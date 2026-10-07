"""
Production-model Brier with the rating features SWAPPED: original Elo vs the ICC-style
ratings, run through the real win-probability pipeline.

This is the metric the repo actually gates on (ball-by-ball second-innings win probability),
unlike scripts/compare_prematch_brier.py which only measures pre-match rating content.

SCOPE -- this is the INTERNATIONAL SUBSET of t20, not the full corpus, so the numbers are NOT
comparable to data/processed/baseline_metrics.json. Reason: the ICC-style engine only rates
internationals, so a full-corpus swap would have to invent league team ratings and would stop
being a like-for-like test. The old system is therefore re-evaluated on the SAME subset as its
own control, which is what makes the comparison valid.

Design
  - Both arms share identical matches, identical folds, identical venue and phase-Elo features.
    ONLY team_*_elo and player effective_elo differ.
  - The control arm uses the PRODUCTION rating parquets (filtered to the subset), not ratings
    recomputed on the subset -- that is what the deployed model actually consumes.
  - New ratings are affine-rescaled onto the old scale's mean/sd before substitution. XGBoost
    is invariant to a monotonic transform of any single feature, but several engineered
    features are DIFFERENCES of ratings (residual_batter_elo = batter_elo - team_elo), so the
    two scales must be commensurate or those features become meaningless.
  - Squad aggregates are recomputed from the new ratings with the same positional/frontline
    weighting production uses, rather than inherited from the old system (which would leak the
    control into the treatment).
  - Models are trained to scratch cache paths; nothing under data/processed is written.

Usage:
    py scripts/measure_rating_swap_brier.py
    py scripts/measure_rating_swap_brier.py --scratch <dir>
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import gc
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.load_data import CricketDataLoader
from src.data_processing.match_metadata import MatchMetadataExtractor
from src.data_processing.build_states import StateBuilder
from src.data_processing.format_spec import spec, INTL_COMPETITIONS, franchise_competitions
from src.features.phase_elo_engine import PhaseEloEngine
from src.features.venue_engine import VenueEngine
from src.features.player_elo_engine import PlayerEloEngine
from src.features.icc_player_rating_engine import AsOfRatingIndex
from src.models.xgb_win_probability import XGBWinProbabilityModel

SQUAD_COLS = ["team_1_batting_squad_elo", "team_1_bowling_squad_elo",
              "team_2_batting_squad_elo", "team_2_bowling_squad_elo"]


def rescale_to(reference: pd.Series, values: pd.Series) -> pd.Series:
    """Affine map putting `values` on `reference`'s mean/sd -- rank-preserving, so it changes
    nothing for a single feature, but makes rating DIFFERENCES commensurate across systems."""
    r_mu, r_sd = float(reference.mean()), float(reference.std())
    v_mu, v_sd = float(values.mean()), float(values.std())
    if v_sd <= 0:
        return pd.Series(np.full(len(values), r_mu), index=values.index)
    return (values - v_mu) / v_sd * r_sd + r_mu


def build_new_elo_df(old_elo_df, icc_teams, fmt):
    """Old elo_df schema, team ratings replaced by rescaled ICC-style ratings."""
    out = old_elo_df.copy()
    icc = icc_teams.set_index("match_id")
    ref = pd.concat([old_elo_df["team_1_elo"], old_elo_df["team_2_elo"]])
    new = pd.concat([icc["team_1_elo"], icc["team_2_elo"]])
    scaled = rescale_to(ref, new)
    n1 = scaled.iloc[:len(icc)]
    n1.index = icc.index
    n2 = scaled.iloc[len(icc):]
    n2.index = icc.index
    out["team_1_elo"] = out["match_id"].map(n1).fillna(out["team_1_elo"])
    out["team_2_elo"] = out["match_id"].map(n2).fillna(out["team_2_elo"])
    return out


def build_new_player_df(old_player_df, icc_players, elo_df, fmt):
    """Old player_elo_df schema, effective_elo replaced by rescaled ICC-style player ratings,
    with squad aggregates recomputed from the NEW ratings using production's own weighting."""
    out = old_player_df.copy()
    key = ["match_id", "player", "role"]
    new = icc_players[key + ["effective_elo"]].rename(columns={"effective_elo": "new_elo"})
    out = out.merge(new, on=key, how="left")

    for role in ("batter", "bowler"):
        m = out["role"] == role
        ref = out.loc[m, "effective_elo"]
        vals = out.loc[m, "new_elo"]
        ok = vals.notna()
        if ok.sum() < 10:
            continue
        scaled = rescale_to(ref, vals[ok])
        out.loc[vals[ok].index, "new_elo"] = scaled
    # any player the new system never rated keeps the neutral default rather than the old value,
    # so the treatment arm contains no old-system information
    neutral = float(pd.concat([out["effective_elo"]]).mean())
    out["effective_elo"] = out["new_elo"].fillna(neutral)
    out = out.drop(columns=["new_elo"])

    # recompute squad aggregates from the new ratings (same weights production uses)
    w_bat = PlayerEloEngine.W_BAT.get(fmt, PlayerEloEngine.W_BAT["t20"])
    w_bowl = PlayerEloEngine.W_BOWL.get(fmt, PlayerEloEngine.W_BOWL["t20"])
    bat_map, bowl_map = {}, {}
    for r in out.itertuples(index=False):
        (bat_map if r.role == "batter" else bowl_map).setdefault(r.match_id, {})[r.player] = r.effective_elo

    squads = elo_df.set_index("match_id")[["team_1_players", "team_2_players"]].to_dict("index")
    rows = {}
    for mid, sq in squads.items():
        bm, wm = bat_map.get(mid, {}), bowl_map.get(mid, {})
        vals = {}
        for i, col in ((1, "team_1_players"), (2, "team_2_players")):
            players = [n.strip() for n in str(sq.get(col, "")).split(",") if n.strip()]
            vals[f"team_{i}_batting_squad_elo"] = PlayerEloEngine._weighted_mean_by_position(
                players, bm, neutral, w_bat)
            vals[f"team_{i}_bowling_squad_elo"] = PlayerEloEngine._weighted_mean_by_bowling_rank(
                players, wm, neutral, w_bowl)
        rows[mid] = vals
    for col in SQUAD_COLS:
        out[col] = out["match_id"].map(lambda m: rows.get(m, {}).get(col, np.nan))
    return out


def build_franchise_elo_df(old_elo_df, bridge_df, meta, fmt):
    """Old elo_df schema, team ratings replaced by squad-composed BRIDGE ratings.

    Franchise sides have no ICC-style team rating of their own -- the bridge rates players, and
    a team is only as good as the XI it fields. So each side's rating is composed per match from
    its own XI's bridge ratings AS OF that match date (production's positional/frontline
    weighting), then rescaled onto the old Elo scale. This is the same composition
    FranchiseBridgeEngine.team_ratings_as_of uses, evaluated per match rather than per window.
    """
    index = AsOfRatingIndex(bridge_df)
    w_bat = PlayerEloEngine.W_BAT.get(fmt, PlayerEloEngine.W_BAT["t20"])
    w_bowl = PlayerEloEngine.W_BOWL.get(fmt, PlayerEloEngine.W_BOWL["t20"])
    dates = meta.set_index("match_id")["date"].to_dict()

    out = old_elo_df.copy()
    vals = {1: {}, 2: {}}
    for r in out.itertuples(index=False):
        as_of = dates.get(r.match_id)
        if as_of is None:
            continue
        for i, col in ((1, "team_1_players"), (2, "team_2_players")):
            players = [n.strip() for n in str(getattr(r, col, "")).split(",") if n.strip()]
            if not players:
                continue
            bm = {p: index.effective_rating(p, "batter", as_of) for p in players}
            wm = {p: index.effective_rating(p, "bowler", as_of) for p in players}
            bat = PlayerEloEngine._weighted_mean_by_position(players, bm, 0.0, w_bat)
            bowl = PlayerEloEngine._weighted_mean_by_bowling_rank(players, wm, 0.0, w_bowl)
            # one team number, balancing both disciplines, mirroring team_ratings_as_of
            vals[i][r.match_id] = (bat + bowl) / 2.0

    ref = pd.concat([old_elo_df["team_1_elo"], old_elo_df["team_2_elo"]])
    raw = pd.Series({**{("1", k): v for k, v in vals[1].items()},
                     **{("2", k): v for k, v in vals[2].items()}})
    scaled = rescale_to(ref, raw)
    for i in (1, 2):
        m = {k[1]: scaled[k] for k in scaled.index if k[0] == str(i)}
        out[f"team_{i}_elo"] = out["match_id"].map(m).fillna(out[f"team_{i}_elo"])
    return out


def build_full_corpus_ratings(old_elo_df, old_player_df, sources, meta, fmt):
    """Combine every rating source into one full-corpus swap.

    Sources are disjoint by match_id (internationals / Tier A / Tier B-C), and each lives on its
    own native scale, so each is standardised SEPARATELY onto a common (1500, 100) target rather
    than pooled first. That is both leak-free (no old-Elo values are consulted) and adequate:
    the model only ever sees the DIFFERENCE between two sides in one match, and matches never
    cross subsets, so cross-subset level alignment does not affect any feature.

    Matches no source covers keep a neutral rating. That handicaps the new arm -- the control
    retains real Elo there -- so the comparison is conservative by construction.
    """
    TARGET_MU, TARGET_SD = 1500.0, 100.0

    def standardise(v):
        v = pd.Series(v, dtype=float)
        mu, sd = float(v.mean()), float(v.std())
        return (v - mu) / sd * TARGET_SD + TARGET_MU if sd > 0 else pd.Series(
            np.full(len(v), TARGET_MU), index=v.index)

    # ---- players: concat the sources, standardising each source x role separately
    frames = []
    for name, (players_df, _) in sources.items():
        f = players_df[["match_id", "player", "role", "effective_elo"]].copy()
        for role in ("batter", "bowler"):
            m = f["role"] == role
            if m.sum() >= 10:
                f.loc[m, "effective_elo"] = standardise(f.loc[m, "effective_elo"]).values
        frames.append(f)
    new_players = pd.concat(frames, ignore_index=True)

    out_p = old_player_df.copy()
    out_p = out_p.merge(new_players.rename(columns={"effective_elo": "new_elo"}),
                        on=["match_id", "player", "role"], how="left")
    out_p["effective_elo"] = out_p["new_elo"].fillna(TARGET_MU)
    out_p = out_p.drop(columns=["new_elo"])

    w_bat = PlayerEloEngine.W_BAT.get(fmt, PlayerEloEngine.W_BAT["t20"])
    w_bowl = PlayerEloEngine.W_BOWL.get(fmt, PlayerEloEngine.W_BOWL["t20"])
    bat_map, bowl_map = {}, {}
    for r in out_p.itertuples(index=False):
        (bat_map if r.role == "batter" else bowl_map).setdefault(r.match_id, {})[r.player] = r.effective_elo

    # ---- teams: each source standardised separately, then merged
    team_vals = {1: {}, 2: {}}
    for name, (_, teams_df) in sources.items():
        if teams_df is None:
            continue
        t = teams_df.set_index("match_id")
        both = pd.concat([t["team_1_elo"], t["team_2_elo"]])
        scaled = standardise(both)
        n = len(t)
        for i, sl in ((1, scaled.iloc[:n]), (2, scaled.iloc[n:])):
            team_vals[i].update(dict(zip(t.index, sl.values)))

    out_e = old_elo_df.copy()
    for i in (1, 2):
        out_e[f"team_{i}_elo"] = out_e["match_id"].map(team_vals[i]).fillna(TARGET_MU)

    # squad aggregates from the new player ratings, production's own weighting
    squads = old_elo_df.set_index("match_id")[["team_1_players", "team_2_players"]].to_dict("index")
    rows = {}
    for mid, sq in squads.items():
        bm, wm = bat_map.get(mid, {}), bowl_map.get(mid, {})
        v = {}
        for i, col in ((1, "team_1_players"), (2, "team_2_players")):
            players = [n.strip() for n in str(sq.get(col, "")).split(",") if n.strip()]
            v[f"team_{i}_batting_squad_elo"] = PlayerEloEngine._weighted_mean_by_position(
                players, bm, TARGET_MU, w_bat)
            v[f"team_{i}_bowling_squad_elo"] = PlayerEloEngine._weighted_mean_by_bowling_rank(
                players, wm, TARGET_MU, w_bowl)
        rows[mid] = v
    for col in SQUAD_COLS:
        out_p[col] = out_p["match_id"].map(lambda m: rows.get(m, {}).get(col, np.nan))
    return out_e, out_p


def brier_for(states, fmt, cache_path, label):
    second = states[states["innings"] == 2].copy()
    model = XGBWinProbabilityModel(format_type=fmt)
    model.fit(second, cache_path=str(cache_path))
    probs = model.predict_batch(second)
    second["pred_prob"] = probs
    second["sq_err"] = (second["pred_prob"] - second["won_match"]) ** 2
    per_match = second.groupby("match_id")["sq_err"].mean()
    brier = float(second["sq_err"].mean())
    ll = float(-(second["won_match"] * np.log(np.clip(second["pred_prob"], 1e-9, 1 - 1e-9))
                 + (1 - second["won_match"]) * np.log(np.clip(1 - second["pred_prob"], 1e-9, 1 - 1e-9))).mean())
    print(f"  {label:<22} Brier={brier:.6f}  logloss={ll:.6f}  n={len(second):,}")
    del model
    gc.collect()
    return brier, ll, len(second), per_match


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--format", default="t20")
    p.add_argument("--scope", default="international",
                   choices=["international", "franchise", "full"])
    p.add_argument("--scratch", default=os.environ.get("TEMP", "/tmp"))
    args = p.parse_args()
    fmt = args.format
    scratch = Path(args.scratch) / "rating_swap"
    scratch.mkdir(parents=True, exist_ok=True)
    fmt_spec = spec(fmt)

    raw = str(PROJECT_ROOT / "data" / "raw")
    deliveries = CricketDataLoader(raw, format_type=fmt).load_all()
    meta = MatchMetadataExtractor(raw, format_type=fmt).extract_all()
    meta = meta[meta["gender"] == fmt_spec["gender"]].copy()

    if args.scope == "international":
        subset = meta[meta["competition"].isin(INTL_COMPETITIONS.get(fmt, ()))
                      & (meta["team_type"] == "international")].copy()
    elif args.scope == "franchise":
        subset = meta[meta["competition"].isin(franchise_competitions(fmt, ("A",)))].copy()
    else:
        subset = meta.copy()
    subset["date"] = pd.to_datetime(subset["date"])
    ids = set(subset["match_id"])
    intl = subset
    deliveries = deliveries[deliveries["match_id"].isin(ids)].copy()
    print(f"{args.scope} subset: {len(subset):,} matches, {len(deliveries):,} deliveries")

    proc = PROJECT_ROOT / "data" / "processed"
    old_elo = pd.read_parquet(proc / f"{fmt}_elo_ratings.parquet")
    old_elo = old_elo[old_elo["match_id"].isin(ids)].copy()
    old_players = pd.read_parquet(proc / f"{fmt}_player_elo_ratings.parquet")
    old_players = old_players[old_players["match_id"].isin(ids)].copy()
    icc_teams = pd.read_parquet(proc / f"{fmt}_icc_team_ratings.parquet")
    icc_players = pd.read_parquet(proc / f"{fmt}_icc_player_ratings.parquet")
    bridge = pd.read_parquet(proc / f"{fmt}_franchise_bridge_tiera.parquet")
    bridge_bc_path = proc / f"{fmt}_franchise_bridge_tierbc.parquet"
    bridge_bc = pd.read_parquet(bridge_bc_path) if bridge_bc_path.exists() else None

    if args.scope == "full":
        intl_ids = set(meta[meta["competition"].isin(INTL_COMPETITIONS.get(fmt, ()))
                            & (meta["team_type"] == "international")]["match_id"])
        a_ids = set(meta[meta["competition"].isin(franchise_competitions(fmt, ("A",)))]["match_id"])
        bc_ids = set(meta[meta["competition"].isin(franchise_competitions(fmt, ("B", "C")))]["match_id"])
        covered = intl_ids | a_ids | bc_ids
        print(f"  coverage: international {len(intl_ids):,} | Tier A {len(a_ids):,} | "
              f"Tier B/C {len(bc_ids):,} | uncovered {len(ids - covered):,} "
              f"({len(ids - covered) / len(ids) * 100:.1f}%, neutral-rated -- handicaps the NEW arm)")
        full_sources = {
            "intl": (icc_players, icc_teams),
            "tierA": (bridge, build_franchise_elo_df(
                old_elo[old_elo["match_id"].isin(a_ids)], bridge, subset, fmt)),
        }
        if bridge_bc is not None and len(bridge_bc):
            full_sources["tierBC"] = (bridge_bc, build_franchise_elo_df(
                old_elo[old_elo["match_id"].isin(bc_ids)], bridge_bc, subset, fmt))

    # shared, identical in both arms
    phase_df = PhaseEloEngine(format_type=fmt).run(deliveries, old_elo)
    venue_df = VenueEngine(format_type=fmt).run(intl, deliveries)

    print("\nBuilding states...")
    arms = {}
    for label, (elo_df, player_df) in {
        "OLD Elo (control)": (old_elo, old_players),
        "NEW ICC-style": (
            (build_new_elo_df(old_elo, icc_teams, fmt) if args.scope == "international"
             else build_franchise_elo_df(old_elo, bridge, subset, fmt))
            if args.scope != "full"
            else build_full_corpus_ratings(old_elo, old_players, full_sources, subset, fmt)[0],
            (build_new_player_df(old_players,
                                 icc_players if args.scope == "international" else bridge,
                                 old_elo, fmt))
            if args.scope != "full"
            else build_full_corpus_ratings(old_elo, old_players, full_sources, subset, fmt)[1]),
    }.items():
        e = elo_df.merge(venue_df, on="match_id", how="left")
        states = StateBuilder(
            deliveries, e, player_elo_df=player_df, phase_elo_df=phase_df,
            total_overs=fmt_spec["total_overs"], format_type=fmt,
            balls_per_over=fmt_spec["balls_per_over"],
        ).build()
        arms[label] = states
        print(f"  {label:<22} states={len(states):,}")

    print("\nTraining + evaluating (second innings win probability):")
    res = {}
    for label, states in arms.items():
        cache = scratch / f"{fmt}_{args.scope}_{'old' if 'OLD' in label else 'new'}_xgb.pkl"
        if cache.exists():
            cache.unlink()
        res[label] = brier_for(states, fmt, cache, label)
        del states
        gc.collect()

    (bo, lo, _, pm_old), (bn, ln, _, pm_new) = res["OLD Elo (control)"], res["NEW ICC-style"]
    print(f"\n  Brier   : old {bo:.6f}  new {bn:.6f}   diff {bn - bo:+.6f} "
          f"({'NEW better' if bn < bo else 'OLD better'})")
    print(f"  Logloss : old {lo:.6f}  new {ln:.6f}   diff {ln - lo:+.6f}")
    # Per-STATE counts massively overstate significance: ball-by-ball states within one match
    # are highly correlated, so the effective sample size is the number of MATCHES, not states.
    # Pair at match level, which is the honest unit.
    from scipy.stats import wilcoxon
    common = pm_old.index.intersection(pm_new.index)
    a, b = pm_old.loc[common], pm_new.loc[common]
    stat, pval = wilcoxon(a, b)
    won = int((b < a).sum())
    print(f"\n  Paired at MATCH level (n={len(common):,} matches -- the honest unit):")
    print(f"    new better in {won:,}/{len(common):,} matches ({won / len(common) * 100:.1f}%)")
    print(f"    Wilcoxon signed-rank p={pval:.3e}   "
          f"median per-match Brier delta={float((b - a).median()):+.6f}")

    if args.scope == "full":
        print("\nScope: FULL corpus -- directly comparable to baseline_metrics.json. Check that")
        print("the control arm reproduces the locked baseline; if it does, the harness is")
        print("faithful and the treatment number can be read against the baseline too.")
    else:
        print(f"\nReminder: {args.scope} subset only -- NOT comparable to baseline_metrics.json.")


if __name__ == "__main__":
    main()
