"""
Fits ICCPlayerRatingEngine.WICKET_VALUE -- the wickets-vs-economy trade-off in
_bowling_points, which was never fitted to anything, just picked at 20.0.

This is the last untested lead on franchise_bridge_plan.md Phase 3's bowling weakness (phase
bias was refuted; measurement noise explains part but not all -- disattenuated agreement is
~0.55 for bowling vs ~0.85 for batting).

Method. _bowling_points is LINEAR in its weights:

    points = WICKET_VALUE * (wickets * oppf)
           + W_ECONOMY_BONUS * (balls * (innings_rate - rpo) / 6 * oppf)
           + W_RESULT * (result_indicator)

so the three bracketed components can be computed ONCE per innings, and any parameterisation
evaluated instantly. The time-weighting is parameter-independent too, so each player's
weighted component sums are also precomputed -- making the whole sweep essentially free.

Two independent objectives are reported per candidate, and a value is only worth trusting if
they agree:
  RELIABILITY -- split-half self-consistency of the resulting bowling rating (does this
                 weighting produce a measure that agrees with itself?). Cleanest criterion,
                 no circularity.
  GATE        -- agreement between the international rating and the franchise-bridge rating
                 (does it make the measure consistent across two independent data sources?).

Usage:
    py scripts/fit_bowling_wicket_value.py
    py scripts/fit_bowling_wicket_value.py --sweep 0 5 10 20 40 80
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.load_data import CricketDataLoader
from src.data_processing.match_metadata import MatchMetadataExtractor
from src.data_processing.format_spec import spec, INTL_COMPETITIONS, franchise_competitions
from src.features.icc_team_rating_engine import ICCTeamRatingEngine
from src.features.icc_player_rating_engine import ICCPlayerRatingEngine


def bowler_components(deliveries, meta, opponent_strength_lookup):
    """One row per bowler-innings with the three parameter-free components of _bowling_points.

    opponent_strength_lookup: (match_id, batting_team) -> opponent strength on the team scale
    (the batting side is the opposition, from the bowler's point of view).
    """
    scorer = ICCPlayerRatingEngine()
    _, bowl_stats, innings_totals = scorer._aggregate_match_stats(deliveries)

    innings_totals = innings_totals.copy()
    innings_totals["innings_rate"] = np.where(
        innings_totals["innings_balls"] > 0,
        innings_totals["innings_runs"] / innings_totals["innings_balls"] * 6.0, 0.0,
    )
    df = bowl_stats.merge(
        innings_totals[["match_id", "batting_team", "innings_rate"]],
        on=["match_id", "batting_team"], how="left",
    )
    df = df[df["balls_bowled"] > 0].copy()

    m = meta[["match_id", "team_1", "team_2", "winner", "date"]].copy()
    m["date"] = pd.to_datetime(m["date"])
    df = df.merge(m, on="match_id", how="inner")

    # the bowler's own team is whichever side is NOT batting
    df["bowling_team"] = np.where(df["batting_team"] == df["team_1"], df["team_2"], df["team_1"])
    has_winner = (
        df["winner"].notna()
        & (df["winner"] != "")
        & ((df["winner"] == df["team_1"]) | (df["winner"] == df["team_2"]))
    )
    df["result"] = np.where(
        ~has_winner, 0.0,
        np.where(df["winner"] == df["bowling_team"], 1.0, -1.0),
    )

    df["opp_strength"] = [
        opponent_strength_lookup.get((mid, bt), ICCTeamRatingEngine.SEED_RATING)
        for mid, bt in zip(df["match_id"], df["batting_team"])
    ]
    oppf = df["opp_strength"] / ICCTeamRatingEngine.SEED_RATING

    df["rpo"] = df["runs_conceded"] / df["balls_bowled"] * 6.0
    df["c_wickets"] = df["wickets"] * oppf
    df["c_econ"] = df["balls_bowled"] * (df["innings_rate"] - df["rpo"]) / 6.0 * oppf
    df["c_result"] = df["result"]
    return df[["match_id", "bowler", "date", "c_wickets", "c_econ", "c_result"]].rename(
        columns={"bowler": "player", "date": "match_date"}
    )


def weighted_component_sums(components: pd.DataFrame, as_of, min_innings, max_inactivity):
    """Per player: time-weighted sums of each component plus the weight total, so a rating for
    ANY parameterisation is a ratio of linear combinations. Applies the same qualification and
    recency floors the leaderboards use."""
    df = components.copy()
    df["age_days"] = (pd.Timestamp(as_of) - pd.to_datetime(df["match_date"])).dt.days
    df = df[df["age_days"] >= 0]
    df["w"] = np.where(
        df["age_days"] < ICCPlayerRatingEngine.FULL_WEIGHT_DAYS, 1.0,
        np.where(df["age_days"] < ICCPlayerRatingEngine.DROP_WEIGHT_DAYS,
                 ICCPlayerRatingEngine.HALF_WEIGHT, 0.0),
    )
    counts = components.groupby("player").size()
    last = pd.to_datetime(components.groupby("player")["match_date"].max())
    eligible = {
        p for p in counts.index
        if counts[p] >= min_innings
        and (pd.Timestamp(as_of) - last[p]).days <= max_inactivity
    }
    df = df[df["player"].isin(eligible) & (df["w"] > 0)]

    g = df.groupby("player")
    out = pd.DataFrame({
        "sw": g["w"].sum(),
        "swk": g.apply(lambda x: float((x["w"] * x["c_wickets"]).sum()), include_groups=False),
        "swe": g.apply(lambda x: float((x["w"] * x["c_econ"]).sum()), include_groups=False),
        "swr": g.apply(lambda x: float((x["w"] * x["c_result"]).sum()), include_groups=False),
    })
    # n for the reliability blend is LIFETIME innings, matching
    # ICCPlayerRatingEngine.get_effective_ratings_from_df -- not just innings inside the decay
    # window. Using in-window counts here silently under-weights long-career players and made
    # this script's reproduction of the gate disagree with the engine's own number.
    out["n"] = counts.reindex(out.index).fillna(0).astype(int)
    return out[out["sw"] > 0]


def rating(sums: pd.DataFrame, wicket_value, w_econ, w_result) -> pd.Series:
    raw = (wicket_value * sums["swk"] + w_econ * sums["swe"] + w_result * sums["swr"]) / sums["sw"]
    reliability = sums["n"] / (sums["n"] + ICCPlayerRatingEngine.MIN_MATCHES_RELIABLE)
    return reliability * raw + (1.0 - reliability) * ICCPlayerRatingEngine.NEUTRAL_BASELINE


def split_half(components, wicket_value, w_econ, w_result, min_innings, trials, seed):
    """Split-half reliability of the resulting rating, Spearman-Brown corrected."""
    counts = components.groupby("player").size()
    eligible = counts[counts >= min_innings].index
    df = components[components["player"].isin(eligible)]
    pts = (wicket_value * df["c_wickets"] + w_econ * df["c_econ"] + w_result * df["c_result"]).to_numpy()
    players = df["player"].to_numpy()

    order = np.argsort(players, kind="stable")
    players_s, pts_s = players[order], pts[order]
    boundaries = np.flatnonzero(np.r_[True, players_s[1:] != players_s[:-1]])
    groups = np.split(pts_s, boundaries[1:])

    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(trials):
        a, b = [], []
        for arr in groups:
            idx = rng.permutation(len(arr))
            h = len(arr) // 2
            if h == 0:
                continue
            a.append(arr[idx[:h]].mean())
            b.append(arr[idx[h:2 * h]].mean())
        rho, _ = spearmanr(a, b)
        rhos.append(rho)
    r_half = float(np.mean(rhos))
    return (2 * r_half) / (1 + r_half) if r_half > -1 else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", default="t20")
    parser.add_argument("--sweep", nargs="+", type=float,
                        default=[0, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 60, 80])
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    fmt = args.format

    fmt_spec = spec(fmt)
    raw = str(PROJECT_ROOT / "data" / "raw")
    deliveries_all = CricketDataLoader(raw, format_type=fmt).load_all()
    meta_all = MatchMetadataExtractor(raw, format_type=fmt).extract_all()
    meta_all = meta_all[meta_all["gender"] == fmt_spec["gender"]].copy()

    proc = PROJECT_ROOT / "data" / "processed"
    icc_elo_df = pd.read_parquet(proc / f"{fmt}_icc_team_ratings.parquet")
    bridge_df = pd.read_parquet(proc / f"{fmt}_franchise_bridge_tiera.parquet")

    # --- international components: opponent strength is the real team rating
    intl_lookup = {}
    for r in icc_elo_df.itertuples(index=False):
        intl_lookup[(r.match_id, r.team_1)] = r.team_1_elo
        intl_lookup[(r.match_id, r.team_2)] = r.team_2_elo
    intl_meta = meta_all[
        meta_all["competition"].isin(INTL_COMPETITIONS.get(fmt, ()))
        & (meta_all["team_type"] == "international")
    ]
    intl_deliveries = deliveries_all[deliveries_all["match_id"].isin(set(intl_meta["match_id"]))]
    intl_comp = bowler_components(intl_deliveries, intl_meta, intl_lookup)
    print(f"international bowler-innings: {len(intl_comp):,}")

    # --- franchise components: opponent strength is the anchor-derived value the bridge saved
    bridge_bowl = bridge_df[bridge_df["role"] == "bowler"]
    fr_lookup = {}
    for r in bridge_bowl.itertuples(index=False):
        # bridge rows store the bowler's team; the opposition is whoever batted
        fr_lookup[(r.match_id, r.player)] = r.opponent_strength
    fr_meta = meta_all[meta_all["competition"].isin(franchise_competitions(fmt, ("A",)))]
    fr_deliveries = deliveries_all[deliveries_all["match_id"].isin(set(fr_meta["match_id"]))]
    # rebuild with a (match_id, batting_team) lookup derived from the bridge's per-innings value
    tmp = bowler_components(fr_deliveries, fr_meta, {})
    tmp = tmp.merge(
        bridge_bowl[["match_id", "player", "opponent_strength"]].drop_duplicates(),
        on=["match_id", "player"], how="left",
    )
    tmp["opponent_strength"] = tmp["opponent_strength"].fillna(ICCTeamRatingEngine.SEED_RATING)
    # components were built with a neutral factor, so rescale to the real one
    scale = tmp["opponent_strength"] / ICCTeamRatingEngine.SEED_RATING
    tmp["c_wickets"] *= scale
    tmp["c_econ"] *= scale
    fr_comp = tmp[["match_id", "player", "match_date", "c_wickets", "c_econ", "c_result"]]
    print(f"franchise bowler-innings:     {len(fr_comp):,}")

    as_of = pd.to_datetime(bridge_df["match_date"]).max() + pd.Timedelta(days=1)
    mi = ICCPlayerRatingEngine.MIN_MATCHES_TO_RANK
    mx = ICCPlayerRatingEngine.MAX_INACTIVITY_DAYS
    intl_sums = weighted_component_sums(intl_comp, as_of, mi, mx)
    fr_sums = weighted_component_sums(fr_comp, as_of, mi, mx)
    print(f"qualified: international={len(intl_sums)}  franchise={len(fr_sums)}  "
          f"common={len(set(intl_sums.index) & set(fr_sums.index))}\n")

    w_econ = ICCPlayerRatingEngine.W_ECONOMY_BONUS
    w_result = ICCPlayerRatingEngine.W_RESULT_BOWL   # role-split 2026-09-02
    print(f"(holding W_ECONOMY_BONUS={w_econ}, W_RESULT={w_result}; only the ratio matters)\n")
    print(f"{'WICKET_VALUE':>13}{'intl reliab':>13}{'bridge reliab':>15}{'GATE spearman':>15}{'disattenuated':>15}")

    best = None
    for wv in args.sweep:
        rel_i = split_half(intl_comp, wv, w_econ, w_result, mi, args.trials, args.seed)
        rel_f = split_half(fr_comp, wv, w_econ, w_result, mi, args.trials, args.seed)
        ri = rating(intl_sums, wv, w_econ, w_result)
        rf = rating(fr_sums, wv, w_econ, w_result)
        common = sorted(set(ri.index) & set(rf.index))
        rho, _ = spearmanr(ri[common].to_numpy(), rf[common].to_numpy())
        ceiling = np.sqrt(rel_i * rel_f) if rel_i > 0 and rel_f > 0 else np.nan
        disatt = rho / ceiling if ceiling and not np.isnan(ceiling) and ceiling > 0 else np.nan
        marker = ""
        if best is None or rho > best[1]:
            best = (wv, rho)
        print(f"{wv:>13.1f}{rel_i:>13.3f}{rel_f:>15.3f}{rho:>15.3f}{disatt:>15.3f}{marker}")

    print(f"\ncurrent WICKET_VALUE = {ICCPlayerRatingEngine.WICKET_VALUE}")
    print(f"best gate at WICKET_VALUE = {best[0]} (spearman {best[1]:.3f})")


if __name__ == "__main__":
    main()
