"""
Fits ICCPlayerRatingEngine.W_RESULT -- the flat match-result bonus in _batting_points and
_bowling_points (+W_RESULT if the player's side won, -W_RESULT if it lost).

Why look: like WICKET_VALUE before it, this was never fitted, just set at 10.0. It is doing a
lot of work for an unfitted constant -- ~40% of a typical batting innings score (mean |points|
24.9) -- and it is conceptually blunt: it credits a batter identically whether they made 5 or
95 in the win. The W_RESULT=0 ablation in franchise_bridge_plan.md barely moved the historical
team rankings, which already hints the term carries little signal.

Both engines' points are LINEAR in W_RESULT:

    points = c_base + W_RESULT * c_result

so c_base and c_result are computed once per innings and every candidate evaluated exactly.
Two independent objectives, same as the WICKET_VALUE fit:
  RELIABILITY -- split-half self-consistency of the rating (no circularity)
  GATE        -- international vs franchise-bridge agreement (cross-source consistency)

Usage:
    py scripts/fit_result_bonus.py
    py scripts/fit_result_bonus.py --sweep 0 5 10 20 40
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

E = ICCPlayerRatingEngine
SEED = ICCTeamRatingEngine.SEED_RATING


def components(deliveries, meta, opp_lookup, role):
    """One row per player-innings: c_base (everything except the result term) and c_result.

    opp_lookup: (match_id, opponent_team) -> opponent strength on the team scale.
    """
    scorer = E()
    bat_stats, bowl_stats, innings = scorer._aggregate_match_stats(deliveries)
    innings = innings.copy()
    innings["innings_rate"] = np.where(
        innings["innings_balls"] > 0,
        innings["innings_runs"] / innings["innings_balls"] * 6.0, 0.0)

    src = bat_stats if role == "batter" else bowl_stats
    df = src.merge(innings[["match_id", "batting_team", "innings_rate"]],
                   on=["match_id", "batting_team"], how="left")

    m = meta[["match_id", "team_1", "team_2", "winner", "date"]].copy()
    m["date"] = pd.to_datetime(m["date"])
    df = df.merge(m, on="match_id", how="inner")

    if role == "batter":
        df = df[df["balls_faced"] > 0].copy()
        df["own_team"] = df["batting_team"]
        # a batter faces the OPPOSING side
        df["opp_team"] = np.where(df["batting_team"] == df["team_1"], df["team_2"], df["team_1"])
        player_col = "batter"
    else:
        df = df[df["balls_bowled"] > 0].copy()
        # for bowling rows, batting_team IS the opposition
        df["opp_team"] = df["batting_team"]
        df["own_team"] = np.where(df["batting_team"] == df["team_1"], df["team_2"], df["team_1"])
        player_col = "bowler"

    df["opp_strength"] = [opp_lookup.get((mid, t), SEED) for mid, t in zip(df["match_id"], df["opp_team"])]
    oppf = df["opp_strength"] / SEED

    has_winner = (df["winner"].notna() & (df["winner"] != "")
                  & ((df["winner"] == df["team_1"]) | (df["winner"] == df["team_2"])))
    df["c_result"] = np.where(~has_winner, 0.0, np.where(df["winner"] == df["own_team"], 1.0, -1.0))

    if role == "batter":
        sr = df["runs"] / df["balls_faced"] * 100.0
        innings_sr = df["innings_rate"] * 100.0 / 6.0
        df["c_base"] = (df["runs"] * oppf
                        + E.W_SR_BONUS * df["balls_faced"] * (sr - innings_sr) / 100.0 * oppf)
    else:
        rpo = df["runs_conceded"] / df["balls_bowled"] * 6.0
        df["c_base"] = (E.WICKET_VALUE * df["wickets"] * oppf
                        + E.W_ECONOMY_BONUS * df["balls_bowled"] * (df["innings_rate"] - rpo) / 6.0 * oppf)

    return df[["match_id", player_col, "date", "c_base", "c_result"]].rename(
        columns={player_col: "player", "date": "match_date"})


def weighted_sums(comp, as_of):
    df = comp.copy()
    df["age_days"] = (pd.Timestamp(as_of) - pd.to_datetime(df["match_date"])).dt.days
    df = df[df["age_days"] >= 0]
    df["w"] = np.where(df["age_days"] < E.FULL_WEIGHT_DAYS, 1.0,
                       np.where(df["age_days"] < E.DROP_WEIGHT_DAYS, E.HALF_WEIGHT, 0.0))
    counts = comp.groupby("player").size()
    last = pd.to_datetime(comp.groupby("player")["match_date"].max())
    eligible = {p for p in counts.index
                if counts[p] >= E.MIN_MATCHES_TO_RANK
                and (pd.Timestamp(as_of) - last[p]).days <= E.MAX_INACTIVITY_DAYS}
    df = df[df["player"].isin(eligible) & (df["w"] > 0)]
    g = df.groupby("player")
    out = pd.DataFrame({
        "sw": g["w"].sum(),
        "sb": g.apply(lambda x: float((x["w"] * x["c_base"]).sum()), include_groups=False),
        "sr": g.apply(lambda x: float((x["w"] * x["c_result"]).sum()), include_groups=False),
    })
    out["n"] = counts.reindex(out.index).fillna(0).astype(int)
    return out[out["sw"] > 0]


def rating(sums, w_result):
    raw = (sums["sb"] + w_result * sums["sr"]) / sums["sw"]
    rel = sums["n"] / (sums["n"] + E.MIN_MATCHES_RELIABLE)
    return rel * raw + (1.0 - rel) * E.NEUTRAL_BASELINE


def split_half(comp, w_result, trials, seed):
    counts = comp.groupby("player").size()
    df = comp[comp["player"].isin(counts[counts >= E.MIN_MATCHES_TO_RANK].index)]
    pts = (df["c_base"] + w_result * df["c_result"]).to_numpy()
    players = df["player"].to_numpy()
    order = np.argsort(players, kind="stable")
    players_s, pts_s = players[order], pts[order]
    bounds = np.flatnonzero(np.r_[True, players_s[1:] != players_s[:-1]])
    groups = np.split(pts_s, bounds[1:])
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(trials):
        a, b = [], []
        for arr in groups:
            idx = rng.permutation(len(arr)); h = len(arr) // 2
            if h == 0: continue
            a.append(arr[idx[:h]].mean()); b.append(arr[idx[h:2 * h]].mean())
        rho, _ = spearmanr(a, b); rhos.append(rho)
    r = float(np.mean(rhos))
    return (2 * r) / (1 + r) if r > -1 else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--format", default="t20")
    p.add_argument("--sweep", nargs="+", type=float, default=[0, 2.5, 5, 10, 20, 40])
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    fmt = args.format

    raw = str(PROJECT_ROOT / "data" / "raw")
    deliveries = CricketDataLoader(raw, format_type=fmt).load_all()
    meta = MatchMetadataExtractor(raw, format_type=fmt).extract_all()
    meta = meta[meta["gender"] == spec(fmt)["gender"]].copy()

    proc = PROJECT_ROOT / "data" / "processed"
    icc_elo = pd.read_parquet(proc / f"{fmt}_icc_team_ratings.parquet")
    bridge = pd.read_parquet(proc / f"{fmt}_franchise_bridge_tiera.parquet")
    as_of = pd.to_datetime(bridge["match_date"]).max() + pd.Timedelta(days=1)

    intl_lookup = {}
    for r in icc_elo.itertuples(index=False):
        intl_lookup[(r.match_id, r.team_1)] = r.team_1_elo
        intl_lookup[(r.match_id, r.team_2)] = r.team_2_elo
    intl_meta = meta[meta["competition"].isin(INTL_COMPETITIONS.get(fmt, ()))
                     & (meta["team_type"] == "international")]
    intl_del = deliveries[deliveries["match_id"].isin(set(intl_meta["match_id"]))]

    fr_meta = meta[meta["competition"].isin(franchise_competitions(fmt, ("A",)))]
    fr_del = deliveries[deliveries["match_id"].isin(set(fr_meta["match_id"]))]

    print(f"(holding WICKET_VALUE={E.WICKET_VALUE}, W_SR_BONUS={E.W_SR_BONUS}, "
          f"W_ECONOMY_BONUS={E.W_ECONOMY_BONUS}; current W_RESULT={E.W_RESULT})\n")

    for role in ("batter", "bowler"):
        ic = components(intl_del, intl_meta, intl_lookup, role)
        # franchise opponent strength comes from what the bridge already saved per innings
        brole = bridge[bridge["role"] == role][["match_id", "player", "opponent_strength"]].drop_duplicates()
        fc = components(fr_del, fr_meta, {}, role).merge(brole, on=["match_id", "player"], how="left")
        fc["opponent_strength"] = fc["opponent_strength"].fillna(SEED)
        fc["c_base"] *= fc["opponent_strength"] / SEED   # components were built neutral; rescale
        fc = fc[["match_id", "player", "match_date", "c_base", "c_result"]]

        isum, fsum = weighted_sums(ic, as_of), weighted_sums(fc, as_of)
        print(f"--- {role} --- (intl innings {len(ic):,}, franchise {len(fc):,}; "
              f"qualified {len(isum)}/{len(fsum)})")
        print(f"{'W_RESULT':>10}{'intl reliab':>13}{'bridge reliab':>15}{'GATE':>10}{'disatt':>9}")
        for wr in args.sweep:
            ri, rf = split_half(ic, wr, args.trials, args.seed), split_half(fc, wr, args.trials, args.seed)
            a, b = rating(isum, wr), rating(fsum, wr)
            common = sorted(set(a.index) & set(b.index))
            rho, _ = spearmanr(a[common].to_numpy(), b[common].to_numpy())
            ceil = np.sqrt(ri * rf) if ri > 0 and rf > 0 else np.nan
            print(f"{wr:>10.1f}{ri:>13.3f}{rf:>15.3f}{rho:>10.3f}{rho / ceil if ceil > 0 else np.nan:>9.3f}")
        print()


if __name__ == "__main__":
    main()
