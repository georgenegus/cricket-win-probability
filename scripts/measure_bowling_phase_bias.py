"""
Diagnostic for franchise_bridge_plan.md Phase 3's open question: WHY does the bridge recover
batting quality well (Spearman 0.758) but bowling only moderately (0.402)?

Leading hypothesis: ICCPlayerRatingEngine._bowling_points scores a bowler's economy against
THE WHOLE INNINGS' run rate. T20 scoring is heavily phase-dependent, so a death-overs
specialist is compared against a baseline that includes cheap powerplay/middle overs and looks
bad by construction, while a powerplay bowler is flattered. Batting has no equivalent problem
(a batter's strike rate is compared against the innings they themselves batted in).

This script does not change any engine. It measures:
  1. BIAS      -- mean economy credit (innings_rate - rpo) by the bowler's death-over share.
                  If the hypothesis holds, credit falls monotonically as death share rises.
  2. CORRECTED -- the same, but comparing each bowler against the run rate in the PHASES THEY
                  ACTUALLY BOWLED (balls-weighted), which is the proposed fix. A flat profile
                  here means the fix removes the bias.
  3. GATE SPLIT -- bridge-vs-international agreement for death-heavy vs powerplay-heavy
                  bowlers. If the hypothesis explains the weak gate, death-heavy bowlers should
                  be where the two disagree most.

Usage:
    py scripts/measure_bowling_phase_bias.py
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
from src.data_processing.format_spec import spec, franchise_competitions
from src.features.icc_player_rating_engine import ICCPlayerRatingEngine
from src.features.franchise_bridge_engine import FranchiseBridgeEngine


def label_phases(df: pd.DataFrame, format_type: str) -> pd.DataFrame:
    """Same phase edges phase_elo_engine._aggregate_match_phase_stats uses."""
    fmt_spec = spec(format_type)
    bpo = float(fmt_spec["balls_per_over"])
    out = df.copy()
    out["over_float"] = out["over"] + (out["ball"] - 1) / bpo
    pp_end = fmt_spec["pp_end_ball"] / fmt_spec["balls_per_over"]
    death_start = fmt_spec["death_start_ball"] / fmt_spec["balls_per_over"]
    out["phase"] = np.where(
        out["over_float"] < pp_end, "powerplay",
        np.where(out["over_float"] < death_start, "middle", "death"),
    )
    if "legal_delivery" not in out.columns:
        out["legal_delivery"] = ((out["is_wide"] == 0) & (out["is_noball"] == 0)).astype(int)
    return out


def build_bowler_innings(deliveries: pd.DataFrame, format_type: str) -> pd.DataFrame:
    df = label_phases(deliveries, format_type)

    # per bowler-innings totals
    per_bowler = (
        df.groupby(["match_id", "batting_team", "bowler"])
        .agg(runs_conceded=("total_runs", "sum"), balls=("legal_delivery", "sum"))
        .reset_index()
    )
    # per bowler-innings, balls in each phase
    per_bowler_phase = (
        df.groupby(["match_id", "batting_team", "bowler", "phase"])["legal_delivery"]
        .sum().reset_index().rename(columns={"legal_delivery": "balls_in_phase"})
    )
    # innings totals, overall and by phase
    innings = (
        df.groupby(["match_id", "batting_team"])
        .agg(innings_runs=("total_runs", "sum"), innings_balls=("legal_delivery", "sum"))
        .reset_index()
    )
    innings_phase = (
        df.groupby(["match_id", "batting_team", "phase"])
        .agg(phase_runs=("total_runs", "sum"), phase_balls=("legal_delivery", "sum"))
        .reset_index()
    )
    innings_phase["phase_rate"] = np.where(
        innings_phase["phase_balls"] > 0,
        innings_phase["phase_runs"] / innings_phase["phase_balls"] * 6.0, np.nan,
    )

    # balls-weighted expected rate across the phases this bowler actually bowled
    w = per_bowler_phase.merge(
        innings_phase[["match_id", "batting_team", "phase", "phase_rate"]],
        on=["match_id", "batting_team", "phase"], how="left",
    )
    w["weighted"] = w["balls_in_phase"] * w["phase_rate"]
    expected = (
        w.groupby(["match_id", "batting_team", "bowler"])
        .agg(weighted_sum=("weighted", "sum"), balls_sum=("balls_in_phase", "sum"))
        .reset_index()
    )
    expected["phase_expected_rate"] = expected["weighted_sum"] / expected["balls_sum"]

    death = (
        per_bowler_phase[per_bowler_phase["phase"] == "death"]
        .rename(columns={"balls_in_phase": "death_balls"})
        .drop(columns=["phase"])
    )

    out = (
        per_bowler
        .merge(innings, on=["match_id", "batting_team"], how="left")
        .merge(expected[["match_id", "batting_team", "bowler", "phase_expected_rate"]],
               on=["match_id", "batting_team", "bowler"], how="left")
        .merge(death, on=["match_id", "batting_team", "bowler"], how="left")
    )
    out["death_balls"] = out["death_balls"].fillna(0)
    out = out[out["balls"] > 0].copy()
    out["death_share"] = out["death_balls"] / out["balls"]
    out["rpo"] = out["runs_conceded"] / out["balls"] * 6.0
    out["innings_rate"] = out["innings_runs"] / out["innings_balls"] * 6.0
    # the credit term as _bowling_points computes it today, and as it would with the fix
    out["credit_current"] = out["innings_rate"] - out["rpo"]
    out["credit_phase_adjusted"] = out["phase_expected_rate"] - out["rpo"]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", default="t20")
    args = parser.parse_args()
    fmt = args.format

    fmt_spec = spec(fmt)
    raw = str(PROJECT_ROOT / "data" / "raw")
    deliveries = CricketDataLoader(raw, format_type=fmt).load_all()
    meta = MatchMetadataExtractor(raw, format_type=fmt).extract_all()
    meta = meta[meta["gender"] == fmt_spec["gender"]].copy()

    tier_a = set(franchise_competitions(fmt, ("A",)))
    ids = set(meta[meta["competition"].isin(tier_a)]["match_id"])
    deliveries = deliveries[deliveries["match_id"].isin(ids)].copy()
    print(f"Tier A franchise deliveries: {len(deliveries):,}")

    bi = build_bowler_innings(deliveries, fmt)
    print(f"bowler-innings: {len(bi):,}\n")

    print("=" * 72)
    print("1. BIAS -- economy credit by death-over share (current whole-innings baseline)")
    print("=" * 72)
    buckets = [(0.0, 0.01, "no death overs"), (0.01, 0.25, "1-25% death"),
               (0.25, 0.5, "25-50%"), (0.5, 0.75, "50-75%"), (0.75, 1.01, "75-100% death")]
    print(f"{'bucket':<16}{'n':>8}{'mean credit':>14}{'phase-adjusted':>16}")
    for lo, hi, label in buckets:
        sel = bi[(bi["death_share"] >= lo) & (bi["death_share"] < hi)]
        if len(sel) == 0:
            continue
        print(f"{label:<16}{len(sel):>8,}{sel['credit_current'].mean():>14.3f}"
              f"{sel['credit_phase_adjusted'].mean():>16.3f}")

    corr_cur, _ = spearmanr(bi["death_share"], bi["credit_current"])
    corr_adj, _ = spearmanr(bi["death_share"], bi["credit_phase_adjusted"])
    print(f"\nSpearman(death_share, credit):  current={corr_cur:.3f}   "
          f"phase-adjusted={corr_adj:.3f}")
    print("A strongly negative 'current' value is the bias: more death overs -> less credit,")
    print("regardless of how well the bowler actually bowled. Nearer zero after adjustment is")
    print("the fix working.")

    print("\n" + "=" * 72)
    print("2. GATE SPLIT -- does the bias explain the weak bowling gate?")
    print("=" * 72)
    proc = PROJECT_ROOT / "data" / "processed"
    bridge_df = pd.read_parquet(proc / f"{fmt}_franchise_bridge_tiera.parquet")
    icc_player_df = pd.read_parquet(proc / f"{fmt}_icc_player_ratings.parquet")
    as_of = pd.to_datetime(bridge_df["match_date"]).max() + pd.Timedelta(days=1)

    career_death = bi.groupby("bowler").apply(
        lambda g: (g["death_balls"].sum() / g["balls"].sum()) if g["balls"].sum() > 0 else np.nan,
        include_groups=False,
    )

    bridge = FranchiseBridgeEngine.get_effective_ratings_from_df(bridge_df, "bowler", as_of_date=as_of)
    intl = ICCPlayerRatingEngine.get_effective_ratings_from_df(icc_player_df, "bowler", as_of_date=as_of)
    common = [p for p in set(bridge) & set(intl) if p in career_death.index and not np.isnan(career_death[p])]
    print(f"bowlers rated in both systems, with phase data: {len(common)}")

    med = float(np.median([career_death[p] for p in common]))
    groups = {
        f"death-light (<{med:.2f} death share)": [p for p in common if career_death[p] < med],
        f"death-heavy (>={med:.2f})": [p for p in common if career_death[p] >= med],
    }
    for label, sel in groups.items():
        if len(sel) < 10:
            print(f"  {label}: n={len(sel)} -- too few")
            continue
        b = np.array([bridge[p] for p in sel])
        i = np.array([intl[p] for p in sel])
        rho, p = spearmanr(b, i)
        print(f"  {label:<34} n={len(sel):<4} spearman={rho:.3f} (p={p:.1e})  "
              f"mean gap={np.mean(b - i):+.2f}")


if __name__ == "__main__":
    main()
