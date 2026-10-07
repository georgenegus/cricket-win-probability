"""
Phase 0 measurement for markdowns/win_predictor/t20_player_elo_revamp_plan.md.

Reads a format's *existing* player-Elo ratings parquet (no model fitting, no pipeline)
and reports the empirical distribution of per-player CAREER PEAK raw Elo, split by role,
for players with >= MIN_MATCHES appearances. This is the ground truth the hold-center
rescale factor `a` is fitted from -- NOT the ~1650 anecdote.

Hold-center calibration: keep raw 1500 -> 1500 and stretch outward so an "elite" anchor
(default: the p99 career peak) maps to TARGET_ELITE (default 2000). The implied engine
knobs are: elo_divisor' = a * 400, and k_runs'/k_wickets' = a * current_k.

Usage:
    py scripts/measure_player_elo_distribution.py                 # t20, min 20 matches, elite=2000
    py scripts/measure_player_elo_distribution.py --format od
    py scripts/measure_player_elo_distribution.py --min-matches 50 --target-elite 1950
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CENTER = 1500.0
PERCENTILES = [1, 10, 25, 50, 75, 90, 99]

# Well-known stars to surface as sanity anchors (cricsheet initials-surname style).
# "average county player" is not a name we can reliably match -- the p25/p50 of the
# peak distribution IS that anchor, so we lean on percentiles for the average case.
STAR_SUBSTRINGS = [
    "Buttler", "Kohli", "Russell", "Maxwell", "Rashid Khan", "Bumrah",
    "de Kock", "Warner", "Narine", "AB de Villiers", "Pollard", "Malinga",
]


def per_player_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the (one-row-per-appearance) ratings frame to one row per (player, role)."""
    # Rows are appended in chronological match order by the engine, and groupby is
    # order-stable within a group, so `last` == most recent rating.
    g = (
        df.groupby(["player", "role"])
        .agg(peak=("raw_elo", "max"),
             current=("raw_elo", "last"),
             low=("raw_elo", "min"),
             matches=("raw_elo", "size"))
        .reset_index()
    )
    return g


def report_role(summary: pd.DataFrame, role: str, min_matches: int) -> dict:
    sub = summary[(summary["role"] == role) & (summary["matches"] >= min_matches)]
    n = len(sub)
    print(f"\n--- {role.upper()}S (>= {min_matches} matches, n={n}) ---")
    if n == 0:
        print("  (none)")
        return {}
    pct = np.percentile(sub["peak"].values, PERCENTILES)
    stats = {p: float(v) for p, v in zip(PERCENTILES, pct)}
    print("  Career PEAK raw_elo percentiles:")
    for p in PERCENTILES:
        print(f"    p{p:<2d} = {stats[p]:7.1f}")
    print(f"    max = {sub['peak'].max():7.1f}   mean = {sub['peak'].mean():7.1f}")

    top = sub.sort_values("peak", ascending=False).head(12)
    print(f"  Top 12 by peak:")
    for _, r in top.iterrows():
        print(f"    {r['player']:<22s} peak={r['peak']:7.1f}  current={r['current']:7.1f}  matches={int(r['matches'])}")
    return stats


def report_anchors(summary: pd.DataFrame):
    print("\n--- NAMED STAR ANCHORS (batter role) ---")
    bats = summary[summary["role"] == "batter"]
    for needle in STAR_SUBSTRINGS:
        hit = bats[bats["player"].str.contains(needle, case=False, na=False)]
        for _, r in hit.iterrows():
            print(f"    {r['player']:<22s} peak={r['peak']:7.1f}  current={r['current']:7.1f}  matches={int(r['matches'])}")


def fit_hold_center(elite_peak: float, target_elite: float, stats_bat: dict, stats_bowl: dict):
    """a = (target_elite - 1500) / (elite_peak - 1500); report implied knobs + mapped anchors."""
    a = (target_elite - CENTER) / (elite_peak - CENTER)

    def mapped(x):
        return CENTER + a * (x - CENTER)

    print("\n" + "=" * 64)
    print("HOLD-CENTER CALIBRATION (raw 1500 -> 1500, stretch by a)")
    print("=" * 64)
    print(f"  Elite anchor (p99 batter peak)  : {elite_peak:.1f}")
    print(f"  Target elite                     : {target_elite:.1f}")
    print(f"  => stretch factor  a             = {a:.3f}")
    print(f"  => elo_divisor'   = 400 * a      = {400.0 * a:.1f}   (from 400)")
    print(f"  => k_runs'  (t20) = 0.3 * a       = {0.3 * a:.3f}   (from 0.3)")
    print(f"  => k_wickets'(t20)= 3.0 * a       = {3.0 * a:.2f}   (from 3.0)")

    print("\n  Where the batter peak percentiles LAND under this map:")
    for p in PERCENTILES:
        if p in stats_bat:
            print(f"    p{p:<2d}: {stats_bat[p]:7.1f} -> {mapped(stats_bat[p]):7.1f}")
    if stats_bowl:
        print("\n  Where the bowler peak percentiles LAND under the same map:")
        for p in PERCENTILES:
            if p in stats_bowl:
                print(f"    p{p:<2d}: {stats_bowl[p]:7.1f} -> {mapped(stats_bowl[p]):7.1f}")
    print("\n  (p50 batter is the 'average established player' anchor; hold-center puts")
    print("   it a touch above 1600 by design -- center + elite p99 are the pinned points.)")
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="t20", choices=["t20", "od", "hnd"])
    ap.add_argument("--min-matches", type=int, default=20)
    ap.add_argument("--target-elite", type=float, default=2000.0)
    args = ap.parse_args()

    path = PROJECT_ROOT / "data" / "processed" / f"{args.format}_player_elo_ratings.parquet"
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)

    df = pd.read_parquet(path, columns=["player", "role", "raw_elo"])
    print("=" * 64)
    print(f"PLAYER-ELO DISTRIBUTION -- format={args.format}  ({len(df):,} appearance-rows)")
    print("=" * 64)

    summary = per_player_summary(df)
    stats_bat = report_role(summary, "batter", args.min_matches)
    stats_bowl = report_role(summary, "bowler", args.min_matches)
    report_anchors(summary)

    if stats_bat:
        fit_hold_center(stats_bat[99], args.target_elite, stats_bat, stats_bowl)


if __name__ == "__main__":
    main()
