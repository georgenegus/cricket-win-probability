"""
Follow-up diagnostic for franchise_bridge_plan.md Phase 3.

The bridge agrees with international ratings at Spearman 0.758 for batting but only 0.402 for
bowling. The phase-bias hypothesis was tested (scripts/measure_bowling_phase_bias.py) and
largely refuted -- death-heavy and death-light bowlers show near-identical agreement.

This tests the remaining explanation: that the CEILING is different. A bowler bowls at most 4
overs per innings, so each observation is far noisier than a batter's innings. If international
bowling ratings are themselves substantially less self-consistent than batting ratings, then
0.402 may be close to the best any method could do, and the bridge is not the weak link.

Method: split-half reliability. Randomly split each player's innings into two halves, compute a
rating from each half independently, and correlate the halves across players. That is an upper
bound on agreement with ANY external measure of the same quantity (a measure cannot correlate
with something else better than it correlates with itself), and is directly comparable between
roles.

Usage:
    py scripts/measure_rating_reliability.py
    py scripts/measure_rating_reliability.py --trials 20 --min-innings 20
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.icc_player_rating_engine import ICCPlayerRatingEngine


def split_half_reliability(df: pd.DataFrame, role: str, min_innings: int, trials: int, seed: int):
    """Mean split-half Spearman across `trials` random splits, Spearman-Brown corrected.

    Each half's rating is the plain mean of that half's points_earned -- deliberately NOT the
    time-weighted/shrunk rating, because a random split destroys the time structure. What is
    being measured is how much signal the per-innings points carry about a player, per role.
    """
    role_df = df[df["role"] == role]
    counts = role_df["player"].value_counts()
    eligible = counts[counts >= min_innings].index
    role_df = role_df[role_df["player"].isin(eligible)]
    if role_df["player"].nunique() < 20:
        return None

    by_player = {p: g["points_earned"].to_numpy(dtype=float)
                 for p, g in role_df.groupby("player", sort=False)}

    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(trials):
        a_vals, b_vals = [], []
        for pts in by_player.values():
            idx = rng.permutation(len(pts))
            half = len(pts) // 2
            a_vals.append(pts[idx[:half]].mean())
            b_vals.append(pts[idx[half:2 * half]].mean())
        rho, _ = spearmanr(a_vals, b_vals)
        rhos.append(rho)

    r_half = float(np.mean(rhos))
    # Spearman-Brown: reliability of the FULL-length measure, given two half-length halves
    r_full = (2 * r_half) / (1 + r_half) if r_half > -1 else float("nan")
    return {
        "n_players": len(by_player),
        "mean_innings": float(np.mean([len(v) for v in by_player.values()])),
        "split_half_rho": round(r_half, 3),
        "full_length_reliability": round(r_full, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", default="t20")
    parser.add_argument("--min-innings", type=int, default=ICCPlayerRatingEngine.MIN_MATCHES_TO_RANK)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    proc = PROJECT_ROOT / "data" / "processed"
    sources = {
        "international": proc / f"{args.format}_icc_player_ratings.parquet",
        "franchise bridge": proc / f"{args.format}_franchise_bridge_tiera.parquet",
    }

    print(f"Split-half reliability (min {args.min_innings} innings, {args.trials} random splits)\n")
    print(f"{'source':<20}{'role':<10}{'players':>9}{'mean inns':>11}{'split-half':>12}{'full-length':>13}")
    results = {}
    for label, path in sources.items():
        if not path.exists():
            print(f"{label:<20} SKIP -- {path.name} not found")
            continue
        df = pd.read_parquet(path)
        for role in ("batter", "bowler"):
            res = split_half_reliability(df, role, args.min_innings, args.trials, args.seed)
            if res is None:
                print(f"{label:<20}{role:<10} too few players")
                continue
            results[(label, role)] = res
            print(f"{label:<20}{role:<10}{res['n_players']:>9}{res['mean_innings']:>11.1f}"
                  f"{res['split_half_rho']:>12}{res['full_length_reliability']:>13}")

    print("\nInterpretation:")
    print("  'full-length reliability' is how well a rating built from these innings correlates")
    print("  with itself -- an UPPER BOUND on agreement with any external measure.")

    # Observed gate values are read from the validation run's own output rather than hardcoded,
    # so this can never print a stale number after a parameter change.
    gate_path = proc / f"{args.format}_franchise_bridge_validation.json"
    if not gate_path.exists():
        print(f"  (run scripts/measure_franchise_bridge_validation.py to compare against the "
              f"observed gate -- {gate_path.name} not found)")
        return
    with open(gate_path, encoding="utf-8") as f:
        gate = json.load(f).get("gate", {})

    print()
    for role_key, label in (("batter", "batting"), ("bowler", "bowling")):
        observed = gate.get(role_key, {}).get("spearman_r")
        bridge_res = results.get(("franchise bridge", role_key))
        intl_res = results.get(("international", role_key))
        if observed is None or not bridge_res or not intl_res:
            continue
        # attenuation: a correlation between two noisy measures is bounded by sqrt(r_xx * r_yy)
        ceiling = float(np.sqrt(intl_res["full_length_reliability"] * bridge_res["full_length_reliability"]))
        pct = observed / ceiling * 100 if ceiling > 0 else float("nan")
        print(f"  {label}: observed {observed:.3f} vs attenuation ceiling {ceiling:.3f} "
              f"-> {pct:.0f}% of attainable (disattenuated true agreement ~{observed / ceiling:.2f})")


if __name__ == "__main__":
    main()
