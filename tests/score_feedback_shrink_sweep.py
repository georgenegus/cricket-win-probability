"""
Phase 1 of mc_runrate_feedback_plan.md -- sweeps SCORE_FEEDBACK_SHRINK at much finer resolution
than the plan's original {0, .25, .5, .75, 1} grid, to find the smallest per-format shrink that
clears od's innings-total bias gate (<=13.0) without needing to guess between coarse grid points.

WHY FINER RESOLUTION THAN THE PLAN ASKED FOR, AND HOW IT'S AFFORDABLE:

  Phase 0 measured the score->runs response as ~linear (quadratic coefficient ~2e-4 in both od
  and t20 -- see mc_runrate_feedback_plan.md's Results). A near-linear response means bias vs.
  shrink is also close to linear, so a 5-point grid leaves real ambiguity about exactly where
  the <=13.0 crossing sits -- the "smallest sufficient shrink" the plan asks for could be
  anywhere between two grid points. This script uses a MUCH finer grid (default 21 points,
  0.00 to 1.00 step 0.05) plus linear interpolation between the two bracketing points to solve
  for the crossing directly, rather than rounding to whichever grid point happens to clear it.

  The naive cost of N grid points is N independent simulate_innings() calls per match. This
  script instead exploits simulate_innings()'s score_feedback_shrink ARRAY form (added
  alongside it in this same phase): one call per match simulates ALL grid points at once, as
  contiguous blocks of `--sims-per-point` sims, so the ~total_balls delivery-loop iterations
  (and their model-call overhead) are paid ONCE per match covering the whole grid, not once per
  grid point. Empirically (2026-08-10, this box) this is roughly a 1.8x speedup over N separate
  calls at the same per-point sim count -- worth taking, though it is NOT the dominant lever;
  see below.

  The bigger lever is that every grid point is evaluated on the EXACT SAME match sample (same
  squads, same actual scores) by construction -- the packed call guarantees this. Match-to-match
  squad variance is the dominant noise source in the innings-total bias metric (this is also why
  generate_mc_calibration_metrics.py's own docstring says raising its n_sims buys little over
  raising its match count) and it cancels out of the SHAPE of bias-vs-shrink even though it
  doesn't cancel out of any single point's absolute value -- which is exactly what a resolution
  sweep needs.

METHODOLOGY NOTES:
  * Only innings 1 is swept here (bias/MAE/coverage/avg_std_dev -- the plan's Phase 1
    Verification "Test 1"+"Test 4" equivalents). Win-prob agreement and O/U Brier (Tests 2/3)
    are deferred to the full-fidelity confirmation of whichever candidate this script picks,
    exactly as the plan's Verification section specifies ("final candidate only, at full
    fidelity"). Re-running all 4 tests at 21 grid points would cost ~4x for information this
    script doesn't need to make its recommendation.
  * Reduced fidelity throughout (matches generate_mc_calibration_metrics.py's own
    --n-sims 1000 "fast" convention) -- this NEVER writes to the canonical
    mc_calibration_metrics*.json path; only tests/one-time/generate_mc_calibration_metrics.py
    at its true defaults (n_sims=3000, matches=100) may do that.
  * Coverage at --sims-per-point=70 measures p5/p95 from a 70-sample empirical distribution --
    coarser than the canonical script's default n_sims=3000, so treat coverage as a directional
    canary here (per the plan's WARNING about damping collapsing simulated variance), not a
    precise number; confirm the chosen candidate's coverage at full fidelity before shipping.

Usage:
    py tests/score_feedback_shrink_sweep.py --formats od t20
    py tests/score_feedback_shrink_sweep.py --formats od --grid-step 0.02   # even finer
"""

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys
import json
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

from src.data_processing.format_spec import spec
from src.models.delivery_outcome_model import DeliveryOutcomeModel
from src.simulation.innings_simulator import simulate_innings
from src.simulation.match_simulator import extract_match_squads

# Mirrors tests/one-time/generate_mc_calibration_metrics.py's _calibration_targets exactly (not
# imported: that module's directory name has a hyphen, which is not importable as a normal
# dotted package). Keep these two in sync if the canonical script's scaling ever changes.
# Scaled by innings_total_sd, not default_venue_score -- see that function's comment for why
# (bias is a dispersion metric; the mean is the wrong basis). Changed 2026-08-11.
_T20_MAE_MAX = 25.0
_T20_BIAS_ABS_MAX = 8.0


def _bias_gate(format_type: str) -> float:
    if format_type == "t20":
        return _T20_BIAS_ABS_MAX
    scale = spec(format_type)["innings_total_sd"] / spec("t20")["innings_total_sd"]
    return round(_T20_BIAS_ABS_MAX * scale, 1)


def _select_matches(format_type: str, n_matches: int) -> pd.DataFrame:
    """Same full-quota-or-all-out innings-1 sample selection as generate_mc_calibration_metrics
    .py's Test 1, so this sweep's matches are the same population the canonical gate judges
    against (not a like-for-like RNG match, but a like-for-like SAMPLE)."""
    fmt_spec = spec(format_type)
    total_balls, bpo = fmt_spec["total_balls"], fmt_spec["balls_per_over"]
    meta = pd.read_parquet(PROJECT_ROOT / f"data/processed/{format_type}_match_metadata.parquet")
    data = pd.read_parquet(PROJECT_ROOT / f"data/processed/{format_type}_cricket_data.parquet",
                           columns=["match_id", "innings", "total_runs", "wicket", "is_wide"])
    if "gender" in meta.columns:
        meta = meta[meta["gender"] == "male"]
    inn1 = data[data["innings"] == 1]
    agg = inn1.groupby("match_id").agg(
        actual_total=("total_runs", "sum"),
        legal_balls=("is_wide", lambda x: (x == 0).sum()),
        wkts=("wicket", "sum"),
    )
    full = agg[(agg["legal_balls"] >= total_balls - bpo) | (agg["wkts"] == 10)].reset_index()
    merged = meta.merge(full, on="match_id").sort_values("match_id")
    sample_idx = np.linspace(0, len(merged) - 1, min(n_matches, len(merged)), dtype=int)
    return merged.iloc[sample_idx].reset_index(drop=True)


def sweep_format(format_type: str, grid: np.ndarray, n_matches: int, sims_per_point: int) -> dict:
    print("=" * 78)
    print(f"SCORE-FEEDBACK SHRINK SWEEP -- {format_type.upper()}")
    print(f"  grid: {len(grid)} points [{grid[0]:.2f}..{grid[-1]:.2f}], "
          f"{sims_per_point} sims/point, {n_matches} matches -> "
          f"{len(grid) * sims_per_point:,} sims/match", flush=True)
    print("=" * 78)

    fmt_spec = spec(format_type)
    total_overs, bpo = fmt_spec["total_overs"], fmt_spec["balls_per_over"]
    n_grid = len(grid)
    shrink_arr = np.repeat(grid, sims_per_point)   # contiguous blocks, one per grid point
    n_sims_total = n_grid * sims_per_point

    model = DeliveryOutcomeModel(format_type=format_type)
    model.fit(None, cache_path=PROJECT_ROOT / f"data/processed/{format_type}_delivery_outcome_model.pkl")

    sample = _select_matches(format_type, n_matches)
    print(f"  {len(sample)} sample matches selected", flush=True)

    # Per grid point: running arrays across matches, so bias/MAE/coverage/avg_std_dev are all
    # computed the same way generate_mc_calibration_metrics.py computes them (mean over matches
    # of a per-match statistic), just re-derived per shrink value instead of once.
    sim_means = np.full((len(sample), n_grid), np.nan)
    sim_p5s = np.full((len(sample), n_grid), np.nan)
    sim_p95s = np.full((len(sample), n_grid), np.nan)
    sim_stds = np.full((len(sample), n_grid), np.nan)
    actuals = np.full(len(sample), np.nan)

    t_start = time.time()
    n_ok = 0
    for i, row in sample.iterrows():
        m_id = row["match_id"]
        try:
            t1, t2, venue_score = extract_match_squads(m_id, format_type=format_type)
            res = simulate_innings(
                model=model, lineup=t1["lineup"], attack=t2["attack"],
                venue_avg_score=venue_score, n_sims=n_sims_total, total_overs=total_overs,
                balls_per_over=bpo, format_type=format_type, innings=1, target=None,
                rng=np.random.default_rng(int(m_id) % 100000), score_feedback_shrink=shrink_arr,
            )
        except Exception as e:
            print(f"  [skip] match {m_id}: {e}")
            continue
        blocks = res["final_score"].reshape(n_grid, sims_per_point)
        sim_means[i] = blocks.mean(axis=1)
        sim_stds[i] = blocks.std(axis=1)
        sim_p5s[i] = np.percentile(blocks, 5, axis=1)
        sim_p95s[i] = np.percentile(blocks, 95, axis=1)
        actuals[i] = float(row["actual_total"])
        n_ok += 1
        if n_ok % 10 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / n_ok * (len(sample) - n_ok)
            print(f"  {n_ok}/{len(sample)} matches ({elapsed/60:.1f}m elapsed, "
                  f"~{eta/60:.1f}m remaining)", flush=True)

    ok = ~np.isnan(actuals)
    actuals_ok = actuals[ok]
    n = int(ok.sum())

    table = []
    for g in range(n_grid):
        sm = sim_means[ok, g]
        bias = float((sm - actuals_ok).mean())
        mae = float(np.abs(sm - actuals_ok).mean())
        coverage = float(((actuals_ok >= sim_p5s[ok, g]) & (actuals_ok <= sim_p95s[ok, g])).mean())
        avg_std = float(sim_stds[ok, g].mean())
        table.append({
            "shrink": round(float(grid[g]), 4), "bias": bias, "mae": mae,
            "coverage": coverage, "avg_std_dev": avg_std,
        })

    bias_gate = _bias_gate(format_type)
    recommended = _smallest_shrink_clearing_gate(table, bias_gate)

    print(f"\n  {'shrink':>7s} {'bias':>8s} {'mae':>7s} {'coverage':>9s} {'avg_std':>8s}")
    for r in table:
        flag = " <=GATE" if abs(r["bias"]) <= bias_gate else ""
        print(f"  {r['shrink']:>7.2f} {r['bias']:>+8.2f} {r['mae']:>7.2f} "
              f"{r['coverage']:>9.1%} {r['avg_std_dev']:>8.1f}{flag}")

    print(f"\n  bias gate: |bias| <= {bias_gate}")
    print(f"  recommended shrink (interpolated, smallest that clears the gate): "
          f"{recommended['shrink'] if recommended else 'NONE -- no grid point clears the gate'}")
    if recommended:
        print(f"    interpolated bias at that shrink: {recommended['interpolated_bias']:+.2f}")

    return {
        "format": format_type,
        "n_matches": n,
        "n_grid": n_grid,
        "sims_per_point": sims_per_point,
        "bias_gate": bias_gate,
        "table": table,
        "recommended": recommended,
    }


def _smallest_shrink_clearing_gate(table, bias_gate):
    """table sorted by ascending shrink (0=full damping .. 1=no-op). Finds the SMALLEST shrink
    (least damping removed, i.e. closest to 1.0... wait -- smallest shrink = MOST damping) at
    which |bias| first drops to/under the gate, scanning from shrink=1.0 DOWN to 0.0 (today's
    behaviour is shrink=1.0; damping is only added until it's just enough). Linearly interpolates
    between the two bracketing grid points for the exact crossing rather than snapping to a
    grid point -- the whole reason for a finer grid over the plan's original 5-point one.

    "Smallest intervention that clears the gate" (mc_runrate_feedback_plan.md Phase 1 step 4)
    means the LARGEST shrink (least damping, closest to the reversible shrink=1.0 no-op) that
    still satisfies the gate -- scanning from shrink=1.0 downward and stopping at the first
    (highest-shrink) point where the gate is met.
    """
    pts = sorted(table, key=lambda r: -r["shrink"])  # 1.0 -> 0.0
    if abs(pts[0]["bias"]) <= bias_gate:
        return {"shrink": pts[0]["shrink"], "interpolated_bias": pts[0]["bias"],
                "note": "no damping needed -- shrink=1.0 already clears the gate"}
    for a, b in zip(pts, pts[1:]):
        # a has the larger shrink (checked already, gate NOT met); b has less shrink.
        if abs(b["bias"]) <= bias_gate:
            # Interpolate bias linearly in shrink between a (fails) and b (passes) and solve
            # for |bias| == bias_gate (bias is monotone in shrink in every case measured so
            # far -- Phase 0 found near-linear, same-signed responses in both formats).
            s_a, s_b = a["shrink"], b["shrink"]
            bias_a, bias_b = a["bias"], b["bias"]
            target = bias_gate if bias_b >= 0 else -bias_gate
            if bias_b == bias_a:
                frac = 1.0
            else:
                frac = float(np.clip((target - bias_a) / (bias_b - bias_a), 0.0, 1.0))
            s_star = s_a + frac * (s_b - s_a)
            interp_bias = bias_a + frac * (bias_b - bias_a)
            return {"shrink": round(float(s_star), 4), "interpolated_bias": round(float(interp_bias), 3),
                    "bracket": [b["shrink"], a["shrink"]]}
    return None  # not even shrink=0.0 clears the gate


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--formats", nargs="+", default=["od", "t20"], choices=["t20", "od"])
    p.add_argument("--grid-step", type=float, default=0.05,
                   help="Grid resolution on shrink in [0,1]. 0.05 -> 21 points.")
    p.add_argument("--matches", type=int, default=100)
    p.add_argument("--sims-per-point", type=int, default=70)
    args = p.parse_args()

    grid = np.round(np.arange(0.0, 1.0 + 1e-9, args.grid_step), 6)
    grid = np.clip(grid, 0.0, 1.0)

    results = {}
    for fmt in args.formats:
        t0 = time.time()
        results[fmt] = sweep_format(fmt, grid, args.matches, args.sims_per_point)
        print(f"  [{fmt}] done in {(time.time()-t0)/60:.1f}m\n", flush=True)
        out_path = PROJECT_ROOT / "outputs" / f"score_feedback_shrink_sweep_{fmt}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results[fmt], indent=2))
        print(f"  Wrote {out_path}")

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for fmt, r in results.items():
        rec = r["recommended"]
        print(f"{fmt}: recommended shrink = "
              f"{rec['shrink'] if rec else 'NONE'} "
              f"(gate |bias|<={r['bias_gate']})")


if __name__ == "__main__":
    main()
