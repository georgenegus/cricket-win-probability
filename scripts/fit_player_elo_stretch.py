"""
Re-fits a format's `PlayerEloEngine.*_ELO_STRETCH` so the batter career-peak p99 lands on a
target (default 2000 -- the "world-class ceiling" the hold-center calibration pins alongside
the fixed 1500 center).

Why this exists as a real script rather than an ad hoc runner: the stretch is NOT a
set-and-forget constant. It is fitted against a distribution that the format's *team* Elo
feeds into (player Elo anchors on the opposing bowling-squad Elo, whose fallback anchor is
`_team_to_player_scale(team_elo)`, and on the Phase-4 opponent-quality weight), and
`_team_to_player_scale` amplifies every team-Elo point by the stretch itself. So any change to
team Elo -- including a legitimate correction like the 2026-08-06 stale-cache rebuild, which
moved t20 team Elo on 3,697/10,634 matches by up to ~400 points -- silently decalibrates the
player scale. t20_player_elo_revamp_plan.md Phase 6 fitted 2.48 to land p99 at 1999.8; by
2026-08-17 that same measurement read 1865.3 with no code change at all. This script is how
you re-pin it without hand-rolling a scratch runner each time.

Method: iterate `PlayerEloEngine` ONLY (never a model retrain) against candidate stretch
values, writing to a scratch cache directory so the production parquets are never touched.
A pure affine rescale would make (p99 - 1500) exactly proportional to the stretch, giving a
closed-form answer in one step; the kept non-affine dynamics phases (experience-graded K,
opponent-quality weighting) break that slightly, so we secant-iterate to the tolerance
instead of trusting the first algebraic guess. This mirrors Phase 6's own 3.22 -> 2.45 -> 2.48
hand-iteration, which is exactly the loop being automated here.

Usage:
    py scripts/fit_player_elo_stretch.py --format t20 --min-matches 50
    py scripts/fit_player_elo_stretch.py --format hnd --min-matches 20

The fitted value is PRINTED, not written to source -- shipping it is a deliberate edit to the
class constant followed by a full production retrain + gate re-run (see the plans' ship step).
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.format_spec import SEED_SOURCE, spec  # noqa: E402
from src.data_processing.load_data import CricketDataLoader  # noqa: E402
from src.features.player_elo_engine import PlayerEloEngine, build_seed_ratings  # noqa: E402

CENTER = 1500.0

# The class attribute each format's stretch lives on. Formats absent here are unstretched by
# design (women's formats are all 1.0, never measured for their own populations) -- refuse to
# "fit" one rather than silently inventing a men's-population-shaped constant for it.
STRETCH_ATTR = {
    "t20": "T20_ELO_STRETCH",
    "od": "OD_ELO_STRETCH",
    "hnd": "HND_ELO_STRETCH",
}


def load_inputs(format_type: str):
    """Deliveries + team Elo for one format, WITHOUT re-running MatchMetadataExtractor.

    Two deliberate choices here:
      - Team Elo is read straight off `{fmt}_elo_ratings.parquet` (the exact output of
        `EloEngine.run(meta_df)`) rather than recomputed. Rebuilding it would require
        MatchMetadataExtractor's spawn-based worker pool, which OOMs this 7.7GB machine when
        anything else is resident, and would produce a byte-identical result anyway.
      - Gender filtering is implicit: that parquet was built from an already gender-filtered
        meta_df, so restricting deliveries to its match_ids reproduces retrain_full's filter
        exactly. Never re-derive the filter here with a hardcoded gender -- see
        .agents/instructions.md.
    """
    elo_path = PROJECT_ROOT / "data" / "processed" / f"{format_type}_elo_ratings.parquet"
    if not elo_path.exists():
        raise SystemExit(f"ERROR: {elo_path} not found -- run a retrain for {format_type} first.")
    elo_df = pd.read_parquet(elo_path)

    loader = CricketDataLoader(str(PROJECT_ROOT / "data" / "raw"), format_type=format_type)
    deliveries = loader.load_all()
    deliveries = deliveries[deliveries["match_id"].isin(set(elo_df["match_id"]))].copy()

    print(f"[{format_type}] {len(elo_df):,} matches, {len(deliveries):,} deliveries "
          f"(gender={spec(format_type)['gender']})")
    return deliveries, elo_df


def batter_peak_p99(player_elo_df: pd.DataFrame, min_matches: int) -> tuple:
    """(p99, p50, n) of career-peak raw_elo over batters with >= min_matches appearances.

    Identical aggregation to scripts/measure_player_elo_distribution.py (career peak =
    max raw_elo per player), so the numbers here are directly comparable to the plans'
    tracked figures rather than a subtly different statistic.
    """
    bat = player_elo_df[player_elo_df["role"] == "batter"]
    per_player = bat.groupby("player")["raw_elo"].agg(peak="max", n="size")
    sub = per_player[per_player["n"] >= min_matches]
    if sub.empty:
        raise SystemExit(f"ERROR: no batters with >= {min_matches} appearances.")
    return (
        float(np.percentile(sub["peak"].values, 99)),
        float(np.percentile(sub["peak"].values, 50)),
        int(len(sub)),
    )


def run_at_stretch(format_type, stretch, deliveries, elo_df, precomputed, seed_ratings, scratch_dir):
    """One PlayerEloEngine run at `stretch`, isolated to `scratch_dir`.

    The constant is set on the CLASS before construction because `__init__` bakes it into
    elo_divisor/k_runs/k_wickets, while `_team_to_player_scale` and run()'s seed remap read it
    live off the class -- setting it in one place keeps all three consistent.
    """
    setattr(PlayerEloEngine, STRETCH_ATTR[format_type], float(stretch))
    engine = PlayerEloEngine(
        format_type=format_type,
        form_window=15,          # production value (cricsheet_importer.retrain_full)
        processed_path=str(scratch_dir),
    )
    assert engine.elo_divisor == 400.0 * stretch, "stretch did not reach elo_divisor"
    return engine.run(
        deliveries, elo_df, force=True,
        precomputed_stats=precomputed, seed_ratings=seed_ratings,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="t20", choices=sorted(STRETCH_ATTR))
    ap.add_argument("--min-matches", type=int, default=50,
                    help="appearance cut for the peak distribution (t20 plans used 50, hnd 20)")
    ap.add_argument("--target", type=float, default=2000.0)
    ap.add_argument("--tol", type=float, default=3.0, help="accept when |p99 - target| <= tol")
    ap.add_argument("--max-iter", type=int, default=6)
    args = ap.parse_args()

    fmt = args.format
    attr = STRETCH_ATTR[fmt]
    baseline = getattr(PlayerEloEngine, attr)
    print("=" * 70)
    print(f"FIT {attr}  target batter-peak p99 = {args.target:.1f}  "
          f"(>= {args.min_matches} matches)  current constant = {baseline}")
    print("=" * 70)

    deliveries, elo_df = load_inputs(fmt)

    # Aggregate deliveries ONCE -- it is stretch-independent (pure counting of runs/balls/
    # wickets), so re-doing it per candidate would dominate runtime for no reason.
    probe = PlayerEloEngine(format_type=fmt, processed_path=str(PROJECT_ROOT / "data" / "processed"))
    precomputed = probe._aggregate_match_stats(deliveries)

    # A pooled format warm-starts from its SEED_SOURCE frozen snapshot. Read it (refresh=False,
    # so the frozen file is untouched) and pass it in: without the seed, hnd's ratings would be
    # computed from a cold start and the fitted constant would not match production's recipe.
    seed_ratings = build_seed_ratings(fmt) if fmt in SEED_SOURCE else None

    scratch_dir = Path(tempfile.mkdtemp(prefix=f"elostretch_{fmt}_"))
    history = []
    try:
        stretch = baseline
        for i in range(1, args.max_iter + 1):
            pdf = run_at_stretch(fmt, stretch, deliveries, elo_df, precomputed, seed_ratings, scratch_dir)
            p99, p50, n = batter_peak_p99(pdf, args.min_matches)
            history.append((stretch, p99, p50))
            print(f"\n  iter {i}: {attr}={stretch:.4f} -> batter peak p99={p99:.1f} "
                  f"(p50={p50:.1f}, n={n}, delta from target {p99 - args.target:+.1f})")

            if abs(p99 - args.target) <= args.tol:
                print(f"\n  CONVERGED within +/-{args.tol}")
                break

            # Affine-implied correction: (p99 - 1500) is proportional to the stretch under a
            # pure rescale, so scale by the ratio of desired to achieved spread. The dynamics
            # phases make this approximate, hence the loop.
            ratio = (args.target - CENTER) / (p99 - CENTER)
            stretch = stretch * ratio
        else:
            print(f"\n  WARNING: hit --max-iter {args.max_iter} without reaching +/-{args.tol}")
    finally:
        setattr(PlayerEloEngine, attr, baseline)   # never leave the class mutated
        shutil.rmtree(scratch_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print(f"{'candidate':>12}  {'peak p99':>9}  {'peak p50':>9}")
    for s, p99, p50 in history:
        print(f"{s:12.4f}  {p99:9.1f}  {p50:9.1f}")
    best = min(history, key=lambda h: abs(h[1] - args.target))
    print(f"\nFITTED {attr} = {best[0]:.4f}   (p99={best[1]:.1f}, was {baseline} at "
          f"p99={history[0][1]:.1f})")
    print("Production parquets untouched -- ship by editing the class constant, then retrain.")


if __name__ == "__main__":
    # Required: CricketDataLoader's ProcessPoolExecutor (cold YAML parse) and
    # MatchMetadataExtractor both use spawn on Windows, which re-imports this module per
    # worker. Without the guard each worker re-runs the whole fit. See the machine-memory
    # note in scripts/import_and_retrain.py.
    main()
