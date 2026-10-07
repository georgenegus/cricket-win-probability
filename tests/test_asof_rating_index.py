"""
Parity test for AsOfRatingIndex (franchise_bridge_plan.md Phase 1).

The index replaces ICCPlayerRatingEngine's per-call ledger scan with prefix sums + binary
search. That is only safe if it returns EXACTLY the same numbers, including at the piecewise
time-weighting boundaries (FULL_WEIGHT_DAYS / DROP_WEIGHT_DAYS), where an off-by-one in the
searchsorted side= or cutoff arithmetic would silently shift a match between weight bands.

Usage:
    py tests/test_asof_rating_index.py
    py tests/test_asof_rating_index.py --format od
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import argparse
import random
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.icc_player_rating_engine import ICCPlayerRatingEngine, AsOfRatingIndex

TOL = 1e-9


def build_ledgers(icc_player_df: pd.DataFrame) -> dict:
    """(player, role) -> [(date, points)], the state run() would have built in-process."""
    ledgers = {}
    for row in icc_player_df.itertuples(index=False):
        ledgers.setdefault((row.player, row.role), []).append((row.match_date, row.points_earned))
    return ledgers


def sample_query_dates(dates, rng) -> list:
    """A spread of as-of dates per player, deliberately including the weight-band boundaries
    where an off-by-one would hide: exactly FULL_WEIGHT_DAYS / DROP_WEIGHT_DAYS after a real
    appearance, and +/- 1 day either side of each."""
    out = []
    anchor = dates[len(dates) // 2]
    for offset_days in (
        0, 1, 7,
        ICCPlayerRatingEngine.FULL_WEIGHT_DAYS - 1,
        ICCPlayerRatingEngine.FULL_WEIGHT_DAYS,
        ICCPlayerRatingEngine.FULL_WEIGHT_DAYS + 1,
        ICCPlayerRatingEngine.DROP_WEIGHT_DAYS - 1,
        ICCPlayerRatingEngine.DROP_WEIGHT_DAYS,
        ICCPlayerRatingEngine.DROP_WEIGHT_DAYS + 1,
        3000,
    ):
        out.append(pd.Timestamp(anchor) + pd.Timedelta(days=offset_days))
    out.append(pd.Timestamp(dates[0]) - pd.Timedelta(days=5))   # before any appearance
    out.append(pd.Timestamp(rng.choice(dates)))                  # exactly on an appearance
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", default="t20")
    parser.add_argument("--players", type=int, default=60, help="How many players to sample.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    path = PROJECT_ROOT / "data" / "processed" / f"{args.format}_icc_player_ratings.parquet"
    if not path.exists():
        print(f"SKIP: {path} not found -- run scripts/measure_icc_rating_crosscheck.py "
              f"--formats {args.format} first.")
        return 0

    df = pd.read_parquet(path)
    df["match_date"] = pd.to_datetime(df["match_date"])

    index = AsOfRatingIndex(df)
    engine = ICCPlayerRatingEngine(format_type=args.format)
    all_ledgers = build_ledgers(df)

    rng = random.Random(args.seed)
    keys = sorted(all_ledgers.keys())
    sampled = rng.sample(keys, min(args.players, len(keys)))

    raw_checks = 0
    eff_checks = 0
    failures = []

    for player, role in sampled:
        ledger = sorted(all_ledgers[(player, role)], key=lambda t: t[0])
        dates = [d for d, _ in ledger]

        for as_of in sample_query_dates(dates, rng):
            # The index is defined on innings STRICTLY BEFORE as_of, which is precisely the
            # ledger state run() holds when computing a pre-match rating. Truncate to
            # reproduce that state, and compare both raw and effective against it.
            truncated = [(d, p) for d, p in ledger if d < as_of]
            engine.ledgers = {(player, role): truncated}

            expected_raw = engine.get_raw_rating(player, role, as_of)
            actual_raw = index.raw_rating(player, role, as_of)
            raw_checks += 1
            if abs(expected_raw - actual_raw) > TOL:
                failures.append(
                    f"RAW  {player!r}/{role} @ {as_of.date()}: engine={expected_raw:.10f} "
                    f"index={actual_raw:.10f} (diff {expected_raw - actual_raw:.2e})"
                )

            expected_eff = engine.get_effective_rating(player, role, as_of)
            actual_eff = index.effective_rating(player, role, as_of)
            eff_checks += 1
            if abs(expected_eff - actual_eff) > TOL:
                failures.append(
                    f"EFF  {player!r}/{role} @ {as_of.date()}: engine={expected_eff:.10f} "
                    f"index={actual_eff:.10f} (diff {expected_eff - actual_eff:.2e})"
                )

    print(f"format={args.format}  players sampled={len(sampled)}  "
          f"raw checks={raw_checks}  effective checks={eff_checks}")
    if failures:
        print(f"\nFAILED: {len(failures)} mismatch(es):")
        for f in failures[:20]:
            print("  " + f)
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        return 1

    print("PASS: AsOfRatingIndex matches ICCPlayerRatingEngine exactly, including at the "
          "FULL_WEIGHT_DAYS / DROP_WEIGHT_DAYS boundaries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
