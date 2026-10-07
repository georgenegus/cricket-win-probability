"""
T20/OD Win-Probability Regression Snapshot.

Freezes XGBWinProbabilityModel.predict_batch() output on a fixed 500-row-per-format
sample of existing t20 and od second-innings states. This is the Phase 0 regression
fixture for markdowns/win_predictor/hundred_brier_improvement_plan.md -- every later
phase that touches xgb_win_probability.py must reproduce these predictions exactly
(np.allclose), proving t20/od behaviour stayed byte-identical while the hnd path changed.

Usage:
    py tests/regression_snapshot_t20od.py             # create if missing, else verify
    py tests/regression_snapshot_t20od.py --create     # force-recreate the snapshot
    py tests/regression_snapshot_t20od.py --verify     # force-verify against existing snapshot
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data_processing.load_data import CricketDataLoader
from src.data_processing.match_metadata import MatchMetadataExtractor
from src.features.elo_engine import EloEngine
from src.features.player_elo_engine import PlayerEloEngine
from src.features.phase_elo_engine import PhaseEloEngine
from src.features.venue_engine import VenueEngine
from src.data_processing.build_states import StateBuilder
from src.data_processing.format_spec import spec
from src.models.xgb_win_probability import XGBWinProbabilityModel

SNAPSHOT_PATH = Path("data/processed/_regression_snapshot_t20od.parquet")
SAMPLE_SIZE_PER_FORMAT = 500


def _build_second_innings_states(format_type: str) -> pd.DataFrame:
    """Mirrors cricsheet_importer.retrain_full()'s feature pipeline exactly (cache-aware, no forced recompute)."""
    print(f"[{format_type.upper()}] Loading deliveries & metadata...")
    loader = CricketDataLoader("data/raw", format_type=format_type)
    deliveries = loader.load_all()

    meta_extractor = MatchMetadataExtractor("data/raw", format_type=format_type)
    meta_df = meta_extractor.extract_all()
    if "gender" in meta_df.columns:
        meta_df = meta_df[meta_df["gender"] == "male"].copy()
        deliveries = deliveries[deliveries["match_id"].isin(meta_df["match_id"])].copy()

    print(f"[{format_type.upper()}] Computing Elo / player Elo / phase Elo / venue features...")
    elo_engine = EloEngine(use_dynamic_k=True, use_margin=True, use_decay=True, use_home_away=False, format_type=format_type)
    elo_df = elo_engine.run(meta_df)

    player_elo_engine = PlayerEloEngine(format_type=format_type, form_window=15)
    player_elo_df = player_elo_engine.run(deliveries, elo_df)

    phase_elo_engine = PhaseEloEngine(format_type=format_type)
    phase_elo_df = phase_elo_engine.run(deliveries, elo_df)

    venue_engine = VenueEngine(format_type=format_type)
    venue_df = venue_engine.run(meta_df, deliveries)
    elo_df = elo_df.merge(venue_df, on="match_id", how="left")

    print(f"[{format_type.upper()}] Building states...")
    fmt_spec = spec(format_type)
    states = StateBuilder(
        deliveries,
        elo_df,
        player_elo_df=player_elo_df,
        phase_elo_df=phase_elo_df,
        total_overs=fmt_spec["total_overs"],
        format_type=format_type,
        balls_per_over=fmt_spec["balls_per_over"],
    ).build()

    second_innings = states[states["innings"] == 2].copy()
    print(f"[{format_type.upper()}] {len(states):,} total states, {len(second_innings):,} second-innings states.")
    return second_innings


def create_snapshot():
    """Builds t20/od second-innings states, samples 500/format, freezes predict_batch() output to parquet."""
    snapshots = []
    for format_type in ("t20", "od"):
        second_innings = _build_second_innings_states(format_type)
        n = min(SAMPLE_SIZE_PER_FORMAT, len(second_innings))
        sample = second_innings.sample(n, random_state=42).copy()

        model = XGBWinProbabilityModel(format_type=format_type)
        model.fit(None, cache_path=f"data/processed/{format_type}_xgb_models.pkl")
        sample["pred_prob_snapshot"] = model.predict_batch(sample)
        sample["_snapshot_format"] = format_type

        print(f"[{format_type.upper()}] Snapshotted {len(sample)} states. "
              f"Mean pred_prob_snapshot={sample['pred_prob_snapshot'].mean():.4f}")
        snapshots.append(sample)

    combined = pd.concat(snapshots, ignore_index=True)
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(SNAPSHOT_PATH, index=False)
    print(f"\nWrote {len(combined)} rows ({SAMPLE_SIZE_PER_FORMAT}/format target) to {SNAPSHOT_PATH}")
    return combined


def verify_snapshot():
    """Reloads the frozen snapshot, recomputes predict_batch() with current code, asserts equality per format."""
    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(f"No snapshot at {SNAPSHOT_PATH} -- run with --create first.")

    snap = pd.read_parquet(SNAPSHOT_PATH)
    all_ok = True
    for format_type, group in snap.groupby("_snapshot_format"):
        model = XGBWinProbabilityModel(format_type=format_type)
        model.fit(None, cache_path=f"data/processed/{format_type}_xgb_models.pkl")
        fresh_probs = model.predict_batch(group)
        frozen_probs = group["pred_prob_snapshot"].values

        max_abs_diff = float(np.max(np.abs(fresh_probs - frozen_probs)))
        ok = bool(np.allclose(fresh_probs, frozen_probs, atol=1e-6))
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(f"[{format_type.upper()}] {status} -- max abs diff = {max_abs_diff:.8f} across {len(group)} states")

    if all_ok:
        print("\nREGRESSION CHECK PASSED: t20/od predictions unchanged.")
    else:
        print("\nREGRESSION CHECK FAILED: t20/od predictions have drifted -- investigate before merging.")
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true", help="Force-recreate the snapshot.")
    parser.add_argument("--verify", action="store_true", help="Force-verify against the existing snapshot.")
    args = parser.parse_args()

    if args.create or (not args.verify and not SNAPSHOT_PATH.exists()):
        create_snapshot()
        print("\nRun again (no flags, or --verify) after future xgb_win_probability.py changes to confirm t20/od parity.")
    else:
        passed = verify_snapshot()
        sys.exit(0 if passed else 1)
