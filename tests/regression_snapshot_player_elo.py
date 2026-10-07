"""
Frozen-Format Player-Elo Regression Snapshot.

Generic freeze gate reused across the player-Elo revamp plans in markdowns/win_predictor/:
shipped 2026-08-03 for t20_player_elo_revamp_plan.md (froze od+hnd while t20 was being
tuned); re-pointed for od_player_elo_revamp_plan.md (froze t20+hnd while od was being
tuned); now re-pointed again for hnd_player_elo_revamp_plan.md (freezes t20+od while hnd is
being tuned -- see FROZEN_FORMATS below).

Freezes PlayerEloEngine output (raw_elo + effective_elo) on a fixed sample of the frozen
formats' player appearances. Every later phase re-runs the engine with force=True (bypassing
cache) and asserts np.allclose against this frozen sample -- the tightest possible gate on
"did the actively-tuned format's edit leak into a frozen one".

(The format under active tuning is deliberately NOT frozen here -- it is the thing changing;
its movement is measured by the win-prob Brier baseline in tests/regression_test.py, not
asserted to zero.)

Usage:
    py tests/regression_snapshot_player_elo.py            # create if missing, else verify
    py tests/regression_snapshot_player_elo.py --create   # force-recreate the snapshot
    py tests/regression_snapshot_player_elo.py --verify    # force-verify (re-runs engine, force=True)
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
from src.features.player_elo_engine import PlayerEloEngine, build_hnd_seed_ratings

SNAPSHOT_PATH = Path("data/processed/_regression_snapshot_player_elo.parquet")
SAMPLE_SIZE_PER_FORMAT = 500
# t20_player_elo_revamp_plan.md (shipped 2026-08-03) froze ("od", "hnd") while t20 was being
# tuned. od_player_elo_revamp_plan.md then tuned od, flipping the frozen set to t20+hnd.
# hnd_player_elo_revamp_plan.md now tunes hnd, so the frozen set flips to t20+od.
FROZEN_FORMATS = ("t20", "od")
KEY = ["match_id", "player", "role"]


def compute_player_elo(format_type: str, force: bool, cache_override: Path = None) -> pd.DataFrame:
    """Runs the pipeline up to (and including) player Elo, mirroring retrain_full() EXACTLY:
    form_window=15 for every format, and hnd warm-seeded from the latest T20 effective_elo.
    (Omitting the seed diverges from the production parquet by ~157 Elo -- proven in Phase 0.)

    force=False reads the valid cache (current-code output); force=True bypasses it to exercise
    whatever the engine code currently produces -- that's what verify must use. cache_override
    redirects the engine's OUTPUT parquet so a force-recompute never clobbers the production
    cache (verify points it at a disposable scratch path).
    """
    loader = CricketDataLoader(str(PROJECT_ROOT / "data" / "raw"), format_type=format_type)
    deliveries = loader.load_all()

    meta_extractor = MatchMetadataExtractor(str(PROJECT_ROOT / "data" / "raw"), format_type=format_type)
    meta_df = meta_extractor.extract_all()
    if "gender" in meta_df.columns:
        meta_df = meta_df[meta_df["gender"] == "male"].copy()
        deliveries = deliveries[deliveries["match_id"].isin(meta_df["match_id"])].copy()

    elo_engine = EloEngine(use_dynamic_k=True, use_margin=True, use_decay=True, use_home_away=False, format_type=format_type)
    elo_df = elo_engine.run(meta_df)

    # hnd warm-seeds from a FROZEN snapshot of T20's effective_elo, not the live parquet --
    # see build_hnd_seed_ratings' docstring. Replicates retrain_full verbatim.
    seed_ratings = build_hnd_seed_ratings() if format_type == "hnd" else None

    player_elo_engine = PlayerEloEngine(format_type=format_type, form_window=15)
    if cache_override is not None:
        player_elo_engine.cache_file = Path(cache_override)
    return player_elo_engine.run(deliveries, elo_df, force=force, seed_ratings=seed_ratings)


def create_snapshot():
    frames = []
    for fmt in FROZEN_FORMATS:
        print(f"[{fmt.upper()}] Computing player Elo (from cache)...")
        pe = compute_player_elo(fmt, force=False)
        n = min(SAMPLE_SIZE_PER_FORMAT, len(pe))
        sample = pe.sample(n, random_state=42)[KEY + ["raw_elo", "effective_elo"]].copy()
        sample["_snapshot_format"] = fmt
        print(f"[{fmt.upper()}] Froze {n} rows. mean raw_elo={sample['raw_elo'].mean():.4f}, "
              f"mean effective_elo={sample['effective_elo'].mean():.4f}")
        frames.append(sample)

    combined = pd.concat(frames, ignore_index=True)
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(SNAPSHOT_PATH, index=False)
    print(f"\nWrote {len(combined)} rows to {SNAPSHOT_PATH}")
    return combined


def verify_snapshot():
    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(f"No snapshot at {SNAPSHOT_PATH} -- run with --create first.")

    snap = pd.read_parquet(SNAPSHOT_PATH)
    all_ok = True
    for fmt in FROZEN_FORMATS:
        frozen = snap[snap["_snapshot_format"] == fmt]
        if frozen.empty:
            continue
        print(f"[{fmt.upper()}] Recomputing player Elo (force=True, scratch cache)...")
        scratch = PROJECT_ROOT / "data" / "processed" / f"_verify_{fmt}_player_elo.parquet"
        fresh = compute_player_elo(fmt, force=True, cache_override=scratch)
        merged = frozen.merge(fresh[KEY + ["raw_elo", "effective_elo"]], on=KEY,
                              how="left", suffixes=("_frozen", "_fresh"))

        missing = merged["raw_elo_fresh"].isna().sum()
        raw_diff = float(np.max(np.abs(merged["raw_elo_frozen"] - merged["raw_elo_fresh"]).fillna(np.inf)))
        eff_diff = float(np.max(np.abs(merged["effective_elo_frozen"] - merged["effective_elo_fresh"]).fillna(np.inf)))
        ok = bool(missing == 0 and np.isclose(raw_diff, 0.0, atol=1e-6) and np.isclose(eff_diff, 0.0, atol=1e-6))
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(f"[{fmt.upper()}] {status} -- {len(frozen)} rows, {missing} missing, "
              f"max raw diff={raw_diff:.8f}, max effective diff={eff_diff:.8f}")

    frozen_label = "/".join(f.upper() for f in FROZEN_FORMATS)
    if all_ok:
        print(f"\nPLAYER-ELO REGRESSION PASSED: {frozen_label} unchanged.")
    else:
        print(f"\nPLAYER-ELO REGRESSION FAILED: {frozen_label} drifted -- an active-format-gated edit leaked.")
    return all_ok


def repair_production_caches():
    """Recompute the frozen formats' player-Elo with the correct production recipe
    (form_window=15, hnd seeded) straight to the PRODUCTION parquet -- restores caches after a
    buggy force-recompute corrupted them (e.g. a wrong form_window clobbering recent_form, or
    hnd computed un-seeded)."""
    for fmt in FROZEN_FORMATS:
        print(f"[{fmt.upper()}] Repairing production player-Elo cache (force=True)...")
        pe = compute_player_elo(fmt, force=True)  # cache_override=None -> production path
        print(f"[{fmt.upper()}] Rewrote {len(pe):,} rows to production cache. "
              f"mean raw_elo={pe['raw_elo'].mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--repair", action="store_true",
                        help="Recompute the frozen formats' production caches with the correct recipe.")
    args = parser.parse_args()

    if args.repair:
        repair_production_caches()
    elif args.create or (not args.verify and not SNAPSHOT_PATH.exists()):
        create_snapshot()
        print(f"\nRun with --verify after player_elo_engine.py changes to confirm {'/'.join(FROZEN_FORMATS)} parity.")
    else:
        passed = verify_snapshot()
        sys.exit(0 if passed else 1)
