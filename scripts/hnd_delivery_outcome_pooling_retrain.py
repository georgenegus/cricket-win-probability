"""
One-off runner for markdowns/win_predictor/hnd_delivery_outcome_pooling_plan.md: retrains ONLY
hnd's delivery-outcome model (the Monte Carlo simulator's per-ball engine), now pooled with T20
deliveries (DeliveryOutcomeModel.DELIVERY_OUTCOME_POOL_SOURCE = {"hnd": "t20"}, wired into
retrain_full()'s step 6 in cricsheet_importer.py).

Deliberately narrower than retrain_full('hnd'): mirrors only that function's steps 1-2 (load hnd
deliveries + compute its own Elo/player-Elo/venue features) plus the delivery-outcome fit itself.
hnd's first-innings regressor and win-prob classifier are NOT retrained here -- neither their
inputs nor DeliveryOutcomeModel change, so retraining them too would spend real compute for zero
expected effect while adding pure retrain-variance risk to metrics that are out of scope for this
change (same "one retrain, only what's needed" discipline as effective_game_elo_plan.md Phase 5's
scope narrowing).

Backs up the existing hnd_delivery_outcome_model.pkl (never deletes) before fitting -- model
caches have no source-hash check (fit(cache_path=...) only tests existence), so a genuine retrain
requires the file to be gone from that path.

Usage:
    py scripts/hnd_delivery_outcome_pooling_retrain.py
"""

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

FORMAT = "hnd"


def main():
    from src.data_processing.load_data import CricketDataLoader
    from src.data_processing.match_metadata import MatchMetadataExtractor
    from src.data_processing.format_spec import spec
    from src.features.elo_engine import EloEngine
    from src.features.player_elo_engine import PlayerEloEngine
    from src.features.venue_engine import VenueEngine
    from src.models.delivery_outcome_model import DeliveryOutcomeModel, build_training_frame
    from src.data_processing.cricsheet_importer import _build_delivery_outcome_pool_frame

    print("=" * 78)
    print("HND DELIVERY-OUTCOME MODEL RETRAIN -- pooled with T20 (DELIVERY_OUTCOME_POOL_SOURCE)")
    print("=" * 78)
    t0 = time.time()

    # ---- steps 1-2 of retrain_full('hnd'), hnd's own data only ----
    print(f"\n[{FORMAT.upper()}] Loading deliveries + metadata...")
    loader = CricketDataLoader("data/raw", format_type=FORMAT)
    deliveries = loader.load_all()

    meta = MatchMetadataExtractor("data/raw", format_type=FORMAT)
    meta_df = meta.extract_all()
    if "gender" in meta_df.columns:
        meta_df = meta_df[meta_df["gender"] == spec(FORMAT)["gender"]].copy()
        deliveries = deliveries[deliveries["match_id"].isin(meta_df["match_id"])].copy()

    print(f"[{FORMAT.upper()}] Computing Elo ratings...")
    elo_engine = EloEngine(use_dynamic_k=True, use_margin=True, use_decay=True, use_home_away=False, format_type=FORMAT)
    elo_df = elo_engine.run(meta_df)

    print(f"[{FORMAT.upper()}] Computing player Elo ratings...")
    player_elo_engine = PlayerEloEngine(format_type=FORMAT, form_window=15)
    player_elo_df = player_elo_engine.run(deliveries, elo_df)

    print(f"[{FORMAT.upper()}] Computing venue features...")
    venue_engine = VenueEngine(format_type=FORMAT)
    venue_df = venue_engine.run(meta_df, deliveries)

    training_frame = build_training_frame(deliveries, player_elo_df, venue_df, format_type=FORMAT)
    print(f"[{FORMAT.upper()}] {len(training_frame):,} deliveries, {training_frame['match_id'].nunique():,} matches")

    # ---- T20 pool (same helper retrain_full() now calls) ----
    print("\n[T20] Building pool frame...")
    pool_frame = _build_delivery_outcome_pool_frame("t20")
    print(f"[T20] {len(pool_frame):,} pool deliveries, {pool_frame['match_id'].nunique():,} matches (train-only)")

    # ---- back up the existing artifact, never delete ----
    dom_path = PROJECT_ROOT / "data" / "processed" / f"{FORMAT}_delivery_outcome_model.pkl"
    if dom_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = dom_path.with_suffix(f".pre_pooling_bak.{stamp}.pkl")
        shutil.copyfile(dom_path, backup_path)
        print(f"\n[backup] {dom_path.name} -> {backup_path.name}")
        dom_path.unlink()
        print(f"[backup] removed {dom_path} to force a genuine refit (no source-hash check on this cache)")

    # ---- fit, pooled ----
    print(f"\n[{FORMAT.upper()}] Fitting pooled DeliveryOutcomeModel...")
    dom_model = DeliveryOutcomeModel(format_type=FORMAT)
    assert dom_model.pooled, "DELIVERY_OUTCOME_POOL_SOURCE lookup failed -- expected hnd to be pooled"
    dom_model.fit(training_frame, cache_path=dom_path, pool_frame=pool_frame)

    print(f"\n[OK] Retrain complete in {time.time() - t0:.0f}s. "
          f"self.features ({len(dom_model.features)}): {dom_model.features}")
    print(f"Saved to {dom_path}")


if __name__ == "__main__":
    main()
