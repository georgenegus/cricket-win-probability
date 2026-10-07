"""
Delivery-outcome (Monte Carlo engine) gate with the rating features SWAPPED: original Elo vs
the new ICC-style + bridge ratings.

Companion to scripts/measure_rating_swap_brier.py, which does the same for the win-probability
model. The delivery-outcome model consumes player ratings too (build_training_frame joins
effective_elo per match/player/role), so the same swap applies and the same gate --
extras/legal OOF log-loss, locked in data/processed/sim_baseline_metrics.json -- can be read.

Both arms are TRAINED FRESH. The production model in data/processed was trained on old ratings,
so reusing it would evaluate the new frame against a model fitted to the old features and
understate the new arm. Fresh training to scratch caches keeps it symmetric; nothing under
data/processed is written.

OOF log-loss is computed exactly the way tests/sim_baseline_metrics.py computes it -- each
match scored by the fold model that did NOT train on it.

Usage:
    py scripts/measure_rating_swap_sim_gate.py --scratch <dir>
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import gc
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.data_processing.load_data import CricketDataLoader
from src.data_processing.match_metadata import MatchMetadataExtractor
from src.data_processing.format_spec import spec, INTL_COMPETITIONS, franchise_competitions
from src.features.venue_engine import VenueEngine
from src.models.delivery_outcome_model import (
    DeliveryOutcomeModel, build_training_frame,
    LEGAL_CLASSES, LEGAL_CLASS_TO_IDX, EXTRAS_CLASSES, EXTRAS_CLASS_TO_IDX,
)
from measure_rating_swap_brier import build_full_corpus_ratings, build_franchise_elo_df


def oof_log_losses(model, frame) -> dict:
    """Mirrors tests/sim_baseline_metrics.evaluate_sim_model's OOF computation exactly."""
    frame = frame.copy()
    frame["_fold"] = frame["match_id"].map(model.match_to_fold)
    n_unmapped = int(frame["_fold"].isna().sum())
    frame = frame.dropna(subset=["_fold"]).copy()
    frame["_fold"] = frame["_fold"].astype(int)

    y_extras = frame["extra_class"].map(EXTRAS_CLASS_TO_IDX).values
    oof_extras = np.zeros((len(frame), len(EXTRAS_CLASSES)))

    legal_mask = (frame["extra_class"] == "legal").values
    legal_frame = frame[legal_mask].reset_index(drop=True)
    y_legal = legal_frame["legal_outcome"].map(LEGAL_CLASS_TO_IDX).values
    oof_legal = np.zeros((len(legal_frame), len(LEGAL_CLASSES)))

    for fold_idx, pack in model.fold_models.items():
        m = (frame["_fold"].values == fold_idx)
        if m.any():
            oof_extras[m] = pack["extras"].predict_proba(model.oof_features(frame.loc[m]).values)
        lm = (legal_frame["_fold"].values == fold_idx)
        if lm.any():
            oof_legal[lm] = pack["legal"].predict_proba(model.oof_features(legal_frame.loc[lm]).values)

    return {
        "extras_log_loss": float(log_loss(y_extras, oof_extras,
                                          labels=list(range(len(EXTRAS_CLASSES))))),
        "legal_log_loss": float(log_loss(y_legal, oof_legal,
                                         labels=list(range(len(LEGAL_CLASSES))))),
        "n_deliveries": int(len(frame)),
        "n_legal": int(len(legal_frame)),
        "n_unmapped_to_fold": n_unmapped,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--format", default="t20")
    p.add_argument("--scratch", default=os.environ.get("TEMP", "/tmp"))
    args = p.parse_args()
    fmt = args.format
    scratch = Path(args.scratch) / "rating_swap_sim"
    scratch.mkdir(parents=True, exist_ok=True)

    raw = str(PROJECT_ROOT / "data" / "raw")
    deliveries = CricketDataLoader(raw, format_type=fmt).load_all()
    meta = MatchMetadataExtractor(raw, format_type=fmt).extract_all()
    meta = meta[meta["gender"] == spec(fmt)["gender"]].copy()
    deliveries = deliveries[deliveries["match_id"].isin(meta["match_id"])].copy()
    meta["date"] = pd.to_datetime(meta["date"])
    print(f"corpus: {len(meta):,} matches, {len(deliveries):,} deliveries")

    proc = PROJECT_ROOT / "data" / "processed"
    old_elo = pd.read_parquet(proc / f"{fmt}_elo_ratings.parquet")
    old_players = pd.read_parquet(proc / f"{fmt}_player_elo_ratings.parquet")
    icc_teams = pd.read_parquet(proc / f"{fmt}_icc_team_ratings.parquet")
    icc_players = pd.read_parquet(proc / f"{fmt}_icc_player_ratings.parquet")
    bridge_a = pd.read_parquet(proc / f"{fmt}_franchise_bridge_tiera.parquet")
    bc_path = proc / f"{fmt}_franchise_bridge_tierbc.parquet"
    bridge_bc = pd.read_parquet(bc_path) if bc_path.exists() else None

    a_ids = set(meta[meta["competition"].isin(franchise_competitions(fmt, ("A",)))]["match_id"])
    bc_ids = set(meta[meta["competition"].isin(franchise_competitions(fmt, ("B", "C")))]["match_id"])
    sources = {
        "intl": (icc_players, icc_teams),
        "tierA": (bridge_a, build_franchise_elo_df(
            old_elo[old_elo["match_id"].isin(a_ids)], bridge_a, meta, fmt)),
    }
    if bridge_bc is not None and len(bridge_bc):
        sources["tierBC"] = (bridge_bc, build_franchise_elo_df(
            old_elo[old_elo["match_id"].isin(bc_ids)], bridge_bc, meta, fmt))

    print("Composing swapped ratings...")
    _, new_players = build_full_corpus_ratings(old_elo, old_players, sources, meta, fmt)

    venue_df = VenueEngine(format_type=fmt).run(meta, deliveries)   # shared by both arms

    results = {}
    for label, players in (("OLD Elo (control)", old_players), ("NEW combined", new_players)):
        print(f"\n--- {label} ---")
        frame = build_training_frame(deliveries, players, venue_df, format_type=fmt)
        print(f"  training frame: {len(frame):,} deliveries")
        cache = scratch / f"{fmt}_{'old' if 'OLD' in label else 'new'}_sim.pkl"
        if cache.exists():
            cache.unlink()
        model = DeliveryOutcomeModel(format_type=fmt)
        model.fit(frame, cache_path=str(cache))
        results[label] = oof_log_losses(model, frame)
        r = results[label]
        print(f"  extras_log_loss={r['extras_log_loss']:.6f}  "
              f"legal_log_loss={r['legal_log_loss']:.6f}  n={r['n_deliveries']:,}")
        del frame, model
        gc.collect()

    old_r, new_r = results["OLD Elo (control)"], results["NEW combined"]
    print("\n" + "=" * 72)
    baseline_path = proc / "sim_baseline_metrics.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            locked = json.load(f).get(fmt, {})
        print(f"locked baseline : extras={locked.get('extras_log_loss')}  "
              f"legal={locked.get('legal_log_loss')}")
    print(f"OLD (control)   : extras={old_r['extras_log_loss']:.6f}  "
          f"legal={old_r['legal_log_loss']:.6f}")
    print(f"NEW combined    : extras={new_r['extras_log_loss']:.6f}  "
          f"legal={new_r['legal_log_loss']:.6f}")
    de = new_r["extras_log_loss"] - old_r["extras_log_loss"]
    dl = new_r["legal_log_loss"] - old_r["legal_log_loss"]
    print(f"\n  extras delta : {de:+.6f}  ({'NEW better' if de < 0 else 'OLD better'})")
    print(f"  legal  delta : {dl:+.6f}  ({'NEW better' if dl < 0 else 'OLD better'})")
    print("\n  (gate tolerance is 0.01 on each; lower log-loss is better)")


if __name__ == "__main__":
    main()
