"""
Delivery-Outcome (simulator engine) OOF baseline lock.

Originally a Phase 0 fixture for t20_player_elo_revamp_plan.md (t20 only; shipped
2026-08-03). Extended for od_player_elo_revamp_plan.md Phase 0 to also cover od -- the
baseline file now holds one entry per format that has been locked (currently t20 and od);
--generate only (re)locks the format(s) passed via --formats, leaving other formats' entries
untouched.

DeliveryOutcomeModel is the OTHER consumer of player Elo (batter_elo, bowler_elo,
matchup_elo_diff, non_striker_elo, recent_form_batter/bowler -- see FEATURES in
delivery_outcome_model.py) besides the win-prob classifier. Each player-Elo revamp plan
retrains this model too (its own Phase 6), so it needs a genuine "before" number alongside
the win-prob Brier baseline (tests/regression_test.py) -- otherwise a sim-engine regression
could ship undetected.

Scored with true GroupKFold-by-match out-of-fold predictions: each delivery is scored by
the fold model that did NOT see its match during training (model.match_to_fold +
model.fold_models), never the final model -- so this is a leakage-free accuracy number,
not a train-set score.

Usage:
    py tests/sim_baseline_metrics.py --generate                 # lock/refresh t20 (default)
    py tests/sim_baseline_metrics.py --generate --formats od    # lock/refresh od only
    py tests/sim_baseline_metrics.py --formats od               # re-check od against its lock
    py tests/sim_baseline_metrics.py                             # re-check t20 (default) against its lock
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import log_loss

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
from src.data_processing.format_spec import spec
from src.features.elo_engine import EloEngine
from src.features.player_elo_engine import PlayerEloEngine
from src.features.venue_engine import VenueEngine
from src.features import rating_source
from src.models.delivery_outcome_model import (
    DeliveryOutcomeModel, build_training_frame,
    LEGAL_CLASSES, LEGAL_CLASS_TO_IDX, EXTRAS_CLASSES, EXTRAS_CLASS_TO_IDX,
)

BASELINE_PATH = PROJECT_ROOT / "data" / "processed" / "sim_baseline_metrics.json"
DEFAULT_TOLERANCE = {"extras_log_loss_max_increase": 0.01, "legal_log_loss_max_increase": 0.01}


def build_frame(format_type: str, source: str = rating_source.DEFAULT_RATING_SOURCE) -> pd.DataFrame:
    """Mirrors retrain_full's delivery-outcome training-frame build exactly (cache-aware).

    source: additive (rating_source_toggle_plan.md Phase 5), default "elo" -- unchanged for
    every existing caller. A non-"elo" source reads the pre-built combined-rating artifact
    (scripts/build_combined_ratings.py) instead of recomputing Elo; same rationale as
    regression_test.py::evaluate_format_models's identical branch."""
    loader = CricketDataLoader(str(PROJECT_ROOT / "data" / "raw"), format_type=format_type)
    deliveries = loader.load_all()

    meta_extractor = MatchMetadataExtractor(str(PROJECT_ROOT / "data" / "raw"), format_type=format_type)
    meta_df = meta_extractor.extract_all()
    if "gender" in meta_df.columns:
        # Filter to this format's own gender -- was hardcoded to "male", which would zero
        # out an all-female corpus like odw/hndw/t20w. See cricsheet_importer.retrain_full's
        # identical fix.
        meta_df = meta_df[meta_df["gender"] == spec(format_type)["gender"]].copy()
        deliveries = deliveries[deliveries["match_id"].isin(meta_df["match_id"])].copy()

    if source == rating_source.DEFAULT_RATING_SOURCE:
        elo_engine = EloEngine(use_dynamic_k=True, use_margin=True, use_decay=True, use_home_away=False, format_type=format_type)
        elo_df = elo_engine.run(meta_df)
        player_elo_engine = PlayerEloEngine(format_type=format_type, form_window=15)
        player_elo_df = player_elo_engine.run(deliveries, elo_df)
    else:
        player_elo_df = rating_source.load_player_ratings(format_type, source)

    venue_engine = VenueEngine(format_type=format_type)
    venue_df = venue_engine.run(meta_df, deliveries)

    return build_training_frame(deliveries, player_elo_df, venue_df, format_type=format_type)


def evaluate_sim_model(format_type: str, source: str = rating_source.DEFAULT_RATING_SOURCE) -> dict:
    """Genuine OOF log-loss for both stages, using each match's HELD-OUT fold model."""
    suffix = "" if source == rating_source.DEFAULT_RATING_SOURCE else f"_{source}"
    cache_path = PROJECT_ROOT / "data" / "processed" / f"{format_type}{suffix}_delivery_outcome_model.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(f"No trained sim model at {cache_path} -- run retrain_full('{format_type}') "
                                f"(or scripts/train_rating_source_models.py for a non-elo source) first.")

    model = DeliveryOutcomeModel(format_type=format_type)
    model.fit(None, cache_path=cache_path)  # cache exists -> loads fold_models/match_to_fold, no training

    frame = build_frame(format_type, source)
    frame["_fold"] = frame["match_id"].map(model.match_to_fold)
    n_unmapped = int(frame["_fold"].isna().sum())
    frame = frame.dropna(subset=["_fold"]).copy()
    frame["_fold"] = frame["_fold"].astype(int)

    y_extras_idx = frame["extra_class"].map(EXTRAS_CLASS_TO_IDX).values
    oof_extras = np.zeros((len(frame), len(EXTRAS_CLASSES)))

    legal_mask = (frame["extra_class"] == "legal").values
    y_legal_idx = frame.loc[legal_mask, "legal_outcome"].map(LEGAL_CLASS_TO_IDX).values
    oof_legal = np.zeros((legal_mask.sum(), len(LEGAL_CLASSES)))
    legal_row_order = []  # track which original frame-row each oof_legal row belongs to, for fold-safe fill

    # Fill OOF predictions fold-by-fold, always with the model that did NOT train on that fold.
    legal_frame = frame[legal_mask].reset_index(drop=True)
    for fold_idx, fold_pack in model.fold_models.items():
        m = (frame["_fold"].values == fold_idx)
        if m.any():
            oof_extras[m] = fold_pack["extras"].predict_proba(model.oof_features(frame.loc[m]).values)

        lm = (legal_frame["_fold"].values == fold_idx)
        if lm.any():
            oof_legal[lm] = fold_pack["legal"].predict_proba(model.oof_features(legal_frame.loc[lm]).values)

    extras_ll = float(log_loss(y_extras_idx, oof_extras, labels=list(range(len(EXTRAS_CLASSES)))))
    legal_ll = float(log_loss(y_legal_idx, oof_legal, labels=list(range(len(LEGAL_CLASSES)))))

    return {
        "extras_log_loss": round(extras_ll, 6),
        "legal_log_loss": round(legal_ll, 6),
        "n_deliveries": int(len(frame)),
        "n_legal": int(legal_mask.sum()),
        "n_unmapped_to_fold": n_unmapped,
        "n_folds": len(model.fold_models),
    }


def _load_existing_baseline() -> dict:
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH) as f:
            return json.load(f)
    return {}


def generate_baseline(formats=("t20",)):
    """(Re)locks the given format(s)' entries in-place; other formats' entries are preserved."""
    result = _load_existing_baseline()
    for format_type in formats:
        print("=" * 60)
        print(f"GENERATING SIM (DELIVERY-OUTCOME) OOF BASELINE -- {format_type.upper()}")
        print("=" * 60)
        r = evaluate_sim_model(format_type)
        result[format_type] = r
        print(f"  Extras-gate OOF log-loss : {r['extras_log_loss']:.6f}  (n={r['n_deliveries']:,})")
        print(f"  Legal-outcome OOF log-loss: {r['legal_log_loss']:.6f}  (n={r['n_legal']:,})")
        if r["n_unmapped_to_fold"]:
            print(f"  WARNING: {r['n_unmapped_to_fold']} deliveries had no fold mapping (excluded).")

    result["tolerance"] = DEFAULT_TOLERANCE
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote baseline to {BASELINE_PATH} (formats: {', '.join(result.keys() - {'tolerance'})})")
    return result


def check_against_baseline(formats=("t20",)):
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"No baseline at {BASELINE_PATH} -- run --generate first.")
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    tol = baseline.get("tolerance", DEFAULT_TOLERANCE)

    ok = True
    for format_type in formats:
        if format_type not in baseline:
            raise KeyError(f"No baseline entry for '{format_type}' -- run --generate --formats {format_type} first.")
        print(f"--- {format_type.upper()} ---")
        actual = evaluate_sim_model(format_type)
        expected = baseline[format_type]

        for key, tol_key in [("extras_log_loss", "extras_log_loss_max_increase"),
                             ("legal_log_loss", "legal_log_loss_max_increase")]:
            max_allowed = expected[key] + tol[tol_key]
            passed = actual[key] <= max_allowed
            ok = ok and passed
            status = "PASS" if passed else "FAIL [REGRESSION]"
            print(f"  {key}: {actual[key]:.6f} (Baseline: {expected[key]:.6f} | "
                  f"Diff: {actual[key]-expected[key]:+.6f}) [{status}]")

    print("\n" + ("SIM BASELINE CHECK PASSED" if ok else "SIM BASELINE CHECK FAILED"))
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--formats", nargs="+", default=["t20"], choices=["t20", "od", "hnd", "t20w", "odw", "hndw"])
    args = parser.parse_args()

    if args.generate or not BASELINE_PATH.exists():
        generate_baseline(formats=args.formats)
    else:
        passed = check_against_baseline(formats=args.formats)
        sys.exit(0 if passed else 1)
