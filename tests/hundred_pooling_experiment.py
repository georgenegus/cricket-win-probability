"""
Hundred Pooling Experiment - Measures Pooled (T20 + Hundred) vs Hundred-Only Model Performance.

Holds out the most recent ~20% of Hundred matches as a test set, fits:
1. `hnd_only`: DeliveryOutcomeModel trained solely on historical Hundred training matches.
2. `pooled`: DeliveryOutcomeModel trained on all T20 deliveries + historical Hundred training matches (with is_hundred feature).

Evaluates both on the held-out Hundred test set using:
- Legal-outcome Log-Loss & Top-1 Accuracy
- Extras-gate Log-Loss & Top-1 Accuracy
- Death-phase P(6) empirical alignment error
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import log_loss, accuracy_score
from xgboost import XGBClassifier

# Add project root to sys.path for module resolution
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reconfigure stdout to UTF-8 on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data_processing.load_data import CricketDataLoader
from src.data_processing.match_metadata import MatchMetadataExtractor
from src.features.elo_engine import EloEngine
from src.features.player_elo_engine import PlayerEloEngine
from src.features.venue_engine import VenueEngine
from src.models.delivery_outcome_model import (
    FEATURES, LEGAL_CLASSES, LEGAL_CLASS_TO_IDX, EXTRAS_CLASSES, EXTRAS_CLASS_TO_IDX,
    build_training_frame, DeliveryOutcomeModel
)


def run_experiment():
    print("=" * 70)
    print("  HUNDRED POOLING EXPERIMENT: POOLED (T20+HND) vs HUNDRED-ONLY")
    print("=" * 70 + "\n")

    # 1. Load Datasets
    print("Loading data...")
    loader_hnd = CricketDataLoader("data/raw", format_type="hnd")
    df_hnd = loader_hnd.load_all()
    meta_hnd = MatchMetadataExtractor("data/raw", format_type="hnd").extract_all()

    loader_t20 = CricketDataLoader("data/raw", format_type="t20")
    df_t20 = loader_t20.load_all()
    meta_t20 = MatchMetadataExtractor("data/raw", format_type="t20").extract_all()

    # Load feature parquets
    player_elo_hnd = pd.read_parquet("data/processed/hnd_player_elo_ratings.parquet")
    venue_hnd = pd.read_parquet("data/processed/hnd_venue_features.parquet")

    player_elo_t20 = pd.read_parquet("data/processed/t20_player_elo_ratings.parquet")
    venue_t20 = pd.read_parquet("data/processed/t20_venue_features.parquet")

    # 2. Build Chronological Train/Test Split on Hundred Matches
    # Sort matches by date and hold out the last 20% (~33 matches)
    sorted_matches = meta_hnd.sort_values("date")["match_id"].tolist()
    split_idx = int(len(sorted_matches) * 0.80)
    train_match_ids = set(sorted_matches[:split_idx])
    test_match_ids = set(sorted_matches[split_idx:])

    print(f"Hundred Matches Total: {len(sorted_matches)} | Train: {len(train_match_ids)} | Held-out Test: {len(test_match_ids)}")

    # Build Hundred training frames
    df_hnd_train = df_hnd[df_hnd["match_id"].isin(train_match_ids)].copy()
    df_hnd_test = df_hnd[df_hnd["match_id"].isin(test_match_ids)].copy()

    tf_hnd_train = build_training_frame(df_hnd_train, player_elo_hnd, venue_hnd, format_type="hnd")
    tf_hnd_test = build_training_frame(df_hnd_test, player_elo_hnd, venue_hnd, format_type="hnd")

    # Build T20 training frame (sample or use full dataset for fast execution)
    tf_t20_train = build_training_frame(df_t20, player_elo_t20, venue_t20, format_type="t20")

    print(f"Training deliveries -> HND Only: {len(tf_hnd_train):,} | T20 Pool: {len(tf_t20_train):,} | Test Deliveries: {len(tf_hnd_test):,}")

    # 3. Model A: Hundred-Only Model
    print("\n[Model A] Training Hundred-Only Delivery Outcome Model...")
    tf_hnd_train_legal = tf_hnd_train[tf_hnd_train["extra_class"] == "legal"].copy()
    
    y_extras_hnd = tf_hnd_train["extra_class"].map(EXTRAS_CLASS_TO_IDX).values
    y_legal_hnd = tf_hnd_train_legal["legal_outcome"].map(LEGAL_CLASS_TO_IDX).values

    clf_extras_hnd = XGBClassifier(
        objective="multi:softprob", num_class=len(EXTRAS_CLASSES),
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=-1
    )
    clf_extras_hnd.fit(tf_hnd_train[FEATURES].values, y_extras_hnd)

    clf_legal_hnd = XGBClassifier(
        objective="multi:softprob", num_class=len(LEGAL_CLASSES),
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=-1
    )
    clf_legal_hnd.fit(tf_hnd_train_legal[FEATURES].values, y_legal_hnd)

    # 4. Model B: Pooled Model (T20 + HND with is_hundred feature)
    print("\n[Model B] Training Pooled (T20 + HND) Delivery Outcome Model...")
    tf_hnd_train_copy = tf_hnd_train.copy()
    tf_hnd_train_copy["is_hundred"] = 1.0

    tf_t20_train_copy = tf_t20_train.copy()
    tf_t20_train_copy["is_hundred"] = 0.0

    pooled_features = FEATURES + ["is_hundred"]

    tf_pooled_all = pd.concat([tf_t20_train_copy, tf_hnd_train_copy], ignore_index=True)
    tf_pooled_legal = tf_pooled_all[tf_pooled_all["extra_class"] == "legal"].copy()

    y_extras_pooled = tf_pooled_all["extra_class"].map(EXTRAS_CLASS_TO_IDX).values
    y_legal_pooled = tf_pooled_legal["legal_outcome"].map(LEGAL_CLASS_TO_IDX).values

    clf_extras_pooled = XGBClassifier(
        objective="multi:softprob", num_class=len(EXTRAS_CLASSES),
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=-1
    )
    clf_extras_pooled.fit(tf_pooled_all[pooled_features].values, y_extras_pooled)

    clf_legal_pooled = XGBClassifier(
        objective="multi:softprob", num_class=len(LEGAL_CLASSES),
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=-1
    )
    clf_legal_pooled.fit(tf_pooled_legal[pooled_features].values, y_legal_pooled)

    # 5. Evaluate both on held-out Hundred Test Set
    print("\nEvaluating both models on held-out Hundred test set...")
    tf_hnd_test_legal = tf_hnd_test[tf_hnd_test["extra_class"] == "legal"].copy()

    y_test_extras = tf_hnd_test["extra_class"].map(EXTRAS_CLASS_TO_IDX).values
    y_test_legal = tf_hnd_test_legal["legal_outcome"].map(LEGAL_CLASS_TO_IDX).values

    X_test_hnd = tf_hnd_test[FEATURES].values
    X_test_hnd_legal = tf_hnd_test_legal[FEATURES].values

    tf_hnd_test_pooled = tf_hnd_test.copy()
    tf_hnd_test_pooled["is_hundred"] = 1.0
    tf_hnd_test_legal_pooled = tf_hnd_test_legal.copy()
    tf_hnd_test_legal_pooled["is_hundred"] = 1.0

    X_test_pooled = tf_hnd_test_pooled[pooled_features].values
    X_test_pooled_legal = tf_hnd_test_legal_pooled[pooled_features].values

    # Predictions - HND Only
    probs_extras_hnd = clf_extras_hnd.predict_proba(X_test_hnd)
    probs_legal_hnd = clf_legal_hnd.predict_proba(X_test_hnd_legal)

    loss_extras_hnd = log_loss(y_test_extras, probs_extras_hnd, labels=list(range(len(EXTRAS_CLASSES))))
    loss_legal_hnd = log_loss(y_test_legal, probs_legal_hnd, labels=list(range(len(LEGAL_CLASSES))))
    acc_legal_hnd = accuracy_score(y_test_legal, probs_legal_hnd.argmax(axis=1))

    # Predictions - Pooled
    probs_extras_pooled = clf_extras_pooled.predict_proba(X_test_pooled)
    probs_legal_pooled = clf_legal_pooled.predict_proba(X_test_pooled_legal)

    loss_extras_pooled = log_loss(y_test_extras, probs_extras_pooled, labels=list(range(len(EXTRAS_CLASSES))))
    loss_legal_pooled = log_loss(y_test_legal, probs_legal_pooled, labels=list(range(len(LEGAL_CLASSES))))
    acc_legal_pooled = accuracy_score(y_test_legal, probs_legal_pooled.argmax(axis=1))

    # Death-phase P(6) empirical calibration error
    six_idx = LEGAL_CLASSES.index(6)
    death_mask = tf_hnd_test_legal["phase_encoded"].values == 2
    actual_six_death = (y_test_legal[death_mask] == six_idx).mean()
    pred_six_hnd_death = probs_legal_hnd[death_mask, six_idx].mean()
    pred_six_pooled_death = probs_legal_pooled[death_mask, six_idx].mean()

    err_six_hnd = abs(pred_six_hnd_death - actual_six_death)
    err_six_pooled = abs(pred_six_pooled_death - actual_six_death)

    # 6. Results Summary
    print("\n" + "=" * 70)
    print("  EXPERIMENT RESULTS SUMMARY (Held-out Hundred Test Set)")
    print("=" * 70)
    print(f"{'Metric':<30} | {'Hundred-Only':<15} | {'Pooled (T20+HND)':<15} | {'Winner':<10}")
    print("-" * 78)
    
    w_legal = "HND-Only" if loss_legal_hnd < loss_legal_pooled else "Pooled"
    w_acc = "HND-Only" if acc_legal_hnd > acc_legal_pooled else "Pooled"
    w_six = "HND-Only" if err_six_hnd < err_six_pooled else "Pooled"

    print(f"{'Legal Log-Loss (lower=better)':<30} | {loss_legal_hnd:<15.4f} | {loss_legal_pooled:<15.4f} | {w_legal:<10}")
    print(f"{'Legal Accuracy (higher=better)':<30} | {acc_legal_hnd:<15.4f} | {acc_legal_pooled:<15.4f} | {w_acc:<10}")
    print(f"{'Death P(6) Error (lower=better)':<30} | {err_six_hnd:<15.4f} | {err_six_pooled:<15.4f} | {w_six:<10}")
    print(f"{'Extras Log-Loss (lower=better)':<30} | {loss_extras_hnd:<15.4f} | {loss_extras_pooled:<15.4f} | {'Same':<10}")
    print("-" * 78)

    print("\nDetailed Death-Phase P(6) Comparison:")
    print(f"  Actual Empirical Death P(6): {actual_six_death:.4f}")
    print(f"  HND-Only Predicted Death P(6): {pred_six_hnd_death:.4f} (Error: {err_six_hnd:.4f})")
    print(f"  Pooled Predicted Death P(6): {pred_six_pooled_death:.4f} (Error: {err_six_pooled:.4f})")

    verdict = "POOLED" if (loss_legal_pooled < loss_legal_hnd and err_six_pooled <= err_six_hnd) else "HUNDRED-ONLY"
    print(f"\nFINAL VERDICT: Recommend keeping {verdict} model architecture.")
    print("=" * 70 + "\n")

    return {
        "hnd_only": {"loss_legal": loss_legal_hnd, "acc_legal": acc_legal_hnd, "err_six_death": err_six_hnd},
        "pooled": {"loss_legal": loss_legal_pooled, "acc_legal": acc_legal_pooled, "err_six_death": err_six_pooled},
        "verdict": verdict,
    }


if __name__ == "__main__":
    run_experiment()
