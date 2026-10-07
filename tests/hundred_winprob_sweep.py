"""
Hundred Win-Probability Hyperparameter Sweep (Phase 4).

GroupKFold-by-match_id hyperparameter search over a thin-data-appropriate grid, ranked by
MEAN OUT-OF-FOLD BRIER (the metric this whole plan targets -- not log-loss). Uses raw
(uncalibrated) OOF probabilities for ranking speed; the winning config is then run through
the full fit() + isotonic-calibration pipeline via retrain_full('hnd') for the real,
calibrated verification number (see hundred_validation.py).

Grid note: the plan's literal spec (max_depth x min_child_weight x 3 paired lr/n_estimators
x reg_alpha x reg_lambda x subsample x colsample_bytree) is 648 configs x 5 folds = 3,240
fits -- too slow for a single background run. Kept full resolution on the three knobs most
directly tied to the stated over-fit problem (max_depth, min_child_weight, lr/n_estimators
pairing); the four secondary regularization knobs are bundled into 3 light/medium/heavy
profiles instead of a full 24-way cross product. 3x3x3x3 = 81 configs x 5 folds = 405 fits.

Usage:
    py tests/hundred_winprob_sweep.py
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import csv
import time
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold

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

RESULTS_PATH = Path("data/processed/hnd_winprob_sweep_results.csv")
TIE_EPSILON = 0.001  # near-ties within this margin of the best OOF Brier are broken by simplicity

# Primary grid -- directly targets the "over-fit on thin data" problem
MAX_DEPTH_GRID = [2, 3, 4]
MIN_CHILD_WEIGHT_GRID = [10, 20, 40]
LR_NEST_PAIRS = [(0.08, 300), (0.05, 500), (0.03, 800)]

# Secondary regularization profiles (reg_alpha, reg_lambda, subsample, colsample_bytree) --
# bundled instead of a full 24-way cross product, see module docstring.
REG_PROFILES = [
    (1, 3, 0.8, 0.8),   # light
    (1, 5, 0.8, 0.7),   # medium
    (3, 10, 0.7, 0.7),  # heavy
]


def build_hnd_second_innings_states() -> pd.DataFrame:
    """Mirrors cricsheet_importer.retrain_full('hnd')'s feature pipeline exactly (cache-aware)."""
    format_type = "hnd"
    print("Loading hnd deliveries & metadata...")
    loader = CricketDataLoader("data/raw", format_type=format_type)
    deliveries = loader.load_all()

    meta_extractor = MatchMetadataExtractor("data/raw", format_type=format_type)
    meta_df = meta_extractor.extract_all()
    if "gender" in meta_df.columns:
        meta_df = meta_df[meta_df["gender"] == "male"].copy()
        deliveries = deliveries[deliveries["match_id"].isin(meta_df["match_id"])].copy()

    print("Computing Elo / player Elo / phase Elo / venue features...")
    elo_engine = EloEngine(use_dynamic_k=True, use_margin=True, use_decay=True, use_home_away=False, format_type=format_type)
    elo_df = elo_engine.run(meta_df)

    t20_pe_path = Path("data/processed/t20_player_elo_ratings.parquet")
    seed_ratings = None
    if t20_pe_path.exists():
        t20_pe = pd.read_parquet(t20_pe_path)
        latest_t20 = t20_pe.sort_values("match_id").groupby(["player", "role"]).last().reset_index()
        b_seeds = latest_t20[latest_t20["role"] == "batter"].set_index("player")["effective_elo"].to_dict()
        w_seeds = latest_t20[latest_t20["role"] == "bowler"].set_index("player")["effective_elo"].to_dict()
        seed_ratings = {"batter": b_seeds, "bowler": w_seeds}
    player_elo_engine = PlayerEloEngine(format_type=format_type, form_window=15)
    player_elo_df = player_elo_engine.run(deliveries, elo_df, seed_ratings=seed_ratings)

    phase_elo_engine = PhaseEloEngine(format_type=format_type)
    phase_elo_df = phase_elo_engine.run(deliveries, elo_df)

    venue_engine = VenueEngine(format_type=format_type)
    venue_df = venue_engine.run(meta_df, deliveries)
    elo_df = elo_df.merge(venue_df, on="match_id", how="left")

    print("Building states...")
    fmt_spec = spec(format_type)
    states = StateBuilder(
        deliveries, elo_df, player_elo_df=player_elo_df, phase_elo_df=phase_elo_df,
        total_overs=fmt_spec["total_overs"], format_type=format_type,
        balls_per_over=fmt_spec["balls_per_over"],
    ).build()

    second_innings = states[states["innings"] == 2].copy()
    print(f"{len(states):,} total states, {len(second_innings):,} second-innings states across {second_innings['match_id'].nunique()} matches.")
    return second_innings


def run_sweep():
    states = build_hnd_second_innings_states()

    # Reuse the model class's own feature extractor + Phase 1-3 fixes (player-Elo features,
    # format_spec-correct decay geometry, monotone constraints) -- the sweep tunes hyperparameters
    # ON TOP OF the already-fixed feature set, per the plan's stated dependency.
    helper = XGBWinProbabilityModel(format_type="hnd")
    print("Extracting features...")
    X, y = helper._extract_features_and_target(states)
    monotone_constraints = helper._build_monotone_constraints(X.columns)
    print(f"{X.shape[1]} features, {len(X):,} rows.")

    match_ids = states["match_id"].values
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X, y, groups=match_ids))  # computed ONCE, reused for every grid point

    combos = list(itertools.product(MAX_DEPTH_GRID, MIN_CHILD_WEIGHT_GRID, LR_NEST_PAIRS, REG_PROFILES))
    print(f"\nSweeping {len(combos)} configurations x 5 folds = {len(combos) * 5} fits...\n")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["max_depth", "min_child_weight", "learning_rate", "n_estimators",
                  "reg_alpha", "reg_lambda", "subsample", "colsample_bytree", "mean_oof_brier"]

    results = []
    t_start = time.time()
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, (max_depth, min_child_weight, (lr, n_estimators), (reg_alpha, reg_lambda, subsample, colsample_bytree)) in enumerate(combos):
            fold_briers = []
            for train_idx, val_idx in splits:
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

                model = XGBClassifier(
                    n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
                    min_child_weight=min_child_weight, subsample=subsample,
                    colsample_bytree=colsample_bytree, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                    objective="binary:logistic", eval_metric="logloss", random_state=42,
                    n_jobs=-1, enable_categorical=True, monotone_constraints=monotone_constraints,
                )
                model.fit(X_train, y_train)
                probs = model.predict_proba(X_val)[:, 1]
                fold_briers.append(float(np.mean((probs - y_val.values) ** 2)))

            row = {
                "max_depth": max_depth, "min_child_weight": min_child_weight,
                "learning_rate": lr, "n_estimators": n_estimators,
                "reg_alpha": reg_alpha, "reg_lambda": reg_lambda,
                "subsample": subsample, "colsample_bytree": colsample_bytree,
                "mean_oof_brier": round(float(np.mean(fold_briers)), 6),
            }
            results.append(row)
            writer.writerow(row)
            f.flush()  # incremental write -- survives an interrupted run

            if (i + 1) % 10 == 0 or (i + 1) == len(combos):
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (len(combos) - (i + 1)) / rate if rate > 0 else 0
                best_so_far = min(r["mean_oof_brier"] for r in results)
                print(f"  [{i+1}/{len(combos)}] elapsed={elapsed:.0f}s eta={eta:.0f}s best_so_far={best_so_far:.6f}")

    df = pd.DataFrame(results)

    # Tie-break: within TIE_EPSILON of the best OOF Brier, prefer fewer effective parameters
    # (shallower depth, fewer trees, higher min_child_weight/reg_alpha/reg_lambda = simpler model)
    df["complexity_score"] = (
        df["max_depth"] * 100
        + df["n_estimators"] / 10.0
        - df["reg_alpha"] * 5
        - df["reg_lambda"] * 2
        - df["min_child_weight"]
    )
    best_brier = df["mean_oof_brier"].min()
    near_best = df[df["mean_oof_brier"] <= best_brier + TIE_EPSILON].sort_values("complexity_score")
    winner = near_best.iloc[0].to_dict()

    df_ranked = df.sort_values("mean_oof_brier").reset_index(drop=True)
    print("\n" + "=" * 100)
    print(f"TOP 15 CONFIGS BY MEAN OOF BRIER  (best={best_brier:.6f}, {len(near_best)} within {TIE_EPSILON} tie-band)")
    print("=" * 100)
    print(df_ranked.head(15).drop(columns=["complexity_score"], errors="ignore").to_string(index=False))

    print("\nWINNING CONFIG (best OOF Brier; ties within tie-band broken toward simpler model):")
    for k in fieldnames:
        print(f"  {k}: {winner[k]}")

    print(f"\nFull results written to {RESULTS_PATH}")
    return winner


if __name__ == "__main__":
    run_sweep()
