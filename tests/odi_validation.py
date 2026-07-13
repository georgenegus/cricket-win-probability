from sklearn import model_selection
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import time
import sys
import gc
from pathlib import Path
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.load_data import CricketDataLoader
from src.data_processing.build_states import StateBuilder
from src.data_processing.match_metadata import MatchMetadataExtractor
from src.features.elo_engine import EloEngine
from src.features.player_elo_engine import PlayerEloEngine
from src.features.venue_engine import VenueEngine
from src.models.xgb_win_probability import XGBWinProbabilityModel
from src.models.first_innings_predictor import FirstInningsPredictor


def main():
    print("==================================================")
    print("ONE-DAY FORMAT MODEL PIPELINE & VALIDATION")
    print("==================================================")

    start_time = time.time()

    # 1. Load Data for OD format
    print("\nLoading raw One-Day deliveries...")
    t_load = time.time()
    loader = CricketDataLoader("data/raw", format_type="od")
    deliveries = loader.load_all()
    print(f"Loaded {len(deliveries):,} deliveries in {time.time() - t_load:.2f}s")

    # 2. Extract metadata
    print("\nExtracting One-Day match metadata...")
    t_meta = time.time()
    meta_extractor = MatchMetadataExtractor("data/raw", format_type="od")
    meta_df = meta_extractor.extract_all()
    
    # Filter for male matches only
    if "gender" in meta_df.columns:
        meta_df = meta_df[meta_df["gender"] == "male"].copy()
        print(f"Filtered for male matches only. Remaining matches: {len(meta_df)}")
        deliveries = deliveries[deliveries["match_id"].isin(meta_df["match_id"])].copy()
        print(f"Filtered deliveries. Remaining rows: {len(deliveries)}")
        
    print(f"Extracted {len(meta_df):,} matches in {time.time() - t_meta:.2f}s")

    # 3. Compute Elo for OD
    print("\nComputing rolling Elo ratings for One-Day teams...")
    t_elo = time.time()
    # One-Day formats play for longer and might need dynamic decays, let's keep it consistent
    elo_engine = EloEngine(use_dynamic_k=True, use_margin=True, use_decay=True, use_home_away=False, format_type="od")
    elo_df = elo_engine.run(meta_df)
    print(f"Elo computation completed in {time.time() - t_elo:.2f}s")

    # 4. Compute Venue features
    print("\nComputing rolling venue features...")
    t_venue = time.time()
    venue_engine = VenueEngine(format_type="od")
    venue_df = venue_engine.run(meta_df, deliveries)
    elo_df = elo_df.merge(venue_df, on="match_id", how="left")
    print(f"Venue computation completed in {time.time() - t_venue:.2f}s")

    # 4.5 Compute Player Elo for OD
    print("\nComputing rolling player Elo ratings...")
    t_player = time.time()
    player_elo_engine = PlayerEloEngine(format_type="od", form_window=5)
    # Clear cache to force recalculation with squad features
    player_elo_engine.clear_cache()
    player_elo_df = player_elo_engine.run(deliveries, elo_df)
    print(f"Player Elo computation completed in {time.time() - t_player:.2f}s")

    # 4.6 Compute Phase Elo for OD
    print("\nComputing rolling phase Elo ratings...")
    t_phase = time.time()
    from src.features.phase_elo_engine import PhaseEloEngine
    phase_elo_engine = PhaseEloEngine(format_type="od")
    phase_elo_engine.clear_cache()
    phase_elo_df = phase_elo_engine.run(deliveries, elo_df)
    print(f"Phase Elo computation completed in {time.time() - t_phase:.2f}s")

    # 5. Build states (total_overs=50 for One-Day format)
    print("\nBuilding match states (50 overs format)...")
    t_states = time.time()
    states = StateBuilder(deliveries, elo_df, player_elo_df=player_elo_df, phase_elo_df=phase_elo_df, total_overs=50).build()
    print(f"Built {len(states):,} match states in {time.time() - t_states:.2f}s")

    # Free memory
    del deliveries
    del elo_df
    del venue_df
    del meta_df
    del player_elo_df
    del phase_elo_df
    gc.collect()

    # ==================================================
    # EVALUATING MODEL 1: FIRST INNINGS PREDICTOR (REGRESSION)
    # ==================================================
    print("\n" + "="*50)
    print("1. EVALUATING FIRST INNINGS PREDICTOR (XGBRegressor)")
    print("="*50)

    # Train First Innings Regressor
    t_fit_reg = time.time()
    reg_model = FirstInningsPredictor(format_type="od")
    reg_model.fit(states, cache_path="data/processed/od_first_innings_regressor.pkl")
    print(f"Regressor trained in {time.time() - t_fit_reg:.2f}s")

    # Evaluate
    first_innings_states = states[states["innings"] == 1].copy()
    sample_reg_size = min(5000, len(first_innings_states))
    print(f"Evaluating regressor on {sample_reg_size:,} random first-innings states...")
    sample_reg = first_innings_states.sample(sample_reg_size, random_state=42).copy()

    actuals_reg = sample_reg["final_first_innings_score"].values
    preds_reg = []
    for _, row in sample_reg.iterrows():
        preds_reg.append(reg_model.predict(row)["predicted_score"])

    sample_reg["predicted_score"] = preds_reg
    mae = mean_absolute_error(actuals_reg, preds_reg)
    rmse = root_mean_squared_error(actuals_reg, preds_reg)

    print(f"\n  Overall MAE : {mae:.2f} runs")
    print(f"  Overall RMSE: {rmse:.2f} runs")
    print("\n  Per-Phase Metrics:")
    for phase, group in sample_reg.groupby("phase"):
        p_act = group["final_first_innings_score"].values
        p_pred = group["predicted_score"].values
        p_mae = mean_absolute_error(p_act, p_pred)
        p_rmse = root_mean_squared_error(p_act, p_pred)
        print(f"    {phase:<12} (count: {len(group):<4}) | MAE: {p_mae:6.2f} | RMSE: {p_rmse:6.2f}")

    # ==================================================
    # EVALUATING MODEL 2: WIN PROBABILITY (CLASSIFICATION)
    # ==================================================
    print("\n" + "="*50)
    print("2. EVALUATING WIN PROBABILITY MODEL (XGBClassifier)")
    print("="*50)

    # Filter strictly for second innings states
    second_innings_states = states[states["innings"] == 2].copy()
    print(f"Loaded {len(second_innings_states):,} second innings states.")

    # Run data leakage detection checks
    check_data_leakage(second_innings_states, format_type="od")

    # Train Classifier
    t_fit_clf = time.time()
    clf_model = XGBWinProbabilityModel(format_type="od")
    clf_model.fit(second_innings_states, cache_path="data/processed/od_xgb_models.pkl")
    print(f"Classifier trained in {time.time() - t_fit_clf:.2f}s")

    # Evaluate
    sample_clf_size = len(second_innings_states)
    print(f"Evaluating classifier on {sample_clf_size:,} random second-innings states...")
    sample_clf = second_innings_states.sample(sample_clf_size, random_state=42).copy()

    actuals_clf = sample_clf["won_match"].values
    
    # Batch prediction (vectorised)
    preds_clf = clf_model.predict_batch(sample_clf)

    sample_clf["pred_prob"] = preds_clf
    sample_clf["error"] = (sample_clf["pred_prob"] - sample_clf["won_match"]) ** 2
    brier = sample_clf["error"].mean()

    print(f"\n  Overall Brier Score: {brier:.4f}")
    print("\n  Per-Phase Brier Score:")
    for phase, group in sample_clf.groupby("phase"):
        p_brier = group["error"].mean()
        print(f"    {phase:<12} (count: {len(group):<4}) | Brier Score: {p_brier:.4f}")

    print("\n" + "="*50)
    print(f"TOTAL PIPELINE COMPLETED IN {time.time() - start_time:.2f} SECONDS")
    print("="*50)


def check_data_leakage(states: pd.DataFrame, format_type="od"):
    """
    Performs data leakage checks on states and splits:
    1. Group split leakage verification.
    2. Suspect early-stage feature correlations to identify target leakage.
    3. Look-ahead temporal features verification.
    """
    print("\n==================================================")
    print("RUNNING DATA LEAKAGE DETECTION CHECKS")
    print("==================================================")
    
    # 1. Split Leakage: Check group split overlap
    from sklearn.model_selection import GroupKFold
    model_helper = XGBWinProbabilityModel(format_type=format_type)
    X, y = model_helper._extract_features_and_target(states)
    groups = states["match_id"].values
    
    gkf = GroupKFold(n_splits=5)
    split_leak_detected = False
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        train_matches = set(groups[train_idx])
        val_matches = set(groups[val_idx])
        overlap = train_matches.intersection(val_matches)
        if overlap:
            print(f"[ERROR] Fold {fold}: Split Leakage DETECTED! Overlapping matches: {len(overlap)}")
            split_leak_detected = True
        else:
            print(f"[OK] Fold {fold}: No match_id leakage across splits.")
            
    if not split_leak_detected:
        print("[OK] Split Leakage: All GroupKFold splits are perfectly isolated by match_id.")

    # 2. Target Leakage: suspicious early correlations (first 5 overs for ODI since it's 50 overs)
    early_mask = states["over"] < 5
    early_states = states[early_mask]
    if len(early_states) > 0:
        X_early = X[early_mask]
        y_early = y[early_mask]
        
        suspicious_corrs = []
        for col in X_early.columns:
            # Skip non-numeric features for correlation check
            from pandas.api.types import is_numeric_dtype
            if not is_numeric_dtype(X_early[col]):
                continue
            if X_early[col].nunique() <= 2:
                continue
            corr = float(X_early[col].corr(y_early))
            if abs(corr) > 0.6:
                suspicious_corrs.append((col, corr))
                
        if suspicious_corrs:
            print("\n[WARNING] Suspiciously high feature correlations with target in early overs (over < 5):")
            for col, corr in suspicious_corrs:
                print(f"  - {col}: Pearson Correlation = {corr:.4f}")
        else:
            print("[OK] Target Leakage: No feature exhibits suspiciously high correlation in early overs (all |r| < 0.6).")

    # 3. Look-Ahead / Temporal Leakage: Pre-match features stability
    sample_matches = states["match_id"].sample(min(5, states["match_id"].nunique()), random_state=42).unique()
    temporal_leak_detected = False
    for mid in sample_matches:
        m_states = states[states["match_id"] == mid]
        if m_states["venue_avg_first_innings_score"].nunique() > 1:
            print(f"[ERROR] Temporal Leakage: 'venue_avg_first_innings_score' changes mid-match for match {mid}!")
            temporal_leak_detected = True
        if m_states["venue_chase_success_rate"].nunique() > 1:
            print(f"[ERROR] Temporal Leakage: 'venue_chase_success_rate' changes mid-match for match {mid}!")
            temporal_leak_detected = True
        if "team_1_elo" in m_states.columns and m_states["team_1_elo"].nunique() > 1:
            print(f"[ERROR] Temporal Leakage: 'team_1_elo' changes mid-match for match {mid}!")
            temporal_leak_detected = True
                
    if not temporal_leak_detected:
        print("[OK] Temporal Leakage: Venue stats and team Elos are constant pre-match ratings.")
    print("==================================================\n")


if __name__ == "__main__":
    main()
