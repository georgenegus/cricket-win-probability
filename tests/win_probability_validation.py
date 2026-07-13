import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import time

from pathlib import Path
import sys

# --------------------------------------------------
# Add project root to path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------
# Imports
# --------------------------------------------------

from src.data_processing.load_data import CricketDataLoader
from src.data_processing.build_states import StateBuilder
from src.models.xgb_win_probability import XGBWinProbabilityModel


# --------------------------------------------------
# Timing utilities
# --------------------------------------------------

script_start = time.time()
timings = {}


def start_timer():
    return time.time()


def stop_timer(name, start):
    timings[name] = time.time() - start


# --------------------------------------------------
# Validation
# --------------------------------------------------

def run_validation(model, states: pd.DataFrame, model_name="KNN", sample_size=1000):

    print("\n==================================================")
    print(f"WIN PROBABILITY MODEL VALIDATION ({model_name})")
    print("==================================================")

    validation_start = time.time()

    results = []

    # --------------------------------------------------
    # Sampling
    # --------------------------------------------------

    sample_size = min(sample_size, len(states))

    sample = states.sample(
        sample_size,
        random_state=42
    )

    print(f"\nTesting {len(sample):,} random states")

    # --------------------------------------------------
    # Batch Prediction (vectorised)
    # --------------------------------------------------

    prediction_start = start_timer()

    # Use predict_batch for vectorised prediction
    batch_probs = model.predict_batch(sample)

    # Build results DataFrame from batch output
    results = []
    for idx, (_, row) in enumerate(sample.iterrows()):
        results.append({
            "match_id": row["match_id"],
            "win_probability": float(batch_probs[idx]),
            "sample_size": len(model.match_to_fold) if model.match_to_fold else 0,
            "status": "success",
            "runs_remaining": row["runs_remaining"],
            "balls_remaining": row["balls_remaining"],
            "wickets_lost": row["wickets_lost"],
            "won_match": row["won_match"],
            "phase": row.get("phase", "unknown")
        })

    stop_timer(
        "predictions",
        prediction_start
    )

    df = pd.DataFrame(results)

    # --------------------------------------------------
    # Status Counts
    # --------------------------------------------------

    print("\n[STATUS COUNTS]")
    print(df["status"].value_counts())

    success = df[
        df["status"] == "success"
    ].copy()

    print("\n[WICKET ANALYSIS]")

    print(
        success["wickets_lost"]
        .describe()
    )

    print("\nUnique wickets values:")

    print(
        sorted(
            success["wickets_lost"]
            .dropna()
            .unique()
        )
    )

    print(
        "\nSuccessful predictions:",
        len(success)
    )

    print(
        "Failed predictions:",
        len(df) - len(success)
    )

    if len(success) == 0:
        print("\nNo successful predictions.")
        return

    # --------------------------------------------------
    # Probability Checks
    # --------------------------------------------------

    print("\n[PROBABILITY CHECKS]")

    print(
        "Min:",
        success["win_probability"].min()
    )

    print(
        "Max:",
        success["win_probability"].max()
    )

    print(
        "Mean:",
        success["win_probability"].mean()
    )

    print(
        "NaNs:",
        success["win_probability"]
        .isna()
        .sum()
    )

    assert success["win_probability"].between(
        0,
        1
    ).all(), (
        "Probabilities outside [0,1]"
    )

    # --------------------------------------------------
    # Sample Size Analysis
    # --------------------------------------------------

    print("\n[SAMPLE SIZE ANALYSIS]")

    print(
        success["sample_size"]
        .describe()
    )

    # --------------------------------------------------
    # Directionality
    # --------------------------------------------------

    print("\n[DIRECTIONALITY CHECKS]")

    corr_runs = (
        success["runs_remaining"]
        .corr(success["win_probability"])
    )

    corr_balls = (
        success["balls_remaining"]
        .corr(success["win_probability"])
    )

    corr_wickets = (
        success["wickets_lost"]
        .corr(success["win_probability"])
    )

    print("\n[WIN PROBABILITY BY WICKETS LOST]")

    print(
        success.groupby("wickets_lost")
        ["win_probability"]
        .mean()
        .round(3)
    )

    print(
        "Corr (runs_remaining):",
        round(corr_runs, 4)
    )

    print(
        "Corr (balls_remaining):",
        round(corr_balls, 4)
    )

    print(
        "Corr (wickets_lost):",
        round(corr_wickets, 4)
    )

    if not np.isnan(corr_runs):
        assert corr_runs < 0, (
            "Runs remaining should "
            "negatively correlate "
            "with win probability"
        )

    # --------------------------------------------------
    # Brier Score
    # --------------------------------------------------

    brier_start = start_timer()

    print("\n[BRIER SCORE]")

    success["error"] = (
        success["win_probability"]
        - success["won_match"]
    ) ** 2

    brier_score = (
        success["error"]
        .mean()
    )

    print(
        "Brier Score:",
        round(brier_score, 4)
    )

    print("\n[PER-PHASE BRIER SCORE]")
    for phase, group in success.groupby("phase"):
        phase_brier = group["error"].mean()
        print(f"  {phase:<12} Brier Score: {phase_brier:.4f} (count: {len(group)})")

    stop_timer(
        "brier_score",
        brier_start
    )

    # --------------------------------------------------
    # Calibration
    # --------------------------------------------------

    calibration_start = start_timer()

    print("\n[CALIBRATION]")

    success["bucket"] = pd.cut(
        success["win_probability"],
        bins=np.arange(
            0,
            1.1,
            0.1
        ),
        include_lowest=True
    )

    calibration = (
        success
        .groupby(
            "bucket",
            observed=False
        )
        .agg(
            predicted=(
                "win_probability",
                "mean"
            ),
            actual=(
                "won_match",
                "mean"
            ),
            count=(
                "won_match",
                "size"
            )
        )
        .round(3)
    )

    print(calibration)

    stop_timer(
        "calibration",
        calibration_start
    )

    # --------------------------------------------------
    # Edge Cases
    # --------------------------------------------------

    print("\n[EDGE CASES]")

    already_won = success[
        success["runs_remaining"] <= 0
    ]

    if len(already_won) > 0:

        print(
            "Average probability when "
            "runs_remaining <= 0:",
            round(
                already_won[
                    "win_probability"
                ].mean(),
                4
            )
        )

    final_over = success[
        success["balls_remaining"] <= 6
    ]

    if len(final_over) > 0:

        print(
            "Average probability when "
            "balls_remaining <= 6:",
            round(
                final_over[
                    "win_probability"
                ].mean(),
                4
            )
        )

    # --------------------------------------------------
    # Runtime
    # --------------------------------------------------

    stop_timer(
        "validation_total",
        validation_start
    )

    # Stop the overall script timer if it was started at the entry point
    if "script_start" in globals():
        stop_timer("total_script_runtime", script_start)

    print("\n[PERFORMANCE REPORT]")
    print("-" * 50)

    for stage, duration in timings.items():
        print(
            f"{stage:<25}"
            f"{duration:>10.3f} sec"
        )

    print("-" * 50)

    # Sum of all measured sub-component runtimes
    non_summed_keys = {"validation_total", "total_script_runtime"}
    total_components = sum(
        v
        for k, v in timings.items()
        if k not in non_summed_keys
    )

    print(
        f"{'TOTAL (measured steps)':<25}"
        f"{total_components:>10.3f} sec"
    )
    if "total_script_runtime" in timings:
        print(
            f"{'TOTAL SCRIPT RUNTIME':<25}"
            f"{timings['total_script_runtime']:>10.3f} sec"
        )
    print("-" * 50)

    # Calculate per-phase Brier scores
    per_phase = {}
    for phase, group in success.groupby("phase"):
        per_phase[phase] = float(group["error"].mean())

    print("\n==================================================")
    print("VALIDATION COMPLETE")
    print("==================================================")

    return {
        "overall": float(brier_score),
        "per_phase": per_phase,
        "prediction_time": timings.get("predictions", 0.0)
    }


def check_data_leakage(states: pd.DataFrame, format_type="t20"):
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

    # 2. Target Leakage: suspicious early correlations
    early_mask = states["over"] < 3
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
            print("\n[WARNING] Suspiciously high feature correlations with target in early overs (over < 3):")
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


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    print("Loading data...")

    t = start_timer()

    loader = CricketDataLoader(
        "data/raw"
    )

    deliveries = loader.load_all()

    stop_timer(
        "load_data",
        t
    )

    print("Building states with Elo ratings...")

    from src.data_processing.match_metadata import MatchMetadataExtractor
    from src.features.elo_engine import EloEngine
    from src.features.player_elo_engine import PlayerEloEngine
    from src.features.venue_engine import VenueEngine

    # Extract metadata
    print("Extracting match metadata...")
    t_meta = start_timer()
    meta_extractor = MatchMetadataExtractor("data/raw")
    meta_df = meta_extractor.extract_all()
    
    # Filter for male matches only
    if "gender" in meta_df.columns:
        meta_df = meta_df[meta_df["gender"] == "male"].copy()
        print(f"Filtered for male matches only. Remaining matches: {len(meta_df)}")
        deliveries = deliveries[deliveries["match_id"].isin(meta_df["match_id"])].copy()
        print(f"Filtered deliveries. Remaining rows: {len(deliveries)}")
        
    stop_timer("extract_metadata", t_meta)

    # Compute Elo
    print("Computing Elo ratings...")
    t_elo = start_timer()
    elo_engine = EloEngine(use_dynamic_k=True, use_margin=True, use_decay=True, use_home_away=False)
    elo_df = elo_engine.run(meta_df)
    stop_timer("compute_elo", t_elo)

    # Compute Player Elo (dynamic fallback loading from cache if computed)
    print("Computing player Elo ratings...")
    t_player_elo = start_timer()
    player_elo_engine = PlayerEloEngine(format_type="t20", form_window=15)
    player_elo_df = player_elo_engine.run(deliveries, elo_df)
    stop_timer("compute_player_elo", t_player_elo)

    # Compute Phase Elo ratings
    print("Computing phase Elo ratings...")
    t_phase_elo = start_timer()
    from src.features.phase_elo_engine import PhaseEloEngine
    phase_elo_engine = PhaseEloEngine(format_type="t20")
    phase_elo_df = phase_elo_engine.run(deliveries, elo_df)
    stop_timer("compute_phase_elo", t_phase_elo)

    # Compute Venue features
    print("Computing Venue features...")
    t_venue = start_timer()
    venue_engine = VenueEngine()
    venue_df = venue_engine.run(meta_df, deliveries)
    stop_timer("compute_venue", t_venue)

    # Merge Venue features into elo_df
    elo_df = elo_df.merge(venue_df, on="match_id", how="left")

    # Build states
    print("Building states...")
    t_states = start_timer()
    states = StateBuilder(
        deliveries,
        elo_df,
        player_elo_df=player_elo_df,
        phase_elo_df=phase_elo_df
    ).build()
    stop_timer("build_states", t_states)

    # Free memory before copying/subsetting
    import gc
    del deliveries
    del elo_df
    del venue_df
    del meta_df
    del player_elo_df
    del phase_elo_df
    gc.collect()

    # Only second innings states (single filtering step)
    states = states[states["innings"] == 2].copy()
    gc.collect()
    print(states.head(5))


    print(
        f"Validation dataset: "
        f"{len(states):,} states"
    )

    # Run data leakage checks
    check_data_leakage(states, format_type="t20")

    print("\nInitializing and fitting XGBoost model...")
    t_fit = start_timer()
    xgb_model = XGBWinProbabilityModel()
    xgb_model.fit(
        states,
        cache_path="data/processed/xgb_models.pkl"
    )
    stop_timer("fit_model", t_fit)

    # --------------------------------------------------
    # Running Model Validation
    # --------------------------------------------------
    # We increase the validation sample size to 5,000 states for highly precise,
    # low-variance Brier Score metrics.
    validation_sample_size = len(states)

    print(f"\nRunning model validation with sample size {validation_sample_size}...")
    run_validation(
        xgb_model,
        states,
        model_name="XGBoost",
        sample_size=validation_sample_size
    )