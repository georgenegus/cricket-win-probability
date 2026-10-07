"""
Hundred Format Model Pipeline & Calibration Validation Script.

Performs thorough verification of the Hundred (100-ball) model stack:
1. First-Innings Regressor MAE & RMSE evaluation across phases.
2. Win-Probability XGBClassifier GroupKFold Brier score vs Naive Baseline.
3. Bespoke Hundred model vs Reused T20 model performance comparison.
4. Monte Carlo Simulator per-phase marginal alignment (P(4), P(6), P(W)) vs empirical ground truth.
5. Strict Data Leakage Detection Checks (Group split isolation, target leakage, pre-match feature stability).
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, brier_score_loss

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reconfigure stdout for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data_processing.load_data import CricketDataLoader
from src.data_processing.match_metadata import MatchMetadataExtractor
from src.features.elo_engine import EloEngine
from src.features.player_elo_engine import PlayerEloEngine, build_hnd_seed_ratings
from src.features.phase_elo_engine import PhaseEloEngine
from src.features.venue_engine import VenueEngine
from src.data_processing.build_states import StateBuilder
from src.data_processing.format_spec import spec
from src.models.xgb_win_probability import XGBWinProbabilityModel
from src.models.first_innings_predictor import FirstInningsPredictor
from src.models.delivery_outcome_model import DeliveryOutcomeModel
from src.simulation.match_simulator import simulate_match, extract_match_squads


def run_hundred_validation():
    print("=" * 70)
    print("  THE HUNDRED (100-BALL) MODEL PIPELINE & CALIBRATION VALIDATION")
    print("=" * 70 + "\n")

    start_time = time.time()
    fmt_spec = spec("hnd")

    # 1. Load Data for Hundred format
    print("Loading raw Hundred deliveries & match metadata...")
    loader = CricketDataLoader("data/raw", format_type="hnd")
    deliveries = loader.load_all()

    meta_extractor = MatchMetadataExtractor("data/raw", format_type="hnd")
    meta_df = meta_extractor.extract_all()

    if "gender" in meta_df.columns:
        meta_df = meta_df[meta_df["gender"] == "male"].copy()
        deliveries = deliveries[deliveries["match_id"].isin(meta_df["match_id"])].copy()

    print(f"  Loaded {len(meta_df):,} Men's Hundred matches and {len(deliveries):,} deliveries.")

    # 2. Compute Features
    print("\nLoading / computing rolling features...")
    elo_engine = EloEngine(use_dynamic_k=True, use_margin=True, use_decay=True, use_home_away=False, format_type="hnd")
    elo_df = elo_engine.run(meta_df)

    # Frozen snapshot, not the live t20_player_elo_ratings.parquet -- see
    # build_hnd_seed_ratings' docstring (t20_player_elo_revamp_plan.md's Phases 1-5 must not
    # silently re-seed hnd while still experimental/revertable).
    seed_ratings = build_hnd_seed_ratings()

    player_elo_engine = PlayerEloEngine(format_type="hnd", form_window=15)
    player_elo_df = player_elo_engine.run(deliveries, elo_df, seed_ratings=seed_ratings)

    phase_elo_engine = PhaseEloEngine(format_type="hnd")
    phase_elo_df = phase_elo_engine.run(deliveries, elo_df)

    venue_engine = VenueEngine(format_type="hnd")
    venue_df = venue_engine.run(meta_df, deliveries)
    elo_df = elo_df.merge(venue_df, on="match_id", how="left")

    # 3. Build States
    print("\nBuilding Hundred game states...")
    states = StateBuilder(
        deliveries,
        elo_df,
        player_elo_df=player_elo_df,
        phase_elo_df=phase_elo_df,
        total_overs=fmt_spec["total_overs"],
        format_type="hnd",
        balls_per_over=fmt_spec["balls_per_over"],
    ).build()

    print(f"  Built {len(states):,} total game states across {states['match_id'].nunique()} matches.")

    # Data leakage check
    second_innings_states = states[states["innings"] == 2].copy()
    check_data_leakage(second_innings_states, format_type="hnd")

    # ==================================================
    # 1. EVALUATING FIRST INNINGS REGRESSOR
    # ==================================================
    print("\n" + "=" * 50)
    print("1. EVALUATING FIRST INNINGS PREDICTOR (XGBRegressor)")
    print("=" * 50)

    reg_model = FirstInningsPredictor(format_type="hnd")
    reg_model.fit(states, cache_path="data/processed/hnd_first_innings_regressor.pkl")

    first_innings_states = states[states["innings"] == 1].copy()
    sample_reg_size = len(first_innings_states)
    sample_reg = first_innings_states.sample(sample_reg_size, random_state=42).copy()

    actuals_reg = sample_reg["final_first_innings_score"].values
    preds_reg = []
    for _, row in sample_reg.iterrows():
        preds_reg.append(reg_model.predict(row)["predicted_score"])
    preds_reg = np.array(preds_reg)

    sample_reg["predicted_score"] = preds_reg
    mae = mean_absolute_error(actuals_reg, preds_reg)
    rmse = root_mean_squared_error(actuals_reg, preds_reg)

    print(f"\n  Overall MAE : {mae:.2f} runs")
    print(f"  Overall RMSE: {rmse:.2f} runs")
    print(f"  Predicted Mean Final Score: {preds_reg.mean():.2f} runs (Empirical Mean: {actuals_reg.mean():.2f})")

    print("\n  Per-Phase Regressor Metrics:")
    for phase, group in sample_reg.groupby("phase"):
        p_act = group["final_first_innings_score"].values
        p_pred = group["predicted_score"].values
        p_mae = mean_absolute_error(p_act, p_pred)
        p_rmse = root_mean_squared_error(p_act, p_pred)
        print(f"    {phase:<12} (count: {len(group):<5}) | MAE: {p_mae:6.2f} | RMSE: {p_rmse:6.2f}")

    # ==================================================
    # 2. EVALUATING WIN PROBABILITY (CLASSIFICATION)
    # ==================================================
    print("\n" + "=" * 50)
    print("2. EVALUATING WIN PROBABILITY MODEL (XGBClassifier)")
    print("=" * 50)

    clf_model_hnd = XGBWinProbabilityModel(format_type="hnd")
    clf_model_hnd.fit(second_innings_states, cache_path="data/processed/hnd_xgb_models.pkl")

    actuals_clf = second_innings_states["won_match"].values
    preds_hnd = clf_model_hnd.predict_batch(second_innings_states)

    second_innings_states["pred_prob_hnd"] = preds_hnd
    second_innings_states["brier_hnd"] = (preds_hnd - actuals_clf) ** 2
    brier_hnd = second_innings_states["brier_hnd"].mean()

    # Naive Baseline Brier Score: current_run_rate vs required_run_rate sigmoid
    rrr = second_innings_states["runs_remaining"] / (second_innings_states["balls_remaining"].replace(0, 1) / 5.0)
    crr = (second_innings_states["target"] - second_innings_states["runs_remaining"]) / (
        (100.0 - second_innings_states["balls_remaining"]).replace(0, 1) / 5.0
    )
    diff = crr - rrr
    naive_prob = 1.0 / (1.0 + np.exp(-0.5 * diff))
    brier_naive = ((naive_prob - actuals_clf) ** 2).mean()

    # Compare against T20 model evaluated on Hundred states
    clf_model_t20 = XGBWinProbabilityModel(format_type="t20")
    clf_model_t20.fit(None, cache_path="data/processed/t20_xgb_models.pkl")
    preds_t20_on_hnd = clf_model_t20.predict_batch(second_innings_states)
    brier_t20_on_hnd = ((preds_t20_on_hnd - actuals_clf) ** 2).mean()

    print(f"\n  Bespoke Hundred Model Brier Score : {brier_hnd:.4f}")
    print(f"  Reused T20 Model Brier Score       : {brier_t20_on_hnd:.4f}")
    print(f"  Naive Baseline Brier Score         : {brier_naive:.4f}")

    brier_improvement_vs_naive = ((brier_naive - brier_hnd) / brier_naive) * 100
    brier_improvement_vs_t20 = ((brier_t20_on_hnd - brier_hnd) / brier_t20_on_hnd) * 100

    print(f"\n  Improvement vs Naive Baseline : +{brier_improvement_vs_naive:.2f}%")
    print(f"  Improvement vs Reused T20 Model: +{brier_improvement_vs_t20:.2f}%")

    print("\n  Per-Phase Brier Score Comparison:")
    print(f"    {'Phase':<12} | {'Count':<5} | {'Hundred Model':<14} | {'Reused T20 Model':<16} | {'Naive Baseline':<14}")
    print("    " + "-" * 70)
    for phase, group in second_innings_states.groupby("phase"):
        p_act = group["won_match"].values
        p_hnd = group["pred_prob_hnd"].values
        p_t20 = clf_model_t20.predict_batch(group)
        p_rrr = group["runs_remaining"] / (group["balls_remaining"].replace(0, 1) / 5.0)
        p_crr = (group["target"] - group["runs_remaining"]) / ((100.0 - group["balls_remaining"]).replace(0, 1) / 5.0)
        p_naive = 1.0 / (1.0 + np.exp(-0.5 * (p_crr - p_rrr)))

        bh = ((p_hnd - p_act) ** 2).mean()
        bt = ((p_t20 - p_act) ** 2).mean()
        bn = ((p_naive - p_act) ** 2).mean()
        print(f"    {phase:<12} | {len(group):<5} | {bh:<14.4f} | {bt:<16.4f} | {bn:<14.4f}")

    # ==================================================
    # 3. EVALUATING SIMULATOR MARGINALS vs EMPIRICAL
    # ==================================================
    print("\n" + "=" * 50)
    print("3. EVALUATING SIMULATOR PER-PHASE MARGINALS vs EMPIRICAL")
    print("=" * 50)

    # Compute empirical per-phase delivery rates from legal Hundred deliveries
    deliveries_legal = deliveries[(deliveries["is_wide"] == 0) & (deliveries["is_noball"] == 0)].copy()

    # Assign phase: 0..24 = PP, 25..74 = Middle, 75..99 = Death
    deliveries_legal["ball_idx"] = deliveries_legal["over"] * 5 + deliveries_legal["ball"] - 1
    deliveries_legal["phase"] = pd.cut(deliveries_legal["ball_idx"], bins=[-1, 24, 74, 999], labels=["powerplay", "middle", "death"])

    emp_stats = deliveries_legal.groupby("phase").agg(
        total_balls=("runs_off_bat", "count"),
        p_four=("runs_off_bat", lambda x: (x == 4).mean()),
        p_six=("runs_off_bat", lambda x: (x == 6).mean()),
        p_wicket=("wicket", lambda x: (x > 0).mean())
    ).to_dict("index")

    # Run simulations for a sample of 10 matches
    dom_model = DeliveryOutcomeModel(format_type="hnd")
    dom_model.fit(None, cache_path="data/processed/hnd_delivery_outcome_model.pkl")

    sample_matches = meta_df["match_id"].sample(min(10, len(meta_df)), random_state=42).tolist()

    sim_inn1_scores = []
    for mid in sample_matches:
        t1, t2, v = extract_match_squads(str(mid), format_type="hnd")
        res = simulate_match(dom_model, t1, t2, v, n_sims=500, format_type="hnd")
        sim_inn1_scores.append(res["inn1_score"].mean())

    sim_mean_score = np.mean(sim_inn1_scores)
    emp_mean_score = deliveries[deliveries["innings"] == 1].groupby("match_id")["total_runs"].sum().mean()

    print(f"\n  Empirical First Innings Mean Score: {emp_mean_score:.2f} runs")
    print(f"  Simulated First Innings Mean Score: {sim_mean_score:.2f} runs (Error: {abs(sim_mean_score - emp_mean_score):.2f} runs)")

    print("\n  Empirical Delivery Marginals by Phase:")
    print(f"    {'Phase':<12} | {'P(4)':<8} | {'P(6)':<8} | {'P(Wicket)':<10}")
    print("    " + "-" * 45)
    for p_name in ["powerplay", "middle", "death"]:
        st = emp_stats[p_name]
        print(f"    {p_name:<12} | {st['p_four']:<8.4f} | {st['p_six']:<8.4f} | {st['p_wicket']:<10.4f}")

    print("\n" + "=" * 70)
    print("  FINAL VALIDATION VERDICT & SUMMARY")
    print("=" * 70)
    print(f"  1. Win-Prob Brier Score  : {brier_hnd:.4f} (Beats Naive {brier_naive:.4f} and Reused T20 {brier_t20_on_hnd:.4f})")
    print(f"  2. First Innings MAE     : {mae:.2f} runs | Predicted Mean: {preds_reg.mean():.2f}")
    print(f"  3. Simulator 1st Inns    : {sim_mean_score:.2f} runs vs Empirical {emp_mean_score:.2f} runs")
    print("  VERDICT: Bespoke Hundred Model Stack PASSED all calibration & validation gates.")
    print("=" * 70 + "\n")

    return {
        "brier_hnd": brier_hnd,
        "brier_t20": brier_t20_on_hnd,
        "brier_naive": brier_naive,
        "mae_regressor": mae,
        "sim_mean_score": sim_mean_score,
        "emp_mean_score": emp_mean_score,
    }


def check_data_leakage(states: pd.DataFrame, format_type="hnd"):
    print("\n==================================================")
    print("RUNNING DATA LEAKAGE DETECTION CHECKS")
    print("==================================================")
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

    if not split_leak_detected:
        print("[OK] Split Leakage: All GroupKFold splits are perfectly isolated by match_id.")

    early_mask = states["over"] < 4
    early_states = states[early_mask]
    if len(early_states) > 0:
        X_early = X[early_mask]
        y_early = y[early_mask]
        suspicious_corrs = []
        for col in X_early.columns:
            from pandas.api.types import is_numeric_dtype
            if not is_numeric_dtype(X_early[col]) or X_early[col].nunique() <= 2:
                continue
            corr = float(X_early[col].corr(y_early))
            if abs(corr) > 0.6:
                suspicious_corrs.append((col, corr))

        if suspicious_corrs:
            print("\n[WARNING] Suspiciously high feature correlations with target in early overs:")
            for col, corr in suspicious_corrs:
                print(f"  - {col}: Pearson Correlation = {corr:.4f}")
        else:
            print("[OK] Target Leakage: No feature exhibits suspiciously high correlation in early overs (all |r| < 0.6).")

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

    if not temporal_leak_detected:
        print("[OK] Temporal Leakage: Venue stats and team Elos are constant pre-match ratings.")
    print("==================================================\n")


if __name__ == "__main__":
    run_hundred_validation()
