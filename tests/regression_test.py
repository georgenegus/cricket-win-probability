import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import time
import gc
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.load_data import CricketDataLoader
from src.data_processing.build_states import StateBuilder
from src.data_processing.match_metadata import MatchMetadataExtractor
from src.features.elo_engine import EloEngine
from src.features.player_elo_engine import PlayerEloEngine
from src.features.phase_elo_engine import PhaseEloEngine
from src.features.venue_engine import VenueEngine
from src.models.xgb_win_probability import XGBWinProbabilityModel
from src.models.first_innings_predictor import FirstInningsPredictor


def load_baseline_metrics(baseline_path: Path = None) -> dict:
    if baseline_path is None:
        baseline_path = PROJECT_ROOT / "data" / "processed" / "baseline_metrics.json"
    
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline metrics file not found: {baseline_path}")
    
    with open(baseline_path, "r") as f:
        return json.load(f)


def evaluate_format_models(format_type: str, seed: int = 42, evaluate_full: bool = True) -> dict:
    """
    Evaluates both regressor and win probability models for a given format type ('t20' or 'od').
    By default, evaluates Win Probability on the full second-innings dataset to guarantee exact metrics.
    """
    total_overs = 20 if format_type == "t20" else 50
    form_window = 15 if format_type == "t20" else 5

    raw_path = PROJECT_ROOT / "data" / "raw"
    loader = CricketDataLoader(str(raw_path), format_type=format_type)
    deliveries = loader.load_all()

    meta_extractor = MatchMetadataExtractor(str(raw_path), format_type=format_type)
    meta_df = meta_extractor.extract_all()
    if "gender" in meta_df.columns:
        meta_df = meta_df[meta_df["gender"] == "male"].copy()
        deliveries = deliveries[deliveries["match_id"].isin(meta_df["match_id"])].copy()

    elo_engine = EloEngine(use_dynamic_k=True, use_margin=True, use_decay=True, use_home_away=False, format_type=format_type)
    elo_df = elo_engine.run(meta_df)

    player_elo_engine = PlayerEloEngine(format_type=format_type, form_window=form_window)
    player_elo_df = player_elo_engine.run(deliveries, elo_df)

    phase_elo_engine = PhaseEloEngine(format_type=format_type)
    phase_elo_df = phase_elo_engine.run(deliveries, elo_df)

    venue_engine = VenueEngine(format_type=format_type)
    venue_df = venue_engine.run(meta_df, deliveries)
    elo_df = elo_df.merge(venue_df, on="match_id", how="left")

    states = StateBuilder(
        deliveries,
        elo_df,
        player_elo_df=player_elo_df,
        phase_elo_df=phase_elo_df,
        total_overs=total_overs
    ).build()

    # --- 1. Regressor Evaluation (Innings 1) ---
    reg_cache_name = f"{format_type}_first_innings_regressor.pkl"
    reg_cache_path = PROJECT_ROOT / "data" / "processed" / reg_cache_name
    reg_model = FirstInningsPredictor(format_type=format_type)
    reg_model.fit(states, cache_path=str(reg_cache_path))

    first_innings_states = states[states["innings"] == 1].copy()
    sample_reg_size = min(5000, len(first_innings_states))
    sample_reg = first_innings_states.sample(sample_reg_size, random_state=seed).copy()

    actuals_reg = sample_reg["final_first_innings_score"].values
    preds_reg = [reg_model.predict(row)["predicted_score"] for _, row in sample_reg.iterrows()]
    sample_reg["predicted_score"] = preds_reg

    reg_overall_mae = float(mean_absolute_error(actuals_reg, preds_reg))
    reg_overall_rmse = float(root_mean_squared_error(actuals_reg, preds_reg))

    reg_phase_mae = {}
    for phase, group in sample_reg.groupby("phase"):
        p_act = group["final_first_innings_score"].values
        p_pred = group["predicted_score"].values
        reg_phase_mae[str(phase)] = round(float(mean_absolute_error(p_act, p_pred)), 4)

    # --- 2. Win Probability Classifier Evaluation (Innings 2) ---
    clf_cache_name = "t20_xgb_models.pkl" if format_type == "t20" else "od_xgb_models.pkl"
    clf_cache_path = PROJECT_ROOT / "data" / "processed" / clf_cache_name
    if not clf_cache_path.exists() and format_type == "t20":
        clf_cache_path = PROJECT_ROOT / "data" / "processed" / "xgb_models.pkl"

    clf_model = XGBWinProbabilityModel(format_type=format_type)
    clf_model.fit(states[states["innings"] == 2], cache_path=str(clf_cache_path))

    second_innings_states = states[states["innings"] == 2].copy()
    if evaluate_full:
        sample_clf = second_innings_states
    else:
        sample_clf_size = min(10000, len(second_innings_states))
        sample_clf = second_innings_states.sample(sample_clf_size, random_state=seed).copy()

    batch_probs = clf_model.predict_batch(sample_clf)
    sample_clf["pred_prob"] = batch_probs
    sample_clf["error"] = (sample_clf["pred_prob"] - sample_clf["won_match"]) ** 2

    clf_overall_brier = float(sample_clf["error"].mean())
    clf_phase_brier = {}
    for phase, group in sample_clf.groupby("phase"):
        clf_phase_brier[str(phase)] = round(float(group["error"].mean()), 6)

    del deliveries, elo_df, venue_df, meta_df, player_elo_df, phase_elo_df, states
    gc.collect()

    return {
        "win_probability": {
            "overall_brier": round(clf_overall_brier, 6),
            "per_phase_brier": clf_phase_brier
        },
        "first_innings": {
            "overall_mae": round(reg_overall_mae, 4),
            "overall_rmse": round(reg_overall_rmse, 4),
            "per_phase_mae": reg_phase_mae
        }
    }


class TestModelRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.baseline = load_baseline_metrics()
        cls.tolerance = cls.baseline.get("tolerance", {
            "overall_brier_max_increase": 0.0010,
            "phase_brier_max_increase": 0.0025,
            "overall_mae_max_increase": 0.50,
            "phase_mae_max_increase": 1.00
        })

    def test_t20_model_regression(self):
        actual = evaluate_format_models("t20", evaluate_full=True)
        expected = self.baseline["t20"]
        self._verify_metrics("T20", actual, expected)

    def test_od_model_regression(self):
        actual = evaluate_format_models("od", evaluate_full=True)
        expected = self.baseline["od"]
        self._verify_metrics("ODI", actual, expected)

    def _verify_metrics(self, label, actual, expected):
        # 1. Win Probability Classifier Brier Score
        actual_brier = actual["win_probability"]["overall_brier"]
        expected_brier = expected["win_probability"]["overall_brier"]
        max_brier = expected_brier + self.tolerance["overall_brier_max_increase"]

        self.assertLessEqual(
            actual_brier,
            max_brier,
            f"[{label}] Overall Brier score regressed from {expected_brier:.4f} to {actual_brier:.4f} (Max allowed: {max_brier:.4f})"
        )

        for phase, exp_p_brier in expected["win_probability"]["per_phase_brier"].items():
            act_p_brier = actual["win_probability"]["per_phase_brier"].get(phase, 0.0)
            max_p_brier = exp_p_brier + self.tolerance["phase_brier_max_increase"]
            self.assertLessEqual(
                act_p_brier,
                max_p_brier,
                f"[{label}] Phase '{phase}' Brier score regressed from {exp_p_brier:.4f} to {act_p_brier:.4f} (Max allowed: {max_p_brier:.4f})"
            )

        # 2. First Innings Regressor MAE
        actual_mae = actual["first_innings"]["overall_mae"]
        expected_mae = expected["first_innings"]["overall_mae"]
        max_mae = expected_mae + self.tolerance["overall_mae_max_increase"]

        self.assertLessEqual(
            actual_mae,
            max_mae,
            f"[{label}] First Innings MAE regressed from {expected_mae:.2f} to {actual_mae:.2f} (Max allowed: {max_mae:.2f})"
        )

        for phase, exp_p_mae in expected["first_innings"]["per_phase_mae"].items():
            act_p_mae = actual["first_innings"]["per_phase_mae"].get(phase, 0.0)
            max_p_mae = exp_p_mae + self.tolerance["phase_mae_max_increase"]
            self.assertLessEqual(
                act_p_mae,
                max_p_mae,
                f"[{label}] Phase '{phase}' MAE regressed from {exp_p_mae:.2f} to {act_p_mae:.2f} (Max allowed: {max_p_mae:.2f})"
            )


def print_comparison_report(baseline_metrics: dict):
    print("\n==================================================")
    print("AUTOMATED MODEL REGRESSION TEST SUITE")
    print("==================================================")

    tolerance = baseline_metrics.get("tolerance", {})
    all_passed = True

    for format_type in ["t20", "od"]:
        fmt_label = "T20" if format_type == "t20" else "ODI"
        print(f"\n--- Running Regression Check: {fmt_label} Format ---")
        
        actual = evaluate_format_models(format_type, evaluate_full=True)
        expected = baseline_metrics[format_type]

        # Classifier report
        act_brier = actual["win_probability"]["overall_brier"]
        exp_brier = expected["win_probability"]["overall_brier"]
        diff_brier = act_brier - exp_brier
        brier_passed = act_brier <= exp_brier + tolerance.get("overall_brier_max_increase", 0.001)
        if not brier_passed:
            all_passed = False

        status_str = "PASS" if brier_passed else "FAIL [REGRESSION]"
        print(f"  Win Probability Brier : {act_brier:.6f} (Baseline: {exp_brier:.6f} | Diff: {diff_brier:+.6f}) [{status_str}]")

        for phase, exp_p_brier in expected["win_probability"]["per_phase_brier"].items():
            act_p_brier = actual["win_probability"]["per_phase_brier"].get(phase, 0.0)
            p_diff = act_p_brier - exp_p_brier
            p_passed = act_p_brier <= exp_p_brier + tolerance.get("phase_brier_max_increase", 0.0025)
            if not p_passed:
                all_passed = False
            p_status = "PASS" if p_passed else "FAIL [REGRESSION]"
            print(f"    Phase '{phase:<9}' : {act_p_brier:.6f} (Baseline: {exp_p_brier:.6f} | Diff: {p_diff:+.6f}) [{p_status}]")

        # Regressor report
        act_mae = actual["first_innings"]["overall_mae"]
        exp_mae = expected["first_innings"]["overall_mae"]
        diff_mae = act_mae - exp_mae
        mae_passed = act_mae <= exp_mae + tolerance.get("overall_mae_max_increase", 0.5)
        if not mae_passed:
            all_passed = False
        mae_status_str = "PASS" if mae_passed else "FAIL [REGRESSION]"
        print(f"  First Innings MAE    : {act_mae:.2f} (Baseline: {exp_mae:.2f} | Diff: {diff_mae:+.2f}) [{mae_status_str}]")

        for phase, exp_p_mae in expected["first_innings"]["per_phase_mae"].items():
            act_p_mae = actual["first_innings"]["per_phase_mae"].get(phase, 0.0)
            p_mae_diff = act_p_mae - exp_p_mae
            p_mae_passed = act_p_mae <= exp_p_mae + tolerance.get("phase_mae_max_increase", 1.0)
            if not p_mae_passed:
                all_passed = False
            p_mae_status = "PASS" if p_mae_passed else "FAIL [REGRESSION]"
            print(f"    Phase '{phase:<9}' : {act_p_mae:.2f} (Baseline: {exp_p_mae:.2f} | Diff: {p_mae_diff:+.2f}) [{p_mae_status}]")

    print("\n==================================================")
    if all_passed:
        print("ALL REGRESSION CHECKS PASSED SUCCESSFULLY [0 REGRESSIONS DETECTED]")
        print("==================================================")
        return 0
    else:
        print("REGRESSION CHECK FAILED [PERFORMANCE REGRESSION DETECTED]")
        print("==================================================")
        return 1


if __name__ == "__main__":
    baseline = load_baseline_metrics()
    exit_code = print_comparison_report(baseline)
    sys.exit(exit_code)
