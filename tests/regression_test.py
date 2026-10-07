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
import argparse
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
from src.data_processing.format_spec import spec, ACTIVE_FORMATS, SEED_SOURCE
from src.features.elo_engine import EloEngine
from src.features.player_elo_engine import PlayerEloEngine, build_seed_ratings
from src.features.phase_elo_engine import PhaseEloEngine
from src.features.venue_engine import VenueEngine
from src.features import rating_source
from src.models.xgb_win_probability import XGBWinProbabilityModel
from src.models.first_innings_predictor import FirstInningsPredictor

DEFAULT_TOLERANCE = {
    "overall_brier_max_increase": 0.0010,
    "phase_brier_max_increase": 0.0025,
    "overall_mae_max_increase": 0.50,
    "phase_mae_max_increase": 1.00
}

FMT_LABELS = {
    "t20": "T20", "od": "ODI", "hnd": "The Hundred", "t20w": "T20 (Women)",
    "odw": "ODI (Women)", "hndw": "The Hundred (Women)",
}


def load_baseline_metrics(baseline_path: Path = None) -> dict:
    if baseline_path is None:
        baseline_path = PROJECT_ROOT / "data" / "processed" / "baseline_metrics.json"

    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline metrics file not found: {baseline_path}. "
            f"Run 'py tests/regression_test.py --generate' to lock a fresh baseline first."
        )

    with open(baseline_path, "r") as f:
        return json.load(f)


def _calibration_buckets(sample_clf: pd.DataFrame) -> list:
    """Predicted-vs-actual win-probability calibration in 0.1-wide buckets.

    Mirrors tests/win_probability_validation.py's long-standing binning exactly (same
    np.arange(0, 1.1, 0.1) edges, same include_lowest, same predicted/actual/count aggregation)
    so the two never disagree about what "the calibration curve" means for a format.

    Added 2026-08-13 for scripts/build_accuracy_report.py, which feeds app.py's /api/accuracy --
    that endpoint previously served HARDCODED calibration curves and Brier scores for every
    format (stale literals that matched no real measurement, and none at all for the women's
    formats). Computed here rather than in a separate script so it reuses the exact same
    out-of-fold predictions the regression gate itself scores, and can never drift from them.
    """
    df = sample_clf[["pred_prob", "won_match"]].copy()
    df["bucket"] = pd.cut(df["pred_prob"], bins=np.arange(0, 1.1, 0.1), include_lowest=True)
    grouped = (
        df.groupby("bucket", observed=False)
          .agg(predicted=("pred_prob", "mean"),
               actual=("won_match", "mean"),
               count=("won_match", "size"))
    )
    out = []
    for bucket, row in grouped.iterrows():
        if int(row["count"]) == 0:
            continue  # empty bin -- omit rather than emit NaN predicted/actual to the UI
        out.append({
            "bucket": str(bucket),
            "predicted": round(float(row["predicted"]), 4),
            "actual": round(float(row["actual"]), 4),
            "count": int(row["count"]),
        })
    return out


def evaluate_format_models(format_type: str, seed: int = 42, evaluate_full: bool = True,
                           include_calibration: bool = False,
                           source: str = rating_source.DEFAULT_RATING_SOURCE) -> dict:
    """
    Evaluates both regressor and win probability models for a given format type ('t20' or 'od').
    By default, evaluates Win Probability on the full second-innings dataset to guarantee exact metrics.

    include_calibration: additive, default OFF -- when True the result also carries a
    "calibration" list (see _calibration_buckets). The regression gate never sets it, so its
    own path and the metrics it locks are byte-identical to before this flag existed.

    source: additive (rating_source_toggle_plan.md Phase 5), default "elo" -- every existing
    call site (the regression gate itself included) is untouched by construction. A non-"elo"
    source skips Elo computation entirely and reads the pre-built combined-rating artifact
    (scripts/build_combined_ratings.py) instead; PhaseEloEngine/VenueEngine stay IDENTICAL
    either way (format-structural only, consume no rating values -- same "shared, identical in
    both arms" design scripts/measure_rating_swap_brier.py validated the swap on). Model cache
    paths get a `_{source}` suffix so a non-elo run can never read or overwrite the deployed
    Elo artifacts.
    """
    fmt_spec = spec(format_type)
    # form_window=15 for every format, matching production (cricsheet_importer.retrain_full)
    # exactly. Was previously 5 for non-t20 formats -- harmless while od/hnd were frozen, but
    # a real discrepancy once od becomes the format under active tuning (od_player_elo_revamp_plan.md
    # Phase 0): a stale/invalidated od cache recomputed here with the wrong window would silently
    # corrupt production's recent_form values (a genuine model feature, not cosmetic).
    form_window = 15

    raw_path = PROJECT_ROOT / "data" / "raw"
    loader = CricketDataLoader(str(raw_path), format_type=format_type)
    deliveries = loader.load_all()

    meta_extractor = MatchMetadataExtractor(str(raw_path), format_type=format_type)
    meta_df = meta_extractor.extract_all()
    if "gender" in meta_df.columns:
        # Filter to this format's own gender -- was hardcoded to "male", which would zero out
        # an all-female corpus like t20w. See cricsheet_importer.retrain_full's identical fix.
        meta_df = meta_df[meta_df["gender"] == fmt_spec["gender"]].copy()
        deliveries = deliveries[deliveries["match_id"].isin(meta_df["match_id"])].copy()

    if source == rating_source.DEFAULT_RATING_SOURCE:
        elo_engine = EloEngine(use_dynamic_k=True, use_margin=True, use_decay=True, use_home_away=False, format_type=format_type)
        elo_df = elo_engine.run(meta_df)

        # Safety net matching production (cricsheet_importer.retrain_full): a pooled format's
        # player Elo seeds from its SEED_SOURCE frozen snapshot (hnd<-t20, hndw<-t20w). On a
        # cache hit run() returns before using this, so it's a no-op for the normal path -- but
        # if the cache is ever invalidated, this prevents a seedless recompute silently
        # overwriting the production parquet with wrong ratings.
        seed_ratings = build_seed_ratings(format_type) if format_type in SEED_SOURCE else None
        player_elo_engine = PlayerEloEngine(format_type=format_type, form_window=form_window)
        player_elo_df = player_elo_engine.run(deliveries, elo_df, seed_ratings=seed_ratings)
    else:
        # Phase 5: the rating BUILD (Phase 1's bridge validation, Phase 2's combining/scaling)
        # is already done and separately verified -- this function's job is training/evaluating
        # models on top of it, not recomputing it.
        elo_df = rating_source.load_team_ratings(format_type, source)
        player_elo_df = rating_source.load_player_ratings(format_type, source)

    phase_elo_engine = PhaseEloEngine(format_type=format_type)
    phase_elo_df = phase_elo_engine.run(deliveries, elo_df)

    venue_engine = VenueEngine(format_type=format_type)
    venue_df = venue_engine.run(meta_df, deliveries)
    elo_df = elo_df.merge(venue_df, on="match_id", how="left")

    # NOTE: format_type/balls_per_over must be threaded through explicitly -- StateBuilder
    # defaults to format_type="t20" internally, which silently mis-keys od's rolling_window
    # (18 instead of 36) and default_econ (8.0 instead of 5.0) if omitted.
    states = StateBuilder(
        deliveries,
        elo_df,
        player_elo_df=player_elo_df,
        phase_elo_df=phase_elo_df,
        total_overs=fmt_spec["total_overs"],
        format_type=format_type,
        balls_per_over=fmt_spec["balls_per_over"],
    ).build()

    source_suffix = "" if source == rating_source.DEFAULT_RATING_SOURCE else f"_{source}"

    # --- 1. Regressor Evaluation (Innings 1) ---
    reg_cache_name = f"{format_type}{source_suffix}_first_innings_regressor.pkl"
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
    clf_cache_name = f"{format_type}{source_suffix}_xgb_models.pkl"
    clf_cache_path = PROJECT_ROOT / "data" / "processed" / clf_cache_name
    # Legacy unsuffixed t20 fallback -- only meaningful for the default Elo source, since no
    # such legacy file has ever existed for a rating-source-toggle model.
    if not clf_cache_path.exists() and format_type == "t20" and source == rating_source.DEFAULT_RATING_SOURCE:
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

    # Computed BEFORE the frames are freed below; only when asked for, so the gate path's
    # memory profile on this 7.7GB box is unchanged.
    calibration = _calibration_buckets(sample_clf) if include_calibration else None

    del deliveries, elo_df, venue_df, meta_df, player_elo_df, phase_elo_df, states
    gc.collect()

    result = {
        "win_probability": {
            "overall_brier": round(clf_overall_brier, 6),
            "sample_size": int(len(sample_clf)),
            "per_phase_brier": clf_phase_brier
        },
        "first_innings": {
            "overall_mae": round(reg_overall_mae, 4),
            "overall_rmse": round(reg_overall_rmse, 4),
            "sample_size": int(len(sample_reg)),
            "per_phase_mae": reg_phase_mae
        }
    }
    if calibration is not None:
        result["calibration"] = calibration
    return result


def generate_baseline(baseline_path: Path = None, tolerance: dict = None):
    """Computes fresh t20/od metrics against the FULL current corpus and locks them as the baseline."""
    if baseline_path is None:
        baseline_path = PROJECT_ROOT / "data" / "processed" / "baseline_metrics.json"
    if tolerance is None:
        tolerance = DEFAULT_TOLERANCE

    print("=" * 60)
    print("GENERATING FRESH T20/OD/HND BASELINE (full corpus, evaluate_full=True)")
    print("=" * 60)

    result = {}
    for format_type in ACTIVE_FORMATS:
        fmt_label = FMT_LABELS[format_type]
        print(f"\n--- Evaluating {fmt_label} on full current corpus ---")
        t_start = time.time()
        result[format_type] = evaluate_format_models(format_type, evaluate_full=True)
        elapsed = time.time() - t_start
        wp = result[format_type]["win_probability"]
        fi = result[format_type]["first_innings"]
        print(f"  Win Probability Brier : {wp['overall_brier']:.6f} (n={wp['sample_size']:,})")
        print(f"  First Innings MAE     : {fi['overall_mae']:.2f} runs (n={fi['sample_size']:,})")
        print(f"  Completed in {elapsed:.1f}s")

    result["tolerance"] = tolerance

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nWrote fresh baseline to {baseline_path}")
    print("Run 'py tests/regression_test.py' (no flags) after future changes to confirm no regression.")
    return result


class TestModelRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.baseline = load_baseline_metrics()
        cls.tolerance = cls.baseline.get("tolerance", DEFAULT_TOLERANCE)

    def test_t20_model_regression(self):
        actual = evaluate_format_models("t20", evaluate_full=True)
        expected = self.baseline["t20"]
        self._verify_metrics("T20", actual, expected)

    def test_od_model_regression(self):
        actual = evaluate_format_models("od", evaluate_full=True)
        expected = self.baseline["od"]
        self._verify_metrics("ODI", actual, expected)

    def test_hnd_model_regression(self):
        if "hnd" not in self.baseline:
            self.skipTest("No hnd baseline locked yet")
        actual = evaluate_format_models("hnd", evaluate_full=True)
        expected = self.baseline["hnd"]
        self._verify_metrics("The Hundred", actual, expected)

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

    for format_type in ACTIVE_FORMATS:
        if format_type not in baseline_metrics:
            print(f"\n--- Skipping {FMT_LABELS[format_type]} (no baseline locked yet) ---")
            continue
        fmt_label = FMT_LABELS[format_type]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Lock a fresh baseline from the current full corpus (overwrites baseline_metrics.json).")
    parser.add_argument("--generate-format", type=str, default=None,
                         help="Lock/update the baseline for a single format only (e.g. t20w), preserving every other format's existing baseline entry.")
    args = parser.parse_args()

    baseline_file = PROJECT_ROOT / "data" / "processed" / "baseline_metrics.json"

    if args.generate_format:
        fmt = args.generate_format
        baseline = load_baseline_metrics(baseline_file) if baseline_file.exists() else {"tolerance": DEFAULT_TOLERANCE}
        print(f"--- Locking baseline for single format: {fmt} ---")
        t_start = time.time()
        baseline[fmt] = evaluate_format_models(fmt, evaluate_full=True)
        elapsed = time.time() - t_start
        wp = baseline[fmt]["win_probability"]
        fi = baseline[fmt]["first_innings"]
        print(f"  Win Probability Brier : {wp['overall_brier']:.6f} (n={wp['sample_size']:,})")
        print(f"  First Innings MAE     : {fi['overall_mae']:.2f} runs (n={fi['sample_size']:,})")
        print(f"  Completed in {elapsed:.1f}s")
        with open(baseline_file, "w") as f:
            json.dump(baseline, f, indent=2)
        print(f"Wrote updated baseline (format={fmt}) to {baseline_file}")
        sys.exit(0)
    elif args.generate or not baseline_file.exists():
        generate_baseline(baseline_file)
        sys.exit(0)
    else:
        baseline = load_baseline_metrics(baseline_file)
        exit_code = print_comparison_report(baseline)
        sys.exit(exit_code)
