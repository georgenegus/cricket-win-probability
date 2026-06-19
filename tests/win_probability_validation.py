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
from src.models.win_probability import WinProbabilityModel
from src.models.similarity_engine import SimpleSimilarityEngine

# --------------------------------------------------
# Timing utilities
# --------------------------------------------------

timings = {}


def start_timer():
    return time.time()


def stop_timer(name, start):
    timings[name] = time.time() - start


# --------------------------------------------------
# Validation
# --------------------------------------------------

def run_validation(model, states: pd.DataFrame):

    print("\n==================================================")
    print("WIN PROBABILITY MODEL VALIDATION")
    print("==================================================")

    validation_start = time.time()

    results = []

    # --------------------------------------------------
    # Sampling
    # --------------------------------------------------

    sample_size = min(500, len(states))

    sample = states.sample(
        sample_size,
        random_state=42
    )

    print(f"\nTesting {len(sample):,} random states")

    # --------------------------------------------------
    # Prediction Loop
    # --------------------------------------------------

    prediction_start = start_timer()

    for idx, (_, row) in enumerate(sample.iterrows()):

        if idx % 50 == 0:
            print(
                f"Processed {idx}/{len(sample)} states..."
            )

        # Prevent data leakage
        history = states[
            states["match_id"] != row["match_id"]
        ]

        out = model.predict(
            row,
            history
        )

        results.append({
            "match_id": row["match_id"],
            "win_probability": out["win_probability"],
            "sample_size": out["sample_size"],
            "status": out["status"],
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

    print("\n[PERFORMANCE REPORT]")
    print("-" * 50)

    for stage, duration in timings.items():

        print(
            f"{stage:<20}"
            f"{duration:>10.3f} sec"
        )

    print("-" * 50)

    total = sum(
        v
        for k, v in timings.items()
        if k != "validation_total"
    )

    # Calculate per-phase Brier scores
    per_phase = {}
    for phase, group in success.groupby("phase"):
        per_phase[phase] = float(group["error"].mean())

    print("\n==================================================")
    print("VALIDATION COMPLETE")
    print("==================================================")

    return {
        "overall": float(brier_score),
        "per_phase": per_phase
    }


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

    print("Building states...")

    t = start_timer()

    states = StateBuilder(
        deliveries
    ).build()

    # print("STATE COLUMNS:", states.columns.tolist())
    states = states[states["innings"] == 2].copy()
    print(states.head(5))

    stop_timer(
        "build_states",
        t
    )

    # Only second innings states
    states = states[
        states["innings"] == 2
    ].copy()

    # Run validation for multiple phase modes to compare
    modes = ["none", "filter", "feature"]
    comparison = {}

    for mode in modes:
        print(f"\n==================================================")
        print(f"RUNNING VALIDATION FOR PHASE MODE: {mode.upper()}")
        print(f"==================================================")
        
        engine = SimpleSimilarityEngine(
            k=100,
            phase_mode=mode
        )

        model = WinProbabilityModel(
            similarity_engine=engine
        )

        res = run_validation(
            model,
            states
        )
        comparison[mode] = res

    print("\n\n" + "="*50)
    print("PHASE-AWARE SIMILARITY COMPARISON RESULT")
    print("="*50)
    print(f"{'Phase Mode':<15} | {'Overall Brier':<15} | {'Powerplay Brier':<15} | {'Middle Brier':<15} | {'Death Brier':<15}")
    print("-"*85)
    for mode in modes:
        res = comparison[mode]
        pp = res["per_phase"].get("powerplay", np.nan)
        mid = res["per_phase"].get("middle", np.nan)
        death = res["per_phase"].get("death", np.nan)
        print(f"{mode:<15} | {res['overall']:15.4f} | {pp:15.4f} | {mid:15.4f} | {death:15.4f}")
    print("="*50)