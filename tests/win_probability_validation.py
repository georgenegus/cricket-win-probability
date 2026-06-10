import pandas as pd
import numpy as np
import time

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_processing.load_data import CricketDataLoader
from src.data_processing.build_states import StateBuilder
from src.models.win_probability import WinProbabilityModel
from src.models.similarity_engine import SimpleSimilarityEngine

engine = SimpleSimilarityEngine(k=200)

model = WinProbabilityModel(
    similarity_engine=engine
)


def run_validation(model, states: pd.DataFrame):

    print("\n==================================================")
    print("WIN PROBABILITY MODEL VALIDATION")
    print("==================================================")

    start = time.time()

    results = []

    # sample to keep runtime manageable
    sample = states.sample(min(5000, len(states)), random_state=42)

    for _, row in sample.iterrows():

        out = model.predict(row, states)

        results.append({
            "win_probability": out["win_probability"],
            "sample_size": out["sample_size"],
            "status": out["status"],
            "runs_remaining": row.get("runs_remaining"),
            "balls_remaining": row.get("balls_remaining"),
            "won_match": row.get("won_match"),
        })

    df = pd.DataFrame(results)

    # -----------------------------
    # 1. Basic breakdown
    # -----------------------------
    print("\n[STATUS COUNTS]")
    print(df["status"].value_counts())

    success = df[df["status"] == "success"].copy()

    print("\nSuccessful predictions:", len(success))
    print("Failed predictions:", len(df) - len(success))

    # -----------------------------
    # 2. Probability sanity checks
    # -----------------------------
    print("\n[PROBABILITY CHECKS]")

    if len(success) > 0:

        print("Min:", success["win_probability"].min())
        print("Max:", success["win_probability"].max())
        print("Mean:", success["win_probability"].mean())

        assert success["win_probability"].between(0, 1).all()

    # -----------------------------
    # 3. Sample size analysis
    # -----------------------------
    print("\n[SAMPLE SIZE ANALYSIS]")

    print(success["sample_size"].describe())

    print("\n% low-sample predictions:",
          (df["status"] == "insufficient_data").mean())

    # -----------------------------
    # 4. Directionality checks
    # -----------------------------
    print("\n[DIRECTIONALITY CHECKS]")

    if len(success) > 50:

        corr_runs = success["runs_remaining"].corr(success["win_probability"])
        corr_balls = success["balls_remaining"].corr(success["win_probability"])

        print("Corr (runs_remaining):", corr_runs)
        print("Corr (balls_remaining):", corr_balls)

        # expected sign check
        assert corr_runs < 0, "Model directionality issue: runs_remaining should be negative correlation"

    # -----------------------------
    # 5. Edge behaviour
    # -----------------------------
    print("\n[EDGE CASES]")

    if len(success) > 0:

        print("Win prob when runs_remaining <= 0:",
              success[success["runs_remaining"] <= 0]["win_probability"].mean())

        print("Win prob when balls_remaining <= 6:",
              success[success["balls_remaining"] <= 6]["win_probability"].mean())

    # -----------------------------
    # 6. Runtime
    # -----------------------------
    print("\nTOTAL TIME:", round(time.time() - start, 2), "sec")

    print("\n==================================================")
    print("VALIDATION COMPLETE")
    print("==================================================")


if __name__ == "__main__":

    print("Loading data...")
    loader = CricketDataLoader("data/raw")
    df = loader.load_all()

    print("Building states...")
    states = StateBuilder(df).build()

    print("Running model validation...")

    model = WinProbabilityModel(
        similarity_engine=None  # plug your engine in here
    )

    run_validation(model, states)