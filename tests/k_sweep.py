import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path("c:/Python/T20_Win_Probability_Model")
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.load_data import CricketDataLoader
from src.data_processing.build_states import StateBuilder

if __name__ == "__main__":
    print("Loading data...")
    loader = CricketDataLoader(str(PROJECT_ROOT / "data/raw"))
    deliveries = loader.load_all()

    print("Building states...")
    states = StateBuilder(deliveries).build()
    states = states[states["innings"] == 2].copy()

    # Sample test set (500 states)
    sample = states.sample(500, random_state=42)
    
    k_values = [30, 50, 100, 150, 200, 300, 400, 500, 750, 1000]
    
    # We will accumulate errors for each k
    errors = {k: [] for k in k_values}

    # Pre-calculate std values for scaling
    sigma_rr = states["runs_remaining"].std()
    sigma_br = states["balls_remaining"].std()
    sigma_wk = states["wickets_lost"].std()

    sigma_rr = sigma_rr if (pd.notna(sigma_rr) and sigma_rr > 0) else 50.0
    sigma_br = sigma_br if (pd.notna(sigma_br) and sigma_br > 0) else 30.0
    sigma_wk = sigma_wk if (pd.notna(sigma_wk) and sigma_wk > 0) else 2.0

    print("Running predictions loop...")
    for idx, (_, row) in enumerate(sample.iterrows()):
        if idx % 50 == 0:
            print(f"Processed {idx}/500 states...")

        # Filter same match to prevent leakage
        history = states[states["match_id"] != row["match_id"]].copy()

        # Calculate distances
        rr = history["runs_remaining"] - row["runs_remaining"]
        br = history["balls_remaining"] - row["balls_remaining"]
        wk = history["wickets_lost"] - row["wickets_lost"]

        distance = (
            (rr / sigma_rr) ** 2 +
            (br / sigma_br) ** 2 +
            (wk / sigma_wk) ** 2
        )

        history = history.assign(distance=distance)
        sorted_history = history.sort_values("distance")

        # Evaluate for all k values
        for k in k_values:
            nearest = sorted_history.head(k)
            weights = np.exp(-nearest["distance"])
            if weights.sum() == 0:
                probability = nearest["won_match"].mean()
            else:
                probability = np.average(nearest["won_match"], weights=weights)
            
            probability = round(float(probability), 4)
            err = (probability - row["won_match"]) ** 2
            errors[k].append(err)

    print("\nSweep Results:")
    for k in k_values:
        brier = np.mean(errors[k])
        print(f"k={k:<5} : {brier:.6f}")
