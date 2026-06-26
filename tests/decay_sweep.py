import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.load_data import CricketDataLoader
from src.data_processing.build_states import StateBuilder
from src.data_processing.match_metadata import MatchMetadataExtractor
from src.features.elo_engine import EloEngine
from src.models.xgb_win_probability import XGBWinProbabilityModel

def main():
    print("==================================================")
    # 1. Load raw data and metadata
    print("Loading raw deliveries...")
    loader = CricketDataLoader("data/raw")
    deliveries = loader.load_all()

    print("Extracting match metadata...")
    meta_extractor = MatchMetadataExtractor("data/raw")
    meta_df = meta_extractor.extract_all()

    # 2. Build base states once (without Elo df to avoid recalculation)
    print("Building base states...")
    base_states = StateBuilder(deliveries, elo_df=None).build()
    base_states = base_states[base_states["innings"] == 2].copy()
    print(f"Base states built: {len(base_states):,} rows.")

    # 3. Create a fixed validation sample of 5,000 states (consistent random_state)
    print("Sampling validation states...")
    validation_sample_size = 5000
    val_sample = base_states.sample(
        min(validation_sample_size, len(base_states)),
        random_state=42
    ).copy()

    # Define grids
    decay_days_grid = [60, 90, 120, 180]
    decay_rate_grid = [0.90, 0.95, 0.98]
    
    results = []

    print("\n==================================================")
    print("STARTING DECAY PARAMETERS GRID SEARCH")
    print("==================================================")

    for days in decay_days_grid:
        for rate in decay_rate_grid:
            print(f"\nEvaluating decay_days={days}, decay_rate={rate}...")
            
            t0 = time.time()
            
            # Compute Elo ratings
            elo_engine = EloEngine(
                use_dynamic_k=True,
                use_margin=True,
                use_decay=True,
                decay_days=days,
                decay_rate=rate
            )
            elo_df = elo_engine.run(meta_df, force=True)
            
            # Merge Elo ratings into states (vectorized and fast)
            states_copy = base_states.drop(columns=["batting_team_elo", "bowling_team_elo", "elo_diff"], errors="ignore")
            states_copy = states_copy.merge(elo_df, on="match_id", how="left")
            states_copy["team_1_elo"] = states_copy["team_1_elo"].fillna(1500.0)
            states_copy["team_2_elo"] = states_copy["team_2_elo"].fillna(1500.0)

            states_copy["batting_team_elo"] = np.where(
                states_copy["batting_team"] == states_copy["team_1"],
                states_copy["team_1_elo"],
                states_copy["team_2_elo"]
            )
            states_copy["bowling_team_elo"] = np.where(
                states_copy["batting_team"] == states_copy["team_1"],
                states_copy["team_2_elo"],
                states_copy["team_1_elo"]
            )
            states_copy["elo_diff"] = states_copy["batting_team_elo"] - states_copy["bowling_team_elo"]
            states_copy = states_copy.drop(columns=["team_1", "team_2", "team_1_elo", "team_2_elo"], errors="ignore")

            # Merge into validation sample
            val_sample_copy = val_sample.drop(columns=["batting_team_elo", "bowling_team_elo", "elo_diff"], errors="ignore")
            val_sample_copy = val_sample_copy.merge(elo_df, on="match_id", how="left")
            val_sample_copy["team_1_elo"] = val_sample_copy["team_1_elo"].fillna(1500.0)
            val_sample_copy["team_2_elo"] = val_sample_copy["team_2_elo"].fillna(1500.0)

            val_sample_copy["batting_team_elo"] = np.where(
                val_sample_copy["batting_team"] == val_sample_copy["team_1"],
                val_sample_copy["team_1_elo"],
                val_sample_copy["team_2_elo"]
            )
            val_sample_copy["bowling_team_elo"] = np.where(
                val_sample_copy["batting_team"] == val_sample_copy["team_1"],
                val_sample_copy["team_2_elo"],
                val_sample_copy["team_1_elo"]
            )
            val_sample_copy["elo_diff"] = val_sample_copy["batting_team_elo"] - val_sample_copy["bowling_team_elo"]
            val_sample_copy = val_sample_copy.drop(columns=["team_1", "team_2", "team_1_elo", "team_2_elo"], errors="ignore")

            # Train XGBoost
            model = XGBWinProbabilityModel()
            model.fit(states_copy, cache_path=None) # No caching during grid search

            # Predict and evaluate Brier
            errors = []
            for _, row in val_sample_copy.iterrows():
                out = model.predict(row)
                if out["status"] == "success":
                    errors.append((out["win_probability"] - row["won_match"]) ** 2)

            brier = np.mean(errors)
            duration = time.time() - t0
            
            print(f"--> Brier Score: {brier:.6f} (took {duration:.1f}s)")
            results.append({
                "decay_days": days,
                "decay_rate": rate,
                "brier_score": brier,
                "duration_seconds": duration
            })

    # Output report
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="brier_score").reset_index(drop=True)
    
    print("\n==================================================")
    print("GRID SEARCH COMPLETE")
    print("==================================================")
    print(df_results.to_string(index=False))
    
    best = df_results.iloc[0]
    print("\n" + "="*50)
    print(f"Optimal Parameters Found:")
    print(f"  decay_days : {int(best['decay_days'])}")
    print(f"  decay_rate : {best['decay_rate']:.2f}")
    print(f"  Brier Score: {best['brier_score']:.6f}")
    print("="*50)

if __name__ == "__main__":
    main()
