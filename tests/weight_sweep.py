import pandas as pd
import numpy as np
import sys
import time
from pathlib import Path
from sklearn.model_selection import KFold

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.load_data import CricketDataLoader
from src.data_processing.build_states import StateBuilder

def precompute_differences(sample_df, states_df, k=100):
    """
    Precomputes feature differences for validation sample against all matches
    to speed up grid search.
    """
    print("Precomputing differences for 500 sample states...")
    precomputed = []
    
    for idx, (_, row) in enumerate(sample_df.iterrows()):
        if idx % 100 == 0:
            print(f"  Precomputed {idx}/{len(sample_df)}...")
            
        # Prevent data leakage
        history = states_df[states_df["match_id"] != row["match_id"]].copy()
        
        # Phase-aware filtering
        same_phase = history[history["phase"] == row["phase"]]
        if len(same_phase) >= k * 2:
            history = same_phase.copy()
            
        # Feature standard deviations
        sigma_rr = history["runs_remaining"].std()
        sigma_br = history["balls_remaining"].std()
        sigma_wk = history["wickets_lost"].std()
        
        sigma_rr = sigma_rr if (pd.notna(sigma_rr) and sigma_rr > 0) else 38.0
        sigma_br = sigma_br if (pd.notna(sigma_br) and sigma_br > 0) else 34.0
        sigma_wk = sigma_wk if (pd.notna(sigma_wk) and sigma_wk > 0) else 2.5
        
        rr_diff = ((history["runs_remaining"] - row["runs_remaining"]) / sigma_rr).values ** 2
        br_diff = ((history["balls_remaining"] - row["balls_remaining"]) / sigma_br).values ** 2
        wk_diff = ((history["wickets_lost"] - row["wickets_lost"]) / sigma_wk).values ** 2
        
        # Pressure index
        df_pressure = history["required_run_rate"] * (1 + history["wickets_lost"] / 10.0)
        state_pressure = row["required_run_rate"] * (1 + row["wickets_lost"] / 10.0)
        
        sigma_p = df_pressure.std()
        sigma_p = sigma_p if (pd.notna(sigma_p) and sigma_p > 0) else 5.0
        
        p_diff = ((df_pressure - state_pressure) / sigma_p).values ** 2
        
        precomputed.append({
            "rr_diff": rr_diff,
            "br_diff": br_diff,
            "wk_diff": wk_diff,
            "p_diff": p_diff,
            "won_match": history["won_match"].values,
            "actual": row["won_match"]
        })
        
    return precomputed

def evaluate_weights(precomputed_list, indices, w_r, w_b, w_w, w_p, k=100):
    """
    Evaluates Brier score for a given set of weights over specified indices.
    """
    errors = []
    for idx in indices:
        item = precomputed_list[idx]
        
        # Weighted distance calculation
        distance = (
            w_r * item["rr_diff"] +
            w_b * item["br_diff"] +
            w_w * item["wk_diff"] +
            w_p * item["p_diff"]
        )
        
        # Find nearest neighbours
        part_idx = np.argpartition(distance, k)[:k]
        sorted_part_idx = part_idx[np.argsort(distance[part_idx])]
        
        k_won = item["won_match"][sorted_part_idx]
        k_dist = distance[sorted_part_idx]
        
        # Gaussian kernel weights
        weights = np.exp(-k_dist)
        if weights.sum() == 0:
            probability = k_won.mean()
        else:
            probability = np.average(k_won, weights=weights)
            
        probability = round(float(probability), 4)
        errors.append((probability - item["actual"]) ** 2)
        
    return np.mean(errors)

def run_grid_search(precomputed_list, train_indices, w_r_grid, w_b_grid, w_w_grid, w_p_grid):
    """
    Finds the weights that minimize the Brier score on the training indices.
    """
    best_brier = 1.0
    best_weights = None
    
    for w_r in w_r_grid:
        for w_b in w_b_grid:
            for w_w in w_w_grid:
                for w_p in w_p_grid:
                    brier = evaluate_weights(precomputed_list, train_indices, w_r, w_b, w_w, w_p)
                    if brier < best_brier:
                        best_brier = brier
                        best_weights = (w_r, w_b, w_w, w_p)
                        
    return best_weights, best_brier

if __name__ == "__main__":
    print("Loading datasets...")
    loader = CricketDataLoader(str(PROJECT_ROOT / "data/raw"))
    deliveries = loader.load_all()

    print("Building states...")
    states = StateBuilder(deliveries).build()
    states = states[states["innings"] == 2].copy()

    # Draw validation sample (500 states)
    sample = states.sample(500, random_state=42).reset_index(drop=True)
    
    precomputed = precompute_differences(sample, states)
    
    # Define grids
    # Stage 1: Coarse Grid
    coarse_grid = [0.5, 1.0, 2.0, 3.0]
    
    print("\n==========================================")
    print("STAGE 1: COARSE GRID SEARCH")
    print("==========================================")
    
    # 5-fold CV to check generalized performance
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_best_weights = []
    fold_test_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(precomputed)):
        print(f"Processing Fold {fold+1}/5...")
        best_w, best_b = run_grid_search(precomputed, train_idx, coarse_grid, coarse_grid, coarse_grid, coarse_grid)
        test_score = evaluate_weights(precomputed, test_idx, *best_w)
        fold_best_weights.append(best_w)
        fold_test_scores.append(test_score)
        print(f"  Fold {fold+1} Best Weights (r, b, w, p): {best_w} | Train Brier: {best_b:.5f} | Test Brier: {test_score:.5f}")
        
    print(f"\nAverage 5-Fold Cross-Validated Brier (Coarse): {np.mean(fold_test_scores):.5f}")
    
    # Run coarse search on all data to find best starting point for Stage 2
    print("\nRunning coarse search on all 500 validation states...")
    global_best_w, global_best_b = run_grid_search(precomputed, range(500), coarse_grid, coarse_grid, coarse_grid, coarse_grid)
    print(f"Global Coarse Best Weights: {global_best_w} with Brier: {global_best_b:.5f}")
    
    # Stage 2: Fine Grid Search (+/- 0.3 with steps of 0.1 around Stage 1 optimal weights)
    print("\n==========================================")
    print("STAGE 2: FINE GRID SEARCH")
    print("==========================================")
    
    def generate_fine_grid(center):
        return [round(center + offset, 1) for offset in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3] if center + offset > 0]
        
    fine_r_grid = generate_fine_grid(global_best_w[0])
    fine_b_grid = generate_fine_grid(global_best_w[1])
    fine_w_grid = generate_fine_grid(global_best_w[2])
    fine_p_grid = generate_fine_grid(global_best_w[3])
    
    print(f"Fine grid w_r: {fine_r_grid}")
    print(f"Fine grid w_b: {fine_b_grid}")
    print(f"Fine grid w_w: {fine_w_grid}")
    print(f"Fine grid w_p: {fine_p_grid}")
    
    # 5-fold CV to evaluate Stage 2
    fold_test_scores_fine = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(precomputed)):
        print(f"Processing Fold {fold+1}/5 (Fine Search)...")
        best_w_fine, best_b_fine = run_grid_search(
            precomputed, train_idx, fine_r_grid, fine_b_grid, fine_w_grid, fine_p_grid
        )
        test_score_fine = evaluate_weights(precomputed, test_idx, *best_w_fine)
        fold_test_scores_fine.append(test_score_fine)
        print(f"  Fold {fold+1} Best Weights (r, b, w, p): {best_w_fine} | Train Brier: {best_b_fine:.5f} | Test Brier: {test_score_fine:.5f}")
        
    print(f"\nAverage 5-Fold Cross-Validated Brier (Fine): {np.mean(fold_test_scores_fine):.5f}")
    
    # Run fine search on all data to get the absolute best weights
    print("\nRunning fine search on all 500 validation states...")
    final_best_w, final_best_b = run_grid_search(
        precomputed, range(500), fine_r_grid, fine_b_grid, fine_w_grid, fine_p_grid
    )
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Final Optimized Weights:")
    print(f"  w_runs    (w_r) : {final_best_w[0]:.1f}")
    print(f"  w_balls   (w_b) : {final_best_w[1]:.1f}")
    print(f"  w_wickets (w_w) : {final_best_w[2]:.1f}")
    print(f"  w_press   (w_p) : {final_best_w[3]:.1f}")
    print(f"Final Optimized Brier Score: {final_best_b:.5f}")
    print("="*50)
