import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter
from src.data_processing.load_data import CricketDataLoader
from src.data_processing.match_metadata import MatchMetadataExtractor

def build_empirical_dls(format_type="t20", data_dir="data"):
    print(f"\nBuilding Empirical DLS lookup table for format: {format_type}...")
    
    # 1. Load deliveries and match metadata
    loader = CricketDataLoader(f"{data_dir}/raw", format_type=format_type)
    deliveries = loader.load_all()
    print(f"Loaded {len(deliveries):,} total deliveries.")
    
    meta_extractor = MatchMetadataExtractor(f"{data_dir}/raw", format_type=format_type)
    meta_df = meta_extractor.extract_all()
    print(f"Loaded {len(meta_df):,} matches metadata.")
    
    # Filter to this format's own gender (was hardcoded to "male", which would zero out an
    # all-female corpus like t20w).
    if "gender" in meta_df.columns:
        from src.data_processing.format_spec import spec
        meta_df = meta_df[meta_df["gender"] == spec(format_type)["gender"]]
        print(f"Filtered for {spec(format_type)['gender']} matches: {len(meta_df):,} matches.")
    
    # Filter out rain-affected/DLS matches to ensure clean historical resource curves
    if "method" in meta_df.columns:
        meta_df = meta_df[
            meta_df["method"].isna() | 
            (meta_df["method"] == "") | 
            (meta_df["method"] == "None") |
            (meta_df["method"] == "none")
        ]
        print(f"Filtered out shortened/DLS matches: {len(meta_df):,} matches.")
        
    deliveries = deliveries[deliveries["match_id"].isin(meta_df["match_id"])]
    print(f"Deliveries in clean matches: {len(deliveries):,}")
    
    # 2. Process first innings data
    df = deliveries[deliveries["innings"] == 1].copy()
    
    # Sort chronologically to compute running totals
    df = df.sort_values(["match_id", "over", "ball"])
    
    # Calculate cumulative runs and wickets
    df["runs_scored"] = df.groupby("match_id")["total_runs"].cumsum()
    df["wickets_lost"] = df.groupby("match_id")["wicket"].cumsum()
    
    # Legal deliveries and balls faced
    df["legal_delivery"] = ((df["is_wide"] == 0) & (df["is_noball"] == 0)).astype(int)
    df["balls_faced"] = df.groupby("match_id")["legal_delivery"].cumsum()
    
    total_balls = 120 if format_type == "t20" else 300
    df["balls_remaining"] = total_balls - df["balls_faced"]
    
    # Only keep completed matches where the first innings reached the target length or was bowled out
    match_status = df.groupby("match_id").agg(
        max_balls=("balls_faced", "max"),
        max_wickets=("wickets_lost", "max"),
        final_score=("runs_scored", "max")
    )
    
    valid_matches = match_status[
        (match_status["max_wickets"] == 10) | (match_status["max_balls"] >= total_balls)
    ].index
    
    df = df[df["match_id"].isin(valid_matches)]
    print(f"Filtered to first innings of completed matches: {len(valid_matches):,} matches, {len(df):,} deliveries.")
    
    # Map final score back to deliveries
    final_scores = match_status.loc[valid_matches, "final_score"]
    df["final_score"] = df["match_id"].map(final_scores)
    
    # Expected runs remaining to be scored in this innings from the current ball's post-delivery state
    df["runs_to_end"] = df["final_score"] - df["runs_scored"]
    
    # Group by (balls_remaining, wickets_lost)
    grouped = df.groupby(["balls_remaining", "wickets_lost"]).agg(
        runs_mean=("runs_to_end", "mean"),
        count=("runs_to_end", "count")
    ).reset_index()
    
    # 3. Create empty lookup matrix (rows = balls_remaining, cols = wickets_lost)
    matrix = np.zeros((total_balls + 1, 11))
    counts = np.zeros((total_balls + 1, 11))
    
    for _, row in grouped.iterrows():
        b = int(row["balls_remaining"])
        w = int(row["wickets_lost"])
        if 0 <= b <= total_balls and 0 <= w <= 10:
            matrix[b, w] = row["runs_mean"]
            counts[b, w] = row["count"]
            
    # Baseline for credibility fill (to handle sparse/unobserved states safely)
    avg_score = 160.0 if format_type == "t20" else 250.0
    runs_per_ball = avg_score / total_balls
    
    # Blend historical mean with baseline using credibility weighting
    C = 15.0  # credibility constant (weight equivalent to 15 deliveries)
    for b in range(total_balls + 1):
        for w in range(11):
            baseline = runs_per_ball * b * (10.0 - w) / 10.0
            historical_mean = matrix[b, w]
            count = counts[b, w]
            
            # Blend
            matrix[b, w] = (historical_mean * count + baseline * C) / (count + C)
            
    # 4. Smooth with 2D Gaussian filter to ensure smooth transitions
    matrix = gaussian_filter(matrix, sigma=1.5)
    
    # 5. Enforce physical boundaries
    matrix[0, :] = 0.0  # 0 balls remaining -> 0 runs remaining
    matrix[:, 10] = 0.0  # 10 wickets lost -> 0 runs remaining
    
    # 6. Enforce monotonicity constraints
    # A. Fewer wickets lost must lead to equal or higher runs remaining
    for b in range(total_balls + 1):
        for w in range(1, 11):
            if matrix[b, w] > matrix[b, w - 1]:
                matrix[b, w] = matrix[b, w - 1]
                
    # B. More balls remaining must lead to equal or higher runs remaining
    for w in range(11):
        for b in range(1, total_balls + 1):
            if matrix[b, w] < matrix[b - 1, w]:
                matrix[b, w] = matrix[b - 1, w]
                
    # Flatten matrix to dataframe for easy parquet storage
    records = []
    for b in range(total_balls + 1):
        for w in range(11):
            records.append({
                "balls_remaining": b,
                "wickets_lost": w,
                "expected_runs_remaining": matrix[b, w]
            })
            
    out_df = pd.DataFrame(records)
    out_path = Path(f"{data_dir}/processed/empirical_dls_{format_type}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path)
    print(f"Saved empirical DLS lookup table for {format_type} to {out_path}")

if __name__ == "__main__":
    build_empirical_dls("t20")
    build_empirical_dls("od")
