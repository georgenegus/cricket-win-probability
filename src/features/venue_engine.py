import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.cache_manager import CacheManager
from src.features.venue_aliases import canonicalize_for_aggregation


class VenueEngine:
    """
    Computes rolling venue-level features prior to each match date:
    - venue_avg_first_innings_score (rolling mean of first innings totals at venue)
    - venue_chase_success_rate (rolling win rate of team batting 2nd at venue)
    Uses empirical global priors for Bayesian shrinkage on venues with few historical matches.
    """

    def __init__(
        self,
        smoothing_runs=None,
        smoothing_chase=None,
        default_avg_score=None,
        processed_path="data/processed",
        format_type="t20"
    ):
        self.processed_path = Path(processed_path)
        self.format_type = format_type
        self.cache_file = self.processed_path / f"{format_type}_venue_features.parquet"
        self.processed_path.mkdir(parents=True, exist_ok=True)

        from src.data_processing.format_spec import spec
        fmt_spec = spec(format_type)

        if format_type in ("od", "odw"):
            self.smoothing_runs = float(smoothing_runs if smoothing_runs is not None else 10.0)
            self.smoothing_chase = float(smoothing_chase if smoothing_chase is not None else 25.0)
            self.default_avg_score = float(default_avg_score if default_avg_score is not None else fmt_spec["default_venue_score"])
        else:
            self.smoothing_runs = float(smoothing_runs if smoothing_runs is not None else 10.0)
            self.smoothing_chase = float(smoothing_chase if smoothing_chase is not None else 10.0)
            self.default_avg_score = float(default_avg_score if default_avg_score is not None else fmt_spec["default_venue_score"])

    def normalize_venue(self, venue):
        if not isinstance(venue, str):
            return ""
        # Route through the shared alias canonicalization (Phase 11) so that
        # raw spelling/punctuation variants of the same ground (e.g.
        # "M.Chinnaswamy Stadium" vs "M Chinnaswamy Stadium, Bengaluru")
        # aggregate into one bucket instead of silently splitting a venue's
        # historical stats across several rows.
        return canonicalize_for_aggregation(venue)

    def run(self, metadata_df: pd.DataFrame, deliveries_df: pd.DataFrame, force=False) -> pd.DataFrame:
        if not force and CacheManager.is_cache_valid(self.cache_file, [Path(__file__)]):
            print(f"Loading venue features from cache: {self.cache_file}")
            return pd.read_parquet(self.cache_file)


        print("Computing rolling venue-level features...")

        # 1. Compute first-innings score for each match from deliveries
        # Sum total_runs in innings == 1
        first_innings = deliveries_df[deliveries_df["innings"] == 1]
        first_innings_scores = (
            first_innings.groupby("match_id")["total_runs"]
            .sum()
            .reset_index()
        )
        first_innings_scores.columns = ["match_id", "first_innings_score"]

        # Merge first-innings scores into match metadata
        df = metadata_df.merge(first_innings_scores, on="match_id", how="left")

        # Sort chronologically by date and match_id to prevent data leakage
        df = df.sort_values(by=["date", "match_id"]).reset_index(drop=True)

        venue_records = []

        # Internal state for running stats per normalized venue
        # venue -> {sum_runs, count_runs, success_chase, count_chase}
        venue_stats = {}

        # Global running averages for smoothing/shrinkage
        global_sum_runs = 0.0
        global_count_runs = 0
        global_success_chase = 0.0
        global_count_chase = 0

        for _, row in df.iterrows():
            match_id = row["match_id"]
            venue = row.get("venue", "")
            norm_venue = self.normalize_venue(venue)

            first_innings_score = row.get("first_innings_score", np.nan)
            winner = row.get("winner", None)
            team_2 = row.get("team_2", None)  # team batting second (chasing)
            win_type = row.get("win_type", None)

            # Get current running global averages
            global_avg_runs = (
                global_sum_runs / global_count_runs
                if global_count_runs > 0
                else self.default_avg_score
            )
            global_avg_chase = (
                global_success_chase / global_count_chase
                if global_count_chase > 0
                else 0.45
            )

            # Get current venue statistics
            v_stats = venue_stats.setdefault(
                norm_venue,
                {
                    "sum_runs": 0.0,
                    "count_runs": 0,
                    "success_chase": 0.0,
                    "count_chase": 0
                }
            )

            # Calculate smoothed rolling features using Bayesian shrinkage
            if v_stats["count_runs"] > 0:
                avg_runs = (
                    v_stats["sum_runs"] +
                    self.smoothing_runs * global_avg_runs
                ) / (v_stats["count_runs"] + self.smoothing_runs)
            else:
                avg_runs = global_avg_runs

            if v_stats["count_chase"] > 0:
                chase_rate = (
                    v_stats["success_chase"] +
                    self.smoothing_chase * global_avg_chase
                ) / (v_stats["count_chase"] + self.smoothing_chase)
            else:
                chase_rate = global_avg_chase

            # Record pre-match features
            venue_records.append({
                "match_id": match_id,
                "venue_avg_first_innings_score": avg_runs,
                "venue_chase_success_rate": chase_rate,
                "venue_match_count": v_stats["count_runs"]
            })

            # Update statistics with the outcome of this match (if valid)
            if pd.notna(first_innings_score) and win_type != "no_result":
                # Update venue stats
                v_stats["sum_runs"] += first_innings_score
                v_stats["count_runs"] += 1

                # Update global stats
                global_sum_runs += first_innings_score
                global_count_runs += 1

                # Update chase records if there is a clear winner
                if (
                    winner and
                    isinstance(winner, str) and
                    isinstance(team_2, str)
                ):
                    chase_won = 1.0 if winner == team_2 else 0.0
                    v_stats["success_chase"] += chase_won
                    v_stats["count_chase"] += 1

                    global_success_chase += chase_won
                    global_count_chase += 1

        venue_df = pd.DataFrame(venue_records)

        # Save match_id features to cache
        print(f"Saving venue features to cache: {self.cache_file}")
        venue_df.to_parquet(self.cache_file, index=False, compression="snappy")
        CacheManager.register_cache(self.cache_file, [Path(__file__)])

        # Build live lookup table keyed by normalized venue
        venue_lookup_records = []
        for norm_v, s in venue_stats.items():
            if not norm_v or norm_v == "":
                continue
            count_runs = s["count_runs"]
            if count_runs > 0:
                global_avg_r = (global_sum_runs / global_count_runs) if global_count_runs > 0 else self.default_avg_score
                global_avg_c = (global_success_chase / global_count_chase) if global_count_chase > 0 else 0.45

                avg_runs = (s["sum_runs"] + self.smoothing_runs * global_avg_r) / (count_runs + self.smoothing_runs)
                chase_rate = (s["success_chase"] + self.smoothing_chase * global_avg_c) / (s["count_chase"] + self.smoothing_chase) if s["count_chase"] > 0 else global_avg_c
            else:
                avg_runs = self.default_avg_score
                chase_rate = 0.45

            venue_lookup_records.append({
                "venue": norm_v,
                "venue_avg_first_innings_score": round(float(avg_runs), 2),
                "venue_chase_success_rate": round(float(chase_rate), 4),
                "venue_match_count": int(count_runs),
            })

        venue_lookup_df = pd.DataFrame(venue_lookup_records)
        lookup_file = self.processed_path / f"{self.format_type}_venue_lookup.parquet"
        print(f"Saving venue lookup table to: {lookup_file}")
        venue_lookup_df.to_parquet(lookup_file, index=False, compression="snappy")
        CacheManager.register_cache(lookup_file, [Path(__file__)])

        return venue_df

    def clear_cache(self):
        CacheManager.clear_cache(self.cache_file)

