import os
import pandas as pd
import numpy as np
from pathlib import Path
from src.models.xgb_win_probability import XGBWinProbabilityModel
from src.models.first_innings_predictor import FirstInningsPredictor


class PreMatchPredictor:
    """
    Computes pre-match win probabilities for upcoming cricket fixtures
    using the trained XGBoost win probability models and Elo ratings.
    """

    def __init__(self, processed_dir="data/processed"):
        self.processed_dir = Path(processed_dir)

        # Load T20 & ODI win probability models
        print("Loading PreMatchPredictor models...")
        self.t20_model = XGBWinProbabilityModel(format_type="t20")
        self.t20_model.fit(None, cache_path=self.processed_dir / "t20_xgb_models.pkl")

        self.od_model = XGBWinProbabilityModel(format_type="od")
        self.od_model.fit(None, cache_path=self.processed_dir / "od_xgb_models.pkl")

        # Load First Innings Predictors
        self.t20_first_innings = FirstInningsPredictor(format_type="t20")
        self.t20_first_innings.fit(None, cache_path=self.processed_dir / "t20_first_innings_regressor.pkl")

        self.od_first_innings = FirstInningsPredictor(format_type="od")
        self.od_first_innings.fit(None, cache_path=self.processed_dir / "od_first_innings_regressor.pkl")

        # Load latest team Elo ratings
        self.t20_elos = self._extract_latest_team_elos(self.processed_dir / "t20_elo_ratings.parquet")
        self.od_elos = self._extract_latest_team_elos(self.processed_dir / "od_elo_ratings.parquet")

        # Load venue averages
        self.t20_venues = self._load_venue_averages(self.processed_dir / "t20_venue_features.parquet", default_score=160.0)
        self.od_venues = self._load_venue_averages(self.processed_dir / "od_venue_features.parquet", default_score=260.0)

    def _extract_latest_team_elos(self, parquet_path):
        if not parquet_path.exists():
            return {}
        try:
            df = pd.read_parquet(parquet_path)
            team_elos = {}
            for _, row in df.iterrows():
                if "team_1" in row and "team_1_elo" in row:
                    team_elos[str(row["team_1"])] = float(row["team_1_elo"])
                if "team_2" in row and "team_2_elo" in row:
                    team_elos[str(row["team_2"])] = float(row["team_2_elo"])
            return team_elos
        except Exception as e:
            print(f"Error loading Elo ratings from {parquet_path}: {e}")
            return {}

    def _load_venue_averages(self, parquet_path, default_score):
        if not parquet_path.exists():
            return {}
        try:
            df = pd.read_parquet(parquet_path)
            if "venue" in df.columns and "venue_avg_first_innings_score" in df.columns:
                return df.set_index("venue")["venue_avg_first_innings_score"].to_dict()
            return {}
        except Exception as e:
            return {}

    def get_team_elo(self, team_name, format_type="t20"):
        """Returns latest Elo rating for a team or default 1500.0 if unknown."""
        elos = self.od_elos if format_type in ["od", "odi"] else self.t20_elos
        return float(elos.get(team_name, 1500.0))

    def get_venue_avg_score(self, venue_name, format_type="t20"):
        """Returns historical venue average first innings score."""
        default_val = 260.0 if format_type in ["od", "odi"] else 160.0
        if not venue_name or str(venue_name).strip().upper() in ["N/A", "NONE", "UNKNOWN", ""]:
            return default_val

        venues = self.od_venues if format_type in ["od", "odi"] else self.t20_venues
        if venue_name in venues:
            return float(venues[venue_name])

        clean_target = str(venue_name).strip().lower()
        for v_name, score in venues.items():
            clean_v = str(v_name).strip().lower()
            if clean_target == clean_v or clean_target in clean_v or clean_v in clean_target:
                return float(score)

        return default_val

    def predict(self, team_a, team_b, venue=None, format_type="t20", custom_target=None):
        """
        Computes pre-match win probability for Team A vs Team B.

        Since the toss outcome is unknown pre-match, we compute:
          P(Team A wins) = 0.5 * P(A chases successfully | target) + 0.5 * (1 - P(B chases successfully | target))

        Args:
            team_a (str): Name of Team A
            team_b (str): Name of Team B
            venue (str, optional): Venue name
            format_type (str): "t20" or "od"
            custom_target (float, optional): Force a specific target score

        Returns: dict:
            {
                "team_a": str, "team_b": str,
                "prob_a": float, "prob_b": float,
                "ci_lower_a": float, "ci_upper_a": float,
                "ci_lower_b": float, "ci_upper_b": float,
                "std_dev": float,
                "elo_a": float, "elo_b": float,
                "venue_avg_score": float,
                "estimated_target": float,
                "format_type": str
            }
        """
        fmt = "od" if format_type.lower() in ["od", "odi", "one-day"] else "t20"
        total_overs = 50.0 if fmt == "od" else 20.0
        total_balls = total_overs * 6.0
        default_target = 260.0 if fmt == "od" else 160.0

        elo_a = self.get_team_elo(team_a, format_type=fmt)
        elo_b = self.get_team_elo(team_b, format_type=fmt)
        venue_avg = self.get_venue_avg_score(venue, format_type=fmt) if venue else default_target

        target = custom_target if custom_target is not None else venue_avg

        model = self.od_model if fmt == "od" else self.t20_model

        # Scenario 1: Team B bats 1st, Team A chases
        state_a_chasing = pd.Series(
            {
                "runs_remaining": target,
                "balls_remaining": total_balls,
                "wickets_lost": 0.0,
                "required_run_rate": target / total_overs,
                "run_rate": 0.0,
                "target": target,
                "phase": "powerplay",
                "match_id": 999999,
                "batting_team_elo": elo_a,
                "bowling_team_elo": elo_b,
                "elo_diff": elo_a - elo_b,
                "batter_elo": 1500.0,
                "bowler_elo": 1500.0,
                "player_elo_diff": 0.0,
                "venue_avg_first_innings_score": venue_avg,
                "competition": "odis" if fmt == "od" else "t20s",
                "total_overs": total_overs,
                "target_overs": total_overs,
                "innings": 2,
            }
        )

        # Scenario 2: Team A bats 1st, Team B chases
        state_b_chasing = pd.Series(
            {
                "runs_remaining": target,
                "balls_remaining": total_balls,
                "wickets_lost": 0.0,
                "required_run_rate": target / total_overs,
                "run_rate": 0.0,
                "target": target,
                "phase": "powerplay",
                "match_id": 999999,
                "batting_team_elo": elo_b,
                "bowling_team_elo": elo_a,
                "elo_diff": elo_b - elo_a,
                "batter_elo": 1500.0,
                "bowler_elo": 1500.0,
                "player_elo_diff": 0.0,
                "venue_avg_first_innings_score": venue_avg,
                "competition": "odis" if fmt == "od" else "t20s",
                "total_overs": total_overs,
                "target_overs": total_overs,
                "innings": 2,
            }
        )

        res_a = model.predict_with_ensemble(state_a_chasing)
        res_b = model.predict_with_ensemble(state_b_chasing)

        prob_a_chases = res_a["win_probability"]
        prob_b_chases = res_b["win_probability"]

        # Pre-match probability (un-tossed 50/50 average)
        prob_a = 0.5 * prob_a_chases + 0.5 * (1.0 - prob_b_chases)
        prob_b = 1.0 - prob_a

        # Ensemble uncertainty bounds
        std_dev = round(float((res_a["std_dev"] + res_b["std_dev"]) / 2.0), 4)
        ci_lower_a = max(0.0, round(prob_a - 1.96 * std_dev, 4))
        ci_upper_a = min(1.0, round(prob_a + 1.96 * std_dev, 4))

        ci_lower_b = max(0.0, round(prob_b - 1.96 * std_dev, 4))
        ci_upper_b = min(1.0, round(prob_b + 1.96 * std_dev, 4))

        return {
            "team_a": team_a,
            "team_b": team_b,
            "prob_a": round(float(prob_a), 4),
            "prob_b": round(float(prob_b), 4),
            "ci_lower_a": ci_lower_a,
            "ci_upper_a": ci_upper_a,
            "ci_lower_b": ci_lower_b,
            "ci_upper_b": ci_upper_b,
            "std_dev": std_dev,
            "elo_a": elo_a,
            "elo_b": elo_b,
            "venue_avg_score": venue_avg,
            "estimated_target": target,
            "format_type": fmt,
        }
