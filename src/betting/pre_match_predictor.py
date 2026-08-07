import os
import pandas as pd
import numpy as np
from pathlib import Path
from src.models.xgb_win_probability import XGBWinProbabilityModel
from src.models.first_innings_predictor import FirstInningsPredictor
from src.features.venue_aliases import resolve_venue_alias, clean_venue_string
from src.features.venue_review_queue import queue_ambiguous_venue


class PreMatchPredictor:
    """
    Computes pre-match win probabilities for upcoming cricket fixtures
    using the trained XGBoost win probability models and Elo ratings.
    """

    def __init__(self, processed_dir="data/processed"):
        self.processed_dir = Path(processed_dir)
        from src.data_processing.format_spec import spec as _spec

        # Load T20, ODI, HND & T20W win probability models
        print("Loading PreMatchPredictor models...")
        self.t20_model = XGBWinProbabilityModel(format_type="t20")
        self.t20_model.fit(None, cache_path=self.processed_dir / "t20_xgb_models.pkl")

        self.od_model = XGBWinProbabilityModel(format_type="od")
        self.od_model.fit(None, cache_path=self.processed_dir / "od_xgb_models.pkl")

        self.hnd_model = XGBWinProbabilityModel(format_type="hnd")
        self.hnd_model.fit(None, cache_path=self.processed_dir / "hnd_xgb_models.pkl")

        self.t20w_model = XGBWinProbabilityModel(format_type="t20w")
        self.t20w_model.fit(None, cache_path=self.processed_dir / "t20w_xgb_models.pkl")

        self.odw_model = XGBWinProbabilityModel(format_type="odw")
        self.odw_model.fit(None, cache_path=self.processed_dir / "odw_xgb_models.pkl")

        self.hndw_model = XGBWinProbabilityModel(format_type="hndw")
        self.hndw_model.fit(None, cache_path=self.processed_dir / "hndw_xgb_models.pkl")

        # Load first innings target regressors
        self.t20_first_innings = FirstInningsPredictor(format_type="t20")
        self.t20_first_innings.fit(None, cache_path=self.processed_dir / "t20_first_innings_regressor.pkl")

        self.od_first_innings = FirstInningsPredictor(format_type="od")
        self.od_first_innings.fit(None, cache_path=self.processed_dir / "od_first_innings_regressor.pkl")

        self.hnd_first_innings = FirstInningsPredictor(format_type="hnd")
        self.hnd_first_innings.fit(None, cache_path=self.processed_dir / "hnd_first_innings_regressor.pkl")

        self.t20w_first_innings = FirstInningsPredictor(format_type="t20w")
        self.t20w_first_innings.fit(None, cache_path=self.processed_dir / "t20w_first_innings_regressor.pkl")

        self.odw_first_innings = FirstInningsPredictor(format_type="odw")
        self.odw_first_innings.fit(None, cache_path=self.processed_dir / "odw_first_innings_regressor.pkl")

        self.hndw_first_innings = FirstInningsPredictor(format_type="hndw")
        self.hndw_first_innings.fit(None, cache_path=self.processed_dir / "hndw_first_innings_regressor.pkl")

        # Load latest team Elo ratings
        self.t20_elos = self._extract_latest_team_elos(self.processed_dir / "t20_elo_ratings.parquet")
        self.od_elos = self._extract_latest_team_elos(self.processed_dir / "od_elo_ratings.parquet")
        self.hnd_elos = self._extract_latest_team_elos(self.processed_dir / "hnd_elo_ratings.parquet")
        self.t20w_elos = self._extract_latest_team_elos(self.processed_dir / "t20w_elo_ratings.parquet")
        self.odw_elos = self._extract_latest_team_elos(self.processed_dir / "odw_elo_ratings.parquet")
        self.hndw_elos = self._extract_latest_team_elos(self.processed_dir / "hndw_elo_ratings.parquet")

        # Load venue averages from lookup parquet (falling back to features parquet)
        self.t20_venues = self._load_venue_averages(self.processed_dir / "t20_venue_lookup.parquet", default_score=160.0)
        self.od_venues = self._load_venue_averages(self.processed_dir / "od_venue_lookup.parquet", default_score=260.0)
        self.hnd_venues = self._load_venue_averages(self.processed_dir / "hnd_venue_lookup.parquet", default_score=140.0)
        self.t20w_venues = self._load_venue_averages(self.processed_dir / "t20w_venue_lookup.parquet", default_score=_spec("t20w")["default_venue_score"])
        self.odw_venues = self._load_venue_averages(self.processed_dir / "odw_venue_lookup.parquet", default_score=_spec("odw")["default_venue_score"])
        self.hndw_venues = self._load_venue_averages(self.processed_dir / "hndw_venue_lookup.parquet", default_score=_spec("hndw")["default_venue_score"])

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
        # Fallback to feature file if lookup file doesn't exist yet
        if not parquet_path.exists():
            fallback_path = parquet_path.parent / parquet_path.name.replace("_lookup.", "_features.")
            if fallback_path.exists():
                parquet_path = fallback_path
            else:
                return {}
        try:
            df = pd.read_parquet(parquet_path)
            if "venue" in df.columns and "venue_avg_first_innings_score" in df.columns:
                return df.set_index("venue")["venue_avg_first_innings_score"].to_dict()
            return {}
        except Exception as e:
            print(f"Error loading venue averages from {parquet_path}: {e}")
            return {}

    def _normalize_fmt(self, format_type):
        """Maps a format string/alias to one of this class's own loaded formats
        (t20/od/hnd/t20w). Any other recognised FORMAT_SPECS code (future odw/hndw)
        passes through unchanged instead of silently defaulting to "t20" -- same fix
        as match_simulator.extract_match_squads (womens_games_ingestion_plan.md Phase 5)."""
        from src.data_processing.format_spec import FORMAT_SPECS
        fmt_clean = str(format_type).lower().strip()
        if fmt_clean in ["hnd", "hundred", "100"]:
            return "hnd"
        if fmt_clean in ["od", "odi", "one-day"]:
            return "od"
        if fmt_clean in ["t20w", "women's t20", "womens t20"]:
            return "t20w"
        if fmt_clean in FORMAT_SPECS:
            return fmt_clean
        return "t20"

    def get_team_elo(self, team_name, format_type="t20"):
        """Returns latest Elo rating for a team or default 1500.0 if unknown."""
        fmt = self._normalize_fmt(format_type)
        elos = {
            "hnd": self.hnd_elos, "od": self.od_elos, "t20w": self.t20w_elos,
            "odw": self.odw_elos, "hndw": self.hndw_elos,
        }.get(fmt, self.t20_elos)
        return float(elos.get(team_name, 1500.0))

    def get_venue_avg_score(self, venue_name, format_type="t20"):
        """Returns historical venue average first innings score."""
        from src.data_processing.format_spec import spec
        fmt = self._normalize_fmt(format_type)

        default_val = float(spec(fmt)["default_venue_score"])
        if not venue_name:
            return default_val

        cleaned = clean_venue_string(venue_name)
        if not cleaned:
            return default_val

        venues = {
            "hnd": self.hnd_venues, "od": self.od_venues, "t20w": self.t20w_venues,
            "odw": self.odw_venues, "hndw": self.hndw_venues,
        }.get(fmt, self.t20_venues)

        resolved_key = resolve_venue_alias(venue_name, venues)
        if resolved_key and resolved_key in venues:
            return float(venues[resolved_key])

        queue_ambiguous_venue(
            raw_identifier=venue_name,
            format_type=fmt,
            reason="unresolved_venue_string",
            context={"source": "PreMatchPredictor.get_venue_avg_score"},
        )
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
        from src.data_processing.format_spec import spec
        fmt = self._normalize_fmt(format_type)

        fmt_spec = spec(fmt)
        total_overs = float(fmt_spec["total_overs"])
        balls_per_over = float(fmt_spec["balls_per_over"])
        total_balls = float(fmt_spec["total_balls"])
        default_target = float(fmt_spec["default_venue_score"])

        elo_a = self.get_team_elo(team_a, format_type=fmt)
        elo_b = self.get_team_elo(team_b, format_type=fmt)
        venue_avg = self.get_venue_avg_score(venue, format_type=fmt) if venue else default_target

        target = custom_target if custom_target is not None else venue_avg

        model = {
            "hnd": self.hnd_model, "od": self.od_model, "t20w": self.t20w_model,
            "odw": self.odw_model, "hndw": self.hndw_model,
        }.get(fmt, self.t20_model)

        # Scenario 1: Team B bats 1st, Team A chases
        state_a_chasing = pd.Series(
            {
                "runs_remaining": target,
                "balls_remaining": total_balls,
                "wickets_lost": 0.0,
                "required_run_rate": target / (total_balls / balls_per_over),
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
                "competition": "hnd" if fmt == "hnd" else ("odis" if fmt == "od" else "t20s"),
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
                "required_run_rate": target / (total_balls / balls_per_over),
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
                "competition": "hnd" if fmt == "hnd" else ("odis" if fmt == "od" else "t20s"),
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
