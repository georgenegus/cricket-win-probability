"""
Live State Builder Module

Translates incremental live delivery updates (DeliveryUpdate) into model-compatible
feature vectors (pd.Series) for in-play win probability prediction.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd

from src.betting.live_scorecard import DeliveryUpdate


class LiveStateBuilder:
    """
    Incremental state builder for live match tracking.
    Accumulates delivery updates and produces pd.Series state vectors formatted for
    XGBWinProbabilityModel.predict_with_ensemble(state).
    """

    def __init__(
        self,
        format_type: str = "t20",
        target: Optional[int] = None,
        batting_team_elo: float = 1500.0,
        bowling_team_elo: float = 1500.0,
        venue_avg_first_innings_score: Optional[float] = None,
        venue_chase_success_rate: float = 0.45,
        competition: str = "t20s",
        player_elo_dict: Optional[Dict[str, float]] = None,
    ):
        self.format_type = format_type.lower()
        self.total_overs = 50.0 if self.format_type == "od" else 20.0
        self.total_balls = self.total_overs * 6.0
        self.target = target

        # Team Elo
        self.batting_team_elo = float(batting_team_elo)
        self.bowling_team_elo = float(bowling_team_elo)
        self.elo_diff = self.batting_team_elo - self.bowling_team_elo

        # Venue parameters
        default_venue = 260.0 if self.format_type == "od" else 160.0
        self.venue_avg_first_innings_score = (
            venue_avg_first_innings_score if venue_avg_first_innings_score is not None else default_venue
        )
        self.venue_chase_success_rate = float(venue_chase_success_rate)
        self.competition = competition.lower()

        # State accumulators
        self.runs_scored = 0
        self.wickets_lost = 0
        self.legal_deliveries = 0
        self.deliveries_history = []

        # Current players
        self.current_striker: Optional[str] = None
        self.current_non_striker: Optional[str] = None
        self.current_bowler: Optional[str] = None

        # Player Elo dictionary mapping player_name -> effective_elo
        self.player_elos: Dict[str, float] = player_elo_dict or {}

    def update(self, delivery: DeliveryUpdate) -> None:
        """
        Processes a new delivery update and updates internal accumulators.

        Handles:
        - Runs scored (including extras)
        - Wickets lost
        - Legal delivery count (wides and no-balls do not count as legal)
        - Current striker, non-striker, bowler tracking
        """
        self.runs_scored += delivery.runs_this_ball

        if delivery.is_wicket:
            self.wickets_lost += 1

        if not delivery.is_wide and not delivery.is_noball:
            self.legal_deliveries += 1

        if delivery.striker:
            self.current_striker = delivery.striker
        if delivery.non_striker:
            self.current_non_striker = delivery.non_striker
        if delivery.bowler:
            self.current_bowler = delivery.bowler

        if delivery.target is not None:
            self.target = delivery.target

        self.deliveries_history.append(delivery)

    def get_phase(self, overs_completed: float) -> str:
        """Determines the current match phase (powerplay, middle, death)."""
        if self.total_overs > 20:
            powerplay_end = 10.0
            death_start = 40.0
        else:
            powerplay_end = self.total_overs * 0.3   # 6.0 for T20
            death_start = self.total_overs * 0.75    # 15.0 for T20

        if overs_completed < powerplay_end:
            return "powerplay"
        elif overs_completed >= death_start:
            return "death"
        else:
            return "middle"

    def get_state(self) -> pd.Series:
        """
        Returns a pd.Series state vector compatible with model.predict_with_ensemble(state).

        Matches feature expectations of XGBWinProbabilityModel._extract_features_single().
        """
        if self.target is None:
            raise ValueError("Cannot compute chase state without a target.")

        runs_remaining = max(float(self.target - self.runs_scored), 0.0)
        balls_remaining = max(float(self.total_balls - self.legal_deliveries), 0.0)
        overs_completed = self.legal_deliveries / 6.0

        if balls_remaining > 0:
            required_run_rate = float(runs_remaining / (balls_remaining / 6.0))
        else:
            required_run_rate = 0.0

        if self.legal_deliveries > 0:
            run_rate = float(self.runs_scored / overs_completed)
        else:
            run_rate = 0.0

        phase = self.get_phase(overs_completed)

        batter_elo = float(self.player_elos.get(self.current_striker, 1500.0))
        bowler_elo = float(self.player_elos.get(self.current_bowler, 1500.0))
        non_striker_elo = float(self.player_elos.get(self.current_non_striker, 1500.0))

        state = pd.Series({
            "runs_remaining": float(runs_remaining),
            "balls_remaining": float(balls_remaining),
            "wickets_lost": float(self.wickets_lost),
            "required_run_rate": float(required_run_rate),
            "run_rate": float(run_rate),
            "target": float(self.target),
            "phase": phase,
            "match_id": 999999,  # Sentinel ID to ensure final ensemble model usage
            "batting_team_elo": float(self.batting_team_elo),
            "bowling_team_elo": float(self.bowling_team_elo),
            "elo_diff": float(self.elo_diff),
            "batter_elo": float(batter_elo),
            "bowler_elo": float(bowler_elo),
            "non_striker_elo": float(non_striker_elo),
            "player_elo_diff": float(batter_elo - bowler_elo),
            "venue_avg_first_innings_score": float(self.venue_avg_first_innings_score),
            "venue_chase_success_rate": float(self.venue_chase_success_rate),
            "competition": self.competition,
            "total_overs": float(self.total_overs),
            "target_overs": float(self.total_overs),
            "innings": 2,
        })

        return state

    @staticmethod
    def load_player_elos(parquet_path: Union[str, Path] = "data/processed/t20_player_elo_ratings.parquet") -> Dict[str, float]:
        """
        Loads the latest player Elo ratings from precomputed parquet storage.

        Returns:
            Dict mapping player_name -> effective_elo
        """
        path = Path(parquet_path)
        if not path.exists():
            return {}

        df = pd.read_parquet(path)
        if "player" not in df.columns or "effective_elo" not in df.columns:
            return {}

        # Get latest effective Elo for each player
        latest_df = df.groupby("player")["effective_elo"].last()
        return latest_df.to_dict()
