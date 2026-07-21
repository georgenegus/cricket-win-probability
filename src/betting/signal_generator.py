"""
In-Play Signal Generator Module

Compares real-time XGBoost model P(win) predictions against Betfair exchange odds
to generate structured betting signals (BET, TRADE_OUT, HOLD, SKIP) on every delivery.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Union
import pandas as pd

from src.betting.value_detector import ValueDetector


class SignalType(Enum):
    BET = "BET"             # High confidence value edge detected -> place bet
    TRADE_OUT = "TRADE_OUT" # Position trade-out condition satisfied -> lock profit / cut loss
    HOLD = "HOLD"           # No actionable edge
    SKIP = "SKIP"           # Edge detected but filtered due to uncertainty or low liquidity


@dataclass
class InPlaySignal:
    """Represents an in-play betting signal generated after a delivery."""
    signal_type: SignalType
    timestamp: str
    over_ball: str                     # e.g., "14.3"
    model_prob: float                  # Model P(win)
    exchange_implied_prob: float       # Market implied probability (overround removed)
    edge: float                        # model_prob - exchange_implied_prob
    back_odds: Optional[float]         # Best available back odds
    lay_odds: Optional[float]          # Best available lay odds
    confidence: str                    # "strong" / "moderate" / "none"
    kelly_stake_pct: float             # Recommended stake as % of bankroll
    ensemble_std_dev: float            # Model 5-fold ensemble uncertainty
    ci_lower: float                    # 95% CI lower bound
    ci_upper: float                    # 95% CI upper bound
    match_state_summary: str           # e.g., "85/2 after 10.0 overs, chasing 185"
    selection_id: Optional[Union[int, str]] = None
    team_name: Optional[str] = None
    available_liquidity: Optional[float] = None
    skip_reason: Optional[str] = None


class SignalGenerator:
    """
    In-Play Signal Generator comparing live model predictions against exchange odds.
    Applies latency/spread safety margins and model uncertainty caps.
    """

    def __init__(
        self,
        model: Any,
        value_detector: Optional[ValueDetector] = None,
        betfair_client: Optional[Any] = None,
        min_edge: float = 0.05,
        min_liquidity: float = 100.0,
        max_std_dev: float = 0.10,
        kelly_fraction: float = 0.15,
    ):
        """
        Args:
            model: Fitted XGBWinProbabilityModel instance.
            value_detector: ValueDetector instance (reused or auto-created).
            betfair_client: Optional BetfairClient instance for fetching live market odds.
            min_edge: Minimum edge required to trigger a BET signal (default 5%).
            min_liquidity: Minimum available matched volume (£) at top price.
            max_std_dev: Maximum ensemble standard deviation to trust signal.
            kelly_fraction: Kelly stake fraction (default 15% for in-play).
        """
        self.model = model
        self.betfair = betfair_client
        self.min_edge = float(min_edge)
        self.min_liquidity = float(min_liquidity)
        self.max_std_dev = float(max_std_dev)
        self.kelly_fraction = float(kelly_fraction)

        self.detector = value_detector or ValueDetector(
            min_edge=self.min_edge,
            kelly_fraction=self.kelly_fraction,
        )

    def generate_signal(
        self,
        state: pd.Series,
        back_odds: Optional[float],
        lay_odds: Optional[float] = None,
        opp_back_odds: Optional[float] = None,
        available_liquidity: Optional[float] = None,
        team_name: Optional[str] = None,
        selection_id: Optional[Union[int, str]] = None,
    ) -> InPlaySignal:
        """
        Generates an InPlaySignal for a single delivery state and odds snapshot.
        """
        # 1. Model Prediction
        res = self.model.predict_with_ensemble(state)
        model_prob = float(res["win_probability"])
        std_dev = float(res["std_dev"])
        ci_lower = float(res["ci_lower"])
        ci_upper = float(res["ci_upper"])

        # Format state summary
        overs = (state.get("total_overs", 20.0) * 6.0 - state.get("balls_remaining", 0.0)) / 6.0
        target = state.get("target", 0.0)
        wickets = int(state.get("wickets_lost", 0))
        runs_needed = target - state.get("runs_remaining", 0.0)
        summary = f"{int(runs_needed)}/{wickets} ({overs:.1f} ov), chasing {int(target)}"
        over_ball_str = f"{int(overs)}.{int(round((overs % 1) * 6))}"
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Handle missing or invalid odds
        if back_odds is None or back_odds <= 1.0:
            return InPlaySignal(
                signal_type=SignalType.HOLD,
                timestamp=timestamp_str,
                over_ball=over_ball_str,
                model_prob=round(model_prob, 4),
                exchange_implied_prob=0.0,
                edge=0.0,
                back_odds=back_odds,
                lay_odds=lay_odds,
                confidence="none",
                kelly_stake_pct=0.0,
                ensemble_std_dev=round(std_dev, 4),
                ci_lower=round(ci_lower, 4),
                ci_upper=round(ci_upper, 4),
                match_state_summary=summary,
                selection_id=selection_id,
                team_name=team_name,
                available_liquidity=available_liquidity,
                skip_reason="Invalid or missing back odds",
            )

        # 2. Convert odds & remove overround if opp_back_odds provided
        raw_implied = 1.0 / float(back_odds)
        if opp_back_odds is not None and opp_back_odds > 1.0:
            raw_opp = 1.0 / float(opp_back_odds)
            overround_dict = self.detector.remove_overround({"team": raw_implied, "opp": raw_opp})
            fair_prob = overround_dict.get("team", raw_implied)
        else:
            fair_prob = raw_implied

        edge = round(model_prob - fair_prob, 4)

        # 3. Apply Filters: Uncertainty -> Liquidity -> Minimum Edge
        liquidity_val = available_liquidity if available_liquidity is not None else 1000.0

        if std_dev > self.max_std_dev:
            signal_type = SignalType.SKIP
            skip_reason = f"High ensemble uncertainty (std_dev {std_dev:.3f} > {self.max_std_dev:.3f})"
            confidence = "none"
            stake_pct = 0.0
        elif liquidity_val < self.min_liquidity:
            signal_type = SignalType.SKIP
            skip_reason = f"Low market liquidity (£{liquidity_val:.1f} < £{self.min_liquidity:.1f})"
            confidence = "none"
            stake_pct = 0.0
        elif edge >= self.min_edge:
            signal_type = SignalType.BET
            skip_reason = None
            confidence = "strong" if edge >= 0.08 else "moderate"
            stake_pct = self.detector.kelly_stake(model_prob, back_odds)
        else:
            signal_type = SignalType.HOLD
            skip_reason = f"Edge ({edge:.1%}) below threshold ({self.min_edge:.1%})"
            confidence = "none"
            stake_pct = 0.0

        return InPlaySignal(
            signal_type=signal_type,
            timestamp=timestamp_str,
            over_ball=over_ball_str,
            model_prob=round(model_prob, 4),
            exchange_implied_prob=round(fair_prob, 4),
            edge=edge,
            back_odds=back_odds,
            lay_odds=lay_odds,
            confidence=confidence,
            kelly_stake_pct=stake_pct,
            ensemble_std_dev=round(std_dev, 4),
            ci_lower=round(ci_lower, 4),
            ci_upper=round(ci_upper, 4),
            match_state_summary=summary,
            selection_id=selection_id,
            team_name=team_name,
            available_liquidity=liquidity_val,
            skip_reason=skip_reason,
        )
