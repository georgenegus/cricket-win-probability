"""
In-Play Execution Engine Module

Manages bet placement, position tracking, Betfair API minimum stake constraints,
trade-out calculations ("greening up" / stop loss), and paper-trading vs live-trading execution modes.
"""

from typing import Any, Dict, Optional, Union

from src.betting.bet_tracker import BetTracker
from src.betting.signal_generator import InPlaySignal, SignalType


class ExecutionEngine:
    """
    In-Play Execution Engine executing bets and trade-outs based on InPlaySignals.
    Supports paper trading (logging only) and live execution via Betfair API.
    """

    def __init__(
        self,
        betfair_client: Optional[Any] = None,
        bet_tracker: Optional[BetTracker] = None,
        mode: str = "paper",
        min_bet_size: float = 2.00,
    ):
        """
        Args:
            betfair_client: BetfairClient instance (required for mode='live').
            bet_tracker: BetTracker instance for persistent bet logging.
            mode: "paper" (simulation mode) or "live" (real Betfair API bet placement).
            min_bet_size: Minimum allowed stake in £ (Betfair minimum is £2.00).
        """
        self.betfair = betfair_client
        self.tracker = bet_tracker or BetTracker()
        self.mode = mode.lower()
        self.min_bet_size = float(min_bet_size)

        # Active open positions: {market_id: position_dict}
        self.open_positions: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def calculate_trade_out(
        entry_back_odds: float,
        entry_stake: float,
        current_lay_odds: Optional[float],
    ) -> Dict[str, Any]:
        """
        Calculates the lay stake needed to equalize profit on both outcomes ("greening up").

        Equal profit formula:
            lay_stake = entry_stake * entry_back_odds / current_lay_odds
            profit_if_team_wins = entry_stake * (entry_back_odds - 1) - lay_stake * (current_lay_odds - 1)
            profit_if_team_loses = lay_stake - entry_stake
        """
        if (
            current_lay_odds is None
            or current_lay_odds <= 1.0
            or entry_back_odds <= 1.0
            or entry_stake <= 0
        ):
            return {
                "lay_stake": 0.0,
                "profit_if_team_wins": 0.0,
                "profit_if_team_loses": 0.0,
                "is_profitable": False,
            }

        lay_stake = round(float(entry_stake) * float(entry_back_odds) / float(current_lay_odds), 2)
        profit_if_wins = float(entry_stake) * (float(entry_back_odds) - 1.0) - lay_stake * (float(current_lay_odds) - 1.0)
        profit_if_loses = lay_stake - float(entry_stake)

        is_prof = profit_if_wins > 0 and profit_if_loses > 0

        return {
            "lay_stake": lay_stake,
            "profit_if_team_wins": round(profit_if_wins, 2),
            "profit_if_team_loses": round(profit_if_loses, 2),
            "is_profitable": is_prof,
        }

    def check_trade_out(
        self,
        signal: InPlaySignal,
        market_id: str,
        overs_remaining: Optional[float] = None,
    ) -> Optional[InPlaySignal]:
        """
        Checks whether an existing open position for market_id should be traded out.

        Triggers:
        1. Profit Lock: Guaranteed trade-out profit >= 50% of initial stake.
        2. Stop Loss: Model P(win) drop causing unrealized loss <= -30% of stake.
        3. Panic Overs: Remaining overs < 2.0 (high volatility panic trade-out).
        """
        pos = self.open_positions.get(str(market_id))
        if not pos:
            return None

        current_lay_odds = signal.lay_odds
        entry_odds = pos["entry_odds"]
        entry_stake = pos["stake"]
        entry_prob = pos.get("entry_prob", 0.5)

        trade_info = self.calculate_trade_out(entry_odds, entry_stake, current_lay_odds) if current_lay_odds else None

        trade_reason = None
        if trade_info and trade_info["is_profitable"] and trade_info["profit_if_team_wins"] >= (entry_stake * 0.50):
            trade_reason = f"PROFIT LOCK (+£{trade_info['profit_if_team_wins']:.2f} guaranteed profit)"

        elif (signal.model_prob - entry_prob) * entry_stake <= (-entry_stake * 0.30):
            trade_reason = f"STOP LOSS (Model P(win) dropped from {entry_prob:.1%} to {signal.model_prob:.1%})"

        elif overs_remaining is not None and overs_remaining < 2.0 and trade_info and trade_info["profit_if_team_wins"] > 0:
            trade_reason = f"PANIC CLOSE ({overs_remaining:.1f} ov remaining, locking +£{trade_info['profit_if_team_wins']:.2f})"

        if trade_reason:
            lay_stake = trade_info["lay_stake"] if trade_info else entry_stake
            curr_bankroll = max(self.tracker.get_current_bankroll(), 1.0)
            return InPlaySignal(
                signal_type=SignalType.TRADE_OUT,
                timestamp=signal.timestamp,
                over_ball=signal.over_ball,
                model_prob=signal.model_prob,
                exchange_implied_prob=signal.exchange_implied_prob,
                edge=signal.edge,
                back_odds=signal.back_odds,
                lay_odds=current_lay_odds,
                confidence="high",
                kelly_stake_pct=round(lay_stake / curr_bankroll, 4),
                ensemble_std_dev=signal.ensemble_std_dev,
                ci_lower=signal.ci_lower,
                ci_upper=signal.ci_upper,
                match_state_summary=f"TRADE OUT: {trade_reason}",
                selection_id=pos.get("selection_id"),
                team_name=pos.get("team_backed"),
                skip_reason=trade_reason,
            )

        return None

    def execute_signal(
        self,
        signal: InPlaySignal,
        market_id: str,
        selection_id: Union[int, str],
        event_id: str,
        match_description: str,
        venue: str = "N/A",
        format_type: str = "t20",
        bankroll: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Executes a bet given an InPlaySignal.
        """
        if signal.signal_type == SignalType.TRADE_OUT:
            return self.execute_trade_out(market_id, signal)

        if signal.signal_type != SignalType.BET:
            return None

        curr_bankroll = bankroll if bankroll is not None else self.tracker.get_current_bankroll()
        stake = round(curr_bankroll * signal.kelly_stake_pct, 2)

        if stake < self.min_bet_size:
            print(
                f"[EXECUTION ENGINE] Calculated stake £{stake:.2f} below minimum "
                f"£{self.min_bet_size:.2f} for market {market_id}. Skipping bet."
            )
            return None

        team_backed = signal.team_name or match_description.split(" v ")[0]

        if self.mode == "live":
            if not self.betfair:
                raise RuntimeError("BetfairClient instance required for live execution mode.")

            print(f"[LIVE BET] Placing order: BACK {team_backed} @ {signal.back_odds} for £{stake:.2f}...")
            result = self.betfair.place_bet(
                market_id=market_id,
                selection_id=selection_id,
                side="BACK",
                price=signal.back_odds,
                size=stake,
            )

            if result.get("status") != "SUCCESS":
                print(f"[LIVE BET FAILED] {result.get('error', 'UNKNOWN_ERROR')}")
                return None

            print(f"[LIVE BET PLACED] Bet ID: {result.get('bet_id')}")

        else:
            print(f"[PAPER BET] BACK {team_backed} @ {signal.back_odds} for £{stake:.2f} ({signal.over_ball} ov)")

        self.tracker.log_bet(
            event_id=event_id,
            match_description=match_description,
            venue=venue,
            format_type=format_type,
            bet_on=team_backed,
            model_prob=signal.model_prob,
            fair_market_prob=signal.exchange_implied_prob,
            edge=signal.edge,
            decimal_odds=signal.back_odds or 2.0,
            stake=stake,
            stake_pct=signal.kelly_stake_pct,
            staking_method=f"inplay_kelly_{signal.kelly_stake_pct:.3f}",
            mode=self.mode,
        )

        position = {
            "market_id": str(market_id),
            "selection_id": selection_id,
            "event_id": str(event_id),
            "match_description": match_description,
            "team_backed": team_backed,
            "entry_odds": signal.back_odds,
            "entry_prob": signal.model_prob,
            "stake": stake,
            "over_ball": signal.over_ball,
            "mode": self.mode,
        }

        self.open_positions[str(market_id)] = position
        return position

    def execute_trade_out(self, market_id: str, trade_signal: InPlaySignal) -> Optional[Dict[str, Any]]:
        """
        Executes a trade-out order (LAY) to close an open position.
        """
        pos = self.open_positions.get(str(market_id))
        if not pos:
            return None

        curr_bankroll = max(self.tracker.get_current_bankroll(), 1.0)
        lay_stake = round(curr_bankroll * trade_signal.kelly_stake_pct, 2)
        lay_odds = trade_signal.lay_odds or pos["entry_odds"]

        if self.mode == "live":
            if not self.betfair:
                raise RuntimeError("BetfairClient instance required for live execution mode.")

            print(f"[LIVE TRADE-OUT] Placing LAY order: {pos['team_backed']} @ {lay_odds} for £{lay_stake:.2f}...")
            result = self.betfair.place_bet(
                market_id=market_id,
                selection_id=pos["selection_id"],
                side="LAY",
                price=lay_odds,
                size=lay_stake,
            )

            if result.get("status") != "SUCCESS":
                print(f"[LIVE TRADE-OUT FAILED] {result.get('error', 'UNKNOWN_ERROR')}")
                return None
        else:
            print(f"[PAPER TRADE-OUT] LAY {pos['team_backed']} @ {lay_odds} for £{lay_stake:.2f} ({trade_signal.skip_reason})")

        closed_pos = self.close_position(market_id)
        return closed_pos

    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """Returns dictionary of currently tracked open positions."""
        return self.open_positions

    def get_position(self, market_id: str) -> Optional[Dict[str, Any]]:
        """Returns open position for a given market ID if present."""
        return self.open_positions.get(str(market_id))

    def close_position(self, market_id: str) -> Optional[Dict[str, Any]]:
        """Removes and returns an open position for a market ID."""
        return self.open_positions.pop(str(market_id), None)
