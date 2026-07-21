"""
In-Play Value Betting Scanner & Orchestrator

Integrates live scorecard feed, real-time state builder, XGBoost win probability engine,
signal generator, and execution engine into an automated in-play betting loop.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.betting.bet_tracker import BetTracker
from src.betting.betfair_client import BetfairClient
from src.betting.execution_engine import ExecutionEngine
from src.betting.live_scorecard import ESPNCricInfoScraper, LiveScorecardFeed, ManualEntryFeed
from src.betting.live_state_builder import LiveStateBuilder
from src.betting.pre_match_predictor import PreMatchPredictor
from src.betting.signal_generator import InPlaySignal, SignalGenerator, SignalType
from src.models.xgb_win_probability import XGBWinProbabilityModel


class InPlayScanner:
    """
    Main Orchestrator for In-Play Value Betting.
    Monitors live match deliveries, constructs features, generates signals, and manages execution.
    """

    MAX_BETS_PER_MATCH = 3          # Maximum 3 bets per match
    MAX_EXPOSURE_PER_MATCH = 0.10   # Maximum 10% of bankroll exposed per match
    COOLDOWN_DELIVERIES = 3         # Minimum 3 legal deliveries wait between bets

    def __init__(
        self,
        format_type: str = "t20",
        mode: str = "paper",
        min_edge: float = 0.05,
        min_liquidity: float = 100.0,
        max_std_dev: float = 0.10,
        kelly_fraction: float = 0.15,
        starting_bankroll: float = 1000.0,
        betfair_client: Optional[BetfairClient] = None,
        bet_tracker: Optional[BetTracker] = None,
        use_dashboard: bool = False,
    ):
        self.format_type = format_type.lower()
        self.mode = mode.lower()
        self.min_edge = float(min_edge)
        self.min_liquidity = float(min_liquidity)
        self.max_std_dev = float(max_std_dev)
        self.kelly_fraction = float(kelly_fraction)
        self.starting_bankroll = float(starting_bankroll)
        self.use_dashboard = use_dashboard

        self.betfair = betfair_client
        self.tracker = bet_tracker or BetTracker(starting_bankroll=self.starting_bankroll)
        self.model: Optional[XGBWinProbabilityModel] = None

        # Safety tracking counters per match
        self.bets_placed_count = 0
        self.total_exposure = 0.0
        self.last_bet_delivery_idx = -999

    def load_model(self) -> None:
        """Loads trained XGBoost model for current format."""
        cache_name = f"{self.format_type}_xgb_models.pkl"
        cache_path = Path("data/processed") / cache_name

        self.model = XGBWinProbabilityModel(format_type=self.format_type)
        if cache_path.exists():
            self.model.fit(None, cache_path=str(cache_path))
        else:
            raise FileNotFoundError(f"Model cache not found at {cache_path}. Train model first.")

    def check_safety_limits(self, bankroll: float, current_delivery_idx: int) -> Tuple[bool, Optional[str]]:
        """
        Validates safety constraints before executing a bet.

        Returns:
            (is_safe: bool, reason_if_unsafe: str)
        """
        if self.bets_placed_count >= self.MAX_BETS_PER_MATCH:
            return False, f"Max bets limit reached ({self.MAX_BETS_PER_MATCH})"

        if (self.total_exposure / bankroll) >= self.MAX_EXPOSURE_PER_MATCH:
            return False, f"Max exposure limit reached ({self.MAX_EXPOSURE_PER_MATCH:.0%})"

        deliveries_since_last_bet = current_delivery_idx - self.last_bet_delivery_idx
        if deliveries_since_last_bet < self.COOLDOWN_DELIVERIES:
            return (
                False,
                f"Cooldown active ({deliveries_since_last_bet}/{self.COOLDOWN_DELIVERIES} deliveries)",
            )

        return True, None

    def run_match(
        self,
        match_url_or_id: str,
        chasing_team: str,
        defending_team: str,
        target: int,
        event_id: str = "manual_event",
        market_id: str = "manual_market",
        chasing_selection_id: Union[int, str] = 1,
        venue: str = "Unknown",
        feed_type: str = "espn",
        poll_interval: float = 10.0,
        mock_odds_stream: Optional[List[Dict[str, Any]]] = None,
    ) -> List[InPlaySignal]:
        """
        Executes the in-play value betting orchestrator loop for a single match chase.
        """
        if not self.model:
            self.load_model()

        # Lookup Elo ratings & venue baseline
        player_elos = LiveStateBuilder.load_player_elos()
        predictor = PreMatchPredictor()
        batting_elo = predictor.get_team_elo(chasing_team, self.format_type)
        bowling_elo = predictor.get_team_elo(defending_team, self.format_type)
        venue_avg = predictor.get_venue_avg_score(venue, self.format_type)

        # Initialize StateBuilder, SignalGenerator, ExecutionEngine
        state_builder = LiveStateBuilder(
            format_type=self.format_type,
            target=target,
            batting_team_elo=batting_elo,
            bowling_team_elo=bowling_elo,
            venue_avg_first_innings_score=venue_avg,
            player_elo_dict=player_elos,
        )

        signal_gen = SignalGenerator(
            model=self.model,
            betfair_client=self.betfair,
            min_edge=self.min_edge,
            min_liquidity=self.min_liquidity,
            max_std_dev=self.max_std_dev,
            kelly_fraction=self.kelly_fraction,
        )

        engine = ExecutionEngine(
            betfair_client=self.betfair,
            bet_tracker=self.tracker,
            mode=self.mode,
        )

        # Connect Scorecard Feed
        feed: LiveScorecardFeed
        if feed_type == "manual":
            feed = ManualEntryFeed()
            feed.set_match_info(
                batting_team=chasing_team,
                bowling_team=defending_team,
                target=target,
                innings=2,
                format_type=self.format_type,
            )
        else:
            feed = ESPNCricInfoScraper()
        feed.connect(match_url_or_id)

        # Reset per-match safety tracking
        self.bets_placed_count = 0
        self.total_exposure = 0.0
        self.last_bet_delivery_idx = -999

        signals_history: List[InPlaySignal] = []
        delivery_idx = 0

        # Setup Dashboard
        dashboard = None
        if self.use_dashboard:
            try:
                from src.betting.in_play_dashboard import InPlayDashboard
                dashboard = InPlayDashboard(
                    match_title=f"{chasing_team} v {defending_team}",
                    target=target,
                    format_type=self.format_type,
                    mode=self.mode,
                )
            except ImportError:
                print("Warning: 'rich' library not installed. Falling back to basic console.")
                self.use_dashboard = False

        if not self.use_dashboard:
            # Print basic header
            print("=" * 80)
            print(f"  TheInsideEdge — In-Play Value Scanner ({self.format_type.upper()})")
            print(f"  Match: {chasing_team} v {defending_team} | Target: {target} runs | Venue: {venue}")
            print(f"  Mode: {self.mode.upper()} | Edge Threshold: {self.min_edge:.1%} | Kelly: {self.kelly_fraction:.0%}")
            print("=" * 80)
            print(f"{'Over':<6} {'Score':<8} {'Model P(Win)':<14} {'Market P':<10} {'Edge':<8} {'Odds':<6} {'Signal':<12} {'Action/Notes'}")
            print("-" * 80)

        def poll_loop():
            nonlocal delivery_idx
            for delivery in feed.poll(interval_seconds=poll_interval):
                delivery_idx += 1

                # Update State
                state_builder.update(delivery)
                state = state_builder.get_state()

                # Determine odds for delivery
                back_odds = None
                lay_odds = None
                liquidity = 1000.0

                if mock_odds_stream and (delivery_idx - 1) < len(mock_odds_stream):
                    odds_item = mock_odds_stream[delivery_idx - 1]
                    back_odds = odds_item.get("back_odds")
                    lay_odds = odds_item.get("lay_odds")
                    liquidity = odds_item.get("liquidity", 1000.0)
                elif self.betfair and market_id != "manual_market":
                    market_book = self.betfair.get_live_market_odds(market_id)
                    if market_book and market_book.get("runners"):
                        for runner in market_book["runners"]:
                            if runner["selection_id"] == chasing_selection_id:
                                back_odds = runner.get("back_price")
                                lay_odds = runner.get("lay_price")
                                liquidity = runner.get("back_size", 0.0)

                # Fallback for manual feed testing if no odds supplied
                if back_odds is None and feed_type == "manual":
                    back_odds = 2.00

                # Generate Signal
                signal = signal_gen.generate_signal(
                    state=state,
                    back_odds=back_odds,
                    lay_odds=lay_odds,
                    available_liquidity=liquidity,
                    team_name=chasing_team,
                    selection_id=chasing_selection_id,
                )
                signals_history.append(signal)

                # Check Trade-Out Triggers for Open Positions
                overs_rem = state.get("balls_remaining", 0.0) / 6.0
                trade_out_sig = engine.check_trade_out(signal, market_id, overs_remaining=overs_rem)

                action_note = ""
                if trade_out_sig:
                    closed_pos = engine.execute_trade_out(market_id, trade_out_sig)
                    if closed_pos:
                        self.total_exposure = max(0.0, self.total_exposure - closed_pos.get("stake", 0.0))
                        action_note = f"TRADE OUT EXECUTED ({trade_out_sig.skip_reason})"
                        signal = trade_out_sig

                elif signal.signal_type == SignalType.BET:
                    bankroll = self.tracker.get_current_bankroll()
                    is_safe, safety_reason = self.check_safety_limits(bankroll, delivery_idx)

                    if is_safe:
                        pos = engine.execute_signal(
                            signal=signal,
                            market_id=market_id,
                            selection_id=chasing_selection_id,
                            event_id=event_id,
                            match_description=f"{chasing_team} v {defending_team}",
                            venue=venue,
                            format_type=self.format_type,
                            bankroll=bankroll,
                        )
                        if pos:
                            self.bets_placed_count += 1
                            self.total_exposure += pos["stake"]
                            self.last_bet_delivery_idx = delivery_idx
                            action_note = f"EXECUTED £{pos['stake']:.2f}"
                        else:
                            action_note = "EXECUTION SKIPPED (min stake)"
                    else:
                        action_note = f"SAFETY BLOCK ({safety_reason})"

                elif signal.skip_reason:
                    action_note = f"({signal.skip_reason})"

                # Render dashboard or basic print
                if dashboard:
                    dashboard.update(
                        state=state,
                        signal=signal,
                        positions=engine.get_open_positions(),
                        bankroll=self.tracker.get_current_bankroll(),
                        action_note=action_note,
                    )
                else:
                    score_str = f"{state_builder.runs_scored}/{state_builder.wickets_lost}"
                    edge_str = f"+{signal.edge:.1%}" if signal.edge > 0 else f"{signal.edge:.1%}"
                    odds_str = f"{signal.back_odds:.2f}" if signal.back_odds else "N/A"
                    mkt_str = f"{signal.exchange_implied_prob:.1%}" if signal.exchange_implied_prob else "N/A"

                    print(
                        f"{signal.over_ball:<6} {score_str:<8} "
                        f"{signal.model_prob:.1%} (±{signal.ensemble_std_dev:.3f})   "
                        f"{mkt_str:<10} {edge_str:<8} {odds_str:<6} "
                        f"[{signal.signal_type.value}] {action_note}"
                    )

        if dashboard:
            from rich.live import Live
            with Live(dashboard.layout, refresh_per_second=4, screen=True):
                poll_loop()
        else:
            poll_loop()

        # Match Complete Summary
        if not dashboard:
            print("-" * 80)
            print("  MATCH COMPLETE")
            print(f"  Total Deliveries Tracked: {delivery_idx}")
            print(f"  Total Signals Generated: {len(signals_history)}")
            print(f"  Total Bets Placed: {self.bets_placed_count} (Exposure: £{self.total_exposure:.2f})")
            print("=" * 80)

        return signals_history


def main():
    parser = argparse.ArgumentParser(description="TheInsideEdge — In-Play Value Betting Scanner")
    parser.add_argument("--match-url", type=str, default="1411242", help="ESPNcricinfo match URL or numeric ID")
    parser.add_argument("--chasing", type=str, required=True, help="Name of chasing team")
    parser.add_argument("--defending", type=str, required=True, help="Name of defending team")
    parser.add_argument("--target", type=int, required=True, help="Chase target score")
    parser.add_argument("--event-id", type=str, default="manual_event", help="Betfair Event ID")
    parser.add_argument("--market-id", type=str, default="manual_market", help="Betfair Market ID")
    parser.add_argument("--format", type=str, default="t20", choices=["t20", "od"], help="Match format")
    parser.add_argument("--venue", type=str, default="Unknown", help="Venue name")
    parser.add_argument("--mode", type=str, default="paper", choices=["paper", "live"], help="Execution mode")
    parser.add_argument("--feed", type=str, default="espn", choices=["espn", "manual"], help="Scorecard feed source")
    parser.add_argument("--min-edge", type=float, default=0.05, help="Minimum edge threshold (default 0.05)")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Starting bankroll £")
    parser.add_argument("--poll-interval", type=float, default=10.0, help="Poll interval in seconds")
    parser.add_argument("--dashboard", action="store_true", help="Enable rich terminal UI dashboard")

    args = parser.parse_args()

    scanner = InPlayScanner(
        format_type=args.format,
        mode=args.mode,
        min_edge=args.min_edge,
        starting_bankroll=args.bankroll,
        use_dashboard=args.dashboard,
    )

    scanner.run_match(
        match_url_or_id=args.match_url,
        chasing_team=args.chasing,
        defending_team=args.defending,
        target=args.target,
        event_id=args.event_id,
        market_id=args.market_id,
        venue=args.venue,
        feed_type=args.feed,
        poll_interval=args.poll_interval,
    )


if __name__ == "__main__":
    main()
