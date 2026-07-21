"""
Rich Terminal Dashboard for In-Play Betting Orchestrator
"""

from typing import Any, Dict, List
from collections import deque

from rich.align import Align
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.betting.signal_generator import InPlaySignal, SignalType


class InPlayDashboard:
    """
    Real-time terminal dashboard using the `rich` library.
    Provides a live-updating UI with panels for match state, odds, positions, and logs.
    """

    def __init__(self, match_title: str, target: int, format_type: str, mode: str):
        self.match_title = match_title
        self.target = target
        self.format_type = format_type.upper()
        self.mode = mode.upper()
        self.signal_history = deque(maxlen=15)
        self.layout = self._build_layout()

    def _build_layout(self) -> Layout:
        """Initialises the overall screen layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=15),
        )
        layout["main"].split_row(
            Layout(name="left_panel", ratio=1),
            Layout(name="right_panel", ratio=2),
        )
        layout["left_panel"].split_column(
            Layout(name="state", ratio=1),
            Layout(name="market", ratio=1),
        )
        return layout

    def update(
        self,
        state: Dict[str, Any],
        signal: InPlaySignal,
        positions: Dict[str, Dict[str, Any]],
        bankroll: float,
        action_note: str = "",
    ):
        """Updates all panels with the latest delivery state and signal."""
        # 1. Header
        header = Table.grid(expand=True)
        header.add_column(justify="center", ratio=1)
        header_text = Text(f"TheInsideEdge — In-Play Scanner ({self.format_type})", style="bold cyan")
        sub_text = Text(f"Match: {self.match_title} | Target: {self.target} | Mode: {self.mode}", style="yellow")
        header.add_row(header_text)
        header.add_row(sub_text)
        self.layout["header"].update(Panel(header, style="blue"))

        # 2. Match State Panel
        runs = state.get("runs_scored", 0)
        wickets = state.get("wickets_lost", 0)
        overs = (state.get("total_balls", 0) - state.get("balls_remaining", 0)) / 6.0
        rrr = state.get("required_run_rate", 0.0)

        state_table = Table(expand=True, show_header=False, box=None)
        state_table.add_column("Metric", style="cyan")
        state_table.add_column("Value", style="bold white")
        state_table.add_row("Score:", f"{runs}/{wickets}")
        state_table.add_row("Overs:", f"{overs:.1f}")
        state_table.add_row("Req. Run Rate:", f"{rrr:.2f}")
        state_table.add_row("Target:", f"{self.target}")
        self.layout["state"].update(Panel(state_table, title="[bold green]Live Match State"))

        # 3. Market Odds & Edge
        market_table = Table(expand=True, show_header=False, box=None)
        market_table.add_column("Metric", style="cyan")
        market_table.add_column("Value", justify="right")

        edge_color = "green" if signal.edge > 0 else "red"
        edge_str = f"[{edge_color}]+{signal.edge:.1%}[/]" if signal.edge > 0 else f"[{edge_color}]{signal.edge:.1%}[/]"

        market_table.add_row("Model P(Win):", f"{signal.model_prob:.1%} (±{signal.ensemble_std_dev:.3f})")
        market_table.add_row("Market P:", f"{signal.exchange_implied_prob:.1%}")
        market_table.add_row("Edge:", edge_str)
        market_table.add_row("Back Odds:", f"[yellow]{signal.back_odds or 'N/A'}[/]")
        market_table.add_row("Lay Odds:", f"[yellow]{signal.lay_odds or 'N/A'}[/]")
        self.layout["market"].update(Panel(market_table, title="[bold magenta]Model vs Market"))

        # 4. Open Positions
        pos_table = Table(expand=True)
        pos_table.add_column("Team", style="bold cyan")
        pos_table.add_column("Entry Odds", justify="right")
        pos_table.add_column("Stake", justify="right", style="green")
        pos_table.add_column("Current Lay", justify="right")
        pos_table.add_column("Status", justify="right")

        for m_id, pos in positions.items():
            team = pos["team_backed"]
            entry_odds = pos["entry_odds"]
            stake = pos["stake"]
            curr_lay = signal.lay_odds or "N/A"
            status = f"P(win) {signal.model_prob:.1%}"
            pos_table.add_row(team, f"{entry_odds:.2f}", f"£{stake:.2f}", str(curr_lay), status)

        if not positions:
            pos_table.add_row("No open positions", "", "", "", "")
        
        bankroll_text = f"\n[bold white]Current Bankroll: £{bankroll:.2f}[/]"
        pos_panel_content = Align.center(pos_table, vertical="top")
        
        pos_layout = Layout()
        pos_layout.split_column(
            Layout(pos_table, ratio=1),
            Layout(Align.left(bankroll_text), size=2)
        )

        self.layout["right_panel"].update(Panel(pos_layout, title="[bold yellow]Open Positions"))

        # 5. Signals Log
        log_table = Table(expand=True)
        log_table.add_column("Over", style="dim", width=6)
        log_table.add_column("Score", width=8)
        log_table.add_column("Model P", width=12)
        log_table.add_column("Market", width=10)
        log_table.add_column("Edge", width=8)
        log_table.add_column("Odds", width=6)
        log_table.add_column("Signal", width=10)
        log_table.add_column("Action / Notes")

        # Add current signal to log
        if action_note or signal.signal_type != SignalType.HOLD:
             # Only log significant actions or non-HOLD signals if we want it cleaner, 
             # but we log every ball for completeness.
             pass

        self.signal_history.append((state, signal, action_note))

        for hist_state, sig, note in list(self.signal_history):
            s_runs = hist_state.get("runs_scored", 0)
            s_wkts = hist_state.get("wickets_lost", 0)
            score_str = f"{s_runs}/{s_wkts}"
            edge_c = "green" if sig.edge > 0 else "red"
            edg_str = f"[{edge_c}]+{sig.edge:.1%}[/]" if sig.edge > 0 else f"[{edge_c}]{sig.edge:.1%}[/]"
            
            sig_col = f"[bold green]{sig.signal_type.value}[/]" if sig.signal_type == SignalType.BET else sig.signal_type.value
            if sig.signal_type == SignalType.TRADE_OUT:
                sig_col = f"[bold cyan]TRADE_OUT[/]"

            log_table.add_row(
                str(sig.over_ball),
                score_str,
                f"{sig.model_prob:.1%}",
                f"{sig.exchange_implied_prob:.1%}",
                edg_str,
                str(sig.back_odds),
                sig_col,
                note if note else (f"[{sig.skip_reason}]" if sig.skip_reason else ""),
            )

        self.layout["footer"].update(Panel(log_table, title="[bold blue]Live Signals Log (Last 15 Deliveries)"))
