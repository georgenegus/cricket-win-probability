import os
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path


class BetTracker:
    """
    Persistent CSV-based bet ledger tracking bet placements, P&L,
    bankroll equity, and aggregate performance metrics.
    """

    COLUMNS = [
        "timestamp",
        "event_id",
        "market_id",
        "selection_id",
        "match_description",
        "venue",
        "format",
        "bet_on",
        "model_prob",
        "fair_market_prob",
        "edge",
        "decimal_odds",
        "stake",
        "stake_pct",
        "staking_method",
        "result",
        "pnl",
        "bankroll_after",
        "mode",
    ]

    def __init__(self, log_path="data/betting/bet_log.csv", starting_bankroll=1000.0):
        self.primary_log_path = Path(log_path)
        self.primary_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.starting_bankroll = float(starting_bankroll)
        self.active_log_path = self.primary_log_path

        if not self.primary_log_path.exists():
            df = pd.DataFrame(columns=self.COLUMNS)
            try:
                df.to_csv(self.primary_log_path, index=False)
            except PermissionError:
                self.active_log_path = self.primary_log_path.parent / "bet_log_latest.csv"
                df.to_csv(self.active_log_path, index=False)

    @property
    def log_path(self):
        return self.active_log_path

    def _load_df(self):
        target_path = self.active_log_path if self.active_log_path.exists() else self.primary_log_path
        if not target_path.exists():
            return pd.DataFrame(columns=self.COLUMNS)
        df = pd.read_csv(target_path, dtype={"event_id": str, "market_id": str, "selection_id": str})
        for col in self.COLUMNS:
            if col not in df.columns:
                df[col] = "" if col in ("market_id", "selection_id") else pd.NA
        return df[self.COLUMNS]

    def _save_df(self, df):
        try:
            df.to_csv(self.primary_log_path, index=False)
            self.active_log_path = self.primary_log_path
        except PermissionError:
            self.active_log_path = self.primary_log_path.parent / "bet_log_latest.csv"
            print(f"\n[WARNING] Could not write to {self.primary_log_path} (file is locked or open in Excel).")
            print(f"[WARNING] Saving bet log to active location: {self.active_log_path}")
            df.to_csv(self.active_log_path, index=False)

    def get_current_bankroll(self):
        """Calculates current bankroll based on initial balance and net settled P&L."""
        df = self._load_df()
        if df.empty or "pnl" not in df.columns:
            return self.starting_bankroll

        settled_pnl = df["pnl"].dropna().sum()
        return round(float(self.starting_bankroll + settled_pnl), 2)

    def log_bet(
        self,
        event_id,
        match_description,
        venue,
        format_type,
        bet_on,
        model_prob,
        fair_market_prob,
        edge,
        decimal_odds,
        stake,
        stake_pct,
        staking_method="kelly_0.25",
        mode="paper",
        market_id=None,
        selection_id=None,
    ):
        """Appends a new bet record with result='pending'."""
        if float(stake or 0.0) <= 0.0:
            print(f"Skipping bet log for {match_description} ({bet_on}): Stake is £0.00.")
            return None

        df = self._load_df()

        # Prevent logging duplicate pending bets for the same event and selection
        if not df.empty:
            duplicate_mask = (df["event_id"].astype(str) == str(event_id)) & (df["bet_on"] == str(bet_on)) & (df["result"] == "pending")
            if duplicate_mask.any():
                print(f"Bet already pending for event {event_id} ({bet_on}), skipping duplicate log.")
                return None

        timestamp = datetime.now(timezone.utc).isoformat()
        current_bankroll = self.get_current_bankroll()

        new_row = {
            "timestamp": timestamp,
            "event_id": str(event_id),
            "market_id": str(market_id or ""),
            "selection_id": str(selection_id or ""),
            "match_description": str(match_description),
            "venue": str(venue or "N/A"),
            "format": str(format_type),
            "bet_on": str(bet_on),
            "model_prob": round(float(model_prob), 4),
            "fair_market_prob": round(float(fair_market_prob), 4),
            "edge": round(float(edge), 4),
            "decimal_odds": round(float(decimal_odds), 2),
            "stake": round(float(stake), 2),
            "stake_pct": round(float(stake_pct), 4),
            "staking_method": str(staking_method),
            "result": "pending",
            "pnl": None,
            "bankroll_after": current_bankroll,
            "mode": str(mode),
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        self._save_df(df)
        print(f"Logged {mode} bet: {match_description} -> {bet_on} @ {decimal_odds} (£{stake:.2f})")
        return new_row

    def settle_pending_bets(self, betfair_client):
        """
        Queries Betfair API for outcome of all pending bets and updates settled status.
        """
        open_bets = self.get_open_bets()
        if open_bets.empty:
            print("No pending bets to settle.")
            return 0

        print(f"Checking settlement status for {len(open_bets)} pending bets...")
        settled_count = 0

        for idx, row in open_bets.iterrows():
            event_id = str(row["event_id"])
            market_id = str(row.get("market_id") or "").strip()
            selection_id = str(row.get("selection_id") or "").strip()
            bet_on = str(row["bet_on"])
            match_desc = str(row["match_description"])

            try:
                if market_id and selection_id:
                    outcome = betfair_client.get_settled_runner_status(market_id, selection_id)
                else:
                    outcome = betfair_client.check_event_outcome(event_id, bet_on)

                if outcome in ["win", "loss", "void"]:
                    if self.update_result(event_id, outcome, bet_on=bet_on):
                        settled_count += 1
            except Exception as e:
                print(f"Error checking settlement for event {event_id} ({match_desc}): {e}")

        print(f"Settled {settled_count} of {len(open_bets)} pending bets.")
        return settled_count

    def update_result(self, event_id, result, bet_on=None):
        """
        Updates a pending bet with actual outcome: 'win', 'loss', or 'void'.

        Args:
            event_id (str): Event ID
            result (str): "win", "loss", or "void"
            bet_on (str, optional): Team backed (if multiple bets exist for event)
        """
        df = self._load_df()
        if df.empty:
            return False

        mask = (df["event_id"].astype(str) == str(event_id)) & (df["result"].isin(["pending", "unresolvable"]))
        if bet_on:
            mask = mask & (df["bet_on"] == bet_on)

        matching_indices = df[mask].index
        if len(matching_indices) == 0:
            print(f"No pending bet found for event {event_id}")
            return False

        idx = matching_indices[0]
        res = result.lower().strip()
        stake = float(df.loc[idx, "stake"])
        odds = float(df.loc[idx, "decimal_odds"])

        if res == "win":
            pnl = stake * (odds - 1.0)
        elif res == "loss":
            pnl = -stake
        elif res == "void":
            pnl = 0.0
        else:
            raise ValueError("Result must be 'win', 'loss', or 'void'")

        df.loc[idx, "result"] = res
        df.loc[idx, "pnl"] = round(pnl, 2)

        # Recalculate running bankroll after each settled bet
        curr = self.starting_bankroll
        for i in range(len(df)):
            if pd.notna(df.loc[i, "pnl"]):
                curr += float(df.loc[i, "pnl"])
            df.loc[i, "bankroll_after"] = round(curr, 2)

        self._save_df(df)
        print(f"Updated event {event_id} ({df.loc[idx, 'match_description']}): {res.upper()} (P&L: £{pnl:+.2f})")
        return True

    def get_open_bets(self):
        """Returns DataFrame of all bets with result='pending'."""
        df = self._load_df()
        if df.empty:
            return pd.DataFrame(columns=self.COLUMNS)
        return df[df["result"] == "pending"]

    def get_summary(self):
        """Returns aggregate performance stats dict."""
        df = self._load_df()
        if df.empty:
            return {
                "total_bets": 0,
                "wins": 0,
                "losses": 0,
                "pending": 0,
                "win_rate": 0.0,
                "total_staked": 0.0,
                "total_pnl": 0.0,
                "roi_pct": 0.0,
                "avg_edge": 0.0,
                "current_bankroll": self.starting_bankroll,
            }

        settled = df[df["result"].isin(["win", "loss"])]
        wins = len(df[df["result"] == "win"])
        losses = len(df[df["result"] == "loss"])
        pending = len(df[df["result"] == "pending"])

        total_settled = wins + losses
        win_rate = (wins / total_settled) if total_settled > 0 else 0.0

        total_staked = settled["stake"].sum() if not settled.empty else 0.0
        total_pnl = settled["pnl"].sum() if not settled.empty else 0.0
        roi_pct = (total_pnl / total_staked * 100.0) if total_staked > 0 else 0.0

        avg_edge = df["edge"].mean() if not df.empty else 0.0

        return {
            "total_bets": len(df),
            "wins": wins,
            "losses": losses,
            "pending": pending,
            "win_rate": round(float(win_rate), 4),
            "total_staked": round(float(total_staked), 2),
            "total_pnl": round(float(total_pnl), 2),
            "roi_pct": round(float(roi_pct), 2),
            "avg_edge": round(float(avg_edge), 4),
            "current_bankroll": self.get_current_bankroll(),
        }
