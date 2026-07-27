import os
import sys
from datetime import datetime, timezone, timedelta
import pandas as pd

sys.path.append(os.getcwd())

from src.betting.bet_tracker import BetTracker
from src.betting.betfair_client import BetfairClient


def reconcile_legacy_bets(max_age_hours=72, log_path="data/betting/bet_log.csv"):
    """
    Attempts to settle legacy pending bets via Betfair API.
    For any bets remaining pending that are older than max_age_hours,
    marks them as 'unresolvable' so they do not retry endlessly.
    """
    print("=" * 70)
    print("  Legacy Bet Reconciliation Script")
    print("=" * 70)

    tracker = BetTracker(log_path=log_path)
    
    # 1. Attempt Betfair settlement
    try:
        betfair = BetfairClient()
        print("\nAttempting settlement via Betfair Client...")
        settled_count = tracker.settle_pending_bets(betfair)
        print(f"Settled {settled_count} bets.")
    except Exception as e:
        print(f"Warning: Betfair settlement failed or unavailable ({e}). Proceeding with legacy timeout check.")

    # 2. Mark remaining old pending bets as unresolvable
    df = tracker._load_df()
    if df.empty:
        print("No bets found in log.")
        return

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max_age_hours)
    
    unresolvable_count = 0
    for idx, row in df.iterrows():
        if row.get("result") == "pending":
            ts_str = row.get("timestamp")
            if pd.notna(ts_str):
                try:
                    ts = datetime.fromisoformat(str(ts_str))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    
                    if ts < cutoff:
                        df.loc[idx, "result"] = "unresolvable"
                        df.loc[idx, "pnl"] = pd.NA
                        unresolvable_count += 1
                        print(f"Marked event {row.get('event_id')} ({row.get('match_description')}) as 'unresolvable'.")
                except Exception as ex:
                    print(f"Error parsing timestamp for row {idx}: {ex}")

    if unresolvable_count > 0:
        tracker._save_df(df)
        print(f"\nUpdated {unresolvable_count} old pending bet(s) to 'unresolvable'.")
    else:
        print("\nNo old pending bets needed marking as 'unresolvable'.")

    print("\nReconciliation summary:")
    print(tracker.get_summary())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Settle pending bets via Betfair; sweep aged-out ones to 'unresolvable'.")
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=72.0,
        help="Pending bets older than this are marked 'unresolvable' (default 72h = ~3 days of settlement attempts).",
    )
    parser.add_argument("--log-path", default="data/betting/bet_log.csv", help="Path to the bet ledger CSV.")
    args = parser.parse_args()

    reconcile_legacy_bets(max_age_hours=args.max_age_hours, log_path=args.log_path)
