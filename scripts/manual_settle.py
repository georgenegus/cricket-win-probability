import argparse
import os
import sys

sys.path.append(os.getcwd())
from src.betting.bet_tracker import BetTracker


def main():
    parser = argparse.ArgumentParser(description="Manually settle a paper bet")
    parser.add_argument("--event-id", required=True, help="Betfair Event ID")
    parser.add_argument("--result", choices=["win", "loss", "void"], required=True, help="Match result")
    parser.add_argument("--bet-on", help="Team backed (optional if unique)")
    args = parser.parse_args()

    tracker = BetTracker()
    success = tracker.update_result(args.event_id, args.result, bet_on=args.bet_on)
    if success:
        print(f"Successfully settled event {args.event_id} as {args.result.upper()}.")
        print("\nUpdated summary:")
        print(tracker.get_summary())
    else:
        print(f"Failed to update event {args.event_id}. Make sure the event_id and bet_on match an open or unresolvable bet.")


if __name__ == "__main__":
    main()
