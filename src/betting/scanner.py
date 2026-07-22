import argparse
import sys
from datetime import datetime, timezone
import pandas as pd

from src.betting.betfair_client import BetfairClient
from src.betting.pre_match_predictor import PreMatchPredictor
from src.betting.value_detector import ValueDetector
from src.betting.bet_tracker import BetTracker


def run_scanner(
    mode="paper",
    hours_ahead=48,
    threshold=0.03,
    kelly_fraction=0.25,
    bankroll=1000.0,
    format_filter="both",
):
    """
    Main Pre-Match Value Betting Scanner Workflow.
    """
    # Ensure UTF-8 output encoding for Windows terminals
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print(f"\n{'='*75}")
    print(f"  TheInsideEdge -- Pre-Match Value Scanner")
    print(f"  Mode: {mode.upper()} | Bankroll: £{bankroll:,.2f} | Min Edge: {threshold:.1%}")
    print(f"{'='*75}\n")

    # 1. Initialise components
    print("Initialising betting pipeline components...")
    betfair = BetfairClient()
    predictor = PreMatchPredictor()
    detector = ValueDetector(min_edge=threshold, kelly_fraction=kelly_fraction)
    tracker = BetTracker(starting_bankroll=bankroll)

    # 1b. Check and settle pending bets from past scans
    print("Checking and settling pending bets from previous scans...")
    try:
        tracker.settle_pending_bets(betfair)
    except Exception as e:
        print(f"Error checking bet settlement: {e}")

    # 2. Fetch upcoming cricket events
    print(f"Fetching upcoming cricket events starting in the next {hours_ahead} hours...")
    try:
        events = betfair.get_upcoming_cricket_events(hours_ahead=hours_ahead)
    except Exception as e:
        print(f"Error connecting to Betfair API: {e}")
        return []

    if not events:
        print("No upcoming cricket events found on Betfair matching criteria.")
        return []

    valid_elo_teams = set(predictor.t20_elos.keys()).union(set(predictor.od_elos.keys()))

    scanned_results = []
    value_bets_count = 0

    print(f"\nProcessing {len(events)} events...\n")

    for ev in events:
        event_id = ev["event_id"]
        event_name = ev["event_name"]

        print(f"Scanning Event: {event_name} (ID: {event_id})...")

        # Fetch Match Odds
        odds_data = betfair.get_match_odds(event_id)
        if not odds_data or not odds_data.get("runners") or len(odds_data["runners"]) < 2:
            print("  -> Skipping: No valid Match Winner market or insufficient runners.")
            continue

        runners = odds_data["runners"]
        runner_a = runners[0]
        runner_b = runners[1]

        # Fuzzy match runner names to Elo team names
        team_a_matched, score_a = betfair.match_team_name(runner_a["runner_name"], valid_elo_teams)
        team_b_matched, score_b = betfair.match_team_name(runner_b["runner_name"], valid_elo_teams)

        if not team_a_matched or not team_b_matched:
            print(f"  -> Skipping: Could not match runner names ({runner_a['runner_name']} / {runner_b['runner_name']}) to Elo database.")
            continue

        # Infer format from event_name, competition_name, and market_name
        comp_name = odds_data.get("competition_name", "")
        market_name = odds_data.get("market_name", "")
        text_to_check = f"{event_name} {comp_name} {market_name}".lower()

        od_keywords = ["odi", "one-day", "one day", "50 over", "50-over", "metro bank", "royal london", "list a", "1-day"]
        t20_keywords = ["t20", "twenty20", "blast", "hundred", "ipl", "cpl", "bbl", "psl", "sa20", "ilt20", "mlc", "lpl", "bpl", "super smash"]

        if any(kw in text_to_check for kw in od_keywords):
            fmt = "od"
        elif any(kw in text_to_check for kw in t20_keywords):
            fmt = "t20"
        else:
            fmt = "t20"

        if format_filter != "both" and format_filter.lower() != fmt:
            print(f"  -> Skipping format {fmt} (filtered for {format_filter}).")
            continue

        back_odds_a = runner_a["back_price"]
        back_odds_b = runner_b["back_price"]

        if not back_odds_a or not back_odds_b:
            print("  -> Skipping: Missing back odds price ladder.")
            continue

        # Extract venue from Betfair market or event data
        match_venue = odds_data.get("venue") or ev.get("venue") or "N/A"
        if match_venue == "N/A":
            match_venue = None

        # Predict pre-match probability with venue
        pred_res = predictor.predict(team_a_matched, team_b_matched, venue=match_venue, format_type=fmt)
        model_prob_a = pred_res["prob_a"]
        model_prob_b = pred_res["prob_b"]

        # Value detection
        val_analysis = detector.analyze_match(
            team_a=team_a_matched,
            team_b=team_b_matched,
            model_prob_a=model_prob_a,
            model_prob_b=model_prob_b,
            back_odds_a=back_odds_a,
            back_odds_b=back_odds_b,
        )

        val_analysis["event_id"] = event_id
        val_analysis["event_name"] = event_name
        val_analysis["format"] = fmt
        val_analysis["venue"] = match_venue or "N/A"

        scanned_results.append(val_analysis)

        # Log value bets (only if Kelly stake is > 0)
        if val_analysis["is_value_a"]:
            stake_amount = round(bankroll * val_analysis["kelly_stake_a"], 2)
            if stake_amount > 0:
                value_bets_count += 1
                tracker.log_bet(
                    event_id=event_id,
                    market_id=odds_data["market_id"],
                    selection_id=runner_a["selection_id"],
                    match_description=f"{team_a_matched} v {team_b_matched}",
                    venue=match_venue or "N/A",
                    format_type=fmt,
                    bet_on=team_a_matched,
                    model_prob=model_prob_a,
                    fair_market_prob=val_analysis["fair_prob_a"],
                    edge=val_analysis["edge_a"],
                    decimal_odds=back_odds_a,
                    stake=stake_amount,
                    stake_pct=val_analysis["kelly_stake_a"],
                    staking_method=f"kelly_{kelly_fraction}",
                    mode=mode,
                )

        if val_analysis["is_value_b"]:
            stake_amount = round(bankroll * val_analysis["kelly_stake_b"], 2)
            if stake_amount > 0:
                value_bets_count += 1
                tracker.log_bet(
                    event_id=event_id,
                    market_id=odds_data["market_id"],
                    selection_id=runner_b["selection_id"],
                    match_description=f"{team_a_matched} v {team_b_matched}",
                    venue=match_venue or "N/A",
                    format_type=fmt,
                    bet_on=team_b_matched,
                    model_prob=model_prob_b,
                    fair_market_prob=val_analysis["fair_prob_b"],
                    edge=val_analysis["edge_b"],
                    decimal_odds=back_odds_b,
                    stake=stake_amount,
                    stake_pct=val_analysis["kelly_stake_b"],
                    staking_method=f"kelly_{kelly_fraction}",
                    mode=mode,
                )

    # 3. Print Summary Table
    print(f"\n{'='*85}")
    print(f"  SCAN SUMMARY RESULTS ({len(scanned_results)} Matches Processed)")
    print(f"{'='*85}")
    print(f"{'Match':<30} | {'Fmt':<4} | {'Model':<7} | {'Market':<7} | {'Edge':<7} | {'Odds':<6} | {'Action'}")
    print(f"{'-'*85}")

    for r in scanned_results:
        match_str = f"{r['team_a']} v {r['team_b']}"[:29]
        fmt_str = r['format'].upper()
        
        if r['edge_a'] >= r['edge_b']:
            m_prob = f"{r['model_prob_a']:.1%}"
            mk_prob = f"{r['fair_prob_a']:.1%}"
            edge_str = f"{r['edge_a']:+.1%}"
            odds_str = f"{r['back_odds_a']:.2f}"
        else:
            m_prob = f"{r['model_prob_b']:.1%}"
            mk_prob = f"{r['fair_prob_b']:.1%}"
            edge_str = f"{r['edge_b']:+.1%}"
            odds_str = f"{r['back_odds_b']:.2f}"

        action = r["recommended_action"]
        print(f"{match_str:<30} | {fmt_str:<4} | {m_prob:<7} | {mk_prob:<7} | {edge_str:<7} | {odds_str:<6} | {action}")

    print(f"{'='*85}\n")
    print(f"Value bets identified: {value_bets_count}")
    print(f"Bet log saved to: {tracker.log_path}\n")

    return scanned_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TheInsideEdge — Pre-Match Value Betting Scanner")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper", help="Trading mode")
    parser.add_argument("--hours-ahead", type=int, default=48, help="Scan window in hours")
    parser.add_argument("--threshold", type=float, default=0.03, help="Minimum edge threshold (default 0.03 = 3%%)")
    parser.add_argument("--kelly-fraction", type=float, default=0.25, help="Fractional Kelly multiplier (default 0.25)")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Starting bankroll balance in GBP")
    parser.add_argument("--format", choices=["t20", "od", "both"], default="both", help="Format filter")

    args = parser.parse_args()

    run_scanner(
        mode=args.mode,
        hours_ahead=args.hours_ahead,
        threshold=args.threshold,
        kelly_fraction=args.kelly_fraction,
        bankroll=args.bankroll,
        format_filter=args.format,
    )
