"""
Exotic Market Value Scanner

Compares Monte Carlo simulated fair odds against exchange exotic-market prices to flag value.
Reuses ValueDetector for edge/Kelly and BetTracker for logging.
"""

from typing import Dict, Any, Optional
from src.betting.value_detector import ValueDetector
from src.betting.bet_tracker import BetTracker
from src.simulation.market_pricer import (
    price_total_runs,
    price_top_batsman,
    price_margin,
    price_player_props
)

def scan_exotic(
    event_id: str,
    match_description: str,
    venue: str,
    format_type: str,
    sim_results: Dict[str, Any],
    market: str,
    exchange_odds: Dict[str, Any],
    line: Optional[float] = None,
    lineup_names: Optional[list] = None,
    target_runs: Optional[float] = None,
    innings: str = "inn1",
    bet_tracker: Optional[BetTracker] = None,
    value_detector: Optional[ValueDetector] = None,
    mode: str = "paper"
) -> list:
    """
    Scans an exotic market for value by comparing simulated probabilities against exchange odds.
    
    Args:
        event_id: Unique match identifier.
        match_description: e.g., "India vs Australia"
        venue: Match venue.
        format_type: e.g., "T20"
        sim_results: Output dict from simulate_match().
        market: One of "total_runs", "top_batsman", "player_props".
        exchange_odds: Dict mapping selections to decimal odds.
        line: The line for total_runs (e.g., 165.5).
        lineup_names: List of batter names for top_batsman/player_props.
        target_runs: Threshold for player_props (e.g., 30.0).
        innings: "inn1", "inn2", or "total".
        bet_tracker: Optional BetTracker instance to log bets.
        value_detector: Optional ValueDetector instance for edge detection.
        mode: "paper" or "live".
        
    Returns:
        List of dictionaries detailing any value bets found (or empty list if none).
    """
    if value_detector is None:
        value_detector = ValueDetector(min_edge=0.03, kelly_fraction=0.25)
    
    if bet_tracker is None:
        bet_tracker = BetTracker(log_path="data/betting/exotic_bet_log.csv")

    recommendations = []

    if market == "total_runs":
        if line is None:
            raise ValueError("Must provide 'line' for total_runs market")
        
        sim_prices = price_total_runs(sim_results, line, innings=innings)
        
        raw_implied = {}
        for sel in ["over", "under"]:
            if sel in exchange_odds:
                raw_implied[sel] = value_detector.odds_to_implied_prob(exchange_odds[sel])
        
        fair_probs = value_detector.remove_overround(raw_implied)
        
        for sel in ["over", "under"]:
            if sel not in exchange_odds:
                continue
                
            sim_prob = sim_prices[f"p_{sel}"]
            fair_market = fair_probs.get(sel, 0.5)
            exchange_odd = exchange_odds[sel]
            
            val = value_detector.detect_value(sim_prob, fair_market)
            stake = value_detector.kelly_stake(sim_prob, exchange_odd) if val["is_value"] else 0.0
            
            if val["is_value"]:
                bet_on = f"Total Runs {innings.upper()} {sel.capitalize()} {line}"
                rec = {
                    "market": "total_runs",
                    "selection": bet_on,
                    "model_prob": sim_prob,
                    "fair_market_prob": fair_market,
                    "edge": val["edge"],
                    "decimal_odds": exchange_odd,
                    "stake_pct": stake
                }
                recommendations.append(rec)
                
                bankroll = bet_tracker.get_current_bankroll()
                nominal_stake = bankroll * stake
                
                bet_tracker.log_bet(
                    event_id=event_id,
                    match_description=match_description,
                    venue=venue,
                    format_type=format_type,
                    bet_on=bet_on,
                    model_prob=sim_prob,
                    fair_market_prob=fair_market,
                    edge=val["edge"],
                    decimal_odds=exchange_odd,
                    stake=nominal_stake,
                    stake_pct=stake,
                    staking_method="kelly_0.25",
                    mode=mode
                )

    elif market == "top_batsman":
        if lineup_names is None:
            raise ValueError("Must provide 'lineup_names' for top_batsman market")
            
        sim_prices = price_top_batsman(sim_results, lineup_names, innings=innings)
        
        raw_implied = {
            name: value_detector.odds_to_implied_prob(odds)
            for name, odds in exchange_odds.items()
        }
        fair_probs = value_detector.remove_overround(raw_implied)
        
        for name, odds in exchange_odds.items():
            if name not in sim_prices:
                continue
                
            sim_prob = sim_prices[name]["p_top"]
            fair_market = fair_probs.get(name, 1.0 / len(lineup_names))
            
            val = value_detector.detect_value(sim_prob, fair_market)
            stake = value_detector.kelly_stake(sim_prob, odds) if val["is_value"] else 0.0
            
            if val["is_value"]:
                bet_on = f"Top Batsman {innings.upper()} - {name}"
                rec = {
                    "market": "top_batsman",
                    "selection": bet_on,
                    "model_prob": sim_prob,
                    "fair_market_prob": fair_market,
                    "edge": val["edge"],
                    "decimal_odds": odds,
                    "stake_pct": stake
                }
                recommendations.append(rec)
                
                bankroll = bet_tracker.get_current_bankroll()
                nominal_stake = bankroll * stake
                
                bet_tracker.log_bet(
                    event_id=event_id,
                    match_description=match_description,
                    venue=venue,
                    format_type=format_type,
                    bet_on=bet_on,
                    model_prob=sim_prob,
                    fair_market_prob=fair_market,
                    edge=val["edge"],
                    decimal_odds=odds,
                    stake=nominal_stake,
                    stake_pct=stake,
                    staking_method="kelly_0.25",
                    mode=mode
                )
                
    elif market == "player_props":
        if lineup_names is None or target_runs is None:
            raise ValueError("Must provide 'lineup_names' and 'target_runs' for player_props market")
            
        sim_prices = price_player_props(sim_results, lineup_names, target_runs=target_runs, innings=innings)
        
        for name, prop_odds in exchange_odds.items():
            if name not in sim_prices:
                continue
                
            sim_prob = sim_prices[name]["p_hit"]
            
            if isinstance(prop_odds, dict) and "yes" in prop_odds and "no" in prop_odds:
                raw_implied = {
                    "yes": value_detector.odds_to_implied_prob(prop_odds["yes"]),
                    "no": value_detector.odds_to_implied_prob(prop_odds["no"])
                }
                fair_probs = value_detector.remove_overround(raw_implied)
                fair_market = fair_probs.get("yes", 0.5)
                odds = prop_odds["yes"]
            else:
                odds = float(prop_odds) if not isinstance(prop_odds, dict) else prop_odds.get("yes", 0.0)
                fair_market = value_detector.odds_to_implied_prob(odds) * 0.95
            
            if odds <= 1.0:
                continue
                
            val = value_detector.detect_value(sim_prob, fair_market)
            stake = value_detector.kelly_stake(sim_prob, odds) if val["is_value"] else 0.0
            
            if val["is_value"]:
                bet_on = f"Player Prop: {name} to score {target_runs}+ Runs"
                rec = {
                    "market": "player_props",
                    "selection": bet_on,
                    "model_prob": sim_prob,
                    "fair_market_prob": fair_market,
                    "edge": val["edge"],
                    "decimal_odds": odds,
                    "stake_pct": stake
                }
                recommendations.append(rec)
                
                bankroll = bet_tracker.get_current_bankroll()
                nominal_stake = bankroll * stake
                
                bet_tracker.log_bet(
                    event_id=event_id,
                    match_description=match_description,
                    venue=venue,
                    format_type=format_type,
                    bet_on=bet_on,
                    model_prob=sim_prob,
                    fair_market_prob=fair_market,
                    edge=val["edge"],
                    decimal_odds=odds,
                    stake=nominal_stake,
                    stake_pct=stake,
                    staking_method="kelly_0.25",
                    mode=mode
                )
    else:
        print(f"Market '{market}' scanning is not implemented yet.")
        
    return recommendations
