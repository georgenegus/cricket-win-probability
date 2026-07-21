"""
Market Pricing Engine for Monte Carlo Match Simulation

Converts simulation outcome matrices into fair decimal odds, Monte Carlo standard error estimates,
and probability distributions for exotic betting markets.
"""

import math
from typing import Any, Dict, List, Optional, Union
import numpy as np


def prob_to_fair_odds(p: float, margin: float = 0.0) -> float:
    """
    Converts a probability `p` to fair decimal odds, with an optional safety margin.

    Odds = 1.0 / (p * (1 + margin))

    Args:
        p: Probability in range [0, 1].
        margin: Optional modelling margin buffer (e.g. 0.05 for 5% safety margin).

    Returns:
        float: Decimal odds rounded to 2 decimal places. Returns 999.0 if p <= 0.
    """
    p_float = float(p)
    if p_float <= 0.0:
        return 999.0

    p_clamped = min(max(p_float, 1e-6), 1.0 - 1e-6)
    p_adj = p_clamped * (1.0 + float(margin))
    p_adj = min(max(p_adj, 1e-6), 0.999999)

    return round(1.0 / p_adj, 2)


def compute_mc_standard_error(p: float, n_sims: int) -> float:
    """
    Computes Monte Carlo standard error SE = sqrt(p * (1 - p) / N).

    Args:
        p: Estimated probability in range [0, 1].
        n_sims: Number of Monte Carlo iterations.

    Returns:
        float: Standard error rounded to 4 decimal places.
    """
    if n_sims <= 0:
        return 0.0
    p_clean = min(max(float(p), 0.0), 1.0)
    se = math.sqrt((p_clean * (1.0 - p_clean)) / float(n_sims))
    return round(se, 4)


def price_total_runs(
    sim: Dict[str, Any],
    line: float,
    innings: str = "inn1",
    margin: float = 0.0,
) -> Dict[str, Any]:
    """
    Prices over/under on total runs for Innings 1, Innings 2, or total match score.

    Args:
        sim: Output dictionary from simulate_match().
        line: The benchmark run total line (e.g. 165.5).
        innings: "inn1", "inn2", or "total".
        margin: Optional safety margin buffer.

    Returns:
        Dict containing line, p_over, p_under, over_odds, under_odds, mc_se_over, mc_se_under,
        mean_score, p5_score, p95_score.
    """
    n_sims = int(sim.get("n_sims", 10000))

    if innings == "total":
        scores = sim["inn1_score"] + sim["inn2_score"]
    elif innings in ["inn2", "inn_2", "2"]:
        scores = sim["inn2_score"]
    else:
        scores = sim["inn1_score"]

    line_val = float(line)
    p_over = float((scores > line_val).mean())
    p_under = float((scores < line_val).mean())

    return {
        "line": line_val,
        "innings": innings,
        "p_over": round(p_over, 4),
        "p_under": round(p_under, 4),
        "over_odds": prob_to_fair_odds(p_over, margin=margin),
        "under_odds": prob_to_fair_odds(p_under, margin=margin),
        "mc_se_over": compute_mc_standard_error(p_over, n_sims),
        "mc_se_under": compute_mc_standard_error(p_under, n_sims),
        "mean_score": round(float(scores.mean()), 1),
        "p5_score": float(np.percentile(scores, 5)),
        "p95_score": float(np.percentile(scores, 95)),
    }


def price_top_batsman(
    sim: Dict[str, Any],
    lineup_names: List[str],
    innings: str = "inn1",
    margin: float = 0.0,
) -> Dict[str, Any]:
    """
    Prices Top Batsman market (highest scorer in an innings) from 2D per-batter run matrices.
    Splits tie counts equally among joint top scorers.

    Args:
        sim: Output dictionary from simulate_match().
        lineup_names: List of batter names corresponding to lineup indices [0..10].
        innings: "inn1" or "inn2".
        margin: Optional safety margin buffer.

    Returns:
        Dict mapping batter names to dicts containing p_top, odds, mc_se.
    """
    n_sims = int(sim.get("n_sims", 10000))
    key = "inn2_batter_runs" if innings in ["inn2", "inn_2", "2"] else "inn1_batter_runs"
    runs = sim[key]  # shape (n_sims, n_batters)

    n_batters = runs.shape[1]
    names = list(lineup_names) + [f"Batter_{i+1}" for i in range(len(lineup_names), n_batters)]

    max_runs = runs.max(axis=1, keepdims=True)
    is_top = (runs == max_runs)
    n_ties = is_top.sum(axis=1, keepdims=True)

    # Split tie credit equally
    split_counts = (is_top.astype(np.float64) / n_ties.astype(np.float64)).sum(axis=0)
    p_top = split_counts / float(n_sims)

    results = {}
    for i in range(n_batters):
        name = names[i]
        p_val = float(p_top[i])
        results[name] = {
            "p_top": round(p_val, 4),
            "odds": prob_to_fair_odds(p_val, margin=margin),
            "mc_se": compute_mc_standard_error(p_val, n_sims),
            "mean_runs": round(float(runs[:, i].mean()), 1),
        }

    return results


def price_margin(sim: Dict[str, Any], margin: float = 0.0) -> Dict[str, Any]:
    """
    Prices match winner markets and victory margin distributions.

    Args:
        sim: Output dictionary from simulate_match().
        margin: Optional safety margin buffer.

    Returns:
        Dict containing match winner odds, win probabilities, and expected victory margins.
    """
    n_sims = int(sim.get("n_sims", 10000))
    p_first_win = float(sim["first_batting_won"].mean())
    p_second_win = float(sim["second_batting_won"].mean())
    p_tie = float(sim["is_tie"].mean())

    run_margins = sim["margin_runs"]
    wkt_margins = sim["margin_wkts"]

    mean_run_margin = float(np.nanmean(run_margins)) if not np.isnan(run_margins).all() else 0.0
    mean_wkt_margin = float(np.nanmean(wkt_margins)) if not np.isnan(wkt_margins).all() else 0.0

    return {
        "first_team_name": sim.get("first_team_name", "Team A"),
        "second_team_name": sim.get("second_team_name", "Team B"),
        "p_first_bat_win": round(p_first_win, 4),
        "p_second_bat_win": round(p_second_win, 4),
        "p_tie": round(p_tie, 4),
        "first_bat_odds": prob_to_fair_odds(p_first_win, margin=margin),
        "second_bat_odds": prob_to_fair_odds(p_second_win, margin=margin),
        "tie_odds": prob_to_fair_odds(p_tie, margin=margin),
        "mc_se_first_win": compute_mc_standard_error(p_first_win, n_sims),
        "mc_se_second_win": compute_mc_standard_error(p_second_win, n_sims),
        "mean_run_margin": round(mean_run_margin, 1),
        "mean_wkt_margin": round(mean_wkt_margin, 1),
    }


def price_player_props(
    sim: Dict[str, Any],
    lineup_names: List[str],
    target_runs: float = 30.0,
    innings: str = "inn1",
    margin: float = 0.0,
) -> Dict[str, Any]:
    """
    Prices individual player performance milestone props (e.g. Batter 30+ runs).

    Args:
        sim: Output dictionary from simulate_match().
        lineup_names: List of batter names corresponding to lineup indices [0..10].
        target_runs: The run threshold (e.g. 30.0 for 30+ runs).
        innings: "inn1" or "inn2".
        margin: Optional safety margin buffer.

    Returns:
        Dict mapping batter names to dicts containing p_hit, odds, mc_se.
    """
    n_sims = int(sim.get("n_sims", 10000))
    key = "inn2_batter_runs" if innings in ["inn2", "inn_2", "2"] else "inn1_batter_runs"
    runs = sim[key]  # shape (n_sims, n_batters)

    n_batters = runs.shape[1]
    names = list(lineup_names) + [f"Batter_{i+1}" for i in range(len(lineup_names), n_batters)]

    hits = (runs >= float(target_runs)).astype(np.float64)
    p_hits = hits.mean(axis=0)

    results = {}
    for i in range(n_batters):
        name = names[i]
        p_val = float(p_hits[i])
        results[name] = {
            "target_runs": float(target_runs),
            "p_hit": round(p_val, 4),
            "odds": prob_to_fair_odds(p_val, margin=margin),
            "mc_se": compute_mc_standard_error(p_val, n_sims),
        }

    return results
