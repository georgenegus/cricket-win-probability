"""
Monte Carlo Match Simulator

Orchestrates full two-innings cricket match simulations across N parallel iterations,
linking Innings 1 score generation to Innings 2 chase targets.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.simulation.innings_simulator import simulate_innings


def simulate_match(
    model: Any,
    team_a: Dict[str, Any],
    team_b: Dict[str, Any],
    venue_avg_score: float,
    n_sims: int = 10000,
    total_overs: float = 20.0,
    bat_first: str = "a",
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Simulates a full two-innings Monte Carlo match across n_sims iterations.

    Args:
        model: Fitted DeliveryOutcomeModel.
        team_a: Dict containing 'lineup' (>=11 players) and 'attack' (>=5 bowlers), optional 'name'.
        team_b: Dict containing 'lineup' (>=11 players) and 'attack' (>=5 bowlers), optional 'name'.
        venue_avg_score: Historical average 1st innings score at venue (e.g. 160.0).
        n_sims: Number of Monte Carlo iterations (default 10,000).
        total_overs: Match overs per innings (20.0 for T20, 50.0 for ODI).
        bat_first: "a" or "b" indicating which team bats first.
        seed: Random seed for reproducibility across the simulation ensemble.

    Returns:
        Dict of length-n_sims outcome arrays and aggregated indicators:
            - "inn1_score": 1st innings total runs
            - "inn2_score": 2nd innings total runs
            - "inn1_wkts": 1st innings wickets lost
            - "inn2_wkts": 2nd innings wickets lost
            - "inn1_balls": 1st innings legal balls bowled
            - "inn2_balls": 2nd innings legal balls bowled
            - "inn1_batter_runs": (n_sims, len(inn1_lineup)) 2D array of per-batter runs
            - "inn2_batter_runs": (n_sims, len(inn2_lineup)) 2D array of per-batter runs
            - "first_batting_won": boolean array (True if 1st batting team won)
            - "second_batting_won": boolean array (True if 2nd batting team won)
            - "is_tie": boolean array (True if scores equal)
            - "margin_runs": run margin when 1st batting team won (np.nan otherwise)
            - "margin_wkts": wicket margin when 2nd batting team won (np.nan otherwise)
            - "first_team_name": Name of 1st batting team
            - "second_team_name": Name of 2nd batting team
            - "bat_first": "a" or "b"
            - "n_sims": int
    """
    bat_first_clean = str(bat_first).lower().strip()
    if bat_first_clean not in ["a", "b"]:
        raise ValueError(f"bat_first must be 'a' or 'b', got '{bat_first}'.")

    if bat_first_clean == "a":
        first_team = team_a
        second_team = team_b
    else:
        first_team = team_b
        second_team = team_a

    first_name = first_team.get("name", "Team A" if bat_first_clean == "a" else "Team B")
    second_name = second_team.get("name", "Team B" if bat_first_clean == "a" else "Team A")

    rng = np.random.default_rng(seed)

    # Innings 1 simulation
    inn1 = simulate_innings(
        model=model,
        lineup=first_team["lineup"],
        attack=second_team["attack"],
        venue_avg_score=venue_avg_score,
        n_sims=n_sims,
        total_overs=total_overs,
        innings=1,
        target=None,
        rng=rng,
    )

    # Chase targets for Innings 2: target is inn1_score + 1
    targets = inn1["final_score"] + 1

    # Innings 2 simulation
    inn2 = simulate_innings(
        model=model,
        lineup=second_team["lineup"],
        attack=first_team["attack"],
        venue_avg_score=venue_avg_score,
        n_sims=n_sims,
        total_overs=total_overs,
        innings=2,
        target=targets,
        rng=rng,
    )

    inn1_score = inn1["final_score"]
    inn2_score = inn2["final_score"]
    inn1_wkts = inn1["wickets"]
    inn2_wkts = inn2["wickets"]

    first_won = inn1_score > inn2_score
    second_won = inn2_score >= targets
    is_tie = inn1_score == inn2_score

    # Margins
    margin_runs = np.where(first_won, (inn1_score - inn2_score).astype(np.float64), np.nan)
    margin_wkts = np.where(second_won, (10 - inn2_wkts).astype(np.float64), np.nan)

    return {
        "inn1_score": inn1_score,
        "inn2_score": inn2_score,
        "inn1_wkts": inn1_wkts,
        "inn2_wkts": inn2_wkts,
        "inn1_balls": inn1["legal_balls"],
        "inn2_balls": inn2["legal_balls"],
        "inn1_batter_runs": inn1["per_batter_runs"],
        "inn2_batter_runs": inn2["per_batter_runs"],
        "first_batting_won": first_won,
        "second_batting_won": second_won,
        "is_tie": is_tie,
        "margin_runs": margin_runs,
        "margin_wkts": margin_wkts,
        "first_team_name": first_name,
        "second_team_name": second_name,
        "bat_first": bat_first_clean,
        "n_sims": n_sims,
    }


def extract_match_squads(
    match_id: Union[int, str],
    format_type: str = "t20",
    processed_dir: str = "data/processed",
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """
    Extracts team_1 and team_2 squad dicts (lineup + attack) and venue_avg_score
    for a historical match_id from dataset parquets.

    Returns:
        (team_1_dict, team_2_dict, venue_avg_score)
    """
    p_dir = Path(processed_dir)
    fmt = "od" if str(format_type).lower() in ["od", "odi"] else "t20"

    meta_path = p_dir / f"{fmt}_match_metadata.parquet"
    player_path = p_dir / f"{fmt}_player_elo_ratings.parquet"
    venue_path = p_dir / f"{fmt}_venue_features.parquet"

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata parquet not found: {meta_path}")

    meta_df = pd.read_parquet(meta_path)
    match_row = meta_df[meta_df["match_id"].astype(str) == str(match_id)]
    if match_row.empty:
        raise ValueError(f"Match ID '{match_id}' not found in {meta_path}")

    row = match_row.iloc[0]
    t1_name = str(row["team_1"])
    t2_name = str(row["team_2"])
    venue_name = str(row.get("venue", "Unknown"))

    def _parse_players(val):
        if isinstance(val, str):
            return [p.strip() for p in val.split(",") if p.strip()]
        if isinstance(val, (list, tuple, np.ndarray)):
            return [str(p).strip() for p in val if str(p).strip()]
        return []

    t1_players = _parse_players(row["team_1_players"])
    t2_players = _parse_players(row["team_2_players"])

    # Venue score lookup
    default_venue_score = 260.0 if fmt == "od" else 160.0
    venue_score = default_venue_score
    if venue_path.exists():
        v_df = pd.read_parquet(venue_path)
        if "match_id" in v_df.columns and "venue_avg_first_innings_score" in v_df.columns:
            v_match = v_df[v_df["match_id"].astype(str) == str(match_id)]
            if not v_match.empty:
                venue_score = float(v_match.iloc[0]["venue_avg_first_innings_score"])

    # Player Elo lookup — batting and bowling ratings kept SEPARATE.
    # Each player has up to two rows (role="batter" / role="bowler"); a single
    # name-keyed dict lets one role's Elo clobber the other's (Bug #1).
    batter_map = {}
    bowler_map = {}
    if player_path.exists():
        p_df = pd.read_parquet(player_path)
        p_match = p_df[p_df["match_id"].astype(str) == str(match_id)]
        for _, p_row in p_match.iterrows():
            p_name = str(p_row["player"])
            entry = {
                "effective_elo": float(p_row.get("effective_elo", 1500.0)),
                "recent_form": float(p_row.get("recent_form", 0.0)),
            }
            if str(p_row.get("role")) == "bowler":
                bowler_map[p_name] = entry
            else:
                batter_map[p_name] = entry

    def _build_team_dict(team_name: str, player_list: list) -> Dict[str, Any]:
        # Batting order as listed; carry each player's BATTING Elo/form.
        lineup = []
        for p in player_list[:11]:
            info = batter_map.get(p, {"effective_elo": 1500.0, "recent_form": 0.0})
            lineup.append({"name": p, "effective_elo": info["effective_elo"], "recent_form": info["recent_form"]})

        # Fill up to 11 if player list had fewer
        while len(lineup) < 11:
            lineup.append({"name": f"Dummy_{len(lineup)+1}", "effective_elo": 1500.0, "recent_form": 0.0})

        # Attack = the top 5 players with the highest BOWLING Elo, descending.
        # We must prioritize players who ACTUALLY have a bowling record. Pure batters
        # who are missing from bowler_map should NOT default to 1500 for sorting, otherwise
        # they get selected ahead of actual part-time bowlers who have <1500 Elo.
        ranked = sorted(
            [p["name"] for p in lineup],
            key=lambda name: bowler_map[name]["effective_elo"] if name in bowler_map else 0.0,
            reverse=True,
        )
        attack = []
        for name in ranked[:5]:
            info = bowler_map.get(name, {"effective_elo": 1500.0, "recent_form": 0.0})
            attack.append({"name": name, "effective_elo": info["effective_elo"], "recent_form": info["recent_form"]})

        return {"name": team_name, "lineup": lineup, "attack": attack}

    t1_dict = _build_team_dict(t1_name, t1_players)
    t2_dict = _build_team_dict(t2_name, t2_players)

    return t1_dict, t2_dict, venue_score
