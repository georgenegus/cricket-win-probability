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
from src.data_processing.format_spec import spec


def simulate_match(
    model: Any,
    team_a: Dict[str, Any],
    team_b: Dict[str, Any],
    venue_avg_score: float,
    n_sims: int = 10000,
    total_overs: Optional[float] = None,
    balls_per_over: Optional[float] = None,
    format_type: Optional[str] = None,
    bat_first: str = "a",
    seed: Optional[int] = 42,
    live_state: Optional[Dict[str, Any]] = None,
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
        live_state: None for a pre-match simulation, or a dict describing an in-progress
            match to RESUME from: {"innings": 1|2, "runs", "wickets", "legal_balls",
            "target" (required when innings==2)}. bat_first must be set so the team currently
            batting is the 1st-batting team when innings==1, or the 2nd-batting (chasing)
            team when innings==2. When innings==2 the 1st innings is already complete, so it
            is fixed at target-1 rather than simulated (its per-batter markets are settled and
            returned as zeros — callers should not price 1st-innings props in that case).

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
            - "inn1_bowler_wkts"/"inn2_bowler_wkts": (n_sims, len(attack)) per-bowler wickets
            - "inn1_bowling_names"/"inn2_bowling_names": names labelling those wicket columns
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

    fmt = format_type or getattr(model, "format_type", "t20")
    fmt_spec = spec(fmt)
    tot_overs = float(total_overs if total_overs is not None else fmt_spec["total_overs"])
    bpo = int(balls_per_over if balls_per_over is not None else fmt_spec["balls_per_over"])

    rng = np.random.default_rng(seed)

    live_innings = int(live_state["innings"]) if live_state else None

    if live_innings == 2:
        # 1st innings is complete — its total is known (target - 1), so don't re-simulate it.
        # Per-batter runs for a settled innings can't be recovered from a snapshot; zeros
        # signal callers not to price 1st-innings props.
        known_target = int(live_state["target"])
        inn1_final = np.full(n_sims, known_target - 1, dtype=np.int32)
        inn1 = {
            "final_score": inn1_final,
            "wickets": np.full(n_sims, 10, dtype=np.int8),
            "legal_balls": np.full(n_sims, int(round(tot_overs * bpo)), dtype=np.int32),
            "per_batter_runs": np.zeros((n_sims, len(first_team["lineup"])), dtype=np.int32),
            # 1st innings is settled on a live inn2 resume; per-bowler wickets can't be
            # recovered from the aggregate snapshot, so zeros signal "don't price these".
            "per_bowler_wkts": np.zeros((n_sims, len(second_team["attack"])), dtype=np.int32),
        }
    else:
        # Fresh 1st innings, or a live 1st innings resumed from the current snapshot.
        inn1_start = None
        if live_innings == 1:
            inn1_start = {
                "runs": live_state["runs"],
                "wickets": live_state["wickets"],
                "legal_balls": live_state["legal_balls"],
            }
        inn1 = simulate_innings(
            model=model,
            lineup=first_team["lineup"],
            attack=second_team["attack"],
            venue_avg_score=venue_avg_score,
            n_sims=n_sims,
            total_overs=tot_overs,
            balls_per_over=bpo,
            format_type=fmt,
            innings=1,
            target=None,
            rng=rng,
            start_state=inn1_start,
        )

    # Chase targets for Innings 2: target is inn1_score + 1
    targets = inn1["final_score"] + 1

    # 2nd innings: resume from the live snapshot when the match is currently in innings 2.
    inn2_start = None
    if live_innings == 2:
        inn2_start = {
            "runs": live_state["runs"],
            "wickets": live_state["wickets"],
            "legal_balls": live_state["legal_balls"],
        }

    inn2 = simulate_innings(
        model=model,
        lineup=second_team["lineup"],
        attack=first_team["attack"],
        venue_avg_score=venue_avg_score,
        n_sims=n_sims,
        total_overs=tot_overs,
        balls_per_over=bpo,
        format_type=fmt,
        innings=2,
        target=targets,
        rng=rng,
        start_state=inn2_start,
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
        # Per-bowler wickets, plus the bowling side's names to label each column. Inn1 is
        # bowled by the 2nd team's attack, inn2 by the 1st team's attack.
        "inn1_bowler_wkts": inn1["per_bowler_wkts"],
        "inn2_bowler_wkts": inn2["per_bowler_wkts"],
        "inn1_bowling_names": [p.get("name", f"Bowler_{i+1}") for i, p in enumerate(second_team["attack"])],
        "inn2_bowling_names": [p.get("name", f"Bowler_{i+1}") for i, p in enumerate(first_team["attack"])],
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
    from src.data_processing.format_spec import FORMAT_SPECS

    p_dir = Path(processed_dir)
    fmt_clean = str(format_type).lower().strip()
    if fmt_clean in ["od", "odi"]:
        fmt = "od"
    elif fmt_clean in ["hnd", "hundred", "100"]:
        fmt = "hnd"
    elif fmt_clean in FORMAT_SPECS:
        # Any other known format code (t20w, and future odw/hndw) passes through unchanged --
        # was falling into the "else: t20" default below, which would have silently redirected
        # every t20w squad lookup to read men's t20 parquets instead of t20w's own.
        fmt = fmt_clean
    else:
        fmt = "t20"

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
    default_venue_score = float(spec(fmt)["default_venue_score"])
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
    # Fallback for named players with no row for THIS match_id — e.g. a batter who was named
    # in the XI but never reached the crease because the innings ended first. Without this,
    # such players silently default to a neutral 1500 and can crater a lineup's simulated
    # strength even though they have a perfectly good rating from an earlier match. Filled
    # below only for players who actually need it, using each one's most recent role rating
    # from strictly BEFORE this match (never a later one, to avoid leaking future form).
    prior_batter_elo = {}
    prior_bowler_elo = {}
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

        named_players = set(t1_players) | set(t2_players)
        missing_batters = named_players - set(batter_map)
        missing_bowlers = named_players - set(bowler_map)
        match_date_ts = pd.to_datetime(row.get("date"), errors="coerce")
        if (missing_batters or missing_bowlers) and pd.notna(match_date_ts):
            needed = missing_batters | missing_bowlers
            p_hist = p_df[p_df["player"].isin(needed)].copy()
            if not p_hist.empty:
                date_map = dict(zip(meta_df["match_id"].astype(str), pd.to_datetime(meta_df["date"], errors="coerce")))
                p_hist["match_date"] = p_hist["match_id"].astype(str).map(date_map)
                p_hist = p_hist[p_hist["match_date"] < match_date_ts].sort_values("match_date")

                def _latest_prior(name, role):
                    rows = p_hist[(p_hist["player"] == name) & (p_hist["role"] == role)]
                    if rows.empty:
                        return None
                    last = rows.iloc[-1]
                    return {
                        "effective_elo": float(last["effective_elo"]),
                        "recent_form": float(last.get("recent_form", 0.0)),
                    }

                for p_name in missing_batters:
                    prior = _latest_prior(p_name, "batter")
                    if prior is not None:
                        prior_batter_elo[p_name] = prior
                for p_name in missing_bowlers:
                    prior = _latest_prior(p_name, "bowler")
                    if prior is not None:
                        prior_bowler_elo[p_name] = prior

    def _build_team_dict(team_name: str, player_list: list) -> Dict[str, Any]:
        # Batting order as listed; carry each player's BATTING Elo/form. Prefer this match's
        # own rating, then the player's most recent prior rating, then a neutral default.
        lineup = []
        for p in player_list[:11]:
            info = batter_map.get(p) or prior_batter_elo.get(p) or {"effective_elo": 1500.0, "recent_form": 0.0}
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
            info = bowler_map.get(name) or prior_bowler_elo.get(name) or {"effective_elo": 1500.0, "recent_form": 0.0}
            attack.append({"name": name, "effective_elo": info["effective_elo"], "recent_form": info["recent_form"]})

        return {"name": team_name, "lineup": lineup, "attack": attack}

    t1_dict = _build_team_dict(t1_name, t1_players)
    t2_dict = _build_team_dict(t2_name, t2_players)

    return t1_dict, t2_dict, venue_score
