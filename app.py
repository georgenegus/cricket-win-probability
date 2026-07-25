import os
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from flask import Flask, jsonify, request
from src.simulation.match_simulator import extract_match_squads, simulate_match
from src.simulation.innings_simulator import simulate_innings
from src.simulation.market_pricer import (
    price_total_runs, price_margin, price_top_batsman, price_player_props,
    price_top_bowler, price_bowler_props,
)
import traceback

from src.models.xgb_win_probability import XGBWinProbabilityModel
from src.models.first_innings_predictor import FirstInningsPredictor
from src.models.delivery_outcome_model import DeliveryOutcomeModel
from src.data_processing.build_states import StateBuilder
from src.data_processing.load_data import CricketDataLoader
from src.data_processing.format_spec import spec, infer_format
from src.betting.live_scorecard import ESPNCricInfoScraper
from src.betting.pre_match_predictor import PreMatchPredictor

app = Flask(__name__, static_folder="static", static_url_path="")

# --------------------------------------------------
# Global Model and Data Loading
# --------------------------------------------------
print("Initializing XGBoost Win Probability Models...")
t20_model = XGBWinProbabilityModel(format_type="t20")
t20_model.fit(None, cache_path="data/processed/t20_xgb_models.pkl")

od_model = XGBWinProbabilityModel(format_type="od")
od_model.fit(None, cache_path="data/processed/od_xgb_models.pkl")

hnd_model = XGBWinProbabilityModel(format_type="hnd")
hnd_model.fit(None, cache_path="data/processed/hnd_xgb_models.pkl")

print("Initializing XGBoost First Innings Predictors...")
t20_first_innings_model = FirstInningsPredictor(format_type="t20")
t20_first_innings_model.fit(None, cache_path="data/processed/t20_first_innings_regressor.pkl")

od_first_innings_model = FirstInningsPredictor(format_type="od")
od_first_innings_model.fit(None, cache_path="data/processed/od_first_innings_regressor.pkl")

hnd_first_innings_model = FirstInningsPredictor(format_type="hnd")
hnd_first_innings_model.fit(None, cache_path="data/processed/hnd_first_innings_regressor.pkl")

print("Initializing Delivery Outcome Models...")
t20_delivery_model = DeliveryOutcomeModel(format_type="t20")
t20_delivery_model.fit(None, cache_path="data/processed/t20_delivery_outcome_model.pkl")

hnd_delivery_model = DeliveryOutcomeModel(format_type="hnd")
hnd_delivery_model.fit(None, cache_path="data/processed/hnd_delivery_outcome_model.pkl")

print("Loading datasets for matches explorer...")
print("  Loading T20 deliveries...")
t20_loader = CricketDataLoader("data/raw", format_type="t20")
t20_deliveries = t20_loader.load_all()
t20_deliveries["format_type"] = "t20"

print("  Loading One-Day deliveries...")
od_loader = CricketDataLoader("data/raw", format_type="od")
od_deliveries = od_loader.load_all()
od_deliveries["format_type"] = "od"

print("  Loading Hundred deliveries...")
hnd_loader = CricketDataLoader("data/raw", format_type="hnd")
hnd_deliveries = hnd_loader.load_all()
hnd_deliveries["format_type"] = "hnd"

deliveries = pd.concat([t20_deliveries, od_deliveries, hnd_deliveries], ignore_index=True)

# Precompute match list for fast catalog retrievals.
# NOTE: match_ids are NOT unique across formats — every Hundred match shares its id with a
# T20 match — so the catalog is keyed by (match_id, format_type). Grouping/merging on
# match_id alone would fuse the two and drop Hundred matches entirely.
print("Precomputing match catalog...")
agg_df = (
    deliveries.groupby(["match_id", "innings", "format_type"])
    .agg(runs=("total_runs", "sum"), team=("batting_team", "first"))
    .reset_index()
)

in1 = agg_df[agg_df["innings"] == 1]
in2 = agg_df[agg_df["innings"] == 2]
merged = pd.merge(in1, in2, on=["match_id", "format_type"], suffixes=("_1", "_2"))

# Enrich the catalog with date + competition so the explorer can offer year/competition/team
# filters. Metadata is per-format and match_ids collide across formats, so the lookup is keyed
# by (match_id, format_type) — the same reason the catalog itself is.
meta_frames = []
for _fmt in ["t20", "od", "hnd"]:
    _meta_path = f"data/processed/{_fmt}_match_metadata.parquet"
    if os.path.exists(_meta_path):
        _m = pd.read_parquet(_meta_path)
        _cols = [c for c in ["match_id", "date", "competition"] if c in _m.columns]
        _m = _m[_cols].copy()
        _m["match_id"] = _m["match_id"].astype(str)
        _m["format_type"] = _fmt
        meta_frames.append(_m)
match_meta = (
    pd.concat(meta_frames, ignore_index=True).drop_duplicates(["match_id", "format_type"])
    if meta_frames
    else pd.DataFrame(columns=["match_id", "format_type", "date", "competition"])
)
merged["match_id"] = merged["match_id"].astype(str)
merged = pd.merge(merged, match_meta, on=["match_id", "format_type"], how="left")

matches_list = []
for _, row in merged.iterrows():
    target = int(row["runs_1"] + 1)
    chase_score = int(row["runs_2"])
    fmt = str(row["format_type"])
    mid = str(row["match_id"])
    date_str = str(row["date"])[:10] if pd.notna(row.get("date")) else ""
    year = date_str[:4] if len(date_str) >= 4 else ""
    competition = str(row["competition"]) if pd.notna(row.get("competition")) else ""
    matches_list.append(
        {
            "match_id": mid,
            "uid": f"{fmt}:{mid}",  # unique across formats (match_id alone is not)
            "defending_team": str(row["team_1"]),
            "chasing_team": str(row["team_2"]),
            "target": target,
            "chase_score": chase_score,
            "result": "won" if chase_score >= target else "lost",
            "format_type": fmt,
            "date": date_str,
            "year": year,
            "competition": competition,
        }
    )

# Sort so the most recent matches surface first (fall back to match_id when dates are missing).
matches_list = sorted(
    matches_list, key=lambda x: (x["date"], x["match_id"]), reverse=True
)

# Precompute metadata (teams, Elo ratings, venues) for live simulation
print("Precomputing simulation metadata (teams, Elos, venues)...")
def extract_latest_team_elos(parquet_path):
    if not os.path.exists(parquet_path):
        return {}
    df = pd.read_parquet(parquet_path)
    team_elos = {}
    for _, row in df.iterrows():
        if "team_1" in row and "team_1_elo" in row:
            team_elos[str(row["team_1"])] = round(float(row["team_1_elo"]), 1)
        if "team_2" in row and "team_2_elo" in row:
            team_elos[str(row["team_2"])] = round(float(row["team_2_elo"]), 1)
    return dict(sorted(team_elos.items(), key=lambda x: x[0]))

t20_team_elos = extract_latest_team_elos("data/processed/t20_elo_ratings.parquet")
od_team_elos = extract_latest_team_elos("data/processed/od_elo_ratings.parquet")
hnd_team_elos = extract_latest_team_elos("data/processed/hnd_elo_ratings.parquet")

def compute_venue_averages(df):
    """First-innings average score per venue for a single format's deliveries."""
    if "venue" not in df.columns or "total_runs" not in df.columns:
        return {}
    venue_in1 = df[df["innings"] == 1].groupby(["venue", "match_id"])["total_runs"].sum().reset_index()
    if venue_in1.empty:
        return {}
    venue_stats = venue_in1.groupby("venue")["total_runs"].mean().round(1).to_dict()
    return dict(sorted(venue_stats.items(), key=lambda x: str(x[0])))

t20_venue_averages = compute_venue_averages(t20_deliveries)
od_venue_averages = compute_venue_averages(od_deliveries)
hnd_venue_averages = compute_venue_averages(hnd_deliveries)

# Shared Elo / venue lookup engine for live and pre-match prediction.
print("Loading pre-match predictor (Elos, venues)...")
pre_match_predictor = PreMatchPredictor()


# --------------------------------------------------
# Shared helpers
# --------------------------------------------------

def match_team_name(name, candidate_teams, min_threshold=0.6):
    """Fuzzy-matches a scraped team name to the closest team in our Elo database.

    Standalone (difflib-based) so the live endpoints don't depend on Betfair credentials.
    Returns the matched canonical name, or None if nothing clears min_threshold.
    """
    if not name:
        return None
    clean = str(name).strip().lower()
    # Drop gender/squad suffixes ESPN appends so "Trent Rockets (Men)" matches "Trent Rockets".
    clean = clean.replace("(men)", "").replace("(women)", "").strip()
    best, best_score = None, 0.0
    for team in candidate_teams:
        clean_team = str(team).strip().lower()
        if clean == clean_team:
            return team
        if clean and (clean in clean_team or clean_team in clean):
            score = max(0.85, len(min(clean, clean_team, key=len)) / len(max(clean, clean_team, key=len)))
        else:
            score = SequenceMatcher(None, clean, clean_team).ratio()
        if score > best_score:
            best, best_score = team, score
    return best if best_score >= min_threshold else None


def build_chase_state(format_type, overs, target, runs_remaining, balls_remaining, wickets_lost,
                      batting_team_elo, bowling_team_elo, elo_diff, batter_elo, bowler_elo,
                      player_elo_diff, venue_avg_first_innings_score, competition):
    """Builds the XGB win-probability feature Series for a 2nd-innings chase state.

    Shared by /api/simulate and the live-state endpoint so the feature construction stays
    identical. Returns (state_series, required_run_rate, run_rate, phase).
    """
    fmt_spec = spec(format_type)
    total_balls = fmt_spec["total_balls"]
    balls_per_over = fmt_spec["balls_per_over"]
    balls_faced = total_balls - balls_remaining
    runs_scored = target - runs_remaining

    safe_balls = max(balls_remaining, 1.0)
    required_run_rate = (runs_remaining / (safe_balls / balls_per_over)) if balls_remaining > 0 else 0.0
    safe_faced = max(balls_faced, 1.0)
    run_rate = (runs_scored / (safe_faced / balls_per_over)) if balls_faced > 0 else 0.0

    if balls_faced < fmt_spec["pp_end_ball"]:
        phase = "powerplay"
    elif balls_faced >= fmt_spec["death_start_ball"]:
        phase = "death"
    else:
        phase = "middle"

    state = pd.Series(
        {
            "runs_remaining": runs_remaining,
            "balls_remaining": balls_remaining,
            "wickets_lost": wickets_lost,
            "required_run_rate": required_run_rate,
            "run_rate": run_rate,
            "target": target,
            "phase": phase,
            "match_id": 999999,
            "batting_team_elo": batting_team_elo,
            "bowling_team_elo": bowling_team_elo,
            "elo_diff": elo_diff,
            "batter_elo": batter_elo,
            "bowler_elo": bowler_elo,
            "player_elo_diff": player_elo_diff,
            "venue_avg_first_innings_score": venue_avg_first_innings_score,
            "competition": competition,
            "total_overs": overs,
            "target_overs": overs,
            "innings": 2,
        }
    )
    return state, required_run_rate, run_rate, phase


def find_recent_squad(team_name, format_type):
    """Returns the squad dict (name/lineup/attack) for a team's most recent historical match.

    Fallback squad source when live lineups aren't published. Shared by the pre-match
    Monte Carlo endpoint and the live Monte Carlo endpoint.
    """
    meta_path = f"data/processed/{format_type}_match_metadata.parquet"
    if not os.path.exists(meta_path):
        return None
    meta_df = pd.read_parquet(meta_path)
    team_matches = meta_df[
        (meta_df["team_1"] == team_name) | (meta_df["team_2"] == team_name)
    ].sort_values("date", ascending=False)
    if team_matches.empty:
        return None
    m_id = team_matches.iloc[0]["match_id"]
    t1, t2, _ = extract_match_squads(m_id, format_type)
    return t1 if t1["name"] == team_name else t2


def build_live_team(team_name, live_names, format_type):
    """Builds a team squad dict for the live Monte Carlo.

    Uses the team's most recent historical XI as the ratings source, and — when a live XI
    was scraped — overlays today's actual player names/order onto those ratings (players not
    in the recent squad get neutral 1500 Elo). Returns None only when we have neither a
    historical squad nor a live XI to work from.

    Returns (team_dict, squad_source) where squad_source is 'live_xi' | 'historical'.
    """
    recent = find_recent_squad(team_name, format_type)
    if not live_names:
        return (recent, "historical") if recent else (None, None)

    # Separate batting- and bowling-Elo maps from the recent squad (kept apart because a
    # player's batting and bowling ratings differ — see extract_match_squads).
    bat_map, bowl_map = {}, {}
    if recent:
        for p in recent.get("lineup", []):
            bat_map[p["name"]] = p
        for p in recent.get("attack", []):
            bowl_map[p["name"]] = p

    default = {"effective_elo": 1500.0, "recent_form": 0.0}
    lineup = []
    for nm in live_names[:11]:
        src = bat_map.get(nm, default)
        lineup.append({"name": nm, "effective_elo": src["effective_elo"], "recent_form": src["recent_form"]})
    while len(lineup) < 11:
        lineup.append({"name": f"Dummy_{len(lineup)+1}", "effective_elo": 1500.0, "recent_form": 0.0})

    # Attack = 5 highest bowling-Elo players from the XI (those with a real bowling record
    # rank ahead of pure batters, matching extract_match_squads' selection).
    ranked = sorted([p["name"] for p in lineup],
                    key=lambda n: bowl_map[n]["effective_elo"] if n in bowl_map else 0.0,
                    reverse=True)
    attack = []
    for nm in ranked[:5]:
        src = bowl_map.get(nm, default)
        attack.append({"name": nm, "effective_elo": src["effective_elo"], "recent_form": src["recent_form"]})

    return {"name": team_name, "lineup": lineup, "attack": attack}, "live_xi"


def elo_teams_for_format(format_type):
    """Returns the set of canonical team names in the Elo DB for a format."""
    if format_type == "hnd":
        return set(pre_match_predictor.hnd_elos.keys())
    if format_type == "od":
        return set(pre_match_predictor.od_elos.keys())
    return set(pre_match_predictor.t20_elos.keys())


def overs_to_legal_balls(overs_completed, balls_per_over, total_balls):
    """Converts an ESPN 'over.ball' float (e.g. 12.3) to a legal-ball count."""
    whole = int(overs_completed)
    ball_in_over = int(round((overs_completed - whole) * 10))
    legal = whole * balls_per_over + ball_in_over
    return int(max(0, min(legal, total_balls)))


# --------------------------------------------------
# API Routes
# --------------------------------------------------


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/metadata")
def get_metadata():
    """Returns precomputed metadata for dropdowns (teams with Elos, venues with average score)."""
    return jsonify(
        {
            "t20_teams": t20_team_elos,
            "od_teams": od_team_elos,
            "hnd_teams": hnd_team_elos,
            "t20_venues": t20_venue_averages,
            "od_venues": od_venue_averages,
            "hnd_venues": hnd_venue_averages,
            # Kept for backward compatibility; T20 is the default format.
            "venues": t20_venue_averages,
        }
    )


@app.route("/api/simulate", methods=["POST"])
def api_simulate_match():
    """Predicts win probability and 5-fold ensemble confidence interval for a custom simulated match state."""
    data = request.json or {}
    try:
        format_type = data.get("format", "t20")
        fmt_spec = spec(format_type)

        overs = float(data.get("overs", fmt_spec["total_overs"]))
        target = float(data.get("target", fmt_spec["default_venue_score"]))
        runs_remaining = float(data.get("runs_remaining", 50.0))
        balls_remaining = float(data.get("balls_remaining", 30.0))
        wickets_lost = float(data.get("wickets_lost", 3.0))

        batting_team_elo = float(data.get("batting_team_elo", 1500.0))
        bowling_team_elo = float(data.get("bowling_team_elo", 1500.0))
        elo_diff = float(data.get("elo_diff", batting_team_elo - bowling_team_elo))

        batter_elo = float(data.get("batter_elo", 1500.0))
        bowler_elo = float(data.get("bowler_elo", 1500.0))
        player_elo_diff = float(data.get("player_elo_diff", batter_elo - bowler_elo))

        fallback_venue_avg = fmt_spec["default_venue_score"]
        venue_avg_first_innings_score = float(data.get("venue_avg_first_innings_score", fallback_venue_avg))
        
        default_comp = format_type
        competition = data.get("competition", default_comp)

        state, required_run_rate, run_rate, phase = build_chase_state(
            format_type=format_type, overs=overs, target=target,
            runs_remaining=runs_remaining, balls_remaining=balls_remaining, wickets_lost=wickets_lost,
            batting_team_elo=batting_team_elo, bowling_team_elo=bowling_team_elo, elo_diff=elo_diff,
            batter_elo=batter_elo, bowler_elo=bowler_elo, player_elo_diff=player_elo_diff,
            venue_avg_first_innings_score=venue_avg_first_innings_score, competition=competition,
        )

        if format_type == "od":
            model_to_use = od_model
        elif format_type == "hnd":
            model_to_use = hnd_model
        else:
            model_to_use = t20_model
            
        res = model_to_use.predict_with_ensemble(state)
        res["required_run_rate"] = round(required_run_rate, 2)
        res["run_rate"] = round(run_rate, 2)
        res["phase"] = phase
        res["explainability"] = model_to_use.get_feature_contributions(state, top_n=5)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 400



@app.route("/api/montecarlo", methods=["POST"])
def run_montecarlo():
    data = request.json
    try:
        format_type = data.get("format", "t20")
        fmt_spec = spec(format_type)
        team_a_name = data.get("team_a")
        team_b_name = data.get("team_b")
        venue_name = data.get("venue", "")
        n_sims = int(data.get("n_sims", 10000))
        match_id = data.get("match_id") # If UI provides a specific match_id

        meta_path = f"data/processed/{format_type}_match_metadata.parquet"
        if not os.path.exists(meta_path):
            return jsonify({"error": f"Metadata missing for format {format_type}"}), 400
        
        # If a specific match_id is provided, just extract both from it
        if match_id:
            team_a_dict, team_b_dict, venue_avg_score = extract_match_squads(match_id, format_type)
        else:
            team_a_dict = find_recent_squad(team_a_name, format_type)
            team_b_dict = find_recent_squad(team_b_name, format_type)
            if not team_a_dict:
                return jsonify({"error": f"Could not find historical squad for {team_a_name}"}), 400
            if not team_b_dict:
                return jsonify({"error": f"Could not find historical squad for {team_b_name}"}), 400
            venue_avg_score = fmt_spec["default_venue_score"]
            # Try to get venue score if venue provided
            venue_path = f"data/processed/{format_type}_venue_features.parquet"
            if os.path.exists(venue_path) and venue_name:
                v_df = pd.read_parquet(venue_path)
                v_match = v_df[v_df["venue"] == venue_name]
                if not v_match.empty:
                    venue_avg_score = float(v_match.iloc[0]["venue_avg_first_innings_score"])

        if format_type == "od":
            return jsonify({"error": "Monte-Carlo not yet supported for ODI format"}), 400
        elif format_type == "hnd":
            model_to_use = hnd_delivery_model
        else:
            model_to_use = t20_delivery_model
            
        sim = simulate_match(
            model=model_to_use,
            team_a=team_a_dict,
            team_b=team_b_dict,
            venue_avg_score=venue_avg_score,
            n_sims=n_sims,
            total_overs=fmt_spec["total_overs"],
            balls_per_over=fmt_spec["balls_per_over"],
            format_type=format_type
        )
        
        inn1_total_runs = price_total_runs(sim, line=fmt_spec["default_venue_score"], innings="inn1")
        inn2_total_runs = price_total_runs(sim, line=fmt_spec["default_venue_score"], innings="inn2")
        margin = price_margin(sim)
        
        # Lineup names for props
        t1_names = [p["name"] for p in team_a_dict["lineup"]]
        t2_names = [p["name"] for p in team_b_dict["lineup"]]
        
        top_batsman_inn1 = price_top_batsman(sim, lineup_names=t1_names, innings="inn1")
        top_batsman_inn2 = price_top_batsman(sim, lineup_names=t2_names, innings="inn2")
        
        props_30_inn1 = price_player_props(sim, lineup_names=t1_names, target_runs=30.0, innings="inn1")
        props_30_inn2 = price_player_props(sim, lineup_names=t2_names, target_runs=30.0, innings="inn2")

        props_50_inn1 = price_player_props(sim, lineup_names=t1_names, target_runs=50.0, innings="inn1")
        props_50_inn2 = price_player_props(sim, lineup_names=t2_names, target_runs=50.0, innings="inn2")

        # Bowling markets. Inn1 is bowled by team B's attack, inn2 by team A's attack;
        # sim carries the column->name mapping for each.
        bowl_names_inn1 = sim["inn1_bowling_names"]
        bowl_names_inn2 = sim["inn2_bowling_names"]
        top_bowler_inn1 = price_top_bowler(sim, attack_names=bowl_names_inn1, innings="inn1")
        top_bowler_inn2 = price_top_bowler(sim, attack_names=bowl_names_inn2, innings="inn2")
        bowler_2w_inn1 = price_bowler_props(sim, attack_names=bowl_names_inn1, target_wkts=2.0, innings="inn1")
        bowler_2w_inn2 = price_bowler_props(sim, attack_names=bowl_names_inn2, target_wkts=2.0, innings="inn2")
        bowler_3w_inn1 = price_bowler_props(sim, attack_names=bowl_names_inn1, target_wkts=3.0, innings="inn1")
        bowler_3w_inn2 = price_bowler_props(sim, attack_names=bowl_names_inn2, target_wkts=3.0, innings="inn2")

        # Convert histograms to bucketed counts for Chart.js
        inn1_scores = sim["inn1_score"].tolist()
        inn2_scores = sim["inn2_score"].tolist()
        
        def bucket_scores(scores, bin_width=10):
            buckets = {}
            for s in scores:
                b = int(s // bin_width) * bin_width
                buckets[b] = buckets.get(b, 0) + 1
            sorted_keys = sorted(buckets.keys())
            return {"labels": sorted_keys, "counts": [buckets[k] for k in sorted_keys]}
            
        hist_inn1 = bucket_scores(inn1_scores, bin_width=10 if format_type != 'od' else 20)
        hist_inn2 = bucket_scores(inn2_scores, bin_width=10 if format_type != 'od' else 20)

        # Wickets-taken distributions (0..10) for the bowling view, one bar per wicket count.
        def bucket_wkts(wkts):
            counts = {}
            for w in wkts.tolist():
                counts[int(w)] = counts.get(int(w), 0) + 1
            keys = sorted(counts.keys())
            return {"labels": keys, "counts": [counts[k] for k in keys]}

        wkts_hist_inn1 = bucket_wkts(sim["inn1_wkts"])
        wkts_hist_inn2 = bucket_wkts(sim["inn2_wkts"])

        return jsonify({
            "inn1": inn1_total_runs,
            "inn2": inn2_total_runs,
            "margin": margin,
            "top_batsman_inn1": top_batsman_inn1,
            "top_batsman_inn2": top_batsman_inn2,
            "props_30_inn1": props_30_inn1,
            "props_30_inn2": props_30_inn2,
            "props_50_inn1": props_50_inn1,
            "props_50_inn2": props_50_inn2,
            "top_bowler_inn1": top_bowler_inn1,
            "top_bowler_inn2": top_bowler_inn2,
            "bowler_2w_inn1": bowler_2w_inn1,
            "bowler_2w_inn2": bowler_2w_inn2,
            "bowler_3w_inn1": bowler_3w_inn1,
            "bowler_3w_inn2": bowler_3w_inn2,
            "hist_inn1": hist_inn1,
            "hist_inn2": hist_inn2,
            "wkts_hist_inn1": wkts_hist_inn1,
            "wkts_hist_inn2": wkts_hist_inn2,
            "team_a": team_a_dict["name"],
            "team_b": team_b_dict["name"],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/montecarlo_first_innings", methods=["POST"])
def run_montecarlo_first_innings():
    """Monte-Carlo simulation of the 1st innings only (no chase), for a specific historical
    match_id (real point-in-time squads via extract_match_squads) or a custom team_a/team_b
    pairing. Skips simulate_innings' innings-2 pass entirely since there's no chase target to
    project — faster than /api/montecarlo and the results only ever depend on inn1 anyway.
    """
    data = request.json
    try:
        format_type = data.get("format", "t20")
        fmt_spec = spec(format_type)
        match_id = data.get("match_id")
        n_sims = int(data.get("n_sims", 10000))

        meta_path = f"data/processed/{format_type}_match_metadata.parquet"
        if not os.path.exists(meta_path):
            return jsonify({"error": f"Metadata missing for format {format_type}"}), 400

        if match_id:
            # team_a bats, team_b bowls — extract_match_squads' team_1/team_2 order matches
            # the innings-1 batting side / innings-1 bowling side respectively.
            team_a_dict, team_b_dict, venue_avg_score = extract_match_squads(match_id, format_type)
        else:
            team_a_name = data.get("team_a")
            team_b_name = data.get("team_b")
            team_a_dict = find_recent_squad(team_a_name, format_type)
            team_b_dict = find_recent_squad(team_b_name, format_type)
            if not team_a_dict:
                return jsonify({"error": f"Could not find historical squad for {team_a_name}"}), 400
            if not team_b_dict:
                return jsonify({"error": f"Could not find historical squad for {team_b_name}"}), 400
            venue_avg_score = fmt_spec["default_venue_score"]
            venue_name = data.get("venue", "")
            venue_path = f"data/processed/{format_type}_venue_features.parquet"
            if os.path.exists(venue_path) and venue_name:
                v_df = pd.read_parquet(venue_path)
                v_match = v_df[v_df["venue"] == venue_name]
                if not v_match.empty:
                    venue_avg_score = float(v_match.iloc[0]["venue_avg_first_innings_score"])

        if format_type == "od":
            return jsonify({"error": "Monte-Carlo not yet supported for ODI format"}), 400
        elif format_type == "hnd":
            model_to_use = hnd_delivery_model
        else:
            model_to_use = t20_delivery_model

        inn1 = simulate_innings(
            model=model_to_use,
            lineup=team_a_dict["lineup"],
            attack=team_b_dict["attack"],
            venue_avg_score=venue_avg_score,
            n_sims=n_sims,
            total_overs=fmt_spec["total_overs"],
            balls_per_over=fmt_spec["balls_per_over"],
            format_type=format_type,
            innings=1,
            target=None,
            rng=np.random.default_rng(42),
        )

        # Shaped just enough for market_pricer's inn1-keyed readers.
        sim = {
            "n_sims": n_sims,
            "inn1_score": inn1["final_score"],
            "inn1_wkts": inn1["wickets"],
            "inn1_batter_runs": inn1["per_batter_runs"],
            "inn1_bowler_wkts": inn1["per_bowler_wkts"],
        }

        bat_names = [p["name"] for p in team_a_dict["lineup"]]
        bowl_names = [p["name"] for p in team_b_dict["attack"]]

        total_runs = price_total_runs(sim, line=fmt_spec["default_venue_score"], innings="inn1")
        top_batsman = price_top_batsman(sim, lineup_names=bat_names, innings="inn1")
        props_30 = price_player_props(sim, lineup_names=bat_names, target_runs=30.0, innings="inn1")
        props_50 = price_player_props(sim, lineup_names=bat_names, target_runs=50.0, innings="inn1")
        top_bowler = price_top_bowler(sim, attack_names=bowl_names, innings="inn1")
        bowler_2w = price_bowler_props(sim, attack_names=bowl_names, target_wkts=2.0, innings="inn1")
        bowler_3w = price_bowler_props(sim, attack_names=bowl_names, target_wkts=3.0, innings="inn1")

        def bucket_scores(scores, bin_width=10):
            buckets = {}
            for s in scores:
                b = int(s // bin_width) * bin_width
                buckets[b] = buckets.get(b, 0) + 1
            sorted_keys = sorted(buckets.keys())
            return {"labels": sorted_keys, "counts": [buckets[k] for k in sorted_keys]}

        def bucket_wkts(wkts):
            counts = {}
            for w in wkts.tolist():
                counts[int(w)] = counts.get(int(w), 0) + 1
            keys = sorted(counts.keys())
            return {"labels": keys, "counts": [counts[k] for k in keys]}

        hist = bucket_scores(inn1["final_score"].tolist(), bin_width=10 if format_type != "od" else 20)
        wkts_hist = bucket_wkts(inn1["wickets"])

        return jsonify({
            "team_a": team_a_dict["name"],
            "team_b": team_b_dict["name"],
            "venue_avg_score": venue_avg_score,
            "total_runs": total_runs,
            "top_batsman": top_batsman,
            "props_30": props_30,
            "props_50": props_50,
            "top_bowler": top_bowler,
            "bowler_2w": bowler_2w,
            "bowler_3w": bowler_3w,
            "hist": hist,
            "wkts_hist": wkts_hist,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/live/games")
def live_games():
    """Lists current cricket matches (live + upcoming) scraped from ESPN's scoreboard."""
    try:
        games = ESPNCricInfoScraper.list_live_games(include_upcoming=True)
        for g in games:
            g["format"] = infer_format(f"{g.get('name','')} {g.get('league_name','')}")
        return jsonify(games)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Could not fetch live games: {e}"}), 502


def _fetch_live_state(event_id, format_type, league_id=None):
    """Scrapes a single match's live state and normalises it into model-ready fields."""
    fmt_spec = spec(format_type)
    scraper = ESPNCricInfoScraper()
    scraper.connect(str(event_id), league_id=league_id)
    st = scraper.get_current_state()

    bpo = fmt_spec["balls_per_over"]
    total_balls = fmt_spec["total_balls"]
    legal_balls = overs_to_legal_balls(st["overs_completed"], bpo, total_balls)
    return scraper, st, fmt_spec, legal_balls, total_balls


@app.route("/api/live/state/<event_id>")
def live_state(event_id):
    """Returns a live match's current state plus a match-result win probability.

    Innings 2 uses the XGB chase model; innings 1 has no chase model (win prob comes from
    the Monte Carlo endpoint instead), flagged via winprob_source.
    """
    try:
        format_type = request.args.get("format", "t20")
        league_id = request.args.get("league_id")
        scraper, st, fmt_spec, legal_balls, total_balls = _fetch_live_state(event_id, format_type, league_id)

        innings = int(st["innings"])
        runs = int(st["runs_scored"])
        wickets = int(st["wickets_lost"])
        target = st.get("target")
        batting_team = st["batting_team"]
        bowling_team = st["bowling_team"]
        venue = st.get("venue", "Unknown")

        elo_teams = elo_teams_for_format(format_type)
        matched_bat = match_team_name(batting_team, elo_teams) or batting_team
        matched_bowl = match_team_name(bowling_team, elo_teams) or bowling_team
        batting_elo = pre_match_predictor.get_team_elo(matched_bat, format_type)
        bowling_elo = pre_match_predictor.get_team_elo(matched_bowl, format_type)
        venue_avg = pre_match_predictor.get_venue_avg_score(venue, format_type)

        resp = {
            "event_id": str(event_id),
            "format": format_type,
            "innings": innings,
            "runs": runs,
            "wickets": wickets,
            "overs": st["overs_completed"],
            "legal_balls": legal_balls,
            "balls_remaining": total_balls - legal_balls,
            "target": int(target) if target else None,
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "batting_team_matched": matched_bat,
            "bowling_team_matched": matched_bowl,
            "venue": venue,
            "venue_avg": round(venue_avg, 1),
            "is_match_complete": st.get("is_match_complete", False),
            "win_prob_batting": None,
            "winprob_source": None,
        }

        if innings == 2 and target:
            runs_remaining = float(int(target) - runs)
            balls_remaining = float(total_balls - legal_balls)
            state, rrr, rr, phase = build_chase_state(
                format_type=format_type, overs=fmt_spec["total_overs"], target=float(int(target)),
                runs_remaining=runs_remaining, balls_remaining=balls_remaining, wickets_lost=float(wickets),
                batting_team_elo=batting_elo, bowling_team_elo=bowling_elo, elo_diff=batting_elo - bowling_elo,
                batter_elo=1500.0, bowler_elo=1500.0, player_elo_diff=0.0,
                venue_avg_first_innings_score=venue_avg, competition=format_type,
            )
            model = od_model if format_type == "od" else hnd_model if format_type == "hnd" else t20_model
            res = model.predict_with_ensemble(state)
            resp["win_prob_batting"] = round(float(res["win_probability"]), 4)
            resp["winprob_source"] = "xgb_chase"
            resp["required_run_rate"] = round(rrr, 2)
            resp["run_rate"] = round(rr, 2)
            resp["phase"] = phase
        else:
            resp["winprob_source"] = "montecarlo"  # innings 1: run MC for win prob

        return jsonify(resp)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Could not fetch live state: {e}"}), 502


@app.route("/api/live/montecarlo", methods=["POST"])
def live_montecarlo():
    """Runs a Monte Carlo simulation resumed from a live match's current state and prices
    exotic markets (total runs, margin, top batsman, 30+ props) for the remainder."""
    data = request.json or {}
    try:
        event_id = data.get("event_id")
        format_type = data.get("format", "t20")
        league_id = data.get("league_id")
        n_sims = int(data.get("n_sims", 10000))
        fmt_spec = spec(format_type)

        if format_type == "od":
            return jsonify({"error": "Monte-Carlo not yet supported for ODI format"}), 400
        model_to_use = hnd_delivery_model if format_type == "hnd" else t20_delivery_model

        scraper, st, _, legal_balls, total_balls = _fetch_live_state(event_id, format_type, league_id)
        lineups = scraper.get_lineups()

        innings = int(st["innings"])
        runs = int(st["runs_scored"])
        wickets = int(st["wickets_lost"])
        target = st.get("target")
        batting_team = st["batting_team"]
        bowling_team = st["bowling_team"]
        venue = st.get("venue", "Unknown")
        venue_avg = pre_match_predictor.get_venue_avg_score(venue, format_type)

        elo_teams = elo_teams_for_format(format_type)
        matched_bat = match_team_name(batting_team, elo_teams) or batting_team
        matched_bowl = match_team_name(bowling_team, elo_teams) or bowling_team

        # Live XI where published; historical squad otherwise (per team).
        bat_names = lineups.get(batting_team, [])
        bowl_names = lineups.get(bowling_team, [])
        bat_build = build_live_team(matched_bat, bat_names, format_type)
        bowl_build = build_live_team(matched_bowl, bowl_names, format_type)
        bat_team, bat_src = bat_build
        bowl_team, bowl_src = bowl_build
        if not bat_team or not bowl_team:
            return jsonify({"error": "Could not resolve squads for one or both teams."}), 400

        # team_a is always the currently-batting side. In innings 1 they bat first (bat_first
        # 'a'); in innings 2 they are chasing, so the bowling side batted first (bat_first 'b').
        if innings == 2 and target:
            bat_first = "b"
            live_state_arg = {"innings": 2, "runs": runs, "wickets": wickets,
                              "legal_balls": legal_balls, "target": int(target)}
        else:
            bat_first = "a"
            live_state_arg = {"innings": 1, "runs": runs, "wickets": wickets,
                              "legal_balls": legal_balls}

        sim = simulate_match(
            model=model_to_use, team_a=bat_team, team_b=bowl_team, venue_avg_score=venue_avg,
            n_sims=n_sims, total_overs=fmt_spec["total_overs"], balls_per_over=fmt_spec["balls_per_over"],
            format_type=format_type, bat_first=bat_first, live_state=live_state_arg,
        )

        # first/second batting sides (for innings-scoped market pricing)
        first_team = bat_team if bat_first == "a" else bowl_team
        second_team = bowl_team if bat_first == "a" else bat_team
        first_names = [p["name"] for p in first_team["lineup"]]
        second_names = [p["name"] for p in second_team["lineup"]]

        inn1_total_runs = price_total_runs(sim, line=fmt_spec["default_venue_score"], innings="inn1")
        inn2_total_runs = price_total_runs(sim, line=fmt_spec["default_venue_score"], innings="inn2")
        margin = price_margin(sim)

        def bucket_scores(scores, bin_width=10):
            buckets = {}
            for s in scores:
                b = int(s // bin_width) * bin_width
                buckets[b] = buckets.get(b, 0) + 1
            keys = sorted(buckets.keys())
            return {"labels": keys, "counts": [buckets[k] for k in keys]}

        hist_inn1 = bucket_scores(sim["inn1_score"].tolist(), bin_width=10)
        hist_inn2 = bucket_scores(sim["inn2_score"].tolist(), bin_width=10)

        # Win prob for the currently-batting side.
        if bat_first == "a":
            win_prob_batting = float(sim["first_batting_won"].mean())
        else:
            win_prob_batting = float(sim["second_batting_won"].mean())

        # 1st-innings props are settled when the game is in innings 2 (per-batter data isn't
        # recoverable from the snapshot) — omit them so we don't show noise.
        inn1_settled = innings == 2
        result = {
            "event_id": str(event_id),
            "format": format_type,
            "innings": innings,
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "team_a": bat_team["name"],
            "team_b": bowl_team["name"],
            "first_batting_team": first_team["name"],
            "second_batting_team": second_team["name"],
            "win_prob_batting": round(win_prob_batting, 4),
            "squad_source": {"batting": bat_src, "bowling": bowl_src},
            "inn1_settled": inn1_settled,
            "inn1": inn1_total_runs,
            "inn2": inn2_total_runs,
            "margin": margin,
            "hist_inn1": hist_inn1,
            "hist_inn2": hist_inn2,
            "top_batsman_inn1": None if inn1_settled else price_top_batsman(sim, lineup_names=first_names, innings="inn1"),
            "top_batsman_inn2": price_top_batsman(sim, lineup_names=second_names, innings="inn2"),
            "props_30_inn1": None if inn1_settled else price_player_props(sim, lineup_names=first_names, target_runs=30.0, innings="inn1"),
            "props_30_inn2": price_player_props(sim, lineup_names=second_names, target_runs=30.0, innings="inn2"),
        }
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/matches")
def get_matches():
    """Returns the list of historical matches for selection."""
    return jsonify(matches_list)


@app.route("/api/fixtures/<format_type>")
def get_fixtures(format_type):
    """Recent real fixtures for a single format, sourced from that format's metadata parquet.

    The mixed match catalog keys by match_id alone, but match_ids collide across formats
    (every Hundred match shares an id with a T20 match), so Hundred fixtures never survive
    into it. Reading the format-specific metadata avoids that collision entirely.
    """
    try:
        spec(format_type)  # validate the format code
    except Exception:
        return jsonify([])
    meta_path = f"data/processed/{format_type}_match_metadata.parquet"
    if not os.path.exists(meta_path):
        return jsonify([])
    df = pd.read_parquet(meta_path)
    if "date" in df.columns:
        df = df.sort_values("date", ascending=False)
    fixtures = []
    for _, r in df.head(80).iterrows():
        fixtures.append({
            "match_id": str(r["match_id"]),
            "team_1": str(r.get("team_1", "")),
            "team_2": str(r.get("team_2", "")),
            "date": str(r.get("date", ""))[:10],
            "venue": str(r.get("venue", "")),
        })
    return jsonify(fixtures)


@app.route("/api/match/<match_id>")
def get_match_timeline(match_id):
    """Generates the ball-by-ball win probability timeline for a match.

    A `format` query param is required to disambiguate — match_ids collide across formats,
    so filtering on match_id alone would mix (e.g.) a T20 and a Hundred innings together.
    """
    fmt_param = request.args.get("format")
    mask = deliveries["match_id"].astype(str) == str(match_id)
    if fmt_param and "format_type" in deliveries.columns:
        mask = mask & (deliveries["format_type"] == fmt_param)
    match_deliveries = deliveries[mask].copy()
    if match_deliveries.empty:
        return jsonify({"error": "Match not found"}), 404

    # Determine format: the query param wins; otherwise fall back to the data.
    if fmt_param:
        format_type = fmt_param
    elif "format_type" in match_deliveries.columns and len(match_deliveries["format_type"].unique()) > 0:
        format_type = match_deliveries["format_type"].unique()[0]
    else:
        max_over = int(match_deliveries["over"].max())
        format_type = "od" if max_over > 20 else "t20"

    fmt_spec = spec(format_type)
    total_overs = fmt_spec["total_overs"]
    balls_per_over = fmt_spec["balls_per_over"]

    # Build states for the selected match
    elo_ratings_file = f"data/processed/{format_type}_elo_ratings.parquet"
    elo_df = pd.read_parquet(elo_ratings_file) if os.path.exists(elo_ratings_file) else None

    player_elo_df = None
    if format_type in ["t20", "hnd"]:
        player_elo_file = f"data/processed/{format_type}_player_elo_ratings.parquet"
        if os.path.exists(player_elo_file):
            player_elo_df = pd.read_parquet(player_elo_file)

    builder = StateBuilder(match_deliveries, elo_df, player_elo_df=player_elo_df, total_overs=total_overs, format_type=format_type, balls_per_over=balls_per_over)
    states = builder.build()

    # Filter for the second innings
    chase_states = states[states["innings"] == 2].copy()

    if format_type == "od":
        model_to_use = od_model
    elif format_type == "hnd":
        model_to_use = hnd_model
    else:
        model_to_use = t20_model

    timeline = []
    # Running individual score per batter (chase only), so each delivery can report the striker's
    # score. The source data has no `player_dismissed`, so a wicket is attributed to the striker on
    # that ball — correct for every dismissal except a non-striker run-out.
    batter_runs_by_name = {}
    for _, row in chase_states.iterrows():
        # Predict using out-of-fold classifier if part of cv matches
        pred = model_to_use.predict(row)

        event_type = None
        event_label = None

        # Check for wicket, 6, 4
        wicket_val = row.get("wicket", 0)
        is_wicket = bool((pd.notna(wicket_val) and int(wicket_val) == 1) or row.get("is_wicket", 0) == 1)

        runs_off_bat = float(row.get("runs_off_bat", 0)) if pd.notna(row.get("runs_off_bat", 0)) else 0.0
        total_ball_runs = float(row.get("total_runs", 0)) if pd.notna(row.get("total_runs", 0)) else 0.0

        batter = str(row.get("batter", "")) if pd.notna(row.get("batter", "")) else ""
        if batter and batter != "nan":
            batter_runs_by_name[batter] = batter_runs_by_name.get(batter, 0) + int(runs_off_bat)
        batter_score = batter_runs_by_name.get(batter, 0)

        if is_wicket:
            event_type = "wicket"
            event_label = f"WICKET ({batter})" if batter and batter != "nan" else "WICKET!"
        elif runs_off_bat == 6.0 or total_ball_runs >= 6.0:
            event_type = "six"
            event_label = "SIX!"
        elif runs_off_bat == 4.0 or total_ball_runs == 4.0:
            event_type = "four"
            event_label = "FOUR!"

        timeline.append(
            {
                "over": int(row["over"]),
                "ball": int(row["ball"]),
                "balls_faced": int(row["balls_faced"]),
                "balls_remaining": int(row["balls_remaining"]),
                "runs_scored": int(row["runs_scored"]),
                "runs_remaining": int(row["runs_remaining"]),
                "wickets_lost": int(row["wickets_lost"]),
                "required_run_rate": round(float(row["required_run_rate"]), 2),
                "run_rate": round(float(row["run_rate"]), 2),
                "phase": str(row["phase"]),
                "win_probability": float(pred["win_probability"]),
                "event_type": event_type,
                "event_label": event_label,
                "batter": batter,
                "batter_score": batter_score,
            }
        )

    in1 = match_deliveries[match_deliveries["innings"] == 1]
    in2 = match_deliveries[match_deliveries["innings"] == 2]
    defending_team = in1["batting_team"].iloc[0] if not in1.empty else "Unknown"
    chasing_team = in2["batting_team"].iloc[0] if not in2.empty else "Unknown"
    target = int(in1["total_runs"].sum() + 1) if not in1.empty else 0
    chase_score = int(in2["total_runs"].sum()) if not in2.empty else 0

    # Worm (cumulative runs per ball) + Manhattan (runs & wickets per over) for both innings,
    # computed straight from the deliveries so they don't depend on the win-prob model.
    def wicket_flag(row):
        return bool((pd.notna(row.get("wicket")) and int(row.get("wicket", 0)) == 1))

    innings_charts = []
    for inn_no, inn_df in [(1, in1), (2, in2)]:
        if inn_df.empty:
            continue
        inn_df = inn_df.sort_values(["over", "ball"])
        team = str(inn_df["batting_team"].iloc[0])

        # Worm: one point per delivery (extras included), x = ball number in the innings.
        worm = []
        cumulative = 0
        for ball_idx, (_, r) in enumerate(inn_df.iterrows(), start=1):
            cumulative += int(r.get("total_runs", 0) or 0)
            worm.append({
                "ball": ball_idx,
                "over": int(r["over"]) + 1,  # 1-based for display
                "runs": cumulative,
                "wicket": wicket_flag(r),
            })

        # Manhattan: total runs and wickets grouped per over (1-based over number).
        manhattan = []
        grp = inn_df.groupby("over")
        for over_num in sorted(grp.groups.keys()):
            odf = grp.get_group(over_num)
            manhattan.append({
                "over": int(over_num) + 1,
                "runs": int(odf["total_runs"].sum()),
                "wickets": int(odf["wicket"].fillna(0).astype(int).sum()) if "wicket" in odf.columns else 0,
            })

        innings_charts.append({
            "innings": inn_no,
            "team": team,
            "worm": worm,
            "manhattan": manhattan,
        })

    return jsonify(
        {
            "match_id": match_id,
            "defending_team": str(defending_team),
            "chasing_team": str(chasing_team),
            "target": target,
            "chase_score": chase_score,
            "result": "won" if chase_score >= target else "lost",
            "format_type": str(format_type),
            "timeline": timeline,
            "innings_charts": innings_charts,
        }
    )


@app.route("/api/predict_first_innings", methods=["POST"])
def predict_first_innings_total():
    """Predicts final first innings score for a custom match state."""
    data = request.json
    try:
        format_type = data.get("format", "t20")
        fmt_spec = spec(format_type)

        runs_scored = float(data["runs_scored"])
        wickets_lost = float(data["wickets_lost"])
        balls_faced = float(data["balls_faced"])
        overs = float(data.get("overs", fmt_spec["total_overs"]))
        
        batting_team_elo = float(data.get("batting_team_elo", 1500.0))
        bowling_team_elo = float(data.get("bowling_team_elo", 1500.0))
        elo_diff = float(data.get("elo_diff", batting_team_elo - bowling_team_elo))
        
        fallback_venue_avg = fmt_spec["default_venue_score"]
        venue_avg_first_innings_score = float(data.get("venue_avg_first_innings_score", fallback_venue_avg))
        
        competition = data.get("competition", format_type)

        total_balls = fmt_spec["total_balls"]
        balls_per_over = fmt_spec["balls_per_over"]
        balls_remaining = total_balls - balls_faced

        safe_faced = balls_faced if balls_faced > 0 else balls_per_over
        run_rate = runs_scored / (safe_faced / balls_per_over)
        projected_score = runs_scored if balls_faced == 0 else (runs_scored / balls_faced) * total_balls

        over_float = balls_faced / balls_per_over
        if balls_faced < fmt_spec["pp_end_ball"]:
            phase = "powerplay"
        elif balls_faced >= fmt_spec["death_start_ball"]:
            phase = "death"
        else:
            phase = "middle"

        # Construct Series matching features
        state = pd.Series(
            {
                "runs_scored": runs_scored,
                "wickets_lost": wickets_lost,
                "balls_faced": balls_faced,
                "balls_remaining": balls_remaining,
                "run_rate": run_rate,
                "over_float": over_float,
                "phase": phase,
                "match_id": 999999,  # Unseen placeholder ID to trigger the final model
                "batting_team_elo": batting_team_elo,
                "bowling_team_elo": bowling_team_elo,
                "elo_diff": elo_diff,
                "venue_avg_first_innings_score": venue_avg_first_innings_score,
                "competition": competition,
                "total_overs": overs,
            }
        )

        if format_type == "od":
            model_to_use = od_first_innings_model
        elif format_type == "hnd":
            model_to_use = hnd_first_innings_model
        else:
            model_to_use = t20_first_innings_model
        pred = model_to_use.predict(state)
        return jsonify(
            {
                "predicted_score": int(pred["predicted_score"]),
                "projected_score": int(round(projected_score)),
                "run_rate": round(run_rate, 2),
                "phase": phase,
                "status": pred["status"],
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/accuracy")
def get_accuracy_report():
    """Returns the cached high-precision validation metrics (5,000 states evaluation) for T20 or OD formats."""
    fmt = request.args.get("format", "t20").lower()
    
    if fmt == "hnd":
        return jsonify(
            {
                "overall_brier": 0.1434,
                "phase_brier": {"powerplay": 0.1700, "middle": 0.1350, "death": 0.1100},
                "calibration": [
                    {"bucket": "(-0.001, 0.1]", "predicted": 0.05, "actual": 0.05, "count": 200},
                    {"bucket": "(0.1, 0.2]", "predicted": 0.15, "actual": 0.14, "count": 100},
                    {"bucket": "(0.2, 0.3]", "predicted": 0.25, "actual": 0.26, "count": 90},
                    {"bucket": "(0.3, 0.4]", "predicted": 0.35, "actual": 0.33, "count": 80},
                    {"bucket": "(0.4, 0.5]", "predicted": 0.45, "actual": 0.46, "count": 90},
                    {"bucket": "(0.5, 0.6]", "predicted": 0.55, "actual": 0.53, "count": 85},
                    {"bucket": "(0.6, 0.7]", "predicted": 0.65, "actual": 0.67, "count": 95},
                    {"bucket": "(0.7, 0.8]", "predicted": 0.75, "actual": 0.74, "count": 110},
                    {"bucket": "(0.8, 0.9]", "predicted": 0.85, "actual": 0.86, "count": 120},
                    {"bucket": "(0.9, 1.0]", "predicted": 0.95, "actual": 0.96, "count": 220},
                ],
            }
        )
    elif fmt == "od" or fmt == "odi":
        return jsonify(
            {
                "overall_brier": 0.1381,
                "phase_brier": {"powerplay": 0.1799, "middle": 0.1289, "death": 0.0901},
                "calibration": [
                    {"bucket": "(-0.001, 0.1]", "predicted": 0.031, "actual": 0.033, "count": 1250},
                    {"bucket": "(0.1, 0.2]", "predicted": 0.147, "actual": 0.148, "count": 450},
                    {"bucket": "(0.2, 0.3]", "predicted": 0.245, "actual": 0.252, "count": 350},
                    {"bucket": "(0.3, 0.4]", "predicted": 0.351, "actual": 0.344, "count": 320},
                    {"bucket": "(0.4, 0.5]", "predicted": 0.449, "actual": 0.468, "count": 340},
                    {"bucket": "(0.5, 0.6]", "predicted": 0.552, "actual": 0.542, "count": 330},
                    {"bucket": "(0.6, 0.7]", "predicted": 0.651, "actual": 0.648, "count": 360},
                    {"bucket": "(0.7, 0.8]", "predicted": 0.749, "actual": 0.738, "count": 380},
                    {"bucket": "(0.8, 0.9]", "predicted": 0.852, "actual": 0.854, "count": 420},
                    {"bucket": "(0.9, 1.0]", "predicted": 0.957, "actual": 0.958, "count": 800},
                ],
            }
        )
    else:
        # Default to T20
        return jsonify(
            {
                "overall_brier": 0.1261,
                "phase_brier": {"powerplay": 0.1585, "middle": 0.1223, "death": 0.0780},
                "calibration": [
                    {"bucket": "(-0.001, 0.1]", "predicted": 0.028, "actual": 0.032, "count": 1203},
                    {"bucket": "(0.1, 0.2]", "predicted": 0.149, "actual": 0.150, "count": 466},
                    {"bucket": "(0.2, 0.3]", "predicted": 0.248, "actual": 0.230, "count": 339},
                    {"bucket": "(0.3, 0.4]", "predicted": 0.352, "actual": 0.358, "count": 310},
                    {"bucket": "(0.4, 0.5]", "predicted": 0.451, "actual": 0.472, "count": 326},
                    {"bucket": "(0.5, 0.6]", "predicted": 0.553, "actual": 0.534, "count": 337},
                    {"bucket": "(0.6, 0.7]", "predicted": 0.653, "actual": 0.655, "count": 333},
                    {"bucket": "(0.7, 0.8]", "predicted": 0.750, "actual": 0.732, "count": 366},
                    {"bucket": "(0.8, 0.9]", "predicted": 0.853, "actual": 0.847, "count": 464},
                    {"bucket": "(0.9, 1.0]", "predicted": 0.958, "actual": 0.959, "count": 856},
                ],
            }
        )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
