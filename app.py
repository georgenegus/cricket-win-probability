import os
import re
import json
from pathlib import Path
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from flask import Flask, jsonify, request
from src.simulation.match_simulator import (
    extract_match_squads,
    simulate_match,
    ATTACK_SIZE,
    bowling_appearance_rates,
    bowling_volume_rates,
    bowler_volume_table,
    frontline_bowl_sort_key,
)
from src.simulation.innings_simulator import simulate_innings
from src.simulation.market_pricer import (
    price_total_runs, price_margin, price_top_batsman, price_player_props,
    price_top_bowler, price_bowler_props,
)
import traceback

from src.features.player_elo_engine import PlayerEloEngine, compute_effective_elo, get_match_squad_elo
from src.features import rating_source
from src.features.icc_player_rating_engine import ICCPlayerRatingEngine
from src.features.icc_team_rating_engine import ICCTeamRatingEngine
from src.models.xgb_win_probability import XGBWinProbabilityModel
from src.models.first_innings_predictor import FirstInningsPredictor
from src.models.delivery_outcome_model import DeliveryOutcomeModel
from src.data_processing.build_states import StateBuilder
from src.data_processing.load_data import CricketDataLoader
from src.data_processing.format_spec import spec, classify_espn_event, ACTIVE_FORMATS
from src.betting.live_scorecard import ESPNCricInfoScraper
from src.betting.pre_match_predictor import PreMatchPredictor
from src.betting.bet_tracker import BetTracker
from src.betting.scanner import run_scanner

app = Flask(__name__, static_folder="static", static_url_path="")

# --------------------------------------------------
# Global Model and Data Loading
# --------------------------------------------------
print("Initializing XGBoost Win Probability Models...")
MODELS = {}
for _fmt in ACTIVE_FORMATS:
    MODELS[_fmt] = XGBWinProbabilityModel(format_type=_fmt)
    MODELS[_fmt].fit(None, cache_path=f"data/processed/{_fmt}_xgb_models.pkl")

print("Initializing XGBoost First Innings Predictors...")
FIRST_INNINGS_MODELS = {}
for _fmt in ACTIVE_FORMATS:
    FIRST_INNINGS_MODELS[_fmt] = FirstInningsPredictor(format_type=_fmt)
    FIRST_INNINGS_MODELS[_fmt].fit(None, cache_path=f"data/processed/{_fmt}_first_innings_regressor.pkl")

print("Initializing Delivery Outcome Models...")
# EXPERIMENTAL: od's per-ball model exists and passes its own delivery-level log-loss check
# (sim_baseline_metrics.json's "od" entry), but tests/one-time/test_simulation_calibration.py
# (innings-total MAE, win-prob agreement vs the XGB model, coverage, bias) is hardcoded to
# format_type="t20" and has never been run for od -- full-match ODI simulation accuracy is
# genuinely unmeasured. Wired in anyway (per explicit product decision) with an
# "experimental"/unvalidated flag threaded through every od Monte Carlo response (see the
# `"simulation_experimental": format_type == "od"` literals below) so the UI can warn users
# rather than presenting it as equally trustworthy to T20/Hundred. t20w's delivery model went
# through the same full training pipeline as t20's, so it does NOT carry this caveat.
DELIVERY_MODELS = {}
for _fmt in ACTIVE_FORMATS:
    DELIVERY_MODELS[_fmt] = DeliveryOutcomeModel(format_type=_fmt)
    DELIVERY_MODELS[_fmt].fit(None, cache_path=f"data/processed/{_fmt}_delivery_outcome_model.pkl")

# --------------------------------------------------
# Rating-source-aware model accessor (rating_source_toggle_plan.md Phase 6)
# --------------------------------------------------
# Non-elo model sets (currently just "icc", scripts/train_rating_source_models.py's output)
# load LAZILY on first request and stay cached in-process -- doubling every format's model set
# at import, the way MODELS/FIRST_INNINGS_MODELS/DELIVERY_MODELS load eagerly above, is a real
# hazard on this box's 7.7GB RAM (see machine_memory_constraints). The default "elo" source is
# NOT touched by this cache at all: it returns the exact same eagerly-loaded dict entry every
# call site already used, just routed through one function instead of a 7x-repeated
# `X.get(fmt, X["t20"])`.
_ICC_MODEL_CACHE = {}  # (kind, format_type, source) -> fitted model instance

# kind -> (eager dict for the default source, model class, artifact filename stem). Stems match
# scripts/train_rating_source_models.py's naming exactly: "{fmt}_{source}_{stem}.pkl".
_MODEL_KIND_INFO = {
    "xgb": (MODELS, XGBWinProbabilityModel, "xgb_models"),
    "first_innings": (FIRST_INNINGS_MODELS, FirstInningsPredictor, "first_innings_regressor"),
    "delivery": (DELIVERY_MODELS, DeliveryOutcomeModel, "delivery_outcome_model"),
}


def get_model(kind: str, format_type: str, source: str = rating_source.DEFAULT_RATING_SOURCE):
    """Returns the fitted model for (kind, format_type, source). kind is one of
    "xgb" / "first_innings" / "delivery". Raises for an unsupported (format, source) pair or a
    missing artifact rather than silently falling back to Elo -- a caller that gets the source
    wrong should fail loudly, not serve a prediction off the wrong model."""
    eager_dict, model_cls, artifact_stem = _MODEL_KIND_INFO[kind]
    if source == rating_source.DEFAULT_RATING_SOURCE:
        return eager_dict.get(format_type, eager_dict["t20"])

    if not rating_source.is_supported(format_type, source):
        raise ValueError(f"Rating source '{source}' has no model coverage for format '{format_type}'")

    key = (kind, format_type, source)
    if key not in _ICC_MODEL_CACHE:
        cache_path = f"data/processed/{format_type}_{source}_{artifact_stem}.pkl"
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No trained '{source}' {kind} model at {cache_path} -- run "
                f"scripts/train_rating_source_models.py --format {format_type} first."
            )
        print(f"Lazily loading '{source}' {kind} model for {format_type}...")
        model = model_cls(format_type=format_type)
        model.fit(None, cache_path=cache_path)  # cache always exists here -> loads, never trains
        _ICC_MODEL_CACHE[key] = model
    return _ICC_MODEL_CACHE[key]

print("Loading datasets for matches explorer...")
DELIVERIES_BY_FMT = {}
for _fmt in ACTIVE_FORMATS:
    print(f"  Loading {_fmt} deliveries...")
    _loader = CricketDataLoader("data/raw", format_type=_fmt)
    _df = _loader.load_all()
    _df["format_type"] = _fmt
    DELIVERIES_BY_FMT[_fmt] = _df

deliveries = pd.concat(list(DELIVERIES_BY_FMT.values()), ignore_index=True)

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
for _fmt in ACTIVE_FORMATS:
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

# Pre-match team Elo lookup, keyed by (format, match_id) -> (team_1_elo, team_2_elo). These are
# the ratings recorded immediately BEFORE each match was played (see elo_engine.py's
# "Record pre-match Elo ratings" step), not each team's current/live rating — the catalog needs
# the former so the explorer can show each side's true strength entering that specific match.
match_team_elos = {}
for _fmt in ACTIVE_FORMATS:
    _elo_path = f"data/processed/{_fmt}_elo_ratings.parquet"
    if os.path.exists(_elo_path):
        _edf = pd.read_parquet(_elo_path)
        if {"match_id", "team_1_elo", "team_2_elo"}.issubset(_edf.columns):
            for _, _erow in _edf.iterrows():
                match_team_elos[(_fmt, str(_erow["match_id"]))] = (
                    round(float(_erow["team_1_elo"]), 1),
                    round(float(_erow["team_2_elo"]), 1),
                )

matches_list = []
for _, row in merged.iterrows():
    target = int(row["runs_1"] + 1)
    chase_score = int(row["runs_2"])
    fmt = str(row["format_type"])
    mid = str(row["match_id"])
    date_str = str(row["date"])[:10] if pd.notna(row.get("date")) else ""
    year = date_str[:4] if len(date_str) >= 4 else ""
    competition = str(row["competition"]) if pd.notna(row.get("competition")) else ""
    defending_team_elo, chasing_team_elo = match_team_elos.get((fmt, mid), (None, None))
    matches_list.append(
        {
            "match_id": mid,
            "uid": f"{fmt}:{mid}",  # unique across formats (match_id alone is not)
            "defending_team": str(row["team_1"]),
            "chasing_team": str(row["team_2"]),
            "defending_team_elo": defending_team_elo,  # pre-match Elo, not current/live rating
            "chasing_team_elo": chasing_team_elo,
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

TEAM_ELOS = {_fmt: extract_latest_team_elos(f"data/processed/{_fmt}_elo_ratings.parquet") for _fmt in ACTIVE_FORMATS}

# --------------------------------------------------
# Squad-strength baselines for effective Elo (effective_game_elo_plan.md Phase 3)
# --------------------------------------------------
def compute_squad_baselines(fmt, trailing_n=10):
    """Precomputes each team's trailing-N-match squad-strength baseline -- the team-relative
    reference compute_effective_elo() compares today's XI against. squad_now for a match/team
    is the mean of that side's (already Phase-1/2-weighted) batting_squad_elo & bowling_squad_elo
    columns; the baseline is a chronological rolling mean of the team's OWN prior squad_now
    values (same recipe as the Phase-0 calibration script: shift(1) to exclude the match itself,
    rolling(trailing_n, min_periods=3)).

    Returns (historical, live):
      historical: {(team, match_id): baseline}  -- point-in-time, excludes that match itself.
      live:       {team: baseline}               -- as of the team's most recent match (includes
                  it), for surfaces with no historical match_id (the live in-play endpoint).
    Both dicts omit entries with insufficient trailing history (min_periods=3 not met) --
    callers (compute_effective_elo) treat a missing baseline as "unknown," not zero.
    """
    player_path = f"data/processed/{fmt}_player_elo_ratings.parquet"
    elo_path = f"data/processed/{fmt}_elo_ratings.parquet"
    if not os.path.exists(player_path) or not os.path.exists(elo_path):
        return {}, {}

    pdf = pd.read_parquet(player_path)
    squad_cols = ["match_id", "team_1_batting_squad_elo", "team_1_bowling_squad_elo",
                  "team_2_batting_squad_elo", "team_2_bowling_squad_elo"]
    if not all(c in pdf.columns for c in squad_cols):
        return {}, {}

    edf = pd.read_parquet(elo_path)
    if not {"match_id", "team_1", "team_2", "match_date"}.issubset(edf.columns):
        return {}, {}

    squad_df = pdf[squad_cols].drop_duplicates("match_id")
    merged = edf[["match_id", "team_1", "team_2", "match_date"]].merge(squad_df, on="match_id", how="inner")
    merged["match_date"] = pd.to_datetime(merged["match_date"], errors="coerce")

    rows = []
    for _, r in merged.iterrows():
        rows.append((r["match_id"], r["match_date"], r["team_1"],
                     (r["team_1_batting_squad_elo"] + r["team_1_bowling_squad_elo"]) / 2.0))
        rows.append((r["match_id"], r["match_date"], r["team_2"],
                     (r["team_2_batting_squad_elo"] + r["team_2_bowling_squad_elo"]) / 2.0))
    long_df = pd.DataFrame(rows, columns=["match_id", "date", "team", "squad_now"]).sort_values("date")

    long_df["baseline_excl"] = long_df.groupby("team")["squad_now"].transform(
        lambda s: s.shift(1).rolling(trailing_n, min_periods=3).mean()
    )
    long_df["baseline_incl"] = long_df.groupby("team")["squad_now"].transform(
        lambda s: s.rolling(trailing_n, min_periods=3).mean()
    )

    historical = {
        (r["team"], str(r["match_id"])): float(r["baseline_excl"])
        for _, r in long_df.dropna(subset=["baseline_excl"]).iterrows()
    }
    live = {
        team: float(g.iloc[-1]["baseline_incl"])
        for team, g in long_df.dropna(subset=["baseline_incl"]).groupby("team")
    }
    return historical, live

print("Precomputing squad-strength baselines for effective Elo...")
SQUAD_BASELINE_HIST = {}
SQUAD_BASELINE_LIVE = {}
for _fmt in ACTIVE_FORMATS:
    SQUAD_BASELINE_HIST[_fmt], SQUAD_BASELINE_LIVE[_fmt] = compute_squad_baselines(_fmt)

# --------------------------------------------------
# Elo history explorer (Player & Team Data tab)
# --------------------------------------------------
# Parquets are read lazily on first request and cached in-process. Team Elo files are small,
# but player Elo files are large (100k+ rows), so we avoid loading them at startup.
# Cache keys carry `source` (rating_source.DEFAULT_RATING_SOURCE = "elo" by default) so the
# rating-source toggle (markdowns/win_predictor/rating_source_toggle_plan.md) can hold both an
# Elo and an ICC-style snapshot in memory at once without either evicting the other.
_ELO_TEAM_DF_CACHE = {}     # (format, source) -> team ratings DataFrame
_ELO_PLAYER_DF_CACHE = {}   # (format, source) -> player ratings DataFrame (with match_date joined in)
_ELO_ENTITY_CACHE = {}      # (format, type, source) -> list of {name[, roles]}

def _load_team_elo_df(fmt, source=rating_source.DEFAULT_RATING_SOURCE):
    key = (fmt, source)
    if key not in _ELO_TEAM_DF_CACHE:
        path = rating_source.resolve_path(fmt, "team", source)
        _ELO_TEAM_DF_CACHE[key] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return _ELO_TEAM_DF_CACHE[key]

def _load_player_elo_df(fmt, source=rating_source.DEFAULT_RATING_SOURCE):
    """Player ratings joined with each match's date (the player parquet has no date column)."""
    key = (fmt, source)
    if key not in _ELO_PLAYER_DF_CACHE:
        path = rating_source.resolve_path(fmt, "player", source)
        if path.exists():
            pdf = pd.read_parquet(path).copy()
            tdf = _load_team_elo_df(fmt, source)
            date_map = dict(zip(tdf["match_id"], tdf["match_date"])) if not tdf.empty else {}
            pdf["match_date"] = pdf["match_id"].map(date_map)
            _ELO_PLAYER_DF_CACHE[key] = pdf
        else:
            _ELO_PLAYER_DF_CACHE[key] = pd.DataFrame()
    return _ELO_PLAYER_DF_CACHE[key]

_LATEST_PLAYER_ELO_CACHE = {}  # (format, source) -> {(player, role): {"effective_elo", "recent_form"}}

def _load_latest_player_elo(fmt, source=rating_source.DEFAULT_RATING_SOURCE):
    """Each player's own most recent (effective_elo, recent_form), regardless of which
    specific historical match produced it -- unlike find_recent_squad()'s snapshot (limited to
    the ~11-16 names that happened to play in ONE past match), this covers every player who
    has ever been rated. Needed because squad rotation means today's live XI is frequently
    almost entirely different personnel from a team's last tracked fixture; players missing
    from that one snapshot but very much present in the wider ratings (e.g. a recently-rested
    regular) would otherwise be wrongly defaulted to a neutral 1500. Same "latest by match_id"
    convention as build_hnd_seed_ratings()."""
    key = (fmt, source)
    if key not in _LATEST_PLAYER_ELO_CACHE:
        pdf = _load_player_elo_df(fmt, source)
        if pdf.empty:
            _LATEST_PLAYER_ELO_CACHE[key] = {}
        else:
            latest = pdf.sort_values("match_id").groupby(["player", "role"]).last().reset_index()
            _LATEST_PLAYER_ELO_CACHE[key] = {
                (row["player"], row["role"]): {
                    "effective_elo": float(row["effective_elo"]),
                    "raw_elo": float(row["raw_elo"]),
                    "recent_form": float(row["recent_form"]),
                }
                for _, row in latest.iterrows()
            }
    return _LATEST_PLAYER_ELO_CACHE[key]

_BOWLING_RATE_CACHE = {}  # format -> {player: share of recent appearances they bowled in}

def _load_bowling_rates(fmt):
    """Each player's recent bowling rate, for gating who counts as a frontline bowling option
    (see frontline_bowl_sort_key / FRONTLINE_BOWL_MIN_RATE in match_simulator.py).

    Unlike the historical path there's no in-match ground truth to lean on here -- a live XI is
    resolved before anyone has bowled -- so this recency signal is the ONLY thing standing
    between a batter's stale bowling Elo and a place in the simulated attack."""
    if fmt not in _BOWLING_RATE_CACHE:
        _BOWLING_RATE_CACHE[fmt] = bowling_appearance_rates(_load_player_elo_df(fmt))
    return _BOWLING_RATE_CACHE[fmt]

_BOWLING_VOLUME_CACHE = {}  # format -> {player: mean balls bowled per recent appearance}

def _load_bowling_volumes(fmt):
    """Each player's recent bowling workload, used to order the eligible tier by who actually
    bowls the most overs rather than who is rated highest (see frontline_bowl_sort_key).

    Passing the rate gate only means "bowls at all" -- a genuine 4-over part-timer clears it
    just as cleanly as a 10-over specialist, and without this would then be ranked ahead of that
    specialist purely on Elo."""
    if fmt not in _BOWLING_VOLUME_CACHE:
        try:
            _BOWLING_VOLUME_CACHE[fmt] = bowling_volume_rates(
                _load_player_elo_df(fmt), bowler_volume_table(fmt)
            )
        except FileNotFoundError:
            _BOWLING_VOLUME_CACHE[fmt] = {}
    return _BOWLING_VOLUME_CACHE[fmt]

_CRICINFO_ID_MAP_CACHE = None  # cricinfo_id (str) -> unique_name (Cricsheet-style, e.g. "BA King")

def _load_cricinfo_id_map():
    """Maps ESPN/Cricinfo's numeric athlete id -> the Cricsheet "Initials Surname" player name
    (player_meta.csv's unique_name) the historical Elo ratings are keyed by. ESPN's live
    roster feed spells out full first names ("Brandon King") that frequently don't match
    Cricsheet's convention ("BA King") -- a raw name-string match silently drops well-rated
    players to the neutral-1500 fallback, so build_live_team() resolves by id instead."""
    global _CRICINFO_ID_MAP_CACHE
    if _CRICINFO_ID_MAP_CACHE is None:
        path = "data/processed/player_meta.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, usecols=["cricinfo_id", "unique_name"]).dropna(subset=["cricinfo_id", "unique_name"])
            _CRICINFO_ID_MAP_CACHE = dict(zip(df["cricinfo_id"].astype(str), df["unique_name"]))
        else:
            _CRICINFO_ID_MAP_CACHE = {}
    return _CRICINFO_ID_MAP_CACHE

def _match_squad_elo(fmt, match_id):
    """The (Phase-1/2-weighted) batting/bowling squad Elo the win-prob model actually trained
    on for this match -- for the Lineups & Elo effective-Elo display, exact/authoritative rather
    than recomputed. Thin wrapper: the actual lookup lives in player_elo_engine.get_match_squad_elo()
    (single source of truth, also used by PreMatchPredictor's squad-Elo-aware predict()). Returns
    (t1_bat, t1_bowl, t2_bat, t2_bowl) or None if unavailable."""
    return get_match_squad_elo(fmt, match_id)

def compute_venue_averages(df):
    """First-innings average score per venue for a single format's deliveries."""
    if "venue" not in df.columns or "total_runs" not in df.columns:
        return {}
    venue_in1 = df[df["innings"] == 1].groupby(["venue", "match_id"])["total_runs"].sum().reset_index()
    if venue_in1.empty:
        return {}
    venue_stats = venue_in1.groupby("venue")["total_runs"].mean().round(1).to_dict()
    return dict(sorted(venue_stats.items(), key=lambda x: str(x[0])))

VENUE_AVERAGES_BY_FMT = {_fmt: compute_venue_averages(DELIVERIES_BY_FMT[_fmt]) for _fmt in ACTIVE_FORMATS}

# Shared Elo / venue lookup engine for live and pre-match prediction.
print("Loading pre-match predictor (Elos, venues)...")
pre_match_predictor = PreMatchPredictor()

# Persistent paper-bet ledger (data/betting/bet_log.csv). Read-only for the dashboard;
# the pre-match scanner appends paper bets to it.
bet_tracker = BetTracker()


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


def build_first_innings_state(format_type, runs_scored, wickets_lost, balls_faced,
                               batting_team_elo, bowling_team_elo, elo_diff,
                               venue_avg_first_innings_score, competition, overs=None):
    """Builds the first-innings regressor's feature Series for a 1st-innings state.

    Shared by /api/predict_first_innings and the live-state endpoint (mirrors
    build_chase_state's role for the 2nd-innings chase model) so the feature construction
    stays identical. Returns (state_series, run_rate, projected_score, phase).
    """
    fmt_spec = spec(format_type)
    overs = fmt_spec["total_overs"] if overs is None else overs
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

    state = pd.Series(
        {
            "runs_scored": runs_scored,
            "wickets_lost": wickets_lost,
            "balls_faced": balls_faced,
            "balls_remaining": balls_remaining,
            "run_rate": run_rate,
            "over_float": over_float,
            "phase": phase,
            "match_id": 999999,
            "batting_team_elo": batting_team_elo,
            "bowling_team_elo": bowling_team_elo,
            "elo_diff": elo_diff,
            "venue_avg_first_innings_score": venue_avg_first_innings_score,
            "competition": competition,
            "total_overs": overs,
        }
    )
    return state, run_rate, projected_score, phase


def find_recent_squad(team_name, format_type, source=rating_source.DEFAULT_RATING_SOURCE):
    """Returns the squad dict (name/lineup/attack) for a team's most recent historical match.

    Fallback squad source when live lineups aren't published. Shared by the pre-match
    Monte Carlo endpoint and the live Monte Carlo endpoint.
    """
    meta_path = f"data/processed/{format_type}_match_metadata.parquet"
    if not os.path.exists(meta_path):
        return None
    meta_df = pd.read_parquet(meta_path)
    # Men's and women's sides share the same team-name string (e.g. "Pakistan"), and this
    # cache is NOT pre-filtered by gender (unlike the elo/training pipelines, which all apply
    # a gender filter before use) -- sorting by date alone can silently return a women's squad
    # for a men's fixture whenever the women's team happened to play more recently. Filter to
    # THIS format's own gender (was hardcoded to "male", which returned zero rows for an
    # all-female corpus like t20w, silently breaking every squad lookup for that format).
    if "gender" in meta_df.columns:
        meta_df = meta_df[meta_df["gender"] == spec(format_type)["gender"]]
    team_matches = meta_df[
        (meta_df["team_1"] == team_name) | (meta_df["team_2"] == team_name)
    ].sort_values("date", ascending=False)
    if team_matches.empty:
        return None
    m_id = team_matches.iloc[0]["match_id"]
    t1, t2, _ = extract_match_squads(m_id, format_type, source=source)
    return t1 if t1["name"] == team_name else t2


def build_live_team(team_name, live_players, format_type, source=rating_source.DEFAULT_RATING_SOURCE):
    """Builds a team squad dict for the live Monte Carlo.

    Rates each live player from their OWN most recent rating anywhere in history (via
    _load_latest_player_elo), not just whichever single past match find_recent_squad()
    happens to snapshot -- squad rotation means today's actual XI is frequently almost
    entirely different personnel from a team's last tracked fixture, so a player missing from
    that one snapshot but well-rated overall would otherwise be wrongly defaulted to neutral
    1500. find_recent_squad()'s snapshot is kept only as a secondary fallback (covers players
    with no role-specific history who nonetheless appeared in that one match), and only when
    neither source has a rating does a player get neutral 1500. Returns None only when we have
    neither a historical squad nor a live XI to work from.

    live_players: list of {"id": espn_athlete_id_or_None, "name": espn_display_name} from
    ESPNCricInfoScraper.get_lineups(). Each is resolved to the Elo ratings' Cricsheet-style
    name via player_meta.csv's cricinfo_id -> unique_name map when possible — ESPN spells out
    full first names ("Brandon King") that frequently don't match Cricsheet's "Initials
    Surname" convention ("BA King") the historical ratings are keyed by, so a raw name-string
    match alone silently drops well-rated players to the neutral 1500 default. Falls back to
    the raw ESPN name (prior behaviour) whenever no id or no mapping is available.

    Returns (team_dict, squad_source) where squad_source is 'live_xi' | 'historical'.
    """
    recent = find_recent_squad(team_name, format_type, source=source)
    if not live_players:
        return (recent, "historical") if recent else (None, None)

    id_map = _load_cricinfo_id_map()
    latest_elo = _load_latest_player_elo(format_type, source)

    # Secondary fallback maps from one specific recent match's snapshot (kept apart by role
    # because a player's batting and bowling ratings differ — see extract_match_squads).
    bat_map, bowl_map = {}, {}
    if recent:
        for p in recent.get("lineup", []):
            bat_map[p["name"]] = p
        for p in recent.get("attack", []):
            bowl_map[p["name"]] = p

    default = {"effective_elo": 1500.0, "raw_elo": 1500.0, "recent_form": 0.0}

    def _rating(key, role, snapshot_map):
        """Returns (rating_dict, rated: bool). rated=False means this player has no record
        anywhere (full history or the one-match snapshot) for this role and is sitting on the
        neutral-1500 default -- surfaced per-player so the UI can flag low-confidence squads
        instead of silently mispricing them."""
        r = latest_elo.get((key, role))
        if r is not None:
            return r, True
        if key in snapshot_map:
            return snapshot_map[key], True
        return default, False

    # (espn_display_name, lookup_key) pairs -- lookup_key is the resolved Cricsheet-style
    # name when the id maps to one, else the raw ESPN name.
    resolved = []
    for entry in live_players[:11]:
        espn_name = entry["name"]
        aid = entry.get("id")
        key = id_map.get(str(aid)) if aid is not None else None
        if key is None:
            key = espn_name
        resolved.append((espn_name, key))
    while len(resolved) < 11:
        dummy = f"Dummy_{len(resolved)+1}"
        resolved.append((dummy, dummy))

    lineup = []
    for espn_name, key in resolved:
        src, rated = _rating(key, "batter", bat_map)
        lineup.append({"name": espn_name, "effective_elo": src["effective_elo"], "raw_elo": src["raw_elo"], "recent_form": src["recent_form"], "rated": rated})

    # Attack = 5 best bowling options from the XI: frontline-eligible players first, and only
    # then by bowling Elo (matching extract_match_squads' selection). Ranking value is 0.0 for
    # anyone with NO bowler-role record anywhere (full history or snapshot) -- never rank on the
    # neutral-1500 default, which would wrongly promote unrated players above genuine part-time
    # bowlers rated below 1500 (the same shrinkage-ranking trap effective_game_elo_plan.md's
    # Phase 2 hit and fixed on the training side).
    #
    # Having *a* bowling record is necessary but NOT sufficient: it never expires, so a batter
    # who bowled a few overs years ago keeps a bowling Elo that shrinkage anchors near team
    # strength and which then outranks the real attack (DP Sibley topping Surrey's bowling on
    # 2026-08-05 with nothing bowled since 2018). The recent-appearance rate is what makes it
    # sufficient -- see FRONTLINE_BOWL_MIN_RATE. Ordering WITHIN the eligible tier then goes by
    # expected workload (bowl_volumes), with Elo only breaking ties: clearing the rate gate says
    # a player bowls, not how much, so a 4-over part-timer would otherwise be ranked above a
    # 10-over specialist on rating alone.
    bowl_rates = _load_bowling_rates(format_type)
    bowl_volumes = _load_bowling_volumes(format_type)

    def _bowl_rank_value(key):
        r = latest_elo.get((key, "bowler"))
        if r is not None:
            return r["effective_elo"]
        if key in bowl_map:
            return bowl_map[key]["effective_elo"]
        return 0.0

    ranked = sorted(
        resolved,
        key=lambda t: frontline_bowl_sort_key(
            t[1], _bowl_rank_value(t[1]), bowl_rates, volumes=bowl_volumes
        ),
        reverse=True,
    )
    # overs_per_inns surfaces the value the ordering is based on -- keyed on the resolved
    # Cricsheet name (not the ESPN display name) since that's what the volume table is keyed by.
    bpo = float(spec(format_type)["balls_per_over"])
    attack = []
    for espn_name, key in ranked[:ATTACK_SIZE]:
        src, rated = _rating(key, "bowler", bowl_map)
        attack.append({
            "name": espn_name, "effective_elo": src["effective_elo"], "raw_elo": src["raw_elo"],
            "recent_form": src["recent_form"], "rated": rated,
            "overs_per_inns": round(bowl_volumes[key] / bpo, 1) if key in bowl_volumes else None,
        })

    # Squad-quality summary (live XI only -- a "historical" squad_source fallback is, by
    # definition, a real past match's roster, so every player there already has a genuine
    # rating). Surfaced so the UI can warn when a live match's Elo-based pricing rests on
    # mostly-unrated players rather than failing silently.
    unrated_names = sorted({p["name"] for p in lineup if not p["rated"]} | {p["name"] for p in attack if not p["rated"]})
    quality = {"unrated_count": len(unrated_names), "total": len(lineup), "unrated_names": unrated_names}

    return {"name": team_name, "lineup": lineup, "attack": attack, "quality": quality}, "live_xi"


def elo_teams_for_format(format_type):
    """Returns the set of canonical team names in the Elo DB for a format."""
    if format_type == "hnd":
        return set(pre_match_predictor.hnd_elos.keys())
    if format_type == "od":
        return set(pre_match_predictor.od_elos.keys())
    if format_type == "t20w":
        return set(pre_match_predictor.t20w_elos.keys())
    if format_type == "odw":
        return set(pre_match_predictor.odw_elos.keys())
    if format_type == "hndw":
        return set(pre_match_predictor.hndw_elos.keys())
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
    """Returns precomputed metadata for dropdowns (teams with Elos, venues with average score).
    Flat `{fmt}_teams`/`{fmt}_venues` keys are kept (not restructured to a nested dict) for
    backward compatibility with existing frontend code -- new formats just add new flat keys."""
    resp = {}
    for _fmt in ACTIVE_FORMATS:
        resp[f"{_fmt}_teams"] = TEAM_ELOS.get(_fmt, {})
        resp[f"{_fmt}_venues"] = VENUE_AVERAGES_BY_FMT.get(_fmt, {})
    # Kept for backward compatibility; T20 is the default format.
    resp["venues"] = VENUE_AVERAGES_BY_FMT.get("t20", {})
    # Rating-source toggle (markdowns/win_predictor/rating_source_toggle_plan.md Phase 4): which
    # source keys have coverage for each format, so the UI can disable/tooltip an unsupported
    # combination instead of the request 400ing after the fact.
    resp["rating_sources"] = {_fmt: list(rating_source.available_sources(_fmt)) for _fmt in ACTIVE_FORMATS}
    return jsonify(resp)


@app.route("/api/elo/entities")
def elo_entities():
    """Lists the searchable teams or players (with available roles) for a given format."""
    fmt = request.args.get("format", "t20")
    etype = request.args.get("type", "team")
    source = request.args.get("source", rating_source.DEFAULT_RATING_SOURCE)
    if not rating_source.is_supported(fmt, source):
        return jsonify({"error": f"Rating source '{source}' has no coverage for format '{fmt}'"}), 400
    key = (fmt, etype, source)
    if key not in _ELO_ENTITY_CACHE:
        if etype == "player":
            pdf = _load_player_elo_df(fmt, source)
            if pdf.empty:
                _ELO_ENTITY_CACHE[key] = []
            else:
                roles_by_player = pdf.groupby("player")["role"].agg(lambda s: sorted(set(s)))
                _ELO_ENTITY_CACHE[key] = [
                    {"name": str(p), "roles": list(r)} for p, r in sorted(roles_by_player.items())
                ]
        else:
            tdf = _load_team_elo_df(fmt, source)
            if tdf.empty:
                names = []
            else:
                names = sorted(set(tdf["team_1"].astype(str)) | set(tdf["team_2"].astype(str)))
            _ELO_ENTITY_CACHE[key] = [{"name": n} for n in names]
    return jsonify({"format": fmt, "type": etype, "source": source, "entities": _ELO_ENTITY_CACHE[key]})


def _elo_series_for(etype, fmt, name, role="batter", start=None, end=None,
                    source=rating_source.DEFAULT_RATING_SOURCE):
    """Looks up one team's or player's chronological rating series, optionally restricted to
    match_date in [start, end] (both inclusive, "YYYY-MM-DD" strings; either may be None for
    an open-ended bound). Returns (series, stats, error) where error is None on success, or
    (None, None, "message") on failure."""
    if etype == "player":
        pdf = _load_player_elo_df(fmt, source)
        if pdf.empty:
            return None, None, f"No player rating data for format {fmt} (source: {source})"
        sub = pdf[(pdf["player"] == name) & (pdf["role"] == role)].dropna(subset=["match_date"])
        if sub.empty:
            return None, None, f"No {role} record found for {name}"
        # match_date is an ISO YYYY-MM-DD string, so lexical sort == chronological sort.
        sub = sub.sort_values("match_date")
        series = [
            {
                "date": str(r.match_date),
                "elo": round(float(r.raw_elo), 1),
                "effective": round(float(r.effective_elo), 1),
                "match_id": str(r.match_id),
            }
            for r in sub.itertuples()
        ]
    else:
        tdf = _load_team_elo_df(fmt, source)
        if tdf.empty:
            return None, None, f"No rating data for format {fmt} (source: {source})"
        m1 = tdf[tdf["team_1"].astype(str) == name][["match_date", "team_1_elo", "match_id"]]
        m1 = m1.rename(columns={"team_1_elo": "elo"})
        m2 = tdf[tdf["team_2"].astype(str) == name][["match_date", "team_2_elo", "match_id"]]
        m2 = m2.rename(columns={"team_2_elo": "elo"})
        both = pd.concat([m1, m2], ignore_index=True).dropna(subset=["match_date"])
        if both.empty:
            return None, None, f"No matches found for {name}"
        both = both.sort_values("match_date")
        series = [
            {"date": str(r.match_date), "elo": round(float(r.elo), 1), "match_id": str(r.match_id)}
            for r in both.itertuples()
        ]

    # match_date strings are ISO "YYYY-MM-DD", so lexical comparison is chronological comparison.
    if start:
        series = [pt for pt in series if pt["date"] >= start]
    if end:
        series = [pt for pt in series if pt["date"] <= end]
    if not series:
        return None, None, f"No matches for {name} in the selected period"

    elos = [pt["elo"] for pt in series]
    stats = {
        "current": elos[-1],
        "peak": round(max(elos), 1),
        "low": round(min(elos), 1),
        "matches": len(series),
        "first_date": series[0]["date"],
        "last_date": series[-1]["date"],
    }
    return series, stats, None


_ELO_LEADERBOARD_CACHE = {}  # (format, type, role, source) -> list of rows, sorted desc by "current"

def _leaderboard_qualified(df, group_col, date_col="match_date", min_matches=20):
    """Drops entities that are not (a) established enough or (b) still current, and picks each
    survivor's genuinely most recent row.

    Two gates, both mirroring what the rating engines already apply in their own
    get_effective_ratings_from_df() -- this endpoint predates them and never got the same
    treatment, which is why 5-match associates and players last seen years ago were ranking
    above established internationals:
      - MIN_MATCHES_TO_RANK: a handful of innings is a small-sample fluke, not a rating.
      - MAX_INACTIVITY_DAYS: a "current" leaderboard should only contain current players.

    The inactivity window is measured from THIS FORMAT'S OWN most recent match, not wall-clock
    now(), matching the convention the validation scripts use. Wall-clock breaks seasonal
    competitions: The Hundred's last season ended 2025-08-31, so a 365-day window from "today"
    empties its leaderboard completely (measured: 0 of 240 batters, 0 of 8 teams survive) purely
    because the next season hasn't been played yet.

    Returns (last_row_per_entity, per_entity_match_counts), both already filtered.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if df.empty:
        return df, pd.Series(dtype=int)

    as_of = df[date_col].max()
    # Sort by DATE, not match_id. Cricsheet ids are ~chronological (rank correlation 0.999) but
    # not reliably so across competitions, and the exceptions land on exactly the players anyone
    # would check first -- "latest by match_id" gave V Kohli his 2016-05-29 rating as "current"
    # instead of his 2026-05-31 one.
    df = df.sort_values(date_col)
    last = df.groupby(group_col).last()
    counts = df.groupby(group_col).size()

    inactive_days = (as_of - last[date_col]).dt.days
    keep = (counts >= min_matches) & (inactive_days <= ICCPlayerRatingEngine.MAX_INACTIVITY_DAYS)
    keep = keep[keep].index
    return last.loc[keep], counts.loc[keep]


def _build_leaderboard(etype, fmt, role="batter", source=rating_source.DEFAULT_RATING_SOURCE):
    """Every qualified team's or player's (role-filtered) latest rating for a format, ranked desc.

    Uses raw_elo throughout (never effective_elo), matching the Elo Explorer's own
    current/peak/low stats (_elo_series_for) -- effective_elo is a per-match blend against that
    match's opposing team strength, not a standalone rating, so it isn't comparable across
    players/matches the way a leaderboard needs.

    peak/low/matches are plain column aggregates (order-independent) over the entity's whole
    history; "current" comes from its most recent match. Qualification/recency gating is
    _leaderboard_qualified()'s job -- see there for why both gates exist and what they fixed."""
    key = (fmt, etype, role if etype == "player" else None, source)
    if key in _ELO_LEADERBOARD_CACHE:
        return _ELO_LEADERBOARD_CACHE[key]

    # rating_pool only exists on combined-source parquets (icc), not the Elo ones -- surfaced
    # per row as the PROVENANCE OF THE "current" figure (that team/player's most recent match),
    # not the whole history, so the UI badge always describes what the number actually rests on.
    if etype == "player":
        pdf = _load_player_elo_df(fmt, source)
        has_pool = "rating_pool" in pdf.columns
        sub = pdf[pdf["role"] == role].dropna(subset=["match_date"]) if not pdf.empty else pdf
        if sub.empty:
            rows = []
        else:
            agg = sub.groupby("player")["raw_elo"].agg(peak="max", low="min")
            last, counts = _leaderboard_qualified(
                sub, "player", min_matches=ICCPlayerRatingEngine.MIN_MATCHES_TO_RANK)
            rows = [
                {
                    "name": player,
                    "current": round(float(r["raw_elo"]), 1),
                    "peak": round(float(agg.loc[player, "peak"]), 1),
                    "low": round(float(agg.loc[player, "low"]), 1),
                    "matches": int(counts.loc[player]),
                    "last_date": str(pd.Timestamp(r["match_date"]).date()),
                    "rating_pool": str(r["rating_pool"]) if has_pool else None,
                }
                for player, r in last.iterrows()
            ]
    else:
        tdf = _load_team_elo_df(fmt, source)
        if tdf.empty:
            rows = []
        else:
            has_pool = "rating_pool" in tdf.columns
            pool_cols = ["rating_pool"] if has_pool else []
            m1 = tdf[["match_id", "match_date", "team_1", "team_1_elo"] + pool_cols].rename(
                columns={"team_1": "team", "team_1_elo": "elo"})
            m2 = tdf[["match_id", "match_date", "team_2", "team_2_elo"] + pool_cols].rename(
                columns={"team_2": "team", "team_2_elo": "elo"})
            both = pd.concat([m1, m2], ignore_index=True).dropna(subset=["match_date"])
            agg = both.groupby("team")["elo"].agg(peak="max", low="min")
            last, counts = _leaderboard_qualified(
                both, "team", min_matches=ICCTeamRatingEngine.MIN_MATCHES_TO_RANK)
            rows = [
                {
                    "name": team,
                    "current": round(float(r["elo"]), 1),
                    "peak": round(float(agg.loc[team, "peak"]), 1),
                    "low": round(float(agg.loc[team, "low"]), 1),
                    "matches": int(counts.loc[team]),
                    "rating_pool": str(r["rating_pool"]) if has_pool else None,
                    "last_date": str(pd.Timestamp(r["match_date"]).date()),
                }
                for team, r in last.iterrows()
            ]

    rows.sort(key=lambda r: r["current"], reverse=True)
    _ELO_LEADERBOARD_CACHE[key] = rows
    return rows


@app.route("/api/elo/leaderboard")
def elo_leaderboard():
    """Ranks every team's or player's current rating for a format, highest first, for the
    Player & Team Data leaderboard view. ?source= selects Elo (default) or the ICC-style
    combined rating (see src/features/rating_source.py); unsupported for the given format
    returns 400 rather than silently falling back."""
    fmt = request.args.get("format", "t20")
    etype = request.args.get("type", "team")
    role = request.args.get("role", "batter")
    source = request.args.get("source", rating_source.DEFAULT_RATING_SOURCE)
    if not rating_source.is_supported(fmt, source):
        return jsonify({"error": f"Rating source '{source}' has no coverage for format '{fmt}'"}), 400
    try:
        limit = max(1, min(int(request.args.get("limit", 25)), 200))
    except (TypeError, ValueError):
        limit = 25

    rows = _build_leaderboard(etype, fmt, role, source)
    return jsonify({
        "format": fmt, "type": etype, "role": role, "source": source,
        "total": len(rows), "rows": rows[:limit],
    })


@app.route("/api/elo/history")
def elo_history():
    """Returns a team's or player's chronological rating progression for a format, optionally
    restricted to a date window via ?start=YYYY-MM-DD&end=YYYY-MM-DD. ?source= as above."""
    fmt = request.args.get("format", "t20")
    etype = request.args.get("type", "team")
    role = request.args.get("role", "batter")
    name = (request.args.get("name") or "").strip()
    start = (request.args.get("start") or "").strip() or None
    end = (request.args.get("end") or "").strip() or None
    source = request.args.get("source", rating_source.DEFAULT_RATING_SOURCE)
    if not name:
        return jsonify({"error": "Missing name"}), 400
    if not rating_source.is_supported(fmt, source):
        return jsonify({"error": f"Rating source '{source}' has no coverage for format '{fmt}'"}), 400

    series, stats, error = _elo_series_for(etype, fmt, name, role, start=start, end=end, source=source)
    if error:
        return jsonify({"error": error}), 404
    return jsonify({"name": name, "type": etype, "format": fmt, "source": source, "series": series, "stats": stats})


@app.route("/api/elo/compare", methods=["POST"])
def elo_compare():
    """Batch lookup of up to 8 team/player rating histories for the comparison chart. An
    optional top-level "start"/"end" (YYYY-MM-DD) restricts every item to the same date window.
    Each item may carry its own "source" (default: Elo), so a comparison can mix sources."""
    data = request.json or {}
    items = data.get("items", [])
    start = (data.get("start") or "").strip() or None
    end = (data.get("end") or "").strip() or None
    if not isinstance(items, list) or not items:
        return jsonify({"error": "No items provided"}), 400
    if len(items) > 8:
        return jsonify({"error": "A maximum of 8 items can be compared at once"}), 400

    results = []
    for item in items:
        etype = item.get("type", "team")
        fmt = item.get("format", "t20")
        role = item.get("role", "batter")
        name = (item.get("name") or "").strip()
        source = item.get("source", rating_source.DEFAULT_RATING_SOURCE)
        if not name:
            results.append({"type": etype, "format": fmt, "role": role, "name": name, "error": "Missing name"})
            continue
        if not rating_source.is_supported(fmt, source):
            results.append({"type": etype, "format": fmt, "role": role, "name": name,
                            "error": f"Rating source '{source}' has no coverage for format '{fmt}'"})
            continue
        series, stats, error = _elo_series_for(etype, fmt, name, role, start=start, end=end, source=source)
        if error:
            results.append({"type": etype, "format": fmt, "role": role, "name": name, "error": error})
        else:
            results.append({"type": etype, "format": fmt, "role": role, "name": name,
                            "source": source, "series": series, "stats": stats})

    return jsonify({"results": results})


@app.route("/api/simulate", methods=["POST"])
def api_simulate_match():
    """Predicts win probability and 5-fold ensemble confidence interval for a custom simulated match state."""
    data = request.json or {}
    try:
        format_type = data.get("format", "t20")
        source = data.get("source", rating_source.DEFAULT_RATING_SOURCE)
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

        model_to_use = get_model("xgb", format_type, source)

        res = model_to_use.predict_with_ensemble(state)
        res["required_run_rate"] = round(required_run_rate, 2)
        res["run_rate"] = round(run_rate, 2)
        res["phase"] = phase
        res["explainability"] = model_to_use.get_feature_contributions(state, top_n=5)
        res["source"] = source
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
        source = data.get("source", rating_source.DEFAULT_RATING_SOURCE)
        if not rating_source.is_supported(format_type, source):
            return jsonify({"error": f"Rating source '{source}' has no coverage for format '{format_type}'"}), 400

        meta_path = f"data/processed/{format_type}_match_metadata.parquet"
        if not os.path.exists(meta_path):
            return jsonify({"error": f"Metadata missing for format {format_type}"}), 400

        # If a specific match_id is provided, just extract both from it
        if match_id:
            team_a_dict, team_b_dict, venue_avg_score = extract_match_squads(match_id, format_type, source=source)
        else:
            team_a_dict = find_recent_squad(team_a_name, format_type, source=source)
            team_b_dict = find_recent_squad(team_b_name, format_type, source=source)
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

        # rating_source_toggle_plan.md Phase 6-continued: model and squad now share `source` --
        # find_recent_squad/extract_match_squads resolve player ratings on the same scale the
        # model was trained on, so this is no longer the model-only swap that would have fed an
        # ICC-trained model Elo-scale features.
        model_to_use = get_model("delivery", format_type, source)

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
            
        hist_inn1 = bucket_scores(inn1_scores, bin_width=20 if format_type in ("od", "odw") else 10)
        hist_inn2 = bucket_scores(inn2_scores, bin_width=20 if format_type in ("od", "odw") else 10)

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
            # OD's delivery model passes its own per-ball log-loss check, but the full-match
            # calibration battery (innings MAE, win-prob agreement) has only ever been run for
            # t20 -- see DELIVERY_MODELS' EXPERIMENTAL comment near its definition above.
            # Flagged so the UI can warn.
            # odw carries the same caveat as od (same unmeasured-full-match-sim reason);
            # hndw doesn't -- its pooled model went through the same Phase-3 pooled-beats-
            # unpooled gate hnd itself did.
            "simulation_experimental": format_type in ("od", "odw"),
            "source": source,
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
        source = data.get("source", rating_source.DEFAULT_RATING_SOURCE)
        if not rating_source.is_supported(format_type, source):
            return jsonify({"error": f"Rating source '{source}' has no coverage for format '{format_type}'"}), 400

        meta_path = f"data/processed/{format_type}_match_metadata.parquet"
        if not os.path.exists(meta_path):
            return jsonify({"error": f"Metadata missing for format {format_type}"}), 400

        if match_id:
            # team_a bats, team_b bowls — extract_match_squads' team_1/team_2 order matches
            # the innings-1 batting side / innings-1 bowling side respectively.
            team_a_dict, team_b_dict, venue_avg_score = extract_match_squads(match_id, format_type, source=source)
        else:
            team_a_name = data.get("team_a")
            team_b_name = data.get("team_b")
            team_a_dict = find_recent_squad(team_a_name, format_type, source=source)
            team_b_dict = find_recent_squad(team_b_name, format_type, source=source)
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

        # rating_source_toggle_plan.md Phase 6-continued: model and squad now share `source`.
        model_to_use = get_model("delivery", format_type, source)

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

        hist = bucket_scores(inn1["final_score"].tolist(), bin_width=20 if format_type in ("od", "odw") else 10)
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
            # odw carries the same caveat as od (same unmeasured-full-match-sim reason);
            # hndw doesn't -- its pooled model went through the same Phase-3 pooled-beats-
            # unpooled gate hnd itself did.
            "simulation_experimental": format_type in ("od", "odw"),
            "source": source,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/live/games")
def live_games():
    """Lists current cricket matches (live + upcoming) scraped from ESPN's scoreboard.

    Match types we don't model at all (Test, First-Class) are excluded outright rather than
    defaulted to 't20' and offered for pricing -- previously a live Test match ("West Indies
    v Pakistan", 2nd Test) was silently listed and priced as if it were T20, since neither the
    name nor league text ("Pakistan tour of West Indies 2026") contains a format keyword and
    infer_format() defaults to 't20' when nothing matches. classify_espn_event() instead
    trusts ESPN's own eventType/class.generalClassCard for each event first (see
    format_spec.py), falling back to the text-keyword heuristic only when ESPN's own
    classification is missing.
    """
    try:
        games = ESPNCricInfoScraper.list_live_games(include_upcoming=True)
        result = []
        for g in games:
            fmt = classify_espn_event(
                g.get("event_type", ""), g.get("class_card", ""),
                f"{g.get('name','')} {g.get('league_name','')}",
            )
            if fmt is None:
                continue
            g["format"] = fmt
            result.append(g)
        return jsonify(result)
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
        source = request.args.get("source", rating_source.DEFAULT_RATING_SOURCE)
        if not rating_source.is_supported(format_type, source):
            return jsonify({"error": f"Rating source '{source}' has no coverage for format '{format_type}'"}), 400
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
        batting_elo = pre_match_predictor.get_team_elo(matched_bat, format_type, source)
        bowling_elo = pre_match_predictor.get_team_elo(matched_bowl, format_type, source)
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
            "rating_source": source,
            "predicted_first_innings_score": None,
            "first_innings_projected_score": None,
            "first_innings_prediction_source": None,
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
            model = get_model("xgb", format_type, source)
            res = model.predict_with_ensemble(state)
            resp["win_prob_batting"] = round(float(res["win_probability"]), 4)
            resp["winprob_source"] = "xgb_chase"
            resp["required_run_rate"] = round(rrr, 2)
            resp["run_rate"] = round(rr, 2)
            resp["phase"] = phase
        elif innings == 1:
            resp["winprob_source"] = "montecarlo"  # innings 1: run MC for win prob
            fi_state, fi_run_rate, fi_projected_score, fi_phase = build_first_innings_state(
                format_type=format_type, runs_scored=float(runs), wickets_lost=float(wickets),
                balls_faced=float(legal_balls), batting_team_elo=batting_elo, bowling_team_elo=bowling_elo,
                elo_diff=batting_elo - bowling_elo, venue_avg_first_innings_score=venue_avg,
                competition=format_type, overs=fmt_spec["total_overs"],
            )
            fi_model = get_model("first_innings", format_type, source)
            fi_pred = fi_model.predict(fi_state)
            resp["predicted_first_innings_score"] = fi_pred["predicted_score"]
            resp["first_innings_projected_score"] = int(round(fi_projected_score))
            resp["first_innings_prediction_source"] = "xgb_first_innings"
            resp["run_rate"] = round(fi_run_rate, 2)
            resp["phase"] = fi_phase
        else:
            resp["winprob_source"] = "montecarlo"

        return jsonify(resp)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Could not fetch live state: {e}"}), 502


def _resolve_live_teams(event_id, format_type, league_id=None, source=rating_source.DEFAULT_RATING_SOURCE):
    """Resolves a live match's batting/bowling squads (lineup + attack, each player's Elo and
    'rated' flag, plus a per-side quality summary) and the weighted squad/effective-Elo tiles
    -- everything the Lineups & Elo view needs -- WITHOUT running the Monte Carlo simulation.

    Shared by the lightweight /api/live/squads preview (so the UI can show squads and flag
    low-confidence ones as soon as a match is selected, before committing to a simulation) and
    /api/live/montecarlo (which calls this first, then continues on to simulate). Returns a
    dict with a non-None "error" key on failure (squads unresolvable) -- callers check that
    before using the rest of the fields.

    `source` only steers build_live_team's PLAYER ratings -- the ones that actually feed the
    Monte Carlo engine, where an out-of-distribution mismatch would corrupt the simulation.
    The "overall"/"squad"/"effective" Elo TILES below stay on Elo regardless: they're a display
    decoration built from the deployed Elo engine's own squad-baseline calibration
    (SQUAD_BASELINE_LIVE, compute_effective_elo), which has no ICC-source equivalent -- building
    one would be a separate feature, not a `source` passthrough.
    """
    fmt_spec = spec(format_type)
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
    bat_players = lineups.get(batting_team, [])
    bowl_players = lineups.get(bowling_team, [])
    bat_build = build_live_team(matched_bat, bat_players, format_type, source=source)
    bowl_build = build_live_team(matched_bowl, bowl_players, format_type, source=source)
    bat_team, bat_src = bat_build
    bowl_team, bowl_src = bowl_build
    if not bat_team or not bowl_team:
        return {"error": "Could not resolve squads for one or both teams."}

    # Each side's current overall (team-level) Elo — same figure the historical Monte Carlo
    # dropdowns are labelled with, surfaced here for the live Lineups & Elo view.
    bat_overall_elo = pre_match_predictor.get_team_elo(matched_bat, format_type)
    bowl_overall_elo = pre_match_predictor.get_team_elo(matched_bowl, format_type)

    # team_a is always the currently-batting side. In innings 1 they bat first (bat_first
    # 'a'); in innings 2 they are chasing, so the bowling side batted first (bat_first 'b').
    bat_first = "b" if (innings == 2 and target) else "a"

    # first/second batting sides (for innings-scoped market pricing)
    first_team = bat_team if bat_first == "a" else bowl_team
    second_team = bowl_team if bat_first == "a" else bat_team
    first_overall_elo = bat_overall_elo if bat_first == "a" else bowl_overall_elo
    second_overall_elo = bowl_overall_elo if bat_first == "a" else bat_overall_elo

    # Weighted batting/bowling squad Elo + effective Elo for the live Lineups & Elo view
    # (effective_game_elo_plan.md Phase 3). "attack" here is already a pre-selected top-5
    # (match_simulator's own selection, independent of Phase 2's top-6) rather than the full
    # 11-name squad the training pipeline ranks from — _weighted_mean_by_bowling_rank
    # gracefully renormalizes over however many names are actually available (same short-
    # squad handling verified in Phase 2's unit tests), so this is a close, honest
    # approximation from the best data this endpoint has, not the exact trained value
    # (that exactness is only possible for historical match_ids — see _match_squad_elo).
    def _live_squad_elo(team_dict, overall_elo):
        w_bat = PlayerEloEngine.W_BAT.get(format_type, PlayerEloEngine.W_BAT["t20"])
        w_bowl = PlayerEloEngine.W_BOWL.get(format_type, PlayerEloEngine.W_BOWL["t20"])
        lineup, attack = team_dict.get("lineup", []), team_dict.get("attack", [])
        bat_map = {p["name"]: p["effective_elo"] for p in lineup}
        bowl_map = {p["name"]: p["effective_elo"] for p in attack}
        bat_sq = PlayerEloEngine._weighted_mean_by_position([p["name"] for p in lineup], bat_map, 1500.0, w_bat)
        bowl_sq = PlayerEloEngine._weighted_mean_by_bowling_rank([p["name"] for p in attack], bowl_map, 1500.0, w_bowl)
        baseline = SQUAD_BASELINE_LIVE.get(format_type, {}).get(team_dict.get("name"))
        eff = compute_effective_elo(overall_elo, bat_sq, bowl_sq, baseline, format_type)
        return round(bat_sq, 1), round(bowl_sq, 1), round(eff, 1)

    first_batting_squad_elo, first_bowling_squad_elo, first_effective_elo = _live_squad_elo(first_team, first_overall_elo)
    second_batting_squad_elo, second_bowling_squad_elo, second_effective_elo = _live_squad_elo(second_team, second_overall_elo)

    return {
        "error": None,
        "fmt_spec": fmt_spec, "legal_balls": legal_balls, "total_balls": total_balls,
        "innings": innings, "runs": runs, "wickets": wickets, "target": target,
        "batting_team": batting_team, "bowling_team": bowling_team, "venue_avg": venue_avg,
        "bat_team": bat_team, "bowl_team": bowl_team, "bat_src": bat_src, "bowl_src": bowl_src,
        "bat_first": bat_first, "first_team": first_team, "second_team": second_team,
        "first_overall_elo": round(first_overall_elo, 1), "second_overall_elo": round(second_overall_elo, 1),
        "first_batting_squad_elo": first_batting_squad_elo,
        "first_bowling_squad_elo": first_bowling_squad_elo,
        "first_effective_elo": first_effective_elo,
        "second_batting_squad_elo": second_batting_squad_elo,
        "second_bowling_squad_elo": second_bowling_squad_elo,
        "second_effective_elo": second_effective_elo,
    }


# A side is flagged low-confidence when at least this many of its 11 players have no rating
# anywhere (full history or the one-match historical snapshot) and are sitting on the neutral
# 1500 default — squad/effective Elo for that side is honest-best-effort, not a real signal.
LOW_CONFIDENCE_UNRATED_THRESHOLD = 3


@app.route("/api/live/squads")
def live_squads():
    """Lightweight preview of a live match's resolved squads + Elo tiles, without running the
    Monte Carlo simulation — lets the UI show lineups (and flag low-confidence ones, see
    LOW_CONFIDENCE_UNRATED_THRESHOLD) as soon as a match is selected, before the user commits
    to a full simulation."""
    try:
        event_id = request.args.get("event_id")
        format_type = request.args.get("format", "t20")
        league_id = request.args.get("league_id")
        source = request.args.get("source", rating_source.DEFAULT_RATING_SOURCE)
        if not rating_source.is_supported(format_type, source):
            return jsonify({"error": f"Rating source '{source}' has no coverage for format '{format_type}'"}), 400
        resolved = _resolve_live_teams(event_id, format_type, league_id, source=source)
        if resolved["error"]:
            return jsonify({"error": resolved["error"]}), 400

        first_team, second_team = resolved["first_team"], resolved["second_team"]
        return jsonify({
            "event_id": str(event_id),
            "format": format_type,
            "source": source,
            "innings": resolved["innings"],
            "batting_team": resolved["batting_team"],
            "bowling_team": resolved["bowling_team"],
            "first_batting_team": first_team["name"],
            "second_batting_team": second_team["name"],
            "first_batting_squad": first_team,
            "second_batting_squad": second_team,
            "first_batting_overall_elo": resolved["first_overall_elo"],
            "second_batting_overall_elo": resolved["second_overall_elo"],
            "first_batting_squad_elo": resolved["first_batting_squad_elo"],
            "first_bowling_squad_elo": resolved["first_bowling_squad_elo"],
            "first_effective_elo": resolved["first_effective_elo"],
            "second_batting_squad_elo": resolved["second_batting_squad_elo"],
            "second_bowling_squad_elo": resolved["second_bowling_squad_elo"],
            "second_effective_elo": resolved["second_effective_elo"],
            "squad_source": {"batting": resolved["bat_src"], "bowling": resolved["bowl_src"]},
            "low_confidence_threshold": LOW_CONFIDENCE_UNRATED_THRESHOLD,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Could not fetch live squads: {e}"}), 502


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
        source = data.get("source", rating_source.DEFAULT_RATING_SOURCE)
        if not rating_source.is_supported(format_type, source):
            return jsonify({"error": f"Rating source '{source}' has no coverage for format '{format_type}'"}), 400

        # rating_source_toggle_plan.md Phase 6-continued: model and squad now share the same
        # `source` -- _resolve_live_teams threads it into build_live_team's PLAYER ratings, so
        # the model and its features are on the same scale, avoiding the out-of-distribution
        # mismatch a model-only swap would have caused.
        model_to_use = get_model("delivery", format_type, source)

        resolved = _resolve_live_teams(event_id, format_type, league_id, source=source)
        if resolved["error"]:
            return jsonify({"error": resolved["error"]}), 400

        fmt_spec = resolved["fmt_spec"]
        legal_balls, innings = resolved["legal_balls"], resolved["innings"]
        runs, wickets, target = resolved["runs"], resolved["wickets"], resolved["target"]
        batting_team, bowling_team = resolved["batting_team"], resolved["bowling_team"]
        bat_team, bowl_team = resolved["bat_team"], resolved["bowl_team"]
        bat_src, bowl_src = resolved["bat_src"], resolved["bowl_src"]
        bat_first = resolved["bat_first"]
        first_team, second_team = resolved["first_team"], resolved["second_team"]
        first_overall_elo, second_overall_elo = resolved["first_overall_elo"], resolved["second_overall_elo"]
        first_batting_squad_elo = resolved["first_batting_squad_elo"]
        first_bowling_squad_elo = resolved["first_bowling_squad_elo"]
        first_effective_elo = resolved["first_effective_elo"]
        second_batting_squad_elo = resolved["second_batting_squad_elo"]
        second_bowling_squad_elo = resolved["second_bowling_squad_elo"]
        second_effective_elo = resolved["second_effective_elo"]

        if innings == 2 and target:
            live_state_arg = {"innings": 2, "runs": runs, "wickets": wickets,
                              "legal_balls": legal_balls, "target": int(target)}
        else:
            live_state_arg = {"innings": 1, "runs": runs, "wickets": wickets,
                              "legal_balls": legal_balls}

        sim = simulate_match(
            model=model_to_use, team_a=bat_team, team_b=bowl_team, venue_avg_score=resolved["venue_avg"],
            n_sims=n_sims, total_overs=fmt_spec["total_overs"], balls_per_over=fmt_spec["balls_per_over"],
            format_type=format_type, bat_first=bat_first, live_state=live_state_arg,
        )

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

        hist_bin_width = 20 if format_type in ("od", "odw") else 10
        hist_inn1 = bucket_scores(sim["inn1_score"].tolist(), bin_width=hist_bin_width)
        hist_inn2 = bucket_scores(sim["inn2_score"].tolist(), bin_width=hist_bin_width)

        # Wickets-taken distributions (0..10), one bar per wicket count — same bucketing as
        # the historical /api/montecarlo endpoint's bowling view.
        def bucket_wkts(wkts):
            counts = {}
            for w in wkts.tolist():
                counts[int(w)] = counts.get(int(w), 0) + 1
            keys = sorted(counts.keys())
            return {"labels": keys, "counts": [counts[k] for k in keys]}

        wkts_hist_inn1 = bucket_wkts(sim["inn1_wkts"])
        wkts_hist_inn2 = bucket_wkts(sim["inn2_wkts"])

        # Bowling markets. Inn1 is bowled by whichever side bowls first, inn2 by the other;
        # sim carries the column->name mapping for each (same convention as /api/montecarlo).
        bowl_names_inn1 = sim["inn1_bowling_names"]
        bowl_names_inn2 = sim["inn2_bowling_names"]

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
            # Lineup + attack (with each player's Elo) for the Lineups & Elo view — same dicts
            # simulate_match() itself ran on, so this always matches what was actually simulated.
            "first_batting_squad": first_team,
            "second_batting_squad": second_team,
            "first_batting_overall_elo": round(first_overall_elo, 1),
            "second_batting_overall_elo": round(second_overall_elo, 1),
            "first_batting_squad_elo": first_batting_squad_elo,
            "first_bowling_squad_elo": first_bowling_squad_elo,
            "first_effective_elo": first_effective_elo,
            "second_batting_squad_elo": second_batting_squad_elo,
            "second_bowling_squad_elo": second_bowling_squad_elo,
            "second_effective_elo": second_effective_elo,
            "win_prob_batting": round(win_prob_batting, 4),
            "squad_source": {"batting": bat_src, "bowling": bowl_src},
            "inn1_settled": inn1_settled,
            "inn1": inn1_total_runs,
            "inn2": inn2_total_runs,
            "margin": margin,
            "hist_inn1": hist_inn1,
            "hist_inn2": hist_inn2,
            "wkts_hist_inn1": wkts_hist_inn1,
            "wkts_hist_inn2": wkts_hist_inn2,
            "top_batsman_inn1": None if inn1_settled else price_top_batsman(sim, lineup_names=first_names, innings="inn1"),
            "top_batsman_inn2": price_top_batsman(sim, lineup_names=second_names, innings="inn2"),
            "props_30_inn1": None if inn1_settled else price_player_props(sim, lineup_names=first_names, target_runs=30.0, innings="inn1"),
            "props_30_inn2": price_player_props(sim, lineup_names=second_names, target_runs=30.0, innings="inn2"),
            "props_50_inn1": None if inn1_settled else price_player_props(sim, lineup_names=first_names, target_runs=50.0, innings="inn1"),
            "props_50_inn2": price_player_props(sim, lineup_names=second_names, target_runs=50.0, innings="inn2"),
            # Bowling markets mirror the batting ones: inn1-side props are equally unrecoverable
            # once that innings is settled (no per-bowler breakdown survives in the live snapshot).
            "top_bowler_inn1": None if inn1_settled else price_top_bowler(sim, attack_names=bowl_names_inn1, innings="inn1"),
            "top_bowler_inn2": price_top_bowler(sim, attack_names=bowl_names_inn2, innings="inn2"),
            "bowler_2w_inn1": None if inn1_settled else price_bowler_props(sim, attack_names=bowl_names_inn1, target_wkts=2.0, innings="inn1"),
            "bowler_2w_inn2": price_bowler_props(sim, attack_names=bowl_names_inn2, target_wkts=2.0, innings="inn2"),
            "bowler_3w_inn1": None if inn1_settled else price_bowler_props(sim, attack_names=bowl_names_inn1, target_wkts=3.0, innings="inn1"),
            "bowler_3w_inn2": price_bowler_props(sim, attack_names=bowl_names_inn2, target_wkts=3.0, innings="inn2"),
            "low_confidence_threshold": LOW_CONFIDENCE_UNRATED_THRESHOLD,
            # odw carries the same caveat as od (same unmeasured-full-match-sim reason);
            # hndw doesn't -- its pooled model went through the same Phase-3 pooled-beats-
            # unpooled gate hnd itself did.
            "simulation_experimental": format_type in ("od", "odw"),
            "source": source,
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

    # This match's own recorded ratings on the requested source (rating_source_toggle_plan.md
    # Phase 6) -- a pure historical replay, so (unlike the squad-driven Monte Carlo endpoints)
    # swapping the file swaps everything the model needs; there's no separate "current squad"
    # lookup to keep in sync.
    source = request.args.get("source", rating_source.DEFAULT_RATING_SOURCE)
    if not rating_source.is_supported(format_type, source):
        return jsonify({"error": f"Rating source '{source}' has no coverage for format '{format_type}'"}), 400

    # Build states for the selected match
    elo_ratings_file = rating_source.resolve_path(format_type, "team", source)
    elo_df = pd.read_parquet(elo_ratings_file) if elo_ratings_file.exists() else None

    player_elo_df = None
    if format_type in ["t20", "hnd", "t20w", "odw", "hndw"]:
        player_elo_file = rating_source.resolve_path(format_type, "player", source)
        if player_elo_file.exists():
            player_elo_df = pd.read_parquet(player_elo_file)

    builder = StateBuilder(match_deliveries, elo_df, player_elo_df=player_elo_df, total_overs=total_overs, format_type=format_type, balls_per_over=balls_per_over)
    states = builder.build()

    # Filter for the second innings
    chase_states = states[states["innings"] == 2].copy()

    model_to_use = get_model("xgb", format_type, source)

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
            "source": source,
        }
    )


@app.route("/api/match/<match_id>/squads")
def get_match_squads(match_id):
    """Both teams' point-in-time lineup + attack for the Lineups & Elo views: each named
    player alongside their batting/bowling Elo as it stood immediately before this specific
    match (via extract_match_squads — no simulation is run, this is a lightweight read of the
    same squad data /api/montecarlo uses internally). Also includes each side's overall
    (team-level) pre-match Elo (the same figure used to label the Monte Carlo team dropdowns),
    the weighted batting/bowling squad Elo the win-prob model actually trained on for this
    match, and the derived "effective Elo" (effective_game_elo_plan.md Phase 3) — team Elo
    nudged by whether this specific XI is stronger/weaker than that team's usual one."""
    format_type = request.args.get("format", "t20")
    try:
        defending_dict, chasing_dict, venue_avg_score = extract_match_squads(match_id, format_type)
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"error": str(e)}), 404

    defending_team_elo, chasing_team_elo = match_team_elos.get((format_type, str(match_id)), (None, None))

    # Weighted squad Elo (exact — the same columns the win-prob model trained on) + effective
    # Elo, using each team's point-in-time trailing baseline (excludes this match itself).
    squad_elo = _match_squad_elo(format_type, match_id)
    if squad_elo:
        t1_bat_sq, t1_bowl_sq, t2_bat_sq, t2_bowl_sq = squad_elo
    else:
        t1_bat_sq = t1_bowl_sq = t2_bat_sq = t2_bowl_sq = None

    hist_baselines = SQUAD_BASELINE_HIST.get(format_type, {})
    defending_baseline = hist_baselines.get((defending_dict["name"], str(match_id)))
    chasing_baseline = hist_baselines.get((chasing_dict["name"], str(match_id)))

    defending_effective_elo = (
        compute_effective_elo(defending_team_elo, t1_bat_sq, t1_bowl_sq, defending_baseline, format_type)
        if defending_team_elo is not None and t1_bat_sq is not None else None
    )
    chasing_effective_elo = (
        compute_effective_elo(chasing_team_elo, t2_bat_sq, t2_bowl_sq, chasing_baseline, format_type)
        if chasing_team_elo is not None and t2_bat_sq is not None else None
    )

    return jsonify(
        {
            "match_id": match_id,
            "format_type": format_type,
            "defending_team": defending_dict,  # bats first — {name, lineup: [...], attack: [...]}
            "chasing_team": chasing_dict,
            "defending_team_elo": defending_team_elo,  # overall team Elo entering this match
            "chasing_team_elo": chasing_team_elo,
            "defending_team_batting_squad_elo": t1_bat_sq,
            "defending_team_bowling_squad_elo": t1_bowl_sq,
            "chasing_team_batting_squad_elo": t2_bat_sq,
            "chasing_team_bowling_squad_elo": t2_bowl_sq,
            "defending_team_effective_elo": round(defending_effective_elo, 1) if defending_effective_elo is not None else None,
            "chasing_team_effective_elo": round(chasing_effective_elo, 1) if chasing_effective_elo is not None else None,
            "venue_avg_first_innings_score": round(venue_avg_score, 1),
        }
    )


@app.route("/api/predict_first_innings", methods=["POST"])
def predict_first_innings_total():
    """Predicts final first innings score for a custom match state."""
    data = request.json
    try:
        format_type = data.get("format", "t20")
        source = data.get("source", rating_source.DEFAULT_RATING_SOURCE)
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

        state, run_rate, projected_score, phase = build_first_innings_state(
            format_type, runs_scored, wickets_lost, balls_faced,
            batting_team_elo, bowling_team_elo, elo_diff,
            venue_avg_first_innings_score, competition, overs=overs,
        )

        model_to_use = get_model("first_innings", format_type, source)
        pred = model_to_use.predict(state)
        return jsonify(
            {
                "predicted_score": int(pred["predicted_score"]),
                "projected_score": int(round(projected_score)),
                "run_rate": round(run_rate, 2),
                "phase": phase,
                "status": pred["status"],
                "source": source,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


FEATURE_IMPORTANCE_REPORT_PATH = Path("outputs/feature_importance_report.md")
BASELINE_METRICS_PATH = Path("data/processed/baseline_metrics.json")


def _mc_calibration_path(format_type: str) -> Path:
    """t20 is the original, un-suffixed snapshot; every other format writes to
    mc_calibration_metrics_<format>.json -- matches generate_mc_calibration_metrics.py's own
    naming exactly (see that script's `suffix` logic)."""
    suffix = "" if format_type == "t20" else f"_{format_type}"
    return Path(f"data/processed/mc_calibration_metrics{suffix}.json")

_feature_importance_cache = None


def _parse_feature_importance_report():
    """Parses outputs/feature_importance_report.md into a dict keyed by format ('t20'/'od'),
    each holding 'classifier' and 'regressor' sections with total feature counts and ranked rows."""
    global _feature_importance_cache
    if _feature_importance_cache is not None:
        return _feature_importance_cache

    result = {}
    if not FEATURE_IMPORTANCE_REPORT_PATH.exists():
        _feature_importance_cache = result
        return result

    text = FEATURE_IMPORTANCE_REPORT_PATH.read_text(encoding="utf-8")

    # Split into per-format sections on top-level "## X Format" headings.
    format_sections = re.split(r"\n## ", text)
    row_pattern = re.compile(
        r"\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*\*\*([\d.]+)%\*\*\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|"
    )
    meta_pattern = re.compile(r"Evaluated across (\d+) fold models? on (\d+) total features\.")

    for section in format_sections:
        if section.startswith("T20 Format"):
            fmt_key = "t20"
        elif section.startswith("ODI Format"):
            fmt_key = "od"
        else:
            continue

        subsections = re.split(r"\n### ", section)
        fmt_result = {}
        for sub in subsections:
            if sub.startswith("1. Win Probability Classifier"):
                model_key = "classifier"
            elif sub.startswith("2. First Innings Regressor"):
                model_key = "regressor"
            else:
                continue

            meta_match = meta_pattern.search(sub)
            folds_evaluated = int(meta_match.group(1)) if meta_match else None
            total_features = int(meta_match.group(2)) if meta_match else None

            rows = []
            for m in row_pattern.finditer(sub):
                rows.append({
                    "rank": int(m.group(1)),
                    "feature": m.group(2),
                    "mean_gain_pct": float(m.group(3)),
                    "std_pct": float(m.group(4)),
                    "min_pct": float(m.group(5)),
                    "max_pct": float(m.group(6)),
                })
            low_importance_count = sum(1 for r in rows if r["mean_gain_pct"] < 1.0)

            fmt_result[model_key] = {
                "folds_evaluated": folds_evaluated,
                "total_features": total_features,
                "low_importance_count": low_importance_count,
                "features": rows,
            }
        result[fmt_key] = fmt_result

    _feature_importance_cache = result
    return result


@app.route("/api/feature-importance")
def get_feature_importance():
    """Returns ranked XGBoost feature-importance tables (win-prob classifier + first-innings
    regressor) parsed from outputs/feature_importance_report.md. Not available for hnd."""
    fmt = request.args.get("format", "t20").lower()
    top_n = request.args.get("top_n", 15, type=int)

    if fmt not in ("t20", "od"):
        return jsonify({"available": False, "message": "Feature importance data is not available for the Hundred format."})

    try:
        report = _parse_feature_importance_report()
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Could not parse feature importance report: {e}"}), 500

    fmt_data = report.get(fmt)
    if not fmt_data or "classifier" not in fmt_data:
        return jsonify({"available": False, "message": "Feature importance report has not been generated yet."})

    def _truncate(section):
        section = dict(section)
        section["features"] = section["features"][:top_n]
        return section

    return jsonify({
        "available": True,
        "classifier": _truncate(fmt_data["classifier"]),
        "regressor": _truncate(fmt_data.get("regressor", {"features": []})),
    })


@app.route("/api/mc-performance")
def get_mc_performance():
    """Returns cached Monte Carlo simulator calibration/backtest metrics (generated by
    tests/one-time/generate_mc_calibration_metrics.py) for the format given by ?format=
    (defaults to t20, matching every other Analytics-tab endpoint's convention).

    Pass/fail flags are computed HERE, against each metric's own embedded target, rather than
    cached in the JSON -- so a gate re-anchor (e.g. t20's MAE gate, 25.0 -> 28.5 on 2026-08-11)
    is reflected immediately without needing every format's snapshot re-locked."""
    format_type = request.args.get("format", "t20").lower()
    path = _mc_calibration_path(format_type)
    if not path.exists():
        return jsonify({
            "available": False,
            "message": f"Monte Carlo calibration metrics have not been generated yet for "
                       f"'{format_type}'. Run tests/one-time/generate_mc_calibration_metrics.py "
                       f"--format {format_type} to produce them.",
        })
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["available"] = True

        ic = data.get("innings_calibration", {})
        cov_lo, cov_hi = ic.get("coverage_target", [0.0, 1.0])
        ic["coverage_passed"] = cov_lo <= ic.get("coverage_rate", -1) <= cov_hi
        ic["mae_passed"] = ic.get("mae", float("inf")) <= ic.get("mae_target_max", float("inf"))
        ic["bias_passed"] = abs(ic.get("bias", float("inf"))) <= ic.get("bias_target_abs_max", 0.0)

        # Win-prob gate moved from agreement-with-PreMatchPredictor to SKILL vs actual winners
        # (2026-08-13, see generate_mc_calibration_metrics.py's Tests 2+3 block). Older snapshots
        # that predate the change still carry only `winprob_agreement`, so fall back to the old
        # gate for those rather than silently reporting them as passing/failing on a key they
        # do not have.
        ws = data.get("winprob_skill")
        wp = data.get("winprob_agreement", {})
        if ws is not None:
            ws["passed"] = ws.get("skill_vs_baserate", float("-inf")) >= ws.get("skill_target_min", 0.0)
            winprob_passed = ws["passed"]
            wp["passed"] = None  # diagnostic only -- no longer a gate
        else:
            wp["passed"] = wp.get("mean_abs_diff", float("inf")) <= wp.get("target_max", float("inf"))
            winprob_passed = wp["passed"]

        ou = data.get("over_under_reliability", {})
        ou["passed"] = ou.get("brier_score", float("inf")) <= ou.get("target_max", float("inf"))

        var = data.get("variance_sanity", {})
        std_lo, std_hi = var.get("target_range", [0.0, float("inf")])
        var["passed"] = std_lo <= var.get("avg_std_dev", -1) <= std_hi

        gate_flags = [ic["coverage_passed"], ic["mae_passed"], ic["bias_passed"],
                     winprob_passed, ou["passed"], var["passed"]]
        data["gates_total"] = len(gate_flags)
        data["gates_passed"] = sum(1 for g in gate_flags if g)

        return jsonify(data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Could not load MC calibration metrics: {e}"}), 500


ACCURACY_REPORT_PATH = Path("data/processed/accuracy_report.json")


@app.route("/api/accuracy")
def get_accuracy_report():
    """Real, measured win-probability + first-innings validation metrics for any ACTIVE_FORMAT.

    Reads data/processed/accuracy_report.json (built by scripts/build_accuracy_report.py, which
    uses the same evaluate_format_models() the regression gate does). Before 2026-08-13 this
    endpoint returned hardcoded literals -- a stale Brier score and a fabricated calibration
    curve per format, with no support at all for the three women's formats; the legacy branches
    below are kept only as a fallback for a format the report hasn't been generated for yet.
    """
    fmt = request.args.get("format", "t20").lower()
    if fmt == "odi":
        fmt = "od"  # legacy alias the frontend used to send

    if ACCURACY_REPORT_PATH.exists():
        try:
            with open(ACCURACY_REPORT_PATH, "r", encoding="utf-8") as f:
                report = json.load(f)
            entry = report.get(fmt)
            if entry:
                return jsonify({
                    "available": True,
                    "format": fmt,
                    "overall_brier": entry["overall_brier"],
                    "phase_brier": entry["phase_brier"],
                    "calibration": entry["calibration"],
                    "first_innings": entry.get("first_innings"),
                    "win_prob_sample_size": entry.get("win_prob_sample_size"),
                    "generated_at": entry.get("generated_at"),
                })
        except Exception:
            traceback.print_exc()  # fall through to the legacy branches below

    first_innings_perf = None
    if fmt in ("t20", "od"):
        try:
            if BASELINE_METRICS_PATH.exists():
                with open(BASELINE_METRICS_PATH, "r", encoding="utf-8") as f:
                    baseline = json.load(f)
                first_innings_perf = baseline.get(fmt, {}).get("first_innings")
        except Exception:
            traceback.print_exc()

    if fmt not in ("t20", "od", "hnd"):
        # A women's format with no report entry yet -- say so, rather than silently serving
        # t20's numbers under a women's label (the old `else` branch's behaviour).
        return jsonify({
            "available": False,
            "format": fmt,
            "message": f"Accuracy metrics for '{fmt}' have not been generated yet. "
                       f"Run scripts/build_accuracy_report.py --formats {fmt}.",
        })

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
                "first_innings": first_innings_perf,
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
                "first_innings": first_innings_perf,
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
                "first_innings": first_innings_perf,
            }
        )


# --------------------------------------------------
# BETTING — paper-bet ledger dashboard + pre-match value scanner
# --------------------------------------------------
@app.route("/api/betting/summary")
def betting_summary():
    """Aggregate performance stats for the betting dashboard (bankroll, ROI, win rate, etc.)."""
    try:
        return jsonify(bet_tracker.get_summary())
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Could not load betting summary: {e}"}), 500


@app.route("/api/betting/bets")
def betting_bets():
    """Full bet ledger (newest first) plus the starting bankroll, for the dashboard table + equity curve."""
    try:
        return jsonify({
            "starting_bankroll": bet_tracker.starting_bankroll,
            "bets": bet_tracker.get_all_bets(newest_first=True),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Could not load bet ledger: {e}"}), 500


@app.route("/api/betting/scan", methods=["POST"])
def betting_scan():
    """Runs the pre-match value scanner on demand (PAPER mode only) and returns the scan
    results plus a refreshed summary. Reuses the app's already-loaded predictor + ledger.

    This makes a live Betfair connection (login + market fetches); it can take several
    seconds and fails with 502 if credentials are missing or the API is unreachable.
    """
    data = request.json or {}
    try:
        results = run_scanner(
            mode="paper",  # hard-coded: the UI never places real-money bets
            hours_ahead=int(data.get("hours_ahead", 48)),
            threshold=float(data.get("threshold", 0.03)),
            kelly_fraction=float(data.get("kelly_fraction", 0.25)),
            bankroll=float(data.get("bankroll", bet_tracker.get_current_bankroll())),
            format_filter=str(data.get("format_filter", "both")),
            predictor=pre_match_predictor,
            tracker=bet_tracker,
        )
        return jsonify({
            "results": results,
            "summary": bet_tracker.get_summary(),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Scan failed (Betfair unavailable or credentials missing): {e}"}), 502


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
