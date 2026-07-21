import os
import math
import numpy as np
import pandas as pd
from pathlib import Path


from src.utils.cache_manager import CacheManager


class PlayerEloEngine:
    """
    Computes chronological rolling Elo ratings for individual batters and bowlers.
    Incorporates Bayesian shrinkage toward team Elo to handle sparse/new players.
    """


    def __init__(
        self,
        k_runs=None,
        k_wickets=None,
        decay_days=180.0,
        decay_rate=0.95,
        default_elo=1500.0,
        min_matches_reliable=10,
        processed_path="data/processed",
        format_type="t20",
        form_window=5,
    ):
        self.format_type = format_type
        if k_runs is None:
            k_runs = 0.8 if format_type == "od" else 0.3
        if k_wickets is None:
            k_wickets = 12.0 if format_type == "od" else 3.0

        self.k_runs = float(k_runs)
        self.k_wickets = float(k_wickets)
        self.decay_days = float(decay_days)
        self.decay_rate = float(decay_rate)
        self.default_elo = float(default_elo)
        self.min_matches_reliable = float(min_matches_reliable)
        self.processed_path = Path(processed_path)
        self.form_window = int(form_window)
        self.cache_file = self.processed_path / f"{format_type}_player_elo_ratings.parquet"
        os.makedirs(self.processed_path, exist_ok=True)

        # Base format averages
        if format_type == "od":
            self.avg_runs_per_ball = 0.85
            self.avg_wickets_per_ball = 0.026
        else:
            self.avg_runs_per_ball = 1.25
            self.avg_wickets_per_ball = 0.054

        # State tracking: ratings and match counts
        self.batter_ratings = {}       # player -> elo
        self.bowler_ratings = {}       # player -> elo
        self.batter_match_counts = {}  # player -> count
        self.bowler_match_counts = {}  # player -> count
        self.batter_last_dates = {}    # player -> date
        self.bowler_last_dates = {}    # player -> date
        
        # Recent deltas tracking for form features (Proposal 5)
        self.batter_recent_deltas = {}  # player -> list of Elo updates
        self.bowler_recent_deltas = {}  # player -> list of Elo updates

    def get_raw_elo(self, player: str, role: str) -> float:
        if role == "batter":
            return self.batter_ratings.setdefault(player, self.default_elo)
        else:
            return self.bowler_ratings.setdefault(player, self.default_elo)

    def get_match_count(self, player: str, role: str) -> int:
        if role == "batter":
            return self.batter_match_counts.setdefault(player, 0)
        else:
            return self.bowler_match_counts.setdefault(player, 0)

    def get_last_date(self, player: str, role: str):
        if role == "batter":
            return self.batter_last_dates.get(player, None)
        else:
            return self.bowler_last_dates.get(player, None)

    def get_effective_elo(self, player: str, role: str, team_elo: float) -> float:
        """
        Bayesian shrinkage toward team Elo for players with limited match experience.
        """
        raw_elo = self.get_raw_elo(player, role)
        n_matches = self.get_match_count(player, role)

        # Blend factor (Sigmoid-like blend)
        reliability = n_matches / (n_matches + self.min_matches_reliable)
        return reliability * raw_elo + (1.0 - reliability) * team_elo

    def calculate_decayed_elo(self, elo: float, last_date, current_date) -> float:
        if not last_date or not current_date:
            return elo
        try:
            d1 = last_date if isinstance(last_date, pd.Timestamp) else pd.to_datetime(last_date)
            d2 = current_date if isinstance(current_date, pd.Timestamp) else pd.to_datetime(current_date)
            days = (d2 - d1).days
            if days > self.decay_days:
                periods = (days - self.decay_days) / self.decay_days
                decay_factor = self.decay_rate ** periods
                return self.default_elo + (elo - self.default_elo) * decay_factor
        except Exception:
            pass
        return elo

    def _aggregate_match_stats(self, deliveries_df: pd.DataFrame):
        """
        Aggregates deliveries into match-level batter and bowler statistics.
        """
        # Create legal delivery flag if not present
        if "legal_delivery" not in deliveries_df.columns:
            deliveries_df = deliveries_df.copy()
            deliveries_df["legal_delivery"] = (
                (deliveries_df["is_wide"] == 0) & (deliveries_df["is_noball"] == 0)
            ).astype(int)

        # Batting stats
        bat_stats = (
            deliveries_df.groupby(["match_id", "batting_team", "batter"])
            .agg(
                runs=("runs_off_bat", "sum"),
                balls_faced=("legal_delivery", "sum"),
                dismissals=("wicket", "sum"),
            )
            .reset_index()
        )
        # Cap dismissals at 1 per innings
        bat_stats["dismissals"] = np.minimum(bat_stats["dismissals"], 1)

        # Bowling stats
        # Opposing team is the one batting (batting_team).
        # Cast batting_team to string first to avoid pandas categorical list aggregation error
        batting_teams_str = deliveries_df[["match_id", "batting_team"]].copy()
        batting_teams_str["batting_team"] = batting_teams_str["batting_team"].astype(str)
        bowling_teams = (
            batting_teams_str.drop_duplicates()
            .groupby("match_id")
            .agg(teams=("batting_team", list))
            .reset_index()
        )

        bowl_stats = (
            deliveries_df.groupby(["match_id", "batting_team", "bowler"])
            .agg(
                runs_conceded=("total_runs", "sum"),
                balls_bowled=("legal_delivery", "sum"),
                wickets=("wicket", "sum"),
            )
            .reset_index()
        )

        return bat_stats, bowl_stats, bowling_teams

    def run(self, deliveries_df: pd.DataFrame, elo_df: pd.DataFrame, force=False, precomputed_stats=None) -> pd.DataFrame:
        """
        Computes rolling player Elo ratings chronologically.

        Args:
            deliveries_df: Raw ball-by-ball deliveries DataFrame.
            elo_df: Team Elo ratings DataFrame.
            force: If True, recompute even if cache exists.
            precomputed_stats: Optional tuple of (bat_stats, bowl_stats, bowling_teams)
                from a prior call to _aggregate_match_stats(). Avoids re-aggregating
                deliveries in sweep scripts where the raw data doesn't change.
        """
        sources = [Path(__file__)]
        registry_file = Path(__file__).parent / "build_player_registry.py"
        if registry_file.exists():
            sources.append(registry_file)

        if not force and CacheManager.is_cache_valid(self.cache_file, sources):
            print(f"Loading player Elo ratings from cache: {self.cache_file}")
            return pd.read_parquet(self.cache_file)


        print("Computing rolling player Elo ratings...")
        # Reset internal states to ensure a clean run
        self.batter_ratings = {}
        self.bowler_ratings = {}
        self.batter_match_counts = {}
        self.bowler_match_counts = {}
        self.batter_last_dates = {}
        self.bowler_last_dates = {}
        self.batter_recent_deltas = {}
        self.bowler_recent_deltas = {}

        # Pre-aggregate match stats (or use precomputed)
        if precomputed_stats is not None:
            bat_stats, bowl_stats, bowling_teams = precomputed_stats
        else:
            bat_stats, bowl_stats, bowling_teams = self._aggregate_match_stats(deliveries_df)

        # Convert to list of dict records grouped by match_id for O(1) loop lookups
        bat_records = bat_stats.to_dict("records")
        bat_groups = {}
        for r in bat_records:
            bat_groups.setdefault(r["match_id"], []).append(r)

        bowl_records = bowl_stats.to_dict("records")
        bowl_groups = {}
        for r in bowl_records:
            bowl_groups.setdefault(r["match_id"], []).append(r)

        # Index bowling teams for fast lookup
        opp_team_map = {}
        for _, row in bowling_teams.iterrows():
            m_id = row["match_id"]
            teams = row["teams"]
            if len(teams) == 2:
                opp_team_map[m_id] = {teams[0]: teams[1], teams[1]: teams[0]}
            else:
                opp_team_map[m_id] = {}

        # Map match_id to pre-match team Elo ratings
        cols = ["team_1", "team_2", "team_1_elo", "team_2_elo"]
        if "team_1_players" in elo_df.columns:
            cols.extend(["team_1_players", "team_2_players"])
        team_elo_map = elo_df.set_index("match_id")[cols].to_dict("index")

        # Sort matches chronologically by match_date
        sorted_matches = elo_df.sort_values(by=["match_date", "match_id"])[["match_id", "match_date"]].drop_duplicates()

        player_elo_records = []

        for _, match_row in sorted_matches.iterrows():
            match_id = match_row["match_id"]
            match_date = match_row["match_date"]

            t_elos = team_elo_map.get(match_id, {})
            if not t_elos:
                continue

            t1, t2 = t_elos["team_1"], t_elos["team_2"]
            t1_elo, t2_elo = t_elos["team_1_elo"], t_elos["team_2_elo"]

            # Dict for opposition team Elo lookup
            team_strength = {t1: t1_elo, t2: t2_elo}

            # Parse squad player lists
            t1_players_str = t_elos.get("team_1_players", "")
            t2_players_str = t_elos.get("team_2_players", "")
            t1_players = [p.strip() for p in t1_players_str.split(",") if p.strip()] if isinstance(t1_players_str, str) else []
            t2_players = [p.strip() for p in t2_players_str.split(",") if p.strip()] if isinstance(t2_players_str, str) else []

            # Pre-decay playing XI players to match_date and build lookup maps of pre-match effective Elo
            match_batter_eff_elos = {}
            match_bowler_eff_elos = {}

            for p in t1_players:
                b_elo = self.get_raw_elo(p, "batter")
                b_elo = self.calculate_decayed_elo(b_elo, self.get_last_date(p, "batter"), match_date)
                self.batter_ratings[p] = b_elo
                match_batter_eff_elos[p] = self.get_effective_elo(p, "batter", t1_elo)

                bowl_elo = self.get_raw_elo(p, "bowler")
                bowl_elo = self.calculate_decayed_elo(bowl_elo, self.get_last_date(p, "bowler"), match_date)
                self.bowler_ratings[p] = bowl_elo
                match_bowler_eff_elos[p] = self.get_effective_elo(p, "bowler", t1_elo)

            for p in t2_players:
                b_elo = self.get_raw_elo(p, "batter")
                b_elo = self.calculate_decayed_elo(b_elo, self.get_last_date(p, "batter"), match_date)
                self.batter_ratings[p] = b_elo
                match_batter_eff_elos[p] = self.get_effective_elo(p, "batter", t2_elo)

                bowl_elo = self.get_raw_elo(p, "bowler")
                bowl_elo = self.calculate_decayed_elo(bowl_elo, self.get_last_date(p, "bowler"), match_date)
                self.bowler_ratings[p] = bowl_elo
                match_bowler_eff_elos[p] = self.get_effective_elo(p, "bowler", t2_elo)

            # Compute squad Elo averages (using pre-match effective Elo ratings)
            if t1_players:
                t1_bat_squad = float(np.mean([match_batter_eff_elos.get(p, t1_elo) for p in t1_players]))
                t1_bowl_squad = float(np.mean([match_bowler_eff_elos.get(p, t1_elo) for p in t1_players]))
            else:
                t1_bat_squad = float(t1_elo)
                t1_bowl_squad = float(t1_elo)

            if t2_players:
                t2_bat_squad = float(np.mean([match_batter_eff_elos.get(p, t2_elo) for p in t2_players]))
                t2_bowl_squad = float(np.mean([match_bowler_eff_elos.get(p, t2_elo) for p in t2_players]))
            else:
                t2_bat_squad = float(t2_elo)
                t2_bowl_squad = float(t2_elo)

            # 1. Batters
            match_batters = bat_groups.get(match_id, [])
            for row in match_batters:
                batter = row["batter"]
                batting_team = row["batting_team"]
                opp_team = opp_team_map.get(match_id, {}).get(batting_team, None)
                opp_elo = team_strength.get(opp_team, 1500.0)

                # Get pre-match Elo (already decayed at loop start)
                b_elo = self.batter_ratings.get(batter, self.default_elo)

                # Get pre-match form (Proposal 5)
                b_deltas = self.batter_recent_deltas.get(batter, [])
                b_form = float(np.mean(b_deltas)) if b_deltas else 0.0

                # Record pre-match ratings
                player_elo_records.append({
                    "match_id": match_id,
                    "player": batter,
                    "role": "batter",
                    "raw_elo": b_elo,
                    "effective_elo": match_batter_eff_elos.get(batter, self.get_effective_elo(batter, "batter", team_strength.get(batting_team, 1500.0))),
                    "recent_form": b_form,
                    "team_1_batting_squad_elo": t1_bat_squad,
                    "team_1_bowling_squad_elo": t1_bowl_squad,
                    "team_2_batting_squad_elo": t2_bat_squad,
                    "team_2_bowling_squad_elo": t2_bowl_squad,
                })

                # Compute updates
                balls = row["balls_faced"]
                if balls > 0:
                    # Opposing bowling quality is squad bowling Elo for T20, team Elo for ODI
                    opp_eff_elo = (t2_bowl_squad if batting_team == t1 else t1_bowl_squad) if self.format_type == "t20" else opp_elo

                    exp_runs_per_ball = np.clip(
                        self.avg_runs_per_ball * (10.0 ** ((b_elo - opp_eff_elo) / 400.0)), 0.1, 3.0
                    )
                    exp_wickets_per_ball = np.clip(
                        self.avg_wickets_per_ball * (10.0 ** ((opp_eff_elo - b_elo) / 400.0)), 0.001, 0.5
                    )

                    expected_runs = exp_runs_per_ball * balls
                    expected_dismissals = exp_wickets_per_ball * balls

                    delta_runs = row["runs"] - expected_runs
                    delta_wickets = row["dismissals"] - expected_dismissals

                    new_elo = b_elo + self.k_runs * delta_runs - self.k_wickets * delta_wickets
                    self.batter_ratings[batter] = new_elo

                    # Track delta for recent form (Proposal 5)
                    self.batter_recent_deltas.setdefault(batter, []).append(new_elo - b_elo)
                    if len(self.batter_recent_deltas[batter]) > self.form_window:
                        self.batter_recent_deltas[batter].pop(0)

                self.batter_match_counts[batter] = self.get_match_count(batter, "batter") + 1
                self.batter_last_dates[batter] = match_date

            # 2. Bowlers
            match_bowlers = bowl_groups.get(match_id, [])
            for row in match_bowlers:
                bowler = row["bowler"]
                # Determine bowling team and opposing batting team
                opp_team = row["batting_team"]
                bowl_team = opp_team_map.get(match_id, {}).get(opp_team, None)

                opp_elo = team_strength.get(opp_team, 1500.0)
                bowl_team_elo = team_strength.get(bowl_team, 1500.0)

                # Get pre-match Elo (already decayed at loop start)
                bowl_elo = self.bowler_ratings.get(bowler, self.default_elo)

                # Get pre-match form (Proposal 5)
                bowl_deltas = self.bowler_recent_deltas.get(bowler, [])
                bowl_form = float(np.mean(bowl_deltas)) if bowl_deltas else 0.0

                # Record pre-match ratings
                player_elo_records.append({
                    "match_id": match_id,
                    "player": bowler,
                    "role": "bowler",
                    "raw_elo": bowl_elo,
                    "effective_elo": match_bowler_eff_elos.get(bowler, self.get_effective_elo(bowler, "bowler", bowl_team_elo)),
                    "recent_form": bowl_form,
                    "team_1_batting_squad_elo": t1_bat_squad,
                    "team_1_bowling_squad_elo": t1_bowl_squad,
                    "team_2_batting_squad_elo": t2_bat_squad,
                    "team_2_bowling_squad_elo": t2_bowl_squad,
                })

                # Compute updates
                balls = row["balls_bowled"]
                if balls > 0:
                    # Opposing batting quality is squad batting Elo for T20, team Elo for ODI
                    opp_eff_elo = (t2_bat_squad if bowl_team == t1 else t1_bat_squad) if self.format_type == "t20" else opp_elo

                    exp_runs_per_ball = np.clip(
                        self.avg_runs_per_ball * (10.0 ** ((opp_eff_elo - bowl_elo) / 400.0)), 0.1, 3.0
                    )
                    exp_wickets_per_ball = np.clip(
                        self.avg_wickets_per_ball * (10.0 ** ((bowl_elo - opp_eff_elo) / 400.0)), 0.001, 0.5
                    )

                    expected_runs = exp_runs_per_ball * balls
                    expected_wickets = exp_wickets_per_ball * balls

                    delta_runs = row["runs_conceded"] - expected_runs
                    delta_wickets = row["wickets"] - expected_wickets

                    new_elo = bowl_elo - self.k_runs * delta_runs + self.k_wickets * delta_wickets
                    self.bowler_ratings[bowler] = new_elo

                    # Track delta for recent form (Proposal 5)
                    self.bowler_recent_deltas.setdefault(bowler, []).append(new_elo - bowl_elo)
                    if len(self.bowler_recent_deltas[bowler]) > self.form_window:
                        self.bowler_recent_deltas[bowler].pop(0)

                self.bowler_match_counts[bowler] = self.get_match_count(bowler, "bowler") + 1
                self.bowler_last_dates[bowler] = match_date

        df = pd.DataFrame(player_elo_records)
        print(f"Saving player Elo ratings to cache: {self.cache_file}")
        df.to_parquet(self.cache_file, index=False, compression="snappy")
        CacheManager.register_cache(self.cache_file, sources)
        return df

    def clear_cache(self):
        CacheManager.clear_cache(self.cache_file)

