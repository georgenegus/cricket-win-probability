import os
import math
import numpy as np
import pandas as pd
from pathlib import Path


class PhaseEloEngine:
    """
    Computes chronological rolling Elo ratings for teams per format, split by match phases:
    - powerplay
    - middle
    - death
    For both batting and bowling roles.
    """

    def __init__(
        self,
        k_runs=None,
        k_wickets=None,
        default_elo=1500.0,
        processed_path="data/processed",
        format_type="t20",
    ):
        self.format_type = format_type
        if k_runs is None:
            self.k_runs = 0.2 if format_type == "od" else 0.5
        else:
            self.k_runs = float(k_runs)

        if k_wickets is None:
            self.k_wickets = 8.0 if format_type == "od" else 5.0
        else:
            self.k_wickets = float(k_wickets)

        self.default_elo = float(default_elo)
        self.processed_path = Path(processed_path)
        self.cache_file = self.processed_path / f"{format_type}_phase_elo_ratings.parquet"
        os.makedirs(self.processed_path, exist_ok=True)

        self.phases = ["powerplay", "middle", "death"]
        self.ratings = {p: {} for p in self.phases}

        # Dynamic baseline rates per ball per phase
        self.baseline_runs = {}
        self.baseline_wickets = {}

    def get_elo(self, team: str, role: str, phase: str) -> float:
        phase_pool = self.ratings.setdefault(phase, {})
        team_ratings = phase_pool.setdefault(team, {})
        return team_ratings.setdefault(role, self.default_elo)

    def _compute_dynamic_baselines(self, deliveries_df: pd.DataFrame):
        df = deliveries_df.copy()
        df["over_float"] = df["over"] + (df["ball"] - 1) / 6.0

        if self.format_type == "od":
            pp_end = 10.0
            death_start = 40.0
        else:
            pp_end = 6.0
            death_start = 15.0

        df["phase"] = np.where(
            df["over_float"] < pp_end,
            "powerplay",
            np.where(
                df["over_float"] < death_start,
                "middle",
                "death"
            )
        )

        if "legal_delivery" not in df.columns:
            df["legal_delivery"] = (
                (df["is_wide"] == 0) & (df["is_noball"] == 0)
            ).astype(int)

        for phase in self.phases:
            phase_df = df[df["phase"] == phase]
            legal_balls = phase_df["legal_delivery"].sum()
            if legal_balls > 0:
                self.baseline_runs[phase] = float(phase_df["total_runs"].sum() / legal_balls)
                self.baseline_wickets[phase] = float(phase_df["wicket"].sum() / legal_balls)
            else:
                # Fallback values
                if self.format_type == "od":
                    self.baseline_runs[phase] = 1.33 if phase == "death" else 0.83
                    self.baseline_wickets[phase] = 0.05 if phase == "death" else 0.025
                else:
                    self.baseline_runs[phase] = 1.50 if phase == "death" else 1.25
                    self.baseline_wickets[phase] = 0.093 if phase == "death" else 0.04

        print(f"Dynamic baselines for {self.format_type}:")
        for phase in self.phases:
            print(f"  {phase}: runs/ball = {self.baseline_runs[phase]:.4f}, wickets/ball = {self.baseline_wickets[phase]:.4f}")

    def _aggregate_match_phase_stats(self, deliveries_df: pd.DataFrame):
        df = deliveries_df.copy()
        df["over_float"] = df["over"] + (df["ball"] - 1) / 6.0

        if self.format_type == "od":
            pp_end = 10.0
            death_start = 40.0
        else:
            pp_end = 6.0
            death_start = 15.0

        df["phase"] = np.where(
            df["over_float"] < pp_end,
            "powerplay",
            np.where(
                df["over_float"] < death_start,
                "middle",
                "death"
            )
        )

        if "legal_delivery" not in df.columns:
            df["legal_delivery"] = (
                (df["is_wide"] == 0) & (df["is_noball"] == 0)
            ).astype(int)

        agg_df = (
            df.groupby(["match_id", "innings", "batting_team", "phase"])
            .agg(
                runs=("total_runs", "sum"),
                wickets=("wicket", "sum"),
                balls=("legal_delivery", "sum")
            )
            .reset_index()
        )
        return agg_df

    def run(self, deliveries_df: pd.DataFrame, elo_df: pd.DataFrame, force=False) -> pd.DataFrame:
        if not force and self.cache_file.exists():
            print(f"Loading phase Elo ratings from cache: {self.cache_file}")
            return pd.read_parquet(self.cache_file)

        print("Computing rolling phase Elo ratings...")
        self.ratings = {p: {} for p in self.phases}
        self._compute_dynamic_baselines(deliveries_df)

        # Aggregate delivery stats
        agg_df = self._aggregate_match_phase_stats(deliveries_df)
        agg_records = agg_df.to_dict("records")

        # Group by match_id for fast retrieval
        match_phase_groups = {}
        for r in agg_records:
            match_phase_groups.setdefault(r["match_id"], []).append(r)

        # Sort matches chronologically by match_date
        sorted_matches = elo_df.sort_values(by=["match_date", "match_id"])[
            ["match_id", "match_date", "team_1", "team_2"]
        ].drop_duplicates()

        phase_elo_records = []

        for _, match_row in sorted_matches.iterrows():
            match_id = match_row["match_id"]
            t1 = match_row["team_1"]
            t2 = match_row["team_2"]

            # Store pre-match phase Elo ratings
            rec = {"match_id": match_id}
            for phase in self.phases:
                rec[f"team_1_batting_elo_{phase}"] = self.get_elo(t1, "bat", phase)
                rec[f"team_1_bowling_elo_{phase}"] = self.get_elo(t1, "bowl", phase)
                rec[f"team_2_batting_elo_{phase}"] = self.get_elo(t2, "bat", phase)
                rec[f"team_2_bowling_elo_{phase}"] = self.get_elo(t2, "bowl", phase)

            phase_elo_records.append(rec)

            # Match lookup mapping for opposition team
            opp_team = {t1: t2, t2: t1}

            # Update Elos based on match stats
            match_stats = match_phase_groups.get(match_id, [])
            for stat in match_stats:
                phase = stat["phase"]
                bat_team = stat["batting_team"]
                bowl_team = opp_team.get(bat_team, None)
                if not bowl_team:
                    continue

                runs = stat["runs"]
                wickets = stat["wickets"]
                balls = stat["balls"]

                if balls > 0:
                    e_bat = self.get_elo(bat_team, "bat", phase)
                    e_bowl = self.get_elo(bowl_team, "bowl", phase)

                    # Expectation models
                    exp_runs_rate = np.clip(
                        self.baseline_runs[phase] * (10.0 ** ((e_bat - e_bowl) / 400.0)),
                        0.1,
                        3.0
                    )
                    exp_wickets_rate = np.clip(
                        self.baseline_wickets[phase] * (10.0 ** ((e_bowl - e_bat) / 400.0)),
                        0.001,
                        0.5
                    )

                    expected_runs = exp_runs_rate * balls
                    expected_wickets = exp_wickets_rate * balls

                    delta_runs = runs - expected_runs
                    delta_wickets = wickets - expected_wickets

                    # Update step
                    delta_bat = self.k_runs * delta_runs - self.k_wickets * delta_wickets
                    delta_bowl = -self.k_runs * delta_runs + self.k_wickets * delta_wickets

                    self.ratings[phase].setdefault(bat_team, {})["bat"] = e_bat + delta_bat
                    self.ratings[phase].setdefault(bowl_team, {})["bowl"] = e_bowl + delta_bowl

        df = pd.DataFrame(phase_elo_records)
        print(f"Saving phase Elo ratings to cache: {self.cache_file}")
        df.to_parquet(self.cache_file, index=False, compression="snappy")
        return df

    def clear_cache(self):
        if self.cache_file.exists():
            self.cache_file.unlink()
            print(f"Cleared phase Elo ratings cache: {self.cache_file}")
