import pandas as pd
import numpy as np


class StateBuilder:

    def __init__(self, df: pd.DataFrame, elo_df: pd.DataFrame = None, player_elo_df: pd.DataFrame = None, phase_elo_df: pd.DataFrame = None, total_overs=20, rolling_window=None):
        self.df = df.copy()
        self.elo_df = elo_df
        self.player_elo_df = player_elo_df
        self.phase_elo_df = phase_elo_df
        self.total_overs = total_overs
        self.rolling_window = rolling_window if rolling_window is not None else (18 if total_overs <= 20 else 36)

    def add_over_info(self):
        self.df["over_float"] = (
            self.df["over"] +
            (self.df["ball"] - 1) / 6
        )
        return self

    def add_phase(self):
        target_overs_col = self.df.get("target_overs", pd.Series(np.nan, index=self.df.index))
        eff_overs = np.where(
            (self.df["innings"] == 2) & target_overs_col.notna(),
            target_overs_col,
            self.total_overs
        )

        if self.total_overs > 20:
            powerplay_end = np.where(eff_overs > 20, 10.0, eff_overs * 0.3)
            death_start = np.where(eff_overs > 20, 40.0, eff_overs * 0.75)
        else:
            powerplay_end = eff_overs * 0.3
            death_start = eff_overs * 0.75

        self.df["phase"] = np.where(
            self.df["over_float"] < powerplay_end,
            "powerplay",
            np.where(
                self.df["over_float"] < death_start,
                "middle",
                "death"
            )
        )
        return self

    def build_innings_state(self):

        self.df = self.df.sort_values(
            ["match_id", "innings", "over", "ball"]
        )

        # --------------------------------------------------
        # Innings cumulative runs
        # --------------------------------------------------

        self.df["runs_scored"] = (
            self.df.groupby(
                ["match_id", "innings"]
            )["total_runs"]
            .cumsum()
        )

        # --------------------------------------------------
        # Innings cumulative wickets
        # --------------------------------------------------

        self.df["wickets_lost"] = (
            self.df.groupby(
                ["match_id", "innings"]
            )["wicket"]
            .cumsum()
        )

        print("WICKET RAW COUNTS")
        print(self.df["wicket"].value_counts(dropna=False))

        # --------------------------------------------------
        # Legal delivery flag
        # --------------------------------------------------

        self.df["legal_delivery"] = (
            (self.df["is_wide"] == 0)
            & (self.df["is_noball"] == 0)
        ).astype(int)

        # --------------------------------------------------
        # Legal balls faced
        # --------------------------------------------------

        self.df["balls_faced"] = (
            self.df.groupby(
                ["match_id", "innings"]
            )["legal_delivery"]
            .cumsum()
        )

        target_overs_col = self.df.get("target_overs", pd.Series(np.nan, index=self.df.index))
        eff_overs = np.where(
            (self.df["innings"] == 2) & target_overs_col.notna(),
            target_overs_col,
            self.total_overs
        )
        eff_overs_floor = np.floor(eff_overs)
        eff_balls_allowed = eff_overs_floor * 6 + np.round((eff_overs - eff_overs_floor) * 10)
        bad = self.df[self.df["balls_faced"] > eff_balls_allowed]

        if len(bad) > 0:
            print(f"\nWARNING: {len(bad)} invalid states found and dropped\n")

            print(
                bad[
                    [
                        "match_id",
                        "innings",
                        "over",
                        "ball",
                        "balls_faced"
                    ]
                ].head(20)
            )

        self.df = self.df[self.df["balls_faced"] <= eff_balls_allowed]

        # --------------------------------------------------
        # Run rate
        # --------------------------------------------------

        self.df["run_rate"] = np.where(
            self.df["balls_faced"] > 0,
            self.df["runs_scored"] /
            (self.df["balls_faced"] / 6),
            0
        )

        # --------------------------------------------------
        # Partnership & Momentum features (Upgrade 2) - Optimized Vectorized Implementation
        # --------------------------------------------------
        prev_match = self.df["match_id"].shift(1)
        prev_innings = self.df["innings"].shift(1)
        prev_wicket = self.df["wicket"].shift(1)

        is_start = (
            (self.df["match_id"] != prev_match) | 
            (self.df["innings"] != prev_innings) | 
            (prev_wicket == 1)
        )
        partnership_id = is_start.cumsum()
        self.df["current_partnership_runs"] = self.df.groupby(partnership_id)["total_runs"].cumsum()

        self.df["recent_runs"] = (
            self.df.groupby(["match_id", "innings"])["total_runs"]
            .rolling(window=self.rolling_window, min_periods=1)
            .sum()
            .droplevel([0, 1])
        )

        self.df["recent_wickets"] = (
            self.df.groupby(["match_id", "innings"])["wicket"]
            .rolling(window=self.rolling_window, min_periods=1)
            .sum()
            .droplevel([0, 1])
        )

        # --------------------------------------------------
        # Bowler Spell Fatigue & Economy
        # --------------------------------------------------
        # runs_conceded_delivery = runs_off_bat on normal balls, total_runs on wides/noballs
        runs_conceded_delivery = np.where(
            (self.df["is_wide"] == 1) | (self.df["is_noball"] == 1),
            self.df["total_runs"],
            self.df["runs_off_bat"]
        )
        
        self.df["bowler_balls_bowled"] = (
            self.df.groupby(["match_id", "innings", "bowler"])["legal_delivery"]
            .cumsum()
        )
        
        self.df["_runs_conceded_temp"] = runs_conceded_delivery
        self.df["bowler_runs_conceded"] = (
            self.df.groupby(["match_id", "innings", "bowler"])["_runs_conceded_temp"]
            .cumsum()
        )
        self.df = self.df.drop(columns=["_runs_conceded_temp"])
        
        default_econ = 8.0 if self.total_overs <= 20 else 5.0
        self.df["bowler_overs_bowled"] = (self.df["bowler_balls_bowled"] / 6.0).astype(float)
        self.df["bowler_economy"] = np.where(
            self.df["bowler_balls_bowled"] > 0,
            self.df["bowler_runs_conceded"] / (self.df["bowler_balls_bowled"] / 6.0),
            default_econ
        ).astype(float)

        return self

    def add_match_context(self):
        # Identify matches with both innings 1 and 2
        valid_matches = self.df.groupby("match_id")["innings"].nunique() == 2
        valid_match_ids = valid_matches[valid_matches].index
        
        df_filtered = self.df[self.df["match_id"].isin(valid_match_ids)].copy()
        
        # Get final score for first and second innings per match
        first_innings_totals = df_filtered[df_filtered["innings"] == 1].groupby("match_id")["runs_scored"].max()
        second_innings_totals = df_filtered[df_filtered["innings"] == 2].groupby("match_id")["runs_scored"].max()
        
        # Map back to dataframe
        first_innings_map = df_filtered["match_id"].map(first_innings_totals)
        second_innings_map = df_filtered["match_id"].map(second_innings_totals)
        
        target_runs_val = df_filtered.get("target_runs", pd.Series(np.nan, index=df_filtered.index))
        df_filtered["target"] = np.where(
            df_filtered["innings"] == 2,
            target_runs_val.fillna(first_innings_map + 1),
            np.nan
        )
        
        df_filtered["final_first_innings_score"] = first_innings_map
        
        df_filtered["runs_remaining"] = np.where(
            df_filtered["innings"] == 2,
            df_filtered["target"] - df_filtered["runs_scored"],
            np.nan
        )
        
        target_overs_val = df_filtered.get("target_overs", pd.Series(np.nan, index=df_filtered.index))
        effective_overs = np.where(
            (df_filtered["innings"] == 2) & target_overs_val.notna(),
            target_overs_val,
            self.total_overs
        )
        eff_overs_floor = np.floor(effective_overs)
        eff_balls_allowed = eff_overs_floor * 6 + np.round((effective_overs - eff_overs_floor) * 10)
        df_filtered["balls_remaining"] = np.maximum(
            eff_balls_allowed - df_filtered["balls_faced"],
            0.0
        )
        
        # Avoid division by zero by using a safe denominator when balls_remaining <= 0
        safe_balls = np.where(df_filtered["balls_remaining"] > 0, df_filtered["balls_remaining"], 6.0)
        required_rr = df_filtered["runs_remaining"] / (safe_balls / 6.0)
        
        df_filtered["required_run_rate"] = np.where(
            df_filtered["innings"] == 2,
            np.where(df_filtered["balls_remaining"] > 0, required_rr, 0.0),
            np.nan
        )
        
        # Chasing team won if second_innings_score > first_innings_score
        chasing_team_won = second_innings_map > first_innings_map
        df_filtered["won_match"] = np.where(
            df_filtered["innings"] == 2,
            chasing_team_won.astype(float),
            np.nan
        )
        
        return df_filtered

    def build(self):

        self.add_over_info()
        self.add_phase()
        self.build_innings_state()

        res = self.add_match_context()

        # Merge Elo ratings if provided
        if self.elo_df is not None:
            res = res.merge(self.elo_df, on="match_id", how="left")
            # If team_1_elo is missing, fill with neutral 1500
            res["team_1_elo"] = res["team_1_elo"].fillna(1500.0)
            res["team_2_elo"] = res["team_2_elo"].fillna(1500.0)

            # Set batting/bowling Elos
            res["batting_team_elo"] = np.where(
                res["batting_team"] == res["team_1"],
                res["team_1_elo"],
                res["team_2_elo"]
            )
            res["bowling_team_elo"] = np.where(
                res["batting_team"] == res["team_1"],
                res["team_2_elo"],
                res["team_1_elo"]
            )
            res["elo_diff"] = res["batting_team_elo"] - res["bowling_team_elo"]

            # Drop temporary merge helper columns (postponed to end)
            pass
        else:
            res["batting_team_elo"] = 1500.0
            res["bowling_team_elo"] = 1500.0
            res["elo_diff"] = 0.0

        # Cast to float64
        res["batting_team_elo"] = res["batting_team_elo"].astype(float)
        res["bowling_team_elo"] = res["bowling_team_elo"].astype(float)
        res["elo_diff"] = res["elo_diff"].astype(float)

        # Merge Phase Elo ratings if provided
        if self.phase_elo_df is not None:
            res = res.merge(self.phase_elo_df, on="match_id", how="left")
            res["batting_team_phase_elo"] = 1500.0
            res["bowling_team_phase_elo"] = 1500.0

            for phase in ["powerplay", "middle", "death"]:
                phase_mask = res["phase"] == phase

                # When team_1 is batting:
                t1_batting = (res["batting_team"] == res["team_1"]) & phase_mask
                res.loc[t1_batting, "batting_team_phase_elo"] = res.loc[t1_batting, f"team_1_batting_elo_{phase}"].fillna(1500.0)
                res.loc[t1_batting, "bowling_team_phase_elo"] = res.loc[t1_batting, f"team_2_bowling_elo_{phase}"].fillna(1500.0)

                # When team_2 is batting:
                t2_batting = (res["batting_team"] == res["team_2"]) & phase_mask
                res.loc[t2_batting, "batting_team_phase_elo"] = res.loc[t2_batting, f"team_2_batting_elo_{phase}"].fillna(1500.0)
                res.loc[t2_batting, "bowling_team_phase_elo"] = res.loc[t2_batting, f"team_1_bowling_elo_{phase}"].fillna(1500.0)

            res["phase_elo_diff"] = res["batting_team_phase_elo"] - res["bowling_team_phase_elo"]

            # Drop temporary phase Elo columns
            phase_drop_cols = []
            for p in ["powerplay", "middle", "death"]:
                phase_drop_cols.extend([
                    f"team_1_batting_elo_{p}", f"team_1_bowling_elo_{p}",
                    f"team_2_batting_elo_{p}", f"team_2_bowling_elo_{p}"
                ])
            res = res.drop(columns=phase_drop_cols, errors="ignore")
        else:
            res["batting_team_phase_elo"] = res["batting_team_elo"]
            res["bowling_team_phase_elo"] = res["bowling_team_elo"]
            res["phase_elo_diff"] = res["elo_diff"]

        res["batting_team_phase_elo"] = res["batting_team_phase_elo"].astype(float)
        res["bowling_team_phase_elo"] = res["bowling_team_phase_elo"].astype(float)
        res["phase_elo_diff"] = res["phase_elo_diff"].astype(float)

        # Merge player Elo ratings if provided
        if self.player_elo_df is not None:
            # Batters
            bat_cols = ["match_id", "player", "effective_elo"]
            if "recent_form" in self.player_elo_df.columns:
                bat_cols.append("recent_form")
            
            bat_elo_df = self.player_elo_df[self.player_elo_df["role"] == "batter"][bat_cols].rename(
                columns={"player": "batter", "effective_elo": "batter_elo", "recent_form": "batter_recent_form"}
            )
            res = res.merge(bat_elo_df, on=["match_id", "batter"], how="left")
            res["batter_elo"] = res["batter_elo"].fillna(1500.0).astype(float)
            res["batter_recent_form"] = res.get("batter_recent_form", pd.Series(0.0, index=res.index)).fillna(0.0).astype(float)

            # Non-strikers (Elo only — recent_form was pruned during Upgrade 4 optimization)
            non_striker_cols = ["match_id", "player", "effective_elo"]
            non_striker_elo_df = self.player_elo_df[self.player_elo_df["role"] == "batter"][non_striker_cols].rename(
                columns={"player": "non_striker", "effective_elo": "non_striker_elo"}
            )
            res = res.merge(non_striker_elo_df, on=["match_id", "non_striker"], how="left")
            res["non_striker_elo"] = res["non_striker_elo"].fillna(1500.0).astype(float)

            # Bowlers
            bowl_cols = ["match_id", "player", "effective_elo"]
            if "recent_form" in self.player_elo_df.columns:
                bowl_cols.append("recent_form")
                
            bowl_elo_df = self.player_elo_df[self.player_elo_df["role"] == "bowler"][bowl_cols].rename(
                columns={"player": "bowler", "effective_elo": "bowler_elo", "recent_form": "bowler_recent_form"}
            )
            res = res.merge(bowl_elo_df, on=["match_id", "bowler"], how="left")
            res["bowler_elo"] = res["bowler_elo"].fillna(1500.0).astype(float)
            res["bowler_recent_form"] = res.get("bowler_recent_form", pd.Series(0.0, index=res.index)).fillna(0.0).astype(float)

            res["player_elo_diff"] = (res["batter_elo"] - res["bowler_elo"]).astype(float)
            res["player_recent_form_diff"] = (res["batter_recent_form"] - res["bowler_recent_form"]).astype(float)

            # Merge playing XI squad average Elos
            squad_cols = ["match_id", "team_1_batting_squad_elo", "team_1_bowling_squad_elo", 
                          "team_2_batting_squad_elo", "team_2_bowling_squad_elo"]
            if all(col in self.player_elo_df.columns for col in squad_cols):
                squad_df = self.player_elo_df[squad_cols].drop_duplicates()
                res = res.merge(squad_df, on="match_id", how="left")
                
                # Fallbacks in case of missing matches or values
                res["team_1_batting_squad_elo"] = res["team_1_batting_squad_elo"].fillna(res["team_1_elo"]).astype(float)
                res["team_1_bowling_squad_elo"] = res["team_1_bowling_squad_elo"].fillna(res["team_1_elo"]).astype(float)
                res["team_2_batting_squad_elo"] = res["team_2_batting_squad_elo"].fillna(res["team_2_elo"]).astype(float)
                res["team_2_bowling_squad_elo"] = res["team_2_bowling_squad_elo"].fillna(res["team_2_elo"]).astype(float)

                # Derive innings-specific squad ratings
                res["batting_squad_elo"] = np.where(res["innings"] == 1, res["team_1_batting_squad_elo"], res["team_2_batting_squad_elo"])
                res["bowling_squad_elo"] = np.where(res["innings"] == 1, res["team_2_bowling_squad_elo"], res["team_1_bowling_squad_elo"])
                res["squad_elo_diff"] = (res["batting_squad_elo"] - res["bowling_squad_elo"]).astype(float)
            else:
                res["batting_squad_elo"] = res["batting_team_elo"]
                res["bowling_squad_elo"] = res["bowling_team_elo"]
                res["squad_elo_diff"] = res["elo_diff"]
        else:
            res["batter_elo"] = 1500.0
            res["bowler_elo"] = 1500.0
            res["non_striker_elo"] = 1500.0
            res["player_elo_diff"] = 0.0
            res["batter_recent_form"] = 0.0
            res["bowler_recent_form"] = 0.0
            res["player_recent_form_diff"] = 0.0
            res["batting_squad_elo"] = res["batting_team_elo"]
            res["bowling_squad_elo"] = res["bowling_team_elo"]
            res["squad_elo_diff"] = res["elo_diff"]

        # Map player styles
        from src.features.build_player_registry import load_player_styles
        player_styles = load_player_styles()

        batter_cats = res["batter_id"].cat.categories if hasattr(res["batter_id"], "cat") else res["batter_id"].unique()
        batter_hand_map = {cat: player_styles.get(str(cat), {}).get("bat", "RHB") for cat in batter_cats}
        res["batter_hand"] = res["batter_id"].map(batter_hand_map).fillna("RHB").astype("category")

        bowler_cats = res["bowler_id"].cat.categories if hasattr(res["bowler_id"], "cat") else res["bowler_id"].unique()
        bowler_style_map = {cat: player_styles.get(str(cat), {}).get("bowl", "RAP") for cat in bowler_cats}
        res["bowler_style"] = res["bowler_id"].map(bowler_style_map).fillna("RAP").astype("category")

        # Spinner matchup definitions
        # is_spin_away: spin spins away from batter (bowler advantage)
        # - LHB vs OB or LAA
        # - RHB vs LB or SLA
        res["is_spin_away"] = np.where(
            ((res["batter_hand"] == "RHB") & res["bowler_style"].isin(["LB", "SLA"])) |
            ((res["batter_hand"] == "LHB") & res["bowler_style"].isin(["OB", "LAA"])),
            1.0,
            0.0
        ).astype(float)

        # is_spin_in: spin spins into batter (batter advantage)
        # - LHB vs LB or SLA
        # - RHB vs OB or LAA
        res["is_spin_in"] = np.where(
            ((res["batter_hand"] == "RHB") & res["bowler_style"].isin(["OB", "LAA"])) |
            ((res["batter_hand"] == "LHB") & res["bowler_style"].isin(["LB", "SLA"])),
            1.0,
            0.0
        ).astype(float)

        # Drop temporary merge helper columns
        res = res.drop(columns=["team_1", "team_2", "team_1_elo", "team_2_elo",
                                "team_1_batting_squad_elo", "team_1_bowling_squad_elo",
                                "team_2_batting_squad_elo", "team_2_bowling_squad_elo",
                                "batter_id", "bowler_id", "non_striker_id",
                                "bowler_balls_bowled", "bowler_runs_conceded"], errors="ignore")
        res["total_overs"] = float(self.total_overs)

        return res
    


    