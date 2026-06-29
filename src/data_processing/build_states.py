import pandas as pd
import numpy as np


class StateBuilder:

    def __init__(self, df: pd.DataFrame, elo_df: pd.DataFrame = None, total_overs=20):
        self.df = df.copy()
        self.elo_df = elo_df
        self.total_overs = total_overs

    def add_over_info(self):
        self.df["over_float"] = (
            self.df["over"] +
            (self.df["ball"] - 1) / 6
        )
        return self

    def add_phase(self):
        if self.total_overs > 20:
            powerplay_end = 10.0
            death_start = 40.0
        else:
            powerplay_end = self.total_overs * 0.3
            death_start = self.total_overs * 0.75

        def get_phase(over):
            if over < powerplay_end:
                return "powerplay"
            elif over < death_start:
                return "middle"
            return "death"

        self.df["phase"] = self.df["over_float"].apply(get_phase)
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

        bad = self.df[self.df["balls_faced"] > (self.total_overs * 6)]

        if len(bad) > 0:
            print(f"\nWARNING: {len(bad)} invalid states found\n")

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

        # --------------------------------------------------
        # Run rate
        # --------------------------------------------------

        self.df["run_rate"] = np.where(
            self.df["balls_faced"] > 0,
            self.df["runs_scored"] /
            (self.df["balls_faced"] / 6),
            0
        )

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
        
        df_filtered["target"] = np.where(
            df_filtered["innings"] == 2,
            first_innings_map + 1,
            np.nan
        )
        
        df_filtered["final_first_innings_score"] = first_innings_map
        
        df_filtered["runs_remaining"] = np.where(
            df_filtered["innings"] == 2,
            df_filtered["target"] - df_filtered["runs_scored"],
            np.nan
        )
        
        df_filtered["balls_remaining"] = (
            (self.total_overs * 6) - df_filtered["balls_faced"]
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

            # Drop temporary merge helper columns
            res = res.drop(columns=["team_1", "team_2", "team_1_elo", "team_2_elo"], errors="ignore")
        else:
            res["batting_team_elo"] = 1500.0
            res["bowling_team_elo"] = 1500.0
            res["elo_diff"] = 0.0

        # Cast to float64
        res["batting_team_elo"] = res["batting_team_elo"].astype(float)
        res["bowling_team_elo"] = res["bowling_team_elo"].astype(float)
        res["elo_diff"] = res["elo_diff"].astype(float)

        res["total_overs"] = float(self.total_overs)

        return res
    


    