import pandas as pd
import numpy as np


class StateBuilder:

    def __init__(self, df: pd.DataFrame, elo_df: pd.DataFrame = None):
        self.df = df.copy()
        self.elo_df = elo_df

    def add_over_info(self):
        self.df["over_float"] = (
            self.df["over"] +
            (self.df["ball"] - 1) / 6
        )
        return self

    def add_phase(self):
        def get_phase(over):
            if over < 6:
                return "powerplay"
            elif over < 15:
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

        bad = self.df[self.df["balls_faced"] > 120]

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

        all_matches = []

        for match_id, match_df in self.df.groupby("match_id"):

            innings_1 = match_df[
                match_df["innings"] == 1
            ]

            innings_2 = match_df[
                match_df["innings"] == 2
            ]

            if innings_1.empty or innings_2.empty:
                continue

            first_innings_score = (
                innings_1["runs_scored"].iloc[-1]
            )

            second_innings_score = (
                innings_2["runs_scored"].iloc[-1]
            )

            chasing_team_won = (
                second_innings_score >
                first_innings_score
            )

            match_df["target"] = np.where(
                match_df["innings"] == 2,
                first_innings_score + 1,
                np.nan
            )

            match_df["runs_remaining"] = np.where(
                match_df["innings"] == 2,
                match_df["target"] -
                match_df["runs_scored"],
                np.nan
            )

            match_df["balls_remaining"] = (
                120 - match_df["balls_faced"]
            )

            # Avoid division by zero by using a safe denominator when balls_remaining <= 0
            safe_balls = np.where(match_df["balls_remaining"] > 0, match_df["balls_remaining"], 6.0)
            required_rr = match_df["runs_remaining"] / (safe_balls / 6.0)

            match_df["required_run_rate"] = np.where(
                match_df["innings"] == 2,
                np.where(match_df["balls_remaining"] > 0, required_rr, 0.0),
                np.nan
            )

            match_df["won_match"] = np.where(
                match_df["innings"] == 2,
                int(chasing_team_won),
                np.nan
            )

            all_matches.append(match_df)

        return pd.concat(
            all_matches,
            ignore_index=True
        )

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

        return res
    


    