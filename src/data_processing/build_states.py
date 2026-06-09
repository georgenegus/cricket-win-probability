import pandas as pd
import numpy as np


class StateBuilder:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

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
            elif over < 16:
                return "middle"
            return "death"

        self.df["phase"] = self.df["over_float"].apply(get_phase)
        return self

    def build_innings_state(self):

        self.df = self.df.sort_values(
            ["match_id", "innings", "over", "ball"]
        )

        self.df["runs_scored"] = (
            self.df.groupby(
                ["match_id", "innings"]
            )["total_runs"]
            .cumsum()
        )

        self.df["wickets_lost"] = (
            self.df.groupby(
                ["match_id", "innings"]
            )["wicket"]
            .cumsum()
        )

        self.df["balls_faced"] = (
            self.df.groupby(
                ["match_id", "innings"]
            )
            .cumcount() + 1
        )

        self.df["run_rate"] = (
            self.df["runs_scored"] /
            (self.df["balls_faced"] / 6)
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

            match_df["required_run_rate"] = np.where(
                match_df["innings"] == 2,
                match_df["runs_remaining"] /
                (match_df["balls_remaining"] / 6),
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

        return self.add_match_context()