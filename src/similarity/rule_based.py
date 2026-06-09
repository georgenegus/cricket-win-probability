import pandas as pd


class RuleBasedSimilarity:

    def __init__(
        self,
        runs_tol=10,
        wickets_tol=1,
        balls_tol=6
    ):
        self.runs_tol = runs_tol
        self.wickets_tol = wickets_tol
        self.balls_tol = balls_tol

    def find_similar_states(
        self,
        state: pd.Series,
        historical_states: pd.DataFrame
    ) -> pd.DataFrame:

        mask = (
            (historical_states["runs_scored"]
                .between(
                    state["runs_scored"] - self.runs_tol,
                    state["runs_scored"] + self.runs_tol
                ))
            &
            (historical_states["wickets_lost"]
                .between(
                    state["wickets_lost"] - self.wickets_tol,
                    state["wickets_lost"] + self.wickets_tol
                ))
            &
            (historical_states["balls_remaining"]
                .between(
                    state["balls_remaining"] - self.balls_tol,
                    state["balls_remaining"] + self.balls_tol
                ))
        )

        return historical_states[mask]