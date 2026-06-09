import pandas as pd


class WinProbabilityModel:

    def __init__(
        self,
        similarity_engine,
        min_samples=30
    ):
        self.similarity_engine = similarity_engine
        self.min_samples = min_samples

    def predict(
        self,
        state: pd.Series,
        historical_states: pd.DataFrame
    ):

        similar_states = (
            self.similarity_engine
            .find_similar_states(
                state,
                historical_states
            )
        )

        sample_size = len(similar_states)

        if sample_size < self.min_samples:

            return {
                "win_probability": None,
                "sample_size": sample_size,
                "status": "insufficient_data"
            }

        probability = (
            similar_states["won_match"]
            .mean()
        )

        return {
            "win_probability": round(
                float(probability),
                4
            ),
            "sample_size": sample_size,
            "status": "success"
        }