import pandas as pd
import numpy as np


class WinProbabilityModel:

    def __init__(
        self,
        similarity_engine,
        min_samples=30,
        calibration_model=None
    ):
        self.similarity_engine = similarity_engine
        self.min_samples = min_samples
        self.calibration_model = calibration_model

    def predict(
        self,
        state: pd.Series,
        historical_states: pd.DataFrame
    ):
        # --------------------------------------------------
        # Logical Boundary Conditions
        # --------------------------------------------------
        if state["runs_remaining"] <= 0:
            return {
                "win_probability": 1.0,
                "sample_size": 0,
                "status": "success"
            }

        if state["wickets_lost"] >= 10:
            return {
                "win_probability": 0.0,
                "sample_size": 0,
                "status": "success"
            }

        if state["balls_remaining"] <= 0:
            return {
                "win_probability": 0.0,
                "sample_size": 0,
                "status": "success"
            }

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

        # Distance weighting using a Gaussian kernel
        weights = np.exp(-similar_states["distance"])
        if weights.sum() == 0:
            probability = similar_states["won_match"].mean()
        else:
            probability = np.average(
                similar_states["won_match"],
                weights=weights
            )

        # Apply Platt calibration if model is fitted
        if self.calibration_model is not None:
            eps = 1e-4
            prob_clipped = np.clip(probability, eps, 1 - eps)
            logit = np.log(prob_clipped / (1 - prob_clipped))
            probability = float(self.calibration_model.predict_proba([[logit]])[0][1])

        return {
            "win_probability": round(
                float(probability),
                4
            ),
            "sample_size": sample_size,
            "status": "success"
        }

    def fit_calibration(self, calibration_states: pd.DataFrame, historical_states: pd.DataFrame, cache_path=None):
        """
        Fits a LogisticRegression calibration model on log-odds (logit) of raw probabilities.
        If cache_path is provided and exists, loads the model from cache instead of fitting.
        """
        import pickle
        from pathlib import Path
        from sklearn.linear_model import LogisticRegression

        if cache_path is not None:
            cache_file = Path(cache_path)
            if cache_file.exists():
                print(f"Loading calibration model from cache: {cache_file}")
                with open(cache_file, "rb") as f:
                    self.calibration_model = pickle.load(f)
                return

        print(f"Fitting Platt calibration on {len(calibration_states)} states...")
        
        # Save current calibration model to restore later or temporarily disable
        temp_cal = self.calibration_model
        self.calibration_model = None
        
        raw_probs = []
        actuals = []
        
        for idx, (_, row) in enumerate(calibration_states.iterrows()):
            if idx % 100 == 0:
                print(f"Predicting state {idx}/{len(calibration_states)}...")
                
            # Exclude current match to prevent leakage (using the engine's built-in mask)
            out = self.predict(row, historical_states)
            if out["status"] == "success":
                raw_probs.append(out["win_probability"])
                actuals.append(row["won_match"])
                
        # Platt scaling on Logit
        raw_probs = np.array(raw_probs)
        actuals = np.array(actuals)
        
        # Convert to logit space
        eps = 1e-4
        probs_clipped = np.clip(raw_probs, eps, 1 - eps)
        logits = np.log(probs_clipped / (1 - probs_clipped))
        
        lr = LogisticRegression()
        lr.fit(logits.reshape(-1, 1), actuals)
        self.calibration_model = lr
        
        if cache_path is not None:
            cache_file = Path(cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving calibration model to cache: {cache_file}")
            with open(cache_file, "wb") as f:
                pickle.dump(self.calibration_model, f)