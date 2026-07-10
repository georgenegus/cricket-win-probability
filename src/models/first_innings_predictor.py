import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold


class FirstInningsPredictor:
    """
    Regression model using XGBRegressor to predict the final first innings score
    at any delivery during the first innings of a match.
    Uses Group K-Fold cross-validation by match_id to prevent validation leakage.
    """

    def __init__(self, n_estimators=300, max_depth=4, learning_rate=0.08,
                 subsample=0.8, colsample_bytree=0.8, random_state=42, format_type="t20"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.format_type = format_type
        self.fold_models = {}  # key: fold_index, value: fitted XGBRegressor
        self.final_model = None  # trained on 100% of the first innings data
        self.match_to_fold = {}  # key: match_id, value: fold_index

    def _extract_features_and_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        features = pd.DataFrame(index=df.index)
        features["over_float"] = df["over_float"].astype(float)
        features["runs_scored"] = df["runs_scored"].astype(float)
        features["wickets_lost"] = df["wickets_lost"].astype(float)
        features["run_rate"] = df["run_rate"].astype(float)
        
        total_overs = df.get("total_overs", pd.Series(20.0, index=df.index)).astype(float)
        total_balls = total_overs * 6.0
        
        features["balls_faced"] = df["balls_faced"].astype(float)
        features["balls_remaining"] = df["balls_remaining"].astype(float)

        # Derived features
        # 1. projected_score = (runs_scored / max(balls_faced, 1)) * total_balls
        fallback_runs = 225.0 if self.format_type == "od" else 160.0
        features["projected_score"] = np.where(
            features["balls_faced"] > 0,
            (features["runs_scored"] / features["balls_faced"]) * total_balls,
            fallback_runs
        )
        
        # 2. resources_remaining = (10 - wickets_lost) * balls_remaining
        features["resources_remaining"] = (10.0 - features["wickets_lost"]) * features["balls_remaining"]

        # Phase mapping: powerplay: 0, middle: 1, death: 2
        phase_map = {"powerplay": 0, "middle": 1, "death": 2}
        features["phase_encoded"] = df["phase"].map(phase_map).fillna(0).astype(int)

        # Elo features
        features["batting_team_elo"] = df["batting_team_elo"].astype(float)
        features["bowling_team_elo"] = df["bowling_team_elo"].astype(float)
        features["elo_diff"] = df["elo_diff"].astype(float)

        # Venue features with safe fallback defaults
        features["venue_avg_first_innings_score"] = (
            df.get("venue_avg_first_innings_score", pd.Series(160.0, index=df.index))
            .fillna(160.0)
            .astype(float)
        )

        if self.format_type == "od":
            comp_map = {
                "odc": 0, "odis": 1, "odms": 2, "rlc": 3, "sft": 4
            }
            default_val = 1.0 # default to odis
        else:
            comp_map = {
                "bbl": 0, "bpl": 1, "cpl": 2, "ctc": 3, "ilt": 4,
                "ipl": 5, "ipt": 6, "lpl": 7, "mct": 8, "mlc": 9,
                "msl": 10, "npl": 11, "ntb": 12, "psl": 13, "sat": 14,
                "ssm": 15, "t20s": 16
            }
            default_val = 16.0 # default to t20s
        features["competition_encoded"] = df["competition"].map(comp_map).fillna(default_val).astype(float)

        target = df["final_first_innings_score"].astype(float)
        return features, target

    def _extract_features_single(self, state: pd.Series) -> pd.DataFrame:
        features = {}
        features["over_float"] = float(state["over_float"])
        features["runs_scored"] = float(state["runs_scored"])
        features["wickets_lost"] = float(state["wickets_lost"])
        features["run_rate"] = float(state["run_rate"])
        
        total_overs = float(state.get("total_overs", 20.0))
        total_balls = total_overs * 6.0
        
        features["balls_faced"] = float(state["balls_faced"])
        features["balls_remaining"] = float(state["balls_remaining"])

        # Derived features
        if features["balls_faced"] > 0:
            features["projected_score"] = (features["runs_scored"] / features["balls_faced"]) * total_balls
        else:
            features["projected_score"] = 225.0 if total_overs > 20 else 160.0
            
        features["resources_remaining"] = (10.0 - features["wickets_lost"]) * features["balls_remaining"]

        # Phase mapping
        phase_map = {"powerplay": 0, "middle": 1, "death": 2}
        phase_val = state.get("phase", "middle")
        features["phase_encoded"] = phase_map.get(phase_val, 1)

        features["batting_team_elo"] = float(state.get("batting_team_elo", 1500.0))
        features["bowling_team_elo"] = float(state.get("bowling_team_elo", 1500.0))
        features["elo_diff"] = float(state.get("elo_diff", 0.0))

        features["venue_avg_first_innings_score"] = float(state.get("venue_avg_first_innings_score", 160.0))

        if self.format_type == "od":
            comp_map = {
                "odc": 0, "odis": 1, "odms": 2, "rlc": 3, "sft": 4
            }
            default_val = 1.0 # default to odis
        else:
            comp_map = {
                "bbl": 0, "bpl": 1, "cpl": 2, "ctc": 3, "ilt": 4,
                "ipl": 5, "ipt": 6, "lpl": 7, "mct": 8, "mlc": 9,
                "msl": 10, "npl": 11, "ntb": 12, "psl": 13, "sat": 14,
                "ssm": 15, "t20s": 16
            }
            default_val = 16.0 # default to t20s
        comp_val = state.get("competition", "t20s")
        features["competition_encoded"] = float(comp_map.get(comp_val, default_val))

        return pd.DataFrame([features])

    def fit(self, states: pd.DataFrame, cache_path=None):
        """
        Fits Group K-Fold XGBRegressor classifiers grouped by match_id,
        as well as a final model trained on all first innings data.
        """
        if cache_path is not None:
            cache_file = Path(cache_path)
            if cache_file.exists():
                print(f"Loading first innings regressor models from cache: {cache_file}")
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                    self.fold_models = cached_data["fold_models"]
                    self.final_model = cached_data["final_model"]
                    self.match_to_fold = cached_data["match_to_fold"]
                return self

        # Filter strictly for first innings states to train regression model
        first_innings_states = states[states["innings"] == 1].copy()
        print(f"Extracting regression features for {len(first_innings_states):,} first innings states...")
        X, y = self._extract_features_and_target(first_innings_states)

        match_ids = first_innings_states["match_id"].values
        unique_matches = first_innings_states["match_id"].unique()
        print(f"Training on {len(unique_matches):,} unique matches.")

        # GroupKFold (5 splits)
        gkf = GroupKFold(n_splits=5)
        self.match_to_fold = {}

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=match_ids)):
            val_match_ids = first_innings_states.iloc[val_idx]["match_id"].unique()
            for m_id in val_match_ids:
                self.match_to_fold[m_id] = fold_idx

            print(f"Training XGBRegressor for Fold {fold_idx}...")
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]

            model = XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            self.fold_models[fold_idx] = model

        print("Training final XGBRegressor on complete first innings dataset...")
        final_model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=-1
        )
        final_model.fit(X, y)
        self.final_model = final_model

        if cache_path is not None:
            cache_file = Path(cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving first innings regressor models to cache: {cache_file}")
            with open(cache_file, "wb") as f:
                pickle.dump({
                    "fold_models": self.fold_models,
                    "final_model": self.final_model,
                    "match_to_fold": self.match_to_fold
                }, f)

        return self

    def predict(self, state: pd.Series):
        """
        Predicts final first innings score for a state.
        Bypasses model for terminal states using logical boundary checks.
        """
        runs_scored = float(state["runs_scored"])
        wickets_lost = float(state["wickets_lost"])
        balls_remaining = float(state["balls_remaining"])

        # Logical Boundary Conditions: If team is all out or all overs bowled,
        # prediction is exactly the runs scored.
        if wickets_lost >= 10.0 or balls_remaining <= 0.0:
            return {
                "predicted_score": int(runs_scored),
                "status": "success"
            }

        # Extract features
        X_single = self._extract_features_single(state)

        # Select appropriate model based on fold to prevent leakage
        match_id = state.get("match_id", None)
        fold_idx = self.match_to_fold.get(match_id, None)

        if fold_idx is not None and fold_idx in self.fold_models:
            model = self.fold_models[fold_idx]
        else:
            model = self.final_model

        if model is None:
            # Fallback projection if model is not fitted
            total_overs = float(state.get("total_overs", 20.0))
            balls_faced = float(state["balls_faced"])
            projected = runs_scored if balls_faced == 0 else (runs_scored / balls_faced) * (total_overs * 6.0)
            return {
                "predicted_score": int(round(projected)),
                "status": "model_not_fitted"
            }

        # Predict score
        pred_val = float(model.predict(X_single)[0])
        
        # Ensure predicted score is at least the current runs_scored
        pred_val = max(pred_val, runs_scored)

        return {
            "predicted_score": int(round(pred_val)),
            "status": "success"
        }
