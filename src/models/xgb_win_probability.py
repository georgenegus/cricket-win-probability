import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold


class XGBWinProbabilityModel:
    """
    Win probability model powered by a Gradient Boosted Decision Tree (XGBoost) classifier.
    Supports Group K-Fold cross-validation by match_id to prevent data leakage during validation.
    """

    def __init__(self, n_estimators=None, max_depth=None, learning_rate=None, 
                 min_child_weight=None, subsample=None, colsample_bytree=None,
                 reg_alpha=None, reg_lambda=None, random_state=42, format_type="t20",
                 baseline_rr=None, pressure_wicket_div=None, b_factor=None, wicket_exp=None,
                 rolling_window=None, elo_death_start=None, elo_powerplay_end=None):
        self.format_type = format_type
        self.random_state = random_state
        self.rolling_window = rolling_window if rolling_window is not None else 36
        
        # Set format-specific defaults
        if format_type == "od":
            self.n_estimators = n_estimators if n_estimators is not None else 339
            self.max_depth = max_depth if max_depth is not None else 4
            self.learning_rate = learning_rate if learning_rate is not None else 0.02
            self.min_child_weight = min_child_weight if min_child_weight is not None else 10
            self.subsample = subsample if subsample is not None else 0.7
            self.colsample_bytree = colsample_bytree if colsample_bytree is not None else 0.7
            self.reg_alpha = reg_alpha if reg_alpha is not None else 0.0
            self.reg_lambda = reg_lambda if reg_lambda is not None else 1.0
            
            self.baseline_rr = float(baseline_rr if baseline_rr is not None else 5.0)
            self.pressure_wicket_div = float(pressure_wicket_div if pressure_wicket_div is not None else 15.0)
            self.total_format_balls = 300.0
            self.b_factor = float(b_factor if b_factor is not None else 0.04)
            self.wicket_exp = float(wicket_exp if wicket_exp is not None else 1.2)
            self.elo_death_start = float(elo_death_start if elo_death_start is not None else 60.0)
            self.elo_powerplay_end = float(elo_powerplay_end if elo_powerplay_end is not None else 240.0)
        else:
            self.n_estimators = n_estimators if n_estimators is not None else 300
            self.max_depth = max_depth if max_depth is not None else 4
            self.learning_rate = learning_rate if learning_rate is not None else 0.08
            self.min_child_weight = min_child_weight if min_child_weight is not None else 10
            self.subsample = subsample if subsample is not None else 0.8
            self.colsample_bytree = colsample_bytree if colsample_bytree is not None else 0.8
            self.reg_alpha = reg_alpha if reg_alpha is not None else 1.0
            self.reg_lambda = reg_lambda if reg_lambda is not None else 3.0
            
            self.baseline_rr = float(baseline_rr if baseline_rr is not None else 8.0)
            self.pressure_wicket_div = float(pressure_wicket_div if pressure_wicket_div is not None else 10.0)
            self.total_format_balls = 120.0
            self.b_factor = float(b_factor if b_factor is not None else 0.09)
            self.wicket_exp = float(wicket_exp if wicket_exp is not None else 1.6)
            
        self.fold_models = {}  # key: fold_index, value: fitted XGBClassifier
        self.final_model = None  # trained on 100% of the data
        self.match_to_fold = {}  # key: match_id, value: fold_index

    def _compute_dls_resource_pct(self, balls_remaining, wickets_lost):
        """
        Approximation of DLS resource percentage based on remaining balls and wickets lost.
        """
        overs_remaining = balls_remaining / 6.0
        wickets_in_hand = 10.0 - wickets_lost
        
        # Non-linear wicket resource curve: first wickets cost less than last wickets
        wicket_factor = 1.0 - (1.0 - wickets_in_hand / 10.0) ** self.wicket_exp
        # Exponential decay curve for remaining overs
        overs_factor = 1.0 - np.exp(-self.b_factor * overs_remaining)
        
        return wicket_factor * overs_factor

    def _extract_features_and_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        features = pd.DataFrame(index=df.index)
        features["runs_remaining"] = df["runs_remaining"].astype(float)
        features["balls_remaining"] = df["balls_remaining"].astype(float)
        features["wickets_lost"] = df["wickets_lost"].astype(float)
        features["required_run_rate"] = df["required_run_rate"].astype(float)
        features["run_rate"] = df["run_rate"].astype(float)
        features["target"] = df["target"].astype(float)

        # pressure_index = required_run_rate * (1 + wickets_lost / 10.0)
        # Normalised required run rate (relative to format baseline)
        normalized_req_rr = features["required_run_rate"] / self.baseline_rr

        # Wicket pressure scaled by innings progress
        wicket_pressure = (
            features["wickets_lost"]
            * (self.total_format_balls / np.maximum(features["balls_remaining"], 1.0))
        )

        # Pressure index with configurable wicket weight
        features["pressure_index"] = normalized_req_rr * (
            1.0 + wicket_pressure / self.pressure_wicket_div
        )

        # Phase mapping: powerplay: 0, middle: 1, death: 2
        phase_map = {"powerplay": 0, "middle": 1, "death": 2}
        features["phase_encoded"] = df["phase"].map(phase_map).fillna(0).astype(int)

        # Elo features with phase-based decay:
        # Full strength in Powerplay, linear decay in Middle, 0 in Death (parameterized for total overs)
        total_overs = df.get("total_overs", pd.Series(20.0, index=df.index)).astype(float)
        total_balls = total_overs * 6.0
        
        is_one_day = total_overs > 20
        death_start_balls = np.where(is_one_day, self.elo_death_start if self.format_type == "od" else 60.0, total_balls * 0.25)
        powerplay_end_balls = np.where(is_one_day, self.elo_powerplay_end if self.format_type == "od" else 240.0, total_balls * 0.70)
        
        middle_duration = np.maximum(powerplay_end_balls - death_start_balls, 1.0)
        decay_weight = np.clip((features["balls_remaining"] - death_start_balls) / middle_duration, 0.0, 1.0)
        features["batting_team_elo"] = 1500.0 + (df["batting_team_elo"].astype(float) - 1500.0) * decay_weight
        features["bowling_team_elo"] = 1500.0 + (df["bowling_team_elo"].astype(float) - 1500.0) * decay_weight
        features["elo_diff"] = df["elo_diff"].astype(float) * decay_weight

        # Venue features with safe fallback defaults (in case they are not in the DataFrame)
        default_venue_score = 260.0 if self.format_type == "od" else 160.0
        features["venue_avg_first_innings_score"] = (
            df.get("venue_avg_first_innings_score", pd.Series(default_venue_score, index=df.index))
            .fillna(default_venue_score)
            .astype(float)
        )
        features["venue_chase_success_rate"] = (
            df.get("venue_chase_success_rate", pd.Series(0.45, index=df.index))
            .fillna(0.45)
            .astype(float)
        )

        # Derived features
        features["resources_remaining"] = (10.0 - features["wickets_lost"]) * features["balls_remaining"]
        if self.format_type == "od":
            features["dls_resource_pct"] = self._compute_dls_resource_pct(features["balls_remaining"], features["wickets_lost"])
            features["current_partnership_runs"] = df.get("current_partnership_runs", pd.Series(0.0, index=df.index)).astype(float).fillna(0.0)
            features["recent_runs"] = df.get("recent_runs", pd.Series(0.0, index=df.index)).astype(float).fillna(0.0)
            features["recent_wickets"] = df.get("recent_wickets", pd.Series(0.0, index=df.index)).astype(float).fillna(0.0)
            features["recent_run_rate"] = features["recent_runs"] / (self.rolling_window / 6.0)
        features["proportion_scored"] = (features["target"] - features["runs_remaining"]) / np.maximum(features["target"], 1.0)
        features["balls_per_run_needed"] = features["balls_remaining"] / np.maximum(features["runs_remaining"], 1.0)
        features["wickets_in_hand"] = 10.0 - features["wickets_lost"]
        features["scoring_acceleration"] = features["run_rate"] / np.maximum(features["required_run_rate"], 0.1)
        features["is_powerplay"] = (features["phase_encoded"] == 0).astype(float)
        features["is_death"] = (features["phase_encoded"] == 2).astype(float)

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

        target = df["won_match"].astype(int)
        return features, target

    def _extract_features_single(self, state: pd.Series) -> pd.DataFrame:
        features = {}
        features["runs_remaining"] = float(state["runs_remaining"])
        features["balls_remaining"] = float(state["balls_remaining"])
        features["wickets_lost"] = float(state["wickets_lost"])
        features["required_run_rate"] = float(state["required_run_rate"])
        features["run_rate"] = float(state["run_rate"])
        features["target"] = float(state["target"])

        # Calculate pressure_index
        normalized_req_rr = features["required_run_rate"] / self.baseline_rr
        wicket_pressure = (
            features["wickets_lost"]
            * (self.total_format_balls / max(features["balls_remaining"], 1.0))
        )
        features["pressure_index"] = normalized_req_rr * (
            1.0 + wicket_pressure / self.pressure_wicket_div
        )

        # Phase mapping
        phase_map = {"powerplay": 0, "middle": 1, "death": 2}
        phase_val = state.get("phase", "middle")
        features["phase_encoded"] = phase_map.get(phase_val, 1)

        # Elo features with phase-based decay:
        # Full strength in Powerplay, linear decay in Middle, 0 in Death (parameterized for total overs)
        total_overs = float(state.get("total_overs", 20.0))
        total_balls = total_overs * 6.0
        
        if total_overs > 20:
            death_start_balls = self.elo_death_start if self.format_type == "od" else 60.0
            powerplay_end_balls = self.elo_powerplay_end if self.format_type == "od" else 240.0
        else:
            death_start_balls = total_balls * 0.25
            powerplay_end_balls = total_balls * 0.70
            
        middle_duration = max(powerplay_end_balls - death_start_balls, 1.0)
        decay_weight = float(np.clip((features["balls_remaining"] - death_start_balls) / middle_duration, 0.0, 1.0))
        batting_elo = float(state.get("batting_team_elo", 1500.0))
        bowling_elo = float(state.get("bowling_team_elo", 1500.0))
        elo_diff = float(state.get("elo_diff", 0.0))

        features["batting_team_elo"] = 1500.0 + (batting_elo - 1500.0) * decay_weight
        features["bowling_team_elo"] = 1500.0 + (bowling_elo - 1500.0) * decay_weight
        features["elo_diff"] = elo_diff * decay_weight

        # Venue features with safe fallback defaults
        default_venue_score = 260.0 if self.format_type == "od" else 160.0
        features["venue_avg_first_innings_score"] = float(state.get("venue_avg_first_innings_score", default_venue_score))
        features["venue_chase_success_rate"] = float(state.get("venue_chase_success_rate", 0.45))

        # Derived features
        features["resources_remaining"] = (10.0 - features["wickets_lost"]) * features["balls_remaining"]
        if self.format_type == "od":
            features["dls_resource_pct"] = float(self._compute_dls_resource_pct(features["balls_remaining"], features["wickets_lost"]))
            features["current_partnership_runs"] = float(state.get("current_partnership_runs", 0.0))
            features["recent_runs"] = float(state.get("recent_runs", 0.0))
            features["recent_wickets"] = float(state.get("recent_wickets", 0.0))
            features["recent_run_rate"] = features["recent_runs"] / (self.rolling_window / 6.0)
        features["proportion_scored"] = (features["target"] - features["runs_remaining"]) / max(features["target"], 1.0)
        features["balls_per_run_needed"] = features["balls_remaining"] / max(features["runs_remaining"], 1.0)
        features["wickets_in_hand"] = 10.0 - features["wickets_lost"]
        features["scoring_acceleration"] = features["run_rate"] / max(features["required_run_rate"], 0.1)
        features["is_powerplay"] = float(features["phase_encoded"] == 0)
        features["is_death"] = float(features["phase_encoded"] == 2)

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

        # Return as a 2D DataFrame with 1 row
        return pd.DataFrame([features])


    def fit(self, states: pd.DataFrame, cache_path=None):
        """
        Fits Group K-Fold XGBoost classifiers grouped by match_id to prevent validation leakage,
        as well as a final model trained on all data.
        """
        if cache_path is not None:
            cache_file = Path(cache_path)
            if cache_file.exists():
                print(f"Loading XGBoost models from cache: {cache_file}")
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                    self.fold_models = cached_data["fold_models"]
                    self.final_model = cached_data["final_model"]
                    self.match_to_fold = cached_data["match_to_fold"]
                return self

        print("Extracting features for training...")
        X, y = self._extract_features_and_target(states)

        # Get unique match_ids for GroupKFold
        match_ids = states["match_id"].values
        unique_matches = states["match_id"].unique()

        print(f"Dataset has {len(states):,} states across {len(unique_matches):,} unique matches.")

        # Perform GroupKFold
        gkf = GroupKFold(n_splits=5)
        self.match_to_fold = {}

        # Map each match_id to its validation fold index
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=match_ids)):
            val_match_ids = states.iloc[val_idx]["match_id"].unique()
            for m_id in val_match_ids:
                self.match_to_fold[m_id] = fold_idx

            print(f"Training XGBoost model for Fold {fold_idx}...")
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]

            model = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_child_weight=self.min_child_weight,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            self.fold_models[fold_idx] = model

        print("Training final XGBoost model on complete dataset...")
        final_model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=-1
        )
        final_model.fit(X, y)
        self.final_model = final_model

        if cache_path is not None:
            cache_file = Path(cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving XGBoost models to cache: {cache_file}")
            with open(cache_file, "wb") as f:
                pickle.dump({
                    "fold_models": self.fold_models,
                    "final_model": self.final_model,
                    "match_to_fold": self.match_to_fold
                }, f)

        return self

    def predict(self, state: pd.Series, historical_states: pd.DataFrame = None):
        """
        Predicts win probability for a state. Bypasses model using logical boundaries
        for terminal states. Selects the appropriate fold model based on match_id to
        prevent validation leakage.
        """
        # 1. Logical Boundary Conditions
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

        # Extract features
        X_single = self._extract_features_single(state)

        # Decide which model to use to prevent data leakage during validation
        match_id = state.get("match_id", None)
        fold_idx = self.match_to_fold.get(match_id, None)

        if fold_idx is not None and fold_idx in self.fold_models:
            model = self.fold_models[fold_idx]
        else:
            model = self.final_model

        if model is None:
            return {
                "win_probability": None,
                "sample_size": 0,
                "status": "model_not_fitted"
            }

        # Predict probability
        probability = float(model.predict_proba(X_single)[0][1])

        return {
            "win_probability": round(probability, 4),
            "sample_size": len(self.match_to_fold) if self.match_to_fold else 0,
            "status": "success"
        }

