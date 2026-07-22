import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold


# Legal-ball outcome classes, in the fixed index order the model emits probabilities for.
LEGAL_CLASSES = [0, 1, 2, 3, 4, 6, "W"]
LEGAL_CLASS_TO_IDX = {str(c): i for i, c in enumerate(LEGAL_CLASSES)}

# Extras-gate classes.
EXTRAS_CLASSES = ["legal", "wide", "noball"]
EXTRAS_CLASS_TO_IDX = {c: i for i, c in enumerate(EXTRAS_CLASSES)}

FEATURES = [
    "over_float", "balls_remaining", "phase_encoded", "innings",
    "wickets_lost", "batter_elo", "bowler_elo", "matchup_elo_diff",
    "venue_avg_first_innings_score", "recent_form_batter", "recent_form_bowler",
    "current_team_score", "striker_balls_faced", "required_run_rate",
    "non_striker_elo",
]


def _join_player_elo(df: pd.DataFrame, player_elo: pd.DataFrame, who: str) -> pd.DataFrame:
    """
    Joins per-match, per-role effective_elo/recent_form onto the deliveries frame.
    who: "batter" or "bowler" — matches both the delivery column name and the player_elo role.
    """
    sub = (
        player_elo[player_elo["role"] == who][["match_id", "player", "effective_elo", "recent_form"]]
        .drop_duplicates(subset=["match_id", "player"])
        .rename(columns={
            "player": who,
            "effective_elo": f"{who}_elo",
            "recent_form": f"recent_form_{who}",
        })
    )
    df = df.merge(sub, on=["match_id", who], how="left")
    df[f"{who}_elo"] = df[f"{who}_elo"].fillna(1500.0)
    df[f"recent_form_{who}"] = df[f"recent_form_{who}"].fillna(0.0)
    return df


def build_training_frame(deliveries: pd.DataFrame, player_elo: pd.DataFrame,
                         venue_features: pd.DataFrame, format_type: str = "t20") -> pd.DataFrame:
    """
    One row per historical delivery with the state features the simulator can reproduce
    at sampling time, plus the two training labels (extra_class, legal_outcome).

    Restricted to innings 1-2 — super overs (innings 3+) are a different scoring regime
    and are not modelled by the Phase 2/3 two-innings simulator.
    """
    total_overs = 50.0 if format_type == "od" else 20.0

    df = deliveries[deliveries["innings"].isin([1, 2])].copy()

    # Collapse rare overthrow noise (runs_off_bat of 5/7 are scorer artefacts, <0.02% of balls).
    df["runs_off_bat"] = df["runs_off_bat"].replace({5: 4, 7: 6})

    # Labels
    df["legal_outcome"] = np.where(df["wicket"] == 1, "W", df["runs_off_bat"].astype(int).astype(str))
    df["extra_class"] = np.select(
        [df["is_wide"] == 1, df["is_noball"] == 1],
        ["wide", "noball"], default="legal"
    )

    # State features (must be exactly reproducible inside the simulator's live loop)
    df["over_float"] = df["over"].astype(float) + (df["ball"].astype(float) - 1.0) / 6.0
    df["balls_remaining"] = ((df["target_overs"].fillna(total_overs) * 6.0)
                             - (df["over"].astype(float) * 6.0 + df["ball"].astype(float))).clip(lower=0.0)
    df["phase_encoded"] = pd.cut(df["over"], bins=[-1, 5, 14, 999], labels=[0, 1, 2]).astype(int)
    df["innings"] = df["innings"].astype(int)

    # Wickets lost BEFORE this ball (cumulative within match+innings, exclusive of the current ball)
    df["wickets_lost"] = (
        df.groupby(["match_id", "innings"])["wicket"].cumsum() - df["wicket"]
    ).astype(float)

    # Current team score BEFORE this ball
    df["current_team_score"] = (
        df.groupby(["match_id", "innings"])["total_runs"].cumsum() - df["total_runs"]
    ).astype(float)
    
    # Striker balls faced BEFORE this ball
    df["striker_balls_faced"] = (
        df.groupby(["match_id", "innings", "batter"]).cumcount()
    ).astype(float)
    
    # Required run rate (only applies to 2nd innings)
    runs_required = np.where(df["innings"] == 2, df["target_runs"].fillna(999.0) - df["current_team_score"], 0.0)
    df["required_run_rate"] = np.where(
        df["innings"] == 2, 
        (np.clip(runs_required, 0.0, None) / (df["balls_remaining"].replace(0, 1) / 6.0)).clip(upper=36.0),
        0.0
    )

    # Batter / bowler matchup Elo
    df = _join_player_elo(df, player_elo, who="batter")
    df = _join_player_elo(df, player_elo, who="bowler")

    # Non-striker batting Elo (partnership pressure)
    ns = (
        player_elo[player_elo["role"] == "batter"][["match_id", "player", "effective_elo"]]
        .drop_duplicates(subset=["match_id", "player"])
        .rename(columns={"player": "non_striker", "effective_elo": "non_striker_elo"})
    )
    df = df.merge(ns, on=["match_id", "non_striker"], how="left")
    df["non_striker_elo"] = df["non_striker_elo"].fillna(1500.0)

    # Recompute matchup difference using raw Elos
    df["matchup_elo_diff"] = df["batter_elo"] - df["bowler_elo"]

    # Venue scoring bias
    default_venue = 260.0 if format_type == "od" else 160.0
    vf = venue_features[["match_id", "venue_avg_first_innings_score"]]
    df = df.merge(vf, on="match_id", how="left")
    df["venue_avg_first_innings_score"] = df["venue_avg_first_innings_score"].fillna(default_venue)

    return df


class DeliveryOutcomeModel:
    """
    Two-stage multinomial delivery-outcome sampler:
      Stage A (extras gate)  — P(legal) vs P(wide) vs P(noball), all deliveries.
      Stage B (legal ball)   — P(0/1/2/3/4/6/WICKET), legal deliveries only.

    Follows the same GroupKFold-by-match_id + final_model ensemble discipline as
    XGBWinProbabilityModel / FirstInningsPredictor, to keep validation leakage-free
    and give the simulator a leakage-safe fold model for historical replay/calibration.
    """

    def __init__(self, format_type="t20", n_splits=5, random_state=42):
        self.format_type = format_type
        self.total_overs = 50.0 if format_type == "od" else 20.0
        self.n_splits = n_splits
        self.random_state = random_state

        self.fold_models = {}                          # fold_idx -> {"extras": clf, "legal": clf}
        self.final_model = {"extras": None, "legal": None}  # trained on 100% of the data
        self.match_to_fold = {}                         # match_id -> fold_idx

    @property
    def extras_model(self):
        return self.final_model["extras"]

    @property
    def legal_model(self):
        return self.final_model["legal"]

    def _make_extras_clf(self):
        return XGBClassifier(
            objective="multi:softprob", num_class=len(EXTRAS_CLASSES),
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            random_state=self.random_state, n_jobs=-1,
        )

    def _make_legal_clf(self):
        return XGBClassifier(
            objective="multi:softprob", num_class=len(LEGAL_CLASSES),
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            random_state=self.random_state, n_jobs=-1,
        )

    def fit(self, frame: pd.DataFrame, cache_path=None):
        """
        Fits Group K-Fold extras/legal classifier pairs grouped by match_id (to prevent
        validation leakage), plus a final pair trained on all data for production sampling.
        """
        if cache_path is not None:
            cache_file = Path(cache_path)
            if cache_file.exists():
                print(f"Loading delivery outcome models from cache: {cache_file}")
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                    self.fold_models = cached["fold_models"]
                    self.final_model = cached["final_model"]
                    self.match_to_fold = cached["match_to_fold"]
                return self

        y_extras_full = frame["extra_class"].map(EXTRAS_CLASS_TO_IDX)
        match_ids = frame["match_id"].values
        unique_matches = frame["match_id"].unique()
        print(f"Training on {len(frame):,} deliveries across {len(unique_matches):,} unique matches.")

        gkf = GroupKFold(n_splits=self.n_splits)
        self.match_to_fold = {}

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(frame, y_extras_full, groups=match_ids)):
            val_match_ids = frame.iloc[val_idx]["match_id"].unique()
            for m_id in val_match_ids:
                self.match_to_fold[m_id] = fold_idx

            train_frame = frame.iloc[train_idx]

            print(f"Training extras-gate model for Fold {fold_idx}...")
            extras_clf = self._make_extras_clf()
            extras_clf.fit(train_frame[FEATURES], train_frame["extra_class"].map(EXTRAS_CLASS_TO_IDX))

            print(f"Training legal-outcome model for Fold {fold_idx}...")
            legal_train = train_frame[train_frame["extra_class"] == "legal"]
            legal_clf = self._make_legal_clf()
            legal_clf.fit(legal_train[FEATURES], legal_train["legal_outcome"].map(LEGAL_CLASS_TO_IDX))

            self.fold_models[fold_idx] = {"extras": extras_clf, "legal": legal_clf}

        print("Training final extras-gate model on complete dataset...")
        final_extras = self._make_extras_clf()
        final_extras.fit(frame[FEATURES], y_extras_full)

        print("Training final legal-outcome model on complete dataset...")
        legal_all = frame[frame["extra_class"] == "legal"]
        final_legal = self._make_legal_clf()
        final_legal.fit(legal_all[FEATURES], legal_all["legal_outcome"].map(LEGAL_CLASS_TO_IDX))

        self.final_model = {"extras": final_extras, "legal": final_legal}

        if cache_path is not None:
            cache_file = Path(cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving delivery outcome models to cache: {cache_file}")
            with open(cache_file, "wb") as f:
                pickle.dump({
                    "fold_models": self.fold_models,
                    "final_model": self.final_model,
                    "match_to_fold": self.match_to_fold,
                }, f)

        return self

    def predict_proba_batch(self, state_matrix: pd.DataFrame, match_id=None) -> dict:
        """
        state_matrix: one row per live simulation (N rows), columns = FEATURES.
        match_id: if given and it maps to a held-out fold (leakage-safe historical replay,
                  e.g. Phase 5 calibration), that fold's models are used instead of the
                  final model. Live/future-match simulation (Phase 2-3) has no match_to_fold
                  entry and correctly falls through to the final model.

        Returns {'extras': (N,3) probs, 'legal': (N,7) probs} for vectorised sampling.
        """
        fold_idx = self.match_to_fold.get(match_id) if match_id is not None else None
        if fold_idx is not None and fold_idx in self.fold_models:
            extras_m = self.fold_models[fold_idx]["extras"]
            legal_m = self.fold_models[fold_idx]["legal"]
        else:
            extras_m = self.final_model["extras"]
            legal_m = self.final_model["legal"]

        X = state_matrix[FEATURES].values
        return {
            "extras": extras_m.predict_proba(X),
            "legal": legal_m.predict_proba(X),
        }
