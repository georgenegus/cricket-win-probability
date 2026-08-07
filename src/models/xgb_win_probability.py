import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold

from src.data_processing.format_spec import spec, POOL_SOURCE
from src.features.player_elo_engine import PlayerEloEngine


class XGBWinProbabilityModel:
    """
    Win probability model powered by a Gradient Boosted Decision Tree (XGBoost) classifier.
    Supports Group K-Fold cross-validation by match_id to prevent data leakage during validation.
    """

    # Monotonic-direction prior for XGBoost's monotone_constraints (Phase 3, hnd-gated): the
    # chasing team's win probability has an unambiguous cricket direction in these features.
    # Anything not listed defaults to 0 (no constraint) -- ambiguous/contextual features
    # (venue, phase flags, competition, style, target, balls_remaining) stay unconstrained.
    MONOTONE_POSITIVE_FEATURES = frozenset({
        "proportion_scored", "wickets_in_hand", "run_rate", "scoring_acceleration",
        "balls_per_run_needed", "resources_remaining", "batting_team_elo",
        "batting_team_phase_elo", "elo_diff", "phase_elo_diff", "batter_elo",
        "batting_squad_elo", "batting_duo_elo_avg",
    })
    MONOTONE_NEGATIVE_FEATURES = frozenset({
        "runs_remaining", "required_run_rate", "pressure_index", "wickets_lost",
        "bowling_team_elo", "bowling_team_phase_elo", "bowler_elo", "bowling_squad_elo",
    })

    def __init__(self, n_estimators=None, max_depth=None, learning_rate=None,
                 min_child_weight=None, subsample=None, colsample_bytree=None,
                 reg_alpha=None, reg_lambda=None, random_state=42, format_type="t20",
                 baseline_rr=None, pressure_wicket_div=None, b_factor=None, wicket_exp=None,
                 rolling_window=None, elo_death_start=None, elo_powerplay_end=None):
        self.format_type = format_type
        self.random_state = random_state
        self._spec = spec(format_type)
        self.balls_per_over = self._spec["balls_per_over"]
        # Spec-derived (byte-identical for t20/od/hnd/t20w: 20/50/20/20 overs -> 18/36/18/18):
        # a >20-over innings gets the wider 36-ball window, matching StateBuilder's own
        # rolling_window rule (build_states.py) so recent_run_rate stays unit-consistent.
        self.rolling_window = rolling_window if rolling_window is not None else (18 if self._spec["total_overs"] <= 20 else 36)

        # Set format-specific defaults. odw inherits od's hyperparameter/regularisation
        # regime (regularisation regime is structural, not a population fit -- see
        # womens_games_ingestion_plan.md odw phase); hndw inherits hnd's below (its sweep
        # ran at 162 matches, statistically the same regime as hndw's 155).
        if format_type in ("od", "odw"):
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
            self.b_factor = float(b_factor if b_factor is not None else 0.04)
            self.wicket_exp = float(wicket_exp if wicket_exp is not None else 1.2)
        elif format_type in ("hnd", "hndw"):
            # Phase 4 sweep winner (tests/hundred_winprob_sweep.py, data/processed/hnd_winprob_sweep_results.csv):
            # mean OOF Brier 0.169295 vs. 0.188889 for the pre-sweep defaults (max_depth=4, reg_alpha=1,
            # reg_lambda=3, subsample/colsample=0.8) -- shallower + more regularized generalizes much
            # better on ~15K states / 162 matches.
            self.n_estimators = n_estimators if n_estimators is not None else 300
            self.max_depth = max_depth if max_depth is not None else 2
            self.learning_rate = learning_rate if learning_rate is not None else 0.08
            self.min_child_weight = min_child_weight if min_child_weight is not None else 10
            self.subsample = subsample if subsample is not None else 0.7
            self.colsample_bytree = colsample_bytree if colsample_bytree is not None else 0.7
            self.reg_alpha = reg_alpha if reg_alpha is not None else 3.0
            self.reg_lambda = reg_lambda if reg_lambda is not None else 10.0
            
            self.baseline_rr = float(baseline_rr if baseline_rr is not None else 8.0)
            self.pressure_wicket_div = float(pressure_wicket_div if pressure_wicket_div is not None else 10.0)
            self.b_factor = float(b_factor if b_factor is not None else 0.09)
            self.wicket_exp = float(wicket_exp if wicket_exp is not None else 1.6)
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
            self.b_factor = float(b_factor if b_factor is not None else 0.09)
            self.wicket_exp = float(wicket_exp if wicket_exp is not None else 1.6)

        # Byte-identical for t20/od/hnd/t20w (120.0/300.0/100.0/120.0 reproduced exactly);
        # was hardcoded per-branch above, which silently gave any new format 120.0 regardless
        # of its real ball count (would have under-scaled pressure_index for odw by 2.5x).
        self.total_format_balls = float(self._spec["total_balls"])

        # Elo phase-decay thresholds, in balls-remaining terms, derived from format_spec so
        # t20/od reproduce their previous hardcoded values exactly while hnd gets the correct
        # 100-ball thresholds instead of the previous (120-ball-derived) ones.
        # Equivalence: t20 120-90=30, 120-36=84; od 300-240=60, 300-60=240; hnd 100-75=25, 100-25=75.
        default_death_start_rem = self._spec["total_balls"] - self._spec["death_start_ball"]
        default_powerplay_end_rem = self._spec["total_balls"] - self._spec["pp_end_ball"]
        self.elo_death_start = float(elo_death_start if elo_death_start is not None else default_death_start_rem)
        self.elo_powerplay_end = float(elo_powerplay_end if elo_powerplay_end is not None else default_powerplay_end_rem)

        self.fold_models = {}  # key: fold_index, value: fitted XGBClassifier
        self.final_model = None  # trained on 100% of the data
        self.match_to_fold = {}  # key: match_id, value: fold_index
        self.fold_calibrators = {}  # key: fold_index, value: fitted IsotonicRegression
        self.final_calibrator = None  # fitted IsotonicRegression on all OOF predictions

        # Load Empirical DLS lookup table if it exists
        self.empirical_dls = None
        dls_path = Path("data/processed") / f"empirical_dls_{self.format_type}.parquet"
        if dls_path.exists():
            try:
                dls_df = pd.read_parquet(dls_path)
                self.empirical_dls = dls_df.set_index(["balls_remaining", "wickets_lost"])["expected_runs_remaining"].to_dict()
            except Exception as e:
                print(f"Error loading empirical DLS parquet cache: {e}")

    def _elo_stretch(self) -> float:
        """The Phase 1 hold-center stretch factor for this instance's format (1.0 = no-op
        for t20w, and for t20/od/hnd before each format's own Phase 1 ships). Single source
        of truth: PlayerEloEngine.T20_ELO_STRETCH / OD_ELO_STRETCH / HND_ELO_STRETCH."""
        if self.format_type == "t20":
            return PlayerEloEngine.T20_ELO_STRETCH
        elif self.format_type == "od":
            return PlayerEloEngine.OD_ELO_STRETCH
        elif self.format_type == "hnd":
            return PlayerEloEngine.HND_ELO_STRETCH
        return 1.0

    def _team_to_player_scale(self, team_elo):
        """
        Maps an absolute team-Elo value (EloEngine's unstretched scale) onto player-Elo's
        scale (hold-center: 1500 fixed). residual_* features below subtract team Elo from
        player/squad Elo -- without this, the subtraction mixes player_elo_engine.py's
        format-specific rescale with an unstretched team Elo and stops being a clean
        per-column function, breaking each format's own Phase 1 "changes nothing" invariant.
        No-op for t20w (and for t20/od/hnd before their own Phase 1 ships). Accepts scalar or
        ndarray/Series.
        """
        return 1500.0 + (team_elo - 1500.0) * self._elo_stretch()

    def _lookup_empirical_runs_remaining_batch(self, balls_remaining_series: pd.Series, wickets_lost_series: pd.Series) -> pd.Series:
        if self.empirical_dls is None:
            return (10.0 - wickets_lost_series) * balls_remaining_series
        
        # Round to nearest integer for safe lookup
        keys = list(zip(balls_remaining_series.round().astype(int), wickets_lost_series.round().astype(int)))
        res = [self.empirical_dls.get(key, 0.0) for key in keys]
        return pd.Series(res, index=balls_remaining_series.index)

    def _lookup_empirical_runs_remaining_single(self, balls_remaining: float, wickets_lost: float) -> float:
        if self.empirical_dls is None:
            return (10.0 - wickets_lost) * balls_remaining
        
        key = (int(round(balls_remaining)), int(round(wickets_lost)))
        return self.empirical_dls.get(key, 0.0)

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
        target_overs = df.get("target_overs", pd.Series(np.nan, index=df.index))
        total_balls = np.where(
            (df["innings"] == 2) & target_overs.notna(),
            target_overs * self.balls_per_over,
            self.total_format_balls
        )
        wicket_pressure = (
            features["wickets_lost"]
            * (total_balls / np.maximum(features["balls_remaining"], 1.0))
        )

        # Pressure index with configurable wicket weight
        features["pressure_index"] = normalized_req_rr * (
            1.0 + wicket_pressure / self.pressure_wicket_div
        )

        # Phase mapping: powerplay: 0, middle: 1, death: 2
        phase_map = {"powerplay": 0, "middle": 1, "death": 2}
        features["phase_encoded"] = df["phase"].map(phase_map).fillna(0).astype(int)

        # Elo features with phase-based decay: full strength in Powerplay, linear decay in
        # Middle, 0 in Death. Thresholds are the format_spec-derived self.elo_death_start /
        # self.elo_powerplay_end scalars set in __init__ (balls-remaining terms).
        middle_duration = max(self.elo_powerplay_end - self.elo_death_start, 1.0)
        decay_weight = np.clip((features["balls_remaining"] - self.elo_death_start) / middle_duration, 0.0, 1.0)
        features["batting_team_elo"] = 1500.0 + (df["batting_team_elo"].astype(float) - 1500.0) * decay_weight
        features["bowling_team_elo"] = 1500.0 + (df["bowling_team_elo"].astype(float) - 1500.0) * decay_weight
        features["elo_diff"] = df["elo_diff"].astype(float) * decay_weight

        # Phase-Specific Elo features
        p_bat = df.get("batting_team_phase_elo", df["batting_team_elo"]).astype(float)
        p_bowl = df.get("bowling_team_phase_elo", df["bowling_team_elo"]).astype(float)
        p_diff = df.get("phase_elo_diff", df["elo_diff"]).astype(float)
        features["batting_team_phase_elo"] = 1500.0 + (p_bat - 1500.0) * decay_weight
        features["bowling_team_phase_elo"] = 1500.0 + (p_bowl - 1500.0) * decay_weight
        features["phase_elo_diff"] = p_diff * decay_weight

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
        features["resources_remaining"] = self._lookup_empirical_runs_remaining_batch(
            features["balls_remaining"],
            features["wickets_lost"]
        )
        if self.format_type in ("od", "odw"):
            # odw is also a 50-over format -- without this, odw would be missing the DLS-
            # resource feature od's model is built around.
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

        # Phase 5/6: is_hundred lets a pooled model (hnd<-t20, hndw<-t20w) distinguish pooled
        # rows (tagged 0.0 by the pooling caller, see fit()) from its own rows (always 1.0
        # here). Gated on POOL_SOURCE membership so t20/od/t20w/odw's feature schema -- and
        # their cached models -- never change. Column name kept as "is_hundred" (not
        # generalised to e.g. "is_pooled_target") because hndw genuinely IS a Hundred format
        # and renaming would change hnd's already-cached feature schema for no reason.
        if self.format_type in POOL_SOURCE:
            features["is_hundred"] = 1.0

        if self.format_type in ("od", "odw"):
            comp_map = {
                "odc": 0, "odis": 1, "odms": 2, "rlc": 3, "sft": 4,
                # odw-only competitions (womens_games_ingestion_plan.md odw phase, measured
                # 2026-08-06: odis 636 / rhf 189 / wod 104) -- "odis" is shared with od (its
                # migrated female matches keep the same comp_dir code); rhf/wod are odw-only.
                "rhf": 5, "wod": 6,
            }
            default_val = 1.0 # default to odis
        else:
            comp_map = {
                "bbl": 0, "bpl": 1, "cpl": 2, "ctc": 3, "ilt": 4,
                "ipl": 5, "ipt": 6, "lpl": 7, "mct": 8, "mlc": 9,
                "msl": 10, "npl": 11, "ntb": 12, "psl": 13, "sat": 14,
                "ssm": 15, "t20s": 16,
                # t20w-only competitions (womens_games_ingestion_plan.md Phase 1/3) -- without
                # these, every non-shared women's competition (WBBL, WPL, etc.) collapses into
                # the generic t20s default, losing real competition-specific signal.
                "wbb": 17, "wpl": 18, "cec": 19, "wsl": 20, "wtb": 21, "blz": 22, "wtc": 23,
            }
            default_val = 16.0 # default to t20s
        features["competition_encoded"] = df["competition"].map(comp_map).fillna(default_val).astype(float)

        # Player Elo features (t20 / od / hnd / t20w / odw / hndw — hnd warm-started from T20
        # ratings via seed_ratings, hndw from t20w). t20w added Phase 3
        # (womens_games_ingestion_plan.md): its 3,200-match corpus is ~20x hnd's (162), which
        # already safely carries this feature. odw (929) and hndw (155, hnd-sized) both clear
        # the same bar.
        if self.format_type in ["t20", "od", "hnd", "t20w", "odw", "hndw"]:
            p_bat = df.get("batter_elo", pd.Series(1500.0, index=df.index)).astype(float)
            p_bowl = df.get("bowler_elo", pd.Series(1500.0, index=df.index)).astype(float)
            p_diff = df.get("player_elo_diff", pd.Series(0.0, index=df.index)).astype(float)
            p_non_striker = df.get("non_striker_elo", pd.Series(1500.0, index=df.index)).astype(float)
            p_non_striker_diff = (p_non_striker - p_bowl).astype(float)
            p_duo_avg = ((p_bat + p_non_striker) / 2.0).astype(float)

            s_bat = df.get("batting_squad_elo", df["batting_team_elo"]).astype(float)
            s_bowl = df.get("bowling_squad_elo", df["bowling_team_elo"]).astype(float)
            s_diff = df.get("squad_elo_diff", df["elo_diff"]).astype(float)

            t_bat = df["batting_team_elo"].astype(float)
            t_bowl = df["bowling_team_elo"].astype(float)
            t_diff = df["elo_diff"].astype(float)

            f_bat = df.get("batter_recent_form", pd.Series(0.0, index=df.index)).astype(float)
            f_bowl = df.get("bowler_recent_form", pd.Series(0.0, index=df.index)).astype(float)
            f_diff = df.get("player_recent_form_diff", pd.Series(0.0, index=df.index)).astype(float)

            # Decay raw player features
            features["batter_elo"] = 1500.0 + (p_bat - 1500.0) * decay_weight
            features["bowler_elo"] = 1500.0 + (p_bowl - 1500.0) * decay_weight
            features["non_striker_elo"] = 1500.0 + (p_non_striker - 1500.0) * decay_weight
            features["player_elo_diff"] = p_diff * decay_weight
            features["non_striker_elo_diff"] = p_non_striker_diff * decay_weight
            features["batting_duo_elo_avg"] = 1500.0 + (p_duo_avg - 1500.0) * decay_weight

            features["batting_squad_elo"] = 1500.0 + (s_bat - 1500.0) * decay_weight
            features["bowling_squad_elo"] = 1500.0 + (s_bowl - 1500.0) * decay_weight
            features["squad_elo_diff"] = s_diff * decay_weight

            # Decay residual player features. Team Elo is mapped onto player scale first --
            # see _team_to_player_scale (no-op for hnd).
            t_bat_r = self._team_to_player_scale(t_bat)
            t_bowl_r = self._team_to_player_scale(t_bowl)
            t_diff_r = t_diff * self._elo_stretch()

            features["residual_batter_elo"] = (p_bat - t_bat_r) * decay_weight
            features["residual_bowler_elo"] = (p_bowl - t_bowl_r) * decay_weight
            features["residual_player_elo_diff"] = (p_diff - t_diff_r) * decay_weight
            features["residual_batting_squad_elo"] = (s_bat - t_bat_r) * decay_weight
            features["residual_bowling_squad_elo"] = (s_bowl - t_bowl_r) * decay_weight
            features["residual_squad_elo_diff"] = (s_diff - t_diff_r) * decay_weight

            # Decay form features
            features["batter_recent_form"] = f_bat * decay_weight
            features["bowler_recent_form"] = f_bowl * decay_weight
            features["player_recent_form_diff"] = f_diff * decay_weight

        # Style-Based Matchup features
        bat_type = pd.CategoricalDtype(categories=["LHB", "RHB"])
        bowl_type = pd.CategoricalDtype(categories=["OB", "LB", "SLA", "LAA", "LAP", "RAP"])
        features["batter_hand"] = df.get("batter_hand", pd.Series("RHB", index=df.index)).astype(bat_type)
        features["bowler_style"] = df.get("bowler_style", pd.Series("RAP", index=df.index)).astype(bowl_type)
        features["is_spin_away"] = df.get("is_spin_away", pd.Series(0.0, index=df.index)).astype(float)
        features["is_spin_in"] = df.get("is_spin_in", pd.Series(0.0, index=df.index)).astype(float)

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
        target_overs = state.get("target_overs")
        innings = state.get("innings", 2)
        if innings == 2 and pd.notna(target_overs) and target_overs > 0:
            total_balls = float(target_overs) * self.balls_per_over
        else:
            total_balls = self.total_format_balls
            
        wicket_pressure = (
            features["wickets_lost"]
            * (total_balls / max(features["balls_remaining"], 1.0))
        )
        features["pressure_index"] = normalized_req_rr * (
            1.0 + wicket_pressure / self.pressure_wicket_div
        )

        # Phase mapping
        phase_map = {"powerplay": 0, "middle": 1, "death": 2}
        phase_val = state.get("phase", "middle")
        features["phase_encoded"] = phase_map.get(phase_val, 1)

        # Elo features with phase-based decay: full strength in Powerplay, linear decay in
        # Middle, 0 in Death. Thresholds are the format_spec-derived self.elo_death_start /
        # self.elo_powerplay_end scalars set in __init__ (balls-remaining terms).
        middle_duration = max(self.elo_powerplay_end - self.elo_death_start, 1.0)
        decay_weight = float(np.clip((features["balls_remaining"] - self.elo_death_start) / middle_duration, 0.0, 1.0))
        batting_elo = float(state.get("batting_team_elo", 1500.0))
        bowling_elo = float(state.get("bowling_team_elo", 1500.0))
        elo_diff = float(state.get("elo_diff", 0.0))

        features["batting_team_elo"] = 1500.0 + (batting_elo - 1500.0) * decay_weight
        features["bowling_team_elo"] = 1500.0 + (bowling_elo - 1500.0) * decay_weight
        features["elo_diff"] = elo_diff * decay_weight

        # Phase-Specific Elo features
        batting_phase_elo = float(state.get("batting_team_phase_elo", batting_elo))
        bowling_phase_elo = float(state.get("bowling_team_phase_elo", bowling_elo))
        phase_elo_diff = float(state.get("phase_elo_diff", elo_diff))

        features["batting_team_phase_elo"] = 1500.0 + (batting_phase_elo - 1500.0) * decay_weight
        features["bowling_team_phase_elo"] = 1500.0 + (bowling_phase_elo - 1500.0) * decay_weight
        features["phase_elo_diff"] = phase_elo_diff * decay_weight

        # Venue features with safe fallback defaults
        default_venue_score = 260.0 if self.format_type == "od" else 160.0
        features["venue_avg_first_innings_score"] = float(state.get("venue_avg_first_innings_score", default_venue_score))
        features["venue_chase_success_rate"] = float(state.get("venue_chase_success_rate", 0.45))

        # Derived features
        features["resources_remaining"] = self._lookup_empirical_runs_remaining_single(
            features["balls_remaining"],
            features["wickets_lost"]
        )
        if self.format_type in ("od", "odw"):
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

        # See _extract_features_and_target -- POOL_SOURCE-gated, keeps t20/od/t20w/odw schema unchanged.
        if self.format_type in POOL_SOURCE:
            features["is_hundred"] = 1.0

        if self.format_type in ("od", "odw"):
            comp_map = {
                "odc": 0, "odis": 1, "odms": 2, "rlc": 3, "sft": 4,
                "rhf": 5, "wod": 6,
            }
            default_val = 1.0 # default to odis
        else:
            comp_map = {
                "bbl": 0, "bpl": 1, "cpl": 2, "ctc": 3, "ilt": 4,
                "ipl": 5, "ipt": 6, "lpl": 7, "mct": 8, "mlc": 9,
                "msl": 10, "npl": 11, "ntb": 12, "psl": 13, "sat": 14,
                "ssm": 15, "t20s": 16,
                "wbb": 17, "wpl": 18, "cec": 19, "wsl": 20, "wtb": 21, "blz": 22, "wtc": 23,
            }
            default_val = 16.0 # default to t20s
        comp_val = state.get("competition", "t20s")
        features["competition_encoded"] = float(comp_map.get(comp_val, default_val))

        # Player Elo features (t20 / od / hnd / t20w / odw / hndw — hnd warm-started from T20
        # ratings via seed_ratings, hndw from t20w)
        if self.format_type in ["t20", "od", "hnd", "t20w", "odw", "hndw"]:
            p_bat = float(state.get("batter_elo", 1500.0))
            p_bowl = float(state.get("bowler_elo", 1500.0))
            p_diff = float(state.get("player_elo_diff", 0.0))
            p_non_striker = float(state.get("non_striker_elo", 1500.0))
            p_non_striker_diff = p_non_striker - p_bowl
            p_duo_avg = (p_bat + p_non_striker) / 2.0

            s_bat = float(state.get("batting_squad_elo", batting_elo))
            s_bowl = float(state.get("bowling_squad_elo", bowling_elo))
            s_diff = float(state.get("squad_elo_diff", elo_diff))

            f_bat = float(state.get("batter_recent_form", 0.0))
            f_bowl = float(state.get("bowler_recent_form", 0.0))
            f_diff = float(state.get("player_recent_form_diff", 0.0))

            # Decay raw player features
            features["batter_elo"] = 1500.0 + (p_bat - 1500.0) * decay_weight
            features["bowler_elo"] = 1500.0 + (p_bowl - 1500.0) * decay_weight
            features["non_striker_elo"] = 1500.0 + (p_non_striker - 1500.0) * decay_weight
            features["player_elo_diff"] = p_diff * decay_weight
            features["non_striker_elo_diff"] = p_non_striker_diff * decay_weight
            features["batting_duo_elo_avg"] = 1500.0 + (p_duo_avg - 1500.0) * decay_weight
            features["batting_squad_elo"] = 1500.0 + (s_bat - 1500.0) * decay_weight
            features["bowling_squad_elo"] = 1500.0 + (s_bowl - 1500.0) * decay_weight
            features["squad_elo_diff"] = s_diff * decay_weight

            # Decay residual player features. Team Elo is mapped onto player scale first --
            # see _team_to_player_scale (no-op for hnd).
            batting_elo_r = self._team_to_player_scale(batting_elo)
            bowling_elo_r = self._team_to_player_scale(bowling_elo)
            elo_diff_r = elo_diff * self._elo_stretch()

            features["residual_batter_elo"] = (p_bat - batting_elo_r) * decay_weight
            features["residual_bowler_elo"] = (p_bowl - bowling_elo_r) * decay_weight
            features["residual_player_elo_diff"] = (p_diff - elo_diff_r) * decay_weight
            features["residual_batting_squad_elo"] = (s_bat - batting_elo_r) * decay_weight
            features["residual_bowling_squad_elo"] = (s_bowl - bowling_elo_r) * decay_weight
            features["residual_squad_elo_diff"] = (s_diff - elo_diff_r) * decay_weight

            # Decay form features
            features["batter_recent_form"] = f_bat * decay_weight
            features["bowler_recent_form"] = f_bowl * decay_weight
            features["player_recent_form_diff"] = f_diff * decay_weight

        # Style-Based Matchup features
        features["batter_hand"] = state.get("batter_hand", "RHB")
        features["bowler_style"] = state.get("bowler_style", "RAP")
        features["is_spin_away"] = float(state.get("is_spin_away", 0.0))
        features["is_spin_in"] = float(state.get("is_spin_in", 0.0))

        # Return as a 2D DataFrame with 1 row
        df_feat = pd.DataFrame([features])
        bat_type = pd.CategoricalDtype(categories=["LHB", "RHB"])
        bowl_type = pd.CategoricalDtype(categories=["OB", "LB", "SLA", "LAA", "LAP", "RAP"])
        df_feat["batter_hand"] = df_feat["batter_hand"].astype(bat_type)
        df_feat["bowler_style"] = df_feat["bowler_style"].astype(bowl_type)
        return df_feat

    def _build_monotone_constraints(self, columns):
        """
        Builds an order-safe monotone_constraints tuple from live column names (hnd-gated).
        Returns None for t20/od, which is identical to XGBoost's own default -- no behaviour change.
        """
        if self.format_type != "hnd":
            return None
        signs = tuple(
            1 if col in self.MONOTONE_POSITIVE_FEATURES
            else -1 if col in self.MONOTONE_NEGATIVE_FEATURES
            else 0
            for col in columns
        )
        assert len(signs) == len(columns)
        return signs

    def fit(self, states: pd.DataFrame, cache_path=None, train_final_model=True, pool_X=None, pool_y=None):
        """
        Fits Group K-Fold XGBoost classifiers grouped by match_id to prevent validation leakage,
        as well as a final model trained on all data.

        pool_X/pool_y (Phase 5/6; POOL_SOURCE-gated formats only, e.g. hnd<-t20, hndw<-t20w):
        pre-extracted feature/target rows from another format pooled into every fold's TRAINING set as extra signal. Pool rows are
        never held out and never scored -- fold validation, OOF calibration, and match_to_fold
        are all computed purely from `states`. When pooling, hyperparameters and monotone
        constraints switch to the validated pooled config (see hundred_winprob_pooling_experiment.py)
        instead of self's own tiny-corpus-tuned attributes. t20/od never pass pool_X, so their
        behaviour is unchanged (byte-identical) from before pooling existed.
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
                    self.fold_calibrators = cached_data.get("fold_calibrators", {})
                    self.final_calibrator = cached_data.get("final_calibrator", None)
                return self

        print("Extracting features for training...")
        X, y = self._extract_features_and_target(states)
        monotone_constraints = self._build_monotone_constraints(X.columns)

        is_pooled = pool_X is not None
        if is_pooled:
            # Byte-identical for hnd (t20 and t20w both fall through the same "else" branch
            # in __init__, so every field xgb_kwargs reads below is equal by value between
            # them) -- see POOL_SOURCE in format_spec.py. Direct index, not .get(fmt, "t20"),
            # so a format missing from POOL_SOURCE fails loudly instead of silently reusing
            # t20's men's-population-tuned hyperparameters.
            pool_hp = XGBWinProbabilityModel(format_type=POOL_SOURCE[self.format_type])
            xgb_kwargs = dict(
                n_estimators=pool_hp.n_estimators, max_depth=pool_hp.max_depth,
                learning_rate=pool_hp.learning_rate, min_child_weight=pool_hp.min_child_weight,
                subsample=pool_hp.subsample, colsample_bytree=pool_hp.colsample_bytree,
                reg_alpha=pool_hp.reg_alpha, reg_lambda=pool_hp.reg_lambda,
                monotone_constraints=None,
            )
        else:
            xgb_kwargs = dict(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.learning_rate, min_child_weight=self.min_child_weight,
                subsample=self.subsample, colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha, reg_lambda=self.reg_lambda,
                monotone_constraints=monotone_constraints,
            )

        # Get unique match_ids for GroupKFold
        match_ids = states["match_id"].values
        unique_matches = states["match_id"].unique()

        print(f"Dataset has {len(states):,} states across {len(unique_matches):,} unique matches.")
        if is_pooled:
            print(f"Pooling in {len(pool_X):,} additional never-held-out training rows.")

        # Perform GroupKFold
        gkf = GroupKFold(n_splits=5)
        self.match_to_fold = {}

        raw_oof_probs = [None] * 5
        oof_targets = [None] * 5

        # Map each match_id to its validation fold index
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=match_ids)):
            val_match_ids = states.iloc[val_idx]["match_id"].unique()
            for m_id in val_match_ids:
                self.match_to_fold[m_id] = fold_idx

            print(f"Training XGBoost model for Fold {fold_idx}...")
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            if is_pooled:
                X_train = pd.concat([X_train, pool_X], ignore_index=True)
                y_train = pd.concat([y_train, pool_y], ignore_index=True)

            model = XGBClassifier(
                **xgb_kwargs,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1,
                enable_categorical=True,
            )
            model.fit(X_train, y_train)
            self.fold_models[fold_idx] = model

            # Predict raw out-of-fold probabilities (val is always pure `states` rows --
            # pool rows are never held out/scored, since we aren't evaluating the pool format here)
            raw_oof_probs[fold_idx] = model.predict_proba(X_val)[:, 1]
            oof_targets[fold_idx] = y_val.values

        # Train out-of-fold isotonic calibrators to prevent data leakage
        from sklearn.isotonic import IsotonicRegression
        self.fold_calibrators = {}
        for fold_idx in range(5):
            print(f"Training Isotonic Calibrator for Fold {fold_idx}...")
            other_probs = np.concatenate([raw_oof_probs[j] for j in range(5) if j != fold_idx])
            other_targets = np.concatenate([oof_targets[j] for j in range(5) if j != fold_idx])
            
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(other_probs, other_targets)
            self.fold_calibrators[fold_idx] = calibrator

        # Train final calibrator on all OOF predictions
        print("Training final Isotonic Calibrator on complete out-of-fold dataset...")
        all_oof_probs = np.concatenate(raw_oof_probs)
        all_oof_targets = np.concatenate(oof_targets)
        
        final_calibrator = IsotonicRegression(out_of_bounds="clip")
        final_calibrator.fit(all_oof_probs, all_oof_targets)
        self.final_calibrator = final_calibrator

        if train_final_model:
            print("Training final XGBoost model on complete dataset...")
            X_final, y_final = X, y
            if is_pooled:
                X_final = pd.concat([X, pool_X], ignore_index=True)
                y_final = pd.concat([y, pool_y], ignore_index=True)
            final_model = XGBClassifier(
                **xgb_kwargs,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1,
                enable_categorical=True,
            )
            final_model.fit(X_final, y_final)
            self.final_model = final_model
        else:
            print("Skipping final model training (sweep mode).")

        if cache_path is not None:
            cache_file = Path(cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving XGBoost models to cache: {cache_file}")
            with open(cache_file, "wb") as f:
                pickle.dump({
                    "fold_models": self.fold_models,
                    "final_model": self.final_model,
                    "match_to_fold": self.match_to_fold,
                    "fold_calibrators": self.fold_calibrators,
                    "final_calibrator": self.final_calibrator
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

        # Apply isotonic calibration
        if fold_idx is not None and fold_idx in self.fold_calibrators:
            probability = float(self.fold_calibrators[fold_idx].predict([probability])[0])
        elif self.final_calibrator is not None:
            probability = float(self.final_calibrator.predict([probability])[0])

        return {
            "win_probability": round(probability, 4),
            "sample_size": len(self.match_to_fold) if self.match_to_fold else 0,
            "status": "success"
        }

    def predict_with_ensemble(self, state: pd.Series):
        """
        Predicts win probability for a state and calculates 5-fold ensemble variance
        and 95% Confidence Interval.
        """
        # 1. Logical Boundary Conditions
        if state["runs_remaining"] <= 0:
            return {
                "win_probability": 1.0,
                "ci_lower": 1.0,
                "ci_upper": 1.0,
                "std_dev": 0.0,
                "fold_probabilities": [1.0] * 5,
                "status": "success"
            }

        if state["wickets_lost"] >= 10:
            return {
                "win_probability": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "std_dev": 0.0,
                "fold_probabilities": [0.0] * 5,
                "status": "success"
            }

        if state["balls_remaining"] <= 0:
            return {
                "win_probability": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "std_dev": 0.0,
                "fold_probabilities": [0.0] * 5,
                "status": "success"
            }

        # Extract features
        X_single = self._extract_features_single(state)

        # Collect predictions from all available fold models
        fold_probs = []
        for fold_idx in range(5):
            if fold_idx in self.fold_models:
                m = self.fold_models[fold_idx]
                p = float(m.predict_proba(X_single)[0][1])
                if fold_idx in self.fold_calibrators:
                    p = float(self.fold_calibrators[fold_idx].predict([p])[0])
                fold_probs.append(round(p, 4))

        # Main point estimate using final model + final calibrator
        if self.final_model is not None:
            point_prob = float(self.final_model.predict_proba(X_single)[0][1])
            if self.final_calibrator is not None:
                point_prob = float(self.final_calibrator.predict([point_prob])[0])
        elif fold_probs:
            point_prob = float(np.mean(fold_probs))
        else:
            return {
                "win_probability": None,
                "ci_lower": None,
                "ci_upper": None,
                "std_dev": None,
                "fold_probabilities": [],
                "status": "model_not_fitted"
            }

        if fold_probs:
            std_dev = float(np.std(fold_probs))
            ci_lower = max(0.0, round(point_prob - 1.96 * std_dev, 4))
            ci_upper = min(1.0, round(point_prob + 1.96 * std_dev, 4))
        else:
            std_dev = 0.0
            ci_lower = round(point_prob, 4)
            ci_upper = round(point_prob, 4)

        return {
            "win_probability": round(point_prob, 4),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "std_dev": round(std_dev, 4),
            "fold_probabilities": fold_probs,
            "status": "success"
        }

    FEATURE_LABELS = {
        "proportion_scored": "Target Chase Progress",
        "balls_per_run_needed": "Delivery Buffer Ratio",
        "required_run_rate": "Required Run Rate",
        "run_rate": "Current Run Rate",
        "scoring_acceleration": "Scoring Acceleration Rate",
        "pressure_index": "Match Pressure Index",
        "elo_diff": "Team Elo Advantage",
        "batting_team_elo": "Batting Team Rating",
        "bowling_team_elo": "Bowling Team Rating",
        "player_elo_diff": "Player Elo Advantage",
        "batter_elo": "Batter Elo Rating",
        "bowler_elo": "Bowler Elo Rating",
        "wickets_in_hand": "Wickets Remaining",
        "wickets_lost": "Wickets Fallen",
        "runs_remaining": "Runs Needed to Win",
        "balls_remaining": "Deliveries Remaining",
        "target": "Target Score",
        "venue_avg_first_innings_score": "Venue Par Score",
        "venue_chase_success_rate": "Venue Historical Chase Rate",
        "dls_resource_pct": "DLS Resource Percentage",
        "recent_run_rate": "Recent Form Run Rate",
        "current_partnership_runs": "Partnership Score",
    }

    def get_feature_contributions(self, state: pd.Series, top_n: int = 5):
        """
        Calculates exact TreeSHAP feature contributions for a single prediction state.
        Returns top N factors with directional impact, SHAP value, and narrative explanation.
        """
        import xgboost as xgb

        # Handle boundary conditions
        runs_rem = float(state.get("runs_remaining", 1.0))
        w_lost = float(state.get("wickets_lost", 0.0))
        b_rem = float(state.get("balls_remaining", 1.0))

        if runs_rem <= 0 or w_lost >= 10 or b_rem <= 0:
            return []

        if self.final_model is None:
            return []

        try:
            X_single = self._extract_features_single(state)
            booster = self.final_model.get_booster()
            dmatrix = xgb.DMatrix(X_single, enable_categorical=True)
            contribs = booster.predict(dmatrix, pred_contribs=True)[0]

            feature_names = list(X_single.columns)
            feature_contribs = contribs[:-1]

            results = []
            for fname, val in zip(feature_names, feature_contribs):
                if abs(val) < 1e-4:
                    continue
                
                label = self.FEATURE_LABELS.get(fname, fname.replace("_", " ").title())
                impact = "positive" if val > 0 else "negative"
                sign_str = f"+{val:.2f}" if val > 0 else f"{val:.2f}"
                
                if val > 0:
                    narrative = f"{label} pushes win probability UP ({sign_str})"
                else:
                    narrative = f"{label} pulls win probability DOWN ({sign_str})"

                results.append({
                    "feature": label,
                    "raw_feature": fname,
                    "contribution": round(float(val), 4),
                    "abs_contribution": round(abs(float(val)), 4),
                    "impact": impact,
                    "narrative": narrative
                })

            results = sorted(results, key=lambda x: x["abs_contribution"], reverse=True)
            return results[:top_n]
        except Exception as e:
            print(f"Error computing feature contributions: {e}")
            return []

    def predict_batch(self, states_df: pd.DataFrame) -> np.ndarray:
        """
        Vectorised batch prediction for multiple states at once.
        Handles boundary conditions, fold model selection, and feature extraction
        in bulk — much faster than calling predict() in a loop.

        Returns a numpy array of win probabilities, one per row.
        """
        n = len(states_df)
        probs = np.full(n, np.nan)

        # 1. Handle boundary conditions vectorially
        won = states_df["runs_remaining"].values <= 0
        all_out = states_df["wickets_lost"].values >= 10
        no_balls = states_df["balls_remaining"].values <= 0

        probs[won] = 1.0
        probs[all_out & ~won] = 0.0
        probs[no_balls & ~won & ~all_out] = 0.0

        # 2. Identify rows needing model prediction
        needs_model = np.isnan(probs)
        if not needs_model.any():
            return np.round(probs, 4)

        model_df = states_df[needs_model].copy()
        model_indices = np.where(needs_model)[0]

        # 3. Extract features in one vectorised call
        features, _ = self._extract_features_and_target(model_df)

        # 4. Group by fold and predict per fold model
        match_ids = model_df["match_id"].values
        fold_assignments = np.array([
            self.match_to_fold.get(mid, -1) for mid in match_ids
        ])

        for fold_idx in range(5):
            mask = fold_assignments == fold_idx
            if mask.any() and fold_idx in self.fold_models:
                fold_proba = self.fold_models[fold_idx].predict_proba(
                    features[mask]
                )[:, 1]
                
                # Apply fold calibrator if available
                if fold_idx in self.fold_calibrators:
                    fold_proba = self.fold_calibrators[fold_idx].predict(fold_proba)
                    
                probs[model_indices[mask]] = fold_proba

        # 5. Handle unseen matches (use final model)
        unseen_mask = fold_assignments == -1
        if unseen_mask.any() and self.final_model is not None:
            final_proba = self.final_model.predict_proba(
                features[unseen_mask]
            )[:, 1]
            
            # Apply final calibrator if available
            if self.final_calibrator is not None:
                final_proba = self.final_calibrator.predict(final_proba)
                
            probs[model_indices[unseen_mask]] = final_proba

        return np.round(probs, 4)
