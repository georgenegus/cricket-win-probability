import numpy as np
import pandas as pd


class SimpleSimilarityEngine:
    """
    Baseline similarity engine using weighted distance
    on core match-state features.
    """

    def __init__(self, k=100, phase_mode="filter", use_pressure=True, use_resources=False,
                 w_r=0.3, w_b=2.2, w_w=1.1, w_p=1.1):
        self.k = k
        self.phase_mode = phase_mode
        self.use_pressure = use_pressure
        self.use_resources = use_resources
        self.w_r = w_r
        self.w_b = w_b
        self.w_w = w_w
        self.w_p = w_p

    def find_similar_states(self, state: pd.Series, historical_states: pd.DataFrame):

        df = historical_states.copy()

        # -----------------------------
        # Ensure required columns exist
        # -----------------------------
        required = ["runs_remaining", "balls_remaining", "wickets_lost"]

        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # -----------------------------
        # Phase-aware filtering
        # -----------------------------
        if self.phase_mode == "filter" and "phase" in df.columns and "phase" in state:
            same_phase = df[df["phase"] == state["phase"]]
            if len(same_phase) >= self.k * 2:
                df = same_phase.copy()

        # -----------------------------
        # Feature extraction & standard deviations
        # -----------------------------
        sigma_rr = df["runs_remaining"].std()
        sigma_br = df["balls_remaining"].std()
        sigma_wk = df["wickets_lost"].std()

        # Fallback to defaults if std is zero or NaN
        sigma_rr = sigma_rr if (pd.notna(sigma_rr) and sigma_rr > 0) else 38.0
        sigma_br = sigma_br if (pd.notna(sigma_br) and sigma_br > 0) else 34.0
        sigma_wk = sigma_wk if (pd.notna(sigma_wk) and sigma_wk > 0) else 2.5

        rr = df["runs_remaining"] - state["runs_remaining"]
        br = df["balls_remaining"] - state["balls_remaining"]
        wk = df["wickets_lost"] - state["wickets_lost"]

        # -----------------------------
        # Normalised distance metric using variance-based scaling
        # -----------------------------
        distance = (
            self.w_r * (rr / sigma_rr) ** 2 +
            self.w_b * (br / sigma_br) ** 2 +
            self.w_w * (wk / sigma_wk) ** 2
        )

        # -----------------------------
        # Phase as a distance feature
        # -----------------------------
        if self.phase_mode == "feature" and "phase" in df.columns and "phase" in state:
            phase_map = {"powerplay": 0, "middle": 1, "death": 2}
            state_phase_val = phase_map.get(state["phase"], 0)
            # Map the pandas series phase column values
            df_phase_val = df["phase"].map(phase_map).fillna(0)
            phase_diff = (df_phase_val - state_phase_val).abs()
            # Phase diff penalty (normalised by 1.0)
            distance = distance + (phase_diff / 1.0) ** 2

        # -----------------------------
        # Feature Interactions (Upgrade 2E)
        # -----------------------------
        if self.use_pressure:
            if "required_run_rate" not in df.columns:
                raise ValueError("required_run_rate column is required for use_pressure")
            
            df_pressure = df["required_run_rate"] * (1 + df["wickets_lost"] / 10.0)
            state_pressure = state["required_run_rate"] * (1 + state["wickets_lost"] / 10.0)
            
            sigma_p = df_pressure.std()
            sigma_p = sigma_p if (pd.notna(sigma_p) and sigma_p > 0) else 5.0
            
            p_diff = df_pressure - state_pressure
            distance = distance + self.w_p * (p_diff / sigma_p) ** 2

        if self.use_resources:
            df_resources = (10 - df["wickets_lost"]) * df["balls_remaining"]
            state_resources = (10 - state["wickets_lost"]) * state["balls_remaining"]
            
            sigma_res = df_resources.std()
            sigma_res = sigma_res if (pd.notna(sigma_res) and sigma_res > 0) else 300.0
            
            res_diff = df_resources - state_resources
            distance = distance + (res_diff / sigma_res) ** 2

        df = df.assign(distance=distance)

        # -----------------------------
        # nearest neighbours
        # -----------------------------
        nearest = df.sort_values("distance").head(self.k)

        return nearest