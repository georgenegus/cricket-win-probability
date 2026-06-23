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

        # -----------------------------
        # Ensure required columns exist
        # -----------------------------
        required = ["runs_remaining", "balls_remaining", "wickets_lost"]

        for col in required:
            if col not in historical_states.columns:
                raise ValueError(f"Missing required column: {col}")

        # -----------------------------
        # Phase-aware filtering (with mask caching)
        # -----------------------------
        has_phase_filter = False
        mask = None
        if self.phase_mode == "filter" and "phase" in historical_states.columns and "phase" in state:
            if not hasattr(self, "_mask_cache") or self._mask_cache.get("_total_len") != len(historical_states):
                self._mask_cache = {"_total_len": len(historical_states)}
            
            phase_val = state["phase"]
            if phase_val not in self._mask_cache:
                m = (historical_states["phase"].values == phase_val)
                if m.sum() >= self.k * 2:
                    self._mask_cache[phase_val] = m
                else:
                    self._mask_cache[phase_val] = None
            
            mask = self._mask_cache[phase_val]
            if mask is not None:
                df = historical_states[mask]
                has_phase_filter = True
            else:
                df = historical_states
        else:
            df = historical_states

        # -----------------------------
        # Feature extraction & standard deviations (with caching)
        # -----------------------------
        if not hasattr(self, "_sigma_cache"):
            self._sigma_cache = {}

        # Cache key is based on the phase and shape of df to ensure correctness
        cache_key = (
            state["phase"] if (self.phase_mode == "filter" and "phase" in state) else "all",
            df.shape[0]
        )

        if cache_key not in self._sigma_cache:
            sigma_rr = df["runs_remaining"].std()
            sigma_br = df["balls_remaining"].std()
            sigma_wk = df["wickets_lost"].std()

            # Fallback to defaults if std is zero or NaN
            sigma_rr = sigma_rr if (pd.notna(sigma_rr) and sigma_rr > 0) else 38.0
            sigma_br = sigma_br if (pd.notna(sigma_br) and sigma_br > 0) else 34.0
            sigma_wk = sigma_wk if (pd.notna(sigma_wk) and sigma_wk > 0) else 2.5

            sigma_p = 5.0
            if self.use_pressure:
                # Direct std calculation using temp array
                temp_p = df["required_run_rate"].values * (1 + df["wickets_lost"].values / 10.0)
                std_p = temp_p.std()
                sigma_p = std_p if (pd.notna(std_p) and std_p > 0) else 5.0

            sigma_res = 300.0
            if self.use_resources:
                temp_res = (10 - df["wickets_lost"].values) * df["balls_remaining"].values
                std_res = temp_res.std()
                sigma_res = std_res if (pd.notna(std_res) and std_res > 0) else 300.0

            self._sigma_cache[cache_key] = (sigma_rr, sigma_br, sigma_wk, sigma_p, sigma_res)

        sigma_rr, sigma_br, sigma_wk, sigma_p, sigma_res = self._sigma_cache[cache_key]

        # -----------------------------
        # Normalised distance metric using variance-based scaling (NumPy vectorized)
        # -----------------------------
        rr = (df["runs_remaining"].values - state["runs_remaining"]) / sigma_rr
        br = (df["balls_remaining"].values - state["balls_remaining"]) / sigma_br
        wk = (df["wickets_lost"].values - state["wickets_lost"]) / sigma_wk

        distance = (
            self.w_r * (rr ** 2) +
            self.w_b * (br ** 2) +
            self.w_w * (wk ** 2)
        )

        # -----------------------------
        # Phase as a distance feature
        # -----------------------------
        if self.phase_mode == "feature" and "phase" in df.columns and "phase" in state:
            phase_map = {"powerplay": 0, "middle": 1, "death": 2}
            state_phase_val = phase_map.get(state["phase"], 0)
            df_phase_val = df["phase"].map(phase_map).fillna(0).values
            phase_diff = np.abs(df_phase_val - state_phase_val)
            distance = distance + (phase_diff / 1.0) ** 2

        # -----------------------------
        # Feature Interactions (with caching of full arrays)
        # -----------------------------
        if self.use_pressure:
            if not hasattr(self, "_full_pressure") or len(self._full_pressure) != len(historical_states):
                self._full_pressure = historical_states["required_run_rate"].values * (1 + historical_states["wickets_lost"].values / 10.0)
            
            df_pressure = self._full_pressure[mask] if has_phase_filter else self._full_pressure
            state_pressure = state["required_run_rate"] * (1 + state["wickets_lost"] / 10.0)
            p_diff = (df_pressure - state_pressure) / sigma_p
            distance = distance + self.w_p * (p_diff ** 2)

        if self.use_resources:
            if not hasattr(self, "_full_resources") or len(self._full_resources) != len(historical_states):
                self._full_resources = (10 - historical_states["wickets_lost"].values) * historical_states["balls_remaining"].values
            
            df_resources = self._full_resources[mask] if has_phase_filter else self._full_resources
            state_resources = (10 - state["wickets_lost"]) * state["balls_remaining"]
            res_diff = (df_resources - state_resources) / sigma_res
            distance = distance + (res_diff ** 2)

        # -----------------------------
        # Zero-copy match exclusion (to prevent data leakage)
        # -----------------------------
        if "match_id" in df.columns and "match_id" in state:
            same_match_mask = df["match_id"].values == state["match_id"]
            distance = np.where(same_match_mask, np.inf, distance)

        # -----------------------------
        # nearest neighbours via argpartition
        # -----------------------------
        n_samples = len(df)
        if n_samples <= self.k:
            idx = np.arange(n_samples)
        else:
            # Partition to find top k smallest distances
            idx = np.argpartition(distance, self.k)[:self.k]
            # Sort only the top k elements
            idx = idx[np.argsort(distance[idx])]

        nearest = df.iloc[idx].copy()
        nearest["distance"] = distance[idx]

        return nearest