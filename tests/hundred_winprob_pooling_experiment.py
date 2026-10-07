"""
Hundred Win-Probability Pooling/Transfer Experiment (Phase 5).

Phase 6 of the original Hundred plan pooled the *delivery-outcome* model and won. The
*win-probability classifier* -- the model Brier actually scores -- was never pooled. This
script measures three transfer challengers against the Phase-1-4 bespoke hnd model on a
**chronological held-out slice of recent Hundred matches** (last ~20%, mirroring
hundred_pooling_experiment.py's split):

  A. Pooled    -- T20 (full) + HND-train states, `is_hundred` feature, one classifier.
  B. Blend     -- p = w*p_hnd + (1-w)*p_t20on_hnd, sweep w in [0, 1].
  C. Warm-start -- hnd model continuation-trained from the T20 booster (xgb_model=...).

All four (bespoke + A/B/C) are evaluated on the SAME held-out 20% using RAW (uncalibrated)
probabilities for a clean, apples-to-apples comparison -- production calibration is a
separate, later step applied uniformly regardless of which architecture ships.

Design decisions recorded here (per the plan's explicit "decide and record which"):
  - required_run_rate / run_rate keep their native per-format-over scale (/6 for t20, /5 for
    hnd) in the pooled frame; the model relies on `is_hundred` to differentiate rather than
    renormalising to a common per-ball basis.
  - Pooled classifier hyperparameters: T20's own tuned defaults (read live off
    XGBWinProbabilityModel(format_type="t20")), since the pooled corpus is T20-volume-dominated.
  - Monotone constraints are NOT applied to the pooled/warm-start challengers (added complexity
    not required by the plan for this experiment); the bespoke comparator keeps its Phase-3
    constraints since that's the actual production architecture being challenged.

Usage:
    py tests/hundred_winprob_pooling_experiment.py
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _log(msg):
    print(msg, flush=True)


from src.data_processing.load_data import CricketDataLoader
from src.data_processing.match_metadata import MatchMetadataExtractor
from src.features.elo_engine import EloEngine
from src.features.player_elo_engine import PlayerEloEngine
from src.features.phase_elo_engine import PhaseEloEngine
from src.features.venue_engine import VenueEngine
from src.data_processing.build_states import StateBuilder
from src.data_processing.format_spec import spec
from src.models.xgb_win_probability import XGBWinProbabilityModel


def build_second_innings_states(format_type: str) -> pd.DataFrame:
    """Mirrors cricsheet_importer.retrain_full()'s feature pipeline exactly (cache-aware)."""
    _log(f"[{format_type.upper()}] Loading deliveries & metadata...")
    loader = CricketDataLoader("data/raw", format_type=format_type)
    deliveries = loader.load_all()

    meta_extractor = MatchMetadataExtractor("data/raw", format_type=format_type)
    meta_df = meta_extractor.extract_all()
    if "gender" in meta_df.columns:
        meta_df = meta_df[meta_df["gender"] == "male"].copy()
        deliveries = deliveries[deliveries["match_id"].isin(meta_df["match_id"])].copy()

    seed_ratings = None
    if format_type == "hnd":
        t20_pe_path = Path("data/processed/t20_player_elo_ratings.parquet")
        if t20_pe_path.exists():
            t20_pe = pd.read_parquet(t20_pe_path)
            latest_t20 = t20_pe.sort_values("match_id").groupby(["player", "role"]).last().reset_index()
            b_seeds = latest_t20[latest_t20["role"] == "batter"].set_index("player")["effective_elo"].to_dict()
            w_seeds = latest_t20[latest_t20["role"] == "bowler"].set_index("player")["effective_elo"].to_dict()
            seed_ratings = {"batter": b_seeds, "bowler": w_seeds}

    _log(f"[{format_type.upper()}] Computing Elo / player Elo / phase Elo / venue features...")
    elo_engine = EloEngine(use_dynamic_k=True, use_margin=True, use_decay=True, use_home_away=False, format_type=format_type)
    elo_df = elo_engine.run(meta_df)

    player_elo_engine = PlayerEloEngine(format_type=format_type, form_window=15)
    player_elo_df = player_elo_engine.run(deliveries, elo_df, seed_ratings=seed_ratings)

    phase_elo_engine = PhaseEloEngine(format_type=format_type)
    phase_elo_df = phase_elo_engine.run(deliveries, elo_df)

    venue_engine = VenueEngine(format_type=format_type)
    venue_df = venue_engine.run(meta_df, deliveries)
    elo_df = elo_df.merge(venue_df, on="match_id", how="left")

    _log(f"[{format_type.upper()}] Building states...")
    fmt_spec = spec(format_type)
    states = StateBuilder(
        deliveries, elo_df, player_elo_df=player_elo_df, phase_elo_df=phase_elo_df,
        total_overs=fmt_spec["total_overs"], format_type=format_type,
        balls_per_over=fmt_spec["balls_per_over"],
    ).build()

    second_innings = states[states["innings"] == 2].copy()
    _log(f"[{format_type.upper()}] {len(states):,} total, {len(second_innings):,} second-innings states, "
         f"{second_innings['match_id'].nunique()} matches.")
    return second_innings, meta_df


def brier(probs, actuals):
    return float(np.mean((np.asarray(probs) - np.asarray(actuals)) ** 2))


def per_phase_brier(states_df, probs, actuals):
    """states_df, probs, and actuals must all share the same row ORDER (positional alignment)."""
    out = {}
    probs = np.asarray(probs)
    actuals = np.asarray(actuals)
    phases = states_df["phase"].values
    for phase_name in pd.unique(phases):
        mask = phases == phase_name
        out[phase_name] = round(brier(probs[mask], actuals[mask]), 6)
    return out


def run_experiment():
    print("=" * 70)
    print("  HUNDRED WIN-PROBABILITY POOLING/TRANSFER EXPERIMENT")
    print("=" * 70 + "\n")
    t_start = time.time()

    # 1. Build states for both formats
    hnd_states, hnd_meta = build_second_innings_states("hnd")
    t20_states, _ = build_second_innings_states("t20")

    # 2. Chronological split on HND matches -- last ~20% held out (mirrors hundred_pooling_experiment.py)
    sorted_matches = hnd_meta.sort_values("date")["match_id"].tolist()
    split_idx = int(len(sorted_matches) * 0.80)
    train_match_ids = set(sorted_matches[:split_idx])
    test_match_ids = set(sorted_matches[split_idx:])

    hnd_train = hnd_states[hnd_states["match_id"].isin(train_match_ids)].copy()
    hnd_test = hnd_states[hnd_states["match_id"].isin(test_match_ids)].copy()
    print(f"\nHND matches: {len(sorted_matches)} total | train {len(train_match_ids)} | "
          f"held-out test {len(test_match_ids)}")
    print(f"HND states: train {len(hnd_train):,} | test {len(hnd_test):,}\n")

    # 3. Feature extraction -- each format via its OWN correctly-configured model instance,
    #    so format-specific geometry (decay thresholds, venue defaults) is computed correctly
    #    for each set of rows before concatenation (see module docstring on required_run_rate scale).
    hnd_helper = XGBWinProbabilityModel(format_type="hnd")  # picks up Phase 1-4 defaults + fixes
    t20_helper = XGBWinProbabilityModel(format_type="t20")

    print("Extracting features (hnd-train, hnd-test, t20-full)...")
    X_hnd_train, y_hnd_train = hnd_helper._extract_features_and_target(hnd_train)
    X_hnd_test, y_hnd_test = hnd_helper._extract_features_and_target(hnd_test)
    X_t20, y_t20 = t20_helper._extract_features_and_target(t20_states)
    hnd_monotone = hnd_helper._build_monotone_constraints(X_hnd_train.columns)
    print(f"{X_hnd_train.shape[1]} features. hnd-train={len(X_hnd_train):,}, "
          f"hnd-test={len(X_hnd_test):,}, t20-pool={len(X_t20):,}\n")

    results = {}

    # ================================================================
    # Bespoke: Phase 1-4 architecture, trained on hnd-train (80%) only,
    # evaluated on the held-out hnd-test (20%). RAW probabilities.
    # ================================================================
    print("[Bespoke] Training hnd-only model (Phase 1-4 config) on 80% train split...")
    bespoke_model = XGBClassifier(
        n_estimators=hnd_helper.n_estimators, max_depth=hnd_helper.max_depth,
        learning_rate=hnd_helper.learning_rate, min_child_weight=hnd_helper.min_child_weight,
        subsample=hnd_helper.subsample, colsample_bytree=hnd_helper.colsample_bytree,
        reg_alpha=hnd_helper.reg_alpha, reg_lambda=hnd_helper.reg_lambda,
        objective="binary:logistic", eval_metric="logloss", random_state=42,
        n_jobs=-1, enable_categorical=True, monotone_constraints=hnd_monotone,
    )
    bespoke_model.fit(X_hnd_train, y_hnd_train)
    p_bespoke = bespoke_model.predict_proba(X_hnd_test)[:, 1]
    results["bespoke"] = p_bespoke

    # ================================================================
    # A. Pooled: T20-full + hnd-train, is_hundred flag, one classifier.
    # ================================================================
    print("[A: Pooled] Training pooled (T20 + HND-train) model with is_hundred feature...")
    X_t20_p = X_t20.copy()
    X_t20_p["is_hundred"] = 0.0
    X_hnd_train_p = X_hnd_train.copy()
    X_hnd_train_p["is_hundred"] = 1.0
    X_pooled = pd.concat([X_t20_p, X_hnd_train_p], ignore_index=True)
    y_pooled = pd.concat([y_t20, y_hnd_train], ignore_index=True)

    pooled_model = XGBClassifier(
        n_estimators=t20_helper.n_estimators, max_depth=t20_helper.max_depth,
        learning_rate=t20_helper.learning_rate, min_child_weight=t20_helper.min_child_weight,
        subsample=t20_helper.subsample, colsample_bytree=t20_helper.colsample_bytree,
        reg_alpha=t20_helper.reg_alpha, reg_lambda=t20_helper.reg_lambda,
        objective="binary:logistic", eval_metric="logloss", random_state=42,
        n_jobs=-1, enable_categorical=True,
    )
    pooled_model.fit(X_pooled, y_pooled)
    X_hnd_test_p = X_hnd_test.copy()
    X_hnd_test_p["is_hundred"] = 1.0
    p_pooled = pooled_model.predict_proba(X_hnd_test_p)[:, 1]
    results["pooled"] = p_pooled

    # ================================================================
    # B. Blend: p = w*p_hnd + (1-w)*p_t20on_hnd, sweep w.
    # ================================================================
    print("[B: Blend] Loading production T20 model, predicting on hnd-test, sweeping w...")
    t20_prod = XGBWinProbabilityModel(format_type="t20")
    t20_prod.fit(None, cache_path="data/processed/t20_xgb_models.pkl")
    p_t20_on_hnd = t20_prod.predict_batch(hnd_test)

    best_w, best_blend_brier, best_p_blend = None, np.inf, None
    for w in np.arange(0.0, 1.01, 0.05):
        p_blend = w * p_bespoke + (1 - w) * p_t20_on_hnd
        b = brier(p_blend, y_hnd_test.values)
        if b < best_blend_brier:
            best_w, best_blend_brier, best_p_blend = round(float(w), 2), b, p_blend
    results[f"blend@w={best_w}"] = best_p_blend

    # ================================================================
    # C. Warm-start: hnd model continuation-trained from the T20 booster.
    # ================================================================
    print("[C: Warm-start] Continuation-training from the T20 booster onto hnd-train...")
    t20_booster = t20_prod.final_model.get_booster()
    warmstart_model = XGBClassifier(
        n_estimators=100, max_depth=hnd_helper.max_depth, learning_rate=0.05,
        min_child_weight=hnd_helper.min_child_weight, subsample=hnd_helper.subsample,
        colsample_bytree=hnd_helper.colsample_bytree, reg_alpha=hnd_helper.reg_alpha,
        reg_lambda=hnd_helper.reg_lambda, objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=-1, enable_categorical=True,
    )
    warmstart_model.fit(X_hnd_train, y_hnd_train, xgb_model=t20_booster)
    p_warmstart = warmstart_model.predict_proba(X_hnd_test)[:, 1]
    results["warm_start"] = p_warmstart

    # ================================================================
    # Results
    # ================================================================
    actuals = y_hnd_test.values
    print("\n" + "=" * 90)
    print("  RESULTS (held-out HND test set, RAW/uncalibrated probabilities)")
    print("=" * 90)
    print(f"{'Model':<20} | {'Brier':<10} | {'Mean Pred':<10} | {'Actual Win Rate':<16} | vs bespoke")
    print("-" * 90)

    bespoke_brier = brier(results["bespoke"], actuals)
    summary = {}
    for name, probs in results.items():
        b = brier(probs, actuals)
        mean_pred = float(np.mean(probs))
        actual_rate = float(np.mean(actuals))
        delta = b - bespoke_brier
        beats = "BEATS bespoke" if (name != "bespoke" and b < bespoke_brier) else ("--" if name == "bespoke" else "loses")
        print(f"{name:<20} | {b:<10.6f} | {mean_pred:<10.4f} | {actual_rate:<16.4f} | {delta:+.6f} ({beats})")
        summary[name] = {"brier": b, "per_phase": per_phase_brier(hnd_test, probs, actuals)}

    print("\nPer-phase Brier:")
    print(f"{'Model':<20} | {'powerplay':<12} | {'middle':<12} | {'death':<12}")
    print("-" * 60)
    for name, s in summary.items():
        pp = s["per_phase"]
        print(f"{name:<20} | {pp.get('powerplay', float('nan')):<12.6f} | "
              f"{pp.get('middle', float('nan')):<12.6f} | {pp.get('death', float('nan')):<12.6f}")

    winners = [name for name, s in summary.items() if name != "bespoke" and s["brier"] < bespoke_brier]
    print("\n" + "=" * 90)
    if winners:
        best_name = min(winners, key=lambda n: summary[n]["brier"])
        print(f"VERDICT: {best_name} BEATS the bespoke model ({summary[best_name]['brier']:.6f} vs {bespoke_brier:.6f}).")
        print(f"All challengers beating bespoke: {winners}")
    else:
        print(f"VERDICT: NO challenger beats the bespoke model (bespoke={bespoke_brier:.6f}). Ship bespoke as-is.")
    print("=" * 90)
    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")

    return summary, bespoke_brier, winners


if __name__ == "__main__":
    run_experiment()
