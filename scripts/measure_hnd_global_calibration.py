"""
Follow-up to the hnd feedback-loop investigation (scripts/measure_score_feedback.py --formats hnd):
that pass found current_team_score's sensitivity slope WRONG-SIGNED for a runaway loop, its
partial-dependence shape non-monotone in every over-bucket, and its ablation cost statistically
zero -- all pointing AWAY from the od-style feedback mechanism as the cause of hnd's measured
-10.34-run MC calibration bias.

This script checks the more boring, more likely alternative: is the delivery-outcome model just
GLOBALLY miscalibrated (predicted class probabilities systematically off from observed
frequencies, unconditional on any specific feature), consistent with under-fitting a genuinely
small training corpus (167 matches, ~32K deliveries -- a small fraction of t20/od's)?

Ships nothing -- pure measurement, OOF throughout (each delivery scored by the fold model that
did NOT train on its match, exactly as sim_baseline_metrics.py's own gate does it).

Reports, for BOTH heads, predicted-mean vs actual-frequency at the whole-corpus level:
  - extras gate   : P(legal) / P(wide) / P(noball)
  - legal outcome : P(0) / P(1) / P(2) / P(3) / P(4) / P(6) / P(wicket)
  - derived       : mean runs-per-legal-ball (bat runs only, wicket credits 0 -- mirrors
                    innings_simulator.LEGAL_RUN_VALUES) and P(wicket), both predicted vs actual,
                    with match-paired significance (the match is this repo's unit of independence
                    for correlated deliveries -- pooling would manufacture false confidence at
                    n=30,716 rows).

Run:
    py scripts/measure_hnd_global_calibration.py
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

from src.models.delivery_outcome_model import (
    DeliveryOutcomeModel, LEGAL_CLASSES, LEGAL_CLASS_TO_IDX,
    EXTRAS_CLASSES, EXTRAS_CLASS_TO_IDX,
)
from tests.sim_baseline_metrics import build_frame

FMT = "hnd"
RUN_VALUE = np.array([0 if c == "W" else int(c) for c in LEGAL_CLASSES], dtype=np.float64)
WICKET_IDX = LEGAL_CLASS_TO_IDX["W"]


def load_model(fmt: str) -> DeliveryOutcomeModel:
    cache_path = PROJECT_ROOT / "data" / "processed" / f"{fmt}_delivery_outcome_model.pkl"
    model = DeliveryOutcomeModel(format_type=fmt)
    model.fit(None, cache_path=cache_path)
    return model


def oof_proba(model, df, fold_models_key):
    """(N, n_classes) OOF probabilities, each row scored by the fold model that did NOT train
    on its match. Rows whose fold has no model are left NaN.

    Goes through model.oof_features() so a pooled model (hnd, since 2026-08-12) gets its
    synthesized is_hundred column automatically."""
    n_classes = len(EXTRAS_CLASSES) if fold_models_key == "extras" else len(LEGAL_CLASSES)
    out = np.full((len(df), n_classes), np.nan)
    folds = df["_fold"].values
    X = model.oof_features(df).values
    for fold_idx, pack in model.fold_models.items():
        m = folds == fold_idx
        if m.any():
            out[m] = pack[fold_models_key].predict_proba(X[m])
    return out


def main():
    print("=" * 78)
    print(f"HND GLOBAL CALIBRATION CHECK (follow-up to the feedback-loop investigation)")
    print("=" * 78)

    model = load_model(FMT)
    frame = build_frame(FMT)
    frame["_fold"] = frame["match_id"].map(model.match_to_fold)
    frame = frame.dropna(subset=["_fold"]).copy()
    frame["_fold"] = frame["_fold"].astype(int)
    print(f"  {len(frame):,} deliveries, {frame['match_id'].nunique():,} matches (all OOF-scored)")

    # ---- extras gate: P(legal)/P(wide)/P(noball), whole corpus ----
    extras_proba = oof_proba(model, frame, "extras")
    ok = ~np.isnan(extras_proba[:, 0])
    actual_extras = frame.loc[ok, "extra_class"].map(EXTRAS_CLASS_TO_IDX).values
    print("\n  [extras gate] predicted mean P(class) vs actual frequency, whole corpus")
    for cls, idx in EXTRAS_CLASS_TO_IDX.items():
        pred_mean = float(extras_proba[ok, idx].mean())
        actual_freq = float((actual_extras == idx).mean())
        print(f"    {cls:8s} predicted={pred_mean:.4f}  actual={actual_freq:.4f}  "
              f"delta={pred_mean - actual_freq:+.4f}")

    # ---- legal outcome: P(0..6,W), legal deliveries only ----
    legal = frame[frame["extra_class"] == "legal"].reset_index(drop=True)
    legal_proba = oof_proba(model, legal, "legal")
    ok_l = ~np.isnan(legal_proba[:, 0])
    legal_ok = legal[ok_l].reset_index(drop=True)
    proba_ok = legal_proba[ok_l]
    actual_legal = legal_ok["legal_outcome"].map(LEGAL_CLASS_TO_IDX).values

    print("\n  [legal outcome] predicted mean P(class) vs actual frequency, legal deliveries only")
    for cls in LEGAL_CLASSES:
        idx = LEGAL_CLASS_TO_IDX[str(cls)]
        pred_mean = float(proba_ok[:, idx].mean())
        actual_freq = float((actual_legal == idx).mean())
        print(f"    {str(cls):8s} predicted={pred_mean:.4f}  actual={actual_freq:.4f}  "
              f"delta={pred_mean - actual_freq:+.4f}")

    # ---- derived: runs-per-legal-ball and P(wicket), match-paired ----
    pred_runs = proba_ok @ RUN_VALUE
    actual_runs = RUN_VALUE[actual_legal]
    pred_pw = proba_ok[:, WICKET_IDX]
    actual_pw = (actual_legal == WICKET_IDX).astype(np.float64)

    print(f"\n  [derived, whole corpus, n={len(legal_ok):,} legal deliveries]")
    print(f"    runs/legal-ball   predicted={pred_runs.mean():.4f}  actual={actual_runs.mean():.4f}  "
          f"delta={pred_runs.mean() - actual_runs.mean():+.4f}")
    print(f"    P(wicket)         predicted={pred_pw.mean():.4f}  actual={actual_pw.mean():.4f}  "
          f"delta={pred_pw.mean() - actual_pw.mean():+.4f}")

    def match_paired(metric_pred, metric_actual, match_ids, label, unit=None):
        d = pd.DataFrame({"match_id": match_ids, "d": metric_actual - metric_pred}) \
            .groupby("match_id")["d"].mean()
        se = d.std(ddof=1) / np.sqrt(len(d))
        t = d.mean() / se if se > 0 else float("nan")
        print(f"    {label}: match-paired mean(actual-pred)={d.mean():+.4f}  "
              f"SE={se:.4f}  t={t:.2f}  n_matches={len(d)}")
        if unit:
            # Projected over a full innings' worth of legal balls -- a back-of-envelope
            # translation of a PER-BALL bias into the INNINGS-TOTAL effect it would produce if
            # it held constant across the innings (a rough scale check, not a simulation).
            avg_legal_balls_per_innings = len(legal_ok) / frame["match_id"].nunique() / 2
            print(f"    -> at ~{avg_legal_balls_per_innings:.0f} legal balls/innings, this "
                  f"per-ball bias projects to ~{d.mean() * avg_legal_balls_per_innings:+.2f} "
                  f"{unit}/innings if it held constant (a rough scale check, not a simulation)")

    match_paired(pred_runs, actual_runs, legal_ok["match_id"].values, "runs/legal-ball", unit="runs")
    match_paired(pred_pw, actual_pw, legal_ok["match_id"].values, "P(wicket)", unit="wickets")


if __name__ == "__main__":
    main()
