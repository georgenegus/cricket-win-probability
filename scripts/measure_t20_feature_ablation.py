"""
Phase 2 of t20_mae_gate_plan.md -- prices candidate delivery-outcome features OOF, on a
subsampled control-relative ablation, BEFORE paying for a full retrain (Phase 3).

Ships nothing. Follows the same methodology as mc_runrate_feedback_plan.md's Phase 0
(scripts/measure_score_feedback.py's run_ablation): subsample by MATCH, GroupKFold by match_id,
hyperparameters identical to production, and measure the log-loss delta against a CONTROL variant
retrained on the exact same subsample and folds -- never against the full-corpus lock, which
differs for reasons unrelated to the features being priced.

Candidate variants (Phase 2's own ordering, cheapest/most-promising first):
  +squad_elo -- batting_squad_elo / bowling_squad_elo, named-XI aggregates pulled straight from
                {fmt}_player_elo_ratings.parquet via get_match_squad_elo -- the SAME leakage-safe,
                strictly-prior, position-weighted squad Elo the win-prob model already trains on.
                This is Phase 0 hypothesis (3): the per-ball model has batter/bowler Elo but no
                squad-level aggregate, so it never sees "strong XI vs weak attack".
  +all       -- squad_elo plus competition (category code), era (season year) and toss (won-toss-
                and-batting / elected-to-field flags), RIDING ALONG WITH squad_elo per the plan's
                own instruction: these were worth ~0.32 match-level MAE runs on their own (18.6%
                -> 20.3% R^2 in the match-level regressor) -- real but small, so only worth a
                retrain if they ride along with squad_elo rather than requiring their own pass.

LEAKAGE DISCIPLINE: squad Elo comes untouched from get_match_squad_elo (named XI, strictly-prior
ratings) -- never recomputed from participants, which would encode the outcome (this plan's
Executive Summary: a naive n_batters_used-style aggregate scored a fake 35.7% R^2 this way once).
competition/era/toss are plain category codes / calendar features, all known before a ball is
bowled.

METHODOLOGY (identical discipline to measure_score_feedback.py's run_ablation, reused here):
  * subsample by MATCH, never by delivery; GroupKFold by match_id, hyperparameters identical to
    production -- the point is to price the feature, not a different model.
  * every variant sees the same matches, same splits, same rows -- feature columns only differ.
  * significance is match-paired (mean per-match log-loss delta), |t| > 2 is the bar; pooling
    correlated deliveries has already turned noise into an apparent 5.8-SE finding once on this
    plan's sibling (mc_runrate_feedback_plan.md).
  * only the legal-outcome head is ablated -- team quality bears on WHICH runs are scored, not on
    the wide/no-ball extras gate, matching measure_score_feedback.py's own scope.

NOT measured here: a direct simulated-MAE read. That would need the new features threaded through
innings_simulator's _assemble_state_matrix as well as build_training_frame -- exactly the
train/serve drift risk Phase 3's own risk table flags -- and is safer paid once, in Phase 3, if
this pricing pass justifies a retrain, than risked twice for a throwaway pricing script. The
match-level regressor's MAE deltas (Executive Summary) are used as the standing proxy instead.

Run:
    py scripts/measure_t20_feature_ablation.py
    py scripts/measure_t20_feature_ablation.py --ablation-matches 800 --ablation-folds 2
"""

import os

# Must precede the numpy/xgboost import -- 7.7GB box, oversubscribed BLAS threads turn a
# subsampled retrain into a swap-thrash (see machine_memory_constraints memory / other scripts).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys
import json
import time
import argparse
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
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss

from src.models.delivery_outcome_model import (
    DeliveryOutcomeModel, FEATURES, LEGAL_CLASSES, LEGAL_CLASS_TO_IDX,
)
from src.features.player_elo_engine import get_match_squad_elo
from tests.sim_baseline_metrics import build_frame, BASELINE_PATH

FMT = "t20"
OUT_DIR = PROJECT_ROOT / "outputs"

SQUAD_ELO_COLS = ["batting_squad_elo", "bowling_squad_elo"]
CONTEXT_COLS = ["comp_code", "era_year", "toss_won_by_batting_side", "toss_decision_field"]


# ----------------------------------------------------------------------------------------
# feature attachment -- all leakage-safe / known pre-match, per the plan's discipline
# ----------------------------------------------------------------------------------------

def attach_squad_elo(frame: pd.DataFrame, fmt: str) -> pd.DataFrame:
    """batting_squad_elo / bowling_squad_elo per delivery, from the named-XI, strictly-prior
    squad Elo already computed for the win-prob model (get_match_squad_elo) -- never recomputed
    from participants. Missing matches (older corpus tail with no squad-Elo columns) fall back to
    1500.0, the same neutral default _join_player_elo uses for missing player Elo."""
    meta = pd.read_parquet(PROJECT_ROOT / f"data/processed/{fmt}_match_metadata.parquet")[
        ["match_id", "team_1", "team_2"]
    ]
    meta["match_id"] = meta["match_id"].astype(str)

    frame = frame.copy()
    frame["match_id"] = frame["match_id"].astype(str)
    frame = frame.merge(meta, on="match_id", how="left")
    frame["bowling_team"] = np.where(
        frame["batting_team"] == frame["team_1"], frame["team_2"], frame["team_1"]
    )

    rows = []
    for mid, t1, t2 in meta[meta["match_id"].isin(frame["match_id"].unique())].itertuples(index=False):
        vals = get_match_squad_elo(fmt, mid, processed_dir=str(PROJECT_ROOT / "data/processed"))
        if vals is None:
            continue
        t1_bat, t1_bowl, t2_bat, t2_bowl = vals
        rows.append({"match_id": mid, "team": t1, "own_bat_sq": t1_bat, "own_bowl_sq": t1_bowl})
        rows.append({"match_id": mid, "team": t2, "own_bat_sq": t2_bat, "own_bowl_sq": t2_bowl})
    long = pd.DataFrame(rows)
    print(f"    squad-Elo coverage: {long['match_id'].nunique():,}/{frame['match_id'].nunique():,} matches", flush=True)

    if long.empty:
        frame["batting_squad_elo"] = 1500.0
        frame["bowling_squad_elo"] = 1500.0
        return frame.drop(columns=["team_1", "team_2", "bowling_team"])

    bat_lu = long.rename(columns={"team": "batting_team", "own_bat_sq": "batting_squad_elo"})[
        ["match_id", "batting_team", "batting_squad_elo"]
    ]
    frame = frame.merge(bat_lu, on=["match_id", "batting_team"], how="left")
    bowl_lu = long.rename(columns={"team": "bowling_team", "own_bowl_sq": "bowling_squad_elo"})[
        ["match_id", "bowling_team", "bowling_squad_elo"]
    ]
    frame = frame.merge(bowl_lu, on=["match_id", "bowling_team"], how="left")

    frame["batting_squad_elo"] = frame["batting_squad_elo"].fillna(1500.0)
    frame["bowling_squad_elo"] = frame["bowling_squad_elo"].fillna(1500.0)
    return frame.drop(columns=["team_1", "team_2", "bowling_team"])


def attach_context_features(frame: pd.DataFrame, fmt: str) -> pd.DataFrame:
    """comp_code (plain category code, NOT a target encoding -- no leakage), era_year (season),
    and toss flags. All known at toss time / from the fixture, never from the match outcome."""
    meta = pd.read_parquet(PROJECT_ROOT / f"data/processed/{fmt}_match_metadata.parquet")[
        ["match_id", "toss_winner", "toss_decision"]
    ]
    meta["match_id"] = meta["match_id"].astype(str)

    frame = frame.copy()
    frame["match_id"] = frame["match_id"].astype(str)
    frame = frame.merge(meta, on="match_id", how="left")

    frame["comp_code"] = frame["competition"].astype("category").cat.codes.astype(float)
    frame["era_year"] = pd.to_datetime(frame["match_date"], errors="coerce").dt.year.astype(float)
    frame["toss_won_by_batting_side"] = (frame["toss_winner"] == frame["batting_team"]).astype(float)
    frame["toss_decision_field"] = (
        frame["toss_decision"].astype(str).str.lower() == "field"
    ).astype(float)
    return frame.drop(columns=["toss_winner", "toss_decision"])


# ----------------------------------------------------------------------------------------
# ablation -- same shape as measure_score_feedback.py's run_ablation, generalised to
# APPENDED feature sets rather than a single drop/replace
# ----------------------------------------------------------------------------------------

VARIANTS = {
    "control": [],
    "+squad_elo": SQUAD_ELO_COLS,
    "+all": SQUAD_ELO_COLS + CONTEXT_COLS,
}


def run_ablation(frame: pd.DataFrame, n_matches: int, n_folds_used: int,
                 n_splits: int = 5, seed: int = 42) -> dict:
    matches = frame["match_id"].unique()
    rng = np.random.default_rng(seed)
    if n_matches and len(matches) > n_matches:
        matches = rng.choice(matches, size=n_matches, replace=False)
    legal = frame[(frame["extra_class"] == "legal")
                  & frame["match_id"].isin(matches)].reset_index(drop=True)

    y = legal["legal_outcome"].map(LEGAL_CLASS_TO_IDX).values
    groups = legal["match_id"].values
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(legal, y, groups=groups))[:n_folds_used]
    n_scored = sum(len(v) for _, v in splits)

    print(f"    ablation corpus: {len(legal):,} legal deliveries over {len(matches):,} matches; "
          f"{n_folds_used}/{n_splits} folds -> {n_scored:,} scored OOF", flush=True)

    labels = list(range(len(LEGAL_CLASSES)))
    proto = DeliveryOutcomeModel(format_type=FMT, n_splits=n_splits, random_state=seed)

    results = {}
    per_row_ll = {}
    order = np.concatenate([v for _, v in splits])
    for variant, extra_cols in VARIANTS.items():
        cols = list(FEATURES) + extra_cols   # append-only: never reorder FEATURES itself
        X = legal[cols].values
        lls, chunks = [], []
        t0 = time.time()
        for fold_idx, (tr, va) in enumerate(splits):
            clf = proto._make_legal_clf()
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[va])
            lls.append(float(log_loss(y[va], p, labels=labels)))
            chunks.append(-np.log(np.clip(p[np.arange(len(va)), y[va]], 1e-15, None)))
            del clf
        per_row_ll[variant] = np.concatenate(chunks)
        results[variant] = {
            "oof_log_loss": float(np.average(lls, weights=[len(v) for _, v in splits])),
            "per_fold_log_loss": [round(x, 6) for x in lls],
            "n_features": len(cols),
            "fit_seconds": round(time.time() - t0, 1),
        }
        print(f"    {variant:12s} OOF legal log-loss = {results[variant]['oof_log_loss']:.6f} "
              f"({results[variant]['fit_seconds']:.0f}s, {len(cols)} features)", flush=True)

    scored_matches = pd.Series(groups[order], name="match_id")
    control_ll = per_row_ll["control"]
    for variant in ("+squad_elo", "+all"):
        diff = pd.DataFrame({
            "match_id": scored_matches.values,
            "d": per_row_ll[variant] - control_ll,
        }).groupby("match_id")["d"].mean()
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        results[variant]["delta_vs_control"] = round(
            results[variant]["oof_log_loss"] - results["control"]["oof_log_loss"], 6)
        results[variant]["match_paired"] = {
            "n_matches": int(len(diff)),
            "mean_delta": round(float(diff.mean()), 6),
            "se": round(float(se), 6),
            "t": round(float(diff.mean() / se), 2) if se > 0 else None,
            "share_worse": round(float((diff > 0).mean()), 3),
        }

    # +all vs +squad_elo directly -- "do competition/era/toss earn their keep ON TOP of
    # squad_elo", the question the plan's own instruction actually asks (see module docstring).
    diff_incremental = pd.DataFrame({
        "match_id": scored_matches.values,
        "d": per_row_ll["+all"] - per_row_ll["+squad_elo"],
    }).groupby("match_id")["d"].mean()
    se_inc = diff_incremental.std(ddof=1) / np.sqrt(len(diff_incremental))
    results["+all"]["delta_vs_squad_elo"] = round(
        results["+all"]["oof_log_loss"] - results["+squad_elo"]["oof_log_loss"], 6)
    results["+all"]["match_paired_vs_squad_elo"] = {
        "n_matches": int(len(diff_incremental)),
        "mean_delta": round(float(diff_incremental.mean()), 6),
        "se": round(float(se_inc), 6),
        "t": round(float(diff_incremental.mean() / se_inc), 2) if se_inc > 0 else None,
    }

    results["_meta"] = {
        "n_matches": int(len(matches)), "n_legal_deliveries": int(len(legal)),
        "n_scored_oof": int(n_scored), "n_folds_used": n_folds_used, "n_splits": n_splits,
    }
    return results


# ----------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--ablation-matches", type=int, default=800,
                   help="Matches subsampled for the ablation retrain (0 = all; very slow).")
    p.add_argument("--ablation-folds", type=int, default=2,
                   help="How many of the 5 GroupKFold folds to actually train (precision vs time).")
    args = p.parse_args()

    print("=" * 78)
    print(f"T20 FEATURE ABLATION (Phase 2, no ship) -- squad_elo / competition / era / toss")
    print("=" * 78, flush=True)

    print("\n  building training frame (cache-aware)...", flush=True)
    frame = build_frame(FMT)
    print(f"  {len(frame):,} deliveries, {frame['match_id'].nunique():,} matches", flush=True)

    print("\n  [attach] batting_squad_elo / bowling_squad_elo (named XI, strictly-prior)")
    frame = attach_squad_elo(frame, FMT)

    print("  [attach] comp_code / era_year / toss flags")
    frame = attach_context_features(frame, FMT)

    print(f"\n  [ablation] retraining the legal head out-of-fold "
          f"(control / +squad_elo / +all)")
    out = {"format": FMT}
    out["ablation"] = run_ablation(frame, args.ablation_matches, args.ablation_folds)

    locked = {}
    if BASELINE_PATH.exists():
        locked = json.loads(BASELINE_PATH.read_text()).get(FMT, {})
    out["ablation"]["_locked_full_corpus_reference"] = {
        "legal_log_loss": locked.get("legal_log_loss"),
        "note": "context only -- not comparable to the subsampled control above",
    }

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    for variant in ("+squad_elo", "+all"):
        mp = out["ablation"][variant]["match_paired"]
        print(f"  {variant:12s} vs control  : {out['ablation'][variant]['delta_vs_control']:+.6f} "
              f"log-loss  (match-paired t={mp['t']}, n={mp['n_matches']}, "
              f"{mp['share_worse']:.0%} of matches worse)")
    mpi = out["ablation"]["+all"]["match_paired_vs_squad_elo"]
    print(f"  +all         vs +squad_elo : {out['ablation']['+all']['delta_vs_squad_elo']:+.6f} "
          f"log-loss  (match-paired t={mpi['t']}, n={mpi['n_matches']})")
    if locked.get("legal_log_loss") is not None:
        print(f"  [context] locked full-corpus legal log-loss: {locked['legal_log_loss']:.6f}; "
              f"sim_baseline_metrics.json gate tolerance +0.01")

    path = OUT_DIR / "t20_feature_ablation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
