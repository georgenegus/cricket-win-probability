"""
hnd counterpart to scripts/measure_t20_mae_decomposition.py, scoped to what's actually needed for
a re-anchor decision on hnd's MAE gate (see t20_mae_gate_plan.md Phase 4's precedent and
hnd_delivery_outcome_pooling_plan.md for the pooling fix that got hnd's bias/coverage/win-prob/
O-U-Brier gates passing but left MAE at 23.06 against a 20.7 gate).

Measures, all leakage-safe / OOF:
  1. Zero-skill floor -- MAE if you always predict the population median.
  2. The honest ceiling -- a direct, leakage-safe match-level regressor (5-fold OOF) on every
     pre-match feature this repo holds for hnd (venue par, team Elo, named-XI squad Elo,
     competition, era, toss), over the FULL 157-match full-length population (not a subsample --
     hnd's whole corpus is smaller than the "100-match sample" t20/od draw from, so there's no
     larger pool to hold back).
  3. Today's simulator, re-measured on the SAME 100-match sample generate_mc_calibration_
     metrics.py uses (byte-identical selection logic, copied from that script -- a mismatch here
     already cost real trust once on this repo's t20 plan), WITH per-match rows persisted so the
     sampling-noise SE that the "ceiling + noise" re-anchor formula needs can actually be computed
     for hnd's own error distribution, not assumed to match t20's.

Ships nothing -- pure measurement.

Run:
    py scripts/measure_hnd_mae_ceiling.py
"""

import os
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
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from src.data_processing.format_spec import spec
from src.models.delivery_outcome_model import DeliveryOutcomeModel
from src.simulation.innings_simulator import simulate_innings
from src.simulation.match_simulator import extract_match_squads
from src.features.player_elo_engine import get_match_squad_elo

FMT = "hnd"
KFOLDS = 5
SEED = 42


# ----------------------------------------------------------------------------------------
# population -- byte-identical to generate_mc_calibration_metrics.py's own selection (same
# "male" gender filter it uses, same full-length definition, same linspace sampling), so the
# "same matches" comparisons below are actually valid.
# ----------------------------------------------------------------------------------------

def _full_length_population() -> pd.DataFrame:
    fmt_spec = spec(FMT)
    total_balls, bpo = fmt_spec["total_balls"], fmt_spec["balls_per_over"]
    meta = pd.read_parquet(PROJECT_ROOT / f"data/processed/{FMT}_match_metadata.parquet")
    if "gender" in meta.columns:
        meta = meta[meta["gender"] == "male"]
    d = pd.read_parquet(PROJECT_ROOT / f"data/processed/{FMT}_cricket_data.parquet",
                        columns=["match_id", "innings", "total_runs", "wicket", "is_wide"])
    if "gender" in meta.columns:
        d = d[d["match_id"].isin(meta["match_id"])]
    agg = d[d["innings"] == 1].groupby("match_id").agg(
        actual_total=("total_runs", "sum"),
        legal_balls=("is_wide", lambda x: (x == 0).sum()),
        wkts=("wicket", "sum"))
    full = agg[(agg["legal_balls"] >= total_balls - bpo) | (agg["wkts"] == 10)].reset_index()
    return meta.merge(full, on="match_id").sort_values("match_id").reset_index(drop=True)


def _select_matches(population: pd.DataFrame, n_matches: int) -> pd.DataFrame:
    idx = np.linspace(0, len(population) - 1, min(n_matches, len(population)), dtype=int)
    return population.iloc[idx].reset_index(drop=True)


# ----------------------------------------------------------------------------------------
# 1. direct regressor ceiling, over the FULL population
# ----------------------------------------------------------------------------------------

def _build_direct_regressor_frame(population: pd.DataFrame) -> pd.DataFrame:
    m = population.copy()
    m["match_id"] = m["match_id"].astype(str)

    venue = pd.read_parquet(PROJECT_ROOT / f"data/processed/{FMT}_venue_features.parquet")
    m = m.merge(venue[["match_id", "venue_avg_first_innings_score"]].drop_duplicates("match_id"),
               on="match_id", how="left")

    elo = pd.read_parquet(PROJECT_ROOT / f"data/processed/{FMT}_elo_ratings.parquet")
    elo_cols = [c for c in elo.columns if "elo" in c.lower()]
    m = m.merge(elo[["match_id"] + elo_cols].drop_duplicates("match_id"), on="match_id", how="left")

    sq = []
    for mid in m["match_id"]:
        vals = get_match_squad_elo(FMT, mid, processed_dir=str(PROJECT_ROOT / "data/processed"))
        sq.append(vals if vals is not None else (np.nan,) * 4)
    sq_df = pd.DataFrame(sq, columns=["t1_bat_sq", "t1_bowl_sq", "t2_bat_sq", "t2_bowl_sq"])
    m = pd.concat([m.reset_index(drop=True), sq_df], axis=1)
    m["batting_squad_elo"] = m["t1_bat_sq"]
    m["bowling_squad_elo"] = m["t2_bowl_sq"]

    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m["year"] = m["date"].dt.year.astype(float)
    m["comp_code"] = m["competition"].astype("category").cat.codes.astype(float)
    m["toss_won_by_batting_side"] = (m["toss_winner"] == m["team_1"]).astype(float)
    m["toss_decision_field"] = (m["toss_decision"].astype(str).str.lower() == "field").astype(float)
    return m


DIRECT_FEATURES = [
    "venue_avg_first_innings_score", "team_1_elo", "team_2_elo",
    "batting_squad_elo", "bowling_squad_elo",
    "comp_code", "year", "toss_won_by_batting_side", "toss_decision_field",
]


def fit_direct_regressor_oof(population: pd.DataFrame) -> dict:
    frame = _build_direct_regressor_frame(population)
    cols = [c for c in DIRECT_FEATURES if c in frame.columns and frame[c].notna().any()]
    frame = frame.dropna(subset=cols + ["actual_total"]).reset_index(drop=True)
    X = frame[cols].astype(float).values
    y = frame["actual_total"].astype(float).values

    oof = np.full(len(y), np.nan)
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)
    for tr, va in kf.split(X):
        model = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                             random_state=SEED, n_jobs=-1)
        model.fit(X[tr], y[tr])
        oof[va] = model.predict(X[va])

    r2 = 1 - ((y - oof) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    mae = float(np.abs(y - oof).mean())
    zero_skill_mae = float(np.abs(y - np.median(y)).mean())
    print(f"  direct regressor: n={len(y):,}  features used={cols}")
    print(f"  OOF R^2={r2:.1%}  OOF MAE={mae:.2f}  |  zero-skill (median) MAE={zero_skill_mae:.2f}")
    return {"n": len(y), "features_used": cols, "r2": r2, "mae": mae, "zero_skill_mae": zero_skill_mae,
            "match_ids": frame["match_id"].tolist()}


# ----------------------------------------------------------------------------------------
# 2. today's simulator, re-measured with per-match rows persisted for the noise SE
# ----------------------------------------------------------------------------------------

def simulate_sample(sample: pd.DataFrame, n_sims: int) -> list:
    fmt_spec = spec(FMT)
    model = DeliveryOutcomeModel(format_type=FMT)
    model.fit(None, cache_path=PROJECT_ROOT / f"data/processed/{FMT}_delivery_outcome_model.pkl")

    rows = []
    t0 = time.time()
    for i, row in sample.iterrows():
        mid = str(row["match_id"])
        try:
            t1, t2, venue_score = extract_match_squads(mid, format_type=FMT)
            res = simulate_innings(
                model=model, lineup=t1["lineup"], attack=t2["attack"], venue_avg_score=venue_score,
                n_sims=n_sims, total_overs=fmt_spec["total_overs"], balls_per_over=fmt_spec["balls_per_over"],
                format_type=FMT, innings=1, target=None,
                rng=np.random.default_rng(int(mid) % 100000),
            )
        except Exception as e:
            print(f"  [skip] {mid}: {e}")
            continue
        scores = res["final_score"].astype(np.float64)
        rows.append({
            "match_id": mid, "actual": float(row["actual_total"]),
            "sim_mean": float(scores.mean()),
        })
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(sample)} matches ({elapsed/60:.1f}m elapsed)", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--matches", type=int, default=100)
    p.add_argument("--n-sims", type=int, default=3000)
    args = p.parse_args()

    print("=" * 78)
    print(f"HND MAE CEILING -- {args.matches}-match sample, n_sims={args.n_sims}")
    print("=" * 78)

    population = _full_length_population()
    print(f"\n[population] {len(population)} full-length hnd matches")

    print("\n[ceiling] fitting the leakage-safe direct regressor over the FULL population...")
    ceiling = fit_direct_regressor_oof(population)

    print(f"\n[sample] selecting the {args.matches}-match calibration sample (same selection as generate_mc_calibration_metrics.py)...")
    sample = _select_matches(population, args.matches)
    sample["match_id"] = sample["match_id"].astype(str)

    print(f"\n[simulate] {args.n_sims} sims/match...")
    rows = simulate_sample(sample, args.n_sims)
    df = pd.DataFrame(rows)

    mae_today = float((df["sim_mean"] - df["actual"]).abs().mean())
    per_match_abs_err = (df["sim_mean"] - df["actual"]).abs()
    se = float(per_match_abs_err.std(ddof=1) / np.sqrt(len(df)))

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"  zero-skill floor (median)         : {ceiling['zero_skill_mae']:.2f}")
    print(f"  simulator today (this sample)     : {mae_today:.2f}  (SE={se:.2f}, n={len(df)} matches)")
    print(f"  simulator today (locked, official): 23.06  [context -- may differ slightly, different sim seed draw]")
    print(f"  ceiling (leakage-safe regressor)  : {ceiling['mae']:.2f}  (OOF R^2={ceiling['r2']:.1%}, n={ceiling['n']})")
    print(f"  gap (today - ceiling)             : {mae_today - ceiling['mae']:+.2f}")
    print(f"\n  candidate re-anchor bases:")
    print(f"    ceiling + 1 SE  = {ceiling['mae'] + se:.2f}")
    print(f"    ceiling + 2 SE  = {ceiling['mae'] + 2*se:.2f}")
    print(f"    current MAE_TARGET (t20-derived scaling) = 20.7")

    out = {
        "format": FMT, "n_sample_matches": len(df), "n_sims": args.n_sims,
        "zero_skill_mae": ceiling["zero_skill_mae"],
        "ceiling_mae": ceiling["mae"], "ceiling_r2": ceiling["r2"],
        "ceiling_n": ceiling["n"], "ceiling_features": ceiling["features_used"],
        "simulator_mae_this_sample": mae_today, "simulator_mae_se": se,
        "simulator_mae_locked_official": 23.06,
        "candidate_gate_ceiling_plus_1se": ceiling["mae"] + se,
        "candidate_gate_ceiling_plus_2se": ceiling["mae"] + 2 * se,
    }
    out_path = PROJECT_ROOT / "outputs" / "hnd_mae_ceiling.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
