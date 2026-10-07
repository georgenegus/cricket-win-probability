"""
Phase 0 of t20_mae_gate_plan.md -- decomposes t20's MC innings-total MAE shortfall (27.57 vs the
25.0 gate) into three mechanisms with completely different fixes and costs:

  1. WRONG POINT ESTIMATOR. MAE is minimised by the median, but generate_mc_calibration_metrics.py
     scores scores.mean(). Simulated innings totals are left-skewed (collapses create a long low
     tail), so mean != median. Cost to fix: one line.
  2. CROSS-MATCH MISCALIBRATION. If the simulator under- or over-differentiates matches, an OOF
     affine correction `a + b*sim_mean` reduces MAE with no model change.
  3. THE PER-BALL MODEL CANNOT SEE TEAM QUALITY. FEATURES has batter_elo/bowler_elo/
     matchup_elo_diff/non_striker_elo but no squad-level aggregate -- each delivery is priced
     knowing only the two players on strike, never whether this is a strong XI against a weak
     attack. Cost to fix: a retrain (Phase 2/3).

Ships nothing -- pure measurement. Decomposes the gap between the simulator's current MAE and a
direct leakage-safe regressor's OOF MAE (the same "ceiling" measured while scoping this plan) into
how much each of (1) and (2) recovers, and how much is left over for (3).

LEAKAGE DISCIPLINE (a trap already hit once while scoping this plan): squad-quality features must
be derived from the NAMED XI with a rating computed STRICTLY BEFORE this match -- never from who
actually batted/bowled, which encodes the outcome (a side that scores heavily loses fewer wickets
and uses fewer batters; naive "participants" aggregates silently leak this). This script uses
`team_1_batting_squad_elo`/`team_2_bowling_squad_elo` straight from
data/processed/t20_player_elo_ratings.parquet (via get_match_squad_elo) -- the SAME pre-match,
position/rank-weighted, named-XI squad Elo the win-probability model already trains on
(player_elo_engine.py), not a hand-rolled aggregate. team_1 is always the innings-1 batting side
in this corpus (verified: 100% of rows), so team_1_batting_squad_elo / team_2_bowling_squad_elo is
exactly the batting/bowling pair for the innings this script forecasts.

Usage:
    py scripts/measure_t20_mae_decomposition.py
    py scripts/measure_t20_mae_decomposition.py --matches 100 --n-sims 3000
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

FMT = "t20"
GATE = 25.0
QUANTILE_GRID = np.round(np.arange(0.40, 0.601, 0.01), 3)
KFOLDS = 5
SEED = 42


# ----------------------------------------------------------------------------------------
# match population -- byte-identical selection to generate_mc_calibration_metrics.py /
# score_feedback_shrink_sweep.py, so "the calibration sample" means the same 100 matches
# everywhere this repo quotes it.
# ----------------------------------------------------------------------------------------

def _full_length_population() -> pd.DataFrame:
    fmt_spec = spec(FMT)
    total_balls, bpo = fmt_spec["total_balls"], fmt_spec["balls_per_over"]
    meta = pd.read_parquet(PROJECT_ROOT / f"data/processed/{FMT}_match_metadata.parquet")
    if "gender" in meta.columns:
        meta = meta[meta["gender"] == spec(FMT)["gender"]].copy()
    d = pd.read_parquet(PROJECT_ROOT / f"data/processed/{FMT}_cricket_data.parquet",
                        columns=["match_id", "innings", "total_runs", "wicket", "is_wide", "is_noball"])
    # legal_balls counts only is_wide==0, matching generate_mc_calibration_metrics.py's
    # population EXACTLY (no-balls are NOT excluded there, unlike a cricket-rules-correct
    # "legal delivery"). This must stay byte-identical to that script's own selection or the
    # "same 100 matches" comparisons this script exists to make become silently invalid --
    # already caught once (2026-08-11): excluding no-balls too shifted the full-length
    # boundary for a handful of matches, changing the population by one match (10,013 vs
    # 10,014) and cascading into 23/100 different linspace-sampled matches.
    agg = d[d["innings"] == 1].groupby("match_id").agg(
        actual_total=("total_runs", "sum"),
        legal_balls=("is_wide", lambda x: (x == 0).sum()),
        wkts=("wicket", "sum"))
    full = agg[(agg["legal_balls"] >= total_balls - bpo) | (agg["wkts"] == 10)].reset_index()
    return meta.merge(full, on="match_id").sort_values("match_id").reset_index(drop=True)


def _select_matches(n_matches: int) -> pd.DataFrame:
    merged = _full_length_population()
    idx = np.linspace(0, len(merged) - 1, min(n_matches, len(merged)), dtype=int)
    return merged.iloc[idx].reset_index(drop=True)


# ----------------------------------------------------------------------------------------
# 1. The "ceiling" -- a direct, leakage-safe match-level regressor, OOF over the WHOLE
#    full-length population (not just the 100-match sample), so its OOF predictions for the
#    100 sampled matches are genuinely out-of-sample and comparable to the simulator 1:1.
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

    # Named-XI, strictly-prior squad Elo -- straight off player_elo_ratings.parquet (see module
    # docstring). team_1 is always the innings-1 batting side in this corpus.
    sq = []
    for mid in m["match_id"]:
        vals = get_match_squad_elo(FMT, mid, processed_dir=str(PROJECT_ROOT / "data/processed"))
        sq.append(vals if vals is not None else (np.nan,) * 4)
    sq_df = pd.DataFrame(sq, columns=["t1_bat_sq", "t1_bowl_sq", "t2_bat_sq", "t2_bowl_sq"])
    m = pd.concat([m.reset_index(drop=True), sq_df], axis=1)
    m["batting_squad_elo"] = m["t1_bat_sq"]     # innings-1 batting side
    m["bowling_squad_elo"] = m["t2_bowl_sq"]    # innings-1 bowling side

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


def fit_direct_regressor_oof(population: pd.DataFrame) -> pd.DataFrame:
    """OOF (actual, pred) for every match in `population`, from the leakage-safe feature set."""
    frame = _build_direct_regressor_frame(population)
    cols = [c for c in DIRECT_FEATURES if c in frame.columns]
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
    print(f"  direct regressor: n={len(y):,}  OOF R^2={r2:.1%}  OOF MAE={mae:.2f}", flush=True)

    out = frame[["match_id"]].copy()
    out["direct_oof_pred"] = oof
    out["direct_oof_r2_global"] = r2
    return out


# ----------------------------------------------------------------------------------------
# 2. Simulate the 100-match calibration sample at full fidelity, keeping full per-match
#    score arrays (needed for the quantile sweep -- no need to re-simulate per quantile).
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
            "sim_mean": float(scores.mean()), "sim_median": float(np.median(scores)),
            "sim_sd": float(scores.std(ddof=1)),
            "quantiles": {float(q): float(np.quantile(scores, q)) for q in QUANTILE_GRID},
        })
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(sample)} matches ({elapsed/60:.1f}m elapsed, "
                  f"~{elapsed/(i+1)*(len(sample)-i-1)/60:.1f}m remaining)", flush=True)
    return rows


# ----------------------------------------------------------------------------------------
# 3/4/5. Estimator sweep, OOF recalibration, headroom decomposition, dispersion sanity check
# ----------------------------------------------------------------------------------------

def estimator_sweep(df: pd.DataFrame) -> dict:
    mae_mean = float((df["sim_mean"] - df["actual"]).abs().mean())
    mae_median = float((df["sim_median"] - df["actual"]).abs().mean())
    q_maes = {}
    for q in QUANTILE_GRID:
        pred = df["quantiles"].apply(lambda d: d[float(q)])
        q_maes[float(q)] = float((pred - df["actual"]).abs().mean())
    best_q = min(q_maes, key=q_maes.get)
    return {"mae_mean": mae_mean, "mae_median": mae_median,
            "quantile_maes": q_maes, "best_quantile": best_q, "mae_best_quantile": q_maes[best_q]}


def oof_recalibration(df: pd.DataFrame, base_col: str) -> dict:
    """OOF-fit actual ~ a + b*base_col, KFold by match. Returns slope stats + OOF MAE."""
    x = df[base_col].astype(float).values
    y = df["actual"].astype(float).values
    n = len(y)
    oof_pred = np.full(n, np.nan)
    slopes = []
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)
    for tr, va in kf.split(x):
        A = np.vstack([x[tr], np.ones(len(tr))]).T
        b, a = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        oof_pred[va] = a + b * x[va]
        slopes.append(b)
    slopes = np.array(slopes)
    mae_recal = float(np.abs(oof_pred - y).mean())
    # Full-data fit for reporting (not for scoring -- oof_pred/mae_recal already are OOF).
    A_full = np.vstack([x, np.ones(n)]).T
    b_full, a_full = np.linalg.lstsq(A_full, y, rcond=None)[0]
    return {
        "slope_b": float(b_full), "intercept_a": float(a_full),
        "slope_b_oof_mean": float(slopes.mean()), "slope_b_oof_se": float(slopes.std(ddof=1) / np.sqrt(KFOLDS)),
        "oof_mae_after_recalibration": mae_recal,
    }


def dispersion_check(df: pd.DataFrame) -> dict:
    avg_sim_sd = float(df["sim_sd"].mean())
    realized_error_sd = float((df["sim_mean"] - df["actual"]).std(ddof=1))
    return {"avg_sim_sd": avg_sim_sd, "realized_error_sd": realized_error_sd,
            "ratio": avg_sim_sd / realized_error_sd if realized_error_sd else float("nan")}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--matches", type=int, default=100)
    p.add_argument("--n-sims", type=int, default=3000)
    args = p.parse_args()

    print("=" * 78)
    print(f"T20 MAE DECOMPOSITION -- {args.matches} matches, n_sims={args.n_sims}")
    print("=" * 78)

    print("\n[setup] fitting the leakage-safe direct regressor over the FULL population...", flush=True)
    population = _full_length_population()
    direct_oof = fit_direct_regressor_oof(population)

    print(f"\n[setup] selecting the {args.matches}-match calibration sample...", flush=True)
    sample = _select_matches(args.matches)
    sample["match_id"] = sample["match_id"].astype(str)

    print(f"\n[simulate] {args.n_sims} sims/match, this is the slow part...", flush=True)
    rows = simulate_sample(sample, args.n_sims)
    df = pd.DataFrame(rows)
    df = df.merge(direct_oof, on="match_id", how="left")
    n_missing_direct = int(df["direct_oof_pred"].isna().sum())
    if n_missing_direct:
        print(f"  [warn] {n_missing_direct} matches missing a direct-regressor OOF prediction "
              f"(dropped from headroom decomposition)")

    # Persist per-match rows (flatten quantiles to a few named columns for the CSV).
    persist_cols = {"match_id": df["match_id"], "actual": df["actual"], "sim_mean": df["sim_mean"],
                    "sim_median": df["sim_median"], "sim_sd": df["sim_sd"],
                    "direct_oof_pred": df["direct_oof_pred"]}
    for q in (0.40, 0.45, 0.50, 0.55, 0.60):
        persist_cols[f"sim_q{int(q*100)}"] = df["quantiles"].apply(lambda d: d[q])
    out_csv = PROJECT_ROOT / "outputs" / "t20_mae_decomposition_rows.csv"
    pd.DataFrame(persist_cols).to_csv(out_csv, index=False)
    print(f"\n[persist] per-match rows -> {out_csv}")

    print("\n" + "=" * 78)
    print("[Q1/2] ESTIMATOR SWEEP")
    print("=" * 78)
    est = estimator_sweep(df)
    print(f"  MAE(mean, today's convention)   = {est['mae_mean']:.2f}")
    print(f"  MAE(median)                     = {est['mae_median']:.2f}")
    print(f"  {'quantile':>10s} {'MAE':>8s}")
    for q, v in est["quantile_maes"].items():
        flag = "  <-- best" if q == est["best_quantile"] else ""
        print(f"  {q:>10.2f} {v:>8.2f}{flag}")

    print("\n" + "=" * 78)
    print("[Q3] OOF RECALIBRATION (actual ~ a + b*sim_mean)")
    print("=" * 78)
    recal = oof_recalibration(df, "sim_mean")
    print(f"  slope b = {recal['slope_b']:.3f}  (OOF mean {recal['slope_b_oof_mean']:.3f} "
          f"+/- {recal['slope_b_oof_se']:.3f} SE)")
    print(f"  intercept a = {recal['intercept_a']:.2f}")
    print(f"  interpretation: b<1 => simulator OVER-differentiates matches (regress toward mean "
          f"helps); b>1 => UNDER-differentiates (needs amplifying); b~1 => already well-scaled")
    print(f"  OOF MAE after recalibration = {recal['oof_mae_after_recalibration']:.2f}")

    print("\n" + "=" * 78)
    print("[Q4] HEADROOM ATTRIBUTION")
    print("=" * 78)
    df_ok = df.dropna(subset=["direct_oof_pred"])
    mae_direct = float((df_ok["direct_oof_pred"] - df_ok["actual"]).abs().mean())
    mae_today = est["mae_mean"]
    mae_best_est = min(est["mae_median"], est["mae_best_quantile"])
    mae_recal = recal["oof_mae_after_recalibration"]
    gap_total = mae_today - mae_direct
    gap_estimator = mae_today - mae_best_est
    gap_recal_only = mae_today - mae_recal
    print(f"  simulator today (mean)                : {mae_today:.2f}")
    print(f"  simulator, best estimator              : {mae_best_est:.2f}  "
          f"(recovers {gap_estimator:+.2f} of the gap)")
    print(f"  simulator, OOF recalibration            : {mae_recal:.2f}  "
          f"(recovers {gap_recal_only:+.2f} of the gap)")
    print(f"  direct regressor ceiling (same matches) : {mae_direct:.2f}")
    print(f"  TOTAL GAP (today - ceiling)             : {gap_total:.2f}")
    residual = mae_today - max(gap_estimator, gap_recal_only) - mae_direct
    print(f"  residual after best zero-retrain fix    : ~{gap_total - max(gap_estimator, gap_recal_only):.2f} "
          f"runs -- this is what Phase 2/3 (squad-context features) would need to close")
    print(f"  n matches in this comparison            : {len(df_ok)}")

    print("\n" + "=" * 78)
    print("[Q5] DISPERSION SANITY CHECK")
    print("=" * 78)
    disp = dispersion_check(df)
    print(f"  avg simulated std dev (per-match)  = {disp['avg_sim_sd']:.2f}")
    print(f"  realised error SD (sim_mean-actual)= {disp['realized_error_sd']:.2f}")
    print(f"  ratio                              = {disp['ratio']:.2f}  "
          f"({'UNDER-dispersed' if disp['ratio']<0.9 else 'OVER-dispersed' if disp['ratio']>1.1 else 'roughly matched'})")

    summary = {
        "format": FMT, "n_matches": len(df), "n_sims": args.n_sims,
        "gate": GATE,
        "estimator_sweep": est,
        "recalibration": recal,
        "headroom": {
            "mae_today": mae_today, "mae_best_estimator": mae_best_est,
            "mae_after_recalibration": mae_recal, "mae_direct_regressor_ceiling": mae_direct,
            "total_gap": gap_total, "n_matches_compared": len(df_ok),
        },
        "dispersion": disp,
    }
    out_path = PROJECT_ROOT / "outputs" / "t20_mae_decomposition_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
