"""
Brier / log-loss comparison of the ORIGINAL Elo engine vs the NEW ICC-style + bridge ratings,
on pre-match win prediction.

SCOPE NOTE -- read before quoting these numbers. This is NOT the production Brier in
data/processed/baseline_metrics.json. That one is ball-by-ball SECOND-INNINGS win probability
from the full XGBoost model, which consumes ~40 features of which team/player rating is only
part. This script measures something narrower and cleaner: given ONLY the two sides' ratings
before a ball is bowled, how well can you predict the winner? That isolates the predictive
content of the rating systems themselves, which is exactly what is being compared, and needs
no retrain.

Method
  - Predict P(team_1 wins) from the pre-match rating difference via logistic regression.
  - The logistic is fitted OUT-OF-SAMPLE (K-fold over matches), so the calibration mapping
    cannot leak. Ratings are already pre-match by construction in both systems.
  - Same matches, same folds, same everything for both systems -- only the rating differs.
  - A no-skill baseline (always predict the base rate) is reported so the numbers are readable.

Scopes
  international -- old team Elo          vs ICC-style team rating
  franchise     -- old squad-mean Elo    vs bridge squad-mean rating
                   (franchise teams have no old-Elo *team* rating that means anything across
                   leagues, so both sides are composed from player ratings for a fair test)

Usage:
    py scripts/compare_prematch_brier.py
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.match_metadata import MatchMetadataExtractor
from src.data_processing.format_spec import spec, INTL_COMPETITIONS, franchise_competitions
from src.features.icc_player_rating_engine import ICCPlayerRatingEngine, AsOfRatingIndex


def evaluate(diffs: np.ndarray, y: np.ndarray, seed: int = 42, folds: int = 5) -> dict:
    """Out-of-fold Brier/log-loss/accuracy for P(win) predicted from a single rating difference."""
    X = diffs.reshape(-1, 1)
    oof = np.zeros(len(y), dtype=float)
    for tr, te in KFold(n_splits=folds, shuffle=True, random_state=seed).split(X):
        model = LogisticRegression(max_iter=1000)
        model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
    return {
        "brier": brier_score_loss(y, oof),
        "log_loss": log_loss(y, oof),
        "accuracy": accuracy_score(y, (oof >= 0.5).astype(int)),
    }


def squad_mean(players_str, index: AsOfRatingIndex, as_of, role: str):
    names = [n.strip() for n in str(players_str).split(",") if n.strip()]
    vals = [index.effective_rating(p, role, as_of) for p in names
            if index.role_innings_before(p, role, as_of) > 0]
    return float(np.mean(vals)) if vals else np.nan


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--format", default="t20")
    args = p.parse_args()
    fmt = args.format

    meta = MatchMetadataExtractor(str(PROJECT_ROOT / "data" / "raw"), format_type=fmt).extract_all()
    meta = meta[meta["gender"] == spec(fmt)["gender"]].copy()
    meta["date"] = pd.to_datetime(meta["date"])

    proc = PROJECT_ROOT / "data" / "processed"
    old_elo_teams = pd.read_parquet(proc / f"{fmt}_elo_ratings.parquet")
    icc_teams = pd.read_parquet(proc / f"{fmt}_icc_team_ratings.parquet")
    old_players = pd.read_parquet(proc / f"{fmt}_player_elo_ratings.parquet")
    bridge = pd.read_parquet(proc / f"{fmt}_franchise_bridge_tiera.parquet")

    results = {}

    # ---------------------------------------------------------------- international
    intl = meta[meta["competition"].isin(INTL_COMPETITIONS.get(fmt, ()))
                & (meta["team_type"] == "international")]
    old_m = old_elo_teams[["match_id", "team_1", "team_2", "team_1_elo", "team_2_elo"]]
    new_m = icc_teams[["match_id", "team_1_elo", "team_2_elo"]].rename(
        columns={"team_1_elo": "n1", "team_2_elo": "n2"})
    df = (intl[["match_id", "team_1", "team_2", "winner"]]
          .merge(old_m[["match_id", "team_1_elo", "team_2_elo"]], on="match_id")
          .merge(new_m, on="match_id"))
    df = df[df["winner"].isin(df["team_1"]) | df["winner"].isin(df["team_2"])]
    df = df[(df["winner"] == df["team_1"]) | (df["winner"] == df["team_2"])]
    y = (df["winner"] == df["team_1"]).astype(int).to_numpy()
    results["international"] = {
        "n": len(df),
        "base_rate": float(y.mean()),
        "old": evaluate((df["team_1_elo"] - df["team_2_elo"]).to_numpy(), y),
        "new": evaluate((df["n1"] - df["n2"]).to_numpy(), y),
    }

    # ---------------------------------------------------------------- franchise (Tier A)
    fr = meta[meta["competition"].isin(franchise_competitions(fmt, ("A",)))].copy()
    fr = fr[(fr["winner"] == fr["team_1"]) | (fr["winner"] == fr["team_2"])]
    old_idx = AsOfRatingIndex(old_players.merge(
        meta[["match_id", "date"]].rename(columns={"date": "match_date"}), on="match_id", how="left")
        .assign(points_earned=lambda d: d["effective_elo"]))
    new_idx = AsOfRatingIndex(bridge)

    rows = []
    for r in fr.itertuples(index=False):
        o1 = squad_mean(r.team_1_players, old_idx, r.date, "batter")
        o2 = squad_mean(r.team_2_players, old_idx, r.date, "batter")
        n1 = squad_mean(r.team_1_players, new_idx, r.date, "batter")
        n2 = squad_mean(r.team_2_players, new_idx, r.date, "batter")
        if np.isnan([o1, o2, n1, n2]).any():
            continue
        rows.append({"old_diff": o1 - o2, "new_diff": n1 - n2,
                     "y": int(r.winner == r.team_1)})
    fdf = pd.DataFrame(rows)
    if len(fdf) > 100:
        yf = fdf["y"].to_numpy()
        results["franchise (Tier A)"] = {
            "n": len(fdf),
            "base_rate": float(yf.mean()),
            "old": evaluate(fdf["old_diff"].to_numpy(), yf),
            "new": evaluate(fdf["new_diff"].to_numpy(), yf),
        }

    print("\nPRE-MATCH WIN PREDICTION from ratings alone (out-of-fold, 5-fold)")
    print("NOT comparable to baseline_metrics.json -- see this script's docstring.\n")
    for scope, r in results.items():
        no_skill = r["base_rate"] * (1 - r["base_rate"])
        print(f"{scope}  (n={r['n']:,}, team_1 win rate {r['base_rate']:.3f}, "
              f"no-skill Brier {no_skill:.4f})")
        print(f"  {'system':<12}{'Brier':>10}{'log-loss':>11}{'accuracy':>10}")
        for k in ("old", "new"):
            m = r[k]
            label = "OLD Elo" if k == "old" else "NEW"
            print(f"  {label:<12}{m['brier']:>10.4f}{m['log_loss']:>11.4f}{m['accuracy']:>10.3f}")
        d = r["old"]["brier"] - r["new"]["brier"]
        better = "NEW" if d > 0 else "OLD"
        print(f"  -> Brier improvement: {d:+.4f}  ({better} better)\n")


if __name__ == "__main__":
    main()
