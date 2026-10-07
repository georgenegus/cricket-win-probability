"""
Phase 3 of markdowns/win_predictor/franchise_bridge_plan.md -- THE VALIDATION GATE.

The bridge claims to carry the international rating scale into franchise cricket. That claim
is testable, because anchors play franchise cricket too: compute an anchor's rating purely
from their FRANCHISE performances (via the bridge) and compare it against their independently
known INTERNATIONAL rating. If the bridge transfers the scale faithfully those should agree;
if they don't, Tier B/C multi-hop must not be attempted.

Reports:
  1. GATE      -- Spearman/Pearson between bridge-derived and international ratings, for
                  players qualifying in both systems, per role.
  2. DENSITY   -- the same correlation bucketed by how well-anchored that player's franchise
                  matches were (does it degrade where anchors are thin?).
  3. BIAS      -- mean signed gap between bridge and international rating (are anchors, who
                  are by construction the best players in any franchise XI, systematically
                  over/under-rated by the bridge?).
  4. COVERAGE  -- anchored share of franchise matches by year (early T20 seasons predate a
                  meaningful pool of 20-cap T20I players, so the bridge cannot anchor them).
  5. FACE      -- top bridge-rated players overall and among franchise-only (non-anchor)
                  players, for eyeball sanity.

Usage:
    py scripts/measure_franchise_bridge_validation.py
    py scripts/measure_franchise_bridge_validation.py --format t20 --tiers A
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.load_data import CricketDataLoader
from src.data_processing.match_metadata import MatchMetadataExtractor
from src.data_processing.format_spec import spec, FRANCHISE_BRIDGE_TIERS
from src.features.icc_player_rating_engine import ICCPlayerRatingEngine
from src.features.franchise_bridge_engine import FranchiseBridgeEngine


def build_bridge(format_type: str, tiers: tuple, force: bool):
    fmt_spec = spec(format_type)
    raw = str(PROJECT_ROOT / "data" / "raw")
    deliveries = CricketDataLoader(raw, format_type=format_type).load_all()
    meta = MatchMetadataExtractor(raw, format_type=format_type).extract_all()
    meta = meta[meta["gender"] == fmt_spec["gender"]].copy()
    deliveries = deliveries[deliveries["match_id"].isin(meta["match_id"])].copy()

    proc = PROJECT_ROOT / "data" / "processed"
    icc_elo_df = pd.read_parquet(proc / f"{format_type}_icc_team_ratings.parquet")
    icc_player_df = pd.read_parquet(proc / f"{format_type}_icc_player_ratings.parquet")

    engine = FranchiseBridgeEngine(format_type=format_type, tiers=tiers)
    bridge_df = engine.run(deliveries, meta, icc_elo_df, icc_player_df, force=force)
    return bridge_df, icc_player_df, meta


def gate(bridge_df, icc_player_df, as_of):
    """Bridge-derived vs true international rating, for players qualifying in both."""
    out = {}
    for role in ("batter", "bowler"):
        bridge = FranchiseBridgeEngine.get_effective_ratings_from_df(bridge_df, role, as_of_date=as_of)
        intl = ICCPlayerRatingEngine.get_effective_ratings_from_df(icc_player_df, role, as_of_date=as_of)
        common = sorted(set(bridge) & set(intl))
        if len(common) < 10:
            out[role] = {"n": len(common), "note": "too few players in both systems"}
            continue
        b = np.array([bridge[p] for p in common])
        i = np.array([intl[p] for p in common])
        rho, p_s = spearmanr(b, i)
        r, p_p = pearsonr(b, i)
        out[role] = {
            "n": len(common),
            "spearman_r": round(float(rho), 3), "spearman_p": float(f"{p_s:.2e}"),
            "pearson_r": round(float(r), 3), "pearson_p": float(f"{p_p:.2e}"),
            "mean_bridge": round(float(b.mean()), 2),
            "mean_intl": round(float(i.mean()), 2),
            "mean_signed_gap": round(float((b - i).mean()), 2),
            "players": common,
            "bridge_vals": {p: round(bridge[p], 2) for p in common},
            "intl_vals": {p: round(intl[p], 2) for p in common},
        }
    return out


def density_buckets(bridge_df, icc_player_df, as_of):
    """Does agreement degrade where the player's franchise matches were thinly anchored?"""
    out = {}
    for role in ("batter", "bowler"):
        role_df = bridge_df[bridge_df["role"] == role]
        mean_anchors = role_df.groupby("player")["opponent_anchors"].mean()

        bridge = FranchiseBridgeEngine.get_effective_ratings_from_df(bridge_df, role, as_of_date=as_of)
        intl = ICCPlayerRatingEngine.get_effective_ratings_from_df(icc_player_df, role, as_of_date=as_of)
        common = [p for p in set(bridge) & set(intl) if p in mean_anchors.index]
        if len(common) < 20:
            out[role] = {"note": "too few players"}
            continue

        rows = []
        for lo, hi, label in ((0, 2, "0-2 anchors"), (2, 4, "2-4"), (4, 6, "4-6"), (6, 99, "6+")):
            sel = [p for p in common if lo <= mean_anchors[p] < hi]
            if len(sel) < 10:
                rows.append({"bucket": label, "n": len(sel), "spearman_r": None})
                continue
            b = np.array([bridge[p] for p in sel])
            i = np.array([intl[p] for p in sel])
            rho, _ = spearmanr(b, i)
            rows.append({"bucket": label, "n": len(sel), "spearman_r": round(float(rho), 3)})
        out[role] = rows
    return out


def coverage_by_year(bridge_df):
    df = bridge_df.copy()
    df["year"] = pd.to_datetime(df["match_date"]).dt.year
    per_year = df.groupby("year").agg(
        innings=("player", "size"),
        mean_opp_anchors=("opponent_anchors", "mean"),
        pct_unanchored=("opponent_anchors", lambda s: round((s == 0).mean() * 100, 1)),
    ).reset_index()
    return per_year


def face_validity(bridge_df, as_of, top_n=15):
    out = {}
    anchor_flag = bridge_df.groupby("player")["is_anchor"].max().to_dict()
    for role in ("batter", "bowler"):
        ratings = FranchiseBridgeEngine.get_effective_ratings_from_df(bridge_df, role, as_of_date=as_of)
        ordered = sorted(ratings, key=ratings.get, reverse=True)
        out[role] = {
            "top_overall": [(p, round(ratings[p], 1), bool(anchor_flag.get(p, False))) for p in ordered[:top_n]],
            "top_franchise_only": [
                (p, round(ratings[p], 1)) for p in ordered if not anchor_flag.get(p, False)
            ][:top_n],
            "n_rated": len(ratings),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", default="t20")
    parser.add_argument("--tiers", nargs="+", default=["A"], choices=["A", "B", "C"])
    parser.add_argument("--force", action="store_true", help="Recompute the bridge, ignoring cache.")
    args = parser.parse_args()

    bridge_df, icc_player_df, meta = build_bridge(args.format, tuple(args.tiers), args.force)
    as_of = pd.to_datetime(bridge_df["match_date"]).max() + pd.Timedelta(days=1)
    print(f"\nas-of date: {as_of.date()}   bridge innings: {len(bridge_df)}   "
          f"players: {bridge_df['player'].nunique()}")

    print("\n" + "=" * 72)
    print("1. GATE -- bridge-derived vs true international rating")
    print("=" * 72)
    g = gate(bridge_df, icc_player_df, as_of)
    for role, d in g.items():
        if "note" in d:
            print(f"{role}: n={d['n']} -- {d['note']}")
            continue
        print(f"{role}: n={d['n']} common players")
        print(f"   Spearman r = {d['spearman_r']} (p={d['spearman_p']:.1e})   "
              f"Pearson r = {d['pearson_r']} (p={d['pearson_p']:.1e})")
        print(f"   mean bridge={d['mean_bridge']}  mean intl={d['mean_intl']}  "
              f"signed gap={d['mean_signed_gap']}")

    print("\n" + "=" * 72)
    print("2. DENSITY -- agreement bucketed by anchors backing the player's matches")
    print("=" * 72)
    for role, rows in density_buckets(bridge_df, icc_player_df, as_of).items():
        if isinstance(rows, dict):
            print(f"{role}: {rows.get('note')}")
            continue
        print(f"{role}:")
        for r in rows:
            print(f"   {r['bucket']:<12} n={r['n']:<5} spearman={r['spearman_r']}")

    print("\n" + "=" * 72)
    print("3. COVERAGE -- anchoring by year")
    print("=" * 72)
    cov = coverage_by_year(bridge_df)
    print(cov.to_string(index=False))

    print("\n" + "=" * 72)
    print("4. FACE VALIDITY")
    print("=" * 72)
    fv = face_validity(bridge_df, as_of)
    for role, d in fv.items():
        print(f"\n{role} (n_rated={d['n_rated']}) -- top overall [(name, rating, is_anchor)]:")
        for p, v, a in d["top_overall"]:
            print(f"   {p:<24} {v:>7}  {'anchor' if a else 'franchise-only'}")
        print(f"  top FRANCHISE-ONLY {role}s:")
        for p, v in d["top_franchise_only"][:10]:
            print(f"   {p:<24} {v:>7}")

    out_path = PROJECT_ROOT / "data" / "processed" / f"{args.format}_franchise_bridge_validation.json"
    payload = {
        "as_of": str(as_of.date()),
        "tiers": args.tiers,
        "gate": {r: {k: v for k, v in d.items() if k not in ("players", "bridge_vals", "intl_vals")}
                 for r, d in g.items()},
        "coverage_by_year": cov.to_dict(orient="records"),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
