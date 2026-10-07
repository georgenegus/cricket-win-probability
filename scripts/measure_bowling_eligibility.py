"""
Phase 0 of markdowns/win_predictor/frontline_bowler_eligibility_plan.md.

Measures, WITHOUT touching PlayerEloEngine, what a recent-bowling-rate eligibility gate would
have done to each format's bowler rank key -- reconstructed directly from the already-computed
*_player_elo_ratings.parquet + *_elo_ratings.parquet, via vectorized merge_asof, rather than
re-running the full chronological delivery-by-delivery Elo loop (nothing about raw-Elo
computation changes here; only which players count as eligible for the top-K bowling rank).

Reuses PlayerEloEngine's own static weighting methods and _team_to_player_scale so the
reconstructed "current" rank key matches production exactly -- see verify_reconstruction().

Usage:
    py scripts/measure_bowling_eligibility.py --formats t20 od hnd t20w odw hndw
    py scripts/measure_bowling_eligibility.py --formats od --sweep
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.player_elo_engine import PlayerEloEngine

MIN_MATCHES_RELIABLE = 10.0  # PlayerEloEngine default, never overridden anywhere in this repo
DEFAULT_WINDOW = 10
DEFAULT_MIN_RATE = 0.2


def _stretch_for(fmt):
    return {
        "t20": PlayerEloEngine.T20_ELO_STRETCH,
        "od": PlayerEloEngine.OD_ELO_STRETCH,
        "hnd": PlayerEloEngine.HND_ELO_STRETCH,
    }.get(fmt, 1.0)


def _parse_players(val):
    if isinstance(val, str):
        return [p.strip() for p in val.split(",") if p.strip()]
    if isinstance(val, (list, tuple, np.ndarray)):
        return [str(p).strip() for p in val if str(p).strip()]
    return []


def load_format(fmt, processed_dir="data/processed"):
    p_dir = Path(processed_dir)
    p_df = pd.read_parquet(p_dir / f"{fmt}_player_elo_ratings.parquet")
    e_df = pd.read_parquet(p_dir / f"{fmt}_elo_ratings.parquet")
    e_df["match_date"] = pd.to_datetime(e_df["match_date"], errors="coerce")
    e_df = e_df.dropna(subset=["match_date"]).sort_values("match_date")
    date_map = dict(zip(e_df["match_id"].astype(str), e_df["match_date"]))
    p_df["match_date"] = p_df["match_id"].astype(str).map(date_map)
    p_df = p_df.dropna(subset=["match_date"])
    return p_df, e_df


def build_query_frame(e_df):
    """One row per (player, team_key, match_id, match_date, team_elo) for every named player in
    every match -- the (player, match) pairs we need rank-key values for."""
    rows = []
    for r in e_df.itertuples():
        for team_key, players_col, elo_col in (
            ("team_1", r.team_1_players, r.team_1_elo),
            ("team_2", r.team_2_players, r.team_2_elo),
        ):
            for p in _parse_players(players_col):
                rows.append((p, team_key, r.match_id, r.match_date, elo_col))
    q = pd.DataFrame(rows, columns=["player", "team_key", "match_id", "match_date", "team_elo"])
    return q.sort_values(["player", "match_date"]).reset_index(drop=True)


def attach_bowling_history(q, p_df):
    """merge_asof (strictly-before) each query row against that player's own bowler-role rows
    to get: raw_elo as of just before this match, and n_prior = count of earlier bowler rows.
    Absent players get raw_elo=1500 (default_elo) and n_prior=0 -- never bowled yet."""
    bowl = p_df[p_df["role"] == "bowler"][["player", "match_date", "raw_elo"]].copy()
    bowl = bowl.sort_values(["player", "match_date"])
    bowl["n_prior_after"] = bowl.groupby("player").cumcount() + 1  # count incl. this row

    merged = pd.merge_asof(
        q.sort_values("match_date"), bowl.sort_values("match_date"),
        on="match_date", by="player", direction="backward", allow_exact_matches=False,
    ).sort_values(["player", "match_date"])
    merged["raw_elo"] = merged["raw_elo"].fillna(1500.0)
    merged["n_prior"] = merged["n_prior_after"].fillna(0).astype(int)
    return merged.drop(columns=["n_prior_after"])


def attach_recent_bowl_rate(q, p_df, window=DEFAULT_WINDOW):
    """Trailing-`window` bowling rate as of strictly before each query match: one row per
    (player, match) collapsing batter+bowler rows to a 'bowled this match' flag, a shifted
    rolling mean per player, then merge_asof onto the query frame."""
    app = (
        p_df.groupby(["player", "match_id", "match_date"])["role"]
        .apply(lambda s: float((s == "bowler").any()))
        .reset_index(name="bowled")
        .sort_values(["player", "match_date"])
    )
    app["recent_rate"] = (
        app.groupby("player")["bowled"]
        .apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    rates = app[["player", "match_date", "recent_rate"]].dropna(subset=["recent_rate"])
    merged = pd.merge_asof(
        q.sort_values("match_date"), rates.sort_values("match_date"),
        on="match_date", by="player", direction="backward", allow_exact_matches=False,
    ).sort_values(["player", "match_date"])
    merged["recent_rate"] = merged["recent_rate"].fillna(0.0)
    return merged


def compute_eff_elo(row, fmt):
    reliability = row["n_prior"] / (row["n_prior"] + MIN_MATCHES_RELIABLE)
    team_elo_scaled = 1500.0 + (row["team_elo"] - 1500.0) * _stretch_for(fmt)
    return reliability * row["raw_elo"] + (1.0 - reliability) * team_elo_scaled


def displacement_stats(merged, fmt, min_rate=DEFAULT_MIN_RATE):
    """For every (match, team), compares the old ('has ever bowled') top-K bowling squad
    against the new (recent-rate-gated) top-K, using PlayerEloEngine's own weighted-mean
    method so the reconstruction matches production exactly."""
    w_bowl = PlayerEloEngine.W_BOWL[fmt]
    k = len(w_bowl)
    merged = merged.copy()
    merged["eff_elo"] = merged.apply(lambda r: compute_eff_elo(r, fmt), axis=1)
    merged["old_rank_val"] = np.where(merged["n_prior"] > 0, merged["eff_elo"], 0.0)
    merged["new_rank_val"] = np.where(merged["recent_rate"] >= min_rate, merged["eff_elo"], 0.0)

    changed = 0
    total = 0
    abs_deltas = []
    displaced_names = []
    for (mid, team_key), g in merged.groupby(["match_id", "team_key"]):
        players = list(g["player"])
        elo_map = dict(zip(g["player"], g["eff_elo"]))
        old_rank = dict(zip(g["player"], g["old_rank_val"]))
        new_rank = dict(zip(g["player"], g["new_rank_val"]))
        fallback = float(g["team_elo"].iloc[0])
        fallback_scaled = 1500.0 + (fallback - 1500.0) * _stretch_for(fmt)

        old_top = set(sorted(players, key=lambda p: old_rank.get(p, 0.0), reverse=True)[:k])
        new_top = set(sorted(players, key=lambda p: new_rank.get(p, 0.0), reverse=True)[:k])
        total += 1
        if old_top != new_top:
            changed += 1
            displaced_names.extend(old_top - new_top)

        old_sq = PlayerEloEngine._weighted_mean_by_bowling_rank(
            players, elo_map, fallback_scaled, w_bowl, rank_key_map=old_rank
        )
        new_sq = PlayerEloEngine._weighted_mean_by_bowling_rank(
            players, elo_map, fallback_scaled, w_bowl, rank_key_map=new_rank
        )
        abs_deltas.append(abs(new_sq - old_sq))

    return {
        "format": fmt,
        "min_rate": min_rate,
        "n_squads": total,
        "pct_squad_changed": 100.0 * changed / total if total else 0.0,
        "mean_abs_delta": float(np.mean(abs_deltas)) if abs_deltas else 0.0,
        "p90_abs_delta": float(np.percentile(abs_deltas, 90)) if abs_deltas else 0.0,
        "max_abs_delta": float(np.max(abs_deltas)) if abs_deltas else 0.0,
        "top_displaced_players": pd.Series(displaced_names).value_counts().head(10).to_dict() if displaced_names else {},
    }


def run_format(fmt, sweep=False):
    print(f"\n{'=' * 70}\n{fmt.upper()}\n{'=' * 70}")
    p_df, e_df = load_format(fmt)
    q = build_query_frame(e_df)

    merged = attach_bowling_history(q, p_df)
    merged = attach_recent_bowl_rate(merged, p_df)

    print(f"query rows (player-match appearances, named XI): {len(merged):,}")
    print("\nRecent-bowling-rate distribution (all named-XI rows):")
    print(merged["recent_rate"].describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_string())

    stats = displacement_stats(merged, fmt, DEFAULT_MIN_RATE)
    print(f"\nDisplacement @ min_rate={DEFAULT_MIN_RATE}, window={DEFAULT_WINDOW}:")
    print(f"  squads analysed        : {stats['n_squads']:,}")
    print(f"  squads with top-{len(PlayerEloEngine.W_BOWL[fmt])} changed : {stats['pct_squad_changed']:.2f}%")
    print(f"  mean |delta| bowling squad elo : {stats['mean_abs_delta']:.3f}")
    print(f"  p90  |delta|                    : {stats['p90_abs_delta']:.3f}")
    print(f"  max  |delta|                    : {stats['max_abs_delta']:.3f}")
    if stats["top_displaced_players"]:
        print(f"  most-frequently-displaced players (false-frontline class):")
        for name, cnt in stats["top_displaced_players"].items():
            print(f"    {name:25s} displaced in {cnt} squads")

    if sweep:
        print(f"\nThreshold sweep:")
        print(f"  {'min_rate':>9s} {'window':>7s} {'%changed':>9s} {'mean|d|':>9s} {'p90|d|':>9s}")
        for window in (5, 10, 15):
            m2 = attach_bowling_history(q, p_df)
            m2 = attach_recent_bowl_rate(m2, p_df, window=window)
            for min_rate in (0.1, 0.15, 0.2, 0.25, 0.3):
                s = displacement_stats(m2, fmt, min_rate)
                print(f"  {min_rate:>9.2f} {window:>7d} {s['pct_squad_changed']:>8.2f}% {s['mean_abs_delta']:>9.3f} {s['p90_abs_delta']:>9.3f}")

    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats", nargs="+", default=["t20", "od", "hnd", "t20w", "odw", "hndw"])
    ap.add_argument("--sweep", action="store_true", help="Also run the threshold/window sweep (t20/od only recommended -- rank order is a no-op for uniform-W_BOWL formats)")
    args = ap.parse_args()

    all_stats = []
    for fmt in args.formats:
        try:
            all_stats.append(run_format(fmt, sweep=args.sweep))
        except FileNotFoundError as e:
            print(f"\n{fmt}: SKIPPED ({e})")

    print(f"\n\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for s in all_stats:
        print(f"{s['format']:6s} squads_changed={s['pct_squad_changed']:6.2f}%  mean|delta|={s['mean_abs_delta']:7.3f}  p90|delta|={s['p90_abs_delta']:7.3f}")
