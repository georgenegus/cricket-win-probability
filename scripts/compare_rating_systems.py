"""
Head-to-head: does the ORIGINAL Elo or the NEW ICC-style/bridge rating better predict what a
player actually does next?

Why this test rather than agreement-with-ICC: the new engine was explicitly built to mirror
ICC's methodology, so scoring it on ICC agreement is close to circular. Forward prediction is
neutral -- neither system has any advantage in the choice of yardstick. It simply asks: rate
every player as of date T, then see whose ratings better anticipate performance over the
following year.

Design
  - Yardsticks are computed straight from deliveries and belong to neither system:
      batting -- runs per innings, and strike rate
      bowling -- wickets per innings, and economy (sign-flipped so higher is always better)
    Reported separately rather than blended, so no weighting choice can tilt the result.
  - Two scopes, each matched to what both systems can legitimately see:
      international   -- old player Elo vs ICC-style international rating
      franchise       -- old player Elo vs the Tier A bridge rating
  - Matched population per evaluation date: every player with enough history before T AND
    enough innings after T, rated by BOTH systems. No qualification floors are applied beyond
    that, so neither system is advantaged by its own presentation rules.
  - Old Elo is read the way production reads it (most recent pre-match effective_elo before T,
    the pre_match_predictor.py convention), which is the honest representation of that system.

Usage:
    py scripts/compare_rating_systems.py
    py scripts/compare_rating_systems.py --from-year 2016 --to-year 2025
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
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.load_data import CricketDataLoader
from src.data_processing.match_metadata import MatchMetadataExtractor
from src.data_processing.format_spec import spec, INTL_COMPETITIONS, franchise_competitions
from src.features.icc_player_rating_engine import ICCPlayerRatingEngine, AsOfRatingIndex


def per_innings_stats(deliveries: pd.DataFrame, format_type: str, match_dates: dict) -> tuple:
    """Neutral per-innings performance records, computed from deliveries only.

    The bowling frame also carries `expected_runs` -- what an average bowler would have
    conceded off exactly the balls this bowler bowled, given WHICH PHASES they bowled them in
    and the prevailing run rate in that phase that season. This is what makes the
    phase-adjusted economy yardstick role-neutral: in T20 the best bowlers are handed the death
    and the powerplay against aggressive batting, so raw economy penalises them for their
    assignment rather than their bowling. Season-specific rates also absorb era drift (T20
    scoring has risen sharply over the corpus).
    """
    scorer = ICCPlayerRatingEngine()
    bat, bowl, _ = scorer._aggregate_match_stats(deliveries)
    bat = bat[bat["balls_faced"] > 0].rename(columns={"batter": "player"})
    bowl = bowl[bowl["balls_bowled"] > 0].rename(columns={"bowler": "player"})

    fmt_spec = spec(format_type)
    bpo = float(fmt_spec["balls_per_over"])
    pp_end = fmt_spec["pp_end_ball"] / bpo
    death_start = fmt_spec["death_start_ball"] / bpo

    d = deliveries.copy()
    d["over_float"] = d["over"] + (d["ball"] - 1) / bpo
    d["phase"] = np.where(d["over_float"] < pp_end, "powerplay",
                          np.where(d["over_float"] < death_start, "middle", "death"))
    if "legal_delivery" not in d.columns:
        d["legal_delivery"] = ((d["is_wide"] == 0) & (d["is_noball"] == 0)).astype(int)
    d["season"] = d["match_id"].map(match_dates).dt.year

    # prevailing runs-per-ball by season x phase -- the "average bowler" reference
    ref = (d.groupby(["season", "phase"])
             .agg(r=("total_runs", "sum"), b=("legal_delivery", "sum")).reset_index())
    ref["rate"] = np.where(ref["b"] > 0, ref["r"] / ref["b"], np.nan)
    rate_lookup = ref.set_index(["season", "phase"])["rate"].to_dict()

    per_phase = (d.groupby(["match_id", "bowler", "season", "phase"])["legal_delivery"]
                   .sum().reset_index())
    per_phase["expected"] = [
        balls * rate_lookup.get((s, ph), np.nan)
        for balls, s, ph in zip(per_phase["legal_delivery"], per_phase["season"], per_phase["phase"])
    ]
    exp = (per_phase.groupby(["match_id", "bowler"])["expected"].sum()
                    .reset_index().rename(columns={"bowler": "player",
                                                   "expected": "expected_runs"}))
    bowl = bowl.merge(exp, on=["match_id", "player"], how="left")
    return bat, bowl


def future_metrics(bat, bowl, match_dates, lo, hi) -> dict:
    """Actual performance in (lo, hi], per role, as {metric: {player: value}}."""
    out = {}
    b = bat[bat["match_id"].map(match_dates).between(lo, hi, inclusive="right")]
    g = b.groupby("player").agg(runs=("runs", "sum"), balls=("balls_faced", "sum"),
                                inns=("runs", "size"))
    g = g[g["inns"] > 0]
    out["batter"] = {
        "runs_per_innings": (g["runs"] / g["inns"]).to_dict(),
        "strike_rate": (g["runs"] / g["balls"] * 100.0).to_dict(),
        "_innings": g["inns"].to_dict(),
    }

    w = bowl[bowl["match_id"].map(match_dates).between(lo, hi, inclusive="right")]
    g = w.groupby("player").agg(wkts=("wickets", "sum"), runs=("runs_conceded", "sum"),
                                balls=("balls_bowled", "sum"), inns=("wickets", "size"),
                                expected=("expected_runs", "sum"))
    g = g[(g["inns"] > 0) & (g["balls"] > 0)]
    out["bowler"] = {
        "wickets_per_innings": (g["wkts"] / g["inns"]).to_dict(),
        # sign-flipped so that, as with every other metric here, higher = better
        "economy_inverted": (-(g["runs"] / g["balls"] * 6.0)).to_dict(),
        # runs saved per over against what an average bowler would have conceded off the SAME
        # phases -- role-neutral, unlike raw economy
        "economy_vs_expected": ((g["expected"] - g["runs"]) / g["balls"] * 6.0).to_dict(),
        "_innings": g["inns"].to_dict(),
    }
    return out


def old_elo_as_of(elo_df, match_dates, role, as_of) -> dict:
    """Most recent pre-match effective_elo strictly before as_of -- how production reads it."""
    df = elo_df[elo_df["role"] == role].copy()
    df["date"] = df["match_id"].map(match_dates)
    df = df[df["date"] < as_of].sort_values("date")
    return df.groupby("player")["effective_elo"].last().to_dict()


def history_counts(stats, match_dates, role_key, as_of) -> dict:
    df = stats[stats["match_id"].map(match_dates) < as_of]
    return df.groupby("player").size().to_dict()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--format", default="t20")
    p.add_argument("--from-year", type=int, default=2016)
    p.add_argument("--to-year", type=int, default=2025)
    p.add_argument("--min-history", type=int, default=10, help="Innings before T required.")
    p.add_argument("--min-future", type=int, default=8, help="Innings after T required.")
    args = p.parse_args()
    fmt = args.format

    raw = str(PROJECT_ROOT / "data" / "raw")
    deliveries = CricketDataLoader(raw, format_type=fmt).load_all()
    meta = MatchMetadataExtractor(raw, format_type=fmt).extract_all()
    meta = meta[meta["gender"] == spec(fmt)["gender"]].copy()
    meta["date"] = pd.to_datetime(meta["date"])
    match_dates = meta.set_index("match_id")["date"].to_dict()
    deliveries = deliveries[deliveries["match_id"].isin(meta["match_id"])].copy()

    proc = PROJECT_ROOT / "data" / "processed"
    old_elo = pd.read_parquet(proc / f"{fmt}_player_elo_ratings.parquet")
    icc_pl = pd.read_parquet(proc / f"{fmt}_icc_player_ratings.parquet")
    bridge = pd.read_parquet(proc / f"{fmt}_franchise_bridge_tiera.parquet")

    scopes = {
        "international": (set(meta[meta["competition"].isin(INTL_COMPETITIONS.get(fmt, ()))
                                   & (meta["team_type"] == "international")]["match_id"]),
                          AsOfRatingIndex(icc_pl), "ICC-style intl"),
        "franchise (Tier A)": (set(meta[meta["competition"].isin(franchise_competitions(fmt, ("A",)))]["match_id"]),
                               AsOfRatingIndex(bridge), "bridge"),
    }

    metric_names = {
        "batter": ["runs_per_innings", "strike_rate"],
        "bowler": ["wickets_per_innings", "economy_inverted", "economy_vs_expected"],
    }

    for scope_name, (match_ids, new_index, new_label) in scopes.items():
        sc_del = deliveries[deliveries["match_id"].isin(match_ids)]
        bat, bowl = per_innings_stats(sc_del, fmt, match_dates)
        stats_by_role = {"batter": bat, "bowler": bowl}

        print("\n" + "=" * 78)
        print(f"SCOPE: {scope_name}   (old Elo  vs  {new_label})")
        print("=" * 78)

        for role in ("batter", "bowler"):
            rows = {m: {"old": [], "new": [], "n": []} for m in metric_names[role]}
            for year in range(args.from_year, args.to_year + 1):
                T = pd.Timestamp(f"{year}-01-01")
                fut = future_metrics(bat, bowl, match_dates, T, T + pd.Timedelta(days=365))[role]
                hist = history_counts(stats_by_role[role], match_dates, role, T)
                old = old_elo_as_of(old_elo, match_dates, role, T)

                eligible = [
                    pl for pl, n in fut["_innings"].items()
                    if n >= args.min_future
                    and hist.get(pl, 0) >= args.min_history
                    and pl in old
                    and new_index.role_innings_before(pl, role, T) >= args.min_history
                ]
                if len(eligible) < 20:
                    continue
                new = {pl: new_index.effective_rating(pl, role, T) for pl in eligible}

                for m in metric_names[role]:
                    y = [fut[m][pl] for pl in eligible]
                    r_old, _ = spearmanr([old[pl] for pl in eligible], y)
                    r_new, _ = spearmanr([new[pl] for pl in eligible], y)
                    rows[m]["old"].append(r_old)
                    rows[m]["new"].append(r_new)
                    rows[m]["n"].append(len(eligible))

            print(f"\n  {role.upper()}  (mean Spearman of rating-at-T vs performance over the "
                  f"next year, {args.from_year}-{args.to_year})")
            print(f"    {'metric':<22}{'years':>7}{'mean n':>9}{'OLD Elo':>10}{'NEW':>10}{'diff':>9}")
            for m in metric_names[role]:
                d = rows[m]
                if not d["old"]:
                    print(f"    {m:<22}  insufficient data")
                    continue
                mo, mn = float(np.mean(d["old"])), float(np.mean(d["new"]))
                flag = "  <-- NEW" if mn > mo else ("  <-- OLD" if mo > mn else "")
                print(f"    {m:<22}{len(d['old']):>7}{np.mean(d['n']):>9.0f}"
                      f"{mo:>10.3f}{mn:>10.3f}{mn - mo:>+9.3f}{flag}")


if __name__ == "__main__":
    main()
