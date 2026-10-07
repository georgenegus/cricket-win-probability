"""
Phases 4-5 of markdowns/win_predictor/franchise_bridge_plan.md: franchise player and team
rankings, for any date in the corpus.

Players are rated by bridging from internationally-rated anchors (FranchiseBridgeEngine);
teams are composed from those player ratings using the same positional/frontline squad
weighting production squad-Elo uses. Both are expressed on the INTERNATIONAL team rating
scale, so a franchise team's number is directly comparable to a national side's.

    py scripts/franchise_rankings.py                          # current rankings
    py scripts/franchise_rankings.py --as-of 2019-06-01       # as they stood then
    py scripts/franchise_rankings.py --as-of 2016-05-01 --top 20
    py scripts/franchise_rankings.py --season-ends            # a snapshot per year

> Anchoring is thin before ~2014 (T20Is began in 2005, so few players had 20 caps early on).
> Snapshots before then are reported with a warning and should not be trusted.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.load_data import CricketDataLoader
from src.data_processing.match_metadata import MatchMetadataExtractor
from src.data_processing.format_spec import spec, FRANCHISE_BRIDGE_TIERS
from src.features.franchise_bridge_engine import FranchiseBridgeEngine

# Below this date the bridge has too few anchors to be meaningful -- see the plan's coverage
# table (2008-09 are 95-100% unanchored; 2014 onward is >99% anchored).
TRUSTWORTHY_FROM = pd.Timestamp("2014-01-01")


def load_all(format_type, tiers, force):
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
    return engine, bridge_df, meta


def report(engine, bridge_df, meta, as_of, top, lookback):
    if as_of < TRUSTWORTHY_FROM:
        print(f"\n  !! {as_of.date()} predates reliable anchoring ({TRUSTWORTHY_FROM.date()}); "
              f"treat the following as unanchored and unreliable.")

    print(f"\n{'=' * 72}\nFRANCHISE RANKINGS as of {as_of.date()}\n{'=' * 72}")

    for role, label in (("batter", "BATTERS"), ("bowler", "BOWLERS")):
        ratings = FranchiseBridgeEngine.get_effective_ratings_from_df(bridge_df, role, as_of_date=as_of)
        if not ratings:
            print(f"\nTop {label}: none qualified at this date")
            continue
        anchor_flag = (bridge_df[bridge_df["role"] == role]
                       .groupby("player")["is_anchor"].max().to_dict())
        ordered = sorted(ratings, key=ratings.get, reverse=True)[:top]
        print(f"\nTop {label}  (of {len(ratings)} qualified)")
        for i, p in enumerate(ordered, 1):
            tag = "" if anchor_flag.get(p) else "  (franchise-only)"
            print(f"  {i:>3}. {p:<26}{ratings[p]:>8.1f}{tag}")

    teams = engine.team_ratings_as_of(bridge_df, meta, as_of_date=as_of, lookback_days=lookback)
    print(f"\nTop TEAMS  (of {len(teams)} with >=5 matches in the last {lookback} days)")
    if teams.empty:
        print("  none qualified at this date")
        return
    print(f"  {'#':>3} {'team':<28}{'rating':>8}{'bat':>8}{'bowl':>8}  comp")
    for i, r in teams.head(top).iterrows():
        print(f"  {i + 1:>3} {r['team']:<28}{r['team_rating']:>8}{r['batting_rating']:>8}"
              f"{r['bowling_rating']:>8}  {r['competition']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--format", default="t20")
    parser.add_argument("--tiers", nargs="+", default=["A"], choices=["A", "B", "C"])
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: latest in corpus)")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--lookback", type=int, default=365,
                        help="Days of squad history used to compose team ratings.")
    parser.add_argument("--season-ends", action="store_true",
                        help="Print a team snapshot for 1 Oct of each year instead.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    engine, bridge_df, meta = load_all(args.format, tuple(args.tiers), args.force)
    latest = pd.to_datetime(bridge_df["match_date"]).max()

    if args.season_ends:
        print(f"\nTEAM RATING HISTORY (top 5 each year, {args.lookback}-day squad window)")
        for year in range(2014, latest.year + 1):
            as_of = pd.Timestamp(f"{year}-10-01")
            if as_of > latest:
                break
            teams = engine.team_ratings_as_of(bridge_df, meta, as_of_date=as_of,
                                              lookback_days=args.lookback)
            if teams.empty:
                continue
            top5 = ", ".join(f"{r['team']} ({r['team_rating']:.0f})"
                             for _, r in teams.head(5).iterrows())
            print(f"  {year}: {top5}")
        return

    as_of = pd.Timestamp(args.as_of) if args.as_of else latest + pd.Timedelta(days=1)
    report(engine, bridge_df, meta, as_of, args.top, args.lookback)


if __name__ == "__main__":
    main()
