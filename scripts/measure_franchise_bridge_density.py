"""
Phase 0 of markdowns/win_predictor/franchise_bridge_plan.md: measures how densely "gold
standard" internationally-rated players appear in each franchise/domestic competition.

This is the feasibility constraint for the whole bridging idea -- a franchise match can only
link its players to the international rating scale if it contains players whose international
rating is already known. Uses the STRICT, time-aware definition: a player counts as an anchor
for a given match only if they had already reached MIN_MATCHES_TO_RANK international innings
BEFORE that match's date (a player who debuts internationally in 2024 is not an anchor for a
2016 IPL match).

Regenerates the numbers frozen in format_spec.FRANCHISE_BRIDGE_TIERS -- re-run as the corpus
grows to check whether any competition has changed tier.

Usage:
    py scripts/measure_franchise_bridge_density.py
    py scripts/measure_franchise_bridge_density.py --format t20 --min-intl-innings 20
    py scripts/measure_franchise_bridge_density.py --check    # non-zero exit if tiers moved
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

from src.data_processing.match_metadata import MatchMetadataExtractor
from src.data_processing.format_spec import spec, INTL_COMPETITIONS, FRANCHISE_BRIDGE_TIERS
from src.features.icc_player_rating_engine import ICCPlayerRatingEngine


def anchor_qualification_dates(format_type: str, min_intl_innings: int) -> dict:
    """player -> the date they reached `min_intl_innings` international appearances.
    A player is an anchor for any match AFTER that date, and not before."""
    ratings_path = PROJECT_ROOT / "data" / "processed" / f"{format_type}_icc_player_ratings.parquet"
    if not ratings_path.exists():
        raise FileNotFoundError(
            f"{ratings_path} not found -- run scripts/measure_icc_rating_crosscheck.py "
            f"--formats {format_type} first to build the international player ratings."
        )
    player_df = pd.read_parquet(ratings_path)
    player_df["match_date"] = pd.to_datetime(player_df["match_date"])

    # one row per player per international match (a player appearing as both batter and
    # bowler in the same match is one appearance, not two)
    appearances = player_df[["player", "match_date"]].drop_duplicates().sort_values("match_date")
    appearances["cum_apps"] = appearances.groupby("player").cumcount() + 1
    qualified = appearances[appearances["cum_apps"] == min_intl_innings]
    return qualified.set_index("player")["match_date"].to_dict()


def count_anchors(players_str: str, match_date, qualification_dates: dict) -> int:
    names = [n.strip() for n in str(players_str).split(",") if n.strip()]
    return sum(
        1 for n in names
        if n in qualification_dates and qualification_dates[n] <= match_date
    )


def measure(format_type: str, min_intl_innings: int) -> pd.DataFrame:
    fmt_spec = spec(format_type)
    meta = MatchMetadataExtractor(str(PROJECT_ROOT / "data" / "raw"), format_type=format_type).extract_all()
    meta = meta[meta["gender"] == fmt_spec["gender"]].copy()
    meta["date"] = pd.to_datetime(meta["date"])

    qualification_dates = anchor_qualification_dates(format_type, min_intl_innings)
    print(f"Anchor pool: {len(qualification_dates)} players ever reached "
          f"{min_intl_innings} international appearances in {format_type}\n")

    intl = set(INTL_COMPETITIONS.get(format_type, ()))
    rows = []
    for comp, sub in meta.groupby("competition"):
        if comp in intl:
            continue  # internationals are the anchor source, not a bridging target
        a1 = sub.apply(lambda r: count_anchors(r["team_1_players"], r["date"], qualification_dates), axis=1)
        a2 = sub.apply(lambda r: count_anchors(r["team_2_players"], r["date"], qualification_dates), axis=1)
        both_min = pd.concat([a1, a2], axis=1).min(axis=1)
        rows.append({
            "competition": comp,
            "matches": len(sub),
            "first": sub["date"].min().date(),
            "last": sub["date"].max().date(),
            "avg_anchors_per_xi": round((a1.mean() + a2.mean()) / 2, 1),
            "both_ge1_pct": round((both_min >= 1).mean() * 100, 1),
            "both_ge3_pct": round((both_min >= 3).mean() * 100, 1),
            "both_ge5_pct": round((both_min >= 5).mean() * 100, 1),
        })

    return pd.DataFrame(rows).sort_values("both_ge3_pct", ascending=False).reset_index(drop=True)


def registered_tier(comp: str, format_type: str):
    """Registered tier for a competition WITHIN ITS FORMAT -- the registry is per-format, since
    the same folder name (e.g. 't20s') means different things across formats."""
    for tier, comps in FRANCHISE_BRIDGE_TIERS.get(format_type, {}).items():
        if comp in comps:
            return tier
    return None


def classify(both_ge3_pct: float) -> str:
    """Tier boundaries as used when FRANCHISE_BRIDGE_TIERS was first frozen (2026-09-02)."""
    if both_ge3_pct >= 50.0:
        return "A"
    if both_ge3_pct >= 10.0:
        return "B"
    return "C"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", default="t20", choices=list(INTL_COMPETITIONS.keys()),
                        help="Bridging only makes sense for a format with an international pool to anchor from.")
    parser.add_argument("--min-intl-innings", type=int, default=ICCPlayerRatingEngine.MIN_MATCHES_TO_RANK,
                        help="International appearances before a player counts as an anchor "
                             "(default: ICCPlayerRatingEngine.MIN_MATCHES_TO_RANK).")
    parser.add_argument("--check", action="store_true",
                        help="Exit non-zero if any competition's measured tier differs from "
                             "the one registered in format_spec.FRANCHISE_BRIDGE_TIERS.")
    args = parser.parse_args()

    df = measure(args.format, args.min_intl_innings)

    print(f"{'comp':<6}{'n':>6}{'anchors/XI':>12}{'both>=1':>9}{'both>=3':>9}{'both>=5':>9}  {'tier':<6}{'registered':<11}")
    drift = []
    for _, r in df.iterrows():
        measured = classify(r["both_ge3_pct"])
        registered = registered_tier(r["competition"], args.format)
        flag = ""
        if registered is None:
            flag = "  <-- UNREGISTERED"
            drift.append((r["competition"], measured, registered))
        elif registered != measured:
            flag = f"  <-- DRIFT (was {registered})"
            drift.append((r["competition"], measured, registered))
        print(f'{r["competition"]:<6}{r["matches"]:>6}{r["avg_anchors_per_xi"]:>12}'
              f'{r["both_ge1_pct"]:>9}{r["both_ge3_pct"]:>9}{r["both_ge5_pct"]:>9}  '
              f'{measured:<6}{str(registered):<11}{flag}')

    if drift:
        print(f"\n{len(drift)} competition(s) differ from format_spec.FRANCHISE_BRIDGE_TIERS -- "
              f"update the registry (and re-check which leagues Tier A covers) before relying on it.")
    else:
        print("\nAll competitions match their registered tier.")

    if args.check and drift:
        sys.exit(1)


if __name__ == "__main__":
    main()
