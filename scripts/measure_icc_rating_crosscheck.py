"""
Phase 3 of markdowns/win_predictor/icc_rating_engine_plan.md: runs ICCTeamRatingEngine and
ICCPlayerRatingEngine on each in-scope format's international-only corpus and cross-checks
the result against tests/fixtures/icc_rankings_snapshot_2026.json (real, currently-published
ICC rankings).

This is a rank-correlation / face-validity check, NOT a claim of exact reproduction -- see
the fixture file's own "_disclaimer" and the engines' module docstrings for why an exact
match isn't the bar. No production files are read or written beyond this script's own
{format}_icc_team_ratings.parquet / {format}_icc_player_ratings.parquet caches.

Usage:
    py scripts/measure_icc_rating_crosscheck.py                  # all in-scope formats
    py scripts/measure_icc_rating_crosscheck.py --formats t20 od
    py scripts/measure_icc_rating_crosscheck.py --force          # bypass engine caches
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.load_data import CricketDataLoader
from src.data_processing.match_metadata import MatchMetadataExtractor
from src.data_processing.format_spec import spec, INTL_COMPETITIONS
from src.features.icc_team_rating_engine import ICCTeamRatingEngine
from src.features.icc_player_rating_engine import ICCPlayerRatingEngine

FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "icc_rankings_snapshot_2026.json"

# Face-validity anchors from markdowns/win_predictor/competition_tier_calibration_analysis.md
# (the current Elo leaderboard's known inversion: Rashid Khan 216th/648, Kohli 126th/786) --
# checked independently of the live ICC fixture, by surname substring against whichever
# cricsheet name variant this corpus actually uses (initials vary: "V Kohli", "JC Buttler", ...).
# Men's T20 only -- the only format that analysis was run against.
FACE_VALIDITY_ANCHORS = {
    "t20": {
        "batter": ["Kohli", "Buttler", "Gayle", "de Villiers"],
        "bowler": ["Bumrah", "Rashid Khan"],
    },
}


def load_format_data(format_type: str):
    fmt_spec = spec(format_type)
    raw_path = PROJECT_ROOT / "data" / "raw"

    deliveries = CricketDataLoader(str(raw_path), format_type=format_type).load_all()
    meta_df = MatchMetadataExtractor(str(raw_path), format_type=format_type).extract_all()
    if "gender" in meta_df.columns:
        meta_df = meta_df[meta_df["gender"] == fmt_spec["gender"]].copy()
        deliveries = deliveries[deliveries["match_id"].isin(meta_df["match_id"])].copy()
    return deliveries, meta_df


def build_player_team_map(icc_elo_df: pd.DataFrame) -> dict:
    """player name (cricsheet form) -> most-frequent team, derived from the international-
    only match squads already carried on icc_elo_df's team_1_players/team_2_players."""
    counts = {}
    for _, row in icc_elo_df.iterrows():
        for team_col, players_col in (("team_1", "team_1_players"), ("team_2", "team_2_players")):
            team = row.get(team_col)
            players_str = row.get(players_col) or ""
            for name in str(players_str).split(","):
                name = name.strip()
                if not name:
                    continue
                counts.setdefault(name, Counter())[team] += 1
    return {name: team_counts.most_common(1)[0][0] for name, team_counts in counts.items()}


def match_fixture_player(fixture_name: str, fixture_team: str, player_team_map: dict, rated_players: set):
    """Best-effort cricsheet-name match for one ICC fixture player, since cricsheet mixes
    'Initials Surname' and 'First Last' naming. Tries an exact match first, then a
    surname-token match constrained to the same team (to avoid cross-team surname collisions
    like multiple '...Khan's). Returns the matched cricsheet name, or None."""
    if fixture_name in rated_players:
        return fixture_name

    surname = fixture_name.split()[-1].lower()
    candidates = [
        name for name in rated_players
        if name.split()[-1].lower() == surname and player_team_map.get(name) == fixture_team
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def crosscheck_teams(format_type: str, fixture: dict) -> dict:
    team_engine = ICCTeamRatingEngine(format_type=format_type)
    meta_df = load_format_data(format_type)[1]
    icc_elo_df = team_engine.run(meta_df)
    current_ratings = ICCTeamRatingEngine.get_effective_ratings_from_df(icc_elo_df, format_type)

    fixture_table = fixture[format_type]["team_ratings"]["table"]
    common = [row for row in fixture_table if row["team"] in current_ratings]

    result = {
        "n_fixture_teams": len(fixture_table),
        "n_matched": len(common),
        "our_top5": sorted(current_ratings, key=current_ratings.get, reverse=True)[:5],
        "icc_top5": [row["team"] for row in fixture_table[:5]],
    }
    if len(common) >= 3:
        icc_ratings = [row["rating"] for row in common]
        our_ratings = [current_ratings[row["team"]] for row in common]
        corr, pval = spearmanr(icc_ratings, our_ratings)
        result["spearman_r"] = round(float(corr), 3)
        result["spearman_p"] = round(float(pval), 4)
    result["top5_overlap"] = len(set(result["our_top5"]) & set(result["icc_top5"]))
    return result, icc_elo_df


def crosscheck_players(format_type: str, fixture: dict, icc_elo_df: pd.DataFrame) -> dict:
    deliveries, meta_df = load_format_data(format_type)
    player_engine = ICCPlayerRatingEngine(format_type=format_type)
    icc_player_df = player_engine.run(deliveries, meta_df, icc_elo_df)

    player_team_map = build_player_team_map(icc_elo_df)

    result = {}
    for role, fixture_key in (("batter", "batting_top10"), ("bowler", "bowling_top10")):
        current_ratings = ICCPlayerRatingEngine.get_effective_ratings_from_df(icc_player_df, role)
        our_rank = {name: i + 1 for i, name in enumerate(sorted(current_ratings, key=current_ratings.get, reverse=True))}

        fixture_table = fixture[format_type][fixture_key]["table"]
        matched, unmatched = [], []
        for row in fixture_table:
            cricsheet_name = match_fixture_player(row["player"], row["team"], player_team_map, set(current_ratings))
            if cricsheet_name is None:
                unmatched.append(row["player"])
                continue
            matched.append({
                "icc_name": row["player"], "icc_rank": row["rank"],
                "cricsheet_name": cricsheet_name, "our_rank": our_rank[cricsheet_name],
                "our_percentile": round(100 * (1 - our_rank[cricsheet_name] / len(our_rank)), 1),
            })

        role_result = {"n_fixture": len(fixture_table), "n_matched": len(matched), "unmatched": unmatched, "matched": matched}
        if len(matched) >= 3:
            corr, pval = spearmanr([m["icc_rank"] for m in matched], [m["our_rank"] for m in matched])
            role_result["spearman_r"] = round(float(corr), 3)
            role_result["spearman_p"] = round(float(pval), 4)
        result[role] = role_result

    anchors = FACE_VALIDITY_ANCHORS.get(format_type, {})
    if anchors:
        anchor_result = {}
        for role, surnames in anchors.items():
            current_ratings = ICCPlayerRatingEngine.get_effective_ratings_from_df(icc_player_df, role)
            our_rank = {name: i + 1 for i, name in enumerate(sorted(current_ratings, key=current_ratings.get, reverse=True))}
            total = len(our_rank)
            hits = []
            for surname in surnames:
                candidates = [n for n in current_ratings if surname.lower() in n.lower()]
                if not candidates:
                    hits.append({"surname": surname, "found": False})
                    continue
                best = min(candidates, key=lambda n: our_rank[n])
                hits.append({
                    "surname": surname, "found": True, "cricsheet_name": best,
                    "rank": our_rank[best], "of": total,
                    "percentile": round(100 * (1 - our_rank[best] / total), 1),
                })
            anchor_result[role] = hits
        result["face_validity_anchors"] = anchor_result

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formats", nargs="+", default=list(INTL_COMPETITIONS.keys()),
                         choices=list(INTL_COMPETITIONS.keys()))
    parser.add_argument("--force", action="store_true", help="Bypass engine caches, recompute from scratch.")
    args = parser.parse_args()

    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        fixture = json.load(f)

    all_results = {}
    for format_type in args.formats:
        print(f"\n{'=' * 70}\n{format_type.upper()} -- ICC rating cross-check\n{'=' * 70}")

        team_result, icc_elo_df = crosscheck_teams(format_type, fixture)
        print(f"\nTeam ratings ({team_result['n_matched']}/{team_result['n_fixture_teams']} fixture teams matched):")
        print(f"  Spearman r = {team_result.get('spearman_r', 'n/a')} (p={team_result.get('spearman_p', 'n/a')})")
        print(f"  Top-5 overlap: {team_result['top5_overlap']}/5")
        print(f"  Our top 5:  {team_result['our_top5']}")
        print(f"  ICC top 5:  {team_result['icc_top5']}")

        player_result = crosscheck_players(format_type, fixture, icc_elo_df)
        for role in ("batter", "bowler"):
            r = player_result[role]
            print(f"\n{role.capitalize()} leaderboard ({r['n_matched']}/{r['n_fixture']} fixture players matched):")
            print(f"  Spearman r = {r.get('spearman_r', 'n/a')} (p={r.get('spearman_p', 'n/a')})")
            if r["unmatched"]:
                print(f"  Unmatched (name-matching gap, not necessarily a rating problem): {r['unmatched']}")
            for m in r["matched"]:
                print(f"    ICC #{m['icc_rank']:>2} {m['icc_name']:<22} -> our rank #{m['our_rank']} (p{m['our_percentile']})")

        if "face_validity_anchors" in player_result:
            print("\nFace-validity anchors (independent of the live ICC fixture):")
            for role, hits in player_result["face_validity_anchors"].items():
                for h in hits:
                    if h["found"]:
                        print(f"  {role} {h['surname']:<14} -> {h['cricsheet_name']:<20} rank {h['rank']}/{h['of']} (p{h['percentile']})")
                    else:
                        print(f"  {role} {h['surname']:<14} -> NOT FOUND in this format's rated players")

        all_results[format_type] = {"teams": team_result, "players": player_result}

    out_path = PROJECT_ROOT / "data" / "processed" / "icc_rating_crosscheck_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
