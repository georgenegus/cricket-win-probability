"""
Phase 2 of markdowns/win_predictor/rating_source_toggle_plan.md: promotes the full-corpus
rating-combination logic already proven in scripts/measure_rating_swap_brier.py
(build_full_corpus_ratings, validated there against the locked t20 baseline) into a real
artifact builder.

Writes, per qualifying format:
    data/processed/{fmt}_icc_combined_team_ratings.parquet
    data/processed/{fmt}_icc_combined_player_ratings.parquet

Schema is IDENTICAL to the Elo parquets (team_1_elo/team_2_elo; player/role/raw_elo/
effective_elo/squad columns) plus one added column, `rating_pool`, tagging which source
produced each row (intl / bridge_a / bridge_bc / uncovered) -- the provenance the UI badge
needs. Every source is standardised SEPARATELY onto a fixed (1500, 100) target before merging
(see build_full_corpus_ratings' own docstring for why), so the files are already on Elo's
scale at rest: no runtime rescaling exists downstream to drift out of sync with what was
trained on.

Per-format source composition (bridge tiers only included once THEIR OWN validation gate
passed -- see rating_source_toggle_plan.md Phase 1):
    t20   : intl + Tier A bridge + Tier B/C bridge   (all three validated, prior session)
    t20w  : intl + Tier A bridge                     (Tier A validated 2026-09-03; Tier C
                                                        was never built or gated -- excluded)
    od    : intl only                                (multi-hop bridge FAILED its gate)
    odw   : intl only                                (multi-hop bridge FAILED its gate)
    hnd/hndw are not callable here -- format_spec.INTL_COMPETITIONS has no entry for them
    (The Hundred has no international pool of its own format to anchor from).

raw_elo is recomputed the same way effective_elo is (standardised per source, per role) rather
than inherited from the Elo parquet the way build_full_corpus_ratings does it for the swap
EXPERIMENT (intentional there -- the experiment holds every other feature constant so only
team_elo/effective_elo are under test). Leaving Elo's raw_elo in a supposedly Elo-independent
production artifact would defeat Phase 5's point of training a genuinely separate model.
recent_form has no analogue in either new-source engine (a separate feature-engineering
question, not this builder's job) and is set neutral (0.0) rather than leaking Elo's value in.

Usage:
    py scripts/build_combined_ratings.py --format t20
    py scripts/build_combined_ratings.py --formats t20 t20w od odw
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.match_metadata import MatchMetadataExtractor
from src.data_processing.format_spec import spec, INTL_COMPETITIONS, franchise_competitions
from scripts.measure_rating_swap_brier import build_franchise_elo_df, build_full_corpus_ratings

TARGET_MU, TARGET_SD = 1500.0, 100.0

# Bridge tiers to include per format, gated on Phase 1's validation results -- NOT derived from
# format_spec.FRANCHISE_BRIDGE_TIERS directly, because that registry only says a tier is dense
# enough to ATTEMPT bridging, not that the attempt cleared the agreement-with-international gate.
VALIDATED_BRIDGE_TIERS = {
    "t20": (("tierA", ("A",)), ("tierBC", ("B", "C"))),
    "t20w": (("tierA", ("A",)),),
    "od": (),
    "odw": (),
}


def standardise(v: pd.Series) -> pd.Series:
    v = pd.Series(v, dtype=float)
    mu, sd = float(v.mean()), float(v.std())
    if sd <= 0:
        return pd.Series(np.full(len(v), TARGET_MU), index=v.index)
    return (v - mu) / sd * TARGET_SD + TARGET_MU


def build_sources_and_pools(fmt, meta, old_elo, proc):
    """(full_sources dict for build_full_corpus_ratings, match_id -> rating_pool dict)."""
    icc_teams = pd.read_parquet(proc / f"{fmt}_icc_team_ratings.parquet")
    icc_players = pd.read_parquet(proc / f"{fmt}_icc_player_ratings.parquet")
    intl_ids = set(meta[meta["competition"].isin(INTL_COMPETITIONS.get(fmt, ()))
                        & (meta["team_type"] == "international")]["match_id"])

    full_sources = {"intl": (icc_players, icc_teams)}
    pool_of = {mid: "intl" for mid in intl_ids}

    for name, tiers in VALIDATED_BRIDGE_TIERS.get(fmt, ()):
        tier_tag = "".join(tiers).lower()
        bridge_path = proc / f"{fmt}_franchise_bridge_tier{tier_tag}.parquet"
        if not bridge_path.exists():
            print(f"  WARNING: {bridge_path.name} not found -- skipping {name} for {fmt}")
            continue
        bridge_df = pd.read_parquet(bridge_path)
        comp_ids = set(meta[meta["competition"].isin(franchise_competitions(fmt, tiers))]["match_id"])
        full_sources[name] = (bridge_df, build_franchise_elo_df(
            old_elo[old_elo["match_id"].isin(comp_ids)], bridge_df, meta, fmt))
        pool_key = "bridge_a" if tiers == ("A",) else "bridge_bc"
        pool_of.update({mid: pool_key for mid in comp_ids})

    return full_sources, pool_of


def build_for_format(fmt: str, proc: Path):
    fmt_spec = spec(fmt)
    if fmt not in INTL_COMPETITIONS:
        print(f"Skipping '{fmt}' -- no international pool registered (format_spec.INTL_COMPETITIONS), "
              f"so there is nothing to anchor a combined artifact on.")
        return

    raw = str(PROJECT_ROOT / "data" / "raw")
    meta = MatchMetadataExtractor(raw, format_type=fmt).extract_all()
    meta = meta[meta["gender"] == fmt_spec["gender"]].copy()
    meta["date"] = pd.to_datetime(meta["date"])

    old_elo = pd.read_parquet(proc / f"{fmt}_elo_ratings.parquet")
    old_players = pd.read_parquet(proc / f"{fmt}_player_elo_ratings.parquet")

    full_sources, pool_of = build_sources_and_pools(fmt, meta, old_elo, proc)
    print(f"\n{fmt}: sources = {list(full_sources.keys())}")

    out_e, out_p = build_full_corpus_ratings(old_elo, old_players, full_sources, meta, fmt)

    # raw_elo: standardise each source's OWN raw_elo the same way effective_elo was, instead of
    # leaving build_full_corpus_ratings' inherited (Elo-derived) value in place -- see module
    # docstring for why that would leak Elo into a supposedly Elo-independent artifact.
    frames = []
    for name, (players_df, _) in full_sources.items():
        f = players_df[["match_id", "player", "role", "raw_elo"]].copy()
        for role in ("batter", "bowler"):
            m = f["role"] == role
            if m.sum() >= 10:
                f.loc[m, "raw_elo"] = standardise(f.loc[m, "raw_elo"]).values
        frames.append(f)
    new_raw = pd.concat(frames, ignore_index=True).rename(columns={"raw_elo": "new_raw_elo"})
    out_p = out_p.merge(new_raw, on=["match_id", "player", "role"], how="left")
    out_p["raw_elo"] = out_p["new_raw_elo"].fillna(TARGET_MU)
    out_p = out_p.drop(columns=["new_raw_elo"])
    out_p["recent_form"] = 0.0

    out_e["rating_pool"] = out_e["match_id"].map(pool_of).fillna("uncovered")
    out_p["rating_pool"] = out_p["match_id"].map(pool_of).fillna("uncovered")

    # ---- schema parity check: identical to the Elo parquets plus rating_pool
    expected_e = set(old_elo.columns) | {"rating_pool"}
    expected_p = set(old_players.columns) | {"rating_pool"}
    assert set(out_e.columns) == expected_e, f"team schema mismatch: {set(out_e.columns) ^ expected_e}"
    assert set(out_p.columns) == expected_p, f"player schema mismatch: {set(out_p.columns) ^ expected_p}"

    team_path = proc / f"{fmt}_icc_combined_team_ratings.parquet"
    player_path = proc / f"{fmt}_icc_combined_player_ratings.parquet"
    out_e.to_parquet(team_path, compression="snappy")
    out_p.to_parquet(player_path, compression="snappy")

    print(f"  wrote {team_path.name} ({len(out_e):,} matches), {player_path.name} ({len(out_p):,} rows)")
    print("  rating_pool coverage (matches):")
    print(f"    {out_e['rating_pool'].value_counts(normalize=True).mul(100).round(1).to_dict()}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--format", default=None)
    p.add_argument("--formats", nargs="+", default=None)
    args = p.parse_args()
    formats = args.formats or ([args.format] if args.format else ["t20", "t20w", "od", "odw"])

    proc = PROJECT_ROOT / "data" / "processed"
    for fmt in formats:
        build_for_format(fmt, proc)


if __name__ == "__main__":
    main()
