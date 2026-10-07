"""
Phase 6 of markdowns/win_predictor/franchise_bridge_plan.md: multi-hop extension to the
thinly-anchored leagues (Tier B/C -- BBL, T20 Blast, Super Smash, CSA T20, MSL, IPT, MCT).

Tier A works because internationally-rated anchors are present on both sides in 55-98% of its
matches. Tier B/C are not directly bridgeable (T20 Blast: 1.3 anchors per XI, both-sides->=3 in
1.9% of matches). The second hop uses players rated by the TIER A pass as anchors instead --
county pros who also play IPL/PSL/CPL carry the scale into leagues that have no international
presence of their own.

Runs Tier A first, then Tier B/C seeded from it, then validates the second hop the same way
Phase 3 validated the first: genuine international anchors DO appear in Tier B/C occasionally,
so their Tier B/C-derived rating can still be checked against their known international rating.

    py scripts/run_multihop_bridge.py
    py scripts/run_multihop_bridge.py --force
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
from src.data_processing.format_spec import spec, FRANCHISE_BRIDGE_TIERS
from src.features.icc_player_rating_engine import ICCPlayerRatingEngine
from src.features.franchise_bridge_engine import FranchiseBridgeEngine


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--format", default="t20")
    p.add_argument("--force", action="store_true")
    p.add_argument("--top", type=int, default=12)
    args = p.parse_args()
    fmt = args.format

    raw = str(PROJECT_ROOT / "data" / "raw")
    deliveries = CricketDataLoader(raw, format_type=fmt).load_all()
    meta = MatchMetadataExtractor(raw, format_type=fmt).extract_all()
    meta = meta[meta["gender"] == spec(fmt)["gender"]].copy()
    deliveries = deliveries[deliveries["match_id"].isin(meta["match_id"])].copy()

    proc = PROJECT_ROOT / "data" / "processed"
    icc_elo = pd.read_parquet(proc / f"{fmt}_icc_team_ratings.parquet")
    icc_pl = pd.read_parquet(proc / f"{fmt}_icc_player_ratings.parquet")

    # Tier selection is per-format: t20/t20w have a Tier A to seed from (hop1=A, hop2=B/C).
    # od/odw register ONLY Tier C -- there is no richer domestic tier to draw a second-hop
    # pool from, so hop1 and hop2 both run over Tier C, and hop2 self-seeds from hop1's own
    # output (domestic performers who built up a rating against DIRECT international anchors
    # become second-hop anchors for the Tier C matches that had none). Because both hops then
    # share one cache file, hop1 is forced fresh every run -- otherwise a stale cache could
    # already hold a self-seeded result and hop2 would compound seeding on top of it.
    fmt_tiers = FRANCHISE_BRIDGE_TIERS.get(fmt, {})
    self_seeding = "A" not in fmt_tiers
    if self_seeding:
        hop1_tiers = hop2_tiers = tuple(t for t in ("B", "C") if t in fmt_tiers) or ("C",)
    else:
        hop1_tiers = ("A",)
        hop2_tiers = tuple(t for t in ("B", "C") if t in fmt_tiers)

    print("=" * 72)
    print(f"HOP 1 -- Tier {'/'.join(hop1_tiers)} (directly anchored)")
    print("=" * 72)
    tier_a = FranchiseBridgeEngine(format_type=fmt, tiers=hop1_tiers)
    a_df = tier_a.run(deliveries, meta, icc_elo, icc_pl, force=args.force or self_seeding)

    if not hop2_tiers:
        print(f"\nNo thinner tier registered for '{fmt}' beyond {hop1_tiers} -- nothing to bridge in hop 2.")
        return

    print("\n" + "=" * 72)
    if self_seeding:
        print(f"HOP 2 -- Tier {'/'.join(hop2_tiers)} again, self-seeded with hop 1's own ratings "
              f"as second-hop anchors")
    else:
        print(f"HOP 2 -- Tier {'/'.join(hop2_tiers)}, seeded with Tier {'/'.join(hop1_tiers)} "
              f"ratings as second-hop anchors")
    print("=" * 72)
    tier_bc = FranchiseBridgeEngine(format_type=fmt, tiers=hop2_tiers)
    bc_df = tier_bc.run(deliveries, meta, icc_elo, icc_pl, force=True, second_hop_df=a_df)

    as_of = pd.to_datetime(bc_df["match_date"]).max() + pd.Timedelta(days=1)
    print(f"\nTier {'/'.join(hop2_tiers)} (hop 2): {len(bc_df):,} innings, {bc_df['player'].nunique():,} players")

    # anchor coverage achieved by the second hop
    print("\nAnchor coverage per competition (opponent anchors available per innings):")
    cov = bc_df.groupby("competition").agg(
        innings=("player", "size"),
        mean_anchors=("opponent_anchors", "mean"),
        pct_unanchored=("opponent_anchors", lambda s: round((s == 0).mean() * 100, 1)),
    ).sort_values("innings", ascending=False)
    print(cov.to_string())

    print("\n" + "=" * 72)
    print("VALIDATION -- does the second hop hold the scale?")
    print("=" * 72)
    for role in ("batter", "bowler"):
        bc = FranchiseBridgeEngine.get_effective_ratings_from_df(bc_df, role, as_of_date=as_of)
        intl = ICCPlayerRatingEngine.get_effective_ratings_from_df(icc_pl, role, as_of_date=as_of)
        a = FranchiseBridgeEngine.get_effective_ratings_from_df(a_df, role, as_of_date=as_of)

        common_i = sorted(set(bc) & set(intl))
        if len(common_i) >= 10:
            rho, pv = spearmanr([bc[p] for p in common_i], [intl[p] for p in common_i])
            gap = np.mean([bc[p] - intl[p] for p in common_i])
            print(f"{role}: vs INTERNATIONAL   n={len(common_i):<4} spearman={rho:.3f} "
                  f"(p={pv:.1e})  mean gap={gap:+.2f}")
        else:
            print(f"{role}: vs INTERNATIONAL   n={len(common_i)} -- too few to judge")

        common_a = sorted(set(bc) & set(a))
        hop1_label = f"TIER {'/'.join(hop1_tiers)} bridge" if not self_seeding else "HOP 1 (direct-anchor-only)"
        if len(common_a) >= 10:
            rho, pv = spearmanr([bc[p] for p in common_a], [a[p] for p in common_a])
            print(f"{role}: vs {hop1_label}   n={len(common_a):<4} spearman={rho:.3f} "
                  f"(p={pv:.1e})")
        else:
            print(f"{role}: vs {hop1_label}   n={len(common_a)} -- too few to judge")

    print("\n" + "=" * 72)
    print(f"NEWLY RATED -- players hop 1 could never reach")
    print("=" * 72)
    a_players = set(a_df["player"])
    for role in ("batter", "bowler"):
        bc = FranchiseBridgeEngine.get_effective_ratings_from_df(bc_df, role, as_of_date=as_of)
        only_bc = {p: v for p, v in bc.items() if p not in a_players}
        ordered = sorted(only_bc, key=only_bc.get, reverse=True)[:args.top]
        print(f"\nTop {role}s rated ONLY via the second hop ({len(only_bc)} such players):")
        for i, pl in enumerate(ordered, 1):
            print(f"  {i:>3}. {pl:<26}{only_bc[pl]:>8.1f}")


if __name__ == "__main__":
    main()
