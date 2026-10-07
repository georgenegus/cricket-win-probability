"""
Venue resolution impact measurement (Phase 10.2).

Answers the question Phase 7 was supposed to answer but didn't: does venue
resolution actually move predictions, and by how much? Phase 7's "validation"
was a generic 5-fold Brier score that never touches any venue-resolution
code — it would report the same number whether or not this feature existed.

Methodology, per historical fixture (team_1 v team_2, known true venue):
  - baseline:      predictor output with no venue info at all (venue=None) —
                    what every fixture got before this feature existed.
  - resolved:      predictor output using VenueHistoryResolver's inferred
                    venue (team-history + overrides only — simulates "Betfair
                    gave us nothing") — what a fixture gets now when Betfair
                    omits the venue.
  - true:          predictor output using the actual historical venue — the
                    best case, "what if we'd known the real ground".

Reports:
  - Coverage: what fraction of fixtures the resolver actually fires on.
  - Accuracy: of fixtures where it fires, how often the resolved venue IS the
    true venue (exact canonical match).
  - Impact: distribution of |resolved_prob_a - baseline_prob_a| — the actual
    swing this feature produces, restricted to fixtures where it fires
    (fixtures it abstains on are, correctly, unaffected).
  - How much of the true-vs-baseline gap resolution actually closes.

Run: python scripts/measure_venue_impact.py [--format t20|od|both] [--sample N]
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import numpy as np

from src.betting.pre_match_predictor import PreMatchPredictor
from src.features.venue_history_resolver import VenueHistoryResolver
from src.features.venue_aliases import resolve_venue_alias


def measure_format(fmt: str, predictor: PreMatchPredictor, resolver: VenueHistoryResolver, sample: int | None, save_csv: bool = False):
    meta_path = PROJECT_ROOT / "data" / "processed" / f"{fmt}_match_metadata.parquet"
    lookup_path = PROJECT_ROOT / "data" / "processed" / f"{fmt}_venue_lookup.parquet"
    if not meta_path.exists() or not lookup_path.exists():
        print(f"[{fmt.upper()}] Missing metadata/lookup — skipping.")
        return None

    meta = pd.read_parquet(meta_path)
    lookup = pd.read_parquet(lookup_path)
    available_venues = set(lookup["venue"].dropna().unique())

    valid = meta[meta["venue"].notna() & meta["team_1"].notna() & meta["team_2"].notna()].copy()
    if sample and len(valid) > sample:
        valid = valid.sample(n=sample, random_state=42)

    rows = []
    for _, row in valid.iterrows():
        team_a = str(row["team_1"]).strip()
        team_b = str(row["team_2"]).strip()
        true_venue = resolve_venue_alias(str(row["venue"]).strip(), available_venues)
        if not true_venue:
            continue

        resolved_venue, confidence, info = resolver.resolve_from_history(team_a, team_b, format_type=fmt)

        baseline_res = predictor.predict(team_a, team_b, venue=None, format_type=fmt)
        true_res = predictor.predict(team_a, team_b, venue=true_venue, format_type=fmt)

        if resolved_venue:
            resolved_res = predictor.predict(team_a, team_b, venue=resolved_venue, format_type=fmt)
        else:
            resolved_res = baseline_res

        rows.append({
            "team_a": team_a,
            "team_b": team_b,
            "true_venue": true_venue,
            "resolved_venue": resolved_venue,
            "resolver_status": info["status"],
            "fired": resolved_venue is not None,
            "exact_match": resolved_venue == true_venue,
            "baseline_prob_a": baseline_res["prob_a"],
            "resolved_prob_a": resolved_res["prob_a"],
            "true_prob_a": true_res["prob_a"],
        })

    if not rows:
        print(f"[{fmt.upper()}] No usable fixtures.")
        return None

    df = pd.DataFrame(rows)
    df["resolved_vs_baseline_delta"] = (df["resolved_prob_a"] - df["baseline_prob_a"]).abs()
    df["true_vs_baseline_delta"] = (df["true_prob_a"] - df["baseline_prob_a"]).abs()
    df["gap_closed"] = np.where(
        df["true_vs_baseline_delta"] > 1e-9,
        1.0 - (df["true_prob_a"] - df["resolved_prob_a"]).abs() / df["true_vs_baseline_delta"],
        np.nan,
    )

    fired = df[df["fired"]]
    n = len(df)
    n_fired = len(fired)

    print(f"\n{'='*70}\n{fmt.upper()} — {n:,} fixtures evaluated\n{'='*70}")
    print(f"Coverage: resolver fired on {n_fired:,}/{n:,} ({n_fired/n:.1%})")
    if n_fired:
        print(f"  by status: {fired['resolver_status'].value_counts().to_dict()}")
        print(f"Accuracy (fired fixtures only, ALL statuses pooled): exact venue match {fired['exact_match'].mean():.1%}")
        print(f"\nImpact on P(team_a wins) when resolver fires (n={n_fired:,}):")
        print(f"  |resolved - baseline| delta: mean={fired['resolved_vs_baseline_delta'].mean():.4f}, "
              f"median={fired['resolved_vs_baseline_delta'].median():.4f}, "
              f"p90={fired['resolved_vs_baseline_delta'].quantile(0.9):.4f}, "
              f"max={fired['resolved_vs_baseline_delta'].max():.4f}")
        print(f"  |true - baseline| delta (ceiling, if venue were always known): "
              f"mean={fired['true_vs_baseline_delta'].mean():.4f}, median={fired['true_vs_baseline_delta'].median():.4f}")
        gap = fired["gap_closed"].dropna()
        if len(gap):
            print(f"  Fraction of true-vs-baseline gap closed (mean, unclipped -- sensitive to outliers): {gap.mean():.1%}")
            print(f"  Fraction of true-vs-baseline gap closed (median, robust): {gap.median():.1%}")

        print(f"\n  --- Breakdown by resolver status (this is the load-bearing table) ---")
        for status, grp in fired.groupby("resolver_status"):
            g = grp["gap_closed"].dropna()
            print(
                f"  {status:20s} n={len(grp):4d}  exact_match={grp['exact_match'].mean():.1%}  "
                f"median|resolved-baseline|={grp['resolved_vs_baseline_delta'].median():.4f}  "
                f"median_gap_closed={g.median():.1%}" if len(g) else
                f"  {status:20s} n={len(grp):4d}  exact_match={grp['exact_match'].mean():.1%}"
            )
    else:
        print("  Resolver never fired on this sample — no impact to report.")

    if save_csv:
        out_dir = PROJECT_ROOT / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"venue_impact_{fmt}.csv"
        df.to_csv(out_path, index=False)
        print(f"\n  Raw per-fixture rows saved to: {out_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Measure venue resolution's real impact on predictions")
    parser.add_argument("--format", choices=["t20", "od", "both"], default="both")
    parser.add_argument("--sample", type=int, default=3000, help="Max fixtures per format (0 = all)")
    parser.add_argument("--save-csv", action="store_true", help="Save per-fixture rows to outputs/venue_impact_<fmt>.csv")
    args = parser.parse_args()
    sample = None if args.sample == 0 else args.sample

    print("Loading PreMatchPredictor and VenueHistoryResolver...")
    predictor = PreMatchPredictor()
    resolver = VenueHistoryResolver()

    formats = ["t20", "od"] if args.format == "both" else [args.format]
    for fmt in formats:
        measure_format(fmt, predictor, resolver, sample, save_csv=args.save_csv)


if __name__ == "__main__":
    main()
