"""
Automated Cricsheet import + feature & model retrain.
Run daily/every 2 days via Windows Task Scheduler or CLI.

Cadence:
  - Every 2 days: Lightweight import + retrain of Elo ratings & venue averages (~30s).
  - Every 2 weeks (14 days): Full model retraining of XGBoost classifiers & First Innings regressors.
"""
import sys
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is on Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.cricsheet_importer import (
    import_recent_matches,
    import_catchup,
    import_full_archive,
    invalidate_caches,
    retrain_lightweight,
    retrain_full,
    log_import,
    validate_import,
)


def is_full_retrain_due(format_type: str = "t20", max_age_days: int = 14) -> bool:
    """
    Checks if a format's XGBoost model is missing or older than max_age_days (default 14 days).
    Parameterized (womens_games_ingestion_plan.md Phase 6) so t20w's retrain cadence is judged
    by its OWN model's age, not tied to men's t20's clock -- t20w was trained fresh in Phase 3,
    so piggybacking on t20's schedule would trigger wasteful, unnecessary early retrains.
    """
    model_path = Path(f"data/processed/{format_type}_xgb_models.pkl")
    if not model_path.exists():
        return True

    mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
    age = datetime.now() - mtime
    return age > timedelta(days=max_age_days)


def main():
    parser = argparse.ArgumentParser(description="TheInsideEdge -- Cricsheet Game Import & Model Retrain")
    parser.add_argument("--days", choices=["2", "7", "30", "60", "all"], default="2", help="Download window in days")
    parser.add_argument("--catchup", action="store_true", help="Run 30-day catch-up import")
    parser.add_argument("--full", action="store_true", help="Run full archive import (all_male.zip)")
    parser.add_argument("--retrain-full", action="store_true", help="Force a full XGBoost model retrain regardless of schedule")
    parser.add_argument("--dry-run", action="store_true", help="Report matches that would be imported without writing files")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*70}")
    print(f"  TheInsideEdge -- Cricsheet Game Importer & Retrainer")

    if args.full or args.days in ("60", "all"):
        mode_str = "FULL ARCHIVE SYNC (all_male.zip)"
    elif args.catchup or args.days == "30":
        mode_str = "CATCH-UP (30 Days)"
    else:
        mode_str = f"{args.days}-Day Window"

    print(f"  Time: {timestamp} | Mode: {mode_str}")
    print(f"{'='*70}\n")

    # 1. Import new matches -- male archive, as before.
    if args.full or args.days in ("60", "all"):
        stats = import_full_archive(dry_run=args.dry_run)
    elif args.catchup or args.days == "30":
        stats = import_catchup(dry_run=args.dry_run)
    else:
        stats = import_recent_matches(window=args.days, dry_run=args.dry_run)

    # 1b. Import new matches -- female archive (womens_games_ingestion_plan.md Phase 6), same
    # window/cadence as the male fetch above. Male and female archives never share a match_id
    # (separate Cricsheet source zips), so the two stats dicts can be summed key-by-key below
    # with no double-counting risk -- new_t20/new_od/new_hnd are always 0 from this call, and
    # new_t20w/new_odw/new_hndw are always 0 from the male call above.
    if args.full or args.days in ("60", "all"):
        female_stats = import_full_archive(dry_run=args.dry_run, gender="female")
    elif args.catchup or args.days == "30":
        female_stats = import_catchup(dry_run=args.dry_run, gender="female")
    else:
        female_stats = import_recent_matches(window=args.days, dry_run=args.dry_run, gender="female")

    for key in ("skipped", "skipped_exists", "errors"):
        stats[key] += female_stats.get(key, 0)
    for key in ("new_t20", "new_od", "new_hnd", "new_t20w", "new_odw", "new_hndw"):
        stats[key] = stats.get(key, 0) + female_stats.get(key, 0)

    total_new = stats["new_t20"] + stats["new_od"]
    total_new_t20w = stats["new_t20w"]
    total_new_odw = stats["new_odw"]
    total_new_hndw = stats["new_hndw"]

    print(f"\nImport Summary:")
    print(f"  New T20 matches:   {stats['new_t20']}")
    print(f"  New OD matches:    {stats['new_od']}")
    print(f"  New Hundred matches: {stats['new_hnd']}")
    print(f"  New T20W matches:  {stats['new_t20w']}")
    print(f"  New ODW matches:   {stats['new_odw']}")
    print(f"  New HNDW matches:  {stats['new_hndw']}")
    print(f"  Already existed:   {stats['skipped_exists']}")
    print(f"  Skipped:           {stats['skipped']}")
    print(f"  Errors:            {stats['errors']}")

    # 2. Check if 2-week full retrain is due or explicitly requested -- t20/od on their own
    # shared clock (unchanged), t20w checked independently (see is_full_retrain_due's docstring
    # -- it must not piggyback on t20's schedule, since t20w has its own fresh training date).
    full_retrain_needed = args.retrain_full or is_full_retrain_due("t20", max_age_days=14)
    t20w_full_retrain_needed = args.retrain_full or is_full_retrain_due("t20w", max_age_days=14)
    odw_full_retrain_needed = args.retrain_full or is_full_retrain_due("odw", max_age_days=14)
    hndw_full_retrain_needed = args.retrain_full or is_full_retrain_due("hndw", max_age_days=14)
    stats["full_retrain"] = (
        full_retrain_needed or t20w_full_retrain_needed
        or odw_full_retrain_needed or hndw_full_retrain_needed
    )

    if not args.dry_run:
        log_import(stats)

    # 3. Trigger appropriate retraining level
    if not args.dry_run:
        formats_to_update = []
        if total_new > 0 or full_retrain_needed:
            formats_to_update = ["t20", "od"]
            invalidate_caches(formats=formats_to_update)

        if full_retrain_needed:
            print(f"\n[!] BI-WEEKLY RETRAIN DUE (>=14 days since last fit or requested) -- Retraining full XGBoost models...")
            for fmt in formats_to_update:
                retrain_full(fmt)
                validate_import(fmt)
        elif total_new > 0:
            print(f"\n{total_new} new matches imported -- Invalidating caches and running lightweight Elo/Venue retrain...")
            for fmt in formats_to_update:
                retrain_lightweight(fmt)
                validate_import(fmt)
        else:
            print("\nNo new male-format matches found and full model retrain not yet due (last fit < 14 days ago).")

        # t20w retrain, independent of the t20/od decision above -- only trained women's format.
        if total_new_t20w > 0 or t20w_full_retrain_needed:
            invalidate_caches(formats=["t20w"])
            if t20w_full_retrain_needed:
                print(f"\n[!] T20W BI-WEEKLY RETRAIN DUE -- Retraining full t20w XGBoost models...")
                retrain_full("t20w")
                validate_import("t20w")
            else:
                print(f"\n{total_new_t20w} new t20w matches imported -- Invalidating caches and running lightweight Elo/Venue retrain...")
                retrain_lightweight("t20w")
                validate_import("t20w")
        else:
            print("No new t20w matches found and t20w full retrain not yet due.")

        # odw retrain, independent of the other formats -- unpooled, its own clock.
        if total_new_odw > 0 or odw_full_retrain_needed:
            invalidate_caches(formats=["odw"])
            if odw_full_retrain_needed:
                print(f"\n[!] ODW BI-WEEKLY RETRAIN DUE -- Retraining full odw XGBoost models...")
                retrain_full("odw")
                validate_import("odw")
            else:
                print(f"\n{total_new_odw} new odw matches imported -- Invalidating caches and running lightweight Elo/Venue retrain...")
                retrain_lightweight("odw")
                validate_import("odw")
        else:
            print("No new odw matches found and odw full retrain not yet due.")

        # hndw retrain, independent of the other formats -- pooled from t20w at fit() time
        # (retrain_full's POOL_SOURCE gating handles this automatically), its own clock.
        if total_new_hndw > 0 or hndw_full_retrain_needed:
            invalidate_caches(formats=["hndw"])
            if hndw_full_retrain_needed:
                print(f"\n[!] HNDW BI-WEEKLY RETRAIN DUE -- Retraining full hndw XGBoost models...")
                retrain_full("hndw")
                validate_import("hndw")
            else:
                print(f"\n{total_new_hndw} new hndw matches imported -- Invalidating caches and running lightweight Elo/Venue retrain...")
                retrain_lightweight("hndw")
                validate_import("hndw")
        else:
            print("No new hndw matches found and hndw full retrain not yet due.")
    else:
        print("\n[DRY RUN] No files modified, caches invalidated, or models retrained.")

    print(f"\nFinished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n")


if __name__ == "__main__":
    main()
