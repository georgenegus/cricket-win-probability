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


def is_full_retrain_due(max_age_days: int = 14) -> bool:
    """
    Checks if XGBoost models are missing or older than max_age_days (default 14 days).
    """
    t20_model_path = Path("data/processed/t20_xgb_models.pkl")
    if not t20_model_path.exists():
        return True

    mtime = datetime.fromtimestamp(t20_model_path.stat().st_mtime)
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

    # 1. Import new matches
    if args.full or args.days in ("60", "all"):
        stats = import_full_archive(dry_run=args.dry_run)
    elif args.catchup or args.days == "30":
        stats = import_catchup(dry_run=args.dry_run)
    else:
        stats = import_recent_matches(window=args.days, dry_run=args.dry_run)

    total_new = stats["new_t20"] + stats["new_od"]

    print(f"\nImport Summary:")
    print(f"  New T20 matches: {stats['new_t20']}")
    print(f"  New OD matches:  {stats['new_od']}")
    print(f"  Already existed: {stats['skipped_exists']}")
    print(f"  Skipped (non-male/non-LOI): {stats['skipped']}")
    print(f"  Errors:          {stats['errors']}")

    # 2. Check if 2-week full retrain is due or explicitly requested
    full_retrain_needed = args.retrain_full or is_full_retrain_due(max_age_days=14)
    stats["full_retrain"] = full_retrain_needed

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
            print("\nNo new matches found and full model retrain not yet due (last fit < 14 days ago).")
    else:
        print("\n[DRY RUN] No files modified, caches invalidated, or models retrained.")

    print(f"\nFinished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n")


if __name__ == "__main__":
    main()
