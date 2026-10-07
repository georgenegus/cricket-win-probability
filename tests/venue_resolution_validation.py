"""
Venue resolution validation (Phase 10.1).

Exercises the venue resolution path end-to-end: alias/normalization layer,
the history-driven resolver (Phase 8), override precedence (Phase 4), and the
manual-review queue (Phase 9). Phase 7's original validation (Brier score /
GroupKFold) never touched any of this code — this script is what actually
checks it.

Run directly: python tests/venue_resolution_validation.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
import tempfile

from src.features.venue_aliases import resolve_venue_alias, clean_venue_string
from src.features.venue_history_resolver import VenueHistoryResolver
from src.features.venue_review_queue import queue_ambiguous_venue, load_queue
from src.betting.pre_match_predictor import PreMatchPredictor

FAILURES = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        FAILURES.append(label)


def section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def run():
    # --------------------------------------------------------------
    # 1. Predictor: resolved venue changes the score, unknown stays baseline
    # --------------------------------------------------------------
    section("1. Predictor non-baseline vs baseline")
    predictor = PreMatchPredictor()

    known_score = predictor.get_venue_avg_score("Lord's", format_type="t20")
    check("Known venue ('Lord's') returns non-baseline score", known_score != 160.0)

    unknown_score = predictor.get_venue_avg_score("Some Totally Made Up Ground XYZ", format_type="t20")
    check("Unknown venue string falls back to baseline (160.0)", unknown_score == 160.0)

    empty_score = predictor.get_venue_avg_score(None, format_type="t20")
    check("No venue supplied falls back to baseline (160.0)", empty_score == 160.0)

    od_known = predictor.get_venue_avg_score("Lord's", format_type="od")
    check("Known venue OD baseline differs from T20 default (260.0)", od_known != 260.0)

    # --------------------------------------------------------------
    # 2. Alias / normalization layer
    # --------------------------------------------------------------
    section("2. Alias resolution and ambiguity guard")

    fake_venues = {
        "kennington oval, london": 1,
        "old trafford, manchester": 1,
        "county ground, bristol": 1,
        "county ground, chelmsford": 1,
    }

    resolved = resolve_venue_alias("The Oval", fake_venues)
    check("'The Oval' resolves via alias map", resolved == "kennington oval, london")

    resolved = resolve_venue_alias("Emirates Old Trafford", fake_venues)
    check("'Emirates Old Trafford' resolves via alias map", resolved == "old trafford, manchester")

    ambiguous = resolve_venue_alias("County Ground", fake_venues)
    check("'County Ground' (bare, collidable) returns None, not a guess", ambiguous is None)

    # Two near-identical-scoring candidates -> reject rather than guess (the
    # <0.05 runner-up gap rule). "Sussex" fuzzy-matches Hove (0.80) and Derby
    # (0.78) within 0.02 of each other — close enough that the guard refuses
    # to pick, even though Hove happens to be the real answer. That's the
    # intended, conservative trade-off: abstain over a confident wrong guess.
    close_call_venues = {
        "county ground, derby": 1,
        "county ground, hove": 1,
        "county ground, bristol": 1,
    }
    close_call = resolve_venue_alias("County Ground, Sussex", close_call_venues)
    check("Near-tied fuzzy candidates (<0.05 apart) reject instead of guessing", close_call is None)

    check(
        "clean_venue_string is idempotent",
        clean_venue_string(clean_venue_string("The Oval, London!!")) == clean_venue_string("The Oval, London!!"),
    )

    # --------------------------------------------------------------
    # 3. History resolver (Phase 8): fires on concentrated teams, abstains on rotators
    # --------------------------------------------------------------
    section("3. History-driven resolver (Phase 8 concentration gate)")

    resolver = VenueHistoryResolver()

    for team in ["Royal Challengers Bangalore", "Chennai Super Kings", "Argentina"]:
        venue, conf, info = resolver.resolve_from_history(team, "Some Rotator XI", "t20")
        check(f"'{team}' abstains (no_data) — not a confident rotator guess", info["status"] == "no_data" and venue is None)

    for team, expected_venue_substr in [
        ("Lancashire", "old trafford"),
        ("Somerset", "taunton"),
        ("Nottinghamshire", "trent bridge"),
        ("Surrey", "kennington oval"),
    ]:
        venue, conf, info = resolver.resolve_from_history(team, "Some Rotator XI", "t20")
        check(
            f"'{team}' resolves to its concentrated home venue",
            info["status"] == "single_concentrated" and venue is not None and expected_venue_substr in venue,
        )

    venue, conf, info = resolver.resolve_from_history("Lancashire", "Surrey", "t20")
    check(
        "Two concentrated teams at different grounds -> tiebreak (not a silent pick)",
        info["status"] == "tiebreak" and venue is not None,
    )

    # --------------------------------------------------------------
    # 4. Override precedence (Phase 4) beats stale/absent history
    # --------------------------------------------------------------
    section("4. Override precedence")

    venue, conf, info = resolver.resolve_from_history("Antigua and Barbuda Falcons", "Somerset", "t20")
    check(
        "Seed override wins outright with confidence 1.0",
        info["status"] == "override" and conf == 1.0 and "antigua" in (venue or "").lower(),
    )

    # --------------------------------------------------------------
    # 5. Review queue (Phase 9): ambiguous strings queue, stay baseline, never expire
    # --------------------------------------------------------------
    section("5. Review queue")

    with tempfile.TemporaryDirectory() as tmp:
        qpath = Path(tmp) / "venue_review_queue.json"

        queue_ambiguous_venue("Weird Test Venue XJ9", "t20", reason="test", queue_path=qpath)
        queue_ambiguous_venue("Weird Test Venue XJ9", "t20", reason="test", queue_path=qpath)

        data = load_queue(queue_path=qpath)
        key = "t20::weird test venue xj9"
        check("Ambiguous identifier is queued", key in data)
        check("Repeated occurrences increment count, not duplicate entries", data.get(key, {}).get("occurrences") == 2)
        check("New entries default to 'pending' (baseline meanwhile, not lost)", data.get(key, {}).get("status") == "pending")

        # Simulate a human marking it genuinely-baseline — must not reopen to pending
        data[key]["status"] = "baseline"
        with open(qpath, "w", encoding="utf-8") as f:
            json.dump(data, f)

        queue_ambiguous_venue("Weird Test Venue XJ9", "t20", reason="test", queue_path=qpath)
        data = load_queue(queue_path=qpath)
        check("Reviewer's 'baseline' status is not reopened by later occurrences", data[key]["status"] == "baseline")

    # --------------------------------------------------------------
    # 6. Data hygiene: Phase 11 dedup fix (spot check on a known collision)
    # --------------------------------------------------------------
    section("6. Duplicate venue key merge (Phase 11)")

    import pandas as pd
    lookup_path = PROJECT_ROOT / "data" / "processed" / "t20_venue_lookup.parquet"
    if lookup_path.exists():
        lk = pd.read_parquet(lookup_path)
        chinnaswamy_rows = lk[lk["venue"].str.contains("chinnaswamy", case=False, na=False)]
        check("Chinnaswamy spelling variants merged into a single lookup row", len(chinnaswamy_rows) == 1)
    else:
        print("  [SKIP] t20_venue_lookup.parquet not found")

    # --------------------------------------------------------------
    section("SUMMARY")
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All venue resolution checks passed.")


if __name__ == "__main__":
    run()
