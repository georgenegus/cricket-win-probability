from pathlib import Path
import pandas as pd
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.load_data import CricketDataLoader
from src.data_processing.build_states import StateBuilder


def main():

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 1000)

    print("Loading data...")
    loader = CricketDataLoader("data/raw")
    deliveries = loader.load_all()

    print("\nColumns loaded:")
    print(deliveries.columns.tolist())

    print("\nBuilding states...")
    states = StateBuilder(deliveries).build()

    # --------------------------------------------------
    # QUICK CHECK: FIRST SECOND INNINGS STATE
    # --------------------------------------------------

    print("\n==================================================")
    print("FIRST SECOND INNINGS INSPECTION")
    print("==================================================")

    second_innings = states[states["innings"] == 2].copy()

    if len(second_innings) == 0:
        print("No second innings found")
    else:
        first_match_id = second_innings["match_id"].iloc[0]

        sample_innings = second_innings[
            second_innings["match_id"] == first_match_id
        ].sort_values(["over", "ball"])

        cols_to_show = [
            c for c in [
                "match_id",
                "innings",
                "over",
                "ball",
                "runs_scored",
                "runs_remaining",
                "balls_remaining",
                "wickets_lost",
                "wickets_remaining",
                "required_run_rate",
                "won_match"
            ]
            if c in sample_innings.columns
        ]

        print(sample_innings[cols_to_show].to_string(index=False))

        print("\nSUMMARY:")
        print("Wickets lost unique:", sorted(sample_innings["wickets_lost"].unique()))

        if "wickets_remaining" in sample_innings.columns:
            print("Wickets remaining unique:", sorted(sample_innings["wickets_remaining"].unique()))
        else:
            print("wickets_remaining column NOT PRESENT")

    print("\nSearching for negative balls_remaining states...\n")

    bad_states = states[states["balls_remaining"] < 0]

    print(f"Found {len(bad_states):,} negative states")

    if len(bad_states) == 0:
        print("No issues found.")
        return

    print("\nFirst 20 problematic rows:\n")

    print(
        bad_states[
            [
                "match_id",
                "innings",
                "over",
                "ball",
                "balls_remaining",
                "runs_remaining",
            ]
        ].head(20)
    )

    # --------------------------------------------------
    # Investigate first problematic state
    # --------------------------------------------------

    first_bad = bad_states.iloc[0]

    match_id = first_bad["match_id"]
    innings = first_bad["innings"]
    over = first_bad["over"]

    print("\n" + "=" * 80)
    print("INVESTIGATING FIRST PROBLEMATIC OVER")
    print("=" * 80)

    print(f"Match ID : {match_id}")
    print(f"Innings  : {innings}")
    print(f"Over     : {over}")

    # --------------------------------------------------
    # ORIGINAL DELIVERY DATA (FULL INNINGS)
    # --------------------------------------------------

    print("\nORIGINAL DELIVERY DATA (FULL INNINGS)")
    print("-" * 80)

    original_innings = deliveries[
        (deliveries["match_id"] == match_id)
        & (deliveries["innings"] == innings)
    ].copy()

    original_innings = original_innings.sort_values(["over", "ball"])

    print(original_innings.to_string(index=False))

    # --------------------------------------------------
    # STATE BUILDER OUTPUT (FULL INNINGS)
    # --------------------------------------------------

    print("\nSTATE BUILDER OUTPUT (FULL INNINGS)")
    print("-" * 80)

    state_innings = states[
        (states["match_id"] == match_id)
        & (states["innings"] == innings)
    ].copy()

    state_innings = state_innings.sort_values(["over", "ball"])

    cols_to_show = [
        c for c in [
            "match_id",
            "innings",
            "over",
            "ball",
            "runs_scored",
            "wickets_lost",
            "balls_remaining",
            "runs_remaining",
            "required_run_rate",
            "won_match",
        ]
        if c in state_innings.columns
    ]

    print(state_innings[cols_to_show].to_string(index=False))

    # --------------------------------------------------
    # BALLS GREATER THAN 6 CHECK
    # --------------------------------------------------

    print("\nBALLS GREATER THAN 6")
    print("-" * 80)

    extra_balls = original_innings[
        original_innings["ball"] > 6
    ]

    if len(extra_balls) == 0:
        print("No balls > 6 found")
    else:
        print(extra_balls.to_string(index=False))


if __name__ == "__main__":
    main()