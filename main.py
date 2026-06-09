from src.data_processing.load_data import CricketDataLoader
from src.data_processing.build_states import StateBuilder

# Load raw data
loader = CricketDataLoader("data/raw")
df = loader.load_all()

print(f"Loaded {len(df):,} deliveries")

# Build state table
states = StateBuilder(df).build()

second = states[
    states["innings"] == 2
]

print(
    second[
        [
            "match_id",
            "runs_remaining",
            "balls_remaining",
            "required_run_rate",
            "won_match"
        ]
    ].head(20)
)