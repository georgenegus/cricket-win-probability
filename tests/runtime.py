from src.data_processing.load_data import CricketDataLoader
from src.data_processing.build_states import StateBuilder
from src.data_processing.timer import Timer

timer = Timer()

# ====================================================
# LOAD DATA
# ====================================================

timer.start("load_data")

loader = CricketDataLoader("data/raw")
df = loader.load_all()

timer.stop("load_data")

# ====================================================
# BUILD STATES
# ====================================================

timer.start("build_states")

states = StateBuilder(df).build()

timer.stop("build_states")

# ====================================================
# FILTER SECOND INNINGS
# ====================================================

timer.start("filter_second_innings")

second = states[
    states["innings"] == 2
].copy()

timer.stop("filter_second_innings")

# ====================================================
# NEGATIVE STATE CHECK
# ====================================================

timer.start("negative_state_check")

negative_states = second[
    second["runs_remaining"] < 0
]

timer.stop("negative_state_check")

# ====================================================
# NULL CHECK
# ====================================================

timer.start("null_check")

critical_columns = [
    "runs_remaining",
    "wickets_lost",
    "balls_remaining",
    "won_match"
]

null_counts = (
    second[critical_columns]
    .isnull()
    .sum()
)

timer.stop("null_check")

# ====================================================
# DISTRIBUTION CHECK
# ====================================================

timer.start("distribution_check")

_ = second["runs_remaining"].describe()
_ = second["wickets_lost"].describe()
_ = second["balls_remaining"].describe()

timer.stop("distribution_check")

# ====================================================
# OUTPUTS
# ====================================================

print(f"Loaded {len(df):,} deliveries")
print(f"Built {len(states):,} states")
print(f"Second innings rows: {len(second):,}")
print(f"Negative states: {len(negative_states):,}")

print("\nNull counts:")
print(null_counts)

timer.report()