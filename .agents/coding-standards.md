# Coding Standards

## Language & Runtime

- **Python 3.14+** (Windows)
- Key dependencies: `pandas`, `numpy`, `xgboost`, `scikit-learn`, `flask`, `pyyaml`, `pyarrow`
- No virtual environment convention — packages are installed globally

## Code Style

### General

- Use **type hints** for function signatures where practical
- Use **docstrings** for classes and public methods
- Use `# Comment` for inline explanations of non-obvious logic
- Prefer **explicit over implicit** — avoid magic numbers without named constants or comments
- Keep functions focused on a single responsibility


### Naming Conventions

- Classes: `PascalCase` (e.g. `StateBuilder`, `EloEngine`, `XGBWinProbabilityModel`)
- Functions/methods: `snake_case` (e.g. `build_innings_state`, `compute_expected`)
- Private methods: prefix with `_` (e.g. `_extract_features_and_target`, `_extract_features_single`)
- Constants: `UPPER_SNAKE_CASE` for true constants, but format-specific defaults are set in `__init__` with conditional logic
- Variables: `snake_case`, descriptive names (e.g. `decay_weight`, `wicket_pressure`, `second_innings_states`)

### File Organisation

- **Source code**: `src/` with subpackages `data_processing/`, `features/`, `models/`
- **Tests & validation**: `tests/` — these are validation scripts, not unit tests. They train models and report metrics
- **Sweep scripts**: `tests/` — named `{format}_{feature}_sweep.py`
- **Static web assets**: `static/` with `index.html`, `css/`, `js/`
- **Documentation**: `docs/` for system-level docs, `markdowns/upgrades/` for historical upgrade plans
- **Cached data**: `data/processed/` — parquet files and pickle model files


### Model Code Conventions

- Feature extraction is split into two methods:
  - `_extract_features_and_target(df)` — vectorised, for training on full DataFrames
  - `_extract_features_single(state: pd.Series)` — scalar, for single-state prediction in the Flask app
- **`predict_batch(states_df)`** — vectorised bulk prediction that handles boundaries, fold routing, and feature extraction in batch. Use this for validation and sweep scripts.
- **Both `_extract_features_and_target` and `_extract_features_single` must produce features in the same column order** — XGBoost's `colsample_bytree` is sensitive to column order under a fixed random seed
- Format-specific defaults are set in `__init__` via `if self.format_type == "od": ... else: ...` blocks
- Format-specific features (e.g. DLS, partnership) are guarded with `if self.format_type == "od":` conditionals
- Models are serialised via `pickle.dump()` to `.pkl` files in `data/processed/`

## Data Integrity Rules

### No Data Leakage

This is the single most important rule in the codebase:

1. **Chronological feature computation**: Elo ratings, venue stats, and player Elo are computed in strict date order. A match's pre-match features only use data from prior matches.
2. **Group K-Fold**: All balls from the same `match_id` are in the same fold. Never split a match across folds.
3. **Pre-match recording**: The Elo engine records ratings BEFORE updating them with the match result.
4. **Validation uses fold models**: During validation, predictions use the fold model that was NOT trained on that match.
5. **Final model for live predictions**: A 6th model trained on 100% of data is used only for new/unseen matches (e.g. in the Flask app).

### Gender Filtering

Both T20 and ODI models are trained on **male-only matches**. The `gender` column (from `match_metadata.py`) must be filtered before Elo computation and state building. This is done in the validation scripts, not in the loaders.

### Cache Management

- When changing features or model parameters, **delete the relevant cached files** in `data/processed/` to force recomputation
- Cache files: `{format}_elo_ratings.parquet`, `{format}_venue_features.parquet`, `{format}_player_elo_ratings.parquet`, `{format}_xgb_models.pkl`, `{format}_first_innings_regressor.pkl`
- The validation scripts handle cache clearing when needed

