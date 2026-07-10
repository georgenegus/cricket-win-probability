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

### Computational Efficiency

- **Always write the most computationally efficient code.** This is imperative for scripts such as the sweeps which are running a large number of combinations to find the optimal value of a parameter.
- Limit multi-threading threads (e.g. set environment variables `OPENBLAS_NUM_THREADS="1"` etc.) when executing nested numpy/pandas loops on Windows to prevent CPU thrashing.
- **Use `predict_batch()` instead of row-by-row `predict()`** for bulk validation. The batch method handles boundary conditions vectorially, extracts features once via `_extract_features_and_target()`, and calls `predict_proba()` per fold in a single pass — ~17× faster than `iterrows` loops.
- **Pre-aggregate expensive computations** before sweep loops. For example, call `PlayerEloEngine._aggregate_match_stats()` once and pass the result via `precomputed_stats=` to avoid re-groupby on 2.6M rows per iteration.
- **Cache partial state builds** in sweep scripts. Build the base state DataFrame once (without player Elo columns) and re-merge only the changing columns each iteration.
- **Skip final model training in sweeps** — pass `train_final_model=False` to `fit()` since sweeps only need the 5 fold models for validation, not the production model.

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

### DataFrame Conventions

- Use `.copy()` when creating filtered subsets to avoid `SettingWithCopyWarning`
- Use `.astype(float)` explicitly when extracting features to ensure consistent dtypes
- Use `df.get("column", default_series)` with `.fillna()` for safe feature extraction with fallbacks
- Sort DataFrames by `["match_id", "innings", "over", "ball"]` before cumulative operations
- Use `groupby().cumsum()` for running totals (runs_scored, wickets_lost, balls_faced)

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

## Validation Protocol

### Brier Score Evaluation

1. Train 5-fold GroupKFold models on all second-innings states
2. Sample 5,000 random states with `random_state=42`
3. For each sampled state, use the fold model that was NOT trained on that match
4. Compute Brier Score: `mean((predicted_probability - actual_outcome)²)`
5. Report overall Brier, per-phase Brier, and calibration buckets

### When Making Model Changes

1. **Clear relevant caches** (`os.remove()` the cached `.pkl` and `.parquet` files)
2. **Run the validation script** (`py tests/win_probability_validation.py` or `py tests/odi_validation.py`)
3. **Compare Brier score** against the baseline
4. **If improved**: Keep the change, update the walkthrough, copy models to app-compatible filenames
5. **If regressed**: Revert the change, restore the baseline

### Copying Models for Flask App

The validation scripts save models to `data/processed/xgb_models.pkl` and `data/processed/first_innings_regressor.pkl`, but the Flask app expects `t20_xgb_models.pkl` and `t20_first_innings_regressor.pkl`. After training:
```python
shutil.copyfile('data/processed/xgb_models.pkl', 'data/processed/t20_xgb_models.pkl')
shutil.copyfile('data/processed/first_innings_regressor.pkl', 'data/processed/t20_first_innings_regressor.pkl')
```

## Flask App Conventions

- Models are loaded once at startup from `.pkl` caches
- All deliveries are loaded into memory at startup (~5.3M rows)
- Match catalog is precomputed at startup
- API routes return JSON
- The `/api/match/:id` endpoint rebuilds states on-the-fly for a single match
- Format detection: `"od" if max(over) > 20 else "t20"`

## Frontend Conventions

- Single-page app with tab navigation
- Chart.js for all visualisations
- Vanilla JS — no framework
- Glassmorphism dark-mode design with Outfit font family
- API calls via `fetch()` to Flask backend on `localhost:5000`
