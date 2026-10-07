# Coding Standards

## Language & Runtime

- **Python 3.14+** (Windows)
- Key dependencies: `pandas`, `numpy`, `xgboost`, `scikit-learn`, `flask`, `pyyaml`, `pyarrow`
- No virtual environment convention — packages are installed globally
- Dev machine has only **7.7GB RAM** — full-corpus retrains (especially `t20`'s 2.9M rows) can OOM if run alongside other heavy processes. Standalone retrain/measurement scripts need an `if __name__ == "__main__":` guard (Windows `spawn` will otherwise silently re-run the whole script multiple times in worker processes)

## Code Style

### General

- Use **type hints** for function signatures where practical
- Use **docstrings** for classes and public methods
- Use `# Comment` for inline explanations of non-obvious logic — prefer explaining *why*, not *what*
- Prefer **explicit over implicit** — avoid magic numbers without named constants or comments
- Keep functions focused on a single responsibility

### Naming Conventions

- Classes: `PascalCase` (e.g. `StateBuilder`, `EloEngine`, `XGBWinProbabilityModel`)
- Functions/methods: `snake_case` (e.g. `build_innings_state`, `compute_expected`)
- Private methods: prefix with `_` (e.g. `_extract_features_and_target`, `_extract_features_single`)
- Constants: `UPPER_SNAKE_CASE` for true constants; format-specific defaults are set in `__init__` with conditional logic keyed on `format_type`
- Variables: `snake_case`, descriptive names (e.g. `decay_weight`, `wicket_pressure`, `second_innings_states`)

### File Organisation

- **Source code**: `src/` with subpackages `data_processing/`, `features/`, `models/`, `simulation/`, `betting/`, `utils/`
- **Tests & validation**: `tests/` — `regression_test.py`/`sim_baseline_metrics.py` are regression gates against locked baselines, not unit tests; most other files here are one-shot validation/experiment scripts
- **Sweep scripts**: `tests/` — named `{format}_{feature}_sweep.py`
- **Automation**: `scripts/` — scheduled import/retrain, one-off measurement/gate scripts
- **Static web assets**: `static/` with `index.html`, `js/`, `css/`
- **Documentation**: `docs/` for system-level docs, `markdowns/` for build plans and upgrade analysis (organised by area, e.g. `markdowns/win_predictor/`)
- **Cached data**: `data/processed/` — parquet feature caches and pickled model files, one set per format (see `.agents/project-overview.md`)

### Model Code Conventions

- Feature extraction is split into two methods:
  - `_extract_features_and_target(df)` — vectorised, for training on full DataFrames
  - `_extract_features_single(state: pd.Series)` — scalar, for single-state prediction in the Flask app
- **`predict_batch(states_df)`** — vectorised bulk prediction that handles boundaries, fold routing, and feature extraction in batch. Use this for validation/sweep scripts, never `iterrows()` + single-state `predict()`.
- **Both `_extract_features_and_target` and `_extract_features_single` must produce features in the same column order** — XGBoost's `colsample_bytree` is sensitive to column order under a fixed random seed
- Format-specific defaults are set in `__init__` via `if self.format_type in (...): ... elif ...: ... else: ...` blocks, or read directly from `spec(format_type)` (`src/data_processing/format_spec.py`) when the value is purely geometric (ball counts, phase edges) rather than an empirically-fit constant
- Format-specific features (e.g. od/odw's DLS resource %) are guarded with `if self.format_type in ("od", "odw"):` conditionals
- Models are serialised via `pickle.dump()` to `.pkl` files in `data/processed/`

## Data Integrity Rules

### No Data Leakage

This is the single most important rule in the codebase:

1. **Chronological feature computation**: Elo ratings, venue stats, and player Elo are computed in strict date order. A match's pre-match features only use data from prior matches.
2. **Group K-Fold**: all balls from the same `match_id` are in the same fold. Never split a match across folds.
3. **Pre-match recording**: the Elo engine records ratings BEFORE updating them with the match result.
4. **Validation uses fold models**: predictions use the fold model that was NOT trained on that match.
5. **Final model for live predictions**: a 6th model trained on 100% of data is used only for new/unseen matches (Flask app, live scanner).

### Gender Filtering — per format, not hardcoded

There is no blanket "male-only" rule. Every format has a `gender` key in `FORMAT_SPECS` (`t20`/`od`/`hnd` → `"male"`; `t20w`/`odw`/`hndw` → `"female"`). Filter with:

```python
meta_df = meta_df[meta_df["gender"] == spec(format_type)["gender"]].copy()
deliveries = deliveries[deliveries["match_id"].isin(meta_df["match_id"])].copy()
```

Never hardcode `== "male"`. Several stale scripts in this repo still do (a known cleanup backlog — check before copying a pattern from an unfamiliar test file). **Women's and men's data must never mix** — no shared Elo pool, par score, or training corpus, ever, even incidentally. Pooling/seeding (see below) is gated **within gender only**, enforced by an assertion in `format_spec.py`.

### Pooling / Seeding for thin corpora

`hnd` pools `t20`'s second-innings states at train time (never held out, never scored) and seeds player Elo from a frozen `t20` snapshot; `hndw` mirrors this pooling `t20w`. Governed by `POOL_SOURCE`/`SEED_SOURCE` dicts in `format_spec.py` — add a new pooled format there, not via a hardcoded `if format_type == "hnd"` check at each call site.

### Cache Management

- Caches are validated by a **content hash of their source `.py` file(s)**, not a manual delete (`src/utils/cache_manager.py`). Editing an engine file automatically invalidates every format's cache built from it on next run.
- **Gotcha**: a pre-existing cache with no manifest entry is trusted without verification on first sight (a "grandfather" clause), so a cache can silently drift out of sync with its source code for a long time if nothing else touches that file's hash. If you suspect this, force a rebuild (`engine.run(..., force=True)`) and diff against the cached parquet before trusting either.
- **Model `.pkl` files have NO hash check at all** — `fit(cache_path=...)` just checks `Path(cache_path).exists()`. To force a genuine retrain, delete (back up first) the `.pkl`, don't rely on touching the source file.
- Cache files: `{format}_elo_ratings.parquet`, `{format}_phase_elo_ratings.parquet`, `{format}_venue_features.parquet`, `{format}_player_elo_ratings.parquet`, `{format}_xgb_models.pkl`, `{format}_first_innings_regressor.pkl`, `{format}_delivery_outcome_model.pkl`
