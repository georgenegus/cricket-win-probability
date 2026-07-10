# Instructions for Working on This Project

## Before Making Any Changes

1. **Read the project overview** at `.agents/project-overview.md` to understand the system architecture.
2. **Read the coding standards** at `.agents/coding-standards.md` for conventions and rules.
3. **Read the system guide** at `docs/system_guide.md` for a deep technical walkthrough.
4. **Check `markdowns/upgrades/`** for existing analysis and upgrade plans before proposing new approaches — many ideas have been explored and documented there.

## Critical Rules

### Data Leakage Prevention (HIGHEST PRIORITY)

Data leakage is the single most damaging mistake in this codebase. Always ensure:

- **All feature engines** (Elo, Venue, Player Elo) compute features **chronologically** — only using past match data
- **Group K-Fold**: All deliveries from the same `match_id` stay in the same fold — never split a match
- **Pre-match recording**: Elo ratings are recorded BEFORE the match result updates them
- When adding new features, verify they use **only information available at the time of that ball** — no future information

### Feature Column Order Matters

XGBoost's `colsample_bytree` with a fixed `random_state` is sensitive to column order. Both `_extract_features_and_target()` and `_extract_features_single()` **must produce features in exactly the same column order**. When adding or reordering features, update BOTH methods identically.

### Gender Filtering

All models are trained on **male-only data**. This filtering happens in the validation scripts (`tests/win_probability_validation.py`, `tests/odi_validation.py`), not in the loaders. When creating new scripts or pipelines, always filter `meta_df = meta_df[meta_df['gender'] == 'male']` and propagate to deliveries.

### Format Isolation

T20 and ODI models have different feature sets. ODI has extra features (DLS resource %, partnership runs, recent runs/wickets, recent run rate). These are guarded with `if self.format_type == "od":` conditionals. When modifying features:
- If T20-only: guard with `if self.format_type in ["t20"]:`
- If ODI-only: guard with `if self.format_type == "od":`
- If both: guard with `if self.format_type in ["t20", "od"]:`
- Always verify the OTHER format still works after changes

### Computational Efficiency

- **Always write the most computationally efficient code.** This is imperative for scripts such as the sweeps which are running a large number of combinations to find the optimal value of a parameter.

### Sweep Script Best Practices

When writing parameter sweep scripts (e.g. `tests/t20_player_form_sweep.py`), follow these patterns to minimise per-iteration time:

1. **Pre-aggregate deliveries once**: Call `PlayerEloEngine._aggregate_match_stats(deliveries)` before the sweep loop and pass the result via `precomputed_stats=` to each iteration
2. **Cache partial states**: Build the base state DataFrame once (without player Elo columns) and re-merge only the changing columns each iteration
3. **Use `predict_batch()`**: Never use `iterrows()` + `predict()` for bulk validation — use `model.predict_batch(states_df)` for vectorised prediction
4. **Skip final model**: Pass `train_final_model=False` to `model.fit()` since sweeps only need fold models for Brier calculation

## Workflow for Model Improvements

### 1. Hypothesis & Analysis

- State what feature/change you expect to improve and why
- Check if this has been tried before in `markdowns/upgrades/`
- If it's a parameterised change, create a sweep script in `tests/`

### 2. Implementation

- Modify `src/models/xgb_win_probability.py` (both `_extract_features_and_target` and `_extract_features_single`)
- If the change requires new data, modify `src/data_processing/build_states.py` or the relevant feature engine
- Guard format-specific features appropriately

### 3. Validation

- **Delete cached models**: Remove `data/processed/xgb_models.pkl` (and `od_xgb_models.pkl` for ODI)
- **Run validation**: `py tests/win_probability_validation.py` (T20) or `py tests/odi_validation.py` (ODI)
- **Compare Brier score** against the established baseline:
  - T20 baseline: **0.1318**
  - ODI baseline: **0.1324**
- Also check per-phase Brier scores and calibration

### 4. Decision

- **If Brier improves or is neutral**: Keep the change
- **If Brier regresses**: Revert all changes. Even a +0.0001 regression should be reverted unless there's a strong structural argument.
- **Document the result** in the walkthrough or upgrade plan markdown

### 5. Deployment

If the change is kept:
1. Copy T20 models to app-compatible filenames:
   ```python
   shutil.copyfile('data/processed/xgb_models.pkl', 'data/processed/t20_xgb_models.pkl')
   shutil.copyfile('data/processed/first_innings_regressor.pkl', 'data/processed/t20_first_innings_regressor.pkl')
   ```
2. Verify the Flask app starts successfully: `py app.py`
3. If the improvement also applies to ODI, implement and validate for ODI separately

## Running the Project

### Validation Scripts
```bash
# T20 Win Probability (trains model + validates)
py tests/win_probability_validation.py

# ODI Win Probability (trains model + validates)
py tests/odi_validation.py

# T20 First Innings Predictor
py tests/first_innings_validation.py
```

### Flask Web App
```bash
py app.py
# Serves on http://127.0.0.1:5000
```

### Main Pipeline (diagnostic — does NOT train models)
```bash
py main.py
```

## Key Hyperparameters

### T20 XGBoost Classifier
- `n_estimators=300`, `max_depth=4`, `learning_rate=0.08`
- `min_child_weight=10`, `subsample=0.8`, `colsample_bytree=0.8`
- `reg_alpha=1.0`, `reg_lambda=3.0`

### ODI XGBoost Classifier
- `n_estimators=339`, `max_depth=4`, `learning_rate=0.02`
- `min_child_weight=10`, `subsample=0.7`, `colsample_bytree=0.7`
- `reg_alpha=0.0`, `reg_lambda=1.0`

### Elo Engine (T20)
- `k_factor=35`, `decay_days=90`, `decay_rate=0.95`
- `margin_runs_coef=0.5`, `margin_runs_div=10`, `margin_wickets_coef=0.1`
- Dynamic K: `k_max=50`, `k_min=20`, `k_scale=30`

### Elo Engine (ODI)
- `k_factor=25`, `decay_days=120`, `decay_rate=0.97`
- `margin_runs_coef=0.3`, `margin_runs_div=25`, `margin_wickets_coef=0.08`
- Dynamic K: `k_max=35`, `k_min=25`, `k_scale=30`

### Player Elo Engine
- `k_runs=0.5`, `k_wickets=10.0`, `decay_days=180`, `decay_rate=0.95`
- `min_matches_reliable=10` (Bayesian shrinkage threshold)

## Common Pitfalls

1. **Forgetting to clear caches**: If you change features or model parameters but the cached `.pkl` model still exists, the old model will be loaded instead of retraining. Always delete relevant caches.

2. **Feature mismatch between training and prediction**: `_extract_features_and_target()` and `_extract_features_single()` must produce identical columns in the same order. A mismatch causes silent accuracy degradation or XGBoost feature name errors.

3. **Testing on the training set**: Always use the fold model that was NOT trained on a given match. The `match_to_fold` mapping handles this automatically.

4. **Mixing women's and men's data**: The model is trained on men's matches only. Including women's matches degrades T20 Brier from 0.1171 to 0.1263 due to structural differences in scoring patterns.

5. **Modifying one format without checking the other**: Always verify changes to shared code don't break the other format. Run both T20 and ODI validation.

6. **Hardcoding format-specific logic**: Use `self.format_type` checks instead of hardcoded values. The `__init__` method already handles format-specific defaults.

## Documentation Locations

| Document | Path | Purpose |
|----------|------|---------|
| System Guide | `docs/system_guide.md` | Full technical walkthrough for newcomers |
| Upgrade Plans | `markdowns/upgrades/*.md` | Historical upgrade proposals and analysis |
| This File | `.agents/instructions.md` | Working instructions for the AI agent |
| Project Overview | `.agents/project-overview.md` | Quick reference for architecture and file locations |
| Coding Standards | `.agents/coding-standards.md` | Code conventions and data integrity rules |
