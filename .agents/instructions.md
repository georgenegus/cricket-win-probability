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
