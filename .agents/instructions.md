# Instructions for Working on This Project

## Before Making Any Changes

1. **Read the project overview** at `.agents/project-overview.md` to understand the system architecture.
2. **Read the coding standards** at `.agents/coding-standards.md` for conventions and rules.
3. **Read the system guide** at `docs/system_guide.md` for a deep technical walkthrough.
4. **Check `markdowns/`** (organised by area, e.g. `markdowns/win_predictor/`) for existing analysis and build plans before proposing new approaches — many ideas have already been explored and documented, including per-phase results.

## Critical Rules

### Data Leakage Prevention (HIGHEST PRIORITY)

Data leakage is the single most damaging mistake in this codebase. Always ensure:

- **All feature engines** (Elo, Venue, Player Elo, Phase Elo) compute features **chronologically** — only using past match data
- **Group K-Fold**: all deliveries from the same `match_id` stay in the same fold — never split a match
- **Pre-match recording**: Elo ratings are recorded BEFORE the match result updates them
- When adding new features, verify they use **only information available at the time of that ball** — no future information

### Feature Column Order Matters

XGBoost's `colsample_bytree` with a fixed `random_state` is sensitive to column order. Both `_extract_features_and_target()` and `_extract_features_single()` **must produce features in exactly the same column order**. When adding or reordering features, update BOTH methods identically.

### Gender Filtering — per format, never hardcoded

Six formats are trained and served: `t20`, `od`, `hnd` (male), `t20w`, `odw`, `hndw` (female). There is **no blanket male-only rule**. Filter using `spec(format_type)["gender"]`, not a hardcoded `"male"` string:

```python
meta_df = meta_df[meta_df["gender"] == spec(format_type)["gender"]].copy()
```

Women's and men's data must never share an Elo pool, par score, or training corpus.

### Format Isolation

Different formats have different feature sets — ODI-family formats (`od`/`odw`) have extra features (DLS resource %, partnership runs, recent runs/wickets, recent run rate), guarded with `if self.format_type in ("od", "odw"):` conditionals. When modifying features:
- If one geometry-family only: guard with `if self.format_type in (...):` listing every format in that family (e.g. `("od", "odw")`, or `("hnd", "hndw")`)
- Prefer deriving from `spec(format_type)` (ball counts, phase edges) over a hardcoded per-format branch when the value is purely geometric
- Never let a new format silently fall through to `else` and inherit a constant that was **empirically fit to a different format's population** (positional batting weights, Elo-stretch factors, dynamic-K multipliers) — either measure it for the new format or use a documented conservative no-op default (uniform weights, stretch=1.0)
- Always verify every OTHER active format still works after changes (run `tests/regression_test.py`)

## Running the Project

### Regression / Validation
```bash
py tests/regression_test.py                        # check all ACTIVE_FORMATS against locked baseline_metrics.json
py tests/regression_test.py --generate              # relock the full baseline (all formats)
py tests/regression_test.py --generate-format hnd   # relock one format only, others untouched
py tests/sim_baseline_metrics.py --formats t20 od   # check delivery-outcome (Monte Carlo) model OOF log-loss
```

### Flask Web App
```bash
py app.py
# Serves on http://127.0.0.1:5000
```

### Full Model Retrain (production path)
```python
from src.data_processing.cricsheet_importer import retrain_full
retrain_full("hnd")   # trains first-innings regressor + win-prob classifier (+ pooling if applicable) + delivery-outcome model
```
`fit(cache_path=...)` only checks whether the `.pkl` file exists — **back up and delete it first** to force a genuine retrain, don't rely on source-code edits to invalidate it.

## Common Pitfalls

1. **Model `.pkl` caches have no content check**: if the `.pkl` exists, `fit()` loads it and returns — even if the features it would be trained on have changed. Delete (after backing up) to force retraining.
2. **Feature engine caches (`.parquet`) DO have a content check** (hash of the source `.py` file) but it can be fooled: a pre-existing cache with no manifest entry is trusted blindly on first sight rather than verified. If a format's cache seems inconsistent with current code, force a rebuild and diff.
3. **Feature mismatch between training and prediction**: `_extract_features_and_target()` and `_extract_features_single()` must produce identical columns in the same order. A mismatch causes silent accuracy degradation or XGBoost feature name errors.
4. **Testing on the training set**: always use the fold model that was NOT trained on a given match. The `match_to_fold` mapping (and `predict_batch`) handles this automatically.
5. **A new format silently inheriting a men's/other-format constant**: any `if format_type == "x": ... else: <default>` ladder is a trap for a new format that falls into `else` — audit these deliberately when adding a format, don't assume the default is safe.
6. **Modifying one format's branch without checking siblings**: always verify changes to shared code (any file under `src/features/` or `src/models/`) don't shift other formats' baselines. Run `tests/regression_test.py` — it iterates every `ACTIVE_FORMATS` entry.
7. **Hardcoding format-specific logic**: use `self.format_type` checks (or `spec(format_type)`) instead of hardcoded values.

## Documentation Locations

| Document | Path | Purpose |
|----------|------|---------|
| System Guide | `docs/system_guide.md` | Full technical walkthrough for newcomers |
| Build Plans | `markdowns/*/*.md` | Per-area build plans, phased implementation tracking, and results |
| This File | `.agents/instructions.md` | Working instructions for the AI agent |
| Project Overview | `.agents/project-overview.md` | Quick reference for architecture, file locations, current baselines |
| Coding Standards | `.agents/coding-standards.md` | Code conventions and data integrity rules |
