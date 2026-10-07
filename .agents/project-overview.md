# Project Overview — T20 Win Probability Model

## What This Project Is

A **machine learning cricket win-probability and Monte Carlo simulation system**, covering **six formats**: men's and women's T20 (`t20`, `t20w`), ODI (`od`, `odw`), and The Hundred (`hnd`, `hndw`). It predicts:
- Ball-by-ball win probability during the second innings (chase)
- First-innings final score projections
- Full-match Monte Carlo simulations (score distributions, player props, betting markets)

Served through a **Flask web application** with a glassmorphism-styled single-page frontend, plus a betting/live-scoring subsystem that prices real fixtures from ESPN's live scoreboard.

Women's formats are **separate format codes, not a gender flag** — `t20w`/`odw`/`hndw` have their own directories, Elo pools, par scores, and trained models, structurally isolated from the men's data. See `markdowns/win_predictor/womens_games_ingestion_plan.md` for why.

## Data Source

All data comes from [Cricsheet.org](https://cricsheet.org). Current corpus (as of 2026-08-07):

| Format | Matches | Deliveries |
|---|---:|---:|
| `t20` | 10,634 | 2,936,761 |
| `od` | 4,125 | 2,640,385 |
| `hnd` | 167 | 31,952 |
| `t20w` | 3,200 | 728,751 |
| `odw` | 929 | 487,315 |
| `hndw` | 155 | 30,445 |

Raw YAMLs live in `data/raw/<format>/<competition>/`. `hndw`'s corpus is thin enough that its win-probability model pools `t20w`'s second-innings states at train time (never held out/scored) and seeds player Elo from a frozen `t20w` snapshot — same mechanism `hnd` uses pooling `t20`.

## Architecture (Pipeline)

```
YAML Files → CricketDataLoader → MatchMetadataExtractor
                                        ↓
              EloEngine + VenueEngine + PlayerEloEngine + PhaseEloEngine
                                        ↓
                                   StateBuilder
                                        ↓
                    XGBWinProbabilityModel (2nd-innings classifier)
                    FirstInningsPredictor  (1st-innings regressor)
                    DeliveryOutcomeModel   (per-ball outcome model, feeds Monte Carlo)
                                        ↓
                    match_simulator.py — Monte Carlo full-match simulation
                                        ↓
              Flask API (app.py) — win-prob / first-innings / MC-sim / live / betting
                                        ↓
                              Web UI (static/)
```

`src/data_processing/format_spec.py` is the single source of truth for per-format geometry (`FORMAT_SPECS`), which formats are trained+served (`ACTIVE_FORMATS`), and the pooling/seeding relationships (`POOL_SOURCE`, `SEED_SOURCE`). Every engine and model class takes a `format_type` string and reads its own defaults from `spec(format_type)` rather than hardcoding per-format branches wherever possible.

## Key Source Files

| File | Purpose |
|------|---------|
| `src/data_processing/format_spec.py` | Per-format geometry, `ACTIVE_FORMATS`, `POOL_SOURCE`/`SEED_SOURCE`, live-ESPN format classification |
| `src/data_processing/load_data.py` | Parses YAML files into a deliveries DataFrame, caches to parquet |
| `src/data_processing/match_metadata.py` | Extracts match-level metadata (teams, result, venue, gender, playing XIs) |
| `src/data_processing/cricsheet_importer.py` | Import from Cricsheet zips, gender routing, `retrain_full()` (the production retrain entry point) |
| `src/data_processing/build_states.py` | Builds ball-by-ball game states, merges all features, phase labelling |
| `src/features/elo_engine.py` | Chronological rolling **team** Elo (dynamic K-factor, margin scaling, decay, home advantage) |
| `src/features/venue_engine.py` | Rolling venue-level features (avg first-innings score, chase success rate) with Bayesian shrinkage |
| `src/features/player_elo_engine.py` | Individual player Elo for batters/bowlers, squad-strength aggregation, seed/stretch mechanisms |
| `src/features/phase_elo_engine.py` | Per-phase (powerplay/middle/death) rolling team Elo |
| `src/models/xgb_win_probability.py` | XGBoost classifier for 2nd-innings win probability, Group K-Fold CV, per-format hyperparameters, pooling support |
| `src/models/first_innings_predictor.py` | XGBoost regressor for 1st-innings final score |
| `src/models/delivery_outcome_model.py` | Per-ball outcome model (extras-gate + legal-outcome), the Monte Carlo engine's core |
| `src/simulation/match_simulator.py` | Full two-innings Monte Carlo simulation, batter/bowler props |
| `src/betting/pre_match_predictor.py` | Pre-match win-prob for live fixtures (all 6 formats), team Elo/venue lookups |
| `src/betting/live_scorecard.py`, `live_state_builder.py`, `in_play_scanner.py` | Live-match state tracking and in-play pricing (t20/od scope for the automated scanner) |
| `src/betting/bet_tracker.py`, `scanner.py` | Bet logging and pre-match value scanning |
| `app.py` | Flask web server — dict-based model registry keyed on `ACTIVE_FORMATS`, serves all APIs and static UI |
| `static/index.html` | Single-page web UI: Live Simulator, First Innings, Match Explorer (incl. Monte Carlo/Lineups sub-panels), Player/Team Data, Betting |
| `static/js/app.js`, `static/js/formats.js` | Frontend JS logic, per-format UI constants |

## Key Test/Validation Files

| File | Purpose |
|------|---------|
| `tests/regression_test.py` | **The** win-probability + first-innings regression gate, run against `baseline_metrics.json`. `--generate` locks a fresh full baseline; `--generate-format <fmt>` locks one format only. This replaced the older per-format `*_validation.py` scripts. |
| `tests/sim_baseline_metrics.py` | OOF log-loss baseline for the delivery-outcome (Monte Carlo) model, one entry per format in `sim_baseline_metrics.json`. `--generate --formats <fmt...>` locks/refreshes. |
| `scripts/import_and_retrain.py` | Scheduled import + retrain (Windows Task Scheduler), independent per-format retrain-due clocks |
| `scripts/hnd_winprob_phase_gate.py`, `measure_player_elo_distribution.py`, `measure_venue_impact.py` | One-off measurement/gate scripts used during specific tuning phases |

## Model Performance (locked baselines, as of 2026-08-07)

| Format | Win-Prob Brier | First-Innings MAE |
|---|---:|---:|
| `t20` | **0.127573** | **16.80** runs |
| `od` | **0.135169** | **31.13** runs |
| `hnd` | **0.139339** (pooled from `t20`) | **20.44** runs |
| `t20w` | **0.106491** | **15.09** runs |
| `odw` | **0.131821** | **31.05** runs |
| `hndw` | **0.143180** (pooled from `t20w`) | **16.06** runs |

t20/od moved from the prior lock (0.127522/0.134128) via `frontline_bowler_eligibility_plan.md`'s
bowler-rank-key fix (see `docs/system_guide.md`'s "Frontline-bowler eligibility" section) — a
real, correctness-driven cost kept within tolerance, not a regression. hnd/t20w/odw/hndw's small
moves are pure corpus drift since the prior lock (confirmed byte-identical on the eligibility
fix itself), not caused by that change.

Delivery-outcome (Monte Carlo engine) OOF log-loss is tracked separately in `data/processed/sim_baseline_metrics.json` (extras-gate / legal-outcome, per format). **All six formats are locked** as of 2026-08-14 — `t20w` was the last gap (extras 0.226622 / legal 1.318837, n=728,505) and is now gated like the rest.

The Monte Carlo *simulator's* own calibration (innings-total bias/MAE/coverage, distinct from the
per-ball log-loss above) is locked in `data/processed/mc_calibration_metrics{,_od,...}.json`. `od`
was re-locked 2026-08-10 (`mc_runrate_feedback_plan.md`, Phase 1's `SCORE_FEEDBACK_SHRINK=0.30`
damping fix for the `current_team_score` feedback loop — see `docs/system_guide.md` section 15):
bias +18.17 → **+12.72** (clears its ≤13.0 gate). `t20` is unaffected (`SCORE_FEEDBACK_SHRINK=1.0`,
a byte-identical no-op) and keeps its prior lock.

**t20's MAE gate was re-anchored 25.0 → 28.5 on 2026-08-11** (`t20_mae_gate_plan.md` Phase 4,
full writeup in `docs/system_guide.md` §15): the honest ceiling on data this repo holds is 26.29
OOF MAE (best leakage-safe regressor on every pre-match feature available), so 25.0 was never
attainable. Median-based MAE point estimator (Phase 1) also shipped, moving the locked value
27.5735 → 27.48. A squad-context retrain was priced (Phase 2, subsampled OOF ablation) and found
**not justified** (no candidate feature set cleared `|t|>2` at n=800 matches) — no retrain was
performed. `_T20_MAE_MAX` is every other format's MAE-gate anchor too (dispersion-based
`innings_total_sd` scaling), so this moved every format's MAE gate at once, though only `t20`'s
and `od`'s snapshots currently exist on disk to reflect it (`hnd`/`t20w`/`odw`/`hndw` have no
`mc_calibration_metrics_<fmt>.json` yet — this battery has never been run for them).

## Cached Data & Models

Everything in `data/processed/` is named `{format}_<artifact>`:
- `{format}_cricket_data.parquet` — parsed deliveries
- `{format}_match_metadata.parquet` — match summaries (incl. `gender` column)
- `{format}_elo_ratings.parquet` — team Elo
- `{format}_phase_elo_ratings.parquet` — per-phase team Elo
- `{format}_player_elo_ratings.parquet` — player Elo
- `{format}_venue_features.parquet` / `_venue_lookup.parquet` — venue stats
- `{format}_team_venue_dist.parquet` — team-venue distributions (live scanner)
- `{format}_xgb_models.pkl` — win-prob fold models + final model
- `{format}_first_innings_regressor.pkl` — first-innings regressor
- `{format}_delivery_outcome_model.pkl` — Monte Carlo delivery model
- `_{target}_seed_from_{source}.parquet` — frozen player-Elo seed snapshots (e.g. `_hnd_seed_from_t20.parquet`, `_hndw_seed_from_t20w.parquet`)

**Cache validity is hash-based** (`src/utils/cache_manager.py`), keyed on the source `.py` file(s) that produce each cache — editing an engine file invalidates that cache for *every* format that uses it, forcing a rebuild on next run. **Gotcha**: a pre-existing cache file with no manifest entry is "grandfathered in" (trusted without verification) rather than rebuilt — this let `t20`'s and `hnd`'s team-Elo caches silently drift out of sync with `elo_engine.py`'s actual code for a long time before being caught (2026-08-07, see `markdowns/win_predictor/womens_games_ingestion_plan.md`). If a regression check fails for a format your edit "shouldn't" have touched, check whether that format's cache was already stale before assuming your edit is buggy.

## Gender / Format Handling

There is **no blanket "male-only" rule**. Every engine and model reads its required gender via `spec(format_type)["gender"]` and filters `meta_df`/`deliveries` accordingly — this must be threaded through any new script that builds a training corpus (`meta_df = meta_df[meta_df["gender"] == spec(format_type)["gender"]]`), not hardcoded to `"male"`. Women's and men's data must never share an Elo pool, par score, or training corpus — this is enforced structurally by separate `data/raw/<fmt>/` directories, not just the gender filter.

## Feature Engineering Highlights

- **Phase-based Elo decay**: team and player Elo features decay from full strength in the powerplay to zero in the death overs
- **Residual player features**: `residual_batter_elo = (batter_elo - team_elo) × decay_weight` — individual quality orthogonal to team strength
- **Effective Elo**: squad-strength-weighted blend of team Elo with batting/bowling squad composition (`compute_effective_elo`)
- **Pressure index**: `(required_run_rate / baseline) × (1 + wicket_pressure / divisor)`
- **Group K-Fold**: all balls from the same match stay in the same fold
- **Bayesian shrinkage**: venue stats and player-Elo cold-start handling
- **od/odw-specific features**: DLS resource percentage approximation, partnership runs, recent runs/wickets (rolling window)
- **Pooling/seeding for thin corpora**: `hnd` pools `t20`, `hndw` pools `t20w` — extra never-held-out training rows plus a frozen player-Elo seed snapshot, gated via `POOL_SOURCE`/`SEED_SOURCE` in `format_spec.py`
