# Project Overview — CreasePulse (T20 & ODI Win Probability Model)

## What This Project Is

CreasePulse is a **machine learning-powered cricket win probability prediction system**. It predicts the probability of the chasing (batting second) team winning a cricket match, ball-by-ball, during the second innings of T20 and ODI games. It also predicts first innings final scores.

The system is served through a **Flask web application** with a glassmorphism-styled single-page frontend.

## Data Source

All data comes from [Cricsheet.org](https://cricsheet.org) — 11,592 T20 YAML files across 17 competitions and 5,967 ODI YAML files across 5 competitions, stored in `data/raw/t20/` and `data/raw/od/`.

## Architecture (9-Stage Pipeline)

```
YAML Files → CricketDataLoader → MatchMetadataExtractor
                                        ↓
                              EloEngine + VenueEngine + PlayerEloEngine
                                        ↓
                                   StateBuilder
                                        ↓
                            XGBWinProbabilityModel (Classifier)
                            FirstInningsPredictor  (Regressor)
                                        ↓
                              Flask API (app.py)
                                        ↓
                              Web UI (static/)
```

## Key Source Files

| File | Purpose |
|------|---------|
| `src/data_processing/load_data.py` | Parses YAML files into a deliveries DataFrame, caches to parquet |
| `src/data_processing/match_metadata.py` | Extracts match-level metadata (teams, result, venue, playing XIs) |
| `src/data_processing/build_states.py` | Builds ball-by-ball game states with cumulative stats and merges all features |
| `src/features/elo_engine.py` | Chronological rolling team Elo ratings with dynamic K-factor, margin scaling, inactivity decay, home advantage |
| `src/features/venue_engine.py` | Rolling venue-level features (avg first innings score, chase success rate) with Bayesian shrinkage |
| `src/features/player_elo_engine.py` | Individual player Elo ratings for batters/bowlers with Bayesian shrinkage toward team Elo |
| `src/models/xgb_win_probability.py` | XGBoost classifier for win probability with 33 features (T20) or 37 features (ODI), Group K-Fold CV |
| `src/models/first_innings_predictor.py` | XGBoost regressor for predicting first innings final score |
| `app.py` | Flask web server — loads models, serves API endpoints and static UI |
| `static/index.html` | Single-page web UI with 5 tabs (Live Predictor, First Innings, Match Explorer, Scenario Explorer, Model Analytics) |
| `static/js/app.js` | Frontend JS logic with Chart.js visualisations |
| `static/css/style.css` | Dark-mode glassmorphism CSS styling |

## Key Test/Validation Files

| File | Purpose |
|------|---------|
| `tests/win_probability_validation.py` | T20 win probability model training, validation, and Brier score reporting |
| `tests/odi_validation.py` | ODI model training, validation, and Brier score reporting |
| `tests/first_innings_validation.py` | T20 first innings regressor training and validation |

## Model Performance (as of July 2026)

| Model | Brier Score | Notes |
|-------|------------|-------|
| T20 Win Probability | **0.1318** | Men-only data, 33 features, 5-fold GroupKFold |
| ODI Win Probability | **0.1324** | Men-only data, 37 features (includes DLS/momentum), 5-fold GroupKFold |
| T20 First Innings | MAE: **16.64 runs** | |
| ODI First Innings | MAE: **31.49 runs** | |

## Cached Data & Models

All intermediate outputs are cached in `data/processed/`:
- `{format}_cricket_data.parquet` — parsed deliveries
- `{format}_match_metadata.parquet` — match summaries
- `{format}_elo_ratings.parquet` — team Elo ratings
- `{format}_venue_features.parquet` — venue statistics
- `{format}_player_elo_ratings.parquet` — player Elo ratings
- `{format}_xgb_models.pkl` — trained XGBoost fold models + final model
- `{format}_first_innings_regressor.pkl` — trained first innings regressors

The Flask app loads models from `t20_xgb_models.pkl`, `od_xgb_models.pkl`, `t20_first_innings_regressor.pkl`, and `od_first_innings_regressor.pkl`.

## Data Filtering

Both T20 and ODI models are trained exclusively on **male matches**. The `gender` column from match metadata is used to filter out women's matches. This was a critical improvement that reduced T20 Brier Score from 0.1263 to 0.1171 (later refined further).

## Feature Engineering Highlights

- **Phase-based Elo decay**: Team and player Elo features decay from full strength in the Powerplay to zero in the Death overs
- **Residual player features**: `residual_batter_elo = (batter_elo - team_elo) × decay_weight` — captures individual quality orthogonal to team strength
- **Pressure index**: `(required_run_rate / baseline) × (1 + wicket_pressure / divisor)` — combines run rate pressure with wicket situational pressure
- **Group K-Fold**: All balls from the same match are in the same fold to prevent data leakage
- **Bayesian shrinkage**: Used in venue stats and player Elo cold-start handling
- **ODI-specific features**: DLS resource percentage approximation, partnership runs, recent runs/wickets (rolling window)
