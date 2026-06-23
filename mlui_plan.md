# T20 Win Probability Model — Implementation Plan (Phase 3)

> **Created:** 2026-06-22  
> **Status:** In Progress  
> **Current Brier Score:** 0.1513  
> **Target Brier Score:** < 0.10

---

## 1. Upgrade Impact Summary (Phase 1 & 2)

All upgrades from the original `upgrade_plan.md` have been completed. The cumulative results are summarised below.

| Upgrade | Brier Score | Δ (Actual) | Δ (Projected) | % Achieved |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline MVP** | **0.1548** | — | — | — |
| + 2A (Variance Scaling) | 0.1544 | -0.0004 | -0.005 to -0.015 | 4–8% |
| + 2B (Distance Weighting) | 0.1544 | 0.0000 | -0.005 to -0.012 | 0% |
| + 2C (Optimal k) | 0.1543 | -0.0001 | -0.002 to -0.008 | 1–5% |
| + 2D (Phase Awareness) | 0.1538 | -0.0005 | -0.008 to -0.020 | 2.5–6% |
| + 2E (Interactions) | 0.1526 | -0.0012 | -0.005 to -0.015 | 8–24% |
| + 2F (Learned Weights) | 0.1516 | -0.0010 | -0.005 to -0.015 | 7–20% |
| + 2G (Cricket-Aware) | 0.1513 | -0.0003 | -0.005 to -0.012 | 2.5–6% |
| **Total** | **0.1513** | **-0.0035** | **-0.035 to -0.097** | **3.6–10%** |

### Why the Upgrades Underperformed

1. **Data is the bottleneck, not the distance metric.** The ~161k second-innings states, when sliced by phase/wickets/score, leave only a few dozen truly "close" neighbours. The hand-tuned constants (50, 30, 2) were surprisingly close to the actual standard deviations (38, 34, 2.5), so 2A barely moved the needle.

2. **Distance weighting needs distance spread.** Upgrade 2B (Gaussian weighting) had zero impact because nearest neighbours are so tightly packed (distances ≈ 0) that all weights ≈ 1.0.

3. **Curse of dimensionality.** Each new feature stretches the distance space. With KNN, adding dimensions without proportionally more data makes every point more isolated, degrading neighbour quality.

4. **KNN has a hard ceiling.** KNN is non-parametric and memory-based — it cannot learn decision boundaries, feature interactions, or non-linear patterns beyond raw feature distances. It averages outcomes from similar situations but cannot generalise or extrapolate.

5. **Cricket is inherently noisy.** Identical match situations produce different outcomes due to pitch, team quality, player form, weather, and pressure. The irreducible noise floor (Bayes error) for T20 prediction without team/player context is estimated at 0.08–0.10.

---

## 2. Proposed Changes (Phase 3)

Each change below will be tested independently and its impact recorded. Only changes that improve the Brier score will be retained.

> **Rule:** Update the "Result" and "Status" columns after each test. Mark with ✅ (kept), ❌ (reverted), or 🔄 (in progress).

---

### 3A: Expand Dataset

**Priority:** 🔴 Critical  
**Difficulty:** Low  
**Expected Brier improvement:** -0.005 to -0.010  
**Status:** `[ ]` Not Started

#### Problem
The model trains on ~161k second-innings states from a limited set of Cricsheet YAML files. More data directly improves KNN quality and is a prerequisite for ML approaches.

#### Solution
Download the full Cricsheet archive (all men's T20Is + domestic T20 leagues). This could triple the dataset to ~500k+ states.

#### Result
| Metric | Before | After | Δ |
| :--- | :---: | :---: | :---: |
| Overall Brier | 0.1513 | — | — |
| Powerplay Brier | — | — | — |
| Middle Brier | — | — | — |
| Death Brier | — | — | — |
| Dataset Size | ~161k | — | — |

---

### 3B: Platt Scaling Calibration Layer

**Priority:** 🟡 Important  
**Difficulty:** Low  
**Expected Brier improvement:** -0.003 to -0.005  
**Status:** `[x]` Completed

#### Problem
The model was overconfident at extremes and underconfident at low/middle probabilities, with some edge case issues (e.g. predicting < 1.0 even after the match was won).

#### Solution
Apply Platt Scaling on Log-Odds (Logit). Fit a logistic regression model on the logit-transformed raw probabilities over a separate 500-state calibration dataset, and use cached `calibration_model.pkl`. Exact logical boundary conditions (`runs_remaining <= 0`, `wickets_lost >= 10`, `balls_remaining <= 0`) bypass the calibration layer.

```python
# Convert to logit space
eps = 1e-4
probs_clipped = np.clip(raw_probs, eps, 1 - eps)
logits = np.log(probs_clipped / (1 - probs_clipped))

# Fit Platt scaling
lr = LogisticRegression()
lr.fit(logits.reshape(-1, 1), actuals)
```

#### Result
| Metric | Before | After | Δ |
| :--- | :---: | :---: | :---: |
| Overall Brier | 0.1502 | 0.1487 | -0.0015 |
| Powerplay Brier | 0.2032 | 0.2009 | -0.0023 |
| Middle Brier | 0.1280 | 0.1283 | +0.0003 |
| Death Brier | 0.0740 | 0.0664 | -0.0076 |
| Edge Case (runs <= 0) | 0.8233 | 1.0000 | +0.1767 |


---

### 3C: Gradient Boosted Trees (XGBoost / LightGBM)

**Priority:** 🔴 Critical  
**Difficulty:** Medium  
**Expected Brier improvement:** -0.020 to -0.040  
**Status:** `[ ]` Not Started

#### Problem
KNN cannot learn decision boundaries or feature interactions. A supervised classifier can learn these from the data directly.

#### Solution
Train an XGBoost binary classifier on the same feature set:
- Input: `runs_remaining`, `balls_remaining`, `wickets_lost`, `required_run_rate`, `phase` (encoded), `pressure_index`
- Target: `won_match` (0/1)
- Use 5-fold cross-validation stratified by `match_id` to prevent data leakage.

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    objective='binary:logistic',
    eval_metric='logloss',
    early_stopping_rounds=50,
)
```

#### Result
| Metric | KNN Baseline | XGBoost | Δ |
| :--- | :---: | :---: | :---: |
| Overall Brier | 0.1513 | — | — |
| Powerplay Brier | — | — | — |
| Middle Brier | — | — | — |
| Death Brier | — | — | — |
| Log Loss | — | — | — |

---

### 3D: Team Strength Features

**Priority:** 🟡 Important  
**Difficulty:** Medium  
**Expected Brier improvement:** -0.005 to -0.010  
**Status:** `[ ]` Not Started

#### Problem
The model treats all teams equally. In reality, Australia chasing 180 is a different proposition to a minnow chasing 180.

#### Solution
Add a rolling team Elo rating or historical win rate as a feature. Calculate from the dataset itself (prior matches).

#### Result
| Metric | Before | After | Δ |
| :--- | :---: | :---: | :---: |
| Overall Brier | — | — | — |

---

### 3E: Venue / Conditions Features

**Priority:** 🟢 Nice to Have  
**Difficulty:** Medium  
**Expected Brier improvement:** -0.002 to -0.005  
**Status:** `[ ]` Not Started

#### Problem
Some grounds have higher average scores, different dew effects, and pitch behaviour that systematically affects chasing outcomes.

#### Solution
Add venue-level average first-innings score, chase success rate, and (if available) day/night indicator.

#### Result
| Metric | Before | After | Δ |
| :--- | :---: | :---: | :---: |
| Overall Brier | — | — | — |

---

### 3F: Neural Network (LSTM / Transformer)

**Priority:** 🔵 Future  
**Difficulty:** High  
**Expected Brier improvement:** -0.030 to -0.050  
**Status:** `[ ]` Not Started

#### Problem
KNN and tree models treat each ball independently. A sequence model can learn temporal patterns — momentum shifts, batting collapses, death-overs acceleration — from the full ball-by-ball trajectory.

#### Solution
Train an LSTM or Transformer model on sequences of ball-by-ball states within an innings, predicting `won_match` at each timestep.

#### Result
| Metric | Best Previous | Neural Net | Δ |
| :--- | :---: | :---: | :---: |
| Overall Brier | — | — | — |

---

## 3. Professional Web UI

**Status:** `[ ]` Not Started

### Architecture
```
Browser (HTML/CSS/JS + Chart.js)
       │  REST API (JSON)
Flask Backend (Python)
       │
Existing Model (CricketDataLoader → StateBuilder → SimilarityEngine → WinProbabilityModel)
```

### Features
| Tab | Description | Status |
| :--- | :--- | :---: |
| **Live Prediction** | Input runs/balls/wickets/over → animated win probability gauge | `[ ]` |
| **Ball-by-Ball** | Line chart of win probability across a match timeline | `[ ]` |
| **Phase Breakdown** | Bar chart showing accuracy by Powerplay / Middle / Death | `[ ]` |
| **Scenario Explorer** | Sliders for runs/balls/wickets with live probability updates | `[ ]` |

### Design
- Dark theme with glassmorphism cards
- Vibrant gradient accents (teal → purple)
- Chart.js for all visualisations
- Google Fonts (Inter, Outfit)
- Fully responsive

### Files to Create
- `app.py` — Flask server with API endpoints
- `static/index.html` — Single-page application
- `static/css/style.css` — Design system and styling
- `static/js/app.js` — Client-side logic and charts

---

## 4. Recommended Implementation Order

```
Phase 3A — Quick Wins
├── 3A  Expand dataset                    ← More data helps everything
├── 3B  Isotonic calibration              ← Low effort, fixes known bias
└── UI  Build the web interface           ← Can run in parallel

Phase 3B — Model Upgrade
├── 3C  XGBoost / LightGBM               ← Biggest expected gain
└── 3D  Team strength features            ← Meaningful context

Phase 3C — Advanced (if needed)
├── 3E  Venue / conditions features
└── 3F  Neural network (LSTM/Transformer)
```

---

## 5. Progress Log

Record each experiment here as it is completed.

| Date | Change | Brier Before | Brier After | Δ | Decision |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 2026-06-23 | 3B Platt Scaling Calibration | 0.1502 | 0.1487 | -0.0015 | ✅ Kept |

---


## 6. Current Best Model Configuration

```python
SimpleSimilarityEngine(
    k=100,
    phase_mode="filter",
    use_pressure=True,
    use_resources=False,
    w_r=0.3,
    w_b=2.2,
    w_w=1.1,
    w_p=1.1,
)
# + Exponential Wickets (use_exponential_wickets=True) — from Upgrade 2G
# Brier Score: 0.1513
```
