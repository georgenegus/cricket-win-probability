# T20 Win Probability Model — Upgrade Plan

---

## 1. Current Architecture

### Overview

The model is a **historical state-based win probability engine** that estimates the likelihood of the chasing team winning at any ball during a T20 second innings. It uses a k-nearest-neighbour (KNN) similarity search over ~341,000 ball-by-ball deliveries parsed from Cricsheet YAML files.

### Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  YAML Files  ──►  DataLoader  ──►  Parquet Cache                   │
│                                       │                            │
│                                  StateBuilder                      │
│                                       │                            │
│                              ┌────────┴────────┐                   │
│                              │  Match States    │                   │
│                              │  (2nd innings)   │                   │
│                              └────────┬────────┘                   │
│                                       │                            │
│                          SimilarityEngine (k=200)                  │
│                                       │                            │
│                          WinProbabilityModel                       │
│                            mean(won_match)                         │
│                                       │                            │
│                              Win Probability                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Module Inventory

| Module | File | Responsibility |
|--------|------|----------------|
| Data Loader | `src/data_processing/load_data.py` | Parse YAML, cache to parquet, dtype optimisation |
| State Builder | `src/data_processing/build_states.py` | Cumulative runs, wickets, balls, phases, match context |
| Similarity Engine | `src/models/similarity_engine.py` | Weighted Euclidean distance, k-nearest-neighbours |
| Rule-Based Similarity | `src/similarity/rule_based.py` | Tolerance-window filter (alternate engine, unused in validation) |
| Win Probability | `src/models/win_probability.py` | Aggregates similar states → mean win rate |
| Validation | `tests/win_probability_validation.py` | Brier score, calibration, directionality, edge cases |
| Timer | `src/data_processing/timer.py` | Performance instrumentation |

### Current Distance Metric

```python
distance = (rr / 50)² + (br / 30)² + (wk / 2)²
```

Where:
- `rr` = runs_remaining difference
- `br` = balls_remaining difference  
- `wk` = wickets_lost difference

Divisors (50, 30, 2) are hand-tuned normalisation constants. `k = 200` nearest neighbours are returned.

### Current Features Used

| Feature | Used in Distance | Used in State |
|---------|:----------------:|:-------------:|
| `runs_remaining` | ✅ | ✅ |
| `balls_remaining` | ✅ | ✅ |
| `wickets_lost` | ✅ | ✅ |
| `run_rate` | ❌ | ✅ |
| `required_run_rate` | ❌ | ✅ |
| `phase` | ❌ | ✅ |
| `target` | ❌ | ✅ |

---

## 2. Validation Findings

### Brier Score

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Brier Score** | **0.1548** | Solid MVP. A perfect model scores 0.0; naive 50/50 guessing scores 0.25. |

For reference:
- **< 0.10** — strong predictive model
- **0.10–0.15** — good, competitive
- **0.15–0.20** — reasonable baseline (← current)
- **> 0.20** — weak

### Calibration

| Predicted | Actual | Gap |
|-----------|--------|-----|
| 0.04 | 0.073 | +0.033 (underconfident) |
| 0.145 | 0.125 | -0.020 |
| 0.254 | 0.205 | -0.049 |
| 0.350 | 0.356 | +0.006 ✅ |
| 0.454 | 0.444 | -0.010 ✅ |
| 0.551 | 0.581 | +0.030 |
| 0.651 | 0.700 | +0.049 |
| 0.753 | 0.735 | -0.018 ✅ |
| 0.855 | 0.840 | -0.015 ✅ |
| 0.966 | 0.909 | -0.057 (overconfident) |

**Key observations:**
- Calibration is strongest in the 0.3–0.5 and 0.7–0.85 ranges
- The model is **underconfident at very low probabilities** (predicts 4%, actual 7%)
- The model is **overconfident at extremes** (predicts 97%, actual 91%)
- Mid-range calibration (0.35–0.85) is good

### Directionality Correlations

| Feature | Correlation with Win Probability | Expected | Status |
|---------|:------:|:--------:|:------:|
| `runs_remaining` | -0.386 | Negative | ✅ Correct |
| `balls_remaining` | +0.107 | Positive | ✅ Correct |
| `wickets_lost` | -0.447 | Negative | ✅ Correct |

All directional signals are correct. Wickets have the strongest signal, which aligns with cricket domain knowledge — losing wickets is the strongest predictor of chase failure.

---

## 3. Strengths of the MVP

1. **Correct directional behaviour** — all three core features influence probability in the right direction
2. **Good mid-range calibration** — the 0.3–0.85 band is well-calibrated
3. **Clean architecture** — modular, testable, and well-separated concerns
4. **Data leakage prevention** — validation correctly excludes same-match states
5. **Sensible data pipeline** — parquet caching, dtype optimisation, legal delivery tracking
6. **Known anomalies documented** — the 121-delivery match is handled transparently
7. **Interpretable** — every prediction can be traced back to specific historical matches

---

## 4. Weaknesses and Improvement Opportunities

### 4.1 Distance Metric Weaknesses

> **Priority: Critical — this is where the biggest Brier score gains will come from.**

| Weakness | Impact | Detail |
|----------|--------|--------|
| **Hand-tuned divisors** | Medium-High | The values (50, 30, 2) have no statistical basis. They implicitly define feature importance but don't reflect the actual variance in the data. |
| **Equal feature weighting** | High | After normalisation, all three features contribute equally. In reality, the *relative importance* of runs, balls, and wickets changes throughout the innings. |
| **No phase awareness** | High | Needing 40 off 30 with 2 wickets lost is dramatically different in the powerplay vs death overs, but the engine treats them identically. |
| **Missing interactions** | High | The engine treats features independently. A team needing 20 off 12 at 0 wickets is very different from 20 off 12 at 8 wickets — this interaction is not captured in the distance. |
| **Missing features** | Medium | `required_run_rate`, `run_rate`, and `phase` are computed but never used in the distance function. |
| **No distance weighting** | Medium | All k=200 neighbours are treated equally regardless of how close they are. A state at distance 0.001 has the same influence as one at distance 0.5. |

### 4.2 Aggregation Weaknesses

| Weakness | Impact |
|----------|--------|
| Unweighted mean of `won_match` — no distance-based weighting | Medium |
| No confidence interval or uncertainty estimate | Low |
| No smoothing across adjacent game states | Low |

### 4.3 Calibration Weaknesses

| Weakness | Impact |
|----------|--------|
| Overconfidence at extremes (0.96 predicted → 0.91 actual) | Medium |
| Underconfidence at low end (0.04 predicted → 0.07 actual) | Medium |
| No post-hoc calibration layer | Low (for now) |

### 4.4 Performance Concerns

| Issue | Impact |
|-------|--------|
| Full dataframe copy on every `find_similar_states` call | Medium |
| Linear scan of all historical states (no indexing) | Low (acceptable at current scale) |

---

## 5. Prioritised Upgrade Roadmap

The upgrades are ordered by **expected Brier score improvement per unit of implementation effort**. Each upgrade is designed to be independently measurable using the existing validation framework.

---

### Upgrade 2A: Variance-Based Feature Scaling ✅

**Priority:** 🔴 Critical — Do this first  
**Difficulty:** Easy  
**Expected Brier improvement:** 0.005–0.015  

#### Problem

The current divisors (50, 30, 2) are arbitrary. If runs_remaining has a standard deviation of 38 but is normalised by 50, and wickets_lost has a standard deviation of 2.5 but is normalised by 2, then wickets are being artificially amplified relative to their natural variance.

#### Solution

Replace hand-tuned divisors with the **actual standard deviation** of each feature computed from the historical dataset.

```python
# Before
distance = (rr / 50)² + (br / 30)² + (wk / 2)²

# After
distance = (rr / σ_runs)² + (br / σ_balls)² + (wk / σ_wickets)²
```

Where `σ` values are computed once at engine initialisation from the training data.

#### Implementation

```python
class ImprovedSimilarityEngine:

    def __init__(self, k=200):
        self.k = k
        self.fitted = False
        self.scales = {}

    def fit(self, historical_states: pd.DataFrame):
        """Compute variance-based scales from the dataset."""
        self.scales = {
            "runs_remaining": historical_states["runs_remaining"].std(),
            "balls_remaining": historical_states["balls_remaining"].std(),
            "wickets_lost": historical_states["wickets_lost"].std(),
        }
        self.fitted = True

    def find_similar_states(self, state, historical_states):
        if not self.fitted:
            self.fit(historical_states)

        rr = historical_states["runs_remaining"] - state["runs_remaining"]
        br = historical_states["balls_remaining"] - state["balls_remaining"]
        wk = historical_states["wickets_lost"] - state["wickets_lost"]

        distance = (
            (rr / self.scales["runs_remaining"]) ** 2 +
            (br / self.scales["balls_remaining"]) ** 2 +
            (wk / self.scales["wickets_lost"]) ** 2
        )

        return historical_states.assign(distance=distance) \
            .sort_values("distance").head(self.k)
```

#### Measurement

Run the existing validation with the new engine and compare Brier scores directly.

---

### Upgrade 2B: Distance-Weighted Probability Aggregation ✅

**Priority:** 🔴 Critical  
**Difficulty:** Easy  
**Expected Brier improvement:** 0.005–0.012  

#### Problem

Currently all 200 nearest neighbours contribute equally:

```python
probability = similar_states["won_match"].mean()
```

A state at distance 0.001 (near-identical) has the same weight as one at distance 0.8 (quite different). This is a well-known weakness in unweighted KNN.

#### Solution

Use **inverse-distance weighting** so closer neighbours have more influence.

```python
# Gaussian kernel weighting
weights = np.exp(-similar_states["distance"])
probability = np.average(
    similar_states["won_match"],
    weights=weights
)
```

Or simpler inverse-distance:

```python
weights = 1.0 / (similar_states["distance"] + epsilon)
probability = np.average(
    similar_states["won_match"],
    weights=weights
)
```

The Gaussian kernel is preferred because it:
- Gives very high weight to near-identical states
- Smoothly decays for more distant states
- Doesn't explode for very small distances (unlike 1/d)

#### Implementation

Modify `WinProbabilityModel.predict()` to accept and use distances.

#### Measurement

Direct Brier score comparison. Also check calibration table — this should particularly improve extreme buckets.

---

### Upgrade 2C: Optimal k Selection ✅

**Priority:** 🟡 Important  
**Difficulty:** Easy  
**Expected Brier improvement:** 0.002–0.008  

#### Problem

`k=200` was chosen without systematic evaluation. Too small a k introduces noise; too large a k over-smooths and destroys signal.

#### Solution

Sweep k values systematically and measure Brier score at each:

```python
k_values = [50, 75, 100, 150, 200, 300, 500]
```

Run the full validation for each k and pick the value that minimises Brier score.

> [!TIP]
> With distance-weighted aggregation (Upgrade 2B), the model becomes less sensitive to k because distant neighbours are naturally down-weighted. This means a larger k is less harmful and may even help. Test k selection **after** implementing 2B.

#### Measurement

Brier score grid search. Plot k vs Brier score to find the optimal region.

---

### Upgrade 2D: Phase-Aware Similarity ✅

**Priority:** 🟡 Important  
**Difficulty:** Medium  
**Expected Brier improvement:** 0.008–0.020  
**Actual Result**: Hard phase filtering (Approach A) reduced overall Brier score from **0.1543** to **0.1538**, primarily improving the Death overs phase (reduced from **0.1018** to **0.0978**).

#### Problem

The current engine treats all overs equally. But cricket is fundamentally phase-dependent:

- **Powerplay (overs 0–5)**: Fielding restrictions, aggressive batting, low wicket cost
- **Middle (overs 6–15)**: Consolidation, spin bowling, acceleration decisions
- **Death (overs 16–19)**: Full aggression, high risk, specialist death bowlers

Needing 60 runs off 36 balls at 2 wickets in over 8 is a completely different situation from needing 60 off 36 at 2 wickets in over 14 — even though the raw numbers are identical.

#### Solution: Phase Filtering + Phase Feature

Two complementary approaches:

**Approach A — Hard phase filter (simpler, do first):**

```python
def find_similar_states(self, state, historical_states):
    # First, filter to same phase
    same_phase = historical_states[
        historical_states["phase"] == state["phase"]
    ]

    # If insufficient data, fall back to all states
    if len(same_phase) < self.k * 2:
        same_phase = historical_states

    # Then apply distance metric on filtered set
    ...
```

**Approach B — Phase as a distance feature (more nuanced):**

Encode phase as a numeric distance component. States in different phases receive a penalty.

```python
# Phase penalty (0 if same phase, 1 if adjacent, 2 if powerplay vs death)
phase_map = {"powerplay": 0, "middle": 1, "death": 2}
phase_diff = abs(phase_map[state_phase] - phase_map[hist_phase])

distance = (
    (rr / σ_r)² +
    (br / σ_b)² +
    (wk / σ_w)² +
    (phase_diff / 1.0)²   # Phase penalty
)
```

#### Recommendation

Start with Approach A (hard filter). Measure. If it helps, keep it. If data sparsity is a problem in some phases, switch to Approach B.

#### Measurement

Brier score overall + per-phase Brier scores. This upgrade should particularly improve powerplay and death overs predictions.

---

### Upgrade 2E: Feature Interactions ✅

**Priority:** 🟡 Important  
**Difficulty:** Medium  
**Expected Brier improvement:** 0.005–0.015  
**Actual Result**: Dynamically adding the **Pressure Index** (`required_run_rate * (1 + wickets_lost / 10)`) as a distance feature reduced overall Brier score from **0.1538** to **0.1526**, primarily helping the Powerplay (reduced from **0.2015** to **0.2007**) and Death (reduced from **0.0978** to **0.0893**) phases.

#### Problem

The current distance treats features independently. But in cricket, features interact:

- **Wickets × Runs remaining**: 50 to get with 2 wickets lost is comfortable; 50 to get with 8 wickets lost is nearly impossible
- **Wickets × Balls remaining**: More wickets in hand matters more in death overs
- **Required run rate**: A composite interaction of runs and balls, but non-linear with wickets

#### Solution

Add interaction features to the distance metric:

```python
# Wickets remaining (derived)
wickets_remaining = 10 - wickets_lost

# Pressure index: required_run_rate × (1 + wickets_lost / 10)
# Captures the escalating danger of losing wickets under run pressure
pressure = required_run_rate * (1 + wickets_lost / 10)

# Resources remaining (simplified Duckworth-Lewis intuition)
resources = wickets_remaining * balls_remaining
```

Add these as additional distance dimensions:

```python
distance = (
    (rr / σ_r)² +
    (br / σ_b)² +
    (wk / σ_w)² +
    (pressure_diff / σ_p)² +
    (resources_diff / σ_res)²
)
```

> [!IMPORTANT]
> Add interaction features **one at a time** and measure each individually. Some may hurt rather than help due to the curse of dimensionality. Only keep features that measurably improve Brier score.

#### Measurement

Brier score with each feature added individually, then in combination. Track which features help and which are redundant.

---

### Upgrade 2F: Learned Feature Weights ✅

**Priority:** 🟠 High — but do after 2A–2E  
**Difficulty:** Medium-Hard  
**Expected Brier improvement:** 0.005–0.015  
**Actual Result**: Optimized feature weights via a two-stage (coarse-to-fine) 5-fold cross-validated grid search. The optimal weights identified are: w_r = 0.3, w_b = 2.2, w_w = 1.1, w_p = 1.1. This reduced overall Brier Score from **0.1526** to **0.1520** (improving Powerplay to **0.1993** and Death to **0.0890**).

#### Problem

Even after variance-based scaling, all features contribute equally to distance. In reality, some features matter more than others, and their relative importance may vary.

#### Solution

Use **cross-validated grid search** to optimise feature weights:

```python
# Parameterised distance
distance = (
    w_r * (rr / σ_r)² +
    w_b * (br / σ_b)² +
    w_w * (wk / σ_w)²
)
```

Search over a grid of `(w_r, w_b, w_w)` values and select weights that minimise Brier score on a held-out validation fold.

```python
weight_grid = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

best_brier = 1.0
best_weights = None

for w_r in weight_grid:
    for w_b in weight_grid:
        for w_w in weight_grid:
            brier = evaluate(w_r, w_b, w_w)
            if brier < best_brier:
                best_brier = brier
                best_weights = (w_r, w_b, w_w)
```

> [!WARNING]
> This requires proper train/validation/test splitting to avoid overfitting the weights to the validation set. Use k-fold cross-validation (e.g., 5-fold stratified by match_id) rather than a single random sample.

#### Measurement

Cross-validated Brier score. Report both in-fold and out-of-fold scores to detect overfitting.

---

### Upgrade 2G: Cricket-Aware Distance (Advanced) ✅

**Priority:** 🔵 Future — after exhausting simpler options  
**Difficulty:** Hard  
**Expected Brier improvement:** 0.005–0.012  
**Actual Result**: Evaluated Exponential Wickets, Run Pressure, and Desperation Index. Adding **Exponential Wickets** (`np.exp(wk / 3.0) - 1.0`) successfully lowered the overall Brier Score to **0.1513** (particularly improving the Death overs phase to **0.0871**). The other non-linear features degraded performance. Thus, `use_exponential_wickets` is set to `True` by default.

#### Problem

A standard Euclidean distance in feature space doesn't capture cricket-specific non-linearities. For example:

- The difference between 0 and 1 wicket lost is much smaller than the difference between 8 and 9 wickets lost
- Runs remaining at the start of an innings matter less per-run than runs remaining with 2 balls left
- The relationship between required run rate and win probability is non-linear (6 RRR is routine; 12 RRR is panic; 18 RRR is near-impossible)

#### Solution

Apply non-linear transformations before computing distance:

```python
# Wicket pressure: exponential scaling
# Makes high wicket counts contribute disproportionately more
wk_transformed = np.exp(wickets_lost / 3) - 1

# Run pressure: scales up as fewer balls remain
run_pressure = runs_remaining / max(balls_remaining / 6, 0.5)

# Desperation index: non-linear combination
desperation = (required_run_rate / 8) ** 1.5 * (1 + wickets_lost / 5)
```

This is effectively hand-crafting features that encode cricket knowledge into the distance function.

#### Measurement

Brier score. Per-phase Brier scores will be particularly informative here.

---

## 6. Recommended Implementation Order

```
Phase 1 — Quick Wins (1-2 days)
├── 2A  Variance-based scaling [Completed ✅]  ← Do first
├── 2B  Distance-weighted aggregation [Completed ✅]  ← Do second
└── 2C  Optimal k selection [Completed ✅]             ← Do after 2B

Phase 2 — Structural Improvements (2-3 days)
├── 2D  Phase-aware similarity          ← Biggest expected gain
└── 2E  Feature interactions            ← Add one feature at a time

Phase 3 — Optimisation (2-3 days)
├── 2F  Learned feature weights         ← Requires cross-validation
└── 2G  Cricket-aware distance          ← Only if Phase 1+2 plateau
```

---

## 7. Expected Cumulative Impact

| After Upgrade | Expected Brier Score | Improvement |
|---------------|:--------------------:|:-----------:|
| **Current MVP** | **0.1548** | — |
| + 2A (variance scaling) | **0.1544** | **-0.0004** (Actual) |
| + 2B (distance weighting) | **0.1544** | **0.0000** (Actual)* |
| + 2C (optimal k) | **0.1543** | **-0.0001** (Actual) |

*Note on 2B: Due to the high density of the historical dataset (~161k states), the 200 nearest neighbours are extremely close to the target state, meaning their calculated distances are near 0. Consequently, all Gaussian weights are very close to 1.0, leading to a weighted average almost identical to the unweighted mean. This will become more impactful when tuning `k` (Upgrade 2C) or using smaller bandwidths/larger feature spaces.
| + 2D (phase awareness) | **0.1538** | **-0.0005** (Actual) |
| + 2E (interactions) | **0.1526** | **-0.0012** (Actual) |
| + 2F (learned weights) | **0.1516** | **-0.0010** (Actual) |
| + 2G (cricket-aware) | **0.1513** | **-0.0003** (Actual) |

> [!NOTE]
> These are estimates. The actual improvements will depend on the data distribution and may be higher or lower. The key principle is: **measure each upgrade independently** before moving to the next. If an upgrade doesn't help, don't keep it.

**Target:** A Brier score below **0.12** should be achievable through similarity engine improvements alone, before any ML models are introduced.

---

## 8. Known Issues

| Issue | Status | Action |
|-------|--------|--------|
| Match 1410492 has 121 legal deliveries | Documented | Retain — genuine anomaly |
| `balls_remaining` can go negative for that match | Known | Warning printed, data retained |
| `calibration.py` and `metrics.py` are empty | Placeholder | Will be populated in later upgrades |
| Validation samples only 500 states | Adequate for now | Increase to 1000+ for weight optimisation (Upgrade 2F) |
| `df.copy()` in `find_similar_states` | Performance | Remove copy — use `.assign()` on original (already partially done) |
| Double filtering of 2nd innings in `win_probability_validation.py` (lines 430 and 439–441) | Minor bug | Harmless but should be cleaned up |

---

## 9. Validation Framework Enhancements

To support the upgrade roadmap, the validation framework should be extended:

### 9.1 A/B Comparison Script

Create a script that runs two engines side-by-side on the same sample and reports the Brier score difference with statistical significance.

### 9.2 Per-Phase Brier Scores

Break down Brier score by phase (powerplay / middle / death) to understand where each upgrade has the most impact.

### 9.3 Calibration Plot

Produce a visual calibration curve (predicted vs actual) to complement the table. This makes it easier to spot systematic biases.

### 9.4 Increase Sample Size for Weight Optimisation

When running grid searches (Upgrade 2F), increase the validation sample from 500 to at least 1000 states to reduce noise in Brier score estimates.

---

## 10. What NOT to Do Yet

| Temptation | Why Wait |
|------------|----------|
| Jump to XGBoost / neural networks | The similarity engine still has substantial untapped potential. ML models will be more effective once the feature space is properly engineered. |
| Add team-specific features | The dataset may not have enough matches per team to support team-level modelling without overfitting. Revisit after Upgrade 3 (feature engineering). |
| Add player-level features | Same sparsity concern. Player features require significant data infrastructure work. |
| Isotonic calibration / Platt scaling | Calibration layers should be added **after** the underlying model is as good as possible. They can't fix a fundamentally noisy model. |

---

## Summary

The MVP is a strong foundation with correct directional behaviour and reasonable calibration. The similarity engine has clear, measurable weaknesses that can be addressed systematically. The recommended approach is:

1. **Fix the scaling** (2A) — remove arbitrary constants
2. **Weight by distance** (2B) — let close neighbours dominate
3. **Tune k** (2C) — find the sweet spot
4. **Add phase awareness** (2D) — biggest structural improvement
5. **Add interactions** (2E) — capture cricket-specific relationships
6. **Optimise weights** (2F) — data-driven feature importance
7. **Cricket-aware transforms** (2G) — non-linear domain knowledge

Each upgrade is independently measurable and should move the Brier score from **0.1548 toward 0.10 or below**.
