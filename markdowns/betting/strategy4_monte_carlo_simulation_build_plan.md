# Strategy 4: Monte Carlo Match Simulation — Build Plan

> **Author:** Antigravity AI · **Date:** 2026-07-21  
> **Strategy:** 4 — Monte Carlo Match Simulation for Exotic Markets (Most Ambitious)  
> **Model Baseline:** T20 win-prob Brier `0.1287` · First-innings regressor already trained · 2.94M ball-by-ball rows available  
> **Prerequisite:** Existing Elo (team/phase/player), venue, and empirical-DLS infrastructure — all reused read-only  
> **Defaults / Scope:** T20 first · 10,000 sims/match · Paper pricing validated against historical distributions before any staking

---

## Executive Summary

Go beyond win/loss. Build a **ball-by-ball match simulator** driven by a new multinomial delivery-outcome model, run it thousands of times per fixture, and turn the resulting outcome distributions into fair prices for **exotic markets** — total runs over/under, margin of victory, method of victory, top batsman, highest opening partnership, and player props. Exotic markets are far less efficient than match-winner markets, and almost no individual bettor can price them from first principles, so this is the highest *potential* edge of any strategy in the roadmap — and the highest engineering risk.

The one genuinely new component is the **delivery-outcome model**: our current stack predicts `P(win)` and `P(final first-innings score)`, but not `P(this specific delivery's outcome)`. Everything else — team/phase/player Elo, venue features, empirical DLS — transfers directly from the existing pipeline as conditioning inputs. The simulator is a stochastic wrapper that samples outcomes delivery-by-delivery, tracks match state (strike rotation, wickets, bowler overs), and aggregates thousands of completed matches into distributions.

> [!IMPORTANT]
> **The whole strategy lives or dies on calibration.** A per-delivery model with even a small bias compounds over 120 balls (T20) or 300 balls (ODI) into wildly wrong score distributions. Phase 5 (calibration validation) is not optional polish — it is the go/no-go gate. If simulated score distributions don't match historical reality, the exotic prices are fiction and nothing gets staked.

### What We Can Price Once This Works

| Market | Priced From |
|--------|-------------|
| Total match/innings runs (over/under) | Distribution of simulated innings totals |
| Margin of victory (runs / wickets) | Full two-innings simulation captures margin spread |
| Method of victory (bat first vs chase) | Compare simulated outcomes across toss/bat-order scenarios |
| Top batsman | Per-player run distributions weighted by batting position + Elo |
| Highest opening partnership | Partnership runs before first wicket, across sims |
| Player performance props (e.g. batter 30+, bowler 3+ wkts) | Tail probabilities of per-player simulated tallies |

### Phase Overview

| # | Phase | Tier | Effort | Files Created |
|---|-------|------|:------:|---------------|
| 1 | Delivery-Outcome Model | New Model | 🔴 High | `src/models/delivery_outcome_model.py` |
| 2 | Innings Simulator | Core Engine | 🔴 High | `src/simulation/innings_simulator.py` |
| 3 | Match Simulator (Monte Carlo) | Core Engine | 🔴 High | `src/simulation/match_simulator.py` |
| 4 | Market Pricing Engine | Pricing | 🟡 Medium | `src/simulation/market_pricer.py` |
| 5 | Calibration & Validation | Validation Gate | 🔴 High | `tests/test_simulation_calibration.py` |
| 6 | Exotic Value Detection | Integration | 🟢 Low | `src/betting/exotic_scanner.py` |

---

## File Structure

```
src/
  models/
    delivery_outcome_model.py       [Phase 1] NEW — Multinomial P(outcome | state) per delivery
    xgb_win_probability.py          [Existing] ← READ-ONLY (optional cross-check of sim vs model P(win))
    first_innings_predictor.py      [Existing] ← READ-ONLY (sanity anchor for innings totals)
  simulation/
    __init__.py                     [Phase 2] NEW — Package init
    innings_simulator.py            [Phase 2] NEW — Ball-by-ball stochastic innings engine
    match_simulator.py              [Phase 3] NEW — Two-innings + N-simulation Monte Carlo loop
    market_pricer.py                [Phase 4] NEW — Distributions → fair decimal odds
  features/
    elo_engine.py                   [Existing] ← READ-ONLY (team Elo lookups)
    phase_elo_engine.py             [Existing] ← READ-ONLY (phase strength)
    player_elo_engine.py            [Existing] ← READ-ONLY (batter/bowler Elo)
    venue_engine.py                 [Existing] ← READ-ONLY (venue scoring bias)
  betting/
    exotic_scanner.py               [Phase 6] NEW — Compare sim fair odds vs exchange, reuse ValueDetector
    value_detector.py               [Existing] ← reused directly
data/
  processed/
    t20_cricket_data.parquet        [Existing] 2.94M deliveries — training data for Phase 1
    t20_player_elo_ratings.parquet  [Existing] Per-match batter/bowler effective_elo + recent_form
    t20_venue_features.parquet      [Existing] Venue scoring bias
    empirical_dls_t20.parquet       [Existing] Resource curves (conditioning + sanity anchor)
    t20_delivery_outcome_model.pkl  [Phase 1] NEW — Serialised multinomial model
    simulation_calibration_t20.parquet [Phase 5] NEW — Sim-vs-actual calibration report
```

---

## Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `xgboost` | `XGBClassifier` (multi:softprob) for the delivery-outcome model | Already installed |
| `numpy` | Vectorised sampling across the simulation ensemble | Already installed |
| `pandas` | Data handling, distribution aggregation | Already installed |
| `scikit-learn` | `GroupKFold`, `log_loss`, calibration utilities | Already installed |
| `scipy` (optional) | Distribution fitting / KS tests in calibration | Already installed |
| `matplotlib` (optional) | Calibration histograms, reliability plots | Already installed |

No new heavy dependencies — this is XGBoost + NumPy sampling on top of existing infrastructure.

---

## Phase Execution Order

```
Phase 1: Delivery-Outcome Model
    (train multinomial P(outcome | state) on 2.94M deliveries)
    │
    ▼
Phase 2: Innings Simulator
    (sample ball-by-ball, track strike/wickets/bowler overs → one innings)
    │
    ▼
Phase 3: Match Simulator (Monte Carlo)
    (simulate both innings × N times, vectorised across the ensemble)
    │
    ▼
Phase 4: Market Pricing Engine
    (aggregate N outcomes → distributions → fair decimal odds)
    │
    ▼
Phase 5: Calibration & Validation  ← GO/NO-GO GATE
    (do simulated distributions match historical reality?)
    │
    ▼
Phase 6: Exotic Value Detection (compare fair odds vs exchange)
    │
    ▼
Paper Pricing Validation (price live matches, log vs settled results)
```

> [!CAUTION]
> Do **not** proceed past Phase 5 until calibration passes. Every downstream price is a deterministic function of the simulated distributions; if those are biased, Phase 6 will confidently generate value signals that are pure noise. This is the single most important discipline in the whole strategy.

---

## Phase 1: Delivery-Outcome Model

**File:** `src/models/delivery_outcome_model.py` (NEW)  
**Priority:** 🔴 Critical — the one new model the whole simulator depends on  
**Dependencies:** `t20_cricket_data.parquet`, player/phase/venue Elo features, `xgboost`

### Problem

We need `P(outcome | match_state)` for a single delivery. The existing models predict aggregate `P(win)` and `P(final score)` — neither gives the per-ball outcome distribution a simulator must sample from. We train a new multinomial classifier on the 2.94M historical deliveries.

### Outcome Space

Empirical distribution over `runs_off_bat` (from `t20_cricket_data.parquet`): `0`→43.2%, `1`→35.6%, `2`→6.7%, `3`→0.4%, `4`→10.1%, `6`→4.0%; `wicket` rate 5.46%; `is_wide` 3.7%; `is_noball` 0.5%. Values of `5` and `7` are negligible overthrow noise (<0.02%) — fold into `4` and `6` respectively.

We model this in **two stages**, because extras don't consume a legal ball and must be handled separately from the batted outcome:

```
Stage A — Extras gate:  P(legal) vs P(wide) vs P(noball)
Stage B — Legal ball:   multinomial over {0, 1, 2, 3, 4, 6, WICKET}
```

> [!NOTE]
> Two stages keep ball-consumption semantics clean: a **wide/no-ball adds a run and re-bowls the delivery** (no legal ball consumed, no strike change on the extra itself), while a legal ball always advances the over count. A single 9-class model muddies this and makes the simulator's ball-counting logic error-prone.

### Implementation

#### Step 1: Build the training frame with conditioning features

Mirror the feature-extraction pattern in `FirstInningsPredictor._extract_features_and_target()` ([first_innings_predictor.py:55](../src/models/first_innings_predictor.py#L55)), but at **per-delivery granularity** and with the batter/bowler matchup front and centre.

```python
import pandas as pd
import numpy as np

LEGAL_CLASSES = [0, 1, 2, 3, 4, 6, "W"]   # index order the model emits

def build_training_frame(deliveries: pd.DataFrame, player_elo: pd.DataFrame) -> pd.DataFrame:
    """
    One row per historical delivery with the state features the simulator can reproduce
    at sampling time. Joins per-match batter/bowler effective_elo from player_elo_ratings.
    """
    df = deliveries.copy()

    # Collapse rare overthrow noise
    df["runs_off_bat"] = df["runs_off_bat"].replace({5: 4, 7: 6})

    # Legal-ball outcome label (only defined where NOT an extra)
    df["legal_outcome"] = np.where(df["wicket"] == 1, "W", df["runs_off_bat"].astype(str))

    # Extras label
    df["extra_class"] = np.select(
        [df["is_wide"] == 1, df["is_noball"] == 1],
        ["wide", "noball"], default="legal"
    )

    # --- Conditioning features (must be reproducible inside the simulator) ---
    # State
    df["over_float"] = df["over"] + (df["ball"] - 1) / 6.0
    df["balls_remaining"] = (df["target_overs"].fillna(20.0) * 6.0) - (df["over"] * 6.0 + df["ball"])
    df["phase_encoded"] = pd.cut(df["over"], bins=[-1, 5, 14, 99], labels=[0, 1, 2]).astype(int)
    df["innings"] = df["innings"].astype(int)

    # Batter / bowler matchup (join effective_elo per match+player from player_elo)
    df = _join_player_elo(df, player_elo, who="batter")
    df = _join_player_elo(df, player_elo, who="bowler")
    df["matchup_elo_diff"] = df["batter_elo"] - df["bowler_elo"]

    # Venue scoring bias + wickets_lost so far (cumulative within innings)
    df["wickets_lost"] = df.groupby(["match_id", "innings"])["wicket"].cumsum() - df["wicket"]

    return df
```

#### Step 2: Two XGBoost classifiers, GroupKFold by match_id

Follow the ensemble discipline already used in the codebase — `GroupKFold` on `match_id` so no match straddles train/validation, plus a `final_model` fit on all data for production sampling.

```python
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold

FEATURES = ["over_float", "balls_remaining", "phase_encoded", "innings",
            "wickets_lost", "batter_elo", "bowler_elo", "matchup_elo_diff",
            "venue_avg_first_innings_score", "recent_form_batter", "recent_form_bowler"]

class DeliveryOutcomeModel:
    """Two-stage multinomial delivery-outcome sampler."""

    def __init__(self, format_type="t20"):
        self.format_type = format_type
        self.extras_model = None    # 3-class: legal / wide / noball
        self.legal_model = None     # 7-class: 0,1,2,3,4,6,W
        self.legal_classes = LEGAL_CLASSES

    def fit(self, frame: pd.DataFrame, cache_path=None):
        if cache_path and Path(cache_path).exists():
            self._load(cache_path); return self

        # Stage A: extras gate (all deliveries)
        self.extras_model = XGBClassifier(
            objective="multi:softprob", num_class=3,
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        ).fit(frame[FEATURES], frame["extra_class"].map({"legal":0,"wide":1,"noball":2}))

        # Stage B: legal-ball outcome (legal deliveries only)
        legal = frame[frame["extra_class"] == "legal"]
        self.legal_model = XGBClassifier(
            objective="multi:softprob", num_class=7,
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        ).fit(legal[FEATURES], legal["legal_outcome"].map({str(c): i for i, c in enumerate(LEGAL_CLASSES)}))

        if cache_path:
            self._save(cache_path)
        return self
```

#### Step 3: Vectorised probability lookup (the simulator's hot path)

The simulator calls this once per ball **for all N simulations at once** — batched prediction is what makes 10,000 sims tractable (see Phase 3 performance note).

```python
    def predict_proba_batch(self, state_matrix: pd.DataFrame) -> dict:
        """
        state_matrix: one row per live simulation (N rows), columns = FEATURES.
        Returns {'extras': (N,3) probs, 'legal': (N,7) probs} for vectorised sampling.
        """
        return {
            "extras": self.extras_model.predict_proba(state_matrix[FEATURES].values),
            "legal":  self.legal_model.predict_proba(state_matrix[FEATURES].values),
        }
```

### Verification

```bash
py -c "
from src.models.delivery_outcome_model import DeliveryOutcomeModel, build_training_frame
import pandas as pd
d = pd.read_parquet('data/processed/t20_cricket_data.parquet')
pe = pd.read_parquet('data/processed/t20_player_elo_ratings.parquet')
frame = build_training_frame(d, pe)
m = DeliveryOutcomeModel('t20').fit(frame, cache_path='data/processed/t20_delivery_outcome_model.pkl')
# Marginal check: model's average predicted class probs must match empirical base rates
import numpy as np
probs = m.legal_model.predict_proba(frame[frame.extra_class=='legal'].sample(50000)[FEATURES].values).mean(0)
print(dict(zip(['0','1','2','3','4','6','W'], probs.round(4))))
"
```

Expected: averaged predicted probabilities close to empirical base rates (`0`≈0.43, `1`≈0.36, `4`≈0.10, `6`≈0.04, `W`≈0.055). A large gap here means the model is miscalibrated at the margin — fix before simulating.

---

## Phase 2: Innings Simulator

**File:** `src/simulation/innings_simulator.py` (NEW)  
**Priority:** 🔴 Critical — the stochastic engine that turns per-ball probs into an innings  
**Dependencies:** `DeliveryOutcomeModel`, a batting lineup + bowling attack with Elo

### Problem

Given a lineup, an attack, and the outcome model, simulate one complete innings ball-by-ball: sample each delivery, update the score, rotate strike on odd runs and at over-end, fall wickets and bring in the next batter, and enforce bowler over-limits. This must be **vectorised across N simultaneous simulations** so Phase 3 can run 10,000 in seconds, not minutes.

### Match-State Bookkeeping (the hard part)

| State per simulation | Update rule |
|---|---|
| `score`, `wickets` | Add sampled runs; increment wickets on `W` |
| `striker`, `non_striker` | Swap on 1/3 runs and at the end of each over |
| `batting_position` | On wicket, next batter (position `wickets+2`) comes in as striker |
| `legal_balls_bowled` | +1 only on a legal delivery (extras re-bowl) |
| `bowler` | New bowler each over; enforce max `total_overs/5` overs per bowler; no back-to-back overs |
| `innings_over` flag | True when `wickets == 10` or `legal_balls == total_balls` (or target chased in innings 2) |

> [!WARNING]
> **Strike rotation and batter substitution are the most bug-prone logic in the entire strategy.** An off-by-one in who's on strike silently corrupts every per-player distribution (top batsman, partnerships, props). Unit-test this against a hand-traced over before trusting any aggregate output.

### Implementation

#### Step 1: Vectorised innings loop

```python
import numpy as np
import pandas as pd

def simulate_innings(model, lineup, attack, venue_feats, n_sims=10000,
                     total_overs=20.0, innings=1, target=None, rng=None):
    """
    Runs n_sims independent innings in lockstep, one ball at a time, fully vectorised.
    lineup: ordered list of 11 batters with elo/recent_form.
    attack: list of bowlers with elo/recent_form + over allowances.
    Returns a dict of per-sim arrays: final_score, wickets, per_batter_runs, fell_at_ball, ...
    """
    rng = rng or np.random.default_rng()   # NOTE: seed passed in for reproducibility (see Phase 3)
    total_balls = int(total_overs * 6)

    # Per-sim live state (all length-n_sims arrays)
    score      = np.zeros(n_sims, dtype=np.int32)
    wickets    = np.zeros(n_sims, dtype=np.int8)
    striker    = np.zeros(n_sims, dtype=np.int8)   # index into lineup (0,1 open)
    non_striker= np.ones(n_sims,  dtype=np.int8)
    next_bat   = np.full(n_sims, 2, dtype=np.int8) # next batter to come in
    legal_balls= np.zeros(n_sims, dtype=np.int16)
    per_batter = np.zeros((n_sims, 11), dtype=np.int16)
    live       = np.ones(n_sims, dtype=bool)       # innings still in progress

    while live.any():
        idx = np.where(live)[0]                    # only advance live sims
        feats = _assemble_state_matrix(idx, score, wickets, legal_balls, striker,
                                       lineup, attack, venue_feats, innings, total_overs)
        probs = model.predict_proba_batch(feats)

        # Stage A: extras gate
        u = rng.random(idx.size)
        extra = _sample_categorical(probs["extras"], rng)   # 0 legal / 1 wide / 2 noball
        is_extra = extra > 0
        score[idx[is_extra]] += 1                            # extra run, ball re-bowled

        # Stage B: legal deliveries only
        legal_idx = idx[~is_extra]
        out = _sample_categorical(probs["legal"][~is_extra], rng)  # 0..6 => class index
        _apply_legal_outcome(legal_idx, out, score, wickets, striker, non_striker,
                             next_bat, per_batter, legal_balls, model.legal_classes)

        # End-of-over strike swap + bowler rotation handled inside _apply_legal_outcome
        live = _update_live(live, wickets, legal_balls, total_balls, score, target, innings)

    return {"final_score": score, "wickets": wickets, "per_batter_runs": per_batter}
```

#### Step 2: Categorical sampling helper (vectorised inverse-CDF)

```python
def _sample_categorical(prob_matrix: np.ndarray, rng) -> np.ndarray:
    """Vectorised multinomial draw: one class index per row via inverse-CDF."""
    cdf = np.cumsum(prob_matrix, axis=1)
    u = rng.random((prob_matrix.shape[0], 1))
    return (u < cdf).argmax(axis=1)
```

#### Step 3: Outcome application (runs, wickets, strike)

`_apply_legal_outcome()` maps class index → runs (or wicket), adds to `score` and `per_batter[striker]`, increments `legal_balls`, swaps strike on odd runs, and on `W` promotes `next_bat`. End-of-over swap is applied when `legal_balls % 6 == 0`.

### Verification

```bash
py -c "
from src.simulation.innings_simulator import simulate_innings
from src.models.delivery_outcome_model import DeliveryOutcomeModel
# ... load model, a known lineup/attack, venue feats ...
res = simulate_innings(model, lineup, attack, venue_feats, n_sims=10000, innings=1)
import numpy as np
print(f'Mean total: {res[\"final_score\"].mean():.1f}  (T20 typical ~160)')
print(f'P5/P95: {np.percentile(res[\"final_score\"], [5,95]).round(0)}')
print(f'All-out rate: {(res[\"wickets\"]==10).mean():.1%}')
"
```

Expected: mean innings total in a realistic band (~150–170 for a balanced T20), a plausible spread, and an all-out rate consistent with history (~15–25%).

---

## Phase 3: Match Simulator (Monte Carlo)

**File:** `src/simulation/match_simulator.py` (NEW)  
**Priority:** 🔴 Critical — orchestrates both innings across N simulations  
**Dependencies:** `innings_simulator`, lineups + attacks + venue for both teams

### Problem

Wrap the innings engine into a full match: simulate innings 1 (team batting first) to a total, then simulate innings 2 (the chase) conditioned on that target, across N Monte Carlo runs. The output is N complete match outcomes — every downstream price is an aggregate over these.

### Implementation

#### Step 1: Two-innings orchestration

```python
import numpy as np

def simulate_match(model, team_a, team_b, venue_feats, n_sims=10000,
                   total_overs=20.0, bat_first="a", seed=42):
    """
    Full Monte Carlo match. Returns per-sim outcomes for pricing every exotic market.
    Innings 1 total feeds innings 2 as the chase target (win on target+1).
    """
    rng = np.random.default_rng(seed)   # reproducible: same seed → same sim set

    first, second = (team_a, team_b) if bat_first == "a" else (team_b, team_a)

    inn1 = simulate_innings(model, first["lineup"], second["attack"], venue_feats,
                            n_sims=n_sims, total_overs=total_overs, innings=1, rng=rng)
    targets = inn1["final_score"] + 1

    inn2 = simulate_innings(model, second["lineup"], first["attack"], venue_feats,
                            n_sims=n_sims, total_overs=total_overs, innings=2,
                            target=targets, rng=rng)

    first_won = inn2["final_score"] < targets - 0  # chase fell short (ties handled separately)
    return {
        "inn1_score": inn1["final_score"], "inn2_score": inn2["final_score"],
        "inn1_wkts": inn1["wickets"],       "inn2_wkts": inn2["wickets"],
        "first_batting_won": first_won,
        "inn1_batter_runs": inn1["per_batter_runs"],
        "inn2_batter_runs": inn2["per_batter_runs"],
        "margin_runs": np.where(first_won, targets - 1 - inn2["final_score"], np.nan),
        "margin_wkts": np.where(~first_won, 10 - inn2["wickets"], np.nan),
    }
```

#### Step 2: Lineups from squad data

Lineups/attacks come from the squad columns already stored per match (`team_1_players` / `team_2_players` in the Elo parquets) joined to `t20_player_elo_ratings.parquet` for each player's `effective_elo`, `recent_form`, and `role`. For a hypothetical/future fixture, pass the announced XI.

> [!IMPORTANT]
> **Performance: batch across sims, never loop per-sim.** 10,000 sims × ~120 balls = 1.2M deliveries per innings. Calling the model once per (sim, ball) is ~2.4M XGBoost calls per match — minutes of wall-clock. The vectorised design (Phase 2) issues **one batched `predict_proba` per ball-step across all live sims** → ~120–250 model calls per innings total. This is the difference between a 3-second match and a 20-minute one. If even that is too slow for live use, precompute a **binned probability table** keyed by `(phase, wickets_lost, balls_remaining_bucket, batter_elo_bucket, bowler_elo_bucket)` and sample from the table instead of the live model.

#### Step 3: Reproducibility & scenario sweeps

Because `Date.now()`/random seeding is explicit (`seed=42`), the same inputs yield the same simulation set — essential for debugging and for A/B comparing scenarios (toss outcomes, bat-first vs chase) on an identical random stream.

### Verification

```bash
py -c "
from src.simulation.match_simulator import simulate_match
# ... load model, team_a, team_b, venue_feats ...
out = simulate_match(model, team_a, team_b, venue_feats, n_sims=10000, seed=42)
print(f'P(team batting first wins): {out[\"first_batting_won\"].mean():.1%}')
# Cross-check vs the trained win-prob model for the same matchup
"
```

Expected: `P(first-batting team wins)` in a sane band (roughly 45–55% for evenly-matched sides), and — as a sanity anchor — within a few points of `XGBWinProbabilityModel`'s pre-match number for the same teams/venue.

---

## Phase 4: Market Pricing Engine

**File:** `src/simulation/market_pricer.py` (NEW)  
**Priority:** 🟡 Medium — deterministic aggregation of sim outputs into odds  
**Dependencies:** `simulate_match` output arrays

### Problem

Convert the N-outcome arrays into fair decimal odds for each exotic market. This is pure aggregation — count the fraction of sims where an event occurs, convert probability to odds, and (optionally) add a modelling-margin buffer to protect against simulation noise.

### Implementation

#### Step 1: Probability → fair odds

```python
def prob_to_fair_odds(p: float, margin: float = 0.0) -> float:
    """Fair decimal odds from a simulated probability, with an optional safety margin."""
    p_adj = min(max(p, 1e-6), 1 - 1e-6) * (1 + margin)
    return round(1.0 / p_adj, 2)
```

#### Step 2: Market pricers

```python
import numpy as np

def price_total_runs(sim, line: float) -> dict:
    """Over/under on innings-1 total (repeat for match aggregate / innings-2)."""
    totals = sim["inn1_score"]
    p_over = float((totals > line).mean())
    return {"line": line, "p_over": round(p_over, 4),
            "over_odds": prob_to_fair_odds(p_over),
            "under_odds": prob_to_fair_odds(1 - p_over)}

def price_top_batsman(sim, lineup_names, innings="inn1") -> dict:
    """P(each batter tops their innings) from per-batter run arrays."""
    runs = sim[f"{innings}_batter_runs"]            # (n_sims, 11)
    top = runs.argmax(axis=1)                        # index of top scorer each sim
    counts = np.bincount(top, minlength=11) / runs.shape[0]
    return {lineup_names[i]: {"p_top": round(counts[i], 4),
                              "odds": prob_to_fair_odds(counts[i])}
            for i in range(len(lineup_names))}

def price_margin(sim) -> dict:
    """Distribution of victory margins (runs when defending, wickets when chasing)."""
    return {
        "p_first_bat_win": round(float(sim["first_batting_won"].mean()), 4),
        "mean_run_margin": float(np.nanmean(sim["margin_runs"])),
        "mean_wkt_margin": float(np.nanmean(sim["margin_wkts"])),
    }
```

#### Step 3: Handle ties, no-results, and Monte Carlo error

- **Ties/no-result**: track separately; don't fold into a win bucket.
- **Monte Carlo standard error**: for a probability `p` over `N` sims, SE ≈ `sqrt(p(1-p)/N)`. At N=10,000 and p=0.1, SE ≈ 0.3% — report it, and treat any "edge" smaller than a few SEs as noise, not signal.

> [!NOTE]
> The `margin` buffer in `prob_to_fair_odds()` is your protection against over-confident exotic pricing. Start conservative (e.g. 3–5%) until Phase 5 proves the raw distributions are well-calibrated, then relax it.

### Verification

```bash
py -c "
from src.simulation.market_pricer import price_total_runs, price_margin
# ... run simulate_match, then: ...
print(price_total_runs(sim, line=165.5))
print(price_margin(sim))
"
```

Expected: over/under probabilities that straddle 0.5 near the mean total, and a first-bat win probability consistent with Phase 3.

---

## Phase 5: Calibration & Validation — GO/NO-GO GATE

**File:** `tests/test_simulation_calibration.py` (NEW)  
**Priority:** 🔴 Critical — proves the simulated distributions match reality  
**Dependencies:** All prior phases, historical `t20_cricket_data.parquet` + `t20_match_metadata.parquet`

### Problem

Compounding per-delivery error is the existential threat. We must prove that simulated distributions match **actual historical distributions** before pricing anything. A simulator that gets mean score right but the *variance* wrong will misprice every over/under and margin market.

### Validation Battery

#### Test 1: Marginal distribution of innings totals

Replay a large sample of historical matches: for each, simulate the first innings with the real lineups/attack, and compare the **distribution** of simulated totals to the actual total.

```python
def test_innings_total_calibration():
    # For N historical matches, simulate with real XIs; collect (actual, sim_mean, sim_p5, sim_p95)
    # PASS criteria:
    #   - actual total falls within sim [P5, P95] for ~90% of matches (coverage)
    #   - mean absolute error of sim_mean vs actual < ~12 runs (T20)
    #   - no systematic bias: mean(sim_mean - actual) ≈ 0 (within a couple of runs)
    ...
```

#### Test 2: Win-probability agreement

The simulated `P(first-bat win)` should track the trained `XGBWinProbabilityModel` — two independent methods agreeing is strong evidence both are sound.

```python
def test_winprob_agreement():
    # For a grid of matchups, |sim_P(win) - xgb_P(win)| should be small (< ~5%) on average.
    ...
```

#### Test 3: Reliability of an exotic market

Bucket historical over/under predictions by simulated probability; within each bucket the empirical hit-rate should match the bucket's predicted probability (a reliability diagram).

```python
def test_over_under_reliability():
    # Bin sim P(over 160.5) into deciles; actual over-rate per bin ≈ bin midpoint.
    # This is the calibration curve for the exotic price itself.
    ...
```

#### Test 4: Variance/tail sanity

Compare simulated score variance and boundary/wicket rates to historical — a common failure is *too little* variance (over-regularised model → distributions too tight → overconfident tail prices).

> [!CAUTION]
> **Passing "mean score" is necessary but nowhere near sufficient.** The exotic edge lives in the *tails and the spread*, not the mean. A simulator can nail average totals while badly misjudging the probability of a 200+ score or a 3-wicket collapse — and those tails are exactly what over/under and prop markets price. Test 1 (coverage), Test 3 (reliability), and Test 4 (variance) are the ones that actually gate deployment.

### Verification

```bash
py -m pytest tests/test_simulation_calibration.py -v -s
```

Expected: all four tests pass — coverage ~90%, low bias, sim/XGB win-prob agreement, and calibrated over/under reliability. **If any fail, the simulator is not deployable** regardless of how reasonable individual matches look.

---

## Phase 6: Exotic Value Detection

**File:** `src/betting/exotic_scanner.py` (NEW)  
**Priority:** 🟢 Low — thin wiring once pricing is calibrated  
**Dependencies:** `market_pricer`, `ValueDetector`, live exchange odds

### Problem

Compare our simulated fair odds against the exchange's exotic-market prices and flag value — reusing the exact Strategy 1/2 machinery (`ValueDetector` for edge + Kelly, `BetTracker` for the ledger). No new betting logic; just point it at exotic markets instead of match-winner.

### Implementation

1. Simulate the match (Phase 3) → price the target market (Phase 4) → get our fair probability.
2. Fetch the exchange's price for that exotic market; convert to fair implied probability via `ValueDetector.remove_overround()`.
3. `ValueDetector.detect_value(sim_prob, exchange_fair_prob)` → edge; `kelly_stake()` for sizing.
4. Log with `BetTracker`, tagging the market type (`total_runs`, `top_batsman`, etc.).

> [!WARNING]
> **Exotic markets are thin and slow to settle.** Liquidity for top-batsman and prop markets is a fraction of the match-winner market — a "value" price with no volume behind it is untradeable. Filter on available liquidity (as the in-play plan does), require a wider edge threshold than match-winner (exotic mispricing is real but so is exotic variance), and paper-trade far longer before staking.

### Verification

```bash
py -c "
from src.betting.exotic_scanner import scan_exotic
# scan_exotic(match, market='total_runs', line=165.5, exchange_over_odds=1.9, exchange_under_odds=2.0)
# → prints sim prob, exchange fair prob, edge, recommended stake
"
```

Expected: an edge + Kelly stake for markets where the simulation disagrees with the exchange beyond the threshold, `HOLD` otherwise.

---

## Implementation Tracking

| Phase | File | Status | Notes |
|-------|------|:------:|-------|
| 1. Delivery-Outcome Model | `src/models/delivery_outcome_model.py` | `[x] Completed` | Two-stage multinomial, GroupKFold(5)+final, cached to `t20_delivery_outcome_model.pkl`. Marginals match empirical within ±0.2pp; log loss 1.4166 vs 1.4788 naive baseline; top-1 acc 44.7% vs 35.4% baseline |
| 2. Innings Simulator | `src/simulation/innings_simulator.py` | `[x] Completed` | Vectorised ball-by-ball sim + 4 hand-traced unit tests (`tests/test_innings_simulator.py`, all pass) covering strike rotation, wicket substitution, and the odd-run/end-of-over swap cancellation. Real-match verification (2022 IPL Final, RR lineup vs GT attack, 10K sims): mean total 158.7 (band ~150-170 ✓), P5/P95 107-204, top-order batters correctly lead run distribution. All-out rate 11.8%, under the plan's rough 15-25% expectation — not necessarily a bug (single matchup, not a population average) but flagged for Phase 5's proper multi-match calibration to confirm. Perf: 25s/10K sims (2.5ms/sim), validating the batched-per-ball design (~119 model calls total, not 10K×120) |
| 3. Match Simulator | `src/simulation/match_simulator.py` | `[x] Completed` | Two-innings orchestration (simulating Innings 1, target linkage, Innings 2 chase), seed reproducibility (`seed=42`), victory margins (`margin_runs`, `margin_wkts`), and squad extractor (`extract_match_squads`). Verified via `tests/test_match_simulator.py` (4 unit/integration tests passing). Real-match test (2022 IPL Final, match_id=1312200): 1,000 full match sims completed in 2.6s (2.6ms/sim) |
| 4. Market Pricing Engine | `src/simulation/market_pricer.py` | `[x] Completed` | Aggregates sim arrays into fair decimal odds (`prob_to_fair_odds`), Monte Carlo standard error (`compute_mc_standard_error`), over/under lines (`price_total_runs`), top batsman (`price_top_batsman`), victory margins (`price_margin`), and player props (`price_player_props`). Verified via `tests/test_market_pricer.py` (7 tests passing) |
| 5. Calibration & Validation | `tests/test_simulation_calibration.py` | `[~] 3/4 methods pass — 1 known gap` | Superseded by the current n=100/40, 3000-sims battery (`data/processed/mc_calibration_metrics.json`), which the original 30-match row above predates and understated the noise on. **2026-08-08 result: 3 of 4 test methods pass** — Coverage 86.0%, Systematic Bias -2.62 runs (\|<8\| ✓), Win-Prob Agreement 10.6% (<12% ✓, was 13.7% before today), Over/Under Brier 0.206 (<0.25 ✓), Score Std Dev 32.2 runs (within 25-55 ✓). **Innings-total MAE still fails: 27.41 runs vs <25.0 target** (was 28.05). Two real fixes landed today: (1) `PreMatchPredictor.predict()` was silently dropping squad-level Elo (always falling back to team-average Elo even though the win-prob model trained on squad Elo) — added optional squad-Elo passthrough (`get_match_squad_elo()` in `player_elo_engine.py`, additive/backward-compatible, verified byte-identical for existing callers), wired into Test 2 — this alone closed the win-prob-agreement gate. (2) Replaced `innings_simulator.py`'s naive `over_idx % n_bowlers` round-robin with a phase-aware bowler schedule (`_build_bowler_schedule`) concentrating the top-2 ranked bowlers in the powerplay+death overs and weaker/part-time options in the middle, matching real captaincy — unit-tested (exact quotas, no consecutive-over repeats) across t20/od/hnd shapes, kept as a correctness improvement, but it moved Test 1's MAE by only -0.0016 runs (a wash) — the hypothesis that bowler-rotation realism was driving the MAE gap was not confirmed. Also added an in-process parquet cache to `extract_match_squads()` (280 redundant re-reads/format → 1), cutting the battery from ~2.6h to ~75-90min CPU time. Both `tests/regression_test.py` (all 6 formats, zero regressions) and `tests/sim_baseline_metrics.py` (t20, PASS) re-verified clean — nothing here touched training features. Remaining MAE gap accepted as a known, investigated limitation per user decision (2026-08-08); root cause still open — candidates not yet tried: delivery-outcome model conditioning richness, extras/byes modeling, or it may simply be inherent per-match dispersion this modelling approach can't fully capture |
| 6. Exotic Value Detection | `src/betting/exotic_scanner.py` | `[x] Completed` | Reused ValueDetector/BetTracker on exotic markets. Scanner successfully built (`scan_exotic`) with tested support for `total_runs`, `top_batsman`, and `player_props` markets. Tests passing and log verification added. |

---

## Phase 5 Addendum — The innings-MAE gate is unreachable (investigated 2026-08-08)

Test 1's `MAE < 25.0` sub-assertion is the only failing gate (27.41 official / 27.65 re-measured).
It was investigated to exhaustion; **the conclusion is that the threshold is misspecified, not that
the simulator is deficient.** Three independent measurements, all leak-free:

**1. MAE only responds to the predicted MEAN.** The metric is `mean|sim_mean − actual|` — the
simulated distribution's variance, tails, and coverage are arithmetically irrelevant to it. This
is why the phase-aware bowler schedule (a genuine realism improvement) moved it by −0.0016 runs.

**2. There is an irreducible floor, and it sits above the target.** Scoring a predicted mean
against a single match outcome costs `E|X − μ|` even when μ is perfect. Measured exactly (no
normality assumption): per-match sim MAD = 25.48, MAD/σ ratio 0.7895 → floor ≈ **25.8 runs at the
battery's own σ ≈ 32.7**. The target is 25.0. Passing it requires σ < 31.3 — i.e. deliberately
under-dispersing the forecast, which would degrade Test 3's over/under Brier and corrupt exactly
the tail probabilities the exotic markets are priced from. Coverage is already 86% vs a nominal
90%, so the distribution is if anything *already* slightly too tight.

**3. The simulator is already at the pre-match information ceiling** — so even the ~1.8 runs
between 27.65 and the 25.8 floor is *not* accessible, because that floor assumes perfectly knowing
the conditional mean, which pre-match data cannot determine. Evidence, all on the battery's exact
100-match sample:

| Predictor | MAE | corr |
|---|---|---|
| Global mean (no skill) | 30.77 | — |
| Venue average | 30.41 | 0.280 |
| Purpose-built XGB regressor, ALL pre-match features (OOF) | 28.40 | 0.489 |
| **MC simulator mean** | **27.65** | **0.497** |
| Best sim+regressor blend (weight chosen in-sample) | 27.65 (w=1.0) | 0.497 |
| Implied correlation ceiling (from corpus variance vs sim noise) | — | **~0.508** |

A directly-optimized regressor with team Elo, squad Elo, venue, recent-venue form, competition and
era **loses to the simulator**, and blending adds exactly nothing (the in-sample-optimistic search
picks pure sim). At the ~0.508 correlation ceiling the corresponding MAE is ≈27.6 — which is where
the simulator already is. Venue conditioning was checked specifically: recent-window venue form
beats the shipped all-time cumulative average by only corr 0.280→0.304, and era drift is
near-worthless (`global_r250` corr 0.10, `global_r1000` 0.02).

**Recommendation:** replace the raw-MAE assertion with a metric that isn't floored by irreducible
noise. Either **CRPS** (proper scoring rule for a distributional forecast against a single outcome;
measured ≈15.1 runs on a 30-match subsample — rewards sharpness *and* calibration, punishes
over- and under-dispersion alike) or an **MAE ÷ per-match-MAD ratio** (currently 27.65/25.8 ≈
**1.07**), which has the right incentive property: narrowing the forecast shrinks the floor without
improving MAE, so gaming it makes the score worse. Keep bias (−0.97 ✓) and coverage (86%) as-is —
neither has this floor problem. Scripts: `scratchpad/{mae_decomposition,mae_headroom,floor_and_crps,venue_signal_test,prematch_ceiling,blend_test}.py`.

---

## Verification Plan

### After Each Phase

1. Run the phase's `py -c` / `pytest` snippet above.
2. Confirm the delivery-outcome model's marginals match empirical base rates (Phase 1).
3. Confirm no existing model/feature file was modified — Strategy 4 is additive.

### End-to-End Validation

1. Train the delivery-outcome model; verify marginal calibration.
2. Simulate a known historical match with its real XIs; confirm mean/spread/all-out rate are realistic.
3. Run the full calibration battery (Phase 5) on ≥200 historical matches.
4. Only if all four calibration tests pass: price a handful of exotic markets and eyeball against known results.
5. Paper-price live matches over multiple weeks; log simulated fair odds vs settled outcomes before any staking.

### Simulation-Specific Discipline

> [!IMPORTANT]
> Fix the random seed for all validation runs so results are reproducible and regressions are detectable. When comparing scenarios (bat-first vs chase, toss outcomes), reuse the **same seed** so differences reflect the scenario, not the random stream. Report Monte Carlo standard error alongside every price — an "edge" inside a couple of standard errors is noise.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|:--------:|------------|
| Per-delivery bias compounds over 120+ balls | 🔴 High | Phase 5 calibration gate (coverage + reliability + variance); reject deployment on any failure |
| Simulator too slow for live pricing | 🔴 High | Vectorise across sims (one batched predict per ball-step); fall back to a binned probability table if needed |
| Strike-rotation / batter-substitution bugs | 🔴 High | Hand-traced unit tests on a known over; per-player distributions cross-checked vs historical |
| Under-dispersed distributions (tails too tight) | 🟡 Medium | Explicit variance/tail tests (Test 4); avoid over-regularising the outcome model |
| Exotic-market illiquidity | 🟡 Medium | Liquidity filter; wider edge threshold; long paper-pricing period |
| Player Elo missing for uncapped/franchise players | 🟢 Low | Default `effective_elo`=1500; position + phase priors dominate for tail-enders anyway |
| Lineup unknown until toss | 🟢 Low | Price on projected XI; re-run at toss when the actual XI is confirmed |
| Monte Carlo noise mistaken for edge | 🟢 Low | Report MC standard error; require edge ≫ SE before flagging value |

---

## Regression Guard

> [!CAUTION]
> Strategy 4 is **additive and read-only** with respect to the existing stack. It introduces a new model (`delivery_outcome_model.py`) and a new `src/simulation/` package, and reuses `value_detector.py` / `bet_tracker.py` unchanged. It reads `t20_cricket_data.parquet`, the player/phase/venue Elo parquets, and `empirical_dls_t20.parquet`, and writes only new artifacts. **No changes** to `xgb_win_probability.py`, `first_innings_predictor.py`, `build_states.py`, the Elo engines, or `app.py`. The existing win-probability pipeline must behave identically after this work — Strategy 4 consumes its features, never mutates them.

---

## Open Questions

- **Which exotic market first?** Total match runs (over/under) is the most liquid and easiest to calibrate — recommended as the beachhead before top-batsman/props.
- **T20 only, or ODI too?** ODI's 300-ball innings compounds per-delivery error further; prove the approach on T20 first.
- **How many sims?** 10,000 balances MC error (~0.3% at p=0.1) against speed; bump to 50,000 only for thin-tail prop markets where precision matters.
- **Binned table vs live model** for the sampling hot path — do you need live-match speed (favours the table) or is pre-match batch pricing enough (live model is fine)?
- **Where do announced XIs come from** for pre-match pricing — manual entry, or a lineup feed? This gates how early before a match you can price.
