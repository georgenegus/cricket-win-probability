# T20 Monte Carlo Innings-Total MAE — Extracting the Remaining Performance

> **Author:** Antigravity AI · **Date:** 2026-08-11
> **Purpose / Strategy:** Close as much of t20's Monte Carlo innings-total MAE shortfall as the available data actually supports, and settle on evidence whether the 25.0 gate is attainable at all
> **Baseline / Metrics:** t20 MC calibration — MAE **27.57** (gate ≤25.0, FAILING), bias **−1.16** ✅, coverage 0.86 ✅, win-prob agreement 0.1063 ✅, O/U Brier 0.2047 ✅, avg_std_dev 32.31 ✅. Zero-skill MAE floor (predict the median every time) = **29.19**
> **Defaults / Scope:** The Monte Carlo simulator and the delivery-outcome model only. The win-probability and first-innings models are out of scope and must not move. `t20` only — no other format's gates, features or baselines change.

---

## Executive Summary

t20's MC simulator sits at MAE **27.57** against a **25.0** gate. Bias is already exhausted as a
lever (`mc_runrate_feedback_plan.md` Phase 3: zeroing all remaining bias moves MAE by **0.02
runs**), so this is entirely a *skill/spread* problem: making the simulator's point forecast track
real innings totals better.

The uncomfortable headline, measured rather than assumed:

| Evidence (measured 2026-08-11, leakage-safe, 5-fold OOF, n=10,013 full-length innings) | Value |
|---|---|
| Innings-total SD / zero-skill MAE floor | 37.74 / **29.19** |
| Simulator today | **27.57** (only **5.6%** better than a zero-skill median guess) |
| Best direct regressor on venue par + team Elo | 27.63 (R² 11.7%) |
| Best direct regressor on **the information the simulator already consumes** (+ named-XI squad Elo) | **26.61** (R² 18.6%) |
| Best direct regressor on **everything available in the repo** (+ competition, era, toss) | **26.29** (R² 20.3%) |
| R² required to reach the 25.0 gate | **31.1%** |

> [!CAUTION]
> **The 25.0 gate is not attainable from the data this repo currently holds.** A direct
> match-level regressor given *every* pre-match feature available — venue par, team Elo, named-XI
> squad Elo, competition, era, toss — tops out at OOF MAE **26.29**, still **+1.29** over the gate.
> The simulator cannot beat a purpose-built point forecaster fed the same inputs, so no amount of
> simulator tuning reaches 25.0. Closing the gate requires **information the repo does not have**
> (pitch/dew/weather, confirmed team news, ground dimensions) — a data-acquisition project, not a
> modelling one. This plan therefore has two tracks and does **not** promise the gate.

**But there is real, worthwhile headroom:** the simulator (27.57) is **~1.3 runs behind** the best
leakage-safe regressor on its *own* inputs (26.29–26.61). That gap is genuine forecasting quality
being left on the table, worth capturing on its merits regardless of where the gate sits. This
plan targets that ~1.3 runs (about **half** the current 2.57-run gap to the gate), then re-anchors
the gate on evidence.

> [!IMPORTANT]
> **A leakage trap was already hit and corrected while scoping this plan — do not re-introduce
> it.** The first feasibility probe aggregated squad Elo over players who *actually batted or
> bowled*, which encodes the outcome: a side that scores heavily loses fewer wickets and therefore
> uses fewer batters. `n_batters_used` **alone** scores **35.7% OOF R²** — pure leak. Rebuilding
> over the **named XI** with **strictly-prior** ratings dropped the squad-Elo tier from a fictitious
> 26.3% R² / 25.20 MAE to an honest **18.6% / 26.61**. Every feature in this plan must be derivable
> from the toss-time team sheet and prior matches only.

### Phase Overview

| # | Phase | Tier | Effort | Retrain? |
|---|-------|------|:------:|:--------:|
| 0 | Diagnose the simulator's ~1.3-run shortfall vs its own inputs | Analysis | 🟡 Medium | No |
| 1 | Zero-retrain wins: point estimator + OOF recalibration | Core Logic | 🟢 Low | **No** |
| 2 | Price candidate features OOF (squad context, competition, era, toss) | Analysis | 🟡 Medium | No (subsampled) |
| 3 | Single retrain of the winning feature set | Integration | 🔴 High | **Yes** |
| 4 | Re-anchor the gate on evidence; re-lock + docs | Validation | 🟢 Low | No |

---

## Phase Execution Flow

```
Phase 0: attribute the 1.3-run gap -- estimator? recalibration? missing squad context?
    │        no production change; decides which of 1/2/3 is worth doing
    ▼
Phase 1: median-vs-mean point estimator + OOF affine recalibration
    │        no retrain -- cheapest possible win, fully reversible
    ├── gap closed enough ──► Phase 4
    ▼
Phase 2: OOF ablation of candidate features (subsampled, control-relative, ships nothing)
    │        prices each feature BEFORE paying for a full retrain
    ▼
Phase 3: ONE retrain of whichever set Phase 2 justifies
    │        gated on sim_baseline log-loss AND all 4 MC calibration tests
    ▼
Phase 4: re-anchor the 25.0 gate on the measured ceiling; re-lock + document
```

---

## Phase 0: Diagnose the Simulator's Shortfall

**Files:** `scripts/measure_t20_mae_decomposition.py` (NEW) · **Priority:** 🔴 Critical · **Retrain:** No

### Problem

We know the simulator is ~1.3 runs behind a regressor on its own inputs. We do **not** know why.
Three candidate mechanisms, each with a completely different fix and cost:

1. **Wrong point estimator.** MAE is minimised by the **median**, but
   `generate_mc_calibration_metrics.py` scores `scores.mean()`. Simulated innings totals are
   left-skewed (collapses form a long low tail), so mean ≠ median and the mean is the wrong
   statistic for an MAE gate. Cost to fix: one line.
2. **Mis-calibrated across matches.** If the simulator under-differentiates matches (its means
   are squashed toward the population mean) or over-differentiates, an affine recalibration
   `a + b·sim_mean` fitted out-of-fold reduces MAE with no model change. `b > 1` ⇒
   under-differentiating; `b < 1` ⇒ over-reacting.
3. **The per-ball model cannot see team quality.** `FEATURES` contains `batter_elo`,
   `bowler_elo`, `matchup_elo_diff`, `non_striker_elo` — but **no squad-level aggregate**. Each
   delivery is priced knowing only the two players involved, never whether this is a strong XI
   against a weak attack. The measured jump from team Elo (11.7% R²) to named-XI squad Elo
   (18.6%) is signal the simulator is structurally blind to. Cost to fix: a retrain.

### Implementation

1. Re-run the t20 innings-1 calibration sample (100 matches, `--n-sims 3000` fidelity) but
   **persist per-match rows**: `sim_mean`, `sim_median`, selected quantiles, `sim_sd`, the actual
   total, and the match's pre-match inputs (venue par, named-XI squad Elo aggregates).
2. **Estimator test.** Recompute MAE using the median, and sweep the predictive quantile
   `q ∈ [0.40, 0.60]` to find the MAE-optimal one. Report MAE at each.
3. **Recalibration test.** Fit `actual ~ a + b·sim_mean` **out-of-fold** (KFold by match; the
   match is the unit of independence — never pool deliveries, per this repo's hard-won rule) and
   report slope `b`, its SE, and the OOF MAE after recalibration.
4. **Headroom attribution.** On the same 100 matches, compute the OOF MAE of the direct
   leakage-safe regressor. Decompose the 1.3-run gap into: estimator choice, recalibration,
   residual (⇒ genuinely needs Phase 2/3).
5. Sanity-check that `sim_sd` (32.31) vs realised error SD (34.5) under-dispersion is not itself
   distorting the mean.

### Verification

```bash
py scripts/measure_t20_mae_decomposition.py
py tests/sim_baseline_metrics.py --formats t20   # unchanged; Phase 0 ships nothing
```

Expected: a per-mechanism attribution table summing to ≈1.3 runs, and a go/no-go for each of
Phases 1–3.

> [!NOTE]
> **100 matches carries ~±3 runs of MAE sampling noise.** Any effect Phase 0 measures that is
> smaller than ~1 run must be confirmed on a larger match sample (300+) before it is believed.
> Prefer raising match count over `--n-sims`: the metric's precision is set by the match sample,
> not by within-match sim noise.

### Results (2026-08-11)

**`scripts/measure_t20_mae_decomposition.py`** — direct regressor over the full corpus (n=10,013,
confirming the earlier probe: OOF R²=20.1%, MAE=26.27), then the 100-match calibration sample
simulated at full fidelity (n_sims=3000, ~50 min).

**On this run, the simulator scored MAE 26.53** — a full run below the locked 27.57. Before
reading anything into that, it had to be checked against exactly the failure mode this repo has
hit four times before (`mc_runrate_feedback_plan.md`'s Phase 3 note): **an underpowered sample
producing an apparent effect that is really just noise.**

| Estimator sweep | MAE |
|---|---:|
| mean (today's convention) | 26.53 |
| median | 26.42 |
| best single quantile (q=0.49–0.50) | 26.42 |

| OOF recalibration (`actual ~ a + b·sim_mean`) | Value |
|---|---:|
| slope `b` | 1.104 ± 0.041 SE |
| OOF MAE after recalibration | 26.85 |

`b ≈ 1.10` is close enough to 1 to read as "already reasonably well-scaled" — not the
`b ≫ 1` or `b ≪ 1` that would indicate a real miscalibration worth correcting.

> [!WARNING]
> **The raw 26.53-vs-27.57 gap is sampling noise, not a real improvement — checked, not assumed.**
> The per-match absolute-error SE on this sample is **2.28 runs** (95% CI [22.06, 31.01]); both
> the Aug-9 lock (27.57) and the Aug-10 fast-fidelity confirmation (27.54, run on the *same*
> post-refresh player-Elo cache as this run — a stale-cache explanation was checked and ruled
> out) fall comfortably inside that interval. This is expected: t20's per-match error SD is ~35
> runs, so 100 matches was never going to pin down the *mean* absolute error tightly. It is a
> textbook illustration of why this repo's own convention is to trust **paired** differences on
> a fixed match sample, not two independent point estimates.

**Paired decomposition** (same 100 matches, same sims — this is the number that matters, and it
tells a different, more sober story than the raw MAE gap above):

| Comparison | Effect | SE | t | Verdict |
|---|---:|---:|---:|---|
| mean vs median estimator | +0.114 runs | 0.079 | **+1.44** | Not significant — closest to it, directionally consistent with left-skew |
| today vs OOF-recalibrated | −0.315 runs | 0.322 | −0.98 | Not significant |
| today vs direct-regressor ceiling | −1.084 runs | 0.878 | −1.24 | Not significant — **cannot establish the simulator is actually behind its own inputs at this sample size** |

**None of the three candidate mechanisms clears this repo's own `|t| > 2` bar at n=100.** The
plan's Executive Summary framed a ~1.3-run gap to decompose; that number came from comparing two
*independent* 100-match measurements (locked simulator MAE vs. the full-corpus regressor's global
MAE) and, per the table above, is not distinguishable from zero once measured as a genuine paired
effect on one fixed sample. This doesn't mean the gap doesn't exist — it means **this measurement
cannot resolve it**, which is a materially different, more honest conclusion.

**Dispersion sanity check:** avg simulated std dev 32.60 vs realised error SD 35.08 (ratio 0.93,
"roughly matched") — no sign of the under/over-dispersion failure mode that would itself distort
the mean.

> [!CAUTION]
> **Correction (2026-08-11, discovered while shipping Phase 1a): the 100-match sample above was
> NOT byte-identical to `generate_mc_calibration_metrics.py`'s official calibration sample.**
> `scripts/measure_t20_mae_decomposition.py`'s population filter excluded no-balls from
> `legal_balls` (a cricket-rules-correct definition); the canonical script excludes only wides.
> That shifted the "full-length innings" boundary for a handful of matches, changing the
> population size by exactly one match (10,013 vs the correct 10,014) — enough to cascade through
> `np.linspace` sampling into **23 of the 100 matches being different** (77% overlap). The bug is
> now fixed in the script (verified byte-identical to the canonical population); this note exists
> so the numbers above are read with the right caveat rather than silently corrected away.
>
> **Practical impact, now measured:** the full-fidelity re-lock of the *correct* population
> (Phase 1's Results, below) puts the real median-vs-mean effect at **only −0.09 runs** (27.48 vs
> 27.57) — smaller than this section's own +0.114-run estimate, and further from significance,
> not closer. **The qualitative conclusion is unchanged and if anything reinforced**: none of the
> three mechanisms was ever going to clear `|t|>2` at n=100, and the honest read is still "this
> measurement cannot resolve a gap this small." Treat the specific numbers in the table above as
> illustrative of the methodology and the order of magnitude, not as precise values — the
> Phase 1 Results section below has the authoritative ones for the change actually shipped.

**What this means for Phases 1–3:**
- The median-estimator swap (Phase 1a) is the only candidate with a *directionally* consistent,
  near-significant signal (t=+1.44), it's free (no retrain), reversible, and independently
  justified on first principles (MAE is minimised by the median on a skewed distribution,
  regardless of what this one sample shows) — **worth shipping on its own merits**, not on this
  measurement's statistical authority.
- OOF recalibration (Phase 1b) has no case at this sample size (t=−0.98, and the point estimate
  is in the *wrong* direction) — **do not pursue** without a larger, separately-powered check.
- The squad-context retrain case (Phase 2/3) is **not established** by this measurement (t=−1.24,
  wrong-signed point estimate). The full-corpus ladder (n=10,013, `26.29` MAE) remains the
  credible ceiling estimate — trust that number over this 100-match slice — but confirming
  the simulator is actually *behind* it, specifically, needs a properly-powered check (300+
  matches, per this phase's own pre-registered caveat) before a retrain is justified.

**Full per-match rows:** `outputs/t20_mae_decomposition_rows.csv`. Summary:
`outputs/t20_mae_decomposition_summary.json`.

---

## Phase 1: Zero-Retrain Wins — Estimator and Recalibration

**Files:** `tests/one-time/generate_mc_calibration_metrics.py`, `src/simulation/market_pricer.py` (MODIFY) · **Priority:** 🔴 High · **Retrain:** No

### Problem

If Phase 0 confirms either mechanism (1) or (2), MAE improves with no model change at all — the
cheapest and most reversible possible win.

### Implementation

1. **Point estimator.** If the median (or an MAE-optimal quantile) beats the mean, adopt it *for
   the MAE metric*, keeping **bias** mean-based — bias is a statement about expectation, MAE about
   absolute error, and each is minimised by a different statistic. Document this split explicitly
   so it never reads as cherry-picking.
2. **Recalibration.** If slope `b` is significantly ≠ 1, apply an OOF-fitted affine correction to
   the simulator's *reported point forecast*. Fit per format; never fit on the same matches it is
   scored on.
3. Guard both behind a named constant defaulting to today's behaviour, so the change is a
   documented, reversible switch (same discipline as `SCORE_FEEDBACK_SHRINK = 1.0`).

> [!CAUTION]
> **Recalibration must not silently distort the other three markets.** Shifting or rescaling the
> point forecast can degrade coverage, O/U Brier and win-prob agreement — all of which currently
> pass. If a recalibration improves MAE but pushes any other metric out of range, it is **not**
> acceptable: the MC output feeds live betting markets, and a better MAE bought with worse
> over/under pricing is a net loss. Prefer applying the correction only to the reported point
> estimate, leaving the sampled distribution (and everything priced from it) untouched.

### Verification

```bash
py tests/one-time/generate_mc_calibration_metrics.py --format t20 --n-sims 1000   # _fast_ file
# final candidate only, at full fidelity:
py tests/one-time/generate_mc_calibration_metrics.py --format t20
```

Gate: MAE improves, **and** bias stays within ±8.0, coverage inside [0.70, 0.95], win-prob
agreement ≤0.12, O/U Brier ≤0.25, avg_std_dev inside [25, 55].

### Results (2026-08-11)

**Only Phase 1a (point estimator) was pursued.** Phase 0 found OOF recalibration had a
wrong-signed, non-significant effect (t=−0.98) — no case to act on, so Phase 1b was not
implemented. `market_pricer.py` was **not touched**: its `mean_score` field is a live betting-UI
display value, a materially different, higher-stakes surface than the calibration gate's own
metric, and nothing in Phase 0 justified changing it.

**Implementation:** `MAE_POINT_ESTIMATE = {"t20": "median"}` — a per-format dict in
`tests/one-time/generate_mc_calibration_metrics.py` (mirroring `SCORE_FEEDBACK_SHRINK`'s
precedent), defaulting every other format to `"mean"` (today's behaviour, unchanged). Confirmed
via smoke test that `od` still reports `estimator=mean`. **`bias` stays mean-based unconditionally
for every format** — bias estimates expectation, MAE estimates absolute error, and the two are
deliberately not the same statistic. The duplicate calibration battery in
`test_simulation_calibration.py` was updated identically (that directory's hyphenated name blocks
a normal import, so the split is duplicated with a "keep in sync" comment, matching this repo's
existing pattern for that file pair). The output JSON now also carries `mae_point_estimate` and
`mae_mean_based_diagnostic` for transparency.

> [!NOTE]
> **A sampling bug was caught and fixed while verifying this, and it changes how much to trust
> Phase 0's specific numbers (not the shipped code).** `scripts/measure_t20_mae_decomposition.py`'s
> population filter subtly diverged from this script's own (excluded no-balls from `legal_balls`;
> the canonical definition excludes only wides), which shifted the full-length-innings boundary
> for a few matches and changed the 100-match sample by 23 matches out of 100. Fixed and verified
> byte-identical to this script's population. The fix was cross-checked for free: `bias` in the
> re-lock below reproduces the old locked value to nine decimal places
> (`-1.1552600000000006`), which is only possible if the underlying simulation — same matches,
> same seeds, same model — is exactly reproduced. See Phase 0's Results section for the full
> correction note.

**Full-fidelity re-lock** (`generate_mc_calibration_metrics.py --format t20`, n_sims=3000, 100
matches, all 4 tests) overwrote the canonical `data/processed/mc_calibration_metrics.json`:

| Metric | Old lock (mean-based MAE) | **New canonical (median-based MAE)** | Gate |
|---|---:|---:|---:|
| MAE | 27.5735 | **27.48** | ≤25.0 ❌ (unchanged — see below) |
| Bias | −1.15526 | −1.15526 *(byte-identical)* | ≤8.0 ✅ |
| Coverage | 0.86 | 0.86 *(byte-identical)* | [0.70, 0.95] ✅ |
| Win-prob agreement | 0.10634 | 0.10634 *(byte-identical)* | ≤0.12 ✅ |
| O/U Brier | 0.20467 | 0.20467 *(byte-identical)* | ≤0.25 ✅ |
| avg_std_dev | 32.306 | 32.306 *(byte-identical)* | [25, 55] ✅ |

**Every metric except MAE reproduced to the last decimal place** — direct proof this was a
clean, fully isolated change with zero side effects on anything else the simulator produces,
exactly the property Phase 1's own CAUTION callout above required before shipping.

**The real effect (−0.093 runs) is smaller than Phase 0's own estimate (+0.114 runs on the
wrong-by-23-matches sample)** — consistent with, and if anything reinforcing, Phase 0's own
conclusion that this effect was never established at `|t|>2` and shouldn't be over-trusted at
its point value. It was shipped on the first-principles argument (median is MAE-optimal on a
skewed distribution), not because either measurement proved a large win.

> [!IMPORTANT]
> **This does not meaningfully move t20 toward its gate.** 27.48 vs the 25.0 target is still a
> 2.48-run miss — barely different from the 2.57-run miss before. Per the Executive Summary's own
> ceiling analysis (best achievable OOF MAE ≈26.29–26.61 on leakage-safe pre-match features), no
> zero-retrain change was ever going to close this gap; Phase 1a was always the "worth doing
> regardless, on principle" option, not a gate-closing one. **Phase 2/3 (squad-context retrain)
> remains unestablished and was not pursued** — Phase 0 found no significant evidence the
> simulator underperforms its own inputs at n=100 (t=−1.24, wrong-signed point estimate), so
> committing to an hours-long retrain isn't justified without a properly-powered (300+ match)
> confirmation first, per this phase's own risk table.

**Regression guard:** `sim_baseline_metrics.py --formats t20` and `tests/regression_test.py`
(all 6 formats) both pass with 0 regressions, run alone (this box OOMs under concurrent load;
`app.py` was stopped for the full-fidelity run).

---

## Phase 2: Price the Candidate Features (no ship)

**Files:** `scripts/measure_t20_feature_ablation.py` (NEW) · **Priority:** 🟡 Medium · **Retrain:** No (subsampled OOF only)

### Problem

A full t20 delivery-outcome retrain is expensive (2.4M deliveries, 5 folds, two heads) on a 7.7GB
box. Price each candidate **before** paying for it — exactly the pattern that worked in
`mc_runrate_feedback_plan.md` Phase 0, where a subsampled control-relative ablation correctly
predicted the full-corpus outcome.

### Implementation

Subsample by **match** (never by delivery), GroupKFold by `match_id`, hyperparameters identical to
production, and measure **control-relative** deltas (a control variant retrained on the same
subsample and folds — never against the full-corpus lock, which differs for reasons unrelated to
the features). Significance assessed match-paired, `|t| > 2`.

Candidate variants, cheapest first:

1. **`batting_squad_elo` / `bowling_squad_elo`** — named-XI aggregates (mean, and optionally max/
   spread) attached to every delivery. This is the mechanism Phase 0 hypothesis (3) targets and
   the measured 11.7% → 18.6% R² jump says it carries the most signal of anything available.
   Must be built from the **named XI with strictly-prior ratings**, not participants.
2. **`competition`** — 17 leagues with materially different scoring regimes; `venue_avg` only
   partially proxies it (a ground hosting both an international and a domestic competition gets
   one blended par). Encode as an OOF target-encoding or a small categorical, never an in-sample
   target encoding.
3. **`era`** — a date/season trend; T20 scoring has inflated materially over the corpus window.
4. **`toss_decision` / bat-first indicator** — known pre-match, currently unused.

Measured expectation from the match-level probe: (1) is worth the most; (2)+(3)+(4) together were
worth **0.32 runs** of MAE (18.6% → 20.3% R²) — real but small, so they only justify a retrain if
they ride along with (1) rather than requiring their own.

### Verification

```bash
py scripts/measure_t20_feature_ablation.py
py tests/sim_baseline_metrics.py --formats t20   # unchanged; Phase 2 ships nothing
```

Expected: per-variant OOF log-loss delta vs control (with match-paired `t`), plus — critically —
an estimate of the **innings-total MAE** effect, since log-loss improving does **not** guarantee
MAE improves. Where feasible, simulate a small match sample with the ablated fold models to get a
direct MAE read rather than inferring from log-loss alone.

### Results (2026-08-11)

**`scripts/measure_t20_feature_ablation.py`** (NEW) — subsampled, control-relative, match-paired,
following `measure_score_feedback.py`'s established `run_ablation` pattern exactly (same
GroupKFold-by-`match_id` discipline, same hyperparameters as production, only the legal-outcome
head ablated). Three variants, all **appending** to `FEATURES` (never reordering it): `control`
(today's 15 features), `+squad_elo` (control + `batting_squad_elo`/`bowling_squad_elo` from
`get_match_squad_elo` — named XI, strictly-prior, same source as Phase 0's regressor), `+all`
(`+squad_elo` plus `comp_code`/`era_year`/`toss_won_by_batting_side`/`toss_decision_field`).
`batting_squad_elo`/`bowling_squad_elo` were attached per delivery via the row's actual
`batting_team`/derived `bowling_team` (not a fixed "team_1 always bats first" assumption, unlike
the match-level scripts — per-ball data spans both innings) — squad-Elo coverage was 10,636/10,636
matches (100%), so no fallback-default rows entered the ablation.

Smoke-tested first at 60 matches / 1 fold (directionally promising: `+squad_elo` −0.0065,
`+all` −0.0078 log-loss vs control) before paying for the real pass — but per this plan's own
n=100 sampling-noise lesson from Phase 0, a small sample's point estimate is not trustworthy on
its own, so the full pass below is the one that counts.

**Full pass** (800 matches, 2/5 GroupKFold folds — the plan's specified scale, matching
`mc_runrate_feedback_plan.md`'s Phase 0 precedent — 174,859 legal deliveries, 69,936 scored OOF):

| Variant | OOF legal log-loss | Δ vs control | match-paired `t` (n=320) | Verdict |
|---|---:|---:|---:|---|
| control (15 features) | 1.452366 | — | — | — |
| `+squad_elo` (17 features) | 1.452299 | −0.000067 | **+0.38** | Not significant — indistinguishable from zero |
| `+all` (21 features) | 1.451187 | −0.001180 | **−1.37** | Not significant |
| `+all` vs `+squad_elo` (incremental) | — | −0.001113 | **−1.74** | Not significant — closest to the bar, still short |

> [!CAUTION]
> **None of the three variants clears this repo's own `\|t\|>2` bar, and the headline result is a
> real reversal from the smoke test.** At 60 matches / 1 fold, `+squad_elo` looked like a solid
> −0.0065 win; at the full 800-match / 2-fold scale it collapsed to −0.00007 (t=+0.38 — noise,
> wrong-signed at the point estimate). This is not a bug — it is exactly the "small sample
> produces an apparent effect that is really noise" failure mode Phase 0 flagged for itself,
> now reproduced independently on this phase's own ablation. Trust the full-scale numbers.
>
> **The per-ball log-loss ablation and the match-level MAE regressor are measuring genuinely
> different things, and they disagree.** Phase 0's direct regressor found squad Elo carries real
> signal for the match's *aggregate* total (R² 11.7%→18.6%). This ablation finds it carries
> almost no signal for any *individual ball's* outcome distribution. Both can be true at once: a
> team-quality effect that is a small, consistent per-ball nudge integrates into a large
> innings-total effect over ~120 balls, while being far too diffuse for a per-ball log-loss metric
> — scored on one outcome out of 7 classes per row — to detect at this sample size. This ablation
> does not contradict Phase 0's ceiling estimate; it says the *retrain-pricing metric this phase
> uses* is not sensitive enough to confirm that ceiling is reachable through this specific
> mechanism.

**What this means for Phase 3:** **Not justified by this measurement.** The plan's own Phase
Execution Flow gates Phase 3 on "whichever set Phase 2 justifies" — here, nothing clears the bar,
including the combined `+all` set that was explicitly supposed to "ride along with squad_elo"
per this phase's own instruction. The `+all`-vs-`+squad_elo` comparison (t=−1.74) is the closest
to significant of the three and directionally consistent with Phase 0's finding that
competition/era/toss are worth a small amount on top of squad_elo — but "closest to significant"
is not "significant," and this repo's standing practice (Phase 0's own conclusion, restated) is
not to spend an hours-long retrain chasing an effect that hasn't cleared its own bar. A
larger-scale rerun (more matches and/or more folds) could resolve whether `+all` is real or
also noise, but that is now a **separately justified follow-up**, not something this pass
resolves for free.

**Direct simulated-MAE read:** not attempted, by design (see module docstring) — threading new
features through `innings_simulator._assemble_state_matrix` for a throwaway pricing script risks
exactly the train/serve drift Phase 3's own risk table warns about, and is safer paid once, in
Phase 3, if a retrain is ever justified.

**Regression guard:** `sim_baseline_metrics.py --formats t20` reruns clean post-ablation
(`extras_log_loss` +0.000003, `legal_log_loss` −0.000331, both PASS) — confirms the script,
as designed, shipped nothing.

**Full ablation output:** `outputs/t20_feature_ablation.json`.

---

## Phase 3: Single Retrain of the Winning Set

**Files:** `src/models/delivery_outcome_model.py` (MODIFY), `data/processed/t20_delivery_outcome_model.pkl` · **Priority:** 🟡 Medium · **Retrain:** Yes

### Problem

Ship whatever Phase 2 justified, in **one** retrain rather than several — each t20 retrain is
hours of wall-clock and the box cannot run anything else meanwhile.

### Implementation

1. Add the winning features to `FEATURES`, **appending** rather than reordering — XGBoost's
   `colsample_bytree` is sensitive to column order under a fixed seed, so a reorder would confound
   the change with a sampling change.
2. Extend `build_training_frame()` to compute them leakage-safely; the simulator's
   `_assemble_state_matrix()` must reproduce each one **exactly** at sampling time. Any drift
   between the two silently skews every prediction the simulator makes.
3. Squad Elo is constant within a match, so it is computed once per innings, not per ball.
4. Back up the existing `.pkl` before retraining — model caches have **no** hash check
   (`fit(cache_path=...)` only tests existence), so a genuine retrain requires deleting the file.

> [!WARNING]
> **`t20` is a `POOL_SOURCE` for `hnd` and a `SEED_SOURCE` for its player Elo.** Changing t20's
> delivery-outcome model does not itself pool into hnd (pooling is a win-prob-model concern), but
> confirm `hnd`'s gates before declaring done — this plan claims t20-only scope and must prove it.
> Women's formats are entirely separate code paths and must not be touched, even incidentally.

### Verification

```bash
py tests/sim_baseline_metrics.py --formats t20            # the retrain gate (+0.01 tolerance)
py tests/one-time/generate_mc_calibration_metrics.py --format t20
py tests/regression_test.py                               # all formats, must be 0 regressions
```

Gate: OOF log-loss within the locked tolerance, **and** MAE improved, **and** no other MC metric
pushed out of range. A variant that improves MAE while degrading per-ball log-loss is trading a
genuinely better model for a metric win and is **not** acceptable.

---

## Phase 4: Re-Anchor the Gate and Document

**Files:** `tests/one-time/generate_mc_calibration_metrics.py`, `data/processed/mc_calibration_metrics.json`, `docs/system_guide.md`, `.agents/project-overview.md`

### Problem

t20's 25.0 was *"empirically chosen"* and has never been validated against a skill ceiling. This
plan's evidence says it sits **beyond** what the available data supports (best honest regressor:
26.29). Leaving a permanently-unreachable gate red is worse than useless — it trains everyone to
ignore a failing gate, which is exactly how a real regression gets missed.

### Implementation

1. Re-anchor t20's MAE gate on a **defensible, stated basis**, with the measured ceiling on
   record. Candidate bases, to be chosen deliberately rather than reverse-engineered from the
   current number:
   - a fixed fraction of the measured zero-skill floor (29.19), e.g. "must beat a median-guess by
     ≥ X%";
   - the measured attainable ceiling (26.29) plus a tolerance for the ~±3-run sampling noise at
     n=100;
   - retain 25.0 as an explicit *aspiration* while the gate itself sits at the attainable level.
2. Whatever basis is chosen, apply it through the **dispersion-based** cross-format scaling
   introduced 2026-08-11 (`innings_total_sd`), so every other format moves consistently and no
   format is silently re-tightened.
3. Re-lock `mc_calibration_metrics.json` at full fidelity (`--n-sims 3000`); `_fast_` files must
   never become the baseline.
4. Document the measured ceiling in `docs/system_guide.md` §15 next to the existing simulator
   notes, so the next person does not re-derive it.

> [!IMPORTANT]
> Re-anchoring is a **judgement call about what standard the simulator should meet**, not a
> measurement, and it moves every format's gate at once through the shared scaling. It should be
> an explicit, recorded decision — never a quiet edit that happens to make a red gate green.

### Results (2026-08-11)

**The re-anchor basis was put to the user as an explicit choice** (per this phase's own IMPORTANT
callout — not something to decide quietly), with three candidate bases and their numeric
consequences spelled out: (A) ceiling (26.29) + ~1 sampling-noise SE (2.28, Phase 0's measured
per-match absolute-error SE at n=100) ≈ **28.5**; (B) lock in current performance with a small
fixed buffer ≈ 28.0; (C) leave 25.0 unchanged and document it as a permanently-red, data-
acquisition-blocked aspiration. **Option A (28.5) was chosen.**

**Implementation:** `_T20_MAE_MAX = 28.5` in `tests/one-time/generate_mc_calibration_metrics.py`
(was `25.0`), with the full rationale recorded in a comment above the constant. The duplicate
gate in `test_simulation_calibration.py` (that directory's hyphenated name blocks importing the
constant directly, matching this plan's existing pattern for `MAE_POINT_ESTIMATE`) was synced to
the same value via a local `mae_gate = 28.5` variable. This is every other format's MAE-gate
anchor too, via `_calibration_targets()`'s `innings_total_sd`-based scaling — re-anchoring t20
moved every format's gate at once, not just t20's, exactly as the IMPORTANT callout above warns.

**Full-fidelity re-lock** (`generate_mc_calibration_metrics.py --format t20`, n_sims=3000, 100
matches, all 4 tests):

| Metric | Value | Gate | Result |
|---|---:|---:|---|
| MAE | 27.48 | ≤28.5 | **PASS** (was FAIL ≤25.0) |
| Bias | −1.15526 | ≤8.0 | PASS |
| Coverage | 0.86 | [0.70, 0.95] | PASS |
| Win-prob agreement | 0.106 | ≤0.12 | PASS |
| O/U Brier | 0.2047 | ≤0.25 | PASS |
| avg_std_dev | 32.3 | [25, 55] | PASS |

**MAE and every other metric reproduced exactly** (bias to 9 decimal places: `-1.1552600000000006`)
— proof this re-lock changed only the recorded gate threshold, not the model, the simulator, or
any measured value. **t20's MC calibration gate now PASSES for the first time** — not because the
simulator got better, but because the threshold now reflects what this repo's data can actually
support, on the record, with the ceiling analysis that justifies it.

**Documentation:** the measured ceiling, the Phase 1/2 findings, and the re-anchor rationale are
now recorded in `docs/system_guide.md` §15 (new subsection, next to the existing simulator notes)
and `.agents/project-overview.md` (Model Performance section), so the next person does not
re-derive any of this. `od`'s own cached `mc_calibration_metrics_od.json` was **deliberately not
regenerated** — its measured values (bias, coverage, etc.) are unaffected by t20's constant, its
own MAE gate is explicitly deferred work per this plan's own Deferred section, and re-simulating
purely to refresh a stale display field wasn't judged worth an hours-long rerun; this is noted as
a known, minor staleness rather than silently left unexplained. `hnd`/`t20w`/`odw`/`hndw` have no
`mc_calibration_metrics_<fmt>.json` on disk at all — this battery has never been run for them, so
there was nothing to go stale.

**Regression guard:** `sim_baseline_metrics.py --formats t20` and `tests/regression_test.py`
(all 6 formats) both re-run clean, 0 regressions — Phase 4 touched only gate thresholds and docs,
never the model or simulator, so this is confirmatory rather than a real risk.

---

## Implementation Tracking

| Phase | Component | Status | Notes |
|-------|-----------|:------:|-------|
| 0 | Diagnose the ~1.3-run shortfall | `[x] Completed` | `scripts/measure_t20_mae_decomposition.py`. Direct regressor confirmed on full corpus (n=10,013, R²=20.1%, MAE=26.27). 100-match sim (n_sims=3000) scored MAE 26.53 -- checked against the locked 27.57 and Aug-10 fast-fidelity 27.54, both fall inside this run's 95% CI (SE=2.28); ruled out a stale-cache explanation (player-Elo cache unchanged since the Aug-10 confirmation). **Paired decomposition (the number that matters): none of the 3 candidate mechanisms clears `\|t\|>2`** -- median vs mean estimator +0.11 runs (t=+1.44, closest to significant), OOF recalibration -0.32 runs (t=-0.98, wrong direction), vs direct-regressor ceiling -1.08 runs (t=-1.24, cannot establish simulator is behind its own inputs). Recommendation: ship the median-estimator swap (Phase 1a) on first-principles grounds (MAE-optimal statistic for a skewed distribution) since it's free and reversible, not because this measurement proves it; do NOT pursue OOF recalibration (Phase 1b) or the squad-context retrain (Phase 2/3) without a larger (300+ match) confirmation first -- this measurement is genuinely underpowered to resolve effects this small. `sim_baseline_metrics.py --formats t20` unchanged, as expected. **Correction (during Phase 1):** this script's population sampling had a bug (23/100 matches differed from the canonical script's sample); fixed and verified byte-identical. Re-measured on the correct population (Phase 1's full-fidelity re-lock), the real median-vs-mean effect is −0.09 runs, smaller than this table's own estimate — reinforces, doesn't overturn, the "not established" conclusion. |
| 1 | Zero-retrain wins: point estimator + OOF recalibration | `[x] Completed (1a only)` | Shipped `MAE_POINT_ESTIMATE = {"t20": "median"}` in `tests/one-time/generate_mc_calibration_metrics.py` (+ synced duplicate in `test_simulation_calibration.py`); `od` confirmed unaffected (`estimator=mean`). 1b (recalibration) **not implemented** — Phase 0 found no case (wrong-signed, t=−0.98). Full-fidelity re-lock: MAE 27.5735→**27.48** (−0.093 runs); bias/coverage/win-prob/O·U-Brier/avg_std_dev all **byte-identical** to the old lock — a fully isolated change, zero side effects. Gate still fails (27.48 vs ≤25.0) — this was never expected to close it (see Executive Summary's ceiling analysis), shipped on first-principles grounds only. `sim_baseline_metrics.py` + `regression_test.py` both pass, 0 regressions. Phase 2/3 (squad-context retrain) **not pursued** — unestablished at n=100 per Phase 0. |
| 2 | OOF feature ablation (no ship) | `[x] Completed` | `scripts/measure_t20_feature_ablation.py`. Subsampled (800 matches, 2/5 GroupKFold folds), control-relative, match-paired, mirroring `measure_score_feedback.py`'s `run_ablation`. **None of `+squad_elo` (t=+0.38), `+all` (t=−1.37), or `+all` vs `+squad_elo` (t=−1.74) clears `\|t\|>2`** — a 60-match/1-fold smoke test had looked promising (`+squad_elo` −0.0065) but collapsed to noise at full scale, the same small-sample failure mode Phase 0 already flagged for itself. Per-ball log-loss and the match-level MAE regressor (Phase 0) appear to be measuring different things: squad Elo's match-aggregate signal (R² 11.7%→18.6%) is apparently too diffuse per-ball for this metric to confirm at n=800. **Phase 3 not justified by this measurement.** `sim_baseline_metrics.py --formats t20` unchanged, as expected (ships nothing). |
| 3 | Single retrain of winning set | `[—] Not pursued` | Not justified — Phase 2's ablation found no feature set clearing `\|t\|>2` (closest: `+all` vs `+squad_elo`, t=−1.74). Skipping a multi-hour retrain that this plan's own gating logic doesn't support, per the same discipline Phase 0 already applied to itself. |
| 4 | Re-anchor gate + re-lock + docs | `[x] Completed` | User chose the re-anchor basis explicitly from 3 costed options: `_T20_MAE_MAX` 25.0 → **28.5** (ceiling 26.29 + ~1 sampling-noise SE) in `generate_mc_calibration_metrics.py` (+ synced `test_simulation_calibration.py`). Cascades to every format via `innings_total_sd` scaling. Full-fidelity re-lock: MAE **27.48 reproduced exactly**, all 4 tests **PASS** — t20's MC calibration gate is green for the first time. Documented in `docs/system_guide.md` §15 and `.agents/project-overview.md`. `od`'s snapshot deliberately left stale (out of scope, its own MAE gate is deferred work); `hnd`/`t20w`/`odw`/`hndw` have no snapshot to go stale. `regression_test.py` + `sim_baseline_metrics.py --formats t20` both pass, 0 regressions. |

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|:--------:|------------|
| Outcome leakage via participant-derived features (already hit once: 35.7% fake R²) | 🔴 High | Named XI + strictly-prior ratings only; sanity-check any new feature's standalone R² and its correlation with wickets lost |
| MAE win bought by degrading O/U Brier / coverage / win-prob (these feed live betting) | 🔴 High | All four MC tests gate every phase, not just MAE; prefer point-estimate-only corrections that leave the sampled distribution intact |
| Chasing a gate that is provably unreachable | 🔴 High | Executive Summary states the ceiling up front; Phase 4 re-anchors on evidence; success is defined as capturing the ~1.3 runs, not hitting 25.0 |
| Train/serve feature drift between `build_training_frame` and `_assemble_state_matrix` | 🔴 High | Any new feature must be reproduced identically in both; verify on a fixed match before retraining |
| Retrain cost (hours) spent on features worth ~0.3 runs | 🟡 Medium | Phase 2 prices everything OOF and subsampled *before* Phase 3 pays; one retrain, not several |
| 7.7GB box OOMs mid-retrain (already hit repeatedly this week) | 🟡 Medium | Keep the Flask app **down** during retrains/calibration — measured ~717MB and the direct cause of three OOMs on 2026-08-11; run heavy jobs strictly one at a time |
| n=100 sampling noise (~±3 runs) mistaken for a real effect | 🟡 Medium | Confirm any sub-1-run effect at 300+ matches; raise match count, not `--n-sims` |
| Log-loss improves but MAE doesn't (they are different objectives) | 🟡 Medium | Phase 2 measures MAE directly on a simulated sample, not only log-loss |

---

## Regression Guard

- **t20 only.** No other format's gates, features, models or baselines change. `hnd` (which
  pools/seeds from t20) must be re-checked before sign-off despite the claim of isolation.
- **Women's formats are untouched** — separate format codes, never a gender flag.
- **The win-probability and first-innings models are out of scope**; `tests/regression_test.py`
  must stay at 0 regressions throughout.
- **Both existing gates must still pass:**
  ```bash
  py tests/regression_test.py
  py tests/sim_baseline_metrics.py --formats t20
  ```
- **Every phase is independently revertible** — Phase 1 behind a default-off constant, Phase 3
  behind a backed-up `.pkl`.

---

## Deferred (explicitly not this plan)

- **Closing the last ~1.3 runs to 25.0.** Provably requires information the repo does not hold:
  pitch and dew reports, weather, confirmed (not named) team news, ground dimensions. That is a
  **data-acquisition** project with its own sourcing, licensing and freshness problems, and should
  be scoped separately if the gate is ever to be met as originally written.
- **od's MAE gate.** Same class of problem, different numbers (od sits at 46.42 against a
  dispersion-corrected 46.3). Out of scope here; see `mc_runrate_feedback_plan.md`'s "MAE Gate
  Follow-Up" section.
- **The residual +10-run od bias** that survives full damping — unattributed by any tracked
  feature or the (wickets, over) joint distribution. Tracked in `mc_runrate_feedback_plan.md`.
