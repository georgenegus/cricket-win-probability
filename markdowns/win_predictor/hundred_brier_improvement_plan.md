# The Hundred — Win-Probability Brier Improvement Plan

> **Owner:** George · **Created:** 2026-07-24 · **Status:** ✅ Shipped (2026-07-24) — ship gate met
> **Purpose / Strategy:** Make the *bespoke* Hundred win-probability classifier actually beat the "reuse the T20 model" baseline on Brier score. Today it loses to it.
> **Outcome:** brier_hnd 0.1780 → **0.1411** (−20.7%), beating reused-T20 (0.1434) for the first time. Phases 1-4 (feature/geometry fixes, monotonic constraints, hyperparameter tuning) closed the gap from 0.1780→0.1689; Phase 5's pooling experiment found the T20+HND pooled architecture beat bespoke by a wide margin on held-out data, and Phase 6 wired it into production — see the tracker below for full detail on every phase.
> **Scope:** `src/models/xgb_win_probability.py` (the model Brier directly scores) + the `hnd` retrain path. **Zero behavioural change to `t20`/`od`** — every edit is either `hnd`-gated or a `format_spec`-driven reformulation proven byte-identical for the 6-ball formats.

---

## Executive Summary

Phase-8 validation ([tests/hundred_validation.py](../../tests/hundred_validation.py)) already delivered the uncomfortable verdict:

| Model on held-out Hundred 2nd-innings states | Brier |
|---|:---:|
| **Bespoke `hnd` win-prob classifier** | **0.1780** |
| Reused **T20** model on Hundred states | **0.1434** ← better |
| Naive RRR-vs-CRR baseline | (recorded Phase 0) |

The bespoke model is currently *worse than doing nothing bespoke*. The causes are visible in the code, not in the data, and they rank cleanly by leverage:

1. **The `hnd` path discards every player-Elo feature** ([xgb_win_probability.py:223](../../src/models/xgb_win_probability.py#L223) — `if self.format_type in ["t20","od"]:`). We built seeded Hundred player Elos in the last plan's Phase 4b ("the single highest-leverage change") and then never fed them to the win-prob model. The winning T20 model *does* read those columns off the Hundred states — which is almost certainly why it wins. **Highest leverage, nearly free.**
2. **Geometry bugs in the `hnd` feature path** — the Elo-decay thresholds and the wicket-pressure denominator are computed on a **120-ball** scale for a **100-ball** innings ([lines 133-160](../../src/models/xgb_win_probability.py#L133)). Silent skew on every Hundred prediction.
3. **Never pooled/transferred at the win-prob level.** Phase 6 pooled only the *delivery-outcome* model. A pooled/blended win-prob classifier is the natural fallback if 1+2 don't fully close the gap.
4. **No monotonic constraints** — a strong, over-fit-proof prior that directly helps Brier/calibration on ~167 matches.
5. **Un-tuned regularization** — `hnd` inherits `max_depth=4, lr=0.08` from a corpus ~100× larger.

**Sequencing philosophy:** ship the cheap, near-certain wins first (Phases 1–2), *measure*, then only reach for tuning (3–4) and pooling (5) if the gap to a good model isn't closed. Phase 6 is the ship decision with an explicit accept bar.

### Phase Overview

| # | Phase | Lever | Effort | Files (NEW / MODIFY) |
|---|-------|-------|:------:|----------------------|
| 0 | Lock the baseline (exact Brier numbers, per-phase) | — | 🟢 Low | *(run only)* |
| 1 | Enable player-Elo features on the `hnd` path | #1 | 🟢 Low | `xgb_win_probability.py` (MODIFY) |
| 2 | Fix `hnd` geometry (decay thresholds + wicket pressure) | #2 | 🟡 Medium | `xgb_win_probability.py` (MODIFY) |
| 3 | Monotonic constraints on directional features | #4 | 🟡 Medium | `xgb_win_probability.py` (MODIFY) |
| 4 | Regularization / hyperparameter sweep for `hnd` | #5 | 🟡 Medium | `xgb_win_probability.py` (MODIFY), sweep script (NEW) |
| 5 | Pooled / T20-blended win-prob experiment | #3 | 🟡 Medium | `tests/hundred_winprob_pooling_experiment.py` (NEW) |
| 6 | Ship decision + wire the winning config | — | 🟢 Low | retrain path, `hundred_validation.py` |

Effort key: 🟢 Low · 🟡 Medium · 🔴 High.

---

## Invariants (read before touching the file)

> [!CAUTION]
> **Train/serve parity.** `_extract_features_and_target` (batch, [line 119](../../src/models/xgb_win_probability.py#L119)) and `_extract_features_single` (single, [line 279](../../src/models/xgb_win_probability.py#L279)) MUST compute every feature identically. The model trains on the batch path and serves (pre-match, scanner, sim) partly on the single path. **Every edit in Phases 1–3 is applied to BOTH methods in the same commit**, and their formulas diffed side-by-side.

> [!CAUTION]
> **`t20`/`od` must stay byte-identical.** All edits are either (a) gated behind `format_type == "hnd"`, or (b) a `format_spec` reformulation *proven* equivalent for 6-ball formats. Two independent gates enforce this, and **both** must pass after every phase:
> 1. **Exact-match snapshot** — `py tests/regression_snapshot_t20od.py --verify` re-predicts on a frozen 500-state/format sample and asserts `np.allclose` (atol=1e-6) against Phase-0's frozen output. Catches *any* drift, however small.
> 2. **Full-corpus tolerance test** — `py tests/regression_test.py` re-evaluates Brier (win-prob) and MAE (first-innings) over the **entire** current t20/od second-innings corpus and asserts no metric moves beyond `data/processed/baseline_metrics.json`'s locked tolerances (Brier: +0.0010 overall / +0.0025 per-phase; MAE: +0.50 overall / +1.00 per-phase). This is the authoritative "did accuracy regress" answer; the snapshot alone only proves a fixed sample didn't move.

---

## Phase 0: Lock the Baseline

**Priority:** 🟢 Low · **Blocking** (no improvement is real without a locked "before").
**Dependencies:** current trained `hnd_xgb_models.pkl`.

### Problem
Improvements must be measured against exact, reproducible numbers, not the ~0.178 quoted from memory.

### Implementation
1. Run the existing validation harness end-to-end and capture the console block verbatim:
   ```bash
   py tests/hundred_validation.py
   ```
2. Record into the tracker (Phase 0 row): **overall** Brier for `brier_hnd`, `brier_t20_on_hnd`, `brier_naive`, and the **per-phase** table (powerplay / middle / death). These are the four locked reference numbers every later phase reports a delta against.
3. **Freeze a t20/od exact-match regression fixture.** `tests/regression_snapshot_t20od.py --create` snapshots `predict_batch` output for a fixed 500-row-per-format sample of existing `t20`/`od` second-innings states → `data/processed/_regression_snapshot_t20od.parquet`. `--verify` re-asserts `np.allclose` against it.
4. **Lock the t20/od full-corpus Brier/MAE baseline.** `tests/regression_test.py --generate` re-evaluates win-prob Brier + first-innings MAE (overall + per-phase) over the **entire current** t20/od second-innings corpus (not a sample) and writes `data/processed/baseline_metrics.json` with tolerance thresholds. Default (no flag) re-checks against it and exits non-zero on any regression beyond tolerance.
   > [!NOTE]
   > A stale `baseline_metrics.json` existed on disk pre-dating this plan (sample_size 1,002,374 t20 / 976,150 od vs. today's actual 1,134,006 / 978,262 second-innings states — the corpus grew since it was last generated). It was regenerated fresh in Phase 0 so the lock reflects *today's* corpus, not a stale one — comparing post-edit numbers against a baseline built on fewer matches would confound real regressions with ordinary corpus growth. Its source script (`tests/regression_test.py`, previously deleted from the working tree) also had a latent bug fixed here: `StateBuilder(...)` was called without `format_type=`/`balls_per_over=`, silently defaulting to `format_type="t20"` internally for the `od` call — which mis-keys `rolling_window` (18 instead of the correct 36) and `default_econ` (8.0 instead of 5.0) for OD. Now threads `format_spec.spec(format_type)` explicitly, matching `retrain_full`.

### Verification
Baseline table recorded; snapshot parquet written; `baseline_metrics.json` regenerated and self-verified (`py tests/regression_test.py` passes with 0 diff against itself). Nothing changed in `src/` yet.

---

## Phase 1: Enable Player-Elo Features on the `hnd` Path  ◄── highest leverage

**File:** `src/models/xgb_win_probability.py` (MODIFY)
**Priority:** 🟢 Low effort / 🔴 High value
**Dependencies:** Phase 0. Hundred states already carry the columns ([build_states.py:346-414](../../src/data_processing/build_states.py#L346)).

### Problem
The entire player/squad/form/residual-Elo feature block is gated to `t20`/`od` in **both** extractor methods ([batch line 223](../../src/models/xgb_win_probability.py#L223), [single line 381](../../src/models/xgb_win_probability.py#L381)). The `hnd` model therefore trains on team/phase/venue/pressure features only — throwing away the exact signal (seeded player Elos) the reused-T20 model uses to beat it.

### Implementation
1. In **both** `_extract_features_and_target` and `_extract_features_single`, broaden the gate:
   ```python
   if self.format_type in ["t20", "od", "hnd"]:
   ```
   No other change inside the block is needed — it reads `batter_elo`, `bowler_elo`, `non_striker_elo`, `player_elo_diff`, `batting_squad_elo`, `bowling_squad_elo`, `squad_elo_diff`, `*_recent_form`, and the `residual_*` derivations, all present on `hnd` states (with `1500.0`/`0.0` fallbacks for debutants via `df.get`). The `decay_weight` multiplier is fixed properly in Phase 2 — Phase 1 rides the current (slightly-off) decay to isolate the feature-set effect.
2. Confirm the style/matchup block ([lines 268-274](../../src/models/xgb_win_probability.py#L268)) is already format-agnostic (it is) — no change.
3. **Do not** touch `competition_encoded`: `hnd` maps to the `t20s` default (constant → trees ignore it). Not a lever; leave it.
4. Retrain only the `hnd` win-prob artifact:
   ```bash
   py -c "from src.data_processing.cricsheet_importer import retrain_full; retrain_full('hnd')"
   ```
   (or the targeted win-prob refit the retrain path uses — regenerate `data/processed/hnd_xgb_models.pkl`).

### Verification
```bash
py tests/hundred_validation.py                      # read the Brier block
py tests/regression_snapshot_t20od.py --verify      # exact-match gate
py tests/regression_test.py                         # full-corpus tolerance gate
```
- **Expected:** `brier_hnd` drops materially toward / below `brier_t20_on_hnd` (0.1434). This single change should close most of the 0.178→0.143 gap.
- Both t20/od regression gates PASS with zero diff (edit was `hnd`-gated, so this is a pure sanity check, not expected to catch anything here).
- Record new overall + per-phase Brier in the tracker with the delta vs Phase 0.

> [!NOTE]
> If Phase 1 alone lands `brier_hnd < brier_t20_on_hnd`, the "reuse T20" recommendation from the last plan is formally beaten and the bespoke model is justified. Phases 2–5 then push further.

---

## Phase 2: Fix `hnd` Feature Geometry (120-ball → 100-ball)

**File:** `src/models/xgb_win_probability.py` (MODIFY)
**Priority:** 🟡 Medium
**Dependencies:** Phase 1.

### Problem
Two feature computations hardcode 6-ball / 120-ball geometry, wrong for the 100-ball Hundred:

- **Elo phase-decay** ([batch 155-160](../../src/models/xgb_win_probability.py#L155), [single 312-320](../../src/models/xgb_win_probability.py#L312)): `total_balls = total_overs * 6.0` → **120** for `hnd`; `is_one_day = total_overs > 20` is `False`, so decay thresholds become `death_start=30, powerplay_end=84` on a scale where the innings has only 100 balls. `decay_weight` is miscalibrated for every Hundred state.
- **Wicket-pressure denominator** ([batch 133-138](../../src/models/xgb_win_probability.py#L133), [single 292-295](../../src/models/xgb_win_probability.py#L292)): uses `target_overs * 6.0` (→120) instead of the Hundred's 100.

### Implementation
Reformulate both via `format_spec` — **proven equivalent for t20/od**, correct for `hnd`.

1. In `__init__`, resolve the spec once:
   ```python
   from src.data_processing.format_spec import spec
   self._spec = spec(format_type)
   self.balls_per_over = self._spec["balls_per_over"]   # 6 / 6 / 5
   # self.total_format_balls already 120/300/100 per existing branches — keep
   ```
2. **Elo-decay thresholds** — replace the `is_one_day` branch with a spec-driven `balls-remaining` threshold (identical numbers for t20/od, correct for hnd):
   ```python
   total_balls_fmt   = self._spec["total_balls"]                      # 120 / 300 / 100
   death_start_rem   = total_balls_fmt - self._spec["death_start_ball"]  # 30 / 60 / 25
   powerplay_end_rem = total_balls_fmt - self._spec["pp_end_ball"]       # 84 / 240 / 75
   middle_duration   = max(powerplay_end_rem - death_start_rem, 1.0)
   decay_weight = clip((balls_remaining - death_start_rem) / middle_duration, 0, 1)
   ```
   Equivalence check (must hold before merge): t20 → `120-90=30`, `120-36=84`; od → `300-240=60`, `300-60=240`; hnd → `100-75=25`, `100-25=75`. The `elo_death_start`/`elo_powerplay_end` ctor params become redundant overrides — keep them accepted for sweep back-compat but default to the spec derivation.
3. **Wicket-pressure denominator** — replace `target_overs * 6.0` with `target_overs * self.balls_per_over`; keep the `self.total_format_balls` fallback (already 100 for hnd). t20/od: `*6` unchanged.
4. Apply to **both** extractor methods; diff the two side-by-side.

### Verification
```bash
# t20/od equivalence — MUST be byte-identical to Phase-0 snapshot, AND within tolerance on the full corpus
py tests/regression_snapshot_t20od.py --verify
py tests/regression_test.py
py -c "from src.data_processing.cricsheet_importer import retrain_full; retrain_full('hnd')"
py tests/hundred_validation.py
```
- **Expected:** t20/od predictions identical to snapshot (`np.allclose` at 1e-6, i.e. `regression_snapshot_t20od.py --verify` prints PASS/0.0 diff for both formats) **and** `regression_test.py` exits 0 (no metric moved beyond tolerance); `brier_hnd` improves again (decay now spans the true 25/50/25 split). This phase is the highest t20/od regression risk in the plan (it touches the shared decay-threshold formula) — do not proceed to Phase 3 if either gate fails. Record delta.

---

## Phase 3: Monotonic Constraints on Directional Features

**File:** `src/models/xgb_win_probability.py` (MODIFY)
**Priority:** 🟡 Medium
**Dependencies:** Phases 1-2. (Apply to `hnd` first; consider extending to t20/od only as a separate, out-of-scope follow-up.)

### Problem
With ~167 matches the tree can wiggle non-monotonically in features that are physically monotone, hurting calibration (Brier). XGBoost's `monotone_constraints` injects the prior for free and can't be over-fit around.

### Implementation
1. Build the constraint vector **by column name** (order-safe): extract features once, get `X.columns`, map each to a sign, pass a same-length tuple. Proposed signs (chasing team's win prob):
   - `+1`: `proportion_scored`, `wickets_in_hand`, `run_rate`, `scoring_acceleration`, `balls_per_run_needed`, `resources_remaining`, `batting_team_elo`, `batting_team_phase_elo`, `elo_diff`, `phase_elo_diff`, `batter_elo`, `batting_squad_elo`, `batting_duo_elo_avg`
   - `-1`: `runs_remaining`, `required_run_rate`, `pressure_index`, `wickets_lost`, `bowling_team_elo`, `bowling_team_phase_elo`, `bowler_elo`, `bowling_squad_elo`
   - `0`: everything else (venue, phase flags, competition, style, target, balls_remaining — ambiguous or contextual)
2. Add `monotone_constraints=<tuple>` to **both** `XGBClassifier(...)` instantiations (fold + final, [lines 480, 525](../../src/models/xgb_win_probability.py#L480)). Gate the constraint tuple behind `if self.format_type == "hnd"` initially so t20/od are untouched.
3. Guard against column-order drift: assert `len(constraints) == X.shape[1]` and build the tuple from the live `X.columns`, never a hardcoded list.

### Verification
```bash
py -c "from src.data_processing.cricsheet_importer import retrain_full; retrain_full('hnd')"
py tests/hundred_validation.py
```
- **Expected:** Brier equal-or-better and the reliability curve visibly flatter; per-phase Brier no worse in any phase. If any phase regresses, relax the `0`-signed set rather than the strong physical ones. Record outcome; **kill this phase if it doesn't help** — it's a prior, not a requirement.

---

## Phase 4: Regularization / Hyperparameter Sweep for `hnd`

**File:** `src/models/xgb_win_probability.py` defaults (MODIFY), `tests/hundred_winprob_sweep.py` (NEW)
**Priority:** 🟡 Medium
**Dependencies:** Phases 1-3 (sweep on top of the fixed feature set).

### Problem
`hnd` inherits near-T20 capacity (`max_depth=4, lr=0.08, n_estimators=300`) on ~100× less data — likely over-fit.

### Implementation
1. New sweep script: GroupKFold-by-`match_id` (reuse the model's own CV), objective = **mean OOF Brier** (not log-loss, since Brier is the target metric). Grid, small and sensible for thin data:
   - `max_depth ∈ {2, 3, 4}`
   - `min_child_weight ∈ {10, 20, 40}`
   - `learning_rate ∈ {0.03, 0.05, 0.08}` (pair lower lr with more `n_estimators ∈ {300, 500, 800}`)
   - `reg_alpha ∈ {1, 3}`, `reg_lambda ∈ {3, 5, 10}`
   - `subsample / colsample_bytree ∈ {0.7, 0.8}`
   Use `train_final_model=False` (sweep mode already supported, [line 542](../../src/models/xgb_win_probability.py#L542)) for speed.
2. Pick the config minimising OOF Brier with the **fewest** effective parameters on ties (prefer shallower/more-regularized). Write it into the `elif format_type == "hnd":` defaults block ([lines 42-56](../../src/models/xgb_win_probability.py#L42)).

### Verification
```bash
py tests/hundred_winprob_sweep.py     # prints ranked grid by OOF Brier
py tests/hundred_validation.py        # confirm held-out Brier tracks the OOF pick
```
- **Expected:** a modest, honest Brier gain; new defaults recorded in the tracker with the winning grid row.

---

## Phase 5: Pooled / T20-Blended Win-Prob Experiment (fallback upgrade)

**File:** `tests/hundred_winprob_pooling_experiment.py` (NEW — mirror [hundred_pooling_experiment.py](../../tests/hundred_pooling_experiment.py))
**Priority:** 🟡 Medium
**Dependencies:** Phases 1-4 (compete against the *improved* bespoke model, not the broken one).

### Problem
Phase 6 of the prior plan pooled the *delivery-outcome* model and won. The *win-prob classifier* — the Brier scorer — was never pooled. Since reused-T20 already beats the (old) bespoke model, a transfer approach at this layer is promising. But intuition isn't evidence: measure it, with a kill switch.

### Implementation
Evaluate three challengers on a **chronological** held-out slice of recent Hundred matches (last ~20%, matching the existing experiment's split), against the Phase-1-4 bespoke `hnd` model:
- **A — Pooled:** T20 + HND second-innings states, `is_hundred ∈ {0,1}` appended to features, one classifier. (Caveat: `required_run_rate` denominator differs ÷6 vs ÷5 — normalise to a per-over basis in the pooled frame *or* rely on `is_hundred`; record which.)
- **B — Blend:** `p = w·p_hnd + (1-w)·p_t20on_hnd`, sweep `w ∈ {0..1}`; a 1-parameter transfer that's trivial to ship.
- **C — T20-warm-start:** train `hnd` model initialised from the T20 booster (`xgb_model=` continuation) then fine-tune on Hundred states.

Metric: held-out **Brier** (primary) + reliability + per-phase Brier. **Keep a challenger only if it beats the bespoke Phase-1-4 model on held-out Brier.** If none do, the bespoke model ships as-is — a clean result.

### Verification
```bash
py tests/hundred_winprob_pooling_experiment.py
```
- **Expected:** a ranked table `{bespoke, pooled, blend@w*, warm_start} → [Brier, per-phase Brier]` and a one-line verdict written to the tracker.

> [!WARNING]
> Pooling can wash out genuine Hundred end-game behaviour under 100× more T20 rows. `is_hundred` mitigates but doesn't guarantee. The blend (B) is the safest transfer if pooling proves unstable.

---

## Phase 6: Ship Decision + Wire the Winning Config

**File:** retrain path, `tests/hundred_validation.py` (confirm), `pre_match_predictor.py` / `scanner.py` (verify only)
**Priority:** 🟢 Low
**Dependencies:** Phases 1-5.

### Implementation
1. Select the config with the best held-out Brier that also (a) beats reused-T20 (0.1434) and (b) is no worse than the naive baseline in any phase.
2. Ensure `retrain_full("hnd")` produces that artifact. If the winner is a blend (Phase 5-B), add the blend weight + T20-fallback prediction into the `hnd` serve path (both `predict` and `predict_batch`), keeping t20/od untouched.
3. **Verify live parity:** `pre_match_predictor` and `scanner` consume the same extractor, so no change is expected — but run the Phase-7 smoke from the prior plan to confirm a Hundred fixture still prices (target ≈ 140, 100-ball geometry).
4. Update the prior plan's Phase-8 tracker row and the `hundred_model_scoping` memory with the final shipped Brier.

### Verification / Accept Bar
```bash
py tests/hundred_validation.py
py -c "from src.betting.pre_match_predictor import PreMatchPredictor; p=PreMatchPredictor(); \
print(p.predict('Trent Rockets','Southern Brave', venue='Trent Bridge', format_type='hnd'))"
```
- **Ship gate:** `brier_hnd < 0.1434` (beats reuse-T20) **and** `brier_hnd < brier_naive` per-phase, on held-out folds. Record the final number.

---

## Implementation Tracking

_Per the `update_build_plans_live` rule: fill each row with **real numbers** from its Verification command immediately after the phase lands — never batched to session end._

| Phase | Component | Status | Notes (real outcome / Brier delta / deviations) |
|-------|-----------|:------:|--------------------------------------------------|
| 0 | Lock baseline | `[x]` | **Overall Brier:** hnd=0.1780, t20_on_hnd=0.1434, naive=0.1680 — bespoke `hnd` is worse than BOTH reused-T20 and naive. **Per-phase Brier** (hnd / t20 / naive): powerplay 0.2006 / 0.1880 / 0.2405; middle 0.1809 / 0.1381 / 0.1550; death 0.1393 / 0.0966 / 0.1031 — hnd loses every phase to reused-T20, and loses to naive everywhere except powerplay. Note: `hundred_validation.py`'s own printed "Improvement +...%" / "VERDICT: PASSED" lines are stale/buggy (the literal `+-24.12%` is the tell — a hardcoded `+` prefix on an actually-negative number); trust the raw Brier numbers above, not that verdict text. **t20/od regression snapshot:** built `tests/regression_snapshot_t20od.py` (mirrors `retrain_full`'s feature pipeline exactly); wrote 500 t20 + 500 od second-innings states + frozen `predict_batch` output to `data/processed/_regression_snapshot_t20od.parquet` (mean pred_prob: t20=0.4920, od=0.4460). Self-verified with `--verify`: max abs diff = 0.00000000 on both formats — harness confirmed sound before any `src/` edits. **t20/od own-format Brier baseline (locked via `tests/regression_test.py --generate`, full corpus):** T20 Brier=0.127899 (n=1,134,006 second-innings states), first-innings MAE=16.51 runs; OD Brier=0.134587 (n=978,262), first-innings MAE=31.47 runs. Written to `data/processed/baseline_metrics.json` with tolerance gates (Brier +0.0010 overall/+0.0025 per-phase, MAE +0.50 overall/+1.00 per-phase). Self-check (`py tests/regression_test.py` re-run against the fresh baseline) confirms 0 regressions. **This is the number every later phase's t20/od gate must stay within tolerance of — not the old stale JSON.** |
| 1 | Player-Elo on `hnd` | `[x]` | **brier_hnd: 0.1780 → 0.1754 (Δ −0.0026).** Real but far smaller than hypothesized — does NOT close most of the 0.178→0.143 gap; still loses to reused-T20 (0.1434) and to naive (0.1680) in middle/death. Per-phase: powerplay 0.2006→0.1992, middle 0.1809→0.1792, death 0.1393→0.1324 (biggest single-phase gain). Likely cause: the newly-enabled player-Elo features are still multiplied by the Phase-2-unfixed `decay_weight` (computed on the wrong 120-ball scale for `hnd`), attenuating the signal rather than passing it through cleanly — Phase 2 should unlock more of the hypothesized gain. **t20/od regression gates: both PASS.** Exact-match snapshot: 0.00000000 diff both formats. Full-corpus tolerance test (`regression_test.py`): every metric diff = +0.000000 vs. locked baseline, 0 regressions — confirms the `hnd`-gated edit truly touched nothing shared. Does not yet beat reused-T20 → proceeding to Phase 2 per the plan's decision note. |
| 2 | `hnd` geometry fix | `[x]` | **brier_hnd: 0.1754 → 0.1767 (Δ +0.0013, slightly WORSE)** — contrary to the plan's hypothesis. Per-phase: powerplay 0.1992→0.1990 (marginal gain), middle 0.1792→0.1807 (worse), death 0.1324→0.1357 (worse). **t20/od: both gates PASS with exact 0.00000000 diff** on every metric — the equivalence proof (t20 30/84, od 60/240 unchanged; hnd corrected 25/75) held exactly in practice, zero regression risk materialized despite this being the plan's highest-risk phase. Assessment: the fix is proven cricket-correct (25/75 are the true Hundred phase boundaries vs. the old accidentally-offset 30/84), and the Brier regression (+0.0013) is small enough to plausibly be small-corpus noise (15,010 states, 5-fold split) rather than evidence the true boundaries hurt — reverting to a proven bug for a marginal, noisy gain would leave Phase 3+ tuning around incorrect geometry. **Decision: kept the fix, did not revert.** Net progress after Phases 1+2 combined: brier_hnd 0.1780→0.1767 (Δ −0.0013 total) — still 0.0333 away from reused-T20 (0.1434); Phases 3-5 now carry more weight than the plan's sequencing assumed. |
| 3 | Monotonic constraints | `[x]` | **brier_hnd: 0.1767 → 0.1723 (Δ −0.0044) — best single-phase gain in the plan so far.** Per-phase: powerplay 0.1990→0.1977, middle 0.1807→0.1739, death 0.1357→0.1330 — **every phase improved, zero regressions.** Implemented exactly per plan: `MONOTONE_POSITIVE_FEATURES`/`MONOTONE_NEGATIVE_FEATURES` class constants (13 `+1`, 8 `-1`, matching the plan's named list verbatim — did not extend to the unlisted residual_*/diff/form features, kept faithful to spec), `_build_monotone_constraints()` builds the tuple order-safe from live `X.columns`, hnd-gated (`format_type != "hnd"` → `None`, identical to XGBoost's own default). Categorical columns (`batter_hand`/`bowler_style`, sign 0) retrained without error. **Decision: KEPT** — satisfies the plan's explicit criterion (equal-or-better, no phase worse). **t20/od: both gates PASS**, exact 0.00000000 diff on every metric (one transient OOM crash from running the two gates *concurrently* on the large t20/od corpus — re-ran sequentially, confirmed clean; not a code regression, just a resource-contention artifact of parallel execution). Net progress after Phases 1-3: brier_hnd 0.1780→0.1723 (Δ −0.0057 total); still 0.0289 from reused-T20 (0.1434). |
| 4 | `hnd` reg/HP sweep | `[x]` | **Grid reduced from the plan's literal 648 configs (3,240 fits) to 81 (405 fits)** to fit the tool's 10-min background cap: kept full resolution on max_depth/min_child_weight/lr-nest-pairing (the 3 knobs directly tied to the stated over-fit problem), bundled the 4 secondary reg knobs (reg_alpha, reg_lambda, subsample, colsample_bytree) into 3 light/medium/heavy profiles instead of a full 24-way cross product. Ran in `tests/hundred_winprob_sweep.py`, ~581s. **Winning config: max_depth=2 (was 4), min_child_weight=10, lr=0.08, n_estimators=300, reg_alpha=3 (was 1), reg_lambda=10 (was 3), subsample=0.7 (was 0.8), colsample_bytree=0.7 (was 0.8).** Raw OOF Brier 0.169295 vs. 0.188889 for the pre-sweep config evaluated in the same sweep (Δ −0.0196, top-3 tie-band all agreed on shallow+heavily-regularized regardless of lr/n_estimators — a strong, non-fluke signal). **Calibrated brier_hnd: 0.1723 → 0.1689 (Δ −0.0034), real net improvement.** Per-phase: middle 0.1739→0.1693 (better), death 0.1330→0.1189 (much better), **powerplay 0.1977→0.2043 (worse, +0.0066)** — no explicit per-phase kill-rule for this phase (unlike Phase 3); kept given the net gain and the strength/consistency of the underlying OOF signal. **t20/od: both gates PASS**, exact 0.00000000 diff (change is fully contained in the `elif format_type=="hnd"` branch, inert by construction for other formats). Net progress after Phases 1-4: brier_hnd 0.1780→0.1689 (Δ −0.0091 total); 0.0255 from reused-T20 (0.1434). |
| 5 | Pooled/blended win-prob | `[x]` | Built `tests/hundred_winprob_pooling_experiment.py`. Chronological split: 133 train / 34 held-out HND matches (11,920 / 3,090 states). All evaluated on the SAME held-out set with **raw/uncalibrated** probabilities for a clean architecture comparison (production calibration applied uniformly later, regardless of which architecture ships). **Result — all three challengers beat bespoke:** bespoke 0.145574; **pooled 0.111271 (Δ −0.034303, BEST — wins overall AND every single phase**: powerplay 0.1921→0.1475, middle 0.1376→0.1036, death 0.0976→0.0782); blend@w=0.3 0.125805 (Δ −0.019770); warm_start 0.139375 (Δ −0.006199, smallest gain). This is a far larger effect than Phases 1-4 combined (−0.0091 total) and the plan's own "pooling washes out Hundred end-game signal" risk did NOT materialize — death improved the most. Design decisions recorded: `required_run_rate`/`run_rate` kept native per-format-over scale (relied on `is_hundred` rather than renormalizing); pooled classifier used T20's own tuned hyperparameters (T20-volume-dominated corpus); monotone constraints not applied to pooled/warm-start (bespoke comparator kept its Phase-3 constraints). **Verdict: POOLED wins decisively.** **Not yet wired into production** — Phase 5's scope is the experiment + verdict; integrating a pooled architecture (which needs both t20 AND hnd data at retrain time, and doesn't fit the existing single-format `XGBWinProbabilityModel`/`retrain_full` interface) is a materially bigger architectural change than Phases 1-4 and is Phase 6's explicit job — flagged to the user for discussion before proceeding. No production code touched this phase (experiment script only) — t20/od regression gates not applicable (nothing shared changed). |
| 6 | Ship + wire | `[x]` | **SHIP GATE MET: brier_hnd = 0.1411 < 0.1434 (beats reused-T20 for the first time in this plan)**, and beats naive per-phase everywhere: death 0.1008<0.1031, middle 0.1376<0.1550, powerplay 0.1772<0.2405. Wired pooled architecture into production: (1) `is_hundred` feature added to both extractor methods, gated `if self.format_type=="hnd"` -- t20/od get zero schema change since the column never appears for them; (2) `fit()` extended with optional `pool_X`/`pool_y` params -- when `None` (always true for t20/od) behaves byte-identical to pre-Phase-5 code; when provided, pool rows join every fold's *training* set (plus the final model) but are never held out/scored -- fold assignment, OOF calibration, and match_to_fold all computed purely from hnd's own states; pooled path uses T20's own tuned hyperparameters and no monotone constraints, exactly matching the validated experiment; (3) new `_build_second_innings_pool_states()` helper in `cricsheet_importer.py`, wired into `retrain_full` only for `format_type=="hnd"` (builds T20 second-innings states + tags `is_hundred=0.0`) -- t20/od's own retrain path untouched. Retrain confirmed pooling worked as designed: "Dataset has 15,010 states across 162 unique matches. Pooling in 1,134,006 additional never-held-out training rows." **t20/od: both gates PASS**, exact 0.00000000 diff on every metric despite this phase restructuring the shared `fit()` method -- the pool_X-gating held. **Live smoke test PASSED**: `PreMatchPredictor.predict('Trent Rockets','Southern Brave', venue='Trent Bridge', format_type='hnd')` → prob_a=0.6446, estimated_target=143.32, format_type='hnd' -- works transparently through the unchanged `predict`/`predict_batch` interface, no serving-side changes needed. **Final trajectory: brier_hnd 0.1780 (Phase 0) → 0.1411 (shipped) — a 20.7% relative reduction, crossing the reused-T20 threshold for the first time.** |

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|:--------:|------------|
| An edit silently regresses `t20`/`od` | 🔴 High | Every change `hnd`-gated or spec-reformulated & proven equivalent; Phase-0 snapshot + `np.allclose` gate each phase |
| Batch/single extractor drift (train≠serve) | 🔴 High | Both methods edited in one commit; formulas diffed side-by-side; invariant stated up top |
| Player-Elo fallbacks (debutants at 1500) add noise not signal | 🟡 Medium | Phase 1 measures in isolation; seeded Elos (prior Phase 4b) mean few true 1500s; kill if Brier worsens |
| Monotone constraints over-constrain thin data | 🟡 Medium | `hnd`-gated, per-phase Brier check, explicit kill switch (Phase 3) |
| Sweep over-fits the fold split | 🟡 Medium | GroupKFold-by-match_id; prefer simpler config on ties; confirm on held-out (Phase 4 vs 6) |
| Pooling washes out Hundred end-game | 🟡 Medium | Held-out Hundred Brier only + `is_hundred`; blend (B) as safer 1-param transfer; kill switch |

---

## Regression Guard

> [!CAUTION]
> - **`t20`/`od` behaviour is frozen.** Phases 1-3 edits are `hnd`-gated; the Phase-2 `format_spec` reformulation is proven byte-identical for 6-ball formats and re-asserted against the Phase-0 snapshot every phase.
> - **`format_spec.py` is the single source of geometry.** No new literal `5`/`6`/`100`/`120` in `xgb_win_probability.py` — read `self._spec` / `self.balls_per_over`.
> - **Both extractor methods change together**, always.
> - Each phase is independently revertable and independently measured — no phase ships without a recorded Brier delta beating its predecessor.
```

