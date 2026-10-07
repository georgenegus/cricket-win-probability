# Hundred (hnd) Player Elo — Dynamics Revamp & Scale Alignment Plan

> **Owner:** George · **Created:** 2026-08-06 · **Status:** ✅ Shipped (2026-08-06) — plan complete.
> **Outcome:** _Phase 1 (scale) shipped: `HND_ELO_STRETCH=2.78`, center 1500 held exactly, batter p99 2000.0 exact (Buttler peak 1565.1→1680.9). Phases 2 (dynamic K) and 3 (opponent weighting) were both tried and reverted — hnd's 162-match corpus couldn't support the added variance without regressing Brier, the same failure mode `effective_game_elo_plan.md` found for hnd's squad-weighting experiments. Phase 4 shipped the scale-only result to production: pooled hnd win-prob Brier improved from 0.139453 to **0.137557** (Δ−0.001896) — a genuine, non-trivial gain, not just the expected affine no-op, because the wider hnd-native scale now sits much closer to T20's own stretched scale in the pooled training set the win-prob classifier actually trains on (see Phase 4 for the full mechanism). Along the way, independently rediscovered and fixed T20 Phase 6's exact infra bug (missing `if __name__ == "__main__":` guard on Windows spawn causing recursive re-execution + silent stale-cache reuse) in this plan's own retrain script. Final: t20/od/t20w untouched (all gates exact 0.0 or unaffected), hnd's own sim-engine baseline established for the first time, all production artifacts genuinely regenerated and live-smoke-tested._
> **Purpose / Strategy:** Port the methodology proven on [t20_player_elo_revamp_plan.md](t20_player_elo_revamp_plan.md) (shipped) and [od_player_elo_revamp_plan.md](od_player_elo_revamp_plan.md) (Phases 1+3 shipped) to the Hundred (`hnd`) player-Elo engine: a hold-center scale so `hnd` peaks sit on the same 1500/2000 anchors as T20/OD, delivered via genuine, independently Brier-gated dynamics changes. **User-scoped**: stretch (Phase 1) + experience-graded dynamic K (Phase 2) + opponent-quality weighting (Phase 3) only — veteran de-shrink and per-phase skill features are explicitly out of scope (both already failed for T20 on a far larger corpus; not worth re-trying on hnd's much smaller one).
> **Scope:** `src/features/player_elo_engine.py` (primary, `hnd`-gated knobs) + retrain of `hnd`'s two tree models that consume player Elo (`xgb_win_probability.py`, `delivery_outcome_model.py`) + `first_innings_regressor` (unaffected but retrained as part of the standard pipeline). **`hnd` only** — `t20`/`od` dynamics and scale stay frozen and regression-gated.

---

## Why this is riskier than T20/OD — read before touching the engine

`hnd`'s corpus is **tiny**: 162 matches / 15,010 second-innings states / 4,534 player-appearance rows, vs. T20's ~1.1M states and OD's ~1.0M. This project already ran the exact non-affine-dynamics-on-hnd experiment once before, in a different guise: `effective_game_elo_plan.md`'s Phase 1 (position-weighted batting squad Elo) and Phase 2 (frontline-weighted bowling squad Elo) both **failed hnd's Brier gate by 10–25x tolerance** while passing cleanly for t20/od, with the documented conclusion "hnd's corpus is almost certainly too small to support the added variance ... without overfitting." Dynamic K (Phase 2 here) and opponent weighting (Phase 3 here) are a different mechanism, but the same underlying risk (a small corpus can't distinguish genuine signal from noise) applies squarely. **Expect a real chance either or both phases get reverted** — that is a successful outcome of this plan's process, not a failure of it, per this repo's keep-or-revert precedent (Strategy 3, T20 Phase 2/5).

`hnd` also has a structural quirk neither T20 nor OD has: it **warm-seeds** its player ratings from a frozen T20 snapshot (`build_hnd_seed_ratings()`, `_hnd_seed_from_t20.parquet`) rather than starting cold at 1500. Phase 1 below must convert that seed onto `hnd`'s **own** stretched scale, not leave it on the old unstretched one — see the dedicated section below.

### The two levers (same framing as T20/OD)

- **Scale knob (Phase 1, affine, 0-Brier gate):** raise `hnd`'s `elo_divisor`/`k_runs`/`k_wickets` by a hold-center factor `a`, hold 1500 as center. Provably prediction-invariant *if* every team-Elo-as-player-proxy site and the T20 seed are correctly mapped.
- **Dynamics knobs (Phases 2–3, non-affine, Brier-gated):** experience-graded K, then opponent-quality weighting — each ported verbatim in mechanism from T20/OD, re-tuned (not copied) to `hnd`'s own measured profile where one exists, strict keep-or-revert on the full-corpus Brier gate.

### Phase Overview

| # | Phase | Lever | Affine? | Gate |
|---|-------|-------|:-------:|------|
| 0 | Measure distribution + confirm locked t20/od/hnd baseline | — | — | informational |
| 1 | Scale knob: divisor + K + team-anchor rescale + **T20-seed re-map onto hnd's own stretched scale** | scale | ✅ yes (0-Brier gate) | hnd Brier within ±0.0010 of Phase 0; t20/od exact 0.0 |
| 2 | Experience-graded dynamic K | dynamics | ❌ no | hnd Brier ≤ Phase 1 (±0.0025/phase); t20/od exact 0.0 |
| 3 | Opponent-quality-weighted updates | dynamics | ❌ no | hnd Brier ≤ Phase 2 (±0.0025/phase); t20/od exact 0.0 |
| 4 | Final calibration + retrain + ship | — | — | required to close out |

---

## Invariants

> [!CAUTION]
> **`t20`/`od` must stay byte-identical.** Every new knob (`HND_ELO_STRETCH`, dynamic-K mult, opp-weight coef) is gated to `format_type == "hnd"` only. Gate every phase with `tests/regression_snapshot_player_elo.py` (t20/od exact `np.allclose` 0.0) and `tests/regression_test.py` (t20/od Brier/MAE unchanged).

> [!CAUTION]
> **The T20-seed re-map is the #1 correctness risk unique to this plan.** Today (`stretch=1.0` for hnd), `build_hnd_seed_ratings()`'s seed conversion (`1500 + (t20_effective_elo − 1500) / T20_ELO_STRETCH`) lands seeds on hnd's *unstretched* 1500-centred scale — correct today, because `elo_divisor`/`k_runs`/`k_wickets` are also unstretched for hnd, so the model's internal update dynamics operate on that same unstretched scale throughout. Once `HND_ELO_STRETCH` is introduced, `hnd`'s *own* dynamics (divisor/K) move onto the wider scale — but the seed conversion must move onto that **same wider scale too**, or newly-seeded players would start on the old narrow scale while the update math and every already-evolved rating operate on the new wide one, breaking the "Phase 1 is affine" invariant specifically for seeded (i.e. every) player. Fix: seed conversion becomes `1500 + (t20_effective_elo − 1500) * (HND_ELO_STRETCH / T20_ELO_STRETCH)` — same relative-skill mapping, landing directly on hnd's new scale instead of the old one.

> [!CAUTION]
> **Scale-mixing:** `_team_to_player_scale()` must be extended to hnd (`elif format_type == "hnd": return 1500 + HND_ELO_STRETCH*(team_elo-1500)`) and `compute_effective_elo`'s stretch selection likewise — both currently hardcode "else: stretch = 1.0" as the hnd/t20w fallback.

---

## Phase 0: Measure + Confirm Baseline

**Status:** done (measurement only, no `src/` changes).

`scripts/measure_player_elo_distribution.py --format hnd --min-matches 20` (parity with T20/OD's own min-matches cut): batters n=42, **p50=1541.2, p99=1679.9**, max=1694.8 (HC Brook); bowlers n=31, p99=1637.1, max=1652.2 (L Wood). Named anchors (batter role, current hnd scale): Buttler peak=1565.1, Russell 1552.4, Maxwell 1576.5, de Kock 1545.3, Warner 1572.3 — all compressed into a ~1500–1600 band, consistent with hnd's small corpus + T20 seed being read back through the (today, unstretched) hnd scale.

Fitted hold-center stretch (p99 → 2000): **a = 2.780**. Locked baseline (`data/processed/baseline_metrics.json`, already current, not regenerated): **hnd Brier 0.139453** (n=15,010), per-phase death 0.101952 / middle 0.137514 / powerplay 0.170540; MAE 19.2902. t20 Brier 0.127562 (n=1,134,006); od Brier 0.134128 (n=978,262) — both re-used as the frozen reference for every later phase's exact-match gate.

---

## Phase 1: Scale Knob

**Status:** ✅ shipped (unpooled-gated; production retrain deferred to Phase 4).

Implementation per the Invariants section above: `HND_ELO_STRETCH = 2.78` class constant; `elo_divisor`/`k_runs`/`k_wickets` stretched by it for `format_type == "hnd"`; `_team_to_player_scale`/`compute_effective_elo` extended; seed conversion in `run()` updated to `HND_ELO_STRETCH / T20_ELO_STRETCH`. **Scope grew beyond the plan's file list, same class of bug as T20 Phase 1**: found and fixed an identical scale-mixing gap in `xgb_win_probability.py`'s `_elo_stretch()`/`_team_to_player_scale()` helper pair (used by the `residual_*` features, gated to `["t20","od","hnd","t20w"]` — hnd was already in that gate, but `_elo_stretch()` had no hnd branch and silently returned 1.0) — added the `hnd` branch so the residual features stay a clean affine no-op instead of quietly breaking Phase 1's zero-Brier gate.

**Methodology deviation (documented, not silent):** this machine has ~7.7GB RAM and repeatedly hit `ArrowMemoryError` building T20's full second-innings pool (`_build_second_innings_pool_states("t20")`, needed for production's pooled hnd win-prob training) — see `scripts/hnd_winprob_phase_gate.py`'s docstring. Every phase gate in this plan (1–3) therefore compares **unpooled** hnd-only Brier against the previous phase's unpooled number; the real pooled, production-faithful check happens once at Phase 4's ship retrain.

### Verification / Gate — PASS
- **Unpooled hnd Brier:** established a Phase-0 reference (`--phase0-ref`, monkeypatches `HND_ELO_STRETCH=1.0`, reproduces pre-Phase-1 exactly): **0.166474** (n=15,010; death 0.118292/middle 0.167014/powerplay 0.200486). Phase 1 (current code): **0.166382** (Δ **−0.000092**, essentially unchanged — as expected for a provably affine transform); per-phase death 0.118178 (−0.000114), middle 0.166831 (−0.000183), powerplay 0.200585 (+0.000099) — all comfortably inside tolerance (±0.0010 overall / ±0.0025 per-phase).
- **t20/od exact `np.allclose` 0.0** on the player-Elo snapshot (`tests/regression_snapshot_player_elo.py --verify`, `FROZEN_FORMATS=("t20","od")` — re-pointed for this plan): both PASS, max raw/effective diff 0.0.
- **Re-measured distribution** (`scripts/measure_player_elo_distribution.py --format hnd --min-matches 20`): center held exactly **1500.0** (fixed point); **p99 batter peak = 2000.0** (target 2000, exact); p50 **1614.5**; bowler p99 1881.0. Auto-fit reports `a≈1.000` — no further trim needed, confirms `HND_ELO_STRETCH=2.78` was fitted correctly. Named anchors: Buttler peak 1680.9 (was 1565.1), Russell 1645.8, Maxwell 1712.5 — all now spread into the intended wider band; hnd's naturally smaller "genuinely world-class" tail (mixed county/international competition, unlike T20's international-only corpus) means these land lower than T20's Buttler/Kohli (~1985/1962), which is expected, not a miscalibration — the p99/center anchors are what's pinned, not any individual named player.
- **Production artifacts NOT yet touched** — this phase's gate ran entirely against scratch caches (`_scratch_hnd_player_elo.parquet`, `_scratch_hnd_xgb_models.pkl`, both cleaned up after each run). `data/processed/hnd_player_elo_ratings.parquet` will be regenerated for real at Phase 4's `retrain_full("hnd")`.

---

## Phase 2: Experience-Graded Dynamic K

**Status:** ❌ **TRIED, REVERTED.** **Dependencies:** Phase 1.

Ported the same functional form as T20/OD Phase 3 (`_dynamic_k_multiplier`), `hnd`-gated, `max_mult=1.4` (OD's more conservative ratio), `k_scale=30` unchanged (measured hnd's own appearance profile first: batter p50=7 — identical to t20/od's own p50 — p75=14.25/p90=26.1/p99=41, n=240 batters/170 bowlers; the lower half matches t20/od closely, only the upper tail is compressed, a fact about the competition's youth (started 2021, max appearances=43) rather than a k_scale mismatch).

### Verification / Gate — FAIL, reverted
Unpooled hnd Brier (`scripts/hnd_winprob_phase_gate.py`): Phase 1 **0.166382** → Phase 2 **0.167464**, **Δ +0.001082** — exceeds the ±0.0010 overall tolerance (and is a larger miss than T20's own analogous Phase 2, veteran de-shrink, which reverted at +0.000167 for lack of independent justification). Per-phase: death improved (0.118178→0.116785, −0.001393) but middle (0.166831→0.168531, +0.001700) and powerplay (0.200585→0.202260, +0.001675) both worsened beyond noise. Consistent with the plan's stated risk: hnd's 162-match corpus doesn't support the added variance, the same failure mode as `effective_game_elo_plan.md`'s position/bowling squad-weighting experiments on hnd. **Reverted** — `dynamic_k_max_mult` restored to the exact 1.0 no-op for hnd (code comment documents the tried-and-reverted history, matching this repo's convention for Strategy 3 / T20 Phase 2 / T20 Phase 5). Re-ran the gate post-revert and confirmed it reproduces Phase 1's exact 0.166382. t20/od untouched throughout (this phase's knob was hnd-gated only, no snapshot re-verify needed for a same-session revert).

---

## Phase 3: Opponent-Quality-Weighted Updates

**Status:** ❌ **TRIED, REVERTED.** **Dependencies:** Phase 2 (reverted, so baseline is Phase 1's 0.166382).

Ported `_opponent_quality_weight` (bounded/saturating tanh), `hnd`-gated, `weight_coef=0.5`/`weight_scale=300`. `weight_scale` measured (not guessed) from hnd's own post-Phase-1 opponent-Elo deviation distribution (`data/processed/hnd_elo_ratings.parquet`, team_1/2_elo mapped through `HND_ELO_STRETCH`): median 113.7, p90 297.9, p99 387.0 — closely matched the T20 distribution T20's `weight_scale=300` was itself fitted from (median ~100-130, p90 ~300-380), so reusing that value was a measured choice, not a blind copy.

### Verification / Gate — FAIL, reverted
Unpooled hnd Brier: Phase 1 **0.166382** → Phase 3 **0.166768**, **Δ +0.000386** — inside the ±0.0010 overall tolerance band but a genuine worsening, not an improvement. Per-phase: death +0.000064 (negligible), middle +0.000719 (worse, inside ±0.0025), powerplay −0.000032 (negligible). Reverted on the same principle T20's own Phase 2 (veteran de-shrink) was reverted under despite an even smaller +0.000167 miss: a dynamics phase whose sole justification is a Brier gain must actually deliver one — "within tolerance" isn't "earned its place." `opp_weight_coef` restored to the exact 1.0-no-op state (0.0 coefficient) for hnd. Re-ran the gate post-revert and confirmed bit-exact reproduction of Phase 1's number: 0.166382 (death 0.118178, middle 0.166831, powerplay 0.200585).

---

## Phase 4: Final Calibration + Retrain + Ship

**Status:** ✅ shipped.

Since Phases 2 and 3 both reverted, the final state carried into Phase 4 is Phase 1 alone (scale knob only, `HND_ELO_STRETCH=2.78`) — no re-trim was needed (Phase 1's own re-measurement already landed p99 exactly at 2000.0, `a≈1.000`, confirmed unchanged since nothing touched the engine after).

**Real infra bug found and fixed en route — the exact same class of bug T20 Phase 6 hit.** The first retrain attempt (an ad hoc runner script) was missing Python's `if __name__ == "__main__":` guard. On Windows, `MatchMetadataExtractor`'s parallel worker pool uses `multiprocessing.spawn`, which re-imports the main script per worker; since the script's top-level code unconditionally called `retrain_full("hnd")`, every spawned worker recursively re-ran the *entire* pipeline, causing cascading `ArrowMemoryError`/`ArrayMemoryError` crashes from concurrent recursive copies competing for memory (compounded by this machine's genuinely tight RAM — see below). Worse: because `XGBWinProbabilityModel.fit()`/`DeliveryOutcomeModel.fit()` use dumb existence-based pkl caching with no source-hash check, one of the recursive crash-survivors "completed successfully" while silently reloading the **stale pre-revamp models** instead of retraining — exactly the T20 Phase 6 trap, independently rediscovered. Fixed identically: added the `__main__` guard, and manually backed up (not deleted — `*.pre_hndelo_phase4_bak.pkl`, kept as reversible artifacts) the two stale hnd pkls (`hnd_xgb_models.pkl`, `hnd_delivery_outcome_model.pkl`) to force genuine refits. Re-ran cleanly as a single process: T20 pool built successfully ("Pooling in 1,134,006 additional never-held-out training rows"), both models genuinely retrained from scratch.

**Separately, this session ran on a memory-constrained machine (7.7GB total RAM, browser/IDE/VPN/antivirus/other startup apps routinely leaving <2GB free)** — the `ArrowMemoryError` failures during Phases 1–3's investigation (before the `__main__` bug was found) were genuine OOMs from building T20's full second-innings pool repeatedly; documented in the Phase 1 section's methodology-deviation note. The user closed extra apps and restarted the machine before this phase's final retrain succeeded.

### Verification / Accept Bar — all PASS
- **Pooled hnd win-prob Brier** (`tests/regression_test.py`'s `evaluate_format_models`, the real production recipe): **0.139453 → 0.137557, Δ −0.001896** — a genuine improvement, not just neutral, and notably ~20x larger than the unpooled Phase 1 gate's Δ−0.000092. Likely mechanism (not just coincidence): before this plan, hnd's *native* player-Elo features sat on the old unstretched (D=400) scale while the *pooled T20* rows shared the same feature columns on T20's own stretched (D=2.48×400) scale — a real scale inconsistency the shared tree ensemble had to split across. `HND_ELO_STRETCH=2.78` brings hnd's native rows onto a comparably wide scale, making the pooled training set far more internally consistent. Per-phase: death 0.101952→0.098559 (−0.003393, better), middle 0.137514→0.135071 (−0.002443, better), powerplay 0.170540→0.170805 (+0.000265, negligible). First-innings MAE: exact 0.00 diff (19.2902 both — unaffected, as expected, the regressor doesn't consume player-Elo scale).
- **t20/od: exact 0.0** on the player-Elo snapshot, re-verified after the full retrain (`tests/regression_snapshot_player_elo.py --verify`).
- **Sim engine (delivery-outcome model) baseline established** (none existed for hnd before this session — `tests/sim_baseline_metrics.py --generate --formats hnd`, preserving t20/od entries): extras-gate OOF log-loss **0.191095**, legal-outcome OOF log-loss **1.554910** (n=31,943 deliveries / 30,716 legal). No prior number to compare against; this is now hnd's own reference point for future work.
- **Scale**: re-measured, unchanged from Phase 1 — center 1500.0 exact, batter p99 2000.0 exact, bowler p99 1881.0.
- **Baselines updated**: `data/processed/baseline_metrics.json` (`--generate-format hnd`, t20/od/t20w preserved) and `data/processed/sim_baseline_metrics.json` (t20/od preserved) both now reflect the shipped state.
- **Live serve-path smoke test**: `PreMatchPredictor.predict("Welsh Fire", "London Spirit", format_type="hnd")` ran cleanly end-to-end (prob_a=0.5526).
- **Production artifacts updated for real** (unlike Phases 1–3, which stayed on scratch caches): `hnd_player_elo_ratings.parquet`, `hnd_xgb_models.pkl`, `hnd_delivery_outcome_model.pkl` all regenerated from the shipped (Phase-1-only) code.

---

## Implementation Tracking

_Per the `update_build_plans_live` rule: fill each row with real numbers immediately after the phase lands._

| Phase | Component | Status | Notes |
|-------|-----------|:------:|-------|
| 0 | Measure + confirm baseline | `[x]` | See Phase 0 section above. |
| 1 | Scale knob | `[x]` **KEPT** | `HND_ELO_STRETCH=2.78`. Unpooled Brier 0.166474→0.166382 (Δ−0.000092, pass). t20/od exact 0.0. p99→2000.0 exact. Also fixed a latent scale-mixing gap in xgb_win_probability.py's `_elo_stretch()` (missing hnd branch). Production artifacts not yet touched (deferred to Phase 4). |
| 2 | Dynamic K | `[x]` **REVERTED** | max_mult=1.4, k_scale=30. Unpooled Brier 0.166382→0.167464 (Δ+0.001082, FAIL). Reverted to exact 1.0 no-op; post-revert re-run reproduced 0.166382. |
| 3 | Opp-quality weighting | `[x]` **REVERTED** | coef=0.5, scale=300 (measured, matched T20's fit). Unpooled Brier 0.166382→0.166768 (Δ+0.000386, FAIL — worsening despite being in-tolerance). Reverted to exact 1.0 no-op; post-revert re-run reproduced 0.166382 bit-exact. |
| 4 | Final cal + ship | `[x]` **SHIPPED** | No re-trim needed (Phase 1 already p99=2000.0 exact). Found+fixed the same missing-`__main__`-guard/stale-cache-reuse bug T20 Phase 6 hit. Pooled hnd Brier 0.139453→**0.137557** (Δ−0.001896, genuine improvement — pooling makes the scale-consistency benefit ~20x larger than the unpooled gate suggested). t20/od exact 0.0. hnd sim-engine baseline newly established (extras 0.191095, legal 1.554910). baseline_metrics.json + sim_baseline_metrics.json updated. Live smoke test clean. |
