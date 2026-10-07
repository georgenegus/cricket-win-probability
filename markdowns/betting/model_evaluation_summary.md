# Strategy 4 Monte Carlo Model: Quality, Validity, and Earning Potential

> [!WARNING]
> **CORRECTION (2026-07-22): the original "GO FOR LAUNCH" verdict below was based on a
> non-representative 30-match / 500-sim sample and does not hold up.** Re-running the calibration
> battery on **100 matches at 3000 sims** — a far tighter estimate — shows **2 of 4 gates failing**.
> The optimistic figures in the original write-up (MAE ~16, bias +1.5) were an artefact of a small,
> lucky sample. The authoritative current numbers are in the table below. The model is **not** cleared
> for launch; see the revised conclusion.

## Current calibration status (100 matches, 3000 sims)

| Metric | Original snapshot (30m/500s) | Baseline (100m/3000s) | After off-by-one fix | Target | Verdict |
|---|---|---|---|---|---|
| Coverage [P5, P95] | 86.7% | 87.0% | 88.0% | 70–95% | ✅ pass |
| MAE | ~16.0 | 28.30 | **27.94** | < 25 | ❌ **FAIL** |
| Systematic Bias | +1.5 | +5.13 | **+0.74** | \|8\| | ✅ pass |
| Brier (O/U 155.5) | 0.2085 | 0.1924 | 0.1953 | < 0.25 | ✅ pass |
| Win-prob agreement | 9.5% | 13.2% | **13.1%** | < 12% | ❌ **FAIL** |
| Std dev | 27.5 | 34.7 | 34.4 | 25–55 | ✅ pass |

**What the off-by-one fix achieved:** correcting a train/serve skew in `balls_remaining` /
`striker_balls_faced` (the simulator was reading each game state as ~1 ball later than reality) cut
**systematic bias from +5.13 to +0.74 runs** — an 86% reduction — with no retrain. This confirms the
skew was the bias source.

**What remains:** MAE (27.94) and win-probability agreement (13.1%) still fail. These are
*dispersion / per-match sharpness* problems, not systematic skew, so the bias fix could not and did
not move them. Closing them requires improving the model's per-match precision (leading candidate:
bowler-usage realism — the simulator currently faces an unrealistically uniform top-5 attack every
match) or additional feature/modelling work — not another cheap correctness fix.

---

## Original write-up (retained for context — numbers superseded by the table above)

## 1. Distribution & Variance: The "Shape" of Reality
*   **Coverage Rate [P5, P95]: ~88%** · **Average Simulated Std Dev: ~34 runs**
*   **Meaning**: The simulator captures the wide unpredictability of T20 cricket rather than
    collapsing to a "safe" mean, and coverage sits close to the nominal 90%.
*   **Earning Potential**: Preserved natural variance makes the engine suited to pricing tail exotics
    (e.g. "Team to score 210+", "bowled out under 100") that crude normal-distribution books misprice
    — *once the point-estimate accuracy (MAE) is brought within gate.*

## 2. Accuracy & Bias: The "Center" of Reality
*   **Systematic Bias: +0.74 runs** (was the headline strength; genuinely good) · **MAE: ~28 runs (FAILING)**
*   **Meaning**: Bias is essentially eliminated — over/under lines are well-grounded on average. But
    the *per-match* error (MAE ~28) is too high: individual projections still miss by enough to erode
    edge on standard totals until sharpened.
*   **Earning Potential**: A near-zero bias means a projected 165 vs a 150 line is not a structural
    artefact — but with ~28-run per-match error, a 15-run delta is within noise. Edge claims on
    standard Total Runs markets should wait until MAE is inside the < 25 gate.

## 3. Market Calibration: Beating the Baseline
*   **Over/Under (155.5) Brier Score: 0.1953** (improved from 0.2085 as Monte-Carlo noise fell)
*   **Meaning**: Comfortably below the 0.25 coin-flip baseline — the model does add probabilistic
    information on standard totals.
*   **Earning Potential**: The strongest of the four signals. A sub-0.25 Brier on a binary market is
    consistent with identifiable +EV, but should be read alongside the failing MAE before sizing.

## 4. Structural Integrity
*   **Win-Probability Agreement: 13.1% (FAILING, target < 12%)**
*   **Meaning**: The delivery-level Monte Carlo win probabilities disagree with the Pre-Match
    predictor (Strategy 1) by slightly more than the 12% tolerance. This tracks the MAE problem — the
    disagreement is downstream of per-match score spread, not of bias.
*   **Earning Potential**: Until this is inside tolerance, treat in-play / derivative pricing from the
    simulator with caution — the micro and macro layers are not yet tightly agreeing.

## Revised Conclusion: NOT YET — one gate-blocker remains (MAE / win-prob)
The engine is **structurally sound and now bias-free**, and Brier/variance/coverage all pass. But
**MAE (~28) and win-prob agreement (13.1%) fail** at an honest sample size, so the model is **not**
cleared for live exotic scanning. Next step is a per-match *sharpness* improvement (bowler-usage
realism is the lead hypothesis), re-measured on the 100-match / 3000-sim battery, before any
launch decision is revisited.
