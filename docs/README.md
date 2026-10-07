# Documentation

| Document | What it covers |
|----------|----------------|
| [setup.md](setup.md) | Installing, downloading the data, training the models, running the checks |
| [system_guide.md](system_guide.md) | Full technical walkthrough of the pipeline, features and models |

## Selected build plans and analyses

Each model change in this project was planned, measured and written up phase by phase. These are a
selection of those working documents, published as written. They link to other plans that are not
published, so some links inside them do not resolve.

| Document | Summary |
|----------|---------|
| [Monte Carlo model evaluation](../markdowns/betting/model_evaluation_summary.md) | Calibration status of the match simulator. Includes a correction: an early "go for launch" verdict from a 30-match sample was withdrawn after a 100-match rerun showed two of four gates failing. |
| [Monte Carlo simulation build plan](../markdowns/betting/strategy4_monte_carlo_simulation_build_plan.md) | Design of the ball-by-ball simulator and the delivery-outcome model that drives it. |
| [T20 innings-total MAE gate](../markdowns/win_predictor/t20_mae_gate_plan.md) | Why the simulator's 25.0-run MAE target could not be met: the best leakage-safe model on the available data reaches 26.29, so the gate was re-anchored on evidence. |
| [Run-rate feedback loop](../markdowns/win_predictor/mc_runrate_feedback_plan.md) | Diagnosis and fix of a loop in which the simulator's own score fed back into the model, biasing ODI totals high (+18.17 to +12.72 runs). |
| [Hundred win-probability improvement](../markdowns/win_predictor/hundred_brier_improvement_plan.md) | The bespoke Hundred model initially lost to simply reusing the T20 model. Pooling T20 data at train time took its Brier score from 0.1780 to 0.1411. |
| [Hundred player-Elo revamp](../markdowns/win_predictor/hnd_player_elo_revamp_plan.md) | A rescale that shipped, and two dynamics changes that were tried and reverted because a 162-match corpus could not support them. |
| [Competition-tier calibration analysis](../markdowns/win_predictor/competition_tier_calibration_analysis.md) | Why associate-nation players topped the player-Elo rankings: a model-structure defect, not a scaling one. |
| [Women's formats](../markdowns/win_predictor/womens_games_ingestion_plan.md) | Adding `t20w`, `odw` and `hndw` as separate formats with their own data, ratings and models. |
