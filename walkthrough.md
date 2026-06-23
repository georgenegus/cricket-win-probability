# Walkthrough — Upgrade 2D: Phase-Aware Similarity

We have completed the implementation and validation of Upgrade 2D: Phase-Aware Similarity.

## Changes Made

### 1. Similarity Engine
We modified `SimpleSimilarityEngine` in [similarity_engine.py](file:///c:/Python/T20_Win_Probability_Model/src/models/similarity_engine.py):
- Added `phase_mode` configurations: `"none"`, `"filter"`, and `"feature"`.
- Implemented Approach B (Phase difference as a distance penalty) under `"feature"` mode using a mapped phase sequence (`{"powerplay": 0, "middle": 1, "death": 2}`).

### 2. Validation Framework
We enhanced [win_probability_validation.py](file:///c:/Python/T20_Win_Probability_Model/tests/win_probability_validation.py):
- Extended prediction loop tracking to record the phase of each sample.
- Added per-phase Brier score calculations (separating Powerplay, Middle, and Death overs).
- Updated the entry point to run a side-by-side comparison of the three modes.

---

## Validation Results

Running the model validation over 500 random states (using standard deviation fallbacks: runs=38.0, balls=34.0, wickets=2.5) produced the following metrics:

| Phase Mode | Overall Brier Score | Powerplay Brier | Middle Brier | Death Brier |
| :--- | :---: | :---: | :---: | :---: |
| **None** (No Phase Awareness) | 0.1543 | 0.2013 | 0.1335 | 0.1018 |
| **Filter** (Approach A - Hard Filter) | **0.1538** | 0.2015 | 0.1336 | **0.0978** |
| **Feature** (Approach B - Distance Penalty) | 0.1546 | 0.2020 | 0.1338 | 0.1012 |

### Key Observations
- **Approach A (Hard Filter)** provides the best results, yielding an overall Brier Score of **0.1538** and significantly improving death overs predictions to **0.0978** (from 0.1018).
- **Approach B (Distance Penalty)** slightly degrades overall performance (to **0.1546**) due to interference in neighbor distance space without custom weight optimization.
- **Recommendation**: Approach A (Hard Filter) is selected as the default mode.
