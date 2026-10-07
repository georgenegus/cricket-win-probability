# Setup — Getting From a Fresh Clone to a Running App

> **Purpose:** How to install dependencies, obtain the data, train the models and run the checks
> **Scope:** All six formats (`t20`, `od`, `hnd`, `t20w`, `odw`, `hndw`) · **Updated:** 2026-10-07

---

## What is and is not in the repository

The repository holds code, documentation and the locked regression baselines. It does **not** hold
the match data or the trained models: the working `data/` directory is about 4.8 GB and is rebuilt
locally.

| Path | In the repo? | How it is produced |
|------|:------------:|--------------------|
| `data/raw/<format>/<competition>/*.yaml` | No | Downloaded from Cricsheet (step 2) |
| `data/processed/*.parquet` | No | Feature caches, built during training (step 3) |
| `data/processed/*.pkl` | No | Trained models, built during training (step 3) |
| `data/processed/baseline_metrics.json` | Yes | Locked win-probability and first-innings baseline |
| `data/processed/sim_baseline_metrics.json` | Yes | Locked delivery-outcome (Monte Carlo) baseline |
| `data/processed/mc_calibration_metrics*.json` | Yes | Locked Monte Carlo simulator calibration results |
| `data/reference/venue_overrides.json` | Yes | Hand-maintained home-venue overrides |
| `credentials.env` | No | Your own Betfair credentials (step 5, optional) |

All match data comes from [Cricsheet](https://cricsheet.org), which publishes ball-by-ball records
under an open licence.

---

## 1. Install

Developed and tested on Python 3.14.5 on Windows 11. Commands below use the `py` launcher; use
`python` on other platforms.

```bash
py -m pip install -r requirements.txt
```

Run every command from the repository root. Paths such as `data/raw` are relative to it.

---

## 2. Download the data

```bash
py scripts/import_and_retrain.py --full --dry-run   # report what would be imported, write nothing
py scripts/import_and_retrain.py --full             # import, then train (see step 3)
```

`--full` downloads Cricsheet's complete `all_male.zip` and `all_female.zip` archives and files each
match under `data/raw/<format>/<competition>/`. Men's and women's matches go to separate format
directories and never share a training corpus.

For reference, the corpus the current baselines were locked on (2026-08-07):

| Format | Matches | Deliveries |
|--------|--------:|-----------:|
| `t20` | 10,634 | 2,936,761 |
| `od` | 4,125 | 2,640,385 |
| `hnd` | 167 | 31,952 |
| `t20w` | 3,200 | 728,751 |
| `odw` | 929 | 487,315 |
| `hndw` | 155 | 30,445 |

---

## 3. Train the models

When no model file exists for a format, `scripts/import_and_retrain.py` treats a full retrain as
due, so the `--full` run in step 2 goes on to train `t20`, `od`, `t20w`, `odw` and `hndw`. To force
a retrain later, pass `--retrain-full`.

Men's Hundred (`hnd`) is not covered by that script. Train it afterwards, once `t20` is built,
because `hnd` pools `t20` data at train time and seeds its player ratings from `t20`:

```python
# save as a script and run from the repository root
from src.data_processing.cricsheet_importer import retrain_full

if __name__ == "__main__":
    retrain_full("hnd")
```

For each format, `retrain_full` builds the team, player, phase and venue features and then fits
three models: the first-innings regressor, the second-innings win-probability classifier and the
per-ball delivery-outcome model that drives the Monte Carlo simulator.

Two things to know before starting:

- **Memory.** Development was done on a machine with 7.7 GB of RAM. A full `t20` retrain
  (2.9 million deliveries) can run out of memory if other heavy processes are running.
- **Model files are not checked against the code.** If a `.pkl` exists, it is loaded as is. Delete
  it to force a genuine retrain.

---

## 4. Run the app and the checks

```bash
py app.py                           # serves http://127.0.0.1:5000
```

The app loads the models for all six formats at startup, so step 3 must have completed for each.

```bash
py tests/regression_test.py         # win-probability and first-innings gate, all six formats
py tests/sim_baseline_metrics.py    # delivery-outcome (Monte Carlo) model gate
```

Both gates compare freshly computed metrics with the locked baselines, within the tolerance stored
in the baseline file. The baselines were locked on the corpus shown in step 2. A fresh Cricsheet
download includes later matches, so expect small differences from corpus drift.

---

## 5. Optional: Betfair odds and automation

The win-probability models, the simulator and the web app work without any credentials. Only the
Betfair odds scanner and bet settlement need them:

```bash
cp credentials.env.example credentials.env    # then fill in the three values
```

`credentials.env` is gitignored. Without it, the scan endpoint returns an error and everything else
runs normally.

The three `scripts/run_*.bat` files are Windows Task Scheduler wrappers for the scheduled import,
the daily scanner and bet settlement. They resolve the project root from their own location and
write to `logs/`.
