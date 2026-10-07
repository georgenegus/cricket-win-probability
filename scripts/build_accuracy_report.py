"""
Builds data/processed/accuracy_report.json -- the real, measured per-format accuracy data that
app.py's /api/accuracy serves to the Analytics tab (Accuracy Overview / First Innings views).

WHY THIS EXISTS: /api/accuracy previously returned HARDCODED literals for every format -- a
stale Brier score and an entirely fabricated calibration curve per format (t20 0.1261, od 0.1381,
hnd 0.1434), none of which matched the real locked baselines, and none at all for the three
women's formats. This script replaces that with genuine measurements, computed by the SAME
evaluate_format_models() the regression gate uses, so the UI can never disagree with
baseline_metrics.json about how good a model is.

Covers all six ACTIVE_FORMATS. Per format it records:
  - win-probability overall + per-phase Brier (full second-innings corpus, out-of-fold)
  - the predicted-vs-actual calibration curve in 0.1-wide buckets
  - first-innings MAE / RMSE / per-phase MAE
  - the sample sizes behind each, so the UI can show what the numbers rest on

Runs the formats sequentially (never concurrently) and frees each format's frames before the
next -- this box has 7.7GB and evaluate_format_models holds a full delivery corpus at peak.

Run:
    py scripts/build_accuracy_report.py                    # all six formats
    py scripts/build_accuracy_report.py --formats t20 t20w # a subset (merges into the existing file)
"""

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys
import gc
import json
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data_processing.format_spec import ACTIVE_FORMATS
from tests.regression_test import evaluate_format_models

OUT_PATH = PROJECT_ROOT / "data" / "processed" / "accuracy_report.json"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--formats", nargs="+", default=list(ACTIVE_FORMATS),
                   choices=list(ACTIVE_FORMATS))
    args = p.parse_args()

    # Merge into whatever is already on disk, so a subset run never silently drops the
    # formats it didn't recompute.
    report = {}
    if OUT_PATH.exists():
        try:
            report = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        except Exception:
            print(f"[warn] could not parse existing {OUT_PATH.name}; starting fresh")

    print("=" * 78)
    print(f"BUILDING ACCURACY REPORT -- {', '.join(args.formats)}")
    print("=" * 78, flush=True)

    for fmt in args.formats:
        t0 = time.time()
        print(f"\n--- {fmt.upper()} ---", flush=True)
        res = evaluate_format_models(fmt, include_calibration=True)

        wp, fi = res["win_probability"], res["first_innings"]
        report[fmt] = {
            "overall_brier": wp["overall_brier"],
            "phase_brier": wp["per_phase_brier"],
            "win_prob_sample_size": wp["sample_size"],
            "calibration": res["calibration"],
            "first_innings": fi,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        print(f"  Brier {wp['overall_brier']:.6f} (n={wp['sample_size']:,})  "
              f"first-innings MAE {fi['overall_mae']:.2f} (n={fi['sample_size']:,})  "
              f"{len(res['calibration'])} calibration buckets  [{time.time()-t0:.0f}s]", flush=True)
        gc.collect()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_PATH} ({', '.join(k for k in report if k != 'tolerance')})")


if __name__ == "__main__":
    main()
