"""
Phase 1 of mc_runrate_feedback_plan.md -- extracts the canonical per-format score trajectory
`src/simulation/innings_simulator.py` loads at runtime (`data/processed/{fmt}_score_trajectory
.json`) from Phase 0's diagnostic measurement (`outputs/score_trajectory_{fmt}.json`, written by
`scripts/measure_score_feedback.py`).

Uses the ALL_REACHING curve, not the plan's originally-specified full_length one: the simulator
needs E[current_team_score at ball B | still batting at ball B] -- exactly what "all_reaching"
measures (mean score over every innings that reached ball B, regardless of how it later ended).
"full_length" instead conditions on how the innings ENDS (used its full quota, or was bowled
out) -- a fact the simulator cannot know mid-innings when it is deciding what score to feed the
model, and which measurably runs low for innings 2 (a chase that used its whole quota is one
that didn't chase comfortably; see mc_runrate_feedback_plan.md's Phase 0 Results, "Par score
trajectory" paragraph). Only 2026-08-09 was measured "full_length"; this script never
recomputes -- it just re-projects the already-measured all_reaching curve into the slim shape
the simulator wants, so it is fast and re-runnable without re-touching the model.

Run once per format after scripts/measure_score_feedback.py has produced the
outputs/score_trajectory_<fmt>.json it reads from; re-run whenever that file is refreshed
(retrain, new data). Not run automatically -- innings_simulator.py just loads whatever is on
disk, same convention as this repo's other data/processed/ caches (see coding-standards.md).

Usage:
    py scripts/build_score_trajectory.py --formats od t20
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def build(format_type: str) -> None:
    src_path = PROJECT_ROOT / "outputs" / f"score_trajectory_{format_type}.json"
    if not src_path.exists():
        raise FileNotFoundError(
            f"No diagnostic trajectory at {src_path} -- run "
            f"`py scripts/measure_score_feedback.py --formats {format_type}` first."
        )
    src = json.loads(src_path.read_text())

    innings_out = {}
    for inn, curves in src["innings"].items():
        vec = curves.get("all_reaching")
        if vec is None:
            raise KeyError(f"{src_path} innings {inn} has no 'all_reaching' curve -- stale format?")
        innings_out[inn] = vec

    out = {
        "format": format_type,
        "source": "all_reaching",
        "source_file": str(src_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "total_balls": src["total_balls"],
        "innings": innings_out,
    }
    out_path = PROJECT_ROOT / "data" / "processed" / f"{format_type}_score_trajectory.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[{format_type}] wrote {out_path} "
          f"(innings 1 ends at {innings_out['1'][-1]:.1f}, "
          f"innings 2 ends at {innings_out['2'][-1]:.1f})")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--formats", nargs="+", default=["od", "t20"],
                   choices=["t20", "od", "hnd", "t20w", "odw", "hndw"])
    args = p.parse_args()
    for fmt in args.formats:
        build(fmt)


if __name__ == "__main__":
    main()
