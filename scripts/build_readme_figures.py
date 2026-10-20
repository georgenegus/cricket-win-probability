"""
Builds the calibration figure shown in README.md from data/processed/accuracy_report.json
(itself built by scripts/build_accuracy_report.py from the same out-of-fold predictions the
regression gate scores). Writes a light and a dark variant so the README can serve whichever
matches the reader's GitHub theme.

Run:
    py scripts/build_readme_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "accuracy_report.json"
OUT_DIR = PROJECT_ROOT / "docs" / "img"

FORMAT_LABELS = {
    "t20": "Men's T20",
    "od": "Men's one-day",
    "hnd": "Men's Hundred",
    "t20w": "Women's T20",
    "odw": "Women's one-day",
    "hndw": "Women's Hundred",
}

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink_2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "series": "#2a78d6",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink_2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "series": "#3987e5",
    },
}


def build_calibration_figure(report: dict, theme_name: str) -> Path:
    t = THEMES[theme_name]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2), sharex=True, sharey=True)
    fig.patch.set_facecolor(t["surface"])

    for ax, (fmt, label) in zip(axes.flat, FORMAT_LABELS.items()):
        entry = report[fmt]
        buckets = entry["calibration"]
        predicted = [b["predicted"] for b in buckets]
        actual = [b["actual"] for b in buckets]

        ax.set_facecolor(t["surface"])
        ax.plot([0, 1], [0, 1], color=t["muted"], linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.plot(predicted, actual, color=t["series"], linewidth=2, zorder=2)
        ax.scatter(predicted, actual, s=42, color=t["series"], edgecolors=t["surface"],
                   linewidths=1.5, zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xticklabels(["0%", "50%", "100%"])
        ax.set_yticklabels(["0%", "50%", "100%"])
        ax.set_aspect("equal")
        ax.grid(color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=t["muted"], labelsize=9, length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(t["axis"])

        ax.set_title(label, loc="left", fontsize=11, color=t["ink"], fontweight="bold", pad=18)
        ax.text(0, 1.03, f"Brier {entry['overall_brier']:.3f} · {entry['win_prob_sample_size']:,} states",
                transform=ax.transAxes, fontsize=8.5, color=t["ink_2"], va="bottom")

    for ax in axes[1]:
        ax.set_xlabel("Predicted chase win probability", fontsize=9, color=t["ink_2"])
    for ax in axes[:, 0]:
        ax.set_ylabel("Actual chase win rate", fontsize=9, color=t["ink_2"])

    fig.suptitle("Second-innings win probability: predicted vs actual (out-of-fold)",
                 x=0.06, ha="left", fontsize=13, color=t["ink"], fontweight="bold")
    fig.text(0.06, 0.935, "Each point is a 10-percentage-point bucket of ball-by-ball predictions. "
             "Dashed line = perfect calibration.", fontsize=9, color=t["ink_2"])
    fig.tight_layout(rect=(0.03, 0, 1, 0.93), h_pad=2.5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"calibration_{theme_name}.png"
    fig.savefig(out_path, dpi=150, facecolor=t["surface"])
    plt.close(fig)
    return out_path


def main():
    report = json.loads(REPORT_PATH.read_text())
    for theme_name in THEMES:
        print(f"Wrote {build_calibration_figure(report, theme_name)}")


if __name__ == "__main__":
    main()
