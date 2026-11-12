"""Plot zero-shot target-domain return per training condition.

Reads the per-seed SAC results and writes a light and a dark version of the
figure used in the README.

    python plot_results.py
"""

import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path("results")
OUT_DIR = Path("assets")
TARGET_ENV = "CustomHopper-target-v0"

# Display order (top to bottom): condition key -> (label, per-seed CSV).
CONDITIONS = [
    ("sac_src-std", "Source only (baseline)", "ablation"),
    ("sac_udr10", "UDR ±10%", "udr_study"),
    ("sac_udr30", "UDR ±30%", "udr_study"),
    ("sac_udr50", "UDR ±50%", "udr_study"),
    ("sac_tdr-std", "Torso-mass DR", "ablation"),
    ("sac_tgt-std", "Trained on target (reference)", "udr_study"),
]

THEMES = {
    "light": {
        "surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "series": "#2a78d6",
    },
    "dark": {
        "surface": "#1a1a19", "text": "#f0efec", "muted": "#c3c2b7",
        "grid": "#2c2c2a", "axis": "#383835", "series": "#3987e5",
    },
}


def load_target_returns():
    """Return {condition: [per-seed mean target return]}."""
    returns = {}
    for folder in {folder for _, _, folder in CONDITIONS}:
        with open(RESULTS / folder / "per_seed.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["env"] == TARGET_ENV:
                    returns.setdefault(row["condition"], []).append(float(row["mean_return"]))
    return returns


def plot(returns, theme_name):
    t = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    n = len(CONDITIONS)
    for i, (key, label, _) in enumerate(CONDITIONS):
        y = n - 1 - i
        values = returns[key]
        mean = statistics.fmean(values)
        ax.plot([min(values), max(values)], [y, y], color=t["grid"], lw=2,
                solid_capstyle="round", zorder=1)
        ax.scatter(values, [y] * len(values), s=64, color=t["series"],
                   edgecolors=t["surface"], linewidths=2, zorder=3)
        ax.plot([mean, mean], [y - 0.28, y + 0.28], color=t["text"], lw=2.5,
                solid_capstyle="round", zorder=4)
        ax.text(1700, y, f"{mean:,.0f}", va="center", ha="right",
                color=t["text"], fontsize=10, fontweight="bold")

    ax.set_yticks(range(n))
    ax.set_yticklabels([label for _, label, _ in reversed(CONDITIONS)],
                       color=t["text"], fontsize=10)
    ax.set_xlim(250, 1720)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Mean episode return on the target domain", color=t["muted"], fontsize=9)
    ax.tick_params(axis="x", colors=t["muted"], labelsize=9, length=0)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.grid(axis="x", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])

    fig.text(0.015, 0.965, "Zero-shot transfer to the target domain",
             color=t["text"], fontsize=13, fontweight="bold", va="top")
    fig.text(0.015, 0.895,
             "SAC, 200k steps. Dots: three training seeds (50 episodes each). "
             "Bar and number: mean of the seeds.",
             color=t["muted"], fontsize=9, va="top")
    fig.subplots_adjust(left=0.27, right=0.985, top=0.80, bottom=0.14)

    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"target_return_{theme_name}.png"
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)
    return path


def main():
    returns = load_target_returns()
    for theme in THEMES:
        print(f"Saved {plot(returns, theme)}")


if __name__ == "__main__":
    main()
