#!/usr/bin/env python3
"""The actor learns to fool the teacher, and becomes more accurate.

Two panels built from the same data and drawing code as the separate figures
they replace (plot_fooling_score.py and plot_seeds in
plot_training_trajectory.py):

(a) mean log p_ind of the training rollouts over the first 100 updates and then
    in four bins of 500 updates;
(b) held-out accuracy at every checkpoint.

Every panel draws a thin line per seed under the bold mean; the square marks the
checkpoint with the highest mean held-out accuracy, at the same step in all
panels. Reads fooling_rate_data.json and training_trajectory_seeds.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from iar_repro.records import results_root

import matplotlib

matplotlib.use("Agg")
import figstyle as fs
import numpy as np

DOMAINS = [(d, fs.LABEL[d], fs.SHORT[d], fs.LINE[d]) for d in fs.DOMAINS]
# The first 100 updates stand for the initial policy (x=0); then four equal
# 500-update bins ending at 500, 1000, 1500 and 2000.
REWARD_BINS = [(0, [0]), (500, [1, 2, 3, 4]), (1000, [5, 6, 7, 8, 9]),
               (1500, [10, 11, 12, 13, 14]), (2000, [15, 16, 17, 18, 19])]


def best_checkpoint(domain_data: dict) -> int:
    """Checkpoint with the highest mean held-out score across seeds."""
    steps = sorted(int(k) for k in domain_data["heldout_pct"])
    means = [float(np.mean(domain_data["heldout_pct"][str(s)])) for s in steps]
    return steps[int(np.argmax(means))]


def band(ax, x, seed_curves, color, mark_step) -> tuple[float, float]:
    """Thin line per seed under the bold mean, open circle at the start, square at the best step."""
    x = np.asarray(x)
    for curve in seed_curves:
        ax.plot(x, curve, color=color, lw=fs.SEED_LW, alpha=fs.SEED_ALPHA, zorder=2)
    mean = seed_curves.mean(axis=0)
    ax.plot(x, mean, color=color, lw=fs.MEAN_LW, marker=fs.MARKER["grpo_identity"],
            markeredgecolor="white", zorder=3)
    ax.scatter([x[0]], [mean[0]], s=(fs.MARKER_SIZE + 1.2) ** 2, facecolor="white",
               edgecolor=color, linewidth=0.8, zorder=4)
    i = int(np.argmin(np.abs(x - mark_step)))
    ax.scatter([x[i]], [mean[i]], s=(fs.MARKER_SIZE + 1.8) ** 2, marker="s", color=color,
               edgecolor="white", linewidth=0.5, zorder=5)
    return float(x[-1]), float(mean[-1])


def plot(fooling: dict, trajectory: dict, output: Path) -> None:
    fig, axes = fs.subplots(1, 2, width=1.0, height=2.0)

    # (a) mean log p_ind over bins of training rollouts, (b) held-out accuracy,
    # as in the trajectory figure.
    ends = ({}, {})
    for domain, _, short, color in DOMAINS:
        d = trajectory["domains"][domain]
        stop = best_checkpoint(d)
        tx = np.array([step for step, _ in REWARD_BINS])
        w = np.array([[np.mean([d["log_p_own_windows"][s][i] for i in idx])
                       for _, idx in REWARD_BINS] for s in trajectory["seeds"]])
        ends[0][short] = band(axes[0], tx, w, color, stop)
        steps = sorted(int(k) for k in d["heldout_pct"])
        vals = np.array([d["heldout_pct"][str(k)] for k in steps])
        ends[1][short] = band(axes[1], np.array(steps), vals.T, color, stop)
    axes[0].set_title("(a) Training self-attribution")
    axes[0].set_ylabel(r"Mean $\log p_{\mathrm{ind}}$")
    axes[1].set_title("(b) Held-out performance")
    axes[1].set_ylabel("Held-out accuracy (%)")
    axes[1].set_ylim(36, 76)
    axes[1].set_yticks([40, 50, 60, 70])
    for ax in axes:
        ax.set_xlabel("Optimizer step")
        ax.set_xticks([0, 500, 1000, 1500, 2000])
        ax.set_xlim(-100, 2100)

    for ax, labels in zip(axes, ends):
        fs.label_ends(ax, labels)
    fs.save(fig, output)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fooling", type=Path, default=results_root() / "figure_data/fooling_rate_data.json")
    parser.add_argument("--trajectory", type=Path, default=results_root() / "figure_data/training_trajectory_seeds.json")
    parser.add_argument("--output", type=Path, default=here / "learning_to_fool.pdf")
    args = parser.parse_args()
    plot(json.loads(args.fooling.read_text()), json.loads(args.trajectory.read_text()), args.output)


if __name__ == "__main__":
    main()
