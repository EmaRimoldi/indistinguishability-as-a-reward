#!/usr/bin/env python3
"""The indistinguishability reward against the teacher's correctness judgment.

Both scores are read on the same training rollouts of the three Indist. seeds:
log p_ind is the reward the policy was trained on, log q is what the same teacher
answers when asked whether the response is correct. Points are sampled evenly from
the three seeds; titles give the mean over seeds. Reads judge_vs_reward_data.json.
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
from matplotlib.lines import Line2D

DOMAINS = list(fs.DOMAINS)


def plot(data: dict, output: Path) -> None:
    fig, axes = fs.subplots(1, 4, height=1.95)
    for domain, axis in zip(DOMAINS, axes, strict=True):
        entry = data["domains"][domain]
        points = np.array(entry["points"], dtype=float)
        for label, color in ((0, fs.LIGHT[domain]), (1, fs.LINE[domain])):
            mask = points[:, 2] == label
            axis.scatter(points[mask, 0], points[mask, 1], s=2.5, linewidths=0, color=color,
                         alpha=.5, rasterized=True)
        axis.set_title(f"{fs.LABEL[domain]}\n"
                       fr"$\rho$ {entry['spearman']:.2f}   $R^2$ {entry['r2_pooled']:.2f}")
        axis.set_xlabel(r"$\log p_{\mathrm{ind}}$ (reward)")
    axes[0].set_ylabel(r"$\log q$ (correctness judgment)")
    keys = [Line2D([0], [0], lw=0, marker="o", markersize=3.5, color=color, label=label)
            for color, label in ((fs.KEY_DARK, "Correct"), (fs.KEY_LIGHT, "Incorrect"))]
    fs.legend_top(fig, keys, handlelength=1.0)
    fs.save(fig, output)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=results_root() / "figure_data/judge_vs_reward_data.json")
    parser.add_argument("--output", type=Path, default=here / "judge_vs_reward.pdf")
    args = parser.parse_args()
    plot(json.loads(args.data.read_text()), args.output)


if __name__ == "__main__":
    main()
