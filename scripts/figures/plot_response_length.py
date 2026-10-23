#!/usr/bin/env python3
"""Response length during training, one panel per domain.

Reads response_length_data.json, written by
scripts/revision_20260922/response_length_data.py from the trainer log
histories. No model calls, no training, no test data.
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
from matplotlib.ticker import LogLocator, ScalarFormatter

DOMAINS = list(fs.DOMAINS)
# Colour separates the domains, marker and dash pattern separate the rewards.
METHODS = list(fs.CONDITIONS)


def plot(data: dict, output: Path) -> None:
    x = np.array(data["bin_end_steps"])
    fig, axes = fs.subplots(1, 4, height=1.8, sharex=True)
    for domain, axis in zip(DOMAINS, axes, strict=True):
        color = fs.LINE[domain]
        # What the teacher itself writes, the length the student is imitating.
        axis.axhline(data["teacher_length"][domain], color=fs.MUTED, lw=0.9,
                     ls=fs.TEACHER_DASH, zorder=1)
        for condition in METHODS:
            per_seed = np.array(list(data["conditions"][condition][domain].values()))
            for curve in per_seed:
                axis.plot(x, curve, color=color, lw=fs.SEED_LW, ls=fs.DASH[condition],
                          alpha=fs.SEED_ALPHA, zorder=2)
            axis.plot(x, per_seed.mean(axis=0), color=color, lw=fs.MEAN_LW, ls=fs.DASH[condition],
                      marker=fs.MARKER[condition], markeredgecolor="white", zorder=3)
        axis.set_yscale("log")
        # Plain numbers on the log axis, and only at 1, 2 and 5 of each decade.
        axis.yaxis.set_major_formatter(ScalarFormatter())
        axis.yaxis.set_minor_locator(LogLocator(subs=(2.0, 5.0), numticks=12))
        axis.yaxis.set_minor_formatter(ScalarFormatter())
        axis.set_title(fs.LABEL[domain])
        axis.set_xlabel("Optimizer step")
        axis.set_xticks([0, 1000, 2000])
        axis.set_xlim(0, 2100)
    axes[0].set_ylabel("Mean response length (tokens)")
    fs.legend_top(fig, fs.condition_line_keys() + [fs.teacher_key()])
    fs.save(fig, output)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=results_root() / "figure_data/response_length_data.json")
    parser.add_argument("--output", type=Path, default=here / "response_length_by_reward.pdf")
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    assert data["bin_updates"] == 200 and len(data["bin_end_steps"]) == 10
    plot(data, args.output)


if __name__ == "__main__":
    main()
