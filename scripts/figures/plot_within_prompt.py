#!/usr/bin/env python3
"""Visualize the p_own evidence chain without accessing any test data.

The figures deliberately use within-domain standardization only for display.  All
tests and original-scale estimates are left in the source audit artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import figstyle as fs
import numpy as np


DOMAINS = fs.DOMAINS


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_alignment() -> dict[str, list[dict]]:
    return {
        domain: read_jsonl(ALIGNMENT / f"rows_{domain}_p_own.jsonl")
        for domain in DOMAINS
    }


def zscore(values: np.ndarray) -> np.ndarray:
    scale = float(np.std(values))
    if scale <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.mean(values))) / scale


def bootstrap_mean_ci(values: np.ndarray, seed: int, draws: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(np.quantile(means, (0.025, 0.975)).tolist())


def mixed_prompt_gaps(rows: list[dict]) -> np.ndarray:
    rewards = zscore(np.asarray([float(row["training_reward"]) for row in rows]))
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row, reward in zip(rows, rewards, strict=True):
        grouped[str(row["task_id"])].append((int(row["correct"]), float(reward)))
    gaps = []
    for values in grouped.values():
        correct = [reward for label, reward in values if label == 1]
        incorrect = [reward for label, reward in values if label == 0]
        if correct and incorrect:
            gaps.append(float(np.mean(correct) - np.mean(incorrect)))
    return np.asarray(gaps)


def plot_within_prompt(rows_by_domain: dict[str, list[dict]], destination: Path) -> None:
    fig, axes = fs.subplots(1, 4, height=2.1, sharey=True)
    rng = np.random.default_rng(20260916)
    for index, (domain, ax) in enumerate(zip(DOMAINS, axes, strict=True)):
        gaps = mixed_prompt_gaps(rows_by_domain[domain])
        violin = ax.violinplot(gaps, positions=[0], widths=0.8, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(fs.FILL[domain])
            body.set_alpha(1.0)
            body.set_edgecolor("none")
        jitter = rng.uniform(-0.25, 0.25, size=len(gaps))
        ax.scatter(jitter, gaps, s=2.5, color=fs.LINE[domain], alpha=0.35, linewidths=0,
                   rasterized=True)
        mean = float(np.mean(gaps))
        low, high = bootstrap_mean_ci(gaps, 100 + index)
        ax.errorbar(0, mean, yerr=[[mean - low], [high - mean]], fmt="o", color=fs.INK,
                    markersize=fs.MARKER_SIZE + 0.6, elinewidth=fs.ERR_LW, capthick=fs.ERR_LW,
                    zorder=3)
        ax.axhline(0, color=fs.MUTED, lw=fs.REF_LW, zorder=1)
        ax.set_xlim(-0.65, 0.65)
        ax.set_xticks([])
        ax.set_title(fs.LABEL[domain])
        ax.spines["bottom"].set_visible(False)
    axes[0].set_ylabel("Correct $-$ incorrect gap in\nstandardized $\\log p_{\\mathrm{ind}}$")
    fs.save(fig, destination.with_suffix(".pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce within-prompt plot from frozen rollout rows.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ALIGNMENT = args.input_dir
    plot_within_prompt(load_alignment(), args.output)
