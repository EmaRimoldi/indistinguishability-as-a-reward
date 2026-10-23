#!/usr/bin/env python3
"""Render the main-paper reward distributions.

Default: the three-seed version from the compact file reward_distributions_seeds.json,
which also carries the conditional means and AUC that Table 3 used to report.
With --input-dir it falls back to the original single-run rendering from rollout rows.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from iar_repro.records import results_root

import matplotlib

matplotlib.use("Agg")
import figstyle as fs
import numpy as np
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DOMAINS = list(fs.DOMAINS)


def read_scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    scores, correct = [], []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            score = row.get("score_probability")
            if score is not None:
                scores.append(float(score))
                correct.append(int(row["correct"]))
    return np.asarray(scores), np.asarray(correct)


def reflected_gaussian_kde(values: np.ndarray, grid: np.ndarray,
                           bandwidth: float = 0.035) -> np.ndarray:
    """Gaussian KDE on [0, 1] with reflection at both probability boundaries."""
    density = np.zeros_like(grid, dtype=float)
    normalizer = math.sqrt(2.0 * math.pi)
    for start in range(0, len(values), 1000):
        batch = values[start:start + 1000]
        for reflected in (batch, -batch, 2.0 - batch):
            z = (grid[:, None] - reflected[None, :]) / bandwidth
            density += np.exp(-0.5 * z * z).sum(axis=1)
    return density / (len(values) * bandwidth * normalizer)


def correctness_keys() -> list:
    """Legend keys valid for every domain: correct = dark tone over a pastel area, incorrect = light tone."""
    return [(Patch(facecolor=fs.KEY_FILL, linewidth=0), Line2D([0], [0], color=fs.KEY_DARK, lw=fs.MEAN_LW)),
            Line2D([0], [0], color=fs.KEY_LIGHT, lw=fs.MEAN_LW)]


def annotate_means(axis, incorrect: float, correct: float, band: float) -> None:
    """Double arrow between the conditional means at height ``band``, the gap above it, and
    each mean beside its end of the arrow (below the end when it would leave the panel)."""
    axis.annotate("", xy=(correct, band), xytext=(incorrect, band),
                  arrowprops={"arrowstyle": "<->", "color": fs.MUTED, "linewidth": 0.7,
                              "shrinkA": 0, "shrinkB": 0, "mutation_scale": 6})
    axis.annotate(rf"$\Delta$ {correct - incorrect:.2f}", xy=((incorrect + correct) / 2, band),
                  xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom",
                  fontsize=7, color=fs.MUTED)
    width = 0.19  # a "0.00" label at 7 pt, in units of p_ind
    halo = [patheffects.withStroke(linewidth=2.0, foreground="white")]
    for mean, side, color in ((incorrect, -1, fs.MUTED), (correct, 1, fs.INK)):
        if 0 <= mean + side * (0.02 + width) <= 1:
            axis.annotate(f"{mean:.2f}", xy=(mean, band), xytext=(side * 2.5, 0),
                          textcoords="offset points", ha="left" if side > 0 else "right",
                          va="center", fontsize=7, color=color, path_effects=halo)
        else:
            axis.annotate(f"{mean:.2f}", xy=(mean, band), xytext=(0, -2.5),
                          textcoords="offset points", ha="center", va="top", fontsize=7,
                          color=color, path_effects=halo)


def plot_seeds(data: dict, output: Path) -> None:
    """Pooled densities with the conditional means marked inside each distribution."""
    grid = np.asarray(data["grid"])
    fig, axes = fs.subplots(1, 4, height=1.75, sharex=True)
    for axis, domain in zip(axes, DOMAINS):
        entry = data["domains"][domain]
        means = {1: np.asarray(entry["mean_correct_per_seed"]),
                 0: np.asarray(entry["mean_incorrect_per_seed"])}
        density = {1: np.asarray(entry["density_correct"]),
                   0: np.asarray(entry["density_incorrect"])}
        top = max(density[0].max(), density[1].max())
        band = top * 1.12
        tone = {1: fs.LINE[domain], 0: fs.LIGHT[domain]}
        axis.fill_between(grid, 0, density[1], color=fs.FILL[domain], alpha=.75, linewidth=0)
        for outcome in (0, 1):
            axis.plot(grid, density[outcome], color=tone[outcome], lw=fs.MEAN_LW,
                      solid_capstyle="round")
            mean = float(means[outcome].mean())
            axis.vlines(mean, 0, band, color=tone[outcome], linewidth=fs.REF_LW)
            if float(means[outcome].max() - means[outcome].min()) > 0.005:
                axis.fill_betweenx([0, band], means[outcome].min(), means[outcome].max(),
                                   color=tone[outcome], alpha=0.25, linewidth=0)
        annotate_means(axis, float(means[0].mean()), float(means[1].mean()), band)
        axis.set_title(fs.LABEL[domain])
        axis.set_xlabel(r"$p_{\mathrm{ind}}$")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, band * 1.18)
    axes[0].set_ylabel("Density")  # shared variable: labelled once, on the leftmost panel
    fs.legend_top(fig, correctness_keys(), labels=["Correct", "Incorrect"])
    fs.save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "reward_distributions_by_correctness.pdf")
    args = parser.parse_args()
    compact = results_root() / "figure_data/reward_distributions_seeds.json"
    if args.input_dir is None:
        plot_seeds(json.loads(compact.read_text()), args.output)
        return

    fig, axes = fs.subplots(1, 4, height=1.75, sharex=True)
    grid = np.linspace(0, 1, 501)
    for axis, domain in zip(axes, DOMAINS):
        scores, correct = read_scores(args.input_dir / f"rows_{domain}_p_own.jsonl")
        tone = {1: fs.LINE[domain], 0: fs.LIGHT[domain]}
        for outcome in (0, 1):
            density = reflected_gaussian_kde(scores[correct == outcome], grid)
            if outcome == 1:
                axis.fill_between(grid, 0, density, color=fs.FILL[domain], alpha=.75, linewidth=0)
            axis.plot(grid, density, color=tone[outcome], lw=fs.MEAN_LW)
        axis.set_title(fs.LABEL[domain])
        axis.set_xlabel(r"$p_{\mathrm{ind}}$")
        axis.set_xlim(0, 1)
        axis.set_ylim(bottom=0)
    axes[0].set_ylabel("Density")
    fs.legend_top(fig, correctness_keys(), labels=["Correct", "Incorrect"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fs.save(fig, args.output)


if __name__ == "__main__":
    main()
