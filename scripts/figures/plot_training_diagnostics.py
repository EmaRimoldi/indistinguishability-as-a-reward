#!/usr/bin/env python3
"""GRPO surrogate loss and training diagnostics over all seeds of each GRPO run.

Reads every trainer_log_history.jsonl of the Indist., Quality and Oracle runs
(three seeds each), merges resumed attempts by step,
and draws per condition the logged values of all seeds (faint) and a 51-step
centered average of their mean across seeds (bold).

Outputs grpo_loss.pdf and training_diagnostics_4x4.pdf next to this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import figstyle as fs
import numpy as np

from iar_repro.records import log_paths

DOMAINS = tuple((d, fs.LABEL[d]) for d in fs.DOMAINS)
CONDITIONS = fs.CONDITIONS
SEEDS = ("seed23", "seed42", "seed7")
METRICS = (("clip_ratio/region_mean", "Clipped-token ratio"), ("grad_norm", "Gradient norm"),
           ("entropy", "Policy entropy (nats)"), ("reward_std", "Generation-batch\nreward SD"))
STEPS = np.arange(1, 2001)


def history(condition: str, domain: str, seed: str) -> dict[str, np.ndarray] | None:
    """Metric -> values on steps 1..2000 (NaN where missing); None if the run is absent."""
    paths = log_paths(condition, domain, seed)
    if not paths:
        return None
    by_step: dict[int, dict] = {}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if "step" in row and "loss" in row:
                        by_step[int(row["step"])] = row
    out = {}
    for metric in ("loss",) + tuple(m for m, _ in METRICS):
        out[metric] = np.array([float(by_step[s][metric]) if s in by_step and metric in by_step[s]
                                else np.nan for s in STEPS])
    return out


def centered_mean(values: np.ndarray, window: int = 51) -> np.ndarray:
    ok = np.isfinite(values)
    weights = np.ones(window)
    num = np.convolve(np.where(ok, values, 0.0), weights, mode="same")
    den = np.convolve(ok.astype(float), weights, mode="same")
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def draw(axis, runs: list[dict], metric: str, domain: str, condition: str) -> None:
    """Every logged value of every seed (faint) under the smoothed mean across seeds; the
    colour is the domain's, the dash pattern and the markers are the condition's."""
    color = fs.LINE[domain]
    stack = np.vstack([r[metric] for r in runs])
    for values in stack:
        axis.plot(STEPS, values, color=color, alpha=.06, linewidth=.3, rasterized=True, zorder=1)
    with np.errstate(all="ignore"):
        mean = np.nanmean(stack, axis=0)
    offset = 100 + 133 * CONDITIONS.index(condition)
    axis.plot(STEPS, centered_mean(mean), color=color, linewidth=1.0, ls=fs.DASH[condition],
              marker=fs.MARKER[condition], markevery=(offset, 400), markersize=2.8,
              markeredgecolor="white", zorder=3)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=here)
    args = parser.parse_args()
    runs = {(d, c): [h for s in SEEDS if (h := history(c, d, s)) is not None]
            for d, _ in DOMAINS for c in CONDITIONS}
    fig, axes = fs.subplots(2, 2, width=0.94, height=3.4, sharex=True)
    for index, (axis, (domain, label)) in enumerate(zip(axes.flat, DOMAINS)):
        for condition in CONDITIONS:
            draw(axis, runs[(domain, condition)], "loss", domain, condition)
        axis.set_yscale("symlog", linthresh=1e-2)
        axis.set_title(label)
        if index >= 2:
            axis.set_xlabel("Optimizer step")
        if index % 2 == 0:
            axis.set_ylabel("GRPO surrogate loss")
        axis.set_xticks([0, 500, 1000, 1500, 2000])
    fs.legend_top(fig, fs.condition_line_keys())
    fs.save(fig, args.out_dir / "grpo_loss.pdf")

    fig, axes = fs.subplots(4, 4, height=5.3, sharex=True)
    for r, (metric, row_label) in enumerate(METRICS):
        for c, (domain, label) in enumerate(DOMAINS):
            axis = axes[r, c]
            for condition in CONDITIONS:
                draw(axis, runs[(domain, condition)], metric, domain, condition)
            if r == 0:
                axis.set_title(label)
            if c == 0:
                axis.set_ylabel(row_label)
            if r == len(METRICS) - 1:
                axis.set_xlabel("Optimizer step")
            if metric == "grad_norm":
                axis.set_yscale("symlog", linthresh=.01)
                axis.set_ylim(bottom=0)
            axis.set_xticks([0, 1000, 2000])
    fs.legend_top(fig, fs.condition_line_keys())
    fs.save(fig, args.out_dir / "training_diagnostics_4x4.pdf")
    for key, value in runs.items():
        print(*key, "seeds:", len(value))


if __name__ == "__main__":
    main()
