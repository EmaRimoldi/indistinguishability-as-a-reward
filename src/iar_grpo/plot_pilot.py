from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "reward": "#168577",
    "quality": "#3976B8",
    "fooling": "#B84468",
    "correct": "#D97706",
    "incorrect": "#7656B3",
    "neutral": "#546E7A",
}


def create_all_plots(run_dir: Path) -> list[Path]:
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths = []
    paths.extend(plot_validation_learning_curves(run_dir, figures))
    paths.extend(plot_training_diagnostics(run_dir, figures))
    paths.extend(plot_reward_distributions(run_dir, figures))
    paths.extend(plot_optimization_diagnostics(run_dir, figures))
    return paths


def plot_validation_learning_curves(run_dir: Path, figures: Path) -> list[Path]:
    summary = json.loads((run_dir / "evaluation_summary.json").read_text())
    states = summary["validation"]
    steps = np.array([int(state["step"]) for state in states])
    quality = np.array([float(state["pass@1"]) for state in states])
    p_own = np.array([float(state["p_own_mean"]) for state in states])
    p_own_sem = np.array([
        float(state["p_own_std"]) / math.sqrt(int(state["n"])) for state in states
    ])
    sampled_fooling = np.array([float(state["sampled_fooling_rate"]) for state in states])
    argmax_fooling = np.array([float(state["argmax_fooling_rate"]) for state in states])
    correct = np.array([
        np.nan if state["mean_p_own_correct"] is None else float(state["mean_p_own_correct"])
        for state in states
    ])
    incorrect = np.array([
        np.nan if state["mean_p_own_incorrect"] is None else float(state["mean_p_own_incorrect"])
        for state in states
    ])
    selected_step = int(summary.get("selection", {}).get("selected_step", steps[-1]))

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.3), sharex=True, constrained_layout=True)
    axes[0].plot(steps, quality, "o-", color=COLORS["quality"], label="validation pass@1")
    lo, hi = zip(*(
        wilson_interval(int(state["passed"]), int(state["n"])) for state in states
    ), strict=True)
    axes[0].fill_between(steps, lo, hi, color=COLORS["quality"], alpha=0.14, linewidth=0)
    axes[0].set_ylabel("pass@1")
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False, loc="best")

    axes[1].plot(steps, p_own, "o-", color=COLORS["reward"], label="mean $p_{own}$")
    axes[1].fill_between(
        steps,
        np.maximum(0, p_own - 1.96 * p_own_sem),
        np.minimum(1, p_own + 1.96 * p_own_sem),
        color=COLORS["reward"], alpha=0.14, linewidth=0,
    )
    axes[1].plot(
        steps, argmax_fooling, "s--", color=COLORS["fooling"], label="argmax fooling"
    )
    axes[1].plot(
        steps,
        sampled_fooling,
        "^:",
        color=COLORS["neutral"],
        label="emitted-label fooling",
    )
    axes[1].set_ylabel("identity signal")
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False, ncol=2, loc="best")

    axes[2].plot(steps, correct, "o-", color=COLORS["correct"], label="correct responses")
    axes[2].plot(steps, incorrect, "s--", color=COLORS["incorrect"], label="incorrect responses")
    axes[2].set_ylabel("mean $p_{own}$")
    axes[2].set_xlabel("processed training groups")
    axes[2].set_ylim(0, 1)
    axes[2].legend(frameon=False, ncol=2, loc="best")
    for axis in axes:
        axis.axvline(
            selected_step,
            color=COLORS["neutral"],
            linestyle=":",
            linewidth=1.1,
            alpha=0.75,
        )
    style_axes(axes)
    fig.suptitle("Validation learning curve", fontweight="bold")
    return save_figure(fig, figures / "pilot_validation_learning_curves")


def plot_training_diagnostics(run_dir: Path, figures: Path) -> list[Path]:
    groups = read_jsonl(run_dir / "training_groups.jsonl")
    x = np.array([int(row["group_index"]) for row in groups])
    reward = np.array([float(row["reward_mean"]) for row in groups])
    spread = np.array([float(row["reward_std"]) for row in groups])
    fooling = np.array([float(row["fooling_rate"]) for row in groups])
    quality = np.array([
        float(row["pass_count"]) / int(row.get("num_rollouts", 4)) for row in groups
    ])
    similarity = np.array([float(row["pairwise_similarity"]) for row in groups])
    mixed = np.array([len(set(str(row["label_pattern"]))) > 1 for row in groups], dtype=float)
    window = 25

    fig, axes = plt.subplots(4, 1, figsize=(7.2, 9.4), sharex=True, constrained_layout=True)
    axes[0].plot(x, reward, color=COLORS["reward"], alpha=0.16, linewidth=0.7)
    axes[0].plot(x, rolling_mean(reward, window), color=COLORS["reward"], linewidth=1.8,
                 label=f"mean log $p_{{own}}$ ({window}-group average)")
    axes[0].set_ylabel("mean log $p_{own}$")
    axes[0].legend(frameon=False, loc="best")

    axes[1].plot(x, rolling_mean(quality, window), color=COLORS["quality"], linewidth=1.8,
                 label="unit-test pass rate")
    axes[1].plot(x, rolling_mean(fooling, window), color=COLORS["fooling"], linewidth=1.8,
                 label="emitted-label fooling")
    axes[1].set_ylabel("rolling rate")
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False, ncol=2, loc="best")

    axes[2].plot(x, rolling_mean(spread, window), color=COLORS["incorrect"], linewidth=1.8,
                 label="within-group reward std")
    axes[2].set_ylabel("reward std")
    axes[2].legend(frameon=False, loc="best")

    axes[3].plot(x, rolling_mean(similarity, window), color=COLORS["correct"], linewidth=1.8,
                 label="pairwise similarity")
    axes[3].plot(x, rolling_mean(mixed, window), color=COLORS["neutral"], linewidth=1.4,
                 linestyle="--", label="mixed-label groups")
    axes[3].set_ylabel("rollout diversity")
    axes[3].set_xlabel("training group order")
    axes[3].set_ylim(0, 1)
    axes[3].legend(frameon=False, ncol=2, loc="best")
    style_axes(axes)
    fig.suptitle("Training-rollout diagnostics", fontweight="bold")
    return save_figure(fig, figures / "pilot_training_diagnostics")


def plot_reward_distributions(run_dir: Path, figures: Path) -> list[Path]:
    summary = json.loads((run_dir / "evaluation_summary.json").read_text())
    validation_steps = [int(state["step"]) for state in summary["validation"]]
    selected_step = int(summary.get("selection", {}).get("selected_step", validation_steps[-1]))
    steps = list(dict.fromkeys([validation_steps[0], selected_step, validation_steps[-1]]))
    if len(steps) < 3 and len(validation_steps) > 2:
        middle = validation_steps[len(validation_steps) // 2]
        steps = list(dict.fromkeys([validation_steps[0], middle, validation_steps[-1]]))
    data: dict[int, list[dict[str, Any]]] = {
        step: read_jsonl(run_dir / "evaluations" / "validation" / f"step_{step:04d}" / "rows.jsonl")
        for step in steps
    }
    palette = [COLORS["quality"], COLORS["reward"], COLORS["fooling"]][: len(steps)]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5), constrained_layout=True)
    cdf_grid = np.linspace(0, 1, 401)
    for step, color in zip(steps, palette, strict=True):
        values = np.sort([pown_value(row) for row in data[step]])
        axes[0].plot(
            cdf_grid,
            reflected_smooth_cdf(values, cdf_grid),
            color=color,
            linewidth=1.8,
            label=f"step {step}",
        )
    axes[0].set_xlabel("$p_{own}$")
    axes[0].set_ylabel("smoothed CDF")
    axes[0].legend(frameon=False, loc="best")

    positions = []
    labels = []
    violin_values = []
    for index, step in enumerate(steps):
        for offset, passed in [(-0.16, True), (0.16, False)]:
            values = [
                pown_value(row)
                for row in data[step]
                if bool(row["unit_tests"]["passed"]) == passed
            ]
            if values:
                positions.append(index + offset)
                labels.append((step, passed))
                violin_values.append(values)
    parts = axes[1].violinplot(violin_values, positions=positions, widths=0.27,
                              showmeans=True, showextrema=False)
    for body, (_, passed) in zip(parts["bodies"], labels, strict=True):
        body.set_facecolor(COLORS["correct"] if passed else COLORS["incorrect"])
        body.set_edgecolor("none")
        body.set_alpha(0.45)
    parts["cmeans"].set_color(COLORS["neutral"])
    for position, (_, passed), values in zip(positions, labels, violin_values, strict=True):
        rng = np.random.default_rng(10_000 + int(position * 100) + int(passed))
        jitter = rng.uniform(-0.075, 0.075, size=len(values))
        axes[1].scatter(
            np.asarray(position) + jitter,
            values,
            s=12,
            color=COLORS["correct"] if passed else COLORS["incorrect"],
            alpha=0.34,
            linewidths=0,
            zorder=3,
        )
    axes[1].set_xticks(range(len(steps)), [str(step) for step in steps])
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("$p_{own}$ by correctness")
    axes[1].plot([], [], color=COLORS["correct"], linewidth=6, alpha=0.45, label="correct")
    axes[1].plot([], [], color=COLORS["incorrect"], linewidth=6, alpha=0.45, label="incorrect")
    axes[1].legend(frameon=False, loc="best")

    bins = np.linspace(-8, 8, 41)
    for step, color in zip(steps, palette, strict=True):
        deltas = [float(row["delta"]) for row in data[step]]
        axes[2].hist(deltas, bins=bins, density=True, histtype="step", linewidth=1.7,
                     color=color, label=f"step {step}")
    axes[2].axvline(0, color=COLORS["neutral"], linewidth=1)
    axes[2].set_xlabel(r"$\log p(1)-\log p(0)$")
    axes[2].set_ylabel("density")
    axes[2].legend(frameon=False, loc="best")
    style_axes(axes)
    fig.suptitle("Validation reward distributions", fontweight="bold")
    return save_figure(fig, figures / "pilot_reward_distributions")


def plot_optimization_diagnostics(run_dir: Path, figures: Path) -> list[Path]:
    history = read_jsonl(run_dir / "trainer_log_history.jsonl")
    rows = [row for row in history if "grad_norm" in row]
    steps = np.arange(1, len(rows) + 1)
    metrics = [
        ("loss", "GRPO loss", COLORS["quality"]),
        ("grad_norm", "gradient norm", COLORS["reward"]),
        ("entropy", "policy entropy", COLORS["correct"]),
        ("clip_ratio/region_mean", "clipped-token ratio", COLORS["fooling"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.5), sharex=True, constrained_layout=True)
    for axis, (key, label, color) in zip(axes.flat, metrics, strict=True):
        values = np.array([float(row.get(key, 0)) for row in rows])
        axis.plot(steps, values, color=color, alpha=0.22, linewidth=0.7)
        axis.plot(steps, rolling_mean(values, 25), color=color, linewidth=1.7)
        axis.set_ylabel(label)
        axis.set_xlabel("optimizer step")
    style_axes(axes.flat)
    fig.suptitle("Optimisation diagnostics", fontweight="bold")
    return save_figure(fig, figures / "pilot_optimization_diagnostics")


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        result[index] = np.mean(values[max(0, index - window + 1) : index + 1])
    return result


def pown_value(row: dict[str, Any]) -> float:
    """Read p_own, reconstructing it from legacy binary log-probability logs."""
    if "p_own" in row:
        return float(row["p_own"])
    delta = float(row.get("delta", float(row["logp_1"]) - float(row["logp_0"])))
    return 1.0 / (1.0 + math.exp(-np.clip(delta, -60, 60)))


def reflected_smooth_cdf(values: np.ndarray | list[float], grid: np.ndarray) -> np.ndarray:
    """KDE-smoothed CDF on [0, 1] with reflection at both boundaries."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.zeros_like(grid)
    scale = min(
        np.std(values, ddof=1) if len(values) > 1 else 0.08,
        (np.percentile(values, 75) - np.percentile(values, 25)) / 1.349 if len(values) > 1 else 0.08,
    )
    bandwidth = float(np.clip(0.9 * max(scale, 0.04) * len(values) ** (-0.2), 0.035, 0.12))
    centered = (grid[:, None] - values[None, :]) / bandwidth
    lower_reflection = (grid[:, None] + values[None, :]) / bandwidth
    upper_reflection = (grid[:, None] - (2 - values[None, :])) / bandwidth
    density = np.mean(
        np.exp(-0.5 * centered**2)
        + np.exp(-0.5 * lower_reflection**2)
        + np.exp(-0.5 * upper_reflection**2),
        axis=1,
    )
    cdf = np.cumsum(density)
    cdf -= cdf[0]
    return cdf / cdf[-1] if cdf[-1] else cdf


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def style_axes(axes) -> None:
    for axis in axes:
        axis.grid(axis="y", color="#CFD8DC", linewidth=0.6, alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)


def save_figure(fig, stem: Path) -> list[Path]:
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [pdf, png]
