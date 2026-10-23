"""Comparable Coder versus non-Coder diagnostics for completed GRPO runs.

The common *full score* is the untransformed probability ``p_own``. Native
training rewards are overlaid whenever the runs optimize the same scalar, and
are deliberately kept on separate axes only when their reward transforms
differ. Validation pass@1 is displayed as a performance audit, never as a
selection criterion in this report.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .plot_pilot import read_jsonl, rolling_mean, save_figure, style_axes, wilson_interval


MODEL_COLORS = {"non_coder": "#D97706", "coder": "#3976B8"}
AUDIT_COLOR = "#546E7A"
WINDOW = 25


@dataclass(frozen=True)
class ModelRun:
    key: str
    label: str
    run_dir: Path
    summary: dict[str, Any]
    native_reward_name: str

    @property
    def validation(self) -> list[dict[str, Any]]:
        return list(self.summary["validation"])

    @property
    def training_groups(self) -> list[dict[str, Any]]:
        return read_jsonl(self.run_dir / "training_groups.jsonl")

    @property
    def trainer_history(self) -> list[dict[str, Any]]:
        return [
            row
            for row in read_jsonl(self.run_dir / "trainer_log_history.jsonl")
            if "grad_norm" in row
        ]


def create_comparison_plots(
    *,
    non_coder_run_dir: Path,
    coder_run_dir: Path,
    output_dir: Path,
) -> list[Path]:
    """Create all four overlaid figures and a machine-readable protocol manifest."""
    runs = [
        load_run("non_coder", "Non-Coder", non_coder_run_dir),
        load_run("coder", "Coder", coder_run_dir),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths.extend(plot_validation_criteria_and_audit(runs, output_dir))
    paths.extend(plot_training_diagnostics(runs, output_dir))
    paths.extend(plot_reward_distributions(runs, output_dir))
    paths.extend(plot_optimisation_diagnostics(runs, output_dir))
    manifest = comparison_manifest(runs, paths)
    manifest_path = output_dir / "comparison_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return [*paths, manifest_path]


def load_run(key: str, label: str, run_dir: Path) -> ModelRun:
    summary_path = run_dir / "evaluation_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing completed evaluation summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise RuntimeError(f"Evaluation is not complete: {run_dir}")
    config_path = run_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    value = str((config.get("reward") or {}).get("value", "p_own"))
    return ModelRun(
        key=key,
        label=label,
        run_dir=run_dir,
        summary=summary,
        native_reward_name=value,
    )


def plot_validation_criteria_and_audit(runs: list[ModelRun], output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 9.0), sharex=True, constrained_layout=True)
    separate_native_scales = uses_separate_native_scales(runs)
    native_axes = [axes[1], axes[1].twinx()] if separate_native_scales else [axes[1]] * len(runs)
    for index, run in enumerate(runs):
        states = run.validation
        steps = np.array([int(state["step"]) for state in states])
        color = MODEL_COLORS[run.key]
        full_score = np.array([summary_full_score(state) for state in states])
        sem = np.array([summary_full_score_sem(state) for state in states])
        fooling = np.array([float(state.get("fooling_rate", state.get("sampled_fooling_rate", np.nan))) for state in states])
        quality = np.array([float(state["pass@1"]) for state in states])

        axes[0].plot(steps, full_score, "o-", color=color, linewidth=1.9, label=f"{run.label}: full score")
        axes[0].fill_between(steps, np.maximum(0, full_score - 1.96 * sem),
                             np.minimum(1, full_score + 1.96 * sem), color=color, alpha=0.12, linewidth=0)
        axes[0].plot(steps, fooling, "--", color=color, alpha=0.8,
                     label=f"{run.label}: emitted-label fooling")

        native_axis = native_axes[index]
        native_axis.plot(steps, [float(state["reward_mean"]) for state in states], "o-",
                         color=color, linewidth=1.9, label=f"{run.label}: native reward")
        if separate_native_scales:
            native_axis.set_ylabel(f"{run.label} native reward ({run.native_reward_name})", color=color)
            native_axis.tick_params(axis="y", colors=color)

        lo, hi = zip(*(wilson_interval(int(state["passed"]), int(state["n"])) for state in states), strict=True)
        axes[2].plot(steps, quality, "o-", color=color, linewidth=1.9, label=run.label)
        axes[2].fill_between(steps, lo, hi, color=color, alpha=0.10, linewidth=0)

    axes[0].set_ylabel("full score ($p_{own}$)")
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False, ncol=2, loc="best")
    if separate_native_scales:
        axes[1].set_ylabel("native reward\n(separate scales)")
        merge_legends(axes[1], native_axes[1], loc="best")
    else:
        axes[1].set_ylabel(f"native reward ({runs[0].native_reward_name})")
        axes[1].legend(frameon=False, ncol=2, loc="best")
    axes[2].set_ylabel("validation pass@1\n(audit only)")
    axes[2].set_xlabel("processed training groups")
    axes[2].set_ylim(0, 1)
    axes[2].legend(frameon=False, ncol=2, loc="best")
    style_axes(axes)
    fig.suptitle("Coder vs non-Coder: primary criteria and performance audit", fontweight="bold")
    return save_figure(fig, output_dir / "coder_vs_non_coder_validation_criteria_and_audit")


def plot_training_diagnostics(runs: list[ModelRun], output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(4, 1, figsize=(8.4, 10.7), sharex=True, constrained_layout=True)
    separate_native_scales = uses_separate_native_scales(runs)
    native_axes = [axes[0], axes[0].twinx()] if separate_native_scales else [axes[0]] * len(runs)
    for index, run in enumerate(runs):
        groups = run.training_groups
        x = np.array([int(row["group_index"]) for row in groups])
        color = MODEL_COLORS[run.key]
        reward = np.array([float(row["reward_mean"]) for row in groups])
        full_score = training_full_scores(run, groups)
        fooling = np.array([float(row["fooling_rate"]) for row in groups])
        quality = np.array([float(row["pass_count"]) / rollout_count(row) for row in groups])
        spread = np.array([float(row["reward_std"]) for row in groups])
        similarity = np.array([float(row["pairwise_similarity"]) for row in groups])

        native_axis = native_axes[index]
        native_axis.plot(x, rolling_mean(reward, WINDOW), color=color, linewidth=1.8,
                         label=f"{run.label}: native reward")
        if separate_native_scales:
            native_axis.set_ylabel(f"{run.label} reward ({run.native_reward_name})", color=color)
            native_axis.tick_params(axis="y", colors=color)
        axes[1].plot(x, rolling_mean(full_score, WINDOW), color=color, linewidth=1.8, label=run.label)
        axes[2].plot(x, rolling_mean(fooling, WINDOW), color=color, linewidth=1.8,
                     label=f"{run.label}: fooling")
        axes[2].plot(x, rolling_mean(quality, WINDOW), color=color, linewidth=1.4,
                     linestyle="--", label=f"{run.label}: train pass rate")
        axes[3].plot(x, rolling_mean(spread, WINDOW), color=color, linewidth=1.8,
                     label=f"{run.label}: reward std")
        axes[3].plot(x, rolling_mean(similarity, WINDOW), color=color, linewidth=1.4,
                     linestyle="--", label=f"{run.label}: similarity")

    if separate_native_scales:
        axes[0].set_ylabel("native reward\n(separate scales)")
        merge_legends(axes[0], native_axes[1], loc="best")
    else:
        axes[0].set_ylabel(f"native reward ({runs[0].native_reward_name})")
        axes[0].legend(frameon=False, ncol=2, loc="best")
    axes[1].set_ylabel("full score ($p_{own}$)")
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False, ncol=2, loc="best")
    axes[2].set_ylabel("rollout diagnostics\n(not selection)")
    axes[2].set_ylim(0, 1)
    axes[2].legend(frameon=False, ncol=2, loc="best")
    axes[3].set_ylabel("spread / diversity")
    axes[3].set_xlabel("training group order")
    axes[3].legend(frameon=False, ncol=2, loc="best")
    style_axes(axes)
    fig.suptitle("Coder vs non-Coder: training diagnostics", fontweight="bold")
    return save_figure(fig, output_dir / "coder_vs_non_coder_training_diagnostics")


def plot_reward_distributions(runs: list[ModelRun], output_dir: Path) -> list[Path]:
    """Plot smooth densities of the observed validation full-score samples.

    The earlier version mixed an empirical-CDF step plot, violins, and a binned
    histogram.  Those are useful summaries, but obscure the distributional
    shift that is the point of this panel.  Here each line is a reflected
    Gaussian KDE fitted directly to the 90 observed validation ``p_own``
    values at the baseline and final checkpoints.  Reflection avoids leaking
    density past the natural [0, 1] support of a probability.
    """
    fig, axes = plt.subplots(1, len(runs), figsize=(9.6, 3.8), sharex=True, sharey=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    grid = np.linspace(0.0, 1.0, 401)
    for axis, run in zip(axes, runs, strict=True):
        states = sorted(int(state["step"]) for state in run.validation)
        color = MODEL_COLORS[run.key]
        for step, linestyle, alpha in [(states[0], "--", 0.18), (states[-1], "-", 0.26)]:
            rows = validation_rows(run, step)
            values = np.asarray([row_full_score(row) for row in rows], dtype=float)
            density = reflected_gaussian_kde(values, grid)
            mean = float(np.mean(values))
            axis.plot(grid, density, color=color, linestyle=linestyle, linewidth=2.1,
                      label=f"step {step}  (mean={mean:.3f})")
            axis.fill_between(grid, density, color=color, alpha=alpha, linewidth=0)
            axis.axvline(mean, color=color, linestyle=linestyle, linewidth=1.0, alpha=0.75)
        axis.set_title(run.label, fontweight="bold")
        axis.set_xlabel("validation full score ($p_{own}$)")
        axis.set_xlim(0, 1)
        axis.legend(frameon=False, fontsize=8, loc="upper center")

    axes[0].set_ylabel("smooth density")
    style_axes(axes)
    fig.suptitle("Coder vs non-Coder: validation full-score distribution shift", fontweight="bold")
    return save_figure(fig, output_dir / "coder_vs_non_coder_reward_distributions")


def reflected_gaussian_kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Smooth samples on a closed probability interval without histogram bins.

    Each observed value contributes a Gaussian kernel plus reflections at 0
    and 1.  The result is a continuous estimate on [0, 1], while the plotted
    curves still derive solely from the raw evaluation rows.
    """
    if values.ndim != 1 or values.size == 0:
        raise ValueError("KDE requires at least one one-dimensional value")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Full-score KDE expects values in [0, 1]")
    scale = min(float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                float(np.subtract(*np.percentile(values, [75, 25])) / 1.34))
    bandwidth = float(np.clip(0.9 * scale * values.size ** (-0.2), 0.035, 0.12))
    reflected = np.concatenate((values, -values, 2.0 - values))
    z = (grid[:, None] - reflected[None, :]) / bandwidth
    kernels = np.exp(-0.5 * z * z) / (math.sqrt(2.0 * math.pi) * bandwidth)
    return np.sum(kernels, axis=1) / values.size


def plot_optimisation_diagnostics(runs: list[ModelRun], output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.9), sharex=True, constrained_layout=True)
    metrics = [
        ("loss", "GRPO loss"),
        ("grad_norm", "gradient norm"),
        ("entropy", "policy entropy"),
        ("clip_ratio/region_mean", "clipped-token ratio"),
    ]
    for axis, (key, label) in zip(axes.flat, metrics, strict=True):
        for run in runs:
            rows = run.trainer_history
            x = np.array([float(row.get("step", index + 1)) for index, row in enumerate(rows)])
            values = np.array([float(row.get(key, 0.0)) for row in rows])
            color = MODEL_COLORS[run.key]
            axis.plot(x, values, color=color, alpha=0.14, linewidth=0.65)
            axis.plot(x, rolling_mean(values, WINDOW), color=color, linewidth=1.7, label=run.label)
        axis.set_ylabel(label)
        axis.set_xlabel("optimizer step")
        axis.legend(frameon=False, loc="best")
    style_axes(axes.flat)
    fig.suptitle("Coder vs non-Coder: optimisation diagnostics", fontweight="bold")
    return save_figure(fig, output_dir / "coder_vs_non_coder_optimization_diagnostics")


def training_full_scores(run: ModelRun, groups: list[dict[str, Any]]) -> np.ndarray:
    """Return common p_own scores per group, reconstructing them when needed."""
    path = run.run_dir / "reward_calls.jsonl"
    if not path.is_file():
        return np.array([float(row["reward_mean"]) for row in groups])
    calls = sorted(
        (row for row in read_jsonl(path) if row.get("status") == "ok"),
        key=lambda row: int(row["call_id"]),
    )
    if len(calls) != len(groups) * rollout_count(groups[0]):
        return np.array([float(row["reward_mean"]) for row in groups])
    group_size = rollout_count(groups[0])
    return np.array([
        np.mean([row_full_score(row) for row in calls[start : start + group_size]])
        for start in range(0, len(calls), group_size)
    ])


def validation_rows(run: ModelRun, step: int) -> list[dict[str, Any]]:
    return read_jsonl(run.run_dir / "evaluations" / "validation" / f"step_{step:04d}" / "rows.jsonl")


def summary_full_score(summary: dict[str, Any]) -> float:
    value = summary.get("p_own_mean", summary.get("reward_mean"))
    if value is None:
        raise ValueError("Validation summary has neither p_own_mean nor reward_mean")
    return float(value)


def summary_full_score_sem(summary: dict[str, Any]) -> float:
    value = summary.get("p_own_std", summary.get("reward_std", 0.0))
    return float(value) / math.sqrt(int(summary["n"]))


def row_full_score(row: dict[str, Any]) -> float:
    value = row.get("p_own", row.get("reward"))
    if value is None:
        raise ValueError("Evaluation row has neither p_own nor reward")
    return float(value)


def rollout_count(row: dict[str, Any]) -> int:
    return int(row.get("num_rollouts", 4))


def merge_legends(left, right, *, loc: str) -> None:
    handles_a, labels_a = left.get_legend_handles_labels()
    handles_b, labels_b = right.get_legend_handles_labels()
    left.legend(handles_a + handles_b, labels_a + labels_b, frameon=False, loc=loc)


def uses_separate_native_scales(runs: list[ModelRun]) -> bool:
    """Native reward values are directly comparable exactly when transforms match."""
    return len({run.native_reward_name for run in runs}) != 1


def comparison_manifest(runs: list[ModelRun], paths: list[Path]) -> dict[str, Any]:
    return {
        "status": "complete",
        "protocol": {
            "primary_criteria": [
                "native_training_reward",
                "full_self_attribution_score_p_own",
            ],
            "performance_audit": "validation_pass_at_1",
            "full_score_definition": "p_own when logged; the historical p_own reward otherwise",
            "native_reward_caveat": native_reward_caveat(runs),
            "selection_caveat": "This comparative report does not make a post-hoc checkpoint selection.",
        },
        "runs": [
            {
                "key": run.key,
                "label": run.label,
                "run_dir": str(run.run_dir),
                "native_reward": run.native_reward_name,
                "validation_steps": [int(state["step"]) for state in run.validation],
            }
            for run in runs
        ],
        "plots": [str(path) for path in paths],
    }


def native_reward_caveat(runs: list[ModelRun]) -> str:
    names = {run.native_reward_name for run in runs}
    if len(names) == 1:
        name = next(iter(names))
        return f"Both runs optimize {name}; native rewards share one directly comparable axis."
    return (
        "The runs use different native reward transforms; native rewards are plotted "
        "on separate axes and are not directly comparable."
    )
