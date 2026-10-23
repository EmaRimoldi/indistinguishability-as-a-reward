from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from .data import read_jsonl, write_json


def correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return float(np.corrcoef(np.asarray(xs), np.asarray(ys))[0, 1])


def paired_audit(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    base = {row["id"]: row for row in baseline}
    cand = {row["id"]: row for row in candidate}
    if base.keys() != cand.keys():
        raise ValueError("Evaluation IDs differ across checkpoints")
    identifiers = list(base)
    transitions = {
        f"{int(before)}_to_{int(after)}": sum(
            bool(base[key]["score"]["correct"]) == before
            and bool(cand[key]["score"]["correct"]) == after
            for key in identifiers
        )
        for before in (False, True)
        for after in (False, True)
    }
    initially_valid = [key for key in identifiers if base[key]["score"]["valid"]]
    initially_invalid = [key for key in identifiers if not base[key]["score"]["valid"]]
    token_deltas = [float(cand[key]["completion_tokens"] - base[key]["completion_tokens"]) for key in identifiers]
    pown_deltas = [float(cand[key]["p_own"] - base[key]["p_own"]) for key in identifiers]
    return {
        "correctness_transitions": transitions,
        "initially_invalid": {
            "n": len(initially_invalid),
            "candidate_correct": sum(cand[key]["score"]["correct"] for key in initially_invalid),
            "mean_delta_p_own": statistics.mean(cand[key]["p_own"] - base[key]["p_own"] for key in initially_invalid),
        },
        "initially_valid": {
            "n": len(initially_valid),
            "baseline_accuracy": statistics.mean(base[key]["score"]["correct"] for key in initially_valid),
            "candidate_accuracy": statistics.mean(cand[key]["score"]["correct"] for key in initially_valid),
            "net_correct_change": sum(
                int(cand[key]["score"]["correct"]) - int(base[key]["score"]["correct"])
                for key in initially_valid
            ),
            "mean_delta_p_own": statistics.mean(cand[key]["p_own"] - base[key]["p_own"] for key in initially_valid),
        },
        "mean_completion_token_delta": statistics.mean(token_deltas),
        "correlation_delta_tokens_delta_p_own": correlation(token_deltas, pown_deltas),
    }


def make_plots(run_dir: Path, evaluation: dict[str, Any]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    history = read_jsonl(run_dir / "trainer_log_history.jsonl")
    rows = [row for row in history if "grad_norm" in row]
    steps = [int(row["step"]) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(steps, [float(row["loss"]) for row in rows], color="#0072B2", linewidth=1)
    axes[0, 0].axhline(0, color="#777777", linewidth=0.7)
    axes[0, 0].set(title="GRPO loss", xlabel="Step", ylabel="Loss")
    axes[0, 1].plot(steps, [float(row["grad_norm"]) for row in rows], color="#D55E00", linewidth=1)
    axes[0, 1].set(title="Gradient norm", xlabel="Step", ylabel="Norm")
    axes[1, 0].plot(steps, [float(row["kl"]) for row in rows], color="#009E73", linewidth=1)
    axes[1, 0].set(title="KL from reference", xlabel="Step", ylabel="KL")
    axes[1, 1].plot(steps, [float(row["entropy"]) for row in rows], color="#CC79A7", linewidth=1)
    axes[1, 1].set(title="Policy entropy", xlabel="Step", ylabel="Entropy")
    fig.tight_layout()
    files: list[str] = []
    for suffix in ("png", "pdf"):
        path = plot_dir / f"short_training_diagnostics.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        files.append(str(path.relative_to(run_dir)))
    plt.close(fig)

    validation = evaluation["validation"]
    checkpoints = [int(row["step"]) for row in validation]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].plot(checkpoints, [row["mean_p_own"] for row in validation], marker="o", color="#0072B2")
    axes[0].set(title=r"Indistinguishability", xlabel="Step", ylabel=r"Mean $p_{\rm own}$")
    axes[1].plot(checkpoints, [row["accuracy"] for row in validation], marker="o", color="#009E73")
    axes[1].set(title="External task quality", xlabel="Step", ylabel="Answer accuracy")
    axes[2].plot(checkpoints, [row["sampled_fooling"] for row in validation], marker="o", label="Sampled", color="#D55E00")
    axes[2].plot(checkpoints, [row["argmax_fooling"] for row in validation], marker="o", label="Argmax", color="#CC79A7")
    axes[2].set(title="Fooling rates", xlabel="Step", ylabel="Rate")
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(checkpoints)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = plot_dir / f"short_validation_learning_curve.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        files.append(str(path.relative_to(run_dir)))
    plt.close(fig)
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    evaluation = json.loads((run_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    base = read_jsonl(run_dir / "evaluations" / "step_0000" / "rows.jsonl")
    audits = {
        str(step): paired_audit(
            base,
            read_jsonl(run_dir / "evaluations" / f"step_{step:04d}" / "rows.jsonl"),
        )
        for step in (50, 100)
    }
    result = {
        "run_id": evaluation["run_id"],
        "paired_format_and_length_audit": audits,
        "interpretation": {
            "accuracy_gain_is_not_entirely_format_independent": True,
            "reward_uses_neither_gold_answer_nor_format_validity": True,
        },
    }
    result["plots"] = make_plots(run_dir, evaluation)
    write_json(run_dir / "post_training_audit.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
