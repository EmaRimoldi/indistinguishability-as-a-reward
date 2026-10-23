from __future__ import annotations

import argparse
import json
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
    token_deltas = [float(cand[key]["completion_tokens"] - base[key]["completion_tokens"]) for key in identifiers]
    pown_deltas = [float(cand[key]["p_own"] - base[key]["p_own"]) for key in identifiers]
    return {
        "correctness_transitions": {
            f"{int(before)}_to_{int(after)}": sum(
                bool(base[key]["score"]["correct"]) == before
                and bool(cand[key]["score"]["correct"]) == after
                for key in identifiers
            )
            for before in (False, True)
            for after in (False, True)
        },
        "mean_completion_token_delta": statistics.mean(token_deltas),
        "correlation_delta_tokens_delta_p_own": correlation(token_deltas, pown_deltas),
        "baseline_truncations": sum(bool(base[key].get("hit_max_new_tokens")) for key in identifiers),
        "candidate_truncations": sum(bool(cand[key].get("hit_max_new_tokens")) for key in identifiers),
    }


def make_plots(run_dir: Path, evaluation: dict[str, Any], *, prefix: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = run_dir / "figures"
    plot_dir.mkdir(exist_ok=True)
    history = read_jsonl(run_dir / "trainer_log_history.jsonl")
    rows = [row for row in history if "grad_norm" in row]
    steps = [int(row["step"]) for row in rows]
    metric_specs = (("loss", "GRPO loss"), ("grad_norm", "Gradient norm"), ("kl", "KL from reference"), ("entropy", "Policy entropy"))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    for ax, (key, title), color in zip(axes.flat, metric_specs, colors):
        ax.plot(steps, [float(row[key]) for row in rows], color=color, linewidth=1)
        ax.set(title=title, xlabel="Step", ylabel=key.replace("_", " ").title())
        ax.grid(alpha=0.2)
    fig.tight_layout()
    files: list[str] = []
    path = plot_dir / f"{prefix}_training_diagnostics.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    files.append(str(path.relative_to(run_dir)))
    plt.close(fig)

    validation = evaluation["validation"]
    checkpoints = [int(row["step"]) for row in validation]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].plot(checkpoints, [row["mean_p_own"] for row in validation], marker="o", color="#0072B2")
    axes[0].set(title="Indistinguishability", xlabel="Step", ylabel="Mean p_own")
    axes[1].plot(checkpoints, [row["accuracy"] for row in validation], marker="o", color="#009E73")
    axes[1].set(title="External GSM8K quality", xlabel="Step", ylabel="Exact-match accuracy")
    axes[2].plot(checkpoints, [row["fooling_sampled"] for row in validation], marker="o", label="Sampled", color="#D55E00")
    axes[2].plot(checkpoints, [row["fooling_argmax"] for row in validation], marker="o", label="Argmax", color="#CC79A7")
    axes[2].set(title="Fooling rates", xlabel="Step", ylabel="Rate")
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(checkpoints)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    path = plot_dir / f"{prefix}_validation_learning_curve.png"
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
    full_validation_root = run_dir / "evaluations" / "validation"
    if full_validation_root.exists():
        validation_root = full_validation_root
        prefix = "official"
    else:
        validation_root = run_dir / "evaluations"
        prefix = "short"
    baseline = read_jsonl(validation_root / "step_0000" / "rows.jsonl")
    candidate_steps = [int(row["step"]) for row in evaluation["validation"] if int(row["step"]) != 0]
    audits = {
        str(step): paired_audit(
            baseline,
            read_jsonl(validation_root / f"step_{step:04d}" / "rows.jsonl"),
        )
        for step in candidate_steps
    }
    if "comparisons_to_step_0" in evaluation:
        comparisons = evaluation["comparisons_to_step_0"]
        feasible = [
            row
            for row in comparisons
            if row.get("quality_feasible")
            and float(row["delta_p_own_lcb_95_one_sided"]) > 0
        ]
        winner = max(
            feasible,
            key=lambda row: float(row["delta_p_own_lcb_95_one_sided"]),
            default=None,
        )
        training_decision = {
            "criterion": "positive paired one-sided 95% LCB of validation delta p_own with accuracy drop <= 0.02",
            "decision": "PROMISING" if winner else "NO-GO-OFFICIAL",
            "selected_step": int(winner["step"]) if winner else 0,
            "objective_value": (
                float(winner["delta_p_own_lcb_95_one_sided"])
                if winner
                else None
            ),
        }
    else:
        training_decision = {
            "criterion": evaluation["selection"]["objective"],
            "decision": "SELECTED" if int(evaluation["selection"]["selected_step"]) else "BASELINE-SELECTED",
            "selected_step": int(evaluation["selection"]["selected_step"]),
        }
    result = {
        "run_id": evaluation["run_id"],
        "training_decision": training_decision,
        "paired_format_length_and_correctness_audit": audits,
        "reward_uses_neither_gold_answer_nor_format_validity": True,
        "plots": make_plots(run_dir, evaluation, prefix=prefix),
    }
    write_json(run_dir / "post_training_audit.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
