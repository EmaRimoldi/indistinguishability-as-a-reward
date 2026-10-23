from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from iar_phase0.config import load_config
from iar_phase0.data import load_mbpp
from iar_phase0.env import load_env_file

from .config import load_grpo_config
from .evaluate_pilot import adapter_for_step, evaluate_state, manifest_for, select_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one short GRPO HPO trial")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_grpo_config(args.config)
    run_dir = config.resolve(config.raw["paths"]["experiments"]) / args.run_id
    checkpoint_root = config.resolve(config.raw["paths"]["checkpoints"]) / args.run_id
    training_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if training_summary.get("status") != "complete":
        raise RuntimeError(f"Training run is not complete: {run_dir}")

    load_env_file(config.resolve(config.raw["paths"]["openrouter_env"]))
    phase0 = load_config(config.resolve(config.raw["dataset"]["phase0_config"]))
    mbpp = load_mbpp(phase0)
    evaluation = config.raw["evaluation"]
    validation_spec = evaluation["validation"]
    validation = select_examples(mbpp, validation_spec)
    checkpoints = [int(step) for step in evaluation["checkpoints"]]

    states: list[dict[str, Any]] = []
    rows_by_step: dict[int, list[dict[str, Any]]] = {}
    for step in checkpoints:
        adapter = adapter_for_step(checkpoint_root, step, checkpoints[-1])
        state_name = f"step_{step:04d}"
        rows, summary = evaluate_state(
            config=config,
            examples=validation,
            adapter_dir=adapter,
            state_name=state_name,
            split_name="validation",
            num_generations=int(validation_spec["num_generations"]),
            temperature=float(validation_spec["temperature"]),
            top_p=float(validation_spec["top_p"]),
            output_dir=run_dir / "evaluations" / "validation" / state_name,
            seed=int(config.raw["project"]["seed"]),
        )
        summary["step"] = step
        states.append(summary)
        rows_by_step[step] = rows

    baseline_step = checkpoints[0]
    baseline = rows_by_step[baseline_step]
    comparisons = [
        paired_comparison(
            baseline,
            rows_by_step[step],
            step=step,
            z=float(evaluation["paired_lcb_z"]),
            max_pass_drop=float(evaluation["max_pass_at_1_drop"]),
        )
        for step in checkpoints[1:]
    ]
    feasible = [comparison for comparison in comparisons if comparison["quality_feasible"]]
    selected = max(feasible, key=lambda row: float(row["p_own_delta_lcb"])) if feasible else None
    objective_value = float(selected["p_own_delta_lcb"]) if selected else -1.0

    hpo = config.raw.get("hpo") or {}
    objective = {
        "schema_version": 1,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "study_name": hpo.get("study_name"),
        "trial_number": hpo.get("trial_number"),
        "seed": int(config.raw["project"]["seed"]),
        "reward": str(config.raw["reward"]["value"]),
        "objective_name": "max_feasible_paired_validation_p_own_delta_lcb",
        "objective_value": objective_value,
        "selected_step": int(selected["step"]) if selected else None,
        "quality_feasible": selected is not None,
        "max_pass_at_1_drop": float(evaluation["max_pass_at_1_drop"]),
        "comparisons": comparisons,
    }
    split_manifest = {
        "training_index_manifest": str(config.raw["dataset"].get("index_manifest")),
        "validation": manifest_for(validation),
    }
    combined = {
        "run_id": args.run_id,
        "status": "complete",
        "split_manifest": split_manifest,
        "validation": states,
        "objective": objective,
    }
    (run_dir / "hpo_evaluation_summary.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (run_dir / "objective.json").write_text(
        json.dumps(objective, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(objective, indent=2, ensure_ascii=False))


def paired_comparison(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    step: int,
    z: float,
    max_pass_drop: float,
) -> dict[str, Any]:
    baseline_by_id = {str(row["task_id"]): row for row in baseline}
    candidate_by_id = {str(row["task_id"]): row for row in candidate}
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("Validation task IDs differ between baseline and checkpoint")

    task_ids = list(baseline_by_id)
    p_deltas = [
        float(candidate_by_id[task_id]["p_own"])
        - float(baseline_by_id[task_id]["p_own"])
        for task_id in task_ids
    ]
    pass_deltas = [
        int(bool(candidate_by_id[task_id]["unit_tests"]["passed"]))
        - int(bool(baseline_by_id[task_id]["unit_tests"]["passed"]))
        for task_id in task_ids
    ]
    mean_delta = statistics.mean(p_deltas)
    standard_error = statistics.stdev(p_deltas) / math.sqrt(len(p_deltas)) if len(p_deltas) > 1 else 0.0
    pass_delta = statistics.mean(pass_deltas)
    return {
        "step": int(step),
        "n": len(task_ids),
        "p_own_delta_mean": mean_delta,
        "p_own_delta_standard_error": standard_error,
        "p_own_delta_lcb": mean_delta - z * standard_error,
        "pass_at_1_delta": pass_delta,
        "net_passed_change": sum(pass_deltas),
        "quality_feasible": pass_delta >= -max_pass_drop,
    }


if __name__ == "__main__":
    main()
