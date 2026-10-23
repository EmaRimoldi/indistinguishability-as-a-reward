from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from iar_phase0.env import load_env_file

from .data import read_jsonl, sha256_json, write_json, write_jsonl
from .evaluate_short import (
    adapter_for_step,
    attach_rewards,
    generate,
    paired,
    summarize,
)
from .train import resolve, select_balanced


def evaluate_state(
    *,
    config: dict[str, Any],
    root: Path,
    run_dir: Path,
    checkpoint_root: Path,
    examples: list[dict[str, Any]],
    split_name: str,
    step: int,
    final_step: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state_dir = run_dir / "evaluations" / split_name / f"step_{step:04d}"
    state_dir.mkdir(parents=True, exist_ok=False)
    rows = generate(
        config,
        root,
        examples,
        adapter_for_step(checkpoint_root, step, final_step),
        step,
    )
    attach_rewards(config, rows, state_dir / "reward_calls.jsonl")
    write_jsonl(state_dir / "rows.jsonl", rows)
    summary = {"split": split_name, "step": step, **summarize(rows)}
    write_json(state_dir / "summary.json", summary)
    return rows, summary


def select_checkpoint(
    baseline: list[dict[str, Any]],
    candidates: dict[int, list[dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    comparisons = []
    maximum_drop = float(config["evaluation"]["maximum_accuracy_drop"])
    selection_rule = str(
        config["evaluation"].get(
            "select_on", "validation_paired_p_own_lcb_subject_to_accuracy"
        )
    )
    for step, rows in sorted(candidates.items()):
        comparison = {
            "step": step,
            **paired(
                baseline,
                rows,
                z=float(config["evaluation"]["paired_lcb_z"]),
            ),
        }
        comparison["quality_feasible"] = float(comparison["accuracy_delta"]) >= -maximum_drop
        comparisons.append(comparison)
    if selection_rule == "validation_paired_p_own_lcb_audit_accuracy_only":
        feasible = [
            row
            for row in comparisons
            if float(row["delta_p_own_lcb_95_one_sided"]) > 0
        ]
        objective = (
            "maximum paired one-sided 95% LCB of validation delta p_own; "
            "task accuracy is audit-only"
        )
    elif selection_rule == "validation_paired_p_own_lcb_subject_to_accuracy":
        feasible = [
            row
            for row in comparisons
            if row["quality_feasible"] and float(row["delta_p_own_lcb_95_one_sided"]) > 0
        ]
        objective = "maximum quality-feasible paired one-sided 95% LCB of validation delta p_own"
    elif selection_rule == "validation_task_accuracy":
        winner = max(
            comparisons,
            key=lambda row: (float(row["accuracy_delta"]), -int(row["step"])),
            default=None,
        )
        if winner is None or float(winner["accuracy_delta"]) <= 0:
            winner = None
        return {
            "objective": "maximum strict validation task accuracy; earliest checkpoint breaks ties",
            "selection_rule": selection_rule,
            "task_accuracy_role": "selection_objective",
            "selected_step": int(winner["step"]) if winner else 0,
            "comparisons": comparisons,
        }
    else:
        raise ValueError(f"Unsupported checkpoint selection rule: {selection_rule}")
    winner = max(feasible, key=lambda row: float(row["delta_p_own_lcb_95_one_sided"]), default=None)
    return {
        "objective": objective,
        "selection_rule": selection_rule,
        "task_accuracy_role": (
            "audit_only"
            if selection_rule == "validation_paired_p_own_lcb_audit_accuracy_only"
            else "selection_constraint"
        ),
        "selected_step": int(winner["step"]) if winner else 0,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CQA full training on validation and held-out test")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Evaluate and select on validation without opening the held-out test file.",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    validation_only = bool(
        args.validation_only or config.get("evaluation", {}).get("validation_only", False)
    )
    root = Path(config["project"]["root"]).resolve()
    run_dir = resolve(root, config["paths"]["experiments"]) / args.run_id
    training = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if training.get("status") != "complete":
        raise RuntimeError("Full training did not complete successfully")
    checkpoint_root = resolve(root, config["paths"]["checkpoints"]) / args.run_id
    split_dir = resolve(root, config["paths"]["split_dir"])
    validation, test = load_requested_splits(
        split_dir, config, validation_only=validation_only
    )
    evaluation_manifest = {
        "validation": {
            "n": len(validation),
            "ids_sha256": sha256_json([row["id"] for row in validation]),
            "ordered_ids": [row["id"] for row in validation],
        },
        "official_test_used": not validation_only,
    }
    if test is not None:
        evaluation_manifest["test"] = {
            "n": len(test),
            "ids_sha256": sha256_json([row["id"] for row in test]),
            "ordered_ids": [row["id"] for row in test],
            "source": "complete official CommonsenseQA validation split",
        }
    write_json(
        run_dir / "parameters" / "evaluation_split_manifest.json",
        evaluation_manifest,
    )
    load_env_file(resolve(root, config["paths"]["openrouter_env"]))
    checkpoints = [int(step) for step in config["evaluation"]["checkpoints"]]
    final_step = checkpoints[-1]
    validation_rows: dict[int, list[dict[str, Any]]] = {}
    validation_summaries = []
    for step in checkpoints:
        rows, summary = evaluate_state(
            config=config,
            root=root,
            run_dir=run_dir,
            checkpoint_root=checkpoint_root,
            examples=validation,
            split_name="validation",
            step=step,
            final_step=final_step,
        )
        validation_rows[step] = rows
        validation_summaries.append(summary)
    selection = select_checkpoint(
        validation_rows[0],
        {step: validation_rows[step] for step in checkpoints[1:]},
        config,
    )
    selected_step = int(selection["selected_step"])
    test_steps = [] if validation_only else ([0] if selected_step == 0 else [0, selected_step])
    test_rows: dict[int, list[dict[str, Any]]] = {}
    test_summaries = []
    if test_steps:
        assert test is not None
    for step in test_steps:
        rows, summary = evaluate_state(
            config=config,
            root=root,
            run_dir=run_dir,
            checkpoint_root=checkpoint_root,
            examples=test,
            split_name="test",
            step=step,
            final_step=final_step,
        )
        test_rows[step] = rows
        test_summaries.append(summary)
    test_comparison = None
    if test_steps and selected_step:
        test_comparison = {
            "step": selected_step,
            **paired(
                test_rows[0],
                test_rows[selected_step],
                z=float(config["evaluation"]["paired_lcb_z"]),
            ),
        }
    result = {
        "run_id": args.run_id,
        "status": "complete",
        "validation_only": validation_only,
        "validation": validation_summaries,
        "selection": selection,
        "test": test_summaries,
        "test_comparison": test_comparison,
    }
    write_json(run_dir / "evaluation_summary.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def load_requested_splits(
    split_dir: Path,
    config: dict[str, Any],
    *,
    validation_only: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Load the held-out test only after the experimental gate authorizes it."""
    validation_source = read_jsonl(split_dir / "validation.jsonl")
    requested = int(config["evaluation"]["validation_count"])
    validation = (
        validation_source
        if requested == len(validation_source)
        else select_balanced(
            validation_source,
            count=requested,
            seed=int(config["project"]["seed"]) + 1_000_003,
        )
    )
    if validation_only:
        return validation, None
    test = read_jsonl(split_dir / "test.jsonl")
    if len(test) != int(config["evaluation"]["test_count"]):
        raise ValueError("Held-out test row count differs from the frozen config")
    return validation, test


if __name__ == "__main__":
    main()
