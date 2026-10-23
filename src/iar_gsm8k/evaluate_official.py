from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from iar_phase0.env import load_env_file

from .data import read_jsonl, sha256_json, write_json, write_jsonl
from .evaluate_short import adapter_for_step, attach_rewards, generate, paired, summarize
from .train import frozen_subset, resolve


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
    state_dir.mkdir(parents=True, exist_ok=True)
    rows_path = state_dir / "rows.jsonl"
    summary_path = state_dir / "summary.json"
    if rows_path.exists() and summary_path.exists():
        return read_jsonl(rows_path), json.loads(summary_path.read_text(encoding="utf-8"))
    rows = generate(
        config,
        root,
        examples,
        adapter_for_step(checkpoint_root, step, final_step),
        step,
    )
    attach_rewards(config, rows, state_dir / "reward_calls.jsonl")
    write_jsonl(rows_path, rows)
    summary = {"split": split_name, "step": step, **summarize(rows)}
    write_json(summary_path, summary)
    return rows, summary


def select_checkpoint(
    baseline: list[dict[str, Any]],
    candidates: dict[int, list[dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
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
    else:
        raise ValueError(f"Unsupported checkpoint selection rule: {selection_rule}")
    winner = max(
        feasible,
        key=lambda row: float(row["delta_p_own_lcb_95_one_sided"]),
        default=None,
    )
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
    parser = argparse.ArgumentParser(
        description="Evaluate an authorized GSM8K official run on validation, then the held-out test"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    requested = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    if str(requested.get("status")) != "PREPARED_NOT_AUTHORIZED":
        raise ValueError("Unexpected official configuration status")
    root = Path(requested["project"]["root"]).resolve()
    run_dir = resolve(root, requested["paths"]["experiments"]) / args.run_id
    config = json.loads((run_dir / "parameters" / "resolved_config.json").read_text(encoding="utf-8"))
    training = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if training.get("status") != "complete":
        raise RuntimeError("Official training did not complete successfully")
    checkpoint_root = resolve(root, config["paths"]["checkpoints"]) / args.run_id
    split_dir = resolve(root, config["paths"]["split_dir"])
    validation = frozen_subset(
        read_jsonl(split_dir / str(config["dataset"]["validation_file"])),
        count=int(config["dataset"]["validation_count"]),
    )
    test = read_jsonl(split_dir / str(config["dataset"]["final_test_file"]))
    if len(test) != int(config["dataset"]["final_test_count"]):
        raise ValueError("Held-out GSM8K test row count differs from the frozen config")
    write_json(
        run_dir / "parameters" / "evaluation_split_manifest.json",
        {
            "validation": {
                "n": len(validation),
                "ids_sha256": sha256_json([row["id"] for row in validation]),
                "ordered_ids": [row["id"] for row in validation],
            },
            "test": {
                "n": len(test),
                "ids_sha256": sha256_json([row["id"] for row in test]),
                "ordered_ids": [row["id"] for row in test],
                "source": "complete official openai/gsm8k main test split",
                "access_policy": "evaluated only after validation checkpoint selection",
            },
        },
    )
    load_env_file(resolve(root, config["paths"]["openrouter_env"]))
    checkpoints = [int(step) for step in config["evaluation"]["checkpoints"]]
    final_step = checkpoints[-1]
    validation_rows: dict[int, list[dict[str, Any]]] = {}
    validation_summaries: list[dict[str, Any]] = []
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

    # The official test is first touched here, after validation has fixed the checkpoint.
    test_steps = [0] if selected_step == 0 else [0, selected_step]
    test_rows: dict[int, list[dict[str, Any]]] = {}
    test_summaries: list[dict[str, Any]] = []
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
    if selected_step:
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
        "validation": validation_summaries,
        "selection": selection,
        "test": test_summaries,
        "test_comparison": test_comparison,
    }
    write_json(run_dir / "evaluation_summary.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
