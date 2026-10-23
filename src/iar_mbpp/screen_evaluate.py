from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from iar_grpo.config import GrpoTrialConfig
from iar_grpo.evaluate_pilot import (
    adapter_for_step,
    evaluate_state,
    select_validation_checkpoint,
)
from iar_phase0.data import MbppExample
from iar_phase0.env import load_env_file

from .data import write_json
from .screen_protocol import (
    CHECKPOINTS,
    VALIDATION_COUNT,
    audit_readiness,
    config_sha256,
    load_frozen_split,
    resolve,
)

SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def validation_examples(config: dict[str, Any], *, root: Path) -> list[MbppExample]:
    rows = load_frozen_split(config, root=root, role="validation")
    examples = [
        MbppExample(
            split="validation",
            index=int(row["source_index"]),
            task_id=str(row["id"]),
            prompt=str(row["prompt"]),
            code=None,
            test_setup_code=str(row["test_setup_code"]),
            tests=[str(test) for test in row["tests"]],
        )
        for row in rows
    ]
    if len(examples) != VALIDATION_COUNT:
        raise RuntimeError("MBPP validation row count changed")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MBPP screen checkpoints on canonical validation only"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not SAFE_RUN_ID.fullmatch(args.run_id):
        raise ValueError("Unsafe MBPP run ID")

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(config["project"]["root"]).resolve()
    audit_readiness(config, root=root, require_ready=True)
    run_dir = resolve(root, config["paths"]["experiments"]) / args.run_id
    training_summary = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )
    if (
        training_summary.get("status") != "complete"
        or training_summary.get("official_test_accessed") is not False
        or training_summary.get("config_sha256") != config_sha256(config_path)
    ):
        raise RuntimeError("MBPP training artifact is incomplete, stale, or unsafe")

    load_env_file(resolve(root, config["paths"]["openrouter_env"]))
    examples = validation_examples(config, root=root)
    trial_config = GrpoTrialConfig(raw=config, path=config_path)
    checkpoint_root = resolve(root, config["paths"]["checkpoints"]) / args.run_id
    rows_by_step: dict[int, list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []
    spec = config["evaluation"]["validation"]
    for step in CHECKPOINTS:
        state = f"step_{step:04d}"
        rows, summary = evaluate_state(
            config=trial_config,
            examples=examples,
            adapter_dir=adapter_for_step(checkpoint_root, step, CHECKPOINTS[-1]),
            state_name=state,
            split_name="validation",
            num_generations=int(spec["num_generations"]),
            temperature=float(spec["temperature"]),
            top_p=float(spec["top_p"]),
            output_dir=run_dir / "evaluations" / "validation" / state,
            seed=int(config["project"]["seed"]),
        )
        summary["step"] = step
        summary["official_test_accessed"] = False
        rows_by_step[step] = rows
        summaries.append(summary)
    selection = select_validation_checkpoint(
        rows_by_step,
        z=float(config["evaluation"]["paired_lcb_z"]),
        max_pass_drop=float(config["evaluation"]["max_pass_at_1_drop"]),
        selection_rule="validation_task_accuracy",
    )
    selection["official_test_accessed"] = False
    write_json(run_dir / "validation_checkpoint_summaries.json", {"rows": summaries})
    write_json(run_dir / "validation_checkpoint_selection.json", selection)
    print(json.dumps(selection, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
