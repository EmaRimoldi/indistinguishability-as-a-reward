from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from iar_analysis.evaluation_resume import load_completed_checkpoint
from iar_grpo.validation_reward import binary_auc
from iar_phase0.env import load_env_file

from .data import write_json, write_jsonl
from .evaluate_short import adapter_for_step, attach_rewards, generate, summarize
from .screen_protocol import (
    CHECKPOINTS,
    SCREEN_VALIDATION,
    audit_readiness,
    config_sha256,
    load_frozen_split,
    resolve,
)

SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def select_checkpoint(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if [int(row["step"]) for row in summaries] != CHECKPOINTS:
        raise RuntimeError("GSM8K checkpoint evaluation is incomplete")
    selected = max(
        summaries, key=lambda row: (float(row["accuracy"]), -int(row["step"]))
    )
    return {
        "selector": "task_utility",
        "task_utility": "exact_answer_accuracy",
        "tie_break": "earliest_checkpoint",
        "selected_step": int(selected["step"]),
        "selected_accuracy": float(selected["accuracy"]),
        "official_test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate GSM8K screen checkpoints on internal validation only"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not SAFE_RUN_ID.fullmatch(args.run_id):
        raise ValueError("Unsafe GSM8K run ID")
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(config["project"]["root"]).resolve()
    audit_readiness(config, root=root, require_ready=True)
    run_dir = resolve(root, config["paths"]["experiments"]) / args.run_id
    training = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if (
        training.get("status") != "complete"
        or training.get("official_test_accessed") is not False
        or training.get("config_sha256") != config_sha256(config_path)
    ):
        raise RuntimeError("GSM8K training artifact is incomplete, stale, or unsafe")
    examples = load_frozen_split(config, root=root, role="validation")
    if len(examples) != SCREEN_VALIDATION:
        raise RuntimeError("GSM8K validation row count changed")
    load_env_file(resolve(root, config["paths"]["openrouter_env"]))
    checkpoint_root = resolve(root, config["paths"]["checkpoints"]) / args.run_id
    summaries = []
    for step in CHECKPOINTS:
        state_dir = run_dir / "evaluations" / "validation" / f"step_{step:05d}"
        cached = load_completed_checkpoint(
            state_dir,
            step=step,
            expected_rows=SCREEN_VALIDATION,
            rows_file="rows.jsonl",
            required_files=("reward_calls.jsonl",),
        )
        if cached is not None:
            summaries.append(cached)
            continue
        state_dir.mkdir(parents=True, exist_ok=False)
        rows = generate(
            config,
            root,
            examples,
            adapter_for_step(checkpoint_root, step, CHECKPOINTS[-1]),
            step,
        )
        attach_rewards(config, rows, state_dir / "reward_calls.jsonl")
        write_jsonl(state_dir / "rows.jsonl", rows)
        summary = {
            "step": step,
            **summarize(rows),
            "reward_auc_for_correctness": binary_auc(
                [bool(row["score"]["correct"]) for row in rows],
                [float(row["p_own"]) for row in rows],
            ),
            "official_test_accessed": False,
        }
        write_json(state_dir / "summary.json", summary)
        summaries.append(summary)
    selection = select_checkpoint(summaries)
    write_json(run_dir / "validation_checkpoint_summaries.json", {"rows": summaries})
    write_json(run_dir / "validation_checkpoint_selection.json", selection)
    print(json.dumps(selection, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
