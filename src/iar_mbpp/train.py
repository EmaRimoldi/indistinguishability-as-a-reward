from __future__ import annotations

import argparse
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset

from iar_grpo.config import GrpoTrialConfig
from iar_grpo.reward import LiveOpenRouterReward, settings_from_config
from iar_grpo.train import run_training
from iar_phase0.data import MbppExample
from iar_phase0.env import load_env_file
from iar_phase0.parameters import initialize_parameter_record
from iar_phase0.prompts import generator_prompt, mbpp_google_problem_statement

from .data import write_json, write_jsonl
from .screen_protocol import (
    TRAIN_COUNT,
    audit_readiness,
    config_sha256,
    load_frozen_split,
    resolve,
    verify_authorization_gate,
)

SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def row_to_example(row: dict[str, Any]) -> MbppExample:
    if row.get("source_split") != "train":
        raise RuntimeError("MBPP optimisation row is not from the official train split")
    return MbppExample(
        split="train",
        index=int(row["source_index"]),
        task_id=str(row["id"]),
        prompt=str(row["prompt"]),
        code=None,
        test_setup_code=str(row["test_setup_code"]),
        tests=[str(test) for test in row["tests"]],
    )


def build_dataset(
    config: dict[str, Any], *, root: Path
) -> tuple[Dataset, list[dict[str, Any]]]:
    source = load_frozen_split(config, root=root, role="optimisation")
    if len(source) != TRAIN_COUNT:
        raise RuntimeError("MBPP optimisation row count changed")
    rows: list[dict[str, Any]] = []
    for row in source:
        example = row_to_example(row)
        task_statement = mbpp_google_problem_statement(example)
        rows.append(
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": generator_prompt(
                            example, variant="plain", fewshot=[]
                        ),
                    }
                ],
                "task_statement": task_statement,
                "task_id": str(example.task_id),
                "dataset_index": int(example.index),
            }
        )
    return Dataset.from_list(rows), rows


def _set_scratch_environment(config: dict[str, Any], *, root: Path) -> None:
    scratch = resolve(root, config["paths"]["scratch_root"])
    paths = {
        "HF_HOME": resolve(root, config["paths"]["hf_cache"]),
        "TMPDIR": resolve(root, config["paths"]["tmp"]),
        "XDG_CACHE_HOME": scratch / "cache/xdg",
        "TRITON_CACHE_DIR": scratch / "cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": scratch / "cache/torchinductor",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the gated MBPP five-domain screen"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization-gate", type=Path, required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(config["project"]["root"]).resolve()
    audit_readiness(config, root=root, require_ready=True)
    verify_authorization_gate(args.authorization_gate, config_path=config_path)
    _set_scratch_environment(config, root=root)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"mbpp_screen_seed17_{timestamp}"
    if not SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("Unsafe MBPP run ID")
    experiment_dir = resolve(root, config["paths"]["experiments"]) / run_id
    checkpoint_dir = resolve(root, config["paths"]["checkpoints"]) / run_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, experiment_dir / "config.yaml")
    initialize_parameter_record(
        experiment_dir, resolved_config=config, source_config=config_path
    )
    shutil.copy2(
        args.authorization_gate.resolve(),
        experiment_dir / "parameters" / "authorization_gate.json",
    )

    dataset, rows = build_dataset(config, root=root)
    write_jsonl(experiment_dir / "train_examples_no_gold.jsonl", rows)
    load_env_file(resolve(root, config["paths"]["openrouter_env"]))
    runtime = LiveOpenRouterReward(
        settings=settings_from_config(config),
        log_path=experiment_dir / "reward_calls.jsonl",
    )
    trial_config = GrpoTrialConfig(raw=config, path=config_path)
    try:
        result, history = run_training(
            trial_config,
            dataset,
            checkpoint_dir,
            experiment_dir,
            runtime,
        )
        runtime.assert_complete()
        write_jsonl(experiment_dir / "trainer_log_history.jsonl", history)
        statistics = runtime.reward_statistics()
        expected_ids = {str(row["task_id"]) for row in rows}
        observed = statistics.get("by_task") or {}
        if set(observed) != expected_ids:
            raise RuntimeError("MBPP reward log does not cover the frozen train split")
        expected_per_task = int(config["training"]["num_generations"]) * int(
            config["training"]["num_epochs"]
        )
        incomplete = {
            task_id: values.get("n")
            for task_id, values in observed.items()
            if int(values.get("n", -1)) != expected_per_task
        }
        if incomplete:
            raise RuntimeError(f"MBPP reward log has incomplete groups: {incomplete}")
        write_json(
            experiment_dir / "summary.json",
            {
                "run_id": run_id,
                "status": "complete",
                "config_sha256": config_sha256(config_path),
                "official_test_accessed": False,
                "dataset_rows": len(rows),
                "reward_audit": runtime.audit_summary(),
                "reward_statistics": statistics,
                "trainer_metrics": result.metrics,
                "checkpoint_dir": str(checkpoint_dir),
            },
        )
    except Exception as exc:
        write_json(
            experiment_dir / "failure.json",
            {
                "run_id": run_id,
                "status": "failed",
                "config_sha256": config_sha256(config_path),
                "official_test_accessed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "reward_audit": runtime.audit_summary(),
            },
        )
        raise


if __name__ == "__main__":
    main()
