from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset

from iar_grpo.reward import LiveOpenRouterReward, settings_from_config
from iar_phase0.env import load_env_file
from iar_phase0.parameters import initialize_parameter_record

from .data import sha256_json, write_json, write_jsonl
from .item4_protocol import (
    OPTIMISATION_COUNT,
    audit_readiness,
    config_sha256,
    load_frozen_split,
    resolve,
)
from .prompts import (
    ATTRIBUTION_PROMPT_VERSION,
    attribution_prompt,
    task_prompt,
    template_hashes,
)
from .train import run_training

SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def build_dataset(
    config: dict[str, Any], *, root: Path
) -> tuple[Dataset, list[dict[str, Any]]]:
    source = load_frozen_split(config, root, "optimisation")
    rows: list[dict[str, Any]] = []
    for item in source:
        task = task_prompt(str(item["question"]), list(item["choices"]))
        rows.append(
            {
                "prompt": [{"role": "user", "content": task}],
                "task_statement": task,
                "task_id": str(item["id"]),
                "dataset_index": int(item["source_index"]),
            }
        )
    if len(rows) != OPTIMISATION_COUNT:
        raise RuntimeError("CQA optimisation row count changed")
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
    parser = argparse.ArgumentParser(description="Run gated CQA Item 4 training")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization-gate", type=Path, required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()

    config_path = args.config.resolve()
    authorization = args.authorization_gate.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(config["project"]["root"]).resolve()
    audit_readiness(
        config,
        root=root,
        config_path=config_path,
        authorization_gate=authorization,
        require_ready=True,
    )
    _set_scratch_environment(config, root=root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"cqa_item4_seed23_{timestamp}"
    if not SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("Unsafe CQA run ID")
    experiment_dir = resolve(root, config["paths"]["experiments"]) / run_id
    checkpoint_dir = resolve(root, config["paths"]["checkpoints"]) / run_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, experiment_dir / "config.yaml")
    initialize_parameter_record(
        experiment_dir, resolved_config=config, source_config=config_path
    )
    shutil.copy2(
        authorization, experiment_dir / "parameters" / "authorization_gate.json"
    )

    dataset, rows = build_dataset(config, root=root)
    write_jsonl(experiment_dir / "train_examples_no_gold.jsonl", rows)
    write_json(
        experiment_dir / "parameters" / "training_pool.json",
        {
            "n": len(rows),
            "ordered_ids_sha256": sha256_json([str(row["task_id"]) for row in rows]),
            "source_split": "official_train_internal_optimisation",
            "official_test_accessed": False,
        },
    )
    load_env_file(resolve(root, config["paths"]["openrouter_env"]))
    runtime = LiveOpenRouterReward(
        settings=settings_from_config(config),
        log_path=experiment_dir / "reward_calls.jsonl",
        prompt_builder=attribution_prompt,
        prompt_template_hash=template_hashes()[ATTRIBUTION_PROMPT_VERSION],
        system_prompt_builder=lambda _name: None,
        candidate_normalizer=str.strip,
    )
    try:
        result, history = run_training(
            config, root, dataset, checkpoint_dir, experiment_dir, runtime
        )
        runtime.assert_complete()
        write_jsonl(experiment_dir / "trainer_log_history.jsonl", history)
        statistics = runtime.reward_statistics()
        expected_ids = {str(row["task_id"]) for row in rows}
        observed = statistics.get("by_task") or {}
        expected_per_task = int(config["training"]["num_generations"]) * int(
            config["training"]["num_epochs"]
        )
        incomplete = {
            task_id: values.get("n")
            for task_id, values in observed.items()
            if int(values.get("n", -1)) != expected_per_task
        }
        if set(observed) != expected_ids or incomplete:
            raise RuntimeError(f"CQA reward coverage is incomplete: {incomplete}")
        summary = {
            "run_id": run_id,
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config_sha256(config_path),
            "official_test_accessed": False,
            "dataset_rows": len(rows),
            "reward_audit": runtime.audit_summary(),
            "reward_statistics": statistics,
            "trainer_metrics": result.metrics,
            "checkpoint_dir": str(checkpoint_dir),
        }
        write_json(experiment_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
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
