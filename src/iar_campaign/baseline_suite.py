from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from iar_campaign.baseline_rewards import (
    QUALITY_PROMPT_VERSION,
    LocalOracleReward,
    quality_prompt,
    quality_prompt_template_sha256,
)
from iar_campaign.four_domain_screen import (
    CAMPAIGN_ROOT,
    DOMAINS,
    _read_json,
    _under,
)
from iar_campaign.generate import prompts_for_domain, read_jsonl
from iar_campaign.reward_pilot import (
    key_lock,
    load_named_secret,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from iar_grpo.config import GrpoTrialConfig, load_grpo_config
from iar_grpo.reward import LiveOpenRouterReward, settings_from_config
from iar_grpo.train import run_training
from iar_phase0.generator import strip_code_fences
from iar_phase0.parameters import initialize_parameter_record

STEP_ROOT = CAMPAIGN_ROOT / "07_baseline_suite"
SELECTION = CAMPAIGN_ROOT / "06_two_domain_selection/results/two_domain_selection_v1.json"
PROTOCOL = STEP_ROOT / "configs/baseline_suite_protocol_v1.json"
TRAINABLE_GRPO_ARMS = ("grpo_quality", "grpo_oracle")


def validate_baseline_grpo_config(
    config: GrpoTrialConfig,
) -> tuple[str, str, Path, Path, dict[str, Any]]:
    raw = config.raw
    campaign = raw.get("campaign") or {}
    domain = str(campaign.get("domain", ""))
    arm = str(campaign.get("arm", ""))
    if domain not in DOMAINS or arm not in TRAINABLE_GRPO_ARMS:
        raise ValueError("Step 7 config has an unsupported domain or GRPO arm")
    serialized = json.dumps(raw, sort_keys=True).lower()
    if "mbpp" in serialized or "ifbench" in serialized or "ifeval" in serialized:
        raise ValueError("Forbidden benchmark in Step 7 config")
    if sha256_file(SELECTION) != str(campaign["domain_selection_sha256"]):
        raise RuntimeError("Step 6 selection hash drifted")
    selection = _read_json(SELECTION)
    if (
        selection.get("status") != "complete"
        or selection.get("protocol") != "campaign-two-domain-selection-v1"
        or selection.get("official_test_accessed") is not False
        or domain not in selection.get("selected_domains", [])
        or selection.get("selected_domains", [None])[0] != "commonsenseqa"
    ):
        raise RuntimeError("Step 7 is not bound to the frozen two-domain selection")
    if sha256_file(PROTOCOL) != str(campaign["baseline_protocol_sha256"]):
        raise RuntimeError("Step 7 baseline protocol hash drifted")
    if config.prompt_count != 1000 or config.num_epochs != 2:
        raise ValueError("Step 7 GRPO requires train_core1000 for two epochs")
    if config.num_generations != 4 or config.expected_reward_calls != 8000:
        raise ValueError("Step 7 GRPO requires G=4 and 8000 logical rewards")
    training = raw["training"]
    if (
        int(training["max_steps"]) != 2000
        or int(training["per_device_train_batch_size"]) != 4
        or int(training["generation_batch_size"]) != 8
        or int(training["save_steps"]) != 500
        or float(training["learning_rate"]) != 3e-5
        or float(training["beta"]) != 0.003
        or float(training["temperature"]) != 0.6
        or str(training["loss_type"]) != "dr_grpo"
        or str(training["scale_rewards"]) != "group"
    ):
        raise ValueError("Step 7 GRPO hyperparameters drifted")
    reward = raw["reward"]
    if arm == "grpo_quality":
        reward_selection = Path(str(campaign["reward_selection"]))
        if sha256_file(reward_selection) != str(campaign["reward_selection_sha256"]):
            raise RuntimeError("Step 4 reward selection hash drifted")
        selected = str(_read_json(reward_selection)["selected_reward_transform"])
        if (
            str(reward["value"]) != selected
            or raw["discriminator"].get("attribution_prompt_version")
            != QUALITY_PROMPT_VERSION
            or raw["discriminator"].get("attribution_prompt_template_sha256")
            != quality_prompt_template_sha256()
        ):
            raise RuntimeError("GRPO-quality is not bound to f*(q) and its frozen prompt")
    elif (
        str(reward["value"]) != "oracle_task_utility"
        or int(reward["max_http_requests"]) != 0
    ):
        raise RuntimeError("GRPO-oracle must use only the local task verifier")
    train_file = Path(str(raw["dataset"]["split_file"]))
    validation_file = Path(str(raw["evaluation"]["validation_file"]))
    if sha256_file(train_file) != str(raw["dataset"]["split_file_sha256"]):
        raise RuntimeError("Step 7 training split hash drifted")
    if sha256_file(validation_file) != str(raw["evaluation"]["validation_file_sha256"]):
        raise RuntimeError("Step 7 validation split hash drifted")
    if any(part.lower() in {"test", "sealed", "final_test"} for part in validation_file.parts):
        raise ValueError("Step 7 cannot access official or sealed test data")
    expected_root = (
        "frozen_historical_seed23"
        if domain in {"commonsenseqa", "gsm8k"}
        else "frozen_seed23_v1"
    )
    expected_train = {
        "commonsenseqa": "train.jsonl",
        "gsm8k": "official_train.jsonl",
        "kodcode": "train_core1000.jsonl",
        "spider": "train_core1000.jsonl",
    }[domain]
    if train_file.parent.name != expected_root or train_file.name != expected_train:
        raise ValueError("Step 7 is not using the frozen train_core1000 IDs")
    if validation_file.name not in {"validation.jsonl", "validation300.jsonl"}:
        raise ValueError("Step 7 is not using validation300")
    return domain, arm, train_file, validation_file, selection


def build_training_dataset(config: GrpoTrialConfig):
    from datasets import Dataset

    domain, _, train_file, _, _ = validate_baseline_grpo_config(config)
    examples = read_jsonl(train_file)
    if len(examples) != 1000 or len({str(row["id"]) for row in examples}) != 1000:
        raise RuntimeError("Step 7 train_core1000 is incomplete or duplicated")
    tables = (
        Path(str(config.raw["evaluation"]["spider_tables_json"]))
        if domain == "spider"
        else None
    )
    prompts = prompts_for_domain(domain, examples, tables_json=tables)["a0"]
    rows = [
        {
            "prompt": [{"role": "user", "content": prompt}],
            "task_statement": prompt,
            "task_id": str(example["id"]),
            "dataset_index": int(example["source_index"]),
        }
        for example, prompt in zip(examples, prompts, strict=True)
    ]
    return Dataset.from_list(rows), rows, examples


def run_grpo_baseline_attempt(
    *,
    config_path: Path,
    run_id: str,
    key_pool: Path | None = None,
    key_name: str | None = None,
    lock_dir: Path | None = None,
) -> dict[str, Any]:
    config = load_grpo_config(config_path)
    domain, arm, train_file, validation_file, _ = validate_baseline_grpo_config(config)
    experiment_root = _under(
        config.resolve(config.raw["paths"]["experiments"]), STEP_ROOT, "Step 7 run root"
    )
    checkpoint_root = _under(
        config.resolve(config.raw["paths"]["checkpoints"]),
        STEP_ROOT,
        "Step 7 checkpoint root",
    )
    experiment_dir = experiment_root / run_id
    checkpoint_dir = checkpoint_root / run_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    (experiment_dir / "config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    initialize_parameter_record(
        experiment_dir, resolved_config=config.raw, source_config=config_path
    )
    dataset, rows, examples = build_training_dataset(config)
    write_jsonl(experiment_dir / "train_examples.jsonl", rows)
    write_json(
        experiment_dir / "campaign_inputs.json",
        {
            "domain": domain,
            "arm": arm,
            "training_split": str(train_file),
            "training_split_sha256": sha256_file(train_file),
            "validation_split": str(validation_file),
            "validation_split_sha256": sha256_file(validation_file),
            "domain_selection": str(SELECTION),
            "domain_selection_sha256": sha256_file(SELECTION),
            "baseline_protocol": str(PROTOCOL),
            "baseline_protocol_sha256": sha256_file(PROTOCOL),
            "official_test_accessed": False,
        },
    )
    if arm == "grpo_quality":
        if key_pool is None or key_name is None or lock_dir is None:
            raise ValueError("GRPO-quality requires a named OpenRouter lane")
        runtime: Any = LiveOpenRouterReward(
            settings=settings_from_config(config.raw),
            log_path=experiment_dir / "reward_calls.jsonl",
            api_key=load_named_secret(key_pool, key_name),
            prompt_builder=quality_prompt,
            prompt_template_hash=quality_prompt_template_sha256(),
            system_prompt_builder=lambda _name: None,
            candidate_normalizer=strip_code_fences if domain == "kodcode" else str.strip,
        )
    else:
        database_root = (
            Path(str(config.raw["evaluation"]["spider_database_root"]))
            if domain == "spider"
            else None
        )
        runtime = LocalOracleReward(
            domain=domain,
            examples=examples,
            log_path=experiment_dir / "reward_calls.jsonl",
            max_reward_calls=8000,
            kodcode_timeout_seconds=int(
                config.raw["evaluation"].get("coding_timeout_seconds", 60)
            ),
            spider_database_root=database_root,
            max_concurrency=4,
        )
    try:
        if arm == "grpo_quality":
            assert key_name is not None and lock_dir is not None
            with key_lock(lock_dir, key_name):
                result, history = run_training(
                    config, dataset, checkpoint_dir, experiment_dir, runtime
                )
        else:
            result, history = run_training(
                config, dataset, checkpoint_dir, experiment_dir, runtime
            )
        runtime.assert_complete()
        write_jsonl(experiment_dir / "trainer_log_history.jsonl", history)
        statistics = runtime.reward_statistics()
        observed = statistics.get("by_task") or {}
        expected_ids = {str(row["task_id"]) for row in rows}
        if set(observed) != expected_ids or any(
            int(value.get("n", -1)) != 8 for value in observed.values()
        ):
            raise RuntimeError("Step 7 reward coverage differs from 1000 × G4 × 2")
        gradients = [float(row["grad_norm"]) for row in history if "grad_norm" in row]
        if not gradients or any(not math.isfinite(value) for value in gradients):
            raise RuntimeError("Step 7 gradient audit is missing or non-finite")
        summary = {
            "schema_version": 1,
            "status": "complete",
            "completed_at": utc_now(),
            "run_id": run_id,
            "domain": domain,
            "arm": arm,
            "reward_value": config.raw["reward"]["value"],
            "dataset_rows": len(rows),
            "num_epochs": 2,
            "expected_reward_calls": 8000,
            "reward_audit": runtime.audit_summary(),
            "reward_statistics": statistics,
            "trainer_metrics": result.metrics,
            "optimization_audit": {
                "logged_steps": len(gradients),
                "nonzero_gradient_steps": sum(value > 0 for value in gradients),
                "min_grad_norm": min(gradients),
                "max_grad_norm": max(gradients),
            },
            "checkpoint_dir": str(checkpoint_dir),
            "official_test_accessed": False,
        }
        write_json(experiment_dir / "summary.json", summary)
        return summary
    except Exception as exc:
        write_json(
            experiment_dir / "failure.json",
            {
                "schema_version": 1,
                "status": "failed",
                "failed_at": utc_now(),
                "run_id": run_id,
                "domain": domain,
                "arm": arm,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "reward_audit": runtime.audit_summary(),
                "official_test_accessed": False,
            },
        )
        raise
