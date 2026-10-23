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
from iar_phase0.parameters import initialize_parameter_record, write_parameter_artifact

from .data import (
    TRAIN_DATASET_ID,
    TRAIN_DATASET_REVISION,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)
from .evaluate_validation import _reject_heldout_artifacts
from .prompts import (
    ATTRIBUTION_PROMPT_VERSION,
    attribution_prompt,
    task_prompt,
    template_hashes,
)
from .screen_protocol import (
    OPTIMISATION_COUNT,
    audit_readiness,
    config_sha256,
    verify_authorization_gate,
)

SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"Blank JSONL line in {path} at {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected JSON object in {path} at {line_number}")
            rows.append(value)
    return rows


def actor_source(root: Path, config: dict[str, Any]) -> str:
    model_id = str(config["actor"]["model_id"])
    if config["actor"].get("local_files_only") is not True:
        raise RuntimeError("IFBench screen forbids an unpinned remote actor download")
    cache = resolve(root, config["paths"]["hf_cache"])
    revision = str(config["actor"]["revision"])
    snapshot = cache / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError(f"No cached snapshot for {model_id}@{revision}")
    return str(snapshot)


def load_optimisation_rows(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    split_dir = _reject_heldout_artifacts(resolve(root, config["paths"]["split_dir"]))
    manifest_path = split_dir / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("IFBench split manifest must be a JSON object")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    if (
        manifest.get("manifest_sha256") != sha256_json(unhashed)
        or manifest.get("manifest_sha256") != config["dataset"]["manifest_sha256"]
    ):
        raise RuntimeError("IFBench frozen split manifest hash mismatch")
    source_metadata = manifest.get("training_source") or {}
    if (
        source_metadata.get("dataset") != TRAIN_DATASET_ID
        or source_metadata.get("revision") != TRAIN_DATASET_REVISION
        or source_metadata.get("source_split") != "train"
    ):
        raise RuntimeError("IFBench optimisation is not from pinned RLVR-IFeval train")
    heldout = manifest.get("heldout_policy") or {}
    if (
        heldout.get("accessed_during_preparation") is not False
        or heldout.get("materialized_during_preparation") is not False
    ):
        raise RuntimeError("IFBench held-out data entered the frozen split")

    split = (manifest.get("splits") or {}).get("optimisation") or {}
    source = split_dir / str(config["dataset"]["source_file"])
    if (
        source.is_symlink()
        or split.get("file") != source.name
        or split.get("file_sha256") != sha256_file(source)
    ):
        raise RuntimeError("IFBench optimisation differs from its frozen manifest")
    rows = read_jsonl(source)
    if len(rows) != OPTIMISATION_COUNT or int(split.get("n", -1)) != len(rows):
        raise RuntimeError(
            f"IFBench optimisation split must contain exactly {OPTIMISATION_COUNT} rows"
        )
    ids = [str(row.get("id", "")) for row in rows]
    if (
        not all(ids)
        or len(ids) != len(set(ids))
        or split.get("ids_sha256") != sha256_json(ids)
        or split.get("rows_sha256") != sha256_json(rows)
    ):
        raise RuntimeError("IFBench optimisation content/order hash mismatch")
    for order, row in enumerate(rows):
        if (
            row.get("source_dataset") != TRAIN_DATASET_ID
            or row.get("source_split") != "train"
            or row.get("experiment_split") != "optimisation"
            or row.get("experiment_order") != order
            or not isinstance(row.get("prompt"), str)
            or not row["prompt"].strip()
        ):
            raise RuntimeError(f"Invalid internal-train optimisation row at {order}")
    return rows


def build_dataset(
    config: dict[str, Any], root: Path
) -> tuple[Dataset, list[dict[str, Any]]]:
    source = load_optimisation_rows(config, root)
    rows: list[dict[str, Any]] = []
    for item in source:
        task = task_prompt(str(item["prompt"]))
        rows.append(
            {
                "prompt": [{"role": "user", "content": task}],
                "task_statement": task,
                "task_id": str(item["id"]),
                "dataset_index": int(item["source_index"]),
            }
        )
    return Dataset.from_list(rows), rows


def run_training(
    config: dict[str, Any],
    root: Path,
    dataset: Dataset,
    checkpoint_dir: Path,
    experiment_dir: Path,
    runtime: LiveOpenRouterReward,
) -> tuple[Any, list[dict[str, Any]]]:
    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from iar_grpo.numerics import (
        guarded_grpo_trainer_class,
        install_stable_grpo_logprobs,
    )

    actor = config["actor"]
    training = config["training"]
    cache = resolve(root, config["paths"]["hf_cache"])
    source = actor_source(root, config)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        cache_dir=str(cache),
        local_files_only=True,
        trust_remote_code=False,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lengths = [
        len(
            tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True
            )
        )
        for row in dataset
    ]
    if max(lengths) > int(training["max_prompt_tokens"]):
        raise ValueError(f"Prompt length {max(lengths)} exceeds configured maximum")

    async def reward_function(
        prompts: list[Any],
        completions: list[Any],
        task_statement: list[str],
        task_id: list[str],
        dataset_index: list[int],
        **_: Any,
    ) -> list[float]:
        return await runtime.score_batch(
            completions=completions,
            task_statement=task_statement,
            task_id=task_id,
            dataset_index=dataset_index,
        )

    reward_function.__name__ = "ifbench_streamlake_indistinguishability_reward"
    args = GRPOConfig(
        output_dir=str(checkpoint_dir),
        run_name=str(config["trial"]["run_name"]),
        seed=int(config["project"]["seed"]),
        data_seed=int(config["project"]["seed"]),
        learning_rate=float(training["learning_rate"]),
        warmup_ratio=float(training["warmup_ratio"]),
        lr_scheduler_type=str(training["lr_scheduler_type"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        num_train_epochs=float(training["num_epochs"]),
        max_steps=int(training["max_steps"]),
        logging_steps=1,
        save_strategy=str(training["save_strategy"]),
        save_steps=int(training["save_steps"]),
        save_total_limit=training.get("save_total_limit"),
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        bf16=True,
        num_generations=int(training["num_generations"]),
        generation_batch_size=int(training["generation_batch_size"]),
        max_completion_length=int(training["max_completion_length"]),
        temperature=float(training["temperature"]),
        top_p=float(training["top_p"]),
        top_k=0,
        beta=float(training["beta"]),
        loss_type=str(training["loss_type"]),
        scale_rewards=str(training["scale_rewards"]),
        mask_truncated_completions=bool(training["mask_truncated_completions"]),
        log_completions=True,
        num_completions_to_print=int(training["num_completions_to_print"]),
        use_vllm=False,
        model_init_kwargs={
            "cache_dir": str(cache),
            "local_files_only": True,
            "trust_remote_code": False,
            "torch_dtype": torch.bfloat16,
        },
    )
    lora = actor["lora"]
    peft = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora["target_modules"]),
    )
    write_parameter_artifact(experiment_dir, "trainer_arguments.json", args.to_dict())
    write_parameter_artifact(experiment_dir, "lora_arguments.json", peft.to_dict())
    numerical_runtime = install_stable_grpo_logprobs()
    numerical_runtime["fail_on_nonfinite_actor_gradients"] = True
    numerical_runtime["anomaly_detection"] = (
        os.environ.get("IAR_GRPO_DETECT_ANOMALY", "0") == "1"
    )
    write_parameter_artifact(
        experiment_dir, "numerical_runtime.json", numerical_runtime
    )
    trainer_class = guarded_grpo_trainer_class(GRPOTrainer)
    trainer = trainer_class(
        model=source,
        reward_funcs=reward_function,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft,
    )
    result = trainer.train()
    final_dir = checkpoint_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    return result, list(trainer.state.log_history)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the gated IFBench 1.5B screen")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization-gate", type=Path, required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(config["project"]["root"]).resolve()
    audit_readiness(config, root=root, require_ready=True)
    verify_authorization_gate(args.authorization_gate, config_path=config_path)

    scratch = resolve(root, config["paths"]["scratch_root"])
    temporary = resolve(root, config["paths"]["tmp"])
    temporary.mkdir(parents=True, exist_ok=True)
    for name, path in {
        "HF_HOME": resolve(root, config["paths"]["hf_cache"]),
        "TMPDIR": temporary,
        "XDG_CACHE_HOME": scratch / "cache/xdg",
        "TRITON_CACHE_DIR": scratch / "cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": scratch / "cache/torchinductor",
    }.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"ifbench_screen_seed23_{timestamp}"
    if not SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("Unsafe IFBench run ID")
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
    dataset, rows = build_dataset(config, root)
    write_jsonl(experiment_dir / "train_examples_no_gold.jsonl", rows)
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
        reward_statistics = runtime.reward_statistics()
        expected_task_ids = {str(row["task_id"]) for row in rows}
        observed = reward_statistics.get("by_task") or {}
        if set(observed) != expected_task_ids:
            raise RuntimeError(
                "IFBench reward log does not cover the frozen optimisation set"
            )
        expected_per_task = int(config["training"]["num_generations"]) * int(
            config["training"]["num_epochs"]
        )
        incomplete = {
            task_id: values.get("n")
            for task_id, values in observed.items()
            if int(values.get("n", -1)) != expected_per_task
        }
        if incomplete:
            raise RuntimeError(
                f"IFBench reward log contains incomplete GRPO groups: {incomplete}"
            )
        write_json(
            experiment_dir / "summary.json",
            {
                "run_id": run_id,
                "protocol": config["protocol"],
                "config_sha256": config_sha256(config_path),
                "status": "complete",
                "official_test_accessed": False,
                "dataset_rows": len(rows),
                "reward_transform": config["reward"]["value"],
                "reward_audit": runtime.audit_summary(),
                "reward_statistics": reward_statistics,
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
                "official_test_accessed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    main()
