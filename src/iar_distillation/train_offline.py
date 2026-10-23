from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset

from iar_phase0.parameters import initialize_parameter_record, write_parameter_artifact

from .preference_data import actor_source, resolve
from .teacher_data import read_jsonl, sha256_file, write_json


def sft_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    projected = []
    for row in rows:
        messages = row["messages"]
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or messages[0].get("role") != "user"
            or messages[1].get("role") != "assistant"
        ):
            raise ValueError(f"Malformed SFT conversation: {row.get('id')}")
        projected.append(
            {
                "id": str(row["id"]),
                "domain": str(row["domain"]),
                "prompt": [messages[0]],
                "completion": [messages[1]],
                "teacher_correct": bool(row["teacher_correct"]),
                "teacher_valid": bool(row["teacher_valid"]),
            }
        )
    return projected


def load_dataset_rows(
    config: dict[str, Any], root: Path, method: str, domain: str
) -> tuple[Dataset, Path]:
    run_dir = resolve(root, config["paths"]["teacher_root"]) / domain
    filename = {
        "sft": "sft_all.jsonl",
        "sft_matched": "sft_dpo_chosen.jsonl",
        "dpo": "dpo_pairs.jsonl",
    }[method]
    path = run_dir / filename
    rows = read_jsonl(path)
    expected = int(config["domains"][domain]["expected_count"])
    if len(rows) != expected:
        raise ValueError(f"{domain}/{method}: expected {expected} rows, found {len(rows)}")
    if method == "sft":
        rows = sft_rows(rows)
    elif method in {"sft_matched", "dpo"}:
        audit = json.loads((run_dir / "dpo_audit.json").read_text(encoding="utf-8"))
        if audit.get("status") != "PASS" or int(audit.get("n", -1)) != expected:
            raise RuntimeError(f"{domain}: DPO dataset has no passing complete audit")
    return Dataset.from_list(rows), path


def lora_config(config: dict[str, Any]):
    from peft import LoraConfig

    lora = config["actor"]["lora"]
    return LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora["target_modules"]),
    )


def length_audit(
    tokenizer,
    dataset: Dataset,
    *,
    method: str,
    max_prompt_length: int,
    max_length: int,
) -> dict[str, Any]:
    prompt_lengths: list[int] = []
    sequence_lengths: list[int] = []
    completion_lengths: list[int] = []
    for row in dataset:
        prompt = list(row["prompt"])
        prompt_ids = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True
        )
        prompt_lengths.append(len(prompt_ids))
        completions = [row["completion"]] if method == "sft" else [row["chosen"], row["rejected"]]
        for completion in completions:
            sequence_ids = tokenizer.apply_chat_template(
                prompt + list(completion), tokenize=True, add_generation_prompt=False
            )
            sequence_lengths.append(len(sequence_ids))
            completion_lengths.append(max(0, len(sequence_ids) - len(prompt_ids)))
    over_prompt = sum(length > max_prompt_length for length in prompt_lengths)
    over_total = sum(length > max_length for length in sequence_lengths)
    result = {
        "n_prompts": len(prompt_lengths),
        "n_sequences": len(sequence_lengths),
        "prompt_tokens_max": max(prompt_lengths),
        "prompt_tokens_total": sum(prompt_lengths),
        "completion_tokens_max": max(completion_lengths),
        "completion_tokens_total": sum(completion_lengths),
        "sequence_tokens_max": max(sequence_lengths),
        "sequence_tokens_total": sum(sequence_lengths),
        "configured_max_prompt_length": max_prompt_length,
        "configured_max_length": max_length,
        "prompts_over_limit": over_prompt,
        "sequences_over_limit": over_total,
    }
    if over_prompt or over_total:
        raise ValueError(f"Token-length preflight would truncate training examples: {result}")
    return result


def train(
    config: dict[str, Any],
    root: Path,
    method: str,
    domain: str,
    run_id: str,
) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

    dataset, dataset_path = load_dataset_rows(config, root, method, domain)
    objective = "dpo" if method == "dpo" else "sft"
    method_cfg = config["training"][objective]
    domain_cfg = config["domains"][domain]
    seed = int(domain_cfg.get("seed", config["project"]["seed"]))
    experiment_dir = resolve(root, config["paths"]["experiments"]) / domain / method / run_id
    checkpoint_dir = resolve(root, config["paths"]["checkpoints"]) / domain / method / run_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    (experiment_dir / "parameters").mkdir(parents=True, exist_ok=True)
    cache = resolve(root, config["paths"]["hf_cache"])
    os.environ.setdefault("HF_HOME", str(cache))
    source = actor_source(root, config)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        cache_dir=str(cache),
        local_files_only=bool(config["actor"].get("local_files_only", True)),
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lengths = length_audit(
        tokenizer,
        dataset,
        method=objective,
        max_prompt_length=int(domain_cfg["max_prompt_length"]),
        max_length=int(domain_cfg["max_length"]),
    )

    common: dict[str, Any] = {
        "output_dir": str(checkpoint_dir),
        "run_name": run_id,
        "seed": seed,
        "data_seed": seed,
        "learning_rate": float(method_cfg["learning_rate"]),
        "warmup_ratio": float(method_cfg["warmup_ratio"]),
        "lr_scheduler_type": str(method_cfg["lr_scheduler_type"]),
        "per_device_train_batch_size": int(method_cfg["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(method_cfg["gradient_accumulation_steps"]),
        "gradient_checkpointing": bool(method_cfg["gradient_checkpointing"]),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "num_train_epochs": float(method_cfg["num_epochs"]),
        "max_steps": int(domain_cfg["max_steps"]),
        "logging_steps": int(method_cfg["logging_steps"]),
        "save_strategy": "steps",
        "save_steps": int(domain_cfg["steps_per_epoch"]),
        "save_total_limit": None,
        "eval_strategy": "no",
        "report_to": "none",
        "bf16": True,
        "include_num_input_tokens_seen": True,
        "include_tokens_per_second": True,
        "skip_memory_metrics": False,
        "remove_unused_columns": True,
        "model_init_kwargs": {
            "cache_dir": str(cache),
            "local_files_only": bool(config["actor"].get("local_files_only", True)),
            "trust_remote_code": False,
            "torch_dtype": torch.bfloat16,
        },
    }
    peft = lora_config(config)
    if objective == "sft":
        trainer_args = SFTConfig(
            **common,
            max_length=int(domain_cfg["max_length"]),
            completion_only_loss=True,
            shuffle_dataset=False,
            packing=False,
        )
        trainer = SFTTrainer(
            model=source,
            args=trainer_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft,
        )
    else:
        trainer_args = DPOConfig(
            **common,
            beta=float(method_cfg["beta"]),
            loss_type="sigmoid",
            max_length=int(domain_cfg["max_length"]),
            max_prompt_length=int(domain_cfg["max_prompt_length"]),
            max_completion_length=int(domain_cfg["max_new_tokens"]),
            precompute_ref_log_probs=bool(method_cfg["precompute_ref_log_probs"]),
            disable_dropout=bool(method_cfg["disable_dropout"]),
        )
        trainer = DPOTrainer(
            model=source,
            ref_model=None,
            args=trainer_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft,
        )

    resolved = {
        **config,
        "resolved_run": {
            "method": method,
            "domain": domain,
            "run_id": run_id,
            "dataset_path": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "dataset_count": len(dataset),
            "actor_source": source,
            "experiment_dir": str(experiment_dir),
            "checkpoint_dir": str(checkpoint_dir),
        },
    }
    initialize_parameter_record(
        experiment_dir,
        resolved_config=resolved,
        source_config=Path(config["_source_config"]),
    )
    shutil.copy2(config["_source_config"], experiment_dir / "config.yaml")
    write_parameter_artifact(experiment_dir, "trainer_arguments.json", trainer_args.to_dict())
    write_parameter_artifact(experiment_dir, "lora_arguments.json", peft.to_dict())
    write_parameter_artifact(experiment_dir, "token_length_audit.json", lengths)

    started = time.time()
    result = trainer.train()
    final_dir = checkpoint_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log_history = list(trainer.state.log_history)
    write_json(experiment_dir / "trainer_log_history.json", log_history)
    summary = {
        "status": "complete",
        "method": method,
        "domain": domain,
        "run_id": run_id,
        "model_id": str(config["actor"]["model_id"]),
        "seed": seed,
        "dataset_count": len(dataset),
        "dataset_sha256": sha256_file(dataset_path),
        "num_epochs": float(method_cfg["num_epochs"]),
        "max_steps": int(domain_cfg["max_steps"]),
        "effective_batch_size": int(method_cfg["per_device_train_batch_size"])
        * int(method_cfg["gradient_accumulation_steps"]),
        "learning_rate": float(method_cfg["learning_rate"]),
        "beta": float(method_cfg["beta"]) if objective == "dpo" else None,
        "train_runtime_seconds": time.time() - started,
        "train_metrics": dict(result.metrics),
        "token_budget_estimate": {
            "sequence_tokens_per_epoch": int(lengths["sequence_tokens_total"]),
            "policy_sequence_tokens": int(
                lengths["sequence_tokens_total"] * float(method_cfg["num_epochs"])
            ),
            "reference_precompute_sequence_tokens": (
                int(lengths["sequence_tokens_total"])
                if objective == "dpo" and bool(method_cfg["precompute_ref_log_probs"])
                else 0
            ),
            "total_forward_sequence_tokens": int(
                lengths["sequence_tokens_total"] * float(method_cfg["num_epochs"])
                + (
                    lengths["sequence_tokens_total"]
                    if objective == "dpo" and bool(method_cfg["precompute_ref_log_probs"])
                    else 0
                )
            ),
        },
        "global_step": int(trainer.state.global_step),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(experiment_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exposure-matched SFT or DPO")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=("sft", "sft_matched", "dpo"), required=True)
    parser.add_argument("--domain", choices=("mbpp", "commonsenseqa", "gsm8k"), required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["_source_config"] = str(config_path)
    root = Path(config["project"]["root"]).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    seed = int(config["domains"][args.domain].get("seed", config["project"]["seed"]))
    run_id = args.run_id or f"{args.domain}_{args.method}_deepseek_v1_seed{seed}_{timestamp}"
    print(json.dumps(train(config, root, args.method, args.domain, run_id), indent=2))


if __name__ == "__main__":
    main()
