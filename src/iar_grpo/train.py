from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from iar_phase0.data import write_jsonl
from iar_phase0.env import load_env_file
from iar_phase0.parameters import initialize_parameter_record, write_parameter_artifact

from .config import load_grpo_config
from .data import build_trial_dataset
from .reward import LiveOpenRouterReward, settings_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one audited GRPO pilot")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()

    config = load_grpo_config(args.config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{config.raw['trial']['run_name']}_{timestamp}"
    experiment_dir = config.resolve(config.raw["paths"]["experiments"]) / run_id
    checkpoint_dir = config.resolve(config.raw["paths"]["checkpoints"]) / run_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config.path, experiment_dir / "config.yaml")
    initialize_parameter_record(
        experiment_dir,
        resolved_config=config.raw,
        source_config=config.path,
    )

    dataset, rows = build_trial_dataset(config)
    write_jsonl(experiment_dir / "train_examples.jsonl", rows)

    env_path = config.resolve(config.raw["paths"]["openrouter_env"])
    load_env_file(env_path)
    runtime = LiveOpenRouterReward(
        settings=settings_from_config(config.raw),
        log_path=experiment_dir / "reward_calls.jsonl",
    )

    try:
        result, log_history = run_training(
            config,
            dataset,
            checkpoint_dir,
            experiment_dir,
            runtime,
        )
        runtime.assert_complete()
        write_jsonl(experiment_dir / "trainer_log_history.jsonl", log_history)
        reward_statistics = runtime.reward_statistics()
        expected_task_ids = {str(row["task_id"]) for row in rows}
        observed = reward_statistics.get("by_task") or {}
        if set(observed) != expected_task_ids:
            missing = sorted(expected_task_ids - set(observed))
            extra = sorted(set(observed) - expected_task_ids)
            raise RuntimeError(f"Reward coverage mismatch; missing={missing}, extra={extra}")
        expected_calls_per_task = config.num_generations * config.num_epochs
        bad_counts = {
            task_id: values.get("n")
            for task_id, values in observed.items()
            if int(values.get("n", -1)) != expected_calls_per_task
        }
        if bad_counts:
            raise RuntimeError(f"Incomplete GRPO groups in reward log: {bad_counts}")
        gradient_rows = [row for row in log_history if "grad_norm" in row]
        summary = {
            "run_id": run_id,
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "actor_model": config.raw["actor"]["model_id"],
            "discriminator_model": config.raw["discriminator"]["model_id"],
            "dataset_rows": len(dataset),
            "num_generations": config.num_generations,
            "num_epochs": config.num_epochs,
            "reward_audit": runtime.audit_summary(),
            "reward_statistics": reward_statistics,
            "trainer_metrics": result.metrics,
            "optimization_audit": {
                "logged_steps": len(gradient_rows),
                "nonzero_gradient_steps": sum(float(row["grad_norm"]) > 0 for row in gradient_rows),
                "min_grad_norm": min(float(row["grad_norm"]) for row in gradient_rows),
                "max_grad_norm": max(float(row["grad_norm"]) for row in gradient_rows),
                "max_clipped_ratio": max(
                    float(row.get("completions/clipped_ratio", 0.0)) for row in gradient_rows
                ),
            },
            "checkpoint_dir": str(checkpoint_dir),
        }
        (experiment_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    except Exception as exc:
        failure = {
            "run_id": run_id,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "reward_audit": runtime.audit_summary(),
        }
        (experiment_dir / "failure.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        raise


def run_training(
    config,
    dataset,
    checkpoint_dir: Path,
    experiment_dir: Path,
    runtime: LiveOpenRouterReward,
    *,
    resume_from_checkpoint: Path | None = None,
):
    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from iar_grpo.numerics import (
        guarded_grpo_trainer_class,
        install_stable_grpo_logprobs,
    )

    actor = config.raw["actor"]
    training = config.raw["training"]
    hf_cache = config.resolve(config.raw["paths"]["hf_cache"])
    actor_source = config.actor_source()
    os.environ.setdefault("HF_HOME", str(hf_cache))

    tokenizer = AutoTokenizer.from_pretrained(
        actor_source,
        cache_dir=str(hf_cache),
        local_files_only=bool(actor["local_files_only"]),
        trust_remote_code=False,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_prompt_tokens = int(training["max_prompt_tokens"])
    lengths = []
    for row in dataset:
        prompt = row["prompt"]
        if isinstance(prompt, list):
            token_ids = tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
            )
        else:
            token_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        lengths.append(len(token_ids))
    if max(lengths) > max_prompt_tokens:
        raise ValueError(f"Prompt length {max(lengths)} exceeds configured limit {max_prompt_tokens}")

    async def indistinguishability_reward(
        prompts: list[str],
        completions: list[str],
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

    indistinguishability_reward.__name__ = "openrouter_indistinguishability_reward"

    grpo_args = GRPOConfig(
        output_dir=str(checkpoint_dir),
        run_name=str(config.raw["trial"]["run_name"]),
        seed=int(config.raw["project"]["seed"]),
        data_seed=int(config.raw["project"]["seed"]),
        learning_rate=float(training["learning_rate"]),
        warmup_ratio=float(training["warmup_ratio"]),
        lr_scheduler_type=str(training["lr_scheduler_type"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        num_train_epochs=float(config.num_epochs),
        max_steps=int(training["max_steps"]),
        logging_steps=1,
        save_strategy=str(training.get("save_strategy", "no")),
        save_steps=int(training.get("save_steps", 500)),
        save_total_limit=training.get("save_total_limit"),
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        bf16=bool(training.get("bf16", True)),
        use_cpu=bool(training.get("use_cpu", False)),
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
            "cache_dir": str(hf_cache),
            "local_files_only": bool(actor["local_files_only"]),
            "trust_remote_code": False,
            "torch_dtype": torch.bfloat16 if training.get("bf16", True) else torch.float32,
        },
    )
    peft_config = LoraConfig(
        r=int(actor["lora"]["r"]),
        lora_alpha=int(actor["lora"]["alpha"]),
        lora_dropout=float(actor["lora"]["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(actor["lora"]["target_modules"]),
    )
    write_parameter_artifact(experiment_dir, "trainer_arguments.json", grpo_args.to_dict())
    write_parameter_artifact(experiment_dir, "lora_arguments.json", peft_config.to_dict())
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
        model=actor_source,
        reward_funcs=indistinguishability_reward,
        args=grpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    result = trainer.train(
        resume_from_checkpoint=(
            str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
        )
    )
    final_dir = checkpoint_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    return result, list(trainer.state.log_history)


if __name__ == "__main__":
    main()
