from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset

from iar_grpo.reward import LiveOpenRouterReward, settings_from_config
from iar_phase0.env import load_env_file
from iar_phase0.parameters import initialize_parameter_record, write_parameter_artifact

from .data import read_jsonl, sha256_json, write_json, write_jsonl
from .prompts import (
    ATTRIBUTION_PROMPT_VERSION,
    attribution_prompt,
    task_prompt,
    template_hashes,
)


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def select_balanced(
    rows: list[dict[str, Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("CQA subset size must be positive")
    labels = "ABCDE"
    per_label, remainder = divmod(count, len(labels))
    # When ``count`` is not divisible by five, allocate the remainder with a
    # frozen seed-dependent label order.  This keeps label counts within one
    # example of each other without silently changing the requested size.
    remainder_order = list(labels)
    random.Random(seed + 104_729).shuffle(remainder_order)
    extra_labels = set(remainder_order[:remainder])
    selected: list[dict[str, Any]] = []
    for label_index, label in enumerate(labels):
        label_count = per_label + int(label in extra_labels)
        candidates = [row for row in rows if row["answer_key"] == label]
        random.Random(seed + 1291 * (label_index + 1)).shuffle(candidates)
        if len(candidates) < label_count:
            raise ValueError(
                f"Not enough training examples with answer {label}: "
                f"need {label_count}, found {len(candidates)}"
            )
        selected.extend(candidates[:label_count])
    random.Random(seed + 65_537).shuffle(selected)
    return selected


def actor_source(root: Path, config: dict[str, Any]) -> str:
    actor = config["actor"]
    model_id = str(actor["model_id"])
    if not bool(actor.get("local_files_only", True)):
        return model_id
    cache = resolve(root, config["paths"]["hf_cache"])
    revision = actor.get("revision") or actor.get("snapshot_revision")
    if revision:
        snapshot = (
            cache
            / f"models--{model_id.replace('/', '--')}"
            / "snapshots"
            / str(revision)
        )
        if not snapshot.is_dir():
            raise FileNotFoundError(f"No cached snapshot for {model_id}@{revision}")
        return str(snapshot)
    snapshots = sorted(
        (cache / f"models--{model_id.replace('/', '--')}" / "snapshots").glob("*")
    )
    if not snapshots:
        raise FileNotFoundError(f"No cached snapshot for {model_id}")
    return str(snapshots[-1])


def build_dataset(
    config: dict[str, Any], root: Path
) -> tuple[Dataset, list[dict[str, Any]]]:
    source_path = (
        resolve(root, config["paths"]["split_dir"]) / config["dataset"]["source_file"]
    )
    source = read_jsonl(source_path)
    selected = select_balanced(
        source,
        count=int(config["dataset"]["prompt_count"]),
        seed=int(config["project"]["seed"]),
    )
    rows = []
    for order, item in enumerate(selected):
        task = task_prompt(item["question"], item["choices"])
        rows.append(
            {
                "prompt": [{"role": "user", "content": task}],
                "task_statement": task,
                "task_id": str(item["id"]),
                "dataset_index": int(item["source_index"]),
                "experiment_order": order,
                "answer_key": item["answer_key"],
            }
        )
    return Dataset.from_list(rows), rows


def validate(config: dict[str, Any]) -> None:
    prompts = int(config["dataset"]["prompt_count"])
    generations = int(config["training"]["num_generations"])
    epochs = int(config["training"]["num_epochs"])
    expected = prompts * generations * epochs
    if expected != int(config["trial"]["expected_reward_calls"]):
        raise ValueError("Reward-call plan does not match prompts * K * epochs")
    if expected != int(config["reward"]["max_reward_calls"]):
        raise ValueError("Reward budget does not match the trial plan")
    if int(config["reward"]["max_http_requests"]) != expected * int(
        config["reward"]["max_attempts_per_call"]
    ):
        raise ValueError("HTTP budget does not match reward calls * attempts")
    global_batch = int(config["training"]["per_device_train_batch_size"]) * int(
        config["training"].get("world_size", 1)
    )
    if int(config["training"]["max_steps"]) != expected // global_batch:
        raise ValueError("max_steps must equal total rollouts / global batch")
    if int(config["training"]["generation_batch_size"]) % generations:
        raise ValueError("generation_batch_size must contain complete K groups")
    checkpoints = [int(step) for step in config["evaluation"]["checkpoints"]]
    if (
        not checkpoints
        or checkpoints[0] != 0
        or checkpoints[-1] != int(config["training"]["max_steps"])
        or checkpoints != sorted(set(checkpoints))
    ):
        raise ValueError(
            "Evaluation checkpoints must be unique, ordered, and span step 0 to max_steps"
        )
    if config["reward"]["value"] not in {
        "absolute_logp_1",
        "log_p_own",
        "p_own",
        "logit_p_own",
        "binary_own",
    }:
        raise ValueError("Unsupported CQA reward transform")
    if (
        config["discriminator"]["attribution_prompt_version"]
        != ATTRIBUTION_PROMPT_VERSION
    ):
        raise ValueError(
            "CQA attribution prompt version does not match the frozen template"
        )


def run_training(
    config: dict[str, Any],
    root: Path,
    dataset: Dataset,
    checkpoint_dir: Path,
    experiment_dir: Path,
    runtime: LiveOpenRouterReward,
):
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
    os.environ.setdefault("HF_HOME", str(cache))
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

    reward_function.__name__ = "cqa_openrouter_indistinguishability_reward"
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
    parser = argparse.ArgumentParser(
        description="Run the gated CommonsenseQA GRPO short trial"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate(config)
    gate = json.loads(args.gate.resolve().read_text(encoding="utf-8"))
    if gate.get("decision") != "GO":
        raise RuntimeError(
            f"Training is not authorized by Phase 0 gate: {gate.get('decision')}"
        )
    root = Path(config["project"]["root"]).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{config['trial']['run_name']}_{timestamp}"
    experiment_dir = resolve(root, config["paths"]["experiments"]) / run_id
    checkpoint_dir = resolve(root, config["paths"]["checkpoints"]) / run_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, experiment_dir / "config.yaml")
    initialize_parameter_record(
        experiment_dir, resolved_config=config, source_config=config_path
    )
    shutil.copy2(
        args.gate.resolve(), experiment_dir / "parameters" / "phase0_gate.json"
    )

    dataset, rows = build_dataset(config, root)
    write_jsonl(experiment_dir / "train_examples.jsonl", rows)
    write_json(
        experiment_dir / "parameters" / "training_subset_manifest.json",
        {
            "n": len(rows),
            "seed": int(config["project"]["seed"]),
            "answer_key_counts": dict(
                sorted(Counter(row["answer_key"] for row in rows).items())
            ),
            "ids_sha256": sha256_json([row["task_id"] for row in rows]),
            "ordered_ids": [row["task_id"] for row in rows],
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
        expected = int(config["training"]["num_generations"]) * int(
            config["training"]["num_epochs"]
        )
        bad = {
            task: values["n"]
            for task, values in statistics["by_task"].items()
            if int(values["n"]) != expected
        }
        if bad or len(statistics["by_task"]) != len(rows):
            raise RuntimeError(f"Incomplete reward groups: {bad}")
        gradient_rows = [row for row in history if "grad_norm" in row]
        summary = {
            "run_id": run_id,
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_rows": len(rows),
            "num_generations": int(config["training"]["num_generations"]),
            "reward_audit": runtime.audit_summary(),
            "reward_statistics": statistics,
            "trainer_metrics": result.metrics,
            "optimization_audit": {
                "logged_steps": len(gradient_rows),
                "nonzero_gradient_steps": sum(
                    float(row["grad_norm"]) > 0 for row in gradient_rows
                ),
                "min_grad_norm": min(float(row["grad_norm"]) for row in gradient_rows),
                "max_grad_norm": max(float(row["grad_norm"]) for row in gradient_rows),
            },
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
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    main()
