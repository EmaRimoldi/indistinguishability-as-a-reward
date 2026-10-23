from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

from iar_campaign.generate import read_jsonl
from iar_campaign.reward_pilot import sha256_file, utc_now, write_json
from iar_phase0.parameters import initialize_parameter_record, write_parameter_artifact

from .baseline_suite import PROTOCOL, SELECTION, STEP_ROOT

ACTOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ACTOR_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
CHECKPOINTS = (0, 125, 250, 375, 500)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _actor_source(cache: Path) -> Path:
    value = cache / "models--Qwen--Qwen2.5-1.5B-Instruct/snapshots" / ACTOR_REVISION
    if not value.is_dir():
        raise FileNotFoundError(value)
    return value


def _project_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for row in rows:
        messages = row.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or messages[0].get("role") != "user"
            or messages[1].get("role") != "assistant"
            or not str(messages[1].get("content", "")).strip()
        ):
            raise ValueError(f"Malformed SFT row: {row.get('id')}")
        projected.append(
            {
                "id": str(row["id"]),
                "prompt": [messages[0]],
                "completion": [messages[1]],
                **{
                    key: bool(row[key])
                    for key in ("teacher_correct", "teacher_valid")
                    if key in row
                },
            }
        )
    return projected


def run_sft_baseline(
    *,
    teacher_summary: Path,
    run_id: str,
    cache_dir: Path,
) -> dict[str, Any]:
    selection = _read_json(SELECTION)
    if (
        selection.get("status") != "complete"
        or selection.get("official_test_accessed") is not False
    ):
        raise RuntimeError("SFT is not bound to a complete Step 6 selection")
    teacher = _read_json(teacher_summary)
    domain = str(teacher.get("domain", ""))
    if (
        teacher.get("status") != "complete"
        or teacher.get("protocol") != "campaign-step7-sft-teacher-v1"
        or int(teacher.get("n", -1)) != 1000
        or domain not in selection.get("selected_domains", [])
        or teacher.get("official_test_accessed") is not False
    ):
        raise RuntimeError("Teacher corpus is incomplete or targets another domain")
    expected_teacher = (
        STEP_ROOT / "domains" / domain / "sft_teacher/runs/teacher_attempt001/summary.json"
    ).resolve()
    if teacher_summary.resolve() != expected_teacher:
        raise ValueError("Unexpected teacher run path")
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    sft_path = Path(str(teacher["sft_all"]))
    if sha256_file(sft_path) != str(teacher["sft_all_sha256"]):
        raise RuntimeError("SFT corpus hash drifted")
    raw_rows = read_jsonl(sft_path)
    if len(raw_rows) != 1000 or len({str(row["id"]) for row in raw_rows}) != 1000:
        raise RuntimeError("SFT requires exactly one teacher response for each prompt")
    rows = _project_rows(raw_rows)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from iar_grpo.numerics import configure_stable_sdpa

    arm = STEP_ROOT / "domains" / domain / "sft_teacher"
    experiment_dir = arm / "sft_runs" / run_id
    checkpoint_dir = arm / "sft_checkpoints" / run_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    random.seed(23)
    torch.manual_seed(23)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(23)
    sdpa = configure_stable_sdpa(torch)
    source = _actor_source(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        local_files_only=True,
        trust_remote_code=False,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_lengths: list[int] = []
    sequence_lengths: list[int] = []
    completion_lengths: list[int] = []
    for row in rows:
        prompt_ids = tokenizer.apply_chat_template(
            row["prompt"], tokenize=True, add_generation_prompt=True
        )
        sequence_ids = tokenizer.apply_chat_template(
            row["prompt"] + row["completion"],
            tokenize=True,
            add_generation_prompt=False,
        )
        prompt_lengths.append(len(prompt_ids))
        sequence_lengths.append(len(sequence_ids))
        completion_lengths.append(max(0, len(sequence_ids) - len(prompt_ids)))
    max_length = max(sequence_lengths)
    if max_length > 4864:
        raise RuntimeError(f"SFT sequence would exceed the frozen 4864-token cap: {max_length}")
    config = {
        "protocol": "campaign-step7-sft-v1",
        "domain": domain,
        "run_id": run_id,
        "teacher_summary": str(teacher_summary.resolve()),
        "teacher_summary_sha256": sha256_file(teacher_summary.resolve()),
        "sft_corpus": str(sft_path),
        "sft_corpus_sha256": sha256_file(sft_path),
        "domain_selection": str(SELECTION),
        "domain_selection_sha256": sha256_file(SELECTION),
        "baseline_protocol": str(PROTOCOL),
        "baseline_protocol_sha256": sha256_file(PROTOCOL),
        "actor_model": ACTOR_MODEL,
        "actor_revision": ACTOR_REVISION,
        "seed": 23,
        "num_epochs": 2,
        "effective_batch_size": 4,
        "learning_rate": 1e-4,
        "checkpoints": list(CHECKPOINTS),
        "official_test_accessed": False,
        "mbpp": "FORBIDDEN",
    }
    initialize_parameter_record(
        experiment_dir, resolved_config=config, source_config=PROTOCOL
    )
    shutil.copy2(PROTOCOL, experiment_dir / "baseline_suite_protocol_v1.json")
    training = SFTConfig(
        output_dir=str(checkpoint_dir),
        run_name=f"campaign_step7_{domain}_sft_seed23",
        seed=23,
        data_seed=23,
        learning_rate=1e-4,
        warmup_ratio=0.0,
        lr_scheduler_type="constant",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_train_epochs=2.0,
        max_steps=500,
        logging_steps=1,
        save_strategy="steps",
        save_steps=125,
        save_total_limit=None,
        eval_strategy="no",
        report_to="none",
        bf16=True,
        max_length=max_length,
        completion_only_loss=True,
        shuffle_dataset=False,
        packing=False,
        include_num_input_tokens_seen=True,
        include_tokens_per_second=True,
        skip_memory_metrics=False,
        remove_unused_columns=True,
        model_init_kwargs={
            "cache_dir": str(cache_dir),
            "local_files_only": True,
            "trust_remote_code": False,
            "torch_dtype": torch.bfloat16,
        },
    )
    lora = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    write_parameter_artifact(experiment_dir, "trainer_arguments.json", training.to_dict())
    write_parameter_artifact(experiment_dir, "lora_arguments.json", lora.to_dict())
    write_parameter_artifact(experiment_dir, "numerical_runtime.json", sdpa)
    write_parameter_artifact(
        experiment_dir,
        "token_length_audit.json",
        {
            "n": len(rows),
            "prompt_tokens_total": sum(prompt_lengths),
            "prompt_tokens_max": max(prompt_lengths),
            "completion_tokens_total": sum(completion_lengths),
            "completion_tokens_max": max(completion_lengths),
            "sequence_tokens_total": sum(sequence_lengths),
            "sequence_tokens_max": max(sequence_lengths),
            "configured_max_length": max_length,
        },
    )
    trainer = SFTTrainer(
        model=source,
        args=training,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        peft_config=lora,
    )
    started = time.time()
    try:
        result = trainer.train()
        trainer.save_model(str(checkpoint_dir / "final_adapter"))
        tokenizer.save_pretrained(str(checkpoint_dir / "final_adapter"))
        history = list(trainer.state.log_history)
        gradients = [float(row["grad_norm"]) for row in history if "grad_norm" in row]
        if not gradients or any(not math.isfinite(value) for value in gradients):
            raise RuntimeError("SFT gradient audit is missing or non-finite")
        write_json(experiment_dir / "trainer_log_history.json", history)
        summary = {
            "schema_version": 1,
            "status": "complete",
            "completed_at": utc_now(),
            "protocol": "campaign-step7-sft-v1",
            "domain": domain,
            "run_id": run_id,
            "dataset_count": len(rows),
            "num_epochs": 2,
            "global_step": int(trainer.state.global_step),
            "train_runtime_seconds": time.time() - started,
            "trainer_metrics": dict(result.metrics),
            "optimization_audit": {
                "logged_steps": len(gradients),
                "nonzero_gradient_steps": sum(value > 0 for value in gradients),
                "min_grad_norm": min(gradients),
                "max_grad_norm": max(gradients),
            },
            "checkpoint_dir": str(checkpoint_dir),
            "official_test_accessed": False,
            "mbpp": "FORBIDDEN",
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
                "domain": domain,
                "run_id": run_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "official_test_accessed": False,
            },
        )
        raise
