from __future__ import annotations

import gc
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from iar_campaign.four_domain_screen import (
    CAMPAIGN_ROOT,
    DOMAINS,
    _adapter_for_step,
    _attribution_template,
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
from iar_campaign.score_baselines import (
    _score_kodcode_rows,
    _score_simple,
    _score_spider_rows,
)
from iar_grpo.config import GrpoTrialConfig, load_grpo_config
from iar_grpo.reward import LiveOpenRouterReward, settings_from_config
from iar_grpo.train import run_training
from iar_phase0.generator import strip_code_fences
from iar_phase0.parameters import initialize_parameter_record

STEP_ROOT = CAMPAIGN_ROOT / "08_model_scale_3b"
SELECTION = CAMPAIGN_ROOT / "06_two_domain_selection/results/two_domain_selection_v1.json"
PROTOCOL = STEP_ROOT / "configs/model_scale_3b_protocol_v1.json"
ACTOR_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ACTOR_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
CHECKPOINTS = (0, 500, 1000, 1500, 2000)


def validate_scale_config(
    config: GrpoTrialConfig,
) -> tuple[str, Path, Path, dict[str, Any]]:
    raw = config.raw
    campaign = raw.get("campaign") or {}
    domain = str(campaign.get("domain", ""))
    if domain not in DOMAINS:
        raise ValueError("Step 8 has an unsupported domain")
    serialized = json.dumps(raw, sort_keys=True).lower()
    if "mbpp" in serialized or "ifbench" in serialized or "ifeval" in serialized:
        raise ValueError("Forbidden benchmark in Step 8 config")
    selection = _read_json(SELECTION)
    if (
        sha256_file(SELECTION) != str(campaign["domain_selection_sha256"])
        or selection.get("status") != "complete"
        or domain not in selection.get("selected_domains", [])
        or selection.get("selected_domains", [None])[0] != "commonsenseqa"
        or selection.get("official_test_accessed") is not False
    ):
        raise RuntimeError("Step 8 is not bound to the frozen domain selection")
    if sha256_file(PROTOCOL) != str(campaign["scale_protocol_sha256"]):
        raise RuntimeError("Step 8 protocol hash drifted")
    if (
        raw["actor"].get("model_id") != ACTOR_MODEL
        or raw["actor"].get("revision") != ACTOR_REVISION
    ):
        raise RuntimeError("Step 8 actor snapshot drifted")
    if config.prompt_count != 1000 or config.num_epochs != 2:
        raise ValueError("Step 8 requires train_core1000 for two epochs")
    if config.num_generations != 4 or config.expected_reward_calls != 8000:
        raise ValueError("Step 8 requires G=4 and 8000 logical rewards")
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
        raise ValueError("Step 8 GRPO hyperparameters drifted")
    reward_path = Path(str(campaign["reward_selection"]))
    reward = _read_json(reward_path)
    if (
        sha256_file(reward_path) != str(campaign["reward_selection_sha256"])
        or reward.get("status") != "complete"
        or str(raw["reward"]["value"])
        != str(reward["selected_reward_transform"])
    ):
        raise RuntimeError("Step 8 identity reward is not frozen")
    train_file = Path(str(raw["dataset"]["split_file"]))
    validation_file = Path(str(raw["evaluation"]["validation_file"]))
    if (
        sha256_file(train_file) != str(raw["dataset"]["split_file_sha256"])
        or sha256_file(validation_file)
        != str(raw["evaluation"]["validation_file_sha256"])
    ):
        raise RuntimeError("Step 8 data hashes drifted")
    if any(part.lower() in {"test", "sealed", "final_test"} for part in validation_file.parts):
        raise ValueError("Step 8 cannot access test data")
    return domain, train_file, validation_file, selection


def build_training_dataset(config: GrpoTrialConfig):
    from datasets import Dataset

    domain, train_file, _, _ = validate_scale_config(config)
    examples = read_jsonl(train_file)
    if len(examples) != 1000 or len({str(row["id"]) for row in examples}) != 1000:
        raise RuntimeError("Step 8 train_core1000 is incomplete")
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
    return Dataset.from_list(rows), rows


def run_training_attempt(
    *, config_path: Path, run_id: str, key_pool: Path, key_name: str, lock_dir: Path
) -> dict[str, Any]:
    config = load_grpo_config(config_path)
    domain, train_file, validation_file, _ = validate_scale_config(config)
    experiment_root = _under(
        config.resolve(config.raw["paths"]["experiments"]), STEP_ROOT, "Step 8 runs"
    )
    checkpoint_root = _under(
        config.resolve(config.raw["paths"]["checkpoints"]), STEP_ROOT, "Step 8 checkpoints"
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
    dataset, rows = build_training_dataset(config)
    write_jsonl(experiment_dir / "train_examples.jsonl", rows)
    write_json(
        experiment_dir / "campaign_inputs.json",
        {
            "domain": domain,
            "training_split": str(train_file),
            "training_split_sha256": sha256_file(train_file),
            "validation_split": str(validation_file),
            "validation_split_sha256": sha256_file(validation_file),
            "domain_selection": str(SELECTION),
            "domain_selection_sha256": sha256_file(SELECTION),
            "scale_protocol": str(PROTOCOL),
            "scale_protocol_sha256": sha256_file(PROTOCOL),
            "official_test_accessed": False,
        },
    )
    version, template_sha = _attribution_template(domain)
    if config.raw["discriminator"].get("attribution_prompt_version") != version:
        raise RuntimeError("Step 8 attribution prompt version drifted")
    from iar_campaign.judge_sanity import attribution_prompt

    runtime = LiveOpenRouterReward(
        settings=settings_from_config(config.raw),
        log_path=experiment_dir / "reward_calls.jsonl",
        api_key=load_named_secret(key_pool, key_name),
        prompt_builder=lambda task, candidate: attribution_prompt(domain, task, candidate),
        prompt_template_hash=template_sha,
        system_prompt_builder=lambda _name: None,
        candidate_normalizer=strip_code_fences if domain == "kodcode" else str.strip,
    )
    try:
        with key_lock(lock_dir, key_name):
            result, history = run_training(
                config, dataset, checkpoint_dir, experiment_dir, runtime
            )
        runtime.assert_complete()
        write_jsonl(experiment_dir / "trainer_log_history.jsonl", history)
        stats = runtime.reward_statistics()
        expected_ids = {str(row["task_id"]) for row in rows}
        observed = stats.get("by_task") or {}
        if set(observed) != expected_ids or any(
            int(value.get("n", -1)) != 8 for value in observed.values()
        ):
            raise RuntimeError("Step 8 reward coverage mismatch")
        gradients = [float(row["grad_norm"]) for row in history if "grad_norm" in row]
        if not gradients or any(not math.isfinite(value) for value in gradients):
            raise RuntimeError("Step 8 gradients are missing or non-finite")
        summary = {
            "schema_version": 1,
            "status": "complete",
            "completed_at": utc_now(),
            "domain": domain,
            "run_id": run_id,
            "actor_model": ACTOR_MODEL,
            "actor_revision": ACTOR_REVISION,
            "reward_value": config.raw["reward"]["value"],
            "dataset_rows": len(rows),
            "expected_reward_calls": 8000,
            "reward_audit": runtime.audit_summary(),
            "reward_statistics": stats,
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
                "status": "failed",
                "failed_at": utc_now(),
                "domain": domain,
                "run_id": run_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "reward_audit": runtime.audit_summary(),
                "official_test_accessed": False,
            },
        )
        raise


def generate_validation_attempt(
    *, config_path: Path, run_id: str, step: int, attempt: int
) -> dict[str, Any]:
    if step not in CHECKPOINTS or attempt <= 0:
        raise ValueError("Invalid Step 8 validation checkpoint or attempt")
    config = load_grpo_config(config_path)
    domain, _, validation_file, _ = validate_scale_config(config)
    run_root = _under(
        config.resolve(config.raw["paths"]["experiments"]) / run_id,
        STEP_ROOT,
        "Step 8 run",
    )
    checkpoint_root = _under(
        config.resolve(config.raw["paths"]["checkpoints"]) / run_id,
        STEP_ROOT,
        "Step 8 checkpoint",
    )
    training = _read_json(run_root / "summary.json")
    if training.get("status") != "complete":
        raise RuntimeError("Step 8 validation cannot precede training completion")
    output_dir = (
        run_root
        / "evaluations/validation"
        / f"step_{step:04d}"
        / f"generation_attempt{attempt:03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from iar_grpo.numerics import configure_stable_sdpa

    examples = read_jsonl(validation_file)
    tables = (
        Path(str(config.raw["evaluation"]["spider_tables_json"]))
        if domain == "spider"
        else None
    )
    prompts = prompts_for_domain(domain, examples, tables_json=tables)["a0"]
    adapter = _adapter_for_step(checkpoint_root, step, CHECKPOINTS[-1])
    random.seed(23)
    torch.manual_seed(23)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(23)
    sdpa = configure_stable_sdpa(torch)
    source = config.actor_source()
    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, trust_remote_code=False, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = None
    model = None
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            source,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model = (
            PeftModel.from_pretrained(base_model, adapter, is_trainable=False)
            if adapter is not None
            else base_model
        )
        model.eval()
        candidates: list[dict[str, Any]] = []
        batch_size = int(config.raw["evaluation"]["batch_size"])
        max_tokens = int(config.raw["evaluation"]["max_new_tokens"])
        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start : start + batch_size]
            batch_prompts = prompts[start : start + batch_size]
            encoded = tokenizer.apply_chat_template(
                [[{"role": "user", "content": prompt}] for prompt in batch_prompts],
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_length = encoded["input_ids"].shape[1]
            for example, output in zip(batch_examples, generated, strict=True):
                completion = output[prompt_length:]
                candidates.append(
                    {
                        "id": str(example["id"]),
                        "source_index": int(example["source_index"]),
                        "domain": domain,
                        "step": step,
                        "run_id": run_id,
                        "response": tokenizer.decode(
                            completion, skip_special_tokens=True
                        ).strip(),
                        "completion_tokens": len(completion),
                        "hit_max_new_tokens": len(completion) >= max_tokens,
                    }
                )
        write_jsonl(output_dir / "candidates.jsonl", candidates)
        summary = {
            "schema_version": 1,
            "status": "complete",
            "completed_at": utc_now(),
            "domain": domain,
            "run_id": run_id,
            "step": step,
            "n": len(candidates),
            "truncated": sum(bool(row["hit_max_new_tokens"]) for row in candidates),
            "candidate_file_sha256": sha256_file(output_dir / "candidates.jsonl"),
            "validation_file": str(validation_file),
            "validation_file_sha256": sha256_file(validation_file),
            "adapter": str(adapter) if adapter is not None else None,
            "sdpa": sdpa,
            "official_test_accessed": False,
        }
        write_json(output_dir / "summary.json", summary)
        return summary
    finally:
        del model
        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def score_validation_attempt(
    *,
    config_path: Path,
    run_id: str,
    step: int,
    generation_attempt: int,
    scoring_attempt: int,
    score_workers: int,
) -> dict[str, Any]:
    config = load_grpo_config(config_path)
    domain, _, validation_file, _ = validate_scale_config(config)
    run_root = Path(str(config.raw["paths"]["experiments"])) / run_id
    generation_dir = (
        run_root
        / "evaluations/validation"
        / f"step_{step:04d}"
        / f"generation_attempt{generation_attempt:03d}"
    )
    generation = _read_json(generation_dir / "summary.json")
    candidates = read_jsonl(generation_dir / "candidates.jsonl")
    examples = read_jsonl(validation_file)
    if (
        generation.get("status") != "complete"
        or generation.get("official_test_accessed") is not False
        or generation.get("candidate_file_sha256")
        != sha256_file(generation_dir / "candidates.jsonl")
        or len(candidates) != 300
        or [str(row["id"]) for row in candidates]
        != [str(row["id"]) for row in examples]
    ):
        raise RuntimeError("Step 8 generation is incomplete or misaligned")
    output_dir = (
        run_root
        / "evaluations/validation"
        / f"step_{step:04d}"
        / f"score_attempt{scoring_attempt:03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    scored_rows = candidates
    protocol: dict[str, Any] | None = None
    if domain in {"commonsenseqa", "gsm8k"}:
        _score_simple(domain, examples, candidates)
    elif domain == "kodcode":
        scored_rows, protocol = _score_kodcode_rows(
            examples,
            candidates,
            scoring_protocol_path=Path(
                str(config.raw["evaluation"]["kodcode_scoring_protocol"])
            ),
            score_workers=score_workers,
            timeout_seconds=int(config.raw["evaluation"]["coding_timeout_seconds"]),
        )
    else:
        protocol = _score_spider_rows(
            examples,
            candidates,
            split_manifest_path=Path(
                str(config.raw["evaluation"]["spider_split_manifest"])
            ),
            scoring_protocol_path=Path(
                str(config.raw["evaluation"]["spider_scoring_protocol"])
            ),
            validation_file=validation_file,
            database_root=Path(str(config.raw["evaluation"]["spider_database_root"])),
            score_workers=score_workers,
        )
    write_jsonl(output_dir / "candidates.jsonl", candidates)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "domain": domain,
        "run_id": run_id,
        "step": step,
        "n": len(candidates),
        "scored_n": len(scored_rows),
        "correct": sum(bool(row["score"]["correct"]) for row in scored_rows),
        "valid": sum(bool(row["score"].get("valid", True)) for row in scored_rows),
        "task_utility": statistics.fmean(
            bool(row["score"]["correct"]) for row in scored_rows
        ),
        "status_counts": dict(
            sorted(
                Counter(
                    str(row["score"].get("status", "scored")) for row in scored_rows
                ).items()
            )
        ),
        "candidate_file_sha256": sha256_file(output_dir / "candidates.jsonl"),
        "generation_summary": str(generation_dir / "summary.json"),
        "generation_summary_sha256": sha256_file(generation_dir / "summary.json"),
        "scoring_protocol": protocol,
        "official_test_accessed": False,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def collect_scale(*, config_paths: list[Path], run_id: str) -> dict[str, Any]:
    domains: dict[str, Any] = {}
    for config_path in config_paths:
        config = load_grpo_config(config_path)
        domain, _, _, _ = validate_scale_config(config)
        run_root = Path(str(config.raw["paths"]["experiments"])) / run_id
        evaluations = []
        for step in CHECKPOINTS:
            score_root = run_root / "evaluations/validation" / f"step_{step:04d}"
            complete = []
            for path in sorted(score_root.glob("score_attempt*/summary.json")):
                value = _read_json(path)
                if value.get("status") == "complete":
                    complete.append((path, value))
            if len(complete) != 1:
                raise RuntimeError(
                    f"Expected one Step 8 score for {domain}/step{step}, found {len(complete)}"
                )
            path, value = complete[0]
            evaluations.append(
                {
                    "step": step,
                    "task_utility": float(value["task_utility"]),
                    "summary": str(path),
                    "summary_sha256": sha256_file(path),
                }
            )
        selected = max(
            evaluations, key=lambda row: (float(row["task_utility"]), -int(row["step"]))
        )
        domains[domain] = {
            "selected_step": int(selected["step"]),
            "selected_task_utility": float(selected["task_utility"]),
            "validation_gain_vs_a0": float(selected["task_utility"])
            - float(evaluations[0]["task_utility"]),
            "evaluations": evaluations,
            "training_summary": str(run_root / "summary.json"),
            "training_summary_sha256": sha256_file(run_root / "summary.json"),
        }
    result = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "protocol": "campaign-selected-domain-model-scale-3b-v1",
        "actor_model": ACTOR_MODEL,
        "actor_revision": ACTOR_REVISION,
        "domains": domains,
        "checkpoint_selection": "maximize validation task utility; earliest checkpoint on ties",
        "official_test_accessed": False,
        "mbpp": "FORBIDDEN",
        "ifbench": "DEFERRED",
    }
    destination = STEP_ROOT / "results/model_scale_3b_validation_v1.json"
    if destination.exists():
        raise FileExistsError(destination)
    write_json(destination, result)
    return result
