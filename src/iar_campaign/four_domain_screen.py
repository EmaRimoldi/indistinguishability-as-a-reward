from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from iar_campaign.generate import prompts_for_domain, read_jsonl
from iar_campaign.judge_sanity import attribution_prompt
from iar_campaign.reward_pilot import (
    _adapter_for_step,
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
from iar_phase0.generator import strip_code_fences
from iar_phase0.parameters import initialize_parameter_record

CAMPAIGN_ROOT = Path(
    "/path/to/runs"
)
STEP_ROOT = CAMPAIGN_ROOT / "identity_seed42"
DOMAINS = ("commonsenseqa", "gsm8k", "kodcode", "spider")
CHECKPOINTS = (0, 500, 1000, 1500, 2000)
REWARD_VALUES = ("log_p_own", "p_own", "logit_p_own", "binary_own")
ACTOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ACTOR_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def _under(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root.expanduser().resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escaped {root}: {resolved}") from exc
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _verify_hashed_input(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise RuntimeError(f"{label} is missing or drifted: {path}")
    value = _read_json(path)
    if value.get("official_test_accessed") is not False:
        raise RuntimeError(f"{label} accessed an official test")
    return value


def validate_screen_config(
    config: GrpoTrialConfig,
) -> tuple[str, Path, Path, dict[str, Any]]:
    raw = config.raw
    campaign = raw.get("campaign") or {}
    domain = str(campaign.get("domain", ""))
    if domain not in DOMAINS:
        raise ValueError(f"Step 5 supports only {DOMAINS}; MBPP is forbidden")
    serialized = json.dumps(raw, sort_keys=True).lower()
    if "mbpp" in serialized or "ifbench" in serialized or "ifeval" in serialized:
        raise ValueError("Forbidden benchmark in Step 5 config")
    if config.prompt_count != 1000 or config.num_epochs != 2:
        raise ValueError("Step 5 requires train_core1000 for exactly two epochs")
    if config.num_generations != 4 or config.expected_reward_calls != 8000:
        raise ValueError("Step 5 requires G=4 and exactly 8000 logical reward calls")
    training = raw["training"]
    if (
        int(training["max_steps"]) != 2000
        or int(training["per_device_train_batch_size"]) != 4
        or int(training["generation_batch_size"]) != 8
        or int(training["save_steps"]) != 500
    ):
        raise ValueError("Step 5 optimizer and checkpoint schedule drifted")
    if tuple(int(value) for value in raw["evaluation"]["checkpoints"]) != CHECKPOINTS:
        raise ValueError("Step 5 checkpoint set drifted")
    if (
        str(raw["actor"]["model_id"]) != ACTOR_MODEL
        or str(raw["actor"]["revision"]) != ACTOR_REVISION
    ):
        raise ValueError("Step 5 actor snapshot drifted")
    reward_result_path = _under(
        Path(campaign["reward_selection"]), CAMPAIGN_ROOT, "reward selection"
    )
    reward_result = _verify_hashed_input(
        reward_result_path,
        str(campaign["reward_selection_sha256"]),
        "Step 4 reward selection",
    )
    selected_reward = str(reward_result.get("selected_reward_transform", ""))
    if (
        reward_result.get("status") != "complete"
        or selected_reward not in REWARD_VALUES
        or str(raw["reward"]["value"]) != selected_reward
    ):
        raise RuntimeError("Step 5 is not bound to a complete Step 4 reward selection")
    cqa_gate_path = _under(Path(campaign["cqa_confirm_gate"]), CAMPAIGN_ROOT, "CQA gate")
    cqa_gate = _verify_hashed_input(
        cqa_gate_path, str(campaign["cqa_confirm_gate_sha256"]), "CQA confirmatory gate"
    )
    if cqa_gate.get("decision") != "GO":
        raise RuntimeError("CQA confirmatory judge gate is not GO")
    baseline_registry_path = _under(
        Path(raw["evaluation"]["baseline_registry"]), CAMPAIGN_ROOT, "baseline registry"
    )
    registry = _verify_hashed_input(
        baseline_registry_path,
        str(raw["evaluation"]["baseline_registry_sha256"]),
        "Step 5 baseline registry",
    )
    if (
        registry.get("status") != "complete"
        or registry.get("protocol") != "campaign-step5-validation-baseline-registry-v1"
        or set(registry.get("baselines") or {}) != set(DOMAINS)
    ):
        raise RuntimeError("Step 5 baseline registry is incomplete")
    split_file = _under(Path(raw["dataset"]["split_file"]), CAMPAIGN_ROOT, "train split")
    validation_file = _under(
        Path(raw["evaluation"]["validation_file"]), CAMPAIGN_ROOT, "validation split"
    )
    expected_root = (
        "frozen_historical_seed23"
        if domain in {"commonsenseqa", "gsm8k"}
        else "frozen_seed23_v1"
    )
    expected_train_name = {
        "commonsenseqa": "train.jsonl",
        "gsm8k": "official_train.jsonl",
        "kodcode": "train_core1000.jsonl",
        "spider": "train_core1000.jsonl",
    }[domain]
    if split_file.name != expected_train_name or split_file.parent.name != expected_root:
        raise ValueError("Step 5 training split is not frozen train_core1000")
    if validation_file.name not in {"validation.jsonl", "validation300.jsonl"}:
        raise ValueError("Step 5 evaluation is not validation300")
    if any(part.lower() in {"test", "sealed", "final_test"} for part in validation_file.parts):
        raise ValueError("Official or sealed data entered Step 5")
    if sha256_file(split_file) != str(raw["dataset"]["split_file_sha256"]):
        raise RuntimeError("Step 5 train split hash drifted")
    if sha256_file(validation_file) != str(raw["evaluation"]["validation_file_sha256"]):
        raise RuntimeError("Step 5 validation split hash drifted")
    if domain == "kodcode":
        protocol_path = _under(
            Path(raw["evaluation"]["kodcode_scoring_protocol"]),
            CAMPAIGN_ROOT,
            "KodCode scoring protocol",
        )
        protocol = _verify_hashed_input(
            protocol_path,
            str(raw["evaluation"]["kodcode_scoring_protocol_sha256"]),
            "KodCode scoring protocol",
        )
        if protocol.get("decision") != "GO" or int(protocol.get("scored_n", -1)) != 299:
            raise RuntimeError("KodCode scoring protocol is not 299/300 GO")
    if domain == "spider":
        split_manifest_path = _under(
            Path(raw["evaluation"]["spider_split_manifest"]),
            CAMPAIGN_ROOT,
            "Spider split manifest",
        )
        if sha256_file(split_manifest_path) != str(
            raw["evaluation"]["spider_split_manifest_sha256"]
        ):
            raise RuntimeError("Spider split manifest hash drifted")
        protocol_path = _under(
            Path(raw["evaluation"]["spider_scoring_protocol"]),
            CAMPAIGN_ROOT,
            "Spider scoring protocol",
        )
        protocol = _verify_hashed_input(
            protocol_path,
            str(raw["evaluation"]["spider_scoring_protocol_sha256"]),
            "Spider scoring protocol",
        )
        if protocol.get("decision") != "GO" or int(protocol.get("scored_n", -1)) != 300:
            raise RuntimeError("Spider scoring protocol is not 300/300 GO")
    return domain, split_file, validation_file, registry


def build_training_dataset(config: GrpoTrialConfig):
    from datasets import Dataset

    domain, split_file, _, _ = validate_screen_config(config)
    rows = read_jsonl(split_file)
    if len(rows) != 1000 or len({str(row["id"]) for row in rows}) != 1000:
        raise RuntimeError("train_core1000 is incomplete or duplicated")
    tables = (
        Path(config.raw["evaluation"]["spider_tables_json"])
        if domain == "spider"
        else None
    )
    prompts = prompts_for_domain(domain, rows, tables_json=tables)["a0"]
    actor_visible_fields = (
        ["question", "choices"]
        if domain == "commonsenseqa"
        else ["question", "schema"]
        if domain == "spider"
        else ["question"]
    )
    training_rows = [
        {
            "prompt": [{"role": "user", "content": prompt}],
            "task_statement": prompt,
            "task_id": str(item["id"]),
            "dataset_index": int(item["source_index"]),
            "actor_visible_fields": actor_visible_fields,
        }
        for item, prompt in zip(rows, prompts, strict=True)
    ]
    return Dataset.from_list(training_rows), training_rows


def _attribution_template(domain: str) -> tuple[str, str]:
    if domain == "commonsenseqa":
        from iar_cqa.prompts import (
            BINARY_CONTRACT_ATTRIBUTION_PROMPT_VERSION as ATTRIBUTION_PROMPT_VERSION,
        )
        from iar_cqa.prompts import (
            template_hashes,
        )
    elif domain == "gsm8k":
        from iar_gsm8k.prompts import ATTRIBUTION_PROMPT_VERSION, template_hashes
    elif domain == "spider":
        from iar_spider.prompts import ATTRIBUTION_PROMPT_VERSION, template_hashes
    else:
        from iar_phase0.prompts import (
            POINTWISE_SELF_ATTRIBUTION_NO_SYSTEM_PROMPT_VERSION as ATTRIBUTION_PROMPT_VERSION,
        )
        from iar_phase0.prompts import (
            pointwise_self_attribution_no_system_template_sha256,
        )

        return ATTRIBUTION_PROMPT_VERSION, pointwise_self_attribution_no_system_template_sha256()
    return ATTRIBUTION_PROMPT_VERSION, template_hashes()[ATTRIBUTION_PROMPT_VERSION]


def reward_runtime(
    *,
    config: GrpoTrialConfig,
    domain: str,
    log_path: Path,
    api_key: str,
    settings=None,
    call_id_offset: int = 0,
    http_request_id_offset: int = 0,
) -> LiveOpenRouterReward:
    version, template_sha = _attribution_template(domain)
    configured = str(config.raw["discriminator"]["attribution_prompt_version"])
    if configured != version:
        raise RuntimeError(f"Attribution prompt version drift for {domain}")
    return LiveOpenRouterReward(
        settings=settings or settings_from_config(config.raw),
        log_path=log_path,
        api_key=api_key,
        call_id_offset=call_id_offset,
        http_request_id_offset=http_request_id_offset,
        prompt_builder=lambda task, candidate: attribution_prompt(
            domain, task, candidate, version=version
        ),
        prompt_template_hash=template_sha,
        system_prompt_builder=lambda _name: None,
        candidate_normalizer=strip_code_fences if domain == "kodcode" else str.strip,
    )


def _prepare_resume_reward_log(
    *, source_log: Path, destination_log: Path, logical_calls: int
) -> tuple[int, int, dict[str, Any]]:
    if not source_log.is_file():
        raise FileNotFoundError(f"Resume reward log is missing: {source_log}")
    if logical_calls <= 0:
        raise ValueError("resume logical calls must be positive")
    rows = read_jsonl(source_log)
    recovered = [
        row for row in rows if int(row.get("call_id", -1)) <= logical_calls
    ]
    successful = [row for row in recovered if row.get("status") == "ok"]
    recovered_ids = {int(row["call_id"]) for row in successful}
    expected_ids = set(range(1, logical_calls + 1))
    if recovered_ids != expected_ids or len(successful) != logical_calls:
        raise RuntimeError(
            "Resume reward log does not contain exactly one successful result for "
            "each logical call through the selected checkpoint"
        )
    http_requests = max(int(row.get("http_request_id", 0)) for row in recovered)
    write_jsonl(destination_log, recovered)
    return logical_calls, http_requests, {
        "source_log": str(source_log),
        "source_log_sha256": sha256_file(source_log),
        "recovered_rows": len(recovered),
        "recovered_successful_rewards": len(successful),
        "logical_reward_calls_before_checkpoint": logical_calls,
        "http_requests_before_checkpoint": http_requests,
    }


def run_training_attempt(
    *,
    config_path: Path,
    run_id: str,
    key_pool: Path,
    key_name: str,
    lock_dir: Path,
    resume_from_checkpoint: Path | None = None,
    resume_reward_log: Path | None = None,
    resume_logical_calls: int | None = None,
) -> dict[str, Any]:
    from iar_grpo.train import run_training

    config = load_grpo_config(config_path)
    domain, split_file, validation_file, _ = validate_screen_config(config)
    experiment_root = _under(
        config.resolve(config.raw["paths"]["experiments"]), STEP_ROOT, "run root"
    )
    checkpoint_root = _under(
        config.resolve(config.raw["paths"]["checkpoints"]), STEP_ROOT, "checkpoint root"
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
            "training_split": str(split_file),
            "training_split_sha256": sha256_file(split_file),
            "validation_split": str(validation_file),
            "validation_split_sha256": sha256_file(validation_file),
            "reward_selection": config.raw["campaign"]["reward_selection"],
            "reward_selection_sha256": config.raw["campaign"]["reward_selection_sha256"],
            "cqa_confirm_gate": config.raw["campaign"]["cqa_confirm_gate"],
            "cqa_confirm_gate_sha256": config.raw["campaign"]["cqa_confirm_gate_sha256"],
            "official_test_accessed": False,
        },
    )
    resume: dict[str, Any] | None = None
    runtime_settings = settings_from_config(config.raw)
    call_id_offset = 0
    http_request_id_offset = 0
    if resume_from_checkpoint is not None:
        if resume_reward_log is None or resume_logical_calls is None:
            raise ValueError(
                "A checkpoint resume requires --resume-reward-log and "
                "--resume-logical-calls"
            )
        if not resume_from_checkpoint.is_dir() or not (
            resume_from_checkpoint / "trainer_state.json"
        ).is_file():
            raise FileNotFoundError(
                f"Resume checkpoint is incomplete: {resume_from_checkpoint}"
            )
        call_id_offset, http_request_id_offset, resume = _prepare_resume_reward_log(
            source_log=resume_reward_log,
            destination_log=experiment_dir / "reward_calls.jsonl",
            logical_calls=int(resume_logical_calls),
        )
        if call_id_offset >= runtime_settings.max_reward_calls:
            raise ValueError("Resume logical-call offset exhausts the reward budget")
        if http_request_id_offset >= runtime_settings.max_http_requests:
            raise ValueError("Resume HTTP offset exhausts the HTTP budget")
        runtime_settings = replace(
            runtime_settings,
            max_reward_calls=runtime_settings.max_reward_calls - call_id_offset,
            max_http_requests=runtime_settings.max_http_requests - http_request_id_offset,
        )
        resume.update(
            {
                "checkpoint": str(resume_from_checkpoint),
                "checkpoint_trainer_state_sha256": sha256_file(
                    resume_from_checkpoint / "trainer_state.json"
                ),
                "remaining_logical_reward_calls": runtime_settings.max_reward_calls,
                "remaining_http_requests": runtime_settings.max_http_requests,
            }
        )
        write_json(experiment_dir / "resume_manifest.json", resume)
    runtime = reward_runtime(
        config=config,
        domain=domain,
        log_path=experiment_dir / "reward_calls.jsonl",
        api_key=load_named_secret(key_pool, key_name),
        settings=runtime_settings,
        call_id_offset=call_id_offset,
        http_request_id_offset=http_request_id_offset,
    )
    try:
        with key_lock(lock_dir, key_name):
            result, history = run_training(
                config,
                dataset,
                checkpoint_dir,
                experiment_dir,
                runtime,
                resume_from_checkpoint=resume_from_checkpoint,
            )
        runtime.assert_complete()
        write_jsonl(experiment_dir / "trainer_log_history.jsonl", history)
        stats = runtime.reward_statistics()
        observed = stats.get("by_task") or {}
        expected_ids = {str(row["task_id"]) for row in rows}
        if set(observed) != expected_ids or any(
            int(value.get("n", -1)) != 8 for value in observed.values()
        ):
            raise RuntimeError("Step 5 reward coverage differs from 1000 × G4 × 2 epochs")
        gradients = [row for row in history if "grad_norm" in row]
        grad_values = [float(row["grad_norm"]) for row in gradients]
        if not grad_values or any(not math.isfinite(value) for value in grad_values):
            raise RuntimeError("Step 5 gradient audit is missing or non-finite")
        summary = {
            "schema_version": 1,
            "status": "complete",
            "completed_at": utc_now(),
            "run_id": run_id,
            "domain": domain,
            "reward_value": config.raw["reward"]["value"],
            "dataset_rows": len(rows),
            "num_epochs": 2,
            "expected_reward_calls": 8000,
            "reward_audit": runtime.audit_summary(),
            "reward_statistics": stats,
            "trainer_metrics": result.metrics,
            "optimization_audit": {
                "logged_steps": len(grad_values),
                "nonzero_gradient_steps": sum(value > 0 for value in grad_values),
                "min_grad_norm": min(grad_values),
                "max_grad_norm": max(grad_values),
            },
            "checkpoint_dir": str(checkpoint_dir),
            "official_test_accessed": False,
        }
        if resume is not None:
            summary["resume"] = resume
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
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from iar_grpo.numerics import configure_stable_sdpa

    config = load_grpo_config(config_path)
    domain, _, validation_file, _ = validate_screen_config(config)
    if step not in CHECKPOINTS[1:] or attempt <= 0:
        raise ValueError("Step 5 generation requires a trained checkpoint and positive attempt")
    run_root = _under(
        config.resolve(config.raw["paths"]["experiments"]) / run_id,
        STEP_ROOT,
        "run root",
    )
    training = _read_json(run_root / "summary.json")
    if training.get("status") != "complete":
        raise RuntimeError("Checkpoint generation cannot start before complete training")
    output_dir = (
        run_root
        / "evaluations"
        / "validation"
        / f"step_{step:04d}"
        / f"generation_attempt{attempt:03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    examples = read_jsonl(validation_file)
    tables = (
        Path(config.raw["evaluation"]["spider_tables_json"])
        if domain == "spider"
        else None
    )
    prompts = prompts_for_domain(domain, examples, tables_json=tables)["a0"]
    checkpoint_root = _under(
        config.resolve(config.raw["paths"]["checkpoints"]) / run_id,
        STEP_ROOT,
        "checkpoint root",
    )
    adapter = _adapter_for_step(checkpoint_root, step, 2000)
    if adapter is None:
        raise RuntimeError("Step 5 trained generation unexpectedly selected A0")
    random.seed(23)
    torch.manual_seed(23)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(23)
    sdpa = configure_stable_sdpa(torch)
    tokenizer = AutoTokenizer.from_pretrained(
        config.actor_source(),
        cache_dir=str(config.resolve(config.raw["paths"]["hf_cache"])),
        local_files_only=True,
        trust_remote_code=False,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = None
    batch_size = int(config.raw["evaluation"]["batch_size"])
    max_new_tokens = int(config.raw["evaluation"]["max_new_tokens"])
    candidates: list[dict[str, Any]] = []
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.actor_source(),
            cache_dir=str(config.resolve(config.raw["paths"]["hf_cache"])),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        model.eval()
        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start : start + batch_size]
            batch_prompts = prompts[start : start + batch_size]
            conversations = [
                [{"role": "user", "content": prompt}] for prompt in batch_prompts
            ]
            encoded = tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                padding_side="left",
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_length = encoded["input_ids"].shape[1]
            for item, prompt, output in zip(
                batch_examples, batch_prompts, generated, strict=True
            ):
                completion = output[prompt_length:]
                candidates.append(
                    {
                        "id": str(item["id"]),
                        "source_index": int(item["source_index"]),
                        "domain": domain,
                        "variant": "grpo",
                        "run_id": run_id,
                        "step": step,
                        "model_id": ACTOR_MODEL,
                        "model_revision": ACTOR_REVISION,
                        "adapter": str(adapter),
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "response": tokenizer.decode(
                            completion, skip_special_tokens=True
                        ).strip(),
                        "completion_tokens": len(completion),
                        "max_new_tokens": max_new_tokens,
                        "hit_max_new_tokens": len(completion) >= max_new_tokens,
                        "seed": 23,
                        "official_test_accessed": False,
                    }
                )
            write_jsonl(output_dir / "candidates.jsonl", candidates)
        if len(candidates) != 300 or len({row["id"] for row in candidates}) != 300:
            raise RuntimeError("Step 5 validation generation coverage is incomplete")
        summary = {
            "schema_version": 1,
            "status": "complete",
            "completed_at": utc_now(),
            "domain": domain,
            "run_id": run_id,
            "step": step,
            "attempt": attempt,
            "candidate_count": len(candidates),
            "candidate_file_sha256": sha256_file(output_dir / "candidates.jsonl"),
            "validation_file": str(validation_file),
            "validation_file_sha256": sha256_file(validation_file),
            "adapter": str(adapter),
            "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
            "truncated_count": sum(bool(row["hit_max_new_tokens"]) for row in candidates),
            "sdpa": sdpa,
            "official_test_accessed": False,
        }
        write_json(output_dir / "summary.json", summary)
        return summary
    except Exception as exc:
        write_json(
            output_dir / "failure.json",
            {
                "schema_version": 1,
                "status": "failed",
                "failed_at": utc_now(),
                "domain": domain,
                "run_id": run_id,
                "step": step,
                "attempt": attempt,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "partial_candidates": len(candidates),
                "official_test_accessed": False,
            },
        )
        raise
    finally:
        if model is not None:
            del model
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
    domain, _, validation_file, _ = validate_screen_config(config)
    run_root = _under(
        config.resolve(config.raw["paths"]["experiments"]) / run_id,
        STEP_ROOT,
        "run root",
    )
    step_root = run_root / "evaluations" / "validation" / f"step_{step:04d}"
    generation_dir = step_root / f"generation_attempt{generation_attempt:03d}"
    generation_summary = _read_json(generation_dir / "summary.json")
    candidates_path = generation_dir / "candidates.jsonl"
    if (
        generation_summary.get("status") != "complete"
        or generation_summary.get("domain") != domain
        or int(generation_summary.get("step", -1)) != step
        or generation_summary.get("candidate_file_sha256") != sha256_file(candidates_path)
    ):
        raise RuntimeError("Step 5 generation input is incomplete or drifted")
    output_dir = step_root / f"score_attempt{scoring_attempt:03d}"
    output_dir.mkdir(parents=True, exist_ok=False)
    examples = read_jsonl(validation_file)
    candidates = read_jsonl(candidates_path)
    by_id = {str(row["id"]): row for row in candidates}
    if len(by_id) != 300 or set(by_id) != {str(row["id"]) for row in examples}:
        raise RuntimeError("Step 5 generated candidates do not cover validation300")
    candidates = [by_id[str(row["id"])] for row in examples]
    scored_rows = candidates
    protocol: dict[str, Any] | None = None
    try:
        if domain in {"commonsenseqa", "gsm8k"}:
            _score_simple(domain, examples, candidates)
        elif domain == "kodcode":
            scored_rows, protocol = _score_kodcode_rows(
                examples,
                candidates,
                scoring_protocol_path=Path(
                    config.raw["evaluation"]["kodcode_scoring_protocol"]
                ),
                score_workers=score_workers,
                timeout_seconds=int(config.raw["evaluation"]["coding_timeout_seconds"]),
            )
        else:
            protocol = _score_spider_rows(
                examples,
                candidates,
                split_manifest_path=Path(
                    config.raw["evaluation"]["spider_split_manifest"]
                ),
                scoring_protocol_path=Path(
                    config.raw["evaluation"]["spider_scoring_protocol"]
                ),
                validation_file=validation_file,
                database_root=Path(config.raw["evaluation"]["spider_database_root"]),
                score_workers=score_workers,
            )
        write_jsonl(output_dir / "candidates.jsonl", candidates)
        correct = sum(bool(row["score"]["correct"]) for row in scored_rows)
        summary = {
            "schema_version": 1,
            "status": "complete",
            "completed_at": utc_now(),
            "domain": domain,
            "run_id": run_id,
            "step": step,
            "generation_attempt": generation_attempt,
            "scoring_attempt": scoring_attempt,
            "n": len(candidates),
            "scored_n": len(scored_rows),
            "correct": correct,
            "valid": sum(bool(row["score"].get("valid", True)) for row in scored_rows),
            "reference_invalid_excluded": len(candidates) - len(scored_rows),
            "task_utility": statistics.fmean(
                bool(row["score"]["correct"]) for row in scored_rows
            ),
            "status_counts": dict(
                sorted(
                    Counter(
                        str(row["score"].get("status", "scored"))
                        for row in scored_rows
                    ).items()
                )
            ),
            "candidate_file_sha256": sha256_file(output_dir / "candidates.jsonl"),
            "generation_summary": str(generation_dir / "summary.json"),
            "generation_summary_sha256": sha256_file(generation_dir / "summary.json"),
            "scoring_protocol": protocol,
            "selection_role": "validation_task_utility",
            "official_test_accessed": False,
        }
        write_json(output_dir / "summary.json", summary)
        return summary
    except Exception as exc:
        write_json(
            output_dir / "failure.json",
            {
                "schema_version": 1,
                "status": "failed",
                "failed_at": utc_now(),
                "domain": domain,
                "run_id": run_id,
                "step": step,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "official_test_accessed": False,
            },
        )
        raise


def collect_screen(*, step_root: Path, run_id: str, config_name: str) -> dict[str, Any]:
    step_root = _under(step_root, CAMPAIGN_ROOT, "Step 5 root")
    domains: dict[str, Any] = {}
    for domain in DOMAINS:
        config_path = step_root / "domains" / domain / "grpo" / "configs" / config_name
        config = load_grpo_config(config_path)
        validate_screen_config(config)
        run = step_root / "domains" / domain / "grpo" / "runs" / run_id
        training = _read_json(run / "summary.json")
        if training.get("status") != "complete":
            raise RuntimeError(f"Incomplete Step 5 training for {domain}")
        registry = _read_json(Path(config.raw["evaluation"]["baseline_registry"]))
        base = registry["baselines"][domain]
        evaluations = [
            {
                "step": 0,
                "task_utility": float(base["a0"]["task_utility"]),
                "summary": base["a0"]["summary"],
                "summary_sha256": base["a0"]["summary_sha256"],
            }
        ]
        for step in CHECKPOINTS[1:]:
            score_root = run / "evaluations" / "validation" / f"step_{step:04d}"
            complete: list[tuple[Path, dict[str, Any]]] = []
            for path in sorted(score_root.glob("score_attempt*/summary.json")):
                value = _read_json(path)
                if value.get("status") == "complete":
                    complete.append((path, value))
            if len(complete) != 1:
                raise RuntimeError(
                    f"Expected one complete Step 5 validation for {domain}/step{step}; "
                    f"found {len(complete)}"
                )
            path, value = complete[0]
            if value.get("status") != "complete" or int(value.get("n", -1)) != 300:
                raise RuntimeError(f"Incomplete Step 5 validation for {domain}/step{step}")
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
            "run_id": run_id,
            "reward_value": training["reward_value"],
            "training_reward_calls": int(training["expected_reward_calls"]),
            "a0_task_utility": float(base["a0"]["task_utility"]),
            "prompt_imitation_task_utility": float(
                base["prompt_imitation"]["task_utility"]
            ),
            "selected_step": int(selected["step"]),
            "selected_task_utility": float(selected["task_utility"]),
            "validation_gain_vs_a0": float(selected["task_utility"])
            - float(base["a0"]["task_utility"]),
            "evaluations": evaluations,
            "training_summary": str(run / "summary.json"),
            "training_summary_sha256": sha256_file(run / "summary.json"),
        }
    reward_values = {value["reward_value"] for value in domains.values()}
    if len(reward_values) != 1:
        raise RuntimeError("Step 5 domains used different reward transforms")
    result = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "protocol": "campaign-four-domain-screen-1p5b-v1",
        "run_id": run_id,
        "reward_value": next(iter(reward_values)),
        "domains": domains,
        "checkpoint_selection": "maximize validation task utility; earliest checkpoint on ties",
        "official_test_accessed": False,
        "mbpp": "FORBIDDEN",
    }
    destination = step_root / "results" / "four_domain_screen_1p5b.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite Step 5 result: {destination}")
    write_json(destination, result)
    return result
