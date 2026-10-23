"""Executable, single-domain GPT-5 Mini K=10 closed-weight condition.

The teacher is queried as an opaque binary judge.  Every actor completion gets
ten independently billed judgments and the GRPO reward is log((m + 1) / 12).
This module deliberately has no logprob path and opens the sealed test only in
``evaluate_heldout`` after a terminal training summary exists.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import math
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
import yaml

from iar_campaign.generate import prompts_for_domain, read_jsonl
from iar_campaign.reward_pilot import key_lock, load_named_secret, sha256_file, utc_now, write_json, write_jsonl
from iar_closed_weight.gemini_pilot import (
    ONE_LABEL_SCHEMA,
    GeminiClosedWeightPilot,
    PilotSettings,
    ProtocolViolation,
    build_judge_prompt,
)
from iar_grpo.config import GrpoTrialConfig, load_grpo_config
from iar_phase0.generator import strip_code_fences
from iar_phase0.parameters import initialize_parameter_record

CAMPAIGN_ROOT = Path("/path/to/runs")
STEP_ROOT = CAMPAIGN_ROOT / "16_kodcode_gpt5mini_k10"
PROTOCOL = STEP_ROOT / "configs/gpt5mini_k10_protocol_v3.json"
# The campaign's published tier names differ from the historical paths embedded
# in the YAML itself; this is the canonical on-cluster source artifact.
BASE_CONFIG = CAMPAIGN_ROOT / "tier1_four_domain_screen_1p5b/domains/kodcode/grpo/configs/screen_prep001.yaml"
DATA_ROOT = CAMPAIGN_ROOT / "evaluation_audits_and_reproducibility/01_data_protocol_runtime_gates/datasets/kodcode/frozen_seed23_v1"
TRAIN_CONFIG = STEP_ROOT / "configs/gpt5mini_k10_train_v3.yaml"
CANARY_GATE = STEP_ROOT / "results/canary_attempt001.json"
MODEL_ID = "openai/gpt-5-mini"
PROVIDER = "OpenAI"
K = 10
JUDGE_TEMPERATURE = 1.0
MAX_LOGICAL_REWARDS = 8_000
MAX_PAID_JUDGMENTS = MAX_LOGICAL_REWARDS * K
HARD_COST_CAP_USD = 20.0
# Amendment transient-retry-v1: transient failures (HTTP 429/5xx, timeouts, transport
# errors, upstream 429/5xx inside a 200 body) are retried; all others stay terminal.
MAX_HTTP_ATTEMPTS = 8


def _run_id() -> str:
    """Resolve an immutable training-attempt name from the batch environment."""
    run_id = os.environ.get("IAR_GPT5MINI_RUN_ID", "attempt001")
    if not re.fullmatch(r"attempt[0-9]{3}", run_id):
        raise ValueError("IAR_GPT5MINI_RUN_ID must look like attempt002")
    return run_id


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _unresolved_errors(judgments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Error rows are admissible only as transient attempts whose call later got a receipt."""
    receipted = {str(row.get("call_id")) for row in judgments if row.get("status") == "ok"}
    return [
        row for row in judgments
        if (row.get("status") == "error" or row.get("billing_status") == "ambiguous")
        and not (row.get("transient") is True and str(row.get("call_id")) in receipted)
    ]


def protocol() -> dict[str, Any]:
    value = _read(PROTOCOL)
    endpoint = value.get("endpoint") or {}
    reward = value.get("reward") or {}
    if (
        value.get("protocol") != "campaign-kodcode-gpt5mini-k10-v3"
        or value.get("domain") != "kodcode"
        or endpoint.get("model_id") != MODEL_ID
        or endpoint.get("provider") != PROVIDER
        or endpoint.get("service_tier") != "flex"
        or endpoint.get("provider_routing") != {"only": ["openai"], "allow_fallbacks": False}
        or endpoint.get("logprobs_requested") is not False
        or reward.get("k") != K
        or reward.get("reward_value") != "log((m+1)/12)"
        or reward.get("maximum_http_attempts_per_physical_call") != MAX_HTTP_ATTEMPTS
        or (value.get("amendment") or {}).get("id") != "transient-retry-v1"
        or value.get("official_test_accessed_at_protocol_write") is not False
    ):
        raise RuntimeError("GPT-5 Mini K=10 protocol drifted")
    return value


def _settings(*, log_dir: Path, canary: bool) -> PilotSettings:
    # GeminiClosedWeightPilot is a transport/audit implementation despite its
    # historical name; all endpoint/model fields below are explicitly GPT-5 Mini.
    return PilotSettings(
        model_id=MODEL_ID, expected_response_provider=PROVIDER, service_tier="flex",
        provider_routing={"only": ["openai"], "allow_fallbacks": False},
        temperatures=(JUDGE_TEMPERATURE,), k_values=(K,), independent_batches=1,
        prompt_count=1, datasets=1, prompts_per_dataset=1, rollouts_per_prompt=1,
        max_tokens=256, top_p=1.0, reasoning={"effort": "minimal", "exclude": True},
        response_format=ONE_LABEL_SCHEMA, max_concurrency=2, timeout_seconds=180,
        max_attempts=MAX_HTTP_ATTEMPTS, max_retry_wait_seconds=60, minimum_label_rate=0.0,
        maximum_label_rate=1.0, minimum_variable_group_fraction=0.0,
        minimum_independent_batch_pearson=-1.0, signal_selection_rule="not_used",
        signal_tie_breaker="not_used", canary_candidates_per_dataset=1,
        canary_max_total_cost_usd=0.5, max_projected_full_cost_usd=HARD_COST_CAP_USD,
        full_hard_cost_cap_usd=HARD_COST_CAP_USD, max_single_request_cost_usd=0.05,
        output_dir=log_dir, registry_manifest=PROTOCOL,
        registry_manifest_sha256=sha256_file(PROTOCOL), canary_gate=CANARY_GATE,
        canary_gate_sha256="not_written", canary_execution_enabled=canary,
        full_execution_enabled=not canary,
    )


class Gpt5MiniK10Reward:
    def __init__(
        self,
        *,
        api_key: str,
        log_dir: Path,
        prior_judgments: list[dict[str, Any]] | None = None,
        prior_groups: list[dict[str, Any]] | None = None,
    ) -> None:
        if (prior_judgments is None) != (prior_groups is None):
            raise ValueError("Resume audit needs both judgment and group prefixes")
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=False)
        self.judgment_log = log_dir / "raw_judgments.jsonl"
        self.group_log = log_dir / "reward_groups.jsonl"
        prior_judgments = prior_judgments or []
        prior_groups = prior_groups or []
        if prior_judgments:
            write_jsonl(self.judgment_log, prior_judgments)
            write_jsonl(self.group_log, prior_groups)
        prior_cost = sum(
            float((row.get("usage") or {}).get("cost", 0.0))
            for row in prior_judgments
            if row.get("status") == "ok"
        )
        self.runner = GeminiClosedWeightPilot(
            settings=_settings(log_dir=log_dir, canary=False),
            api_key=api_key,
            log_path=self.judgment_log,
            initial_observed_cost_usd=prior_cost,
        )
        self.logical_calls = len(prior_groups)
        self._group_lock = asyncio.Lock()

    async def score_batch(self, *, completions: list[Any], task_statement: list[str], task_id: list[str], dataset_index: list[int], **_: Any) -> list[float]:
        from iar_grpo.reward import completion_to_text
        size = len(completions)
        if not (size == len(task_statement) == len(task_id) == len(dataset_index)):
            raise ValueError("Reward metadata length mismatch")
        if self.logical_calls + size > MAX_LOGICAL_REWARDS:
            raise RuntimeError("K=10 logical-reward budget exceeded")
        first = self.logical_calls
        self.logical_calls += size
        specs: list[list[dict[str, Any]]] = []
        rows: list[dict[str, Any]] = []
        per_task_rollout: defaultdict[str, int] = defaultdict(int)
        for offset, completion in enumerate(completions):
            task = str(task_id[offset])
            rollout = per_task_rollout[task]
            per_task_rollout[task] += 1
            candidate = strip_code_fences(completion_to_text(completion))
            candidate_id = hashlib.sha256(f"{first + offset + 1}|{task}|{rollout}|{candidate}".encode()).hexdigest()[:24]
            prompt = build_judge_prompt(str(task_statement[offset]), candidate)
            specs.append([
                {"call_id": hashlib.sha256(f"{candidate_id}|{replicate}".encode()).hexdigest()[:24], "logical_call_id": candidate_id,
                 "candidate_id": candidate_id, "task_id": task, "dataset_index": int(dataset_index[offset]), "rollout_index": rollout,
                 "replicate_index": replicate, "temperature": JUDGE_TEMPERATURE, "prompt": prompt}
                for replicate in range(K)
            ])
            rows.append({"logical_reward_id": first + offset + 1, "candidate_id": candidate_id, "task_id": task, "rollout_index": rollout,
                         "dataset_index": int(dataset_index[offset]), "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest()})
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=30.0, read=None, write=30.0, pool=60.0)) as client:
            results = await asyncio.gather(*[self.runner.judge_one(client, spec, phase="full") for group in specs for spec in group], return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise RuntimeError(f"K=10 judgment batch has {len(failures)} terminal failures") from failures[0]
        rewards: list[float] = []
        group_rows: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            labels = [int(value) for value in results[index * K:(index + 1) * K]]
            m = sum(labels)
            reward = math.log((m + 1) / (K + 2))
            rewards.append(reward)
            group_rows.append({**row, "k": K, "labels": labels, "m": m, "p_own": (m + 1) / (K + 2), "reward": reward, "status": "ok"})
        async with self._group_lock:
            with self.group_log.open("a", encoding="utf-8") as handle:
                for row in group_rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        return rewards

    def audit(self) -> dict[str, Any]:
        judgments = read_jsonl(self.judgment_log) if self.judgment_log.exists() else []
        groups = read_jsonl(self.group_log) if self.group_log.exists() else []
        receipts = [row for row in judgments if row.get("status") == "ok"]
        return {"logical_rewards": self.logical_calls, "reward_groups": len(groups), "successful_judgments": len(receipts),
                "observed_cost_usd": sum(float((row.get("usage") or {}).get("cost", 0.0)) for row in receipts),
                "ambiguous_errors": sum(row.get("billing_status") == "ambiguous" for row in judgments),
                "retried_transient_errors": sum(row.get("status") == "error" and row.get("transient") is True for row in judgments),
                "unresolved_errors": len(_unresolved_errors(judgments))}

    def assert_complete(self) -> None:
        audit = self.audit()
        if audit["logical_rewards"] != MAX_LOGICAL_REWARDS or audit["reward_groups"] != MAX_LOGICAL_REWARDS or audit["successful_judgments"] != MAX_PAID_JUDGMENTS or audit["unresolved_errors"]:
            raise RuntimeError(f"Incomplete or ambiguous K=10 reward audit: {audit}")


def prepare() -> dict[str, Any]:
    p = protocol()
    if TRAIN_CONFIG.exists():
        # Retry is allowed only after a scheduler-level failure before the
        # worker starts.  The generated config itself is immutable.
        raw = yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8"))
        if int(raw["discriminator"].get("judge_max_tokens", 0)) != 256:
            raw["discriminator"]["judge_max_tokens"] = 256
            TRAIN_CONFIG.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        return {"status": "PREPARED_NOT_LAUNCHED", "config": str(TRAIN_CONFIG), "config_sha256": sha256_file(TRAIN_CONFIG), "protocol_sha256": sha256_file(PROTOCOL), "official_test_accessed": False}
    raw = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    arm = STEP_ROOT / "training"
    raw["campaign"] = {"campaign_id": "campaign-kodcode-gpt5mini-k10-v2", "step": 16, "domain": "kodcode", "protocol": str(PROTOCOL), "protocol_sha256": sha256_file(PROTOCOL)}
    raw["paths"]["experiments"] = str(arm / "runs")
    raw["paths"]["checkpoints"] = str(arm / "checkpoints")
    raw["dataset"]["split_file"] = str(DATA_ROOT / "train_core1000.jsonl")
    raw["dataset"]["split_file_sha256"] = p["training"]["dataset"]["sha256"]
    raw["trial"]["run_name"] = "kodcode_gpt5mini_k10_seed23"
    raw["discriminator"] = {"model_id": MODEL_ID, "target_model_name": "GPT-5 Mini", "expected_provider": PROVIDER,
        "provider_routing": p["endpoint"]["provider_routing"], "request_overrides": {"service_tier": "flex", "reasoning": {"effort": "minimal", "exclude": True}},
        "temperature": JUDGE_TEMPERATURE, "judge_max_tokens": 256, "top_logprobs": 0, "strict_logprobs": False,
        "attribution_prompt_version": "phase0-pointwise-self-attribution-v2-no-system"}
    raw["reward"].update({"type": "closed_weight_repeated_binary", "value": "closed_weight_log_p_own_k10", "judgments_per_reward": K, "smoothing": "(m+1)/12", "max_reward_calls": MAX_LOGICAL_REWARDS, "max_http_requests": MAX_PAID_JUDGMENTS * MAX_HTTP_ATTEMPTS, "max_attempts_per_call": MAX_HTTP_ATTEMPTS, "max_concurrency": 2, "minimum_request_interval_seconds": 0.0, "timeout_seconds": 180, "max_retry_wait_seconds": 60, "store_raw_response": True})
    raw["evaluation"] = {"held_out_only": True, "held_out_split": p["held_out_evaluation"]["split"], "held_out_sha256": p["held_out_evaluation"]["sha256"], "held_out_count": 1000, "checkpoints": [0, 500, 1000, 1500, 2000], "checkpoints_to_score": [2000], "official_test_accessed": False}
    TRAIN_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_CONFIG.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    result = {"status": "PREPARED_NOT_LAUNCHED", "config": str(TRAIN_CONFIG), "config_sha256": sha256_file(TRAIN_CONFIG), "protocol_sha256": sha256_file(PROTOCOL), "official_test_accessed": False}
    write_json(STEP_ROOT / "manifests/prepared_attempt001.json", result)
    return result


def _load_train_config() -> GrpoTrialConfig:
    config = load_grpo_config(TRAIN_CONFIG)
    raw = config.raw
    if config.prompt_count != 1000 or config.num_epochs != 2 or config.num_generations != 4 or config.expected_reward_calls != MAX_LOGICAL_REWARDS or raw["reward"]["value"] != "closed_weight_log_p_own_k10" or int(raw["reward"]["max_http_requests"]) != MAX_PAID_JUDGMENTS * MAX_HTTP_ATTEMPTS or int(raw["reward"]["max_attempts_per_call"]) != MAX_HTTP_ATTEMPTS:
        raise RuntimeError("Training contract drifted")
    if sha256_file(Path(str(raw["dataset"]["split_file"]))) != protocol()["training"]["dataset"]["sha256"]:
        raise RuntimeError("Frozen KodCode train split drifted")
    return config


def run_canary(*, key_pool: Path, key_name: str) -> dict[str, Any]:
    if CANARY_GATE.exists():
        raise FileExistsError(CANARY_GATE)
    config = _load_train_config()
    example = read_jsonl(Path(str(config.raw["dataset"]["split_file"])))[0]
    task = prompts_for_domain("kodcode", [example], tables_json=None)["a0"][0]
    log_dir = STEP_ROOT / "canary/attempt001"
    runner = GeminiClosedWeightPilot(settings=_settings(log_dir=log_dir, canary=True), api_key=load_named_secret(key_pool, key_name), log_path=log_dir / "raw_judgments.jsonl")
    candidate = "def solve():\n    pass\n"
    async def execute() -> list[int]:
        async with httpx.AsyncClient(timeout=180) as client:
            return await asyncio.gather(*[runner.judge_one(client, {"call_id": f"canary-{i}", "logical_call_id": f"canary-{i}", "replicate_index": i, "temperature": JUDGE_TEMPERATURE, "prompt": build_judge_prompt(task, candidate)}, phase="canary") for i in range(K)])
    labels = asyncio.run(execute())
    rows = read_jsonl(log_dir / "raw_judgments.jsonl")
    receipts = [row for row in rows if row.get("status") == "ok"]
    costs = [float((row.get("usage") or {})["cost"]) for row in receipts]
    passed = len(labels) == K and len(receipts) == K and all(0 < cost <= 0.05 for cost in costs) and sum(costs) <= 0.5
    result = {"status": "GO" if passed else "NO_GO", "passed": passed, "completed_at": utc_now(), "model_id": MODEL_ID, "provider": PROVIDER, "service_tier": "flex", "k": K, "labels": labels, "observed_cost_usd": sum(costs), "projected_full_cost_usd": statistics.fmean(costs) * MAX_PAID_JUDGMENTS if costs else None, "receipt_count": len(receipts), "protocol_sha256": sha256_file(PROTOCOL), "official_test_accessed": False}
    write_json(CANARY_GATE, result)
    return result


def _resume_prefix(*, logical_rewards_per_step: int) -> dict[str, Any] | None:
    source_attempt = os.environ.get("IAR_GPT5MINI_RESUME_ATTEMPT")
    if source_attempt is None:
        return None
    if not re.fullmatch(r"attempt[0-9]{3}", source_attempt):
        raise ValueError("IAR_GPT5MINI_RESUME_ATTEMPT must look like attempt002")
    try:
        step = int(os.environ["IAR_GPT5MINI_RESUME_STEP"])
    except (KeyError, ValueError) as exc:
        raise ValueError("IAR_GPT5MINI_RESUME_STEP must be an integer") from exc
    if step <= 0:
        raise ValueError("Resume step must be positive")
    expected_groups = step * logical_rewards_per_step
    source_run = STEP_ROOT / "training/runs" / source_attempt
    source_checkpoint = STEP_ROOT / "training/checkpoints" / source_attempt / f"checkpoint-{step}"
    if not source_checkpoint.is_dir():
        raise FileNotFoundError(source_checkpoint)
    groups = read_jsonl(source_run / "reward_calls/reward_groups.jsonl")
    prefix = [row for row in groups if int(row.get("logical_reward_id", -1)) <= expected_groups]
    prefix.sort(key=lambda row: int(row["logical_reward_id"]))
    if [int(row["logical_reward_id"]) for row in prefix] != list(range(1, expected_groups + 1)):
        raise RuntimeError("Resume reward-group prefix is incomplete or reordered")
    if any(row.get("status") != "ok" or row.get("k") != K or len(row.get("labels", [])) != K for row in prefix):
        raise RuntimeError("Resume reward-group prefix violates the frozen K=10 audit")
    candidate_ids = {str(row["candidate_id"]) for row in prefix}
    if len(candidate_ids) != expected_groups:
        raise RuntimeError("Resume reward-group candidate IDs are not unique")
    judgments = [row for row in read_jsonl(source_run / "reward_calls/raw_judgments.jsonl") if str(row.get("candidate_id")) in candidate_ids]
    receipts = [row for row in judgments if row.get("status") == "ok"]
    if len(receipts) != expected_groups * K:
        raise RuntimeError("Resume judgment prefix has missing or duplicate receipts")
    if _unresolved_errors(judgments):
        raise RuntimeError("Resume judgment prefix contains an unresolved or non-transient error")
    return {
        "source_attempt": source_attempt,
        "source_checkpoint": source_checkpoint,
        "step": step,
        "logical_rewards": expected_groups,
        "prior_groups": prefix,
        "prior_judgments": judgments,
        "prior_observed_cost_usd": sum(float((row.get("usage") or {}).get("cost", 0.0)) for row in receipts),
    }


def train(*, key_pool: Path, key_name: str, lock_dir: Path) -> dict[str, Any]:
    # Importing the trainer also imports Torch/TRL.  Keep it out of the
    # prepare/submit and CPU-canary paths, which do not need a GPU runtime.
    from iar_grpo.train import run_training
    gate = _read(CANARY_GATE)
    if gate.get("passed") is not True:
        raise RuntimeError("Training is blocked until the GPT-5 Mini canary is GO")
    config = _load_train_config()
    run_id = _run_id()
    run_root = Path(str(config.raw["paths"]["experiments"])) / run_id
    checkpoint_root = Path(str(config.raw["paths"]["checkpoints"])) / run_id
    if run_root.exists() or checkpoint_root.exists():
        raise FileExistsError("Immutable training attempt already exists")
    run_root.mkdir(parents=True)
    checkpoint_root.mkdir(parents=True)
    (run_root / "config.yaml").write_text(TRAIN_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    initialize_parameter_record(run_root, resolved_config=config.raw, source_config=TRAIN_CONFIG)
    examples = read_jsonl(Path(str(config.raw["dataset"]["split_file"])))
    prompts = prompts_for_domain("kodcode", examples, tables_json=None)["a0"]
    from datasets import Dataset
    rows = [{"prompt": [{"role": "user", "content": prompt}], "task_statement": prompt, "task_id": str(example["id"]), "dataset_index": int(example["source_index"])} for example, prompt in zip(examples, prompts, strict=True)]
    write_jsonl(run_root / "train_examples.jsonl", rows)
    resume = _resume_prefix(logical_rewards_per_step=config.num_generations)
    if resume is not None:
        write_json(
            run_root / "resume_provenance.json",
            {key: (str(value) if isinstance(value, Path) else value) for key, value in resume.items() if key not in {"prior_groups", "prior_judgments"}},
        )
    runtime = Gpt5MiniK10Reward(
        api_key=load_named_secret(key_pool, key_name),
        log_dir=run_root / "reward_calls",
        prior_judgments=(resume or {}).get("prior_judgments"),
        prior_groups=(resume or {}).get("prior_groups"),
    )
    try:
        with key_lock(lock_dir, key_name):
            result, history = run_training(
                config,
                Dataset.from_list(rows),
                checkpoint_root,
                run_root,
                runtime,
                resume_from_checkpoint=(resume or {}).get("source_checkpoint"),
            )
        runtime.assert_complete()
        write_jsonl(run_root / "trainer_log_history.jsonl", history)
        summary = {"status": "complete", "completed_at": utc_now(), "run_id": run_id, "model_id": MODEL_ID, "provider": PROVIDER, "k": K, "reward": "log((m+1)/12)", "reward_audit": runtime.audit(), "trainer_metrics": result.metrics, "checkpoint_dir": str(checkpoint_root), "official_test_accessed": False}
        write_json(run_root / "summary.json", summary)
        return summary
    except Exception as exc:
        write_json(run_root / "failure.json", {"status": "failed", "failed_at": utc_now(), "error_type": type(exc).__name__, "error": str(exc), "reward_audit": runtime.audit(), "official_test_accessed": False})
        raise


def evaluate_heldout() -> dict[str, Any]:
    """Generate and score the sealed KodCode test exactly once, from step 2000."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from iar_campaign.score_baselines import _score_kodcode
    config = _load_train_config()
    run_id = _run_id()
    run_root = Path(str(config.raw["paths"]["experiments"])) / run_id
    checkpoint_root = Path(str(config.raw["paths"]["checkpoints"])) / run_id
    if _read(run_root / "summary.json").get("status") != "complete":
        raise RuntimeError("Held-out test is blocked until training is complete")
    p = protocol()["held_out_evaluation"]
    test_path = DATA_ROOT / "sealed/sealed_test1000.jsonl"
    if sha256_file(test_path) != p["sha256"]:
        raise RuntimeError("Sealed KodCode test hash drifted")
    output = run_root / "evaluations/sealed_test1000/step_2000"
    if output.exists():
        raise FileExistsError("Held-out evaluation is immutable and already exists")
    output.mkdir(parents=True)
    examples = read_jsonl(test_path)
    prompts = prompts_for_domain("kodcode", examples, tables_json=None)["a0"]
    tokenizer = AutoTokenizer.from_pretrained(config.actor_source(), local_files_only=True, trust_remote_code=False, padding_side="left")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(config.actor_source(), local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16, device_map="auto")
    adapter = checkpoint_root / "checkpoint-2000"
    if not adapter.is_dir(): raise FileNotFoundError(adapter)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    model.eval(); candidates = []
    try:
        for start in range(0, len(examples), 4):
            batch_examples, batch_prompts = examples[start:start+4], prompts[start:start+4]
            encoded = tokenizer.apply_chat_template([[{"role": "user", "content": q}] for q in batch_prompts], tokenize=True, add_generation_prompt=True, padding=True, return_tensors="pt", return_dict=True).to(model.device)
            with torch.inference_mode(): generated = model.generate(**encoded, max_new_tokens=768, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            prompt_length = encoded["input_ids"].shape[1]
            for example, tokens in zip(batch_examples, generated, strict=True): candidates.append({"id": str(example["id"]), "source_index": int(example["source_index"]), "response": tokenizer.decode(tokens[prompt_length:], skip_special_tokens=True).strip()})
    finally:
        del model, base
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    write_jsonl(output / "candidates.jsonl", candidates)
    payloads = [(row["response"], str(example["hidden_tests"]), 60) for row, example in zip(candidates, examples, strict=True)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool: scores = list(pool.map(_score_kodcode, payloads))
    for row, score in zip(candidates, scores, strict=True): row["score"] = score
    write_jsonl(output / "scored_candidates.jsonl", candidates)
    correct = sum(bool(row["score"]["correct"]) for row in candidates)
    result = {"status": "complete", "completed_at": utc_now(), "domain": "kodcode", "checkpoint": 2000, "n": len(candidates), "correct": correct, "task_utility": correct / len(candidates), "test_sha256": sha256_file(test_path), "candidate_sha256": sha256_file(output / "candidates.jsonl"), "scored_candidate_sha256": sha256_file(output / "scored_candidates.jsonl"), "official_test_accessed": True, "checkpoint_selection": "not permitted on held-out test"}
    write_json(output / "summary.json", result)
    return result
