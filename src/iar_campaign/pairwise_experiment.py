from __future__ import annotations

import gc
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx

from iar_campaign.four_domain_screen import (
    ACTOR_REVISION,
    CAMPAIGN_ROOT,
    DOMAINS,
    _adapter_for_step,
    _read_json,
    _under,
)
from iar_campaign.generate import prompts_for_domain, read_jsonl, sha256_text
from iar_campaign.judge_sanity import append_event, attribution_prompt, binary_mass_gate
from iar_campaign.pairwise_reward import (
    PAIRWISE_PROMPT_VERSION,
    PairwiseOpenRouterReward,
    normalize_response,
    pairwise_prompt,
    pairwise_prompt_template_sha256,
)
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
from iar_grpo.reward import settings_from_config
from iar_grpo.train import run_training
from iar_phase0.openrouter import (
    OPENROUTER_URL,
    binary_probability_from_logprobs,
    make_chat_payload,
    response_text,
)
from iar_phase0.parameters import initialize_parameter_record

STEP_ROOT = CAMPAIGN_ROOT / "09_pairwise_generate_distinguish"
STEP7_ROOT = CAMPAIGN_ROOT / "07_baseline_suite"
SELECTION = CAMPAIGN_ROOT / "06_two_domain_selection/results/two_domain_selection_v1.json"
PROTOCOL = STEP_ROOT / "configs/pairwise_protocol_v1.json"
ACTOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
JUDGE_MODEL = "deepseek/deepseek-v4-flash"
JUDGE_PROVIDER = "StreamLake"
CHECKPOINTS = (0, 500, 1000, 1500, 2000)
PROBE_GATE = STEP_ROOT / "results/pairwise_probe_gate_v1.json"


def validate_pairwise_config(
    config: GrpoTrialConfig,
) -> tuple[str, Path, Path, dict[str, Any]]:
    raw = config.raw
    campaign = raw.get("campaign") or {}
    domain = str(campaign.get("domain", ""))
    if domain not in DOMAINS or campaign.get("arm") != "grpo_pairwise":
        raise ValueError("Step 9 config has an unsupported domain or arm")
    serialized = json.dumps(raw, sort_keys=True).lower()
    if "mbpp" in serialized or "ifbench" in serialized or "ifeval" in serialized:
        raise ValueError("Forbidden benchmark in Step 9 config")
    selection = _read_json(SELECTION)
    if (
        sha256_file(SELECTION) != str(campaign["domain_selection_sha256"])
        or selection.get("status") != "complete"
        or selection.get("protocol") != "campaign-two-domain-selection-v1"
        or selection.get("official_test_accessed") is not False
        or domain not in selection.get("selected_domains", [])
        or selection.get("selected_domains", [None])[0] != "commonsenseqa"
    ):
        raise RuntimeError("Step 9 is not bound to the frozen domain selection")
    if sha256_file(PROTOCOL) != str(campaign["pairwise_protocol_sha256"]):
        raise RuntimeError("Step 9 pairwise protocol hash drifted")
    if raw["actor"].get("model_id") != ACTOR_MODEL or raw["actor"].get("revision") != ACTOR_REVISION:
        raise RuntimeError("Step 9 actor snapshot drifted")
    if config.prompt_count != 1000 or config.num_epochs != 2:
        raise ValueError("Step 9 full training requires train_core1000 for two epochs")
    if config.num_generations != 4 or config.expected_reward_calls != 8000:
        raise ValueError("Step 9 full training requires G=4 and 8000 pair rewards")
    training = raw["training"]
    if (
        int(training["max_steps"]) != 2000
        or int(training["per_device_train_batch_size"]) != 4
        or int(training["generation_batch_size"]) != 8
        or int(training["save_steps"]) != 500
        or float(training["learning_rate"]) != 3e-5
        or float(training["beta"]) != 0.003
        or float(training["temperature"]) != 0.6
        or float(training["top_p"]) != 1.0
        or str(training["loss_type"]) != "dr_grpo"
        or str(training["scale_rewards"]) != "group"
    ):
        raise ValueError("Step 9 GRPO hyperparameters drifted")
    reward = raw["reward"]
    discriminator = raw["discriminator"]
    routing = discriminator.get("provider_routing") or {}
    if (
        reward.get("value") != "log_p_pair"
        or int(reward["max_reward_calls"]) != 8000
        or int(reward["max_http_requests"]) != 80000
        or discriminator.get("model_id") != JUDGE_MODEL
        or discriminator.get("expected_provider") != JUDGE_PROVIDER
        or discriminator.get("attribution_prompt_version") != PAIRWISE_PROMPT_VERSION
        or discriminator.get("attribution_prompt_template_sha256")
        != pairwise_prompt_template_sha256()
        or routing.get("order") != [JUDGE_PROVIDER]
        or routing.get("allow_fallbacks") is not False
        or routing.get("require_parameters") is not True
    ):
        raise RuntimeError("Step 9 pairwise reward endpoint or budget drifted")
    train_file = Path(str(raw["dataset"]["split_file"]))
    validation_file = Path(str(raw["evaluation"]["validation_file"]))
    if (
        sha256_file(train_file) != str(raw["dataset"]["split_file_sha256"])
        or sha256_file(validation_file) != str(raw["evaluation"]["validation_file_sha256"])
    ):
        raise RuntimeError("Step 9 data hashes drifted")
    if any(part.lower() in {"test", "sealed", "final_test"} for part in validation_file.parts):
        raise ValueError("Step 9 cannot access official or sealed test data")
    expected_train = {
        "commonsenseqa": "train.jsonl",
        "gsm8k": "official_train.jsonl",
        "kodcode": "train_core1000.jsonl",
        "spider": "train_core1000.jsonl",
    }[domain]
    if train_file.name != expected_train or validation_file.name not in {
        "validation.jsonl",
        "validation300.jsonl",
    }:
        raise ValueError("Step 9 is not using frozen train1000/validation300")
    _under(
        config.resolve(raw["paths"]["experiments"]),
        STEP_ROOT,
        "Step 9 experiment root",
    )
    _under(
        config.resolve(raw["paths"]["checkpoints"]),
        STEP_ROOT,
        "Step 9 checkpoint root",
    )
    return domain, train_file, validation_file, selection


def _teacher_cache(
    config: GrpoTrialConfig,
    examples: list[dict[str, Any]],
    prompts: list[str],
) -> dict[str, str]:
    campaign = config.raw["campaign"]
    summary_path = Path(str(campaign["teacher_summary"]))
    summary = _read_json(summary_path)
    if (
        summary.get("status") != "complete"
        or summary.get("domain") != campaign["domain"]
        or int(summary.get("n", -1)) != 1000
        or summary.get("model_id") != JUDGE_MODEL
        or summary.get("provider") != JUDGE_PROVIDER
        or summary.get("official_test_accessed") is not False
    ):
        raise RuntimeError("Step 7 teacher corpus is incomplete")
    response_path = Path(str(summary["responses"]))
    if sha256_file(response_path) != str(summary["responses_sha256"]):
        raise RuntimeError("Step 7 teacher response hash drifted")
    rows = read_jsonl(response_path)
    by_id = {str(row["id"]): row for row in rows}
    if len(rows) != 1000 or len(by_id) != 1000:
        raise RuntimeError("Step 7 teacher response coverage drifted")
    result = {}
    for example, prompt in zip(examples, prompts, strict=True):
        row = by_id.get(str(example["id"]))
        if row is None or row.get("prompt") != prompt or not str(row.get("response", "")).strip():
            raise RuntimeError(f"Teacher response misaligned for {example['id']}")
        result[prompt] = str(row["response"])
    if len(result) != len(examples):
        raise RuntimeError("Task prompts are not unique in the teacher cache")
    return result


def build_training_dataset(config: GrpoTrialConfig):
    from datasets import Dataset

    domain, train_file, _, _ = validate_pairwise_config(config)
    examples = read_jsonl(train_file)
    if len(examples) != 1000 or len({str(row["id"]) for row in examples}) != 1000:
        raise RuntimeError("Step 9 train_core1000 is incomplete")
    tables = Path(str(config.raw["evaluation"]["spider_tables_json"])) if domain == "spider" else None
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
    return Dataset.from_list(rows), rows, examples, prompts


def run_training_attempt(
    *, config_path: Path, run_id: str, key_pool: Path, key_name: str, lock_dir: Path
) -> dict[str, Any]:
    config = load_grpo_config(config_path)
    domain, train_file, validation_file, _ = validate_pairwise_config(config)
    gate = _read_json(PROBE_GATE)
    if (
        gate.get("status") != "complete"
        or gate.get("domains", {}).get(domain, {}).get("decision") != "GO"
        or gate.get("official_test_accessed") is not False
    ):
        raise RuntimeError(f"Step 9 full training gate is not GO for {domain}")
    experiment_root = _under(
        config.resolve(config.raw["paths"]["experiments"]), STEP_ROOT, "Step 9 runs"
    )
    checkpoint_root = _under(
        config.resolve(config.raw["paths"]["checkpoints"]), STEP_ROOT, "Step 9 checkpoints"
    )
    experiment_dir = experiment_root / run_id
    checkpoint_dir = checkpoint_root / run_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    (experiment_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    initialize_parameter_record(experiment_dir, resolved_config=config.raw, source_config=config_path)
    dataset, rows, examples, prompts = build_training_dataset(config)
    write_jsonl(experiment_dir / "train_examples.jsonl", rows)
    teacher = _teacher_cache(config, examples, prompts)
    runtime = PairwiseOpenRouterReward(
        domain=domain,
        settings=settings_from_config(config.raw),
        teacher_by_task=teacher,
        log_dir=experiment_dir / "reward_calls",
        api_key=load_named_secret(key_pool, key_name),
    )
    write_json(
        experiment_dir / "campaign_inputs.json",
        {
            "domain": domain,
            "training_split": str(train_file),
            "training_split_sha256": sha256_file(train_file),
            "validation_split": str(validation_file),
            "validation_split_sha256": sha256_file(validation_file),
            "teacher_summary": config.raw["campaign"]["teacher_summary"],
            "teacher_response_count": len(teacher),
            "pairwise_protocol": str(PROTOCOL),
            "pairwise_protocol_sha256": sha256_file(PROTOCOL),
            "official_test_accessed": False,
        },
    )
    try:
        with key_lock(lock_dir, key_name):
            result, history = run_training(config, dataset, checkpoint_dir, experiment_dir, runtime)
        runtime.assert_complete()
        write_jsonl(experiment_dir / "trainer_log_history.jsonl", history)
        reward_stats = runtime.reward_statistics()
        expected_ids = {str(row["task_id"]) for row in rows}
        observed = reward_stats.get("by_task") or {}
        if set(observed) != expected_ids or any(int(value.get("n", -1)) != 8 for value in observed.values()):
            raise RuntimeError("Step 9 reward coverage differs from 1000 × G4 × 2")
        gradients = [float(row["grad_norm"]) for row in history if "grad_norm" in row]
        if not gradients or any(not math.isfinite(value) for value in gradients):
            raise RuntimeError("Step 9 gradients are missing or non-finite")
        summary = {
            "schema_version": 1,
            "status": "complete",
            "completed_at": utc_now(),
            "domain": domain,
            "run_id": run_id,
            "reward_value": "log_p_pair",
            "dataset_rows": len(rows),
            "expected_pair_rewards": 8000,
            "planned_judge_calls": 16000,
            "reward_audit": runtime.audit_summary(),
            "reward_statistics": reward_stats,
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
                "domain": domain,
                "run_id": run_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "reward_audit": runtime.audit_summary(),
                "official_test_accessed": False,
            },
        )
        raise


def validate_probe_config(config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = _read_json(config_path)
    domain = str(config.get("domain", ""))
    if domain not in DOMAINS:
        raise ValueError("Unsupported Step 9 probe domain")
    serialized = json.dumps(config, sort_keys=True).lower()
    if "mbpp" in serialized or "ifbench" in serialized or "ifeval" in serialized:
        raise ValueError("Forbidden benchmark in Step 9 probe config")
    selection = _read_json(SELECTION)
    protocol = _read_json(PROTOCOL)
    if (
        sha256_file(SELECTION) != str(config["domain_selection_sha256"])
        or selection.get("status") != "complete"
        or domain not in selection.get("selected_domains", [])
        or sha256_file(PROTOCOL) != str(config["pairwise_protocol_sha256"])
        or config.get("gate_thresholds") != protocol.get("gate_thresholds")
        or config.get("official_test_accessed") is not False
    ):
        raise RuntimeError("Step 9 probe is not bound to frozen inputs")
    probe = Path(str(config["probe_split"]))
    if sha256_file(probe) != str(config["probe_split_sha256"]):
        raise RuntimeError("Step 9 probe split hash drifted")
    rows = read_jsonl(probe)
    if len(rows) != 100 or len({str(row["id"]) for row in rows}) != 100:
        raise RuntimeError("Step 9 probe requires 100 unique IDs")
    output_root = _under(Path(str(config["output_root"])), STEP_ROOT, "Step 9 probe output")
    if domain not in output_root.parts:
        raise ValueError("Step 9 probe output domain mismatch")
    grpo_path = Path(str(config["grpo_config"]))
    if sha256_file(grpo_path) != str(config["grpo_config_sha256"]):
        raise RuntimeError("Step 9 GRPO config hash drifted")
    grpo = load_grpo_config(grpo_path)
    grpo_domain, train_file, _, _ = validate_pairwise_config(grpo)
    if grpo_domain != domain:
        raise RuntimeError("Step 9 probe and GRPO domains differ")
    train_ids = {str(row["id"]) for row in read_jsonl(train_file)}
    if not {str(row["id"]) for row in rows} <= train_ids:
        raise RuntimeError("Step 9 probe100 is not nested in train_core1000")
    return config, rows


def generate_probe_candidates(*, config_path: Path, run_id: str) -> dict[str, Any]:
    config, examples = validate_probe_config(config_path)
    grpo = load_grpo_config(Path(str(config["grpo_config"])))
    domain = str(config["domain"])
    output = Path(str(config["output_root"])) / "runs" / run_id / "generation_attempt001"
    output.mkdir(parents=True, exist_ok=False)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from iar_grpo.numerics import configure_stable_sdpa

    tables = Path(str(grpo.raw["evaluation"]["spider_tables_json"])) if domain == "spider" else None
    prompts = prompts_for_domain(domain, examples, tables_json=tables)["a0"]
    random.seed(23)
    torch.manual_seed(23)
    torch.cuda.manual_seed_all(23)
    sdpa = configure_stable_sdpa(torch)
    source = grpo.actor_source()
    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, trust_remote_code=False, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            source,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
        candidates: list[dict[str, Any]] = []
        batch_size = int(config["generation_batch_size"])
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
                    max_new_tokens=768,
                    do_sample=True,
                    temperature=0.6,
                    top_p=1.0,
                    num_return_sequences=4,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_length = encoded["input_ids"].shape[1]
            for example_index, example in enumerate(batch_examples):
                prompt = batch_prompts[example_index]
                for rollout in range(4):
                    generated_index = example_index * 4 + rollout
                    completion = generated[generated_index][prompt_length:]
                    response = normalize_response(
                        domain, tokenizer.decode(completion, skip_special_tokens=True)
                    )
                    candidates.append(
                        {
                            "logical_key": f"{domain}::{example['id']}::{rollout}",
                            "id": str(example["id"]),
                            "source_index": int(example["source_index"]),
                            "rollout_index": rollout,
                            "domain": domain,
                            "task_prompt": prompt,
                            "task_prompt_sha256": sha256_text(prompt),
                            "response": response,
                            "response_sha256": sha256_text(response),
                            "completion_tokens": len(completion),
                            "hit_max_new_tokens": len(completion) >= 768,
                        }
                    )
        if len(candidates) != 400 or len({row["logical_key"] for row in candidates}) != 400:
            raise RuntimeError("Step 9 probe generation coverage mismatch")
        write_jsonl(output / "candidates.jsonl", candidates)
        summary = {
            "schema_version": 1,
            "status": "complete",
            "completed_at": utc_now(),
            "domain": domain,
            "run_id": run_id,
            "n_prompts": 100,
            "rollouts_per_prompt": 4,
            "n_candidates": len(candidates),
            "truncated": sum(bool(row["hit_max_new_tokens"]) for row in candidates),
            "candidates": str(output / "candidates.jsonl"),
            "candidates_sha256": sha256_file(output / "candidates.jsonl"),
            "probe_split": config["probe_split"],
            "probe_split_sha256": config["probe_split_sha256"],
            "sdpa": sdpa,
            "official_test_accessed": False,
        }
        write_json(output / "summary.json", summary)
        return summary
    except Exception as exc:
        write_json(
            output / "failure.json",
            {
                "status": "failed",
                "failed_at": utc_now(),
                "domain": domain,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "official_test_accessed": False,
            },
        )
        raise
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _validated_judgment(raw: dict[str, Any]) -> dict[str, Any]:
    if raw.get("model") != JUDGE_MODEL or raw.get("provider") != JUDGE_PROVIDER:
        raise RuntimeError("misrouted_response")
    parsed = binary_probability_from_logprobs(raw)
    if not parsed.get("ok"):
        raise RuntimeError(f"judge_logprob_contract:{parsed.get('error')}")
    emitted = response_text(raw).strip()
    argmax = "1" if float(parsed["logp_1"]) > float(parsed["logp_0"]) else "0"
    return {
        "p_response_1": float(parsed["p1"]),
        "logp_0": float(parsed["logp_0"]),
        "logp_1": float(parsed["logp_1"]),
        "binary_label_mass": float(parsed["binary_label_mass"]),
        "judge_token": emitted if emitted in {"0", "1"} else None,
        "judge_argmax_token": argmax,
        "emitted_argmax_consistent": emitted == argmax,
        "raw_model": raw.get("model"),
        "raw_provider": raw.get("provider"),
        "usage": raw.get("usage") or {},
    }


def run_probe_judge(
    *,
    config_path: Path,
    run_id: str,
    key_pool: Path,
    key_name: str,
    lock_dir: Path,
) -> dict[str, Any]:
    config, examples = validate_probe_config(config_path)
    domain = str(config["domain"])
    generation = Path(str(config["output_root"])) / "runs" / run_id / "generation_attempt001"
    generation_summary = _read_json(generation / "summary.json")
    candidate_path = Path(str(generation_summary["candidates"]))
    if (
        generation_summary.get("status") != "complete"
        or sha256_file(candidate_path) != generation_summary.get("candidates_sha256")
    ):
        raise RuntimeError("Step 9 probe generation is incomplete")
    candidates = read_jsonl(candidate_path)
    grpo = load_grpo_config(Path(str(config["grpo_config"])))
    train_examples = read_jsonl(Path(str(grpo.raw["dataset"]["split_file"])))
    tables = Path(str(grpo.raw["evaluation"]["spider_tables_json"])) if domain == "spider" else None
    train_prompts = prompts_for_domain(domain, train_examples, tables_json=tables)["a0"]
    teacher_by_task = _teacher_cache(grpo, train_examples, train_prompts)
    probe_ids = {str(row["id"]) for row in examples}
    if {str(row["id"]) for row in candidates} != probe_ids:
        raise RuntimeError("Step 9 probe candidates and IDs are misaligned")
    tasks = []
    for candidate in candidates:
        task = str(candidate["task_prompt"])
        teacher = normalize_response(domain, teacher_by_task[task])
        actor = normalize_response(domain, str(candidate["response"]))
        for order in ("actor_first", "actor_second"):
            response_1, response_2 = (actor, teacher) if order == "actor_first" else (teacher, actor)
            prompt = pairwise_prompt(task, response_1, response_2)
            tasks.append(
                {
                    "logical_key": f"judge::{candidate['logical_key']}::{order}",
                    "candidate_key": candidate["logical_key"],
                    "id": str(candidate["id"]),
                    "source_index": int(candidate["source_index"]),
                    "rollout_index": int(candidate["rollout_index"]),
                    "order": order,
                    "candidate_sha256": candidate["response_sha256"],
                    "teacher_sha256": sha256_text(teacher),
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                }
            )
    if len(tasks) != 800 or len({row["logical_key"] for row in tasks}) != 800:
        raise RuntimeError("Step 9 probe judgment plan is incomplete")
    output = Path(str(config["output_root"])) / "runs" / run_id / "judge_attempt001"
    output.mkdir(parents=True, exist_ok=True)
    response_path = output / "responses.jsonl"
    event_path = output / "events.jsonl"
    receipt_path = output / "api_receipts.jsonl"
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    responses = read_jsonl(response_path) if response_path.is_file() else []
    done = {str(row["logical_key"]): row for row in responses}
    receipts_list = read_jsonl(receipt_path) if receipt_path.is_file() else []
    receipts = {str(row["logical_key"]): row for row in receipts_list}
    if len(done) != len(responses) or len(receipts) != len(receipts_list):
        raise RuntimeError("Step 9 probe contains duplicate responses or receipts")
    states: dict[str, str] = {}
    for event in read_jsonl(event_path) if event_path.is_file() else []:
        states[str(event["logical_key"])] = str(event["event"])
    ambiguous = sorted(
        key
        for key, state in states.items()
        if state in {"REQUEST_STARTED", "REQUEST_AMBIGUOUS"} and key not in done and key not in receipts
    )
    if ambiguous:
        raise RuntimeError(f"Fail-closed ambiguous Step 9 probe requests: {ambiguous[:3]}")
    task_map = {str(row["logical_key"]): row for row in tasks}
    if not set(done) <= set(task_map) or not set(receipts) <= set(task_map):
        raise RuntimeError("Step 9 probe resume state does not match the frozen plan")
    secret = load_named_secret(key_pool, key_name)
    headers = {
        "Authorization": f"Bearer {secret}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "IAR campaign Step 9 pairwise probe",
        "HTTP-Referer": "https://example.invalid/iar-code",
    }
    transport = config["transport"]
    started = time.time()
    http_requests = 0
    with key_lock(lock_dir, key_name), httpx.Client(timeout=float(transport["timeout_seconds"])) as client:
        last_request = 0.0
        for task in tasks:
            logical_key = str(task["logical_key"])
            if logical_key in done:
                continue
            if logical_key in receipts:
                receipt = receipts[logical_key]
                if receipt.get("prompt_sha256") != task["prompt_sha256"]:
                    raise RuntimeError(f"Step 9 receipt drift: {logical_key}")
                parsed = _validated_judgment(dict(receipt["raw_response"]))
                p_actor = parsed["p_response_1"] if task["order"] == "actor_first" else 1.0 - parsed["p_response_1"]
                row = {**{key: value for key, value in task.items() if key != "prompt"}, **parsed, "p_actor": p_actor, "recovered_from_receipt": True}
                responses.append(row)
                write_jsonl(response_path, responses)
                done[logical_key] = row
                append_event(event_path, {"event": "RESPONSE_RECOVERED_FROM_RECEIPT", "logical_key": logical_key, "time_unix": time.time()})
                continue
            payload = make_chat_payload(
                model_id=JUDGE_MODEL,
                prompt=str(task["prompt"]),
                system_prompt=None,
                max_tokens=1,
                temperature=0,
                logprobs=True,
                top_logprobs=5,
                provider_routing={"order": [JUDGE_PROVIDER], "require_parameters": True, "allow_fallbacks": False},
                request_overrides={"top_p": 1.0, "reasoning": {"enabled": False}},
            )
            last_error: str | None = None
            for attempt in range(1, int(transport["max_attempts_per_call"]) + 1):
                wait = float(transport["minimum_request_interval_seconds"]) - (time.monotonic() - last_request)
                if wait > 0:
                    time.sleep(wait)
                payload_hash = sha256_text(json.dumps(payload, sort_keys=True))
                append_event(event_path, {"event": "REQUEST_STARTED", "logical_key": logical_key, "attempt": attempt, "payload_sha256": payload_hash, "time_unix": time.time()})
                last_request = time.monotonic()
                http_requests += 1
                try:
                    response = client.post(OPENROUTER_URL, headers=headers, json=payload)
                except httpx.TransportError as exc:
                    append_event(event_path, {"event": "REQUEST_AMBIGUOUS", "logical_key": logical_key, "attempt": attempt, "error_type": type(exc).__name__, "time_unix": time.time()})
                    raise RuntimeError(f"Ambiguous Step 9 probe transport for {logical_key}") from exc
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"http_{response.status_code}"
                    append_event(event_path, {"event": "RETRYABLE_HTTP_RESPONSE", "logical_key": logical_key, "attempt": attempt, "status_code": response.status_code, "time_unix": time.time()})
                    if attempt < int(transport["max_attempts_per_call"]):
                        time.sleep(min(5.0 * 2 ** (attempt - 1), float(transport["max_retry_wait_seconds"])))
                        continue
                    break
                response.raise_for_status()
                try:
                    raw = response.json()
                    parsed = _validated_judgment(raw)
                except (ValueError, RuntimeError) as exc:
                    last_error = str(exc)
                    retryable = last_error == "misrouted_response" or last_error.startswith("judge_logprob_contract:missing_")
                    append_event(event_path, {"event": "INVALID_API_RESPONSE", "logical_key": logical_key, "attempt": attempt, "error": last_error, "retryable": retryable, "time_unix": time.time()})
                    if retryable and attempt < int(transport["max_attempts_per_call"]):
                        if "missing_binary_label_in_top_logprobs" in last_error:
                            payload = {**payload, "top_logprobs": 20}
                        time.sleep(2.0)
                        continue
                    break
                receipt = {"event": "API_RESPONSE_RECEIVED", "logical_key": logical_key, "attempt": attempt, "prompt_sha256": task["prompt_sha256"], "payload_sha256": payload_hash, "openrouter_response_id": raw.get("id"), "raw_response": raw, "time_unix": time.time()}
                append_event(receipt_path, receipt)
                receipts[logical_key] = receipt
                p_actor = parsed["p_response_1"] if task["order"] == "actor_first" else 1.0 - parsed["p_response_1"]
                row = {**{key: value for key, value in task.items() if key != "prompt"}, **parsed, "p_actor": p_actor, "request_top_logprobs": payload.get("top_logprobs"), "recovered_from_receipt": False}
                responses.append(row)
                write_jsonl(response_path, responses)
                done[logical_key] = row
                append_event(event_path, {"event": "REQUEST_SUCCEEDED", "logical_key": logical_key, "attempt": attempt, "openrouter_response_id": raw.get("id"), "time_unix": time.time()})
                last_error = None
                break
            if last_error is not None:
                raise RuntimeError(f"Step 9 probe request failed for {logical_key}: {last_error}")
    if set(done) != set(task_map):
        raise RuntimeError("Step 9 probe judge ended with incomplete coverage")
    responses.sort(key=lambda row: str(row["logical_key"]))
    write_jsonl(response_path, responses)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "domain": domain,
        "run_id": run_id,
        "logical_pair_rewards": 400,
        "successful_judge_calls": len(responses),
        "http_requests_this_process": http_requests,
        "responses": str(response_path),
        "responses_sha256": sha256_file(response_path),
        "events": str(event_path),
        "events_sha256": sha256_file(event_path),
        "api_receipts": str(receipt_path),
        "api_receipts_sha256": sha256_file(receipt_path),
        "elapsed_seconds": time.time() - started,
        "key_fingerprint": sha256_text(secret)[:16],
        "official_test_accessed": False,
    }
    write_json(summary_path, summary)
    return summary


def _bootstrap_interval(values: list[float], *, seed: int, replicates: int) -> list[float]:
    generator = random.Random(seed)
    samples = sorted(
        statistics.fmean(generator.choice(values) for _ in values)
        for _ in range(replicates)
    )
    return [samples[int(0.025 * replicates)], samples[min(replicates - 1, int(0.975 * replicates))]]


def collect_probe_gate(*, config_paths: list[Path], run_id: str) -> dict[str, Any]:
    per_domain: dict[str, Any] = {}
    passing = []
    for config_path in config_paths:
        config, _ = validate_probe_config(config_path)
        domain = str(config["domain"])
        judge_dir = Path(str(config["output_root"])) / "runs" / run_id / "judge_attempt001"
        summary = _read_json(judge_dir / "summary.json")
        responses = read_jsonl(Path(str(summary["responses"])))
        if summary.get("status") != "complete" or sha256_file(Path(str(summary["responses"]))) != summary.get("responses_sha256") or len(responses) != 800:
            raise RuntimeError(f"Incomplete Step 9 probe judge for {domain}")
        grouped: defaultdict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in responses:
            grouped[str(row["candidate_key"])][str(row["order"])] = row
        if len(grouped) != 400 or any(set(value) != {"actor_first", "actor_second"} for value in grouped.values()):
            raise RuntimeError(f"Step 9 two-order coverage mismatch for {domain}")
        pair_rows = []
        task_probabilities: defaultdict[str, list[float]] = defaultdict(list)
        for key, orders in sorted(grouped.items()):
            first = float(orders["actor_first"]["p_actor"])
            second = float(orders["actor_second"]["p_actor"])
            probability = (first + second) / 2.0
            pair_rows.append({"candidate_key": key, "id": orders["actor_first"]["id"], "p_actor_first": first, "p_actor_second": second, "p_pair": probability, "order_effect": first - second})
            task_probabilities[str(orders["actor_first"]["id"])].append(probability)
        probabilities = [float(row["p_pair"]) for row in pair_rows]
        effects = [float(row["order_effect"]) for row in pair_rows]
        ranges = [max(values) - min(values) for values in task_probabilities.values()]
        thresholds = config["gate_thresholds"]
        mass_gate = binary_mass_gate([float(row["binary_label_mass"]) for row in responses], thresholds)
        metrics = {
            "logical_pair_rewards": len(pair_rows),
            "judge_calls": len(responses),
            "mean_p_pair": statistics.fmean(probabilities),
            "std_p_pair": statistics.stdev(probabilities),
            "mean_order_effect": statistics.fmean(effects),
            "order_effect_bootstrap_95_interval": _bootstrap_interval(effects, seed=23 + DOMAINS.index(domain), replicates=int(thresholds["bootstrap_replicates"])),
            "fraction_groups_with_required_range": statistics.fmean(value >= float(thresholds["minimum_within_group_range"]) for value in ranges),
            "median_within_group_range": statistics.median(ranges),
            "emitted_argmax_consistency": statistics.fmean(bool(row["emitted_argmax_consistent"]) for row in responses),
            "binary_mass_gate": mass_gate,
            "usage": {
                key: sum(
                    float((row.get("usage") or {}).get(key, 0) or 0)
                    for row in responses
                )
                for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
            },
        }
        checks = {
            "complete_logical_pair_coverage": len(pair_rows) == int(thresholds["complete_logical_pair_coverage"]),
            "complete_judge_call_coverage": len(responses) == int(thresholds["complete_judge_call_coverage"]),
            "endpoint_pinned": {row["raw_model"] for row in responses} == {JUDGE_MODEL} and {row["raw_provider"] for row in responses} == {JUDGE_PROVIDER},
            "binary_label_support": bool(mass_gate["pass"]),
            "emitted_argmax_consistency": metrics["emitted_argmax_consistency"] >= float(thresholds["minimum_emitted_argmax_consistency"]),
            "nondegenerate_mean": float(thresholds["minimum_mean_actor_probability"]) <= metrics["mean_p_pair"] <= float(thresholds["maximum_mean_actor_probability"]),
            "nondegenerate_standard_deviation": metrics["std_p_pair"] >= float(thresholds["minimum_actor_probability_standard_deviation"]),
            "within_group_variation": metrics["fraction_groups_with_required_range"] >= float(thresholds["minimum_fraction_groups_with_range"]),
            "approximate_order_invariance": abs(metrics["mean_order_effect"]) <= float(thresholds["maximum_absolute_mean_order_effect"]),
            "no_ambiguous_requests": True,
        }
        decision = "GO" if all(checks.values()) else "NO_GO"
        if decision == "GO":
            passing.append(domain)
        per_domain[domain] = {"decision": decision, "checks": checks, "metrics": metrics, "judge_summary": str(judge_dir / "summary.json"), "judge_summary_sha256": sha256_file(judge_dir / "summary.json")}
        write_jsonl(STEP_ROOT / "results" / f"pairwise_probe_pairs_{domain}_v1.jsonl", pair_rows)
    selection = _read_json(SELECTION)
    selected = list(selection["selected_domains"])
    if set(per_domain) != set(selected):
        raise RuntimeError("Step 9 gate does not cover both selected domains")
    result = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "protocol": "campaign-pairwise-generate-distinguish-probe-gate-v1",
        "selected_domains": selected,
        "passing_domains": [domain for domain in selected if domain in passing],
        "domains": per_domain,
        "progression_rule": "each domain proceeds independently only when every precommitted check passes",
        "official_test_accessed": False,
        "mbpp": "FORBIDDEN",
        "ifbench": "DEFERRED",
    }
    destination = STEP_ROOT / "results/pairwise_probe_gate_v1.json"
    if destination.exists():
        raise FileExistsError(destination)
    write_json(destination, result)
    return result


def generate_validation_attempt(
    *, config_path: Path, run_id: str, step: int, attempt: int
) -> dict[str, Any]:
    if step not in CHECKPOINTS or attempt <= 0:
        raise ValueError("Invalid Step 9 validation checkpoint or attempt")
    config = load_grpo_config(config_path)
    domain, _, validation_file, _ = validate_pairwise_config(config)
    run_root = _under(
        config.resolve(config.raw["paths"]["experiments"]) / run_id,
        STEP_ROOT,
        "Step 9 run",
    )
    checkpoint_root = _under(
        config.resolve(config.raw["paths"]["checkpoints"]) / run_id,
        STEP_ROOT,
        "Step 9 checkpoint",
    )
    training = _read_json(run_root / "summary.json")
    if training.get("status") != "complete":
        raise RuntimeError("Step 9 validation cannot precede training completion")
    output = (
        run_root
        / "evaluations/validation"
        / f"step_{step:04d}"
        / f"generation_attempt{attempt:03d}"
    )
    output.mkdir(parents=True, exist_ok=False)
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
        candidates = []
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
            for example, generated_row in zip(batch_examples, generated, strict=True):
                completion = generated_row[prompt_length:]
                candidates.append(
                    {
                        "id": str(example["id"]),
                        "source_index": int(example["source_index"]),
                        "domain": domain,
                        "step": step,
                        "run_id": run_id,
                        "response": normalize_response(
                            domain,
                            tokenizer.decode(completion, skip_special_tokens=True),
                        ),
                        "completion_tokens": len(completion),
                        "hit_max_new_tokens": len(completion) >= max_tokens,
                    }
                )
        if len(candidates) != 300:
            raise RuntimeError("Step 9 validation generation is incomplete")
        write_jsonl(output / "candidates.jsonl", candidates)
        summary = {
            "schema_version": 1,
            "status": "complete",
            "completed_at": utc_now(),
            "domain": domain,
            "run_id": run_id,
            "step": step,
            "n": len(candidates),
            "truncated": sum(bool(row["hit_max_new_tokens"]) for row in candidates),
            "candidate_file_sha256": sha256_file(output / "candidates.jsonl"),
            "validation_file": str(validation_file),
            "validation_file_sha256": sha256_file(validation_file),
            "adapter": str(adapter) if adapter is not None else None,
            "sdpa": sdpa,
            "official_test_accessed": False,
        }
        write_json(output / "summary.json", summary)
        return summary
    except Exception as exc:
        write_json(
            output / "failure.json",
            {
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
    domain, _, validation_file, _ = validate_pairwise_config(config)
    run_root = _under(
        config.resolve(config.raw["paths"]["experiments"]) / run_id,
        STEP_ROOT,
        "Step 9 run",
    )
    generation = (
        run_root
        / "evaluations/validation"
        / f"step_{step:04d}"
        / f"generation_attempt{generation_attempt:03d}"
    )
    generation_summary = _read_json(generation / "summary.json")
    candidates = read_jsonl(generation / "candidates.jsonl")
    examples = read_jsonl(validation_file)
    if (
        generation_summary.get("status") != "complete"
        or generation_summary.get("official_test_accessed") is not False
        or generation_summary.get("candidate_file_sha256")
        != sha256_file(generation / "candidates.jsonl")
        or len(candidates) != 300
        or [str(row["id"]) for row in candidates]
        != [str(row["id"]) for row in examples]
    ):
        raise RuntimeError("Step 9 validation generation is incomplete or misaligned")
    output = (
        run_root
        / "evaluations/validation"
        / f"step_{step:04d}"
        / f"score_attempt{scoring_attempt:03d}"
    )
    output.mkdir(parents=True, exist_ok=False)
    try:
        scored = candidates
        scoring_protocol: dict[str, Any] | None = None
        if domain in {"commonsenseqa", "gsm8k"}:
            _score_simple(domain, examples, candidates)
        elif domain == "kodcode":
            scored, scoring_protocol = _score_kodcode_rows(
                examples,
                candidates,
                scoring_protocol_path=Path(
                    str(config.raw["evaluation"]["kodcode_scoring_protocol"])
                ),
                score_workers=score_workers,
                timeout_seconds=int(
                    config.raw["evaluation"]["coding_timeout_seconds"]
                ),
            )
        else:
            scoring_protocol = _score_spider_rows(
                examples,
                candidates,
                split_manifest_path=Path(
                    str(config.raw["evaluation"]["spider_split_manifest"])
                ),
                scoring_protocol_path=Path(
                    str(config.raw["evaluation"]["spider_scoring_protocol"])
                ),
                validation_file=validation_file,
                database_root=Path(
                    str(config.raw["evaluation"]["spider_database_root"])
                ),
                score_workers=score_workers,
            )
        write_jsonl(output / "candidates.jsonl", candidates)
        summary = {
            "schema_version": 1,
            "status": "complete",
            "completed_at": utc_now(),
            "domain": domain,
            "run_id": run_id,
            "step": step,
            "n": len(candidates),
            "scored_n": len(scored),
            "correct": sum(bool(row["score"]["correct"]) for row in scored),
            "valid": sum(bool(row["score"].get("valid", True)) for row in scored),
            "task_utility": statistics.fmean(
                bool(row["score"]["correct"]) for row in scored
            ),
            "status_counts": dict(
                sorted(
                    Counter(
                        str(row["score"].get("status", "scored")) for row in scored
                    ).items()
                )
            ),
            "candidate_file_sha256": sha256_file(output / "candidates.jsonl"),
            "generation_summary": str(generation / "summary.json"),
            "generation_summary_sha256": sha256_file(generation / "summary.json"),
            "scoring_protocol": scoring_protocol,
            "official_test_accessed": False,
        }
        write_json(output / "summary.json", summary)
        return summary
    except Exception as exc:
        write_json(
            output / "failure.json",
            {
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


def collect_task_validation(*, config_paths: list[Path], run_id: str) -> dict[str, Any]:
    gate = _read_json(PROBE_GATE)
    passing = list(gate.get("passing_domains") or [])
    configs: dict[str, GrpoTrialConfig] = {}
    for path in config_paths:
        config = load_grpo_config(path)
        domain, _, _, _ = validate_pairwise_config(config)
        configs[domain] = config
    if set(configs) != set(passing):
        raise RuntimeError("Step 9 task collector configs do not equal gate passing domains")
    screen = _read_json(
        CAMPAIGN_ROOT / "05_four_domain_screen_1p5b/results/four_domain_screen_1p5b.json"
    )
    domains = {}
    for domain in passing:
        config = configs[domain]
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
                    f"Expected one Step 9 score for {domain}/step{step}, found {len(complete)}"
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
            evaluations,
            key=lambda row: (float(row["task_utility"]), -int(row["step"])),
        )
        primary = screen["domains"][domain]
        domains[domain] = {
            "selected_step": int(selected["step"]),
            "selected_task_utility": float(selected["task_utility"]),
            "validation_gain_vs_a0": float(selected["task_utility"])
            - float(evaluations[0]["task_utility"]),
            "validation_gain_vs_primary_single_output": float(
                selected["task_utility"]
            )
            - float(primary["selected_task_utility"]),
            "primary_single_output_selected_step": int(primary["selected_step"]),
            "primary_single_output_task_utility": float(
                primary["selected_task_utility"]
            ),
            "evaluations": evaluations,
            "training_summary": str(run_root / "summary.json"),
            "training_summary_sha256": sha256_file(run_root / "summary.json"),
        }
    result = {
        "schema_version": 1,
        "status": "task_validation_complete_identity_readout_pending",
        "completed_at": utc_now(),
        "protocol": "campaign-pairwise-generate-distinguish-full-v1",
        "probe_gate": str(PROBE_GATE),
        "probe_gate_sha256": sha256_file(PROBE_GATE),
        "passing_domains": passing,
        "domains": domains,
        "checkpoint_selection": "maximize validation task utility; earliest checkpoint on ties",
        "official_test_accessed": False,
        "mbpp": "FORBIDDEN",
        "ifbench": "DEFERRED",
    }
    destination = STEP_ROOT / "results/pairwise_task_validation_v1.json"
    if destination.exists():
        raise FileExistsError(destination)
    write_json(destination, result)
    return result


def _selected_candidate_rows(
    evaluations: list[dict[str, Any]], selected_step: int
) -> tuple[list[dict[str, Any]], Path]:
    item = next(row for row in evaluations if int(row["step"]) == selected_step)
    summary_path = Path(str(item["summary"]))
    summary = _read_json(summary_path)
    candidate_path = summary_path.parent / "candidates.jsonl"
    if (
        summary.get("status") != "complete"
        or summary.get("official_test_accessed") is not False
        or summary.get("candidate_file_sha256") != sha256_file(candidate_path)
    ):
        raise RuntimeError(f"Selected validation candidates are invalid: {summary_path}")
    rows = read_jsonl(candidate_path)
    if len(rows) != 300:
        raise RuntimeError(f"Selected validation candidate count is not 300: {candidate_path}")
    return rows, summary_path


def run_identity_readout(
    *,
    config_path: Path,
    run_id: str,
    key_pool: Path,
    key_name: str,
    lock_dir: Path,
    attempt: int,
) -> dict[str, Any]:
    config = load_grpo_config(config_path)
    domain, _, validation_file, _ = validate_pairwise_config(config)
    task_result_path = STEP_ROOT / "results/pairwise_task_validation_v1.json"
    task_result = _read_json(task_result_path)
    if (
        task_result.get("status")
        != "task_validation_complete_identity_readout_pending"
        or domain not in task_result.get("passing_domains", [])
        or task_result.get("official_test_accessed") is not False
    ):
        raise RuntimeError(f"Step 9 identity readout is not enabled for {domain}")
    pair_info = task_result["domains"][domain]
    pairwise_rows, pairwise_summary = _selected_candidate_rows(
        pair_info["evaluations"], int(pair_info["selected_step"])
    )
    screen = _read_json(
        CAMPAIGN_ROOT / "05_four_domain_screen_1p5b/results/four_domain_screen_1p5b.json"
    )
    primary_info = screen["domains"][domain]
    primary_rows, primary_summary = _selected_candidate_rows(
        primary_info["evaluations"], int(primary_info["selected_step"])
    )
    examples = read_jsonl(validation_file)
    tables = (
        Path(str(config.raw["evaluation"]["spider_tables_json"]))
        if domain == "spider"
        else None
    )
    prompts = prompts_for_domain(domain, examples, tables_json=tables)["a0"]
    prompt_by_id = {
        str(example["id"]): prompt
        for example, prompt in zip(examples, prompts, strict=True)
    }
    if [str(row["id"]) for row in pairwise_rows] != [
        str(row["id"]) for row in primary_rows
    ] or set(prompt_by_id) != {str(row["id"]) for row in pairwise_rows}:
        raise RuntimeError("Step 9 pointwise identity candidate IDs are misaligned")
    tasks = []
    for method, candidates in (
        ("primary_single_output", primary_rows),
        ("pairwise", pairwise_rows),
    ):
        for candidate in candidates:
            row_id = str(candidate["id"])
            response = normalize_response(domain, str(candidate["response"]))
            prompt = attribution_prompt(domain, prompt_by_id[row_id], response)
            tasks.append(
                {
                    "logical_key": f"identity::{domain}::{method}::{row_id}",
                    "domain": domain,
                    "method": method,
                    "id": row_id,
                    "source_index": int(candidate["source_index"]),
                    "candidate_sha256": sha256_text(response),
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                }
            )
    output = STEP_ROOT / "domains" / domain / "identity_readout" / "runs" / run_id / f"attempt{attempt:03d}"
    output.mkdir(parents=True, exist_ok=True)
    response_path = output / "responses.jsonl"
    event_path = output / "events.jsonl"
    receipt_path = output / "api_receipts.jsonl"
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    responses = read_jsonl(response_path) if response_path.is_file() else []
    done = {str(row["logical_key"]): row for row in responses}
    receipt_rows = read_jsonl(receipt_path) if receipt_path.is_file() else []
    receipts = {str(row["logical_key"]): row for row in receipt_rows}
    if len(done) != len(responses) or len(receipts) != len(receipt_rows):
        raise RuntimeError("Step 9 identity readout has duplicate rows")
    states: dict[str, str] = {}
    for event in read_jsonl(event_path) if event_path.is_file() else []:
        states[str(event["logical_key"])] = str(event["event"])
    ambiguous = sorted(
        key
        for key, state in states.items()
        if state in {"REQUEST_STARTED", "REQUEST_AMBIGUOUS"}
        and key not in done
        and key not in receipts
    )
    if ambiguous:
        raise RuntimeError(f"Fail-closed ambiguous identity requests: {ambiguous[:3]}")
    task_map = {str(row["logical_key"]): row for row in tasks}
    if not set(done) <= set(task_map) or not set(receipts) <= set(task_map):
        raise RuntimeError("Step 9 identity resume state does not match its plan")
    reward = config.raw["reward"]
    secret = load_named_secret(key_pool, key_name)
    headers = {
        "Authorization": f"Bearer {secret}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "IAR campaign Step 9 pointwise identity readout",
        "HTTP-Referer": "https://example.invalid/iar-code",
    }
    started = time.time()
    http_requests = 0
    with key_lock(lock_dir, key_name), httpx.Client(timeout=float(reward["timeout_seconds"])) as client:
        last_request = 0.0
        for task in tasks:
            logical_key = str(task["logical_key"])
            if logical_key in done:
                continue
            if logical_key in receipts:
                receipt = receipts[logical_key]
                if receipt.get("prompt_sha256") != task["prompt_sha256"]:
                    raise RuntimeError(f"Identity receipt drift: {logical_key}")
                parsed = _validated_judgment(dict(receipt["raw_response"]))
                row = {
                    **{key: value for key, value in task.items() if key != "prompt"},
                    **parsed,
                    "p_own": parsed["p_response_1"],
                    "recovered_from_receipt": True,
                }
                responses.append(row)
                write_jsonl(response_path, responses)
                done[logical_key] = row
                append_event(
                    event_path,
                    {
                        "event": "RESPONSE_RECOVERED_FROM_RECEIPT",
                        "logical_key": logical_key,
                        "time_unix": time.time(),
                    },
                )
                continue
            payload = make_chat_payload(
                model_id=JUDGE_MODEL,
                prompt=str(task["prompt"]),
                system_prompt=None,
                max_tokens=1,
                temperature=0,
                logprobs=True,
                top_logprobs=5,
                provider_routing={
                    "order": [JUDGE_PROVIDER],
                    "require_parameters": True,
                    "allow_fallbacks": False,
                },
                request_overrides={"top_p": 1.0, "reasoning": {"enabled": False}},
            )
            last_error: str | None = None
            for request_attempt in range(1, int(reward["max_attempts_per_call"]) + 1):
                wait = float(reward["minimum_request_interval_seconds"]) - (
                    time.monotonic() - last_request
                )
                if wait > 0:
                    time.sleep(wait)
                payload_hash = sha256_text(json.dumps(payload, sort_keys=True))
                append_event(
                    event_path,
                    {
                        "event": "REQUEST_STARTED",
                        "logical_key": logical_key,
                        "attempt": request_attempt,
                        "payload_sha256": payload_hash,
                        "time_unix": time.time(),
                    },
                )
                last_request = time.monotonic()
                http_requests += 1
                try:
                    response = client.post(OPENROUTER_URL, headers=headers, json=payload)
                except httpx.TransportError as exc:
                    append_event(
                        event_path,
                        {
                            "event": "REQUEST_AMBIGUOUS",
                            "logical_key": logical_key,
                            "attempt": request_attempt,
                            "error_type": type(exc).__name__,
                            "time_unix": time.time(),
                        },
                    )
                    raise RuntimeError(
                        f"Ambiguous Step 9 identity transport for {logical_key}"
                    ) from exc
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"http_{response.status_code}"
                    append_event(
                        event_path,
                        {
                            "event": "RETRYABLE_HTTP_RESPONSE",
                            "logical_key": logical_key,
                            "attempt": request_attempt,
                            "status_code": response.status_code,
                            "time_unix": time.time(),
                        },
                    )
                    if request_attempt < int(reward["max_attempts_per_call"]):
                        time.sleep(
                            min(
                                5.0 * 2 ** (request_attempt - 1),
                                float(reward["max_retry_wait_seconds"]),
                            )
                        )
                        continue
                    break
                response.raise_for_status()
                try:
                    raw = response.json()
                    parsed = _validated_judgment(raw)
                except (ValueError, RuntimeError) as exc:
                    last_error = str(exc)
                    retryable = last_error == "misrouted_response" or last_error.startswith(
                        "judge_logprob_contract:missing_"
                    )
                    append_event(
                        event_path,
                        {
                            "event": "INVALID_API_RESPONSE",
                            "logical_key": logical_key,
                            "attempt": request_attempt,
                            "error": last_error,
                            "retryable": retryable,
                            "time_unix": time.time(),
                        },
                    )
                    if retryable and request_attempt < int(reward["max_attempts_per_call"]):
                        if "missing_binary_label_in_top_logprobs" in last_error:
                            payload = {**payload, "top_logprobs": 20}
                        time.sleep(2.0)
                        continue
                    break
                receipt = {
                    "event": "API_RESPONSE_RECEIVED",
                    "logical_key": logical_key,
                    "attempt": request_attempt,
                    "prompt_sha256": task["prompt_sha256"],
                    "payload_sha256": payload_hash,
                    "openrouter_response_id": raw.get("id"),
                    "raw_response": raw,
                    "time_unix": time.time(),
                }
                append_event(receipt_path, receipt)
                receipts[logical_key] = receipt
                row = {
                    **{key: value for key, value in task.items() if key != "prompt"},
                    **parsed,
                    "p_own": parsed["p_response_1"],
                    "request_top_logprobs": payload.get("top_logprobs"),
                    "recovered_from_receipt": False,
                }
                responses.append(row)
                write_jsonl(response_path, responses)
                done[logical_key] = row
                append_event(
                    event_path,
                    {
                        "event": "REQUEST_SUCCEEDED",
                        "logical_key": logical_key,
                        "attempt": request_attempt,
                        "openrouter_response_id": raw.get("id"),
                        "time_unix": time.time(),
                    },
                )
                last_error = None
                break
            if last_error is not None:
                raise RuntimeError(
                    f"Step 9 identity request failed for {logical_key}: {last_error}"
                )
    if set(done) != set(task_map):
        raise RuntimeError("Step 9 identity readout ended with incomplete coverage")
    responses.sort(key=lambda row: str(row["logical_key"]))
    write_jsonl(response_path, responses)
    methods = {
        method: [float(row["p_own"]) for row in responses if row["method"] == method]
        for method in ("primary_single_output", "pairwise")
    }
    summary = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "domain": domain,
        "run_id": run_id,
        "attempt": attempt,
        "calls": len(responses),
        "http_requests_this_process": http_requests,
        "mean_p_own": {
            method: statistics.fmean(values) for method, values in methods.items()
        },
        "delta_p_own_pairwise_minus_primary": statistics.fmean(methods["pairwise"])
        - statistics.fmean(methods["primary_single_output"]),
        "usage": {
            key: sum(
                float((row.get("usage") or {}).get(key, 0) or 0)
                for row in responses
            )
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
        },
        "primary_selected_summary": str(primary_summary),
        "primary_selected_summary_sha256": sha256_file(primary_summary),
        "pairwise_selected_summary": str(pairwise_summary),
        "pairwise_selected_summary_sha256": sha256_file(pairwise_summary),
        "responses": str(response_path),
        "responses_sha256": sha256_file(response_path),
        "events": str(event_path),
        "events_sha256": sha256_file(event_path),
        "api_receipts": str(receipt_path),
        "api_receipts_sha256": sha256_file(receipt_path),
        "elapsed_seconds": time.time() - started,
        "key_fingerprint": sha256_text(secret)[:16],
        "official_test_accessed": False,
    }
    write_json(summary_path, summary)
    return summary


def collect_final_result(
    *, config_paths: list[Path], run_id: str, identity_attempt: int
) -> dict[str, Any]:
    task_path = STEP_ROOT / "results/pairwise_task_validation_v1.json"
    task = _read_json(task_path)
    passing = list(task.get("passing_domains") or [])
    configs = {}
    for path in config_paths:
        config = load_grpo_config(path)
        domain, _, _, _ = validate_pairwise_config(config)
        configs[domain] = config
    if set(configs) != set(passing):
        raise RuntimeError("Step 9 final collector configs do not match passing domains")
    domains = {}
    for domain in passing:
        identity_summary = (
            STEP_ROOT
            / "domains"
            / domain
            / "identity_readout"
            / "runs"
            / run_id
            / f"attempt{identity_attempt:03d}/summary.json"
        )
        identity = _read_json(identity_summary)
        if identity.get("status") != "complete" or identity.get("official_test_accessed") is not False:
            raise RuntimeError(f"Step 9 identity readout incomplete for {domain}")
        domains[domain] = {
            **task["domains"][domain],
            "pointwise_identity": {
                "mean_p_own": identity["mean_p_own"],
                "delta_p_own_pairwise_minus_primary": identity[
                    "delta_p_own_pairwise_minus_primary"
                ],
                "summary": str(identity_summary),
                "summary_sha256": sha256_file(identity_summary),
            },
        }
    result = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "protocol": "campaign-pairwise-generate-distinguish-full-v1",
        "probe_gate": str(PROBE_GATE),
        "probe_gate_sha256": sha256_file(PROBE_GATE),
        "task_validation": str(task_path),
        "task_validation_sha256": sha256_file(task_path),
        "passing_domains": passing,
        "domains": domains,
        "checkpoint_selection": "maximize validation task utility; earliest checkpoint on ties",
        "official_test_accessed": False,
        "mbpp": "FORBIDDEN",
        "ifbench": "DEFERRED",
    }
    destination = STEP_ROOT / "results/pairwise_full_validation_v1.json"
    if destination.exists():
        raise FileExistsError(destination)
    write_json(destination, result)
    return result
