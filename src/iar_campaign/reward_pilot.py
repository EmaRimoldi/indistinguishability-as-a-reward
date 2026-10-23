from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import gc
import hashlib
import json
import math
import os
import random
import re
import resource
import statistics
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from iar_campaign.generate import coding_plain, prompts_for_domain, read_jsonl
from iar_grpo.config import GrpoTrialConfig, load_grpo_config
from iar_grpo.reward import LiveOpenRouterReward, settings_from_config
from iar_phase0.generator import strip_code_fences
from iar_phase0.parameters import initialize_parameter_record, write_parameter_artifact
from iar_phase0.prompts import (
    pointwise_self_attribution_no_system_prompt_from_strings,
    pointwise_self_attribution_no_system_template_sha256,
)

CAMPAIGN_ROOT = Path(
    "/path/to/runs"
)
DOMAINS = ("gsm8k", "kodcode")
REWARD_VALUES = ("log_p_own", "p_own", "logit_p_own", "binary_own")
KEY_NAMES = (
    "OPENROUTER_API_KEY",
    "OPENROUTER_API_KEY_2",
    "OPENROUTER_API_KEY_3",
)


def reward_arm_slug(reward: str) -> str:
    if reward not in REWARD_VALUES:
        raise ValueError(f"Unknown reward transform: {reward}")
    return "binary" if reward == "binary_own" else reward


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def ensure_under(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root.expanduser().resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escaped the campaign root: {resolved}") from exc
    return resolved


def load_named_secret(path: Path, name: str) -> str:
    if re.fullmatch(r"OPENROUTER_API_KEY(?:_[2-9][0-9]*)?", name) is None:
        raise ValueError(f"Unsupported key lane: {name}")
    if os.environ.get(name, "").strip():
        return os.environ[name].strip()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == name:
            secret = value.strip().strip('"').strip("'")
            if not secret:
                raise ValueError(f"Empty credential: {name}")
            return secret
    raise KeyError(f"Missing credential: {name}")


@contextmanager
def key_lock(lock_dir: Path, key_name: str):
    lock_dir.mkdir(parents=True, exist_ok=True)
    handle = (lock_dir / f"{key_name}.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def validate_campaign_config(config: GrpoTrialConfig) -> tuple[str, Path, Path]:
    raw = config.raw
    campaign = raw.get("campaign") or {}
    domain = str(campaign.get("domain", ""))
    if domain not in DOMAINS:
        raise ValueError(f"Reward pilot supports only {DOMAINS}, found {domain!r}")
    if str(raw["reward"]["value"]) not in REWARD_VALUES:
        raise ValueError("Reward value is outside the frozen four-transform pilot")
    if config.prompt_count != 374 or config.num_epochs != 1:
        raise ValueError("The frozen reward pilot is exactly pilot374 for one epoch")
    if config.num_generations != 4 or config.expected_reward_calls != 1496:
        raise ValueError("The frozen pilot requires G=4 and exactly 1496 reward calls")
    if int(raw["training"]["max_steps"]) != 374:
        raise ValueError("The one-epoch pilot must contain exactly 374 optimizer steps")
    if [int(value) for value in raw["evaluation"]["checkpoints"]] != [0, 187, 374]:
        raise ValueError("The frozen pilot checkpoints are 0, 187, and 374")
    if int(raw["evaluation"]["validation_count"]) != 300:
        raise ValueError("Checkpoint selection requires the frozen validation300")
    if str(raw["evaluation"]["select_on"]) != "validation_task_utility":
        raise ValueError("Checkpoint selection must use validation task utility")
    if int(raw["evaluation"]["max_new_tokens"]) != 768:
        raise ValueError("Campaign generation length is frozen at 768 tokens")
    baseline_summary = ensure_under(
        Path(raw["evaluation"]["baseline_summary"]),
        CAMPAIGN_ROOT,
        "shared validation baseline",
    )
    if baseline_summary.name != "summary.json" or "shared_baseline" not in baseline_summary.parts:
        raise ValueError("Step 4 must use the registered shared validation baseline")
    if str(raw["actor"].get("revision", "")) != (
        "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    ):
        raise ValueError("Actor revision is not the frozen Qwen2.5-1.5B snapshot")
    split_file = ensure_under(Path(raw["dataset"]["split_file"]), CAMPAIGN_ROOT, "training split")
    validation_file = ensure_under(
        Path(raw["evaluation"]["validation_file"]), CAMPAIGN_ROOT, "validation split"
    )
    expected_parent = "frozen_historical_seed23" if domain == "gsm8k" else "frozen_seed23_v1"
    if split_file.name != "pilot374.jsonl" or split_file.parent.name != expected_parent:
        raise ValueError("Training input is not the frozen domain pilot374")
    if validation_file.name not in {"validation.jsonl", "validation300.jsonl"}:
        raise ValueError("Evaluation input is not the frozen validation300")
    if domain == "kodcode":
        protocol_path = ensure_under(
            Path(raw["evaluation"]["kodcode_scoring_protocol"]),
            CAMPAIGN_ROOT,
            "KodCode scoring protocol",
        )
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol.get("decision") != "GO":
            raise RuntimeError("KodCode reference-valid scoring protocol is not GO")
        if protocol.get("validation_file_sha256") != sha256_file(validation_file):
            raise RuntimeError("KodCode scoring protocol does not match validation300")
    forbidden_parts = {"test", "sealed", "final_test", "train_extension"}
    if any(
        part.lower() in forbidden_parts
        for path in (split_file, validation_file)
        for part in path.parts
    ):
        raise ValueError("A sealed or extension split was requested before final selection")
    return domain, split_file, validation_file


def build_training_dataset(config: GrpoTrialConfig):
    from datasets import Dataset

    domain, split_file, _ = validate_campaign_config(config)
    source = read_jsonl(split_file)
    if len(source) != 374 or len({str(row["id"]) for row in source}) != 374:
        raise RuntimeError("pilot374 has incomplete or duplicate coverage")
    if domain == "gsm8k":
        from iar_gsm8k.prompts import task_prompt

        tasks = [task_prompt(str(row["question"])) for row in source]
    else:
        tasks = [coding_plain(str(row["question"])) for row in source]
    rows = [
        {
            "prompt": [{"role": "user", "content": task}],
            "task_statement": task,
            "task_id": str(item["id"]),
            "dataset_index": int(item["source_index"]),
            "actor_visible_fields": ["question"],
        }
        for item, task in zip(source, tasks, strict=True)
    ]
    return Dataset.from_list(rows), rows


def reward_runtime(
    *, config: GrpoTrialConfig, domain: str, log_path: Path, api_key: str
) -> LiveOpenRouterReward:
    kwargs: dict[str, Any]
    if domain == "gsm8k":
        from iar_gsm8k.prompts import (
            ATTRIBUTION_PROMPT_VERSION,
            attribution_prompt,
            template_hashes,
        )

        kwargs = {
            "prompt_builder": attribution_prompt,
            "prompt_template_hash": template_hashes()[ATTRIBUTION_PROMPT_VERSION],
            "system_prompt_builder": lambda _name: None,
            "candidate_normalizer": str.strip,
        }
    else:
        kwargs = {
            "prompt_builder": pointwise_self_attribution_no_system_prompt_from_strings,
            "prompt_template_hash": pointwise_self_attribution_no_system_template_sha256(),
            "system_prompt_builder": lambda _name: None,
            "candidate_normalizer": strip_code_fences,
        }
    return LiveOpenRouterReward(
        settings=settings_from_config(config.raw),
        log_path=log_path,
        api_key=api_key,
        **kwargs,
    )


def run_training_attempt(
    *, config_path: Path, run_id: str, key_pool: Path, key_name: str, lock_dir: Path
) -> dict[str, Any]:
    from iar_grpo.train import run_training

    config = load_grpo_config(config_path)
    domain, split_file, validation_file = validate_campaign_config(config)
    experiment_root = ensure_under(config.resolve(config.raw["paths"]["experiments"]), CAMPAIGN_ROOT, "run root")
    checkpoint_root = ensure_under(config.resolve(config.raw["paths"]["checkpoints"]), CAMPAIGN_ROOT, "checkpoint root")
    experiment_dir = experiment_root / run_id
    checkpoint_dir = checkpoint_root / run_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    (experiment_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    initialize_parameter_record(experiment_dir, resolved_config=config.raw, source_config=config_path)
    dataset, rows = build_training_dataset(config)
    write_jsonl(experiment_dir / "train_examples.jsonl", rows)
    write_parameter_artifact(
        experiment_dir,
        "campaign_inputs.json",
        {
            "domain": domain,
            "training_split": str(split_file),
            "training_split_sha256": sha256_file(split_file),
            "validation_split": str(validation_file),
            "validation_split_sha256": sha256_file(validation_file),
            "official_test_accessed": False,
        },
    )
    secret = load_named_secret(key_pool, key_name)
    runtime = reward_runtime(
        config=config,
        domain=domain,
        log_path=experiment_dir / "reward_calls.jsonl",
        api_key=secret,
    )
    try:
        with key_lock(lock_dir, key_name):
            result, history = run_training(config, dataset, checkpoint_dir, experiment_dir, runtime)
        runtime.assert_complete()
        write_jsonl(experiment_dir / "trainer_log_history.jsonl", history)
        reward_statistics = runtime.reward_statistics()
        observed = reward_statistics.get("by_task") or {}
        expected_ids = {str(row["task_id"]) for row in rows}
        if set(observed) != expected_ids:
            raise RuntimeError("Training reward coverage differs from pilot374")
        if any(int(values.get("n", -1)) != 4 for values in observed.values()):
            raise RuntimeError("At least one training prompt lacks its complete G=4 group")
        gradients = [row for row in history if "grad_norm" in row]
        if not gradients or any(not _finite(float(row["grad_norm"])) for row in gradients):
            raise RuntimeError("Training gradient audit is incomplete or non-finite")
        summary = {
            "schema_version": 1,
            "status": "complete",
            "completed_at": utc_now(),
            "run_id": run_id,
            "domain": domain,
            "reward_value": config.raw["reward"]["value"],
            "dataset_rows": len(rows),
            "expected_reward_calls": config.expected_reward_calls,
            "reward_audit": runtime.audit_summary(),
            "reward_statistics": reward_statistics,
            "trainer_metrics": result.metrics,
            "optimization_audit": {
                "logged_steps": len(gradients),
                "nonzero_gradient_steps": sum(float(row["grad_norm"]) > 0 for row in gradients),
                "min_grad_norm": min(float(row["grad_norm"]) for row in gradients),
                "max_grad_norm": max(float(row["grad_norm"]) for row in gradients),
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
                "reward_value": config.raw["reward"]["value"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "reward_audit": runtime.audit_summary(),
                "official_test_accessed": False,
            },
        )
        raise


def _finite(value: float) -> bool:
    return math.isfinite(value)


def _adapter_for_step(checkpoint_root: Path, step: int, final_step: int) -> Path | None:
    if step == 0:
        return None
    path = checkpoint_root / ("final_adapter" if step == final_step else f"checkpoint-{step}")
    if not (path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Missing LoRA adapter: {path}")
    return path


def _limit_child() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    memory = 16 * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    file_size = 128 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_size, file_size))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _score_kodcode(payload: tuple[str, str, int]) -> dict[str, Any]:
    response, hidden_tests, timeout_seconds = payload
    code = strip_code_fences(response)
    with tempfile.TemporaryDirectory(prefix="iar_kodcode_") as temporary:
        solution_path = Path(temporary) / "solution.py"
        solution_path.write_text(code.rstrip() + "\n", encoding="utf-8")
        tests = hidden_tests.rstrip()
        if "from solution import *" not in tests:
            tests = "from solution import *  # noqa: F403\n\n" + tests
        test_path = Path(temporary) / "test_solution.py"
        test_path.write_text(tests + "\n", encoding="utf-8")
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.environ.get("IAR_KODCODE_TEST_PYTHONPATH", ""),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
        }
        test_python = os.environ.get("IAR_KODCODE_TEST_PYTHON", sys.executable)
        try:
            completed = subprocess.run(
                [test_python, "-m", "pytest", "-q", str(test_path)],
                cwd=temporary,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                preexec_fn=_limit_child if sys.platform.startswith("linux") else None,
                check=False,
            )
            return {
                "correct": completed.returncode == 0,
                "valid": completed.returncode in {0, 1},
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
                "timed_out": False,
                "extracted_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            }
        except subprocess.TimeoutExpired:
            return {
                "correct": False,
                "valid": False,
                "returncode": -2,
                "stdout_tail": "",
                "stderr_tail": f"Timeout after {timeout_seconds}s",
                "timed_out": True,
                "extracted_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            }


def _generate_validation(
    *, config: GrpoTrialConfig, run_id: str, step: int, output_dir: Path
) -> list[dict[str, Any]]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from iar_grpo.numerics import configure_stable_sdpa

    domain, _, validation_file = validate_campaign_config(config)
    examples = read_jsonl(validation_file)
    if len(examples) != 300 or len({str(row["id"]) for row in examples}) != 300:
        raise RuntimeError("Frozen validation split is not unique validation300")
    prompts = prompts_for_domain(domain, examples, tables_json=None)["a0"]
    actor = config.raw["actor"]
    source = config.actor_source()
    cache = config.resolve(config.raw["paths"]["hf_cache"])
    checkpoint_root = ensure_under(
        config.resolve(config.raw["paths"]["checkpoints"]) / run_id,
        CAMPAIGN_ROOT,
        "checkpoint root",
    )
    adapter = _adapter_for_step(
        checkpoint_root, step, int(config.raw["training"]["max_steps"])
    )
    random.seed(int(config.raw["project"]["seed"]))
    torch.manual_seed(int(config.raw["project"]["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.raw["project"]["seed"]))
    configure_stable_sdpa(torch)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        cache_dir=str(cache),
        local_files_only=bool(actor["local_files_only"]),
        trust_remote_code=False,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        source,
        cache_dir=str(cache),
        local_files_only=bool(actor["local_files_only"]),
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    output_path = output_dir / "candidates.jsonl"
    existing = read_jsonl(output_path) if output_path.exists() else []
    by_id = {str(row["id"]): row for row in existing}
    planned_ids = {str(row["id"]) for row in examples}
    if len(by_id) != len(existing) or not set(by_id) <= planned_ids:
        raise RuntimeError("Existing validation ledger is duplicate or belongs to another split")
    batch_size = int(config.raw["evaluation"]["batch_size"])
    max_new_tokens = int(config.raw["evaluation"]["max_new_tokens"])
    pending = [
        (row, prompt)
        for row, prompt in zip(examples, prompts, strict=True)
        if str(row["id"]) not in by_id
    ]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        conversations = [[{"role": "user", "content": prompt}] for _, prompt in batch]
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
        for (item, prompt), output in zip(batch, generated, strict=True):
            completion = output[prompt_length:]
            response = tokenizer.decode(completion, skip_special_tokens=True).strip()
            row = {
                "id": str(item["id"]),
                "source_index": int(item["source_index"]),
                "domain": domain,
                "step": step,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "response": response,
                "completion_tokens": len(completion),
                "hit_max_new_tokens": len(completion) >= max_new_tokens,
                "official_test_accessed": False,
            }
            if domain == "gsm8k":
                from iar_gsm8k.scoring import score_response

                row["score"] = score_response(response, str(item["answer_value"]))
            by_id[row["id"]] = row
        write_jsonl(output_path, [by_id[key] for key in sorted(by_id)])
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [by_id[str(row["id"])] for row in examples]


def run_validation_attempt(
    *, config_path: Path, run_id: str, step: int, score_workers: int
) -> dict[str, Any]:
    config = load_grpo_config(config_path)
    domain, _, validation_file = validate_campaign_config(config)
    run_root = ensure_under(
        config.resolve(config.raw["paths"]["experiments"]) / run_id,
        CAMPAIGN_ROOT,
        "run root",
    )
    if step > 0:
        training_summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
        if training_summary.get("status") != "complete":
            raise RuntimeError("Validation cannot start before complete training")
    output_dir = run_root / "evaluations" / "validation" / f"step_{step:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _generate_validation(config=config, run_id=run_id, step=step, output_dir=output_dir)
    scored_rows = rows
    scoring_protocol_path: Path | None = None
    if domain == "kodcode":
        examples = {str(row["id"]): row for row in read_jsonl(validation_file)}
        scoring_protocol_path = Path(
            config.raw["evaluation"]["kodcode_scoring_protocol"]
        ).resolve()
        scoring_protocol = json.loads(scoring_protocol_path.read_text(encoding="utf-8"))
        scored_ids = {str(value) for value in scoring_protocol["scored_ids"]}
        if (
            not scored_ids
            or not scored_ids <= set(examples)
            or len(scored_ids) != int(scoring_protocol["scored_n"])
        ):
            raise RuntimeError("KodCode scoring protocol has invalid scored IDs")
        scored_rows = [row for row in rows if str(row["id"]) in scored_ids]
        payloads = [
            (
                str(row["response"]),
                str(examples[str(row["id"])]["hidden_tests"]),
                int(config.raw["evaluation"]["coding_timeout_seconds"]),
            )
            for row in scored_rows
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=score_workers) as pool:
            scores = list(pool.map(_score_kodcode, payloads))
        for row, score in zip(scored_rows, scores, strict=True):
            row["score"] = score
        for row in rows:
            if str(row["id"]) not in scored_ids:
                row["score"] = {
                    "correct": False,
                    "valid": False,
                    "excluded_reference_invalid": True,
                    "reason": "reference_solution_failed_frozen_hidden_tests",
                }
        write_jsonl(output_dir / "candidates.jsonl", rows)
    accuracy = statistics.fmean(
        bool(row["score"]["correct"]) for row in scored_rows
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "domain": domain,
        "run_id": run_id,
        "step": step,
        "n": len(rows),
        "scored_n": len(scored_rows),
        "correct": sum(bool(row["score"]["correct"]) for row in scored_rows),
        "valid": sum(bool(row["score"]["valid"]) for row in scored_rows),
        "reference_invalid_excluded": len(rows) - len(scored_rows),
        "task_utility": accuracy,
        "candidate_file_sha256": sha256_file(output_dir / "candidates.jsonl"),
        "validation_file": str(validation_file),
        "validation_file_sha256": sha256_file(validation_file),
        "selection_role": "validation_task_utility",
        "kodcode_scoring_protocol": (
            str(scoring_protocol_path) if scoring_protocol_path is not None else None
        ),
        "kodcode_scoring_protocol_sha256": (
            sha256_file(scoring_protocol_path)
            if scoring_protocol_path is not None
            else None
        ),
        "official_test_accessed": False,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def task_outcomes(summary_path: Path) -> dict[str, bool]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates_path = summary_path.with_name("candidates.jsonl")
    if sha256_file(candidates_path) != summary.get("candidate_file_sha256"):
        raise RuntimeError(f"Validation candidate hash mismatch: {candidates_path}")
    rows = read_jsonl(candidates_path)
    outcomes = {
        str(row["id"]): bool(row["score"]["correct"])
        for row in rows
        if not bool(row["score"].get("excluded_reference_invalid", False))
    }
    if len(outcomes) != int(summary.get("scored_n", summary["n"])):
        raise RuntimeError(f"Validation outcome coverage mismatch: {summary_path}")
    return outcomes


def bootstrap_macro_gains(
    per_example: dict[str, dict[str, list[float]]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, list[float]]:
    if replicates <= 1:
        raise ValueError("Bootstrap requires at least two replicates")
    rewards = tuple(per_example)
    if set(rewards) != set(REWARD_VALUES):
        raise ValueError("Bootstrap requires all four reward transforms")
    domains = tuple(per_example[rewards[0]])
    if set(domains) != set(DOMAINS):
        raise ValueError("Bootstrap requires both reward-pilot domains")
    sizes = {domain: len(per_example[rewards[0]][domain]) for domain in domains}
    if any(size <= 1 for size in sizes.values()):
        raise ValueError("Each bootstrap domain requires at least two paired outcomes")
    for reward in rewards:
        if tuple(per_example[reward]) != domains:
            raise ValueError("Bootstrap domain order changed between transforms")
        if any(len(per_example[reward][domain]) != sizes[domain] for domain in domains):
            raise ValueError("Bootstrap paired-outcome coverage differs between transforms")
    generator = random.Random(seed)
    samples = {reward: [] for reward in rewards}
    for _ in range(replicates):
        indices = {
            domain: [generator.randrange(sizes[domain]) for _ in range(sizes[domain])]
            for domain in domains
        }
        for reward in rewards:
            domain_means = [
                statistics.fmean(
                    per_example[reward][domain][index]
                    for index in indices[domain]
                )
                for domain in domains
            ]
            samples[reward].append(statistics.fmean(domain_means))
    return samples


def select_reward_with_logp_tie_rule(
    observed: dict[str, float], bootstrap: dict[str, list[float]]
) -> dict[str, Any]:
    raw_winner = max(
        REWARD_VALUES,
        key=lambda reward: (observed[reward], -REWARD_VALUES.index(reward)),
    )
    logp = "log_p_own"
    differences = [
        left - right
        for left, right in zip(bootstrap[raw_winner], bootstrap[logp], strict=True)
    ]
    difference_se = statistics.stdev(differences)
    observed_gap = observed[raw_winner] - observed[logp]
    within_one_se = observed_gap <= difference_se
    selected = logp if raw_winner != logp and within_one_se else raw_winner
    return {
        "selected_reward_transform": selected,
        "raw_best_reward_transform": raw_winner,
        "raw_best_minus_log_p_own": observed_gap,
        "bootstrap_se_raw_best_minus_log_p_own": difference_se,
        "raw_best_and_log_p_own_within_one_bootstrap_se": within_one_se,
        "tie_rule_applied": selected == logp and raw_winner != logp,
    }


def collect_reward_pilot(step_root: Path, config_name: str) -> dict[str, Any]:
    step_root = ensure_under(step_root, CAMPAIGN_ROOT, "Step 4 root")
    domains: dict[str, Any] = {}
    macro: dict[str, Any] = {}
    per_example: dict[str, dict[str, list[float]]] = {
        reward: {} for reward in REWARD_VALUES
    }
    for domain in DOMAINS:
        domains[domain] = {}
        for reward in REWARD_VALUES:
            arm = step_root / "arms" / domain / reward_arm_slug(reward)
            config_path = arm / "configs" / config_name
            config = load_grpo_config(config_path)
            validate_campaign_config(config)
            complete_runs: list[tuple[Path, dict[str, Any]]] = []
            for summary_path in sorted((arm / "runs").glob("attempt*/summary.json")):
                value = json.loads(summary_path.read_text(encoding="utf-8"))
                if value.get("status") == "complete":
                    complete_runs.append((summary_path.parent, value))
            if len(complete_runs) != 1:
                raise RuntimeError(
                    f"Expected one complete training for {domain}/{reward}; "
                    f"found {len(complete_runs)}"
                )
            run, training = complete_runs[0]
            run_id = run.name
            if training.get("domain") != domain or training.get("reward_value") != reward:
                raise RuntimeError(f"Training provenance mismatch: {domain}/{reward}")
            checkpoints = [int(value) for value in config.raw["evaluation"]["checkpoints"]]
            evaluations = []
            evaluation_paths: dict[int, Path] = {}
            for step in checkpoints:
                summary_path = (
                    ensure_under(
                        Path(config.raw["evaluation"]["baseline_summary"]),
                        step_root,
                        "shared validation baseline",
                    )
                    if step == 0
                    else run / "evaluations" / "validation" / f"step_{step:04d}" / "summary.json"
                )
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary.get("status") != "complete" or int(summary.get("n", -1)) != 300:
                    raise RuntimeError(f"Incomplete validation: {domain}/{reward}/step{step}")
                evaluations.append(summary)
                evaluation_paths[step] = summary_path
            selected = max(evaluations, key=lambda row: (float(row["task_utility"]), -int(row["step"])))
            baseline = next(row for row in evaluations if int(row["step"]) == 0)
            selected_outcomes = task_outcomes(evaluation_paths[int(selected["step"])])
            baseline_outcomes = task_outcomes(evaluation_paths[0])
            paired_ids = sorted(set(selected_outcomes) & set(baseline_outcomes))
            if set(paired_ids) != set(selected_outcomes) or set(paired_ids) != set(
                baseline_outcomes
            ):
                raise RuntimeError(f"Unpaired validation outcomes: {domain}/{reward}")
            per_example[reward][domain] = [
                float(selected_outcomes[row_id]) - float(baseline_outcomes[row_id])
                for row_id in paired_ids
            ]
            domains[domain][reward] = {
                "run_id": run_id,
                "selected_step": int(selected["step"]),
                "baseline_task_utility": float(baseline["task_utility"]),
                "selected_task_utility": float(selected["task_utility"]),
                "validation_gain": float(selected["task_utility"]) - float(baseline["task_utility"]),
                "paired_validation_n": len(paired_ids),
                "training_reward_calls": int(training["expected_reward_calls"]),
                "source_hashes": {
                    "config": sha256_file(config_path),
                    "training_summary": sha256_file(run / "summary.json"),
                    **{
                        f"validation_step_{row['step']}": sha256_file(
                            Path(config.raw["evaluation"]["baseline_summary"])
                            if int(row["step"]) == 0
                            else run
                            / "evaluations"
                            / "validation"
                            / f"step_{int(row['step']):04d}"
                            / "summary.json"
                        )
                        for row in evaluations
                    },
                },
            }
    for reward in REWARD_VALUES:
        gains = [float(domains[domain][reward]["validation_gain"]) for domain in DOMAINS]
        macro[reward] = {
            "equal_domain_mean_validation_gain": statistics.fmean(gains),
            "domain_gains": dict(zip(DOMAINS, gains, strict=True)),
        }
    bootstrap_replicates = 10_000
    bootstrap = bootstrap_macro_gains(
        per_example,
        seed=23,
        replicates=bootstrap_replicates,
    )
    for reward in REWARD_VALUES:
        macro[reward]["bootstrap_standard_error"] = statistics.stdev(
            bootstrap[reward]
        )
        macro[reward]["bootstrap_replicates"] = bootstrap_replicates
    selection = select_reward_with_logp_tie_rule(
        {
            reward: float(macro[reward]["equal_domain_mean_validation_gain"])
            for reward in REWARD_VALUES
        },
        bootstrap,
    )
    result = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "selection_rule": (
            "maximize equal-domain mean validation task-utility gain; select the earliest "
            "checkpoint on within-arm ties and the preregistered transform order on macro ties"
        ),
        "reward_transform_order": list(REWARD_VALUES),
        "domains": domains,
        "macro": macro,
        **selection,
        "run_ids_by_arm": {
            f"{domain}/{reward}": str(domains[domain][reward]["run_id"])
            for domain in DOMAINS
            for reward in REWARD_VALUES
        },
        "official_test_accessed": False,
    }
    destination = step_root / "results" / "reward_transform_selection.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite an existing selection: {destination}")
    write_json(destination, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Campaign Step 4 reward-transform pilot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--run-id", required=True)
    train.add_argument("--key-pool", type=Path, required=True)
    train.add_argument("--key-name", choices=KEY_NAMES, required=True)
    train.add_argument("--lock-dir", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--step", type=int, choices=(0, 187, 374), required=True)
    evaluate.add_argument("--score-workers", type=int, default=12)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--step-root", type=Path, required=True)
    collect.add_argument("--config-name", required=True)
    args = parser.parse_args()
    if args.command == "train":
        result = run_training_attempt(
            config_path=args.config.resolve(),
            run_id=args.run_id,
            key_pool=args.key_pool.resolve(),
            key_name=args.key_name,
            lock_dir=args.lock_dir.resolve(),
        )
    elif args.command == "evaluate":
        result = run_validation_attempt(
            config_path=args.config.resolve(),
            run_id=args.run_id,
            step=args.step,
            score_workers=args.score_workers,
        )
    else:
        result = collect_reward_pilot(args.step_root.resolve(), args.config_name)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
