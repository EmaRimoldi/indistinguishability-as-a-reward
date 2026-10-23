from __future__ import annotations

import gc
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from iar_campaign.four_domain_screen import _adapter_for_step, _read_json, _under
from iar_campaign.generate import prompts_for_domain, read_jsonl
from iar_campaign.reward_pilot import sha256_file, utc_now, write_json, write_jsonl
from iar_campaign.score_baselines import (
    _score_kodcode_rows,
    _score_simple,
    _score_spider_rows,
)
from iar_grpo.config import load_grpo_config

from .baseline_suite import STEP_ROOT, validate_baseline_grpo_config
from .sft_baseline import CHECKPOINTS as SFT_CHECKPOINTS

GRPO_CHECKPOINTS = (0, 500, 1000, 1500, 2000)
METHODS = ("sft_teacher", "grpo_quality", "grpo_oracle")


def _run_and_checkpoints(
    *, domain: str, method: str, run_id: str
) -> tuple[Path, Path, tuple[int, ...]]:
    arm = STEP_ROOT / "domains" / domain / method
    if method == "sft_teacher":
        return (
            arm / "sft_runs" / run_id,
            arm / "sft_checkpoints" / run_id,
            SFT_CHECKPOINTS,
        )
    return arm / "runs" / run_id, arm / "checkpoints" / run_id, GRPO_CHECKPOINTS


def generate_validation_attempt(
    *,
    config_path: Path,
    method: str,
    run_id: str,
    step: int,
    attempt: int,
) -> dict[str, Any]:
    if method not in METHODS or attempt <= 0:
        raise ValueError("Invalid Step 7 evaluation method or attempt")
    config = load_grpo_config(config_path)
    domain, _, _, validation_file, _ = validate_baseline_grpo_config(config)
    run_root, checkpoint_root, checkpoints = _run_and_checkpoints(
        domain=domain, method=method, run_id=run_id
    )
    if step not in checkpoints[1:]:
        raise ValueError(f"Invalid {method} checkpoint: {step}")
    run_root = _under(run_root, STEP_ROOT, "Step 7 evaluation run")
    checkpoint_root = _under(checkpoint_root, STEP_ROOT, "Step 7 evaluation checkpoint")
    training = _read_json(run_root / "summary.json")
    if training.get("status") != "complete" or training.get("official_test_accessed") is not False:
        raise RuntimeError("Step 7 validation cannot start before complete training")
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
    if len(examples) != 300:
        raise RuntimeError("Step 7 evaluation requires validation300")
    tables = (
        Path(str(config.raw["evaluation"]["spider_tables_json"]))
        if domain == "spider"
        else None
    )
    prompts = prompts_for_domain(domain, examples, tables_json=tables)["a0"]
    adapter = _adapter_for_step(checkpoint_root, step, checkpoints[-1])
    if adapter is None:
        raise RuntimeError("A trained Step 7 checkpoint resolved to A0")
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
        model = PeftModel.from_pretrained(base_model, adapter, is_trainable=False)
        model.eval()
        candidates: list[dict[str, Any]] = []
        batch_size = int(config.raw["evaluation"]["batch_size"])
        max_tokens = int(config.raw["evaluation"]["max_new_tokens"])
        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start : start + batch_size]
            batch_prompts = prompts[start : start + batch_size]
            conversations = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]
            encoded = tokenizer.apply_chat_template(
                conversations,
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
            for example, prompt, output in zip(
                batch_examples, batch_prompts, generated, strict=True
            ):
                completion = output[prompt_length:]
                response = tokenizer.decode(completion, skip_special_tokens=True).strip()
                candidates.append(
                    {
                        "id": str(example["id"]),
                        "source_index": int(example["source_index"]),
                        "domain": domain,
                        "method": method,
                        "step": step,
                        "run_id": run_id,
                        "response": response,
                        "completion_tokens": len(completion),
                        "max_new_tokens": max_tokens,
                        "hit_max_new_tokens": len(completion) >= max_tokens,
                    }
                )
        write_jsonl(output_dir / "candidates.jsonl", candidates)
        summary = {
            "schema_version": 1,
            "status": "complete",
            "completed_at": utc_now(),
            "domain": domain,
            "method": method,
            "run_id": run_id,
            "step": step,
            "generation_attempt": attempt,
            "n": len(candidates),
            "truncated": sum(bool(row["hit_max_new_tokens"]) for row in candidates),
            "candidate_file_sha256": sha256_file(output_dir / "candidates.jsonl"),
            "validation_file": str(validation_file),
            "validation_file_sha256": sha256_file(validation_file),
            "adapter": str(adapter),
            "sdpa": sdpa,
            "official_test_accessed": False,
        }
        write_json(output_dir / "summary.json", summary)
        return summary
    except Exception as exc:
        write_json(
            output_dir / "failure.json",
            {
                "status": "failed",
                "failed_at": utc_now(),
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
    method: str,
    run_id: str,
    step: int,
    generation_attempt: int,
    scoring_attempt: int,
    score_workers: int,
) -> dict[str, Any]:
    if method not in METHODS or scoring_attempt <= 0:
        raise ValueError("Invalid Step 7 scoring method or attempt")
    config = load_grpo_config(config_path)
    domain, _, _, validation_file, _ = validate_baseline_grpo_config(config)
    run_root, _, checkpoints = _run_and_checkpoints(domain=domain, method=method, run_id=run_id)
    if step not in checkpoints[1:]:
        raise ValueError(f"Invalid {method} checkpoint: {step}")
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
        or [str(row["id"]) for row in candidates] != [str(row["id"]) for row in examples]
    ):
        raise RuntimeError("Step 7 generation is incomplete or misaligned")
    output_dir = (
        run_root
        / "evaluations/validation"
        / f"step_{step:04d}"
        / f"score_attempt{scoring_attempt:03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        scored_rows = candidates
        protocol: dict[str, Any] | None = None
        if domain in {"commonsenseqa", "gsm8k"}:
            _score_simple(domain, examples, candidates)
        elif domain == "kodcode":
            protocol_path = Path(str(config.raw["evaluation"]["kodcode_scoring_protocol"]))
            scored_rows, protocol = _score_kodcode_rows(
                examples,
                candidates,
                scoring_protocol_path=protocol_path,
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
                database_root=Path(
                    str(config.raw["evaluation"]["spider_database_root"])
                ),
                score_workers=score_workers,
            )
        write_jsonl(output_dir / "candidates.jsonl", candidates)
        summary = {
            "schema_version": 1,
            "status": "complete",
            "completed_at": utc_now(),
            "domain": domain,
            "method": method,
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
    except Exception as exc:
        write_json(
            output_dir / "failure.json",
            {
                "status": "failed",
                "failed_at": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "official_test_accessed": False,
            },
        )
        raise


def collect_method_validation(
    *, config_path: Path, method: str, run_id: str
) -> dict[str, Any]:
    config = load_grpo_config(config_path)
    domain, _, _, _, _ = validate_baseline_grpo_config(config)
    run_root, _, checkpoints = _run_and_checkpoints(domain=domain, method=method, run_id=run_id)
    baseline_registry = _read_json(
        Path(str(config.raw["evaluation"]["baseline_registry"]))
    )
    a0 = baseline_registry["baselines"][domain]["a0"]
    evaluations = [
        {
            "step": 0,
            "task_utility": float(a0["task_utility"]),
            "summary": a0["summary"],
            "summary_sha256": a0["summary_sha256"],
        }
    ]
    for step in checkpoints[1:]:
        score_root = run_root / "evaluations/validation" / f"step_{step:04d}"
        complete = []
        for path in sorted(score_root.glob("score_attempt*/summary.json")):
            value = _read_json(path)
            if value.get("status") == "complete":
                complete.append((path, value))
        if len(complete) != 1:
            raise RuntimeError(
                f"Expected one successful score for {domain}/{method}/step{step}, found {len(complete)}"
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
    result = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "protocol": "campaign-step7-validation-checkpoint-selection-v1",
        "domain": domain,
        "method": method,
        "run_id": run_id,
        "selected_step": int(selected["step"]),
        "selected_task_utility": float(selected["task_utility"]),
        "validation_gain_vs_a0": float(selected["task_utility"])
        - float(a0["task_utility"]),
        "evaluations": evaluations,
        "checkpoint_selection": "maximize validation task utility; earliest checkpoint on ties",
        "official_test_accessed": False,
        "mbpp": "FORBIDDEN",
    }
    output = run_root / "evaluations/validation/selection.json"
    if output.exists():
        raise FileExistsError(output)
    write_json(output, result)
    return result


def collect_baseline_suite(*, preparation_manifest: Path) -> dict[str, Any]:
    preparation = _read_json(preparation_manifest)
    if (
        preparation.get("status") != "PREPARED_NOT_LAUNCHED"
        or preparation.get("official_test_accessed") is not False
        or preparation.get("mbpp") != "FORBIDDEN"
    ):
        raise RuntimeError("Step 7 preparation manifest is invalid")
    selected_domains = list(preparation.get("selected_domains") or [])
    if len(selected_domains) != 2 or selected_domains[0] != "commonsenseqa":
        raise RuntimeError("Step 7 collector requires CQA plus one selected domain")
    results: dict[str, dict[str, Any]] = {}
    for domain in selected_domains:
        results[domain] = {}
        reused = preparation["reused_artifacts"][domain]
        for method in ("a0", "prompt_imitation"):
            summary = Path(str(reused[method]["summary"]))
            if sha256_file(summary) != str(reused[method]["summary_sha256"]):
                raise RuntimeError(f"Reused Step 5 baseline drifted: {domain}/{method}")
            results[domain][method] = {
                "source": "reused_step5",
                "task_utility": float(reused[method]["task_utility"]),
                "summary": str(summary),
                "summary_sha256": sha256_file(summary),
            }
        identity = reused["grpo_identity"]
        training = Path(str(identity["training_summary"]))
        selected_summary = Path(str(identity["selected_validation_summary"]))
        if (
            sha256_file(training) != str(identity["training_summary_sha256"])
            or not selected_summary.is_file()
        ):
            raise RuntimeError(f"Reused Step 5 identity arm drifted: {domain}")
        results[domain]["grpo_identity"] = {
            "source": "reused_step5",
            "selected_step": int(identity["selected_step"]),
            "task_utility": float(identity["selected_task_utility"]),
            "training_summary": str(training),
            "training_summary_sha256": sha256_file(training),
            "validation_summary": str(selected_summary),
            "validation_summary_sha256": sha256_file(selected_summary),
        }
        for method in METHODS:
            run_root, _, _ = _run_and_checkpoints(
                domain=domain, method=method, run_id="attempt001"
            )
            selection_path = run_root / "evaluations/validation/selection.json"
            selection = _read_json(selection_path)
            if (
                selection.get("status") != "complete"
                or selection.get("domain") != domain
                or selection.get("method") != method
                or selection.get("official_test_accessed") is not False
            ):
                raise RuntimeError(f"Incomplete Step 7 validation selection: {domain}/{method}")
            results[domain][method] = {
                "source": "new_step7",
                "selected_step": int(selection["selected_step"]),
                "task_utility": float(selection["selected_task_utility"]),
                "validation_gain_vs_a0": float(selection["validation_gain_vs_a0"]),
                "selection": str(selection_path),
                "selection_sha256": sha256_file(selection_path),
            }
    result = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "protocol": "campaign-selected-domain-baseline-suite-v1",
        "selected_domains": selected_domains,
        "preparation_manifest": str(preparation_manifest.resolve()),
        "preparation_manifest_sha256": sha256_file(preparation_manifest.resolve()),
        "results": results,
        "checkpoint_selection": "maximize validation task utility; earliest checkpoint on ties",
        "official_test_accessed": False,
        "mbpp": "FORBIDDEN",
        "ifbench": "DEFERRED",
    }
    destination = STEP_ROOT / "results/baseline_suite_validation_v1.json"
    if destination.exists():
        raise FileExistsError(destination)
    write_json(destination, result)
    return result
