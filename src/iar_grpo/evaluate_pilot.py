from __future__ import annotations

import argparse
import asyncio
import difflib
import gc
import json
import math
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any

from iar_phase0.config import load_config
from iar_phase0.data import MbppExample, load_mbpp, materialize_split, write_jsonl
from iar_phase0.env import load_env_file
from iar_phase0.evaluate import run_mbpp_tests
from iar_phase0.generator import strip_code_fences
from iar_phase0.prompts import generator_prompt, mbpp_google_problem_statement

from .config import GrpoTrialConfig, load_grpo_config
from .reward import LiveOpenRouterReward, settings_from_config, summarize_reward_log
from .validation_reward import binary_auc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate every checkpoint of a completed GRPO pilot"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Evaluate and select on validation without reading pilot or official-test rows.",
    )
    args = parser.parse_args()

    config = load_grpo_config(args.config)
    validation_only = bool(
        args.validation_only
        or config.raw.get("evaluation", {}).get("validation_only", False)
    )
    run_dir = config.resolve(config.raw["paths"]["experiments"]) / args.run_id
    checkpoint_root = config.resolve(config.raw["paths"]["checkpoints"]) / args.run_id
    summary_path = run_dir / "summary.json"
    if (
        not summary_path.exists()
        or json.loads(summary_path.read_text())["status"] != "complete"
    ):
        raise RuntimeError(f"Training run is not complete: {run_dir}")

    load_env_file(config.resolve(config.raw["paths"]["openrouter_env"]))
    phase0_config = load_config(config.resolve(config.raw["dataset"]["phase0_config"]))
    mbpp = load_mbpp(phase0_config)
    evaluation = config.raw["evaluation"]

    validation, pilot, final_test = select_requested_splits(
        mbpp, evaluation, validation_only=validation_only
    )
    validation_spec = evaluation["validation"]

    split_manifest = {
        "training": {
            "source_split": config.raw["dataset"]["split"],
            "count": config.prompt_count,
        },
        "validation": manifest_for(validation),
        "official_test_used": not validation_only,
    }
    if not validation_only:
        assert pilot is not None and final_test is not None
        pilot_spec = evaluation["pilot_subset"]
        test_spec = evaluation["final_test"]
        excluded_prefix = selection_start_index(test_spec)
        split_manifest.update(
            {
                "pilot_subset": manifest_for(pilot),
                "final_test": manifest_for(final_test),
                "excluded_from_final_test": manifest_for(
                    materialize_split(
                        mbpp, str(test_spec["split"]), limit=excluded_prefix
                    )
                ),
            }
        )
    (run_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    checkpoints = [int(step) for step in evaluation["checkpoints"]]
    validation_summaries: list[dict[str, Any]] = []
    validation_rows: dict[int, list[dict[str, Any]]] = {}
    for step in checkpoints:
        adapter = adapter_for_step(checkpoint_root, step, checkpoints[-1])
        state_name = f"step_{step:04d}"
        rows, summary = evaluate_state(
            config=config,
            examples=validation,
            adapter_dir=adapter,
            state_name=state_name,
            split_name="validation",
            num_generations=int(validation_spec["num_generations"]),
            temperature=float(validation_spec["temperature"]),
            top_p=float(validation_spec["top_p"]),
            output_dir=run_dir / "evaluations" / "validation" / state_name,
            seed=int(config.raw["project"]["seed"]),
        )
        summary["step"] = step
        validation_summaries.append(summary)
        validation_rows[step] = rows

    selection = select_validation_checkpoint(
        validation_rows,
        z=float(evaluation["paired_lcb_z"]),
        max_pass_drop=float(evaluation["max_pass_at_1_drop"]),
        selection_rule=str(
            evaluation.get(
                "select_on", "max_feasible_paired_validation_p_own_delta_lcb"
            )
        ),
    )
    endpoint_steps = list(
        dict.fromkeys(
            [checkpoints[0], int(selection["selected_step"]), checkpoints[-1]]
        )
    )
    pilot_summaries: list[dict[str, Any]] = []
    test_summaries: list[dict[str, Any]] = []
    if not validation_only:
        assert pilot is not None and final_test is not None
        for step in endpoint_steps:
            adapter = adapter_for_step(checkpoint_root, step, checkpoints[-1])
            state_name = f"step_{step:04d}"
            _, pilot_summary = evaluate_state(
                config=config,
                examples=pilot,
                adapter_dir=adapter,
                state_name=state_name,
                split_name="pilot_subset",
                num_generations=int(pilot_spec["num_generations"]),
                temperature=float(pilot_spec["temperature"]),
                top_p=float(pilot_spec["top_p"]),
                output_dir=run_dir / "evaluations" / "pilot_subset" / state_name,
                seed=int(config.raw["project"]["seed"]),
            )
            pilot_summary["step"] = step
            pilot_summaries.append(pilot_summary)

            _, test_summary = evaluate_state(
                config=config,
                examples=final_test,
                adapter_dir=adapter,
                state_name=state_name,
                split_name="final_test",
                num_generations=int(test_spec["num_generations"]),
                temperature=float(test_spec["temperature"]),
                top_p=float(test_spec["top_p"]),
                output_dir=run_dir / "evaluations" / "final_test" / state_name,
                seed=int(config.raw["project"]["seed"]),
            )
            test_summary["step"] = step
            test_summaries.append(test_summary)

    training_summary = audit_training_rollouts(config, run_dir)
    combined = {
        "run_id": args.run_id,
        "status": "complete",
        "validation_only": validation_only,
        "split_manifest": split_manifest,
        "validation": validation_summaries,
        "selection": selection,
        "pilot_subset": pilot_summaries,
        "final_test": test_summaries,
        "training_rollouts": training_summary,
    }
    (run_dir / "evaluation_summary.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    from .plot_pilot import create_all_plots

    plot_paths = create_all_plots(run_dir)
    combined["plots"] = [str(path) for path in plot_paths]
    (run_dir / "evaluation_summary.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(combined, indent=2, ensure_ascii=False))


def select_requested_splits(
    dataset,
    evaluation: dict[str, Any],
    *,
    validation_only: bool,
) -> tuple[list[MbppExample], list[MbppExample] | None, list[MbppExample] | None]:
    """Materialize only the splits authorized for this evaluation stage."""
    validation = select_examples(dataset, evaluation["validation"])
    if validation_only:
        return validation, None, None
    return (
        validation,
        select_examples(dataset, evaluation["pilot_subset"]),
        select_examples(dataset, evaluation["final_test"]),
    )


def paired_validation_comparison(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    step: int,
    z: float,
    max_pass_drop: float,
) -> dict[str, Any]:
    """Compare a checkpoint to step zero without looking at held-out test data."""
    baseline_by_id = {str(row["task_id"]): row for row in baseline}
    candidate_by_id = {str(row["task_id"]): row for row in candidate}
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("Validation task IDs differ between baseline and checkpoint")

    task_ids = list(baseline_by_id)
    p_deltas = [
        float(candidate_by_id[task_id]["p_own"])
        - float(baseline_by_id[task_id]["p_own"])
        for task_id in task_ids
    ]
    pass_deltas = [
        int(bool(candidate_by_id[task_id]["unit_tests"]["passed"]))
        - int(bool(baseline_by_id[task_id]["unit_tests"]["passed"]))
        for task_id in task_ids
    ]
    mean_delta = statistics.mean(p_deltas)
    standard_error = (
        statistics.stdev(p_deltas) / math.sqrt(len(p_deltas))
        if len(p_deltas) > 1
        else 0.0
    )
    pass_delta = statistics.mean(pass_deltas)
    return {
        "step": int(step),
        "n": len(task_ids),
        "p_own_delta_mean": mean_delta,
        "p_own_delta_standard_error": standard_error,
        "p_own_delta_lcb": mean_delta - z * standard_error,
        "pass_at_1_delta": pass_delta,
        "net_passed_change": sum(pass_deltas),
        "quality_feasible": pass_delta >= -max_pass_drop,
        "sampled_fooling_delta": statistics.mean(
            str(candidate_by_id[task_id]["judge_token"]) == "1" for task_id in task_ids
        )
        - statistics.mean(
            str(baseline_by_id[task_id]["judge_token"]) == "1" for task_id in task_ids
        ),
        "argmax_fooling_delta": statistics.mean(
            float(candidate_by_id[task_id]["delta"]) > 0 for task_id in task_ids
        )
        - statistics.mean(
            float(baseline_by_id[task_id]["delta"]) > 0 for task_id in task_ids
        ),
    }


def select_validation_checkpoint(
    rows_by_step: dict[int, list[dict[str, Any]]],
    *,
    z: float,
    max_pass_drop: float,
    selection_rule: str = "max_feasible_paired_validation_p_own_delta_lcb",
) -> dict[str, Any]:
    """Select a checkpoint on validation before evaluating held-out test data."""
    steps = sorted(rows_by_step)
    if not steps or steps[0] != 0:
        raise ValueError(
            "Checkpoint selection requires a step-zero validation baseline"
        )
    comparisons = [
        paired_validation_comparison(
            rows_by_step[0],
            rows_by_step[step],
            step=step,
            z=z,
            max_pass_drop=max_pass_drop,
        )
        for step in steps[1:]
    ]
    if selection_rule == "validation_task_accuracy":
        winner = max(
            comparisons,
            key=lambda row: (float(row["pass_at_1_delta"]), -int(row["step"])),
            default=None,
        )
        if winner is None or float(winner["pass_at_1_delta"]) <= 0:
            winner = None
        return {
            "objective_name": "maximum_validation_pass_at_1_earliest_tie",
            "selected_step": int(winner["step"]) if winner else 0,
            "objective_value": float(winner["pass_at_1_delta"]) if winner else 0.0,
            "quality_feasible": True,
            "max_pass_at_1_drop": max_pass_drop,
            "comparisons": comparisons,
        }
    if selection_rule != "max_feasible_paired_validation_p_own_delta_lcb":
        raise ValueError(f"Unsupported checkpoint selection rule: {selection_rule}")
    feasible = [row for row in comparisons if bool(row["quality_feasible"])]
    winner = max(feasible, key=lambda row: float(row["p_own_delta_lcb"]), default=None)
    if winner is None or float(winner["p_own_delta_lcb"]) <= 0:
        selected_step = 0
        objective_value = 0.0
        selected_quality_feasible = True
    else:
        selected_step = int(winner["step"])
        objective_value = float(winner["p_own_delta_lcb"])
        selected_quality_feasible = bool(winner["quality_feasible"])
    return {
        "objective_name": "max_feasible_paired_validation_p_own_delta_lcb",
        "selected_step": selected_step,
        "objective_value": objective_value,
        "quality_feasible": selected_quality_feasible,
        "max_pass_at_1_drop": max_pass_drop,
        "comparisons": comparisons,
    }


def select_examples(dataset, spec: dict[str, Any]) -> list[MbppExample]:
    split = str(spec["split"])
    all_examples = materialize_split(dataset, split)
    start = selection_start_index(spec)
    count = int(spec["count"])
    selected = all_examples[start : start + count]
    if len(selected) != count:
        raise ValueError(
            f"Expected {count} examples from {split}[{start}:], found {len(selected)}"
        )
    return selected


def selection_start_index(spec: dict[str, Any]) -> int:
    """Resolve the backward-compatible prefix excluded from an evaluation split."""
    return int(spec.get("start_index", spec.get("exclude_first", 0)))


def manifest_for(examples: list[MbppExample]) -> dict[str, Any]:
    return {
        "source_split": examples[0].split if examples else None,
        "count": len(examples),
        "dataset_indices": [example.index for example in examples],
        "task_ids": [str(example.task_id) for example in examples],
    }


def adapter_for_step(checkpoint_root: Path, step: int, final_step: int) -> Path | None:
    if step == 0:
        return None
    if step == final_step:
        path = checkpoint_root / "final_adapter"
    else:
        path = checkpoint_root / f"checkpoint-{step}"
    if not (path / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"LoRA adapter not found: {path}")
    return path


def evaluate_state(
    *,
    config: GrpoTrialConfig,
    examples: list[MbppExample],
    adapter_dir: Path | None,
    state_name: str,
    split_name: str,
    num_generations: int,
    temperature: float,
    top_p: float,
    output_dir: Path,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cached = load_completed_evaluation_state(
        output_dir,
        state_name=state_name,
        split_name=split_name,
        expected_rows=len(examples) * num_generations,
        num_generations=num_generations,
        temperature=temperature,
        top_p=top_p,
    )
    if cached is not None:
        return cached

    output_dir.mkdir(parents=True, exist_ok=False)
    rows = generate_candidates(
        config=config,
        examples=examples,
        adapter_dir=adapter_dir,
        num_generations=num_generations,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    attach_unit_tests(
        rows, examples, int(config.raw["evaluation"]["unit_test_timeout_seconds"])
    )
    attach_live_rewards(config, rows, output_dir / "reward_calls.jsonl")
    write_jsonl(output_dir / "rows.jsonl", rows)
    summary = summarize_rows(rows, num_generations)
    summary.update(
        {
            "state": state_name,
            "split": split_name,
            "adapter": str(adapter_dir) if adapter_dir else None,
            "num_generations": num_generations,
            "temperature": temperature,
            "top_p": top_p,
            "reward_usage": summarize_reward_log(output_dir / "reward_calls.jsonl")[
                "usage"
            ],
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return rows, summary


def load_completed_evaluation_state(
    output_dir: Path,
    *,
    state_name: str,
    split_name: str,
    expected_rows: int,
    num_generations: int,
    temperature: float,
    top_p: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    """Reuse a fully written state after an interrupted evaluation run.

    A state is reusable only when its rows, reward log, and summary are all
    present and its immutable evaluation parameters agree with this request.
    This prevents an interrupted parent report from re-billing already
    completed live-judge calls while refusing ambiguous partial directories.
    """
    if not output_dir.exists():
        return None

    rows_path = output_dir / "rows.jsonl"
    reward_log_path = output_dir / "reward_calls.jsonl"
    summary_path = output_dir / "summary.json"
    required = [rows_path, reward_log_path, summary_path]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"Evaluation state directory is incomplete and cannot be resumed: {output_dir} "
            f"(missing {', '.join(missing)})"
        )

    try:
        rows = [
            json.loads(line)
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Evaluation state directory is unreadable: {output_dir}"
        ) from exc

    if len(rows) != expected_rows or int(summary.get("n", -1)) != expected_rows:
        raise RuntimeError(
            f"Evaluation state directory has an unexpected row count: {output_dir} "
            f"(expected {expected_rows}, found rows={len(rows)}, summary={summary.get('n')})"
        )
    if (
        summary.get("state") != state_name
        or summary.get("split") != split_name
        or int(summary.get("num_generations", -1)) != num_generations
        or not math.isclose(float(summary.get("temperature", math.nan)), temperature)
        or not math.isclose(float(summary.get("top_p", math.nan)), top_p)
    ):
        raise RuntimeError(
            f"Evaluation state metadata does not match requested state: {output_dir}"
        )
    return rows, summary


def generate_candidates(
    *,
    config: GrpoTrialConfig,
    examples: list[MbppExample],
    adapter_dir: Path | None,
    num_generations: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> list[dict[str, Any]]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    actor = config.raw["actor"]
    hf_cache = config.resolve(config.raw["paths"]["hf_cache"])
    actor_source = config.actor_source()
    tokenizer = AutoTokenizer.from_pretrained(
        actor_source,
        cache_dir=str(hf_cache),
        local_files_only=bool(actor["local_files_only"]),
        padding_side="left",
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        actor_source,
        cache_dir=str(hf_cache),
        local_files_only=bool(actor["local_files_only"]),
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()
    set_seed(seed)

    expanded: list[tuple[MbppExample, int]] = [
        (example, rollout_index)
        for example in examples
        for rollout_index in range(num_generations)
    ]
    batch_size = int(config.raw["evaluation"]["batch_size"])
    max_new_tokens = int(config.raw["training"]["max_completion_length"])
    rows: list[dict[str, Any]] = []
    for start in range(0, len(expanded), batch_size):
        batch = expanded[start : start + batch_size]
        conversations = [
            [
                {
                    "role": "user",
                    "content": generator_prompt(example, variant="plain", fewshot=[]),
                }
            ]
            for example, _ in batch
        ]
        inputs = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs.update(
                {"temperature": temperature, "top_p": top_p, "top_k": 0}
            )
        with torch.inference_mode():
            outputs = model.generate(**inputs, **generation_kwargs)
        prompt_length = inputs["input_ids"].shape[1]
        for (example, rollout_index), output in zip(batch, outputs, strict=True):
            completion_ids = output[prompt_length:]
            eos_positions = (completion_ids == tokenizer.eos_token_id).nonzero(
                as_tuple=True
            )[0]
            terminated = len(eos_positions) > 0
            if terminated:
                completion_ids = completion_ids[: int(eos_positions[0]) + 1]
            raw = tokenizer.decode(completion_ids, skip_special_tokens=True)
            rows.append(
                {
                    "task_id": str(example.task_id),
                    "dataset_index": int(example.index),
                    "source_split": example.split,
                    "rollout_index": rollout_index,
                    "task_statement": mbpp_google_problem_statement(example),
                    "candidate_raw": raw,
                    "candidate": strip_code_fences(raw),
                    "completion_tokens": len(completion_ids),
                    "terminated_with_eos": terminated,
                }
            )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def attach_unit_tests(
    rows: list[dict[str, Any]], examples: list[MbppExample], timeout: int
) -> None:
    by_task = {str(example.task_id): example for example in examples}

    def run(row: dict[str, Any]) -> dict[str, Any]:
        example = by_task[str(row["task_id"])]
        result = run_mbpp_tests(
            row["candidate"],
            example.tests,
            timeout_seconds=timeout,
            test_setup_code=example.test_setup_code,
        )
        return {
            "passed": result.passed,
            "returncode": result.returncode,
            "stderr": result.stderr[-1000:],
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, rows))
    for row, result in zip(rows, results, strict=True):
        row["unit_tests"] = result


def attach_live_rewards(
    config: GrpoTrialConfig, rows: list[dict[str, Any]], log_path: Path
) -> None:
    attempts = int(config.raw["evaluation"]["max_reward_attempts"])
    settings = replace(
        settings_from_config(config.raw),
        max_reward_calls=len(rows),
        max_http_requests=len(rows) * attempts,
        max_attempts_per_call=attempts,
        max_concurrency=int(config.raw["evaluation"]["max_concurrency"]),
        timeout_seconds=float(config.raw["evaluation"]["timeout_seconds"]),
    )
    runtime = LiveOpenRouterReward(settings=settings, log_path=log_path)
    rewards = asyncio.run(
        runtime.score_batch(
            completions=[row["candidate_raw"] for row in rows],
            task_statement=[row["task_statement"] for row in rows],
            task_id=[row["task_id"] for row in rows],
            dataset_index=[row["dataset_index"] for row in rows],
        )
    )
    runtime.assert_complete()
    ok_calls = sorted(
        (
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ),
        key=lambda row: (int(row["call_id"]), int(row["attempt"])),
    )
    ok_calls = [row for row in ok_calls if row.get("status") == "ok"]
    if len(ok_calls) != len(rows):
        raise RuntimeError(
            f"Expected {len(rows)} successful evaluation rewards, got {len(ok_calls)}"
        )
    for row, reward, call in zip(rows, rewards, ok_calls, strict=True):
        row.update(
            {
                "reward": float(reward),
                "judge_token": str(call["judge_token"]),
                "logprob_first_token": str(call.get("logprob_first_token", "")),
                "logp_1": float(call["logp_1"]),
                "logp_0": float(call["logp_0"]),
                "p_own": float(call["p_own"]),
                "delta": float(call["delta"]),
                "raw_model": str(call["raw_model"]),
            }
        )


def summarize_rows(
    rows: list[dict[str, Any]],
    num_generations: int,
    group_size: int | None = None,
) -> dict[str, Any]:
    rewards = [float(row["reward"]) for row in rows]
    p_own_values = [float(row.get("p_own", row["reward"])) for row in rows]
    passed = [bool(row["unit_tests"]["passed"]) for row in rows]
    correct_rewards = [
        float(row["reward"]) for row in rows if row["unit_tests"]["passed"]
    ]
    wrong_rewards = [
        float(row["reward"]) for row in rows if not row["unit_tests"]["passed"]
    ]
    correct_p_own = [
        float(row.get("p_own", row["reward"]))
        for row in rows
        if row["unit_tests"]["passed"]
    ]
    wrong_p_own = [
        float(row.get("p_own", row["reward"]))
        for row in rows
        if not row["unit_tests"]["passed"]
    ]
    strict_matches = 0
    ties = 0
    contradictions = 0
    for row in rows:
        delta = float(row["delta"])
        token = str(row["judge_token"])
        if delta == 0:
            ties += 1
        elif (delta > 0 and token == "1") or (delta < 0 and token == "0"):
            strict_matches += 1
        else:
            contradictions += 1
    result: dict[str, Any] = {
        "n": len(rows),
        "pass@1": statistics.mean(passed),
        "passed": sum(passed),
        "reward_mean": statistics.mean(rewards),
        "reward_std": statistics.pstdev(rewards),
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "p_own_mean": statistics.mean(p_own_values),
        "p_own_std": statistics.pstdev(p_own_values),
        "p_own_min": min(p_own_values),
        "p_own_max": max(p_own_values),
        "fooling_rate": statistics.mean(str(row["judge_token"]) == "1" for row in rows),
        "sampled_fooling_rate": statistics.mean(
            str(row["judge_token"]) == "1" for row in rows
        ),
        "argmax_fooling_rate": statistics.mean(float(row["delta"]) > 0 for row in rows),
        "mean_reward_correct": statistics.mean(correct_rewards)
        if correct_rewards
        else None,
        "mean_reward_incorrect": statistics.mean(wrong_rewards)
        if wrong_rewards
        else None,
        "mean_p_own_correct": statistics.mean(correct_p_own) if correct_p_own else None,
        "mean_p_own_incorrect": statistics.mean(wrong_p_own) if wrong_p_own else None,
        "reward_auc_for_correctness": binary_auc(passed, p_own_values),
        "logprob_consistency": {
            "strict_matches": strict_matches,
            "ties": ties,
            "contradictions": contradictions,
        },
    }
    if all("terminated_with_eos" in row for row in rows):
        result["termination_rate"] = statistics.mean(
            bool(row["terminated_with_eos"]) for row in rows
        )
    if num_generations > 1:
        groups = group_statistics(rows, group_size=group_size)
        result["groups"] = {
            "n": len(groups),
            "at_least_one_correct": sum(group["pass_count"] > 0 for group in groups),
            "all_correct": sum(
                group["pass_count"] == num_generations for group in groups
            ),
            "mixed_correctness": sum(
                0 < group["pass_count"] < num_generations for group in groups
            ),
            "none_correct": sum(group["pass_count"] == 0 for group in groups),
            "mean_pairwise_similarity": statistics.mean(
                group["pairwise_similarity"] for group in groups
            ),
            "all_rollouts_textually_unique": all(
                group["unique_candidates"] == num_generations for group in groups
            ),
        }
    return result


def group_statistics(
    rows: list[dict[str, Any]], group_size: int | None = None
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if group_size is None:
            key = str(row["task_id"])
        else:
            if group_size <= 0 or "call_id" not in row:
                raise ValueError(
                    "Training-group reconstruction requires call_id and group_size > 0"
                )
            key = str((int(row["call_id"]) - 1) // group_size)
        grouped[key].append(row)
    results = []
    for group_key, group in grouped.items():
        ordered = sorted(
            group,
            key=lambda row: (
                int(row["call_id"])
                if group_size is not None
                else int(row["rollout_index"])
            ),
        )
        if group_size is not None and len(ordered) != group_size:
            raise ValueError(
                f"Reconstructed training group {group_key} has {len(ordered)} rows; "
                f"expected {group_size}"
            )
        similarities = [
            difflib.SequenceMatcher(None, left["candidate"], right["candidate"]).ratio()
            for left, right in combinations(ordered, 2)
        ]
        rewards = [float(row["reward"]) for row in ordered]
        labels = [str(row["judge_token"]) for row in ordered]
        results.append(
            {
                "task_id": str(ordered[0]["task_id"]),
                "source_group_index": int(group_key)
                if group_size is not None
                else None,
                "num_rollouts": len(ordered),
                "pass_count": sum(bool(row["unit_tests"]["passed"]) for row in ordered),
                "reward_mean": statistics.mean(rewards),
                "reward_std": statistics.pstdev(rewards),
                "fooling_rate": statistics.mean(label == "1" for label in labels),
                "label_pattern": "".join(labels),
                "pairwise_similarity": statistics.mean(similarities)
                if similarities
                else 1.0,
                "unique_candidates": len({row["candidate"] for row in ordered}),
            }
        )
    return results


def audit_training_rollouts(config: GrpoTrialConfig, run_dir: Path) -> dict[str, Any]:
    phase0_config = load_config(config.resolve(config.raw["dataset"]["phase0_config"]))
    mbpp = load_mbpp(phase0_config)
    examples = materialize_split(
        mbpp, str(config.raw["dataset"]["split"]), limit=config.prompt_count
    )
    reward_rows = [
        json.loads(line)
        for line in (run_dir / "reward_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    reward_rows = sorted(
        (row for row in reward_rows if row.get("status") == "ok"),
        key=lambda row: int(row["call_id"]),
    )
    rows = [
        {
            **row,
            "candidate": strip_code_fences(str(row["candidate_raw"])),
        }
        for row in reward_rows
    ]
    attach_unit_tests(
        rows, examples, int(config.raw["evaluation"]["unit_test_timeout_seconds"])
    )
    write_jsonl(run_dir / "training_rollout_audit.jsonl", rows)
    groups = sorted(
        group_statistics(rows, group_size=config.num_generations),
        key=lambda group: int(group["source_group_index"]),
    )
    for index, group in enumerate(groups, start=1):
        group["group_index"] = index
    write_jsonl(run_dir / "training_groups.jsonl", groups)
    summary = summarize_rows(
        rows,
        config.num_generations,
        group_size=config.num_generations,
    )
    summary["group_count"] = len(groups)
    return summary


if __name__ == "__main__":
    main()
