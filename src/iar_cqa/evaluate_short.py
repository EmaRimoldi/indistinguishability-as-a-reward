from __future__ import annotations

import argparse
import asyncio
import gc
import json
import math
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from iar_grpo.reward import LiveOpenRouterReward, settings_from_config
from iar_phase0.env import load_env_file

from .data import read_jsonl, sha256_json, write_json, write_jsonl
from .prompts import ATTRIBUTION_PROMPT_VERSION, attribution_prompt, task_prompt, template_hashes
from .scoring import response_features, score_response
from .train import actor_source, resolve, select_balanced


def adapter_for_step(checkpoint_root: Path, step: int, final_step: int) -> Path | None:
    if step == 0:
        return None
    path = checkpoint_root / ("final_adapter" if step == final_step else f"checkpoint-{step}")
    if not (path / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"LoRA adapter not found: {path}")
    return path


def generate(
    config: dict[str, Any],
    root: Path,
    examples: list[dict[str, Any]],
    adapter: Path | None,
    step: int,
) -> list[dict[str, Any]]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    cache = resolve(root, config["paths"]["hf_cache"])
    source = actor_source(root, config)
    tokenizer = AutoTokenizer.from_pretrained(
        source, cache_dir=str(cache), local_files_only=True, padding_side="left", trust_remote_code=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        source,
        cache_dir=str(cache),
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    set_seed(int(config["project"]["seed"]))
    evaluation = config["evaluation"]
    batch_size = int(evaluation["batch_size"])
    rows: list[dict[str, Any]] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        tasks = [task_prompt(item["question"], item["choices"]) for item in batch]
        conversations = [[{"role": "user", "content": task}] for task in tasks]
        inputs = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(evaluation["max_new_tokens"]),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_length = inputs["input_ids"].shape[1]
        for item, task, output in zip(batch, tasks, outputs, strict=True):
            completion = output[prompt_length:]
            eos = (completion == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            terminated = len(eos) > 0
            if terminated:
                completion = completion[: int(eos[0]) + 1]
            response = tokenizer.decode(completion, skip_special_tokens=True).strip()
            rows.append(
                {
                    "id": item["id"],
                    "source_index": item["source_index"],
                    "step": step,
                    "task_statement": task,
                    "response": response,
                    "score": score_response(response, item["answer_key"]),
                    "features": response_features(response),
                    "completion_tokens": int(len(completion)),
                    "terminated_with_eos": terminated,
                }
            )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def attach_rewards(config: dict[str, Any], rows: list[dict[str, Any]], log_path: Path) -> None:
    attempts = int(config["evaluation"]["max_reward_attempts"])
    settings = replace(
        settings_from_config(config),
        max_reward_calls=len(rows),
        max_http_requests=len(rows) * attempts,
        max_attempts_per_call=attempts,
        max_concurrency=int(config["evaluation"]["max_concurrency"]),
        timeout_seconds=float(config["evaluation"]["timeout_seconds"]),
    )
    runtime = LiveOpenRouterReward(
        settings=settings,
        log_path=log_path,
        prompt_builder=attribution_prompt,
        prompt_template_hash=template_hashes()[ATTRIBUTION_PROMPT_VERSION],
        system_prompt_builder=lambda _name: None,
        candidate_normalizer=str.strip,
    )
    rewards = asyncio.run(
        runtime.score_batch(
            completions=[row["response"] for row in rows],
            task_statement=[row["task_statement"] for row in rows],
            task_id=[row["id"] for row in rows],
            dataset_index=[row["source_index"] for row in rows],
        )
    )
    runtime.assert_complete()
    calls = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    calls = sorted((row for row in calls if row.get("status") == "ok"), key=lambda row: int(row["call_id"]))
    if len(calls) != len(rows):
        raise RuntimeError(f"Expected {len(rows)} successful validation rewards, found {len(calls)}")
    for row, reward, call in zip(rows, rewards, calls, strict=True):
        row.update(
            {
                "reward": float(reward),
                "p_own": float(call["p_own"]),
                "logp_0": float(call["logp_0"]),
                "logp_1": float(call["logp_1"]),
                "delta": float(call["delta"]),
                "binary_label_mass": float(call["binary_label_mass"]),
                "judge_token": str(call["judge_token"]),
                "judge_argmax_token": "1" if float(call["delta"]) > 0 else "0",
                "raw_model": call["raw_model"],
                "raw_provider": call["raw_provider"],
            }
        )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "correct": sum(row["score"]["correct"] for row in rows),
        "accuracy": statistics.mean(row["score"]["correct"] for row in rows),
        "valid": sum(row["score"]["valid"] for row in rows),
        "valid_rate": statistics.mean(row["score"]["valid"] for row in rows),
        "mean_reward": statistics.mean(float(row["reward"]) for row in rows),
        "mean_p_own": statistics.mean(float(row["p_own"]) for row in rows),
        "sampled_fooling": statistics.mean(str(row["judge_token"]) == "1" for row in rows),
        "argmax_fooling": statistics.mean(float(row["delta"]) > 0 for row in rows),
        "mean_completion_tokens": statistics.mean(int(row["completion_tokens"]) for row in rows),
        "termination_rate": statistics.mean(bool(row["terminated_with_eos"]) for row in rows),
        "minimum_binary_label_mass": min(float(row["binary_label_mass"]) for row in rows),
        "emitted_argmax_consistency": statistics.mean(
            str(row["judge_token"]) == str(row["judge_argmax_token"]) for row in rows
        ),
    }


def paired(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]], *, z: float) -> dict[str, Any]:
    base = {row["id"]: row for row in baseline}
    cand = {row["id"]: row for row in candidate}
    if base.keys() != cand.keys():
        raise ValueError("Validation IDs differ across checkpoints")
    ids = list(base)
    deltas = [float(cand[key]["p_own"]) - float(base[key]["p_own"]) for key in ids]
    accuracy_deltas = [int(cand[key]["score"]["correct"]) - int(base[key]["score"]["correct"]) for key in ids]
    mean_delta = statistics.mean(deltas)
    se = statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
    return {
        "n": len(ids),
        "delta_p_own": mean_delta,
        "delta_p_own_standard_error": se,
        "delta_p_own_lcb_95_one_sided": mean_delta - z * se,
        "accuracy_delta": statistics.mean(accuracy_deltas),
        "net_correct_change": sum(accuracy_deltas),
        "sampled_fooling_delta": statistics.mean(str(cand[key]["judge_token"]) == "1" for key in ids)
        - statistics.mean(str(base[key]["judge_token"]) == "1" for key in ids),
        "argmax_fooling_delta": statistics.mean(float(cand[key]["delta"]) > 0 for key in ids)
        - statistics.mean(float(base[key]["delta"]) > 0 for key in ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the gated CQA short trial")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    root = Path(config["project"]["root"]).resolve()
    run_dir = resolve(root, config["paths"]["experiments"]) / args.run_id
    training_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if training_summary.get("status") != "complete":
        raise RuntimeError("Training did not complete successfully")
    checkpoint_root = resolve(root, config["paths"]["checkpoints"]) / args.run_id
    source = read_jsonl(resolve(root, config["paths"]["split_dir"]) / "validation.jsonl")
    examples = select_balanced(
        source,
        count=int(config["evaluation"]["validation_count"]),
        seed=int(config["project"]["seed"]) + 1_000_003,
    )
    write_json(
        run_dir / "parameters" / "validation_subset_manifest.json",
        {
            "n": len(examples),
            "seed": int(config["project"]["seed"]) + 1_000_003,
            "ids_sha256": sha256_json([row["id"] for row in examples]),
            "ordered_ids": [row["id"] for row in examples],
        },
    )
    load_env_file(resolve(root, config["paths"]["openrouter_env"]))
    checkpoints = [int(step) for step in config["evaluation"]["checkpoints"]]
    rows_by_step: dict[int, list[dict[str, Any]]] = {}
    summaries = []
    for step in checkpoints:
        state_dir = run_dir / "evaluations" / f"step_{step:04d}"
        state_dir.mkdir(parents=True, exist_ok=False)
        rows = generate(
            config,
            root,
            examples,
            adapter_for_step(checkpoint_root, step, checkpoints[-1]),
            step,
        )
        attach_rewards(config, rows, state_dir / "reward_calls.jsonl")
        write_jsonl(state_dir / "rows.jsonl", rows)
        state_summary = {"step": step, **summarize(rows)}
        write_json(state_dir / "summary.json", state_summary)
        rows_by_step[step] = rows
        summaries.append(state_summary)
    comparisons = [
        {"step": step, **paired(rows_by_step[0], rows_by_step[step], z=float(config["evaluation"]["paired_lcb_z"]))}
        for step in checkpoints[1:]
    ]
    maximum_drop = float(config["evaluation"]["maximum_accuracy_drop"])
    for comparison in comparisons:
        comparison["quality_feasible"] = float(comparison["accuracy_delta"]) >= -maximum_drop
    result = {
        "run_id": args.run_id,
        "status": "complete",
        "validation": summaries,
        "comparisons_to_step_0": comparisons,
        "training_effect_detected": any(
            row["quality_feasible"] and float(row["delta_p_own_lcb_95_one_sided"]) > 0
            for row in comparisons
        ),
    }
    write_json(run_dir / "evaluation_summary.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
