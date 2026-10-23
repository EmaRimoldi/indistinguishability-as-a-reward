from __future__ import annotations

import argparse
import gc
import json
import re
import statistics
from pathlib import Path
from typing import Any

import yaml

from iar_analysis.evaluation_resume import load_completed_checkpoint
from iar_phase0.env import load_env_file

from .analyze import auc
from .data import sha256_json, write_json, write_jsonl
from .evaluate_short import attach_rewards
from .item4_protocol import (
    CHECKPOINTS,
    VALIDATION_COUNT,
    audit_readiness,
    exact_actor_snapshot,
    load_frozen_split,
    resolve,
)
from .prompts import task_prompt
from .scoring import response_features, score_response

SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def adapter_for_step(checkpoint_root: Path, step: int) -> Path | None:
    if step == 0:
        return None
    path = checkpoint_root / (
        "final_adapter" if step == CHECKPOINTS[-1] else f"checkpoint-{step}"
    )
    if not (path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"LoRA adapter not found: {path}")
    return path


def generate(
    config: dict[str, Any],
    *,
    root: Path,
    examples: list[dict[str, Any]],
    adapter: Path | None,
    step: int,
) -> list[dict[str, Any]]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    source = exact_actor_snapshot(config, root)
    cache = resolve(root, config["paths"]["hf_cache"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(source),
        cache_dir=str(cache),
        local_files_only=True,
        trust_remote_code=False,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(source),
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
        prompts = [
            task_prompt(str(row["question"]), list(row["choices"])) for row in batch
        ]
        conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
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
        for example, task, output in zip(batch, prompts, outputs, strict=True):
            completion = output[prompt_length:]
            response = tokenizer.decode(completion, skip_special_tokens=True).strip()
            rows.append(
                {
                    "id": str(example["id"]),
                    "source_index": int(example["source_index"]),
                    "step": step,
                    "task_statement": task,
                    "response": response,
                    "score": score_response(response, str(example["answer_key"])),
                    "features": response_features(response),
                    "completion_tokens": len(completion),
                    "official_test_accessed": False,
                }
            )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def summarize(step: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != VALIDATION_COUNT:
        raise RuntimeError("Incomplete CQA Item 4 validation evaluation")
    correct = [bool(row["score"]["correct"]) for row in rows]
    valid = [bool(row["score"]["valid"]) for row in rows]
    return {
        "step": step,
        "n": len(rows),
        "correct": sum(correct),
        "accuracy": statistics.fmean(correct),
        "valid": sum(valid),
        "valid_rate": statistics.fmean(valid),
        "mean_reward": statistics.fmean(float(row["reward"]) for row in rows),
        "mean_p_own": statistics.fmean(float(row["p_own"]) for row in rows),
        "sampled_fooling": statistics.fmean(
            str(row["judge_token"]) == "1" for row in rows
        ),
        "argmax_fooling": statistics.fmean(float(row["delta"]) > 0 for row in rows),
        "reward_auc_for_correctness": auc(
            [float(row["p_own"]) for row in rows], correct
        ),
        "official_test_accessed": False,
    }


def select_checkpoint(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if [int(row["step"]) for row in summaries] != CHECKPOINTS:
        raise RuntimeError("Checkpoint evaluation is incomplete or out of order")
    selected = max(
        summaries,
        key=lambda row: (float(row["accuracy"]), -int(row["step"])),
    )
    return {
        "selector": "task_utility",
        "task_utility": "exact_answer_accuracy",
        "tie_break": "earliest_checkpoint",
        "selected_step": int(selected["step"]),
        "selected_accuracy": float(selected["accuracy"]),
        "official_test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CQA Item 4 on internal validation only"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not SAFE_RUN_ID.fullmatch(args.run_id):
        raise ValueError("Unsafe CQA run ID")
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(config["project"]["root"]).resolve()
    run_dir = resolve(root, config["paths"]["experiments"]) / args.run_id
    authorization_gate = run_dir / "parameters" / "authorization_gate.json"
    audit_readiness(
        config,
        root=root,
        config_path=config_path,
        authorization_gate=authorization_gate,
        require_ready=True,
    )
    training = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if (
        training.get("status") != "complete"
        or training.get("official_test_accessed") is not False
    ):
        raise RuntimeError("Training run is incomplete or violates held-out policy")
    examples = load_frozen_split(config, root, "validation")
    load_env_file(resolve(root, config["paths"]["openrouter_env"]))
    write_json(
        run_dir / "parameters" / "validation_pool.json",
        {
            "n": len(examples),
            "ordered_ids_sha256": sha256_json([str(row["id"]) for row in examples]),
            "source_split": "official_train_internal_validation",
            "official_test_accessed": False,
        },
    )
    checkpoint_root = resolve(root, config["paths"]["checkpoints"]) / args.run_id
    summaries = []
    for step in CHECKPOINTS:
        state_dir = run_dir / "evaluations" / f"step_{step:05d}"
        cached = load_completed_checkpoint(
            state_dir,
            step=step,
            expected_rows=VALIDATION_COUNT,
            rows_file="rows.jsonl",
            required_files=("reward_calls.jsonl",),
        )
        if cached is not None:
            summaries.append(cached)
            continue
        state_dir.mkdir(parents=True, exist_ok=False)
        rows = generate(
            config,
            root=root,
            examples=examples,
            adapter=adapter_for_step(checkpoint_root, step),
            step=step,
        )
        attach_rewards(config, rows, state_dir / "reward_calls.jsonl")
        write_jsonl(state_dir / "rows.jsonl", rows)
        summary = summarize(step, rows)
        write_json(state_dir / "summary.json", summary)
        summaries.append(summary)
    selection = select_checkpoint(summaries)
    write_json(run_dir / "validation_checkpoint_summaries.json", {"rows": summaries})
    write_json(run_dir / "validation_checkpoint_selection.json", selection)
    print(json.dumps(selection, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
