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
from iar_grpo.validation_reward import attach_validation_rewards, binary_auc
from iar_phase0.env import load_env_file

from .data import write_json, write_jsonl
from .evaluate_validation import evaluate_validation, load_validation_protocol
from .prompts import (
    ATTRIBUTION_PROMPT_VERSION,
    attribution_prompt,
    task_prompt,
    template_hashes,
)
from .screen_protocol import CHECKPOINTS, audit_readiness, config_sha256
from .train import actor_source, resolve

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


def generate_predictions(
    config: dict[str, Any],
    *,
    root: Path,
    examples: list[dict[str, Any]],
    adapter: Path | None,
) -> list[dict[str, Any]]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    cache = resolve(root, config["paths"]["hf_cache"])
    source = actor_source(root, config)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        cache_dir=str(cache),
        local_files_only=True,
        trust_remote_code=False,
        padding_side="left",
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
    rows: list[dict[str, Any]] = []
    batch_size = int(evaluation["batch_size"])
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        prompts = [task_prompt(str(row["prompt"])) for row in batch]
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
            response = tokenizer.decode(
                output[prompt_length:], skip_special_tokens=True
            ).strip()
            rows.append(
                {
                    "id": str(example["id"]),
                    "source_index": int(example["source_index"]),
                    "task_statement": task,
                    "output": response,
                }
            )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def select_checkpoint(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if [int(row["step"]) for row in summaries] != CHECKPOINTS:
        raise RuntimeError("Checkpoint evaluation is incomplete or out of order")
    if any(int(row.get("n", -1)) != 1_501 for row in summaries):
        raise RuntimeError("Checkpoint evaluation does not cover all validation rows")
    winner = max(
        summaries,
        key=lambda row: (float(row["accuracy"]), -int(row["step"])),
    )
    return {
        "selector": "task_utility",
        "task_utility": "ifeval_old_constraint_accuracy",
        "tie_break": "earliest_checkpoint",
        "selected_step": int(winner["step"]),
        "selected_accuracy": float(winner["accuracy"]),
        "official_test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and evaluate every IFBench-screen checkpoint on internal validation"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not SAFE_RUN_ID.fullmatch(args.run_id):
        raise ValueError("Unsafe IFBench run ID")
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(config["project"]["root"]).resolve()
    audit_readiness(config, root=root, require_ready=True)
    run_dir = resolve(root, config["paths"]["experiments"]) / args.run_id
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if (
        summary.get("status") != "complete"
        or summary.get("official_test_accessed") is not False
        or summary.get("config_sha256") != config_sha256(config_path)
    ):
        raise RuntimeError(
            "Training artifact is incomplete, stale, or violates held-out policy"
        )
    split_dir = resolve(root, config["paths"]["split_dir"])
    examples, _ = load_validation_protocol(split_dir)
    checkpoint_root = resolve(root, config["paths"]["checkpoints"]) / args.run_id
    open_instruct_root = resolve(root, config["paths"]["open_instruct_root"])
    load_env_file(resolve(root, config["paths"]["openrouter_env"]))
    summaries: list[dict[str, Any]] = []
    for step in CHECKPOINTS:
        state_dir = run_dir / "evaluations" / f"step_{step:05d}"
        cached = load_completed_checkpoint(
            state_dir,
            step=step,
            expected_rows=len(examples),
            rows_file="reward_rows.jsonl",
            required_files=(
                "predictions.jsonl",
                "reward_calls.jsonl",
                "task_evaluation/summary.json",
                "task_evaluation/evaluation_manifest.json",
            ),
        )
        if cached is not None:
            summaries.append(cached)
            continue
        state_dir.mkdir(parents=True, exist_ok=False)
        predictions = generate_predictions(
            config,
            root=root,
            examples=examples,
            adapter=adapter_for_step(checkpoint_root, step),
        )
        predictions_path = state_dir / "predictions.jsonl"
        write_jsonl(
            predictions_path,
            [{"id": row["id"], "output": row["output"]} for row in predictions],
        )
        result = evaluate_validation(
            split_dir=split_dir,
            predictions_path=predictions_path,
            open_instruct_root=open_instruct_root,
            output_dir=state_dir / "task_evaluation",
        )
        attach_validation_rewards(
            config=config,
            rows=predictions,
            log_path=state_dir / "reward_calls.jsonl",
            prompt_builder=attribution_prompt,
            prompt_template_hash=template_hashes()[ATTRIBUTION_PROMPT_VERSION],
        )
        write_jsonl(state_dir / "reward_rows.jsonl", predictions)
        passed_by_id = {str(row["id"]): bool(row["passed"]) for row in result["rows"]}
        labels = [passed_by_id[str(row["id"])] for row in predictions]
        state_summary = {
                "step": step,
                **result["summary"],
                "mean_p_own": statistics.fmean(
                    float(row["p_own"]) for row in predictions
                ),
                "mean_reward": statistics.fmean(
                    float(row["reward"]) for row in predictions
                ),
                "sampled_fooling": statistics.fmean(
                    str(row["judge_token"]) == "1" for row in predictions
                ),
                "argmax_fooling": statistics.fmean(
                    float(row["delta"]) > 0 for row in predictions
                ),
                "reward_auc_for_correctness": binary_auc(
                    labels, [float(row["p_own"]) for row in predictions]
                ),
                "official_test_accessed": False,
            }
        write_json(state_dir / "summary.json", state_summary)
        summaries.append(state_summary)
    selection = select_checkpoint(summaries)
    write_json(run_dir / "validation_checkpoint_summaries.json", {"rows": summaries})
    write_json(run_dir / "validation_checkpoint_selection.json", selection)
    print(json.dumps(selection, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
