from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone

from iar_phase0.data import write_jsonl
from iar_phase0.generator import strip_code_fences
from iar_phase0.parameters import initialize_parameter_record

from .config import load_grpo_config
from .data import build_trial_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Check GRPO rollout termination without OpenRouter calls")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    config = load_grpo_config(args.config)
    dataset, _ = build_trial_dataset(config)
    actor = config.raw["actor"]
    training = config.raw["training"]
    hf_cache = config.resolve(config.raw["paths"]["hf_cache"])
    run_id = datetime.now(timezone.utc).strftime("grpo_generation_preflight_%Y%m%dT%H%M%SZ")
    out_dir = config.resolve(config.raw["paths"]["experiments"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    initialize_parameter_record(
        out_dir,
        resolved_config=config.raw,
        source_config=config.path,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        actor["model_id"], cache_dir=str(hf_cache), local_files_only=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        actor["model_id"],
        cache_dir=str(hf_cache),
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
    model.eval()
    set_seed(int(config.raw["project"]["seed"]))

    rows = []
    for example in dataset:
        conversations = [example["prompt"] for _ in range(config.num_generations)]
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
                max_new_tokens=int(training["max_completion_length"]),
                do_sample=True,
                temperature=float(training["temperature"]),
                top_p=float(training["top_p"]),
                top_k=0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_length = inputs["input_ids"].shape[1]
        for rollout_index, output in enumerate(outputs):
            completion_ids = output[prompt_length:]
            eos_positions = (completion_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            terminated = len(eos_positions) > 0
            if terminated:
                completion_ids = completion_ids[: int(eos_positions[0]) + 1]
            raw = tokenizer.decode(completion_ids, skip_special_tokens=True)
            rows.append(
                {
                    "task_id": example["task_id"],
                    "dataset_index": int(example["dataset_index"]),
                    "rollout_index": rollout_index,
                    "terminated_with_eos": terminated,
                    "completion_tokens": len(completion_ids),
                    "candidate_raw": raw,
                    "candidate": strip_code_fences(raw),
                }
            )

    write_jsonl(out_dir / "completions.jsonl", rows)
    lengths = [row["completion_tokens"] for row in rows]
    terminated = sum(bool(row["terminated_with_eos"]) for row in rows)
    unique_by_task = {
        task_id: len({row["candidate_raw"] for row in rows if row["task_id"] == task_id})
        for task_id in sorted({row["task_id"] for row in rows}, key=int)
    }
    summary = {
        "run_id": run_id,
        "status": "complete",
        "openrouter_calls": 0,
        "prompt_format": "conversational_chat",
        "n": len(rows),
        "terminated_with_eos": terminated,
        "terminated_rate": terminated / len(rows),
        "truncated": len(rows) - terminated,
        "completion_tokens": {
            "min": min(lengths),
            "mean": statistics.mean(lengths),
            "max": max(lengths),
        },
        "unique_completions_by_task": unique_by_task,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(out_dir), "summary": summary}, indent=2))
    if summary["terminated_rate"] < 0.9:
        raise RuntimeError("Conversational rollout termination remains below the 90% preflight threshold")


if __name__ == "__main__":
    main()
