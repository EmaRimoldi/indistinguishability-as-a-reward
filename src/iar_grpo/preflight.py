from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics

from iar_phase0.env import load_env_file

from .config import load_grpo_config
from .data import build_trial_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an audited GRPO pilot without launching training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--require-training-deps", action="store_true")
    args = parser.parse_args()

    config = load_grpo_config(args.config)
    dataset, rows = build_trial_dataset(config)
    env_path = config.resolve(config.raw["paths"]["openrouter_env"])
    load_env_file(env_path)
    has_key = bool(os.environ.get("OPENROUTER_API_KEY"))
    if not has_key:
        raise RuntimeError(f"OpenRouter key not configured at {env_path}")

    actor_cache = config.resolve(config.raw["paths"]["hf_cache"])
    actor_source = config.actor_source()

    packages: dict[str, str] = {}
    for package in ["torch", "transformers", "datasets", "accelerate", "trl", "peft"]:
        try:
            module = importlib.import_module(package)
            packages[package] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            packages[package] = "missing"
    if args.require_training_deps and any(packages[name] == "missing" for name in ["trl", "peft"]):
        raise RuntimeError("GRPO dependencies are missing; run scripts/setup_grpo_env.sh")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        actor_source,
        cache_dir=str(actor_cache),
        local_files_only=bool(config.raw["actor"]["local_files_only"]),
        trust_remote_code=False,
    )
    prompt_lengths = [
        len(
            tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True
            )
        )
        for row in rows
    ]
    max_prompt_tokens = int(config.raw["training"]["max_prompt_tokens"])
    if max(prompt_lengths) > max_prompt_tokens:
        raise ValueError(
            f"Prompt length {max(prompt_lengths)} exceeds configured limit {max_prompt_tokens}"
        )
    sorted_lengths = sorted(prompt_lengths)
    p95_index = min(len(sorted_lengths) - 1, int(0.95 * len(sorted_lengths)))

    print(
        json.dumps(
            {
                "status": "ready",
                "dataset_rows": len(dataset),
                "prompt_format": "conversational_chat",
                "task_ids": [row["task_id"] for row in rows],
                "num_generations": config.num_generations,
                "expected_reward_calls": config.expected_reward_calls,
                "openrouter_key_configured": has_key,
                "actor_snapshot": actor_source,
                "prompt_tokens": {
                    "min": min(prompt_lengths),
                    "median": statistics.median(prompt_lengths),
                    "p95": sorted_lengths[p95_index],
                    "max": max(prompt_lengths),
                    "configured_max": max_prompt_tokens,
                },
                "packages": packages,
                "training_will_start": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
