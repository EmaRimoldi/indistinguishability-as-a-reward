#!/usr/bin/env python3
"""Run a paid-call-free GRPO canary through a production trainer path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class DeterministicLocalReward:
    """Completion-dependent finite reward used only for numerical diagnostics."""

    def __init__(self, value: str) -> None:
        self.value = value

    async def score_batch(
        self,
        *,
        completions: list[str],
        task_statement: list[str],
        task_id: list[str],
        dataset_index: list[int],
    ) -> list[float]:
        del task_statement, dataset_index
        scores: list[float] = []
        for identifier, completion in zip(task_id, completions, strict=True):
            digest = hashlib.sha256(
                f"{identifier}\0{completion}".encode("utf-8")
            ).digest()
            probability = (int.from_bytes(digest[:8], "big") + 1) / (2**64 + 1)
            probability = min(max(probability, 1e-6), 1 - 1e-6)
            if self.value == "log_p_own":
                score = math.log(probability)
            elif self.value == "p_own":
                score = probability
            elif self.value == "logit_p_own":
                score = math.log(probability / (1 - probability))
            else:
                raise ValueError(f"Unsupported canary reward: {self.value}")
            scores.append(score)
        return scores


def canary_config(
    source: dict[str, Any],
    output_root: Path,
    steps: int,
    *,
    domain: str = "cqa",
    beta: float | None = None,
    generation_batch_size: int | None = None,
    project_root: Path | None = None,
    hf_cache: Path | None = None,
    seed: int | None = None,
    max_completion_length: int | None = None,
    split_dir: Path | None = None,
) -> dict[str, Any]:
    if steps <= 0 or steps % 2:
        raise ValueError("Canary steps must be a positive even number")
    config = deepcopy(source)
    if project_root is not None:
        config["project"]["root"] = str(project_root.resolve())
    if hf_cache is not None:
        config["paths"]["hf_cache"] = str(hf_cache.resolve())
    if seed is not None:
        config["project"]["seed"] = int(seed)
    if split_dir is not None:
        config["paths"]["split_dir"] = str(split_dir.resolve())
        if domain == "cqa":
            config["dataset"]["source_file"] = "train.jsonl"
    if domain == "cqa":
        # The CQA canary intentionally uses exactly ``steps`` examples.
        config["dataset"]["prompt_count"] = steps
    elif domain == "kodcode":
        config["trial"]["prompt_count"] = steps
        config["dataset"]["source_file"] = "probe100.jsonl"
    else:
        raise ValueError(f"Unsupported numerical-canary domain: {domain}")
    global_batch = int(config["training"]["per_device_train_batch_size"]) * int(
        config["training"].get("world_size", 1)
    )
    calls = steps * global_batch
    config["trial"]["run_name"] = f"{domain}_grpo_numerical_canary"
    config["trial"]["expected_reward_calls"] = calls
    config["reward"]["max_reward_calls"] = calls
    config["reward"]["max_http_requests"] = calls
    config["reward"]["max_attempts_per_call"] = 1
    config["training"]["num_epochs"] = 1
    config["training"]["max_steps"] = steps
    config["training"]["save_strategy"] = "no"
    if generation_batch_size is not None:
        generations = int(config["training"]["num_generations"])
        if (
            generation_batch_size <= 0
            or generation_batch_size % generations
            or generation_batch_size % global_batch
            or calls % generation_batch_size
        ):
            raise ValueError(
                "Canary generation_batch_size must divide the rollout budget and "
                "contain complete GRPO/global batches"
            )
        config["training"]["generation_batch_size"] = generation_batch_size
    if beta is not None:
        if beta < 0:
            raise ValueError("Canary beta cannot be negative")
        config["training"]["beta"] = float(beta)
    if max_completion_length is not None:
        if max_completion_length <= 0:
            raise ValueError("Canary max_completion_length must be positive")
        config["training"]["max_completion_length"] = int(max_completion_length)
    config["evaluation"]["checkpoints"] = [0, steps]
    config["paths"]["experiments"] = str(output_root / "runs")
    config["paths"]["checkpoints"] = str(output_root / "checkpoints")
    config["numerical_canary"] = {
        "domain": domain,
        "paid_api_calls": 0,
        "steps": steps,
        "seed": int(config["project"]["seed"]),
        "max_completion_length": int(config["training"]["max_completion_length"]),
        "preserve_full_training_pool": False,
    }
    return config


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_domain_inputs(
    domain: str,
    config: dict[str, Any],
    config_path: Path,
) -> tuple[Any, list[dict[str, Any]], Any, Any]:
    """Resolve the domain's real dataset builder and production train function."""

    root = Path(config["project"]["root"]).resolve()
    if domain == "cqa":
        from iar_cqa.train import build_dataset, run_training

        dataset, rows = build_dataset(config, root)
        return dataset, rows, config, run_training
    if domain == "kodcode":
        from datasets import Dataset
        from iar_grpo.config import GrpoTrialConfig
        from iar_grpo.train import run_training
        from iar_phase0.data import read_jsonl

        typed_config = GrpoTrialConfig(raw=config, path=config_path)
        split_dir = Path(config["paths"]["split_dir"]).resolve()
        source = read_jsonl(split_dir / config["dataset"]["source_file"])
        if len(source) != 100:
            raise RuntimeError("KodCode numerical probe must contain exactly 100 rows")
        rows = []
        for item in source:
            question = str(item["question"]).strip()
            task = (
                "You are an expert Python programmer. Solve the following task with a "
                "complete executable Python solution. Return only Python code after "
                f"[BEGIN].\n\nTask:\n{question}\n\n[BEGIN]\n"
            )
            rows.append(
                {
                    "prompt": [{"role": "user", "content": task}],
                    "task_statement": question,
                    "task_id": str(item["id"]),
                    "dataset_index": int(item["source_index"]),
                    "actor_visible_fields": ["question"],
                }
            )
        dataset = Dataset.from_list(rows)
        return dataset, rows, typed_config, run_training
    raise ValueError(f"Unsupported numerical-canary domain: {domain}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--domain", choices=("cqa", "kodcode"), default="cqa")
    parser.add_argument("--beta", type=float)
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument(
        "--reward-value",
        choices=("log_p_own", "p_own", "logit_p_own"),
        default="log_p_own",
    )
    parser.add_argument("--detect-anomaly", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-completion-length", type=int)
    parser.add_argument("--split-dir", type=Path)
    args = parser.parse_args()

    source_config = args.config.resolve()
    source = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    project_root = source_config.parents[1]
    hf_cache = Path(os.environ.get("HF_HOME", source["paths"]["hf_cache"]))
    config = canary_config(
        source,
        output,
        args.steps,
        domain=args.domain,
        beta=args.beta,
        generation_batch_size=args.generation_batch_size,
        project_root=project_root,
        hf_cache=hf_cache,
        seed=args.seed,
        max_completion_length=args.max_completion_length,
        split_dir=args.split_dir,
    )
    config["reward"]["value"] = args.reward_value
    config_path = output / "canary_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    dataset, rows, training_config, run_training = build_domain_inputs(
        args.domain, config, config_path
    )
    experiment_dir = output / "run"
    checkpoint_dir = output / "checkpoints"
    experiment_dir.mkdir()
    checkpoint_dir.mkdir()
    if args.detect_anomaly:
        os.environ["IAR_GRPO_DETECT_ANOMALY"] = "1"
    try:
        if args.domain == "cqa":
            result, history = run_training(
                training_config,
                Path(config["project"]["root"]).resolve(),
                dataset,
                checkpoint_dir,
                experiment_dir,
                DeterministicLocalReward(args.reward_value),
            )
        else:
            result, history = run_training(
                training_config,
                dataset,
                checkpoint_dir,
                experiment_dir,
                DeterministicLocalReward(args.reward_value),
            )
    except Exception as exc:
        write_json(
            output / "failure.json",
            {
                "status": "failed",
                "domain": args.domain,
                "paid_api_calls": 0,
                "steps_requested": args.steps,
                "seed": int(config["project"]["seed"]),
                "max_completion_length": int(config["training"]["max_completion_length"]),
                "beta": float(config["training"]["beta"]),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    numeric_fields = ("loss", "grad_norm", "kl", "entropy")
    bad = [
        {key: row.get(key) for key in numeric_fields if key in row}
        for row in history
        if any(
            key in row and not math.isfinite(float(row[key]))
            for key in numeric_fields
        )
    ]
    summary = {
        "status": "complete" if not bad else "failed",
        "domain": args.domain,
        "paid_api_calls": 0,
        "steps": args.steps,
        "seed": int(config["project"]["seed"]),
        "max_completion_length": int(config["training"]["max_completion_length"]),
        "training_examples": len(rows),
        "reward_value": args.reward_value,
        "beta": float(config["training"]["beta"]),
        "nonfinite_metric_rows": bad,
        "trainer_metrics": result.metrics,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    if bad:
        raise RuntimeError("Numerical canary produced non-finite trainer metrics")


if __name__ == "__main__":
    main()
