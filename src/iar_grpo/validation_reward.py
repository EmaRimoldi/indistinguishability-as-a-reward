from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from .reward import LiveOpenRouterReward, settings_from_config


def binary_auc(labels: list[bool], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores, strict=True) if label]
    negatives = [
        score for label, score in zip(labels, scores, strict=True) if not label
    ]
    if not positives or not negatives:
        return None
    wins = sum(
        float(positive > negative) + 0.5 * float(positive == negative)
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def attach_validation_rewards(
    *,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    log_path: Path,
    prompt_builder: Callable[[str, str], str],
    prompt_template_hash: str,
    completion_key: str = "output",
) -> None:
    """Attach audited discriminator evidence to one fresh validation state."""
    if log_path.exists():
        raise FileExistsError(
            f"Refusing to append ambiguous validation log: {log_path}"
        )
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
        prompt_builder=prompt_builder,
        prompt_template_hash=prompt_template_hash,
        system_prompt_builder=lambda _name: None,
        candidate_normalizer=str.strip,
    )
    rewards = asyncio.run(
        runtime.score_batch(
            completions=[str(row[completion_key]) for row in rows],
            task_statement=[str(row["task_statement"]) for row in rows],
            task_id=[str(row["id"]) for row in rows],
            dataset_index=[int(row["source_index"]) for row in rows],
        )
    )
    runtime.assert_complete()
    with log_path.open(encoding="utf-8") as handle:
        calls = [json.loads(line) for line in handle if line.strip()]
    calls = sorted(
        (row for row in calls if row.get("status") == "ok"),
        key=lambda row: int(row["call_id"]),
    )
    if len(calls) != len(rows):
        raise RuntimeError(
            f"Expected {len(rows)} successful validation rewards, found {len(calls)}"
        )
    for row, reward, call in zip(rows, rewards, calls, strict=True):
        row.update(
            {
                "reward": float(reward),
                "p_own": float(call["p_own"]),
                "logp_0": float(call["logp_0"]),
                "logp_1": float(call["logp_1"]),
                "delta": float(call["delta"]),
                "judge_token": str(call["judge_token"]),
                "raw_model": str(call["raw_model"]),
                "raw_provider": str(call["raw_provider"]),
            }
        )
