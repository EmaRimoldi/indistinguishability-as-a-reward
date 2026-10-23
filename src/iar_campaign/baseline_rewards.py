from __future__ import annotations

import asyncio
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from iar_grpo.reward import completion_to_text

QUALITY_PROMPT_VERSION = "campaign-direct-quality-v1-no-reference-no-system"
QUALITY_PROMPT_TEMPLATE = """You are a strict task-solution correctness evaluator.

Judge only whether the candidate fully and correctly solves the supplied task. Do not judge authorship, style, similarity to another model, or whether you would have written it. You are not given a reference answer. Return exactly one character: 1 for a fully correct solution, or 0 otherwise.

Task:
<TASK>

Candidate solution:
<CANDIDATE>

Label (0 or 1):"""


def quality_prompt(task_statement: str, candidate: str) -> str:
    return QUALITY_PROMPT_TEMPLATE.replace("<TASK>", task_statement.strip()).replace(
        "<CANDIDATE>", candidate.strip()
    )


def quality_prompt_template_sha256() -> str:
    return hashlib.sha256(QUALITY_PROMPT_TEMPLATE.encode()).hexdigest()


class LocalOracleReward:
    """Deterministic privileged task-verifier reward with the live-runtime interface."""

    def __init__(
        self,
        *,
        domain: str,
        examples: list[dict[str, Any]],
        log_path: Path,
        max_reward_calls: int,
        kodcode_timeout_seconds: int = 60,
        spider_database_root: Path | None = None,
        max_concurrency: int = 4,
    ) -> None:
        if domain not in {"commonsenseqa", "gsm8k", "kodcode", "spider"}:
            raise ValueError(f"Unsupported oracle domain: {domain}; MBPP is forbidden")
        if max_reward_calls <= 0 or max_concurrency <= 0:
            raise ValueError("Oracle reward budgets must be positive")
        self.domain = domain
        self.examples = {str(row["id"]): row for row in examples}
        if len(self.examples) != len(examples):
            raise ValueError("Oracle examples have duplicate IDs")
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_reward_calls = max_reward_calls
        self.kodcode_timeout_seconds = kodcode_timeout_seconds
        self.spider_database_root = spider_database_root
        self.max_concurrency = max_concurrency
        self.logical_calls = 0
        self.rows: list[dict[str, Any]] = []

    def _score_one(self, example: dict[str, Any], response: str) -> dict[str, Any]:
        if self.domain == "commonsenseqa":
            from iar_cqa.scoring import score_response

            return score_response(response, str(example["answer_key"]))
        if self.domain == "gsm8k":
            from iar_gsm8k.scoring import score_response

            return score_response(response, str(example["answer_value"]))
        if self.domain == "kodcode":
            from iar_campaign.reward_pilot import _score_kodcode

            return _score_kodcode(
                (
                    response,
                    str(example["hidden_tests"]),
                    self.kodcode_timeout_seconds,
                )
            )
        if self.spider_database_root is None:
            raise ValueError("Spider oracle reward requires a database root")
        from iar_campaign.score_baselines import _score_spider_one

        value = _score_spider_one(
            (example, {"response": response}, self.spider_database_root)
        )
        return dict(value["score"])

    async def score_batch(
        self,
        *,
        completions: list[Any],
        task_statement: list[str],
        task_id: list[str],
        dataset_index: list[int],
        rollout_index: list[int] | None = None,
    ) -> list[float]:
        size = len(completions)
        if not (len(task_statement) == len(task_id) == len(dataset_index) == size):
            raise ValueError("Oracle metadata and completions must have equal lengths")
        if rollout_index is not None and len(rollout_index) != size:
            raise ValueError("rollout_index and completions must have equal lengths")
        if self.logical_calls + size > self.max_reward_calls:
            raise RuntimeError("Oracle reward-call budget exceeded")
        first_call = self.logical_calls + 1
        self.logical_calls += size
        semaphore = asyncio.Semaphore(self.max_concurrency)
        within_task: defaultdict[str, int] = defaultdict(int)

        async def score(offset: int) -> tuple[float, dict[str, Any]]:
            row_id = str(task_id[offset])
            example = self.examples.get(row_id)
            if example is None:
                raise RuntimeError(f"Unknown oracle task ID: {row_id}")
            response = completion_to_text(completions[offset]).strip()
            async with semaphore:
                outcome = await asyncio.to_thread(self._score_one, example, response)
            reward = float(bool(outcome["correct"]))
            if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
                raise RuntimeError(f"Invalid oracle reward for {row_id}: {reward}")
            current_rollout = (
                int(rollout_index[offset])
                if rollout_index is not None
                else within_task[row_id]
            )
            within_task[row_id] += 1
            return reward, {
                "call_id": first_call + offset,
                "task_id": row_id,
                "dataset_index": int(dataset_index[offset]),
                "rollout_index": current_rollout,
                "candidate": response,
                "reward": reward,
                "correct": bool(outcome["correct"]),
                "valid": bool(outcome.get("valid", True)),
                "status": str(outcome.get("status", "scored")),
                "reward_value": "oracle_task_utility",
                "reference_exposed_to_actor": False,
                "official_test_accessed": False,
            }

        results = await asyncio.gather(*(score(offset) for offset in range(size)))
        new_rows = [row for _, row in results]
        with self.log_path.open("a", encoding="utf-8") as handle:
            for row in new_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self.rows.extend(new_rows)
        return [reward for reward, _ in results]

    def assert_complete(self) -> None:
        if self.logical_calls != self.max_reward_calls:
            raise RuntimeError(
                f"Expected {self.max_reward_calls} oracle rewards, observed {self.logical_calls}"
            )

    def audit_summary(self) -> dict[str, Any]:
        return {
            "logical_reward_calls": self.logical_calls,
            "http_requests": 0,
            "max_reward_calls": self.max_reward_calls,
            "reward_value": "oracle_task_utility",
            "domain": self.domain,
        }

    def reward_statistics(self) -> dict[str, Any]:
        by_task: defaultdict[str, list[float]] = defaultdict(list)
        for row in self.rows:
            by_task[str(row["task_id"])].append(float(row["reward"]))
        rewards = [float(row["reward"]) for row in self.rows]
        return {
            "n": len(rewards),
            "mean": statistics.fmean(rewards) if rewards else None,
            "status_counts": dict(Counter(str(row["status"]) for row in self.rows)),
            "by_task": {
                task_id: {"n": len(values), "mean": statistics.fmean(values)}
                for task_id, values in sorted(by_task.items())
            },
        }
