from __future__ import annotations

import asyncio
import json
from pathlib import Path

from iar_campaign.baseline_rewards import (
    LocalOracleReward,
    quality_prompt,
    quality_prompt_template_sha256,
)


def test_quality_prompt_is_reference_free_and_identity_free() -> None:
    prompt = quality_prompt("Question: 1+1?", "Answer: 2")
    assert "Question: 1+1?" in prompt
    assert "Answer: 2" in prompt
    assert "reference answer" in prompt
    assert "authorship" in prompt
    assert "Return exactly one character" in prompt
    assert len(quality_prompt_template_sha256()) == 64


def test_cqa_oracle_runtime_scores_and_audits(tmp_path: Path) -> None:
    runtime = LocalOracleReward(
        domain="commonsenseqa",
        examples=[{"id": "q1", "answer_key": "B"}],
        log_path=tmp_path / "oracle.jsonl",
        max_reward_calls=2,
    )
    rewards = asyncio.run(
        runtime.score_batch(
            completions=["Answer: B", "Answer: C"],
            task_statement=["task", "task"],
            task_id=["q1", "q1"],
            dataset_index=[3, 3],
        )
    )
    runtime.assert_complete()
    rows = [json.loads(line) for line in (tmp_path / "oracle.jsonl").read_text().splitlines()]
    assert rewards == [1.0, 0.0]
    assert [row["reference_exposed_to_actor"] for row in rows] == [False, False]
    assert runtime.reward_statistics()["by_task"]["q1"]["n"] == 2
