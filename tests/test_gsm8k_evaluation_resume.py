from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

from iar_gsm8k.evaluate_short import attach_rewards


def _call(call_id: int, task_id: str, reward: float) -> dict:
    return {
        "call_id": call_id,
        "http_request_id": call_id,
        "status": "ok",
        "task_id": task_id,
        "reward": reward,
        "p_own": 0.4 + call_id / 10,
        "logp_0": -1.0,
        "logp_1": reward,
        "delta": reward + 1.0,
        "binary_label_mass": 0.999,
        "judge_token": "1",
        "raw_model": "deepseek/deepseek-v4-flash",
        "raw_provider": "DigitalOcean",
    }


class FakeReward:
    def __init__(self, *, settings, log_path, call_id_offset=0, http_request_id_offset=0, **_):
        self.settings = settings
        self.log_path = log_path
        self.call_id_offset = call_id_offset
        self.http_request_id_offset = http_request_id_offset
        self.completed = 0

    async def score_batch(self, *, completions, task_statement, task_id, dataset_index):
        assert len(completions) == len(task_statement) == len(task_id) == len(dataset_index) == 1
        row = _call(self.call_id_offset + 1, task_id[0], -0.25)
        row["http_request_id"] = self.http_request_id_offset + 1
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        self.completed = 1
        return [row["reward"]]

    def assert_complete(self):
        assert self.completed == self.settings.max_reward_calls


def test_attach_rewards_resumes_a_contiguous_successful_prefix() -> None:
    project = Path(__file__).parents[1]
    config = yaml.safe_load((project / "configs" / "gsm8k_grpo_short_v1.yaml").read_text())
    rows = [
        {"id": "a", "response": "one", "task_statement": "q1", "source_index": 1},
        {"id": "b", "response": "two", "task_statement": "q2", "source_index": 2},
    ]
    with tempfile.TemporaryDirectory() as directory:
        log_path = Path(directory) / "reward_calls.jsonl"
        log_path.write_text(json.dumps(_call(1, "a", -0.5)) + "\n", encoding="utf-8")
        with patch("iar_gsm8k.evaluate_short.LiveOpenRouterReward", FakeReward):
            attach_rewards(config, rows, log_path)

    assert rows[0]["reward"] == -0.5
    assert rows[1]["reward"] == -0.25
    assert rows[0]["raw_provider"] == rows[1]["raw_provider"] == "DigitalOcean"
