from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import ClassVar

import pytest

import iar_campaign.pairwise_reward as pairwise_module
from iar_campaign.pairwise_reward import (
    PAIRWISE_PROMPT_VERSION,
    PairwiseOpenRouterReward,
    pairwise_prompt,
    pairwise_prompt_template_sha256,
)
from iar_grpo.config import GrpoTrialConfig, validate_grpo_config
from iar_grpo.reward import RewardSettings


def test_pairwise_prompt_freezes_binary_order_contract() -> None:
    prompt = pairwise_prompt("Solve 2+2", "answer A", "answer B")
    assert "Solve 2+2" in prompt
    assert "Response 1:\nanswer A" in prompt
    assert "Response 2:\nanswer B" in prompt
    assert "1 if Response 1" in prompt
    assert "0 if Response 2" in prompt
    assert "<TASK>" not in prompt
    assert len(pairwise_prompt_template_sha256()) == 64


class _FakeLiveReward:
    instances: ClassVar[list[_FakeLiveReward]] = []

    def __init__(self, *, settings, **kwargs) -> None:
        self.settings = settings
        self.logical_calls = 0
        self.http_requests = 0
        self.index = len(self.instances)
        self.instances.append(self)

    async def score_batch(self, *, completions, **kwargs):
        self.logical_calls += len(completions)
        self.http_requests += len(completions)
        value = 0.8 if self.index == 0 else 0.7
        return [value] * len(completions)

    def assert_complete(self) -> None:
        assert self.logical_calls == self.settings.max_reward_calls

    def audit_summary(self):
        return {"logical_reward_calls": self.logical_calls}


def test_pairwise_runtime_averages_actor_probability_across_orders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _FakeLiveReward.instances = []
    monkeypatch.setattr(pairwise_module, "LiveOpenRouterReward", _FakeLiveReward)
    settings = RewardSettings(
        model_id="deepseek/deepseek-v4-flash",
        target_model_name="DeepSeek V4 Flash",
        max_reward_calls=1,
        max_http_requests=10,
        max_attempts_per_call=5,
        max_concurrency=1,
        timeout_seconds=5,
        temperature=0,
        max_tokens=1,
        top_logprobs=5,
        strict_logprobs=True,
        provider_routing={"order": ["StreamLake"]},
        request_overrides={},
        expected_provider="StreamLake",
        reward_value="log_p_pair",
        attribution_prompt_version=PAIRWISE_PROMPT_VERSION,
        minimum_request_interval_seconds=0,
    )
    runtime = PairwiseOpenRouterReward(
        domain="gsm8k",
        settings=settings,
        teacher_by_task={"task": "teacher"},
        log_dir=tmp_path,
        api_key="test",
    )
    rewards = asyncio.run(
        runtime.score_batch(
            completions=["actor"],
            task_statement=["task"],
            task_id=["q1"],
            dataset_index=[1],
        )
    )
    runtime.assert_complete()
    assert rewards == pytest.approx([math.log(0.55 + 1.0e-6)])
    assert runtime.rows[0]["p_actor_when_first"] == pytest.approx(0.8)
    assert runtime.rows[0]["p_actor_when_second"] == pytest.approx(0.3)
    assert runtime.audit_summary()["http_requests"] == 2


def test_grpo_config_accounts_for_two_http_calls_per_pairwise_reward() -> None:
    raw = {
        "project": {"root": "/tmp", "seed": 23},
        "paths": {"hf_cache": "/tmp/cache"},
        "dataset": {"selection": "frozen"},
        "trial": {"prompt_count": 2, "expected_reward_calls": 16},
        "training": {
            "num_epochs": 2,
            "num_generations": 4,
            "generation_batch_size": 8,
            "per_device_train_batch_size": 4,
            "world_size": 1,
            "max_steps": 4,
            "temperature": 0.6,
            "save_strategy": "no",
        },
        "reward": {
            "value": "log_p_pair",
            "max_reward_calls": 16,
            "max_attempts_per_call": 5,
            "max_http_requests": 160,
            "epsilon": 1.0e-6,
            "minimum_request_interval_seconds": 0,
            "max_retry_wait_seconds": 30,
        },
        "discriminator": {"attribution_prompt_version": PAIRWISE_PROMPT_VERSION},
    }
    validate_grpo_config(GrpoTrialConfig(raw=raw, path=Path("/tmp/config.yaml")))
    raw["reward"]["max_http_requests"] = 80
    with pytest.raises(ValueError, match="HTTP calls per reward"):
        validate_grpo_config(GrpoTrialConfig(raw=raw, path=Path("/tmp/config.yaml")))
