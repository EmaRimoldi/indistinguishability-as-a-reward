from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from iar_grpo.config import load_grpo_config
from iar_grpo.reward import attribution_prompt_template_sha256, training_attribution_prompt
from iar_grpo.run_reward_probe import summarize_probe_groups
from iar_phase0.data import MbppExample
from iar_phase0.prompts import (
    pointwise_self_attribution_prompt,
    pointwise_self_attribution_template_sha256,
)


def test_phase0_and_grpo_use_byte_identical_attribution_prompt() -> None:
    example = MbppExample(
        split="test",
        index=0,
        task_id="1",
        prompt="Return one.",
        code=None,
        test_setup_code="",
        tests=["assert f() == 1"],
    )
    candidate = "def f():\n    return 1"
    assert pointwise_self_attribution_prompt(example, candidate) == training_attribution_prompt(
        "You are an expert Python programmer, and here is your task: Return one. "
        "Your code should pass these tests:\n\nassert f() == 1",
        candidate,
    )
    assert pointwise_self_attribution_template_sha256() == attribution_prompt_template_sha256()


def test_coder_trial17_preserves_selected_hpo_and_plain_actor_prompt() -> None:
    config = load_grpo_config("configs/coder_grpo_trial17.yaml")
    assert config.raw["actor"]["model_id"] == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert config.raw["dataset"]["phase0_config"] == "configs/coder_plain_test.yaml"
    assert config.raw["reward"]["value"] == "log_p_own"
    assert config.raw["training"]["num_generations"] == 4
    assert config.raw["training"]["learning_rate"] == 3.0e-5
    assert config.raw["training"]["beta"] == 3.0e-3
    assert config.raw["training"]["scale_rewards"] == "group"
    assert config.raw["evaluation"]["checkpoints"] == [0, 50, 100]
    assert config.raw["reward_probe"]["prompt_count"] == 25


def test_reward_probe_reports_groupwise_signal_and_route_provenance() -> None:
    rows = [
        {
            "status": "ok",
            "task_id": "1",
            "rollout_index": 0,
            "candidate": "a",
            "p_own": 0.2,
            "binary_label_mass": 1.0,
            "raw_model": "deepseek/deepseek-v4-flash",
            "raw_provider": "DeepSeek",
        },
        {
            "status": "ok",
            "task_id": "1",
            "rollout_index": 1,
            "candidate": "b",
            "p_own": 0.8,
            "binary_label_mass": 1.0,
            "raw_model": "deepseek/deepseek-v4-flash",
            "raw_provider": "DeepSeek",
        },
        {
            "status": "ok",
            "task_id": "2",
            "rollout_index": 0,
            "candidate": "c",
            "p_own": 0.1,
            "binary_label_mass": 1.0,
            "raw_model": "deepseek/deepseek-v4-flash",
            "raw_provider": "DeepSeek",
        },
        {
            "status": "ok",
            "task_id": "2",
            "rollout_index": 1,
            "candidate": "d",
            "p_own": 0.9,
            "binary_label_mass": 1.0,
            "raw_model": "deepseek/deepseek-v4-flash",
            "raw_provider": "DeepSeek",
        },
    ]
    audit = summarize_probe_groups(rows, expected_rollouts=2)
    assert audit["all_groups_complete"] is True
    assert audit["all_complete_groups_have_nonzero_p_own_variance"] is True
    assert audit["raw_models"] == {"deepseek/deepseek-v4-flash": 4}
    assert audit["raw_providers"] == {"DeepSeek": 4}


def test_stratified_bootstrap_ci_is_reproducible() -> None:
    script_path = Path("scripts/summarize_pointwise_attribution.py")
    module_spec = importlib.util.spec_from_file_location("pointwise_summary", script_path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    own = [{"score": 0.8}, {"score": 0.9}]
    non_own = [{"score": 0.2}, {"score": 0.3}]
    interval = module.stratified_bootstrap_ci(
        own,
        non_own,
        lambda o, n: module.mean([row["score"] for row in o]) - module.mean([row["score"] for row in n]),
        replicates=200,
        seed=7,
    )
    assert interval["point_estimate"] == pytest.approx(0.6)
    assert interval["lower"] is not None
    assert interval["upper"] is not None
    assert interval["replicates_valid"] == 200
