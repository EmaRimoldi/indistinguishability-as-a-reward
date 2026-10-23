from pathlib import Path

import yaml

from scripts.grpo_numerical_canary import DeterministicLocalReward, canary_config


def test_canary_config_preserves_one_pass_and_disables_http_budget():
    source = yaml.safe_load(Path("configs/reward_pilot_cqa_base.yaml").read_text())
    config = canary_config(
        source,
        Path("/tmp/iar-numerical-canary"),
        100,
        project_root=Path("/tmp/iar-source"),
        hf_cache=Path("/tmp/iar-cache"),
        seed=17,
        max_completion_length=768,
        split_dir=Path("/tmp/iar-frozen-cqa"),
    )
    assert config["project"]["root"] == str(Path("/tmp/iar-source").resolve())
    assert config["paths"]["hf_cache"] == str(Path("/tmp/iar-cache").resolve())
    assert config["dataset"]["prompt_count"] == 100
    assert config["training"]["max_steps"] == 100
    assert config["training"]["save_strategy"] == "no"
    assert config["training"]["generation_batch_size"] == 8
    assert config["trial"]["expected_reward_calls"] == 400
    assert config["reward"]["max_attempts_per_call"] == 1
    assert config["evaluation"]["checkpoints"] == [0, 100]
    assert config["project"]["seed"] == 17
    assert config["training"]["max_completion_length"] == 768
    assert config["paths"]["split_dir"] == str(Path("/tmp/iar-frozen-cqa").resolve())
    assert config["dataset"]["source_file"] == "train.jsonl"


def test_kodcode_canary_uses_probe_and_can_ablate_kl():
    source = yaml.safe_load(Path("configs/reward_pilot_cqa_base.yaml").read_text())
    config = canary_config(
        source,
        Path("/tmp/iar-numerical-kodcode"),
        100,
        domain="kodcode",
        beta=0.0,
        generation_batch_size=40,
        split_dir=Path("/tmp/iar-kodcode"),
    )
    assert config["trial"]["prompt_count"] == 100
    assert config["dataset"]["source_file"] == "probe100.jsonl"
    assert config["training"]["max_steps"] == 100
    assert config["training"]["beta"] == 0.0
    assert config["training"]["generation_batch_size"] == 40
    assert config["trial"]["expected_reward_calls"] == 400
    assert config["numerical_canary"]["preserve_full_training_pool"] is False


def test_canary_rejects_inexact_generation_batches():
    import pytest

    source = yaml.safe_load(Path("configs/reward_pilot_cqa_base.yaml").read_text())
    with pytest.raises(ValueError, match="must divide the rollout budget"):
        canary_config(
            source,
            Path("/tmp/iar-numerical-kodcode"),
            18,
            domain="kodcode",
            generation_batch_size=40,
            split_dir=Path("/tmp/iar-kodcode"),
        )


def test_local_reward_is_deterministic_and_finite():
    import asyncio
    import math

    runtime = DeterministicLocalReward("log_p_own")
    kwargs = {
        "completions": ["Answer: A", "Answer: B"],
        "task_statement": ["x", "x"],
        "task_id": ["one", "one"],
        "dataset_index": [1, 1],
    }
    first = asyncio.run(runtime.score_batch(**kwargs))
    second = asyncio.run(runtime.score_batch(**kwargs))
    assert first == second
    assert first[0] != first[1]
    assert all(math.isfinite(value) for value in first)
