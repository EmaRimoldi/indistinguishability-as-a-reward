import hashlib
import json
from pathlib import Path

import pytest

import iar_campaign.reward_pilot as pilot
from iar_grpo.config import GrpoTrialConfig


def minimal_config(root: Path, domain: str = "kodcode") -> GrpoTrialConfig:
    frozen = "frozen_seed23_v1" if domain == "kodcode" else "frozen_historical_seed23"
    validation = "validation300.jsonl" if domain == "kodcode" else "validation.jsonl"
    split_root = root / "01_data_protocol_runtime_gates" / "datasets" / domain / frozen
    split_root.mkdir(parents=True)
    (split_root / "pilot374.jsonl").write_text("", encoding="utf-8")
    (split_root / validation).write_text("", encoding="utf-8")
    scoring_protocol = root / "04_reward_transform_selection" / "results" / "kodcode.json"
    scoring_protocol.parent.mkdir(parents=True, exist_ok=True)
    scoring_protocol.write_text(
        json.dumps(
            {
                "decision": "GO",
                "validation_file_sha256": hashlib.sha256(b"").hexdigest(),
                "scored_ids": ["kept"],
            }
        ),
        encoding="utf-8",
    )
    baseline = (
        root
        / "04_reward_transform_selection"
        / "arms"
        / domain
        / "log_p_own"
        / "runs"
        / "shared_baseline"
        / "evaluations"
        / "validation"
        / "step_0000"
        / "summary.json"
    )
    raw = {
        "campaign": {"domain": domain},
        "project": {"seed": 23, "root": str(root)},
        "paths": {
            "experiments": str(root / "04_reward_transform_selection" / "runs"),
            "checkpoints": str(root / "04_reward_transform_selection" / "checkpoints"),
        },
        "dataset": {"split_file": str(split_root / "pilot374.jsonl")},
        "trial": {"prompt_count": 374, "expected_reward_calls": 1496},
        "actor": {
            "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        },
        "reward": {"value": "log_p_own"},
        "training": {"num_epochs": 1, "num_generations": 4, "max_steps": 374},
        "evaluation": {
            "checkpoints": [0, 187, 374],
            "validation_count": 300,
            "validation_file": str(split_root / validation),
            "baseline_summary": str(baseline),
            "select_on": "validation_task_utility",
            "max_new_tokens": 768,
            "kodcode_scoring_protocol": str(scoring_protocol),
        },
    }
    return GrpoTrialConfig(raw=raw, path=root / "pilot.yaml")


@pytest.mark.parametrize("domain", pilot.DOMAINS)
def test_frozen_reward_pilot_contract(tmp_path: Path, monkeypatch, domain: str):
    monkeypatch.setattr(pilot, "CAMPAIGN_ROOT", tmp_path)
    config = minimal_config(tmp_path, domain)
    observed_domain, split, validation = pilot.validate_campaign_config(config)
    assert observed_domain == domain
    assert split.name == "pilot374.jsonl"
    assert validation.name in {"validation.jsonl", "validation300.jsonl"}


def test_reward_pilot_rejects_test_as_validation(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(pilot, "CAMPAIGN_ROOT", tmp_path)
    config = minimal_config(tmp_path)
    config.raw["evaluation"]["validation_file"] = str(tmp_path / "sealed_test1000.jsonl")
    with pytest.raises(ValueError, match="validation300"):
        pilot.validate_campaign_config(config)


def test_secret_loader_accepts_a_fourth_numbered_lane(tmp_path: Path):
    key_pool = tmp_path / "keys.env"
    key_pool.write_text("OPENROUTER_API_KEY_4=fourth\n", encoding="utf-8")
    assert pilot.load_named_secret(key_pool, "OPENROUTER_API_KEY_4") == "fourth"
    with pytest.raises(ValueError, match="Unsupported key lane"):
        pilot.load_named_secret(key_pool, "ARBITRARY_SECRET")


def test_kodcode_hidden_tests_score_candidate():
    tests = "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5"
    correct = pilot._score_kodcode(("def add(a, b):\n    return a + b", tests, 10))
    wrong = pilot._score_kodcode(("def add(a, b):\n    return a - b", tests, 10))
    assert correct["correct"] is True
    assert correct["valid"] is True
    assert wrong["correct"] is False
    assert wrong["valid"] is True


def test_kodcode_tests_without_solution_import_are_supported():
    tests = "def test_add():\n    assert add(2, 3) == 5"
    result = pilot._score_kodcode(("def add(a, b):\n    return a + b", tests, 10))
    assert result["correct"] is True


def test_kodcode_explicit_import_tests_see_solution_imports():
    response = "import math\n\ndef lcm(a, b):\n    return math.lcm(a, b)"
    tests = (
        "from solution import lcm\n\n"
        "def test_lcm():\n"
        "    assert lcm(2, 3) == math.lcm(2, 3)"
    )
    result = pilot._score_kodcode((response, tests, 10))
    assert result["correct"] is True


def test_reward_selection_prefers_logp_within_one_bootstrap_se():
    observed = {
        "log_p_own": 0.10,
        "p_own": 0.11,
        "logit_p_own": 0.08,
        "binary_own": 0.07,
    }
    bootstrap = {
        "log_p_own": [0.08, 0.12, 0.09, 0.11],
        "p_own": [0.13, 0.09, 0.14, 0.08],
        "logit_p_own": [0.07, 0.09, 0.08, 0.08],
        "binary_own": [0.06, 0.08, 0.07, 0.07],
    }
    result = pilot.select_reward_with_logp_tie_rule(observed, bootstrap)
    assert result["raw_best_reward_transform"] == "p_own"
    assert result["selected_reward_transform"] == "log_p_own"
    assert result["tie_rule_applied"] is True


def test_reward_selection_keeps_clear_raw_winner():
    observed = {
        "log_p_own": 0.10,
        "p_own": 0.30,
        "logit_p_own": 0.08,
        "binary_own": 0.07,
    }
    bootstrap = {
        "log_p_own": [0.09, 0.11, 0.10, 0.10],
        "p_own": [0.29, 0.31, 0.30, 0.30],
        "logit_p_own": [0.07, 0.09, 0.08, 0.08],
        "binary_own": [0.06, 0.08, 0.07, 0.07],
    }
    result = pilot.select_reward_with_logp_tie_rule(observed, bootstrap)
    assert result["selected_reward_transform"] == "p_own"
    assert result["tie_rule_applied"] is False
