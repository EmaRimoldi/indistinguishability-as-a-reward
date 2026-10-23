from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import yaml

from iar_cqa.evaluate_full import select_checkpoint as select_cqa
from iar_cqa.train import validate as validate_cqa
from iar_gsm8k.evaluate_official import select_checkpoint as select_gsm
from iar_gsm8k.train import validate as validate_gsm


ROOT = Path(__file__).parents[1]


def rows(p_own: float, accuracy: float, n: int = 20):
    correct = round(accuracy * n)
    return [
        {
            "id": str(index),
            "p_own": p_own,
            "score": {"correct": index < correct},
            "judge_token": "1" if p_own >= 0.5 else "0",
            "delta": 1.0 if p_own >= 0.5 else -1.0,
        }
        for index in range(n)
    ]


def test_new_configs_use_log_p_own_and_full_score_selection():
    cqa = yaml.safe_load((ROOT / "configs" / "cqa_grpo_full_log_p_own_v2.yaml").read_text())
    gsm = yaml.safe_load((ROOT / "configs" / "gsm8k_grpo_official_log_p_own_v2.yaml").read_text())
    validate_cqa(cqa)
    with patch.dict(os.environ, {"IAR_GSM8K_OFFICIAL_AUTHORIZED": "YES"}):
        validate_gsm(gsm)
    for config in (cqa, gsm):
        assert config["reward"]["value"] == "log_p_own"
        assert config["dataset"]["prompt_count"] == 1000
        assert config["training"]["num_epochs"] == 2
        assert config["training"]["num_generations"] == 4
        assert config["training"]["max_steps"] == 2000
        assert config["trial"]["expected_reward_calls"] == 8000
        assert config["evaluation"]["select_on"] == "validation_paired_p_own_lcb_audit_accuracy_only"


def test_audit_only_selection_does_not_gate_on_task_accuracy():
    baseline = rows(0.2, 1.0)
    moderate = rows(0.4, 1.0)
    high_full_score_low_accuracy = rows(0.8, 0.5)
    config = {
        "evaluation": {
            "maximum_accuracy_drop": 0.02,
            "paired_lcb_z": 1.6448536269514722,
            "select_on": "validation_paired_p_own_lcb_audit_accuracy_only",
        }
    }
    for selector in (select_cqa, select_gsm):
        selection = selector(
            baseline,
            {1000: moderate, 2000: high_full_score_low_accuracy},
            config,
        )
        assert selection["selected_step"] == 2000
        assert selection["task_accuracy_role"] == "audit_only"
        assert selection["comparisons"][1]["quality_feasible"] is False
