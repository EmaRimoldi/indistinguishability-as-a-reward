from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from iar_grpo.analyze_hpo import analyze_study, reward_log_metrics


class AnalyzeHpoTest(unittest.TestCase):
    def test_distinguishes_response_log_alignment_from_argmax_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "reward_calls.jsonl"
            rows = [
                reward_row("1", "1", 1.0),
                reward_row("0", "0", 1.0),
                reward_row("0", "0", -1.0),
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            metrics = reward_log_metrics(path)
            self.assertEqual(metrics["instrumented_token_count"], 3)
            self.assertEqual(metrics["emitted_logprob_token_mismatches"], 0)
            self.assertEqual(metrics["emitted_argmax_mismatches"], 1)

    def test_reports_soft_and_hard_fooling_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            study = Path(temp_dir)
            run = study / "trials/hpo_logp_trial_0003_seed17"
            run.mkdir(parents=True)
            config = {
                "project": {"seed": 17},
                "hpo": {
                    "trial_number": 3,
                    "source": "test",
                    "parameters": {
                        "learning_rate": 1e-5,
                        "beta": 1e-3,
                        "temperature": 0.6,
                        "lora_r": 32,
                        "lora_alpha": 32,
                        "lora_dropout": 0.0,
                        "scale_rewards": "group",
                    },
                },
            }
            (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
            objective = {
                "objective_value": 0.01,
                "selected_step": 100,
                "quality_feasible": True,
                "comparisons": [
                    {
                        "step": 100,
                        "p_own_delta_mean": 0.04,
                        "p_own_delta_lcb": 0.01,
                        "pass_at_1_delta": 0.02,
                        "net_passed_change": 2,
                    }
                ],
            }
            (run / "objective.json").write_text(json.dumps(objective), encoding="utf-8")
            evaluation = {
                "validation": [
                    state(0, 0.2, 0.1, 0.4, 40),
                    state(100, 0.24, 0.3, 0.42, 42),
                ]
            }
            (run / "hpo_evaluation_summary.json").write_text(
                json.dumps(evaluation), encoding="utf-8"
            )
            (run / "reward_calls.jsonl").write_text(
                "".join(
                    json.dumps({"usage": {"cost": cost}}) + "\n"
                    for cost in [0.01, 0.02]
                ),
                encoding="utf-8",
            )

            outputs = analyze_study(study)
            rows = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
            self.assertEqual(rows[0]["p_own_delta"], 0.04)
            self.assertAlmostEqual(rows[0]["hard_fooling_delta"], 0.2)
            self.assertEqual(rows[0]["net_passed_change"], 2)
            self.assertAlmostEqual(rows[0]["total_openrouter_cost"], 0.23)
            costs = json.loads(Path(outputs["cost_summary"]).read_text())
            self.assertAlmostEqual(costs["total_openrouter_cost"], 0.23)
            self.assertIn("Δ sampled fooling", Path(outputs["markdown"]).read_text())


def state(step: int, p_own: float, fooling: float, pass_at_1: float, passed: int) -> dict:
    return {
        "step": step,
        "p_own_mean": p_own,
        "fooling_rate": fooling,
        "argmax_fooling_rate": fooling,
        "pass@1": pass_at_1,
        "passed": passed,
        "reward_usage": {"cost": 0.1},
        "logprob_consistency": {"contradictions": 0},
    }


def reward_row(judge: str, logged: str, delta: float) -> dict:
    return {
        "status": "ok",
        "judge_token": judge,
        "logprob_first_token": logged,
        "delta": delta,
        "binary_label_mass": 0.999,
    }


if __name__ == "__main__":
    unittest.main()
