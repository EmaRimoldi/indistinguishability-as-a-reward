from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from iar_grpo.config import GrpoTrialConfig, load_grpo_config
from iar_grpo.evaluate_hpo import paired_comparison
from iar_grpo.hpo import materialize_trial, read_yaml, validate_refinement_params


class HpoProtocolTest(unittest.TestCase):
    def test_offline_actor_source_is_a_cached_snapshot_path(self) -> None:
        template = read_yaml(Path("configs/grpo_hpo_trial_template.yaml"))
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template["project"]["root"] = str(root)
            snapshot = (
                root
                / "models/cache/huggingface/models--Qwen--Qwen2.5-1.5B-Instruct"
                / "snapshots/test-snapshot"
            )
            snapshot.mkdir(parents=True)
            config = GrpoTrialConfig(raw=template, path=Path("test.yaml"))
            self.assertEqual(config.actor_source(), str(snapshot))

    def test_short_trial_template_has_fixed_protocol(self) -> None:
        config = load_grpo_config("configs/grpo_hpo_trial_template.yaml")
        self.assertEqual(config.prompt_count, 100)
        self.assertEqual(config.num_generations, 4)
        self.assertEqual(config.expected_reward_calls, 400)
        self.assertEqual(config.raw["evaluation"]["checkpoints"], [0, 50, 100])
        self.assertEqual(config.raw["evaluation"]["validation"]["count"], 90)
        self.assertEqual(config.raw["reward"]["value"], "log_p_own")

    def test_paired_lcb_and_quality_constraint(self) -> None:
        baseline = [row("1", 0.2, False), row("2", 0.3, True), row("3", 0.4, True)]
        candidate = [row("1", 0.4, False), row("2", 0.5, False), row("3", 0.6, True)]
        result = paired_comparison(
            baseline,
            candidate,
            step=50,
            z=1.6448536269514722,
            max_pass_drop=0.02,
        )
        self.assertAlmostEqual(result["p_own_delta_mean"], 0.2)
        self.assertEqual(result["net_passed_change"], -1)
        self.assertFalse(result["quality_feasible"])

    def test_materialized_trial_records_all_optimized_parameters(self) -> None:
        template = read_yaml(Path("configs/grpo_hpo_trial_template.yaml"))
        search = read_yaml(Path("configs/hpo_short_trials.yaml"))
        params = {
            "learning_rate": 1.0e-5,
            "beta": 1.0e-3,
            "temperature": 0.6,
            "lora_r": 64,
            "lora_alpha": 32,
            "lora_dropout": 0.0,
            "scale_rewards": "group",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            study_dir = Path(temp_dir)
            (study_dir / "configs").mkdir()
            batch_dir = study_dir / "batches" / "test"
            (batch_dir / "results").mkdir(parents=True)
            spec = materialize_trial(
                study_dir=study_dir,
                batch_dir=batch_dir,
                template=template,
                search=search,
                trial_number=7,
                params=params,
                seed=17,
                source="test",
            )
            generated = read_yaml(Path(spec["config_path"]))
            self.assertEqual(generated["training"]["learning_rate"], 1.0e-5)
            self.assertEqual(generated["training"]["beta"], 1.0e-3)
            self.assertEqual(generated["actor"]["lora"]["r"], 64)
            self.assertEqual(generated["reward"]["value"], "log_p_own")
            self.assertEqual(generated["hpo"]["parameters"], params)
            json.dumps(spec)

    def test_refinement_must_use_the_registered_space(self) -> None:
        search = read_yaml(Path("configs/hpo_short_trials.yaml"))
        params = {
            name: values[0] for name, values in search["search"]["space"].items()
        }
        validate_refinement_params(params, search["search"]["space"])
        params["learning_rate"] = 2.0e-5
        with self.assertRaises(ValueError):
            validate_refinement_params(params, search["search"]["space"])


def row(task_id: str, p_own: float, passed: bool) -> dict:
    return {
        "task_id": task_id,
        "p_own": p_own,
        "unit_tests": {"passed": passed},
    }


if __name__ == "__main__":
    unittest.main()
