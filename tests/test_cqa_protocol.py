import unittest
from pathlib import Path

import yaml

from iar_cqa.analyze import go_no_go
from iar_cqa.data import stratified_partition
from iar_cqa.evaluate_short import paired
from iar_cqa.prompts import (
    ATTRIBUTION_PROMPT_VERSION,
    attribution_prompt,
    protocol_manifest,
    task_prompt,
)
from iar_cqa.scoring import accuracy, parse_answer, response_features, score_response
from iar_cqa.train import select_balanced, validate


class CqaProtocolTest(unittest.TestCase):
    def test_task_prompt_has_five_choices_and_no_gold(self):
        prompt = task_prompt("Where?", ["one", "two", "three", "four", "five"])
        self.assertIn("A. one", prompt)
        self.assertIn("E. five", prompt)
        self.assertIn("Answer: X", prompt)

    def test_attribution_is_one_user_message_protocol(self):
        prompt = attribution_prompt("<TASK>", "<RESPONSE>")
        self.assertIn("not to judge whether the response is better in general", prompt)
        self.assertNotIn("Do not base your decision", prompt)
        self.assertEqual(prompt.count("Return exactly one token:"), 1)
        manifest = protocol_manifest()
        self.assertEqual(manifest["message_roles"], ["user"])
        self.assertFalse(manifest["system_prompt_present"])
        self.assertIn(ATTRIBUTION_PROMPT_VERSION, manifest["template_sha256"])

    def test_answer_parser_is_strict_and_uses_last_answer(self):
        self.assertEqual(parse_answer("Reason.\nAnswer: B"), "B")
        self.assertEqual(parse_answer("Reason.\n**Answer: D**"), "D")
        self.assertEqual(parse_answer("Reason.\n__Answer: C__"), "C")
        self.assertEqual(parse_answer("Answer: A\nCorrection.\nAnswer: E"), "E")
        self.assertIsNone(parse_answer("The answer is B"))
        self.assertTrue(score_response("Answer: C", "C")["correct"])

    def test_response_features(self):
        features = response_features("- Option A is weak.\n- Choice B fits.\nAnswer: B")
        self.assertEqual(features["bullets"], 2)
        self.assertTrue(features["has_answer_line"])
        self.assertEqual(features["mentioned_option_letters"], ["A", "B"])

    def test_accuracy_serializes_invalid_predictions_with_a_string_key(self):
        rows = [
            {"score": {"valid": False, "correct": False, "prediction": None}},
            {"score": {"valid": True, "correct": True, "prediction": "A"}},
        ]
        result = accuracy(rows)
        self.assertEqual(result["prediction_counts"], {"INVALID": 1, "A": 1})

    def test_stratified_split_is_balanced_and_disjoint(self):
        rows = []
        for label in "ABCDE":
            for index in range(100):
                rows.append({"id": f"{label}-{index}", "answer_key": label})
        split = stratified_partition(rows, seed=23, sizes={"audit": 50, "train": 100})
        for values, expected in ((split["audit"], 10), (split["train"], 20)):
            counts = {label: sum(row["answer_key"] == label for row in values) for label in "ABCDE"}
            self.assertEqual(set(counts.values()), {expected})
        self.assertFalse({r["id"] for r in split["audit"]} & {r["id"] for r in split["train"]})

    def test_phase0_judge_concurrency_is_positive(self):
        config_path = Path(__file__).parents[1] / "configs" / "cqa_phase0_v1.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertGreater(int(config["discriminator"]["transport"]["max_concurrency"]), 0)

    def test_short_training_config_and_subset_are_frozen(self):
        config_path = Path(__file__).parents[1] / "configs" / "cqa_grpo_short_v1.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        validate(config)
        self.assertEqual(config["reward"]["value"], "absolute_logp_1")
        self.assertEqual(config["training"]["num_generations"], 4)
        self.assertEqual(config["training"]["max_steps"], 100)
        self.assertEqual(config["evaluation"]["checkpoints"], [0, 50, 100])
        self.assertEqual(config["evaluation"]["validation_count"], 300)

        rows = [
            {"id": f"{label}-{index}", "answer_key": label}
            for label in "ABCDE"
            for index in range(40)
        ]
        first = select_balanced(rows, count=100, seed=23)
        second = select_balanced(rows, count=100, seed=23)
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        self.assertEqual(
            {label: sum(row["answer_key"] == label for row in first) for label in "ABCDE"},
            {label: 20 for label in "ABCDE"},
        )

    def test_full_training_config_matches_the_authorized_two_epoch_plan(self):
        config_path = Path(__file__).parents[1] / "configs" / "cqa_grpo_full_v1.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        validate(config)
        self.assertEqual(config["dataset"]["prompt_count"], 1000)
        self.assertEqual(config["training"]["num_epochs"], 2)
        self.assertEqual(config["training"]["num_generations"], 4)
        self.assertEqual(config["trial"]["expected_reward_calls"], 8000)
        self.assertEqual(config["training"]["max_steps"], 2000)
        self.assertEqual(config["evaluation"]["checkpoints"], [0, 1000, 2000])
        self.assertEqual(config["evaluation"]["test_count"], 1221)

    def test_short_evaluation_is_paired_by_dataset_id(self):
        baseline = [
            {"id": "a", "p_own": 0.2, "score": {"correct": False}, "judge_token": "0", "delta": -1},
            {"id": "b", "p_own": 0.4, "score": {"correct": True}, "judge_token": "0", "delta": -1},
        ]
        candidate = [
            {"id": "b", "p_own": 0.7, "score": {"correct": True}, "judge_token": "1", "delta": 1},
            {"id": "a", "p_own": 0.3, "score": {"correct": True}, "judge_token": "0", "delta": -1},
        ]
        result = paired(baseline, candidate, z=0.0)
        self.assertAlmostEqual(result["delta_p_own"], 0.2)
        self.assertEqual(result["net_correct_change"], 1)
        self.assertAlmostEqual(result["sampled_fooling_delta"], 0.5)

    def test_go_no_go_returns_a_machine_readable_decision(self):
        baseline = {
            "groups": {
                "B_own": {"accuracy": {"accuracy": 0.8}, "mean_p_own": 0.7},
                "A0_plain": {
                    "accuracy": {"accuracy": 0.6},
                    "mean_p_own": 0.2,
                    "attribution_auc": 0.8,
                },
            },
            "protocol_audit": {
                "binary_label_mass": {"min": 0.9},
                "emitted_argmax_consistency": 0.8,
            },
        }
        probe = {
            "feature_audit": {"chars": {"within_task_correlation_with_reward": 0.1}},
            "selected_answer_audit": {
                "A": {"within_task_indicator_correlation_with_reward": 0.1}
            },
            "group_signal": {"groups_with_at_least_two_distinct_rewards": 100},
            "correctness_relation": {"correct_beats_incorrect_pairwise_rate": 0.7},
            "protocol_audit": {
                "binary_label_mass": {"min": 0.9},
                "emitted_argmax_consistency": 0.8,
            },
        }
        gates = {
            "minimum_b_minus_a_accuracy": 0.05,
            "minimum_attribution_auc": 0.7,
            "minimum_groups_with_reward_variation": 80,
            "minimum_correct_beats_incorrect_rate": 0.55,
            "maximum_absolute_within_task_surface_correlation": 0.4,
        }
        result = go_no_go(baseline, probe, gates)
        self.assertEqual(result["decision"], "GO")
        self.assertEqual(result["n_passed"], result["n_checks"])


if __name__ == "__main__":
    unittest.main()
