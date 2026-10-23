import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

from iar_gsm8k.analyze import SURFACE_FEATURES, baseline_summary, go_no_go, lexical_audit, probe_summary
from iar_gsm8k.data import deterministic_partition
from iar_gsm8k.evaluate_official import select_checkpoint
from iar_gsm8k.prompts import (
    ATTRIBUTION_PROMPT_VERSION,
    attribution_prompt,
    imitation_prompt,
    protocol_manifest,
    task_prompt,
)
from iar_gsm8k.repeatability import icc_one_way
from iar_gsm8k.judge import probe_candidates_path, refresh_candidate_scores
from iar_gsm8k.scoring import (
    normalize_number,
    parse_answer,
    parse_reference_answer,
    response_features,
    score_response,
)
from iar_gsm8k.shortcuts import (
    TRANSFORMS,
    analyze_replicates,
    change_answer_marker,
    compact_whitespace,
    flatten_layout,
    minimize_punctuation,
    remove_conclusion_markers,
)
from iar_gsm8k.train import choose_completion_length, validate


class Gsm8kProtocolTest(unittest.TestCase):
    def test_prompts_are_frozen_one_user_message_without_shortcut_instruction(self):
        task = task_prompt("How many?")
        self.assertEqual(
            task,
            "Question:\nHow many?\n\nSolve the problem step by step.\n\n"
            "End with a separate line in the format:\nAnswer: X\n\n"
            "where X is the final numerical answer.",
        )
        imitation = imitation_prompt("How many?")
        self.assertIn("answer as DeepSeek V4 Flash would", imitation)
        prompt = attribution_prompt(task, "Reasoning.\nAnswer: 2")
        self.assertIn("not to judge whether the solution is better in general", prompt)
        self.assertNotIn("Do not base your decision", prompt)
        self.assertNotIn("punctuation", prompt)
        manifest = protocol_manifest()
        self.assertEqual(manifest["message_roles"], ["user"])
        self.assertFalse(manifest["system_prompt_present"])
        self.assertIn(ATTRIBUTION_PROMPT_VERSION, manifest["template_sha256"])

    def test_numeric_normalization(self):
        cases = {
            "12": "12",
            "-12": "-12",
            "12.500": "12.5",
            "1,234": "1234",
            "$ 1,234.50": "1234.5",
            "-$1,234.50": "-1234.5",
            "€0.50": "0.5",
            "(42)": "-42",
            "($42)": "-42",
            " 0.000 ": "0",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_number(raw), expected)
        for invalid in ("", "one", "1,23", "2 and 3", "1/2", "NaN"):
            with self.subTest(invalid=invalid):
                self.assertIsNone(normalize_number(invalid))

    def test_parser_uses_only_the_last_explicit_answer_line(self):
        self.assertEqual(parse_answer("We use 10 and 20.\nAnswer: $1,234.50"), "1234.5")
        self.assertEqual(parse_answer("Therefore this works. Answer: 10"), "10")
        self.assertEqual(parse_answer("### Answer: -100000"), "-100000")
        self.assertEqual(parse_answer("**Answer: 108**"), "108")
        self.assertEqual(parse_answer("__Answer: 12__"), "12")
        self.assertEqual(parse_answer("Answer: 1\nCorrection.\nAnswer: -2"), "-2")
        self.assertIsNone(parse_answer("The answer is 2"))
        self.assertIsNone(parse_answer("Answer: 2\nCorrection.\nAnswer: two"))
        self.assertTrue(score_response("Work.\nAnswer: 1,200", "1200")["correct"])

    def test_cached_generations_are_rescored_without_changing_text(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            response = "The calculation gives ten. Answer: 10"
            row = {
                "id": "x",
                "response": response,
                "score": {"valid": False, "correct": False, "prediction": None},
                "features": {},
            }
            (run_dir / "actor_probe_k4_responses.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            refresh_candidate_scores(
                run_dir,
                "probe",
                {"x": {"question": "How many?", "answer_value": "10"}},
            )
            refreshed = json.loads((run_dir / "actor_probe_k4_responses.jsonl").read_text())
            self.assertEqual(refreshed["response"], response)
            self.assertTrue(refreshed["score"]["correct"])
            self.assertEqual(refreshed["parser_version"], "explicit-answer-marker-v2")

    def test_reference_parser(self):
        self.assertEqual(parse_reference_answer("Some work.\n#### 1,200"), "1200")
        with self.assertRaises(ValueError):
            parse_reference_answer("No final marker")

    def test_response_features_cover_math_shortcuts(self):
        features = response_features(
            "Step 1: Therefore, 2 + 3 = 5.\nAnswer: 5",
            "If we add 2 and 3, how many?",
        )
        self.assertEqual(features["numbered_steps"], 1)
        self.assertEqual(features["equation_lines"], 1)
        self.assertEqual(features["conclusion_markers"], 1)
        self.assertTrue(features["answer_is_final_nonempty_line"])

    def test_split_is_deterministic_exhaustive_and_disjoint(self):
        rows = [{"id": str(index)} for index in range(50)]
        first = deterministic_partition(rows, seed=23, sizes={"audit": 10, "short": 10})
        second = deterministic_partition(rows, seed=23, sizes={"audit": 10, "short": 10})
        self.assertEqual(
            {name: [row["id"] for row in values] for name, values in first.items()},
            {name: [row["id"] for row in values] for name, values in second.items()},
        )
        ids = [row["id"] for values in first.values() for row in values]
        self.assertEqual(len(ids), 50)
        self.assertEqual(len(set(ids)), 50)
        self.assertEqual(len(first["reserve"]), 30)

    def test_shortcut_transformations_preserve_numeric_content_except_marker_parser(self):
        original = "Step 1: Therefore, 2 + 3 = 5; this is enough.\n\nAnswer: 5"
        for transform in (flatten_layout, compact_whitespace, remove_conclusion_markers, minimize_punctuation):
            with self.subTest(transform=transform.__name__):
                changed = transform(original)
                self.assertNotEqual(changed, original)
                self.assertEqual(parse_answer(changed), "5")
        changed_marker = change_answer_marker(original)
        self.assertIn("Final answer: 5", changed_marker)
        self.assertIsNone(parse_answer(changed_marker))

        inline = "We calculate 2 + 3 = 5. Answer: 5"
        for transform in (flatten_layout, compact_whitespace, remove_conclusion_markers, minimize_punctuation):
            with self.subTest(inline_transform=transform.__name__):
                self.assertEqual(parse_answer(transform(inline)), "5")

    def test_phase0_config_does_not_gate_on_binary_mass_threshold(self):
        path = Path(__file__).parents[1] / "configs" / "gsm8k_phase0_v1.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(config["reward"]["value"], "absolute_logp_1")
        self.assertNotIn("minimum_binary_label_mass", config["gates"])
        self.assertEqual(config["actor"]["probe"]["num_generations"], 4)
        self.assertEqual(config["actor"]["probe"]["length_reaudit_caps"], [768, 1024])
        self.assertNotIn("seed", config["discriminator"]["request_overrides"])
        self.assertEqual(config["discriminator"]["own_generation"]["temperature"], 0)

    def test_resolved_probe_artifact_is_preferred_without_overwriting_original(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            original = run_dir / "actor_probe_k4_responses.jsonl"
            resolved = run_dir / "actor_probe_k4_responses_resolved.jsonl"
            original.write_text("{}\n", encoding="utf-8")
            self.assertEqual(probe_candidates_path(run_dir), original)
            resolved.write_text("{}\n", encoding="utf-8")
            self.assertEqual(probe_candidates_path(run_dir), resolved)

    def test_lexical_audit_is_within_task_and_reports_recurring_tokens(self):
        rows = []
        for task in ("a", "b", "c"):
            for index, token in enumerate(("therefore", "hence", "therefore", "hence")):
                rows.append({"id": task, "response": f"{token} calculate answer", "reward": float(index)})
        audit = lexical_audit(rows, within_task=True)
        tokens = {row["token"] for row in audit["top_tokens"]}
        self.assertIn("therefore", tokens)
        self.assertIn("hence", tokens)
        self.assertIsNotNone(audit["maximum_absolute_correlation"])

    def test_repeatability_icc_separates_stable_from_unstable_scores(self):
        stable = [[0.1, 0.1, 0.1, 0.1], [0.9, 0.9, 0.9, 0.9], [0.5, 0.5, 0.5, 0.5]]
        unstable = [[0.1, 0.9, 0.1, 0.9], [0.9, 0.1, 0.9, 0.1], [0.5, 0.2, 0.8, 0.4]]
        self.assertAlmostEqual(icc_one_way(stable), 1.0)
        self.assertLess(icc_one_way(unstable), 0.0)

    def test_replicated_shortcut_audit_averages_fixed_states(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            primary = []
            additional = []
            for transform in TRANSFORMS:
                for pair_index in range(25):
                    pair_id = f"{transform}:{pair_index:03d}"
                    for state, p_own in (("original", 0.2), ("transformed", 0.25)):
                        base = {
                            "pair_id": pair_id,
                            "pair_state": state,
                            "p_own": p_own,
                            "reward": -1.0 + p_own,
                            "binary_label_mass": 0.999,
                            "emitted_argmax_consistent": True,
                            "raw_model": "deepseek/deepseek-v4-flash",
                            "raw_provider": "DigitalOcean",
                            "judge_system_prompt_present": False,
                            "judge_usage": {"cost": 0.001},
                        }
                        primary.append(base)
                        for replicate_index in (1, 2, 3):
                            additional.append({**base, "replicate_index": replicate_index})
            (run_dir / "shortcut_judgments.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in primary), encoding="utf-8"
            )
            (run_dir / "shortcut_replicate_judgments.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in additional), encoding="utf-8"
            )
            audit = analyze_replicates(
                {"shortcut_audit": {"pairs_per_transform": 25, "judgments_per_state": 4}},
                run_dir,
            )
            self.assertEqual(audit["n_judgments"], 1000)
            self.assertAlmostEqual(audit["maximum_absolute_mean_delta_p_own"], 0.05)
            self.assertTrue(audit["primary_single_judgment_audit_preserved"])

    def test_short_training_config_is_gated_and_frozen(self):
        path = Path(__file__).parents[1] / "configs" / "gsm8k_grpo_short_v1.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        validate(config)
        self.assertEqual(config["dataset"]["source_file"], "short_train.jsonl")
        self.assertEqual(config["dataset"]["prompt_count"], 100)
        self.assertEqual(config["training"]["num_generations"], 4)
        self.assertEqual(config["training"]["max_steps"], 100)
        self.assertEqual(config["evaluation"]["checkpoints"], [0, 50, 100])

    def test_official_config_is_prepared_but_not_authorized(self):
        path = Path(__file__).parents[1] / "configs" / "gsm8k_grpo_official_prepared_v1.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(config["status"], "PREPARED_NOT_AUTHORIZED")
        self.assertEqual(config["dataset"]["prompt_count"], 1000)
        self.assertEqual(config["training"]["num_epochs"], 2)
        self.assertEqual(config["trial"]["expected_reward_calls"], 8000)
        self.assertTrue(config["evaluation"]["test_after_selection_only"])
        with self.assertRaises(RuntimeError):
            validate(config)
        with patch.dict("os.environ", {"IAR_GSM8K_OFFICIAL_AUTHORIZED": "YES"}):
            validate(config)

    def test_official_checkpoint_selection_uses_lcb_and_quality_constraint(self):
        baseline = [
            {"id": str(index), "p_own": 0.2, "score": {"correct": True}, "judge_token": "0", "delta": -1.0}
            for index in range(20)
        ]
        improved = [
            {"id": str(index), "p_own": 0.4, "score": {"correct": True}, "judge_token": "1", "delta": 1.0}
            for index in range(20)
        ]
        degraded = [
            {"id": str(index), "p_own": 0.8, "score": {"correct": index < 10}, "judge_token": "1", "delta": 1.0}
            for index in range(20)
        ]
        selection = select_checkpoint(
            baseline,
            {1000: improved, 2000: degraded},
            {"evaluation": {"maximum_accuracy_drop": 0.02, "paired_lcb_z": 1.6448536269514722}},
        )
        self.assertEqual(selection["selected_step"], 1000)
        self.assertTrue(selection["comparisons"][0]["quality_feasible"])
        self.assertFalse(selection["comparisons"][1]["quality_feasible"])

    def test_completion_length_is_selected_from_audit_and_truncation_is_guarded(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            rows = [
                {"completion_tokens": 100 + (index % 20), "hit_max_new_tokens": False}
                for index in range(400)
            ]
            (run_dir / "actor_probe_k4_responses.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            result = choose_completion_length(run_dir, 512)
            self.assertEqual(result["selected_max_completion_length"], 160)
            rows[0]["hit_max_new_tokens"] = True
            rows[1]["hit_max_new_tokens"] = True
            rows[2]["hit_max_new_tokens"] = True
            rows[3]["hit_max_new_tokens"] = True
            rows[4]["hit_max_new_tokens"] = True
            (run_dir / "actor_probe_k4_responses.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                choose_completion_length(run_dir, 512)

    def test_go_no_go_requires_completed_paired_shortcut_audit(self):
        group = {
            "accuracy": {"accuracy": 0.8},
            "mean_p_own": 0.7,
            "S": 0.7,
        }
        baseline = {
            "groups": {
                "B_own": group,
                "A0_plain": {
                    "accuracy": {"accuracy": 0.6},
                    "mean_p_own": 0.2,
                    "S": 0.2,
                    "paired_gap": {"lcb_95_one_sided": 0.3},
                    "attribution_auc": 0.8,
                },
            },
            "protocol_audit": {
                "all_logprobs_finite": True,
                "emitted_argmax_consistency": 0.7,
                "single_token_output_rate": 1.0,
                "binary_label_mass": {"min": 0.1},
            },
        }
        probe = {
            "feature_audit": {"chars": {"within_task_correlation_with_reward": 0.1}},
            "group_signal": {"groups_with_at_least_two_distinct_rewards": 100},
            "correctness_relation": {
                "correct_only_reward_std": 0.2,
                "reward_auc_for_correctness": 0.7,
                "top_reward_accuracy_all_groups": 0.7,
                "random_candidate_expected_accuracy": 0.6,
            },
            "protocol_audit": baseline["protocol_audit"],
        }
        gates = {
            "minimum_b_minus_a_accuracy": 0.05,
            "minimum_attribution_auc": 0.7,
            "minimum_groups_with_reward_variation": 80,
            "minimum_correct_only_reward_std": 0.01,
            "maximum_correctness_auc": 0.99,
            "maximum_absolute_within_task_surface_correlation": 0.4,
            "maximum_absolute_paired_shortcut_effect": 0.1,
            "minimum_single_binary_token_rate": 0.99,
            "minimum_reward_repeatability_icc": 0.8,
            "maximum_sample_probability_calibration_gap": 0.1,
            "expected_model": "deepseek/deepseek-v4-flash",
            "expected_provider": "DigitalOcean",
        }
        repeatability = {
            "reward_reliability": {"icc_1_1": 0.9},
            "decision_behavior": {"mean_emitted_own": 0.25, "mean_p_own": 0.24},
        }
        for audit in (baseline["protocol_audit"], probe["protocol_audit"]):
            audit.update(
                {
                    "models": ["deepseek/deepseek-v4-flash"],
                    "providers": ["DigitalOcean"],
                    "message_roles": [("user",)],
                    "system_prompt_present": [False],
                }
            )
        pending = go_no_go(baseline, probe, None, gates, repeatability)
        self.assertEqual(pending["decision"], "PENDING-SHORTCUT-AUDIT")
        go = go_no_go(
            baseline,
            probe,
            {"maximum_absolute_mean_delta_p_own": 0.05},
            gates,
            repeatability,
        )
        self.assertEqual(go["decision"], "GO")
        self.assertEqual(go["non_gating_diagnostics"]["probe_emitted_argmax_consistency"], 0.7)

    def test_complete_synthetic_phase0_metrics(self):
        def protocol_fields(label, p_own):
            return {
                "p_own": p_own,
                "reward": -0.2 if label == "1" else -2.0,
                "logp_0": -2.0 if label == "1" else -0.2,
                "logp_1": -0.2 if label == "1" else -2.0,
                "binary_label_mass": 0.999,
                "judge_token": label,
                "judge_argmax_token": label,
                "emitted_argmax_consistent": True,
                "raw_model": "deepseek/deepseek-v4-flash",
                "raw_provider": "DigitalOcean",
                "judge_prompt_version": ATTRIBUTION_PROMPT_VERSION,
                "judge_prompt_template_sha256": "hash",
                "judge_message_roles": ["user"],
                "judge_system_prompt_present": False,
                "judge_usage": {"cost": 0.00001},
            }

        baseline_rows = []
        for index in range(100):
            common = {
                "id": f"id-{index}",
                "rollout_index": 0,
                "score": {"valid": True, "correct": True, "prediction": "1"},
                "completion_tokens": 20,
                "terminated_with_eos": True,
            }
            baseline_rows.append({**common, "source": "B_own", "variant": "own", "usage": {"cost": 0.001}, **protocol_fields("1", 0.8)})
            baseline_rows.append({**common, "source": "A0", "variant": "plain", **protocol_fields("0", 0.2)})
            baseline_rows.append({**common, "source": "A0", "variant": "imitation", **protocol_fields("0", 0.3)})
        baseline = baseline_summary(baseline_rows, 1.6448536269514722)
        self.assertEqual(baseline["groups"]["B_own"]["accuracy"]["accuracy"], 1.0)
        self.assertGreater(baseline["groups"]["A0_plain"]["paired_gap"]["lcb_95_one_sided"], 0)

        probe_rows = []
        for index in range(100):
            for rollout in range(4):
                correct = rollout >= 2
                reward = -2.0 + rollout * 0.5
                label = "1" if reward > -1 else "0"
                features = {feature: float((index + rollout) % 3) for feature in SURFACE_FEATURES}
                probe_rows.append(
                    {
                        "id": f"id-{index}",
                        "source": "A0",
                        "variant": "plain",
                        "rollout_index": rollout,
                        "response": f"work {rollout}\nAnswer: {int(correct)}",
                        "score": {"valid": True, "correct": correct, "prediction": str(int(correct))},
                        "features": features,
                        **protocol_fields(label, 0.1 + 0.2 * rollout),
                        "reward": reward,
                        "logp_1": reward,
                    }
                )
        probe = probe_summary(probe_rows)
        self.assertEqual(probe["group_signal"]["complete_k4_groups"], 100)
        self.assertEqual(probe["pass_at_1"], 0)
        self.assertEqual(probe["pass_at_4"], 1)
        self.assertEqual(probe["correctness_relation"]["top_reward_accuracy_all_groups"], 1)


if __name__ == "__main__":
    unittest.main()
