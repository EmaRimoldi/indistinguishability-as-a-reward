from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from iar_grpo.confirm_hpo import (
    aggregate_candidate,
    compare_candidates,
    confirmation_spec_sort_key,
    reward_log_signature,
)


class ConfirmHpoTest(unittest.TestCase):
    def test_confirmation_jobs_are_interleaved_by_seed_then_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = []
            for trial, seed in [
                (15, 42),
                (12, 42),
                (15, 23),
                (12, 23),
                (15, 17),
                (12, 17),
            ]:
                path = Path(temp_dir) / f"{trial}_{seed}.json"
                path.write_text(
                    json.dumps({"trial_number": trial, "seed": seed}),
                    encoding="utf-8",
                )
                paths.append(path)
            ordered = sorted(paths, key=confirmation_spec_sort_key)
            self.assertEqual(
                [json.loads(path.read_text()) for path in ordered],
                [
                    {"trial_number": 12, "seed": 17},
                    {"trial_number": 15, "seed": 17},
                    {"trial_number": 12, "seed": 23},
                    {"trial_number": 15, "seed": 23},
                    {"trial_number": 12, "seed": 42},
                    {"trial_number": 15, "seed": 42},
                ],
            )

    def test_robust_candidate_requires_soft_hard_and_quality_agreement(self) -> None:
        rows = [seed_row(seed, 0.05, 0.03, 0.02) for seed in [17, 23, 42]]
        result = aggregate_candidate(rows)
        self.assertGreater(result["pooled_p_own_delta_lcb"], 0)
        self.assertTrue(result["robust"])
        self.assertAlmostEqual(result["incremental_confirmation_cost"], 0.06)
        self.assertAlmostEqual(result["reused_source_cost"], 0.0)

        rows[2]["argmax_fooling_delta"] = -0.2
        result = aggregate_candidate(rows)
        self.assertFalse(result["robust"])

        rows = [seed_row(seed, 0.05, 0.03, 0.02) for seed in [17, 23, 42]]
        rows[1]["protocol_ok"] = False
        result = aggregate_candidate(rows)
        self.assertFalse(result["protocol_ok"])
        self.assertFalse(result["robust"])

        rows = [seed_row(seed, 0.05, 0.0, 0.02) for seed in [17, 23, 42]]
        result = aggregate_candidate(rows)
        self.assertEqual(result["mean_sampled_fooling_delta"], 0.0)
        self.assertFalse(result["robust"])

        rows = [seed_row(seed, 0.05, 0.03, 0.02) for seed in [17, 23, 42]]
        rows[2]["observed_providers"] = ["AnotherProvider"]
        result = aggregate_candidate(rows)
        self.assertFalse(result["protocol_ok"])
        self.assertFalse(result["robust"])

        rows = [seed_row(seed, 0.05, 0.03, 0.02) for seed in [17, 23, 42]]
        rows[2]["baseline_generation_manifest_sha256"] = "changed-baseline"
        result = aggregate_candidate(rows)
        self.assertFalse(result["protocol_ok"])
        self.assertFalse(result["robust"])

    def test_one_negative_seed_is_not_robust(self) -> None:
        rows = [seed_row(17, 0.05, 0.02, 0.02), seed_row(23, 0.04, 0.02, 0.02)]
        rows.append(seed_row(42, -0.01, 0.02, 0.02))
        result = aggregate_candidate(rows)
        self.assertFalse(result["all_seed_p_own_deltas_positive"])
        self.assertFalse(result["robust"])

    def test_reward_log_signature_covers_prompt_model_and_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "reward_calls.jsonl"
            rows = [
                {
                    "status": "ok",
                    "raw_provider": "DigitalOcean",
                    "raw_model": "deepseek/deepseek-v4-flash",
                    "reward_value": "log_p_own",
                    "judge_prompt_template_version": "v1",
                    "judge_prompt_template_sha256": "prompt-hash",
                    "judge_system_prompt_sha256": "system-hash",
                },
                {
                    "status": "error",
                    "raw_provider": "ignored",
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            signature = reward_log_signature(path)
            self.assertEqual(signature["n_ok"], 1)
            self.assertEqual(signature["providers"], {"DigitalOcean"})
            self.assertEqual(
                signature["raw_models"], {"deepseek/deepseek-v4-flash"}
            )
            self.assertEqual(signature["reward_values"], {"log_p_own"})
            self.assertEqual(signature["prompt_versions"], {"v1"})
            self.assertEqual(signature["prompt_hashes"], {"prompt-hash"})
            self.assertEqual(signature["system_hashes"], {"system-hash"})

    def test_direct_candidate_comparison_is_paired_by_seed_and_task(self) -> None:
        left_rows = [seed_row(seed, 0.05, 0.03, 0.02) for seed in [17, 23, 42]]
        right_rows = [seed_row(seed, 0.02, 0.01, 0.01) for seed in [17, 23, 42]]
        left = {"trial_number": 17, "seeds": left_rows}
        right = {"trial_number": 12, "seeds": right_rows}
        result = compare_candidates(left, right)
        self.assertAlmostEqual(result["p_own_delta_difference_mean"], 0.03)
        self.assertAlmostEqual(result["p_own_delta_difference_standard_error"], 0.0)
        self.assertAlmostEqual(result["p_own_delta_difference_lcb"], 0.03)
        self.assertTrue(result["left_superiority_supported"])
        self.assertEqual(
            [row["seed"] for row in result["per_seed"]], [17, 23, 42]
        )


def seed_row(seed: int, delta: float, sampled: float, argmax: float) -> dict:
    return {
        "seed": seed,
        "p_own_delta": delta,
        "p_own_delta_lcb": delta - 0.01,
        "pass_at_1_delta": 0.0,
        "net_passed_change": 0,
        "quality_feasible": True,
        "sampled_fooling_delta": sampled,
        "argmax_fooling_delta": argmax,
        "total_openrouter_cost": 0.02,
        "reused": False,
        "training_prompt_manifest_sha256": "fixed-prompts",
        "baseline_generation_manifest_sha256": "fixed-baseline",
        "protocol_ok": True,
        "observed_providers": ["DigitalOcean"],
        "task_p_own_deltas": {str(task): delta for task in range(90)},
    }


if __name__ == "__main__":
    unittest.main()
