from __future__ import annotations

import unittest

from iar_grpo.evaluate_pilot import (
    group_statistics,
    select_validation_checkpoint,
    selection_start_index,
    summarize_rows,
)


class EvaluationSummaryTest(unittest.TestCase):
    def test_group_and_logprob_audits(self) -> None:
        rows = [
            self.row(0, True, 0.8, "1", 1.0, "def f(): return 1"),
            self.row(1, True, 0.6, "1", 0.2, "def f():\n    return 1"),
            self.row(2, False, 0.5, "0", 0.0, "def f(): return 0"),
            self.row(3, False, 0.2, "0", -1.0, "def f():\n    return 0"),
        ]
        summary = summarize_rows(rows, num_generations=4)
        self.assertEqual(summary["passed"], 2)
        self.assertEqual(summary["logprob_consistency"]["strict_matches"], 3)
        self.assertEqual(summary["logprob_consistency"]["ties"], 1)
        self.assertEqual(summary["logprob_consistency"]["contradictions"], 0)
        self.assertEqual(summary["groups"]["mixed_correctness"], 1)
        groups = group_statistics(rows)
        self.assertEqual(groups[0]["label_pattern"], "1100")
        self.assertEqual(groups[0]["unique_candidates"], 4)

    def test_training_groups_are_not_merged_across_epochs(self) -> None:
        rows = []
        for call_id in range(1, 9):
            row = self.row(
                (call_id - 1) % 4,
                call_id % 2 == 0,
                -float(call_id),
                "1" if call_id % 2 == 0 else "0",
                1.0 if call_id % 2 == 0 else -1.0,
                f"def f(): return {call_id}",
            )
            row["call_id"] = call_id
            rows.append(row)
        groups = group_statistics(rows, group_size=4)
        self.assertEqual(len(groups), 2)
        self.assertEqual([group["source_group_index"] for group in groups], [0, 1])
        self.assertTrue(all(group["unique_candidates"] == 4 for group in groups))

    def test_checkpoint_selection_uses_validation_lcb_and_quality_gate(self) -> None:
        baseline = [
            self.validation_row("1", 0.1, False, "0"),
            self.validation_row("2", 0.2, True, "0"),
            self.validation_row("3", 0.3, True, "0"),
            self.validation_row("4", 0.4, False, "0"),
        ]
        candidate = [
            self.validation_row("1", 0.5, False, "1"),
            self.validation_row("2", 0.6, True, "1"),
            self.validation_row("3", 0.7, True, "1"),
            self.validation_row("4", 0.8, False, "1"),
        ]
        selection = select_validation_checkpoint(
            {0: baseline, 100: candidate}, z=1.6448536269514722, max_pass_drop=0.02
        )
        self.assertEqual(selection["selected_step"], 100)
        self.assertGreater(selection["objective_value"], 0)
        self.assertTrue(selection["quality_feasible"])

    def test_split_prefix_supports_current_and_legacy_keys(self) -> None:
        self.assertEqual(selection_start_index({"start_index": 0}), 0)
        self.assertEqual(selection_start_index({"start_index": 10}), 10)
        self.assertEqual(selection_start_index({"exclude_first": 7}), 7)
        self.assertEqual(selection_start_index({}), 0)

    @staticmethod
    def validation_row(task_id: str, p_own: float, passed: bool, token: str):
        return {
            "task_id": task_id,
            "p_own": p_own,
            "unit_tests": {"passed": passed},
            "judge_token": token,
            "delta": 1.0 if token == "1" else -1.0,
        }

    @staticmethod
    def row(index: int, passed: bool, reward: float, token: str, delta: float, code: str):
        return {
            "task_id": "1",
            "rollout_index": index,
            "candidate": code,
            "unit_tests": {"passed": passed},
            "reward": reward,
            "judge_token": token,
            "delta": delta,
            "terminated_with_eos": True,
        }


if __name__ == "__main__":
    unittest.main()
