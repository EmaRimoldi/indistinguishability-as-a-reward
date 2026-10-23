from __future__ import annotations

import unittest

from iar_grpo.plot_confirmation import candidate_color, candidate_labels


class PlotConfirmationTest(unittest.TestCase):
    def test_candidate_labels_and_colors_are_stable(self) -> None:
        rows = [{"trial_number": 12}, {"trial_number": 17}]
        self.assertEqual(candidate_labels(rows), ["trial 12", "trial 17"])
        self.assertNotEqual(candidate_color(rows[0]), candidate_color(rows[1]))


if __name__ == "__main__":
    unittest.main()
