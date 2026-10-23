from __future__ import annotations

import unittest

from iar_grpo.plot_hpo import source_group


class PlotHpoTest(unittest.TestCase):
    def test_source_groups_are_stable(self) -> None:
        self.assertEqual(source_group("initial:turing_center"), "initial")
        self.assertEqual(source_group("tpe:multivariate"), "tpe")
        self.assertEqual(source_group("refinement:trial12_rank64"), "refinement")
        self.assertEqual(source_group(None), "other")


if __name__ == "__main__":
    unittest.main()
