from __future__ import annotations

from iar_grpo.validation_reward import binary_auc


def test_binary_auc_handles_ties_and_direction() -> None:
    assert binary_auc([True, True, False, False], [0.9, 0.5, 0.5, 0.1]) == 0.875
    assert binary_auc([True, False], [0.1, 0.9]) == 0.0
    assert binary_auc([True], [0.9]) is None
