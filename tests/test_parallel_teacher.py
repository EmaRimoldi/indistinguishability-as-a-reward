from pathlib import Path

import pytest

from iar_distillation.parallel_teacher import balanced_assignments, load_named_secret


def test_balanced_assignments_are_complete_disjoint_and_balanced() -> None:
    source_ids = [f"row-{index}" for index in range(11)]
    shards = balanced_assignments(source_ids, 3)

    flattened = [source_id for shard in shards for source_id in shard]
    assert sorted(flattened) == sorted(source_ids)
    assert len(flattened) == len(set(flattened))
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_load_named_secret_selects_only_requested_variable(tmp_path: Path) -> None:
    env_path = tmp_path / "keys.env"
    env_path.write_text(
        "OPENROUTER_API_KEY=first\nOPENROUTER_API_KEY_2='second'\n",
        encoding="utf-8",
    )

    assert load_named_secret(env_path, "OPENROUTER_API_KEY_2") == "second"
    with pytest.raises(KeyError):
        load_named_secret(env_path, "OPENROUTER_API_KEY_3")


def test_balanced_assignments_rejects_zero_shards() -> None:
    with pytest.raises(ValueError):
        balanced_assignments(["row"], 0)
