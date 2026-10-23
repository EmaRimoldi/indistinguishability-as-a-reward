from __future__ import annotations

import json
from typing import Any

from datasets import Dataset

from iar_phase0.config import load_config
from iar_phase0.data import load_mbpp, materialize_split
from iar_phase0.prompts import generator_prompt, mbpp_google_problem_statement

from .config import GrpoTrialConfig


def build_trial_dataset(config: GrpoTrialConfig) -> tuple[Dataset, list[dict[str, Any]]]:
    """Build one fixed MBPP training pass using the Phase 0 prompt policy."""
    phase0_config = load_config(config.resolve(config.raw["dataset"]["phase0_config"]))
    dataset = load_mbpp(phase0_config)
    split = str(config.raw["dataset"]["split"])
    examples = _select_training_examples(config, dataset, split)
    if len(examples) != config.prompt_count:
        raise ValueError(f"Expected {config.prompt_count} MBPP examples, found {len(examples)}")

    rows: list[dict[str, Any]] = []
    for example in examples:
        rows.append(
            {
                # Qwen2.5-Instruct must receive the same chat wrapper used in
                # Phase 0 inference. A raw string makes the model continue the
                # benchmark text instead of completing an assistant turn.
                "prompt": [
                    {
                        "role": "user",
                        "content": generator_prompt(example, variant="plain", fewshot=[]),
                    }
                ],
                "task_statement": mbpp_google_problem_statement(example),
                "task_id": str(example.task_id),
                "dataset_index": int(example.index),
                "split": example.split,
            }
        )
    return Dataset.from_list(rows), rows


def _select_training_examples(config: GrpoTrialConfig, dataset, split: str):
    manifest_value = config.raw["dataset"].get("index_manifest")
    if not manifest_value:
        return materialize_split(dataset, split=split, limit=config.prompt_count)

    manifest_path = config.resolve(manifest_value)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("split")) != split:
        raise ValueError(
            f"Training-index manifest split {manifest.get('split')!r} does not match {split!r}"
        )
    indices = [int(index) for index in manifest["dataset_indices"]]
    if len(indices) != config.prompt_count or len(set(indices)) != len(indices):
        raise ValueError("Training-index manifest must contain the configured number of unique rows")

    all_examples = materialize_split(dataset, split=split)
    if min(indices) < 0 or max(indices) >= len(all_examples):
        raise IndexError(f"Training-index manifest contains an out-of-range index for {split}")
    examples = [all_examples[index] for index in indices]
    expected_task_ids = manifest.get("task_ids")
    if expected_task_ids is not None:
        actual = [str(example.task_id) for example in examples]
        expected = [str(task_id) for task_id in expected_task_ids]
        if actual != expected:
            raise ValueError("Training-index manifest task IDs do not match the cached dataset")
    return examples
