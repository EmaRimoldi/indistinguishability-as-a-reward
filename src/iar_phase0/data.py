from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from .config import ProjectConfig


@dataclass(frozen=True)
class MbppExample:
    split: str
    index: int
    task_id: int | str
    prompt: str
    code: str | None
    test_setup_code: str
    tests: list[str]


def load_mbpp(config: ProjectConfig) -> DatasetDict:
    dataset_cfg = config.raw["dataset"]
    cache_dir = config.resolve(config.raw["paths"]["data_cache"])
    hf_cache = config.resolve(config.raw["paths"]["hf_cache"])
    hf_cache.mkdir(parents=True, exist_ok=True)
    name = dataset_cfg["name"]
    config_name = dataset_cfg.get("config_name")
    kwargs: dict[str, Any] = {
        "cache_dir": str(cache_dir),
    }
    if config_name:
        dataset = load_dataset(name, config_name, **kwargs)
    else:
        dataset = load_dataset(name, **kwargs)
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict, got {type(dataset)!r}")
    return dataset


def materialize_split(
    dataset: DatasetDict,
    split: str,
    limit: int | None = None,
) -> list[MbppExample]:
    if split not in dataset:
        raise KeyError(f"Dataset split not found: {split}. Available: {list(dataset.keys())}")
    ds: Dataset = dataset[split]
    count = len(ds) if limit is None else min(limit, len(ds))
    examples: list[MbppExample] = []
    for i in range(count):
        row = ds[i]
        tests = _extract_tests(row)
        examples.append(
            MbppExample(
                split=split,
                index=i,
                task_id=row.get("task_id", f"{split}-{i}"),
                prompt=row.get("text") or row.get("prompt") or row.get("question") or "",
                code=row.get("code"),
                test_setup_code=str(row.get("test_setup_code") or ""),
                tests=tests,
            )
        )
    return examples


def fewshot_examples(dataset: DatasetDict, split: str, count: int) -> list[MbppExample]:
    return materialize_split(dataset, split=split, limit=count)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _extract_tests(row: dict[str, Any]) -> list[str]:
    for key in ["test_list", "tests"]:
        value = row.get(key)
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, str):
            return [value]
    return []
