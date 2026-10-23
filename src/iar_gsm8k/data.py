from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from .scoring import parse_reference_answer

DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
EXPECTED_TRAIN = 7_473
EXPECTED_TEST = 1_319
SCREEN_VALIDATION = 747
SCREEN_OPTIMISATION = EXPECTED_TRAIN - SCREEN_VALIDATION
SCREEN_PROTOCOL = "gsm8k-full-train-90-10-validation-only-v1"
SCREEN_TRAIN_FILE = "main/train-00000-of-00001.parquet"
SCREEN_TRAIN_SHA256 = "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in values
        ),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dataset_revision() -> str:
    return DATASET_REVISION


def load_source_split(
    split: str, *, cache_dir: Path, revision: str | None
) -> list[dict[str, Any]]:
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported GSM8K source split: {split}")
    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=split,
        cache_dir=str(cache_dir),
        revision=revision,
    )
    expected = EXPECTED_TRAIN if split == "train" else EXPECTED_TEST
    if len(dataset) != expected:
        raise ValueError(f"Unexpected GSM8K {split} size: {len(dataset)} != {expected}")
    rows: list[dict[str, Any]] = []
    for source_index, row in enumerate(dataset):
        question = str(row["question"])
        reference_solution = str(row["answer"])
        identifier = f"gsm8k-main-{split}-{source_index:05d}"
        rows.append(
            {
                "id": identifier,
                "source_split": split,
                "source_index": source_index,
                "question": question,
                "answer_value": parse_reference_answer(reference_solution),
                "reference_solution": reference_solution,
                "source_row_sha256": sha256_json(
                    {"question": question, "answer": reference_solution}
                ),
            }
        )
    return rows


def download_pinned_screen_train(
    *,
    cache_dir: Path,
    downloader: Callable[..., str] = hf_hub_download,
) -> Path:
    """Download exactly the train parquet without initializing all splits."""
    path = Path(
        downloader(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=SCREEN_TRAIN_FILE,
            revision=DATASET_REVISION,
            cache_dir=str(cache_dir),
        )
    ).resolve()
    if (
        not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != SCREEN_TRAIN_SHA256
    ):
        raise RuntimeError("Pinned GSM8K train parquet is missing or hash-mismatched")
    return path


def normalize_screen_train(path: Path) -> list[dict[str, Any]]:
    source = pq.read_table(path).to_pylist()
    if len(source) != EXPECTED_TRAIN:
        raise ValueError(f"Unexpected GSM8K train size: {len(source)}")
    rows: list[dict[str, Any]] = []
    for source_index, row in enumerate(source):
        question = str(row["question"])
        reference_solution = str(row["answer"])
        rows.append(
            {
                "id": f"gsm8k-main-train-{source_index:05d}",
                "source_split": "train",
                "source_index": source_index,
                "question": question,
                "answer_value": parse_reference_answer(reference_solution),
                "reference_solution": reference_solution,
                "source_row_sha256": sha256_json(
                    {"question": question, "answer": reference_solution}
                ),
            }
        )
    return rows


def prepare_screen_splits(
    output_dir: Path,
    *,
    cache_dir: Path,
    seed: int = 23,
    validation_size: int = SCREEN_VALIDATION,
    downloader: Callable[..., str] = hf_hub_download,
) -> dict[str, Any]:
    """Freeze a 90/10 partition of official train without touching official test."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite GSM8K screen split: {output_dir}")
    if validation_size != SCREEN_VALIDATION:
        raise ValueError(
            f"GSM8K screen validation must contain {SCREEN_VALIDATION} rows"
        )
    source = normalize_screen_train(
        download_pinned_screen_train(cache_dir=cache_dir, downloader=downloader)
    )
    shuffled = [dict(row) for row in source]
    random.Random(seed).shuffle(shuffled)
    validation = shuffled[:validation_size]
    optimisation = shuffled[validation_size:]
    partitions = {"optimisation": optimisation, "validation": validation}
    output_dir.mkdir(parents=True, exist_ok=False)
    split_specs: dict[str, Any] = {}
    for role, rows in partitions.items():
        for order, row in enumerate(rows):
            row["experiment_split"] = role
            row["experiment_order"] = order
        path = output_dir / f"{role}.jsonl"
        write_jsonl(path, rows)
        split_specs[role] = {
            "source_split": "train",
            "file": path.name,
            "n": len(rows),
            "ids_sha256": sha256_json([row["id"] for row in rows]),
            "rows_sha256": sha256_json(rows),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    if {row["id"] for row in optimisation} & {row["id"] for row in validation}:
        raise RuntimeError("GSM8K optimisation and validation overlap")
    manifest = {
        "schema_version": 1,
        "protocol": SCREEN_PROTOCOL,
        "seed": seed,
        "partition_method": "single deterministic shuffle of pinned official train",
        "source": {
            "dataset": DATASET_ID,
            "config": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "train_count": EXPECTED_TRAIN,
            "train_file": SCREEN_TRAIN_FILE,
            "train_file_sha256": SCREEN_TRAIN_SHA256,
        },
        "splits": split_specs,
        "heldout_policy": {
            "official_test_accessed_during_preparation": False,
            "official_test_materialized_during_preparation": False,
            "official_test_count": EXPECTED_TEST,
            "access_only_after_all_selection_gates": True,
        },
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_json(output_dir / "split_manifest.json", manifest)
    return manifest


def deterministic_partition(
    rows: list[dict[str, Any]], *, seed: int, sizes: dict[str, int]
) -> dict[str, list[dict[str, Any]]]:
    if any(size < 0 for size in sizes.values()):
        raise ValueError("Split sizes must be non-negative")
    if sum(sizes.values()) > len(rows):
        raise ValueError("Requested splits exceed the source dataset")
    shuffled = [dict(row) for row in rows]
    random.Random(seed).shuffle(shuffled)
    result: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for name, size in sizes.items():
        values = shuffled[cursor : cursor + size]
        cursor += size
        for order, row in enumerate(values):
            row["experiment_split"] = name
            row["experiment_order"] = order
        result[name] = values
    reserve = shuffled[cursor:]
    for order, row in enumerate(reserve):
        row["experiment_split"] = "reserve"
        row["experiment_order"] = order
    result["reserve"] = reserve
    ids = [row["id"] for values in result.values() for row in values]
    if len(ids) != len(set(ids)) or len(ids) != len(rows):
        raise ValueError("Internal GSM8K splits are not exhaustive and disjoint")
    return result


def prepare_experiment_splits(
    output_dir: Path,
    *,
    cache_dir: Path,
    seed: int = 23,
    audit_size: int = 100,
    short_train_size: int = 100,
    official_train_size: int = 1_000,
    validation_size: int = 300,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_path = output_dir / "split_manifest.json"
    requested_sizes = {
        "reward_audit": audit_size,
        "short_train": short_train_size,
        "official_train": official_train_size,
        "validation": validation_size,
        "reserve": EXPECTED_TRAIN
        - audit_size
        - short_train_size
        - official_train_size
        - validation_size,
        "test": EXPECTED_TEST,
    }
    if existing_path.exists():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        observed_sizes = {
            name: int(values["n"])
            for name, values in existing.get("splits", {}).items()
        }
        compatible = (
            existing.get("seed") == seed
            and existing.get("source", {}).get("dataset") == DATASET_ID
            and existing.get("source", {}).get("config") == DATASET_CONFIG
            and existing.get("source", {}).get("revision") == DATASET_REVISION
            and observed_sizes == requested_sizes
            and all(
                (output_dir / values["file"]).is_file()
                for values in existing.get("splits", {}).values()
            )
        )
        if not compatible:
            raise FileExistsError(
                f"Existing frozen GSM8K split is incompatible: {existing_path}"
            )
        return existing
    revision = dataset_revision()
    train = load_source_split("train", cache_dir=cache_dir, revision=revision)
    test = load_source_split("test", cache_dir=cache_dir, revision=revision)
    partitions = deterministic_partition(
        train,
        seed=seed,
        sizes={
            "reward_audit": audit_size,
            "short_train": short_train_size,
            "official_train": official_train_size,
            "validation": validation_size,
        },
    )
    partitions["test"] = [
        dict(row, experiment_split="test", experiment_order=index)
        for index, row in enumerate(test)
    ]
    split_summaries: dict[str, Any] = {}
    for name, values in partitions.items():
        file_path = output_dir / f"{name}.jsonl"
        write_jsonl(file_path, values)
        split_summaries[name] = {
            "n": len(values),
            "source_split": sorted({row["source_split"] for row in values}),
            "file": file_path.name,
            "ordered_ids": [row["id"] for row in values],
            "ids_sha256": sha256_json([row["id"] for row in values]),
            "rows_sha256": sha256_json(values),
        }
    internal_ids = {
        row["id"]
        for name, values in partitions.items()
        if name != "test"
        for row in values
    }
    test_ids = {row["id"] for row in partitions["test"]}
    if internal_ids & test_ids:
        raise ValueError("Official test overlaps an internal training-derived split")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "partition_method": "single deterministic shuffle of official train using Python random.Random(seed)",
        "cache_policy": "reuse cached Hugging Face artifacts before downloading the pinned revision",
        "source": {
            "dataset": DATASET_ID,
            "config": DATASET_CONFIG,
            "revision": revision,
            "source_counts": {"train": len(train), "test": len(test)},
        },
        "splits": split_summaries,
        "exclusions": [],
        "policy": {
            "gold_answer_exposed_to_models": False,
            "reference_solution_exposed_to_models": False,
            "official_test_used_before_final_checkpoint_selection": False,
            "official_test_kept_complete": True,
        },
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_json(output_dir / "split_manifest.json", manifest)
    return manifest
