from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DATASET_ID = "google-research-datasets/mbpp"
DATASET_CONFIG = "full"
DATASET_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
TRAIN_FILE = "full/train-00000-of-00001.parquet"
VALIDATION_FILE = "full/validation-00000-of-00001.parquet"
TRAIN_SHA256 = "09d125ca31edacb7800be8c67c45abff618faf0214ff551291817d06bdb914ae"
VALIDATION_SHA256 = "3f0ec060987432d99fe8fb409d31e6c67445b208a01741c5583517c80a10fe80"
TRAIN_COUNT = 374
VALIDATION_COUNT = 90
PROTOCOL = "mbpp-full-canonical-train-validation-only-v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"Expected JSON objects in {path}")
    return rows


def download_pinned_split(
    split: str,
    *,
    cache_dir: Path,
    downloader: Callable[..., str] = hf_hub_download,
) -> Path:
    specs = {
        "train": (TRAIN_FILE, TRAIN_SHA256),
        "validation": (VALIDATION_FILE, VALIDATION_SHA256),
    }
    if split not in specs:
        raise ValueError(
            "Only MBPP train and validation are authorized before final testing"
        )
    filename, expected_hash = specs[split]
    path = Path(
        downloader(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=filename,
            revision=DATASET_REVISION,
            cache_dir=str(cache_dir),
        )
    ).resolve()
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise RuntimeError(f"Pinned MBPP {split} parquet is missing or hash-mismatched")
    return path


def normalize_split(
    path: Path, *, split: str, expected_count: int
) -> list[dict[str, Any]]:
    source = pq.read_table(path).to_pylist()
    if len(source) != expected_count:
        raise ValueError(
            f"MBPP {split} has {len(source)} rows, expected {expected_count}"
        )
    rows = []
    for source_index, item in enumerate(source):
        task_id = item.get("task_id")
        prompt = item.get("text")
        code = item.get("code")
        tests = item.get("test_list")
        setup = item.get("test_setup_code")
        if (
            isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or not isinstance(prompt, str)
            or not prompt.strip()
            or not isinstance(code, str)
            or not isinstance(tests, list)
            or not tests
            or not all(isinstance(test, str) and test.strip() for test in tests)
            or not isinstance(setup, str)
        ):
            raise ValueError(f"Malformed MBPP {split} row {source_index}")
        fingerprint = {
            "task_id": task_id,
            "text": prompt,
            "code": code,
            "test_list": tests,
            "test_setup_code": setup,
            "challenge_test_list": item.get("challenge_test_list"),
        }
        rows.append(
            {
                "id": f"mbpp-{split}-{task_id}",
                "task_id": task_id,
                "source_split": split,
                "source_index": source_index,
                "prompt": prompt.strip(),
                "reference_code": code,
                "tests": tests,
                "test_setup_code": setup,
                "source_row_sha256": sha256_json(fingerprint),
            }
        )
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate MBPP IDs in {split}")
    return rows


def prepare_registered_splits(
    output_dir: Path,
    *,
    cache_dir: Path,
    downloader: Callable[..., str] = hf_hub_download,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite MBPP protocol: {output_dir}")
    train_path = download_pinned_split(
        "train", cache_dir=cache_dir, downloader=downloader
    )
    validation_path = download_pinned_split(
        "validation", cache_dir=cache_dir, downloader=downloader
    )
    train = normalize_split(train_path, split="train", expected_count=TRAIN_COUNT)
    validation = normalize_split(
        validation_path, split="validation", expected_count=VALIDATION_COUNT
    )
    if {row["task_id"] for row in train} & {row["task_id"] for row in validation}:
        raise RuntimeError("MBPP train and validation task IDs overlap")
    output_dir.mkdir(parents=True, exist_ok=False)
    split_rows = {"optimisation": train, "validation": validation}
    split_manifest: dict[str, Any] = {}
    for role, rows in split_rows.items():
        file = output_dir / f"{role}.jsonl"
        write_jsonl(file, rows)
        split_manifest[role] = {
            "source_split": "train" if role == "optimisation" else "validation",
            "file": file.name,
            "n": len(rows),
            "file_sha256": sha256_file(file),
            "rows_sha256": sha256_json(rows),
            "ids_sha256": sha256_json([row["id"] for row in rows]),
        }
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "manifest_hash_scope": "canonical JSON excluding manifest_sha256",
        "dataset": DATASET_ID,
        "config": DATASET_CONFIG,
        "revision": DATASET_REVISION,
        "source_files": {
            "train": {"path": TRAIN_FILE, "sha256": TRAIN_SHA256},
            "validation": {
                "path": VALIDATION_FILE,
                "sha256": VALIDATION_SHA256,
            },
        },
        "splits": split_manifest,
        "heldout_policy": {
            "official_test_accessed_during_preparation": False,
            "official_test_materialized_during_preparation": False,
            "official_test_count": 500,
            "access_only_after_selection_gates": True,
        },
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_json(output_dir / "split_manifest.json", manifest)
    return manifest
