from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

CQA_DATASET = "tau/commonsense_qa"
CQA_CONFIG = "default"
COSE_DATASET = "Salesforce/cos_e"
COSE_CONFIG = "v1.11"
LABELS = ("A", "B", "C", "D", "E")
SCREEN_VALIDATION_COUNT = 973
SCREEN_OPTIMISATION_COUNT = 8_768
SCREEN_PROTOCOL = "cqa-full-train-90-10-validation-only-v2"
CQA_TRAIN_FILE = "data/train-00000-of-00001.parquet"
CQA_TRAIN_SHA256 = "b0449767ed986bfc2ca52b1244a46ef12f732756727f3cb0a4ab69ac8b3d282b"
COSE_TRAIN_FILE = "v1.11/train-00000-of-00001.parquet"
COSE_TRAIN_SHA256 = "a03cc8f4ae917b0939c5ad35805f407e21cd68e8c046fe90db64001b1d73f16c"


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
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dataset_revisions() -> dict[str, str | None]:
    api = HfApi()
    revisions: dict[str, str | None] = {}
    for dataset_name in (CQA_DATASET, COSE_DATASET):
        try:
            revisions[dataset_name] = api.dataset_info(dataset_name).sha
        except Exception:  # noqa: BLE001 - record an unavailable remote revision
            revisions[dataset_name] = None
    return revisions


def load_joined_split(
    split: str,
    *,
    cache_dir: Path,
    revisions: dict[str, str | None] | None = None,
) -> list[dict[str, Any]]:
    revisions = revisions or {}
    cqa = load_dataset(
        CQA_DATASET,
        CQA_CONFIG,
        split=split,
        cache_dir=str(cache_dir),
        revision=revisions.get(CQA_DATASET),
    )
    cose = load_dataset(
        COSE_DATASET,
        COSE_CONFIG,
        split=split,
        cache_dir=str(cache_dir),
        revision=revisions.get(COSE_DATASET),
    )
    cose_by_id = {str(row["id"]): row for row in cose}
    if len(cose_by_id) != len(cose):
        raise ValueError(f"Duplicate CoS-E IDs in {split}")
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for source_index, row in enumerate(cqa):
        identifier = str(row["id"])
        explanation = cose_by_id.get(identifier)
        if explanation is None:
            missing.append(identifier)
            continue
        labels = list(row["choices"]["label"])
        choices = list(row["choices"]["text"])
        if labels != list(LABELS) or len(choices) != 5:
            raise ValueError(f"Unexpected choice labels for {identifier}: {labels}")
        answer_key = str(row["answerKey"])
        if answer_key not in LABELS:
            raise ValueError(f"Invalid answer key for {identifier}: {answer_key!r}")
        answer_index = LABELS.index(answer_key)
        if choices[answer_index] != str(explanation["answer"]):
            raise ValueError(
                f"CommonsenseQA/CoS-E answer mismatch for {identifier}: "
                f"{choices[answer_index]!r} != {explanation['answer']!r}"
            )
        rows.append(
            {
                "id": identifier,
                "source_split": split,
                "source_index": source_index,
                "question": str(row["question"]),
                "question_concept": str(row["question_concept"]),
                "choice_labels": labels,
                "choices": choices,
                "answer_key": answer_key,
                "answer_text": choices[answer_index],
                "reference_abstractive_explanation": str(
                    explanation["abstractive_explanation"]
                ),
                "reference_extractive_explanation": str(
                    explanation["extractive_explanation"]
                ),
            }
        )
    if missing:
        raise ValueError(f"Missing {len(missing)} CoS-E rows in {split}: {missing[:5]}")
    if len(rows) != len(cqa) or len(rows) != len(cose):
        raise ValueError(
            f"Join is not one-to-one for {split}: CQA={len(cqa)}, CoS-E={len(cose)}, joined={len(rows)}"
        )
    return rows


def stratified_partition(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    sizes: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    if any(size % len(LABELS) for size in sizes.values()):
        raise ValueError("Every internal split size must be divisible by five")
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[row["answer_key"]].append(row)
    selected: dict[str, list[dict[str, Any]]] = {name: [] for name in sizes}
    for label_index, label in enumerate(LABELS):
        candidates = list(by_label[label])
        random.Random(seed + 1009 * (label_index + 1)).shuffle(candidates)
        cursor = 0
        for name, size in sizes.items():
            count = size // len(LABELS)
            chosen = candidates[cursor : cursor + count]
            if len(chosen) != count:
                raise ValueError(f"Not enough {label} examples for {name}")
            selected[name].extend(chosen)
            cursor += count
    for split_index, (name, split_rows) in enumerate(selected.items()):
        random.Random(seed + 7919 * (split_index + 1)).shuffle(split_rows)
        for order, row in enumerate(split_rows):
            row["experiment_split"] = name
            row["experiment_order"] = order
    return selected


def stratified_holdout(
    rows: list[dict[str, Any]], *, seed: int, holdout_size: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition every row into a proportional holdout and optimization pool."""
    if not 0 < holdout_size < len(rows):
        raise ValueError(
            "holdout_size must leave non-empty holdout and optimization pools"
        )
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = str(row["answer_key"])
        if label not in LABELS:
            raise ValueError(f"Invalid answer label: {label!r}")
        by_label[label].append(row)
    if set(by_label) != set(LABELS):
        raise ValueError("All answer labels must be represented")

    exact = {label: holdout_size * len(by_label[label]) / len(rows) for label in LABELS}
    quotas = {label: int(exact[label]) for label in LABELS}
    remaining = holdout_size - sum(quotas.values())
    for label in sorted(LABELS, key=lambda item: (-(exact[item] - quotas[item]), item))[
        :remaining
    ]:
        quotas[label] += 1

    optimization: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for label_index, label in enumerate(LABELS):
        candidates = list(by_label[label])
        random.Random(seed + 1009 * (label_index + 1)).shuffle(candidates)
        holdout.extend(dict(row) for row in candidates[: quotas[label]])
        optimization.extend(dict(row) for row in candidates[quotas[label] :])
    random.Random(seed + 65_537).shuffle(optimization)
    random.Random(seed + 1_000_003).shuffle(holdout)
    for name, selected_rows in (("train", optimization), ("validation", holdout)):
        for order, row in enumerate(selected_rows):
            row["experiment_split"] = name
            row["experiment_order"] = order
    return optimization, holdout


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_pinned_train(
    *,
    repo_id: str,
    revision: str,
    filename: str,
    expected_sha256: str,
    cache_dir: Path,
    downloader: Any = hf_hub_download,
) -> Path:
    path = Path(
        downloader(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            revision=revision,
            cache_dir=str(cache_dir),
        )
    ).resolve()
    if not path.is_file() or _sha256_file(path) != expected_sha256:
        raise RuntimeError(
            f"Pinned train parquet is missing or hash-mismatched: {repo_id}"
        )
    return path


def load_pinned_joined_train(
    *, cache_dir: Path, downloader: Any = hf_hub_download
) -> list[dict[str, Any]]:
    """Read only the two pinned train parquets needed by the CQA screen."""
    cqa_path = _download_pinned_train(
        repo_id=CQA_DATASET,
        revision="94630fe30dad47192a8546eb75f094926d47e155",
        filename=CQA_TRAIN_FILE,
        expected_sha256=CQA_TRAIN_SHA256,
        cache_dir=cache_dir,
        downloader=downloader,
    )
    cose_path = _download_pinned_train(
        repo_id=COSE_DATASET,
        revision="f01396594b768e0026841176bf770afe58b82fba",
        filename=COSE_TRAIN_FILE,
        expected_sha256=COSE_TRAIN_SHA256,
        cache_dir=cache_dir,
        downloader=downloader,
    )
    cqa = pq.read_table(cqa_path).to_pylist()
    cose = pq.read_table(cose_path).to_pylist()
    cose_by_id = {str(row["id"]): row for row in cose}
    if len(cqa) != 9_741 or len(cose) != 9_741 or len(cose_by_id) != len(cose):
        raise ValueError("Pinned CQA/CoS-E train counts are not 9741 one-to-one rows")
    rows: list[dict[str, Any]] = []
    for source_index, row in enumerate(cqa):
        identifier = str(row["id"])
        explanation = cose_by_id.get(identifier)
        if explanation is None:
            raise ValueError(f"Missing CoS-E explanation for {identifier}")
        labels = [str(value) for value in row["choices"]["label"]]
        choices = [str(value) for value in row["choices"]["text"]]
        answer_key = str(row["answerKey"])
        if labels != list(LABELS) or answer_key not in LABELS:
            raise ValueError(f"Malformed CQA labels for {identifier}")
        answer_index = LABELS.index(answer_key)
        if choices[answer_index] != str(explanation["answer"]):
            raise ValueError(f"CQA/CoS-E answer mismatch for {identifier}")
        rows.append(
            {
                "id": identifier,
                "source_split": "train",
                "source_index": source_index,
                "question": str(row["question"]),
                "question_concept": str(row["question_concept"]),
                "choice_labels": labels,
                "choices": choices,
                "answer_key": answer_key,
                "answer_text": choices[answer_index],
                "reference_abstractive_explanation": str(
                    explanation["abstractive_explanation"]
                ),
                "reference_extractive_explanation": str(
                    explanation["extractive_explanation"]
                ),
            }
        )
    return rows


def prepare_screen_splits(
    output_dir: Path,
    *,
    cache_dir: Path,
    seed: int = 23,
    downloader: Any = hf_hub_download,
) -> dict[str, Any]:
    """Freeze the CQA screen from official train without reading held-out data."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite CQA screen split: {output_dir}")
    source = load_pinned_joined_train(cache_dir=cache_dir, downloader=downloader)
    optimisation, validation = stratified_holdout(
        source, seed=seed, holdout_size=SCREEN_VALIDATION_COUNT
    )
    if len(optimisation) != SCREEN_OPTIMISATION_COUNT:
        raise RuntimeError("Unexpected CQA optimisation count")
    output_dir.mkdir(parents=True, exist_ok=False)
    specs: dict[str, Any] = {}
    for role, rows in (("optimisation", optimisation), ("validation", validation)):
        path = output_dir / f"{role}.jsonl"
        write_jsonl(path, rows)
        specs[role] = {
            "source_split": "train",
            "file": path.name,
            "n": len(rows),
            "file_sha256": _sha256_file(path),
            "ids_sha256": sha256_json([row["id"] for row in rows]),
            "rows_sha256": sha256_json(rows),
        }
    manifest = {
        "schema_version": 1,
        "protocol": SCREEN_PROTOCOL,
        "seed": seed,
        "stratification": "proportional by answerKey A-E",
        "sources": {
            "commonsenseqa": {
                "dataset": CQA_DATASET,
                "revision": "94630fe30dad47192a8546eb75f094926d47e155",
                "train_file": CQA_TRAIN_FILE,
                "train_file_sha256": CQA_TRAIN_SHA256,
            },
            "cos_e": {
                "dataset": COSE_DATASET,
                "revision": "f01396594b768e0026841176bf770afe58b82fba",
                "train_file": COSE_TRAIN_FILE,
                "train_file_sha256": COSE_TRAIN_SHA256,
            },
        },
        "splits": specs,
        "heldout_policy": {
            "official_validation_accessed_during_preparation": False,
            "official_validation_materialized_during_preparation": False,
            "official_validation_count": 1_221,
            "official_test_used": False,
            "access_only_after_all_selection_gates": True,
        },
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_json(output_dir / "split_manifest.json", manifest)
    return manifest


def prepare_final_splits(
    output_dir: Path,
    *,
    cache_dir: Path,
    seed: int = 23,
    validation_size: int = 974,
) -> dict[str, Any]:
    """Create the final CQA 90/10 train-validation partition and held-out test."""
    output_dir.mkdir(parents=True, exist_ok=False)
    revisions = dataset_revisions()
    train_source = load_joined_split("train", cache_dir=cache_dir, revisions=revisions)
    official_validation = load_joined_split(
        "validation", cache_dir=cache_dir, revisions=revisions
    )
    optimization, validation = stratified_holdout(
        train_source, seed=seed, holdout_size=validation_size
    )
    test = [
        dict(row, experiment_split="test", experiment_order=index)
        for index, row in enumerate(official_validation)
    ]
    partitions = {"train": optimization, "validation": validation, "test": test}

    split_summaries: dict[str, Any] = {}
    seen: set[str] = set()
    for name, split_rows in partitions.items():
        ids = [str(row["id"]) for row in split_rows]
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError("Final CQA splits overlap or contain duplicate IDs")
        seen.update(ids)
        path = output_dir / f"{name}.jsonl"
        write_jsonl(path, split_rows)
        split_summaries[name] = {
            "n": len(split_rows),
            "source_split": sorted({str(row["source_split"]) for row in split_rows}),
            "answer_key_counts": dict(
                sorted(Counter(str(row["answer_key"]) for row in split_rows).items())
            ),
            "ids_sha256": sha256_json(ids),
            "rows_sha256": sha256_json(split_rows),
            "file": path.name,
        }

    manifest = {
        "schema_version": 2,
        "seed": seed,
        "stratification": "proportional by official answerKey A-E",
        "sources": {
            "commonsenseqa": {
                "dataset": CQA_DATASET,
                "config": CQA_CONFIG,
                "revision": revisions.get(CQA_DATASET),
            },
            "cos_e": {
                "dataset": COSE_DATASET,
                "config": COSE_CONFIG,
                "revision": revisions.get(COSE_DATASET),
            },
        },
        "source_counts": {
            "train": len(train_source),
            "validation": len(official_validation),
        },
        "splits": split_summaries,
        "policy": {
            "gold_answer_exposed_to_models": False,
            "reference_explanations_exposed_to_models": False,
            "official_validation_used_as_test": True,
            "official_test_used": False,
        },
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_json(output_dir / "split_manifest.json", manifest)
    return manifest


def prepare_experiment_splits(
    output_dir: Path,
    *,
    cache_dir: Path,
    seed: int = 23,
    audit_size: int = 100,
    train_size: int = 1000,
    validation_size: int = 300,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    revisions = dataset_revisions()
    train_source = load_joined_split("train", cache_dir=cache_dir, revisions=revisions)
    official_validation = load_joined_split(
        "validation", cache_dir=cache_dir, revisions=revisions
    )
    partitions = stratified_partition(
        train_source,
        seed=seed,
        sizes={
            "reward_audit": audit_size,
            "train": train_size,
            "validation": validation_size,
        },
    )
    test_rows = [
        dict(row, experiment_split="test", experiment_order=index)
        for index, row in enumerate(official_validation)
    ]
    partitions["test"] = test_rows

    all_selected_ids: list[str] = []
    split_summaries: dict[str, Any] = {}
    for name, split_rows in partitions.items():
        ids = [row["id"] for row in split_rows]
        all_selected_ids.extend(ids)
        path = output_dir / f"{name}.jsonl"
        write_jsonl(path, split_rows)
        split_summaries[name] = {
            "n": len(split_rows),
            "source_split": sorted({row["source_split"] for row in split_rows}),
            "answer_key_counts": dict(
                sorted(Counter(row["answer_key"] for row in split_rows).items())
            ),
            "ids_sha256": sha256_json(ids),
            "rows_sha256": sha256_json(split_rows),
            "file": path.name,
        }
    if len(all_selected_ids) != len(set(all_selected_ids)):
        raise ValueError("Experiment splits overlap by dataset ID")

    used_train_ids = {
        row["id"]
        for name in ("reward_audit", "train", "validation")
        for row in partitions[name]
    }
    unused_train = [row for row in train_source if row["id"] not in used_train_ids]
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "stratification": "exactly balanced by official answerKey A-E for reward_audit/train/validation",
        "sources": {
            "commonsenseqa": {
                "dataset": CQA_DATASET,
                "config": CQA_CONFIG,
                "revisions": revisions.get(CQA_DATASET),
            },
            "cos_e": {
                "dataset": COSE_DATASET,
                "config": COSE_CONFIG,
                "revisions": revisions.get(COSE_DATASET),
            },
        },
        "source_counts": {
            "train": len(train_source),
            "validation": len(official_validation),
        },
        "splits": split_summaries,
        "unused_official_train": {
            "n": len(unused_train),
            "ids_sha256": sha256_json([row["id"] for row in unused_train]),
        },
        "policy": {
            "gold_answer_exposed_to_models": False,
            "reference_explanations_exposed_to_models": False,
            "official_validation_used_as_test": True,
            "official_test_used": False,
        },
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_json(output_dir / "split_manifest.json", manifest)
    return manifest
