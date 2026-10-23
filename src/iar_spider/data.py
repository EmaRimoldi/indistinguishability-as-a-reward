from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DATASET_ID = "xlangai/spider"
DATASET_CONFIG = "spider"
DATASET_REVISION = "0c350918f3f29ec754f1181c65cdce76cd6c133c"
OFFICIAL_REPOSITORY = "https://github.com/taoyds/spider.git"
OFFICIAL_REPOSITORY_REVISION = "b7b5b8c890cd30e35427348bb9eb8c6d1350ca7c"
OFFICIAL_TRAIN_SPLIT = "train"
OFFICIAL_HELDOUT_SPLIT = "validation"
EXPECTED_TRAIN = 7_000
EXPECTED_HELDOUT = 1_034
DEFAULT_VALIDATION_COUNT = 700
DEFAULT_SEED = 23
TRAIN_PARQUET_FILE = "spider/train-00000-of-00001.parquet"
TRAIN_PARQUET_SHA256 = "cb4b681558f6f8f428e516fb94c5a1cb19c5a0a0c153c0618c8cc4a28115d4cb"


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
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in values
        ),
        encoding="utf-8",
    )


def load_source_split(
    split: str,
    *,
    cache_dir: Path,
    revision: str = DATASET_REVISION,
) -> list[dict[str, Any]]:
    """Load the pinned train parquet without invoking a multi-split builder."""
    if split != OFFICIAL_TRAIN_SPLIT:
        raise ValueError(
            "The validation-only loader is restricted to Spider official train"
        )
    parquet_path = download_pinned_train_parquet(
        cache_dir=cache_dir,
        revision=revision,
    )
    dataset = pq.read_table(parquet_path).to_pylist()
    if len(dataset) != EXPECTED_TRAIN:
        raise ValueError(
            f"Unexpected Spider {split} size: {len(dataset)} != {EXPECTED_TRAIN}"
        )

    rows: list[dict[str, Any]] = []
    for source_index, source in enumerate(dataset):
        db_id = str(source["db_id"]).strip()
        question = str(source["question"]).strip()
        reference_query = str(source["query"]).strip()
        if not db_id or not question or not reference_query:
            raise ValueError(f"Empty required field in Spider {split} row {source_index}")
        rows.append(
            {
                "id": f"spider-1.0-{split}-{source_index:05d}",
                "source_split": split,
                "source_index": source_index,
                "db_id": db_id,
                "question": question,
                # Stored for external evaluation only.  Prompt builders must
                # never expose this field to the actor or discriminator.
                "reference_query": reference_query,
                "source_row_sha256": sha256_json(
                    {
                        "db_id": db_id,
                        "question": question,
                        "query": reference_query,
                    }
                ),
            }
        )
    return rows


def download_pinned_train_parquet(
    *,
    cache_dir: Path,
    revision: str = DATASET_REVISION,
    expected_sha256: str = TRAIN_PARQUET_SHA256,
    downloader: Callable[..., str] = hf_hub_download,
) -> Path:
    """Download exactly the train shard and verify its immutable content hash."""
    path = Path(
        downloader(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=TRAIN_PARQUET_FILE,
            revision=revision,
            cache_dir=str(cache_dir),
        )
    ).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pinned Spider train parquet is missing: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(
            "Pinned Spider train parquet hash mismatch: "
            f"expected {expected_sha256}, observed {observed}"
        )
    return path


def _stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}|{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _validation_quotas(
    by_database: dict[str, list[dict[str, Any]]], validation_count: int, seed: int
) -> dict[str, int]:
    total = sum(len(rows) for rows in by_database.values())
    if not 0 < validation_count < total:
        raise ValueError("validation_count must lie strictly between zero and the train size")

    exact = {
        db_id: len(rows) * validation_count / total
        for db_id, rows in by_database.items()
    }
    quotas = {db_id: math.floor(value) for db_id, value in exact.items()}
    remaining = validation_count - sum(quotas.values())
    order = sorted(
        by_database,
        key=lambda db_id: (
            -(exact[db_id] - quotas[db_id]),
            _stable_seed(seed + 7_919, db_id),
            db_id,
        ),
    )
    for db_id in order[:remaining]:
        quotas[db_id] += 1
    if sum(quotas.values()) != validation_count:
        raise AssertionError("Largest-remainder allocation did not reach the requested size")
    return quotas


def stratified_train_validation_partition(
    rows: list[dict[str, Any]],
    *,
    validation_count: int = DEFAULT_VALIDATION_COUNT,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[dict[str, Any]]]:
    """Split official train while preserving its database distribution.

    Rows are sampled independently within each ``db_id`` according to a
    largest-remainder proportional allocation.  The official development set
    is not read and therefore cannot influence membership or ordering.
    """
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Spider train rows contain duplicate IDs")
    by_database: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        db_id = str(row.get("db_id", "")).strip()
        if not db_id:
            raise ValueError(f"Spider row {row.get('id')!r} has no db_id")
        by_database[db_id].append(dict(row))

    quotas = _validation_quotas(by_database, validation_count, seed)
    validation: list[dict[str, Any]] = []
    optimisation: list[dict[str, Any]] = []
    for db_id in sorted(by_database):
        candidates = by_database[db_id]
        random.Random(_stable_seed(seed, db_id)).shuffle(candidates)
        cut = quotas[db_id]
        validation.extend(candidates[:cut])
        optimisation.extend(candidates[cut:])

    random.Random(seed + 65_537).shuffle(validation)
    random.Random(seed + 131_071).shuffle(optimisation)
    for split_name, split_rows in (
        ("validation", validation),
        ("optimisation", optimisation),
    ):
        for order, row in enumerate(split_rows):
            row["experiment_split"] = split_name
            row["experiment_order"] = order

    partitioned_ids = [row["id"] for row in validation + optimisation]
    if len(partitioned_ids) != len(ids) or set(partitioned_ids) != set(ids):
        raise AssertionError("Spider internal partition is not exhaustive and disjoint")
    return {"optimisation": optimisation, "validation": validation}


def audit_schema_assets(
    db_ids: Iterable[str], *, tables_json: Path, database_root: Path
) -> dict[str, Any]:
    """Verify and fingerprint the official schemas and SQLite databases."""
    expected = sorted({str(db_id) for db_id in db_ids})
    if not tables_json.is_file():
        raise FileNotFoundError(f"Missing Spider tables.json: {tables_json}")
    if not database_root.is_dir():
        raise FileNotFoundError(f"Missing Spider database directory: {database_root}")
    tables = json.loads(tables_json.read_text(encoding="utf-8"))
    if not isinstance(tables, list):
        raise TypeError("Spider tables.json must contain a JSON list")
    schema_by_id = {
        str(row.get("db_id")): row
        for row in tables
        if isinstance(row, dict) and row.get("db_id") is not None
    }
    schema_ids = set(schema_by_id)
    missing_schemas = sorted(set(expected) - schema_ids)
    required_schema_lists = {
        "table_names_original",
        "table_names",
        "column_names_original",
        "column_names",
        "column_types",
        "primary_keys",
        "foreign_keys",
    }
    malformed_schemas = sorted(
        db_id
        for db_id in expected
        if db_id in schema_by_id
        and any(
            not isinstance(schema_by_id[db_id].get(field), list)
            for field in required_schema_lists
        )
    )
    database_files = {
        db_id: database_root / db_id / f"{db_id}.sqlite" for db_id in expected
    }
    missing_databases = [db_id for db_id, path in database_files.items() if not path.is_file()]
    malformed_databases = []
    for db_id, path in database_files.items():
        if path.is_file():
            with path.open("rb") as handle:
                if handle.read(16) != b"SQLite format 3\x00":
                    malformed_databases.append(db_id)
    if missing_schemas or malformed_schemas or missing_databases or malformed_databases:
        raise FileNotFoundError(
            "Incomplete Spider assets: "
            f"missing schema entries={missing_schemas[:10]}, "
            f"malformed schema entries={malformed_schemas[:10]}, "
            f"missing SQLite databases={missing_databases[:10]}, "
            f"malformed SQLite databases={malformed_databases[:10]}"
        )
    return {
        "official_repository": OFFICIAL_REPOSITORY,
        "official_repository_revision": OFFICIAL_REPOSITORY_REVISION,
        "tables_json": str(tables_json.resolve()),
        "tables_json_sha256": sha256_file(tables_json),
        "database_root": str(database_root.resolve()),
        "database_count": len(database_files),
        "database_sha256": {
            db_id: sha256_file(path) for db_id, path in sorted(database_files.items())
        },
    }


def prepare_validation_only_splits(
    output_dir: Path,
    *,
    cache_dir: Path,
    tables_json: Path,
    database_root: Path,
    seed: int = DEFAULT_SEED,
    validation_count: int = DEFAULT_VALIDATION_COUNT,
    source_loader: Callable[..., list[dict[str, Any]]] = load_source_split,
) -> dict[str, Any]:
    """Freeze optimisation/validation rows without touching official held-out data."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite frozen Spider split: {output_dir}")
    train = source_loader(
        OFFICIAL_TRAIN_SPLIT,
        cache_dir=cache_dir,
        revision=DATASET_REVISION,
    )
    if len(train) != EXPECTED_TRAIN:
        raise ValueError(f"Unexpected Spider train size: {len(train)} != {EXPECTED_TRAIN}")
    assets = audit_schema_assets(
        (row["db_id"] for row in train),
        tables_json=tables_json,
        database_root=database_root,
    )
    partitions = stratified_train_validation_partition(
        train, validation_count=validation_count, seed=seed
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    summaries: dict[str, Any] = {}
    for name in ("optimisation", "validation"):
        rows = partitions[name]
        path = output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        ordered_ids = [row["id"] for row in rows]
        summaries[name] = {
            "n": len(rows),
            "file": path.name,
            "file_sha256": sha256_file(path),
            "ids_sha256": sha256_json(ordered_ids),
            "rows_sha256": sha256_json(rows),
            "database_counts": dict(sorted(Counter(row["db_id"] for row in rows).items())),
        }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "spider-1.0-validation-only-v1",
        "manifest_hash_scope": "canonical JSON excluding manifest_sha256",
        "seed": seed,
        "partition_method": (
            "database-stratified largest-remainder allocation from pinned official train; "
            "deterministic within-database and final split shuffles"
        ),
        "source": {
            "dataset": DATASET_ID,
            "config": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "access_method": "single pinned Hub parquet shard; no dataset builder",
            "train_parquet_file": TRAIN_PARQUET_FILE,
            "train_parquet_sha256": TRAIN_PARQUET_SHA256,
            "loaded_splits": [OFFICIAL_TRAIN_SPLIT],
            "official_train_count": EXPECTED_TRAIN,
        },
        "splits": summaries,
        "schema_assets": assets,
        "heldout_policy": {
            "dataset_split": OFFICIAL_HELDOUT_SPLIT,
            "expected_count": EXPECTED_HELDOUT,
            "accessed_during_preparation": False,
            "materialized_during_preparation": False,
            "may_be_accessed_only_after_all_selection_gates_are_frozen": True,
        },
        "gold_policy": {
            "reference_query_stored_for_external_evaluation": True,
            "reference_query_exposed_to_actor_or_discriminator": False,
        },
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_json(output_dir / "split_manifest.json", manifest)
    return manifest
