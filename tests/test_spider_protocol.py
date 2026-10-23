from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from iar_spider.data import (
    DATASET_REVISION,
    TRAIN_PARQUET_FILE,
    audit_schema_assets,
    download_pinned_train_parquet,
    prepare_validation_only_splits,
    sha256_file,
    sha256_json,
    stratified_train_validation_partition,
)


def synthetic_rows(databases: int = 20, rows_per_database: int = 350) -> list[dict]:
    return [
        {
            "id": f"spider-1.0-train-{index:05d}",
            "source_split": "train",
            "source_index": index,
            "db_id": f"db_{index // rows_per_database:03d}",
            "question": f"question {index}",
            "reference_query": f"SELECT {index}",
            "source_row_sha256": f"hash-{index}",
        }
        for index in range(databases * rows_per_database)
    ]


def make_assets(root: Path, db_ids: set[str]) -> tuple[Path, Path]:
    tables = root / "tables.json"
    tables.write_text(
        json.dumps(
            [
                {
                    "db_id": db_id,
                    "table_names_original": ["items"],
                    "table_names": ["items"],
                    "column_names_original": [[-1, "*"], [0, "id"]],
                    "column_names": [[-1, "*"], [0, "id"]],
                    "column_types": ["text", "number"],
                    "primary_keys": [1],
                    "foreign_keys": [],
                }
                for db_id in sorted(db_ids)
            ]
        ),
        encoding="utf-8",
    )
    databases = root / "database"
    for db_id in db_ids:
        path = databases / db_id / f"{db_id}.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE items (id INTEGER PRIMARY KEY)")
    return tables, databases


def test_partition_is_exact_deterministic_and_database_stratified() -> None:
    rows = synthetic_rows()
    first = stratified_train_validation_partition(rows, validation_count=700, seed=23)
    second = stratified_train_validation_partition(rows, validation_count=700, seed=23)
    assert len(first["validation"]) == 700
    assert len(first["optimisation"]) == 6_300
    assert [row["id"] for row in first["validation"]] == [
        row["id"] for row in second["validation"]
    ]
    assert Counter(row["db_id"] for row in first["validation"]) == {
        f"db_{index:03d}": 35 for index in range(20)
    }
    validation_ids = {row["id"] for row in first["validation"]}
    optimisation_ids = {row["id"] for row in first["optimisation"]}
    assert not validation_ids & optimisation_ids
    assert len(validation_ids | optimisation_ids) == 7_000


def test_preparation_never_loads_or_materializes_official_heldout(
    tmp_path: Path,
) -> None:
    rows = synthetic_rows()
    tables, databases = make_assets(tmp_path, {row["db_id"] for row in rows})
    calls: list[tuple[str, str]] = []

    def loader(split: str, *, cache_dir: Path, revision: str) -> list[dict]:
        calls.append((split, revision))
        return rows

    output = tmp_path / "frozen"
    manifest = prepare_validation_only_splits(
        output,
        cache_dir=tmp_path / "cache",
        tables_json=tables,
        database_root=databases,
        source_loader=loader,
    )
    assert calls == [("train", DATASET_REVISION)]
    assert manifest["splits"]["validation"]["n"] == 700
    assert manifest["splits"]["optimisation"]["n"] == 6_300
    assert manifest["heldout_policy"]["accessed_during_preparation"] is False
    assert not (output / "test.jsonl").exists()
    assert not (output / "official_validation.jsonl").exists()
    stored = json.loads((output / "split_manifest.json").read_text(encoding="utf-8"))
    assert stored["manifest_sha256"] == manifest["manifest_sha256"]
    manifest_without_hash = dict(stored)
    del manifest_without_hash["manifest_sha256"]
    assert stored["manifest_sha256"] == sha256_json(manifest_without_hash)
    for split in ("optimisation", "validation"):
        split_manifest = stored["splits"][split]
        assert split_manifest["file_sha256"] == sha256_file(
            output / split_manifest["file"]
        )
    assert len(stored["schema_assets"]["database_sha256"]) == 20


def test_train_downloader_requests_only_the_pinned_train_shard(tmp_path: Path) -> None:
    parquet = tmp_path / "train.parquet"
    parquet.write_bytes(b"pinned train only")
    calls: list[dict] = []

    def downloader(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(parquet)

    path = download_pinned_train_parquet(
        cache_dir=tmp_path / "cache",
        expected_sha256=sha256_file(parquet),
        downloader=downloader,
    )

    assert path == parquet.resolve()
    assert calls == [
        {
            "repo_id": "xlangai/spider",
            "repo_type": "dataset",
            "filename": TRAIN_PARQUET_FILE,
            "revision": DATASET_REVISION,
            "cache_dir": str(tmp_path / "cache"),
        }
    ]


def test_missing_database_assets_fail_before_a_protocol_is_frozen(
    tmp_path: Path,
) -> None:
    tables = tmp_path / "tables.json"
    tables.write_text('[{"db_id":"db_a"}]', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Missing Spider database directory"):
        audit_schema_assets(
            ["db_a"], tables_json=tables, database_root=tmp_path / "missing"
        )


def test_malformed_schema_or_sqlite_assets_are_rejected(tmp_path: Path) -> None:
    tables = tmp_path / "tables.json"
    tables.write_text('[{"db_id":"db_a"}]', encoding="utf-8")
    database = tmp_path / "database" / "db_a" / "db_a.sqlite"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"not sqlite")
    with pytest.raises(FileNotFoundError, match="malformed schema entries=.*db_a"):
        audit_schema_assets(
            ["db_a"], tables_json=tables, database_root=tmp_path / "database"
        )
