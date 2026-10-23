from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pytest

from iar_spider.data import (
    DATASET_ID,
    DATASET_REVISION,
    OFFICIAL_REPOSITORY,
    OFFICIAL_REPOSITORY_REVISION,
    TRAIN_PARQUET_FILE,
    TRAIN_PARQUET_SHA256,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)
from iar_spider.evaluate_validation import (
    ExecutionLimits,
    GoldExecutionError,
    evaluate_validation,
    extract_sql,
    validate_read_only_statement,
)


def _database(path: Path, rows: list[tuple[int, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE items (id INTEGER, name TEXT, category TEXT)")
        connection.executemany("INSERT INTO items VALUES (?, ?, ?)", rows)


def _protocol(tmp_path: Path, gold_queries: list[str]) -> tuple[Path, Path, Path, list[dict]]:
    split_dir = tmp_path / "splits"
    database_root = tmp_path / "database"
    canonical = database_root / "db_a" / "db_a.sqlite"
    _database(canonical, [(1, "alpha", "target"), (2, "beta", "other")])
    tables = tmp_path / "tables.json"
    tables.write_text(
        json.dumps(
            [
                {
                    "db_id": "db_a",
                    "table_names_original": ["items"],
                    "table_names": ["items"],
                    "column_names_original": [[-1, "*"], [0, "id"], [0, "name"]],
                    "column_names": [[-1, "*"], [0, "id"], [0, "name"]],
                    "column_types": ["text", "number", "text"],
                    "primary_keys": [],
                    "foreign_keys": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    validation = [
        {
            "id": f"spider-1.0-train-{index:05d}",
            "source_split": "train",
            "source_index": index,
            "db_id": "db_a",
            "question": f"question {index}",
            "reference_query": query,
            "source_row_sha256": f"source-{index}",
            "experiment_split": "validation",
            "experiment_order": index,
        }
        for index, query in enumerate(gold_queries)
    ]
    validation_path = split_dir / "validation.jsonl"
    write_jsonl(validation_path, validation)
    manifest = {
        "schema_version": 1,
        "protocol": "spider-1.0-validation-only-v1",
        "source": {
            "dataset": DATASET_ID,
            "revision": DATASET_REVISION,
            "loaded_splits": ["train"],
            "train_parquet_file": TRAIN_PARQUET_FILE,
            "train_parquet_sha256": TRAIN_PARQUET_SHA256,
        },
        "splits": {
            "validation": {
                "n": len(validation),
                "file": "validation.jsonl",
                "file_sha256": sha256_file(validation_path),
                "ids_sha256": sha256_json([row["id"] for row in validation]),
                "rows_sha256": sha256_json(validation),
            }
        },
        "schema_assets": {
            "official_repository": OFFICIAL_REPOSITORY,
            "official_repository_revision": OFFICIAL_REPOSITORY_REVISION,
            "tables_json_sha256": sha256_file(tables),
            "database_sha256": {"db_a": sha256_file(canonical)},
        },
        "heldout_policy": {
            "accessed_during_preparation": False,
            "materialized_during_preparation": False,
        },
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_json(split_dir / "split_manifest.json", manifest)
    return split_dir, tables, database_root, validation


def _predictions(path: Path, validation: list[dict], outputs: list[str]) -> Path:
    write_jsonl(
        path,
        [
            {"id": row["id"], "response": output}
            for row, output in zip(validation, outputs, strict=True)
        ],
    )
    return path


def _limits() -> ExecutionLimits:
    return ExecutionLimits(
        example_timeout_seconds=3.0,
        query_timeout_seconds=0.25,
        sqlite_progress_ops=100,
        max_result_rows=1_000,
        max_result_bytes=100_000,
    )


def test_execution_accuracy_uses_official_denotation_semantics(tmp_path: Path) -> None:
    gold = [
        "SELECT name FROM items WHERE id = 1",
        "SELECT id, name FROM items ORDER BY id",
        "SELECT count(*) FROM items",
    ]
    split_dir, tables, databases, validation = _protocol(tmp_path, gold)
    predictions = _predictions(
        tmp_path / "predictions.jsonl",
        validation,
        [
            "```sql\nSELECT name FROM items WHERE id = 1\n```",
            "SQL: SELECT name, id FROM items ORDER BY id",
            "SELECT count(*) FROM items",
        ],
    )

    result = evaluate_validation(
        split_dir=split_dir,
        predictions_path=predictions,
        tables_json=tables,
        database_root=databases,
        output_dir=tmp_path / "evaluation",
        metric_mode="execution",
        limits=_limits(),
        expected_count=3,
    )

    assert result["summary"]["accuracy"] == 1.0
    assert result["manifest"]["official_dev_or_test_read"] is False
    assert result["manifest"]["execution_safety"]["process_isolation"] is True
    assert result["manifest"]["database_inventory"]["db_a"][0]["file"] == (
        "db_a/db_a.sqlite"
    )
    stored = json.loads(
        (tmp_path / "evaluation" / "evaluation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(stored["official_test_suite"]["revision"]) == 40
    unhashed = dict(stored)
    unhashed.pop("manifest_sha256")
    assert stored["manifest_sha256"] == sha256_json(unhashed)


def test_unicode_line_separator_inside_json_string_is_not_a_record_boundary(
    tmp_path: Path,
) -> None:
    query = "SELECT '\u2028'"
    split_dir, tables, databases, validation = _protocol(tmp_path, [query])
    predictions = _predictions(
        tmp_path / "predictions.jsonl", validation, [query]
    )
    result = evaluate_validation(
        split_dir=split_dir,
        predictions_path=predictions,
        tables_json=tables,
        database_root=databases,
        output_dir=tmp_path / "evaluation",
        metric_mode="execution",
        limits=_limits(),
        expected_count=1,
    )
    assert result["summary"]["accuracy"] == 1.0


def test_test_suite_mode_rejects_single_db_false_positive(tmp_path: Path) -> None:
    split_dir, tables, databases, validation = _protocol(
        tmp_path, ["SELECT name FROM items WHERE category = 'target'"]
    )
    predictions = _predictions(
        tmp_path / "predictions.jsonl",
        validation,
        ["SELECT name FROM items WHERE id = 1"],
    )
    execution = evaluate_validation(
        split_dir=split_dir,
        predictions_path=predictions,
        tables_json=tables,
        database_root=databases,
        output_dir=tmp_path / "execution",
        metric_mode="execution",
        limits=_limits(),
        expected_count=1,
    )
    assert execution["summary"]["accuracy"] == 1.0

    _database(
        databases / "db_a" / "db_a_testsuite_1.sqlite",
        [(1, "alpha", "other"), (2, "beta", "target")],
    )
    suite = evaluate_validation(
        split_dir=split_dir,
        predictions_path=predictions,
        tables_json=tables,
        database_root=databases,
        output_dir=tmp_path / "suite",
        metric_mode="test_suite",
        limits=_limits(),
        expected_count=1,
    )
    assert suite["summary"]["accuracy"] == 0.0
    assert suite["rows"][0]["score"]["status"] == "denotation_mismatch"


@pytest.mark.parametrize(
    "output,error",
    [
        ("DELETE FROM items", "only_one_select_statement_is_allowed|mutating_sql_keyword"),
        ("SELECT 1; SELECT 2", "multiple_or_missing_sql_statements"),
        ("```sql\nSELECT 1\n``` trailing", "ambiguous_fenced_output"),
    ],
)
def test_unsafe_or_ambiguous_outputs_are_blocked_before_execution(
    tmp_path: Path, output: str, error: str
) -> None:
    split_dir, tables, databases, validation = _protocol(tmp_path, ["SELECT count(*) FROM items"])
    predictions = _predictions(tmp_path / "predictions.jsonl", validation, [output])

    result = evaluate_validation(
        split_dir=split_dir,
        predictions_path=predictions,
        tables_json=tables,
        database_root=databases,
        output_dir=tmp_path / "evaluation",
        metric_mode="execution",
        limits=_limits(),
        expected_count=1,
    )

    assert result["summary"]["accuracy"] == 0.0
    assert result["rows"][0]["score"]["status"] == "invalid_or_blocked_sql"
    assert re_search(error, result["rows"][0]["score"]["error"])
    with sqlite3.connect(databases / "db_a" / "db_a.sqlite") as connection:
        assert connection.execute("SELECT count(*) FROM items").fetchone()[0] == 2


def re_search(pattern: str, value: str) -> bool:
    return re.search(pattern, value) is not None


def test_recursive_prediction_is_interrupted_and_scored_wrong(tmp_path: Path) -> None:
    split_dir, tables, databases, validation = _protocol(tmp_path, ["SELECT 1"])
    predictions = _predictions(
        tmp_path / "predictions.jsonl",
        validation,
        [
            (
                "WITH RECURSIVE loop(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM loop) "
                "SELECT sum(x) FROM loop"
            )
        ],
    )
    result = evaluate_validation(
        split_dir=split_dir,
        predictions_path=predictions,
        tables_json=tables,
        database_root=databases,
        output_dir=tmp_path / "evaluation",
        metric_mode="execution",
        limits=_limits(),
        expected_count=1,
    )
    assert result["summary"]["accuracy"] == 0.0
    assert result["rows"][0]["score"]["status"] in {
        "prediction_execution_error",
        "prediction_timeout",
    }


def test_invalid_gold_fails_the_entire_evaluation(tmp_path: Path) -> None:
    split_dir, tables, databases, validation = _protocol(
        tmp_path, ["SELECT missing_column FROM items"]
    )
    predictions = _predictions(tmp_path / "predictions.jsonl", validation, ["SELECT 1"])
    with pytest.raises(GoldExecutionError, match="Reference query failed"):
        evaluate_validation(
            split_dir=split_dir,
            predictions_path=predictions,
            tables_json=tables,
            database_root=databases,
            output_dir=tmp_path / "evaluation",
            metric_mode="execution",
            limits=_limits(),
            expected_count=1,
        )
    assert not (tmp_path / "evaluation").exists()


def test_heldout_artifact_and_database_symlink_fail_closed(tmp_path: Path) -> None:
    split_dir, tables, databases, validation = _protocol(tmp_path, ["SELECT 1"])
    predictions = _predictions(tmp_path / "predictions.jsonl", validation, ["SELECT 1"])
    (split_dir / "dev.jsonl").write_text("not-json-and-must-not-be-read", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Held-out Spider artifact present"):
        evaluate_validation(
            split_dir=split_dir,
            predictions_path=predictions,
            tables_json=tables,
            database_root=databases,
            output_dir=tmp_path / "evaluation",
            metric_mode="execution",
            limits=_limits(),
            expected_count=1,
        )

    (split_dir / "dev.jsonl").unlink()
    canonical = databases / "db_a" / "db_a.sqlite"
    outside = tmp_path / "outside.sqlite"
    canonical.replace(outside)
    canonical.symlink_to(outside)
    with pytest.raises(ValueError, match="Spider database escapes"):
        evaluate_validation(
            split_dir=split_dir,
            predictions_path=predictions,
            tables_json=tables,
            database_root=databases,
            output_dir=tmp_path / "evaluation",
            metric_mode="execution",
            limits=_limits(),
            expected_count=1,
        )


def test_sql_extraction_and_validation_are_conservative() -> None:
    assert extract_sql("SQL: SELECT 1") == "SELECT 1"
    assert extract_sql("```sqlite\nSELECT 1\n```") == "SELECT 1"
    validate_read_only_statement(
        "WITH values_cte AS (SELECT 1 AS x) SELECT x FROM values_cte"
    )
    with pytest.raises(ValueError, match="multiple_or_missing"):
        validate_read_only_statement("SELECT ';'; DELETE FROM items")
