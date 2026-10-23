from __future__ import annotations

import json
import math
import multiprocessing
import re
import sqlite3
import statistics
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import sqlparse
from sqlparse import tokens as sql_tokens

from .data import (
    DATASET_ID,
    DATASET_REVISION,
    DEFAULT_VALIDATION_COUNT,
    OFFICIAL_REPOSITORY,
    OFFICIAL_REPOSITORY_REVISION,
    TRAIN_PARQUET_FILE,
    TRAIN_PARQUET_SHA256,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)
from .official_test_suite import (
    OFFICIAL_EXEC_EVAL_SHA256,
    OFFICIAL_LICENSE_SHA256,
    OFFICIAL_TEST_SUITE_REPOSITORY,
    OFFICIAL_TEST_SUITE_REVISION,
    postprocess,
    remove_distinct,
    replace_current_year,
    result_equal,
)

SAFE_DB_ID = re.compile(r"^[A-Za-z0-9_-]+$")
FENCE = re.compile(r"```(?:sqlite|sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
OUTPUT_FIELDS = ("response", "output", "candidate", "completion")
MUTATING_KEYWORDS = {
    "ALTER",
    "ANALYZE",
    "ATTACH",
    "CREATE",
    "DELETE",
    "DETACH",
    "DROP",
    "INSERT",
    "PRAGMA",
    "REINDEX",
    "REPLACE",
    "UPDATE",
    "VACUUM",
}


@dataclass(frozen=True)
class ExecutionLimits:
    example_timeout_seconds: float = 10.0
    query_timeout_seconds: float = 4.0
    sqlite_progress_ops: int = 1_000
    max_result_rows: int = 100_000
    max_result_bytes: int = 16 * 1024 * 1024

    def validate(self) -> None:
        if not math.isfinite(self.example_timeout_seconds) or self.example_timeout_seconds <= 0:
            raise ValueError("example_timeout_seconds must be positive and finite")
        if not math.isfinite(self.query_timeout_seconds) or self.query_timeout_seconds <= 0:
            raise ValueError("query_timeout_seconds must be positive and finite")
        if self.sqlite_progress_ops <= 0 or self.max_result_rows <= 0 or self.max_result_bytes <= 0:
            raise ValueError("SQLite execution limits must be positive")


class GoldExecutionError(RuntimeError):
    """The pinned reference query or database failed; the evaluation is invalid."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    # File iteration splits only on physical newline delimiters.  ``str.splitlines``
    # would also split valid JSON strings containing U+2028/U+2029.
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"Expected JSON objects in {path}")
    return rows


def _under(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root.expanduser().resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes its authorized root: {resolved}") from exc
    return resolved


def extract_sql(output: str) -> str:
    """Extract one SQL-only response or one isolated SQL code fence."""
    text = output.strip()
    if not text:
        raise ValueError("empty_output")
    fences = FENCE.findall(text)
    if fences:
        if len(fences) != 1 or FENCE.sub("", text).strip():
            raise ValueError("ambiguous_fenced_output")
        text = fences[0].strip()
    elif text[:4].lower() == "sql:":
        text = text[4:].strip()
    if not text:
        raise ValueError("empty_sql")
    return text


def validate_read_only_statement(query: str) -> None:
    """Reject multiple statements and everything except a read-only SELECT/CTE."""
    statements = [statement for statement in sqlparse.parse(query) if str(statement).strip()]
    if len(statements) != 1:
        raise ValueError("multiple_or_missing_sql_statements")
    statement = statements[0]
    meaningful = [
        token
        for token in statement.flatten()
        if not token.is_whitespace and token.ttype not in sql_tokens.Comment
    ]
    if not meaningful:
        raise ValueError("empty_sql_statement")
    keyword_values = {
        token.value.upper()
        for token in meaningful
        if token.ttype in sql_tokens.Keyword
        or token.ttype in sql_tokens.Keyword.DML
        or token.ttype in sql_tokens.Keyword.DDL
    }
    forbidden = sorted(keyword_values & MUTATING_KEYWORDS)
    if forbidden:
        raise ValueError(f"mutating_sql_keyword:{','.join(forbidden)}")
    dml = [token.value.upper() for token in meaningful if token.ttype in sql_tokens.Keyword.DML]
    if not dml or any(operation != "SELECT" for operation in dml):
        raise ValueError("only_select_operations_are_allowed")


def normalize_query(query: str, *, keep_distinct: bool) -> str:
    validate_read_only_statement(query)
    normalized = postprocess(query)
    if not keep_distinct:
        normalized = remove_distinct(normalized)
    validate_read_only_statement(normalized)
    return normalized


def _authorizer(action: int, first: str | None, second: str | None, *_: Any) -> int:
    denied_names = (
        "SQLITE_ALTER_TABLE",
        "SQLITE_ANALYZE",
        "SQLITE_ATTACH",
        "SQLITE_CREATE_INDEX",
        "SQLITE_CREATE_TABLE",
        "SQLITE_CREATE_TEMP_INDEX",
        "SQLITE_CREATE_TEMP_TABLE",
        "SQLITE_CREATE_TEMP_TRIGGER",
        "SQLITE_CREATE_TEMP_VIEW",
        "SQLITE_CREATE_TRIGGER",
        "SQLITE_CREATE_VIEW",
        "SQLITE_DELETE",
        "SQLITE_DETACH",
        "SQLITE_DROP_INDEX",
        "SQLITE_DROP_TABLE",
        "SQLITE_DROP_TEMP_INDEX",
        "SQLITE_DROP_TEMP_TABLE",
        "SQLITE_DROP_TEMP_TRIGGER",
        "SQLITE_DROP_TEMP_VIEW",
        "SQLITE_DROP_TRIGGER",
        "SQLITE_DROP_VIEW",
        "SQLITE_INSERT",
        "SQLITE_PRAGMA",
        "SQLITE_REINDEX",
        "SQLITE_TRANSACTION",
        "SQLITE_UPDATE",
    )
    denied = {getattr(sqlite3, name) for name in denied_names if hasattr(sqlite3, name)}
    if action in denied:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and (second or first or "").lower() == "load_extension":
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _execute_read_only(path: str, query: str, limits: ExecutionLimits) -> list[tuple[Any, ...]]:
    database = Path(path).resolve()
    uri = f"{database.as_uri()}?mode=ro&immutable=1"
    connection = sqlite3.connect(
        uri,
        uri=True,
        timeout=min(1.0, limits.query_timeout_seconds),
        isolation_level=None,
    )
    try:
        connection.text_factory = lambda value: value.decode(errors="ignore")
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.set_authorizer(_authorizer)
        deadline = time.monotonic() + limits.query_timeout_seconds
        connection.set_progress_handler(
            lambda: int(time.monotonic() >= deadline), limits.sqlite_progress_ops
        )
        cursor = connection.execute(replace_current_year(query))
        result = cursor.fetchmany(limits.max_result_rows + 1)
        if len(result) > limits.max_result_rows:
            raise RuntimeError("result_row_limit_exceeded")
        if len(repr(result).encode("utf-8", errors="replace")) > limits.max_result_bytes:
            raise RuntimeError("result_byte_limit_exceeded")
        return result
    finally:
        connection.close()


def _execution_worker(
    sender: Connection,
    stage: Any,
    database_paths: tuple[str, ...],
    predicted_query: str,
    gold_query: str,
    order_matters: bool,
    limits: ExecutionLimits,
) -> None:
    try:
        checked = 0
        for database_path in database_paths:
            stage.value = 0
            try:
                gold_result = _execute_read_only(database_path, gold_query, limits)
            except (sqlite3.Error, OSError, RuntimeError, ValueError) as exc:
                sender.send(
                    {
                        "status": "gold_error",
                        "database": Path(database_path).name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                return
            stage.value = 1
            try:
                predicted_result = _execute_read_only(
                    database_path, predicted_query, limits
                )
            except (sqlite3.Error, OSError, RuntimeError, ValueError) as exc:
                sender.send(
                    {
                        "status": "prediction_execution_error",
                        "database": Path(database_path).name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                return
            if not result_equal(
                gold_result,
                predicted_result,
                order_matters=order_matters,
                seed=0,
            ):
                sender.send(
                    {
                        "status": "denotation_mismatch",
                        "database": Path(database_path).name,
                        "databases_checked": checked + 1,
                    }
                )
                return
            checked += 1
        sender.send({"status": "correct", "databases_checked": checked})
    finally:
        sender.close()


def execute_example(
    predicted_query: str,
    gold_query: str,
    database_paths: Iterable[Path],
    *,
    limits: ExecutionLimits,
) -> dict[str, Any]:
    """Evaluate one prediction in a disposable process."""
    limits.validate()
    paths = tuple(str(path.resolve()) for path in database_paths)
    if not paths:
        raise ValueError("At least one SQLite database is required")
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    stage = context.Value("i", 0)
    process = context.Process(
        target=_execution_worker,
        args=(
            sender,
            stage,
            paths,
            predicted_query,
            gold_query,
            "order by" in gold_query.lower(),
            limits,
        ),
    )
    process.start()
    sender.close()
    process.join(limits.example_timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        if process.is_alive():
            process.kill()
            process.join()
        if stage.value == 0:
            raise GoldExecutionError("Reference query timed out")
        return {"status": "prediction_timeout", "correct": False}
    if not receiver.poll():
        if stage.value == 0:
            raise GoldExecutionError(
                f"Reference-query worker failed with exit code {process.exitcode}"
            )
        return {
            "status": "prediction_worker_failure",
            "correct": False,
            "exit_code": process.exitcode,
        }
    result = receiver.recv()
    receiver.close()
    if result["status"] == "gold_error":
        raise GoldExecutionError(
            f"Reference query failed on {result['database']}: {result['error']}"
        )
    result["correct"] = result["status"] == "correct"
    return result


def _prediction_output(row: dict[str, Any]) -> tuple[str, str]:
    present = [field for field in OUTPUT_FIELDS if str(row.get(field, "")).strip()]
    if len(present) != 1:
        raise ValueError(
            f"Prediction {row.get('id')!r} must contain exactly one of {OUTPUT_FIELDS}"
        )
    field = present[0]
    return field, str(row[field])


def load_validation_protocol(
    split_dir: Path,
    *,
    tables_json: Path,
    database_root: Path,
    expected_count: int = DEFAULT_VALIDATION_COUNT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load exactly the internal-train validation split and verify every source hash."""
    split_dir = split_dir.resolve()
    manifest_path = _under(split_dir / "split_manifest.json", split_dir, "split manifest")
    validation_path = _under(split_dir / "validation.jsonl", split_dir, "validation split")
    if not manifest_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("Frozen Spider validation protocol is incomplete")
    for forbidden in ("dev.jsonl", "test.jsonl", "official_validation.jsonl"):
        if (split_dir / forbidden).exists() or (split_dir / forbidden).is_symlink():
            raise RuntimeError(f"Held-out Spider artifact present in validation-only split: {forbidden}")
    manifest = _read_json(manifest_path)
    if manifest.get("protocol") != "spider-1.0-validation-only-v1":
        raise ValueError("Unexpected Spider split protocol")
    stored_manifest_hash = manifest.get("manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    if stored_manifest_hash != sha256_json(unhashed):
        raise RuntimeError("Spider split manifest hash mismatch")
    source = manifest.get("source") or {}
    if (
        source.get("dataset") != DATASET_ID
        or source.get("revision") != DATASET_REVISION
        or source.get("loaded_splits") != ["train"]
        or source.get("train_parquet_file") != TRAIN_PARQUET_FILE
        or source.get("train_parquet_sha256") != TRAIN_PARQUET_SHA256
    ):
        raise RuntimeError("Spider validation is not derived solely from pinned official train")
    heldout = manifest.get("heldout_policy") or {}
    if (
        heldout.get("accessed_during_preparation") is not False
        or heldout.get("materialized_during_preparation") is not False
    ):
        raise RuntimeError("Spider held-out data entered the frozen validation protocol")
    split = (manifest.get("splits") or {}).get("validation") or {}
    if split.get("file") != "validation.jsonl" or int(split.get("n", -1)) != expected_count:
        raise RuntimeError("Unexpected Spider internal validation split")
    if split.get("file_sha256") != sha256_file(validation_path):
        raise RuntimeError("Spider validation file hash mismatch")
    rows = _read_jsonl(validation_path)
    if len(rows) != expected_count:
        raise RuntimeError(f"Expected {expected_count} Spider validation rows, found {len(rows)}")
    ids = [str(row.get("id", "")) for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Spider validation IDs are missing or duplicated")
    if any(
        row.get("source_split") != "train"
        or row.get("experiment_split") != "validation"
        or not str(row.get("reference_query", "")).strip()
        for row in rows
    ):
        raise RuntimeError("Spider validation rows violate the internal-train gold policy")
    if split.get("ids_sha256") != sha256_json(ids) or split.get("rows_sha256") != sha256_json(rows):
        raise RuntimeError("Spider validation content hash mismatch")

    tables_json = tables_json.resolve()
    database_root = database_root.resolve()
    assets = manifest.get("schema_assets") or {}
    if (
        assets.get("official_repository") != OFFICIAL_REPOSITORY
        or assets.get("official_repository_revision")
        != OFFICIAL_REPOSITORY_REVISION
        or assets.get("tables_json_sha256") != sha256_file(tables_json)
    ):
        raise RuntimeError("Spider tables.json differs from the frozen protocol")
    canonical_hashes = assets.get("database_sha256") or {}
    for db_id in sorted({str(row["db_id"]) for row in rows}):
        if not SAFE_DB_ID.fullmatch(db_id):
            raise ValueError(f"Unsafe Spider db_id: {db_id!r}")
        database = _under(
            database_root / db_id / f"{db_id}.sqlite",
            database_root,
            "Spider database",
        )
        if not database.is_file() or canonical_hashes.get(db_id) != sha256_file(database):
            raise RuntimeError(f"Spider canonical database hash mismatch: {db_id}")
    return rows, manifest


def database_suite(
    db_id: str,
    *,
    database_root: Path,
    metric_mode: str,
) -> tuple[Path, ...]:
    if not SAFE_DB_ID.fullmatch(db_id):
        raise ValueError(f"Unsafe Spider db_id: {db_id!r}")
    root = database_root.resolve()
    directory = _under(root / db_id, root, "Spider database directory")
    canonical = _under(directory / f"{db_id}.sqlite", root, "Spider canonical database")
    if metric_mode == "execution":
        paths = (canonical,)
    elif metric_mode == "test_suite":
        paths = tuple(
            sorted(
                _under(path, root, "Spider test-suite database")
                for path in directory.glob("*.sqlite")
            )
        )
        if len(paths) < 2:
            raise RuntimeError(
                f"Official test-suite assets are missing for {db_id}; "
                "refusing to label single-database execution as test-suite accuracy"
            )
    else:
        raise ValueError("metric_mode must be 'execution' or 'test_suite'")
    if not all(path.is_file() for path in paths) or canonical not in paths:
        raise FileNotFoundError(f"Incomplete Spider database suite for {db_id}")
    return paths


def evaluate_validation(
    *,
    split_dir: Path,
    predictions_path: Path,
    tables_json: Path,
    database_root: Path,
    output_dir: Path,
    metric_mode: str,
    keep_distinct: bool = False,
    limits: ExecutionLimits | None = None,
    expected_count: int = DEFAULT_VALIDATION_COUNT,
) -> dict[str, Any]:
    """Score a complete prediction file without opening Spider dev/test."""
    limits = limits or ExecutionLimits()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Spider evaluation: {output_dir}")
    validation, split_manifest = load_validation_protocol(
        split_dir,
        tables_json=tables_json,
        database_root=database_root,
        expected_count=expected_count,
    )
    predictions_path = predictions_path.resolve()
    predictions = _read_jsonl(predictions_path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in predictions:
        row_id = str(row.get("id", ""))
        if not row_id or row_id in by_id:
            raise ValueError(f"Missing or duplicate prediction ID: {row_id!r}")
        by_id[row_id] = row
    expected_ids = [str(row["id"]) for row in validation]
    if set(by_id) != set(expected_ids) or len(by_id) != expected_count:
        raise RuntimeError("Prediction IDs do not exactly match Spider internal validation")

    db_suites = {
        db_id: database_suite(
            db_id, database_root=database_root, metric_mode=metric_mode
        )
        for db_id in sorted({str(row["db_id"]) for row in validation})
    }
    scored = []
    for gold in validation:
        row_id = str(gold["id"])
        field, raw_output = _prediction_output(by_id[row_id])
        score: dict[str, Any]
        predicted_sql = None
        try:
            predicted_sql = normalize_query(
                extract_sql(raw_output), keep_distinct=keep_distinct
            )
        except ValueError as exc:
            score = {"correct": False, "status": "invalid_or_blocked_sql", "error": str(exc)}
        else:
            try:
                gold_sql = normalize_query(
                    str(gold["reference_query"]), keep_distinct=keep_distinct
                )
            except ValueError as exc:
                raise GoldExecutionError(
                    f"Invalid reference SQL for {row_id}: {exc}"
                ) from exc
            score = execute_example(
                predicted_sql,
                gold_sql,
                db_suites[str(gold["db_id"])],
                limits=limits,
            )
        scored.append(
            {
                "id": row_id,
                "source_index": int(gold["source_index"]),
                "db_id": str(gold["db_id"]),
                "prediction_output_field": field,
                "predicted_sql": predicted_sql,
                "score": score,
            }
        )

    database_inventory = {
        db_id: [
            {
                "file": str(path.relative_to(database_root.resolve())),
                "sha256": sha256_file(path),
            }
            for path in paths
        ]
        for db_id, paths in db_suites.items()
    }
    local_official_adapter = Path(__file__).with_name("official_test_suite.py")
    vendored_license = (
        Path(__file__).parents[2] / "reproduction/third_party" / "test-suite-sql-eval" / "LICENSE"
    )
    if sha256_file(vendored_license) != OFFICIAL_LICENSE_SHA256:
        raise RuntimeError("Vendored official test-suite license hash mismatch")
    correct = [bool(row["score"]["correct"]) for row in scored]
    summary = {
        "status": "complete",
        "metric": (
            "official_spider_single_database_execution_accuracy"
            if metric_mode == "execution"
            else "official_spider_distilled_test_suite_accuracy"
        ),
        "n": len(scored),
        "correct": sum(correct),
        "accuracy": statistics.fmean(correct),
        "status_counts": dict(sorted(Counter(row["score"]["status"] for row in scored).items())),
    }
    manifest = {
        "schema_version": 1,
        "protocol": "spider-1.0-internal-validation-official-exec-v1",
        "manifest_hash_scope": "canonical JSON excluding manifest_sha256",
        "validation_only": True,
        "official_dev_or_test_read": False,
        "metric_mode": metric_mode,
        "keep_distinct": keep_distinct,
        "split_manifest_sha256": sha256_file(split_dir.resolve() / "split_manifest.json"),
        "split_protocol_manifest_sha256": split_manifest["manifest_sha256"],
        "predictions": {
            "file": str(predictions_path),
            "sha256": sha256_file(predictions_path),
        },
        "spider_source": {
            "repository": OFFICIAL_REPOSITORY,
            "revision": OFFICIAL_REPOSITORY_REVISION,
            "dataset": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
        },
        "official_test_suite": {
            "repository": OFFICIAL_TEST_SUITE_REPOSITORY,
            "revision": OFFICIAL_TEST_SUITE_REVISION,
            "upstream_exec_eval_sha256": OFFICIAL_EXEC_EVAL_SHA256,
            "plug_value": False,
            "license": str(vendored_license),
            "license_sha256": OFFICIAL_LICENSE_SHA256,
            "local_hardened_adapter": str(local_official_adapter),
            "local_hardened_adapter_sha256": sha256_file(local_official_adapter),
            "compatibility": (
                "upstream postprocessing and denotation-equivalence semantics; "
                "query execution replaced by read-only process isolation"
            ),
        },
        "execution_safety": {
            "sqlite_uri": "mode=ro&immutable=1",
            "query_only": True,
            "sqlite_authorizer": "mutations, attach/detach, pragma and load_extension denied",
            "single_statement_select_only": True,
            "process_isolation": True,
            **limits.__dict__,
        },
        "database_inventory": database_inventory,
        "outputs": {
            "rows_content_sha256": sha256_json(scored),
            "summary_content_sha256": sha256_json(summary),
        },
        "summary": summary,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(output_dir / "rows.jsonl", scored)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "evaluation_manifest.json", manifest)
    return {"summary": summary, "manifest": manifest, "rows": scored}
