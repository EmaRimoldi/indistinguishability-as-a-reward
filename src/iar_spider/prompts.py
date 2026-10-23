from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

TASK_PROMPT_VERSION = "spider-1.0-sql-only-v1-no-system"
ATTRIBUTION_PROMPT_VERSION = "spider-1.0-pointwise-self-attribution-v1-no-system"
TARGET_MODEL_NAME = "DeepSeek V4 Flash"

_TASK_TEMPLATE = """Database: {db_id}
Schema:
{schema}

Question: {question}

Return exactly one executable SQLite query and nothing else. Do not use Markdown fences or add an explanation."""

_ATTRIBUTION_TEMPLATE = """You are {target_model_name}. You are given a text-to-SQL task and one candidate response.

Task:
{task_statement}

Candidate response:
{candidate}

Would you plausibly have produced this candidate response yourself for this task?
Return exactly one token: 1 for yes or 0 for no."""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_schema_catalog(tables_json: Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(tables_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError("Spider tables.json must contain a JSON list")
    catalog: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not str(row.get("db_id", "")).strip():
            raise ValueError("Malformed Spider schema entry")
        db_id = str(row["db_id"])
        if db_id in catalog:
            raise ValueError(f"Duplicate Spider schema: {db_id}")
        catalog[db_id] = row
    return catalog


def render_schema(schema: dict[str, Any]) -> str:
    tables = list(schema.get("table_names_original") or [])
    columns = list(schema.get("column_names_original") or [])
    types = list(schema.get("column_types") or [])
    if len(columns) != len(types) or not tables:
        raise ValueError(f"Malformed schema for {schema.get('db_id')!r}")
    by_table: list[list[str]] = [[] for _ in tables]
    for (table_index, column), column_type in zip(columns, types, strict=True):
        table_index = int(table_index)
        if table_index < 0:
            continue
        if table_index >= len(tables):
            raise ValueError("Spider column references an unknown table")
        by_table[table_index].append(f"{column} {column_type}")
    lines = [
        f"{table}({', '.join(by_table[index])})"
        for index, table in enumerate(tables)
    ]
    foreign_keys = []
    for left, right in schema.get("foreign_keys") or []:
        left_table, left_column = columns[int(left)]
        right_table, right_column = columns[int(right)]
        foreign_keys.append(
            f"{tables[int(left_table)]}.{left_column} = "
            f"{tables[int(right_table)]}.{right_column}"
        )
    if foreign_keys:
        lines.append("Foreign keys: " + "; ".join(foreign_keys))
    return "\n".join(lines)


def task_prompt(*, db_id: str, question: str, schema: dict[str, Any]) -> str:
    if str(schema.get("db_id")) != db_id:
        raise ValueError("Schema db_id does not match the example")
    return _TASK_TEMPLATE.format(
        db_id=db_id,
        schema=render_schema(schema),
        question=question.strip(),
    )


def attribution_prompt(task_statement: str, candidate: str) -> str:
    return _ATTRIBUTION_TEMPLATE.format(
        target_model_name=TARGET_MODEL_NAME,
        task_statement=task_statement.strip(),
        candidate=candidate.strip(),
    )


def template_hashes() -> dict[str, str]:
    return {
        TASK_PROMPT_VERSION: sha256_text(_TASK_TEMPLATE),
        ATTRIBUTION_PROMPT_VERSION: sha256_text(_ATTRIBUTION_TEMPLATE),
    }
