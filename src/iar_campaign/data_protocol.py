from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

KODCODE_DATASET = "KodCode/KodCode-Light-RL-10K"
KODCODE_REVISION = "dcf78a8bbba9a613b596ce993c4921a38687dfcc"
KODCODE_SOURCE_SHA256 = (
    "cd777b078dd700d7e87c2f42ae2cb1d28de91625da92066fffda219eb9193c93"
)
KODCODE_EXPECTED_ROWS = 10_000
SEED = 23


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _hash_order(seed: int, *parts: str) -> str:
    return hashlib.sha256(
        (str(seed) + "\0" + "\0".join(parts)).encode("utf-8")
    ).hexdigest()


def _largest_remainder_targets(
    counts: Counter[str], size: int, total: int, seed: int, role: str
) -> dict[str, int]:
    exact = {key: value * size / total for key, value in counts.items()}
    target = {key: math.floor(value) for key, value in exact.items()}
    remainder = size - sum(target.values())
    order = sorted(
        counts,
        key=lambda key: (
            -(exact[key] - target[key]),
            _hash_order(seed, role, key),
        ),
    )
    for key in order[:remainder]:
        target[key] += 1
    return target


def grouped_stratified_partition(
    rows: list[dict[str, Any]],
    *,
    role_sizes: Mapping[str, int],
    group_key: Callable[[dict[str, Any]], str],
    stratum_key: Callable[[dict[str, Any]], str],
    seed: int = SEED,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Assign whole source groups while closely matching proportional strata.

    Multi-row groups are placed first by a deterministic deficit cost.  The
    many singleton groups then fill every role exactly, interleaving roles to
    avoid starving later partitions.  This keeps source variants together and
    provides deterministic, auditable deviations from largest-remainder
    targets.
    """

    if not rows or sum(role_sizes.values()) != len(rows):
        raise ValueError("Role sizes must be positive and exhaust the source rows")
    if any(size <= 0 for size in role_sizes.values()):
        raise ValueError("Every role must have a positive size")
    identifiers = [str(row.get("id", "")) for row in rows]
    if any(not value for value in identifiers) or len(set(identifiers)) != len(rows):
        raise ValueError("Rows must have unique non-empty IDs")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        key = group_key(source)
        if not key:
            raise ValueError(f"Empty partition group for {source['id']}")
        groups[key].append(source)

    source_counts = Counter(stratum_key(row) for row in rows)
    targets = {
        role: _largest_remainder_targets(source_counts, size, len(rows), seed, role)
        for role, size in role_sizes.items()
    }
    actual = {role: Counter() for role in role_sizes}
    remaining = dict(role_sizes)
    assignments: dict[str, str] = {}

    multi = sorted(
        ((key, values) for key, values in groups.items() if len(values) > 1),
        key=lambda item: (-len(item[1]), _hash_order(seed, "multi", item[0])),
    )
    for key, values in multi:
        group_counts = Counter(stratum_key(row) for row in values)
        candidates = [role for role, capacity in remaining.items() if capacity >= len(values)]
        if not candidates:
            raise RuntimeError(f"No role can hold grouped source variants: {key}")

        def cost(role: str) -> tuple[float, str]:
            delta = 0.0
            for stratum, count in group_counts.items():
                wanted = targets[role].get(stratum, 0)
                before = actual[role][stratum] - wanted
                after = before + count
                delta += (after * after - before * before) / max(wanted, 1)
            return delta, _hash_order(seed, "tie", key, role)

        role = min(candidates, key=cost)
        assignments[key] = role
        remaining[role] -= len(values)
        actual[role].update(group_counts)

    singleton_pools: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for key, values in groups.items():
        if len(values) == 1:
            singleton_pools[stratum_key(values[0])].append((key, values[0]))
    for stratum, values in singleton_pools.items():
        values.sort(key=lambda item: _hash_order(seed, "singleton", stratum, item[0]))

    singleton_needs = dict(remaining)
    singleton_done = {role: 0 for role in role_sizes}
    while any(remaining.values()):
        roles = [role for role, capacity in remaining.items() if capacity]
        role = min(
            roles,
            key=lambda name: (
                singleton_done[name] / max(singleton_needs[name], 1),
                _hash_order(seed, "role", name, str(singleton_done[name])),
            ),
        )
        strata = [name for name, values in singleton_pools.items() if values]
        if not strata:
            raise RuntimeError("Singleton groups were exhausted before exact sizing")

        def singleton_cost(stratum: str) -> tuple[float, str]:
            wanted = targets[role].get(stratum, 0)
            before = actual[role][stratum] - wanted
            delta = ((before + 1) ** 2 - before**2) / max(wanted, 1)
            return delta, _hash_order(seed, "stratum", role, stratum)

        stratum = min(strata, key=singleton_cost)
        key, _ = singleton_pools[stratum].pop(0)
        assignments[key] = role
        actual[role][stratum] += 1
        remaining[role] -= 1
        singleton_done[role] += 1

    result: dict[str, list[dict[str, Any]]] = {role: [] for role in role_sizes}
    group_roles: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        key = group_key(row)
        role = assignments[key]
        value = dict(row)
        value["partition_group"] = key
        value["partition_role"] = role
        result[role].append(value)
        group_roles[key].add(role)
    if any(len(values) != 1 for values in group_roles.values()):
        raise AssertionError("A source-variant group leaked across roles")
    for role, values in result.items():
        values.sort(key=lambda row: _hash_order(seed, "row", role, str(row["id"])))
        if len(values) != role_sizes[role]:
            raise AssertionError(f"Exact role size failed for {role}")
        for index, value in enumerate(values):
            value["partition_order"] = index

    deviations = {
        role: {
            stratum: actual[role][stratum] - targets[role][stratum]
            for stratum in sorted(source_counts)
        }
        for role in role_sizes
    }
    audit = {
        "method": "deterministic grouped proportional stratification v1",
        "seed": seed,
        "source_rows": len(rows),
        "group_count": len(groups),
        "multirow_group_count": sum(len(values) > 1 for values in groups.values()),
        "maximum_group_size": max(len(values) for values in groups.values()),
        "source_stratum_counts": dict(sorted(source_counts.items())),
        "role_targets": targets,
        "role_actual": {role: dict(sorted(values.items())) for role, values in actual.items()},
        "role_deviations": deviations,
        "maximum_absolute_stratum_deviation": max(
            abs(value)
            for role_values in deviations.values()
            for value in role_values.values()
        ),
        "group_assignment_sha256": sha256_json(sorted(assignments.items())),
    }
    return result, audit


def _file_spec(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "file": path.name,
        "n": len(rows),
        "file_sha256": sha256_file(path),
        "ids_sha256": sha256_json([row["id"] for row in rows]),
        "rows_sha256": sha256_json(rows),
    }


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def prepare_kodcode_splits(
    source_parquet: Path, output_dir: Path, *, seed: int = SEED
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite frozen KodCode split: {output_dir}")
    if sha256_file(source_parquet) != KODCODE_SOURCE_SHA256:
        raise RuntimeError("KodCode pinned source parquet hash mismatch")
    source = pq.read_table(source_parquet).to_pylist()
    if len(source) != KODCODE_EXPECTED_ROWS:
        raise ValueError("Unexpected KodCode source row count")

    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(source):
        metadata = raw.get("metadata") or {}
        identifier = str(raw.get("question_id") or "").strip()
        question = str(raw.get("question") or "").strip()
        solution = str(raw.get("solution") or "").strip()
        tests = str(raw.get("test") or "").strip()
        subset = str(raw.get("subset") or "").strip()
        difficulty = str(raw.get("gpt_difficulty") or "").strip().lower()
        if not all((identifier, question, solution, tests, subset, difficulty)):
            raise ValueError(f"KodCode row {index} has an empty required field")
        prompt_id = str(metadata.get("prompt_id") or "").strip()
        group = f"prompt:{prompt_id}" if prompt_id else f"question:{sha256_json(_normalize_space(question))}"
        rows.append(
            {
                "id": identifier,
                "source_split": "train",
                "source_index": index,
                "subset": subset,
                "gpt_difficulty": difficulty,
                "question": question,
                "reference_solution": solution,
                "hidden_tests": tests,
                "prompt_id": prompt_id,
                "source_variant_group": group,
                "source_row_sha256": sha256_json(raw),
            }
        )

    role_sizes = {
        "probe100": 100,
        "pilot_delta274": 274,
        "train_core_delta626": 626,
        "validation300": 300,
        "sealed_test1000": 1_000,
        "train_extension3000": 3_000,
        "reserve": 4_700,
    }
    roles, audit = grouped_stratified_partition(
        rows,
        role_sizes=role_sizes,
        group_key=lambda row: str(row["source_variant_group"]),
        stratum_key=lambda row: f"{row['subset']}|{row['gpt_difficulty']}",
        seed=seed,
    )
    derived = {
        "probe100": roles["probe100"],
        "pilot374": roles["probe100"] + roles["pilot_delta274"],
        "train_core1000": (
            roles["probe100"]
            + roles["pilot_delta274"]
            + roles["train_core_delta626"]
        ),
        "validation300": roles["validation300"],
        "reserve": roles["reserve"],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    specs: dict[str, Any] = {}
    for name, values in derived.items():
        path = output_dir / f"{name}.jsonl"
        write_jsonl(path, values)
        specs[name] = _file_spec(path, values)
    sealed_dir = output_dir / "sealed"
    sealed_dir.mkdir(mode=0o700)
    for name in ("sealed_test1000", "train_extension3000"):
        path = sealed_dir / f"{name}.jsonl"
        write_jsonl(path, roles[name])
        path.chmod(0o600)
        spec = _file_spec(path, roles[name])
        spec["file"] = f"sealed/{path.name}"
        specs[name] = spec

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "kodcode-light-rl-10k-campaign-seed23-v1",
        "source": {
            "dataset": KODCODE_DATASET,
            "revision": KODCODE_REVISION,
            "source_file": "data/train-00000-of-00001.parquet",
            "source_file_sha256": KODCODE_SOURCE_SHA256,
            "row_count": KODCODE_EXPECTED_ROWS,
        },
        "seed": seed,
        "partition_audit": audit,
        "splits": specs,
        "nesting": "probe100 subset of pilot374 subset of train_core1000",
        "gold_policy": (
            "reference_solution and hidden_tests are evaluator-only; actor and "
            "discriminator prompts may read question only"
        ),
        "sealed_policy": (
            "sealed_test1000 and train_extension3000 are excluded from every "
            "current training, validation and model-selection config"
        ),
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_json(output_dir / "split_manifest.json", manifest)
    return manifest


def prepare_spider_campaign_splits(
    output_dir: Path,
    *,
    cache_dir: Path,
    tables_json: Path,
    database_root: Path,
    seed: int = SEED,
) -> dict[str, Any]:
    from iar_spider.data import (
        DATASET_CONFIG,
        DATASET_ID,
        DATASET_REVISION,
        EXPECTED_HELDOUT,
        EXPECTED_TRAIN,
        OFFICIAL_HELDOUT_SPLIT,
        OFFICIAL_TRAIN_SPLIT,
        TRAIN_PARQUET_FILE,
        TRAIN_PARQUET_SHA256,
        audit_schema_assets,
        load_source_split,
    )

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite frozen Spider split: {output_dir}")
    rows = load_source_split(OFFICIAL_TRAIN_SPLIT, cache_dir=cache_dir)
    if len(rows) != EXPECTED_TRAIN:
        raise ValueError("Unexpected Spider train row count")
    assets = audit_schema_assets(
        (row["db_id"] for row in rows),
        tables_json=tables_json,
        database_root=database_root,
    )
    for row in rows:
        row["source_variant_group"] = "question:" + sha256_json(
            _normalize_space(str(row["question"]))
        )

    role_sizes = {
        "probe100": 100,
        "pilot_delta274": 274,
        "train_core_delta626": 626,
        "validation300": 300,
        "train_extension3000": 3_000,
        "reserve": 2_700,
    }
    roles, audit = grouped_stratified_partition(
        rows,
        role_sizes=role_sizes,
        group_key=lambda row: str(row["source_variant_group"]),
        stratum_key=lambda row: str(row["db_id"]),
        seed=seed,
    )
    derived = {
        "probe100": roles["probe100"],
        "pilot374": roles["probe100"] + roles["pilot_delta274"],
        "train_core1000": (
            roles["probe100"]
            + roles["pilot_delta274"]
            + roles["train_core_delta626"]
        ),
        "validation300": roles["validation300"],
        "reserve": roles["reserve"],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    specs: dict[str, Any] = {}
    for name, values in derived.items():
        path = output_dir / f"{name}.jsonl"
        write_jsonl(path, values)
        specs[name] = _file_spec(path, values)
    sealed_dir = output_dir / "sealed"
    sealed_dir.mkdir(mode=0o700)
    extension_path = sealed_dir / "train_extension3000.jsonl"
    write_jsonl(extension_path, roles["train_extension3000"])
    extension_path.chmod(0o600)
    specs["train_extension3000"] = _file_spec(
        extension_path, roles["train_extension3000"]
    )
    specs["train_extension3000"]["file"] = f"sealed/{extension_path.name}"

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "spider-1.0-campaign-seed23-v1",
        "source": {
            "dataset": DATASET_ID,
            "config": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "train_parquet_file": TRAIN_PARQUET_FILE,
            "train_parquet_sha256": TRAIN_PARQUET_SHA256,
            "loaded_splits": [OFFICIAL_TRAIN_SPLIT],
            "official_train_count": EXPECTED_TRAIN,
        },
        "seed": seed,
        "partition_audit": audit,
        "splits": specs,
        "schema_assets": assets,
        "nesting": "probe100 subset of pilot374 subset of train_core1000",
        "gold_policy": (
            "reference_query is evaluator-only; actor and discriminator prompts "
            "may read question and database schema only"
        ),
        "heldout_policy": {
            "dataset_split": OFFICIAL_HELDOUT_SPLIT,
            "expected_count": EXPECTED_HELDOUT,
            "accessed_during_preparation": False,
            "materialized_during_preparation": False,
            "access_only_after_all_selection_gates": True,
        },
        "sealed_policy": (
            "train_extension3000 is excluded from current training, validation "
            "and model-selection configs"
        ),
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_json(output_dir / "split_manifest.json", manifest)
    return manifest

