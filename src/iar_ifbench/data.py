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

TRAIN_DATASET_ID = "allenai/RLVR-IFeval"
TRAIN_DATASET_CONFIG = "default"
TRAIN_DATASET_SPLIT = "train"
TRAIN_DATASET_REVISION = "47c03c73621c4aab2b824b7818681117d662770e"
EXPECTED_TRAIN_INSTANCES = 14_973
TRAIN_PARQUET_FILE = "data/train-00000-of-00001.parquet"
TRAIN_PARQUET_SHA256 = "75f1b1f63039044034fe75d22f51f77f257a79291d0084262bb43695d4aac971"

HELDOUT_DATASET_ID = "allenai/IFBench_test"
HELDOUT_DATASET_CONFIG = "default"
# The Hub repository calls its only split "train", but it is the benchmark's
# released semantic test set.  Selection code must never load this split.
HELDOUT_SOURCE_SPLIT = "train"
HELDOUT_DATASET_REVISION = "2e8a48de45ff3bf41242f927254ca81b59ca3ae2"
EXPECTED_HELDOUT_INSTANCES = 300

DATA_LICENSE = "ODC-BY-1.0"
RESPONSIBLE_USE_URL = "https://allenai.org/responsible-use"
IFBENCH_REPOSITORY = "https://github.com/allenai/IFBench.git"
IFBENCH_REPOSITORY_REVISION = "fcd289db21d43aaa96c6d9291d32561cd6e19305"
OPEN_INSTRUCT_REPOSITORY = "https://github.com/allenai/open-instruct.git"
OPEN_INSTRUCT_REPOSITORY_REVISION = "ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5"
CODE_LICENSE = "Apache-2.0"

DEFAULT_VALIDATION_COUNT = 1_501
DEFAULT_SEED = 23


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


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


def load_training_pool(
    *,
    cache_dir: Path,
    revision: str = TRAIN_DATASET_REVISION,
) -> list[dict[str, Any]]:
    """Load and normalize only the pinned RLVR-IFeval training pool.

    There is deliberately no held-out loader in this module.  IFBench_test is
    a separate Hub repository and cannot be reached by the selection entrypoint.
    """
    parquet_path = download_pinned_train_parquet(
        cache_dir=cache_dir,
        revision=revision,
    )
    dataset = pq.read_table(parquet_path).to_pylist()
    if len(dataset) != EXPECTED_TRAIN_INSTANCES:
        raise ValueError(
            "Unexpected RLVR-IFeval train size: "
            f"{len(dataset)} != {EXPECTED_TRAIN_INSTANCES}"
        )

    rows: list[dict[str, Any]] = []
    for source_index, source in enumerate(dataset):
        messages = source.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 1
            or not isinstance(messages[0], dict)
            or messages[0].get("role") != "user"
        ):
            raise ValueError(
                f"RLVR-IFeval row {source_index} must contain exactly one user message"
            )
        prompt = str(messages[0].get("content", "")).strip()
        constraint_type = str(source.get("constraint_type", "")).strip()
        constraint = str(source.get("constraint", "")).strip()
        try:
            verifier_args = json.loads(str(source.get("ground_truth", "")))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid verifier JSON in RLVR-IFeval row {source_index}"
            ) from exc
        if (
            not prompt
            or not constraint_type
            or not constraint
            or not isinstance(verifier_args, dict)
            or not str(verifier_args.get("func_name", "")).strip()
        ):
            raise ValueError(f"Incomplete RLVR-IFeval row {source_index}")

        fingerprint_payload = {
            "messages": messages,
            "ground_truth": verifier_args,
            "dataset": source.get("dataset"),
            "constraint_type": constraint_type,
            "constraint": constraint,
        }
        rows.append(
            {
                "id": f"rlvr-ifeval-train-{source_index:05d}",
                "source_index": source_index,
                "source_dataset": TRAIN_DATASET_ID,
                "source_split": TRAIN_DATASET_SPLIT,
                "prompt": prompt,
                # Verifier metadata is retained for external task scoring only.
                "verifier_args": verifier_args,
                "constraint_type": constraint_type,
                "constraint": constraint,
                "source_row_sha256": sha256_json(fingerprint_payload),
            }
        )
    return rows


def download_pinned_train_parquet(
    *,
    cache_dir: Path,
    revision: str = TRAIN_DATASET_REVISION,
    expected_sha256: str = TRAIN_PARQUET_SHA256,
    downloader: Callable[..., str] = hf_hub_download,
) -> Path:
    """Download exactly the pinned train shard and verify immutable contents."""
    path = Path(
        downloader(
            repo_id=TRAIN_DATASET_ID,
            repo_type="dataset",
            filename=TRAIN_PARQUET_FILE,
            revision=revision,
            cache_dir=str(cache_dir),
        )
    ).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pinned RLVR-IFeval train parquet is missing: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(
            "Pinned RLVR-IFeval train parquet hash mismatch: "
            f"expected {expected_sha256}, observed {observed}"
        )
    return path


def _stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}|{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _validation_quotas(
    by_constraint: dict[str, list[dict[str, Any]]],
    validation_count: int,
    seed: int,
) -> dict[str, int]:
    total = sum(len(rows) for rows in by_constraint.values())
    if not 0 < validation_count < total:
        raise ValueError("validation_count must lie strictly between zero and train size")
    exact = {
        constraint: len(rows) * validation_count / total
        for constraint, rows in by_constraint.items()
    }
    quotas = {constraint: math.floor(value) for constraint, value in exact.items()}
    remaining = validation_count - sum(quotas.values())
    order = sorted(
        by_constraint,
        key=lambda constraint: (
            -(exact[constraint] - quotas[constraint]),
            _stable_seed(seed + 7_919, constraint),
            constraint,
        ),
    )
    for constraint in order[:remaining]:
        quotas[constraint] += 1
    if sum(quotas.values()) != validation_count:
        raise AssertionError("Largest-remainder allocation missed validation target")
    return quotas


def _select_prompt_groups(
    groups: list[list[dict[str, Any]]], *, quota: int
) -> set[int]:
    """Find an exact group-level subset so duplicate prompts cannot leak."""
    predecessor: list[tuple[int, int] | None] = [None] * (quota + 1)
    predecessor[0] = (-1, -1)
    for group_index, group in enumerate(groups):
        size = len(group)
        for previous_total in range(quota - size, -1, -1):
            next_total = previous_total + size
            if predecessor[previous_total] is not None and predecessor[next_total] is None:
                predecessor[next_total] = (previous_total, group_index)
    if predecessor[quota] is None:
        sizes = Counter(len(group) for group in groups)
        raise ValueError(f"Cannot allocate exact validation quota {quota}; group sizes={sizes}")
    selected: set[int] = set()
    total = quota
    while total:
        step = predecessor[total]
        if step is None:
            raise AssertionError("Broken prompt-group subset reconstruction")
        total, group_index = step
        selected.add(group_index)
    return selected


def stratified_train_validation_partition(
    rows: list[dict[str, Any]],
    *,
    validation_count: int = DEFAULT_VALIDATION_COUNT,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[dict[str, Any]]]:
    """Stratify by constraint type while keeping exact duplicate prompts together."""
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("RLVR-IFeval rows contain duplicate IDs")

    prompt_constraints: dict[str, set[str]] = defaultdict(set)
    prompt_verifiers: dict[str, set[str]] = defaultdict(set)
    by_constraint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_row in rows:
        row = dict(source_row)
        prompt = str(row.get("prompt", "")).strip()
        constraint_type = str(row.get("constraint_type", "")).strip()
        if not prompt or not constraint_type:
            raise ValueError(f"Incomplete normalized row {row.get('id')!r}")
        prompt_constraints[prompt].add(constraint_type)
        prompt_verifiers[prompt].add(canonical_json(row.get("verifier_args")))
        by_constraint[constraint_type].append(row)
    inconsistent_prompts = sorted(
        prompt
        for prompt in prompt_constraints
        if len(prompt_constraints[prompt]) != 1 or len(prompt_verifiers[prompt]) != 1
    )
    if inconsistent_prompts:
        raise ValueError(
            "Duplicate prompts disagree on constraint/verifier metadata: "
            f"{inconsistent_prompts[:3]}"
        )

    quotas = _validation_quotas(by_constraint, validation_count, seed)
    validation: list[dict[str, Any]] = []
    optimisation: list[dict[str, Any]] = []
    for constraint_type in sorted(by_constraint):
        prompt_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in by_constraint[constraint_type]:
            prompt_groups[row["prompt"]].append(row)
        groups = list(prompt_groups.values())
        random.Random(_stable_seed(seed, constraint_type)).shuffle(groups)
        selected = _select_prompt_groups(groups, quota=quotas[constraint_type])
        for group_index, group in enumerate(groups):
            destination = validation if group_index in selected else optimisation
            destination.extend(group)

    random.Random(seed + 65_537).shuffle(validation)
    random.Random(seed + 131_071).shuffle(optimisation)
    for split_name, split_rows in (
        ("validation", validation),
        ("optimisation", optimisation),
    ):
        for order, row in enumerate(split_rows):
            row["experiment_split"] = split_name
            row["experiment_order"] = order

    validation_prompts = {row["prompt"] for row in validation}
    optimisation_prompts = {row["prompt"] for row in optimisation}
    if validation_prompts & optimisation_prompts:
        raise AssertionError("An exact duplicate prompt leaked across internal splits")
    partitioned_ids = [row["id"] for row in validation + optimisation]
    if len(partitioned_ids) != len(ids) or set(partitioned_ids) != set(ids):
        raise AssertionError("Internal IF partition is not exhaustive and disjoint")
    return {"optimisation": optimisation, "validation": validation}


def _duplicate_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["prompt"] for row in rows)
    return {
        "instances": len(rows),
        "unique_prompts": len(counts),
        "duplicate_prompt_groups": sum(count > 1 for count in counts.values()),
        "duplicate_instances_beyond_first": sum(count - 1 for count in counts.values()),
    }


def prepare_validation_only_splits(
    output_dir: Path,
    *,
    cache_dir: Path,
    seed: int = DEFAULT_SEED,
    validation_count: int = DEFAULT_VALIDATION_COUNT,
    training_loader: Callable[..., list[dict[str, Any]]] = load_training_pool,
) -> dict[str, Any]:
    """Freeze training/validation data without loading the IFBench test repo."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite frozen IF split: {output_dir}")
    train = training_loader(cache_dir=cache_dir, revision=TRAIN_DATASET_REVISION)
    if len(train) != EXPECTED_TRAIN_INSTANCES:
        raise ValueError(
            "Unexpected RLVR-IFeval train size: "
            f"{len(train)} != {EXPECTED_TRAIN_INSTANCES}"
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
        summaries[name] = {
            "n": len(rows),
            "unique_prompts": len({row["prompt"] for row in rows}),
            "file": path.name,
            "file_sha256": sha256_file(path),
            "ids_sha256": sha256_json([row["id"] for row in rows]),
            "rows_sha256": sha256_json(rows),
            "constraint_type_counts": dict(
                sorted(Counter(row["constraint_type"] for row in rows).items())
            ),
        }

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "rlvr-ifeval-to-ifbench-ood-validation-only-v1",
        "manifest_hash_scope": "canonical JSON excluding manifest_sha256",
        "seed": seed,
        "partition_method": (
            "constraint-type-stratified largest-remainder allocation from the pinned "
            "RLVR-IFeval pool; exact duplicate prompts are indivisible groups"
        ),
        "training_source": {
            "dataset": TRAIN_DATASET_ID,
            "config": TRAIN_DATASET_CONFIG,
            "source_split": TRAIN_DATASET_SPLIT,
            "revision": TRAIN_DATASET_REVISION,
            "license": DATA_LICENSE,
            "responsible_use_url": RESPONSIBLE_USE_URL,
            "contains_third_party_model_output_subject_to_additional_terms": True,
            **_duplicate_summary(train),
        },
        "splits": summaries,
        "heldout_policy": {
            "dataset": HELDOUT_DATASET_ID,
            "config": HELDOUT_DATASET_CONFIG,
            "source_split_name": HELDOUT_SOURCE_SPLIT,
            "semantic_role": "official IFBench released test set",
            "revision": HELDOUT_DATASET_REVISION,
            "license": DATA_LICENSE,
            "expected_count": EXPECTED_HELDOUT_INSTANCES,
            "accessed_during_preparation": False,
            "materialized_during_preparation": False,
            "may_be_accessed_only_after_all_selection_gates_are_frozen": True,
        },
        "evaluation_sources": {
            "ifbench": {
                "repository": IFBENCH_REPOSITORY,
                "revision": IFBENCH_REPOSITORY_REVISION,
                "license": CODE_LICENSE,
                "role": "official OOD test checkers and prompt-level loose accuracy",
            },
            "open_instruct": {
                "repository": OPEN_INSTRUCT_REPOSITORY,
                "revision": OPEN_INSTRUCT_REPOSITORY_REVISION,
                "license": CODE_LICENSE,
                "role": "RLVR-IFeval legacy training verifier (IFEvalVerifierOld)",
            },
        },
        "verifier_policy": {
            "verifier_args_stored_for_external_task_scoring": True,
            "verifier_args_exposed_to_actor_or_discriminator": False,
        },
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    write_json(output_dir / "split_manifest.json", manifest)
    return manifest
