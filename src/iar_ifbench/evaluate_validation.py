from __future__ import annotations

import importlib.util
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from .data import (
    DATA_LICENSE,
    DEFAULT_SEED,
    DEFAULT_VALIDATION_COUNT,
    EXPECTED_TRAIN_INSTANCES,
    HELDOUT_DATASET_ID,
    HELDOUT_DATASET_REVISION,
    OPEN_INSTRUCT_REPOSITORY_REVISION,
    TRAIN_DATASET_CONFIG,
    TRAIN_DATASET_ID,
    TRAIN_DATASET_REVISION,
    TRAIN_DATASET_SPLIT,
    TRAIN_PARQUET_FILE,
    TRAIN_PARQUET_SHA256,
    sha256_file,
    sha256_json,
    write_json,
    write_jsonl,
)
from .official_ifeval_old import (
    EXPECTED_FUNCTION_NAMES,
    GROUND_TRUTH_UTILS_SHA256,
    IF_FUNCTIONS_SHA256,
    OPEN_INSTRUCT_LICENSE,
    OPEN_INSTRUCT_LICENSE_SHA256,
    OPEN_INSTRUCT_REPOSITORY,
    OPEN_INSTRUCT_REVISION,
    score_with_ifeval_verifier_old,
)

FROZEN_SPLIT_DIR_NAME = "final_seed23_10pct_v2"
FROZEN_SPLIT_MANIFEST_SHA256 = (
    "4a22386b4ff6fff63ec1f4b7fa6949638ad680eb451c80c4d9281f444d9d2304"
)
SPLIT_PROTOCOL = "rlvr-ifeval-to-ifbench-ood-validation-only-v1"
MAX_PREDICTIONS_BYTES = 512 * 1024 * 1024
MAX_OUTPUT_CHARACTERS = 1_000_000
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
ROW_ID = re.compile(r"^rlvr-ifeval-train-[0-9]{5}$")
HELDOUT_NAME_MARKERS = (
    "ifbench_test",
    "ifbench-test",
    "official_test",
    "official-test",
)
HELDOUT_EXACT_FILENAMES = frozenset({"test.json", "test.jsonl", "test.parquet"})
VALIDATION_ROW_FIELDS = frozenset(
    {
        "id",
        "source_index",
        "source_dataset",
        "source_split",
        "prompt",
        "verifier_args",
        "constraint_type",
        "constraint",
        "source_row_sha256",
        "experiment_split",
        "experiment_order",
    }
)


class IFBenchEvaluationError(RuntimeError):
    """Raised when the validation-only contract cannot be proven."""


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IFBenchEvaluationError(f"Cannot parse JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise IFBenchEvaluationError(f"Expected a JSON object in {path}")
    return value


def _load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise IFBenchEvaluationError(
                        f"Blank JSONL line in {path} at line {line_number}"
                    )
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise IFBenchEvaluationError(
                        f"Invalid JSONL in {path} at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise IFBenchEvaluationError(
                        f"JSONL row {line_number} in {path} is not an object"
                    )
                rows.append(value)
    except (OSError, UnicodeError) as exc:
        raise IFBenchEvaluationError(f"Cannot read {path}: {exc}") from exc
    return rows


def _is_heldout_name(name: str) -> bool:
    normalized = name.lower()
    return normalized in HELDOUT_EXACT_FILENAMES or any(
        marker in normalized for marker in HELDOUT_NAME_MARKERS
    )


def _reject_heldout_artifacts(split_dir: Path) -> Path:
    """Inspect names only; never open a held-out artifact."""
    if split_dir.is_symlink():
        raise IFBenchEvaluationError("Frozen split directory may not be a symlink")
    resolved = split_dir.resolve()
    if not resolved.is_dir():
        raise IFBenchEvaluationError(f"Frozen split directory is missing: {resolved}")
    if any(_is_heldout_name(part) for part in resolved.parts):
        raise IFBenchEvaluationError(
            "Split path appears to reference held-out IFBench data"
        )
    for root, directory_names, file_names in os.walk(resolved, followlinks=False):
        root_path = Path(root)
        for name in directory_names + file_names:
            candidate = root_path / name
            if _is_heldout_name(name):
                raise IFBenchEvaluationError(
                    f"Held-out/official-test artifact is forbidden: {candidate}"
                )
            if candidate.is_symlink():
                raise IFBenchEvaluationError(
                    f"Symlinks are forbidden inside the frozen split: {candidate}"
                )
    return resolved


def _validate_split_manifest(
    split_dir: Path,
    *,
    expected_count: int,
    expected_manifest_sha256: str,
    expected_dir_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split_dir = _reject_heldout_artifacts(split_dir)
    if split_dir.name != expected_dir_name:
        raise IFBenchEvaluationError(
            f"Expected frozen split directory {expected_dir_name!r}, got {split_dir.name!r}"
        )
    manifest_path = split_dir / "split_manifest.json"
    if not manifest_path.is_file():
        raise IFBenchEvaluationError("Frozen split_manifest.json is missing")
    manifest = _load_json_object(manifest_path)
    stored_manifest_sha256 = manifest.get("manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    computed_manifest_sha256 = sha256_json(unhashed)
    if stored_manifest_sha256 != computed_manifest_sha256:
        raise IFBenchEvaluationError("Frozen split manifest self-hash mismatch")
    if stored_manifest_sha256 != expected_manifest_sha256:
        raise IFBenchEvaluationError(
            f"Frozen split manifest is not the pre-registered {expected_dir_name}"
        )

    expected_top_level = {
        "schema_version": 1,
        "protocol": SPLIT_PROTOCOL,
        "seed": DEFAULT_SEED,
    }
    for key, expected in expected_top_level.items():
        if manifest.get(key) != expected:
            raise IFBenchEvaluationError(
                f"Frozen split manifest {key!r} mismatch: {manifest.get(key)!r}"
            )
    source = manifest.get("training_source")
    expected_source = {
        "dataset": TRAIN_DATASET_ID,
        "config": TRAIN_DATASET_CONFIG,
        "source_split": TRAIN_DATASET_SPLIT,
        "revision": TRAIN_DATASET_REVISION,
        "license": DATA_LICENSE,
        "instances": EXPECTED_TRAIN_INSTANCES,
        "unique_prompts": 14_970,
    }
    if not isinstance(source, dict) or any(
        source.get(key) != expected for key, expected in expected_source.items()
    ):
        raise IFBenchEvaluationError("RLVR-IFeval training provenance mismatch")
    heldout = manifest.get("heldout_policy")
    if (
        not isinstance(heldout, dict)
        or heldout.get("dataset") != HELDOUT_DATASET_ID
        or heldout.get("revision") != HELDOUT_DATASET_REVISION
        or heldout.get("accessed_during_preparation") is not False
        or heldout.get("materialized_during_preparation") is not False
        or heldout.get("may_be_accessed_only_after_all_selection_gates_are_frozen")
        is not True
    ):
        raise IFBenchEvaluationError("Held-out IFBench policy/provenance mismatch")
    evaluation_sources = manifest.get("evaluation_sources")
    open_instruct = (
        evaluation_sources.get("open_instruct")
        if isinstance(evaluation_sources, dict)
        else None
    )
    if (
        not isinstance(open_instruct, dict)
        or open_instruct.get("repository") != OPEN_INSTRUCT_REPOSITORY
        or open_instruct.get("revision") != OPEN_INSTRUCT_REPOSITORY_REVISION
        or open_instruct.get("license") != OPEN_INSTRUCT_LICENSE
    ):
        raise IFBenchEvaluationError("Pinned Open Instruct provenance mismatch")

    splits = manifest.get("splits")
    validation = splits.get("validation") if isinstance(splits, dict) else None
    if (
        not isinstance(validation, dict)
        or validation.get("file") != "validation.jsonl"
        or validation.get("n") != expected_count
    ):
        raise IFBenchEvaluationError("Frozen validation split metadata mismatch")
    validation_path = split_dir / "validation.jsonl"
    if not validation_path.is_file() or validation_path.is_symlink():
        raise IFBenchEvaluationError("Frozen validation.jsonl must be a regular file")
    if validation.get("file_sha256") != sha256_file(validation_path):
        raise IFBenchEvaluationError("Frozen validation file hash mismatch")
    rows = _load_jsonl_objects(validation_path)
    if len(rows) != expected_count:
        raise IFBenchEvaluationError(
            f"Expected {expected_count} validation instances, found {len(rows)}"
        )
    if validation.get("rows_sha256") != sha256_json(rows):
        raise IFBenchEvaluationError("Frozen validation row-content hash mismatch")
    ids = [row.get("id") for row in rows]
    if validation.get("ids_sha256") != sha256_json(ids):
        raise IFBenchEvaluationError("Frozen validation ID hash mismatch")
    if len(set(ids)) != expected_count:
        raise IFBenchEvaluationError("Frozen validation IDs are not unique")

    constraint_counts: Counter[str] = Counter()
    source_indexes: set[int] = set()
    for order, row in enumerate(rows):
        if set(row) != VALIDATION_ROW_FIELDS:
            raise IFBenchEvaluationError(
                f"Validation row {order} has unexpected/missing fields"
            )
        row_id = row["id"]
        source_index = row["source_index"]
        verifier_args = row["verifier_args"]
        if not isinstance(row_id, str) or not ROW_ID.fullmatch(row_id):
            raise IFBenchEvaluationError(f"Invalid validation row ID at order {order}")
        if (
            not isinstance(source_index, int)
            or isinstance(source_index, bool)
            or not 0 <= source_index < EXPECTED_TRAIN_INSTANCES
            or source_index in source_indexes
        ):
            raise IFBenchEvaluationError(f"Invalid/duplicate source index for {row_id}")
        source_indexes.add(source_index)
        if (
            row["source_dataset"] != TRAIN_DATASET_ID
            or row["source_split"] != TRAIN_DATASET_SPLIT
            or row["experiment_split"] != "validation"
            or row["experiment_order"] != order
        ):
            raise IFBenchEvaluationError(
                f"Split/source provenance mismatch for {row_id}"
            )
        if not isinstance(row["prompt"], str) or not row["prompt"].strip():
            raise IFBenchEvaluationError(f"Missing prompt for {row_id}")
        if not isinstance(row["constraint"], str) or not row["constraint"].strip():
            raise IFBenchEvaluationError(f"Missing constraint description for {row_id}")
        if (
            not isinstance(row["constraint_type"], str)
            or not row["constraint_type"].strip()
        ):
            raise IFBenchEvaluationError(f"Missing constraint type for {row_id}")
        if (
            not isinstance(verifier_args, dict)
            or not isinstance(verifier_args.get("func_name"), str)
            or not verifier_args["func_name"]
        ):
            raise IFBenchEvaluationError(
                f"Invalid legacy verifier arguments for {row_id}"
            )
        if not isinstance(row["source_row_sha256"], str) or not HEX_SHA256.fullmatch(
            row["source_row_sha256"]
        ):
            raise IFBenchEvaluationError(f"Invalid source-row hash for {row_id}")
        constraint_counts[row["constraint_type"]] += 1
    if validation.get("constraint_type_counts") != dict(
        sorted(constraint_counts.items())
    ):
        raise IFBenchEvaluationError("Validation constraint-family counts mismatch")
    if validation.get("unique_prompts") != len({row["prompt"] for row in rows}):
        raise IFBenchEvaluationError("Validation unique-prompt count mismatch")
    return manifest, rows


def load_validation_protocol(
    split_dir: Path,
    *,
    expected_count: int = DEFAULT_VALIDATION_COUNT,
    expected_manifest_sha256: str = FROZEN_SPLIT_MANIFEST_SHA256,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load only the frozen internal validation rows after all provenance gates."""
    manifest, rows = _validate_split_manifest(
        split_dir,
        expected_count=expected_count,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_dir_name=FROZEN_SPLIT_DIR_NAME,
    )
    return rows, manifest


def _load_predictions(
    path: Path, expected_ids: list[str], *, max_output_characters: int
) -> tuple[dict[str, str], int]:
    if path.is_symlink():
        raise IFBenchEvaluationError("Predictions JSONL may not be a symlink")
    path = path.resolve()
    if any(_is_heldout_name(part) for part in path.parts):
        raise IFBenchEvaluationError(
            "Predictions path appears to reference held-out/official-test data"
        )
    if not path.is_file():
        raise IFBenchEvaluationError(f"Predictions JSONL is missing: {path}")
    size = path.stat().st_size
    if size > MAX_PREDICTIONS_BYTES:
        raise IFBenchEvaluationError("Predictions JSONL exceeds the safety size limit")
    rows = _load_jsonl_objects(path)
    predictions: dict[str, str] = {}
    for line_number, row in enumerate(rows, start=1):
        if set(row) != {"id", "output"}:
            raise IFBenchEvaluationError(
                f"Prediction row {line_number} must contain exactly 'id' and 'output'"
            )
        row_id, output = row["id"], row["output"]
        if not isinstance(row_id, str) or not row_id:
            raise IFBenchEvaluationError(f"Invalid prediction ID at line {line_number}")
        if not isinstance(output, str):
            raise IFBenchEvaluationError(
                f"Prediction output for {row_id} is not a string"
            )
        if len(output) > max_output_characters:
            raise IFBenchEvaluationError(f"Prediction output for {row_id} is too large")
        if row_id in predictions:
            raise IFBenchEvaluationError(f"Duplicate prediction ID: {row_id}")
        predictions[row_id] = output
    expected = set(expected_ids)
    observed = set(predictions)
    if observed != expected or len(rows) != len(expected_ids):
        missing = sorted(expected - observed)[:3]
        extra = sorted(observed - expected)[:3]
        raise IFBenchEvaluationError(
            f"Prediction IDs do not exactly match validation (missing={missing}, extra={extra})"
        )
    return predictions, size


def _load_module(path: Path) -> ModuleType:
    module_name = f"_iar_pinned_if_functions_{sha256_file(path)[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise IFBenchEvaluationError(f"Cannot load pinned verifier source: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise IFBenchEvaluationError(
            f"Cannot import pinned IF verifier: {exc}"
        ) from exc
    return module


def _load_pinned_function_map(
    open_instruct_root: Path,
    *,
    expected_if_functions_sha256: str,
    expected_ground_truth_sha256: str,
    expected_license_sha256: str,
    expected_function_names: frozenset[str],
) -> tuple[Mapping[str, Callable[..., Any]], dict[str, Any]]:
    if open_instruct_root.is_symlink():
        raise IFBenchEvaluationError("Open Instruct root may not be a symlink")
    root = open_instruct_root.resolve()
    if not root.is_dir():
        raise IFBenchEvaluationError(f"Open Instruct checkout is missing: {root}")
    source_paths = {
        "if_functions.py": root / "open_instruct" / "if_functions.py",
        "ground_truth_utils.py": root / "open_instruct" / "ground_truth_utils.py",
        "LICENSE": root / "LICENSE",
    }
    if any(not path.is_file() or path.is_symlink() for path in source_paths.values()):
        raise IFBenchEvaluationError(
            "Pinned Open Instruct source/license files are missing"
        )
    observed_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    expected_hashes = {
        "if_functions.py": expected_if_functions_sha256,
        "ground_truth_utils.py": expected_ground_truth_sha256,
        "LICENSE": expected_license_sha256,
    }
    if observed_hashes != expected_hashes:
        raise IFBenchEvaluationError(
            f"Open Instruct revision/source hash mismatch: {observed_hashes}"
        )
    module = _load_module(source_paths["if_functions.py"])
    function_map = getattr(module, "IF_FUNCTIONS_MAP", None)
    if (
        not isinstance(function_map, dict)
        or set(function_map) != expected_function_names
    ):
        raise IFBenchEvaluationError("Pinned IF_FUNCTIONS_MAP has an unexpected schema")
    if any(not callable(function) for function in function_map.values()):
        raise IFBenchEvaluationError("Pinned IF_FUNCTIONS_MAP contains a non-callable")
    provenance = {
        "repository": OPEN_INSTRUCT_REPOSITORY,
        "revision": OPEN_INSTRUCT_REVISION,
        "license": OPEN_INSTRUCT_LICENSE,
        "files_sha256": observed_hashes,
    }
    return function_map, provenance


def _group_metrics(result_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for row in result_rows:
        grouped[row["constraint_type"]].append(row["score"])
    return {
        name: {
            "n": len(scores),
            "passed": int(sum(scores)),
            "accuracy": statistics.fmean(scores),
        }
        for name, scores in sorted(grouped.items())
    }


def evaluate_validation(
    *,
    split_dir: Path,
    predictions_path: Path,
    open_instruct_root: Path,
    output_dir: Path,
    expected_count: int = DEFAULT_VALIDATION_COUNT,
    expected_manifest_sha256: str = FROZEN_SPLIT_MANIFEST_SHA256,
    expected_dir_name: str = FROZEN_SPLIT_DIR_NAME,
    expected_if_functions_sha256: str = IF_FUNCTIONS_SHA256,
    expected_ground_truth_sha256: str = GROUND_TRUTH_UTILS_SHA256,
    expected_license_sha256: str = OPEN_INSTRUCT_LICENSE_SHA256,
    expected_function_names: frozenset[str] = EXPECTED_FUNCTION_NAMES,
    max_output_characters: int = MAX_OUTPUT_CHARACTERS,
) -> dict[str, Any]:
    """Evaluate only the frozen RLVR-IFeval internal validation split."""
    if expected_count <= 0 or max_output_characters <= 0:
        raise ValueError("Evaluation limits must be positive")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {output_dir}")
    split_manifest, validation_rows = _validate_split_manifest(
        split_dir,
        expected_count=expected_count,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_dir_name=expected_dir_name,
    )
    expected_ids = [row["id"] for row in validation_rows]
    predictions, predictions_bytes = _load_predictions(
        predictions_path, expected_ids, max_output_characters=max_output_characters
    )
    function_map, verifier_provenance = _load_pinned_function_map(
        open_instruct_root,
        expected_if_functions_sha256=expected_if_functions_sha256,
        expected_ground_truth_sha256=expected_ground_truth_sha256,
        expected_license_sha256=expected_license_sha256,
        expected_function_names=expected_function_names,
    )

    result_rows: list[dict[str, Any]] = []
    for row in validation_rows:
        row_id = row["id"]
        try:
            score = score_with_ifeval_verifier_old(
                predictions[row_id], row["verifier_args"], function_map
            )
        except Exception as exc:
            raise IFBenchEvaluationError(
                f"Pinned IFEvalVerifierOld failed for {row_id}: {type(exc).__name__}: {exc}"
            ) from exc
        result_rows.append(
            {
                "id": row_id,
                "source_index": row["source_index"],
                "experiment_order": row["experiment_order"],
                "constraint_type": row["constraint_type"],
                "verifier_function": row["verifier_args"]["func_name"],
                "score": score,
                "passed": score == 1.0,
                "output_sha256": sha256_json(predictions[row_id]),
            }
        )

    scores = [row["score"] for row in result_rows]
    summary: dict[str, Any] = {
        "metric": "ifeval_old_constraint_accuracy",
        "n": len(scores),
        "passed": int(sum(scores)),
        "accuracy": statistics.fmean(scores),
        "strict_accuracy": None,
        "loose_accuracy": None,
        "strict_loose_status": (
            "not_defined_by_pinned_ifeval_verifier_old_for_single-constraint_rlvr_rows"
        ),
        "by_constraint_type": _group_metrics(result_rows),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir / "rows.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(rows_path, result_rows)
    write_json(summary_path, summary)
    local_adapter = Path(__file__).with_name("official_ifeval_old.py")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "rlvr-ifeval-internal-validation-evaluation-v1",
        "manifest_hash_scope": "canonical JSON excluding manifest_sha256",
        "dataset": {
            "id": TRAIN_DATASET_ID,
            "revision": TRAIN_DATASET_REVISION,
            "pinned_train_parquet": TRAIN_PARQUET_FILE,
            "pinned_train_parquet_sha256": TRAIN_PARQUET_SHA256,
            "source_split": TRAIN_DATASET_SPLIT,
            "experiment_split": "validation",
            "instances": expected_count,
            "frozen_directory_name": expected_dir_name,
            "split_manifest_sha256": sha256_file(
                split_dir.resolve() / "split_manifest.json"
            ),
            "split_protocol_manifest_sha256": split_manifest["manifest_sha256"],
        },
        "heldout_safety": {
            "dataset": HELDOUT_DATASET_ID,
            "revision": HELDOUT_DATASET_REVISION,
            "accessed": False,
            "materialized": False,
            "name_scan_passed_before_any_split_file_read": True,
        },
        "verifier": {
            "class": "IFEvalVerifierOld",
            **verifier_provenance,
            "local_adapter_sha256": sha256_file(local_adapter),
            "function_names": sorted(function_map),
            "strict_accuracy_defined": False,
            "loose_accuracy_defined": False,
        },
        "prediction_contract": {
            "format": "JSONL objects with exactly id and output",
            "all_validation_ids_exactly_once": True,
            "max_file_bytes": MAX_PREDICTIONS_BYTES,
            "max_output_characters": max_output_characters,
            "input_bytes": predictions_bytes,
            "input_sha256": sha256_file(predictions_path),
        },
        "outputs": {
            "rows": {"file": rows_path.name, "sha256": sha256_file(rows_path)},
            "summary": {
                "file": summary_path.name,
                "sha256": sha256_file(summary_path),
            },
        },
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    manifest_path = output_dir / "evaluation_manifest.json"
    write_json(manifest_path, manifest)
    return {"summary": summary, "manifest": manifest, "rows": result_rows}
