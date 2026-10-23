from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .data import (
    DEFAULT_SEED,
    HELDOUT_DATASET_ID,
    TRAIN_DATASET_ID,
    TRAIN_DATASET_REVISION,
    sha256_file,
    sha256_json,
)
from .evaluate_validation import (
    FROZEN_SPLIT_DIR_NAME,
    IFBenchEvaluationError,
    _reject_heldout_artifacts,
)
from .official_ifeval_old import (
    GROUND_TRUTH_UTILS_SHA256,
    IF_FUNCTIONS_SHA256,
    OPEN_INSTRUCT_LICENSE_SHA256,
)
from .prompts import ATTRIBUTION_PROMPT_VERSION, TASK_PROMPT_VERSION

PROTOCOL = "ifbench-five-domain-screen-1p5b-v1"
ACTOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ACTOR_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
JUDGE_MODEL = "deepseek/deepseek-v4-flash"
JUDGE_PROVIDER = "StreamLake"
OPTIMISATION_COUNT = 13_472
VALIDATION_COUNT = 1_501
CHECKPOINTS = [0, 3368, 6736, 10104, 13472, 16840, 20208, 23576, 26944]
CHECKPOINT_EPOCHS = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
REWARD_TRANSFORMS = {
    "absolute_logp_1",
    "log_p_own",
    "p_own",
    "logit_p_own",
    "binary_own",
}
PENDING = "PENDING_GATE"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _under(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} is not under the scratch root: {path}") from exc


def config_sha256(config_path: Path) -> str:
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def validate_static_config(config: dict[str, Any], *, root: Path) -> None:
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected IFBench screen protocol")
    if int(config["project"].get("seed", -1)) != DEFAULT_SEED:
        raise ValueError("IFBench screen seed changed")
    if (
        config["actor"].get("model_id") != ACTOR_MODEL
        or config["actor"].get("revision") != ACTOR_REVISION
        or config["actor"].get("local_files_only") is not True
    ):
        raise ValueError("IFBench screen is frozen to the pinned local Qwen actor")
    dataset = config["dataset"]
    if (
        dataset.get("dataset_id") != TRAIN_DATASET_ID
        or dataset.get("revision") != TRAIN_DATASET_REVISION
        or dataset.get("source_file") != "optimisation.jsonl"
        or dataset.get("validation_file") != "validation.jsonl"
        or int(dataset.get("optimisation_count", -1)) != OPTIMISATION_COUNT
        or int(dataset.get("validation_count", -1)) != VALIDATION_COUNT
        or dataset.get("actor_prompt_version") != TASK_PROMPT_VERSION
    ):
        raise ValueError("IFBench dataset/split/prompt contract changed")
    if Path(config["paths"]["split_dir"]).name != FROZEN_SPLIT_DIR_NAME:
        raise ValueError("IFBench screen must use final_seed23_10pct_v2")

    training = config["training"]
    evaluation = config["evaluation"]
    if (
        int(training.get("num_generations", -1)) != 4
        or int(training.get("num_epochs", -1)) != 2
        or int(training.get("max_steps", -1)) != CHECKPOINTS[-1]
        or int(training.get("save_steps", -1)) != CHECKPOINTS[1]
        or list(evaluation.get("checkpoints") or []) != CHECKPOINTS
        or list(evaluation.get("checkpoint_epochs") or []) != CHECKPOINT_EPOCHS
    ):
        raise ValueError("IFBench G/epoch/checkpoint schedule changed")
    if int(training.get("generation_batch_size", -1)) != 32:
        raise ValueError("IFBench exact epochs require generation_batch_size=32")
    unique_prompts_per_generation_batch = int(training["generation_batch_size"]) // int(
        training["num_generations"]
    )
    if OPTIMISATION_COUNT % unique_prompts_per_generation_batch:
        raise ValueError("IFBench optimisation rows must fill every generation batch")
    expected_calls = OPTIMISATION_COUNT * 4 * 2
    if (
        int(config["trial"].get("expected_reward_calls", -1)) != expected_calls
        or int(config["reward"].get("max_reward_calls", -1)) != expected_calls
    ):
        raise ValueError("Reward-call budget is not 13472 * G=4 * 2 epochs")
    attempts = int(config["reward"].get("max_attempts_per_call", -1))
    if (
        attempts != 3
        or int(config["reward"].get("max_http_requests", -1))
        != expected_calls * attempts
    ):
        raise ValueError(
            "HTTP budget must allow at most three attempts per logical call"
        )
    reward_value = str(config["reward"].get("value", ""))
    if reward_value != PENDING and reward_value not in REWARD_TRANSFORMS:
        raise ValueError("IFBench reward must remain pending or use a gated transform")

    discriminator = config["discriminator"]
    routing = discriminator.get("provider_routing") or {}
    if (
        discriminator.get("model_id") != JUDGE_MODEL
        or discriminator.get("expected_provider") != JUDGE_PROVIDER
        or routing.get("order") != [JUDGE_PROVIDER]
        or routing.get("allow_fallbacks") is not False
        or routing.get("require_parameters") is not True
        or discriminator.get("strict_logprobs") is not True
        or discriminator.get("attribution_prompt_version") != ATTRIBUTION_PROMPT_VERSION
    ):
        raise ValueError("Judge must be exact StreamLake, no-fallback, strict-logprobs")
    if evaluation.get("validation_only") is not True:
        raise ValueError("IFBench screen evaluation must be validation-only")
    if evaluation.get("entrypoint") != "iar_ifbench.evaluate_validation":
        raise ValueError("Unexpected IFBench validation evaluator")
    if (
        evaluation.get("select_on") != "task_utility"
        or evaluation.get("task_utility") != "ifeval_old_constraint_accuracy"
        or evaluation.get("tie_break") != "earliest_checkpoint"
        or evaluation.get("characterize_reward") is not True
        or int(evaluation.get("expected_reward_calls", -1))
        != VALIDATION_COUNT * len(CHECKPOINTS)
    ):
        raise ValueError("Checkpoint selector must use frozen IF task utility")

    serialized = json.dumps(config, sort_keys=True).lower()
    for forbidden in (
        "ifbench_test",
        "ifbench-test",
        "official_test",
        "official-test",
        "test_file",
        "test_path",
        "test_dataset",
    ):
        if forbidden in serialized:
            raise ValueError(f"Forbidden held-out field in IFBench config: {forbidden}")
    scratch = _resolve(root, config["paths"]["scratch_root"])
    for key in (
        "split_dir",
        "open_instruct_root",
        "hf_cache",
        "experiments",
        "checkpoints",
        "tmp",
    ):
        _under(_resolve(root, config["paths"][key]), scratch, key)


def _audit_frozen_split(
    config: dict[str, Any], root: Path
) -> tuple[list[str], str | None]:
    blockers: list[str] = []
    split_dir = _resolve(root, config["paths"]["split_dir"])
    try:
        split_dir = _reject_heldout_artifacts(split_dir)
    except (IFBenchEvaluationError, OSError) as exc:
        return [f"frozen split safety check failed: {exc}"], None
    manifest_path = split_dir / "split_manifest.json"
    try:
        manifest = _read_json(manifest_path)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        return [f"missing/invalid frozen split manifest: {exc}"], None
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    expected_manifest = str(config["dataset"]["manifest_sha256"])
    if manifest.get("manifest_sha256") != sha256_json(unhashed):
        blockers.append("frozen split manifest self-hash mismatch")
    if manifest.get("manifest_sha256") != expected_manifest:
        blockers.append("frozen split manifest_sha256 mismatch")
    source = manifest.get("training_source") or {}
    if (
        source.get("dataset") != TRAIN_DATASET_ID
        or source.get("revision") != TRAIN_DATASET_REVISION
        or source.get("source_split") != "train"
    ):
        blockers.append("frozen split is not from pinned RLVR-IFeval train")
    heldout = manifest.get("heldout_policy") or {}
    if (
        heldout.get("dataset") != HELDOUT_DATASET_ID
        or heldout.get("accessed_during_preparation") is not False
        or heldout.get("materialized_during_preparation") is not False
    ):
        blockers.append("official IFBench held-out data entered the frozen split")
    for name, count, filename in (
        ("optimisation", OPTIMISATION_COUNT, "optimisation.jsonl"),
        ("validation", VALIDATION_COUNT, "validation.jsonl"),
    ):
        split = (manifest.get("splits") or {}).get(name) or {}
        path = split_dir / filename
        if (
            int(split.get("n", -1)) != count
            or split.get("file") != filename
            or not path.is_file()
            or path.is_symlink()
            or split.get("file_sha256") != sha256_file(path)
        ):
            blockers.append(f"{name} split/hash is not the frozen {count}-row artifact")
    return blockers, str(manifest_path)


def _audit_open_instruct(
    config: dict[str, Any], root: Path
) -> tuple[list[str], str | None]:
    checkout = _resolve(root, config["paths"]["open_instruct_root"])
    files = {
        checkout / "open_instruct" / "if_functions.py": IF_FUNCTIONS_SHA256,
        checkout / "open_instruct" / "ground_truth_utils.py": GROUND_TRUTH_UTILS_SHA256,
        checkout / "LICENSE": OPEN_INSTRUCT_LICENSE_SHA256,
    }
    blockers = []
    for path, expected_hash in files.items():
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != expected_hash
        ):
            blockers.append(f"missing/tampered pinned Open Instruct file: {path}")
    return blockers, str(checkout) if not blockers else None


def audit_readiness(
    config: dict[str, Any],
    *,
    root: Path,
    require_ready: bool = False,
) -> dict[str, Any]:
    validate_static_config(config, root=root)
    blockers, split_manifest = _audit_frozen_split(config, root)
    verifier_blockers, verifier_checkout = _audit_open_instruct(config, root)
    blockers.extend(verifier_blockers)
    verified: dict[str, Any] = {}
    if split_manifest is not None:
        verified["split_manifest"] = split_manifest
    if verifier_checkout is not None:
        verified["open_instruct"] = verifier_checkout

    gate_specs = config.get("gates") or {}
    for name in ("split", "reward_transform", "checkpoint_selector", "endpoint"):
        spec = gate_specs.get(name) or {}
        expected_hash = str(spec.get("file_sha256", ""))
        if not expected_hash or expected_hash == PENDING:
            blockers.append(f"{name} gate hash is not frozen")
            continue
        path = _resolve(root, str(spec.get("path", "")))
        if not path.is_file():
            blockers.append(f"missing {name} gate: {path}")
            continue
        if sha256_file(path) != expected_hash:
            blockers.append(f"{name} gate file hash mismatch")
            continue
        gate = _read_json(path)
        if name == "split":
            if (
                gate.get("outcome_use_allowed") is not True
                or gate.get("manifest_sha256") != config["dataset"]["manifest_sha256"]
                or (gate.get("source") or {}).get("dataset") != TRAIN_DATASET_ID
                or (gate.get("partition") or {}).get("optimisation_instances")
                != OPTIMISATION_COUNT
                or (gate.get("partition") or {}).get("validation_instances")
                != VALIDATION_COUNT
            ):
                blockers.append("split gate does not authorize the frozen IF split")
        elif name == "reward_transform":
            selected = str(gate.get("selected_reward_transform", ""))
            if gate.get("decision") != "GO" or selected not in REWARD_TRANSFORMS:
                blockers.append("reward-transform gate has no valid GO winner")
            elif config["reward"].get("value") != selected:
                blockers.append("config reward transform differs from frozen gate")
        elif name == "checkpoint_selector":
            selected = (gate.get("macro_gate") or {}).get("frozen_selector")
            if selected != "task":
                blockers.append("checkpoint selector gate is not frozen to task")
        elif name == "endpoint" and (
            gate.get("decision") != "GO"
            or gate.get("model") != JUDGE_MODEL
            or gate.get("provider") != JUDGE_PROVIDER
            or gate.get("allow_fallbacks") is not False
            or gate.get("logprobs") is not True
        ):
            blockers.append("endpoint gate does not prove exact StreamLake logprobs")
        verified[name] = str(path)

    report = {
        "protocol": config["protocol"],
        "ready": not blockers,
        "blockers": blockers,
        "verified": verified,
        "official_test_accessed": False,
        "planned_train_command": (
            "python -m iar_ifbench.train --config <config> --authorization-gate <gate>"
        ),
        "planned_validation_evaluator": "iar_ifbench.evaluate_validation",
    }
    if require_ready and blockers:
        raise RuntimeError("IFBench screen is not ready: " + "; ".join(blockers))
    return report


def verify_authorization_gate(path: Path, *, config_path: Path) -> dict[str, Any]:
    gate = _read_json(path.resolve())
    if (
        gate.get("decision") != "GO"
        or gate.get("protocol") != PROTOCOL
        or gate.get("config_sha256") != config_sha256(config_path.resolve())
        or gate.get("official_test_access_authorized") is not False
    ):
        raise RuntimeError(
            "IFBench run authorization gate is missing, stale, or over-broad"
        )
    return gate
