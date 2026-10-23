from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .data import DATASET_REVISION, sha256_file
from .prompts import ATTRIBUTION_PROMPT_VERSION, TASK_PROMPT_VERSION

ACTOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ACTOR_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
JUDGE_MODEL = "deepseek/deepseek-v4-flash"
JUDGE_PROVIDER = "StreamLake"
OPTIMISATION_COUNT = 6_300
VALIDATION_COUNT = 700
CHECKPOINTS = [0, 1575, 3150, 4725, 6300, 7875, 9450, 11025, 12600]
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
    if config.get("protocol") != "spider-five-domain-screen-1p5b-v1":
        raise ValueError("Unexpected Spider screen protocol")
    if (
        config["actor"]["model_id"] != ACTOR_MODEL
        or config["actor"].get("revision") != ACTOR_REVISION
    ):
        raise ValueError("The five-domain screen is frozen to Qwen2.5-1.5B-Instruct")
    dataset = config["dataset"]
    if (
        dataset.get("revision") != DATASET_REVISION
        or int(dataset.get("optimisation_count", -1)) != OPTIMISATION_COUNT
        or int(dataset.get("validation_count", -1)) != VALIDATION_COUNT
        or dataset.get("actor_prompt_version") != TASK_PROMPT_VERSION
    ):
        raise ValueError("Spider dataset/split/prompt contract changed")
    training = config["training"]
    if (
        int(training.get("num_generations", -1)) != 4
        or int(training.get("num_epochs", -1)) != 2
        or int(training.get("max_steps", -1)) != CHECKPOINTS[-1]
        or int(training.get("save_steps", -1)) != CHECKPOINTS[1]
        or list(config["evaluation"].get("checkpoints") or []) != CHECKPOINTS
    ):
        raise ValueError("Spider G/epoch/checkpoint schedule changed")
    expected = OPTIMISATION_COUNT * 4 * 2
    if (
        int(config["trial"].get("expected_reward_calls", -1)) != expected
        or int(config["reward"].get("max_reward_calls", -1)) != expected
    ):
        raise ValueError("Reward-call budget is not 6300 * G=4 * 2 epochs")
    attempts = int(config["reward"].get("max_attempts_per_call", -1))
    if (
        attempts != 3
        or int(config["reward"].get("max_http_requests", -1)) != expected * attempts
    ):
        raise ValueError(
            "HTTP budget must allow at most three attempts per logical call"
        )
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
    if config["evaluation"].get("validation_only") is not True:
        raise ValueError("Spider screen evaluation must be validation-only")
    if config["evaluation"].get("entrypoint") != "iar_spider.evaluate_validation":
        raise ValueError("Unexpected Spider validation evaluator")
    if (
        config["evaluation"].get("select_on") != "task_utility"
        or config["evaluation"].get("characterize_reward") is not True
        or int(config["evaluation"].get("expected_reward_calls", -1))
        != VALIDATION_COUNT * len(CHECKPOINTS)
    ):
        raise ValueError("Checkpoint selector must use frozen task utility")
    serialized = json.dumps(config, sort_keys=True).lower()
    for forbidden in (
        "test_file",
        "dev_file",
        "official_test_path",
        "official_dev_path",
    ):
        if forbidden in serialized:
            raise ValueError(f"Forbidden held-out field in screen config: {forbidden}")
    scratch = _resolve(root, config["paths"]["scratch_root"])
    for key in (
        "split_dir",
        "tables_json",
        "database_root",
        "hf_cache",
        "experiments",
        "checkpoints",
        "tmp",
    ):
        _under(_resolve(root, config["paths"][key]), scratch, key)


def audit_readiness(
    config: dict[str, Any],
    *,
    root: Path,
    require_ready: bool = False,
) -> dict[str, Any]:
    validate_static_config(config, root=root)
    blockers: list[str] = []
    verified: dict[str, Any] = {}
    split_dir = _resolve(root, config["paths"]["split_dir"])
    tables_json = _resolve(root, config["paths"]["tables_json"])
    database_root = _resolve(root, config["paths"]["database_root"])
    if not tables_json.is_file():
        blockers.append(f"missing frozen Spider tables.json: {tables_json}")
    else:
        verified["tables_json"] = str(tables_json)
    if not database_root.is_dir():
        blockers.append(f"missing frozen Spider database root: {database_root}")
    else:
        verified["database_root"] = str(database_root)
    manifest_path = split_dir / "split_manifest.json"
    if not manifest_path.is_file():
        blockers.append(f"missing frozen split manifest: {manifest_path}")
    else:
        manifest = _read_json(manifest_path)
        expected_manifest = str(config["dataset"]["manifest_sha256"])
        if manifest.get("manifest_sha256") != expected_manifest:
            blockers.append("frozen split manifest_sha256 mismatch")
        split = manifest.get("splits") or {}
        if int((split.get("optimisation") or {}).get("n", -1)) != OPTIMISATION_COUNT:
            blockers.append("optimisation split is not exactly 6300")
        if int((split.get("validation") or {}).get("n", -1)) != VALIDATION_COUNT:
            blockers.append("validation split is not exactly 700")
        heldout = manifest.get("heldout_policy") or {}
        if heldout.get("accessed_during_preparation") is not False:
            blockers.append("official held-out split was accessed")
        verified["split_manifest"] = str(manifest_path)

    gate_specs = config.get("gates") or {}
    for name in (
        "split",
        "evaluator",
        "reward_transform",
        "checkpoint_selector",
        "endpoint",
    ):
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
            source = gate.get("source") or {}
            partition = gate.get("partition") or {}
            heldout = gate.get("heldout") or {}
            if (
                gate.get("outcome_use_allowed") is not True
                or gate.get("manifest_sha256")
                != config["dataset"]["manifest_sha256"]
                or gate.get("remote_output_dir") != str(split_dir)
                or source.get("dataset") != config["dataset"]["dataset_id"]
                or source.get("revision") != config["dataset"]["revision"]
                or partition.get("optimisation_instances") != OPTIMISATION_COUNT
                or partition.get("validation_instances") != VALIDATION_COUNT
                or heldout.get("accessed_during_preparation") is not False
            ):
                blockers.append("split gate does not authorize the frozen Spider split")
        elif name == "evaluator":
            if (
                gate.get("slurm_state") != "COMPLETED"
                or gate.get("validation_only") is not True
                or gate.get("official_dev_or_test_read") is not False
                or gate.get("split_protocol_manifest_sha256")
                != config["dataset"]["manifest_sha256"]
                or gate.get("instances") != VALIDATION_COUNT
                or gate.get("accuracy") != 1.0
            ):
                blockers.append("evaluator gate does not prove the frozen Spider metric")
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
        elif name == "endpoint":
            if (
                gate.get("decision") != "GO"
                or gate.get("model") != JUDGE_MODEL
                or gate.get("provider") != JUDGE_PROVIDER
                or gate.get("allow_fallbacks") is not False
                or gate.get("logprobs") is not True
            ):
                blockers.append(
                    "endpoint gate does not prove exact StreamLake logprobs"
                )
        verified[name] = str(path)

    report = {
        "protocol": config["protocol"],
        "ready": not blockers,
        "blockers": blockers,
        "verified": verified,
        "official_test_accessed": False,
        "planned_train_command": "python -m iar_spider.train --config <config> --authorization-gate <gate>",
        "planned_validation_evaluator": "iar_spider.evaluate_validation",
    }
    if require_ready and blockers:
        raise RuntimeError("Spider screen is not ready: " + "; ".join(blockers))
    return report


def verify_authorization_gate(path: Path, *, config_path: Path) -> dict[str, Any]:
    gate = _read_json(path.resolve())
    if (
        gate.get("decision") != "GO"
        or gate.get("protocol") != "spider-five-domain-screen-1p5b-v1"
        or gate.get("config_sha256") != config_sha256(config_path.resolve())
        or gate.get("official_test_access_authorized") is not False
    ):
        raise RuntimeError("Run authorization gate is missing, stale, or over-broad")
    return gate
