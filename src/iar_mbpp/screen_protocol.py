from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from iar_grpo.config import GrpoTrialConfig, validate_grpo_config

from .data import (
    DATASET_REVISION,
    TRAIN_COUNT,
    VALIDATION_COUNT,
    read_jsonl,
    sha256_file,
    sha256_json,
)
from .data import PROTOCOL as DATA_PROTOCOL

PROTOCOL = "mbpp-five-domain-screen-1p5b-v1"
ACTOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ACTOR_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
JUDGE_MODEL = "deepseek/deepseek-v4-flash"
JUDGE_PROVIDER = "StreamLake"
CHECKPOINTS = [0, 187, 374, 561, 748]
PENDING = "PENDING_GATE"
REWARD_TRANSFORMS = {
    "absolute_logp_1",
    "log_p_own",
    "p_own",
    "logit_p_own",
    "binary_own",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_static_config(config: dict[str, Any], *, root: Path) -> None:
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected MBPP screen protocol")
    actor = config["actor"]
    if (
        actor.get("model_id") != ACTOR_MODEL
        or actor.get("revision") != ACTOR_REVISION
        or actor.get("local_files_only") is not True
    ):
        raise ValueError("MBPP screen actor is not the frozen Qwen snapshot")
    dataset = config["dataset"]
    if (
        dataset.get("revision") != DATASET_REVISION
        or int(dataset.get("optimisation_count", -1)) != TRAIN_COUNT
        or int(dataset.get("validation_count", -1)) != VALIDATION_COUNT
        or dataset.get("source_file") != "optimisation.jsonl"
        or dataset.get("validation_file") != "validation.jsonl"
    ):
        raise ValueError("MBPP canonical split contract changed")
    expected_calls = TRAIN_COUNT * 4 * 2
    if (
        int(config["trial"].get("prompt_count", -1)) != TRAIN_COUNT
        or int(config["trial"].get("expected_reward_calls", -1)) != expected_calls
        or int(config["reward"].get("max_reward_calls", -1)) != expected_calls
    ):
        raise ValueError("MBPP reward-call plan changed")
    if (
        int(config["training"].get("num_generations", -1)) != 4
        or int(config["training"].get("num_epochs", -1)) != 2
        or int(config["training"].get("max_steps", -1)) != 748
        or int(config["training"].get("save_steps", -1)) != 187
        or list(config["evaluation"].get("checkpoints") or []) != CHECKPOINTS
        or config["evaluation"].get("validation_only") is not True
        or config["evaluation"].get("select_on") != "validation_task_accuracy"
        or config["evaluation"].get("characterize_reward") is not True
        or int(config["evaluation"].get("expected_reward_calls", -1))
        != VALIDATION_COUNT * len(CHECKPOINTS)
    ):
        raise ValueError("MBPP training or validation schedule changed")
    judge = config["discriminator"]
    routing = judge.get("provider_routing") or {}
    if (
        judge.get("model_id") != JUDGE_MODEL
        or judge.get("expected_provider") != JUDGE_PROVIDER
        or routing.get("order") != [JUDGE_PROVIDER]
        or routing.get("allow_fallbacks") is not False
        or routing.get("require_parameters") is not True
        or judge.get("strict_logprobs") is not True
    ):
        raise ValueError("MBPP judge must be exact StreamLake with strict logprobs")
    attempts = int(config["reward"].get("max_attempts_per_call", -1))
    if attempts != 3 or int(config["reward"].get("max_http_requests", -1)) != (
        expected_calls * attempts
    ):
        raise ValueError("MBPP HTTP budget must allow at most three attempts")
    serialized = json.dumps(config, sort_keys=True).lower()
    for forbidden in ("test_file", "official_test_path", "pilot_subset", "final_test"):
        if forbidden in serialized:
            raise ValueError(f"Forbidden pre-selection test field: {forbidden}")
    scratch = resolve(root, config["paths"]["scratch_root"])
    for key in ("split_dir", "data_cache", "experiments", "checkpoints", "tmp"):
        try:
            resolve(root, config["paths"][key]).relative_to(scratch)
        except ValueError as exc:
            raise ValueError(f"{key} is not under the scratch root") from exc


def load_frozen_split(
    config: dict[str, Any], *, root: Path, role: str
) -> list[dict[str, Any]]:
    if role not in {"optimisation", "validation"}:
        raise ValueError(
            "Only MBPP optimisation/validation may be read before selection"
        )
    split_dir = resolve(root, config["paths"]["split_dir"])
    manifest = _read_json(split_dir / "split_manifest.json")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    if (
        manifest.get("protocol") != DATA_PROTOCOL
        or manifest.get("manifest_sha256") != sha256_json(unhashed)
        or manifest.get("manifest_sha256") != config["dataset"]["manifest_sha256"]
        or (manifest.get("heldout_policy") or {}).get(
            "official_test_accessed_during_preparation"
        )
        is not False
    ):
        raise RuntimeError("MBPP frozen split manifest is invalid")
    spec = (manifest.get("splits") or {}).get(role) or {}
    path = split_dir / str(spec.get("file", ""))
    rows = read_jsonl(path)
    expected = TRAIN_COUNT if role == "optimisation" else VALIDATION_COUNT
    if (
        len(rows) != expected
        or spec.get("n") != expected
        or spec.get("file_sha256") != sha256_file(path)
        or spec.get("rows_sha256") != sha256_json(rows)
        or spec.get("ids_sha256") != sha256_json([row["id"] for row in rows])
    ):
        raise RuntimeError(f"MBPP frozen {role} content/hash mismatch")
    return rows


def audit_readiness(
    config: dict[str, Any], *, root: Path, require_ready: bool = False
) -> dict[str, Any]:
    validate_static_config(config, root=root)
    blockers: list[str] = []
    try:
        load_frozen_split(config, root=root, role="optimisation")
        load_frozen_split(config, root=root, role="validation")
    except Exception as exc:  # noqa: BLE001 - dry-run reports all blockers
        blockers.append(str(exc))
    for name in ("reward_transform", "checkpoint_selector", "endpoint"):
        spec = (config.get("gates") or {}).get(name) or {}
        expected_hash = str(spec.get("file_sha256", ""))
        if not expected_hash or expected_hash == PENDING:
            blockers.append(f"{name} gate hash is not frozen")
            continue
        path = resolve(root, str(spec.get("path", "")))
        if not path.is_file() or sha256_file(path) != expected_hash:
            blockers.append(f"{name} gate is missing or hash-mismatched")
            continue
        gate = _read_json(path)
        if name == "reward_transform":
            selected = str(gate.get("selected_reward_transform", ""))
            if (
                gate.get("decision") != "GO"
                or selected not in REWARD_TRANSFORMS
                or config["reward"].get("value") != selected
            ):
                blockers.append("reward-transform gate/config disagreement")
        elif name == "checkpoint_selector":
            if (gate.get("macro_gate") or {}).get("frozen_selector") != "task":
                blockers.append("checkpoint selector is not task utility")
        elif (
            gate.get("decision") != "GO"
            or gate.get("model") != JUDGE_MODEL
            or gate.get("provider") != JUDGE_PROVIDER
            or gate.get("allow_fallbacks") is not False
            or gate.get("logprobs") is not True
        ):
            blockers.append("endpoint gate does not prove exact StreamLake logprobs")
    if not blockers and config["reward"].get("value") != PENDING:
        validate_grpo_config(GrpoTrialConfig(raw=config, path=Path("screen.yaml")))
    report = {
        "protocol": PROTOCOL,
        "ready": not blockers,
        "blockers": blockers,
        "official_test_accessed": False,
        "training_or_api_started": False,
    }
    if require_ready and blockers:
        raise RuntimeError("MBPP screen is not ready: " + "; ".join(blockers))
    return report


def verify_authorization_gate(path: Path, *, config_path: Path) -> dict[str, Any]:
    gate = _read_json(path.resolve())
    if (
        gate.get("decision") != "GO"
        or gate.get("protocol") != PROTOCOL
        or gate.get("config_sha256") != config_sha256(config_path.resolve())
        or gate.get("official_test_access_authorized") is not False
    ):
        raise RuntimeError("MBPP authorization gate is missing, stale, or over-broad")
    return gate
