from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .data import (
    DATASET_ID,
    DATASET_REVISION,
    SCREEN_OPTIMISATION,
    SCREEN_PROTOCOL,
    SCREEN_VALIDATION,
    read_jsonl,
    sha256_json,
)
from .prompts import ATTRIBUTION_PROMPT_VERSION, TASK_PROMPT_VERSION

PROTOCOL = "gsm8k-five-domain-screen-1p5b-v1"
ACTOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ACTOR_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
JUDGE_MODEL = "deepseek/deepseek-v4-flash"
JUDGE_PROVIDER = "StreamLake"
CHECKPOINTS = [0, 3363, 6726, 10089, 13452]
PENDING = "PENDING_GATE"
REWARD_TRANSFORMS = {
    "absolute_logp_1",
    "log_p_own",
    "p_own",
    "logit_p_own",
    "binary_own",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_sha256(path: Path) -> str:
    return sha256_file(path.resolve())


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def validate_static_config(config: dict[str, Any], *, root: Path) -> None:
    if config.get("protocol") != PROTOCOL:
        raise ValueError("Unexpected GSM8K screen protocol")
    actor = config["actor"]
    if (
        actor.get("model_id") != ACTOR_MODEL
        or actor.get("revision") != ACTOR_REVISION
        or actor.get("local_files_only") is not True
    ):
        raise ValueError("GSM8K actor snapshot is not frozen")
    dataset = config["dataset"]
    if (
        dataset.get("dataset_id") != DATASET_ID
        or dataset.get("revision") != DATASET_REVISION
        or int(dataset.get("optimisation_count", -1)) != SCREEN_OPTIMISATION
        or int(dataset.get("validation_count", -1)) != SCREEN_VALIDATION
        or dataset.get("source_file") != "optimisation.jsonl"
        or dataset.get("validation_file") != "validation.jsonl"
        or dataset.get("task_prompt_version") != TASK_PROMPT_VERSION
    ):
        raise ValueError("GSM8K dataset/split/prompt contract changed")
    expected_calls = SCREEN_OPTIMISATION * 4 * 2
    training = config["training"]
    evaluation = config["evaluation"]
    if (
        int(config["trial"].get("prompt_count", -1)) != SCREEN_OPTIMISATION
        or int(config["trial"].get("expected_reward_calls", -1)) != expected_calls
        or int(config["reward"].get("max_reward_calls", -1)) != expected_calls
        or int(training.get("num_generations", -1)) != 4
        or int(training.get("num_epochs", -1)) != 2
        or int(training.get("generation_batch_size", -1)) != 8
        or int(training.get("max_steps", -1)) != CHECKPOINTS[-1]
        or int(training.get("save_steps", -1)) != CHECKPOINTS[1]
        or list(evaluation.get("checkpoints") or []) != CHECKPOINTS
        or evaluation.get("validation_only") is not True
        or evaluation.get("select_on") != "task_utility"
        or evaluation.get("task_utility") != "exact_answer_accuracy"
        or evaluation.get("characterize_reward") is not True
        or int(evaluation.get("expected_reward_calls", -1))
        != SCREEN_VALIDATION * len(CHECKPOINTS)
    ):
        raise ValueError("GSM8K training/evaluation schedule changed")
    judge = config["discriminator"]
    routing = judge.get("provider_routing") or {}
    if (
        judge.get("model_id") != JUDGE_MODEL
        or judge.get("expected_provider") != JUDGE_PROVIDER
        or judge.get("attribution_prompt_version") != ATTRIBUTION_PROMPT_VERSION
        or routing.get("order") != [JUDGE_PROVIDER]
        or routing.get("require_parameters") is not True
        or routing.get("allow_fallbacks") is not False
        or judge.get("strict_logprobs") is not True
    ):
        raise ValueError("GSM8K judge must be exact StreamLake with strict logprobs")
    attempts = int(config["reward"].get("max_attempts_per_call", -1))
    if attempts != 3 or int(config["reward"].get("max_http_requests", -1)) != (
        expected_calls * attempts
    ):
        raise ValueError("GSM8K HTTP budget must permit exactly three attempts")
    serialized = json.dumps(config, sort_keys=True).lower()
    for forbidden in ("test_file", "test_path", "official_test", "final_test"):
        if forbidden in serialized:
            raise ValueError(f"Forbidden pre-selection test field: {forbidden}")
    scratch = resolve(root, config["paths"]["scratch_root"])
    for key in ("split_dir", "data_cache", "experiments", "checkpoints", "tmp"):
        try:
            resolve(root, config["paths"][key]).relative_to(scratch)
        except ValueError as exc:
            raise ValueError(f"{key} is not under scratch_root") from exc


def load_frozen_split(
    config: dict[str, Any], *, root: Path, role: str
) -> list[dict[str, Any]]:
    if role not in {"optimisation", "validation"}:
        raise ValueError("Only GSM8K optimisation/validation may be loaded")
    split_dir = resolve(root, config["paths"]["split_dir"])
    manifest = _read_object(split_dir / "split_manifest.json")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    if (
        manifest.get("protocol") != SCREEN_PROTOCOL
        or manifest.get("manifest_sha256") != sha256_json(unhashed)
        or manifest.get("manifest_sha256") != config["dataset"]["manifest_sha256"]
        or (manifest.get("heldout_policy") or {}).get(
            "official_test_accessed_during_preparation"
        )
        is not False
    ):
        raise RuntimeError("GSM8K frozen split manifest is invalid")
    spec = (manifest.get("splits") or {}).get(role) or {}
    path = split_dir / str(spec.get("file", ""))
    rows = read_jsonl(path)
    expected = SCREEN_OPTIMISATION if role == "optimisation" else SCREEN_VALIDATION
    ids = [str(row.get("id", "")) for row in rows]
    if (
        len(rows) != expected
        or int(spec.get("n", -1)) != expected
        or not path.is_file()
        or path.is_symlink()
        or spec.get("file_sha256") != sha256_file(path)
        or spec.get("rows_sha256") != sha256_json(rows)
        or spec.get("ids_sha256") != sha256_json(ids)
        or not all(ids)
        or len(ids) != len(set(ids))
        or any(row.get("source_split") != "train" for row in rows)
    ):
        raise RuntimeError(f"GSM8K frozen {role} content/hash mismatch")
    return rows


def audit_readiness(
    config: dict[str, Any], *, root: Path, require_ready: bool = False
) -> dict[str, Any]:
    validate_static_config(config, root=root)
    blockers: list[str] = []
    try:
        load_frozen_split(config, root=root, role="optimisation")
        load_frozen_split(config, root=root, role="validation")
    except Exception as exc:  # noqa: BLE001 - report exact dry-run blocker
        blockers.append(str(exc))
    for name in ("reward_transform", "checkpoint_selector", "endpoint"):
        spec = (config.get("gates") or {}).get(name) or {}
        expected = str(spec.get("file_sha256", ""))
        if not expected or expected == PENDING:
            blockers.append(f"{name} gate hash is not frozen")
            continue
        path = resolve(root, str(spec.get("path", "")))
        if not path.is_file() or sha256_file(path) != expected:
            blockers.append(f"{name} gate is missing or hash-mismatched")
            continue
        gate = _read_object(path)
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
    result = {
        "protocol": PROTOCOL,
        "ready": not blockers,
        "blockers": blockers,
        "official_test_accessed": False,
        "training_or_api_started": False,
    }
    if require_ready and blockers:
        raise RuntimeError("GSM8K screen is not ready: " + "; ".join(blockers))
    return result


def verify_authorization_gate(path: Path, *, config_path: Path) -> dict[str, Any]:
    gate = _read_object(path.resolve())
    if (
        gate.get("decision") != "GO"
        or gate.get("protocol") != PROTOCOL
        or gate.get("config_sha256") != config_sha256(config_path.resolve())
        or gate.get("official_test_access_authorized") is not False
    ):
        raise RuntimeError("GSM8K authorization gate is missing, stale, or over-broad")
    return gate
