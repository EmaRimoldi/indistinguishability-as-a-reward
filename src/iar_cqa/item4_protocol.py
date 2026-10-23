from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .data import SCREEN_PROTOCOL as DATA_PROTOCOL
from .data import read_jsonl, sha256_json
from .prompts import ATTRIBUTION_PROMPT_VERSION, TASK_PROMPT_VERSION

PROTOCOL = "cqa-item4-full-1p5b-v1"
ACTOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ACTOR_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
JUDGE_MODEL = "deepseek/deepseek-v4-flash"
JUDGE_PROVIDER = "StreamLake"
OPTIMISATION_COUNT = 8_768
VALIDATION_COUNT = 973
CHECKPOINTS = [0, 4_384, 8_768, 13_152, 17_536]
PENDING = "PENDING_GATE"
REWARD_TRANSFORMS = {
    "absolute_logp_1",
    "log_p_own",
    "p_own",
    "logit_p_own",
    "binary_own",
}


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_sha256(path: Path) -> str:
    return sha256_file(path.resolve())


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _under(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} is not under scratch_root: {path}") from exc


def _keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [str(key) for key in value] + [
            key for item in value.values() for key in _keys(item)
        ]
    if isinstance(value, list):
        return [key for item in value for key in _keys(item)]
    return []


def validate_static_config(config: dict[str, Any], *, root: Path) -> None:
    if (
        config.get("protocol") != PROTOCOL
        or config.get("status") != "PREPARED_NOT_AUTHORIZED"
    ):
        raise ValueError("Unexpected or prematurely authorized CQA Item 4 protocol")
    actor = config["actor"]
    if (
        actor.get("model_id") != ACTOR_MODEL
        or actor.get("revision") != ACTOR_REVISION
        or actor.get("local_files_only") is not True
    ):
        raise ValueError("Actor model/revision/offline policy changed")
    dataset = config["dataset"]
    if (
        int(dataset.get("optimisation_count", -1)) != OPTIMISATION_COUNT
        or int(dataset.get("validation_count", -1)) != VALIDATION_COUNT
        or dataset.get("optimisation_file") != "optimisation.jsonl"
        or dataset.get("validation_file") != "validation.jsonl"
        or dataset.get("task_prompt_version") != TASK_PROMPT_VERSION
    ):
        raise ValueError("CQA pool sizes or task prompt changed")
    training = config["training"]
    if (
        int(training.get("num_generations", -1)) != 4
        or int(training.get("num_epochs", -1)) != 2
        or int(training.get("generation_batch_size", -1)) != 8
        or int(training.get("max_steps", -1)) != CHECKPOINTS[-1]
        or int(training.get("save_steps", -1)) != CHECKPOINTS[1]
    ):
        raise ValueError("CQA Item 4 GRPO schedule changed")
    expected_calls = OPTIMISATION_COUNT * 4 * 2
    unique_prompts_per_generation_batch = int(
        training.get("generation_batch_size", -1)
    ) // int(training.get("num_generations", -1))
    if (
        int(config["trial"].get("prompt_count", -1)) != OPTIMISATION_COUNT
        or int(config["trial"].get("expected_reward_calls", -1)) != expected_calls
        or OPTIMISATION_COUNT % unique_prompts_per_generation_batch
    ):
        raise ValueError("CQA exact two-epoch rollout plan changed")
    if list(config["evaluation"].get("checkpoints") or []) != CHECKPOINTS:
        raise ValueError("CQA Item 4 checkpoint schedule changed")
    if (
        config["evaluation"].get("validation_only") is not True
        or config["evaluation"].get("select_on") != "task_utility"
        or config["evaluation"].get("task_utility") != "exact_answer_accuracy"
        or config["evaluation"].get("characterize_reward") is not True
        or int(config["evaluation"].get("expected_reward_calls", -1))
        != VALIDATION_COUNT * len(CHECKPOINTS)
    ):
        raise ValueError("Checkpoint selection must be validation task accuracy")
    reward = config["reward"]
    if (
        int(reward.get("max_reward_calls", -1)) != expected_calls
        or int(reward.get("max_attempts_per_call", -1)) != 3
        or int(reward.get("max_http_requests", -1)) != expected_calls * 3
    ):
        raise ValueError("CQA reward/HTTP budgets changed")
    discriminator = config["discriminator"]
    route = discriminator.get("provider_routing") or {}
    if (
        discriminator.get("model_id") != JUDGE_MODEL
        or discriminator.get("expected_provider") != JUDGE_PROVIDER
        or discriminator.get("attribution_prompt_version") != ATTRIBUTION_PROMPT_VERSION
        or discriminator.get("strict_logprobs") is not True
        or route.get("order") != [JUDGE_PROVIDER]
        or route.get("require_parameters") is not True
        or route.get("allow_fallbacks") is not False
    ):
        raise ValueError("Judge must be exact StreamLake/no-fallback/strict-logprobs")
    forbidden = {
        "test_file",
        "test_path",
        "official_test",
        "official_test_path",
        "official_validation_file",
    }
    present = forbidden.intersection(_keys(config))
    if present:
        raise ValueError(f"Held-out access fields are forbidden: {sorted(present)}")
    scratch = resolve(root, config["paths"]["scratch_root"])
    for key in ("split_dir", "hf_cache", "experiments", "checkpoints", "tmp"):
        _under(resolve(root, config["paths"][key]), scratch, key)


def _verify_manifest(
    config: dict[str, Any], root: Path
) -> tuple[Path | None, list[str]]:
    blockers: list[str] = []
    split_dir = resolve(root, config["paths"]["split_dir"])
    manifest_path = split_dir / "split_manifest.json"
    expected = str(config["dataset"].get("split_manifest_sha256", ""))
    if not expected or expected == PENDING:
        blockers.append("split manifest hash is not frozen")
        return None, blockers
    if not manifest_path.is_file():
        blockers.append(f"missing split manifest: {manifest_path}")
        return None, blockers
    manifest = _read_object(manifest_path)
    stored = manifest.get("manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    if stored != sha256_json(unhashed) or stored != expected:
        blockers.append("split manifest content hash mismatch")
    if manifest.get("protocol") != DATA_PROTOCOL:
        blockers.append("unexpected CQA split protocol")
    heldout = manifest.get("heldout_policy") or {}
    if (
        heldout.get("official_validation_accessed_during_preparation") is not False
        or heldout.get("official_validation_materialized_during_preparation")
        is not False
    ):
        blockers.append("held-out CQA validation entered split preparation")
    splits = manifest.get("splits") or {}
    for role, manifest_name, count in (
        ("optimisation", "optimisation", OPTIMISATION_COUNT),
        ("validation", "validation", VALIDATION_COUNT),
    ):
        summary = splits.get(manifest_name) or {}
        configured_name = config["dataset"].get(f"{role}_file")
        path = split_dir / str(configured_name or "")
        if int(summary.get("n", -1)) != count:
            blockers.append(f"frozen {role} pool is not {count}")
            continue
        if (
            summary.get("file") != configured_name
            or not path.is_file()
            or path.is_symlink()
        ):
            blockers.append(f"missing or unsafe frozen {role} file")
            continue
        try:
            rows = read_jsonl(path)
            ids = [str(row.get("id", "")) for row in rows]
            if (
                len(rows) != count
                or not all(ids)
                or len(ids) != len(set(ids))
                or summary.get("ids_sha256") != sha256_json(ids)
                or summary.get("rows_sha256") != sha256_json(rows)
                or any(row.get("source_split") != "train" for row in rows)
            ):
                blockers.append(f"frozen {role} content/hash mismatch")
        except Exception as exc:  # noqa: BLE001 - include exact preflight blocker
            blockers.append(f"cannot audit frozen {role}: {exc}")
    return manifest_path, blockers


def _verify_gate(
    config: dict[str, Any], root: Path, name: str
) -> tuple[dict[str, Any] | None, list[str]]:
    spec = (config.get("gates") or {}).get(name) or {}
    expected = str(spec.get("file_sha256", ""))
    if not expected or expected == PENDING:
        return None, [f"{name} gate hash is not frozen"]
    path = resolve(root, str(spec.get("path", "")))
    if not path.is_file():
        return None, [f"missing {name} gate: {path}"]
    if sha256_file(path) != expected:
        return None, [f"{name} gate file hash mismatch"]
    return _read_object(path), []


def audit_readiness(
    config: dict[str, Any],
    *,
    root: Path,
    config_path: Path,
    authorization_gate: Path | None = None,
    require_ready: bool = False,
) -> dict[str, Any]:
    validate_static_config(config, root=root)
    _, blockers = _verify_manifest(config, root)
    reward_gate, errors = _verify_gate(config, root, "reward_transform")
    blockers.extend(errors)
    if reward_gate is not None:
        selected = str(reward_gate.get("selected_reward_transform", ""))
        if reward_gate.get("decision") != "GO" or selected not in REWARD_TRANSFORMS:
            blockers.append("reward-transform gate has no valid GO winner")
        elif config["reward"].get("value") != selected:
            blockers.append("configured reward differs from the frozen winner")
    checkpoint_gate, errors = _verify_gate(config, root, "checkpoint_selector")
    blockers.extend(errors)
    if checkpoint_gate is not None and (
        (checkpoint_gate.get("macro_gate") or {}).get("frozen_selector") != "task"
    ):
        blockers.append("checkpoint selector is not task utility")
    endpoint_gate, errors = _verify_gate(config, root, "endpoint")
    blockers.extend(errors)
    if endpoint_gate is not None and (
        endpoint_gate.get("decision") != "GO"
        or endpoint_gate.get("model") != JUDGE_MODEL
        or endpoint_gate.get("provider") != JUDGE_PROVIDER
        or endpoint_gate.get("allow_fallbacks") is not False
        or endpoint_gate.get("logprobs") is not True
    ):
        blockers.append("endpoint gate does not prove exact StreamLake logprobs")
    if authorization_gate is None:
        blockers.append("config authorization gate was not supplied")
    elif not authorization_gate.is_file():
        blockers.append(f"missing config authorization gate: {authorization_gate}")
    else:
        gate = _read_object(authorization_gate.resolve())
        if (
            gate.get("decision") != "GO"
            or gate.get("protocol") != PROTOCOL
            or gate.get("config_sha256") != config_sha256(config_path)
            or gate.get("official_test_access_authorized") is not False
        ):
            blockers.append("config authorization gate is stale or over-broad")
    report = {
        "protocol": PROTOCOL,
        "ready": not blockers,
        "blockers": blockers,
        "training_or_api_started": False,
        "official_test_accessed": False,
    }
    if require_ready and blockers:
        raise RuntimeError("CQA Item 4 is not ready: " + "; ".join(blockers))
    return report


def load_frozen_split(
    config: dict[str, Any], root: Path, split: str
) -> list[dict[str, Any]]:
    if split not in {"optimisation", "validation"}:
        raise ValueError("Only optimisation and validation may be loaded")
    manifest_path, blockers = _verify_manifest(config, root)
    if blockers or manifest_path is None:
        raise RuntimeError("Cannot load unfrozen CQA pool: " + "; ".join(blockers))
    manifest = _read_object(manifest_path)
    manifest_name = split
    configured_name = config["dataset"][f"{split}_file"]
    summary = (manifest.get("splits") or {}).get(manifest_name) or {}
    if summary.get("file") != configured_name:
        raise RuntimeError(f"Unexpected file for frozen {split} split")
    path = manifest_path.parent / configured_name
    rows = read_jsonl(path)
    expected_n = OPTIMISATION_COUNT if split == "optimisation" else VALIDATION_COUNT
    ids = [str(row.get("id", "")) for row in rows]
    if (
        len(rows) != expected_n
        or int(summary.get("n", -1)) != expected_n
        or not all(ids)
        or len(ids) != len(set(ids))
        or summary.get("ids_sha256") != sha256_json(ids)
        or summary.get("rows_sha256") != sha256_json(rows)
        or any(row.get("source_split") != "train" for row in rows)
    ):
        raise RuntimeError(f"Frozen {split} rows failed integrity checks")
    return rows


def exact_actor_snapshot(config: dict[str, Any], root: Path) -> Path:
    cache = resolve(root, config["paths"]["hf_cache"])
    slug = f"models--{ACTOR_MODEL.replace('/', '--')}"
    candidates = [
        cache / slug / "snapshots" / ACTOR_REVISION,
        cache / "hub" / slug / "snapshots" / ACTOR_REVISION,
    ]
    matches = [path.resolve() for path in candidates if path.is_dir()]
    if len(matches) != 1 or matches[0].name != ACTOR_REVISION:
        raise FileNotFoundError(
            f"Exact offline Actor snapshot {ACTOR_REVISION} is unavailable"
        )
    return matches[0]
