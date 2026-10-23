from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from iar_distillation.teacher_data import (
    SCHEMA_VERSION,
    SCORER_VERSIONS,
    build_summary,
    export_sft,
    load_sources,
    read_jsonl,
    resolve,
    score_output,
    sha256_file,
    sha256_text,
    utc_now,
    verified_chat_completion,
    write_json,
    write_jsonl,
)
from iar_phase0.openrouter import OpenRouterClient, make_chat_payload, response_text


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def balanced_assignments(ids: list[str], shard_count: int) -> list[list[str]]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    shards = [[] for _ in range(shard_count)]
    for index, source_id in enumerate(ids):
        shards[index % shard_count].append(source_id)
    return shards


def load_named_secret(path: Path, name: str) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == name:
            secret = value.strip().strip('"').strip("'")
            if not secret:
                raise ValueError(f"Empty credential variable: {name}")
            return secret
    raise KeyError(f"Credential variable not found: {name}")


def load_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(config["project"]["root"]).resolve()
    return config, root


def validate_base_rows(
    responses: list[dict[str, Any]], source_rows: list[dict[str, Any]]
) -> None:
    sources_by_id = {str(row["id"]): row for row in source_rows}
    ids = [str(row.get("id")) for row in responses]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs in base responses")
    orders = [int(row["experiment_order"]) for row in responses]
    if orders != sorted(orders):
        raise ValueError("Base responses are not sorted")
    for row in responses:
        source_id = str(row["id"])
        if source_id not in sources_by_id:
            raise ValueError(f"Unknown base response ID: {source_id}")
        source = sources_by_id[source_id]
        if any(
            row.get(key) != source.get(key)
            for key in (
                "id",
                "task_id",
                "dataset_index",
                "experiment_order",
                "prompt",
                "prompt_version",
            )
        ):
            raise ValueError(f"Base response/source mismatch: {source_id}")
        if row.get("prompt_sha256") != sha256_text(str(source["prompt"])):
            raise ValueError(f"Base prompt hash mismatch: {source_id}")


def parallel_dir(root: Path, config: dict[str, Any], domain: str, name: str) -> Path:
    return resolve(root, str(config["paths"]["output_root"])) / domain / "parallel" / name


def prepare_plan(
    config_path: Path,
    domain: str,
    shard_count: int,
    name: str,
) -> Path:
    config, root = load_config(config_path)
    domain_config = dict(config["domains"][domain])
    source_rows = load_sources(root, domain, domain_config)
    main_dir = resolve(root, str(config["paths"]["output_root"])) / domain
    main_response_path = main_dir / "responses.jsonl"
    base_responses = read_jsonl(main_response_path)
    validate_base_rows(base_responses, source_rows)
    base_ids = {str(row["id"]) for row in base_responses}
    remaining_ids = [str(row["id"]) for row in source_rows if str(row["id"]) not in base_ids]
    assignments = balanced_assignments(remaining_ids, shard_count)
    run_dir = parallel_dir(root, config, domain, name)
    run_dir.mkdir(parents=True, exist_ok=True)
    plan_path = run_dir / "plan.json"
    snapshot_path = run_dir / "base_responses.jsonl"
    if plan_path.exists() or snapshot_path.exists():
        raise FileExistsError(f"Parallel plan already exists: {run_dir}")
    shutil.copy2(main_response_path, snapshot_path)
    plan = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "prepared",
        "name": name,
        "domain": domain,
        "expected_count": len(source_rows),
        "base_count": len(base_responses),
        "remaining_count": len(remaining_ids),
        "shard_count": shard_count,
        "assignments": assignments,
        "base_responses_file": str(snapshot_path),
        "base_responses_sha256": sha256_file(snapshot_path),
        "main_responses_file": str(main_response_path),
        "config_file": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "model_id": str(config["model"]["model_id"]),
        "expected_provider": str(config["model"]["expected_provider"]),
    }
    write_json(plan_path, plan)
    return plan_path


def generate_shard(
    config_path: Path,
    plan_path: Path,
    shard_index: int,
    key_env_file: Path,
    key_env_name: str,
) -> Path:
    config, root = load_config(config_path)
    plan = read_json(plan_path)
    domain = str(plan["domain"])
    shard_count = int(plan["shard_count"])
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"Invalid shard index {shard_index}/{shard_count}")
    if sha256_file(config_path) != plan["config_sha256"]:
        raise ValueError("Configuration drift after shard preparation")
    snapshot_path = Path(str(plan["base_responses_file"]))
    if sha256_file(snapshot_path) != plan["base_responses_sha256"]:
        raise ValueError("Base snapshot drift")
    source_rows = load_sources(root, domain, dict(config["domains"][domain]))
    sources_by_id = {str(row["id"]): row for row in source_rows}
    assigned_ids = [str(value) for value in plan["assignments"][shard_index]]
    selected = [sources_by_id[source_id] for source_id in assigned_ids]
    shard_dir = plan_path.parent / f"shard_{shard_index:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    selected_source_path = shard_dir / "source_prompts.jsonl"
    write_jsonl(
        selected_source_path,
        [
            {
                key: row[key]
                for key in (
                    "id",
                    "task_id",
                    "dataset_index",
                    "experiment_order",
                    "prompt",
                    "prompt_version",
                )
            }
            for row in selected
        ],
    )

    secret = load_named_secret(key_env_file, key_env_name)
    key_fingerprint = hashlib.sha256(secret.encode("utf-8")).hexdigest()[:16]
    os.environ["OPENROUTER_API_KEY"] = secret
    model = config["model"]
    transport = config["transport"]
    client = OpenRouterClient.from_env(
        max_attempts=int(transport["max_attempts"]),
        max_retry_wait_seconds=float(transport["max_retry_wait_seconds"]),
    )
    response_path = shard_dir / "responses.jsonl"
    responses = read_jsonl(response_path) if response_path.exists() else []
    completed = {str(row["id"]): row for row in responses}
    if len(completed) != len(responses):
        raise ValueError(f"Duplicate shard responses: {response_path}")
    if not set(completed).issubset(set(assigned_ids)):
        raise ValueError("Shard checkpoint contains an unassigned ID")

    domain_config = dict(config["domains"][domain])
    generation = {
        "temperature": float(model["temperature"]),
        "top_p": float(model["top_p"]),
        "max_tokens": int(domain_config["max_tokens"]),
        "reasoning_enabled": False,
    }
    manifest = {
        "schema_version": 1,
        "status": "running",
        "domain": domain,
        "plan_file": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "assigned_count": len(assigned_ids),
        "completed_rows": len(responses),
        "key_env_name": key_env_name,
        "key_fingerprint": key_fingerprint,
        "model_id": str(model["model_id"]),
        "expected_provider": str(model["expected_provider"]),
        "message_roles": ["user"],
        "system_prompt_present": False,
        "generation": generation,
        "scorer_version": SCORER_VERSIONS[domain],
        "started_at": utc_now(),
    }
    write_json(shard_dir / "manifest.json", manifest)

    progress = tqdm(
        total=len(selected),
        initial=len(responses),
        desc=f"DeepSeek teacher {domain} shard {shard_index}/{shard_count}",
    )
    for source in selected:
        source_id = str(source["id"])
        if source_id in completed:
            if completed[source_id].get("prompt_sha256") != sha256_text(source["prompt"]):
                raise ValueError(f"Prompt drift while resuming shard row {source_id}")
            continue
        payload = make_chat_payload(
            model_id=str(model["model_id"]),
            prompt=str(source["prompt"]),
            system_prompt=None,
            max_tokens=int(domain_config["max_tokens"]),
            temperature=float(model["temperature"]),
            provider_routing=dict(model["provider_routing"]),
            request_overrides={
                "top_p": float(model["top_p"]),
                "reasoning": {"enabled": False},
            },
        )
        started = time.time()
        raw = verified_chat_completion(
            client=client,
            payload=payload,
            source_id=source_id,
            expected_model=str(model["model_id"]),
            expected_provider=str(model["expected_provider"]),
            event_path=shard_dir / "transport_events.jsonl",
            max_attempts=int(transport["logical_max_attempts"]),
            retry_wait_seconds=float(transport["logical_retry_wait_seconds"]),
        )
        raw_text = response_text(raw)
        normalized, score = score_output(
            domain,
            raw_text,
            source,
            int(domain_config.get("unit_test_timeout_seconds", 5)),
        )
        choice = raw["choices"][0]
        row = {
            "schema_version": SCHEMA_VERSION,
            "id": source["id"],
            "task_id": source["task_id"],
            "domain": domain,
            "dataset_index": source["dataset_index"],
            "experiment_order": source["experiment_order"],
            "prompt": source["prompt"],
            "prompt_version": source["prompt_version"],
            "prompt_sha256": sha256_text(source["prompt"]),
            "message_roles": ["user"],
            "system_prompt_present": False,
            "source": "DeepSeek_teacher_parallel_shard",
            "requested_model": str(model["model_id"]),
            "raw_model": str(raw["model"]),
            "raw_provider": str(raw["provider"]),
            "raw_response": raw_text,
            "response": normalized,
            "response_sha256": sha256_text(normalized),
            "score": score,
            "scorer_version": SCORER_VERSIONS[domain],
            "finish_reason": choice.get("finish_reason"),
            "native_finish_reason": choice.get("native_finish_reason"),
            "usage": raw.get("usage", {}),
            "generation": generation,
            "latency_seconds": time.time() - started,
            "created_at": utc_now(),
            "parallel_shard": {
                "plan": str(plan_path),
                "index": shard_index,
                "count": shard_count,
                "key_fingerprint": key_fingerprint,
            },
        }
        responses.append(row)
        completed[source_id] = row
        responses.sort(key=lambda value: int(value["experiment_order"]))
        write_jsonl(response_path, responses)
        manifest["completed_rows"] = len(responses)
        manifest["updated_at"] = utc_now()
        write_json(shard_dir / "manifest.json", manifest)
        progress.update(1)
        time.sleep(float(transport["inter_request_delay_seconds"]))
    progress.close()
    export_sft(shard_dir, responses)
    summary = build_summary(domain, responses)
    write_json(shard_dir / "summary.json", summary)
    manifest.update(
        {
            "status": "complete",
            "completed_rows": len(responses),
            "responses_sha256": sha256_file(response_path),
            "completed_at": summary["completed_at"],
        }
    )
    write_json(shard_dir / "manifest.json", manifest)
    return shard_dir


def merge_shards(config_path: Path, plan_path: Path) -> Path:
    config, root = load_config(config_path)
    plan = read_json(plan_path)
    domain = str(plan["domain"])
    if sha256_file(config_path) != plan["config_sha256"]:
        raise ValueError("Configuration drift before merge")
    snapshot_path = Path(str(plan["base_responses_file"]))
    if sha256_file(snapshot_path) != plan["base_responses_sha256"]:
        raise ValueError("Base snapshot drift before merge")
    main_response_path = Path(str(plan["main_responses_file"]))
    if sha256_file(main_response_path) != plan["base_responses_sha256"]:
        raise ValueError("Main responses changed after shard preparation")
    merged = read_jsonl(snapshot_path)
    shard_audit: list[dict[str, Any]] = []
    for shard_index, raw_assigned in enumerate(plan["assignments"]):
        assigned = {str(value) for value in raw_assigned}
        shard_dir = plan_path.parent / f"shard_{shard_index:02d}"
        manifest = read_json(shard_dir / "manifest.json")
        rows = read_jsonl(shard_dir / "responses.jsonl")
        observed = {str(row["id"]) for row in rows}
        if manifest.get("status") != "complete":
            raise ValueError(f"Shard {shard_index} is not complete")
        if observed != assigned or len(rows) != len(assigned):
            raise ValueError(f"Shard {shard_index} assignment mismatch")
        if manifest.get("responses_sha256") != sha256_file(shard_dir / "responses.jsonl"):
            raise ValueError(f"Shard {shard_index} response hash mismatch")
        merged.extend(rows)
        shard_audit.append(
            {
                "shard_index": shard_index,
                "rows": len(rows),
                "key_fingerprint": manifest.get("key_fingerprint"),
                "responses_sha256": manifest.get("responses_sha256"),
            }
        )
    ids = [str(row["id"]) for row in merged]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs after shard merge")
    expected = int(plan["expected_count"])
    if len(merged) != expected:
        raise ValueError(f"Merged {len(merged)} rows, expected {expected}")
    merged.sort(key=lambda row: int(row["experiment_order"]))
    if [int(row["experiment_order"]) for row in merged] != list(range(expected)):
        raise ValueError("Merged experiment order is not contiguous")
    validate_base_rows(merged, load_sources(root, domain, dict(config["domains"][domain])))
    write_jsonl(main_response_path, merged)
    merge_manifest = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "domain": domain,
        "plan_file": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "base_count": int(plan["base_count"]),
        "merged_count": len(merged),
        "responses_file": str(main_response_path),
        "responses_sha256": sha256_file(main_response_path),
        "shards": shard_audit,
    }
    write_json(plan_path.parent / "merge_manifest.json", merge_manifest)
    return main_response_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel, checkpoint-safe teacher generation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--config", type=Path, required=True)
    prepare_parser.add_argument("--domain", required=True)
    prepare_parser.add_argument("--shard-count", type=int, required=True)
    prepare_parser.add_argument("--name", required=True)

    shard_parser = subparsers.add_parser("run-shard")
    shard_parser.add_argument("--config", type=Path, required=True)
    shard_parser.add_argument("--plan", type=Path, required=True)
    shard_parser.add_argument("--shard-index", type=int, required=True)
    shard_parser.add_argument("--key-env-file", type=Path, required=True)
    shard_parser.add_argument("--key-env-name", required=True)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--config", type=Path, required=True)
    merge_parser.add_argument("--plan", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        output = prepare_plan(args.config, args.domain, args.shard_count, args.name)
    elif args.command == "run-shard":
        output = generate_shard(
            args.config,
            args.plan,
            args.shard_index,
            args.key_env_file,
            args.key_env_name,
        )
    else:
        output = merge_shards(args.config, args.plan)
    print(output)


if __name__ == "__main__":
    main()
