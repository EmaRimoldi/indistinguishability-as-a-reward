from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import random
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import yaml

from iar_phase0.openrouter import (
    OPENROUTER_URL,
    binary_probability_from_logprobs,
    make_chat_payload,
    response_text,
)

from .generate import prompts_for_domain, read_jsonl, sha256_text

CAMPAIGN_ROOT = Path("/path/to/runs")
MODEL_ID = "deepseek/deepseek-v4-flash"
PROVIDER = "StreamLake"
KEY_NAMES = (
    "OPENROUTER_API_KEY",
    "OPENROUTER_API_KEY_2",
    "OPENROUTER_API_KEY_3",
)
DOMAINS = ("commonsenseqa", "gsm8k", "kodcode", "spider")
PROTOCOLS = {
    "iar-judge-sanity-four-domain-v1",
    "iar-judge-sanity-cqa-confirm-v2",
    "iar-judge-sanity-cqa-confirm-v3-binary-contract",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def read_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Judge-sanity config must be a mapping")
    return value


def ensure_under(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root.expanduser().resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escaped {root}: {resolved}") from exc
    return resolved


def validate_config(config: dict[str, Any], *, require_candidates: bool = True) -> None:
    serialized = json.dumps(config, sort_keys=True).lower()
    if "mbpp" in serialized or "ifbench" in serialized or "ifeval" in serialized:
        raise ValueError("Forbidden benchmark in active campaign config")
    if config.get("protocol") not in PROTOCOLS:
        raise ValueError("Unexpected judge-sanity protocol")
    endpoint = config.get("endpoint") or {}
    routing = endpoint.get("provider_routing") or {}
    if (
        endpoint.get("model_id") != MODEL_ID
        or endpoint.get("expected_provider") != PROVIDER
        or routing.get("order") != [PROVIDER]
        or routing.get("allow_fallbacks") is not False
        or routing.get("require_parameters") is not True
        or endpoint.get("key_names") != list(KEY_NAMES)
        or int(endpoint.get("concurrency_per_key", -1)) != 1
    ):
        raise ValueError(
            "Judge endpoint is not the frozen StreamLake three-lane endpoint"
        )
    domains = config.get("domains") or {}
    if tuple(domains) != DOMAINS:
        raise ValueError(
            "Judge sanity requires the four frozen domains in canonical order"
        )
    enabled_domains = tuple(config.get("enabled_domains") or DOMAINS)
    if (
        not enabled_domains
        or len(set(enabled_domains)) != len(enabled_domains)
        or any(domain not in DOMAINS for domain in enabled_domains)
    ):
        raise ValueError("enabled_domains must be a unique non-empty subset of the campaign domains")
    if config.get("protocol") == "iar-judge-sanity-cqa-confirm-v2" and enabled_domains != (
        "commonsenseqa",
    ):
        raise ValueError("The confirmatory v2 protocol is restricted to CommonsenseQA")
    for domain, spec in domains.items():
        split_file = ensure_under(Path(spec["split_file"]), CAMPAIGN_ROOT, "split_file")
        candidate_file = ensure_under(
            Path(spec["candidate_file"]), CAMPAIGN_ROOT, "candidate_file"
        )
        run_dir = ensure_under(Path(spec["run_dir"]), CAMPAIGN_ROOT, "run_dir")
        if domain not in str(run_dir):
            raise ValueError(f"Domain run directory mismatch: {domain}")
        if not split_file.is_file() or (require_candidates and domain in enabled_domains and not candidate_file.is_file()):
            raise FileNotFoundError(f"Missing frozen input for {domain}")
    if int(config["teacher_generation"]["expected_per_domain"]) != 100:
        raise ValueError("Teacher generation must use probe100")
    if int(config["judging"]["expected_per_domain"]) != 300:
        raise ValueError("Judging must cover B-own, A0 and prompt imitation")
    if int(config["judging"]["top_logprobs"]) != 5:
        raise ValueError("Frozen binary judge requires top_logprobs=5")


def load_named_secret(path: Path, name: str) -> str:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == name:
            secret = value.strip().strip('"').strip("'")
            if not secret:
                raise ValueError(f"Empty credential: {name}")
            return secret
    raise KeyError(f"Missing credential: {name}")


@contextmanager
def key_lock(lock_dir: Path, key_name: str):
    lock_dir.mkdir(parents=True, exist_ok=True)
    handle = (lock_dir / f"{key_name}.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _domain_inputs(
    config: dict[str, Any], domain: str
) -> tuple[list[dict[str, Any]], list[str]]:
    spec = config["domains"][domain]
    rows = read_jsonl(Path(spec["split_file"]))
    tables = Path(config["paths"]["spider_tables_json"]) if domain == "spider" else None
    prompts = prompts_for_domain(domain, rows, tables_json=tables)["a0"]
    if len(rows) != 100 or len(prompts) != 100:
        raise ValueError(f"{domain} is not probe100")
    return rows, prompts


def attribution_prompt(
    domain: str, task: str, candidate: str, *, version: str | None = None
) -> str:
    if domain == "commonsenseqa":
        from iar_cqa.prompts import (
            BINARY_CONTRACT_ATTRIBUTION_PROMPT_VERSION,
            binary_contract_attribution_prompt,
        )
        from iar_cqa.prompts import (
            attribution_prompt as legacy_builder,
        )

        if version == BINARY_CONTRACT_ATTRIBUTION_PROMPT_VERSION:
            return binary_contract_attribution_prompt(task, candidate)
        return legacy_builder(task, candidate)
    if domain == "gsm8k":
        from iar_gsm8k.prompts import attribution_prompt as builder

        return builder(task, candidate)
    if domain == "spider":
        from iar_spider.prompts import attribution_prompt as builder

        return builder(task, candidate)
    if domain == "kodcode":
        from iar_phase0.prompts import (
            pointwise_self_attribution_no_system_prompt_from_strings,
        )

        return pointwise_self_attribution_no_system_prompt_from_strings(task, candidate)
    raise ValueError(domain)


def build_tasks(config: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    prompt_version = None
    if config.get("protocol") == "iar-judge-sanity-cqa-confirm-v3-binary-contract":
        from iar_cqa.prompts import BINARY_CONTRACT_ATTRIBUTION_PROMPT_VERSION

        prompt_version = BINARY_CONTRACT_ATTRIBUTION_PROMPT_VERSION
    for domain in tuple(config.get("enabled_domains") or DOMAINS):
        rows, task_prompts = _domain_inputs(config, domain)
        if phase == "teacher":
            for row, task in zip(rows, task_prompts, strict=True):
                logical_key = f"teacher::{domain}::{row['id']}"
                tasks.append(
                    {
                        "logical_key": logical_key,
                        "domain": domain,
                        "id": str(row["id"]),
                        "source_index": int(row["source_index"]),
                        "task_prompt": task,
                        "task_prompt_sha256": sha256_text(task),
                    }
                )
            continue
        if phase != "judge":
            raise ValueError(phase)
        own_path = Path(config["domains"][domain]["run_dir"]) / "b_own_responses.jsonl"
        own = read_jsonl(own_path)
        actor = read_jsonl(Path(config["domains"][domain]["candidate_file"]))
        candidates = own + actor
        by_id = {
            str(row["id"]): task for row, task in zip(rows, task_prompts, strict=True)
        }
        for candidate in candidates:
            candidate_id = str(candidate["id"])
            variant = str(candidate["variant"])
            if variant not in {"own", "a0", "prompt_imitation"}:
                raise ValueError(f"Unexpected candidate variant: {variant}")
            candidate_text = str(candidate["response"]).strip()
            if not candidate_text:
                raise ValueError(f"Empty candidate: {domain}/{candidate_id}/{variant}")
            prompt = attribution_prompt(
                domain,
                by_id[candidate_id],
                candidate_text,
                version=prompt_version,
            )
            tasks.append(
                {
                    "logical_key": f"judge::{domain}::{candidate_id}::{variant}",
                    "domain": domain,
                    "id": candidate_id,
                    "source_index": int(candidate["source_index"]),
                    "variant": variant,
                    "ground_truth_own": variant == "own",
                    "candidate_sha256": sha256_text(candidate_text),
                    "judge_prompt": prompt,
                    "judge_prompt_sha256": sha256_text(prompt),
                }
            )
    return sorted(tasks, key=lambda row: str(row["logical_key"]))


def shard_tasks(
    tasks: list[dict[str, Any]], shard_index: int, num_shards: int
) -> list[dict[str, Any]]:
    if not 0 <= shard_index < num_shards:
        raise ValueError("Invalid shard index")
    return [
        row
        for row in tasks
        if int(hashlib.sha256(str(row["logical_key"]).encode()).hexdigest(), 16)
        % num_shards
        == shard_index
    ]


def _events_state(path: Path) -> dict[str, str]:
    state: dict[str, str] = {}
    for event in read_jsonl(path) if path.is_file() else []:
        state[str(event["logical_key"])] = str(event["event"])
    return state


def _request_payload(
    config: dict[str, Any], task: dict[str, Any], phase: str
) -> dict[str, Any]:
    endpoint = config["endpoint"]
    if phase == "teacher":
        phase_config = config["teacher_generation"]
        return make_chat_payload(
            model_id=MODEL_ID,
            prompt=str(task["task_prompt"]),
            max_tokens=int(phase_config["max_tokens"]),
            temperature=float(phase_config["temperature"]),
            provider_routing=dict(endpoint["provider_routing"]),
            request_overrides=dict(endpoint["request_overrides"]),
        )
    phase_config = config["judging"]
    return make_chat_payload(
        model_id=MODEL_ID,
        prompt=str(task["judge_prompt"]),
        max_tokens=int(phase_config["max_tokens"]),
        temperature=float(phase_config["temperature"]),
        logprobs=True,
        top_logprobs=int(phase_config["top_logprobs"]),
        provider_routing=dict(endpoint["provider_routing"]),
        request_overrides=dict(endpoint["request_overrides"]),
    )


def _validated_result(
    raw: dict[str, Any], task: dict[str, Any], phase: str
) -> dict[str, Any]:
    if raw.get("model") != MODEL_ID or raw.get("provider") != PROVIDER:
        raise RuntimeError("misrouted_response")
    emitted = response_text(raw).strip()
    if phase == "teacher":
        if not emitted:
            raise RuntimeError("empty_teacher_response")
        return {
            **{key: value for key, value in task.items() if key != "task_prompt"},
            "response": emitted,
            "response_sha256": sha256_text(emitted),
            "raw_model": raw.get("model"),
            "raw_provider": raw.get("provider"),
            "finish_reason": raw["choices"][0].get("finish_reason"),
            "usage": raw.get("usage") or {},
        }
    parsed = binary_probability_from_logprobs(raw)
    if not parsed.get("ok"):
        raise RuntimeError(f"judge_logprob_contract:{parsed.get('error')}")
    label = emitted if emitted in {"0", "1"} else None
    argmax = "1" if float(parsed["logp_1"]) > float(parsed["logp_0"]) else "0"
    return {
        **{key: value for key, value in task.items() if key != "judge_prompt"},
        "p_own": float(parsed["p1"]),
        "logp_0": float(parsed["logp_0"]),
        "logp_1": float(parsed["logp_1"]),
        "binary_label_mass": float(parsed["binary_label_mass"]),
        "judge_token": label,
        "judge_argmax_token": argmax,
        "emitted_argmax_consistent": label == argmax,
        "raw_model": raw.get("model"),
        "raw_provider": raw.get("provider"),
        "usage": raw.get("usage") or {},
    }


def run_lane(
    *,
    config_path: Path,
    phase: str,
    shard_index: int,
    num_shards: int,
    key_name: str,
    key_pool: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = read_config(config_path)
    validate_config(config, require_candidates=phase == "judge")
    if num_shards != len(KEY_NAMES) or key_name != KEY_NAMES[shard_index]:
        raise ValueError("Shard-to-key mapping changed")
    output_dir = ensure_under(output_dir, CAMPAIGN_ROOT, "lane output")
    output_dir.mkdir(parents=True, exist_ok=True)
    response_path = output_dir / "responses.jsonl"
    event_path = output_dir / "events.jsonl"
    existing = read_jsonl(response_path) if response_path.is_file() else []
    done = {str(row["logical_key"]) for row in existing}
    states = _events_state(event_path)
    ambiguous = sorted(
        key
        for key, state in states.items()
        if state in {"REQUEST_STARTED", "REQUEST_AMBIGUOUS"} and key not in done
    )
    if ambiguous:
        raise RuntimeError(f"Fail-closed ambiguous prior requests: {ambiguous[:3]}")
    tasks = shard_tasks(build_tasks(config, phase), shard_index, num_shards)
    planned = {str(row["logical_key"]) for row in tasks}
    if not done <= planned:
        raise RuntimeError("Existing lane response ledger does not match launch")
    secret = load_named_secret(key_pool, key_name)
    headers = {
        "Authorization": f"Bearer {secret}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "IAR campaign judge sanity",
        "HTTP-Referer": "https://example.invalid/iar-code",
    }
    transport = config["transport"]
    max_attempts = int(transport["max_attempts_per_call"])
    timeout = float(transport["timeout_seconds"])
    started_at = time.time()
    http_requests = 0
    with (
        key_lock(Path(config["paths"]["global_lock_dir"]), key_name),
        httpx.Client(timeout=timeout) as client,
    ):
        for task in tasks:
            logical_key = str(task["logical_key"])
            if logical_key in done:
                continue
            payload = _request_payload(config, task, phase)
            last_error = None
            for attempt in range(1, max_attempts + 1):
                payload_hash = sha256_text(json.dumps(payload, sort_keys=True))
                append_event(
                    event_path,
                    {
                        "event": "REQUEST_STARTED",
                        "logical_key": logical_key,
                        "attempt": attempt,
                        "payload_sha256": payload_hash,
                        "request_top_logprobs": payload.get("top_logprobs"),
                        "time_unix": time.time(),
                    },
                )
                http_requests += 1
                try:
                    response = client.post(
                        OPENROUTER_URL, headers=headers, json=payload
                    )
                except httpx.TransportError as exc:
                    append_event(
                        event_path,
                        {
                            "event": "REQUEST_AMBIGUOUS",
                            "logical_key": logical_key,
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                            "time_unix": time.time(),
                        },
                    )
                    raise RuntimeError(
                        f"Ambiguous transport failure for {logical_key}"
                    ) from exc
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"http_{response.status_code}"
                    append_event(
                        event_path,
                        {
                            "event": "RETRYABLE_HTTP_RESPONSE",
                            "logical_key": logical_key,
                            "attempt": attempt,
                            "status_code": response.status_code,
                            "time_unix": time.time(),
                        },
                    )
                    if attempt < max_attempts:
                        delay = min(
                            5.0 * 2 ** (attempt - 1),
                            float(transport["max_retry_wait_seconds"]),
                        )
                        time.sleep(delay)
                        continue
                    break
                if response.status_code >= 400:
                    append_event(
                        event_path,
                        {
                            "event": "HARD_HTTP_RESPONSE",
                            "logical_key": logical_key,
                            "attempt": attempt,
                            "status_code": response.status_code,
                            "time_unix": time.time(),
                        },
                    )
                    response.raise_for_status()
                raw = response.json()
                try:
                    result = _validated_result(raw, task, phase)
                except RuntimeError as exc:
                    last_error = str(exc)
                    retryable = (
                        last_error == "misrouted_response"
                        or last_error.startswith("judge_logprob_contract:missing_")
                    )
                    append_event(
                        event_path,
                        {
                            "event": "INVALID_API_RESPONSE",
                            "logical_key": logical_key,
                            "attempt": attempt,
                            "error": last_error,
                            "retryable": retryable,
                            "time_unix": time.time(),
                        },
                    )
                    if retryable and attempt < max_attempts:
                        if "missing_binary_label_in_top_logprobs" in last_error:
                            payload = {**payload, "top_logprobs": 20}
                        time.sleep(2.0)
                        continue
                    break
                if phase == "judge":
                    result["request_top_logprobs"] = payload.get("top_logprobs")
                existing.append(result)
                existing.sort(key=lambda row: str(row["logical_key"]))
                write_jsonl(response_path, existing)
                done.add(logical_key)
                append_event(
                    event_path,
                    {
                        "event": "REQUEST_SUCCEEDED",
                        "logical_key": logical_key,
                        "attempt": attempt,
                        "time_unix": time.time(),
                    },
                )
                last_error = None
                break
            if last_error is not None:
                raise RuntimeError(
                    f"API request failed closed for {logical_key}: {last_error}"
                )
    if done != planned:
        raise RuntimeError("Lane ended with incomplete coverage")
    summary = {
        "schema_version": 1,
        "status": "complete",
        "phase": phase,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "key_name": key_name,
        "key_fingerprint": hashlib.sha256(secret.encode()).hexdigest()[:16],
        "planned_calls": len(tasks),
        "successful_calls": len(existing),
        "http_requests_this_process": http_requests,
        "elapsed_seconds": time.time() - started_at,
        "response_sha256": sha256_file(response_path),
        "event_sha256": sha256_file(event_path),
        "config_sha256": sha256_file(config_path),
        "official_test_accessed": False,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def auc(positive: list[float], negative: list[float]) -> float:
    if not positive or not negative:
        raise ValueError("AUC requires both classes")
    wins = sum(
        1.0 if left > right else 0.5 if left == right else 0.0
        for left in positive
        for right in negative
    )
    return wins / (len(positive) * len(negative))


def paired_summary(
    deltas: list[float], *, seed: int, replicates: int
) -> dict[str, Any]:
    if not deltas:
        raise ValueError("Paired summary requires values")
    generator = random.Random(seed)
    bootstrap = sorted(
        statistics.mean(generator.choice(deltas) for _ in deltas)
        for _ in range(replicates)
    )
    index = max(0, math.ceil(0.05 * replicates) - 1)
    return {
        "n": len(deltas),
        "mean_delta": statistics.mean(deltas),
        "standard_error": statistics.stdev(deltas) / math.sqrt(len(deltas)),
        "bootstrap_one_sided_95_lcb": bootstrap[index],
        "bootstrap_replicates": replicates,
    }


def binary_mass_gate(values: list[float], thresholds: dict[str, Any]) -> dict[str, Any]:
    if not values:
        raise ValueError("Binary-mass gate requires observations")
    if "minimum_binary_label_mass" in thresholds:
        minimum = float(thresholds["minimum_binary_label_mass"])
        return {
            "policy": "minimum_only",
            "pass": min(values) >= minimum,
            "minimum_observed": min(values),
            "minimum_required": minimum,
        }
    floor = float(thresholds["binary_label_mass_floor"])
    target = float(thresholds["binary_label_mass_target"])
    required_coverage = float(thresholds["minimum_target_coverage"])
    coverage = statistics.mean(value >= target for value in values)
    return {
        "policy": "floor_and_coverage",
        "pass": min(values) >= floor and coverage >= required_coverage,
        "minimum_observed": min(values),
        "floor_required": floor,
        "target": target,
        "target_coverage_observed": coverage,
        "target_coverage_required": required_coverage,
    }


def collect_phase(config_path: Path, phase: str) -> dict[str, Any]:
    config = read_config(config_path)
    validate_config(config, require_candidates=phase == "judge")
    phase_root = Path(config["paths"][f"{phase}_lane_root"])
    lane_rows: list[dict[str, Any]] = []
    lane_summaries = []
    for index in range(len(KEY_NAMES)):
        lane = phase_root / f"lane_{index}"
        summary = json.loads((lane / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "complete":
            raise RuntimeError(f"Incomplete {phase} lane {index}")
        lane_summaries.append(summary)
        lane_rows.extend(read_jsonl(lane / "responses.jsonl"))
    expected = build_tasks(config, phase)
    expected_keys = {str(row["logical_key"]) for row in expected}
    observed_keys = [str(row["logical_key"]) for row in lane_rows]
    if (
        len(observed_keys) != len(set(observed_keys))
        or set(observed_keys) != expected_keys
    ):
        raise RuntimeError(f"{phase} collection coverage mismatch")
    if phase == "teacher":
        active_domains = tuple(config.get("enabled_domains") or DOMAINS)
        by_domain: dict[str, list[dict[str, Any]]] = {domain: [] for domain in active_domains}
        for row in lane_rows:
            by_domain[str(row["domain"])].append(
                {
                    "id": str(row["id"]),
                    "source_index": int(row["source_index"]),
                    "domain": str(row["domain"]),
                    "source": "B_own",
                    "variant": "own",
                    "model_id": MODEL_ID,
                    "response": str(row["response"]),
                    "response_sha256": str(row["response_sha256"]),
                    "prompt_sha256": str(row["task_prompt_sha256"]),
                    "raw_model": str(row["raw_model"]),
                    "raw_provider": str(row["raw_provider"]),
                    "finish_reason": row.get("finish_reason"),
                    "usage": row.get("usage") or {},
                }
            )
        for domain, rows in by_domain.items():
            if len(rows) != 100:
                raise RuntimeError(f"Teacher coverage mismatch for {domain}")
            if any(row.get("finish_reason") != "stop" for row in rows):
                raise RuntimeError(
                    f"Teacher generation did not terminate normally for {domain}"
                )
            rows.sort(key=lambda row: str(row["id"]))
            write_jsonl(
                Path(config["domains"][domain]["run_dir"]) / "b_own_responses.jsonl",
                rows,
            )
        report = {
            "schema_version": 1,
            "phase": "teacher",
            "status": "complete",
            "calls": len(lane_rows),
            "calls_per_domain": {
                domain: len(rows) for domain, rows in by_domain.items()
            },
            "models": sorted({str(row["raw_model"]) for row in lane_rows}),
            "providers": sorted({str(row["raw_provider"]) for row in lane_rows}),
            "lane_summaries": lane_summaries,
            "official_test_accessed": False,
        }
        write_json(Path(config["paths"]["teacher_collection"]), report)
        return report
    thresholds = config["gate_thresholds"]
    per_domain: dict[str, Any] = {}
    active_domains = tuple(config.get("enabled_domains") or DOMAINS)
    for domain in active_domains:
        rows = [row for row in lane_rows if row["domain"] == domain]
        groups = {
            variant: [row for row in rows if row["variant"] == variant]
            for variant in ("own", "a0", "prompt_imitation")
        }
        p_by_variant = {
            variant: [float(row["p_own"]) for row in values]
            for variant, values in groups.items()
        }
        per_id = {
            str(row["id"]): {
                variant: next(
                    float(item["p_own"])
                    for item in groups[variant]
                    if str(item["id"]) == str(row["id"])
                )
                for variant in groups
            }
            for row in groups["own"]
        }
        paired_a0 = paired_summary(
            [values["own"] - values["a0"] for values in per_id.values()],
            seed=int(config["seed"]) + DOMAINS.index(domain),
            replicates=int(thresholds["bootstrap_replicates"]),
        )
        paired_imitation = paired_summary(
            [values["own"] - values["prompt_imitation"] for values in per_id.values()],
            seed=int(config["seed"]) + 100 + DOMAINS.index(domain),
            replicates=int(thresholds["bootstrap_replicates"]),
        )
        mass_values = [float(row["binary_label_mass"]) for row in rows]
        mass_gate = binary_mass_gate(mass_values, thresholds)
        emitted_consistency = statistics.mean(
            float(bool(row["emitted_argmax_consistent"])) for row in rows
        )
        protocol_ok = (
            len(rows) == 300
            and all(len(values) == 100 for values in groups.values())
            and {str(row["raw_model"]) for row in rows} == {MODEL_ID}
            and {str(row["raw_provider"]) for row in rows} == {PROVIDER}
            and all(row["judge_token"] in {"0", "1"} for row in rows)
        )
        control_auc = auc(p_by_variant["own"], p_by_variant["a0"])
        checks = {
            "complete_protocol": protocol_ok,
            "binary_label_support": bool(mass_gate["pass"]),
            "emitted_argmax_consistency": emitted_consistency
            >= float(thresholds.get("minimum_emitted_argmax_consistency", 0.0)),
            "positive_negative_auc": control_auc >= float(thresholds["minimum_auc"]),
            "positive_negative_mean_gap": paired_a0["mean_delta"]
            >= float(thresholds["minimum_mean_p_own_gap"]),
            "positive_negative_lcb_above_zero": paired_a0["bootstrap_one_sided_95_lcb"]
            > 0,
        }
        per_domain[domain] = {
            "decision": "GO" if all(checks.values()) else "NO_GO",
            "checks": checks,
            "n": len(rows),
            "mean_p_own": {
                variant: statistics.mean(values)
                for variant, values in p_by_variant.items()
            },
            "auc_own_vs_a0": control_auc,
            "auc_own_vs_prompt_imitation": auc(
                p_by_variant["own"], p_by_variant["prompt_imitation"]
            ),
            "paired_own_minus_a0": paired_a0,
            "paired_own_minus_prompt_imitation": paired_imitation,
            "minimum_binary_label_mass": min(
                float(row["binary_label_mass"]) for row in rows
            ),
            "binary_label_mass_gate": mass_gate,
            "emitted_argmax_consistency": emitted_consistency,
        }
    report = {
        "schema_version": 1,
        "protocol": config["protocol"],
        "enabled_domains": list(active_domains),
        "decision": "GO"
        if all(result["decision"] == "GO" for result in per_domain.values())
        else "NO_GO",
        "calls": len(lane_rows),
        "domains": per_domain,
        "thresholds": thresholds,
        "models": sorted({str(row["raw_model"]) for row in lane_rows}),
        "providers": sorted({str(row["raw_provider"]) for row in lane_rows}),
        "lane_summaries": lane_summaries,
        "config_sha256": sha256_file(config_path),
        "official_test_accessed": False,
    }
    output = Path(config["paths"]["judge_gate"])
    write_json(output, report)
    return report
