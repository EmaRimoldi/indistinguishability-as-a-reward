from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx

from iar_campaign.generate import prompts_for_domain, read_jsonl, sha256_text
from iar_campaign.judge_sanity import append_event
from iar_campaign.reward_pilot import (
    key_lock,
    load_named_secret,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from iar_phase0.generator import strip_code_fences
from iar_phase0.openrouter import OPENROUTER_URL, make_chat_payload, response_text

from .baseline_suite import PROTOCOL, SELECTION, STEP_ROOT

MODEL_ID = "deepseek/deepseek-v4-flash"
PROVIDER = "StreamLake"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def validate_teacher_config(config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = _read_json(config_path)
    domain = str(config.get("domain", ""))
    if domain not in {"commonsenseqa", "gsm8k", "kodcode", "spider"}:
        raise ValueError(f"Unsupported teacher domain: {domain}; MBPP is forbidden")
    serialized = json.dumps(config, sort_keys=True).lower()
    if "mbpp" in serialized or "ifbench" in serialized or "ifeval" in serialized:
        raise ValueError("Forbidden benchmark in Step 7 teacher config")
    selection = _read_json(SELECTION)
    if (
        sha256_file(SELECTION) != str(config.get("domain_selection_sha256"))
        or selection.get("status") != "complete"
        or selection.get("protocol")
        != "campaign-four-domain-baseline-extension-v1"
        or selection.get("selected_domains")
        != ["commonsenseqa", "gsm8k", "kodcode", "spider"]
        or selection.get("extension_domains") != ["kodcode", "spider"]
        or domain not in selection.get("extension_domains", [])
        or selection.get("official_test_accessed") is not False
    ):
        raise RuntimeError("Teacher config is not bound to the Step 15 extension")
    if sha256_file(PROTOCOL) != str(config.get("baseline_protocol_sha256")):
        raise RuntimeError("Teacher config baseline protocol drifted")
    routing = config.get("provider_routing") or {}
    if (
        config.get("protocol") != "campaign-step15-sft-teacher-v1"
        or config.get("model_id") != MODEL_ID
        or config.get("expected_provider") != PROVIDER
        or routing.get("order") != [PROVIDER]
        or routing.get("allow_fallbacks") is not False
        or routing.get("require_parameters") is not True
        or float(config.get("temperature", -1)) != 0
        or float(config.get("top_p", -1)) != 1
        or int(config.get("expected_count", -1)) != 1000
        or config.get("official_test_accessed") is not False
    ):
        raise RuntimeError("Teacher endpoint or data contract drifted")
    train = Path(str(config["training_split"]))
    if sha256_file(train) != str(config["training_split_sha256"]):
        raise RuntimeError("Teacher training split hash drifted")
    if any(part.lower() in {"test", "sealed", "final_test"} for part in train.parts):
        raise ValueError("Teacher corpus cannot use official or sealed test data")
    examples = read_jsonl(train)
    if len(examples) != 1000 or len({str(row["id"]) for row in examples}) != 1000:
        raise RuntimeError("Teacher corpus requires 1000 unique training IDs")
    expected_root = (STEP_ROOT / "domains" / domain / "sft_teacher").resolve()
    if Path(str(config["output_root"])).resolve() != expected_root:
        raise ValueError("Teacher output escaped its canonical arm directory")
    return config, examples


def _outcome(
    *,
    domain: str,
    response: str,
    example: dict[str, Any],
    spider_database_root: Path | None,
) -> tuple[str, dict[str, Any]]:
    normalized = strip_code_fences(response) if domain in {"kodcode", "spider"} else response.strip()
    if domain == "commonsenseqa":
        from iar_cqa.scoring import score_response

        score = score_response(normalized, str(example["answer_key"]))
    elif domain == "gsm8k":
        from iar_gsm8k.scoring import score_response

        score = score_response(normalized, str(example["answer_value"]))
    elif domain == "kodcode":
        from iar_campaign.reward_pilot import _score_kodcode

        score = _score_kodcode((normalized, str(example["hidden_tests"]), 60))
    else:
        if spider_database_root is None:
            raise ValueError("Spider teacher scoring requires a database root")
        from iar_campaign.score_baselines import _score_spider_one

        score = _score_spider_one(
            (example, {"response": normalized}, spider_database_root)
        )["score"]
    return normalized, dict(score)


def _event_state(path: Path) -> dict[str, str]:
    state: dict[str, str] = {}
    if path.is_file():
        for row in read_jsonl(path):
            state[str(row["logical_key"])] = str(row["event"])
    return state


def _teacher_row(
    *,
    domain: str,
    task: dict[str, Any],
    raw: dict[str, Any],
    attempt: int,
    spider_database_root: Path | None,
) -> dict[str, Any]:
    choice = raw["choices"][0]
    normalized, score = _outcome(
        domain=domain,
        response=response_text(raw),
        example=task["example"],
        spider_database_root=spider_database_root,
    )
    return {
        "schema_version": 1,
        "logical_key": task["logical_key"],
        "id": task["id"],
        "source_index": task["source_index"],
        "experiment_order": task["experiment_order"],
        "domain": domain,
        "prompt": task["prompt"],
        "prompt_sha256": task["prompt_sha256"],
        "response": normalized,
        "response_sha256": sha256_text(normalized),
        "score": score,
        "requested_model": MODEL_ID,
        "raw_model": raw.get("model"),
        "raw_provider": raw.get("provider"),
        "openrouter_response_id": raw.get("id"),
        "finish_reason": choice.get("finish_reason"),
        "usage": raw.get("usage") or {},
        "raw_response": raw,
        "attempt": attempt,
        "created_at": utc_now(),
        "official_test_accessed": False,
    }


def run_teacher_corpus(
    *,
    config_path: Path,
    run_id: str,
    key_pool: Path,
    key_name: str,
    lock_dir: Path,
) -> dict[str, Any]:
    config, examples = validate_teacher_config(config_path)
    domain = str(config["domain"])
    run_root = Path(str(config["output_root"])) / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    response_path = run_root / "responses.jsonl"
    event_path = run_root / "events.jsonl"
    receipt_path = run_root / "api_receipts.jsonl"
    summary_path = run_root / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Teacher run already finalized: {summary_path}")
    existing = read_jsonl(response_path) if response_path.is_file() else []
    done = {str(row["logical_key"]): row for row in existing}
    if len(done) != len(existing):
        raise RuntimeError("Teacher response ledger has duplicate logical keys")
    states = _event_state(event_path)
    receipt_rows = read_jsonl(receipt_path) if receipt_path.is_file() else []
    receipts = {str(row["logical_key"]): row for row in receipt_rows}
    if len(receipts) != len(receipt_rows):
        raise RuntimeError("Teacher API receipt ledger has duplicate logical keys")
    ambiguous = sorted(
        key
        for key, state in states.items()
        if state in {"REQUEST_STARTED", "REQUEST_AMBIGUOUS"}
        and key not in done
        and key not in receipts
    )
    if ambiguous:
        raise RuntimeError(f"Fail-closed ambiguous teacher requests: {ambiguous[:3]}")
    tables = Path(str(config["spider_tables_json"])) if domain == "spider" else None
    prompts = prompts_for_domain(domain, examples, tables_json=tables)["a0"]
    tasks = [
        {
            "logical_key": f"teacher::{domain}::{example['id']}",
            "id": str(example["id"]),
            "source_index": int(example["source_index"]),
            "experiment_order": order,
            "prompt": prompt,
            "prompt_sha256": sha256_text(prompt),
            "example": example,
        }
        for order, (example, prompt) in enumerate(zip(examples, prompts, strict=True))
    ]
    planned = {str(task["logical_key"]) for task in tasks}
    if not set(done) <= planned:
        raise RuntimeError("Existing teacher ledger is not a subset of the frozen plan")
    secret = load_named_secret(key_pool, key_name)
    headers = {
        "Authorization": f"Bearer {secret}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "IAR campaign Step 15 teacher corpus",
        "HTTP-Referer": "https://local.cluster/indistinguishability-as-a-reward",
    }
    transport = config["transport"]
    max_attempts = int(transport["max_attempts_per_call"])
    http_requests = 0
    started = time.time()
    database_root = (
        Path(str(config["spider_database_root"])) if domain == "spider" else None
    )
    with (
        key_lock(lock_dir, key_name),
        httpx.Client(timeout=float(transport["timeout_seconds"])) as client,
    ):
        last_request = 0.0
        for task in tasks:
            logical_key = str(task["logical_key"])
            if logical_key in done:
                if done[logical_key].get("prompt_sha256") != task["prompt_sha256"]:
                    raise RuntimeError(f"Teacher prompt drift while resuming {logical_key}")
                continue
            if logical_key in receipts:
                receipt = receipts[logical_key]
                if receipt.get("prompt_sha256") != task["prompt_sha256"]:
                    raise RuntimeError(f"Teacher receipt prompt drift: {logical_key}")
                row = _teacher_row(
                    domain=domain,
                    task=task,
                    raw=dict(receipt["raw_response"]),
                    attempt=int(receipt["attempt"]),
                    spider_database_root=database_root,
                )
                existing.append(row)
                write_jsonl(response_path, existing)
                done[logical_key] = row
                append_event(
                    event_path,
                    {
                        "event": "RESPONSE_RECOVERED_FROM_RECEIPT",
                        "logical_key": logical_key,
                        "time_unix": time.time(),
                    },
                )
                continue
            payload = make_chat_payload(
                model_id=MODEL_ID,
                prompt=str(task["prompt"]),
                system_prompt=None,
                max_tokens=int(config["max_tokens"]),
                temperature=0,
                provider_routing=dict(config["provider_routing"]),
                request_overrides={
                    "top_p": 1.0,
                    **dict(config.get("request_overrides") or {}),
                },
            )
            last_error: str | None = None
            for attempt in range(1, max_attempts + 1):
                interval = float(transport["minimum_request_interval_seconds"])
                wait = interval - (time.monotonic() - last_request)
                if wait > 0:
                    time.sleep(wait)
                payload_hash = sha256_text(json.dumps(payload, sort_keys=True))
                append_event(
                    event_path,
                    {
                        "event": "REQUEST_STARTED",
                        "logical_key": logical_key,
                        "attempt": attempt,
                        "payload_sha256": payload_hash,
                        "time_unix": time.time(),
                    },
                )
                last_request = time.monotonic()
                http_requests += 1
                try:
                    response = client.post(OPENROUTER_URL, headers=headers, json=payload)
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
                        f"Ambiguous teacher transport failure for {logical_key}"
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
                        time.sleep(
                            min(
                                5.0 * 2 ** (attempt - 1),
                                float(transport["max_retry_wait_seconds"]),
                            )
                        )
                        continue
                    break
                if response.status_code >= 400:
                    response.raise_for_status()
                try:
                    raw = response.json()
                    if (
                        raw.get("model") != MODEL_ID
                        or raw.get("provider") != PROVIDER
                        or not response_text(raw).strip()
                    ):
                        raise RuntimeError("misrouted_or_incomplete_teacher_response")
                    choice = raw["choices"][0]
                    if choice.get("finish_reason") != "stop":
                        raise RuntimeError(
                            f"teacher_generation_not_stopped:{choice.get('finish_reason')}"
                        )
                except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                    last_error = str(exc)
                    append_event(
                        event_path,
                        {
                            "event": "INVALID_API_RESPONSE",
                            "logical_key": logical_key,
                            "attempt": attempt,
                            "error": last_error,
                            "time_unix": time.time(),
                        },
                    )
                    if attempt < max_attempts:
                        time.sleep(2.0)
                        continue
                    break
                receipt = {
                    "event": "API_RESPONSE_RECEIVED",
                    "logical_key": logical_key,
                    "attempt": attempt,
                    "prompt_sha256": task["prompt_sha256"],
                    "payload_sha256": payload_hash,
                    "openrouter_response_id": raw.get("id"),
                    "raw_response": raw,
                    "time_unix": time.time(),
                }
                append_event(receipt_path, receipt)
                receipts[logical_key] = receipt
                row = _teacher_row(
                    domain=domain,
                    task=task,
                    raw=raw,
                    attempt=attempt,
                    spider_database_root=database_root,
                )
                existing.append(row)
                write_jsonl(response_path, existing)
                done[logical_key] = row
                append_event(
                    event_path,
                    {
                        "event": "REQUEST_SUCCEEDED",
                        "logical_key": logical_key,
                        "attempt": attempt,
                        "openrouter_response_id": raw.get("id"),
                        "time_unix": time.time(),
                    },
                )
                last_error = None
                break
            if last_error is not None:
                raise RuntimeError(f"Teacher request failed for {logical_key}: {last_error}")
    if set(done) != planned:
        raise RuntimeError("Teacher corpus ended with incomplete coverage")
    existing.sort(key=lambda row: int(row["experiment_order"]))
    write_jsonl(response_path, existing)
    sft = [
        {
            "id": row["id"],
            "domain": domain,
            "messages": [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]},
            ],
            "teacher_correct": bool(row["score"].get("correct")),
            "teacher_valid": bool(row["score"].get("valid", True)),
            "source_response_sha256": row["response_sha256"],
        }
        for row in existing
    ]
    write_jsonl(run_root / "sft_all.jsonl", sft)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "protocol": "campaign-step15-sft-teacher-v1",
        "domain": domain,
        "run_id": run_id,
        "n": len(existing),
        "valid": sum(bool(row["score"].get("valid", True)) for row in existing),
        "correct": sum(bool(row["score"].get("correct")) for row in existing),
        "http_requests_this_process": http_requests,
        "elapsed_seconds": time.time() - started,
        "responses": str(response_path),
        "responses_sha256": sha256_file(response_path),
        "sft_all": str(run_root / "sft_all.jsonl"),
        "sft_all_sha256": sha256_file(run_root / "sft_all.jsonl"),
        "events": str(event_path),
        "events_sha256": sha256_file(event_path),
        "api_receipts": str(receipt_path),
        "api_receipts_sha256": sha256_file(receipt_path),
        "model_id": MODEL_ID,
        "provider": PROVIDER,
        "official_test_accessed": False,
        "mbpp": "FORBIDDEN",
    }
    write_json(summary_path, summary)
    return summary
