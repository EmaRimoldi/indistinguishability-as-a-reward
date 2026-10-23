from __future__ import annotations

import concurrent.futures
import gc
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx

from iar_campaign.four_domain_screen import (
    CAMPAIGN_ROOT,
    DOMAINS,
    _adapter_for_step,
    _attribution_template,
    _read_json,
    _under,
)
from iar_campaign.generate import prompts_for_domain, read_jsonl, sha256_text
from iar_campaign.judge_sanity import (
    append_event,
    auc,
    binary_mass_gate,
    paired_summary,
)
from iar_campaign.judge_sanity import (
    attribution_prompt as domain_attribution_prompt,
)
from iar_campaign.reward_pilot import (
    _score_kodcode,
    key_lock,
    load_named_secret,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from iar_campaign.score_baselines import (
    _score_kodcode_rows,
    _score_simple,
    _score_spider_one,
    _score_spider_rows,
)
from iar_grpo.config import GrpoTrialConfig, load_grpo_config
from iar_grpo.reward import LiveOpenRouterReward, settings_from_config
from iar_grpo.train import run_training
from iar_phase0.generator import strip_code_fences
from iar_phase0.openrouter import (
    OPENROUTER_URL,
    binary_probability_from_logprobs,
    make_chat_payload,
    response_text,
)
from iar_phase0.parameters import initialize_parameter_record

STEP_ROOT = CAMPAIGN_ROOT / "10_discriminator_transfer"
STEP5_ROOT = CAMPAIGN_ROOT / "05_four_domain_screen_1p5b"
STEP9_ROOT = CAMPAIGN_ROOT / "09_pairwise_generate_distinguish"
SELECTION = (
    CAMPAIGN_ROOT / "06_two_domain_selection/results/two_domain_selection_v1.json"
)
PROTOCOL = STEP_ROOT / "configs/discriminator_transfer_protocol_v1.json"
BASELINE_REGISTRY = STEP5_ROOT / "results/validation_baseline_registry_v1.json"
CHECKPOINTS = (0, 500, 1000, 1500, 2000)


def _candidate_by_slug(protocol: dict[str, Any], slug: str) -> dict[str, Any]:
    matches = [
        row for row in protocol["candidate_discriminators"] if row["slug"] == slug
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicated discriminator slug: {slug}")
    return dict(matches[0])


def _validate_selection() -> list[str]:
    value = _read_json(SELECTION)
    selected = list(value.get("selected_domains") or [])
    if (
        value.get("status") != "complete"
        or len(selected) != 2
        or selected[0] != "commonsenseqa"
        or any(domain not in DOMAINS for domain in selected)
        or value.get("official_test_accessed") is not False
    ):
        raise RuntimeError("Step 6 selection is incomplete or invalid")
    return selected


def validate_screen_config(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    config = _read_json(path)
    serialized = json.dumps(config, sort_keys=True).lower()
    if "mbpp" in serialized or "ifbench" in serialized or "ifeval" in serialized:
        raise ValueError("Forbidden benchmark in Step 10 screen")
    protocol = _read_json(PROTOCOL)
    selected = _validate_selection()
    domain = str(config.get("domain", ""))
    if (
        config.get("protocol") != "campaign-discriminator-transfer-screen-v1"
        or sha256_file(PROTOCOL) != str(config.get("transfer_protocol_sha256"))
        or sha256_file(SELECTION) != str(config.get("domain_selection_sha256"))
        or domain not in selected
        or config.get("gate_thresholds") != protocol.get("gate_thresholds")
        or config.get("official_test_accessed") is not False
    ):
        raise RuntimeError("Step 10 screen is not bound to frozen campaign inputs")
    candidate = _candidate_by_slug(protocol, str(config["candidate_slug"]))
    endpoint = config["endpoint"]
    if (
        endpoint.get("model_id") != candidate["model_id"]
        or endpoint.get("target_model_name") != candidate["target_model_name"]
        or endpoint.get("expected_provider") != candidate["provider"]
        or endpoint.get("provider_routing")
        != {
            "order": [candidate["provider"]],
            "require_parameters": True,
            "allow_fallbacks": False,
        }
        or endpoint.get("request_overrides")
        != {"top_p": 1.0, "reasoning": {"enabled": False}}
    ):
        raise RuntimeError("Step 10 endpoint drifted from the frozen protocol")
    split = _under(
        Path(str(config["probe_split"])), CAMPAIGN_ROOT, "Step 10 probe split"
    )
    actors = _under(
        Path(str(config["actor_candidates"])), CAMPAIGN_ROOT, "Step 10 actor candidates"
    )
    if sha256_file(split) != str(config["probe_split_sha256"]):
        raise RuntimeError("Step 10 probe split hash drifted")
    actor_summary = actors.with_name("summary.json")
    if (
        not actor_summary.is_file()
        or _read_json(actor_summary).get("status") != "complete"
    ):
        raise RuntimeError("Step 9 actor probe generation is incomplete")
    if sha256_file(actors) != str(_read_json(actor_summary)["candidates_sha256"]):
        raise RuntimeError("Step 9 actor candidate hash drifted")
    examples = read_jsonl(split)
    actor_rows = read_jsonl(actors)
    probe_ids = {str(row["id"]) for row in examples}
    grouped = Counter(str(row["id"]) for row in actor_rows)
    if (
        len(examples) != 100
        or len(probe_ids) != 100
        or len(actor_rows) != 400
        or set(grouped) != probe_ids
        or set(grouped.values()) != {4}
        or any(str(row.get("domain")) != domain for row in actor_rows)
    ):
        raise RuntimeError("Step 10 requires exactly probe100 x four A0 rollouts")
    output_root = _under(
        Path(str(config["output_root"])), STEP_ROOT, "Step 10 screen output"
    )
    if (
        domain not in output_root.parts
        or str(candidate["slug"]) not in output_root.parts
    ):
        raise ValueError(
            "Step 10 screen output does not match its candidate-domain pair"
        )
    return config, candidate, examples


def _messages_have_reasoning(raw: dict[str, Any]) -> bool:
    try:
        message = raw["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return True
    return bool(message.get("reasoning") or message.get("reasoning_details"))


def _validate_route(raw: dict[str, Any], candidate: dict[str, Any]) -> None:
    if (
        raw.get("model") != candidate["model_id"]
        or raw.get("provider") != candidate["provider"]
    ):
        raise RuntimeError("misrouted_response")


def _parse_judgment(raw: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    _validate_route(raw, candidate)
    parsed = binary_probability_from_logprobs(raw)
    if not parsed.get("ok"):
        raise RuntimeError(f"judge_logprob_contract:{parsed.get('error')}")
    judge_token = response_text(raw).strip()
    argmax = "1" if float(parsed["logp_1"]) > float(parsed["logp_0"]) else "0"
    return {
        "p_own": float(parsed["p1"]),
        "logp_0": float(parsed["logp_0"]),
        "logp_1": float(parsed["logp_1"]),
        "binary_label_mass": float(parsed["binary_label_mass"]),
        "judge_token": judge_token,
        "argmax_token": argmax,
        "emitted_argmax_consistent": judge_token == argmax,
        "raw_model": raw.get("model"),
        "raw_provider": raw.get("provider"),
        "reasoning_output_absent": not _messages_have_reasoning(raw),
        "usage": raw.get("usage") or {},
    }


def _request_with_wal(
    *,
    client: httpx.Client,
    headers: dict[str, str],
    payload: dict[str, Any],
    logical_key: str,
    prompt_sha256: str,
    event_path: Path,
    receipt_path: Path,
    candidate: dict[str, Any],
    last_request: list[float],
    minimum_interval: float,
    max_attempts: int,
    max_retry_wait: float,
    require_binary: bool,
) -> tuple[dict[str, Any], int]:
    http_requests = 0
    last_error: str | None = None
    payload = dict(payload)
    for attempt in range(1, max_attempts + 1):
        wait = minimum_interval - (time.monotonic() - last_request[0])
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
        last_request[0] = time.monotonic()
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
                f"Ambiguous Step 10 transport for {logical_key}"
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
                time.sleep(min(5.0 * 2 ** (attempt - 1), max_retry_wait))
                continue
            break
        if response.status_code >= 400:
            append_event(
                event_path,
                {
                    "event": "REQUEST_FAILED_KNOWN",
                    "logical_key": logical_key,
                    "attempt": attempt,
                    "status_code": response.status_code,
                    "time_unix": time.time(),
                },
            )
            response.raise_for_status()
        try:
            raw = response.json()
        except ValueError:
            raw = None
        if not isinstance(raw, dict):
            last_error = "invalid_json_response"
            append_event(
                event_path,
                {
                    "event": "INVALID_API_RESPONSE",
                    "logical_key": logical_key,
                    "attempt": attempt,
                    "error": last_error,
                    "retryable": True,
                    "time_unix": time.time(),
                },
            )
            if attempt < max_attempts:
                time.sleep(min(2.0 * attempt, max_retry_wait))
                continue
            break
        if (
            raw.get("model") != candidate["model_id"]
            or raw.get("provider") != candidate["provider"]
        ):
            last_error = "misrouted_response"
            append_event(
                event_path,
                {
                    "event": "INVALID_API_RESPONSE",
                    "logical_key": logical_key,
                    "attempt": attempt,
                    "error": last_error,
                    "retryable": True,
                    "raw_model": raw.get("model"),
                    "raw_provider": raw.get("provider"),
                    "time_unix": time.time(),
                },
            )
            if attempt < max_attempts:
                time.sleep(min(2.0 * attempt, max_retry_wait))
                continue
            break
        if require_binary:
            parsed = binary_probability_from_logprobs(raw)
            if not parsed.get("ok"):
                last_error = f"judge_logprob_contract:{parsed.get('error')}"
                retryable = parsed.get("error") in {
                    "missing_or_invalid_choices",
                    "missing_logprobs",
                    "missing_logprobs_content",
                    "missing_binary_label_in_top_logprobs",
                }
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
                    if parsed.get("error") == "missing_binary_label_in_top_logprobs":
                        payload["top_logprobs"] = 20
                    time.sleep(min(2.0 * attempt, max_retry_wait))
                    continue
                break
        receipt = {
            "event": "API_RESPONSE_RECEIVED",
            "logical_key": logical_key,
            "attempt": attempt,
            "prompt_sha256": prompt_sha256,
            "payload_sha256": payload_hash,
            "openrouter_response_id": raw.get("id"),
            "raw_response": raw,
            "time_unix": time.time(),
        }
        append_event(receipt_path, receipt)
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
        return raw, http_requests
    raise RuntimeError(f"Step 10 request failed for {logical_key}: {last_error}")


def run_screen_attempt(
    *, config_path: Path, run_id: str, key_pool: Path, key_name: str, lock_dir: Path
) -> dict[str, Any]:
    config, candidate, examples = validate_screen_config(config_path)
    domain = str(config["domain"])
    output = Path(str(config["output_root"])) / "runs" / run_id
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    event_path = output / "events.jsonl"
    receipt_path = output / "api_receipts.jsonl"
    response_path = output / "responses.jsonl"
    responses = read_jsonl(response_path) if response_path.is_file() else []
    done = {str(row["logical_key"]): row for row in responses}
    receipt_rows = read_jsonl(receipt_path) if receipt_path.is_file() else []
    receipts = {str(row["logical_key"]): row for row in receipt_rows}
    if len(done) != len(responses) or len(receipts) != len(receipt_rows):
        raise RuntimeError("Duplicate Step 10 responses or receipts")
    states: dict[str, str] = {}
    for row in read_jsonl(event_path) if event_path.is_file() else []:
        states[str(row["logical_key"])] = str(row["event"])
    ambiguous = sorted(
        key
        for key, state in states.items()
        if state in {"REQUEST_STARTED", "REQUEST_AMBIGUOUS"}
        and key not in done
        and key not in receipts
    )
    if ambiguous:
        raise RuntimeError(f"Fail-closed ambiguous Step 10 requests: {ambiguous[:3]}")

    actor_rows = read_jsonl(Path(str(config["actor_candidates"])))
    tables = (
        Path(str(config["scoring"]["spider_tables_json"]))
        if domain == "spider"
        else None
    )
    prompts = prompts_for_domain(domain, examples, tables_json=tables)["a0"]
    prompt_by_id = {
        str(row["id"]): prompt for row, prompt in zip(examples, prompts, strict=True)
    }
    tasks: list[dict[str, Any]] = []
    for actor in actor_rows:
        row_id = str(actor["id"])
        tasks.append(
            {
                "logical_key": f"judge::actor::{actor['logical_key']}",
                "phase": "judge",
                "source": "actor",
                "id": row_id,
                "source_index": int(actor["source_index"]),
                "rollout_index": int(actor["rollout_index"]),
                "task_prompt": prompt_by_id[row_id],
                "response": str(actor["response"]),
            }
        )
    for row in examples:
        row_id = str(row["id"])
        tasks.append(
            {
                "logical_key": f"own::{domain}::{row_id}",
                "phase": "own",
                "source": "own",
                "id": row_id,
                "source_index": int(row["source_index"]),
                "rollout_index": 0,
                "task_prompt": prompt_by_id[row_id],
            }
        )
    # The first operation is a one-call logprob canary on an A0 response.
    # Remaining actor judgments follow, then own generations and own judgments.
    actor_tasks = tasks[:400]
    own_tasks = tasks[400:]
    ordered = actor_tasks + own_tasks
    secret = load_named_secret(key_pool, key_name)
    headers = {
        "Authorization": f"Bearer {secret}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "IAR campaign Step 10 discriminator screen",
        "HTTP-Referer": "https://local.cluster/indistinguishability-as-a-reward",
    }
    transport = config["transport"]
    http_requests = 0
    started = time.time()
    last_request = [0.0]

    def task_prompt(task: dict[str, Any]) -> str:
        if task["phase"] == "judge":
            return domain_attribution_prompt(
                domain, str(task["task_prompt"]), str(task["response"])
            )
        return str(task["task_prompt"])

    def recover(task: dict[str, Any]) -> dict[str, Any] | None:
        key = str(task["logical_key"])
        if key in done:
            return done[key]
        if key not in receipts:
            return None
        receipt = receipts[key]
        raw = dict(receipt["raw_response"])
        if receipt.get("prompt_sha256") != sha256_text(task_prompt(task)):
            raise RuntimeError(f"Step 10 receipt prompt drift: {key}")
        if task["phase"] == "own":
            _validate_route(raw, candidate)
            row = {
                **task,
                "response": response_text(raw).strip(),
                "response_sha256": sha256_text(response_text(raw).strip()),
                "raw_model": raw.get("model"),
                "raw_provider": raw.get("provider"),
                "finish_reason": raw["choices"][0].get("finish_reason"),
                "reasoning_output_absent": not _messages_have_reasoning(raw),
                "usage": raw.get("usage") or {},
                "recovered_from_receipt": True,
            }
        else:
            row = {
                **task,
                **_parse_judgment(raw, candidate),
                "recovered_from_receipt": True,
            }
        responses.append(row)
        write_jsonl(response_path, responses)
        done[key] = row
        append_event(
            event_path,
            {
                "event": "RESPONSE_RECOVERED_FROM_RECEIPT",
                "logical_key": key,
                "time_unix": time.time(),
            },
        )
        return row

    with (
        key_lock(lock_dir, key_name),
        httpx.Client(timeout=float(transport["timeout_seconds"])) as client,
    ):
        # Canary first. It is also the first planned actor judgment.
        canary_task = ordered[0]
        if str(canary_task["logical_key"]) not in done and recover(canary_task) is None:
            prompt = task_prompt(canary_task)
            payload = make_chat_payload(
                model_id=str(candidate["model_id"]),
                prompt=prompt,
                system_prompt=None,
                max_tokens=1,
                temperature=0,
                logprobs=True,
                top_logprobs=5,
                provider_routing=config["endpoint"]["provider_routing"],
                request_overrides=config["endpoint"]["request_overrides"],
            )
            raw, used = _request_with_wal(
                client=client,
                headers=headers,
                payload=payload,
                logical_key=str(canary_task["logical_key"]),
                prompt_sha256=sha256_text(prompt),
                event_path=event_path,
                receipt_path=receipt_path,
                candidate=candidate,
                last_request=last_request,
                minimum_interval=float(transport["minimum_request_interval_seconds"]),
                max_attempts=int(transport["max_attempts_per_call"]),
                max_retry_wait=float(transport["max_retry_wait_seconds"]),
                require_binary=True,
            )
            http_requests += used
            parsed = _parse_judgment(raw, candidate)
            if not parsed["reasoning_output_absent"]:
                raise RuntimeError(
                    "Step 10 canary returned reasoning despite disabled reasoning"
                )
            row = {**canary_task, **parsed, "recovered_from_receipt": False}
            responses.append(row)
            write_jsonl(response_path, responses)
            done[str(canary_task["logical_key"])] = row
        canary = done[str(canary_task["logical_key"])]
        if not bool(canary["reasoning_output_absent"]):
            raise RuntimeError(
                "Step 10 canary returned reasoning despite disabled reasoning"
            )
        write_json(
            output / "endpoint_canary.json",
            {
                "schema_version": 1,
                "decision": "GO",
                "model_id": candidate["model_id"],
                "provider": candidate["provider"],
                "logprobs_ok": True,
                "reasoning_output_absent": bool(canary["reasoning_output_absent"]),
                "binary_label_mass": float(canary["binary_label_mass"]),
                "official_test_accessed": False,
            },
        )
        for task in ordered[1:]:
            key = str(task["logical_key"])
            if task["phase"] == "own":
                row = done.get(key) or recover(task)
                if row is None:
                    prompt = task_prompt(task)
                    payload = make_chat_payload(
                        model_id=str(candidate["model_id"]),
                        prompt=prompt,
                        system_prompt=None,
                        max_tokens=768,
                        temperature=0,
                        logprobs=False,
                        provider_routing=config["endpoint"]["provider_routing"],
                        request_overrides=config["endpoint"]["request_overrides"],
                    )
                    raw, used = _request_with_wal(
                        client=client,
                        headers=headers,
                        payload=payload,
                        logical_key=key,
                        prompt_sha256=sha256_text(prompt),
                        event_path=event_path,
                        receipt_path=receipt_path,
                        candidate=candidate,
                        last_request=last_request,
                        minimum_interval=float(
                            transport["minimum_request_interval_seconds"]
                        ),
                        max_attempts=int(transport["max_attempts_per_call"]),
                        max_retry_wait=float(transport["max_retry_wait_seconds"]),
                        require_binary=False,
                    )
                    http_requests += used
                    text = response_text(raw).strip()
                    row = {
                        **task,
                        "response": text,
                        "response_sha256": sha256_text(text),
                        "raw_model": raw.get("model"),
                        "raw_provider": raw.get("provider"),
                        "finish_reason": raw["choices"][0].get("finish_reason"),
                        "reasoning_output_absent": not _messages_have_reasoning(raw),
                        "usage": raw.get("usage") or {},
                        "recovered_from_receipt": False,
                    }
                    responses.append(row)
                    write_jsonl(response_path, responses)
                    done[key] = row
                # Immediately obtain the self-attribution score for the generated answer.
                text = str(row["response"])
                judge_task = {
                    **task,
                    "logical_key": f"judge::own::{domain}::{task['id']}",
                    "phase": "judge",
                    "response": text,
                }
                judge_key = str(judge_task["logical_key"])
                if judge_key not in done and recover(judge_task) is None:
                    judge_prompt = task_prompt(judge_task)
                    judge_payload = make_chat_payload(
                        model_id=str(candidate["model_id"]),
                        prompt=judge_prompt,
                        system_prompt=None,
                        max_tokens=1,
                        temperature=0,
                        logprobs=True,
                        top_logprobs=5,
                        provider_routing=config["endpoint"]["provider_routing"],
                        request_overrides=config["endpoint"]["request_overrides"],
                    )
                    raw_j, used_j = _request_with_wal(
                        client=client,
                        headers=headers,
                        payload=judge_payload,
                        logical_key=judge_key,
                        prompt_sha256=sha256_text(judge_prompt),
                        event_path=event_path,
                        receipt_path=receipt_path,
                        candidate=candidate,
                        last_request=last_request,
                        minimum_interval=float(
                            transport["minimum_request_interval_seconds"]
                        ),
                        max_attempts=int(transport["max_attempts_per_call"]),
                        max_retry_wait=float(transport["max_retry_wait_seconds"]),
                        require_binary=True,
                    )
                    http_requests += used_j
                    judge_row = {
                        **judge_task,
                        **_parse_judgment(raw_j, candidate),
                        "recovered_from_receipt": False,
                    }
                    responses.append(judge_row)
                    write_jsonl(response_path, responses)
                    done[judge_key] = judge_row
                continue
            if key in done or recover(task) is not None:
                continue
            prompt = task_prompt(task)
            payload = make_chat_payload(
                model_id=str(candidate["model_id"]),
                prompt=prompt,
                system_prompt=None,
                max_tokens=1,
                temperature=0,
                logprobs=True,
                top_logprobs=5,
                provider_routing=config["endpoint"]["provider_routing"],
                request_overrides=config["endpoint"]["request_overrides"],
            )
            raw, used = _request_with_wal(
                client=client,
                headers=headers,
                payload=payload,
                logical_key=key,
                prompt_sha256=sha256_text(prompt),
                event_path=event_path,
                receipt_path=receipt_path,
                candidate=candidate,
                last_request=last_request,
                minimum_interval=float(transport["minimum_request_interval_seconds"]),
                max_attempts=int(transport["max_attempts_per_call"]),
                max_retry_wait=float(transport["max_retry_wait_seconds"]),
                require_binary=True,
            )
            parsed = _parse_judgment(raw, candidate)
            http_requests += used
            row = {
                **task,
                **parsed,
                "request_top_logprobs": payload["top_logprobs"],
                "recovered_from_receipt": False,
            }
            responses.append(row)
            write_jsonl(response_path, responses)
            done[key] = row

    expected = {str(row["logical_key"]) for row in actor_tasks}
    expected |= {str(row["logical_key"]) for row in own_tasks}
    expected |= {f"judge::own::{domain}::{row['id']}" for row in examples}
    if set(done) != expected:
        raise RuntimeError(
            f"Step 10 screen coverage mismatch: {len(done)} != {len(expected)}"
        )
    responses.sort(key=lambda row: str(row["logical_key"]))
    write_jsonl(response_path, responses)
    event_rows = read_jsonl(event_path)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "candidate_slug": candidate["slug"],
        "model_id": candidate["model_id"],
        "provider": candidate["provider"],
        "domain": domain,
        "run_id": run_id,
        "own_generations": 100,
        "actor_judgments": 400,
        "own_judgments": 100,
        "successful_api_outputs": len(responses),
        "http_requests_this_process": http_requests,
        "ambiguous_request_events": sum(
            row["event"] == "REQUEST_AMBIGUOUS" for row in event_rows
        ),
        "responses": str(response_path),
        "responses_sha256": sha256_file(response_path),
        "events": str(event_path),
        "events_sha256": sha256_file(event_path),
        "api_receipts": str(receipt_path),
        "api_receipts_sha256": sha256_file(receipt_path),
        "endpoint_canary": str(output / "endpoint_canary.json"),
        "endpoint_canary_sha256": sha256_file(output / "endpoint_canary.json"),
        "usage": {
            key: sum(
                float((row.get("usage") or {}).get(key, 0) or 0) for row in responses
            )
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
        },
        "elapsed_seconds": time.time() - started,
        "key_fingerprint": hashlib.sha256(secret.encode()).hexdigest()[:16],
        "official_test_accessed": False,
    }
    write_json(summary_path, summary)
    return summary


def score_screen_attempt(
    *, config_path: Path, run_id: str, attempt: int, score_workers: int
) -> dict[str, Any]:
    if attempt <= 0 or score_workers <= 0:
        raise ValueError("Invalid Step 10 screen scoring attempt")
    config, candidate, examples = validate_screen_config(config_path)
    domain = str(config["domain"])
    run_root = Path(str(config["output_root"])) / "runs" / run_id
    screen = _read_json(run_root / "summary.json")
    responses = read_jsonl(Path(str(screen["responses"])))
    own = [row for row in responses if row.get("phase") == "own"]
    judgments = [row for row in responses if row.get("phase") == "judge"]
    if len(own) != 100 or len(judgments) != 500:
        raise RuntimeError("Step 10 screen API phase is incomplete")
    judge_by_key = {str(row["logical_key"]): row for row in judgments}
    candidates: list[dict[str, Any]] = []
    for row in own:
        key = f"judge::own::{domain}::{row['id']}"
        candidates.append(
            {
                "logical_key": f"candidate::own::{domain}::{row['id']}",
                "source": "own",
                "id": str(row["id"]),
                "source_index": int(row["source_index"]),
                "rollout_index": 0,
                "domain": domain,
                "response": str(row["response"]),
                "p_own": float(judge_by_key[key]["p_own"]),
            }
        )
    for row in judgments:
        if row.get("source") != "actor":
            continue
        candidates.append(
            {
                "logical_key": str(row["logical_key"]).replace(
                    "judge::", "candidate::", 1
                ),
                "source": "actor",
                "id": str(row["id"]),
                "source_index": int(row["source_index"]),
                "rollout_index": int(row["rollout_index"]),
                "domain": domain,
                "response": str(row["response"]),
                "p_own": float(row["p_own"]),
            }
        )
    example_by_id = {str(row["id"]): row for row in examples}
    aligned_examples = [example_by_id[str(row["id"])] for row in candidates]
    if domain in {"commonsenseqa", "gsm8k"}:
        _score_simple(domain, aligned_examples, candidates)
        scored_rows = candidates
        scoring_protocol: dict[str, Any] | None = None
    elif domain == "kodcode":
        payloads = [
            (
                str(row["response"]),
                str(example_by_id[str(row["id"])]["hidden_tests"]),
                int(config["scoring"]["coding_timeout_seconds"]),
            )
            for row in candidates
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=score_workers) as pool:
            scores = list(pool.map(_score_kodcode, payloads))
        for row, score in zip(candidates, scores, strict=True):
            row["score"] = score
        scored_rows = candidates
        scoring_protocol = {
            "metric": "frozen_hidden_test_execution",
            "timeout_seconds": int(config["scoring"]["coding_timeout_seconds"]),
        }
    else:
        database_root = Path(str(config["scoring"]["spider_database_root"]))
        with concurrent.futures.ThreadPoolExecutor(max_workers=score_workers) as pool:
            scores = list(
                pool.map(
                    _score_spider_one,
                    [
                        (example, row, database_root)
                        for example, row in zip(
                            aligned_examples, candidates, strict=True
                        )
                    ],
                )
            )
        for row, result in zip(candidates, scores, strict=True):
            row.update(result)
        scored_rows = candidates
        scoring_protocol = {
            "metric": "frozen_single_database_execution",
            "database_root": str(database_root),
        }
    output = run_root / f"score_attempt{attempt:03d}"
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "candidates.jsonl", candidates)
    result = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "candidate_slug": candidate["slug"],
        "domain": domain,
        "run_id": run_id,
        "n": len(candidates),
        "scored_n": len(scored_rows),
        "correct": sum(bool(row["score"]["correct"]) for row in scored_rows),
        "task_utility": statistics.fmean(
            bool(row["score"]["correct"]) for row in scored_rows
        ),
        "scoring_protocol": scoring_protocol,
        "candidates": str(output / "candidates.jsonl"),
        "candidates_sha256": sha256_file(output / "candidates.jsonl"),
        "official_test_accessed": False,
    }
    write_json(output / "summary.json", result)
    return result


def collect_screen_gate(
    *, config_paths: list[Path], run_id: str, scoring_attempt: int
) -> dict[str, Any]:
    protocol = _read_json(PROTOCOL)
    thresholds = protocol["gate_thresholds"]
    selected = _validate_selection()
    expected_pairs = [
        (str(candidate["slug"]), domain)
        for candidate in protocol["candidate_discriminators"]
        for domain in selected
    ]
    configs: dict[tuple[str, str], dict[str, Any]] = {}
    for path in config_paths:
        config, candidate, _ = validate_screen_config(path)
        configs[(str(candidate["slug"]), str(config["domain"]))] = config
    if set(configs) != set(expected_pairs):
        raise RuntimeError(
            "Step 10 screen gate does not cover all four candidate-domain pairs"
        )
    pairs: dict[str, Any] = {}
    passing: list[str] = []
    for pair_index, (slug, domain) in enumerate(expected_pairs):
        config = configs[(slug, domain)]
        run_root = Path(str(config["output_root"])) / "runs" / run_id
        screen = _read_json(run_root / "summary.json")
        scored_summary = _read_json(
            run_root / f"score_attempt{scoring_attempt:03d}/summary.json"
        )
        scored = read_jsonl(Path(str(scored_summary["candidates"])))
        judgments = [
            row
            for row in read_jsonl(Path(str(screen["responses"])))
            if row.get("phase") == "judge"
        ]
        own = [row for row in scored if row["source"] == "own"]
        actor = [row for row in scored if row["source"] == "actor"]
        own_p = [float(row["p_own"]) for row in own]
        actor_p = [float(row["p_own"]) for row in actor]
        own_u = [float(bool(row["score"]["correct"])) for row in own]
        actor_u = [float(bool(row["score"]["correct"])) for row in actor]
        group_p: dict[str, list[float]] = defaultdict(list)
        group_u: dict[str, list[float]] = defaultdict(list)
        for row in actor:
            group_p[str(row["id"])].append(float(row["p_own"]))
            group_u[str(row["id"])].append(float(bool(row["score"]["correct"])))
        own_by_id = {str(row["id"]): row for row in own}
        identity_deltas = [
            float(own_by_id[row_id]["p_own"]) - statistics.fmean(values)
            for row_id, values in group_p.items()
        ]
        utility_deltas = [
            float(bool(own_by_id[row_id]["score"]["correct"]))
            - statistics.fmean(group_u[row_id])
            for row_id in group_u
        ]
        correctness_positive = [
            float(row["p_own"]) for row in scored if bool(row["score"]["correct"])
        ]
        correctness_negative = [
            float(row["p_own"]) for row in scored if not bool(row["score"]["correct"])
        ]
        correctness_auc = (
            auc(correctness_positive, correctness_negative)
            if correctness_positive and correctness_negative
            else None
        )
        ranges = [max(values) - min(values) for values in group_p.values()]
        masses = [float(row["binary_label_mass"]) for row in judgments]
        mass_gate = binary_mass_gate(masses, thresholds)
        endpoint_canary = _read_json(Path(str(screen["endpoint_canary"])))
        metrics = {
            "U_D": statistics.fmean(own_u),
            "U_A0": statistics.fmean(actor_u),
            "gamma_0": statistics.fmean(utility_deltas),
            "G_D0": statistics.fmean(identity_deltas),
            "AUC_identity": auc(own_p, actor_p),
            "AUC_correct": correctness_auc,
            "actor_p_own_std": statistics.stdev(actor_p),
            "fraction_groups_with_required_range": statistics.fmean(
                value >= float(thresholds["minimum_within_group_range"])
                for value in ranges
            ),
            "emitted_argmax_consistency": statistics.fmean(
                bool(row["emitted_argmax_consistent"]) for row in judgments
            ),
            "identity_runway_paired": paired_summary(
                identity_deltas,
                seed=23 + pair_index,
                replicates=int(protocol["screen"]["bootstrap_replicates"]),
            ),
            "competence_gap_paired": paired_summary(
                utility_deltas,
                seed=123 + pair_index,
                replicates=int(protocol["screen"]["bootstrap_replicates"]),
            ),
            "binary_label_mass_gate": mass_gate,
        }
        checks = {
            "endpoint_canary": endpoint_canary.get("decision") == "GO"
            and endpoint_canary.get("reasoning_output_absent") is True,
            "complete_coverage": len(own)
            == int(thresholds["complete_own_generation_coverage"])
            and len(actor) == int(thresholds["complete_actor_candidate_coverage"])
            and len(judgments) == int(thresholds["complete_judge_call_coverage"]),
            "own_generation_complete": all(
                row.get("finish_reason") == "stop"
                and bool(row.get("reasoning_output_absent"))
                for row in read_jsonl(Path(str(screen["responses"])))
                if row.get("phase") == "own"
            ),
            "no_ambiguous_requests": int(screen["ambiguous_request_events"])
            <= int(thresholds["maximum_ambiguous_requests"]),
            "endpoint_pinned": {row["raw_model"] for row in judgments}
            == {config["endpoint"]["model_id"]}
            and {row["raw_provider"] for row in judgments}
            == {config["endpoint"]["expected_provider"]},
            "reasoning_disabled": all(
                bool(row["reasoning_output_absent"]) for row in judgments
            ),
            "binary_label_support": bool(mass_gate["pass"]),
            "emitted_argmax_consistency": metrics["emitted_argmax_consistency"]
            >= float(thresholds["minimum_emitted_argmax_consistency"]),
            "competence_gap": metrics["gamma_0"]
            >= float(thresholds["minimum_discriminator_task_utility_gap"]),
            "identity_auc": metrics["AUC_identity"]
            >= float(thresholds["minimum_identity_auc_own_vs_actor"]),
            "identity_runway": metrics["G_D0"]
            >= float(thresholds["minimum_identity_runway"]),
            "correctness_auc": correctness_auc is not None
            and correctness_auc >= float(thresholds["minimum_correctness_auc"]),
            "reward_variance": metrics["actor_p_own_std"]
            >= float(thresholds["minimum_actor_probability_standard_deviation"]),
            "within_group_variation": metrics["fraction_groups_with_required_range"]
            >= float(thresholds["minimum_fraction_groups_with_range"]),
        }
        pair_id = f"{slug}__{domain}"
        decision = "GO" if all(checks.values()) else "NO_GO"
        if decision == "GO":
            passing.append(pair_id)
        pairs[pair_id] = {
            "candidate_slug": slug,
            "domain": domain,
            "decision": decision,
            "checks": checks,
            "metrics": metrics,
            "screen_summary": str(run_root / "summary.json"),
            "screen_summary_sha256": sha256_file(run_root / "summary.json"),
            "score_summary": str(
                run_root / f"score_attempt{scoring_attempt:03d}/summary.json"
            ),
            "score_summary_sha256": sha256_file(
                run_root / f"score_attempt{scoring_attempt:03d}/summary.json"
            ),
        }
    result = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "protocol": "campaign-discriminator-transfer-screen-gate-v1",
        "selected_domains": selected,
        "expected_pairs": [f"{slug}__{domain}" for slug, domain in expected_pairs],
        "passing_pairs": passing,
        "pairs": pairs,
        "progression_rule": "each candidate-domain pair progresses only when every frozen check passes",
        "official_test_accessed": False,
        "mbpp": "FORBIDDEN",
        "ifbench": "DEFERRED",
    }
    destination = STEP_ROOT / "results/discriminator_transfer_screen_gate_v1.json"
    if destination.exists():
        raise FileExistsError(destination)
    write_json(destination, result)
    return result


def validate_grpo_config(
    config: GrpoTrialConfig,
) -> tuple[str, str, Path, Path, dict[str, Any]]:
    raw = config.raw
    campaign = raw.get("campaign") or {}
    domain = str(campaign.get("domain", ""))
    slug = str(campaign.get("candidate_slug", ""))
    protocol = _read_json(PROTOCOL)
    candidate = _candidate_by_slug(protocol, slug)
    if domain not in _validate_selection():
        raise ValueError("Step 10 GRPO domain is not selected")
    serialized = json.dumps(raw, sort_keys=True).lower()
    if "mbpp" in serialized or "ifbench" in serialized or "ifeval" in serialized:
        raise ValueError("Forbidden benchmark in Step 10 GRPO config")
    if sha256_file(PROTOCOL) != str(campaign.get("transfer_protocol_sha256")):
        raise RuntimeError("Step 10 protocol hash drifted")
    if sha256_file(SELECTION) != str(campaign.get("domain_selection_sha256")):
        raise RuntimeError("Step 10 domain-selection hash drifted")
    gate_path = Path(str(campaign.get("screen_gate", "")))
    gate = _read_json(gate_path)
    pair_id = f"{slug}__{domain}"
    if (
        gate.get("status") != "complete"
        or pair_id not in gate.get("passing_pairs", [])
        or (gate.get("pairs") or {}).get(pair_id, {}).get("decision") != "GO"
        or gate.get("official_test_accessed") is not False
    ):
        raise RuntimeError("Step 10 GRPO pair did not pass every frozen screen gate")
    discriminator = raw["discriminator"]
    expected_prompt_version, expected_prompt_sha = _attribution_template(domain)
    if (
        discriminator.get("model_id") != candidate["model_id"]
        or discriminator.get("target_model_name") != candidate["target_model_name"]
        or discriminator.get("expected_provider") != candidate["provider"]
        or discriminator.get("attribution_prompt_version") != expected_prompt_version
        or discriminator.get("attribution_prompt_template_sha256")
        != expected_prompt_sha
        or discriminator.get("provider_routing")
        != {
            "order": [candidate["provider"]],
            "require_parameters": True,
            "allow_fallbacks": False,
        }
        or discriminator.get("request_overrides")
        != {"top_p": 1.0, "reasoning": {"enabled": False}}
    ):
        raise RuntimeError("Step 10 discriminator configuration drifted")
    reward = raw["reward"]
    training = raw["training"]
    if (
        config.prompt_count != 1000
        or config.num_epochs != 2
        or config.num_generations != 4
        or config.expected_reward_calls != 8000
        or str(reward["value"]) != "log_p_own"
        or int(reward["max_http_requests"]) != 40000
        or int(reward["max_attempts_per_call"]) != 5
        or int(reward["max_concurrency"]) != 1
        or int(training["max_steps"]) != 2000
        or int(training["per_device_train_batch_size"]) != 4
        or int(training["generation_batch_size"]) != 8
        or int(training["save_steps"]) != 500
        or float(training["learning_rate"]) != 3e-5
        or float(training["beta"]) != 0.003
        or float(training["temperature"]) != 0.6
        or str(training["loss_type"]) != "dr_grpo"
        or str(training["scale_rewards"]) != "group"
    ):
        raise ValueError("Step 10 GRPO hyperparameters drifted")
    train_file = Path(str(raw["dataset"]["split_file"]))
    validation_file = Path(str(raw["evaluation"]["validation_file"]))
    if sha256_file(train_file) != str(
        raw["dataset"]["split_file_sha256"]
    ) or sha256_file(validation_file) != str(
        raw["evaluation"]["validation_file_sha256"]
    ):
        raise RuntimeError("Step 10 frozen split hash drifted")
    if any(
        part.lower() in {"test", "sealed", "final_test"}
        for part in validation_file.parts
    ):
        raise ValueError("Step 10 cannot access official or sealed test data")
    expected_root = (
        "frozen_historical_seed23"
        if domain in {"commonsenseqa", "gsm8k"}
        else "frozen_seed23_v1"
    )
    expected_train = {
        "commonsenseqa": "train.jsonl",
        "gsm8k": "official_train.jsonl",
        "kodcode": "train_core1000.jsonl",
        "spider": "train_core1000.jsonl",
    }[domain]
    if train_file.parent.name != expected_root or train_file.name != expected_train:
        raise ValueError("Step 10 is not using the frozen train_core1000 IDs")
    if validation_file.name not in {"validation.jsonl", "validation300.jsonl"}:
        raise ValueError("Step 10 is not using validation300")
    return domain, slug, train_file, validation_file, candidate


def build_training_dataset(config: GrpoTrialConfig):
    from datasets import Dataset

    domain, _, train_file, _, _ = validate_grpo_config(config)
    examples = read_jsonl(train_file)
    if len(examples) != 1000 or len({str(row["id"]) for row in examples}) != 1000:
        raise RuntimeError("Step 10 requires 1000 unique train IDs")
    tables = (
        Path(str(config.raw["evaluation"]["spider_tables_json"]))
        if domain == "spider"
        else None
    )
    prompts = prompts_for_domain(domain, examples, tables_json=tables)["a0"]
    rows = [
        {
            "prompt": [{"role": "user", "content": prompt}],
            "task_statement": prompt,
            "task_id": str(example["id"]),
            "dataset_index": int(example["source_index"]),
        }
        for example, prompt in zip(examples, prompts, strict=True)
    ]
    return Dataset.from_list(rows), rows


def run_training_attempt(
    *, config_path: Path, run_id: str, key_pool: Path, key_name: str, lock_dir: Path
) -> dict[str, Any]:
    config = load_grpo_config(config_path)
    domain, slug, train_file, validation_file, candidate = validate_grpo_config(config)
    experiment_root = _under(
        config.resolve(config.raw["paths"]["experiments"]),
        STEP_ROOT,
        "Step 10 run root",
    )
    checkpoint_root = _under(
        config.resolve(config.raw["paths"]["checkpoints"]),
        STEP_ROOT,
        "Step 10 checkpoint root",
    )
    experiment_dir = experiment_root / run_id
    checkpoint_dir = checkpoint_root / run_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    (experiment_dir / "config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    initialize_parameter_record(
        experiment_dir, resolved_config=config.raw, source_config=config_path
    )
    dataset, rows = build_training_dataset(config)
    write_jsonl(experiment_dir / "train_examples.jsonl", rows)
    runtime = LiveOpenRouterReward(
        settings=settings_from_config(config.raw),
        log_path=experiment_dir / "reward_calls.jsonl",
        api_key=load_named_secret(key_pool, key_name),
        prompt_builder=lambda task, response: domain_attribution_prompt(
            domain, task, response
        ),
        prompt_template_hash=_attribution_template(domain)[1],
        system_prompt_builder=lambda _name: None,
        candidate_normalizer=strip_code_fences if domain == "kodcode" else str.strip,
    )
    try:
        with key_lock(lock_dir, key_name):
            result, history = run_training(
                config, dataset, checkpoint_dir, experiment_dir, runtime
            )
        runtime.assert_complete()
        write_jsonl(experiment_dir / "trainer_log_history.jsonl", history)
        stats = runtime.reward_statistics()
        observed = stats.get("by_task") or {}
        if set(observed) != {str(row["task_id"]) for row in rows} or any(
            int(value.get("n", -1)) != 8 for value in observed.values()
        ):
            raise RuntimeError("Step 10 reward coverage differs from 1000 x G4 x 2")
        gradients = [float(row["grad_norm"]) for row in history if "grad_norm" in row]
        if not gradients or any(not math.isfinite(value) for value in gradients):
            raise RuntimeError("Step 10 gradient audit is missing or non-finite")
        summary = {
            "schema_version": 1,
            "status": "complete",
            "completed_at": utc_now(),
            "run_id": run_id,
            "domain": domain,
            "candidate_slug": slug,
            "model_id": candidate["model_id"],
            "provider": candidate["provider"],
            "dataset_rows": len(rows),
            "num_epochs": 2,
            "expected_reward_calls": 8000,
            "reward_audit": runtime.audit_summary(),
            "reward_statistics": stats,
            "trainer_metrics": result.metrics,
            "optimization_audit": {
                "logged_steps": len(gradients),
                "nonzero_gradient_steps": sum(value > 0 for value in gradients),
                "min_grad_norm": min(gradients),
                "max_grad_norm": max(gradients),
            },
            "training_split": str(train_file),
            "training_split_sha256": sha256_file(train_file),
            "validation_split": str(validation_file),
            "validation_split_sha256": sha256_file(validation_file),
            "checkpoint_dir": str(checkpoint_dir),
            "official_test_accessed": False,
        }
        write_json(experiment_dir / "summary.json", summary)
        return summary
    except Exception as exc:
        write_json(
            experiment_dir / "failure.json",
            {
                "schema_version": 1,
                "status": "failed",
                "failed_at": utc_now(),
                "run_id": run_id,
                "domain": domain,
                "candidate_slug": slug,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "reward_audit": runtime.audit_summary(),
                "official_test_accessed": False,
            },
        )
        raise


def generate_validation(
    *, config_path: Path, run_id: str, attempt: int
) -> dict[str, Any]:
    if attempt <= 0:
        raise ValueError("Invalid Step 10 generation attempt")
    config = load_grpo_config(config_path)
    domain, slug, _, validation_file, _ = validate_grpo_config(config)
    run_root = _under(
        config.resolve(config.raw["paths"]["experiments"]) / run_id,
        STEP_ROOT,
        "Step 10 run",
    )
    checkpoint_root = _under(
        config.resolve(config.raw["paths"]["checkpoints"]) / run_id,
        STEP_ROOT,
        "Step 10 checkpoints",
    )
    if _read_json(run_root / "summary.json").get("status") != "complete":
        raise RuntimeError("Step 10 validation cannot precede training completion")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from iar_grpo.numerics import configure_stable_sdpa

    examples = read_jsonl(validation_file)
    if len(examples) != 300:
        raise RuntimeError("Step 10 validation requires validation300")
    tables = (
        Path(str(config.raw["evaluation"]["spider_tables_json"]))
        if domain == "spider"
        else None
    )
    prompts = prompts_for_domain(domain, examples, tables_json=tables)["a0"]
    random.seed(23)
    torch.manual_seed(23)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(23)
    sdpa = configure_stable_sdpa(torch)
    tokenizer = AutoTokenizer.from_pretrained(
        config.actor_source(),
        local_files_only=True,
        trust_remote_code=False,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = None
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            config.actor_source(),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        for step in CHECKPOINTS[1:]:
            output = (
                run_root
                / "evaluations/validation"
                / f"step_{step:04d}"
                / f"generation_attempt{attempt:03d}"
            )
            output.mkdir(parents=True, exist_ok=False)
            adapter = _adapter_for_step(checkpoint_root, step, CHECKPOINTS[-1])
            if adapter is None:
                raise RuntimeError("Step 10 trained checkpoint resolved to A0")
            model = PeftModel.from_pretrained(base_model, adapter, is_trainable=False)
            model.eval()
            rows: list[dict[str, Any]] = []
            batch_size = int(config.raw["evaluation"]["batch_size"])
            for start in range(0, len(examples), batch_size):
                batch_examples = examples[start : start + batch_size]
                batch_prompts = prompts[start : start + batch_size]
                encoded = tokenizer.apply_chat_template(
                    [[{"role": "user", "content": prompt}] for prompt in batch_prompts],
                    tokenize=True,
                    add_generation_prompt=True,
                    padding=True,
                    return_tensors="pt",
                    return_dict=True,
                ).to(model.device)
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=int(config.raw["evaluation"]["max_new_tokens"]),
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                prompt_length = encoded["input_ids"].shape[1]
                for example, output_ids in zip(batch_examples, generated, strict=True):
                    completion = output_ids[prompt_length:]
                    rows.append(
                        {
                            "id": str(example["id"]),
                            "source_index": int(example["source_index"]),
                            "domain": domain,
                            "candidate_slug": slug,
                            "step": step,
                            "run_id": run_id,
                            "response": tokenizer.decode(
                                completion, skip_special_tokens=True
                            ).strip(),
                            "completion_tokens": len(completion),
                            "hit_max_new_tokens": len(completion)
                            >= int(config.raw["evaluation"]["max_new_tokens"]),
                        }
                    )
            write_jsonl(output / "candidates.jsonl", rows)
            write_json(
                output / "summary.json",
                {
                    "schema_version": 1,
                    "status": "complete",
                    "completed_at": utc_now(),
                    "domain": domain,
                    "candidate_slug": slug,
                    "run_id": run_id,
                    "step": step,
                    "n": len(rows),
                    "truncated": sum(bool(row["hit_max_new_tokens"]) for row in rows),
                    "candidate_file_sha256": sha256_file(output / "candidates.jsonl"),
                    "validation_file": str(validation_file),
                    "validation_file_sha256": sha256_file(validation_file),
                    "adapter": str(adapter),
                    "sdpa": sdpa,
                    "official_test_accessed": False,
                },
            )
            model.unload()
            del model
            gc.collect()
            torch.cuda.empty_cache()
        return {
            "status": "complete",
            "domain": domain,
            "candidate_slug": slug,
            "steps": list(CHECKPOINTS[1:]),
            "official_test_accessed": False,
        }
    finally:
        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def score_validation(
    *,
    config_path: Path,
    run_id: str,
    generation_attempt: int,
    scoring_attempt: int,
    score_workers: int,
) -> dict[str, Any]:
    config = load_grpo_config(config_path)
    domain, slug, _, validation_file, _ = validate_grpo_config(config)
    run_root = _under(
        config.resolve(config.raw["paths"]["experiments"]) / run_id,
        STEP_ROOT,
        "Step 10 run",
    )
    examples = read_jsonl(validation_file)
    summaries = []
    for step in CHECKPOINTS[1:]:
        generation = (
            run_root
            / "evaluations/validation"
            / f"step_{step:04d}"
            / f"generation_attempt{generation_attempt:03d}"
        )
        candidates = read_jsonl(generation / "candidates.jsonl")
        generated = _read_json(generation / "summary.json")
        if (
            generated.get("status") != "complete"
            or generated.get("candidate_file_sha256")
            != sha256_file(generation / "candidates.jsonl")
            or [str(row["id"]) for row in candidates]
            != [str(row["id"]) for row in examples]
        ):
            raise RuntimeError(f"Step 10 validation generation incomplete at {step}")
        output = (
            run_root
            / "evaluations/validation"
            / f"step_{step:04d}"
            / f"score_attempt{scoring_attempt:03d}"
        )
        output.mkdir(parents=True, exist_ok=False)
        protocol: dict[str, Any] | None = None
        scored = candidates
        if domain in {"commonsenseqa", "gsm8k"}:
            _score_simple(domain, examples, candidates)
        elif domain == "kodcode":
            scored, protocol = _score_kodcode_rows(
                examples,
                candidates,
                scoring_protocol_path=Path(
                    str(config.raw["evaluation"]["kodcode_scoring_protocol"])
                ),
                score_workers=score_workers,
                timeout_seconds=int(config.raw["evaluation"]["coding_timeout_seconds"]),
            )
        else:
            protocol = _score_spider_rows(
                examples,
                candidates,
                split_manifest_path=Path(
                    str(config.raw["evaluation"]["spider_split_manifest"])
                ),
                scoring_protocol_path=Path(
                    str(config.raw["evaluation"]["spider_scoring_protocol"])
                ),
                validation_file=validation_file,
                database_root=Path(
                    str(config.raw["evaluation"]["spider_database_root"])
                ),
                score_workers=score_workers,
            )
        write_jsonl(output / "candidates.jsonl", candidates)
        summary = {
            "schema_version": 1,
            "status": "complete",
            "completed_at": utc_now(),
            "domain": domain,
            "candidate_slug": slug,
            "run_id": run_id,
            "step": step,
            "n": len(candidates),
            "scored_n": len(scored),
            "correct": sum(bool(row["score"]["correct"]) for row in scored),
            "valid": sum(bool(row["score"].get("valid", True)) for row in scored),
            "task_utility": statistics.fmean(
                bool(row["score"]["correct"]) for row in scored
            ),
            "scoring_protocol": protocol,
            "candidate_file_sha256": sha256_file(output / "candidates.jsonl"),
            "validation_file": str(validation_file),
            "validation_file_sha256": sha256_file(validation_file),
            "official_test_accessed": False,
        }
        write_json(output / "summary.json", summary)
        summaries.append(summary)
    return {
        "status": "complete",
        "domain": domain,
        "candidate_slug": slug,
        "summaries": summaries,
        "official_test_accessed": False,
    }


def collect_pair_validation(
    *, config_path: Path, run_id: str, scoring_attempt: int
) -> dict[str, Any]:
    config = load_grpo_config(config_path)
    domain, slug, _, _, candidate = validate_grpo_config(config)
    run_root = _under(
        config.resolve(config.raw["paths"]["experiments"]) / run_id,
        STEP_ROOT,
        "Step 10 run",
    )
    registry = _read_json(Path(str(config.raw["evaluation"]["baseline_registry"])))
    a0 = registry["baselines"][domain]["a0"]
    evaluations = [
        {
            "step": 0,
            "task_utility": float(a0["task_utility"]),
            "summary": a0["summary"],
            "summary_sha256": a0["summary_sha256"],
        }
    ]
    for step in CHECKPOINTS[1:]:
        path = (
            run_root
            / "evaluations/validation"
            / f"step_{step:04d}"
            / f"score_attempt{scoring_attempt:03d}/summary.json"
        )
        value = _read_json(path)
        if value.get("status") != "complete":
            raise RuntimeError(f"Incomplete Step 10 validation at {step}")
        evaluations.append(
            {
                "step": step,
                "task_utility": float(value["task_utility"]),
                "summary": str(path),
                "summary_sha256": sha256_file(path),
            }
        )
    selected = max(
        evaluations, key=lambda row: (float(row["task_utility"]), -int(row["step"]))
    )
    result = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "protocol": "campaign-step10-validation-selection-v1",
        "candidate_slug": slug,
        "model_id": candidate["model_id"],
        "provider": candidate["provider"],
        "domain": domain,
        "run_id": run_id,
        "selected_step": int(selected["step"]),
        "selected_task_utility": float(selected["task_utility"]),
        "validation_gain_vs_a0": float(selected["task_utility"])
        - float(a0["task_utility"]),
        "evaluations": evaluations,
        "checkpoint_selection": "maximize validation task utility; earliest checkpoint on ties",
        "official_test_accessed": False,
        "mbpp": "FORBIDDEN",
    }
    output = run_root / "evaluations/validation/selection.json"
    if output.exists():
        raise FileExistsError(output)
    write_json(output, result)
    return result


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or statistics.pstdev(x) == 0 or statistics.pstdev(y) == 0:
        return None
    mx, my = statistics.fmean(x), statistics.fmean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / math.sqrt(
        sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)
    )


def _ranks(values: list[float]) -> list[float]:
    result = [0.0] * len(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            result[order[position]] = rank
        start = end
    return result


def _leave_one_out_linear(x: list[float], y: list[float]) -> dict[str, Any] | None:
    if len(x) < 3:
        return None
    predictions = []
    for held_out in range(len(x)):
        train_x = [value for index, value in enumerate(x) if index != held_out]
        train_y = [value for index, value in enumerate(y) if index != held_out]
        mean_x = statistics.fmean(train_x)
        mean_y = statistics.fmean(train_y)
        denominator = sum((value - mean_x) ** 2 for value in train_x)
        slope = (
            sum(
                (left - mean_x) * (right - mean_y)
                for left, right in zip(train_x, train_y, strict=True)
            )
            / denominator
            if denominator > 0
            else 0.0
        )
        prediction = mean_y + slope * (x[held_out] - mean_x)
        predictions.append(
            {
                "index": held_out,
                "observed": y[held_out],
                "predicted": prediction,
                "error": prediction - y[held_out],
            }
        )
    return {
        "predictions": predictions,
        "rmse": math.sqrt(statistics.fmean(row["error"] ** 2 for row in predictions)),
    }


def collect_final(*, preparation_manifest: Path, run_id: str) -> dict[str, Any]:
    preparation = _read_json(preparation_manifest)
    gate = _read_json(STEP_ROOT / "results/discriminator_transfer_screen_gate_v1.json")
    passing = list(gate.get("passing_pairs") or [])
    rows = []
    for pair_id in passing:
        item = preparation["pairs"][pair_id]
        config = load_grpo_config(Path(str(item["grpo_config"]["path"])))
        domain, slug, _, _, candidate = validate_grpo_config(config)
        run_root = Path(str(config.raw["paths"]["experiments"])) / run_id
        selection_path = run_root / "evaluations/validation/selection.json"
        selected = _read_json(selection_path)
        metrics = gate["pairs"][pair_id]["metrics"]
        rows.append(
            {
                "pair_id": pair_id,
                "candidate_slug": slug,
                "model_id": candidate["model_id"],
                "provider": candidate["provider"],
                "domain": domain,
                "U_D": float(metrics["U_D"]),
                "G_D0": float(metrics["G_D0"]),
                "gamma_0": float(metrics["gamma_0"]),
                "AUC_correct": float(metrics["AUC_correct"]),
                "AUC_identity": float(metrics["AUC_identity"]),
                "validation_gain_vs_a0": float(selected["validation_gain_vs_a0"]),
                "selected_step": int(selected["selected_step"]),
                "selection": str(selection_path),
                "selection_sha256": sha256_file(selection_path),
            }
        )
    predictors = ["U_D", "G_D0", "gamma_0", "AUC_correct"]
    correlations: dict[str, Any] = {}
    outcomes = [row["validation_gain_vs_a0"] for row in rows]
    for predictor in predictors:
        values = [row[predictor] for row in rows]
        correlations[predictor] = {
            "pearson": _pearson(values, outcomes),
            "spearman": _pearson(_ranks(values), _ranks(outcomes)),
            "leave_one_out_linear": _leave_one_out_linear(values, outcomes),
        }
    result = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "protocol": "campaign-discriminator-transfer-final-v1",
        "screen_gate": str(
            STEP_ROOT / "results/discriminator_transfer_screen_gate_v1.json"
        ),
        "screen_gate_sha256": sha256_file(
            STEP_ROOT / "results/discriminator_transfer_screen_gate_v1.json"
        ),
        "passing_pairs": passing,
        "rows": rows,
        "correlations": correlations,
        "inference_scope": "descriptive/exploratory",
        "official_test_accessed": False,
        "mbpp": "FORBIDDEN",
        "ifbench": "DEFERRED",
    }
    destination = STEP_ROOT / "results/discriminator_transfer_validation_v1.json"
    if destination.exists():
        raise FileExistsError(destination)
    write_json(destination, result)
    return result
