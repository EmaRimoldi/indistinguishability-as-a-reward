from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import math
import random
import statistics
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

from iar_campaign.reward_pilot import (
    key_lock,
    load_named_secret,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from iar_closed_weight.gemini_pilot import (
    EXPECTED_RESPONSE_PROVIDER,
    MAX_PRICE,
    MODEL_ID,
    ONE_LABEL_SCHEMA,
    PROVIDER_ENDPOINT,
    SERVICE_TIER,
    GeminiClosedWeightPilot,
    PilotSettings,
    ProtocolViolation,
    audit_phase_log,
    build_judge_prompt,
    call_plan_sha256,
    iter_call_plan,
    prepare_or_verify_phase_dir,
    select_canary_plan,
    summarize_canary,
)
from iar_grpo.config import GrpoTrialConfig, load_grpo_config
from iar_grpo.train import run_training
from iar_phase0.generator import strip_code_fences
from iar_phase0.parameters import initialize_parameter_record

CAMPAIGN_ROOT = Path("/orcd/pool/008/erimoldi/iar/experiments/campaign_20260908_v2")
STEP_ROOT = CAMPAIGN_ROOT / "11_closed_weight_teacher_k10"
SELECTION = (
    CAMPAIGN_ROOT / "06_two_domain_selection/results/two_domain_selection_v1.json"
)
STEP9_PREPARATION = (
    CAMPAIGN_ROOT
    / "09_pairwise_generate_distinguish/manifests/pairwise_prepared_attempt001.json"
)
PROTOCOL = STEP_ROOT / "configs/closed_weight_k10_protocol_v1.json"
REGISTRY = STEP_ROOT / "pilot/pilot_registry_attempt001.json"
CANARY_GATE = STEP_ROOT / "results/closed_weight_canary_gate_v1.json"
PILOT_GATE = STEP_ROOT / "results/closed_weight_pilot_gate_v1.json"
BASELINE_REGISTRY = (
    CAMPAIGN_ROOT
    / "05_four_domain_screen_1p5b/results/validation_baseline_registry_v1.json"
)
TEMPERATURES = (0.7, 1.0, 1.3)
K = 10
LANES = 3
CALL_ORDER_SEED = 230911
CHECKPOINTS = (0, 500, 1000, 1500, 2000)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _hash_rank(domain: str, prompt_id: str) -> str:
    return hashlib.sha256(
        f"{CALL_ORDER_SEED}|{domain}|{prompt_id}".encode()
    ).hexdigest()


def validate_protocol() -> dict[str, Any]:
    protocol = _read(PROTOCOL)
    serialized = json.dumps(protocol, sort_keys=True).lower()
    if (
        protocol.get("protocol") != "campaign-closed-weight-teacher-k10-v1"
        or protocol.get("reward", {}).get("k") != K
        or tuple(protocol.get("reward", {}).get("temperatures", [])) != TEMPERATURES
        or protocol.get("reward", {}).get("smoothing") != "(m+1)/(K+2)"
        or protocol.get("endpoint", {}).get("model_id") != MODEL_ID
        or protocol.get("endpoint", {}).get("provider_tag") != PROVIDER_ENDPOINT
        or protocol.get("endpoint", {}).get("provider") != EXPECTED_RESPONSE_PROVIDER
        or protocol.get("endpoint", {}).get("service_tier") != SERVICE_TIER
        or protocol.get("official_test_accessed") is not False
        or "mbpp" not in serialized
        or protocol.get("mbpp") != "FORBIDDEN"
    ):
        raise RuntimeError("Frozen Step 11 protocol is missing or drifted")
    return protocol


def prepare_pilot_registry() -> dict[str, Any]:
    protocol = validate_protocol()
    selection = _read(SELECTION)
    step9 = _read(STEP9_PREPARATION)
    domains = list(selection.get("selected_domains") or [])
    if (
        selection.get("status") != "complete"
        or len(domains) != 2
        or domains[0] != "commonsenseqa"
        or step9.get("status") != "PREPARED_NOT_LAUNCHED"
        or step9.get("selected_domains") != domains
        or selection.get("official_test_accessed") is not False
    ):
        raise RuntimeError("Step 11 requires the frozen Step 6/9 inputs")
    if REGISTRY.exists():
        raise FileExistsError(REGISTRY)
    candidate_rows: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    for domain in domains:
        probe_config = _read(Path(str(step9["probe_configs"][domain]["path"])))
        generation = (
            Path(str(probe_config["output_root"]))
            / "runs/probe_attempt001/generation_attempt001"
        )
        summary_path = generation / "summary.json"
        summary = _read(summary_path)
        source_path = Path(str(summary.get("candidates", "")))
        if (
            summary.get("status") != "complete"
            or summary.get("domain") != domain
            or int(summary.get("n_candidates", -1)) != 400
            or int(summary.get("rollouts_per_prompt", -1)) != 4
            or sha256_file(source_path) != summary.get("candidates_sha256")
        ):
            raise RuntimeError(f"Step 9 probe candidates are incomplete: {domain}")
        rows = _read_jsonl(source_path)
        by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_prompt[str(row["id"])].append(row)
        if len(by_prompt) != 100 or any(
            len(group) != 4 for group in by_prompt.values()
        ):
            raise RuntimeError(f"Step 9 G4 groups are incomplete: {domain}")
        selected_ids = sorted(by_prompt, key=lambda value: _hash_rank(domain, value))[
            :50
        ]
        for prompt_id in selected_ids:
            group = sorted(
                by_prompt[prompt_id], key=lambda row: int(row["rollout_index"])
            )
            if [int(row["rollout_index"]) for row in group] != list(range(4)):
                raise RuntimeError(f"Invalid rollout indices: {domain}/{prompt_id}")
            for row in group:
                candidate_rows.append(
                    {
                        "candidate_id": str(row["logical_key"]),
                        "dataset": domain,
                        "prompt_id": prompt_id,
                        "rollout_index": int(row["rollout_index"]),
                        "split_role": "train_probe100_nested50",
                        "task_statement": str(row["task_prompt"]),
                        "candidate": str(row["response"]),
                        "task_statement_sha256": str(row["task_prompt_sha256"]),
                        "candidate_sha256": str(row["response_sha256"]),
                    }
                )
        sources[domain] = {
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "candidates": str(source_path),
            "candidates_sha256": sha256_file(source_path),
            "selected_prompt_ids": selected_ids,
        }
    candidate_path = STEP_ROOT / "pilot/candidates_attempt001.jsonl"
    if candidate_path.exists():
        raise FileExistsError(candidate_path)
    write_jsonl(candidate_path, candidate_rows)
    manifest = {
        "schema_version": 1,
        "status": "PREPARED_NOT_LAUNCHED",
        "created_at": utc_now(),
        "protocol": "campaign-closed-weight-teacher-k10-registry-v1",
        "protocol_path": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "domain_selection": str(SELECTION),
        "domain_selection_sha256": sha256_file(SELECTION),
        "selected_domains": domains,
        "split_role": "train_probe100_nested50",
        "source_is_official_test": False,
        "official_test_accessed": False,
        "prompt_count": 100,
        "prompts_per_domain": 50,
        "rollouts_per_prompt": 4,
        "candidate_count": 400,
        "candidate_file": str(candidate_path),
        "candidate_file_sha256": sha256_file(candidate_path),
        "sources": sources,
        "planned_judgments": 400 * K * len(TEMPERATURES),
        "mbpp": "FORBIDDEN",
        "ifbench": "DEFERRED",
        "cost_protocol": protocol["pilot"],
    }
    write_json(REGISTRY, manifest)
    return manifest


def load_registry(path: Path = REGISTRY) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _read(path)
    candidate_path = Path(str(manifest.get("candidate_file", "")))
    rows = _read_jsonl(candidate_path)
    if (
        manifest.get("status") != "PREPARED_NOT_LAUNCHED"
        or manifest.get("official_test_accessed") is not False
        or manifest.get("source_is_official_test") is not False
        or sha256_file(candidate_path) != manifest.get("candidate_file_sha256")
        or len(rows) != 400
        or len({str(row["candidate_id"]) for row in rows}) != 400
    ):
        raise ProtocolViolation("Step 11 pilot registry is incomplete or drifted")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["prompt_id"]))].append(row)
    if len(groups) != 100 or any(len(group) != 4 for group in groups.values()):
        raise ProtocolViolation(
            "Step 11 registry does not contain 100 complete G4 groups"
        )
    return manifest, rows


def _settings(
    *,
    registry_manifest: Path,
    output_dir: Path,
    canary_enabled: bool,
    full_enabled: bool,
) -> PilotSettings:
    protocol = validate_protocol()
    pilot = protocol["pilot"]
    canary = protocol["canary"]
    return PilotSettings(
        model_id=MODEL_ID,
        expected_response_provider=EXPECTED_RESPONSE_PROVIDER,
        service_tier=SERVICE_TIER,
        provider_routing={
            "only": [PROVIDER_ENDPOINT],
            "allow_fallbacks": False,
            "require_parameters": True,
            "max_price": MAX_PRICE,
        },
        temperatures=TEMPERATURES,
        k_values=(K,),
        independent_batches=1,
        prompt_count=100,
        datasets=2,
        prompts_per_dataset=50,
        rollouts_per_prompt=4,
        max_tokens=32,
        top_p=1.0,
        reasoning={"effort": "medium", "exclude": True},
        response_format=ONE_LABEL_SCHEMA,
        max_concurrency=4,
        timeout_seconds=300.0,
        max_attempts=1,
        max_retry_wait_seconds=60.0,
        minimum_label_rate=float(pilot["gate_thresholds"]["minimum_label_rate"]),
        maximum_label_rate=float(pilot["gate_thresholds"]["maximum_label_rate"]),
        minimum_variable_group_fraction=float(
            pilot["gate_thresholds"]["minimum_variable_g4_group_fraction"]
        ),
        minimum_independent_batch_pearson=float(
            pilot["gate_thresholds"]["minimum_half_split_pearson"]
        ),
        signal_selection_rule=str(pilot["selection_rule"]),
        signal_tie_breaker="lower_temperature",
        canary_candidates_per_dataset=1,
        canary_max_total_cost_usd=float(canary["maximum_total_cost_usd"]),
        max_projected_full_cost_usd=float(pilot["maximum_projected_cost_usd"]),
        full_hard_cost_cap_usd=float(pilot["hard_cost_cap_usd"]),
        max_single_request_cost_usd=float(canary["maximum_single_request_cost_usd"]),
        output_dir=output_dir,
        registry_manifest=registry_manifest,
        registry_manifest_sha256=sha256_file(registry_manifest),
        canary_gate=CANARY_GATE,
        canary_gate_sha256=sha256_file(CANARY_GATE)
        if CANARY_GATE.exists()
        else "PENDING",
        canary_execution_enabled=canary_enabled,
        full_execution_enabled=full_enabled,
    )


def _raw_fingerprint_config(manifest: Path) -> dict[str, Any]:
    protocol = validate_protocol()
    return {
        "schema_version": 1,
        "endpoint": protocol["endpoint"],
        "pilot": protocol["pilot"],
        "transport": {"max_concurrency": 4, "max_attempts": 1, "timeout_seconds": 300},
        "cost_control": {"canary": protocol["canary"], "pilot": protocol["pilot"]},
        "signal_gate": protocol["pilot"]["gate_thresholds"],
        "paths": {"registry": str(manifest)},
    }


def _manifest(
    *, mode: str, plan: list[dict[str, Any]], registry: Path, lane: int | None
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": "campaign-closed-weight-teacher-k10-v1",
        "mode": mode,
        "lane": lane,
        "official_test_accessed": False,
        "model_id": MODEL_ID,
        "provider_endpoint": PROVIDER_ENDPOINT,
        "k": K,
        "temperatures": list(TEMPERATURES),
        "registry_manifest": str(registry),
        "registry_manifest_sha256": sha256_file(registry),
        "protocol_sha256": sha256_file(PROTOCOL),
        "call_plan_sha256": call_plan_sha256(plan),
        "planned_requests": len(plan),
        "max_http_attempts_per_logical_request": 1,
        "response_cache_enabled": False,
    }


def run_canary(*, registry: Path, key_pool: Path, key_name: str) -> dict[str, Any]:
    manifest, rows = load_registry(registry)
    run_dir = STEP_ROOT / "canary/runs/canary_attempt001"
    settings = _settings(
        registry_manifest=registry,
        output_dir=run_dir,
        canary_enabled=True,
        full_enabled=False,
    )
    plan = select_canary_plan(rows, settings)
    log_path = run_dir / "raw_judgments.jsonl"
    prepare_or_verify_phase_dir(
        run_dir, _manifest(mode="canary", plan=plan, registry=registry, lane=None)
    )
    audit = (
        audit_phase_log(log_path, plan=plan, phase="canary")
        if log_path.exists()
        else {"completed_call_ids": set(), "observed_cost_usd": 0.0}
    )
    secret = load_named_secret(key_pool, key_name)
    runner = GeminiClosedWeightPilot(
        settings=settings,
        api_key=secret,
        log_path=log_path,
        initial_observed_cost_usd=float(audit["observed_cost_usd"]),
    )
    with key_lock(CAMPAIGN_ROOT / "00_campaign_control/locks/openrouter", key_name):
        asyncio.run(
            runner.run_canary(plan, completed_call_ids=set(audit["completed_call_ids"]))
        )
    gate = summarize_canary(
        log_path,
        raw=_raw_fingerprint_config(registry),
        settings=settings,
        registry_rows=rows,
    )
    gate.update(
        {
            "protocol": "campaign-closed-weight-teacher-k10-v1",
            "protocol_sha256": sha256_file(PROTOCOL),
            "registry_manifest": str(registry),
            "registry_manifest_sha256": sha256_file(registry),
            "selected_domains": manifest["selected_domains"],
            "completed_at": utc_now(),
        }
    )
    if CANARY_GATE.exists():
        raise FileExistsError(CANARY_GATE)
    write_json(CANARY_GATE, gate)
    return gate


def _full_plan(registry: Path) -> list[dict[str, Any]]:
    _, rows = load_registry(registry)
    settings = _settings(
        registry_manifest=registry,
        output_dir=STEP_ROOT / "pilot",
        canary_enabled=False,
        full_enabled=True,
    )
    plan = list(iter_call_plan(rows, settings))
    for spec in plan:
        spec["logical_call_id"] = spec["call_id"]
        spec["transport_attempt"] = 1
    return plan


def shard_plan(plan: Iterable[Mapping[str, Any]], lane: int) -> list[dict[str, Any]]:
    if lane not in range(LANES):
        raise ValueError("Pilot lane must be 0, 1 or 2")
    return [
        dict(spec)
        for spec in plan
        if int(hashlib.sha256(str(spec["call_id"]).encode()).hexdigest(), 16) % LANES
        == lane
    ]


def _safe_explicit_http_error(exc: BaseException) -> bool:
    cause: BaseException | None = exc
    while cause is not None:
        if isinstance(cause, httpx.HTTPStatusError):
            response = cause.response
            if response.status_code != 429 and not 500 <= response.status_code <= 599:
                return False
            try:
                body = response.json()
            except ValueError:
                return False
            if not isinstance(body, dict) or body.get("id"):
                return False
            usage = body.get("usage")
            cost = usage.get("cost") if isinstance(usage, dict) else None
            return cost in {None, 0}
        cause = cause.__cause__
    return False


async def _judge_with_known_failure_recovery(
    runner: GeminiClosedWeightPilot,
    client: httpx.AsyncClient,
    spec: Mapping[str, Any],
    *,
    phase: str,
    maximum_transport_attempts: int = 5,
) -> int:
    base_id = str(spec.get("logical_call_id") or spec["call_id"])
    for attempt in range(1, maximum_transport_attempts + 1):
        current = dict(spec)
        current["logical_call_id"] = base_id
        current["transport_attempt"] = attempt
        current["call_id"] = (
            base_id
            if attempt == 1
            else hashlib.sha256(
                f"{base_id}|known-unbilled|{attempt}".encode()
            ).hexdigest()[:24]
        )
        try:
            return await runner.judge_one(client, current, phase=phase)
        except RuntimeError as exc:
            if attempt >= maximum_transport_attempts or not _safe_explicit_http_error(
                exc
            ):
                raise
            await asyncio.sleep(min(5.0 * 2 ** (attempt - 1), 60.0))
    raise AssertionError("unreachable")


def audit_resilient_log(
    log_path: Path, *, plan: list[dict[str, Any]], phase: str
) -> dict[str, Any]:
    expected = {
        str(spec.get("logical_call_id") or spec["call_id"]): spec for spec in plan
    }
    by_logical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows = _read_jsonl(log_path) if log_path.exists() else []
    for row in rows:
        logical = str(row.get("logical_call_id") or row.get("call_id", ""))
        spec = expected.get(logical)
        if spec is None or row.get("phase") != phase:
            raise ProtocolViolation("Paid log contains an unknown logical call")
        if (
            str(row.get("candidate_id")) != str(spec["candidate_id"])
            or float(row.get("temperature")) != float(spec["temperature"])
            or int(row.get("replicate_index")) != int(spec["replicate_index"])
        ):
            raise ProtocolViolation("Paid recovery log drifted from its frozen call")
        by_logical[logical].append(row)
    completed: set[str] = set()
    cost = 0.0
    for logical, attempts in by_logical.items():
        if len(attempts) % 2:
            raise ProtocolViolation(f"Ambiguous paid WAL without receipt: {logical}")
        pairs = [attempts[index : index + 2] for index in range(0, len(attempts), 2)]
        successes = 0
        for index, pair in enumerate(pairs):
            if pair[0].get("status") != "intent" or pair[1].get("status") not in {
                "ok",
                "error",
            }:
                raise ProtocolViolation(f"Malformed paid WAL sequence: {logical}")
            if pair[1].get("status") == "ok":
                successes += 1
                if index != len(pairs) - 1:
                    raise ProtocolViolation(
                        f"Paid call continued after success: {logical}"
                    )
                cost += float((pair[1].get("usage") or {}).get("cost", 0.0))
            else:
                status = int(pair[1].get("response_status_code", 0))
                if (
                    pair[1].get("billing_status")
                    != "explicit_error_no_generation_receipt"
                    or status != 429
                    and not 500 <= status <= 599
                ):
                    raise ProtocolViolation(
                        f"Ambiguous or unsafe paid failure: {logical}"
                    )
        if successes > 1:
            raise ProtocolViolation(f"Duplicate paid success: {logical}")
        if successes == 1:
            completed.add(logical)
    return {
        "completed_call_ids": completed,
        "completed_calls": len(completed),
        "pending_calls": len(plan) - len(completed),
        "observed_cost_usd": cost,
        "rows": len(rows),
    }


def run_pilot_lane(
    *, registry: Path, lane: int, key_pool: Path, key_name: str
) -> dict[str, Any]:
    canary = _read(CANARY_GATE)
    if (
        canary.get("passed") is not True
        or canary.get("official_test_accessed") is not False
    ):
        raise ProtocolViolation("Pilot is blocked by the closed-weight canary")
    settings = _settings(
        registry_manifest=registry,
        output_dir=STEP_ROOT / "pilot",
        canary_enabled=False,
        full_enabled=True,
    )
    plan = shard_plan(_full_plan(registry), lane)
    run_dir = STEP_ROOT / f"pilot/runs/pilot_attempt001/lane_{lane}"
    log_path = run_dir / "raw_judgments.jsonl"
    prepare_or_verify_phase_dir(
        run_dir, _manifest(mode="pilot", plan=plan, registry=registry, lane=lane)
    )
    audit = audit_resilient_log(log_path, plan=plan, phase="pilot")
    secret = load_named_secret(key_pool, key_name)
    runner = GeminiClosedWeightPilot(
        settings=replace(settings, full_hard_cost_cap_usd=20.0 / LANES),
        api_key=secret,
        log_path=log_path,
        initial_observed_cost_usd=float(audit["observed_cost_usd"]),
    )
    with key_lock(CAMPAIGN_ROOT / "00_campaign_control/locks/openrouter", key_name):

        async def execute() -> None:
            pending = [
                spec
                for spec in plan
                if str(spec["logical_call_id"]) not in set(audit["completed_call_ids"])
            ]
            async with httpx.AsyncClient(
                timeout=runner.settings.timeout_seconds
            ) as client:
                for start in range(0, len(pending), 16):
                    results = await asyncio.gather(
                        *(
                            _judge_with_known_failure_recovery(
                                runner, client, spec, phase="pilot"
                            )
                            for spec in pending[start : start + 16]
                        ),
                        return_exceptions=True,
                    )
                    failures = [
                        value for value in results if isinstance(value, BaseException)
                    ]
                    if failures:
                        raise RuntimeError(
                            f"Pilot lane {lane} stopped after "
                            f"{len(failures)} paid-call failures"
                        ) from failures[0]

        asyncio.run(execute())
    final = audit_resilient_log(log_path, plan=plan, phase="pilot")
    if final["pending_calls"] != 0:
        raise ProtocolViolation(f"Pilot lane {lane} is incomplete")
    summary = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "lane": lane,
        "planned_requests": len(plan),
        "completed_requests": final["completed_calls"],
        "observed_cost_usd": final["observed_cost_usd"],
        "log": str(log_path),
        "log_sha256": sha256_file(log_path),
        "official_test_accessed": False,
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    lm, rm = statistics.mean(left), statistics.mean(right)
    numerator = sum((x - lm) * (y - rm) for x, y in zip(left, right, strict=True))
    ls = math.sqrt(sum((x - lm) ** 2 for x in left))
    rs = math.sqrt(sum((y - rm) ** 2 for y in right))
    return None if ls == 0 or rs == 0 else numerator / (ls * rs)


def collect_pilot(*, registry: Path) -> dict[str, Any]:
    protocol = validate_protocol()
    manifest, registry_rows = load_registry(registry)
    plan = _full_plan(registry)
    planned = {str(spec["call_id"]): spec for spec in plan}
    ok: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    for lane in range(LANES):
        run_dir = STEP_ROOT / f"pilot/runs/pilot_attempt001/lane_{lane}"
        summary = _read(run_dir / "summary.json")
        log_path = Path(str(summary["log"]))
        if summary.get("status") != "complete" or sha256_file(log_path) != summary.get(
            "log_sha256"
        ):
            raise ProtocolViolation(f"Pilot lane {lane} is incomplete")
        lane_plan = shard_plan(plan, lane)
        audit_resilient_log(log_path, plan=lane_plan, phase="pilot")
        for row in _read_jsonl(log_path):
            all_rows.append(row)
            if row.get("status") == "ok":
                call_id = str(row.get("logical_call_id") or row["call_id"])
                if call_id in ok:
                    raise ProtocolViolation(f"Duplicate paid judgment: {call_id}")
                ok[call_id] = row
    if set(ok) != set(planned) or len(ok) != 12_000:
        raise ProtocolViolation(
            "Pilot does not contain exactly 12,000 resolved judgments"
        )
    by_condition: dict[tuple[float, str, str], list[dict[str, Any]]] = defaultdict(list)
    costs = []
    for call_id, row in ok.items():
        spec = planned[call_id]
        by_condition[
            (
                float(spec["temperature"]),
                str(spec["dataset"]),
                str(spec["candidate_id"]),
            )
        ].append(row)
        costs.append(float((row.get("usage") or {}).get("cost", 0.0)))
    candidate_meta = {str(row["candidate_id"]): row for row in registry_rows}
    metrics: list[dict[str, Any]] = []
    candidates_by_temperature_domain: dict[tuple[float, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for (temperature, domain, candidate_id), rows in sorted(by_condition.items()):
        rows.sort(key=lambda row: int(row["replicate_index"]))
        if len(rows) != K or [int(row["replicate_index"]) for row in rows] != list(
            range(K)
        ):
            raise ProtocolViolation(
                f"Incomplete K=10 group: {candidate_id}/{temperature}"
            )
        labels = [int(row["label"]) for row in rows]
        item = {
            "temperature": temperature,
            "domain": domain,
            "candidate_id": candidate_id,
            "prompt_id": str(candidate_meta[candidate_id]["prompt_id"]),
            "rollout_index": int(candidate_meta[candidate_id]["rollout_index"]),
            "m": sum(labels),
            "p_own": (sum(labels) + 1) / 12,
            "reward": math.log((sum(labels) + 1) / 12),
            "p_first5": (sum(labels[:5]) + 1) / 7,
            "p_last5": (sum(labels[5:]) + 1) / 7,
        }
        candidates_by_temperature_domain[(temperature, domain)].append(item)
    thresholds = protocol["pilot"]["gate_thresholds"]
    eligible_by_temperature: dict[float, bool] = {value: True for value in TEMPERATURES}
    for temperature in TEMPERATURES:
        for domain in manifest["selected_domains"]:
            rows = candidates_by_temperature_domain[(temperature, domain)]
            groups: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                groups[str(row["prompt_id"])].append(float(row["p_own"]))
            variable = sum(
                len(values) == 4 and statistics.pvariance(values) > 0
                for values in groups.values()
            )
            label_rate = sum(int(row["m"]) for row in rows) / (len(rows) * K)
            reliability = _pearson(
                [float(row["p_first5"]) for row in rows],
                [float(row["p_last5"]) for row in rows],
            )
            variable_fraction = variable / len(groups)
            eligible = (
                float(thresholds["minimum_label_rate"])
                <= label_rate
                <= float(thresholds["maximum_label_rate"])
                and variable_fraction
                >= float(thresholds["minimum_variable_g4_group_fraction"])
                and reliability is not None
                and reliability >= float(thresholds["minimum_half_split_pearson"])
            )
            eligible_by_temperature[temperature] &= eligible
            metrics.append(
                {
                    "temperature": temperature,
                    "domain": domain,
                    "candidate_count": len(rows),
                    "judgment_count": len(rows) * K,
                    "label_rate": label_rate,
                    "variable_g4_group_fraction": variable_fraction,
                    "half_split_pearson": reliability,
                    "p_own_mean": statistics.mean(float(row["p_own"]) for row in rows),
                    "p_own_std": statistics.pstdev(float(row["p_own"]) for row in rows),
                    "eligible": eligible,
                }
            )
    eligible_temperatures = [
        value for value in TEMPERATURES if eligible_by_temperature[value]
    ]

    def rank(value: float) -> tuple[float, float, float, float]:
        rows = [row for row in metrics if float(row["temperature"]) == value]
        return (
            min(float(row["variable_g4_group_fraction"]) for row in rows),
            min(float(row["half_split_pearson"]) for row in rows),
            -max(abs(float(row["label_rate"]) - 0.5) for row in rows),
            -value,
        )

    selected = max(eligible_temperatures, key=rank) if eligible_temperatures else None
    pilot_cost = sum(costs)
    projected_training_cost = pilot_cost * (160_000 / 12_000) * 1.25
    cost_passed = (
        pilot_cost <= float(protocol["pilot"]["hard_cost_cap_usd"])
        and projected_training_cost
        <= float(protocol["full_training_if_go"]["maximum_projected_total_cost_usd"])
        and max(costs, default=0.0)
        <= float(protocol["pilot"]["maximum_single_request_cost_usd"])
    )
    passed = selected is not None and cost_passed
    result = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "protocol": "campaign-closed-weight-teacher-k10-v1",
        "protocol_sha256": sha256_file(PROTOCOL),
        "registry_manifest": str(registry),
        "registry_manifest_sha256": sha256_file(registry),
        "successful_judgments": len(ok),
        "k": K,
        "smoothing": "(m+1)/12",
        "metrics": metrics,
        "selected_temperature": selected,
        "signal_passed": selected is not None,
        "observed_pilot_cost_usd": pilot_cost,
        "projected_full_training_cost_usd_with_25pct_margin": projected_training_cost,
        "cost_passed": cost_passed,
        "passed": passed,
        "decision": "GO" if passed else "NO_GO",
        "official_test_accessed": False,
    }
    if PILOT_GATE.exists():
        raise FileExistsError(PILOT_GATE)
    write_json(PILOT_GATE, result)
    candidate_output = STEP_ROOT / "pilot/runs/pilot_attempt001/candidate_rewards.jsonl"
    write_jsonl(
        candidate_output,
        [
            row
            for key in sorted(candidates_by_temperature_domain)
            for row in candidates_by_temperature_domain[key]
        ],
    )
    return result


class ClosedWeightK10Reward:
    """Convert ten audited Gemini binary judgments into one GRPO reward."""

    def __init__(
        self,
        *,
        domain: str,
        temperature: float,
        log_dir: Path,
        api_key: str,
        max_reward_calls: int = 8000,
    ) -> None:
        if temperature not in TEMPERATURES:
            raise ValueError("Closed-weight temperature is outside the frozen grid")
        self.domain = domain
        self.temperature = temperature
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=False)
        self.group_log = self.log_dir / "reward_groups.jsonl"
        self.judgment_log = self.log_dir / "raw_judgments.jsonl"
        self.max_reward_calls = max_reward_calls
        self.logical_calls = 0
        settings = _settings(
            registry_manifest=REGISTRY,
            output_dir=log_dir,
            canary_enabled=False,
            full_enabled=True,
        )
        protocol = validate_protocol()
        self.runner = GeminiClosedWeightPilot(
            settings=replace(
                settings,
                temperatures=(temperature,),
                full_hard_cost_cap_usd=float(
                    protocol["full_training_if_go"]["hard_total_cost_cap_usd"]
                )
                / 2,
            ),
            api_key=api_key,
            log_path=self.judgment_log,
        )
        self._group_lock = asyncio.Lock()

    async def score_batch(
        self,
        *,
        completions: list[Any],
        task_statement: list[str],
        task_id: list[str],
        dataset_index: list[int],
        rollout_index: list[int] | None = None,
    ) -> list[float]:
        from iar_grpo.reward import completion_to_text

        size = len(completions)
        if not (len(task_statement) == len(task_id) == len(dataset_index) == size):
            raise ValueError("Closed-weight reward metadata length mismatch")
        if self.logical_calls + size > self.max_reward_calls:
            raise RuntimeError("Closed-weight logical reward budget exceeded")
        first = self.logical_calls
        self.logical_calls += size
        inferred: defaultdict[str, int] = defaultdict(int)
        specs_by_candidate: list[list[dict[str, Any]]] = []
        metadata: list[dict[str, Any]] = []
        for offset, completion in enumerate(completions):
            task = str(task_id[offset])
            if rollout_index is None:
                rollout = inferred[task]
                inferred[task] += 1
            else:
                rollout = int(rollout_index[offset])
            raw_candidate = completion_to_text(completion)
            candidate = (
                strip_code_fences(raw_candidate)
                if self.domain == "kodcode"
                else raw_candidate.strip()
            )
            logical_index = first + offset + 1
            candidate_id = hashlib.sha256(
                f"train|{self.domain}|{logical_index}|{task}|{rollout}|{candidate}".encode()
            ).hexdigest()[:24]
            prompt = build_judge_prompt(str(task_statement[offset]), candidate)
            specs_by_candidate.append(
                [
                    {
                        "call_id": hashlib.sha256(
                            f"{candidate_id}|{self.temperature}|{replicate}".encode()
                        ).hexdigest()[:24],
                        "candidate_id": candidate_id,
                        "dataset": self.domain,
                        "prompt_id": task,
                        "rollout_index": rollout,
                        "temperature": self.temperature,
                        "batch_id": 0,
                        "replicate_index": replicate,
                        "prompt": prompt,
                    }
                    for replicate in range(K)
                ]
            )
            for spec in specs_by_candidate[-1]:
                spec["logical_call_id"] = spec["call_id"]
                spec["transport_attempt"] = 1
            metadata.append(
                {
                    "logical_reward_id": logical_index,
                    "candidate_id": candidate_id,
                    "task_id": task,
                    "dataset_index": int(dataset_index[offset]),
                    "rollout_index": rollout,
                    "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
                }
            )
        async with httpx.AsyncClient(
            timeout=self.runner.settings.timeout_seconds
        ) as client:
            flat_labels = await asyncio.gather(
                *(
                    _judge_with_known_failure_recovery(
                        self.runner, client, spec, phase="pilot"
                    )
                    for specs in specs_by_candidate
                    for spec in specs
                ),
                return_exceptions=True,
            )
        failures = [value for value in flat_labels if isinstance(value, BaseException)]
        if failures:
            raise RuntimeError(
                f"Closed-weight batch stopped after {len(failures)} paid-call failures"
            ) from failures[0]
        rewards: list[float] = []
        group_rows: list[dict[str, Any]] = []
        for offset, meta in enumerate(metadata):
            labels = [
                int(value) for value in flat_labels[offset * K : (offset + 1) * K]
            ]
            m = sum(labels)
            p_own = (m + 1) / 12
            reward = math.log(p_own)
            rewards.append(reward)
            group_rows.append(
                {
                    **meta,
                    "domain": self.domain,
                    "temperature": self.temperature,
                    "k": K,
                    "labels": labels,
                    "m": m,
                    "p_own": p_own,
                    "reward": reward,
                    "reward_value": "closed_weight_log_p_own_k10",
                    "completed_at_unix": time.time(),
                    "status": "ok",
                }
            )
        async with self._group_lock:
            with self.group_log.open("a", encoding="utf-8") as handle:
                for row in group_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
        return rewards

    def audit_summary(self) -> dict[str, Any]:
        judgments = _read_jsonl(self.judgment_log) if self.judgment_log.exists() else []
        unsafe_errors = [
            row
            for row in judgments
            if row.get("status") == "error"
            and row.get("billing_status") != "explicit_error_no_generation_receipt"
        ]
        return {
            "logical_reward_calls": self.logical_calls,
            "successful_judgments": sum(row.get("status") == "ok" for row in judgments),
            "judgment_rows": len(judgments),
            "k": K,
            "maximum_logical_reward_calls": self.max_reward_calls,
            "maximum_judgments": self.max_reward_calls * K,
            "known_unbilled_replacement_attempts": sum(
                row.get("status") == "error" for row in judgments
            ),
            "unsafe_or_ambiguous_errors": len(unsafe_errors),
        }

    def reward_statistics(self) -> dict[str, Any]:
        rows = _read_jsonl(self.group_log) if self.group_log.exists() else []
        rewards = [float(row["reward"]) for row in rows]
        by_task: dict[str, Any] = {}
        for task_id in sorted({str(row["task_id"]) for row in rows}):
            group = [row for row in rows if str(row["task_id"]) == task_id]
            values = [float(row["reward"]) for row in group]
            by_task[task_id] = {
                "n": len(group),
                "mean": statistics.mean(values),
                "std": statistics.pstdev(values),
                "min": min(values),
                "max": max(values),
            }
        return {
            "n": len(rows),
            "reward": {
                "mean": statistics.mean(rewards) if rewards else None,
                "std": statistics.pstdev(rewards) if rewards else None,
                "min": min(rewards) if rewards else None,
                "max": max(rewards) if rewards else None,
            },
            "by_task": by_task,
        }

    def assert_complete(self) -> None:
        groups = _read_jsonl(self.group_log)
        judgments = _read_jsonl(self.judgment_log)
        if (
            self.logical_calls != self.max_reward_calls
            or len(groups) != self.max_reward_calls
        ):
            raise RuntimeError("Closed-weight logical reward coverage is incomplete")
        if (
            sum(row.get("status") == "ok" for row in judgments)
            != self.max_reward_calls * K
        ):
            raise RuntimeError("Closed-weight K=10 judgment coverage is incomplete")
        if any(
            row.get("status") == "error"
            and (
                row.get("billing_status") != "explicit_error_no_generation_receipt"
                or (
                    int(row.get("response_status_code") or 0) != 429
                    and not 500 <= int(row.get("response_status_code") or 0) <= 599
                )
            )
            for row in judgments
        ):
            raise RuntimeError("Closed-weight paid log contains an unsafe failure")
        successful_ids = [
            str(row.get("logical_call_id") or row.get("call_id"))
            for row in judgments
            if row.get("status") == "ok"
        ]
        if len(successful_ids) != len(set(successful_ids)):
            raise RuntimeError("Closed-weight paid log contains duplicate successes")
        if any(int(row["k"]) != K or len(row["labels"]) != K for row in groups):
            raise RuntimeError("Closed-weight reward groups violate K=10")


def validate_training_config(
    config: GrpoTrialConfig,
) -> tuple[str, Path, Path, float]:
    validate_protocol()
    gate = _read(PILOT_GATE)
    selection = _read(SELECTION)
    raw = config.raw
    campaign = raw.get("campaign") or {}
    domain = str(campaign.get("domain", ""))
    selected = list(selection.get("selected_domains") or [])
    temperature = float(raw["discriminator"]["temperature"])
    if (
        gate.get("passed") is not True
        or float(gate.get("selected_temperature", -1)) != temperature
        or domain not in selected
        or selected[0] != "commonsenseqa"
        or sha256_file(PROTOCOL) != str(campaign.get("closed_weight_protocol_sha256"))
        or sha256_file(PILOT_GATE)
        != str(campaign.get("closed_weight_pilot_gate_sha256"))
    ):
        raise RuntimeError("Step 11 training is not bound to the frozen GO gate")
    if (
        config.prompt_count != 1000
        or config.num_generations != 4
        or config.num_epochs != 2
        or config.expected_reward_calls != 8000
        or raw["reward"].get("value") != "closed_weight_log_p_own_k10"
        or int(raw["reward"].get("judgments_per_reward", 0)) != K
        or int(raw["reward"].get("max_http_requests", 0)) != 80_000
        or int(raw["reward"].get("max_attempts_per_call", 0)) != 1
        or raw["discriminator"].get("model_id") != MODEL_ID
        or raw["discriminator"].get("expected_provider") != EXPECTED_RESPONSE_PROVIDER
    ):
        raise RuntimeError("Step 11 K=10 model, reward or budget drifted")
    training = raw["training"]
    if (
        int(training["max_steps"]) != 2000
        or int(training["save_steps"]) != 500
        or float(training["learning_rate"]) != 3e-5
        or float(training["beta"]) != 0.003
        or float(training["temperature"]) != 0.6
        or float(training["top_p"]) != 1.0
        or str(training["loss_type"]) != "dr_grpo"
        or str(training["scale_rewards"]) != "group"
    ):
        raise RuntimeError("Step 11 GRPO hyperparameters drifted")
    train_file = Path(str(raw["dataset"]["split_file"]))
    validation_file = Path(str(raw["evaluation"]["validation_file"]))
    if (
        sha256_file(train_file) != raw["dataset"].get("split_file_sha256")
        or sha256_file(validation_file)
        != raw["evaluation"].get("validation_file_sha256")
        or any(
            part.lower() in {"test", "sealed", "final_test"}
            for part in validation_file.parts
        )
    ):
        raise RuntimeError("Step 11 frozen train/validation split drifted")
    return domain, train_file, validation_file, temperature


def build_training_dataset(config: GrpoTrialConfig):
    from datasets import Dataset

    from iar_campaign.generate import prompts_for_domain, read_jsonl

    domain, train_file, _, _ = validate_training_config(config)
    examples = read_jsonl(train_file)
    if len(examples) != 1000 or len({str(row["id"]) for row in examples}) != 1000:
        raise RuntimeError("Step 11 train_core1000 is incomplete")
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
    domain, train_file, validation_file, temperature = validate_training_config(config)
    run_root = Path(str(config.raw["paths"]["experiments"])).resolve() / run_id
    checkpoint_root = Path(str(config.raw["paths"]["checkpoints"])).resolve() / run_id
    if (
        STEP_ROOT.resolve() not in run_root.parents
        or STEP_ROOT.resolve() not in checkpoint_root.parents
    ):
        raise ValueError("Step 11 output escaped its canonical directory")
    run_root.mkdir(parents=True, exist_ok=False)
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    (run_root / "config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    initialize_parameter_record(
        run_root, resolved_config=config.raw, source_config=config_path
    )
    dataset, rows = build_training_dataset(config)
    write_jsonl(run_root / "train_examples.jsonl", rows)
    runtime = ClosedWeightK10Reward(
        domain=domain,
        temperature=temperature,
        log_dir=run_root / "reward_calls",
        api_key=load_named_secret(key_pool, key_name),
    )
    try:
        with key_lock(lock_dir, key_name):
            result, history = run_training(
                config, dataset, checkpoint_root, run_root, runtime
            )
        runtime.assert_complete()
        write_jsonl(run_root / "trainer_log_history.jsonl", history)
        stats = runtime.reward_statistics()
        observed = stats.get("by_task") or {}
        if set(observed) != {str(row["task_id"]) for row in rows} or any(
            int(value.get("n", -1)) != 8 for value in observed.values()
        ):
            raise RuntimeError("Step 11 rewards do not cover train1000 x G4 x 2")
        gradients = [float(row["grad_norm"]) for row in history if "grad_norm" in row]
        if not gradients or any(not math.isfinite(value) for value in gradients):
            raise RuntimeError("Step 11 gradients are missing or non-finite")
        summary = {
            "schema_version": 1,
            "status": "complete",
            "completed_at": utc_now(),
            "run_id": run_id,
            "domain": domain,
            "model_id": MODEL_ID,
            "provider": EXPECTED_RESPONSE_PROVIDER,
            "temperature": temperature,
            "k": K,
            "smoothing": "(m+1)/12",
            "expected_logical_rewards": 8000,
            "expected_paid_judgments": 80000,
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
            "checkpoint_dir": str(checkpoint_root),
            "official_test_accessed": False,
        }
        write_json(run_root / "summary.json", summary)
        return summary
    except Exception as exc:
        write_json(
            run_root / "failure.json",
            {
                "schema_version": 1,
                "status": "failed",
                "failed_at": utc_now(),
                "run_id": run_id,
                "domain": domain,
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
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from iar_campaign.four_domain_screen import _adapter_for_step
    from iar_campaign.generate import prompts_for_domain, read_jsonl
    from iar_grpo.numerics import configure_stable_sdpa

    if attempt <= 0:
        raise ValueError("Invalid Step 11 validation attempt")
    config = load_grpo_config(config_path)
    domain, _, validation_file, _ = validate_training_config(config)
    run_root = Path(str(config.raw["paths"]["experiments"])).resolve() / run_id
    checkpoint_root = Path(str(config.raw["paths"]["checkpoints"])).resolve() / run_id
    if _read(run_root / "summary.json").get("status") != "complete":
        raise RuntimeError("Step 11 validation cannot precede training")
    examples = read_jsonl(validation_file)
    if len(examples) != 300:
        raise RuntimeError("Step 11 requires validation300")
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
    base_model = AutoModelForCausalLM.from_pretrained(
        config.actor_source(),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    try:
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
                raise RuntimeError("Step 11 checkpoint unexpectedly resolved to A0")
            model = PeftModel.from_pretrained(base_model, adapter, is_trainable=False)
            model.eval()
            candidates = []
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
                    candidates.append(
                        {
                            "id": str(example["id"]),
                            "source_index": int(example["source_index"]),
                            "domain": domain,
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
            write_jsonl(output / "candidates.jsonl", candidates)
            write_json(
                output / "summary.json",
                {
                    "schema_version": 1,
                    "status": "complete",
                    "completed_at": utc_now(),
                    "domain": domain,
                    "run_id": run_id,
                    "step": step,
                    "n": len(candidates),
                    "truncated": sum(
                        bool(row["hit_max_new_tokens"]) for row in candidates
                    ),
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
    from iar_campaign.generate import read_jsonl
    from iar_campaign.score_baselines import (
        _score_kodcode_rows,
        _score_simple,
        _score_spider_rows,
    )

    config = load_grpo_config(config_path)
    domain, _, validation_file, _ = validate_training_config(config)
    run_root = Path(str(config.raw["paths"]["experiments"])).resolve() / run_id
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
        generated = _read(generation / "summary.json")
        if (
            generated.get("status") != "complete"
            or generated.get("candidate_file_sha256")
            != sha256_file(generation / "candidates.jsonl")
            or [str(row["id"]) for row in candidates]
            != [str(row["id"]) for row in examples]
        ):
            raise RuntimeError(f"Incomplete Step 11 generation at {step}")
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
            "official_test_accessed": False,
        }
        write_json(output / "summary.json", summary)
        summaries.append(summary)
    return {
        "status": "complete",
        "domain": domain,
        "summaries": summaries,
        "official_test_accessed": False,
    }


def collect_validation(
    *, config_path: Path, run_id: str, scoring_attempt: int
) -> dict[str, Any]:
    config = load_grpo_config(config_path)
    domain, _, _, temperature = validate_training_config(config)
    run_root = Path(str(config.raw["paths"]["experiments"])).resolve() / run_id
    registry = _read(Path(str(config.raw["evaluation"]["baseline_registry"])))
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
        value = _read(path)
        if value.get("status") != "complete":
            raise RuntimeError(f"Incomplete Step 11 score at {step}")
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
        "protocol": "campaign-step11-closed-weight-validation-v1",
        "domain": domain,
        "run_id": run_id,
        "temperature": temperature,
        "k": K,
        "selected_step": int(selected["step"]),
        "selected_task_utility": float(selected["task_utility"]),
        "validation_gain_vs_a0": float(selected["task_utility"])
        - float(a0["task_utility"]),
        "evaluations": evaluations,
        "checkpoint_selection": "maximize validation task utility; earliest checkpoint on ties",
        "official_test_accessed": False,
        "mbpp": "FORBIDDEN",
        "ifbench": "DEFERRED",
    }
    output = run_root / "evaluations/validation/selection.json"
    if output.exists():
        raise FileExistsError(output)
    write_json(output, result)
    return result


def collect_final(*, preparation_manifest: Path) -> dict[str, Any]:
    prepared = _read(preparation_manifest)
    results = {}
    for domain, entry in prepared["configs"].items():
        config = load_grpo_config(Path(str(entry["path"])))
        validate_training_config(config)
        selection = (
            Path(str(config.raw["paths"]["experiments"]))
            / "attempt001/evaluations/validation/selection.json"
        )
        value = _read(selection)
        if value.get("status") != "complete" or value.get("domain") != domain:
            raise RuntimeError(f"Incomplete Step 11 validation: {domain}")
        results[domain] = {
            **value,
            "selection": str(selection),
            "selection_sha256": sha256_file(selection),
        }
    result = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "protocol": "campaign-step11-closed-weight-final-v1",
        "selected_domains": prepared["selected_domains"],
        "selected_temperature": prepared["selected_temperature"],
        "k": K,
        "smoothing": "(m+1)/12",
        "domains": results,
        "official_test_accessed": False,
        "mbpp": "FORBIDDEN",
        "ifbench": "DEFERRED",
    }
    output = STEP_ROOT / "results/closed_weight_validation_v1.json"
    if output.exists():
        raise FileExistsError(output)
    write_json(output, result)
    return result
