from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import httpx
import yaml

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "google/gemini-3.8-flash"
PROVIDER_ENDPOINT = "google-ai-studio/flex"
EXPECTED_RESPONSE_PROVIDER = "Google AI Studio"
SERVICE_TIER = "flex"
TEMPERATURES = (0.7, 1.0, 1.3)
K_VALUES = (4, 8, 16)
INDEPENDENT_BATCHES = 2
FLEX_INPUT_USD_PER_MILLION = 0.375
FLEX_OUTPUT_USD_PER_MILLION = 1.875
TOKEN_PROJECTION_SAFETY_FACTOR = 1.25
TOKEN_PROJECTION_PER_REQUEST_OVERHEAD = 64
CALL_ORDER_SEED = 230907
MAX_PRICE = {"prompt": 0.375, "completion": 1.875}
SIGNAL_TIE_BREAKER = "smaller_k_then_lower_temperature"

SENSITIVE_KEYS = {
    "api-key",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "openrouter_api_key",
    "proxy-authorization",
    "set-cookie",
    "x-api-key",
}
FORBIDDEN_OUTCOME_FIELDS = {
    "answer",
    "correct",
    "correctness",
    "gold",
    "ground_truth",
    "reference_answer",
    "reference_query",
    "reward",
    "task_score",
    "unit_tests",
}

ONE_LABEL_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "binary_self_attribution",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"label": {"type": "integer", "enum": [0, 1]}},
            "required": ["label"],
            "additionalProperties": False,
        },
    },
}


class ProtocolViolation(RuntimeError):
    """A non-transient response or experiment-protocol failure."""


class UpstreamTransientError(RuntimeError):
    """An upstream 429/5xx reported inside an otherwise successful HTTP response."""



@dataclass(frozen=True)
class PilotSettings:
    model_id: str
    expected_response_provider: str
    service_tier: str
    provider_routing: dict[str, Any]
    temperatures: tuple[float, ...]
    k_values: tuple[int, ...]
    independent_batches: int
    prompt_count: int
    datasets: int
    prompts_per_dataset: int
    rollouts_per_prompt: int
    max_tokens: int
    top_p: float
    reasoning: dict[str, Any]
    response_format: dict[str, Any]
    max_concurrency: int
    timeout_seconds: float
    max_attempts: int
    max_retry_wait_seconds: float
    minimum_label_rate: float
    maximum_label_rate: float
    minimum_variable_group_fraction: float
    minimum_independent_batch_pearson: float
    signal_selection_rule: str
    signal_tie_breaker: str
    canary_candidates_per_dataset: int
    canary_max_total_cost_usd: float
    max_projected_full_cost_usd: float
    full_hard_cost_cap_usd: float
    max_single_request_cost_usd: float
    output_dir: Path
    registry_manifest: Path
    registry_manifest_sha256: str
    canary_gate: Path
    canary_gate_sha256: str
    canary_execution_enabled: bool
    full_execution_enabled: bool

    @property
    def candidate_count(self) -> int:
        return self.prompt_count * self.rollouts_per_prompt

    @property
    def judgments_per_candidate(self) -> int:
        return max(self.k_values) * self.independent_batches * len(self.temperatures)

    @property
    def planned_judgments(self) -> int:
        return self.candidate_count * self.judgments_per_candidate

    @property
    def planned_canary_requests(self) -> int:
        return (
            self.datasets * self.canary_candidates_per_dataset * len(self.temperatures)
        )


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a YAML mapping: {path}")
    return value


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically publish one JSON artifact without ever exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable artifact: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to reuse temporary artifact: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def exclusive_phase_lock(phase_dir: Path) -> Iterator[TextIO]:
    """Hold a non-blocking process lock for a paid phase's complete lifecycle."""
    lock_path = phase_dir.with_name(f".{phase_dir.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProtocolViolation(
                f"Another closed-weight process owns the phase lock: {lock_path}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield handle
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    # File iteration splits only on physical newlines. ``str.splitlines`` also
    # treats U+2028/U+2029 inside otherwise valid JSON strings as delimiters.
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"Expected JSON objects in {path}")
    return rows


def _resolve_under(project_root: Path, value: str, label: str) -> Path:
    root = project_root.resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{label} must remain under project root {root}: {path}"
        ) from exc
    return path


def settings_from_config(
    raw: Mapping[str, Any], *, project_root: Path
) -> PilotSettings:
    if raw.get("schema_version") != 1:
        raise ValueError("Closed-weight pilot config must use schema_version=1")
    endpoint = raw.get("endpoint") or {}
    pilot = raw.get("pilot") or {}
    transport = raw.get("transport") or {}
    signal_gate = raw.get("signal_gate") or {}
    cost_control = raw.get("cost_control") or {}
    paths = raw.get("paths") or {}
    execution = raw.get("execution") or {}
    settings = PilotSettings(
        model_id=str(endpoint.get("model_id", "")),
        expected_response_provider=str(endpoint.get("expected_response_provider", "")),
        service_tier=str(endpoint.get("service_tier", "")),
        provider_routing=dict(endpoint.get("provider") or {}),
        temperatures=tuple(float(value) for value in pilot.get("temperatures", [])),
        k_values=tuple(int(value) for value in pilot.get("k_values", [])),
        independent_batches=int(pilot.get("independent_batches", 0)),
        prompt_count=int(pilot.get("prompt_count", 0)),
        datasets=int(pilot.get("datasets", 0)),
        prompts_per_dataset=int(pilot.get("prompts_per_dataset", 0)),
        rollouts_per_prompt=int(pilot.get("rollouts_per_prompt", 0)),
        max_tokens=int(endpoint.get("max_tokens", 0)),
        top_p=float(endpoint.get("top_p", 0.0)),
        reasoning=dict(endpoint.get("reasoning") or {}),
        response_format=dict(endpoint.get("response_format") or {}),
        max_concurrency=int(transport.get("max_concurrency", 0)),
        timeout_seconds=float(transport.get("timeout_seconds", 0.0)),
        max_attempts=int(transport.get("max_attempts", 0)),
        max_retry_wait_seconds=float(transport.get("max_retry_wait_seconds", 0.0)),
        minimum_label_rate=float(signal_gate.get("minimum_label_rate", -1.0)),
        maximum_label_rate=float(signal_gate.get("maximum_label_rate", -1.0)),
        minimum_variable_group_fraction=float(
            signal_gate.get("minimum_variable_group_fraction", -1.0)
        ),
        minimum_independent_batch_pearson=float(
            signal_gate.get("minimum_independent_batch_pearson", -2.0)
        ),
        signal_selection_rule=str(signal_gate.get("selection_rule", "")),
        signal_tie_breaker=str(signal_gate.get("tie_breaker", "")),
        canary_candidates_per_dataset=int(
            cost_control.get("canary_candidates_per_dataset", 0)
        ),
        canary_max_total_cost_usd=float(
            cost_control.get("canary_max_total_cost_usd", 0.0)
        ),
        max_projected_full_cost_usd=float(
            cost_control.get("max_projected_full_cost_usd", 0.0)
        ),
        full_hard_cost_cap_usd=float(cost_control.get("full_hard_cost_cap_usd", 0.0)),
        max_single_request_cost_usd=float(
            cost_control.get("max_single_request_cost_usd", 0.0)
        ),
        output_dir=_resolve_under(
            project_root, str(paths.get("output_dir", "")), "output_dir"
        ),
        registry_manifest=_resolve_under(
            project_root,
            str(paths.get("validation_registry_manifest", "")),
            "validation_registry_manifest",
        ),
        registry_manifest_sha256=str(
            paths.get("validation_registry_manifest_sha256", "")
        ),
        canary_gate=_resolve_under(
            project_root, str(paths.get("canary_gate", "")), "canary_gate"
        ),
        canary_gate_sha256=str(paths.get("canary_gate_sha256", "")),
        canary_execution_enabled=execution.get("canary_enabled") is True,
        full_execution_enabled=execution.get("full_enabled") is True,
    )
    validate_static_protocol(settings, raw)
    return settings


def validate_static_protocol(settings: PilotSettings, raw: Mapping[str, Any]) -> None:
    if settings.model_id != MODEL_ID:
        raise ValueError(f"model_id must be exactly {MODEL_ID!r}")
    if settings.expected_response_provider != EXPECTED_RESPONSE_PROVIDER:
        raise ValueError(
            f"expected_response_provider must be exactly {EXPECTED_RESPONSE_PROVIDER!r}"
        )
    if settings.service_tier != SERVICE_TIER:
        raise ValueError(f"service_tier must be exactly {SERVICE_TIER!r}")
    expected_routing = {
        "only": [PROVIDER_ENDPOINT],
        "allow_fallbacks": False,
        "require_parameters": True,
        "max_price": MAX_PRICE,
    }
    if settings.provider_routing != expected_routing:
        raise ValueError(f"provider routing must be exactly {expected_routing}")
    if settings.temperatures != TEMPERATURES:
        raise ValueError(f"temperature grid must be exactly {TEMPERATURES}")
    if settings.k_values != K_VALUES:
        raise ValueError(f"K grid must be exactly {K_VALUES}")
    if settings.independent_batches != INDEPENDENT_BATCHES:
        raise ValueError("independent_batches must be exactly 2")
    if (settings.prompt_count, settings.datasets, settings.prompts_per_dataset) != (
        100,
        2,
        50,
    ):
        raise ValueError(
            "pilot must contain 100 prompts, balanced 50/50 over two datasets"
        )
    if settings.rollouts_per_prompt != 4:
        raise ValueError("rollouts_per_prompt must be exactly 4")
    if (raw.get("pilot") or {}).get("k_design") != "nested_prefix_of_k_max":
        raise ValueError(
            "K design must reuse nested prefixes of each independent K=16 batch"
        )
    if settings.reasoning != {"effort": "medium", "exclude": True}:
        raise ValueError("reasoning must use fixed medium effort with hidden reasoning")
    if settings.response_format != ONE_LABEL_SCHEMA:
        raise ValueError("response_format must be the frozen one-label JSON schema")
    if settings.max_attempts != 1:
        raise ValueError("closed-weight judgments must use exactly one HTTP attempt")
    if settings.max_concurrency <= 0 or settings.timeout_seconds <= 0:
        raise ValueError("transport concurrency and timeout must be positive")
    if settings.max_retry_wait_seconds < 0:
        raise ValueError("max_retry_wait_seconds cannot be negative")
    if settings.max_tokens <= 0 or not 0 < settings.top_p <= 1:
        raise ValueError("max_tokens and top_p are invalid")
    if settings.canary_candidates_per_dataset != 1:
        raise ValueError("canary_candidates_per_dataset must be exactly 1")
    if (
        settings.canary_max_total_cost_usd != 0.10
        or settings.max_projected_full_cost_usd != 15.00
        or settings.full_hard_cost_cap_usd != 20.00
        or settings.max_single_request_cost_usd != 0.01
    ):
        raise ValueError(
            "Closed-weight cost ceilings differ from the approved protocol"
        )
    if settings.max_projected_full_cost_usd > settings.full_hard_cost_cap_usd:
        raise ValueError("Projected cost ceiling cannot exceed the full hard cost cap")
    if "logprobs" in (raw.get("endpoint") or {}):
        raise ValueError("closed-weight pilot must not request token log-probabilities")
    if (
        not 0 <= settings.minimum_label_rate < settings.maximum_label_rate <= 1
        or not 0 <= settings.minimum_variable_group_fraction <= 1
        or not -1 <= settings.minimum_independent_batch_pearson <= 1
        or settings.signal_selection_rule
        != "maximin_variable_groups_then_reliability_then_balance"
        or settings.signal_tie_breaker != SIGNAL_TIE_BREAKER
    ):
        raise ValueError("Closed-weight signal-gate thresholds are invalid")


def build_judge_prompt(task_statement: str, candidate: str) -> str:
    return (
        "You will be shown one task and one anonymized candidate response.\n\n"
        "The response may have been written by you, or by another model solving the same "
        "task. Decide only whether it could plausibly be your own response.\n\n"
        f"Task:\n{task_statement.strip()}\n\n"
        f"Candidate response:\n{candidate.strip()}\n\n"
        "Return exactly one JSON object with no explanation: "
        '{"label": 1} if it could plausibly be your response, or '
        '{"label": 0} otherwise.'
    )


def build_payload(
    settings: PilotSettings, *, prompt: str, temperature: float
) -> dict[str, Any]:
    if temperature not in settings.temperatures:
        raise ValueError(f"Temperature {temperature} is outside the frozen grid")
    payload = {
        "model": settings.model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": settings.max_tokens,
        "temperature": temperature,
        "top_p": settings.top_p,
        "service_tier": settings.service_tier,
        "provider": dict(settings.provider_routing),
        "reasoning": dict(settings.reasoning),
        "response_format": dict(settings.response_format),
        "usage": {"include": True},
    }
    # GPT-5 endpoints are routed only with their native sampling defaults.
    # Supplying OpenAI-incompatible temperature/top_p values makes OpenRouter
    # exclude otherwise eligible OpenAI endpoints during parameter filtering.
    if settings.model_id.startswith("openai/gpt-5"):
        payload.pop("temperature")
        payload.pop("top_p")
    return payload


def parse_binary_label(body: Mapping[str, Any]) -> int:
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProtocolViolation(
            "Response is missing choices[0].message.content"
        ) from exc
    try:
        value = json.loads(str(content))
    except json.JSONDecodeError as exc:
        raise ProtocolViolation("Binary judgment is not valid JSON") from exc
    if not isinstance(value, dict) or set(value) != {"label"}:
        raise ProtocolViolation("Binary judgment must contain only the label field")
    label = value["label"]
    if isinstance(label, bool) or not isinstance(label, int) or label not in {0, 1}:
        raise ProtocolViolation("Binary label must be the integer 0 or 1")
    return label


def verify_response_receipt(
    body: Mapping[str, Any], settings: PilotSettings
) -> dict[str, float]:
    generation_id = body.get("id")
    if not isinstance(generation_id, str) or not generation_id.strip():
        raise ProtocolViolation("Response is missing a generation ID")
    if body.get("model") != settings.model_id:
        raise ProtocolViolation(
            f"Returned model {body.get('model')!r} != {settings.model_id!r}"
        )
    if body.get("provider") != settings.expected_response_provider:
        raise ProtocolViolation(
            "Returned provider "
            f"{body.get('provider')!r} != {settings.expected_response_provider!r}"
        )
    if body.get("service_tier") != settings.service_tier:
        raise ProtocolViolation(
            "Returned service_tier "
            f"{body.get('service_tier')!r} != {settings.service_tier!r}"
        )
    usage = body.get("usage")
    if not isinstance(usage, dict):
        raise ProtocolViolation("Response is missing usage metadata")
    parsed: dict[str, float] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
        raw_value = usage.get(key)
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ProtocolViolation(f"usage.{key} is missing or non-numeric")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0:
            raise ProtocolViolation(f"usage.{key} must be finite and non-negative")
        parsed[key] = value
    if (
        parsed["prompt_tokens"] <= 0
        or parsed["completion_tokens"] <= 0
        or parsed["cost"] <= 0
        or parsed["completion_tokens"] > settings.max_tokens
        or not math.isclose(
            parsed["total_tokens"],
            parsed["prompt_tokens"] + parsed["completion_tokens"],
            abs_tol=0.0,
        )
    ):
        raise ProtocolViolation(
            "Usage token counts or cost are internally inconsistent"
        )
    maximum_route_cost = (
        parsed["prompt_tokens"] * FLEX_INPUT_USD_PER_MILLION
        + parsed["completion_tokens"] * FLEX_OUTPUT_USD_PER_MILLION
    ) / 1_000_000
    if parsed["cost"] > maximum_route_cost * 1.05 + 1e-8:
        raise ProtocolViolation("Receipt cost exceeds the frozen Flex route pricing")
    metadata = body.get("openrouter_metadata")
    if not isinstance(metadata, dict) or metadata.get("requested") != settings.model_id:
        raise ProtocolViolation("Response lacks matching OpenRouter router metadata")
    attempts = metadata.get("attempts")
    available = (metadata.get("endpoints") or {}).get("available")
    selected = (
        [item for item in available if item.get("selected") is True]
        if isinstance(available, list)
        else []
    )
    # OpenRouter's current GPT-5 response metadata is v2: it attests the one
    # selected endpoint but omits the legacy per-attempt array.  The root
    # receipt already pins model/provider/tier above, and this branch still
    # requires exactly one OpenAI endpoint selected on the first attempt.
    if settings.model_id.startswith("openai/gpt-5"):
        metadata_valid = (
            metadata.get("attempt") == 1
            and len(selected) == 1
            and selected[0].get("provider") == settings.expected_response_provider
        )
    else:
        metadata_valid = (
            metadata.get("attempt") == 1
            and isinstance(attempts, list)
            and len(attempts) == 1
            and attempts[0].get("provider") == settings.expected_response_provider
            and attempts[0].get("status") == 200
            and len(selected) == 1
            and selected[0].get("provider") == settings.expected_response_provider
        )
    if not metadata_valid:
        raise ProtocolViolation(
            "Router metadata does not prove the frozen single endpoint"
        )
    return parsed


def smoothed_probability(own_labels: int, k: int) -> float:
    if k <= 0 or own_labels < 0 or own_labels > k:
        raise ValueError("Require 0 <= own_labels <= K and K > 0")
    return (own_labels + 1) / (k + 2)


def sanitize_for_log(value: Any, *, secrets: Iterable[str] = ()) -> Any:
    secret_values = tuple(secret for secret in secrets if secret)
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if str(key).lower() in SENSITIVE_KEYS
                else sanitize_for_log(item, secrets=secret_values)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_for_log(item, secrets=secret_values) for item in value]
    if isinstance(value, str):
        result = value
        for secret in secret_values:
            result = result.replace(secret, "[REDACTED]")
        return result
    return value


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, UpstreamTransientError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or 500 <= status <= 599
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


def _retry_delay(exc: Exception, attempt: int, *, max_wait: float) -> float:
    fallback = min(5.0 * (2 ** (attempt - 1)), max_wait)
    if not isinstance(exc, httpx.HTTPStatusError) or exc.response.status_code != 429:
        return fallback
    retry_after = exc.response.headers.get("Retry-After")
    if retry_after is None:
        return fallback
    try:
        return min(max(float(retry_after), fallback), max_wait)
    except ValueError:
        return fallback


def _get_dotted(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def audit_required_gates(
    raw: Mapping[str, Any], *, project_root: Path
) -> dict[str, Any]:
    entries = (raw.get("gates") or {}).get("required")
    if not isinstance(entries, list) or not entries:
        raise ValueError("At least one prerequisite gate must be configured")
    results: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict) or not str(entry.get("name", "")).strip():
            raise ValueError("Every gate must be a named mapping")
        path = _resolve_under(project_root, str(entry.get("path", "")), "gate path")
        expected_sha256 = str(entry.get("sha256", ""))
        issues: list[str] = []
        actual_sha256: str | None = None
        if expected_sha256 in {"", "PENDING"}:
            issues.append("sha256_not_frozen")
        if not path.is_file():
            issues.append("file_missing")
        else:
            actual_sha256 = sha256_file(path)
            if (
                expected_sha256 not in {"", "PENDING"}
                and actual_sha256 != expected_sha256
            ):
                issues.append("sha256_mismatch")
            gate = read_json(path)
            expected = entry.get("expected") or {}
            if not isinstance(expected, dict):
                raise TypeError("gate.expected must be a mapping")
            for dotted, wanted in expected.items():
                if _get_dotted(gate, str(dotted)) != wanted:
                    issues.append(f"unexpected:{dotted}")
        results.append(
            {
                "name": str(entry["name"]),
                "path": str(path),
                "expected_sha256": expected_sha256,
                "actual_sha256": actual_sha256,
                "ready": not issues,
                "issues": issues,
            }
        )
    return {"ready": all(item["ready"] for item in results), "gates": results}


def load_validation_registry(
    manifest_path: Path, settings: PilotSettings, *, project_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ProtocolViolation("Validation registry must use schema_version=1")
    if manifest.get("split_role") != "validation":
        raise ProtocolViolation("Pilot registry must contain validation data only")
    if manifest.get("official_test_accessed") is not False:
        raise ProtocolViolation("Registry must attest official_test_accessed=false")
    if manifest.get("source_is_official_test") is not False:
        raise ProtocolViolation("Registry source cannot be an official test split")
    rows_path = _resolve_under(
        project_root, str(manifest.get("candidate_file", "")), "candidate_file"
    )
    expected_hash = str(manifest.get("candidate_file_sha256", ""))
    if (
        not rows_path.is_file()
        or not expected_hash
        or sha256_file(rows_path) != expected_hash
    ):
        raise ProtocolViolation("Candidate registry file is missing or hash-mismatched")
    rows = read_jsonl(rows_path)
    if len(rows) != settings.candidate_count:
        raise ProtocolViolation(
            f"Expected {settings.candidate_count} candidate rows, found {len(rows)}"
        )
    required = {
        "candidate_id",
        "dataset",
        "prompt_id",
        "rollout_index",
        "split_role",
        "task_statement",
        "candidate",
    }
    candidate_ids: set[str] = set()
    prompt_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required <= set(row):
            raise ProtocolViolation(f"Candidate row {index} is incomplete")
        forbidden = FORBIDDEN_OUTCOME_FIELDS & set(row)
        if forbidden:
            raise ProtocolViolation(
                f"Candidate row {index} exposes task outcomes: {sorted(forbidden)}"
            )
        if row["split_role"] != "validation":
            raise ProtocolViolation(f"Candidate row {index} is not validation-only")
        candidate_id = str(row["candidate_id"])
        if not candidate_id or candidate_id in candidate_ids:
            raise ProtocolViolation("Candidate IDs must be unique and non-empty")
        candidate_ids.add(candidate_id)
        rollout_index = row["rollout_index"]
        if isinstance(rollout_index, bool) or not isinstance(rollout_index, int):
            raise ProtocolViolation("rollout_index must be an integer")
        if not str(row["task_statement"]).strip() or not str(row["candidate"]).strip():
            raise ProtocolViolation("Task statements and candidates cannot be empty")
        prompt_rows[(str(row["dataset"]), str(row["prompt_id"]))].append(row)
    if len(prompt_rows) != settings.prompt_count:
        raise ProtocolViolation(
            f"Expected {settings.prompt_count} unique prompts, found {len(prompt_rows)}"
        )
    by_dataset = Counter(dataset for dataset, _ in prompt_rows)
    if len(by_dataset) != settings.datasets or set(by_dataset.values()) != {
        settings.prompts_per_dataset
    }:
        raise ProtocolViolation(
            "Pilot prompts must be balanced across exactly two datasets: "
            f"{dict(by_dataset)}"
        )
    for key, group in prompt_rows.items():
        indices = sorted(int(row["rollout_index"]) for row in group)
        if indices != list(range(settings.rollouts_per_prompt)):
            raise ProtocolViolation(
                f"Prompt {key} does not contain rollout indices 0..3"
            )
    if manifest.get("prompt_count") != settings.prompt_count:
        raise ProtocolViolation("Registry manifest prompt_count disagrees with config")
    if manifest.get("candidate_count") != settings.candidate_count:
        raise ProtocolViolation(
            "Registry manifest candidate_count disagrees with config"
        )
    return manifest, rows


def iter_call_plan(
    rows: Iterable[Mapping[str, Any]], settings: PilotSettings
) -> Iterator[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for temperature in settings.temperatures:
        for batch_id in range(settings.independent_batches):
            for row in rows:
                for replicate_index in range(max(settings.k_values)):
                    candidate_id = str(row["candidate_id"])
                    call_id = sha256_text(
                        f"{candidate_id}|{temperature}|{batch_id}|{replicate_index}"
                    )[:24]
                    plan.append(
                        {
                            "call_id": call_id,
                            "candidate_id": candidate_id,
                            "dataset": str(row["dataset"]),
                            "prompt_id": str(row["prompt_id"]),
                            "rollout_index": int(row["rollout_index"]),
                            "temperature": temperature,
                            "batch_id": batch_id,
                            "replicate_index": replicate_index,
                            "prompt": build_judge_prompt(
                                str(row["task_statement"]), str(row["candidate"])
                            ),
                        }
                    )
    # Interleave temperatures, batches, prompts, and replicates deterministically
    # so temporal provider drift cannot masquerade as a condition effect.
    plan.sort(key=lambda spec: sha256_text(f"{CALL_ORDER_SEED}|{spec['call_id']}"))
    yield from plan


def protocol_fingerprint(raw: Mapping[str, Any]) -> str:
    """Fingerprint the scientific/cost protocol while excluding mutable run gates."""
    return sha256_text(
        canonical_json(
            {
                "schema_version": raw.get("schema_version"),
                "endpoint": raw.get("endpoint"),
                "pilot": raw.get("pilot"),
                "transport": raw.get("transport"),
                "cost_control": raw.get("cost_control"),
                "signal_gate": raw.get("signal_gate"),
                "judge_prompt_template_sha256": sha256_text(
                    build_judge_prompt("{TASK}", "{CANDIDATE}")
                ),
                "call_order_seed": CALL_ORDER_SEED,
                "runtime_source_sha256": sha256_file(Path(__file__).resolve()),
                "pricing_usd_per_million": {
                    "input": FLEX_INPUT_USD_PER_MILLION,
                    "output": FLEX_OUTPUT_USD_PER_MILLION,
                },
            }
        )
    )


def select_canary_plan(
    rows: Iterable[Mapping[str, Any]], settings: PilotSettings
) -> list[dict[str, Any]]:
    """Select the longest candidate per dataset and judge it at every temperature."""
    by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
    if len(by_dataset) != settings.datasets:
        raise ProtocolViolation(
            "Canary selection did not find the frozen dataset count"
        )
    selected: list[Mapping[str, Any]] = []
    for dataset in sorted(by_dataset):
        ordered = sorted(
            by_dataset[dataset],
            key=lambda row: (
                len(
                    build_judge_prompt(
                        str(row["task_statement"]), str(row["candidate"])
                    ).encode("utf-8")
                ),
                str(row["candidate_id"]),
            ),
            reverse=True,
        )
        selected.extend(ordered[: settings.canary_candidates_per_dataset])
    plan: list[dict[str, Any]] = []
    for row in selected:
        for temperature in settings.temperatures:
            candidate_id = str(row["candidate_id"])
            call_id = sha256_text(f"canary|{candidate_id}|{temperature}")[:24]
            plan.append(
                {
                    "call_id": call_id,
                    "candidate_id": candidate_id,
                    "dataset": str(row["dataset"]),
                    "prompt_id": str(row["prompt_id"]),
                    "rollout_index": int(row["rollout_index"]),
                    "temperature": temperature,
                    "batch_id": 0,
                    "replicate_index": 0,
                    "prompt": build_judge_prompt(
                        str(row["task_statement"]), str(row["candidate"])
                    ),
                }
            )
            plan[-1]["prompt_utf8_bytes"] = len(plan[-1]["prompt"].encode("utf-8"))
    if len(plan) != settings.planned_canary_requests:
        raise ProtocolViolation("Canary plan does not match the frozen request count")
    return plan


def call_plan_sha256(plan: Iterable[Mapping[str, Any]]) -> str:
    frozen = []
    for spec in plan:
        frozen.append(
            {
                "call_id": str(spec["call_id"]),
                "candidate_id": str(spec["candidate_id"]),
                "dataset": str(spec["dataset"]),
                "prompt_id": str(spec["prompt_id"]),
                "rollout_index": int(spec["rollout_index"]),
                "temperature": float(spec["temperature"]),
                "batch_id": int(spec["batch_id"]),
                "replicate_index": int(spec["replicate_index"]),
                "judge_prompt_sha256": sha256_text(str(spec["prompt"])),
            }
        )
    return sha256_text(canonical_json(frozen))


def expected_run_manifest(
    *,
    raw: Mapping[str, Any],
    settings: PilotSettings,
    mode: str,
    plan: list[dict[str, Any]],
    config_sha256: str,
) -> dict[str, Any]:
    if mode not in {"canary", "full"}:
        raise ValueError("Paid run manifest mode must be canary or full")
    log_phase = "canary" if mode == "canary" else "pilot"
    gate_hashes = {
        str(entry["name"]): str(entry["sha256"])
        for entry in (raw.get("gates") or {}).get("required", [])
    }
    return {
        "schema_version": 1,
        "protocol": "gemini-3.8-flash-closed-weight-pilot-v1",
        "mode": mode,
        "log_phase": log_phase,
        "official_test_accessed": False,
        "model_id": settings.model_id,
        "provider_endpoint": PROVIDER_ENDPOINT,
        "protocol_fingerprint": protocol_fingerprint(raw),
        "config_sha256": config_sha256,
        "validation_registry_manifest_sha256": settings.registry_manifest_sha256,
        "required_gate_sha256": gate_hashes,
        "canary_gate_sha256": (settings.canary_gate_sha256 if mode == "full" else None),
        "call_plan_sha256": call_plan_sha256(plan),
        "planned_requests": len(plan),
        "max_http_attempts_per_logical_request": settings.max_attempts,
        "response_cache_enabled": False,
    }


def prepare_or_verify_phase_dir(
    phase_dir: Path, expected_manifest: Mapping[str, Any]
) -> None:
    """Create a paid phase once, or prove an interrupted phase is the same run."""
    manifest_path = phase_dir / "run_manifest.json"
    if not phase_dir.exists():
        phase_dir.mkdir(parents=True, exist_ok=False)
    if manifest_path.exists():
        actual = read_json(manifest_path)
        if actual != dict(expected_manifest):
            raise ProtocolViolation(
                f"Existing paid phase manifest does not match this protocol: {manifest_path}"
            )
        return
    unexpected = [path.name for path in phase_dir.iterdir()]
    if unexpected:
        raise ProtocolViolation(
            "Paid phase directory has artifacts but no run manifest; refusing an "
            f"ambiguous resume: {sorted(unexpected)}"
        )
    write_json_atomic(manifest_path, expected_manifest)


def audit_phase_log(
    log_path: Path,
    *,
    plan: list[dict[str, Any]],
    phase: str,
) -> dict[str, Any]:
    """Prove which logical calls can be skipped without issuing a duplicate."""
    if phase not in {"canary", "pilot"}:
        raise ValueError("phase must be canary or pilot")
    plan_by_id = {str(spec["call_id"]): spec for spec in plan}
    if len(plan_by_id) != len(plan):
        raise ProtocolViolation("Paid call plan contains duplicate call IDs")
    rows = read_jsonl(log_path) if log_path.exists() else []
    by_call_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        call_id = str(row.get("call_id", ""))
        spec = plan_by_id.get(call_id)
        if spec is None:
            raise ProtocolViolation(
                f"Raw log row {row_index} references an unknown call ID"
            )
        if row.get("phase") != phase or row.get("attempt") != 1:
            raise ProtocolViolation(
                f"Raw log row {row_index} has an unexpected phase or attempt"
            )
        for key, caster in (
            ("candidate_id", str),
            ("dataset", str),
            ("prompt_id", str),
            ("rollout_index", int),
            ("temperature", float),
            ("batch_id", int),
            ("replicate_index", int),
        ):
            try:
                observed = caster(row.get(key))
            except (TypeError, ValueError) as exc:
                raise ProtocolViolation(
                    f"Raw log row {row_index} has invalid {key}"
                ) from exc
            if observed != caster(spec[key]):
                raise ProtocolViolation(
                    f"Raw log row {row_index} disagrees with the frozen plan on {key}"
                )
        if row.get("judge_prompt_sha256") != sha256_text(str(spec["prompt"])):
            raise ProtocolViolation(
                f"Raw log row {row_index} uses a different judge prompt"
            )
        if "prompt_utf8_bytes" in spec and int(row.get("prompt_utf8_bytes", -1)) != int(
            spec["prompt_utf8_bytes"]
        ):
            raise ProtocolViolation(
                f"Raw log row {row_index} has a different prompt byte count"
            )
        if row.get("status") not in {"intent", "ok", "error"}:
            raise ProtocolViolation(f"Raw log row {row_index} has an invalid status")
        by_call_id[call_id].append(row)

    completed: set[str] = set()
    generation_ids: set[str] = set()
    observed_cost = 0.0
    ambiguous: list[str] = []
    for call_id, call_rows in by_call_id.items():
        statuses = [str(row["status"]) for row in call_rows]
        if statuses != ["intent", "ok"]:
            ambiguous.append(call_id)
            continue
        ok_row = call_rows[1]
        generation_id = ok_row.get("generation_id")
        if (
            not isinstance(generation_id, str)
            or not generation_id
            or generation_id in generation_ids
        ):
            raise ProtocolViolation(
                "Successful receipts have missing/duplicate generation IDs"
            )
        generation_ids.add(generation_id)
        raw_cost = (ok_row.get("usage") or {}).get("cost")
        if (
            isinstance(raw_cost, bool)
            or not isinstance(raw_cost, (int, float))
            or not math.isfinite(float(raw_cost))
            or float(raw_cost) <= 0
        ):
            raise ProtocolViolation(
                "Successful receipt has no positive accountable cost"
            )
        observed_cost += float(raw_cost)
        completed.add(call_id)
    if ambiguous:
        raise ProtocolViolation(
            "Paid phase contains unresolved or failed logical calls; automatic retry "
            f"could duplicate billing: {sorted(ambiguous)[:8]}"
        )
    return {
        "rows": len(rows),
        "attempted_calls": len(by_call_id),
        "completed_call_ids": completed,
        "completed_calls": len(completed),
        "pending_calls": len(plan) - len(completed),
        "observed_cost_usd": observed_cost,
        "generation_ids": len(generation_ids),
    }


def validate_canary_gate(
    raw: Mapping[str, Any], settings: PilotSettings
) -> dict[str, Any]:
    issues: list[str] = []
    actual_sha256: str | None = None
    gate: dict[str, Any] | None = None
    if settings.canary_gate_sha256 in {"", "PENDING"}:
        issues.append("sha256_not_frozen")
    if not settings.canary_gate.is_file():
        issues.append("file_missing")
    else:
        actual_sha256 = sha256_file(settings.canary_gate)
        if (
            settings.canary_gate_sha256 not in {"", "PENDING"}
            and actual_sha256 != settings.canary_gate_sha256
        ):
            issues.append("sha256_mismatch")
        try:
            gate = read_json(settings.canary_gate)
            expected = {
                "schema_version": 1,
                "passed": True,
                "official_test_accessed": False,
                "model_id": settings.model_id,
                "provider_endpoint": PROVIDER_ENDPOINT,
                "response_provider": settings.expected_response_provider,
                "service_tier": settings.service_tier,
                "protocol_fingerprint": protocol_fingerprint(raw),
                "validation_registry_manifest_sha256": settings.registry_manifest_sha256,
                "successful_requests": settings.planned_canary_requests,
            }
            for key, wanted in expected.items():
                if gate.get(key) != wanted:
                    issues.append(f"unexpected:{key}")
            observed = gate.get("observed_total_cost_usd")
            maximum = gate.get("maximum_request_cost_usd")
            projected = gate.get("conservative_full_cost_projection_usd")
            mean_projected = gate.get("mean_full_cost_projection_usd")
            maximum_projected = gate.get("naive_max_receipt_full_cost_projection_usd")
            for key, value, ceiling in (
                (
                    "observed_total_cost_usd",
                    observed,
                    settings.canary_max_total_cost_usd,
                ),
                (
                    "maximum_request_cost_usd",
                    maximum,
                    settings.max_single_request_cost_usd,
                ),
                (
                    "conservative_full_cost_projection_usd",
                    projected,
                    settings.max_projected_full_cost_usd,
                ),
                (
                    "mean_full_cost_projection_usd",
                    mean_projected,
                    settings.max_projected_full_cost_usd,
                ),
                (
                    "naive_max_receipt_full_cost_projection_usd",
                    maximum_projected,
                    settings.max_projected_full_cost_usd,
                ),
            ):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0
                    or float(value) > ceiling
                ):
                    issues.append(f"invalid_or_over_budget:{key}")
        except Exception as exc:  # noqa: BLE001 - report malformed gates fail-closed
            issues.append(f"invalid_gate:{exc}")
    return {
        "ready": not issues,
        "path": str(settings.canary_gate),
        "expected_sha256": settings.canary_gate_sha256,
        "actual_sha256": actual_sha256,
        "issues": issues,
        "gate": gate,
    }


class GeminiClosedWeightPilot:
    def __init__(
        self,
        *,
        settings: PilotSettings,
        api_key: str,
        log_path: Path,
        transport: httpx.AsyncBaseTransport | None = None,
        initial_observed_cost_usd: float = 0.0,
    ) -> None:
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing")
        self.settings = settings
        self.api_key = api_key
        self.log_path = log_path
        self.transport = transport
        self._log_lock = asyncio.Lock()
        if (
            not math.isfinite(initial_observed_cost_usd)
            or initial_observed_cost_usd < 0
        ):
            raise ValueError("Initial observed cost must be finite and non-negative")
        self._budget_condition = asyncio.Condition()
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        self._observed_cost_usd = initial_observed_cost_usd
        self._reserved_cost_usd = 0.0

    async def _append_log(self, row: dict[str, Any]) -> None:
        sanitized = sanitize_for_log(row, secrets=(self.api_key,))
        async with self._log_lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sanitized, ensure_ascii=False) + "\n")

    def _phase_ceiling(self, phase: str) -> float:
        return (
            self.settings.canary_max_total_cost_usd
            if phase == "canary"
            else self.settings.full_hard_cost_cap_usd
        )

    async def _reserve_cost(self, *, phase: str) -> None:
        reservation = self.settings.max_single_request_cost_usd
        ceiling = self._phase_ceiling(phase)
        async with self._budget_condition:
            while (
                self._observed_cost_usd + self._reserved_cost_usd + reservation
                > ceiling
            ):
                if self._reserved_cost_usd == 0:
                    raise ProtocolViolation(
                        f"No {phase} request can start without exceeding the "
                        f"frozen ${ceiling:.2f} ceiling"
                    )
                await self._budget_condition.wait()
            self._reserved_cost_usd += reservation

    async def _release_cost_reservation(self) -> None:
        async with self._budget_condition:
            self._reserved_cost_usd -= self.settings.max_single_request_cost_usd
            if self._reserved_cost_usd < -1e-12:
                raise RuntimeError("Cost-reservation accounting underflow")
            self._reserved_cost_usd = max(0.0, self._reserved_cost_usd)
            self._budget_condition.notify_all()

    async def _settle_cost(self, cost: float, *, phase: str) -> None:
        ceiling = self._phase_ceiling(phase)
        violation: str | None = None
        async with self._budget_condition:
            self._reserved_cost_usd -= self.settings.max_single_request_cost_usd
            self._reserved_cost_usd = max(0.0, self._reserved_cost_usd)
            self._observed_cost_usd += cost
            if cost > self.settings.max_single_request_cost_usd:
                violation = (
                    f"Request cost ${cost:.6f} exceeds the frozen per-request ceiling"
                )
            elif self._observed_cost_usd > ceiling:
                violation = (
                    f"Observed {phase} cost ${self._observed_cost_usd:.6f} exceeds "
                    f"the frozen ${ceiling:.2f} ceiling"
                )
            self._budget_condition.notify_all()
        if violation is not None:
            raise ProtocolViolation(violation)

    async def judge_one(
        self,
        client: httpx.AsyncClient,
        spec: Mapping[str, Any],
        *,
        phase: str = "pilot",
    ) -> int:
        payload = build_payload(
            self.settings,
            prompt=str(spec["prompt"]),
            temperature=float(spec["temperature"]),
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Metadata": "enabled",
            "X-OpenRouter-Cache": "false",
            "X-OpenRouter-Title": "iar-gemini-closed-weight-pilot",
        }
        last_error: Exception | None = None
        for attempt in range(1, self.settings.max_attempts + 1):
            started = time.perf_counter()
            raw_body: Any = None
            response_status_code: int | None = None
            response_headers: dict[str, str] = {}
            reserved = False
            settled = False
            try:
                await self._reserve_cost(phase=phase)
                reserved = True
                await self._append_log(
                    {
                        **{
                            key: value for key, value in spec.items() if key != "prompt"
                        },
                        "phase": phase,
                        "attempt": attempt,
                        "status": "intent",
                        "billing_status": "unresolved",
                        "judge_prompt_sha256": sha256_text(str(spec["prompt"])),
                    }
                )
                async with self._semaphore:
                    response = await client.post(
                        OPENROUTER_URL, headers=headers, json=payload
                    )
                response_status_code = response.status_code
                response_headers = {
                    key.lower(): value
                    for key, value in response.headers.items()
                    if key.lower() in {"x-generation-id", "x-openrouter-cache-status"}
                }
                try:
                    raw_body = response.json()
                except ValueError:
                    raw_body = {"non_json_body": response.text}
                response.raise_for_status()
                if not isinstance(raw_body, dict):
                    raise ProtocolViolation(
                        "OpenRouter response body is not a JSON object"
                    )
                raw_cost = (raw_body.get("usage") or {}).get("cost")
                if (
                    not isinstance(raw_cost, bool)
                    and isinstance(raw_cost, (int, float))
                    and math.isfinite(float(raw_cost))
                    and float(raw_cost) >= 0
                ):
                    settled = True
                    await self._settle_cost(float(raw_cost), phase=phase)
                upstream_error = raw_body.get("error")
                if isinstance(upstream_error, dict):
                    code = upstream_error.get("code")
                    if (
                        not isinstance(code, bool)
                        and isinstance(code, int)
                        and (code == 429 or 500 <= code <= 599)
                    ):
                        raise UpstreamTransientError(
                            f"Upstream error {code} inside HTTP {response_status_code}: "
                            f"{str(upstream_error.get('message', ''))[:200]}"
                        )
                usage = verify_response_receipt(raw_body, self.settings)
                if not settled:
                    raise ProtocolViolation(
                        "Successful response has no accountable cost"
                    )
                label = parse_binary_label(raw_body)
                await self._append_log(
                    {
                        **{
                            key: value for key, value in spec.items() if key != "prompt"
                        },
                        "phase": phase,
                        "attempt": attempt,
                        "status": "ok",
                        "billing_status": "receipt_recorded",
                        "generation_id": raw_body.get("id"),
                        "label": label,
                        "judge_prompt_sha256": sha256_text(str(spec["prompt"])),
                        "request_parameters": {
                            key: payload[key]
                            for key in (
                                "model",
                                "max_tokens",
                                "temperature",
                                "top_p",
                                "service_tier",
                                "provider",
                                "reasoning",
                                "response_format",
                                "usage",
                            )
                            if key in payload
                        },
                        "returned_model": raw_body.get("model"),
                        "returned_provider": raw_body.get("provider"),
                        "returned_service_tier": raw_body.get("service_tier"),
                        "response_headers": response_headers,
                        "usage": usage,
                        "latency_seconds": time.perf_counter() - started,
                        "raw_response": raw_body,
                    }
                )
                return label
            except Exception as exc:  # noqa: BLE001 - terminal failures are audited
                if reserved and not settled:
                    await self._release_cost_reservation()
                last_error = exc
                error_cost = (
                    (raw_body.get("usage") or {}).get("cost")
                    if isinstance(raw_body, dict)
                    and isinstance(raw_body.get("usage"), dict)
                    else None
                )
                explicit_error_without_generation = (
                    response_status_code is not None
                    and response_status_code >= 400
                    and not (
                        isinstance(raw_body, dict)
                        and (
                            raw_body.get("id")
                            or (
                                not isinstance(error_cost, bool)
                                and isinstance(error_cost, (int, float))
                                and math.isfinite(float(error_cost))
                                and float(error_cost) > 0
                            )
                        )
                    )
                )
                await self._append_log(
                    {
                        **{
                            key: value for key, value in spec.items() if key != "prompt"
                        },
                        "phase": phase,
                        "attempt": attempt,
                        "status": "error",
                        "billing_status": (
                            "receipt_recorded"
                            if settled
                            else (
                                "explicit_error_no_generation_receipt"
                                if explicit_error_without_generation
                                else "ambiguous"
                            )
                        ),
                        "generation_id": (
                            raw_body.get("id") if isinstance(raw_body, dict) else None
                        ),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "transient": _is_transient(exc),
                        "judge_prompt_sha256": sha256_text(str(spec["prompt"])),
                        "latency_seconds": time.perf_counter() - started,
                        "response_headers": response_headers,
                        "response_status_code": response_status_code,
                        "usage": (
                            raw_body.get("usage")
                            if isinstance(raw_body, dict)
                            else None
                        ),
                        "raw_response": raw_body,
                    }
                )
                if attempt >= self.settings.max_attempts or not _is_transient(exc):
                    break
                await asyncio.sleep(
                    _retry_delay(
                        exc,
                        attempt,
                        max_wait=self.settings.max_retry_wait_seconds,
                    )
                )
        assert last_error is not None
        raise RuntimeError(
            f"Judgment failed for call_id={spec['call_id']}"
        ) from last_error

    async def run_canary(
        self,
        plan: list[dict[str, Any]],
        *,
        completed_call_ids: set[str] | None = None,
    ) -> None:
        if not self.settings.canary_execution_enabled:
            raise ProtocolViolation(
                "Paid canary execution is disabled in the frozen config"
            )
        async with httpx.AsyncClient(
            timeout=self.settings.timeout_seconds,
            transport=self.transport,
        ) as client:
            # Sequential execution keeps the six-request canary's cost stop exact.
            completed = completed_call_ids or set()
            for spec in plan:
                if str(spec["call_id"]) in completed:
                    continue
                await self.judge_one(client, spec, phase="canary")

    async def run_full(
        self,
        plan: list[dict[str, Any]],
        *,
        completed_call_ids: set[str] | None = None,
    ) -> None:
        if not self.settings.full_execution_enabled:
            raise ProtocolViolation(
                "Full paid execution is disabled in the frozen config"
            )
        completed = completed_call_ids or set()
        async with httpx.AsyncClient(
            timeout=self.settings.timeout_seconds,
            transport=self.transport,
        ) as client:
            chunk_size = max(1, self.settings.max_concurrency * 4)
            pending: list[dict[str, Any]] = []
            for spec in plan:
                if str(spec["call_id"]) in completed:
                    continue
                pending.append(spec)
                if len(pending) == chunk_size:
                    await asyncio.gather(
                        *(self.judge_one(client, item) for item in pending)
                    )
                    pending.clear()
            if pending:
                await asyncio.gather(
                    *(self.judge_one(client, item) for item in pending)
                )


def preflight(raw: Mapping[str, Any], *, project_root: Path) -> dict[str, Any]:
    settings = settings_from_config(raw, project_root=project_root)
    gate_report = audit_required_gates(raw, project_root=project_root)
    registry_report: dict[str, Any]
    if settings.registry_manifest.is_file():
        try:
            actual_manifest_sha256 = sha256_file(settings.registry_manifest)
            if settings.registry_manifest_sha256 in {"", "PENDING"}:
                raise ProtocolViolation(
                    "Validation registry manifest SHA-256 is not frozen"
                )
            if actual_manifest_sha256 != settings.registry_manifest_sha256:
                raise ProtocolViolation(
                    "Validation registry manifest SHA-256 mismatches config"
                )
            manifest, rows = load_validation_registry(
                settings.registry_manifest, settings, project_root=project_root
            )
            registry_report = {
                "ready": True,
                "manifest": str(settings.registry_manifest),
                "manifest_sha256": actual_manifest_sha256,
                "candidate_count": len(rows),
                "prompt_count": manifest["prompt_count"],
            }
        except Exception as exc:  # noqa: BLE001 - report all validation blockers
            registry_report = {
                "ready": False,
                "manifest": str(settings.registry_manifest),
                "error": str(exc),
            }
    else:
        registry_report = {
            "ready": False,
            "manifest": str(settings.registry_manifest),
            "error": "validation registry manifest is missing",
        }
    canary_gate_report = validate_canary_gate(raw, settings)
    prerequisites_ready = gate_report["ready"] and registry_report["ready"]
    return {
        "protocol": "gemini-3.8-flash-closed-weight-pilot-v1",
        "paid_calls_started": False,
        "static_protocol_ready": True,
        "gates": gate_report,
        "registry": registry_report,
        "canary_gate": canary_gate_report,
        "canary_execution_enabled": settings.canary_execution_enabled,
        "full_execution_enabled": settings.full_execution_enabled,
        "ready_to_run_canary": (
            prerequisites_ready and settings.canary_execution_enabled
        ),
        "ready_to_run_full": (
            prerequisites_ready
            and canary_gate_report["ready"]
            and settings.full_execution_enabled
        ),
        "planned_judgments": settings.planned_judgments,
        "planned_canary_requests": settings.planned_canary_requests,
        "total_planned_requests_before_retries": (
            settings.planned_judgments + settings.planned_canary_requests
        ),
        "maximum_http_attempts": (
            settings.planned_judgments + settings.planned_canary_requests
        )
        * settings.max_attempts,
        "cost_control": {
            "canary_max_total_cost_usd": settings.canary_max_total_cost_usd,
            "max_projected_full_cost_usd": settings.max_projected_full_cost_usd,
            "full_hard_cost_cap_usd": settings.full_hard_cost_cap_usd,
            "max_single_request_cost_usd": settings.max_single_request_cost_usd,
        },
        "model": settings.model_id,
        "provider_endpoint": PROVIDER_ENDPOINT,
        "expected_response_provider": settings.expected_response_provider,
        "service_tier": settings.service_tier,
    }


def summarize_canary(
    log_path: Path,
    *,
    raw: Mapping[str, Any],
    settings: PilotSettings,
    registry_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    registry_rows = list(registry_rows)
    canary_plan = select_canary_plan(registry_rows, settings)
    resume_audit = audit_phase_log(
        log_path,
        plan=canary_plan,
        phase="canary",
    )
    rows = read_jsonl(log_path)
    canary_rows = [row for row in rows if row.get("phase") == "canary"]
    successful = [row for row in canary_rows if row.get("status") == "ok"]
    logical_ids = {str(row.get("call_id", "")) for row in canary_rows}
    success_ids = {str(row.get("call_id", "")) for row in successful}
    costs = [float((row.get("usage") or {}).get("cost", 0.0)) for row in successful]
    observed_total = float(resume_audit["observed_cost_usd"])
    maximum_request = max(costs, default=0.0)
    token_ratios = [
        float((row.get("usage") or {}).get("prompt_tokens", 0.0))
        / int(row.get("prompt_utf8_bytes", 0))
        for row in successful
        if int(row.get("prompt_utf8_bytes", 0)) > 0
        and float((row.get("usage") or {}).get("prompt_tokens", 0.0)) > 0
    ]
    if len(token_ratios) != settings.planned_canary_requests:
        raise ProtocolViolation("Canary receipts cannot calibrate prompt-token density")
    maximum_tokens_per_byte = max(max(token_ratios), 1.0)
    registry_prompt_bytes = sum(
        len(
            build_judge_prompt(
                str(row["task_statement"]), str(row["candidate"])
            ).encode("utf-8")
        )
        for row in registry_rows
    )
    projected_input_tokens = math.ceil(
        registry_prompt_bytes
        * settings.judgments_per_candidate
        * maximum_tokens_per_byte
        * TOKEN_PROJECTION_SAFETY_FACTOR
        + settings.planned_judgments * TOKEN_PROJECTION_PER_REQUEST_OVERHEAD
    )
    projected_output_tokens = settings.planned_judgments * settings.max_tokens
    projected = (
        projected_input_tokens * FLEX_INPUT_USD_PER_MILLION
        + projected_output_tokens * FLEX_OUTPUT_USD_PER_MILLION
    ) / 1_000_000
    terminal_errors = sum(
        row.get("status") == "error" and row.get("transient") is not True
        for row in canary_rows
    )
    duplicate_successes = len(successful) - len(success_ids)
    mean_receipt_projection = (
        statistics.mean(costs) * settings.planned_judgments if costs else math.inf
    )
    maximum_receipt_projection = (
        maximum_request * settings.planned_judgments if costs else math.inf
    )
    passed = (
        len(successful) == settings.planned_canary_requests
        and len(logical_ids) == settings.planned_canary_requests
        and len(success_ids) == settings.planned_canary_requests
        and duplicate_successes == 0
        and terminal_errors == 0
        and observed_total <= settings.canary_max_total_cost_usd
        and maximum_request <= settings.max_single_request_cost_usd
        and projected <= settings.max_projected_full_cost_usd
        and mean_receipt_projection <= settings.max_projected_full_cost_usd
        and maximum_receipt_projection <= settings.max_projected_full_cost_usd
    )
    return {
        "schema_version": 1,
        "passed": passed,
        "official_test_accessed": False,
        "model_id": settings.model_id,
        "provider_endpoint": PROVIDER_ENDPOINT,
        "response_provider": settings.expected_response_provider,
        "service_tier": settings.service_tier,
        "protocol_fingerprint": protocol_fingerprint(raw),
        "validation_registry_manifest_sha256": settings.registry_manifest_sha256,
        "successful_requests": len(successful),
        "logical_requests": len(logical_ids),
        "duplicate_successes": duplicate_successes,
        "transient_error_attempts": sum(
            row.get("status") == "error" and row.get("transient") is True
            for row in canary_rows
        ),
        "terminal_error_attempts": terminal_errors,
        "observed_total_cost_usd": observed_total,
        "mean_request_cost_usd": statistics.mean(costs) if costs else None,
        "maximum_request_cost_usd": maximum_request,
        "maximum_observed_prompt_tokens_per_utf8_byte": maximum_tokens_per_byte,
        "token_projection_safety_factor": TOKEN_PROJECTION_SAFETY_FACTOR,
        "token_projection_per_request_overhead": TOKEN_PROJECTION_PER_REQUEST_OVERHEAD,
        "registry_prompt_utf8_bytes": registry_prompt_bytes,
        "projected_full_input_tokens": projected_input_tokens,
        "projected_full_output_tokens_at_request_ceiling": projected_output_tokens,
        "mean_full_cost_projection_usd": mean_receipt_projection,
        "naive_max_receipt_full_cost_projection_usd": maximum_receipt_projection,
        "conservative_full_cost_projection_usd": projected,
        "canary_max_total_cost_usd": settings.canary_max_total_cost_usd,
        "max_projected_full_cost_usd": settings.max_projected_full_cost_usd,
        "full_hard_cost_cap_usd": settings.full_hard_cost_cap_usd,
        "max_single_request_cost_usd": settings.max_single_request_cost_usd,
        "raw_log": str(log_path),
        "resume_audit": {
            key: value
            for key, value in resume_audit.items()
            if key not in {"completed_call_ids"}
        },
        "note": (
            "Longest validation candidate per dataset at each frozen temperature. "
            "The gated projection scales the maximum observed prompt-token/UTF-8-byte "
            "ratio (floored at one token per UTF-8 byte) over every registry prompt, "
            "adds 64 input tokens per request, applies a 25% safety factor, prices every "
            "output at max_tokens, and also gates mean/max receipt extrapolations; no task "
            "outcomes were read."
        ),
    }


def summarize_pilot(
    log_path: Path, *, k_values: tuple[int, ...] = K_VALUES
) -> dict[str, Any]:
    rows = read_jsonl(log_path)
    pilot_rows = [row for row in rows if row.get("phase") == "pilot"]
    ok = [row for row in pilot_rows if row.get("status") == "ok"]
    by_call_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pilot_rows:
        by_call_id[str(row.get("call_id", ""))].append(row)
    resolved_calls = sum(
        any(row.get("status") == "ok" for row in attempts)
        for attempts in by_call_id.values()
    )
    duplicate_successes = sum(
        max(0, sum(row.get("status") == "ok" for row in attempts) - 1)
        for attempts in by_call_id.values()
    )
    by_setting_candidate: dict[tuple[float, int, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in ok:
        key = (
            float(row["temperature"]),
            int(row["batch_id"]),
            str(row["candidate_id"]),
        )
        by_setting_candidate[key].append(row)
    conditions: list[dict[str, Any]] = []
    estimate_maps: dict[tuple[float, int, int], dict[str, float]] = {}
    for temperature in sorted({key[0] for key in by_setting_candidate}):
        for batch_id in sorted(
            {key[1] for key in by_setting_candidate if key[0] == temperature}
        ):
            candidate_keys = sorted(
                key
                for key in by_setting_candidate
                if key[0] == temperature and key[1] == batch_id
            )
            for k in k_values:
                estimates: list[tuple[dict[str, Any], float]] = []
                selected_calls: list[dict[str, Any]] = []
                complete = True
                for key in candidate_keys:
                    candidate_rows = sorted(
                        by_setting_candidate[key],
                        key=lambda row: int(row["replicate_index"]),
                    )
                    prefix = [
                        row for row in candidate_rows if int(row["replicate_index"]) < k
                    ]
                    if len(prefix) != k:
                        complete = False
                        continue
                    selected_calls.extend(prefix)
                    estimates.append(
                        (
                            prefix[0],
                            smoothed_probability(
                                sum(row["label"] for row in prefix), k
                            ),
                        )
                    )
                groups: dict[tuple[str, str], list[float]] = defaultdict(list)
                for metadata, estimate in estimates:
                    groups[
                        (str(metadata["dataset"]), str(metadata["prompt_id"]))
                    ].append(estimate)
                variable_groups = sum(
                    len(values) == 4 and statistics.pvariance(values) > 0
                    for values in groups.values()
                )
                labels = [int(row["label"]) for row in selected_calls]
                costs = [
                    float((row.get("usage") or {}).get("cost", 0.0))
                    for row in selected_calls
                ]
                p_values = [estimate for _, estimate in estimates]
                estimate_maps[(temperature, batch_id, k)] = {
                    str(metadata["candidate_id"]): estimate
                    for metadata, estimate in estimates
                }
                conditions.append(
                    {
                        "temperature": temperature,
                        "batch_id": batch_id,
                        "k": k,
                        "complete": complete,
                        "candidate_count": len(estimates),
                        "judgment_count": len(labels),
                        "own_label_rate": statistics.mean(labels) if labels else None,
                        "p_own_smoothed_mean": statistics.mean(p_values)
                        if p_values
                        else None,
                        "p_own_smoothed_std": statistics.pstdev(p_values)
                        if p_values
                        else None,
                        "prompt_groups": len(groups),
                        "variable_g4_groups": variable_groups,
                        "variable_g4_group_fraction": (
                            variable_groups / len(groups) if groups else None
                        ),
                        "observed_cost_usd": sum(costs),
                    }
                )
    comparisons: list[dict[str, Any]] = []
    for temperature in sorted({key[0] for key in estimate_maps}):
        for k in k_values:
            first = estimate_maps.get((temperature, 0, k), {})
            second = estimate_maps.get((temperature, 1, k), {})
            candidate_ids = sorted(set(first) & set(second))
            first_values = [first[candidate_id] for candidate_id in candidate_ids]
            second_values = [second[candidate_id] for candidate_id in candidate_ids]
            comparisons.append(
                {
                    "temperature": temperature,
                    "k": k,
                    "complete": bool(candidate_ids)
                    and len(candidate_ids) == len(first) == len(second),
                    "paired_candidates": len(candidate_ids),
                    "mean_absolute_p_own_difference": (
                        statistics.mean(
                            abs(left - right)
                            for left, right in zip(
                                first_values, second_values, strict=True
                            )
                        )
                        if candidate_ids
                        else None
                    ),
                    "mean_signed_p_own_difference_batch1_minus_batch0": (
                        statistics.mean(
                            right - left
                            for left, right in zip(
                                first_values, second_values, strict=True
                            )
                        )
                        if candidate_ids
                        else None
                    ),
                    "pearson_p_own": _pearson(first_values, second_values),
                }
            )
    return {
        "raw_log": str(log_path),
        "rows": len(rows),
        "successful_pilot_judgments": len(ok),
        "logical_pilot_calls": len(by_call_id),
        "resolved_pilot_calls": resolved_calls,
        "unresolved_pilot_calls": len(by_call_id) - resolved_calls,
        "duplicate_successes": duplicate_successes,
        "transient_retry_attempts": sum(
            row.get("status") == "error" and row.get("transient") is True
            for row in pilot_rows
        ),
        "terminal_error_attempts": sum(
            row.get("status") == "error" and row.get("transient") is not True
            for row in pilot_rows
        ),
        "status_counts": dict(Counter(str(row.get("status")) for row in rows)),
        "conditions": conditions,
        "independent_batch_comparisons": comparisons,
        "note": "No task outcomes are read or used by this summary.",
    }


def build_signal_gate(
    summary: Mapping[str, Any], settings: PilotSettings
) -> dict[str, Any]:
    """Select a usable closed-weight signal without consulting task outcomes."""
    raw_conditions = summary.get("conditions")
    raw_comparisons = summary.get("independent_batch_comparisons")
    if not isinstance(raw_conditions, list) or not isinstance(raw_comparisons, list):
        raise ProtocolViolation("Signal gate requires complete pilot summaries")
    conditions: dict[tuple[float, int, int], Mapping[str, Any]] = {}
    for condition in raw_conditions:
        if not isinstance(condition, Mapping):
            raise ProtocolViolation("Signal-gate condition is malformed")
        key = (
            float(condition.get("temperature")),
            int(condition.get("batch_id")),
            int(condition.get("k")),
        )
        if key in conditions:
            raise ProtocolViolation("Signal-gate conditions contain duplicates")
        conditions[key] = condition
    comparisons: dict[tuple[float, int], Mapping[str, Any]] = {}
    for comparison in raw_comparisons:
        if not isinstance(comparison, Mapping):
            raise ProtocolViolation("Signal-gate comparison is malformed")
        key = (float(comparison.get("temperature")), int(comparison.get("k")))
        if key in comparisons:
            raise ProtocolViolation("Signal-gate comparisons contain duplicates")
        comparisons[key] = comparison

    candidates: list[dict[str, Any]] = []
    for temperature in settings.temperatures:
        for k in settings.k_values:
            batch_conditions = [
                conditions.get((temperature, batch_id, k))
                for batch_id in range(settings.independent_batches)
            ]
            comparison = comparisons.get((temperature, k))
            if any(item is None for item in batch_conditions) or comparison is None:
                raise ProtocolViolation(
                    "Signal gate is missing a temperature/K condition"
                )
            assert comparison is not None
            concrete = [item for item in batch_conditions if item is not None]
            try:
                label_rates = [float(item["own_label_rate"]) for item in concrete]
                variable_fractions = [
                    float(item["variable_g4_group_fraction"]) for item in concrete
                ]
                raw_reliability = comparison["pearson_p_own"]
                reliability = (
                    None if raw_reliability is None else float(raw_reliability)
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ProtocolViolation(
                    "Signal gate encountered a malformed non-numeric condition"
                ) from exc
            complete = all(item.get("complete") is True for item in concrete) and (
                comparison.get("complete") is True
            )
            label_rate_within_bounds = all(
                settings.minimum_label_rate <= rate <= settings.maximum_label_rate
                for rate in label_rates
            )
            minimum_variable_fraction = min(variable_fractions)
            average_label_rate = statistics.mean(label_rates)
            balance_score = 1.0 - 2.0 * abs(average_label_rate - 0.5)
            eligible = (
                complete
                and label_rate_within_bounds
                and minimum_variable_fraction
                >= settings.minimum_variable_group_fraction
                and reliability is not None
                and reliability >= settings.minimum_independent_batch_pearson
            )
            candidates.append(
                {
                    "temperature": temperature,
                    "k": k,
                    "complete": complete,
                    "batch_label_rates": label_rates,
                    "label_rate_within_bounds": label_rate_within_bounds,
                    "batch_variable_g4_group_fractions": variable_fractions,
                    "minimum_variable_g4_group_fraction": minimum_variable_fraction,
                    "independent_batch_pearson": reliability,
                    "average_label_rate": average_label_rate,
                    "balance_score": balance_score,
                    "eligible": eligible,
                }
            )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = (
        max(
            eligible,
            key=lambda candidate: (
                float(candidate["minimum_variable_g4_group_fraction"]),
                float(candidate["independent_batch_pearson"]),
                float(candidate["balance_score"]),
                -int(candidate["k"]),
                -float(candidate["temperature"]),
            ),
        )
        if eligible
        else None
    )
    return {
        "schema_version": 1,
        "passed": selected is not None,
        "decision": "GO" if selected is not None else "NO_GO_DEGENERATE_SIGNAL",
        "official_test_accessed": False,
        "selection_uses_task_outcomes": False,
        "thresholds": {
            "minimum_label_rate": settings.minimum_label_rate,
            "maximum_label_rate": settings.maximum_label_rate,
            "minimum_variable_group_fraction": (
                settings.minimum_variable_group_fraction
            ),
            "minimum_independent_batch_pearson": (
                settings.minimum_independent_batch_pearson
            ),
        },
        "selection_rule": settings.signal_selection_rule,
        "tie_breaker": settings.signal_tie_breaker,
        "selected_condition": selected,
        "candidates": candidates,
        "note": (
            "Selection uses only binary-label non-degeneracy, variation across the "
            "four actor rollouts, and reproducibility across independent judgment "
            "batches; task correctness is intentionally unavailable."
        ),
    }


def validate_complete_summary(
    summary: Mapping[str, Any], settings: PilotSettings
) -> None:
    if summary.get("successful_pilot_judgments") != settings.planned_judgments:
        raise ProtocolViolation("Pilot summary does not contain every planned judgment")
    if (
        summary.get("logical_pilot_calls") != settings.planned_judgments
        or summary.get("resolved_pilot_calls") != settings.planned_judgments
        or summary.get("unresolved_pilot_calls") != 0
        or summary.get("duplicate_successes") != 0
        or summary.get("transient_retry_attempts") != 0
        or summary.get("terminal_error_attempts") != 0
    ):
        raise ProtocolViolation(
            "Pilot summary contains unresolved or duplicate judgments"
        )
    conditions = summary.get("conditions")
    expected_conditions = (
        len(settings.temperatures)
        * settings.independent_batches
        * len(settings.k_values)
    )
    if not isinstance(conditions, list) or len(conditions) != expected_conditions:
        raise ProtocolViolation("Pilot summary has an incomplete condition grid")
    for condition in conditions:
        k = int(condition.get("k", 0))
        if (
            condition.get("complete") is not True
            or condition.get("candidate_count") != settings.candidate_count
            or condition.get("prompt_groups") != settings.prompt_count
            or condition.get("judgment_count") != settings.candidate_count * k
        ):
            raise ProtocolViolation("Pilot summary contains an incomplete condition")
    comparisons = summary.get("independent_batch_comparisons")
    expected_comparisons = len(settings.temperatures) * len(settings.k_values)
    if not isinstance(comparisons, list) or len(comparisons) != expected_comparisons:
        raise ProtocolViolation(
            "Pilot summary has incomplete independent-batch comparisons"
        )
    signal_gate = summary.get("signal_gate")
    if (
        not isinstance(signal_gate, Mapping)
        or signal_gate.get("passed") is not True
        or signal_gate.get("decision") != "GO"
        or signal_gate.get("official_test_accessed") is not False
        or signal_gate.get("selection_uses_task_outcomes") is not False
        or not isinstance(signal_gate.get("selected_condition"), Mapping)
    ):
        raise ProtocolViolation(
            "Closed-weight pilot did not pass its frozen signal gate"
        )


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True)
    )
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def run_from_config(
    config_path: Path, *, project_root: Path, mode: str = "preflight"
) -> dict[str, Any]:
    if mode not in {"preflight", "canary", "full"}:
        raise ValueError("mode must be one of: preflight, canary, full")
    raw = read_yaml(config_path)
    report = preflight(raw, project_root=project_root)
    if mode == "preflight":
        return report
    readiness_key = "ready_to_run_canary" if mode == "canary" else "ready_to_run_full"
    if not report[readiness_key]:
        raise ProtocolViolation(
            f"Paid {mode} run is blocked: preflight gates are not ready"
        )
    settings = settings_from_config(raw, project_root=project_root)
    _, rows = load_validation_registry(
        settings.registry_manifest, settings, project_root=project_root
    )
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing")
    phase_dir = settings.output_dir / mode
    log_path = phase_dir / "raw_judgments.jsonl"
    plan = (
        select_canary_plan(rows, settings)
        if mode == "canary"
        else list(iter_call_plan(rows, settings))
    )
    log_phase = "canary" if mode == "canary" else "pilot"
    run_manifest = expected_run_manifest(
        raw=raw,
        settings=settings,
        mode=mode,
        plan=plan,
        config_sha256=sha256_file(config_path),
    )
    with exclusive_phase_lock(phase_dir):
        prepare_or_verify_phase_dir(phase_dir, run_manifest)
        resume = audit_phase_log(log_path, plan=plan, phase=log_phase)
        runner = GeminiClosedWeightPilot(
            settings=settings,
            api_key=api_key,
            log_path=log_path,
            initial_observed_cost_usd=float(resume["observed_cost_usd"]),
        )
        completed_call_ids = set(resume["completed_call_ids"])
        if mode == "canary":
            asyncio.run(
                runner.run_canary(
                    plan,
                    completed_call_ids=completed_call_ids,
                )
            )
            final_audit = audit_phase_log(log_path, plan=plan, phase="canary")
            if final_audit["completed_calls"] != len(plan):
                raise ProtocolViolation("Canary ended without every frozen receipt")
            gate = summarize_canary(
                log_path,
                raw=raw,
                settings=settings,
                registry_rows=rows,
            )
            gate = {
                **gate,
                "run_manifest_sha256": sha256_file(phase_dir / "run_manifest.json"),
            }
            if settings.canary_gate.exists():
                if read_json(settings.canary_gate) != gate:
                    raise ProtocolViolation(
                        "Existing canary gate disagrees with the immutable raw ledger"
                    )
            else:
                write_json_atomic(settings.canary_gate, gate)
            return {
                **gate,
                "canary_gate_sha256": sha256_file(settings.canary_gate),
            }

        asyncio.run(
            runner.run_full(
                plan,
                completed_call_ids=completed_call_ids,
            )
        )
        final_audit = audit_phase_log(log_path, plan=plan, phase="pilot")
        if final_audit["completed_calls"] != len(plan):
            raise ProtocolViolation("Full pilot ended without every frozen receipt")
        summary = summarize_pilot(log_path)
        summary["schema_version"] = 1
        summary["official_test_accessed"] = False
        summary["protocol_fingerprint"] = protocol_fingerprint(raw)
        summary["run_manifest_sha256"] = sha256_file(phase_dir / "run_manifest.json")
        summary["resume_audit"] = {
            key: value
            for key, value in final_audit.items()
            if key != "completed_call_ids"
        }
        summary["signal_gate"] = build_signal_gate(summary, settings)
        summary["decision"] = summary["signal_gate"]["decision"]
        summary_path = phase_dir / "summary.json"
        signal_gate_path = phase_dir / "signal_gate.json"
        for artifact_path, artifact in (
            (summary_path, summary),
            (signal_gate_path, summary["signal_gate"]),
        ):
            if artifact_path.exists():
                if read_json(artifact_path) != artifact:
                    raise ProtocolViolation(
                        f"Existing artifact disagrees with the raw ledger: {artifact_path}"
                    )
            else:
                write_json_atomic(artifact_path, artifact)
        validate_complete_summary(summary, settings)
        return summary
