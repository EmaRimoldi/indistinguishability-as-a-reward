from __future__ import annotations

import asyncio
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from iar_phase0.generator import strip_code_fences
from iar_phase0.openrouter import (
    OPENROUTER_URL,
    binary_probability_from_logprobs,
    make_chat_payload,
    response_text,
)
from iar_phase0.prompts import (
    POINTWISE_SELF_ATTRIBUTION_PROMPT_VERSION,
    TARGET_MODEL_NAME,
    pointwise_self_attribution_prompt_for_version,
    pointwise_self_attribution_system_prompt_for_version,
    pointwise_self_attribution_template_sha256_for_version,
    sha256_text,
)

ATTRIBUTION_PROMPT_VERSION = POINTWISE_SELF_ATTRIBUTION_PROMPT_VERSION


class MisroutedJudgeResponseError(RuntimeError):
    """The response provably did not come from the pinned judge deployment.

    Raised when the routed model or provider differs from the pinned target, or
    when the response body is not valid JSON.  Such a response must never be
    used as a judgment, but the identical request may be retried: the failure
    is in OpenRouter's routing/transport layer, not in the judge's semantics.
    Responses from the correct deployment whose logprobs are present but violate
    the binary-label contract remain non-retryable."""


class IncompleteJudgeResponseError(RuntimeError):
    """A correctly-routed 200 response that omitted the requested logprobs.

    Raised when the response comes from the pinned deployment but the top
    log-probability payload the request explicitly asked for is structurally
    absent (missing choices, missing ``logprobs``, or empty ``content``).  This
    is a transient serialization/transport incompleteness on the provider side:
    the identical request may be retried and typically returns the payload.

    It is deliberately narrower than a semantic contract violation: when the
    logprobs ARE delivered but the binary labels are not in the top-k
    distribution, that signals a calibration problem that retrying cannot fix,
    so it stays a non-retryable ``RuntimeError``.  Retries are bounded by the
    existing ``max_attempts_per_call``; if the payload never arrives the caller
    still fails closed rather than fabricating a reward."""


class TruncatedBinaryLogprobDistributionError(RuntimeError):
    """The endpoint returned logprobs but top-k omitted one binary label.

    The generated label can be valid even when the alternative label ranks just
    below the requested top-k.  A bounded retry requests a wider top-logprob
    view without changing the prompt, generated-token budget, or reward
    definition.  If both labels remain absent, the call still fails closed.
    """


# Structurally-incomplete logprob payloads on an otherwise valid 200 response.
# These are transient (a fresh identical request usually returns the payload).
_TRANSIENT_LOGPROB_ERRORS = frozenset(
    {"missing_or_invalid_choices", "missing_logprobs", "missing_logprobs_content"}
)


def training_attribution_prompt(
    task_statement: str,
    candidate: str,
    version: str = ATTRIBUTION_PROMPT_VERSION,
) -> str:
    """The Phase 0 pointwise attribution question, parameterized by raw strings."""
    return pointwise_self_attribution_prompt_for_version(
        task_statement, candidate, version
    )


def attribution_prompt_template_sha256(
    version: str = ATTRIBUTION_PROMPT_VERSION,
) -> str:
    return pointwise_self_attribution_template_sha256_for_version(version)


def completion_to_text(completion: Any) -> str:
    """Normalize TRL standard or conversational completion payloads."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        contents = [
            str(message.get("content", ""))
            for message in completion
            if isinstance(message, dict) and message.get("role") == "assistant"
        ]
        if contents:
            return contents[-1]
    raise TypeError(f"Unsupported completion payload: {type(completion)!r}")


@dataclass(frozen=True)
class RewardSettings:
    model_id: str
    target_model_name: str
    max_reward_calls: int
    max_http_requests: int
    max_attempts_per_call: int
    max_concurrency: int
    timeout_seconds: float
    temperature: float
    max_tokens: int
    top_logprobs: int
    strict_logprobs: bool
    provider_routing: dict[str, Any]
    request_overrides: dict[str, Any]
    expected_provider: str | None = None
    reward_value: str = "log_p_own"
    probability_epsilon: float = 1e-6
    attribution_prompt_version: str = ATTRIBUTION_PROMPT_VERSION
    minimum_request_interval_seconds: float = 0.0
    max_retry_wait_seconds: float = 60.0
    store_raw_response: bool = False


def reward_from_binary_logprobs(
    parsed: dict[str, Any], settings: RewardSettings
) -> tuple[float, dict[str, float | str | None]]:
    """Transform the Discriminator's binary evidence into a training reward.

    ``log_p_own`` is the adopted reward.  The other monotone transforms are
    exposed for the preregistered reward pilot.  All probability-based
    transforms use the same explicit numerical guard.
    """
    epsilon = float(settings.probability_epsilon)
    if not 0.0 < epsilon < 0.5:
        raise ValueError("probability_epsilon must be strictly between 0 and 0.5")
    delta = float(parsed["logp_1"]) - float(parsed["logp_0"])
    p_own_raw = (
        1.0 / (1.0 + math.exp(-delta))
        if delta >= 0
        else math.exp(delta) / (1.0 + math.exp(delta))
    )
    p_own = min(max(p_own_raw, epsilon), 1.0 - epsilon)
    log_p_own = math.log(p_own)
    logit_p_own = math.log(p_own) - math.log1p(-p_own)
    binary_own = 1.0 if float(parsed["logp_1"]) > float(parsed["logp_0"]) else 0.0
    diagnostics: dict[str, float | str | None] = {
        "reward_value": settings.reward_value,
        "p_own_raw": p_own_raw,
        "p_own": p_own,
        "log_p_own": log_p_own,
        "logit_p_own": logit_p_own,
        "binary_own": binary_own,
        "probability_epsilon": epsilon,
        "delta": delta,
    }
    if settings.reward_value == "log_p_own":
        return log_p_own, diagnostics
    if settings.reward_value == "p_own":
        return p_own, diagnostics
    if settings.reward_value == "logit_p_own":
        return logit_p_own, diagnostics
    if settings.reward_value == "binary_own":
        return binary_own, diagnostics
    if settings.reward_value == "absolute_logp_1":
        return float(parsed["logp_1"]), diagnostics
    raise ValueError(f"Unsupported reward value: {settings.reward_value!r}")


class LiveOpenRouterReward:
    """Score every GRPO rollout through a live, audited OpenRouter request."""

    def __init__(
        self,
        *,
        settings: RewardSettings,
        log_path: Path,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        call_id_offset: int = 0,
        http_request_id_offset: int = 0,
        prompt_builder: Callable[[str, str], str] | None = None,
        prompt_template_hash: str | None = None,
        system_prompt_builder: Callable[[str], str | None] | None = None,
        candidate_normalizer: Callable[[str], str] | None = None,
    ) -> None:
        self.settings = settings
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing")
        self.transport = transport
        self.call_id_offset = int(call_id_offset)
        self.http_request_id_offset = int(http_request_id_offset)
        self.prompt_builder = prompt_builder
        self.prompt_template_hash = prompt_template_hash
        self.system_prompt_builder = system_prompt_builder
        self.candidate_normalizer = candidate_normalizer or strip_code_fences
        self.logical_calls = 0
        self.http_requests = 0

    async def score_batch(
        self,
        *,
        completions: list[Any],
        task_statement: list[str],
        task_id: list[str],
        dataset_index: list[int],
        rollout_index: list[int] | None = None,
    ) -> list[float]:
        size = len(completions)
        if not (len(task_statement) == len(task_id) == len(dataset_index) == size):
            raise ValueError("Reward metadata and completions must have equal lengths")
        if rollout_index is not None and len(rollout_index) != size:
            raise ValueError("rollout_index and completions must have equal lengths")

        if self.logical_calls + size > self.settings.max_reward_calls:
            raise RuntimeError(
                f"Reward-call budget exceeded: {self.logical_calls}+{size}>"
                f"{self.settings.max_reward_calls}"
            )
        first_call_id = self.call_id_offset + self.logical_calls + 1
        self.logical_calls += size

        counter_lock = asyncio.Lock()
        log_lock = asyncio.Lock()
        pacing_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(self.settings.max_concurrency)
        last_request_started = [0.0]

        rollout_counts: defaultdict[str, int] = defaultdict(int)
        async with httpx.AsyncClient(
            timeout=self.settings.timeout_seconds,
            transport=self.transport,
        ) as client:
            jobs = []
            for offset, completion in enumerate(completions):
                key = str(task_id[offset])
                if rollout_index is None:
                    current_rollout_index = rollout_counts[key]
                    rollout_counts[key] += 1
                else:
                    current_rollout_index = int(rollout_index[offset])
                raw_candidate = completion_to_text(completion)
                jobs.append(
                    self._score_one(
                        client=client,
                        call_id=first_call_id + offset,
                        task_statement=str(task_statement[offset]),
                        raw_candidate=raw_candidate,
                        task_id=key,
                        dataset_index=int(dataset_index[offset]),
                        rollout_index=current_rollout_index,
                        counter_lock=counter_lock,
                        log_lock=log_lock,
                        pacing_lock=pacing_lock,
                        last_request_started=last_request_started,
                        semaphore=semaphore,
                    )
                )
            return list(await asyncio.gather(*jobs))

    async def _reserve_http_request(self, counter_lock: asyncio.Lock) -> int:
        async with counter_lock:
            if self.http_requests >= self.settings.max_http_requests:
                raise RuntimeError(f"HTTP-request budget exhausted at {self.http_requests}")
            self.http_requests += 1
            return self.http_request_id_offset + self.http_requests

    async def _score_one(
        self,
        *,
        client: httpx.AsyncClient,
        call_id: int,
        task_statement: str,
        raw_candidate: str,
        task_id: str,
        dataset_index: int,
        rollout_index: int,
        counter_lock: asyncio.Lock,
        log_lock: asyncio.Lock,
        pacing_lock: asyncio.Lock,
        last_request_started: list[float],
        semaphore: asyncio.Semaphore,
    ) -> float:
        candidate = self.candidate_normalizer(raw_candidate)
        prompt_version = self.settings.attribution_prompt_version
        if self.prompt_builder is None:
            prompt = training_attribution_prompt(
                task_statement, candidate, version=prompt_version
            )
        else:
            prompt = self.prompt_builder(task_statement, candidate)
        if self.system_prompt_builder is None:
            system_prompt = pointwise_self_attribution_system_prompt_for_version(
                prompt_version, self.settings.target_model_name
            )
        else:
            system_prompt = self.system_prompt_builder(self.settings.target_model_name)
        base_payload = make_chat_payload(
            model_id=self.settings.model_id,
            system_prompt=system_prompt,
            prompt=prompt,
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature,
            logprobs=True,
            top_logprobs=self.settings.top_logprobs,
            provider_routing=self.settings.provider_routing,
            request_overrides=self.settings.request_overrides,
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": "iar-code-grpo",
            "HTTP-Referer": "https://example.invalid/iar-code",
        }

        last_error: Exception | None = None
        for attempt in range(1, self.settings.max_attempts_per_call + 1):
            payload = dict(base_payload)
            if isinstance(last_error, TruncatedBinaryLogprobDistributionError):
                payload["top_logprobs"] = max(
                    int(payload.get("top_logprobs") or 0), 20
                )
            request_id = await self._reserve_http_request(counter_lock)
            started = time.perf_counter()
            body: Any = None
            response_status_code: int | None = None
            response_headers: dict[str, str] = {}
            try:
                async with semaphore:
                    async with pacing_lock:
                        wait_seconds = (
                            self.settings.minimum_request_interval_seconds
                            - (time.perf_counter() - last_request_started[0])
                        )
                        if wait_seconds > 0:
                            await asyncio.sleep(wait_seconds)
                        last_request_started[0] = time.perf_counter()
                    response = await client.post(OPENROUTER_URL, headers=headers, json=payload)
                    response_status_code = response.status_code
                    response_headers = {
                        key.lower(): value
                        for key, value in response.headers.items()
                        if key.lower()
                        in {
                            "retry-after",
                            "x-generation-id",
                            "x-openrouter-cache-status",
                            "x-ratelimit-limit",
                            "x-ratelimit-remaining",
                            "x-ratelimit-reset",
                        }
                    }
                    try:
                        body = response.json()
                    except ValueError as decode_error:
                        raise MisroutedJudgeResponseError(
                            "Response body is not valid JSON"
                        ) from decode_error
                    response.raise_for_status()
                if not isinstance(body, dict):
                    raise MisroutedJudgeResponseError(
                        "Response body is not a JSON object"
                    )
                if body.get("model") != self.settings.model_id:
                    raise MisroutedJudgeResponseError(
                        f"Unexpected routed model {body.get('model')!r}; "
                        f"expected {self.settings.model_id!r}"
                    )
                if (
                    self.settings.expected_provider is not None
                    and body.get("provider") != self.settings.expected_provider
                ):
                    raise MisroutedJudgeResponseError(
                        f"Unexpected routed provider {body.get('provider')!r}; "
                        f"expected {self.settings.expected_provider!r}"
                    )
                parsed = binary_probability_from_logprobs(body)
                if self.settings.strict_logprobs and not parsed.get("ok"):
                    if parsed.get("error") in _TRANSIENT_LOGPROB_ERRORS:
                        raise IncompleteJudgeResponseError(
                            f"Judge response omitted requested logprobs: {parsed}"
                        )
                    if parsed.get("error") == "missing_binary_label_in_top_logprobs":
                        raise TruncatedBinaryLogprobDistributionError(
                            "Judge top-k distribution omitted a binary label: "
                            f"{parsed}"
                        )
                    raise RuntimeError(f"Binary log-probabilities unavailable: {parsed}")
                binary_label_mass = float(parsed["binary_label_mass"])
                reward, reward_diagnostics = reward_from_binary_logprobs(
                    parsed, self.settings
                )
                judge_token = response_text(body).strip()
                logprob_first_token = str(parsed.get("first_token", "")).strip()
                argmax_token = (
                    "1" if float(parsed["logp_1"]) > float(parsed["logp_0"]) else "0"
                )
                log_row: dict[str, Any] = {
                    "call_id": call_id,
                    "http_request_id": request_id,
                    "openrouter_response_id": body.get("id"),
                    "attempt": attempt,
                    "task_id": task_id,
                    "dataset_index": dataset_index,
                    "rollout_index": rollout_index,
                    "candidate": candidate,
                    "candidate_raw": raw_candidate,
                    "reward": reward,
                    **reward_diagnostics,
                    "logp_1": float(parsed["logp_1"]),
                    "logp_0": float(parsed["logp_0"]),
                    "binary_label_mass": binary_label_mass,
                    "judge_token": judge_token,
                    "logprob_first_token": logprob_first_token,
                    "argmax_token": argmax_token,
                    "emitted_argmax_consistent": judge_token == argmax_token,
                    "emitted_logprob_token_consistent": judge_token == logprob_first_token,
                    "raw_model": body.get("model"),
                    "raw_provider": body.get("provider"),
                    "judge_prompt_template_version": prompt_version,
                    "judge_prompt_template_sha256": self.prompt_template_hash
                    or attribution_prompt_template_sha256(prompt_version),
                    "judge_system_prompt_present": system_prompt is not None,
                    "judge_system_prompt_sha256": (
                        sha256_text(system_prompt) if system_prompt is not None else None
                    ),
                    "judge_message_roles": [
                        message["role"] for message in payload["messages"]
                    ],
                    "judge_user_prompt_sha256": sha256_text(prompt),
                    "request_parameters": {
                        "temperature": payload.get("temperature"),
                        "top_p": payload.get("top_p"),
                        "max_tokens": payload.get("max_tokens"),
                        "top_logprobs": payload.get("top_logprobs"),
                        "provider": payload.get("provider"),
                        "reasoning": payload.get("reasoning"),
                    },
                    "usage": body.get("usage"),
                    "response_status_code": response_status_code,
                    "response_headers": response_headers,
                    "latency_seconds": time.perf_counter() - started,
                    "status": "ok",
                }
                if self.settings.store_raw_response:
                    log_row["raw_response"] = _redact_sensitive(body)
                await self._append_log(
                    log_lock,
                    log_row,
                )
                return reward
            except Exception as exc:  # noqa: BLE001 - log every terminal failure
                last_error = exc
                error_row: dict[str, Any] = {
                        "call_id": call_id,
                        "http_request_id": request_id,
                        "attempt": attempt,
                        "task_id": task_id,
                        "dataset_index": dataset_index,
                        "rollout_index": rollout_index,
                        "candidate": candidate,
                        "candidate_raw": raw_candidate,
                        "latency_seconds": time.perf_counter() - started,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "retryable": _is_retryable_exception(exc),
                        "response_status_code": response_status_code,
                        "response_headers": response_headers,
                        "request_top_logprobs": payload.get("top_logprobs"),
                        "raw_model": body.get("model") if isinstance(body, dict) else None,
                        "raw_provider": (
                            body.get("provider") if isinstance(body, dict) else None
                        ),
                    }
                if self.settings.store_raw_response and body is not None:
                    error_row["raw_response"] = _redact_sensitive(body)
                await self._append_log(log_lock, error_row)
                if (
                    attempt < self.settings.max_attempts_per_call
                    and _is_retryable_exception(exc)
                ):
                    await asyncio.sleep(
                        _retry_delay_seconds(
                            exc,
                            attempt,
                            max_wait=self.settings.max_retry_wait_seconds,
                        )
                    )
                else:
                    break
        assert last_error is not None
        raise RuntimeError(f"Reward request failed for call_id={call_id}") from last_error

    async def _append_log(self, log_lock: asyncio.Lock, row: dict[str, Any]) -> None:
        async with log_lock:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def audit_summary(self) -> dict[str, int]:
        return {
            "logical_reward_calls": self.logical_calls,
            "http_requests": self.http_requests,
            "call_id_offset": self.call_id_offset,
            "http_request_id_offset": self.http_request_id_offset,
            "max_reward_calls": self.settings.max_reward_calls,
            "max_http_requests": self.settings.max_http_requests,
        }

    def reward_statistics(self) -> dict[str, Any]:
        return summarize_reward_log(self.log_path)

    def assert_complete(self) -> None:
        if self.logical_calls != self.settings.max_reward_calls:
            raise RuntimeError(
                f"Expected {self.settings.max_reward_calls} rewards, observed {self.logical_calls}"
            )
        if self.http_requests > self.settings.max_http_requests:
            raise RuntimeError(
                f"HTTP request budget exceeded: {self.http_requests}>"
                f"{self.settings.max_http_requests}"
            )


def settings_from_config(raw: dict[str, Any]) -> RewardSettings:
    discriminator = raw["discriminator"]
    reward = raw["reward"]
    return RewardSettings(
        model_id=str(discriminator["model_id"]),
        target_model_name=str(discriminator.get("target_model_name", TARGET_MODEL_NAME)),
        max_reward_calls=int(reward["max_reward_calls"]),
        max_http_requests=int(reward["max_http_requests"]),
        max_attempts_per_call=int(reward["max_attempts_per_call"]),
        max_concurrency=int(reward["max_concurrency"]),
        timeout_seconds=float(reward["timeout_seconds"]),
        temperature=float(discriminator["temperature"]),
        max_tokens=int(discriminator["judge_max_tokens"]),
        top_logprobs=int(discriminator["top_logprobs"]),
        strict_logprobs=bool(discriminator["strict_logprobs"]),
        provider_routing=dict(discriminator["provider_routing"]),
        request_overrides=dict(discriminator.get("request_overrides") or {}),
        expected_provider=(
            str(discriminator["expected_provider"])
            if discriminator.get("expected_provider") is not None
            else None
        ),
        reward_value=str(reward.get("value", "log_p_own")),
        probability_epsilon=float(reward.get("epsilon", 1e-6)),
        attribution_prompt_version=str(
            discriminator.get(
                "attribution_prompt_version", ATTRIBUTION_PROMPT_VERSION
            )
        ),
        minimum_request_interval_seconds=float(
            reward.get("minimum_request_interval_seconds", 0.0)
        ),
        max_retry_wait_seconds=float(reward.get("max_retry_wait_seconds", 60.0)),
        store_raw_response=bool(reward.get("store_raw_response", False)),
    )


def _retry_delay_seconds(
    exc: Exception,
    attempt: int,
    *,
    max_wait: float = 60.0,
) -> float:
    """Return a conservative retry delay, honoring provider rate-limit advice.

    A short 1/2-second retry loop is insufficient when concurrent Slurm trials
    share one OpenRouter account.  This changes only transport resilience: the
    request payload, reward and HPO protocol remain identical.
    """
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


def _is_retryable_exception(exc: Exception) -> bool:
    """Retry only failures recoverable without changing reward semantics."""
    if isinstance(
        exc,
        (
            MisroutedJudgeResponseError,
            IncompleteJudgeResponseError,
            TruncatedBinaryLogprobDistributionError,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or 500 <= status <= 599
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


def _redact_sensitive(value: Any) -> Any:
    """Recursively redact credential-shaped fields in stored raw responses."""
    if isinstance(value, dict):
        sensitive = {"api-key", "apikey", "authorization", "x-api-key"}
        return {
            key: (
                "[REDACTED]"
                if str(key).casefold().replace("_", "-") in sensitive
                else _redact_sensitive(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    return value


def summarize_reward_log(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    rewards = [float(row["reward"]) for row in ok_rows]
    p_own_values = [float(row.get("p_own", row["reward"])) for row in ok_rows]
    deltas = [float(row["delta"]) for row in ok_rows]
    if not rewards:
        return {"n": len(rows), "n_ok": 0}
    by_task: dict[str, Any] = {}
    task_ids = {str(row["task_id"]) for row in ok_rows}
    for task_id in sorted(task_ids, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)):
        group = [row for row in ok_rows if str(row["task_id"]) == task_id]
        values = [float(row["reward"]) for row in group]
        by_task[task_id] = {
            "n": len(group),
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
            "min": min(values),
            "max": max(values),
            "unique_candidates": len({str(row["candidate_raw"]) for row in group}),
        }
    usage_keys = ["prompt_tokens", "completion_tokens", "total_tokens", "cost"]
    usage = {
        key: sum(float((row.get("usage") or {}).get(key, 0) or 0) for row in ok_rows)
        for key in usage_keys
    }
    latencies = [float(row["latency_seconds"]) for row in ok_rows]
    return {
        "n": len(rows),
        "n_ok": len(ok_rows),
        "reward": {
            "mean": statistics.mean(rewards),
            "std": statistics.pstdev(rewards),
            "min": min(rewards),
            "max": max(rewards),
        },
        "p_own": {
            "mean": statistics.mean(p_own_values),
            "std": statistics.pstdev(p_own_values),
            "min": min(p_own_values),
            "max": max(p_own_values),
        },
        "delta": {
            "mean": statistics.mean(deltas),
            "std": statistics.pstdev(deltas),
            "min": min(deltas),
            "max": max(deltas),
        },
        "reward_values": dict(
            Counter(str(row.get("reward_value", "p_own")) for row in ok_rows)
        ),
        "judge_tokens": dict(Counter(str(row.get("judge_token")) for row in ok_rows)),
        "raw_models": dict(Counter(str(row.get("raw_model")) for row in ok_rows)),
        "raw_providers": dict(Counter(str(row.get("raw_provider")) for row in ok_rows)),
        "judge_prompt_template_versions": dict(
            Counter(str(row.get("judge_prompt_template_version")) for row in ok_rows)
        ),
        "judge_prompt_template_hashes": dict(
            Counter(str(row.get("judge_prompt_template_sha256")) for row in ok_rows)
        ),
        "judge_system_prompt_hashes": dict(
            Counter(str(row.get("judge_system_prompt_sha256")) for row in ok_rows)
        ),
        "latency_seconds": {"mean": statistics.mean(latencies), "max": max(latencies)},
        "usage": usage,
        "by_task": by_task,
    }
