from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass(frozen=True)
class OpenRouterClient:
    api_key: str
    app_name: str = "iar-code-phase0"
    site_url: str = "https://example.invalid/iar-code"
    timeout_seconds: int = 120
    max_attempts: int = 4
    max_retry_wait_seconds: float = 60.0

    @classmethod
    def from_env(
        cls,
        *,
        max_attempts: int = 4,
        max_retry_wait_seconds: float = 60.0,
    ) -> "OpenRouterClient":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing.")
        return cls(
            api_key=api_key,
            app_name=os.environ.get("OPENROUTER_APP_NAME", cls.app_name),
            site_url=os.environ.get("OPENROUTER_SITE_URL", cls.site_url),
            max_attempts=max_attempts,
            max_retry_wait_seconds=max_retry_wait_seconds,
        )

    def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": self.app_name,
            "HTTP-Referer": self.site_url,
        }
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(OPENROUTER_URL, headers=headers, json=payload)
                    response.raise_for_status()
                    return response.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_error = exc
                if attempt >= self.max_attempts or not _is_retryable_request_error(exc):
                    raise
                time.sleep(
                    _request_retry_delay_seconds(
                        exc,
                        attempt,
                        max_wait=self.max_retry_wait_seconds,
                    )
                )
        assert last_error is not None
        raise last_error


def _is_retryable_request_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and (
        exc.response.status_code == 429 or exc.response.status_code >= 500
    )


def _request_retry_delay_seconds(
    exc: Exception,
    attempt: int,
    *,
    max_wait: float,
) -> float:
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


def make_chat_payload(
    *,
    model_id: str,
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int,
    temperature: float,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    provider_routing: dict[str, Any] | None = None,
    request_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if logprobs:
        payload["logprobs"] = True
    if top_logprobs is not None:
        payload["top_logprobs"] = top_logprobs
    if provider_routing:
        payload["provider"] = provider_routing
    if request_overrides:
        payload.update(request_overrides)
    return payload


def response_text(response: dict[str, Any]) -> str:
    return response["choices"][0]["message"]["content"]


def binary_probability_from_logprobs(response: dict[str, Any]) -> dict[str, Any]:
    """Extract P(label=1 | label in {0,1}) from first-token top log-probs.

    Returns a diagnostic dictionary. It does not silently invent missing label
    probabilities because missing labels usually mean the endpoint/tokenization
    must be calibrated.
    """
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return {
            "ok": False,
            "error": "missing_or_invalid_choices",
            "response_keys": sorted(str(key) for key in response),
        }
    choice = choices[0]
    logprobs = choice.get("logprobs")
    if not logprobs:
        return {"ok": False, "error": "missing_logprobs"}
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not content:
        return {"ok": False, "error": "missing_logprobs_content"}
    first: dict[str, Any] | None = None
    label_logprobs: dict[str, float] = {}
    content_position: int | None = None
    seen_positions: list[dict[str, Any]] = []
    for position, candidate in enumerate(content):
        top = candidate.get("top_logprobs") or []
        candidate_labels: dict[str, float] = {}
        for item in top:
            token = str(item.get("token", ""))
            normalized = token.strip()
            if normalized in {"0", "1"} and normalized not in candidate_labels:
                candidate_labels[normalized] = float(item["logprob"])
        seen_positions.append({
            "position": position,
            "emitted_token": str(candidate.get("token", "")),
            "top_tokens": [str(item.get("token", "")) for item in top[:20]],
        })
        if "0" in candidate_labels and "1" in candidate_labels:
            first = candidate
            label_logprobs = candidate_labels
            content_position = position
            break

    if first is None:
        return {
            "ok": False,
            "error": "missing_binary_label_in_top_logprobs",
            "seen_positions": seen_positions,
        }

    l0 = label_logprobs["0"]
    l1 = label_logprobs["1"]
    binary_label_mass = math.exp(l0) + math.exp(l1)
    max_log = max(l0, l1)
    p1 = math.exp(l1 - max_log) / (math.exp(l1 - max_log) + math.exp(l0 - max_log))
    return {
        "ok": True,
        "p1": p1,
        "logp_0": l0,
        "logp_1": l1,
        "binary_label_mass": binary_label_mass,
        "first_token": str(first.get("token", "")),
        "content_position": content_position,
    }
