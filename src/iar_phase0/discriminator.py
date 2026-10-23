from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .data import MbppExample
from .generator import strip_code_fences
from .openrouter import (
    OpenRouterClient,
    binary_probability_from_logprobs,
    make_chat_payload,
    response_text,
)
from .prompts import discriminator_judge_prompt, discriminator_own_answer_prompt, sha256_text


@dataclass
class Discriminator:
    client: OpenRouterClient
    model_id: str
    own_answer_max_tokens: int
    judge_max_tokens: int
    temperature: float
    top_logprobs: int
    provider_routing: dict[str, Any]
    request_overrides: dict[str, Any] | None = None
    strict_logprobs: bool = True

    def own_answer(self, example: MbppExample) -> dict[str, Any]:
        prompt = discriminator_own_answer_prompt(example)
        payload = make_chat_payload(
            model_id=self.model_id,
            prompt=prompt,
            max_tokens=self.own_answer_max_tokens,
            temperature=0.2,
            provider_routing=self.provider_routing,
            request_overrides=self.request_overrides,
        )
        response = self.client.chat_completion(payload)
        return {
            "text": strip_code_fences(response_text(response)),
            "raw_model": response.get("model"),
            "raw_provider": response.get("provider"),
            "user_prompt_sha256": sha256_text(prompt),
            "usage": response.get("usage"),
        }

    def judge(self, example: MbppExample, candidate_code: str) -> dict[str, Any]:
        prompt = discriminator_judge_prompt(example, candidate_code)
        payload = make_chat_payload(
            model_id=self.model_id,
            prompt=prompt,
            max_tokens=self.judge_max_tokens,
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=self.top_logprobs,
            provider_routing=self.provider_routing,
            request_overrides=self.request_overrides,
        )
        response = self.client.chat_completion(payload)
        parsed = binary_probability_from_logprobs(response)
        if self.strict_logprobs and not parsed["ok"]:
            raise RuntimeError(f"Could not extract binary log-probabilities: {parsed}")
        return {
            "text": response_text(response),
            "parsed": parsed,
            "raw_model": response.get("model"),
            "raw_provider": response.get("provider"),
            "user_prompt_sha256": sha256_text(prompt),
            "usage": response.get("usage"),
        }
