from __future__ import annotations

import unittest

from iar_phase0.data import MbppExample
from iar_phase0.prompts import (
    POINTWISE_SELF_ATTRIBUTION_NO_SYSTEM_PROMPT_VERSION,
    pointwise_self_attribution_template_sha256_for_version,
)
from iar_phase0.run_pointwise_self_attribution import judge_trial


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat_completion(self, payload: dict) -> dict:
        self.calls += 1
        if self.calls == 1:
            return {
                "model": "openai/gpt-oss-120b",
                "provider": "CoreWeave",
                "choices": [{"message": {"content": "0"}}],
            }
        return {
            "model": "openai/gpt-oss-120b",
            "provider": "CoreWeave",
            "choices": [
                {
                    "message": {"content": "0"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "0",
                                "top_logprobs": [
                                    {"token": "0", "logprob": -0.1},
                                    {"token": "1", "logprob": -2.1},
                                ],
                            }
                        ]
                    },
                }
            ],
        }


class PointwiseRetryTest(unittest.TestCase):
    def test_retries_http_200_response_missing_logprobs(self) -> None:
        client = FakeClient()
        version = POINTWISE_SELF_ATTRIBUTION_NO_SYSTEM_PROMPT_VERSION
        example = MbppExample(
            split="test",
            index=0,
            task_id=11,
            prompt="Write f.",
            code=None,
            test_setup_code="",
            tests=["assert f() == 1"],
        )
        trial = {
            "trial_id": 0,
            "split": "test",
            "index": 0,
            "task_id": 11,
            "candidate_source": "A0",
            "candidate_variant": "plain",
            "candidate_source_key": "plain",
            "candidate_unit_tests": {"passed": True},
            "candidate_text": "def f(): return 1",
            "ground_truth": 0,
        }
        result = judge_trial(
            trial=trial,
            example=example,
            client=client,  # type: ignore[arg-type]
            disc_cfg={
                "model_id": "openai/gpt-oss-120b",
                "judge_max_tokens": 256,
                "temperature": 0,
                "top_logprobs": 5,
                "strict_logprobs": True,
                "provider_routing": {},
                "request_overrides": {},
            },
            system_prompt=None,
            prompt_version=version,
            prompt_template_sha256=pointwise_self_attribution_template_sha256_for_version(
                version
            ),
            invalid_response_attempts=2,
            invalid_response_retry_delay_seconds=0,
        )
        self.assertEqual(client.calls, 2)
        self.assertEqual(result["judge"]["validation_attempt"], 2)
        self.assertTrue(result["judge"]["parsed"]["ok"])
        self.assertEqual(result["judge"]["message_roles"], ["user"])


if __name__ == "__main__":
    unittest.main()
