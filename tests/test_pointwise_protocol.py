from __future__ import annotations

import unittest

from iar_phase0.openrouter import make_chat_payload
from iar_phase0.prompts import (
    POINTWISE_SELF_ATTRIBUTION_NO_SYSTEM_PROMPT_VERSION,
    pointwise_self_attribution_prompt_for_version,
    pointwise_self_attribution_system_prompt_for_version,
    pointwise_self_attribution_template_sha256_for_version,
)


class PointwiseProtocolV2Test(unittest.TestCase):
    def test_v2_is_one_user_message_with_exact_attribution_question(self) -> None:
        version = POINTWISE_SELF_ATTRIBUTION_NO_SYSTEM_PROMPT_VERSION
        prompt = pointwise_self_attribution_prompt_for_version(
            "STANDARD MBPP TASK AND TESTS",
            "def answer():\n    return 1",
            version,
        )
        system_prompt = pointwise_self_attribution_system_prompt_for_version(
            version, "OpenAI GPT-OSS 120B"
        )
        payload = make_chat_payload(
            model_id="openai/gpt-oss-120b",
            system_prompt=system_prompt,
            prompt=prompt,
            max_tokens=256,
            temperature=0,
            logprobs=True,
            top_logprobs=5,
        )

        self.assertIsNone(system_prompt)
        self.assertEqual([message["role"] for message in payload["messages"]], ["user"])
        self.assertEqual(payload["messages"][0]["content"], prompt)
        self.assertIn("Was this solution written by you?", prompt)
        self.assertIn("1 = yes, this was written by me", prompt)
        self.assertIn("0 = no, this was not written by me", prompt)
        self.assertNotIn("plausibly correct", prompt)
        self.assertNotIn("unnecessarily verbose", prompt)
        self.assertNotIn("oddly formatted", prompt)

    def test_v2_template_hash_is_stable(self) -> None:
        version = POINTWISE_SELF_ATTRIBUTION_NO_SYSTEM_PROMPT_VERSION
        self.assertEqual(
            pointwise_self_attribution_template_sha256_for_version(version),
            "a71239a72e820f2d89472bcfb8aefe1d49ca8b71c90eb82b5c5e6715579b230f",
        )


if __name__ == "__main__":
    unittest.main()
