from __future__ import annotations

import asyncio
import json
import math
import tempfile
import unittest
from pathlib import Path

import httpx

from iar_phase0.prompts import POINTWISE_SELF_ATTRIBUTION_NO_SYSTEM_PROMPT_VERSION

from iar_grpo.reward import (
    ATTRIBUTION_PROMPT_VERSION,
    LiveOpenRouterReward,
    RewardSettings,
    _retry_delay_seconds,
    completion_to_text,
    reward_from_binary_logprobs,
    summarize_reward_log,
)
from iar_grpo.run_reward_probe import summarize_probe_groups


def fake_openrouter_response(_: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "deepseek/deepseek-v4-flash",
            "provider": "DeepSeek",
            "choices": [
                {
                    "message": {"content": "1"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "1",
                                "top_logprobs": [
                                    {"token": "1", "logprob": -0.2204174099},
                                    {"token": "0", "logprob": -1.6204174099},
                                ],
                            }
                        ]
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        },
    )


class RewardRuntimeTest(unittest.TestCase):
    def test_v2_reward_payload_contains_only_one_user_message(self) -> None:
        captured: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            response = fake_openrouter_response(request)
            body = json.loads(response.content)
            body["model"] = "openai/gpt-oss-120b"
            return httpx.Response(200, json=body)

        settings = RewardSettings(
            model_id="openai/gpt-oss-120b",
            target_model_name="OpenAI GPT-OSS 120B",
            max_reward_calls=1,
            max_http_requests=1,
            max_attempts_per_call=1,
            max_concurrency=1,
            timeout_seconds=5,
            temperature=0,
            max_tokens=256,
            top_logprobs=5,
            strict_logprobs=True,
            provider_routing={},
            request_overrides={"reasoning": {"effort": "low", "exclude": True}},
            attribution_prompt_version=POINTWISE_SELF_ATTRIBUTION_NO_SYSTEM_PROMPT_VERSION,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = LiveOpenRouterReward(
                settings=settings,
                log_path=Path(temp_dir) / "calls.jsonl",
                api_key="test-key",
                transport=httpx.MockTransport(handler),
            )
            asyncio.run(
                runtime.score_batch(
                    completions=["def f(): return 1"],
                    task_statement=["Write f."],
                    task_id=["1"],
                    dataset_index=[0],
                )
            )
            rows = [json.loads(line) for line in runtime.log_path.read_text().splitlines()]

        self.assertEqual([message["role"] for message in captured[0]["messages"]], ["user"])
        self.assertEqual(rows[0]["judge_message_roles"], ["user"])
        self.assertFalse(rows[0]["judge_system_prompt_present"])
        self.assertIsNone(rows[0]["judge_system_prompt_sha256"])

    def test_probe_summary_audits_reward_variation_labels_and_correctness(self) -> None:
        reward_rows = []
        completion_rows = []
        for task_id in ["1", "2"]:
            for rollout_index in range(4):
                reward_rows.append(
                    {
                        "status": "ok",
                        "task_id": task_id,
                        "rollout_index": rollout_index,
                        "reward": -float(rollout_index + 1),
                        "p_own": 0.1 * (rollout_index + 1),
                        "logp_0": -0.1 if rollout_index < 2 else -2.0,
                        "logp_1": -2.0 if rollout_index < 2 else -0.1,
                        "judge_token": "0" if rollout_index < 2 else "1",
                        "candidate": f"candidate-{task_id}-{rollout_index}",
                        "binary_label_mass": 1.0,
                        "raw_model": "b",
                        "raw_provider": "provider",
                    }
                )
                completion_rows.append(
                    {
                        "task_id": task_id,
                        "rollout_index": rollout_index,
                        "completion_tokens": 10,
                        "unit_tests": {"passed": rollout_index == 0},
                    }
                )

        summary = summarize_probe_groups(
            reward_rows,
            expected_rollouts=4,
            completion_rows=completion_rows,
        )
        self.assertEqual(summary["n_complete_groups"], 2)
        self.assertEqual(summary["n_groups_with_nonzero_reward_variance"], 2)
        self.assertEqual(summary["fooling_emitted"], 0.5)
        self.assertEqual(summary["fooling_argmax"], 0.5)
        self.assertEqual(summary["emitted_argmax_consistency"], 1.0)
        self.assertEqual(summary["correctness"]["groups_with_at_least_one_correct"], 2)
        self.assertEqual(summary["correctness"]["groups_with_mixed_correctness"], 2)

    def test_rate_limit_retry_honors_provider_delay_and_conservative_floor(self) -> None:
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        response = httpx.Response(429, headers={"Retry-After": "12"}, request=request)
        error = httpx.HTTPStatusError("rate limited", request=request, response=response)
        self.assertEqual(_retry_delay_seconds(error, attempt=1), 12.0)

        short_response = httpx.Response(429, headers={"Retry-After": "1"}, request=request)
        short_error = httpx.HTTPStatusError(
            "rate limited", request=request, response=short_response
        )
        self.assertEqual(_retry_delay_seconds(short_error, attempt=2), 10.0)

    def test_adopted_reward_is_log_probability_of_own_label(self) -> None:
        parsed = {"p1": 0.9, "logp_1": -0.1, "logp_0": -4.1}
        base = RewardSettings(
            model_id="b",
            target_model_name="B",
            max_reward_calls=1,
            max_http_requests=1,
            max_attempts_per_call=1,
            max_concurrency=1,
            timeout_seconds=5,
            temperature=0,
            max_tokens=1,
            top_logprobs=5,
            strict_logprobs=True,
            provider_routing={},
            request_overrides={},
        )
        log_reward, diagnostics = reward_from_binary_logprobs(parsed, base)
        self.assertAlmostEqual(log_reward, math.log(1 / (1 + math.exp(-4.0))))
        self.assertAlmostEqual(diagnostics["delta"], 4.0)
        self.assertAlmostEqual(diagnostics["log_p_own"], log_reward)

        historical = RewardSettings(
            **{
                **base.__dict__,
                "reward_value": "p_own",
            }
        )
        p_reward, _ = reward_from_binary_logprobs(parsed, historical)
        self.assertAlmostEqual(p_reward, 1 / (1 + math.exp(-4.0)))

        absolute = RewardSettings(
            **{
                **base.__dict__,
                "reward_value": "absolute_logp_1",
            }
        )
        absolute_reward, _ = reward_from_binary_logprobs(parsed, absolute)
        self.assertEqual(absolute_reward, -0.1)

    def test_runtime_accepts_a_domain_specific_single_user_prompt(self) -> None:
        captured: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return fake_openrouter_response(request)

        settings = RewardSettings(
            model_id="deepseek/deepseek-v4-flash",
            target_model_name="DeepSeek V4 Flash",
            max_reward_calls=1,
            max_http_requests=1,
            max_attempts_per_call=1,
            max_concurrency=1,
            timeout_seconds=5,
            temperature=0,
            max_tokens=1,
            top_logprobs=5,
            strict_logprobs=True,
            provider_routing={},
            request_overrides={},
            reward_value="absolute_logp_1",
            attribution_prompt_version="cqa-test-v1",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = LiveOpenRouterReward(
                settings=settings,
                log_path=Path(temp_dir) / "calls.jsonl",
                api_key="test-key",
                transport=httpx.MockTransport(handler),
                prompt_builder=lambda task, candidate: f"TASK={task}\nCANDIDATE={candidate}",
                prompt_template_hash="fixed-hash",
                system_prompt_builder=lambda _name: None,
                candidate_normalizer=str.strip,
            )
            rewards = asyncio.run(
                runtime.score_batch(
                    completions=["  response  "],
                    task_statement=["question"],
                    task_id=["hex-id"],
                    dataset_index=[0],
                )
            )
            rows = [json.loads(line) for line in runtime.log_path.read_text().splitlines()]
            summary = summarize_reward_log(runtime.log_path)

        self.assertEqual(rewards, [-0.2204174099])
        self.assertEqual(captured[0]["messages"], [{"role": "user", "content": "TASK=question\nCANDIDATE=response"}])
        self.assertEqual(rows[0]["judge_prompt_template_sha256"], "fixed-hash")
        self.assertFalse(rows[0]["judge_system_prompt_present"])
        self.assertEqual(summary["n_ok"], 1)

    def test_conversational_completion_is_normalized(self) -> None:
        payload = [[{"role": "assistant", "content": "```python\ndef f():\n    return 1\n```"}]]
        self.assertIn("def f", completion_to_text(payload[0]))

    def test_scores_and_logs_one_request_per_rollout(self) -> None:
        task_ids = [str(task_id) for task_id in range(601, 611) for _ in range(4)]
        dataset_indices = [index for index in range(10) for _ in range(4)]
        settings = RewardSettings(
            model_id="deepseek/deepseek-v4-flash",
            target_model_name="DeepSeek V4 Flash",
            max_reward_calls=40,
            max_http_requests=120,
            max_attempts_per_call=3,
            max_concurrency=4,
            timeout_seconds=5,
            temperature=0,
            max_tokens=1,
            top_logprobs=5,
            strict_logprobs=True,
            provider_routing={"require_parameters": True, "allow_fallbacks": False},
            request_overrides={"reasoning": {"enabled": False}},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "reward_calls.jsonl"
            runtime = LiveOpenRouterReward(
                settings=settings,
                log_path=log_path,
                api_key="test-key",
                transport=httpx.MockTransport(fake_openrouter_response),
            )
            rewards = asyncio.run(
                runtime.score_batch(
                    completions=["def f(): return 1"] * 40,
                    task_statement=["Write f."] * 40,
                    task_id=task_ids,
                    dataset_index=dataset_indices,
                )
            )
            expected = math.log(1.0 / (1.0 + math.exp(-1.4)))
            self.assertTrue(all(abs(value - expected) < 1e-8 for value in rewards))
            runtime.assert_complete()
            rows = [json.loads(line) for line in log_path.read_text().splitlines()]
            self.assertEqual(len(rows), 40)
            self.assertEqual({row["rollout_index"] for row in rows}, {0, 1, 2, 3})
            self.assertEqual({row["status"] for row in rows}, {"ok"})
            self.assertEqual({row["logprob_first_token"] for row in rows}, {"1"})
            self.assertEqual({row["raw_provider"] for row in rows}, {"DeepSeek"})
            self.assertEqual(
                {row["judge_prompt_template_version"] for row in rows},
                {ATTRIBUTION_PROMPT_VERSION},
            )
            self.assertEqual(len({row["judge_prompt_template_sha256"] for row in rows}), 1)
            self.assertEqual(len({row["judge_system_prompt_sha256"] for row in rows}), 1)
            self.assertTrue(all(row["binary_label_mass"] > 0.999 for row in rows))
            summary = summarize_reward_log(log_path)
            self.assertEqual(summary["n_ok"], 40)
            self.assertEqual(len(summary["by_task"]), 10)
            self.assertEqual(summary["usage"]["total_tokens"], 440.0)
            self.assertEqual(summary["raw_providers"], {"DeepSeek": 40})
            self.assertEqual(
                summary["judge_prompt_template_versions"],
                {ATTRIBUTION_PROMPT_VERSION: 40},
            )
            self.assertEqual(runtime.http_requests, 40)

    def test_reward_budget_is_hard(self) -> None:
        settings = RewardSettings(
            model_id="b",
            target_model_name="B",
            max_reward_calls=1,
            max_http_requests=1,
            max_attempts_per_call=1,
            max_concurrency=1,
            timeout_seconds=5,
            temperature=0,
            max_tokens=1,
            top_logprobs=5,
            strict_logprobs=True,
            provider_routing={},
            request_overrides={},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = LiveOpenRouterReward(
                settings=settings,
                log_path=Path(temp_dir) / "calls.jsonl",
                api_key="test-key",
                transport=httpx.MockTransport(fake_openrouter_response),
            )
            with self.assertRaises(RuntimeError):
                asyncio.run(
                    runtime.score_batch(
                        completions=["a", "b"],
                        task_statement=["t", "t"],
                        task_id=["1", "1"],
                        dataset_index=[0, 0],
                    )
                )

    def test_runtime_supports_multiple_async_batches(self) -> None:
        settings = RewardSettings(
            model_id="deepseek/deepseek-v4-flash",
            target_model_name="DeepSeek V4 Flash",
            max_reward_calls=2,
            max_http_requests=6,
            max_attempts_per_call=3,
            max_concurrency=1,
            timeout_seconds=5,
            temperature=0,
            max_tokens=1,
            top_logprobs=5,
            strict_logprobs=True,
            provider_routing={},
            request_overrides={},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = LiveOpenRouterReward(
                settings=settings,
                log_path=Path(temp_dir) / "calls.jsonl",
                api_key="test-key",
                transport=httpx.MockTransport(fake_openrouter_response),
            )
            for task_id in ["1", "2"]:
                rewards = asyncio.run(
                    runtime.score_batch(
                        completions=["def f(): return 1"],
                        task_statement=["Write f."],
                        task_id=[task_id],
                        dataset_index=[int(task_id)],
                    )
                )
                self.assertEqual(len(rewards), 1)
            runtime.assert_complete()
            self.assertEqual(runtime.http_requests, 2)

    def test_resume_preserves_explicit_rollout_indices_and_unique_ids(self) -> None:
        settings = RewardSettings(
            model_id="deepseek/deepseek-v4-flash",
            target_model_name="DeepSeek V4 Flash",
            max_reward_calls=2,
            max_http_requests=2,
            max_attempts_per_call=1,
            max_concurrency=1,
            timeout_seconds=5,
            temperature=0,
            max_tokens=1,
            top_logprobs=5,
            strict_logprobs=True,
            provider_routing={},
            request_overrides={},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "calls.jsonl"
            runtime = LiveOpenRouterReward(
                settings=settings,
                log_path=log_path,
                api_key="test-key",
                transport=httpx.MockTransport(fake_openrouter_response),
                call_id_offset=17,
                http_request_id_offset=23,
            )
            asyncio.run(
                runtime.score_batch(
                    completions=["def f(): return 1", "def f(): return 2"],
                    task_statement=["Write f.", "Write f."],
                    task_id=["1", "1"],
                    dataset_index=[0, 0],
                    rollout_index=[1, 3],
                )
            )
            runtime.assert_complete()
            rows = [json.loads(line) for line in log_path.read_text().splitlines()]

        self.assertEqual([row["rollout_index"] for row in rows], [1, 3])
        self.assertEqual([row["call_id"] for row in rows], [18, 19])
        self.assertEqual([row["http_request_id"] for row in rows], [24, 25])


if __name__ == "__main__":
    unittest.main()
