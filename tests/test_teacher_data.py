import pytest

from iar_distillation.teacher_data import (
    SCORER_VERSIONS,
    build_summary,
    one_user_prompt,
    rescore_cached_responses,
)


def test_one_user_prompt_is_strict() -> None:
    row = {"prompt": [{"role": "user", "content": "Question"}]}
    assert one_user_prompt(row) == "Question"


def test_summary_reports_accuracy_and_provider() -> None:
    rows = [
        {
            "score": {"valid": True, "correct": True},
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.1},
            "raw_model": "deepseek/deepseek-v4-flash",
            "raw_provider": "DigitalOcean",
        },
        {
            "score": {"valid": True, "correct": False},
            "finish_reason": "length",
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12, "cost": 0.05},
            "raw_model": "deepseek/deepseek-v4-flash",
            "raw_provider": "DigitalOcean",
        },
    ]
    summary = build_summary("gsm8k", rows)
    assert summary["n"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["valid_rate"] == 1.0
    assert summary["termination_rate"] == 0.5
    assert summary["openrouter_cost"] == pytest.approx(0.15)
    assert summary["providers"] == ["DigitalOcean"]


def test_rescore_cached_cqa_response_accepts_markdown_emphasis() -> None:
    rows = [
        {
            "id": "example-1",
            "raw_response": "Reasoning.\n**Answer: D**",
            "response": "Reasoning.\n**Answer: D**",
            "response_sha256": "stale",
            "score": {"valid": False, "correct": False},
        }
    ]
    sources = [{"id": "example-1", "gold": {"answer_key": "D"}}]
    rescored = rescore_cached_responses("commonsenseqa", rows, sources, timeout=5)
    assert rescored[0]["score"] == {
        "prediction": "D",
        "gold": "D",
        "valid": True,
        "correct": True,
    }
    assert rescored[0]["scorer_version"] == SCORER_VERSIONS["commonsenseqa"]
