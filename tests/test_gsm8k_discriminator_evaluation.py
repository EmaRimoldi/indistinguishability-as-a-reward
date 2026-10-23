from iar_gsm8k.evaluate_discriminator import summarize


def test_summarize_discriminator_validation() -> None:
    rows = [
        {
            "score": {"valid": True, "correct": True},
            "terminated_normally": True,
            "completion_tokens": 10,
            "usage": {"cost": 0.01},
            "raw_model": "deepseek/deepseek-v4-flash",
            "raw_provider": "DigitalOcean",
            "message_roles": ["user"],
            "system_prompt_present": False,
            "prompt_version": "gsm8k-task-plain-v1-no-system",
        },
        {
            "score": {"valid": False, "correct": False},
            "terminated_normally": False,
            "completion_tokens": 20,
            "usage": {"cost": 0.02},
            "raw_model": "deepseek/deepseek-v4-flash",
            "raw_provider": "DigitalOcean",
            "message_roles": ["user"],
            "system_prompt_present": False,
            "prompt_version": "gsm8k-task-plain-v1-no-system",
        },
    ]

    result = summarize(rows)

    assert result["accuracy"] == 0.5
    assert result["valid_rate"] == 0.5
    assert result["termination_rate"] == 0.5
    assert result["mean_completion_tokens"] == 15
    assert result["openrouter_cost_usd"] == 0.03
    assert result["system_prompt_present"] == [False]
