from iar_phase0.openrouter import binary_probability_from_logprobs


def test_binary_parser_skips_provider_channel_tokens() -> None:
    response = {
        "choices": [{
            "logprobs": {
                "content": [
                    {
                        "token": "<|channel|>",
                        "top_logprobs": [{"token": "<|channel|>", "logprob": 0.0}],
                    },
                    {
                        "token": "1",
                        "top_logprobs": [
                            {"token": "1", "logprob": -0.1},
                            {"token": "0", "logprob": -2.1},
                        ],
                    },
                ]
            }
        }]
    }

    parsed = binary_probability_from_logprobs(response)

    assert parsed["ok"] is True
    assert parsed["first_token"] == "1"
    assert parsed["content_position"] == 1
    assert parsed["p1"] > 0.5


def test_binary_parser_rejects_http_200_payload_without_choices() -> None:
    parsed = binary_probability_from_logprobs(
        {"error": {"message": "transient upstream response"}}
    )

    assert parsed["ok"] is False
    assert parsed["error"] == "missing_or_invalid_choices"
    assert parsed["response_keys"] == ["error"]
