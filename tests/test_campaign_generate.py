from iar_campaign.generate import coding_imitation, coding_plain, prompts_for_domain


def test_kodcode_prompts_never_include_gold_fields():
    row = {
        "id": "x",
        "question": "Write add(a, b).",
        "reference_solution": "SECRET_SOLUTION",
        "hidden_tests": "SECRET_TESTS",
    }
    prompts = prompts_for_domain("kodcode", [row], tables_json=None)
    assert set(prompts) == {"a0", "prompt_imitation"}
    for value in prompts.values():
        assert "Write add(a, b)." in value[0]
        assert "SECRET_SOLUTION" not in value[0]
        assert "SECRET_TESTS" not in value[0]
    assert "DeepSeek V4 Flash" not in coding_plain(row["question"])
    assert "DeepSeek V4 Flash" in coding_imitation(row["question"])
