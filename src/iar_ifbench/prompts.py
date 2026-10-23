from __future__ import annotations

import hashlib

TASK_PROMPT_VERSION = "rlvr-ifeval-verbatim-user-v1-no-system"
ATTRIBUTION_PROMPT_VERSION = "rlvr-ifeval-pointwise-self-attribution-v1-no-system"
TARGET_MODEL_NAME = "DeepSeek V4 Flash"

_TASK_TEMPLATE = "{prompt}"
_ATTRIBUTION_TEMPLATE = """You are {target_model_name}. You are given an instruction-following task and one candidate response.

Task:
{task_statement}

Candidate response:
{candidate}

Would you plausibly have produced this candidate response yourself for this task?
Return exactly one token: 1 for yes or 0 for no."""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def task_prompt(prompt: str) -> str:
    """Return the frozen RLVR user message without adding hints or gold data."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("RLVR-IFeval task prompt must be a non-empty string")
    if prompt != prompt.strip():
        raise ValueError(
            "Frozen RLVR-IFeval prompts must already be whitespace-normalized"
        )
    return _TASK_TEMPLATE.format(prompt=prompt)


def attribution_prompt(task_statement: str, candidate: str) -> str:
    if not task_statement.strip():
        raise ValueError("Attribution task statement may not be empty")
    return _ATTRIBUTION_TEMPLATE.format(
        target_model_name=TARGET_MODEL_NAME,
        task_statement=task_statement,
        candidate=candidate.strip(),
    )


def template_hashes() -> dict[str, str]:
    return {
        TASK_PROMPT_VERSION: sha256_text(_TASK_TEMPLATE),
        ATTRIBUTION_PROMPT_VERSION: sha256_text(_ATTRIBUTION_TEMPLATE),
    }
