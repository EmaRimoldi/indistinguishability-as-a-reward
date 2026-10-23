from __future__ import annotations

import hashlib
from collections.abc import Sequence

TASK_PROMPT_VERSION = "commonsenseqa-explain-answer-v1"
IMITATION_PROMPT_VERSION = "commonsenseqa-deepseek-imitation-v1"
ATTRIBUTION_PROMPT_VERSION = "commonsenseqa-pointwise-self-attribution-v1-no-system"
BINARY_CONTRACT_ATTRIBUTION_PROMPT_VERSION = (
    "commonsenseqa-pointwise-self-attribution-v2-binary-contract-no-system"
)
TARGET_MODEL_NAME = "DeepSeek V4 Flash"
LABELS = ("A", "B", "C", "D", "E")


def task_prompt(question: str, choices: Sequence[str]) -> str:
    if len(choices) != 5:
        raise ValueError(f"CommonsenseQA requires five choices, received {len(choices)}")
    choice_block = "\n".join(f"{label}. {choice}" for label, choice in zip(LABELS, choices))
    return (
        f"Question:\n{question.strip()}\n\n"
        f"Choices:\n{choice_block}\n\n"
        "Explain why one choice is the best answer.\n\n"
        "End with a separate line in the format:\n"
        "Answer: X\n\n"
        "where X is one of A, B, C, D, or E."
    )


def imitation_prompt(question: str, choices: Sequence[str]) -> str:
    return (
        f"For this response, answer as {TARGET_MODEL_NAME} would.\n\n"
        "Match its level of commonsense reasoning and answer quality, not merely its\n"
        f"surface wording. Produce a response that would be indistinguishable from one\n"
        f"written by {TARGET_MODEL_NAME}.\n\n"
        "Do not mention the model name in your response.\n\n"
        f"{task_prompt(question, choices)}"
    )


def attribution_prompt(task: str, candidate_response: str) -> str:
    """Domain adaptation of the frozen MBPP v2 pointwise prompt.

    Paragraph order, binary labels, and absence of a system message are kept
    isomorphic to the Qwen/MBPP experiment. Only domain-specific nouns and the
    observable response characteristics are changed.
    """
    return (
        "You will be shown a commonsense multiple-choice question and one anonymized candidate response.\n\n"
        "The response may have been written by you, or it may have been written by another\n"
        "model answering the same question.\n\n"
        "Your objective is not to judge whether the response is better in general.\n"
        "Your objective is to decide whether this response was written by you, based on\n"
        "your own reasoning choices, reasoning strategy, use of commonsense knowledge,\n"
        "explanation structure, treatment of alternatives, and level of detail.\n\n"
        f"Task:\n{task.strip()}\n\n"
        f"Candidate response:\n{candidate_response.strip()}\n\n"
        "Was this response written by you?\n\n"
        "Return exactly one token:\n"
        "1 = yes, this was written by me\n"
        "0 = no, this was not written by me\n\n"
        "Answer:"
    )


def binary_contract_attribution_prompt(task: str, candidate_response: str) -> str:
    """Confirmatory CQA prompt with an explicit one-character output contract."""
    return (
        "You will be shown a commonsense multiple-choice question and one anonymized candidate response.\n\n"
        "The response may have been written by you, or it may have been written by another\n"
        "model answering the same question.\n\n"
        "Your objective is not to judge whether the response is better in general.\n"
        "Your objective is to decide whether this response was written by you, based on\n"
        "your own reasoning choices, reasoning strategy, use of commonsense knowledge,\n"
        "explanation structure, treatment of alternatives, and level of detail.\n\n"
        f"Task:\n{task.strip()}\n\n"
        f"Candidate response:\n{candidate_response.strip()}\n\n"
        "Was this response written by you?\n\n"
        "Your entire response must be exactly one ASCII character: 0 or 1.\n"
        "Do not output a word, prefix, whitespace, punctuation, or explanation.\n\n"
        "Return exactly one token:\n"
        "1 = yes, this was written by me\n"
        "0 = no, this was not written by me\n\n"
        "Output:"
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def template_hashes() -> dict[str, str]:
    choices = ["<CHOICE_A>", "<CHOICE_B>", "<CHOICE_C>", "<CHOICE_D>", "<CHOICE_E>"]
    return {
        TASK_PROMPT_VERSION: sha256_text(task_prompt("<QUESTION>", choices)),
        IMITATION_PROMPT_VERSION: sha256_text(imitation_prompt("<QUESTION>", choices)),
        ATTRIBUTION_PROMPT_VERSION: sha256_text(
            attribution_prompt("<TASK>", "<CANDIDATE_RESPONSE>")
        ),
        BINARY_CONTRACT_ATTRIBUTION_PROMPT_VERSION: sha256_text(
            binary_contract_attribution_prompt("<TASK>", "<CANDIDATE_RESPONSE>")
        ),
    }


def protocol_manifest() -> dict[str, object]:
    return {
        "task_prompt_version": TASK_PROMPT_VERSION,
        "imitation_prompt_version": IMITATION_PROMPT_VERSION,
        "attribution_prompt_version": ATTRIBUTION_PROMPT_VERSION,
        "template_sha256": template_hashes(),
        "message_roles": ["user"],
        "system_prompt_present": False,
        "target_model_name": TARGET_MODEL_NAME,
    }
