from __future__ import annotations

import hashlib

from .data import MbppExample

TARGET_MODEL_NAME = "DeepSeek V4 Flash"
POINTWISE_SELF_ATTRIBUTION_PROMPT_VERSION = "phase0-pointwise-self-attribution-v1"
POINTWISE_SELF_ATTRIBUTION_NO_SYSTEM_PROMPT_VERSION = (
    "phase0-pointwise-self-attribution-v2-no-system"
)


def build_fewshot_block(examples: list[MbppExample]) -> str:
    blocks: list[str] = []
    for idx, example in enumerate(examples, start=1):
        solution = (example.code or "").strip()
        blocks.append(f"{mbpp_google_prompt(example)}\n{solution}\n[DONE]")
    return "\n---\n".join(blocks)


def generator_prompt(example: MbppExample, variant: str, fewshot: list[MbppExample]) -> str:
    if variant == "plain":
        return mbpp_google_prompt(example)
    if variant == "imitation":
        return target_imitation_prompt(example, target_model=TARGET_MODEL_NAME)
    if variant == "fewshot_imitation":
        return (
            "The following are examples of Python programming tasks and their solutions.\n\n"
            f"{build_fewshot_block(fewshot)}\n\n"
            f"{target_imitation_prompt(example, target_model=TARGET_MODEL_NAME)}"
        )
    raise ValueError(f"Unknown generator variant: {variant}")


def discriminator_own_answer_prompt(example: MbppExample) -> str:
    return mbpp_google_prompt(example)


def discriminator_judge_prompt(example: MbppExample, candidate_code: str) -> str:
    return (
        "You are judging whether a candidate Python solution could plausibly be a solution "
        "that you yourself would produce.\n\n"
        f"{mbpp_google_problem_statement(example)}\n\n"
        f"Candidate solution:\n{candidate_code.strip()}\n\n"
        "Answer with exactly one token:\n"
        "1 = this could plausibly be my own solution\n"
        "0 = this is unlikely to be my own solution\n\n"
        "Use 1 only if the solution is plausibly correct, executable, and stylistically compatible "
        "with your own likely answer.\n"
        "Use 0 if the solution is incorrect, incomplete, superficial, unnecessarily verbose, "
        "oddly formatted, or not compatible with your likely answer.\n\n"
        "Answer:"
    )


def pointwise_self_attribution_system_prompt(target_model: str) -> str:
    return (
        f"You are {target_model}. You are participating in a blinded self-attribution test. "
        "You must decide whether a single Python solution was written by you. "
        "Do not explain your reasoning. Return exactly one token: 1 or 0."
    )


def pointwise_self_attribution_prompt(example: MbppExample, candidate_code: str) -> str:
    return pointwise_self_attribution_prompt_from_strings(
        mbpp_google_problem_statement(example),
        candidate_code,
    )


def pointwise_self_attribution_prompt_from_strings(
    task_statement: str,
    candidate_code: str,
) -> str:
    """Return the canonical pointwise attribution user prompt.

    Phase 0 and the live GRPO reward use this exact function so their judge
    prompts cannot silently drift apart.
    """
    return (
        "You will be shown a Python programming task and one anonymized candidate solution.\n\n"
        "The solution may have been written by you, or it may have been written by another "
        "model solving the same task.\n\n"
        "Your objective is not to judge whether the solution is better in general. "
        "Your objective is to decide whether this solution was written by you, based on "
        "your own coding choices, implementation strategy, Python idioms, formatting, "
        "naming conventions, edge-case handling, and level of detail.\n\n"
        f"Task:\n{task_statement.strip()}\n\n"
        f"Candidate solution:\n{candidate_code.strip()}\n\n"
        "Was this solution written by you?\n\n"
        "Return exactly one token:\n"
        "1 = yes, this was written by me\n"
        "0 = no, this was not written by me\n\n"
        "Answer:"
    )


def pointwise_self_attribution_no_system_prompt(
    example: MbppExample,
    candidate_code: str,
) -> str:
    return pointwise_self_attribution_no_system_prompt_from_strings(
        mbpp_google_problem_statement(example),
        candidate_code,
    )


def pointwise_self_attribution_no_system_prompt_from_strings(
    task_statement: str,
    candidate_code: str,
) -> str:
    """Return the frozen v2 pointwise prompt sent as one user message.

    This intentionally preserves the historical self-attribution question while
    removing the separate system message.  Do not add task-quality or style
    penalties to this template: those are external audits, not the reward.
    """
    return (
        "You will be shown a Python programming task and one anonymized candidate solution.\n\n"
        "The solution may have been written by you, or it may have been written by another\n"
        "model solving the same task.\n\n"
        "Your objective is not to judge whether the solution is better in general.\n"
        "Your objective is to decide whether this solution was written by you, based on\n"
        "your own coding choices, implementation strategy, Python idioms, formatting,\n"
        "naming conventions, edge-case handling, and level of detail.\n\n"
        f"Task:\n{task_statement.strip()}\n\n"
        f"Candidate solution:\n{candidate_code.strip()}\n\n"
        "Was this solution written by you?\n\n"
        "Return exactly one token:\n"
        "1 = yes, this was written by me\n"
        "0 = no, this was not written by me\n\n"
        "Answer:"
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def pointwise_self_attribution_template_sha256() -> str:
    """Fingerprint the invariant Phase-0/GRPO attribution template."""
    return sha256_text(
        pointwise_self_attribution_prompt_from_strings("<TASK>", "<CANDIDATE>")
    )


def pointwise_self_attribution_no_system_template_sha256() -> str:
    return sha256_text(
        pointwise_self_attribution_no_system_prompt_from_strings(
            "<TASK>", "<CANDIDATE>"
        )
    )


def pointwise_self_attribution_prompt_for_version(
    task_statement: str,
    candidate_code: str,
    version: str,
) -> str:
    if version == POINTWISE_SELF_ATTRIBUTION_PROMPT_VERSION:
        return pointwise_self_attribution_prompt_from_strings(
            task_statement, candidate_code
        )
    if version == POINTWISE_SELF_ATTRIBUTION_NO_SYSTEM_PROMPT_VERSION:
        return pointwise_self_attribution_no_system_prompt_from_strings(
            task_statement, candidate_code
        )
    raise ValueError(f"Unsupported pointwise attribution prompt version: {version!r}")


def pointwise_self_attribution_template_sha256_for_version(version: str) -> str:
    if version == POINTWISE_SELF_ATTRIBUTION_PROMPT_VERSION:
        return pointwise_self_attribution_template_sha256()
    if version == POINTWISE_SELF_ATTRIBUTION_NO_SYSTEM_PROMPT_VERSION:
        return pointwise_self_attribution_no_system_template_sha256()
    raise ValueError(f"Unsupported pointwise attribution prompt version: {version!r}")


def pointwise_self_attribution_system_prompt_for_version(
    version: str,
    target_model: str,
) -> str | None:
    if version == POINTWISE_SELF_ATTRIBUTION_PROMPT_VERSION:
        return pointwise_self_attribution_system_prompt(target_model)
    if version == POINTWISE_SELF_ATTRIBUTION_NO_SYSTEM_PROMPT_VERSION:
        return None
    raise ValueError(f"Unsupported pointwise attribution prompt version: {version!r}")


def pairwise_code_attribution_system_prompt() -> str:
    return (
        "You are a blinded pairwise code-attribution evaluator. Apply only the supplied "
        "evaluation rubric. You must choose one solution, even when the evidence is weak "
        "or tied. Do not provide explanations or chain-of-thought. Return exactly one "
        "valid JSON object and no additional text."
    )


def pairwise_code_attribution_prompt(
    example: MbppExample,
    *,
    target_model: str,
    solution_1: str,
    solution_2: str,
) -> str:
    return (
        f"Target generator:\n{target_model}\n\n"
        "Select the solution whose observable coding behavior is more consistent with the "
        "target generator.\n\n"
        "The task is provided only to interpret the code in context, not to reward the "
        "solution that is better in general.\n\n"
        "Consider implementation strategy, decomposition choices, use of Python idioms, "
        "control flow, edge-case handling, naming conventions, comments or docstrings, "
        "concision, formatting, and level of detail.\n\n"
        "Ignore solution position, explicit mentions of model names, comments about the "
        "generation process, and assumptions about how the solutions were generated. "
        "Do not choose solely because one solution appears more correct or better in general. "
        "Do not assume that exactly one solution was produced by the target generator. You "
        "must nevertheless choose the solution that is more consistent with it.\n\n"
        f"Task:\n{mbpp_google_problem_statement(example)}\n\n"
        f"Solution 1:\n{solution_1.strip()}\n\n"
        f"Solution 2:\n{solution_2.strip()}\n\n"
        "Return exactly one of:\n"
        "{\"choice\": 1}\n"
        "{\"choice\": 2}"
    )


def mbpp_google_problem_statement(example: MbppExample) -> str:
    tests = "\n".join(example.tests)
    return (
        "You are an expert Python programmer, and here is your task: "
        f"{example.prompt.strip()} "
        "Your code should pass these tests:\n\n"
        f"{tests}"
    )


def mbpp_google_prompt(example: MbppExample) -> str:
    return f"{mbpp_google_problem_statement(example)}\n[BEGIN]\n"


def target_imitation_prompt(example: MbppExample, target_model: str) -> str:
    return (
        f"For this response, act as {target_model}. Do not merely imitate its surface style: "
        f"assume the same level of programming competence, problem-solving ability, and code quality as {target_model}.\n"
        f"Your goal is to produce a Python solution that would be indistinguishable from an answer written by {target_model} "
        "for the same task.\n"
        "Do not mention the model name in the answer. Output only executable Python code after [BEGIN].\n\n"
        f"{mbpp_google_prompt(example)}"
    )
