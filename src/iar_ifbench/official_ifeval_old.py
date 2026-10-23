"""Small, pinned adapter for Open Instruct's ``IFEvalVerifierOld``.

The verifier class is defined in ``open_instruct/ground_truth_utils.py`` and
dispatches to ``IF_FUNCTIONS_MAP`` in ``open_instruct/if_functions.py``.  The
evaluation entrypoint validates both files byte-for-byte against revision
``ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5`` before loading the function map.
This module reproduces only the class's local dispatch semantics so importing
the full training stack is unnecessary.

Upstream: https://github.com/allenai/open-instruct
License: Apache-2.0 (the exact upstream LICENSE is validated at runtime).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from typing import Any

OPEN_INSTRUCT_REPOSITORY = "https://github.com/allenai/open-instruct.git"
OPEN_INSTRUCT_REVISION = "ebd0c8e0c1d778085a4f26aacc2fd4d96f3515f5"
GROUND_TRUTH_UTILS_SHA256 = (
    "9c5a5fe88e44bc8f6bc11da622e28947da8ef75d91d16511ddc30d45b4943dd4"
)
IF_FUNCTIONS_SHA256 = "afea345e3bc100e6d5588ca14e7269141e8810e1489e60ad64a59cd57360201c"
OPEN_INSTRUCT_LICENSE_SHA256 = (
    "b998a796c55d006a974a48c73aa32674e6901e8d49aa98d74722e0c1063b40b6"
)
OPEN_INSTRUCT_LICENSE = "Apache-2.0"

EXPECTED_FUNCTION_NAMES = frozenset(
    {
        "validate_choice",
        "validate_end",
        "validate_forbidden_words",
        "validate_frequency_capital_words",
        "validate_highlighted_sections",
        "validate_json_format",
        "validate_lowercase",
        "validate_no_commas",
        "validate_paragraphs",
        "validate_placeholders",
        "validate_quotation",
        "validate_repeat_prompt",
        "validate_response_language",
        "validate_sections",
        "validate_title",
        "validate_two_responses",
        "validate_uppercase",
        "validate_word_constraint",
        "verify_bullet_points",
        "verify_keyword_frequency",
        "verify_keywords",
        "verify_letter_frequency",
        "verify_paragraph_count",
        "verify_postscript",
        "verify_sentence_constraint",
    }
)


def remove_thinking_section(prediction: str) -> str:
    """Match the helper called by the pinned legacy verifier."""
    prediction = prediction.replace("<|assistant|>", "").strip()
    prediction = prediction.split("</think>")[-1]
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")
    return prediction.strip()


def score_with_ifeval_verifier_old(
    prediction: str,
    label: str | Mapping[str, Any],
    function_map: Mapping[str, Callable[..., Any]],
) -> float:
    """Return the score produced by the pinned ``IFEvalVerifierOld``.

    The upstream implementation mutates dictionary labels by popping
    ``func_name``.  Copying here preserves identical scoring without mutating
    the frozen validation record.
    """
    constraint: Any = label
    answer = remove_thinking_section(prediction)
    if isinstance(constraint, str):
        constraint = json.loads(constraint)
    if not isinstance(constraint, Mapping):
        raise TypeError("IFEvalVerifierOld label must be a JSON object")
    constraint = dict(constraint)
    if "func_name" not in constraint:
        return 0.0
    func_name = constraint.pop("func_name")
    if not isinstance(func_name, str) or func_name not in function_map:
        raise KeyError(f"Unknown IFEvalVerifierOld function: {func_name!r}")
    function = function_map[func_name]
    non_none_args = {
        key: value for key, value in constraint.items() if value is not None
    }
    result = function(answer) if not constraint else function(answer, **non_none_args)
    score = float(result)
    if not math.isfinite(score) or score not in (0.0, 1.0):
        raise ValueError(f"Legacy IF verifier returned non-binary score {score!r}")
    return score
