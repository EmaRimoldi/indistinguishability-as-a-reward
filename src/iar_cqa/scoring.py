from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable


ANSWER_RE = re.compile(
    r"(?im)^[ \t]*(?:\*\*|__)?Answer\s*:\s*([A-E])"
    r"(?:\*\*|__)?[ \t]*$"
)


def parse_answer(text: str) -> str | None:
    matches = ANSWER_RE.findall(text or "")
    return matches[-1].upper() if matches else None


def score_response(text: str, gold: str) -> dict[str, Any]:
    prediction = parse_answer(text)
    return {
        "prediction": prediction,
        "gold": gold,
        "valid": prediction is not None,
        "correct": prediction == gold,
    }


def response_features(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    words = re.findall(r"\b\w+\b", stripped)
    lines = stripped.splitlines()
    sentence_marks = re.findall(r"[.!?](?:\s|$)", stripped)
    bullets = sum(bool(re.match(r"^\s*(?:[-*]|\d+[.)])\s+", line)) for line in lines)
    mentioned_options = sorted(set(re.findall(r"(?i)\b(?:option|choice)\s+([A-E])\b", stripped)))
    repeated_choice_text = len(re.findall(r"(?im)^\s*[A-E][.)]\s+", stripped))
    return {
        "chars": len(stripped),
        "words": len(words),
        "lines": len(lines),
        "sentences": max(len(sentence_marks), 1 if stripped else 0),
        "bullets": bullets,
        "mentioned_option_letters": mentioned_options,
        "n_mentioned_option_letters": len(mentioned_options),
        "repeated_choice_lines": repeated_choice_text,
        "has_answer_line": parse_answer(stripped) is not None,
        "commas": stripped.count(","),
        "semicolons": stripped.count(";"),
        "parentheses": stripped.count("(") + stripped.count(")"),
    }


def accuracy(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    valid = sum(bool(row["score"]["valid"]) for row in rows)
    correct = sum(bool(row["score"]["correct"]) for row in rows)
    return {
        "n": len(rows),
        "valid": valid,
        "valid_rate": valid / len(rows) if rows else None,
        "correct": correct,
        "accuracy": correct / len(rows) if rows else None,
        "prediction_counts": dict(
            Counter(
                str(row["score"]["prediction"])
                if row["score"]["prediction"] is not None
                else "INVALID"
                for row in rows
            )
        ),
    }
