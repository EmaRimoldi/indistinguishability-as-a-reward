from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable


ANSWER_MARKER_RE = re.compile(
    r"(?im)(?:^|(?<=[.!?])\s+)(?:#{1,6}\s*)?(?:\*\*|__)?"
    r"Answer\s*:\s*([^\r\n]*?)(?:\*\*|__)?(?=\s*$)"
)
GOLD_LINE_RE = re.compile(r"(?m)^####\s*(.*?)\s*$")
NUMERIC_RE = re.compile(
    r"^[+-]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?$"
)


def normalize_number(value: str) -> str | None:
    """Return a canonical decimal string, without guessing missing content."""
    candidate = (value or "").strip()
    if candidate.endswith(".") and candidate.count(".") >= 1:
        candidate = candidate[:-1].rstrip()
    if candidate.startswith("(") and candidate.endswith(")"):
        candidate = "-" + candidate[1:-1].strip()
    candidate = re.sub(r"^([+-]?)\s*[\$£€¥]\s*", r"\1", candidate)
    candidate = re.sub(r"\s*[\$£€¥]$", "", candidate)
    candidate = candidate.strip()
    candidate = re.sub(r"\s+", "", candidate)
    if not NUMERIC_RE.fullmatch(candidate):
        return None
    try:
        number = Decimal(candidate.replace(",", ""))
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    if number == 0:
        return "0"
    canonical = format(number.normalize(), "f")
    return canonical.rstrip("0").rstrip(".") if "." in canonical else canonical


def parse_answer(text: str) -> str | None:
    """Parse the final explicit ``Answer:`` marker without guessing a number."""
    matches = ANSWER_MARKER_RE.findall(text or "")
    return normalize_number(matches[-1]) if matches else None


def parse_reference_answer(reference_solution: str) -> str:
    matches = GOLD_LINE_RE.findall(reference_solution or "")
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one GSM8K #### answer, found {len(matches)}")
    value = normalize_number(matches[0])
    if value is None:
        raise ValueError(f"Invalid GSM8K reference answer: {matches[0]!r}")
    return value


def score_response(text: str, gold: str) -> dict[str, Any]:
    prediction = parse_answer(text)
    normalized_gold = normalize_number(gold)
    if normalized_gold is None:
        raise ValueError(f"Invalid gold answer: {gold!r}")
    return {
        "prediction": prediction,
        "gold": normalized_gold,
        "valid": prediction is not None,
        "correct": prediction == normalized_gold,
    }


def response_features(text: str, question: str = "") -> dict[str, Any]:
    stripped = (text or "").strip()
    lines = stripped.splitlines()
    words = re.findall(r"\b\w+\b", stripped)
    alphabetic_words = re.findall(r"[A-Za-z]+", stripped)
    sentence_marks = re.findall(r"[.!?](?:\s|$)", stripped)
    equation_lines = sum("=" in line for line in lines)
    numbered_steps = sum(bool(re.match(r"^\s*(?:step\s+)?\d+[.):]\s+", line, re.I)) for line in lines)
    conclusion_markers = len(re.findall(r"(?i)\b(?:therefore|thus|hence|so)\b", stripped))
    normalized_question = " ".join((question or "").lower().split())
    normalized_response = " ".join(stripped.lower().split())
    answer_matches = ANSWER_MARKER_RE.findall(stripped)
    return {
        "chars": len(stripped),
        "words": len(words),
        "lexical_diversity": len({word.lower() for word in alphabetic_words}) / len(alphabetic_words)
        if alphabetic_words else 0.0,
        "mean_word_length": sum(len(word) for word in alphabetic_words) / len(alphabetic_words)
        if alphabetic_words else 0.0,
        "lines": len(lines),
        "sentences": max(len(sentence_marks), 1 if stripped else 0),
        "numbered_steps": numbered_steps,
        "equation_lines": equation_lines,
        "equals_signs": stripped.count("="),
        "arithmetic_operators": len(re.findall(r"(?<!\w)[+×÷*/-](?!\w)", stripped)),
        "commas": stripped.count(","),
        "semicolons": stripped.count(";"),
        "punctuation_marks": len(re.findall(r"[,;:.!?]", stripped)),
        "markdown_markers": len(re.findall(r"(?:\*\*|#{1,6}|^\s*[-*+]\s+)", stripped, re.M)),
        "parentheses": stripped.count("(") + stripped.count(")"),
        "currency_symbols": len(re.findall(r"[\$£€¥]", stripped)),
        "conclusion_markers": conclusion_markers,
        "answer_line_count": len(answer_matches),
        "has_answer_line": parse_answer(stripped) is not None,
        "answer_is_final_nonempty_line": bool(
            lines and re.match(r"^\s*(?:#{1,6}\s*)?Answer\s*:", lines[-1], re.I)
        ),
        "repeats_question": bool(normalized_question and normalized_question in normalized_response),
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
