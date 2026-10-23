from __future__ import annotations

import ast
import concurrent.futures
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DOMAINS = ("commonsenseqa", "gsm8k", "kodcode", "spider")
SELECTED_QUALITY_DOMAINS = ("commonsenseqa", "gsm8k")
DEFAULT_CAMPAIGN_ROOT = Path(
    "/path/to/runs"
)
FORBIDDEN_INPUT_PARTS = {
    "final_test",
    "sealed_inputs",
    "sealed_test1000",
    "test",
    "official_test",
}
QUALITY_PROMPT_VERSION = "campaign-direct-quality-v1-no-reference-no-system"
QUALITY_PROMPT_SHA256 = (
    "8f6750400eee7bb35657ad41f913401cfb3fb3c12c85091b2251d68b050166d6"
)
IDENTITY_PROMPT_PROVENANCE = {
    "commonsenseqa": (
        "commonsenseqa-pointwise-self-attribution-v2-binary-contract-no-system",
        "c3fcd0cb546a99d173d6d19a00e6720c1a86a170f73d8a6671bfb073f1b24444",
    ),
    "gsm8k": (
        "gsm8k-pointwise-self-attribution-v1-no-system",
        "f4a40ecd84793bcea469bfd89325481bf06bd7bf60ec050b4799af50993f80cb",
    ),
    "kodcode": (
        "phase0-pointwise-self-attribution-v2-no-system",
        "a71239a72e820f2d89472bcfb8aefe1d49ca8b71c90eb82b5c5e6715579b230f",
    ),
    "spider": (
        "spider-1.0-pointwise-self-attribution-v1-no-system",
        "8a798f70adb3221fc976aaf7fd4716f1418d75a136dadfba8bee278e003b9652",
    ),
}


@dataclass(frozen=True)
class RunSpec:
    domain: str
    signal: str
    run_id: str
    reward_calls: Path
    train_examples: Path
    dataset: Path
    max_completion_tokens: int = 768


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected an object at {path}:{line_number}")
            output.append(value)
    return output


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_nonofficial_input(path: Path, campaign_root: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = campaign_root.expanduser().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Input escaped campaign root: {resolved}") from exc
    lowered = {part.lower() for part in relative.parts}
    forbidden = sorted(lowered & FORBIDDEN_INPUT_PARTS)
    if forbidden:
        raise ValueError(
            f"Official/sealed test access is forbidden ({', '.join(forbidden)}): {resolved}"
        )
    return resolved


def _dataset_for_domain(campaign_root: Path, domain: str) -> Path:
    data_root = campaign_root / "01_data_protocol_runtime_gates/datasets"
    paths = {
        "commonsenseqa": data_root
        / "commonsenseqa/frozen_historical_seed23/train.jsonl",
        "gsm8k": data_root / "gsm8k/frozen_historical_seed23/official_train.jsonl",
        "kodcode": data_root / "kodcode/frozen_seed23_v1/train_core1000.jsonl",
        "spider": data_root / "spider/frozen_seed23_v1/train_core1000.jsonl",
    }
    return _assert_nonofficial_input(paths[domain], campaign_root)


def discover_run_specs(campaign_root: Path) -> list[RunSpec]:
    """Resolve only canonical non-test training rollouts from Steps 5 and 7."""
    campaign_root = campaign_root.expanduser().resolve()
    step5 = campaign_root / "05_four_domain_screen_1p5b"
    step7 = campaign_root / "07_baseline_suite"
    suite_path = _assert_nonofficial_input(
        step7 / "results/baseline_suite_validation_v1.json", campaign_root
    )
    suite = read_json(suite_path)
    if (
        suite.get("status") != "complete"
        or suite.get("official_test_accessed") is not False
    ):
        raise RuntimeError(
            "Step 7 canonical validation result is incomplete or test-tainted"
        )

    specs: list[RunSpec] = []
    for domain in DOMAINS:
        run_dir = step5 / f"domains/{domain}/grpo/runs/attempt001"
        specs.append(
            RunSpec(
                domain=domain,
                signal="p_own",
                run_id="step5_grpo_identity_attempt001",
                reward_calls=_assert_nonofficial_input(
                    run_dir / "reward_calls.jsonl", campaign_root
                ),
                train_examples=_assert_nonofficial_input(
                    run_dir / "train_examples.jsonl", campaign_root
                ),
                dataset=_dataset_for_domain(campaign_root, domain),
            )
        )

    for domain in SELECTED_QUALITY_DOMAINS:
        domain_result = suite["results"][domain]
        for method, signal in (("grpo_quality", "q"), ("grpo_oracle", "oracle")):
            run_id = str(domain_result[method]["run_id"])
            run_dir = step7 / f"domains/{domain}/{method}/runs/{run_id}"
            specs.append(
                RunSpec(
                    domain=domain,
                    signal=signal,
                    run_id=f"step7_{method}_{run_id}",
                    reward_calls=_assert_nonofficial_input(
                        run_dir / "reward_calls.jsonl", campaign_root
                    ),
                    train_examples=_assert_nonofficial_input(
                        run_dir / "train_examples.jsonl", campaign_root
                    ),
                    dataset=_dataset_for_domain(campaign_root, domain),
                )
            )
    for spec in specs:
        for path in (spec.reward_calls, spec.train_examples, spec.dataset):
            if not path.is_file():
                raise FileNotFoundError(path)
    return specs


def _candidate_text(row: dict[str, Any]) -> str:
    for field in ("candidate_raw", "candidate", "response"):
        if field in row:
            return str(row[field])
    raise KeyError("Reward row has no candidate text")


def _score_payload(
    payload: tuple[str, dict[str, Any], str, int, str | None],
) -> dict[str, Any]:
    domain, example, candidate, timeout_seconds, spider_database_root = payload
    if domain == "commonsenseqa":
        from iar_cqa.scoring import score_response

        return score_response(candidate, str(example["answer_key"]))
    if domain == "gsm8k":
        from iar_gsm8k.scoring import score_response

        return score_response(candidate, str(example["answer_value"]))
    if domain == "kodcode":
        from iar_campaign.reward_pilot import _score_kodcode

        return _score_kodcode(
            (candidate, str(example["hidden_tests"]), timeout_seconds)
        )
    if domain == "spider":
        if spider_database_root is None:
            raise ValueError("Spider scoring requires --spider-database-root")
        from iar_campaign.score_baselines import _score_spider_one

        value = _score_spider_one(
            (example, {"response": candidate}, Path(spider_database_root))
        )
        return dict(value["score"])
    raise ValueError(f"Unsupported domain: {domain}")


def _batched_token_lengths(
    texts: Sequence[str], *, model_id: str, cache_dir: Path | None
) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        local_files_only=True,
    )
    output: list[int] = []
    for start in range(0, len(texts), 256):
        batch = tokenizer(
            list(texts[start : start + 256]),
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        output.extend(len(tokens) for tokens in batch)
    return output


WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?!\w)")
ANSWER_RE = re.compile(r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*|__)?answer\s*:")
HEDGE_RE = re.compile(
    r"(?i)\b(?:maybe|perhaps|likely|probably|possibly|seems?|appears?|I think)\b"
)
FIRST_PERSON_RE = re.compile(r"(?i)\b(?:I|me|my|mine|we|us|our|ours)\b")
CONCLUSION_RE = re.compile(r"(?i)\b(?:therefore|thus|hence|so|consequently)\b")


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def style_features(text: str, *, domain: str, prompt: str) -> dict[str, float]:
    stripped = (text or "").strip()
    lines = stripped.splitlines()
    words = WORD_RE.findall(stripped)
    nonempty_lines = [line for line in lines if line.strip()]
    bullet_lines = sum(
        bool(re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", line)) for line in lines
    )
    numbered_steps = sum(
        bool(re.match(r"^\s*(?:step\s+)?\d+[.):]\s+", line, re.IGNORECASE))
        for line in lines
    )
    answer_lines = ANSWER_RE.findall(stripped)
    prompt_words = {word.lower() for word in WORD_RE.findall(prompt) if len(word) > 3}
    response_words = {word.lower() for word in words if len(word) > 3}
    overlap = _safe_ratio(
        len(prompt_words & response_words), len(prompt_words | response_words)
    )
    features: dict[str, float] = {
        "style_log_chars": math.log1p(len(stripped)),
        "style_log_words": math.log1p(len(words)),
        "style_log_lines": math.log1p(len(nonempty_lines)),
        "style_mean_word_length": _safe_ratio(
            sum(len(word) for word in words), len(words)
        ),
        "style_numeric_density": _safe_ratio(
            len(NUMBER_RE.findall(stripped)), len(words)
        ),
        "style_punctuation_density": _safe_ratio(
            len(re.findall(r"[,;:.!?]", stripped)), len(stripped)
        ),
        "style_bullet_fraction": _safe_ratio(bullet_lines, len(nonempty_lines)),
        "style_numbered_step_fraction": _safe_ratio(
            numbered_steps, len(nonempty_lines)
        ),
        "style_answer_marker": float(bool(answer_lines)),
        "style_answer_is_final_line": float(
            bool(nonempty_lines and ANSWER_RE.match(nonempty_lines[-1]))
        ),
        "style_markdown_fence": float("```" in stripped),
        "style_markdown_emphasis": float(
            bool(re.search(r"\*\*|__|^#{1,6}\s", stripped, re.MULTILINE))
        ),
        "style_first_person_density": _safe_ratio(
            len(FIRST_PERSON_RE.findall(stripped)), len(words)
        ),
        "style_hedge_density": _safe_ratio(len(HEDGE_RE.findall(stripped)), len(words)),
        "style_conclusion_density": _safe_ratio(
            len(CONCLUSION_RE.findall(stripped)), len(words)
        ),
        "style_prompt_overlap": overlap,
        "style_equation_line_fraction": _safe_ratio(
            sum("=" in line for line in lines), len(nonempty_lines)
        ),
        "style_comment_line_fraction": _safe_ratio(
            sum(line.lstrip().startswith("#") for line in lines), len(nonempty_lines)
        ),
    }
    if domain == "kodcode":
        code = stripped
        try:
            from iar_phase0.generator import strip_code_fences

            code = strip_code_fences(stripped)
        except (ImportError, AttributeError):
            code = re.sub(
                r"^```(?:python)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE
            )
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            syntax_valid = False
            function_count = 0
        else:
            syntax_valid = True
            function_count = sum(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                for node in ast.walk(tree)
            )
        features.update(
            {
                "style_domain_format_valid": float(syntax_valid),
                "style_function_count": float(function_count),
                "style_type_hint_count": float(
                    len(re.findall(r"(?:->|:\s*[A-Z][A-Za-z0-9_\[\], ]*)", code))
                ),
                "style_example_or_test_mentions": float(
                    len(re.findall(r"(?i)\b(?:example|test|sample)\b", code))
                ),
            }
        )
    elif domain == "spider":
        normalized = stripped.upper()
        keywords = re.findall(
            r"\b(?:SELECT|FROM|WHERE|JOIN|GROUP BY|ORDER BY|HAVING|UNION|INTERSECT|EXCEPT|LIMIT)\b",
            normalized,
        )
        format_valid = (
            bool(re.match(r"(?is)^\s*(?:WITH\b.*?\bSELECT|SELECT)\b", stripped))
            and "```" not in stripped
        )
        features.update(
            {
                "style_domain_format_valid": float(format_valid),
                "style_sql_keyword_count": float(len(keywords)),
                "style_sql_join_count": float(
                    len(re.findall(r"(?i)\bJOIN\b", stripped))
                ),
                "style_sql_alias_count": float(
                    len(re.findall(r"(?i)\bAS\s+[A-Za-z_]\w*", stripped))
                ),
            }
        )
    else:
        features["style_domain_format_valid"] = features["style_answer_marker"]
    return features


def difficulty_features(example: dict[str, Any], *, domain: str) -> dict[str, float]:
    question = str(example.get("question", ""))
    reference = str(
        example.get("reference_solution")
        or example.get("reference_query")
        or example.get("reference_abstractive_explanation")
        or ""
    )
    difficulty = str(example.get("gpt_difficulty", "")).lower()
    declared = {"easy": 0.0, "medium": 0.5, "hard": 1.0}.get(difficulty, 0.5)
    reference_complexity = 0.0
    if domain == "spider":
        reference_complexity = float(
            len(
                re.findall(
                    r"(?i)\b(?:JOIN|GROUP BY|ORDER BY|HAVING|UNION|INTERSECT|EXCEPT|SELECT)\b",
                    reference,
                )
            )
        )
    elif domain == "kodcode":
        reference_complexity = float(
            len(re.findall(r"(?m)^\s*(?:def|class|for|while|if)\b", reference))
        )
    else:
        reference_complexity = float(len(NUMBER_RE.findall(reference)))
    return {
        "difficulty_prompt_log_chars": math.log1p(len(question)),
        "difficulty_prompt_numeric_count": float(len(NUMBER_RE.findall(question))),
        "difficulty_reference_log_chars": math.log1p(len(reference)),
        "difficulty_reference_complexity": reference_complexity,
        "difficulty_declared": declared,
    }


def _prompt_from_training_row(row: dict[str, Any]) -> str:
    if row.get("task_statement"):
        return str(row["task_statement"])
    messages = row.get("prompt") or []
    return "\n".join(str(message.get("content", "")) for message in messages)


def _score_rows(
    *,
    domain: str,
    examples: dict[str, dict[str, Any]],
    reward_rows: list[dict[str, Any]],
    workers: int,
    timeout_seconds: int,
    spider_database_root: Path | None,
) -> list[dict[str, Any]]:
    unique_payloads: list[tuple[str, dict[str, Any], str, int, str | None]] = []
    unique_index: dict[tuple[str, str], int] = {}
    row_to_unique: list[int] = []
    for row in reward_rows:
        task_id = str(row["task_id"])
        candidate = _candidate_text(row)
        key = (task_id, hashlib.sha256(candidate.encode()).hexdigest())
        if key not in unique_index:
            unique_index[key] = len(unique_payloads)
            unique_payloads.append(
                (
                    domain,
                    examples[task_id],
                    candidate,
                    timeout_seconds,
                    str(spider_database_root)
                    if spider_database_root is not None
                    else None,
                )
            )
        row_to_unique.append(unique_index[key])
    if domain in {"commonsenseqa", "gsm8k"}:
        unique_scores = [_score_payload(payload) for payload in unique_payloads]
        return [unique_scores[index] for index in row_to_unique]
    # Both expensive evaluators already isolate each candidate in its own child
    # process (pytest for KodCode and the SQL worker for Spider). Threads only
    # coordinate those subprocesses; a second process pool adds memory pressure
    # and can deadlock during forkserver cleanup after worker termination.
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        unique_scores = list(executor.map(_score_payload, unique_payloads, chunksize=4))
    return [unique_scores[index] for index in row_to_unique]


def extract_run(
    spec: RunSpec,
    *,
    campaign_root: Path,
    output_dir: Path,
    workers: int = 8,
    timeout_seconds: int = 60,
    tokenizer_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    tokenizer_cache: Path | None = None,
    spider_database_root: Path | None = None,
) -> dict[str, Any]:
    """Create one compact, deterministically scored rollout table."""
    for path in (spec.reward_calls, spec.train_examples, spec.dataset):
        _assert_nonofficial_input(path, campaign_root)
    output_path = output_dir / f"rows_{spec.domain}_{spec.signal}.jsonl"
    provenance_path = output_dir / f"provenance_{spec.domain}_{spec.signal}.json"
    if output_path.exists() or provenance_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing extraction: {output_path}"
        )

    all_reward_rows = read_jsonl(spec.reward_calls)
    expected_status = "scored" if spec.signal == "oracle" else "ok"
    reward_rows = [
        row for row in all_reward_rows if str(row.get("status")) == expected_status
    ]
    if len(reward_rows) != 8000:
        raise RuntimeError(
            f"Expected exactly 8,000 canonical reward rows for {spec.domain}/{spec.signal}; "
            f"observed {len(reward_rows)}/{len(all_reward_rows)} successful/total. "
            "Failed retry attempts are retained upstream but excluded here."
        )
    if any(row.get("official_test_accessed") is True for row in reward_rows):
        raise RuntimeError("A reward row reports official test access")
    call_ids = [int(row["call_id"]) for row in reward_rows]
    if len(set(call_ids)) != 8000 or set(call_ids) != set(range(1, 8001)):
        raise RuntimeError(
            f"Logical call coverage drifted for {spec.domain}/{spec.signal}"
        )
    reward_prompt_provenance: dict[str, Any]
    if spec.signal == "q":
        versions = {
            str(row.get("judge_prompt_template_version")) for row in reward_rows
        }
        hashes = {str(row.get("judge_prompt_template_sha256")) for row in reward_rows}
        if versions != {QUALITY_PROMPT_VERSION} or hashes != {QUALITY_PROMPT_SHA256}:
            raise RuntimeError(f"Quality prompt provenance drifted: {versions}")
        reward_prompt_provenance = {
            "version": QUALITY_PROMPT_VERSION,
            "sha256": QUALITY_PROMPT_SHA256,
        }
    if spec.signal == "p_own":
        expected_version, expected_hash = IDENTITY_PROMPT_PROVENANCE[spec.domain]
        versions = {
            str(row.get("judge_prompt_template_version")) for row in reward_rows
        }
        hashes = {str(row.get("judge_prompt_template_sha256")) for row in reward_rows}
        if versions != {expected_version} or hashes != {expected_hash}:
            raise RuntimeError(
                f"Identity prompt provenance drifted for {spec.domain}: "
                f"versions={versions}, hashes={hashes}"
            )
        reward_prompt_provenance = {
            "version": expected_version,
            "sha256": expected_hash,
        }
    if spec.signal == "oracle":
        reward_prompt_provenance = {"type": "deterministic_correctness_label"}

    training = read_jsonl(spec.train_examples)
    prompt_by_id = {
        str(row["task_id"]): _prompt_from_training_row(row) for row in training
    }
    dataset_rows = read_jsonl(spec.dataset)
    examples = {str(row["id"]): row for row in dataset_rows}
    task_ids = {str(row["task_id"]) for row in reward_rows}
    if (
        len(task_ids) != 1000
        or not task_ids <= set(examples)
        or not task_ids <= set(prompt_by_id)
    ):
        raise RuntimeError(
            f"Training/dataset coverage mismatch for {spec.domain}/{spec.signal}: "
            f"tasks={len(task_ids)}"
        )

    if spec.signal == "oracle":
        scores = [
            {
                "correct": bool(row["correct"]),
                "valid": bool(row.get("valid", True)),
                "status": str(row.get("status", "scored")),
            }
            for row in reward_rows
        ]
    else:
        scores = _score_rows(
            domain=spec.domain,
            examples=examples,
            reward_rows=reward_rows,
            workers=workers,
            timeout_seconds=timeout_seconds,
            spider_database_root=spider_database_root,
        )
    candidates = [_candidate_text(row) for row in reward_rows]
    token_lengths = _batched_token_lengths(
        candidates, model_id=tokenizer_model, cache_dir=tokenizer_cache
    )

    occurrence: defaultdict[tuple[str, int], int] = defaultdict(int)
    compact: list[dict[str, Any]] = []
    for offset, (reward, candidate, score, token_length) in enumerate(
        zip(reward_rows, candidates, scores, token_lengths, strict=True)
    ):
        task_id = str(reward["task_id"])
        rollout_index = int(reward["rollout_index"])
        key = (task_id, rollout_index)
        occurrence[key] += 1
        example = examples[task_id]
        prompt = prompt_by_id[task_id]
        if spec.signal == "oracle":
            probability = float(reward["reward"])
            training_reward = probability
        else:
            probability = float(reward["p_own"])
            training_reward = float(reward["log_p_own"])
        if not 0.0 <= probability <= 1.0 or not math.isfinite(training_reward):
            raise RuntimeError(f"Non-finite reward at {spec.run_id}:{offset}")
        features = style_features(candidate, domain=spec.domain, prompt=prompt)
        features.update(difficulty_features(example, domain=spec.domain))
        features["style_evaluator_valid"] = float(bool(score.get("valid", True)))
        features["style_truncated"] = float(token_length >= spec.max_completion_tokens)
        features["style_near_truncation"] = float(
            token_length >= math.floor(0.98 * spec.max_completion_tokens)
        )
        compact.append(
            {
                "schema_version": 1,
                "domain": spec.domain,
                "task_type": {
                    "commonsenseqa": "multiple_choice_commonsense",
                    "gsm8k": "numeric_reasoning",
                    "kodcode": "executable_code",
                    "spider": "executable_sql",
                }[spec.domain],
                "signal": spec.signal,
                "run_id": spec.run_id,
                "call_id": int(reward["call_id"]),
                "task_id": task_id,
                "dataset_index": int(reward["dataset_index"]),
                "rollout_index": rollout_index,
                "occurrence": occurrence[key],
                "training_progress": offset / max(1, len(reward_rows) - 1),
                "score_probability": probability,
                "training_reward": training_reward,
                "correct": int(bool(score["correct"])),
                "evaluator_valid": int(bool(score.get("valid", True))),
                "score_status": str(score.get("status", "scored")),
                "completion_tokens": token_length,
                "max_completion_tokens": spec.max_completion_tokens,
                "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
                "features": features,
                "official_test_accessed": False,
            }
        )
    occurrence_counts = Counter(
        (str(row["task_id"]), int(row["rollout_index"])) for row in compact
    )
    if {row["rollout_index"] for row in compact} != {0, 1, 2, 3} or set(
        occurrence_counts.values()
    ) != {2}:
        raise RuntimeError(
            f"Expected two epochs of four rollouts for every prompt: {spec.run_id}"
        )
    write_jsonl(output_path, compact)
    provenance = {
        "schema_version": 1,
        "status": "complete",
        "domain": spec.domain,
        "signal": spec.signal,
        "run_id": spec.run_id,
        "reward_prompt_provenance": reward_prompt_provenance,
        "n": len(compact),
        "source_rows_total_including_failed_attempts": len(all_reward_rows),
        "source_status_counts": dict(
            Counter(str(row.get("status")) for row in all_reward_rows)
        ),
        "n_tasks": len(task_ids),
        "rollout_counts": dict(Counter(row["rollout_index"] for row in compact)),
        "correct_rate": sum(row["correct"] for row in compact) / len(compact),
        "valid_rate": sum(row["evaluator_valid"] for row in compact) / len(compact),
        "truncation_rate": sum(row["features"]["style_truncated"] for row in compact)
        / len(compact),
        "inputs": {
            label: {"path": str(path), "sha256": sha256_file(path)}
            for label, path in (
                ("reward_calls", spec.reward_calls),
                ("train_examples", spec.train_examples),
                ("dataset", spec.dataset),
            )
        },
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "tokenizer": {"model_id": tokenizer_model, "local_files_only": True},
        "official_test_accessed": False,
    }
    write_json(provenance_path, provenance)
    return provenance


def extract_domain(
    *,
    domain: str,
    campaign_root: Path,
    output_dir: Path,
    workers: int = 8,
    timeout_seconds: int = 60,
    tokenizer_cache: Path | None = None,
    spider_database_root: Path | None = None,
) -> list[dict[str, Any]]:
    if domain not in DOMAINS:
        raise ValueError(f"Unsupported domain: {domain}")
    specs = [
        spec for spec in discover_run_specs(campaign_root) if spec.domain == domain
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        extract_run(
            spec,
            campaign_root=campaign_root,
            output_dir=output_dir,
            workers=workers,
            timeout_seconds=timeout_seconds,
            tokenizer_cache=tokenizer_cache,
            spider_database_root=spider_database_root,
        )
        for spec in specs
    ]


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_delta)
        * sum(value * value for value in right_delta)
    )
    if denominator <= 0.0:
        return None
    return (
        sum(a * b for a, b in zip(left_delta, right_delta, strict=True)) / denominator
    )


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return pearson(_average_ranks(left), _average_ranks(right))


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(int(label) for label in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = _average_ranks(scores)
    rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def expected_calibration_error(
    labels: Sequence[int], scores: Sequence[float], bins: int = 10
) -> float | None:
    if not labels:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    error = 0.0
    for bin_index in range(bins):
        start = math.floor(bin_index * len(order) / bins)
        end = math.floor((bin_index + 1) * len(order) / bins)
        members = order[start:end]
        if not members:
            continue
        observed = sum(labels[index] for index in members) / len(members)
        predicted = sum(scores[index] for index in members) / len(members)
        error += len(members) / len(order) * abs(observed - predicted)
    return error


def within_prompt_metrics(
    rows: Sequence[dict[str, Any]],
) -> dict[str, float | int | None]:
    by_task: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    centered_scores: list[float] = []
    centered_rewards: list[float] = []
    centered_labels: list[float] = []
    concordance: list[float] = []
    top_correct: list[float] = []
    mixed = 0
    for task_rows in by_task.values():
        scores = [float(row["score_probability"]) for row in task_rows]
        rewards = [float(row["training_reward"]) for row in task_rows]
        labels = [float(row["correct"]) for row in task_rows]
        score_mean = sum(scores) / len(scores)
        reward_mean = sum(rewards) / len(rewards)
        label_mean = sum(labels) / len(labels)
        centered_scores.extend(value - score_mean for value in scores)
        centered_rewards.extend(value - reward_mean for value in rewards)
        centered_labels.extend(value - label_mean for value in labels)
        correct = [score for score, label in zip(scores, labels, strict=True) if label]
        incorrect = [
            score for score, label in zip(scores, labels, strict=True) if not label
        ]
        if not correct or not incorrect:
            continue
        mixed += 1
        for positive in correct:
            for negative in incorrect:
                concordance.append(
                    1.0 if positive > negative else 0.5 if positive == negative else 0.0
                )
        best = max(scores)
        tied = [
            label for score, label in zip(scores, labels, strict=True) if score == best
        ]
        top_correct.append(sum(tied) / len(tied))
    return {
        "mixed_prompts": mixed,
        "within_prompt_pearson_score": pearson(centered_scores, centered_labels),
        "within_prompt_pearson_training_reward": pearson(
            centered_rewards, centered_labels
        ),
        "within_prompt_concordance": (
            sum(concordance) / len(concordance) if concordance else None
        ),
        "top_reward_correct_rate_mixed": (
            sum(top_correct) / len(top_correct) if top_correct else None
        ),
    }


def calibration_slope_intercept(
    labels: Sequence[int], scores: Sequence[float]
) -> tuple[float | None, float | None]:
    """Fit correctness ~ intercept + slope * logit(score) via small IRLS."""
    try:
        import numpy as np
    except ImportError:
        return None, None
    if len(set(labels)) < 2 or len(scores) < 10:
        return None, None
    clipped = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    predictor = np.log(clipped / (1.0 - clipped))
    if float(np.std(predictor)) <= 1e-12:
        return None, None
    design = np.column_stack([np.ones(len(predictor)), predictor])
    target = np.asarray(labels, dtype=float)
    beta = np.zeros(2)
    penalty = np.diag([1e-8, 1e-6])
    for _ in range(100):
        linear = np.clip(design @ beta, -30, 30)
        probability = 1.0 / (1.0 + np.exp(-linear))
        weight = np.clip(probability * (1.0 - probability), 1e-8, None)
        working = linear + (target - probability) / weight
        system = design.T @ (weight[:, None] * design) + penalty
        update = np.linalg.solve(system, design.T @ (weight * working))
        if float(np.max(np.abs(update - beta))) < 1e-8:
            beta = update
            break
        beta = update
    if not np.all(np.isfinite(beta)) or float(np.max(np.abs(beta))) > 1e4:
        return None, None
    return float(beta[0]), float(beta[1])


def metric_bundle(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(row["correct"]) for row in rows]
    scores = [float(row["score_probability"]) for row in rows]
    rewards = [float(row["training_reward"]) for row in rows]
    if labels and all(
        abs(score - label) <= 1e-12 for score, label in zip(scores, labels, strict=True)
    ):
        # The deterministic oracle is already perfectly calibrated. A logistic
        # recalibration fit has complete separation, so finite intercept/slope
        # estimates are arbitrary and should not be reported.
        intercept, slope = None, None
    else:
        intercept, slope = calibration_slope_intercept(labels, scores)
    output = {
        "n": len(rows),
        "n_prompts": len({str(row["task_id"]) for row in rows}),
        "correct_rate": sum(labels) / len(labels) if labels else None,
        "mean_score": sum(scores) / len(scores) if scores else None,
        "roc_auc": roc_auc(labels, scores),
        "pearson_score": pearson(scores, labels),
        "spearman_score": spearman(scores, labels),
        "pearson_training_reward": pearson(rewards, labels),
        "brier": (
            sum(
                (score - label) ** 2
                for score, label in zip(scores, labels, strict=True)
            )
            / len(labels)
            if labels
            else None
        ),
        "calibration_in_the_large": (
            sum(scores) / len(scores) - sum(labels) / len(labels) if labels else None
        ),
        "ece_equal_frequency_10": expected_calibration_error(labels, scores),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "valid_rate": (
            sum(int(row["evaluator_valid"]) for row in rows) / len(rows)
            if rows
            else None
        ),
        "truncation_rate": (
            sum(float(row["features"]["style_truncated"]) for row in rows) / len(rows)
            if rows
            else None
        ),
    }
    output.update(within_prompt_metrics(rows))
    return output


def _bootstrap_sample_by_prompt(
    rows: Sequence[dict[str, Any]], generator: Any
) -> list[dict[str, Any]]:
    by_task: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    task_ids = sorted(by_task)
    sampled = generator.choice(task_ids, size=len(task_ids), replace=True)
    output: list[dict[str, Any]] = []
    for replica, task_id in enumerate(sampled):
        for row in by_task[str(task_id)]:
            copied = dict(row)
            copied["task_id"] = f"bootstrap::{replica}::{task_id}"
            output.append(copied)
    return output


BOOTSTRAP_METRICS = (
    "roc_auc",
    "pearson_score",
    "spearman_score",
    "pearson_training_reward",
    "brier",
    "calibration_in_the_large",
    "ece_equal_frequency_10",
    "within_prompt_pearson_training_reward",
    "within_prompt_concordance",
)


def bootstrap_metrics(
    rows: Sequence[dict[str, Any]], *, replicates: int, seed: int
) -> dict[str, dict[str, float | int | None]]:
    import numpy as np

    generator = np.random.default_rng(seed)
    values: dict[str, list[float]] = {metric: [] for metric in BOOTSTRAP_METRICS}
    for _ in range(replicates):
        sampled = _bootstrap_sample_by_prompt(rows, generator)
        labels = [int(row["correct"]) for row in sampled]
        scores = [float(row["score_probability"]) for row in sampled]
        rewards = [float(row["training_reward"]) for row in sampled]
        within = within_prompt_metrics(sampled)
        bundle = {
            "roc_auc": roc_auc(labels, scores),
            "pearson_score": pearson(scores, labels),
            "spearman_score": spearman(scores, labels),
            "pearson_training_reward": pearson(rewards, labels),
            "brier": sum(
                (score - label) ** 2
                for score, label in zip(scores, labels, strict=True)
            )
            / len(labels),
            "calibration_in_the_large": sum(scores) / len(scores)
            - sum(labels) / len(labels),
            "ece_equal_frequency_10": expected_calibration_error(labels, scores),
            **within,
        }
        for metric in BOOTSTRAP_METRICS:
            value = bundle.get(metric)
            if value is not None and math.isfinite(float(value)):
                values[metric].append(float(value))
    output: dict[str, dict[str, float | int | None]] = {}
    for metric, observed in values.items():
        if not observed:
            output[metric] = {"low": None, "high": None, "valid_replicates": 0}
            continue
        output[metric] = {
            "low": float(np.quantile(observed, 0.025)),
            "high": float(np.quantile(observed, 0.975)),
            "valid_replicates": len(observed),
        }
    return output


def _rows_for_scope(
    rows: Sequence[dict[str, Any]], rollout_index: int | None
) -> list[dict[str, Any]]:
    if rollout_index is None:
        return list(rows)
    return [row for row in rows if int(row["rollout_index"]) == rollout_index]


def analyze_metrics(
    rows: Sequence[dict[str, Any]], *, bootstrap_replicates: int, seed: int
) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["domain"]), str(row["signal"]))].append(row)
    output: list[dict[str, Any]] = []
    for group_index, ((domain, signal), group_rows) in enumerate(
        sorted(grouped.items())
    ):
        for rollout_index in (None, 0, 1, 2, 3):
            scoped = _rows_for_scope(group_rows, rollout_index)
            bundle = metric_bundle(scoped)
            intervals = bootstrap_metrics(
                scoped,
                replicates=bootstrap_replicates,
                seed=seed + group_index * 100 + (rollout_index or 0),
            )
            record = {
                "domain": domain,
                "signal": signal,
                "scope": "pooled"
                if rollout_index is None
                else f"rollout_{rollout_index}",
                "rollout_index": rollout_index,
                "bootstrap_unit": "task_id",
                "bootstrap_replicates": bootstrap_replicates,
                **bundle,
            }
            for metric, interval in intervals.items():
                record[f"{metric}_ci_low"] = interval["low"]
                record[f"{metric}_ci_high"] = interval["high"]
                record[f"{metric}_bootstrap_valid"] = interval["valid_replicates"]
            output.append(record)
    return output


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _feature_names(rows: Sequence[dict[str, Any]], prefix: str) -> list[str]:
    return sorted(
        {
            str(name)
            for row in rows
            for name in (row.get("features") or {})
            if str(name).startswith(prefix)
        }
    )


def _design_value(row: dict[str, Any], name: str) -> float:
    if name.startswith("feature::"):
        return float(
            (row.get("features") or {}).get(name.removeprefix("feature::"), 0.0)
        )
    if name == "training_progress":
        return float(row["training_progress"])
    if name == "occurrence":
        return float(row["occurrence"])
    if name.startswith("rollout_is_"):
        return float(int(row["rollout_index"]) == int(name.rsplit("_", 1)[-1]))
    if name == "training_reward":
        return float(row["training_reward"])
    if name == "correct":
        return float(row["correct"])
    raise KeyError(name)


def _matrix(rows: Sequence[dict[str, Any]], names: Sequence[str]) -> Any:
    import numpy as np

    return np.asarray(
        [[_design_value(row, name) for name in names] for row in rows], dtype=float
    )


def _cv_predictions(
    rows: Sequence[dict[str, Any]],
    *,
    feature_sets: dict[str, list[str]],
    target_name: str,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    del seed  # GroupKFold is deterministic and preserves group integrity.
    labels = np.asarray([_design_value(row, target_name) for row in rows], dtype=float)
    groups = np.asarray([str(row["task_id"]) for row in rows])
    splitter = GroupKFold(n_splits=folds)
    predictions = {name: np.full(len(rows), np.nan) for name in feature_sets}
    for train_index, test_index in splitter.split(np.zeros(len(rows)), labels, groups):
        for name, features in feature_sets.items():
            train = _matrix([rows[index] for index in train_index], features)
            test = _matrix([rows[index] for index in test_index], features)
            if target_name == "correct":
                model = make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    LogisticRegression(
                        C=1.0,
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=0,
                    ),
                )
                model.fit(train, labels[train_index].astype(int))
                predictions[name][test_index] = model.predict_proba(test)[:, 1]
            else:
                model = make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    Ridge(alpha=1.0),
                )
                model.fit(train, labels[train_index])
                predictions[name][test_index] = model.predict(test)
    if any(not np.all(np.isfinite(values)) for values in predictions.values()):
        raise RuntimeError("Grouped cross-validation left non-finite predictions")
    return {"target": labels, "predictions": predictions, "groups": groups}


def _log_loss(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    clipped = [min(max(float(value), 1e-12), 1.0 - 1e-12) for value in probabilities]
    return -sum(
        label * math.log(probability) + (1 - label) * math.log(1 - probability)
        for label, probability in zip(labels, clipped, strict=True)
    ) / len(labels)


def _r_squared(target: Sequence[float], predicted: Sequence[float]) -> float | None:
    if not target:
        return None
    mean = sum(target) / len(target)
    denominator = sum((value - mean) ** 2 for value in target)
    if denominator <= 0.0:
        return None
    return (
        1.0
        - sum(
            (value - estimate) ** 2
            for value, estimate in zip(target, predicted, strict=True)
        )
        / denominator
    )


def _prediction_metrics(
    target: Sequence[float], predictions: dict[str, Sequence[float]], *, binary: bool
) -> dict[str, dict[str, float | None]]:
    output: dict[str, dict[str, float | None]] = {}
    for name, predicted in predictions.items():
        if binary:
            labels = [int(value) for value in target]
            output[name] = {
                "roc_auc": roc_auc(labels, list(predicted)),
                "log_loss": _log_loss(labels, list(predicted)),
                "brier": sum(
                    (float(probability) - label) ** 2
                    for probability, label in zip(predicted, labels, strict=True)
                )
                / len(labels),
            }
        else:
            output[name] = {
                "r_squared": _r_squared(list(target), list(predicted)),
                "rmse": math.sqrt(
                    sum(
                        (float(value) - float(estimate)) ** 2
                        for value, estimate in zip(target, predicted, strict=True)
                    )
                    / len(target)
                ),
            }
    return output


def _bootstrap_incremental(
    *,
    target: Sequence[float],
    predictions: dict[str, Sequence[float]],
    groups: Sequence[str],
    binary: bool,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float | int | None]]:
    import numpy as np

    by_group: defaultdict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        by_group[str(group)].append(index)
    group_names = sorted(by_group)
    generator = np.random.default_rng(seed)
    samples: defaultdict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        selected_groups = generator.choice(
            group_names, size=len(group_names), replace=True
        )
        indices = [index for group in selected_groups for index in by_group[str(group)]]
        y = [float(target[index]) for index in indices]
        predicted = {
            name: [float(values[index]) for index in indices]
            for name, values in predictions.items()
        }
        metrics = _prediction_metrics(y, predicted, binary=binary)
        if binary:
            deltas = {
                "reward_delta_auc_over_style": (
                    metrics["full"]["roc_auc"] - metrics["style"]["roc_auc"]
                    if metrics["full"]["roc_auc"] is not None
                    and metrics["style"]["roc_auc"] is not None
                    else None
                ),
                "style_delta_auc_over_reward": (
                    metrics["full"]["roc_auc"] - metrics["reward"]["roc_auc"]
                    if metrics["full"]["roc_auc"] is not None
                    and metrics["reward"]["roc_auc"] is not None
                    else None
                ),
                "reward_log_loss_improvement_over_style": (
                    metrics["style"]["log_loss"] - metrics["full"]["log_loss"]
                ),
                "style_log_loss_improvement_over_reward": (
                    metrics["reward"]["log_loss"] - metrics["full"]["log_loss"]
                ),
            }
        else:
            deltas = {
                "correctness_delta_r2_over_style": (
                    metrics["full"]["r_squared"] - metrics["style"]["r_squared"]
                    if metrics["full"]["r_squared"] is not None
                    and metrics["style"]["r_squared"] is not None
                    else None
                ),
                "style_delta_r2_over_correctness": (
                    metrics["full"]["r_squared"] - metrics["correctness"]["r_squared"]
                    if metrics["full"]["r_squared"] is not None
                    and metrics["correctness"]["r_squared"] is not None
                    else None
                ),
            }
        for name, value in deltas.items():
            if value is not None and math.isfinite(float(value)):
                samples[name].append(float(value))
    output: dict[str, dict[str, float | int | None]] = {}
    for name, values in samples.items():
        output[name] = {
            "low": float(np.quantile(values, 0.025)) if values else None,
            "high": float(np.quantile(values, 0.975)) if values else None,
            "valid_replicates": len(values),
        }
    return output


def _demean_by_task(rows: Sequence[dict[str, Any]], values: Any) -> Any:
    import numpy as np

    output = np.asarray(values, dtype=float).copy()
    by_task: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_task[str(row["task_id"])].append(index)
    for indices in by_task.values():
        output[indices] -= np.mean(output[indices], axis=0)
    return output


def _within_prompt_style_controlled_association(
    rows: Sequence[dict[str, Any]],
    style_names: Sequence[str],
    design_names: Sequence[str],
) -> dict[str, float | None]:
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    reward = _demean_by_task(
        rows, np.asarray([float(row["training_reward"]) for row in rows])
    )
    correct = _demean_by_task(rows, np.asarray([float(row["correct"]) for row in rows]))
    style = _demean_by_task(rows, _matrix(rows, style_names))
    controls = _demean_by_task(rows, _matrix(rows, [*style_names, *design_names]))
    varying = np.std(controls, axis=0) > 1e-12
    if not np.any(varying):
        return {
            "within_prompt_partial_corr_reward_correct_given_style": pearson(
                reward.tolist(), correct.tolist()
            ),
            "within_prompt_style_r2_for_reward": 0.0,
            "within_prompt_correctness_r2_for_reward": _r_squared(
                reward.tolist(),
                Ridge(alpha=1.0)
                .fit(correct[:, None], reward)
                .predict(correct[:, None])
                .tolist(),
            ),
        }
    style_varying = np.std(style, axis=0) > 1e-12
    style = StandardScaler().fit_transform(style[:, style_varying])
    controls = StandardScaler().fit_transform(controls[:, varying])
    reward_style_model = Ridge(alpha=1.0).fit(style, reward)
    reward_control_model = Ridge(alpha=1.0).fit(controls, reward)
    correct_control_model = Ridge(alpha=1.0).fit(controls, correct)
    reward_residual = reward - reward_control_model.predict(controls)
    correct_residual = correct - correct_control_model.predict(controls)
    correctness_model = Ridge(alpha=1.0).fit(correct[:, None], reward)
    return {
        "within_prompt_partial_corr_reward_correct_given_style": pearson(
            reward_residual.tolist(), correct_residual.tolist()
        ),
        "within_prompt_style_r2_for_reward": _r_squared(
            reward.tolist(), reward_style_model.predict(style).tolist()
        ),
        "within_prompt_correctness_r2_for_reward": _r_squared(
            reward.tolist(), correctness_model.predict(correct[:, None]).tolist()
        ),
    }


def analyze_incremental_models(
    rows: Sequence[dict[str, Any]], *, folds: int, bootstrap_replicates: int, seed: int
) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["domain"]), str(row["signal"]))].append(row)
    output: list[dict[str, Any]] = []
    for group_index, ((domain, signal), group_rows) in enumerate(
        sorted(grouped.items())
    ):
        style_names = [
            f"feature::{name}" for name in _feature_names(group_rows, "style_")
        ]
        difficulty_names = [
            f"feature::{name}" for name in _feature_names(group_rows, "difficulty_")
        ]
        design_names = [
            "training_progress",
            "occurrence",
            "rollout_is_1",
            "rollout_is_2",
            "rollout_is_3",
        ]
        controls = difficulty_names + design_names
        correctness_sets = {
            "controls": controls,
            "style": controls + style_names,
            "reward": controls + ["training_reward"],
            "full": controls + style_names + ["training_reward"],
        }
        reward_sets = {
            "controls": controls,
            "correctness": controls + ["correct"],
            "style": controls + style_names,
            "full": controls + style_names + ["correct"],
        }
        correctness_cv = _cv_predictions(
            group_rows,
            feature_sets=correctness_sets,
            target_name="correct",
            folds=folds,
            seed=seed + group_index,
        )
        reward_cv = _cv_predictions(
            group_rows,
            feature_sets=reward_sets,
            target_name="training_reward",
            folds=folds,
            seed=seed + 100 + group_index,
        )
        correctness_metrics = _prediction_metrics(
            correctness_cv["target"].tolist(),
            {
                name: value.tolist()
                for name, value in correctness_cv["predictions"].items()
            },
            binary=True,
        )
        reward_metrics = _prediction_metrics(
            reward_cv["target"].tolist(),
            {name: value.tolist() for name, value in reward_cv["predictions"].items()},
            binary=False,
        )
        incremental = {
            "reward_delta_auc_over_style": (
                correctness_metrics["full"]["roc_auc"]
                - correctness_metrics["style"]["roc_auc"]
            ),
            "style_delta_auc_over_reward": (
                correctness_metrics["full"]["roc_auc"]
                - correctness_metrics["reward"]["roc_auc"]
            ),
            "reward_log_loss_improvement_over_style": (
                correctness_metrics["style"]["log_loss"]
                - correctness_metrics["full"]["log_loss"]
            ),
            "style_log_loss_improvement_over_reward": (
                correctness_metrics["reward"]["log_loss"]
                - correctness_metrics["full"]["log_loss"]
            ),
            "correctness_delta_r2_over_style": (
                reward_metrics["full"]["r_squared"]
                - reward_metrics["style"]["r_squared"]
            ),
            "style_delta_r2_over_correctness": (
                reward_metrics["full"]["r_squared"]
                - reward_metrics["correctness"]["r_squared"]
            ),
        }
        correctness_ci = _bootstrap_incremental(
            target=correctness_cv["target"].tolist(),
            predictions={
                name: value.tolist()
                for name, value in correctness_cv["predictions"].items()
            },
            groups=correctness_cv["groups"].tolist(),
            binary=True,
            replicates=bootstrap_replicates,
            seed=seed + 1000 + group_index,
        )
        reward_ci = _bootstrap_incremental(
            target=reward_cv["target"].tolist(),
            predictions={
                name: value.tolist() for name, value in reward_cv["predictions"].items()
            },
            groups=reward_cv["groups"].tolist(),
            binary=False,
            replicates=bootstrap_replicates,
            seed=seed + 2000 + group_index,
        )
        record: dict[str, Any] = {
            "domain": domain,
            "signal": signal,
            "n": len(group_rows),
            "n_prompts": len({str(row["task_id"]) for row in group_rows}),
            "grouped_cv_folds": folds,
            "bootstrap_replicates": bootstrap_replicates,
            "n_style_features": len(style_names),
            "n_difficulty_features": len(difficulty_names),
            **incremental,
            **_within_prompt_style_controlled_association(
                group_rows, style_names, design_names
            ),
        }
        for model_name, values in correctness_metrics.items():
            for metric_name, value in values.items():
                record[f"correctness_model_{model_name}_{metric_name}"] = value
        for model_name, values in reward_metrics.items():
            for metric_name, value in values.items():
                record[f"reward_model_{model_name}_{metric_name}"] = value
        for name, interval in {**correctness_ci, **reward_ci}.items():
            record[f"{name}_ci_low"] = interval["low"]
            record[f"{name}_ci_high"] = interval["high"]
            record[f"{name}_bootstrap_valid"] = interval["valid_replicates"]
        output.append(record)
    return output


def snapshot_campaign_context(
    *, campaign_root: Path, output_dir: Path
) -> dict[str, Any]:
    campaign_root = campaign_root.expanduser().resolve()
    step5_path = _assert_nonofficial_input(
        campaign_root
        / "05_four_domain_screen_1p5b/results/four_domain_screen_1p5b.json",
        campaign_root,
    )
    step7_path = _assert_nonofficial_input(
        campaign_root / "07_baseline_suite/results/baseline_suite_validation_v1.json",
        campaign_root,
    )
    step5 = read_json(step5_path)
    step7 = read_json(step7_path)
    if any(
        value.get("status") != "complete"
        or value.get("official_test_accessed") is not False
        for value in (step5, step7)
    ):
        raise RuntimeError("Canonical validation context is incomplete or test-tainted")
    downstream: dict[str, Any] = {}
    for domain in DOMAINS:
        screen = step5["domains"][domain]
        downstream[domain] = {
            "a0": float(screen["a0_task_utility"]),
            "prompt_imitation": float(screen["prompt_imitation_task_utility"]),
            "grpo_identity": float(screen["selected_task_utility"]),
            "grpo_identity_gain_vs_a0": float(screen["validation_gain_vs_a0"]),
        }
        if domain in SELECTED_QUALITY_DOMAINS:
            baseline = step7["results"][domain]
            downstream[domain].update(
                {
                    "sft_teacher": float(baseline["sft_teacher"]["task_utility"]),
                    "grpo_quality": float(baseline["grpo_quality"]["task_utility"]),
                    "grpo_oracle": float(baseline["grpo_oracle"]["task_utility"]),
                }
            )
    context = {
        "schema_version": 1,
        "campaign": campaign_root.name,
        "scope": "nonofficial training rollouts and validation summaries only",
        "coverage": {
            domain: {
                "p_own": True,
                "q": domain in SELECTED_QUALITY_DOMAINS,
                "oracle": domain in SELECTED_QUALITY_DOMAINS,
            }
            for domain in DOMAINS
        },
        "downstream_validation_task_utility": downstream,
        "sources": {
            "step5": {"path": str(step5_path), "sha256": sha256_file(step5_path)},
            "step7": {"path": str(step7_path), "sha256": sha256_file(step7_path)},
        },
        "official_test_accessed": False,
    }
    path = output_dir / "campaign_context.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite campaign context: {path}")
    write_json(path, context)
    return context


def load_extracted_rows(output_dir: Path) -> list[dict[str, Any]]:
    expected = [
        (domain, signal)
        for domain in DOMAINS
        for signal in (
            ("p_own", "q", "oracle")
            if domain in SELECTED_QUALITY_DOMAINS
            else ("p_own",)
        )
    ]
    all_rows: list[dict[str, Any]] = []
    for domain, signal in expected:
        rows_path = output_dir / f"rows_{domain}_{signal}.jsonl"
        provenance_path = output_dir / f"provenance_{domain}_{signal}.json"
        if not rows_path.is_file() or not provenance_path.is_file():
            raise FileNotFoundError(f"Missing extraction for {domain}/{signal}")
        provenance = read_json(provenance_path)
        if (
            provenance.get("status") != "complete"
            or provenance.get("official_test_accessed") is not False
            or provenance.get("output", {}).get("sha256") != sha256_file(rows_path)
        ):
            raise RuntimeError(f"Invalid provenance for {domain}/{signal}")
        rows = read_jsonl(rows_path)
        if len(rows) != 8000 or any(
            row.get("official_test_accessed") is not False
            or row.get("domain") != domain
            or row.get("signal") != signal
            for row in rows
        ):
            raise RuntimeError(f"Invalid compact rows for {domain}/{signal}")
        all_rows.extend(rows)
    return all_rows


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "NA"
    return f"{numeric:.{digits}f}"


def _ci(row: dict[str, Any], metric: str, digits: int = 3) -> str:
    value = _fmt(row.get(metric), digits)
    low = row.get(f"{metric}_ci_low")
    high = row.get(f"{metric}_ci_high")
    if low is None or high is None:
        return value
    return f"{value} [{_fmt(low, digits)}, {_fmt(high, digits)}]"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _alignment_label(row: dict[str, Any]) -> str:
    if row["signal"] == "oracle":
        return "oracle deterministico (allineamento tautologico)"
    correctness = float(row["correctness_delta_r2_over_style"])
    style = float(row["style_delta_r2_over_correctness"])
    partial = abs(
        float(row.get("within_prompt_partial_corr_reward_correct_given_style") or 0.0)
    )
    margin = 0.01
    if correctness > max(style + margin, 1.5 * max(style, 0.0)) and partial >= 0.10:
        return "più allineato a competenza/correttezza"
    if style > max(correctness + margin, 1.5 * max(correctness, 0.0)):
        return "più allineato a stile/formato"
    return "misto / non separabile"


def _make_figures(
    *,
    rows: Sequence[dict[str, Any]],
    metrics: Sequence[dict[str, Any]],
    models: Sequence[dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    pooled = [row for row in metrics if row["scope"] == "pooled"]
    labels = [f"{row['domain']} / {row['signal']}" for row in pooled]
    values = np.asarray([float(row["roc_auc"]) for row in pooled])
    lower = np.asarray([float(row["roc_auc_ci_low"]) for row in pooled])
    upper = np.asarray([float(row["roc_auc_ci_high"]) for row in pooled])
    colors = [
        {"p_own": "#167d8d", "q": "#d17c0b", "oracle": "#555555"}[row["signal"]]
        for row in pooled
    ]
    fig, axis = plt.subplots(figsize=(9, max(4.5, 0.5 * len(labels))))
    positions = np.arange(len(labels))
    axis.errorbar(
        values,
        positions,
        xerr=np.vstack([values - lower, upper - values]),
        fmt="none",
        ecolor="#777777",
        capsize=3,
        linewidth=1.5,
    )
    axis.scatter(values, positions, c=colors, s=45, zorder=3)
    axis.axvline(0.5, color="#999999", linestyle="--", linewidth=1)
    axis.set(
        yticks=positions,
        yticklabels=labels,
        xlabel="ROC AUC per la correttezza deterministica",
    )
    axis.set_xlim(0.45, 1.02)
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    auc_path = figure_dir / "reward_correctness_auc.png"
    fig.savefig(auc_path, dpi=180)
    plt.close(fig)

    nonoracle = [row for row in models if row["signal"] != "oracle"]
    labels = [f"{row['domain']} / {row['signal']}" for row in nonoracle]
    correctness = [float(row["correctness_delta_r2_over_style"]) for row in nonoracle]
    style = [float(row["style_delta_r2_over_correctness"]) for row in nonoracle]
    positions = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9, max(4.5, 0.6 * len(labels))))
    axis.barh(
        positions - width / 2,
        correctness,
        height=width,
        label="correttezza aggiunta dopo lo stile",
        color="#2d7f5e",
    )
    axis.barh(
        positions + width / 2,
        style,
        height=width,
        label="stile/formato aggiunto dopo la correttezza",
        color="#b66a25",
    )
    axis.axvline(0.0, color="#555555", linewidth=0.8)
    axis.set(
        yticks=positions,
        yticklabels=labels,
        xlabel="R² incrementale in CV raggruppata per predire il reward osservato",
    )
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="best")
    fig.tight_layout()
    incremental_path = figure_dir / "incremental_reward_explanation.png"
    fig.savefig(incremental_path, dpi=180)
    plt.close(fig)

    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["domain"]), str(row["signal"]))].append(row)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
    for axis, ((domain, signal), group_rows) in zip(
        axes.flat, sorted(grouped.items()), strict=True
    ):
        labels_array = np.asarray([int(row["correct"]) for row in group_rows])
        scores_array = np.asarray(
            [float(row["score_probability"]) for row in group_rows]
        )
        order = np.argsort(scores_array)
        chunks = np.array_split(order, 10)
        x = [float(np.mean(scores_array[chunk])) for chunk in chunks if len(chunk)]
        y = [float(np.mean(labels_array[chunk])) for chunk in chunks if len(chunk)]
        axis.plot([0, 1], [0, 1], "--", color="#999999", linewidth=0.8)
        axis.plot(
            x,
            y,
            "o-",
            color={"p_own": "#167d8d", "q": "#d17c0b", "oracle": "#555555"}[signal],
        )
        axis.set_title(f"{domain} / {signal}")
        axis.grid(alpha=0.15)
    fig.supxlabel("Probabilità reward media nel bin a frequenza uguale")
    fig.supylabel("Correttezza deterministica osservata")
    fig.tight_layout()
    calibration_path = figure_dir / "reward_calibration.png"
    fig.savefig(calibration_path, dpi=180)
    plt.close(fig)
    return [str(auc_path), str(incremental_path), str(calibration_path)]


def render_report(
    *,
    context: dict[str, Any],
    metrics: Sequence[dict[str, Any]],
    models: Sequence[dict[str, Any]],
    output_dir: Path,
    figure_paths: Sequence[str],
) -> str:
    pooled = [row for row in metrics if row["scope"] == "pooled"]
    downstream = context["downstream_validation_task_utility"]
    pooled_by_key = {(str(row["domain"]), str(row["signal"])): row for row in pooled}
    models_by_key = {(str(row["domain"]), str(row["signal"])): row for row in models}
    coverage_rows = []
    for domain in DOMAINS:
        available = {
            str(row["signal"]): int(row["n"])
            for row in pooled
            if row["domain"] == domain
        }
        coverage_rows.append(
            [
                domain,
                available.get("p_own", "not collected"),
                available.get("q", "not collected"),
                available.get("oracle", "not collected"),
            ]
        )
    primary_rows = [
        [
            row["domain"],
            row["signal"],
            row["n"],
            _ci(row, "roc_auc"),
            _ci(row, "pearson_score"),
            _ci(row, "pearson_training_reward"),
            _ci(row, "within_prompt_pearson_training_reward"),
            _ci(row, "brier"),
            _ci(row, "calibration_in_the_large"),
            _fmt(row["calibration_slope"]),
            _ci(row, "ece_equal_frequency_10"),
            _fmt(row["valid_rate"]),
            _fmt(row["truncation_rate"]),
        ]
        for row in pooled
    ]
    incremental_rows = [
        [
            row["domain"],
            row["signal"],
            _ci(row, "reward_delta_auc_over_style"),
            _ci(row, "style_delta_auc_over_reward"),
            _ci(row, "correctness_delta_r2_over_style"),
            _ci(row, "style_delta_r2_over_correctness"),
            _fmt(row["within_prompt_partial_corr_reward_correct_given_style"]),
            _alignment_label(row),
        ]
        for row in models
    ]
    downstream_rows = []
    for domain in DOMAINS:
        values = downstream[domain]
        downstream_rows.append(
            [
                domain,
                _fmt(values["a0"]),
                _fmt(values["grpo_identity"]),
                _fmt(values["grpo_identity_gain_vs_a0"]),
                _fmt(values.get("grpo_quality")),
                _fmt(values.get("grpo_oracle")),
            ]
        )
    rollout_rows = [
        [
            row["domain"],
            row["signal"],
            row["scope"],
            row["n"],
            _ci(row, "roc_auc"),
            _ci(row, "pearson_training_reward"),
            _ci(row, "within_prompt_pearson_training_reward"),
            _ci(row, "brier"),
            _fmt(row["correct_rate"]),
        ]
        for row in metrics
        if row["scope"] != "pooled"
    ]
    labels = "; ".join(
        f"{row['domain']}/{row['signal']}: {_alignment_label(row)}" for row in models
    )
    task_types = {
        "commonsenseqa": "scelta multipla di senso comune",
        "gsm8k": "ragionamento aritmetico con risposta numerica",
        "kodcode": "generazione di codice verificata da test",
        "spider": "text-to-SQL verificato per esecuzione",
    }
    task_readings = []
    for domain in DOMAINS:
        p_metric = pooled_by_key[(domain, "p_own")]
        p_model = models_by_key[(domain, "p_own")]
        values = downstream[domain]
        reading = (
            f"- **{domain} ({task_types[domain]}).** `p_own`: AUC "
            f"{_fmt(p_metric['roc_auc'])}, correlazione within-prompt "
            f"{_fmt(p_metric['within_prompt_pearson_training_reward'])}; "
            f"ΔR² correttezza/stile "
            f"{_fmt(p_model['correctness_delta_r2_over_style'])}/"
            f"{_fmt(p_model['style_delta_r2_over_correctness'])} "
            f"({_alignment_label(p_model)}). La task utility di validation passa "
            f"da {_fmt(values['a0'])} a {_fmt(values['grpo_identity'])} "
            f"(Δ {_fmt(values['grpo_identity_gain_vs_a0'])})."
        )
        if domain in SELECTED_QUALITY_DOMAINS:
            q_metric = pooled_by_key[(domain, "q")]
            q_model = models_by_key[(domain, "q")]
            reading += (
                f" `q`: AUC {_fmt(q_metric['roc_auc'])}, correlazione "
                f"within-prompt "
                f"{_fmt(q_metric['within_prompt_pearson_training_reward'])}, "
                f"ΔR² correttezza/stile "
                f"{_fmt(q_model['correctness_delta_r2_over_style'])}/"
                f"{_fmt(q_model['style_delta_r2_over_correctness'])} "
                f"({_alignment_label(q_model)}); utility GRPO "
                f"{_fmt(values['grpo_quality'])}. L'oracle ottiene "
                f"{_fmt(values['grpo_oracle'])}, ma il suo allineamento locale è "
                "tautologico perché coincide con la label."
            )
        else:
            reading += " `q` e oracle non sono stati raccolti per questo dominio."
        if domain == "kodcode":
            reading += (
                " Nel modello complementare di correttezza, tuttavia, lo stile/formato "
                f"aggiunto dopo il reward vale ΔAUC "
                f"{_fmt(p_model['style_delta_auc_over_reward'])}, contro "
                f"{_fmt(p_model['reward_delta_auc_over_style'])} del reward dopo lo "
                "stile: la validità formale resta una feature task-relevant che "
                "`p_own` non riassume completamente."
            )
        task_readings.append(reading)
    task_readings_markdown = "\n".join(task_readings)
    p_models = [models_by_key[(domain, "p_own")] for domain in DOMAINS]
    p_metrics = [pooled_by_key[(domain, "p_own")] for domain in DOMAINS]
    p_min_auc = min(float(row["roc_auc"]) for row in p_metrics)
    p_max_auc = max(float(row["roc_auc"]) for row in p_metrics)
    p_min_partial = min(
        float(row["within_prompt_partial_corr_reward_correct_given_style"])
        for row in p_models
    )
    p_max_partial = max(
        float(row["within_prompt_partial_corr_reward_correct_given_style"])
        for row in p_models
    )
    executive_findings = f"""
- `p_own` contiene informazione incrementale di correttezza in tutti e quattro i tipi di task: AUC {_fmt(p_min_auc)}–{_fmt(p_max_auc)} e correlazione parziale within-prompt dopo stile/formato {_fmt(p_min_partial)}–{_fmt(p_max_partial)}. Nel modello che spiega il reward, il ΔR² della correttezza supera quello dello stile in ogni dominio.
- L'allineamento identity più forte è su text-to-SQL (Spider: AUC {_fmt(pooled_by_key[("spider", "p_own")]["roc_auc"])}, partial r {_fmt(models_by_key[("spider", "p_own")]["within_prompt_partial_corr_reward_correct_given_style"])}); il più debole è sulla scelta multipla CQA (AUC {_fmt(pooled_by_key[("commonsenseqa", "p_own")]["roc_auc"])}, partial r {_fmt(models_by_key[("commonsenseqa", "p_own")]["within_prompt_partial_corr_reward_correct_given_style"])}).
- `q` è nettamente competence-aligned su CQA, ma su GSM8K è misto: correttezza e stile aggiungono ΔR² simili ({_fmt(models_by_key[("gsm8k", "q")]["correctness_delta_r2_over_style"])} e {_fmt(models_by_key[("gsm8k", "q")]["style_delta_r2_over_correctness"])}).
- I risultati per rollout sono stabili: nessun range AUC tra indici 0–3 inverte l'ordinamento sostantivo. La calibrazione assoluta non lo è: `p_own` sottostima fortemente la correttezza in CQA/GSM8K, mentre `q` la sovrastima.
- L'allineamento locale non ordina la utility finale nel singolo seed: su CQA `q` ha AUC maggiore ma GRPO identity ottiene {_fmt(downstream["commonsenseqa"]["grpo_identity"])} contro {_fmt(downstream["commonsenseqa"]["grpo_quality"])}; su GSM8K `q` ottiene invece {_fmt(downstream["gsm8k"]["grpo_quality"])} contro {_fmt(downstream["gsm8k"]["grpo_identity"])} di identity.
""".strip()
    bootstrap_replicates = metrics[0]["bootstrap_replicates"] if metrics else 0
    figure_rel = [str(Path(path).relative_to(output_dir)) for path in figure_paths]
    return f"""# Audit di allineamento dei reward — campaign_20260908_v2

## Risposta in breve

Questo audit misura associazioni, non identifica un meccanismo causale. La lettura descrittiva delle analisi incrementali è: {labels}. Le righe oracle sono un ceiling di controllo, non evidenza indipendente: in questa campagna l'oracle reward coincide con il target di correttezza deterministica.

Il confronto è completo per `p_own` nei quattro domini. `q` e il braccio oracle sono stati raccolti soltanto per CommonsenseQA e GSM8K; riportarli per KodCode o Spider richiederebbe nuove chiamate al judge o nuovi bracci, quindi quelle celle sono marcate intenzionalmente come non disponibili.

### Esito sostantivo

{executive_findings}

## Copertura dei dati e protezione dal leakage

{_markdown_table(["domain", "p_own rows", "q rows", "oracle rows"], coverage_rows)}

Tutti gli input sono rollout di training degli Step 5 e 7; la performance downstream è esclusivamente di validation. L'estrattore rifiuta i path di test o sigillati, verifica gli hash delle fonti e registra `official_test_accessed=false` in ogni riga e record di provenienza.

## Associazione tra reward e correttezza

Gli intervalli sono intervalli percentile al 95% da cluster bootstrap sui prompt ({bootstrap_replicates} repliche). `Pearson(log reward)` usa il reward effettivamente ottimizzato: `log(p_own)` o `log(q)`; per l'oracle usa lo score binario. Le correlazioni within-prompt rimuovono ogni effetto fisso del prompt.

{_markdown_table(["domain", "signal", "n", "AUC", "Pearson(prob.)", "Pearson(log reward)", "within-prompt Pearson", "Brier", "cal. gap", "cal. slope", "ECE-10", "valid rate", "trunc. rate"], primary_rows)}

La calibrazione è descrittiva: `p_own` stima l'autorialità e `q` è un giudizio di qualità senza reference, quindi nessuno dei due è per definizione una probabilità di correttezza. Brier ed ECE quantificano questo disallineamento.

![AUC with prompt-clustered confidence intervals]({figure_rel[0]})

![Calibration curves]({figure_rel[2]})

## Correttezza contro stile/formato come spiegazioni del reward

Sono stimati due modelli complementari con cross-validation raggruppata a cinque fold. Il primo predice la correttezza da controlli di difficoltà/disegno, feature di stile/formato, reward o entrambi. Il secondo predice il reward osservato da controlli più correttezza deterministica, stile/formato o entrambi. Il raggruppamento per prompt impedisce che lo stesso problema entri nei fold di training e validation. Lo stile include lunghezza, formato valido, troncamento al cap del tokenizer, marker di risposta, Markdown, struttura in passi/equazioni, overlap lessicale e forma specifica di codice/SQL.

Un `ΔR² correttezza` positivo indica che la correttezza spiega il reward oltre lo stile; un `ΔR² stile` positivo indica che lo stile spiega il reward oltre la correttezza. La correlazione parziale within-prompt rimuove anche gli effetti fissi del problema e le feature stilistiche misurate. Sono decomposizioni predittive, non mediatori o effetti causali.

{_markdown_table(["domain", "signal", "ΔAUC reward after style", "ΔAUC style after reward", "ΔR² correctness after style", "ΔR² style after correctness", "within-prompt partial r", "descriptive reading"], incremental_rows)}

![Incremental reward explanation]({figure_rel[1]})

## Lettura per natura del task

{task_readings_markdown}

Il confronto tra AUC locale e utility finale è deliberatamente descrittivo. Per esempio, un reward può ordinare bene i rollout della propria policy senza produrre la policy con la migliore utility, perché cambiano esplorazione, distribuzione dei candidati, scala/varianza del gradiente e traiettoria di ottimizzazione.

## Relazione con la performance downstream su validation

{_markdown_table(["domain", "A0", "GRPO identity", "identity gain", "GRPO q", "GRPO oracle"], downstream_rows)}

Queste differenze mostrano che la scelta del reward cambia la policy addestrata, ma è disponibile un solo seed e le decomposizioni dei rollout sono osservazioni post-trattamento da policy che evolvono diversamente. Non identificano quindi se un gain downstream sia stato *causato attraverso* la correttezza anziché lo stile. Per una conclusione causale servirebbero riscritture controfattuali o ablation sugli stessi candidati.

## Risultati per indice di rollout

Ogni prompt ha quattro rollout campionati in ciascuna di due epoche. La tabella ripete le diagnostiche separatamente per gli indici 0–3; il bootstrap resta clusterizzato per prompt.

{_markdown_table(["domain", "signal", "scope", "n", "AUC", "Pearson(log reward)", "within-prompt Pearson", "Brier", "correct rate"], rollout_rows)}

## Riproduzione e limiti interpretativi

- Lo score oracle coincide con la label di correttezza per CommonsenseQA/GSM8K: il suo allineamento rispetto a quella label è tautologico.
- `q` è salvato upstream in campi generici `p_own`, ma viene rinominato solo dopo la verifica di versione e hash del prompt di qualità.
- Il troncamento è ricalcolato con il tokenizer Qwen2.5 congelato e il cap di 768 token; la validità di formato è domain-native.
- Il bootstrap per prompt gestisce la dipendenza tra rollout dello stesso problema, non il singolo seed né il distribution shift.
- AUC, correlazioni, Brier, ECE e gap di calibrazione hanno CI bootstrap; intercetta e slope di calibrazione sono diagnostiche puntuali per evitare migliaia di fit logistici instabili nei campioni di rollout più piccoli.
- L'estrazione KodCode completa è stata eseguita con Python 3.13 su copie byte-identiche degli input canonici (tre hash verificati sul cluster). I tentativi Python 3.14 sul cluster non hanno prodotto file parziali utilizzabili per problemi NFS, timeout o OOM causato da process tree avviati da alcuni candidati; le stime KodCode vanno quindi considerate runtime-specific finché non sarà disponibile una replica 3.14 con limiti sull'intero process tree. La ricevuta `kodcode_extraction_reconciliation.json` documenta input, job e hash; lo script Slurm limita ora KodCode a due evaluator concorrenti.
- L'assenza di `q`/oracle su codice e SQL è un limite degli artefatti, non evidenza di assenza di relazione.
- Nessun artefatto di test ufficiale o sigillato è stato letto o materializzato.
"""


def render_saved_analysis(*, output_dir: Path, seed: int = 23) -> dict[str, Any]:
    """Render saved statistics without repeating bootstrap/CV computation."""
    context = read_json(output_dir / "campaign_context.json")
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    models = json.loads(
        (output_dir / "incremental_models.json").read_text(encoding="utf-8")
    )
    if not isinstance(metrics, list) or not isinstance(models, list):
        raise TypeError("Saved metrics and incremental models must be JSON arrays")
    rows = load_extracted_rows(output_dir)
    if context.get("official_test_accessed") is not False:
        raise RuntimeError("Campaign context is test-tainted")
    if len(metrics) != 40 or len(models) != 8:
        raise RuntimeError("Saved analysis has incomplete domain/signal coverage")
    bootstrap_values = {int(row["bootstrap_replicates"]) for row in metrics + models}
    fold_values = {int(row["grouped_cv_folds"]) for row in models}
    if len(bootstrap_values) != 1 or len(fold_values) != 1:
        raise RuntimeError("Saved analysis configuration is inconsistent")
    figures = _make_figures(
        rows=rows, metrics=metrics, models=models, output_dir=output_dir
    )
    report = render_report(
        context=context,
        metrics=metrics,
        models=models,
        output_dir=output_dir,
        figure_paths=figures,
    )
    report_path = output_dir / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    bootstrap_replicates = bootstrap_values.pop()
    folds = fold_values.pop()
    artifact_paths = [
        output_dir / "metrics.json",
        output_dir / "metrics.csv",
        output_dir / "incremental_models.json",
        output_dir / "incremental_models.csv",
        report_path,
    ]
    for supporting_name in (
        "kodcode_extraction_reconciliation.json",
        "reward_prompt_provenance.json",
    ):
        supporting_path = output_dir / supporting_name
        if supporting_path.is_file():
            artifact_paths.append(supporting_path)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "n_rows": len(rows),
        "n_domain_signal_groups": len(models),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_unit": "task_id",
        "grouped_cv_folds": folds,
        "seed": seed,
        "implementation": {
            "module_path": str(Path(__file__).resolve()),
            "module_sha256": sha256_file(Path(__file__).resolve()),
            "python": sys.version,
        },
        "artifacts": {
            path.name: {"path": str(path), "sha256": sha256_file(path)}
            for path in artifact_paths
        },
        "figures": [
            {"path": path, "sha256": sha256_file(Path(path))} for path in figures
        ],
        "official_test_accessed": False,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def analyze_extractions(
    *,
    output_dir: Path,
    bootstrap_replicates: int = 1000,
    folds: int = 5,
    seed: int = 23,
) -> dict[str, Any]:
    context_path = output_dir / "campaign_context.json"
    if not context_path.is_file():
        raise FileNotFoundError(context_path)
    context = read_json(context_path)
    if context.get("official_test_accessed") is not False:
        raise RuntimeError("Campaign context is test-tainted")
    rows = load_extracted_rows(output_dir)
    metrics = analyze_metrics(
        rows, bootstrap_replicates=bootstrap_replicates, seed=seed
    )
    models = analyze_incremental_models(
        rows,
        folds=folds,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    write_json(output_dir / "metrics.json", metrics)
    write_csv(output_dir / "metrics.csv", metrics)
    write_json(output_dir / "incremental_models.json", models)
    write_csv(output_dir / "incremental_models.csv", models)
    return render_saved_analysis(output_dir=output_dir, seed=seed)
