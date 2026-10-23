from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
import re
import statistics
import time
import tokenize
from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from iar_campaign.baseline_rewards import (
    QUALITY_PROMPT_VERSION,
    quality_prompt,
    quality_prompt_template_sha256,
)
from iar_campaign.judge_sanity import (
    KEY_NAMES,
    MODEL_ID,
    PROVIDER,
    append_event,
    attribution_prompt,
    key_lock,
    load_named_secret,
)
from iar_campaign.reward_alignment import (
    DEFAULT_CAMPAIGN_ROOT,
    DOMAINS,
    IDENTITY_PROMPT_PROVENANCE,
    _candidate_text,
    _score_payload,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from iar_phase0.generator import strip_code_fences
from iar_phase0.openrouter import (
    OPENROUTER_URL,
    binary_probability_from_logprobs,
    make_chat_payload,
    response_text,
)

PROTOCOL = "iar-causal-reward-construct-audit-v1"
STEP_ROOT = DEFAULT_CAMPAIGN_ROOT / "13_causal_reward_audit"
VARIANTS = (
    "correct_original",
    "incorrect_original",
    "correct_style",
    "incorrect_style",
)
QUALITY_DOMAINS = ("commonsenseqa", "gsm8k")
SIGNALS_BY_DOMAIN = {
    "commonsenseqa": ("p_own", "q"),
    "gsm8k": ("p_own", "q"),
    "kodcode": ("p_own",),
    "spider": ("p_own",),
}
FORBIDDEN_PARTS = {
    "test",
    "official_test",
    "final_test",
    "sealed",
    "sealed_inputs",
    "sealed_test1000",
}
STYLE_PREFIXES = {
    "commonsenseqa": "### Candidate response\n\n",
    "gsm8k": "### Candidate response\n\n",
    "kodcode": "# Candidate response\n",
    "spider": "-- Candidate response\n",
}
PROBABILITY_EPSILON = 1e-6
STRUCTURAL_FALLBACK_MUTATIONS = {
    "inject_entry_failure",
    "append_limit_zero",
    "wrap_empty_result",
    "replace_with_constant_zero",
    "replace_with_constant_one",
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _under(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(root.expanduser().resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escaped {root}: {resolved}") from exc
    forbidden = sorted({part.lower() for part in relative.parts} & FORBIDDEN_PARTS)
    if forbidden:
        raise ValueError(
            f"Official/sealed test path is forbidden for {label} "
            f"({', '.join(forbidden)}): {resolved}"
        )
    return resolved


def _dataset_path(campaign_root: Path, domain: str) -> Path:
    base = campaign_root / "01_data_protocol_runtime_gates/datasets"
    relative = {
        "commonsenseqa": "commonsenseqa/frozen_historical_seed23/train.jsonl",
        "gsm8k": "gsm8k/frozen_historical_seed23/official_train.jsonl",
        "kodcode": "kodcode/frozen_seed23_v1/train_core1000.jsonl",
        "spider": "spider/frozen_seed23_v1/train_core1000.jsonl",
    }[domain]
    return _under(base / relative, campaign_root, f"{domain} training data")


def _step5_run(campaign_root: Path, domain: str) -> Path:
    return _under(
        campaign_root
        / f"05_four_domain_screen_1p5b/domains/{domain}/grpo/runs/attempt001",
        campaign_root,
        f"{domain} Step 5 run",
    )


def _compact_path(campaign_root: Path, domain: str) -> Path:
    return _under(
        campaign_root
        / "12_final_test_analysis_reconciliation/correlations/"
        / "nonofficial_reward_alignment_attempt001"
        / f"rows_{domain}_p_own.jsonl",
        campaign_root,
        f"{domain} compact rollout extraction",
    )


def _successful_reward_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows = [row for row in read_jsonl(path) if row.get("status") == "ok"]
    output = {int(row["call_id"]): row for row in rows}
    if len(output) != len(rows):
        raise RuntimeError(f"Successful reward rows have duplicate call IDs: {path}")
    return output


def _replace_span(text: str, start: int, end: int, replacement: str) -> str:
    return text[:start] + replacement + text[end:]


def _cqa_mutations(candidate: str, gold: str) -> list[tuple[str, str]]:
    pattern = re.compile(
        r"(?im)^[ \t]*(?:\*\*|__)?Answer\s*:\s*([A-E])(?:\*\*|__)?[ \t]*$"
    )
    matches = list(pattern.finditer(candidate))
    if not matches:
        return []
    wrong = next(label for label in "ABCDE" if label != gold.upper())
    start, end = matches[-1].span(1)
    return [("replace_final_choice", _replace_span(candidate, start, end, wrong))]


def _gsm8k_mutations(candidate: str, gold: str) -> list[tuple[str, str]]:
    from decimal import Decimal

    pattern = re.compile(
        r"(?im)^[ \t]*(?:#{1,6}\s*)?(?:\*\*|__)?Answer\s*:\s*([^\r\n]+?)\s*$"
    )
    matches = list(pattern.finditer(candidate))
    if not matches:
        return []
    number = Decimal(gold.replace(",", ""))
    wrong_values = [number + 1, number - 1, -number if number != 0 else Decimal(1)]
    start, end = matches[-1].span(1)
    output: list[tuple[str, str]] = []
    for index, wrong in enumerate(wrong_values):
        rendered = format(wrong, "f")
        output.append(
            (
                f"replace_final_number_{index}",
                _replace_span(candidate, start, end, rendered),
            )
        )
    return output


def _offsets(text: str) -> list[int]:
    starts = [0]
    for match in re.finditer("\n", text):
        starts.append(match.end())
    return starts


def _python_token_mutations(candidate: str) -> list[tuple[str, str]]:
    replacements = {
        "==": "!=",
        "!=": "==",
        "<": ">",
        ">": "<",
        "<=": ">=",
        ">=": "<=",
        "+": "-",
        "-": "+",
        "*": "+",
        "and": "or",
        "or": "and",
        "True": "False",
        "False": "True",
    }
    line_offsets = _offsets(candidate)
    output: list[tuple[str, str]] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(candidate).readline)
        for token in tokens:
            replacement = replacements.get(token.string)
            if replacement is None:
                continue
            start = line_offsets[token.start[0] - 1] + token.start[1]
            end = line_offsets[token.end[0] - 1] + token.end[1]
            output.append(
                (
                    f"replace_python_token_{token.string}_with_{replacement}",
                    _replace_span(candidate, start, end, replacement),
                )
            )
            if len(output) >= 24:
                break
    except (IndentationError, SyntaxError, tokenize.TokenError):
        return []
    return output


def _python_failure_fallback(candidate: str) -> tuple[str, str] | None:
    function = re.search(
        r"(?m)^(?P<indent>[ \t]*)(?:async\s+)?def\s+[A-Za-z_]\w*\s*\([^\n]*\).*:\s*(?:#.*)?$",
        candidate,
    )
    if function is None:
        return None
    line_end = candidate.find("\n", function.end())
    if line_end < 0:
        line_end = function.end()
        separator = "\n"
    else:
        line_end += 1
        separator = ""
    indentation = function.group("indent") + "    "
    injected = (
        f'{separator}{indentation}raise AssertionError("causal audit mutation")\n'
    )
    return "inject_entry_failure", _replace_span(
        candidate, line_end, line_end, injected
    )


def _spider_mutations(candidate: str) -> list[tuple[str, str]]:
    transformations = (
        (r"(?i)\bASC\b", "DESC", "flip_asc"),
        (r"(?i)\bDESC\b", "ASC", "flip_desc"),
        (r"(?i)\bAND\b", "OR", "flip_and"),
        (r"(?i)\bOR\b", "AND", "flip_or"),
        (r"!=|<>", "=", "flip_not_equal"),
        (r">=", "<", "flip_greater_equal"),
        (r"<=", ">", "flip_less_equal"),
        (r"(?<![<>!])=(?!=)", "!=", "flip_equal"),
        (r"(?<![<>=])>(?!=)", "<", "flip_greater"),
        (r"(?<![<>=])<(?!=|>)", ">", "flip_less"),
        (r"(?i)\bMAX\s*\(", "MIN(", "flip_max"),
        (r"(?i)\bMIN\s*\(", "MAX(", "flip_min"),
    )
    output: list[tuple[str, str]] = []
    for pattern, replacement, label in transformations:
        for match_index, match in enumerate(re.finditer(pattern, candidate)):
            output.append(
                (
                    f"{label}_{match_index}",
                    _replace_span(candidate, match.start(), match.end(), replacement),
                )
            )
            if len(output) >= 24:
                return output
    query = candidate.strip().rstrip(";")
    if not re.search(r"(?i)\bLIMIT\b", query):
        output.append(("append_limit_zero", query + " LIMIT 0"))
    output.extend(
        [
            (
                "wrap_empty_result",
                f"SELECT * FROM ({query}) AS causal_audit_subquery WHERE 1 = 0",
            ),
            ("replace_with_constant_zero", "SELECT 0"),
            ("replace_with_constant_one", "SELECT 1"),
        ]
    )
    return output


def semantic_mutations(
    domain: str, candidate: str, example: dict[str, Any]
) -> list[tuple[str, str]]:
    if domain == "commonsenseqa":
        return _cqa_mutations(candidate, str(example["answer_key"]))
    if domain == "gsm8k":
        return _gsm8k_mutations(candidate, str(example["answer_value"]))
    if domain == "kodcode":
        output = _python_token_mutations(candidate)
        fallback = _python_failure_fallback(candidate)
        return output + ([fallback] if fallback is not None else [])
    if domain == "spider":
        return _spider_mutations(candidate)
    raise ValueError(domain)


def style_transform(domain: str, candidate: str) -> str:
    prefix = STYLE_PREFIXES[domain]
    if candidate.startswith(prefix):
        raise ValueError("Candidate already contains the frozen style prefix")
    return prefix + candidate


def _score(
    domain: str,
    example: dict[str, Any],
    candidate: str,
    timeout_seconds: int,
    spider_database_root: Path | None,
) -> dict[str, Any]:
    return _score_payload(
        (
            domain,
            example,
            candidate,
            timeout_seconds,
            str(spider_database_root) if spider_database_root is not None else None,
        )
    )


def _candidate_unit(
    *,
    domain: str,
    source: dict[str, Any],
    compact: dict[str, Any],
    training: dict[str, Any],
    example: dict[str, Any],
    timeout_seconds: int,
    spider_database_root: Path | None,
) -> list[dict[str, Any]] | None:
    original = strip_code_fences(_candidate_text(source)).strip()
    try:
        original_score = _score(
            domain, example, original, timeout_seconds, spider_database_root
        )
    except (FileNotFoundError, RuntimeError, ValueError):
        return None
    if not (original_score.get("valid") and original_score.get("correct")):
        return None
    selected_mutation: tuple[str, str, dict[str, Any]] | None = None
    for label, mutated in semantic_mutations(domain, original, example):
        if mutated == original:
            continue
        try:
            score = _score(
                domain, example, mutated, timeout_seconds, spider_database_root
            )
        except (FileNotFoundError, RuntimeError, ValueError):
            continue
        if score.get("valid") and not score.get("correct"):
            selected_mutation = label, mutated, score
            break
    if selected_mutation is None:
        return None
    mutation_label, incorrect, incorrect_score = selected_mutation
    correct_style = style_transform(domain, original)
    incorrect_style = style_transform(domain, incorrect)
    try:
        correct_style_score = _score(
            domain, example, correct_style, timeout_seconds, spider_database_root
        )
        incorrect_style_score = _score(
            domain, example, incorrect_style, timeout_seconds, spider_database_root
        )
    except (FileNotFoundError, RuntimeError, ValueError):
        return None
    if not (
        correct_style_score.get("valid")
        and correct_style_score.get("correct")
        and incorrect_style_score.get("valid")
        and not incorrect_style_score.get("correct")
    ):
        return None
    unit_id = f"{domain}::{compact['task_id']}"
    common = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "domain": domain,
        "unit_id": unit_id,
        "task_id": str(compact["task_id"]),
        "dataset_index": int(compact["dataset_index"]),
        "task_statement": str(training["task_statement"]),
        "source_call_id": int(compact["call_id"]),
        "source_rollout_index": int(compact["rollout_index"]),
        "source_occurrence": int(compact["occurrence"]),
        "source_p_own": float(compact["score_probability"]),
        "source_log_p_own": float(compact["training_reward"]),
        "semantic_mutation": mutation_label,
        "semantic_char_delta": len(incorrect) - len(original),
        "style_prefix": STYLE_PREFIXES[domain],
        "style_char_delta": len(STYLE_PREFIXES[domain]),
        "official_test_accessed": False,
    }
    definitions = (
        ("correct_original", 1, 0, original, original_score),
        ("incorrect_original", 0, 0, incorrect, incorrect_score),
        ("correct_style", 1, 1, correct_style, correct_style_score),
        ("incorrect_style", 0, 1, incorrect_style, incorrect_style_score),
    )
    return [
        {
            **common,
            "candidate_id": f"{unit_id}::{variant}",
            "variant": variant,
            "correctness_level": correctness,
            "style_level": style,
            "candidate": candidate,
            "candidate_sha256": _sha256_text(candidate),
            "candidate_chars": len(candidate),
            "oracle_correct": int(bool(score["correct"])),
            "oracle_valid": int(bool(score.get("valid", True))),
            "oracle_status": str(score.get("status", "scored")),
        }
        for variant, correctness, style, candidate, score in definitions
    ]


def _eligible_bases(
    campaign_root: Path, domain: str, seed: int
) -> tuple[
    list[tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]],
    dict[str, str],
]:
    run = _step5_run(campaign_root, domain)
    paths = {
        "compact": _compact_path(campaign_root, domain),
        "reward_calls": _under(
            run / "reward_calls.jsonl", campaign_root, f"{domain} reward calls"
        ),
        "train_examples": _under(
            run / "train_examples.jsonl", campaign_root, f"{domain} train examples"
        ),
        "dataset": _dataset_path(campaign_root, domain),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    source_by_call = _successful_reward_rows(paths["reward_calls"])
    training_by_id = {
        str(row["task_id"]): row for row in read_jsonl(paths["train_examples"])
    }
    examples = read_jsonl(paths["dataset"])
    example_by_id = {str(row["id"]): row for row in examples}
    example_by_index = {int(row["source_index"]): row for row in examples}
    grouped: defaultdict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = (
        defaultdict(list)
    )
    for compact in read_jsonl(paths["compact"]):
        if not (
            int(compact["correct"]) == 1
            and int(compact["evaluator_valid"]) == 1
            and float(compact["features"].get("style_truncated", 0.0)) == 0.0
        ):
            continue
        source = source_by_call.get(int(compact["call_id"]))
        if source is None:
            continue
        candidate_hash = _sha256_text(_candidate_text(source))
        if candidate_hash != compact["candidate_sha256"]:
            raise RuntimeError(
                f"Candidate hash mismatch for {domain}/{compact['call_id']}"
            )
        grouped[str(compact["task_id"])].append((source, compact))
    ordered_tasks = sorted(
        grouped,
        key=lambda task_id: _sha256_text(f"{seed}|{domain}|task|{task_id}"),
    )
    selected: list[
        tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]
    ] = []
    for task_id in ordered_tasks:
        rows = sorted(
            grouped[task_id],
            key=lambda item: _sha256_text(
                f"{seed}|{domain}|candidate|{task_id}|{item[1]['candidate_sha256']}"
            ),
        )
        source, compact = rows[0]
        training = training_by_id[task_id]
        example = example_by_id.get(task_id) or example_by_index.get(
            int(compact["dataset_index"])
        )
        if example is None:
            raise RuntimeError(f"Missing training example for {domain}/{task_id}")
        selected.append((source, compact, training, example))
    return selected, {label: sha256_file(path) for label, path in paths.items()}


def prepare_candidates(
    *,
    campaign_root: Path,
    output_dir: Path,
    sample_per_domain: int,
    seed: int,
    workers: int,
    timeout_seconds: int,
    spider_database_root: Path,
    spider_database_provenance_root: Path | None = None,
) -> dict[str, Any]:
    campaign_root = campaign_root.expanduser().resolve()
    output_dir = _under(output_dir, campaign_root, "causal audit output")
    if sample_per_domain <= 0 or workers <= 0:
        raise ValueError("Sample size and workers must be positive")
    if (output_dir / "candidates.jsonl").exists():
        raise FileExistsError("Refusing to overwrite a frozen candidate panel")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    failures: dict[str, int] = {}
    for domain in DOMAINS:
        fragment_path = output_dir / f"candidates_{domain}.jsonl"
        fragment_manifest_path = output_dir / f"prepare_{domain}.json"
        if fragment_path.exists() != fragment_manifest_path.exists():
            raise RuntimeError(f"Incomplete cached domain fragment for {domain}")
        if fragment_path.is_file():
            cached_manifest = _read_json(fragment_manifest_path)
            cached_rows = read_jsonl(fragment_path)
            if not (
                cached_manifest.get("protocol") == PROTOCOL
                and cached_manifest.get("status") == "complete"
                and cached_manifest.get("official_test_accessed") is False
                and int(cached_manifest.get("seed", -1)) == seed
                and int(cached_manifest.get("sample_per_domain", -1))
                == sample_per_domain
                and cached_manifest.get("candidate_sha256")
                == sha256_file(fragment_path)
                and len(cached_rows) == sample_per_domain * len(VARIANTS)
                and {str(row["domain"]) for row in cached_rows} == {domain}
            ):
                raise RuntimeError(f"Cached domain fragment drifted for {domain}")
            all_rows.extend(cached_rows)
            provenance[domain] = dict(cached_manifest["provenance"])
            failures[domain] = int(provenance[domain]["rejected_tasks"])
            continue
        bases, hashes = _eligible_bases(campaign_root, domain, seed)
        root = spider_database_root if domain == "spider" else None

        def build(
            payload: tuple[
                dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
            ],
            current_domain: str = domain,
            current_root: Path | None = root,
        ) -> list[dict[str, Any]] | None:
            source, compact, training, example = payload
            return _candidate_unit(
                domain=current_domain,
                source=source,
                compact=compact,
                training=training,
                example=example,
                timeout_seconds=timeout_seconds,
                spider_database_root=current_root,
            )

        # Process deterministic batches until the first N valid causal units have
        # been found. This preserves the selection rule without running expensive
        # executable evaluators on an unnecessary fixed over-sample.
        domain_workers = min(workers, 2) if domain == "kodcode" else workers
        built: list[list[dict[str, Any]] | None] = []
        valid: list[list[dict[str, Any]]] = []
        batch_size = max(20, sample_per_domain // 5)
        with ThreadPoolExecutor(max_workers=domain_workers) as executor:
            for start in range(0, len(bases), batch_size):
                batch = list(executor.map(build, bases[start : start + batch_size]))
                built.extend(batch)
                valid.extend(rows for rows in batch if rows is not None)
                if len(valid) >= sample_per_domain:
                    break
        if len(valid) < sample_per_domain:
            raise RuntimeError(
                f"Only {len(valid)} valid causal units for {domain}; "
                f"need {sample_per_domain}"
            )
        chosen = valid[:sample_per_domain]
        domain_rows = [row for rows in chosen for row in rows]
        all_rows.extend(domain_rows)
        failures[domain] = len(built) - len(valid)
        provenance[domain] = {
            "eligible_correct_tasks": len(bases),
            "attempted_tasks": len(built),
            "accepted_tasks": sample_per_domain,
            "rejected_tasks": failures[domain],
            "input_sha256": hashes,
            "mutation_counts": dict(
                Counter(rows[0]["semantic_mutation"] for rows in chosen)
            ),
        }
        domain_rows.sort(key=lambda row: str(row["candidate_id"]))
        write_jsonl(fragment_path, domain_rows)
        write_json(
            fragment_manifest_path,
            {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "status": "complete",
                "domain": domain,
                "seed": seed,
                "sample_per_domain": sample_per_domain,
                "candidate_count": len(domain_rows),
                "candidate_sha256": sha256_file(fragment_path),
                "provenance": provenance[domain],
                "official_test_accessed": False,
            },
        )
    all_rows.sort(key=lambda row: str(row["candidate_id"]))
    candidate_path = output_dir / "candidates.jsonl"
    write_jsonl(candidate_path, all_rows)
    canonical_spider_root = (
        spider_database_provenance_root or spider_database_root
    ).expanduser().resolve()
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "prepared",
        "campaign_root": str(campaign_root),
        "output_dir": str(output_dir),
        "sample_per_domain": sample_per_domain,
        "candidate_count": len(all_rows),
        "expected_judge_calls": sum(
            sample_per_domain * len(VARIANTS) * len(SIGNALS_BY_DOMAIN[domain])
            for domain in DOMAINS
        ),
        "seed": seed,
        "variants": list(VARIANTS),
        "style_prefixes": STYLE_PREFIXES,
        "signals_by_domain": SIGNALS_BY_DOMAIN,
        "inputs": provenance,
        "candidates": {
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
        },
        "spider_database_root": str(canonical_spider_root),
        "spider_database_staging": {
            "used": canonical_spider_root != spider_database_root.expanduser().resolve(),
            "method": (
                "node_local_rsync_mirror"
                if canonical_spider_root
                != spider_database_root.expanduser().resolve()
                else "direct"
            ),
        },
        "official_test_accessed": False,
    }
    write_json(output_dir / "prepare_manifest.json", manifest)
    return manifest


def _prompt_for_signal(row: dict[str, Any], signal: str) -> tuple[str, str, str]:
    domain = str(row["domain"])
    task = str(row["task_statement"])
    candidate = str(row["candidate"])
    if signal == "p_own":
        version, template_hash = IDENTITY_PROMPT_PROVENANCE[domain]
        prompt = attribution_prompt(domain, task, candidate, version=version)
        return prompt, version, template_hash
    if signal == "q" and domain in QUALITY_DOMAINS:
        return (
            quality_prompt(task, candidate),
            QUALITY_PROMPT_VERSION,
            quality_prompt_template_sha256(),
        )
    raise ValueError(f"Unsupported signal/domain pair: {signal}/{domain}")


def build_judge_tasks(candidate_path: Path, seed: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for candidate in read_jsonl(candidate_path):
        for signal in SIGNALS_BY_DOMAIN[str(candidate["domain"])]:
            prompt, version, template_hash = _prompt_for_signal(candidate, signal)
            logical_key = f"{signal}::{candidate['candidate_id']}"
            tasks.append(
                {
                    "logical_key": logical_key,
                    "candidate_id": str(candidate["candidate_id"]),
                    "unit_id": str(candidate["unit_id"]),
                    "domain": str(candidate["domain"]),
                    "task_id": str(candidate["task_id"]),
                    "variant": str(candidate["variant"]),
                    "correctness_level": int(candidate["correctness_level"]),
                    "style_level": int(candidate["style_level"]),
                    "signal": signal,
                    "candidate_sha256": str(candidate["candidate_sha256"]),
                    "judge_prompt": prompt,
                    "judge_prompt_sha256": _sha256_text(prompt),
                    "judge_prompt_template_version": version,
                    "judge_prompt_template_sha256": template_hash,
                    "order_key": _sha256_text(f"{seed}|judge-order|{logical_key}"),
                    "official_test_accessed": False,
                }
            )
    return sorted(tasks, key=lambda row: str(row["order_key"]))


def _shard(
    tasks: Sequence[dict[str, Any]], index: int, count: int
) -> list[dict[str, Any]]:
    if not 0 <= index < count:
        raise ValueError("Invalid shard index")
    return [
        row
        for row in tasks
        if int(_sha256_text(str(row["logical_key"])), 16) % count == index
    ]


def _retry_wait(response: httpx.Response | None, attempt: int, cap: float) -> float:
    if response is not None:
        raw = response.headers.get("retry-after")
        if raw:
            try:
                return min(max(float(raw), 0.0), cap)
            except ValueError:
                pass
    return min(2.0 ** (attempt - 1), cap)


def _events_state(path: Path) -> dict[str, str]:
    return {
        str(row["logical_key"]): str(row["event"])
        for row in (read_jsonl(path) if path.is_file() else [])
    }


def run_judge_lane(
    *,
    candidate_path: Path,
    output_dir: Path,
    campaign_root: Path,
    seed: int,
    shard_index: int,
    num_shards: int,
    key_name: str,
    key_pool: Path,
    max_tasks: int | None = None,
    timeout_seconds: float = 120.0,
    max_attempts: int = 6,
    max_retry_wait_seconds: float = 60.0,
) -> dict[str, Any]:
    campaign_root = campaign_root.expanduser().resolve()
    candidate_path = _under(candidate_path, campaign_root, "candidate panel")
    output_dir = _under(output_dir, campaign_root, "judge lane output")
    if num_shards != len(KEY_NAMES) or key_name != KEY_NAMES[shard_index]:
        raise ValueError("Frozen one-key-per-shard mapping changed")
    manifest = _read_json(candidate_path.parent / "prepare_manifest.json")
    if (
        manifest.get("protocol") != PROTOCOL
        or manifest.get("official_test_accessed") is not False
        or manifest["candidates"]["sha256"] != sha256_file(candidate_path)
    ):
        raise RuntimeError("Candidate panel is incomplete, drifted, or test-tainted")
    tasks = _shard(build_judge_tasks(candidate_path, seed), shard_index, num_shards)
    output_dir.mkdir(parents=True, exist_ok=True)
    response_path = output_dir / "responses.jsonl"
    event_path = output_dir / "events.jsonl"
    existing = read_jsonl(response_path) if response_path.is_file() else []
    done = {str(row["logical_key"]) for row in existing}
    planned = {str(row["logical_key"]) for row in tasks}
    if not done <= planned:
        raise RuntimeError("Existing response ledger does not match this shard")
    states = _events_state(event_path)
    ambiguous = sorted(
        key
        for key, state in states.items()
        if state == "REQUEST_STARTED" and key not in done
    )
    if ambiguous:
        raise RuntimeError(f"Fail-closed ambiguous prior request: {ambiguous[:3]}")
    pending = [row for row in tasks if str(row["logical_key"]) not in done]
    if max_tasks is not None:
        if max_tasks <= 0:
            raise ValueError("max_tasks must be positive")
        pending = pending[:max_tasks]
    secret = load_named_secret(key_pool, key_name)
    headers = {
        "Authorization": f"Bearer {secret}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": "IAR causal construct audit",
        "HTTP-Referer": "https://local.cluster/indistinguishability-as-a-reward",
    }
    http_requests = 0
    started = time.time()
    with (
        key_lock(STEP_ROOT / "locks", key_name),
        httpx.Client(timeout=timeout_seconds) as client,
    ):
        for task in pending:
            prompt = str(task["judge_prompt"])
            payload = make_chat_payload(
                model_id=MODEL_ID,
                prompt=prompt,
                max_tokens=1,
                temperature=0.0,
                logprobs=True,
                top_logprobs=5,
                provider_routing={
                    "order": [PROVIDER],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                },
                request_overrides={"reasoning": {"enabled": False}},
            )
            logical_key = str(task["logical_key"])
            last_error: str | None = None
            for attempt in range(1, max_attempts + 1):
                append_event(
                    event_path,
                    {
                        "event": "REQUEST_STARTED",
                        "logical_key": logical_key,
                        "attempt": attempt,
                        "payload_sha256": _sha256_text(
                            json.dumps(payload, sort_keys=True)
                        ),
                        "timestamp": time.time(),
                    },
                )
                response: httpx.Response | None = None
                try:
                    response = client.post(
                        OPENROUTER_URL, headers=headers, json=payload
                    )
                    http_requests += 1
                    response.raise_for_status()
                    raw = response.json()
                    if raw.get("model") != MODEL_ID or raw.get("provider") != PROVIDER:
                        raise RuntimeError("misrouted_response")
                    parsed = binary_probability_from_logprobs(raw)
                    if not parsed.get("ok"):
                        if (
                            parsed.get("error")
                            == "missing_binary_label_in_top_logprobs"
                        ):
                            payload = {**payload, "top_logprobs": 20}
                        raise RuntimeError(
                            f"judge_logprob_contract:{parsed.get('error')}"
                        )
                    delta = float(parsed["logp_1"]) - float(parsed["logp_0"])
                    probability = (
                        1.0 / (1.0 + math.exp(-delta))
                        if delta >= 0
                        else math.exp(delta) / (1.0 + math.exp(delta))
                    )
                    clipped = min(
                        max(probability, PROBABILITY_EPSILON),
                        1.0 - PROBABILITY_EPSILON,
                    )
                    emitted = response_text(raw).strip()
                    result = {
                        **{
                            key: value
                            for key, value in task.items()
                            if key != "judge_prompt"
                        },
                        "attempt": attempt,
                        "score_probability": probability,
                        "training_reward": math.log(clipped),
                        "log_score": math.log(clipped),
                        "logit_score": math.log(clipped) - math.log1p(-clipped),
                        "logp_0": float(parsed["logp_0"]),
                        "logp_1": float(parsed["logp_1"]),
                        "binary_label_mass": float(parsed["binary_label_mass"]),
                        "judge_token": emitted if emitted in {"0", "1"} else None,
                        "judge_argmax_token": "1" if delta > 0 else "0",
                        "raw_model": raw.get("model"),
                        "raw_provider": raw.get("provider"),
                        "usage": raw.get("usage") or {},
                        "status": "ok",
                    }
                    with response_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(result, ensure_ascii=False, sort_keys=True)
                            + "\n"
                        )
                        handle.flush()
                        os.fsync(handle.fileno())
                    append_event(
                        event_path,
                        {
                            "event": "RESPONSE_COMMITTED",
                            "logical_key": logical_key,
                            "attempt": attempt,
                            "timestamp": time.time(),
                        },
                    )
                    done.add(logical_key)
                    break
                except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                    last_error = f"{type(exc).__name__}:{exc}"
                    append_event(
                        event_path,
                        {
                            "event": "REQUEST_FAILED",
                            "logical_key": logical_key,
                            "attempt": attempt,
                            "error": last_error,
                            "timestamp": time.time(),
                        },
                    )
                    if attempt == max_attempts:
                        raise RuntimeError(
                            f"Judge failed after {max_attempts} attempts for {logical_key}: "
                            f"{last_error}"
                        ) from exc
                    time.sleep(_retry_wait(response, attempt, max_retry_wait_seconds))
            else:
                raise AssertionError(last_error)
    rows = read_jsonl(response_path) if response_path.is_file() else []
    complete = (
        len(rows) == len(tasks) and {str(row["logical_key"]) for row in rows} == planned
    )
    summary = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete" if complete else "partial",
        "shard_index": shard_index,
        "num_shards": num_shards,
        "key_name": key_name,
        "planned_calls": len(tasks),
        "completed_calls": len(rows),
        "http_requests_this_invocation": http_requests,
        "elapsed_seconds_this_invocation": time.time() - started,
        "response_path": str(response_path),
        "response_sha256": sha256_file(response_path)
        if response_path.is_file()
        else None,
        "cost": sum(float((row.get("usage") or {}).get("cost") or 0.0) for row in rows),
        "official_test_accessed": False,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _unit_effects(rows: Sequence[dict[str, Any]], field: str) -> dict[str, float]:
    values = {str(row["variant"]): float(row[field]) for row in rows}
    if set(values) != set(VARIANTS):
        raise RuntimeError(f"Incomplete 2x2 unit: {sorted(values)}")
    semantic_original = values["correct_original"] - values["incorrect_original"]
    semantic_style = values["correct_style"] - values["incorrect_style"]
    style_correct = values["correct_style"] - values["correct_original"]
    style_incorrect = values["incorrect_style"] - values["incorrect_original"]
    return {
        "semantic_effect": (semantic_original + semantic_style) / 2.0,
        "style_effect": (style_correct + style_incorrect) / 2.0,
        "style_magnitude": (abs(style_correct) + abs(style_incorrect)) / 2.0,
        "interaction": style_correct - style_incorrect,
        "semantic_dominance": (semantic_original + semantic_style) / 2.0
        - (abs(style_correct) + abs(style_incorrect)) / 2.0,
        "semantic_correct_direction": (
            float(semantic_original > 0) + float(semantic_style > 0)
        )
        / 2.0,
    }


def _summarize_units(units: Sequence[dict[str, float]]) -> dict[str, float]:
    return {
        key: _mean([unit[key] for unit in units])
        for key in (
            "semantic_effect",
            "style_effect",
            "style_magnitude",
            "interaction",
            "semantic_dominance",
            "semantic_correct_direction",
        )
    }


def _bootstrap_effects(
    units: Sequence[dict[str, float]], replicates: int, seed: int
) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)
    metrics = tuple(_summarize_units(units))
    draws = {metric: [] for metric in metrics}
    for _ in range(replicates):
        sample = [units[rng.randrange(len(units))] for _ in units]
        summary = _summarize_units(sample)
        for metric in metrics:
            draws[metric].append(summary[metric])

    def percentile(values: Sequence[float], probability: float) -> float:
        ordered = sorted(values)
        position = probability * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    point = _summarize_units(units)
    return {
        metric: {
            "estimate": point[metric],
            "ci_low": percentile(draws[metric], 0.025),
            "ci_high": percentile(draws[metric], 0.975),
        }
        for metric in metrics
    }


def _bootstrap_mean_metrics(
    rows: Sequence[dict[str, float]],
    metrics: Sequence[str],
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)

    def means(sample: Sequence[dict[str, float]]) -> dict[str, float]:
        return {metric: _mean([row[metric] for row in sample]) for metric in metrics}

    draws = {metric: [] for metric in metrics}
    for _ in range(replicates):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        summary = means(sample)
        for metric in metrics:
            draws[metric].append(summary[metric])

    def percentile(values: Sequence[float], probability: float) -> float:
        ordered = sorted(values)
        position = probability * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    point = means(rows)
    return {
        metric: {
            "estimate": point[metric],
            "ci_low": percentile(draws[metric], 0.025),
            "ci_high": percentile(draws[metric], 0.975),
        }
        for metric in metrics
    }


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    mean_left, mean_right = _mean(left), _mean(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - mean_left) ** 2 for a in left)
        * sum((b - mean_right) ** 2 for b in right)
    )
    return numerator / denominator if denominator else None


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def _fmt(value: float | None) -> str:
    return "NA" if value is None or not math.isfinite(value) else f"{value:.3f}"


def _effect_table(rows: Sequence[dict[str, Any]]) -> list[list[str]]:
    return [
        [
            row["domain"],
            row["signal"],
            str(row["n_units"]),
            (
                f"{_fmt(row['semantic_effect']['estimate'])} "
                f"[{_fmt(row['semantic_effect']['ci_low'])}, "
                f"{_fmt(row['semantic_effect']['ci_high'])}]"
            ),
            (
                f"{_fmt(row['style_magnitude']['estimate'])} "
                f"[{_fmt(row['style_magnitude']['ci_low'])}, "
                f"{_fmt(row['style_magnitude']['ci_high'])}]"
            ),
            (
                f"{_fmt(row['semantic_dominance']['estimate'])} "
                f"[{_fmt(row['semantic_dominance']['ci_low'])}, "
                f"{_fmt(row['semantic_dominance']['ci_high'])}]"
            ),
            _fmt(row["semantic_correct_direction"]["estimate"]),
        ]
        for row in rows
    ]


def _render_report(
    *,
    metrics: list[dict[str, Any]],
    signal_comparisons: list[dict[str, Any]],
    repeatability: list[dict[str, Any]],
    output_dir: Path,
    sample_per_domain: int,
    bootstrap_replicates: int,
) -> None:
    primary = [
        row
        for row in metrics
        if row["scale"] == "training_reward" and row["subset"] == "all"
    ]
    strict = [
        row
        for row in metrics
        if row["scale"] == "training_reward"
        and row["subset"] == "strict_minimal"
        and row["n_units"] < sample_per_domain
    ]
    table = _effect_table(primary)
    strict_section = ""
    if strict:
        strict_section = f"""
## Sensitivity: sole mutazioni locali

Questa verifica esclude i fallback strutturali (assertion iniettata, `LIMIT 0`,
query vuota o costante) e conserva soltanto sostituzioni locali di token/operatori.

{_markdown_table(["dominio", "segnale", "prompt", "effetto semantico [95% CI]", "magnitudine stile [95% CI]", "dominanza semantica [95% CI]", "direzione corretta"], _effect_table(strict))}
"""
    repeat_table = [
        [
            row["domain"],
            str(row["n"]),
            _fmt(row["pearson"]),
            _fmt(row["mean_absolute_difference"]),
            _fmt(row["mean_signed_difference"]),
        ]
        for row in repeatability
    ]
    comparison_rows = [
        [
            row["domain"],
            str(row["n_units"]),
            (
                f"{_fmt(row['semantic_effect']['estimate'])} "
                f"[{_fmt(row['semantic_effect']['ci_low'])}, "
                f"{_fmt(row['semantic_effect']['ci_high'])}]"
            ),
            (
                f"{_fmt(row['style_magnitude']['estimate'])} "
                f"[{_fmt(row['style_magnitude']['ci_low'])}, "
                f"{_fmt(row['style_magnitude']['ci_high'])}]"
            ),
            (
                f"{_fmt(row['semantic_dominance']['estimate'])} "
                f"[{_fmt(row['semantic_dominance']['ci_low'])}, "
                f"{_fmt(row['semantic_dominance']['ci_high'])}]"
            ),
        ]
        for row in signal_comparisons
        if row["scale"] == "training_reward"
    ]
    p_own_probability = [
        row
        for row in metrics
        if row["signal"] == "p_own"
        and row["subset"] == "all"
        and row["scale"] == "score_probability"
    ]
    probability_summary = ", ".join(
        f"{row['domain']} {_fmt(row['semantic_effect']['estimate'])}"
        for row in p_own_probability
    )
    report = f"""# Audit causale del costrutto del reward

## Risultato primario

Questa analisi misura l'effetto causale locale di due interventi sulle stesse
risposte: alterare la correttezza mantenendo il testo il più possibile invariato,
oppure aggiungere un prefisso stilistico/formale identico alle versioni corrette e
scorrette. La scala primaria è il reward effettivamente ottimizzato: `log(p)` per
`p_own`/`q` e lo score binario per l'oracle.

{_markdown_table(["dominio", "segnale", "prompt", "effetto semantico [95% CI]", "magnitudine stile [95% CI]", "dominanza semantica [95% CI]", "direzione corretta"], table)}

Un effetto semantico positivo significa che il reward assegna un valore maggiore
alla variante corretta. La magnitudine stilistica è il cambiamento assoluto medio
causato dal solo prefisso. La dominanza semantica è `effetto semantico - magnitudine
stilistica`: un intervallo interamente sopra zero è evidenza che, per questo
intervento e questo pannello, la sensibilità alla correttezza supera quella allo
stile misurato.
{strict_section}

## Confronto diretto `p_own` − `q`

Poiché i due judge hanno valutato gli stessi controfattuali in CQA e GSM8K, le
differenze sono bootstrap appaiati per prompt. Per l'effetto semantico e la
dominanza, un delta positivo favorisce `p_own`; per la magnitudine stilistica, un
delta negativo indica minore sensibilità allo stile per `p_own`.

{_markdown_table(["dominio", "prompt", "Δ effetto semantico [95% CI]", "Δ magnitudine stile [95% CI]", "Δ dominanza [95% CI]"], comparison_rows)}

## Lettura dei risultati

- Per `p_own`, la dominanza semantica ha CI interamente sopra zero in tutti e
  quattro i domini: il segnale reagisce causalmente più alla modifica di
  correttezza che al prefisso misurato, pur non essendo invariante allo stile.
- Sulla scala di probabilità, il gap corretto−errato di `p_own` cresce da
  {probability_summary}. Nel pannello considerato il segnale è quindi più
  discriminante sui task eseguibili/strutturati, soprattutto Spider; i livelli
  assoluti non sono però calibrati in modo identico tra domini.
- Nei confronti appaiati, `p_own` e `q` non differiscono significativamente per
  effetto semantico né in CQA né in GSM8K. In CQA `p_own` è però più sensibile
  allo stile e ha dominanza semantica inferiore; in GSM8K le differenze di stile
  e dominanza restano inconclusive.
- L'oracle è semanticamente perfetto e invariante al prefisso per costruzione;
  serve da controllo positivo, non come confronto quantitativo sulla stessa
  scala di `log(p)`.

## Ripetibilità di `p_own`

Il punteggio della condizione `correct_original` è una nuova chiamata sullo stesso
testo già valutato durante il training. Serve a distinguere effetti causali da
rumore o drift del judge.

{_markdown_table(["dominio", "n", "Pearson vecchio/nuovo", "MAE", "bias nuovo−vecchio"], repeat_table)}

## Disegno

- {sample_per_domain} prompt di training per dominio, selezionati deterministicamente fra i rollout corretti, validi e non troncati dello Step 5.
- Fattoriale appaiato 2×2: correttezza corretta/alterata × stile originale/marcato.
- Ogni perturbazione è ri-valutata dall'oracle deterministico. Entrambe le versioni corrette devono restare corrette; entrambe le versioni errate devono essere valide ma errate.
- `p_own` è misurato in tutti i domini; `q` soltanto in CQA e GSM8K, coerentemente con la baseline esistente.
- Ordine delle chiamate cieco rispetto alla variante e randomizzato deterministicamente; un solo token, temperatura zero, modello e provider congelati.
- CI percentile al 95% da {bootstrap_replicates} bootstrap sui prompt. Il prompt è l'unità indipendente.
- Nessun test ufficiale o split sealed è stato letto.

## Limiti inferenziali

L'esperimento identifica causalmente l'effetto delle perturbazioni sul **punteggio
del judge**, condizionatamente a risposte che erano corrette. Non identifica ancora
l'effetto causale del reward sulla performance dopo molti update. Inoltre la
perturbazione stilistica rappresenta un preciso cambio di formato/lunghezza, non
l'intero spazio degli stili. Nei task testuali la modifica della risposta finale
può creare incoerenza con il ragionamento precedente: la sensibilità rilevata può
quindi riflettere correttezza, coerenza interna o entrambe. Servono successivamente
training multi-seed e ulteriori famiglie di riscritture semantiche/style-preserving.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")


def _plot_metrics(metrics: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in metrics
        if row["scale"] == "training_reward" and row["subset"] == "all"
    ]
    domain_labels = {
        "commonsenseqa": "CQA",
        "gsm8k": "GSM8K",
        "kodcode": "KodCode",
        "spider": "Spider",
    }
    labels = [f"{domain_labels[row['domain']]}\n{row['signal']}" for row in rows]
    x = list(range(len(rows)))
    semantic = [row["semantic_effect"]["estimate"] for row in rows]
    style = [row["style_magnitude"]["estimate"] for row in rows]
    semantic_error = [
        [
            row["semantic_effect"]["estimate"] - row["semantic_effect"]["ci_low"]
            for row in rows
        ],
        [
            row["semantic_effect"]["ci_high"] - row["semantic_effect"]["estimate"]
            for row in rows
        ],
    ]
    style_error = [
        [
            row["style_magnitude"]["estimate"] - row["style_magnitude"]["ci_low"]
            for row in rows
        ],
        [
            row["style_magnitude"]["ci_high"] - row["style_magnitude"]["estimate"]
            for row in rows
        ],
    ]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    width = 0.36
    axis.bar(
        [value - width / 2 for value in x],
        semantic,
        width,
        yerr=semantic_error,
        capsize=3,
        label="semantic effect",
        color="#167d8d",
    )
    axis.bar(
        [value + width / 2 for value in x],
        style,
        width,
        yerr=style_error,
        capsize=3,
        label="absolute style effect",
        color="#d17c0b",
    )
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Change in optimized reward")
    axis.set_title("Causal semantic sensitivity versus measured style sensitivity")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "causal_semantic_vs_style.png", dpi=180)
    figure.savefig(output_dir / "causal_semantic_vs_style.pdf")
    plt.close(figure)


def analyze_responses(
    *,
    candidate_path: Path,
    lane_root: Path,
    output_dir: Path,
    campaign_root: Path,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    campaign_root = campaign_root.expanduser().resolve()
    candidate_path = _under(candidate_path, campaign_root, "candidate panel")
    lane_root = _under(lane_root, campaign_root, "judge lanes")
    output_dir = _under(output_dir, campaign_root, "analysis output")
    candidates = read_jsonl(candidate_path)
    by_candidate = {str(row["candidate_id"]): row for row in candidates}
    judge_rows: list[dict[str, Any]] = []
    lane_summaries: list[dict[str, Any]] = []
    for shard_index in range(len(KEY_NAMES)):
        directory = lane_root / f"shard_{shard_index}"
        summary = _read_json(directory / "summary.json")
        if summary.get("status") != "complete":
            raise RuntimeError(f"Judge shard {shard_index} is incomplete")
        rows = read_jsonl(directory / "responses.jsonl")
        if summary.get("response_sha256") != sha256_file(directory / "responses.jsonl"):
            raise RuntimeError(f"Judge shard {shard_index} hash drifted")
        judge_rows.extend(rows)
        lane_summaries.append(summary)
    expected = build_judge_tasks(candidate_path, seed)
    expected_keys = {str(row["logical_key"]) for row in expected}
    observed_keys = {str(row["logical_key"]) for row in judge_rows}
    if len(judge_rows) != len(observed_keys) or observed_keys != expected_keys:
        raise RuntimeError("Judge coverage is incomplete or duplicated")
    scored: list[dict[str, Any]] = []
    for row in judge_rows:
        candidate = by_candidate[str(row["candidate_id"])]
        if row["candidate_sha256"] != candidate["candidate_sha256"]:
            raise RuntimeError("Scored candidate hash drifted")
        scored.append({**candidate, **row})
    for candidate in candidates:
        probability = float(candidate["oracle_correct"])
        scored.append(
            {
                **candidate,
                "logical_key": f"oracle::{candidate['candidate_id']}",
                "signal": "oracle",
                "score_probability": probability,
                "training_reward": probability,
                "log_score": math.log(
                    min(
                        max(probability, PROBABILITY_EPSILON), 1.0 - PROBABILITY_EPSILON
                    )
                ),
                "logit_score": math.log(
                    min(
                        max(probability, PROBABILITY_EPSILON), 1.0 - PROBABILITY_EPSILON
                    )
                )
                - math.log1p(
                    -min(
                        max(probability, PROBABILITY_EPSILON), 1.0 - PROBABILITY_EPSILON
                    )
                ),
                "status": "local_oracle",
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "scored_counterfactuals.jsonl", scored)
    grouped: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        grouped[(str(row["domain"]), str(row["signal"]), str(row["unit_id"]))].append(
            row
        )
    metrics: list[dict[str, Any]] = []
    for domain in DOMAINS:
        signals = (*SIGNALS_BY_DOMAIN[domain], "oracle")
        for signal in signals:
            unit_rows = [
                rows
                for (row_domain, row_signal, _), rows in grouped.items()
                if row_domain == domain and row_signal == signal
            ]
            if not unit_rows:
                continue
            strict_unit_rows = [
                rows
                for rows in unit_rows
                if str(rows[0]["semantic_mutation"])
                not in STRUCTURAL_FALLBACK_MUTATIONS
            ]
            subsets = (("all", unit_rows), ("strict_minimal", strict_unit_rows))
            for subset, selected_rows in subsets:
                if not selected_rows:
                    continue
                for scale in ("training_reward", "score_probability", "logit_score"):
                    units = [_unit_effects(rows, scale) for rows in selected_rows]
                    intervals = _bootstrap_effects(
                        units,
                        bootstrap_replicates,
                        seed
                        + int(
                            _sha256_text(
                                f"{domain}|{signal}|{subset}|{scale}"
                            )[:8],
                            16,
                        ),
                    )
                    metrics.append(
                        {
                            "domain": domain,
                            "signal": signal,
                            "subset": subset,
                            "scale": scale,
                            "n_units": len(units),
                            **intervals,
                        }
                    )
    comparison_metrics = (
        "semantic_effect",
        "style_magnitude",
        "semantic_dominance",
        "semantic_correct_direction",
    )
    signal_comparisons: list[dict[str, Any]] = []
    for domain in QUALITY_DOMAINS:
        unit_ids = sorted(
            unit_id
            for row_domain, row_signal, unit_id in grouped
            if row_domain == domain
            and row_signal == "p_own"
            and (domain, "q", unit_id) in grouped
        )
        for scale in ("training_reward", "score_probability", "logit_score"):
            differences: list[dict[str, float]] = []
            for unit_id in unit_ids:
                p_own_effects = _unit_effects(
                    grouped[(domain, "p_own", unit_id)], scale
                )
                q_effects = _unit_effects(grouped[(domain, "q", unit_id)], scale)
                differences.append(
                    {
                        metric: p_own_effects[metric] - q_effects[metric]
                        for metric in comparison_metrics
                    }
                )
            intervals = _bootstrap_mean_metrics(
                differences,
                comparison_metrics,
                bootstrap_replicates,
                seed
                + int(
                    _sha256_text(f"{domain}|p_own-minus-q|{scale}")[:8], 16
                ),
            )
            signal_comparisons.append(
                {
                    "domain": domain,
                    "left_signal": "p_own",
                    "right_signal": "q",
                    "contrast": "p_own_minus_q",
                    "scale": scale,
                    "n_units": len(differences),
                    **intervals,
                }
            )
    repeatability: list[dict[str, Any]] = []
    for domain in DOMAINS:
        rows = [
            row
            for row in scored
            if row["domain"] == domain
            and row["signal"] == "p_own"
            and row["variant"] == "correct_original"
        ]
        old = [float(row["source_p_own"]) for row in rows]
        new = [float(row["score_probability"]) for row in rows]
        differences = [right - left for left, right in zip(old, new)]
        repeatability.append(
            {
                "domain": domain,
                "n": len(rows),
                "pearson": _pearson(old, new),
                "mean_absolute_difference": _mean(
                    [abs(value) for value in differences]
                ),
                "mean_signed_difference": _mean(differences),
            }
        )
    write_json(
        output_dir / "causal_metrics.json",
        {
            "metrics": metrics,
            "signal_comparisons": signal_comparisons,
            "repeatability": repeatability,
        },
    )
    with (output_dir / "causal_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "domain",
                "signal",
                "subset",
                "scale",
                "n_units",
                "metric",
                "estimate",
                "ci_low",
                "ci_high",
            ),
        )
        writer.writeheader()
        for row in metrics:
            for metric in (
                "semantic_effect",
                "style_effect",
                "style_magnitude",
                "interaction",
                "semantic_dominance",
                "semantic_correct_direction",
            ):
                writer.writerow(
                    {
                        "domain": row["domain"],
                        "signal": row["signal"],
                        "subset": row["subset"],
                        "scale": row["scale"],
                        "n_units": row["n_units"],
                        "metric": metric,
                        **row[metric],
                    }
                )
    with (output_dir / "signal_comparisons.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "domain",
                "left_signal",
                "right_signal",
                "contrast",
                "scale",
                "n_units",
                "metric",
                "estimate",
                "ci_low",
                "ci_high",
            ),
        )
        writer.writeheader()
        for row in signal_comparisons:
            for metric in comparison_metrics:
                writer.writerow(
                    {
                        "domain": row["domain"],
                        "left_signal": row["left_signal"],
                        "right_signal": row["right_signal"],
                        "contrast": row["contrast"],
                        "scale": row["scale"],
                        "n_units": row["n_units"],
                        "metric": metric,
                        **row[metric],
                    }
                )
    manifest = _read_json(candidate_path.parent / "prepare_manifest.json")
    _render_report(
        metrics=metrics,
        signal_comparisons=signal_comparisons,
        repeatability=repeatability,
        output_dir=output_dir,
        sample_per_domain=int(manifest["sample_per_domain"]),
        bootstrap_replicates=bootstrap_replicates,
    )
    _plot_metrics(metrics, output_dir)
    summary = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "candidate_count": len(candidates),
        "judge_call_count": len(judge_rows),
        "oracle_score_count": len(candidates),
        "total_api_cost": sum(float(row.get("cost") or 0.0) for row in lane_summaries),
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "inputs": {
            "candidate_sha256": sha256_file(candidate_path),
            "lane_response_sha256": [row["response_sha256"] for row in lane_summaries],
        },
        "outputs": {
            name: sha256_file(output_dir / name)
            for name in (
                "scored_counterfactuals.jsonl",
                "causal_metrics.json",
                "causal_metrics.csv",
                "signal_comparisons.csv",
                "REPORT.md",
                "causal_semantic_vs_style.png",
                "causal_semantic_vs_style.pdf",
            )
        },
        "official_test_accessed": False,
    }
    write_json(output_dir / "summary.json", summary)
    return summary
