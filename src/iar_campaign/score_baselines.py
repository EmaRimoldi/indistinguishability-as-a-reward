from __future__ import annotations

import concurrent.futures
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from iar_campaign.generate import read_jsonl
from iar_campaign.reward_pilot import (
    _score_kodcode,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)

CAMPAIGN_ROOT = Path(
    "/path/to/runs"
)
DOMAINS = ("commonsenseqa", "gsm8k", "kodcode", "spider")
VARIANTS = ("a0", "prompt_imitation")
ACTOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ACTOR_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SPIDER_EXECUTION_LIMITS = {
    "example_timeout_seconds": 60.0,
    "query_timeout_seconds": 20.0,
    "sqlite_progress_ops": 1_000,
    "max_result_rows": 100_000,
    "max_result_bytes": 16 * 1024 * 1024,
}


def _under(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root.expanduser().resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escaped its authorized root: {resolved}") from exc
    return resolved


def _load_inputs(
    *,
    domain: str,
    variant: str,
    generation_dir: Path,
    validation_file: Path,
    campaign_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if domain not in DOMAINS:
        raise ValueError(f"Unsupported domain: {domain}; MBPP is forbidden")
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported baseline variant: {variant}")
    generation_dir = _under(generation_dir, campaign_root, "generation directory")
    validation_file = _under(validation_file, campaign_root, "validation split")
    if validation_file.name not in {"validation.jsonl", "validation300.jsonl"}:
        raise ValueError("Baseline scoring is validation-only")
    if any(part.lower() in {"test", "sealed", "final_test"} for part in validation_file.parts):
        raise ValueError("Official or sealed data cannot enter baseline scoring")

    summary_path = generation_dir / "summary.json"
    candidates_path = generation_dir / "candidates.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "complete"
        or summary.get("domain") != domain
        or summary.get("variants") != [variant]
        or summary.get("model_id") != ACTOR_MODEL
        or summary.get("model_revision") != ACTOR_REVISION
        or summary.get("official_test_accessed") is not False
        or int(summary.get("candidate_count", -1)) != 300
        or int(summary.get("truncated_count", -1)) != 0
    ):
        raise RuntimeError("Generation summary violates the frozen Step 5 baseline protocol")
    if summary.get("candidate_file_sha256") != sha256_file(candidates_path):
        raise RuntimeError("Generated candidate hash mismatch")
    if summary.get("split_file_sha256") != sha256_file(validation_file):
        raise RuntimeError("Generation and scoring validation splits differ")

    examples = read_jsonl(validation_file)
    candidates = read_jsonl(candidates_path)
    example_by_id = {str(row["id"]): row for row in examples}
    candidate_by_id = {str(row["id"]): row for row in candidates}
    if (
        len(examples) != 300
        or len(candidates) != 300
        or len(example_by_id) != 300
        or len(candidate_by_id) != 300
        or set(example_by_id) != set(candidate_by_id)
    ):
        raise RuntimeError("Baseline candidates do not exactly cover validation300")
    for row_id, candidate in candidate_by_id.items():
        example = example_by_id[row_id]
        if (
            candidate.get("domain") != domain
            or candidate.get("variant") != variant
            or int(candidate["source_index"]) != int(example["source_index"])
            or candidate.get("model_id") != ACTOR_MODEL
            or candidate.get("model_revision") != ACTOR_REVISION
        ):
            raise RuntimeError(f"Candidate provenance mismatch for {row_id}")
    ordered_candidates = [candidate_by_id[str(row["id"])] for row in examples]
    return examples, ordered_candidates, summary


def _score_simple(
    domain: str,
    examples: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> None:
    if domain == "commonsenseqa":
        from iar_cqa.scoring import score_response

        gold_field = "answer_key"
    elif domain == "gsm8k":
        from iar_gsm8k.scoring import score_response

        gold_field = "answer_value"
    else:
        raise ValueError(f"No simple scorer for {domain}")
    for example, candidate in zip(examples, candidates, strict=True):
        candidate["score"] = score_response(
            str(candidate["response"]), str(example[gold_field])
        )


def _score_kodcode_rows(
    examples: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    scoring_protocol_path: Path,
    score_workers: int,
    timeout_seconds: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol = json.loads(scoring_protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("decision") != "GO"
        or int(protocol.get("scored_n", -1)) != 299
        or protocol.get("validation_file_sha256") is None
    ):
        raise RuntimeError("KodCode scoring protocol is not the frozen 299/300 GO protocol")
    scored_ids = {str(value) for value in protocol["scored_ids"]}
    example_by_id = {str(row["id"]): row for row in examples}
    if len(scored_ids) != 299 or not scored_ids <= set(example_by_id):
        raise RuntimeError("KodCode scoring protocol contains invalid IDs")
    scored_rows = [row for row in candidates if str(row["id"]) in scored_ids]
    payloads = [
        (
            str(row["response"]),
            str(example_by_id[str(row["id"])]["hidden_tests"]),
            timeout_seconds,
        )
        for row in scored_rows
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=score_workers) as pool:
        scores = list(pool.map(_score_kodcode, payloads))
    for row, score in zip(scored_rows, scores, strict=True):
        row["score"] = score
    for row in candidates:
        if str(row["id"]) not in scored_ids:
            row["score"] = {
                "correct": False,
                "valid": False,
                "excluded_reference_invalid": True,
                "reason": "reference_solution_failed_frozen_hidden_tests",
            }
    return scored_rows, {
        "path": str(scoring_protocol_path),
        "sha256": sha256_file(scoring_protocol_path),
    }


def _spider_limits():
    from iar_spider.evaluate_validation import ExecutionLimits

    return ExecutionLimits(**SPIDER_EXECUTION_LIMITS)


def _score_spider_one(
    payload: tuple[dict[str, Any], dict[str, Any], Path]
) -> dict[str, Any]:
    example, candidate, database_root = payload
    from iar_spider.evaluate_validation import (
        database_suite,
        execute_example,
        extract_sql,
        normalize_query,
    )

    try:
        predicted = normalize_query(
            extract_sql(str(candidate["response"])), keep_distinct=False
        )
    except ValueError as exc:
        score = {
            "correct": False,
            "valid": False,
            "status": "invalid_or_blocked_sql",
            "error": str(exc),
        }
        predicted = None
    else:
        gold = normalize_query(str(example["reference_query"]), keep_distinct=False)
        databases = database_suite(
            str(example["db_id"]), database_root=database_root, metric_mode="execution"
        )
        score = execute_example(
            predicted,
            gold,
            databases,
            limits=_spider_limits(),
        )
        score["valid"] = score["status"] != "invalid_or_blocked_sql"
    return {"predicted_sql": predicted, "score": score}


def _audit_spider_reference_one(
    payload: tuple[dict[str, Any], Path]
) -> dict[str, Any]:
    example, database_root = payload
    from iar_spider.evaluate_validation import (
        GoldExecutionError,
        database_suite,
        execute_example,
        normalize_query,
    )

    row_id = str(example["id"])
    try:
        gold = normalize_query(str(example["reference_query"]), keep_distinct=False)
        databases = database_suite(
            str(example["db_id"]), database_root=database_root, metric_mode="execution"
        )
        score = execute_example(gold, gold, databases, limits=_spider_limits())
        if not bool(score.get("correct")):
            raise RuntimeError(f"Reference self-comparison was not correct: {score}")
        return {"id": row_id, "db_id": str(example["db_id"]), "valid": True, "score": score}
    except (GoldExecutionError, FileNotFoundError, RuntimeError, ValueError) as exc:
        return {
            "id": row_id,
            "db_id": str(example["db_id"]),
            "valid": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _validate_spider_inputs(
    *,
    examples: list[dict[str, Any]],
    split_manifest_path: Path,
    validation_file: Path,
    database_root: Path,
) -> tuple[Path, list[str]]:
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    split = (manifest.get("splits") or {}).get("validation300") or {}
    heldout = manifest.get("heldout_policy") or {}
    assets = manifest.get("schema_assets") or {}
    if (
        manifest.get("protocol") != "spider-1.0-campaign-seed23-v1"
        or split.get("file") != "validation300.jsonl"
        or int(split.get("n", -1)) != 300
        or split.get("file_sha256") != sha256_file(validation_file)
        or heldout.get("accessed_during_preparation") is not False
        or heldout.get("materialized_during_preparation") is not False
    ):
        raise RuntimeError("Spider validation manifest violates the frozen campaign protocol")
    canonical_hashes = assets.get("database_sha256") or {}
    database_root = database_root.resolve()
    used_db_ids = sorted({str(row["db_id"]) for row in examples})
    for db_id in used_db_ids:
        database = database_root / db_id / f"{db_id}.sqlite"
        if not database.is_file() or canonical_hashes.get(db_id) != sha256_file(database):
            raise RuntimeError(f"Spider database hash mismatch: {db_id}")
    return database_root, used_db_ids


def audit_spider_references(
    *,
    validation_file: Path,
    split_manifest_path: Path,
    database_root: Path,
    output_path: Path,
    score_workers: int = 4,
    campaign_root: Path = CAMPAIGN_ROOT,
) -> dict[str, Any]:
    validation_file = _under(validation_file, campaign_root, "Spider validation split")
    split_manifest_path = _under(
        split_manifest_path, campaign_root, "Spider split manifest"
    )
    output_path = _under(output_path, campaign_root, "Spider reference audit")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite Spider reference audit: {output_path}")
    examples = read_jsonl(validation_file)
    if len(examples) != 300 or len({str(row["id"]) for row in examples}) != 300:
        raise RuntimeError("Spider reference audit requires exactly validation300")
    database_root, used_db_ids = _validate_spider_inputs(
        examples=examples,
        split_manifest_path=split_manifest_path,
        validation_file=validation_file,
        database_root=database_root,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=score_workers) as pool:
        rows = list(
            pool.map(
                _audit_spider_reference_one,
                [(example, database_root) for example in examples],
            )
        )
    valid_ids = [str(row["id"]) for row in rows if bool(row["valid"])]
    decision = "GO" if len(valid_ids) == 300 else "NO_GO"
    payload = {
        "schema_version": 1,
        "protocol": "spider-validation300-reference-self-execution-v1",
        "decision": decision,
        "completed_at": utc_now(),
        "validation_file": str(validation_file),
        "validation_file_sha256": sha256_file(validation_file),
        "split_manifest": str(split_manifest_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "database_root": str(database_root),
        "database_count": len(used_db_ids),
        "execution_limits": SPIDER_EXECUTION_LIMITS,
        "score_workers": score_workers,
        "n": len(rows),
        "scored_n": len(valid_ids),
        "scored_ids": valid_ids,
        "failures": [row for row in rows if not bool(row["valid"])],
        "rows": rows,
        "official_test_accessed": False,
    }
    write_json(output_path, payload)
    return payload


def _score_spider_rows(
    examples: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    split_manifest_path: Path,
    scoring_protocol_path: Path,
    validation_file: Path,
    database_root: Path,
    score_workers: int,
) -> dict[str, Any]:
    database_root, used_db_ids = _validate_spider_inputs(
        examples=examples,
        split_manifest_path=split_manifest_path,
        validation_file=validation_file,
        database_root=database_root,
    )
    scoring_protocol = json.loads(scoring_protocol_path.read_text(encoding="utf-8"))
    scored_ids = {str(value) for value in scoring_protocol.get("scored_ids", [])}
    if (
        scoring_protocol.get("protocol")
        != "spider-validation300-reference-self-execution-v1"
        or scoring_protocol.get("decision") != "GO"
        or scoring_protocol.get("validation_file_sha256") != sha256_file(validation_file)
        or scoring_protocol.get("execution_limits") != SPIDER_EXECUTION_LIMITS
        or int(scoring_protocol.get("scored_n", -1)) != 300
        or scored_ids != {str(row["id"]) for row in examples}
    ):
        raise RuntimeError("Spider reference scoring protocol is missing, drifted, or not GO")
    payloads = [
        (example, candidate, database_root)
        for example, candidate in zip(examples, candidates, strict=True)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=score_workers) as pool:
        scores = list(pool.map(_score_spider_one, payloads))
    for candidate, result in zip(candidates, scores, strict=True):
        candidate.update(result)
    return {
        "split_manifest": str(split_manifest_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "database_root": str(database_root),
        "database_count": len(used_db_ids),
        "metric": "official_spider_single_database_execution_accuracy",
        "execution_limits": SPIDER_EXECUTION_LIMITS,
        "reference_protocol": str(scoring_protocol_path),
        "reference_protocol_sha256": sha256_file(scoring_protocol_path),
    }


def score_baseline(
    *,
    domain: str,
    variant: str,
    generation_dir: Path,
    validation_file: Path,
    output_dir: Path,
    campaign_root: Path = CAMPAIGN_ROOT,
    kodcode_scoring_protocol: Path | None = None,
    spider_split_manifest: Path | None = None,
    spider_scoring_protocol: Path | None = None,
    spider_database_root: Path | None = None,
    score_workers: int = 8,
    coding_timeout_seconds: int = 60,
) -> dict[str, Any]:
    output_dir = _under(output_dir, campaign_root, "scoring output")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite baseline score: {output_dir}")
    output_dir.mkdir(parents=True)
    try:
        examples, candidates, generation_summary = _load_inputs(
            domain=domain,
            variant=variant,
            generation_dir=generation_dir,
            validation_file=validation_file,
            campaign_root=campaign_root,
        )
        scored_rows = candidates
        protocol: dict[str, Any] | None = None
        if domain in {"commonsenseqa", "gsm8k"}:
            _score_simple(domain, examples, candidates)
        elif domain == "kodcode":
            if kodcode_scoring_protocol is None:
                raise ValueError("KodCode requires its frozen scoring protocol")
            protocol_path = _under(
                kodcode_scoring_protocol, campaign_root, "KodCode scoring protocol"
            )
            protocol_value = json.loads(protocol_path.read_text(encoding="utf-8"))
            if protocol_value.get("validation_file_sha256") != sha256_file(validation_file):
                raise RuntimeError("KodCode scoring protocol targets another validation split")
            scored_rows, protocol = _score_kodcode_rows(
                examples,
                candidates,
                scoring_protocol_path=protocol_path,
                score_workers=score_workers,
                timeout_seconds=coding_timeout_seconds,
            )
        else:
            if (
                spider_split_manifest is None
                or spider_scoring_protocol is None
                or spider_database_root is None
            ):
                raise ValueError(
                    "Spider requires its split manifest, reference protocol and database root"
                )
            protocol = _score_spider_rows(
                examples,
                candidates,
                split_manifest_path=spider_split_manifest.resolve(),
                scoring_protocol_path=_under(
                    spider_scoring_protocol, campaign_root, "Spider scoring protocol"
                ),
                validation_file=validation_file.resolve(),
                database_root=spider_database_root,
                score_workers=score_workers,
            )
        if not scored_rows:
            raise RuntimeError("No validation rows were scored")
        write_jsonl(output_dir / "candidates.jsonl", candidates)
        correct = sum(bool(row["score"]["correct"]) for row in scored_rows)
        valid = sum(bool(row["score"].get("valid", True)) for row in scored_rows)
        summary = {
            "schema_version": 1,
            "status": "complete",
            "completed_at": utc_now(),
            "domain": domain,
            "variant": variant,
            "n": len(candidates),
            "scored_n": len(scored_rows),
            "correct": correct,
            "valid": valid,
            "reference_invalid_excluded": len(candidates) - len(scored_rows),
            "task_utility": statistics.fmean(
                bool(row["score"]["correct"]) for row in scored_rows
            ),
            "status_counts": dict(
                sorted(Counter(str(row["score"].get("status", "scored")) for row in scored_rows).items())
            ),
            "candidate_file_sha256": sha256_file(output_dir / "candidates.jsonl"),
            "validation_file": str(validation_file.resolve()),
            "validation_file_sha256": sha256_file(validation_file.resolve()),
            "generation_dir": str(generation_dir.resolve()),
            "generation_candidate_file_sha256": generation_summary["candidate_file_sha256"],
            "scoring_protocol": protocol,
            "selection_role": "validation_task_utility",
            "official_test_accessed": False,
        }
        write_json(output_dir / "summary.json", summary)
        return summary
    except Exception as exc:
        write_json(
            output_dir / "failure.json",
            {
                "schema_version": 1,
                "status": "failed",
                "failed_at": utc_now(),
                "domain": domain,
                "variant": variant,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "official_test_accessed": False,
            },
        )
        raise


def collect_validation_baselines(
    *,
    step_root: Path,
    output_path: Path,
    campaign_root: Path = CAMPAIGN_ROOT,
) -> dict[str, Any]:
    step_root = _under(step_root, campaign_root, "Step 5 root")
    output_path = _under(output_path, step_root, "baseline registry")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite baseline registry: {output_path}")
    baselines: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        baselines[domain] = {}
        for variant in VARIANTS:
            run = step_root / "domains" / domain / variant / "runs" / "validation_attempt002"
            summaries = sorted(
                run.glob("evaluations/validation/score_attempt*/summary.json")
            )
            complete: list[tuple[Path, dict[str, Any]]] = []
            for path in summaries:
                value = json.loads(path.read_text(encoding="utf-8"))
                if value.get("status") == "complete":
                    complete.append((path, value))
            if len(complete) != 1:
                raise RuntimeError(
                    f"Expected exactly one successful score for {domain}/{variant}, "
                    f"found {len(complete)}"
                )
            path, value = complete[0]
            if (
                value.get("domain") != domain
                or value.get("variant") != variant
                or int(value.get("n", -1)) != 300
                or value.get("official_test_accessed") is not False
            ):
                raise RuntimeError(f"Invalid baseline score summary: {path}")
            candidates = path.with_name("candidates.jsonl")
            if value.get("candidate_file_sha256") != sha256_file(candidates):
                raise RuntimeError(f"Baseline candidate hash mismatch: {path}")
            baselines[domain][variant] = {
                "summary": str(path),
                "summary_sha256": sha256_file(path),
                "candidates": str(candidates),
                "candidates_sha256": sha256_file(candidates),
                "task_utility": float(value["task_utility"]),
                "correct": int(value["correct"]),
                "scored_n": int(value["scored_n"]),
                "valid": int(value["valid"]),
            }
    result = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "protocol": "campaign-step5-validation-baseline-registry-v1",
        "model_id": ACTOR_MODEL,
        "model_revision": ACTOR_REVISION,
        "baselines": baselines,
        "selection_role": "validation_task_utility",
        "official_test_accessed": False,
        "mbpp": "FORBIDDEN",
    }
    write_json(output_path, result)
    return result
