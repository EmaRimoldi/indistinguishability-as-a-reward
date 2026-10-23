from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from iar_campaign.four_domain_screen import (
    CHECKPOINTS,
    STEP_ROOT,
    _read_json,
    _under,
    generate_validation_attempt,
    score_validation_attempt,
    validate_screen_config,
)
from iar_campaign.reward_pilot import sha256_file
from iar_grpo.config import load_grpo_config

VALIDATION_CHECKPOINTS = CHECKPOINTS[1:]


def _run_root(config_path: Path, run_id: str) -> tuple[Any, str, Path]:
    config = load_grpo_config(config_path)
    domain, _, _, _ = validate_screen_config(config)
    run_root = _under(
        config.resolve(config.raw["paths"]["experiments"]) / run_id,
        STEP_ROOT,
        "run root",
    )
    return config, domain, run_root


def _complete_generation_attempts(
    *, step_root: Path, domain: str, run_id: str, step: int
) -> list[tuple[int, Path, dict[str, Any]]]:
    complete: list[tuple[int, Path, dict[str, Any]]] = []
    for path in sorted(step_root.glob("generation_attempt*/summary.json")):
        value = _read_json(path)
        if value.get("status") != "complete":
            continue
        attempt_text = path.parent.name.removeprefix("generation_attempt")
        if not attempt_text.isdigit():
            raise RuntimeError(f"Malformed generation attempt directory: {path.parent}")
        attempt = int(attempt_text)
        candidates = path.parent / "candidates.jsonl"
        if (
            value.get("domain") != domain
            or value.get("run_id") != run_id
            or int(value.get("step", -1)) != step
            or int(value.get("attempt", -1)) != attempt
            or int(value.get("candidate_count", -1)) != 300
            or value.get("official_test_accessed") is not False
            or not candidates.is_file()
            or value.get("candidate_file_sha256") != sha256_file(candidates)
        ):
            raise RuntimeError(
                f"Complete generation summary is corrupt or drifted: {path}"
            )
        complete.append((attempt, path, value))
    return complete


def _complete_score_attempts(
    *, step_root: Path, domain: str, run_id: str, step: int
) -> list[tuple[int, Path, dict[str, Any]]]:
    complete: list[tuple[int, Path, dict[str, Any]]] = []
    for path in sorted(step_root.glob("score_attempt*/summary.json")):
        value = _read_json(path)
        if value.get("status") != "complete":
            continue
        attempt_text = path.parent.name.removeprefix("score_attempt")
        if not attempt_text.isdigit():
            raise RuntimeError(f"Malformed score attempt directory: {path.parent}")
        attempt = int(attempt_text)
        candidates = path.parent / "candidates.jsonl"
        utility = float(value.get("task_utility", float("nan")))
        generation_summary = Path(str(value.get("generation_summary", "")))
        if (
            value.get("domain") != domain
            or value.get("run_id") != run_id
            or int(value.get("step", -1)) != step
            or int(value.get("scoring_attempt", -1)) != attempt
            or int(value.get("n", -1)) != 300
            or value.get("selection_role") != "validation_task_utility"
            or value.get("official_test_accessed") is not False
            or not math.isfinite(utility)
            or not candidates.is_file()
            or value.get("candidate_file_sha256") != sha256_file(candidates)
            or not generation_summary.is_file()
            or value.get("generation_summary_sha256") != sha256_file(generation_summary)
        ):
            raise RuntimeError(f"Complete score summary is corrupt or drifted: {path}")
        complete.append((attempt, path, value))
    return complete


def generate_domain_checkpoints(
    *, config_path: Path, run_id: str, attempt: int
) -> dict[str, Any]:
    if attempt <= 0:
        raise ValueError("Generation attempt must be positive")
    _, domain, run_root = _run_root(config_path, run_id)
    rows: list[dict[str, Any]] = []
    for step in VALIDATION_CHECKPOINTS:
        step_root = run_root / "evaluations" / "validation" / f"step_{step:04d}"
        complete = _complete_generation_attempts(
            step_root=step_root, domain=domain, run_id=run_id, step=step
        )
        if len(complete) > 1:
            raise RuntimeError(
                f"Multiple complete generations for {domain}/step{step}: "
                f"{[str(row[1]) for row in complete]}"
            )
        if complete:
            rows.append(
                {
                    "step": step,
                    "attempt": complete[0][0],
                    "status": "reused_complete",
                    "summary": str(complete[0][1]),
                }
            )
            continue
        destination = step_root / f"generation_attempt{attempt:03d}"
        if destination.exists():
            raise RuntimeError(
                f"Generation attempt is partial or failed: {destination}; "
                "rerun with a new validation launch attempt"
            )
        result = generate_validation_attempt(
            config_path=config_path, run_id=run_id, step=step, attempt=attempt
        )
        rows.append(
            {
                "step": step,
                "attempt": attempt,
                "status": "generated",
                "summary": str(destination / "summary.json"),
                "candidate_file_sha256": result["candidate_file_sha256"],
            }
        )
    return {
        "schema_version": 1,
        "status": "complete",
        "stage": "step5_validation_generation_domain_batch",
        "domain": domain,
        "run_id": run_id,
        "checkpoints": list(VALIDATION_CHECKPOINTS),
        "items": rows,
        "official_test_accessed": False,
        "mbpp": "FORBIDDEN",
    }


def score_domain_checkpoints(
    *, config_path: Path, run_id: str, attempt: int, score_workers: int
) -> dict[str, Any]:
    if attempt <= 0 or score_workers <= 0:
        raise ValueError("Scoring attempt and worker count must be positive")
    _, domain, run_root = _run_root(config_path, run_id)
    rows: list[dict[str, Any]] = []
    for step in VALIDATION_CHECKPOINTS:
        step_root = run_root / "evaluations" / "validation" / f"step_{step:04d}"
        complete_scores = _complete_score_attempts(
            step_root=step_root, domain=domain, run_id=run_id, step=step
        )
        if len(complete_scores) > 1:
            raise RuntimeError(
                f"Multiple complete scores for {domain}/step{step}: "
                f"{[str(row[1]) for row in complete_scores]}"
            )
        if complete_scores:
            rows.append(
                {
                    "step": step,
                    "attempt": complete_scores[0][0],
                    "status": "reused_complete",
                    "summary": str(complete_scores[0][1]),
                    "task_utility": float(complete_scores[0][2]["task_utility"]),
                }
            )
            continue
        generations = _complete_generation_attempts(
            step_root=step_root, domain=domain, run_id=run_id, step=step
        )
        if len(generations) != 1:
            raise RuntimeError(
                f"Expected one complete generation for {domain}/step{step}; "
                f"found {len(generations)}"
            )
        destination = step_root / f"score_attempt{attempt:03d}"
        if destination.exists():
            raise RuntimeError(
                f"Scoring attempt is partial or failed: {destination}; "
                "rerun with a new validation launch attempt"
            )
        result = score_validation_attempt(
            config_path=config_path,
            run_id=run_id,
            step=step,
            generation_attempt=generations[0][0],
            scoring_attempt=attempt,
            score_workers=score_workers,
        )
        rows.append(
            {
                "step": step,
                "attempt": attempt,
                "generation_attempt": generations[0][0],
                "status": "scored",
                "summary": str(destination / "summary.json"),
                "task_utility": float(result["task_utility"]),
            }
        )
    return {
        "schema_version": 1,
        "status": "complete",
        "stage": "step5_validation_scoring_domain_batch",
        "domain": domain,
        "run_id": run_id,
        "checkpoints": list(VALIDATION_CHECKPOINTS),
        "items": rows,
        "score_workers": score_workers,
        "scoring_backend": "local_deterministic_task_utility",
        "openrouter_calls": 0,
        "official_test_accessed": False,
        "mbpp": "FORBIDDEN",
    }
