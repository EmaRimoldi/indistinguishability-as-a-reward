from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REWARDS = ("log_p_own", "p_own", "logit_p_own", "binary_own")
DOMAINS = ("MBPP", "CommonsenseQA")
EXPECTED_BATCH_ID = "reward-transform-pilot-v10-20260907"
FORBIDDEN_EVALUATION_SPLITS = ("test", "final_test", "pilot_subset")


@dataclass(frozen=True)
class CallAudit:
    successful_calls: int
    http_rows: int
    retries: int
    cost_usd: float


@dataclass(frozen=True)
class PilotRun:
    domain: str
    reward: str
    run_id: str
    run_dir: Path
    ids: tuple[str, ...]
    selected_step: int
    baseline_correct: tuple[int, ...]
    selected_correct: tuple[int, ...]
    p_own_deltas: tuple[float, ...]
    training_calls: CallAudit
    validation_calls: CallAudit
    source_config_sha256: str
    source_hashes: dict[str, str]

    @property
    def task_deltas(self) -> tuple[int, ...]:
        return tuple(
            selected - baseline
            for baseline, selected in zip(
                self.baseline_correct, self.selected_correct, strict=True
            )
        )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a YAML mapping: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"Expected JSON objects in {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_under(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root.expanduser().resolve())
    except ValueError as exc:
        raise ValueError(f"{label} is outside the authorized root: {resolved}") from exc
    return resolved


def _authorized_file(path: Path, root: Path, label: str) -> Path:
    resolved = _ensure_under(path, root, label)
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def _domain(entry: dict[str, Any]) -> str:
    module = str(entry.get("train_module"))
    if module == "iar_grpo.train":
        return "MBPP"
    if module == "iar_cqa.train":
        return "CommonsenseQA"
    raise ValueError(f"Unsupported pilot train module: {module!r}")


def _row_id(row: dict[str, Any]) -> str:
    for key in ("task_id", "id"):
        if key in row:
            return str(row[key])
    raise ValueError("Validation row has no task_id or id")


def _correct(row: dict[str, Any]) -> int:
    unit_tests = row.get("unit_tests")
    if isinstance(unit_tests, dict) and "passed" in unit_tests:
        return int(bool(unit_tests["passed"]))
    score = row.get("score")
    if isinstance(score, dict) and "correct" in score:
        return int(bool(score["correct"]))
    raise ValueError("Validation row has no supported correctness field")


def _mean(values: Iterable[float | int]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("Cannot average an empty collection")
    return statistics.fmean(materialized)


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot take a percentile of an empty collection")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def audit_call_log(path: Path, *, expected_calls: int) -> CallAudit:
    rows = _read_jsonl(path)
    by_call: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        call_id = int(row["call_id"])
        by_call.setdefault(call_id, []).append(row)
    expected_ids = set(range(1, expected_calls + 1))
    if set(by_call) != expected_ids:
        raise RuntimeError(
            f"Incomplete judge log {path}: expected call IDs 1..{expected_calls}, "
            f"found {sorted(by_call)}"
        )
    successes: list[dict[str, Any]] = []
    for call_id, call_rows in by_call.items():
        ok = [row for row in call_rows if row.get("status") == "ok"]
        if len(ok) != 1:
            raise RuntimeError(
                f"Logical judge call {call_id} in {path} has {len(ok)} successful responses"
            )
        successes.extend(ok)
    if len(successes) != expected_calls or len(by_call) != expected_calls:
        raise RuntimeError(
            f"Incomplete judge log {path}: expected {expected_calls}, "
            f"found {len(successes)} complete logical calls"
        )
    cost = 0.0
    for row in successes:
        usage = row.get("usage")
        if not isinstance(usage, dict) or usage.get("cost") is None:
            raise ValueError(f"Missing successful-call cost in {path}")
        value = float(usage["cost"])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid cost in {path}: {value!r}")
        cost += value
    return CallAudit(
        successful_calls=len(successes),
        http_rows=len(rows),
        retries=len(rows) - len(successes),
        cost_usd=cost,
    )


def _load_validation_step(path: Path, *, expected_rows: int) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = _row_id(row)
        if row_id in by_id:
            raise ValueError(f"Duplicate validation ID {row_id!r} in {path}")
        p_own = float(row["p_own"])
        if not math.isfinite(p_own) or not 0.0 <= p_own <= 1.0:
            raise ValueError(f"Invalid p_own={p_own!r} in {path}")
        _correct(row)
        by_id[row_id] = row
    if len(by_id) != expected_rows:
        raise RuntimeError(
            f"Validation row count mismatch in {path}: expected {expected_rows}, found {len(by_id)}"
        )
    return by_id


def _task_selected_step(rows: dict[int, dict[str, dict[str, Any]]]) -> int:
    return max(
        sorted(rows),
        key=lambda step: (
            _mean(_correct(row) for row in rows[step].values()),
            -step,
        ),
    )


def _reject_test_artifacts(run_dir: Path, domain: str) -> None:
    evaluations = run_dir / "evaluations"
    for split in FORBIDDEN_EVALUATION_SPLITS:
        if (evaluations / split).exists() or (evaluations / split).is_symlink():
            raise RuntimeError(f"Forbidden {split!r} evaluation artifact in {run_dir}")
    if domain == "MBPP":
        manifest_path = run_dir / "split_manifest.json"
    else:
        manifest_path = run_dir / "parameters" / "evaluation_split_manifest.json"
    manifest_path = _authorized_file(manifest_path, run_dir, "split manifest")
    manifest = _read_json(manifest_path)
    if manifest.get("official_test_used") is not False:
        raise RuntimeError(f"Validation-only manifest is missing its no-test assertion: {manifest_path}")
    if any(key in manifest for key in ("test", "final_test", "pilot_subset")):
        raise RuntimeError(f"Test or pilot split leaked into {manifest_path}")


def load_pilot_run(
    entry: dict[str, Any], *, batch_dir: Path, pool_root: Path
) -> PilotRun:
    domain = _domain(entry)
    reward = str(entry.get("reward_value"))
    if reward not in REWARDS:
        raise ValueError(f"Unexpected reward transform: {reward!r}")
    expected_evaluation_module = (
        "iar_grpo.evaluate_pilot" if domain == "MBPP" else "iar_cqa.evaluate_full"
    )
    if entry.get("evaluation_module") != expected_evaluation_module:
        raise ValueError(
            f"Unexpected evaluation module for {domain}: {entry.get('evaluation_module')!r}"
        )
    config_path = _ensure_under(Path(str(entry["config"])), batch_dir, "pilot config")
    config_hash = _sha256(config_path)
    if entry.get("config_sha256") != config_hash:
        raise RuntimeError(f"Materialized pilot config drift: {config_path}")
    source_config_hash = str(entry.get("source_config_sha256", ""))
    if len(source_config_hash) != 64:
        raise ValueError(f"Missing source-config provenance for {entry.get('name')}")
    config = _read_yaml(config_path)
    if str((config.get("reward") or {}).get("value")) != reward:
        raise RuntimeError(f"Reward/config mismatch for {entry.get('name')}")
    evaluation = config.get("evaluation") or {}
    if evaluation.get("validation_only") is not True:
        raise RuntimeError(f"Pilot is not registered validation-only: {config_path}")
    if str(evaluation.get("select_on")) != "validation_task_accuracy":
        raise RuntimeError(f"Pilot does not use the frozen task selector: {config_path}")

    experiments_root = _ensure_under(
        Path(str(config["paths"]["experiments"])), pool_root, "experiment root"
    )
    run_id = str(entry["run_id"])
    run_dir = _ensure_under(experiments_root / run_id, pool_root, "pilot run")
    training_summary_path = _authorized_file(
        run_dir / "summary.json", run_dir, "training summary"
    )
    training_summary = _read_json(training_summary_path)
    if training_summary.get("status") != "complete":
        raise RuntimeError(f"Training run is incomplete: {run_dir}")
    evaluation_summary_path = _authorized_file(
        run_dir / "evaluation_summary.json", run_dir, "evaluation summary"
    )
    evaluation_summary = _read_json(evaluation_summary_path)
    if evaluation_summary.get("status") != "complete":
        raise RuntimeError(f"Validation evaluation is incomplete: {run_dir}")
    if evaluation_summary.get("validation_only") is not True:
        raise RuntimeError(f"Evaluation was not validation-only: {run_dir}")
    if evaluation_summary.get("test") or evaluation_summary.get("test_comparison"):
        raise RuntimeError(f"CQA test result leaked into validation-only run: {run_dir}")
    if evaluation_summary.get("final_test") or evaluation_summary.get("pilot_subset"):
        raise RuntimeError(f"MBPP test result leaked into validation-only run: {run_dir}")
    _reject_test_artifacts(run_dir, domain)

    checkpoints = tuple(int(step) for step in evaluation["checkpoints"])
    if not checkpoints or checkpoints[0] != 0 or checkpoints != tuple(sorted(set(checkpoints))):
        raise ValueError(f"Invalid checkpoint schedule in {config_path}")
    expected_rows = int(
        evaluation["validation"]["count"]
        if domain == "MBPP"
        else evaluation["validation_count"]
    )
    by_step: dict[int, dict[str, dict[str, Any]]] = {}
    validation_calls: list[CallAudit] = []
    source_hashes = {
        str(training_summary_path.relative_to(run_dir)): _sha256(training_summary_path),
        str(evaluation_summary_path.relative_to(run_dir)): _sha256(evaluation_summary_path),
        str(config_path): config_hash,
    }
    validation_root = run_dir / "evaluations" / "validation"
    present_steps = {
        int(path.name.removeprefix("step_"))
        for path in validation_root.glob("step_*")
        if path.is_dir()
    }
    if present_steps != set(checkpoints):
        raise RuntimeError(
            f"Validation checkpoint mismatch for {run_dir}: "
            f"expected {set(checkpoints)}, found {present_steps}"
        )
    for step in checkpoints:
        state = validation_root / f"step_{step:04d}"
        rows_path = _authorized_file(state / "rows.jsonl", run_dir, "validation rows")
        calls_path = _authorized_file(
            state / "reward_calls.jsonl", run_dir, "validation judge log"
        )
        by_step[step] = _load_validation_step(rows_path, expected_rows=expected_rows)
        validation_calls.append(audit_call_log(calls_path, expected_calls=expected_rows))
        source_hashes[str(rows_path.relative_to(run_dir))] = _sha256(rows_path)
        source_hashes[str(calls_path.relative_to(run_dir))] = _sha256(calls_path)
    ids = tuple(sorted(by_step[0]))
    if any(tuple(sorted(rows)) != ids for rows in by_step.values()):
        raise RuntimeError(f"Validation IDs change across checkpoints in {run_dir}")

    selected_step = _task_selected_step(by_step)
    reported_step = int((evaluation_summary.get("selection") or {}).get("selected_step", -1))
    if reported_step != selected_step:
        raise RuntimeError(
            f"Frozen task selector mismatch in {run_dir}: reported {reported_step}, "
            f"recomputed {selected_step}"
        )
    baseline = by_step[0]
    selected = by_step[selected_step]
    expected_training_calls = int(config["trial"]["expected_reward_calls"])
    training_log_path = _authorized_file(
        run_dir / "reward_calls.jsonl", run_dir, "training judge log"
    )
    training_calls = audit_call_log(
        training_log_path, expected_calls=expected_training_calls
    )
    source_hashes[str(training_log_path.relative_to(run_dir))] = _sha256(training_log_path)
    combined_validation = CallAudit(
        successful_calls=sum(item.successful_calls for item in validation_calls),
        http_rows=sum(item.http_rows for item in validation_calls),
        retries=sum(item.retries for item in validation_calls),
        cost_usd=sum(item.cost_usd for item in validation_calls),
    )
    return PilotRun(
        domain=domain,
        reward=reward,
        run_id=run_id,
        run_dir=run_dir,
        ids=ids,
        selected_step=selected_step,
        baseline_correct=tuple(_correct(baseline[row_id]) for row_id in ids),
        selected_correct=tuple(_correct(selected[row_id]) for row_id in ids),
        p_own_deltas=tuple(
            float(selected[row_id]["p_own"]) - float(baseline[row_id]["p_own"])
            for row_id in ids
        ),
        training_calls=training_calls,
        validation_calls=combined_validation,
        source_config_sha256=source_config_hash,
        source_hashes=source_hashes,
    )


def _interval(values: list[float]) -> dict[str, float]:
    return {
        "bootstrap_se": statistics.stdev(values) if len(values) > 1 else 0.0,
        "ci_95_low": _percentile(values, 0.025),
        "ci_95_high": _percentile(values, 0.975),
    }


def _run_summary(run: PilotRun, bootstrap: list[float]) -> dict[str, Any]:
    gain = _mean(run.task_deltas)
    training = run.training_calls
    validation = run.validation_calls
    total_calls = training.successful_calls + validation.successful_calls
    total_cost = training.cost_usd + validation.cost_usd
    return {
        "run_id": run.run_id,
        "run_dir": str(run.run_dir),
        "n_validation": len(run.ids),
        "selected_step": run.selected_step,
        "baseline_task_utility": _mean(run.baseline_correct),
        "selected_task_utility": _mean(run.selected_correct),
        "task_gain": gain,
        "task_gain_pp": 100.0 * gain,
        **_interval(bootstrap),
        "mean_p_own_gain": _mean(run.p_own_deltas),
        "training_successful_judge_calls": training.successful_calls,
        "training_retries": training.retries,
        "training_cost_usd": training.cost_usd,
        "validation_successful_judge_calls": validation.successful_calls,
        "validation_retries": validation.retries,
        "validation_cost_usd": validation.cost_usd,
        "total_successful_judge_calls": total_calls,
        "total_cost_usd": total_cost,
        "task_gain_pp_per_1000_training_successful_judge_calls": (
            100.0 * gain * 1000.0 / training.successful_calls
        ),
        "training_cost_usd_per_1000_successful_judge_calls": (
            training.cost_usd * 1000.0 / training.successful_calls
        ),
        "total_cost_usd_per_1000_successful_judge_calls": (
            total_cost * 1000.0 / total_calls
        ),
        "source_hashes": run.source_hashes,
    }


def analyze_batch(
    batch_dir: Path,
    *,
    bootstrap_reps: int = 10_000,
    bootstrap_seed: int = 20260907,
    expected_batch_id: str = EXPECTED_BATCH_ID,
) -> dict[str, Any]:
    if bootstrap_reps < 2:
        raise ValueError("bootstrap_reps must be at least two")
    batch_dir = batch_dir.expanduser().resolve()
    batch_path = batch_dir / "batch.json"
    batch = _read_json(batch_path)
    if int(batch.get("schema_version", -1)) != 1:
        raise ValueError("Reward-pilot batch must use schema_version 1")
    if str(batch.get("batch_id")) != expected_batch_id:
        raise ValueError(
            f"Expected batch {expected_batch_id!r}, found {batch.get('batch_id')!r}"
        )
    if batch_dir.name != expected_batch_id:
        raise ValueError(
            f"Batch directory name {batch_dir.name!r} does not match {expected_batch_id!r}"
        )
    pool_root = Path(str(batch["pool_root"])).expanduser().resolve()
    _ensure_under(batch_dir, pool_root, "batch directory")
    entries = batch.get("entries")
    if not isinstance(entries, list):
        raise TypeError("batch.entries must be a list")

    runs: dict[str, dict[str, PilotRun]] = {domain: {} for domain in DOMAINS}
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("Every batch entry must be an object")
        run = load_pilot_run(entry, batch_dir=batch_dir, pool_root=pool_root)
        if run.reward in runs[run.domain]:
            raise ValueError(f"Duplicate {run.domain}/{run.reward} pilot")
        runs[run.domain][run.reward] = run
    expected = set(REWARDS)
    for domain in DOMAINS:
        if set(runs[domain]) != expected:
            raise RuntimeError(
                f"Incomplete reward matrix for {domain}: expected {expected}, "
                f"found {set(runs[domain])}"
            )
        reference = runs[domain][REWARDS[0]]
        for reward in REWARDS[1:]:
            candidate = runs[domain][reward]
            if candidate.ids != reference.ids:
                raise RuntimeError(f"Validation IDs differ across {domain} reward arms")
            if candidate.baseline_correct != reference.baseline_correct:
                raise RuntimeError(f"Frozen actor baseline differs across {domain} reward arms")
            if candidate.source_config_sha256 != reference.source_config_sha256:
                raise RuntimeError(f"Source config differs across {domain} reward arms")

    rng = random.Random(bootstrap_seed)
    domain_bootstrap = {
        domain: {reward: [] for reward in REWARDS} for domain in DOMAINS
    }
    macro_bootstrap = {reward: [] for reward in REWARDS}
    for _ in range(bootstrap_reps):
        sampled_domain_gains: dict[str, dict[str, float]] = {}
        for domain in DOMAINS:
            n = len(runs[domain][REWARDS[0]].ids)
            indices = [rng.randrange(n) for _ in range(n)]
            sampled_domain_gains[domain] = {}
            for reward in REWARDS:
                deltas = runs[domain][reward].task_deltas
                value = _mean(deltas[index] for index in indices)
                sampled_domain_gains[domain][reward] = value
                domain_bootstrap[domain][reward].append(value)
        for reward in REWARDS:
            macro_bootstrap[reward].append(
                _mean(sampled_domain_gains[domain][reward] for domain in DOMAINS)
            )

    domains: dict[str, Any] = {}
    for domain in DOMAINS:
        domains[domain] = {
            "winner": max(
                REWARDS,
                key=lambda reward: (
                    _mean(runs[domain][reward].task_deltas),
                    -REWARDS.index(reward),
                ),
            ),
            "rewards": {
                reward: _run_summary(
                    runs[domain][reward], domain_bootstrap[domain][reward]
                )
                for reward in REWARDS
            },
        }

    macro_gain = {
        reward: _mean(
            domains[domain]["rewards"][reward]["task_gain"] for domain in DOMAINS
        )
        for reward in REWARDS
    }
    macro_resources = {}
    for reward in REWARDS:
        training_calls = sum(
            runs[domain][reward].training_calls.successful_calls for domain in DOMAINS
        )
        validation_calls = sum(
            runs[domain][reward].validation_calls.successful_calls for domain in DOMAINS
        )
        training_cost = sum(
            runs[domain][reward].training_calls.cost_usd for domain in DOMAINS
        )
        validation_cost = sum(
            runs[domain][reward].validation_calls.cost_usd for domain in DOMAINS
        )
        macro_resources[reward] = {
            "training_successful_judge_calls": training_calls,
            "validation_successful_judge_calls": validation_calls,
            "total_successful_judge_calls": training_calls + validation_calls,
            "training_cost_usd": training_cost,
            "validation_cost_usd": validation_cost,
            "total_cost_usd": training_cost + validation_cost,
            "mean_task_gain_pp_per_1000_training_successful_judge_calls": (
                100.0 * macro_gain[reward] * 1000.0 / training_calls
            ),
            "training_cost_usd_per_1000_successful_judge_calls": (
                training_cost * 1000.0 / training_calls
            ),
        }
    ordered = sorted(REWARDS, key=lambda reward: (-macro_gain[reward], REWARDS.index(reward)))
    raw_best, runner_up = ordered[:2]
    difference = macro_gain[raw_best] - macro_gain[runner_up]
    difference_bootstrap = [
        left - right
        for left, right in zip(
            macro_bootstrap[raw_best], macro_bootstrap[runner_up], strict=True
        )
    ]
    difference_se = statistics.stdev(difference_bootstrap)
    within_one_se = difference <= difference_se + 1e-15
    domain_winners = {domain: domains[domain]["winner"] for domain in DOMAINS}
    winners_disagree = len(set(domain_winners.values())) > 1
    fallback_reasons = []
    if within_one_se:
        fallback_reasons.append("leading_macro_transforms_within_one_bootstrap_se")
    if winners_disagree and within_one_se:
        fallback_reasons.append("domain_winners_disagree_without_clear_macro_advantage")
    frozen = "log_p_own" if fallback_reasons else raw_best

    return {
        "schema_version": 1,
        "analysis": "reward_transform_pilot_gate_v1",
        "batch_id": expected_batch_id,
        "batch_file": str(batch_path),
        "batch_file_sha256": _sha256(batch_path),
        "access_policy": {
            "validation_only": True,
            "official_test_read": False,
            "forbidden_evaluation_splits": list(FORBIDDEN_EVALUATION_SPLITS),
        },
        "selector": {
            "name": "validation_task_accuracy",
            "tie_break": "earliest checkpoint",
            "source_gate": "checkpoint_selector_gate_v2",
        },
        "bootstrap": {
            "reps": bootstrap_reps,
            "seed": bootstrap_seed,
            "unit": "paired validation instance within domain; domains macro-averaged equally",
        },
        "domains": domains,
        "macro": {
            reward: {
                "mean_task_gain": macro_gain[reward],
                "mean_task_gain_pp": 100.0 * macro_gain[reward],
                **_interval(macro_bootstrap[reward]),
                **macro_resources[reward],
            }
            for reward in REWARDS
        },
        "decision": {
            "raw_best_reward": raw_best,
            "runner_up_reward": runner_up,
            "raw_best_minus_runner_up": difference,
            "difference_bootstrap_se": difference_se,
            "leading_within_one_bootstrap_se": within_one_se,
            "domain_winners": domain_winners,
            "domain_winners_disagree": winners_disagree,
            "fallback_reasons": fallback_reasons,
            "frozen_reward": frozen,
            "tie_rule": (
                "Choose the largest unweighted two-domain macro gain; retain log_p_own "
                "when the leaders are within one paired-bootstrap SE or when domain "
                "winners disagree without a clear macro advantage."
            ),
        },
    }
