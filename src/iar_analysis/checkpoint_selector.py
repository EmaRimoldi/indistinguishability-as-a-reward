from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

SELECTORS = ("task", "mean_p_own", "p_own_lcb", "task_then_p_own")
ONE_SIDED_95_Z = 1.6448536269514722


@dataclass(frozen=True)
class CheckpointTable:
    name: str
    source: Path
    steps: tuple[int, ...]
    ids: tuple[str, ...]
    correct: dict[int, dict[str, bool]]
    p_own: dict[int, dict[str, float]]
    source_hashes: dict[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_id(row: dict[str, Any]) -> str:
    for key in ("task_id", "id", "source_index", "dataset_index"):
        if key in row:
            return str(row[key])
    raise ValueError("Validation row has no stable identifier")


def _correct(row: dict[str, Any]) -> bool:
    if isinstance(row.get("unit_tests"), dict) and "passed" in row["unit_tests"]:
        return bool(row["unit_tests"]["passed"])
    if isinstance(row.get("score"), dict) and "correct" in row["score"]:
        return bool(row["score"]["correct"])
    if "correct" in row:
        return bool(row["correct"])
    raise ValueError("Validation row has no correctness field")


def load_checkpoint_table(name: str, run_dir: Path) -> CheckpointTable:
    validation = run_dir / "evaluations" / "validation"
    paths = sorted(validation.glob("step_*/rows.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No validation checkpoints under {validation}")

    correct: dict[int, dict[str, bool]] = {}
    p_own: dict[int, dict[str, float]] = {}
    hashes: dict[str, str] = {}
    reference_ids: set[str] | None = None
    for path in paths:
        try:
            step = int(path.parent.name.removeprefix("step_"))
        except ValueError as exc:
            raise ValueError(f"Invalid checkpoint directory: {path.parent}") from exc
        step_correct: dict[str, bool] = {}
        step_p_own: dict[str, float] = {}
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row_id = _row_id(row)
                if row_id in step_correct:
                    raise ValueError(f"Duplicate id {row_id!r} in {path}:{line_number}")
                value = float(row["p_own"])
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"Invalid p_own={value!r} in {path}:{line_number}")
                step_correct[row_id] = _correct(row)
                step_p_own[row_id] = value
        ids = set(step_correct)
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            missing = sorted(reference_ids - ids)[:5]
            extra = sorted(ids - reference_ids)[:5]
            raise ValueError(
                f"Checkpoint id mismatch for {name} step {step}: missing={missing}, extra={extra}"
            )
        correct[step] = step_correct
        p_own[step] = step_p_own
        hashes[str(path.relative_to(run_dir))] = _sha256(path)

    assert reference_ids is not None
    steps = tuple(sorted(correct))
    if steps[0] != 0:
        raise ValueError(f"{name} has no step-0 checkpoint")
    return CheckpointTable(
        name=name,
        source=run_dir,
        steps=steps,
        ids=tuple(sorted(reference_ids)),
        correct=correct,
        p_own=p_own,
        source_hashes=hashes,
    )


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("Cannot average an empty collection")
    return fmean(materialized)


def _p_own_lcb(values: list[float], z: float = ONE_SIDED_95_Z) -> float:
    if len(values) < 2:
        return values[0]
    return fmean(values) - z * stdev(values) / math.sqrt(len(values))


def _pick_step(table: CheckpointTable, ids: tuple[str, ...], selector: str) -> int:
    rows: list[tuple[int, float, float, float]] = []
    for step in table.steps:
        task = _mean(float(table.correct[step][row_id]) for row_id in ids)
        p_values = [table.p_own[step][row_id] for row_id in ids]
        rows.append((step, task, fmean(p_values), _p_own_lcb(p_values)))

    if selector == "task":
        return max(rows, key=lambda row: (row[1], -row[0]))[0]
    if selector == "mean_p_own":
        return max(rows, key=lambda row: (row[2], -row[0]))[0]
    if selector == "p_own_lcb":
        return max(rows, key=lambda row: (row[3], row[2], -row[0]))[0]
    if selector == "task_then_p_own":
        return max(rows, key=lambda row: (row[1], row[2], -row[0]))[0]
    raise ValueError(f"Unknown selector: {selector}")


def _stratified_half_split(
    table: CheckpointTable, *, rng: random.Random, repeat: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    baseline = table.correct[0]
    strata = {
        False: [row_id for row_id in table.ids if not baseline[row_id]],
        True: [row_id for row_id in table.ids if baseline[row_id]],
    }
    selection: list[str] = []
    audit: list[str] = []
    for key in (False, True):
        values = strata[key][:]
        rng.shuffle(values)
        cut = len(values) // 2 + (1 if len(values) % 2 and repeat % 2 == int(key) else 0)
        selection.extend(values[:cut])
        audit.extend(values[cut:])
    if not selection or not audit:
        raise ValueError(f"Degenerate split for {table.name}")
    return tuple(sorted(selection)), tuple(sorted(audit))


def evaluate_dataset(
    table: CheckpointTable, *, seed: int, repeats: int = 2000
) -> dict[str, Any]:
    rng = random.Random(seed)
    regrets = {selector: [] for selector in SELECTORS}
    selected = {selector: Counter() for selector in SELECTORS}
    split_sizes: Counter[tuple[int, int]] = Counter()

    for repeat in range(repeats):
        selection_ids, audit_ids = _stratified_half_split(table, rng=rng, repeat=repeat)
        split_sizes[(len(selection_ids), len(audit_ids))] += 1
        # Evaluate both complementary directions.  This uses every validation
        # row for out-of-fold auditing while keeping selection and audit rows
        # disjoint within each direction.
        for select_fold, audit_fold in (
            (selection_ids, audit_ids),
            (audit_ids, selection_ids),
        ):
            audit_task = {
                step: _mean(float(table.correct[step][row_id]) for row_id in audit_fold)
                for step in table.steps
            }
            oracle = max(audit_task.values())
            for selector in SELECTORS:
                step = _pick_step(table, select_fold, selector)
                selected[selector][step] += 1
                regret = oracle - audit_task[step]
                if regret < -1e-12:
                    raise AssertionError("Regret cannot be negative")
                regrets[selector].append(max(0.0, regret))

    summaries: dict[str, Any] = {}
    for selector in SELECTORS:
        values = regrets[selector]
        summaries[selector] = {
            "mean_regret": fmean(values),
            "regret_sd_across_splits": stdev(values) if len(values) > 1 else 0.0,
            "zero_regret_fraction": sum(value == 0 for value in values) / len(values),
            "selected_step_counts": {
                str(step): selected[selector][step] for step in sorted(selected[selector])
            },
        }
    return {
        "dataset": table.name,
        "source": str(table.source),
        "n": len(table.ids),
        "steps": list(table.steps),
        "baseline_accuracy": _mean(float(table.correct[0][row_id]) for row_id in table.ids),
        "repeats": repeats,
        "crossfit_directions_per_repeat": 2,
        "split_sizes": {f"{a}/{b}": count for (a, b), count in sorted(split_sizes.items())},
        "selectors": summaries,
        "source_hashes": table.source_hashes,
    }


def choose_macro_selector(
    datasets: list[dict[str, Any]], *, seed: int = 23017, bootstrap_reps: int = 10000
) -> dict[str, Any]:
    if not datasets:
        raise ValueError("At least one dataset is required")
    per_dataset = {
        item["dataset"]: {
            selector: float(item["selectors"][selector]["mean_regret"])
            for selector in SELECTORS
        }
        for item in datasets
    }
    macro = {
        selector: fmean(values[selector] for values in per_dataset.values())
        for selector in SELECTORS
    }
    best = min(SELECTORS, key=lambda selector: (macro[selector], SELECTORS.index(selector)))

    rng = random.Random(seed)
    names = sorted(per_dataset)
    bootstrap_values = {selector: [] for selector in SELECTORS}
    bootstrap_differences = {selector: [] for selector in SELECTORS}
    for _ in range(bootstrap_reps):
        sampled = [rng.choice(names) for _ in names]
        sampled_means = {
            selector: fmean(per_dataset[name][selector] for name in sampled)
            for selector in SELECTORS
        }
        for selector in SELECTORS:
            bootstrap_values[selector].append(sampled_means[selector])
            bootstrap_differences[selector].append(sampled_means[selector] - sampled_means[best])

    summary: dict[str, Any] = {}
    tied: list[str] = []
    for selector in SELECTORS:
        se = stdev(bootstrap_values[selector]) if bootstrap_reps > 1 else 0.0
        diff_se = stdev(bootstrap_differences[selector]) if bootstrap_reps > 1 else 0.0
        difference = macro[selector] - macro[best]
        within_one_se = difference <= diff_se + 1e-15
        if within_one_se:
            tied.append(selector)
        summary[selector] = {
            "macro_mean_regret": macro[selector],
            "dataset_bootstrap_se": se,
            "difference_from_best": difference,
            "difference_bootstrap_se": diff_se,
            "within_one_se_of_best": within_one_se,
        }

    chosen = "task" if "task" in tied else best
    return {
        "per_dataset_mean_regret": per_dataset,
        "selectors": summary,
        "raw_best_selector": best,
        "selectors_within_one_bootstrap_se": tied,
        "frozen_selector": chosen,
        "tie_rule": "Prefer external task utility when it is within one paired dataset-bootstrap SE of the raw best.",
        "bootstrap_reps": bootstrap_reps,
        "bootstrap_seed": seed,
    }
