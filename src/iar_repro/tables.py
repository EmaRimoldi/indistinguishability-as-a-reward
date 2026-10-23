"""Recompute paper accuracies from recorded per-item verifier outcomes."""

from __future__ import annotations

import ast
import json
import math
import statistics
from pathlib import Path

from .paths import DOMAINS, ROOT, SEEDS
from .records import manifest, materialize, result_path


def scores(relative: str) -> dict[str, dict]:
    out = {}
    for line in materialize(relative).read_text().splitlines():
        row = json.loads(line)
        score = (
            ast.literal_eval(row["score"])
            if isinstance(row["score"], str)
            else row["score"]
        )
        if str(row["id"]) in out:
            raise ValueError("Duplicate evaluation item")
        if not score.get("excluded_reference_invalid"):
            out[str(row["id"])] = score
    return out


def recompute(output: Path) -> dict:
    index = {
        (r.get("domain"), r.get("condition")): r["local"]
        for r in manifest()
        if r.get("kind") in ("heldout", "heldout_extra")
    }
    expected = json.loads(
        (result_path("results/heldout_accuracy_per_seed.json")).read_text()
    )
    expected_k10 = json.loads(
        (result_path("results/k10_heldout_stats.json")).read_text()
    )
    expected_teacher = json.loads(
        (result_path("results/teacher_heldout.json")).read_text()
    )
    result = {}
    for domain in DOMAINS:
        a0 = scores(index[(domain, "a0")])
        n = len(a0)
        accuracy = lambda rows: (
            100 * sum(bool(r.get("correct")) for r in rows.values()) / len(rows)
        )
        a0_pct = accuracy(a0)
        assert math.isclose(a0_pct, expected[domain]["a0"], abs_tol=1e-10)
        data = {"n": n, "a0_pct": a0_pct, "conditions": {}}
        for condition, source in (
            ("indist", "identity"),
            ("quality", "grpo_quality"),
            ("oracle", "grpo_oracle"),
            ("sft", "sft_teacher"),
        ):
            per_seed = {}
            for seed in SEEDS:
                key = (
                    f"identity_s{seed}"
                    if condition == "indist"
                    else condition
                    if seed == 23
                    else f"{source}_s{seed}"
                )
                rows = scores(index[(domain, key)])
                if rows.keys() != a0.keys():
                    raise ValueError(f"Held-out item mismatch: {domain}/{key}")
                value = accuracy(rows)
                if not math.isclose(
                    value, expected[domain][condition][str(seed)], abs_tol=1e-10
                ):
                    raise ValueError(f"Recorded accuracy mismatch: {domain}/{key}")
                per_seed[str(seed)] = value
            values = list(per_seed.values())
            data["conditions"][condition] = {
                "per_seed_pct": per_seed,
                "mean_pct": statistics.mean(values),
                "sample_sd_pct": statistics.stdev(values),
            }
        for name, reference, field in (
            ("k10", expected_k10, "k10_pct"),
            ("teacher", expected_teacher, "teacher_pct"),
        ):
            rows = scores(index[(domain, name)])
            if rows.keys() != a0.keys():
                raise ValueError(f"Held-out item mismatch: {domain}/{name}")
            value = accuracy(rows)
            if not math.isclose(value, reference[domain][field], abs_tol=1e-10):
                raise ValueError(f"Recorded accuracy mismatch: {domain}/{name}")
            data[name] = {"accuracy_pct": value, "gain_pp": value - a0_pct}
        result[domain] = data
    report = {
        "status": "passed",
        "source": "recorded per-item outcomes; no model inference",
        "matches_frozen_summaries": True,
        "model_calls": 0,
        "full_experiments_rerun": False,
        "domains": result,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
