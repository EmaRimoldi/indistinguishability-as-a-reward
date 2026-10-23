"""Held-out statistics from stored outputs (A3, A10, A12); no regeneration.

* A3: invalid-output rates by condition and domain.
* A10: paired Identity-vs-baseline tests (exact McNemar, paired bootstrap, Holm).
* A12: Identity mean and SD over seeds 23, 42 and 7, with per-seed tests vs A0.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DOMAINS = ("commonsenseqa", "gsm8k", "kodcode", "spider")
SEEDS = ("identity_s23", "identity_s42", "identity_s7")
BASELINES = ("sft", "quality", "oracle")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 23


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def outcome(domain: str, score: dict[str, Any]) -> tuple[bool, bool] | None:
    """Return (correct, invalid) or None for rows excluded from the denominator."""
    if score.get("excluded_reference_invalid"):
        return None
    correct = bool(score.get("correct"))
    if domain == "kodcode":
        invalid = score.get("returncode") not in (0, 1)
    elif domain == "spider":
        invalid = score.get("status") == "invalid_or_blocked_sql"
    else:
        invalid = not bool(score.get("valid"))
    return correct, invalid


def load_condition(domain: str, path: str) -> dict[str, tuple[bool, bool]]:
    rows = {}
    for row in read_jsonl(path):
        res = outcome(domain, row["score"])
        if res is not None:
            rows[str(row["id"])] = res
    return rows


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_bootstrap(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    diff = x.astype(float) - y.astype(float)
    idx = rng.integers(0, len(diff), size=(BOOTSTRAP_REPLICATES, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    adjusted, running = {}, 0.0
    for rank, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p))
        adjusted[key] = running
    return adjusted


def compare(a: dict[str, tuple[bool, bool]], b: dict[str, tuple[bool, bool]]) -> dict[str, Any]:
    ids = sorted(set(a) & set(b))
    x = np.array([a[i][0] for i in ids])
    y = np.array([b[i][0] for i in ids])
    only_a = int(np.sum(x & ~y))
    only_b = int(np.sum(~x & y))
    low, high = paired_bootstrap(x, y)
    return {
        "n": len(ids),
        "diff_pp": 100 * float(x.mean() - y.mean()),
        "ci95_pp": [100 * low, 100 * high],
        "discordant": [only_a, only_b],
        "mcnemar_p": mcnemar_exact(only_a, only_b),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    files = json.loads(args.files.read_text())
    result: dict[str, Any] = {}
    for domain in DOMAINS:
        conds = {name: load_condition(domain, path) for name, path in files[domain].items()}
        acc = {n: 100 * float(np.mean([v[0] for v in c.values()])) for n, c in conds.items()}
        invalid = {n: {"count": int(sum(v[1] for v in c.values())), "n": len(c),
                       "rate_pct": 100 * float(np.mean([v[1] for v in c.values()]))}
                   for n, c in conds.items()}
        seed_acc = [acc[s] for s in SEEDS]
        vs_a0 = {s: compare(conds[s], conds["a0"]) for s in SEEDS}
        vs_a0_holm = holm({s: r["mcnemar_p"] for s, r in vs_a0.items()})
        pairwise: dict[str, Any] = {}
        for seed in SEEDS:
            tests = {base: compare(conds[seed], conds[base]) for base in BASELINES + ("prompt_imitation",)}
            adj = holm({b: tests[b]["mcnemar_p"] for b in BASELINES})
            for base in BASELINES:
                tests[base]["holm_p"] = adj[base]
            pairwise[seed] = tests
        mean_diff = {base: float(np.mean([pairwise[s][base]["diff_pp"] for s in SEEDS]))
                     for base in BASELINES + ("prompt_imitation",)}
        result[domain] = {
            "accuracy_pct": acc,
            "invalid": invalid,
            "identity_seeds": {"values": seed_acc, "mean": float(np.mean(seed_acc)),
                               "sd": float(np.std(seed_acc, ddof=1))},
            "identity_vs_a0": {s: {**vs_a0[s], "holm_p": vs_a0_holm[s]} for s in SEEDS},
            "identity_vs_baselines": pairwise,
            "mean_over_seeds_diff_pp": mean_diff,
        }
        logger.info("%s identity %.1f±%.1f | invalid a0 %.1f%%", domain,
                    result[domain]["identity_seeds"]["mean"], result[domain]["identity_seeds"]["sd"],
                    invalid["a0"]["rate_pct"])
    args.output.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
