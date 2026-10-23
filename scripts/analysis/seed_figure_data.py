"""Compact three-seed data for Figure 2 (training trajectory) and Figure 3 / Table 4
(correctness by attribution decile). Reads archived logs and labels only.

Outputs (next to the paper's plotting scripts):
  training_trajectory_seeds.json, correctness_deciles_seeds.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from rollout_signal import ALIGN, read_jsonl, reward_log

logger = logging.getLogger(__name__)

ROOT = Path("/path/to/iar_code")
LABELS = ROOT / "results/revision_20260922/seed_rollout_labels"
THREE_SEED_CSV = Path("/path/to/runs/"
                      "identity_seed42/seed7/test_evaluations/three_seed_identity_heldout.csv")
DOMAINS = ("commonsenseqa", "gsm8k", "kodcode", "spider")
SEEDS = ("seed23", "seed42", "seed7")
A0_CORRECT = {"commonsenseqa": (707, 1221), "gsm8k": (735, 1319), "kodcode": (385, 992), "spider": (495, 1034)}


def labelled_rows(domain: str, seed: str) -> list[dict[str, Any]]:
    """Rows with p_own, correct, task_id, occurrence, rollout_index, call_id."""
    cluster = LABELS / f"rows_{domain}_{seed}.jsonl"
    if seed == "seed23" and not cluster.exists():
        rows = read_jsonl(ALIGN / f"rows_{domain}_p_own.jsonl")
        return [{"call_id": r["call_id"], "task_id": r["task_id"], "occurrence": r["occurrence"],
                 "rollout_index": r["rollout_index"], "p_own": r["score_probability"],
                 "correct": int(r["correct"])} for r in rows]
    rows = sorted(read_jsonl(LABELS / f"rows_{domain}_{seed}.jsonl"), key=lambda r: r["call_id"])
    seen: dict[tuple[str, int], int] = defaultdict(int)
    out = []
    for r in rows:
        if r["rollout_index"] == 0:
            seen[(r["task_id"], 0)] += 1
        occ = seen[(r["task_id"], 0)]
        out.append({**r, "occurrence": occ})
    return out


def deciles(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    ordered = sorted(rows, key=lambda r: (r["p_own"], r["task_id"], r["occurrence"], r["rollout_index"]))
    size = len(ordered) // 10
    return [{"decile": k + 1,
             "correct_rate": float(np.mean([r["correct"] for r in ordered[k * size:(k + 1) * size]])),
             "mean_p_own": float(np.mean([r["p_own"] for r in ordered[k * size:(k + 1) * size]]))}
            for k in range(10)]


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = scores.argsort()
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def summary_table(rows: list[dict[str, Any]]) -> dict[str, float]:
    y = np.array([r["correct"] for r in rows])
    p = np.array([r["p_own"] for r in rows])
    return {"correct_rate": float(y.mean()), "p_own_correct": float(p[y == 1].mean()),
            "p_own_incorrect": float(p[y == 0].mean()), "auc": auc(y, p)}


def reflected_kde(values: np.ndarray, grid: np.ndarray, bandwidth: float = 0.035) -> np.ndarray:
    """Gaussian KDE on [0, 1] with reflection at both probability boundaries."""
    density = np.zeros_like(grid, dtype=float)
    for start in range(0, len(values), 2000):
        batch = values[start:start + 2000]
        for reflected in (batch, -batch, 2.0 - batch):
            z = (grid[:, None] - reflected[None, :]) / bandwidth
            density += np.exp(-0.5 * z * z).sum(axis=1)
    return density / (len(values) * bandwidth * np.sqrt(2.0 * np.pi))


def distributions(per_domain: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    """Pooled densities of p_own by correctness, with per-seed means and AUC."""
    grid = np.linspace(0, 1, 501)
    out: dict[str, Any] = {"grid": grid.tolist(), "bandwidth": 0.035, "domains": {}}
    for domain, per_seed in per_domain.items():
        pooled = [r for rows in per_seed.values() for r in rows]
        p = np.array([r["p_own"] for r in pooled])
        y = np.array([r["correct"] for r in pooled])
        stats = {seed: summary_table(rows) for seed, rows in per_seed.items()}
        out["domains"][domain] = {
            "n_pooled": len(pooled),
            "density_correct": reflected_kde(p[y == 1], grid).tolist(),
            "density_incorrect": reflected_kde(p[y == 0], grid).tolist(),
            "mean_correct_per_seed": [stats[s]["p_own_correct"] for s in SEEDS],
            "mean_incorrect_per_seed": [stats[s]["p_own_incorrect"] for s in SEEDS],
            "auc_per_seed": [stats[s]["auc"] for s in SEEDS],
            "correct_rate_per_seed": [stats[s]["correct_rate"] for s in SEEDS],
        }
    return out


def trajectory() -> dict[str, Any]:
    heldout: dict[str, dict[int, list[float]]] = defaultdict(dict)
    with open(THREE_SEED_CSV, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            heldout[row["domain"]][int(row["step"])] = [float(row[s]) for s in SEEDS]
    out = {}
    for domain in DOMAINS:
        k, n = A0_CORRECT[domain]
        steps = {0: [100 * k / n] * 3, **heldout[domain]}
        windows, log_windows = {}, {}
        for seed in SEEDS:
            ok = sorted((r for r in read_jsonl(reward_log("grpo_identity", domain, seed))
                         if r.get("status") == "ok"), key=lambda r: r["call_id"])
            windows[seed] = [float(np.mean([r["p_own"] for r in ok[i * 400:(i + 1) * 400]])) for i in range(20)]
            log_windows[seed] = [float(np.mean([r["log_p_own"] for r in ok[i * 400:(i + 1) * 400]]))
                                 for i in range(20)]
        out[domain] = {"heldout_pct": {str(s): v for s, v in sorted(steps.items())},
                       "p_own_windows": windows, "log_p_own_windows": log_windows,
                       "window_end_steps": [100 * (i + 1) for i in range(20)]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    archived = json.loads((args.out_dir / "correctness_deciles_data.json").read_text())
    dec, tab, rows_by_domain = {}, {}, {}
    for domain in DOMAINS:
        per_seed = {seed: labelled_rows(domain, seed) for seed in SEEDS}
        rows_by_domain[domain] = per_seed
        dec[domain] = {seed: deciles(rows) for seed, rows in per_seed.items()}
        tab[domain] = {seed: summary_table(rows) for seed, rows in per_seed.items()}
        ref = sorted((r for r in archived["rows"] if r["domain"] == domain), key=lambda r: r["decile"])
        diff = max(abs(a["correct_rate"] - b["correct_rate"]) for a, b in zip(ref, dec[domain]["seed23"], strict=True))
        if not (LABELS / f"rows_{domain}_seed23.jsonl").exists():
            assert diff < 1e-12, f"seed-23 deciles do not reproduce the archive for {domain}: {diff}"
        else:
            logger.info("%s: seed 23 relabelled on the cluster; max decile change vs archive %.3f", domain, diff)
        logger.info("%s AUC %s", domain, [round(tab[domain][s]["auc"], 3) for s in SEEDS])
    note = ("Seeds 42 and 7 are labelled with the frozen reward-alignment scorer on the cluster (Linux, Python 3.14). "
            "Seed 23 uses the archived labels, except KodCode, which is relabelled on the cluster for consistency.")
    (args.out_dir / "correctness_deciles_seeds.json").write_text(
        json.dumps({"note": note, "seeds": list(SEEDS), "deciles": dec, "table": tab}, indent=1) + "\n")
    (args.out_dir / "reward_distributions_seeds.json").write_text(
        json.dumps({"note": note, "seeds": list(SEEDS), **distributions(rows_by_domain)}) + "\n")
    (args.out_dir / "training_trajectory_seeds.json").write_text(
        json.dumps({"seeds": list(SEEDS), "domains": trajectory()}, indent=1) + "\n")


if __name__ == "__main__":
    main()
