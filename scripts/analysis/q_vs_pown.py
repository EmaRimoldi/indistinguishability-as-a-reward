"""How much of log p_own does the Quality judge q explain, and does p_own add
information about correctness beyond q? (revision 2026-09-22)

Inputs: q scored on the 8,000 seed-23 Identity rollouts per domain
(score_q_on_identity.py) and the verifier labels of the same rollouts.

Reported per domain:
* R^2 of log p_own on log q: pooled, and within prompt group (both demeaned by group).
* Pearson and Spearman correlation of p_own and q.
* Logistic regression of correctness on log q, and on log q + log p_own:
  likelihood-ratio test, AUC of each model, and prompt-bootstrap intervals.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from rollout_signal import ALIGN, read_jsonl

logger = logging.getLogger(__name__)

ROOT = Path("/path/to/iar_code")
Q_DIR = ROOT / "results/revision_20260922/q_on_identity"
CLUSTER_LABELS = ROOT / "results/revision_20260922/seed_rollout_labels"
DOMAINS = ("commonsenseqa", "gsm8k", "kodcode", "spider")
BOOT = 1000


def labels(domain: str) -> dict[int, int]:
    """Seed-23 verifier labels; KodCode uses the complete cluster re-execution."""
    if domain == "kodcode":
        return {r["call_id"]: int(r["correct"]) for r in read_jsonl(CLUSTER_LABELS / "rows_kodcode_seed23.jsonl")}
    return {r["call_id"]: int(r["correct"]) for r in read_jsonl(ALIGN / f"rows_{domain}_p_own.jsonl")}


def logistic(x: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[np.ndarray, float]:
    """IRLS logistic regression with intercept; returns coefficients and log-likelihood."""
    X = np.column_stack([np.ones(len(y)), x])
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-9, None)
        step = np.linalg.solve(X.T @ (X * w[:, None]) + 1e-8 * np.eye(X.shape[1]), X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = np.clip(1 / (1 + np.exp(-(X @ beta))), 1e-12, 1 - 1e-12)
    return beta, float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def average_ranks(s: np.ndarray) -> np.ndarray:
    """Deterministic average ranks, including ties (required for Spearman)."""
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    # Average tied ranks in sorted order; avoid a full-array scan per unique value.
    sorted_scores = s[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1, len(s)]
    ranks[order] = np.repeat((bounds[:-1] + bounds[1:] + 1) / 2, np.diff(bounds))
    return ranks


def auc(y: np.ndarray, s: np.ndarray) -> float:
    ranks = average_ranks(s)
    pos = y == 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def r2(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def demean(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = values.astype(float).copy()
    for g in np.unique(groups):
        idx = groups == g
        out[idx] -= values[idx].mean()
    return out


def analyse(domain: str) -> dict:
    rows = read_jsonl(Q_DIR / f"q_rows_{domain}.jsonl")
    lab = labels(domain)
    visits: dict[tuple[str, int], int] = defaultdict(int)
    group_ids = []
    for r in sorted(rows, key=lambda r: r["identity_call_id"]):
        if r["rollout_index"] == 0:
            visits[(r["task_id"], 0)] += 1
        group_ids.append(f'{r["task_id"]}#{visits[(r["task_id"], 0)]}')
    rows = sorted(rows, key=lambda r: r["identity_call_id"])
    lp = np.array([r["log_p_own"] for r in rows])
    lq = np.array([r["log_q"] for r in rows])
    y = np.array([lab[r["identity_call_id"]] for r in rows], dtype=float)
    groups = np.array(group_ids)
    prompts = np.array([r["task_id"] for r in rows])

    def stats(idx: np.ndarray) -> dict:
        _, ll_q = logistic(lq[idx], y[idx])
        _, ll_qp = logistic(np.column_stack([lq[idx], lp[idx]]), y[idx])
        b_qp, _ = logistic(np.column_stack([lq[idx], lp[idx]]), y[idx])
        s_q = lq[idx]
        s_qp = np.column_stack([np.ones(idx.sum() if idx.dtype == bool else len(idx)), lq[idx], lp[idx]]) @ b_qp
        return {"r2_pooled": r2(lq[idx], lp[idx]),
                "lr_stat": 2 * (ll_qp - ll_q), "auc_q": auc(y[idx], s_q), "auc_q_plus_pown": auc(y[idx], s_qp),
                "auc_pown": auc(y[idx], lp[idx])}

    full = np.arange(len(rows))
    point = stats(full)
    point["r2_within_prompt"] = r2(demean(lq, groups), demean(lp, groups))
    point["pearson_p_q"] = float(np.corrcoef(np.exp(lp), np.exp(lq))[0, 1])
    rank = average_ranks
    point["spearman_p_q"] = float(np.corrcoef(rank(lp), rank(lq))[0, 1])
    point["lr_p_value"] = math.erfc(math.sqrt(max(point["lr_stat"], 0) / 2))
    point["n"] = len(rows)
    point["correct_rate"] = float(y.mean())
    rng = np.random.default_rng(23)
    uniq = np.unique(prompts)
    index = {p: np.where(prompts == p)[0] for p in uniq}
    boots = defaultdict(list)
    for _ in range(BOOT):
        pick = np.concatenate([index[p] for p in rng.choice(uniq, len(uniq))])
        b = stats(pick)
        for k in ("r2_pooled", "auc_q", "auc_q_plus_pown"):
            boots[k].append(b[k])
        boots["delta_auc"].append(b["auc_q_plus_pown"] - b["auc_q"])
    point["ci95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
    point["delta_auc"] = point["auc_q_plus_pown"] - point["auc_q"]
    logger.info("%s R2=%.3f (within %.3f) dAUC=%.3f LR p=%.2g", domain, point["r2_pooled"],
                point["r2_within_prompt"], point["delta_auc"], point["lr_p_value"])
    return point


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = {d: analyse(d) for d in DOMAINS if (Q_DIR / f"q_rows_{d}.jsonl").exists()}
    args.output.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
