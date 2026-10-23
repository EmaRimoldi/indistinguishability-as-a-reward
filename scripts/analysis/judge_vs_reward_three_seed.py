"""The indistinguishability reward against the teacher's correctness judgment, three seeds.

For each Identity seed (23, 42, 7) and domain, reads log q (score_q_on_identity.py)
and log p_ind on the 8,000 training rollouts, with the verifier labels, and computes
the statistics of the seed-23 analysis: Spearman and Pearson correlation, pooled and
within-prompt R^2 of log p_ind on log q, and the AUC of each score for correctness.
Reports every seed and the mean and SD over seeds, and samples scatter points evenly
from the three seeds for the figure.

Output: paper/figures/judge_vs_reward_data.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/path/to/iar_code")
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from q_vs_pown import Q_DIR, auc, average_ranks, demean, labels as seed23_labels, r2  # noqa: E402
from rollout_signal import read_jsonl  # noqa: E402

logger = logging.getLogger(__name__)
DOMAINS = ("commonsenseqa", "gsm8k", "kodcode", "spider")
SEEDS = ("seed23", "seed42", "seed7")
LABELS = ROOT / "results/revision_20260922/seed_rollout_labels"
POINTS_PER_SEED = 834
STATS = ("spearman", "pearson", "r2_pooled", "r2_within_prompt", "auc_pown", "auc_q")


def load(domain: str, seed: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """log p_ind, log q, correctness and prompt-visit group of every rollout, by call id."""
    suffix = "" if seed == "seed23" else f"_{seed}"
    rows = sorted(read_jsonl(Q_DIR / f"q_rows_{domain}{suffix}.jsonl"), key=lambda r: r["identity_call_id"])
    assert len(rows) == 8000, (domain, seed, len(rows))
    if seed == "seed23":
        lab = seed23_labels(domain)
    else:
        lab = {r["call_id"]: int(r["correct"]) for r in read_jsonl(LABELS / f"rows_{domain}_{seed}.jsonl")}
    visits: dict[str, int] = defaultdict(int)
    groups = []
    for r in rows:
        if r["rollout_index"] == 0:
            visits[r["task_id"]] += 1
        groups.append(f'{r["task_id"]}#{visits[r["task_id"]]}')
    return (np.array([r["log_p_own"] for r in rows]), np.array([r["log_q"] for r in rows]),
            np.array([lab[r["identity_call_id"]] for r in rows], dtype=float), np.array(groups))


def seed_stats(lp: np.ndarray, lq: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    rank = average_ranks
    return {"spearman": float(np.corrcoef(rank(lp), rank(lq))[0, 1]),
            "pearson": float(np.corrcoef(np.exp(lp), np.exp(lq))[0, 1]),
            "r2_pooled": r2(lq, lp), "r2_within_prompt": r2(demean(lq, groups), demean(lp, groups)),
            "auc_pown": auc(y, lp), "auc_q": auc(y, lq)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "paper/figures/judge_vs_reward_data.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(23)
    out = {"seeds": list(SEEDS), "n_plotted": POINTS_PER_SEED * len(SEEDS), "domains": {}}
    for domain in DOMAINS:
        per_seed, points = {}, []
        for seed in SEEDS:
            lp, lq, y, groups = load(domain, seed)
            per_seed[seed] = seed_stats(lp, lq, y, groups)
            pick = rng.choice(len(lp), POINTS_PER_SEED, replace=False)
            points += [[round(float(lp[i]), 4), round(float(lq[i]), 4), int(y[i])] for i in pick]
            logger.info("%s %s %s", domain, seed, {k: round(v, 3) for k, v in per_seed[seed].items()})
        entry = {"per_seed": per_seed, "points": points}
        for stat in STATS:
            values = [per_seed[s][stat] for s in SEEDS]
            entry[stat] = float(np.mean(values))
            entry[f"{stat}_sd"] = float(np.std(values, ddof=1))
        out["domains"][domain] = entry
    args.output.write_text(json.dumps(out) + "\n", encoding="utf-8")
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
