"""Held-out accuracy of the K=10 GPT-5 Mini runs against A0 (seed 23, one run per domain).

Pairs every held-out item of the terminal K=10 checkpoint with the same item of A0
by id, and reports accuracy, valid-output share, and an exact two-sided McNemar
test on the discordant pairs. KodCode items whose reference solution fails its
own tests are excluded, as in the main evaluation. Domains whose evaluation has not finished are
skipped.

Output: results/revision_20260925/k10_heldout_stats.json
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
from math import comb
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path("/path/to/iar_code")
CAMPAIGN = Path("/path/to/runs")
TEST_FILES = ROOT / "results/revision_20260922/test_files.json"
K10 = {"commonsenseqa": CAMPAIGN / "17_gpt5mini_k10_multidomain/commonsenseqa",
       "gsm8k": CAMPAIGN / "17_gpt5mini_k10_multidomain/gsm8k",
       "kodcode": CAMPAIGN / "16_kodcode_gpt5mini_k10",
       "spider": CAMPAIGN / "17_gpt5mini_k10_multidomain/spider"}
EVAL = "test_evaluation/step2000_attempt001/scored_candidates.jsonl"
FINAL_ATTEMPT = {"commonsenseqa": "attempt002", "gsm8k": "attempt004", "kodcode": "attempt010",
                 "spider": "attempt002"}
WINDOW = 1000


def training_stats(domain: str) -> dict[str, float]:
    """Fooling rate (m > K/2) and constant-group share in the first and last 1,000 rollouts."""
    path = K10[domain] / "training/runs" / FINAL_ATTEMPT[domain] / "reward_calls/reward_groups.jsonl"
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[int(row["logical_reward_id"])] = (int(row["m"]), int(row["k"]))
    m = [rows[i] for i in sorted(rows)]
    assert len(m) == 8000, domain

    def window(part: list) -> dict[str, float]:
        groups = [part[i:i + 4] for i in range(0, len(part), 4)]
        return {"fooling_pct": 100 * sum(v > k / 2 for v, k in part) / len(part),
                "mean_m": sum(v for v, _ in part) / len(part),
                "constant_group_pct": 100 * sum(len({v for v, _ in g}) == 1 for g in groups) / len(groups)}
    return {"first": window(m[:WINDOW]), "last": window(m[-WINDOW:]), "all": window(m)}


def read_scores(path: Path) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            score = row["score"]
            score = ast.literal_eval(score) if isinstance(score, str) else score
            if not score.get("excluded_reference_invalid"):
                out[row["id"]] = score
    return out


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/revision_20260925/k10_heldout_stats.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    files = json.loads(TEST_FILES.read_text(encoding="utf-8"))
    out = {}
    for domain, run in K10.items():
        path = run / EVAL
        if not path.exists():
            logger.info("%s: evaluation not finished", domain)
            continue
        k10, a0 = read_scores(path), read_scores(Path(files[domain]["a0"]))
        ids = k10.keys() & a0.keys()
        assert len(ids) == len(k10), domain
        a0 = {i: a0[i] for i in ids}
        b = sum(k10[i]["correct"] and not a0[i]["correct"] for i in k10)
        c = sum(a0[i]["correct"] and not k10[i]["correct"] for i in k10)
        n = len(k10)
        acc = lambda s: 100 * sum(bool(v["correct"]) for v in s.values()) / n
        valid = lambda s: 100 * sum(bool(v.get("valid")) for v in s.values()) / n
        out[domain] = {"n": n, "k10_pct": acc(k10), "a0_pct": acc(a0), "gain_pp": acc(k10) - acc(a0),
                       "k10_valid_pct": valid(k10), "a0_valid_pct": valid(a0),
                       "k10_only_correct": b, "a0_only_correct": c, "mcnemar_p": mcnemar_exact(b, c)}
        out[domain]["training"] = training_stats(domain)
        logger.info("%s %s", domain, out[domain])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
