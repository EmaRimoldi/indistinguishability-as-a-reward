"""Reward and correctness on the two visits to each training prompt, three Identity seeds.

For every Identity run, the rollouts of each prompt's first visit (epoch 1) are
compared with those of its second visit (epoch 2): change in mean p_ind and in
the share of verifier-correct rollouts, plus, on Spider, mean p_ind of correct
and incorrect rollouts on each visit.

Output: results/revision_20260924/repeated_prompts_three_seed.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from seed_figure_data import DOMAINS, SEEDS, labelled_rows

logger = logging.getLogger(__name__)


def visit_stats(rows: list[dict]) -> dict[str, float]:
    first = [r for r in rows if r["occurrence"] == 1]
    second = [r for r in rows if r["occurrence"] == 2]
    mean = lambda rs, key: float(np.mean([r[key] for r in rs]))
    by = lambda rs, c: float(np.mean([r["p_own"] for r in rs if r["correct"] == c]))
    return {"delta_p_own": mean(second, "p_own") - mean(first, "p_own"),
            "delta_correct_pp": 100 * (mean(second, "correct") - mean(first, "correct")),
            "p_own_incorrect": [by(first, 0), by(second, 0)],
            "p_own_correct": [by(first, 1), by(second, 1)],
            "rollouts": [len(first), len(second)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = {}
    for domain in DOMAINS:
        out[domain] = {seed: visit_stats(labelled_rows(domain, seed)) for seed in SEEDS}
        for seed, s in out[domain].items():
            logger.info("%s %s dp=%+.3f dcorrect=%+.1f n=%s", domain, seed, s["delta_p_own"],
                        s["delta_correct_pp"], s["rollouts"])
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
