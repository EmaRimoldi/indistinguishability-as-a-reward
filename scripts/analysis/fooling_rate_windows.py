"""Fooling rate of every Identity training run over all of its rollouts.

The fooling rate is the share of training rollouts that the teacher attributes to
itself, i.e. with p_ind above one half. Each run's 8,000 successful reward calls
are taken in call order and split into 20 consecutive windows of 400 rollouts
(100 optimizer updates each), the same windows used for the mean log p_ind in
training_trajectory_seeds.json. "start" and "end" keep the first and last window.

Output: paper/figures/fooling_rate_data.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rollout_signal import read_jsonl, reward_log

logger = logging.getLogger(__name__)

DOMAINS = ("commonsenseqa", "gsm8k", "kodcode", "spider")
SEEDS = ("seed23", "seed42", "seed7")
WINDOW = 400
WINDOWS = 20


def fooling_windows(domain: str, seed: str) -> list[float]:
    """Percentage of rollouts with p_ind > 1/2 in each 400-rollout window."""
    ok = sorted((r for r in read_jsonl(reward_log("grpo_identity", domain, seed))
                 if r.get("status") == "ok"), key=lambda r: r["call_id"])
    if len(ok) != WINDOW * WINDOWS:
        raise RuntimeError(f"{domain}/{seed}: expected {WINDOW * WINDOWS} calls, found {len(ok)}")
    return [100.0 * sum(r["p_own"] > 0.5 for r in ok[i * WINDOW:(i + 1) * WINDOW]) / WINDOW
            for i in range(WINDOWS)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    domains = {}
    for domain in DOMAINS:
        domains[domain] = {}
        for seed in SEEDS:
            windows = fooling_windows(domain, seed)
            domains[domain][seed] = {"start": windows[0], "end": windows[-1], "windows": windows}
            logger.info("%s %s: %.1f%% -> %.1f%%", domain, seed, windows[0], windows[-1])
    payload = {"window_rollouts": WINDOW, "window_end_steps": [100 * (i + 1) for i in range(WINDOWS)],
               "domains": domains}
    args.output.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
