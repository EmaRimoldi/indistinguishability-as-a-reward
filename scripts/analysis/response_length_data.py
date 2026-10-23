#!/usr/bin/env python3
"""Collect mean response length during training, per reward, domain and seed.

Reads only the trainer log histories of the completed runs; no model calls, no
test data. Writes the compact file the main-paper figure plots.
"""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path
from statistics import mean

logger = logging.getLogger(__name__)

CAMPAIGN = Path("/path/to/runs")
MAIN = Path("/path/to/iar_code/experiments/main")
BASELINES = CAMPAIGN / "baseline_seeds_20260922"
OUTPUT = Path("/path/to/iar_code/paper/"
              "figures/response_length_data.json")

DOMAINS = ("commonsenseqa", "gsm8k", "kodcode", "spider")
SEEDS = ("seed23", "seed42", "seed7")
CONDITIONS = ("grpo_identity", "grpo_quality", "grpo_oracle")
MAX_STEPS = 2000
BIN = 200


def log_paths(condition: str, domain: str, seed: str) -> list[str]:
    """All attempts of one run; a resumed run splits its history over several."""
    if seed == "seed23":
        root = MAIN / condition / domain / "seed23" / "runs"
    elif condition == "grpo_identity":
        base = CAMPAIGN / "identity_seed42" / ("domains" if seed == "seed42" else "seed7/domains")
        root = base / domain / "grpo" / "runs"
    else:
        root = BASELINES / domain / condition / seed / "runs"
    return sorted(glob.glob(str(root / "*" / "trainer_log_history.jsonl")))


def lengths_by_step(paths: list[str]) -> dict[int, float]:
    steps: dict[int, float] = {}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "completions/mean_length" in row:
                    steps[int(row["step"])] = float(row["completions/mean_length"])
    return steps


def binned(steps: dict[int, float]) -> list[float]:
    """Mean length over consecutive BIN-update bins; the x axis is the bin end."""
    out = []
    for start in range(0, MAX_STEPS, BIN):
        values = [v for s, v in steps.items() if start < s <= start + BIN]
        if not values:
            raise RuntimeError(f"no logged updates in ({start}, {start + BIN}]")
        out.append(mean(values))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    data = {"bin_updates": BIN,
            "bin_end_steps": list(range(BIN, MAX_STEPS + 1, BIN)),
            "conditions": {}}
    for condition in CONDITIONS:
        data["conditions"][condition] = {}
        for domain in DOMAINS:
            per_seed = {}
            for seed in SEEDS:
                paths = log_paths(condition, domain, seed)
                if not paths:
                    continue
                per_seed[seed] = binned(lengths_by_step(paths))
            if not per_seed:
                raise RuntimeError(f"no runs for {condition}/{domain}")
            data["conditions"][condition][domain] = per_seed
            logger.info("%s %s seeds=%s", condition, domain, sorted(per_seed))
    OUTPUT.write_text(json.dumps(data, indent=1) + "\n")
    logger.info("wrote %s", OUTPUT)


if __name__ == "__main__":
    main()
