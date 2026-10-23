#!/usr/bin/env python3
"""How often a reward separates a group, and how often it separates it correctly.

Two quantities per reward and domain:

* contrast   -- share of four-response groups whose rewards are not all equal,
                read from the trainer logs of the runs themselves
                (``frac_reward_zero_std``); a group with equal rewards has all
                four advantages zero and does not enter the update.
* direction  -- among the groups that hold both a correct and an incorrect
                response, the share in which the reward puts the correct ones
                above the incorrect ones. Chance is 0.5; the task verifier is
                1.0 by construction.

Direction for log p_ind is measured on the indistinguishability rollouts of each
seed. Direction for the judged reward q is measured on the same rollouts of the
three indistinguishability seeds, scored a second time by the Quality judge, so
it says how q ranks responses rather than what q's own policy produced.
Contrast uses only runs that reached the end of training.

Reads logged rollouts only. No model calls, no training, no test data.
"""

from __future__ import annotations

import glob
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

logger = logging.getLogger(__name__)

ROOT = Path("/path/to/iar_code")
CAMPAIGN = Path("/path/to/runs")
MAIN = ROOT / "experiments/main"
BASELINES = CAMPAIGN / "baseline_seeds_20260922"
ALIGN = (CAMPAIGN / "evaluation_audits_and_reproducibility"
         / "12_final_test_analysis_reconciliation/correlations"
         / "nonofficial_reward_alignment_attempt001")
LABELS = ROOT / "results/revision_20260922/seed_rollout_labels"
QROWS = ROOT / "results/revision_20260922/q_on_identity"
OUTPUT = ROOT / "paper/figures/reward_contrast_data.json"

DOMAINS = ("commonsenseqa", "gsm8k", "kodcode", "spider")
SEEDS = ("seed23", "seed42", "seed7")
GROUP = 4


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def trainer_logs(condition: str, domain: str, seed: str) -> list[str]:
    if seed == "seed23":
        root = MAIN / condition / domain / "seed23" / "runs"
    elif condition == "grpo_identity":
        base = CAMPAIGN / "identity_seed42" / ("domains" if seed == "seed42" else "seed7/domains")
        root = base / domain / "grpo" / "runs"
    else:
        root = BASELINES / domain / condition / seed / "runs"
    return sorted(glob.glob(str(root / "*" / "trainer_log_history.jsonl")))


def contrast_pct(condition: str, domain: str, seed: str) -> float | None:
    """Share of groups whose four rewards are not all equal, from the run's log."""
    steps: dict[int, float] = {}
    for path in trainer_logs(condition, domain, seed):
        for row in read_jsonl(Path(path)):
            if "frac_reward_zero_std" in row:
                steps[int(row["step"])] = float(row["frac_reward_zero_std"])
    if not steps or max(steps) < 1999:  # logged on odd steps up to 1999
        return None
    return 100.0 * (1.0 - mean(steps.values()))


def labelled_rows(domain: str, seed: str) -> list[dict]:
    """Rollouts with log p_ind and a correctness label, one run per seed."""
    path = LABELS / f"rows_{domain}_{seed}.jsonl"
    if path.exists():
        return read_jsonl(path)
    if seed != "seed23":
        raise FileNotFoundError(path)
    # The seed-23 alignment rows log the probability; the reward is its clipped log.
    epsilon = 1e-6
    rows = read_jsonl(ALIGN / f"rows_{domain}_p_own.jsonl")
    for row in rows:
        p_own = min(max(float(row["score_probability"]), epsilon), 1.0 - epsilon)
        row["log_p_own"] = math.log(p_own)
    return rows


def direction_pct(rows: list[dict], score_key: str, correct_key: str = "correct") -> tuple[float, int]:
    """Share of mixed groups in which the mean score of the correct responses wins."""
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(int(row["call_id"]) - 1) // GROUP].append(row)
    wins = mixed = 0
    for members in groups.values():
        if len(members) != GROUP:
            continue
        good = [float(r[score_key]) for r in members if int(r[correct_key]) == 1]
        bad = [float(r[score_key]) for r in members if int(r[correct_key]) == 0]
        if not good or not bad:
            continue
        mixed += 1
        wins += mean(good) > mean(bad)
    return 100.0 * wins / mixed, mixed


def q_rows_with_labels(domain: str, seed: str) -> list[dict]:
    """Join the judged reward q onto the Indist. rollouts of one seed it was scored on."""
    labels = {int(r["call_id"]): int(r["correct"]) for r in labelled_rows(domain, seed)}
    suffix = "" if seed == "seed23" else f"_{seed}"
    out = []
    for row in read_jsonl(QROWS / f"q_rows_{domain}{suffix}.jsonl"):
        call_id = int(row["identity_call_id"])
        if call_id in labels:
            out.append({"call_id": call_id, "log_q": float(row["log_q"]),
                        "correct": labels[call_id]})
    return out


def summarize(values: list[float]) -> dict[str, float | int]:
    return {"mean": mean(values), "sd": stdev(values) if len(values) > 1 else 0.0,
            "n": len(values)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    data: dict = {"groups_per_run": 2000, "domains": {}}
    for domain in DOMAINS:
        entry: dict = {}
        for condition in ("grpo_identity", "grpo_quality", "grpo_oracle"):
            values = [c for seed in SEEDS if (c := contrast_pct(condition, domain, seed)) is not None]
            entry[condition] = {"contrast": summarize(values)}
        directions = []
        for seed in SEEDS:
            pct, mixed = direction_pct(labelled_rows(domain, seed), "log_p_own")
            directions.append(pct)
            logger.info("%s %s direction %.1f%% over %d mixed groups", domain, seed, pct, mixed)
        entry["grpo_identity"]["direction"] = summarize(directions)
        q_directions = []
        for seed in SEEDS:
            q_pct, q_mixed = direction_pct(q_rows_with_labels(domain, seed), "log_q")
            q_directions.append(q_pct)
            logger.info("%s %s q direction %.1f%% over %d mixed groups", domain, seed, q_pct, q_mixed)
        entry["grpo_quality"]["direction"] = {**summarize(q_directions),
                                              "note": "q scored on the Indist. rollouts of each seed"}
        entry["grpo_oracle"]["direction"] = {"mean": 100.0, "sd": 0.0, "n": 3,
                                             "note": "the verifier defines correctness"}
        data["domains"][domain] = entry
        logger.info("%s contrast: %s", domain,
                    {c: round(entry[c]["contrast"]["mean"], 1) for c in entry})
    OUTPUT.write_text(json.dumps(data, indent=1) + "\n")
    logger.info("wrote %s", OUTPUT)


if __name__ == "__main__":
    main()
