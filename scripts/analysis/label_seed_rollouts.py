"""Verifier labels for the Identity training rollouts of seeds 42 and 7 (and a seed-23 check).

Uses ``_score_payload`` from the frozen reward-alignment source that produced the
seed-23 labels, so all seeds are labelled by the same code. Training data only;
no official test file is read and no model or API is called.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path("/path/to/iar_code")
VIEW = ROOT / "experiments/main/grpo_identity"
DATA = Path("/path/to/runs/"
            "evaluation_audits_and_reproducibility/01_data_protocol_runtime_gates/datasets")
DATASETS = {
    "commonsenseqa": DATA / "commonsenseqa/frozen_historical_seed23/train.jsonl",
    "gsm8k": DATA / "gsm8k/frozen_historical_seed23/official_train.jsonl",
    "kodcode": DATA / "kodcode/frozen_seed23_v1/train_core1000.jsonl",
    "spider": DATA / "spider/frozen_seed23_v1/train_core1000.jsonl",
}
SPIDER_DB = "/path/to/datasets/spider-release-1403EGqz/spider_data/database"
CHECK_ROWS = 200


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def reward_log(domain: str, seed: str) -> Path:
    for path in reversed(sorted(glob.glob(str(VIEW / domain / seed / "runs/*/reward_calls.jsonl")))):
        if sum(1 for r in read_jsonl(path) if r.get("status") == "ok") == 8000:
            return Path(path)
    raise FileNotFoundError(f"no complete reward log for {domain}/{seed}")


def candidate_text(row: dict[str, Any]) -> str:
    for field in ("candidate_raw", "candidate", "response"):
        if field in row:
            return str(row[field])
    raise KeyError("reward row has no candidate text")


def label(domain: str, rows: list[dict[str, Any]], workers: int, timeout: int) -> list[dict[str, Any]]:
    from iar_campaign.reward_alignment import _score_payload

    examples = {str(r["id"]): r for r in read_jsonl(DATASETS[domain])}
    keys, payloads, index = [], [], {}
    for row in rows:
        cand = candidate_text(row)
        key = (str(row["task_id"]), hashlib.sha256(cand.encode()).hexdigest())
        if key not in index:
            index[key] = len(payloads)
            payloads.append((domain, examples[key[0]], cand, timeout,
                             SPIDER_DB if domain == "spider" else None))
        keys.append(index[key])
    if domain in ("commonsenseqa", "gsm8k"):
        scores = [_score_payload(p) for p in payloads]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            scores = list(pool.map(_score_payload, payloads, chunksize=4))
    out = []
    for row, k in zip(rows, keys, strict=True):
        s = scores[k]
        out.append({"call_id": int(row["call_id"]), "task_id": str(row["task_id"]),
                    "dataset_index": int(row["dataset_index"]), "rollout_index": int(row["rollout_index"]),
                    "p_own": float(row["p_own"]), "log_p_own": float(row["log_p_own"]),
                    "correct": int(bool(s.get("correct"))), "evaluator_valid": int(bool(s.get("valid", True)))})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=sorted(DATASETS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--seeds", nargs="+", default=["seed42", "seed7"])
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_check:
        archived = None
    # Sanity check: re-label a fixed sample of seed-23 rollouts and compare with the archive.
    archived = None if args.skip_check else {r["call_id"]: r for r in read_jsonl(
        Path("/path/to/runs/evaluation_audits_and_reproducibility/"
             "12_final_test_analysis_reconciliation/correlations/nonofficial_reward_alignment_attempt001")
        / f"rows_{args.domain}_p_own.jsonl")}
    if archived is not None:
        s23 = [r for r in read_jsonl(reward_log(args.domain, "seed23")) if r.get("status") == "ok"]
        sample = random.Random(23).sample(s23, CHECK_ROWS)
        check = label(args.domain, sample, args.workers, args.timeout)
        agree = sum(int(c["correct"] == archived[c["call_id"]]["correct"]) for c in check)
        logger.info("%s seed23 check: %d/%d labels agree", args.domain, agree, CHECK_ROWS)
        (args.output_dir / f"check_{args.domain}_seed23.json").write_text(
            json.dumps({"agree": agree, "n": CHECK_ROWS}))

    for seed in args.seeds:
        path = reward_log(args.domain, seed)
        rows = sorted((r for r in read_jsonl(path) if r.get("status") == "ok"), key=lambda r: r["call_id"])
        labelled = label(args.domain, rows, args.workers, args.timeout)
        with open(args.output_dir / f"rows_{args.domain}_{seed}.jsonl", "w", encoding="utf-8") as handle:
            for row in labelled:
                handle.write(json.dumps(row) + "\n")
        logger.info("%s %s: %d rows, correct rate %.3f (%s)", args.domain, seed, len(labelled),
                    sum(r["correct"] for r in labelled) / len(labelled), path)


if __name__ == "__main__":
    main()
