"""Score the Identity training rollouts of one seed with the Quality judge q (revision 2026-09-22).

Every rollout of the seed-23 Identity run is sent, with its task statement, to the frozen
GRPO-quality judge: the same prompt, model, provider route, decoding settings and
candidate normalization as the seed-23 Quality training run of the same domain. Only
the rollouts differ. Each call is logged by the frozen client.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

VIEW = Path("/path/to/iar_code/experiments/main")
KEY_POOL = Path("/path/to/iar_code/keys/key_pool.env")
BATCH = 64


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--key-name", required=True)
    parser.add_argument("--seed", default="seed23", choices=("seed23", "seed42", "seed7"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0, help="score only the first N rollouts (smoke test)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from iar_campaign.baseline_rewards import quality_prompt, quality_prompt_template_sha256
    from iar_grpo.reward import LiveOpenRouterReward, settings_from_config
    from iar_phase0.generator import strip_code_fences
    from iar_campaign.reward_pilot import load_named_secret

    from iar_repro.configure import prepare
    from iar_repro.records import reward_log
    qcfg = prepare(args.domain, "quality", 23).raw
    raw = json.loads(json.dumps(qcfg))
    raw["reward"]["max_reward_calls"] = 9000
    raw["reward"]["max_http_requests"] = 20000
    raw["reward"]["max_concurrency"] = args.concurrency
    raw["reward"]["minimum_request_interval_seconds"] = 0.0
    settings = settings_from_config(raw)

    run_dir = reward_log("grpo_identity", args.domain, args.seed).parent
    suffix = "" if args.seed == "seed23" else f"_{args.seed}"
    tasks = {r["task_id"]: r["task_statement"] for r in read_jsonl(run_dir / "train_examples.jsonl")}
    rollouts = sorted((r for r in read_jsonl(run_dir / "reward_calls.jsonl") if r.get("status") == "ok"),
                      key=lambda r: r["call_id"])
    if args.limit:
        rollouts = rollouts[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    judge = LiveOpenRouterReward(
        settings=settings, log_path=args.output_dir / f"q_calls_{args.domain}{suffix}.jsonl",
        api_key=load_named_secret(KEY_POOL, args.key_name), prompt_builder=quality_prompt,
        prompt_template_hash=quality_prompt_template_sha256(), system_prompt_builder=lambda _n: None,
        candidate_normalizer=strip_code_fences if args.domain == "kodcode" else str.strip)
    out = []
    for start in range(0, len(rollouts), BATCH):
        batch = rollouts[start:start + BATCH]
        rewards = asyncio.run(judge.score_batch(
            completions=[r["candidate_raw"] for r in batch],
            task_statement=[tasks[r["task_id"]] for r in batch],
            task_id=[r["task_id"] for r in batch],
            dataset_index=[int(r["dataset_index"]) for r in batch],
            rollout_index=[int(r["rollout_index"]) for r in batch]))
        for r, reward in zip(batch, rewards, strict=True):
            out.append({"identity_call_id": r["call_id"], "task_id": r["task_id"],
                        "rollout_index": r["rollout_index"], "p_own": r["p_own"],
                        "log_p_own": r["log_p_own"], "log_q": float(reward)})
        logger.info("%s: %d/%d scored", args.domain, len(out), len(rollouts))
    with open(args.output_dir / f"q_rows_{args.domain}{suffix}.jsonl", "w", encoding="utf-8") as handle:
        for row in out:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
