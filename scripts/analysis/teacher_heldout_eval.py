"""Held-out accuracy of the teacher, DeepSeek V4 Flash, for the capability gap gamma_mu.

The teacher answers the held-out test items with the protocol of its SFT corpus
(StreamLake, temperature 0, top-p 1, reasoning off, at most 768 tokens, the A0
prompt of each domain), and its responses are scored exactly as the actor's are
by tools/heldout_eval.py: the stripped response goes to the same scorers, with the
same KodCode/Spider reference gate. Responses are appended to responses.jsonl as they arrive, so an interrupted
run resumes without repeating calls.

Outputs in --output-dir: responses.jsonl, scored_candidates.jsonl, summary.json.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import threading
import time
from collections import Counter
from pathlib import Path

import httpx

from iar_campaign.generate import prompts_for_domain, read_jsonl
from iar_campaign.reward_pilot import load_named_secret, sha256_file, utc_now, write_json, write_jsonl
from iar_campaign.score_baselines import _score_kodcode, _score_simple, _score_spider_one
from iar_grpo.config import load_grpo_config
from iar_phase0.openrouter import OPENROUTER_URL, make_chat_payload, response_text

logger = logging.getLogger(__name__)
MODEL_ID = "deepseek/deepseek-v4-flash"
PROVIDER = "StreamLake"
ROUTING = {"order": [PROVIDER], "allow_fallbacks": False, "require_parameters": True}
MAX_TOKENS = 768
MAX_ATTEMPTS = 8
KEY_POOL = Path("/path/to/iar_code/keys/key_pool.env")


def valid_ids(domain: str, gate_path: Path | None, test_sha256: str) -> set[str] | None:
    """Reference-gate IDs, as in tools/heldout_eval.py (_valid_ids)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("iar_heldout_gate", Path(__file__).resolve().parents[2] / "scripts/tools/heldout_eval.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    return evaluator._valid_ids(domain=domain, gate_path=gate_path, test_sha256=test_sha256)


def ask(client: httpx.Client, headers: dict, prompt: str) -> dict:
    payload = make_chat_payload(model_id=MODEL_ID, prompt=prompt, system_prompt=None,
                                max_tokens=MAX_TOKENS, temperature=0, provider_routing=dict(ROUTING),
                                request_overrides={"top_p": 1.0, "reasoning": {"enabled": False}})
    last = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.post(OPENROUTER_URL, headers=headers, json=payload)
            response.raise_for_status()
            raw = response.json()
            if raw.get("provider") != PROVIDER or not raw.get("choices"):
                raise RuntimeError(f"unexpected response from {raw.get('provider')}")
            return raw
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            last = exc
            time.sleep(min(120, 5 * 2 ** (attempt - 1)))
    raise RuntimeError(f"teacher request failed after {MAX_ATTEMPTS} attempts: {last}")


def generate(domain: str, examples: list, prompts: list, out: Path, key_name: str, workers: int) -> dict:
    path = out / "responses.jsonl"
    done = {r["id"]: r for r in read_jsonl(path)} if path.exists() else {}
    todo = [(e, p) for e, p in zip(examples, prompts, strict=True) if str(e["id"]) not in done]
    logger.info("%s: %d cached, %d to generate", domain, len(done), len(todo))
    headers = {"Authorization": f"Bearer {load_named_secret(KEY_POOL, key_name)}",
               "Content-Type": "application/json"}
    lock = threading.Lock()
    with httpx.Client(timeout=120) as client, open(path, "a", encoding="utf-8") as sink:
        def one(item):
            example, prompt = item
            raw = ask(client, headers, prompt)
            text = response_text(raw) or ""
            row = {"id": str(example["id"]), "source_index": int(example["source_index"]), "domain": domain,
                   "response": text.strip(),
                   "finish_reason": raw["choices"][0].get("finish_reason"),
                   "completion_tokens": (raw.get("usage") or {}).get("completion_tokens"),
                   "cost": (raw.get("usage") or {}).get("cost"), "raw_provider": raw.get("provider")}
            with lock:
                sink.write(json.dumps(row) + "\n")
                sink.flush()
                done[row["id"]] = row
                if len(done) % 200 == 0:
                    logger.info("%s: %d/%d", domain, len(done), len(examples))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, todo))
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Any GRPO config of the domain")
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--reference-gate", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--key-name", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = load_grpo_config(args.config.resolve())
    domain = str(config.raw["campaign"]["domain"])
    examples = read_jsonl(args.test_file)
    test_sha256 = sha256_file(args.test_file)
    tables = Path(config.raw["evaluation"]["spider_tables_json"]) if domain == "spider" else None
    prompts = prompts_for_domain(domain, examples, tables_json=tables)["a0"]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    keep = valid_ids(domain, args.reference_gate, test_sha256)

    done = generate(domain, examples, prompts, out, args.key_name, args.workers)
    candidates = [{k: v for k, v in done[str(e["id"])].items()} for e in examples]
    for row in candidates:
        row["max_new_tokens"] = MAX_TOKENS
        row["hit_max_new_tokens"] = row.get("finish_reason") == "length"
    by_id = {str(e["id"]): e for e in examples}
    scored = candidates if keep is None else [r for r in candidates if r["id"] in keep]
    if domain in {"commonsenseqa", "gsm8k"}:
        _score_simple(domain, examples, candidates)
    elif domain == "kodcode":
        os.environ.setdefault("IAR_KODCODE_TEST_PYTHONPATH", os.environ.get("PYTHONPATH", ""))
        payloads = [(str(r["response"]), str(by_id[r["id"]]["hidden_tests"]), 60) for r in scored]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for row, score in zip(scored, pool.map(_score_kodcode, payloads), strict=True):
                row["score"] = score
    else:
        root = Path(config.raw["evaluation"]["spider_database_root"])
        payloads = [(by_id[r["id"]], r, root) for r in scored]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for row, score in zip(scored, pool.map(_score_spider_one, payloads), strict=True):
                row.update(score)
    if keep is not None:
        for row in candidates:
            if row["id"] not in keep:
                row["score"] = {"correct": False, "valid": False, "excluded_reference_invalid": True}
    correct = sum(bool(r["score"]["correct"]) for r in scored)
    write_jsonl(out / "scored_candidates.jsonl", candidates)
    write_json(out / "summary.json", {
        "status": "complete", "completed_at": utc_now(), "domain": domain, "model_id": MODEL_ID,
        "provider": PROVIDER, "decoding": {"temperature": 0, "top_p": 1.0, "max_tokens": MAX_TOKENS},
        "n": len(candidates), "scored_n": len(scored), "correct": correct,
        "task_utility": correct / len(scored), "reference_invalid_excluded": len(candidates) - len(scored),
        "valid_output_n": sum(bool(r["score"].get("valid", True)) for r in scored),
        "status_counts": dict(Counter(str(r["score"].get("status", "scored")) for r in scored)),
        "truncated": sum(r["hit_max_new_tokens"] for r in candidates),
        "cost_usd": sum(float(r.get("cost") or 0) for r in candidates),
        "test_file": str(args.test_file), "test_sha256": test_sha256, "official_test_accessed": True})
    logger.info("%s: %d/%d correct (%.1f%%)", domain, correct, len(scored), 100 * correct / len(scored))


if __name__ == "__main__":
    main()
