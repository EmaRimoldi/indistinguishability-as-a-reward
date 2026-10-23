"""Additional training seeds for the SFT and GRPO-oracle baselines (revision 2026-09-22).

Replicates the seed-23 campaign runs exactly, changing only the training seed and the
output location. Configs are the frozen seed-23 configs with campaign directory names
mapped to the reorganized layout; hyperparameters and data files are verified against
the originals. The frozen campaign functions (``run_training``, ``LocalOracleReward``,
the Step-7 SFT recipe) are imported from the identity-seed replication source.

Subcommands:
  prepare     write per-(domain, arm, seed) configs
  train-oracle --domain D --seed S
  train-sft    --domain D --seed S
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

CAMPAIGN = Path("/path/to/runs")
VIEW = Path("/path/to/iar_code/experiments/main")
OUT = CAMPAIGN / "baseline_seeds_20260922"
DOMAINS = ("commonsenseqa", "gsm8k", "kodcode", "spider")
RENAMES = {
    "/07_baseline_suite": "/tier1_baseline_suite",
    "/05_four_domain_screen_1p5b": "/tier1_four_domain_screen_1p5b",
    "/15_four_domain_baseline_extension": "/tier1_four_domain_baseline_extension",
    "/04_reward_transform_selection": "/tier3_reward_transform_selection",
    "/01_data_protocol_runtime_gates": "/evaluation_audits_and_reproducibility/01_data_protocol_runtime_gates",
    "/06_two_domain_selection": "/evaluation_audits_and_reproducibility/06_two_domain_selection",
    "/03_judge_sanity": "/evaluation_audits_and_reproducibility/03_judge_sanity",
    "/12_final_test_analysis_reconciliation": "/evaluation_audits_and_reproducibility/12_final_test_analysis_reconciliation",
}
ORACLE_CONCURRENCY = 16
KEY_POOL = Path("/path/to/iar_code/keys/key_pool.env")


def remap(value: Any) -> Any:
    """Map old campaign directory names to the reorganized layout, recursively."""
    if isinstance(value, dict):
        return {k: remap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [remap(v) for v in value]
    if isinstance(value, str) and str(CAMPAIGN) in value and not Path(value).exists():
        for old, new in RENAMES.items():
            if old in value:
                candidate = value.replace(old, new)
                if Path(candidate).exists():
                    return candidate
    return value


def run_dirs(domain: str, arm: str, seed: int) -> tuple[Path, Path, Path]:
    base = OUT / domain / arm / f"seed{seed}"
    return base / "configs", base / "runs", base / "checkpoints"


def prepare(seeds: list[int], arm: str = "grpo_oracle") -> None:
    for domain in DOMAINS:
        src = VIEW / arm / domain / "seed23" / "runs" / "attempt001" / "config.yaml"
        raw = yaml.safe_load(src.read_text())
        for seed in seeds:
            cfg = remap(json.loads(json.dumps(raw)))
            configs, runs, ckpts = run_dirs(domain, arm, seed)
            cfg["project"]["seed"] = seed
            cfg["trial"]["run_name"] = f"revision_{domain}_{arm}_seed{seed}"
            cfg["paths"]["experiments"] = str(runs)
            cfg["paths"]["checkpoints"] = str(ckpts)
            configs.mkdir(parents=True, exist_ok=True)
            (configs / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            logger.info("prepared %s", configs / "config.yaml")


def train_oracle(domain: str, seed: int, run_id: str) -> None:
    from iar_campaign.baseline_rewards import LocalOracleReward
    from iar_campaign.generate import prompts_for_domain, read_jsonl
    from iar_campaign.reward_pilot import sha256_file, utc_now, write_json, write_jsonl
    from iar_grpo.config import load_grpo_config
    from iar_grpo.train import run_training
    from datasets import Dataset

    configs, runs, ckpts = run_dirs(domain, "grpo_oracle", seed)
    config_path = configs / "config.yaml"
    config = load_grpo_config(config_path)
    raw = config.raw
    ref = yaml.safe_load((VIEW / "grpo_oracle" / domain / "seed23" / "runs" / "attempt001" / "config.yaml").read_text())
    assert raw["training"] == ref["training"] and raw["reward"] == ref["reward"], "hyperparameters drifted"
    assert raw["reward"]["value"] == "oracle_task_utility"
    train_file = Path(raw["dataset"]["split_file"])
    assert sha256_file(train_file) == raw["dataset"]["split_file_sha256"], "training split drifted"
    examples = read_jsonl(train_file)
    assert len(examples) == 1000
    tables = Path(raw["evaluation"]["spider_tables_json"]) if domain == "spider" else None
    prompts = prompts_for_domain(domain, examples, tables_json=tables)["a0"]
    rows = [{"prompt": [{"role": "user", "content": p}], "task_statement": p,
             "task_id": str(e["id"]), "dataset_index": int(e["source_index"])}
            for e, p in zip(examples, prompts, strict=True)]
    experiment_dir, checkpoint_dir = runs / run_id, ckpts / run_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    (experiment_dir / "config.yaml").write_text(config_path.read_text())
    write_jsonl(experiment_dir / "train_examples.jsonl", rows)
    runtime = LocalOracleReward(
        domain=domain, examples=examples, log_path=experiment_dir / "reward_calls.jsonl",
        max_reward_calls=8000,
        kodcode_timeout_seconds=int(raw["evaluation"].get("coding_timeout_seconds", 60)),
        spider_database_root=Path(raw["evaluation"]["spider_database_root"]) if domain == "spider" else None,
        max_concurrency=ORACLE_CONCURRENCY,
    )
    result, history = run_training(config, Dataset.from_list(rows), checkpoint_dir, experiment_dir, runtime)
    runtime.assert_complete()
    write_jsonl(experiment_dir / "trainer_log_history.jsonl", history)
    grads = [float(r["grad_norm"]) for r in history if "grad_norm" in r]
    assert grads and all(math.isfinite(g) for g in grads)
    write_json(experiment_dir / "summary.json", {
        "schema_version": 1, "status": "complete", "completed_at": utc_now(), "run_id": run_id,
        "domain": domain, "arm": "grpo_oracle", "seed": seed, "reward_audit": runtime.audit_summary(),
        "trainer_metrics": result.metrics, "checkpoint_dir": str(checkpoint_dir),
        "oracle_concurrency": ORACLE_CONCURRENCY, "official_test_accessed": False})


def train_quality(domain: str, seed: int, run_id: str, key_name: str,
                  resume_from: Path | None = None) -> None:
    """GRPO with the reference-free teacher correctness judge q (frozen Step-7 client).

    ``resume_from`` continues an interrupted run from one of its saved checkpoints.
    The optimizer, scheduler and RNG state come from that checkpoint, so the
    remaining updates follow the schedule the interrupted run would have taken;
    only the reward calls of those remaining updates are issued and logged here.
    """
    from iar_campaign.baseline_rewards import quality_prompt, quality_prompt_template_sha256
    from iar_campaign.generate import prompts_for_domain, read_jsonl
    from iar_campaign.reward_pilot import load_named_secret, sha256_file, utc_now, write_json, write_jsonl
    from iar_grpo.config import load_grpo_config
    from iar_grpo.reward import LiveOpenRouterReward, settings_from_config
    from iar_grpo.train import run_training
    from iar_phase0.generator import strip_code_fences
    from datasets import Dataset

    configs, runs, ckpts = run_dirs(domain, "grpo_quality", seed)
    config_path = configs / "config.yaml"
    config = load_grpo_config(config_path)
    raw = config.raw
    ref = yaml.safe_load((VIEW / "grpo_quality" / domain / "seed23" / "runs" / "attempt001" / "config.yaml").read_text())
    assert raw["training"] == ref["training"] and raw["reward"] == ref["reward"], "hyperparameters drifted"
    assert raw["discriminator"] == ref["discriminator"], "judge settings drifted"
    assert raw["discriminator"]["attribution_prompt_template_sha256"] == quality_prompt_template_sha256()
    train_file = Path(raw["dataset"]["split_file"])
    assert sha256_file(train_file) == raw["dataset"]["split_file_sha256"], "training split drifted"
    examples = read_jsonl(train_file)
    tables = Path(raw["evaluation"]["spider_tables_json"]) if domain == "spider" else None
    prompts = prompts_for_domain(domain, examples, tables_json=tables)["a0"]
    rows = [{"prompt": [{"role": "user", "content": p}], "task_statement": p,
             "task_id": str(e["id"]), "dataset_index": int(e["source_index"])}
            for e, p in zip(examples, prompts, strict=True)]
    resumed_step = 0
    if resume_from is not None:
        resumed_step = int(json.loads((resume_from / "trainer_state.json").read_text())["global_step"])
        assert 0 < resumed_step < int(raw["training"]["max_steps"])
    experiment_dir, checkpoint_dir = runs / run_id, ckpts / run_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    (experiment_dir / "config.yaml").write_text(config_path.read_text())
    write_jsonl(experiment_dir / "train_examples.jsonl", rows)
    runtime = LiveOpenRouterReward(
        settings=settings_from_config(raw), log_path=experiment_dir / "reward_calls.jsonl",
        api_key=load_named_secret(KEY_POOL, key_name), prompt_builder=quality_prompt,
        prompt_template_hash=quality_prompt_template_sha256(), system_prompt_builder=lambda _n: None,
        candidate_normalizer=strip_code_fences if domain == "kodcode" else str.strip,
        # Continue the call numbering of the interrupted run so the two logs concatenate.
        call_id_offset=resumed_step * int(raw["training"]["num_generations"]),
        http_request_id_offset=resumed_step * int(raw["training"]["num_generations"]))
    result, history = run_training(config, Dataset.from_list(rows), checkpoint_dir, experiment_dir,
                                   runtime, resume_from_checkpoint=resume_from)
    if resume_from is None:
        runtime.assert_complete()
    else:
        # The judge is queried once per rollout of the updates that still had to run.
        expected = ((int(raw["training"]["max_steps"]) - resumed_step)
                    * int(raw["training"]["num_generations"]))
        observed = runtime.audit_summary()["logical_reward_calls"]
        assert observed == expected, f"expected {expected} rewards after resume, observed {observed}"
    write_jsonl(experiment_dir / "trainer_log_history.jsonl", history)
    grads = [float(r["grad_norm"]) for r in history if "grad_norm" in r]
    assert grads and all(math.isfinite(g) for g in grads)
    write_json(experiment_dir / "summary.json", {
        "schema_version": 1, "status": "complete", "completed_at": utc_now(), "run_id": run_id,
        "domain": domain, "arm": "grpo_quality", "seed": seed, "key_name": key_name,
        "resumed_from": str(resume_from) if resume_from is not None else None,
        "resumed_step": resumed_step,
        "reward_audit": runtime.audit_summary(), "trainer_metrics": result.metrics,
        "checkpoint_dir": str(checkpoint_dir), "official_test_accessed": False})


def teacher_corpus(domain: str) -> Path:
    runs = sorted((VIEW / "sft_teacher" / domain / "seed23" / "runs").glob("teacher_attempt*/summary.json"))
    summary = json.loads(runs[-1].read_text())
    path = Path(remap(summary["sft_all"]))
    return path


def train_sft(domain: str, seed: int, run_id: str, cache_dir: Path) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from iar_campaign.generate import read_jsonl
    from iar_campaign.reward_pilot import sha256_file, utc_now, write_json
    from iar_campaign.sft_baseline import ACTOR_REVISION, _actor_source, _project_rows
    from iar_grpo.numerics import configure_stable_sdpa

    ref_dir = sorted((VIEW / "sft_teacher" / domain / "seed23" / "sft_runs").glob("attempt*"))[-1]
    ref_args = json.loads((ref_dir / "parameters" / "trainer_arguments.json").read_text())
    corpus = teacher_corpus(domain)
    rows = _project_rows(read_jsonl(corpus))
    assert len(rows) == 1000
    _, runs, ckpts = run_dirs(domain, "sft_teacher", seed)
    experiment_dir, checkpoint_dir = runs / run_id, ckpts / run_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    sdpa = configure_stable_sdpa(torch)
    source = _actor_source(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False,
                                              padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    training = SFTConfig(
        output_dir=str(checkpoint_dir), run_name=f"revision_{domain}_sft_seed{seed}",
        seed=seed, data_seed=seed, learning_rate=1e-4, warmup_ratio=0.0, lr_scheduler_type="constant",
        per_device_train_batch_size=1, gradient_accumulation_steps=4, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, num_train_epochs=2.0, max_steps=500,
        logging_steps=1, save_strategy="steps", save_steps=125, save_total_limit=None,
        eval_strategy="no", report_to="none", bf16=True, max_length=int(ref_args["max_length"]),
        completion_only_loss=True, shuffle_dataset=False, packing=False,
        include_num_input_tokens_seen=True, remove_unused_columns=True,
        model_init_kwargs={"cache_dir": str(cache_dir), "local_files_only": True,
                           "trust_remote_code": False, "torch_dtype": torch.bfloat16})
    for key in ("learning_rate", "max_steps", "per_device_train_batch_size", "gradient_accumulation_steps",
                "max_length", "save_steps", "shuffle_dataset", "completion_only_loss"):
        assert getattr(training, key) == ref_args[key], f"SFT {key} drifted"
    lora = LoraConfig(r=32, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    trainer = SFTTrainer(model=str(source), args=training, train_dataset=Dataset.from_list(rows),
                         processing_class=tokenizer, peft_config=lora)
    started = time.time()
    result = trainer.train()
    trainer.save_model(str(checkpoint_dir / "final_adapter"))
    history = list(trainer.state.log_history)
    write_json(experiment_dir / "trainer_log_history.json", history)
    write_json(experiment_dir / "summary.json", {
        "schema_version": 1, "status": "complete", "completed_at": utc_now(), "domain": domain,
        "run_id": run_id, "seed": seed, "actor_revision": ACTOR_REVISION, "sft_corpus": str(corpus),
        "sft_corpus_sha256": sha256_file(corpus), "global_step": int(trainer.state.global_step),
        "train_runtime_seconds": time.time() - started, "trainer_metrics": dict(result.metrics),
        "sdpa": sdpa, "checkpoint_dir": str(checkpoint_dir), "official_test_accessed": False})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 7])
    p.add_argument("--arm", default="grpo_oracle", choices=("grpo_oracle", "grpo_quality"))
    for name in ("train-oracle", "train-sft", "train-quality"):
        t = sub.add_parser(name)
        t.add_argument("--domain", choices=DOMAINS, required=True)
        t.add_argument("--seed", type=int, required=True)
        t.add_argument("--run-id", default="attempt001")
        t.add_argument("--key-name", default="")
        t.add_argument("--resume-from", type=Path, default=None,
                       help="checkpoint directory of an interrupted run to continue")
        t.add_argument("--cache-dir", type=Path, default=Path("/path/to/cache/huggingface"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.command == "prepare":
        prepare(args.seeds, args.arm)
    elif args.command == "train-quality":
        train_quality(args.domain, args.seed, args.run_id, args.key_name, args.resume_from)
    elif args.command == "train-oracle":
        train_oracle(args.domain, args.seed, args.run_id)
    else:
        train_sft(args.domain, args.seed, args.run_id, args.cache_dir)


if __name__ == "__main__":
    main()
