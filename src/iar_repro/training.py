from __future__ import annotations

import importlib
import json
import os
import shutil
from pathlib import Path

from .configure import prepare
from .paths import ROOT, defaults, sha256


def train(
    domain: str,
    condition: str,
    seed: int,
    run_id: str,
    *,
    resume: Path | None = None,
    reward_form: str | None = None,
) -> dict:
    """Run the frozen trainer/reward implementations with portable, verified inputs."""
    from datasets import Dataset
    from iar_campaign.generate import prompts_for_domain, read_jsonl
    from iar_campaign.four_domain_screen import reward_runtime
    from iar_campaign.baseline_rewards import (
        LocalOracleReward,
        quality_prompt,
        quality_prompt_template_sha256,
    )
    from iar_grpo.reward import LiveOpenRouterReward, settings_from_config
    from iar_grpo.train import run_training
    from iar_phase0.generator import strip_code_fences
    from iar_phase0.parameters import initialize_parameter_record

    if condition == "reward_form_pilot":
        from .pilot import prepare as prepare_pilot

        config = prepare_pilot(domain, reward_form, run_id)
    else:
        config = prepare(domain, condition, seed, run_id=run_id)
    raw = config.raw
    if resume and condition in ("oracle", "reward_form_pilot"):
        raise ValueError("Start a fresh immutable attempt for this condition")
    if condition != "oracle" and not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise ValueError("Export OPENROUTER_API_KEY before starting paid training")
    config.actor_source()  # Fail before creating an attempt if weights are absent.
    examples = read_jsonl(Path(raw["dataset"]["split_file"]))
    if (
        len(examples) != config.prompt_count
        or len({str(r["id"]) for r in examples}) != config.prompt_count
    ):
        raise ValueError("Training requires the complete unique frozen prompt pool")
    tables = (
        Path(raw["evaluation"]["spider_tables_json"]) if domain == "spider" else None
    )
    prompts = prompts_for_domain(domain, examples, tables_json=tables)["a0"]
    rows = [
        {
            "prompt": [{"role": "user", "content": prompt}],
            "task_statement": prompt,
            "task_id": str(example["id"]),
            "dataset_index": int(example["source_index"]),
        }
        for example, prompt in zip(examples, prompts, strict=True)
    ]
    run = Path(raw["paths"]["experiments"]) / run_id
    checkpoint = Path(raw["paths"]["checkpoints"]) / run_id
    if run.exists() or checkpoint.exists():
        raise FileExistsError(
            "Choose a new run-id; reproduction attempts are immutable"
        )
    resumed_step = 0
    if resume is not None:
        state = json.loads((resume / "trainer_state.json").read_text())
        resumed_step = int(state["global_step"])
        if not 0 < resumed_step < 2000:
            raise ValueError("Resume checkpoint must be between steps 1 and 1,999")
        if condition == "closed_weight_k10":
            raise ValueError(
                "Use the audited historical K=10 resume command; its receipt prefix must be preserved"
            )
    run.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    shutil.copy2(config.path, run / "config.yaml")
    shutil.copy2(
        config.path.with_suffix(".provenance.json"), run / "config.provenance.json"
    )
    initialize_parameter_record(run, resolved_config=raw, source_config=config.path)
    (run / "train_examples.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    key = os.environ.get("OPENROUTER_API_KEY", "")
    offset = resumed_step * 4
    if condition == "reward_form_pilot":
        from iar_campaign.reward_pilot import reward_runtime as pilot_runtime

        runtime = pilot_runtime(
            config=config,
            domain=domain,
            log_path=run / "reward_calls.jsonl",
            api_key=key,
        )
    elif condition == "indist":
        runtime = reward_runtime(
            config=config,
            domain=domain,
            log_path=run / "reward_calls.jsonl",
            api_key=key,
            call_id_offset=offset,
            http_request_id_offset=offset,
        )
    elif condition == "quality":
        runtime = LiveOpenRouterReward(
            settings=settings_from_config(raw),
            log_path=run / "reward_calls.jsonl",
            api_key=key,
            prompt_builder=quality_prompt,
            prompt_template_hash=quality_prompt_template_sha256(),
            system_prompt_builder=lambda _: None,
            candidate_normalizer=strip_code_fences
            if domain == "kodcode"
            else str.strip,
            call_id_offset=offset,
            http_request_id_offset=offset,
        )
    elif condition == "oracle":
        if resume:
            raise ValueError("Oracle resume is not exposed by this portable command")
        runtime = LocalOracleReward(
            domain=domain,
            examples=examples,
            log_path=run / "reward_calls.jsonl",
            max_reward_calls=8000,
            kodcode_timeout_seconds=int(
                raw["evaluation"].get("coding_timeout_seconds", 60)
            ),
            spider_database_root=Path(raw["evaluation"]["spider_database_root"])
            if domain == "spider"
            else None,
            max_concurrency=16,
        )
    else:
        os.environ["IAR_GPT5MINI_DOMAIN"] = domain
        name = (
            "kodcode_gpt5mini_k10"
            if domain == "kodcode"
            else "gpt5mini_k10_multidomain"
        )
        module = importlib.import_module("iar_campaign." + name)
        if domain != "kodcode" and module.DOMAIN != domain:
            raise ValueError("Start each K=10 domain in a separate process")
        module.PROTOCOL = ROOT / "configs/closed_weight_k10" / f"{domain}_protocol.json"
        module.protocol()  # Verify provider, sampling law, K, retries, and reward contract.
        runtime = module.Gpt5MiniK10Reward(api_key=key, log_dir=run / "reward_calls")
    try:
        result, history = run_training(
            config,
            Dataset.from_list(rows),
            checkpoint,
            run,
            runtime,
            resume_from_checkpoint=resume,
        )
        if not resume:
            runtime.assert_complete()
        audit = (
            runtime.audit_summary()
            if hasattr(runtime, "audit_summary")
            else runtime.audit()
        )
        observed = audit.get("logical_reward_calls", audit.get("logical_rewards"))
        expected = (int(raw["training"]["max_steps"]) - resumed_step) * 4
        if observed != expected:
            raise RuntimeError(f"Expected {expected} rewards; observed {observed}")
        summary = {
            "status": "complete",
            "domain": domain,
            "condition": condition,
            "seed": seed,
            "run_id": run_id,
            "resumed_from": str(resume) if resume else None,
            "resumed_step": resumed_step,
            "reward_audit": audit,
            "trainer_metrics": result.metrics,
            "checkpoint_dir": str(checkpoint),
            "config_sha256": sha256(run / "config.yaml"),
            "official_test_accessed": False,
        }
        (run / "trainer_log_history.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in history)
        )
        (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    except Exception as exc:
        (run / "failure.json").write_text(
            json.dumps({"status": "failed", "error": str(exc)}, indent=2)
        )
        raise


def train_sft(domain: str, seed: int, run_id: str) -> dict:
    """Use the exact cached teacher corpus and recorded SFT/LoRA arguments."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    from iar_campaign.sft_baseline import _project_rows, _actor_source
    from iar_grpo.numerics import configure_stable_sdpa
    from iar_phase0.parameters import (
        initialize_parameter_record,
        write_parameter_artifact,
    )

    config = prepare(domain, "sft", seed, run_id=run_id)
    corpus = ROOT / "data/teacher_corpus" / f"{domain}.jsonl"
    record = next(
        r
        for r in json.loads(
            (ROOT / "reproduction/provenance/sft-manifest.json").read_text()
        )
        if r["domain"] == domain and r["kind"] == "corpus"
    )
    if sha256(corpus) != record["sha256"]:
        raise ValueError("Frozen teacher corpus drifted")
    rows = _project_rows(
        [json.loads(line) for line in corpus.read_text().splitlines() if line.strip()]
    )
    if len(rows) != 1000:
        raise ValueError("SFT requires all 1,000 teacher responses")
    # Match the frozen SFT driver: adapter initialization happens before the
    # trainer's own seeding, so TrainingArguments.seed alone is insufficient.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cache = defaults()["cache"]
    source = _actor_source(cache)
    ref = json.loads(
        (
            ROOT / "reproduction/provenance/sft" / domain / "trainer_arguments.json"
        ).read_text()
    )
    base = defaults()["runs"] / "reproduction/sft" / domain / f"seed{seed}"
    run, checkpoint = base / "runs" / run_id, base / "checkpoints" / run_id
    if run.exists() or checkpoint.exists():
        raise FileExistsError("Choose a new run-id")
    run.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    shutil.copy2(config.path, run / "config.yaml")
    shutil.copy2(
        config.path.with_suffix(".provenance.json"), run / "config.provenance.json"
    )
    initialize_parameter_record(
        run, resolved_config=config.raw, source_config=config.path
    )
    configure_stable_sdpa(torch)
    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, trust_remote_code=False, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    args = SFTConfig(
        output_dir=str(checkpoint),
        run_name=f"sft_{domain}_seed{seed}",
        seed=seed,
        data_seed=seed,
        learning_rate=1e-4,
        warmup_ratio=0.0,
        lr_scheduler_type="constant",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_train_epochs=2.0,
        max_steps=500,
        logging_steps=1,
        save_strategy="steps",
        save_steps=125,
        save_total_limit=None,
        eval_strategy="no",
        report_to="none",
        bf16=True,
        max_length=int(ref["max_length"]),
        completion_only_loss=True,
        shuffle_dataset=False,
        packing=False,
        include_num_input_tokens_seen=True,
        remove_unused_columns=True,
        model_init_kwargs={
            "cache_dir": str(cache),
            "local_files_only": True,
            "trust_remote_code": False,
            "torch_dtype": torch.bfloat16,
        },
    )
    for key in (
        "learning_rate",
        "max_steps",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_length",
        "save_steps",
        "shuffle_dataset",
        "completion_only_loss",
    ):
        if getattr(args, key) != ref[key]:
            raise ValueError(f"SFT {key} differs from recorded paper settings")
    lora = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.raw["actor"]["lora"]["target_modules"],
    )
    write_parameter_artifact(run, "trainer_arguments.json", args.to_dict())
    trainer = SFTTrainer(
        model=str(source),
        args=args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        peft_config=lora,
    )
    result = trainer.train()
    trainer.save_model(str(checkpoint / "final_adapter"))
    summary = {
        "status": "complete",
        "domain": domain,
        "condition": "sft",
        "seed": seed,
        "run_id": run_id,
        "global_step": trainer.state.global_step,
        "teacher_corpus_sha256": sha256(corpus),
        "trainer_metrics": result.metrics,
        "checkpoint_dir": str(checkpoint),
        "official_test_accessed": False,
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
