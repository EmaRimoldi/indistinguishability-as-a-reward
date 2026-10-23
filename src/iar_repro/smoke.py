"""Tiny random-weight training checks. These never represent paper results."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import yaml

from .paths import ROOT


class SyntheticReward:
    def __init__(self):
        self.calls = 0

    async def score_batch(self, *, completions, **kwargs):
        self.calls += len(completions)
        return [float(i % 2) for i in range(len(completions))]


def run(output: Path) -> dict:
    import torch
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
    from iar_grpo.config import GrpoTrialConfig, validate_grpo_config
    from iar_grpo.train import run_training
    from iar_campaign.reward_pilot import _score_kodcode

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(1)
    torch.manual_seed(23)
    core = Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "<eos>": 1,
                "<unk>": 2,
                "hello": 3,
                "world": 4,
                "one": 5,
                "two": 6,
                "three": 7,
            },
            unk_token="<unk>",
        )
    )
    core.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        model_input_names=["input_ids", "attention_mask"],
    )
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=8,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=0,
        )
    )
    source = output / "random_model"
    model.save_pretrained(source, safe_serialization=True)
    tokenizer.save_pretrained(source)
    raw = copy.deepcopy(
        yaml.safe_load((ROOT / "configs/indist/gsm8k.yaml").read_text())
    )
    raw["project"] = {"root": str(output), "seed": 23}
    raw["paths"]["hf_cache"] = str(output / "cache")
    raw["actor"].update(model_id=str(source), local_files_only=False, revision=None)
    raw["actor"]["lora"].update(r=2, alpha=2, dropout=0.0)
    raw["trial"].update(
        prompt_count=2, expected_reward_calls=4, run_name="SYNTHETIC_NOT_A_PAPER_RUN"
    )
    raw["reward"].update(
        max_reward_calls=4, max_http_requests=4, max_attempts_per_call=1
    )
    raw["training"].update(
        num_epochs=1,
        num_generations=2,
        generation_batch_size=4,
        per_device_train_batch_size=2,
        max_steps=2,
        gradient_checkpointing=False,
        max_prompt_tokens=64,
        max_completion_length=16,
        bf16=False,
        use_cpu=True,
        save_strategy="steps",
        save_steps=1,
    )
    raw["evaluation"]["checkpoints"] = [0, 1, 2]
    config_path = output / "synthetic_config.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    config = GrpoTrialConfig(raw, config_path)
    validate_grpo_config(config)
    rows = [
        {
            "prompt": [{"role": "user", "content": word}],
            "task_statement": word,
            "task_id": str(i),
            "dataset_index": i,
        }
        for i, word in enumerate(("hello", "world"))
    ]
    reward = SyntheticReward()
    checkpoints, attempt = output / "checkpoints", output / "attempt"
    checkpoints.mkdir()
    attempt.mkdir()
    result, history = run_training(
        config, Dataset.from_list(rows), checkpoints, attempt, reward
    )
    if (
        reward.calls != 4
        or not (checkpoints / "checkpoint-2/adapter_model.safetensors").is_file()
    ):
        raise RuntimeError("Synthetic trainer failed reward/checkpoint coverage")
    norms = [float(row["grad_norm"]) for row in history if "grad_norm" in row]
    if (
        not norms
        or not all(torch.isfinite(torch.tensor(v)) for v in norms)
        or not any(v > 0 for v in norms)
    ):
        raise RuntimeError("Synthetic training produced no finite nonzero update")
    good = _score_kodcode(
        ("def add(a,b): return a+b", "def test_add(): assert add(1,2)==3", 15)
    )
    bad = _score_kodcode(
        ("def add(a,b): return 0", "def test_add(): assert add(1,2)==3", 15)
    )
    if not good.get("correct") or bad.get("correct"):
        raise RuntimeError("Synthetic KodCode scorer failed")
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    sft_args = SFTConfig(
        output_dir=str(output / "sft"),
        use_cpu=True,
        bf16=False,
        max_steps=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        completion_only_loss=True,
        shuffle_dataset=False,
        packing=False,
        max_length=64,
        save_strategy="no",
        report_to="none",
        logging_steps=1,
        seed=23,
        model_init_kwargs={"local_files_only": True, "torch_dtype": torch.float32},
    )
    sft = SFTTrainer(
        model=str(source),
        args=sft_args,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(
            [
                {"prompt": "hello ", "completion": "world <eos>"},
                {"prompt": "one ", "completion": "two three <eos>"},
            ]
        ),
        peft_config=LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    sft_result = sft.train()
    sft.save_model(str(output / "sft/final_adapter"))
    if sft.state.global_step != 2 or not torch.isfinite(
        torch.tensor(sft_result.training_loss)
    ):
        raise RuntimeError("Synthetic SFT training failed")
    report = {
        "status": "passed",
        "synthetic_random_weights": True,
        "optimizer_steps": 2,
        "logical_rewards": reward.calls,
        "paid_model_calls": 0,
        "model_downloads": 0,
        "finite_nonzero_gradients": True,
        "checkpoint_saved": True,
        "kodcode_positive_and_negative_checks": True,
        "trainer_metrics": result.metrics,
        "sft_optimizer_steps": 2,
        "sft_completion_only_loss": True,
        "sft_checkpoint_saved": True,
        "full_paper_experiment": False,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
