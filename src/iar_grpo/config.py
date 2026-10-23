from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from iar_phase0.prompts import (
    POINTWISE_SELF_ATTRIBUTION_NO_SYSTEM_PROMPT_VERSION,
    POINTWISE_SELF_ATTRIBUTION_PROMPT_VERSION,
)

CAMPAIGN_DOMAIN_ATTRIBUTION_PROMPT_VERSIONS = {
    "campaign-direct-quality-v1-no-reference-no-system",
    "campaign-pairwise-generate-distinguish-v1-no-system",
    "commonsenseqa-pointwise-self-attribution-v1-no-system",
    "commonsenseqa-pointwise-self-attribution-v2-binary-contract-no-system",
    "gsm8k-pointwise-self-attribution-v1-no-system",
    "spider-1.0-pointwise-self-attribution-v1-no-system",
}


@dataclass(frozen=True)
class GrpoTrialConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def root(self) -> Path:
        return Path(self.raw["project"]["root"]).expanduser()

    def resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.root / path

    @property
    def prompt_count(self) -> int:
        return int(self.raw["trial"]["prompt_count"])

    @property
    def num_generations(self) -> int:
        return int(self.raw["training"]["num_generations"])

    @property
    def num_epochs(self) -> int:
        return int(self.raw["training"].get("num_epochs", 1))

    @property
    def expected_reward_calls(self) -> int:
        return self.prompt_count * self.num_generations * self.num_epochs

    def actor_source(self) -> str:
        """Resolve a local snapshot path when offline actor loading is requested."""
        actor = self.raw["actor"]
        model_id = str(actor["model_id"])
        if not bool(actor.get("local_files_only")):
            return model_id
        cache = self.resolve(self.raw["paths"]["hf_cache"])
        slug = model_id.replace("/", "--")
        revision = actor.get("revision")
        if revision:
            snapshot = cache / f"models--{slug}" / "snapshots" / str(revision)
            if not snapshot.is_dir():
                raise FileNotFoundError(
                    f"No cached snapshot for offline actor {model_id!r}@{revision}"
                )
            return str(snapshot)
        snapshots = sorted((cache / f"models--{slug}" / "snapshots").glob("*"))
        if not snapshots:
            raise FileNotFoundError(f"No cached snapshot for offline actor {model_id!r}")
        return str(snapshots[-1])


def load_grpo_config(path: str | Path) -> GrpoTrialConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = GrpoTrialConfig(raw=raw, path=config_path)
    validate_grpo_config(config)
    return config


def validate_grpo_config(config: GrpoTrialConfig) -> None:
    trial = config.raw["trial"]
    training = config.raw["training"]
    reward = config.raw["reward"]

    if config.prompt_count <= 0:
        raise ValueError("trial.prompt_count must be positive")
    if config.num_generations < 2:
        raise ValueError("GRPO requires at least two generations per prompt")
    if config.num_epochs <= 0:
        raise ValueError("training.num_epochs must be positive")
    index_manifest = config.raw["dataset"].get("index_manifest")
    if index_manifest and str(config.raw["dataset"].get("selection")) != "manifest":
        raise ValueError("dataset.selection must be 'manifest' when index_manifest is configured")

    expected = config.expected_reward_calls
    declared = int(trial["expected_reward_calls"])
    if declared != expected:
        raise ValueError(
            f"trial.expected_reward_calls={declared}, but prompt_count * num_generations "
            f"* num_epochs={expected}"
        )

    generation_batch_size = int(training["generation_batch_size"])
    if generation_batch_size > expected:
        raise ValueError("generation_batch_size cannot exceed the complete rollout count")
    if generation_batch_size % config.num_generations:
        raise ValueError("generation_batch_size must contain complete GRPO groups")

    world_size = int(training.get("world_size", 1))
    local_batch = int(training["per_device_train_batch_size"]) * world_size
    if generation_batch_size % local_batch:
        raise ValueError("generation_batch_size must be divisible by the global training batch size")
    expected_steps = expected // local_batch
    if int(training["max_steps"]) != expected_steps:
        raise ValueError(
            "max_steps must equal "
            f"total_rollouts/global_batch_size={expected_steps}"
        )
    groups_per_generation_batch = generation_batch_size // config.num_generations
    prompt_visits = config.prompt_count * config.num_epochs
    if prompt_visits % groups_per_generation_batch:
        raise ValueError(
            "prompt_count * num_epochs must be divisible by the number of prompt groups "
            "in each generation batch so that every epoch contains complete GRPO batches"
        )

    if int(reward["max_reward_calls"]) != expected:
        raise ValueError("reward.max_reward_calls must equal the planned rollout count")
    reward_value = str(reward.get("value", "log_p_own"))
    if reward_value not in {
        "p_own",
        "log_p_own",
        "logit_p_own",
        "binary_own",
        "absolute_logp_1",
        "oracle_task_utility",
        "log_p_pair",
        "closed_weight_log_p_own_k10",
    }:
        raise ValueError(f"Unsupported reward.value: {reward_value!r}")
    epsilon = float(reward.get("epsilon", 1e-6))
    if not 0.0 < epsilon < 0.5:
        raise ValueError("reward.epsilon must be strictly between 0 and 0.5")
    attribution_prompt_version = str(
        config.raw["discriminator"].get(
            "attribution_prompt_version", POINTWISE_SELF_ATTRIBUTION_PROMPT_VERSION
        )
    )
    if attribution_prompt_version not in {
        POINTWISE_SELF_ATTRIBUTION_PROMPT_VERSION,
        POINTWISE_SELF_ATTRIBUTION_NO_SYSTEM_PROMPT_VERSION,
        *CAMPAIGN_DOMAIN_ATTRIBUTION_PROMPT_VERSIONS,
    }:
        raise ValueError(
            "Unsupported discriminator.attribution_prompt_version: "
            f"{attribution_prompt_version!r}"
        )
    attempts = int(reward["max_attempts_per_call"])
    if attempts <= 0:
        raise ValueError("reward.max_attempts_per_call must be positive")
    if reward_value == "log_p_pair":
        http_requests_per_reward = 2
    elif reward_value == "closed_weight_log_p_own_k10":
        http_requests_per_reward = int(reward.get("judgments_per_reward", 0))
        if http_requests_per_reward != 10:
            raise ValueError("closed-weight reward requires exactly 10 judgments")
    else:
        http_requests_per_reward = 1
    expected_http_requests = (
        0
        if reward_value == "oracle_task_utility"
        else expected * attempts * http_requests_per_reward
    )
    if int(reward["max_http_requests"]) != expected_http_requests:
        raise ValueError(
            "reward.max_http_requests must equal zero for the local oracle, or "
            "expected_reward_calls * max_attempts_per_call * HTTP calls per reward"
        )
    if float(reward.get("minimum_request_interval_seconds", 0.0)) < 0:
        raise ValueError("reward.minimum_request_interval_seconds cannot be negative")
    if float(reward.get("max_retry_wait_seconds", 60.0)) <= 0:
        raise ValueError("reward.max_retry_wait_seconds must be positive")
    if float(training["temperature"]) <= 0:
        raise ValueError("GRPO rollouts require stochastic sampling; temperature must be positive")

    save_strategy = str(training.get("save_strategy", "no"))
    if save_strategy == "steps":
        save_steps = int(training["save_steps"])
        if save_steps <= 0 or int(training["max_steps"]) % save_steps:
            raise ValueError("training.save_steps must divide max_steps exactly")

    evaluation = config.raw.get("evaluation") or {}
    checkpoints = [int(step) for step in evaluation.get("checkpoints", [])]
    if checkpoints and (checkpoints[0] != 0 or checkpoints[-1] != int(training["max_steps"])):
        raise ValueError("evaluation.checkpoints must start at 0 and end at max_steps")
