from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from iar_grpo.config import GrpoTrialConfig, validate_grpo_config
from .paths import ROOT, DOMAINS, SEEDS, check_hashed_paths, defaults, relocate, sha256

CONDITIONS = ("indist", "quality", "oracle", "closed_weight_k10", "sft", "a0")


def prepare(
    domain: str,
    condition: str,
    seed: int,
    *,
    run_id: str = "attempt001",
    locations: dict | None = None,
) -> GrpoTrialConfig:
    if domain not in DOMAINS or condition not in CONDITIONS or seed not in SEEDS:
        raise ValueError("Choose a paper domain, condition, and seed (23, 42, 7)")
    if condition == "closed_weight_k10" and seed != 23:
        raise ValueError("The paper reports K=10 only for seed 23")
    if condition == "a0" and seed != 23:
        raise ValueError("A0 is frozen; it has no training-seed replications")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run-id must be a single safe directory name")
    folder = "indist" if condition == "a0" else condition
    filename = f"{domain}_eval.yaml" if condition == "sft" else f"{domain}.yaml"
    template = ROOT / "configs" / folder / filename
    raw = yaml.safe_load(template.read_text())
    loc = locations or defaults()
    raw = relocate(raw, **loc)
    raw["project"]["seed"] = seed
    base = loc["runs"] / "reproduction" / condition / domain / f"seed{seed}"
    raw["paths"]["experiments"] = str(base / "runs")
    raw["paths"]["checkpoints"] = str(base / "checkpoints")
    raw["trial"]["run_name"] = f"{condition}_{domain}_seed{seed}"
    config_path = base / "configs" / f"{run_id}.yaml"
    config = GrpoTrialConfig(raw, config_path)
    validate_grpo_config(config)
    if raw["training"]["num_generations"] != 4 or raw["training"]["max_steps"] != 2000:
        raise ValueError("Paper GRPO group size/step budget drifted")
    if raw["actor"]["revision"] != "989aa7980e4cf806f80c7fef2b1adb7bc71aa306":
        raise ValueError("Paper actor revision drifted")
    if int(raw["evaluation"].get("max_new_tokens", 768)) != 768:
        raise ValueError("Paper evaluation token budget drifted")
    inputs = check_hashed_paths(raw)
    frozen = json.loads(
        (ROOT / "reproduction/provenance/desktop-original-sha256.json").read_text()
    )
    for key in ("split_file", "validation_file"):
        value = raw["dataset"].get(key) or raw["evaluation"].get(key)
        if value:
            path = Path(value)
            relative = str(path.relative_to(ROOT))
            if relative not in frozen or sha256(path) != frozen[relative]:
                raise ValueError(f"Frozen data split missing or changed: {relative}")
    raw["_reproduction_condition"] = condition
    if condition == "sft":
        ref = json.loads(
            (
                ROOT / "reproduction/provenance/sft" / domain / "trainer_arguments.json"
            ).read_text()
        )
        raw["training"] = {
            "algorithm": "SFT",
            "num_epochs": 2,
            **{
                key: ref[key]
                for key in (
                    "max_steps",
                    "learning_rate",
                    "per_device_train_batch_size",
                    "gradient_accumulation_steps",
                    "max_length",
                    "save_steps",
                    "shuffle_dataset",
                    "completion_only_loss",
                )
            },
        }
        raw["trial"]["expected_reward_calls"] = 0
        raw["evaluation"]["checkpoints"] = [0, 125, 250, 375, 500]
        raw["reward"] = {"type": "sft_no_reward", "max_http_requests": 0}
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    (config_path.with_suffix(".provenance.json")).write_text(
        json.dumps(
            {
                "template": str(template.relative_to(ROOT)),
                "template_sha256": sha256(template),
                "resolved_config_sha256": sha256(config_path),
                "domain": domain,
                "condition": condition,
                "seed": seed,
                "verified_hashed_inputs": inputs,
                "changes": ["runtime path relocation", "training seed", "output paths"]
                + (
                    ["SFT arguments recovered from recorded trainer_arguments.json"]
                    if condition == "sft"
                    else []
                ),
                "full_experiment_rerun": False,
            },
            indent=2,
        )
        + "\n"
    )
    return config
