"""Portable short reward-form pilot; preserves its separate validation protocol."""

import json
from pathlib import Path
import yaml
from iar_grpo.config import GrpoTrialConfig, validate_grpo_config
from .paths import ROOT, defaults, relocate, check_hashed_paths, sha256, input_index

FORMS = ("log_p_own", "p_own", "logit_p_own", "binary_own")


def prepare(domain, form, run_id="attempt001"):
    from .configure import prepare as main_prepare

    # Reuse the safe run-id and frozen-input checks before resolving this distinct protocol.
    main_prepare(domain, "indist", 23, run_id=run_id)
    if domain not in ("gsm8k", "kodcode") or form not in FORMS:
        raise ValueError(
            "Pilot supports GSM8K/KodCode and the four preregistered transforms"
        )
    template = ROOT / "configs/reward_form_pilot" / f"{domain}_{form}.yaml"
    raw = relocate(yaml.safe_load(template.read_text()), **defaults())
    if domain == "kodcode":
        gate = next(
            item
            for item in input_index()
            if item["original_relative"].endswith(
                "kodcode_validation300_scoring_protocol_v1.json"
            )
        )
        raw["evaluation"]["kodcode_scoring_protocol"] = str(
            ROOT / "reproduction/provenance/protocols" / gate["relative"]
        )
        raw["evaluation"]["kodcode_scoring_protocol_sha256"] = gate["sha256"]
    base = defaults()["runs"] / "reproduction/reward_form_pilot" / domain / form
    raw["paths"]["experiments"] = str(base / "runs")
    raw["paths"]["checkpoints"] = str(base / "checkpoints")
    raw["_reproduction_condition"] = "reward_form_pilot"
    path = base / "configs" / f"{run_id}.yaml"
    config = GrpoTrialConfig(raw, path)
    validate_grpo_config(config)
    if (
        config.prompt_count != 374
        or config.num_epochs != 1
        or config.num_generations != 4
        or raw["training"]["max_steps"] != 374
    ):
        raise ValueError("Reward pilot budget drifted")
    checked = check_hashed_paths(raw)
    original = json.loads(
        (ROOT / "reproduction/provenance/desktop-original-sha256.json").read_text()
    )
    for p in (
        Path(raw["dataset"]["split_file"]),
        Path(raw["evaluation"]["validation_file"]),
    ):
        if sha256(p) != original[str(p.relative_to(ROOT))]:
            raise ValueError("Pilot/validation data drifted")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    path.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "template": str(template.relative_to(ROOT)),
                "template_sha256": sha256(template),
                "resolved_config_sha256": sha256(path),
                "verified_hashed_inputs": checked,
                "changes": ["runtime paths"],
                "full_experiment_rerun": False,
            },
            indent=2,
        )
        + "\n"
    )
    return config
