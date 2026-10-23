from __future__ import annotations

import json
from pathlib import Path

from .configure import prepare
from .paths import ROOT, DOMAINS, SEEDS, input_index, sha256


def verify(*, external: bool = False) -> dict:
    checked = []
    integrity = ROOT / "reproduction/provenance/package-integrity.json"
    protected = json.loads(integrity.read_text())
    for relative, digest in protected.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Package source/input integrity mismatch: {relative}")
    for record in input_index():
        path = ROOT / "reproduction/provenance/protocols" / record["relative"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"Frozen gate/protocol missing or changed: {path}")
        checked.append(str(path.relative_to(ROOT)))
    original = json.loads(
        (ROOT / "reproduction/provenance/desktop-original-sha256.json").read_text()
    )
    for relative, digest in original.items():
        if relative.startswith("data/"):
            path = ROOT / relative
            if sha256(path) != digest:
                raise ValueError(f"Original data split changed: {relative}")
            checked.append(relative)
    for item in json.loads(
        (ROOT / "reproduction/provenance/sft-manifest.json").read_text()
    ):
        if sha256(ROOT / item["local"]) != item["sha256"]:
            raise ValueError(f"SFT provenance/corpus changed: {item['local']}")
        checked.append(item["local"])
    configurations = []
    for domain in DOMAINS:
        for condition in (
            "indist",
            "quality",
            "oracle",
            "sft",
            "closed_weight_k10",
            "a0",
        ):
            seeds = (23,) if condition in ("closed_weight_k10", "a0") else SEEDS
            for seed in seeds:
                config = prepare(domain, condition, seed)
                if external:
                    config.actor_source()
                    if domain == "spider":
                        for key in ("spider_database_root", "spider_tables_json"):
                            if not Path(config.raw["evaluation"][key]).exists():
                                raise FileNotFoundError(config.raw["evaluation"][key])
                configurations.append(
                    dict(
                        domain=domain,
                        condition=condition,
                        seed=seed,
                        config=str(config.path),
                    )
                )
    from .pilot import FORMS, prepare as prepare_pilot
    from .teacher import corpus

    for domain in ("gsm8k", "kodcode"):
        for form in FORMS:
            config = prepare_pilot(domain, form)
            configurations.append(
                dict(
                    domain=domain,
                    condition="reward_form_pilot",
                    form=form,
                    seed=23,
                    config=str(config.path),
                )
            )
    for domain in DOMAINS:
        path = corpus(domain, "attempt001")
        configurations.append(
            dict(domain=domain, condition="teacher_corpus", config=str(path))
        )
    from .pairwise import configure as pairwise_config
    from .screen import configure as screen_config

    for domain in DOMAINS:
        path, _, _ = pairwise_config(domain)
        configurations.append(
            dict(domain=domain, condition="pairwise_probe", config=str(path))
        )
        path = screen_config(domain)
        configurations.append(
            dict(domain=domain, condition="teacher_screen", config=str(path))
        )
    return {
        "status": "passed",
        "verified_artifacts": len(checked),
        "protected_package_files": len(protected),
        "resolved_configurations": len(configurations),
        "configurations": configurations,
        "external_weights_and_databases_checked": external,
        "model_calls": 0,
        "full_experiments_rerun": False,
    }
