"""Regenerate teacher selection from the exact frozen probe task inputs."""

from __future__ import annotations
import json
import re
import shutil
import yaml
from pathlib import Path
from .configure import prepare as main_prepare
from .paths import ROOT, DOMAINS, defaults, sha256


def configure(domain, run_id="attempt001"):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run-id must be a single safe directory name")
    import iar_campaign.judge_sanity as original

    main_prepare(domain, "indist", 23, run_id="screen_" + run_id)
    locations = defaults()
    root = locations["runs"] / "reproduction/teacher_screen" / domain / run_id
    template = (
        ROOT
        / "configs/teacher_screen"
        / (
            "cqa_confirm_v4.yaml"
            if domain == "commonsenseqa"
            else "four_domain_v1.yaml"
        )
    )
    raw = yaml.safe_load(template.read_text())
    raw["enabled_domains"] = [domain]
    for name in DOMAINS:
        folder = root / "inputs" / name
        folder.mkdir(parents=True, exist_ok=True)
        source = ROOT / "data/screen" / f"{name}.jsonl"
        target = folder / "split.jsonl"
        if not target.exists():
            shutil.copy2(source, target)
        elif sha256(target) != sha256(source):
            raise ValueError("Frozen screening split changed")
        raw["domains"][name] = {
            "split_file": str(folder / "split.jsonl"),
            "candidate_file": str(folder / "candidates.jsonl"),
            "run_dir": str(folder),
        }
    raw["paths"] = {
        "spider_tables_json": str(
            (locations["spider"] or ROOT / "external/spider")
            / "spider_data/tables.json"
        ),
        "global_lock_dir": str(root / "locks"),
        "teacher_lane_root": str(root / "teacher"),
        "judge_lane_root": str(root / "judge"),
        "teacher_collection": str(root / "teacher_collection.json"),
        "judge_gate": str(root / "judge_gate.json"),
    }
    original.CAMPAIGN_ROOT = root
    original.validate_config(raw, require_candidates=False)
    config = root / "config.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False))
    config.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "template": str(template.relative_to(ROOT)),
                "template_sha256": sha256(template),
                "resolved_config_sha256": sha256(config),
                "source": "Frozen 100-task inputs; actor and teacher responses must be generated locally.",
                "full_experiment_rerun": False,
            },
            indent=2,
        )
        + "\n"
    )
    return config


def run(domain, action, run_id="attempt001", shard=0):
    import iar_campaign.judge_sanity as original
    from iar_campaign.reward_pilot import load_named_secret

    path = configure(domain, run_id)
    root = path.parent
    original.load_named_secret = load_named_secret
    if action == "configure":
        return {"config": str(path), "model_calls": 0}
    if action == "actors":
        from iar_campaign.generate import run_generation

        config = main_prepare(domain, "indist", 23, run_id="screen_" + run_id)
        folder = root / "inputs" / domain
        if (folder / "candidates.jsonl").exists():
            raise FileExistsError("Choose a new run-id; actor responses already exist")
        return run_generation(
            domain=domain,
            split_file=folder / "split.jsonl",
            output_dir=folder,
            cache_dir=defaults()["cache"],
            model_id=config.raw["actor"]["model_id"],
            revision=config.raw["actor"]["revision"],
            tables_json=Path(config.raw["evaluation"]["spider_tables_json"])
            if domain == "spider"
            else None,
            variants=("a0", "prompt_imitation"),
            batch_size=4 if domain == "kodcode" else 8,
            max_new_tokens=768,
            seed=23,
            campaign_root=defaults()["runs"],
        )
    if action.startswith("collect-"):
        return original.collect_phase(path, action.removeprefix("collect-"))
    if action not in ("teacher", "judge") or shard not in (0, 1, 2):
        raise ValueError("Choose a screening phase and shard 0/1/2")
    return original.run_lane(
        config_path=path,
        phase=action,
        shard_index=shard,
        num_shards=3,
        key_name=original.KEY_NAMES[shard],
        key_pool=defaults()["runs"] / "secrets.env",
        output_dir=root / action / f"lane_{shard}",
    )
