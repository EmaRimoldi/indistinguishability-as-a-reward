"""Regenerate the reported pairwise probe from frozen task/training inputs.

Generation and judging call the original implementations. Portable validation
binds bundled input hashes and the derived local config instead of cluster paths.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
import yaml
from .configure import prepare
from .paths import ROOT, defaults, sha256


def configure(domain, run_id="attempt001"):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run-id must be a single safe directory name")
    main = prepare(domain, "indist", 23, run_id="pairwise_" + run_id)
    base = defaults()["runs"] / "reproduction/pairwise" / domain
    base.mkdir(parents=True, exist_ok=True)
    template = ROOT / "configs/pairwise" / f"{domain}_probe.json"
    raw = json.loads(template.read_text())
    dataset = [
        json.loads(line)
        for line in Path(main.raw["dataset"]["split_file"]).read_text().splitlines()
        if line.strip()
    ]
    examples = dataset[:100]
    if len(dataset) != 1000 or len({str(row["id"]) for row in examples}) != 100:
        raise ValueError("Pairwise training prefix must contain 100 unique tasks")
    # SFT messages contain the original teacher responses, without evaluation labels.
    corpus = ROOT / "data/teacher_corpus" / f"{domain}.jsonl"
    teachers = [json.loads(line) for line in corpus.read_text().splitlines()]
    teacher_file = base / "teacher_responses.jsonl"
    teacher_file.write_text(
        "".join(
            json.dumps(
                {
                    "id": row["id"],
                    "prompt": row["messages"][0]["content"],
                    "response": row["messages"][1]["content"],
                },
                ensure_ascii=False,
            )
            + "\n"
            for row in teachers
        )
    )
    split = base / "probe_split.jsonl"
    split.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in examples
        )
    )
    raw["probe_split"] = str(split)
    raw["probe_split_sha256"] = sha256(split)
    summary = {
        "status": "complete",
        "domain": domain,
        "n": 1000,
        "model_id": "deepseek/deepseek-v4-flash",
        "provider": "StreamLake",
        "official_test_accessed": False,
        "responses": str(teacher_file),
        "responses_sha256": sha256(teacher_file),
        "note": "Derived from hash-verified SFT training messages; no new generation.",
    }
    summary_path = base / "teacher_cache.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    main.raw["campaign"]["teacher_summary"] = str(summary_path)
    grpo_path = base / "actor_config.yaml"
    grpo_path.write_text(yaml.safe_dump(main.raw, sort_keys=False))
    raw.update(
        output_root=str(base),
        grpo_config=str(grpo_path),
        grpo_config_sha256=sha256(grpo_path),
        _portable_template_sha256=sha256(template) if template.is_file() else None,
    )
    cfg = base / "probe_config.json"
    cfg.write_text(json.dumps(raw, indent=2) + "\n")
    return cfg, raw, examples


def run(domain, action, run_id="attempt001"):
    import iar_campaign.pairwise_experiment as original

    config_path, raw, examples = configure(domain, run_id)

    # Validate the known, hash-verified fixed probe rather than requiring a
    # mutable copy of the entire historical campaign and its scheduler layout.
    def validated(path):
        if (
            Path(path).resolve() != config_path.resolve()
            or json.loads(path.read_text()) != raw
        ):
            raise ValueError("Portable pairwise configuration changed")
        return raw, examples

    original.validate_probe_config = validated
    original.STEP_ROOT = defaults()["runs"] / "reproduction/pairwise"
    if action == "configure":
        return {
            "config": str(config_path),
            "actor_rollouts": 400,
            "judge_calls": 800,
            "model_calls": 0,
        }
    if action == "generate":
        return original.generate_probe_candidates(
            config_path=config_path, run_id=run_id
        )
    if action == "judge":
        generation = (
            Path(raw["output_root"]) / "runs" / run_id / "generation_attempt001"
        )
        if not (generation / "summary.json").is_file():
            raise FileNotFoundError(
                "Run pairwise generate first; recorded candidates are not bundled."
            )
        return original.run_probe_judge(
            config_path=config_path,
            run_id=run_id,
            key_pool=defaults()["runs"] / "secrets.env",
            key_name="OPENROUTER_API_KEY",
            lock_dir=defaults()["runs"] / "locks",
        )
    if action == "collect":
        return original.collect_probe_gate(config_paths=[config_path], run_id=run_id)
    raise ValueError(action)
