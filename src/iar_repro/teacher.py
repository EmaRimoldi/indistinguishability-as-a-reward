"""Prepare/run the frozen SFT teacher-corpus generator with relocated inputs."""

import json
from pathlib import Path
from .paths import ROOT, defaults, relocate, check_hashed_paths


def corpus(domain: str, run_id: str, *, execute: bool = False) -> Path | dict:
    from .configure import prepare

    if domain in ("kodcode", "spider"):
        import iar_campaign.teacher_corpus_extension as teacher
    else:
        import iar_campaign.teacher_corpus as teacher
    prepare(domain, "indist", 23, run_id=run_id)
    raw = relocate(
        json.loads((ROOT / "configs/teacher_corpus" / f"{domain}.json").read_text()),
        **defaults(),
    )
    root = defaults()["runs"] / "reproduction/teacher_corpus"
    raw["output_root"] = str(root / "domains" / domain / "sft_teacher")
    check_hashed_paths(raw)
    teacher.SELECTION = Path(raw["domain_selection"])
    teacher.PROTOCOL = Path(raw["baseline_protocol"])
    teacher.STEP_ROOT = root
    path = root / "configs" / f"{domain}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, indent=2) + "\n")
    teacher.validate_teacher_config(path)
    if not execute:
        return path
    return teacher.run_teacher_corpus(
        config_path=path,
        run_id=run_id,
        key_pool=defaults()["runs"] / "secrets.env",
        key_name="OPENROUTER_API_KEY",
        lock_dir=defaults()["runs"] / "locks",
    )
