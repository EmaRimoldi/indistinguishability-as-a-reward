#!/usr/bin/env python3
"""Submit the immutable canary → train → sealed-test DAG for Step 16."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess
from datetime import datetime, timezone
from pathlib import Path
from iar_campaign.kodcode_gpt5mini_k10 import STEP_ROOT, prepare
from iar_campaign.reward_pilot import write_json

def submit(name: str, script: Path, source: Path, log_dir: Path, deps: list[int], key: str | None = None) -> int:
    command = ["sbatch", "--parsable", f"--job-name={name}"]
    if deps: command += ["--dependency=afterok:" + ":".join(map(str, deps)), "--kill-on-invalid-dep=yes"]
    exports = {"IAR_CAMPAIGN_SOURCE": str(source), "IAR_GPT5MINI_LOG_DIR": str(log_dir)}
    if key: exports["IAR_OPENROUTER_KEY_NAME"] = key
    command += ["--export=ALL," + ",".join(f"{k}={v}" for k, v in exports.items()), str(script)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return int(result.stdout.strip().split(";", 1)[0])

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--source", type=Path, required=True); parser.add_argument("--key-name", default="OPENROUTER_API_KEY"); args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_dir() or not str(source).startswith("/path/to/iar_code/"): raise ValueError("source must be an immutable source snapshot")
    prepared = prepare()
    manifest = STEP_ROOT / "manifests/launch_attempt001.json"
    if manifest.exists(): raise FileExistsError(manifest)
    scripts = source / "scripts"
    logs = STEP_ROOT / "logs"
    canary = submit("s16a1-canary-kod", scripts / "slurm_kodcode_gpt5mini_k10_canary.sbatch", source, logs / "canary", [], args.key_name)
    training = submit("s16a1-train-kod", scripts / "slurm_kodcode_gpt5mini_k10_train.sbatch", source, logs / "training", [canary], args.key_name)
    test = submit("s16a1-test-kod", scripts / "slurm_kodcode_gpt5mini_k10_test.sbatch", source, logs / "test", [training])
    result = {"status": "SUBMITTED", "submitted_at": datetime.now(timezone.utc).isoformat(), "source": str(source), "source_files_sha256": hashlib.sha256((source / "SOURCE_FILES.sha256").read_bytes()).hexdigest() if (source / "SOURCE_FILES.sha256").is_file() else None, "prepared": prepared, "jobs": {"canary": canary, "training": training, "sealed_test": test}, "dag": "canary(10 judgments) -> training(8000 rewards x K10) -> sealed_test(checkpoint 2000)", "official_test_accessed": False}
    write_json(manifest, result); print(json.dumps(result, indent=2, sort_keys=True))
if __name__ == "__main__": main()
