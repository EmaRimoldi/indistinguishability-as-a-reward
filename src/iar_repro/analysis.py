"""Portable inputs for the original numerical analysis functions."""

from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path

from .paths import DOMAINS, ROOT
from .records import (
    analysis_inputs,
    result_path,
    reanalysis_inputs,
    materialize,
    log_paths,
    reward_log,
)


def run(output: Path, *, extended: bool = False) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = reanalysis_inputs()
    alignment = analysis_inputs()
    for name in ("metrics", "incremental_models"):
        shutil.copy2(
            materialize(f"results/analysis_reference/{name}.json.gz"),
            alignment / f"{name}.json",
        )
    sys.path.insert(0, str(ROOT / "scripts/analysis"))
    rollout = importlib.import_module("rollout_signal")
    rollout.ALIGN = alignment
    rollout.PARTIAL = materialize("results/partial/kodcode.jsonl.gz")
    rollout.reward_log = reward_log
    rollout.trainer_log = lambda condition, domain: [
        r
        for p in log_paths(condition, domain, "seed23")
        for r in rollout.read_jsonl(p)
        if "frac_reward_zero_std" in r
    ]
    seed = importlib.import_module("seed_figure_data")
    seed.ROOT = ROOT
    seed.LABELS = data / "seed_rollout_labels"
    seed.THREE_SEED_CSV = data / "three_seed_identity_heldout.csv"
    seed.reward_log = reward_log
    q = importlib.import_module("q_vs_pown")
    q.ROOT = ROOT
    q.Q_DIR = data / "q_on_identity"
    q.CLUSTER_LABELS = data / "seed_rollout_labels"
    judge = importlib.import_module("judge_vs_reward_three_seed")
    judge.Q_DIR = q.Q_DIR
    judge.LABELS = seed.LABELS
    contrast = importlib.import_module("reward_contrast_data")
    contrast.ALIGN = alignment
    contrast.LABELS = seed.LABELS
    contrast.QROWS = q.Q_DIR
    contrast.trainer_logs = log_paths
    contrast.OUTPUT = output / "reward_contrast_data.json"
    lengths = importlib.import_module("response_length_data")
    lengths.log_paths = log_paths
    lengths.OUTPUT = output / "response_length_data.json"
    repeated = importlib.import_module("repeated_prompts_three_seed")
    shutil.copy2(
        result_path("results/figure_data/correctness_deciles_data.json"),
        output / "correctness_deciles_data.json",
    )
    jobs = [
        (seed, ["--out-dir", str(output)]),
        (judge, ["--output", str(output / "judge_vs_reward_data.json")]),
        (repeated, ["--output", str(output / "repeated_prompts_three_seed.json")]),
        (contrast, []),
        (lengths, []),
        (rollout, ["--output", str(output / "rollout_signal.json")]),
    ]
    if extended:
        controls = importlib.import_module("controls_three_seed")
        controls.ARCHIVE = alignment
        work = output / "controls_work"
        for seed_name in ("seed42", "seed7"):
            (work / seed_name).mkdir(parents=True, exist_ok=True)
            for domain in DOMAINS:
                name = f"rows_{domain}_p_own.jsonl"
                shutil.copy2(
                    data / "controls" / seed_name / name, work / seed_name / name
                )
        jobs += [
            (q, ["--output", str(output / "q_vs_pown.json")]),
            (
                controls,
                [
                    "--work-dir",
                    str(work),
                    "--output",
                    str(output / "controls_three_seed.json"),
                ],
            ),
        ]
    previous = sys.argv
    try:
        for module, args in jobs:
            sys.argv = [module.__name__] + args
            module.main()
    finally:
        sys.argv = previous
    report = {
        "status": "passed",
        "analyses": [m.__name__ for m, _ in jobs],
        "model_calls": 0,
        "full_experiments_rerun": False,
        "note": "Recomputed numerical analyses from recorded rollouts/verifier labels and logged judge scores.",
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
