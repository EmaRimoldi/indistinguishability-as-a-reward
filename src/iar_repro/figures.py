from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .paths import ROOT, sha256
from .records import analysis_inputs, results_root

EXPECTED = (
    "turing_reward_overview",
    "learning_to_fool",
    "reward_signal",
    "reward_distributions_by_correctness",
    "grpo_loss",
    "training_diagnostics_4x4",
    "within_prompt_correctness_gap",
    "judge_vs_reward",
    "response_length_by_reward",
)


def build(output: Path) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scripts = ROOT / "scripts/figures"
    data = results_root() / "figure_data"
    if not data.is_dir():
        raise FileNotFoundError(
            "Figure data is external. Set IAR_RESULTS_ROOT to your local result collection."
        )
    jobs = [
        (
            "plot_learning_to_fool.py",
            ["--output", str(output / "learning_to_fool.pdf")],
        ),
        ("plot_reward_signal.py", ["--output", str(output / "reward_signal.pdf")]),
        (
            "plot_reward_distributions.py",
            ["--output", str(output / "reward_distributions_by_correctness.pdf")],
        ),
        (
            "plot_response_length.py",
            ["--output", str(output / "response_length_by_reward.pdf")],
        ),
        ("plot_judge_vs_reward.py", ["--output", str(output / "judge_vs_reward.pdf")]),
        ("plot_training_diagnostics.py", ["--out-dir", str(output)]),
        (
            "plot_within_prompt.py",
            [
                "--input-dir",
                str(analysis_inputs()),
                "--output",
                str(output / "within_prompt_correctness_gap.pdf"),
            ],
        ),
        (
            "method_diagram/build.py",
            ["--out", str(output), "--stem", "turing_reward_overview", "--no-pptx"],
        ),
    ]
    env = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "IAR_FIGURE_RASTER_DIR": str(output / "png"),
    }
    for script, arguments in jobs:
        subprocess.run(
            [sys.executable, str(scripts / script), *arguments], env=env, check=True
        )
    files = {}
    for stem in EXPECTED:
        path = output / f"{stem}.pdf"
        if not path.is_file() or not path.read_bytes().startswith(b"%PDF"):
            raise RuntimeError(f"Missing/invalid figure output: {path}")
        files[path.name] = sha256(path)
    report = {
        "status": "passed",
        "pdfs": files,
        "figure_data": {p.name: sha256(p) for p in sorted(data.glob("*.json"))},
        "model_calls": 0,
        "full_experiments_rerun": False,
        "note": "Regenerated from recorded experimental data, not fresh model runs; PDF metadata/fonts may differ.",
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
