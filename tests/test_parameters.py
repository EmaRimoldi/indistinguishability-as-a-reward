from __future__ import annotations

import json
from pathlib import Path

from iar_phase0.parameters import initialize_parameter_record, write_parameter_artifact


def test_parameter_record_contains_requested_and_resolved_values(tmp_path: Path) -> None:
    run_dir = tmp_path / "experiments" / "trial_001"
    run_dir.mkdir(parents=True)
    source = tmp_path / "config.yaml"
    source.write_text("training:\n  learning_rate: 1.0e-5\n", encoding="utf-8")

    parameters_dir = initialize_parameter_record(
        run_dir,
        resolved_config={"training": {"learning_rate": 1.0e-5}},
        source_config=source,
    )
    write_parameter_artifact(run_dir, "trainer_arguments.json", {"max_steps": 100})

    assert (parameters_dir / "requested_config.yaml").read_text() == source.read_text()
    resolved = json.loads((parameters_dir / "resolved_parameters.json").read_text())
    assert resolved["resolved_config"]["training"]["learning_rate"] == 1.0e-5
    trainer = json.loads((parameters_dir / "trainer_arguments.json").read_text())
    assert trainer == {"max_steps": 100}
    manifest = json.loads((parameters_dir / "manifest.json").read_text())
    assert manifest["contains_secrets"] is False
