from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_completed_checkpoint(
    state_dir: Path,
    *,
    step: int,
    expected_rows: int,
    rows_file: str,
    required_files: tuple[str, ...],
) -> dict[str, Any] | None:
    """Reuse only a complete, self-consistent validation checkpoint state."""
    if not state_dir.exists():
        return None
    required = {"summary.json", rows_file, *required_files}
    missing = [name for name in sorted(required) if not (state_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Incomplete validation checkpoint {state_dir}; missing {', '.join(missing)}"
        )
    try:
        summary = json.loads((state_dir / "summary.json").read_text(encoding="utf-8"))
        with (state_dir / rows_file).open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unreadable validation checkpoint: {state_dir}") from exc
    if (
        not isinstance(summary, dict)
        or int(summary.get("step", -1)) != step
        or int(summary.get("n", -1)) != expected_rows
        or summary.get("official_test_accessed") is not False
        or len(rows) != expected_rows
        or not all(isinstance(row, dict) for row in rows)
    ):
        raise RuntimeError(f"Validation checkpoint metadata drifted: {state_dir}")
    return summary
