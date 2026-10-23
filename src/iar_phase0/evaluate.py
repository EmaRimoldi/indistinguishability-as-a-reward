from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UnitTestResult:
    passed: bool
    returncode: int
    stdout: str
    stderr: str


def run_mbpp_tests(
    code: str,
    tests: list[str],
    timeout_seconds: int,
    test_setup_code: str = "",
) -> UnitTestResult:
    if not tests:
        return UnitTestResult(
            passed=False,
            returncode=-1,
            stdout="",
            stderr="No tests available for this example.",
        )
    program = "\n\n".join(
        part.strip()
        for part in [code, test_setup_code, "\n".join(tests)]
        if part and part.strip()
    ) + "\n"
    with tempfile.TemporaryDirectory(prefix="iar_mbpp_") as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(program, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(path)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            return UnitTestResult(
                passed=completed.returncode == 0,
                returncode=completed.returncode,
                stdout=completed.stdout[-4000:],
                stderr=completed.stderr[-4000:],
            )
        except subprocess.TimeoutExpired as exc:
            return UnitTestResult(
                passed=False,
                returncode=-2,
                stdout=(exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                stderr=f"Timeout after {timeout_seconds}s",
            )
