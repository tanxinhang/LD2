"""Subprocess boundary for pre-V2 command implementations."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
KNOWN_ENTRYPOINTS = {
    "train": "scripts/run_mappo.py",
    "pilot": "tools/run_strict_distributed_pilot.py",
    "refresh-results": "tools/run_strict_distributed_bank.py",
}


class LegacyPythonProcessRunner:
    """Execute one allowlisted legacy script from the repository root."""

    def run(self, entrypoint: str, arguments: Tuple[str, ...]) -> int:
        relative = KNOWN_ENTRYPOINTS.get(entrypoint)
        if relative is None:
            raise ValueError(f"unknown legacy entrypoint: {entrypoint!r}")
        script = (REPOSITORY_ROOT / relative).resolve()
        if not script.is_relative_to(REPOSITORY_ROOT) or not script.is_file():
            raise FileNotFoundError(f"legacy entrypoint is unavailable: {script}")
        completed = subprocess.run(
            [sys.executable, str(script), *arguments],
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        return int(completed.returncode)
