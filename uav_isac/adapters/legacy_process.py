"""Managed V2 artifact boundary for stable pre-V2 numerical executors."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Tuple

from uav_isac.domain import RunCompletion, RunManifest

from .configuration import load_resolved_configuration
from .filesystem_artifacts import FileSystemArtifactStore
from .repository_identity import build_repository_identity


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
KNOWN_ENTRYPOINTS = {
    "train": "scripts/run_mappo.py",
    "pilot": "tools/run_strict_distributed_pilot.py",
    "refresh-results": "tools/run_strict_distributed_bank.py",
}
DEFAULT_CONFIGS = {
    "train": "config/default.yaml",
    "pilot": "config/exp_strict_distributed_no_truth_pilot.yaml",
    "refresh-results": "config/exp_strict_distributed_k16q16.yaml",
}
PRIMARY_OUTPUT_FLAGS = {
    "train": ("--out-dir", "raw/output"),
    "pilot": ("--output", "raw/result.json"),
    "refresh-results": ("--output", "raw/result.json"),
}
AUXILIARY_OUTPUT_FLAGS = {
    "--teacher-output", "--evidence-trace-output",
    "--structure-teacher-trace-output", "--n5-counterfactual-output",
}


def _value(arguments: Tuple[str, ...], flag: str, default: str | None = None):
    if flag not in arguments:
        return default
    index = arguments.index(flag)
    if index + 1 >= len(arguments):
        raise ValueError(f"{flag} requires a value")
    return arguments[index + 1]


def _rewrite_outputs(
    entrypoint: str, arguments: Tuple[str, ...], run_directory: Path,
) -> Tuple[str, ...]:
    primary_flag, primary_relative = PRIMARY_OUTPUT_FLAGS[entrypoint]
    output_flags = {primary_flag, *AUXILIARY_OUTPUT_FLAGS}
    rewritten = []
    index = 0
    seen_primary = False
    while index < len(arguments):
        item = arguments[index]
        if item not in output_flags:
            rewritten.append(item)
            index += 1
            continue
        if index + 1 >= len(arguments):
            raise ValueError(f"{item} requires a value")
        if item == primary_flag:
            relative = primary_relative
            seen_primary = True
        else:
            supplied = Path(arguments[index + 1])
            if not supplied.name:
                raise ValueError(f"{item} requires a file name")
            relative = f"raw/aux/{supplied.name}"
        rewritten.extend((item, str(run_directory / relative)))
        index += 2
    if not seen_primary:
        rewritten.extend((primary_flag, str(run_directory / primary_relative)))
    return tuple(rewritten)


def _seeds(
    entrypoint: str, arguments: Tuple[str, ...], resolved, repository: Path,
) -> Tuple[int, ...]:
    if entrypoint == "train":
        return (int(_value(arguments, "--seed", "42")),)
    if entrypoint == "pilot":
        raw = str(_value(arguments, "--seeds", "7"))
        return tuple(int(value) for value in raw.split(",") if value.strip())
    bank_path = Path(resolved.value.marl.eval_seed_bank_path)
    if not bank_path.is_absolute():
        bank_path = repository / bank_path
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    split = str(_value(arguments, "--split", "test"))
    values = tuple(int(seed) for seed in bank["splits"][split])
    limit = int(_value(arguments, "--limit", "0"))
    return values[:limit] if limit > 0 else values


class ManagedLegacyPythonProcessRunner:
    """Run a stable numerical executor inside one immutable V2 namespace."""

    def __init__(self, repository_root: Path | str = REPOSITORY_ROOT):
        self._repository = Path(repository_root).resolve()

    def run(self, entrypoint: str, arguments: Tuple[str, ...]) -> int:
        relative = KNOWN_ENTRYPOINTS.get(entrypoint)
        if relative is None:
            raise ValueError(f"unknown managed entrypoint: {entrypoint!r}")
        script = (self._repository / relative).resolve()
        if not script.is_relative_to(self._repository) or not script.is_file():
            raise FileNotFoundError(f"managed executor is unavailable: {script}")

        config = str(_value(arguments, "--config", DEFAULT_CONFIGS[entrypoint]))
        resolved = load_resolved_configuration(config)
        identity = build_repository_identity(self._repository)
        seeds = _seeds(entrypoint, arguments, resolved, self._repository)
        identity_payload = json.dumps({
            "entrypoint": entrypoint, "arguments": arguments,
            "code": identity.sha256, "config": resolved.sha256, "seeds": seeds,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
        run_id = f"{entrypoint}-{hashlib.sha256(identity_payload).hexdigest()[:20]}"
        store = FileSystemArtifactStore(self._repository / "artifacts")
        run_directory = store.root / "runs" / run_id
        managed_arguments = _rewrite_outputs(entrypoint, arguments, run_directory)
        store.create_run(RunManifest(
            run_id=run_id,
            run_type=entrypoint,
            state="created",
            created_at=datetime.now(timezone.utc).isoformat(),
            command=(sys.executable, relative, *managed_arguments),
            config_sha256=resolved.sha256,
            code_identity=identity.sha256,
            seeds=seeds,
            inputs={"config": resolved.source, "executor": relative},
        ))
        completed = subprocess.run(
            [sys.executable, str(script), *managed_arguments],
            cwd=self._repository,
            check=False,
        )
        artifacts = {}
        for path in sorted(run_directory.rglob("*")):
            if path.is_file() and path.name not in {"manifest.json", "completion.json"}:
                name = path.relative_to(run_directory).as_posix()
                artifacts[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        state = "completed" if completed.returncode == 0 and artifacts else "failed"
        store.complete_run(RunCompletion(
            run_id=run_id,
            state=state,
            completed_at=datetime.now(timezone.utc).isoformat(),
            artifacts=artifacts,
        ))
        return int(completed.returncode if state == "completed" else 1)


LegacyPythonProcessRunner = ManagedLegacyPythonProcessRunner
