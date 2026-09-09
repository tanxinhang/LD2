"""Machine-readable phase gate for repository-wide operations.

The gate is deliberately independent of simulation and learning modules so it
can be imported by CLIs before heavyweight dependencies are loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple

import yaml


DEFAULT_PHASE_FILE = Path(__file__).resolve().parent / "data" / "project_phase.yaml"
KNOWN_PHASES = {
    "architecture_migration",
    "architecture_audit",
    "reproduction",
    "result_refresh",
    "algorithm_research",
}

# A phase may only be advertised after the corresponding scientific workflow
# gates have been completed.  Keeping this mapping in code prevents a phase
# label from drifting ahead of its reproducibility evidence.
PHASE_REQUIRED_GATE = {
    "architecture_migration": "migration_freeze_declared",
    "architecture_audit": "architecture_migrated",
    "reproduction": "architecture_audited",
    "result_refresh": "result_refresh_approved",
    "algorithm_research": "algorithm_optimization_approved",
}


class OperationBlockedError(RuntimeError):
    """Raised when an operation is disabled by the current project phase."""


@dataclass(frozen=True)
class ProjectPhase:
    phase: str
    operations: Mapping[str, bool]
    required_gate_order: Tuple[str, ...]
    completed_gates: Tuple[str, ...]
    source: Path

    def allows(self, operation: str) -> bool:
        """Return whether *operation* is explicitly enabled.

        Unknown operations fail closed instead of inheriting permission from a
        broad phase name.
        """

        return self.operations.get(operation) is True


def load_project_phase(path: Path | str = DEFAULT_PHASE_FILE) -> ProjectPhase:
    """Load and validate the project phase contract."""

    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)

    if not isinstance(payload, dict):
        raise ValueError("project phase file must contain a YAML mapping")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported project phase schema_version")

    phase = payload.get("phase")
    if phase not in KNOWN_PHASES:
        raise ValueError(f"unknown project phase: {phase!r}")

    operations = payload.get("operations")
    if not isinstance(operations, dict) or not operations:
        raise ValueError("operations must be a non-empty mapping")
    if any(not isinstance(name, str) or not isinstance(value, bool)
           for name, value in operations.items()):
        raise ValueError("operations must map string names to booleans")

    required = tuple(payload.get("required_gate_order", ()))
    completed = tuple(payload.get("completed_gates", ()))
    if not required or len(required) != len(set(required)):
        raise ValueError("required_gate_order must be non-empty and unique")
    if any(gate not in required for gate in completed):
        raise ValueError("completed_gates contains an unknown gate")
    completed_positions = [required.index(gate) for gate in completed]
    if completed_positions != list(range(len(completed_positions))):
        raise ValueError("completed_gates must be a contiguous prefix of gate order")

    phase_gate = PHASE_REQUIRED_GATE[phase]
    if phase_gate not in required:
        raise ValueError(
            f"required_gate_order omits the gate required by phase {phase!r}: "
            f"{phase_gate!r}"
        )
    if phase_gate not in completed:
        raise ValueError(
            f"phase {phase!r} requires completed gate {phase_gate!r}"
        )

    return ProjectPhase(
        phase=phase,
        operations=dict(operations),
        required_gate_order=required,
        completed_gates=completed,
        source=source,
    )


def assert_operation_allowed(
    operation: str,
    path: Path | str = DEFAULT_PHASE_FILE,
) -> ProjectPhase:
    """Return the phase or raise when *operation* is not explicitly allowed."""

    phase = load_project_phase(path)
    if not phase.allows(operation):
        raise OperationBlockedError(
            f"operation {operation!r} is blocked during phase {phase.phase!r}; "
            f"see {phase.source}"
        )
    return phase
