from pathlib import Path

import pytest

from uav_isac.governance.project_phase import (
    OperationBlockedError,
    assert_operation_allowed,
    load_project_phase,
)


def test_repository_is_ready_for_preregistered_algorithm_research():
    phase = load_project_phase()

    assert phase.phase == "algorithm_research"
    assert not phase.allows("architecture_migration")
    assert phase.allows("characterization_tests")
    assert phase.allows("algorithm_optimization")
    assert not phase.allows("full_result_refresh")
    assert not phase.allows("destructive_data_cleanup")


def test_unknown_operation_fails_closed():
    with pytest.raises(OperationBlockedError):
        assert_operation_allowed("unregistered_operation")


def test_completed_gates_must_be_a_prefix(tmp_path: Path):
    phase_file = tmp_path / "phase.yaml"
    phase_file.write_text(
        """schema_version: 1
phase: architecture_migration
operations: {architecture_migration: true}
required_gate_order: [one, two, three]
completed_gates: [one, three]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contiguous prefix"):
        load_project_phase(phase_file)


def test_result_refresh_requires_its_approval_gate(tmp_path: Path):
    phase_file = tmp_path / "phase.yaml"
    phase_file.write_text(
        """schema_version: 1
phase: result_refresh
operations: {full_result_refresh: true}
required_gate_order:
  - migration_freeze_declared
  - baseline_characterized
  - architecture_migrated
  - architecture_audited
  - reproduction_passed
  - result_refresh_approved
completed_gates:
  - migration_freeze_declared
  - baseline_characterized
  - architecture_migrated
  - architecture_audited
  - reproduction_passed
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires completed gate"):
        load_project_phase(phase_file)
