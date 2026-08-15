import numpy as np
import pytest

from uav_isac.evaluation.shadow_horizon_router import (
    ShadowBranchObservation,
    arbitrate_shadow_horizon,
    isac_action_digest,
    isac_structure_digest,
)


def _digest(*, power_shift: float = 0.0) -> str:
    selected = np.zeros((2, 2, 1), dtype=bool)
    selected[0, 1, 0] = True
    owner = np.asarray([1], dtype=np.int64)
    return isac_action_digest(
        selected=selected,
        role=np.asarray([1, 0], dtype=np.int8),
        owner=owner,
        sensing_power_w=np.asarray([
            [0.7 + power_shift], [0.6 - power_shift],
        ]),
        comm_power_w=np.asarray([0.3, 0.4]),
    )


def _branch(
    accept: bool,
    digest: str | None,
    *,
    latency: float = 0.04,
    energy: float = 0.01,
    evaluated: bool = True,
    hard_feasible: bool = True,
    resource_accounting_complete: bool = True,
    structure_digest: str | None = None,
) -> ShadowBranchObservation:
    return ShadowBranchObservation(
        evaluated=evaluated,
        accept=accept,
        action_digest=digest,
        structure_digest=structure_digest,
        latency_s=latency,
        control_energy_j=energy,
        hard_feasible=hard_feasible,
        resource_accounting_complete=resource_accounting_complete,
    )


def test_shadow_router_only_recommends_identical_consensus_candidate():
    digest = _digest()
    decision = arbitrate_shadow_horizon(
        _branch(True, digest, latency=0.03, energy=0.01),
        _branch(True, digest, latency=0.08, energy=0.02),
        control_period_s=0.1,
        max_control_energy_j=0.04,
        shared_control_energy_j=0.005,
    )
    assert decision.live_action == "one_step_candidate"
    assert decision.shadow_action == "consensus_candidate"
    assert decision.reason == "consensus_candidate"
    assert np.isclose(decision.parallel_latency_s, 0.08)
    assert np.isclose(decision.shared_control_energy_j, 0.005)
    assert np.isclose(decision.total_control_energy_j, 0.035)
    assert not decision.commit_authority


@pytest.mark.parametrize(
    ("one_step", "horizon", "expected"),
    [
        (_branch(True, _digest()), _branch(False, None),
         "acceptance_disagreement"),
        (_branch(True, _digest()), _branch(True, _digest(power_shift=0.01)),
         "candidate_identity_disagreement"),
        (_branch(True, None), _branch(True, None),
         "candidate_identity_missing"),
        (_branch(False, None), _branch(False, None), "consensus_noop"),
    ],
)
def test_shadow_router_fails_closed_on_disagreement_or_noop(
    one_step, horizon, expected,
):
    decision = arbitrate_shadow_horizon(
        one_step, horizon, control_period_s=0.1)
    assert decision.shadow_action == "noop"
    assert decision.reason == expected


def test_shadow_router_fails_closed_on_deadline_energy_and_missing_branch():
    digest = _digest()
    late = arbitrate_shadow_horizon(
        _branch(True, digest, latency=0.02),
        _branch(True, digest, latency=0.11),
        control_period_s=0.1,
    )
    assert late.reason == "parallel_deadline_failure"

    expensive = arbitrate_shadow_horizon(
        _branch(True, digest, energy=0.03),
        _branch(True, digest, energy=0.03),
        control_period_s=0.1,
        max_control_energy_j=0.05,
    )
    assert expensive.reason == "combined_energy_failure"

    incomplete = arbitrate_shadow_horizon(
        _branch(True, digest),
        _branch(False, None, evaluated=False),
        control_period_s=0.1,
    )
    assert incomplete.reason == "incomplete_branch_observation"
    assert incomplete.live_action == "one_step_candidate"
    assert incomplete.shadow_action == "noop"

    unmetered = arbitrate_shadow_horizon(
        _branch(True, digest),
        _branch(True, digest, resource_accounting_complete=False),
        control_period_s=0.1,
    )
    assert unmetered.reason == "incomplete_resource_accounting"


def test_complete_package_energy_is_charged_once_above_branch_rf_energy():
    digest = _digest()
    decision = arbitrate_shadow_horizon(
        _branch(True, digest, energy=0.01),
        _branch(True, digest, energy=0.02),
        control_period_s=0.1,
        shared_control_energy_j=0.04,
        max_control_energy_j=0.071,
    )
    assert decision.reason == "consensus_candidate"
    assert decision.total_control_energy_j == pytest.approx(0.07)
    failed = arbitrate_shadow_horizon(
        _branch(True, digest, energy=0.01),
        _branch(True, digest, energy=0.02),
        control_period_s=0.1,
        shared_control_energy_j=0.04,
        max_control_energy_j=0.069,
    )
    assert failed.reason == "combined_energy_failure"
    with pytest.raises(ValueError, match="shared_control_energy_j"):
        arbitrate_shadow_horizon(
            _branch(True, digest),
            _branch(True, digest),
            control_period_s=0.1,
            shared_control_energy_j=float("inf"),
        )


def test_complete_compute_latency_is_charged_once_above_parallel_network():
    digest = _digest()
    decision = arbitrate_shadow_horizon(
        _branch(True, digest, latency=0.02),
        _branch(True, digest, latency=0.03),
        control_period_s=0.1,
        shared_control_latency_s=0.04,
    )
    assert decision.reason == "consensus_candidate"
    assert decision.shared_control_latency_s == pytest.approx(0.04)
    assert decision.parallel_latency_s == pytest.approx(0.07)
    failed = arbitrate_shadow_horizon(
        _branch(True, digest, latency=0.02),
        _branch(True, digest, latency=0.03),
        control_period_s=0.069,
        shared_control_latency_s=0.04,
    )
    assert failed.reason == "parallel_deadline_failure"
    with pytest.raises(ValueError, match="shared_control_latency_s"):
        arbitrate_shadow_horizon(
            _branch(True, digest), _branch(True, digest),
            control_period_s=0.1,
            shared_control_latency_s=float("inf"))


def test_action_digest_binds_power_and_shape():
    assert _digest() != _digest(power_shift=0.01)
    with pytest.raises(ValueError, match="selected"):
        isac_action_digest(
            selected=np.zeros((2, 1, 1), dtype=bool),
            role=np.zeros(2, dtype=np.int8),
            owner=np.zeros(1, dtype=np.int64),
            sensing_power_w=np.zeros((2, 1)),
            comm_power_w=np.zeros(2),
        )


def test_structure_consensus_can_select_already_certified_horizon_power():
    selected = np.zeros((2, 2, 1), dtype=bool)
    selected[0, 1, 0] = True
    structure = isac_structure_digest(
        selected=selected,
        role=np.asarray([1, 0], dtype=np.int8),
        owner=np.asarray([1], dtype=np.int64),
    )
    decision = arbitrate_shadow_horizon(
        _branch(True, _digest(), structure_digest=structure),
        _branch(
            True,
            _digest(power_shift=0.01),
            structure_digest=structure,
        ),
        control_period_s=0.1,
        allow_horizon_power_on_structure_consensus=True,
    )

    assert not decision.candidate_identity_agrees
    assert decision.structure_identity_agrees
    assert decision.shadow_action == "horizon_power_candidate"
    assert decision.reason == "structure_consensus_horizon_power"
    assert not decision.commit_authority

    late = arbitrate_shadow_horizon(
        _branch(True, _digest(), structure_digest=structure),
        _branch(
            True,
            _digest(power_shift=0.01),
            structure_digest=structure,
            latency=0.11,
        ),
        control_period_s=0.1,
        allow_horizon_power_on_structure_consensus=True,
    )
    assert late.shadow_action == "noop"
    assert late.reason == "parallel_deadline_failure"


def test_structure_digest_binds_roles_and_owners():
    selected = np.zeros((3, 3, 1), dtype=bool)
    selected[0, 2, 0] = True
    baseline = isac_structure_digest(
        selected=selected,
        role=np.asarray([1, 1, 0], dtype=np.int8),
        owner=np.asarray([2], dtype=np.int64),
    )
    changed_role = isac_structure_digest(
        selected=selected,
        role=np.asarray([1, 0, 0], dtype=np.int8),
        owner=np.asarray([2], dtype=np.int64),
    )
    changed_owner = isac_structure_digest(
        selected=selected,
        role=np.asarray([1, 1, 0], dtype=np.int8),
        owner=np.asarray([1], dtype=np.int64),
    )
    assert baseline != changed_role
    assert baseline != changed_owner
