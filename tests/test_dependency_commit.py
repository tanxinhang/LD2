import numpy as np

from uav_isac.coordination.dependency_commit import (
    DependencyCommitLayout,
    certify_best_dependency_commit,
    certify_dependency_commit,
    dependency_closure,
    minimum_uniform_control_reserve,
    quantize_unit_interval_lower,
    quantize_unit_interval_upper,
)
from uav_isac.coordination.local_exchange_oracle import LocalMove
from uav_isac.environment.communication import InterUAVCommunicationModel


def _case():
    selected = np.zeros((3, 3, 2), dtype=bool)
    selected[0, 1, 0] = True
    role = np.asarray([0, 1, 0], dtype=np.int8)
    owner = np.asarray([1, -1], dtype=np.int64)
    proposal = selected.copy()
    proposal[0, 1, 0] = False
    proposal[0, 2, 0] = True
    move = LocalMove(
        kind="receiver_exchange",
        selected=proposal,
        role=role.copy(),
        owner=np.asarray([2, -1], dtype=np.int64),
    )
    positions = np.asarray([
        [0.0, 0.0, 100.0],
        [10.0, 0.0, 100.0],
        [20.0, 0.0, 100.0],
    ])
    return selected, role, owner, move, positions


def _model(**overrides):
    values = dict(
        rate_bits_per_dim=[0, 8],
        header_bits=64,
        bandwidth_hz=1.0e6,
        deadline_s=0.05,
        processing_delay_s=2.0e-4,
        snr_threshold_db=-20.0,
        antenna_gain_dbi=0.0,
        carrier_hz=2.4e9,
        tx_power_w=0.2,
        kT=4.0e-21,
        noise_figure_db=5.0,
        dt=0.1,
    )
    values.update(overrides)
    return InterUAVCommunicationModel(**values)


def _identity(num_agents=3):
    return {
        "certificate_epoch_ids": np.full(
            num_agents, 7, dtype=np.int64),
        "certificate_digests": np.full(
            num_agents, 0x1234ABCD, dtype=np.uint64),
    }


def test_layout_and_dependency_closure_have_exact_finite_counts():
    selected, role, owner, move, _ = _case()
    closure = dependency_closure(selected, role, owner, move)
    assert closure.affected_targets == (0,)
    assert closure.participants == (0, 1, 2)
    assert closure.changed_roles == 0
    assert closure.changed_owners == 1
    assert closure.toggled_edges == 2

    layout = DependencyCommitLayout(3, 2)
    assert layout.prepare_bits(
        changed_roles=0, changed_owners=1, toggled_edges=2) == 237
    assert layout.vote_bits == 147
    assert layout.decision_bits == 147


def test_commit_certificate_uses_three_physical_rounds_and_power_simplex():
    selected, role, owner, move, positions = _case()
    certificate = certify_dependency_commit(
        selected,
        role,
        owner,
        move,
        proposer=0,
        positions=positions,
        comm_power_w=np.full(3, 0.2),
        sensing_power_w=np.full((3, 2), 0.4),
        state_versions=np.full(3, 75, dtype=np.int64),
        **_identity(),
        communication_model=_model(),
    )
    assert certificate.feasible
    assert tuple(report.name for report in certificate.rounds) == (
        "prepare", "vote", "decision")
    assert certificate.rounds[1].active_senders == (1, 2)
    assert certificate.total_over_air_bits == 237 + 2 * 147 + 147
    assert certificate.total_latency_s <= 0.1
    assert certificate.total_energy_j > 0.0
    np.testing.assert_allclose(certificate.power_excess_w, 0.0, atol=1e-12)


def test_power_or_link_violation_cannot_be_offset_by_protocol_completion():
    selected, role, owner, move, positions = _case()
    power_failure = certify_dependency_commit(
        selected,
        role,
        owner,
        move,
        proposer=0,
        positions=positions,
        comm_power_w=np.full(3, 0.3),
        sensing_power_w=np.full(3, 0.8),
        state_versions=np.full(3, 75, dtype=np.int64),
        **_identity(),
        communication_model=_model(),
    )
    assert not power_failure.feasible
    assert all(f"power:uav:{k}" in power_failure.reasons for k in range(3))

    link_failure = certify_dependency_commit(
        selected,
        role,
        owner,
        move,
        proposer=0,
        positions=positions * np.asarray([1.0e7, 1.0e7, 1.0]),
        comm_power_w=np.full(3, 0.2),
        sensing_power_w=np.full(3, 0.8),
        state_versions=np.full(3, 75, dtype=np.int64),
        **_identity(),
        communication_model=_model(deadline_s=1.0e-5, snr_threshold_db=40.0),
    )
    assert not link_failure.feasible
    assert any(":snr:" in reason or ":deadline:" in reason
               for reason in link_failure.reasons)


def test_best_proposer_election_is_deterministic_and_fail_closed():
    selected, role, owner, move, positions = _case()
    first = certify_best_dependency_commit(
        selected,
        role,
        owner,
        move,
        positions=positions,
        comm_power_w=np.full(3, 0.2),
        sensing_power_w=np.full(3, 0.8),
        state_versions=np.full(3, 75, dtype=np.int64),
        **_identity(),
        communication_model=_model(),
    )
    second = certify_best_dependency_commit(
        selected,
        role,
        owner,
        move,
        positions=positions,
        comm_power_w=np.full(3, 0.2),
        sensing_power_w=np.full(3, 0.8),
        state_versions=np.full(3, 75, dtype=np.int64),
        **_identity(),
        communication_model=_model(),
    )
    assert first.feasible
    assert first.proposer == second.proposer
    assert first.total_latency_s == second.total_latency_s


def test_single_participant_change_needs_no_network_exchange():
    selected = np.zeros((2, 2, 1), dtype=bool)
    role = np.asarray([0, 0], dtype=np.int8)
    owner = np.asarray([-1], dtype=np.int64)
    move = LocalMove(
        kind="local_role",
        selected=selected.copy(),
        role=np.asarray([1, 0], dtype=np.int8),
        owner=owner.copy(),
    )
    certificate = certify_dependency_commit(
        selected,
        role,
        owner,
        move,
        proposer=0,
        positions=np.asarray([[0.0, 0.0, 100.0], [1.0, 0.0, 100.0]]),
        comm_power_w=np.zeros(2),
        sensing_power_w=np.ones(2),
        state_versions=np.zeros(2, dtype=np.int64),
        **_identity(2),
        communication_model=_model(),
    )
    assert certificate.feasible
    assert certificate.total_over_air_bits == 0
    assert certificate.total_energy_j == 0.0
    assert certificate.total_latency_s == 0.0


def test_probability_quantizers_are_one_sided_and_grid_bounded():
    values = np.asarray([0.0, 0.1, 0.6, 1.0])
    lower = quantize_unit_interval_lower(values, bits=8)
    upper = quantize_unit_interval_upper(values, bits=8)
    assert np.all(lower <= values)
    assert np.all(upper >= values)
    assert np.max(values - lower) <= 1.0 / 255.0 + 1e-15
    assert np.max(upper - values) <= 1.0 / 255.0 + 1e-15


def test_version_disagreement_forces_abort():
    selected, role, owner, move, positions = _case()
    certificate = certify_dependency_commit(
        selected,
        role,
        owner,
        move,
        proposer=0,
        positions=positions,
        comm_power_w=np.full(3, 0.2),
        sensing_power_w=np.full(3, 0.8),
        state_versions=np.asarray([75, 75, 74], dtype=np.int64),
        **_identity(),
        communication_model=_model(),
    )
    assert not certificate.feasible
    assert "protocol:state_version:uav:2" in certificate.reasons


def test_epoch_or_digest_disagreement_forces_abort():
    selected, role, owner, move, positions = _case()
    certificate = certify_dependency_commit(
        selected,
        role,
        owner,
        move,
        proposer=0,
        positions=positions,
        comm_power_w=np.full(3, 0.2),
        sensing_power_w=np.full(3, 0.8),
        state_versions=np.full(3, 75, dtype=np.int64),
        certificate_epoch_ids=np.asarray([7, 7, 8], dtype=np.int64),
        certificate_digests=np.asarray(
            [0x1234ABCD, 0x1234ABCD, 0xDEADBEEF], dtype=np.uint64),
        communication_model=_model(),
    )
    assert not certificate.feasible
    assert "protocol:certificate_epoch:uav:2" in certificate.reasons
    assert "protocol:certificate_digest:uav:2" in certificate.reasons


def test_minimum_control_reserve_solves_monotone_link_feasibility():
    selected, role, owner, move, positions = _case()
    result = minimum_uniform_control_reserve(
        selected,
        role,
        owner,
        move,
        positions=positions,
        current_comm_power_w=np.zeros(3),
        current_sensing_power_w=np.full((3, 2), 0.5),
        sensing_weights=np.full((3, 2), 0.5),
        state_versions=np.full(3, 75, dtype=np.int64),
        **_identity(),
        communication_model=_model(),
        reserve_upper_w=0.2,
        tolerance_w=1e-7,
    )
    assert result.feasible
    assert result.uniform_comm_floor_w is not None
    assert 0.0 < result.uniform_comm_floor_w <= 0.2
    assert result.certificate.feasible
    np.testing.assert_allclose(
        np.sum(np.asarray(result.sensing_power_w), axis=1)
        + np.asarray(result.comm_power_w),
        1.0,
        atol=1e-12,
    )
