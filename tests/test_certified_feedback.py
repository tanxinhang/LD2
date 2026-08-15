import numpy as np
import pytest

from uav_isac.environment.communication import InterUAVCommunicationModel
from uav_isac.evaluation.certified_feedback import (
    CertificateViolationEProcess,
    OwnerFeedbackLayout,
    build_residual_feedback_features,
    calibrate_frozen_feedback_epoch,
    certify_owner_feedback_transport,
    feedback_adjusted_reconfiguration_is_certified,
    fit_event_balanced_residual_model,
    minimum_joint_commit_feedback_reserve,
)
from uav_isac.coordination.dependency_commit import DependencyCommitLayout
from uav_isac.coordination.local_exchange_oracle import LocalMove


def _model(**overrides):
    values = dict(
        rate_bits_per_dim=[0, 8],
        header_bits=64,
        bandwidth_hz=1.0e6,
        deadline_s=0.02,
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


def _identity(num_agents):
    return {
        "certificate_epoch_ids": np.full(
            num_agents, 7, dtype=np.int64),
        "certificate_digests": np.full(
            num_agents, 0x1234ABCD, dtype=np.uint64),
    }


def _training_model(repeat_first_event=1):
    first_x = np.asarray([[0.0], [1.0]])
    first_y = np.asarray([0.0, 1.0])
    first_x = np.repeat(first_x, repeat_first_event, axis=0)
    first_y = np.repeat(first_y, repeat_first_event, axis=0)
    feature_events = [first_x, np.asarray([[2.0], [3.0]])]
    predicted = [np.zeros(first_y.shape), np.zeros(2)]
    observed = [first_y, np.asarray([2.0, 3.0])]
    return fit_event_balanced_residual_model(
        feature_events,
        predicted,
        observed,
        event_ids=["train-a", "train-b"],
        ridge=1e-6,
    )


def test_event_balanced_fit_is_invariant_to_within_event_replication():
    base = _training_model(repeat_first_event=1)
    repeated = _training_model(repeat_first_event=20)
    grid = np.arange(4, dtype=np.float64)[:, None]
    np.testing.assert_allclose(
        base.predict(grid), repeated.predict(grid), atol=1e-10)


def test_feedback_feature_map_uses_only_bounded_observable_inputs():
    features = build_residual_feedback_features(
        token_age_frames=np.asarray([0.0, 2.0]),
        packet_loss_fraction=0.1,
        quantization_step=1.0 / 255.0,
        switch_fraction=np.asarray([0.0, 0.5]),
        horizon_fraction=np.asarray([0.0, 1.0]),
        tail_fraction=0.5,
    )
    assert features.shape == (2, 8)
    assert np.all(np.isfinite(features))
    with pytest.raises(ValueError):
        build_residual_feedback_features(0, 1.1, 0.1, 0.1, 0.1, 0.1)


def test_calibration_is_event_disjoint_and_epoch_is_fail_closed():
    model = _training_model()
    feature_events = [np.asarray([[0.5], [1.5]]) for _ in range(20)]
    predicted = [np.full(2, -0.2) for _ in range(20)]
    observed = [np.full(2, -0.2) for _ in range(20)]
    uncertainty = [np.full(2, 0.1) for _ in range(20)]
    epoch = calibrate_frozen_feedback_epoch(
        model,
        feature_events,
        predicted,
        observed,
        uncertainty,
        event_ids=[f"cal-{index}" for index in range(20)],
        miscoverage=0.05,
        epoch_id="epoch-1",
        proposal_pipeline_digests=["pipeline-v1"] * 20,
    )
    assert np.isfinite(epoch.multiplier)
    features = np.asarray([[0.5], [1.5]])
    assert not feedback_adjusted_reconfiguration_is_certified(
        epoch,
        features,
        np.full(2, -10.0),
        np.full(2, 0.1),
        commit_feasible=True,
        structural_feasible=True,
        drift_locked=True,
        proposal_pipeline_digest="pipeline-v1",
    )
    assert not feedback_adjusted_reconfiguration_is_certified(
        epoch,
        features,
        np.full(2, -10.0),
        np.full(2, 0.1),
        commit_feasible=True,
        structural_feasible=True,
        drift_locked=False,
        proposal_pipeline_digest="pipeline-v2",
    )
    with pytest.raises(ValueError, match="leakage"):
        calibrate_frozen_feedback_epoch(
            model,
            [np.asarray([[0.5]])],
            [np.asarray([0.0])],
            [np.asarray([0.0])],
            [np.asarray([0.1])],
            event_ids=["train-a"],
            miscoverage=0.05,
            epoch_id="bad",
            proposal_pipeline_digests=["pipeline-v1"],
        )


def test_feedback_epoch_can_jointly_calibrate_transition_and_channel():
    model = _training_model()
    count = 20
    epoch = calibrate_frozen_feedback_epoch(
        model,
        [np.asarray([[0.5], [1.5]]) for _ in range(count)],
        [np.full(2, -0.2) for _ in range(count)],
        [np.full(2, -0.2) for _ in range(count)],
        [np.full(2, 0.1) for _ in range(count)],
        event_ids=[f"joint-cal-{index}" for index in range(count)],
        miscoverage=0.05,
        epoch_id="joint-feedback-1",
        proposal_pipeline_digests=["pipeline-v1"] * count,
        channel_event_scores=np.full(count, 3.0),
        snr_resolution_db=1.0,
        latency_resolution_s=1.0e-4,
    )
    assert epoch.multiplier == pytest.approx(3.0)
    assert epoch.physical_margins == pytest.approx((3.0, 3.0e-4))
    assert epoch.calibration_event_scores == pytest.approx(np.full(count, 3.0))
    assert len(epoch.transition_calibration_event_scores) == count


def test_violation_e_process_locks_after_repeated_exceedances():
    monitor = CertificateViolationEProcess(
        miscoverage=0.05,
        false_alarm_probability=0.01,
    )
    for _ in range(20):
        state = monitor.update(event_score=2.0, multiplier=1.0)
        if state.alarm:
            break
    assert state.alarm
    assert state.e_value >= 100.0
    # Alarm is sticky even after a non-violation.
    assert monitor.update(event_score=0.0, multiplier=1.0).alarm


def test_owner_feedback_layout_and_physical_transport_are_explicit():
    layout = OwnerFeedbackLayout(6, 6, horizon_frames=5)
    assert layout.shared_bits == 153
    assert layout.entry_bits == 23
    assert layout.packet_bits(6) == 291
    positions = np.asarray([
        [float(k * 10), 0.0, 100.0] for k in range(6)])
    certificate = certify_owner_feedback_transport(
        {k: 6 for k in range(6)},
        coordinator=0,
        positions=positions,
        comm_power_w=np.full(6, 0.2),
        sensing_power_w=np.full(6, 0.8),
        state_versions=np.full(6, 80, dtype=np.int64),
        **_identity(6),
        communication_model=_model(),
        layout=layout,
    )
    assert certificate.feasible
    assert certificate.senders == (1, 2, 3, 4, 5)
    assert certificate.total_over_air_bits == 5 * 291
    assert certificate.total_energy_j > 0.0

    failed = certify_owner_feedback_transport(
        {k: 6 for k in range(6)},
        coordinator=0,
        positions=positions,
        comm_power_w=np.asarray([0.2, 0.2, 0.0, 0.2, 0.2, 0.2]),
        sensing_power_w=np.asarray([0.8, 0.8, 1.0, 0.8, 0.8, 0.8]),
        state_versions=np.asarray([80, 80, 79, 80, 80, 80]),
        **_identity(6),
        communication_model=_model(),
        layout=layout,
    )
    assert not failed.feasible
    assert "protocol:state_version:uav:2" in failed.reasons
    assert any(reason.startswith("feedback:snr:2") for reason in failed.reasons)

    binding_failed = certify_owner_feedback_transport(
        {k: 6 for k in range(6)},
        coordinator=0,
        positions=positions,
        comm_power_w=np.full(6, 0.2),
        sensing_power_w=np.full(6, 0.8),
        state_versions=np.full(6, 80, dtype=np.int64),
        certificate_epoch_ids=np.asarray([7, 7, 8, 7, 7, 7]),
        certificate_digests=np.asarray(
            [1, 1, 2, 1, 1, 1], dtype=np.uint64),
        communication_model=_model(),
        layout=layout,
    )
    assert not binding_failed.feasible
    assert "protocol:certificate_epoch:uav:2" in binding_failed.reasons
    assert "protocol:certificate_digest:uav:2" in binding_failed.reasons


def test_joint_reserve_elects_one_commit_and_feedback_coordinator():
    selected = np.zeros((3, 3, 2), dtype=bool)
    selected[0, 1, 0] = True
    role = np.asarray([0, 1, 0], dtype=np.int8)
    owner = np.asarray([1, 2], dtype=np.int64)
    proposal = selected.copy()
    proposal[0, 1, 0] = False
    proposal[0, 2, 0] = True
    move = LocalMove(
        "exchange",
        proposal,
        role.copy(),
        np.asarray([2, 2], dtype=np.int64),
    )
    positions = np.asarray([
        [0.0, 0.0, 100.0],
        [10.0, 0.0, 100.0],
        [20.0, 0.0, 100.0],
    ])
    result = minimum_joint_commit_feedback_reserve(
        selected,
        role,
        owner,
        move,
        feedback_entries_by_owner={2: 12},
        positions=positions,
        current_comm_power_w=np.zeros(3),
        current_sensing_power_w=np.full((3, 2), 0.5),
        sensing_weights=np.full((3, 2), 0.5),
        state_versions=np.full(3, 75, dtype=np.int64),
        **_identity(3),
        communication_model=_model(),
        commit_layout=DependencyCommitLayout(3, 2),
        feedback_layout=OwnerFeedbackLayout(3, 2, horizon_frames=5),
        reserve_upper_w=0.2,
        tolerance_w=1.0e-14,
    )
    assert result.feasible
    assert result.uniform_comm_floor_w is not None
    assert 0.0 < result.uniform_comm_floor_w <= 0.2
    assert result.commit_certificate.feasible
    assert result.feedback_certificate.feasible
    assert result.coordinator == result.commit_certificate.proposer
    assert result.coordinator == result.feedback_certificate.coordinator

    robust = minimum_joint_commit_feedback_reserve(
        selected,
        role,
        owner,
        move,
        feedback_entries_by_owner={2: 12},
        positions=positions,
        current_comm_power_w=np.zeros(3),
        current_sensing_power_w=np.full((3, 2), 0.5),
        sensing_weights=np.full((3, 2), 0.5),
        state_versions=np.full(3, 75, dtype=np.int64),
        **_identity(3),
        communication_model=_model(),
        commit_layout=DependencyCommitLayout(3, 2),
        feedback_layout=OwnerFeedbackLayout(3, 2, horizon_frames=5),
        reserve_upper_w=0.2,
        tolerance_w=1.0e-14,
        snr_margin_db=6.0,
        latency_margin_s=5.0e-4,
    )
    assert robust.feasible
    assert robust.uniform_comm_floor_w is not None
    assert robust.uniform_comm_floor_w > result.uniform_comm_floor_w
