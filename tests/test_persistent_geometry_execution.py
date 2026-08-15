import numpy as np

from uav_isac.coordination.persistent_geometry_execution import (
    FrozenTokenInbox,
    replace_with_geometry_command,
    start_committed_joint_plan_tube,
)
from uav_isac.coordination.causal_joint_plan import (
    commit_zero_order_hold_joint_plan,
)
from uav_isac.environment.communication import InterUAVCommunicationModel


def _model():
    return InterUAVCommunicationModel(
        rate_bits_per_dim=[0, 4], header_bits=64,
        bandwidth_hz=100_000.0, deadline_s=0.005,
        processing_delay_s=0.0002, snr_threshold_db=0.0,
        antenna_gain_dbi=0.0, carrier_hz=28.0e9,
        tx_power_w=0.25, kT=4.0e-21, noise_figure_db=4.0,
        dt=0.1, message_dim=4,
    )


def test_recorded_mobility_is_not_added_to_geometry_command():
    recorded = np.ones((2, 2), dtype=np.float64)
    assert np.array_equal(
        replace_with_geometry_command(recorded, None, frame=1), recorded)


def test_rejected_commitment_executes_as_atomic_baseline_transaction():
    commitment = commit_zero_order_hold_joint_plan(
        np.asarray([[0.5, 0.0], [-0.5, 0.0]]),
        np.asarray([0.2, 0.2]),
        np.asarray([[0.4], [0.4]]),
        decision_frame=4,
        horizon_steps=3,
    )
    selected = np.zeros((2, 2, 1), dtype=bool)
    selected[0, 1, 0] = True
    tube = start_committed_joint_plan_tube(
        commitment,
        selected=selected,
        role=np.asarray([0, 1]),
        certified_support=selected,
        escrow_flight_energy_per_step_j=10.0,
    )
    assert not tube.proof_available
    assert tube.execution_mode == "committed_baseline_only"
    assert np.array_equal(tube.command(5), commitment.movement_plan_m[0])
    assert np.array_equal(
        tube.escrow_flight_energy_plan_j, np.full((3, 2), 10.0))


def test_geometry_tube_execution_mode_is_explicit():
    from uav_isac.coordination.certified_geometry_repair import (
        CertifiedGeometryRepairConfig,
        certified_trust_region_geometry_repair,
    )
    from uav_isac.coordination.owner_local_physics import (
        OwnerLocalKinematicState,
        OwnerTargetInvariantCache,
    )
    from uav_isac.coordination.persistent_geometry_execution import (
        start_certified_geometry_tube,
    )
    from uav_isac.coordination.target_invariant_transport import (
        TargetInvariantWireLayout,
    )

    positions = np.asarray([
        [100.0, 200.0, 20.0], [300.0, 200.0, 20.0]])
    targets = np.zeros((2, 1, 4), dtype=np.float64)
    targets[:, 0, :2] = 200.0
    state = OwnerLocalKinematicState(
        uav_position_m=positions,
        uav_velocity_mps=np.zeros((2, 3)),
        target_mean_by_owner=targets,
        target_cov_diag_by_owner=np.zeros((2, 1, 4)),
        target_aoi_frames_by_owner=np.zeros((2, 1)),
    )
    selected = np.zeros((2, 2, 1), dtype=bool)
    selected[0, 1, 0] = True
    coefficient = np.zeros_like(selected, dtype=np.float64)
    coefficient[0, 1, 0] = 0.05
    target_position = np.asarray([200.0, 200.0, 0.0])
    invariant = (
        0.05
        * np.sum((positions[0] - target_position) ** 2)
        * np.sum((positions[1] - target_position) ** 2))
    layout = TargetInvariantWireLayout(2, 1)
    cache = OwnerTargetInvariantCache(
        target_invariant=np.asarray([
            layout.quantize_invariant_lower(invariant)]),
        age_frames=np.zeros(1, dtype=np.int64),
        version=np.ones(1, dtype=np.int64),
    )
    decision = certified_trust_region_geometry_repair(
        coefficient, coefficient, state, cache, selected, selected,
        np.asarray([1]), np.asarray([[0.5], [0.0]]), np.full(2, 0.25),
        np.full(2, 1000.0), communication_model=_model(),
        control_period_s=0.1, p_fa=0.1, qos_floor=0.6, dt_s=0.1,
        max_speed_mps=25.0, area_size_m=(400.0, 400.0),
        safe_separation_m=20.0, static_flight_power_w=80.0,
        quadratic_flight_power_coeff=0.05, carrier_hz=28.0e9,
        delta_f_hz=15_625.0, symbol_period_s=64.0e-6,
        delay_bins=64, doppler_bins=16, covariance_radius=0.0,
        dd_support_threshold=0.0, dd_additive_margin=0.0,
        residual_log_margin=0.0,
        config=CertifiedGeometryRepairConfig(
            minimum_worst_pd_improvement=1.0e-7),
        invariant_layout=layout,
        joint_plan_digest="0123456789abcdef",
    )
    repair = start_certified_geometry_tube(
        decision, decision_frame=1, selected=selected,
        role=np.asarray([0, 1]), certified_support=selected,
        origin_position_m=state.uav_position_m,
        execution_mode="repair")
    baseline = start_certified_geometry_tube(
        decision, decision_frame=1, selected=selected,
        role=np.asarray([0, 1]), certified_support=selected,
        origin_position_m=state.uav_position_m,
        execution_mode="committed_baseline")
    assert repair.joint_plan_digest == "0123456789abcdef"
    assert baseline.joint_plan_digest == repair.joint_plan_digest
    assert not np.array_equal(repair.movement_plan_m, baseline.movement_plan_m)


def test_token_visibility_is_recomputed_at_modified_geometry():
    initial = FrozenTokenInbox.from_observation(
        np.zeros((2, 2, 1), dtype=bool),
        np.full((2, 2, 1), np.inf),
        ttl_frames=1,
    )
    messages = np.zeros((2, 4), dtype=np.float64)
    rates = np.ones(2, dtype=np.int64)
    masks = np.ones((2, 1), dtype=np.float64)
    close, stats = initial.advance(
        messages, rates, masks,
        np.asarray([[0.0, 0.0, 20.0], [20.0, 0.0, 20.0]]),
        np.full(2, 0.25), _model(),
    )
    assert stats.delivered_links == 2
    assert close.visible[0, 1, 0]
    assert close.visible[1, 0, 0]

    far, stats = close.advance(
        messages, rates, masks,
        np.asarray([[0.0, 0.0, 20.0], [1.0e8, 0.0, 20.0]]),
        np.full(2, 0.25), _model(),
    )
    assert stats.delivered_links == 0
    assert far.visible[0, 1, 0]
    assert far.visible[1, 0, 0]
    assert not far.visible[0, 0, 0]
    assert not far.visible[1, 1, 0]
    expired, _ = far.advance(
        messages, np.zeros(2, dtype=np.int64), masks,
        np.asarray([[0.0, 0.0, 20.0], [1.0e8, 0.0, 20.0]]),
        np.full(2, 0.25), _model(),
    )
    assert not np.any(expired.visible)


def test_common_geometry_delivery_commits_only_branch_intersection():
    initial = FrozenTokenInbox.from_observation(
        np.zeros((2, 2, 1), dtype=bool),
        np.full((2, 2, 1), np.inf),
        ttl_frames=1,
    )
    messages = np.zeros((2, 4), dtype=np.float64)
    rates = np.ones(2, dtype=np.int64)
    masks = np.ones((2, 1), dtype=np.float64)
    close = np.asarray([
        [0.0, 0.0, 20.0], [20.0, 0.0, 20.0]])
    far = np.asarray([
        [0.0, 0.0, 20.0], [1.0e8, 0.0, 20.0]])
    common, energy = initial.advance_common_geometry_deliveries(
        messages, rates, masks, close, far, np.full(2, 0.25), _model())

    assert not np.any(common.visible)
    assert np.all(energy >= 0.0)
