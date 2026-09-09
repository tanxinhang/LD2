import numpy as np
import pytest

from config.params import load_config
from uav_isac.coordination.hyperedge import (
    annular_alternating_movement_delta,
    bistatic_geometry_tail_ratio,
    deterministic_bottleneck_cost_assignment,
    deterministic_bottleneck_matching,
    decode_offer_stream,
    encode_offer_stream,
    factorized_endpoint_capabilities,
    gauss_southwell_bistatic_geometry_step,
    gauss_southwell_bistatic_geometry_sweep,
    mutual_endpoint_consensus,
    plan_budget_certified_hyperedges,
    plan_local_hyperedges,
    plan_refined_endpoint_hyperedges,
    plan_reserved_endpoint_hyperedges,
    project_pairwise_safe_movement,
    is_pairwise_safe_movement,
    plan_sparse_coalition_endpoint_hyperedges,
    refine_role_mask_local_search,
    reconstruct_bistatic_coefficient_from_public_state,
    reconstruct_bistatic_coefficient_upper_from_public_state,
    reconstruct_selected_bistatic_coefficients_from_public_state,
    reconstruct_selected_bistatic_coefficients_batch_from_public_state,
    reconstruct_bistatic_pair_value,
    role_aware_bistatic_movement_cost,
    select_sparse_tx_coalition_role,
    update_consensus_streak,
)
from uav_isac.environment.env_wrapper import UAVISACEnv


def _plan(tx, rx, visible, deficit):
    return plan_local_hyperedges(
        tx, rx, visible, deficit,
        target_pair_limit=1,
        deficit_gain=2.0,
        proxy_floor=0.25,
    )


def test_offer_stream_round_trip_is_bounded():
    tx = np.array([0.1, 0.7, 1.2])
    rx = np.array([0.9, 0.2, -0.1])
    gap = np.array([0.0, 0.4, 1.0])
    decoded = decode_offer_stream(encode_offer_stream(tx, rx, gap))
    np.testing.assert_allclose(decoded[:, 0], np.clip(tx, 0.0, 1.0))
    np.testing.assert_allclose(decoded[:, 1], np.clip(rx, 0.0, 1.0))
    np.testing.assert_allclose(decoded[:, 2], np.clip(gap, 0.0, 1.0))


def test_selected_coefficient_kernel_matches_dense_nominal_and_certificates():
    rng = np.random.default_rng(1701)
    K, Q = 6, 4
    positions = rng.uniform(0.0, 800.0, size=(K, Q, 2))
    velocities = rng.uniform(-20.0, 20.0, size=(K, Q, 2))
    targets = np.column_stack((
        rng.uniform(0.0, 800.0, size=(Q, 2)), np.zeros(Q)))
    target_velocities = np.column_stack((
        rng.uniform(-5.0, 5.0, size=(Q, 2)), np.zeros(Q)))
    visible = rng.random((K, Q)) > 0.15
    edges = ((0, 1, 0), (2, 3, 1), (4, 5, 2), (0, 3, 3))
    nominal_uncertainty = np.linspace(0.5, 2.0, K)
    certificate_uncertainty = np.linspace(1.0, 3.0, K)
    target_uncertainty = np.linspace(2.0, 5.0, Q)
    velocity_uncertainty = np.linspace(0.1, 0.6, K)
    target_velocity_uncertainty = np.linspace(0.2, 0.8, Q)
    common = dict(
        uav_height_m=20.0,
        fc_hz=28.0e9,
        rcs_m2=1.0,
        delta_f_hz=15625.0,
        symbol_period_s=6.4e-5,
        delay_bins=64,
        doppler_bins=16,
        dd_gate_min=0.5,
        coefficient_scale=1.0e15,
        dd_gain_mode="continuous",
    )
    nominal = reconstruct_bistatic_coefficient_from_public_state(
        positions, velocities, targets, target_velocities, visible,
        position_uncertainty_m=nominal_uncertainty,
        robust_dd_uncertainty=False,
        **common,
    )
    lower = reconstruct_bistatic_coefficient_from_public_state(
        positions, velocities, targets, target_velocities, visible,
        position_uncertainty_m=certificate_uncertainty,
        target_position_uncertainty_m=target_uncertainty,
        velocity_uncertainty_mps=velocity_uncertainty,
        target_velocity_uncertainty_mps=target_velocity_uncertainty,
        robust_dd_uncertainty=True,
        **common,
    )
    upper = reconstruct_bistatic_coefficient_upper_from_public_state(
        positions, targets, visible,
        uav_height_m=common["uav_height_m"],
        fc_hz=common["fc_hz"],
        rcs_m2=common["rcs_m2"],
        coefficient_scale=common["coefficient_scale"],
        position_uncertainty_m=certificate_uncertainty,
        target_position_uncertainty_m=target_uncertainty,
    )
    sparse = reconstruct_selected_bistatic_coefficients_from_public_state(
        positions, velocities, targets, target_velocities, visible, edges,
        nominal_position_uncertainty_m=nominal_uncertainty,
        certificate_position_uncertainty_m=certificate_uncertainty,
        target_position_uncertainty_m=target_uncertainty,
        velocity_uncertainty_mps=velocity_uncertainty,
        target_velocity_uncertainty_mps=target_velocity_uncertainty,
        **common,
    )
    index = tuple(np.asarray(edges, dtype=np.int64).T)
    np.testing.assert_allclose(sparse.nominal, nominal[index], rtol=1e-13)
    np.testing.assert_allclose(sparse.lower, lower[index], rtol=1e-13)
    np.testing.assert_allclose(sparse.upper, upper[index], rtol=1e-13)


def test_batched_selected_kernel_matches_each_private_view():
    rng = np.random.default_rng(1702)
    V, K, Q = 5, 6, 4
    positions = rng.uniform(0.0, 800.0, size=(V, K, Q, 2))
    velocities = rng.uniform(-20.0, 20.0, size=(V, K, Q, 2))
    targets = np.concatenate((
        rng.uniform(0.0, 800.0, size=(V, Q, 2)),
        np.zeros((V, Q, 1)),
    ), axis=2)
    target_velocities = np.concatenate((
        rng.uniform(-5.0, 5.0, size=(V, Q, 2)),
        np.zeros((V, Q, 1)),
    ), axis=2)
    visible = rng.random((V, K, Q)) > 0.15
    edges = ((0, 1, 0), (2, 3, 1), (4, 5, 2), (0, 3, 3))
    nominal_uncertainty = rng.uniform(0.5, 2.0, size=(V, K))
    certificate_uncertainty = rng.uniform(1.0, 3.0, size=(V, K))
    target_uncertainty = rng.uniform(2.0, 5.0, size=(V, Q))
    velocity_uncertainty = rng.uniform(0.1, 0.6, size=(V, K))
    target_velocity_uncertainty = rng.uniform(0.2, 0.8, size=(V, Q))
    common = dict(
        uav_height_m=20.0,
        fc_hz=28.0e9,
        rcs_m2=1.0,
        delta_f_hz=15625.0,
        symbol_period_s=6.4e-5,
        delay_bins=64,
        doppler_bins=16,
        dd_gate_min=0.5,
        coefficient_scale=1.0e15,
        dd_gain_mode="continuous",
    )
    batched = reconstruct_selected_bistatic_coefficients_batch_from_public_state(
        positions,
        velocities,
        targets,
        target_velocities,
        visible,
        edges,
        nominal_position_uncertainty_m=nominal_uncertainty,
        certificate_position_uncertainty_m=certificate_uncertainty,
        target_position_uncertainty_m=target_uncertainty,
        velocity_uncertainty_mps=velocity_uncertainty,
        target_velocity_uncertainty_mps=target_velocity_uncertainty,
        **common,
    )
    for viewer in range(V):
        single = reconstruct_selected_bistatic_coefficients_from_public_state(
            positions[viewer],
            velocities[viewer],
            targets[viewer],
            target_velocities[viewer],
            visible[viewer],
            edges,
            nominal_position_uncertainty_m=nominal_uncertainty[viewer],
            certificate_position_uncertainty_m=(
                certificate_uncertainty[viewer]),
            target_position_uncertainty_m=target_uncertainty[viewer],
            velocity_uncertainty_mps=velocity_uncertainty[viewer],
            target_velocity_uncertainty_mps=(
                target_velocity_uncertainty[viewer]),
            **common,
        )
        np.testing.assert_allclose(
            batched.nominal[viewer], single.nominal, rtol=1e-13)
        np.testing.assert_allclose(
            batched.lower[viewer], single.lower, rtol=1e-13)
        np.testing.assert_allclose(
            batched.upper[viewer], single.upper, rtol=1e-13)


def test_bottleneck_matching_minimizes_worst_travel_not_greedy_edge():
    nodes = np.array([[18.0, 4.0], [10.0, 5.0], [0.0, 15.0]])
    targets = np.array([[1.0, 5.0], [9.0, 9.0], [2.0, 19.0]])
    assignment = deterministic_bottleneck_matching(nodes, targets)
    distances = np.linalg.norm(
        nodes - targets[assignment], axis=1)
    np.testing.assert_allclose(np.max(distances), 10.295630140987)
    assert sorted(assignment.tolist()) == [0, 1, 2]


def test_generic_bottleneck_assignment_minimizes_arbitrary_cost():
    cost = np.asarray([
        [1.0, 2.0, 100.0],
        [2.0, 100.0, 1.0],
        [100.0, 1.0, 2.0],
    ])
    assignment = deterministic_bottleneck_cost_assignment(cost)
    np.testing.assert_array_equal(assignment, [0, 2, 1])
    assert np.max(cost[np.arange(3), assignment]) == pytest.approx(1.0)


def test_role_aware_matching_reduces_bistatic_product_bottleneck():
    nodes = np.asarray([
        [89.2552328811, 20.0275379918],
        [41.4741217076, 71.3155187564],
        [73.8102595904, 45.1177826679],
        [63.8618576608, 66.8918889573],
    ])
    targets = np.asarray([
        [35.3122794762, 65.2073573735],
        [57.7422460713, 69.7385953568],
        [51.1645027287, 33.6431083862],
        [43.4402220761, 92.2461339567],
    ])
    roles = np.asarray([True, False, True, False])
    cost = role_aware_bistatic_movement_cost(
        nodes, targets, roles,
        height_m=20.0, movement_step_m=2.5)
    travel_assignment = deterministic_bottleneck_matching(nodes, targets)
    bistatic_assignment = deterministic_bottleneck_cost_assignment(cost)

    travel_worst = np.max(cost[np.arange(4), travel_assignment])
    bistatic_worst = np.max(cost[np.arange(4), bistatic_assignment])
    assert bistatic_worst < 0.6 * travel_worst


def test_gauss_southwell_selects_one_coordinate_and_reduces_tail():
    nodes = np.asarray([
        [0.0, 0.0],
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 0.0],
    ])
    targets = np.asarray([[0.0, 0.0], [100.0, 0.0]])
    roles = np.asarray([True, False, True, False])
    desired, winner, gain, before, after = (
        gauss_southwell_bistatic_geometry_step(
            nodes,
            targets,
            roles,
            height_m=20.0,
            maximum_step_m=2.5,
        ))
    assert winner == (2, 1)
    assert np.flatnonzero(np.linalg.norm(desired, axis=1)).tolist() == [2]
    assert np.linalg.norm(desired[2]) == pytest.approx(2.5)
    assert gain > 0.0
    assert after < before
    assert gain == pytest.approx(np.log(before / after))


def test_gauss_southwell_threshold_can_hold_geometry_block():
    desired, winner, gain, before, after = (
        gauss_southwell_bistatic_geometry_step(
            np.asarray([[0.0, 0.0], [10.0, 0.0]]),
            np.asarray([[100.0, 0.0]]),
            np.asarray([True, False]),
            height_m=20.0,
            maximum_step_m=2.5,
            minimum_log_improvement=10.0,
        ))
    assert winner is None
    assert gain == 0.0
    assert before == pytest.approx(after)
    np.testing.assert_array_equal(desired, np.zeros((2, 2)))


def test_gauss_southwell_sweep_uses_distinct_nodes_and_is_monotone():
    nodes = np.asarray([
        [0.0, 0.0], [0.0, 20.0], [20.0, 0.0], [20.0, 20.0],
    ])
    targets = np.asarray([[100.0, 0.0], [100.0, 20.0]])
    roles = np.asarray([True, True, False, False])
    desired, coordinates, gain, before, after = (
        gauss_southwell_bistatic_geometry_sweep(
            nodes,
            targets,
            roles,
            height_m=20.0,
            maximum_step_m=2.5,
        ))
    selected_nodes = [node for node, _target in coordinates]
    assert len(selected_nodes) > 1
    assert len(selected_nodes) == len(set(selected_nodes))
    assert np.sum(np.linalg.norm(desired, axis=1) > 1.0e-12) == len(
        selected_nodes)
    assert gain > 0.0
    assert after < before


@pytest.mark.parametrize(
    ('node', 'phase', 'expected_state', 'expected_delta'),
    [
        ([300.0, 0.0], 'strategy', 'far_recovery', [-2.5, 0.0]),
        ([50.0, 0.0], 'strategy', 'near_recovery', [2.5, 0.0]),
        ([150.0, 0.0], 'range', 'range_hold', [0.0, 0.0]),
    ],
)
def test_annular_alternating_range_block_has_boundary_priority(
    node,
    phase,
    expected_state,
    expected_delta,
):
    delta, state = annular_alternating_movement_delta(
        np.asarray(node),
        np.zeros(2),
        np.asarray([100.0, 0.0]),
        phase=phase,
        inner_radius_m=75.0,
        outer_radius_m=225.0,
        maximum_step_m=2.5,
    )
    assert state == expected_state
    np.testing.assert_allclose(delta, expected_delta, atol=1.0e-12)


def test_annular_strategy_block_preserves_range_and_improves_angle():
    node = np.asarray([150.0, 0.0])
    target = np.zeros(2)
    complement = np.asarray([100.0, 0.0])
    delta, state = annular_alternating_movement_delta(
        node,
        target,
        complement,
        phase='strategy',
        inner_radius_m=75.0,
        outer_radius_m=225.0,
        maximum_step_m=2.5,
        desired_bistatic_angle_deg=90.0,
        orientation_sign=1,
    )
    next_node = node + delta
    initial_angle = np.arctan2(node[1], node[0])
    next_angle = np.arctan2(next_node[1], next_node[0])
    assert state == 'strategy_tangent'
    assert next_angle > initial_angle
    assert np.linalg.norm(delta) <= 2.5 + 1.0e-12
    assert np.linalg.norm(next_node - target) == pytest.approx(
        np.linalg.norm(node - target), abs=1.0e-10)


def test_bistatic_tail_ratio_is_scale_free_at_zero_height():
    nodes = np.asarray([
        [0.0, 0.0], [0.0, 2.0], [8.0, 0.0], [8.0, 2.0],
    ])
    targets = np.asarray([[1.0, 1.0], [4.0, 1.0], [20.0, 1.0]])
    roles = np.asarray([True, False, True, False])
    ratio = bistatic_geometry_tail_ratio(
        nodes, targets, roles, height_m=0.0)
    scaled = bistatic_geometry_tail_ratio(
        7.0 * nodes, 7.0 * targets, roles, height_m=0.0)
    assert ratio > 1.0
    assert scaled == pytest.approx(ratio)


def test_bistatic_tail_gate_latch_survives_environment_state_restore():
    cfg = load_config(
        "config/exp_800_k12q12_distributed_replicated_l1_"
        "nearfield32_inertia25_bistatic_move_gamma50_tail40.yaml")
    env = UAVISACEnv(config=cfg, seed=725)
    env.reset()
    expected = np.asarray([1, 0] * 6, dtype=np.int8)
    env.core._distributed_bistatic_gate_latch[:] = expected
    state = env.core.get_state()
    env.core._distributed_bistatic_gate_latch[:] = -1
    env.core.set_state(state)
    np.testing.assert_array_equal(
        env.core._distributed_bistatic_gate_latch, expected)


def test_ack_free_local_movement_assignment_survives_state_restore():
    cfg = load_config(
        "config/exp_800_k10q10_distributed_tail40_safe_v2_"
        "selfstabilizing.yaml")
    env = UAVISACEnv(config=cfg, seed=725)
    env.reset()
    expected = np.tile(
        np.arange(env.core.K, dtype=np.int64),
        (env.core.K, 1),
    )
    env.core._distributed_movement_local_assignment[:] = expected
    state = env.core.get_state()
    env.core._distributed_movement_local_assignment[:] = -1
    env.core.set_state(state)
    np.testing.assert_array_equal(
        env.core._distributed_movement_local_assignment, expected)
    assert env.core._distributed_movement_local_assignment_cache_enabled
    assert env.core._distributed_movement_public_max_age_frames == 1
    assert env.core._distributed_movement_stale_fail_closed
    assert env.core._distributed_movement_execute_projected_public_action


def test_anchor_beacon_reuses_charged_header_and_survives_restore():
    cfg = load_config(
        'config/exp_800_k12q12_distributed_v2_dynamic_u2u_'
        'robustbelief_anchor_feedback5_pilot.yaml')
    env = UAVISACEnv(config=cfg, seed=451)
    env.reset(seed=451)
    core = env.core
    core.t = 0
    core._prepare_hyperedge_submission()

    sender, receiver, target = 1, 0, 1
    protocol = core._pending_hyperedge_protocol[sender]
    assert core._pending_hyperedge_protocol_frame[sender] == 0
    original_dim = 7  # x,y,vx,vy plus the multiplexed three-field header
    assert protocol.shape == (core.Q, original_dim)
    np.testing.assert_allclose(
        core._hyperedge_local_offer[sender, :, 4:7],
        np.tile(np.asarray([1.0, 0.5, 0.5]), (core.Q, 1)),
    )
    core.t = 1  # packet generated at submit(0), merged during step(1)
    core._merge_received_hyperedge_packet(
        receiver,
        sender,
        protocol,
        core._pending_comm_token_masks[sender],
        sent_frame=0,
    )
    assert core._distributed_movement_anchor_last_seen[
        receiver, sender, target] == 0
    expected = core.belief_mgr.mean[sender, target, :2]
    received = core._distributed_movement_anchor_target_xy[
        receiver, sender, target]
    assert np.linalg.norm(received - expected) <= (
        np.linalg.norm(core.area_size[:2]) / 255.0 + 1.0e-9)
    np.testing.assert_allclose(
        core._hyperedge_received_offer[receiver, sender, :, 4:7],
        np.tile(np.asarray([1.0, 0.5, 0.5]), (core.Q, 1)),
    )

    state = core.get_state()
    core._distributed_movement_anchor_last_seen[:] = -10**9
    core.set_state(state)
    assert core._distributed_movement_anchor_last_seen[
        receiver, sender, target] == 0


def test_anchor_beacon_uses_current_transmit_frame_through_physical_transport():
    cfg = load_config(
        'config/exp_800_k12q12_distributed_v2_dynamic_u2u_'
        'robustbelief_anchor_feedback5_pilot.yaml')
    cfg.scenario.T = 2
    env = UAVISACEnv(config=cfg, seed=451)
    env.reset(seed=451)
    core = env.core
    messages = {
        k: np.zeros(core.Q * core._comm_target_token_dim)
        for k in range(core.K)
    }
    core.submit_learned_communications(
        messages,
        {k: 1 for k in range(core.K)},
        {k: 0.2 for k in range(core.K)},
        {k: np.full(core.Q, 1.0 / core.Q) for k in range(core.K)},
        token_masks={k: np.ones(core.Q) for k in range(core.K)},
    )
    # Fresh encoding now occurs after the current action is applied.  Put the
    # next step on an anchor-broadcast frame and verify that the transported
    # timestamp is that frame, not the earlier action-submission frame.
    core.t = 4
    actions = {
        str(k): {'delta_p': np.zeros(2), 'role': 2}
        for k in range(core.K)
    }
    env.step(actions)
    coverage = np.mean(np.sum(np.any(
        core._distributed_movement_anchor_last_seen > -10**8,
        axis=1,
    ), axis=1) / core.Q)
    assert coverage > 0.95
    assert np.max(core._distributed_movement_anchor_last_seen) == 5


def test_movement_anchor_map_uses_recent_source_median():
    cfg = load_config(
        'config/exp_800_k12q12_distributed_v2_dynamic_u2u_'
        'robustbelief_anchor_feedback5_pilot.yaml')
    env = UAVISACEnv(config=cfg, seed=451)
    env.reset(seed=451)
    core = env.core
    core.t = 10
    viewer, target = 0, 0
    core.belief_mgr.mean[viewer, target, 2:4] = 0.0
    core._distributed_movement_anchor_target_xy[
        viewer, :3, target] = np.asarray([
            [100.0, 100.0],
            [102.0, 98.0],
            [700.0, 700.0],
        ])
    core._distributed_movement_anchor_last_seen[
        viewer, :3, target] = 10
    position, _ = core._movement_target_state_for_viewer(viewer)
    np.testing.assert_allclose(position[target, :2], [102.0, 100.0])


def test_stale_public_view_returns_explicit_zero_movement_overrides():
    cfg = load_config(
        "config/exp_800_k10q10_distributed_tail40_safe_v2_"
        "selfstabilizing.yaml")
    env = UAVISACEnv(config=cfg, seed=725)
    env.reset()
    movement = env.core._distributed_greedy_matching_movement_delta()
    assert set(movement) == set(range(env.core.K))
    for value in movement.values():
        np.testing.assert_array_equal(value, np.zeros(2))


def test_dynamic_distributed_coordination_uses_viewer_local_belief_targets():
    cfg = load_config(
        "config/exp_800_k10q10_distributed_tail40_safe_v2_"
        "selfstabilizing.yaml")
    cfg.marl.tracking_enabled = True
    cfg.marl.distributed_coordination_use_local_belief_targets = True
    cfg.marl.distributed_movement_stale_fail_closed = False
    cfg.marl.distributed_movement_safety_projection_enabled = False
    cfg.marl.distributed_id_movement_standoff_m = 0.0
    env = UAVISACEnv(config=cfg, seed=725)
    env.reset()

    origin = np.asarray(env.core.uavs[0].pos[:2], dtype=np.float64)
    env.core.belief_mgr.mean[0, 0, :2] = origin + np.asarray([200.0, 0.0])
    env.core.belief_mgr.mean[0, 0, 2:4] = np.asarray([3.0, -2.0])
    position, velocity = env.core._coordination_target_state_for_viewer(0)
    np.testing.assert_allclose(position[0, :2], origin + [200.0, 0.0])
    np.testing.assert_allclose(velocity[0, :2], [3.0, -2.0])

    movement = env.core._distributed_greedy_matching_movement_delta()
    assert 0 in movement
    assert movement[0][0] > 0.0
    assert abs(movement[0][1]) <= 1.0e-10


def test_strict_local_coordination_never_falls_back_to_target_truth(monkeypatch):
    cfg = load_config(
        "config/exp_800_k10q10_distributed_tail40_safe_v2_"
        "selfstabilizing.yaml")
    cfg.marl.tracking_enabled = True
    cfg.marl.distributed_coordination_use_local_belief_targets = True
    cfg.marl.distributed_movement_stale_fail_closed = False
    env = UAVISACEnv(config=cfg, seed=725)
    env.reset()

    def forbidden_truth_read(*_args, **_kwargs):
        raise AssertionError('distributed coordination read target truth')

    target_type = type(env.core.targets[0])
    monkeypatch.setattr(target_type, 'get_position_3d', forbidden_truth_read)
    monkeypatch.setattr(target_type, 'get_velocity', forbidden_truth_read)

    env.core._prepare_hyperedge_submission()
    env.core._resolve_hyperedge_negotiation()
    env.core._distributed_greedy_matching_movement_delta()


def test_gain_scheduled_block_alternates_and_stays_behind_d_safe_shield():
    cfg = load_config(
        'config/exp_800_k12q12_distributed_v2_dynamic_u2u_'
        'robustbelief_gain_schedule2_pilot.yaml')
    env = UAVISACEnv(config=cfg, seed=451)
    env.reset(seed=451)
    core = env.core
    assert core._distributed_gain_scheduled_movement_enabled
    assert core._distributed_movement_safety_projection_enabled
    assert core._distributed_movement_execute_projected_public_action

    public_xy = np.asarray([
        [100.0 + 100.0 * (node % 4),
         100.0 + 100.0 * (node // 4)]
        for node in range(core.K)
    ])
    target_xy = np.asarray([
        [150.0 + 70.0 * (target % 4),
         550.0 + 50.0 * (target // 4)]
        for target in range(core.Q)
    ])
    assignment = np.arange(core.K, dtype=np.int64) % core.Q
    movement_step = float(core.uavs[0].v_max * core.uavs[0].dt)

    core.t = 2
    range_desired, range_states = core._distributed_public_desired_movement(
        public_xy, target_xy, assignment, movement_step)
    assert set(range_states) == {'gain_far_range'}
    assert np.all(np.linalg.norm(range_desired, axis=1) > 1.0e-12)

    # Isolate the near-regime scheduler after verifying the configured
    # reachability gate; deployment never disables this gate at runtime.
    core._distributed_gain_scheduled_far_range_gate_m = 0.0
    desired, states = core._distributed_public_desired_movement(
        public_xy, target_xy, assignment, movement_step)
    assert states.count('gain_selected') > 1
    assert np.sum(np.linalg.norm(desired, axis=1) > 1.0e-12) == (
        states.count('gain_selected'))

    projected = project_pairwise_safe_movement(
        public_xy,
        desired,
        minimum_distance_m=float(cfg.uav.d_safe),
        maximum_step_m=movement_step,
        area_size_xy=tuple(core.area_size[:2]),
    )
    next_xy = public_xy + projected
    pair_distance = np.linalg.norm(
        next_xy[:, None, :] - next_xy[None, :, :], axis=-1)
    assert np.min(pair_distance[np.triu_indices(core.K, k=1)]) >= (
        float(cfg.uav.d_safe) - 1.0e-7)

    core.t = 3
    held, states = core._distributed_public_desired_movement(
        public_xy, target_xy, assignment, movement_step)
    np.testing.assert_array_equal(held, np.zeros_like(held))
    assert set(states) == {'gain_strategy_hold'}


def test_held_protocol_edge_is_not_filtered_by_simulator_physical_truth():
    cfg = load_config(
        'config/exp_800_q4_architecture_v2_hyperedge_gate.yaml')
    env = UAVISACEnv(config=cfg, seed=29)
    env.reset(seed=29)
    core = env.core
    held = ((0, 1, 0),)
    core._hyperedge_selected_set = held
    core._hyperedge_last_update_frame = core.t
    core._hyperedge_assignment_hold_frames = 5
    core._distributed_replicated_power_enabled = False

    assert core._resolve_hyperedge_negotiation() == held


def test_pairwise_safety_projection_prevents_near_pair_approach():
    positions = np.asarray([[40.0, 50.0], [61.0, 50.0], [90.0, 90.0]])
    desired = np.asarray([[2.5, 0.0], [-2.5, 0.0], [0.0, -2.5]])
    projected = project_pairwise_safe_movement(
        positions,
        desired,
        minimum_distance_m=20.0,
        maximum_step_m=2.5,
        area_size_xy=(100.0, 100.0),
    )
    next_positions = positions + projected
    distance = np.linalg.norm(
        next_positions[:, None, :] - next_positions[None, :, :], axis=-1)
    assert np.min(distance[np.triu_indices(3, k=1)]) >= 20.0 - 1e-7
    assert np.all(np.linalg.norm(projected, axis=1) <= 2.5 + 1e-7)
    relative = positions[0] - positions[1]
    assert np.dot(
        positions[0] - positions[1],
        projected[0] - projected[1],
    ) >= 0.5 * (20.0 ** 2 - np.dot(relative, relative)) - 1e-7
    times = np.linspace(0.0, 1.0, 1001)
    relative_path = (
        relative[None, :]
        + times[:, None] * (projected[0] - projected[1])[None, :])
    assert np.min(np.linalg.norm(relative_path, axis=1)) >= 20.0 - 1e-7


def test_pairwise_safety_projection_is_noop_when_motion_is_safe():
    positions = np.asarray([[20.0, 20.0], [80.0, 20.0]])
    desired = np.asarray([[0.0, 2.5], [0.0, 2.5]])
    projected = project_pairwise_safe_movement(
        positions,
        desired,
        minimum_distance_m=20.0,
        maximum_step_m=2.5,
        area_size_xy=(100.0, 100.0),
    )
    np.testing.assert_allclose(projected, desired)
    assert is_pairwise_safe_movement(
        positions,
        desired,
        minimum_distance_m=20.0,
        maximum_step_m=2.5,
        area_size_xy=(100.0, 100.0),
    )


def test_pairwise_safety_noop_certificate_rejects_intervention():
    positions = np.asarray([[40.0, 50.0], [61.0, 50.0]])
    desired = np.asarray([[2.5, 0.0], [-2.5, 0.0]])
    assert not is_pairwise_safe_movement(
        positions,
        desired,
        minimum_distance_m=20.0,
        maximum_step_m=2.5,
        area_size_xy=(100.0, 100.0),
    )
    # Jointly safe equal motion is not independently composable: if the
    # second packet is stale and holds, the first endpoint would get too close.
    equal_motion = np.asarray([[2.5, 0.0], [2.5, 0.0]])
    assert is_pairwise_safe_movement(
        positions,
        equal_motion,
        minimum_distance_m=20.0,
        maximum_step_m=2.5,
        area_size_xy=(100.0, 100.0),
    )
    assert not is_pairwise_safe_movement(
        positions,
        equal_motion,
        minimum_distance_m=20.0,
        maximum_step_m=2.5,
        area_size_xy=(100.0, 100.0),
        independently_composable=True,
    )

    # The projector rescales even an arbitrarily small strict speed excess;
    # an exact no-op certificate must therefore reject it as well.
    over_step = np.asarray([[2.5 + 1.0e-12, 0.0], [0.0, 0.0]])
    assert not is_pairwise_safe_movement(
        positions,
        over_step,
        minimum_distance_m=20.0,
        maximum_step_m=2.5,
        area_size_xy=(100.0, 100.0),
    )


def test_pairwise_safety_projection_reports_intervention_diagnostics():
    positions = np.asarray([[40.0, 50.0], [61.0, 50.0]])
    desired = np.asarray([[2.5, 0.0], [-2.5, 0.0]])
    projected, diagnostics = project_pairwise_safe_movement(
        positions,
        desired,
        minimum_distance_m=20.0,
        maximum_step_m=2.5,
        area_size_xy=(100.0, 100.0),
        return_diagnostics=True,
    )
    assert diagnostics['intervened'] is True
    assert diagnostics['fail_closed'] is False
    assert diagnostics['solve_time_s'] >= 0.0
    assert diagnostics['pairwise_constraint_count'] == 1
    assert diagnostics['initially_safe'] is True
    assert diagnostics['minimum_initial_distance_m'] == pytest.approx(21.0)
    assert np.linalg.norm(
        positions[0] + projected[0] - positions[1] - projected[1]
    ) >= 20.0 - 1.0e-7


def test_pairwise_safety_projection_recovers_outside_invariant_gradually():
    positions = np.asarray([[40.0, 50.0], [61.0, 50.0]])
    desired = np.asarray([[2.5, 0.0], [-2.5, 0.0]])
    projected, diagnostics = project_pairwise_safe_movement(
        positions,
        desired,
        minimum_distance_m=32.0,
        maximum_step_m=2.5,
        area_size_xy=(100.0, 100.0),
        return_diagnostics=True,
        outside_invariant_recovery=True,
        recovery_gain=1.0,
    )
    next_distance = np.linalg.norm(
        positions[0] + projected[0]
        - positions[1] - projected[1])
    assert diagnostics['initially_safe'] is False
    assert diagnostics['recovery_pair_count'] == 1
    assert diagnostics['fail_closed'] is False
    assert next_distance >= 21.0 - 1.0e-7
    assert next_distance > 21.0 + 1.0e-3


def test_pairwise_safety_projection_is_composable_with_stale_peer_hold():
    """A fresh local action plus a stale peer hold preserves separation."""
    positions = np.asarray([[40.0, 50.0], [61.0, 50.0]])
    desired = np.asarray([[2.5, 0.0], [2.5, 0.0]])
    ordinary = project_pairwise_safe_movement(
        positions,
        desired,
        minimum_distance_m=20.0,
        maximum_step_m=2.5,
        area_size_xy=(100.0, 100.0),
    )
    ordinary_assembled = np.asarray([ordinary[0], np.zeros(2)])
    ordinary_distance = np.linalg.norm(
        positions[0] + ordinary_assembled[0]
        - positions[1] - ordinary_assembled[1])
    assert ordinary_distance < 20.0

    composable = project_pairwise_safe_movement(
        positions,
        desired,
        minimum_distance_m=20.0,
        maximum_step_m=2.5,
        area_size_xy=(100.0, 100.0),
        independently_composable=True,
    )
    assembled = np.asarray([composable[0], np.zeros(2)])
    next_distance = np.linalg.norm(
        positions[0] + assembled[0] - positions[1] - assembled[1])
    assert next_distance >= 20.0 - 1.0e-7


def test_composable_linear_qp_reduction_is_exact_and_reported():
    """Removing redundant speed balls preserves the unique projection."""
    positions = np.asarray([
        [30.0, 30.0], [51.0, 30.0], [72.0, 30.0],
        [30.0, 51.0], [51.0, 51.0], [72.0, 51.0],
    ])
    desired = np.asarray([
        [2.5, 0.0], [-2.5, 0.0], [0.0, -2.5],
        [1.8, 1.7], [-1.9, -1.6], [0.0, 2.5],
    ])
    kwargs = dict(
        minimum_distance_m=20.0,
        maximum_step_m=2.5,
        area_size_xy=(100.0, 100.0),
        independently_composable=True,
        return_diagnostics=True,
    )
    reference, reference_info = project_pairwise_safe_movement(
        positions, desired, analytic_composable_projection=False, **kwargs)
    reduced, reduced_info = project_pairwise_safe_movement(
        positions, desired, analytic_composable_projection=True, **kwargs)

    assert reference_info['projection_solver'] == 'slsqp'
    assert reduced_info['projection_solver'] == 'reduced_linear_qp'
    assert not reference_info['fail_closed']
    assert not reduced_info['fail_closed']
    np.testing.assert_allclose(reduced, reference, atol=1.0e-9, rtol=1.0e-9)


def test_composable_linear_qp_reduction_matches_randomized_full_problem():
    rng = np.random.default_rng(20260830)
    base = np.asarray([
        [30.0, 30.0], [51.0, 30.0], [72.0, 30.0],
        [30.0, 51.0], [51.0, 51.0], [72.0, 51.0],
    ])
    for _ in range(40):
        positions = base + rng.uniform(-0.35, 0.35, size=base.shape)
        desired = rng.normal(0.0, 2.5, size=base.shape)
        kwargs = dict(
            minimum_distance_m=20.0,
            maximum_step_m=2.5,
            area_size_xy=(100.0, 100.0),
            independently_composable=True,
        )
        reference = project_pairwise_safe_movement(
            positions,
            desired,
            analytic_composable_projection=False,
            **kwargs,
        )
        reduced = project_pairwise_safe_movement(
            positions,
            desired,
            analytic_composable_projection=True,
            **kwargs,
        )
        np.testing.assert_allclose(
            reduced, reference, atol=2.0e-8, rtol=2.0e-8)
        next_positions = positions + reduced
        pair_distance = np.linalg.norm(
            next_positions[:, None, :] - next_positions[None, :, :],
            axis=-1,
        )[np.triu_indices(positions.shape[0], k=1)]
        assert np.min(pair_distance) >= 20.0 - 1.0e-7
        assert np.max(np.linalg.norm(reduced, axis=1)) <= 2.5 + 1.0e-7


def test_inverse_square_endpoint_capability_is_bounded_and_physical():
    distance = np.array([50.0, 150.0, 300.0])
    sensing = np.array([0.25, 0.25, 0.25])
    tx, rx = factorized_endpoint_capabilities(
        distance,
        sensing,
        distance_scale_m=150.0,
        mode="inverse_square",
    )
    assert np.all((0.0 <= tx) & (tx <= 1.0))
    assert np.all((0.0 <= rx) & (rx <= 1.0))
    assert rx[0] > rx[1] > rx[2]
    np.testing.assert_allclose(tx, 0.5 * rx)


def test_public_state_pair_reconstruction_needs_visible_endpoints():
    tx = np.full((3, 1), 0.8)
    positions = np.array([
        [[10.0, 0.0]],
        [[90.0, 0.0]],
        [[50.0, 0.0]],
    ])
    targets = np.array([[0.0, 0.0]])
    visible = np.ones((3, 1), dtype=bool)
    value = reconstruct_bistatic_pair_value(
        tx,
        positions,
        targets,
        visible,
        distance_scale_m=150.0,
    )
    assert value.shape == (3, 3, 1)
    assert np.max(value) == 1.0
    assert value[0, 2, 0] > value[1, 2, 0]

    visible[0, 0] = False
    masked = reconstruct_bistatic_pair_value(
        tx,
        positions,
        targets,
        visible,
        distance_scale_m=150.0,
    )
    np.testing.assert_allclose(masked[0, :, 0], 0.0)
    np.testing.assert_allclose(masked[:, 0, 0], 0.0)


def test_public_state_coefficient_reconstruction_is_absolute_and_fail_closed():
    positions = np.array([
        [[10.0, 0.0]],
        [[80.0, 0.0]],
        [[160.0, 0.0]],
    ])
    velocities = np.zeros_like(positions)
    targets = np.array([[0.0, 0.0, 0.0]])
    target_velocities = np.zeros((1, 3))
    visible = np.ones((3, 1), dtype=bool)
    coefficient = reconstruct_bistatic_coefficient_from_public_state(
        positions,
        velocities,
        targets,
        target_velocities,
        visible,
        uav_height_m=20.0,
        fc_hz=28.0e9,
        rcs_m2=1.0,
        delta_f_hz=15_625.0,
        symbol_period_s=64.0e-6,
        delay_bins=64,
        doppler_bins=16,
        dd_gate_min=0.0,
        coefficient_scale=1.0,
    )
    assert coefficient.shape == (3, 3, 1)
    assert coefficient[0, 1, 0] > coefficient[2, 1, 0] > 0.0
    np.testing.assert_allclose(np.diagonal(
        coefficient[:, :, 0]), 0.0)
    robust = reconstruct_bistatic_coefficient_from_public_state(
        positions, velocities, targets, target_velocities, visible,
        uav_height_m=20.0, fc_hz=28.0e9, rcs_m2=1.0,
        delta_f_hz=15_625.0, symbol_period_s=64.0e-6,
        delay_bins=64, doppler_bins=16, dd_gate_min=0.0,
        coefficient_scale=1.0, position_uncertainty_m=3.0,
    )
    assert np.all(robust <= coefficient + 1.0e-15)
    assert robust[0, 1, 0] < coefficient[0, 1, 0]
    target_robust = reconstruct_bistatic_coefficient_from_public_state(
        positions, velocities, targets, target_velocities, visible,
        uav_height_m=20.0, fc_hz=28.0e9, rcs_m2=1.0,
        delta_f_hz=15_625.0, symbol_period_s=64.0e-6,
        delay_bins=64, doppler_bins=16, dd_gate_min=0.0,
        coefficient_scale=1.0,
        target_position_uncertainty_m=np.asarray([5.0]),
    )
    assert np.all(target_robust <= coefficient + 1.0e-15)
    assert target_robust[0, 1, 0] < coefficient[0, 1, 0]
    local_exact = reconstruct_bistatic_coefficient_from_public_state(
        positions, velocities, targets, target_velocities, visible,
        uav_height_m=20.0, fc_hz=28.0e9, rcs_m2=1.0,
        delta_f_hz=15_625.0, symbol_period_s=64.0e-6,
        delay_bins=64, doppler_bins=16, dd_gate_min=0.0,
        coefficient_scale=1.0,
        position_uncertainty_m=np.asarray([0.0, 3.0, 3.0]),
    )
    assert local_exact[0, 1, 0] > robust[0, 1, 0]
    assert local_exact[2, 1, 0] == pytest.approx(robust[2, 1, 0])
    with pytest.raises(ValueError, match="scalar or K-vector"):
        reconstruct_bistatic_coefficient_from_public_state(
            positions, velocities, targets, target_velocities, visible,
            uav_height_m=20.0, fc_hz=28.0e9, rcs_m2=1.0,
            delta_f_hz=15_625.0, symbol_period_s=64.0e-6,
            delay_bins=64, doppler_bins=16, dd_gate_min=0.0,
            coefficient_scale=1.0,
            position_uncertainty_m=np.ones(2),
        )

    visible[0, 0] = False
    masked = reconstruct_bistatic_coefficient_from_public_state(
        positions,
        velocities,
        targets,
        target_velocities,
        visible,
        uav_height_m=20.0,
        fc_hz=28.0e9,
        rcs_m2=1.0,
        delta_f_hz=15_625.0,
        symbol_period_s=64.0e-6,
        delay_bins=64,
        doppler_bins=16,
        dd_gate_min=0.0,
        coefficient_scale=1.0,
    )
    np.testing.assert_allclose(masked[0, :, 0], 0.0)
    np.testing.assert_allclose(masked[:, 0, 0], 0.0)


def test_budget_certified_public_plan_obeys_roles_owners_and_budgets():
    coefficient = np.zeros((4, 4, 2), dtype=np.float64)
    coefficient[0, 3, 0] = 100.0
    coefficient[1, 3, 1] = 90.0
    coefficient[2, 3, :] = 40.0
    budget = np.array([0.02, 0.03, 0.04, 0.0])
    plan = plan_budget_certified_hyperedges(
        coefficient,
        budget,
        np.array([0.8, 0.2]),
        p_fa=0.001,
        p_d_floor=0.60,
        target_pair_limit=2,
        reports_per_receiver=4,
    )
    assert plan.selected
    tx = {i for i, _j, _q in plan.selected}
    rx = {j for _i, j, _q in plan.selected}
    assert tx.isdisjoint(rx)
    for q in range(2):
        edges = [edge for edge in plan.selected if edge[2] == q]
        assert 1 <= len(edges) <= 2
        assert len({edge[1] for edge in edges}) == 1
    assert np.all(plan.proxy_target_value > 0.0)


def test_role_refinement_repairs_a_bad_fixed_bipartition():
    gain = np.full((4, 4, 2), 1.0e-4)
    for node in range(4):
        gain[node, node] = 0.0
    # Under the initial even/odd cut, both dominant pairs lie within a role.
    gain[0, 2, 0] = gain[2, 0, 0] = 1.0
    gain[1, 3, 1] = gain[3, 1, 1] = 1.0
    budget = np.ones(4)
    initial = np.array([True, False, True, False])
    refined = refine_role_mask_local_search(
        gain, budget, target_pair_limit=1,
        initial_role_mask=initial,
    )
    assert refined[0] != refined[2]
    assert refined[1] != refined[3]


def test_refined_endpoint_plans_are_reciprocal_on_common_public_state():
    gain = np.full((4, 4, 2), 0.05)
    for node in range(4):
        gain[node, node] = 0.0
    gain[0, 2, 0] = gain[2, 0, 0] = 2.0
    gain[1, 3, 1] = gain[3, 1, 1] = 2.0
    visible = np.ones((4, 2), dtype=bool)
    plans = [
        plan_refined_endpoint_hyperedges(
            gain, visible, np.ones(4), viewer=viewer,
            target_pair_limit=1,
        )
        for viewer in range(4)
    ]
    mutual = mutual_endpoint_consensus(
        plans, num_uavs=4, num_targets=2, target_pair_limit=1)
    assert {target for _, _, target in mutual} == {0, 1}
    assert not ({tx for tx, _, _ in mutual} & {rx for _, rx, _ in mutual})


def test_sparse_coalition_is_bounded_and_budget_coupled():
    gain = np.full((5, 5, 3), 0.1)
    for node in range(5):
        gain[node, node] = 0.0
    # Three specialist transmitters can each support one target, but the
    # configured degree-two coalition must make an explicit budget tradeoff.
    gain[0, 4, 0] = 8.0
    gain[1, 4, 1] = 8.0
    gain[2, 4, 2] = 8.0
    role = select_sparse_tx_coalition_role(
        gain, np.ones(5), target_pair_limit=2, max_transmitters=2)
    assert 1 <= int(np.sum(role)) <= 2
    plans = [
        plan_sparse_coalition_endpoint_hyperedges(
            gain, np.ones((5, 3), dtype=bool), np.ones(5),
            viewer=viewer, target_pair_limit=2, max_transmitters=2,
        )
        for viewer in range(5)
    ]
    assert all(np.array_equal(plan.role_mask, role) for plan in plans)


def test_reserved_endpoint_consensus_needs_only_two_endpoint_views():
    coefficient = np.zeros((4, 4, 2), dtype=np.float64)
    coefficient[0, 1, 0] = 10.0
    coefficient[2, 1, 0] = 5.0
    coefficient[0, 3, 1] = 4.0
    coefficient[2, 3, 1] = 9.0
    budget = np.ones(4)
    role = np.array([True, False, True, False])
    plans = []
    for viewer in range(4):
        visible = np.zeros((4, 2), dtype=bool)
        visible[viewer] = True
        # Only the two endpoints need reciprocal packet state. Viewers do not
        # share a globally identical four-node snapshot.
        if viewer == 0:
            visible[1, 0] = True
        elif viewer == 1:
            visible[0, 0] = True
        elif viewer == 2:
            visible[3, 1] = True
        elif viewer == 3:
            visible[2, 1] = True
        plans.append(plan_reserved_endpoint_hyperedges(
            coefficient,
            visible,
            budget,
            viewer=viewer,
            tx_role_mask=role,
            target_pair_limit=1,
        ))
    mutual = mutual_endpoint_consensus(
        plans,
        num_uavs=4,
        num_targets=2,
        target_pair_limit=1,
    )
    assert mutual == ((0, 1, 0), (2, 3, 1))


def test_reserved_information_greedy_prefers_complementary_geometry():
    coefficient = np.zeros((4, 4, 1), dtype=np.float64)
    coefficient[0, 3, 0] = 10.0
    coefficient[1, 3, 0] = 9.0
    coefficient[2, 3, 0] = 8.0
    information = np.zeros((4, 4, 1, 4, 4), dtype=np.float64)
    # Tx 0 and 1 repeat one strong direction; Tx 2 is slightly weaker in
    # Deflection but supplies an independent position direction.
    information[0, 3, 0, 0, 0] = 10.0
    information[1, 3, 0, 0, 0] = 10.0
    information[2, 3, 0, 1, 1] = 10.0
    plan = plan_reserved_endpoint_hyperedges(
        coefficient,
        np.ones((4, 1), dtype=bool),
        np.ones(4),
        viewer=3,
        tx_role_mask=np.asarray([True, True, True, False]),
        target_pair_limit=2,
        normalized_edge_information=information,
        information_weight=1.0,
    )
    assert set(plan.selected) == {(0, 3, 0), (2, 3, 0)}


def test_local_plan_enforces_directed_single_roles_and_target_capacity():
    tx = np.array([
        [0.95, 0.90],
        [0.85, 0.80],
        [0.20, 0.15],
        [0.10, 0.25],
    ])
    rx = np.array([
        [0.10, 0.20],
        [0.15, 0.10],
        [0.90, 0.85],
        [0.80, 0.95],
    ])
    plan = _plan(tx, rx, np.ones_like(tx, dtype=bool), np.array([0.6, 0.4]))
    assert len(plan.selected) == 2
    tx_nodes = {edge[0] for edge in plan.selected}
    rx_nodes = {edge[1] for edge in plan.selected}
    assert not (tx_nodes & rx_nodes)
    assert {edge[2] for edge in plan.selected} == {0, 1}


def test_local_plan_is_equivariant_without_score_ties():
    tx = np.array([
        [0.91, 0.62],
        [0.73, 0.81],
        [0.22, 0.31],
        [0.16, 0.27],
    ])
    rx = np.array([
        [0.18, 0.21],
        [0.25, 0.17],
        [0.88, 0.74],
        [0.69, 0.93],
    ])
    visible = np.ones_like(tx, dtype=bool)
    deficit = np.array([0.7, 0.3])
    base = _plan(tx, rx, visible, deficit)

    permutation = np.array([2, 0, 3, 1])
    inverse = np.argsort(permutation)
    permuted = _plan(
        tx[permutation], rx[permutation], visible[permutation], deficit)
    mapped = tuple(sorted(
        (int(permutation[edge[0]]), int(permutation[edge[1]]), edge[2])
        for edge in permuted.selected
    ))
    # ``permutation[new] = old`` maps the permuted node index back to the
    # original index. The inverse is deliberately kept to catch map reversal.
    assert inverse.shape == permutation.shape
    assert mapped == base.selected


def test_reconstructable_pair_value_overrides_scalar_endpoint_proxy():
    tx = np.array([[0.9], [0.8], [0.1]])
    rx = np.array([[0.1], [0.2], [0.9]])
    visible = np.ones_like(tx, dtype=bool)
    pair_value = np.zeros((3, 3, 1))
    pair_value[1, 2, 0] = 9.0
    pair_value[0, 2, 0] = 1.0
    plan = plan_local_hyperedges(
        tx, rx, visible, np.array([0.8]),
        target_pair_limit=1,
        deficit_gain=2.0,
        proxy_floor=0.25,
        pair_value=pair_value,
    )
    assert plan.selected == ((1, 2, 0),)


def test_local_plan_uses_one_receiver_owner_per_target():
    tx = np.array([
        [0.95],
        [0.90],
        [0.20],
        [0.15],
    ])
    rx = np.array([
        [0.10],
        [0.15],
        [0.92],
        [0.88],
    ])
    plan = plan_local_hyperedges(
        tx,
        rx,
        np.ones_like(tx, dtype=bool),
        np.array([0.8]),
        target_pair_limit=2,
        deficit_gain=2.0,
        proxy_floor=0.25,
    )
    assert len(plan.selected) == 2
    assert len({receiver for _, receiver, _ in plan.selected}) == 1


def test_mutual_consensus_requires_both_endpoints():
    tx = np.array([
        [0.95],
        [0.10],
        [0.20],
    ])
    rx = np.array([
        [0.10],
        [0.90],
        [0.50],
    ])
    full = np.ones_like(tx, dtype=bool)
    plans = [_plan(tx, rx, full, np.array([0.8])) for _ in range(3)]
    mutual = mutual_endpoint_consensus(
        plans, num_uavs=3, num_targets=1, target_pair_limit=1)
    assert mutual == plans[0].selected

    # Receiver 1 loses sender 0's offer. Its local plan no longer endorses the
    # same directed edge, so the edge must fail closed.
    lost = full.copy()
    lost[0, 0] = False
    plans[1] = _plan(tx, rx, lost, np.array([0.8]))
    mutual = mutual_endpoint_consensus(
        plans, num_uavs=3, num_targets=1, target_pair_limit=1)
    assert mutual == tuple()


def test_consensus_streak_requires_configured_rounds():
    previous = np.zeros((3, 3, 2), dtype=np.int64)
    edge = (0, 1, 0)
    first, active_first = update_consensus_streak(
        previous, [edge], consensus_rounds=2)
    assert active_first == tuple()
    second, active_second = update_consensus_streak(
        first, [edge], consensus_rounds=2)
    assert active_second == (edge,)
    reset, active_reset = update_consensus_streak(
        second, [], consensus_rounds=2)
    assert active_reset == tuple()
    assert np.max(reset) == 0


def test_physical_hyperedge_stream_is_charged_and_uses_two_rounds():
    cfg = load_config(
        'config/exp_800_q4_architecture_v2_hyperedge_gate.yaml')
    cfg.scenario.T = 3
    env = UAVISACEnv(cfg, seed=37)
    env.reset(seed=37)
    core = env.core
    K, Q = core.K, core.Q
    token_dim = core._comm_target_token_dim
    messages = {k: np.zeros(Q * token_dim) for k in range(K)}
    rates = {k: 1 for k in range(K)}
    fractions = {k: 0.2 for k in range(K)}
    weights = {k: np.full(Q, 1.0 / Q) for k in range(K)}
    actions = {
        str(k): {'delta_p': np.zeros(2), 'role': 2}
        for k in range(K)
    }

    protocol_used = []
    for _ in range(2):
        core.submit_learned_communications(
            messages, rates, fractions, weights,
            token_masks={k: np.ones(Q) for k in range(K)},
        )
        assert all(
            protocol.shape == (Q, 3)
            for protocol in core._pending_hyperedge_protocol.values())
        _, _, _, _, info = env.step(actions)
        protocol_used.append(info['hyperedge_protocol_used'])
        expected_per_sender = 64 + 4 * Q * (token_dim + 3)
        assert info['learned_comm_bits'] == K * expected_per_sender
        combined = (
            info['isac_per_uav_comm_power_w']
            + info['isac_per_uav_sensing_power_w']
        )
        assert np.all(combined <= env.cfg.uav.P_isac_total + 1e-12)
        assert np.all(
            info['isac_per_uav_sensing_power_w']
            <= env.cfg.uav.P_sense_max + 1e-12)
        assert info['isac_max_power_budget_violation_w'] < 1e-12

    assert protocol_used == [0.0, 1.0]


def test_hyperedge_snapshot_restore_is_exact():
    cfg = load_config(
        'config/exp_800_q4_architecture_v2_hyperedge_gate.yaml')
    cfg.scenario.T = 2
    env = UAVISACEnv(cfg, seed=41)
    env.reset(seed=41)
    core = env.core
    core._hyperedge_local_offer[:] = 0.37
    core._hyperedge_received_offer[:] = 0.21
    core._hyperedge_received_last_seen[:] = 3
    core._hyperedge_consensus_streak[0, 1, 0] = 2
    core._hyperedge_selected_set = ((0, 1, 0),)
    core._hyperedge_last_update_frame = 7
    snapshot = core.get_state()

    core._hyperedge_local_offer[:] = 0.0
    core._hyperedge_received_offer[:] = 0.0
    core._hyperedge_received_last_seen[:] = -99
    core._hyperedge_consensus_streak[:] = 0
    core._hyperedge_selected_set = tuple()
    core._hyperedge_last_update_frame = -99
    core.set_state(snapshot)

    np.testing.assert_allclose(
        core._hyperedge_local_offer,
        snapshot['hyperedge_local_offer'])
    np.testing.assert_allclose(
        core._hyperedge_received_offer,
        snapshot['hyperedge_received_offer'])
    np.testing.assert_array_equal(
        core._hyperedge_received_last_seen,
        snapshot['hyperedge_received_last_seen'])
    np.testing.assert_array_equal(
        core._hyperedge_consensus_streak,
        snapshot['hyperedge_consensus_streak'])
    assert core._hyperedge_selected_set == ((0, 1, 0),)
    assert core._hyperedge_last_update_frame == 7


def test_budget_reconstructable_relay_merges_physical_origin_state():
    cfg = load_config(
        'config/exp_800_k8q8_distributed_replicated_l1_nearfield32_'
        'inertia25_snr41_relay.yaml')
    cfg.scenario.T = 2
    env = UAVISACEnv(cfg, seed=9)
    env.reset(seed=9)
    core = env.core
    assert core._hyperedge_base_state_dim == 7
    assert core._hyperedge_protocol_dim == 15

    own_state = np.linspace(0.1, 0.7, 7)
    relay_state = np.linspace(0.2, 0.8, 7)
    decoded = np.concatenate([
        own_state,
        np.asarray([2.0 / cfg.scenario.K]),
        relay_state,
    ])
    protocol = np.repeat(
        (2.0 * decoded - 1.0)[None, :], cfg.scenario.Q, axis=0)
    token_mask = np.zeros(cfg.scenario.Q)
    token_mask[0] = 1.0

    core._merge_received_hyperedge_packet(
        receiver=1, sender=0, protocol=protocol, token_mask=token_mask)

    np.testing.assert_allclose(
        core._hyperedge_received_offer[1, 0, :, :7],
        np.repeat(own_state[None, :], cfg.scenario.Q, axis=0),
    )
    np.testing.assert_allclose(
        core._hyperedge_received_offer[1, 2, :, :7],
        np.repeat(relay_state[None, :], cfg.scenario.Q, axis=0),
    )
    assert np.all(core._hyperedge_received_last_seen[1, 2] == (
        core.t - core._comm_message_ttl_frames))
    env.close()


def test_hyperedge_round_robin_beacons_use_deterministic_sender_cohorts():
    cfg = load_config(
        'config/exp_800_k12q12_distributed_v2_dynamic_u2u_'
        'robustbelief_gapcoverage_enveloped_pilot.yaml')
    cfg.scenario.T = 2
    env = UAVISACEnv(config=cfg, seed=451)
    env.reset(seed=451)
    core = env.core
    assert core._distributed_replicated_power_process_parallel_enabled
    assert core._distributed_replicated_power_process_workers == 4
    assert core._hyperedge_beacon_round_robin_period == 2
    assert core._hyperedge_protocol_header_bits == 10
    assert core._hyperedge_state_field_bits == (8, 8, 8, 8, 4, 8, 8)
    assert core._hyperedge_state_codec_service_envelope_enabled
    assert core._evidence_packet_layout.header_bits == 10
    assert core._evidence_packet_layout.timestamp_bits == 0
    assert core._evidence_service_envelope_layout.header_bits == 64

    core.t = 0
    core._prepare_hyperedge_submission()
    assert set(core._pending_hyperedge_protocol) == set(range(0, core.K, 2))
    assert set(core._pending_hyperedge_protocol_frame.values()) == {0}

    core.t = 1
    core._prepare_hyperedge_submission()
    assert set(core._pending_hyperedge_protocol) == set(range(1, core.K, 2))
    assert set(core._pending_hyperedge_protocol_frame.values()) == {1}
    env.close()
