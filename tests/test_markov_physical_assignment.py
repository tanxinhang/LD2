import copy

import numpy as np

from uav_isac.coordination.maxmin_power import (
    fixed_owner_gain_matrix,
    solve_fixed_structure_maxmin_power_lp,
)
from uav_isac.physical.deflection import DeflectionComputer
from uav_isac.physical.detection import compute_detection_probabilities
from uav_isac.prediction.markov_assignment import (
    solve_markov_assignment_scenario_tree,
)
from uav_isac.prediction.markov_physical_assignment import (
    MarkovPhysicalAssignmentModel,
    cubature_sigma_points,
)


def _fixture():
    dc = DeflectionComputer(
        fc=2.8e10, delta_f=1.5625e4, T_sym=6.4e-5, M=64, N=16,
        kT=4.0e-21, B=1.0e6, NF_dB=4.0,
        P_sense=0.0251, P_report=0.25, ric_K=6.0,
        rcs=1.0, g_min=0.5, rng=np.random.default_rng(61),
        g_tx_dBi=16.0, g_rx_dBi=16.0, use_los_prob=True,
        use_swerling=True, use_report_link=True, dd_gain_mode="continuous",
        sync_delay_error_bins=0.12, sync_doppler_error_bins=-0.17,
    )
    roles = np.array([0, 0, 1, 1], dtype=np.int64)
    selected = ((0, 2, 0), (1, 3, 1))
    fusion = np.array([500.0, 500.0, 0.0])
    model = MarkovPhysicalAssignmentModel(
        dc, selected, np.array([0.0251, 0.0251, 0.0, 0.0]), roles,
        fusion, num_targets=2, dt_s=0.1, movement_step_m=2.5,
        area_size_m=(1000.0, 1000.0), false_alarm_probability=0.01,
        safe_distance_m=20.0, weak_count=2, weak_weight=0.25,
        quadrature_order=16,
    )
    uav_pos = np.array([
        [100.0, 100.0, 100.0], [850.0, 150.0, 100.0],
        [150.0, 800.0, 100.0], [850.0, 850.0, 100.0],
    ])
    uav_vel = np.zeros((4, 3))
    target_pos = np.array([[400.0, 500.0, 0.0], [650.0, 400.0, 0.0]])
    target_vel = np.array([[4.0, -2.0, 0.0], [-3.0, 5.0, 0.0]])
    state = model.pack_state(uav_pos, uav_vel, target_pos, target_vel)
    return dc, model, state


def test_transition_is_bounded_and_advances_reflecting_target_cv():
    _dc, model, state = _fixture()
    next_state = model.transition(state, np.array([0, 1, 0, 1]), 0)
    uav0, _uv0, target0, _tv0 = model.unpack_state(state)
    uav1, uv1, target1, tv1 = model.unpack_state(next_state)
    travel = np.linalg.norm(uav1[:, :2] - uav0[:, :2], axis=1)
    assert np.all(travel <= 2.5 + 1.0e-12)
    np.testing.assert_allclose(target1[:, :2], target0[:, :2] + 0.1 * tv1[:, :2])
    np.testing.assert_allclose(uv1[:, :2], (uav1[:, :2] - uav0[:, :2]) / 0.1)


def test_transition_projects_crossing_uav_paths_to_safe_separation():
    _dc, model, state = _fixture()
    uav_pos, uav_vel, target_pos, target_vel = model.unpack_state(state)
    uav_pos = uav_pos.copy()
    uav_pos[0, :2] = [100.0, 100.0]
    uav_pos[1, :2] = [121.0, 100.0]
    target_pos = target_pos.copy()
    target_pos[0, :2] = [130.0, 100.0]
    target_pos[1, :2] = [90.0, 100.0]
    close_state = model.pack_state(uav_pos, uav_vel, target_pos, target_vel)
    next_state = model.transition(close_state, np.array([0, 1, 0, 1]), 0)
    next_uav, _velocity, _target, _target_velocity = model.unpack_state(next_state)
    assert np.linalg.norm(next_uav[0, :2] - next_uav[1, :2]) >= 20.0 - 1.0e-8


def test_stage_evaluation_matches_explicit_expected_physics_and_exact_lp():
    dc, model, state = _fixture()
    result = model.evaluate(state)
    uav_pos, uav_vel, target_pos, target_vel = model.unpack_state(state)
    dense = dc.compute_expected_dense(
        uav_pos, uav_vel, target_pos, target_vel,
        model.roles, model.fc_position,
        sensing_power_w=np.ones((model.K, model.Q)),
        quadrature_order=model.quadrature_order,
    )
    gain, _ = fixed_owner_gain_matrix(dense.d_eff, model.selected_edges)
    expected_lp = solve_fixed_structure_maxmin_power_lp(gain, model.budget)
    expected_pd = compute_detection_probabilities(expected_lp.deflection, model.p_fa)
    np.testing.assert_allclose(result.power_w, expected_lp.power_w)
    np.testing.assert_allclose(result.deflection, expected_lp.deflection)
    np.testing.assert_allclose(result.detection_probability, expected_pd)
    np.testing.assert_allclose(result.target_price, expected_lp.prices)
    assert result.cost == -(np.min(expected_pd) + 0.25 * np.mean(expected_pd))


def test_physical_scenario_tree_is_repeatable_and_does_not_advance_live_rng():
    dc, model, state = _fixture()
    candidates = np.array([
        [0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1],
    ])
    kwargs = dict(
        candidate_assignments=candidates,
        incumbent_assignment=candidates[0],
        initial_state=state,
        horizon_steps=3,
        transition=model.transition,
        stage_cost=model.stage_cost,
        switch_penalty=0.02,
        discount=0.95,
        beam_width=9,
    )
    before = copy.deepcopy(dc.rng.bit_generator.state)
    first = solve_markov_assignment_scenario_tree(**kwargs)
    middle = copy.deepcopy(dc.rng.bit_generator.state)
    second = solve_markov_assignment_scenario_tree(**kwargs)
    after = copy.deepcopy(dc.rng.bit_generator.state)
    assert before == middle == after
    assert first.action_indices == second.action_indices
    assert first.total_cost == second.total_cost
    for left, right in zip(first.states, second.states):
        np.testing.assert_array_equal(left, right)


def test_cubature_points_reproduce_gaussian_mean_and_covariance():
    mean = np.array([2.0, -1.0, 0.5, 4.0])
    factor = np.array([
        [1.0, 0.0, 0.0, 0.0], [0.2, 2.0, 0.0, 0.0],
        [0.1, -0.3, 0.5, 0.0], [0.0, 0.4, 0.2, 1.5],
    ])
    covariance = factor @ factor.T
    points, weights = cubature_sigma_points(mean, covariance)
    recovered_mean = np.sum(weights[:, None] * points, axis=0)
    centered = points - recovered_mean
    recovered_covariance = np.einsum("s,si,sj->ij", weights, centered, centered)
    np.testing.assert_allclose(recovered_mean, mean, atol=1.0e-14)
    np.testing.assert_allclose(recovered_covariance, covariance, atol=1.0e-13)


def test_covariance_aware_stage_integrates_marginal_target_physics_rng_free():
    dc, base, state = _fixture()
    covariance_horizon = np.zeros((2, 2, 4, 4))
    covariance_horizon[:, :, 0, 0] = 400.0
    covariance_horizon[:, :, 1, 1] = 225.0
    covariance_horizon[:, :, 2, 2] = 4.0
    covariance_horizon[:, :, 3, 3] = 4.0
    model = MarkovPhysicalAssignmentModel(
        dc, base.selected_edges, base.budget, base.roles, base.fc_position,
        num_targets=base.Q, dt_s=base.dt_s,
        movement_step_m=base.movement_step_m, area_size_m=base.area_size_m,
        false_alarm_probability=base.p_fa,
        safe_distance_m=base.safe_distance_m, weak_count=base.weak_count,
        weak_weight=base.weak_weight, quadrature_order=base.quadrature_order,
        target_covariance_horizon=covariance_horizon,
    )
    before = copy.deepcopy(dc.rng.bit_generator.state)
    first = model.evaluate(state, stage_index=0)
    second = model.evaluate(state, stage_index=0)
    after = copy.deepcopy(dc.rng.bit_generator.state)
    assert before == after
    np.testing.assert_array_equal(first.deflection, second.deflection)
    assert np.all(np.isfinite(first.detection_probability))
    assert first.feasible_geometry


def test_uncertainty_penalty_builds_a_lower_confidence_physical_score():
    dc, base, state = _fixture()
    covariance_horizon = np.zeros((1, 2, 4, 4))
    covariance_horizon[:, :, 0, 0] = 900.0
    covariance_horizon[:, :, 1, 1] = 900.0

    def build(penalty):
        return MarkovPhysicalAssignmentModel(
            dc, base.selected_edges, base.budget, base.roles, base.fc_position,
            num_targets=base.Q, dt_s=base.dt_s,
            movement_step_m=base.movement_step_m,
            area_size_m=base.area_size_m,
            false_alarm_probability=base.p_fa,
            target_covariance_horizon=covariance_horizon,
            uncertainty_penalty_std=penalty,
        )

    mean_model = build(0.0)
    conservative_model = build(1.0)
    mean_result = mean_model.evaluate(state, stage_index=0)
    conservative_result = conservative_model.evaluate(state, stage_index=0)
    assert conservative_result.worst_detection_probability <= (
        mean_result.worst_detection_probability + 1.0e-14)


def test_marginal_covariance_columns_match_explicit_full_target_evaluations():
    dc, model, state = _fixture()
    uav_pos, uav_vel, target_pos, target_vel = model.unpack_state(state)
    covariance = np.zeros((2, 4, 4))
    covariance[:, 0, 0] = 225.0
    covariance[:, 1, 1] = 100.0
    optimized = model._expected_coefficient(
        uav_pos, uav_vel, target_pos, target_vel, covariance)

    explicit = np.zeros_like(optimized)
    from uav_isac.prediction.markov_physical_assignment import (
        cubature_sigma_points,
    )
    for target in range(2):
        mean = np.array([
            target_pos[target, 0], target_pos[target, 1],
            target_vel[target, 0], target_vel[target, 1],
        ])
        points, weights = cubature_sigma_points(mean, covariance[target])
        for point, weight in zip(points, weights):
            positions = target_pos.copy()
            velocities = target_vel.copy()
            positions[target, :2] = point[:2]
            velocities[target, :2] = point[2:]
            dense = dc.compute_expected_dense(
                uav_pos, uav_vel, positions, velocities,
                model.roles, model.fc_position,
                sensing_power_w=np.ones((4, 2)),
                quadrature_order=model.quadrature_order)
            explicit[:, :, target] += weight * dense.d_eff[:, :, target]
    np.testing.assert_allclose(optimized, explicit, rtol=2.0e-14, atol=0.0)
