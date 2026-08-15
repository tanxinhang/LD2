import numpy as np

from uav_isac.coordination.bottleneck_router import RepairRoute
from uav_isac.coordination.joint_structure_relaxation import (
    certified_hierarchical_qos_route,
    dual_guided_sparse_candidate_mask,
    owner_pair_relaxed_target_ceiling,
    solve_joint_structure_maxmin_lp_relaxation,
)
from uav_isac.coordination.qos_threshold_feasibility import (
    solve_maxmin_qos_boundary_bisection,
)


def _gain() -> np.ndarray:
    gain = np.zeros((4, 4, 3), dtype=np.float64)
    gain[0, 2] = [7.0, 2.0, 4.0]
    gain[1, 2] = [5.0, 4.0, 1.0]
    gain[3, 2] = [2.0, 6.0, 5.0]
    gain[0, 3] = [6.0, 1.0, 3.0]
    gain[1, 3] = [4.0, 7.0, 2.0]
    gain[2, 3] = [1.0, 3.0, 8.0]
    gain[2, 0] = [3.0, 5.0, 6.0]
    gain[3, 0] = [2.0, 4.0, 7.0]
    return gain


def test_lp_relaxation_upper_bounds_integer_qos_boundary():
    gain = _gain()
    budget = np.asarray([0.8, 0.7, 0.9, 0.6])
    relaxation = solve_joint_structure_maxmin_lp_relaxation(
        gain,
        budget,
        target_pair_limit=2,
        reports_per_receiver=4,
    )
    boundary = solve_maxmin_qos_boundary_bisection(
        gain,
        budget,
        p_fa=0.1,
        target_pair_limit=2,
        reports_per_receiver=4,
        probability_tolerance=2.0e-3,
        max_iterations=12,
    )
    assert relaxation.upper_deflection + 1.0e-8 >= boundary.lower_deflection
    assert np.all(relaxation.target_deflection + 1.0e-7 >= -0.0)
    assert np.isclose(np.sum(relaxation.target_prices), 1.0)
    assert np.all(relaxation.tx_fraction + relaxation.rx_fraction <= 1.0 + 1e-9)


def test_owner_pair_ceiling_tightens_receiver_free_relaxation():
    gain = _gain()
    budget = np.asarray([0.8, 0.7, 0.9, 0.6])
    owner_ceiling = owner_pair_relaxed_target_ceiling(
        gain, budget, target_pair_limit=2)
    receiver_free = np.sum(np.max(gain, axis=1) * budget[:, None], axis=0)
    assert np.all(owner_ceiling <= receiver_free + 1.0e-12)
    assert np.any(owner_ceiling < receiver_free - 1.0e-6)


def test_dual_sparse_candidates_preserve_target_owner_ceiling():
    gain = _gain()
    budget = np.asarray([0.8, 0.7, 0.9, 0.6])
    relaxation = solve_joint_structure_maxmin_lp_relaxation(
        gain,
        budget,
        target_pair_limit=2,
        reports_per_receiver=4,
    )
    candidate = dual_guided_sparse_candidate_mask(
        gain,
        budget,
        relaxation,
        target_pair_limit=2,
        minimum_owner_groups_per_target=1,
        additional_owner_groups=2,
        transmitters_per_owner=2,
    )
    np.testing.assert_allclose(candidate.retained_ceiling_ratio, 1.0)
    assert candidate.candidate_edge_count < candidate.full_edge_count
    assert candidate.candidate_edge_count <= 2 * (gain.shape[2] + 2)


def test_relaxation_and_sparse_ceiling_are_permutation_equivariant():
    gain = _gain()
    budget = np.asarray([0.8, 0.7, 0.9, 0.6])
    base = solve_joint_structure_maxmin_lp_relaxation(
        gain, budget, target_pair_limit=2, reports_per_receiver=4)
    base_candidate = dual_guided_sparse_candidate_mask(
        gain,
        budget,
        base,
        target_pair_limit=2,
        minimum_owner_groups_per_target=1,
        additional_owner_groups=2,
        transmitters_per_owner=2,
    )
    uav_permutation = np.asarray([2, 0, 3, 1])
    target_permutation = np.asarray([1, 2, 0])
    permuted_gain = gain[
        uav_permutation[:, None, None],
        uav_permutation[None, :, None],
        target_permutation[None, None, :],
    ]
    permuted = solve_joint_structure_maxmin_lp_relaxation(
        permuted_gain,
        budget[uav_permutation],
        target_pair_limit=2,
        reports_per_receiver=4,
    )
    permuted_candidate = dual_guided_sparse_candidate_mask(
        permuted_gain,
        budget[uav_permutation],
        permuted,
        target_pair_limit=2,
        minimum_owner_groups_per_target=1,
        additional_owner_groups=2,
        transmitters_per_owner=2,
    )
    assert np.isclose(base.upper_deflection, permuted.upper_deflection)
    np.testing.assert_allclose(
        np.sort(base.target_prices), np.sort(permuted.target_prices), atol=1e-8)
    np.testing.assert_allclose(
        np.sort(base_candidate.retained_ceiling_ratio),
        np.sort(permuted_candidate.retained_ceiling_ratio),
    )
    assert base_candidate.candidate_edge_count == (
        permuted_candidate.candidate_edge_count)


def test_hierarchical_route_uses_true_relaxation_upper_bound():
    gain = np.zeros((3, 3, 2), dtype=np.float64)
    gain[0, 2, 0] = 20.0
    gain[1, 2, 1] = 20.0
    selected = ((0, 2, 0), (1, 2, 1))
    result = certified_hierarchical_qos_route(
        gain,
        selected,
        np.full(3, 0.8),
        current_lower_pd=0.2,
        p_fa=0.1,
        qos_floor=0.6,
        target_pair_limit=1,
        reports_per_receiver=2,
    )
    assert result.decision.route == RepairRoute.POWER
    assert result.fixed_power is not None
    assert result.joint_relaxation is None
