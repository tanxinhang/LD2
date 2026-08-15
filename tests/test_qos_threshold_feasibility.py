import numpy as np

from uav_isac.coordination.qos_threshold_feasibility import (
    solve_maxmin_qos_boundary_bisection,
    solve_qos_threshold_feasibility_milp,
)


def _two_pair_gain() -> np.ndarray:
    gain = np.zeros((4, 4, 2), dtype=np.float64)
    gain[0, 2, 0] = 4.0
    gain[1, 3, 1] = 4.0
    return gain


def test_threshold_milp_respects_roles_owners_capacity_and_power():
    result = solve_qos_threshold_feasibility_milp(
        _two_pair_gain(),
        np.ones(4),
        requested_pd=0.6,
        p_fa=0.1,
        target_pair_limit=1,
        reports_per_receiver=1,
    )

    assert result.feasible
    assert result.selected_set == ((0, 2, 0), (1, 3, 1))
    assert np.array_equal(result.tx_role, [True, True, False, False])
    assert np.array_equal(result.rx_role, [False, False, True, True])
    assert np.array_equal(result.receiver_owner, [2, 3])
    assert np.all(result.target_pd >= 0.6 - 1.0e-8)
    assert np.all(np.sum(result.sensing_power_w, axis=1) <= 1.0 + 1.0e-9)


def test_threshold_milp_proves_role_conflict_infeasible():
    gain = np.zeros((3, 3, 2), dtype=np.float64)
    gain[0, 1, 0] = 8.0
    gain[1, 2, 1] = 8.0
    result = solve_qos_threshold_feasibility_milp(
        gain,
        np.ones(3),
        requested_pd=0.7,
        p_fa=0.1,
        target_pair_limit=1,
        reports_per_receiver=1,
    )

    assert not result.feasible
    assert result.proven_infeasible


def test_bisection_returns_a_valid_fairness_bracket():
    result = solve_maxmin_qos_boundary_bisection(
        _two_pair_gain(),
        np.ones(4),
        p_fa=0.1,
        target_pair_limit=1,
        reports_per_receiver=1,
        initial_feasible_pd=0.6,
        probability_tolerance=2.0e-3,
        max_iterations=12,
    )

    assert result.best_feasible.feasible
    assert result.lower_pd <= np.min(result.best_feasible.target_pd) + 1.0e-8
    assert result.upper_pd - result.lower_pd <= 2.0e-3
    assert result.exact_infeasible_upper


def test_bisection_rejects_an_infeasible_initial_floor():
    gain = np.zeros((3, 3, 2), dtype=np.float64)
    gain[0, 1, 0] = 8.0
    gain[1, 2, 1] = 8.0
    try:
        solve_maxmin_qos_boundary_bisection(
            gain,
            np.ones(3),
            p_fa=0.1,
            target_pair_limit=1,
            reports_per_receiver=1,
            initial_feasible_pd=0.7,
        )
    except RuntimeError as error:
        assert "initial detection floor" in str(error)
    else:
        raise AssertionError("an infeasible initial floor must be rejected")


def test_threshold_input_is_permutation_equivariant():
    gain = _two_pair_gain()
    base = solve_qos_threshold_feasibility_milp(
        gain,
        np.ones(4),
        requested_pd=0.6,
        p_fa=0.1,
        target_pair_limit=1,
        reports_per_receiver=1,
    )
    uav_permutation = np.asarray([2, 0, 3, 1])
    target_permutation = np.asarray([1, 0])
    permuted_gain = gain[
        uav_permutation[:, None, None],
        uav_permutation[None, :, None],
        target_permutation[None, None, :],
    ]
    permuted = solve_qos_threshold_feasibility_milp(
        permuted_gain,
        np.ones(4)[uav_permutation],
        requested_pd=0.6,
        p_fa=0.1,
        target_pair_limit=1,
        reports_per_receiver=1,
    )

    assert base.feasible and permuted.feasible
    assert np.allclose(
        np.sort(base.target_pd), np.sort(permuted.target_pd), atol=1.0e-8)
