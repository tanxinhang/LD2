import numpy as np

from uav_isac.physical.feasibility_oracle import (
    reachable_hungarian_geometry,
    solve_joint_pair_power_oracle,
)


def _synthetic_coefficients():
    # Four UAVs, two targets.  Each ordered pair has a different but positive
    # per-watt deflection so role/power optimization is non-trivial.
    coefficient = np.zeros((4, 4, 2), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            coefficient[i, j, 0] = 8.0 + i + 0.25 * j
            coefficient[i, j, 1] = 7.0 + j + 0.25 * i
    return coefficient


def test_reachable_hungarian_geometry_respects_travel_limit():
    uav = np.array([
        [0.0, 0.0, 20.0],
        [100.0, 0.0, 20.0],
    ])
    target = np.array([
        [20.0, 0.0, 0.0],
        [200.0, 0.0, 0.0],
    ])
    final, direction, assignment = reachable_hungarian_geometry(
        uav, target, travel_distance_m=30.0)
    np.testing.assert_allclose(final[:, 2], 20.0)
    np.testing.assert_allclose(final[0, :2], [20.0, 0.0])
    np.testing.assert_allclose(final[1, :2], [130.0, 0.0])
    np.testing.assert_array_equal(assignment, [0, 1])
    assert np.linalg.norm(direction[1, :2]) == 1.0


def test_single_role_oracle_respects_power_and_pair_capacities():
    solution = solve_joint_pair_power_oracle(
        _synthetic_coefficients(),
        P_FA=0.001,
        total_power_w=1.0,
        communication_reserve_w=0.25,
        target_pair_limit=2,
        reports_per_receiver=2,
        full_duplex=False,
        alternating_iterations=4,
        random_starts=1,
        seed=7,
    )
    assert solution.mode == 'single_role'
    assert set(solution.tx_indices).isdisjoint(solution.rx_indices)
    np.testing.assert_array_less(
        solution.sensing_power_w.sum(axis=1),
        np.full(4, 0.7500001))
    for i in range(4):
        if i not in solution.tx_indices:
            np.testing.assert_allclose(solution.sensing_power_w[i], 0.0)
    for q in range(2):
        assert sum(edge[2] == q for edge in solution.selected_set) <= 2
    for j in range(4):
        assert sum(edge[1] == j for edge in solution.selected_set) <= 2
    assert solution.worst > 0.001


def test_full_duplex_oracle_is_not_worse_than_single_role():
    coefficient = _synthetic_coefficients()
    single = solve_joint_pair_power_oracle(
        coefficient, P_FA=0.001, target_pair_limit=2,
        reports_per_receiver=2, full_duplex=False,
        alternating_iterations=4, random_starts=1, seed=11)
    duplex = solve_joint_pair_power_oracle(
        coefficient, P_FA=0.001, target_pair_limit=2,
        reports_per_receiver=2, full_duplex=True,
        alternating_iterations=4, random_starts=1, seed=11)
    assert duplex.worst + 1e-10 >= single.worst
    assert duplex.weak3 + 1e-10 >= single.weak3
