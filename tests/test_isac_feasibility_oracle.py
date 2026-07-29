import numpy as np

from uav_isac.physical.feasibility_oracle import (
    reachable_hungarian_geometry,
    solve_maxmin_single_role_pairs,
    solve_pair_only_oracle,
    solve_power_only_oracle,
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


def test_isolated_pair_and_power_oracles_preserve_fixed_variable():
    coefficient = _synthetic_coefficients()
    current_power = np.full((4, 2), 0.25)
    pair_only = solve_pair_only_oracle(
        coefficient,
        current_power,
        P_FA=0.001,
        target_pair_limit=2,
        reports_per_receiver=2,
    )
    np.testing.assert_allclose(pair_only.sensing_power_w, current_power)
    power_only = solve_power_only_oracle(
        coefficient,
        pair_only.selected_set,
        P_FA=0.001,
        total_power_w=1.0,
        communication_reserve_w=0.25,
    )
    assert power_only.selected_set == pair_only.selected_set
    np.testing.assert_array_less(
        power_only.sensing_power_w.sum(axis=1),
        np.full(4, 0.7500001),
    )


def test_local_only_pair_oracle_never_fuses_different_receivers():
    coefficient = _synthetic_coefficients()
    current_power = np.full((4, 2), 0.25)
    solution = solve_pair_only_oracle(
        coefficient,
        current_power,
        P_FA=0.001,
        target_pair_limit=2,
        reports_per_receiver=4,
        fusion_mode="local_only",
    )
    receiver_d = np.zeros((4, 2), dtype=np.float64)
    for i, j, q in solution.selected_set:
        receiver_d[j, q] += coefficient[i, j, q] * current_power[i, q]
    np.testing.assert_allclose(solution.D_q, receiver_d.max(axis=0))
    for q in range(2):
        assert len({j for _, j, target in solution.selected_set
                    if target == q}) <= 1


def test_local_only_power_oracle_uses_receiver_max_not_global_sum():
    coefficient = _synthetic_coefficients()
    selected = (
        (0, 2, 0), (1, 3, 0),
        (0, 2, 1), (1, 3, 1),
    )
    local = solve_power_only_oracle(
        coefficient,
        selected,
        P_FA=0.001,
        total_power_w=1.0,
        communication_reserve_w=0.25,
        fusion_mode="local_only",
    )
    central = solve_power_only_oracle(
        coefficient,
        selected,
        P_FA=0.001,
        total_power_w=1.0,
        communication_reserve_w=0.25,
        fusion_mode="central_oracle",
    )
    receiver_d = np.zeros((4, 2), dtype=np.float64)
    for i, j, q in selected:
        receiver_d[j, q] += coefficient[i, j, q] * local.sensing_power_w[i, q]
    np.testing.assert_allclose(local.D_q, receiver_d.max(axis=0))
    assert central.worst + 1.0e-10 >= local.worst


def test_local_only_joint_oracle_preserves_receiver_boundary():
    coefficient = _synthetic_coefficients()
    solution = solve_joint_pair_power_oracle(
        coefficient,
        P_FA=0.001,
        target_pair_limit=2,
        reports_per_receiver=4,
        full_duplex=False,
        alternating_iterations=2,
        random_starts=0,
        seed=13,
        fusion_mode="local_only",
    )
    receiver_d = np.zeros((4, 2), dtype=np.float64)
    for i, j, q in solution.selected_set:
        receiver_d[j, q] += (
            coefficient[i, j, q] * solution.sensing_power_w[i, q]
        )
    np.testing.assert_allclose(solution.D_q, receiver_d.max(axis=0))


def test_single_milp_maxmin_pairs_enforce_endpoint_roles():
    from uav_isac.utils.types import DeflectionEntry

    entries = []
    coefficient = _synthetic_coefficients()
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            for q in range(2):
                entries.append(DeflectionEntry(
                    i=i, j=j, q=q,
                    tau=0.0, nu=0.0, alpha=1.0,
                    d_raw=coefficient[i, j, q],
                    g_dd=1.0, chi_rep=1.0,
                    d_eff=coefficient[i, j, q],
                ))
    selected, D_q = solve_maxmin_single_role_pairs(
        entries,
        num_uavs=4,
        num_targets=2,
        target_pair_limit=2,
        reports_per_receiver=2,
        p_fa=0.001,
        p_d_floor=0.60,
        target_priority=np.ones(2),
    )
    tx = {i for i, _, _ in selected}
    rx = {j for _, j, _ in selected}
    assert tx.isdisjoint(rx)
    assert np.min(D_q) > 0.0
    for q in range(2):
        assert sum(edge[2] == q for edge in selected) <= 2


def test_single_role_p0_local_fusion_uses_one_receiver_per_target():
    from uav_isac.utils.types import DeflectionEntry

    coefficient = _synthetic_coefficients()
    entries = [
        DeflectionEntry(
            i=i, j=j, q=q,
            tau=0.0, nu=0.0, alpha=1.0,
            d_raw=coefficient[i, j, q],
            g_dd=1.0, chi_rep=1.0,
            d_eff=coefficient[i, j, q],
        )
        for i in range(4)
        for j in range(4)
        if i != j
        for q in range(2)
    ]
    selected, D_q = solve_maxmin_single_role_pairs(
        entries,
        num_uavs=4,
        num_targets=2,
        target_pair_limit=2,
        reports_per_receiver=4,
        p_fa=0.001,
        p_d_floor=0.60,
        fusion_mode="local_only",
    )
    receiver_d = np.zeros((4, 2), dtype=np.float64)
    for i, j, q in selected:
        receiver_d[j, q] += coefficient[i, j, q]
    np.testing.assert_allclose(D_q, receiver_d.max(axis=0))
    for q in range(2):
        assert len({j for _, j, target in selected if target == q}) <= 1
