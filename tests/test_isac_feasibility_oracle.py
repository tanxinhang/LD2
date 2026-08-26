import numpy as np
from types import SimpleNamespace

from uav_isac.utils.types import DeflectionEntry
import uav_isac.physical.feasibility_oracle as feasibility_oracle
from uav_isac.coordination.maxmin_power import (
    fixed_owner_gain_matrix,
    solve_fixed_structure_maxmin_power_lp,
)
from uav_isac.physical.feasibility_oracle import (
    repair_invalid_local_only_edges_min_change,
    reachable_hungarian_geometry,
    solve_budget_coupled_local_only_pairs,
    solve_enumerated_role_ceiling_local_pairs,
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


def test_power_oracles_respect_separate_sensing_pa_cap():
    coefficient = _synthetic_coefficients()
    joint = solve_joint_pair_power_oracle(
        coefficient,
        P_FA=0.001,
        total_power_w=1.0,
        sensing_power_cap_w=0.0251,
        target_pair_limit=2,
        reports_per_receiver=2,
        alternating_iterations=2,
        random_starts=0,
    )
    np.testing.assert_array_less(
        joint.sensing_power_w.sum(axis=1), np.full(4, 0.0251001))
    fixed = solve_power_only_oracle(
        coefficient,
        joint.selected_set,
        P_FA=0.001,
        total_power_w=1.0,
        sensing_power_cap_w=0.0251,
    )
    np.testing.assert_array_less(
        fixed.sensing_power_w.sum(axis=1), np.full(4, 0.0251001))


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


def test_budget_coupled_local_structure_respects_power_and_roles():
    coefficient = _synthetic_coefficients()
    budget = np.array([0.02, 0.03, 0.04, 0.05])
    solution = solve_budget_coupled_local_only_pairs(
        coefficient,
        budget,
        P_FA=0.001,
        p_d_floor=0.60,
        target_pair_limit=2,
        reports_per_receiver=4,
        target_priority=np.array([3.0, 1.0]),
    )
    assert solution.selected_set
    assert set(solution.tx_indices).isdisjoint(solution.rx_indices)
    assert np.all(np.sum(solution.sensing_power_w, axis=1) <= budget + 1e-10)
    for q in range(2):
        owners = {j for _i, j, target in solution.selected_set if target == q}
        assert len(owners) == 1
        assert sum(target == q for _i, _j, target in solution.selected_set) <= 2

    reconstructed = np.zeros(2, dtype=np.float64)
    for i, j, q in solution.selected_set:
        reconstructed[q] += (
            coefficient[i, j, q] * solution.sensing_power_w[i, q])
    np.testing.assert_allclose(solution.D_q, reconstructed, atol=1e-10)
    fixed_gain, _ = fixed_owner_gain_matrix(
        coefficient, solution.selected_set)
    downstream = solve_fixed_structure_maxmin_power_lp(fixed_gain, budget)
    assert downstream.worst_deflection + 1e-8 >= float(np.min(solution.D_q))


def test_budget_coupled_live_entrypoint_uses_more_than_one_tx_when_needed():
    # One transmitter cannot reach D=1 on both targets under its shared
    # budget, while two transmitters can.  The legacy unit-power proxy misses
    # this coupling because it gives one transmitter a full unit per target.
    coefficient = np.zeros((4, 4, 2), dtype=np.float64)
    coefficient[0, 3, :] = 10.0
    coefficient[1, 3, 0] = 9.0
    coefficient[2, 3, 1] = 9.0
    entries = [
        DeflectionEntry(
            i=i, j=j, q=q, tau=0.0, nu=0.0, alpha=1.0,
            d_raw=coefficient[i, j, q], g_dd=1.0, chi_rep=1.0,
            d_eff=coefficient[i, j, q],
        )
        for i in range(4) for j in range(4) if i != j
        for q in range(2) if coefficient[i, j, q] > 0.0
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
        per_uav_sensing_budget_w=np.ones(4),
    )
    assert len({i for i, _j, _q in selected}) >= 2
    assert np.min(D_q) >= 9.0 - 1e-6


def test_budget_coupled_accepts_verified_time_limit_incumbent(monkeypatch):
    real_milp = feasibility_oracle.milp

    def time_limited_with_incumbent(*args, **kwargs):
        solved = real_milp(*args, **kwargs)
        assert solved.x is not None
        return SimpleNamespace(
            success=False,
            x=np.asarray(solved.x, dtype=np.float64),
            message="test time limit",
        )

    monkeypatch.setattr(
        feasibility_oracle, "milp", time_limited_with_incumbent)
    solution = solve_budget_coupled_local_only_pairs(
        _synthetic_coefficients(),
        np.array([0.02, 0.03, 0.04, 0.05]),
        P_FA=0.001,
        p_d_floor=0.60,
        target_pair_limit=2,
        reports_per_receiver=4,
        secondary_objective_enabled=False,
    )
    assert solution.selected_set
    assert set(solution.tx_indices).isdisjoint(solution.rx_indices)


def test_enumerated_role_ceiling_returns_lp_certified_feasible_support():
    coefficient = _synthetic_coefficients()
    budget = np.array([0.02, 0.03, 0.04, 0.05])
    solution = solve_enumerated_role_ceiling_local_pairs(
        coefficient,
        budget,
        P_FA=0.001,
        p_d_floor=0.60,
        target_pair_limit=2,
        reports_per_receiver=4,
        target_priority=np.array([3.0, 1.0]),
    )
    assert solution.selected_set
    assert set(solution.tx_indices).isdisjoint(solution.rx_indices)
    assert np.all(np.sum(solution.sensing_power_w, axis=1) <= budget + 1e-10)
    fixed_gain, _ = fixed_owner_gain_matrix(
        coefficient, solution.selected_set)
    downstream = solve_fixed_structure_maxmin_power_lp(fixed_gain, budget)
    np.testing.assert_allclose(solution.D_q, downstream.deflection)
    for q in range(2):
        target_edges = [edge for edge in solution.selected_set if edge[2] == q]
        assert 1 <= len(target_edges) <= 2
        assert len({j for _i, j, _q in target_edges}) == 1


def test_satisficing_portfolio_keeps_feasible_single_tx_and_repairs_failure():
    coefficient = np.zeros((4, 4, 2), dtype=np.float64)
    coefficient[0, 3, :] = 100.0
    entries = [
        DeflectionEntry(
            i=i, j=j, q=q, tau=0.0, nu=0.0, alpha=1.0,
            d_raw=coefficient[i, j, q], g_dd=1.0, chi_rep=1.0,
            d_eff=coefficient[i, j, q],
        )
        for i in range(4) for j in range(4) if i != j
        for q in range(2) if coefficient[i, j, q] > 0.0
    ]
    selected, _ = solve_maxmin_single_role_pairs(
        entries,
        num_uavs=4,
        num_targets=2,
        target_pair_limit=2,
        reports_per_receiver=4,
        p_fa=0.001,
        p_d_floor=0.60,
        fusion_mode="local_only",
        per_uav_sensing_budget_w=np.ones(4),
        joint_solver_mode="satisficing_legacy_then_enumerated",
        task_qos_floors=(0.60, 0.70, 0.80, 2),
    )
    assert len({i for i, _j, _q in selected}) == 1

    coefficient[0, 3, :] = 10.0
    coefficient[1, 3, 0] = 100.0
    coefficient[2, 3, 1] = 100.0
    entries = [
        DeflectionEntry(
            i=i, j=j, q=q, tau=0.0, nu=0.0, alpha=1.0,
            d_raw=coefficient[i, j, q], g_dd=1.0, chi_rep=1.0,
            d_eff=coefficient[i, j, q],
        )
        for i in range(4) for j in range(4) if i != j
        for q in range(2) if coefficient[i, j, q] > 0.0
    ]
    repaired, _ = solve_maxmin_single_role_pairs(
        entries,
        num_uavs=4,
        num_targets=2,
        target_pair_limit=2,
        reports_per_receiver=4,
        p_fa=0.001,
        p_d_floor=0.60,
        fusion_mode="local_only",
        per_uav_sensing_budget_w=np.ones(4),
        joint_solver_mode="satisficing_legacy_then_enumerated",
        task_qos_floors=(0.90, 0.90, 0.90, 2),
    )
    assert len({i for i, _j, _q in repaired}) >= 2


def test_min_change_topology_repair_preserves_roles_owners_and_capacity():
    coefficient = np.zeros((4, 4, 2), dtype=np.float64)
    # Cached edge (0,3,0) has left the physical support.  UAV 2 already has
    # the transmitter role and can replace it without changing target owner.
    coefficient[1, 3, 0] = 8.0
    coefficient[2, 3, 0] = 7.0
    coefficient[2, 3, 1] = 9.0
    selected = ((0, 3, 0), (1, 3, 0), (2, 3, 1))
    budget = np.full(4, 0.1)
    repaired = repair_invalid_local_only_edges_min_change(
        coefficient,
        selected,
        budget,
        P_FA=0.001,
        target_pair_limit=2,
        reports_per_receiver=3,
    )
    assert repaired is not None
    assert repaired.mode == "topology_min_change_local_only"
    assert len(repaired.selected_set) == len(selected)
    assert (0, 3, 0) not in repaired.selected_set
    assert (2, 3, 0) in repaired.selected_set
    assert repaired.tx_indices == (0, 1, 2)
    assert repaired.rx_indices == (3,)
    assert {j for _i, j, _q in repaired.selected_set} == {3}
    assert all(
        sum(edge[2] == q for edge in repaired.selected_set) <= 2
        for q in range(2)
    )
    fixed_gain, _ = fixed_owner_gain_matrix(
        coefficient, repaired.selected_set)
    exact = solve_fixed_structure_maxmin_power_lp(fixed_gain, budget)
    np.testing.assert_allclose(repaired.D_q, exact.deflection)
    assert np.all(np.sum(repaired.sensing_power_w, axis=1) <= budget + 1e-10)


def test_min_change_topology_repair_falls_back_without_role_preserving_edge():
    coefficient = np.zeros((3, 3, 2), dtype=np.float64)
    coefficient[0, 2, 1] = 5.0
    selected = ((0, 2, 0), (0, 2, 1))
    repaired = repair_invalid_local_only_edges_min_change(
        coefficient,
        selected,
        np.full(3, 0.1),
        P_FA=0.001,
        target_pair_limit=1,
        reports_per_receiver=2,
    )
    assert repaired is None
