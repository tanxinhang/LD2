from uav_isac.coordination.permission_cut_master import (
    core_guided_minimum_permission_block,
    CertifiedFeasibility,
    CertifiedMonotonePermissionOracle,
    MonotonePermissionOracle,
    PermissionCutIterationLimit,
    conflict_cut_minimum_permission_block,
    permission_block_with_closed_set,
    objective_layer_common_zeros,
    objective_layer_separating_permission_block,
    shrink_infeasible_permission_core,
)
from uav_isac.coordination.fixing_conflict_filter import PermissionBlock


def test_conflict_cut_master_finds_global_minimum_permission_cardinality():
    def feasible(block):
        return 1 in block.targets and bool({0, 2} & set(block.uavs))

    result = conflict_cut_minimum_permission_block(3, 2, feasible)
    assert result.permission_cardinality == 2
    assert result.block.targets == (1,)
    assert set(result.block.uavs) <= {0, 2}
    assert result.conflict_cuts


def test_forced_backbone_permission_is_preserved():
    result = conflict_cut_minimum_permission_block(
        2, 1, lambda block: 0 in block.targets and 1 in block.role_uavs,
        forced_permissions=("target:0",))
    assert result.block.targets == (0,)
    assert result.block.role_uavs == (1,)
    assert result.block.uavs == (1,)


def test_master_call_cap_returns_auditable_limit_error():
    try:
        conflict_cut_minimum_permission_block(
            3, 2, lambda block: len(block.uavs) + len(block.targets) >= 4,
            max_iterations=1)
    except PermissionCutIterationLimit as error:
        assert error.oracle_calls == 1
        assert len(error.conflict_cuts) == 1
    else:
        raise AssertionError("one-call cap should stop this master")


def test_conflict_shrink_finds_oracle_certified_two_core():
    # A repair is feasible exactly when target 0 or target 1 is released.
    feasible = lambda block: bool({0, 1} & set(block.targets))
    result = shrink_infeasible_permission_core(
        0, 3, ("target:0", "target:1", "target:2"), feasible)
    assert result.core == ("target:0", "target:1")
    assert not feasible(permission_block_with_closed_set(0, 3, result.core))
    for label in result.core:
        reduced = tuple(item for item in result.core if item != label)
        assert feasible(permission_block_with_closed_set(0, 3, reduced))


def test_conflict_shrink_unifies_singleton_backbone_and_three_core():
    singleton = shrink_infeasible_permission_core(
        0, 3, ("target:0", "target:1", "target:2"),
        lambda block: 0 in block.targets)
    assert singleton.core == ("target:0",)

    three = shrink_infeasible_permission_core(
        0, 3, ("target:0", "target:1", "target:2"),
        lambda block: bool(block.targets))
    assert three.core == ("target:0", "target:1", "target:2")


def test_closed_uav_permission_respects_role_dependency():
    block = permission_block_with_closed_set(2, 1, ("uav:1",))
    assert block.uavs == (0,)
    assert block.role_uavs == (0,)


def test_core_guided_master_converges_and_certifies_global_cardinality():
    def feasible(block):
        return 0 in block.targets and (
            1 in block.targets or 0 in block.uavs)

    result = core_guided_minimum_permission_block(
        1, 2, feasible, max_iterations=3)
    assert result.permission_cardinality == 2
    assert 0 in result.block.targets
    assert 1 in result.block.targets or 0 in result.block.uavs
    assert tuple(map(len, result.conflict_cores)) == (2, 1)
    assert result.master_oracle_calls == 3
    assert result.shrink_oracle_calls > 0


def test_quickxplain_returns_irreducible_core_with_fewer_queries():
    feasible = lambda block: bool({0, 1} & set(block.targets))
    sequential = shrink_infeasible_permission_core(
        0, 8, tuple(f"target:{q}" for q in range(8)), feasible)
    quick = shrink_infeasible_permission_core(
        0, 8, tuple(f"target:{q}" for q in range(8)), feasible,
        strategy="quickxplain")
    assert quick.core == ("target:0", "target:1")
    assert quick.oracle_calls < sequential.oracle_calls
    for label in quick.core:
        reduced = tuple(item for item in quick.core if item != label)
        assert feasible(permission_block_with_closed_set(0, 8, reduced))


def test_quickxplain_preserves_uav_role_dependency_semantics():
    # Feasible iff UAV 1 is released; closing UAV 1 semantically closes role 1.
    result = shrink_infeasible_permission_core(
        2, 1,
        ("target:0", "uav:0", "uav:1", "role:0", "role:1"),
        lambda block: 1 in block.uavs,
        strategy="quickxplain")
    assert result.core == ("uav:1",)


def test_monotone_oracle_antichain_inference_is_directionally_exact():
    exact_calls = []

    def exact(block):
        exact_calls.append(block)
        return 0 in block.targets and 0 in block.uavs

    oracle = MonotonePermissionOracle(exact)
    assert oracle(PermissionBlock((0,), (0,), ()))
    assert oracle(PermissionBlock((0, 1), (0, 1), (1,)))  # feasible superset
    assert not oracle(PermissionBlock((), (0,), ()))
    assert not oracle(PermissionBlock((), (), ()))  # infeasible subset
    assert oracle.exact_calls == 2
    assert len(exact_calls) == 2
    assert oracle.inferred_feasible == 1
    assert oracle.inferred_infeasible == 1


def test_support_lex_objective_prioritizes_participant_target_support():
    # Two targets and one UAV+role both have cardinality two.  The exact
    # lexicographic objective prefers one primary support plus one role.
    def feasible(block):
        return ({0, 1} <= set(block.targets)) or (0 in block.role_uavs)

    result = core_guided_minimum_permission_block(
        1, 2, feasible, objective_mode="support_lex")
    assert result.block.uavs == (0,)
    assert result.block.role_uavs == (0,)
    assert result.permission_objective == (1, 1)


def test_certified_monotone_oracle_never_caches_unresolved_as_conflict():
    answers = iter((CertifiedFeasibility.UNRESOLVED,
                    CertifiedFeasibility.CERT_INFEASIBLE))
    oracle = CertifiedMonotonePermissionOracle(lambda _block: next(answers))
    empty = PermissionBlock((), (), ())
    assert oracle(empty) is CertifiedFeasibility.UNRESOLVED
    assert oracle(empty) is CertifiedFeasibility.CERT_INFEASIBLE
    assert oracle.exact_calls == 2
    assert oracle.unresolved == 1


def test_objective_layer_common_zero_cut_strictly_raises_support_lex_bound():
    first = objective_layer_common_zeros(
        1, 2, (("uav:0", "role:0"),), objective_mode="support_lex")
    assert first.permission_objective == (1, 0)
    assert set(first.common_zero_permissions) == {
        "target:0", "target:1", "role:0"}
    second = objective_layer_common_zeros(
        1, 2,
        (("uav:0", "role:0"), first.common_zero_permissions),
        objective_mode="support_lex")
    assert second.scalar_cost > first.scalar_cost


def test_objective_sublevel_common_zeros_shrink_monotonically_with_cost():
    cores = (("uav:0", "role:0"),)
    level2 = objective_layer_common_zeros(
        1, 2, cores, objective_mode="support_lex", scalar_cost_upper=2)
    level3 = objective_layer_common_zeros(
        1, 2, cores, objective_mode="support_lex", scalar_cost_upper=3)
    level4 = objective_layer_common_zeros(
        1, 2, cores, objective_mode="support_lex", scalar_cost_upper=4)
    assert set(level3.common_zero_permissions) < set(level2.common_zero_permissions)
    assert set(level4.common_zero_permissions) < set(level3.common_zero_permissions)


def test_objective_layer_separation_finds_global_support_lex_repair():
    def oracle(block):
        feasible = 0 in block.targets and (
            1 in block.targets or 0 in block.uavs)
        return (CertifiedFeasibility.CERT_FEASIBLE if feasible else
                CertifiedFeasibility.CERT_INFEASIBLE)

    result = objective_layer_separating_permission_block(1, 2, oracle)
    assert result.permission_objective == (2, 0)
    assert result.lower_bound_trace[-1] > result.lower_bound_trace[0]
    assert 0 in result.block.targets


def test_objective_layer_separation_stops_on_unresolved_without_cut():
    try:
        objective_layer_separating_permission_block(
            1, 1, lambda _block: CertifiedFeasibility.UNRESOLVED)
    except RuntimeError as error:
        assert "unresolved" in str(error)
    else:
        raise AssertionError("unresolved oracle state must stop separation")
