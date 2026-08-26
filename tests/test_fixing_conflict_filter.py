from uav_isac.coordination.fixing_conflict_filter import (
    PermissionBlock,
    exact_full_block_repair_backbone,
    irreducible_feasible_permission_block,
)


def test_deletion_filter_returns_irreducible_monotone_permission_block():
    # Feasible iff target 1 and either UAV 0 or UAV 2 are released.
    def feasible(block):
        return 1 in block.targets and bool({0, 2} & set(block.uavs))

    result = irreducible_feasible_permission_block(
        PermissionBlock((0, 1, 2), (0, 1), (0, 1, 2)), feasible)
    assert result.block.targets == (1,)
    assert len(result.block.uavs) == 1
    assert set(result.block.uavs) <= {0, 2}
    assert result.feasibility_calls == 7
    assert result.retained_certificates


def test_deletion_filter_rejects_infeasible_full_block():
    try:
        irreducible_feasible_permission_block(
            PermissionBlock((0,), (0,), (0,)), lambda _block: False)
    except ValueError as error:
        assert "must be feasible" in str(error)
    else:
        raise AssertionError("infeasible full block must be rejected")


def test_bisect_filter_preserves_irreducibility_with_fewer_batch_calls():
    def feasible(block):
        return 1 in block.targets and 0 in block.uavs

    result = irreducible_feasible_permission_block(
        PermissionBlock((0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3)),
        feasible, strategy="bisect")
    assert result.block.targets == (1,)
    assert result.block.uavs == (0,)
    assert result.block.role_uavs == ()
    assert set(result.retained_certificates) == {"target:1", "uav:0"}


def test_full_block_backbone_is_order_invariant_global_necessity():
    def feasible(block):
        return 1 in block.targets and bool({0, 2} & set(block.uavs))

    result = exact_full_block_repair_backbone(
        PermissionBlock((0, 1, 2), (0, 1), (0, 1, 2)), feasible)
    assert result.backbone == ("target:1",)
    assert result.feasibility_calls == 1 + 2 + 3 + 3
