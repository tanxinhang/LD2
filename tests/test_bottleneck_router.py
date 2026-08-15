import math

import pytest

from uav_isac.coordination.bottleneck_router import (
    RepairRoute,
    route_isac_repair,
)


@pytest.mark.parametrize(
    "current,fixed,joint,expected",
    [
        (0.70, 0.90, 0.95, RepairRoute.NO_OP),
        (0.40, 0.80, 0.95, RepairRoute.POWER),
        (0.40, 0.50, 0.80, RepairRoute.STRUCTURE_POWER),
        (0.40, 0.50, 0.55, RepairRoute.GEOMETRY),
    ],
)
def test_router_selects_minimal_sufficient_layer(current, fixed, joint, expected):
    decision = route_isac_repair(
        current,
        fixed_power_upper=fixed,
        joint_structure_upper=joint,
        qos_floor=0.60,
    )
    assert decision.route is expected


def test_router_fails_closed_on_missing_or_nonfinite_ceiling():
    missing = route_isac_repair(
        0.4,
        fixed_power_upper=None,
        joint_structure_upper=0.9,
        qos_floor=0.6,
    )
    invalid = route_isac_repair(
        0.4,
        fixed_power_upper=math.nan,
        joint_structure_upper=0.9,
        qos_floor=0.6,
    )
    assert missing.route is RepairRoute.HOLD_UNVERIFIED
    assert invalid.route is RepairRoute.HOLD_UNVERIFIED


def test_router_subtracts_declared_upper_uncertainty():
    decision = route_isac_repair(
        0.4,
        fixed_power_upper=0.62,
        joint_structure_upper=0.75,
        qos_floor=0.60,
        upper_uncertainty_margin=0.03,
    )
    assert decision.route is RepairRoute.STRUCTURE_POWER


def test_router_rejects_invalid_probability_input():
    with pytest.raises(ValueError, match="probability"):
        route_isac_repair(
            -0.1,
            fixed_power_upper=0.8,
            joint_structure_upper=0.9,
            qos_floor=0.6,
        )
