"""Certificate-routed two-timescale ISAC repair decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class RepairRoute(str, Enum):
    NO_OP = "no_op"
    POWER = "fixed_structure_power"
    STRUCTURE_POWER = "joint_structure_power"
    GEOMETRY = "slow_geometry"
    HOLD_UNVERIFIED = "hold_unverified"


@dataclass(frozen=True)
class BottleneckDecision:
    route: RepairRoute
    reason: str
    current_lower: float
    fixed_power_upper: float | None
    joint_structure_upper: float | None
    qos_floor: float


def route_isac_repair(
    current_lower: float,
    *,
    fixed_power_upper: float | None,
    joint_structure_upper: float | None,
    qos_floor: float,
    upper_uncertainty_margin: float = 0.0,
) -> BottleneckDecision:
    """Route only to the cheapest layer whose certified ceiling can help.

    Upper ceilings must be genuine same-state upper certificates.  Approximate
    solvers can be used only after subtracting a declared uncertainty margin.
    Missing/non-finite ceilings fail closed instead of triggering an expensive
    structural or movement change.
    """
    lower = float(current_lower)
    floor = float(qos_floor)
    margin = float(upper_uncertainty_margin)
    if not np.isfinite(lower) or not np.isfinite(floor):
        raise ValueError("current lower bound and QoS floor must be finite")
    if not (0.0 <= lower <= 1.0 and 0.0 < floor <= 1.0):
        raise ValueError("probability bounds must lie in [0,1]")
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("upper uncertainty margin must be finite and non-negative")

    def usable(value: float | None) -> float | None:
        if value is None:
            return None
        result = float(value)
        if not np.isfinite(result) or not (0.0 <= result <= 1.0):
            return None
        return max(result - margin, 0.0)

    fixed = usable(fixed_power_upper)
    joint = usable(joint_structure_upper)
    common = dict(
        current_lower=lower,
        fixed_power_upper=fixed,
        joint_structure_upper=joint,
        qos_floor=floor,
    )
    if lower >= floor:
        return BottleneckDecision(
            RepairRoute.NO_OP,
            "certified current state already meets the QoS floor",
            **common,
        )
    if fixed is None:
        return BottleneckDecision(
            RepairRoute.HOLD_UNVERIFIED,
            "fixed-structure ceiling is unavailable or invalid",
            **common,
        )
    if fixed >= floor:
        return BottleneckDecision(
            RepairRoute.POWER,
            "fixed structure can meet QoS; avoid a structural switch",
            **common,
        )
    if joint is None:
        return BottleneckDecision(
            RepairRoute.HOLD_UNVERIFIED,
            "power is insufficient but the joint structural ceiling is unverified",
            **common,
        )
    if joint >= floor:
        return BottleneckDecision(
            RepairRoute.STRUCTURE_POWER,
            "fixed structure is insufficient but a same-geometry joint repair can help",
            **common,
        )
    return BottleneckDecision(
        RepairRoute.GEOMETRY,
        "same-geometry power and structure ceilings are below the QoS floor",
        **common,
    )
