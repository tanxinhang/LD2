"""Deterministic target-wise decisions from calibrated physical intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhysicsIntervalDecision:
    accept: bool
    target_safe: np.ndarray
    worst_delta_lower: float


def certified_physics_interval_decision(
    *,
    candidate_lower: np.ndarray,
    noop_lower: np.ndarray,
    noop_upper: np.ndarray,
    qos_floor: float,
    charged_cost: float = 0.0,
) -> PhysicsIntervalDecision:
    """Accept only target-preserving positive gain on one joint interval.

    The caller must construct simultaneous lower/upper probability bounds.
    Monotonicity of ``min`` gives
    ``min(P_candidate)-min(P_noop) >= min(L_candidate)-min(U_noop)``.
    No independence between targets or edges is assumed.
    """
    candidate = np.asarray(candidate_lower, dtype=np.float64).reshape(-1)
    noop_l = np.asarray(noop_lower, dtype=np.float64).reshape(-1)
    noop_u = np.asarray(noop_upper, dtype=np.float64).reshape(-1)
    floor = float(qos_floor)
    cost = float(charged_cost)
    if (
        candidate.size < 1 or candidate.shape != noop_l.shape
        or candidate.shape != noop_u.shape
        or np.any(~np.isfinite(candidate))
        or np.any(~np.isfinite(noop_l)) or np.any(~np.isfinite(noop_u))
        or np.any((candidate < 0.0) | (candidate > 1.0))
        or np.any((noop_l < 0.0) | (noop_l > 1.0))
        or np.any((noop_u < 0.0) | (noop_u > 1.0))
        or np.any(noop_l > noop_u + 1.0e-12)
        or not np.isfinite(floor) or not 0.0 < floor <= 1.0
        or not np.isfinite(cost) or cost < 0.0
    ):
        raise ValueError("physical probability intervals are invalid")
    target_safe = candidate + 1.0e-12 >= np.minimum(noop_l, floor)
    worst_delta_lower = float(np.min(candidate) - np.min(noop_u))
    return PhysicsIntervalDecision(
        accept=bool(np.all(target_safe) and worst_delta_lower > cost),
        target_safe=target_safe,
        worst_delta_lower=worst_delta_lower,
    )
