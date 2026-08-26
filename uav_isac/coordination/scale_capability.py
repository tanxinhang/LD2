"""Cap-aware bistatic scale characterization and shadow-only routing.

The module contains no live controller hook.  It distinguishes constructive
feasibility (a supplied allocation reaches every floor) from certified
infeasibility (even a componentwise relaxed physical ceiling misses a floor).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_isac.coordination.maxmin_power import (
    relaxed_same_geometry_target_ceiling,
)


@dataclass(frozen=True)
class BistaticScaleCapability:
    sensing_budget_w: np.ndarray
    best_single_pair_capability: np.ndarray
    owner_consistent_target_capability: np.ndarray
    relaxed_target_ceiling: np.ndarray
    geometry_bottleneck: float
    owner_consistent_bottleneck: float
    relaxed_worst_ceiling: float
    power_concentration_max: float | None = None
    power_concentration_hhi: float | None = None


@dataclass(frozen=True)
class CapabilityRoute:
    region: str
    action: str
    fixed_gauge: float
    joint_gauge: float | None
    relaxed_gauge: float
    certified: bool
    reason: str


def task_detection_metrics(
    detection_probability: np.ndarray,
    task_floors: tuple[float, float, float, int] | list[float],
) -> dict[str, float]:
    """Return the three task metrics and normalized feasibility ratio.

    The task is feasible iff ``feasibility_ratio <= 1``.  This ratio is not a
    Minkowski/resource gauge because detection probability is nonlinear and
    the ratio is not positively homogeneous.  Unlike a componentwise floor,
    this preserves the distinct worst-target, bottom-k mean, and global mean
    requirements used by the deployed controller.
    """
    pd = np.asarray(detection_probability, dtype=np.float64).reshape(-1)
    if pd.size == 0 or np.any(~np.isfinite(pd)) or np.any((pd < 0.0) | (pd > 1.0)):
        raise ValueError("detection probabilities must be a nonempty vector in [0,1]")
    if len(task_floors) != 4:
        raise ValueError("task_floors must be (rho_min, rho_tail, rho_avg, k)")
    rho_min, rho_tail, rho_avg = (float(value) for value in task_floors[:3])
    k = int(task_floors[3])
    if (
        not 0.0 < rho_min <= 1.0
        or not 0.0 < rho_tail <= 1.0
        or not 0.0 < rho_avg <= 1.0
        or k < 1 or k > pd.size
    ):
        raise ValueError("task floors must be in (0,1] and k within the target count")
    bottom_k = float(np.mean(np.partition(pd, k - 1)[:k]))
    worst = float(np.min(pd))
    average = float(np.mean(pd))
    def ratio(required: float, achieved: float) -> float:
        return required / achieved if achieved > 0.0 else float("inf")

    feasibility_ratio = max(
        ratio(rho_min, worst),
        ratio(rho_tail, bottom_k),
        ratio(rho_avg, average),
    )
    return {
        "worst": worst,
        "bottom_k": bottom_k,
        "average": average,
        "feasibility_ratio": float(feasibility_ratio),
        # Backward-compatible output key for existing shadow artifacts only.
        "gauge": float(feasibility_ratio),
        "worst_slack": worst - rho_min,
        "bottom_k_slack": bottom_k - rho_tail,
        "average_slack": average - rho_avg,
    }


def route_task_capability_shadow(
    fixed_structure_pd: np.ndarray,
    relaxed_ceiling_pd: np.ndarray,
    task_floors: tuple[float, float, float, int] | list[float],
    *,
    joint_feasible_pd: np.ndarray | None = None,
    tolerance: float = 1.0e-12,
) -> CapabilityRoute:
    """Route using the deployed three-floor sensing task without scalarization.

    A joint solution is only a constructive certificate.  Region III requires
    failure of the optimistic componentwise same-geometry ceiling, so a failed
    heuristic search can never by itself trigger geometry repair.
    """
    fixed = np.asarray(fixed_structure_pd, dtype=np.float64).reshape(-1)
    relaxed = np.asarray(relaxed_ceiling_pd, dtype=np.float64).reshape(-1)
    if fixed.shape != relaxed.shape or fixed.size == 0:
        raise ValueError("fixed and relaxed probabilities must be equal nonempty vectors")
    if np.any(fixed > relaxed + tolerance):
        raise ValueError("relaxed ceiling must dominate fixed-structure probability")
    fixed_gauge = task_detection_metrics(
        fixed, task_floors)["feasibility_ratio"]
    relaxed_gauge = task_detection_metrics(
        relaxed, task_floors)["feasibility_ratio"]
    joint_gauge = None
    if joint_feasible_pd is not None:
        joint = np.asarray(joint_feasible_pd, dtype=np.float64).reshape(-1)
        if joint.shape != fixed.shape or np.any(joint > relaxed + tolerance):
            raise ValueError("joint probability must match and not exceed relaxed ceiling")
        joint_gauge = task_detection_metrics(
            joint, task_floors)["feasibility_ratio"]

    if fixed_gauge <= 1.0 + tolerance:
        return CapabilityRoute(
            "I", "hold", fixed_gauge, joint_gauge, relaxed_gauge, True,
            "current fixed structure satisfies worst, bottom-k, and average floors")
    if joint_gauge is not None and joint_gauge <= 1.0 + tolerance:
        return CapabilityRoute(
            "II", "l2_structure_repair", fixed_gauge, joint_gauge,
            relaxed_gauge, True,
            "a same-geometry joint witness satisfies all three task floors")
    if relaxed_gauge > 1.0 + tolerance:
        return CapabilityRoute(
            "III", "l3_geometry_repair", fixed_gauge, joint_gauge,
            relaxed_gauge, True,
            "even the same-geometry componentwise upper ceiling fails a task floor")
    return CapabilityRoute(
        "U", "unresolved_shadow", fixed_gauge, joint_gauge, relaxed_gauge,
        False,
        "same-geometry feasibility is unresolved; heuristic failure is not a proof")


def cap_aware_sensing_budget(
    total_isac_power_w: float | np.ndarray,
    communication_power_w: float | np.ndarray,
    sensing_pa_cap_w: float | np.ndarray,
) -> np.ndarray:
    """Return per-UAV sensing budget under both joint and sensing-PA caps."""
    total, communication, cap = np.broadcast_arrays(
        np.asarray(total_isac_power_w, dtype=np.float64),
        np.asarray(communication_power_w, dtype=np.float64),
        np.asarray(sensing_pa_cap_w, dtype=np.float64),
    )
    if (
        np.any(~np.isfinite(total)) or np.any(total <= 0.0)
        or np.any(~np.isfinite(communication)) or np.any(communication < 0.0)
        or np.any(~np.isfinite(cap)) or np.any(cap <= 0.0)
    ):
        raise ValueError("power limits must be finite with total/cap positive and communication non-negative")
    return np.minimum(np.maximum(total - communication, 0.0), cap)


def characterize_bistatic_scale_capability(
    coefficient_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
    *,
    deployed_target_power_w: np.ndarray | None = None,
) -> BistaticScaleCapability:
    """Compute cap-aware single-pair and coupling-relaxed capabilities.

    ``coefficient_per_watt[i,j,q]`` may already include both path legs, DD
    support and reporting attenuation.  The diagonal is ignored because a
    bistatic pair requires ``i != j``.
    """
    coefficient = np.asarray(coefficient_per_watt, dtype=np.float64).copy()
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    if (
        coefficient.ndim != 3
        or coefficient.shape[0] != coefficient.shape[1]
        or coefficient.shape[0] != budget.size
        or coefficient.shape[2] < 1
    ):
        raise ValueError("coefficient must have shape (K,K,Q) matching budget")
    if np.any(~np.isfinite(coefficient)) or np.any(coefficient < 0.0):
        raise ValueError("coefficient must be finite and non-negative")
    if np.any(~np.isfinite(budget)) or np.any(budget < 0.0):
        raise ValueError("sensing budget must be finite and non-negative")
    diagonal = np.arange(coefficient.shape[0])
    coefficient[diagonal, diagonal, :] = 0.0
    best_receiver = np.max(coefficient, axis=1)
    best_single_pair = np.max(budget[:, None] * best_receiver, axis=0)
    owner_consistent = owner_consistent_bistatic_capability(
        coefficient, budget)
    relaxed = relaxed_same_geometry_target_ceiling(coefficient, budget)

    concentration_max = None
    concentration_hhi = None
    if deployed_target_power_w is not None:
        target_power = np.asarray(
            deployed_target_power_w, dtype=np.float64).reshape(-1)
        if (
            target_power.shape != (coefficient.shape[2],)
            or np.any(~np.isfinite(target_power))
            or np.any(target_power < 0.0)
        ):
            raise ValueError("deployed target power must be a non-negative Q-vector")
        total = float(np.sum(target_power))
        if total > 0.0:
            shares = target_power / total
            concentration_max = float(np.max(shares))
            concentration_hhi = float(np.sum(shares * shares))
    return BistaticScaleCapability(
        sensing_budget_w=budget.copy(),
        best_single_pair_capability=best_single_pair,
        owner_consistent_target_capability=owner_consistent,
        relaxed_target_ceiling=relaxed,
        geometry_bottleneck=float(np.min(best_single_pair)),
        owner_consistent_bottleneck=float(np.min(owner_consistent)),
        relaxed_worst_ceiling=float(np.min(relaxed)),
        power_concentration_max=concentration_max,
        power_concentration_hhi=concentration_hhi,
    )


def owner_consistent_bistatic_capability(
    coefficient_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
) -> np.ndarray:
    """Return ``C_q=max_j sum_{i!=j} b_i a_ijq`` for every target.

    Unlike the fully relaxed ceiling, all transmitters contributing to target
    ``q`` must report to one common receiver owner ``j``.  Cross-target power
    sharing, role exclusivity and receiver capacity remain relaxed, so this is
    a physical difficulty descriptor rather than a deployable allocation.
    """
    coefficient = np.asarray(coefficient_per_watt, dtype=np.float64).copy()
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    if (
        coefficient.ndim != 3
        or coefficient.shape[0] != coefficient.shape[1]
        or coefficient.shape[0] != budget.size
        or coefficient.shape[2] < 1
    ):
        raise ValueError("coefficient must have shape (K,K,Q) matching budget")
    if np.any(~np.isfinite(coefficient)) or np.any(coefficient < 0.0):
        raise ValueError("coefficient must be finite and non-negative")
    if np.any(~np.isfinite(budget)) or np.any(budget < 0.0):
        raise ValueError("sensing budget must be finite and non-negative")
    diagonal = np.arange(coefficient.shape[0])
    coefficient[diagonal, diagonal, :] = 0.0
    owner_capability = np.sum(
        budget[:, None, None] * coefficient, axis=0)
    return np.max(owner_capability, axis=0)


def _gauge(required: np.ndarray, achieved: np.ndarray) -> float:
    ratio = np.divide(
        required,
        achieved,
        out=np.full_like(required, np.inf),
        where=achieved > 0.0,
    )
    ratio[(required == 0.0) & (achieved == 0.0)] = 0.0
    return float(np.max(ratio))


def route_capability_shadow(
    fixed_structure_deflection: np.ndarray,
    relaxed_target_ceiling: np.ndarray,
    required_deflection: np.ndarray,
    *,
    joint_feasible_deflection: np.ndarray | None = None,
    tolerance: float = 1.0e-12,
) -> CapabilityRoute:
    """Classify Hold/L2/L3 only when the corresponding claim is certified.

    A feasible joint allocation is a constructive lower certificate.  Failure
    of a heuristic joint solver is not an impossibility proof.  L3 is certified
    only when the componentwise relaxed upper ceiling misses a task floor.
    """
    fixed = np.asarray(fixed_structure_deflection, dtype=np.float64).reshape(-1)
    relaxed = np.asarray(relaxed_target_ceiling, dtype=np.float64).reshape(-1)
    required = np.asarray(required_deflection, dtype=np.float64).reshape(-1)
    if fixed.size == 0 or fixed.shape != relaxed.shape or fixed.shape != required.shape:
        raise ValueError("fixed, relaxed, and required must be equal nonempty vectors")
    if (
        np.any(~np.isfinite(fixed)) or np.any(fixed < 0.0)
        or np.any(~np.isfinite(relaxed)) or np.any(relaxed < 0.0)
        or np.any(~np.isfinite(required)) or np.any(required < 0.0)
    ):
        raise ValueError("capability vectors must be finite and non-negative")
    if np.any(fixed > relaxed + tolerance):
        raise ValueError("relaxed ceiling must dominate fixed-structure capability")
    fixed_gauge = _gauge(required, fixed)
    relaxed_gauge = _gauge(required, relaxed)
    joint_gauge = None
    if joint_feasible_deflection is not None:
        joint = np.asarray(joint_feasible_deflection, dtype=np.float64).reshape(-1)
        if joint.shape != required.shape or np.any(~np.isfinite(joint)) or np.any(joint < 0.0):
            raise ValueError("joint feasible deflection must be a non-negative Q-vector")
        if np.any(joint > relaxed + tolerance):
            raise ValueError("joint feasible capability cannot exceed relaxed ceiling")
        joint_gauge = _gauge(required, joint)

    if fixed_gauge <= 1.0 + tolerance:
        return CapabilityRoute(
            "I", "hold", fixed_gauge, joint_gauge, relaxed_gauge, True,
            "current fixed structure constructively satisfies every task floor")
    if joint_gauge is not None and joint_gauge <= 1.0 + tolerance:
        return CapabilityRoute(
            "II", "l2_structure_repair", fixed_gauge, joint_gauge,
            relaxed_gauge, True,
            "a same-geometry joint feasible witness satisfies every task floor")
    if relaxed_gauge > 1.0 + tolerance:
        return CapabilityRoute(
            "III", "l3_geometry_repair", fixed_gauge, joint_gauge,
            relaxed_gauge, True,
            "even the coupling-relaxed same-geometry upper ceiling misses a floor")
    return CapabilityRoute(
        "U", "unresolved_shadow", fixed_gauge, joint_gauge, relaxed_gauge,
        False,
        "same-geometry feasibility is unresolved; heuristic failure is not an impossibility proof")
