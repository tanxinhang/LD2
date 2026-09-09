"""Domain-level risk-budgeted safety gates for ISAC reconfiguration.

The learned policy or local-search layer may rank candidates, but it is not
allowed to commit one.  This module consumes already calibrated lower/upper
detection envelopes and applies three independent safeguards:

* target-wise no-harm must hold at every predicted transition step;
* the discounted conservative worst-target gain must pay the declared
  objective cost; and
* data-dependent routing between feedback and physics certificates, together
  with link, runtime and energy resource certificates, is charged to one
  explicit episode-level union risk budget.

The module intentionally does not create confidence intervals.  Each route's
``miscoverage`` must describe the frozen calibration procedure that produced
its simultaneous horizon-by-target envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CertificateRiskBudget:
    """Episode-level Bonferroni budget for routed safety certificates."""

    total: float
    feedback: float = 0.0
    physics: float = 0.0
    link: float = 0.0
    runtime: float = 0.0
    energy: float = 0.0

    def __post_init__(self) -> None:
        values = np.asarray([
            self.total, self.feedback, self.physics, self.link,
            self.runtime, self.energy,
        ], dtype=np.float64)
        if np.any(~np.isfinite(values)):
            raise ValueError("risk budgets must be finite")
        if not 0.0 < float(self.total) < 1.0:
            raise ValueError("total miscoverage must lie in (0,1)")
        if np.any(values[1:] < 0.0) or np.any(values[1:] >= 1.0):
            raise ValueError("component miscoverage must lie in [0,1)")
        if self.allocated > float(self.total) + 1.0e-15:
            raise ValueError(
                "feedback + physics + link + runtime + energy risk exceeds "
                "the total budget")

    @property
    def allocated(self) -> float:
        return float(
            self.feedback + self.physics + self.link
            + self.runtime + self.energy)

    def allocation_for(self, source: str) -> float:
        normalized = str(source).strip().lower()
        if normalized not in {
            "feedback", "physics", "link", "runtime", "energy",
        }:
            raise ValueError(f"unsupported certificate source: {source}")
        return float(getattr(self, normalized))


@dataclass(frozen=True)
class ResourceEpochRiskLedger:
    """Validated resource-epoch contribution to the system risk budget.

    ``complete`` is deliberately stricter than merely having a valid total
    budget: all three resource epochs must be present and each must have a
    positive allocation large enough for its own frozen miscoverage.  The
    coverage floor is the system Bonferroni floor and therefore does not rely
    on independence between communication, computation and metrology errors.
    """

    link_epoch_miscoverage: float | None
    runtime_epoch_miscoverage: float | None
    energy_epoch_miscoverage: float | None
    link_allocation: float
    runtime_allocation: float
    energy_allocation: float
    allocated_resource_miscoverage: float
    system_total_miscoverage: float
    joint_coverage_floor_union_bound: float
    complete: bool


def validate_resource_epoch_risk_budget(
    risk_budget: CertificateRiskBudget,
    *,
    link_miscoverage: float | None,
    runtime_miscoverage: float | None,
    energy_miscoverage: float | None,
) -> ResourceEpochRiskLedger:
    """Bind frozen resource epochs to their pre-allocated system risks."""

    epoch_values = {
        "link": link_miscoverage,
        "runtime": runtime_miscoverage,
        "energy": energy_miscoverage,
    }
    normalized: dict[str, float | None] = {}
    for source, raw_value in epoch_values.items():
        if raw_value is None:
            normalized[source] = None
            continue
        value = float(raw_value)
        if not np.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(
                f"{source} epoch miscoverage must lie in (0,1)")
        allocation = risk_budget.allocation_for(source)
        if allocation <= 0.0:
            raise ValueError(
                f"{source} epoch has no system risk allocation")
        if value > allocation + 1.0e-15:
            raise ValueError(
                f"{source} epoch miscoverage exceeds its system risk "
                "allocation")
        normalized[source] = value

    allocations = {
        source: risk_budget.allocation_for(source)
        for source in ("link", "runtime", "energy")
    }
    complete = bool(
        all(normalized[source] is not None for source in normalized)
        and all(value > 0.0 for value in allocations.values())
    )
    return ResourceEpochRiskLedger(
        link_epoch_miscoverage=normalized["link"],
        runtime_epoch_miscoverage=normalized["runtime"],
        energy_epoch_miscoverage=normalized["energy"],
        link_allocation=float(allocations["link"]),
        runtime_allocation=float(allocations["runtime"]),
        energy_allocation=float(allocations["energy"]),
        allocated_resource_miscoverage=float(sum(allocations.values())),
        system_total_miscoverage=float(risk_budget.total),
        joint_coverage_floor_union_bound=float(1.0 - risk_budget.total),
        complete=complete,
    )


@dataclass(frozen=True)
class RiskBudgetedRouteDecision:
    accept: bool
    accepted_sources: tuple[str, ...]
    allocated_miscoverage: float
    total_miscoverage: float
    hard_feasible: bool


def risk_budgeted_certificate_union(
    branch_acceptance: Mapping[str, bool],
    *,
    risk_budget: CertificateRiskBudget,
    hard_feasible: bool,
) -> RiskBudgetedRouteDecision:
    """Apply a Bonferroni-safe OR over calibrated certificate routes.

    The allocation is charged for every configured branch, not only the branch
    that happens to accept.  That detail is required because choosing a route
    after inspecting both certificates is itself data dependent.
    """
    normalized = {
        str(name).strip().lower(): bool(value)
        for name, value in branch_acceptance.items()
    }
    unsupported = set(normalized) - {"feedback", "physics"}
    if unsupported:
        raise ValueError(
            f"unsupported certificate sources: {sorted(unsupported)}")
    for source, accepted in normalized.items():
        if accepted and risk_budget.allocation_for(source) <= 0.0:
            raise ValueError(
                f"accepted {source} route has no allocated risk budget")
    accepted_sources = tuple(sorted(
        source for source, accepted in normalized.items() if accepted
    ))
    feasible = bool(hard_feasible)
    return RiskBudgetedRouteDecision(
        accept=bool(feasible and accepted_sources),
        accepted_sources=accepted_sources if feasible else tuple(),
        allocated_miscoverage=float(risk_budget.allocated),
        total_miscoverage=float(risk_budget.total),
        hard_feasible=feasible,
    )


@dataclass(frozen=True)
class HorizonTransitionBounds:
    """One simultaneous route envelope over ``(horizon, target)``."""

    source: str
    candidate_lower: np.ndarray
    noop_lower: np.ndarray
    noop_upper: np.ndarray
    miscoverage: float

    def __post_init__(self) -> None:
        source = str(self.source).strip().lower()
        if source not in {"feedback", "physics"}:
            raise ValueError("horizon source must be feedback or physics")
        candidate = np.asarray(self.candidate_lower, dtype=np.float64)
        noop_l = np.asarray(self.noop_lower, dtype=np.float64)
        noop_u = np.asarray(self.noop_upper, dtype=np.float64)
        if (
            candidate.ndim != 2 or candidate.shape[0] < 1
            or candidate.shape[1] < 1
            or candidate.shape != noop_l.shape
            or candidate.shape != noop_u.shape
        ):
            raise ValueError(
                "horizon bounds must share a non-empty (H,Q) shape")
        values = np.concatenate((
            candidate.reshape(-1), noop_l.reshape(-1), noop_u.reshape(-1),
        ))
        if np.any(~np.isfinite(values)) or np.any(
            (values < 0.0) | (values > 1.0)
        ):
            raise ValueError("horizon probability bounds must lie in [0,1]")
        if np.any(noop_l > noop_u + 1.0e-12):
            raise ValueError("No-op lower bound cannot exceed its upper bound")
        alpha = float(self.miscoverage)
        if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("route miscoverage must lie in (0,1)")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "candidate_lower", candidate)
        object.__setattr__(self, "noop_lower", noop_l)
        object.__setattr__(self, "noop_upper", noop_u)
        object.__setattr__(self, "miscoverage", alpha)


@dataclass(frozen=True)
class HorizonRouteDecision:
    source: str
    accept: bool
    target_safe: np.ndarray
    worst_delta_lower: np.ndarray
    discounted_gain_lower: float
    charged_objective_cost: float
    hard_feasible: bool
    certificate_mode: str = "independent_probability_extrema"
    gain_unit: str = "detection_probability"


def certify_horizon_transition(
    bounds: HorizonTransitionBounds,
    *,
    qos_floor: float,
    discount: float = 1.0,
    objective_cost: float = 0.0,
    structural_feasible: bool = True,
    transport_feasible: bool = True,
) -> HorizonRouteDecision:
    """Certify target preservation and positive conservative horizon value."""
    floor = float(qos_floor)
    gamma = float(discount)
    cost = float(objective_cost)
    if not 0.0 < floor <= 1.0:
        raise ValueError("qos_floor must lie in (0,1]")
    if not np.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("discount must lie in (0,1]")
    if not np.isfinite(cost) or cost < 0.0:
        raise ValueError("objective_cost must be finite and non-negative")
    candidate = bounds.candidate_lower
    noop_l = bounds.noop_lower
    noop_u = bounds.noop_upper
    # The No-op upper envelope is the conservative target-wise comparator.
    # Comparing only with its lower envelope would establish service-floor
    # preservation, but could still hide a target degradation.
    no_harm = candidate + 1.0e-12 >= noop_u
    service_safe = candidate + 1.0e-12 >= np.minimum(noop_l, floor)
    target_safe = no_harm & service_safe
    worst_delta = np.min(candidate, axis=1) - np.min(noop_u, axis=1)
    weights = gamma ** np.arange(candidate.shape[0], dtype=np.float64)
    discounted_gain = float(np.dot(weights, worst_delta))
    hard_feasible = bool(structural_feasible and transport_feasible)
    accept = bool(
        hard_feasible
        and np.all(target_safe)
        and discounted_gain > cost
    )
    return HorizonRouteDecision(
        source=bounds.source,
        accept=accept,
        target_safe=target_safe,
        worst_delta_lower=worst_delta,
        discounted_gain_lower=discounted_gain,
        charged_objective_cost=cost,
        hard_feasible=hard_feasible,
    )


def certify_horizon_paired_deflection_transition(
    bounds: HorizonTransitionBounds,
    paired_deflection_lower: np.ndarray,
    *,
    qos_floor: float,
    discount: float = 1.0,
    objective_cost_deflection: float = 0.0,
    structural_feasible: bool = True,
    transport_feasible: bool = True,
) -> HorizonRouteDecision:
    """Certify a Pareto improvement under one common coefficient tensor.

    If candidate and No-op experience the same unknown coefficient tensor
    ``a`` in the calibrated box, their target-q Deflection difference is
    linear: ``Delta_q(a)=sum_e w_e,q a_e,q``.  Its exact box minimum is passed
    as ``paired_deflection_lower``.  Non-negativity therefore implies
    candidate ``P_D`` no-harm by monotonicity of the Gaussian-deflection
    detector, without combining two mutually incompatible interval extrema.

    The objective is the discounted sum of these target-wise lower gains.
    Because every component must first be non-negative, a positive total is
    a strict Pareto improvement.  Its unit is Deflection, so probability-unit
    prices must not be supplied to this route.
    """
    paired = np.asarray(paired_deflection_lower, dtype=np.float64)
    if paired.shape != bounds.candidate_lower.shape:
        raise ValueError("paired Deflection lower bound must have shape (H,Q)")
    if np.any(~np.isfinite(paired)):
        raise ValueError("paired Deflection lower bound must be finite")
    floor = float(qos_floor)
    gamma = float(discount)
    cost = float(objective_cost_deflection)
    if not 0.0 < floor <= 1.0:
        raise ValueError("qos_floor must lie in (0,1]")
    if not np.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("discount must lie in (0,1]")
    if not np.isfinite(cost) or cost < 0.0:
        raise ValueError(
            "objective_cost_deflection must be finite non-negative")

    no_harm = paired >= -1.0e-12
    service_safe = (
        bounds.candidate_lower + 1.0e-12
        >= np.minimum(bounds.noop_lower, floor)
    )
    target_safe = no_harm & service_safe
    per_step_total = np.sum(np.maximum(paired, 0.0), axis=1)
    weights = gamma ** np.arange(paired.shape[0], dtype=np.float64)
    discounted_gain = float(np.dot(weights, per_step_total))
    hard_feasible = bool(structural_feasible and transport_feasible)
    accept = bool(
        hard_feasible
        and np.all(target_safe)
        and discounted_gain > cost + 1.0e-15
    )
    return HorizonRouteDecision(
        source=bounds.source,
        accept=accept,
        target_safe=target_safe,
        worst_delta_lower=np.min(paired, axis=1),
        discounted_gain_lower=discounted_gain,
        charged_objective_cost=cost,
        hard_feasible=hard_feasible,
        certificate_mode="common_box_paired_deflection",
        gain_unit="deflection",
    )


@dataclass(frozen=True)
class HorizonCertificateDecision:
    accept: bool
    mode: str
    accepted_sources: tuple[str, ...]
    route_decisions: tuple[HorizonRouteDecision, ...]
    allocated_miscoverage: float
    total_miscoverage: float


def certify_risk_budgeted_horizon(
    routes: Sequence[HorizonTransitionBounds],
    *,
    risk_budget: CertificateRiskBudget,
    qos_floor: float,
    discount: float = 1.0,
    objective_cost: float = 0.0,
    structural_feasible: bool = True,
    transport_feasible: bool = True,
    mode: str = "union",
) -> HorizonCertificateDecision:
    """Certify finite-horizon reconfiguration with an explicit route budget."""
    items = tuple(routes)
    if not items:
        raise ValueError("at least one horizon certificate route is required")
    names = [item.source for item in items]
    if len(names) != len(set(names)):
        raise ValueError("certificate route sources must be unique")
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"union", "intersection"}:
        raise ValueError("mode must be union or intersection")
    for item in items:
        allocation = risk_budget.allocation_for(item.source)
        if allocation <= 0.0:
            raise ValueError(
                f"route {item.source} has no allocated risk budget")
        if item.miscoverage > allocation + 1.0e-15:
            raise ValueError(
                f"route {item.source} miscoverage exceeds its allocation")
    decisions = tuple(
        certify_horizon_transition(
            item,
            qos_floor=qos_floor,
            discount=discount,
            objective_cost=objective_cost,
            structural_feasible=structural_feasible,
            transport_feasible=transport_feasible,
        )
        for item in items
    )
    accepted = tuple(sorted(
        item.source for item in decisions if item.accept
    ))
    if normalized_mode == "union":
        final_accept = bool(accepted)
    else:
        final_accept = bool(len(accepted) == len(decisions))
    return HorizonCertificateDecision(
        accept=final_accept,
        mode=normalized_mode,
        accepted_sources=accepted if final_accept else tuple(),
        route_decisions=decisions,
        allocated_miscoverage=float(risk_budget.allocated),
        total_miscoverage=float(risk_budget.total),
    )
