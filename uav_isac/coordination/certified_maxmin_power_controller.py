"""Fail-closed finite-round max-min sensing-power controller.

The learned policy retains motion and reporting-structure decisions. Given a
fixed unique-owner structure, this controller reserves the physical U2U wire
budget, solves the remaining linear max-min power problem with finite-round
column generation, and commits only a certified QoS-safe improvement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from uav_isac.coordination.maxmin_power import (
    MaxMinPowerResult,
    distributed_column_generation_maxmin_power,
    fixed_owner_gain_matrix,
)
from uav_isac.coordination.power_repair_transport import (
    PowerRepairTransportCertificate,
    PowerRepairWireLayout,
    certify_power_repair_transport,
)
from uav_isac.domain.communication import CommunicationTransport
from uav_isac.physical.detection import (
    compute_detection_probabilities,
    minimum_deflection_for_detection_probability,
)


@dataclass(frozen=True)
class CertifiedMaxMinPowerConfig:
    rounds: int = 4
    price_bits: int = 6
    feedback_bits: int = 16
    snr_margin_db: float = 3.0
    latency_margin_s: float = 5.0e-4
    qos_floor: float = 0.60
    minimum_worst_pd_improvement: float = 0.0
    probability_tolerance: float = 1.0e-9

    def __post_init__(self) -> None:
        if int(self.rounds) < 1:
            raise ValueError("rounds must be positive")
        if not 0 <= int(self.price_bits) <= 24:
            raise ValueError("price_bits must lie in [0,24]")
        if int(self.feedback_bits) not in (16, 32, 64):
            raise ValueError("feedback_bits must be one of {16,32,64}")
        if not 0.0 <= float(self.qos_floor) < 1.0:
            raise ValueError("qos_floor must lie in [0,1)")
        if float(self.minimum_worst_pd_improvement) < 0.0:
            raise ValueError("minimum improvement must be non-negative")
        if float(self.probability_tolerance) < 0.0:
            raise ValueError("probability tolerance must be non-negative")


@dataclass(frozen=True)
class CertifiedMaxMinPowerDecision:
    accepted: bool
    reason: str
    sensing_power_w: np.ndarray
    communication_power_w: np.ndarray
    candidate_sensing_power_w: np.ndarray
    candidate_communication_power_w: np.ndarray
    noop_lower_pd: np.ndarray
    noop_upper_pd: np.ndarray
    candidate_lower_pd: np.ndarray
    candidate_upper_pd: np.ndarray
    protected_pd: np.ndarray
    optimizer: MaxMinPowerResult | None
    transport: PowerRepairTransportCertificate

    @property
    def certified_worst_pd_improvement(self) -> float:
        return float(
            np.min(self.candidate_lower_pd) - np.min(self.noop_upper_pd))


def _validate_coefficient_envelope(
    lower_coefficient: np.ndarray,
    upper_coefficient: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(lower_coefficient, dtype=np.float64)
    upper = np.asarray(upper_coefficient, dtype=np.float64)
    if (
        lower.ndim != 3 or lower.shape[0] != lower.shape[1]
        or lower.shape[2] < 1 or upper.shape != lower.shape
        or np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper))
        or np.any(lower < 0.0) or np.any(upper < lower)
    ):
        raise ValueError(
            "coefficient envelope must satisfy finite 0 <= lower <= upper "
            "with shape (K,K,Q)")
    return lower, upper


def _feasible_incumbent(
    power_w: np.ndarray,
    budget_w: np.ndarray,
) -> np.ndarray:
    incumbent = np.maximum(np.asarray(power_w, dtype=np.float64), 0.0).copy()
    for transmitter, budget in enumerate(budget_w):
        total = float(np.sum(incumbent[transmitter]))
        if total > float(budget) and total > 0.0:
            incumbent[transmitter] *= float(budget) / total
    return incumbent


def certified_fixed_structure_maxmin_power_repair(
    lower_coefficient: np.ndarray,
    upper_coefficient: np.ndarray,
    selected: Sequence[tuple[int, int, int]],
    positions: np.ndarray,
    existing_communication_power_w: np.ndarray,
    existing_sensing_power_w: np.ndarray,
    *,
    communication_model: CommunicationTransport,
    control_period_s: float,
    p_fa: float,
    config: CertifiedMaxMinPowerConfig = CertifiedMaxMinPowerConfig(),
) -> CertifiedMaxMinPowerDecision:
    """Return a certified power repair or the unchanged no-op allocation.

    Coefficient tensors are simultaneous lower/upper envelopes for the next
    control event. The candidate is accepted only when its lower-bound
    worst-target probability improves on the no-op upper bound and every
    target remains above ``min(noop upper, qos_floor)``.
    """
    lower, upper = _validate_coefficient_envelope(
        lower_coefficient, upper_coefficient)
    K, _, Q = lower.shape
    positions_array = np.asarray(positions, dtype=np.float64)
    existing_comm = np.asarray(
        existing_communication_power_w, dtype=np.float64).reshape(-1)
    existing_sensing = np.asarray(
        existing_sensing_power_w, dtype=np.float64)
    if (
        positions_array.shape != (K, 3) or existing_comm.shape != (K,)
        or existing_sensing.shape != (K, Q)
        or np.any(~np.isfinite(positions_array))
        or np.any(~np.isfinite(existing_comm))
        or np.any(~np.isfinite(existing_sensing))
        or np.any(existing_comm < 0.0) or np.any(existing_comm >= 1.0)
        or np.any(existing_sensing < 0.0)
        or np.any(
            np.sum(existing_sensing, axis=1) + existing_comm > 1.0 + 1.0e-9)
    ):
        raise ValueError("positions and RF allocations are outside support")

    lower_gain, owners = fixed_owner_gain_matrix(lower, selected)
    upper_gain, upper_owners = fixed_owner_gain_matrix(upper, selected)
    if not np.array_equal(owners, upper_owners):
        raise AssertionError("coefficient envelope changed target ownership")

    noop_lower_deflection = np.sum(lower_gain * existing_sensing, axis=0)
    noop_upper_deflection = np.sum(upper_gain * existing_sensing, axis=0)
    noop_lower_pd = compute_detection_probabilities(
        noop_lower_deflection, float(p_fa))
    noop_upper_pd = compute_detection_probabilities(
        noop_upper_deflection, float(p_fa))
    qos_deflection = minimum_deflection_for_detection_probability(
        np.full(Q, float(config.qos_floor), dtype=np.float64), float(p_fa))
    reserve = np.minimum(noop_upper_deflection, qos_deflection)
    protected_pd = compute_detection_probabilities(reserve, float(p_fa))

    layout = PowerRepairWireLayout(
        num_agents=K,
        num_targets=Q,
        rounds=int(config.rounds),
        price_bits=int(config.price_bits),
        deflection_bits=int(config.feedback_bits),
    )
    transport = certify_power_repair_transport(
        owners,
        positions_array,
        existing_comm,
        communication_model=communication_model,
        layout=layout,
        control_period_s=float(control_period_s),
        snr_margin_db=float(config.snr_margin_db),
        latency_margin_s=float(config.latency_margin_s),
    )
    candidate_comm = transport.projected_comm_power_w.copy()
    if transport.feasible:
        budget = np.maximum(1.0 - candidate_comm, 0.0)
        incumbent = _feasible_incumbent(existing_sensing, budget)
        optimizer = distributed_column_generation_maxmin_power(
            lower_gain,
            budget,
            rounds=int(config.rounds),
            price_bits=int(config.price_bits),
            feedback_bits=int(config.feedback_bits),
            minimum_deflection=reserve,
            incumbent_power_w=incumbent,
        )
        candidate_sensing = optimizer.power_w.copy()
    else:
        optimizer = None
        candidate_sensing = existing_sensing.copy()
        candidate_comm = existing_comm.copy()

    candidate_lower_deflection = np.sum(
        lower_gain * candidate_sensing, axis=0)
    candidate_upper_deflection = np.sum(
        upper_gain * candidate_sensing, axis=0)
    candidate_lower_pd = compute_detection_probabilities(
        candidate_lower_deflection, float(p_fa))
    candidate_upper_pd = compute_detection_probabilities(
        candidate_upper_deflection, float(p_fa))

    tolerance = float(config.probability_tolerance)
    improvement = float(
        np.min(candidate_lower_pd) - np.min(noop_upper_pd))
    reasons = []
    if not transport.feasible:
        reasons.append("transport")
    if optimizer is not None and not optimizer.reserve_feasible:
        reasons.append("qos_reserve")
    if np.any(candidate_lower_pd + tolerance < protected_pd):
        reasons.append("target_protection")
    if improvement <= float(config.minimum_worst_pd_improvement) + tolerance:
        reasons.append("no_certified_improvement")
    if np.any(
        np.sum(candidate_sensing, axis=1) + candidate_comm > 1.0 + 1.0e-9
    ):
        reasons.append("rf_budget")
    accepted = not reasons
    return CertifiedMaxMinPowerDecision(
        accepted=accepted,
        reason="accepted" if accepted else ",".join(reasons),
        sensing_power_w=(
            candidate_sensing.copy() if accepted else existing_sensing.copy()),
        communication_power_w=(
            candidate_comm.copy() if accepted else existing_comm.copy()),
        candidate_sensing_power_w=candidate_sensing,
        candidate_communication_power_w=candidate_comm,
        noop_lower_pd=noop_lower_pd,
        noop_upper_pd=noop_upper_pd,
        candidate_lower_pd=candidate_lower_pd,
        candidate_upper_pd=candidate_upper_pd,
        protected_pd=protected_pd,
        optimizer=optimizer,
        transport=transport,
    )
