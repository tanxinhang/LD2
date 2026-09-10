"""Deterministic fixed-repetition envelope for bounded U2U erasures.

Every logical protocol packet is transmitted a fixed ``R`` times.  Hence the
receiver obtains at least one copy whenever at most ``R-1`` copies of each
logical packet are erased.  This is a deterministic support statement; it
does not infer a packet-error probability or assume independent erasures.

The controller may execute parallel compute/transport branches.  Repetition
therefore multiplies only the communication latency in each branch, not the
local computation.  Both parallel branches still consume RF bits and energy.
"""

# ----------------------------------------------------------------------
# AUDIT/RESEARCH-ONLY MODULE (2026-08-16 audit remediation)
#
# This module is consumed only by tools/ audit scripts and tests. It is
# NOT part of the deployment execution path (env_core / trainer) and its
# results must not be described as deployed behaviour. It exists to keep
# a specific research question reproducible; see
# docs/EXPERIMENT_LOG.md for the associated gate.
# ----------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BoundedRepetitionPathCertificate:
    feasible: bool
    reasons: tuple[str, ...]
    repetition_count: int
    tolerated_erasures_per_logical_packet: int
    excess_queue_bound_s: float
    nominal_total_latency_s: float
    worst_case_total_latency_s: float
    nominal_over_air_bits: int
    worst_case_over_air_bits: int
    nominal_rf_energy_j: float
    worst_case_rf_energy_j: float


def _nonnegative_scalar(value: float, name: str) -> float:
    result = float(value)
    if np.isnan(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and not NaN")
    return result


def _nonnegative_sequence(
    values: Sequence[float], name: str,
) -> tuple[float, ...]:
    result = tuple(
        _nonnegative_scalar(value, f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    return result


def certify_bounded_repetition_path(
    *,
    common_compute_s: float,
    parallel_branch_compute_s: Sequence[float],
    parallel_branch_protocol_latency_s: Sequence[float],
    serial_compute_s: float,
    serial_protocol_latency_s: Sequence[float],
    nominal_over_air_bits: int,
    nominal_rf_energy_j: float,
    repetition_count: int,
    excess_queue_bound_s: float,
    deadline_s: float,
    nominal_transport_feasible: bool,
) -> BoundedRepetitionPathCertificate:
    """Certify a complete compute/transport path under fixed repetition.

    For common compute ``C0``, parallel branch compute/communication
    ``(Cb,Lb)``, serial compute ``Cs`` and serial protocol latencies ``Ls``,

    ``T_R = C0 + max_b(Cb + R Lb) + sum(Ls * R) + Cs + J``.

    The ordered additions deliberately preserve the nominal controller's
    expression at ``R=1, J=0``.  Fixed repetition multiplies transmitted bits
    and modeled RF energy by ``R`` while leaving instantaneous RF power and
    all sensing coefficients unchanged.
    """
    repetitions = int(repetition_count)
    if repetitions < 1:
        raise ValueError("repetition_count must be positive")
    branch_compute = _nonnegative_sequence(
        parallel_branch_compute_s, "parallel_branch_compute_s")
    branch_latency = _nonnegative_sequence(
        parallel_branch_protocol_latency_s,
        "parallel_branch_protocol_latency_s",
    )
    if not branch_compute or len(branch_compute) != len(branch_latency):
        raise ValueError(
            "parallel compute and protocol latency require equal nonzero size")
    serial_latency = _nonnegative_sequence(
        serial_protocol_latency_s, "serial_protocol_latency_s")
    common_compute = _nonnegative_scalar(
        common_compute_s, "common_compute_s")
    serial_compute = _nonnegative_scalar(serial_compute_s, "serial_compute_s")
    queue_bound = _nonnegative_scalar(
        excess_queue_bound_s, "excess_queue_bound_s")
    deadline = _nonnegative_scalar(deadline_s, "deadline_s")
    nominal_energy = _nonnegative_scalar(
        nominal_rf_energy_j, "nominal_rf_energy_j")
    nominal_bits = int(nominal_over_air_bits)
    if nominal_bits < 0 or nominal_bits != nominal_over_air_bits:
        raise ValueError("nominal_over_air_bits must be a non-negative integer")

    nominal_latency = common_compute + max(
        compute + latency
        for compute, latency in zip(branch_compute, branch_latency)
    )
    for latency in serial_latency:
        nominal_latency += latency
    nominal_latency += serial_compute

    robust_latency = common_compute + max(
        compute + repetitions * latency
        for compute, latency in zip(branch_compute, branch_latency)
    )
    for latency in serial_latency:
        robust_latency += repetitions * latency
    robust_latency += serial_compute
    robust_latency += queue_bound

    reasons: list[str] = []
    if not bool(nominal_transport_feasible):
        reasons.append("transport:nominal_infeasible")
    if robust_latency > deadline + 1.0e-12:
        reasons.append("repetition:end_to_end_deadline")
    return BoundedRepetitionPathCertificate(
        feasible=not reasons,
        reasons=tuple(reasons),
        repetition_count=repetitions,
        tolerated_erasures_per_logical_packet=repetitions - 1,
        excess_queue_bound_s=queue_bound,
        nominal_total_latency_s=float(nominal_latency),
        worst_case_total_latency_s=float(robust_latency),
        nominal_over_air_bits=nominal_bits,
        worst_case_over_air_bits=int(repetitions * nominal_bits),
        nominal_rf_energy_j=nominal_energy,
        worst_case_rf_energy_j=float(repetitions * nominal_energy),
    )
