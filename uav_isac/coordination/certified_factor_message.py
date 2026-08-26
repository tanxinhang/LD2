"""Certified interval primitives and bit accounting for factor messages."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Callable

import numpy as np

from uav_isac.physical.detection import compute_detection_probabilities


@dataclass(frozen=True)
class IntervalCode:
    lower: np.ndarray
    upper: np.ndarray
    scale_lower: float
    scale_upper: float
    bits_per_value: int


@dataclass(frozen=True)
class CapabilityPayloadBits:
    header_bits: int
    scale_bits: int
    value_bits: int
    dd_bits: int

    @property
    def total_bits(self) -> int:
        return self.header_bits + self.scale_bits + self.value_bits + self.dd_bits


@dataclass(frozen=True)
class FixedPlanCertificate:
    status: str
    lower_metrics: tuple[float, float, float]
    upper_metrics: tuple[float, float, float]


@dataclass(frozen=True)
class CapabilityRouteCertificate:
    """Fail-closed L1 route certificate induced by a coefficient interval.

    ``gamma_lower`` is the optimistic gauge evaluated at the upper gain
    envelope and ``gamma_upper`` is the conservative gauge evaluated at the
    lower gain envelope.  Gain monotonicity gives

        gamma_lower <= gamma_true <= gamma_upper.

    Missing gauge values are never reinterpreted as infeasibility because the
    current LP API uses ``None`` for both physical-ceiling and solver failure.
    """

    status: str
    gamma_lower: float | None
    gamma_upper: float | None


def certify_capability_route(
    gain_lower: np.ndarray,
    gain_upper: np.ndarray,
    gauge: Callable[[np.ndarray], float | None],
    *,
    threshold: float = 1.0,
    tolerance: float = 1.0e-9,
) -> CapabilityRouteCertificate:
    """Certify fixed-structure L1 feasible/fallback from monotone gain bounds.

    The callable must evaluate the *same* capability-gauge model for every
    gain matrix.  A feasible route is certified only when the worst-gain gauge
    is below the threshold; fallback is certified only when even the
    best-gain gauge exceeds it.  Every ambiguous or failed evaluation remains
    unresolved.
    """
    lower = np.asarray(gain_lower, dtype=np.float64)
    upper = np.asarray(gain_upper, dtype=np.float64)
    if lower.ndim != 2 or lower.shape != upper.shape:
        raise ValueError("gain bounds must share a two-dimensional shape")
    if (np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper))
            or np.any(lower < 0.0) or np.any(upper < lower)):
        raise ValueError("invalid gain interval")
    if not np.isfinite(threshold) or not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("invalid route threshold/tolerance")
    gamma_lower = gauge(upper)
    gamma_upper = gauge(lower)
    optimistic = None if gamma_lower is None else float(gamma_lower)
    conservative = None if gamma_upper is None else float(gamma_upper)
    if (optimistic is None or conservative is None
            or not np.isfinite(optimistic) or not np.isfinite(conservative)):
        return CapabilityRouteCertificate("UNRESOLVED", optimistic, conservative)
    # A reversed interval indicates a broken monotonicity/model contract.  It
    # must fail closed instead of issuing a route decision.
    if optimistic > conservative + tolerance:
        return CapabilityRouteCertificate("UNRESOLVED", optimistic, conservative)
    if conservative <= float(threshold) + tolerance:
        status = "CERT_L1_FEASIBLE"
    elif optimistic > float(threshold) + tolerance:
        status = "CERT_L1_FALLBACK"
    else:
        status = "UNRESOLVED"
    return CapabilityRouteCertificate(status, optimistic, conservative)


def log_interval_quantize(values: np.ndarray, bits: int) -> IntervalCode:
    """Quantize positive values into closed log-domain bins.

    The transmitted range endpoints are modeled as outward-rounded float32
    values.  Returned intervals therefore contain the float64 inputs even
    after scale serialization.
    """
    value = np.asarray(values, dtype=np.float64)
    width = int(bits)
    if width < 1:
        raise ValueError("bits must be positive")
    if np.any(~np.isfinite(value)) or np.any(value <= 0.0):
        raise ValueError("log quantization requires finite positive values")
    logged = np.log(value)
    lower32 = np.nextafter(
        np.float32(np.min(logged)), np.float32(-np.inf), dtype=np.float32)
    upper32 = np.nextafter(
        np.float32(np.max(logged)), np.float32(np.inf), dtype=np.float32)
    scale_lower = float(lower32)
    scale_upper = float(upper32)
    levels = 1 << width
    step = (scale_upper - scale_lower) / levels
    if step <= 0.0:
        return IntervalCode(
            lower=value.copy(), upper=value.copy(),
            scale_lower=scale_lower, scale_upper=scale_upper,
            bits_per_value=width)
    index = np.floor((logged - scale_lower) / step).astype(np.int64)
    index = np.clip(index, 0, levels - 1)
    log_lower = scale_lower + index * step
    log_upper = scale_lower + (index + 1) * step
    return IntervalCode(
        lower=np.exp(log_lower), upper=np.exp(log_upper),
        scale_lower=scale_lower, scale_upper=scale_upper,
        bits_per_value=width)


def linear_interval_quantize_positive(values: np.ndarray, bits: int) -> IntervalCode:
    """Quantize positive values into closed linear bins with outward scale."""
    value = np.asarray(values, dtype=np.float64)
    width = int(bits)
    if width < 1 or np.any(~np.isfinite(value)) or np.any(value <= 0.0):
        raise ValueError("linear quantization requires positive finite values and bits")
    scale_upper = float(np.nextafter(
        np.float32(np.max(value)), np.float32(np.inf), dtype=np.float32))
    levels = 1 << width
    step = scale_upper / levels
    index = np.floor(value / step).astype(np.int64)
    index = np.clip(index, 0, levels - 1)
    return IntervalCode(
        lower=index * step, upper=(index + 1) * step,
        scale_lower=0.0, scale_upper=scale_upper,
        bits_per_value=width)


def factor_coefficient_envelope(
    tx_code: IntervalCode,
    rx_code: IntervalCode,
    dd_active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return elementwise coefficient bounds with diagonal/DD zeros exact."""
    tx_lower = np.asarray(tx_code.lower, dtype=np.float64).reshape(-1)
    tx_upper = np.asarray(tx_code.upper, dtype=np.float64).reshape(-1)
    rx_lower = np.asarray(rx_code.lower, dtype=np.float64).reshape(-1)
    rx_upper = np.asarray(rx_code.upper, dtype=np.float64).reshape(-1)
    active = np.asarray(dd_active, dtype=bool)
    K = tx_lower.size
    if tx_upper.size != K or rx_lower.size != K or rx_upper.size != K:
        raise ValueError("factor intervals must have the same length")
    if active.shape != (K, K):
        raise ValueError("dd_active must have shape (K,K)")
    support = active & ~np.eye(K, dtype=bool)
    lower = np.where(support, np.outer(tx_lower, rx_lower), 0.0)
    upper = np.where(support, np.outer(tx_upper, rx_upper), 0.0)
    return lower, upper


def dense_capability_payload_bits(K: int, Q: int, bits: int) -> CapabilityPayloadBits:
    """Semantic-record cost for direct dense linear quantization."""
    if K < 2 or Q < 1 or bits < 1:
        raise ValueError("invalid dense payload dimensions")
    # protocol=4, epoch=16, bit-depth=5; one float32 global linear scale.
    return CapabilityPayloadBits(
        header_bits=25, scale_bits=32,
        value_bits=Q * K * (K - 1) * bits, dd_bits=0)


def factor_capability_payload_bits(
    K: int,
    Q: int,
    bits: int,
    dd_exceptions: int,
) -> CapabilityPayloadBits:
    """Semantic-record cost for log factors plus adaptive DD coding."""
    if K < 2 or Q < 1 or bits < 1:
        raise ValueError("invalid factor payload dimensions")
    entries = Q * K * (K - 1)
    exceptions = int(dd_exceptions)
    if exceptions < 0 or exceptions > entries:
        raise ValueError("invalid DD exception count")
    # protocol=4, epoch=16, bit-depth=5, encoding=1, DD codec=1.
    header = 27
    count_bits = int(ceil(log2(entries + 1)))
    index_bits = int(ceil(log2(entries)))
    sparse_list = count_bits + exceptions * index_bits
    full_mask = entries
    return CapabilityPayloadBits(
        header_bits=header, scale_bits=64,
        value_bits=2 * K * Q * bits,
        dd_bits=min(sparse_list, full_mask))


def task_equivalent_payload_bits(
    method: str,
    K: int,
    Q: int,
    bits: int,
    dd_exceptions: int,
) -> CapabilityPayloadBits:
    """Fair M4 record: shared header/DD support, method-specific scales/values."""
    name = str(method).strip().lower()
    if name not in {"dense_linear", "dense_log", "factor_log"}:
        raise ValueError("unknown task-equivalent payload method")
    entries = Q * K * (K - 1)
    exceptions = int(dd_exceptions)
    if K < 2 or Q < 1 or bits < 1 or exceptions < 0 or exceptions > entries:
        raise ValueError("invalid task-equivalent payload dimensions")
    count_bits = int(ceil(log2(entries + 1)))
    index_bits = int(ceil(log2(entries)))
    dd_bits = min(count_bits + exceptions * index_bits, entries)
    scales = 32 if name == "dense_linear" else 64
    values = (2 * K * Q if name == "factor_log" else entries - exceptions) * bits
    return CapabilityPayloadBits(27, scales, values, dd_bits)


def certify_fixed_plan_three_floors(
    coefficient_lower: np.ndarray,
    coefficient_upper: np.ndarray,
    owner: np.ndarray,
    sensing_power_w: np.ndarray,
    *,
    p_fa: float,
    task_floors: tuple[float, float, float, int] | list[float],
) -> FixedPlanCertificate:
    """Propagate coefficient intervals to a three-floor fixed-plan certificate."""
    lower = np.asarray(coefficient_lower, dtype=np.float64)
    upper = np.asarray(coefficient_upper, dtype=np.float64)
    owner_index = np.asarray(owner, dtype=np.int64).reshape(-1)
    power = np.asarray(sensing_power_w, dtype=np.float64)
    if lower.shape != upper.shape or lower.ndim != 3 or lower.shape[0] != lower.shape[1]:
        raise ValueError("coefficient bounds must share shape (K,K,Q)")
    K, _, Q = lower.shape
    if power.shape != (K, Q) or owner_index.shape != (Q,):
        raise ValueError("owner/power shapes do not match coefficient bounds")
    if np.any(lower < 0.0) or np.any(upper < lower) or np.any(power < 0.0):
        raise ValueError("invalid coefficient interval or sensing power")
    if np.any((owner_index < 0) | (owner_index >= K)):
        raise ValueError("owner index is out of range")
    if len(task_floors) != 4 or not 1 <= int(task_floors[3]) <= Q:
        raise ValueError("invalid task floors")
    d_lower = np.asarray([
        np.dot(lower[:, owner_index[q], q], power[:, q]) for q in range(Q)
    ])
    d_upper = np.asarray([
        np.dot(upper[:, owner_index[q], q], power[:, q]) for q in range(Q)
    ])
    pd_lower = compute_detection_probabilities(d_lower, float(p_fa))
    pd_upper = compute_detection_probabilities(d_upper, float(p_fa))
    tail_k = int(task_floors[3])

    def metrics(pd: np.ndarray) -> tuple[float, float, float]:
        return (
            float(np.min(pd)),
            float(np.mean(np.partition(pd, tail_k - 1)[:tail_k])),
            float(np.mean(pd)),
        )

    lower_metrics = metrics(pd_lower)
    upper_metrics = metrics(pd_upper)
    floors = tuple(float(value) for value in task_floors[:3])
    if all(value >= floor for value, floor in zip(lower_metrics, floors)):
        status = "CERT_FEASIBLE"
    elif any(value < floor for value, floor in zip(upper_metrics, floors)):
        status = "CERT_INFEASIBLE"
    else:
        status = "UNRESOLVED"
    return FixedPlanCertificate(status, lower_metrics, upper_metrics)
