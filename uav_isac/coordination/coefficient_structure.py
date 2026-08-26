"""Spectral diagnostics for dense bistatic ISAC coefficient matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectralStructure:
    singular_values: tuple[float, ...]
    rank_90: int
    rank_95: int
    rank_99: int
    stable_rank: float
    entropy_effective_rank: float
    rank1_relative_frobenius_error: float


@dataclass(frozen=True)
class BistaticFactorization:
    tx_factor: np.ndarray
    rx_factor: np.ndarray
    inactive_offdiagonal: tuple[tuple[int, int], ...]

    def dense(self) -> np.ndarray:
        value = np.outer(self.tx_factor, self.rx_factor)
        np.fill_diagonal(value, 0.0)
        for i, j in self.inactive_offdiagonal:
            value[i, j] = 0.0
        return value


def balance_bistatic_factor_gauge(
    factorization: BistaticFactorization,
) -> BistaticFactorization:
    """Center TX/RX log ranges using the exact outer-product gauge freedom."""
    tx = np.asarray(factorization.tx_factor, dtype=np.float64)
    rx = np.asarray(factorization.rx_factor, dtype=np.float64)
    if np.any(tx <= 0.0) or np.any(rx <= 0.0):
        raise ValueError("factor gauge balancing requires positive factors")
    tx_mid = 0.5 * (float(np.min(np.log(tx))) + float(np.max(np.log(tx))))
    rx_mid = 0.5 * (float(np.min(np.log(rx))) + float(np.max(np.log(rx))))
    shift = 0.5 * (rx_mid - tx_mid)
    return BistaticFactorization(
        tx_factor=tx * np.exp(shift),
        rx_factor=rx * np.exp(-shift),
        inactive_offdiagonal=factorization.inactive_offdiagonal,
    )


def spectral_structure(matrix: np.ndarray) -> SpectralStructure:
    """Return scale-invariant energy-rank diagnostics for one coefficient matrix."""
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or np.any(~np.isfinite(value)):
        raise ValueError("matrix must be a finite two-dimensional array")
    singular = np.linalg.svd(value, compute_uv=False)
    energy = singular ** 2
    total = float(np.sum(energy))
    if total == 0.0:
        return SpectralStructure(
            tuple(float(x) for x in singular), 0, 0, 0, 0.0, 0.0, 0.0)
    cumulative = np.cumsum(energy) / total

    def energy_rank(level: float) -> int:
        return int(np.searchsorted(cumulative, level, side="left") + 1)

    probability = energy[energy > 0.0] / total
    entropy_rank = float(np.exp(-np.sum(probability * np.log(probability))))
    rank1_error = float(np.sqrt(max(0.0, 1.0 - energy[0] / total)))
    return SpectralStructure(
        singular_values=tuple(float(x) for x in singular),
        rank_90=energy_rank(0.90),
        rank_95=energy_rank(0.95),
        rank_99=energy_rank(0.99),
        stable_rank=float(total / energy[0]),
        entropy_effective_rank=entropy_rank,
        rank1_relative_frobenius_error=rank1_error,
    )


def exact_bistatic_factorization(
    path_bistatic: np.ndarray,
    report_reliability: np.ndarray,
    dd_active: np.ndarray,
    *,
    receiver_only_tolerance: float = 1.0e-12,
) -> BistaticFactorization:
    """Recover the exact outer-product completion plus sparse DD exceptions.

    ``path_bistatic`` is the squared path magnitude with a zero diagonal.  The
    inverse-range radar law makes its off-diagonal entries ``w_i w_j``.  Report
    reliability must be receiver-only, as in the current physical model.
    """
    path = np.asarray(path_bistatic, dtype=np.float64)
    report = np.asarray(report_reliability, dtype=np.float64)
    active = np.asarray(dd_active, dtype=bool)
    if path.ndim != 2 or path.shape[0] != path.shape[1] or path.shape[0] < 3:
        raise ValueError("path_bistatic must be square with at least three UAVs")
    if report.shape != path.shape or active.shape != path.shape:
        raise ValueError("report_reliability and dd_active must match path")
    if np.any(~np.isfinite(path)) or np.any(path < 0.0):
        raise ValueError("path_bistatic must be finite and nonnegative")
    K = path.shape[0]
    tx = np.empty(K, dtype=np.float64)
    for i in range(K):
        others = [index for index in range(K) if index != i]
        j, k = others[:2]
        if path[j, k] <= 0.0 or path[i, j] <= 0.0 or path[i, k] <= 0.0:
            raise ValueError("positive off-diagonal path entries are required")
        tx[i] = np.sqrt(path[i, j] * path[i, k] / path[j, k])
    completed = np.outer(tx, tx)
    off_diagonal = ~np.eye(K, dtype=bool)
    if not np.allclose(
        completed[off_diagonal], path[off_diagonal],
        rtol=1.0e-10, atol=0.0):
        raise ValueError("path matrix does not satisfy the bistatic outer-product law")
    receiver_report = np.empty(K, dtype=np.float64)
    for j in range(K):
        values = report[np.arange(K) != j, j]
        receiver_report[j] = float(np.mean(values))
        if not np.allclose(
            values, receiver_report[j], rtol=receiver_only_tolerance,
            atol=receiver_only_tolerance):
            raise ValueError("report reliability is not receiver-only")
    inactive = tuple(
        (i, j) for i in range(K) for j in range(K)
        if i != j and not active[i, j]
    )
    return BistaticFactorization(
        tx_factor=tx,
        rx_factor=tx * receiver_report,
        inactive_offdiagonal=inactive,
    )
