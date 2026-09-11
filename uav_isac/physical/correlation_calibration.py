"""Conservative fusion of correlated OTFS matched-filter evidence.

The fixed-structure power layer relies on a linear lower bound.  For target
``q``, let the normalized edge statistics have covariance ``R_q`` and mean
components satisfying ``|mu_e|^2 = a_e p_e``.  Then

    mu^H R_q^{-1} mu >= ||mu||_2^2 / lambda_max(R_q).

Consequently, dividing every selected coefficient for target ``q`` by
``lambda_max(R_q)`` preserves the LP while lower-bounding the joint
Deflection.  ``R_q`` below is an explicit Gram matrix of finite OTFS
delay--Doppler steering atoms, so it is Hermitian positive semidefinite by
construction; no ad-hoc pairwise penalty is introduced.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np

from uav_isac.utils.types import DeflectionEntry


def _normalized_steering(coordinate_bins: np.ndarray, size: int) -> np.ndarray:
    """Return unit-norm finite-grid steering rows for fractional bin locations."""
    if int(size) < 1:
        raise ValueError("steering size must be positive")
    coordinate = np.asarray(coordinate_bins, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(coordinate)):
        raise ValueError("steering coordinates must be finite")
    sample = np.arange(int(size), dtype=np.float64)
    return np.exp(
        2j * np.pi * coordinate[:, None] * sample[None, :] / float(size)
    ) / np.sqrt(float(size))


def otfs_template_gram(
    delay_bins: np.ndarray,
    doppler_bins: np.ndarray,
    *,
    delay_size: int,
    doppler_size: int,
) -> np.ndarray:
    """Return the separable normalized OTFS template Gram matrix.

    The inner product of the two-dimensional atoms equals the product of the
    delay and Doppler steering-vector inner products.  This is the finite-grid
    Dirichlet correlation, including phase, rather than an absolute-sinc
    surrogate that need not remain positive semidefinite.
    """
    delay = np.asarray(delay_bins, dtype=np.float64).reshape(-1)
    doppler = np.asarray(doppler_bins, dtype=np.float64).reshape(-1)
    if delay.shape != doppler.shape:
        raise ValueError("delay_bins and doppler_bins must have equal length")
    if delay.size == 0:
        return np.zeros((0, 0), dtype=np.complex128)
    delay_atoms = _normalized_steering(delay, int(delay_size))
    doppler_atoms = _normalized_steering(doppler, int(doppler_size))
    gram = (
        delay_atoms @ delay_atoms.conj().T
    ) * (
        doppler_atoms @ doppler_atoms.conj().T
    )
    # Remove floating anti-Hermitian noise before eigvalsh/certificate checks.
    return 0.5 * (gram + gram.conj().T)


def selected_otfs_correlation_factors(
    selected: Sequence[tuple[int, int, int]],
    entries: Iterable[DeflectionEntry],
    *,
    num_targets: int,
    delay_size: int,
    doppler_size: int,
    delta_f_hz: float,
    symbol_time_s: float,
) -> np.ndarray:
    """Compute ``c_q=max(1, lambda_max(R_q))`` for a selected structure.

    Only edges with positive effective Deflection participate.  A missing or
    zero-gain edge contributes no statistic and therefore no correlation
    penalty.  Duplicate selected edges are rejected because counting one
    physical observation twice would invalidate both the additive model and
    its lower bound.
    """
    Q = int(num_targets)
    if Q < 1:
        raise ValueError("num_targets must be positive")
    if not np.isfinite(delta_f_hz) or delta_f_hz <= 0.0:
        raise ValueError("delta_f_hz must be finite and positive")
    if not np.isfinite(symbol_time_s) or symbol_time_s <= 0.0:
        raise ValueError("symbol_time_s must be finite and positive")

    lookup: Mapping[tuple[int, int, int], DeflectionEntry] = {
        (int(entry.i), int(entry.j), int(entry.q)): entry for entry in entries
    }
    edges = tuple(tuple(int(value) for value in edge) for edge in selected)
    if len(set(edges)) != len(edges):
        raise ValueError("selected structure contains a duplicate edge")
    coordinates: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for edge in edges:
        if len(edge) != 3:
            raise ValueError("selected edges must be (tx,rx,target) triples")
        q = edge[2]
        if not 0 <= q < Q:
            raise ValueError("selected target is outside configured support")
        entry = lookup.get(edge)
        if entry is None or float(entry.d_eff) <= 0.0:
            continue
        if not np.isfinite(entry.tau) or not np.isfinite(entry.nu):
            raise ValueError("selected OTFS coordinates must be finite")
        coordinates[q].append((
            float(entry.tau) * int(delay_size) * float(delta_f_hz),
            float(entry.nu) * int(doppler_size) * float(symbol_time_s),
        ))

    factors = np.ones(Q, dtype=np.float64)
    for q, target_coordinates in coordinates.items():
        if len(target_coordinates) <= 1:
            continue
        coordinate = np.asarray(target_coordinates, dtype=np.float64)
        gram = otfs_template_gram(
            coordinate[:, 0], coordinate[:, 1],
            delay_size=int(delay_size), doppler_size=int(doppler_size),
        )
        eigenvalues = np.linalg.eigvalsh(gram)
        if float(eigenvalues[0]) < -1.0e-10:
            raise RuntimeError("OTFS template Gram matrix is not PSD")
        factors[q] = max(1.0, float(eigenvalues[-1]))
    return factors


def selected_otfs_correlation_factors_from_arrays(
    selected: np.ndarray,
    coefficient_per_watt: np.ndarray,
    delay_s: np.ndarray,
    doppler_hz: np.ndarray,
    *,
    delay_size: int,
    doppler_size: int,
    delta_f_hz: float,
    symbol_time_s: float,
) -> np.ndarray:
    """Array-native counterpart used by structural candidate evaluation.

    Correlation is a property of the scheduled unit-power statistics, not of
    the incumbent power allocation.  Therefore every selected edge with a
    positive unit coefficient participates even when its current power is
    zero.  Invalid/off-support edges have zero coefficient and are excluded.
    """
    mask = np.asarray(selected, dtype=bool)
    coefficient = np.asarray(coefficient_per_watt, dtype=np.float64)
    tau = np.asarray(delay_s, dtype=np.float64)
    nu = np.asarray(doppler_hz, dtype=np.float64)
    if (
        mask.shape != coefficient.shape or tau.shape != mask.shape
        or nu.shape != mask.shape or mask.ndim != 3
        or mask.shape[0] != mask.shape[1]
    ):
        raise ValueError(
            "selected, coefficient, delay and Doppler must share (K,K,Q)")
    if np.any(~np.isfinite(coefficient)) or np.any(coefficient < 0.0):
        raise ValueError("coefficient must be finite and non-negative")
    active = mask & (coefficient > 0.0)
    if np.any(~np.isfinite(tau[active])) or np.any(~np.isfinite(nu[active])):
        raise ValueError("active OTFS coordinates must be finite")
    Q = mask.shape[2]
    factors = np.ones(Q, dtype=np.float64)
    for q in range(Q):
        factors[q] = otfs_target_correlation_factor_from_arrays(
            active[:, :, q],
            tau[:, :, q],
            nu[:, :, q],
            delay_size=int(delay_size),
            doppler_size=int(doppler_size),
            delta_f_hz=float(delta_f_hz),
            symbol_time_s=float(symbol_time_s),
        )
    return factors


def otfs_target_correlation_factor_from_arrays(
    active_target: np.ndarray,
    delay_s: np.ndarray,
    doppler_hz: np.ndarray,
    *,
    delay_size: int,
    doppler_size: int,
    delta_f_hz: float,
    symbol_time_s: float,
) -> float:
    """Return one exact target factor for an already validated active mask.

    This target-local primitive makes sparse structural updates cacheable.  It
    performs the same Gram construction/eigendecomposition as the dense-Q
    routine and introduces no approximation.
    """
    active = np.asarray(active_target, dtype=bool)
    tau = np.asarray(delay_s, dtype=np.float64)
    nu = np.asarray(doppler_hz, dtype=np.float64)
    if active.ndim != 2 or active.shape[0] != active.shape[1]:
        raise ValueError("active_target must have shape (K,K)")
    if tau.shape != active.shape or nu.shape != active.shape:
        raise ValueError("target delay/Doppler arrays must match active_target")
    if int(delay_size) < 1 or int(doppler_size) < 1:
        raise ValueError("OTFS grid dimensions must be positive")
    if not np.isfinite(delta_f_hz) or float(delta_f_hz) <= 0.0:
        raise ValueError("delta_f_hz must be finite and positive")
    if not np.isfinite(symbol_time_s) or float(symbol_time_s) <= 0.0:
        raise ValueError("symbol_time_s must be finite and positive")
    indices = np.argwhere(active)
    if indices.shape[0] <= 1:
        return 1.0
    target_tau = tau[active]
    target_nu = nu[active]
    if np.any(~np.isfinite(target_tau)) or np.any(~np.isfinite(target_nu)):
        raise ValueError("active OTFS coordinates must be finite")
    gram = otfs_template_gram(
        target_tau * int(delay_size) * float(delta_f_hz),
        target_nu * int(doppler_size) * float(symbol_time_s),
        delay_size=int(delay_size),
        doppler_size=int(doppler_size),
    )
    eigenvalues = np.linalg.eigvalsh(gram)
    if float(eigenvalues[0]) < -1.0e-10:
        raise RuntimeError("OTFS template Gram matrix is not PSD")
    return max(1.0, float(eigenvalues[-1]))


def apply_target_correlation_factors(
    gain_per_watt: np.ndarray,
    factors: np.ndarray,
) -> np.ndarray:
    """Apply target factors to a non-negative ``(...,Q)`` gain tensor."""
    gain = np.asarray(gain_per_watt, dtype=np.float64)
    penalty = np.asarray(factors, dtype=np.float64).reshape(-1)
    if gain.ndim < 1 or gain.shape[-1] != penalty.size:
        raise ValueError("gain target dimension must match factors")
    if (
        np.any(~np.isfinite(gain)) or np.any(gain < 0.0)
        or np.any(~np.isfinite(penalty)) or np.any(penalty < 1.0)
    ):
        raise ValueError("gain must be non-negative and factors at least one")
    return gain / penalty


def calibrate_selected_entries(
    entries: Iterable[DeflectionEntry],
    selected: Sequence[tuple[int, int, int]],
    factors: np.ndarray,
) -> list[DeflectionEntry]:
    """Scale only scheduled evidence, leaving counterfactual edges unchanged."""
    penalty = np.asarray(factors, dtype=np.float64).reshape(-1)
    selected_edges = {
        tuple(int(value) for value in edge) for edge in selected
    }
    calibrated: list[DeflectionEntry] = []
    for entry in entries:
        edge = (int(entry.i), int(entry.j), int(entry.q))
        if edge in selected_edges:
            q = edge[2]
            if not 0 <= q < penalty.size:
                raise ValueError("entry target is outside factor support")
            calibrated.append(entry._replace(
                d_eff=float(entry.d_eff) / float(penalty[q])))
        else:
            calibrated.append(entry)
    return calibrated
