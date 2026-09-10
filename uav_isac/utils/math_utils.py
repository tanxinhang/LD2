"""Numerically stable mathematical utilities.

Includes Q-function (Gaussian right-tail probability), inverse Q-function,
and safe numerical operations.
"""

import numpy as np
from functools import lru_cache
from scipy.special import erfc, erfinv
from typing import Optional


def symmetric_2x2_max_eigenvalue(matrix: np.ndarray) -> np.ndarray:
    """Largest eigenvalue of batched real symmetric 2x2 matrices.

    For ``[[a, b], [b, d]]`` the eigenvalues are

    ``(a + d +/- hypot(a - d, 2*b)) / 2``.

    The closed form is algebraically exact and avoids dispatching thousands of
    tiny matrices through a general eigensolver.  Leading batch dimensions are
    preserved.  Inputs are symmetrized in the same way as the covariance
    caller, so harmless round-off asymmetry cannot change the result.
    """
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim < 2 or value.shape[-2:] != (2, 2):
        raise ValueError("matrix must have trailing shape (2,2)")
    if np.any(~np.isfinite(value)):
        raise ValueError("matrix must be finite")
    a = value[..., 0, 0]
    d = value[..., 1, 1]
    b = 0.5 * (value[..., 0, 1] + value[..., 1, 0])
    return 0.5 * (a + d + np.hypot(a - d, 2.0 * b))


def Q_function(x: np.ndarray) -> np.ndarray:
    """Q(x) = 0.5 * erfc(x / sqrt(2)) — Gaussian right-tail probability.

    Numerically stable for large |x|. Works with scalar or array inputs.
    """
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * erfc(x / np.sqrt(2.0))


# O7 (roadmap 2026-08-29): P_FA-class inversions are hot-path scalar calls
# (33 call sites, ~32 of them size==1; e.g. inner_solver marginal-gain loops).
# The scalar path is memoized; the vectorized path (detection.py:121 requested
# array) keeps the exact same formula.  Bit-for-bit identical to recomputation
# on the same platform/library because erfinv(1-2p) is deterministic.
@lru_cache(maxsize=64)
def _q_inverse_scalar(p: float) -> float:
    # Clamp to avoid numerical issues at boundaries (same as vector path)
    p_clamped = float(np.clip(p, 1e-15, 1.0 - 1e-15))
    return float(np.sqrt(2.0) * erfinv(1.0 - 2.0 * p_clamped))


def Q_inverse(p: np.ndarray) -> np.ndarray:
    """Inverse Q-function: Q^{-1}(p) = sqrt(2) * erfinv(1 - 2p).

    Numerically stable for p in (0, 1). Clamps extreme values.
    Scalar (size==1) inputs hit the memoized scalar path; array inputs use
    the fully vectorized formula.  Output shape matches the input.
    """
    p = np.asarray(p, dtype=np.float64)
    flat = p.reshape(-1)
    if flat.size == 1:
        return np.full_like(p, _q_inverse_scalar(float(flat[0])))
    # Clamp to avoid numerical issues at boundaries
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return np.sqrt(2.0) * erfinv(1.0 - 2.0 * p)


def compute_PD(D_q: np.ndarray, P_FA: float, eps: float = 1e-10) -> np.ndarray:
    """Compute detection probability from cumulative Deflection.

    P_D^q = Q(Q^{-1}(P_FA) - sqrt(D_q^*))

    Args:
        D_q: Cumulative effective Deflection per target, shape (Q,)
        P_FA: False alarm probability
        eps: Small value for numerical stability in sqrt

    Returns:
        P_D: Detection probability per target, shape (Q,)
    """
    D_q = np.asarray(D_q, dtype=np.float64)
    q_inv = Q_inverse(np.array(P_FA))
    sqrt_D = np.sqrt(np.maximum(D_q, eps))
    return Q_function(q_inv - sqrt_D)


def utility_from_D(D_q: np.ndarray, P_FA: float) -> np.ndarray:
    """Monotone increasing utility function from Deflection.

    U_q(D_q) = -log(1 - P_D(D_q) + eps)
    Monotone increasing. NOTE: it is CONVEX (not concave) in P_D, and
    empirically NON-concave in D_q, so it does NOT make the P0 objective
    submodular (see docs/CURRENT_SYSTEM_MODEL.md §6.3). The P0 greedy is heuristic.

    Args:
        D_q: Cumulative effective Deflection per target, shape (Q,)
        P_FA: False alarm probability

    Returns:
        U_q: Utility per target, shape (Q,)
    """
    P_D = compute_PD(D_q, P_FA)
    # Clamp P_D away from 1 for log stability
    P_D_safe = np.clip(P_D, 0.0, 1.0 - 1e-12)
    return -np.log(np.maximum(1.0 - P_D_safe, 1e-12))


def marginal_utility_gain(D_q_current: float, d_eff_new: float, P_FA: float) -> float:
    """Compute marginal utility gain from adding one effective deflection.

    ΔU = U(D_q + d_eff) - U(D_q)

    Args:
        D_q_current: Current cumulative Deflection for target q
        d_eff_new: Effective Deflection of the candidate edge
        P_FA: False alarm probability

    Returns:
        Marginal utility gain (non-negative)
    """
    D_before = np.array([D_q_current])
    D_after = np.array([D_q_current + d_eff_new])
    U_before = utility_from_D(D_before, P_FA)
    U_after = utility_from_D(D_after, P_FA)
    return float(U_after[0] - U_before[0])


def marginal_utility_gain_batch(
    D_q_current: float,
    d_eff_new: np.ndarray,
    P_FA: float,
) -> np.ndarray:
    """Vectorized marginal utility for candidates of one target.

    All candidates share the same current cumulative deflection, so the
    baseline utility is evaluated once.  The formula and numerical clamps are
    otherwise identical to :func:`marginal_utility_gain`.
    """
    increments = np.asarray(d_eff_new, dtype=np.float64)
    baseline = utility_from_D(
        np.asarray([D_q_current], dtype=np.float64), P_FA,
    )[0]
    updated = utility_from_D(D_q_current + increments, P_FA)
    return np.asarray(updated - baseline, dtype=np.float64)
