"""Numerically stable mathematical utilities.

Includes Q-function (Gaussian right-tail probability), inverse Q-function,
and safe numerical operations.
"""

import numpy as np
from scipy.special import erfc, erfinv
from typing import Optional


def Q_function(x: np.ndarray) -> np.ndarray:
    """Q(x) = 0.5 * erfc(x / sqrt(2)) — Gaussian right-tail probability.

    Numerically stable for large |x|. Works with scalar or array inputs.
    """
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * erfc(x / np.sqrt(2.0))


def Q_inverse(p: np.ndarray) -> np.ndarray:
    """Inverse Q-function: Q^{-1}(p) = sqrt(2) * erfinv(1 - 2p).

    Numerically stable for p in (0, 1). Clamps extreme values.
    """
    p = np.asarray(p, dtype=np.float64)
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


def safe_sqrt(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Safe square root: sqrt(max(x, eps))."""
    return np.sqrt(np.maximum(np.asarray(x, dtype=np.float64), eps))


def utility_from_D(D_q: np.ndarray, P_FA: float) -> np.ndarray:
    """Monotone increasing utility function from Deflection.

    U_q(D_q) = -log(1 - P_D(D_q) + eps)
    Monotone increasing. NOTE: it is CONVEX (not concave) in P_D, and
    empirically NON-concave in D_q, so it does NOT make the P0 objective
    submodular (see docs/KNOWN_ISSUES.md B8). The P0 greedy is heuristic.

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
