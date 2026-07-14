"""Detection performance from cumulative Deflection.

Computes P_D (detection probability) from cumulative effective Deflection
using the Gaussian approximation from the OTFS-ISAC detection theory.
"""

import numpy as np
from uav_isac.utils.math_utils import compute_PD, utility_from_D


def compute_detection_probabilities(
    D_q_star: np.ndarray,  # (Q,) cumulative Deflection per target
    P_FA: float            # false alarm probability
) -> np.ndarray:
    """Compute per-target detection probabilities.

    P_D^q = Q(Q^{-1}(P_FA) - sqrt(D_q^*))

    This is the standard relationship for a deflection-based detector
    under Gaussian statistics.

    Args:
        D_q_star: Cumulative effective Deflection per target
        P_FA: False alarm probability

    Returns:
        P_D: (Q,) detection probabilities in [0, 1]
    """
    return compute_PD(D_q_star, P_FA)


def compute_target_utilities(
    D_q_star: np.ndarray,
    P_FA: float
) -> np.ndarray:
    """Compute per-target utilities from cumulative Deflection.

    U_q = -log(1 - P_D^q)

    WARNING (see docs/KNOWN_ISSUES.md B8): this is monotone INCREASING but
    NOT concave in D_q (it is convex in P_D, and empirically U''(D)>0 over
    ~99.6% of the relevant range). It therefore does NOT make the inner P0
    objective submodular; the P0 greedy has no (1-1/e) guarantee and must be
    described as a heuristic. Use a saturating utility (e.g. 1-exp(-kD)) to
    recover concavity/submodularity.

    Args:
        D_q_star: Cumulative effective Deflection per target
        P_FA: False alarm probability

    Returns:
        U_q: (Q,) utilities per target
    """
    return utility_from_D(D_q_star, P_FA)


def compute_weighted_utility(
    D_q_star: np.ndarray,
    P_FA: float,
    omega_q: np.ndarray  # (Q,) target priorities
) -> float:
    """Compute weighted sum of target utilities.

    U_total = sum_q omega_q * U_q(D_q)

    Args:
        D_q_star: Cumulative Deflection per target
        P_FA: False alarm probability
        omega_q: Target priority weights, must sum to 1

    Returns:
        Total weighted utility (scalar)
    """
    U_q = compute_target_utilities(D_q_star, P_FA)
    return float(np.dot(omega_q, U_q))


def compute_team_reward(
    D_q_star: np.ndarray,
    P_FA: float,
    omega_q: np.ndarray,
    total_bits: float,
    lambda_report: float = 0.001
) -> float:
    """Compute team reward from detection performance and communication cost.

    r_team = sum_q omega_q * U_q(D_q) - lambda_report * total_bits

    This is the base reward before marginal contribution shaping.

    Args:
        D_q_star: Cumulative Deflection per target
        P_FA: False alarm probability
        omega_q: Target priority weights
        total_bits: Total soft information bits reported this frame
        lambda_report: Communication cost coefficient

    Returns:
        Team reward (scalar)
    """
    utility = compute_weighted_utility(D_q_star, P_FA, omega_q)
    return float(utility - lambda_report * total_bits)
