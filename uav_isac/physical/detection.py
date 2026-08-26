"""Detection performance from cumulative Deflection.

Computes P_D (detection probability) from cumulative effective Deflection
using the Gaussian approximation from the OTFS-ISAC detection theory.
"""

import numpy as np
from uav_isac.utils.math_utils import Q_inverse, compute_PD, utility_from_D


DETECTOR_CONVENTION = "real_gaussian_shift"
DETECTOR_SCALE = 1.0


def gaussian_shift_parameters(
    deflection: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parameters of the declared G2-0.5 sufficient-statistic model.

    The detector is explicitly

    ``H0: Z ~ N(0,1)``, ``H1: Z ~ N(sqrt(D),1)``.

    Therefore ``D=(mu1-mu0)^2/var0`` and the Neyman--Pearson threshold is
    ``Q^-1(P_FA)``.  We use the real-equivalent matched-filter convention,
    so ``D=E_signal/E_noise`` and ``c_det=1``.  A complex convention whose
    noise energy is ``E|n|^2`` can introduce a factor two; it is deliberately
    not mixed into this model.
    """
    values = np.asarray(deflection, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("deflection must be finite and non-negative")
    mu0 = np.zeros_like(values)
    mu1 = np.sqrt(values)
    variance0 = np.ones_like(values)
    variance1 = np.ones_like(values)
    return mu0, mu1, variance0, variance1


def monte_carlo_gaussian_shift_roc(
    deflection: np.ndarray,
    P_FA: float,
    *,
    samples: int,
    seed: int,
) -> dict[str, np.ndarray | float | int]:
    """Monte-Carlo audit of the declared ``D -> ROC`` normalization."""
    values = np.asarray(deflection, dtype=np.float64).reshape(-1)
    mu0, mu1, _var0, _var1 = gaussian_shift_parameters(values)
    p_fa = float(P_FA)
    count = int(samples)
    if not np.isfinite(p_fa) or not 0.0 < p_fa < 1.0:
        raise ValueError("P_FA must lie in (0,1)")
    if count < 1000:
        raise ValueError("samples must be at least 1000")
    threshold = float(Q_inverse(np.asarray(p_fa)))
    rng = np.random.default_rng(int(seed))
    h0 = rng.standard_normal(count)
    h1_noise = rng.standard_normal((values.size, count))
    h1 = mu1[:, None] + h1_noise
    empirical_pfa = float(np.mean(h0 > threshold))
    empirical_pd = np.mean(h1 > threshold, axis=1)
    empirical_d = np.square(
        np.mean(h1, axis=1) - float(np.mean(h0))) / float(np.var(h0))
    return {
        "samples": count,
        "threshold": threshold,
        "analytical_pfa": p_fa,
        "empirical_pfa": empirical_pfa,
        "analytical_pd": compute_PD(values, p_fa),
        "empirical_pd": empirical_pd,
        "analytical_deflection": values.copy(),
        "empirical_deflection": empirical_d,
    }


def compute_detection_probabilities(
    D_q_star: np.ndarray,  # (Q,) cumulative Deflection per target
    P_FA: float            # false alarm probability
) -> np.ndarray:
    """Compute per-target detection probabilities.

    P_D^q = Q(Q^{-1}(P_FA) - sqrt(D_q^*))

    This relationship is exact for the declared real Gaussian shift model
    returned by :func:`gaussian_shift_parameters`, not a convention-free
    identity for every real/complex detector.

    Args:
        D_q_star: Cumulative effective Deflection per target
        P_FA: False alarm probability

    Returns:
        P_D: (Q,) detection probabilities in [0, 1]
    """
    return compute_PD(D_q_star, P_FA)


def minimum_deflection_for_detection_probability(
    probability: np.ndarray,
    P_FA: float,
) -> np.ndarray:
    """Invert the Gaussian Deflection detector monotonically.

    From ``P_D=Q(Q^{-1}(P_FA)-sqrt(D))``, the least non-negative Deflection
    attaining a requested probability is
    ``max(Q^{-1}(P_FA)-Q^{-1}(P_D), 0)^2``.  Requests below the false-alarm
    floor need no sensing Deflection; exact probability one is intentionally
    rejected because it requires unbounded Deflection in this model.
    """
    requested = np.asarray(probability, dtype=np.float64)
    p_fa = float(P_FA)
    if (
        np.any(~np.isfinite(requested))
        or np.any(requested < 0.0) or np.any(requested >= 1.0)
        or not np.isfinite(p_fa) or not 0.0 < p_fa < 1.0
    ):
        raise ValueError(
            "probability must lie in [0,1) and P_FA in (0,1)")
    root = np.maximum(
        float(Q_inverse(np.asarray(p_fa))) - Q_inverse(requested), 0.0)
    return np.square(root)


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
