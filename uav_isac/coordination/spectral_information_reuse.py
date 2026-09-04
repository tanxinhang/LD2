"""Spectrally certified information reuse primitives.

This module does not schedule waveforms.  It provides the mathematical
boundary that a later distributed scheduler must obey when one physical
waveform is credited to several target-estimation tasks under correlated
matched-filter noise/interference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SpectralInformationCertificate:
    """Auditable lower bound for one joint multi-target observation."""

    coupling_norm: float
    information_discount: float
    lower_information: np.ndarray
    actual_information: np.ndarray
    minimum_slack_eigenvalue: float


def _symmetric(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _positive_definite_inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(_symmetric(matrix))
    scale = max(float(np.max(np.abs(values), initial=0.0)), 1.0)
    if float(np.min(values, initial=np.inf)) <= 1.0e-12 * scale:
        raise ValueError("noise covariance blocks must be positive definite")
    return (vectors * (1.0 / np.sqrt(values))[None, :]) @ vectors.T


def _block_diagonal_jacobian(
    jacobian_blocks: Sequence[np.ndarray],
) -> tuple[np.ndarray, tuple[int, ...]]:
    blocks = tuple(np.asarray(block, dtype=np.float64)
                   for block in jacobian_blocks)
    if not blocks:
        raise ValueError("at least one target Jacobian block is required")
    if any(block.ndim != 2 or min(block.shape) < 1 for block in blocks):
        raise ValueError("every target Jacobian must be a non-empty matrix")
    if any(not np.all(np.isfinite(block)) for block in blocks):
        raise ValueError("target Jacobians must be finite")
    row_sizes = tuple(int(block.shape[0]) for block in blocks)
    total_rows = sum(row_sizes)
    total_cols = sum(int(block.shape[1]) for block in blocks)
    joint = np.zeros((total_rows, total_cols), dtype=np.float64)
    row = 0
    col = 0
    for block in blocks:
        rows, cols = block.shape
        joint[row:row + rows, col:col + cols] = block
        row += rows
        col += cols
    return joint, row_sizes


def spectrally_certified_information(
    jacobian_blocks: Sequence[np.ndarray],
    noise_covariance: np.ndarray,
) -> SpectralInformationCertificate:
    """Return a PSD-safe information credit under correlated interference.

    Let the joint matched-filter covariance be ``R = D + E``, where ``D``
    keeps only the within-target diagonal blocks.  With

        rho = ||D^{-1/2} E D^{-1/2}||_2,

    ``R <= (1+rho)D`` and therefore

        H' R^{-1} H >= (1+rho)^{-1} H' D^{-1} H.

    The returned lower matrix is the right-hand side.  It is the maximum
    information this primitive permits a scheduler to credit without treating
    correlated target returns as independent evidence.
    """
    H, row_sizes = _block_diagonal_jacobian(jacobian_blocks)
    R = np.asarray(noise_covariance, dtype=np.float64)
    if R.shape != (H.shape[0], H.shape[0]):
        raise ValueError("noise covariance does not match Jacobian rows")
    if not np.all(np.isfinite(R)):
        raise ValueError("noise covariance must be finite")
    R = _symmetric(R)
    _positive_definite_inverse_sqrt(R)

    D = np.zeros_like(R)
    start = 0
    for size in row_sizes:
        stop = start + size
        block = R[start:stop, start:stop]
        _positive_definite_inverse_sqrt(block)
        D[start:stop, start:stop] = block
        start = stop
    D_inverse_sqrt = _positive_definite_inverse_sqrt(D)
    normalized_coupling = _symmetric(
        D_inverse_sqrt @ (R - D) @ D_inverse_sqrt)
    coupling_norm = float(np.linalg.norm(normalized_coupling, ord=2))
    discount = float(1.0 / (1.0 + coupling_norm))

    independent_information = _symmetric(H.T @ np.linalg.solve(D, H))
    actual_information = _symmetric(H.T @ np.linalg.solve(R, H))
    lower_information = discount * independent_information
    slack = _symmetric(actual_information - lower_information)
    minimum_slack = float(np.min(np.linalg.eigvalsh(slack), initial=np.inf))
    numerical_scale = max(
        float(np.linalg.norm(actual_information, ord=2)), 1.0)
    if minimum_slack < -1.0e-9 * numerical_scale:
        raise RuntimeError("spectral information lower bound failed")
    return SpectralInformationCertificate(
        coupling_norm=coupling_norm,
        information_discount=discount,
        lower_information=lower_information,
        actual_information=actual_information,
        minimum_slack_eigenvalue=minimum_slack,
    )


def information_factor(
    information: np.ndarray,
    *,
    relative_tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Return ``U`` such that a PSD information matrix is ``U U'``."""
    matrix = np.asarray(information, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("information must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("information must be finite")
    values, vectors = np.linalg.eigh(_symmetric(matrix))
    scale = max(float(np.max(np.abs(values), initial=0.0)), 1.0)
    if float(np.min(values, initial=np.inf)) < -relative_tolerance * scale:
        raise ValueError("information must be positive semidefinite")
    keep = values > relative_tolerance * scale
    return vectors[:, keep] * np.sqrt(np.maximum(values[keep], 0.0))[None, :]


def logdet_information_gain(
    prior_information: np.ndarray,
    increment_factor: np.ndarray,
) -> float:
    """Compute log-det gain with the matrix determinant lemma."""
    prior = np.asarray(prior_information, dtype=np.float64)
    factor = np.asarray(increment_factor, dtype=np.float64)
    if (prior.ndim != 2 or prior.shape[0] != prior.shape[1]
            or factor.ndim != 2 or factor.shape[0] != prior.shape[0]):
        raise ValueError("prior/factor dimensions are incompatible")
    _positive_definite_inverse_sqrt(prior)
    gram = _symmetric(factor.T @ np.linalg.solve(prior, factor))
    sign, value = np.linalg.slogdet(np.eye(gram.shape[0]) + gram)
    if sign <= 0.0:
        raise RuntimeError("information gain is not positive definite")
    return float(value)


def trace_logdet_upper_bound(
    prior_covariance: np.ndarray,
    increment_factor: np.ndarray,
) -> float:
    """Cheap upper bound ``log det(I + P U U') <= tr(U' P U)``."""
    covariance = np.asarray(prior_covariance, dtype=np.float64)
    factor = np.asarray(increment_factor, dtype=np.float64)
    if (covariance.ndim != 2
            or covariance.shape[0] != covariance.shape[1]
            or factor.ndim != 2
            or factor.shape[0] != covariance.shape[0]):
        raise ValueError("covariance/factor dimensions are incompatible")
    _positive_definite_inverse_sqrt(covariance)
    gram = _symmetric(factor.T @ covariance @ factor)
    return float(np.trace(gram))


def rank_trace_logdet_upper_bound(
    prior_covariance: np.ndarray,
    increment_factor: np.ndarray,
) -> float:
    """Tighter rank-aware upper bound for a low-rank log-det gain.

    If ``r`` is the number of columns of ``U`` and
    ``t = tr(U' P U)``, concavity of ``log(1+x)`` gives
    ``log det(I + U' P U) <= r log(1 + t/r)``.  The expression is no
    larger than the plain trace bound while retaining a strict certificate.
    """
    covariance = np.asarray(prior_covariance, dtype=np.float64)
    factor = np.asarray(increment_factor, dtype=np.float64)
    if (covariance.ndim != 2
            or covariance.shape[0] != covariance.shape[1]
            or factor.ndim != 2
            or factor.shape[0] != covariance.shape[0]):
        raise ValueError("covariance/factor dimensions are incompatible")
    _positive_definite_inverse_sqrt(covariance)
    rank_cap = int(factor.shape[1])
    if rank_cap == 0:
        return 0.0
    trace = float(np.trace(_symmetric(factor.T @ covariance @ factor)))
    return float(rank_cap * np.log1p(max(trace, 0.0) / rank_cap))


def woodbury_covariance_update(
    prior_covariance: np.ndarray,
    increment_factor: np.ndarray,
) -> np.ndarray:
    """Update covariance for ``Y_new = P^{-1} + U U'`` in low rank."""
    covariance = np.asarray(prior_covariance, dtype=np.float64)
    factor = np.asarray(increment_factor, dtype=np.float64)
    if (covariance.ndim != 2
            or covariance.shape[0] != covariance.shape[1]
            or factor.ndim != 2
            or factor.shape[0] != covariance.shape[0]):
        raise ValueError("covariance/factor dimensions are incompatible")
    _positive_definite_inverse_sqrt(covariance)
    middle = np.eye(factor.shape[1]) + factor.T @ covariance @ factor
    updated = covariance - (
        covariance @ factor
        @ np.linalg.solve(middle, factor.T @ covariance))
    return _symmetric(updated)
