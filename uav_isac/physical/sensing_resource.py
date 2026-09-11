"""General sensing-resource and separability contracts.

This module is deliberately a thin, offline contract layer.  It does not
schedule waveforms, solve the target-power LP, or alter online behavior.

Physical streams are indexed independently of target truth.  When a detector
uses a bank of hypothesis-cell signatures, separability is measured by their
noise-whitened, column-normalized Gram matrix.  A mode label never certifies
separability by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np


class SensingMode(str, Enum):
    """Declared interpretation of physical sensing streams."""

    COMMON_PROBE = "common_probe"
    TARGET_SEPARABLE = "target_separable"


@dataclass(frozen=True)
class ResourceOccupancy:
    """Discrete time-frequency occupancy of physical streams.

    ``mask`` has shape ``(K,L,T,F)``.  Code and spatial resources are labels,
    not extra scalar multipliers: unlike time-frequency area, they do not have
    commensurate units and must not be silently collapsed into one number.

    ``hypothesis_ids`` identifies intended detector cells only in the declared
    target-separable specialization.  It must never contain simulator truth
    that is unavailable to the runtime scheduler.
    """

    mode: SensingMode
    mask: np.ndarray
    stream_ids: tuple[str, ...]
    hypothesis_ids: tuple[int | None, ...]
    code_ids: tuple[str | None, ...] = ()
    spatial_ids: tuple[str | None, ...] = ()

    def validated(self) -> "ResourceOccupancy":
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 4 or any(size < 1 for size in mask.shape):
            raise ValueError("resource mask must have shape (K,L,T,F)")
        streams = mask.shape[1]
        if len(self.stream_ids) != streams or len(set(self.stream_ids)) != streams:
            raise ValueError("stream_ids must uniquely identify every stream")
        if len(self.hypothesis_ids) != streams:
            raise ValueError("hypothesis_ids must match the stream dimension")
        for labels, name in (
            (self.code_ids, "code_ids"),
            (self.spatial_ids, "spatial_ids"),
        ):
            if labels and len(labels) != streams:
                raise ValueError(f"{name} must be empty or match stream dimension")
        mode = SensingMode(self.mode)
        if mode is SensingMode.COMMON_PROBE:
            if any(value is not None for value in self.hypothesis_ids):
                raise ValueError(
                    "common-probe streams cannot carry target/hypothesis identity")
        else:
            valid_hypotheses = all(
                isinstance(value, (int, np.integer))
                and not isinstance(value, (bool, np.bool_))
                and int(value) >= 0
                for value in self.hypothesis_ids
            )
            if not valid_hypotheses:
                raise ValueError(
                    "target-separable streams require observable hypothesis IDs")
        return self


@dataclass(frozen=True)
class SensingResourceCaps:
    """Numerical resource caps for one sensing epoch."""

    epoch_duration_s: float
    bandwidth_hz: float
    peak_rf_power_w: np.ndarray
    sensing_energy_cap_j: np.ndarray
    total_rf_energy_cap_j: np.ndarray
    maximum_tf_fraction: np.ndarray


@dataclass(frozen=True)
class SensingResourceAudit:
    """Conservation result without any scheduling side effect."""

    passed: bool
    failure_reasons: tuple[str, ...]
    sensing_energy_j: np.ndarray
    total_rf_energy_j: np.ndarray
    maximum_concurrent_power_envelope_w: np.ndarray
    occupied_tf_fraction: np.ndarray
    occupied_tf_area_hz_s: np.ndarray
    energy_accounting_mode: str


@dataclass(frozen=True)
class SeparationDiagnostics:
    """Scale-invariant diagnostics for a detector hypothesis bank."""

    normalized_whitened_gram: np.ndarray
    mutual_coherence: float
    normalized_identity_error: float
    minimum_eigenvalue: float
    condition_number: float
    numerical_rank: int


@dataclass(frozen=True)
class LPSpecializationCertificate:
    """Fail-closed certificate for the existing target-power LP abstraction."""

    passed: bool
    failure_reasons: tuple[str, ...]
    resource_audit: SensingResourceAudit
    separation_diagnostics: SeparationDiagnostics | None
    equivalent_average_power_w: np.ndarray | None


def _nonnegative_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (int(size),) or np.any(~np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite length-K vector")
    if np.any(vector < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return vector


def _power_trace(value: np.ndarray, K: int, T: int, name: str) -> np.ndarray:
    trace = np.asarray(value, dtype=np.float64)
    if trace.shape != (K, T) or np.any(~np.isfinite(trace)):
        raise ValueError(f"{name} must be a finite (K,T) array")
    if np.any(trace < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return trace


def audit_sensing_resource_conservation(
    occupancy: ResourceOccupancy,
    stream_energy_j: np.ndarray,
    stream_power_envelope_w: np.ndarray,
    caps: SensingResourceCaps,
    *,
    communication_energy_j: np.ndarray | None = None,
    communication_power_envelope_w: np.ndarray | None = None,
    measured_total_rf_energy_j: np.ndarray | None = None,
    measured_total_rf_power_envelope_w: np.ndarray | None = None,
) -> SensingResourceAudit:
    """Audit power, energy, and time-frequency occupancy conservation.

    For disjoint communication and sensing components, total RF energy is the
    explicit sum of both ledgers.  For a genuinely shared ISAC waveform that
    cannot be uniquely decomposed, callers must provide measured total RF
    energy and its total RF power envelope; the function then avoids
    double-counting semantic roles. Disjoint non-zero communication energy
    likewise requires a communication power envelope on the sensing time grid.

    ``stream_power_envelope_w`` is a conservative per-stream simultaneous
    envelope, not a claim about OFDM/OTFS sample-level PAPR.  A later waveform
    realization must still pass its own sampled/continuous peak check.
    """
    occupancy.validated()
    mask = np.asarray(occupancy.mask, dtype=bool)
    K, L, time_cells, frequency_cells = mask.shape
    energy = np.asarray(stream_energy_j, dtype=np.float64)
    envelope = np.asarray(stream_power_envelope_w, dtype=np.float64)
    if energy.shape != (K, L) or envelope.shape != (K, L):
        raise ValueError("stream energy/power must have shape (K,L)")
    if (
        np.any(~np.isfinite(energy))
        or np.any(~np.isfinite(envelope))
        or np.any(energy < 0.0)
        or np.any(envelope < 0.0)
    ):
        raise ValueError("stream energy/power must be finite and non-negative")
    duration = float(caps.epoch_duration_s)
    bandwidth = float(caps.bandwidth_hz)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("epoch_duration_s must be finite and positive")
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth_hz must be finite and positive")
    peak_cap = _nonnegative_vector(caps.peak_rf_power_w, K, "peak_rf_power_w")
    sense_cap = _nonnegative_vector(
        caps.sensing_energy_cap_j, K, "sensing_energy_cap_j")
    total_cap = _nonnegative_vector(
        caps.total_rf_energy_cap_j, K, "total_rf_energy_cap_j")
    tf_cap = np.asarray(caps.maximum_tf_fraction, dtype=np.float64).reshape(-1)
    if (
        tf_cap.shape != (K,)
        or np.any(~np.isfinite(tf_cap))
        or np.any(tf_cap < 0.0)
        or np.any(tf_cap > 1.0)
    ):
        raise ValueError("maximum_tf_fraction must lie in [0,1]")

    active_stream = np.any(mask, axis=(2, 3))
    if np.any((energy > 0.0) & ~active_stream):
        raise ValueError("positive stream energy requires occupied resources")
    active_time = np.any(mask, axis=3)
    support_duration = np.sum(active_time, axis=2) * duration / time_cells
    sensing_energy = np.sum(energy, axis=1)
    sensing_concurrent_envelope = np.sum(
        envelope[:, :, None] * active_time, axis=1)
    union_mask = np.any(mask, axis=1)
    occupied_count = np.sum(union_mask, axis=(1, 2))
    occupied_fraction = occupied_count / float(time_cells * frequency_cells)
    occupied_area = occupied_fraction * duration * bandwidth

    failures: list[str] = []
    if np.any(energy > envelope * support_duration + 1.0e-12):
        failures.append("stream_energy_exceeds_power_time_envelope")
    if np.any(sensing_energy > sense_cap + 1.0e-12):
        failures.append("sensing_energy_cap_exceeded")
    if np.any(occupied_fraction > tf_cap + 1.0e-12):
        failures.append("time_frequency_occupancy_exceeded")

    if measured_total_rf_energy_j is not None:
        if (
            communication_energy_j is not None
            or communication_power_envelope_w is not None
        ):
            raise ValueError(
                "shared total RF measurements cannot use additive communication ledgers")
        if measured_total_rf_power_envelope_w is None:
            raise ValueError("shared waveform audit requires a total RF power envelope")
        total_energy = np.asarray(
            measured_total_rf_energy_j, dtype=np.float64).reshape(-1)
        total_power_trace = _power_trace(
            measured_total_rf_power_envelope_w,
            K,
            time_cells,
            "measured_total_rf_power_envelope_w",
        )
        accounting_mode = "measured_shared_waveform_total"
    else:
        if measured_total_rf_power_envelope_w is not None:
            raise ValueError("total RF power envelope requires measured total RF energy")
        communication = (
            np.zeros(K, dtype=np.float64)
            if communication_energy_j is None
            else np.asarray(communication_energy_j, dtype=np.float64).reshape(-1)
        )
        if communication.shape != (K,) or np.any(~np.isfinite(communication)):
            raise ValueError("communication_energy_j must be a finite length-K vector")
        if np.any(communication < 0.0):
            raise ValueError("communication_energy_j must be non-negative")
        if communication_power_envelope_w is None:
            if np.any(communication > 0.0):
                raise ValueError(
                    "non-zero communication energy requires a power envelope")
            communication_power = np.zeros(
                (K, time_cells), dtype=np.float64)
        else:
            communication_power = _power_trace(
                communication_power_envelope_w,
                K,
                time_cells,
                "communication_power_envelope_w",
            )
        total_power_trace = sensing_concurrent_envelope + communication_power
        total_energy = sensing_energy + communication
        accounting_mode = "disjoint_additive_components"
    if total_energy.shape != (K,) or np.any(~np.isfinite(total_energy)):
        raise ValueError("measured_total_rf_energy_j must be finite and length K")
    if np.any(total_energy < sensing_energy - 1.0e-12):
        failures.append("total_rf_energy_below_sensing_energy")
    envelope_energy = np.sum(total_power_trace, axis=1) * duration / time_cells
    if np.any(total_energy > envelope_energy + 1.0e-12):
        failures.append("total_rf_energy_exceeds_power_time_envelope")
    maximum_concurrent = np.max(total_power_trace, axis=1)
    if np.any(maximum_concurrent > peak_cap + 1.0e-12):
        failures.append("peak_rf_power_exceeded")
    if np.any(total_energy > total_cap + 1.0e-12):
        failures.append("total_rf_energy_cap_exceeded")
    return SensingResourceAudit(
        passed=not failures,
        failure_reasons=tuple(failures),
        sensing_energy_j=sensing_energy,
        total_rf_energy_j=total_energy,
        maximum_concurrent_power_envelope_w=maximum_concurrent,
        occupied_tf_fraction=occupied_fraction,
        occupied_tf_area_hz_s=occupied_area,
        energy_accounting_mode=accounting_mode,
    )


def separation_diagnostics(
    hypothesis_signatures: np.ndarray,
    noise_covariance: np.ndarray | None = None,
) -> SeparationDiagnostics:
    """Measure hypothesis separability after noise whitening and normalization.

    Columns are detector hypothesis cells, not simulator target identities.
    The normalized matrix is invariant to independent positive rescaling of
    each signature. Independent phase rotations leave the scalar diagnostics
    invariant, although they rotate off-diagonal Gram phases.
    """
    signatures = np.asarray(hypothesis_signatures, dtype=np.complex128)
    if signatures.ndim != 2 or min(signatures.shape) < 1:
        raise ValueError("hypothesis_signatures must have shape (observations,H)")
    if np.any(~np.isfinite(signatures)):
        raise ValueError("hypothesis signatures must be finite")
    observations, hypotheses = signatures.shape
    if noise_covariance is None:
        whitened = signatures
    else:
        covariance = np.asarray(noise_covariance, dtype=np.complex128)
        if covariance.shape != (observations, observations):
            raise ValueError("noise_covariance has incompatible shape")
        if np.any(~np.isfinite(covariance)):
            raise ValueError("noise_covariance must be finite")
        if not np.allclose(covariance, covariance.conj().T, atol=1.0e-12):
            raise ValueError("noise_covariance must be Hermitian")
        try:
            cholesky = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("noise_covariance must be positive definite") from exc
        whitened = np.linalg.solve(cholesky, signatures)
    gram = whitened.conj().T @ whitened
    norm = np.sqrt(np.maximum(np.real(np.diag(gram)), 0.0))
    if np.any(norm <= 1.0e-12):
        raise ValueError("every hypothesis signature must have non-zero energy")
    normalized = gram / np.outer(norm, norm)
    normalized = 0.5 * (normalized + normalized.conj().T)
    identity = np.eye(hypotheses, dtype=np.complex128)
    off_diagonal = normalized - identity
    if hypotheses == 1:
        coherence = 0.0
    else:
        coherence = float(np.max(np.abs(off_diagonal)))
    eigenvalues = np.linalg.eigvalsh(normalized)
    minimum = float(max(eigenvalues[0], 0.0))
    maximum = float(max(eigenvalues[-1], 0.0))
    condition = float("inf") if minimum <= 1.0e-12 else maximum / minimum
    rank = int(np.linalg.matrix_rank(normalized, tol=1.0e-10))
    return SeparationDiagnostics(
        normalized_whitened_gram=normalized,
        mutual_coherence=coherence,
        normalized_identity_error=float(
            np.linalg.norm(off_diagonal, ord="fro") / np.sqrt(hypotheses)),
        minimum_eigenvalue=minimum,
        condition_number=condition,
        numerical_rank=rank,
    )


def separation_is_certified(
    diagnostics: SeparationDiagnostics,
    *,
    maximum_mutual_coherence: float,
    minimum_eigenvalue: float,
) -> bool:
    """Apply explicit, caller-owned thresholds to separation diagnostics."""
    coherence_limit = float(maximum_mutual_coherence)
    eigenvalue_floor = float(minimum_eigenvalue)
    if (
        not np.isfinite(coherence_limit)
        or not 0.0 <= coherence_limit < 1.0
        or not np.isfinite(eigenvalue_floor)
        or not 0.0 < eigenvalue_floor <= 1.0
    ):
        raise ValueError("separation thresholds lie outside their valid ranges")
    return bool(
        diagnostics.mutual_coherence <= coherence_limit
        and diagnostics.minimum_eigenvalue >= eigenvalue_floor
    )


def equivalent_frame_average_power(
    stream_energy_j: np.ndarray,
    sensing_epoch_duration_s: float,
) -> np.ndarray:
    """Map target-separable stream energy to the existing LP power variable."""
    energy = np.asarray(stream_energy_j, dtype=np.float64)
    duration = float(sensing_epoch_duration_s)
    if (
        np.any(~np.isfinite(energy))
        or np.any(energy < 0.0)
        or not np.isfinite(duration)
        or duration <= 0.0
    ):
        raise ValueError("energy and sensing duration must be physically valid")
    return energy / duration


def certify_target_separable_lp_specialization(
    occupancy: ResourceOccupancy,
    stream_energy_j: np.ndarray,
    stream_power_envelope_w: np.ndarray,
    caps: SensingResourceCaps,
    hypothesis_signatures: np.ndarray,
    *,
    maximum_mutual_coherence: float,
    minimum_eigenvalue: float,
    noise_covariance: np.ndarray | None = None,
    communication_energy_j: np.ndarray | None = None,
    communication_power_envelope_w: np.ndarray | None = None,
    measured_total_rf_energy_j: np.ndarray | None = None,
    measured_total_rf_power_envelope_w: np.ndarray | None = None,
) -> LPSpecializationCertificate:
    """Certify when stream energy may be represented by existing ``p_iq``.

    This combines resource and separation checks in one call so a passing
    resource audit cannot accidentally be paired with another occupancy.  The
    mapping remains offline and does not execute or modify the LP.
    """
    occupancy.validated()
    resource = audit_sensing_resource_conservation(
        occupancy,
        stream_energy_j,
        stream_power_envelope_w,
        caps,
        communication_energy_j=communication_energy_j,
        communication_power_envelope_w=communication_power_envelope_w,
        measured_total_rf_energy_j=measured_total_rf_energy_j,
        measured_total_rf_power_envelope_w=(
            measured_total_rf_power_envelope_w),
    )
    reasons = list(resource.failure_reasons)
    if SensingMode(occupancy.mode) is not SensingMode.TARGET_SEPARABLE:
        reasons.append("common_probe_has_no_target_separable_lp_mapping")
        return LPSpecializationCertificate(
            passed=False,
            failure_reasons=tuple(reasons),
            resource_audit=resource,
            separation_diagnostics=None,
            equivalent_average_power_w=None,
        )

    hypothesis_ids = tuple(int(value) for value in occupancy.hypothesis_ids)
    if len(set(hypothesis_ids)) != len(hypothesis_ids):
        reasons.append("hypothesis_ids_are_not_one_to_one_with_lp_columns")
    signatures = np.asarray(hypothesis_signatures)
    streams = np.asarray(occupancy.mask).shape[1]
    if signatures.ndim != 2 or signatures.shape[1] != streams:
        raise ValueError("signature columns must match target-separable streams")
    diagnostic = separation_diagnostics(signatures, noise_covariance)
    if not separation_is_certified(
        diagnostic,
        maximum_mutual_coherence=maximum_mutual_coherence,
        minimum_eigenvalue=minimum_eigenvalue,
    ):
        reasons.append("separation_not_certified")
    passed = not reasons
    power = (
        equivalent_frame_average_power(
            stream_energy_j, float(caps.epoch_duration_s))
        if passed else None
    )
    return LPSpecializationCertificate(
        passed=passed,
        failure_reasons=tuple(reasons),
        resource_audit=resource,
        separation_diagnostics=diagnostic,
        equivalent_average_power_w=power,
    )
