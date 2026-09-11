"""Reconstruct-then-hash certificate for correlation-aware candidates.

This module deliberately stops one step short of online actuation.  Each UAV
must independently reconstruct the same fixed-structure physical model and LP
solution.  A SHA-256 digest is carried by the existing atomic three-round
protocol, while commit authority additionally requires byte-for-byte equality
of the reconstructed records.  A digest match alone is therefore never an
execution certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from time import perf_counter
from typing import Callable, Sequence

import numpy as np

from uav_isac.coordination.correlation_aware_exchange import (
    correlation_calibrated_structure_gain,
)
from uav_isac.coordination.dependency_commit import (
    DependencyCommitCertificate,
    DependencyCommitLayout,
    certify_best_dependency_commit,
    dependency_closure,
)
from uav_isac.coordination.local_exchange_oracle import (
    LocalMove,
    role_owner_from_structure,
)
from uav_isac.coordination.maxmin_power import (
    solve_fixed_structure_maxmin_power_lp,
)
from uav_isac.domain.communication import CommunicationTransport


_DOMAIN = b"uav-isac/correlation-candidate/v1\x00"


@dataclass(frozen=True)
class CorrelationCandidateRecord:
    """Complete deterministic meaning of one proposed atomic update."""

    generation_id: int
    dependency_versions: np.ndarray
    selected: np.ndarray
    role: np.ndarray
    owner: np.ndarray
    delay_size: int
    doppler_size: int
    delta_f_hz: float
    symbol_time_s: float
    correlation_factors: np.ndarray
    calibrated_gain_per_watt: np.ndarray
    sensing_budget_w: np.ndarray
    minimum_deflection: np.ndarray
    sensing_power_w: np.ndarray
    deflection: np.ndarray
    worst_deflection: float
    target_prices: np.ndarray
    dual_upper_bound: float


@dataclass(frozen=True)
class ReconstructionCertificate:
    feasible: bool
    reasons: tuple[str, ...]
    participants: tuple[int, ...]
    digest: int | None
    canonical_bytes: int


@dataclass(frozen=True)
class CorrelationCandidateCommitCertificate:
    feasible: bool
    reasons: tuple[str, ...]
    reconstruction: ReconstructionCertificate
    transport: DependencyCommitCertificate


@dataclass(frozen=True)
class CorrelationExchangeShadowResult:
    """Non-actuating audit result for one active atomic epoch."""

    attempted: bool
    accepted_by_exact_lp: bool
    would_commit: bool
    failure_reason: str | None
    baseline_worst_deflection: float
    candidate_worst_deflection: float
    improvement: float
    candidate_count: int
    dual_upper_pruned_count: int
    raw_gain_upper_pruned_count: int
    physical_prefilter_rejected_count: int
    exact_verification_count: int
    gram_compute_time_s: float
    exact_lp_time_s: float
    correlation_factor_cache_hits: int
    correlation_factor_cache_misses: int
    dual_bound_early_stopped: bool
    reconstruction_total_time_s: float
    reconstruction_parallel_critical_path_s: float
    canonical_record_bytes: int
    protocol_over_air_bits: int
    protocol_latency_s: float
    protocol_energy_j: float
    replica_agreement: bool
    active_structure_unchanged: bool


def _finite_float_array(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    # IEEE -0 and +0 are mathematically identical; normalize their encoding.
    return np.where(result == 0.0, 0.0, result)


def _array_bytes(values: np.ndarray, dtype: str) -> bytes:
    array = np.asarray(values, dtype=np.dtype(dtype))
    shape = struct.pack(">B", array.ndim) + b"".join(
        struct.pack(">I", int(size)) for size in array.shape)
    return shape + np.ascontiguousarray(array).tobytes(order="C")


def _bool_bytes(values: np.ndarray) -> bytes:
    array = np.asarray(values, dtype=bool)
    shape = struct.pack(">B", array.ndim) + b"".join(
        struct.pack(">I", int(size)) for size in array.shape)
    packed = np.packbits(array.reshape(-1), bitorder="big").tobytes()
    return shape + struct.pack(">I", len(packed)) + packed


def validate_correlation_candidate_record(
    record: CorrelationCandidateRecord,
    *,
    tolerance: float = 1.0e-10,
) -> None:
    """Check physical dimensions, simplex feasibility and LP identities."""
    generation = int(record.generation_id)
    if generation < 0 or generation != record.generation_id:
        raise ValueError("generation_id must be a non-negative integer")
    selected = np.asarray(record.selected, dtype=bool)
    if selected.ndim != 3 or selected.shape[0] != selected.shape[1]:
        raise ValueError("selected must have shape (K,K,Q)")
    K, _, Q = selected.shape
    role = np.asarray(record.role, dtype=np.int8).reshape(-1)
    owner = np.asarray(record.owner, dtype=np.int64).reshape(-1)
    versions = np.asarray(record.dependency_versions).reshape(-1)
    if role.shape != (K,) or owner.shape != (Q,) or versions.size < 1:
        raise ValueError("role, owner and dependency version shapes disagree")
    if not set(int(value) for value in role.tolist()).issubset({-1, 0, 1}):
        raise ValueError("role values must be idle(-1), receiver(0), or transmitter(1)")
    if any(int(value) < 0 or int(value) != value for value in versions.tolist()):
        raise ValueError("dependency_versions must be non-negative integers")
    inferred_role, inferred_owner = role_owner_from_structure(
        selected, fallback_role=role)
    if not np.array_equal(role, inferred_role):
        raise ValueError("role is inconsistent with selected structure")
    if not np.array_equal(owner, inferred_owner) or np.any(owner < 0):
        raise ValueError("owner is inconsistent with a complete structure")
    if np.any(np.diagonal(selected, axis1=0, axis2=1)):
        raise ValueError("self sensing edges are invalid")

    factors = _finite_float_array(record.correlation_factors, "factors")
    gain = _finite_float_array(record.calibrated_gain_per_watt, "gain")
    budget = _finite_float_array(record.sensing_budget_w, "budget").reshape(-1)
    reserve = _finite_float_array(
        record.minimum_deflection, "minimum_deflection").reshape(-1)
    power = _finite_float_array(record.sensing_power_w, "power")
    deflection = _finite_float_array(record.deflection, "deflection").reshape(-1)
    prices = _finite_float_array(record.target_prices, "target_prices").reshape(-1)
    if (
        factors.shape != (Q,) or gain.shape != (K, Q)
        or budget.shape != (K,) or reserve.shape != (Q,)
        or power.shape != (K, Q) or deflection.shape != (Q,)
        or prices.shape != (Q,)
    ):
        raise ValueError("candidate numerical arrays have inconsistent shapes")
    if (
        np.any(factors < 1.0) or np.any(gain < 0.0)
        or np.any(budget < 0.0) or np.any(reserve < 0.0)
        or np.any(power < 0.0) or np.any(prices < 0.0)
    ):
        raise ValueError("candidate violates non-negativity/correlation bounds")
    if (
        int(record.delay_size) < 1 or int(record.doppler_size) < 1
        or int(record.delay_size) != record.delay_size
        or int(record.doppler_size) != record.doppler_size
    ):
        raise ValueError("OTFS grid dimensions must be positive")
    delta_f = float(record.delta_f_hz)
    symbol_time = float(record.symbol_time_s)
    if not np.isfinite(delta_f) or delta_f <= 0.0:
        raise ValueError("delta_f_hz must be finite and positive")
    if not np.isfinite(symbol_time) or symbol_time <= 0.0:
        raise ValueError("symbol_time_s must be finite and positive")
    scale = max(1.0, float(np.max(budget)))
    tol = float(tolerance) * scale
    if np.any(np.sum(power, axis=1) > budget + tol):
        raise ValueError("sensing power violates a UAV budget simplex")
    recomputed = np.sum(gain * power, axis=0)
    deflection_scale = np.maximum(1.0, np.abs(recomputed))
    if np.any(np.abs(recomputed - deflection) > float(tolerance) * deflection_scale):
        raise ValueError("deflection is inconsistent with gain and power")
    worst = float(record.worst_deflection)
    dual = float(record.dual_upper_bound)
    if not np.isfinite(worst) or not np.isfinite(dual):
        raise ValueError("primal and dual objectives must be finite")
    objective_tol = float(tolerance) * max(1.0, abs(worst), abs(dual))
    if abs(worst - float(np.min(deflection))) > objective_tol:
        raise ValueError("worst_deflection is inconsistent with deflection")
    if np.any(deflection + float(tolerance) * np.maximum(1.0, deflection) < reserve):
        raise ValueError("minimum deflection reserve is violated")
    if abs(float(np.sum(prices)) - 1.0) > float(tolerance):
        raise ValueError("target prices must lie on the probability simplex")
    recomputed_dual = float(np.sum(
        budget * np.max(gain * prices[None, :], axis=1)))
    # The LP layer permits at most 1e-7 relative roundoff when clamping a
    # certified sub-tolerance raw dual below the reconstructed primal.
    dual_tolerance = max(float(tolerance), 1.0e-7) * max(1.0, abs(dual))
    if abs(dual - recomputed_dual) > dual_tolerance:
        raise ValueError("dual upper bound is inconsistent with gain and prices")
    if dual + objective_tol < worst:
        raise ValueError("dual upper bound lies below the primal objective")


def canonical_correlation_candidate_bytes(
    record: CorrelationCandidateRecord,
) -> bytes:
    """Serialize a validated record without repr/JSON float ambiguity."""
    validate_correlation_candidate_record(record)
    chunks = [
        _DOMAIN,
        struct.pack(">Q", int(record.generation_id)),
        _array_bytes(record.dependency_versions, ">u8"),
        _bool_bytes(record.selected),
        _array_bytes(record.role, ">i1"),
        _array_bytes(record.owner, ">i8"),
        struct.pack(">II", int(record.delay_size), int(record.doppler_size)),
        struct.pack(">dd", float(record.delta_f_hz), float(record.symbol_time_s)),
    ]
    for values in (
        record.correlation_factors,
        record.calibrated_gain_per_watt,
        record.sensing_budget_w,
        record.minimum_deflection,
        record.sensing_power_w,
        record.deflection,
        np.asarray([record.worst_deflection], dtype=np.float64),
        record.target_prices,
        np.asarray([record.dual_upper_bound], dtype=np.float64),
    ):
        normalized = _finite_float_array(values, "candidate field")
        chunks.append(_array_bytes(normalized, ">f8"))
    return b"".join(chunks)


def correlation_candidate_digest(record: CorrelationCandidateRecord) -> int:
    """Return the unsigned 256-bit SHA-256 record identity."""
    return int.from_bytes(
        hashlib.sha256(canonical_correlation_candidate_bytes(record)).digest(),
        byteorder="big",
        signed=False,
    )


def build_correlation_candidate_record(
    move: LocalMove,
    coefficient_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
    delay_s: np.ndarray,
    doppler_hz: np.ndarray,
    *,
    generation_id: int,
    dependency_versions: np.ndarray,
    delay_size: int,
    doppler_size: int,
    delta_f_hz: float,
    symbol_time_s: float,
    minimum_deflection: np.ndarray | None = None,
) -> CorrelationCandidateRecord:
    """Independently reconstruct the calibrated model and exact LP result."""
    gain, owner, factors = correlation_calibrated_structure_gain(
        coefficient_per_watt,
        move.selected,
        delay_s,
        doppler_hz,
        delay_size=int(delay_size),
        doppler_size=int(doppler_size),
        delta_f_hz=float(delta_f_hz),
        symbol_time_s=float(symbol_time_s),
    )
    if not np.array_equal(owner, np.asarray(move.owner, dtype=np.int64)):
        raise ValueError("move.owner disagrees with its selected structure")
    inferred_role, _ = role_owner_from_structure(
        move.selected, fallback_role=move.role)
    if not np.array_equal(inferred_role, np.asarray(move.role, dtype=np.int8)):
        raise ValueError("move.role disagrees with its selected structure")
    Q = gain.shape[1]
    reserve = (
        np.zeros(Q, dtype=np.float64)
        if minimum_deflection is None
        else np.asarray(minimum_deflection, dtype=np.float64).reshape(-1)
    )
    solved = solve_fixed_structure_maxmin_power_lp(
        gain,
        sensing_budget_w,
        minimum_deflection=(
            None if minimum_deflection is None else reserve),
    )
    record = CorrelationCandidateRecord(
        generation_id=int(generation_id),
        dependency_versions=np.asarray(dependency_versions).copy(),
        selected=np.asarray(move.selected, dtype=bool).copy(),
        role=np.asarray(move.role, dtype=np.int8).copy(),
        owner=np.asarray(move.owner, dtype=np.int64).copy(),
        delay_size=int(delay_size),
        doppler_size=int(doppler_size),
        delta_f_hz=float(delta_f_hz),
        symbol_time_s=float(symbol_time_s),
        correlation_factors=np.asarray(factors, dtype=np.float64).copy(),
        calibrated_gain_per_watt=np.asarray(gain, dtype=np.float64).copy(),
        sensing_budget_w=np.asarray(sensing_budget_w, dtype=np.float64).copy(),
        minimum_deflection=reserve.copy(),
        sensing_power_w=np.asarray(solved.power_w, dtype=np.float64).copy(),
        deflection=np.asarray(solved.deflection, dtype=np.float64).copy(),
        worst_deflection=float(solved.worst_deflection),
        target_prices=np.asarray(solved.prices, dtype=np.float64).copy(),
        dual_upper_bound=float(solved.dual_upper_bound),
    )
    validate_correlation_candidate_record(record)
    return record


def verify_reconstructed_correlation_candidates(
    records: Sequence[CorrelationCandidateRecord],
    participants: Sequence[int],
    *,
    _digest_fn: Callable[[bytes], bytes] = lambda payload: hashlib.sha256(payload).digest(),
) -> ReconstructionCertificate:
    """Require valid, same-generation, byte-identical participant replicas."""
    closure = tuple(sorted(set(int(value) for value in participants)))
    if not closure:
        return ReconstructionCertificate(True, tuple(), tuple(), None, 0)
    reasons: list[str] = []
    payloads: dict[int, bytes] = {}
    digests: dict[int, bytes] = {}
    for participant in closure:
        if participant < 0 or participant >= len(records):
            reasons.append(f"protocol:missing_candidate:uav:{participant}")
            continue
        try:
            payload = canonical_correlation_candidate_bytes(records[participant])
            digest = bytes(_digest_fn(payload))
            if len(digest) != 32:
                raise ValueError("digest function must return exactly 32 bytes")
            payloads[participant] = payload
            digests[participant] = digest
        except (TypeError, ValueError) as exc:
            reasons.append(
                f"protocol:invalid_candidate:uav:{participant}:{type(exc).__name__}")
    if not payloads:
        return ReconstructionCertificate(
            False, tuple(reasons), closure, None, 0)
    reference = next(iter(payloads))
    reference_payload = payloads[reference]
    reference_digest = digests[reference]
    for participant in closure:
        if participant not in payloads:
            continue
        if digests[participant] != reference_digest:
            reasons.append(f"protocol:candidate_digest:uav:{participant}")
        # This comparison is mandatory even when hashes agree.
        if payloads[participant] != reference_payload:
            reasons.append(f"protocol:canonical_record:uav:{participant}")
    return ReconstructionCertificate(
        feasible=not reasons,
        reasons=tuple(reasons),
        participants=closure,
        digest=int.from_bytes(reference_digest, "big"),
        canonical_bytes=len(reference_payload),
    )


def correlation_candidate_commit_layout(
    num_agents: int,
    num_targets: int,
) -> DependencyCommitLayout:
    """Wire layout with a full SHA-256 digest and 32-bit generation field."""
    return DependencyCommitLayout(
        int(num_agents), int(num_targets), epoch_bits=32, digest_bits=256)


def certify_correlation_candidate_commit(
    selected: np.ndarray,
    role: np.ndarray,
    owner: np.ndarray,
    move: LocalMove,
    *,
    replica_records: Sequence[CorrelationCandidateRecord],
    positions: np.ndarray,
    comm_power_w: np.ndarray,
    communication_model: CommunicationTransport,
    total_power_w: float = 1.0,
    total_deadline_s: float | None = None,
) -> CorrelationCandidateCommitCertificate:
    """Combine semantic replica agreement with the physical atomic protocol."""
    current = np.asarray(selected, dtype=bool)
    K, _, Q = current.shape
    closure = dependency_closure(current, role, owner, move)
    reconstructed = verify_reconstructed_correlation_candidates(
        replica_records, closure.participants)

    # Populate per-node identities from each local reconstruction.  Invalid or
    # missing entries deliberately disagree and force the transport layer shut.
    epochs = np.zeros(K, dtype=np.uint64)
    digests = np.zeros(K, dtype=object)
    versions = np.zeros(K, dtype=np.uint64)
    for participant in closure.participants:
        if 0 <= participant < len(replica_records):
            record = replica_records[participant]
            try:
                epochs[participant] = int(record.generation_id)
                digests[participant] = correlation_candidate_digest(record)
                # The dependency vector itself is digest-bound.  The legacy
                # scalar state-version slot carries the common generation so
                # heterogeneous per-resource version numbers are not falsely
                # required to equal one another.
                versions[participant] = int(record.generation_id)
            except (TypeError, ValueError, IndexError, OverflowError):
                epochs[participant] = participant + 1
                digests[participant] = participant + 1
                versions[participant] = participant + 1
    reference_record = (
        replica_records[closure.participants[0]]
        if closure.participants
        and closure.participants[0] < len(replica_records)
        else None
    )
    sensing_power = (
        np.zeros((K, Q), dtype=np.float64)
        if reference_record is None
        else np.asarray(reference_record.sensing_power_w, dtype=np.float64)
    )
    transport = certify_best_dependency_commit(
        current,
        role,
        owner,
        move,
        positions=positions,
        comm_power_w=comm_power_w,
        sensing_power_w=sensing_power,
        state_versions=versions,
        certificate_epoch_ids=epochs,
        certificate_digests=digests,
        communication_model=communication_model,
        total_power_w=float(total_power_w),
        total_deadline_s=total_deadline_s,
        layout=correlation_candidate_commit_layout(K, Q),
    )
    reasons = tuple(reconstructed.reasons) + tuple(transport.reasons)
    return CorrelationCandidateCommitCertificate(
        feasible=reconstructed.feasible and transport.feasible,
        reasons=reasons,
        reconstruction=reconstructed,
        transport=transport,
    )


def evaluate_correlation_exchange_shadow(
    selected: np.ndarray,
    role: np.ndarray,
    owner: np.ndarray,
    coefficient_per_watt: np.ndarray,
    candidate_mask: np.ndarray,
    sensing_budget_w: np.ndarray,
    delay_s: np.ndarray,
    doppler_hz: np.ndarray,
    *,
    generation_id: int,
    dependency_versions: np.ndarray,
    dependency_ready: bool,
    positions: np.ndarray,
    comm_power_w: np.ndarray,
    communication_model: CommunicationTransport,
    delay_size: int,
    doppler_size: int,
    delta_f_hz: float,
    symbol_time_s: float,
    target_pair_limit: int,
    reports_per_receiver: int,
    top_m: int = 12,
    weak_target_count: int = 2,
    minimum_deflection: np.ndarray | None = None,
    total_power_w: float = 1.0,
    remaining_deadline_s: float | None = None,
) -> CorrelationExchangeShadowResult:
    """Evaluate, reconstruct and physically certify without mutating inputs.

    The routine intentionally returns only a counterfactual result.  It has no
    callback or state-machine handle with which to stage or activate ``move``.
    """
    from uav_isac.coordination.correlation_aware_exchange import (
        correlation_aware_dual_pruned_exact_exchange,
    )

    active_before = np.asarray(selected, dtype=bool).copy()
    K = active_before.shape[0]

    def physical_prefilter(move: LocalMove) -> bool:
        identity = np.full(K, int(generation_id), dtype=np.uint64)
        digest = np.ones(K, dtype=object)
        certificate = certify_best_dependency_commit(
            active_before,
            role,
            owner,
            move,
            positions=positions,
            comm_power_w=comm_power_w,
            sensing_power_w=sensing_budget_w,
            state_versions=identity,
            certificate_epoch_ids=identity,
            certificate_digests=digest,
            communication_model=communication_model,
            total_power_w=float(total_power_w),
            total_deadline_s=remaining_deadline_s,
            layout=correlation_candidate_commit_layout(
                active_before.shape[0], active_before.shape[2]),
        )
        return certificate.feasible

    exchange = correlation_aware_dual_pruned_exact_exchange(
        active_before,
        coefficient_per_watt,
        candidate_mask,
        role,
        sensing_budget_w,
        delay_s,
        doppler_hz,
        delay_size=int(delay_size),
        doppler_size=int(doppler_size),
        delta_f_hz=float(delta_f_hz),
        symbol_time_s=float(symbol_time_s),
        target_pair_limit=int(target_pair_limit),
        reports_per_receiver=int(reports_per_receiver),
        top_m=int(top_m),
        weak_target_count=int(weak_target_count),
        minimum_deflection=minimum_deflection,
        pre_candidate_gate=(physical_prefilter if dependency_ready else None),
    )
    baseline_worst = float(exchange.baseline_worst_deflection)

    common = dict(
        attempted=True,
        accepted_by_exact_lp=bool(exchange.accepted),
        baseline_worst_deflection=baseline_worst,
        candidate_worst_deflection=float(
            exchange.power_result.worst_deflection),
        improvement=float(
            exchange.power_result.worst_deflection - baseline_worst),
        candidate_count=int(exchange.candidate_count),
        dual_upper_pruned_count=int(exchange.dual_upper_pruned_count),
        raw_gain_upper_pruned_count=int(
            exchange.raw_gain_upper_pruned_count),
        physical_prefilter_rejected_count=int(
            exchange.physical_prefilter_rejected_count),
        exact_verification_count=int(exchange.exact_verification_count),
        gram_compute_time_s=float(exchange.gram_compute_time_s),
        exact_lp_time_s=float(exchange.exact_lp_time_s),
        correlation_factor_cache_hits=int(
            exchange.correlation_factor_cache_hits),
        correlation_factor_cache_misses=int(
            exchange.correlation_factor_cache_misses),
        dual_bound_early_stopped=bool(exchange.dual_bound_early_stopped),
        active_structure_unchanged=bool(np.array_equal(
            selected, active_before)),
    )
    if not exchange.accepted:
        return CorrelationExchangeShadowResult(
            **common,
            would_commit=False,
            failure_reason="no_exact_improvement",
            reconstruction_total_time_s=0.0,
            reconstruction_parallel_critical_path_s=0.0,
            canonical_record_bytes=0,
            protocol_over_air_bits=0,
            protocol_latency_s=0.0,
            protocol_energy_j=0.0,
            replica_agreement=False,
        )
    if not dependency_ready:
        return CorrelationExchangeShadowResult(
            **common,
            would_commit=False,
            failure_reason="atomic_dependencies_not_ready",
            reconstruction_total_time_s=0.0,
            reconstruction_parallel_critical_path_s=0.0,
            canonical_record_bytes=0,
            protocol_over_air_bits=0,
            protocol_latency_s=0.0,
            protocol_energy_j=0.0,
            replica_agreement=False,
        )

    move = LocalMove(
        kind=str(exchange.accepted_kind),
        selected=exchange.selected.copy(),
        role=exchange.role.copy(),
        owner=exchange.owner.copy(),
    )
    records: list[CorrelationCandidateRecord] = []
    rebuild_times: list[float] = []
    for _viewer in range(active_before.shape[0]):
        started = perf_counter()
        records.append(build_correlation_candidate_record(
            move,
            coefficient_per_watt,
            sensing_budget_w,
            delay_s,
            doppler_hz,
            generation_id=int(generation_id),
            dependency_versions=dependency_versions,
            delay_size=int(delay_size),
            doppler_size=int(doppler_size),
            delta_f_hz=float(delta_f_hz),
            symbol_time_s=float(symbol_time_s),
            minimum_deflection=minimum_deflection,
        ))
        rebuild_times.append(perf_counter() - started)
    commit = certify_correlation_candidate_commit(
        active_before,
        role,
        owner,
        move,
        replica_records=records,
        positions=positions,
        comm_power_w=comm_power_w,
        communication_model=communication_model,
        total_power_w=float(total_power_w),
        total_deadline_s=remaining_deadline_s,
    )
    failure = None if commit.feasible else ";".join(commit.reasons)
    return CorrelationExchangeShadowResult(
        **common,
        would_commit=bool(commit.feasible),
        failure_reason=failure,
        reconstruction_total_time_s=float(sum(rebuild_times)),
        reconstruction_parallel_critical_path_s=float(max(
            rebuild_times, default=0.0)),
        canonical_record_bytes=int(
            commit.reconstruction.canonical_bytes),
        protocol_over_air_bits=int(commit.transport.total_over_air_bits),
        protocol_latency_s=float(commit.transport.total_latency_s),
        protocol_energy_j=float(commit.transport.total_energy_j),
        replica_agreement=bool(commit.reconstruction.feasible),
    )
