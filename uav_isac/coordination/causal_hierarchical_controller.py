"""Causal coefficient envelopes for the certified hierarchical controller.

Only lagged, actually excited sensing edges may refresh the target invariant.
The invariant is directionally quantized and physically transported before a
one-step inverse-range/OTFS-DD envelope is constructed.  A frozen episode-level
split-conformal log margin covers the remaining simultaneous model residual.
Future coefficients never enter ranking, routing, or commit decisions.
"""

# ----------------------------------------------------------------------
# AUDIT/RESEARCH-ONLY MODULE (2026-08-16 audit remediation)
#
# This module is consumed only by tools/ audit scripts and tests. It is
# NOT part of the deployment execution path (env_core / trainer) and its
# results must not be described as deployed behaviour. It exists to keep
# a specific research question reproducible; see
# docs/EXPERIMENT_LOG.md for the associated gate.
# ----------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from uav_isac.coordination.certified_hierarchical_controller import (
    CertifiedHierarchicalControllerConfig,
    CertifiedHierarchicalDecision,
    certified_hierarchical_isac_control,
)
from uav_isac.coordination.owner_local_physics import (
    OwnerLocalHorizonCoefficientBounds,
    OwnerLocalKinematicState,
    OwnerTargetInvariantCache,
    owner_local_horizon_coefficient_bounds,
    update_target_invariant_cache,
)
from uav_isac.coordination.target_invariant_transport import (
    TargetInvariantToken,
    TargetInvariantTransportCertificate,
    TargetInvariantWireLayout,
    certify_target_invariant_transport,
    decode_owner_target_invariant_tokens,
    encode_owner_target_invariant_tokens,
)
from uav_isac.domain.communication import CommunicationTransport


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenCausalEnvelopeCalibration:
    alpha: float
    coverage_floor: float
    calibration_episode_count: int
    base_current_log_margin: float
    transition_residual_log_margin: float
    horizon_steps: int
    source_paths: tuple[str, ...]
    source_sha256: tuple[str, ...]
    envelope_calibration_ready: bool
    system_certificate_ready: bool
    observability_mode: str = "unspecified"
    target_invariant_token_transport_enabled: bool = False
    action_aligned_owner_state_enabled: bool = False
    target_invariant_cache_max_age_frames: int = 0
    dd_covariance_radius: float = 0.0
    dd_additive_margin: float = 0.0
    calibrated_support_policy: str = "unspecified"
    calibration_episode_ids: tuple[int, ...] = ()
    source_configs: tuple[str, ...] = ()

    @property
    def current_log_margin(self) -> float:
        return float(
            self.base_current_log_margin
            + self.transition_residual_log_margin)

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, object],
    ) -> "FrozenCausalEnvelopeCalibration":
        if payload.get("method") != (
            "episode_level_split_conformal_simultaneous_log_envelope"
        ):
            raise ValueError("unsupported causal envelope calibration method")
        design = payload.get("horizon_design")
        if not isinstance(design, Mapping):
            raise ValueError("calibration lacks a frozen horizon design")
        source_items = payload.get("sources", ())
        if not isinstance(source_items, list) or not source_items:
            raise ValueError("calibration lacks immutable source records")
        paths = []
        hashes = []
        for item in source_items:
            if not isinstance(item, Mapping):
                raise ValueError("calibration source record is invalid")
            paths.append(str(item.get("path", "")))
            hashes.append(str(item.get("sha256", "")))
        values = np.asarray([
            payload.get("alpha", np.nan),
            payload.get("finite_sample_coverage_floor", np.nan),
            design.get("base_current_log_margin", np.nan),
            payload.get("frozen_transition_residual_log_margin", np.nan),
        ], dtype=np.float64)
        if (
            np.any(~np.isfinite(values)) or not 0.0 < values[0] < 1.0
            or not 0.0 < values[1] <= 1.0 or np.any(values[2:] < 0.0)
        ):
            raise ValueError("causal envelope calibration scalars are invalid")
        episode_count = int(payload.get("calibration_episode_count", 0))
        horizon_steps = int(design.get("steps", 0))
        ready = bool(payload.get("envelope_calibration_ready", False))
        target = float(payload.get("target_coverage", 1.0 - values[0]))
        if (
            episode_count < 1 or horizon_steps < 1 or not ready
            or values[1] + 1.0e-15 < target
            or any(not path or len(digest) != 64
                   for path, digest in zip(paths, hashes))
        ):
            raise ValueError("causal envelope calibration is not frozen-ready")
        return cls(
            alpha=float(values[0]),
            coverage_floor=float(values[1]),
            calibration_episode_count=episode_count,
            base_current_log_margin=float(values[2]),
            transition_residual_log_margin=float(values[3]),
            horizon_steps=horizon_steps,
            source_paths=tuple(paths),
            source_sha256=tuple(hashes),
            envelope_calibration_ready=ready,
            system_certificate_ready=bool(
                payload.get("system_certificate_ready", False)),
        )

    @classmethod
    def from_json(
        cls,
        path: Path,
        *,
        source_root: Path | None = None,
        verify_sources: bool = True,
    ) -> "FrozenCausalEnvelopeCalibration":
        calibration_path = Path(path)
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        result = cls.from_mapping(payload)
        if verify_sources:
            root = Path.cwd() if source_root is None else Path(source_root)
            source_metadata = []
            calibration_episode_ids: list[int] = []
            source_configs: list[str] = []
            for raw_path, expected in zip(
                result.source_paths, result.source_sha256
            ):
                source = Path(raw_path)
                if not source.is_absolute():
                    source = root / source
                if not source.is_file() or _sha256(source) != expected:
                    raise ValueError(
                        "causal envelope calibration source hash mismatch")
                source_payload = json.loads(source.read_text(encoding="utf-8"))
                seed_order = [
                    int(seed) for seed in source_payload.get("seed_order", ())
                ]
                if (
                    not seed_order or len(seed_order) != len(set(seed_order))
                    or set(seed_order) & set(calibration_episode_ids)
                ):
                    raise ValueError(
                        "causal calibration episode identities are invalid")
                calibration_episode_ids.extend(seed_order)
                source_config = str(source_payload.get("config", ""))
                if not source_config:
                    raise ValueError("causal calibration source lacks config")
                source_configs.append(source_config)
                diagnostic = source_payload.get(
                    "horizon_future_calibration_diagnostic", {})
                metadata = (
                    int(source_payload.get("schema_version", 0)),
                    str(source_payload.get("observability_mode", "")),
                    bool(source_payload.get(
                        "target_invariant_token_transport_enabled", False)),
                    bool(source_payload.get(
                        "action_aligned_owner_state_enabled", False)),
                    int(source_payload.get(
                        "target_invariant_cache_max_age_frames", 0)),
                    float(source_payload.get(
                        "owner_local_dd_covariance_radius", np.nan)),
                    float(source_payload.get(
                        "frozen_owner_local_dd_additive_margin", np.nan)),
                    float(source_payload.get(
                        "frozen_unknown_edge_lower_log_margin", np.nan)),
                    float(source_payload.get(
                        "frozen_unknown_edge_upper_log_margin", np.nan)),
                    str(diagnostic.get("score", "")),
                    bool(diagnostic.get(
                        "future_information_used_by_controller", True)),
                )
                source_metadata.append(metadata)
            if len(set(source_metadata)) != 1:
                raise ValueError("causal calibration sources changed design")
            metadata = source_metadata[0]
            if (
                metadata[0] < 6
                or metadata[1] != "owner_local_conformal"
                or not metadata[2] or not metadata[3]
                or metadata[4] < 1
                or not np.isfinite(metadata[5]) or metadata[5] < 0.0
                or not np.isfinite(metadata[6]) or metadata[6] < 0.0
                or not np.isclose(
                    max(metadata[7], metadata[8]),
                    result.base_current_log_margin,
                    rtol=0.0,
                    atol=1.0e-15,
                )
                or "current-provenance edges" not in metadata[9]
                or metadata[10]
            ):
                raise ValueError(
                    "causal calibration source domain is incompatible")
            result = replace(
                result,
                observability_mode=metadata[1],
                target_invariant_token_transport_enabled=metadata[2],
                action_aligned_owner_state_enabled=metadata[3],
                target_invariant_cache_max_age_frames=metadata[4],
                dd_covariance_radius=metadata[5],
                dd_additive_margin=metadata[6],
                calibrated_support_policy=(
                    "reciprocal_target_token_edges_or_excited_persistent_edges"
                ),
                calibration_episode_ids=tuple(calibration_episode_ids),
                source_configs=tuple(source_configs),
            )
        return result


@dataclass(frozen=True)
class CausalCoefficientEnvelope:
    available: bool
    reason: str
    lower: np.ndarray
    upper: np.ndarray
    target_invariant_cache: OwnerTargetInvariantCache
    tokens: tuple[TargetInvariantToken, ...]
    transport: TargetInvariantTransportCertificate
    physical_bounds: OwnerLocalHorizonCoefficientBounds | None
    calibration: FrozenCausalEnvelopeCalibration


@dataclass(frozen=True)
class CausalHierarchicalControlDecision:
    accepted: bool
    reason: str
    selected: np.ndarray
    role: np.ndarray
    sensing_power_w: np.ndarray
    communication_power_w: np.ndarray
    envelope: CausalCoefficientEnvelope
    controller: CertifiedHierarchicalDecision | None
    total_over_air_bits: int
    total_protocol_latency_s: float
    total_energy_j: float
    deployment_certificate: bool


def _empty_transport(
    communication_power_w: np.ndarray,
) -> TargetInvariantTransportCertificate:
    comm = np.asarray(communication_power_w, dtype=np.float64).reshape(-1)
    zeros = np.zeros_like(comm)
    return TargetInvariantTransportCertificate(
        feasible=True,
        reasons=tuple(),
        packet_count=0,
        record_count=0,
        total_over_air_bits=0,
        max_packet_latency_s=0.0,
        total_protocol_latency_s=0.0,
        total_energy_j=0.0,
        min_snr_db=float("inf"),
        required_comm_power_w=zeros,
        projected_comm_power_w=comm.copy(),
    )


def build_causal_coefficient_envelope(
    previous_coefficient: np.ndarray,
    previous_observed_mask: np.ndarray,
    previous_state: OwnerLocalKinematicState,
    current_state: OwnerLocalKinematicState,
    target_owner: np.ndarray,
    current_support: np.ndarray,
    positions: np.ndarray,
    existing_communication_power_w: np.ndarray,
    *,
    current_frame: int,
    elapsed_frames: int,
    max_age_frames: int,
    calibration: FrozenCausalEnvelopeCalibration,
    communication_model: CommunicationTransport,
    carrier_hz: float,
    delta_f_hz: float,
    symbol_period_s: float,
    delay_bins: int,
    doppler_bins: int,
    dd_support_threshold: float,
    covariance_radius: float = 0.0,
    dd_additive_margin: float = 0.0,
    snr_margin_db: float = 3.0,
    latency_margin_s: float = 5.0e-4,
    prior_cache: OwnerTargetInvariantCache | None = None,
    tracking_enabled: bool = False,
    use_swerling: bool = False,
    invariant_layout: TargetInvariantWireLayout | None = None,
) -> CausalCoefficientEnvelope:
    """Build one current-event envelope without current physical outcomes."""
    previous = np.asarray(previous_coefficient, dtype=np.float64)
    observed = np.asarray(previous_observed_mask, dtype=bool)
    support = np.asarray(current_support, dtype=bool)
    comm = np.asarray(
        existing_communication_power_w, dtype=np.float64).reshape(-1)
    if previous.ndim != 3 or previous.shape[0] != previous.shape[1]:
        raise ValueError("previous_coefficient must have shape (K,K,Q)")
    K, _, Q = previous.shape
    if observed.shape != previous.shape or support.shape != previous.shape:
        raise ValueError("observed/current support must match coefficients")
    if bool(tracking_enabled) or bool(use_swerling):
        raise ValueError(
            "this frozen calibration supports static non-Swerling targets only")
    if not calibration.envelope_calibration_ready:
        raise ValueError("causal envelope calibration is not ready")
    if (
        calibration.observability_mode not in (
            "owner_local_conformal", "synthetic_test")
        or not calibration.target_invariant_token_transport_enabled
        or not calibration.action_aligned_owner_state_enabled
        or int(max_age_frames)
        != int(calibration.target_invariant_cache_max_age_frames)
        or not np.isclose(
            float(covariance_radius), calibration.dd_covariance_radius,
            rtol=0.0, atol=1.0e-15)
        or not np.isclose(
            float(dd_additive_margin), calibration.dd_additive_margin,
            rtol=0.0, atol=1.0e-15)
    ):
        raise ValueError("runtime causal envelope design differs from calibration")
    layout = invariant_layout or TargetInvariantWireLayout(K, Q)
    if layout.num_agents != K or layout.num_targets != Q:
        raise ValueError("target invariant wire dimensions do not match")
    cache = update_target_invariant_cache(
        previous,
        observed,
        previous_state,
        prior_cache=prior_cache,
        elapsed_frames=int(elapsed_frames),
        max_age_frames=int(max_age_frames),
    )
    owner = np.asarray(target_owner, dtype=np.int64).reshape(-1)
    tokens = encode_owner_target_invariant_tokens(
        cache, owner, layout=layout, sent_frame=int(current_frame))
    transport = certify_target_invariant_transport(
        tokens,
        positions=np.asarray(positions, dtype=np.float64),
        existing_comm_power_w=comm,
        communication_model=communication_model,
        layout=layout,
        snr_margin_db=float(snr_margin_db),
        latency_margin_s=float(latency_margin_s),
    )
    decoded = decode_owner_target_invariant_tokens(
        tokens,
        owner,
        layout=layout,
        current_frame=int(current_frame),
        max_age_frames=int(max_age_frames),
    )
    empty = np.zeros_like(previous)
    if not transport.feasible:
        return CausalCoefficientEnvelope(
            available=False,
            reason="target_invariant_transport_failed",
            lower=empty,
            upper=empty,
            target_invariant_cache=decoded,
            tokens=tokens,
            transport=transport,
            physical_bounds=None,
            calibration=calibration,
        )
    if np.any(decoded.target_invariant <= 0.0):
        return CausalCoefficientEnvelope(
            available=False,
            reason="incomplete_target_invariant",
            lower=empty,
            upper=empty,
            target_invariant_cache=decoded,
            tokens=tokens,
            transport=transport,
            physical_bounds=None,
            calibration=calibration,
        )
    invariant_upper = np.asarray([
        layout.invariant_cell_upper(value)
        for value in decoded.target_invariant
    ], dtype=np.float64)
    bounds = owner_local_horizon_coefficient_bounds(
        (current_state,),
        decoded.target_invariant,
        invariant_upper,
        support,
        carrier_hz=float(carrier_hz),
        delta_f_hz=float(delta_f_hz),
        symbol_period_s=float(symbol_period_s),
        delay_bins=int(delay_bins),
        doppler_bins=int(doppler_bins),
        covariance_radius=float(covariance_radius),
        dd_support_threshold=float(dd_support_threshold),
        dd_additive_margin=float(dd_additive_margin),
        residual_log_margin=float(calibration.current_log_margin),
    )
    return CausalCoefficientEnvelope(
        available=True,
        reason="available",
        lower=bounds.lower[0].copy(),
        upper=bounds.upper[0].copy(),
        target_invariant_cache=decoded,
        tokens=tokens,
        transport=transport,
        physical_bounds=bounds,
        calibration=calibration,
    )


def certified_causal_hierarchical_isac_control(
    previous_coefficient: np.ndarray,
    previous_observed_mask: np.ndarray,
    previous_state: OwnerLocalKinematicState,
    current_state: OwnerLocalKinematicState,
    initial_selected: np.ndarray,
    initial_role: np.ndarray,
    positions: np.ndarray,
    existing_communication_power_w: np.ndarray,
    existing_sensing_power_w: np.ndarray,
    current_support: np.ndarray,
    *,
    current_frame: int,
    elapsed_frames: int,
    max_age_frames: int,
    calibration: FrozenCausalEnvelopeCalibration,
    communication_model: CommunicationTransport,
    control_period_s: float,
    p_fa: float,
    controller_config: CertifiedHierarchicalControllerConfig,
    carrier_hz: float,
    delta_f_hz: float,
    symbol_period_s: float,
    delay_bins: int,
    doppler_bins: int,
    dd_support_threshold: float,
    tracking_enabled: bool = False,
    use_swerling: bool = False,
    prior_cache: OwnerTargetInvariantCache | None = None,
) -> CausalHierarchicalControlDecision:
    """Run D0.47 from a lagged causal envelope and one explicit prefix."""
    selected = np.asarray(initial_selected, dtype=bool)
    role = np.asarray(initial_role, dtype=np.int8).reshape(-1)
    comm = np.asarray(
        existing_communication_power_w, dtype=np.float64).reshape(-1)
    sensing = np.asarray(existing_sensing_power_w, dtype=np.float64)
    from uav_isac.coordination.local_exchange_oracle import (
        role_owner_from_structure,
    )
    inferred_role, owner = role_owner_from_structure(
        selected, fallback_role=role)
    if not np.array_equal(inferred_role, role) or np.any(owner < 0):
        raise ValueError("initial causal structure is inconsistent")
    envelope = build_causal_coefficient_envelope(
        previous_coefficient,
        previous_observed_mask,
        previous_state,
        current_state,
        owner,
        current_support,
        positions,
        comm,
        current_frame=int(current_frame),
        elapsed_frames=int(elapsed_frames),
        max_age_frames=int(max_age_frames),
        calibration=calibration,
        communication_model=communication_model,
        carrier_hz=float(carrier_hz),
        delta_f_hz=float(delta_f_hz),
        symbol_period_s=float(symbol_period_s),
        delay_bins=int(delay_bins),
        doppler_bins=int(doppler_bins),
        dd_support_threshold=float(dd_support_threshold),
        snr_margin_db=float(controller_config.snr_margin_db),
        latency_margin_s=float(controller_config.latency_margin_s),
        prior_cache=prior_cache,
        tracking_enabled=bool(tracking_enabled),
        use_swerling=bool(use_swerling),
    )
    prefix = envelope.transport
    prefix_extra = bool(np.any(
        prefix.projected_comm_power_w > comm + 1.0e-12))
    remaining_period = float(
        control_period_s - prefix.total_protocol_latency_s)
    if (
        not envelope.available or remaining_period <= 0.0
        or np.any(selected & ~np.asarray(current_support, dtype=bool))
        or (controller_config.require_pre_reserved_comm_power and prefix_extra)
    ):
        reason = envelope.reason
        if envelope.available and remaining_period <= 0.0:
            reason = "target_invariant_control_period"
        elif envelope.available and np.any(
            selected & ~np.asarray(current_support, dtype=bool)
        ):
            reason = "incomplete_noop_provenance"
        elif envelope.available and prefix_extra:
            reason = "target_invariant_unreserved_comm"
        return CausalHierarchicalControlDecision(
            accepted=False,
            reason=reason,
            selected=selected.copy(),
            role=role.copy(),
            sensing_power_w=sensing.copy(),
            communication_power_w=comm.copy(),
            envelope=envelope,
            controller=None,
            total_over_air_bits=int(prefix.total_over_air_bits),
            total_protocol_latency_s=float(prefix.total_protocol_latency_s),
            total_energy_j=float(prefix.total_energy_j),
            deployment_certificate=False,
        )
    controller = certified_hierarchical_isac_control(
        envelope.lower,
        envelope.upper,
        selected,
        role,
        np.asarray(positions, dtype=np.float64),
        prefix.projected_comm_power_w,
        sensing,
        communication_model=communication_model,
        control_period_s=remaining_period,
        p_fa=float(p_fa),
        config=controller_config,
    )
    resources = controller.protocol_resources
    bits = int(prefix.total_over_air_bits + (
        0 if resources is None else resources.total_over_air_bits))
    latency = float(prefix.total_protocol_latency_s + (
        0.0 if resources is None else resources.total_protocol_latency_s))
    energy = float(prefix.total_energy_j + (
        0.0 if resources is None else resources.total_energy_j))
    return CausalHierarchicalControlDecision(
        accepted=bool(controller.accepted),
        reason=(
            "accepted:" + controller.route.value
            if controller.accepted else controller.reason),
        selected=(
            controller.selected.copy() if controller.accepted
            else selected.copy()),
        role=(controller.role.copy() if controller.accepted else role.copy()),
        sensing_power_w=(
            controller.sensing_power_w.copy() if controller.accepted
            else sensing.copy()),
        communication_power_w=(
            controller.communication_power_w.copy() if controller.accepted
            else comm.copy()),
        envelope=envelope,
        controller=controller,
        total_over_air_bits=bits,
        total_protocol_latency_s=latency,
        total_energy_j=energy,
        deployment_certificate=bool(
            calibration.system_certificate_ready),
    )
