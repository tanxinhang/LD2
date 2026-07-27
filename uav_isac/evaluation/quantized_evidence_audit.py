"""Offline Gate 1b audit for quantized structured receiver evidence."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from uav_isac.physical.evidence import (
    EvidencePacketLayout,
    local_quality_topk_mask,
    quantize_llr,
)
from uav_isac.environment.communication import InterUAVCommunicationModel
from uav_isac.utils.math_utils import compute_PD


def _apply_content_control(
    quantized_llr: np.ndarray,
    mode: str,
    null_reference: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply a content-only intervention without changing packet transport."""
    normalized = str(mode).strip().lower()
    if normalized == "normal":
        return quantized_llr
    if normalized == "zero":
        return np.zeros_like(quantized_llr)
    if normalized == "value_roll":
        # Keep sender, packet count, target IDs, delivery, and bit rate fixed,
        # but attach each selected target ID to another target's LLR value.
        # Under target-specific H1_q, every other target remains under H0;
        # therefore an H1 detection simulation must roll null evidence rather
        # than unrealistically treating all targets as simultaneously present.
        source = (
            quantized_llr if null_reference is None
            else np.asarray(null_reference, dtype=np.float64))
        if source.shape != quantized_llr.shape:
            raise ValueError(
                "null_reference must match quantized_llr shape")
        return np.roll(source, shift=1, axis=-1)
    raise ValueError(
        "content_mode must be one of normal/zero/value_roll")


def evidence_inclusion(
    receiver_deflection: np.ndarray,
    topk: int,
    owner_aware: bool = False,
    delivery_matrix: Optional[np.ndarray] = None,
    peer_deflection_estimate: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Build pre-evidence owner and peer-routing masks."""
    receiver_d = np.asarray(receiver_deflection, dtype=np.float64)
    if receiver_d.ndim != 3:
        raise ValueError("receiver_deflection must have shape (F, K, Q)")
    frames, agents, targets = receiver_d.shape
    owner = np.argmax(receiver_d, axis=1).astype(np.int64)
    owner_mask = np.zeros(
        (frames, agents, targets), dtype=bool)
    frame_index = np.arange(frames)[:, None]
    target_index = np.arange(targets)[None, :]
    owner_mask[frame_index, owner, target_index] = True
    selection_quality = receiver_d
    if owner_aware:
        selection_quality = receiver_d.copy()
        selection_quality[owner_mask] = 0.0
    selected = local_quality_topk_mask(selection_quality, topk)
    peer_mask = selected & ~owner_mask
    if delivery_matrix is not None:
        delivery = np.asarray(delivery_matrix, dtype=bool)
        if delivery.shape != (frames, agents, agents):
            raise ValueError(
                "delivery_matrix must have shape (F, K, K)")
        source_index = np.arange(agents)[None, :, None]
        frame_for_delivery = np.arange(frames)[:, None, None]
        owner_for_delivery = owner[:, None, :]
        delivered_to_owner = delivery[
            frame_for_delivery,
            source_index,
            owner_for_delivery,
        ]
        peer_mask &= delivered_to_owner
    included = owner_mask | peer_mask
    fused_d = np.sum(receiver_d * included, axis=1)
    if peer_deflection_estimate is None:
        estimated_peer_d = receiver_d
    else:
        estimated_peer_d = np.asarray(
            peer_deflection_estimate, dtype=np.float64)
        if estimated_peer_d.shape != receiver_d.shape:
            raise ValueError(
                "peer_deflection_estimate must have shape (F, K, Q)")
        if (
            np.any(~np.isfinite(estimated_peer_d))
            or np.any(estimated_peer_d < 0.0)
        ):
            raise ValueError(
                "peer deflection estimates must be finite and non-negative")
    threshold_d = np.sum(
        np.where(
            owner_mask,
            receiver_d,
            np.where(peer_mask, estimated_peer_d, 0.0),
        ),
        axis=1,
    )
    return {
        "selected_mask": selected,
        "owner": owner,
        "owner_mask": owner_mask,
        "peer_mask": peer_mask,
        "included_mask": included,
        "fused_deflection": fused_d,
        "threshold_deflection": threshold_d,
    }


def calibrate_standardized_threshold(
    receiver_deflection: np.ndarray,
    *,
    topk: int,
    bits: int,
    clip_max: float,
    p_fa: float,
    owner_aware: bool = False,
    delivery_matrix: Optional[np.ndarray] = None,
    peer_deflection_estimate: Optional[np.ndarray] = None,
    content_mode: str = "normal",
    threshold_mode: str = "standardized",
    samples: int = 1_000_000,
    seed: int = 20260724,
) -> Dict[str, float]:
    """Calibrate one method-level standardized H0 threshold.

    The threshold has form ``-D/2 + sqrt(D) * tau``.  Quantization makes the
    standardized null statistic non-Gaussian, so ``tau`` is learned only from
    the independent calibration trace.
    """
    receiver_d = np.asarray(receiver_deflection, dtype=np.float64)
    routing = evidence_inclusion(
        receiver_d,
        topk,
        owner_aware=owner_aware,
        delivery_matrix=delivery_matrix,
        peer_deflection_estimate=peer_deflection_estimate,
    )
    frames, agents, targets = receiver_d.shape
    count = max(1, int(samples))
    rng = np.random.default_rng(int(seed))
    frame_ids = rng.integers(0, frames, size=count)
    target_ids = rng.integers(0, targets, size=count)
    d = receiver_d[frame_ids, :, target_ids]
    owner = routing["owner"][frame_ids, target_ids]
    peer = routing["peer_mask"][frame_ids, :, target_ids]
    included = routing["included_mask"][frame_ids, :, target_ids]

    noise = rng.standard_normal((count, agents))
    local_h0 = -0.5 * d + np.sqrt(d) * noise
    quantized_h0 = quantize_llr(local_h0, bits, clip_max)
    quantized_h0 = _apply_content_control(
        quantized_h0, content_mode)
    owner_values = local_h0[np.arange(count), owner]
    fused_h0 = owner_values + np.sum(
        np.where(peer, quantized_h0, 0.0), axis=1)
    threshold_d_all = routing["threshold_deflection"]
    threshold_d = threshold_d_all[frame_ids, target_ids]
    valid = threshold_d > 0.0
    if not np.any(valid):
        raise ValueError("calibration trace has no included evidence")
    normalized_threshold_mode = str(
        threshold_mode).strip().lower()
    if normalized_threshold_mode == "standardized":
        calibration_statistic = (
            fused_h0[valid] + 0.5 * threshold_d[valid]
        ) / np.sqrt(threshold_d[valid])
    elif normalized_threshold_mode == "global_llr":
        # A single deployable threshold uses only the received LLR sum.  It
        # does not require the owner to know an untransmitted peer deflection.
        calibration_statistic = fused_h0[valid]
    else:
        raise ValueError(
            "threshold_mode must be standardized or global_llr")
    tau = float(np.quantile(
        calibration_statistic, 1.0 - float(p_fa)))
    calibration_pfa = float(np.mean(
        calibration_statistic > tau))
    return {
        "threshold_mode": normalized_threshold_mode,
        "threshold_value": tau,
        "standardized_threshold": tau,
        "calibration_pfa": calibration_pfa,
        "valid_calibration_samples": int(np.sum(valid)),
    }


def simulate_quantized_detection(
    receiver_deflection: np.ndarray,
    *,
    topk: int,
    bits: int,
    clip_max: float,
    standardized_threshold: float,
    p_fa: float,
    owner_aware: bool = False,
    delivery_matrix: Optional[np.ndarray] = None,
    peer_deflection_estimate: Optional[np.ndarray] = None,
    content_mode: str = "normal",
    threshold_mode: str = "standardized",
    draws_per_frame: int = 2048,
    batch_frames: int = 16,
    seed: int = 20260725,
) -> Dict[str, object]:
    """Estimate per-frame P_D and aggregate P_FA on an independent trace."""
    receiver_d = np.asarray(receiver_deflection, dtype=np.float64)
    routing = evidence_inclusion(
        receiver_d,
        topk,
        owner_aware=owner_aware,
        delivery_matrix=delivery_matrix,
        peer_deflection_estimate=peer_deflection_estimate,
    )
    frames, agents, targets = receiver_d.shape
    draws = max(1, int(draws_per_frame))
    batch_width = max(1, int(batch_frames))
    rng = np.random.default_rng(int(seed))
    pd = np.zeros((frames, targets), dtype=np.float64)
    observed_pfa = np.zeros((frames, targets), dtype=np.float64)
    clipped_count = 0
    value_count = 0
    squared_error = 0.0
    sign_flip_count = 0

    for start in range(0, frames, batch_width):
        stop = min(frames, start + batch_width)
        d = receiver_d[start:stop]
        owner_mask = routing["owner_mask"][start:stop]
        peer_mask = routing["peer_mask"][start:stop]
        fused_d = routing["fused_deflection"][start:stop]
        threshold_d = routing[
            "threshold_deflection"][start:stop]
        shape = (draws,) + d.shape
        noise_h0 = rng.standard_normal(shape)
        noise_h1 = rng.standard_normal(shape)
        local_h0 = -0.5 * d[None] + np.sqrt(d[None]) * noise_h0
        local_h1 = +0.5 * d[None] + np.sqrt(d[None]) * noise_h1
        quantized_h0 = quantize_llr(local_h0, bits, clip_max)
        quantized_h1 = quantize_llr(local_h1, bits, clip_max)
        quantized_h0 = _apply_content_control(
            quantized_h0, content_mode)
        quantized_h1 = _apply_content_control(
            quantized_h1,
            content_mode,
            null_reference=quantize_llr(
                local_h0, bits, clip_max),
        )
        owner_broadcast = owner_mask[None]
        peer_broadcast = peer_mask[None]
        fused_h0 = np.sum(np.where(
            owner_broadcast,
            local_h0,
            np.where(peer_broadcast, quantized_h0, 0.0),
        ), axis=2)
        fused_h1 = np.sum(np.where(
            owner_broadcast,
            local_h1,
            np.where(peer_broadcast, quantized_h1, 0.0),
        ), axis=2)
        normalized_threshold_mode = str(
            threshold_mode).strip().lower()
        if normalized_threshold_mode == "standardized":
            threshold = (
                -0.5 * threshold_d
                + np.sqrt(threshold_d) * float(standardized_threshold)
            )
        elif normalized_threshold_mode == "global_llr":
            threshold = np.full_like(
                fused_d, float(standardized_threshold))
        else:
            raise ValueError(
                "threshold_mode must be standardized or global_llr")
        positive = threshold_d > 0.0
        batch_pfa = np.mean(
            fused_h0 > threshold[None], axis=0)
        batch_pd = np.mean(
            fused_h1 > threshold[None], axis=0)
        observed_pfa[start:stop] = np.where(
            positive, batch_pfa, float(p_fa))
        pd[start:stop] = np.where(
            positive, batch_pd, float(p_fa))

        transmitted = np.broadcast_to(peer_broadcast, local_h0.shape)
        for raw, quantized in (
                (local_h0, quantized_h0), (local_h1, quantized_h1)):
            chosen_raw = raw[transmitted]
            chosen_quantized = quantized[transmitted]
            if chosen_raw.size == 0:
                continue
            clipped_count += int(np.sum(
                np.abs(chosen_raw) > float(clip_max)))
            value_count += int(chosen_raw.size)
            squared_error += float(np.sum(
                (chosen_quantized - chosen_raw) ** 2))
            nonzero = np.abs(chosen_raw) > 1e-15
            sign_flip_count += int(np.sum(
                np.signbit(chosen_raw[nonzero])
                != np.signbit(chosen_quantized[nonzero])))

    return {
        "pd": pd,
        "pfa_by_frame_target": observed_pfa,
        "aggregate_pfa": float(np.mean(observed_pfa)),
        "clip_rate": float(clipped_count / max(value_count, 1)),
        "quantization_mse": float(
            squared_error / max(value_count, 1)),
        "sign_flip_rate": float(
            sign_flip_count / max(value_count, 1)),
        "transmitted_value_count": int(value_count),
        "fused_deflection": routing["fused_deflection"],
        "selected_mask": routing["selected_mask"],
    }


def episode_detection_metrics(
    pd_history: np.ndarray,
    episode_index: np.ndarray,
    steady_window: int = 20,
) -> np.ndarray:
    """Return per-episode [steady, weak3, worst] metrics."""
    pd = np.asarray(pd_history, dtype=np.float64)
    episodes = np.asarray(episode_index, dtype=np.int64)
    if pd.ndim != 2 or episodes.shape != (pd.shape[0],):
        raise ValueError("P_D history and episode indices must align")
    rows = []
    for episode in np.unique(episodes):
        history = pd[episodes == episode]
        width = min(max(1, int(steady_window)), len(history))
        per_target = np.mean(history[-width:], axis=0)
        ordered = np.sort(per_target)
        rows.append([
            float(np.mean(per_target)),
            float(np.mean(ordered[:min(3, ordered.size)])),
            float(ordered[0]),
        ])
    return np.asarray(rows, dtype=np.float64)


def summarize_quantized_method(
    method_pd: np.ndarray,
    local_pd: np.ndarray,
    central_pd: np.ndarray,
    episode_index: np.ndarray,
    *,
    steady_window: int = 20,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260726,
) -> Dict[str, object]:
    """Summarize paired recovery of central evidence gain."""
    method_metrics = episode_detection_metrics(
        method_pd, episode_index, steady_window)
    local_metrics = episode_detection_metrics(
        local_pd, episode_index, steady_window)
    central_metrics = episode_detection_metrics(
        central_pd, episode_index, steady_window)
    if not (
        method_metrics.shape == local_metrics.shape == central_metrics.shape
    ):
        raise ValueError("method/local/central episode metrics must align")
    names = ("steady", "weak3", "worst")
    rng = np.random.default_rng(int(bootstrap_seed))
    episode_count = len(method_metrics)
    indices = rng.integers(
        0, episode_count,
        size=(max(1, int(bootstrap_samples)), episode_count),
    )
    result: Dict[str, object] = {
        "episode_count": int(episode_count),
    }
    for column, name in enumerate(names):
        method_mean = float(np.mean(method_metrics[:, column]))
        local_mean = float(np.mean(local_metrics[:, column]))
        central_mean = float(np.mean(central_metrics[:, column]))
        delta = method_metrics[:, column] - local_metrics[:, column]
        oracle = central_metrics[:, column] - local_metrics[:, column]
        recovery = (
            (method_mean - local_mean) / (central_mean - local_mean)
            if central_mean - local_mean > 1e-12 else 0.0)
        boot_delta = np.mean(delta[indices], axis=1)
        boot_oracle = np.mean(oracle[indices], axis=1)
        boot_recovery = np.divide(
            boot_delta,
            boot_oracle,
            out=np.zeros_like(boot_delta),
            where=boot_oracle > 1e-12,
        )
        result[name] = method_mean
        result[f"{name}_delta_vs_local"] = float(np.mean(delta))
        result[f"{name}_delta_vs_local_ci95"] = [
            float(np.quantile(boot_delta, 0.025)),
            float(np.quantile(boot_delta, 0.975)),
        ]
        result[f"{name}_recovery"] = float(recovery)
        result[f"{name}_recovery_ci95"] = [
            float(np.quantile(boot_recovery, 0.025)),
            float(np.quantile(boot_recovery, 0.975)),
        ]
    return result


def summarize_paired_detection_delta(
    first_pd: np.ndarray,
    second_pd: np.ndarray,
    episode_index: np.ndarray,
    *,
    steady_window: int = 20,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260727,
) -> Dict[str, object]:
    """Paired episode delta of one evidence method over another."""
    first = episode_detection_metrics(
        first_pd, episode_index, steady_window)
    second = episode_detection_metrics(
        second_pd, episode_index, steady_window)
    if first.shape != second.shape:
        raise ValueError("paired method metrics must align")
    delta = first - second
    rng = np.random.default_rng(int(bootstrap_seed))
    episode_count = len(delta)
    indices = rng.integers(
        0, episode_count,
        size=(max(1, int(bootstrap_samples)), episode_count),
    )
    result: Dict[str, object] = {
        "episode_count": int(episode_count),
    }
    for column, name in enumerate(("steady", "weak3", "worst")):
        bootstrap_delta = np.mean(
            delta[indices, column], axis=1)
        result[f"{name}_delta"] = float(np.mean(delta[:, column]))
        result[f"{name}_delta_ci95"] = [
            float(np.quantile(bootstrap_delta, 0.025)),
            float(np.quantile(bootstrap_delta, 0.975)),
        ]
    return result


def mean_structured_bits_per_frame(
    selected_mask: np.ndarray,
    layout: EvidencePacketLayout,
    llr_bits: int,
) -> Tuple[float, float]:
    """Return mean team bits/frame and mean active broadcasts/frame."""
    selected = np.asarray(selected_mask, dtype=bool)
    if selected.ndim != 3:
        raise ValueError("selected_mask must have shape (F, K, Q)")
    entries = np.sum(selected, axis=-1)
    per_sender = np.vectorize(
        lambda count: layout.broadcast_bits(int(count), llr_bits),
        otypes=[np.int64],
    )(entries)
    return (
        float(np.mean(np.sum(per_sender, axis=1))),
        float(np.mean(np.sum(entries > 0, axis=1))),
    )


def structured_broadcast_transport(
    positions: np.ndarray,
    comm_power_w: np.ndarray,
    selected_mask: np.ndarray,
    layout: EvidencePacketLayout,
    llr_bits: int,
    model: InterUAVCommunicationModel,
) -> Dict[str, object]:
    """Evaluate structured broadcasts through the existing physical U2U link."""
    xyz = np.asarray(positions, dtype=np.float64)
    power = np.asarray(comm_power_w, dtype=np.float64)
    selected = np.asarray(selected_mask, dtype=bool)
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError("positions must have shape (F, K, 3)")
    frames, agents, _ = xyz.shape
    if power.shape != (frames, agents):
        raise ValueError("comm_power_w must have shape (F, K)")
    if selected.shape[:2] != (frames, agents):
        raise ValueError("selected_mask must have shape (F, K, Q)")

    entries = np.sum(selected, axis=-1)
    payload_bits = np.vectorize(
        lambda count: layout.broadcast_bits(int(count), llr_bits),
        otypes=[np.int64],
    )(entries)
    delivery = np.zeros((frames, agents, agents), dtype=bool)
    latency = []
    attempted = delivered = expired = 0
    energy_j = 0.0
    for frame in range(frames):
        active = np.flatnonzero(payload_bits[frame] > 0)
        if active.size == 0:
            continue
        effective_bandwidth = model.bandwidth_hz / active.size
        for sender in active:
            sender_airtime = 0.0
            for receiver in range(agents):
                if receiver == sender:
                    continue
                snr_db, _rate, serialization_s, latency_s = model._link(
                    xyz[frame, sender],
                    xyz[frame, receiver],
                    int(payload_bits[frame, sender]),
                    effective_bandwidth,
                    float(power[frame, sender]),
                )
                attempted += 1
                latency.append(float(latency_s))
                sender_airtime = max(sender_airtime, float(serialization_s))
                success = bool(
                    snr_db >= model.snr_threshold_db
                    and latency_s <= model.deadline_s
                )
                delivery[frame, sender, receiver] = success
                if success:
                    delivered += 1
                else:
                    expired += 1
            energy_j += float(power[frame, sender]) * sender_airtime

    latency_array = np.asarray(latency, dtype=np.float64)
    return {
        "delivery_matrix": delivery,
        "payload_bits": payload_bits,
        "attempted_links": int(attempted),
        "delivered_links": int(delivered),
        "expired_links": int(expired),
        "delivery_rate": float(delivered / max(attempted, 1)),
        "deadline_violation_rate": float(expired / max(attempted, 1)),
        "mean_latency_s": float(np.mean(
            latency_array)) if latency_array.size else 0.0,
        "p95_latency_s": float(np.percentile(
            latency_array, 95)) if latency_array.size else 0.0,
        "max_latency_s": float(np.max(
            latency_array)) if latency_array.size else 0.0,
        "energy_j_per_frame": float(energy_j / max(frames, 1)),
        "bits_per_frame": float(np.mean(np.sum(payload_bits, axis=1))),
        "active_senders_per_frame": float(np.mean(
            np.sum(payload_bits > 0, axis=1))),
    }


def analytic_trace_bounds(
    receiver_deflection: np.ndarray,
    p_fa: float,
) -> Dict[str, np.ndarray]:
    """Analytic best-local and central P_D histories for the same trace."""
    receiver_d = np.asarray(receiver_deflection, dtype=np.float64)
    local_d = np.max(receiver_d, axis=1)
    central_d = np.sum(receiver_d, axis=1)
    return {
        "local_pd": compute_PD(local_d, p_fa),
        "central_pd": compute_PD(central_d, p_fa),
    }
