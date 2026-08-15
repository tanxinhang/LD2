"""Event-level conformal margins for physical coordination links.

One reconfiguration event can contain many dependent links, protocol rounds
and retransmission observations.  They are therefore reduced to one joint
nonconformity score before split-conformal calibration.  SNR error (dB) and
latency error (seconds) are normalized only by fixed, pre-registered physical
resolution scales; a maximum, rather than a dimensionally invalid sum, forms
the joint score.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from uav_isac.evaluation.transition_certificate import (
    split_conformal_upper_multiplier,
)


def event_joint_channel_underestimate_score(
    predicted_snr_db: np.ndarray,
    observed_snr_db: np.ndarray,
    predicted_latency_s: np.ndarray,
    observed_latency_s: np.ndarray,
    *,
    snr_resolution_db: float,
    latency_resolution_s: float,
    valid_mask: np.ndarray | None = None,
    delivered_mask: np.ndarray | None = None,
) -> float:
    """Collapse all physical errors in one event to one safety score.

    Positive SNR overestimation and latency underestimation are unsafe.  The
    returned value is

    ``max(0, (pred_snr-obs_snr)/r_snr,
               (obs_latency-pred_latency)/r_delay)``

    over every valid link/round item.  Thus an event with thousands of packets
    still contributes exactly one exchangeable calibration observation.
    """
    predicted_snr = np.asarray(predicted_snr_db, dtype=np.float64)
    observed_snr = np.asarray(observed_snr_db, dtype=np.float64)
    predicted_latency = np.asarray(predicted_latency_s, dtype=np.float64)
    observed_latency = np.asarray(observed_latency_s, dtype=np.float64)
    shape = predicted_snr.shape
    if not (
        observed_snr.shape == shape
        and predicted_latency.shape == shape
        and observed_latency.shape == shape
    ):
        raise ValueError("all channel-error arrays must have the same shape")
    snr_resolution = float(snr_resolution_db)
    latency_resolution = float(latency_resolution_s)
    if not np.isfinite(snr_resolution) or snr_resolution <= 0.0:
        raise ValueError("snr_resolution_db must be finite and positive")
    if not np.isfinite(latency_resolution) or latency_resolution <= 0.0:
        raise ValueError("latency_resolution_s must be finite and positive")
    if valid_mask is None:
        valid = np.ones(shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != shape:
            raise ValueError("valid_mask must match the channel-error arrays")
    if not np.any(valid):
        raise ValueError("event has no required channel observations")
    if delivered_mask is not None:
        delivered = np.asarray(delivered_mask, dtype=bool)
        if delivered.shape != shape:
            raise ValueError("delivered_mask must match channel-error arrays")
        if np.any(valid & ~delivered):
            return float("inf")
    finite = (
        np.isfinite(predicted_snr)
        & np.isfinite(observed_snr)
        & np.isfinite(predicted_latency)
        & np.isfinite(observed_latency)
    )
    if np.any(valid & ~finite):
        return float("inf")
    snr_score = np.max(
        (predicted_snr[valid] - observed_snr[valid]) / snr_resolution)
    latency_score = np.max(
        (observed_latency[valid] - predicted_latency[valid])
        / latency_resolution)
    return float(max(0.0, float(snr_score), float(latency_score)))


def event_joint_transport_residual_score(
    predicted_snr_db: np.ndarray,
    observed_snr_db: np.ndarray,
    observed_latency_s: np.ndarray,
    payload_bits: np.ndarray,
    effective_bandwidth_hz: np.ndarray,
    *,
    processing_delay_s: np.ndarray | float,
    predicted_excess_latency_s: np.ndarray | float = 0.0,
    snr_resolution_db: float,
    excess_latency_resolution_s: float,
    valid_mask: np.ndarray | None = None,
    delivered_mask: np.ndarray | None = None,
) -> float:
    """Joint score with Shannon serialization removed from delay residual.

    A lower observed SNR already increases ``bits / R(SNR)``.  Calibrating the
    raw total-latency error and then lowering SNR again in the robust link
    budget would count that same fading realization twice.  This preferred
    score instead defines

    ``d_excess = d_observed - bits/R(SNR_observed) - d_processing``

    and calibrates only the unmodelled queue/scheduling/processing excess.
    The robust transport later combines the lower-SNR Shannon serialization
    with the conformal upper bound on this orthogonal excess component.
    """
    arrays = np.broadcast_arrays(
        np.asarray(predicted_snr_db, dtype=np.float64),
        np.asarray(observed_snr_db, dtype=np.float64),
        np.asarray(observed_latency_s, dtype=np.float64),
        np.asarray(payload_bits, dtype=np.float64),
        np.asarray(effective_bandwidth_hz, dtype=np.float64),
        np.asarray(processing_delay_s, dtype=np.float64),
        np.asarray(predicted_excess_latency_s, dtype=np.float64),
    )
    (
        predicted_snr,
        observed_snr,
        observed_latency,
        bits,
        bandwidth,
        processing,
        predicted_excess,
    ) = arrays
    shape = predicted_snr.shape
    required = (
        np.ones(shape, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if required.shape != shape:
        raise ValueError("valid_mask must match broadcast transport shape")
    if not np.any(required):
        raise ValueError("event has no required transport observations")
    if delivered_mask is not None:
        delivered = np.asarray(delivered_mask, dtype=bool)
        if delivered.shape != shape:
            raise ValueError("delivered_mask must match transport shape")
        if np.any(required & ~delivered):
            return float("inf")
    finite = np.logical_and.reduce(tuple(np.isfinite(value) for value in arrays))
    if np.any(required & ~finite):
        return float("inf")
    if np.any(required & (bits < 0.0)):
        raise ValueError("payload_bits must be non-negative")
    if np.any(required & (bandwidth <= 0.0)):
        raise ValueError("effective bandwidth must be positive")
    if np.any(required & (processing < 0.0)):
        raise ValueError("processing delay must be non-negative")
    snr_linear = np.power(
        10.0, np.clip(observed_snr / 10.0, -300.0, 300.0))
    rate = bandwidth * np.log2(1.0 + snr_linear)
    serialization = np.divide(
        bits,
        rate,
        out=np.full(shape, np.inf, dtype=np.float64),
        where=rate > 0.0,
    )
    serialization = np.where(bits == 0.0, 0.0, serialization)
    observed_excess = observed_latency - serialization - processing
    return event_joint_channel_underestimate_score(
        predicted_snr,
        observed_snr,
        predicted_excess,
        observed_excess,
        snr_resolution_db=snr_resolution_db,
        latency_resolution_s=excess_latency_resolution_s,
        valid_mask=required,
        delivered_mask=delivered_mask,
    )


@dataclass(frozen=True)
class FrozenChannelMarginEpoch:
    """One immutable, event-calibrated joint physical margin."""

    epoch_id: str
    multiplier: float
    miscoverage: float
    snr_resolution_db: float
    latency_resolution_s: float
    calibration_event_ids: tuple[str, ...]
    calibration_event_scores: tuple[float, ...]

    @property
    def snr_margin_db(self) -> float:
        return float(self.multiplier * self.snr_resolution_db)

    @property
    def latency_margin_s(self) -> float:
        return float(self.multiplier * self.latency_resolution_s)

    @property
    def finite(self) -> bool:
        return bool(
            math.isfinite(self.snr_margin_db)
            and math.isfinite(self.latency_margin_s)
        )


@dataclass(frozen=True)
class FrozenJointSafetyEpoch:
    """One conformal multiplier shared by transition and channel bounds.

    Calibrating the event-wise maximum gives simultaneous marginal coverage
    without assuming that transition error and channel error are independent.
    It also avoids spending the risk budget twice through a union bound.
    """

    epoch_id: str
    multiplier: float
    miscoverage: float
    snr_resolution_db: float
    latency_resolution_s: float
    calibration_event_ids: tuple[str, ...]
    transition_event_scores: tuple[float, ...]
    channel_event_scores: tuple[float, ...]
    joint_event_scores: tuple[float, ...]

    @property
    def snr_margin_db(self) -> float:
        return float(self.multiplier * self.snr_resolution_db)

    @property
    def latency_margin_s(self) -> float:
        return float(self.multiplier * self.latency_resolution_s)

    @property
    def finite(self) -> bool:
        return bool(
            math.isfinite(self.multiplier)
            and math.isfinite(self.snr_margin_db)
            and math.isfinite(self.latency_margin_s)
        )


def calibrate_frozen_joint_safety_epoch(
    transition_event_scores: Sequence[float],
    channel_event_scores: Sequence[float],
    *,
    event_ids: Sequence[str],
    miscoverage: float,
    snr_resolution_db: float,
    latency_resolution_s: float,
    epoch_id: str,
    training_event_ids: Sequence[str] = (),
) -> FrozenJointSafetyEpoch:
    """Calibrate one post-selection bound for both safety mechanisms.

    Each input must already be one score per independent event.  The event
    IDs must align, be unique, and be disjoint from residual-model training.
    The same calibrated multiplier is then used in the transition upper bound
    and in both physical channel margins.
    """
    transition = np.asarray(
        transition_event_scores, dtype=np.float64).reshape(-1)
    channel = np.asarray(channel_event_scores, dtype=np.float64).reshape(-1)
    normalized_ids = tuple(str(value) for value in event_ids)
    if transition.shape != channel.shape or transition.size != len(normalized_ids):
        raise ValueError("joint safety event collections must align")
    if transition.size < 1:
        raise ValueError("at least one joint calibration event is required")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("joint calibration event IDs must be unique")
    training_ids = {str(value) for value in training_event_ids}
    overlap = training_ids & set(normalized_ids)
    if overlap:
        raise ValueError(
            f"training/joint-calibration event leakage: {sorted(overlap)}")
    if (
        np.any(np.isnan(transition)) or np.any(np.isneginf(transition))
        or np.any(np.isnan(channel)) or np.any(np.isneginf(channel))
        or np.any(channel < 0.0)
    ):
        raise ValueError(
            "channel scores must be non-negative; no score may be NaN/-infinity")
    snr_resolution = float(snr_resolution_db)
    latency_resolution = float(latency_resolution_s)
    if not np.isfinite(snr_resolution) or snr_resolution <= 0.0:
        raise ValueError("snr_resolution_db must be finite and positive")
    if not np.isfinite(latency_resolution) or latency_resolution <= 0.0:
        raise ValueError("latency_resolution_s must be finite and positive")
    joint = np.maximum(transition, channel)
    multiplier = split_conformal_upper_multiplier(
        joint,
        miscoverage=float(miscoverage),
    )
    return FrozenJointSafetyEpoch(
        epoch_id=str(epoch_id),
        multiplier=float(multiplier),
        miscoverage=float(miscoverage),
        snr_resolution_db=snr_resolution,
        latency_resolution_s=latency_resolution,
        calibration_event_ids=normalized_ids,
        transition_event_scores=tuple(float(value) for value in transition),
        channel_event_scores=tuple(float(value) for value in channel),
        joint_event_scores=tuple(float(value) for value in joint),
    )


def calibrate_frozen_channel_margin_epoch(
    predicted_snr_events_db: Sequence[np.ndarray],
    observed_snr_events_db: Sequence[np.ndarray],
    predicted_latency_events_s: Sequence[np.ndarray],
    observed_latency_events_s: Sequence[np.ndarray],
    *,
    event_ids: Sequence[str],
    miscoverage: float,
    snr_resolution_db: float,
    latency_resolution_s: float,
    epoch_id: str,
    valid_mask_events: Sequence[np.ndarray] | None = None,
    delivered_mask_events: Sequence[np.ndarray] | None = None,
) -> FrozenChannelMarginEpoch:
    """Calibrate one joint margin from independent reconfiguration events.

    Resolution scales must be fixed before looking at calibration outcomes.
    They define units, not an economic trade-off: the maximum score requires
    both the SNR and delay inequalities to hold simultaneously.
    """
    count = len(predicted_snr_events_db)
    if not (
        count == len(observed_snr_events_db)
        == len(predicted_latency_events_s)
        == len(observed_latency_events_s)
        == len(event_ids)
    ):
        raise ValueError("channel calibration event collections must align")
    if count < 1:
        raise ValueError("at least one channel calibration event is required")
    normalized_ids = tuple(str(value) for value in event_ids)
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("channel calibration event IDs must be unique")
    if valid_mask_events is None:
        masks: Sequence[np.ndarray | None] = [None] * count
    else:
        if len(valid_mask_events) != count:
            raise ValueError("valid-mask events must align with calibration")
        masks = valid_mask_events
    if delivered_mask_events is None:
        delivered_masks: Sequence[np.ndarray | None] = [None] * count
    else:
        if len(delivered_mask_events) != count:
            raise ValueError("delivery-mask events must align with calibration")
        delivered_masks = delivered_mask_events
    scores = tuple(
        event_joint_channel_underestimate_score(
            predicted_snr,
            observed_snr,
            predicted_latency,
            observed_latency,
            snr_resolution_db=snr_resolution_db,
            latency_resolution_s=latency_resolution_s,
            valid_mask=mask,
            delivered_mask=delivered_mask,
        )
        for (
            predicted_snr,
            observed_snr,
            predicted_latency,
            observed_latency,
            mask,
            delivered_mask,
        )
        in zip(
            predicted_snr_events_db,
            observed_snr_events_db,
            predicted_latency_events_s,
            observed_latency_events_s,
            masks,
            delivered_masks,
        )
    )
    multiplier = split_conformal_upper_multiplier(
        np.asarray(scores, dtype=np.float64),
        miscoverage=float(miscoverage),
    )
    return FrozenChannelMarginEpoch(
        epoch_id=str(epoch_id),
        multiplier=float(multiplier),
        miscoverage=float(miscoverage),
        snr_resolution_db=float(snr_resolution_db),
        latency_resolution_s=float(latency_resolution_s),
        calibration_event_ids=normalized_ids,
        calibration_event_scores=tuple(float(value) for value in scores),
    )
