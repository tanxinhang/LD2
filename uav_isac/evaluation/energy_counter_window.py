"""Fail-closed acquisition windows for cumulative controller energy meters."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from uav_isac.evaluation.compute_energy_calibration import (
    ComputeEnergyEventObservation,
)


@dataclass(frozen=True)
class EnergyCounterSample:
    counter_id: str
    monotonic_ns: int
    counter_j: float | None
    counter_resolution_j: float | None
    reading_uncertainty_j: float | None
    counter_modulus_j: float | None
    generation_id: str
    wrap_index: int | None = None
    valid: bool = True
    energy_rate_upper_w: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.valid, (bool, np.bool_)):
            raise ValueError("valid must be boolean")
        if isinstance(self.monotonic_ns, (bool, np.bool_)):
            raise ValueError("monotonic_ns must be an integer")
        timestamp = int(self.monotonic_ns)
        if timestamp != self.monotonic_ns or timestamp < 0:
            raise ValueError("monotonic_ns must be a non-negative integer")
        if not str(self.counter_id).strip() or not str(self.generation_id).strip():
            raise ValueError("counter_id/generation_id must be non-empty")
        if not bool(self.valid):
            return
        required = (
            self.counter_j,
            self.counter_resolution_j,
            self.reading_uncertainty_j,
        )
        if any(value is None for value in required):
            raise ValueError("valid samples require counter/resolution/uncertainty")
        counter, resolution, uncertainty = (float(value) for value in required)
        if (
            not np.isfinite(counter) or counter < 0.0
            or not np.isfinite(resolution) or resolution <= 0.0
            or not np.isfinite(uncertainty) or uncertainty < 0.0
        ):
            raise ValueError("valid counter sample values are outside support")
        if self.counter_modulus_j is not None:
            modulus = float(self.counter_modulus_j)
            if (
                not np.isfinite(modulus) or modulus <= 0.0
                or counter >= modulus
            ):
                raise ValueError("counter/modulus are inconsistent")
        if self.energy_rate_upper_w is not None:
            rate = float(self.energy_rate_upper_w)
            if not np.isfinite(rate) or rate <= 0.0:
                raise ValueError("energy_rate_upper_w must be finite positive")
        if self.wrap_index is not None:
            if isinstance(self.wrap_index, (bool, np.bool_)):
                raise ValueError("wrap_index must be an integer")
            wrap_index = int(self.wrap_index)
            if wrap_index != self.wrap_index or wrap_index < 0:
                raise ValueError("wrap_index must be a non-negative integer")


@dataclass(frozen=True)
class ClosedEnergyCounterWindow:
    observation: ComputeEnergyEventObservation
    duration_s: float
    censored_reason: str | None

    @property
    def censored(self) -> bool:
        return bool(self.observation.window_censored)


def _censored_window(
    *, episode_id: str, event_id: str, duration_s: float, reason: str,
) -> ClosedEnergyCounterWindow:
    return ClosedEnergyCounterWindow(
        observation=ComputeEnergyEventObservation(
            episode_id=str(episode_id),
            event_id=str(event_id),
            counter_start_j=None,
            counter_end_j=None,
            counter_resolution_j=None,
            window_censored=True,
        ),
        duration_s=float(duration_s),
        censored_reason=str(reason),
    )


def close_energy_counter_window(
    start: EnergyCounterSample,
    end: EnergyCounterSample,
    *,
    episode_id: str,
    event_id: str,
) -> ClosedEnergyCounterWindow:
    """Convert two meter samples without guessing resets or counter wraps."""
    if int(end.monotonic_ns) <= int(start.monotonic_ns):
        raise ValueError("energy counter window must have positive duration")
    duration_s = float(
        (int(end.monotonic_ns) - int(start.monotonic_ns)) * 1.0e-9)
    if not start.valid or not end.valid:
        return _censored_window(
            episode_id=episode_id, event_id=event_id,
            duration_s=duration_s, reason="invalid_counter_sample")
    if start.counter_id != end.counter_id:
        return _censored_window(
            episode_id=episode_id, event_id=event_id,
            duration_s=duration_s, reason="counter_identity_changed")
    if start.generation_id != end.generation_id:
        return _censored_window(
            episode_id=episode_id, event_id=event_id,
            duration_s=duration_s, reason="counter_generation_changed")
    if (
        start.counter_resolution_j != end.counter_resolution_j
        or start.counter_modulus_j != end.counter_modulus_j
        or start.energy_rate_upper_w != end.energy_rate_upper_w
    ):
        return _censored_window(
            episode_id=episode_id, event_id=event_id,
            duration_s=duration_s, reason="counter_metrology_changed")

    if start.wrap_index is None and end.wrap_index is None:
        if start.counter_modulus_j is not None:
            if start.energy_rate_upper_w is None:
                return _censored_window(
                    episode_id=episode_id, event_id=event_id,
                    duration_s=duration_s,
                    reason="missing_hidden_wrap_exclusion_bound")
            if (
                float(start.energy_rate_upper_w) * duration_s
                >= float(start.counter_modulus_j)
            ):
                return _censored_window(
                    episode_id=episode_id, event_id=event_id,
                    duration_s=duration_s,
                    reason="hidden_wrap_cannot_be_excluded")
        if float(end.counter_j) < float(start.counter_j):
            return _censored_window(
                episode_id=episode_id, event_id=event_id,
                duration_s=duration_s,
                reason="decreasing_counter_without_wrap_evidence")
        wraps = 0
    elif start.wrap_index is None or end.wrap_index is None:
        return _censored_window(
            episode_id=episode_id, event_id=event_id,
            duration_s=duration_s, reason="incomplete_wrap_evidence")
    else:
        wraps = int(end.wrap_index) - int(start.wrap_index)
        if wraps < 0:
            return _censored_window(
                episode_id=episode_id, event_id=event_id,
                duration_s=duration_s, reason="wrap_index_regressed")
        if wraps > 0 and start.counter_modulus_j is None:
            return _censored_window(
                episode_id=episode_id, event_id=event_id,
                duration_s=duration_s, reason="wrap_has_no_counter_modulus")
    raw_delta = (
        float(end.counter_j) - float(start.counter_j)
        + wraps * float(start.counter_modulus_j or 0.0))
    if raw_delta < 0.0:
        return _censored_window(
            episode_id=episode_id, event_id=event_id,
            duration_s=duration_s,
            reason="wrap_evidence_cannot_explain_counter_drop")
    if start.energy_rate_upper_w is not None:
        metrology_slack = float(
            float(start.reading_uncertainty_j)
            + float(end.reading_uncertainty_j)
            + float(start.counter_resolution_j))
        if raw_delta > (
            float(start.energy_rate_upper_w) * duration_s
            + metrology_slack + 1.0e-15
        ):
            return _censored_window(
                episode_id=episode_id, event_id=event_id,
                duration_s=duration_s,
                reason="counter_delta_exceeds_energy_rate_bound")

    observation = ComputeEnergyEventObservation(
        episode_id=str(episode_id),
        event_id=str(event_id),
        counter_start_j=float(start.counter_j),
        counter_end_j=float(end.counter_j),
        counter_resolution_j=float(start.counter_resolution_j),
        counter_wrap_count=int(wraps),
        counter_modulus_j=(
            None if start.counter_modulus_j is None
            else float(start.counter_modulus_j)),
        window_censored=False,
        counter_difference_uncertainty_j=float(
            float(start.reading_uncertainty_j)
            + float(end.reading_uncertainty_j)),
    )
    return ClosedEnergyCounterWindow(
        observation=observation,
        duration_s=duration_s,
        censored_reason=None,
    )
