"""Two-timescale residual feedback for fail-closed reconfiguration.

The fast control loop uses one immutable calibration epoch.  Feedback from an
executed event is delayed data for a later epoch; it cannot change the safety
threshold that authorized the same event.  A residual model may use every
candidate/horizon/tail item for prediction, but fitting gives each independent
event equal total weight and calibration reduces each event to one maximum
standardized underestimate.

The module also accounts for the physical transport of owner feedback.  It is
not a free centralized label channel: packet bits, orthogonal bandwidth
sharing, SNR, latency, RF energy, state version and the per-UAV ISAC power
budget are all explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from uav_isac.environment.communication import InterUAVCommunicationModel
from uav_isac.coordination.dependency_commit import (
    DependencyCommitCertificate,
    DependencyCommitLayout,
    certify_dependency_commit,
    dependency_closure,
)
from uav_isac.coordination.local_exchange_oracle import LocalMove
from uav_isac.evaluation.transition_certificate import (
    event_max_standardized_underestimate,
    safe_reconfiguration_is_certified,
    split_conformal_upper_multiplier,
)


def _index_bits(cardinality: int) -> int:
    return max(1, int(np.ceil(np.log2(max(int(cardinality), 2)))))


def build_residual_feedback_features(
    token_age_frames: np.ndarray,
    packet_loss_fraction: np.ndarray,
    quantization_step: np.ndarray,
    switch_fraction: np.ndarray,
    horizon_fraction: np.ndarray,
    tail_fraction: np.ndarray,
) -> np.ndarray:
    """Build dimensionless, owner-observable residual features.

    Inputs broadcast to a common shape.  The feature map deliberately excludes
    privileged global geometry.  Age is compressed by ``log1p``; loss,
    switching, horizon and tail indices are normalized to [0,1].
    """
    age, loss, quant, switch, horizon, tail = np.broadcast_arrays(
        np.asarray(token_age_frames, dtype=np.float64),
        np.asarray(packet_loss_fraction, dtype=np.float64),
        np.asarray(quantization_step, dtype=np.float64),
        np.asarray(switch_fraction, dtype=np.float64),
        np.asarray(horizon_fraction, dtype=np.float64),
        np.asarray(tail_fraction, dtype=np.float64),
    )
    values = (age, loss, quant, switch, horizon, tail)
    if any(np.any(~np.isfinite(value)) for value in values):
        raise ValueError("feedback features must be finite")
    if np.any(age < 0.0) or np.any(quant < 0.0):
        raise ValueError("age and quantization step must be non-negative")
    for name, value in (
        ("packet_loss_fraction", loss),
        ("switch_fraction", switch),
        ("horizon_fraction", horizon),
        ("tail_fraction", tail),
    ):
        if np.any((value < 0.0) | (value > 1.0)):
            raise ValueError(f"{name} must lie in [0,1]")
    log_age = np.log1p(age)
    return np.stack((
        log_age,
        loss,
        quant,
        switch,
        horizon,
        tail,
        log_age * loss,
        switch * horizon,
    ), axis=-1)


@dataclass(frozen=True)
class FrozenResidualModel:
    """Immutable event-balanced ridge model for additive safety residual."""

    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]
    ridge: float
    training_event_ids: tuple[str, ...]

    def predict(self, features: np.ndarray) -> np.ndarray:
        value = np.asarray(features, dtype=np.float64)
        dimension = len(self.coefficients)
        if value.ndim < 1 or value.shape[-1] != dimension:
            raise ValueError(
                f"features must end in model dimension {dimension}")
        if np.any(~np.isfinite(value)):
            raise ValueError("features must be finite")
        mean = np.asarray(self.feature_mean, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        coefficient = np.asarray(self.coefficients, dtype=np.float64)
        normalized = (value - mean) / scale
        return float(self.intercept) + np.sum(
            normalized * coefficient, axis=-1)


def _solve_strictly_positive_system(
    matrix: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Deterministic scalar Cholesky solve for a small SPD ridge system.

    The feedback feature dimension is small.  Keeping this solve independent
    of platform BLAS avoids a known Windows MKL abort after long Torch/SciPy
    test processes, while positive ridge regularization supplies the strict
    definiteness required by Cholesky.
    """
    value = np.asarray(matrix, dtype=np.float64)
    target = np.asarray(rhs, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("matrix must be square")
    if target.shape != (value.shape[0],):
        raise ValueError("rhs shape does not match matrix")
    size = value.shape[0]
    lower = np.zeros_like(value)
    for row in range(size):
        for column in range(row + 1):
            subtotal = 0.0
            for index in range(column):
                subtotal += lower[row, index] * lower[column, index]
            if row == column:
                diagonal = float(value[row, row] - subtotal)
                if not np.isfinite(diagonal) or diagonal <= 0.0:
                    raise ValueError(
                        "ridge normal matrix is not strictly positive definite")
                lower[row, column] = math.sqrt(diagonal)
            else:
                lower[row, column] = (
                    float(value[row, column]) - subtotal
                ) / lower[column, column]
    forward = np.zeros(size, dtype=np.float64)
    for row in range(size):
        subtotal = sum(
            lower[row, column] * forward[column]
            for column in range(row)
        )
        forward[row] = (target[row] - subtotal) / lower[row, row]
    solution = np.zeros(size, dtype=np.float64)
    for row in range(size - 1, -1, -1):
        subtotal = sum(
            lower[column, row] * solution[column]
            for column in range(row + 1, size)
        )
        solution[row] = (forward[row] - subtotal) / lower[row, row]
    return solution


def _flatten_event(
    features: np.ndarray,
    predicted_loss: np.ndarray,
    observed_loss: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    feature = np.asarray(features, dtype=np.float64)
    predicted = np.asarray(predicted_loss, dtype=np.float64)
    observed = np.asarray(observed_loss, dtype=np.float64)
    if feature.ndim < 2:
        raise ValueError("each feature event must have shape (...,D)")
    if predicted.shape != observed.shape or feature.shape[:-1] != predicted.shape:
        raise ValueError("event feature/loss shapes are inconsistent")
    flat_feature = feature.reshape(-1, feature.shape[-1])
    residual = (observed - predicted).reshape(-1)
    valid = (
        np.all(np.isfinite(flat_feature), axis=1)
        & np.isfinite(residual)
    )
    if not np.any(valid):
        raise ValueError("event has no finite residual samples")
    return flat_feature[valid], residual[valid]


def fit_event_balanced_residual_model(
    feature_events: Sequence[np.ndarray],
    predicted_loss_events: Sequence[np.ndarray],
    observed_loss_events: Sequence[np.ndarray],
    *,
    event_ids: Sequence[str],
    ridge: float = 1.0e-3,
    minimum_feature_scale: float = 1.0e-9,
) -> FrozenResidualModel:
    """Fit additive error feedback with equal total weight per event."""
    count = len(feature_events)
    if not (
        count == len(predicted_loss_events)
        == len(observed_loss_events) == len(event_ids)
    ):
        raise ValueError("training event collections must have equal length")
    if count < 1:
        raise ValueError("at least one training event is required")
    normalized_ids = tuple(str(value) for value in event_ids)
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("training event IDs must be unique")
    penalty = float(ridge)
    if not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("ridge must be finite and positive")

    features: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    dimension = None
    for feature, predicted, observed in zip(
        feature_events, predicted_loss_events, observed_loss_events,
    ):
        x_event, y_event = _flatten_event(feature, predicted, observed)
        if dimension is None:
            dimension = x_event.shape[1]
        elif x_event.shape[1] != dimension:
            raise ValueError("all feedback events must use one feature dimension")
        features.append(x_event)
        residuals.append(y_event)
        weights.append(np.full(
            y_event.size, 1.0 / (count * y_event.size), dtype=np.float64))
    x = np.concatenate(features, axis=0)
    y = np.concatenate(residuals, axis=0)
    weight = np.concatenate(weights, axis=0)
    mean = np.sum(weight[:, None] * x, axis=0)
    variance = np.sum(weight[:, None] * (x - mean) ** 2, axis=0)
    scale = np.sqrt(np.maximum(variance, float(minimum_feature_scale) ** 2))
    normalized = (x - mean) / scale
    design = np.column_stack((np.ones(x.shape[0]), normalized))
    parameter_count = design.shape[1]
    gram = np.zeros((parameter_count, parameter_count), dtype=np.float64)
    rhs = np.zeros(parameter_count, dtype=np.float64)
    # Scalar accumulation avoids the same long-process Windows MKL failure as
    # the solver below.  Feature dimension is deliberately small, so this is
    # deterministic and negligible relative to event replay.
    for sample in range(design.shape[0]):
        sample_weight = float(weight[sample])
        for row in range(parameter_count):
            row_value = float(design[sample, row])
            rhs[row] += sample_weight * row_value * float(y[sample])
            for column in range(row + 1):
                contribution = (
                    sample_weight * row_value
                    * float(design[sample, column])
                )
                gram[row, column] += contribution
                if row != column:
                    gram[column, row] += contribution
    for index in range(1, parameter_count):
        gram[index, index] += penalty
    parameter = _solve_strictly_positive_system(gram, rhs)
    return FrozenResidualModel(
        feature_mean=tuple(float(value) for value in mean),
        feature_scale=tuple(float(value) for value in scale),
        intercept=float(parameter[0]),
        coefficients=tuple(float(value) for value in parameter[1:]),
        ridge=penalty,
        training_event_ids=normalized_ids,
    )


@dataclass(frozen=True)
class FrozenFeedbackEpoch:
    """One atomically deployed residual model and conformal multiplier."""

    epoch_id: str
    model: FrozenResidualModel
    multiplier: float
    miscoverage: float
    calibration_event_ids: tuple[str, ...]
    calibration_event_scores: tuple[float, ...]
    proposal_pipeline_digest: str
    transition_calibration_event_scores: tuple[float, ...] = tuple()
    channel_calibration_event_scores: tuple[float, ...] = tuple()
    snr_resolution_db: float | None = None
    latency_resolution_s: float | None = None

    def corrected_prediction(
        self,
        predicted_loss: np.ndarray,
        features: np.ndarray,
    ) -> np.ndarray:
        predicted = np.asarray(predicted_loss, dtype=np.float64)
        correction = self.model.predict(features)
        if correction.shape != predicted.shape:
            raise ValueError("feedback correction shape must match predicted loss")
        return predicted + correction

    @property
    def physical_margins(self) -> tuple[float, float] | None:
        """Return joint-epoch channel margins, if physically calibrated."""
        if self.snr_resolution_db is None or self.latency_resolution_s is None:
            return None
        return (
            float(self.multiplier * self.snr_resolution_db),
            float(self.multiplier * self.latency_resolution_s),
        )


def calibrate_frozen_feedback_epoch(
    model: FrozenResidualModel,
    feature_events: Sequence[np.ndarray],
    predicted_loss_events: Sequence[np.ndarray],
    observed_loss_events: Sequence[np.ndarray],
    uncertainty_events: Sequence[np.ndarray],
    *,
    event_ids: Sequence[str],
    miscoverage: float,
    epoch_id: str,
    proposal_pipeline_digests: Sequence[str],
    channel_event_scores: Sequence[float] | None = None,
    snr_resolution_db: float | None = None,
    latency_resolution_s: float | None = None,
) -> FrozenFeedbackEpoch:
    """Calibrate only on events disjoint from residual-model training.

    When one event-level physical channel score is supplied per event, the
    conformal score is the maximum of transition and channel scores.  One
    multiplier then certifies both mechanisms simultaneously, avoiding a
    second miscoverage allocation and any independence assumption.
    """
    count = len(feature_events)
    if not (
        count == len(predicted_loss_events) == len(observed_loss_events)
        == len(uncertainty_events) == len(event_ids)
    ):
        raise ValueError("calibration event collections must have equal length")
    if count < 1:
        raise ValueError("at least one calibration event is required")
    if len(proposal_pipeline_digests) != count:
        raise ValueError("proposal pipeline digests must align with events")
    pipeline_digests = tuple(
        str(value).strip() for value in proposal_pipeline_digests)
    if any(not value for value in pipeline_digests):
        raise ValueError("proposal pipeline digest cannot be empty")
    if len(set(pipeline_digests)) != 1:
        raise ValueError(
            "one calibration epoch cannot mix proposal pipelines")
    calibration_ids = tuple(str(value) for value in event_ids)
    if len(set(calibration_ids)) != len(calibration_ids):
        raise ValueError("calibration event IDs must be unique")
    overlap = set(calibration_ids) & set(model.training_event_ids)
    if overlap:
        raise ValueError(
            f"training/calibration event leakage: {sorted(overlap)}")

    transition_scores = []
    for feature, predicted, observed, uncertainty in zip(
        feature_events,
        predicted_loss_events,
        observed_loss_events,
        uncertainty_events,
    ):
        predicted_array = np.asarray(predicted, dtype=np.float64)
        observed_array = np.asarray(observed, dtype=np.float64)
        uncertainty_array = np.asarray(uncertainty, dtype=np.float64)
        corrected = predicted_array + model.predict(feature)
        transition_scores.append(event_max_standardized_underestimate(
            corrected, observed_array, uncertainty_array))
    if channel_event_scores is None:
        if snr_resolution_db is not None or latency_resolution_s is not None:
            raise ValueError(
                "physical resolutions require channel event scores")
        channel_scores: tuple[float, ...] = tuple()
        scores = np.asarray(transition_scores, dtype=np.float64)
        normalized_snr_resolution = None
        normalized_latency_resolution = None
    else:
        channel = np.asarray(
            channel_event_scores, dtype=np.float64).reshape(-1)
        if channel.size != count:
            raise ValueError("channel scores must provide one value per event")
        if (
            np.any(np.isnan(channel)) or np.any(np.isneginf(channel))
            or np.any(channel < 0.0)
        ):
            raise ValueError(
                "channel scores must be non-negative and not NaN/-infinity")
        if snr_resolution_db is None or latency_resolution_s is None:
            raise ValueError(
                "joint calibration requires both physical resolutions")
        normalized_snr_resolution = float(snr_resolution_db)
        normalized_latency_resolution = float(latency_resolution_s)
        if (
            not np.isfinite(normalized_snr_resolution)
            or normalized_snr_resolution <= 0.0
        ):
            raise ValueError("snr_resolution_db must be finite and positive")
        if (
            not np.isfinite(normalized_latency_resolution)
            or normalized_latency_resolution <= 0.0
        ):
            raise ValueError(
                "latency_resolution_s must be finite and positive")
        channel_scores = tuple(float(value) for value in channel)
        scores = np.maximum(
            np.asarray(transition_scores, dtype=np.float64), channel)
    multiplier = split_conformal_upper_multiplier(
        scores,
        miscoverage=float(miscoverage),
    )
    return FrozenFeedbackEpoch(
        epoch_id=str(epoch_id),
        model=model,
        multiplier=float(multiplier),
        miscoverage=float(miscoverage),
        calibration_event_ids=calibration_ids,
        calibration_event_scores=tuple(float(value) for value in scores),
        proposal_pipeline_digest=pipeline_digests[0],
        transition_calibration_event_scores=tuple(
            float(value) for value in transition_scores),
        channel_calibration_event_scores=channel_scores,
        snr_resolution_db=normalized_snr_resolution,
        latency_resolution_s=normalized_latency_resolution,
    )


def feedback_adjusted_reconfiguration_is_certified(
    epoch: FrozenFeedbackEpoch,
    features: np.ndarray,
    predicted_loss: np.ndarray,
    uncertainty_scale: np.ndarray,
    *,
    commit_feasible: bool,
    structural_feasible: bool,
    drift_locked: bool,
    proposal_pipeline_digest: str,
    safety_margin: float = 0.0,
) -> bool:
    """Apply one frozen epoch; a drift alarm has absolute No-op priority."""
    if bool(drift_locked):
        return False
    if str(proposal_pipeline_digest).strip() != epoch.proposal_pipeline_digest:
        return False
    corrected = epoch.corrected_prediction(predicted_loss, features)
    return safe_reconfiguration_is_certified(
        corrected,
        uncertainty_scale,
        multiplier=epoch.multiplier,
        commit_feasible=commit_feasible,
        structural_feasible=structural_feasible,
        safety_margin=safety_margin,
    )


@dataclass(frozen=True)
class ViolationMonitorState:
    events: int
    violations: int
    e_value: float
    alarm: bool


class CertificateViolationEProcess:
    """Anytime drift alarm for repeated frozen-certificate exceedances.

    Under the explicit null assumption
    ``P(violation_t | past) <= miscoverage``, each fixed-alternative likelihood
    ratio is a non-negative supermartingale.  Their uniform mixture is also an
    e-process, so Ville's inequality bounds the probability of ever crossing
    ``1/false_alarm_probability``.  The alarm is sticky and forces No-op.
    """

    def __init__(
        self,
        *,
        miscoverage: float,
        false_alarm_probability: float = 0.01,
        alternative_rates: Sequence[float] | None = None,
    ) -> None:
        alpha = float(miscoverage)
        delta = float(false_alarm_probability)
        if not 0.0 < alpha < 1.0:
            raise ValueError("miscoverage must lie in (0,1)")
        if not 0.0 < delta < 1.0:
            raise ValueError("false_alarm_probability must lie in (0,1)")
        if alternative_rates is None:
            alternatives = tuple(
                alpha + (1.0 - alpha) * fraction
                for fraction in (0.10, 0.25, 0.50, 0.75)
            )
        else:
            alternatives = tuple(float(value) for value in alternative_rates)
        if not alternatives or any(
            not alpha < value < 1.0 for value in alternatives
        ):
            raise ValueError("every alternative rate must lie in (alpha,1)")
        self.miscoverage = alpha
        self.false_alarm_probability = delta
        self.alternative_rates = alternatives
        self._log_e = np.zeros(len(alternatives), dtype=np.float64)
        self._events = 0
        self._violations = 0
        self._alarm = False

    def update(self, *, event_score: float, multiplier: float) -> ViolationMonitorState:
        score = float(event_score)
        threshold = float(multiplier)
        if math.isnan(score) or math.isnan(threshold):
            raise ValueError("event score and multiplier cannot be NaN")
        violation = bool(score > threshold)
        self._events += 1
        self._violations += int(violation)
        alpha = self.miscoverage
        for index, alternative in enumerate(self.alternative_rates):
            if violation:
                self._log_e[index] += math.log(alternative / alpha)
            else:
                self._log_e[index] += math.log(
                    (1.0 - alternative) / (1.0 - alpha))
        maximum = float(np.max(self._log_e))
        mixture_log = maximum + math.log(float(np.mean(
            np.exp(self._log_e - maximum))))
        e_value = float(math.exp(min(mixture_log, 700.0)))
        if e_value >= 1.0 / self.false_alarm_probability:
            self._alarm = True
        return ViolationMonitorState(
            events=self._events,
            violations=self._violations,
            e_value=e_value,
            alarm=self._alarm,
        )

    @property
    def state(self) -> ViolationMonitorState:
        maximum = float(np.max(self._log_e))
        mixture_log = maximum + math.log(float(np.mean(
            np.exp(self._log_e - maximum))))
        return ViolationMonitorState(
            events=self._events,
            violations=self._violations,
            e_value=float(math.exp(min(mixture_log, 700.0))),
            alarm=self._alarm,
        )


@dataclass(frozen=True)
class OwnerFeedbackLayout:
    num_agents: int
    num_targets: int
    horizon_frames: int
    header_bits: int = 64
    epoch_bits: int = 16
    digest_bits: int = 64
    observed_lower_bound_bits: int = 16

    def __post_init__(self) -> None:
        if self.num_agents < 1 or self.num_targets < 1:
            raise ValueError("num_agents and num_targets must be positive")
        if self.horizon_frames < 0:
            raise ValueError("horizon_frames must be non-negative")
        for name in (
            "header_bits", "epoch_bits", "digest_bits",
            "observed_lower_bound_bits",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def maximum_entries(self) -> int:
        return self.num_targets * (self.horizon_frames + 1)

    @property
    def shared_bits(self) -> int:
        return int(
            self.header_bits + self.epoch_bits + self.digest_bits
            + _index_bits(self.num_agents)
            + _index_bits(self.maximum_entries + 1)
        )

    @property
    def entry_bits(self) -> int:
        return int(
            _index_bits(self.num_targets)
            + _index_bits(self.horizon_frames + 1)
            + self.observed_lower_bound_bits
            + 1  # valid/missing indicator
        )

    def packet_bits(self, num_entries: int) -> int:
        entries = int(num_entries)
        if not 0 <= entries <= self.maximum_entries:
            raise ValueError("num_entries is outside feedback layout capacity")
        if entries == 0:
            return 0
        return self.shared_bits + entries * self.entry_bits


@dataclass(frozen=True)
class OwnerFeedbackTransportCertificate:
    feasible: bool
    reasons: tuple[str, ...]
    coordinator: int
    certificate_epoch_id: int
    certificate_digest: int
    senders: tuple[int, ...]
    total_over_air_bits: int
    max_latency_s: float
    total_energy_j: float
    min_snr_db: float
    per_sender_bits: tuple[int, ...]
    per_uav_energy_j: tuple[float, ...]
    power_excess_w: tuple[float, ...]


def certify_owner_feedback_transport(
    entries_by_owner: Mapping[int, int],
    *,
    coordinator: int,
    positions: np.ndarray,
    comm_power_w: np.ndarray,
    sensing_power_w: np.ndarray,
    state_versions: np.ndarray,
    certificate_epoch_ids: np.ndarray,
    certificate_digests: np.ndarray,
    communication_model: InterUAVCommunicationModel,
    layout: OwnerFeedbackLayout,
    total_power_w: float = 1.0,
    power_tolerance_w: float = 1.0e-12,
    snr_margin_db: float = 0.0,
    latency_margin_s: float = 0.0,
) -> OwnerFeedbackTransportCertificate:
    """Certify one delayed owner-feedback collection round."""
    K = int(layout.num_agents)
    coordinator_id = int(coordinator)
    if not 0 <= coordinator_id < K:
        raise ValueError("coordinator is outside [0,K)")
    pos = np.asarray(positions, dtype=np.float64)
    comm = np.asarray(comm_power_w, dtype=np.float64).reshape(-1)
    sensing = np.asarray(sensing_power_w, dtype=np.float64)
    versions = np.asarray(state_versions)
    epoch_ids = np.asarray(certificate_epoch_ids)
    digests = np.asarray(certificate_digests)
    if pos.shape != (K, 3) or comm.shape != (K,):
        raise ValueError("positions/power shapes do not match layout")
    if sensing.shape == (K,):
        sensing_total = sensing
    elif sensing.ndim == 2 and sensing.shape[0] == K:
        sensing_total = np.sum(sensing, axis=1)
    else:
        raise ValueError("sensing_power_w must have shape (K,) or (K,Q)")
    if (
        versions.shape != (K,) or epoch_ids.shape != (K,)
        or digests.shape != (K,)
    ):
        raise ValueError("feedback identity vectors must have shape (K,)")

    def integer_tuple(values: np.ndarray, name: str) -> tuple[int, ...]:
        normalized = []
        for raw in values.tolist():
            try:
                value = int(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must contain integers") from exc
            if isinstance(raw, (float, np.floating)) and (
                not np.isfinite(raw) or float(raw) != float(value)
            ):
                raise ValueError(f"{name} must contain integers")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            normalized.append(value)
        return tuple(normalized)

    version_values = integer_tuple(versions, "state_versions")
    epoch_values = integer_tuple(epoch_ids, "certificate_epoch_ids")
    digest_values = integer_tuple(digests, "certificate_digests")
    if any(value >= (1 << int(layout.epoch_bits)) for value in epoch_values):
        raise ValueError("feedback epoch does not fit the wire layout")
    if any(value >= (1 << int(layout.digest_bits)) for value in digest_values):
        raise ValueError("feedback digest does not fit the wire layout")
    if (
        np.any(~np.isfinite(pos)) or np.any(~np.isfinite(comm))
        or np.any(~np.isfinite(sensing_total)) or np.any(comm < 0.0)
        or np.any(sensing_total < 0.0)
    ):
        raise ValueError("feedback transport state must be finite/non-negative")
    snr_margin = float(snr_margin_db)
    latency_margin = float(latency_margin_s)
    if np.isnan(snr_margin) or snr_margin < 0.0:
        raise ValueError("snr_margin_db must be non-negative and not NaN")
    if np.isnan(latency_margin) or latency_margin < 0.0:
        raise ValueError("latency_margin_s must be non-negative and not NaN")
    normalized_entries = {int(k): int(v) for k, v in entries_by_owner.items()}
    if any(not 0 <= owner < K for owner in normalized_entries):
        raise ValueError("feedback owner is outside [0,K)")
    for entries in normalized_entries.values():
        layout.packet_bits(entries)
    senders = tuple(sorted(
        owner for owner, entries in normalized_entries.items()
        if owner != coordinator_id and entries > 0
    ))
    reasons = [
        f"power:uav:{k}" for k in np.flatnonzero(
            comm + sensing_total - float(total_power_w)
            > float(power_tolerance_w))
    ]
    participants = tuple(sorted(set(senders) | {coordinator_id}))
    coordinator_version = version_values[coordinator_id]
    coordinator_epoch = epoch_values[coordinator_id]
    coordinator_digest = digest_values[coordinator_id]
    reasons.extend(
        f"protocol:state_version:uav:{participant}"
        for participant in participants
        if version_values[participant] != coordinator_version
    )
    reasons.extend(
        f"protocol:certificate_epoch:uav:{participant}"
        for participant in participants
        if epoch_values[participant] != coordinator_epoch
    )
    reasons.extend(
        f"protocol:certificate_digest:uav:{participant}"
        for participant in participants
        if digest_values[participant] != coordinator_digest
    )
    per_sender_bits = np.zeros(K, dtype=np.int64)
    per_uav_energy = np.zeros(K, dtype=np.float64)
    max_latency = 0.0
    min_snr = float("inf")
    bandwidth = (
        communication_model.bandwidth_hz / len(senders)
        if senders else communication_model.bandwidth_hz
    )
    for sender in senders:
        bits = layout.packet_bits(normalized_entries[sender])
        per_sender_bits[sender] = bits
        snr_db, _rate, serialization_s, latency_s = (
            communication_model.robust_link_budget(
                pos[sender], pos[coordinator_id], bits, bandwidth,
                float(comm[sender]),
                snr_margin_db=snr_margin,
                latency_margin_s=latency_margin,
            )
        )
        min_snr = min(min_snr, float(snr_db))
        max_latency = max(max_latency, float(latency_s))
        per_uav_energy[sender] = (
            float("inf")
            if not np.isfinite(serialization_s)
            else float(comm[sender]) * float(serialization_s)
        )
        if snr_db < communication_model.snr_threshold_db:
            reasons.append(f"feedback:snr:{sender}->{coordinator_id}")
        if latency_s > communication_model.deadline_s:
            reasons.append(f"feedback:deadline:{sender}->{coordinator_id}")
    return OwnerFeedbackTransportCertificate(
        feasible=not reasons,
        reasons=tuple(reasons),
        coordinator=coordinator_id,
        certificate_epoch_id=coordinator_epoch,
        certificate_digest=coordinator_digest,
        senders=senders,
        total_over_air_bits=int(np.sum(per_sender_bits)),
        max_latency_s=float(max_latency),
        total_energy_j=float(np.sum(per_uav_energy)),
        min_snr_db=float(min_snr),
        per_sender_bits=tuple(int(value) for value in per_sender_bits),
        per_uav_energy_j=tuple(float(value) for value in per_uav_energy),
        power_excess_w=tuple(float(value) for value in (
            comm + sensing_total - float(total_power_w))),
    )


@dataclass(frozen=True)
class JointCommitFeedbackReserveResult:
    feasible: bool
    uniform_comm_floor_w: float | None
    coordinator: int
    iterations: int
    commit_certificate: DependencyCommitCertificate
    feedback_certificate: OwnerFeedbackTransportCertificate
    comm_power_w: tuple[float, ...]
    sensing_power_w: tuple[tuple[float, ...], ...]


def minimum_joint_commit_feedback_reserve(
    selected: np.ndarray,
    role: np.ndarray,
    owner: np.ndarray,
    move: LocalMove,
    *,
    feedback_entries_by_owner: Mapping[int, int],
    positions: np.ndarray,
    current_comm_power_w: np.ndarray,
    current_sensing_power_w: np.ndarray,
    sensing_weights: np.ndarray,
    state_versions: np.ndarray,
    certificate_epoch_ids: np.ndarray,
    certificate_digests: np.ndarray,
    communication_model: InterUAVCommunicationModel,
    commit_layout: DependencyCommitLayout,
    feedback_layout: OwnerFeedbackLayout,
    reserve_upper_w: float,
    total_power_w: float = 1.0,
    total_deadline_s: float | None = None,
    tolerance_w: float = 1.0e-9,
    max_iterations: int = 60,
    snr_margin_db: float = 0.0,
    latency_margin_s: float = 0.0,
) -> JointCommitFeedbackReserveResult:
    """Minimize a common RF floor jointly over commit and delayed feedback.

    Proposer/coordinator election is inside the feasibility problem.  For each
    power floor every dependency-closure participant is evaluated as a common
    commit proposer and feedback coordinator; the feasible choice with minimum
    combined latency and then RF energy wins.  This avoids a sequential design
    where a commit-optimal proposer makes the feedback path infeasible.
    """
    current = np.asarray(selected, dtype=bool)
    if current.ndim != 3:
        raise ValueError("selected must have shape (K,K,Q)")
    K, K2, Q = current.shape
    if K != K2:
        raise ValueError("selected graph must have equal UAV axes")
    comm0 = np.asarray(current_comm_power_w, dtype=np.float64).reshape(-1)
    sensing0 = np.asarray(current_sensing_power_w, dtype=np.float64)
    weights = np.asarray(sensing_weights, dtype=np.float64)
    if comm0.shape != (K,) or sensing0.shape != (K, Q):
        raise ValueError("current powers do not match selected cardinalities")
    if weights.shape != (K, Q):
        raise ValueError("sensing_weights must have shape (K,Q)")
    if (
        np.any(~np.isfinite(comm0)) or np.any(~np.isfinite(sensing0))
        or np.any(~np.isfinite(weights)) or np.any(comm0 < 0.0)
        or np.any(sensing0 < 0.0) or np.any(weights < 0.0)
    ):
        raise ValueError("powers and weights must be finite/non-negative")
    mass = np.sum(weights, axis=1, keepdims=True)
    if np.any(mass <= 0.0):
        raise ValueError("each UAV needs positive sensing-weight mass")
    weights = weights / mass
    budget = float(total_power_w)
    upper = float(reserve_upper_w)
    tolerance = float(tolerance_w)
    if not 0.0 <= upper <= budget:
        raise ValueError("reserve_upper_w must lie in [0,total_power_w]")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance_w must be finite and positive")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be positive")
    closure = dependency_closure(current, role, owner, move)
    if not closure.participants:
        raise ValueError("joint reserve is undefined for an empty closure")

    def allocation(floor_w: float) -> tuple[np.ndarray, np.ndarray]:
        comm = comm0.copy()
        sensing = sensing0.copy()
        for participant in closure.participants:
            comm[participant] = max(comm[participant], float(floor_w))
            sensing[participant] = (
                budget - comm[participant]
            ) * weights[participant]
        return comm, sensing

    def evaluate(floor_w: float):
        comm, sensing = allocation(floor_w)
        alternatives = []
        for coordinator in closure.participants:
            commit = certify_dependency_commit(
                current,
                role,
                owner,
                move,
                proposer=coordinator,
                positions=positions,
                comm_power_w=comm,
                sensing_power_w=sensing,
                state_versions=state_versions,
                certificate_epoch_ids=certificate_epoch_ids,
                certificate_digests=certificate_digests,
                communication_model=communication_model,
                total_power_w=budget,
                total_deadline_s=total_deadline_s,
                layout=commit_layout,
                snr_margin_db=snr_margin_db,
                latency_margin_s=latency_margin_s,
            )
            feedback = certify_owner_feedback_transport(
                feedback_entries_by_owner,
                coordinator=coordinator,
                positions=positions,
                comm_power_w=comm,
                sensing_power_w=sensing,
                state_versions=state_versions,
                certificate_epoch_ids=certificate_epoch_ids,
                certificate_digests=certificate_digests,
                communication_model=communication_model,
                layout=feedback_layout,
                total_power_w=budget,
                snr_margin_db=snr_margin_db,
                latency_margin_s=latency_margin_s,
            )
            feasible = commit.feasible and feedback.feasible
            alternatives.append((
                feasible,
                commit,
                feedback,
                coordinator,
                comm,
                sensing,
            ))
        return min(
            alternatives,
            key=lambda item: (
                not item[0],
                len(item[1].reasons) + len(item[2].reasons),
                item[1].total_latency_s + item[2].max_latency_s,
                item[1].total_energy_j + item[2].total_energy_j,
                item[3],
            ),
        )

    lower = evaluate(0.0)
    if lower[0]:
        return JointCommitFeedbackReserveResult(
            feasible=True,
            uniform_comm_floor_w=0.0,
            coordinator=int(lower[3]),
            iterations=0,
            commit_certificate=lower[1],
            feedback_certificate=lower[2],
            comm_power_w=tuple(float(value) for value in lower[4]),
            sensing_power_w=tuple(
                tuple(float(value) for value in row) for row in lower[5]),
        )
    upper_result = evaluate(upper)
    if not upper_result[0]:
        return JointCommitFeedbackReserveResult(
            feasible=False,
            uniform_comm_floor_w=None,
            coordinator=int(upper_result[3]),
            iterations=0,
            commit_certificate=upper_result[1],
            feedback_certificate=upper_result[2],
            comm_power_w=tuple(float(value) for value in upper_result[4]),
            sensing_power_w=tuple(
                tuple(float(value) for value in row)
                for row in upper_result[5]),
        )

    low, high = 0.0, upper
    best = upper_result
    iterations = 0
    while high - low > tolerance and iterations < int(max_iterations):
        middle = 0.5 * (low + high)
        candidate = evaluate(middle)
        iterations += 1
        if candidate[0]:
            high = middle
            best = candidate
        else:
            low = middle
    return JointCommitFeedbackReserveResult(
        feasible=True,
        uniform_comm_floor_w=float(high),
        coordinator=int(best[3]),
        iterations=iterations,
        commit_certificate=best[1],
        feedback_certificate=best[2],
        comm_power_w=tuple(float(value) for value in best[4]),
        sensing_power_w=tuple(
            tuple(float(value) for value in row) for row in best[5]),
    )
