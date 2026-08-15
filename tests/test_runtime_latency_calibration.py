import numpy as np
import pytest

from uav_isac.evaluation.runtime_latency_calibration import (
    RuntimeLatencyEventObservation,
    calibrate_frozen_runtime_latency_epoch,
    episode_runtime_latency_scores,
    event_complete_compute_latency_upper_s,
    validate_frozen_runtime_latency_epoch,
)


def _event(episode, event, latency, *, censored=False):
    return RuntimeLatencyEventObservation(
        episode_id=str(episode),
        event_id=str(event),
        observed_complete_compute_latency_s=(
            None if censored else float(latency)),
        timer_resolution_s=(None if censored else 1.0e-6),
        timer_difference_uncertainty_s=(None if censored else 2.0e-6),
        window_censored=bool(censored),
    )


def test_runtime_upper_adds_resolution_and_timer_uncertainty():
    assert event_complete_compute_latency_upper_s(
        _event("e", "0", 0.02)) == pytest.approx(0.020003)


def test_runtime_episode_uses_maximum_and_censored_window_is_infinite():
    scores = episode_runtime_latency_scores(
        [_event("a", "0", 0.01), _event("a", "1", 0.03),
         _event("b", "0", 0.0, censored=True)],
        expected_episode_ids=("a", "b"),
    )
    assert scores[0].complete_compute_latency_upper_s == pytest.approx(0.030003)
    assert np.isinf(scores[1].complete_compute_latency_upper_s)


def _calibration(count):
    ids = tuple(f"cal-{index}" for index in range(count))
    return ids, [
        _event(episode, "0", 0.01 + index * 0.001)
        for index, episode in enumerate(ids)]


def test_runtime_quantile_is_finite_at_19_and_infinite_at_18_for_5_percent():
    ids, events = _calibration(19)
    epoch = calibrate_frozen_runtime_latency_epoch(
        events, calibration_episode_ids=ids, miscoverage=0.05,
        epoch_id="runtime-finite")
    assert epoch.finite
    assert epoch.rank_one_based == 19
    assert epoch.complete_compute_latency_bound_s == pytest.approx(0.028003)
    short_ids, short_events = _calibration(18)
    short = calibrate_frozen_runtime_latency_epoch(
        short_events, calibration_episode_ids=short_ids, miscoverage=0.05,
        epoch_id="runtime-unresolved")
    assert not short.finite


def test_runtime_validation_is_episode_level_and_disjoint():
    ids, events = _calibration(19)
    epoch = calibrate_frozen_runtime_latency_epoch(
        events, calibration_episode_ids=ids,
        training_episode_ids=("train-0",), miscoverage=0.05,
        epoch_id="runtime-validation")
    validation = validate_frozen_runtime_latency_epoch(
        epoch, [_event("val-0", "0", 0.02),
                _event("val-1", "0", 0.04)],
        validation_episode_ids=("val-0", "val-1"))
    assert validation.failure_episode_ids == ("val-1",)
    with pytest.raises(ValueError, match="leakage"):
        validate_frozen_runtime_latency_epoch(
            epoch, [_event(ids[0], "new", 0.01)],
            validation_episode_ids=(ids[0],))


@pytest.mark.parametrize(
    "observation",
    [
        RuntimeLatencyEventObservation("e", "0", -0.1, 1.0e-6),
        RuntimeLatencyEventObservation("e", "0", 0.1, 0.0),
        RuntimeLatencyEventObservation("e", "0", 0.1, 1.0e-6, -1.0e-6),
        RuntimeLatencyEventObservation(
            "e", "0", 0.1, 1.0e-6, 0.0, "false"),
    ],
)
def test_invalid_runtime_windows_are_rejected(observation):
    with pytest.raises(ValueError):
        event_complete_compute_latency_upper_s(observation)
