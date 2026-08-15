import numpy as np
import pytest

from uav_isac.evaluation.compute_energy_calibration import (
    ComputeEnergyEventObservation,
    calibrate_frozen_compute_energy_epoch,
    episode_compute_energy_scores,
    event_package_energy_upper_j,
    validate_frozen_compute_energy_epoch,
)


def _event(episode, event, energy, *, resolution=0.001, censored=False):
    return ComputeEnergyEventObservation(
        episode_id=str(episode),
        event_id=str(event),
        counter_start_j=(None if censored else 10.0),
        counter_end_j=(None if censored else 10.0 + float(energy)),
        counter_resolution_j=(None if censored else float(resolution)),
        window_censored=bool(censored),
    )


def test_energy_delta_adds_one_counter_resolution():
    assert event_package_energy_upper_j(_event("e", "0", 0.25)) == pytest.approx(
        0.251)


def test_meter_difference_uncertainty_is_added_once_and_must_be_valid():
    base = _event("e", "0", 0.25)
    observation = ComputeEnergyEventObservation(
        **{
            **base.__dict__,
            "counter_difference_uncertainty_j": 0.02,
        }
    )
    assert event_package_energy_upper_j(observation) == pytest.approx(0.271)
    invalid = ComputeEnergyEventObservation(
        **{
            **base.__dict__,
            "counter_difference_uncertainty_j": -0.01,
        }
    )
    with pytest.raises(ValueError, match="uncertainty"):
        event_package_energy_upper_j(invalid)


def test_energy_counter_wrap_is_explicit_and_exactly_charged():
    wrapped = ComputeEnergyEventObservation(
        "e", "0", 9.8, 0.2, 0.01,
        counter_wrap_count=1,
        counter_modulus_j=10.0,
    )
    assert event_package_energy_upper_j(wrapped) == pytest.approx(0.41)
    with pytest.raises(ValueError, match="explicit positive wrap"):
        event_package_energy_upper_j(ComputeEnergyEventObservation(
            "e", "1", 9.8, 0.2, 0.01,
            counter_wrap_count=0,
            counter_modulus_j=10.0,
        ))


def test_episode_energy_is_one_maximum_and_censoring_is_infinite():
    scores = episode_compute_energy_scores(
        [
            _event("a", "0", 0.1),
            _event("a", "1", 0.3),
            _event("b", "0", 0.0, censored=True),
        ],
        expected_episode_ids=("a", "b"),
    )
    assert scores[0].package_energy_upper_j == pytest.approx(0.301)
    assert scores[0].event_count == 2
    assert np.isinf(scores[1].package_energy_upper_j)


def _calibration(count):
    ids = tuple(f"cal-{index}" for index in range(count))
    events = [_event(episode, "0", 0.1 + index * 0.01) for index, episode in enumerate(ids)]
    return ids, events


def test_energy_quantile_is_finite_at_19_and_unresolved_at_18_for_5_percent():
    ids, events = _calibration(19)
    epoch = calibrate_frozen_compute_energy_epoch(
        events,
        calibration_episode_ids=ids,
        miscoverage=0.05,
        epoch_id="finite-energy",
    )
    assert epoch.finite
    assert epoch.rank_one_based == 19
    assert epoch.package_energy_bound_j == pytest.approx(0.281)
    short_ids, short_events = _calibration(18)
    short = calibrate_frozen_compute_energy_epoch(
        short_events,
        calibration_episode_ids=short_ids,
        miscoverage=0.05,
        epoch_id="unresolved-energy",
    )
    assert not short.finite
    assert np.isinf(short.package_energy_bound_j)


def test_energy_validation_is_disjoint_and_episode_level():
    ids, events = _calibration(19)
    epoch = calibrate_frozen_compute_energy_epoch(
        events,
        calibration_episode_ids=ids,
        training_episode_ids=("train-0",),
        miscoverage=0.05,
        epoch_id="energy-validation",
    )
    validation = validate_frozen_compute_energy_epoch(
        epoch,
        [_event("val-0", "0", 0.2), _event("val-1", "0", 0.4)],
        validation_episode_ids=("val-0", "val-1"),
    )
    assert validation.failure_episode_ids == ("val-1",)
    with pytest.raises(ValueError, match="leakage"):
        validate_frozen_compute_energy_epoch(
            epoch,
            [_event(ids[0], "new", 0.1)],
            validation_episode_ids=(ids[0],),
        )


@pytest.mark.parametrize(
    "observation",
    [
        ComputeEnergyEventObservation("e", "0", 1.0, 1.1, 0.0),
        ComputeEnergyEventObservation("e", "0", 1.0, 1.1, -0.1),
        ComputeEnergyEventObservation("e", "0", 1.0, 1.1, 0.1, -1),
        ComputeEnergyEventObservation("e", "0", 1.0, 1.1, 0.1, 0, None, "false"),
    ],
)
def test_invalid_energy_windows_are_rejected(observation):
    with pytest.raises(ValueError):
        event_package_energy_upper_j(observation)
