import numpy as np
import pytest

from uav_isac.evaluation.link_reliability_calibration import (
    LinkReliabilityEventObservation,
    calibrate_frozen_link_reliability_epoch,
    episode_link_reliability_scores,
    validate_frozen_link_reliability_epoch,
)


def _event(
    episode,
    event,
    erased=0,
    queue=0.0,
    *,
    censored=False,
):
    return LinkReliabilityEventObservation(
        episode_id=str(episode),
        event_id=str(event),
        erased_copies_before_success=erased,
        observed_complete_protocol_latency_s=(
            None if censored else 0.01 + float(queue)),
        observed_snr_modeled_protocol_latency_s=(
            None if censored else 0.01),
        delivery_censored=bool(censored),
    )


def test_episode_score_is_one_maximum_not_event_pseudoreplication():
    scores = episode_link_reliability_scores(
        [
            _event("a", "0", erased=0, queue=0.001),
            _event("a", "1", erased=2, queue=0.0005),
            _event("b", "0", erased=1, queue=0.003),
        ],
        expected_episode_ids=("a", "b"),
    )
    assert len(scores) == 2
    assert scores[0].erasure_score == 2
    assert scores[0].excess_queue_score_s == pytest.approx(0.001)
    assert scores[0].event_count == 2
    assert scores[1].erasure_score == 1
    assert scores[1].excess_queue_score_s == pytest.approx(0.003)


def test_censoring_is_infinite_and_missing_frozen_episode_is_rejected():
    score = episode_link_reliability_scores(
        [_event("a", "0", censored=True)],
        expected_episode_ids=("a",),
    )[0]
    assert np.isinf(score.erasure_score)
    assert np.isinf(score.excess_queue_score_s)
    with pytest.raises(ValueError, match="no link observations"):
        episode_link_reliability_scores(
            [_event("a", "0")], expected_episode_ids=("a", "b"))


def _calibration(count, *, censored_index=None):
    ids = tuple(f"cal-{index}" for index in range(count))
    events = [
        _event(
            episode,
            "0",
            erased=index % 3,
            queue=index * 1.0e-5,
            censored=index == censored_index,
        )
        for index, episode in enumerate(ids)
    ]
    return ids, events


def test_finite_sample_rank_requires_19_episodes_at_five_percent():
    ids, events = _calibration(19)
    epoch = calibrate_frozen_link_reliability_epoch(
        events,
        calibration_episode_ids=ids,
        total_miscoverage=0.10,
        erasure_miscoverage=0.05,
        queue_miscoverage=0.05,
        epoch_id="finite",
    )
    assert epoch.finite
    assert epoch.erasure_rank_one_based == 19
    assert epoch.queue_rank_one_based == 19
    assert epoch.erasure_bound == 2
    assert epoch.recommended_repetition_count == 3
    assert epoch.queue_bound_s == pytest.approx(18.0e-5)
    assert epoch.joint_coverage_floor == pytest.approx(0.90)

    short_ids, short_events = _calibration(18)
    short = calibrate_frozen_link_reliability_epoch(
        short_events,
        calibration_episode_ids=short_ids,
        total_miscoverage=0.10,
        erasure_miscoverage=0.05,
        queue_miscoverage=0.05,
        epoch_id="unresolved",
    )
    assert not short.finite
    assert short.recommended_repetition_count is None
    assert np.isinf(short.erasure_bound)
    assert np.isinf(short.queue_bound_s)


def test_infinite_calibration_score_forces_the_corresponding_bound_infinite():
    ids, events = _calibration(19, censored_index=18)
    epoch = calibrate_frozen_link_reliability_epoch(
        events,
        calibration_episode_ids=ids,
        total_miscoverage=0.10,
        erasure_miscoverage=0.05,
        queue_miscoverage=0.05,
        epoch_id="censored",
    )
    assert not epoch.finite
    assert np.isinf(epoch.erasure_bound)
    assert np.isinf(epoch.queue_bound_s)


def test_risk_budget_and_episode_splits_are_strictly_disjoint():
    ids, events = _calibration(19)
    with pytest.raises(ValueError, match="risk exceeds"):
        calibrate_frozen_link_reliability_epoch(
            events,
            calibration_episode_ids=ids,
            total_miscoverage=0.05,
            erasure_miscoverage=0.04,
            queue_miscoverage=0.04,
            epoch_id="overspent",
        )
    with pytest.raises(ValueError, match="leakage"):
        calibrate_frozen_link_reliability_epoch(
            events,
            calibration_episode_ids=ids,
            training_episode_ids=(ids[0],),
            total_miscoverage=0.10,
            erasure_miscoverage=0.05,
            queue_miscoverage=0.05,
            epoch_id="leaky",
        )


def test_validation_is_episode_level_and_reports_joint_union_failures():
    ids, events = _calibration(19)
    epoch = calibrate_frozen_link_reliability_epoch(
        events,
        calibration_episode_ids=ids,
        total_miscoverage=0.10,
        erasure_miscoverage=0.05,
        queue_miscoverage=0.05,
        epoch_id="validated",
    )
    validation = validate_frozen_link_reliability_epoch(
        epoch,
        [
            _event("val-0", "0", erased=2, queue=0.0),
            _event("val-1", "0", erased=3, queue=0.0),
            _event("val-2", "0", erased=0, queue=1.0e-3),
        ],
        validation_episode_ids=("val-0", "val-1", "val-2"),
    )
    assert validation.episode_count == 3
    assert validation.erasure_failure_episode_ids == ("val-1",)
    assert validation.queue_failure_episode_ids == ("val-2",)
    assert validation.joint_failure_episode_ids == ("val-1", "val-2")
    with pytest.raises(ValueError, match="leakage"):
        validate_frozen_link_reliability_epoch(
            epoch,
            [_event(ids[0], "new")],
            validation_episode_ids=(ids[0],),
        )


@pytest.mark.parametrize(
    "observation",
    [
        _event("a", "0", erased=-1),
        _event("a", "0", erased=1.5),
        LinkReliabilityEventObservation("a", "0", 0, np.nan, 0.0),
        LinkReliabilityEventObservation("a", "0", 0, 0.0, -1.0),
        LinkReliabilityEventObservation("a", "0", 0, 0.0, 0.0, "false"),
    ],
)
def test_invalid_uncensored_link_events_are_rejected(observation):
    with pytest.raises(ValueError):
        episode_link_reliability_scores(
            [observation], expected_episode_ids=("a",))
