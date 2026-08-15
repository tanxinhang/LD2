import numpy as np

from uav_isac.evaluation.horizon_future_audit import (
    consecutive_horizon_indices,
    evaluate_horizon_future_outcome,
    simultaneous_log_envelope_score,
)


def test_consecutive_horizon_indices_are_episode_and_frame_strict():
    seeds = np.asarray([7, 7, 7, 9, 9])
    frames = np.asarray([1, 2, 3, 1, 3])

    assert np.array_equal(
        consecutive_horizon_indices(seeds, frames, 0, 3),
        np.asarray([0, 1, 2]),
    )
    assert consecutive_horizon_indices(seeds, frames, 1, 3) is None
    assert consecutive_horizon_indices(seeds, frames, 3, 2) is None


def test_simultaneous_log_score_matches_smallest_symmetric_margin():
    lower = np.asarray([[[[2.0, 0.0]]]])
    upper = np.asarray([[[[4.0, 4.0]]]])
    actual = np.asarray([[[[1.0, 8.0]]]])
    score = simultaneous_log_envelope_score(
        lower, upper, actual, np.ones_like(actual, dtype=bool))

    assert np.isclose(score.lower, np.log(2.0))
    assert np.isclose(score.upper, np.log(2.0))
    assert np.isclose(score.joint, np.log(2.0))
    assert score.audited_coefficient_count == 2


def test_zero_support_error_cannot_be_fixed_by_finite_log_margin():
    score = simultaneous_log_envelope_score(
        np.asarray([1.0]),
        np.asarray([2.0]),
        np.asarray([0.0]),
        np.asarray([True]),
    )

    assert np.isinf(score.lower)
    assert np.isinf(score.joint)
    assert score.lower_zero_support_failure_count == 1


def test_future_outcome_uses_each_frames_physical_coefficient():
    coefficient = np.zeros((2, 2, 2, 1), dtype=np.float64)
    coefficient[0, 0, 1, 0] = 1.0
    coefficient[1, 0, 1, 0] = 2.0
    selected = np.zeros((2, 2, 1), dtype=bool)
    selected[0, 1, 0] = True
    power = np.zeros((2, 2, 1), dtype=np.float64)
    power[:, 0, 0] = 0.5
    lower = 0.5 * coefficient
    upper = 2.0 * coefficient
    mask = np.broadcast_to(selected, coefficient.shape)

    outcome = evaluate_horizon_future_outcome(
        coefficient,
        selected,
        selected,
        power,
        power,
        np.zeros((2, 1)),
        np.ones((2, 1)),
        lower,
        upper,
        mask,
        false_alarm_probability=1.0e-6,
    )

    assert outcome.candidate_pd[1, 0] > outcome.candidate_pd[0, 0]
    assert np.allclose(outcome.candidate_pd, outcome.noop_pd)
    assert not outcome.candidate_lower_failure
    assert not outcome.noop_upper_failure
    assert np.all(outcome.target_no_harm)
    assert outcome.coefficient_score.joint == 0.0
