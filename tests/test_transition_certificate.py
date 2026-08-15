import numpy as np
import pytest

from uav_isac.evaluation.transition_certificate import (
    event_max_standardized_underestimate,
    safe_reconfiguration_is_certified,
    split_conformal_upper_multiplier,
    transition_candidate_is_certified,
    transition_loss_upper_bound,
)


def test_event_score_keeps_candidates_and_horizons_dependent():
    predicted = np.asarray([
        [[-0.2, -0.1], [-0.1, -0.1]],
        [[-0.3, -0.2], [-0.2, -0.2]],
    ])
    observed = predicted.copy()
    observed[1, 0, 1] += 0.30
    scale = np.full_like(predicted, 0.10)
    score = event_max_standardized_underestimate(
        predicted, observed, scale)
    assert np.isclose(score, 3.0)


def test_split_conformal_multiplier_uses_finite_sample_correction():
    scores = np.arange(30, dtype=np.float64)
    # ceil(31*0.95)=30: the maximum is required.
    assert split_conformal_upper_multiplier(
        scores, miscoverage=0.05) == 29.0
    # ceil(31*0.99)=31: 30 events cannot resolve 99% coverage.
    assert np.isinf(split_conformal_upper_multiplier(
        scores, miscoverage=0.01))


def test_missing_required_event_data_and_infinite_scores_fail_closed():
    score = event_max_standardized_underestimate(
        np.asarray([0.0, 0.0]),
        np.asarray([0.0, np.nan]),
        np.asarray([0.1, 0.1]),
    )
    assert np.isinf(score)
    # At 5% with 19 events the maximum is the conformal order statistic, so
    # one required missing event must make the frozen threshold infinite.
    scores = np.zeros(19, dtype=np.float64)
    scores[-1] = np.inf
    assert np.isinf(split_conformal_upper_multiplier(
        scores, miscoverage=0.05))
    with pytest.raises(ValueError):
        split_conformal_upper_multiplier(
            np.asarray([0.0, np.nan]), miscoverage=0.5)


def test_transition_decision_is_joint_and_fail_closed():
    predicted = np.asarray([-0.20, -0.10])
    scale = np.asarray([0.05, 0.05])
    np.testing.assert_allclose(
        transition_loss_upper_bound(
            predicted, scale, multiplier=1.0),
        [-0.15, -0.05],
    )
    assert transition_candidate_is_certified(
        predicted, scale, multiplier=1.0)
    assert not transition_candidate_is_certified(
        predicted, scale, multiplier=2.5)
    assert not transition_candidate_is_certified(
        predicted, scale, multiplier=float("inf"))


def test_event_score_rejects_nonpositive_scale():
    with pytest.raises(ValueError):
        event_max_standardized_underestimate(
            np.asarray([0.0]),
            np.asarray([0.0]),
            np.asarray([0.0]),
        )


def test_safe_reconfiguration_is_an_intersection_not_a_weighted_sum():
    predicted = np.asarray([-0.20, -0.10])
    scale = np.asarray([0.01, 0.01])
    assert safe_reconfiguration_is_certified(
        predicted,
        scale,
        multiplier=1.0,
        structural_feasible=True,
        commit_feasible=True,
    )
    assert not safe_reconfiguration_is_certified(
        predicted,
        scale,
        multiplier=1.0,
        structural_feasible=True,
        commit_feasible=False,
    )
    assert not safe_reconfiguration_is_certified(
        predicted,
        scale,
        multiplier=1.0,
        structural_feasible=False,
        commit_feasible=True,
    )
