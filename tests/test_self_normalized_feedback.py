import numpy as np

from uav_isac.evaluation.self_normalized_feedback import (
    certified_targetwise_feedback_decision,
    episode_max_normalized_scores,
    targetwise_normalized_score,
)


def test_targetwise_score_is_zero_for_conservative_predictions():
    score = targetwise_normalized_score(
        candidate_estimated=np.array([0.6, 0.7]),
        candidate_realized=np.array([0.7, 0.8]),
        noop_estimated=np.array([0.5, 0.6]),
        noop_realized=np.array([0.5, 0.6]),
        absolute_uncertainty=np.array([0.1, 0.1]),
        noop_upper_uncertainty=np.array([0.1, 0.1]),
    )
    assert score == 0.0


def test_targetwise_score_uses_largest_normalized_component():
    score = targetwise_normalized_score(
        candidate_estimated=np.array([0.8, 0.7]),
        candidate_realized=np.array([0.6, 0.6]),
        noop_estimated=np.array([0.5, 0.5]),
        noop_realized=np.array([0.5, 0.5]),
        absolute_uncertainty=np.array([0.1, 0.1]),
        noop_upper_uncertainty=np.array([0.1, 0.1]),
    )
    assert np.isclose(score, 2.0)


def test_episode_scores_collapse_frames_by_seed():
    base = {
        "causal_routed_available": True,
        "causal_routed_estimated_pd": [0.8],
        "causal_routed_realized_pd": [0.7],
        "causal_routed_noop_estimated_pd": [0.5],
        "deployed_pd": [0.5],
        "causal_routed_absolute_uncertainty": [0.1],
        "causal_routed_noop_upper_uncertainty": [0.1],
    }
    rows = [
        {**base, "seed": 1},
        {**base, "seed": 1,
         "causal_routed_realized_pd": [0.6]},
        {**base, "seed": 2,
         "causal_routed_realized_pd": [0.8]},
    ]
    scores = episode_max_normalized_scores(rows)
    assert np.isclose(scores[1], 2.0)
    assert np.isclose(scores[2], 0.0)


def test_targetwise_gate_requires_every_target_and_net_gain():
    accepted = certified_targetwise_feedback_decision(
        candidate_estimated=np.array([0.75, 0.70]),
        noop_estimated=np.array([0.60, 0.55]),
        absolute_uncertainty=np.array([0.02, 0.02]),
        noop_upper_uncertainty=np.array([0.02, 0.02]),
        normalized_margin=1.0,
        qos_floor=0.60,
        switch_cost=0.01,
    )
    assert accepted.accept
    assert np.all(accepted.target_safe)

    unsafe = certified_targetwise_feedback_decision(
        candidate_estimated=np.array([0.75, 0.50]),
        noop_estimated=np.array([0.60, 0.55]),
        absolute_uncertainty=np.array([0.02, 0.02]),
        noop_upper_uncertainty=np.array([0.02, 0.02]),
        normalized_margin=1.0,
        qos_floor=0.60,
    )
    assert not unsafe.accept
    assert not unsafe.target_safe[1]
