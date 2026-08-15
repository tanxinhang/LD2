import numpy as np
import pytest

from uav_isac.evaluation.episode_joint_conformal import (
    certified_feedback_decision,
    episode_max_scores,
    joint_overprediction_score,
    split_conformal_upper,
)


def test_joint_score_covers_paired_improvement_error():
    score = joint_overprediction_score(
        candidate_estimated=0.8,
        candidate_realized=0.7,
        noop_estimated=0.4,
        noop_realized=0.5,
    )
    assert score == pytest.approx(0.2)


def test_episode_collapse_does_not_count_repeated_resolves_as_independent():
    rows = [
        {
            "seed": 11,
            "causal_geometry_available": True,
            "causal_geometry_estimated_worst": 0.8,
            "causal_geometry_worst": 0.7,
            "causal_geometry_noop_estimated_worst": 0.5,
            "causal_geometry_calibration_score": 0.1,
            "causal_geometry_noop_calibration_score": 0.0,
            "deployed_worst": 0.5,
        },
        {
            "seed": 11,
            "causal_geometry_available": True,
            "causal_geometry_estimated_worst": 0.9,
            "causal_geometry_worst": 0.6,
            "causal_geometry_noop_estimated_worst": 0.5,
            "causal_geometry_calibration_score": 0.3,
            "causal_geometry_noop_calibration_score": 0.0,
            "deployed_worst": 0.5,
        },
    ]
    scores = episode_max_scores(rows)
    assert list(scores) == [11]
    assert scores[11] == pytest.approx(0.3)


def test_twenty_episode_five_percent_margin_uses_maximum():
    margin, coverage, rank = split_conformal_upper(
        np.arange(20, dtype=np.float64), alpha=0.05)
    assert margin == 19.0
    assert rank == 20
    assert coverage == 20.0 / 21.0


def test_feedback_gate_requires_strict_gain_after_all_costs():
    rejected = certified_feedback_decision(
        candidate_estimated=0.72,
        noop_estimated=0.60,
        joint_margin=0.10,
        qos_floor=0.60,
        switch_cost=0.01,
        bit_cost=0.01,
    )
    accepted = certified_feedback_decision(
        candidate_estimated=0.75,
        noop_estimated=0.60,
        joint_margin=0.10,
        qos_floor=0.60,
        switch_cost=0.01,
        bit_cost=0.01,
    )
    assert not rejected.accept
    assert accepted.accept


def test_feedback_gate_charges_physical_transport_energy():
    decision = certified_feedback_decision(
        candidate_estimated=0.72,
        noop_estimated=0.60,
        joint_margin=0.05,
        qos_floor=0.60,
        bit_cost=0.02,
        delay_cost=0.02,
        energy_cost=0.04,
    )
    assert not decision.accept
    assert decision.charged_cost == pytest.approx(0.08)
