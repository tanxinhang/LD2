import json

import numpy as np
import pytest

from tools.calibrate_horizon_future_envelope import calibrate
from tools.summarize_paired_horizon_confirmation import (
    _two_sided_sign_p,
    _wilson,
)


def _audit(path, seeds, scores):
    payload = {
        "schema_version": 6,
        "seed_order": seeds,
        "horizon_diagnostic": {
            "steps": 3,
            "base_current_log_margin": 1.0e-5,
            "frozen_transition_residual_log_margin": 0.0,
            "uav_reachable_position_radius_m": [0.0, 2.5, 5.0],
            "uav_reachable_velocity_radius_mps": [0.0, 25.0, 25.0],
            "future_outcome_policy": (
                "post_decision_privileged_frozen_movement_action_tape"),
        },
        "horizon_future_calibration_diagnostic": {
            "score": "joint log score",
            "exchangeability_unit": "episode_seed",
            "future_information_used_by_controller": False,
            "finite_episode_scores": {
                str(seed): score for seed, score in scores.items()
            },
            "infinite_score_seeds": [],
        },
        "summary": {
            "horizon_future_outcome_event_count": 2,
            "horizon_future_coefficient_audit_count": 20,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_calibration_keeps_empty_event_episodes_as_zero_scores(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _audit(first, list(range(30)), {0: 0.1})
    _audit(second, list(range(30, 60)), {30: 0.2})

    result = calibrate([first, second], alpha=0.02)

    assert result["calibration_episode_count"] == 60
    assert result["rank_one_based"] == 60
    assert np.isclose(
        result["finite_sample_coverage_floor"], 60.0 / 61.0)
    assert result["frozen_transition_residual_log_margin"] == 0.2
    assert result["episode_scores"]["59"] == 0.0
    assert result["envelope_calibration_ready"]


def test_calibration_rejects_episode_overlap(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _audit(first, [1, 2], {})
    _audit(second, [2, 3], {})

    with pytest.raises(ValueError, match="overlap"):
        calibrate([first, second], alpha=0.1)


def test_episode_sign_test_and_zero_failure_wilson_are_independent_units():
    assert np.isclose(_two_sided_sign_p(7, 0), 0.015625)
    interval = _wilson(0, 288)
    assert interval[0] == 0.0
    assert 0.0 < interval[1] < 0.02
