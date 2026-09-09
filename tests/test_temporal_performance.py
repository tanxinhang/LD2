import numpy as np
import pytest

from uav_isac.evaluation.temporal_performance import (
    convergence_profile,
    rolling_multiframe_detection,
    summarize_closed_loop_convergence,
    summarize_multiframe_detection,
)
from uav_isac.physical.detection import compute_detection_probabilities


def test_multiframe_detection_adds_deflection_not_probability():
    history = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    actual = rolling_multiframe_detection(history, 0.001, window=2)
    expected = np.stack([
        compute_detection_probabilities(np.array([4.0, 6.0]), 0.001),
        compute_detection_probabilities(np.array([8.0, 10.0]), 0.001),
    ])
    np.testing.assert_allclose(actual, expected)


def test_convergence_profile_distinguishes_first_hit_from_sustained():
    values = [0.2, 0.7, 0.4, 0.72, 0.73, 0.74, 0.74, 0.74]
    profile = convergence_profile(
        values,
        threshold=0.7,
        rolling_window=1,
        terminal_window=3,
        patience=3,
        absolute_band=0.01,
        std_tolerance=0.01,
    )
    assert profile["first_hit_frame"] == 2
    assert profile["sustained_threshold_frame"] == 4
    assert profile["converged"]


def test_joint_qos_reports_tail_hold_instead_of_only_terminal_mean():
    pd = np.array([
        [0.9, 0.8, 0.7],
        [0.9, 0.8, 0.5],
        [0.9, 0.8, 0.7],
        [0.9, 0.8, 0.7],
    ])
    result = summarize_closed_loop_convergence(
        pd, rolling_window=1, terminal_window=4, patience=2)
    assert result["joint_qos"]["first_hit_frame"] == 1
    assert result["joint_qos"]["sustained_frame"] == 3
    assert result["joint_qos"]["terminal_hold_rate"] == pytest.approx(0.75)


def test_multiframe_summary_keeps_window_one_as_single_frame_baseline():
    deflection = np.full((5, 2), 1.0)
    result = summarize_multiframe_detection(
        deflection, 0.001, windows=(1, 2, 10), tail_window=5)
    assert set(result["windows"]) == {"1", "2"}
    assert result["windows"]["2"]["worst"] > result["windows"]["1"]["worst"]

