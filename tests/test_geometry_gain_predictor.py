import numpy as np

from uav_isac.coordination.geometry_gain_predictor import (
    predict_edge_gain_tensor_from_geometry,
    predict_fixed_owner_gain_from_geometry,
)


def _coefficient(uav, target, constant=7.0):
    ranges = np.linalg.norm(uav[:, None, :] - target[None, :, :], axis=-1)
    K, Q = ranges.shape
    result = np.zeros((K, K, Q), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            if i != j:
                result[i, j] = constant / (
                    ranges[i] ** 2 * ranges[j] ** 2)
    return result


def test_geometry_ratio_is_exact_for_fixed_waveform_radar_law():
    previous_uav = np.asarray([[0.0, 0.0, 10.0], [20.0, 0.0, 10.0]])
    current_uav = np.asarray([[2.0, 0.0, 10.0], [18.0, 0.0, 10.0]])
    previous_target = np.asarray([[10.0, 10.0, 0.0]])
    current_target = np.asarray([[11.0, 10.0, 0.0]])
    previous = _coefficient(previous_uav, previous_target)
    expected = _coefficient(current_uav, current_target)
    result = predict_fixed_owner_gain_from_geometry(
        previous,
        previous_uav,
        current_uav,
        previous_target,
        current_target,
        [(0, 1, 0)],
    )
    assert result.gain_per_watt[0, 0] == expected[0, 1, 0]
    assert result.direct_edge_count == 1
    assert result.fallback_edge_count == 0


def test_new_edge_uses_robust_target_invariant():
    uav = np.asarray([
        [0.0, 0.0, 10.0],
        [20.0, 0.0, 10.0],
        [10.0, 20.0, 10.0],
    ])
    target = np.asarray([[10.0, 10.0, 0.0]])
    previous = _coefficient(uav, target)
    expected = previous[2, 1, 0]
    previous[2, 1, 0] = 0.0
    result = predict_fixed_owner_gain_from_geometry(
        previous, uav, uav, target, target, [(2, 1, 0)])
    assert result.gain_per_watt[2, 0] == expected
    assert result.fallback_edge_count == 1


def test_missing_calibration_fails_closed_to_zero():
    coefficient = np.zeros((2, 2, 1), dtype=np.float64)
    uav = np.asarray([[0.0, 0.0, 10.0], [20.0, 0.0, 10.0]])
    target = np.asarray([[10.0, 10.0]])
    result = predict_fixed_owner_gain_from_geometry(
        coefficient, uav, uav, target, target, [(0, 1, 0)])
    assert not result.calibration_available
    assert result.gain_per_watt[0, 0] == 0.0


def test_current_support_gate_is_fail_closed():
    uav = np.asarray([[0.0, 0.0, 10.0], [20.0, 0.0, 10.0]])
    target = np.asarray([[10.0, 10.0, 0.0]])
    coefficient = _coefficient(uav, target)
    support = np.zeros_like(coefficient, dtype=bool)
    result = predict_fixed_owner_gain_from_geometry(
        coefficient,
        uav,
        uav,
        target,
        target,
        [(0, 1, 0)],
        current_support=support,
    )
    assert result.gain_per_watt[0, 0] == 0.0


def test_full_edge_predictor_supports_causal_structural_candidates():
    previous_uav = np.asarray([
        [0.0, 0.0, 10.0], [20.0, 0.0, 10.0], [10.0, 20.0, 10.0]])
    current_uav = previous_uav + np.asarray([1.0, 0.0, 0.0])
    target = np.asarray([[10.0, 10.0, 0.0]])
    previous = _coefficient(previous_uav, target)
    previous[2, 1, 0] = 0.0
    support = np.ones_like(previous, dtype=bool)
    expected = _coefficient(current_uav, target)
    result = predict_edge_gain_tensor_from_geometry(
        previous,
        previous_uav,
        current_uav,
        target,
        target,
        current_support=support,
    )
    assert result.coefficient_per_watt[0, 1, 0] == expected[0, 1, 0]
    assert result.coefficient_per_watt[2, 1, 0] == expected[2, 1, 0]
    assert result.direct_edge_count == 5
    assert result.fallback_edge_count == 1
