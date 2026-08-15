import numpy as np
import pytest

from uav_isac.evaluation.n5_counterfactual_audit import (
    cumulative_detection_deficit,
    protected_target_pd,
    tail_deficit_dominates,
    targetwise_no_op_safe,
)


def test_targetwise_safety_preserves_below_floor_and_floor_caps_safe_target():
    baseline = np.asarray([0.40, 0.80, 0.60], dtype=np.float64)
    np.testing.assert_allclose(
        protected_target_pd(baseline, 0.60),
        [0.40, 0.60, 0.60],
    )
    assert targetwise_no_op_safe(
        np.asarray([0.40, 0.61, 0.60]),
        baseline,
        p_d_floor=0.60,
    )
    assert not targetwise_no_op_safe(
        np.asarray([0.39, 0.90, 0.90]),
        baseline,
        p_d_floor=0.60,
    )
    assert not targetwise_no_op_safe(
        np.asarray([0.50, 0.59, 0.90]),
        baseline,
        p_d_floor=0.60,
    )


def test_targetwise_safety_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        targetwise_no_op_safe(
            np.asarray([0.5]),
            np.asarray([0.5, 0.6]),
            p_d_floor=0.60,
        )


def test_tail_deficit_dominance_protects_all_worst_k_tails():
    baseline = np.asarray([0.20, 0.40, 0.90], dtype=np.float64)
    np.testing.assert_allclose(
        cumulative_detection_deficit(baseline, p_d_floor=0.60),
        [0.40, 0.60, 0.60],
    )
    assert tail_deficit_dominates(
        np.asarray([0.30, 0.35, 0.90]),
        baseline,
        p_d_floor=0.60,
    )
    assert not tail_deficit_dominates(
        np.asarray([0.30, 0.25, 0.90]),
        baseline,
        p_d_floor=0.60,
    )


def test_tail_deficit_dominance_allows_fair_label_redistribution():
    baseline = np.asarray([
        0.27558999265016626,
        1.0,
        1.0,
        0.5792097359435214,
        0.1763891989878697,
        1.0,
    ])
    candidate = np.asarray([
        1.0,
        1.0,
        0.9983543626868194,
        0.4990838340567189,
        0.7708497607398734,
        1.0,
    ])
    assert tail_deficit_dominates(
        candidate,
        baseline,
        p_d_floor=0.60,
    )
    assert not targetwise_no_op_safe(
        candidate,
        baseline,
        p_d_floor=0.60,
    )
