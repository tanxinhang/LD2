import numpy as np
import pytest

from uav_isac.evaluation.expert_arbitration import (
    blend_target_edge_values,
    conservative_target_expert_choice,
    target_support_ceiling,
)


def test_support_ceiling_uses_top_b_transmitters_and_best_receiver():
    values = np.zeros((3, 3, 2), dtype=np.float64)
    mask = np.ones_like(values, dtype=bool)
    values[0, 1, 0] = 2.0
    values[2, 1, 0] = 3.0
    values[0, 2, 0] = 4.0
    values[1, 2, 0] = 0.5
    values[0, 1, 1] = 7.0
    score = target_support_ceiling(values, mask, pair_limit=2)
    np.testing.assert_allclose(score, [5.0, 7.0])


def test_support_ceiling_ignores_diagonal_mask_and_negative_values():
    values = np.zeros((2, 2, 1), dtype=np.float64)
    mask = np.ones_like(values, dtype=bool)
    values[0, 0, 0] = 100.0
    values[0, 1, 0] = -2.0
    values[1, 0, 0] = 1.5
    np.testing.assert_allclose(
        target_support_ceiling(values, mask, pair_limit=2), [1.5])


def test_conservative_choice_keeps_primary_on_tie_and_margin_boundary():
    primary = np.asarray([1.0, 2.0, 0.0])
    alternate = np.asarray([1.0, 2.2, 0.1])
    choice = conservative_target_expert_choice(
        primary, alternate, relative_margin=0.1, absolute_margin=0.1)
    np.testing.assert_array_equal(choice, [False, False, False])


def test_target_blend_never_mixes_edges_inside_one_target():
    primary = np.zeros((2, 2, 3), dtype=np.float64)
    alternate = np.ones_like(primary)
    blended = blend_target_edge_values(
        primary, alternate, np.asarray([False, True, False]))
    np.testing.assert_array_equal(blended[..., 0], 0.0)
    np.testing.assert_array_equal(blended[..., 1], 1.0)
    np.testing.assert_array_equal(blended[..., 2], 0.0)


def test_nonfinite_scores_fail_closed():
    with pytest.raises(ValueError, match="finite"):
        conservative_target_expert_choice(
            np.asarray([0.0]), np.asarray([np.inf]))

