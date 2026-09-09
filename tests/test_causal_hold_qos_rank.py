import numpy as np
import pytest

from tools.audit_causal_hold_qos_rank import (
    align_proposal_to_baseline_scale,
    hold_bottleneck_weights,
    residual_blend,
    seed_split,
)
from tools.audit_causal_hold_qos_rank import parse_args


def test_seed_split_is_ordered_and_disjoint() -> None:
    values = np.asarray([9, 9, 3, 3, 7, 7, 5, 5])
    train, validation, test = seed_split(
        values, train_count=2, validation_count=1, test_count=1)
    assert train == [9, 3]
    assert validation == [7]
    assert test == [5]
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)


def test_seed_split_rejects_oversubscription() -> None:
    with pytest.raises(ValueError, match="requested 4 seeds"):
        seed_split(
            np.asarray([1, 2, 3]),
            train_count=2,
            validation_count=1,
            test_count=1,
        )


def test_hold_bottleneck_weights_emphasize_weak_target() -> None:
    hold = np.zeros((1, 2, 2, 2), dtype=np.float64)
    selected = np.zeros_like(hold, dtype=bool)
    selected[0, 0, 1, 0] = True
    selected[0, 1, 0, 1] = True
    hold[0, 0, 1, 0] = 100.0
    hold[0, 1, 0, 1] = 0.0
    weight = hold_bottleneck_weights(
        hold,
        selected,
        p_fa=0.001,
        qos_floor=0.60,
        strength=3.0,
    )
    assert weight.shape == (1, 2)
    assert weight[0, 1] > weight[0, 0]


def test_residual_blend_zero_is_exact_baseline() -> None:
    baseline = np.asarray([[[[1.0], [3.0]], [[2.0], [0.0]]]])
    proposal = np.asarray([[[[9.0], [1.0]], [[5.0], [0.0]]]])
    admitted = np.asarray([[[[True], [True]], [[True], [False]]]])
    aligned = align_proposal_to_baseline_scale(
        baseline, proposal, admitted)
    np.testing.assert_array_equal(
        residual_blend(baseline, aligned, 0.0), baseline)
    # At alpha=1 the proposal ordering is retained but the baseline score
    # distribution (1, 2, 3) is preserved.
    final = residual_blend(baseline, aligned, 1.0)
    np.testing.assert_array_equal(
        np.sort(final[admitted]), np.asarray([1.0, 2.0, 3.0]))
    assert final[0, 0, 0, 0] > final[0, 1, 0, 0] > final[0, 0, 1, 0]


def test_residual_training_mode_is_explicit() -> None:
    args = parse_args([
        "--trace", "trace.npz",
        "--student-checkpoint", "student.pt",
        "--output", "summary.json",
        "--training-mode", "baseline_residual",
    ])
    assert args.training_mode == "baseline_residual"
