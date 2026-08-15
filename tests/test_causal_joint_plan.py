import numpy as np
import pytest

from uav_isac.coordination.causal_joint_plan import (
    CausalJointPlanCommitment,
    commit_movement_plan_with_held_rf,
    commit_zero_order_hold_joint_plan,
)


def test_zero_order_hold_uses_only_current_joint_action():
    movement = np.asarray([[1.0, -0.5], [0.25, 0.75]])
    communication = np.asarray([0.2, 0.3])
    sensing = np.asarray([[0.4, 0.4], [0.2, 0.5]])
    commitment = commit_zero_order_hold_joint_plan(
        movement,
        communication,
        sensing,
        decision_frame=7,
        horizon_steps=3,
    )
    assert commitment.movement_plan_m.shape == (3, 2, 2)
    assert np.array_equal(commitment.movement_plan_m[2], movement)
    assert np.array_equal(
        commitment.communication_power_plan_w[1], communication)
    assert np.array_equal(commitment.sensing_power_plan_w[1], sensing)
    assert commitment.source == "current_actor_zero_order_hold"


def test_joint_plan_is_immutable_and_digest_binds_rf_action():
    movement = np.zeros((1, 2))
    communication = np.asarray([0.2])
    sensing = np.asarray([[0.3, 0.5]])
    first = commit_zero_order_hold_joint_plan(
        movement, communication, sensing,
        decision_frame=1, horizon_steps=3)
    second = CausalJointPlanCommitment(
        decision_frame=1,
        movement_plan_m=first.movement_plan_m,
        communication_power_plan_w=np.full((3, 1), 0.25),
        sensing_power_plan_w=np.repeat(
            np.asarray([[[0.3, 0.45]]]), 3, axis=0),
    )
    assert first.digest != second.digest
    assert len(first.digest) == 16
    with pytest.raises(ValueError):
        first.movement_plan_m[0, 0, 0] = 1.0


def test_joint_plan_rejects_power_budget_violation():
    with pytest.raises(ValueError, match="RF budget"):
        commit_zero_order_hold_joint_plan(
            np.zeros((1, 2)),
            np.asarray([0.6]),
            np.asarray([[0.3, 0.2]]),
            decision_frame=0,
            horizon_steps=3,
        )


def test_explicit_movement_plan_holds_current_rf_exactly():
    movement = np.arange(24, dtype=np.float64).reshape(3, 4, 2) / 100.0
    communication = np.full(4, 0.2)
    sensing = np.full((4, 3), 0.1)
    plan = commit_movement_plan_with_held_rf(
        movement,
        communication,
        sensing,
        decision_frame=8,
        source="equivariant_test",
    )
    assert np.array_equal(plan.movement_plan_m, movement)
    assert np.array_equal(
        plan.communication_power_plan_w,
        np.repeat(communication[None], 3, axis=0),
    )
    assert np.array_equal(
        plan.sensing_power_plan_w,
        np.repeat(sensing[None], 3, axis=0),
    )
