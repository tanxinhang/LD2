"""Causal finite-horizon joint-plan commitments for slow ISAC control."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


def _immutable_copy(value: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CausalJointPlanCommitment:
    """Digest-bound H-step commitment formed from one current Actor action.

    The adapter is a zero-order hold: it repeats the currently available
    mobility and RF action.  It therefore consumes no future observation or
    future Actor evaluation.  A slow controller may replace only the mobility
    plan after the complete joint plan has entered the prepare/commit protocol.
    """

    decision_frame: int
    movement_plan_m: np.ndarray
    communication_power_plan_w: np.ndarray
    sensing_power_plan_w: np.ndarray
    source: str = "current_actor_zero_order_hold"

    def __post_init__(self) -> None:
        movement = np.asarray(self.movement_plan_m, dtype=np.float64)
        communication = np.asarray(
            self.communication_power_plan_w, dtype=np.float64)
        sensing = np.asarray(self.sensing_power_plan_w, dtype=np.float64)
        if (
            movement.ndim != 3 or movement.shape[2] != 2
            or communication.ndim != 2 or sensing.ndim != 3
            or movement.shape[:2] != communication.shape
            or sensing.shape[:2] != communication.shape
            or movement.shape[0] < 1
            or np.any(~np.isfinite(movement))
            or np.any(~np.isfinite(communication))
            or np.any(~np.isfinite(sensing))
            or np.any(communication < 0.0) or np.any(sensing < 0.0)
            or np.any(
                communication + np.sum(sensing, axis=2) > 1.0 + 1.0e-12)
        ):
            raise ValueError("joint plan dimensions or RF budget are invalid")
        if int(self.decision_frame) < 0:
            raise ValueError("decision_frame must be non-negative")
        if not str(self.source):
            raise ValueError("joint-plan source must be non-empty")
        object.__setattr__(
            self, "movement_plan_m",
            _immutable_copy(movement, dtype=np.float64))
        object.__setattr__(
            self, "communication_power_plan_w",
            _immutable_copy(communication, dtype=np.float64))
        object.__setattr__(
            self, "sensing_power_plan_w",
            _immutable_copy(sensing, dtype=np.float64))

    @property
    def horizon_steps(self) -> int:
        return int(self.movement_plan_m.shape[0])

    @property
    def digest(self) -> str:
        payload = hashlib.sha256()
        payload.update(np.asarray(
            [int(self.decision_frame), self.horizon_steps],
            dtype="<i8").tobytes())
        payload.update(str(self.source).encode("utf-8"))
        for value in (
            self.movement_plan_m,
            self.communication_power_plan_w,
            self.sensing_power_plan_w,
        ):
            array = np.ascontiguousarray(value, dtype="<f8")
            payload.update(np.asarray(array.shape, dtype="<i8").tobytes())
            payload.update(array.tobytes())
        return payload.hexdigest()[:16]


def commit_zero_order_hold_joint_plan(
    movement_command_m: np.ndarray,
    communication_power_w: np.ndarray,
    sensing_power_w: np.ndarray,
    *,
    decision_frame: int,
    horizon_steps: int,
) -> CausalJointPlanCommitment:
    """Repeat one current joint action into a causal finite-horizon contract."""
    movement = np.asarray(movement_command_m, dtype=np.float64)
    communication = np.asarray(communication_power_w, dtype=np.float64)
    sensing = np.asarray(sensing_power_w, dtype=np.float64)
    horizon = int(horizon_steps)
    if (
        movement.ndim != 2 or movement.shape[1] != 2
        or communication.shape != (movement.shape[0],)
        or sensing.ndim != 2 or sensing.shape[0] != movement.shape[0]
        or horizon < 1
    ):
        raise ValueError("current joint action dimensions are inconsistent")
    return CausalJointPlanCommitment(
        decision_frame=int(decision_frame),
        movement_plan_m=np.repeat(movement[None, ...], horizon, axis=0),
        communication_power_plan_w=np.repeat(
            communication[None, ...], horizon, axis=0),
        sensing_power_plan_w=np.repeat(sensing[None, ...], horizon, axis=0),
    )


def commit_movement_plan_with_held_rf(
    movement_plan_m: np.ndarray,
    communication_power_w: np.ndarray,
    sensing_power_w: np.ndarray,
    *,
    decision_frame: int,
    source: str,
) -> CausalJointPlanCommitment:
    """Commit an explicit causal movement plan while holding current RF."""
    movement = np.asarray(movement_plan_m, dtype=np.float64)
    communication = np.asarray(communication_power_w, dtype=np.float64)
    sensing = np.asarray(sensing_power_w, dtype=np.float64)
    if movement.ndim != 3 or movement.shape[2] != 2:
        raise ValueError("movement_plan_m must have shape (H,K,2)")
    if (
        communication.shape != (movement.shape[1],)
        or sensing.ndim != 2
        or sensing.shape[0] != movement.shape[1]
    ):
        raise ValueError("current RF action dimensions are inconsistent")
    return CausalJointPlanCommitment(
        decision_frame=int(decision_frame),
        movement_plan_m=movement,
        communication_power_plan_w=np.repeat(
            communication[None, :], movement.shape[0], axis=0),
        sensing_power_plan_w=np.repeat(
            sensing[None, ...], movement.shape[0], axis=0),
        source=str(source),
    )
