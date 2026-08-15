import numpy as np

from uav_isac.evaluation.physics_interval_gate import (
    certified_physics_interval_decision,
)


def test_physics_interval_gate_accepts_simultaneous_safe_gain():
    result = certified_physics_interval_decision(
        candidate_lower=np.asarray([0.7, 0.8]),
        noop_lower=np.asarray([0.5, 0.7]),
        noop_upper=np.asarray([0.55, 0.75]),
        qos_floor=0.6,
    )

    assert result.accept
    assert np.all(result.target_safe)
    assert np.isclose(result.worst_delta_lower, 0.15)


def test_physics_interval_gate_rejects_target_loss_or_nonpositive_gain():
    target_loss = certified_physics_interval_decision(
        candidate_lower=np.asarray([0.59, 0.9]),
        noop_lower=np.asarray([0.8, 0.8]),
        noop_upper=np.asarray([0.85, 0.85]),
        qos_floor=0.6,
    )
    no_gain = certified_physics_interval_decision(
        candidate_lower=np.asarray([0.7, 0.8]),
        noop_lower=np.asarray([0.5, 0.7]),
        noop_upper=np.asarray([0.75, 0.9]),
        qos_floor=0.6,
    )

    assert not target_loss.accept
    assert not target_loss.target_safe[0]
    assert not no_gain.accept
