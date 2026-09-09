import numpy as np

from uav_isac.evaluation.pareto_no_regret import (
    componentwise_hold_pareto_gate,
)


def test_componentwise_gate_rejects_target_tradeoff() -> None:
    data = {
        "privileged_d_eff": np.zeros((2, 2, 2, 2), dtype=np.float64),
        "p_fa": np.asarray([0.001]),
        "episode": np.asarray([0, 0]),
        "seed": np.asarray([7, 7]),
        "frame": np.asarray([0, 1]),
        "p0_resolved": np.asarray([1, 0], dtype=np.uint8),
    }
    baseline = np.zeros((2, 2, 2, 2), dtype=bool)
    proposal = np.zeros_like(baseline)
    baseline[:, 0, 1, :] = True
    proposal[:, 1, 0, :] = True
    data["privileged_d_eff"][:, 0, 1, :] = np.asarray([2.0, 2.0])
    data["privileged_d_eff"][:, 1, 0, :] = np.asarray([3.0, 1.0])
    gated, audit = componentwise_hold_pareto_gate(
        data, baseline, proposal)
    np.testing.assert_array_equal(gated, baseline)
    assert audit["accepted_segments"] == 0


def test_componentwise_gate_accepts_pareto_improvement() -> None:
    data = {
        "privileged_d_eff": np.zeros((2, 2, 2, 2), dtype=np.float64),
        "p_fa": np.asarray([0.001]),
        "episode": np.asarray([0, 0]),
        "seed": np.asarray([7, 7]),
        "frame": np.asarray([0, 1]),
        "p0_resolved": np.asarray([1, 0], dtype=np.uint8),
    }
    baseline = np.zeros((2, 2, 2, 2), dtype=bool)
    proposal = np.zeros_like(baseline)
    baseline[:, 0, 1, :] = True
    proposal[:, 1, 0, :] = True
    data["privileged_d_eff"][:, 0, 1, :] = np.asarray([2.0, 2.0])
    data["privileged_d_eff"][:, 1, 0, :] = np.asarray([3.0, 2.0])
    gated, audit = componentwise_hold_pareto_gate(
        data, baseline, proposal)
    np.testing.assert_array_equal(gated, proposal)
    assert audit["accepted_segments"] == 1
