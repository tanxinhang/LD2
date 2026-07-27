import numpy as np

from uav_isac.evaluation.temporal_credit_audit import (
    compute_frame_and_macro_gae,
    summarize_temporal_credit_proxy,
)


def test_macro_gae_preserves_within_hold_micro_discount():
    rewards = np.arange(1.0, 11.0).reshape(10, 1)
    boundaries, frame, macro = compute_frame_and_macro_gae(
        rewards, gamma=0.9, gae_lambda=0.0, interval=5)
    np.testing.assert_array_equal(boundaries, np.array([0, 5]))
    assert frame[0, 0] == 1.0
    expected = sum((0.9 ** i) * (i + 1.0) for i in range(5))
    np.testing.assert_allclose(macro[0, 0], expected, rtol=1e-12)


def test_temporal_proxy_requires_independent_episodes_for_gate():
    rewards = np.ones((15, 2), dtype=np.float64)
    pd = np.linspace(0.1, 0.9, 30).reshape(15, 2)
    summary = summarize_temporal_credit_proxy(
        [rewards], [pd], gamma=0.99, gae_lambda=0.95,
        interval=5, bootstrap_samples=4)
    assert summary["eval_temporal_proxy_boundary_count"] == 2
    assert summary["eval_temporal_proxy_gate_pass"] is False
