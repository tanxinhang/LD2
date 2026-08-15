"""Unit tests for the certified staleness bounds."""

import numpy as np
import pytest

from uav_isac.coordination.power_staleness import (
    dd_gate_crossing,
    gain_geometry_drift_bound,
    held_power_staleness_bound,
    maxmin_value_perturbation_bound,
)
from uav_isac.coordination.maxmin_power import solve_fixed_structure_maxmin_power_lp


def test_value_perturbation_bound_is_valid():
    rng = np.random.default_rng(21)
    for _ in range(200):
        K, Q = 5, 4
        gain = rng.uniform(0.1, 3.0, size=(K, Q))
        budget = rng.uniform(0.1, 1.0, size=K)
        noise = rng.uniform(-0.5, 0.5, size=(K, Q))
        gain_new = np.maximum(gain + noise, 0.0)
        t_old = solve_fixed_structure_maxmin_power_lp(
            gain, budget).worst_deflection
        t_new = solve_fixed_structure_maxmin_power_lp(
            gain_new, budget).worst_deflection
        bound = maxmin_value_perturbation_bound(gain, gain_new, budget)
        assert abs(t_new - t_old) <= bound + 1e-9, (
            f"Lipschitz bound violated: |{t_new}-{t_old}|={abs(t_new-t_old)} "
            f"> bound={bound}")


def test_held_power_staleness_bound_is_valid():
    rng = np.random.default_rng(22)
    for _ in range(200):
        K, Q = 5, 4
        gain = rng.uniform(0.1, 3.0, size=(K, Q))
        budget = rng.uniform(0.1, 1.0, size=K)
        noise = rng.uniform(-0.5, 0.5, size=(K, Q))
        gain_new = np.maximum(gain + noise, 0.0)
        held = solve_fixed_structure_maxmin_power_lp(gain, budget)
        t_new = solve_fixed_structure_maxmin_power_lp(
            gain_new, budget).worst_deflection
        # Held power applied to the new gain.
        t_held = float(np.min(np.sum(gain_new * held.power_w, axis=0)))
        bound = held_power_staleness_bound(gain, gain_new, budget)
        assert t_new - t_held <= bound + 1e-9, (
            f"staleness bound violated: {t_new}-{t_held}={t_new-t_held} "
            f"> bound={bound}")


def test_geometry_drift_bound_is_valid_for_friis_gain():
    rng = np.random.default_rng(23)
    # Fixed owner receiver j = q % K.  Gain follows a_iq ~ 1/(R_iq^2 R_jq^2).
    K, Q = 6, 6
    scale = 10.0
    pos = rng.uniform(0.0, 800.0, size=(K, 2))
    target = rng.uniform(0.0, 800.0, size=(Q, 2))
    owner = np.array([q % K for q in range(Q)])

    def distances(p):
        rtx = np.linalg.norm(p[:, None, :] - target[None, :, :], axis=2)
        rrx = np.linalg.norm(p[owner] - target, axis=1)  # (Q,)
        return rtx, rrx

    def gain(p):
        rtx, rrx = distances(p)
        return scale / (rtx ** 2 * rrx[None, :] ** 2)

    delta = 1.0  # max displacement bound
    rtx, rrx = distances(pos)
    g = gain(pos)
    per_iq = gain_geometry_drift_bound(
        g, rtx, owner, rrx,
        np.full(K, delta), np.full(Q, delta),
    )
    # Perturb each node within the delta ball and check the bound.
    for _ in range(100):
        dp = rng.uniform(-delta, delta, size=(K, 2))
        g_new = gain(pos + dp)  # targets static for simplicity
        actual = np.abs(g_new - g)
        assert np.all(actual <= per_iq + 1e-6), (
            "geometry-drift bound violated")


def test_geometry_drift_bound_rejects_bad_input():
    g = np.ones((3, 3))
    owner = np.array([0, 1, 2])
    with pytest.raises(ValueError):
        gain_geometry_drift_bound(
            g, np.zeros((3, 3)), owner, np.zeros(3),
            np.zeros(3), np.zeros(3))  # zero distance invalid
    with pytest.raises(ValueError):
        gain_geometry_drift_bound(
            g, np.ones((3, 3)), owner, np.ones(3),
            np.ones(2), np.ones(3))  # wrong K


def test_dd_gate_crossing():
    g_min = 0.5
    old = np.array([[[0.6, 0.4]], [[0.3, 0.7]]])
    new = np.array([[[0.4, 0.4]], [[0.3, 0.6]]])
    crossing = dd_gate_crossing(old, new, g_min)
    assert bool(crossing[0, 0, 0]) is True   # 0.6 -> 0.4 crosses down
    assert bool(crossing[0, 0, 1]) is False  # stays below
    assert bool(crossing[1, 0, 0]) is False  # stays below
    assert bool(crossing[1, 0, 1]) is False  # 0.7 -> 0.6 stays above
