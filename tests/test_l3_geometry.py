"""Tests for L3 capability-sensitivity geometry (L3-G1)."""

import numpy as np
import pytest

from uav_isac.coordination.capability import (
    capability_gauge_pwl_lp_full,
    capability_geometry_gradient,
)
from uav_isac.coordination.pwl_pd import (
    chord_lower_bound,
    curvature_breakpoints,
)
from uav_isac.physical.detection import (
    minimum_deflection_for_detection_probability,
)

P_FA = 0.001
XI = (0.60, 0.70, 0.80, 3)


def _setup(rng, K=5, Q=4):
    uav = rng.uniform(0.0, 100.0, size=(K, 2))
    target = rng.uniform(0.0, 100.0, size=(Q, 2))
    owner = np.array([q % K for q in range(Q)])
    scale = 1e8  # Friis scale so per-watt gain is ~1..10 at R~50m

    def gain_of(p):
        rtx = np.linalg.norm(p[:, None, :] - target[None, :, :], axis=2)
        rrx = np.linalg.norm(p[owner] - target, axis=1)
        return scale / (rtx ** 2 * rrx[None, :] ** 2)

    gain = gain_of(uav)
    budget = np.full(K, 3.0)
    d_min = float(minimum_deflection_for_detection_probability(
        np.asarray([0.60]), P_FA)[0])
    d_max = float(np.max(np.sum(gain * budget[:, None], axis=0))) + 1.0
    bps = curvature_breakpoints(P_FA, d_min, d_max, epsilon=1e-3)
    cs, ci = chord_lower_bound(P_FA, bps)
    return uav, target, owner, gain_of, gain, budget, d_min, cs, ci


def _gamma_of(gain, budget, d_min, cs, ci):
    out = capability_gauge_pwl_lp_full(gain, budget, P_FA, XI, cs, ci, d_min)
    return out


def test_gradient_direction_decreases_gamma_one_step():
    rng = np.random.default_rng(60)
    for trial in range(10):
        uav, target, owner, gain_of, gain, budget, d_min, cs, ci = _setup(rng)
        out = _gamma_of(gain, budget, d_min, cs, ci)
        if out is None:
            continue
        gamma, p, pi = out
        grad = capability_geometry_gradient(gain, owner, uav, target, p, pi)
        if not np.any(grad != 0.0):
            continue
        # Trust-region step: small, capped step along the negative gradient.
        step_norm = np.linalg.norm(grad)
        alpha = 1.0 / max(step_norm, 1e-9) * 5.0  # cap the step to ~5 m
        delta = -alpha * grad
        # Cap individual moves to a trust radius.
        radius = 5.0
        if np.max(np.linalg.norm(delta, axis=1)) > radius:
            delta = delta / np.max(np.linalg.norm(delta, axis=1)) * radius
        new_uav = uav + delta
        new_gain = gain_of(new_uav)
        out2 = _gamma_of(new_gain, budget, d_min, cs, ci)
        if out2 is None:
            continue
        gamma_new = out2[0]
        # Strict decrease (allow small numerical tolerance).
        assert gamma_new <= gamma + 1e-6, (
            f"trial {trial}: gamma {gamma} -> {gamma_new} increased; "
            f"grad norm {step_norm}")


def test_gradient_matches_numerical_geometry():
    """Verify the Friis gradient against a finite difference of gamma w.r.t.
    a single UAV position (Tx and Rx/owner effects both present)."""
    rng = np.random.default_rng(61)
    uav, target, owner, gain_of, gain, budget, d_min, cs, ci = _setup(rng)
    out = _gamma_of(gain, budget, d_min, cs, ci)
    assert out is not None
    _, p, pi = out
    grad = capability_geometry_gradient(gain, owner, uav, target, p, pi)

    eps = 1e-2
    for k in (0, 2):
        for d in (0, 1):
            up = uav.copy()
            up[k, d] += eps
            down = uav.copy()
            down[k, d] -= eps
            gp = _gamma_of(gain_of(up), budget, d_min, cs, ci)
            gm = _gamma_of(gain_of(down), budget, d_min, cs, ci)
            if gp is None or gm is None:
                continue
            fd = (gp[0] - gm[0]) / (2.0 * eps)
            assert np.isclose(fd, grad[k, d], atol=5e-2), (
                f"uav {k} dim {d}: fd {fd} vs grad {grad[k,d]}")
