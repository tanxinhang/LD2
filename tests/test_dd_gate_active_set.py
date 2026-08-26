"""P0-3 (advice 014): local active-set certificate for the DD gate (C4).

The DD gate ``a_ijq = 1[g_dd >= g_min] * tilde a_ijq`` is a support
discontinuity.  Instead of smoothing the gate, we certify *locally*: with a
Lipschitz constant ``L_g`` of ``g_dd`` w.r.t. position displacement
(velocity held, i.e. the within-frame trust region), every edge with

    |g_dd,ijq - g_min| > L_g * r

has a CONSTANT indicator inside a radius-``r`` trust region (``a_ijq`` is
locally smooth/Lipschitz there); edges failing the condition are
gate-uncertain and the certificate fails closed (re-evaluate exactly at the
moved geometry, never assume smoothness).  Across frames the per-frame
velocity model (v = step/dt) makes the Doppler alignment swing by O(1) per
frame, so no across-frame certificate exists -- the deployment re-solves the
tensor exactly every frame, which is the fail-closed behaviour.

These tests validate (a) the certificate logic and (b) -- critically -- that
the analytic ``L_g`` is a *sound* (never under-estimating) Lipschitz constant
of the actual geometry -> g_dd map for position-only displacements.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from uav_isac.coordination.power_staleness import (
    dd_gate_active_set_certificate,
    dd_gate_position_lipschitz_constant,
)
from uav_isac.physical.geometry import compute_all_bistatic_params
from uav_isac.physical.otfs import compute_dd_effectiveness

M, DELTA_F, T_SYM, N, FC = 64, 1.5625e4, 6.4e-5, 16, 2.8e10
G_MIN = 0.5
C = 3.0e8
V_MAX = 25.0
TGT_V_MAX = 5.0
R = 2.5  # one frame step (v_max * dt)


def _g_dd_from_geometry(uav_pos, uav_vel, tgt_pos, tgt_vel):
    """Per-edge g_dd tensor (K,K,Q); i==j entries are -1 (invalid)."""
    K = uav_pos.shape[0]
    Q = tgt_pos.shape[0]
    roles = np.zeros(K, dtype=np.int64)  # ignored: role_agnostic=True
    tau, nu, _ = compute_all_bistatic_params(
        uav_pos, uav_vel, tgt_pos, tgt_vel, roles, FC, role_agnostic=True)
    g = np.full((K, K, Q), -1.0, dtype=np.float64)
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            for q in range(Q):
                g[i, j, q] = compute_dd_effectiveness(
                    tau[i, j, q], nu[i, j, q], DELTA_F, T_SYM, M, N, G_MIN)
    return g


def _random_geometry(K, Q, rng, scale=800.0, min_dist=40.0):
    """Random geometry with every UAV-target distance >= min_dist."""
    uav_pos = rng.uniform(0, scale, size=(K, 3))
    uav_pos[:, 2] = 20.0
    tgt_pos = rng.uniform(0, scale, size=(Q, 3))
    tgt_pos[:, 2] = 0.0
    # keep targets away from UAVs (avoid tiny R_min blowing up the bound)
    for _ in range(20):
        d = np.linalg.norm(uav_pos[:, None, :2] - tgt_pos[None, :, :2],
                           axis=-1)
        if d.min() >= min_dist:
            break
        close = np.argwhere(d < min_dist)
        for i, q in close[:3]:
            tgt_pos[q, :2] = uav_pos[i, :2] + rng.uniform(
                min_dist, min_dist + 40.0, size=2)
    uav_vel = rng.uniform(-V_MAX, V_MAX, size=(K, 3))
    uav_vel[:, 2] = 0.0
    tgt_vel = rng.uniform(-TGT_V_MAX, TGT_V_MAX, size=(Q, 3))
    tgt_vel[:, 2] = 0.0
    return uav_pos, uav_vel, tgt_pos, tgt_vel


def _r_min(uav_pos, tgt_pos):
    d = np.linalg.norm(uav_pos[:, None, :2] - tgt_pos[None, :, :2], axis=-1)
    return float(d.min())


def _Lg(r_min):
    return dd_gate_position_lipschitz_constant(
        M, DELTA_F, T_SYM, N, FC, C, V_MAX, TGT_V_MAX, r_min)


# ---------------------------------------------------------------------------
# 1. Certificate logic
# ---------------------------------------------------------------------------

def test_certificate_masks_and_r_max():
    rng = np.random.default_rng(0)
    g_dd = rng.uniform(0.0, 1.0, size=(4, 4, 3))
    L_g = _Lg(100.0)
    certified, uncertain, r_max = dd_gate_active_set_certificate(
        g_dd, G_MIN, R, L_g)
    assert certified.shape == g_dd.shape
    assert np.all(certified != uncertain)
    margin = np.abs(g_dd - G_MIN)
    assert np.isclose(r_max, margin.min() / L_g)
    assert np.all(certified == (margin > L_g * R))
    # certification is monotone in the trust-region radius: smaller r certifies
    # a superset of edges
    certified_big, _, _ = dd_gate_active_set_certificate(g_dd, G_MIN, R, L_g)
    certified_small, _, _ = dd_gate_active_set_certificate(
        g_dd, G_MIN, 1e-3, L_g)
    assert np.all(certified_big <= certified_small)
    assert np.any(certified_small)


def test_certificate_requires_positive_lipschitz():
    g_dd = np.full((2, 2, 2), 0.7)
    with pytest.raises(ValueError):
        dd_gate_active_set_certificate(g_dd, G_MIN, R, 0.0)
    with pytest.raises(ValueError):
        dd_gate_active_set_certificate(np.ones((2, 2)), G_MIN, R, 1.0)


# ---------------------------------------------------------------------------
# 2. Soundness of the analytic Lipschitz constant (the crux of C4)
# ---------------------------------------------------------------------------

def test_analytic_lipschitz_constant_is_sound_position_only():
    """For VELOCITY-HELD position displacements the analytic L_g must bound
    |Delta g_dd|/r.  If it does not, the certificate could falsely certify an
    edge whose gate actually flips inside the trust region."""
    rng = np.random.default_rng(1)
    measured_ratios = []
    min_r_min = np.inf
    for _ in range(50):
        K, Q = 4, 3
        uav_pos, uav_vel, tgt_pos, tgt_vel = _random_geometry(K, Q, rng)
        min_r_min = min(min_r_min, _r_min(uav_pos, tgt_pos))
        g_old = _g_dd_from_geometry(uav_pos, uav_vel, tgt_pos, tgt_vel)
        for _ in range(6):
            delta = rng.uniform(-R, R, size=(K, 3))
            delta[:, 2] = 0.0
            norm = np.linalg.norm(delta, axis=1)
            scale = np.minimum(1.0, R / np.maximum(norm, 1e-12))
            delta = delta * scale[:, None]
            new_pos = uav_pos + delta
            new_vel = uav_vel.copy()  # velocity HELD (within-frame)
            g_new = _g_dd_from_geometry(new_pos, new_vel, tgt_pos, tgt_vel)
            valid = (g_old >= 0.0) & (g_new >= 0.0)
            diff = np.abs(g_new[valid] - g_old[valid])
            step_r = float(np.linalg.norm(delta).max())
            if step_r > 1e-9:
                measured_ratios.append(float(diff.max()) / step_r)
    measured_L = float(np.max(measured_ratios))
    # conservative global constant: use the smallest R_min seen (largest L_g)
    L_g = _Lg(min_r_min)
    assert L_g >= measured_L, (
        f"analytic L_g = {L_g:.4f} < measured max |d g_dd|/r = {measured_L:.4f}: "
        f"the certificate would be unsound")


def test_certified_edges_never_flip_within_trust_region():
    """The theorem's key property: certified edges keep their gate state for
    VELOCITY-HELD displacements of size <= r (r <= r_max certifies every edge,
    so no edge may flip)."""
    rng = np.random.default_rng(2)
    for _ in range(40):
        K, Q = 4, 3
        uav_pos, uav_vel, tgt_pos, tgt_vel = _random_geometry(K, Q, rng)
        L_g = _Lg(_r_min(uav_pos, tgt_pos))
        g_old = _g_dd_from_geometry(uav_pos, uav_vel, tgt_pos, tgt_vel)
        valid = g_old >= 0.0
        margin = np.abs(g_old - G_MIN)
        r_max = float(np.min(margin[valid])) / L_g
        if r_max <= 1e-2:
            continue  # geometry sits on the gate; nothing meaningful to certify
        r = float(r_max) * 0.5
        certified, _, _ = dd_gate_active_set_certificate(g_old, G_MIN, r, L_g)
        assert np.all(certified[valid]), "r <= r_max must certify every edge"
        for _ in range(5):
            delta = rng.uniform(-r, r, size=(K, 3))
            delta[:, 2] = 0.0
            norm = np.linalg.norm(delta, axis=1)
            scale = np.minimum(1.0, r / np.maximum(norm, 1e-12))
            delta = delta * scale[:, None]
            g_new = _g_dd_from_geometry(
                uav_pos + delta, uav_vel.copy(), tgt_pos, tgt_vel)
            old_state = g_old[valid] >= G_MIN
            new_state = g_new[valid] >= G_MIN
            assert np.array_equal(old_state, new_state), (
                "a certified edge flipped its DD gate inside the trust region")


def test_across_frame_velocity_jump_not_certified():
    """Across frames (velocity = step/dt) the gate CAN flip even for certified
    edges -> the position-only certificate must NOT be used there; the
    deployment re-solves exactly every frame (fail-closed)."""
    rng = np.random.default_rng(5)
    flips_seen = 0
    for _ in range(30):
        K, Q = 4, 3
        uav_pos, uav_vel, tgt_pos, tgt_vel = _random_geometry(K, Q, rng)
        L_g = _Lg(_r_min(uav_pos, tgt_pos))
        g_old = _g_dd_from_geometry(uav_pos, uav_vel, tgt_pos, tgt_vel)
        valid = g_old >= 0.0
        margin = np.abs(g_old - G_MIN)
        r_max = float(np.min(margin[valid])) / L_g
        if r_max <= 1e-2:
            continue
        r = float(r_max) * 0.5
        certified, _, _ = dd_gate_active_set_certificate(g_old, G_MIN, r, L_g)
        # step with the per-frame velocity model: v' = delta/dt (old v can be
        # anything up to v_max, i.e. up to ~R/dt)
        delta = rng.uniform(-r, r, size=(K, 3))
        delta[:, 2] = 0.0
        new_pos = uav_pos + delta
        new_vel = delta / 0.1
        g_new = _g_dd_from_geometry(new_pos, new_vel, tgt_pos, tgt_vel)
        flipped = (g_old[valid] >= G_MIN) != (g_new[valid] >= G_MIN)
        flips_seen += int(np.any(flipped & certified[valid]))
    assert flips_seen > 0, (
        "expected at least one across-frame gate flip to demonstrate "
        "fail-closed is necessary")


def test_fail_closed_when_certificate_violated():
    """When r > r_max the certificate fails closed (uncertain edges exist)."""
    rng = np.random.default_rng(3)
    g_dd = rng.uniform(0.2, 0.9, size=(4, 4, 3))
    L_g = _Lg(100.0)
    _, _, r_max = dd_gate_active_set_certificate(g_dd, G_MIN, 1e-3, L_g)
    certified, uncertain, _ = dd_gate_active_set_certificate(
        g_dd, G_MIN, float(r_max) * 2.0 + 1e-6, L_g)
    assert not np.all(certified)
    assert np.any(uncertain)
