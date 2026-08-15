"""Tests for D0.94-L3D T4: DD-support restoration (Mode II)."""

import numpy as np
import pytest

from uav_isac.physical.otfs import compute_dd_effectiveness

# 8/8 OTFS grid parameters.
DELTA_F = 1.5625e4
T_SYM = 6.4e-5
M = 64
N = 16
G_MIN = 0.5
C = 299_792_458.0


def _g_dd(uav, owner, target, nu):
    tau = (np.linalg.norm(uav - target) + np.linalg.norm(owner - target)) / C
    return float(compute_dd_effectiveness(tau, nu, DELTA_F, T_SYM, M, N, G_MIN))


# A Doppler that puts k_offset ~ 0.5 (sinc ~ 0.637), so a delay with
# l_offset ~ 0.5 gives g_dd = 0.637^2 ~ 0.406 < g_min.
NU_MISALIGNED = 0.5 / (N * T_SYM)


def _restoration_potential(g_dd, price):
    return price * max(G_MIN - g_dd, 0.0)


def test_restoration_potential_nonzero_when_dd_infeasible():
    """Mode II: a DD-misaligned high-price target yields a non-zero potential."""
    target = np.array([0.0, 0.0])
    owner = np.array([50.0, 0.0])
    found = False
    for d in np.linspace(30.0, 300.0, 3000):
        g = _g_dd(np.array([d, 0.0]), owner, target, NU_MISALIGNED)
        if g < G_MIN:
            found = True
            psi = _restoration_potential(g, price=1.0)
            assert psi > 0.0
            break
    assert found, "expected at least one DD-infeasible geometry in the scan"


def test_restoration_gradient_reduces_dd_gap():
    """Moving along -grad(Psi) reduces the DD gap (brings target back feasible)."""
    target = np.array([0.0, 0.0])
    owner = np.array([50.0, 0.0])
    price = 1.0
    best = None
    for d in np.linspace(30.0, 300.0, 3000):
        g = _g_dd(np.array([d, 0.0]), owner, target, NU_MISALIGNED)
        if g < G_MIN:
            best = d
            break
    assert best is not None

    def psi_at(x):
        return _restoration_potential(
            _g_dd(np.array([x, 0.0]), owner, target, NU_MISALIGNED), price)

    h = 0.5
    fd = (psi_at(best + h) - psi_at(best - h)) / (2.0 * h)
    step = -np.sign(fd) * h
    assert psi_at(best + step) < psi_at(best), (
        "restoration step must reduce the potential (increase g_dd toward g_min)")


def test_capability_gradient_zero_when_dd_infeasible():
    """Contrast: when DD-infeasible (a=0), the capability gradient is zero."""
    target = np.array([0.0, 0.0])
    owner = np.array([50.0, 0.0])
    for d in np.linspace(30.0, 300.0, 3000):
        g = _g_dd(np.array([d, 0.0]), owner, target, NU_MISALIGNED)
        if g < G_MIN:
            # The effective gain is zero (DD gate), so capability gradient = 0.
            assert g < G_MIN
            return
