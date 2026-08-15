"""Two-sided piecewise-linear (PWL) bounds of the detection function.

P_D(D) = Phi(sqrt(D) - c), c = Q^{-1}(P_FA), is concave for D >= c^2/3 (the
high-D working region).  For a concave function:

- the *chord* between two breakpoints lies BELOW the function (a lower bound);
- the *tangent* at any breakpoint lies ABOVE the function (an upper bound).

Both bounds are concave piecewise-linear, so the capability gauge that uses
them becomes a **linear program** (concave PWL = pointwise min of affine
minorants).  The chord lower bound gives the conservative gauge
``gamma*(chord) >= gamma*``; the tangent upper bound gives the optimistic
``gamma*(tangent) <= gamma*``, hence ``gamma_optimistic <= gamma* <=
gamma_conservative``.

Breakpoint placement uses the classical chord-interpolation error bound: with
``M_m = max_{D in I_m} |P_D''(D)|``,

    0 <= P_D(D) - chord(D) <= (M_m/8) (d_{m+1}-d_m)^2,

so a per-segment error ``<= eps`` needs ``Delta d_m <= sqrt(8 eps / M_m)``.
This makes breakpoints dense in high-curvature regions and sparse elsewhere,
rather than a fixed 8/16-segment sweep.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc

from uav_isac.utils.math_utils import Q_inverse


def p_d(d: np.ndarray, p_fa: float) -> np.ndarray:
    """P_D(D) = Phi(sqrt(D) - c), c = Q^{-1}(P_FA)."""
    d = np.asarray(d, dtype=np.float64)
    c = Q_inverse(np.asarray(float(p_fa)))
    return 0.5 * erfc((c - np.sqrt(np.maximum(d, 0.0))) / np.sqrt(2.0))


def p_d_first_derivative(d: np.ndarray, p_fa: float) -> np.ndarray:
    """dP_D/dD (analytic)."""
    d = np.asarray(d, dtype=np.float64)
    c = Q_inverse(np.asarray(float(p_fa)))
    x = np.sqrt(np.maximum(d, 1e-12))
    phi = np.exp(-((x - c) ** 2) / 2.0) / np.sqrt(2.0 * np.pi)
    return phi / (2.0 * x)


def p_d_second_derivative(d: np.ndarray, p_fa: float) -> np.ndarray:
    """d^2 P_D / dD^2 (analytic): -phi(x-c)[(x-c)+1/x]/(4x^2), x=sqrt(D)."""
    d = np.asarray(d, dtype=np.float64)
    c = Q_inverse(np.asarray(float(p_fa)))
    x = np.sqrt(np.maximum(d, 1e-12))
    phi = np.exp(-((x - c) ** 2) / 2.0) / np.sqrt(2.0 * np.pi)
    return -phi * ((x - c) + 1.0 / x) / (4.0 * x * x)


def curvature_breakpoints(
    p_fa: float,
    d_min: float,
    d_max: float,
    epsilon: float,
    max_segments: int = 400,
) -> np.ndarray:
    """Adaptive breakpoints so each chord segment has error <= ~epsilon.

    Solve ``step = sqrt(8 eps / |P_D''(d + step)|)`` by fixed-point iteration:
    the curvature is evaluated at the segment's far end, which over-estimates
    ``M_m`` in the increasing-curvature region (smaller, safer steps) and is
    exact where curvature decreases.
    """
    pts = [float(d_min)]
    d = float(d_min)
    step = (float(d_max) - float(d_min)) / 64.0
    while d < float(d_max) and len(pts) < max_segments:
        step = min(step, float(d_max) - d)
        for _ in range(30):
            far = min(d + step, float(d_max))
            m = abs(float(p_d_second_derivative(np.asarray([far]), p_fa)[0]))
            new_step = np.sqrt(8.0 * float(epsilon) / max(m, 1e-14))
            new_step = min(new_step, float(d_max) - d)
            if abs(new_step - step) < 1e-10 or new_step <= 0.0:
                step = new_step
                break
            step = new_step
        if step <= 1e-12:
            break
        d += step
        pts.append(d)
    if pts[-1] < float(d_max):
        pts.append(float(d_max))
    return np.asarray(pts, dtype=np.float64)


def chord_lower_bound(p_fa: float, breakpoints: np.ndarray) -> np.ndarray:
    """Chord lower bound: (slopes, intercepts) with slope*D + intercept <= P_D."""
    bps = np.asarray(breakpoints, dtype=np.float64)
    y = p_d(bps, p_fa)
    slopes = np.diff(y) / np.diff(bps)
    intercepts = y[:-1] - slopes * bps[:-1]
    return slopes, intercepts


def tangent_upper_bound(p_fa: float, breakpoints: np.ndarray) -> np.ndarray:
    """Tangent upper bound: (slopes, intercepts) with slope*D + intercept >= P_D."""
    bps = np.asarray(breakpoints, dtype=np.float64)
    y = p_d(bps, p_fa)
    slopes = p_d_first_derivative(bps, p_fa)
    intercepts = y - slopes * bps
    return slopes, intercepts


def evaluate_pwl(d: np.ndarray, slopes: np.ndarray, intercepts: np.ndarray) -> np.ndarray:
    """Pointwise min of the affine pieces (the concave PWL function)."""
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    vals = slopes[:, None] * d[None, :] + intercepts[:, None]
    return np.min(vals, axis=0)
