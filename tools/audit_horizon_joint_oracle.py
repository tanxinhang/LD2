#!/usr/bin/env python
"""D1.0-A Horizon Joint Oracle: algorithmic reachability diagnostic (advice 009).

The D0.95 analytical stack is a *first-order reactive* controller: L3 moves each
UAV by one normalized gradient step per frame, L2 is a per-watt P0 ranking, and
L1 is an exact max-min LP.  The same-geometry waterfall shows

    deployed 0.31 -> power_only 0.675 -> single 0.700 -> duplex 0.706 -> relaxed 0.868,

but none of these rungs lets the UAVs *move*.  This tool adds the missing rung:

    horizon_joint = joint structure(L2) x power(L1) x geometry(L3) planning
                    over H frames, from the same per-seed starting geometry.

It is deliberately an ORACLE: it may use global information (true target
positions, full per-watt tensor), it re-optimises structure and power exactly
at every candidate geometry, and its geometry step is a coordinate trust-region
search (multi-candidate SCP-lite with exact physics re-evaluation) instead of a
single normalized gradient.  This answers the D1.0-A gate question: *can a
strong horizon joint planner convert the relaxed headroom into executed
geometry, i.e. reach worst ~0.72/0.75 and P_QoS ~0.80 on the 20 existing
8/8 seeds?*  If the oracle cannot, the gap is physical/feasible-set; if it can,
the gap is the algorithm (finite-horizon first-order controller vs multi-step
joint optimizer).

Method per seed (start geometry = final resolved frame by default):
  for step in 1..H:
    repeat R rounds:
      L2: price-driven structure repair  (priced_structure_repair)
      L1: exact fixed-owner max-min LP   (solve_fixed_structure_maxmin_power_lp)
      L3: coordinate trust-region search (per UAV, 21 candidate moves, exact
          Friis rescale + exact LP per candidate, accept best improving move)
    advance; rescale the full (K,K,Q) per-watt tensor to the moved geometry.

Objective: max-min worst deflection (D1.0-A) or the advice-009 windowed tail
deficit sum_q [D_tar - D_q]_+^2 with D_tar = D(P_D=0.75).  The DD gate g_dd and
reporting reliability chi_rep are frozen at trace values (first-order Friis
approximation used by the existing L3 audits; static targets only).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from uav_isac.coordination.capability import (  # noqa: E402
    capability_geometry_gradient,
    capability_gauge_pwl_lp_full,
)
from uav_isac.coordination.maxmin_power import (  # noqa: E402
    fixed_owner_gain_matrix,
    relaxed_same_geometry_target_ceiling,
    solve_fixed_structure_maxmin_power_lp,
)
from uav_isac.coordination.priced_structure import (  # noqa: E402
    priced_structure_repair,
)
from uav_isac.coordination.pwl_pd import (  # noqa: E402
    chord_lower_bound,
    curvature_breakpoints,
)
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)
from uav_isac.physical.detection import (  # noqa: E402
    compute_detection_probabilities,
    minimum_deflection_for_detection_probability,
)

DEFAULT_TRACE = (
    ROOT
    / "results"
    / "architecture_v2_scale_k8q8_teacher_trace_d079_test20"
    / "teacher_trace.npz"
)
DEFAULT_CONFIG = ROOT / "config" / (
    "exp_800_k8q8_architecture_v2_maxmin_local_fusion_fullgraph_hold5_scale.yaml"
)


# --------------------------------------------------------------------------
# geometry / physics helpers (mirror the existing L3 audit conventions)
# --------------------------------------------------------------------------

def _full_coeff(data: dict, row: int, cfg) -> np.ndarray:
    """Per-watt deflection tensor (K,K,Q) reconstructed from trace observables."""
    return per_watt_deflection_tensor_from_observables(
        np.asarray(data["privileged_alpha"][row], dtype=np.float64),
        np.asarray(data["privileged_g_dd"][row], dtype=np.float64),
        np.asarray(data["privileged_chi_rep"][row], dtype=np.float64),
        T_sym=float(cfg.otfs.T_sym),
        M=int(cfg.otfs.M),
        N=int(cfg.otfs.N),
        kT=float(cfg.channel.kT),
        bandwidth_hz=float(cfg.otfs.B),
        noise_figure_db=float(cfg.channel.NF),
        g_tx_dbi=float(cfg.otfs.g_tx_dBi),
        g_rx_dbi=float(cfg.otfs.g_rx_dBi),
        n_cpi=int(cfg.otfs.n_cpi),
        g_min=float(cfg.detection.g_min),
        use_swerling=bool(cfg.channel.use_swerling),
    )


def _budget(data: dict, row: int) -> np.ndarray:
    """Per-UAV sensing budget b_i = 1 - P_comm,i recorded by the trace."""
    rate = np.asarray(data["outgoing_rate"][row], dtype=np.int64)
    mask = np.asarray(data["outgoing_token_mask"][row], dtype=bool)
    active = (rate > 0) & np.any(mask, axis=1)
    frac = np.clip(
        np.asarray(data["comm_fraction"][row], dtype=np.float64), 0.0, 1.0)
    return 1.0 - np.where(active, frac, 0.0)


def _uav(data: dict, row: int) -> np.ndarray:
    return np.asarray(data["uav_positions"][row], dtype=np.float64)[:, :2].copy()


def _tgt(data: dict, row: int) -> np.ndarray:
    return np.asarray(data["target_states"][row], dtype=np.float64)[:, :2].copy()


def friis_rescale_tensor(
    coeff: np.ndarray, uav: np.ndarray, tgt: np.ndarray, new_uav: np.ndarray,
) -> np.ndarray:
    """Rescale the full (K,K,Q) per-watt tensor under 1/(R_tx^2 R_rx^2)."""
    r = np.linalg.norm(uav[:, None, :] - tgt[None, :, :], axis=2)      # (K,Q)
    rn = np.linalg.norm(new_uav[:, None, :] - tgt[None, :, :], axis=2)
    c = coeff * (r[:, None, :] ** 2) * (r[None, :, :] ** 2)
    return c / (rn[:, None, :] ** 2 * rn[None, :, :] ** 2)


def rescale_fixed_gain(
    gain: np.ndarray, owner: np.ndarray,
    uav: np.ndarray, tgt: np.ndarray, new_uav: np.ndarray,
) -> np.ndarray:
    """Rescale the fixed-owner gain (K,Q) after a geometry move (structure fixed).

    a_iq = C_iq / (R_iq^2 * R_owner(q),q^2), so moving UAV k changes only row k
    (its own Tx gains) and the rows of targets whose owner is k (Rx effect).
    """
    rtx = np.linalg.norm(uav[:, None, :] - tgt[None, :, :], axis=2)      # (K,Q)
    rrx = np.linalg.norm(uav[owner] - tgt, axis=1)                        # (Q,)
    rtx_n = np.linalg.norm(new_uav[:, None, :] - tgt[None, :, :], axis=2)
    rrx_n = np.linalg.norm(new_uav[owner] - tgt, axis=1)
    c = gain * (rtx ** 2) * (rrx[None, :] ** 2)
    return c / (rtx_n ** 2 * rrx_n[None, :] ** 2)


def maxmin_geometry_gradient(
    k: int, owner: np.ndarray, uav: np.ndarray, tgt: np.ndarray,
    gain: np.ndarray, power: np.ndarray, prices: np.ndarray,
) -> np.ndarray:
    """Ascent direction of the max-min worst deflection w.r.t. UAV k position.

    d t*/d x_k = sum_q lambda*_q [ p*_kq d a_kq/d x_k
                 + 1[k=owner_q] sum_i p*_iq d a_iq/d x_k ],
    d a_iq/d x_k = -2 a_iq (x_k - x_q) / R_kq^2   (points toward the target).
    """
    K, Q = gain.shape
    g = np.zeros(2, dtype=np.float64)
    for q in range(Q):
        w = prices[q]
        if abs(w) < 1e-15:
            continue
        r = max(float(np.linalg.norm(uav[k] - tgt[q])), 1e-9)
        dirq = -2.0 * (uav[k] - tgt[q]) / (r * r)
        tx_term = power[k, q] * gain[k, q]
        rx_term = 0.0
        if owner[q] == k:
            rx_term = float(np.sum(power[:, q] * gain[:, q]))
        g += w * (tx_term + rx_term) * dirq
    return g


def deficit_geometry_gradient(
    k: int, owner: np.ndarray, uav: np.ndarray, tgt: np.ndarray,
    gain: np.ndarray, budget: np.ndarray, d_min: float,
) -> np.ndarray:
    """Ascent direction of the squared max-min ceiling deficit (D0.94 Phase 1).

    Used when the fixed-owner ceiling sum_i a_iq b_i is below the worst floor
    for some target: the max-min LP is then infeasible-relative-to-floor and
    its dual is degenerate, so the deficit gradient is the correct driver.
    """
    K, Q = gain.shape
    ceiling = np.sum(gain * budget[:, None], axis=0)
    deficit = np.maximum(0.0, d_min - ceiling)
    g = np.zeros(2, dtype=np.float64)
    for q in range(Q):
        w = deficit[q] * budget[k]
        if abs(w) < 1e-15:
            continue
        a = gain[k, q]
        r = max(float(np.linalg.norm(uav[k] - tgt[q])), 1e-9)
        g += w * (2.0 * a) * (uav[k] - tgt[q]) / (r * r)
    for q in range(Q):
        if owner[q] != k:
            continue
        r = max(float(np.linalg.norm(uav[k] - tgt[q])), 1e-9)
        for i in range(K):
            w = deficit[q] * budget[i]
            if abs(w) < 1e-15:
                continue
            a = gain[i, q]
            g += w * (2.0 * a) * (uav[k] - tgt[q]) / (r * r)
    return g


def candidate_moves(
    grad_k: np.ndarray, step_m: float,
) -> list[np.ndarray]:
    """Trust-region candidate set: stay + 8 compass x {1, 1/2} + gradient x {1, 1/2}."""
    cands: list[np.ndarray] = [np.zeros(2, dtype=np.float64)]
    for i in range(8):
        angle = i * np.pi / 4.0
        d = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        cands.append(step_m * d)
        cands.append(0.5 * step_m * d)
    n = float(np.linalg.norm(grad_k))
    if n > 1e-12:
        g = grad_k / n
        cands.append(step_m * g)
        cands.append(0.5 * step_m * g)
        cands.append(-step_m * g)
        cands.append(-0.5 * step_m * g)
    return cands


def greedy_target_assignment(
    gain: np.ndarray, budget: np.ndarray, uav: np.ndarray, tgt: np.ndarray,
    prices: np.ndarray | None, d_min: float,
) -> np.ndarray:
    """D1.0-C: target responsibility auction (advice 009 §5).

    Each UAV is assigned the target where its *predicted marginal geometry
    value* is highest:

        G_kq = urgency_q * b_k * a_kq / R_kq^2,

    proportional to the Friis sensitivity |d a_kq / d x_k|.  ``urgency`` is the
    max-min dual price (bottleneck), the gauge price, or the worst-floor deficit
    when no price is available.  A greedy auction assigns each UAV at most one
    target (highest-G pairs first); targets may receive many UAVs (herding onto
    the bottleneck is exactly what max-min wants).  Returns ``assign`` (K,)
    with the assigned target index per UAV (-1 when none).
    """
    K, Q = gain.shape
    if prices is None:
        ceiling = np.sum(gain * budget[:, None], axis=0)
        urgency = np.maximum(0.0, d_min - ceiling) + 1e-9
    else:
        urgency = np.maximum(np.asarray(prices, dtype=np.float64), 0.0)
    if float(np.max(urgency)) <= 0.0:
        urgency = np.full(Q, 1.0 / Q, dtype=np.float64)
    r = np.linalg.norm(uav[:, None, :] - tgt[None, :, :], axis=2)  # (K,Q)
    r = np.maximum(r, 1e-9)
    G = urgency[None, :] * budget[:, None] * gain / (r * r)
    assign = np.full(K, -1, dtype=np.int64)
    order = np.argsort(-G.reshape(-1), kind="stable")
    for flat in order:
        k, q = divmod(int(flat), Q)
        if assign[k] == -1 and G[k, q] > 0.0:
            assign[k] = q
    return assign


# --------------------------------------------------------------------------
# per-seed horizon planner
# --------------------------------------------------------------------------

def _score_deflection(deflection: np.ndarray, objective: str, d_tar: float) -> float:
    if objective == "tail":
        return -float(np.sum(np.maximum(d_tar - deflection, 0.0) ** 2))
    return float(np.min(deflection))  # max-min


# Gauge inner solver (three floors: worst 0.60 / weak3 0.70 / steady 0.80).
XI = (0.60, 0.70, 0.80, 3)
# D1.1-A robust margin floors (advice 010): controller targets slightly above
# the evaluation standard so the strict `>=` QoS check passes robustly.
XI_LEX = (0.61, 0.71, 0.81, 3)
# Hybrid score: maximize worst deflection, penalising gamma* > 1 (floors unmet)
# hard enough that feasibility dominates, then the worst is pushed beyond the
# floors.  Scale: gamma is O(0.5-2), min(D) is O(8-16).
FLOOR_PENALTY = 100.0


def _gauge_solve(
    gain: np.ndarray, budget: np.ndarray, p_fa: float, d_min: float,
    xi: tuple[float, float, float, int] = XI,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Capability gauge with PWL certificates; None when a target is unreachable."""
    ceiling = np.sum(gain * budget[:, None], axis=0)
    if np.any(ceiling < d_min - 1e-9):
        return None
    d_max = float(np.max(ceiling)) + 1.0
    bps = curvature_breakpoints(p_fa, d_min, d_max, epsilon=1e-3)
    cs, ci = chord_lower_bound(p_fa, bps)
    return capability_gauge_pwl_lp_full(gain, budget, p_fa, xi, cs, ci, d_min)


def _qos_constrained_maxmin(
    gain: np.ndarray, budget: np.ndarray, p_fa: float,
    xi: tuple[float, float, float, int],
    slopes: np.ndarray, intercepts: np.ndarray, d_min: float,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Stage B of lexicographic (advice 010): max-min worst subject to hard floors.

        max_{p, D, y, tau, z, t}  t
        s.t.  D_q = sum_i a_iq p_iq
              D_q >= t                       (max-min)
              D_q >= d_min                   (worst floor)
              y_q <= slope_m D_q + intercept_m   (PWL chord lower bound, P_D >= y)
              sum_q y_q >= Q * steady_floor      (steady floor)
              z_q >= tau - y_q, z_q >= 0         (bottom-k linearization)
              k*tau - sum_q z_q >= k * weak3_floor
              sum_q p_iq <= b_i, p >= 0

    Returns (t*, p*, D*), or None when infeasible.  The gauge's own solution is
    a feasible point of this LP, so the lexicographic optimum is no worse than
    the gauge on BOTH the QoS floors and the max-min worst.
    """
    from scipy.optimize import linprog

    rho_min, rho_tail, rho_avg, k = xi
    K, Q = gain.shape
    n_p = K * Q
    o_D = n_p
    o_y = o_D + Q
    o_tau = o_y + Q
    o_z = o_tau + 1
    o_t = o_z + Q
    n_vars = o_t + 1
    c = np.zeros(n_vars)
    c[o_t] = -1.0  # maximize t

    eq_rows = []
    eq_ub = []
    for q in range(Q):
        row = np.zeros(n_vars)
        row[o_D + q] = 1.0
        for i in range(K):
            row[i * Q + q] = -gain[i, q]
        eq_rows.append(row)
        eq_ub.append(0.0)

    rows = []
    upper = []
    # max-min: D_q >= t  ->  -D_q + t <= 0
    for q in range(Q):
        row = np.zeros(n_vars)
        row[o_D + q] = -1.0
        row[o_t] = 1.0
        rows.append(row)
        upper.append(0.0)
    # worst floor: D_q >= d_min
    for q in range(Q):
        row = np.zeros(n_vars)
        row[o_D + q] = -1.0
        rows.append(row)
        upper.append(-d_min)
    # PWL chord: y_q <= slope_m D_q + intercept_m
    M = slopes.size
    for m in range(M):
        for q in range(Q):
            row = np.zeros(n_vars)
            row[o_D + q] = -slopes[m]
            row[o_y + q] = 1.0
            rows.append(row)
            upper.append(float(intercepts[m]))
    # steady: sum_q y_q >= Q * rho_avg
    row = np.zeros(n_vars)
    row[o_y:o_y + Q] = -1.0
    rows.append(row)
    upper.append(-Q * rho_avg)
    # bottom-k: z_q - tau + y_q >= 0  ->  -z_q + tau - y_q <= 0
    for q in range(Q):
        row = np.zeros(n_vars)
        row[o_z + q] = -1.0
        row[o_tau] = 1.0
        row[o_y + q] = -1.0
        rows.append(row)
        upper.append(0.0)
    # bottom-k: -k*tau + sum_q z_q <= -k*rho_tail
    row = np.zeros(n_vars)
    row[o_tau] = -k
    row[o_z:o_z + Q] = 1.0
    rows.append(row)
    upper.append(-k * rho_tail)
    # budget: sum_q p_iq <= b_i
    for i in range(K):
        row = np.zeros(n_vars)
        row[i * Q:(i + 1) * Q] = 1.0
        rows.append(row)
        upper.append(float(budget[i]))

    res = linprog(
        c,
        A_ub=np.stack(rows),
        b_ub=np.asarray(upper, dtype=np.float64),
        A_eq=np.stack(eq_rows),
        b_eq=np.asarray(eq_ub, dtype=np.float64),
        bounds=[(0.0, None)] * n_vars,
        method="highs",
    )
    if not res.success or res.x is None:
        return None
    p = res.x[:n_p].reshape(K, Q)
    d = res.x[o_D:o_D + Q]
    return float(res.x[o_t]), p, d


def _pwl_chord(gain, budget, p_fa, d_min):
    ceiling = np.sum(gain * budget[:, None], axis=0)
    d_max = float(np.max(ceiling)) + 1.0
    bps = curvature_breakpoints(p_fa, d_min, d_max, epsilon=1e-3)
    return chord_lower_bound(p_fa, bps)


def _gauge_score(
    out: tuple[float, np.ndarray, np.ndarray] | None,
    gain: np.ndarray, budget: np.ndarray, p_fa: float, d_min: float,
) -> float:
    if out is not None:
        return -float(out[0])  # lower gamma is better
    ceiling = np.sum(gain * budget[:, None], axis=0)
    return -1e6 - float(np.max(np.maximum(d_min - ceiling, 0.0)))


# --------------------------------------------------------------------------
# advice 011 / T2 Central Oracle: low-exposure + comm coupling + safety
# --------------------------------------------------------------------------

def _min_comm_power_budget(
    uav: np.ndarray, cfg, packet_bits: float = 600.0,
) -> np.ndarray:
    """b_i = 1 - P_comm^min(X) (advice 011 §2): analytic minimum broadcast power.

    Mirrors env_core._compute_analytical_min_comm_power: for sender i, the
    power meeting SNR threshold AND Shannon serialization + processing within
    the U2U deadline for EVERY receiver.  Harder U2U geometry -> higher
    P_comm^min -> smaller sensing budget b_i (true comm-sensing coupling).
    """
    K = int(uav.shape[0])
    fc = float(cfg.otfs.fc)
    lam = 299_792_458.0 / max(fc, 1.0)
    bw = float(getattr(cfg.marl, 'comm_bandwidth_hz', 1e5))
    deadline = float(getattr(cfg.marl, 'comm_deadline_s', 0.005))
    proc = float(getattr(cfg.marl, 'comm_processing_delay_s', 2e-4))
    snr_thr_db = float(getattr(cfg.marl, 'comm_snr_threshold_db', 0.0))
    antenna_dbi = float(getattr(cfg.marl, 'comm_antenna_gain_dbi', 0.0))
    antenna_lin = 10.0 ** (2.0 * antenna_dbi / 10.0)
    kT = float(cfg.channel.kT)
    nf_lin = 10.0 ** (float(cfg.channel.NF) / 10.0)
    n_active = max(1, K)
    b_eff = bw / n_active
    n0_b = kT * b_eff * nf_lin
    t_win = max(deadline - proc, 1e-12)
    gamma_th = 10.0 ** (snr_thr_db / 10.0)
    p = np.zeros(K, dtype=np.float64)
    r_req = float(packet_bits) / t_win
    # Cap the required SNR against float overflow for very large packets.
    gamma_rate = min(float(2.0 ** (r_req / b_eff) - 1.0), 1e6)
    gamma_req = max(gamma_th, gamma_rate)
    for i in range(K):
        for j in range(K):
            if j == i:
                continue
            d = max(float(np.linalg.norm(uav[i] - uav[j])), 1.0)
            g = antenna_lin * (lam / (4.0 * np.pi * d)) ** 2
            p[i] = max(p[i], gamma_req * n0_b / max(g, 1e-30))
    return np.clip(1.0 - p, 0.0, 1.0)


def _exposure_coefficients(
    uav: np.ndarray, tgt: np.ndarray, cfg,
) -> np.ndarray:
    """c[i,q] = G_tx_lin * (lambda/(4 pi d_3d))^2: per-watt leakage at target q.

    Simplest observer model (advice 011 §4): each target is its own potential
    observer; c is the free-space path gain from UAV i to target q (with the
    UAV height in the 3D distance).  Exposure E_q = sum_i c[i,q] p_iq.
    """
    fc = float(cfg.otfs.fc)
    lam = 299_792_458.0 / max(fc, 1.0)
    g_tx_lin = 10.0 ** (float(cfg.otfs.g_tx_dBi) / 10.0)
    h = float(cfg.scenario.height)
    d2 = np.linalg.norm(uav[:, None, :] - tgt[None, :, :], axis=2) ** 2  # (K,Q) xy^2
    d3 = np.sqrt(d2 + h * h)
    return g_tx_lin * (lam / (4.0 * np.pi * np.maximum(d3, 1e-6))) ** 2


# advice 012: opponent capability tiers (theta_w) for the counter-detection model.
# G_w: RX array gain (linear); B_w: listen bandwidth (Hz); NF_w: noise figure
# (dB); T_int_w: integration time (s); K_w: waveform-prior / matched-filter gain.
#
# NOTE (audit 2024-12, tier re-parameterisation): in the intercept model the
# per-watt Deflection gain is proportional to 1/(kT * B_w * NF_w), so a NARROW
# listen bandwidth (matched filter on a known subcarrier) makes the opponent
# MORE sensitive, not less.  The tiers below are therefore ordered by physical
# sensitivity:
#   * weak    — wideband search over ~100 MHz (no waveform knowledge), short
#               dwell, high NF, no array: sees almost nothing at these ranges.
#   * medium  — knows the band (~10 MHz), moderate dwell/array, partial
#               waveform structure: selectively binding at close range.
#   * strong  — matched filter on the exact pilot/subcarrier (1 kHz), long
#               coherent integration, large array, full waveform prior:
#               binding everywhere at 1 W / 28 GHz.
OPPONENT_CAPABILITIES = {
    "weak": dict(G_w=1.0, B_w=1e8, NF_w=12.0, T_int_w=1e-5, K_w=1.0),
    "medium": dict(G_w=3.0, B_w=1e7, NF_w=8.0, T_int_w=1e-4, K_w=2.0),
    "strong": dict(G_w=100.0, B_w=1e3, NF_w=3.0, T_int_w=1e-1, K_w=10.0),
}


def _intercept_coefficients(
    uav: np.ndarray, tgt: np.ndarray, cfg, theta: dict,
) -> np.ndarray:
    """a^I[i,q]: per-watt counter-detection Deflection gain at observer q.

    Mirror of the legitimate sensing abstraction (advice 012 §2/§4): the
    observer's detection statistic is assumed Gaussian-shift, so its Deflection
    from UAV i's power p_iq at observer q is

        D_q^I = sum_i a^I[i,q] p_iq,
        a^I[i,q] = K_w G_tx G_w (lambda/(4 pi d_3d))^2 T_int,w / (kT B_w NF_w).

    K_w encodes how much the opponent knows of the waveform/frame structure
    (energy detection -> matched-filter).  d_3d includes the UAV height.
    """
    fc = float(cfg.otfs.fc)
    lam = 299_792_458.0 / max(fc, 1.0)
    g_tx_lin = 10.0 ** (float(cfg.otfs.g_tx_dBi) / 10.0)
    h = float(cfg.scenario.height)
    d2 = np.linalg.norm(uav[:, None, :] - tgt[None, :, :], axis=2) ** 2
    d3 = np.sqrt(d2 + h * h)
    path = (lam / (4.0 * np.pi * np.maximum(d3, 1e-6))) ** 2
    g_w = float(theta["G_w"])
    b_w = float(theta["B_w"])
    nf_lin = 10.0 ** (float(theta["NF_w"]) / 10.0)
    t_int = float(theta["T_int_w"])
    k_w = float(theta["K_w"])
    noise = max(float(cfg.channel.kT) * b_w * nf_lin, 1e-30)
    return k_w * g_tx_lin * g_w * path * t_int / noise


def _intercept_deflection_limit(p_fa_i: float, eps: float) -> float:
    """bar D^I = [Q^-1(P_FA^I) - Q^-1(eps)]^2 (advice 012 §3)."""
    from uav_isac.utils.math_utils import Q_inverse
    root = float(Q_inverse(np.asarray(float(p_fa_i)))) - float(
        Q_inverse(np.asarray(float(eps))))
    return root * root


def _exposure_maxmin_lp(
    gain: np.ndarray, budget: np.ndarray, exposure_c: np.ndarray,
    gamma_exp: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Inner (I) of advice 011 §5: max-min with low-exposure hard constraints.

        max_{p,t}  t
        s.t.  sum_i a_iq p_iq >= t,  for all q
              sum_q p_iq <= b_i,     for all i
              sum_i c[i,q] p_iq <= Gamma_q,  for all q   (target-side exposure)
              p >= 0

    Returns (t*, p*, lambda*, beta*, mu*) — the unified dual prices: lambda =
    sensing bottleneck price, beta = local UAV RF price, mu = exposure price.
    The LP is exact; the local net value of UAV i on target q becomes
    s_iq = lambda_q a_iq - mu_q c[i,q]  (bottleneck gain minus exposure cost).
    """
    from scipy.optimize import linprog

    K, Q = gain.shape
    n = K * Q
    c_obj = np.zeros(n + 1)
    c_obj[-1] = -1.0  # maximize t
    rows = []
    ub = []
    for q in range(Q):  # sum_i a_iq p_iq >= t  ->  -sum_i a_iq p_iq + t <= 0
        row = np.zeros(n + 1)
        row[-1] = 1.0
        for i in range(K):
            row[i * Q + q] = -gain[i, q]
        rows.append(row)
        ub.append(0.0)
    for i in range(K):  # sum_q p_iq <= b_i
        row = np.zeros(n + 1)
        row[i * Q:(i + 1) * Q] = 1.0
        rows.append(row)
        ub.append(float(budget[i]))
    for q in range(Q):  # exposure: sum_i c[i,q] p_iq <= Gamma_q (normalised to
        # unit scale so HiGHS' relative feasibility tolerance is not blown up by
        # the tiny absolute Gamma ~1e-9).
        row = np.zeros(n + 1)
        gq = max(float(gamma_exp[q]), 1e-300)
        for i in range(K):
            row[i * Q + q] = float(exposure_c[i, q]) / gq
        rows.append(row)
        ub.append(1.0)

    res = linprog(
        c_obj,
        A_ub=np.stack(rows),
        b_ub=np.asarray(ub, dtype=np.float64),
        bounds=[(0.0, None)] * (n + 1),
        method="highs",
    )
    if not res.success or res.x is None:
        return None
    p = res.x[:n].reshape(K, Q)
    t_star = float(res.x[-1])
    # Dual prices from marginals (min-LP with <= rows: price = -marginal >= 0).
    marg = np.asarray(res.ineqlin.marginals, dtype=np.float64)
    lam = np.maximum(-marg[:Q], 0.0)
    beta = np.maximum(-marg[Q:Q + K], 0.0)
    mu = np.maximum(-marg[Q + K:], 0.0)
    # Turn budget inequalities into the exact per-UAV RF equality.  Slack is
    # added to the target with the LARGEST exposure headroom (never the raw
    # best-gain target), so the exposure constraints stay respected after fill.
    p = np.maximum(p, 0.0).copy()
    c_norm = exposure_c / np.maximum(gamma_exp[None, :], 1e-300)  # (K,Q)
    for i in range(K):
        slack = float(budget[i] - np.sum(p[i]))
        while slack > 1e-10:
            e_q = np.sum(c_norm * p, axis=0)  # normalised exposure per target
            room = (1.0 - e_q) / np.maximum(c_norm[i, :], 1e-30)
            q_fill = int(np.argmax(room))
            if room[q_fill] <= 1e-12:
                break  # no exposure headroom left on any target for UAV i
            add = min(slack, max(float(room[q_fill]), 0.0))
            if add <= 1e-12:
                break
            p[i, q_fill] += add
            slack -= add
    return t_star, p, lam, beta, mu


def _safety_violated(
    uav: np.ndarray, tgt: np.ndarray,
    d_sep: float, d_standoff: float,
) -> bool:
    """Advice 011 §3 hard safety: |x_i-x_j|>=d_sep and |x_i-y_q|>=d_standoff."""
    K = int(uav.shape[0])
    for i in range(K):
        for j in range(i + 1, K):
            if float(np.linalg.norm(uav[i] - uav[j])) < d_sep - 1e-9:
                return True
    for i in range(K):
        for q in range(int(tgt.shape[0])):
            if float(np.linalg.norm(uav[i] - tgt[q])) < d_standoff - 1e-9:
                return True
    return False


def plan_seed(
    data: dict,
    row0: int,
    cfg,
    *,
    horizon: int,
    rounds: int,
    step_m: float,
    objective: str,
    inner: str,
    all_sensing: bool,
    move_penalty: float,
    assignment: bool,
    qos_floors: tuple[float, float, float, int],
    d_sep: float,
    d_standoff: float,
    gamma_exp: float | None,
    comm_coupling: bool,
    packet_bits: float,
    theta_w: dict | None,
    intercept_eps: float,
    intercept_pfa: float,
    area: tuple[float, float],
) -> dict:
    """Run the receding-horizon joint planner from one trace row's geometry."""
    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    d_tar = float(minimum_deflection_for_detection_probability(
        np.asarray([0.75]), p_fa)[0])
    d_min = float(minimum_deflection_for_detection_probability(
        np.asarray([0.60]), p_fa)[0])
    # D1.1-A robust floors (advice 010): the gauge/lex inner solvers target the
    # margin floors; the evaluation standard stays 0.60/0.70/0.80.
    gauge_xi = qos_floors
    gauge_d_min = float(minimum_deflection_for_detection_probability(
        np.asarray([float(gauge_xi[0])]), p_fa)[0])

    coeff_orig = _full_coeff(data, row0, cfg)
    uav = _uav(data, row0)
    tgt = _tgt(data, row0)
    if all_sensing:
        budget = np.ones(K, dtype=np.float64)
    elif comm_coupling:
        # advice 011 §2: b_i = 1 - P_comm^min(X) — recomputed as UAVs move.
        budget = _min_comm_power_budget(uav, cfg, packet_bits)
    else:
        budget = _budget(data, row0)
    gamma_exp_vec = (
        np.full(Q, float(gamma_exp)) if gamma_exp is not None else None)
    d_bar_intercept = (
        np.full(Q, _intercept_deflection_limit(intercept_pfa, intercept_eps))
        if theta_w is not None else None)
    coeff = coeff_orig.copy()

    # Waterfall anchors at the start geometry (same convention as
    # audit_structure_trace_physical_bottleneck: final resolved frame).
    pair = np.asarray(data["teacher_pair"][row0], dtype=bool)
    selected = tuple(tuple(int(v) for v in e) for e in np.argwhere(pair))
    d_eff = np.asarray(data["privileged_d_eff"][row0], dtype=np.float64)
    receiver_d = np.sum(d_eff * pair, axis=0)
    deployed_pd = compute_detection_probabilities(
        np.max(receiver_d, axis=0), p_fa)
    trace_gain, _ = fixed_owner_gain_matrix(coeff_orig, selected)
    power_only = solve_fixed_structure_maxmin_power_lp(trace_gain, budget)
    power_only_pd = compute_detection_probabilities(power_only.deflection, p_fa)
    single_gain, _ = priced_structure_repair(
        coeff_orig, budget, target_pair_limit=3)
    single = solve_fixed_structure_maxmin_power_lp(single_gain, budget)
    single_pd = compute_detection_probabilities(single.deflection, p_fa)
    relaxed_def = relaxed_same_geometry_target_ceiling(coeff_orig, budget)
    relaxed_pd = compute_detection_probabilities(relaxed_def, p_fa)

    # Horizon planner.  One planning step == one 0.1 s frame, so every UAV may
    # move at most step_m (= v_max*dt) in total within the step, across all
    # inner rounds.  step_budget enforces the physical kinematic constraint.
    trajectory: list[dict] = []
    total_move_m = 0.0
    for step in range(max(1, int(horizon))):
        # advice 011 §2: with comm coupling, the sensing budget follows the
        # U2U geometry (P_comm^min rises as the fleet scatters).
        if comm_coupling and not all_sensing:
            budget = _min_comm_power_budget(uav, cfg, packet_bits)
        step_budget = np.full(K, float(step_m), dtype=np.float64)
        step_moves = 0.0
        # Inner-solver closures over (gain) -> (score, deflection, power, prices).
        if inner == "gauge":
            def _eval(g):
                out = _gauge_solve(g, budget, p_fa, gauge_d_min, gauge_xi)
                if out is None:
                    ceiling = np.sum(g * budget[:, None], axis=0)
                    return (-1e6 - float(np.max(
                        np.maximum(gauge_d_min - ceiling, 0.0))),
                        ceiling, np.zeros_like(g), None)
                gamma, power, prices = out
                if float(gamma) <= 1.0 + 1e-6:
                    # Gauge power is physically feasible (within the 1 W budget).
                    return (-float(gamma), np.sum(g * power, axis=0),
                            power, prices)
                # gamma > 1: the gauge allocation would need > 1 W.  Keep -gamma
                # as the score (drive feasibility) but report the physically
                # valid best-effort max-min allocation, so the displayed QoS is
                # never inflated by a budget-violating power.
                res = solve_fixed_structure_maxmin_power_lp(g, budget)
                return (-float(gamma), res.deflection, res.power_w, res.prices)

            def _grad(k, g, o, ev):
                if ev[3] is None:
                    return deficit_geometry_gradient(
                        k, o, uav, tgt, g, budget, gauge_d_min)
                power, prices = ev[2], ev[3]
                gf = capability_geometry_gradient(g, o, uav, tgt, power, prices)
                return gf[k]
        elif inner == "lexicographic":
            # D1.1-A (advice 010): lexicographic QoS-constrained max-min.
            # Stage A: gauge proves the three margin floors are feasible.
            # Stage B: maximize the worst WITHIN the feasible region (the gauge
            # solution is itself a feasible point, so lex dominates gauge on both
            # QoS and worst by construction).  Infeasible -> reserve-first
            # max-min (worst-floor reserve, best effort).
            def _eval(g):
                out = _gauge_solve(g, budget, p_fa, gauge_d_min, gauge_xi)
                if out is not None and float(out[0]) <= 1.0 + 1e-6:
                    cs, ci = _pwl_chord(g, budget, p_fa, gauge_d_min)
                    st = _qos_constrained_maxmin(
                        g, budget, p_fa, gauge_xi, cs, ci, gauge_d_min)
                    if st is not None:
                        t, p, d = st
                        return (float(t), d, p, None)
                # Fallback: reserve-first max-min (worst-floor reserve, best
                # effort); if the reserve is unreachable, plain max-min.
                try:
                    res = solve_fixed_structure_maxmin_power_lp(
                        g, budget,
                        minimum_deflection=np.full(Q, gauge_d_min))
                except RuntimeError:
                    res = solve_fixed_structure_maxmin_power_lp(g, budget)
                return (float(np.min(res.deflection)), res.deflection,
                        res.power_w, res.prices)

            def _grad(k, g, o, ev):
                out = _gauge_solve(g, budget, p_fa, gauge_d_min, gauge_xi)
                feasible = (
                    out is not None and float(out[0]) <= 1.0 + 1e-6)
                if feasible:
                    res = solve_fixed_structure_maxmin_power_lp(g, budget)
                    return maxmin_geometry_gradient(
                        k, o, uav, tgt, g, res.power_w, res.prices)
                if out is None:
                    return deficit_geometry_gradient(
                        k, o, uav, tgt, g, budget, gauge_d_min)
                _, power, prices = out
                gf = capability_geometry_gradient(g, o, uav, tgt, power, prices)
                return gf[k]
        elif inner == "exposure":
            # advice 011 §5: unified inner (I) — max-min with low-exposure
            # constraints.  Score = max-min worst; the exposure LP enforces
            # E_q = sum_i c[i,q] p_iq <= Gamma_q at the current geometry and
            # returns the unified dual prices (lambda, beta, mu).  Geometry is
            # driven by the max-min dual gradient direction; the exact exposure
            # LP scores every candidate.
            def _eval(g):
                if gamma_exp_vec is None:
                    res = solve_fixed_structure_maxmin_power_lp(g, budget)
                    return (float(np.min(res.deflection)), res.deflection,
                            res.power_w, res.prices)
                c = _exposure_coefficients(uav, tgt, cfg)
                out = _exposure_maxmin_lp(g, budget, c, gamma_exp_vec)
                if out is None:
                    ceiling = np.sum(g * budget[:, None], axis=0)
                    return (-1e6 - float(np.max(np.maximum(
                        gauge_d_min - ceiling, 0.0))),
                        ceiling, np.zeros_like(g), None)
                t_star, p, lam, beta, mu = out
                return (float(t_star), np.sum(g * p, axis=0), p, lam)

            def _grad(k, g, o, ev):
                ceiling = np.sum(g * budget[:, None], axis=0)
                if np.any(ceiling < gauge_d_min - 1e-9):
                    return deficit_geometry_gradient(
                        k, o, uav, tgt, g, budget, gauge_d_min)
                res = solve_fixed_structure_maxmin_power_lp(g, budget)
                return maxmin_geometry_gradient(
                    k, o, uav, tgt, g, res.power_w, res.prices)
        elif inner == "intercept":
            # advice 012: counter-detection-constrained max-min.  Same LP
            # structure as exposure, but the per-watt coefficient is the
            # opponent's detection Deflection gain a^I (opponent capability
            # theta_w) and the bound is D_bar^I from (P_FA^I, eps).  The dual
            # price mu_w is now the *detection-capability cost* of each watt.
            # a^I is recomputed from the CURRENT uav geometry on every
            # evaluation (same first-order convention as exposure c), so the
            # "standoff" knob couples naturally through the constraint.
            def _eval(g):
                aI = _intercept_coefficients(uav, tgt, cfg, theta_w)
                out = _exposure_maxmin_lp(g, budget, aI, d_bar_intercept)
                if out is None:
                    ceiling = np.sum(g * budget[:, None], axis=0)
                    return (-1e6 - float(np.max(np.maximum(
                        gauge_d_min - ceiling, 0.0))),
                        ceiling, np.zeros_like(g), None)
                t_star, p, lam, beta, mu = out
                return (float(t_star), np.sum(g * p, axis=0), p, mu)

            def _grad(k, g, o, ev):
                ceiling = np.sum(g * budget[:, None], axis=0)
                if np.any(ceiling < gauge_d_min - 1e-9):
                    return deficit_geometry_gradient(
                        k, o, uav, tgt, g, budget, gauge_d_min)
                res = solve_fixed_structure_maxmin_power_lp(g, budget)
                return maxmin_geometry_gradient(
                    k, o, uav, tgt, g, res.power_w, res.prices)
        elif inner == "hybrid":
            # D1.0-B hybrid: score = max-min worst DEFLECTION minus a hard
            # penalty when the gauge floors are unmet (gamma > 1).  When the
            # floors are feasible the planner pushes the worst exactly like the
            # max-min inner; when infeasible the penalty dominates so it first
            # restores feasibility.  The trajectory therefore shows the
            # realized max-min worst, consistent with the score.
            def _eval(g):
                res = solve_fixed_structure_maxmin_power_lp(g, budget)
                mm = float(np.min(res.deflection))
                out = _gauge_solve(g, budget, p_fa, d_min)
                if out is None:
                    penalty = FLOOR_PENALTY * 10.0
                else:
                    penalty = FLOOR_PENALTY * max(float(out[0]) - 1.0, 0.0)
                return (mm - penalty, res.deflection, res.power_w, res.prices)

            def _grad(k, g, o, ev):
                ceiling = np.sum(g * budget[:, None], axis=0)
                out = _gauge_solve(g, budget, p_fa, d_min)
                feasible = out is not None and float(out[0]) <= 1.0 + 1e-6
                if feasible:
                    return maxmin_geometry_gradient(
                        k, o, uav, tgt, g, ev[2], ev[3])
                if out is None:
                    return deficit_geometry_gradient(
                        k, o, uav, tgt, g, budget, d_min)
                _, power, prices = out
                gf = capability_geometry_gradient(g, o, uav, tgt, power, prices)
                return gf[k]
        else:  # maxmin
            def _eval(g):
                res = solve_fixed_structure_maxmin_power_lp(g, budget)
                return (_score_deflection(res.deflection, objective, d_tar),
                        res.deflection, res.power_w, res.prices)

            def _grad(k, g, o, ev):
                ceiling = np.sum(g * budget[:, None], axis=0)
                if np.any(ceiling < d_min - 1e-9):
                    return deficit_geometry_gradient(
                        k, o, uav, tgt, g, budget, d_min)
                _, _, power, prices = ev
                return maxmin_geometry_gradient(k, o, uav, tgt, g, power, prices)

        for _round in range(max(1, int(rounds))):
            gain, owner = priced_structure_repair(
                coeff, budget, target_pair_limit=3)
            ev = _eval(gain)
            score_cur = float(ev[0])
            # D1.0-C joint responsibility move: assign each UAV a target and
            # move the whole fleet together, breaking the max-min equalization
            # plateau that stalls single-UAV coordinate ascent (seed 591 etc.).
            if assignment:
                assign = greedy_target_assignment(
                    gain, budget, uav, tgt, ev[3], d_min)
                jd = np.zeros((K, 2), dtype=np.float64)
                any_move = False
                for k in range(K):
                    q = int(assign[k])
                    if q < 0 or step_budget[k] <= 1e-9:
                        continue
                    dvec = tgt[q] - uav[k]
                    n = float(np.linalg.norm(dvec))
                    if n <= 1e-9:
                        continue
                    step = min(float(step_budget[k]), step_m)
                    jd[k] = dvec * (step / n)
                    any_move = True
                if any_move:
                    nu = np.clip(uav + jd, 0.0, area)
                    if (d_sep > 0.0 or d_standoff > 0.0) and _safety_violated(
                            nu, tgt, d_sep, d_standoff):
                        any_move = False
                    else:
                        ng = rescale_fixed_gain(gain, owner, uav, tgt, nu)
                        nev = _eval(ng)
                        cscore = float(nev[0])
                        if move_penalty > 0.0:
                            cscore -= move_penalty * float(np.sum(
                                np.linalg.norm(jd, axis=1)))
                        if cscore > score_cur + 1e-9:
                            old_uav = uav.copy()
                            uav = nu
                            coeff = friis_rescale_tensor(coeff, old_uav, tgt, uav)
                            gain = ng
                            ev = nev
                            step_budget -= np.linalg.norm(jd, axis=1)
                            step_moves += float(np.sum(
                                np.linalg.norm(jd, axis=1)))
                            score_cur = cscore
            for k in range(K):
                if step_budget[k] <= 1e-9:
                    continue
                grad_k = _grad(k, gain, owner, ev)
                best_score = score_cur
                best_delta: np.ndarray | None = None
                best_gain = gain
                best_ev = ev
                for delta in candidate_moves(grad_k, step_m):
                    nrm = float(np.linalg.norm(delta))
                    if nrm > step_budget[k]:
                        if nrm <= 1e-12:
                            continue
                        delta = delta * (step_budget[k] / nrm)
                    nu = uav.copy()
                    nu[k] = np.clip(uav[k] + delta, 0.0, area)
                    # Hard safety constraints (advice 011 §3): reject candidates
                    # that violate UAV separation or target standoff.
                    if (d_sep > 0.0 or d_standoff > 0.0) and _safety_violated(
                            nu, tgt, d_sep, d_standoff):
                        continue
                    ng = rescale_fixed_gain(gain, owner, uav, tgt, nu)
                    nev = _eval(ng)
                    cand_score = float(nev[0])
                    if move_penalty > 0.0 and np.any(delta != 0.0):
                        cand_score -= move_penalty * float(np.linalg.norm(delta))
                    if cand_score > best_score + 1e-9:
                        best_score = cand_score
                        best_delta = delta
                        best_gain = ng
                        best_ev = nev
                if best_delta is not None and np.any(best_delta != 0.0):
                    old_uav = uav.copy()
                    uav[k] = np.clip(uav[k] + best_delta, 0.0, area)
                    coeff = friis_rescale_tensor(coeff, old_uav, tgt, uav)
                    gain = best_gain
                    ev = best_ev
                    step_budget[k] -= float(np.linalg.norm(best_delta))
                    step_moves += float(np.linalg.norm(best_delta))
                    score_cur = best_score
        total_move_m += step_moves
        # Best structure+power at the end of this planning step (trajectory PD).
        # For the constrained inners (exposure / intercept) the honest PD is
        # the constrained allocation the planner actually deploys, not the
        # unconstrained max-min LP (which would overstate QoS while the system
        # is forced to whisper).
        gain, owner = priced_structure_repair(coeff, budget, target_pair_limit=3)
        if inner in ("exposure", "intercept"):
            nev = _eval(gain)
            deflection = np.asarray(nev[1], dtype=np.float64)
        elif inner == "gauge":
            gout = _gauge_solve(gain, budget, p_fa, d_min)
            if gout is not None:
                _, gpower, _ = gout
                deflection = np.sum(gain * gpower, axis=0)
            else:
                deflection = solve_fixed_structure_maxmin_power_lp(
                    gain, budget).deflection
        else:
            deflection = solve_fixed_structure_maxmin_power_lp(
                gain, budget).deflection
        pd = compute_detection_probabilities(deflection, p_fa)
        ordered = np.sort(pd)
        trajectory.append({
            "step": step + 1,
            "pd": pd.tolist(),
            "worst": float(ordered[0]),
            "weak3": float(np.mean(ordered[:min(3, Q)])),
            "steady": float(np.mean(ordered)),
            "worst_deflection": float(np.min(deflection)),
            "cumulative_move_m": float(total_move_m),
        })

    # Steady window: last min(horizon, 20) planned steps, aggregated per the
    # evaluation convention (trainer.py): per-target temporal mean over the
    # window, then worst = min, weak3 = bottom-3 mean, steady = overall mean.
    window = trajectory[-min(max(1, int(horizon)), 20):]
    steady_per_target = np.mean(
        np.asarray([entry["pd"] for entry in window], dtype=np.float64), axis=0)
    sorted_q = np.sort(steady_per_target)
    # advice 012 §11: realized opponent detection audit.  With the FINAL power
    # allocation p* actually used by the planner at the end of the horizon and
    # the FINAL geometry, compute the realized intercept Deflection
    # D^I_q = sum_i a^I[i,q] p*_iq and the opponent's detection probability
    # P_D^I_q = Q(Q^-1(P_FA^I) - sqrt(D^I_q)).  This is reported for EVERY
    # inner mode (maxmin / exposure / intercept) so the three-way comparison
    # shows that power-only and exposure-constrained fail to keep P_D^I <= eps
    # while detection-constrained always does.
    if theta_w is not None and ev is not None and ev[2] is not None:
        p_final = np.maximum(ev[2], 0.0)
        aI_final = _intercept_coefficients(uav, tgt, cfg, theta_w)
        d_intercept = np.sum(aI_final * p_final, axis=0)  # (Q,)
        pd_intercept = compute_detection_probabilities(d_intercept, intercept_pfa)
        intercept_pd_max = float(np.max(pd_intercept))
        intercept_pd_mean = float(np.mean(pd_intercept))
        intercept_d_max = float(np.max(d_intercept))
        intercept_violated = bool(intercept_pd_max > intercept_eps + 1e-9)
    else:
        intercept_pd_max = intercept_pd_mean = intercept_d_max = None
        intercept_violated = False
    return {
        "seed": int(np.asarray(data["seed"][row0]).reshape(-1)[0]),
        "frame": int(np.asarray(data["frame"][row0]).reshape(-1)[0]),
        "deployed_worst": float(np.min(deployed_pd)),
        "power_only_worst": float(np.min(power_only_pd)),
        "single_worst": float(np.min(single_pd)),
        "relaxed_worst": float(np.min(relaxed_pd)),
        "final_worst": float(trajectory[-1]["worst"]),
        "steady_window_worst": float(sorted_q[0]),
        "steady_window_weak3": float(np.mean(sorted_q[:min(3, Q)])),
        "steady_window_steady": float(np.mean(sorted_q)),
        "total_move_m": float(total_move_m),
        "trajectory": trajectory,
        "qos_feasible": bool(
            float(np.mean(sorted_q)) >= 0.80 - 1e-6
            and float(np.mean(sorted_q[:min(3, Q)])) >= 0.70 - 1e-6
            and float(sorted_q[0]) >= 0.60 - 1e-6),
        "intercept_pd_max": intercept_pd_max,
        "intercept_pd_mean": intercept_pd_mean,
        "intercept_d_max": intercept_d_max,
        "intercept_violated": intercept_violated,
    }


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------

def run(
    trace_path: Path,
    config_path: Path,
    *,
    seed_limit: int,
    seeds: tuple[int, ...],
    horizon: int,
    rounds: int,
    step_m: float,
    objective: str,
    inner: str,
    start: str,
    all_sensing: bool,
    move_penalty: float,
    assignment: bool,
    qos_floors: tuple[float, float, float, int],
    d_sep: float,
    d_standoff: float,
    gamma_exp: float | None,
    comm_coupling: bool,
    packet_bits: float,
    theta_w: dict | None,
    intercept_eps: float,
    intercept_pfa: float,
) -> dict:
    with np.load(trace_path, allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    cfg = load_config(str(config_path))
    seeds_arr = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    resolved = np.asarray(data["p0_resolved"], dtype=bool)
    raw_order = list(dict.fromkeys(int(s) for s in seeds_arr.reshape(-1)))
    if seeds:
        order = [s for s in raw_order if s in set(int(x) for x in seeds)]
        if not order:
            raise ValueError(f"none of the requested seeds {seeds} are in the trace")
    else:
        order = raw_order[: max(1, int(seed_limit))]
    area = tuple(float(v) for v in cfg.scenario.region_size)

    rows = []
    t0 = time.time()
    for seed in order:
        idx = np.flatnonzero(seeds_arr == seed)
        if start == "first":
            row0 = int(idx[np.argmin(frames[idx])])
        else:  # final resolved frame (waterfall convention)
            cand = idx[resolved[idx]]
            if not len(cand):
                raise ValueError(f"seed {seed} has no resolved frame")
            row0 = int(cand[np.argmax(frames[cand])])
        rows.append(plan_seed(
            data, row0, cfg,
            horizon=horizon, rounds=rounds, step_m=step_m,
            objective=objective, inner=inner, all_sensing=all_sensing,
            move_penalty=move_penalty, assignment=assignment,
            qos_floors=qos_floors, d_sep=d_sep, d_standoff=d_standoff,
            gamma_exp=gamma_exp, comm_coupling=comm_coupling,
            packet_bits=packet_bits, theta_w=theta_w,
            intercept_eps=intercept_eps, intercept_pfa=intercept_pfa,
            area=area,
        ))

    def _mean(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows]))

    return {
        "schema_version": 1,
        "trace": str(trace_path),
        "config": str(config_path),
        "start": start,
        "seed_limit": len(rows),
        "horizon": int(horizon),
        "rounds": int(rounds),
        "step_m": float(step_m),
        "objective": objective,
        "inner": inner,
        "all_sensing": bool(all_sensing),
        "move_penalty": float(move_penalty),
        "assignment": bool(assignment),
        "qos_floors": list(qos_floors),
        "duration_s": float(time.time() - t0),
        "mean": {
            "deployed_worst": _mean("deployed_worst"),
            "power_only_worst": _mean("power_only_worst"),
            "single_worst": _mean("single_worst"),
            "relaxed_worst": _mean("relaxed_worst"),
            "final_worst": _mean("final_worst"),
            "steady_window_worst": _mean("steady_window_worst"),
            "steady_window_weak3": _mean("steady_window_weak3"),
            "steady_window_steady": _mean("steady_window_steady"),
            "total_move_m": _mean("total_move_m"),
            "intercept_pd_max": float(np.mean([
                float(row["intercept_pd_max"])
                for row in rows if row["intercept_pd_max"] is not None
            ])) if any(row["intercept_pd_max"] is not None for row in rows) else None,
            "intercept_pd_mean": float(np.mean([
                float(row["intercept_pd_mean"])
                for row in rows if row["intercept_pd_mean"] is not None
            ])) if any(row["intercept_pd_mean"] is not None for row in rows) else None,
            "intercept_d_max": float(np.mean([
                float(row["intercept_d_max"])
                for row in rows if row["intercept_d_max"] is not None
            ])) if any(row["intercept_d_max"] is not None for row in rows) else None,
        },
        "intercept_violated_rate": float(np.mean([
            float(row["intercept_violated"]) for row in rows
        ])) if any(row["intercept_violated"] for row in rows) else 0.0,
        "qos_feasible_rate": float(np.mean([
            float(row["qos_feasible"]) for row in rows
        ])),
        "worst_ge_0p72_rate": float(np.mean([
            float(row["steady_window_worst"]) >= 0.72 for row in rows
        ])),
        "worst_ge_0p75_rate": float(np.mean([
            float(row["steady_window_worst"]) >= 0.75 for row in rows
        ])),
        "rows": rows,
        "note": (
            "ORACLE diagnostic: global information, exact structure+power at "
            "every candidate, multi-candidate trust-region geometry; the DD "
            "gate g_dd and chi_rep are frozen at trace values (Friis rescale "
            "only); targets static."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--seed-limit", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="restrict to specific trace seeds (overrides seed-limit)")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--step-m", type=float, default=2.5)
    ap.add_argument("--objective", choices=("maxmin", "tail"), default="maxmin")
    ap.add_argument("--inner", choices=("maxmin", "gauge", "hybrid",
                                        "lexicographic", "exposure", "intercept"),
                    default="maxmin",
                    help="inner power solver: max-min LP (worst-only), the "
                         "capability gauge (three floors), hybrid (maximize "
                         "worst with hard floor penalty), lexicographic "
                         "QoS-constrained max-min (advice 010), exposure "
                         "(advice 011 low-exposure max-min), or intercept "
                         "(advice 012 detection-capability-constrained)")
    ap.add_argument("--start", choices=("final", "first"), default="final")
    ap.add_argument("--all-sensing", action="store_true",
                    help="use b_i = 1 for every UAV (no comm power) instead of "
                         "the trace-recorded sensing budget")
    ap.add_argument("--move-penalty", type=float, default=0.0)
    ap.add_argument("--qos-floors", type=str, default="0.61,0.71,0.81,3",
                    help="gauge/lex inner-solver floor targets "
                         "[worst,weak3,steady,k] (D1.x robust margin); the "
                         "evaluation standard stays 0.60/0.70/0.80")
    ap.add_argument("--assignment", action="store_true",
                    help="D1.0-C: greedy target responsibility auction + joint "
                         "fleet move each round (breaks max-min equalization "
                         "plateaus)")
    # advice 011 / T2 Central Oracle options.
    ap.add_argument("--d-sep", type=float, default=0.0,
                    help="UAV-UAV separation hard constraint (m); 0=off")
    ap.add_argument("--d-standoff", type=float, default=0.0,
                    help="UAV-target standoff hard constraint (m); 0=off")
    ap.add_argument("--exposure-gamma", type=float, default=None,
                    help="target-side exposure limit Gamma (W at the target); "
                         "None=off.  With --inner exposure, enforces "
                         "E_q = sum_i c[i,q] p_iq <= Gamma in the power LP.")
    ap.add_argument("--comm-coupling", action="store_true",
                    help="b_i = 1 - P_comm^min(X) analytic comm-sensing budget "
                         "(recomputed as the fleet moves)")
    ap.add_argument("--packet-bits", type=float, default=600.0,
                    help="nominal U2U packet bits for the comm-coupling budget")
    # advice 012: opponent detection-capability constraint.
    ap.add_argument("--intercept-capability", choices=tuple(OPPONENT_CAPABILITIES),
                    default=None,
                    help="opponent capability tier for --inner intercept "
                         "(weak/medium/strong); None=off")
    ap.add_argument("--intercept-eps", type=float, default=0.1,
                    help="allowed opponent detection probability eps_w")
    ap.add_argument("--intercept-pfa", type=float, default=1e-3,
                    help="opponent false-alarm probability P_FA,w^I")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    raw_floors = [float(v) for v in args.qos_floors.split(",")]
    if len(raw_floors) != 4 or raw_floors[3] < 1:
        raise SystemExit("--qos-floors must be [worst,weak3,steady,k]")
    qos_floors = (raw_floors[0], raw_floors[1], raw_floors[2], int(raw_floors[3]))
    result = run(
        args.trace, args.config,
        seed_limit=args.seed_limit,
        seeds=tuple(args.seeds) if args.seeds else (),
        horizon=args.horizon, rounds=args.rounds,
        step_m=args.step_m, objective=args.objective, inner=args.inner,
        start=args.start, all_sensing=args.all_sensing,
        move_penalty=args.move_penalty, assignment=args.assignment,
        qos_floors=qos_floors,
        d_sep=float(args.d_sep), d_standoff=float(args.d_standoff),
        gamma_exp=(float(args.exposure_gamma)
                   if args.exposure_gamma is not None else None),
        comm_coupling=bool(args.comm_coupling),
        packet_bits=float(args.packet_bits),
        theta_w=(dict(OPPONENT_CAPABILITIES[args.intercept_capability])
                 if args.intercept_capability is not None else None),
        intercept_eps=float(args.intercept_eps),
        intercept_pfa=float(args.intercept_pfa),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "mean": result["mean"],
        "qos_feasible_rate": result["qos_feasible_rate"],
        "worst_ge_0p72_rate": result["worst_ge_0p72_rate"],
        "worst_ge_0p75_rate": result["worst_ge_0p75_rate"],
        "duration_s": result["duration_s"],
        "rows": [{
            key: row[key]
            for key in (
                "seed", "frame", "deployed_worst", "power_only_worst",
                "single_worst", "relaxed_worst", "final_worst",
                "steady_window_worst", "steady_window_weak3",
                "steady_window_steady", "total_move_m", "qos_feasible",
            )
        } for row in result["rows"]],
    }, indent=2))


if __name__ == "__main__":
    main()
