#!/usr/bin/env python
"""D0.93-F (L0): communication slack recovery audit.

Before attributing gamma_P* > 1 to "structure insufficient", reclaim any
over-allocated communication power.  Keeping the transport semantics EXACTLY
unchanged (active Tx set, token content/quantisation, packet bits, receiver
set, bandwidth split, role/owner/structure, deadline, delivery criterion), the
minimum broadcast power per sender is the max over its required receivers of

    P_ij^req = Gamma_req * N0 * B_eff / g_ij,
    Gamma_req = max(Gamma_th, 2^(L_i/(B_eff*(T_ddl-T_proc))) - 1),
    g_ij = antenna_gain * (lambda/(4*pi*d_ij))^2,  B_eff = B / n_active.

This is a direct inversion of the current orthogonal-U2U Shannon link model,
not a heuristic.  The reclaimed budget b'_i = 1 - P_comm,i^min >= b_i is then
fed back into the capability gauge; by monotonicity gamma*(A,b') <= gamma*(A,b)
(strictly no-harm for sensing).  The diagnostic output is the transition of the
"certified escalate" epochs: how many become stay / ambiguous after L0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from uav_isac.coordination.capability import capability_gauge  # noqa: E402
from uav_isac.coordination.maxmin_power import fixed_owner_gain_matrix  # noqa: E402
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)


def _gain_and_budget(data, row, cfg):
    coefficient = per_watt_deflection_tensor_from_observables(
        np.asarray(data["privileged_alpha"][row], dtype=np.float64),
        np.asarray(data["privileged_g_dd"][row], dtype=np.float64),
        np.asarray(data["privileged_chi_rep"][row], dtype=np.float64),
        T_sym=float(cfg.otfs.T_sym), M=int(cfg.otfs.M), N=int(cfg.otfs.N),
        kT=float(cfg.channel.kT), bandwidth_hz=float(cfg.otfs.B),
        noise_figure_db=float(cfg.channel.NF),
        g_tx_dbi=float(cfg.otfs.g_tx_dBi), g_rx_dbi=float(cfg.otfs.g_rx_dBi),
        n_cpi=int(cfg.otfs.n_cpi), g_min=float(cfg.detection.g_min),
        use_swerling=bool(cfg.channel.use_swerling),
    )
    pair = np.asarray(data["teacher_pair"][row], dtype=bool)
    selected = tuple(tuple(int(v) for v in e) for e in np.argwhere(pair))
    gain, _ = fixed_owner_gain_matrix(coefficient, selected)
    return gain


def _min_comm_power(data, row, cfg):
    """Analytic minimum broadcast power per sender (W), direct link inversion."""
    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    positions = np.asarray(data["uav_positions"][row], dtype=np.float64)
    rate = np.asarray(data["outgoing_rate"][row], dtype=np.int64)
    mask = np.asarray(data["outgoing_token_mask"][row], dtype=bool)
    frac = np.clip(np.asarray(data["comm_fraction"][row], dtype=np.float64), 0, 1)
    active = (rate > 0) & np.any(mask, axis=1)
    n_active = max(1, int(np.sum(active)))

    ma = cfg.marl
    wavelength = 299_792_458.0 / float(cfg.otfs.fc)
    antenna_gain = float(10.0 ** (2.0 * ma.comm_antenna_gain_dbi / 10.0))
    kT = float(cfg.channel.kT)
    NF = float(10.0 ** (cfg.channel.NF / 10.0))
    B = float(ma.comm_bandwidth_hz)
    B_eff = B / n_active
    T_ddl = float(ma.comm_deadline_s)
    T_proc = float(ma.comm_processing_delay_s)
    Gamma_th = float(10.0 ** (ma.comm_snr_threshold_db / 10.0))
    header = int(ma.comm_header_bits)
    rate_bits = list(ma.comm_rate_bits_per_dim)
    # message_dim for target tokens: Q * comm_target_token_dim.
    dims_per_token = int(ma.comm_target_token_dim)
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])

    p_min = np.zeros(K, dtype=np.float64)
    for i in range(K):
        if not active[i]:
            continue
        n_tokens = int(np.sum(mask[i] > 0.5))
        active_dims = n_tokens * dims_per_token
        bits_per_dim = rate_bits[max(0, min(int(rate[i]), len(rate_bits) - 1))]
        if bits_per_dim <= 0:
            continue
        L = header + active_dims * bits_per_dim
        if L <= 0:
            continue
        R_req = L / max(T_ddl - T_proc, 1e-12)
        Gamma_rate = float(2.0 ** (R_req / B_eff) - 1.0)
        Gamma_req = max(Gamma_th, Gamma_rate)
        N0_B = kT * B_eff * NF
        for j in range(K):
            if j == i:
                continue
            d = max(float(np.linalg.norm(positions[i] - positions[j])), 1.0)
            path_gain = (wavelength / (4.0 * np.pi * d)) ** 2
            g = antenna_gain * path_gain
            p_min[i] = max(p_min[i], Gamma_req * N0_B / max(g, 1e-30))
    return p_min, active


def run(trace_path, config_path, *, xi, horizon, seed_limit):
    with np.load(trace_path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    cfg = load_config(str(config_path))
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    order = list(dict.fromkeys(int(s) for s in seeds.reshape(-1)))[:seed_limit]

    slack_total = 0.0
    slack_frames = 0
    transition = {"escalate_to_stay": 0, "escalate_to_ambiguous": 0,
                  "escalate_to_escalate": 0}
    before_gauge = []
    after_gauge = []

    for seed in order:
        rows = np.flatnonzero(seeds == seed)
        rows = rows[np.argsort(frames[rows])]
        gains = []
        budgets = []
        budgets_after = []
        for row in rows:
            gain = _gain_and_budget(data, int(row), cfg)
            p_min, active = _min_comm_power(data, int(row), cfg)
            frac = np.clip(np.asarray(data["comm_fraction"][row], dtype=np.float64), 0, 1)
            p_actor = np.where(active, frac, 0.0)
            b_before = 1.0 - p_actor
            b_after = 1.0 - p_min  # p_min <= p_actor, so b_after >= b_before
            gains.append(gain)
            budgets.append(b_before)
            budgets_after.append(b_after)
            slack = float(np.sum(np.maximum(p_actor - p_min, 0.0)))
            slack_total += slack
            slack_frames += 1

        # Epoch envelope before/after L0.
        for start in range(0, len(rows), max(1, horizon)):
            chunk_g = gains[start:start + horizon]
            chunk_b = budgets[start:start + horizon]
            chunk_ba = budgets_after[start:start + horizon]
            if not chunk_g:
                continue
            A_min = np.min(np.stack(chunk_g), axis=0)
            A_max = np.max(np.stack(chunk_g), axis=0)
            b_min = np.min(np.stack(chunk_b), axis=0)
            b_max = np.max(np.stack(chunk_b), axis=0)
            ba_min = np.min(np.stack(chunk_ba), axis=0)
            ba_max = np.max(np.stack(chunk_ba), axis=0)
            lo = capability_gauge(A_max, b_max, p_fa, xi)
            hi = capability_gauge(A_min, b_min, p_fa, xi)
            lo_a = capability_gauge(A_max, ba_max, p_fa, xi)
            hi_a = capability_gauge(A_min, ba_min, p_fa, xi)
            gamma_L = lo[0] if lo is not None else float("inf")
            gamma_U = hi[0] if hi is not None else float("inf")
            gamma_L_a = lo_a[0] if lo_a is not None else float("inf")
            gamma_U_a = hi_a[0] if hi_a is not None else float("inf")

            before = ("stay" if gamma_U <= 1.0
                      else "escalate" if gamma_L > 1.0 else "ambiguous")
            after = ("stay" if gamma_U_a <= 1.0
                     else "escalate" if gamma_L_a > 1.0 else "ambiguous")
            before_gauge.append(gamma_L)
            after_gauge.append(gamma_L_a)
            if before == "escalate":
                transition[f"escalate_to_{after}"] += 1

    n_esc = sum(transition.values())
    recoverable = (
        (transition["escalate_to_stay"] + transition["escalate_to_ambiguous"])
        / max(1, n_esc)
    )
    return {
        "schema_version": 1,
        "trace": str(trace_path),
        "config": str(config_path),
        "horizon_frames": int(horizon),
        "mean_comm_slack_w": slack_total / max(1, slack_frames),
        "escalate_epochs": int(n_esc),
        "transition": transition,
        "comm_recoverable_ratio": float(recoverable),
        "escalate_to_stay_ratio": transition["escalate_to_stay"] / max(1, n_esc),
        "mean_gamma_L_before": float(np.mean(before_gauge)) if before_gauge else float("nan"),
        "mean_gamma_L_after": float(np.mean(after_gauge)) if after_gauge else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--seed-limit", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--rho-min", type=float, default=0.60)
    ap.add_argument("--rho-tail", type=float, default=0.70)
    ap.add_argument("--rho-avg", type=float, default=0.80)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    xi = (args.rho_min, args.rho_tail, args.rho_avg, args.k)
    result = run(args.trace, args.config, xi=xi, horizon=max(1, args.horizon),
                 seed_limit=max(1, args.seed_limit))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
