#!/usr/bin/env python
"""D0.93-E: epoch-level capability envelope and three-state routing.

Turns the per-frame capability oracle gamma*_t into an *epoch-level* certificate
using the monotonicity of gamma*(A, b) in (A, b) (advice/005.md).  Over each
control window we form elementwise coefficient/budget bounds
A^- <= A_t <= A^+ and b^- <= b_t <= b^+, solve the optimistic/conservative
gauges gamma^L = gamma*(A^+, b^+) and gamma^U = gamma*(A^-, b^-), and verify

    gamma^L <= gamma*_t <= gamma^U   for every frame t in the window.

Three-state routing (no empirical threshold):
    gamma^U <= 1        -> certified stay (power-only)
    gamma^L >  1        -> certified escalate (structure)
    otherwise           -> ambiguous (defer/refine)

This avoids per-frame structure churn: only a *persistent* certified deficit
escalates, while instantaneous hard frames and uncertainty do not.
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
    rate = np.asarray(data["outgoing_rate"][row], dtype=np.int64)
    mask = np.asarray(data["outgoing_token_mask"][row], dtype=bool)
    active = (rate > 0) & np.any(mask, axis=1)
    frac = np.clip(np.asarray(data["comm_fraction"][row], dtype=np.float64), 0, 1)
    budget = 1.0 - np.where(active, frac, 0.0)
    return gain, budget


def run(trace_path, config_path, *, xi, horizon, seed_limit):
    with np.load(trace_path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    cfg = load_config(str(config_path))
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    order = list(dict.fromkeys(int(s) for s in seeds.reshape(-1)))[:seed_limit]

    # Per-frame oracle gauges.
    per_frame_gamma = []
    # Envelope stats.
    sandwich_violations = 0
    sandwich_checks = 0
    states = {"stay": 0, "escalate": 0, "ambiguous": 0}
    per_frame_escalate = 0  # frames with gamma*_t > 1

    for seed in order:
        rows = np.flatnonzero(seeds == seed)
        rows = rows[np.argsort(frames[rows])]
        # Collect per-frame gain/budget for this episode.
        gains = []
        budgets = []
        for row in rows:
            gain, budget = _gain_and_budget(data, int(row), cfg)
            gains.append(gain)
            budgets.append(budget)
        K, Q = gains[0].shape
        # Per-frame oracle.
        frame_gammas = []
        for gain, budget in zip(gains, budgets):
            out = capability_gauge(gain, budget, p_fa, xi)
            frame_gammas.append(None if out is None else out[0])
        # Epoch envelopes.
        for start in range(0, len(rows), max(1, horizon)):
            chunk_g = gains[start:start + horizon]
            chunk_b = budgets[start:start + horizon]
            if not chunk_g:
                continue
            A_min = np.min(np.stack(chunk_g), axis=0)
            A_max = np.max(np.stack(chunk_g), axis=0)
            b_min = np.min(np.stack(chunk_b), axis=0)
            b_max = np.max(np.stack(chunk_b), axis=0)
            lo = capability_gauge(A_max, b_max, p_fa, xi)
            hi = capability_gauge(A_min, b_min, p_fa, xi)
            gamma_L = lo[0] if lo is not None else float("inf")
            gamma_U = hi[0] if hi is not None else float("inf")

            # Verify the sandwich on each frame of this epoch.
            for idx in range(start, min(start + horizon, len(rows))):
                gt = frame_gammas[idx]
                if gt is None:
                    continue
                sandwich_checks += 1
                per_frame_gamma.append(gt)
                if gt > 1.0:
                    per_frame_escalate += 1
                if not (gamma_L - 1e-6 <= gt <= gamma_U + 1e-6):
                    sandwich_violations += 1

            if gamma_U <= 1.0:
                states["stay"] += 1
            elif gamma_L > 1.0:
                states["escalate"] += 1
            else:
                states["ambiguous"] += 1

    return {
        "schema_version": 1,
        "trace": str(trace_path),
        "config": str(config_path),
        "horizon_frames": int(horizon),
        "task_xi": {"rho_min": float(xi[0]), "rho_tail": float(xi[1]),
                    "rho_avg": float(xi[2]), "k": int(xi[3])},
        "sandwich_checks": int(sandwich_checks),
        "sandwich_violations": int(sandwich_violations),
        "sandwich_holds": bool(sandwich_violations == 0),
        "epoch_states": states,
        "epoch_states_rate": {k: v / max(1, sum(states.values()))
                              for k, v in states.items()},
        "per_frame_escalate_frames": int(per_frame_escalate),
        "epoch_escalate_frames": int(states["escalate"] * horizon),
        "saved_structure_decisions": int(
            per_frame_escalate - states["escalate"] * horizon),
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
