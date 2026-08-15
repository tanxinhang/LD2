#!/usr/bin/env python
"""D0.94-L3D closed-loop: distributed price-mediated local capability descent.

On the real 8/8 hard frames, run a short closed loop of
    price (peer-to-peer consensus, T0) -> local gradient (T2) -> local move
    -> Friis rescale -> re-solve capability gauge,
and report how many borderline hard states (gamma slightly > 1) reach gamma <= 1,
versus no movement.  This demonstrates the concrete advantage of the distributed
geometry layer in simulation, not just the unit-level gradient checks.
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
from uav_isac.coordination.capability import (  # noqa: E402
    capability_gauge_pwl_lp_full,
    local_capability_gradient_k,
)
from uav_isac.coordination.maxmin_power import fixed_owner_gain_matrix  # noqa: E402
from uav_isac.coordination.pwl_pd import (  # noqa: E402
    chord_lower_bound,
    curvature_breakpoints,
)
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)
from uav_isac.physical.detection import (  # noqa: E402
    minimum_deflection_for_detection_probability,
)

XI = (0.60, 0.70, 0.80, 3)


def _solve(gain, budget, p_fa):
    d_min = float(minimum_deflection_for_detection_probability(
        np.asarray([XI[0]]), p_fa)[0])
    ceiling = np.sum(gain * budget[:, None], axis=0)
    if np.any(ceiling < d_min - 1e-9):
        return None
    d_max = float(np.max(ceiling)) + 1.0
    bps = curvature_breakpoints(p_fa, d_min, d_max, epsilon=1e-3)
    cs, ci = chord_lower_bound(p_fa, bps)
    return capability_gauge_pwl_lp_full(gain, budget, p_fa, XI, cs, ci, d_min)


def _frame(data, row, cfg):
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
    gain, owner = fixed_owner_gain_matrix(coefficient, selected)
    uav = np.asarray(data["uav_positions"][row], dtype=np.float64)[:, :2]
    tgt = np.asarray(data["target_states"][row], dtype=np.float64)[:, :2]
    rate = np.asarray(data["outgoing_rate"][row], dtype=np.int64)
    mask = np.asarray(data["outgoing_token_mask"][row], dtype=bool)
    active = (rate > 0) & np.any(mask, axis=1)
    frac = np.clip(np.asarray(data["comm_fraction"][row], dtype=np.float64), 0, 1)
    budget = 1.0 - np.where(active, frac, 0.0)
    return gain, owner, uav, tgt, budget


def _friis_rescale(gain, owner, uav, tgt, new_uav):
    rtx = np.linalg.norm(uav[:, None, :] - tgt[None, :, :], axis=2)
    rrx = np.linalg.norm(uav[owner] - tgt, axis=1)
    rtx_n = np.linalg.norm(new_uav[:, None, :] - tgt[None, :, :], axis=2)
    rrx_n = np.linalg.norm(new_uav[owner] - tgt, axis=1)
    c = gain * (rtx ** 2) * (rrx[None, :] ** 2)
    return c / (rtx_n ** 2 * rrx_n[None, :] ** 2)


def run(trace_path, config_path, *, horizon, step_m, seed_limit, max_frames):
    with np.load(trace_path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    cfg = load_config(str(config_path))
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    order = list(dict.fromkeys(int(s) for s in seeds.reshape(-1)))[:seed_limit]

    hard = []
    for seed in order:
        rows = np.flatnonzero(seeds == seed)
        rows = rows[np.argsort(frames[rows])]
        for row in rows:
            gain, owner, uav, tgt, budget = _frame(data, int(row), cfg)
            out = _solve(gain, budget, p_fa)
            if out is not None and 1.0 < out[0] <= 1.5:
                hard.append((int(row), gain, owner, uav, tgt, budget))
    rng = np.random.default_rng(0)
    if len(hard) > max_frames:
        hard = [hard[i] for i in rng.choice(len(hard), max_frames, replace=False)]

    reached = 0
    decreases = []
    K, Q = hard[0][1].shape if hard else (0, 0)
    for row, gain, owner, uav, tgt, budget in hard:
        out = _solve(gain, budget, p_fa)
        if out is None:
            continue
        gamma0, power, prices = out
        for _ in range(horizon):
            # Distributed local gradient (T2): each UAV from local info.
            support = {q: {i: (power[i, q], gain[i, q]) for i in range(K)}
                       for q in range(Q)}
            delta = np.zeros_like(uav)
            for k in range(K):
                g_k = local_capability_gradient_k(
                    k, owner, prices, power[k], gain[k], uav[k], tgt, support)
                n = np.linalg.norm(g_k)
                if n > 1e-12:
                    delta[k] = -step_m * g_k / n
            new_uav = uav + delta
            new_gain = _friis_rescale(gain, owner, uav, tgt, new_uav)
            out2 = _solve(new_gain, budget, p_fa)
            if out2 is None:
                break
            if out2[0] >= out[0] - 1e-9:
                break
            gain, uav, out = new_gain, new_uav, out2
            power, prices = out[1], out[2]
        decreases.append(gamma0 - out[0])
        if out[0] <= 1.0:
            reached += 1

    n = len(decreases)
    return {
        "schema_version": 1,
        "trace": str(trace_path),
        "config": str(config_path),
        "borderline_frames": len(hard),
        "solved_frames": n,
        "reached_gamma_le_1": int(reached),
        "reach_rate": reached / max(1, n),
        "mean_gamma_decrease": float(np.mean(decreases)) if decreases else float("nan"),
        "horizon": int(horizon),
        "step_m": float(step_m),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--seed-limit", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--step-m", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=12)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    result = run(args.trace, args.config, horizon=max(1, args.horizon),
                 step_m=float(args.step_m), seed_limit=max(1, args.seed_limit),
                 max_frames=max(1, args.max_frames))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
