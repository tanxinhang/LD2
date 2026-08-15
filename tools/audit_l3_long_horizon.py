#!/usr/bin/env python
"""D0.94-L3H long-horizon distributed geometry: deficit -> capability descent.

The short-horizon L3 (D0.94-L3D) only touched solvable borderline frames
(gamma in (1, 1.5]).  The real performance gap is the *unsolvable* frames
(gamma = inf, i.e. the max-min ceiling D_q^max = sum_i a_iq b_i is below the
worst-target deflection floor d_min for at least one target).  Those are the
frames whose geometry (path loss) is too weak for ANY power allocation, and
they dominate the steady-P_D shortfall (0.736 vs 0.80).

This audit runs a long receding horizon on the unsolvable frames:

  Phase 1 (feasibility repair): move each UAV along the *deficit gradient*
      d/dt x_k = -sum_q  [d_min - D_q^max]_+ * b_k * d a_kq / d x_k
    (Tx term) plus the owner/Rx term.  This is the exact derivative of the
    squared max-min ceiling deficit, so it is the steepest-descent direction of
    the (differentiable, non-negative) feasibility violation, and it uses only
    broadcast deficits + delivered power/gain records -> distributed.

  Phase 2 (capability descent): once the frame becomes gauge-solvable, switch
    to the price-weighted capability gradient (T2) until gamma <= 1.

The path loss is updated exactly (Friis 1/R_tx^2 1/R_rx^2 rescale); the DD gate
g_dd and reporting link chi_rep are held at the trace values (first-order: for
static targets only ~3% of pairs are DD-gated).  Step size is capped at
v_max*dt so the horizon is a physically realisable trajectory.

Reports the gamma reach-rate, the ceiling deficit reduction, and the movement
cost, versus the no-movement baseline.
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


def _ceiling_deficit(gain, budget, d_min):
    ceiling = np.sum(gain * budget[:, None], axis=0)
    return np.maximum(0.0, d_min - ceiling), ceiling


def _deficit_gradient(k, owner, uav, tgt, gain, budget, d_min):
    """Distributed steepest descent of the squared max-min ceiling deficit."""
    Q = int(gain.shape[1])
    ceiling = np.sum(gain * budget[:, None], axis=0)
    deficit = np.maximum(0.0, d_min - ceiling)
    g = np.zeros_like(uav[k])
    # True gradient of the deficit delta_q = max(0, d_min - sum_i a_iq b_i):
    #   d delta_q / d x_k = + b_k a_kq * 2 (x_k - x_q) / R_kq^2   (Tx term)
    #                     + 1[k=owner_q] sum_i b_i a_iq * 2 (x_k - x_q) / R_kq^2
    # (the caller moves in -step * g / |g|, i.e. steepest descent toward the
    # bottleneck target, which is the physically correct inward direction).
    for q in range(Q):
        w = deficit[q] * budget[k]
        if abs(w) < 1e-15:
            continue
        a = gain[k, q]
        r = float(np.linalg.norm(uav[k] - tgt[q]))
        g += w * (2.0 * a) * (uav[k] - tgt[q]) / max(r ** 2, 1e-9)
    for q in range(Q):
        if owner[q] != k:
            continue
        r = float(np.linalg.norm(uav[k] - tgt[q]))
        for i in range(gain.shape[0]):
            w = deficit[q] * budget[i]
            if abs(w) < 1e-15:
                continue
            a = gain[i, q]
            g += w * (2.0 * a) * (uav[k] - tgt[q]) / max(r ** 2, 1e-9)
    return g


def run(trace_path, config_path, *, horizon, step_m, seed_limit, max_frames):
    with np.load(trace_path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    cfg = load_config(str(config_path))
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    d_min = float(minimum_deflection_for_detection_probability(
        np.asarray([XI[0]]), p_fa)[0])
    order = list(dict.fromkeys(int(s) for s in seeds.reshape(-1)))[:seed_limit]

    # Collect unsolvable frames (gamma = inf: ceiling below the worst floor).
    hard = []
    for seed in order:
        rows = np.flatnonzero(seeds == seed)
        rows = rows[np.argsort(frames[rows])]
        for row in rows:
            gain, owner, uav, tgt, budget = _frame(data, int(row), cfg)
            if _solve(gain, budget, p_fa) is None:
                hard.append((int(row), gain, owner, uav, tgt, budget))
    rng = np.random.default_rng(0)
    if len(hard) > max_frames:
        hard = [hard[i] for i in rng.choice(len(hard), max_frames, replace=False)]

    reached = 0
    solvable = 0
    moves = []
    deficit_reductions = []
    dd_gated = 0
    area = tuple(float(v) for v in cfg.scenario.region_size)
    for row, gain, owner, uav, tgt, budget in hard:
        K, Q = gain.shape
        # DD-gate diagnostic: target with zero gain for every transmitter is
        # DD-infeasible under the frozen-g_dd approximation (recoverable only
        # by full physics, i.e. recomputing the delay-Doppler gate).
        if np.any(np.sum(gain > 1e-15, axis=0) == 0):
            dd_gated += 1

        gain0, uav0 = gain, uav
        def_start = float(np.max(_ceiling_deficit(gain, budget, d_min)[0]))
        dist = 0.0

        def _clip(pos):
            return np.clip(pos, 0.0, area)

        def _cost(g, b):
            o = _solve(g, b, p_fa)
            if o is not None:
                return float(o[0])
            return 1e6 + float(np.max(_ceiling_deficit(g, b, d_min)[0]))

        for _ in range(horizon):
            out = _solve(gain, budget, p_fa)
            cost_cur = _cost(gain, budget)
            if out is not None and out[0] <= 1.0:
                break
            delta = np.zeros_like(uav)
            if out is None:
                # Phase 1: deficit descent (ceiling below worst floor).
                for k in range(K):
                    g_k = _deficit_gradient(k, owner, uav, tgt, gain, budget, d_min)
                    n = np.linalg.norm(g_k)
                    if n > 1e-12:
                        delta[k] = -step_m * g_k / n
            else:
                gamma_cur, power, prices = out
                support = {q: {i: (power[i, q], gain[i, q]) for i in range(K)}
                           for q in range(Q)}
                for k in range(K):
                    g_k = local_capability_gradient_k(
                        k, owner, prices, power[k], gain[k], uav[k], tgt, support)
                    n = np.linalg.norm(g_k)
                    if n > 1e-12:
                        delta[k] = -step_m * g_k / n
            new_uav = _clip(uav + delta)
            new_gain = _friis_rescale(gain, owner, uav, tgt, new_uav)
            cost_new = _cost(new_gain, budget)
            if cost_new >= cost_cur - 1e-9:
                break
            dist += float(np.linalg.norm(delta, axis=1).sum())
            gain, uav = new_gain, new_uav
        moves.append(dist)
        out = _solve(gain, budget, p_fa)
        if out is not None:
            solvable += 1
            if out[0] <= 1.0:
                reached += 1
        else:
            deficit_reductions.append(
                def_start - float(np.max(_ceiling_deficit(gain, budget, d_min)[0])))
    n = len(hard)
    return {
        "schema_version": 1,
        "trace": str(trace_path),
        "config": str(config_path),
        "unsolvable_frames": n,
        "dd_gated_frames": int(dd_gated),
        "became_solvable": int(solvable),
        "reached_gamma_le_1": int(reached),
        "reach_rate": reached / max(1, n),
        "solvable_rate": solvable / max(1, n),
        "mean_deficit_reduction": (
            float(np.mean(deficit_reductions)) if deficit_reductions else float("nan")),
        "mean_total_movement_m": float(np.mean(moves)) if moves else float("nan"),
        "max_total_movement_m": float(np.max(moves)) if moves else float("nan"),
        "horizon": int(horizon),
        "step_m": float(step_m),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--seed-limit", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--step-m", type=float, default=2.5)
    ap.add_argument("--max-frames", type=int, default=24)
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
