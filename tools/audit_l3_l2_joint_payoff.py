#!/usr/bin/env python
"""Compare single-lever vs joint payoff on hard frames: L2 vs L3 vs L2+L3.

Answers "which lever has higher payoff" with one controlled experiment:
on the SAME hard-frame sample, measure
  - L2 (structure): greedy best owner + top-3 TX, exact max-min LP (power sharing)
  - L3 (geometry): long-horizon deficit->capability descent, fixed teacher structure
  - Joint: best structure THEN the same L3 descent
and report the full-gauge (gamma <= 1) and worst-floor (>= d_min) recovery rates.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from uav_isac.coordination.maxmin_power import (  # noqa: E402
    fixed_owner_gain_matrix, solve_fixed_structure_maxmin_power_lp)
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables)
from uav_isac.physical.detection import (  # noqa: E402
    minimum_deflection_for_detection_probability)

import importlib.util  # reuse the long-horizon audit helpers
_spec = importlib.util.spec_from_file_location(
    "lh", str(ROOT / "tools" / "audit_l3_long_horizon.py"))
_lh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lh)

XI = (0.60, 0.70, 0.80, 3)


def _full_coeff(data, row, cfg):
    return per_watt_deflection_tensor_from_observables(
        np.asarray(data["privileged_alpha"][row], dtype=np.float64),
        np.asarray(data["privileged_g_dd"][row], dtype=np.float64),
        np.asarray(data["privileged_chi_rep"][row], dtype=np.float64),
        T_sym=float(cfg.otfs.T_sym), M=int(cfg.otfs.M), N=int(cfg.otfs.N),
        kT=float(cfg.channel.kT), bandwidth_hz=float(cfg.otfs.B),
        noise_figure_db=float(cfg.channel.NF),
        g_tx_dbi=float(cfg.otfs.g_tx_dBi), g_rx_dbi=float(cfg.otfs.g_rx_dBi),
        n_cpi=int(cfg.otfs.n_cpi), g_min=float(cfg.detection.g_min),
        use_swerling=bool(cfg.channel.use_swerling))


def _budget(data, row):
    rate = np.asarray(data["outgoing_rate"][row], dtype=np.int64)
    mask = np.asarray(data["outgoing_token_mask"][row], dtype=bool)
    active = (rate > 0) & np.any(mask, axis=1)
    frac = np.clip(np.asarray(data["comm_fraction"][row], dtype=np.float64), 0, 1)
    return 1.0 - np.where(active, frac, 0.0)


def best_structure(coeff, budget):
    """Greedy best owner + top-3 TX per target -> fixed-owner gain + owner."""
    K, Q = coeff.shape[0], coeff.shape[2]
    gain = np.zeros((K, Q))
    owner = np.zeros(Q, dtype=np.int64)
    for q in range(Q):
        best_val, best_j, best_tx = -1.0, -1, None
        for j in range(K):
            col = coeff[:, j, q] * budget
            top = np.argsort(col)[::-1][:3]
            val = float(col[top].sum())
            if val > best_val:
                best_val, best_j, best_tx = val, j, top
        owner[q] = best_j
        for i in best_tx:
            gain[i, q] = coeff[i, best_j, q]
    return gain, owner


def _run_l3(gain, owner, uav, tgt, budget, d_min, p_fa, horizon, step_m, area):
    def cost(g):
        o = _lh._solve(g, budget, p_fa)
        return float(o[0]) if o is not None else 1e6 + float(
            np.max(_lh._ceiling_deficit(g, budget, d_min)[0]))
    for _ in range(horizon):
        out = _lh._solve(gain, budget, p_fa)
        if out is not None and out[0] <= 1.0:
            break
        delta = np.zeros_like(uav)
        if out is None:
            for k in range(gain.shape[0]):
                g_k = _lh._deficit_gradient(k, owner, uav, tgt, gain, budget, d_min)
                n = np.linalg.norm(g_k)
                if n > 1e-12:
                    delta[k] = -step_m * g_k / n
        else:
            _, power, prices = out
            support = {q: {i: (power[i, q], gain[i, q]) for i in range(gain.shape[0])}
                       for q in range(gain.shape[1])}
            for k in range(gain.shape[0]):
                g_k = _lh.local_capability_gradient_k(
                    k, owner, prices, power[k], gain[k], uav[k], tgt, support)
                n = np.linalg.norm(g_k)
                if n > 1e-12:
                    delta[k] = -step_m * g_k / n
        new_uav = np.clip(uav + delta, 0.0, area)
        new_gain = _lh._friis_rescale(gain, owner, uav, tgt, new_uav)
        if cost(new_gain) >= cost(gain) - 1e-9:
            break
        gain, uav = new_gain, new_uav
    out = _lh._solve(gain, budget, p_fa)
    return out[0] if out is not None else float("inf")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--seed-limit", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--step-m", type=float, default=2.5)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    with np.load(args.trace, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    cfg = load_config(str(args.config))
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    d_min = float(minimum_deflection_for_detection_probability(
        np.asarray([XI[0]]), p_fa)[0])
    area = tuple(float(v) for v in cfg.scenario.region_size)
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    order = list(dict.fromkeys(int(s) for s in seeds.reshape(-1)))[:args.seed_limit]

    hard = []
    for seed in order:
        rows = np.flatnonzero(seeds == seed)
        rows = rows[np.argsort(frames[rows])]
        for row in rows:
            gain, owner, uav, tgt, budget = _lh._frame(data, int(row), cfg)
            if _lh._solve(gain, budget, p_fa) is None:
                hard.append((int(row), gain, owner, uav, tgt, budget,
                             _full_coeff(data, int(row), cfg)))
    rng = np.random.default_rng(0)
    if len(hard) > args.max_frames:
        hard = [hard[i] for i in rng.choice(len(hard), args.max_frames, replace=False)]

    l2_worst = l3_full = joint_full = joint_worst = 0
    for row, gain_t, owner_t, uav, tgt, budget, coeff in hard:
        # L2: best structure, exact max-min (worst floor)
        g_l2, o_l2 = best_structure(coeff, budget)
        res = solve_fixed_structure_maxmin_power_lp(g_l2, budget)
        l2_worst += int(res.worst_deflection >= d_min - 1e-9)
        # L3: fixed teacher structure, long horizon (full gauge)
        g3 = _lh._solve(gain_t, budget, p_fa)
        g_l3 = _run_l3(gain_t.copy(), owner_t, uav, tgt, budget, d_min, p_fa,
                       args.horizon, args.step_m, area)
        l3_full += int(g_l3 <= 1.0)
        # Joint: best structure THEN L3 descent (full gauge + worst floor)
        g_j = _run_l3(g_l2.copy(), o_l2, uav, tgt, budget, d_min, p_fa,
                      args.horizon, args.step_m, area)
        joint_full += int(g_j <= 1.0)
        res_j = solve_fixed_structure_maxmin_power_lp(g_l2, budget)
        joint_worst += int(res_j.worst_deflection >= d_min - 1e-9)

    n = len(hard)
    out = {
        "schema_version": 1,
        "hard_frames": n,
        "horizon": args.horizon, "step_m": args.step_m,
        "L2_structure_worstfloor_rate": l2_worst / max(1, n),
        "L3_geometry_fullgauge_rate": l3_full / max(1, n),
        "joint_fullgauge_rate": joint_full / max(1, n),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
