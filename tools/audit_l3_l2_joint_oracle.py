#!/usr/bin/env python
"""Honest joint L2+L3 payoff with the single-role (half-duplex) structure oracle.

Corrects the relaxed greedy best-structure estimate (which let one UAV be both
TX and RX).  The proper L2 block is ``solve_joint_pair_power_oracle``
(``full_duplex=False``): it enumerates role partitions and, per partition,
solves the MILP for reporting pairs + power, so a UAV is EITHER transmitter OR
receiver in a frame.  After the oracle picks the structure we run the SAME
long-horizon L3 geometry descent, then measure the capability-gauge gamma.

Reports: L2-alone (oracle full task), L3-alone (teacher structure), and the
honest joint (oracle structure + L3 descent).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables)
from uav_isac.physical.detection import (  # noqa: E402
    minimum_deflection_for_detection_probability)
from uav_isac.physical.feasibility_oracle import solve_joint_pair_power_oracle  # noqa: E402

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


def _owner_gain_from_selected(coeff, selected, K, Q):
    owner = np.full(Q, -1, dtype=np.int64)
    gain = np.zeros((K, Q))
    for i, j, q in selected:
        owner[q] = j
        gain[i, q] += coeff[i, j, q]
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
    ap.add_argument("--max-frames", type=int, default=10)
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
    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])

    hard = []
    for seed in order:
        rows = np.flatnonzero(seeds == seed)
        rows = rows[np.argsort(frames[rows])]
        for row in rows:
            gain_t, owner_t, uav, tgt, budget = _lh._frame(data, int(row), cfg)
            if _lh._solve(gain_t, budget, p_fa) is None:
                hard.append((int(row), gain_t, owner_t, uav, tgt, budget,
                             _full_coeff(data, int(row), cfg)))
    rng = np.random.default_rng(0)
    if len(hard) > args.max_frames:
        hard = [hard[i] for i in rng.choice(len(hard), args.max_frames, replace=False)]

    l2_full = l3_full = joint_full = 0
    records = []
    for row, gain_t, owner_t, uav, tgt, budget, coeff in hard:
        comm_power = 1.0 - float(np.mean(budget))
        try:
            sol = solve_joint_pair_power_oracle(
                coeff, P_FA=p_fa, total_power_w=1.0,
                communication_reserve_w=comm_power,
                target_pair_limit=int(cfg.detection.K_q_max),
                reports_per_receiver=Q * int(cfg.detection.K_q_max),
                full_duplex=False, alternating_iterations=3, random_starts=1,
                seed=row, fusion_mode="local_only")
        except Exception:
            continue
        l2_feasible = (sol.worst >= 0.60 and sol.weak3 >= 0.70 and sol.steady >= 0.80)
        l2_full += int(l2_feasible)
        g_l2, o_l2 = _owner_gain_from_selected(coeff, sol.selected_set, K, Q)
        g_l3 = _run_l3(gain_t.copy(), owner_t, uav, tgt, budget, d_min, p_fa,
                       args.horizon, args.step_m, area)
        l3_full += int(g_l3 <= 1.0)
        g_j = _run_l3(g_l2.copy(), o_l2, uav, tgt, budget, d_min, p_fa,
                      args.horizon, args.step_m, area)
        joint_full += int(g_j <= 1.0)
        records.append({"row": int(row), "l2_feasible": bool(l2_feasible),
                        "l3_gamma": float(g_l3), "joint_gamma": float(g_j)})

    n = len(records)
    out = {
        "schema_version": 1,
        "hard_frames": n,
        "horizon": args.horizon, "step_m": args.step_m,
        "L2_oracle_fulltask_rate": l2_full / max(1, n),
        "L3_geometry_fullgauge_rate": l3_full / max(1, n),
        "joint_fullgauge_rate": joint_full / max(1, n),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
