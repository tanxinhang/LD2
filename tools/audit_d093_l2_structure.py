#!/usr/bin/env python
"""D0.93 L2: structure capability expansion audit.

For the "certified escalate" frames (power-only capability gauge gamma_P* > 1),
ask whether a *structure* change (optimal single-duplex role/owner/edge, with
optimal power) can make the FULL task feasible.  Reuse the joint pair+power
oracle ``solve_joint_pair_power_oracle(full_duplex=False)``:

    structure-recoverable : oracle (worst, weak3, steady) satisfies the task
    geometry-limited      : even the optimal structure cannot satisfy it

This splits the 77% structure bottleneck into "L2 can fix it" vs "needs L3
geometry".  The oracle enumerates all role partitions, so it is sampled over a
bounded number of escalate frames.
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
from uav_isac.physical.feasibility_oracle import solve_joint_pair_power_oracle  # noqa: E402


def _coefficient_and_fixed_gain(data, row, cfg):
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
    return coefficient, gain


def run(trace_path, config_path, *, xi, max_frames, seed_limit):
    with np.load(trace_path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    cfg = load_config(str(config_path))
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    order = list(dict.fromkeys(int(s) for s in seeds.reshape(-1)))[:seed_limit]

    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])
    escalate_rows = []
    for seed in order:
        rows = np.flatnonzero(seeds == seed)
        rows = rows[np.argsort(frames[rows])]
        for row in rows:
            coefficient, gain = _coefficient_and_fixed_gain(data, int(row), cfg)
            rate = np.asarray(data["outgoing_rate"][row], dtype=np.int64)
            mask = np.asarray(data["outgoing_token_mask"][row], dtype=bool)
            active = (rate > 0) & np.any(mask, axis=1)
            frac = np.clip(np.asarray(data["comm_fraction"][row], dtype=np.float64), 0, 1)
            budget = 1.0 - np.where(active, frac, 0.0)
            out = capability_gauge(gain, budget, p_fa, xi)
            # Escalate: unsolvable (None) or gamma* > 1.
            if out is None or out[0] > 1.0:
                escalate_rows.append((int(row), coefficient, budget))

    # Sample a bounded number of escalate frames.
    rng = np.random.default_rng(0)
    if len(escalate_rows) > max_frames:
        escalate_rows = [escalate_rows[i] for i in
                         rng.choice(len(escalate_rows), max_frames, replace=False)]

    results = []
    for row, coefficient, budget in escalate_rows:
        comm_power = 1.0 - float(np.mean(budget))
        try:
            single = solve_joint_pair_power_oracle(
                coefficient,
                P_FA=p_fa,
                total_power_w=1.0,
                communication_reserve_w=comm_power,
                target_pair_limit=int(cfg.detection.K_q_max),
                reports_per_receiver=Q * int(cfg.detection.K_q_max),
                full_duplex=False,
                alternating_iterations=4,
                random_starts=1,
                seed=row,
                fusion_mode="local_only",
            )
        except Exception:
            continue
        feasible = (single.worst >= 0.60 and single.weak3 >= 0.70
                    and single.steady >= 0.80)
        results.append({
            "row": int(row),
            "single_worst": float(single.worst),
            "single_weak3": float(single.weak3),
            "single_steady": float(single.steady),
            "structure_feasible": bool(feasible),
        })

    n = len(results)
    n_recoverable = sum(1 for r in results if r["structure_feasible"])
    return {
        "schema_version": 1,
        "trace": str(trace_path),
        "config": str(config_path),
        "escalate_frames_total": len(escalate_rows),
        "sampled_frames": n,
        "structure_recoverable": n_recoverable,
        "geometry_limited": n - n_recoverable,
        "structure_recoverable_rate": n_recoverable / max(1, n),
        "single_worst_mean": float(np.mean([r["single_worst"] for r in results])) if n else float("nan"),
        "single_steady_mean": float(np.mean([r["single_steady"] for r in results])) if n else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--seed-limit", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=15)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    xi = (0.60, 0.70, 0.80, 3)
    result = run(args.trace, args.config, xi=xi,
                 max_frames=max(1, args.max_frames),
                 seed_limit=max(1, args.seed_limit))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
