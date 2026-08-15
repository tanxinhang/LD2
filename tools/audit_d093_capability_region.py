#!/usr/bin/env python
"""D0.93-R Phase 1: fixed-structure (worst, steady) capability region.

Answers the decisive question the D0.92 result raised: for the fixed structure
and geometry of the 8/8 trace, does the power layer alone contain an allocation
with BOTH worst >= 0.60 and steady >= 0.80?  If yes, the steady-vs-worst
"tension" is a scalarization artifact (max-min and bargaining just pick
different Pareto points), and a *constrained* allocation fixes it.  If no, the
power layer is exhausted and the structure/geometry layer is required.

For each frame we sweep the worst-target floor w and solve

    max_p  mean_q P_D( sum_i a_iq p_iq )
    s.t.   sum_i a_iq p_iq >= d_w   for every q (worst >= w),
           sum_q p_iq = b_i,  p >= 0.

For w >= 0.60 the per-target Deflection floor d_w is large enough that
P_D(D) = Q(Q^{-1}(P_FA)-sqrt(D)) is concave in D, so this is a concave
maximisation over a polytope (convex program): SLSQP finds the global optimum.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from uav_isac.coordination.maxmin_power import fixed_owner_gain_matrix  # noqa: E402
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)
from uav_isac.physical.detection import (  # noqa: E402
    compute_detection_probabilities,
    minimum_deflection_for_detection_probability,
)

STEADY_WINDOW = 20


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


def _max_steady_subject_to_worst(gain, budget, p_fa, w):
    K, Q = gain.shape
    d_w = float(minimum_deflection_for_detection_probability(
        np.asarray([w]), p_fa)[0])
    # Feasibility: every target's same-feasible-set ceiling must reach d_w.
    ceiling = np.sum(gain * budget[:, None], axis=0)
    if np.any(ceiling < d_w - 1e-9):
        return None  # some target cannot reach w under this structure

    def steady_neg(p):
        p = p.reshape(K, Q)
        d = np.sum(gain * p, axis=0)
        pd = compute_detection_probabilities(d, p_fa)
        return -float(np.mean(pd))

    # Equality: sum_q p_iq = b_i.
    def eq(p):
        p = p.reshape(K, Q)
        return np.sum(p, axis=1) - budget

    # Inequality: sum_i a_iq p_iq - d_w >= 0  ->  d_w - sum_i a_iq p_iq <= 0.
    def ineq(p):
        p = p.reshape(K, Q)
        d = np.sum(gain * p, axis=0)
        return d - d_w

    p0 = np.repeat(budget, Q) / max(Q, 1)
    cons = [
        {"type": "eq", "fun": eq},
        {"type": "ineq", "fun": ineq},
    ]
    res = minimize(
        steady_neg, p0, method="SLSQP", constraints=cons,
        bounds=[(0.0, None)] * (K * Q),
        options={"maxiter": 2000, "ftol": 1e-10},
    )
    p = res.x.reshape(K, Q)
    d = np.sum(gain * p, axis=0)
    pd = compute_detection_probabilities(d, p_fa)
    return float(np.min(pd)), float(np.mean(pd))


def run(trace_path, config_path, *, worst_grid, seed_limit):
    with np.load(trace_path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    cfg = load_config(str(config_path))
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    order = list(dict.fromkeys(int(s) for s in seeds.reshape(-1)))[:seed_limit]

    frontier = {f"{w:.2f}": [] for w in worst_grid}
    for seed in order:
        rows = np.flatnonzero(seeds == seed)
        rows = rows[np.argsort(frames[rows])]
        # Sample final-resolve frames only (structure held stable).
        for row in rows:
            gain, budget = _gain_and_budget(data, int(row), cfg)
            if np.all(np.sum(gain, axis=0) <= 0.0):
                continue  # fully degenerate: no reachable target
            for w in worst_grid:
                try:
                    out = _max_steady_subject_to_worst(gain, budget, p_fa, w)
                except Exception:
                    out = None
                if out is None:
                    continue
                achieved_w, steady = out
                frontier[f"{w:.2f}"].append(steady)

    summary = {
        str(w): {
            "mean_steady": float(np.mean(v)) if v else float("nan"),
            "count": len(v),
        }
        for w, v in frontier.items()
    }
    summary["key_check"] = {
        "worst_floor": 0.60,
        "steady_at_w0.60_mean": summary["0.60"]["mean_steady"],
        "power_layer_satisfies_both": bool(
            summary.get("0.60", {}).get("mean_steady", 0.0) >= 0.80),
    }
    return {
        "schema_version": 1,
        "trace": str(trace_path),
        "config": str(config_path),
        "seed_limit": int(seed_limit),
        "frontier": summary,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--seed-limit", type=int, default=3)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    grid = [round(0.30 + 0.05 * i, 2) for i in range(11)]  # 0.30..0.80
    result = run(args.trace, args.config, worst_grid=grid,
                 seed_limit=max(1, args.seed_limit))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["frontier"], indent=2))


if __name__ == "__main__":
    main()
