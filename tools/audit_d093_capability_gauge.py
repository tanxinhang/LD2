#!/usr/bin/env python
"""D0.93-R v2: full task-set capability gauge gamma_P*.

Per fixed-structure frame, compute the *capability gauge*

    gamma_P* = min_{p, gamma}  gamma
               s.t.  D_q = sum_i a_iq p_iq,
                     W(D) >= rho_min,  T_k(D) >= rho_tail,  A(D) >= rho_avg,
                     sum_q p_iq <= gamma * b_i,  p >= 0.

gamma_P* <= 1  <=>  the power layer alone satisfies the FULL task set
(worst + bottom-k + average); gamma_P* - 1 is the fractional extra sensing
budget the fixed structure would need.  This is a convex program: P_D(D) is
concave in D over the high-D region, so W/T_k/A are all concave superlevel
constraints (the bottom-k superlevel is an intersection of convex superlevel
sets).  We also re-run the D0.93-R sanity check (S*(w2) <= S*(w1) on common
feasible frames) that the previous audit omitted.
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


def _pd_from_deflection(d, p_fa):
    return compute_detection_probabilities(d, p_fa)


def _bottom_k_sum(pd, k):
    return float(np.sum(np.sort(pd)[:k]))


def _capability_gauge(gain, budget, p_fa, xi):
    """min gamma s.t. full task set, budget <= gamma*b.

    Bottom-k uses the O(Q) order-statistic representation (advice/004.md §7):
        T_k >= rho_tail  <=>  exists tau, z_q >= 0:
            z_q >= tau - y_q,  k*tau - sum_q z_q >= k*rho_tail,
    so the nonsmooth "sum of k smallest" becomes linear in (tau, z).  The
    solution is *verified* against the true P_D constraints before returning.
    """
    rho_min, rho_tail, rho_avg, k = xi
    K, Q = gain.shape
    d_min = float(minimum_deflection_for_detection_probability(
        np.asarray([rho_min]), p_fa)[0])
    ceiling = np.sum(gain * budget[:, None], axis=0)
    if np.any(ceiling < d_min - 1e-9):
        return None  # worst floor unreachable for some target

    # Variables: p (K*Q), gamma (1), tau (1), z (Q).
    n_p = K * Q

    def unpack(x):
        p = x[:n_p].reshape(K, Q)
        gamma = float(x[n_p])
        tau = float(x[n_p + 1])
        z = x[n_p + 2:]
        return p, gamma, tau, z

    def pd_of(x):
        p = x[:n_p].reshape(K, Q)
        d = np.sum(gain * p, axis=0)
        return _pd_from_deflection(d, p_fa)

    def objective(x):
        return float(x[n_p])

    def worst_cons(x):
        return pd_of(x) - rho_min

    def avg_cons(x):
        return float(np.mean(pd_of(x))) - rho_avg

    def z_cons(x):
        p, _, tau, z = unpack(x)
        return z - (tau - pd_of(x))

    def tail_cons(x):
        _, _, tau, z = unpack(x)
        return k * tau - float(np.sum(z)) - k * rho_tail

    def budget_cons(x):
        p, gamma, _, _ = unpack(x)
        return gamma * budget - np.sum(p, axis=1)

    p0 = np.repeat(budget, Q) / max(Q, 1)
    x0 = np.concatenate([p0, [1.0], [rho_tail], np.zeros(Q)])
    cons = [
        {"type": "ineq", "fun": worst_cons},
        {"type": "ineq", "fun": avg_cons},
        {"type": "ineq", "fun": z_cons},
        {"type": "ineq", "fun": tail_cons},
        {"type": "ineq", "fun": budget_cons},
    ]
    res = minimize(
        objective, x0, method="SLSQP", constraints=cons,
        bounds=[(0.0, None)] * (n_p + 2 + Q),
        options={"maxiter": 4000, "ftol": 1e-11},
    )
    p, gamma, _, _ = unpack(res.x)
    d = np.sum(gain * p, axis=0)
    pd = _pd_from_deflection(d, p_fa)
    # Verify the true constraints (not just the SLSQP success flag).
    viol = {
        "worst": float(np.min(pd) - rho_min),
        "bottom_k": _bottom_k_sum(pd, k) - k * rho_tail,
        "avg": float(np.mean(pd)) - rho_avg,
        "budget": float(np.max(np.sum(p, axis=1) - gamma * budget)),
    }
    feasible = all(v >= -1e-6 for v in viol.values()) and gamma <= 1.0 + 1e-6
    return float(gamma), d, pd, viol, feasible


def run(trace_path, config_path, *, xi, seed_limit):
    with np.load(trace_path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    cfg = load_config(str(config_path))
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    order = list(dict.fromkeys(int(s) for s in seeds.reshape(-1)))[:seed_limit]

    gauges = []
    feasible = 0
    total = 0
    violations = {"worst": [], "bottom_k": [], "avg": [], "budget": []}
    for seed in order:
        rows = np.flatnonzero(seeds == seed)
        rows = rows[np.argsort(frames[rows])]
        for row in rows:
            gain, budget = _gain_and_budget(data, int(row), cfg)
            if np.all(np.sum(gain, axis=0) <= 0.0):
                continue
            total += 1
            out = _capability_gauge(gain, budget, p_fa, xi)
            if out is None:
                continue
            gamma, d, pd, viol, is_feasible = out
            gauges.append(gamma)
            if is_feasible:
                feasible += 1
            for key in violations:
                violations[key].append(viol[key])

    g = np.asarray(gauges, dtype=np.float64)
    return {
        "schema_version": 1,
        "trace": str(trace_path),
        "config": str(config_path),
        "task_xi": {
            "rho_min": float(xi[0]), "rho_tail": float(xi[1]),
            "rho_avg": float(xi[2]), "k": int(xi[3]),
        },
        "frames_processed": total,
        "frames_gauge_solved": int(g.size),
        "frames_feasible": feasible,
        "feasible_rate": feasible / max(1, total),
        "capability_gauge_mean": float(np.mean(g)) if g.size else float("nan"),
        "capability_gauge_p50": float(np.median(g)) if g.size else float("nan"),
        "capability_gauge_p90": float(np.quantile(g, 0.9)) if g.size else float("nan"),
        "capability_gauge_max": float(np.max(g)) if g.size else float("nan"),
        "capability_deficit_mean": float(np.mean(np.maximum(g - 1.0, 0.0))) if g.size else float("nan"),
        "max_constraint_violation": {
            key: float(np.max(np.abs(v))) if v else 0.0
            for key, v in violations.items()
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--seed-limit", type=int, default=3)
    ap.add_argument("--rho-min", type=float, default=0.60)
    ap.add_argument("--rho-tail", type=float, default=0.70)
    ap.add_argument("--rho-avg", type=float, default=0.80)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    xi = (args.rho_min, args.rho_tail, args.rho_avg, args.k)
    result = run(args.trace, args.config, xi=xi,
                 seed_limit=max(1, args.seed_limit))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
