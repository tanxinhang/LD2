#!/usr/bin/env python
"""V3-T1 (advice 015): structure resource-competition diagnostic.

Given a structure-teacher trace (teacher_trace.npz with privileged alpha/g_dd/
chi_rep, teacher_pair, physical_pd, ...), reconstruct the fixed-owner max-min
LP per resolved frame and quantify the RESOURCE-COMPETITION signature that the
V3 hypothesis attributes to the 6/6 failures:

  - pi_q        : target dual prices lambda*_q (shadow value of target q's floor)
  - eta_i       : UAV scarcity price  eta_i = max_q lambda*_q * a_iq
                  (the exact Lagrange multiplier of UAV i's 1 W budget in the
                  dual:  min_lambda sum_i b_i max_q lambda_q a_iq  -- the per-UAV
                  term is the marginal value of one extra watt on UAV i)
  - tx_reuse    : for each TX i, the number of targets served with p_iq > eps
                  (1 W split count -- the seed-615 coupling signature)
  - owner_load  : for each receiver j, the number of owned targets
  - support     : per-target support size (number of contributing TXs)
  - comp_index  : share of frames where SOME TX serves >= 2 binding (lambda*>0)
                  targets with positive power (multi-target budget contention)

The tool aggregates per seed and compares against the recorded physical worst,
so failing vs passing seeds can be contrasted on the same statistics.

Usage: python tools/diagnose_structure_competition.py --trace PATH --config PATH
       [--out PATH] [--seed-limit N] [--frames all|steady]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from tools.audit_structure_trace_physical_bottleneck import (  # noqa: E402
    _ordered_unique,
    _recorded_power,
)
from uav_isac.coordination.maxmin_power import (  # noqa: E402
    fixed_owner_gain_matrix,
    optimal_maxmin_dual_prices,
    solve_fixed_structure_maxmin_power_lp,
)
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)

EPS = 1e-9


def _coefficient(data: dict, row: int, cfg: object) -> np.ndarray:
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


def frame_stats(
    data: dict, row: int, cfg: object, steady_from: int,
) -> dict | None:
    """Competition statistics for one trace row (resolved frames only)."""
    if not bool(np.asarray(data["p0_resolved"][row])):
        return None
    coeff = _coefficient(data, row, cfg)
    pair = np.asarray(data["teacher_pair"][row], dtype=bool)
    selected = tuple(
        tuple(int(v) for v in edge) for edge in np.argwhere(pair))
    try:
        gain, owners = fixed_owner_gain_matrix(coeff, selected)
    except ValueError:
        return None
    comm_power, _, _ = _recorded_power(data, row)
    budget = 1.0 - comm_power
    if np.any(budget <= 0.0):
        return None
    try:
        res = solve_fixed_structure_maxmin_power_lp(gain, budget)
    except ValueError:
        return None
    lam, _ = optimal_maxmin_dual_prices(gain, budget)
    lam = np.abs(np.asarray(lam, dtype=np.float64))
    p = np.asarray(res.power_w, dtype=np.float64)
    K, Q = p.shape
    eta = np.max(lam[None, :] * gain, axis=1)  # UAV scarcity price
    tx_reuse = np.sum(p > EPS, axis=1)          # targets served per TX
    owner_load = np.zeros(K, dtype=np.int64)
    for q in range(Q):
        if owners[q] >= 0:
            owner_load[owners[q]] += 1
    support = np.sum(p > EPS, axis=0)           # TXs per target
    binding_served = np.sum((p > EPS) & (lam[None, :] > EPS), axis=1)
    comp = bool(np.any(binding_served >= 2))
    return {
        "t_star": float(res.worst_deflection),
        "worst_pd": float(np.min(np.asarray(
            data["physical_pd"][row], dtype=np.float64))),
        "pi_max": float(np.max(lam)),
        "pi_mean": float(np.mean(lam)),
        "eta_mean": float(np.mean(eta)),
        "eta_max": float(np.max(eta)),
        "tx_reuse_max": int(np.max(tx_reuse)),
        "tx_reuse_mean": float(np.mean(tx_reuse)),
        "owner_load_max": int(np.max(owner_load)),
        "support_mean": float(np.mean(support)),
        "comp_index": int(comp),
        "steady_ok": bool(int(np.asarray(data["frame"][row])) >= steady_from),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed-limit", type=int, default=0)
    ap.add_argument("--steady-frames", type=int, default=20)
    args = ap.parse_args()

    with np.load(args.trace, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg = load_config(args.config)
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    T = int(np.max(frames)) + 1
    steady_from = T - max(1, int(args.steady_frames))
    order = _ordered_unique(seeds)
    if args.seed_limit > 0:
        order = order[: args.seed_limit]

    per_seed: dict[str, dict] = {}
    for seed in order:
        rows = np.flatnonzero(seeds == seed)
        stats = [s for s in (frame_stats(data, r, cfg, steady_from)
                             for r in rows) if s is not None]
        if not stats:
            continue
        comp_all = sum(s["comp_index"] for s in stats) / len(stats)
        comp_steady = sum(s["comp_index"] for s in stats
                          if s["steady_ok"]) / max(
                              1, sum(s["steady_ok"] for s in stats))
        per_seed[str(seed)] = {
            "frames": len(stats),
            "steady_frames": sum(s["steady_ok"] for s in stats),
            "mean_worst_pd": float(np.mean([s["worst_pd"] for s in stats])),
            "steady_mean_worst_pd": float(np.mean([
                s["worst_pd"] for s in stats if s["steady_ok"]])),
            "mean_pi_max": float(np.mean([s["pi_max"] for s in stats])),
            "mean_pi_mean": float(np.mean([s["pi_mean"] for s in stats])),
            "mean_eta_max": float(np.mean([s["eta_max"] for s in stats])),
            "mean_eta_mean": float(np.mean([s["eta_mean"] for s in stats])),
            "mean_tx_reuse_max": float(np.mean(
                [s["tx_reuse_max"] for s in stats])),
            "mean_tx_reuse": float(np.mean([s["tx_reuse_mean"] for s in stats])),
            "mean_owner_load_max": float(np.mean(
                [s["owner_load_max"] for s in stats])),
            "mean_support": float(np.mean([s["support_mean"] for s in stats])),
            "comp_index_all": float(comp_all),
            "comp_index_steady": float(comp_steady),
            "mean_t_star": float(np.mean([s["t_star"] for s in stats])),
        }

    result = {"seed_count": len(per_seed), "per_seed": per_seed}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
    # compact print
    print(f"{'seed':>6} {'worst':>6} {'pi_max':>6} {'eta_max':>6} "
          f"{'txReuse':>7} {'ownLoad':>7} {'compAll':>6} {'compSteady':>8}")
    for seed, s in per_seed.items():
        print(f"{seed:>6} {s['steady_mean_worst_pd']:>6.3f} "
              f"{s['mean_pi_max']:>6.3f} {s['mean_eta_max']:>6.3f} "
              f"{s['mean_tx_reuse_max']:>7.1f} {s['mean_owner_load_max']:>7.1f} "
              f"{s['comp_index_all']:>6.2f} {s['comp_index_steady']:>8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
