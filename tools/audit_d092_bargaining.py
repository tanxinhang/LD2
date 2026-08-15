#!/usr/bin/env python
"""D0.92-A smoke: max-min LP vs bargaining LP on the 8/8 trace.

The key question: does the reference-normalized bargaining objective stop the
steady collapse that pure max-min causes (D0.89-A showed steady 0.773 -> 0.692)?
Same fixed-owner structure, same budget, only the power objective changes.
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
from uav_isac.coordination.maxmin_power import (  # noqa: E402
    fixed_owner_gain_matrix,
    solve_fixed_structure_maxmin_power_lp,
)
from uav_isac.coordination.bargaining_power import (  # noqa: E402
    solve_fixed_structure_bargaining_lp,
)
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)
from uav_isac.physical.detection import compute_detection_probabilities  # noqa: E402

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


def run(trace_path, config_path):
    with np.load(trace_path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    cfg = load_config(str(config_path))
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    order = list(dict.fromkeys(int(s) for s in seeds.reshape(-1)))

    mm_worst, mm_steady, mm_eta = [], [], []
    bg_worst, bg_steady, bg_eta = [], [], []
    for seed in order:
        rows = np.flatnonzero(seeds == seed)
        rows = rows[np.argsort(frames[rows])]
        mm_pd, bg_pd = [], []
        for row in rows:
            gain, budget = _gain_and_budget(data, int(row), cfg)
            try:
                mm = solve_fixed_structure_maxmin_power_lp(gain, budget)
                bg = solve_fixed_structure_bargaining_lp(gain, budget)
            except RuntimeError:
                continue
            mm_pd.append(compute_detection_probabilities(mm.deflection, p_fa))
            bg_pd.append(compute_detection_probabilities(bg.deflection, p_fa))
            mm_eta.append(mm.worst_deflection)
            bg_eta.append(bg.bargaining_value)
        for pd, (wl, sl) in ((mm_pd, (mm_worst, mm_steady)), (bg_pd, (bg_worst, bg_steady))):
            w = np.asarray(pd, dtype=np.float64)[-STEADY_WINDOW:]
            per = np.mean(w, axis=0)
            o = np.sort(per)
            wl.append(float(o[0])); sl.append(float(np.mean(o)))

    def stat(v):
        v = np.asarray(v, dtype=np.float64)
        return float(np.mean(v))

    return {
        "episodes": len(order),
        "maxmin_worst_mean": stat(mm_worst),
        "maxmin_steady_mean": stat(mm_steady),
        "bargaining_worst_mean": stat(bg_worst),
        "bargaining_steady_mean": stat(bg_steady),
        "bargaining_eta_mean": stat(bg_eta),
        "steady_delta": stat(bg_steady) - stat(mm_steady),
        "worst_delta": stat(bg_worst) - stat(mm_worst),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    result = run(args.trace, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
