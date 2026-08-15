#!/usr/bin/env python
"""Scale-aware steady/worst ceiling audit.

For a frozen trace, compute the relaxed same-geometry per-target ceiling

    D_q^relax = sum_i b_i * max_j a_ijq,

which relaxes cross-target power coupling, common receiver owner, role and
capacity, so it is a componentwise upper bound on EVERY feasible same-geometry
allocation.  Report both the worst and the steady (mean) ceiling, and decide
whether the QoS floors (worst >= 0.60 AND steady >= 0.80) are even physically
reachable on this scale.  If the relaxed steady ceiling is below 0.80, no
coordination algorithm can reach it and the requirement must be reframed.
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
    relaxed_same_geometry_target_ceiling,
)
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)
from uav_isac.physical.detection import compute_detection_probabilities  # noqa: E402

STEADY_WINDOW = 20


def audit(trace_path: Path, config_path: Path) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg = load_config(str(config_path))
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    seed_order = list(dict.fromkeys(int(s) for s in seeds.reshape(-1)))

    ceiling_worst = []
    ceiling_steady = []
    for seed in seed_order:
        rows = np.flatnonzero(seeds == seed)
        rows = rows[np.argsort(frames[rows])]
        pd_frames = []
        for row in rows:
            coefficient = per_watt_deflection_tensor_from_observables(
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
            rate = np.asarray(data["outgoing_rate"][row], dtype=np.int64)
            token_mask = np.asarray(
                data["outgoing_token_mask"][row], dtype=bool)
            active = (rate > 0) & np.any(token_mask, axis=1)
            fraction = np.clip(np.asarray(
                data["comm_fraction"][row], dtype=np.float64), 0.0, 1.0)
            budget = 1.0 - np.where(active, fraction, 0.0)
            ceiling = relaxed_same_geometry_target_ceiling(
                coefficient, budget)
            pd_frames.append(compute_detection_probabilities(ceiling, p_fa))
        window = np.asarray(pd_frames, dtype=np.float64)[-STEADY_WINDOW:]
        per_target = np.mean(window, axis=0)
        ordered = np.sort(per_target)
        ceiling_steady.append(float(np.mean(ordered)))
        ceiling_worst.append(float(ordered[0]))

    steady = np.asarray(ceiling_steady, dtype=np.float64)
    worst = np.asarray(ceiling_worst, dtype=np.float64)
    feasible = (steady >= 0.80) & (worst >= 0.60)
    return {
        "schema_version": 1,
        "trace": str(trace_path),
        "config": str(config_path),
        "episodes": int(steady.size),
        "ceiling_steady_mean": float(np.mean(steady)),
        "ceiling_steady_min": float(np.min(steady)),
        "ceiling_worst_mean": float(np.mean(worst)),
        "ceiling_worst_min": float(np.min(worst)),
        "ceiling_qos_feasible_rate": float(np.mean(feasible)),
        "steady_0p80_reachable": bool(
            np.mean(steady) >= 0.80),
        "verdict": (
            "steady_floor_unreachable"
            if float(np.mean(steady)) < 0.80
            else ("steady_floor_reachable_but_worst_binding"
                  if float(np.mean(worst)) < 0.60
                  else "both_floors_reachable")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.trace, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
