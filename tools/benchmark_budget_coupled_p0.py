"""Replay traced scale frames through the budget-coupled local P0 solver."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uav_isac.coordination.maxmin_power import (
    solve_fixed_structure_maxmin_power_lp,
)
from uav_isac.physical.feasibility_oracle import (
    solve_budget_coupled_local_only_pairs,
    solve_enumerated_role_ceiling_local_pairs,
)
from uav_isac.utils.math_utils import compute_PD


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--secondary", action="store_true")
    parser.add_argument("--time-limit-s", type=float, default=5.0)
    parser.add_argument(
        "--solver", choices=("milp", "enumerated_role_ceiling"),
        default="milp")
    args = parser.parse_args()

    trace = np.load(args.trace, allow_pickle=False)
    indices = np.arange(0, len(trace["frame"]), max(1, args.stride))[
        :max(1, args.max_frames)
    ]
    records: list[dict[str, float | int]] = []
    for frame in indices:
        coefficient = np.asarray(
            trace["per_watt_coefficient"][frame], dtype=np.float64)
        budget = np.asarray(
            trace["frame_sensing_budget_w"][frame], dtype=np.float64)
        priority = np.exp(np.clip(
            float(trace["deficit_priority_gain"][0])
            * (float(trace["qos_floor"][0])
               - np.asarray(trace["coord_pd_ema"][frame], dtype=np.float64)),
            -6.0,
            6.0,
        ))
        started = time.perf_counter()
        common = dict(
            P_FA=float(trace["p_fa"][0]),
            p_d_floor=float(trace["qos_floor"][0]),
            target_pair_limit=int(trace["target_pair_limit"][0]),
            reports_per_receiver=int(trace["reports_per_receiver"][0]),
            target_priority=priority,
        )
        solution = (
            solve_enumerated_role_ceiling_local_pairs(
                coefficient, budget, **common)
            if args.solver == "enumerated_role_ceiling"
            else solve_budget_coupled_local_only_pairs(
                coefficient,
                budget,
                time_limit_s=float(args.time_limit_s),
                secondary_objective_enabled=bool(args.secondary),
                **common,
            )
        )
        elapsed = time.perf_counter() - started
        fixed_gain = np.zeros(
            (coefficient.shape[0], coefficient.shape[2]), dtype=np.float64)
        for i, j, q in solution.selected_set:
            fixed_gain[i, q] = coefficient[i, j, q]
        downstream = solve_fixed_structure_maxmin_power_lp(fixed_gain, budget)
        downstream_pd = compute_PD(
            downstream.deflection, float(trace["p_fa"][0]))
        records.append({
            "frame": int(trace["frame"][frame]),
            "solve_time_s": float(elapsed),
            "tx_count": len(solution.tx_indices),
            "rx_count": len(solution.rx_indices),
            "edge_count": len(solution.selected_set),
            "proxy_worst_pd": float(np.min(solution.P_D_q)),
            "downstream_worst_pd": float(np.min(downstream_pd)),
        })

    def mean(key: str) -> float:
        return float(np.mean([float(item[key]) for item in records]))

    print(json.dumps({
        "trace": str(args.trace),
        "secondary": bool(args.secondary),
        "solver": args.solver,
        "frames": len(records),
        "mean_solve_time_s": mean("solve_time_s"),
        "mean_tx_count": mean("tx_count"),
        "mean_rx_count": mean("rx_count"),
        "mean_edge_count": mean("edge_count"),
        "min_proxy_worst_pd": float(min(
            item["proxy_worst_pd"] for item in records)),
        "min_downstream_worst_pd": float(min(
            item["downstream_worst_pd"] for item in records)),
        "records": records,
    }, indent=2))


if __name__ == "__main__":
    main()
