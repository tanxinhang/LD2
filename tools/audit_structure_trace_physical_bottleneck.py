#!/usr/bin/env python
"""Decompose same-geometry structure, power and duplex bottlenecks.

The audit uses the final resolved frame of each selected development episode.
It reconstructs the exact 1 W communication/sensing allocation recorded by a
trace without a structural side packet, then compares pair-only, power-only,
joint single-role and joint full-duplex optimized physical benchmarks.
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
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    evaluate_physical_feasibility_oracles,
    per_watt_deflection_tensor,
    per_watt_deflection_tensor_from_observables,
)
from uav_isac.coordination.maxmin_power import (  # noqa: E402
    distributed_column_generation_maxmin_power,
    fixed_owner_gain_matrix,
    relaxed_same_geometry_target_ceiling,
    solve_fixed_structure_maxmin_power_lp,
)
from uav_isac.coordination.bottleneck_router import (  # noqa: E402
    route_isac_repair,
)
from uav_isac.physical.detection import (  # noqa: E402
    compute_detection_probabilities,
)
from uav_isac.utils.types import DeflectionEntry  # noqa: E402


def _ordered_unique(values: np.ndarray) -> list[int]:
    return list(dict.fromkeys(int(value) for value in values.reshape(-1)))


def _entries(data: dict[str, np.ndarray], row: int) -> list[DeflectionEntry]:
    candidate = np.asarray(data["privileged_candidate"][row], dtype=bool)
    d_eff = np.asarray(data["privileged_d_eff"][row], dtype=np.float64)
    d_raw = np.asarray(data["privileged_d_raw"][row], dtype=np.float64)
    alpha = np.asarray(data["privileged_alpha"][row], dtype=np.float64)
    g_dd = np.asarray(data["privileged_g_dd"][row], dtype=np.float64)
    chi_rep = np.asarray(data["privileged_chi_rep"][row], dtype=np.float64)
    K, _, Q = d_eff.shape
    return [
        DeflectionEntry(
            i=i,
            j=j,
            q=q,
            tau=0.0,
            nu=0.0,
            alpha=float(alpha[i, j, q]),
            d_raw=float(d_raw[i, j, q]),
            g_dd=float(g_dd[i, j, q]),
            chi_rep=float(chi_rep[i, j, q]),
            d_eff=float(d_eff[i, j, q]),
        )
        for i in range(K)
        for j in range(K)
        for q in range(Q)
        if i != j and candidate[i, j, q] and d_eff[i, j, q] > 0.0
    ]


def _recorded_power(
    data: dict[str, np.ndarray], row: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    weights = np.asarray(data["sensing_weights"][row], dtype=np.float64)
    if np.any(weights < -1.0e-8):
        raise ValueError("recorded sensing weights must be non-negative")
    normalizer = np.sum(weights, axis=1, keepdims=True)
    if np.any(normalizer <= 0.0):
        raise ValueError("recorded sensing weights must have positive mass")
    weights = weights / normalizer
    rate = np.asarray(data["outgoing_rate"][row], dtype=np.int64)
    token_mask = np.asarray(
        data["outgoing_token_mask"][row], dtype=bool)
    active = (rate > 0) & np.any(token_mask, axis=1)
    fraction = np.clip(np.asarray(
        data["comm_fraction"][row], dtype=np.float64), 0.0, 1.0)
    comm_power = np.where(active, fraction, 0.0)
    sensing_power = (1.0 - comm_power[:, None]) * weights
    error = float(np.max(np.abs(
        comm_power + np.sum(sensing_power, axis=1) - 1.0)))
    return comm_power, sensing_power, error


def audit(
    trace_path: Path,
    seed_limit: int,
    *,
    dual_rounds: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    price_bits: int = 6,
    feedback_bits: int = 16,
    config_path: Path | None = None,
) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    resolved = np.asarray(data["p0_resolved"], dtype=bool)
    seed_order = _ordered_unique(seeds)[:max(1, int(seed_limit))]
    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    pair_limit = int(np.asarray(data["target_pair_limit"]).reshape(-1)[0])
    reports = int(np.asarray(data["reports_per_receiver"]).reshape(-1)[0])
    cfg = load_config(str(config_path)) if config_path is not None else None
    rows = []
    for seed in seed_order:
        candidates = np.flatnonzero(resolved & (seeds == seed))
        if not len(candidates):
            raise ValueError(f"seed {seed} has no resolved frame")
        row = int(candidates[np.argmax(frames[candidates])])
        comm_power, sensing_power, power_error = _recorded_power(data, row)
        d_eff = np.asarray(data["privileged_d_eff"][row], dtype=np.float64)
        pair = np.asarray(data["teacher_pair"][row], dtype=bool)
        receiver_d = np.sum(d_eff * pair, axis=0)
        deployed_pd = compute_detection_probabilities(
            np.max(receiver_d, axis=0), p_fa)
        selected = tuple(
            tuple(int(value) for value in edge)
            for edge in np.argwhere(pair)
        )
        coefficient = (
            per_watt_deflection_tensor(
                _entries(data, row), sensing_power, K, Q)
            if cfg is None
            else per_watt_deflection_tensor_from_observables(
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
        )
        fixed_gain, owners = fixed_owner_gain_matrix(coefficient, selected)
        sensing_budget = 1.0 - comm_power
        exact = solve_fixed_structure_maxmin_power_lp(
            fixed_gain, sensing_budget)
        exact_pd = compute_detection_probabilities(exact.deflection, p_fa)
        relaxed_deflection = relaxed_same_geometry_target_ceiling(
            coefficient, sensing_budget)
        relaxed_pd = compute_detection_probabilities(relaxed_deflection, p_fa)
        routing = route_isac_repair(
            float(np.min(deployed_pd)),
            fixed_power_upper=float(np.min(exact_pd)),
            joint_structure_upper=float(np.min(relaxed_pd)),
            qos_floor=0.60,
        )
        distributed = {}
        for count in dual_rounds:
            finite = distributed_column_generation_maxmin_power(
                fixed_gain,
                sensing_budget,
                rounds=int(count),
                price_bits=int(price_bits),
                feedback_bits=int(feedback_bits),
            )
            finite_pd = compute_detection_probabilities(
                finite.deflection, p_fa)
            distributed[str(int(count))] = {
                "worst": float(np.min(finite_pd)),
                "worst_deflection": float(finite.worst_deflection),
                "exact_deflection_fraction": float(
                    finite.worst_deflection
                    / max(exact.worst_deflection, 1.0e-300)
                ),
                "dual_upper_bound": float(finite.dual_upper_bound),
                "primal_dual_gap": float(finite.primal_dual_gap),
                "power_balance_error_w": float(np.max(np.abs(
                    np.sum(finite.power_w, axis=1) - sensing_budget
                ))),
                "payload_bits_per_resolve": int(
                    Q * int(feedback_bits)
                    + int(count) * Q * (int(price_bits) + int(feedback_bits))
                ),
                "actual_payload_bits_per_resolve": int(
                    Q * int(feedback_bits)
                    + int(finite.rounds) * Q
                    * (int(price_bits) + int(feedback_bits))
                ),
                "actual_rounds": int(finite.rounds),
            }
        result = evaluate_physical_feasibility_oracles(
            _entries(data, row),
            sensing_power,
            deployed_pd,
            selected,
            num_uavs=K,
            num_targets=Q,
            p_fa=p_fa,
            total_power_w=1.0,
            communication_reserve_w=float(np.mean(comm_power)),
            target_pair_limit=pair_limit,
            reports_per_receiver=reports,
            seed=int(seed) * 1000 + int(frames[row]),
            detection_fusion_mode="local_only",
            coefficient_override=coefficient,
        )
        result.update({
            "seed": int(seed),
            "frame": int(frames[row]),
            "communication_reserve_w": float(np.mean(comm_power)),
            "power_balance_error_w": power_error,
            "fixed_owner": owners.tolist(),
            "fixed_owner_exact_worst": float(np.min(exact_pd)),
            "fixed_owner_exact_worst_deflection": float(
                exact.worst_deflection),
            "same_geometry_relaxed_upper_worst": float(np.min(relaxed_pd)),
            "repair_route": routing.route.value,
            "repair_route_reason": routing.reason,
            "fixed_owner_exact_power_balance_error_w": float(np.max(np.abs(
                np.sum(exact.power_w, axis=1) - sensing_budget
            ))),
            "distributed": distributed,
        })
        rows.append(result)

    keys = (
        "deployed_worst",
        "pair_only_worst",
        "power_only_worst",
        "single_worst",
        "duplex_worst",
        "pair_only_worst_gap",
        "power_only_worst_gap",
        "single_worst_gap",
        "joint_over_best_isolated_worst_gap",
        "duplex_over_single_worst_gap",
    )
    distributed_mean = {
        str(count): {
            key: float(np.mean([
                float(row["distributed"][str(count)][key])
                for row in rows
            ]))
            for key in (
                "worst", "worst_deflection", "exact_deflection_fraction",
                "dual_upper_bound", "primal_dual_gap",
                "power_balance_error_w", "payload_bits_per_resolve",
                "actual_payload_bits_per_resolve", "actual_rounds",
            )
        }
        for count in dual_rounds
    }
    return {
        "schema_version": 1,
        "scope": (
            "selection-seed final-resolve same-geometry optimized benchmark; "
            "alternating joint pair/power solve is not a proof of global optimality"
        ),
        "trace": str(trace_path),
        "config": str(config_path) if config_path is not None else None,
        "coefficient_source": (
            "alpha/g_dd/chi_rep analytic per-watt reconstruction"
            if cfg is not None
            else "legacy realized-deflection divided by sensing power"
        ),
        "seed_limit": len(seed_order),
        "dual_rounds": list(dual_rounds),
        "price_bits": int(price_bits),
        "feedback_bits": int(feedback_bits),
        "distributed_payload_scope": (
            "conservative payload-only upper count for target price plus "
            "owner Deflection feedback; early stopping not deducted; "
            "headers, contention and retransmission excluded"
        ),
        "distributed_method": (
            "target-cover initialization plus distributed Dantzig-Wolfe "
            "column generation; every returned primal is RF-feasible"
        ),
        "mean": {
            key: float(np.mean([float(row[key]) for row in rows]))
            for key in keys
        },
        "qos_feasible_rate": {
            name: float(np.mean([
                float(row[f"{name}_worst"]) >= 0.60 for row in rows
            ]))
            for name in ("deployed", "pair_only", "power_only", "single", "duplex")
        },
        "fixed_owner_exact_mean_worst": float(np.mean([
            float(row["fixed_owner_exact_worst"]) for row in rows
        ])),
        "fixed_owner_exact_qos_feasible_rate": float(np.mean([
            float(row["fixed_owner_exact_worst"]) >= 0.60 for row in rows
        ])),
        "distributed_mean": distributed_mean,
        "distributed_qos_feasible_rate": {
            str(count): float(np.mean([
                float(row["distributed"][str(count)]["worst"]) >= 0.60
                for row in rows
            ]))
            for count in dual_rounds
        },
        "same_geometry_relaxed_upper_mean_worst": float(np.mean([
            float(row["same_geometry_relaxed_upper_worst"]) for row in rows
        ])),
        "repair_route_count": {
            route: int(sum(row["repair_route"] == route for row in rows))
            for route in sorted({str(row["repair_route"]) for row in rows})
        },
        "rows": rows,
        "fresh_test_consumed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed-limit", type=int, default=10)
    parser.add_argument(
        "--dual-rounds", type=int, nargs="+",
        default=(1, 2, 4, 8, 16, 32),
    )
    parser.add_argument("--price-bits", type=int, default=6)
    parser.add_argument("--feedback-bits", type=int, default=16)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rounds = tuple(sorted({max(1, int(value)) for value in args.dual_rounds}))
    result = audit(
        args.trace,
        max(1, int(args.seed_limit)),
        dual_rounds=rounds,
        price_bits=int(args.price_bits),
        feedback_bits=int(args.feedback_bits),
        config_path=args.config,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "mean": result["mean"],
        "qos_feasible_rate": result["qos_feasible_rate"],
        "fixed_owner_exact_mean_worst": result[
            "fixed_owner_exact_mean_worst"],
        "distributed_mean": result["distributed_mean"],
        "distributed_qos_feasible_rate": result[
            "distributed_qos_feasible_rate"],
        "rows": [{
            key: row[key]
            for key in (
                "seed", "frame", "deployed_worst", "pair_only_worst",
                "power_only_worst", "single_worst", "duplex_worst",
            )
        } for row in result["rows"]],
    }, indent=2))


if __name__ == "__main__":
    main()
