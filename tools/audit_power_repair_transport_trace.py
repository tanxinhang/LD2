#!/usr/bin/env python
"""Replay causal power repair with an explicit physical U2U wire budget.

The audit preserves the frozen owner graph.  It predicts per-watt bistatic
gain from the previous resolved event and current geometry, reserves the
communication power required by the finite-round column-generation protocol,
and only then solves the sensing-power repair.  No evaluation seed is used to
fit a predictor inside this tool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from tools.audit_structure_trace_physical_bottleneck import (  # noqa: E402
    _ordered_unique,
    _recorded_power,
)
from uav_isac.coordination.geometry_gain_predictor import (  # noqa: E402
    predict_fixed_owner_gain_from_geometry,
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
from uav_isac.coordination.power_repair_transport import (  # noqa: E402
    PowerRepairWireLayout,
    certify_power_repair_transport,
)
from uav_isac.environment.communication import (  # noqa: E402
    InterUAVCommunicationModel,
)
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)
from uav_isac.physical.detection import (  # noqa: E402
    compute_detection_probabilities,
)


def _worst_pd(deflection: np.ndarray, p_fa: float) -> float:
    return float(np.min(compute_detection_probabilities(deflection, p_fa)))


def _episode_mean(rows: list[dict[str, object]], key: str) -> float:
    seed_ids = sorted({int(row["seed"]) for row in rows})
    return float(np.mean([
        np.mean([float(row[key]) for row in rows if int(row["seed"]) == seed])
        for seed in seed_ids
    ]))


def _communication_model(
    config_path: Path,
    *,
    message_dim: int = 16,
) -> tuple[object, InterUAVCommunicationModel]:
    cfg = load_config(str(config_path))
    ma = cfg.marl
    return cfg, InterUAVCommunicationModel(
        rate_bits_per_dim=list(ma.comm_rate_bits_per_dim),
        header_bits=int(ma.comm_header_bits),
        bandwidth_hz=float(ma.comm_bandwidth_hz),
        deadline_s=float(ma.comm_deadline_s),
        processing_delay_s=float(ma.comm_processing_delay_s),
        snr_threshold_db=float(ma.comm_snr_threshold_db),
        antenna_gain_dbi=float(ma.comm_antenna_gain_dbi),
        carrier_hz=float(cfg.otfs.fc),
        tx_power_w=float(ma.comm_tx_power_w),
        kT=float(cfg.channel.kT),
        noise_figure_db=float(cfg.channel.NF),
        dt=float(cfg.scenario.dt),
        message_dim=max(1, int(message_dim)),
    )


def _coefficient_from_trace(
    data: dict[str, np.ndarray], row: int, cfg: object,
) -> np.ndarray:
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


def audit(
    trace_path: Path,
    config_path: Path,
    *,
    seed_limit: int,
    rounds: int,
    price_bits: int,
    feedback_bits: int,
    snr_margin_db: float,
    latency_margin_s: float,
    qos_floor: float,
) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg, comm_model = _communication_model(config_path)
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    resolved = np.asarray(data["p0_resolved"], dtype=bool)
    seed_order = _ordered_unique(seeds)[:max(1, int(seed_limit))]
    indices = np.flatnonzero(resolved & np.isin(seeds, seed_order))
    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])
    if K != int(cfg.scenario.K) or Q != int(cfg.scenario.Q):
        raise ValueError("trace dimensions do not match the supplied config")
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    layout = PowerRepairWireLayout(
        num_agents=K,
        num_targets=Q,
        rounds=int(rounds),
        header_bits=int(cfg.marl.comm_header_bits),
        price_bits=int(price_bits),
        deflection_bits=int(feedback_bits),
    )
    previous_coefficient: dict[int, np.ndarray] = {}
    previous_uav_positions: dict[int, np.ndarray] = {}
    previous_target_states: dict[int, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for index in indices:
        row_id = int(index)
        seed = int(seeds[row_id])
        comm_power, sensing_power, recorded_error = _recorded_power(data, row_id)
        coefficient = _coefficient_from_trace(data, row_id, cfg)
        pair = np.asarray(data["teacher_pair"][row_id], dtype=bool)
        current_support = (
            np.asarray(data["privileged_candidate"][row_id], dtype=bool)
            & (
                np.asarray(data["privileged_g_dd"][row_id], dtype=np.float64)
                >= float(cfg.detection.g_min)
            )
        )
        selected = tuple(
            tuple(int(value) for value in edge) for edge in np.argwhere(pair)
        )
        true_gain, owners = fixed_owner_gain_matrix(coefficient, selected)
        deployed_deflection = np.sum(true_gain * sensing_power, axis=0)
        deployed_pd = compute_detection_probabilities(deployed_deflection, p_fa)
        report: dict[str, object] = {
            "seed": seed,
            "frame": int(frames[row_id]),
            "deployed_worst": float(np.min(deployed_pd)),
            "recorded_power_balance_error_w": float(recorded_error),
        }
        previous = previous_coefficient.get(seed)
        if previous is None:
            report.update({
                "causal_geometry_available": False,
                "causal_geometry_worst": float(np.min(deployed_pd)),
                "causal_geometry_estimated_worst": float(np.min(deployed_pd)),
                "causal_geometry_noop_estimated_worst": float(np.min(deployed_pd)),
                "causal_geometry_calibration_score": 0.0,
                "causal_geometry_noop_calibration_score": 0.0,
                "causal_geometry_delta_calibration_score": 0.0,
                "causal_geometry_power_total_variation": 0.0,
                "transport_feasible": False,
                "transport_reasons": ["causal:previous_event_unavailable"],
                "transport_total_bits": 0,
                "transport_max_packet_latency_s": 0.0,
                "transport_total_protocol_latency_s": 0.0,
                "transport_total_energy_j": 0.0,
                "transport_extra_comm_power_mean_w": 0.0,
                "transport_extra_comm_power_max_w": 0.0,
                "transport_power_balance_error_w": 0.0,
            })
        else:
            geometry = predict_fixed_owner_gain_from_geometry(
                previous,
                previous_uav_positions[seed],
                np.asarray(data["uav_positions"][row_id], dtype=np.float64),
                previous_target_states[seed],
                np.asarray(data["target_states"][row_id], dtype=np.float64),
                selected,
                current_support=current_support,
            )
            transport = certify_power_repair_transport(
                owners,
                np.asarray(data["uav_positions"][row_id], dtype=np.float64),
                comm_power,
                communication_model=comm_model,
                layout=layout,
                control_period_s=float(cfg.scenario.dt),
                snr_margin_db=float(snr_margin_db),
                latency_margin_s=float(latency_margin_s),
            )
            available = bool(transport.feasible)
            candidate_power = sensing_power
            estimated_pd = deployed_pd
            realized_pd = deployed_pd
            if available:
                sensing_budget = 1.0 - transport.projected_comm_power_w
                fixed_exact = solve_fixed_structure_maxmin_power_lp(
                    true_gain, sensing_budget)
                relaxed_ceiling = relaxed_same_geometry_target_ceiling(
                    coefficient, sensing_budget)
                fixed_upper_worst = _worst_pd(fixed_exact.deflection, p_fa)
                joint_upper_worst = _worst_pd(relaxed_ceiling, p_fa)
                route = route_isac_repair(
                    float(np.min(deployed_pd)),
                    fixed_power_upper=fixed_upper_worst,
                    joint_structure_upper=joint_upper_worst,
                    qos_floor=float(qos_floor),
                )
                repaired = distributed_column_generation_maxmin_power(
                    geometry.gain_per_watt,
                    sensing_budget,
                    rounds=int(rounds),
                    price_bits=int(price_bits),
                    feedback_bits=int(feedback_bits),
                )
                candidate_power = repaired.power_w
                estimated_pd = compute_detection_probabilities(
                    repaired.deflection, p_fa)
                realized_pd = compute_detection_probabilities(
                    np.sum(true_gain * candidate_power, axis=0), p_fa)
            else:
                fixed_upper_worst = float(np.min(deployed_pd))
                joint_upper_worst = float(np.min(deployed_pd))
                route = None
            noop_estimated_pd = compute_detection_probabilities(
                np.sum(geometry.gain_per_watt * sensing_power, axis=0), p_fa)
            predicted_delta = float(
                np.min(estimated_pd) - np.min(noop_estimated_pd))
            realized_delta = float(
                np.min(realized_pd) - np.min(deployed_pd))
            extra_comm = transport.projected_comm_power_w - comm_power
            total_power_error = float(np.max(np.abs(
                transport.projected_comm_power_w
                + np.sum(candidate_power, axis=1) - 1.0
            ))) if available else 0.0
            report.update({
                "causal_geometry_available": available,
                "causal_geometry_worst": float(np.min(realized_pd)),
                "causal_geometry_estimated_worst": float(np.min(estimated_pd)),
                "causal_geometry_noop_estimated_worst": float(
                    np.min(noop_estimated_pd)),
                "causal_geometry_calibration_score": float(max(
                    float(np.max(estimated_pd - realized_pd)), 0.0)),
                "causal_geometry_noop_calibration_score": float(max(
                    float(np.max(noop_estimated_pd - deployed_pd)), 0.0)),
                "causal_geometry_delta_calibration_score": float(max(
                    predicted_delta - realized_delta, 0.0)),
                "causal_geometry_power_total_variation": float(np.mean(
                    0.5 * (
                        np.abs(transport.projected_comm_power_w - comm_power)
                        + np.sum(
                            np.abs(candidate_power - sensing_power), axis=1)
                    ))),
                "causal_geometry_harmed": bool(
                    np.min(realized_pd) < np.min(deployed_pd) - 1.0e-12),
                "causal_geometry_owner_pd_abs_error": float(np.mean(
                    np.abs(estimated_pd - realized_pd))),
                "causal_geometry_owner_pd_overprediction": float(np.max(
                    estimated_pd - realized_pd)),
                "causal_geometry_direct_edge_count": int(
                    geometry.direct_edge_count),
                "causal_geometry_fallback_edge_count": int(
                    geometry.fallback_edge_count),
                "diagnostic_fixed_power_upper_worst": float(
                    fixed_upper_worst),
                "diagnostic_joint_structure_upper_worst": float(
                    joint_upper_worst),
                "diagnostic_bottleneck_route": (
                    route.route.value if route is not None else "hold_unverified"),
                "transport_feasible": available,
                "transport_reasons": list(transport.reasons),
                "transport_coordinator": int(transport.coordinator),
                "transport_total_bits": int(transport.total_over_air_bits),
                "transport_max_packet_latency_s": float(
                    transport.max_packet_latency_s),
                "transport_total_protocol_latency_s": float(
                    transport.total_protocol_latency_s),
                "transport_total_energy_j": float(transport.total_energy_j),
                "transport_min_snr_db": float(transport.min_snr_db),
                "transport_feedback_sender_count": int(
                    transport.feedback_sender_count),
                "transport_extra_comm_power_mean_w": float(np.mean(extra_comm)),
                "transport_extra_comm_power_max_w": float(np.max(extra_comm)),
                "transport_projected_comm_power_max_w": float(np.max(
                    transport.projected_comm_power_w)),
                "transport_power_balance_error_w": total_power_error,
            })
        previous_coefficient[seed] = coefficient.copy()
        previous_uav_positions[seed] = np.asarray(
            data["uav_positions"][row_id], dtype=np.float64).copy()
        previous_target_states[seed] = np.asarray(
            data["target_states"][row_id], dtype=np.float64).copy()
        rows.append(report)

    causal_rows = [
        row for row in rows if bool(row["causal_geometry_available"])
    ]
    attempted_rows = [
        row for row in rows
        if "transport_reasons" in row
        and row["transport_reasons"] != ["causal:previous_event_unavailable"]
    ]
    if not causal_rows:
        raise ValueError("no physically feasible causal transport event found")
    summary = {
        "available_event_count": len(causal_rows),
        "transport_attempt_event_count": len(attempted_rows),
        "transport_feasible_rate": float(np.mean([
            bool(row["transport_feasible"]) for row in attempted_rows])),
        "episode_mean_worst": _episode_mean(
            causal_rows, "causal_geometry_worst"),
        "event_mean_worst": float(np.mean([
            float(row["causal_geometry_worst"]) for row in causal_rows])),
        "event_noop_mean_worst": float(np.mean([
            float(row["deployed_worst"]) for row in causal_rows])),
        "event_qos_feasible_rate": float(np.mean([
            float(row["causal_geometry_worst"]) >= float(qos_floor)
            for row in causal_rows
        ])),
        "harm_rate_against_same_event_noop": float(np.mean([
            bool(row["causal_geometry_harmed"]) for row in causal_rows])),
        "owner_pd_mean_absolute_error": float(np.mean([
            float(row["causal_geometry_owner_pd_abs_error"])
            for row in causal_rows
        ])),
        "owner_pd_p95_max_overprediction": float(np.quantile([
            float(row["causal_geometry_owner_pd_overprediction"])
            for row in causal_rows
        ], 0.95)),
        "mean_total_bits": float(np.mean([
            int(row["transport_total_bits"]) for row in causal_rows])),
        "mean_max_packet_latency_s": float(np.mean([
            float(row["transport_max_packet_latency_s"])
            for row in causal_rows
        ])),
        "max_total_protocol_latency_s": float(np.max([
            float(row["transport_total_protocol_latency_s"])
            for row in causal_rows
        ])),
        "mean_total_energy_j": float(np.mean([
            float(row["transport_total_energy_j"]) for row in causal_rows])),
        "mean_extra_comm_power_w": float(np.mean([
            float(row["transport_extra_comm_power_mean_w"])
            for row in causal_rows
        ])),
        "max_extra_comm_power_w": float(np.max([
            float(row["transport_extra_comm_power_max_w"])
            for row in causal_rows
        ])),
        "max_isac_power_balance_error_w": float(np.max([
            float(row["transport_power_balance_error_w"])
            for row in causal_rows
        ])),
        "diagnostic_bottleneck_route_counts": {
            route: int(sum(
                row["diagnostic_bottleneck_route"] == route
                for row in causal_rows
            ))
            for route in (
                "no_op", "fixed_structure_power", "joint_structure_power",
                "slow_geometry", "hold_unverified",
            )
        },
        "diagnostic_fixed_power_upper_qos_rate": float(np.mean([
            float(row["diagnostic_fixed_power_upper_worst"]) >= float(qos_floor)
            for row in causal_rows
        ])),
        "diagnostic_joint_structure_upper_qos_rate": float(np.mean([
            float(row["diagnostic_joint_structure_upper_worst"]) >= float(qos_floor)
            for row in causal_rows
        ])),
    }
    return {
        "schema_version": 1,
        "scope": (
            "causal fixed-owner four-round power repair with explicit U2U "
            "headers, orthogonal owner feedback, price broadcast, latency, "
            "energy and exact 1 W communication+sensing projection"
        ),
        "trace": str(trace_path),
        "config": str(config_path),
        "seed_order": seed_order,
        "event_count": len(rows),
        "rounds": int(rounds),
        "qos_floor": float(qos_floor),
        "wire_layout": {
            "shared_bits": int(layout.shared_bits),
            "price_packet_bits": int(layout.price_packet_bits),
            "price_quantization_bits": int(price_bits),
            "deflection_quantization_bits": int(feedback_bits),
        },
        "robustness_margins": {
            "snr_margin_db": float(snr_margin_db),
            "latency_margin_s": float(latency_margin_s),
            "statistically_calibrated": False,
        },
        "summary": summary,
        "certificate_ready": False,
        "certificate_blockers": [
            "deterministic free-space U2U replay has no stochastic packet-loss residual",
            "the 3 dB/latency engineering margins are not yet data-calibrated",
            "the owner structure is frozen rather than jointly optimized",
        ],
        "fresh_test_consumed": False,
        "elapsed_seconds": float(time.perf_counter() - started),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed-limit", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--price-bits", type=int, default=6)
    parser.add_argument("--feedback-bits", type=int, default=16)
    parser.add_argument("--snr-margin-db", type=float, default=3.0)
    parser.add_argument("--latency-margin-s", type=float, default=5.0e-4)
    parser.add_argument("--qos-floor", type=float, default=0.60)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(
        args.trace,
        args.config,
        seed_limit=max(1, int(args.seed_limit)),
        rounds=max(1, int(args.rounds)),
        price_bits=int(args.price_bits),
        feedback_bits=int(args.feedback_bits),
        snr_margin_db=float(args.snr_margin_db),
        latency_margin_s=float(args.latency_margin_s),
        qos_floor=float(args.qos_floor),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "event_count": result["event_count"],
        "summary": result["summary"],
        "certificate_ready": result["certificate_ready"],
    }, indent=2))


if __name__ == "__main__":
    main()
