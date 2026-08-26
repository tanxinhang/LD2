#!/usr/bin/env python
"""M3 three-floor certificates from quantized factor intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from uav_isac.coordination.certified_factor_message import (  # noqa: E402
    IntervalCode, certify_fixed_plan_three_floors,
    factor_coefficient_envelope, log_interval_quantize,
)
from uav_isac.coordination.coefficient_structure import (  # noqa: E402
    BistaticFactorization, balance_bistatic_factor_gauge,
    exact_bistatic_factorization,
)
from uav_isac.coordination.minimum_intervention_repair import (  # noqa: E402
    solve_minimum_intervention_task_repair_milp,
)
from uav_isac.coordination.scale_capability import cap_aware_sensing_budget  # noqa: E402
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)
from uav_isac.physical.detection import compute_detection_probabilities  # noqa: E402


def _metrics(pd: np.ndarray, tail_k: int) -> tuple[float, float, float]:
    return (float(np.min(pd)),
            float(np.mean(np.partition(pd, tail_k - 1)[:tail_k])),
            float(np.mean(pd)))


def audit(trace_path: Path, config_path: Path, g4a_path: Path) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg = load_config(str(config_path))
    g4a = json.loads(g4a_path.read_text(encoding="utf-8"))
    region = {(int(row["seed"]), int(row["frame"])) for row in g4a["rows"]}
    bits_set = (3, 4, 6, 8, 10, 12, 14, 16)
    power_scales = (0.0, 0.25, 0.5, 0.75, 1.0)
    design_headrooms = (0.0025, 0.005, 0.01)
    rows = []
    headroom_rows = []
    floors = tuple(cfg.marl.task_constrained_qos_floors)
    tail_k = int(floors[3])
    for index in range(np.asarray(data["seed"]).size):
        key = (int(data["seed"][index]), int(data["frame"][index]))
        if key not in region:
            continue
        alpha = np.asarray(data["privileged_alpha"][index], dtype=np.float64)
        report = np.asarray(data["privileged_chi_rep"][index], dtype=np.float64)
        dd = np.asarray(data["privileged_g_dd"][index], dtype=np.float64)
        active_dd = dd >= float(cfg.detection.g_min)
        coefficient = per_watt_deflection_tensor_from_observables(
            alpha, dd, report,
            T_sym=float(cfg.otfs.T_sym), M=int(cfg.otfs.M), N=int(cfg.otfs.N),
            kT=float(cfg.channel.kT), bandwidth_hz=float(cfg.otfs.B),
            noise_figure_db=float(cfg.channel.NF),
            g_tx_dbi=float(cfg.otfs.g_tx_dBi), g_rx_dbi=float(cfg.otfs.g_rx_dBi),
            n_cpi=int(cfg.otfs.n_cpi), g_min=float(cfg.detection.g_min),
            use_swerling=bool(cfg.channel.use_swerling))
        rate = np.asarray(data["outgoing_rate"][index], dtype=np.int64)
        mask = np.asarray(data["outgoing_token_mask"][index], dtype=bool)
        active_comm = (rate > 0) & np.any(mask, axis=1)
        comm = np.where(active_comm, np.clip(np.asarray(
            data["comm_fraction"][index]), 0, 1) * float(cfg.uav.P_isac_total), 0)
        budget = cap_aware_sensing_budget(
            float(cfg.uav.P_isac_total), comm, float(cfg.uav.P_sense_max))
        role = np.asarray(data["teacher_role"][index], dtype=np.int8)
        witness = solve_minimum_intervention_task_repair_milp(
            coefficient, budget,
            np.asarray(data["teacher_pair"][index], dtype=bool),
            role == 0, role == 1,
            np.asarray(data["teacher_receiver_owner"][index], dtype=np.int64),
            p_fa=float(cfg.detection.P_FA), task_floors=floors,
            target_pair_limit=int(np.asarray(data["target_pair_limit"]).reshape(-1)[0]),
            reports_per_receiver=int(np.asarray(data["reports_per_receiver"]).reshape(-1)[0]),
            pwl_epsilon=1.0e-3, time_limit_s=30.0)
        if not witness.feasible:
            raise RuntimeError(f"G4-A witness unexpectedly failed for {key}")
        robust_witnesses = []
        for headroom in design_headrooms:
            tightened = (
                float(floors[0]) + headroom,
                float(floors[1]) + headroom,
                float(floors[2]) + headroom,
                int(floors[3]),
            )
            robust = solve_minimum_intervention_task_repair_milp(
                coefficient, budget,
                np.asarray(data["teacher_pair"][index], dtype=bool),
                role == 0, role == 1,
                np.asarray(data["teacher_receiver_owner"][index], dtype=np.int64),
                p_fa=float(cfg.detection.P_FA), task_floors=tightened,
                target_pair_limit=int(np.asarray(data["target_pair_limit"]).reshape(-1)[0]),
                reports_per_receiver=int(np.asarray(data["reports_per_receiver"]).reshape(-1)[0]),
                pwl_epsilon=1.0e-3, time_limit_s=30.0)
            robust_witnesses.append((headroom, robust))
        K, _, Q = coefficient.shape
        factors = []
        for q in range(Q):
            raw = exact_bistatic_factorization(
                alpha[:, :, q] ** 2, report[:, :, q], active_dd[:, :, q])
            dense = raw.dense()
            positive = dense > 0.0
            scale = float(np.median(coefficient[:, :, q][positive] / dense[positive]))
            factors.append(balance_bistatic_factor_gauge(BistaticFactorization(
                raw.tx_factor, raw.rx_factor * scale,
                raw.inactive_offdiagonal)))
        values = np.concatenate([
            np.concatenate([factor.tx_factor, factor.rx_factor])
            for factor in factors])
        for bits in bits_set:
            code = log_interval_quantize(values, bits)
            coefficient_lower = np.zeros_like(coefficient)
            coefficient_upper = np.zeros_like(coefficient)
            offset = 0
            for q in range(Q):
                tx = IntervalCode(code.lower[offset:offset + K],
                                  code.upper[offset:offset + K],
                                  code.scale_lower, code.scale_upper, bits)
                rx = IntervalCode(code.lower[offset + K:offset + 2 * K],
                                  code.upper[offset + K:offset + 2 * K],
                                  code.scale_lower, code.scale_upper, bits)
                offset += 2 * K
                coefficient_lower[:, :, q], coefficient_upper[:, :, q] = (
                    factor_coefficient_envelope(tx, rx, active_dd[:, :, q]))
            for power_scale in power_scales:
                power = witness.sensing_power_w * power_scale
                certificate = certify_fixed_plan_three_floors(
                    coefficient_lower, coefficient_upper,
                    witness.receiver_owner, power,
                    p_fa=float(cfg.detection.P_FA), task_floors=floors)
                true_deflection = np.asarray([
                    np.dot(coefficient[:, witness.receiver_owner[q], q], power[:, q])
                    for q in range(Q)])
                true_metrics = _metrics(compute_detection_probabilities(
                    true_deflection, float(cfg.detection.P_FA)), tail_k)
                true_feasible = all(
                    value >= float(floor) - 1.0e-10
                    for value, floor in zip(true_metrics, floors[:3]))
                false_feasible = certificate.status == "CERT_FEASIBLE" and not true_feasible
                false_infeasible = certificate.status == "CERT_INFEASIBLE" and true_feasible
                rows.append({
                    "seed": key[0], "frame": key[1], "bits": bits,
                    "power_scale": power_scale,
                    "true_feasible": true_feasible,
                    "status": certificate.status,
                    "false_feasible": false_feasible,
                    "false_infeasible": false_infeasible,
                    "true_metrics": list(true_metrics),
                    "lower_metrics": list(certificate.lower_metrics),
                    "upper_metrics": list(certificate.upper_metrics),
                })
            for headroom, robust in robust_witnesses:
                if not robust.feasible:
                    headroom_rows.append({
                        "seed": key[0], "frame": key[1], "bits": bits,
                        "headroom": headroom, "design_feasible": False,
                        "status": "NO_WITNESS",
                    })
                    continue
                certificate = certify_fixed_plan_three_floors(
                    coefficient_lower, coefficient_upper,
                    robust.receiver_owner, robust.sensing_power_w,
                    p_fa=float(cfg.detection.P_FA), task_floors=floors)
                headroom_rows.append({
                    "seed": key[0], "frame": key[1], "bits": bits,
                    "headroom": headroom, "design_feasible": True,
                    "status": certificate.status,
                    "power_w": robust.total_sensing_power_w,
                    "nominal_power_w": witness.total_sensing_power_w,
                })
    summary = {}
    for bits in bits_set:
        selected = [row for row in rows if row["bits"] == bits]
        summary[str(bits)] = {
            "plans": len(selected),
            "true_feasible": sum(row["true_feasible"] for row in selected),
            "cert_feasible": sum(row["status"] == "CERT_FEASIBLE" for row in selected),
            "cert_infeasible": sum(row["status"] == "CERT_INFEASIBLE" for row in selected),
            "unresolved": sum(row["status"] == "UNRESOLVED" for row in selected),
            "false_feasible": sum(row["false_feasible"] for row in selected),
            "false_infeasible": sum(row["false_infeasible"] for row in selected),
        }
    headroom_summary = {}
    for headroom in design_headrooms:
        by_bits = {}
        for bits in bits_set:
            selected = [row for row in headroom_rows
                        if row["bits"] == bits and row["headroom"] == headroom]
            feasible = [row for row in selected if row["design_feasible"]]
            by_bits[str(bits)] = {
                "design_feasible": len(feasible),
                "cert_feasible": sum(
                    row["status"] == "CERT_FEASIBLE" for row in feasible),
                "unresolved": sum(
                    row["status"] == "UNRESOLVED" for row in feasible),
                "median_power_increase": float(np.median([
                    row["power_w"] - row["nominal_power_w"] for row in feasible
                ])) if feasible else None,
            }
        headroom_summary[str(headroom)] = by_bits
    return {
        "gate": "M3-static-three-floor-fixed-plan-certificate",
        "scope": "21 G4-A witnesses x five preregistered power scales; h=0 exact DD support",
        "frames": len(region), "plans_per_bits": len(region) * len(power_scales),
        "power_scales": list(power_scales), "summary": summary,
        "design_headrooms": list(design_headrooms),
        "headroom_summary": headroom_summary,
        "gate_pass": all(
            item["false_feasible"] == 0 and item["false_infeasible"] == 0
            for item in summary.values()),
        "rows": rows, "headroom_rows": headroom_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--g4a", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.trace, args.config, args.g4a)
    payload = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"rows", "headroom_rows"}},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
