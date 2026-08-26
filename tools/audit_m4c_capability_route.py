#!/usr/bin/env python
"""M4-C-light fixed-structure interval capability-route audit.

This is a centralized semantic shadow.  It asks only whether quantized factor
intervals preserve the L1 decision ``gamma* <= 1`` for a fixed G4-A structure;
it does not require equality of power vectors, LP bases, or dual solutions.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from uav_isac.coordination.capability import capability_gauge_pwl_lp  # noqa: E402
from uav_isac.coordination.certified_factor_message import (  # noqa: E402
    IntervalCode, certify_capability_route, factor_coefficient_envelope,
    log_interval_quantize, task_equivalent_payload_bits,
)
from uav_isac.coordination.coefficient_structure import (  # noqa: E402
    BistaticFactorization, balance_bistatic_factor_gauge,
    exact_bistatic_factorization,
)
from uav_isac.coordination.minimum_intervention_repair import (  # noqa: E402
    solve_minimum_intervention_task_repair_milp,
)
from uav_isac.coordination.pwl_pd import saturating_chord_lower_bound  # noqa: E402
from uav_isac.coordination.scale_capability import cap_aware_sensing_budget  # noqa: E402
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)
from uav_isac.physical.detection import (  # noqa: E402
    minimum_deflection_for_detection_probability,
)


BITS = (3, 4, 6, 8, 10, 12, 14, 16)
TARGET_GAMMAS = (0.90, 0.97, 0.99, 1.01, 1.03, 1.10)


def _fixed_gain(coefficient: np.ndarray, selected_set, owner: np.ndarray) -> np.ndarray:
    K, _, Q = coefficient.shape
    selected = set(tuple(map(int, edge)) for edge in selected_set)
    gain = np.zeros((K, Q), dtype=np.float64)
    for q in range(Q):
        j = int(owner[q])
        for i in range(K):
            if (i, j, q) in selected:
                gain[i, q] = coefficient[i, j, q]
    return gain


def audit(trace_path: Path, config_path: Path, g4a_path: Path) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg = load_config(str(config_path))
    g4a = json.loads(g4a_path.read_text(encoding="utf-8"))
    region = {(int(row["seed"]), int(row["frame"])) for row in g4a["rows"]}
    floors = tuple(cfg.marl.task_constrained_qos_floors)
    p_fa = float(cfg.detection.P_FA)
    d_min = float(minimum_deflection_for_detection_probability(
        np.asarray([float(floors[0])]), p_fa)[0])
    rows = []
    containment_violations = {str(bits): 0 for bits in BITS}

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
        token_mask = np.asarray(data["outgoing_token_mask"][index], dtype=bool)
        comm_active = (rate > 0) & np.any(token_mask, axis=1)
        comm = np.where(comm_active, np.clip(np.asarray(
            data["comm_fraction"][index]), 0, 1) * float(cfg.uav.P_isac_total), 0)
        budget = cap_aware_sensing_budget(
            float(cfg.uav.P_isac_total), comm, float(cfg.uav.P_sense_max))
        role = np.asarray(data["teacher_role"][index], dtype=np.int8)
        witness = solve_minimum_intervention_task_repair_milp(
            coefficient, budget,
            np.asarray(data["teacher_pair"][index], dtype=bool),
            role == 0, role == 1,
            np.asarray(data["teacher_receiver_owner"][index], dtype=np.int64),
            p_fa=p_fa, task_floors=floors,
            target_pair_limit=int(np.asarray(data["target_pair_limit"]).reshape(-1)[0]),
            reports_per_receiver=int(np.asarray(data["reports_per_receiver"]).reshape(-1)[0]),
            pwl_epsilon=1.0e-3, time_limit_s=30.0)
        if not witness.feasible:
            raise RuntimeError(f"G4-A witness unexpectedly failed for {key}")

        K, _, Q = coefficient.shape
        support = active_dd & ~np.eye(K, dtype=bool)[:, :, None]
        exceptions = int(Q * K * (K - 1) - np.sum(support))
        factors = []
        for q in range(Q):
            raw = exact_bistatic_factorization(
                alpha[:, :, q] ** 2, report[:, :, q], active_dd[:, :, q])
            raw_dense = raw.dense()
            positive = raw_dense > 0.0
            detector_scale = float(np.median(
                coefficient[:, :, q][positive] / raw_dense[positive]))
            factors.append(balance_bistatic_factor_gauge(BistaticFactorization(
                raw.tx_factor, raw.rx_factor * detector_scale,
                raw.inactive_offdiagonal)))
        factor_values = np.concatenate([
            np.concatenate([factor.tx_factor, factor.rx_factor])
            for factor in factors])
        envelopes = {}
        for bits in BITS:
            code = log_interval_quantize(factor_values, bits)
            lower = np.zeros_like(coefficient)
            upper = np.zeros_like(coefficient)
            offset = 0
            for q in range(Q):
                tx = IntervalCode(code.lower[offset:offset + K],
                                  code.upper[offset:offset + K],
                                  code.scale_lower, code.scale_upper, bits)
                rx = IntervalCode(code.lower[offset + K:offset + 2 * K],
                                  code.upper[offset + K:offset + 2 * K],
                                  code.scale_lower, code.scale_upper, bits)
                offset += 2 * K
                lower[:, :, q], upper[:, :, q] = factor_coefficient_envelope(
                    tx, rx, active_dd[:, :, q])
            containment_violations[str(bits)] += int(np.sum(
                (coefficient < lower * (1.0 - 1.0e-12))
                | (coefficient > upper * (1.0 + 1.0e-12))))
            envelopes[bits] = (lower, upper)

        exact_gain = _fixed_gain(
            coefficient, witness.selected_set, witness.receiver_owner)
        # Ensure the saturating constant is present.  It then remains a global
        # high-D lower bound even when the diagnostic gauge has gamma > 1.
        d_saturation = float(minimum_deflection_for_detection_probability(
            np.asarray([0.999]), p_fa)[0])
        d_max = max(float(np.max(np.sum(
            exact_gain * budget[:, None], axis=0))) * 2.0,
                    d_saturation + 1.0, d_min + 1.0)
        slopes, intercepts, _ = saturating_chord_lower_bound(
            p_fa, d_min, d_max, 1.0e-3)

        def evaluate(gain: np.ndarray, local_budget: np.ndarray) -> float | None:
            return capability_gauge_pwl_lp(
                gain, local_budget, p_fa, floors,
                slopes, intercepts, d_min)

        baseline_gamma = evaluate(exact_gain, budget)
        if baseline_gamma is None or baseline_gamma > 1.0 + 2.0e-5:
            raise RuntimeError(f"fixed G4-A structure gauge failed for {key}")
        for target_gamma in TARGET_GAMMAS:
            budget_scale = float(baseline_gamma) / float(target_gamma)
            if budget_scale > 1.0 + 1.0e-7:
                continue
            scaled_budget = budget * min(budget_scale, 1.0)
            true_gamma = evaluate(exact_gain, scaled_budget)
            if true_gamma is None:
                raise RuntimeError(f"exact route gauge failed for {key}")
            true_status = ("L1_FEASIBLE" if true_gamma <= 1.0 + 1.0e-8
                           else "L1_FALLBACK")
            for bits in BITS:
                lower_tensor, upper_tensor = envelopes[bits]
                lower_gain = _fixed_gain(
                    lower_tensor, witness.selected_set, witness.receiver_owner)
                upper_gain = _fixed_gain(
                    upper_tensor, witness.selected_set, witness.receiver_owner)
                certificate = certify_capability_route(
                    lower_gain, upper_gain,
                    lambda gain, b=scaled_budget: evaluate(gain, b),
                    tolerance=1.0e-8)
                certified_route = {
                    "CERT_L1_FEASIBLE": "L1_FEASIBLE",
                    "CERT_L1_FALLBACK": "L1_FALLBACK",
                }.get(certificate.status)
                bracket_violation = (
                    certificate.gamma_lower is not None
                    and certificate.gamma_upper is not None
                    and not (certificate.gamma_lower - 1.0e-7 <= true_gamma
                             <= certificate.gamma_upper + 1.0e-7))
                rows.append({
                    "seed": key[0], "frame": key[1], "bits": bits,
                    "target_gamma": target_gamma,
                    "budget_scale": budget_scale,
                    "true_gamma": float(true_gamma),
                    "decision_margin": abs(1.0 - float(true_gamma)),
                    "true_status": true_status,
                    "status": certificate.status,
                    "gamma_lower": certificate.gamma_lower,
                    "gamma_upper": certificate.gamma_upper,
                    "gauge_bracket_violation": bracket_violation,
                    "false_route": (certified_route is not None
                                    and certified_route != true_status),
                    "payload_bits": task_equivalent_payload_bits(
                        "factor_log", K, Q, bits, exceptions).total_bits,
                })

    scenarios = sorted({(row["seed"], row["frame"], row["target_gamma"])
                        for row in rows})
    adaptive = []
    for scenario in scenarios:
        candidates = sorted(
            (row for row in rows
             if (row["seed"], row["frame"], row["target_gamma"]) == scenario
             and row["status"] != "UNRESOLVED"),
            key=lambda row: (row["payload_bits"], row["bits"]))
        base = next(row for row in rows
                    if (row["seed"], row["frame"], row["target_gamma"]) == scenario)
        adaptive.append({
            "seed": scenario[0], "frame": scenario[1],
            "target_gamma": scenario[2], "true_gamma": base["true_gamma"],
            "decision_margin": base["decision_margin"],
            "true_status": base["true_status"],
            "min_bits": candidates[0]["bits"] if candidates else None,
            "payload_bits": candidates[0]["payload_bits"] if candidates else None,
        })

    by_target = {}
    for target in TARGET_GAMMAS:
        selected = [row for row in adaptive if row["target_gamma"] == target]
        certified = [row for row in selected if row["min_bits"] is not None]
        by_target[str(target)] = {
            "scenarios": len(selected), "certified": len(certified),
            "median_true_gamma": (float(np.median([row["true_gamma"] for row in selected]))
                                  if selected else None),
            "median_decision_margin": (float(np.median([
                row["decision_margin"] for row in selected])) if selected else None),
            "min_bit_distribution": dict(sorted(Counter(
                row["min_bits"] for row in certified).items())),
            "median_payload_bits": (float(np.median([
                row["payload_bits"] for row in certified])) if certified else None),
        }
    correlation_rows = [row for row in adaptive if row["min_bits"] is not None]
    correlation = spearmanr(
        [row["decision_margin"] for row in correlation_rows],
        [row["min_bits"] for row in correlation_rows]) if len(correlation_rows) > 1 else None
    false_routes = sum(row["false_route"] for row in rows)
    bracket_violations = sum(row["gauge_bracket_violation"] for row in rows)
    return {
        "gate": "M4-C-light-fixed-structure-capability-route",
        "scope": ("21 correlated development frames; centralized semantic shadow; "
                  "counterfactual residual-budget scaling; fixed G4-A structure"),
        "decision_rule": "L1 feasible iff conservative-PWL gamma* <= 1",
        "non_claims": ["power-vector equality", "LP-basis equality",
                       "distributed scale agreement", "over-air admissibility"],
        "bits": list(BITS), "target_gammas": list(TARGET_GAMMAS),
        "containment_violations": containment_violations,
        "gauge_bracket_violations": bracket_violations,
        "false_certified_routes": false_routes,
        "scenarios": len(scenarios),
        "certified_scenarios": sum(row["min_bits"] is not None for row in adaptive),
        "by_target_gamma": by_target,
        "margin_vs_min_bits_spearman": ({
            "rho_descriptive_only": float(correlation.statistic),
            "n_correlated_scenarios": len(correlation_rows),
            "independent_episodes": 5,
        } if correlation is not None else None),
        "safety_gate_pass": false_routes == 0 and bracket_violations == 0 and all(
            value == 0 for value in containment_violations.values()),
        "route_value_gate_pass": all(
            item["certified"] == item["scenarios"] and item["scenarios"] > 0
            for item in by_target.values()),
        "adaptive": adaptive, "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--g4a", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.trace, args.config, args.g4a)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"rows", "adaptive"}},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
