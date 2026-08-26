#!/usr/bin/env python
"""M4-D-light lexicographic L2 decision-rate audit on a frozen candidate pool."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from uav_isac.coordination.capability import minimum_total_power_pwl_lp  # noqa: E402
from uav_isac.coordination.certified_factor_message import (  # noqa: E402
    IntervalCode, factor_coefficient_envelope, log_interval_quantize,
    task_equivalent_payload_bits,
)
from uav_isac.coordination.coefficient_structure import (  # noqa: E402
    BistaticFactorization, balance_bistatic_factor_gauge,
    exact_bistatic_factorization,
)
from uav_isac.coordination.fixing_conflict_filter import (  # noqa: E402
    PermissionBlock, irreducible_feasible_permission_block,
)
from uav_isac.coordination.minimum_intervention_repair import (  # noqa: E402
    MinimumInterventionRepairResult, solve_minimum_intervention_task_repair_milp,
)
from uav_isac.coordination.pwl_pd import saturating_chord_lower_bound  # noqa: E402
from uav_isac.coordination.scale_capability import cap_aware_sensing_budget  # noqa: E402
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)
from uav_isac.physical.detection import (  # noqa: E402
    minimum_deflection_for_detection_probability,
)


METHODS = ("dense_log", "factor_log")
BITS = (3, 4, 6, 8, 10, 12, 14, 16)
POWER_EQ_TOLERANCE_W = 1.0e-8
NEAR_TIE_RELATIVE_GAP = 1.0e-3


@dataclass(frozen=True)
class Candidate:
    name: str
    result: MinimumInterventionRepairResult
    exact_power_w: float

    @property
    def discrete_key(self) -> tuple[int, int]:
        return self.result.closure_cardinality, self.result.prepare_bits


def _signature(result: MinimumInterventionRepairResult) -> tuple:
    return (
        tuple(result.selected_set), tuple(bool(x) for x in result.tx_role),
        tuple(bool(x) for x in result.rx_role),
        tuple(int(x) for x in result.receiver_owner),
    )


def _fixed_gain(coefficient: np.ndarray, result: MinimumInterventionRepairResult) -> np.ndarray:
    K, _, Q = coefficient.shape
    selected = set(tuple(map(int, edge)) for edge in result.selected_set)
    gain = np.zeros((K, Q), dtype=np.float64)
    for q in range(Q):
        owner = int(result.receiver_owner[q])
        for i in range(K):
            if (i, owner, q) in selected:
                gain[i, q] = coefficient[i, owner, q]
    return gain


def _b2b_repair(
    coefficient, budget, selected, owner, role, floors, common, strategy,
) -> MinimumInterventionRepairResult:
    """Replay one already-established G4-B2b permission-filter candidate."""
    K, _, Q = coefficient.shape

    def solve(block: PermissionBlock, feasibility_only: bool):
        return solve_minimum_intervention_task_repair_milp(
            coefficient, budget, selected, role == 0, role == 1, owner,
            task_floors=floors, changeable_uavs=block.uavs,
            changeable_targets=block.targets,
            changeable_role_uavs=block.role_uavs,
            feasibility_only=feasibility_only, **common)

    filtered = irreducible_feasible_permission_block(
        PermissionBlock(tuple(range(K)), tuple(range(Q)), tuple(range(K))),
        lambda block: solve(block, True).feasible, strategy=strategy)
    return solve(filtered.block, False)


def _strictly_better_upper_than_lower(
    challenger: Candidate, incumbent: Candidate,
    challenger_power_upper: float | None,
    incumbent_power_lower: float | None,
) -> bool:
    if challenger.discrete_key[0] != incumbent.discrete_key[0]:
        return challenger.discrete_key[0] < incumbent.discrete_key[0]
    if challenger.discrete_key[1] != incumbent.discrete_key[1]:
        return challenger.discrete_key[1] < incumbent.discrete_key[1]
    return (
        challenger_power_upper is not None
        and incumbent_power_lower is not None
        and challenger_power_upper < incumbent_power_lower - POWER_EQ_TOLERANCE_W
    )


def _ambiguous_indices(
    candidates: list[Candidate],
    lower_power: list[float | None] | None,
    upper_power: list[float | None] | None,
) -> tuple[int, ...]:
    # At B=0 only coefficient-independent lex stages may eliminate candidates.
    if lower_power is None or upper_power is None:
        best_discrete = min(candidate.discrete_key for candidate in candidates)
        return tuple(i for i, candidate in enumerate(candidates)
                     if candidate.discrete_key == best_discrete)
    ambiguous = []
    for i, candidate in enumerate(candidates):
        eliminated = any(
            j != i and _strictly_better_upper_than_lower(
                challenger, candidate, upper_power[j], lower_power[i])
            for j, challenger in enumerate(candidates))
        if not eliminated:
            ambiguous.append(i)
    return tuple(ambiguous)


def audit(trace_path: Path, config_path: Path, g4a_path: Path) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg = load_config(str(config_path))
    g4a = json.loads(g4a_path.read_text(encoding="utf-8"))
    g4a_reference = {(int(row["seed"]), int(row["frame"])): row
                     for row in g4a["rows"]}
    region = set(g4a_reference)
    floors = tuple(cfg.marl.task_constrained_qos_floors)
    p_fa = float(cfg.detection.P_FA)
    d_min = float(minimum_deflection_for_detection_probability(
        np.asarray([float(floors[0])]), p_fa)[0])
    rows = []
    frames = []
    containment = {method: {str(bits): 0 for bits in BITS} for method in METHODS}
    bracket_violations = 0
    replayed_g4a_closures = []

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
        K, _, Q = coefficient.shape
        rate = np.asarray(data["outgoing_rate"][index], dtype=np.int64)
        mask = np.asarray(data["outgoing_token_mask"][index], dtype=bool)
        active_comm = (rate > 0) & np.any(mask, axis=1)
        comm = np.where(active_comm, np.clip(np.asarray(
            data["comm_fraction"][index]), 0, 1) * float(cfg.uav.P_isac_total), 0)
        budget = cap_aware_sensing_budget(
            float(cfg.uav.P_isac_total), comm, float(cfg.uav.P_sense_max))
        selected = np.asarray(data["teacher_pair"][index], dtype=bool)
        owner = np.asarray(data["teacher_receiver_owner"][index], dtype=np.int64)
        role = np.asarray(data["teacher_role"][index], dtype=np.int8)
        common = dict(
            p_fa=p_fa,
            target_pair_limit=int(np.asarray(data["target_pair_limit"]).reshape(-1)[0]),
            reports_per_receiver=int(np.asarray(data["reports_per_receiver"]).reshape(-1)[0]),
            pwl_epsilon=1.0e-3, time_limit_s=30.0)
        exact = solve_minimum_intervention_task_repair_milp(
            coefficient, budget, selected, role == 0, role == 1, owner,
            task_floors=floors, **common)
        replayed_g4a_closures.append(int(exact.closure_cardinality))
        raw_candidates = [("g4a_exact", exact)]
        for strategy in ("sequential", "bisect"):
            raw_candidates.append((
                f"g4b2b_{strategy}",
                _b2b_repair(
                    coefficient, budget, selected, owner, role, floors,
                    common, strategy)))

        ceiling = np.sum(coefficient.max(axis=1) * budget[:, None], axis=0)
        d_saturation = float(minimum_deflection_for_detection_probability(
            np.asarray([0.999]), p_fa)[0])
        d_max = max(float(np.max(ceiling)), d_saturation + 1.0, d_min + 1.0)
        slopes, intercepts, _ = saturating_chord_lower_bound(
            p_fa, d_min, d_max, 1.0e-3)

        unique = {}
        for name, result in raw_candidates:
            if not result.feasible:
                continue
            signature = _signature(result)
            if signature in unique:
                unique[signature][0] += f"+{name}"
                continue
            power = minimum_total_power_pwl_lp(
                _fixed_gain(coefficient, result), budget, p_fa, floors,
                slopes, intercepts, d_min)
            if power is not None:
                unique[signature] = [name, result, power[0]]
        candidates = [Candidate(value[0], value[1], value[2])
                      for value in unique.values()]
        if not candidates:
            raise RuntimeError(f"no feasible frozen candidate for {key}")

        best_discrete = min(candidate.discrete_key for candidate in candidates)
        discrete_survivors = [i for i, candidate in enumerate(candidates)
                              if candidate.discrete_key == best_discrete]
        best_power = min(candidates[i].exact_power_w for i in discrete_survivors)
        optimal = tuple(i for i in discrete_survivors
                        if candidates[i].exact_power_w
                        <= best_power + POWER_EQ_TOLERANCE_W)
        nonoptimal = [i for i in range(len(candidates)) if i not in optimal]
        if nonoptimal:
            competitor = min(nonoptimal, key=lambda i: (
                candidates[i].discrete_key, candidates[i].exact_power_w))
            if candidates[competitor].discrete_key[0] != best_discrete[0]:
                first_stage = 1
                margin = float(candidates[competitor].discrete_key[0] - best_discrete[0])
                margin_units = "closure_count"
            elif candidates[competitor].discrete_key[1] != best_discrete[1]:
                first_stage = 2
                margin = float(candidates[competitor].discrete_key[1] - best_discrete[1])
                margin_units = "prepare_bits"
            else:
                first_stage = 3
                margin = float(candidates[competitor].exact_power_w - best_power)
                margin_units = "watt"
        else:
            first_stage, margin, margin_units = None, None, None
        if len(candidates) == 1:
            group = "DEGENERATE_SINGLETON"
        elif len(optimal) > 1:
            group = "D3_equivalent"
        elif first_stage in (1, 2) or first_stage is None:
            group = "D1_discrete"
        elif margin / max(best_power, 1.0e-15) <= NEAR_TIE_RELATIVE_GAP:
            group = "D3_near_tie"
        else:
            group = "D2_capability"

        support = active_dd & ~np.eye(K, dtype=bool)[:, :, None]
        exceptions = int(Q * K * (K - 1) - np.sum(support))
        dense_values = coefficient[support]
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

        ambiguity = {method: {0: _ambiguous_indices(candidates, None, None)}
                     for method in METHODS}
        for bits in BITS:
            dense_code = log_interval_quantize(dense_values, bits)
            dense_lower = np.zeros_like(coefficient)
            dense_upper = np.zeros_like(coefficient)
            dense_lower[support] = dense_code.lower
            dense_upper[support] = dense_code.upper
            factor_code = log_interval_quantize(factor_values, bits)
            factor_lower = np.zeros_like(coefficient)
            factor_upper = np.zeros_like(coefficient)
            offset = 0
            for q in range(Q):
                tx = IntervalCode(factor_code.lower[offset:offset + K],
                                  factor_code.upper[offset:offset + K],
                                  factor_code.scale_lower, factor_code.scale_upper, bits)
                rx = IntervalCode(factor_code.lower[offset + K:offset + 2 * K],
                                  factor_code.upper[offset + K:offset + 2 * K],
                                  factor_code.scale_lower, factor_code.scale_upper, bits)
                offset += 2 * K
                factor_lower[:, :, q], factor_upper[:, :, q] = (
                    factor_coefficient_envelope(tx, rx, active_dd[:, :, q]))
            for method, lower_tensor, upper_tensor in (
                ("dense_log", dense_lower, dense_upper),
                ("factor_log", factor_lower, factor_upper),
            ):
                containment[method][str(bits)] += int(np.sum(
                    (coefficient < lower_tensor * (1.0 - 1.0e-12))
                    | (coefficient > upper_tensor * (1.0 + 1.0e-12))))
                power_lower = []
                power_upper = []
                for candidate in candidates:
                    optimistic = minimum_total_power_pwl_lp(
                        _fixed_gain(upper_tensor, candidate.result), budget,
                        p_fa, floors, slopes, intercepts, d_min)
                    conservative = minimum_total_power_pwl_lp(
                        _fixed_gain(lower_tensor, candidate.result), budget,
                        p_fa, floors, slopes, intercepts, d_min)
                    low = None if optimistic is None else optimistic[0]
                    high = None if conservative is None else conservative[0]
                    power_lower.append(low)
                    power_upper.append(high)
                    if not (low is not None and low - 1.0e-8
                            <= candidate.exact_power_w
                            and (high is None or candidate.exact_power_w
                                 <= high + 1.0e-8)):
                        bracket_violations += 1
                ambiguous = _ambiguous_indices(
                    candidates, power_lower, power_upper)
                ambiguity[method][bits] = ambiguous
                certified = set(ambiguous) == set(optimal)
                rows.append({
                    "seed": key[0], "frame": key[1], "method": method,
                    "bits": bits, "candidate_count": len(candidates),
                    "optimal_count": len(optimal), "group": group,
                    "first_distinguishing_stage": first_stage,
                    "margin": margin, "margin_units": margin_units,
                    "ambiguous_count": len(ambiguous),
                    "certified": certified,
                    "certified_nonoptimal": certified and any(
                        item not in optimal for item in ambiguous),
                    "payload_bits": task_equivalent_payload_bits(
                        method, K, Q, bits, exceptions).total_bits,
                })

        frame_summary = {
            "seed": key[0], "frame": key[1], "candidate_count": len(candidates),
            "candidate_names": [candidate.name for candidate in candidates],
            "objectives": [[*candidate.discrete_key, candidate.exact_power_w]
                           for candidate in candidates],
            "optimal_indices": list(optimal), "optimal_count": len(optimal),
            "group": group, "first_distinguishing_stage": first_stage,
            "margin": margin, "margin_units": margin_units,
        }
        for method in METHODS:
            zero_certified = set(ambiguity[method][0]) == set(optimal)
            choices = ([{"bits": 0, "payload_bits": 0}]
                       if zero_certified else [
                row for row in rows if row["seed"] == key[0]
                and row["frame"] == key[1] and row["method"] == method
                and row["certified"]])
            choice = min(choices, key=lambda item: item["payload_bits"], default=None)
            frame_summary[f"{method}_min_bits"] = (
                choice["bits"] if choice is not None else None)
            frame_summary[f"{method}_payload_bits"] = (
                choice["payload_bits"] if choice is not None else None)
            frame_summary[f"{method}_ambiguity_path"] = {
                str(bits): len(indices) for bits, indices in ambiguity[method].items()}
        frames.append(frame_summary)

    summary = {}
    for method in METHODS:
        certified = [frame for frame in frames
                     if frame[f"{method}_min_bits"] is not None]
        summary[method] = {
            "certified_frames": len(certified),
            "min_bit_distribution": dict(sorted(Counter(
                frame[f"{method}_min_bits"] for frame in certified).items())),
            "median_payload_bits": float(np.median([
                frame[f"{method}_payload_bits"] for frame in certified]))
            if certified else None,
        }
    capability_frames = [frame for frame in frames
                         if frame["first_distinguishing_stage"] == 3]
    comparison_frames = [frame for frame in capability_frames
                         if all(frame[f"{method}_payload_bits"] is not None
                                for method in METHODS)]
    factor_wins = sum(frame["factor_log_payload_bits"]
                      < frame["dense_log_payload_bits"]
                      for frame in comparison_frames)
    correlation_frames = [frame for frame in capability_frames
                          if frame["factor_log_min_bits"] is not None
                          and frame["margin"] is not None and frame["margin"] > 0.0]
    correlation = (spearmanr(
        [frame["margin"] for frame in correlation_frames],
        [frame["factor_log_min_bits"] for frame in correlation_frames])
        if len(correlation_frames) > 1 else None)
    false_nonoptimal = sum(row["certified_nonoptimal"] for row in rows)
    nondegenerate_frames = sum(
        frame["candidate_count"] >= 2 for frame in frames)
    candidate_pool_gate = nondegenerate_frames == len(frames) and len(frames) > 0
    return {
        "gate": "M4-D-light-frozen-candidate-lex-decision-rate",
        "scope": ("21 correlated development frames; candidate pool frozen before "
                  "quantization from replayable G4-A and two existing G4-B2b "
                  "permission filters; G4-B ladders excluded because physical_pd "
                  "is absent"),
        "candidate_generation_nonclaim": True,
        "lex_objective": ["dependency_closure", "prepare_bits", "total_sensing_power"],
        "power_equivalence_tolerance_w": POWER_EQ_TOLERANCE_W,
        "near_tie_relative_gap": NEAR_TIE_RELATIVE_GAP,
        "methods": list(METHODS), "bits": list(BITS),
        "frames": len(frames),
        "gate_outcome": ("EVALUABLE" if candidate_pool_gate
                         else "INVALID_INPUT_DEGENERATE_CANDIDATE_POOL"),
        "provenance_drift": {
            "legacy_g4a_nonzero_closure_frames": sum(
                int(row["closure_cardinality"]) > 0
                for row in g4a_reference.values()),
            "replayed_g4a_nonzero_closure_frames": sum(
                value > 0 for value in replayed_g4a_closures),
            "legacy_g4a_median_closure": float(np.median([
                int(row["closure_cardinality"])
                for row in g4a_reference.values()])),
            "replayed_g4a_median_closure": float(np.median(
                replayed_g4a_closures)),
        },
        "nondegenerate_frames": nondegenerate_frames,
        "candidate_pool_gate_pass": candidate_pool_gate,
        "candidate_count_distribution": dict(sorted(Counter(
            frame["candidate_count"] for frame in frames).items())),
        "group_distribution": dict(sorted(Counter(
            frame["group"] for frame in frames).items())),
        "summary": summary,
        "containment_violations": containment,
        "power_bracket_violations": bracket_violations,
        "certified_nonoptimal": false_nonoptimal,
        "capability_dependent_frames": len(capability_frames),
        "factor_vs_dense_comparable_frames": len(comparison_frames),
        "factor_payload_wins": factor_wins,
        "factor_margin_vs_min_bits_spearman": ({
            "rho_descriptive_only": float(correlation.statistic),
            "n_correlated_frames": len(correlation_frames),
            "independent_episodes": 5,
        } if correlation is not None else None),
        "safety_gate_pass": false_nonoptimal == 0 and bracket_violations == 0
        and all(value == 0 for method in containment.values() for value in method.values()),
        "decision_gate_pass": candidate_pool_gate and all(
            frame["factor_log_min_bits"] is not None for frame in frames),
        "communication_value_gate_pass": (
            len(comparison_frames) > 0 and factor_wins == len(comparison_frames)),
        "frames_detail": frames, "rows": rows,
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
                      if key not in {"rows", "frames_detail"}},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
