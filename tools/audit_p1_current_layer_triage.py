#!/usr/bin/env python
"""P1 natural-trajectory D/L1/L2/III/U layer re-triage."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from uav_isac.coordination.capability import (  # noqa: E402
    capability_gauge_pwl_lp,
    minimum_total_power_pwl_lp,
)
from uav_isac.coordination.maxmin_power import (  # noqa: E402
    fixed_owner_gain_matrix,
)
from uav_isac.coordination.minimum_intervention_repair import (  # noqa: E402
    solve_minimum_intervention_task_repair_milp,
)
from uav_isac.coordination.pwl_pd import (  # noqa: E402
    saturating_chord_lower_bound,
)
from uav_isac.coordination.scale_capability import (  # noqa: E402
    characterize_bistatic_scale_capability,
    task_detection_metrics,
)
from uav_isac.evaluation.layer_provenance import (  # noqa: E402
    audit_layer_trace_provenance,
    canonical_json_sha256,
)
from uav_isac.physical.detection import (  # noqa: E402
    compute_detection_probabilities,
    minimum_deflection_for_detection_probability,
)
from uav_isac.utils.provenance import sha256_file  # noqa: E402


def _selected_edges(mask: np.ndarray) -> list[tuple[int, int, int]]:
    return [tuple(map(int, edge)) for edge in np.argwhere(mask)]


def _is_feasible(metrics: dict[str, float], tolerance: float = 1.0e-9) -> bool:
    return float(metrics["feasibility_ratio"]) <= 1.0 + tolerance


def audit(
    trace_path: Path,
    config_path: Path,
    *,
    stride: int = 30,
    time_limit_s: float = 30.0,
) -> dict[str, object]:
    if stride < 1:
        raise ValueError("stride must be positive")
    p0 = audit_layer_trace_provenance(trace_path)
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {name: loaded[name] for name in loaded.files}
    provenance = json.loads(str(np.asarray(data["provenance_json"]).reshape(-1)[0]))
    cfg = load_config(str(config_path))
    resolved = asdict(cfg) if is_dataclass(cfg) else cfg
    if canonical_json_sha256(resolved) != provenance["config_sha256"]:
        raise ValueError("supplied config does not match trace config hash")

    floors = tuple(cfg.marl.task_constrained_qos_floors)
    p_fa = float(cfg.detection.P_FA)
    d_min = float(minimum_deflection_for_detection_probability(
        np.asarray([float(floors[0])]), p_fa)[0])
    d_saturation = float(minimum_deflection_for_detection_probability(
        np.asarray([0.999]), p_fa)[0])
    indices = [
        index for index, frame in enumerate(np.asarray(data["frame"]))
        if int(frame) % stride == 0
    ]
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for index in indices:
        coefficient = np.asarray(
            data["per_watt_coefficient"][index], dtype=np.float64)
        budget = np.asarray(
            data["frame_sensing_budget_w"][index], dtype=np.float64)
        selected_mask = np.asarray(
            data["deployed_selected"][index], dtype=bool)
        selected = _selected_edges(selected_mask)
        deployed_power = np.asarray(
            data["deployed_sensing_power_w"][index], dtype=np.float64)
        deployed_pd = np.asarray(data["physical_pd"][index], dtype=np.float64)
        deployed_metrics = task_detection_metrics(deployed_pd, floors)
        relaxed = characterize_bistatic_scale_capability(coefficient, budget)
        relaxed_pd = compute_detection_probabilities(
            relaxed.relaxed_target_ceiling, p_fa)
        relaxed_metrics = task_detection_metrics(relaxed_pd, floors)
        d_max = max(
            2.0 * float(np.max(relaxed.relaxed_target_ceiling)),
            d_saturation + 1.0,
            d_min + 1.0,
        )
        slopes, intercepts, _ = saturating_chord_lower_bound(
            p_fa, d_min, d_max, 1.0e-3)

        fixed_gain = None
        gamma_fixed = float("inf")
        minimum_fixed_power = None
        try:
            fixed_gain, _ = fixed_owner_gain_matrix(coefficient, selected)
            gamma_value = capability_gauge_pwl_lp(
                fixed_gain, budget, p_fa, floors,
                slopes, intercepts, d_min)
            if gamma_value is not None:
                gamma_fixed = float(gamma_value)
            minimum = minimum_total_power_pwl_lp(
                fixed_gain, budget, p_fa, floors,
                slopes, intercepts, d_min)
            if minimum is not None:
                minimum_fixed_power = float(minimum[0])
        except ValueError:
            # Missing/ambiguous fixed owner is a genuine L1 failure; the joint
            # layer may still repair it.  It is not a geometry certificate.
            pass

        repair = None
        gamma_joint_witness = None
        joint_metrics = None
        joint_conservative_metrics = None
        joint_power_violation_w = None
        if _is_feasible(deployed_metrics):
            layer = "D"
        elif gamma_fixed <= 1.0 + 2.0e-5:
            layer = "L1"
        else:
            role = np.asarray(data["deployed_role"][index], dtype=np.int8)
            repair = solve_minimum_intervention_task_repair_milp(
                coefficient, budget, selected_mask,
                role == 0, role == 1,
                np.asarray(
                    data["deployed_receiver_owner"][index], dtype=np.int64),
                p_fa=p_fa,
                task_floors=floors,
                target_pair_limit=int(
                    np.asarray(data["target_pair_limit"]).reshape(-1)[0]),
                reports_per_receiver=int(
                    np.asarray(data["reports_per_receiver"]).reshape(-1)[0]),
                pwl_epsilon=1.0e-3,
                time_limit_s=float(time_limit_s),
            )
            if repair.feasible:
                joint_metrics = task_detection_metrics(
                    repair.target_pd, floors)
                joint_conservative_metrics = task_detection_metrics(
                    repair.conservative_pd, floors)
                joint_power_violation_w = float(np.max(
                    np.sum(repair.sensing_power_w, axis=1) - budget))
                if (
                    not _is_feasible(joint_metrics, tolerance=1.0e-7)
                    or not _is_feasible(
                        joint_conservative_metrics, tolerance=1.0e-7)
                    or joint_power_violation_w > 1.0e-8
                ):
                    raise RuntimeError("joint repair witness failed exact verification")
                joint_gain, _ = fixed_owner_gain_matrix(
                    coefficient, repair.selected_set)
                joint_gamma_value = capability_gauge_pwl_lp(
                    joint_gain, budget, p_fa, floors,
                    slopes, intercepts, d_min)
                if joint_gamma_value is not None:
                    gamma_joint_witness = float(joint_gamma_value)
                layer = "L2"
            elif not _is_feasible(relaxed_metrics):
                layer = "III"
            else:
                layer = "U"

        rows.append({
            "seed": int(data["seed"][index]),
            "episode": int(data["episode"][index]),
            "frame": int(data["frame"][index]),
            "layer": layer,
            "deployed": deployed_metrics,
            "gamma_fixed": gamma_fixed if np.isfinite(gamma_fixed) else None,
            "gamma_joint_witness": gamma_joint_witness,
            "delta_structure_witness": (
                None if gamma_joint_witness is None or not np.isfinite(gamma_fixed)
                else float(gamma_fixed - gamma_joint_witness)
            ),
            "deployed_sensing_power_w": float(np.sum(deployed_power)),
            "minimum_fixed_sensing_power_w": minimum_fixed_power,
            "delta_power_l1_w": (
                None if minimum_fixed_power is None
                else float(np.sum(deployed_power) - minimum_fixed_power)
            ),
            "relaxed_ceiling": relaxed_metrics,
            "joint_repair_feasible": (
                None if repair is None else bool(repair.feasible)),
            "joint_repair_closure": (
                None if repair is None or not repair.feasible
                else int(repair.closure_cardinality)),
            "joint_witness": joint_metrics,
            "joint_conservative_witness": joint_conservative_metrics,
            "joint_power_violation_w": joint_power_violation_w,
        })

    counts = Counter(str(row["layer"]) for row in rows)
    total = len(rows)
    fractions = {
        layer: float(counts.get(layer, 0) / total)
        for layer in ("D", "L1", "L2", "III", "U")
    }
    power_deltas = [
        float(row["delta_power_l1_w"]) for row in rows
        if row["delta_power_l1_w"] is not None
    ]
    structure_deltas = [
        float(row["delta_structure_witness"]) for row in rows
        if row["delta_structure_witness"] is not None
    ]
    return {
        "gate": "P1_CURRENT_LAYER_RETRIAGE",
        "scope": "exposed_development_natural_current_policy_systematic_frames",
        "audit_code_sha256": sha256_file(Path(__file__)),
        "p0_provenance": p0,
        "trace": str(trace_path),
        "config": str(config_path),
        "sampling": {
            "rule": "frame modulo stride equals zero; no outcome/difficulty filtering",
            "stride": int(stride),
            "frames": total,
            "episodes": len(set(int(row["episode"]) for row in rows)),
        },
        "task_floors": {
            "rho_min": float(floors[0]),
            "rho_tail": float(floors[1]),
            "rho_avg": float(floors[2]),
            "k": int(floors[3]),
        },
        "layer_counts": {
            layer: int(counts.get(layer, 0))
            for layer in ("D", "L1", "L2", "III", "U")
        },
        "layer_fractions": fractions,
        "delta_power_l1_w": {
            "count": len(power_deltas),
            "median": float(np.median(power_deltas)) if power_deltas else None,
            "mean": float(np.mean(power_deltas)) if power_deltas else None,
        },
        "delta_structure_witness": {
            "count": len(structure_deltas),
            "median": float(np.median(structure_deltas)) if structure_deltas else None,
            "meaning": (
                "gamma_fixed minus gamma of the lex repair witness; not a "
                "global joint-gauge optimum"),
        },
        "m4d_route": (
            "RESUME_ONLY_ON_OBSERVED_L2"
            if counts.get("L2", 0) > 0
            else "CLOSE_M4D_L2_RARE_EMERGENCY_LAYER"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--time-limit-s", type=float, default=30.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(
        args.trace, args.config, stride=args.stride,
        time_limit_s=args.time_limit_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(
        {key: value for key, value in result.items() if key != "rows"},
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
