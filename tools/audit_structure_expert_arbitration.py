#!/usr/bin/env python
"""Same-state audit of target-wise arbitration between frozen Students.

This is an information-equivalent development diagnostic.  It uses only
recorded local inputs for expert prediction, evaluates selected structures on
the trace's privileged physical edge values, and does not claim a closed-loop
or deployable safety guarantee.
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

from tools.audit_structure_teacher_trace import (  # noqa: E402
    _infer_observation_slices,
    _per_target_features,
    _replay_teacher_projection,
)
from uav_isac.agents.frozen_structure_student import (  # noqa: E402
    FrozenStructureStudent,
)
from uav_isac.environment.communication import (  # noqa: E402
    InterUAVCommunicationModel,
)
from uav_isac.evaluation.expert_arbitration import (  # noqa: E402
    blend_target_edge_values,
    conservative_target_expert_choice,
    target_support_ceiling,
)
from uav_isac.physical.detection import (  # noqa: E402
    compute_detection_probabilities,
)


def _ordered_unique(values: np.ndarray) -> list[int]:
    return list(dict.fromkeys(int(value) for value in values.reshape(-1)))


def _parse_margins(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("margins must be finite and non-negative")
    return tuple(sorted(set(values)))


def _prediction(
    student: FrozenStructureStudent,
    features: np.ndarray,
    bits_per_dim: int,
) -> tuple[np.ndarray, float]:
    protocol = student.encode_endpoint_protocol(features)
    quantized = InterUAVCommunicationModel.quantize_values_at_bits(
        protocol, int(bits_per_dim))
    return (
        student.predict_from_endpoint_protocol(quantized),
        float(np.mean(np.abs(protocol - quantized))),
    )


def _steady_episode_summary(
    data: dict[str, np.ndarray],
    frame_indices: np.ndarray,
    pair_matrix: np.ndarray,
    *,
    steady_window: int = 20,
) -> dict[str, object]:
    realized = np.asarray(
        data["privileged_d_eff"][frame_indices], dtype=np.float64)
    receiver_d = np.sum(realized * pair_matrix.astype(bool), axis=1)
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    floor = float(np.asarray(data["qos_floor"]).reshape(-1)[0])
    pd = compute_detection_probabilities(
        np.max(receiver_d, axis=1), p_fa)
    seeds = np.asarray(data["seed"])[frame_indices]
    frames = np.asarray(data["frame"])[frame_indices]
    rows = []
    for seed in _ordered_unique(seeds):
        selected = np.flatnonzero(seeds == seed)
        final_frame = int(np.max(frames[selected]))
        steady = selected[frames[selected] >= final_frame - int(steady_window) + 1]
        per_target = np.mean(pd[steady], axis=0)
        ordered = np.sort(per_target)
        rows.append({
            "seed": int(seed),
            "steady": float(np.mean(per_target)),
            "weak3": float(np.mean(ordered[:min(3, ordered.size)])),
            "worst": float(ordered[0]),
            "qos_feasible": bool(np.all(per_target >= floor)),
        })
    return {
        "episode_count": len(rows),
        "steady": float(np.mean([row["steady"] for row in rows])),
        "weak3": float(np.mean([row["weak3"] for row in rows])),
        "mean_worst": float(np.mean([row["worst"] for row in rows])),
        "qos_feasible_rate": float(np.mean([
            row["qos_feasible"] for row in rows])),
        "episodes": rows,
    }


def audit(
    trace_path: Path,
    primary_checkpoint: Path,
    alternate_checkpoint: Path,
    *,
    bits_per_dim: int,
    seed_limit: int,
    relative_margins: tuple[float, ...],
) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    resolved = np.asarray(data["p0_resolved"], dtype=bool)
    seed_order = _ordered_unique(np.asarray(data["seed"]))
    selected_seeds = seed_order[:max(1, int(seed_limit))]
    frame_indices = np.flatnonzero(
        resolved & np.isin(np.asarray(data["seed"]), selected_seeds))
    if not len(frame_indices):
        raise ValueError("selected development seeds have no resolve frames")
    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])
    slices = _infer_observation_slices(data["local_obs"].shape[-1], K, Q)
    features = _per_target_features(
        data, "local_obs", frame_indices, slices)
    primary = FrozenStructureStudent(primary_checkpoint)
    alternate = FrozenStructureStudent(alternate_checkpoint)
    primary_edge, primary_qmae = _prediction(
        primary, features, bits_per_dim)
    alternate_edge, alternate_qmae = _prediction(
        alternate, features, bits_per_dim)
    candidate = np.asarray(
        data["privileged_candidate"][frame_indices], dtype=bool)
    pair_limit = int(np.asarray(data["target_pair_limit"]).reshape(-1)[0])
    primary_score = target_support_ceiling(
        primary_edge, candidate, pair_limit=pair_limit)
    alternate_score = target_support_ceiling(
        alternate_edge, candidate, pair_limit=pair_limit)

    variants: dict[str, object] = {}
    edge_variants = {
        "primary": primary_edge,
        "alternate": alternate_edge,
    }
    choices: dict[str, np.ndarray] = {}
    for margin in relative_margins:
        name = f"targetwise_margin_{margin:g}"
        choice = conservative_target_expert_choice(
            primary_score,
            alternate_score,
            relative_margin=float(margin),
        )
        choices[name] = choice
        edge_variants[name] = blend_target_edge_values(
            primary_edge, alternate_edge, choice)

    for name, edge_values in edge_variants.items():
        replay = _replay_teacher_projection(
            data,
            resolved,
            frame_indices=frame_indices,
            predicted_d_eff=edge_values,
            return_pairs=True,
        )
        pairs = np.asarray(replay.pop("_pair_matrix"), dtype=np.uint8)
        item: dict[str, object] = {
            "same_state_frame_metrics": replay,
            "steady_episode_metrics": _steady_episode_summary(
                data, frame_indices, pairs),
        }
        if name in choices:
            item["alternate_target_fraction"] = float(np.mean(choices[name]))
            item["expert_switches_per_resolve"] = float(np.mean(np.sum(
                choices[name], axis=-1)))
        variants[name] = item

    endpoint_bits = {
        "primary": int(Q * 2 * primary.model.endpoint_dim * bits_per_dim),
        "alternate": int(Q * 2 * alternate.model.endpoint_dim * bits_per_dim),
    }
    return {
        "schema_version": 1,
        "scope": (
            "selection-seed same-state information-equivalent diagnostic; "
            "not closed-loop and not a deployment certificate"
        ),
        "trace": str(trace_path),
        "selected_seeds": selected_seeds,
        "resolve_frames": int(len(frame_indices)),
        "bits_per_dim": int(bits_per_dim),
        "primary_checkpoint": str(primary_checkpoint),
        "alternate_checkpoint": str(alternate_checkpoint),
        "quantization_mae": {
            "primary": primary_qmae,
            "alternate": alternate_qmae,
        },
        "wire_cost_per_sender_per_resolve_bits": {
            **endpoint_bits,
            "dual_expert": int(endpoint_bits["primary"] + endpoint_bits["alternate"]),
            "note": "payload only; headers and retransmission are excluded",
        },
        "arbitration_rule": (
            "alternate target iff its relaxed receiver-local Top-B predicted "
            "support exceeds primary by the stated relative margin; final "
            "structure is re-solved under role/owner/capacity constraints"
        ),
        "variants": variants,
        "fresh_test_consumed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--primary-checkpoint", required=True, type=Path)
    parser.add_argument("--alternate-checkpoint", required=True, type=Path)
    parser.add_argument("--bits-per-dim", type=int, default=8)
    parser.add_argument("--seed-limit", type=int, default=10)
    parser.add_argument("--relative-margins", default="0,0.05,0.1,0.25")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(
        args.trace,
        args.primary_checkpoint,
        args.alternate_checkpoint,
        bits_per_dim=max(1, int(args.bits_per_dim)),
        seed_limit=max(1, int(args.seed_limit)),
        relative_margins=_parse_margins(args.relative_margins),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "selected_seeds": result["selected_seeds"],
        "variants": {
            name: values["steady_episode_metrics"]
            for name, values in result["variants"].items()
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
