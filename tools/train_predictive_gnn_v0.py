"""Train/evaluate the one-frame-ahead shadow GNN with seed-disjoint splits."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as functional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_isac.prediction.certified_gnn import (  # noqa: E402
    CertifiedBipartiteGNN,
    PREDICTIVE_CHECKPOINT_SCHEMA,
    append_causal_feature_residual,
    augment_temporal_protocol_features,
    build_endpoint_features,
    decode_candidate_pool,
)
from uav_isac.prediction.constrained_objective import (  # noqa: E402
    constrained_detection_objective,
    detection_probability_from_deflection,
    project_power_to_row_budget,
)
from uav_isac.prediction.gradient_surgery import (  # noqa: E402
    physical_anchor_pcgrad,
)
from uav_isac.utils.checkpoint_loading import safe_torch_load  # noqa: E402


def transition_indices(data: np.lib.npyio.NpzFile) -> np.ndarray:
    return np.asarray([
        index for index in range(len(data["frame"]) - 1)
        if data["seed"][index] == data["seed"][index + 1]
        and data["frame"][index + 1] == data["frame"][index] + 1
    ], dtype=np.int64)


def exact_labels(
    data: np.lib.npyio.NpzFile, next_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    _, viewers, nodes, targets = data["visible"].shape
    owner = np.full((len(next_indices), targets), -1, dtype=np.int64)
    transmitters = np.zeros(
        (len(next_indices), nodes, targets), dtype=np.float32)
    for sample, frame_index in enumerate(next_indices):
        edges = data["edge_index"][frame_index][data["edge_mask"][frame_index]]
        for tx, rx, target in edges:
            previous = owner[sample, target]
            if previous not in (-1, rx):
                raise ValueError("teacher has more than one owner for a target")
            owner[sample, target] = rx
            transmitters[sample, tx, target] = 1.0
    if np.any(owner < 0):
        raise ValueError("teacher lacks full target coverage")
    return owner, transmitters


def _single_frame_features(
    data: np.lib.npyio.NpzFile, indices: np.ndarray
) -> np.ndarray:
    frames, viewers, nodes, targets, _ = data["endpoint_position"][indices].shape
    features = build_endpoint_features(
        data["endpoint_position"][indices].reshape(-1, nodes, targets, 2),
        data["endpoint_velocity"][indices].reshape(-1, nodes, targets, 2),
        data["target_position"][indices].reshape(-1, targets, 3),
        data["target_velocity"][indices].reshape(-1, targets, 3),
        data["target_position_uncertainty"][indices].reshape(-1, targets),
        data["target_velocity_uncertainty"][indices].reshape(-1, targets),
        data["visible"][indices].reshape(-1, nodes, targets),
        distance_scale_m=1000.0,
        velocity_scale_mps=50.0,
    )
    features = features.reshape(frames, viewers, nodes, targets, -1)
    features = augment_temporal_protocol_features(
        features,
        data["edge_index"][indices],
        data["edge_mask"][indices],
        data["frame"][indices],
        data["nominal_gain"][indices],
        data["lower_gain"][indices],
        data["upper_gain"][indices],
        data["power"][indices],
        data["dual_price"][indices],
    )
    return features


def feature_tensor(
    data: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    *,
    two_frame_residual: bool = False,
) -> torch.Tensor:
    """Build causal current-state features with an optional history residual.

    The residual is zero with a cleared history flag at each seed's first
    frame, so no state can leak across episodes or from a future label.
    """
    current = _single_frame_features(data, indices)
    if not two_frame_residual:
        return torch.from_numpy(current)
    previous_indices = np.asarray(indices, dtype=np.int64).copy()
    history_valid = np.zeros(len(previous_indices), dtype=np.float32)
    for sample, index in enumerate(previous_indices):
        if (
            index > 0
            and data["seed"][index - 1] == data["seed"][index]
            and data["frame"][index - 1] + 1 == data["frame"][index]
        ):
            previous_indices[sample] = index - 1
            history_valid[sample] = 1.0
    previous = _single_frame_features(data, previous_indices)
    return torch.from_numpy(append_causal_feature_residual(
        current, previous, history_valid))


def prediction_loss(
    model: CertifiedBipartiteGNN,
    features: torch.Tensor,
    visible: torch.Tensor,
    owner: torch.Tensor,
    transmitters: torch.Tensor,
    dual: torch.Tensor,
    robust_gain: torch.Tensor,
    teacher_power: torch.Tensor,
    previous_power: torch.Tensor,
    actual_detection: torch.Tensor,
    previous_detection: torch.Tensor,
    *,
    p_fa: float = 0.001,
    qos_floor: float = 0.60,
    joint_weight: float = 1.0,
    qos_dual_weight: float = 0.0,
    detection_forecast_weight: float = 0.05,
    detection_delta_weight: float = 0.02,
    objective_group: str = "joint",
) -> tuple[torch.Tensor, torch.Tensor]:
    samples, viewers, nodes, targets, feature_dim = features.shape
    flat_features = features.reshape(-1, nodes, targets, feature_dim)
    flat_visible = visible.reshape(-1, nodes, targets)
    prediction = model(flat_features, flat_visible)
    owner_repeated = owner[:, None, :].expand(-1, viewers, -1).reshape(-1, targets)
    owner_loss = functional.cross_entropy(
        prediction.owner_logits.permute(0, 2, 1).reshape(-1, nodes),
        owner_repeated.reshape(-1),
    )
    conditional = model.conditional_transmitter_logits(
        prediction, owner_repeated)
    tx_repeated = transmitters[:, None].expand(
        -1, viewers, -1, -1).reshape(-1, nodes, targets)
    not_receiver = torch.ones_like(flat_visible)
    not_receiver.scatter_(1, owner_repeated[:, None, :], False)
    tx_mask = flat_visible & not_receiver
    tx_loss = functional.binary_cross_entropy_with_logits(
        conditional[tx_mask], tx_repeated[tx_mask])
    dual_target = dual.reshape(-1, targets)
    dual_loss = functional.smooth_l1_loss(
        torch.log1p(prediction.dual_warm_start),
        torch.log1p(torch.clamp(dual_target, min=0.0)),
    )
    if actual_detection.shape != (samples, targets):
        raise ValueError("actual_detection must have shape (B,Q)")
    if previous_detection.shape != actual_detection.shape:
        raise ValueError("previous_detection must match actual_detection")
    detection_target = actual_detection[:, None, :].expand(
        -1, viewers, -1).reshape(-1, targets)
    detection_forecast_loss = functional.smooth_l1_loss(
        torch.sigmoid(prediction.detection_logits), detection_target)
    delta_target = torch.clamp(
        actual_detection - previous_detection, min=-1.0, max=1.0)
    delta_target = delta_target[:, None, :].expand(
        -1, viewers, -1).reshape(-1, targets)
    detection_delta_loss = functional.smooth_l1_loss(
        prediction.detection_delta, delta_target)
    if robust_gain.shape != (samples, nodes, targets):
        raise ValueError("robust_gain must be the executed joint (B,K,Q) view")
    if teacher_power.shape != robust_gain.shape or previous_power.shape != robust_gain.shape:
        raise ValueError("joint power labels must match robust_gain")
    if viewers != nodes:
        raise ValueError("executor-row stitching requires one viewer per node")
    viewer_power_logits = prediction.power_logits.reshape(
        samples, viewers, nodes, targets)
    executor = torch.arange(nodes, device=viewer_power_logits.device)
    stitched_logits = viewer_power_logits[:, executor, executor, :]
    stitched_risk = prediction.risk_radius.reshape(
        samples, viewers, targets).mean(dim=1)
    stitched_prediction = replace(
        prediction,
        power_logits=stitched_logits,
        risk_radius=stitched_risk,
    )
    sensing_budget = torch.sum(teacher_power, dim=-1)
    teacher_deflection = torch.sum(robust_gain * teacher_power, dim=1)
    teacher_pd = detection_probability_from_deflection(
        teacher_deflection, p_fa=float(p_fa))
    constructive_feasible = torch.all(
        teacher_pd >= float(qos_floor), dim=-1, keepdim=True)
    qos_constraint_mask = constructive_feasible.expand(-1, targets)
    joint = constrained_detection_objective(
        stitched_prediction,
        robust_gain,
        sensing_budget,
        robust_gain > 0.0,
        teacher_power,
        previous_power,
        p_fa=float(p_fa),
        qos_floor=float(qos_floor),
        qos_constraint_mask=qos_constraint_mask,
        qos_dual_weight=float(qos_dual_weight),
    )
    structure_total = owner_loss + tx_loss
    physical_total = (
        0.1 * dual_loss
        + float(detection_forecast_weight) * detection_forecast_loss
        + float(detection_delta_weight) * detection_delta_loss
        + float(joint_weight) * joint.total
    )
    if objective_group == "structure":
        total = structure_total
    elif objective_group == "physical":
        total = physical_total
    elif objective_group == "joint":
        total = structure_total + physical_total
    else:
        raise ValueError("objective_group must be joint, structure, or physical")
    return total, joint.qos_residual.detach()


def evaluate(
    model: CertifiedBipartiteGNN,
    data: np.lib.npyio.NpzFile,
    input_indices: np.ndarray,
    features: torch.Tensor,
    owner_beam: int,
    transmitters_per_owner: int,
    p_fa: float = 0.001,
    qos_floor: float = 0.60,
    robust_mix: float = 0.25,
) -> dict[str, float | int]:
    owner_hits = 0
    owner_total = 0
    edge_hits = 0
    edge_total = 0
    complete = 0
    candidate_total = 0
    persistence_hits = 0
    persistence_complete = 0
    union_hits = 0
    union_complete = 0
    union_candidate_total = 0
    boundary_edge_hits = 0
    boundary_edge_total = 0
    boundary_complete = 0
    boundary_frames = 0
    regular_edge_hits = 0
    regular_edge_total = 0
    regular_complete = 0
    regular_frames = 0
    robust_worst_pd: list[float] = []
    robust_qos_complete: list[float] = []
    power_share_error: list[float] = []
    budget_violation: list[float] = []
    detection_forecast_error: list[float] = []
    detection_delta_error: list[float] = []
    model.eval()
    with torch.inference_mode():
        for sample, input_index in enumerate(input_indices):
            visible_np = data["visible"][input_index]
            prediction = model(
                features[sample], torch.from_numpy(visible_np))
            next_index = input_index + 1
            next_power = torch.from_numpy(
                data["executed_power"][next_index].astype(np.float32))
            budget = torch.sum(next_power, dim=-1)
            robust_gain = torch.from_numpy((
                (1.0 - float(robust_mix))
                * data["executed_nominal_gain"][next_index]
                + float(robust_mix)
                * data["executed_robust_gain"][next_index]
            ).astype(np.float32))
            viewer_power_logits = prediction.power_logits
            if viewer_power_logits.shape[0] != viewer_power_logits.shape[1]:
                raise ValueError("evaluation requires one viewer per executor")
            executor = torch.arange(viewer_power_logits.shape[1])
            stitched_logits = viewer_power_logits[executor, executor, :][None]
            projected_power = project_power_to_row_budget(
                stitched_logits, budget[None], (robust_gain > 0.0)[None],
            )[0]
            robust_deflection = torch.sum(
                robust_gain * projected_power, dim=0, keepdim=True)
            robust_pd = detection_probability_from_deflection(
                robust_deflection, p_fa=float(p_fa))
            robust_worst_pd.extend(
                float(value) for value in torch.min(robust_pd, dim=-1).values)
            robust_qos_complete.extend(
                float(value) for value in torch.all(
                    robust_pd >= float(qos_floor), dim=-1))
            actual_detection = torch.from_numpy(
                data["actual_detection"][next_index].astype(np.float32))
            previous_detection = torch.from_numpy(
                data["actual_detection"][input_index].astype(np.float32))
            predicted_detection = torch.sigmoid(
                prediction.detection_logits).mean(dim=0)
            predicted_delta = prediction.detection_delta.mean(dim=0)
            detection_forecast_error.append(float(torch.mean(torch.abs(
                predicted_detection - actual_detection))))
            detection_delta_error.append(float(torch.mean(torch.abs(
                predicted_delta - (actual_detection - previous_detection)))))
            scale = torch.clamp(budget.unsqueeze(-1), min=1e-12)
            power_share_error.append(float(torch.mean(torch.abs(
                projected_power / scale - next_power / scale))))
            budget_violation.append(float(torch.max(torch.relu(
                projected_power.sum(dim=-1) - budget))))
            candidates = set(decode_candidate_pool(
                prediction,
                visible_np,
                model=model,
                owner_beam=owner_beam,
                transmitters_per_owner=transmitters_per_owner,
            ))
            teacher = {
                tuple(int(value) for value in edge)
                for edge in data["edge_index"][next_index][
                    data["edge_mask"][next_index]]
            }
            current = {
                tuple(int(value) for value in edge)
                for edge in data["edge_index"][input_index][
                    data["edge_mask"][input_index]]
            }
            union = candidates | current
            predicted_owners = {
                (receiver, target) for _, receiver, target in candidates}
            teacher_owners = {
                (receiver, target) for _, receiver, target in teacher}
            owner_hits += len(predicted_owners & teacher_owners)
            owner_total += len(teacher_owners)
            hits = len(candidates & teacher)
            edge_hits += hits
            edge_total += len(teacher)
            complete += int(hits == len(teacher))
            candidate_total += len(candidates)
            persistent_hit_count = len(current & teacher)
            persistence_hits += persistent_hit_count
            persistence_complete += int(persistent_hit_count == len(teacher))
            union_hit_count = len(union & teacher)
            union_hits += union_hit_count
            union_complete += int(union_hit_count == len(teacher))
            union_candidate_total += len(union)
            is_boundary = int(data["frame"][input_index]) % 5 == 0
            if is_boundary:
                boundary_edge_hits += hits
                boundary_edge_total += len(teacher)
                boundary_complete += int(hits == len(teacher))
                boundary_frames += 1
            else:
                regular_edge_hits += hits
                regular_edge_total += len(teacher)
                regular_complete += int(hits == len(teacher))
                regular_frames += 1
    count = len(input_indices)
    return {
        "transitions": count,
        "owner_recall": owner_hits / max(owner_total, 1),
        "edge_recall": edge_hits / max(edge_total, 1),
        "complete_frame_rate": complete / max(count, 1),
        "mean_candidate_edges": candidate_total / max(count, 1),
        "persistence_edge_recall": persistence_hits / max(edge_total, 1),
        "persistence_complete_frame_rate": persistence_complete / max(count, 1),
        "residual_union_edge_recall": union_hits / max(edge_total, 1),
        "residual_union_complete_frame_rate": union_complete / max(count, 1),
        "residual_union_mean_candidate_edges": (
            union_candidate_total / max(count, 1)),
        "boundary_edge_recall": boundary_edge_hits / max(boundary_edge_total, 1),
        "boundary_complete_frame_rate": boundary_complete / max(boundary_frames, 1),
        "boundary_transitions": boundary_frames,
        "regular_edge_recall": regular_edge_hits / max(regular_edge_total, 1),
        "regular_complete_frame_rate": regular_complete / max(regular_frames, 1),
        "regular_transitions": regular_frames,
        "robust_worst_pd_mean": float(np.mean(robust_worst_pd)),
        "robust_qos_complete_rate": float(np.mean(robust_qos_complete)),
        "power_share_mae": float(np.mean(power_share_error)),
        "native_budget_max_violation_w": float(max(
            budget_violation, default=0.0)),
        "actual_detection_forecast_mae": float(np.mean(
            detection_forecast_error)),
        "actual_detection_delta_mae": float(np.mean(
            detection_delta_error)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--owner-beam", type=int, default=2)
    parser.add_argument("--tx-per-owner", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--p-fa", type=float, default=0.001)
    parser.add_argument("--qos-floor", type=float, default=0.60)
    parser.add_argument("--joint-weight", type=float, default=1.0)
    parser.add_argument(
        "--robust-mix", type=float, default=0.25,
        help="fraction of execution lower gain blended into nominal physics")
    parser.add_argument("--dual-step", type=float, default=2.0)
    parser.add_argument("--dual-max", type=float, default=20.0)
    parser.add_argument("--constraint-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--convergence-window", type=int, default=8)
    parser.add_argument("--convergence-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--detection-forecast-weight", type=float, default=0.05)
    parser.add_argument("--detection-delta-weight", type=float, default=0.02)
    parser.add_argument(
        "--boundary-repeat", type=int, default=4,
        help="training multiplicity for topology-change source frames")
    parser.add_argument(
        "--two-frame-residual", action="store_true",
        help="append the previous-frame feature residual and history flag")
    parser.add_argument(
        "--boundary-power-residual", action="store_true",
        help="allow the boundary branch to alter power logits (ablation)")
    parser.add_argument(
        "--optimization-mode", choices=("joint", "alternating", "pcgrad"),
        default="joint",
        help="alternate structural and physical steps to avoid gradient cancellation")
    parser.add_argument("--load-checkpoint")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.dual_step < 0.0 or args.dual_max < 0.0:
        parser.error("dual step/max must be non-negative")
    if args.constraint_tolerance < 0.0:
        parser.error("constraint tolerance must be non-negative")
    if args.convergence_window < 2 or args.convergence_tolerance < 0.0:
        parser.error("invalid convergence window/tolerance")
    if args.detection_forecast_weight < 0.0 or args.detection_delta_weight < 0.0:
        parser.error("detection auxiliary weights must be non-negative")
    if args.boundary_repeat < 1:
        parser.error("boundary repeat must be positive")

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = np.load(args.dataset)
    indices = transition_indices(data)
    seeds = np.unique(data["seed"][indices])
    if len(seeds) < 3:
        raise ValueError("at least three seeds are required for a disjoint split")
    validation_seeds = set(int(seed) for seed in seeds[-2:])
    train_indices = indices[
        [int(data["seed"][i]) not in validation_seeds for i in indices]]
    validation_indices = indices[
        [int(data["seed"][i]) in validation_seeds for i in indices]]

    train_features = feature_tensor(
        data, train_indices, two_frame_residual=args.two_frame_residual)
    validation_features = feature_tensor(
        data, validation_indices, two_frame_residual=args.two_frame_residual)
    train_owner_np, train_tx_np = exact_labels(data, train_indices + 1)
    train_owner = torch.from_numpy(train_owner_np)
    train_tx = torch.from_numpy(train_tx_np)
    train_visible = torch.from_numpy(data["visible"][train_indices])
    train_dual = torch.from_numpy(
        data["dual_price"][train_indices + 1].astype(np.float32))
    if not 0.0 <= args.robust_mix <= 1.0:
        parser.error("--robust-mix must lie in [0,1]")
    required_native = {
        "executed_nominal_gain", "executed_robust_gain",
        "executed_power", "actual_deflection", "actual_detection",
    }
    missing_native = sorted(required_native.difference(data.files))
    if missing_native:
        raise ValueError(
            "joint constrained training requires v2 native labels: "
            + ", ".join(missing_native))
    train_robust_gain = torch.from_numpy((
        (1.0 - args.robust_mix)
        * data["executed_nominal_gain"][train_indices + 1]
        + args.robust_mix
        * data["executed_robust_gain"][train_indices + 1]
    ).astype(np.float32))
    train_teacher_power = torch.from_numpy(
        data["executed_power"][train_indices + 1].astype(np.float32))
    train_previous_power = torch.from_numpy(
        data["executed_power"][train_indices].astype(np.float32))
    train_actual_detection = torch.from_numpy(
        data["actual_detection"][train_indices + 1].astype(np.float32))
    train_previous_detection = torch.from_numpy(
        data["actual_detection"][train_indices].astype(np.float32))

    model = CertifiedBipartiteGNN(
        feature_dim=int(train_features.shape[-1]),
        hidden_dim=args.hidden,
        message_rounds=args.rounds,
        boundary_feature_index=11 if args.two_frame_residual else None,
        boundary_power_residual=args.boundary_power_residual,
    )
    if args.load_checkpoint:
        checkpoint_payload = safe_torch_load(
            args.load_checkpoint,
            map_location="cpu",
            description="predictive GNN training checkpoint",
            state_dict_keys=("state_dict",),
        )
        incompatible = model.load_state_dict(
            checkpoint_payload["state_dict"], strict=False)
        allowed_missing_prefixes = (
            "power_head.", "detection_head.", "detection_delta_head.",
            "boundary_owner_head.", "boundary_transmitter_head.",
            "boundary_power_head.")
        unsupported_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        if unsupported_missing or incompatible.unexpected_keys:
            raise ValueError(
                "checkpoint is not compatible with the native-constraint model: "
                f"missing={unsupported_missing}, "
                f"unexpected={incompatible.unexpected_keys}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1e-4)
    losses: list[float] = []
    constraint_residuals: list[float] = []
    qos_dual_history: list[float] = []
    gradient_cosines: list[float] = []
    qos_dual = 0.0
    for _ in range(args.epochs):
        model.train()
        repeat = np.where(
            data["frame"][train_indices] % 5 == 0,
            args.boundary_repeat,
            1,
        )
        training_pool = np.repeat(np.arange(len(train_indices)), repeat)
        order = np.random.permutation(training_pool)
        epoch_losses = []
        epoch_residuals = []
        for start in range(0, len(order), args.batch_size):
            batch = torch.from_numpy(order[start:start + args.batch_size])
            batch_args = (
                model, train_features[batch], train_visible[batch],
                train_owner[batch], train_tx[batch], train_dual[batch],
                train_robust_gain[batch], train_teacher_power[batch],
                train_previous_power[batch], train_actual_detection[batch],
                train_previous_detection[batch],
            )
            batch_kwargs = dict(
                p_fa=args.p_fa, qos_floor=args.qos_floor,
                joint_weight=args.joint_weight, qos_dual_weight=qos_dual,
                detection_forecast_weight=args.detection_forecast_weight,
                detection_delta_weight=args.detection_delta_weight,
            )
            grouped_losses = []
            constraint_residual = torch.tensor(0.0)
            if args.optimization_mode == "pcgrad":
                structure_loss, _ = prediction_loss(
                    *batch_args, **batch_kwargs, objective_group="structure")
                physical_loss, constraint_residual = prediction_loss(
                    *batch_args, **batch_kwargs, objective_group="physical")
                optimizer.zero_grad(set_to_none=True)
                gradient_cosines.append(physical_anchor_pcgrad(
                    structure_loss, physical_loss, model.parameters()))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                grouped_losses.extend([
                    float(structure_loss.detach()),
                    float(physical_loss.detach()),
                ])
            else:
                groups = (
                    ("structure", "physical")
                    if args.optimization_mode == "alternating" else ("joint",)
                )
                for group in groups:
                    loss, constraint_residual = prediction_loss(
                        *batch_args, **batch_kwargs, objective_group=group)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    grouped_losses.append(float(loss.detach()))
            epoch_losses.append(float(np.sum(grouped_losses)))
            epoch_residuals.append(float(constraint_residual))
        losses.append(float(np.mean(epoch_losses)))
        mean_residual = float(np.mean(epoch_residuals))
        constraint_residuals.append(mean_residual)
        qos_dual = float(np.clip(
            qos_dual
            + args.dual_step * (mean_residual - args.constraint_tolerance),
            0.0,
            args.dual_max,
        ))
        qos_dual_history.append(qos_dual)

    window = max(2, min(int(args.convergence_window), len(losses)))
    if len(losses) >= window:
        recent_loss = np.asarray(losses[-window:], dtype=np.float64)
        recent_residual = np.asarray(
            constraint_residuals[-window:], dtype=np.float64)
        loss_relative_span = float(
            np.ptp(recent_loss) / max(abs(float(np.mean(recent_loss))), 1e-12))
        residual_span = float(np.ptp(recent_residual))
    else:
        loss_relative_span = float("inf")
        residual_span = float("inf")
    optimizer_converged = bool(
        loss_relative_span <= args.convergence_tolerance
        and residual_span <= args.convergence_tolerance
        and constraint_residuals[-1] <= args.constraint_tolerance
    )

    train_metrics = evaluate(
        model, data, train_indices, train_features,
        args.owner_beam, args.tx_per_owner, args.p_fa, args.qos_floor,
        args.robust_mix)
    validation_metrics = evaluate(
        model, data, validation_indices, validation_features,
        args.owner_beam, args.tx_per_owner, args.p_fa, args.qos_floor,
        args.robust_mix)
    payload = {
        "schema_version": PREDICTIVE_CHECKPOINT_SCHEMA,
        "dataset": args.dataset,
        "train_seeds": [int(seed) for seed in seeds[:-2]],
        "validation_seeds": sorted(validation_seeds),
        "epochs": args.epochs,
        "loss_initial": losses[0] if losses else None,
        "loss_final": losses[-1] if losses else None,
        "owner_beam": args.owner_beam,
        "transmitters_per_owner": args.tx_per_owner,
        "p_fa": args.p_fa,
        "qos_floor": args.qos_floor,
        "joint_weight": args.joint_weight,
        "robust_mix": args.robust_mix,
        "detection_forecast_weight": args.detection_forecast_weight,
        "detection_delta_weight": args.detection_delta_weight,
        "boundary_repeat": args.boundary_repeat,
        "two_frame_residual": args.two_frame_residual,
        "boundary_power_residual": args.boundary_power_residual,
        "optimization_mode": args.optimization_mode,
        "gradient_conflict_rate": (
            float(np.mean(np.asarray(gradient_cosines) < 0.0))
            if gradient_cosines else None),
        "gradient_cosine_mean": (
            float(np.mean(gradient_cosines)) if gradient_cosines else None),
        "projected_qos_dual_final": qos_dual,
        "constraint_residual_initial": constraint_residuals[0],
        "constraint_residual_final": constraint_residuals[-1],
        "convergence": {
            "scope": "joint training objective and native QoS residual",
            "window": window,
            "loss_relative_span": loss_relative_span,
            "constraint_residual_span": residual_span,
            "tolerance": args.convergence_tolerance,
            "constraint_tolerance": args.constraint_tolerance,
            "optimizer_converged": optimizer_converged,
        },
        "train": train_metrics,
        "validation": validation_metrics,
        "production_eligible": False,
    }
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": payload["schema_version"],
        "feature_dim": model.feature_dim,
        "hidden_dim": model.hidden_dim,
        "message_rounds": model.message_rounds,
        "boundary_feature_index": model.boundary_feature_index,
        "boundary_power_residual": model.boundary_power_residual,
        "state_dict": model.state_dict(),
        "metrics": payload,
    }, checkpoint)
    metrics = Path(args.metrics)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    if not args.quiet:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
