#!/usr/bin/env python
"""Train and audit the Gate-C1 finite-round factor-graph coordinator."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uav_isac.agents.frozen_structure_student import (  # noqa: E402
    FrozenStructureStudent,
    build_structure_student_features,
)
from uav_isac.coordination.factor_graph_coordinator import (  # noqa: E402
    FiniteRoundFactorGraphCoordinator,
    assert_hard_structure_invariants,
    decode_lightweight_feasible,
    factor_graph_edge_features,
)
from uav_isac.coordination.finite_round_hyperedge import (  # noqa: E402
    solve_replicated_candidate_graph,
)
from uav_isac.environment.observation_slices import (  # noqa: E402
    ObservationSlices,
)
from uav_isac.evaluation.local_candidate_audit import (  # noqa: E402
    episode_detection_summary,
    realized_pd_history,
    receiver_owner_from_pairs,
)


def _infer_observation_slices(obs_dim: int, K: int, Q: int) -> ObservationSlices:
    base_without_tokens = 8 + 9 * Q + 8 * Q + 3 + 8 * (K - 1) + Q + 16
    token_count = (K - 1) * Q
    numerator = obs_dim - base_without_tokens - token_count
    if token_count <= 0 or numerator % token_count:
        raise ValueError("cannot infer target-token observation layout")
    return ObservationSlices.from_config(
        K=K,
        Q=Q,
        use_p0=False,
        use_rel_features=True,
        use_comm_tokens=True,
        comm_token_dim=numerator // token_count,
        comm_tokens_per_sender=Q,
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _subset_trace(
    raw: dict[str, np.ndarray],
    seeds: np.ndarray,
) -> dict[str, np.ndarray]:
    keep = np.isin(raw["seed"], seeds)
    frames = len(raw["seed"])
    return {
        key: (value[keep] if value.ndim and value.shape[0] == frames else value)
        for key, value in raw.items()
    }


def _edge_values(
    data: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    checkpoint: Path,
) -> np.ndarray:
    K = int(data["num_uavs"][0])
    Q = int(data["num_targets"][0])
    slices = _infer_observation_slices(data["local_obs"].shape[-1], K, Q)
    features = build_structure_student_features(
        data["local_obs"],
        slices,
        outgoing_message=data["outgoing_message"],
        outgoing_token_mask=data["outgoing_token_mask"],
        outgoing_rate=data["outgoing_rate"],
        comm_fraction=data["comm_fraction"],
        sensing_weights=data["sensing_weights"],
        rate_scale=max(float(np.max(data["outgoing_rate"])), 1.0),
        neighbor_subset_mask=candidate["local_neighbors"].astype(bool),
    )
    return FrozenStructureStudent(checkpoint).predict(features).astype(np.float32)


def _replicated_teacher_sequence(
    edge_value: np.ndarray,
    candidate_mask: np.ndarray,
    data: dict[str, np.ndarray],
    cache: Path,
) -> np.ndarray:
    if cache.exists():
        cached = _load_npz(cache)
        pair = cached["teacher_pair"].astype(bool)
        if pair.shape != candidate_mask.shape:
            raise ValueError(f"teacher cache shape mismatch: {cache}")
        return pair
    frames, K, _, Q = candidate_mask.shape
    pair = np.zeros_like(candidate_mask, dtype=bool)
    current = np.zeros((K, K, Q), dtype=bool)
    previous_episode: int | None = None
    resolve_indices = np.flatnonzero(data["p0_resolved"])
    completed = 0
    for frame in range(frames):
        episode = int(data["episode"][frame])
        if episode != previous_episode:
            current.fill(False)
            previous_episode = episode
        if bool(data["p0_resolved"][frame]):
            current.fill(False)
            if np.any(candidate_mask[frame]):
                selected = solve_replicated_candidate_graph(
                    edge_value[frame],
                    candidate_mask[frame],
                    target_pair_limit=int(data["target_pair_limit"][0]),
                    reports_per_receiver=int(data["reports_per_receiver"][0]),
                )
                for edge in selected:
                    current[edge] = True
            completed += 1
            if completed % 25 == 0:
                print(
                    f"teacher {completed}/{len(resolve_indices)} resolve frames",
                    flush=True,
                )
        pair[frame] = current
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        teacher_pair=pair.astype(np.uint8),
        episode=data["episode"],
        seed=data["seed"],
        frame=data["frame"],
    )
    return pair


def _targets_from_pair(pair: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if pair.ndim != 4:
        raise ValueError("pair must have shape (F,K,K,Q)")
    frames, K, _, Q = pair.shape
    role = np.zeros((frames, K), dtype=np.int64)
    role[np.any(pair, axis=(2, 3))] = 1
    role[np.any(pair, axis=(1, 3))] = 2
    owner = receiver_owner_from_pairs(pair)
    owner = np.where(owner >= 0, owner, K).astype(np.int64)
    return role, owner


def _equivalent_owner_sets(
    teacher_pair: np.ndarray,
    candidate_mask: np.ndarray,
    realized_d_eff: np.ndarray,
    *,
    target_pair_limit: int,
    evidence_ratio: float,
) -> np.ndarray:
    """Return set-valued owner labels based on retained physical evidence."""
    pair = np.asarray(teacher_pair, dtype=bool)
    candidate = np.asarray(candidate_mask, dtype=bool)
    d_eff = np.asarray(realized_d_eff, dtype=np.float64)
    if not (pair.shape == candidate.shape == d_eff.shape):
        raise ValueError("teacher, candidate and d_eff shapes must match")
    frames, K, _, Q = pair.shape
    valid = np.zeros((frames, Q, K + 1), dtype=bool)
    teacher_evidence = np.sum(d_eff * pair, axis=(1, 2))
    limit = max(1, int(target_pair_limit))
    for frame in range(frames):
        for target in range(Q):
            reference = float(teacher_evidence[frame, target])
            if reference <= 1.0e-12:
                valid[frame, target, K] = True
                continue
            for receiver in range(K):
                evidence = d_eff[frame, :, receiver, target][
                    candidate[frame, :, receiver, target]]
                retained = np.sort(np.maximum(evidence, 0.0))[-limit:]
                valid[frame, target, receiver] = (
                    float(np.sum(retained))
                    >= float(evidence_ratio) * reference)
            # The actual teacher owner is always admitted despite rounding.
            teacher_receiver = np.flatnonzero(np.any(
                pair[frame, :, :, target], axis=0))
            if len(teacher_receiver):
                valid[frame, target, int(teacher_receiver[0])] = True
    return valid


def _batch_loss(
    model: FiniteRoundFactorGraphCoordinator,
    edge_value: torch.Tensor,
    mask: torch.Tensor,
    teacher_pair: torch.Tensor,
    role_target: torch.Tensor,
    owner_target: torch.Tensor,
    equivalent_owner: torch.Tensor,
    realized_d_eff: torch.Tensor,
    *,
    evidence_ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    edge_features = factor_graph_edge_features(edge_value, mask)
    output = model(edge_features, mask)
    valid_logits = output.edge_logits[mask]
    valid_target = teacher_pair[mask].to(dtype=torch.float32)
    positive = valid_target.sum().clamp_min(1.0)
    negative = (1.0 - valid_target).sum().clamp_min(1.0)
    pos_weight = (negative / positive).clamp(1.0, 8.0)
    edge_loss = F.binary_cross_entropy_with_logits(
        valid_logits, valid_target, pos_weight=pos_weight)
    role_loss = F.cross_entropy(
        output.role_logits.reshape(-1, 3), role_target.reshape(-1))
    owner_log_probability = torch.log_softmax(output.owner_logits, dim=-1)
    owner_loss = -torch.logsumexp(
        owner_log_probability.masked_fill(~equivalent_owner, -torch.inf),
        dim=-1,
    ).mean()

    probability = torch.sigmoid(output.edge_logits) * mask
    role_probability = torch.softmax(output.role_logits, dim=-1)
    owner_probability = torch.softmax(output.owner_logits, dim=-1)[..., :-1]
    joint_probability = probability * (
        role_probability[:, :, None, None, 1]
        * role_probability[:, None, :, None, 2]
        * owner_probability.permute(0, 2, 1)[:, None, :, :]
    )
    teacher_evidence = torch.sum(
        realized_d_eff * teacher_pair, dim=(1, 2))
    predicted_evidence = torch.sum(
        realized_d_eff * joint_probability, dim=(1, 2))
    evidence_loss = torch.relu(
        float(evidence_ratio) * teacher_evidence - predicted_evidence)
    evidence_valid = teacher_evidence > 1.0e-12
    evidence_loss = torch.mean((
        evidence_loss / teacher_evidence.clamp_min(1.0e-6)
    )[evidence_valid])
    teacher_count = torch.sum(teacher_pair, dim=(1, 2, 3))
    predicted_count = torch.sum(joint_probability, dim=(1, 2, 3))
    cardinality_loss = torch.mean(
        torch.abs(predicted_count - teacher_count)
        / teacher_count.clamp_min(1.0))

    tx_probability = role_probability[..., 1]
    rx_probability = role_probability[..., 2]
    compatibility = probability * (
        (1.0 - tx_probability[:, :, None, None])
        + (1.0 - rx_probability[:, None, :, None])
        + (1.0 - owner_probability.permute(0, 2, 1)[:, None, :, :])
    )
    constraint_loss = compatibility[mask].mean()
    auxiliary = torch.stack([
        F.binary_cross_entropy_with_logits(logits[mask], valid_target)
        for logits in output.round_edge_logits
    ]).mean()
    total = (
        0.25 * edge_loss
        + 0.75 * role_loss
        + 1.00 * owner_loss
        + 2.00 * evidence_loss
        + 0.25 * cardinality_loss
        + 0.50 * constraint_loss
        + 0.10 * auxiliary
    )
    parts = {
        "total": float(total.detach()),
        "edge": float(edge_loss.detach()),
        "role": float(role_loss.detach()),
        "owner": float(owner_loss.detach()),
        "evidence": float(evidence_loss.detach()),
        "cardinality": float(cardinality_loss.detach()),
        "constraint": float(constraint_loss.detach()),
        "auxiliary": float(auxiliary.detach()),
    }
    return total, parts


def _batch_certificate_loss(
    model: FiniteRoundFactorGraphCoordinator,
    edge_value: torch.Tensor,
    mask: torch.Tensor,
    alternative_pair: torch.Tensor,
    alternative_role: torch.Tensor,
    alternative_owner: torch.Tensor,
    alternative_valid: torch.Tensor,
    realized_d_eff: torch.Tensor,
    *,
    set_temperature: float,
    evidence_ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Best-of-set loss over complete, jointly feasible certificates."""
    output = model(factor_graph_edge_features(edge_value, mask), mask)
    B, A = alternative_pair.shape[:2]
    valid_edge = mask[:, None, :, :, :]
    edge_logits = output.edge_logits[:, None, :, :, :].expand(
        B, A, *output.edge_logits.shape[1:])
    edge_nll = F.binary_cross_entropy_with_logits(
        edge_logits,
        alternative_pair.to(dtype=torch.float32),
        reduction="none",
    )
    edge_nll = (edge_nll * valid_edge).sum(dim=(2, 3, 4)) / (
        valid_edge.sum(dim=(2, 3, 4)).clamp_min(1))

    # Certificate roles are 1=Tx and 0=Rx for the complete latent partition.
    role_target = torch.where(
        alternative_role == 1,
        torch.ones_like(alternative_role),
        torch.full_like(alternative_role, 2),
    ).clamp_min(0)
    role_log_probability = torch.log_softmax(output.role_logits, dim=-1)
    role_nll = -torch.gather(
        role_log_probability[:, None, :, :].expand(B, A, -1, -1),
        -1,
        role_target[..., None],
    ).squeeze(-1).mean(dim=-1)
    K = output.role_logits.shape[1]
    owner_target = torch.where(
        alternative_owner >= 0,
        alternative_owner,
        torch.full_like(alternative_owner, K),
    ).clamp(0, K)
    owner_log_probability = torch.log_softmax(output.owner_logits, dim=-1)
    owner_nll = -torch.gather(
        owner_log_probability[:, None, :, :].expand(B, A, -1, -1),
        -1,
        owner_target[..., None],
    ).squeeze(-1).mean(dim=-1)
    structure_nll = 0.35 * edge_nll + 0.75 * role_nll + owner_nll
    temperature = max(float(set_temperature), 1.0e-3)
    masked_nll = structure_nll.masked_fill(~alternative_valid, torch.inf)
    count = alternative_valid.sum(dim=1).clamp_min(1).to(edge_value.dtype)
    set_loss = -temperature * (
        torch.logsumexp(-masked_nll / temperature, dim=1)
        - torch.log(count)
    )
    set_loss = set_loss.mean()

    # Physical evidence remains a secondary team-objective term. The leading
    # certificate is the teacher reference; alternatives already satisfy its
    # joint near-equivalence thresholds.
    probability = torch.sigmoid(output.edge_logits) * mask
    role_probability = torch.softmax(output.role_logits, dim=-1)
    owner_probability = torch.softmax(output.owner_logits, dim=-1)[..., :-1]
    joint_probability = probability * (
        role_probability[:, :, None, None, 1]
        * role_probability[:, None, :, None, 2]
        * owner_probability.permute(0, 2, 1)[:, None, :, :]
    )
    reference = alternative_pair[:, 0]
    teacher_evidence = torch.sum(
        realized_d_eff * reference, dim=(1, 2))
    predicted_evidence = torch.sum(
        realized_d_eff * joint_probability, dim=(1, 2))
    evidence_valid = teacher_evidence > 1.0e-12
    evidence_loss = torch.relu(
        float(evidence_ratio) * teacher_evidence - predicted_evidence)
    evidence_loss = torch.mean((
        evidence_loss / teacher_evidence.clamp_min(1.0e-6)
    )[evidence_valid])
    teacher_count = reference.sum(dim=(1, 2, 3))
    predicted_count = joint_probability.sum(dim=(1, 2, 3))
    cardinality_loss = torch.mean(
        torch.abs(predicted_count - teacher_count)
        / teacher_count.clamp_min(1.0))
    tx_probability = role_probability[..., 1]
    rx_probability = role_probability[..., 2]
    compatibility = probability * (
        (1.0 - tx_probability[:, :, None, None])
        + (1.0 - rx_probability[:, None, :, None])
        + (1.0 - owner_probability.permute(0, 2, 1)[:, None, :, :])
    )
    constraint_loss = compatibility[mask].mean()
    auxiliary = F.binary_cross_entropy_with_logits(
        output.edge_logits[mask], reference[mask].to(torch.float32))
    total = (
        set_loss
        + 0.75 * evidence_loss
        + 0.20 * cardinality_loss
        + 0.40 * constraint_loss
        + 0.05 * auxiliary
    )
    return total, {
        "total": float(total.detach()),
        "set": float(set_loss.detach()),
        "evidence": float(evidence_loss.detach()),
        "cardinality": float(cardinality_loss.detach()),
        "constraint": float(constraint_loss.detach()),
        "auxiliary": float(auxiliary.detach()),
    }


def _certificate_rows_for_frames(
    certificate: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
) -> np.ndarray:
    rows = np.full(len(data["frame"]), -1, dtype=np.int64)
    resolve_index = certificate["resolve_index"].astype(np.int64)
    if np.any(resolve_index < 0) or np.any(resolve_index >= len(rows)):
        raise ValueError("certificate resolve indices are outside the trace")
    for key in ("episode", "seed", "frame"):
        if not np.array_equal(
                certificate[key], data[key][resolve_index]):
            raise ValueError(f"certificate does not align with trace: {key}")
    rows[resolve_index] = np.arange(len(resolve_index))
    return rows


def _predict_resolve(
    model: FiniteRoundFactorGraphCoordinator,
    edge_value: np.ndarray,
    candidate_mask: np.ndarray,
    data: dict[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, float]]:
    frames, K, _, Q = candidate_mask.shape
    result = np.zeros_like(candidate_mask, dtype=bool)
    current = np.zeros((K, K, Q), dtype=bool)
    indices = np.flatnonzero(
        np.asarray(data["p0_resolved"], dtype=bool)
        & np.any(candidate_mask, axis=(1, 2, 3)))
    outputs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_index = indices[start:start + batch_size]
            value = torch.as_tensor(
                edge_value[batch_index], dtype=torch.float32, device=device)
            mask = torch.as_tensor(
                candidate_mask[batch_index], dtype=torch.bool, device=device)
            output = model(factor_graph_edge_features(value, mask), mask)
            edge = output.edge_logits.cpu().numpy()
            owner = output.owner_logits.cpu().numpy()
            role = output.role_logits.cpu().numpy()
            for offset, frame in enumerate(batch_index):
                outputs[int(frame)] = (edge[offset], owner[offset], role[offset])

    previous_episode: int | None = None
    repairs = []
    score_delta = []
    fallbacks = 0
    owner_repairs = 0
    resolved = 0
    violations = 0
    for frame in range(frames):
        episode = int(data["episode"][frame])
        if episode != previous_episode:
            current.fill(False)
            previous_episode = episode
        if bool(data["p0_resolved"][frame]):
            current.fill(False)
            if frame in outputs:
                edge, owner, role = outputs[frame]
                decoded = decode_lightweight_feasible(
                    edge,
                    owner,
                    role,
                    candidate_mask[frame],
                    target_pair_limit=int(data["target_pair_limit"][0]),
                    reports_per_receiver=int(data["reports_per_receiver"][0]),
                    edge_value=edge_value[frame],
                )
                current[:] = decoded.selected
                repairs.append(decoded.edge_change_rate)
                score_delta.append(decoded.score_delta)
                fallbacks += int(decoded.role_fallback)
                owner_repairs += int(decoded.owner_repairs)
                resolved += 1
                try:
                    assert_hard_structure_invariants(
                        current,
                        reports_per_receiver=int(
                            data["reports_per_receiver"][0]),
                    )
                except AssertionError:
                    violations += 1
        result[frame] = current
    return result, {
        "projection_repair_rate": float(np.mean(repairs)) if repairs else 0.0,
        "projection_score_delta": float(np.mean(score_delta)) if score_delta else 0.0,
        "role_fallback_rate": fallbacks / max(resolved, 1),
        "owner_repairs_per_resolve": owner_repairs / max(resolved, 1),
        "hard_violation_rate": violations / max(resolved, 1),
        "resolved_nonempty": resolved,
    }


def _structure_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    d_eff: np.ndarray,
    resolved: np.ndarray,
    *,
    evidence_ratio: float,
) -> dict[str, float]:
    frames = np.asarray(resolved, dtype=bool) & np.any(
        reference, axis=(1, 2, 3))
    predicted = prediction[frames]
    teacher = reference[frames]
    tp = int(np.sum(predicted & teacher))
    predicted_count = int(np.sum(predicted))
    teacher_count = int(np.sum(teacher))
    predicted_owner = receiver_owner_from_pairs(predicted)
    teacher_owner = receiver_owner_from_pairs(teacher)
    valid_owner = teacher_owner >= 0
    teacher_evidence = np.sum(
        d_eff[frames] * teacher, axis=(1, 2))
    predicted_evidence = np.sum(
        d_eff[frames] * predicted, axis=(1, 2))
    valid_evidence = teacher_evidence > 1.0e-12
    ratio = np.ones_like(teacher_evidence)
    ratio[valid_evidence] = (
        predicted_evidence[valid_evidence]
        / teacher_evidence[valid_evidence])
    return {
        "exact_edge_precision": tp / max(predicted_count, 1),
        "exact_edge_recall": tp / max(teacher_count, 1),
        "owner_accuracy": float(np.mean(
            predicted_owner[valid_owner] == teacher_owner[valid_owner])),
        "equivalent_evidence_recall": float(np.mean(
            ratio[valid_evidence] >= float(evidence_ratio))),
        "evidence_ratio_mean": float(np.mean(np.clip(
            ratio[valid_evidence], 0.0, 1.0))),
        "predicted_edges_per_resolve": predicted_count / max(
            int(np.sum(frames)), 1),
        "teacher_edges_per_resolve": teacher_count / max(
            int(np.sum(frames)), 1),
    }


def _evaluate(
    model: FiniteRoundFactorGraphCoordinator,
    edge_value: np.ndarray,
    candidate_mask: np.ndarray,
    teacher_pair: np.ndarray,
    data: dict[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
    args: argparse.Namespace,
) -> tuple[dict[str, object], np.ndarray]:
    prediction, projection = _predict_resolve(
        model,
        edge_value,
        candidate_mask,
        data,
        device=device,
        batch_size=batch_size,
    )
    p_fa = float(data["p_fa"][0])
    d_eff = np.asarray(data["privileged_d_eff"], dtype=np.float64)
    predicted_pd = realized_pd_history(prediction, d_eff, p_fa)
    teacher_pd = realized_pd_history(teacher_pair, d_eff, p_fa)
    thresholds = (args.steady_floor, args.weak3_floor, args.worst_floor)
    predicted_summary, rows = episode_detection_summary(
        predicted_pd,
        data["episode"],
        data["seed"],
        steady_window=args.steady_window,
        qos_thresholds=thresholds,
    )
    teacher_summary, _ = episode_detection_summary(
        teacher_pd,
        data["episode"],
        data["seed"],
        steady_window=args.steady_window,
        qos_thresholds=thresholds,
    )
    structure = _structure_metrics(
        prediction,
        teacher_pair,
        d_eff,
        data["p0_resolved"],
        evidence_ratio=args.evidence_ratio,
    )
    gaps = {
        key: float(teacher_summary[key]) - float(predicted_summary[key])
        for key in ("steady", "weak3", "worst", "cvar", "qos_feasible")
    }
    checks = {
        "worst_reference_gap": gaps["worst"] <= args.max_worst_gap,
        "cvar_reference_gap": gaps["cvar"] <= args.max_cvar_gap,
        "hard_constraints": projection["hard_violation_rate"] == 0.0,
        "projection_repair_rate": (
            projection["projection_repair_rate"] <= args.max_projection_repair),
        "equivalent_evidence": (
            structure["equivalent_evidence_recall"]
            >= args.min_equivalent_evidence),
    }
    return {
        "teacher_reference": teacher_summary,
        "coordinator": predicted_summary,
        "reference_minus_coordinator": gaps,
        "structure": structure,
        "projection": projection,
        "gate_c1": {"checks": checks, "pass": bool(all(checks.values()))},
        "episode_rows": rows,
    }, prediction


def _aligned_data(
    trace_path: Path,
    candidate_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    raw = _load_npz(trace_path)
    candidate = _load_npz(candidate_path)
    data = _subset_trace(raw, np.unique(candidate["seed"]))
    for key in ("episode", "seed", "frame"):
        if not np.array_equal(data[key], candidate[key]):
            raise ValueError(f"candidate does not align with trace: {key}")
    return data, candidate


def train(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    started = time.perf_counter()
    train_data, train_candidate = _aligned_data(
        args.trace, args.train_candidate)
    eval_data, eval_candidate = _aligned_data(
        args.trace, args.eval_candidate)
    train_value = _edge_values(
        train_data, train_candidate, args.student_checkpoint)
    eval_value = _edge_values(
        eval_data, eval_candidate, args.student_checkpoint)
    train_mask = train_candidate["candidate_mask"].astype(bool)
    eval_mask = eval_candidate["candidate_mask"].astype(bool)
    train_teacher_cache = (
        args.train_teacher_cache
        if args.train_teacher_cache is not None
        else args.output_dir / "train_teacher_cache.npz"
    )
    train_teacher = _replicated_teacher_sequence(
        train_value,
        train_mask,
        train_data,
        train_teacher_cache,
    )
    eval_cache = args.eval_teacher_cache
    if eval_cache is not None and eval_cache.exists():
        cached = _load_npz(eval_cache)
        eval_teacher = cached["pairs_replicated_consensus"].astype(bool)
    else:
        eval_teacher = _replicated_teacher_sequence(
            eval_value,
            eval_mask,
            eval_data,
            args.output_dir / "eval_teacher_cache.npz",
        )

    train_role, train_owner = _targets_from_pair(train_teacher)
    train_equivalent_owner = _equivalent_owner_sets(
        train_teacher,
        train_mask,
        train_data["privileged_d_eff"],
        target_pair_limit=int(train_data["target_pair_limit"][0]),
        evidence_ratio=args.evidence_ratio,
    )
    certificate = None
    certificate_row = None
    if args.train_certificate is not None:
        certificate = _load_npz(args.train_certificate)
        certificate_row = _certificate_rows_for_frames(
            certificate, train_data)
    sample = (
        train_data["p0_resolved"].astype(bool)
        & np.any(train_mask, axis=(1, 2, 3)))
    seeds = np.unique(train_data["seed"])
    validation_count = max(1, int(round(len(seeds) * args.validation_fraction)))
    validation_seeds = seeds[-validation_count:]
    validation = sample & np.isin(train_data["seed"], validation_seeds)
    fitting = sample & ~np.isin(train_data["seed"], validation_seeds)
    fit_indices = np.flatnonzero(fitting)
    validation_indices = np.flatnonzero(validation)
    if not len(fit_indices) or not len(validation_indices):
        raise ValueError("empty fit/validation split")

    model = FiniteRoundFactorGraphCoordinator(
        hidden_dim=args.hidden_dim,
        rounds=args.rounds,
        coupling_strength=args.coupling_strength,
        coupling_rounds=args.coupling_rounds,
        use_global_context=args.global_context,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_state = None
    best_validation = float("inf")
    history = []
    generator = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = generator.permutation(fit_indices)
        train_parts: list[dict[str, float]] = []
        for start in range(0, len(shuffled), args.batch_size):
            index = shuffled[start:start + args.batch_size]
            value = torch.as_tensor(
                train_value[index], dtype=torch.float32, device=device)
            mask = torch.as_tensor(
                train_mask[index], dtype=torch.bool, device=device)
            target = torch.as_tensor(
                train_teacher[index], dtype=torch.float32, device=device)
            role = torch.as_tensor(
                train_role[index], dtype=torch.long, device=device)
            owner = torch.as_tensor(
                train_owner[index], dtype=torch.long, device=device)
            realized_d_eff = torch.as_tensor(
                train_data["privileged_d_eff"][index],
                dtype=torch.float32,
                device=device,
            )
            if certificate is None or certificate_row is None:
                equivalent_owner = torch.as_tensor(
                    train_equivalent_owner[index],
                    dtype=torch.bool,
                    device=device,
                )
                loss, parts = _batch_loss(
                    model, value, mask, target, role, owner,
                    equivalent_owner, realized_d_eff,
                    evidence_ratio=args.evidence_ratio)
            else:
                row = certificate_row[index]
                if np.any(row < 0):
                    raise ValueError("training frame lacks a certificate")
                alternative_count = certificate["alternative_count"][row]
                A = certificate["selected"].shape[1]
                alternative_valid = (
                    np.arange(A)[None, :] < alternative_count[:, None])
                loss, parts = _batch_certificate_loss(
                    model,
                    value,
                    mask,
                    torch.as_tensor(
                        certificate["selected"][row],
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.as_tensor(
                        certificate["role"][row],
                        dtype=torch.long,
                        device=device,
                    ),
                    torch.as_tensor(
                        certificate["owner"][row],
                        dtype=torch.long,
                        device=device,
                    ),
                    torch.as_tensor(
                        alternative_valid,
                        dtype=torch.bool,
                        device=device,
                    ),
                    realized_d_eff,
                    set_temperature=args.set_temperature,
                    evidence_ratio=args.evidence_ratio,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_parts.append(parts)
        model.eval()
        with torch.inference_mode():
            index = validation_indices
            validation_value_tensor = torch.as_tensor(
                train_value[index], dtype=torch.float32, device=device)
            validation_mask_tensor = torch.as_tensor(
                train_mask[index], dtype=torch.bool, device=device)
            validation_d_eff = torch.as_tensor(
                train_data["privileged_d_eff"][index],
                dtype=torch.float32,
                device=device,
            )
            if certificate is None or certificate_row is None:
                validation_loss, validation_parts = _batch_loss(
                    model,
                    validation_value_tensor,
                    validation_mask_tensor,
                    torch.as_tensor(
                        train_teacher[index],
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.as_tensor(
                        train_role[index], dtype=torch.long, device=device),
                    torch.as_tensor(
                        train_owner[index], dtype=torch.long, device=device),
                    torch.as_tensor(
                        train_equivalent_owner[index],
                        dtype=torch.bool,
                        device=device,
                    ),
                    validation_d_eff,
                    evidence_ratio=args.evidence_ratio,
                )
            else:
                row = certificate_row[index]
                alternative_count = certificate["alternative_count"][row]
                A = certificate["selected"].shape[1]
                validation_loss, validation_parts = _batch_certificate_loss(
                    model,
                    validation_value_tensor,
                    validation_mask_tensor,
                    torch.as_tensor(
                        certificate["selected"][row],
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.as_tensor(
                        certificate["role"][row],
                        dtype=torch.long,
                        device=device,
                    ),
                    torch.as_tensor(
                        certificate["owner"][row],
                        dtype=torch.long,
                        device=device,
                    ),
                    torch.as_tensor(
                        np.arange(A)[None, :] < alternative_count[:, None],
                        dtype=torch.bool,
                        device=device,
                    ),
                    validation_d_eff,
                    set_temperature=args.set_temperature,
                    evidence_ratio=args.evidence_ratio,
                )
        validation_value = float(validation_loss)
        if validation_value < best_validation:
            best_validation = validation_value
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        epoch_row = {
            "epoch": epoch,
            "train_total": float(np.mean([
                item["total"] for item in train_parts])),
            **{f"validation_{key}": value
               for key, value in validation_parts.items()},
        }
        history.append(epoch_row)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(json.dumps(epoch_row), flush=True)
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)

    train_report, train_prediction = _evaluate(
        model, train_value, train_mask, train_teacher, train_data,
        device=device, batch_size=args.batch_size, args=args)
    eval_report, eval_prediction = _evaluate(
        model, eval_value, eval_mask, eval_teacher, eval_data,
        device=device, batch_size=args.batch_size, args=args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "factor_graph_coordinator.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "hidden_dim": args.hidden_dim,
        "rounds": args.rounds,
        "coupling_strength": args.coupling_strength,
        "coupling_rounds": args.coupling_rounds,
        "use_global_context": args.global_context,
        "edge_feature_dim": 4,
        "schema": (
            "gate_c1_factor_graph_coordinator_v3_joint_certificate"
            if certificate is not None
            else "gate_c1_factor_graph_coordinator_v2_equivalent_supervision"),
    }, checkpoint)
    np.savez_compressed(
        args.output_dir / "per_frame.npz",
        train_prediction=train_prediction.astype(np.uint8),
        train_teacher=train_teacher.astype(np.uint8),
        eval_prediction=eval_prediction.astype(np.uint8),
        eval_teacher=eval_teacher.astype(np.uint8),
        eval_episode=eval_data["episode"],
        eval_seed=eval_data["seed"],
        eval_frame=eval_data["frame"],
    )
    with (args.output_dir / "history.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (args.output_dir / "episode_metrics.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        rows = eval_report.pop("episode_rows")
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    train_report.pop("episode_rows")
    report = {
        "protocol": (
            "gate_c1_factor_graph_coordinator_v3_joint_certificate"
            if certificate is not None
            else "gate_c1_factor_graph_coordinator_v2_equivalent_supervision"),
        "train_seeds": [int(value) for value in seeds],
        "validation_seeds": [int(value) for value in validation_seeds],
        "eval_seeds": [int(value) for value in np.unique(eval_data["seed"])],
        "fit_resolve_frames": int(len(fit_indices)),
        "validation_resolve_frames": int(len(validation_indices)),
        "model": {
            "hidden_dim": args.hidden_dim,
            "rounds": args.rounds,
            "coupling_strength": args.coupling_strength,
            "coupling_rounds": args.coupling_rounds,
            "use_global_context": args.global_context,
            "parameters": int(sum(parameter.numel()
                                  for parameter in model.parameters())),
        },
        "thresholds": {
            "max_worst_reference_gap": args.max_worst_gap,
            "max_cvar_reference_gap": args.max_cvar_gap,
            "max_projection_repair_rate": args.max_projection_repair,
            "min_equivalent_evidence_recall": args.min_equivalent_evidence,
        },
        "train_reference_replay": train_report,
        "independent_eval": eval_report,
        "checkpoint": str(checkpoint),
        "train_certificate": (
            str(args.train_certificate)
            if args.train_certificate is not None else None),
        "runtime_s": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace", type=Path, default=Path(
            "results/architecture_v2_scale_k6q6_teacher_trace_selection20/teacher_trace.npz"))
    parser.add_argument(
        "--train-candidate", type=Path, default=Path(
            "results/architecture_v2_scale_k6q6_local_candidate_value_lu4_lq4_r1_gate10/per_frame.npz"))
    parser.add_argument(
        "--eval-candidate", type=Path, default=Path(
            "results/architecture_v2_scale_k6q6_local_candidate_value_lu4_lq4_r1_gate_a2_holdout10/per_frame.npz"))
    parser.add_argument(
        "--eval-teacher-cache", type=Path, default=Path(
            "results/architecture_v2_scale_k6q6_replicated_consensus_holdout10/per_frame.npz"))
    parser.add_argument("--train-teacher-cache", type=Path, default=None)
    parser.add_argument("--train-certificate", type=Path, default=None)
    parser.add_argument(
        "--student-checkpoint", type=Path, default=Path(
            "results/architecture_v2_teacher_cleanreset_trace_gate100/frozen_structure_student_endpoint8.pt"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path(
            "results/architecture_v2_scale_k6q6_factor_graph_gate_c1"))
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--coupling-strength", type=float, default=1.0)
    parser.add_argument("--coupling-rounds", type=int, default=0)
    parser.add_argument("--global-context", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--evidence-ratio", type=float, default=0.95)
    parser.add_argument("--set-temperature", type=float, default=0.25)
    parser.add_argument("--max-worst-gap", type=float, default=0.02)
    parser.add_argument("--max-cvar-gap", type=float, default=0.01)
    parser.add_argument("--max-projection-repair", type=float, default=0.05)
    parser.add_argument("--min-equivalent-evidence", type=float, default=0.90)
    parser.add_argument("--steady-floor", type=float, default=0.80)
    parser.add_argument("--weak3-floor", type=float, default=0.70)
    parser.add_argument("--worst-floor", type=float, default=0.60)
    parser.add_argument("--steady-window", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    result = train(parse_args())
    print(json.dumps(result, indent=2))
