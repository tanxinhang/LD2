#!/usr/bin/env python
"""Audit whether a frozen centralized P0 teacher is distillable.

The audit separates three questions:

1. Can the existing Top-2 Token/sensing claims represent the teacher edges?
2. Does pre-decision local information predict transmitter/receiver ownership?
3. How much of the remaining ambiguity comes from the teacher's privileged
   same-frame physical candidate graph?

Seed-grouped train/test splits prevent adjacent Hold-5 frames from leaking
teacher assignments across the offline probe boundary.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uav_isac.environment.observation_slices import ObservationSlices
from uav_isac.agents.frozen_structure_student import (
    FactorizedStructureStudent,
    build_structure_student_features,
    save_frozen_structure_student,
)
from uav_isac.physical.feasibility_oracle import (
    solve_maxmin_single_role_pairs,
)
from uav_isac.physical.detection import (
    compute_detection_probabilities,
)
from uav_isac.utils.types import DeflectionEntry


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks with deterministic tie handling (SciPy-free)."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while (end < len(values)
               and sorted_values[end] == sorted_values[start]):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(values_a: np.ndarray, values_b: np.ndarray) -> float:
    rank_a = _rankdata(values_a)
    rank_b = _rankdata(values_b)
    rank_a = rank_a - np.mean(rank_a)
    rank_b = rank_b - np.mean(rank_b)
    norm_a = float(np.sqrt(np.sum(rank_a * rank_a)))
    norm_b = float(np.sqrt(np.sum(rank_b * rank_b)))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return float("nan")
    return float(np.sum(rank_a * rank_b) / (norm_a * norm_b))


def _binary_metrics(score: np.ndarray, label: np.ndarray) -> dict[str, float]:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    label = np.asarray(label, dtype=bool).reshape(-1)
    pred = score >= 0.5
    tp = int(np.sum(pred & label))
    fp = int(np.sum(pred & ~label))
    fn = int(np.sum(~pred & label))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    positives = score[label]
    negatives = score[~label]
    if positives.size and negatives.size:
        # Pairwise definition of AUROC; deterministic and tie aware.
        comparison = positives[:, None] - negatives[None, :]
        auroc = float(np.mean(
            (comparison > 0.0) + 0.5 * (comparison == 0.0)))
    else:
        auroc = 0.5
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "positive_rate": float(np.mean(label)),
        "predicted_positive_rate": float(np.mean(pred)),
    }


def _claim_metrics(
    claim: np.ndarray,
    tx: np.ndarray,
    rx: np.ndarray,
    pair: np.ndarray,
) -> dict[str, float]:
    claim = np.asarray(claim, dtype=bool)
    endpoint = np.logical_or(tx, rx)
    endpoint_stats = _binary_metrics(claim.astype(np.float64), endpoint)
    edge_supported = (
        claim[:, :, None, :]
        & claim[:, None, :, :]
    )
    selected = np.asarray(pair, dtype=bool)
    selected_count = int(np.sum(selected))
    return {
        **{f"endpoint_{key}": value
           for key, value in endpoint_stats.items()},
        "teacher_edge_support_rate": float(
            np.sum(edge_supported & selected) / max(selected_count, 1)),
        "team_exact_endpoint_rate": float(np.mean(np.all(
            claim == endpoint, axis=(1, 2)))),
    }


def _infer_observation_slices(
    obs_dim: int,
    K: int,
    Q: int,
) -> ObservationSlices:
    base_without_tokens = (
        8 + 9 * Q + 8 * Q + 3 + 8 * (K - 1) + Q + 16)
    token_count = (K - 1) * Q
    token_dim_numerator = obs_dim - base_without_tokens - token_count
    if token_count <= 0 or token_dim_numerator % token_count:
        raise ValueError(
            "cannot infer target-token observation layout from trace")
    token_dim = token_dim_numerator // token_count
    if token_dim < 6:
        raise ValueError("inferred communication token is too short")
    return ObservationSlices.from_config(
        K=K,
        Q=Q,
        use_p0=False,
        use_rel_features=True,
        use_comm_tokens=True,
        comm_token_dim=token_dim,
        comm_tokens_per_sender=Q,
    )


def _per_target_features(
    data: dict[str, np.ndarray],
    obs_key: str,
    indices: np.ndarray,
    slices: ObservationSlices,
) -> np.ndarray:
    return build_structure_student_features(
        np.asarray(data[obs_key][indices], dtype=np.float32),
        slices,
        outgoing_message=data["outgoing_message"][indices],
        outgoing_token_mask=data["outgoing_token_mask"][indices],
        outgoing_rate=data["outgoing_rate"][indices],
        comm_fraction=data["comm_fraction"][indices],
        sensing_weights=data["sensing_weights"][indices],
        rate_scale=max(float(np.max(data["outgoing_rate"])), 1.0),
    )


class _SharedTargetScorer(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class _PersistentResidualScorer(torch.nn.Module):
    """Predict only a bounded correction to the previous feasible structure."""

    def __init__(self, input_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.residual = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2),
        )
        torch.nn.init.zeros_(self.residual[-1].weight)
        torch.nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        previous: torch.Tensor,
    ) -> torch.Tensor:
        persistence_logit = torch.where(
            previous > 0.5,
            torch.full_like(previous, 2.1972246),
            torch.full_like(previous, -2.1972246),
        )
        return persistence_logit + self.residual(features)


def _fit_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    *,
    epochs: int,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(int(seed))
    x_train_np = train_features.reshape(-1, train_features.shape[-1])
    y_train_np = train_labels.reshape(-1, 2).astype(np.float32)
    x_test_np = test_features.reshape(-1, test_features.shape[-1])
    mean = x_train_np.mean(axis=0, keepdims=True)
    scale = x_train_np.std(axis=0, keepdims=True)
    scale = np.maximum(scale, 1.0e-5)
    x_train = torch.as_tensor(
        (x_train_np - mean) / scale, dtype=torch.float32)
    y_train = torch.as_tensor(y_train_np, dtype=torch.float32)
    x_test = torch.as_tensor(
        (x_test_np - mean) / scale, dtype=torch.float32)
    positive = y_train.sum(dim=0)
    negative = y_train.shape[0] - positive
    pos_weight = negative / positive.clamp_min(1.0)
    model = _SharedTargetScorer(x_train.shape[-1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    batch = min(1024, len(x_train))
    generator = torch.Generator().manual_seed(int(seed))
    for _ in range(max(1, int(epochs))):
        order = torch.randperm(
            len(x_train), generator=generator)
        for start in range(0, len(order), batch):
            index = order[start:start + batch]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train[index]), y_train[index])
            loss.backward()
            optimizer.step()
    with torch.inference_mode():
        score = torch.sigmoid(model(x_test)).cpu().numpy()
    return score.reshape(*test_features.shape[:-1], 2)


def _fit_persistent_residual_probe(
    train_features: np.ndarray,
    train_previous: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_previous: np.ndarray,
    validation_labels: np.ndarray,
    test_features: np.ndarray,
    test_previous: np.ndarray,
    *,
    epochs: int,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(int(seed))
    feature_dim = train_features.shape[-1]
    mean = train_features.reshape(-1, feature_dim).mean(
        axis=0, keepdims=True)
    scale = np.maximum(
        train_features.reshape(-1, feature_dim).std(
            axis=0, keepdims=True),
        1.0e-5,
    )

    def tensor(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32)

    x_train = tensor((train_features - mean) / scale)
    p_train = tensor(train_previous.astype(np.float32))
    y_train = tensor(train_labels.astype(np.float32))
    x_validation = tensor((validation_features - mean) / scale)
    p_validation = tensor(validation_previous.astype(np.float32))
    y_validation = tensor(validation_labels.astype(np.float32))
    x_test = tensor((test_features - mean) / scale)
    p_test = tensor(test_previous.astype(np.float32))

    model = _PersistentResidualScorer(feature_dim)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5.0e-4, weight_decay=1.0e-4)
    tx_loss_fn = torch.nn.BCEWithLogitsLoss()
    owner_loss_fn = torch.nn.CrossEntropyLoss()
    role_loss_fn = torch.nn.BCEWithLogitsLoss()

    def structured_loss(
        features: torch.Tensor,
        previous: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        logits = model(
            features.reshape(-1, feature_dim),
            previous.reshape(-1, 2),
        ).reshape(*features.shape[:-1], 2)
        tx_loss = tx_loss_fn(logits[..., 0], label[..., 0])
        # Exactly one receiver owns each target. Receiver prediction is a
        # K-way categorical projection, not K independent Bernoulli labels.
        owner_logits = logits[..., 1].permute(0, 2, 1)
        owner_label = label[..., 1].argmax(dim=1)
        owner_loss = owner_loss_fn(
            owner_logits.reshape(-1, owner_logits.shape[-1]),
            owner_label.reshape(-1),
        )
        rx_role_label = label[..., 1].amax(dim=-1)
        role_logit = (
            torch.logsumexp(logits[..., 1], dim=-1)
            - torch.logsumexp(logits[..., 0], dim=-1)
        )
        role_loss = role_loss_fn(role_logit, rx_role_label)
        return tx_loss + owner_loss + 0.5 * role_loss

    batch = min(64, len(x_train))
    generator = torch.Generator().manual_seed(int(seed))
    best_loss = float(structured_loss(
        x_validation, p_validation, y_validation).item())
    best_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    stale = 0
    for _ in range(max(1, int(epochs))):
        order = torch.randperm(
            len(x_train), generator=generator)
        for start in range(0, len(order), batch):
            index = order[start:start + batch]
            optimizer.zero_grad()
            loss = structured_loss(
                x_train[index], p_train[index], y_train[index])
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            validation_loss = float(structured_loss(
                x_validation,
                p_validation,
                y_validation,
            ).item())
        if validation_loss < best_loss - 1.0e-5:
            best_loss = validation_loss
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= 20:
            break
    model.load_state_dict(best_state)
    with torch.inference_mode():
        score = torch.sigmoid(
            model(
                x_test.reshape(-1, feature_dim),
                p_test.reshape(-1, 2),
            )).cpu().numpy()
    return score.reshape(*test_features.shape[:-1], 2)


def _fit_factorized_edge_value_probe(
    train_features: np.ndarray,
    train_d_eff: np.ndarray,
    train_selected: np.ndarray,
    validation_features: np.ndarray,
    validation_d_eff: np.ndarray,
    validation_selected: np.ndarray,
    test_features: np.ndarray,
    train_base_d_eff: np.ndarray | None = None,
    validation_base_d_eff: np.ndarray | None = None,
    test_base_d_eff: np.ndarray | None = None,
    train_target_weight: np.ndarray | None = None,
    validation_target_weight: np.ndarray | None = None,
    selected_edge_weight: float = 2.0,
    selected_underprediction_weight: float = 0.0,
    selection_distribution_weight: float = 0.0,
    artifact_output: Path | None = None,
    artifact_metadata: dict[str, object] | None = None,
    artifact_rate_scale: float = 1.0,
    endpoint_dim: int = 32,
    *,
    epochs: int,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(int(seed))
    feature_dim = train_features.shape[-1]
    K = train_features.shape[1]
    diagonal_mask = ~np.eye(K, dtype=bool)[None, :, :, None]
    mean = train_features.reshape(-1, feature_dim).mean(
        axis=0, keepdims=True)
    scale = np.maximum(
        train_features.reshape(-1, feature_dim).std(
            axis=0, keepdims=True),
        1.0e-5,
    )

    def feature_tensor(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            (values - mean) / scale, dtype=torch.float32)

    def target_tensor(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            np.log1p(np.maximum(values, 0.0)), dtype=torch.float32)

    x_train = feature_tensor(train_features)
    y_train = target_tensor(train_d_eff)
    selected_train = torch.as_tensor(
        train_selected, dtype=torch.float32)
    x_validation = feature_tensor(validation_features)
    y_validation = target_tensor(validation_d_eff)
    selected_validation = torch.as_tensor(
        validation_selected, dtype=torch.float32)
    if train_target_weight is None:
        train_target_weight_tensor = torch.ones(
            train_features.shape[0],
            train_features.shape[2],
            dtype=torch.float32,
        )
        validation_target_weight_tensor = torch.ones(
            validation_features.shape[0],
            validation_features.shape[2],
            dtype=torch.float32,
        )
    else:
        if validation_target_weight is None:
            raise ValueError(
                "validation target weights are required with train weights")
        train_target_weight_tensor = torch.as_tensor(
            train_target_weight, dtype=torch.float32)
        validation_target_weight_tensor = torch.as_tensor(
            validation_target_weight, dtype=torch.float32)
    x_test = feature_tensor(test_features)
    if train_base_d_eff is None:
        train_base_log = torch.zeros_like(y_train)
        validation_base_log = torch.zeros_like(y_validation)
        test_base_log = torch.zeros(
            test_features.shape[0],
            K,
            K,
            test_features.shape[2],
            dtype=torch.float32,
        )
    else:
        if (validation_base_d_eff is None
                or test_base_d_eff is None):
            raise ValueError(
                "all physics-base tensors must be provided together")
        train_base_log = target_tensor(train_base_d_eff)
        validation_base_log = target_tensor(
            validation_base_d_eff)
        test_base_log = target_tensor(test_base_d_eff)
    valid_mask = torch.as_tensor(
        diagonal_mask, dtype=torch.bool)

    model = FactorizedStructureStudent(
        feature_dim, endpoint_dim=max(1, int(endpoint_dim)))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=7.5e-4, weight_decay=1.0e-4)

    def loss_fn(
        prediction: torch.Tensor,
        target: torch.Tensor,
        selected: torch.Tensor,
        target_weight: torch.Tensor,
    ) -> torch.Tensor:
        mask = valid_mask.expand_as(prediction)
        target_weight = target_weight[:, None, None, :]
        weight = target_weight * (
            1.0 + float(selected_edge_weight) * selected)
        if selected_underprediction_weight > 0.0:
            underpredicted = (
                (prediction < target).to(prediction.dtype) * selected)
            weight = weight * (
                1.0
                + float(selected_underprediction_weight) * underpredicted)
        error = torch.nn.functional.smooth_l1_loss(
            prediction, target, reduction="none")
        regression = (weight[mask] * error[mask]).sum() / (
            weight[mask].sum().clamp_min(1.0))
        if selection_distribution_weight <= 0.0:
            return regression

        # Auxiliary structured distillation.  The teacher-selected edge values
        # define a soft target distribution for each target; unlike a hard
        # edge BCE this preserves the continuous deflection ordering and only
        # asks the student to place probability mass on projection-relevant
        # edges.
        F, _, _, Q = prediction.shape
        prediction_flat = prediction.permute(
            0, 3, 1, 2).reshape(F, Q, -1)
        target_flat = target.permute(
            0, 3, 1, 2).reshape(F, Q, -1)
        selected_flat = selected.permute(
            0, 3, 1, 2).reshape(F, Q, -1)
        edge_valid = valid_mask.expand_as(prediction).permute(
            0, 3, 1, 2).reshape(F, Q, -1)
        prediction_logits = prediction_flat.masked_fill(
            ~edge_valid, -1.0e4)
        teacher_logits = (
            target_flat + 1.0 * selected_flat).masked_fill(
                ~edge_valid, -1.0e4)
        teacher_distribution = torch.softmax(
            teacher_logits.detach(), dim=-1)
        selection_cross_entropy = -torch.sum(
            teacher_distribution
            * torch.log_softmax(prediction_logits, dim=-1),
            dim=-1,
        )
        selection_loss = (
            selection_cross_entropy * target_weight[:, 0, 0, :]
        ).sum() / target_weight[:, 0, 0, :].sum().clamp_min(1.0)
        return (
            regression
            + float(selection_distribution_weight) * selection_loss)

    best_loss = float(loss_fn(
        validation_base_log + model(x_validation),
        y_validation,
        selected_validation,
        validation_target_weight_tensor,
    ).item())
    best_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    batch = min(32, len(x_train))
    generator = torch.Generator().manual_seed(int(seed))
    stale = 0
    for _ in range(max(1, int(epochs))):
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), batch):
            index = order[start:start + batch]
            optimizer.zero_grad()
            prediction = (
                train_base_log[index] + model(x_train[index]))
            loss = loss_fn(
                prediction,
                y_train[index],
                selected_train[index],
                train_target_weight_tensor[index],
            )
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            validation_loss = float(loss_fn(
                validation_base_log + model(x_validation),
                y_validation,
                selected_validation,
                validation_target_weight_tensor,
            ).item())
        if validation_loss < best_loss - 1.0e-5:
            best_loss = validation_loss
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= 30:
            break
    model.load_state_dict(best_state)
    if artifact_output is not None:
        if train_base_d_eff is not None:
            raise ValueError(
                "physics-residual students are not deployable artifacts")
        with torch.inference_mode():
            endpoint_samples = []
            for start in range(0, len(x_train), batch):
                tx_endpoint, rx_endpoint = model.encode_endpoints(
                    x_train[start:start + batch])
                endpoint_samples.append(torch.cat([
                    tx_endpoint.reshape(-1, model.endpoint_dim),
                    rx_endpoint.reshape(-1, model.endpoint_dim),
                ], dim=-1))
            endpoint_values = torch.cat(endpoint_samples, dim=0)
            # Robust per-dimension calibration maps almost all trained endpoint
            # values into [-1, 1] while preventing a few outliers from wasting
            # the low-bit U2U codebook.
            endpoint_scale = torch.quantile(
                endpoint_values.abs(), 0.995, dim=0,
            ).clamp_min(1.0e-5).cpu().numpy()
        save_frozen_structure_student(
            artifact_output,
            model,
            mean,
            scale,
            num_uavs=K,
            num_targets=train_features.shape[2],
            rate_scale=float(artifact_rate_scale),
            endpoint_scale=endpoint_scale,
            metadata={
                "objective": "continuous_log1p_edge_value",
                "seed": int(seed),
                "selected_edge_weight": float(selected_edge_weight),
                "selected_underprediction_weight": float(
                    selected_underprediction_weight),
                "selection_distribution_weight": float(
                    selection_distribution_weight),
                **dict(artifact_metadata or {}),
            },
        )
    with torch.inference_mode():
        prediction = (
            test_base_log + model(x_test)).cpu().numpy()
    prediction = np.maximum(np.expm1(prediction), 0.0)
    for k in range(K):
        prediction[:, k, k, :] = 0.0
    return prediction


def _edge_value_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    K = predicted.shape[1]
    mask = np.broadcast_to(
        ~np.eye(K, dtype=bool)[None, :, :, None],
        predicted.shape,
    )
    log_prediction = np.log1p(predicted[mask])
    log_target = np.log1p(target[mask])
    rmse = float(np.sqrt(np.mean(
        (log_prediction - log_target) ** 2)))
    frame_rank = []
    for frame_prediction, frame_target in zip(predicted, target):
        correlation = _spearman(
            frame_prediction[~np.eye(K, dtype=bool), :].reshape(-1),
            frame_target[~np.eye(K, dtype=bool), :].reshape(-1),
        )
        if np.isfinite(correlation):
            frame_rank.append(correlation)
    return {
        "log1p_rmse": rmse,
        "frame_edge_spearman": float(np.mean(frame_rank))
        if frame_rank else 0.0,
    }


def _teacher_bottleneck_target_weights(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    strength: float = 3.0,
    temperature: float = 0.15,
) -> np.ndarray:
    """Weight targets by the teacher's same-state QoS bottleneck severity.

    This quantity is used only to choose which continuous edge errors receive
    gradient.  It is not exposed to the student at inference time.
    """
    d_eff = np.asarray(
        data["privileged_d_eff"][indices], dtype=np.float64)
    selected = np.asarray(
        data["teacher_pair"][indices], dtype=np.float64)
    receiver_d = np.sum(d_eff * selected, axis=1)
    target_d = np.max(receiver_d, axis=1)
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    target_pd = compute_detection_probabilities(target_d, p_fa)
    floor = float(np.asarray(data["qos_floor"]).reshape(-1)[0])
    hardness = np.exp(np.clip(
        (floor - target_pd) / max(float(temperature), 1.0e-6),
        -6.0,
        6.0,
    ))
    hardness /= np.maximum(
        np.mean(hardness, axis=-1, keepdims=True), 1.0e-8)
    return np.asarray(
        1.0 + float(strength) * hardness, dtype=np.float32)


def _physics_loglinear_edge_value(
    data: dict[str, np.ndarray],
    fit_indices: np.ndarray,
    test_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Calibrate the bistatic inverse-range law on frozen teacher traces."""
    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])

    def design(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        uav = np.asarray(
            data["uav_positions"][indices], dtype=np.float64)
        target_xy = np.asarray(
            data["target_states"][indices, :, :2], dtype=np.float64)
        target = np.concatenate([
            target_xy,
            np.zeros((*target_xy.shape[:-1], 1), dtype=np.float64),
        ], axis=-1)
        range_m = np.linalg.norm(
            uav[:, :, None, :] - target[:, None, :, :],
            axis=-1,
        ).clip(min=1.0)
        sensing_power = (
            (1.0 - np.asarray(
                data["comm_fraction"][indices], dtype=np.float64))
            [..., None]
            * np.asarray(
                data["sensing_weights"][indices], dtype=np.float64)
        ).clip(min=1.0e-8)
        log_power = np.log(sensing_power)[:, :, None, :]
        log_tx_range = np.log(range_m)[:, :, None, :]
        log_rx_range = np.log(range_m)[:, None, :, :]
        shape = (len(indices), K, K, Q)
        return np.stack([
            np.ones(shape, dtype=np.float64),
            np.broadcast_to(log_power, shape),
            np.broadcast_to(log_tx_range, shape),
            np.broadcast_to(log_rx_range, shape),
        ], axis=-1), range_m

    fit_design, _ = design(fit_indices)
    test_design, _ = design(test_indices)
    fit_target = np.asarray(
        data["privileged_d_eff"][fit_indices], dtype=np.float64)
    fit_selected = np.asarray(
        data["teacher_pair"][fit_indices], dtype=np.float64)
    off_diagonal = np.broadcast_to(
        ~np.eye(K, dtype=bool)[None, :, :, None],
        fit_target.shape,
    )
    positive = off_diagonal & (fit_target > 1.0e-20)
    x = fit_design[positive]
    y = np.log(fit_target[positive])
    weight = np.sqrt(1.0 + 2.0 * fit_selected[positive])
    weighted_x = x * weight[:, None]
    weighted_y = y * weight
    gram = np.zeros((4, 4), dtype=np.float64)
    rhs = np.zeros(4, dtype=np.float64)
    for row, target_value in zip(weighted_x, weighted_y):
        for a in range(4):
            rhs[a] += row[a] * target_value
            for b in range(4):
                gram[a, b] += row[a] * row[b]
    # Four-dimensional Gaussian elimination avoids loading a second BLAS/OpenMP
    # runtime after PyTorch on Windows.
    augmented = np.concatenate([gram, rhs[:, None]], axis=1)
    for column in range(4):
        pivot = column + int(np.argmax(np.abs(
            augmented[column:, column])))
        if abs(float(augmented[pivot, column])) <= 1.0e-12:
            raise ValueError("physics calibration design is rank deficient")
        if pivot != column:
            augmented[[column, pivot]] = augmented[[pivot, column]]
        augmented[column] /= augmented[column, column]
        for row_index in range(4):
            if row_index == column:
                continue
            augmented[row_index] -= (
                augmented[row_index, column] * augmented[column])
    coefficient = augmented[:, -1]
    log_prediction = np.sum(
        test_design * coefficient, axis=-1)
    prediction = np.exp(np.clip(log_prediction, -40.0, 40.0))
    for k in range(K):
        prediction[:, k, k, :] = 0.0
    return prediction, {
        "coefficients": {
            "intercept": float(coefficient[0]),
            "log_sensing_power": float(coefficient[1]),
            "log_tx_range": float(coefficient[2]),
            "log_rx_range": float(coefficient[3]),
        },
        "expected_physics": {
            "log_sensing_power": 1.0,
            "log_tx_range": -2.0,
            "log_rx_range": -2.0,
        },
    }


def _probe_metrics(
    score: np.ndarray,
    label: np.ndarray,
) -> dict[str, object]:
    tx_score = score[..., 0]
    rx_score = score[..., 1]
    tx_label = label[..., 0].astype(bool)
    rx_label = label[..., 1].astype(bool)
    predicted_role = (
        rx_score.max(axis=-1) > tx_score.max(axis=-1)).astype(np.int8)
    teacher_role = (
        rx_label.any(axis=-1) > tx_label.any(axis=-1)).astype(np.int8)
    predicted_owner = rx_score.argmax(axis=1)
    teacher_owner = np.where(
        rx_label.any(axis=1),
        rx_label.argmax(axis=1),
        -1,
    )
    valid_owner = teacher_owner >= 0
    return {
        "tx_target": _binary_metrics(tx_score, tx_label),
        "rx_target": _binary_metrics(rx_score, rx_label),
        "role_accuracy": float(np.mean(
            predicted_role == teacher_role)),
        "team_role_exact_rate": float(np.mean(np.all(
            predicted_role == teacher_role, axis=1))),
        "receiver_owner_accuracy": float(np.mean(
            predicted_owner[valid_owner] == teacher_owner[valid_owner]))
        if np.any(valid_owner) else 0.0,
    }


def _previous_frame_indices(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
) -> np.ndarray:
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    index_by_key = {
        (int(seed), int(frame)): index
        for index, (seed, frame) in enumerate(zip(seeds, frames))
    }
    return np.asarray([
        index_by_key.get(
            (int(seeds[index]), int(frames[index]) - 1),
            -1,
        )
        for index in indices
    ], dtype=np.int64)


def _persistence_metrics(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, float]:
    previous = _previous_frame_indices(data, indices)
    valid = previous >= 0
    current_index = indices[valid]
    previous = previous[valid]
    current_role = data["teacher_role"][current_index]
    previous_role = data["teacher_role"][previous]
    current_owner = data["teacher_receiver_owner"][current_index]
    previous_owner = data["teacher_receiver_owner"][previous]
    current_pair = data["teacher_pair"][current_index].astype(bool)
    previous_pair = data["teacher_pair"][previous].astype(bool)
    intersection = np.sum(
        current_pair & previous_pair, axis=(1, 2, 3))
    union = np.sum(
        current_pair | previous_pair, axis=(1, 2, 3))
    return {
        "evaluated_resolve_frames": int(np.sum(valid)),
        "role_accuracy": float(np.mean(
            current_role == previous_role)),
        "team_role_exact_rate": float(np.mean(np.all(
            current_role == previous_role, axis=1))),
        "receiver_owner_accuracy": float(np.mean(
            current_owner == previous_owner)),
        "pair_jaccard": float(np.mean(
            intersection / np.maximum(union, 1))),
    }


def _previous_teacher_target_state(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    previous = _previous_frame_indices(data, indices)
    valid = previous >= 0
    previous_valid = previous[valid]
    state = np.stack([
        data["teacher_tx_target"][previous_valid],
        data["teacher_rx_target"][previous_valid],
    ], axis=-1).astype(np.float32)
    return state, valid


def _physical_audit(
    data: dict[str, np.ndarray],
    resolved: np.ndarray,
) -> dict[str, float]:
    d_eff = np.asarray(data["privileged_d_eff"], dtype=np.float64)
    owner = np.asarray(data["teacher_receiver_owner"], dtype=np.int64)
    positions = np.asarray(data["uav_positions"], dtype=np.float64)
    targets = np.asarray(data["target_states"], dtype=np.float64)[..., :2]
    sensing = np.asarray(data["sensing_weights"], dtype=np.float64)
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)

    incoming = d_eff.sum(axis=1)
    current_owner = incoming.argmax(axis=1)
    nearest_owner = np.linalg.norm(
        positions[:, :, None, :2] - targets[:, None, :, :],
        axis=-1,
    ).argmin(axis=1)
    sensing_owner = sensing.argmax(axis=1)
    valid = resolved[:, None] & (owner >= 0)

    lag_owner = np.full_like(owner, -1)
    rank_correlations = []
    index_by_key = {
        (int(seed), int(frame)): index
        for index, (seed, frame) in enumerate(zip(seeds, frames))
    }
    for index in np.flatnonzero(resolved):
        previous = index_by_key.get(
            (int(seeds[index]), int(frames[index]) - 1))
        if previous is None:
            continue
        lag_owner[index] = incoming[previous].argmax(axis=0)
        current_flat = d_eff[index].reshape(-1)
        lag_flat = d_eff[previous].reshape(-1)
        corr = _spearman(current_flat, lag_flat)
        if np.isfinite(corr):
            rank_correlations.append(float(corr))
    lag_valid = valid & (lag_owner >= 0)
    return {
        "current_privileged_owner_accuracy": float(np.mean(
            current_owner[valid] == owner[valid])),
        "one_frame_lag_owner_accuracy": float(np.mean(
            lag_owner[lag_valid] == owner[lag_valid]))
        if np.any(lag_valid) else 0.0,
        "geometry_nearest_owner_accuracy": float(np.mean(
            nearest_owner[valid] == owner[valid])),
        "sensing_argmax_owner_accuracy": float(np.mean(
            sensing_owner[valid] == owner[valid])),
        "same_edge_current_vs_lag_spearman": float(np.mean(
            rank_correlations)) if rank_correlations else 0.0,
    }


def _replay_teacher_projection(
    data: dict[str, np.ndarray],
    resolved: np.ndarray,
    *,
    frame_indices: np.ndarray | None = None,
    predicted_d_eff: np.ndarray | None = None,
    return_pairs: bool = False,
) -> dict[str, object]:
    required = {
        "coord_pd_ema",
        "coord_pd_ema_valid",
        "p_fa",
        "qos_floor",
        "deficit_priority_gain",
        "target_pair_limit",
        "reports_per_receiver",
    }
    if not required.issubset(data):
        return {
            "available": False,
            "exact_set_rate": 0.0,
            "edge_f1": 0.0,
        }
    candidate = data["privileged_candidate"].astype(bool)
    d_eff = np.asarray(data["privileged_d_eff"], dtype=np.float64)
    d_raw = np.asarray(data["privileged_d_raw"], dtype=np.float64)
    alpha = np.asarray(data["privileged_alpha"], dtype=np.float64)
    g_dd = np.asarray(data["privileged_g_dd"], dtype=np.float64)
    chi_rep = np.asarray(data["privileged_chi_rep"], dtype=np.float64)
    teacher_pair = data["teacher_pair"].astype(bool)
    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    floor = float(np.asarray(data["qos_floor"]).reshape(-1)[0])
    gain = float(
        np.asarray(data["deficit_priority_gain"]).reshape(-1)[0])
    pair_limit = int(
        np.asarray(data["target_pair_limit"]).reshape(-1)[0])
    reports = int(
        np.asarray(data["reports_per_receiver"]).reshape(-1)[0])
    exact = []
    role_exact = []
    owner_exact = []
    true_positive = false_positive = false_negative = 0
    replay_pd_rows = []
    teacher_pd_rows = []
    replay_pair_rows = []
    if frame_indices is None:
        frame_indices = np.flatnonzero(resolved)
    else:
        frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if predicted_d_eff is not None:
        predicted_d_eff = np.asarray(
            predicted_d_eff, dtype=np.float64)
        expected_shape = (
            len(frame_indices),
            K,
            K,
            Q,
        )
        if predicted_d_eff.shape != expected_shape:
            raise ValueError(
                f"predicted_d_eff must have shape {expected_shape}")
    for replay_index, frame_index in enumerate(frame_indices):
        entries = []
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                for q in range(Q):
                    if not candidate[frame_index, i, j, q]:
                        continue
                    entries.append(DeflectionEntry(
                        i=i,
                        j=j,
                        q=q,
                        tau=0.0,
                        nu=0.0,
                        alpha=float(alpha[frame_index, i, j, q]),
                        d_raw=float(d_raw[frame_index, i, j, q]),
                        g_dd=float(g_dd[frame_index, i, j, q]),
                        chi_rep=float(chi_rep[frame_index, i, j, q]),
                        d_eff=float(
                            predicted_d_eff[replay_index, i, j, q]
                            if predicted_d_eff is not None
                            else d_eff[frame_index, i, j, q]),
                    ))
        if bool(data["coord_pd_ema_valid"][frame_index]):
            priority = np.exp(np.clip(
                gain * (
                    floor
                    - np.asarray(
                        data["coord_pd_ema"][frame_index],
                        dtype=np.float64)
                ),
                -6.0,
                6.0,
            ))
        else:
            priority = np.ones(Q, dtype=np.float64)
        selected, _ = solve_maxmin_single_role_pairs(
            entries,
            num_uavs=K,
            num_targets=Q,
            target_pair_limit=pair_limit,
            reports_per_receiver=reports,
            p_fa=p_fa,
            p_d_floor=floor,
            target_priority=priority,
            fusion_mode="local_only",
        )
        replay_pair = np.zeros((K, K, Q), dtype=bool)
        for i, j, q in selected:
            replay_pair[int(i), int(j), int(q)] = True
        if return_pairs:
            replay_pair_rows.append(replay_pair.copy())
        expected = teacher_pair[frame_index]
        exact.append(bool(np.array_equal(replay_pair, expected)))
        true_positive += int(np.sum(replay_pair & expected))
        false_positive += int(np.sum(replay_pair & ~expected))
        false_negative += int(np.sum(~replay_pair & expected))
        replay_tx = replay_pair.any(axis=(1, 2))
        replay_rx = replay_pair.any(axis=(0, 2))
        expected_tx = expected.any(axis=(1, 2))
        expected_rx = expected.any(axis=(0, 2))
        role_exact.append(bool(
            np.array_equal(replay_tx, expected_tx)
            and np.array_equal(replay_rx, expected_rx)))
        replay_owner = replay_pair.any(axis=0).argmax(axis=0)
        expected_owner = expected.any(axis=0).argmax(axis=0)
        owner_exact.append(bool(np.array_equal(
            replay_owner, expected_owner)))
        realized_d = d_eff[frame_index]
        replay_receiver_d = np.sum(
            realized_d * replay_pair, axis=0)
        teacher_receiver_d = np.sum(
            realized_d * expected, axis=0)
        replay_pd_rows.append(compute_detection_probabilities(
            np.max(replay_receiver_d, axis=0),
            p_fa,
        ))
        teacher_pd_rows.append(compute_detection_probabilities(
            np.max(teacher_receiver_d, axis=0),
            p_fa,
        ))
    precision = true_positive / max(
        true_positive + false_positive, 1)
    recall = true_positive / max(
        true_positive + false_negative, 1)
    replay_pd = np.asarray(replay_pd_rows, dtype=np.float64)
    teacher_pd = np.asarray(teacher_pd_rows, dtype=np.float64)

    def pd_summary(values: np.ndarray) -> dict[str, float]:
        bottom = np.sort(values, axis=-1)
        weak_count = min(3, values.shape[-1])
        return {
            "mean": float(np.mean(values)),
            "weak3": float(np.mean(bottom[:, :weak_count])),
            "worst": float(np.mean(bottom[:, 0])),
            "qos_feasible_rate": float(np.mean(
                np.all(values >= floor, axis=-1))),
        }

    replay_summary = pd_summary(replay_pd)
    teacher_summary = pd_summary(teacher_pd)
    result: dict[str, object] = {
        "available": True,
        "exact_set_rate": float(np.mean(exact)),
        "role_exact_rate": float(np.mean(role_exact)),
        "owner_exact_rate": float(np.mean(owner_exact)),
        "edge_precision": float(precision),
        "edge_recall": float(recall),
        "edge_f1": float(
            2.0 * precision * recall
            / max(precision + recall, 1e-12)),
        "realized_pd": replay_summary,
        "teacher_realized_pd": teacher_summary,
        "realized_pd_gap": {
            key: float(replay_summary[key] - teacher_summary[key])
            for key in ("mean", "weak3", "worst", "qos_feasible_rate")
        },
    }
    if return_pairs:
        result["_pair_matrix"] = np.asarray(
            replay_pair_rows, dtype=np.uint8)
    return result


def audit(
    trace_path: Path,
    epochs: int,
    seed: int,
    student_output: Path | None = None,
    endpoint_dim: int = 32,
) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])
    resolved = np.asarray(data["p0_resolved"], dtype=bool)
    if not np.any(resolved):
        raise ValueError("trace contains no centralized P0 resolve frames")
    pair = data["teacher_pair"][resolved]
    tx = data["teacher_tx_target"][resolved]
    rx = data["teacher_rx_target"][resolved]
    endpoint_count = data["teacher_endpoint_target"][resolved].sum(axis=(1, 2))

    result: dict[str, object] = {
        "trace": str(trace_path),
        "frames": int(len(resolved)),
        "resolve_frames": int(np.sum(resolved)),
        "seeds": [int(value) for value in np.unique(data["seed"])],
        "teacher": {
            "selected_edges_mean": float(
                pair.sum(axis=(1, 2, 3)).mean()),
            "selected_edges_range": [
                int(pair.sum(axis=(1, 2, 3)).min()),
                int(pair.sum(axis=(1, 2, 3)).max()),
            ],
            "endpoints_per_team_mean": float(endpoint_count.mean()),
            "target_coverage": float(np.mean(
                pair.sum(axis=(1, 2)) > 0)),
            "multi_receiver_owner_rate": float(np.mean(
                data["teacher_receiver_owner"][resolved] == -2)),
            "selected_non_candidate_count": int(np.sum(
                (pair > 0)
                & (data["privileged_candidate"][resolved] == 0))),
        },
        "existing_protocol": {
            "outgoing_token_mask": _claim_metrics(
                data["outgoing_token_mask"][resolved], tx, rx, pair),
        },
        "privileged_physics": _physical_audit(data, resolved),
        "teacher_projection_replay": _replay_teacher_projection(
            data, resolved),
        "hold5_persistence": _persistence_metrics(
            data, np.flatnonzero(resolved)),
    }
    # Build the top-2 mask explicitly; comparing argsort indices directly is
    # a common and silent evaluation error.
    sensing = data["sensing_weights"][resolved]
    sensing_order = np.argsort(-sensing, axis=-1, kind="stable")
    sensing_top2 = np.zeros_like(sensing, dtype=bool)
    np.put_along_axis(
        sensing_top2,
        sensing_order[..., :min(2, Q)],
        True,
        axis=-1,
    )
    result["existing_protocol"]["sensing_top2"] = _claim_metrics(
        sensing_top2, tx, rx, pair)

    unique_seeds = np.unique(data["seed"])
    if unique_seeds.size >= 3:
        test_count = max(1, int(np.ceil(0.1 * unique_seeds.size)))
        validation_count = max(
            1, int(np.ceil(0.1 * unique_seeds.size)))
        if test_count + validation_count >= unique_seeds.size:
            validation_count = 1
            test_count = 1
        train_seed = unique_seeds[:-test_count]
        test_seed = unique_seeds[-test_count:]
        fit_seed = train_seed[:-validation_count]
        validation_seed = train_seed[-validation_count:]
        train_index = np.flatnonzero(
            resolved & np.isin(data["seed"], train_seed))
        fit_index = np.flatnonzero(
            resolved & np.isin(data["seed"], fit_seed))
        validation_index = np.flatnonzero(
            resolved & np.isin(data["seed"], validation_seed))
        test_index = np.flatnonzero(
            resolved & np.isin(data["seed"], test_seed))
        slices = _infer_observation_slices(
            data["local_obs"].shape[-1], K, Q)
        train_features = _per_target_features(
            data, "local_obs", train_index, slices)
        test_features = _per_target_features(
            data, "local_obs", test_index, slices)
        train_labels = np.stack([
            data["teacher_tx_target"][train_index],
            data["teacher_rx_target"][train_index],
        ], axis=-1)
        test_labels = np.stack([
            data["teacher_tx_target"][test_index],
            data["teacher_rx_target"][test_index],
        ], axis=-1)
        local_score = _fit_probe(
            train_features,
            train_labels,
            test_features,
            epochs=epochs,
            seed=seed,
        )
        team_mean_train = np.repeat(
            train_features.mean(axis=1, keepdims=True),
            K,
            axis=1,
        )
        team_max_train = np.repeat(
            train_features.max(axis=1, keepdims=True),
            K,
            axis=1,
        )
        team_mean_test = np.repeat(
            test_features.mean(axis=1, keepdims=True),
            K,
            axis=1,
        )
        team_max_test = np.repeat(
            test_features.max(axis=1, keepdims=True),
            K,
            axis=1,
        )
        central_train = np.concatenate(
            [train_features, team_mean_train, team_max_train], axis=-1)
        central_test = np.concatenate(
            [test_features, team_mean_test, team_max_test], axis=-1)
        central_score = _fit_probe(
            central_train,
            train_labels,
            central_test,
            epochs=epochs,
            seed=seed + 1,
        )
        fit_features = _per_target_features(
            data, "local_obs", fit_index, slices)
        validation_features = _per_target_features(
            data, "local_obs", validation_index, slices)
        fit_labels = np.stack([
            data["teacher_tx_target"][fit_index],
            data["teacher_rx_target"][fit_index],
        ], axis=-1)
        validation_labels = np.stack([
            data["teacher_tx_target"][validation_index],
            data["teacher_rx_target"][validation_index],
        ], axis=-1)
        fit_previous, fit_valid = _previous_teacher_target_state(
            data, fit_index)
        validation_previous, validation_valid = (
            _previous_teacher_target_state(data, validation_index))
        test_previous, test_valid = _previous_teacher_target_state(
            data, test_index)
        persistent_score = _fit_persistent_residual_probe(
            fit_features[fit_valid],
            fit_previous,
            fit_labels[fit_valid],
            validation_features[validation_valid],
            validation_previous,
            validation_labels[validation_valid],
            test_features[test_valid],
            test_previous,
            epochs=epochs,
            seed=seed + 2,
        )
        predicted_edge_value = _fit_factorized_edge_value_probe(
            fit_features,
            data["privileged_d_eff"][fit_index],
            data["teacher_pair"][fit_index],
            validation_features,
            data["privileged_d_eff"][validation_index],
            data["teacher_pair"][validation_index],
            test_features,
            endpoint_dim=endpoint_dim,
            epochs=epochs,
            seed=seed + 3,
        )
        edge_value_metrics = _edge_value_metrics(
            predicted_edge_value,
            data["privileged_d_eff"][test_index],
        )
        edge_projection_metrics = _replay_teacher_projection(
            data,
            resolved,
            frame_indices=test_index,
            predicted_d_eff=predicted_edge_value,
        )
        fit_bottleneck_weight = _teacher_bottleneck_target_weights(
            data, fit_index)
        validation_bottleneck_weight = (
            _teacher_bottleneck_target_weights(
                data, validation_index))
        qos_edge_value = _fit_factorized_edge_value_probe(
            fit_features,
            data["privileged_d_eff"][fit_index],
            data["teacher_pair"][fit_index],
            validation_features,
            data["privileged_d_eff"][validation_index],
            data["teacher_pair"][validation_index],
            test_features,
            train_target_weight=fit_bottleneck_weight,
            validation_target_weight=validation_bottleneck_weight,
            selected_edge_weight=5.0,
            selected_underprediction_weight=1.0,
            endpoint_dim=endpoint_dim,
            epochs=epochs,
            seed=seed + 5,
        )
        qos_value_metrics = _edge_value_metrics(
            qos_edge_value,
            data["privileged_d_eff"][test_index],
        )
        qos_projection_metrics = _replay_teacher_projection(
            data,
            resolved,
            frame_indices=test_index,
            predicted_d_eff=qos_edge_value,
        )
        qos_structured_edge_value = _fit_factorized_edge_value_probe(
            fit_features,
            data["privileged_d_eff"][fit_index],
            data["teacher_pair"][fit_index],
            validation_features,
            data["privileged_d_eff"][validation_index],
            data["teacher_pair"][validation_index],
            test_features,
            train_target_weight=fit_bottleneck_weight,
            validation_target_weight=validation_bottleneck_weight,
            selected_edge_weight=5.0,
            selected_underprediction_weight=1.0,
            selection_distribution_weight=0.05,
            endpoint_dim=endpoint_dim,
            artifact_output=student_output,
            artifact_rate_scale=max(
                float(np.max(data["outgoing_rate"])), 1.0),
            artifact_metadata={
                "trace": str(trace_path),
                "selection_rule": (
                    "projected worst/qos feasibility on held-out seeds"),
                "endpoint_dim": int(endpoint_dim),
                "fit_seeds": [int(value) for value in fit_seed],
                "validation_seeds": [
                    int(value) for value in validation_seed],
                "held_out_test_seeds": [
                    int(value) for value in test_seed],
            },
            epochs=epochs,
            seed=seed + 6,
        )
        qos_structured_value_metrics = _edge_value_metrics(
            qos_structured_edge_value,
            data["privileged_d_eff"][test_index],
        )
        qos_structured_projection_metrics = _replay_teacher_projection(
            data,
            resolved,
            frame_indices=test_index,
            predicted_d_eff=qos_structured_edge_value,
        )
        physics_edge_value, physics_parameters = (
            _physics_loglinear_edge_value(
                data,
                fit_index,
                test_index,
            ))
        physics_fit_value, _ = _physics_loglinear_edge_value(
            data,
            fit_index,
            fit_index,
        )
        physics_validation_value, _ = (
            _physics_loglinear_edge_value(
                data,
                fit_index,
                validation_index,
            ))
        physics_value_metrics = _edge_value_metrics(
            physics_edge_value,
            data["privileged_d_eff"][test_index],
        )
        physics_projection_metrics = _replay_teacher_projection(
            data,
            resolved,
            frame_indices=test_index,
            predicted_d_eff=physics_edge_value,
        )
        hybrid_edge_value = _fit_factorized_edge_value_probe(
            fit_features,
            data["privileged_d_eff"][fit_index],
            data["teacher_pair"][fit_index],
            validation_features,
            data["privileged_d_eff"][validation_index],
            data["teacher_pair"][validation_index],
            test_features,
            train_base_d_eff=physics_fit_value,
            validation_base_d_eff=physics_validation_value,
            test_base_d_eff=physics_edge_value,
            endpoint_dim=endpoint_dim,
            epochs=epochs,
            seed=seed + 4,
        )
        hybrid_value_metrics = _edge_value_metrics(
            hybrid_edge_value,
            data["privileged_d_eff"][test_index],
        )
        hybrid_projection_metrics = _replay_teacher_projection(
            data,
            resolved,
            frame_indices=test_index,
            predicted_d_eff=hybrid_edge_value,
        )
        copy_previous_score = test_previous
        result["offline_probe"] = {
            "split": {
                "fit_seeds": [int(value) for value in fit_seed],
                "validation_seeds": [
                    int(value) for value in validation_seed],
                "test_seeds": [int(value) for value in test_seed],
                "fit_resolve_frames": int(len(fit_index)),
                "validation_resolve_frames": int(
                    len(validation_index)),
                "test_resolve_frames": int(len(test_index)),
            },
            "local_predecision_shared_target_scorer": _probe_metrics(
                local_score, test_labels),
            "copy_previous_structure": _probe_metrics(
                copy_previous_score,
                test_labels[test_valid],
            ),
            "persistent_switch_residual_scorer": {
                "note": (
                    "teacher-forced previous state; validation may retain the "
                    "zero-residual persistence checkpoint"),
                **_probe_metrics(
                    persistent_score,
                    test_labels[test_valid],
                ),
            },
            "factorized_edge_value_scorer": {
                "note": (
                    "shared Tx/Rx endpoint encoders and a directed pair "
                    "decoder; projection uses the frozen teacher constraints"),
                **edge_value_metrics,
                "projected_structure": edge_projection_metrics,
            },
            "qos_weighted_factorized_edge_value_scorer": {
                "note": (
                    "continuous edge regression weighted toward same-state "
                    "teacher bottleneck targets and selected-edge "
                    "underprediction; no extra inference input"),
                **qos_value_metrics,
                "projected_structure": qos_projection_metrics,
            },
            "qos_structured_factorized_edge_value_scorer": {
                "note": (
                    "QoS-weighted continuous regression plus a small soft "
                    "teacher edge-distribution objective"),
                **qos_structured_value_metrics,
                "projected_structure": (
                    qos_structured_projection_metrics),
            },
            "physics_loglinear_edge_scorer": {
                "note": (
                    "calibrated sensing-power and bistatic inverse-range "
                    "control; no learned latent Token content"),
                **physics_parameters,
                **physics_value_metrics,
                "projected_structure": physics_projection_metrics,
            },
            "physics_residual_edge_scorer": {
                "note": (
                    "factorized endpoint network predicts a zero-initialized "
                    "residual over the calibrated bistatic inverse-range law"),
                **hybrid_value_metrics,
                "projected_structure": hybrid_projection_metrics,
            },
            "pooled_observation_probe_not_upper_bound": _probe_metrics(
                central_score, test_labels),
        }
    else:
        result["offline_probe"] = {
            "status": "not_run",
            "reason": "at least three independent seeds are required",
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--student-output", type=Path, default=None)
    parser.add_argument("--endpoint-dim", type=int, default=32)
    args = parser.parse_args()
    result = audit(
        args.trace,
        args.epochs,
        args.seed,
        student_output=args.student_output,
        endpoint_dim=args.endpoint_dim,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
