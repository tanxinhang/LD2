#!/usr/bin/env python
"""Train one cardinality-equivariant structure Student on several scales.

The training batches are never padded or concatenated across cardinalities.
Instead, one shared endpoint encoder/decoder is updated with an equal number
of batches from every scale.  Seed-grouped splits and a frozen 4/4
preservation teacher prevent scale fitting from silently consuming the
deployment test set or destroying an already safe policy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.audit_structure_teacher_trace import (
    _edge_value_metrics,
    _infer_observation_slices,
    _per_target_features,
    _replay_teacher_projection,
    _teacher_bottleneck_target_weights,
)
from uav_isac.agents.frozen_structure_student import (
    CardinalityResidualStructureStudent,
    FactorizedStructureStudent,
    FrozenStructureStudent,
    save_cardinality_residual_structure_student,
    save_frozen_structure_student,
)
from uav_isac.coordination.structure_regret import (
    dual_edge_weights,
    tstar_of,
)
from uav_isac.physical.detection import compute_detection_probabilities

# Gate 2 (advice/016 §4.1 + §6): torch-differentiable P_D for the task-regret
# hinge.  Q(x) = 0.5*erfc(x/sqrt(2)); Q^-1(p) = sqrt(2)*erfinv(1-2p).
import torch.special as _torch_special


def _torch_pd_from_deflection(D_q: torch.Tensor, p_fa: float) -> torch.Tensor:
    """Differentiable P_D = Q(Q^-1(P_FA) - sqrt(D)) over (..., Q)."""
    q_inv = float(np.sqrt(2.0) * _torch_special.erfinv(
        torch.as_tensor(1.0 - 2.0 * float(p_fa), dtype=torch.float64)))
    sqrt_d = torch.sqrt(torch.clamp(D_q, min=1.0e-10))
    z = (q_inv - sqrt_d) / float(np.sqrt(2.0))
    return 0.5 * torch.erfc(z)


@dataclass
class ScaleDataset:
    path: Path
    data: dict[str, np.ndarray]
    K: int
    Q: int
    split_seed: dict[str, np.ndarray]
    split_index: dict[str, np.ndarray]
    features: dict[str, np.ndarray]
    target: dict[str, np.ndarray]
    selected: dict[str, np.ndarray]
    target_weight: dict[str, np.ndarray]
    easy: dict[str, np.ndarray]
    preserve_log: dict[str, np.ndarray | None]
    preserve_protocol: dict[str, np.ndarray | None]
    excluded_seeds: np.ndarray
    # Gate 2 (advice/016 §6): per-frame edge task-sensitivity w_iq = pi_q*p_iq
    # and per-frame teacher max-min t* (deflection units) + feasibility.
    dual_weight: dict[str, np.ndarray] | None = None
    teacher_tstar: dict[str, np.ndarray] | None = None
    teacher_feasible: dict[str, np.ndarray] | None = None

    @property
    def name(self) -> str:
        return f"{self.K}/{self.Q}"


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _seed_grouped_split(
    data: dict[str, np.ndarray],
    excluded_seeds: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    resolved = np.asarray(data["p0_resolved"], dtype=bool)
    if excluded_seeds is not None and len(excluded_seeds):
        resolved &= ~np.isin(data["seed"], excluded_seeds)
    seeds = np.unique(np.asarray(data["seed"])[resolved])
    if len(seeds) < 5:
        raise ValueError("each scale trace needs at least five resolved seeds")
    test_count = max(1, int(np.ceil(0.1 * len(seeds))))
    validation_count = max(1, int(np.ceil(0.1 * len(seeds))))
    fit_count = len(seeds) - validation_count - test_count
    if fit_count < 3:
        raise ValueError("seed split leaves fewer than three fit seeds")
    split_seed = {
        "fit": seeds[:fit_count],
        "validation": seeds[fit_count:fit_count + validation_count],
        "test": seeds[fit_count + validation_count:],
    }
    split_index = {
        key: np.flatnonzero(
            resolved & np.isin(data["seed"], values))
        for key, values in split_seed.items()
    }
    return split_seed, split_index


def _teacher_easy_mask(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
) -> np.ndarray:
    d_eff = np.asarray(data["privileged_d_eff"][indices], dtype=np.float64)
    selected = np.asarray(data["teacher_pair"][indices], dtype=np.float64)
    receiver_d = np.sum(d_eff * selected, axis=1)
    target_d = np.max(receiver_d, axis=1)
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    target_pd = compute_detection_probabilities(target_d, p_fa)
    floor = float(np.asarray(data["qos_floor"]).reshape(-1)[0])
    return np.min(target_pd, axis=-1) >= floor


def _deflection_floor_for_pd(pd_floor: float, p_fa: float) -> float:
    """Invert P_D = Q(Q^-1(P_FA) - sqrt(D)): D = (Q^-1(P_FA) - Q^-1(P_D))^2."""
    from uav_isac.utils.math_utils import Q_inverse
    root = max(float(Q_inverse(np.asarray(p_fa)))
               - float(Q_inverse(np.asarray(pd_floor))), 0.0)
    return root ** 2


def _teacher_frame_gain(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
) -> np.ndarray:
    """Per-frame teacher per-watt gain matrix (K,Q) from the trace.

    Reconstructed from the trace's own physics: Deflection is linear in
    sensing power (``d_eff = a_iq * p_iq``), so the per-watt gain on the
    teacher-selected (i,owner_q,q) edge is ``a_iq = d_eff / p_iq``.  This
    works uniformly across traces/cardinalities (the raw chi/alpha/g_dd
    coefficient reconstruction degenerates on some traces, e.g. the 4/4
    gate100 trace stores near-zero alpha while d_eff is physical).  The
    global config scale is already inside d_eff, so t* and the dual weights
    are in the same physical deflection units the QoS floor uses.
    """
    deff = np.asarray(data["privileged_d_eff"], dtype=np.float64)
    weights = np.asarray(data["sensing_weights"], dtype=np.float64)
    comm_fraction = np.asarray(
        data.get("comm_fraction", np.zeros((len(deff), deff.shape[1]))),
        dtype=np.float64)
    budget = np.clip(1.0 - comm_fraction, 0.0, 1.0)
    pair = np.asarray(data["teacher_pair"], dtype=np.float64)
    owners = np.asarray(data["teacher_receiver_owner"], dtype=np.int64)
    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])
    gain = np.zeros((len(indices), K, Q), dtype=np.float64)
    for t, f in enumerate(indices):
        for q in range(Q):
            j = int(owners[f][q])
            if not (0 <= j < K):
                continue
            for i in range(K):
                if pair[f, i, j, q]:
                    # The trace stores normalized target weights, not watts.
                    # Actual p_iq is residual sensing budget times that weight.
                    denom = max(
                        float(budget[f, i] * weights[f, i, q]), 1.0e-9)
                    gain[t, i, q] += float(deff[f, i, j, q]) / denom
    return gain


def _make_dataset(
    path: Path,
    preservation_student: FrozenStructureStudent | None,
    excluded_seeds: np.ndarray | None = None,
    task_regret: bool = False,
) -> ScaleDataset:
    data = _load_trace(path)
    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])
    excluded = np.asarray(
        [] if excluded_seeds is None else excluded_seeds, dtype=np.int64)
    split_seed, split_index = _seed_grouped_split(data, excluded)
    slices = _infer_observation_slices(data["local_obs"].shape[-1], K, Q)
    features: dict[str, np.ndarray] = {}
    target: dict[str, np.ndarray] = {}
    selected: dict[str, np.ndarray] = {}
    target_weight: dict[str, np.ndarray] = {}
    easy: dict[str, np.ndarray] = {}
    preserve_log: dict[str, np.ndarray | None] = {}
    preserve_protocol: dict[str, np.ndarray | None] = {}
    dual_weight: dict[str, np.ndarray] = {}
    teacher_tstar: dict[str, np.ndarray] = {}
    teacher_feasible: dict[str, np.ndarray] = {}
    comm_fraction = np.asarray(
        data.get("comm_fraction", np.zeros((len(data["seed"]), K))),
        dtype=np.float64)
    budget_all = np.clip(1.0 - comm_fraction, 0.0, 1.0)
    if task_regret:
        gain_all = _teacher_frame_gain(data, np.arange(len(data["seed"])))
    preserve_here = (
        preservation_student is not None
        and preservation_student.num_uavs == K
        and preservation_student.num_targets == Q
    )
    for split, indices in split_index.items():
        features[split] = _per_target_features(
            data, "local_obs", indices, slices).astype(np.float32)
        target[split] = np.asarray(
            data["privileged_d_eff"][indices], dtype=np.float32)
        selected[split] = np.asarray(
            data["teacher_pair"][indices], dtype=np.float32)
        target_weight[split] = _teacher_bottleneck_target_weights(
            data, indices)
        easy[split] = _teacher_easy_mask(data, indices)
        preserve_log[split] = None
        preserve_protocol[split] = None
        if task_regret:
            gain = gain_all[indices]
            d_req = float(np.asarray(data["qos_floor"]).reshape(-1)[0])
            d_req_deflection = float(
                _deflection_floor_for_pd(d_req, float(data["p_fa"][0])))
            dual_weight[split] = np.stack([
                dual_edge_weights(
                    gain[t], budget_all[indices[t]], min_power_floor=1e-3)
                for t in range(len(indices))
            ], axis=0).astype(np.float32)
            teacher_tstar[split] = np.asarray([
                tstar_of(gain[t], budget_all[indices[t]])
                for t in range(len(indices))
            ], dtype=np.float32)
            # feasibility in deflection units: t*(A) >= d_req floor
            teacher_feasible[split] = np.asarray(
                teacher_tstar[split] >= d_req_deflection - 1e-9,
                dtype=np.float32)
        if preserve_here:
            preserve_log[split] = np.log1p(
                preservation_student.predict(features[split])).astype(
                    np.float32)
            preserve_protocol[split] = (
                preservation_student.encode_endpoint_protocol(
                    features[split]).astype(np.float32))
    return ScaleDataset(
        path=path,
        data=data,
        K=K,
        Q=Q,
        split_seed=split_seed,
        split_index=split_index,
        features=features,
        target=target,
        selected=selected,
        target_weight=target_weight,
        easy=easy,
        preserve_log=preserve_log,
        preserve_protocol=preserve_protocol,
        excluded_seeds=excluded,
        dual_weight=dual_weight if task_regret else None,
        teacher_tstar=teacher_tstar if task_regret else None,
        teacher_feasible=teacher_feasible if task_regret else None,
    )


def _balanced_normalization(
    datasets: list[ScaleDataset],
) -> tuple[np.ndarray, np.ndarray]:
    means = []
    second_moments = []
    for dataset in datasets:
        flat = dataset.features["fit"].reshape(
            -1, dataset.features["fit"].shape[-1]).astype(np.float64)
        means.append(np.mean(flat, axis=0))
        second_moments.append(np.mean(np.square(flat), axis=0))
    mean = np.mean(means, axis=0)
    variance = np.maximum(
        np.mean(second_moments, axis=0) - np.square(mean), 1.0e-10)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _initialize_from_preservation_student(
    model: FactorizedStructureStudent,
    student: FrozenStructureStudent,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> None:
    """Change input normalization without changing the frozen raw function."""
    if (
        model.input_dim != student.model.input_dim
        or model.hidden_dim != student.model.hidden_dim
        or model.endpoint_dim != student.model.endpoint_dim
    ):
        raise ValueError("preservation Student architecture does not match")
    state = {
        key: value.detach().clone()
        for key, value in student.model.state_dict().items()
    }
    old_mean = torch.as_tensor(
        student.feature_mean.reshape(-1), dtype=torch.float32)
    old_scale = torch.as_tensor(
        student.feature_scale.reshape(-1), dtype=torch.float32)
    new_mean = torch.as_tensor(feature_mean, dtype=torch.float32)
    new_scale = torch.as_tensor(feature_scale, dtype=torch.float32)
    ratio = new_scale / old_scale
    shift = (new_mean - old_mean) / old_scale
    for prefix in ("tx_encoder.0", "rx_encoder.0"):
        weight_key = f"{prefix}.weight"
        bias_key = f"{prefix}.bias"
        old_weight = state[weight_key].clone()
        state[weight_key] = old_weight * ratio[None, :]
        state[bias_key] = state[bias_key] + old_weight @ shift
    model.load_state_dict(state, strict=True)


def _tensor(values: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32)


def _batch_loss(
    model: torch.nn.Module,
    dataset: ScaleDataset,
    split: str,
    indices: np.ndarray,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    *,
    preservation_weight: float,
    protocol_preservation_weight: float,
    preservation_scope: str,
    preservation_endpoint_scale: torch.Tensor,
    selection_weight: float,
    dual_weight_coef: float = 0.0,
    task_regret_coef: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    feature = _tensor(dataset.features[split][indices])
    normalized = (feature - feature_mean) / feature_scale
    tx_endpoint, rx_endpoint = model.encode_endpoints(normalized)
    prediction = model.decode_edges(tx_endpoint, rx_endpoint)
    target = torch.log1p(_tensor(dataset.target[split][indices]).clamp_min(0))
    selected = _tensor(dataset.selected[split][indices])
    target_weight = _tensor(dataset.target_weight[split][indices])
    K = dataset.K
    valid = torch.as_tensor(~np.eye(K, dtype=bool))[None, :, :, None]
    valid = valid.expand_as(prediction)
    weight = target_weight[:, None, None, :] * (1.0 + 5.0 * selected)
    # Gate 2 (advice/016 §6): dual-consistent edge weights w_iq = pi_q*p_iq.
    # The envelope-theorem sensitivity is per (i,q); broadcast over the
    # receiver axis j.  Combined with the target weight and selection bias.
    if dual_weight_coef > 0.0 and dataset.dual_weight is not None:
        dual_w = _tensor(dataset.dual_weight[split][indices])  # (F,K,Q)
        weight = weight * (1.0 + dual_weight_coef * dual_w[:, :, None, :])
    under = (prediction < target).to(prediction.dtype) * selected
    weight = weight * (1.0 + under)
    error = torch.nn.functional.smooth_l1_loss(
        prediction, target, reduction="none")
    regression = (weight[valid] * error[valid]).sum() / (
        weight[valid].sum().clamp_min(1.0))

    F, _, _, Q = prediction.shape
    prediction_flat = prediction.permute(0, 3, 1, 2).reshape(F, Q, -1)
    target_flat = target.permute(0, 3, 1, 2).reshape(F, Q, -1)
    selected_flat = selected.permute(0, 3, 1, 2).reshape(F, Q, -1)
    edge_valid = valid.permute(0, 3, 1, 2).reshape(F, Q, -1)
    prediction_logits = prediction_flat.masked_fill(~edge_valid, -1.0e4)
    teacher_logits = (target_flat + selected_flat).masked_fill(
        ~edge_valid, -1.0e4)
    teacher_distribution = torch.softmax(
        teacher_logits.detach(), dim=-1)
    cross_entropy = -torch.sum(
        teacher_distribution * torch.log_softmax(prediction_logits, dim=-1),
        dim=-1)
    selection = (cross_entropy * target_weight).sum() / (
        target_weight.sum().clamp_min(1.0))

    # Gate 2 (advice/016 §4.1): lexicographic task-regret surrogate.
    #   R_gamma = mean over frames where the teacher is feasible of
    #             relu(d_req - t*(student structure)) in P_D space
    #   R_t     = mean over frames where BOTH feasible of
    #             relu(t*(teacher) - t*(student)) in P_D space
    # P_D-domain: d_req is the deflection floor for the QoS P_D floor;
    # the per-frame teacher max-min t* is precomputed from the trace.
    task_regret = prediction.new_zeros(())
    if task_regret_coef > 0.0 and dataset.teacher_tstar is not None:
        p_fa = float(np.asarray(dataset.data["p_fa"]).reshape(-1)[0])
        qos_floor_pd = float(np.asarray(
            dataset.data["qos_floor"]).reshape(-1)[0])
        d_req = float(_deflection_floor_for_pd(
            qos_floor_pd, p_fa))
        # Differentiable Student-decoded structure: select one edge per target
        # with a softmax relaxation of the deployment argmax.  The previous
        # implementation multiplied by the TEACHER selected mask, so its
        # purported Student regret could not penalize a Student structure flip.
        pred_d = torch.expm1(prediction.clamp_min(0.0)).clamp_min(0.0)
        pred_d_flat = pred_d.permute(0, 3, 1, 2).reshape(F, Q, -1)
        soft_structure = torch.softmax(prediction_logits, dim=-1)
        target_d = torch.sum(soft_structure * pred_d_flat, dim=-1)
        student_pd = _torch_pd_from_deflection(target_d, p_fa)
        student_min_pd = student_pd.min(dim=-1).values  # (F,)
        teacher_t = _tensor(dataset.teacher_tstar[split][indices])
        teacher_feas = _tensor(dataset.teacher_feasible[split][indices])
        # R_gamma: teacher feasible but student drops below the QoS floor.
        floor_mask = teacher_feas
        r_gamma = (
            floor_mask
            * torch.relu(qos_floor_pd - student_min_pd)
        ).mean()
        # R_t: both feasible; residual max-min loss in P_D space.  The
        # teacher's max-min t* is in deflection units; invert to P_D so the
        # regret is comparable to the student's P_D.
        teacher_pd = _torch_pd_from_deflection(
            teacher_t.clamp_min(0.0), p_fa)
        r_t = (
            teacher_feas
            * (student_min_pd >= qos_floor_pd).to(prediction.dtype)
            * torch.relu(teacher_pd - student_min_pd)
        ).mean()
        task_regret = task_regret_coef * (r_gamma + r_t)

    preservation = prediction.new_zeros(())
    protocol_preservation = prediction.new_zeros(())
    preserve_target = dataset.preserve_log[split]
    preserve_protocol = dataset.preserve_protocol[split]
    preserve_frame = (
        np.ones(len(indices), dtype=bool)
        if preservation_scope == "all"
        else dataset.easy[split][indices]
    )
    if preserve_target is not None and np.any(preserve_frame):
        old = _tensor(preserve_target[indices])
        frame_mask = torch.as_tensor(
            preserve_frame, dtype=torch.bool)[:, None, None, None]
        preserve_mask = valid & frame_mask.expand_as(valid)
        preservation = torch.nn.functional.smooth_l1_loss(
            prediction[preserve_mask], old[preserve_mask])
        if preserve_protocol is not None:
            protocol = torch.cat(
                [tx_endpoint, rx_endpoint], dim=-1)
            protocol = torch.clamp(
                protocol / preservation_endpoint_scale, -1.0, 1.0)
            protocol_target = _tensor(preserve_protocol[indices])
            protocol_preservation = torch.nn.functional.smooth_l1_loss(
                protocol[torch.as_tensor(preserve_frame)],
                protocol_target[torch.as_tensor(preserve_frame)],
            )
    total = (
        regression
        + float(selection_weight) * selection
        + task_regret
        + float(preservation_weight) * preservation
        + float(protocol_preservation_weight) * protocol_preservation
    )
    return total, {
        "regression": float(regression.detach()),
        "selection": float(selection.detach()),
        "task_regret": float(task_regret.detach()),
        "preservation": float(preservation.detach()),
        "protocol_preservation": float(protocol_preservation.detach()),
    }


def _predict(
    model: torch.nn.Module,
    features: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    rows = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            feature = _tensor(features[start:start + batch_size])
            log_value = model(
                (feature - _tensor(feature_mean).reshape(1, 1, 1, -1))
                / _tensor(feature_scale).reshape(1, 1, 1, -1))
            rows.append(torch.expm1(log_value).clamp_min(0).cpu().numpy())
    prediction = np.concatenate(rows, axis=0)
    for k in range(min(prediction.shape[1], prediction.shape[2])):
        prediction[:, k, k, :] = 0.0
    return prediction


def _projection_report(
    model: torch.nn.Module,
    datasets: list[ScaleDataset],
    split: str,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        prediction = _predict(
            model,
            dataset.features[split],
            feature_mean,
            feature_scale,
            batch_size,
        )
        report[dataset.name] = {
            "edge_value": _edge_value_metrics(
                prediction, dataset.target[split]),
            "projection": _replay_teacher_projection(
                dataset.data,
                np.asarray(dataset.data["p0_resolved"], dtype=bool),
                frame_indices=dataset.split_index[split],
                predicted_d_eff=prediction,
            ),
            "seeds": [int(value) for value in dataset.split_seed[split]],
            "frames": int(len(prediction)),
            "teacher_easy_fraction": float(np.mean(dataset.easy[split])),
        }
    return report


def _selection_key(report: dict[str, dict[str, Any]]) -> tuple[float, ...]:
    feasible = [
        float(value["projection"]["realized_pd"]["qos_feasible_rate"])
        for value in report.values()
    ]
    worst = [
        float(value["projection"]["realized_pd"]["worst"])
        for value in report.values()
    ]
    # Scale robustness is lexicographic: protect the weakest cardinality first.
    return (
        min(feasible),
        min(worst),
        float(np.mean(feasible)),
        float(np.mean(worst)),
    )


def _endpoint_scale(
    model: torch.nn.Module,
    datasets: list[ScaleDataset],
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    group_scales = []
    mean = _tensor(feature_mean).reshape(1, 1, 1, -1)
    scale = _tensor(feature_scale).reshape(1, 1, 1, -1)
    with torch.inference_mode():
        for dataset in datasets:
            samples = []
            features = dataset.features["fit"]
            for start in range(0, len(features), batch_size):
                values = (_tensor(features[start:start + batch_size]) - mean) / scale
                tx, rx = model.encode_endpoints(values)
                samples.append(torch.cat([tx, rx], dim=-1).reshape(
                    -1, 2 * model.endpoint_dim))
            endpoints = torch.cat(samples, dim=0)
            group_scales.append(torch.quantile(
                endpoints.abs(), 0.995, dim=0).clamp_min(1.0e-5))
    return torch.stack(group_scales).max(dim=0).values.cpu().numpy()


def train(
    trace_paths: list[Path],
    output: Path,
    report_path: Path,
    preservation_checkpoint: Path,
    *,
    epochs: int,
    checkpoint_every: int,
    batch_size: int,
    learning_rate: float,
    preservation_weight: float,
    protocol_preservation_weight: float,
    preservation_scope: str,
    selection_weight: float,
    recalibrate_endpoint_scale: bool,
    cardinality_residual: bool,
    seed: int,
    dual_weight_coef: float = 0.0,
    task_regret_coef: float = 0.0,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    preservation_student = FrozenStructureStudent(preservation_checkpoint)
    if preservation_scope not in {"easy", "all"}:
        raise ValueError("preservation_scope must be 'easy' or 'all'")
    datasets = []
    used_seed_values: set[int] = set()
    for path in trace_paths:
        header = _load_trace(path)
        resolved = np.asarray(header["p0_resolved"], dtype=bool)
        available = {
            int(value) for value in np.unique(header["seed"][resolved])}
        excluded = np.asarray(
            sorted(available & used_seed_values), dtype=np.int64)
        datasets.append(_make_dataset(
            path, preservation_student, excluded_seeds=excluded,
            task_regret=(dual_weight_coef > 0.0 or task_regret_coef > 0.0)))
        used_seed_values.update(available - set(excluded.tolist()))
    cardinalities = {(data.K, data.Q) for data in datasets}
    if len(cardinalities) != len(datasets):
        raise ValueError("trace cardinalities must be unique")
    feature_dims = {data.features["fit"].shape[-1] for data in datasets}
    if len(feature_dims) != 1:
        raise ValueError("per-target feature width differs across scales")
    base_dataset = next(
        data for data in datasets
        if data.K == preservation_student.num_uavs
        and data.Q == preservation_student.num_targets
    )
    if cardinality_residual:
        if preservation_student.schema_version != 1:
            raise ValueError(
                "cardinality residual currently requires a schema-v1 anchor")
        non_anchor = [
            data for data in datasets
            if (data.K, data.Q) != (base_dataset.K, base_dataset.Q)
        ]
        if not non_anchor:
            raise ValueError(
                "cardinality residual requires a non-anchor trace")
        reference = min(
            non_anchor,
            key=lambda data: (
                abs(data.K - base_dataset.K)
                + abs(data.Q - base_dataset.Q)),
        )
        feature_mean = preservation_student.feature_mean.reshape(-1).copy()
        feature_scale = preservation_student.feature_scale.reshape(-1).copy()
        model = CardinalityResidualStructureStudent(
            input_dim=feature_mean.size,
            hidden_dim=preservation_student.model.hidden_dim,
            endpoint_dim=preservation_student.model.endpoint_dim,
            anchor_num_uavs=base_dataset.K,
            anchor_num_targets=base_dataset.Q,
            reference_num_uavs=reference.K,
            reference_num_targets=reference.Q,
        )
        model.base.load_state_dict(
            preservation_student.model.state_dict(), strict=True)
        model.freeze_base()
        training_datasets = non_anchor
    else:
        feature_mean, feature_scale = _balanced_normalization(datasets)
        model = FactorizedStructureStudent(
            input_dim=feature_mean.size,
            hidden_dim=preservation_student.model.hidden_dim,
            endpoint_dim=preservation_student.model.endpoint_dim,
        )
        _initialize_from_preservation_student(
            model, preservation_student, feature_mean, feature_scale)
        training_datasets = datasets
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad],
        lr=float(learning_rate),
        weight_decay=1.0e-4,
    )
    mean_tensor = _tensor(feature_mean).reshape(1, 1, 1, -1)
    scale_tensor = _tensor(feature_scale).reshape(1, 1, 1, -1)
    preservation_endpoint_scale = _tensor(
        preservation_student.endpoint_scale).reshape(1, 1, 1, -1)
    generator = np.random.default_rng(int(seed))

    history = []
    validation = _projection_report(
        model, datasets, "validation", feature_mean, feature_scale, batch_size)
    best_key = _selection_key(validation)
    best_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    best_epoch = 0
    history.append({
        "epoch": 0,
        "selection_key": list(best_key),
        "validation": validation,
    })

    steps = max(
        int(np.ceil(len(data.features["fit"]) / batch_size))
        for data in training_datasets)
    orders = {
        data.name: generator.permutation(len(data.features["fit"]))
        for data in training_datasets
    }
    positions = {data.name: 0 for data in training_datasets}
    for epoch in range(1, max(1, int(epochs)) + 1):
        epoch_terms = []
        for _ in range(steps):
            for dataset_index in generator.permutation(
                    len(training_datasets)):
                dataset = training_datasets[int(dataset_index)]
                position = positions[dataset.name]
                order = orders[dataset.name]
                if position + batch_size > len(order):
                    order = generator.permutation(len(dataset.features["fit"]))
                    orders[dataset.name] = order
                    position = 0
                index = order[position:position + batch_size]
                positions[dataset.name] = position + len(index)
                optimizer.zero_grad()
                loss, terms = _batch_loss(
                    model,
                    dataset,
                    "fit",
                    index,
                    mean_tensor,
                    scale_tensor,
                    preservation_weight=preservation_weight,
                    protocol_preservation_weight=(
                        protocol_preservation_weight),
                    preservation_scope=preservation_scope,
                    preservation_endpoint_scale=(
                        preservation_endpoint_scale),
                    selection_weight=selection_weight,
                    dual_weight_coef=dual_weight_coef,
                    task_regret_coef=task_regret_coef,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                epoch_terms.append(float(loss.detach()))
        if epoch % max(1, checkpoint_every) == 0 or epoch == epochs:
            validation = _projection_report(
                model,
                datasets,
                "validation",
                feature_mean,
                feature_scale,
                batch_size,
            )
            key = _selection_key(validation)
            history.append({
                "epoch": int(epoch),
                "train_loss": float(np.mean(epoch_terms)),
                "selection_key": list(key),
                "validation": validation,
            })
            if key > best_key:
                best_key = key
                best_epoch = int(epoch)
                best_state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }
            print({
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_terms)),
                "selection_key": key,
                "best_epoch": best_epoch,
            })

    model.load_state_dict(best_state)
    endpoint_scale = (
        _endpoint_scale(
            model, datasets, feature_mean, feature_scale, batch_size)
        if recalibrate_endpoint_scale
        else preservation_student.endpoint_scale.reshape(-1).copy()
    )
    rate_scale = max(
        float(np.max(data.data["outgoing_rate"])) for data in datasets)
    artifact_metadata = {
            "objective": (
                "anchor_preserving_cardinality_residual"
                if cardinality_residual
                else "balanced_multicardinality_edge_value"),
            "trained_cardinalities": [
                {"K": data.K, "Q": data.Q, "trace": str(data.path)}
                for data in datasets
            ],
            "preservation_checkpoint": str(preservation_checkpoint),
            "preservation_weight": float(preservation_weight),
            "protocol_preservation_weight": float(
                protocol_preservation_weight),
            "preservation_scope": str(preservation_scope),
            "endpoint_scale_source": (
                "mixed_fit_quantile" if recalibrate_endpoint_scale
                else "preservation_checkpoint"),
            "selection_distribution_weight": float(selection_weight),
            "checkpoint_selection": (
                "min_scale_qos,min_scale_worst,mean_qos,mean_worst"),
            "best_epoch": int(best_epoch),
            "seed": int(seed),
        }
    save_arguments = {
        "output": output,
        "model": model,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "num_uavs": base_dataset.K,
        "num_targets": base_dataset.Q,
        "rate_scale": max(rate_scale, 1.0),
        "endpoint_scale": endpoint_scale,
        "metadata": artifact_metadata,
    }
    if cardinality_residual:
        save_cardinality_residual_structure_student(**save_arguments)
    else:
        save_frozen_structure_student(**save_arguments)
    result = {
        "output": str(output),
        "best_epoch": int(best_epoch),
        "best_selection_key": list(best_key),
        "feature_dim": int(feature_mean.size),
        "endpoint_dim": int(model.endpoint_dim),
        "cardinality_residual": bool(cardinality_residual),
        "datasets": {
            data.name: {
                "trace": str(data.path),
                "split_seeds": {
                    split: [int(value) for value in values]
                    for split, values in data.split_seed.items()
                },
                "split_frames": {
                    split: int(len(values))
                    for split, values in data.split_index.items()
                },
                "excluded_overlapping_seed_values": [
                    int(value) for value in data.excluded_seeds],
            }
            for data in datasets
        },
        "history": history,
        "test": _projection_report(
            model, datasets, "test", feature_mean, feature_scale, batch_size),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--preservation-checkpoint", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--preservation-weight", type=float, default=0.5)
    parser.add_argument(
        "--protocol-preservation-weight", type=float, default=0.5)
    parser.add_argument(
        "--preservation-scope", choices=("easy", "all"), default="easy")
    parser.add_argument("--selection-weight", type=float, default=0.05)
    parser.add_argument("--recalibrate-endpoint-scale", action="store_true")
    parser.add_argument("--cardinality-residual", action="store_true")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--dual-weight-coef", type=float, default=0.0,
        help="Gate 2 (advice/016 §6): envelope dual edge weights "
             "w_iq = pi_q*p_iq multiplier on the regression loss")
    parser.add_argument(
        "--task-regret-coef", type=float, default=0.0,
        help="Gate 2 (advice/016 §4.1): lexicographic task-regret "
             "(R_gamma feasibility-flip + R_t max-min residual) multiplier")
    args = parser.parse_args()
    result = train(
        args.trace,
        args.output,
        args.report,
        args.preservation_checkpoint,
        epochs=args.epochs,
        checkpoint_every=args.checkpoint_every,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        preservation_weight=args.preservation_weight,
        protocol_preservation_weight=args.protocol_preservation_weight,
        preservation_scope=args.preservation_scope,
        selection_weight=args.selection_weight,
        recalibrate_endpoint_scale=args.recalibrate_endpoint_scale,
        cardinality_residual=args.cardinality_residual,
        seed=args.seed,
        dual_weight_coef=args.dual_weight_coef,
        task_regret_coef=args.task_regret_coef,
    )
    print(json.dumps({
        "output": result["output"],
        "best_epoch": result["best_epoch"],
        "best_selection_key": result["best_selection_key"],
    }, indent=2))


if __name__ == "__main__":
    main()
