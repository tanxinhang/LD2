"""Factorized spatio-relational GNN for next-frame hyperedge prefetch.

The model proposes structure and solver warm starts.  It never certifies its
own output: exact selected-edge physics, omitted-edge upper bounds, LP
residuals and the incumbent fallback remain analytical responsibilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional


PREDICTIVE_CHECKPOINT_SCHEMA = "predictive-gnn-shadow/v2-temporal-residual"


def audit_predictive_checkpoint(
    payload: Mapping[str, Any],
    model: nn.Module,
) -> dict[str, Any]:
    """Audit schema and tensor compatibility without mutating ``model``."""
    issues: list[str] = []
    schema = str(payload.get("schema_version", "MISSING"))
    if schema != PREDICTIVE_CHECKPOINT_SCHEMA:
        issues.append(
            f"unsupported schema {schema!r}; expected "
            f"{PREDICTIVE_CHECKPOINT_SCHEMA!r}")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        issues.append("state_dict is missing or is not a mapping")
        state = {}
    expected = model.state_dict()
    missing = sorted(set(expected).difference(state))
    unexpected = sorted(set(state).difference(expected))
    shape_mismatches = sorted(
        key for key in set(expected).intersection(state)
        if not isinstance(state[key], torch.Tensor)
        or tuple(state[key].shape) != tuple(expected[key].shape)
    )
    if missing:
        issues.append("missing model tensors: " + ", ".join(missing))
    if unexpected:
        issues.append("unexpected model tensors: " + ", ".join(unexpected))
    if shape_mismatches:
        issues.append(
            "shape-incompatible model tensors: "
            + ", ".join(shape_mismatches))
    return {
        "schema_version": "predictive-checkpoint-audit/v1",
        "checkpoint_schema": schema,
        "expected_checkpoint_schema": PREDICTIVE_CHECKPOINT_SCHEMA,
        "compatible": not issues,
        "issues": issues,
        "missing_tensors": missing,
        "unexpected_tensors": unexpected,
        "shape_mismatches": shape_mismatches,
    }


def require_compatible_predictive_checkpoint(
    payload: Mapping[str, Any],
    model: nn.Module,
) -> dict[str, Any]:
    """Fail before execution when a predictive checkpoint is from another API."""
    audit = audit_predictive_checkpoint(payload, model)
    if not audit["compatible"]:
        raise ValueError(
            "predictive GNN checkpoint is incompatible: "
            + "; ".join(audit["issues"]))
    return audit


@dataclass(frozen=True)
class FactorizedPrediction:
    """One batched factorized prediction without a pair-dense tensor."""

    owner_logits: torch.Tensor  # (B,K,Q)
    transmitter_logits: torch.Tensor  # (B,K,Q)
    power_logits: torch.Tensor  # (B,K,Q), projected onto row budgets downstream
    dual_warm_start: torch.Tensor  # (B,Q)
    risk_radius: torch.Tensor  # (B,Q)
    detection_logits: torch.Tensor  # (B,Q), next-frame actual P_D forecast
    detection_delta: torch.Tensor  # (B,Q), bounded temporal P_D change
    endpoint_embedding: torch.Tensor  # (B,K,Q,H), used for sparse pair decoding


@dataclass(frozen=True)
class PrefetchedPrediction:
    """Immutable prediction tagged for the frame that may consume it."""

    source_frame: int
    target_frame: int
    prediction: FactorizedPrediction


def build_endpoint_features(
    endpoint_position_xy: np.ndarray,
    endpoint_velocity_xy: np.ndarray,
    target_position: np.ndarray,
    target_velocity: np.ndarray,
    target_position_uncertainty: np.ndarray,
    target_velocity_uncertainty: np.ndarray,
    visible: np.ndarray,
    *,
    distance_scale_m: float,
    velocity_scale_mps: float,
) -> np.ndarray:
    """Build finite public-state-only endpoint tokens ``(B,K,Q,9)``."""
    endpoint_position = np.asarray(endpoint_position_xy, dtype=np.float64)
    endpoint_velocity = np.asarray(endpoint_velocity_xy, dtype=np.float64)
    targets = np.asarray(target_position, dtype=np.float64)
    target_velocities = np.asarray(target_velocity, dtype=np.float64)
    position_radius = np.asarray(
        target_position_uncertainty, dtype=np.float64)
    velocity_radius = np.asarray(
        target_velocity_uncertainty, dtype=np.float64)
    seen = np.asarray(visible, dtype=bool)
    if endpoint_position.ndim != 4 or endpoint_position.shape[-1] != 2:
        raise ValueError("endpoint positions must have shape (B,K,Q,2)")
    B, K, Q, _ = endpoint_position.shape
    if (
        endpoint_velocity.shape != endpoint_position.shape
        or targets.shape != (B, Q, 3)
        or target_velocities.shape != (B, Q, 3)
        or position_radius.shape != (B, Q)
        or velocity_radius.shape != (B, Q)
        or seen.shape != (B, K, Q)
    ):
        raise ValueError("endpoint feature shapes are inconsistent")
    distance_scale = float(distance_scale_m)
    velocity_scale = float(velocity_scale_mps)
    if distance_scale <= 0.0 or velocity_scale <= 0.0:
        raise ValueError("feature scales must be positive")
    delta_position = targets[:, None, :, :2] - endpoint_position
    delta_velocity = (
        target_velocities[:, None, :, :2] - endpoint_velocity)
    distance = np.linalg.norm(delta_position, axis=-1)
    features = np.stack([
        delta_position[..., 0] / distance_scale,
        delta_position[..., 1] / distance_scale,
        delta_velocity[..., 0] / velocity_scale,
        delta_velocity[..., 1] / velocity_scale,
        np.log1p(distance) / np.log1p(distance_scale),
        seen.astype(np.float64),
        np.broadcast_to(
            position_radius[:, None, :] / distance_scale, (B, K, Q)),
        np.broadcast_to(
            velocity_radius[:, None, :] / velocity_scale, (B, K, Q)),
        1.0 / (1.0 + (distance / distance_scale) ** 2),
    ], axis=-1)
    if np.any(~np.isfinite(features)):
        raise ValueError("endpoint features must be finite")
    return features.astype(np.float32)


def augment_temporal_protocol_features(
    endpoint_features: np.ndarray,
    edge_index: np.ndarray,
    edge_mask: np.ndarray,
    frame: np.ndarray,
    nominal_gain: np.ndarray,
    lower_gain: np.ndarray,
    upper_gain: np.ndarray,
    power: np.ndarray,
    dual_price: np.ndarray,
    *,
    hold_period: int = 5,
    power_scale_w: float = 0.0251,
) -> np.ndarray:
    """Add causal protocol/certificate state for next-frame residual learning.

    Inputs include a leading frame dimension. No future label is included.
    The result has 18 features and remains ``(frame,viewer,node,target,F)``.
    """
    base = np.asarray(endpoint_features, dtype=np.float32)
    if base.ndim != 5 or base.shape[-1] != 9:
        raise ValueError("endpoint_features must have shape (F,V,K,Q,9)")
    frames, viewers, nodes, targets, _ = base.shape
    selected = np.asarray(edge_index)
    selected_mask = np.asarray(edge_mask, dtype=bool)
    if selected.shape[:2] != selected_mask.shape or selected.shape[-1] != 3:
        raise ValueError("edge index/mask shapes are inconsistent")
    owner = np.zeros((frames, nodes, targets), dtype=np.float32)
    transmitter = np.zeros_like(owner)
    for sample in range(frames):
        for tx, rx, target in selected[sample][selected_mask[sample]]:
            owner[sample, int(rx), int(target)] = 1.0
            transmitter[sample, int(tx), int(target)] = 1.0
    owner = np.broadcast_to(owner[:, None], (frames, viewers, nodes, targets))
    transmitter = np.broadcast_to(
        transmitter[:, None], (frames, viewers, nodes, targets))
    period = max(int(hold_period), 1)
    phase = np.mod(np.asarray(frame, dtype=np.int64), period)
    boundary = (phase == 0).astype(np.float32)
    phase = phase.astype(np.float32) / float(period)
    boundary = np.broadcast_to(boundary[:, None, None, None], owner.shape)
    phase = np.broadcast_to(phase[:, None, None, None], owner.shape)

    def log_scaled(array: np.ndarray, maximum: float) -> np.ndarray:
        value = np.asarray(array, dtype=np.float64)
        if value.shape != (frames, viewers, nodes, targets):
            raise ValueError("certificate feature shape mismatch")
        return (np.log1p(np.maximum(value, 0.0)) / np.log1p(maximum)).astype(
            np.float32)

    nominal = log_scaled(nominal_gain, 1.0e8)
    lower = log_scaled(lower_gain, 1.0e8)
    upper = log_scaled(upper_gain, 1.0e8)
    power_feature = np.asarray(power, dtype=np.float32) / float(power_scale_w)
    if power_feature.shape != owner.shape:
        raise ValueError("power feature shape mismatch")
    dual = np.asarray(dual_price, dtype=np.float32)
    if dual.shape != (frames, viewers, targets):
        raise ValueError("dual price shape mismatch")
    dual = np.broadcast_to(
        np.log1p(np.maximum(dual, 0.0))[:, :, None, :] / np.log(2.0),
        owner.shape,
    )
    result = np.concatenate([
        base,
        owner[..., None],
        transmitter[..., None],
        boundary[..., None],
        phase[..., None],
        nominal[..., None],
        lower[..., None],
        upper[..., None],
        power_feature[..., None],
        dual[..., None],
    ], axis=-1)
    if np.any(~np.isfinite(result)):
        raise ValueError("temporal protocol features must be finite")
    return result.astype(np.float32, copy=False)


def append_causal_feature_residual(
    current: np.ndarray,
    previous: np.ndarray,
    history_valid: np.ndarray,
) -> np.ndarray:
    """Append a one-step residual without leaking state across episodes."""
    current_array = np.asarray(current, dtype=np.float32)
    previous_array = np.asarray(previous, dtype=np.float32)
    valid = np.asarray(history_valid, dtype=np.float32)
    if current_array.shape != previous_array.shape or current_array.ndim != 5:
        raise ValueError("current/previous features must share shape (F,V,K,Q,D)")
    if valid.shape != (current_array.shape[0],):
        raise ValueError("history_valid must have one value per frame")
    if np.any((valid < 0.0) | (valid > 1.0)):
        raise ValueError("history_valid must lie in [0,1]")
    gate = valid[:, None, None, None, None]
    residual = (current_array - previous_array) * gate
    flag = np.broadcast_to(gate, current_array.shape[:-1] + (1,))
    return np.concatenate([current_array, residual, flag], axis=-1).astype(
        np.float32, copy=False)


class CertifiedBipartiteGNN(nn.Module):
    """Small permutation-equivariant UAV--target message-passing model."""

    def __init__(
        self,
        feature_dim: int = 9,
        hidden_dim: int = 32,
        message_rounds: int = 2,
        boundary_feature_index: int | None = None,
        boundary_power_residual: bool = True,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or hidden_dim < 4 or message_rounds < 1:
            raise ValueError("invalid predictive GNN dimensions")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.message_rounds = int(message_rounds)
        if boundary_feature_index is not None and not (
            0 <= int(boundary_feature_index) < self.feature_dim
        ):
            raise ValueError("boundary feature index is out of range")
        self.boundary_feature_index = (
            None if boundary_feature_index is None
            else int(boundary_feature_index)
        )
        self.boundary_power_residual = bool(boundary_power_residual)
        self.endpoint_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.message_updates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3 * self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            for _ in range(self.message_rounds)
        ])
        self.message_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim)
            for _ in range(self.message_rounds)
        ])
        self.owner_head = nn.Linear(2 * self.hidden_dim, 1)
        self.transmitter_head = nn.Linear(2 * self.hidden_dim, 1)
        self.power_head = nn.Linear(2 * self.hidden_dim, 1)
        self.receiver_query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.transmitter_key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dual_head = nn.Linear(self.hidden_dim, 1)
        self.risk_head = nn.Linear(self.hidden_dim, 1)
        self.detection_head = nn.Linear(self.hidden_dim, 1)
        self.detection_delta_head = nn.Linear(self.hidden_dim, 1)
        # Refresh frames have a discontinuous combinatorial target.  A gated
        # residual branch gives them dedicated capacity while leaving hold
        # frames exactly on the shared equivariant path.
        if self.boundary_feature_index is not None:
            self.boundary_owner_head = nn.Linear(2 * self.hidden_dim, 1)
            self.boundary_transmitter_head = nn.Linear(
                2 * self.hidden_dim, 1)
            if self.boundary_power_residual:
                self.boundary_power_head = nn.Linear(2 * self.hidden_dim, 1)

    def forward(
        self,
        endpoint_features: torch.Tensor,
        visible: torch.Tensor,
    ) -> FactorizedPrediction:
        if endpoint_features.ndim != 4:
            raise ValueError("endpoint_features must have shape (B,K,Q,F)")
        if endpoint_features.shape[-1] != self.feature_dim:
            raise ValueError("endpoint feature dimension mismatch")
        if visible.shape != endpoint_features.shape[:-1]:
            raise ValueError("visible mask shape mismatch")
        mask = visible.to(dtype=endpoint_features.dtype).unsqueeze(-1)
        hidden = self.endpoint_encoder(endpoint_features)
        for update, norm in zip(self.message_updates, self.message_norms):
            node_denominator = torch.clamp(mask.sum(dim=2), min=1.0)
            target_denominator = torch.clamp(mask.sum(dim=1), min=1.0)
            node_context = (hidden * mask).sum(dim=2) / node_denominator
            target_context = (hidden * mask).sum(dim=1) / target_denominator
            message = torch.cat([
                hidden,
                node_context[:, :, None, :].expand_as(hidden),
                target_context[:, None, :, :].expand_as(hidden),
            ], dim=-1)
            hidden = norm(hidden + update(message))
        target_denominator = torch.clamp(mask.sum(dim=1), min=1.0)
        target_context = (hidden * mask).sum(dim=1) / target_denominator
        edge_context = torch.cat([
            hidden,
            target_context[:, None, :, :].expand_as(hidden),
        ], dim=-1)
        invalid = ~visible.to(dtype=torch.bool)
        owner_logits = self.owner_head(edge_context).squeeze(-1)
        transmitter_logits = self.transmitter_head(edge_context).squeeze(-1)
        power_logits = self.power_head(edge_context).squeeze(-1)
        if self.boundary_feature_index is not None:
            boundary_gate = endpoint_features[
                ..., self.boundary_feature_index
            ].clamp(0.0, 1.0)
            owner_logits = owner_logits + boundary_gate * (
                self.boundary_owner_head(edge_context).squeeze(-1))
            transmitter_logits = transmitter_logits + boundary_gate * (
                self.boundary_transmitter_head(edge_context).squeeze(-1))
            if self.boundary_power_residual:
                power_logits = power_logits + boundary_gate * (
                    self.boundary_power_head(edge_context).squeeze(-1))
        owner_logits = owner_logits.masked_fill(invalid, -1.0e9)
        transmitter_logits = transmitter_logits.masked_fill(invalid, -1.0e9)
        power_logits = power_logits.masked_fill(invalid, -1.0e9)
        return FactorizedPrediction(
            owner_logits=owner_logits,
            transmitter_logits=transmitter_logits,
            power_logits=power_logits,
            dual_warm_start=functional.softplus(
                self.dual_head(target_context).squeeze(-1)),
            risk_radius=functional.softplus(
                self.risk_head(target_context).squeeze(-1)),
            detection_logits=self.detection_head(
                target_context).squeeze(-1),
            detection_delta=torch.tanh(
                self.detection_delta_head(target_context).squeeze(-1)),
            endpoint_embedding=hidden,
        )

    def conditional_transmitter_logits(
        self,
        prediction: FactorizedPrediction,
        receiver_index: torch.Tensor,
    ) -> torch.Tensor:
        """Score transmitters conditioned on selected receivers in O(BKQH).

        ``receiver_index`` has shape ``(B,Q)``. This sparse pair decoder can
        represent bistatic Tx--Rx compatibility without ever producing KxK.
        """
        hidden = prediction.endpoint_embedding
        if receiver_index.shape != (hidden.shape[0], hidden.shape[2]):
            raise ValueError("receiver_index must have shape (B,Q)")
        gather_index = receiver_index[:, None, :, None].expand(
            -1, 1, -1, hidden.shape[-1])
        receiver = torch.gather(hidden, 1, gather_index).squeeze(1)
        query = self.receiver_query(receiver)[:, None, :, :]
        key = self.transmitter_key(hidden)
        compatibility = (query * key).sum(dim=-1) / self.hidden_dim ** 0.5
        return prediction.transmitter_logits + compatibility


class PredictionPrefetchCache:
    """Double-buffer contract for one-frame-ahead asynchronous inference."""

    def __init__(self) -> None:
        self._ready: PrefetchedPrediction | None = None

    def publish(
        self,
        source_frame: int,
        prediction: FactorizedPrediction,
        *,
        horizon: int = 1,
    ) -> None:
        if horizon < 1:
            raise ValueError("prediction horizon must be positive")
        self._ready = PrefetchedPrediction(
            source_frame=int(source_frame),
            target_frame=int(source_frame) + int(horizon),
            prediction=prediction,
        )

    def consume(self, target_frame: int) -> PrefetchedPrediction | None:
        ready = self._ready
        if ready is None or ready.target_frame != int(target_frame):
            return None
        self._ready = None
        return ready


def decode_candidate_pool(
    prediction: FactorizedPrediction,
    visible: np.ndarray,
    *,
    model: CertifiedBipartiteGNN | None = None,
    owner_beam: int = 2,
    transmitters_per_owner: int = 3,
) -> tuple[tuple[int, int, int], ...]:
    """Decode a conservative common candidate pool from viewer predictions."""
    owner = prediction.owner_logits.detach().cpu().numpy()
    transmitter = prediction.transmitter_logits.detach().cpu().numpy()
    seen = np.asarray(visible, dtype=bool)
    if owner.ndim != 3 or owner.shape != transmitter.shape or seen.shape != owner.shape:
        raise ValueError("candidate prediction shapes are inconsistent")
    _, K, Q = owner.shape
    owner_score = np.sum(np.where(seen, owner, 0.0), axis=0) / np.maximum(
        np.sum(seen, axis=0), 1)
    transmitter_score = (
        np.sum(np.where(seen, transmitter, 0.0), axis=0)
        / np.maximum(np.sum(seen, axis=0), 1))
    beam_size = max(1, min(int(owner_beam), K))
    owner_indices = np.argsort(
        -owner_score, axis=0, kind="stable")[:beam_size]
    conditional_scores: list[np.ndarray] = []
    if model is not None:
        for beam in range(beam_size):
            # Every viewer evaluates the same consensus receiver per target.
            receiver_tensor = torch.from_numpy(
                np.broadcast_to(owner_indices[beam], (owner.shape[0], Q)).copy()
            ).to(device=prediction.owner_logits.device, dtype=torch.long)
            with torch.inference_mode():
                conditional = model.conditional_transmitter_logits(
                    prediction, receiver_tensor)
            conditional_np = conditional.detach().cpu().numpy()
            conditional_scores.append(
                np.sum(np.where(seen, conditional_np, 0.0), axis=0)
                / np.maximum(np.sum(seen, axis=0), 1)
            )

    candidates = set()
    transmitter_count = max(
        1, min(int(transmitters_per_owner), K - 1))
    for beam in range(beam_size):
        for target in range(Q):
            receiver = int(owner_indices[beam, target])
            score = (
                transmitter_score[:, target].copy()
                if model is None
                else conditional_scores[beam][:, target].copy()
            )
            score[receiver] = -np.inf
            transmitters = np.argsort(-score, kind="stable")[
                :transmitter_count]
            candidates.update(
                (int(tx), receiver, int(target))
                for tx in transmitters
                if np.isfinite(score[int(tx)])
            )
    return tuple(sorted(candidates))


def validate_selected_prediction(
    selected: Iterable[tuple[int, int, int]],
    *,
    num_uavs: int,
    num_targets: int,
    target_pair_limit: int,
) -> tuple[bool, tuple[str, ...]]:
    """Check structural feasibility before exact physical certification."""
    edges = tuple(tuple(int(value) for value in edge) for edge in selected)
    reasons: list[str] = []
    owners = np.full(int(num_targets), -1, dtype=np.int64)
    counts = np.zeros(int(num_targets), dtype=np.int64)
    if len(set(edges)) != len(edges):
        reasons.append("duplicate_edge")
    for transmitter, receiver, target in edges:
        if not (
            0 <= transmitter < int(num_uavs)
            and 0 <= receiver < int(num_uavs)
            and 0 <= target < int(num_targets)
            and transmitter != receiver
        ):
            reasons.append("edge_out_of_range")
            continue
        if owners[target] not in (-1, receiver):
            reasons.append("non_unique_owner")
        owners[target] = receiver
        counts[target] += 1
    if np.any(owners < 0):
        reasons.append("incomplete_target_coverage")
    if np.any(counts > int(target_pair_limit)):
        reasons.append("target_pair_limit")
    unique_reasons = tuple(dict.fromkeys(reasons))
    return not unique_reasons, unique_reasons
