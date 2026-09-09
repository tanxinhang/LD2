"""Frozen distributed surrogate for centralized structure edge values.

Each UAV encodes one feature vector per target.  A shared directed decoder
combines a transmitter endpoint embedding with a receiver endpoint embedding.
Consequently every node can reconstruct the same K-by-K-by-Q edge-value graph
after exchanging the endpoint embeddings; no UAV identity or fixed target ID
is consumed by the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from uav_isac.domain.observation_slices import ObservationSlices
from uav_isac.utils.checkpoint_loading import safe_torch_load


class FactorizedStructureStudent(torch.nn.Module):
    """Shared endpoint encoders plus a directed bistatic edge decoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        endpoint_dim: int = 32,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.endpoint_dim = int(endpoint_dim)

        def endpoint_encoder() -> torch.nn.Sequential:
            return torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, self.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_dim, self.endpoint_dim),
            )

        self.tx_encoder = endpoint_encoder()
        self.rx_encoder = endpoint_encoder()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(
                3 * self.endpoint_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )
        torch.nn.init.zeros_(self.decoder[-1].weight)
        torch.nn.init.zeros_(self.decoder[-1].bias)

    def encode_endpoints(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tx_encoder(features), self.rx_encoder(features)

    def decode_edges(
        self,
        tx: torch.Tensor,
        rx: torch.Tensor,
    ) -> torch.Tensor:
        tx_pair = tx[:, :, None, :, :]
        rx_pair = rx[:, None, :, :, :]
        pair_features = torch.cat([
            tx_pair.expand(-1, -1, rx.shape[1], -1, -1),
            rx_pair.expand(-1, tx.shape[1], -1, -1, -1),
            tx_pair * rx_pair,
        ], dim=-1)
        return self.decoder(pair_features).squeeze(-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tx, rx = self.encode_endpoints(features)
        return self.decode_edges(tx, rx)


class CardinalityResidualStructureStudent(torch.nn.Module):
    """Frozen anchor Student plus a scale-gated equivariant residual.

    The gate is a permutation-invariant function of the tensor cardinality.
    It is exactly zero at the anchor K/Q, so neither endpoint Tokens nor edge
    values can change there.  Only the residual path is trainable.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        endpoint_dim: int = 32,
        *,
        anchor_num_uavs: int,
        anchor_num_targets: int,
        reference_num_uavs: int,
        reference_num_targets: int,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.endpoint_dim = int(endpoint_dim)
        self.anchor_num_uavs = int(anchor_num_uavs)
        self.anchor_num_targets = int(anchor_num_targets)
        self.reference_num_uavs = int(reference_num_uavs)
        self.reference_num_targets = int(reference_num_targets)
        if min(
            self.anchor_num_uavs,
            self.anchor_num_targets,
            self.reference_num_uavs,
            self.reference_num_targets,
        ) < 1:
            raise ValueError("student cardinalities must be positive")

        self.base = FactorizedStructureStudent(
            self.input_dim, self.hidden_dim, self.endpoint_dim)

        def endpoint_residual() -> torch.nn.Sequential:
            network = torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, self.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_dim, self.endpoint_dim),
            )
            torch.nn.init.zeros_(network[-1].weight)
            torch.nn.init.zeros_(network[-1].bias)
            return network

        self.tx_residual = endpoint_residual()
        self.rx_residual = endpoint_residual()
        self.decoder_residual = torch.nn.Sequential(
            torch.nn.Linear(3 * self.endpoint_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )
        torch.nn.init.zeros_(self.decoder_residual[-1].weight)
        torch.nn.init.zeros_(self.decoder_residual[-1].bias)
        self.freeze_base()

    def freeze_base(self) -> None:
        self.base.eval()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        # Avoid changing future stateful base modules if the architecture is
        # extended with normalization or dropout.
        self.base.eval()
        return self

    def cardinality_gate(self, num_uavs: int, num_targets: int) -> float:
        uav_denominator = max(
            abs(self.reference_num_uavs - self.anchor_num_uavs), 1)
        target_denominator = max(
            abs(self.reference_num_targets - self.anchor_num_targets), 1)
        distance = max(
            abs(int(num_uavs) - self.anchor_num_uavs) / uav_denominator,
            abs(int(num_targets) - self.anchor_num_targets)
            / target_denominator,
        )
        return float(np.clip(distance, 0.0, 1.0))

    @staticmethod
    def _pair_features(
        tx: torch.Tensor,
        rx: torch.Tensor,
    ) -> torch.Tensor:
        tx_pair = tx[:, :, None, :, :]
        rx_pair = rx[:, None, :, :, :]
        return torch.cat([
            tx_pair.expand(-1, -1, rx.shape[1], -1, -1),
            rx_pair.expand(-1, tx.shape[1], -1, -1, -1),
            tx_pair * rx_pair,
        ], dim=-1)

    def encode_endpoints(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        K, Q = int(features.shape[1]), int(features.shape[2])
        gate = self.cardinality_gate(K, Q)
        base_tx, base_rx = self.base.encode_endpoints(features)
        tx = base_tx + gate * self.tx_residual(features)
        rx = base_rx + gate * self.rx_residual(features)
        return tx, rx

    def decode_edges(
        self,
        tx: torch.Tensor,
        rx: torch.Tensor,
    ) -> torch.Tensor:
        K, Q = int(tx.shape[1]), int(tx.shape[2])
        gate = self.cardinality_gate(K, Q)
        base_value = self.base.decode_edges(tx, rx)
        residual = self.decoder_residual(
            self._pair_features(tx, rx)).squeeze(-1)
        return base_value + gate * residual

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tx, rx = self.encode_endpoints(features)
        return self.decode_edges(tx, rx)


def build_structure_student_features(
    local_obs: np.ndarray,
    slices: ObservationSlices,
    *,
    outgoing_message: np.ndarray,
    outgoing_token_mask: np.ndarray,
    outgoing_rate: np.ndarray,
    comm_fraction: np.ndarray,
    sensing_weights: np.ndarray,
    rate_scale: float,
    neighbor_subset_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build permutation-equivariant per-target endpoint features.

    Inputs may have shape ``(K, ...)`` for one frame or ``(F, K, ...)`` for a
    batch.  The returned tensor always has shape ``(F, K, Q, D)``.
    """
    obs = np.asarray(local_obs, dtype=np.float32)
    single_frame = obs.ndim == 2
    if single_frame:
        obs = obs[None, ...]
    if obs.ndim != 3:
        raise ValueError("local_obs must have shape (K,D) or (F,K,D)")
    F, K, _ = obs.shape
    Q = slices.Q

    def with_frame_axis(
        values: np.ndarray,
        expected_ndim: int,
    ) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim == expected_ndim - 1:
            array = array[None, ...]
        if array.ndim != expected_ndim or array.shape[0] != F:
            raise ValueError("student action field has incompatible shape")
        return array

    self_features = np.repeat(
        slices.extract_self(obs)[:, :, None, :], Q, axis=2)
    belief = slices.extract_beliefs(obs)
    geometry = slices.extract_geometry(obs)
    pd_hist = slices.extract_pd_hist(obs)[..., None]
    own_memory = slices.extract_comm(obs)
    own_claim = own_memory[..., :Q, None]
    own_mask = own_memory[..., Q:2 * Q, None]

    received = slices.extract_comm_tokens(obs).reshape(
        F, K, K - 1, Q, slices.comm_token_per_sender)
    received_mask = slices.extract_comm_mask(obs).reshape(
        F, K, K - 1, Q, 1).astype(np.float32)
    local_neighbor_count = np.full(
        (F, K, 1, 1), max(K - 1, 1), dtype=np.float32)
    if neighbor_subset_mask is not None:
        subset = np.asarray(neighbor_subset_mask, dtype=bool)
        if subset.ndim == 2:
            subset = subset[None, ...]
        if subset.shape != (F, K, K):
            raise ValueError(
                "neighbor_subset_mask must have shape (F,K,K) or (K,K)")
        peer_subset = np.zeros((F, K, K - 1, 1, 1), dtype=np.float32)
        for viewer in range(K):
            peers = [peer for peer in range(K) if peer != viewer]
            peer_subset[:, viewer, :, 0, 0] = subset[:, viewer, peers]
        received_mask = received_mask * peer_subset
        local_neighbor_count = np.maximum(
            np.sum(peer_subset, axis=2), 1.0).astype(np.float32)
    # Remove sender/target identifiers.  Latent content and channel state are
    # pooled as a set, preserving neighbor permutation equivariance.
    received_equivariant = np.concatenate(
        [received[..., :-6], received[..., -4:]], axis=-1)
    received_sum = np.sum(
        received_equivariant * received_mask, axis=2)
    received_count = np.sum(received_mask, axis=2)
    received_mean = received_sum / np.maximum(received_count, 1.0)
    received_max = np.max(np.where(
        received_mask > 0.5,
        received_equivariant,
        -1.0e4,
    ), axis=2)
    received_max = np.where(
        received_count > 0.0, received_max, 0.0)
    received_fraction = received_count / local_neighbor_count

    outgoing = with_frame_axis(
        outgoing_message, 3).astype(np.float32)
    outgoing = outgoing.reshape(F, K, Q, -1)
    outgoing_mask = with_frame_axis(
        outgoing_token_mask, 3).astype(np.float32)[..., None]
    rate = with_frame_axis(
        outgoing_rate, 2).astype(np.float32)[:, :, None, None]
    rate = np.repeat(rate, Q, axis=2) / max(float(rate_scale), 1.0)
    comm = with_frame_axis(
        comm_fraction, 2).astype(np.float32)[:, :, None, None]
    comm = np.repeat(comm, Q, axis=2)
    sensing = with_frame_axis(
        sensing_weights, 3).astype(np.float32)[..., None]

    features = np.concatenate([
        self_features,
        belief,
        geometry,
        pd_hist,
        own_claim,
        own_mask,
        received_mean,
        received_max,
        received_fraction,
        outgoing,
        outgoing_mask,
        rate,
        comm,
        sensing,
    ], axis=-1)
    return features


def save_frozen_structure_student(
    output: str | Path,
    model: FactorizedStructureStudent,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    *,
    num_uavs: int,
    num_targets: int,
    rate_scale: float,
    endpoint_scale: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": 1,
        "model": {
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "endpoint_dim": model.endpoint_dim,
        },
        "state_dict": model.state_dict(),
        "feature_mean": np.asarray(
            feature_mean, dtype=np.float32).reshape(-1),
        "feature_scale": np.asarray(
            feature_scale, dtype=np.float32).reshape(-1),
        "num_uavs": int(num_uavs),
        "num_targets": int(num_targets),
        "rate_scale": float(rate_scale),
        "endpoint_scale": np.asarray(
            endpoint_scale
            if endpoint_scale is not None
            else np.ones(2 * model.endpoint_dim),
            dtype=np.float32,
        ).reshape(-1),
        "metadata": dict(metadata or {}),
    }, path)


def save_cardinality_residual_structure_student(
    output: str | Path,
    model: CardinalityResidualStructureStudent,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    *,
    num_uavs: int,
    num_targets: int,
    rate_scale: float,
    endpoint_scale: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a schema-v2 anchor-preserving residual Student."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": 2,
        "model": {
            "kind": "cardinality_residual",
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "endpoint_dim": model.endpoint_dim,
            "anchor_num_uavs": model.anchor_num_uavs,
            "anchor_num_targets": model.anchor_num_targets,
            "reference_num_uavs": model.reference_num_uavs,
            "reference_num_targets": model.reference_num_targets,
        },
        "state_dict": model.state_dict(),
        "feature_mean": np.asarray(
            feature_mean, dtype=np.float32).reshape(-1),
        "feature_scale": np.asarray(
            feature_scale, dtype=np.float32).reshape(-1),
        "num_uavs": int(num_uavs),
        "num_targets": int(num_targets),
        "rate_scale": float(rate_scale),
        "endpoint_scale": np.asarray(
            endpoint_scale, dtype=np.float32).reshape(-1),
        "metadata": dict(metadata or {}),
    }, path)


class FrozenStructureStudent:
    """Inference wrapper with immutable normalization and network weights."""

    # The encoders are shared over (UAV,target) entities, the decoder is shared
    # over directed endpoint pairs, and feature construction removes absolute
    # UAV/target identifiers. Runtime migration remains opt-in because the
    # learned normalization was fitted on one training cardinality.
    cardinality_equivariant = True

    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        payload = safe_torch_load(
            checkpoint,
            map_location=device,
            description="frozen structure-student checkpoint",
            allow_legacy_numpy_float32=True,
            required_keys=(
                "schema_version", "feature_mean", "feature_scale",
                "num_uavs", "num_targets", "rate_scale",
            ),
            mapping_keys=("model",),
            state_dict_keys=("state_dict",),
        )
        self.schema_version = int(payload.get("schema_version", -1))
        if self.schema_version not in {1, 2}:
            raise ValueError("unsupported structure-student schema")
        model_config = payload["model"]
        if self.schema_version == 1:
            self.model = FactorizedStructureStudent(
                input_dim=int(model_config["input_dim"]),
                hidden_dim=int(model_config["hidden_dim"]),
                endpoint_dim=int(model_config["endpoint_dim"]),
            ).to(device)
        else:
            if model_config.get("kind") != "cardinality_residual":
                raise ValueError("unsupported schema-v2 Student kind")
            self.model = CardinalityResidualStructureStudent(
                input_dim=int(model_config["input_dim"]),
                hidden_dim=int(model_config["hidden_dim"]),
                endpoint_dim=int(model_config["endpoint_dim"]),
                anchor_num_uavs=int(model_config["anchor_num_uavs"]),
                anchor_num_targets=int(model_config["anchor_num_targets"]),
                reference_num_uavs=int(model_config["reference_num_uavs"]),
                reference_num_targets=int(model_config["reference_num_targets"]),
            ).to(device)
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = torch.device(device)
        self.feature_mean = np.asarray(
            payload["feature_mean"], dtype=np.float32).reshape(1, 1, 1, -1)
        self.feature_scale = np.asarray(
            payload["feature_scale"], dtype=np.float32).reshape(1, 1, 1, -1)
        self.num_uavs = int(payload["num_uavs"])
        self.num_targets = int(payload["num_targets"])
        self.rate_scale = float(payload["rate_scale"])
        self.endpoint_scale = np.asarray(
            payload.get(
                "endpoint_scale",
                np.ones(2 * self.model.endpoint_dim),
            ),
            dtype=np.float32,
        ).reshape(1, 1, 1, -1)
        if self.endpoint_scale.shape[-1] != 2 * self.model.endpoint_dim:
            raise ValueError("structure-student endpoint scale mismatch")
        self.endpoint_scale = np.maximum(
            self.endpoint_scale, 1.0e-5)
        self.metadata = dict(payload.get("metadata", {}))

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 4:
            raise ValueError("features must have shape (F,K,Q,D)")
        if values.shape[-1] != self.feature_mean.shape[-1]:
            raise ValueError("structure-student feature width mismatch")
        normalized = (values - self.feature_mean) / self.feature_scale
        with torch.inference_mode():
            prediction_log = self.model(torch.as_tensor(
                normalized, dtype=torch.float32, device=self.device))
            prediction = torch.expm1(prediction_log).clamp_min(0.0)
        result = prediction.detach().cpu().numpy()
        for k in range(min(result.shape[1], result.shape[2])):
            result[:, k, k, :] = 0.0
        return result

    def encode_endpoint_protocol(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """Encode local target features into a bounded public Tx/Rx stream."""
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 4:
            raise ValueError("features must have shape (F,K,Q,D)")
        normalized = (values - self.feature_mean) / self.feature_scale
        with torch.inference_mode():
            tx, rx = self.model.encode_endpoints(torch.as_tensor(
                normalized, dtype=torch.float32, device=self.device))
            endpoints = torch.cat([tx, rx], dim=-1).cpu().numpy()
        return np.clip(
            endpoints / self.endpoint_scale, -1.0, 1.0)

    def predict_from_endpoint_protocol(
        self,
        protocol: np.ndarray,
        valid_senders: np.ndarray | None = None,
    ) -> np.ndarray:
        """Decode a quantized/cached public endpoint table into edge values."""
        values = np.asarray(protocol, dtype=np.float32)
        if values.ndim == 3:
            values = values[None, ...]
        if values.ndim != 4:
            raise ValueError(
                "endpoint protocol must have shape (K,Q,2E) or (F,K,Q,2E)")
        if values.shape[-1] != 2 * self.model.endpoint_dim:
            raise ValueError("endpoint protocol width mismatch")
        endpoints = values * self.endpoint_scale
        endpoint_dim = self.model.endpoint_dim
        with torch.inference_mode():
            endpoint_tensor = torch.as_tensor(
                endpoints, dtype=torch.float32, device=self.device)
            prediction_log = self.model.decode_edges(
                endpoint_tensor[..., :endpoint_dim],
                endpoint_tensor[..., endpoint_dim:],
            )
            prediction = torch.expm1(prediction_log).clamp_min(0.0)
        result = prediction.cpu().numpy()
        if valid_senders is not None:
            valid = np.asarray(valid_senders, dtype=bool)
            if valid.ndim == 1:
                valid = valid[None, ...]
            if valid.shape != result.shape[:2]:
                raise ValueError("valid_senders shape mismatch")
            valid_edges = (
                valid[:, :, None, None]
                & valid[:, None, :, None]
            )
            result = np.where(valid_edges, result, 0.0)
        for k in range(min(result.shape[1], result.shape[2])):
            result[:, k, k, :] = 0.0
        return result
