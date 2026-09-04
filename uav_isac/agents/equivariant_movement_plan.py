"""Causal set-equivariant residual movement commitments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from uav_isac.utils.checkpoint_loading import safe_torch_load


@dataclass(frozen=True)
class EquivariantMovementPlanMetadata:
    """Frozen deployment contract carried by a selected plan head."""

    architecture: str
    horizon_steps: int
    movement_decision_interval: int
    region_size_m: tuple[float, float]
    maximum_displacement_m: float
    residual_limit_fraction: float
    training_episode_ids: tuple[int, ...]
    selection_episode_ids: tuple[int, ...]
    selection_admitted: bool
    training_episode_keys: tuple[str, ...] = ()
    selection_episode_keys: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EquivariantMovementPlanMetadata":
        return cls(
            architecture=str(value.get(
                "architecture",
                "shared-uav-target-set-invariant-zoh-residual-v1",
            )),
            horizon_steps=int(value["horizon_steps"]),
            movement_decision_interval=int(value["movement_decision_interval"]),
            region_size_m=tuple(float(item) for item in value["region_size_m"]),
            maximum_displacement_m=float(value["maximum_displacement_m"]),
            residual_limit_fraction=float(value["residual_limit_fraction"]),
            training_episode_ids=tuple(
                int(item) for item in value["training_episode_ids"]),
            selection_episode_ids=tuple(
                int(item) for item in value["selection_episode_ids"]),
            selection_admitted=bool(value["selection_admitted"]),
            training_episode_keys=tuple(
                str(item) for item in value.get("training_episode_keys", ())),
            selection_episode_keys=tuple(
                str(item) for item in value.get("selection_episode_keys", ())),
        )


class EquivariantResidualMovementPlan(nn.Module):
    """Predict a bounded H-step residual around the current movement action.

    The same map is applied to every UAV. Target tokens are encoded by one
    shared function and reduced by symmetric mean/max/attention operators.
    Thus UAV relabeling permutes outputs and target relabeling leaves movement
    outputs unchanged. A zero final layer makes initialization exactly ZOH.
    """

    def __init__(
        self,
        *,
        horizon_steps: int,
        movement_decision_interval: int,
        region_size_m: tuple[float, float],
        maximum_displacement_m: float,
        hidden_dim: int = 64,
        residual_limit_fraction: float = 0.75,
    ) -> None:
        super().__init__()
        if int(horizon_steps) < 1 or int(movement_decision_interval) < 1:
            raise ValueError("horizon and movement interval must be positive")
        region = np.asarray(region_size_m, dtype=np.float64)
        if region.shape != (2,) or np.any(region <= 0.0):
            raise ValueError("region_size_m must contain two positive values")
        if maximum_displacement_m <= 0.0:
            raise ValueError("maximum_displacement_m must be positive")
        if not 0.0 < residual_limit_fraction <= 2.0:
            raise ValueError("residual_limit_fraction must lie in (0,2]")
        self.horizon_steps = int(horizon_steps)
        self.movement_decision_interval = int(movement_decision_interval)
        self.maximum_displacement_m = float(maximum_displacement_m)
        self.residual_limit_fraction = float(residual_limit_fraction)
        self.register_buffer(
            "region_size_m", torch.as_tensor(region, dtype=torch.float32))
        hidden = int(hidden_dim)
        self.target_encoder = nn.Sequential(
            nn.Linear(7, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.target_attention = nn.Linear(hidden, 1)
        self.self_encoder = nn.Sequential(
            nn.Linear(10, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.plan_trunk = nn.Sequential(
            nn.Linear(4 * hidden, 2 * hidden), nn.SiLU(),
            nn.Linear(2 * hidden, hidden), nn.SiLU(),
        )
        self.plan_output = nn.Linear(hidden, 2 * self.horizon_steps)
        nn.init.zeros_(self.plan_output.weight)
        nn.init.zeros_(self.plan_output.bias)

    def _guaranteed_hold_mask(self, decision_frame: torch.Tensor) -> torch.Tensor:
        phase = torch.remainder(
            decision_frame.to(dtype=torch.long) - 1,
            self.movement_decision_interval,
        )
        remaining = self.movement_decision_interval - 1 - phase
        offsets = torch.arange(
            1, self.horizon_steps + 1, device=decision_frame.device)
        return offsets[None, :] <= remaining[:, None]

    def forward(
        self,
        uav_position_m: torch.Tensor,
        target_position_m: torch.Tensor,
        current_movement_m: torch.Tensor,
        communication_power_w: torch.Tensor,
        sensing_power_w: torch.Tensor,
        decision_frame: torch.Tensor,
    ) -> torch.Tensor:
        if uav_position_m.ndim != 3 or uav_position_m.shape[-1] != 2:
            raise ValueError("uav_position_m must have shape (B,K,2)")
        if target_position_m.ndim != 3 or target_position_m.shape[-1] != 2:
            raise ValueError("target_position_m must have shape (B,Q,2)")
        batch, num_uavs, _ = uav_position_m.shape
        num_targets = target_position_m.shape[1]
        if (
            current_movement_m.shape != (batch, num_uavs, 2)
            or communication_power_w.shape != (batch, num_uavs)
            or sensing_power_w.shape != (batch, num_uavs, num_targets)
            or decision_frame.shape != (batch,)
        ):
            raise ValueError("movement-plan input dimensions are inconsistent")

        region = self.region_size_m.to(
            device=uav_position_m.device, dtype=uav_position_m.dtype)
        relative = (
            target_position_m[:, None, :, :]
            - uav_position_m[:, :, None, :]
        )
        normalized_relative = relative / region[None, None, None, :]
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        diagonal = torch.linalg.vector_norm(region)
        direction = relative / torch.clamp(distance, min=1.0e-6)
        target_features = torch.cat((
            normalized_relative,
            distance / diagonal,
            direction,
            sensing_power_w[..., None],
            communication_power_w[:, :, None, None].expand(
                -1, -1, num_targets, -1),
        ), dim=-1)
        target_latent = self.target_encoder(target_features)
        target_mean = torch.mean(target_latent, dim=2)
        target_max = torch.amax(target_latent, dim=2)
        attention = torch.softmax(self.target_attention(target_latent), dim=2)
        target_attention = torch.sum(attention * target_latent, dim=2)

        normalized_movement = current_movement_m / self.maximum_displacement_m
        movement_norm = torch.linalg.vector_norm(
            normalized_movement, dim=-1, keepdim=True)
        phase = torch.remainder(
            decision_frame.to(dtype=uav_position_m.dtype) - 1.0,
            float(self.movement_decision_interval),
        ) / float(self.movement_decision_interval)
        phase_angle = 2.0 * torch.pi * phase
        phase_features = torch.stack((
            torch.sin(phase_angle), torch.cos(phase_angle)), dim=-1)
        phase_features = phase_features[:, None, :].expand(-1, num_uavs, -1)
        self_features = torch.cat((
            uav_position_m / region[None, None, :],
            normalized_movement,
            movement_norm,
            communication_power_w[..., None],
            torch.sum(sensing_power_w, dim=-1, keepdim=True),
            phase_features,
            torch.ones_like(movement_norm),
        ), dim=-1)
        self_latent = self.self_encoder(self_features)
        context = torch.cat((
            self_latent, target_mean, target_max, target_attention), dim=-1)
        raw_residual = self.plan_output(self.plan_trunk(context)).reshape(
            batch, num_uavs, self.horizon_steps, 2)
        residual = self.residual_limit_fraction * torch.tanh(raw_residual)
        hold_mask = self._guaranteed_hold_mask(decision_frame)
        residual = torch.where(
            hold_mask[:, None, :, None],
            torch.zeros_like(residual), residual)
        plan = normalized_movement[:, :, None, :] + residual
        norm = torch.linalg.vector_norm(plan, dim=-1, keepdim=True)
        plan = plan / torch.clamp(norm, min=1.0)
        return (
            plan * self.maximum_displacement_m
        ).permute(0, 2, 1, 3).contiguous()


class DoubleSetEquivariantResidualMovementPlan(nn.Module):
    """Use target and peer-UAV sets for scale-aware coordinated planning.

    For every UAV, target tokens and all other UAV tokens are encoded by two
    shared maps and reduced by symmetric mean, max and attention operators.
    The target branch is invariant to target relabeling.  Relabeling UAVs
    permutes both the focal-UAV rows and their peer sets, so the output is UAV
    permutation equivariant.  Every geometric feature is dimensionless,
    which also gives exact covariance under a uniform region rescaling.
    """

    def __init__(
        self,
        *,
        horizon_steps: int,
        movement_decision_interval: int,
        region_size_m: tuple[float, float],
        maximum_displacement_m: float,
        hidden_dim: int = 64,
        residual_limit_fraction: float = 0.75,
    ) -> None:
        super().__init__()
        if int(horizon_steps) < 1 or int(movement_decision_interval) < 1:
            raise ValueError("horizon and movement interval must be positive")
        region = np.asarray(region_size_m, dtype=np.float64)
        if region.shape != (2,) or np.any(region <= 0.0):
            raise ValueError("region_size_m must contain two positive values")
        if maximum_displacement_m <= 0.0:
            raise ValueError("maximum_displacement_m must be positive")
        if not 0.0 < residual_limit_fraction <= 2.0:
            raise ValueError("residual_limit_fraction must lie in (0,2]")
        self.horizon_steps = int(horizon_steps)
        self.movement_decision_interval = int(movement_decision_interval)
        self.maximum_displacement_m = float(maximum_displacement_m)
        self.residual_limit_fraction = float(residual_limit_fraction)
        self.register_buffer(
            "region_size_m", torch.as_tensor(region, dtype=torch.float32))
        hidden = int(hidden_dim)
        self.target_encoder = nn.Sequential(
            nn.Linear(7, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.target_attention = nn.Linear(hidden, 1)
        self.peer_encoder = nn.Sequential(
            nn.Linear(9, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.peer_attention = nn.Linear(hidden, 1)
        self.self_encoder = nn.Sequential(
            nn.Linear(10, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.plan_trunk = nn.Sequential(
            nn.Linear(7 * hidden, 3 * hidden), nn.SiLU(),
            nn.Linear(3 * hidden, hidden), nn.SiLU(),
        )
        self.plan_output = nn.Linear(hidden, 2 * self.horizon_steps)
        nn.init.zeros_(self.plan_output.weight)
        nn.init.zeros_(self.plan_output.bias)

    def _guaranteed_hold_mask(self, decision_frame: torch.Tensor) -> torch.Tensor:
        phase = torch.remainder(
            decision_frame.to(dtype=torch.long) - 1,
            self.movement_decision_interval,
        )
        remaining = self.movement_decision_interval - 1 - phase
        offsets = torch.arange(
            1, self.horizon_steps + 1, device=decision_frame.device)
        return offsets[None, :] <= remaining[:, None]

    @staticmethod
    def _masked_peer_pool(
        latent: torch.Tensor,
        score: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = mask[..., None]
        count = torch.sum(valid, dim=2).clamp(min=1)
        mean = torch.sum(torch.where(
            valid, latent, torch.zeros_like(latent)), dim=2) / count
        maximum = torch.amax(torch.where(
            valid, latent, torch.full_like(latent, -torch.inf)), dim=2)
        masked_score = torch.where(
            valid, score, torch.full_like(score, -torch.inf))
        attention = torch.softmax(masked_score, dim=2)
        attended = torch.sum(attention * latent, dim=2)
        return mean, maximum, attended

    def forward(
        self,
        uav_position_m: torch.Tensor,
        target_position_m: torch.Tensor,
        current_movement_m: torch.Tensor,
        communication_power_w: torch.Tensor,
        sensing_power_w: torch.Tensor,
        decision_frame: torch.Tensor,
    ) -> torch.Tensor:
        if uav_position_m.ndim != 3 or uav_position_m.shape[-1] != 2:
            raise ValueError("uav_position_m must have shape (B,K,2)")
        if target_position_m.ndim != 3 or target_position_m.shape[-1] != 2:
            raise ValueError("target_position_m must have shape (B,Q,2)")
        batch, num_uavs, _ = uav_position_m.shape
        num_targets = target_position_m.shape[1]
        if num_uavs < 2:
            raise ValueError("double-set planning requires at least two UAVs")
        if (
            current_movement_m.shape != (batch, num_uavs, 2)
            or communication_power_w.shape != (batch, num_uavs)
            or sensing_power_w.shape != (batch, num_uavs, num_targets)
            or decision_frame.shape != (batch,)
        ):
            raise ValueError("movement-plan input dimensions are inconsistent")

        region = self.region_size_m.to(
            device=uav_position_m.device, dtype=uav_position_m.dtype)
        diagonal = torch.linalg.vector_norm(region)
        normalized_movement = current_movement_m / self.maximum_displacement_m
        movement_norm = torch.linalg.vector_norm(
            normalized_movement, dim=-1, keepdim=True)
        sensing_total = torch.sum(sensing_power_w, dim=-1, keepdim=True)

        target_relative = (
            target_position_m[:, None, :, :]
            - uav_position_m[:, :, None, :]
        )
        target_distance = torch.linalg.vector_norm(
            target_relative, dim=-1, keepdim=True)
        target_direction = target_relative / torch.clamp(
            target_distance, min=1.0e-6)
        target_features = torch.cat((
            target_relative / region[None, None, None, :],
            target_distance / diagonal,
            target_direction,
            sensing_power_w[..., None],
            communication_power_w[:, :, None, None].expand(
                -1, -1, num_targets, -1),
        ), dim=-1)
        target_latent = self.target_encoder(target_features)
        target_mean = torch.mean(target_latent, dim=2)
        target_max = torch.amax(target_latent, dim=2)
        target_attention = torch.sum(
            torch.softmax(self.target_attention(target_latent), dim=2)
            * target_latent,
            dim=2,
        )

        peer_relative = (
            uav_position_m[:, None, :, :]
            - uav_position_m[:, :, None, :]
        )
        peer_distance = torch.linalg.vector_norm(
            peer_relative, dim=-1, keepdim=True)
        peer_direction = peer_relative / torch.clamp(
            peer_distance, min=1.0e-6)
        peer_features = torch.cat((
            peer_relative / region[None, None, None, :],
            peer_distance / diagonal,
            peer_direction,
            normalized_movement[:, None, :, :].expand(
                -1, num_uavs, -1, -1),
            communication_power_w[:, None, :, None].expand(
                -1, num_uavs, -1, -1),
            sensing_total[:, None, :, :].expand(
                -1, num_uavs, -1, -1),
        ), dim=-1)
        peer_latent = self.peer_encoder(peer_features)
        peer_mask = ~torch.eye(
            num_uavs, dtype=torch.bool, device=uav_position_m.device,
        )[None, :, :].expand(batch, -1, -1)
        peer_mean, peer_max, peer_attention = self._masked_peer_pool(
            peer_latent, self.peer_attention(peer_latent), peer_mask)

        phase = torch.remainder(
            decision_frame.to(dtype=uav_position_m.dtype) - 1.0,
            float(self.movement_decision_interval),
        ) / float(self.movement_decision_interval)
        phase_angle = 2.0 * torch.pi * phase
        phase_features = torch.stack((
            torch.sin(phase_angle), torch.cos(phase_angle)), dim=-1)
        phase_features = phase_features[:, None, :].expand(-1, num_uavs, -1)
        self_features = torch.cat((
            uav_position_m / region[None, None, :],
            normalized_movement,
            movement_norm,
            communication_power_w[..., None],
            sensing_total,
            phase_features,
            torch.ones_like(movement_norm),
        ), dim=-1)
        self_latent = self.self_encoder(self_features)
        context = torch.cat((
            self_latent,
            target_mean, target_max, target_attention,
            peer_mean, peer_max, peer_attention,
        ), dim=-1)
        raw_residual = self.plan_output(self.plan_trunk(context)).reshape(
            batch, num_uavs, self.horizon_steps, 2)
        residual = self.residual_limit_fraction * torch.tanh(raw_residual)
        hold_mask = self._guaranteed_hold_mask(decision_frame)
        residual = torch.where(
            hold_mask[:, None, :, None],
            torch.zeros_like(residual), residual)
        plan = normalized_movement[:, :, None, :] + residual
        norm = torch.linalg.vector_norm(plan, dim=-1, keepdim=True)
        plan = plan / torch.clamp(norm, min=1.0)
        return (
            plan * self.maximum_displacement_m
        ).permute(0, 2, 1, 3).contiguous()


class FrozenEquivariantMovementPlanner:
    """Inference-only wrapper with split and physical-domain validation."""

    def __init__(
        self,
        model: EquivariantResidualMovementPlan,
        metadata: EquivariantMovementPlanMetadata,
        *,
        runtime_region_size_m: tuple[float, float],
        uniform_region_scale: float,
    ) -> None:
        self.model = model.eval()
        self.metadata = metadata
        self.runtime_region_size_m = tuple(
            float(item) for item in runtime_region_size_m)
        self.uniform_region_scale = float(uniform_region_scale)

    @property
    def uniform_region_scaling_applied(self) -> bool:
        return not np.isclose(
            self.uniform_region_scale, 1.0, rtol=0.0, atol=1.0e-12)

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        *,
        validation_episode_ids: tuple[int, ...],
        horizon_steps: int,
        movement_decision_interval: int,
        region_size_m: tuple[float, float],
        maximum_displacement_m: float,
        allow_uniform_region_scaling: bool = False,
        validation_domain_key: str | None = None,
    ) -> "FrozenEquivariantMovementPlanner":
        payload = safe_torch_load(
            path,
            map_location="cpu",
            description="equivariant movement-plan checkpoint",
            required_keys=("hidden_dim",),
            mapping_keys=("metadata",),
            state_dict_keys=("model_state_dict",),
        )
        metadata = EquivariantMovementPlanMetadata.from_dict(payload["metadata"])
        if not metadata.selection_admitted:
            raise ValueError("movement-plan checkpoint failed holdout admission")
        development_keys = (
            set(metadata.training_episode_keys)
            | set(metadata.selection_episode_keys))
        if development_keys:
            if not validation_domain_key:
                raise ValueError(
                    "domain-keyed checkpoint requires validation_domain_key")
            validation_keys = {
                f"{validation_domain_key}:{int(item)}"
                for item in validation_episode_ids
            }
            overlap = development_keys & validation_keys
        else:
            development_ids = (
                set(metadata.training_episode_ids)
                | set(metadata.selection_episode_ids))
            overlap = development_ids & set(
                int(item) for item in validation_episode_ids)
        if overlap:
            raise ValueError(
                f"movement-plan development/validation overlap: {sorted(overlap)}")
        expected_region = np.asarray(region_size_m, dtype=np.float64)
        source_region = np.asarray(metadata.region_size_m, dtype=np.float64)
        region_ratio = expected_region / source_region
        uniform_region_scale = float(region_ratio[0])
        region_matches = bool(np.allclose(
            source_region, expected_region, rtol=0.0, atol=1.0e-12))
        uniform_region_migration = bool(
            allow_uniform_region_scaling
            and np.all(np.isfinite(region_ratio))
            and np.all(region_ratio > 0.0)
            and np.allclose(
                region_ratio,
                uniform_region_scale,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        )
        if (
            metadata.horizon_steps != int(horizon_steps)
            or metadata.movement_decision_interval
            != int(movement_decision_interval)
            or not (region_matches or uniform_region_migration)
            or not np.isclose(
                metadata.maximum_displacement_m,
                float(maximum_displacement_m),
            )
        ):
            raise ValueError("movement-plan checkpoint physical contract differs")
        architectures = {
            "shared-uav-target-set-invariant-zoh-residual-v1": (
                EquivariantResidualMovementPlan),
            "shared-uav-peer-target-double-set-equivariant-zoh-residual-v2": (
                DoubleSetEquivariantResidualMovementPlan),
        }
        model_type = architectures.get(metadata.architecture)
        if model_type is None:
            raise ValueError(
                f"unsupported movement-plan architecture: "
                f"{metadata.architecture}")
        model = model_type(
            horizon_steps=metadata.horizon_steps,
            movement_decision_interval=metadata.movement_decision_interval,
            region_size_m=metadata.region_size_m,
            maximum_displacement_m=metadata.maximum_displacement_m,
            hidden_dim=int(payload["hidden_dim"]),
            residual_limit_fraction=metadata.residual_limit_fraction,
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.region_size_m.copy_(torch.as_tensor(
            expected_region, dtype=model.region_size_m.dtype))
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return cls(
            model,
            metadata,
            runtime_region_size_m=tuple(float(item) for item in expected_region),
            uniform_region_scale=uniform_region_scale,
        )

    def predict(
        self,
        uav_position_m: np.ndarray,
        target_position_m: np.ndarray,
        current_movement_m: np.ndarray,
        communication_power_w: np.ndarray,
        sensing_power_w: np.ndarray,
        *,
        decision_frame: int,
    ) -> np.ndarray:
        with torch.no_grad():
            plan = self.model(
                torch.as_tensor(
                    np.asarray(uav_position_m, dtype=np.float32)[None, :, :2]),
                torch.as_tensor(
                    np.asarray(target_position_m, dtype=np.float32)[None, :, :2]),
                torch.as_tensor(
                    np.asarray(current_movement_m, dtype=np.float32)[None]),
                torch.as_tensor(
                    np.asarray(communication_power_w, dtype=np.float32)[None]),
                torch.as_tensor(
                    np.asarray(sensing_power_w, dtype=np.float32)[None]),
                torch.as_tensor([int(decision_frame)], dtype=torch.long),
            )
        return np.asarray(plan[0].cpu(), dtype=np.float64)
