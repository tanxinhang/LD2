"""Lightweight shared scorer for cardinality-agnostic local moves."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from uav_isac.coordination.local_move_ranker import FEATURE_NAMES


class LocalMoveRanker(nn.Module):
    def __init__(
        self,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        *,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        mean = np.asarray(feature_mean, dtype=np.float32)
        std = np.asarray(feature_std, dtype=np.float32)
        if mean.shape != (len(FEATURE_NAMES),) or std.shape != mean.shape:
            raise ValueError("feature normalization width mismatch")
        self.register_buffer("feature_mean", torch.from_numpy(mean))
        self.register_buffer(
            "feature_std", torch.from_numpy(np.maximum(std, 1.0e-4)))
        self.backbone = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.rank_head = nn.Linear(hidden_dim, 1)
        self.positive_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (
            features - self.feature_mean) / self.feature_std
        hidden = self.backbone(normalized)
        return (
            self.rank_head(hidden).squeeze(-1),
            self.positive_head(hidden).squeeze(-1),
        )


class FrozenLocalMoveRanker:
    """CPU inference wrapper with a stable NumPy interface."""

    def __init__(self, checkpoint: str | Path) -> None:
        payload = torch.load(
            Path(checkpoint), map_location="cpu", weights_only=False)
        self.alpha = float(payload["alpha"])
        self.model = LocalMoveRanker(
            payload["feature_mean"],
            payload["feature_std"],
            hidden_dim=int(payload["hidden_dim"]),
        )
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(features, dtype=torch.float32)
        rank_score, positive_logit = self.model(tensor)
        return (rank_score + self.alpha * positive_logit).cpu().numpy()

