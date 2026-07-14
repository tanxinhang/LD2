"""Bounded residual actor: frozen DAgger base + small trainable residual.

a = clip(a_DAgger + δ_max × tanh(Δa_ψ(o)), -max_dp, max_dp)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class ResidualHead(nn.Module):
    """Small MLP for bounded residual displacement."""
    def __init__(self, hidden_dim=64, hidden_units=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_units), nn.Tanh(),
            nn.Linear(hidden_units, 2),
        )
        # Zero-init last layer
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h):
        return self.net(h)


class ResidualActor(nn.Module):
    """Wrapper: frozen DAgger base + trainable bounded residual."""

    def __init__(self, base_actor, max_dp=2.5, delta_max=0.03, device='cuda'):
        super().__init__()
        self.base = base_actor
        self.max_dp = max_dp
        self.delta_max = delta_max
        self.residual = ResidualHead(hidden_dim=64, hidden_units=32).to(device)

        # Freeze base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()

        # Trainable: residual only
        self.trainable_params = list(self.residual.parameters())

    def forward(self, obs, h_prev=None):
        """Forward: base action + bounded residual."""
        with torch.no_grad():
            dp_base, log_std, role_logits, comm_msg, _, h_new = self.base(obs, h_prev)

        # Get latent from base for residual (use the attention output h)
        # We need to re-run part of base to get the hidden state
        # For simplicity, use a separate forward through base's entity encoders
        # Actually, let's extract h from the base's forward by hooking
        # Simplest: run base again but capture intermediate h
        # The base's forward returns dp, log_std, role, comm, _, h_new
        # h (the 64-dim post-gate feature) is NOT returned directly
        # Use self_enc as a proxy: h ≈ self_enc(self_state)
        result = self.base._parse_obs(obs)
        self_s, targets, neighbors, global_f, comm_agg, n_frames = result
        if targets is None:
            # Fallback: residual = 0
            return dp_base, log_std, role_logits, comm_msg, torch.zeros(obs.shape[0], 1), h_new

        # Use self encoding as proxy for hidden state
        h_self = self.base.self_enc(self_s[..., -1])  # (B, 64)
        delta_raw = self.residual(h_self)  # (B, 2)
        delta = self.delta_max * torch.tanh(delta_raw)

        # Add residual to base dp_mean
        dp_mean = dp_base + delta
        # Clip to max_dp
        dp_norm = torch.norm(dp_mean, dim=-1, keepdim=True)
        scale = torch.clamp(dp_norm / self.max_dp, max=1.0)
        dp_mean = dp_mean / (dp_norm + 1e-8) * scale * self.max_dp

        return dp_mean, log_std, role_logits, comm_msg, torch.zeros(obs.shape[0], 1), h_new
