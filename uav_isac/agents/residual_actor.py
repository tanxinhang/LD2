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


class RiskGatedResidualActor(nn.Module):
    """Frozen-base actor with a bounded decentralized movement residual.

    All non-movement outputs and auxiliary methods are delegated to ``base``.
    The residual acts in normalized movement space and is exactly neutral at
    initialization. Its gate consumes only the frozen local actor latent,
    local P_D history, assignment entropy and delivered-token availability.
    """

    _RISK_DIM = 6

    def __init__(
        self,
        base_actor: nn.Module,
        *,
        delta_max: float = 0.06,
        hidden_units: int = 32,
        gate_bias: float = -2.0,
        risk_floor: float = 0.60,
        directional_basis_enabled: bool = False,
    ) -> None:
        super().__init__()
        if not hasattr(base_actor, "self_enc"):
            raise TypeError("risk residual requires a structured base actor")
        self.base = base_actor
        self.delta_max = float(np.clip(delta_max, 0.0, 0.5))
        self.risk_floor = float(np.clip(risk_floor, 0.0, 1.0))
        self.directional_basis_enabled = bool(directional_basis_enabled)
        latent_dim = int(base_actor.self_enc[0].out_features)
        input_dim = latent_dim + self._RISK_DIM
        width = max(8, int(hidden_units))

        self.residual = nn.Sequential(
            nn.Linear(input_dim, width), nn.Tanh(), nn.Linear(width, 2))
        # A scalar policy controls a physically meaningful direction basis:
        # move along the communication-refined responsibility assigned to weak
        # targets.  The free 2-D branch remains available for corrections that
        # cannot be expressed by this radial basis.
        self.direction_scale = nn.Sequential(
            nn.Linear(input_dim, width), nn.Tanh(), nn.Linear(width, 1))
        self.risk_gate = nn.Sequential(
            nn.Linear(input_dim, width), nn.Tanh(), nn.Linear(width, 1))
        for module in (
                self.residual[0], self.direction_scale[0], self.risk_gate[0]):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(module.bias)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.zeros_(self.direction_scale[-1].weight)
        nn.init.zeros_(self.direction_scale[-1].bias)
        nn.init.zeros_(self.risk_gate[-1].weight)
        nn.init.constant_(self.risk_gate[-1].bias, float(gate_bias))

        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        if not self.directional_basis_enabled:
            for parameter in self.direction_scale.parameters():
                parameter.requires_grad_(False)
        self.base.eval()
        self.trainable_params = list(self.residual.parameters())
        if self.directional_basis_enabled:
            self.trainable_params += list(self.direction_scale.parameters())
        self.trainable_params += list(self.risk_gate.parameters())
        self.last_risk_features = None
        self.last_residual_gate = None
        self.last_normalized_movement_delta = None
        self.last_directional_movement_basis = None
        self.last_directional_movement_scale = None

    def __getattr__(self, name):
        """Delegate the current StructuredActorNetwork contract to ``base``."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = super().__getattr__("base")
            return getattr(base, name)

    def train(self, mode: bool = True):
        super().train(mode)
        # Module.train recurses into children. The foundation must remain
        # frozen and deterministic even while the adapter is being optimized.
        self.base.eval()
        return self

    def parameter_groups(self):
        return {"encoder": [], "attention": [], "head": self.trainable_params}

    def _local_risk_features(self, obs: torch.Tensor) -> torch.Tensor:
        batch = int(obs.shape[0])
        zeros = torch.zeros(
            batch, self._RISK_DIM, dtype=obs.dtype, device=obs.device)
        if obs.ndim != 2:
            return zeros
        try:
            parsed = self.base._parse_obs(obs)
            pd_hist = parsed[6]
            comm_mask = parsed[8]
        except (AttributeError, IndexError, RuntimeError, ValueError):
            return zeros
        if pd_hist is None or pd_hist.ndim < 3:
            return zeros

        pd_last = pd_hist[..., -1].clamp(0.0, 1.0)
        pd_min = pd_last.min(dim=-1).values
        pd_mean = pd_last.mean(dim=-1)
        pd_std = pd_last.std(dim=-1, unbiased=False)
        deficit = torch.relu(self.risk_floor - pd_min)

        assignment = getattr(self.base, "last_movement_assignment", None)
        if assignment is None or assignment.shape[0] != batch:
            assignment_entropy = torch.zeros_like(pd_min)
        else:
            probability = assignment.detach().clamp_min(1e-8)
            entropy = -(probability * probability.log()).sum(dim=-1)
            assignment_entropy = entropy / np.log(
                max(int(probability.shape[-1]), 2))

        if comm_mask is None or comm_mask.numel() == 0:
            availability = torch.zeros_like(pd_min)
        else:
            availability = comm_mask.to(pd_min.dtype).mean(dim=-1)
        return torch.stack([
            pd_min, pd_mean, pd_std, deficit,
            assignment_entropy, availability,
        ], dim=-1)

    def _weak_target_direction(self, obs: torch.Tensor) -> torch.Tensor:
        """Build a decentralized, communication-conditioned direction basis."""
        batch = int(obs.shape[0])
        zero = torch.zeros(batch, 2, dtype=obs.dtype, device=obs.device)
        if obs.ndim != 2:
            return zero
        try:
            parsed = self.base._parse_obs(obs)
            targets = parsed[1]
            pd_hist = parsed[6]
        except (AttributeError, IndexError, RuntimeError, ValueError):
            return zero
        if (targets is None or pd_hist is None or targets.ndim != 4
                or pd_hist.ndim < 3 or targets.shape[2] < 11):
            return zero

        rel_xy = targets[..., -1][:, :, 9:11]
        rel_norm = rel_xy.norm(dim=-1, keepdim=True)
        radial = rel_xy / rel_norm.clamp_min(1e-8)
        radial = torch.where(rel_norm > 1e-8, radial, torch.zeros_like(radial))
        pd_last = pd_hist[..., -1].clamp(0.0, 1.0)
        weakness = torch.softmax(
            (self.risk_floor - pd_last) / 0.10, dim=-1)

        assignment = getattr(self.base, "last_movement_assignment", None)
        if assignment is None or assignment.shape != weakness.shape:
            responsibility = torch.full_like(
                weakness, 1.0 / max(int(weakness.shape[-1]), 1))
        else:
            responsibility = assignment.detach().clamp_min(1e-4)
        weights = weakness * responsibility
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        direction = torch.sum(weights.unsqueeze(-1) * radial, dim=1)
        direction_norm = direction.norm(dim=-1, keepdim=True)
        return torch.where(
            direction_norm > 1e-8,
            direction / direction_norm.clamp_min(1.0),
            torch.zeros_like(direction),
        )

    def forward(
        self,
        obs: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
        detach_h_new: bool = True,
        window_mask: Optional[torch.Tensor] = None,
        comm_round_phase: Optional[torch.Tensor] = None,
        agent_identity: Optional[torch.Tensor] = None,
    ):
        with torch.no_grad():
            outputs = self.base(
                obs,
                h_prev,
                detach_h_new=detach_h_new,
                window_mask=window_mask,
                comm_round_phase=comm_round_phase,
                agent_identity=agent_identity,
            )
        dp_base, log_std, role_logits, comm_msg, pd_pred, h_new = outputs
        latent = self.base.last_policy_latent
        if latent is None or latent.shape[0] != obs.shape[0]:
            raise RuntimeError("base actor did not expose an aligned policy latent")
        risk = self._local_risk_features(obs).detach()
        adapter_input = torch.cat([latent.detach(), risk], dim=-1)
        gate = torch.sigmoid(self.risk_gate(adapter_input))
        if self.directional_basis_enabled:
            direction = self._weak_target_direction(obs).detach()
            direction_scale = torch.tanh(self.direction_scale(adapter_input))
        else:
            direction = torch.zeros(
                obs.shape[0], 2, dtype=obs.dtype, device=obs.device)
            direction_scale = torch.zeros(
                obs.shape[0], 1, dtype=obs.dtype, device=obs.device)
        residual_proposal = (
            torch.tanh(self.residual(adapter_input))
            + direction_scale * direction)
        proposal_norm = residual_proposal.norm(dim=-1, keepdim=True)
        residual_proposal = residual_proposal / proposal_norm.clamp_min(1.0)
        delta = gate * self.delta_max * residual_proposal

        base_action = torch.tanh(dp_base)
        guided_action = torch.clamp(base_action + delta, -0.999, 0.999)
        guided_mean = torch.atanh(guided_action)
        logit_delta = guided_mean - dp_base
        # Keep the zero adapter bitwise neutral without blocking its first
        # gradient through the tanh/atanh conversion.
        neutral_straight_through = (
            dp_base + logit_delta - logit_delta.detach())
        active = delta.abs().sum(dim=-1, keepdim=True) > 0.0
        dp_mean = torch.where(active, guided_mean, neutral_straight_through)

        self.last_risk_features = risk.detach()
        self.last_residual_gate = gate.detach()
        self.last_normalized_movement_delta = delta.detach()
        self.last_directional_movement_basis = direction.detach()
        self.last_directional_movement_scale = direction_scale.detach()
        return dp_mean, log_std, role_logits, comm_msg, pd_pred, h_new

    def communication_parameters(self, comm_mean: torch.Tensor):
        return self.base.communication_parameters(comm_mean)

    def isac_resource_parameters(self, comm_mean: torch.Tensor):
        return self.base.isac_resource_parameters(comm_mean)

    def set_capacity_matching_blend(self, blend: float) -> None:
        self.base.set_capacity_matching_blend(blend)

    def set_target_allocation_movement_blend(self, blend: float) -> None:
        self.base.set_target_allocation_movement_blend(blend)
