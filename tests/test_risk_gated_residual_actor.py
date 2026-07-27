from __future__ import annotations

import torch
from torch import nn

from uav_isac.agents.residual_actor import RiskGatedResidualActor


class _FakeStructuredBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_enc = nn.Sequential(nn.Linear(11, 64), nn.ReLU())
        self.dp_log_std = nn.Parameter(torch.zeros(2))
        self.comm_rate_head = nn.Linear(8, 3)
        self.last_policy_latent = None
        self.last_movement_assignment = None
        self.capacity_blend = 0.0
        self.movement_blend = 0.0

    def _parse_obs(self, obs):
        batch = obs.shape[0]
        pd_hist = torch.full((batch, 4, 1), 0.5, device=obs.device)
        comm_mask = torch.ones(batch, 12, device=obs.device)
        return (None, None, None, None, None, None, pd_hist, None, comm_mask)

    def forward(self, obs, h_prev=None, **kwargs):
        batch = obs.shape[0]
        latent = torch.arange(64, dtype=obs.dtype, device=obs.device)
        self.last_policy_latent = latent.unsqueeze(0).expand(batch, -1) / 64.0
        self.last_movement_assignment = torch.full(
            (batch, 4), 0.25, dtype=obs.dtype, device=obs.device)
        dp = torch.stack([obs[:, 0] * 0.1, obs[:, 1] * -0.1], dim=-1)
        role = torch.zeros(batch, 3, device=obs.device)
        comm = torch.full((batch, 8), 0.25, device=obs.device)
        pd = torch.full((batch, 1), 0.5, device=obs.device)
        return dp, self.dp_log_std, role, comm, pd, h_prev

    def communication_parameters(self, comm_mean):
        return torch.zeros_like(comm_mean[0]), self.comm_rate_head(comm_mean)

    def isac_resource_parameters(self, comm_mean):
        batch = comm_mean.shape[0]
        return (torch.zeros(batch, device=comm_mean.device),
                torch.zeros(1, device=comm_mean.device),
                torch.zeros(batch, 4, device=comm_mean.device),
                torch.zeros(4, device=comm_mean.device))

    def set_capacity_matching_blend(self, blend):
        self.capacity_blend = float(blend)

    def set_target_allocation_movement_blend(self, blend):
        self.movement_blend = float(blend)


class _DirectionalFakeBase(_FakeStructuredBase):
    def _parse_obs(self, obs):
        batch = obs.shape[0]
        targets = torch.zeros(batch, 4, 18, 1, device=obs.device)
        targets[:, 0, 9, 0] = 1.0
        targets[:, 1, 10, 0] = 1.0
        targets[:, 2, 9, 0] = -1.0
        targets[:, 3, 10, 0] = -1.0
        pd_hist = torch.tensor(
            [0.1, 0.8, 0.8, 0.8], device=obs.device,
        ).reshape(1, 4, 1).expand(batch, -1, -1)
        comm_mask = torch.ones(batch, 12, device=obs.device)
        return (None, targets, None, None, None, None, pd_hist, None, comm_mask)

    def forward(self, obs, h_prev=None, **kwargs):
        result = super().forward(obs, h_prev, **kwargs)
        batch = obs.shape[0]
        assignment = torch.tensor(
            [0.7, 0.1, 0.1, 0.1], dtype=obs.dtype, device=obs.device)
        self.last_movement_assignment = assignment.expand(batch, -1)
        return result


def test_zero_init_is_exact_and_only_adapter_is_trainable() -> None:
    base = _FakeStructuredBase()
    actor = RiskGatedResidualActor(base, delta_max=0.06)
    obs = torch.tensor([[1.0, 2.0, 0.0], [-1.0, 0.5, 0.0]])
    expected = base(obs)[0].clone()
    actual = actor(obs)[0]
    assert torch.equal(actual, expected)
    assert all(not parameter.requires_grad for parameter in base.parameters())
    assert all(parameter.requires_grad for parameter in actor.trainable_params)
    assert actor.last_risk_features.shape == (2, 6)
    assert torch.all(actor.last_normalized_movement_delta == 0.0)


def test_residual_is_bounded_and_gate_receives_gradient_after_opening() -> None:
    actor = RiskGatedResidualActor(_FakeStructuredBase(), delta_max=0.06)
    with torch.no_grad():
        actor.residual[-1].bias.fill_(2.0)
    obs = torch.tensor([[0.2, -0.3, 0.0]])
    dp_base = actor.base(obs)[0]
    dp = actor(obs)[0]
    action_delta = torch.tanh(dp) - torch.tanh(dp_base)
    assert torch.all(action_delta.abs() <= 0.06 + 1e-6)
    dp.sum().backward()
    assert actor.residual[-1].bias.grad is not None
    assert actor.risk_gate[-1].bias.grad is not None


def test_auxiliary_contract_and_base_eval_mode_are_transparent() -> None:
    base = _FakeStructuredBase()
    actor = RiskGatedResidualActor(base)
    actor.train()
    assert actor.training
    assert not actor.base.training
    comm = torch.zeros(3, 8)
    assert actor.communication_parameters(comm)[1].shape == (3, 3)
    assert actor.isac_resource_parameters(comm)[2].shape == (3, 4)
    actor.set_capacity_matching_blend(0.7)
    actor.set_target_allocation_movement_blend(0.4)
    assert base.capacity_blend == 0.7
    assert base.movement_blend == 0.4


def test_directional_branch_uses_weak_assigned_target_basis() -> None:
    actor = RiskGatedResidualActor(
        _DirectionalFakeBase(), delta_max=0.06,
        directional_basis_enabled=True)
    with torch.no_grad():
        actor.direction_scale[-1].bias.fill_(2.0)
    obs = torch.tensor([[0.2, -0.3, 0.0]])
    base_dp = actor.base(obs)[0]
    guided_dp = actor(obs)[0]
    action_delta = torch.tanh(guided_dp) - torch.tanh(base_dp)
    assert action_delta[0, 0] > 0.0
    detached_delta = action_delta.detach()
    assert abs(float(detached_delta[0, 1])) < float(detached_delta[0, 0])
    assert torch.all(action_delta.abs() <= 0.06 + 1e-6)
    assert actor.last_directional_movement_basis.shape == (1, 2)
