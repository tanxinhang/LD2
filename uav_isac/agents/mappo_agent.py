"""MAPPO agent with Centralized Training Decentralized Execution (CTDE).

- Actor: MLP [256, 256], input=local_obs, output=(dp_mean, dp_log_std, role_logits)
- Critic: MLP [256, 256], input=global_state, output=scalar value

Both actor and critic are shared across agents (parameter sharing).
Each agent has its own observation but the same policy network.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from copy import deepcopy

from uav_isac.agents.base_agent import BaseAgent
from uav_isac.agents.networks import ActorNetwork, CriticNetwork, StructuredActorNetwork
from uav_isac.domain.action import ActionSpace
from uav_isac.utils.types import Action


class MAPPOAgent(BaseAgent):
    """MAPPO CTDE agent with shared actor and centralized critic."""

    def __init__(
        self,
        agent_id: int,
        obs_dim: int,
        global_state_dim: int,
        action_space: ActionSpace,
        num_agents: int = 4,
        hidden_layers: Optional[list] = None,
        lr: float = 3e-4,
        critic_lr_mult: float = 5.0,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
        centralized_critic: bool = True,
        num_targets: int = 4,
        gru_hidden_dim: int = 0,
        single_frame_dim: int = 0,
        comm_num_rate_levels: int = 4,
        comm_log_std_init: float = -1.0,
        comm_entropy_scale: float = 1.0,
        use_comm_cross_attention: bool = False,
        comm_token_dim: int = 21,
        comm_tokens_per_sender: int = 1,
        comm_payload_dim: int = 16,
        comm_target_token_enabled: bool = False,
        comm_target_token_dim: int = 16,
        use_target_allocation: bool = False,
        use_team_sinkhorn: bool = False,
        capacity_matching_enabled: bool = False,
        capacity_matching_row_capacity: int = 2,
        capacity_matching_column_capacity: int = 2,
        capacity_matching_temperature: float = 0.35,
        capacity_matching_iterations: int = 32,
        capacity_matching_blend: float = 0.0,
        target_allocation_temperature: float = 1.0,
        target_allocation_straight_through: bool = False,
        target_allocation_movement_blend: float = 0.0,
        target_allocation_movement_confidence_gating_enabled: bool = False,
        target_allocation_movement_confidence_floor: float = 0.0,
        target_allocation_movement_confidence_power: float = 2.0,
        hierarchical_dual_assignment_enabled: bool = False,
        movement_team_matching_enabled: bool = False,
        movement_team_matching_temperature: float = 0.35,
        movement_team_matching_iterations: int = 16,
        movement_team_matching_blend: float = 0.0,
        movement_team_matching_intrinsic_bid_mix: float = 0.0,
        target_allocation_resource_blend: float = 0.0,
        round_negotiation_enabled: bool = False,
        round_negotiation_strength: float = 0.5,
        round_negotiation_temperature: float = 0.5,
        sparse_claim_enabled: bool = False,
        sparse_claim_share_topk: int = 2,
        sparse_claim_desired_endpoints: int = 2,
        sparse_claim_full_penalty: float = 2.0,
        sparse_claim_vacant_bonus: float = 0.5,
        sparse_claim_temperature: float = 0.25,
        comm_aided_sensing_enabled: bool = False,
        comm_aided_sensing_blend: float = 1.0,
        comm_semantic_decoder_enabled: bool = False,
        comm_semantic_capacity_bid_enabled: bool = False,
        comm_semantic_capacity_bid_gain: float = 0.0,
        comm_semantic_extra_token_enabled: bool = False,
        comm_semantic_extra_token_threshold: float = 0.10,
        semantic_kinematic_field_enabled: bool = False,
        semantic_kinematic_field_gain: float = 0.15,
        target_conditioned_movement_enabled: bool = False,
        target_conditioned_movement_gain: float = 0.15,
        architecture_v2_enabled: bool = False,
        architecture_v2_prior_gain: float = 1.0,
        architecture_v2_distance_weight: float = 0.25,
        architecture_v2_qos_floor: float = 0.60,
        architecture_v2_comm_prior_gain: float = 2.0,
        architecture_v2_comm_crisis_threshold: float = 0.25,
        architecture_v2_consensus_enabled: bool = True,
        architecture_v2_matching_temperature: float = 0.35,
        architecture_v2_movement_consensus_blend: float = 1.0,
        architecture_v2_endpoint_consensus_gain: float = 2.0,
        architecture_v2_bid_residual_scale: float = 0.25,
        architecture_v2_sensing_aligned_claims_enabled: bool = False,
        architecture_v2_modular_coordination_enabled: bool = False,
        architecture_v2_modular_num_experts: int = 3,
        architecture_v2_modular_gain: float = 0.25,
        architecture_v2_modular_temperature: float = 0.75,
        scale_equivariant_comm_heads_enabled: bool = False,
        permutation_equivariant_round_encoding_enabled: bool = False,
        comm_channel_feedback_rate_enabled: bool = False,
        comm_channel_feedback_dim: int = 6,
        equivariant_value_critic_enabled: bool = False,
        set_risk_critic_enabled: bool = False,
        risk_critic_hidden_dim: int = 128,
        risk_critic_num_quantiles: int = 16,
        risk_critic_cvar_alpha: float = 0.20,
        risk_critic_monotonic_quantiles_enabled: bool = False,
        isac_power_log_std_init: float = -1.0,
        sensing_allocation_log_std_init: float = -1.0,
    ):
        """
        Args:
            agent_id: UAV identifier
            obs_dim: Local observation dimension
            global_state_dim: Global state dimension
            action_space: Action space definition
            hidden_layers: Hidden layer sizes
            lr: Adam learning rate
            max_grad_norm: Gradient clipping max norm
            device: "cpu" or "cuda"
            num_targets: Number of targets (Q) — from config, NOT hardcoded
            gru_hidden_dim: GRU hidden dimension (0 = flat-MLP actor, no GRU)
        """
        super().__init__(agent_id)

        if hidden_layers is None:
            hidden_layers = [256, 256]

        self.device = torch.device(device)
        self.action_space = action_space
        self.max_grad_norm = max_grad_norm
        self.critic_lr_mult = critic_lr_mult
        self.num_targets = num_targets
        self.gru_hidden_dim = gru_hidden_dim
        self.comm_entropy_scale = float(max(0.0, comm_entropy_scale))
        self.comm_payload_dim = max(1, int(comm_payload_dim))
        # centralized critic (MAPPO/CTDE, input=global state) vs
        # decentralized critic (IPPO, input=local obs). de Witt et al. 2020.
        self.centralized_critic = centralized_critic
        if set_risk_critic_enabled and not centralized_critic:
            raise ValueError(
                'set_risk_critic_enabled requires MAPPO centralized training; '
                'the risk critic is removed at decentralized execution')
        if equivariant_value_critic_enabled and not centralized_critic:
            raise ValueError(
                'equivariant_value_critic_enabled requires MAPPO '
                'centralized training')
        critic_state_dim = global_state_dim if centralized_critic else obs_dim
        self._critic_base_state_dim = int(critic_state_dim)
        self._critic_aux_dim = int(num_agents + 16)

        # Networks
        if getattr(action_space, 'structured_actor', False):
            entity_dim = getattr(action_space, 'structured_entity_dim', 64)
            self.actor = StructuredActorNetwork(
                obs_dim=obs_dim, K=num_agents, Q=num_targets,
                entity_dim=entity_dim, max_dp=action_space.max_dp,
                single_frame_dim=single_frame_dim,
                comm_num_rate_levels=comm_num_rate_levels,
                comm_log_std_init=comm_log_std_init,
                use_comm_cross_attention=use_comm_cross_attention,
                comm_token_dim=comm_token_dim,
                comm_tokens_per_sender=comm_tokens_per_sender,
                comm_payload_dim=self.comm_payload_dim,
                comm_target_token_enabled=comm_target_token_enabled,
                comm_target_token_dim=comm_target_token_dim,
                use_target_allocation=use_target_allocation,
                use_team_sinkhorn=use_team_sinkhorn,
                capacity_matching_enabled=capacity_matching_enabled,
                capacity_matching_row_capacity=(
                    capacity_matching_row_capacity),
                capacity_matching_column_capacity=(
                    capacity_matching_column_capacity),
                capacity_matching_temperature=(
                    capacity_matching_temperature),
                capacity_matching_iterations=capacity_matching_iterations,
                capacity_matching_blend=capacity_matching_blend,
                target_allocation_temperature=(
                    target_allocation_temperature),
                target_allocation_straight_through=(
                    target_allocation_straight_through),
                target_allocation_movement_blend=(
                    target_allocation_movement_blend),
                target_allocation_movement_confidence_gating_enabled=(
                    target_allocation_movement_confidence_gating_enabled),
                target_allocation_movement_confidence_floor=(
                    target_allocation_movement_confidence_floor),
                target_allocation_movement_confidence_power=(
                    target_allocation_movement_confidence_power),
                hierarchical_dual_assignment_enabled=(
                    hierarchical_dual_assignment_enabled),
                movement_team_matching_enabled=(
                    movement_team_matching_enabled),
                movement_team_matching_temperature=(
                    movement_team_matching_temperature),
                movement_team_matching_iterations=(
                    movement_team_matching_iterations),
                movement_team_matching_blend=(
                    movement_team_matching_blend),
                movement_team_matching_intrinsic_bid_mix=(
                    movement_team_matching_intrinsic_bid_mix),
                target_allocation_resource_blend=(
                    target_allocation_resource_blend),
                round_negotiation_enabled=round_negotiation_enabled,
                round_negotiation_strength=round_negotiation_strength,
                round_negotiation_temperature=(
                    round_negotiation_temperature),
                sparse_claim_enabled=sparse_claim_enabled,
                sparse_claim_share_topk=sparse_claim_share_topk,
                sparse_claim_desired_endpoints=(
                    sparse_claim_desired_endpoints),
                sparse_claim_full_penalty=sparse_claim_full_penalty,
                sparse_claim_vacant_bonus=sparse_claim_vacant_bonus,
                sparse_claim_temperature=sparse_claim_temperature,
                comm_aided_sensing_enabled=comm_aided_sensing_enabled,
                comm_aided_sensing_blend=comm_aided_sensing_blend,
                comm_semantic_decoder_enabled=(
                    comm_semantic_decoder_enabled),
                comm_semantic_capacity_bid_enabled=(
                    comm_semantic_capacity_bid_enabled),
                comm_semantic_capacity_bid_gain=(
                    comm_semantic_capacity_bid_gain),
                comm_semantic_extra_token_enabled=(
                    comm_semantic_extra_token_enabled),
                comm_semantic_extra_token_threshold=(
                    comm_semantic_extra_token_threshold),
                semantic_kinematic_field_enabled=(
                    semantic_kinematic_field_enabled),
                semantic_kinematic_field_gain=(
                    semantic_kinematic_field_gain),
                target_conditioned_movement_enabled=(
                    target_conditioned_movement_enabled),
                target_conditioned_movement_gain=(
                    target_conditioned_movement_gain),
                architecture_v2_enabled=architecture_v2_enabled,
                architecture_v2_prior_gain=architecture_v2_prior_gain,
                architecture_v2_distance_weight=(
                    architecture_v2_distance_weight),
                architecture_v2_qos_floor=architecture_v2_qos_floor,
                architecture_v2_comm_prior_gain=(
                    architecture_v2_comm_prior_gain),
                architecture_v2_comm_crisis_threshold=(
                    architecture_v2_comm_crisis_threshold),
                architecture_v2_consensus_enabled=(
                    architecture_v2_consensus_enabled),
                architecture_v2_matching_temperature=(
                    architecture_v2_matching_temperature),
                architecture_v2_movement_consensus_blend=(
                    architecture_v2_movement_consensus_blend),
                architecture_v2_endpoint_consensus_gain=(
                    architecture_v2_endpoint_consensus_gain),
                architecture_v2_bid_residual_scale=(
                    architecture_v2_bid_residual_scale),
                architecture_v2_sensing_aligned_claims_enabled=(
                    architecture_v2_sensing_aligned_claims_enabled),
                architecture_v2_modular_coordination_enabled=(
                    architecture_v2_modular_coordination_enabled),
                architecture_v2_modular_num_experts=(
                    architecture_v2_modular_num_experts),
                architecture_v2_modular_gain=(
                    architecture_v2_modular_gain),
                architecture_v2_modular_temperature=(
                    architecture_v2_modular_temperature),
                scale_equivariant_comm_heads_enabled=(
                    scale_equivariant_comm_heads_enabled),
                permutation_equivariant_round_encoding_enabled=(
                    permutation_equivariant_round_encoding_enabled),
                comm_channel_feedback_rate_enabled=(
                    comm_channel_feedback_rate_enabled),
                comm_channel_feedback_dim=comm_channel_feedback_dim,
                isac_power_log_std_init=isac_power_log_std_init,
                sensing_allocation_log_std_init=(
                    sensing_allocation_log_std_init),
            ).to(self.device)
        else:
            self.actor = ActorNetwork(
                obs_dim=obs_dim,
                hidden_layers=hidden_layers,
                max_dp=action_space.max_dp,
                comm_num_rate_levels=comm_num_rate_levels,
                comm_log_std_init=comm_log_std_init,
                num_targets=num_targets,
                comm_payload_dim=self.comm_payload_dim,
                isac_power_log_std_init=isac_power_log_std_init,
                sensing_allocation_log_std_init=(
                    sensing_allocation_log_std_init),
            ).to(self.device)

        self.critic = CriticNetwork(
            state_dim=critic_state_dim,
            hidden_layers=[h*2 for h in hidden_layers] + [hidden_layers[-1]*2],  # 512×3
            num_agents=num_agents,
            comm_dim=16,
            num_targets=num_targets,
            equivariant_value_critic_enabled=(
                equivariant_value_critic_enabled),
            set_risk_critic_enabled=set_risk_critic_enabled,
            risk_hidden_dim=risk_critic_hidden_dim,
            risk_num_quantiles=risk_critic_num_quantiles,
            risk_cvar_alpha=risk_critic_cvar_alpha,
            risk_monotonic_quantiles_enabled=(
                risk_critic_monotonic_quantiles_enabled),
        ).to(self.device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr * critic_lr_mult
        )

    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[Action, float, float]:
        """Select action from local observation.

        Args:
            obs: Local observation vector (obs_dim,)
            deterministic: If True, use mean/mode

        Returns:
            (Action, log_prob, value) — value is 0 for decentralized execution
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            dp_mean, dp_log_std, role_logits, _, _, _ = self.actor(obs_t)

            dp_mean_np = dp_mean.squeeze(0).detach().cpu().numpy()
            dp_std_np = dp_log_std.detach().cpu().numpy()
            role_logits_np = role_logits.squeeze(0).detach().cpu().numpy()

        action, log_prob = self.action_space.decode(
            dp_mean_np, dp_std_np, role_logits_np, deterministic=deterministic
        )

        return action, log_prob, 0.0

    def sample_communication(
        self,
        comm_mean: torch.Tensor,
        deterministic: bool = False,
        return_components: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample learned message content and a variable-rate/silence action.

        Rate index zero is silence.  For active rates, message content is a
        stochastic Gaussian action.  Both terms enter the PPO log-probability,
        allowing sensing reward and explicit transport cost to train the
        sending policy rather than only the receiving network.
        """
        if not hasattr(self.actor, 'communication_parameters'):
            raise RuntimeError('actor does not support cost-aware communication')
        comm_log_std, rate_logits = self.actor.communication_parameters(comm_mean)
        comm_std = torch.exp(comm_log_std).expand_as(comm_mean)
        msg_dist = torch.distributions.Normal(comm_mean, comm_std)
        rate_dist = torch.distributions.Categorical(logits=rate_logits)

        if deterministic:
            comm_action = comm_mean
            rate_action = torch.argmax(rate_logits, dim=-1)
        else:
            comm_action = msg_dist.sample()
            rate_action = rate_dist.sample()

        active = (rate_action > 0).to(comm_mean.dtype)
        msg_log_prob = msg_dist.log_prob(comm_action).sum(dim=-1) * active
        rate_log_prob = rate_dist.log_prob(rate_action)
        total_log_prob = msg_log_prob + rate_log_prob
        # Keep the entropy bonus on the same scale as the other action heads;
        # the PPO log-probability remains the exact joint (summed) density.
        entropy = rate_dist.entropy() + msg_dist.entropy().mean(dim=-1) * active
        if return_components:
            return (comm_action, rate_action, total_log_prob, entropy,
                    {'message': msg_log_prob, 'rate': rate_log_prob})
        return comm_action, rate_action, total_log_prob, entropy

    def sample_isac_resources(
        self,
        comm_mean: torch.Tensor,
        rate_action: torch.Tensor,
        deterministic: bool = False,
        comm_fraction_min: float = 0.0,
        comm_fraction_max: float = 1.0,
    ):
        """Sample variable communication power and a multi-target sensing split.

        The stored actions are unconstrained logistic-normal samples.  The
        environment receives their sigmoid/softmax transforms, while PPO uses
        the exact Gaussian log probability of the stored raw actions.
        """
        if not hasattr(self.actor, 'isac_resource_parameters'):
            raise RuntimeError('actor does not support joint ISAC resources')
        p_mean, p_log_std, s_mean, s_log_std = (
            self.actor.isac_resource_parameters(comm_mean))
        p_std = torch.exp(p_log_std).expand_as(p_mean)
        s_std = torch.exp(s_log_std).expand_as(s_mean)
        p_dist = torch.distributions.Normal(p_mean, p_std)
        s_dist = torch.distributions.Normal(s_mean, s_std)
        if deterministic:
            power_raw = p_mean
            sensing_raw = s_mean
        else:
            power_raw = p_dist.sample()
            sensing_raw = s_dist.sample()
        active = (rate_action > 0).to(comm_mean.dtype)
        lo = float(np.clip(comm_fraction_min, 0.0, 1.0))
        hi = float(np.clip(comm_fraction_max, lo, 1.0))
        comm_fraction = (lo + (hi - lo) * torch.sigmoid(power_raw)) * active
        sensing_weights = torch.softmax(sensing_raw, dim=-1)
        log_prob = p_dist.log_prob(power_raw) * active
        log_prob = log_prob + s_dist.log_prob(sensing_raw).sum(dim=-1)
        entropy = p_dist.entropy() * active + s_dist.entropy().mean(dim=-1)
        return (power_raw, sensing_raw, comm_fraction, sensing_weights,
                log_prob, entropy)

    def _movement_message_resource_log_probs(
        self,
        dp_mean: torch.Tensor,
        dp_log_std: torch.Tensor,
        role_logits: torch.Tensor,
        comm_msgs: torch.Tensor,
        actions_dp: torch.Tensor,
        actions_role: torch.Tensor,
        movement_action_mask: Optional[torch.Tensor],
        actions_comm: Optional[torch.Tensor],
        actions_comm_rate: Optional[torch.Tensor],
        actions_isac_power_raw: Optional[torch.Tensor],
        actions_sensing_raw: Optional[torch.Tensor],
        detach_head_context: bool = False,
        return_components: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single source of the movement+message+resource log-prob math.

        P2-1 (CODE_STRUCTURE_MAP 2026-08-25): this is the exact body that was
        previously duplicated byte-for-byte between ``evaluate_actions`` and
        ``verify_old_log_prob_consistency``.  The PPO ratio guard only holds
        while both consumers agree, so the math now lives in exactly one
        place; the verify path calls this under ``no_grad`` and evaluates with
        the graph when training.

        Returns ``(new_log_probs, entropies)``.
        """
        N = dp_mean.shape[0]
        new_log_probs = torch.zeros(N, device=self.device)
        entropies = torch.zeros(N, device=self.device)

        # Tanh-squashed Gaussian log prob for delta_p
        # Inverse through tanh: atanh(y) where y = dp / dp_scale
        # (MUST use the same dp_scale as ActionSpace.decode/compute_log_prob,
        #  otherwise old_log_prob (sampling) != new_log_prob (update) -> broken ratio)
        # Under dp_parameterization="smooth_disk" (B5, advice 014) the inverse is
        # z = dp / sqrt(d_max^2 - |dp|^2) and the exact change-of-variables
        # correction +2*log(1+|z|^2) - 2*log(d_max) is added, so the PPO ratio is
        # exact for every executed action (no many-to-one radial projection).
        if self.action_space.dp_parameterization == "smooth_disk":
            # float64 intermediates and the SAME formula as ActionSpace.
            # compute_log_prob (numpy): standardized = (z-mu)/max(sigma,1e-12),
            # log N = -0.5*std^2 - log(sigma) - 0.5*log(2pi).  A var+eps form
            # would perturb the quadratic term by ~1e-6 relative, which is
            # amplified to ~1e-4 for the rare large-|z| near-rim actions.
            d_max = self.action_space.max_dp
            dp64 = actions_dp.double()
            norm_sq = (dp64 ** 2).sum(dim=-1)
            denom_sq = (d_max ** 2 - norm_sq).clamp(min=1e-30)
            dp_raw = dp64 / denom_sq.sqrt().unsqueeze(-1)
            dp_std_pos = torch.exp(torch.clamp(
                dp_log_std.double(), -20, 2))
            var = dp_std_pos ** 2  # also used by the entropy term below
            std_safe = dp_std_pos.clamp(min=1e-12)
            standardized = (dp_raw - dp_mean.double()) / std_safe
            log_prob_dp = (
                -0.5 * (standardized ** 2).sum(dim=-1)
                - torch.log(std_safe).sum(dim=-1)
                - dp_raw.shape[-1] * 0.5 * np.log(2.0 * np.pi)
                + 2.0 * torch.log1p((dp_raw ** 2).sum(dim=-1))
                - 2.0 * np.log(d_max)
            ).float()
        else:
            dp_norm = actions_dp / self.action_space.dp_scale
            dp_norm = torch.clamp(dp_norm, -0.999, 0.999)
            dp_raw = torch.atanh(dp_norm)
            dp_std_pos = torch.exp(torch.clamp(dp_log_std, -20, 2))
            var = dp_std_pos ** 2
            log_prob_dp = -0.5 * (
                ((dp_raw - dp_mean) ** 2) / (var + 1e-6)
                + torch.log(2 * np.pi * var + 1e-6)
            ).sum(dim=-1)
            log_prob_dp -= torch.log(1.0 - dp_norm ** 2 + 1e-6).sum(dim=-1)

        # Gaussian (delta_p) entropy
        entropy_dp = 0.5 * torch.log(2 * np.pi * np.e * var + 1e-6).sum(dim=-1)

        movement_mask = (
            torch.ones(N, dtype=log_prob_dp.dtype, device=self.device)
            if movement_action_mask is None
            else movement_action_mask.to(
                device=self.device, dtype=log_prob_dp.dtype).reshape(-1)
        )

        if not getattr(self.action_space, 'learn_roles', True):
            # Role is assigned by the env's P0 solver, not the policy -> drop the
            # role term from BOTH log-prob and entropy so the role head carries no
            # gradient. MUST match ActionSpace.compute_log_prob (which also drops
            # it) to keep old_log_prob == new_log_prob in the PPO ratio.
            new_log_probs = log_prob_dp * movement_mask
            entropies = entropy_dp * movement_mask
        else:
            # Categorical log prob for role
            log_probs_role = torch.log_softmax(role_logits, dim=-1)
            log_prob_role = log_probs_role.gather(1, actions_role.unsqueeze(-1)).squeeze(-1)
            new_log_probs = (log_prob_dp + log_prob_role) * movement_mask

            # Role (categorical) entropy
            probs = torch.softmax(role_logits, dim=-1)
            entropy_role = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            entropies = (entropy_dp + entropy_role) * movement_mask

        # Per-head log-prob components (audit/headwise-credit consumers; zeros
        # when the corresponding head is not executed this frame).
        movement_lp = new_log_probs
        message_lp = torch.zeros_like(new_log_probs)
        rate_lp = torch.zeros_like(new_log_probs)
        resource_lp = torch.zeros_like(new_log_probs)

        # Optional cost-aware communication action. Legacy/off modes omit these
        # tensors and therefore retain the historical action distribution.
        if actions_comm is not None and actions_comm_rate is not None:
            # In headwise-credit mode the rate/resource policies may condition
            # numerically on the current token mean, but their losses must not
            # rewrite the token encoder. Movement and message still share the
            # actor representation; only cross-head output leakage is stopped.
            head_context = (
                comm_msgs.detach()
                if detach_head_context else comm_msgs)
            comm_log_std, rate_logits = self.actor.communication_parameters(
                head_context)
            comm_std = torch.exp(comm_log_std).expand_as(comm_msgs)
            msg_dist = torch.distributions.Normal(comm_msgs, comm_std)
            rate_dist = torch.distributions.Categorical(logits=rate_logits)
            active = (actions_comm_rate > 0).to(comm_msgs.dtype)
            message_lp = msg_dist.log_prob(actions_comm).sum(dim=-1) * active
            rate_lp = rate_dist.log_prob(actions_comm_rate)
            new_log_probs = new_log_probs + message_lp + rate_lp
            comm_entropy = rate_dist.entropy() + msg_dist.entropy().mean(dim=-1) * active
            entropies = entropies + self.comm_entropy_scale * comm_entropy

            if (actions_isac_power_raw is not None
                    and actions_sensing_raw is not None):
                p_mean, p_log_std, s_mean, s_log_std = (
                    self.actor.isac_resource_parameters(head_context))
                p_dist = torch.distributions.Normal(
                    p_mean, torch.exp(p_log_std).expand_as(p_mean))
                s_dist = torch.distributions.Normal(
                    s_mean, torch.exp(s_log_std).expand_as(s_mean))
                active = (actions_comm_rate > 0).to(comm_msgs.dtype)
                resource_lp = (
                    p_dist.log_prob(actions_isac_power_raw) * active
                    + s_dist.log_prob(actions_sensing_raw).sum(dim=-1)
                )
                new_log_probs = new_log_probs + resource_lp
                resource_entropy = (
                    p_dist.entropy() * active
                    + s_dist.entropy().mean(dim=-1)
                )
                entropies = (
                    entropies + self.comm_entropy_scale * resource_entropy)

        if return_components:
            return (new_log_probs, entropies, {
                'movement_lp': movement_lp,
                'message_lp': message_lp,
                'rate_lp': rate_lp,
                'resource_lp': resource_lp,
            })
        return new_log_probs, entropies

    def evaluate_actions(
        self,
        obs: torch.Tensor,               # (batch, obs_dim) or (batch, L, obs_dim)
        global_state: torch.Tensor,      # (batch, global_state_dim)
        actions_dp: torch.Tensor,        # (batch, 2)
        actions_role: torch.Tensor,      # (batch,)
        h_prev: torch.Tensor = None,     # (1, batch*(K-1), D) GRU hidden states
        window_mask: torch.Tensor = None,  # (batch, L) TICA window mask
        actions_comm: torch.Tensor = None,       # (batch, 16) sampled message
        actions_comm_rate: torch.Tensor = None,  # (batch,) 0=silence
        actions_isac_power_raw: torch.Tensor = None,  # (batch,)
        actions_sensing_raw: torch.Tensor = None,     # (batch,Q)
        movement_action_mask: torch.Tensor = None,    # (batch,), 1=new action
        comm_round_phase: torch.Tensor = None,        # (batch,), 0=proposal/1=response
        agent_identity: torch.Tensor = None,          # (batch,), local UAV id
        return_log_prob_components: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update (computational graph enabled).

        P0 FIX: Accepts and passes h_prev (GRU hidden states) to the actor,
        ensuring the PPO ratio compares distributions conditioned on the SAME
        hidden state as during rollout. Without this, the ratio r_t can deviate
        from 1 even before the first optimizer step.

        Args:
            obs: Batch of local observations
            global_state: Batch of global states
            actions_dp: Taken delta_p actions
            actions_role: Taken role actions
            h_prev: (1, batch*(K-1), D) GRU hidden states from rollout buffer,
                    or None for flat-MLP actor (no GRU).

        Returns:
            (new_log_probs, values, entropies, dp_means)
        """
        dp_mean, dp_log_std, role_logits, comm_msgs, pd_pred, _h = self.actor(
            obs, h_prev, window_mask=window_mask,
            comm_round_phase=comm_round_phase,
            agent_identity=agent_identity)
        # Normal trainer batches already append agent one-hot + 16-D message
        # summary. Keep evaluate_actions compatible with diagnostic callers
        # that pass only the base state; the omitted auxiliary context is then
        # explicitly interpreted as unavailable/zero rather than causing an
        # opaque matrix-shape failure.
        if global_state.shape[-1] == self._critic_base_state_dim:
            aux = torch.zeros(
                *global_state.shape[:-1], self._critic_aux_dim,
                dtype=global_state.dtype, device=global_state.device)
            global_state = torch.cat([global_state, aux], dim=-1)
        credit_values = None
        if return_log_prob_components:
            values, credit_values = self.critic.forward_with_credit(global_state)
        else:
            values = self.critic(global_state)

        if return_log_prob_components:
            new_log_probs, entropies, components = (
                self._movement_message_resource_log_probs(
                    dp_mean, dp_log_std, role_logits, comm_msgs,
                    actions_dp, actions_role, movement_action_mask,
                    actions_comm, actions_comm_rate,
                    actions_isac_power_raw, actions_sensing_raw,
                    return_components=True))
            head_outputs = {
                'log_probs': torch.stack([
                    components['movement_lp'],
                    components['message_lp'],
                    components['rate_lp'],
                    components['resource_lp'],
                ], dim=-1),
                'credit_values': credit_values,
            }
            return (new_log_probs, values, entropies, dp_mean, pd_pred,
                    comm_msgs, head_outputs)
        new_log_probs, entropies = self._movement_message_resource_log_probs(
            dp_mean, dp_log_std, role_logits, comm_msgs,
            actions_dp, actions_role, movement_action_mask,
            actions_comm, actions_comm_rate,
            actions_isac_power_raw, actions_sensing_raw)
        return new_log_probs, values, entropies, dp_mean, pd_pred, comm_msgs

    def verify_old_log_prob_consistency(
        self,
        obs: torch.Tensor,
        actions_dp: torch.Tensor,
        actions_role: torch.Tensor,
        old_log_probs: torch.Tensor,
        h_prev: torch.Tensor = None,
        tolerance: float = 1e-4,
        window_mask: torch.Tensor = None,
        actions_comm: torch.Tensor = None,
        actions_comm_rate: torch.Tensor = None,
        actions_isac_power_raw: torch.Tensor = None,
        actions_sensing_raw: torch.Tensor = None,
        movement_action_mask: torch.Tensor = None,
        comm_round_phase: torch.Tensor = None,
        agent_identity: torch.Tensor = None,
    ) -> Tuple[bool, float]:
        """P0 ASSERTION: verify that recomputed log-probs match stored old log-probs.

        This MUST be run before any optimizer step. If max|old - recomputed| >
        tolerance, the PPO ratio is invalid from the start and training results
        are contaminated.

        Args:
            obs: Batch observations (same as used in rollout)
            actions_dp: Taken dp actions
            actions_role: Taken role actions
            old_log_probs: Stored log-probs from rollout
            h_prev: GRU hidden states from rollout (must match)
            tolerance: Maximum allowed absolute difference

        Returns:
            (passed, max_abs_diff)
        """
        with torch.no_grad():
            # Use the same code path as evaluate_actions but with no_grad
            dp_mean, dp_log_std, role_logits, comm_mean, _, _ = self.actor(
                obs, h_prev, window_mask=window_mask,
                comm_round_phase=comm_round_phase,
                agent_identity=agent_identity)

            new_log_probs, _ = (
                self._movement_message_resource_log_probs(
                    dp_mean, dp_log_std, role_logits, comm_mean,
                    actions_dp, actions_role, movement_action_mask,
                    actions_comm, actions_comm_rate,
                    actions_isac_power_raw, actions_sensing_raw))
            max_diff = (old_log_probs - new_log_probs).abs().max().item()
            passed = max_diff < tolerance
        return passed, max_diff

    def update(self, rollout_data: Dict) -> Dict[str, float]:
        """Single PPO update is handled by the trainer.

        Args:
            rollout_data: Training batch data (unused — trainer handles this)

        Returns:
            Empty dict (training is done by MAPPTrainer)
        """
        return {}

    def load_actor_state_dict_compatible(self, state_dict: Dict):
        """Load an actor checkpoint while expanding its rate-action levels.

        Adding a higher quantization rate changes only the first dimension of
        `comm_rate_head`. PyTorch's `strict=False` still rejects shape
        mismatches, so copy all historical rows and retain the actor's neutral
        initialization for any newly appended rate levels.
        """
        migrated = dict(state_dict)
        current = self.actor.state_dict()
        token_dim = int(getattr(
            self.actor, 'comm_target_token_dim', 0))

        # The identity-free modular residual is behavior-neutral because its
        # router starts exactly uniform and expert outputs are mixed after
        # subtracting that uniform distribution. Preserve its distinct small
        # expert bases during checkpoint migration; generic zero-init would
        # make both the router and every expert zero and block first-step
        # teacher gradients.
        if getattr(
                self.actor,
                '_architecture_v2_modular_coordination_enabled', False):
            for key, value in current.items():
                if (key.startswith('v2_module_router.')
                        or key.startswith('v2_coordination_experts.')):
                    migrated.setdefault(key, value.clone())

        # Convert fixed-Q flattened heads into local set heads. With uniform
        # pooling, summing the legacy per-target blocks is the least-assumption
        # projection of y=sum_q W_q z_q+b onto y=W_set mean_q(z_q)+b.
        if (getattr(
                self.actor, '_scale_equivariant_comm_heads_enabled', False)
                and token_dim > 0):
            for old_prefix, new_prefix in (
                    ('comm_rate_head', 'comm_set_rate_head'),
                    ('isac_power_mean_head', 'isac_set_power_head')):
                old_weight = state_dict.get(f'{old_prefix}.weight')
                old_bias = state_dict.get(f'{old_prefix}.bias')
                new_weight_key = f'{new_prefix}.weight'
                new_bias_key = f'{new_prefix}.bias'
                if (old_weight is not None and old_weight.ndim == 2
                        and old_weight.shape[1] % token_dim == 0
                        and new_weight_key in current):
                    blocks = old_weight.reshape(
                        old_weight.shape[0], -1, token_dim)
                    migrated[new_weight_key] = blocks.sum(dim=1).to(
                        current[new_weight_key].device)
                    if old_bias is not None and new_bias_key in current:
                        migrated[new_bias_key] = old_bias.to(
                            current[new_bias_key].device)
            for key in (
                    'comm_set_attention.weight',
                    'comm_set_attention.bias'):
                if key in current:
                    migrated[key] = current[key].clone()

        # Remove absolute K-dependent identity while retaining the learned
        # proposal/response phase effect. The mean old identity column becomes
        # a shared bias, which is invariant to UAV relabelling.
        if getattr(
                self.actor,
                '_permutation_equivariant_round_encoding_enabled', False):
            old_weight = state_dict.get('round_phase_enc.0.weight')
            old_bias = state_dict.get('round_phase_enc.0.bias')
            new_weight_key = 'round_phase_equivariant_enc.0.weight'
            new_bias_key = 'round_phase_equivariant_enc.0.bias'
            if (old_weight is not None and old_weight.ndim == 2
                    and old_weight.shape[1] >= 2
                    and new_weight_key in current):
                migrated[new_weight_key] = old_weight[:, :2].to(
                    current[new_weight_key].device)
                base_bias = (
                    old_bias
                    if old_bias is not None
                    else torch.zeros(
                        old_weight.shape[0],
                        dtype=old_weight.dtype,
                        device=old_weight.device))
                identity_columns = old_weight[:, 2:]
                shared_identity = (
                    identity_columns.mean(dim=1)
                    if identity_columns.numel() > 0
                    else torch.zeros_like(base_bias))
                migrated[new_bias_key] = (
                    base_bias + shared_identity).to(
                        current[new_bias_key].device)
            for suffix in ('weight', 'bias'):
                old_key = f'round_phase_enc.2.{suffix}'
                new_key = f'round_phase_equivariant_enc.2.{suffix}'
                if (old_key in state_dict and new_key in current
                        and state_dict[old_key].shape == current[new_key].shape):
                    migrated[new_key] = state_dict[old_key].to(
                        current[new_key].device)
        expandable_weights = {
            'comm_rate_head.weight', 'intent_head.weight',
            'isac_power_mean_head.weight',
            'isac_sensing_mean_head.weight',
        }
        for key in list(migrated):
            if key not in migrated or key not in current:
                continue
            old_value = migrated[key]
            new_value = current[key]
            if old_value.shape == new_value.shape:
                continue
            expanded = None
            if key == 'comm_token_enc.0.weight':
                expanded = new_value.clone()
                expanded.zero_()
                rows = min(old_value.shape[0], new_value.shape[0])
                if old_value.shape[1] == 21 and new_value.shape[1] == 22:
                    expanded[:rows, :16].copy_(old_value[:rows, :16])
                    expanded[:rows, 16].copy_(old_value[:rows, 16])
                    expanded[:rows, 18:22].copy_(old_value[:rows, 17:21])
                else:
                    cols = min(old_value.shape[1], new_value.shape[1])
                    expanded[:rows, :cols].copy_(old_value[:rows, :cols])
            elif (key in expandable_weights and old_value.ndim == 2
                  and new_value.ndim == 2):
                expanded = new_value.clone()
                expanded.zero_()
                rows = min(old_value.shape[0], new_value.shape[0])
                if new_value.shape[1] % old_value.shape[1] == 0:
                    repeats = new_value.shape[1] // old_value.shape[1]
                    tiled = old_value[:rows].repeat(1, repeats) / repeats
                    expanded[:rows].copy_(tiled)
                else:
                    cols = min(old_value.shape[1], new_value.shape[1])
                    expanded[:rows, :cols].copy_(old_value[:rows, :cols])
            elif key == 'comm_log_std' and old_value.ndim == 1:
                if new_value.numel() % old_value.numel() == 0:
                    expanded = old_value.repeat(
                        new_value.numel() // old_value.numel())
            elif old_value.shape[1:] == new_value.shape[1:]:
                expanded = new_value.clone()
                rows = min(old_value.shape[0], new_value.shape[0])
                expanded[:rows].copy_(old_value[:rows])
            if expanded is None:
                # strict=False still rejects shape mismatches; retain the new
                # module's initialization when no safe migration exists.
                migrated.pop(key)
            else:
                migrated[key] = expanded.to(new_value.device)
        result = self.actor.load_state_dict(migrated, strict=False)
        # A compatible warm-start must be behaviour-preserving.  Leaving newly
        # introduced modules at their random constructor values makes an
        # ablation depend on the process seed before it has seen one gradient.
        # Zero only parameters absent from the migrated checkpoint; expanded
        # historical parameters above retain their copied rows.
        if result.missing_keys and hasattr(self.actor, 'zero_init_new_layers'):
            known_keys = set(current).difference(result.missing_keys)
            self.actor.zero_init_new_layers(known_keys)
            # The capacity bid is bilinear: <W_msg m, W_target t>.  Zeroing
            # both projections preserves behaviour but also makes both
            # gradients exactly zero.  An identity target projection plus a
            # zero message projection remains a neutral (zero-logit) start
            # while allowing W_msg to learn on the first update.
            missing = set(result.missing_keys)
            target_weight = 'neighbor_bid_target_proj.weight'
            if (target_weight in missing
                    and hasattr(self.actor, 'neighbor_bid_target_proj')):
                with torch.no_grad():
                    weight = self.actor.neighbor_bid_target_proj.weight
                    weight.zero_()
                    diagonal = min(weight.shape)
                    weight[:diagonal, :diagonal].copy_(
                        torch.eye(diagonal, device=weight.device,
                                  dtype=weight.dtype))
            # The hierarchical motion head replaces the historical shared
            # assignment output.  Copy that trained prior rather than leaving
            # a new random or all-zero policy, while keeping future gradients
            # independent from endpoint-capacity learning.
            if (any(key.startswith('movement_commitment_head.')
                    for key in missing)
                    and hasattr(self.actor, 'movement_commitment_head')
                    and hasattr(self.actor, 'target_assignment_head')):
                self.actor.movement_commitment_head.load_state_dict(
                    self.actor.target_assignment_head.state_dict())
        return result

    def get_params(self) -> Dict:
        """Get trainable parameters for federated aggregation (Phase 2)."""
        return {
            'actor': deepcopy(self.actor.state_dict()),
            'critic': deepcopy(self.critic.state_dict()),
        }

    def set_params(self, params: Dict) -> None:
        """Set parameters from federated aggregation (Phase 2)."""
        self.actor.load_state_dict(params['actor'])
        self.critic.load_state_dict(params['critic'])
