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
from uav_isac.environment.action import ActionSpace
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
        # centralized critic (MAPPO/CTDE, input=global state) vs
        # decentralized critic (IPPO, input=local obs). de Witt et al. 2020.
        self.centralized_critic = centralized_critic
        critic_state_dim = global_state_dim if centralized_critic else obs_dim

        # Networks
        if getattr(action_space, 'structured_actor', False):
            entity_dim = getattr(action_space, 'structured_entity_dim', 64)
            self.actor = StructuredActorNetwork(
                obs_dim=obs_dim, K=num_agents, Q=num_targets,
                entity_dim=entity_dim, max_dp=action_space.max_dp,
            ).to(self.device)
        else:
            self.actor = ActorNetwork(
                obs_dim=obs_dim,
                hidden_layers=hidden_layers,
                max_dp=action_space.max_dp,
            ).to(self.device)

        self.critic = CriticNetwork(
            state_dim=critic_state_dim,
            hidden_layers=[h*2 for h in hidden_layers] + [hidden_layers[-1]*2],  # 512×3
            num_agents=num_agents,
            comm_dim=16,
            num_targets=num_targets,
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

    def act_batch(
        self,
        obs_batch: np.ndarray,  # (batch, obs_dim)
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Select actions for a batch of observations.

        Args:
            obs_batch: (N, obs_dim)
            deterministic: If True, use mean/mode

        Returns:
            (actions_dp (N,2), actions_role (N,), log_probs (N,), values (N,), entropies (N,))
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
            dp_mean, dp_log_std, role_logits, _, _, _ = self.actor(obs_t)

        dp_mean_np = dp_mean.detach().cpu().numpy()
        dp_std_np = dp_log_std.detach().cpu().numpy()
        role_logits_np = role_logits.detach().cpu().numpy()

        N = obs_batch.shape[0]
        actions_dp = np.zeros((N, 2), dtype=np.float64)
        actions_role = np.zeros(N, dtype=np.int32)
        log_probs = np.zeros(N, dtype=np.float64)

        for i in range(N):
            action, lp = self.action_space.decode(
                dp_mean_np[i], dp_std_np, role_logits_np[i], deterministic=deterministic
            )
            actions_dp[i] = action.delta_p
            actions_role[i] = action.role
            log_probs[i] = lp

        return actions_dp, actions_role, log_probs, np.zeros(N), np.zeros(N)

    def evaluate_actions(
        self,
        obs: torch.Tensor,               # (batch, obs_dim)
        global_state: torch.Tensor,      # (batch, global_state_dim)
        actions_dp: torch.Tensor,        # (batch, 2)
        actions_role: torch.Tensor,      # (batch,)
        h_prev: torch.Tensor = None,     # (1, batch*(K-1), D) GRU hidden states
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
        dp_mean, dp_log_std, role_logits, comm_msgs, pd_pred, _h = self.actor(obs, h_prev)
        values = self.critic(global_state)

        # Compute log probs and entropy
        N = obs.shape[0]
        new_log_probs = torch.zeros(N, device=self.device)
        entropies = torch.zeros(N, device=self.device)

        # Tanh-squashed Gaussian log prob for delta_p
        # Inverse through tanh: atanh(y) where y = dp / dp_scale
        # (MUST use the same dp_scale as ActionSpace.decode/compute_log_prob,
        #  otherwise old_log_prob (sampling) != new_log_prob (update) -> broken ratio)
        dp_norm = actions_dp / self.action_space.dp_scale
        dp_norm = torch.clamp(dp_norm, -0.999, 0.999)
        dp_raw = torch.atanh(dp_norm)

        dp_std_pos = torch.exp(torch.clamp(dp_log_std, -20, 2))

        # Log prob: log N(dp_raw | dp_mean, std) - sum log(1 - tanh^2(dp_raw))
        var = dp_std_pos ** 2
        log_prob_dp = -0.5 * (
            ((dp_raw - dp_mean) ** 2) / (var + 1e-6)
            + torch.log(2 * np.pi * var + 1e-6)
        ).sum(dim=-1)
        log_prob_dp -= torch.log(1.0 - dp_norm ** 2 + 1e-6).sum(dim=-1)

        # Gaussian (delta_p) entropy
        entropy_dp = 0.5 * torch.log(2 * np.pi * np.e * var + 1e-6).sum(dim=-1)

        if not getattr(self.action_space, 'learn_roles', True):
            # Role is assigned by the env's P0 solver, not the policy -> drop the
            # role term from BOTH log-prob and entropy so the role head carries no
            # gradient. MUST match ActionSpace.compute_log_prob (which also drops
            # it) to keep old_log_prob == new_log_prob in the PPO ratio.
            return log_prob_dp, values, entropy_dp, dp_mean, pd_pred, comm_msgs

        # Categorical log prob for role
        log_probs_role = torch.log_softmax(role_logits, dim=-1)
        log_prob_role = log_probs_role.gather(1, actions_role.unsqueeze(-1)).squeeze(-1)

        new_log_probs = log_prob_dp + log_prob_role

        # Role (categorical) entropy
        probs = torch.softmax(role_logits, dim=-1)
        entropy_role = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

        entropies = entropy_dp + entropy_role

        return new_log_probs, values, entropies, dp_mean, pd_pred, comm_msgs

    def verify_old_log_prob_consistency(
        self,
        obs: torch.Tensor,
        actions_dp: torch.Tensor,
        actions_role: torch.Tensor,
        old_log_probs: torch.Tensor,
        h_prev: torch.Tensor = None,
        tolerance: float = 1e-4,
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
            dp_mean, dp_log_std, role_logits, _, _, _ = self.actor(obs, h_prev)

            N = obs.shape[0]
            new_log_probs = torch.zeros(N, device=self.device)

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

            if getattr(self.action_space, 'learn_roles', True):
                log_probs_role = torch.log_softmax(role_logits, dim=-1)
                log_prob_role = log_probs_role.gather(1, actions_role.unsqueeze(-1)).squeeze(-1)
                new_log_probs = log_prob_dp + log_prob_role
            else:
                new_log_probs = log_prob_dp

        diff = (old_log_probs - new_log_probs).abs()
        max_diff = diff.max().item()
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
