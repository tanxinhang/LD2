"""Rollout buffer with Generalized Advantage Estimation (GAE).

Stores trajectories from multiple agents and computes GAE advantages
for PPO training.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


class RolloutBuffer:
    """Stores rollout data for MAPPO/PPO training.

    Stores per-agent: obs, actions (dp, role), log_probs, values, rewards, dones.
    Computes GAE advantages and returns after a rollout completes.
    """

    def __init__(self, buffer_size: int, num_agents: int,
                 obs_dim: int, global_state_dim: int,
                 gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Args:
            buffer_size: Maximum steps in buffer (e.g., 4096)
            num_agents: Number of UAV agents (K)
            obs_dim: Local observation dimension
            global_state_dim: Global state dimension (for critic)
            gamma: Discount factor
            gae_lambda: GAE λ parameter
        """
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ptr = 0
        self.full = False

        # Per-agent storage
        self.obs = np.zeros((buffer_size, num_agents, obs_dim), dtype=np.float64)
        self.global_states = np.zeros((buffer_size, global_state_dim), dtype=np.float64)
        self.actions_dp = np.zeros((buffer_size, num_agents, 2), dtype=np.float64)
        self.actions_role = np.zeros((buffer_size, num_agents), dtype=np.int32)
        self.log_probs = np.zeros((buffer_size, num_agents), dtype=np.float64)
        self.values = np.zeros((buffer_size, num_agents), dtype=np.float64)
        self.rewards = np.zeros((buffer_size, num_agents), dtype=np.float64)
        self.dones = np.zeros((buffer_size, num_agents), dtype=np.float64)
        self.masks = np.ones((buffer_size, num_agents), dtype=np.float64)
        self.oracle_masks = np.ones((buffer_size, num_agents), dtype=np.float64)
        # Per-target storage (S3b: diagnostics)
        self.num_targets = 8
        self.per_target_rewards = np.zeros((buffer_size, num_agents, 8), dtype=np.float64)
        self.per_target_values = np.zeros((buffer_size, num_agents, 8), dtype=np.float64)

        # GAE results (computed after rollout)
        self.advantages = None
        self.returns = None
        self.per_target_advantages = None

    def store(
        self,
        obs: Dict[int, np.ndarray],
        global_state: np.ndarray,
        actions_dp: np.ndarray,      # (K, 2)
        actions_role: np.ndarray,    # (K,)
        log_probs: np.ndarray,       # (K,)
        values: np.ndarray,          # (K,)
        rewards: Dict[int, float],
        dones: Dict[int, bool],
        oracle_mask: np.ndarray = None,  # (K,) 1=actor, 0=oracle
        per_target_rewards: np.ndarray = None,  # (K, Q) per-target P_D
        per_target_values: np.ndarray = None,   # (K, Q) per-target V_q(s)
    ) -> None:
        """Store one transition for all agents.

        Args:
            obs: Dict mapping agent_id → observation vector
            global_state: Global state vector
            actions_dp: (K, 2) delta_p actions
            actions_role: (K,) role actions
            log_probs: (K,) action log probabilities
            values: (K,) critic values
            rewards: Dict mapping agent_id → reward
            dones: Dict mapping agent_id → done flag
        """
        if self.ptr >= self.buffer_size:
            self.full = True
            return

        for k in range(self.num_agents):
            if k in obs:
                self.obs[self.ptr, k] = obs[k]
            self.actions_dp[self.ptr, k] = actions_dp[k]
            self.actions_role[self.ptr, k] = int(actions_role[k])
            self.log_probs[self.ptr, k] = log_probs[k] if k < len(log_probs) else 0.0
            self.values[self.ptr, k] = values[k] if k < len(values) else 0.0
            self.rewards[self.ptr, k] = rewards.get(k, 0.0)
            self.dones[self.ptr, k] = float(dones.get(k, False))
            self.masks[self.ptr, k] = 1.0 - float(dones.get(k, False))
            self.oracle_masks[self.ptr, k] = float(oracle_mask[k]) if oracle_mask is not None else 1.0
            if per_target_rewards is not None:
                self.per_target_rewards[self.ptr, k] = per_target_rewards[k]
            if per_target_values is not None:
                self.per_target_values[self.ptr, k] = per_target_values[k]

        self.global_states[self.ptr] = global_state
        self.ptr += 1

    def compute_gae(self, next_values: np.ndarray) -> None:
        """Compute GAE advantages and returns.

        δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
        A_t = δ_t + γλ * (1 - done) * A_{t+1}
        R_t = A_t + V(s_t)

        Supports both single-env (next_values shape: (K,)) and
        multi-env interleaved (next_values shape: (num_envs*K,)) layouts.
        In multi-env mode, each env's trajectory is at indices n, n+N, n+2N, ...

        Args:
            next_values: (K,) for single env, or (N*K,) for N parallel envs
        """
        actual_size = self.ptr
        self.advantages = np.zeros((actual_size, self.num_agents), dtype=np.float64)
        self.returns = np.zeros((actual_size, self.num_agents), dtype=np.float64)

        # Detect multi-env layout: if next_values has more entries than agents
        num_envs = len(next_values) // self.num_agents

        if num_envs == 1:
            # ── Original single-env GAE ──
            for k in range(self.num_agents):
                gae = 0.0
                for t in reversed(range(actual_size)):
                    if t == actual_size - 1:
                        next_v = next_values[k] if k < len(next_values) else 0.0
                    else:
                        next_v = self.values[t + 1, k]

                    delta = (
                        self.rewards[t, k]
                        + self.gamma * next_v * self.masks[t, k]
                        - self.values[t, k]
                    )
                    gae = delta + self.gamma * self.gae_lambda * self.masks[t, k] * gae
                    self.advantages[t, k] = gae
                    self.returns[t, k] = gae + self.values[t, k]
        else:
            # ── Multi-env interleaved GAE ──
            # Each env n's trajectory: indices n, n+N, n+2N, ..., n+(T-1)*N
            # Next-values are ordered: [env0_k0, env0_k1, ..., env0_kK-1,
            #                          env1_k0, env1_k1, ..., env1_kK-1, ...]
            T = actual_size // num_envs
            for k in range(self.num_agents):
                for n in range(num_envs):
                    gae = 0.0
                    for s in reversed(range(T)):
                        t = n + s * num_envs  # buffer index for env n, step s
                        if s == T - 1:
                            next_v = next_values[n * self.num_agents + k]
                        else:
                            next_t = n + (s + 1) * num_envs
                            next_v = self.values[next_t, k]

                        delta = (
                            self.rewards[t, k]
                            + self.gamma * next_v * self.masks[t, k]
                            - self.values[t, k]
                        )
                        gae = delta + self.gamma * self.gae_lambda * self.masks[t, k] * gae
                        self.advantages[t, k] = gae
                        self.returns[t, k] = gae + self.values[t, k]

    def get_training_data(self) -> Dict[str, torch.Tensor]:
        """Get all buffer data as torch tensors for PPO update.

        Returns:
            Dict of tensors: obs, global_states, actions_dp, actions_role,
            old_log_probs, advantages, returns, old_values
        """
        actual_size = self.ptr
        if self.advantages is None or self.returns is None:
            raise RuntimeError("Must call compute_gae() before get_training_data()")
        advantages: np.ndarray = self.advantages
        returns: np.ndarray = self.returns

        # Flatten agent dimension into batch
        # (T, K, dim) → (T*K, dim)
        obs_flat = self.obs[:actual_size].reshape(-1, self.obs.shape[-1])
        gs_flat = self.global_states[:actual_size]
        dp_flat = self.actions_dp[:actual_size].reshape(-1, 2)
        role_flat = self.actions_role[:actual_size].reshape(-1)
        lp_flat = self.log_probs[:actual_size].reshape(-1)
        adv_flat = advantages.reshape(-1)
        ret_flat = returns.reshape(-1)
        val_flat = self.values[:actual_size].reshape(-1)

        # Zero out advantages for oracle-guided transitions
        oracle_mask_flat = self.oracle_masks[:actual_size].reshape(-1)
        adv_flat = adv_flat * oracle_mask_flat
        ret_flat = ret_flat * oracle_mask_flat  # also mask returns for consistency

        # Normalize advantages (in-place on local, doesn't modify stored advantages)
        adv_mean = adv_flat[adv_flat != 0].mean() if (adv_flat != 0).any() else 0.0
        adv_std = adv_flat[adv_flat != 0].std() + 1e-8
        adv_flat_norm = np.where(adv_flat != 0, (adv_flat - adv_mean) / adv_std, 0.0)

        return {
            'obs': torch.as_tensor(obs_flat, dtype=torch.float32),
            'global_states': torch.as_tensor(gs_flat, dtype=torch.float32),
            'actions_dp': torch.as_tensor(dp_flat, dtype=torch.float32),
            'actions_role': torch.as_tensor(role_flat, dtype=torch.long),
            'old_log_probs': torch.as_tensor(lp_flat, dtype=torch.float32),
            'advantages': torch.as_tensor(adv_flat_norm, dtype=torch.float32),
            'returns': torch.as_tensor(ret_flat, dtype=torch.float32),
            'old_values': torch.as_tensor(val_flat, dtype=torch.float32),
        }

    def clear(self) -> None:
        """Reset buffer for next rollout."""
        self.ptr = 0
        self.full = False
        self.advantages = None
        self.returns = None

    def is_ready(self) -> bool:
        """Check if buffer has enough data for an update."""
        return self.full or self.ptr >= self.buffer_size

    def __len__(self) -> int:
        return self.ptr
