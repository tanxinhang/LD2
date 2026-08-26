"""Rollout buffer with Generalized Advantage Estimation (GAE).

Stores trajectories from multiple agents and computes GAE advantages
for PPO training. Supports recurrent policies via stored GRU hidden states.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


class RolloutBuffer:
    """Stores rollout data for MAPPO/PPO training.

    Stores per-agent: obs, actions (dp, role), log_probs, values, rewards, dones,
    and GRU hidden states for recurrent policy consistency.
    Computes GAE advantages and returns after a rollout completes.
    """

    def __init__(self, buffer_size: int, num_agents: int,
                 obs_dim: int, global_state_dim: int,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 num_targets: int = 4, gru_hidden_dim: int = 0,
                 comm_dim: int = 16):
        """
        Args:
            buffer_size: Maximum steps in buffer (e.g., 4096)
            num_agents: Number of UAV agents (K)
            obs_dim: Local observation dimension
            global_state_dim: Global state dimension (for critic)
            gamma: Discount factor
            gae_lambda: GAE λ parameter
            num_targets: Number of targets (Q) — from config, NOT hardcoded
            gru_hidden_dim: GRU hidden dimension (0 = no recurrent storage)
        """
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ptr = 0
        self.full = False
        self.num_targets = num_targets
        self.gru_hidden_dim = gru_hidden_dim
        self._has_gru = gru_hidden_dim > 0
        self._window_len = 0
        self.comm_dim = int(comm_dim)
        self._has_comm_actions = False
        self._has_isac_resource_actions = False
        self.credit_head_names = ('movement', 'message', 'rate', 'resource')
        self.num_credit_heads = len(self.credit_head_names)
        self._has_headwise_credit = False
        self._has_causal_teacher = False

        # Per-agent storage
        self.obs = np.zeros((buffer_size, num_agents, obs_dim), dtype=np.float64)
        self.global_states = np.zeros((buffer_size, global_state_dim), dtype=np.float64)
        self.actions_dp = np.zeros((buffer_size, num_agents, 2), dtype=np.float64)
        self.actions_role = np.zeros((buffer_size, num_agents), dtype=np.int32)
        self.actions_comm = np.zeros(
            (buffer_size, num_agents, self.comm_dim), dtype=np.float64)
        self.actions_comm_rate = np.zeros(
            (buffer_size, num_agents), dtype=np.int64)
        self.actions_isac_power_raw = np.zeros(
            (buffer_size, num_agents), dtype=np.float64)
        self.actions_sensing_raw = np.zeros(
            (buffer_size, num_agents, num_targets), dtype=np.float64)
        self.log_probs = np.zeros((buffer_size, num_agents), dtype=np.float64)
        self.head_log_probs = np.zeros(
            (buffer_size, num_agents, self.num_credit_heads), dtype=np.float64)
        # 1 when movement/role was newly sampled at this frame, 0 when the
        # previous movement commitment was held while communication continued.
        self.movement_action_masks = np.ones(
            (buffer_size, num_agents), dtype=np.float64)
        self.comm_round_phases = np.zeros(
            (buffer_size, num_agents), dtype=np.float64)
        self.values = np.zeros((buffer_size, num_agents), dtype=np.float64)
        self.rewards = np.zeros((buffer_size, num_agents), dtype=np.float64)
        self.credit_rewards = np.zeros(
            (buffer_size, num_agents, self.num_credit_heads), dtype=np.float64)
        self.credit_values = np.zeros(
            (buffer_size, num_agents, self.num_credit_heads), dtype=np.float64)
        # Sparse paired-intervention labels. A separate mask distinguishes a
        # measured zero effect from a sender that was not intervened on.
        self.causal_teacher_effects = np.zeros(
            (buffer_size, num_agents, num_targets), dtype=np.float64)
        self.causal_teacher_masks = np.zeros(
            (buffer_size, num_agents), dtype=np.float64)
        self.dones = np.zeros((buffer_size, num_agents), dtype=np.float64)
        self.masks = np.ones((buffer_size, num_agents), dtype=np.float64)
        self.oracle_masks = np.ones((buffer_size, num_agents), dtype=np.float64)

        # GRU hidden states: (T, K, K-1, D) for per-neighbor GRU states
        if self._has_gru:
            self.h_prev = np.zeros(
                (buffer_size, num_agents, num_agents - 1, gru_hidden_dim),
                dtype=np.float32,
            )

        # TICA observation window: (T, K, L, obs_dim) + mask
        self.obs_window = None
        self.window_mask = None
        self._window_len = 0

        # Per-target storage (S3b: diagnostics)
        self.per_target_rewards = np.zeros(
            (buffer_size, num_agents, num_targets), dtype=np.float64,
        )
        self.per_target_values = np.zeros(
            (buffer_size, num_agents, num_targets), dtype=np.float64,
        )

        # GAE results (computed after rollout)
        self.advantages = None
        self.returns = None
        self.per_target_advantages = None
        self.per_target_returns = None
        self.credit_advantages = None
        self.credit_returns = None

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
        h_prev: np.ndarray = None,  # (K, K-1, D) GRU hidden states
        obs_window: np.ndarray = None,  # (K, L, obs_dim) TICA window
        window_mask: np.ndarray = None,  # (K, L) TICA mask
        actions_comm: np.ndarray = None,       # (K, comm_dim)
        actions_comm_rate: np.ndarray = None,  # (K,), 0=silence
        actions_isac_power_raw: np.ndarray = None,  # (K,)
        actions_sensing_raw: np.ndarray = None,     # (K,Q)
        movement_action_mask: np.ndarray = None,    # (K,), 1=new movement
        comm_round_phase: np.ndarray = None,        # (K,), 0=proposal/1=response
        head_log_probs: np.ndarray = None,          # (K,4) move/msg/rate/resource
        credit_rewards: np.ndarray = None,          # (K,4) head-specific rewards
        credit_values: np.ndarray = None,           # (K,4) CTDE baselines
        causal_teacher_effects: np.ndarray = None,  # (K,Q) paired P_D effects
        causal_teacher_masks: np.ndarray = None,    # (K,) intervention present
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
            oracle_mask: (K,) 1=actor, 0=oracle
            per_target_rewards: (K, Q) per-target P_D
            per_target_values: (K, Q) per-target V_q(s)
            h_prev: (K, K-1, D) GRU hidden states used in this forward pass
        """
        if self.ptr >= self.buffer_size:
            self.full = True
            return

        for k in range(self.num_agents):
            if k in obs:
                self.obs[self.ptr, k] = obs[k]
            self.actions_dp[self.ptr, k] = actions_dp[k]
            self.actions_role[self.ptr, k] = int(actions_role[k])
            if actions_comm is not None:
                self.actions_comm[self.ptr, k] = actions_comm[k]
            if actions_comm_rate is not None:
                self.actions_comm_rate[self.ptr, k] = int(actions_comm_rate[k])
            if actions_isac_power_raw is not None:
                self.actions_isac_power_raw[self.ptr, k] = float(
                    actions_isac_power_raw[k])
            if actions_sensing_raw is not None:
                self.actions_sensing_raw[self.ptr, k] = actions_sensing_raw[k]
            self.log_probs[self.ptr, k] = log_probs[k] if k < len(log_probs) else 0.0
            if head_log_probs is not None:
                self.head_log_probs[self.ptr, k] = head_log_probs[k]
            self.movement_action_masks[self.ptr, k] = (
                float(movement_action_mask[k])
                if movement_action_mask is not None else 1.0)
            self.comm_round_phases[self.ptr, k] = (
                float(comm_round_phase[k])
                if comm_round_phase is not None else 0.0)
            self.values[self.ptr, k] = values[k] if k < len(values) else 0.0
            self.rewards[self.ptr, k] = rewards.get(k, 0.0)
            if credit_rewards is not None:
                self.credit_rewards[self.ptr, k] = credit_rewards[k]
            if credit_values is not None:
                self.credit_values[self.ptr, k] = credit_values[k]
            if causal_teacher_effects is not None:
                self.causal_teacher_effects[self.ptr, k] = (
                    causal_teacher_effects[k])
            if causal_teacher_masks is not None:
                self.causal_teacher_masks[self.ptr, k] = float(
                    causal_teacher_masks[k])
            self.dones[self.ptr, k] = float(dones.get(k, False))
            self.masks[self.ptr, k] = 1.0 - float(dones.get(k, False))
            self.oracle_masks[self.ptr, k] = float(oracle_mask[k]) if oracle_mask is not None else 1.0
            if per_target_rewards is not None:
                Q_actual = per_target_rewards.shape[1]
                self.per_target_rewards[self.ptr, k, :Q_actual] = per_target_rewards[k]
            if per_target_values is not None:
                Q_actual = per_target_values.shape[1]
                self.per_target_values[self.ptr, k, :Q_actual] = per_target_values[k]

        # Remember whether this rollout contains the communication action head.
        if actions_comm is not None and actions_comm_rate is not None:
            self._has_comm_actions = True
        if (actions_isac_power_raw is not None
                and actions_sensing_raw is not None):
            self._has_isac_resource_actions = True
        if (head_log_probs is not None and credit_rewards is not None
                and credit_values is not None):
            self._has_headwise_credit = True
        if causal_teacher_masks is not None:
            self._has_causal_teacher = True

        # Store GRU hidden states
        if self._has_gru and h_prev is not None:
            self.h_prev[self.ptr] = h_prev  # (K, K-1, D)

        # Store TICA observation window + mask
        if obs_window is not None:
            if self.obs_window is None:
                L = obs_window.shape[-2] if obs_window.ndim >= 3 else 1
                self.obs_window = np.zeros(
                    (self.buffer_size, self.num_agents, L, self.obs.shape[-1]),
                    dtype=np.float64)
                self.window_mask = np.zeros(
                    (self.buffer_size, self.num_agents, L), dtype=bool)
                self._window_len = L
            self.obs_window[self.ptr] = obs_window
            if window_mask is not None:
                self.window_mask[self.ptr] = window_mask

        self.global_states[self.ptr] = global_state
        self.ptr += 1

    def compute_gae(self, next_values: np.ndarray,
                    next_per_target_values: np.ndarray = None,
                    next_credit_values: np.ndarray = None) -> None:
        """Compute GAE advantages and returns.

        δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
        A_t = δ_t + γλ * (1 - done) * A_{t+1}
        R_t = A_t + V(s_t)

        Supports both single-env (next_values shape: (K,)) and
        multi-env interleaved (next_values shape: (num_envs*K,)) layouts.

        Also computes per-target GAE when per_target_values are stored.

        Args:
            next_values: (K,) for single env, or (N*K,) for N parallel envs
            next_per_target_values: (N*K, Q) per-target bootstrap values.
                Only terminal episodes (mask=0) get zero bootstrap;
                buffer-truncated rollouts use the critic's V_q(s_{t+1}).
        """
        actual_size = self.ptr
        self.advantages = np.zeros((actual_size, self.num_agents), dtype=np.float64)
        self.returns = np.zeros((actual_size, self.num_agents), dtype=np.float64)

        # Detect multi-env layout
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
            T = actual_size // num_envs
            for k in range(self.num_agents):
                for n in range(num_envs):
                    gae = 0.0
                    for s in reversed(range(T)):
                        t = n + s * num_envs
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

        # Separate action-head GAE. Each stream has its own CTDE baseline, so
        # communication/rate rewards no longer leak into the movement ratio.
        self.credit_advantages = None
        self.credit_returns = None
        if self._has_headwise_credit and next_credit_values is not None:
            self.credit_advantages = np.zeros(
                (actual_size, self.num_agents, self.num_credit_heads),
                dtype=np.float64)
            self.credit_returns = np.zeros_like(self.credit_advantages)
            if num_envs == 1:
                for k in range(self.num_agents):
                    for h in range(self.num_credit_heads):
                        gae = 0.0
                        for t in reversed(range(actual_size)):
                            next_v = (
                                next_credit_values[k, h]
                                if t == actual_size - 1
                                else self.credit_values[t + 1, k, h])
                            delta = (
                                self.credit_rewards[t, k, h]
                                + self.gamma * next_v * self.masks[t, k]
                                - self.credit_values[t, k, h])
                            gae = (delta + self.gamma * self.gae_lambda
                                   * self.masks[t, k] * gae)
                            self.credit_advantages[t, k, h] = gae
                            self.credit_returns[t, k, h] = (
                                gae + self.credit_values[t, k, h])
            else:
                T = actual_size // num_envs
                for k in range(self.num_agents):
                    for h in range(self.num_credit_heads):
                        for n in range(num_envs):
                            gae = 0.0
                            for s in reversed(range(T)):
                                t = n + s * num_envs
                                next_v = (
                                    next_credit_values[
                                        n * self.num_agents + k, h]
                                    if s == T - 1
                                    else self.credit_values[
                                        n + (s + 1) * num_envs, k, h])
                                delta = (
                                    self.credit_rewards[t, k, h]
                                    + self.gamma * next_v * self.masks[t, k]
                                    - self.credit_values[t, k, h])
                                gae = (delta + self.gamma * self.gae_lambda
                                       * self.masks[t, k] * gae)
                                self.credit_advantages[t, k, h] = gae
                                self.credit_returns[t, k, h] = (
                                    gae + self.credit_values[t, k, h])

        # ── Per-target GAE (S3c: target-wise critic) ──
        # P1 FIX: use next_per_target_values for proper bootstrap.
        # Previously hardcoded V_q=0 for the last step even when the
        # rollout was truncated (not terminal) — that biased A_{t,q} low.
        self.per_target_advantages = np.zeros_like(self.per_target_rewards)
        self.per_target_returns = np.zeros_like(self.per_target_rewards)

        _have_pt_next = next_per_target_values is not None

        if num_envs == 1:
            for k in range(self.num_agents):
                for q in range(self.num_targets):
                    gae = 0.0
                    for t in reversed(range(actual_size)):
                        if t == actual_size - 1:
                            # Use critic bootstrap, masked by done flag
                            if _have_pt_next and k < len(next_per_target_values):
                                next_v_pt = next_per_target_values[k, q]
                            else:
                                next_v_pt = 0.0
                        else:
                            next_v_pt = self.per_target_values[t + 1, k, q]

                        delta_pt = (
                            self.per_target_rewards[t, k, q]
                            + self.gamma * next_v_pt * self.masks[t, k]
                            - self.per_target_values[t, k, q]
                        )
                        gae = delta_pt + self.gamma * self.gae_lambda * self.masks[t, k] * gae
                        self.per_target_advantages[t, k, q] = gae
                        self.per_target_returns[t, k, q] = gae + self.per_target_values[t, k, q]
        else:
            T = actual_size // num_envs
            for k in range(self.num_agents):
                for q in range(self.num_targets):
                    for n in range(num_envs):
                        gae = 0.0
                        for s in reversed(range(T)):
                            t = n + s * num_envs
                            if s == T - 1:
                                if _have_pt_next:
                                    next_v_pt = next_per_target_values[n * self.num_agents + k, q]
                                else:
                                    next_v_pt = 0.0
                            else:
                                next_t = n + (s + 1) * num_envs
                                next_v_pt = self.per_target_values[next_t, k, q]

                            delta_pt = (
                                self.per_target_rewards[t, k, q]
                                + self.gamma * next_v_pt * self.masks[t, k]
                                - self.per_target_values[t, k, q]
                            )
                            gae = delta_pt + self.gamma * self.gae_lambda * self.masks[t, k] * gae
                            self.per_target_advantages[t, k, q] = gae
                            self.per_target_returns[t, k, q] = gae + self.per_target_values[t, k, q]

    def get_training_data(self) -> Dict[str, torch.Tensor]:
        """Get all buffer data as torch tensors for PPO update.

        Returns:
            Dict of tensors: obs, global_states, actions_dp, actions_role,
            old_log_probs, advantages, returns, old_values, and optionally
            h_prev, per_target_advantages, per_target_returns.
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
        movement_mask_flat = self.movement_action_masks[:actual_size].reshape(-1)
        round_phase_flat = self.comm_round_phases[:actual_size].reshape(-1)
        adv_flat = advantages.reshape(-1)
        ret_flat = returns.reshape(-1)
        val_flat = self.values[:actual_size].reshape(-1)

        # Zero out advantages for oracle-guided transitions
        oracle_mask_flat = self.oracle_masks[:actual_size].reshape(-1)
        adv_flat = adv_flat * oracle_mask_flat
        ret_flat = ret_flat * oracle_mask_flat

        # Normalize advantages (in-place on local, doesn't modify stored advantages)
        active_adv = adv_flat != 0
        adv_flat_norm = np.zeros_like(adv_flat)
        if np.any(active_adv):
            adv_values = adv_flat[active_adv]
            adv_mean = adv_values.mean()
            adv_std = adv_values.std() + 1e-8
            adv_flat_norm[active_adv] = (
                adv_values - adv_mean) / adv_std

        result = {
            'obs': torch.as_tensor(obs_flat, dtype=torch.float32),
            'global_states': torch.as_tensor(gs_flat, dtype=torch.float32),
            'actions_dp': torch.as_tensor(dp_flat, dtype=torch.float32),
            'actions_role': torch.as_tensor(role_flat, dtype=torch.long),
            'old_log_probs': torch.as_tensor(lp_flat, dtype=torch.float32),
            'movement_action_masks': torch.as_tensor(
                movement_mask_flat, dtype=torch.float32),
            'comm_round_phases': torch.as_tensor(
                round_phase_flat, dtype=torch.float32),
            'advantages': torch.as_tensor(adv_flat_norm, dtype=torch.float32),
            'returns': torch.as_tensor(ret_flat, dtype=torch.float32),
            'old_values': torch.as_tensor(val_flat, dtype=torch.float32),
            # Retain episode-boundary masks for delayed-message and temporal
            # commitment auxiliaries. Shape remains (T*K,) after flattening.
            'masks': torch.as_tensor(
                self.masks[:actual_size].reshape(-1), dtype=torch.float32),
        }

        if self._has_comm_actions:
            comm_flat = self.actions_comm[:actual_size].reshape(-1, self.comm_dim)
            comm_rate_flat = self.actions_comm_rate[:actual_size].reshape(-1)
            result['actions_comm'] = torch.as_tensor(comm_flat, dtype=torch.float32)
            result['actions_comm_rate'] = torch.as_tensor(comm_rate_flat, dtype=torch.long)
        if self._has_isac_resource_actions:
            power_flat = self.actions_isac_power_raw[:actual_size].reshape(-1)
            sensing_flat = self.actions_sensing_raw[:actual_size].reshape(
                -1, self.num_targets)
            result['actions_isac_power_raw'] = torch.as_tensor(
                power_flat, dtype=torch.float32)
            result['actions_sensing_raw'] = torch.as_tensor(
                sensing_flat, dtype=torch.float32)

        if (self._has_headwise_credit and self.credit_advantages is not None
                and self.credit_returns is not None):
            head_lp_flat = self.head_log_probs[:actual_size].reshape(
                -1, self.num_credit_heads)
            credit_adv = self.credit_advantages[:actual_size].reshape(
                -1, self.num_credit_heads).copy()
            credit_ret = self.credit_returns[:actual_size].reshape(
                -1, self.num_credit_heads)
            credit_val = self.credit_values[:actual_size].reshape(
                -1, self.num_credit_heads)
            # Oracle replacement invalidates only the sampled movement action;
            # communication and resource decisions remain genuine samples.
            credit_adv[:, 0] *= oracle_mask_flat
            credit_adv_norm = np.zeros_like(credit_adv)
            for h in range(self.num_credit_heads):
                active_h = credit_adv[:, h] != 0
                if np.any(active_h):
                    mean_h = credit_adv[active_h, h].mean()
                    std_h = credit_adv[active_h, h].std() + 1e-8
                    credit_adv_norm[active_h, h] = (
                        credit_adv[active_h, h] - mean_h) / std_h
            result['old_head_log_probs'] = torch.as_tensor(
                head_lp_flat, dtype=torch.float32)
            result['credit_advantages'] = torch.as_tensor(
                credit_adv_norm, dtype=torch.float32)
            result['credit_returns'] = torch.as_tensor(
                credit_ret, dtype=torch.float32)
            result['old_credit_values'] = torch.as_tensor(
                credit_val, dtype=torch.float32)

        if self._has_causal_teacher:
            result['causal_teacher_effects'] = torch.as_tensor(
                self.causal_teacher_effects[:actual_size].reshape(
                    -1, self.num_targets), dtype=torch.float32)
            result['causal_teacher_masks'] = torch.as_tensor(
                self.causal_teacher_masks[:actual_size].reshape(-1),
                dtype=torch.float32)
        # Target-level sensing supervision is useful independently of CCP.
        # Every transition stores the achieved P_D vector, so expose it to the
        # communication-aided sensing auxiliary on all rollouts.
        result['per_target_rewards_raw'] = torch.as_tensor(
            self.per_target_rewards[:actual_size].reshape(
                -1, self.num_targets), dtype=torch.float32)

        # GRU hidden states: (T, K, K-1, D) → (T*K, K-1, D)
        if self._has_gru:
            h_flat = self.h_prev[:actual_size].reshape(-1, self.num_agents - 1,
                                                       self.gru_hidden_dim)
            result['h_prev'] = torch.as_tensor(h_flat, dtype=torch.float32)

        # TICA observation window: (T, K, L, obs_dim) → (T*K, L, obs_dim)
        if self.obs_window is not None:
            w_flat = self.obs_window[:actual_size].reshape(-1, self._window_len,
                                                           self.obs.shape[-1])
            result['obs_window'] = torch.as_tensor(w_flat, dtype=torch.float32)
        # TICA window mask: (T, K, L) → (T*K, L)
        if self.window_mask is not None:
            wm_flat = self.window_mask[:actual_size].reshape(-1, self._window_len)
            result['window_mask'] = torch.as_tensor(wm_flat, dtype=torch.bool)

        # Per-target advantages and returns
        if self.per_target_advantages is not None:
            pta_flat = self.per_target_advantages[:actual_size].reshape(-1, self.num_targets)
            ptr_flat = self.per_target_returns[:actual_size].reshape(-1, self.num_targets)
            result['per_target_advantages'] = torch.as_tensor(pta_flat, dtype=torch.float32)
            result['per_target_returns'] = torch.as_tensor(ptr_flat, dtype=torch.float32)

        return result

    def clear(self) -> None:
        """Reset buffer for next rollout."""
        self.ptr = 0
        self.full = False
        self.advantages = None
        self.returns = None
        self.per_target_advantages = None
        self.per_target_returns = None
        self.credit_advantages = None
        self.credit_returns = None
        self._has_comm_actions = False
        self._has_isac_resource_actions = False
        self._has_headwise_credit = False
        self._has_causal_teacher = False

    def is_ready(self) -> bool:
        """Check if buffer has enough data for an update."""
        return self.full or self.ptr >= self.buffer_size

    def __len__(self) -> int:
        return self.ptr
