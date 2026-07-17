"""MAPPO Trainer: PPO-clip + GAE + Lagrangian penalty for constraints.

Orchestrates:
1. Rollout collection (parallel envs or sequential)
2. GAE advantage computation
3. PPO-clip update (multiple epochs, minibatches)
4. Lagrangian multiplier adaptation
5. Entropy coefficient decay
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import time

def compute_cvar_k(Q: int, tail_fraction: float = 0.25) -> int:
    """Number of worst targets for CVaR tail-risk constraint.

    Args:
        Q: number of targets
        tail_fraction: fraction of targets in the tail (default 0.25)

    Returns:
        k: number of worst targets to average over (≥1)
    """
    return max(1, int(np.ceil(tail_fraction * Q)))


from uav_isac.agents.mappo_agent import MAPPOAgent
from uav_isac.agents.buffer import RolloutBuffer
from uav_isac.agents.networks import (
    split_param_groups, ATTENTION_PARAM_PREFIXES,
    ENCODER_PARAM_PREFIXES, HEAD_PARAM_PREFIXES,
)
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.utils.seeding import set_seed
from uav_isac.utils.types import Action


class MAPPTrainer:
    """MAPPO training orchestrator."""

    def __init__(
        self,
        env: UAVISACEnv,
        agents: List[MAPPOAgent],
        config,  # MasterConfig
        device: str = "cpu",
    ):
        """
        Args:
            env: UAVISAC environment
            agents: List of MAPPOAgent (one per UAV)
            config: MasterConfig
            device: "cpu" or "cuda"
        """
        self.agents = agents
        self.cfg = config
        self.device = torch.device(device)
        self.K = len(agents)
        self.Q = config.scenario.Q

        ma = config.marl
        self.gamma = ma.gamma
        self.gae_lambda = ma.gae_lambda
        self.ppo_clip = ma.ppo_clip
        self.ppo_epochs = ma.ppo_epochs
        self.minibatch_size = ma.minibatch_size
        self.entropy_init = ma.entropy_init
        self.entropy_final = ma.entropy_final
        self.entropy_decay_frames = ma.entropy_decay_frames
        self.vf_coef = ma.vf_coef
        self.max_grad_norm = ma.max_grad_norm
        self.num_episodes = ma.num_episodes
        self.rollout_steps = ma.rollout_steps
        self.lagrangian_lr = ma.lagrangian_lr
        self.num_envs = ma.num_envs
        self._lambda_report = ma.lambda_report
        self.target_kl = getattr(ma, 'target_kl', 0.03)  # KL early-stop threshold
        # CTDE (centralized critic, MAPPO) vs decentralized critic (IPPO).
        # Single source of truth = how the agent's critic was actually built.
        self.centralized_critic = getattr(agents[0], 'centralized_critic', True)

        # Convergence-based early stopping (deterministic eval on a plateau)
        self.early_stop = getattr(ma, 'early_stop', True)
        self.eval_interval = getattr(ma, 'eval_interval', 50)
        self.eval_episodes = getattr(ma, 'eval_episodes', 3)
        self.early_stop_patience = getattr(ma, 'early_stop_patience', 12)
        self.early_stop_min_delta = getattr(ma, 'early_stop_min_delta', 0.005)
        # Fixed evaluation scenarios: the SAME seeds are replayed every eval and
        # across the four decode modes, so any score difference is attributable to
        # the policy/decode mode, not scenario luck. Configurable via marl.eval_seeds.
        self.eval_seeds = list(getattr(ma, 'eval_seeds',
                                       [10001, 10002, 10003, 10004, 10005]))
        self.best_score = -float('inf')
        self.best_params = None
        self._patience = 0
        self.converged_episode = None

        # Lagrangian multiplier (for constraint violations)
        self.lagrangian_lambda = 0.0
        self.max_violation_rate = ma.max_violation_rate  # target: fraction of steps with violations
        self.lagrangian_max = ma.lagrangian_max          # upper bound for stability

        # Entropy coefficient (linear decay)
        self.entropy_coef = self.entropy_init

        # Create parallel environments
        self.env = env  # primary (for eval / step_info access)
        self.envs: List[UAVISACEnv] = [env]
        for n in range(1, self.num_envs):
            env_n = UAVISACEnv(config=config, seed=env.seed_val + n * 1000)
            self.envs.append(env_n)

        # ── Fix #6: persistent env state across rollouts ──
        self._current_obs = [None] * self.num_envs

        # Steps per env to maintain total transitions ~= rollout_steps
        self.steps_per_env = max(1, self.rollout_steps // self.num_envs)
        self.macro_interval = getattr(ma, 'actor_decision_interval', 1)
        self.gamma_micro = self.gamma
        if self.macro_interval > 1:
            self.steps_per_env = max(1, self.steps_per_env // self.macro_interval)

        # Shared buffer (size = steps_per_env * num_envs)
        obs_test, _ = env.reset(seed=0)
        obs_dim = obs_test['0'].shape[0]
        global_dim = env.core.obs_builder.get_global_state_dim() + 16

        # TICA window ring buffer: per-env, per-agent, L frames
        self._window_len = getattr(ma, 'obs_history_frames', 1)
        self._use_window = (self._window_len > 1)
        if self._use_window:
            self._obs_ring = np.zeros(
                (self.num_envs, self.K, self._window_len, obs_dim),
                dtype=np.float64)
            self._window_mask = np.zeros(
                (self.num_envs, self.K, self._window_len), dtype=bool)
            print(f'[WINDOW] L={self._window_len}, ring buffer allocated')

        # GRU hidden dim for recurrent buffer storage.
        # StructuredActorNetwork uses entity_dim as GRU hidden size;
        # flat-MLP actor has no GRU → gru_hidden_dim=0.
        _gru_dim = 0
        if getattr(config.marl, 'structured_actor', False):
            # Match the entity_dim used in StructuredActorNetwork.__init__
            _gru_dim = getattr(config.marl, 'hidden_layers', [256, 256])[-1]
            # Actually, the GRU hidden dim is the entity_dim, not the hidden layer dim.
            # The default entity_dim is 128 for StructuredActorNetwork, but let's
            # use the action_space attribute if available, else default 64.
            if hasattr(agents[0], 'actor') and hasattr(agents[0].actor, 'neighbor_gru'):
                _gru_dim = agents[0].actor.neighbor_gru.hidden_size

        # P1 FIX: auto-detect single_frame_dim from ObservationBuilder if not
        # explicitly set. Replaces hardcoded single_dim=227 in networks.py.
        if hasattr(agents[0].actor, 'single_frame_dim') and agents[0].actor.single_frame_dim == 0:
            agents[0].actor.single_frame_dim = env.core.obs_builder.get_single_frame_dim()

        # ── STARTUP DIAGNOSTIC: confirm config/code actually loaded ──
        # (entropy=0 in logs would be impossible if sigma-floor were really 0.37,
        #  so print the EFFECTIVE values to catch stale-cache / non-loaded changes.)
        with torch.no_grad():
            _, _ls, _, _, _, _ = agents[0].actor(torch.zeros(1, obs_dim, device=self.device))
            _sigma = torch.exp(_ls).cpu().numpy().ravel()
        print(f"[CONFIG CHECK] entropy_coef init/final={self.entropy_init}/{self.entropy_final} "
              f"ppo_epochs={self.ppo_epochs} target_kl={self.target_kl} "
              f"actor init sigma={_sigma}  (sigma_floor should match LOG_STD_MIN; "
              f"if entropy later prints ~0 while floor>=exp(-1)=0.37, the change did NOT load)")
        buffer_total = self.steps_per_env * self.num_envs
        self.buffer = RolloutBuffer(
            buffer_size=buffer_total,
            num_agents=self.K,
            obs_dim=obs_dim,
            global_state_dim=global_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            num_targets=self.Q,
            gru_hidden_dim=_gru_dim,
        )

        # Pre-allocate pinned tensors for GPU transfers (reused each step)
        self._obs_gpu = torch.empty(
            self.num_envs * self.K, obs_dim,
            dtype=torch.float32, device=self.device
        )
        self._gs_gpu = torch.empty(
            self.num_envs, env.core.obs_builder.get_global_state_dim(),  # raw gs (65), comm added separately
            dtype=torch.float32, device=self.device
        )

        # BC anchor (optional warm-start regularizer)
        self._bc_actor = None        # frozen copy of the BC policy
        self._bc_beta_init = getattr(ma, 'bc_beta_init', 0.05)
        self._bc_beta = self._bc_beta_init

        # ═══════════════════════════════════════════════════════════════
        # S1: Selective plasticity + communication mode
        # ═══════════════════════════════════════════════════════════════
        self._comm_mode = getattr(ma, 'learned_comm_mode', 'on')
        self._comm_off = (self._comm_mode == 'off')
        self._adv_mode = getattr(ma, 'advantage_mode', 'scalar')
        self._resp_tau_m = getattr(ma, 'target_responsibility_tau_m', 50.0)
        freeze_attn = getattr(ma, 'freeze_attention', False)
        use_per_lr = getattr(ma, 'use_per_module_lr', False)

        enc_params, head_params, attn_params = split_param_groups(
            agents[0].actor.named_parameters())

        if self._comm_off:
            # Freeze all communication-related heads
            comm_head_names = ['comm_head.', 'comm_proj.', 'gate.', 'intent_head.']
            for n, p in agents[0].actor.named_parameters():
                if any(n.startswith(prefix) for prefix in comm_head_names):
                    p.requires_grad_(False)

        if freeze_attn:
            for p in attn_params:
                p.requires_grad_(False)

        # Build optimizer with per-module LR when requested.
        # Full:  encoder=1e-5, attention=1e-5, head=5e-5
        # EH:    encoder=1e-5, attention=0,    head=5e-5
        # This isolates Attention trainability as the ONLY variable.
        if use_per_lr or freeze_attn:
            attn_lr = 0.0 if freeze_attn else 1e-5
            param_groups = [
                {'params': [p for p in enc_params if p.requires_grad], 'lr': 1e-5},
                {'params': [p for p in attn_params if p.requires_grad], 'lr': attn_lr},
                {'params': [p for p in head_params if p.requires_grad], 'lr': 5e-5},
            ]
            # Filter empty groups
            param_groups = [g for g in param_groups if len(g['params']) > 0]
            agents[0].actor_optimizer = torch.optim.Adam(param_groups)
            frozen_count = sum(1 for p in agents[0].actor.parameters() if not p.requires_grad)
            print(f'[S1] comm={self._comm_mode} freeze_attn={freeze_attn} '
                  f'per_module_lr={use_per_lr or freeze_attn} '
                  f'enc_lr=1e-5 attn_lr={attn_lr} head_lr=5e-5 '
                  f'frozen={frozen_count} params')

        # CVaR target-tail-risk constraint
        self._cvar_tau = getattr(ma, 'cvar_tau', 0.0)
        self._cvar_lambda = 0.5
        self._cvar_epsilon = getattr(ma, 'cvar_epsilon', 0.05)

        # DAgger reference KL anchor (conservative fine-tuning)
        self._ref_beta = 2.0

        # Oracle-guided exploration
        self._oracle_alpha = 0.0
        self._oracle_decay_episodes = 200
        self._oracle_ep_count = 0

        # Training metrics
        self.metrics_history: List[Dict] = []
        self.total_frames = 0

        # Shared networks across agents (parameter sharing)
        if len(agents) > 1:
            for k in range(1, len(agents)):
                agents[k].actor = agents[0].actor
                agents[k].critic = agents[0].critic
                agents[k].actor_optimizer = agents[0].actor_optimizer
                agents[k].critic_optimizer = agents[0].critic_optimizer

    def _effective_comm(self, comm_msgs: torch.Tensor) -> torch.Tensor:
        """Return zeroed comm if comm_off, else original. Single entry point."""
        if self._comm_off:
            return torch.zeros_like(comm_msgs)
        return comm_msgs

    def _compute_target_wise_advantage(
        self, obs: torch.Tensor,
        per_target_advantages: torch.Tensor,
        tau_d: float = 50.0,
    ) -> torch.Tensor:
        """Aggregate per-target advantages via detached distance responsibility.

        Pre-computed ONCE before PPO epochs on the full batch; minibatch
        then indexes into the result. This keeps advantages invariant to
        minibatch split and shuffle order.

        Args:
            obs: (B, obs_dim) observation batch (full rollout, flattened)
            per_target_advantages: (B, Q) per-target advantages from buffer
            tau_d: temperature for inverse-distance softmax (meters)

        Returns:
            target_wise_adv: (B,) aggregated advantage per sample
        """
        import math
        B = obs.shape[0]
        Q = per_target_advantages.shape[1]

        # Observation layout (with rel_features=True):
        #   self(8) | beliefs(Q*9) | geometry(Q*8) | physics(3) | ...
        # Geometry per target: dx, dy, dist_norm, sin, cos, d_s1, d_s2, d_s3
        # dist_norm = raw_distance / diagonal — need to convert back to meters
        area_w, area_h = self.cfg.scenario.region_size
        diag_m = math.hypot(area_w, area_h)

        dist_norm = torch.zeros(B, Q, device=obs.device)
        for q in range(Q):
            offset = 8 + Q * 9 + q * 8 + 2  # self + beliefs + geom(q) + dist_idx
            if offset < obs.shape[1]:
                dist_norm[:, q] = obs[:, offset].abs().clamp(min=1e-8)

        dist_m = dist_norm * diag_m  # normalized → meters

        # Inverse-distance softmax responsibility (detached)
        rho = torch.softmax(-dist_m / tau_d, dim=-1).detach()

        # Per-target advantage: independently normalize each target across batch
        pt_adv_norm = torch.zeros_like(per_target_advantages)
        for q in range(Q):
            aq = per_target_advantages[:, q]
            aq_mean = aq.mean()
            aq_std = aq.std(unbiased=False).clamp(min=1e-8)
            pt_adv_norm[:, q] = (aq - aq_mean) / aq_std

        # Aggregate: weighted sum of normalized per-target advantages
        target_wise_adv = (rho * pt_adv_norm).sum(dim=-1)

        # Re-normalize to match scalar advantage scale (full-batch, once)
        tw_mean = target_wise_adv.mean()
        tw_std = target_wise_adv.std(unbiased=False).clamp(min=1e-8)
        return (target_wise_adv - tw_mean) / tw_std

    def collect_rollout(self) -> bool:
        """Collect a full rollout with parallel environments for GPU batching.

        N environments run in parallel. Observations from all N envs are batched
        into a single GPU forward pass (batch = N*K instead of K), dramatically
        reducing GPU kernel launch overhead and idle time.

        Returns:
            True if any episode ended during rollout
        """
        self.buffer.clear()

        # ── Reset accumulators for this rollout ──
        episode_ended = False
        N = self.num_envs
        K = self.K

        # Reset only envs without current state (first call or after episode end)
        all_obs = []
        for n, env in enumerate(self.envs):
            if self._current_obs[n] is None:
                obs, _ = env.reset(seed=int(env.rng.integers(0, 2**31 - 1)))
                self._current_obs[n] = obs
                # Clear ring buffer on fresh reset
                if self._use_window:
                    self._obs_ring[n].fill(0)
                    self._window_mask[n].fill(False)
            all_obs.append(self._current_obs[n])

        self._rollout_team_rewards = []
        self._rollout_pd = []
        self._rollout_constraint_costs = []
        self._rollout_comm_agent_vars = []    # diagnostic: cross-agent comm variance
        self._rollout_utility = []            # diagnostic: utility (before comm cost)
        self._rollout_comm_cost = []          # diagnostic: comm cost per step
        self._rollout_pd_tensor = None        # (T, Q) tensor for aux loss lookup
        self._rollout_cvar_deficits = []      # CVaR deficit per step

        # Pre-allocate numpy buffers (reused each step)
        dp_mean_np = np.empty((N * K, 2), dtype=np.float64)
        role_logits_np = np.empty((N * K, 3), dtype=np.float64)
        values_np = np.empty(N * K, dtype=np.float64)
        actions_dp = np.zeros((K, 2), dtype=np.float64)
        actions_role = np.zeros(K, dtype=np.int32)
        log_probs = np.zeros(K, dtype=np.float64)

        for step in range(self.steps_per_env):
            # ── Build observation batch from all envs ──
            all_gs_list = [env.core.get_global_state() for env in self.envs]

            obs_batch_list = []
            for n in range(N):
                o = all_obs[n]
                obs_batch_list.append(np.stack([o[str(k)] for k in range(K)]))
            obs_batch = np.concatenate(obs_batch_list)  # (N*K, obs_dim)
            all_gs = np.stack(all_gs_list)               # (N, gs_dim)

            # ── Single GPU forward pass (batch = N*K) ──
            with torch.inference_mode():
                # Copy to pre-allocated GPU tensors
                self._obs_gpu[:N*K].copy_(torch.as_tensor(obs_batch, dtype=torch.float32))
                self._gs_gpu[:N].copy_(torch.as_tensor(all_gs, dtype=torch.float32))

                # Batch GRU hidden states: per-neighbor (N*K*(K-1) total)
            h_prev_list = []
            for n in range(N):
                for k in range(K):
                    for kk in range(K):
                        if kk == k: continue
                        key = (k, kk)  # (agent_id, neighbor_id)
                        h = self.envs[n].core._gru_hidden.get(key)
                        if h is None:
                            h = np.zeros(64, dtype=np.float32)
                        h_prev_list.append(h)
            total_neighbors = N * K * (K-1)
            if h_prev_list:
                h_prev_batch = torch.as_tensor(np.stack(h_prev_list), dtype=torch.float32, device=self.device)
                h_prev_batch = h_prev_batch.unsqueeze(0)  # (1, total_neighbors, D)
            else:
                h_prev_batch = None

            # ── Window construction for TICA actor ──
            actor_window = None
            actor_wmask = None
            if self._use_window:
                # Update ring buffer with current observations
                for n in range(N):
                    for k in range(K):
                        self._obs_ring[n, k, :-1] = self._obs_ring[n, k, 1:]
                        self._obs_ring[n, k, -1] = all_obs[n][str(k)]
                        self._window_mask[n, k, :-1] = self._window_mask[n, k, 1:]
                        self._window_mask[n, k, -1] = True
                # Build window batch: (N*K, L, obs_dim)
                actor_window = torch.as_tensor(
                    self._obs_ring.reshape(N * K, self._window_len, -1),
                    dtype=torch.float32, device=self.device)
                actor_wmask = torch.as_tensor(
                    self._window_mask.reshape(N * K, self._window_len),
                    dtype=torch.bool, device=self.device)

            dp_mean, dp_log_std, role_logits, comm_msgs, _pd_pred, h_new = self.agents[0].actor(
                actor_window if self._use_window else self._obs_gpu[:N*K],
                h_prev_batch,
                window_mask=actor_wmask if self._use_window else None)

            # Apply comm mode BEFORE any downstream use (rollout critic, env, buffer)
            comm_msgs = self._effective_comm(comm_msgs)

            # P0 FIX: save h_prev numpy per env for buffer storage.
            # h_prev_list is ordered: env0_k0_nbr0, env0_k0_nbr1, ..., env0_k1_nbr0, ...
            # Reshape to (N, K, K-1, D) then index per env.
            h_prev_arr = None
            if h_prev_list:
                h_prev_arr = np.stack(h_prev_list).reshape(N, K, K-1, -1)

            # Store comm + GRU hidden state per-neighbor for next frame
            comm_np = comm_msgs.detach().cpu().numpy()  # (N*K, 16) — already effective_comm'd
            h_new_np = h_new.cpu().numpy() if h_new is not None else None
            if h_new_np is not None:
                h_new_np = h_new_np.reshape(N, K, K-1, -1)
            for n in range(N):
                for k in range(K):
                    self.envs[n].core._comm_msgs[k] = comm_np[n*K + k].copy()
                    if h_new_np is not None:
                        ni = 0
                        for kk in range(K):
                            if kk == k: continue
                            self.envs[n].core._gru_hidden[(k, kk)] = h_new_np[n, k, ni].copy()
                            ni += 1

                # Critic input: global state + comm (MAPPO/CTDE) or local obs (IPPO)
                agent_ids = torch.arange(K, device=self.device).repeat(N)
                agent_oh = torch.nn.functional.one_hot(agent_ids, K).float()
                # Aggregate comm per env (mean across K agents), repeat for each agent
                comm_agg = comm_msgs.reshape(N, K, -1).mean(dim=1)  # (N, 16)
                comm_agg_rep = comm_agg.repeat_interleave(K, dim=0)  # (N*K, 16)
                if self.centralized_critic:
                    base = self._gs_gpu[:N].repeat_interleave(K, dim=0)  # (N*K, gs_dim)
                else:
                    base = self._obs_gpu[:N*K]                          # IPPO: local obs
                gs_with_id = torch.cat([base, agent_oh, comm_agg_rep], dim=-1)
                values_t = self.agents[0].critic(gs_with_id)
                # S3b: per-target values (diagnostic)
                _, target_values_t = self.agents[0].critic.forward_with_targets(gs_with_id)
                target_v_np = target_values_t.detach().cpu().numpy() if target_values_t is not None else None

            # Copy results back to CPU (single transfer per rollout step)
            dp_mean_np[:] = dp_mean.detach().cpu().numpy()
            dp_std_np = dp_log_std.detach().cpu().numpy()  # (2,) shared param
            role_logits_np[:] = role_logits.detach().cpu().numpy()
            values_np[:] = values_t.detach().cpu().numpy()

            # ── Per-env action decode + step + buffer store ──
            for n in range(N):
                env = self.envs[n]
                obs = all_obs[n]
                idx0 = n * K
                idx1 = idx0 + K

                # Decode actions for this env's agents
                for k in range(K):
                    local_idx = idx0 + k
                    action, lp = self.agents[k].action_space.decode(
                        dp_mean_np[local_idx], dp_std_np, role_logits_np[local_idx]
                    )
                    actions_dp[k] = action.delta_p
                    actions_role[k] = action.role
                    log_probs[k] = lp

                # Oracle-guided exploration: replace actions with Greedy-Approach
                # with probability α (decaying). Oracle actions are NOT trained on.
                oracle_mask = np.ones(K, dtype=np.float64)  # 1=actor, 0=oracle
                r = self.envs[n].core.rng.random()
                if r < self._oracle_alpha:
                    tgt_pos = np.array([t.get_position_3d()
                                       for t in self.envs[n].core.targets])
                    max_dp = self.cfg.uav.v_max * self.cfg.scenario.dt
                    for k in range(K):
                        pos = self.envs[n].core.uavs[k].pos[:2]
                        q = int(np.argmin(
                            [np.linalg.norm(tgt_pos[qq][:2] - pos)
                             for qq in range(self.Q)]))
                        d = tgt_pos[q][:2] - pos
                        norm = np.linalg.norm(d)
                        oracle_dp = d / norm * max_dp if norm > 1e-6 else np.zeros(2)
                        actions_dp[k] = oracle_dp
                        oracle_mask[k] = 0.0  # exclude from PPO loss
                        log_probs[k] = 0.0   # neutral log_prob for ratio

                # Build actions dict once per macro step
                actions_dict = {
                    str(k): {'delta_p': actions_dp[k], 'role': int(actions_role[k])}
                    for k in range(K)
                }

                # Macro-loop: hold same action for macro_interval micro-frames
                macro_rewards = {k: 0.0 for k in range(K)}
                macro_pd = []
                macro_costs = []
                for micro in range(self.macro_interval):
                    next_obs, rewards, terminated, truncated, info = env.step(actions_dict)
                    w = self.gamma_micro ** micro
                    for k in range(K):
                        macro_rewards[k] += w * float(rewards[str(k)])
                    macro_pd.append(info.get('P_D_q', np.zeros(self.Q)).copy())
                    macro_costs.append(float(info.get('constraint_info', {}).get('any_violation', 0.0)))
                    if terminated.get('__all__', False) or truncated.get('__all__', False):
                        break

                # Lagrangian penalty on mean constraint cost over macro step
                constraint_cost = float(np.mean(macro_costs)) if macro_costs else 0.0

                # ── Fix #1: convert string keys to int ──
                obs_int = {int(k): v for k, v in obs.items()}
                rewards_int = {k: float(v) for k, v in macro_rewards.items()}

                # Augment reward with Lagrangian constraint penalty
                for k in range(K):
                    rewards_int[k] = rewards_int[k] - self.lagrangian_lambda * constraint_cost

                # CVaR target-deficit penalty (TRC: Target-Tail-Risk Constraint)
                tau_cvar = getattr(self, '_cvar_tau', 0.3)
                if tau_cvar > 0:
                    pd_frame = np.array(macro_pd[-1]) if macro_pd else np.zeros(self.Q)
                    deficits = np.maximum(0.0, tau_cvar - pd_frame)  # (Q,)
                    cvar_k = compute_cvar_k(self.Q)
                    sorted_def = np.sort(deficits)[::-1]
                    cvar_deficit = float(np.mean(sorted_def[:cvar_k]))
                    self._rollout_cvar_deficits.append(cvar_deficit)
                    cvar_penalty = self._cvar_lambda * cvar_deficit
                    for k in range(K):
                        rewards_int[k] = rewards_int[k] - cvar_penalty

                # ── Defensive: ensure __all__ propagates to per-agent dones ──
                done_all = bool(
                    terminated.get('__all__', False)
                    or truncated.get('__all__', False)
                )
                dones_dict = {
                    k: bool(
                        done_all
                        or terminated.get(str(k), False)
                        or truncated.get(str(k), False)
                    )
                    for k in range(K)
                }

                # Aggregate comm for critic: mean of all agents' messages this step
                comm_agg_n = np.mean(comm_np[n*K:(n+1)*K], axis=0)  # (16,)
                gs_with_comm = np.concatenate([all_gs_list[n], comm_agg_n])  # (65+16=81)
                # Per-target rewards: P_D_q for each agent (same for all, from macro_pd)
                pt_rewards = np.tile(np.mean(macro_pd, axis=0) if macro_pd else np.zeros(Q), (K, 1))
                # Per-target values from critic
                pt_values = target_v_np[idx0:idx1] if target_v_np is not None else None
                # P0 FIX: extract h_prev for this env
                env_h_prev = h_prev_arr[n] if h_prev_arr is not None else None
                # TICA window: save pre-action window for this env
                env_window = (self._obs_ring[n].copy() if self._use_window
                              else None)
                self.buffer.store(
                    obs=obs_int,
                    global_state=gs_with_comm,
                    actions_dp=actions_dp,
                    actions_role=actions_role,
                    log_probs=log_probs,
                    values=values_np[idx0:idx1],
                    rewards=rewards_int,
                    dones=dones_dict,
                    oracle_mask=oracle_mask,
                    per_target_rewards=pt_rewards,
                    per_target_values=pt_values,
                    h_prev=env_h_prev,
                    obs_window=env_window,
                )
                self.total_frames += self.macro_interval

                # Accumulate rollout-level metrics (mean over macro frames)
                if macro_pd:
                    self._rollout_pd.append(np.mean(macro_pd, axis=0))
                self._rollout_constraint_costs.append(constraint_cost)
                # Fix: populate team_reward from env info
                self._rollout_team_rewards.append(float(info.get('team_reward', 0.0)))
                # Diagnostic: comm message variance (across K agents)
                self._rollout_comm_agent_vars.append(float(np.var(comm_np[n*K:(n+1)*K])))
                # Diagnostic: utility from P_D (without comm cost)
                avg_pd_frame = np.mean(info.get('P_D_q', np.zeros(self.Q)))
                self._rollout_utility.append(float(avg_pd_frame))
                # Approximate comm cost
                p0_bits = info.get('total_bits', 0.0)
                lambda_r = getattr(self, '_lambda_report', 1e-5)
                self._rollout_comm_cost.append(float(lambda_r * p0_bits))

                # Decay entropy
                decay_progress = min(1.0, self.total_frames / self.entropy_decay_frames)
                self.entropy_coef = (
                    self.entropy_init
                    + decay_progress * (self.entropy_final - self.entropy_init)
                )

                # ── Only reset when episode actually ends ──
                if done_all:
                    episode_ended = True
                    new_obs, _ = env.reset(
                        seed=int(env.rng.integers(0, 2**31 - 1))
                    )
                    all_obs[n] = new_obs
                    # Clear ring buffer on episode boundary
                    if self._use_window:
                        self._obs_ring[n].fill(0)
                        self._window_mask[n].fill(False)
                else:
                    all_obs[n] = next_obs

        # Save per-env state for next rollout continuity
        self._current_obs = all_obs

        # ── Compute GAE: final values for each env ──
        # P1 FIX: use final observations from all_obs (NOT stale _obs_gpu),
        # final GRU hidden states from envs, and compute per-target bootstrap.
        with torch.inference_mode():
            agent_ids = torch.arange(K, device=self.device).repeat(N)
            agent_oh = torch.nn.functional.one_hot(agent_ids, K).float()

            # Build final obs batch from actual final observations
            final_obs_batch = np.concatenate(
                [np.stack([all_obs[n][str(k)] for k in range(K)]) for n in range(N)])
            self._obs_gpu[:N*K].copy_(torch.as_tensor(final_obs_batch, dtype=torch.float32))

            # Build final GRU hidden state batch from envs
            final_h_prev_list = []
            for n in range(N):
                for k in range(K):
                    for kk in range(K):
                        if kk == k: continue
                        key = (k, kk)
                        h = self.envs[n].core._gru_hidden.get(key)
                        if h is None:
                            h = np.zeros(self.buffer.gru_hidden_dim if self.buffer._has_gru else 64, dtype=np.float32)
                        final_h_prev_list.append(h)
            final_h_batch = None
            if final_h_prev_list:
                final_h_batch = torch.as_tensor(
                    np.stack(final_h_prev_list), dtype=torch.float32, device=self.device
                ).unsqueeze(0)  # (1, N*K*(K-1), D)

            # Forward actor on final obs WITH final GRU state
            _, _, _, final_comm, _, _ = self.agents[0].actor(
                self._obs_gpu[:N*K], final_h_batch)
            final_comm = self._effective_comm(final_comm)
            final_comm_agg = final_comm.reshape(N, K, -1).mean(dim=1).repeat_interleave(K, dim=0)

            if self.centralized_critic:
                final_gs_batch = np.stack([e.core.get_global_state() for e in self.envs])
                self._gs_gpu[:N].copy_(torch.as_tensor(final_gs_batch, dtype=torch.float32))
                base = self._gs_gpu[:N].repeat_interleave(K, dim=0)
            else:
                base = self._obs_gpu[:N*K]  # already has final obs
            gs_with_id = torch.cat([base, agent_oh, final_comm_agg], dim=-1)

            # Scalar and per-target next values
            next_values = self.agents[0].critic(gs_with_id).detach().cpu().numpy()
            _, next_target_values_t = self.agents[0].critic.forward_with_targets(gs_with_id)
            next_pt_values = (next_target_values_t.detach().cpu().numpy()
                              if next_target_values_t is not None else None)

        # Effective gamma between macro transitions
        gamma_eff = self.gamma_micro ** self.macro_interval
        self.buffer.gamma = gamma_eff
        self.buffer.compute_gae(next_values, next_pt_values)

        return episode_ended

    def update(self) -> Dict[str, float]:
        """Perform PPO-clip update with Lagrangian penalty.

        Returns:
            Dict of training metrics
        """
        if not self.buffer.is_ready():
            return {}

        data = self.buffer.get_training_data()

        # Move to device
        obs = data['obs'].to(self.device)
        global_states = data['global_states'].to(self.device)
        actions_dp = data['actions_dp'].to(self.device)
        actions_role = data['actions_role'].to(self.device)
        old_log_probs = data['old_log_probs'].to(self.device)
        advantages = data['advantages'].to(self.device)
        returns = data['returns'].to(self.device)
        old_values = data['old_values'].to(self.device)

        total_size = obs.shape[0]
        indices = np.arange(total_size)

        metrics = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy': 0.0,
            'approx_kl': 0.0,
            'clip_fraction': 0.0,
            '_n_minibatches': 0,
        }

        agent = self.agents[0]  # shared networks

        # Ensure train mode (eval may have been called between updates)
        agent.actor.train()
        agent.critic.train()

        # S4: pre-compute target-wise advantages ONCE before PPO epochs.
        # This keeps advantages invariant to minibatch split and shuffle.
        tw_advantages = None
        if (self._adv_mode == 'target_wise'
                and 'per_target_advantages' in data):
            tw_advantages = self._compute_target_wise_advantage(
                obs, data['per_target_advantages'].to(self.device),
                tau_d=self._resp_tau_m)

        # ═══════════════════════════════════════════════════════════════
        # P0 ASSERTION: old-log-prob consistency check.
        # Verifies that recomputing log-probs with stored h_prev reproduces
        # the old log-probs from rollout. If this fails, the PPO ratio is
        # invalid BEFORE any optimizer step — all training results are suspect.
        # ═══════════════════════════════════════════════════════════════
        if not hasattr(self, '_consistency_checked'):
            self._consistency_checked = True
            _check_n = min(512, total_size)
            _check_idx = np.arange(_check_n)
            _check_obs = obs[_check_idx]
            _check_dp = actions_dp[_check_idx]
            _check_role = actions_role[_check_idx]
            _check_old_lp = old_log_probs[_check_idx]
            _check_h = None
            if 'h_prev' in data:
                _check_h_full = data['h_prev'][_check_idx]
                _check_h = _check_h_full.reshape(1, -1, _check_h_full.shape[-1]).to(self.device)

            _check_w = None
            _check_wm = None
            if 'obs_window' in data:
                _check_w = data['obs_window'][_check_idx].to(self.device)

            passed, max_diff = agent.verify_old_log_prob_consistency(
                _check_w if _check_w is not None else _check_obs,
                _check_dp, _check_role, _check_old_lp,
                h_prev=_check_h,
            )
            if not passed:
                print(f'[PPO RATIO ERROR] old_log_prob != recomputed_log_prob: '
                      f'max|diff|={max_diff:.6f} > tolerance=1e-4')
                print(f'  → PPO ratio r_t ≠ 1 before any optimizer step. '
                      f'Training results are CONTAMINATED.')
                print(f'  → Likely cause: GRU h_prev mismatch between rollout and update, '
                      f'or dp_scale mismatch between ActionSpace.decode and evaluate_actions.')
            else:
                print(f'[PPO RATIO OK] old_log_prob matches recomputed: max|diff|={max_diff:.6f} < 1e-4')

        kl_stop = False
        for epoch in range(self.ppo_epochs):
            if kl_stop:
                break
            np.random.shuffle(indices)

            for start in range(0, total_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]

                mb_obs = obs[mb_idx]
                # Critic input: global state (MAPPO/CTDE) or local obs (IPPO) + agent one-hot.
                # obs index b → timestep row = b // K, agent = b % K
                agent_ids_mb = torch.as_tensor(mb_idx % self.K, device=self.device)
                agent_oh_mb = torch.nn.functional.one_hot(agent_ids_mb, self.K).float()
                if self.centralized_critic:
                    mb_gs_indices = mb_idx // self.K
                    mb_base = global_states[mb_gs_indices]
                else:
                    mb_base = mb_obs   # IPPO: local observation
                    # IPPO critic expects obs_dim + K + comm_dim.
                    # Buffer stores global_state+comm; for IPPO, pad local obs with comm.
                    comm_pad = torch.zeros(len(mb_idx), 16, device=self.device)
                    mb_base = torch.cat([mb_base, comm_pad], dim=-1)
                mb_gs = torch.cat([mb_base, agent_oh_mb], dim=-1)

                mb_actions_dp = actions_dp[mb_idx]
                mb_actions_role = actions_role[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                # advantages already normalized once globally in buffer.get_training_data();
                # do NOT re-normalize per minibatch (that was a double normalization).
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_old_values = old_values[mb_idx]

                # S4: use pre-computed target-wise advantage (invariant to minibatch)
                if tw_advantages is not None:
                    mb_advantages = tw_advantages[mb_idx]

                # P0 FIX: pass stored GRU hidden states so PPO ratio compares
                # distributions conditioned on the SAME h_prev as rollout.
                mb_h_prev = None
                if 'h_prev' in data:
                    mb_h_prev_full = data['h_prev'][mb_idx].to(self.device)  # (mb, K-1, D)
                    # Reshape to (1, mb*(K-1), D) for GRU forward
                    mb_h_prev = mb_h_prev_full.reshape(1, -1, mb_h_prev_full.shape[-1])

                # P0 FIX: pass stored GRU hidden states + TICA window
                mb_window = None
                mb_wmask = None
                if 'obs_window' in data:
                    mb_window = data['obs_window'][mb_idx].to(self.device)
                    # window_mask: check if buffer stores it
                    if 'window_mask' in data:
                        mb_wmask = data['window_mask'][mb_idx].to(self.device)

                # Evaluate actions — obs can be (B, obs_dim) or (B, L, obs_dim)
                new_log_probs, values, entropies, dp_means, _, comm_batch = agent.evaluate_actions(
                    mb_window if mb_window is not None else mb_obs,
                    mb_gs, mb_actions_dp, mb_actions_role,
                    h_prev=mb_h_prev,
                    window_mask=mb_wmask,
                )

                # PPO-clip loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # PPO-clip loss
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss: Huber (smooth L1), robust to return outliers
                critic_loss = torch.nn.functional.smooth_l1_loss(
                    values, mb_returns
                )

                # Entropy bonus
                entropy = entropies.mean()

                # DAgger reference KL anchor: KL(π_ref || π_θ)
                # Keeps policy near DAgger during conservative fine-tuning
                ref_kl_loss = 0.0
                if self._bc_actor is not None and self._ref_beta > 0:
                    with torch.no_grad():
                        ref_mean, ref_log_std, _, _, _, _ = self._bc_actor(mb_obs)
                    log_std_new = self.agents[0].actor.dp_log_std
                    std_new = torch.exp(log_std_new)
                    std_ref = torch.exp(ref_log_std)
                    kl_per_dim = (ref_log_std - log_std_new
                        + (std_new.pow(2) + (dp_means - ref_mean).pow(2)) / (2 * std_ref.pow(2))
                        - 0.5)
                    ref_kl_loss = kl_per_dim.mean()
                bc_loss = ref_kl_loss  # replace MSE BC with KL anchor

                # Comm loss: disabled when learned_comm_mode='off'
                if self._comm_off:
                    loss_comm = torch.tensor(0.0, device=self.device)
                else:
                    comm_var = comm_batch.var(dim=0).mean()
                    comm_coeff = 0.001 if comm_var > 0.05 else 0.01
                    loss_comm = -comm_coeff * comm_var

                    # Intention: force comm to encode "which target I'm flying toward"
                    if hasattr(agent.actor, 'intent_head'):
                        intent_logits = agent.actor.intent_head(comm_batch)
                        Q = self.Q
                        target_dists = []
                        for q in range(Q):
                            offset = 8 + 9*Q + 8*q + 2
                            target_dists.append(mb_obs[:, offset:offset+1])
                        true_target = torch.cat(target_dists, dim=-1).argmin(dim=-1)
                        loss_intent = torch.nn.functional.cross_entropy(intent_logits, true_target)
                        loss_comm = loss_comm + 0.05 * loss_intent

                # Total loss
                loss = (
                    actor_loss
                    + self.vf_coef * critic_loss
                    - self.entropy_coef * entropy
                    + self._bc_beta * bc_loss
                    + loss_comm
                )

                # Backward
                agent.actor_optimizer.zero_grad()
                agent.critic_optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(agent.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(agent.critic.parameters(), self.max_grad_norm)

                agent.actor_optimizer.step()
                agent.critic_optimizer.step()

                # Track metrics
                metrics['actor_loss'] += actor_loss.item()
                metrics['critic_loss'] += critic_loss.item()
                metrics['entropy'] += entropy.item()
                metrics['approx_kl'] += ((ratio - 1.0) - torch.log(ratio)).mean().item()
                metrics['clip_fraction'] += ((ratio < 1.0 - self.ppo_clip) | (ratio > 1.0 + self.ppo_clip)).float().mean().item()
                metrics['_n_minibatches'] += 1

                # KL early-stop DISABLED: too aggressive for MARL dynamics.
                # PPO-clip (ε=0.1) already prevents destructive updates.
                # with torch.no_grad():
                #     approx_kl_mb = ((ratio - 1.0) - torch.log(ratio)).mean().item()
                # if np.isnan(approx_kl_mb) or approx_kl_mb > 1.5 * self.target_kl:
                #     kl_stop = True
                #     break

        # Average metrics (use ACTUAL minibatch count, not expected)
        n_actual = max(metrics.pop('_n_minibatches', 1), 1)
        for k in metrics:
            metrics[k] /= max(n_actual, 1)

        # ── Lagrangian update from full rollout statistics ──
        if self._rollout_constraint_costs:
            mean_violation = float(np.mean(self._rollout_constraint_costs))
        else:
            mean_violation = 0.0

        self.lagrangian_lambda = float(np.clip(
            self.lagrangian_lambda
            + self.lagrangian_lr * (mean_violation - self.max_violation_rate),
            0.0,
            self.lagrangian_max,
        ))

        metrics['lagrangian_lambda'] = self.lagrangian_lambda
        metrics['entropy_coef'] = self.entropy_coef

        # CVaR Lagrangian update
        if hasattr(self, '_cvar_lambda') and self._rollout_cvar_deficits:
            mean_cvar = float(np.mean(self._rollout_cvar_deficits))
            cvar_epsilon = getattr(self, '_cvar_epsilon', 0.05)
            self._cvar_lambda = float(np.clip(
                self._cvar_lambda + 0.01 * (mean_cvar - cvar_epsilon), 0.0, 2.0))
            metrics['cvar_deficit'] = mean_cvar
            metrics['cvar_lambda'] = self._cvar_lambda

        # ── Diagnostic: is the critic fitting? are returns trending up/down? ──
        # value≈0 while return≈tens => critic not fitting (value-clip on raw scale).
        # return trending DOWN over training => reward/advantage direction problem.
        metrics['mean_return'] = float(returns.mean().item())
        metrics['mean_value'] = float(old_values.mean().item())
        metrics['mean_adv_abs'] = float(advantages.abs().mean().item())  # ~0.8 if normalized

        return metrics

    def train_episode(self) -> Dict[str, float]:
        """Train for one episode (one rollout + one update).

        Returns:
            Dict of episode metrics
        """
        episode_ended = self.collect_rollout()
        metrics = self.update()
        # MSE BC anchor: hold constant (no decay).
        if self._bc_beta_init > 0:
            self._bc_beta = self._bc_beta_init
        # LR decay + optional Actor freeze
        self._oracle_ep_count += 1
        freeze_after = getattr(self.cfg.marl, 'freeze_actor_after', 0)
        if freeze_after > 0 and self._oracle_ep_count == freeze_after:
            # Snapshot actor hash at freeze point
            self._freeze_hash = hash(str([
                p.sum().item() for p in self.agents[0].actor.parameters()]))
            print(f'[FREEZE] Ep {freeze_after}: Actor hash={self._freeze_hash}')
        if freeze_after > 0 and self._oracle_ep_count >= freeze_after:
            for agent in self.agents:
                for pg in agent.actor_optimizer.param_groups: pg['lr'] = 0.0
            self._bc_beta = 0.0
            # Verify hash unchanged
            if hasattr(self, '_freeze_hash'):
                cur_hash = hash(str([
                    p.sum().item() for p in self.agents[0].actor.parameters()]))
                if cur_hash != self._freeze_hash:
                    print(f'[FREEZE VIOLATION] Ep {self._oracle_ep_count}: hash changed! {self._freeze_hash}→{cur_hash}')
                    self._freeze_hash = cur_hash
        else:
            current_actor_lr = self.agents[0].actor_optimizer.param_groups[0]['lr']
            if current_actor_lr > 1e-4:
                if self._oracle_ep_count < 100: lr = 3e-4
                elif self._oracle_ep_count < 200: lr = 1e-4
                else: lr = 3e-5
                for agent in self.agents:
                    for pg in agent.actor_optimizer.param_groups: pg['lr'] = lr
                    for pg in agent.critic_optimizer.param_groups: pg['lr'] = lr * 5.0
        return metrics

    def _evaluate(self, n_episodes: int = 5, steady_window: int = 20,
                  dp_deterministic: bool = True, role_deterministic: bool = True,
                  eval_seeds: Optional[List[int]] = None) -> Dict[str, float]:
        """Evaluation on fixed replayable scenarios (no exploration noise).

        All fairness metrics (worst, weak3, tstd) are computed from the STEADY
        WINDOW (last W frames) per episode, then averaged across episodes.
        This prevents early-transient frames from contaminating convergence metrics.

        Returns:
          eval_steady_P_D:   mean over episodes of mean over last-W frames
          eval_worst_P_D:    mean over episodes of MIN_q P_D in last-W frames
          eval_weak3_P_D:    mean over episodes of bottom-3 avg in last-W frames
          eval_target_std:   mean over episodes of std_q in last-W frames
          eval_full_P_D:     mean over episodes of full-episode mean P_D
        """
        actor = self.agents[0].actor
        aspace = self.agents[0].action_space
        K, Q = self.K, self.Q
        if eval_seeds is None:
            eval_seeds = self.eval_seeds[:n_episodes]
        eval_env = UAVISACEnv(config=self.cfg, seed=12345)

        ep_full_means = []        # full-episode mean P_D per episode
        ep_steady_means = []      # steady-window mean P_D per episode
        ep_worst = []             # per-episode steady-window worst target
        ep_weak3 = []             # per-episode steady-window bottom-3 avg
        ep_tstd = []              # per-episode steady-window target std
        ep_per_target = []        # (n_eps, Q) steady-window per-target means
        vp_frames = notx_frames = samerole_frames = total_frames = 0
        W = steady_window

        for ep_seed in eval_seeds:
            obs, _ = eval_env.reset(seed=int(ep_seed))
            pd_hist = []   # list of mean P_D_q per frame
            pd_per_target = []  # list of (Q,) per frame
            # P0 FIX: maintain streaming GRU hidden state during eval.
            # Previously eval called actor(obs) without h_prev → h=0 every frame,
            # inconsistent with rollout (which uses cumulative GRU state).
            eval_h_prev = None  # None → zero-init on first frame
            while True:
                ob = np.stack([obs[str(k)] for k in range(K)])
                with torch.inference_mode():
                    ob_t = torch.as_tensor(ob, dtype=torch.float32, device=self.device)
                    dp_mean, dp_log_std, role_logits, _, _, h_new = actor(ob_t, eval_h_prev)
                    # Save h_new for next frame (detach already in inference_mode)
                    eval_h_prev = h_new
                dpm = dp_mean.detach().cpu().numpy()
                dps = dp_log_std.detach().cpu().numpy()
                rl = role_logits.detach().cpu().numpy()
                actions = {}
                for k in range(K):
                    a, _ = aspace.decode(dpm[k], dps, rl[k],
                                         dp_deterministic=dp_deterministic,
                                         role_deterministic=role_deterministic)
                    actions[str(k)] = {'delta_p': a.delta_p, 'role': a.role}
                obs, _, term, trunc, info = eval_env.step(actions)
                pd_q = info['P_D_q'].copy()
                pd_hist.append(np.mean(pd_q))
                pd_per_target.append(pd_q)
                total_frames += 1
                vp_frames += int(info.get('valid_pair', False))
                notx_frames += int(info.get('no_tx', False))
                samerole_frames += int(info.get('all_same_role', False))
                if term.get('__all__', False) or trunc.get('__all__', False):
                    break
            if pd_hist:
                ep_full_means.append(float(np.mean(pd_hist)))
                w = min(W, len(pd_hist))
                # Steady window: per-target matrix (w, Q)
                steady_pd = np.array(pd_per_target[-w:])  # (w, Q)
                steady_per_target = steady_pd.mean(axis=0)  # (Q,)
                ep_steady_means.append(float(np.mean(steady_per_target)))
                ep_per_target.append(steady_per_target)
                # Episode-wise: min and bottom-3 within THIS episode's steady window
                sorted_q = np.sort(steady_per_target)
                ep_worst.append(float(sorted_q[0]))
                ep_weak3.append(float(np.mean(sorted_q[:3])))
                ep_tstd.append(float(steady_per_target.std()))

        tf = max(total_frames, 1)
        n_eps_completed = len(ep_steady_means)
        if n_eps_completed == 0:
            return {'eval_steady_P_D': 0.0, 'eval_worst_P_D': 0.0,
                    'eval_weak3_P_D': 0.0, 'eval_target_std': 0.0,
                    'eval_full_P_D': 0.0}

        # Per-target matrix for fixed-identity tracking
        if ep_per_target:
            per_target_mat = np.array(ep_per_target)  # (E, Q)
            per_target_avg = per_target_mat.mean(axis=0)  # (Q,)
        else:
            per_target_avg = np.zeros(Q)

        return {
            'eval_steady_P_D': float(np.mean(ep_steady_means)),
            'eval_full_P_D': float(np.mean(ep_full_means)),
            # Episode-wise: average of per-episode worst/bottom3/std
            'eval_worst_P_D': float(np.mean(ep_worst)),
            'eval_weak3_P_D': float(np.mean(ep_weak3)),
            'eval_target_std': float(np.mean(ep_tstd)),
            # Per-target identity tracking
            'eval_per_target': per_target_avg.tolist(),
            # Pairing diagnostics
            'valid_pair_rate': vp_frames / tf,
            'no_TX_rate': notx_frames / tf,
            'all_same_role_rate': samerole_frames / tf,
        }

    def _evaluate_modes(self, n_episodes: Optional[int] = None,
                        steady_window: int = 20) -> Dict[str, Dict[str, float]]:
        """Run all four decode modes on the IDENTICAL fixed scenarios (P0 diagnostic).

        Isolates whether eval collapse comes from the continuous action or the
        discrete role by holding the scenario set constant and toggling only the
        determinism of each head:

          dp_det_role_stoch : continuous frozen, role sampled
          dp_stoch_role_det : role frozen, continuous sampled
          full_greedy       : both frozen   (== legacy deterministic eval)
          full_stochastic   : both sampled

        If dp-frozen modes (dp_det_role_stoch / full_greedy) >> dp-sampled modes,
        the continuous head is fine and the role head is the problem; the reverse
        implicates the continuous action. Returns a dict keyed by mode name, each
        holding the _evaluate() metrics.
        """
        n = n_episodes if n_episodes is not None else len(self.eval_seeds)
        seeds = self.eval_seeds[:n]
        # Descriptive names (no ambiguous A/B/C/D letters): (dp_deterministic, role_deterministic)
        modes = {
            'dp_det_role_stoch': (True, False),
            'dp_stoch_role_det': (False, True),
            'full_greedy':       (True, True),
            'full_stochastic':   (False, False),
        }
        out: Dict[str, Dict[str, float]] = {}
        for name, (dp_det, role_det) in modes.items():
            out[name] = self._evaluate(
                n_episodes=n, steady_window=steady_window,
                dp_deterministic=dp_det, role_deterministic=role_det,
                eval_seeds=seeds,
            )
        return out

    def train(self, num_episodes: Optional[int] = None, log_interval: int = 10,
              eval_interval: int = 100) -> List[Dict]:
        """Main training loop.

        Args:
            num_episodes: Number of episodes; defaults to config value
            log_interval: Print metrics every N episodes
            eval_interval: Run evaluation every N episodes

        Returns:
            List of per-episode metrics dicts
        """
        n_episodes: int = self.num_episodes if num_episodes is None else num_episodes

        all_metrics = []

        for ep in range(n_episodes):
            ep_start = time.time()

            metrics = self.train_episode()

            # ── Fix #9: rollout-average metrics ──
            if self._rollout_team_rewards:
                metrics['team_reward'] = float(np.mean(self._rollout_team_rewards))
            if self._rollout_pd:
                all_pd = np.stack(self._rollout_pd)           # (T, Q)
                metrics['avg_P_D'] = float(np.mean(all_pd))
                # worst-target: min over targets of time-averaged P_D
                per_target_mean = np.mean(all_pd, axis=0)     # (Q,)
                metrics['worst_P_D'] = float(np.min(per_target_mean))
            if self._rollout_constraint_costs:
                metrics['constraint_violation_rate'] = float(
                    np.mean(self._rollout_constraint_costs)
                )
            if self._rollout_utility:
                metrics['mean_utility'] = float(np.mean(self._rollout_utility))
            if self._rollout_comm_agent_vars:
                metrics['comm_agent_var'] = float(np.mean(self._rollout_comm_agent_vars))
            if self._rollout_comm_cost:
                metrics['comm_cost'] = float(np.mean(self._rollout_comm_cost))
            # Build P_D tensor for aux loss lookup during update
            if self._rollout_pd:
                self._rollout_pd_tensor = torch.as_tensor(
                    np.stack(self._rollout_pd), dtype=torch.float32, device=self.device)
            else:
                self._rollout_pd_tensor = None

            metrics['episode'] = ep
            metrics['total_frames'] = self.total_frames
            metrics['time'] = time.time() - ep_start

            all_metrics.append(metrics)

            if ep % log_interval == 0 or ep == n_episodes - 1:
                pd_str = ""
                if 'avg_P_D' in metrics:
                    pd_str = f" avg_P_D={metrics['avg_P_D']:.3f}"
                print(
                    f"Ep {ep:4d}/{n_episodes} | "
                    f"actor_loss={metrics.get('actor_loss', 0):.4f} "
                    f"critic_loss={metrics.get('critic_loss', 0):.4f} "
                    f"entropy={metrics.get('entropy', 0):.3f} "
                    f"kl={metrics.get('approx_kl', 0):.4f} "
                    f"λ={self.lagrangian_lambda:.3f} "
                    # DIAGNOSTIC: ret=return scale, val=critic output (should converge to ret),
                    # advA=|normalized adv| (~0.8); reward=mean team reward (should trend UP)
                    f"ret={metrics.get('mean_return', 0):.2f} "
                    f"val={metrics.get('mean_value', 0):.2f} "
                    f"reward={metrics.get('team_reward', 0):.3f} "
                    # REWARD DIAGNOSTIC: util=mean P_D (proxy for utility), cVar=comm msg variance
                    f"util={metrics.get('mean_utility', 0):.3f} "
                    f"cAgVar={metrics.get('comm_agent_var', 0):.3f} "
                    f"CVaR={metrics.get('cvar_deficit', 0):.3f} "
                    f"cλ={metrics.get('cvar_lambda', 0):.3f}"
                    + pd_str
                )

            # ── Convergence-based eval + early stopping ──
            if self.early_stop and (ep % self.eval_interval == 0 or ep == n_episodes - 1):
                ev = self._evaluate(self.eval_episodes)
                score = ev['eval_steady_P_D']
                metrics.update(ev)
                improved = score > self.best_score + self.early_stop_min_delta
                if improved:
                    self.best_score = score
                    self.best_params = self.agents[0].get_params()  # snapshot best policy
                    self._patience = 0
                else:
                    self._patience += 1
                worst = ev.get('eval_worst_P_D', 0)
                weak3 = ev.get('eval_weak3_P_D', 0)
                tstd = ev.get('eval_target_std', 0)
                full = ev.get('eval_full_P_D', 0)
                print(f"  [eval] ep {ep}: steady={score:.3f} worst={worst:.3f} weak3={weak3:.3f} tstd={tstd:.3f} full={full:.3f} "
                      f"(best={self.best_score:.3f})")
                if self._patience >= self.early_stop_patience:
                    self.converged_episode = ep
                    print(f"  [early-stop] converged at ep {ep}: "
                          f"no >{self.early_stop_min_delta} improvement for "
                          f"{self.early_stop_patience} evals. best steady_P_D={self.best_score:.3f}")
                    break

        # Restore best policy (return the converged optimum, not the last step)
        if self.best_params is not None:
            self.agents[0].set_params(self.best_params)
            print(f"[train] restored best policy: steady_P_D={self.best_score:.3f}"
                  + (f" (converged @ ep {self.converged_episode})" if self.converged_episode else ""))

        return all_metrics
