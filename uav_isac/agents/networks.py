"""Neural network architectures for actor and critic.

Actor: shared MLP [256, 256] → heads: dp_mean(2), dp_log_std(2), role_logits(3)
  (shared = obs → 256 → 256, then head: 256 → output)
Critic: MLP [256, 256] → scalar value + per-target value heads
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


def mlp(input_dim: int, hidden_dims: list, output_dim: int,
        activation=nn.ReLU, output_activation=None) -> nn.Sequential:
    """Build an MLP with configurable hidden layers."""
    layers = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(activation())
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class ActorNetwork(nn.Module):
    """MAPPO actor network.

    Input: local observation (obs_dim,)
    Output:
      - dp_mean: (batch, 2) latent Gaussian mean (tanh-squashed in distribution)
      - dp_log_std: (2,) learnable log-std (broadcast over batch)
      - role_logits: (batch, 3) logits for tx/rx/idle
    """

    def __init__(self, obs_dim: int, hidden_layers: list = [256, 256],
                 max_dp: float = 2.5):
        super().__init__()
        self.max_dp = max_dp

        # Shared feature extractor: 2 hidden ReLU layers (obs→256→256→256).
        # FIX: previously hidden_layers[:-1] dropped the 2nd hidden layer
        # (only 1 effective ReLU). Use full hidden_layers so depth matches docs.
        self.shared = mlp(obs_dim, hidden_layers, hidden_layers[-1])

        # Heads
        self.dp_mean_head = nn.Linear(hidden_layers[-1], 2)
        self.comm_head = nn.Linear(hidden_layers[-1], 16)  # communication message (16-dim)
        self.pd_aux_head = nn.Linear(hidden_layers[-1], 1) # auxiliary P_D predictor
        # init 0: with range (-1,1), tanh(0)=0 -> log_std=0 -> sigma=1 (matches the
        # high-entropy regime that learned early).
        self.dp_log_std = nn.Parameter(torch.zeros(2))  # learnable, shared
        self.role_head = nn.Linear(hidden_layers[-1], 3)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """PPO-style init: hidden layers sqrt(2), output heads small."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'head' in name or name.endswith('role_head'):
                    # Policy output: small weights for stable initial exploration
                    nn.init.orthogonal_(module.weight, gain=0.01)
                else:
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (batch, obs_dim)

        Returns:
            dp_mean: (batch, 2) latent Gaussian mean (unbounded; tanh-squash in distribution)
            dp_log_std: (2,) learnable log-std parameter (broadcast over batch)
            role_logits: (batch, 3) logits for tx/rx/idle
        """
        h = self.shared(obs)
        dp_mean = self.dp_mean_head(h)  # no tanh — handled by tanh-squash distribution
        role_logits = self.role_head(h)
        comm_msg = torch.tanh(self.comm_head(h))  # no saturation
        pd_pred = torch.sigmoid(self.pd_aux_head(h))  # aux P_D prediction (0~1)
        LOG_STD_MIN, LOG_STD_MAX = -1.0, 1.0
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (torch.tanh(self.dp_log_std) + 1.0)
        return dp_mean, log_std, role_logits, comm_msg, pd_pred


class StructuredActorNetwork(nn.Module):
    """Entity-based actor with cross-attention and gated communication.

    Instead of a flat MLP over 248-dim obs, this encodes self, targets, and
    neighbors as SEPARATE entities, pools them via cross-attention, and gates
    the communication channel. This:
      - eliminates permutation sensitivity (targets/neighbors are sets)
      - lets attention automatically ignore distant/irrelevant entities
      - protects the BC-learned physical policy from comm noise via a gate

    P1 FIX: PD_hist (previous-frame per-target detection probability) is now
    projected and added to the target entity encoding, so the Actor can
    directly condition on which targets had low/high detection.

    NOTE: PD_hist in the local observation IS the UAV's own LOCAL detection
    confidence — P_D computed from deflection of bistatic pairs where THIS UAV
    is tx or rx (not global fused P_D). Each UAV sees different PD_hist values.
    This respects the decentralized information boundary: no free global
    fusion-centre broadcast. See env_core.py prev_P_D_local (2026-07-14 fix).
    """

    def __init__(self, obs_dim: int, K: int = 8, Q: int = 8,
                 entity_dim: int = 128, max_dp: float = 2.5,
                 single_frame_dim: int = 0,
                 use_corrected_parser: bool = False,
                 use_p0: bool = False):
        super().__init__()
        self.K, self.Q = K, Q
        self.max_dp = max_dp
        D = entity_dim
        self.single_frame_dim = single_frame_dim
        self._use_corrected_parser = use_corrected_parser
        self._use_p0 = use_p0
        if use_corrected_parser:
            from uav_isac.environment.observation_slices import ObservationSlices
            self._obs_slices = ObservationSlices.from_config(
                K=K, Q=Q, use_p0=use_p0, use_rel_features=True)

        # ── Entity encoders ──
        self.self_enc = nn.Sequential(
            nn.Linear(11, D), nn.ReLU(), nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D),
        )
        self.target_enc = nn.Sequential(
            nn.Linear(18, D), nn.ReLU(), nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D),
        )
        # PD_hist projector: maps scalar P_D (per target) → entity-dim signal
        # that modulates the target encoding, so Actor knows which targets
        # had low detection in the previous frame.
        self.pd_hist_proj = nn.Linear(1, D)

        # GRU for neighbor temporal encoding (handles window=1 gracefully)
        self.neighbor_gru = nn.GRU(input_size=9, hidden_size=D, batch_first=True)
        self.neighbor_proj = nn.Linear(D, D)  # project GRU output
        self.global_enc = nn.Linear(2, D)

        # ── Cross-attention ──
        self.attn = nn.MultiheadAttention(D, num_heads=8, batch_first=True)
        self.attn_norm = nn.LayerNorm(D)

        # ── Communication gate ──
        self.comm_proj = nn.Linear(16, D)
        self.gate = nn.Linear(D + D, 1)

        # ── Output heads ──
        self.dp_head = nn.Linear(D, 2)
        self.comm_head = nn.Linear(D, 16)
        self.intent_head = nn.Linear(16, Q)  # comm→target: force semantic encoding
        self.role_head = nn.Linear(D, 3)
        self.dp_log_std = nn.Parameter(torch.zeros(2))

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                g = 0.01 if 'head' in name else np.sqrt(2)
                nn.init.orthogonal_(m.weight, gain=g)
                nn.init.constant_(m.bias, 0.0)

    def zero_init_new_layers(self, known_keys: set):
        """Zero-initialize layers NOT present in an old checkpoint.

        When loading a DAgger checkpoint that predates pd_hist_proj (or any
        future layer addition), new layers get random orthogonal weights from
        __init__ → _init_weights. That random init changes the policy, so the
        "DAgger baseline" is no longer the true DAgger policy.

        Call this AFTER load_state_dict(..., strict=False) to zero out any
        parameter whose name is NOT in known_keys. For Linear layers, both
        weight and bias are zeroed, making them identity-through-zero:
        e_{kq} = e_{kq}^{base} + 0 = e_{kq}^{base}.

        Args:
            known_keys: set of parameter names present in the old checkpoint.
        """
        with torch.no_grad():
            for n, p in self.named_parameters():
                if n not in known_keys:
                    p.zero_()
                    print(f'  [zero_init] {n} ← zeros (not in old checkpoint)')

    def _parse_obs(self, obs: torch.Tensor):
        """Parse flat obs. Returns entity tensors as sequences (B, N, W, D)."""
        B = obs.shape[0]
        obs_dim = obs.shape[1]
        single_dim = self.single_frame_dim if self.single_frame_dim > 0 else obs_dim

        parse_fn = (self._parse_one_corrected if self._use_corrected_parser
                    else self._parse_one)

        if obs_dim > single_dim + 20:
            w = obs_dim // single_dim
            frames = []
            for i in range(w):
                s, t, n, g, c, pd = parse_fn(obs[:, i*single_dim:(i+1)*single_dim], B)
                frames.append((s, t, n, g, c, pd))
            s_seq = torch.stack([f[0] for f in frames], dim=-1)
            t_seq = torch.stack([f[1] for f in frames], dim=-1)
            n_seq = torch.stack([f[2] for f in frames], dim=-1)
            g_seq = torch.stack([f[3] for f in frames], dim=-1)
            pd_seq = torch.stack([f[5] for f in frames], dim=-1)
            return s_seq, t_seq, n_seq, g_seq, frames[-1][4], w, pd_seq
        else:
            s, t, n, g, c, pd = parse_fn(obs, B)
            return (s.unsqueeze(-1), t.unsqueeze(-1), n.unsqueeze(-1),
                    g.unsqueeze(-1), c, 1, pd.unsqueeze(-1))

    def _parse_one_corrected(self, obs, B):
        """Parse using ObservationSlices — correct block-based layout."""
        K, Q = self.K, self.Q
        sl = self._obs_slices

        # Self: 8 + physics(3) = 11
        self_raw = obs[:, sl.self_start:sl.self_start + sl.self_len]
        phys = obs[:, sl.physics_start:sl.physics_start + sl.physics_len]
        self_state = torch.cat([self_raw, phys], dim=-1)  # (B, 11)

        # Beliefs block (Q × 9) → reshape
        beliefs = obs[:, sl.belief_start:sl.belief_start + Q * sl.belief_per_target]
        beliefs = beliefs.reshape(B, Q, sl.belief_per_target)  # (B, Q, 9)

        # Geometry block (Q × 8) when rel_features
        if sl.has_rel_features and sl.geom_per_target > 0:
            geometry = obs[:, sl.geom_start:sl.geom_start + Q * sl.geom_per_target]
            geometry = geometry.reshape(B, Q, sl.geom_per_target)  # (B, Q, 8)
        else:
            geometry = torch.zeros(B, Q, 0, device=obs.device)

        # Per-target: cat belief + geometry
        targets = []
        for q in range(Q):
            tq = torch.cat([beliefs[:, q, :], geometry[:, q, :]], dim=-1)  # 17 dims
            targets.append(tq)
        target_stack = torch.stack(targets, dim=1)  # (B, Q, 17)

        # Coverage (P0)
        if sl.has_p0 and sl.coverage_len > 0:
            cov = obs[:, sl.coverage_start:sl.coverage_start + sl.coverage_len]
            target_stack = torch.cat([target_stack, cov.unsqueeze(-1)], dim=-1)  # → 18
        if target_stack.shape[-1] < 18:
            target_stack = torch.cat([
                target_stack,
                torch.zeros(B, Q, 18 - target_stack.shape[-1], device=obs.device)
            ], dim=-1)

        # Neighbors block — always pad to 9
        n_dim = sl.neighbor_per_agent
        n_raw = obs[:, sl.neighbor_start:sl.neighbor_start + (K-1) * n_dim]
        neighbors = n_raw.reshape(B, K-1, n_dim)
        if n_dim == 8:
            neighbors = torch.cat([
                neighbors[:, :, :7],
                torch.zeros(B, K-1, 1, device=obs.device),
                neighbors[:, :, 7:]
            ], dim=-1)
        neighbor_stack = neighbors  # (B, K-1, 9)

        # Global
        if sl.has_p0 and sl.global_len > 0:
            global_feat = obs[:, sl.global_start:sl.global_start + sl.global_len]
        else:
            global_feat = torch.zeros(B, 2, device=obs.device)

        # PD_hist and comm
        pd_hist = obs[:, sl.pd_hist_start:sl.pd_hist_start + sl.pd_hist_len]
        comm_agg = obs[:, sl.comm_start:sl.comm_start + sl.comm_len]

        return self_state, target_stack, neighbor_stack, global_feat, comm_agg, pd_hist

    def _parse_one(self, obs, B):
        """Parse single frame (227 dims) into entity tensors.

        Returns:
            self_state, target_stack, neighbor_stack, global_feat, comm_agg, pd_hist
        """
        K, Q = self.K, self.Q
        obs_dim_real = obs.shape[1]

        # Support both 251 (with P0 in_pair) and 227 (no P0) layouts
        ptr = 0
        self_state = obs[:, ptr:ptr+8]; ptr += 8  # 8

        # Targets: belief(9) + geom(8) = 17 per target
        targets = []
        for _ in range(Q):
            b_mean  = obs[:, ptr:ptr+4]; ptr += 4
            b_cov   = obs[:, ptr:ptr+4]; ptr += 4
            b_aoi   = obs[:, ptr:ptr+1]; ptr += 1
            g_dx    = obs[:, ptr:ptr+1]; ptr += 1
            g_dy    = obs[:, ptr:ptr+1]; ptr += 1
            g_dist  = obs[:, ptr:ptr+1]; ptr += 1
            g_sin   = obs[:, ptr:ptr+1]; ptr += 1
            g_cos   = obs[:, ptr:ptr+1]; ptr += 1
            g_s1    = obs[:, ptr:ptr+1]; ptr += 1
            g_s2    = obs[:, ptr:ptr+1]; ptr += 1
            g_s3    = obs[:, ptr:ptr+1]; ptr += 1
            targets.append(torch.cat([
                b_mean, b_cov, b_aoi,
                g_dx, g_dy, g_dist, g_sin, g_cos, g_s1, g_s2, g_s3
            ], dim=-1))  # 17 dims per target

        # Physics features: nearest_dist(1) + bearing_sin(1) + cos(1) = 3
        physics_feat = obs[:, ptr:ptr+3]; ptr += 3

        # Coverage(Q) + pairing(K-1) — may be absent in non-P0 configs
        remaining = obs_dim_real - ptr
        without_p0 = (K-1)*9 + 2 + Q + 16  # neighbors + global + P_D + comm
        has_p0 = (remaining > without_p0 + 2)  # heuristic: P0 adds Q+(K-1) dims

        coverage = None
        if has_p0:
            coverage = obs[:, ptr:ptr+Q]; ptr += Q  # Q
            _pairing = obs[:, ptr:ptr+(K-1)]; ptr += (K-1)  # skip pairing

        # Neighbors: (K-1) × 9 (rel_pos(2)+rel_vel(2)+role(1)+heading(2)+in_pair(1)+nearest(1))
        neighbor_dim = 9 if has_p0 else 8  # in_pair is only there if P0
        neighbors = []
        for _ in range(K-1):
            n = obs[:, ptr:ptr+neighbor_dim]; ptr += neighbor_dim
            if neighbor_dim == 8:
                # Pad missing in_pair with zero
                n = torch.cat([n[:, :7], torch.zeros(B, 1, device=obs.device), n[:, 7:]], dim=-1)
            neighbors.append(n)

        if has_p0:
            global_feat = obs[:, ptr:ptr+2]; ptr += 2  # 2
        else:
            global_feat = torch.zeros(B, 2, device=obs.device)  # no global coord

        # P1 FIX: PD_hist is now KEPT and returned (was previously read and discarded).
        # This is the per-target detection probability from the previous frame,
        # critical for Actor to know which targets are being missed.
        pd_hist = obs[:, ptr:ptr+Q]; ptr += Q  # P_D history (B, Q)

        comm_agg = obs[:, ptr:ptr+16]  # 16

        # Append physics to self_state for encoder
        self_state = torch.cat([self_state, physics_feat], dim=-1)  # (B, 11)

        target_stack = torch.stack(targets, dim=1)      # (B, Q, 17)
        neighbor_stack = torch.stack(neighbors, dim=1)   # (B, K-1, 9)
        if coverage is not None:
            coverage = coverage.unsqueeze(-1)             # (B, Q, 1)
            # Append coverage to target features
            target_stack = torch.cat([target_stack, coverage], dim=-1)  # (B, Q, 18)

        # Ensure target_stack is always 18-dim (pad with zero if no P0 coverage)
        if target_stack.shape[-1] < 18:
            B = target_stack.shape[0]
            pad = torch.zeros(B, Q, 18 - target_stack.shape[-1], device=target_stack.device)
            target_stack = torch.cat([target_stack, pad], dim=-1)

        return self_state, target_stack, neighbor_stack, global_feat, comm_agg, pd_hist

    def forward(self, obs: torch.Tensor, h_prev: torch.Tensor = None,
                detach_h_new: bool = True, window_mask: torch.Tensor = None):
        """Forward with optional streaming GRU hidden state.

        Args:
            obs: (B, obs_dim) observation batch
            h_prev: (1, B*(K-1), D) GRU hidden states, or None for zero-init.
            detach_h_new: If True (default), detach h_new before returning.
            window_mask: ignored (interface compat with TICA).
                True for rollout, evaluation, and PPO single-step updates
                (prevents cross-timestep computation graphs).
                False for DAgger chunk BPTT training (allows gradient flow
                within a chunk; caller must detach at chunk boundaries).

        Returns:
            dp_mean, log_std, role_logits, comm_msg, pd_pred, h_new
        """
        result = self._parse_obs(obs)
        self_s, targets, neighbors, global_f, comm_agg, n_frames, pd_hist = result
        B = obs.shape[0]
        D = self.self_enc[0].out_features
        LOG_STD_MIN, LOG_STD_MAX = -1.0, 1.0
        log_std = LOG_STD_MIN + 0.5*(LOG_STD_MAX-LOG_STD_MIN)*(torch.tanh(self.dp_log_std)+1.0)

        if targets is None:
            h = self.self_enc(torch.zeros(B, 11, device=obs.device))
            return self.dp_head(h), log_std, self.role_head(h), torch.tanh(self.comm_head(h)), torch.zeros(B, 1), None

        # Encode entities: use last timestep (dim=-1 is seq)
        se = self.self_enc(self_s[..., -1]).unsqueeze(1)          # (B, 1, D)
        te_base = self.target_enc(targets[..., -1])                # (B, Q, D)
        ge = self.global_enc(global_f[..., -1]).unsqueeze(1)      # (B, 1, D)

        # P1 FIX: Project PD_hist into target entity encoding.
        # This gives Actor direct knowledge of which targets had low detection
        # in the previous frame — the most direct signal of target failure.
        # PD_hist shape: (B, Q, W) → use last timestep
        pd_last = pd_hist[..., -1]  # (B, Q)
        pd_feat = self.pd_hist_proj(pd_last.unsqueeze(-1))  # (B, Q, D)
        te = te_base + pd_feat  # residual modulation by detection history

        # Streaming GRU for neighbors: (B*Nn, 1, 9) with per-neighbor state
        Nn = neighbors.shape[1]
        n_feat = neighbors[..., -1]  # (B, Nn, 9) — last timestep
        n_input = n_feat.reshape(B * Nn, 1, 9)  # (B*Nn, 1, 9)
        if h_prev is not None:
            _, hn = self.neighbor_gru(n_input, h_prev)
        else:
            _, hn = self.neighbor_gru(n_input)
        ne = self.neighbor_proj(hn.squeeze(0)).reshape(B, Nn, -1)
        h_new = hn.detach() if detach_h_new else hn  # (1, B*Nn, D)

        entities = torch.cat([ge, te, ne], dim=1)
        ctx, _ = self.attn(se, entities, entities)
        h_physical = self.attn_norm(se + ctx).squeeze(1)

        cp = self.comm_proj(comm_agg)
        gate = torch.sigmoid(self.gate(torch.cat([h_physical, cp], dim=-1)))
        h = h_physical + gate * cp

        dp_mean = self.dp_head(h)
        comm_msg = torch.tanh(self.comm_head(h))
        role_logits = self.role_head(h)

        return dp_mean, log_std, role_logits, comm_msg, torch.sigmoid(torch.zeros_like(dp_mean[:, :1])), h_new


# ── Parameter group names for selective plasticity (S1) ──
# These are the canonical module prefixes. Use explicit inclusion (not
# string-exclusion) to avoid silently missing heads like intent_head, dp_log_std.
ENCODER_PARAM_PREFIXES = (
    'self_enc.', 'target_enc.', 'pd_hist_proj.',
    'neighbor_gru.', 'neighbor_proj.', 'global_enc.',
)
HEAD_PARAM_PREFIXES = (
    'dp_head.', 'comm_head.', 'intent_head.',
    'role_head.', 'comm_proj.', 'gate.',
    'dp_log_std',  # nn.Parameter, no trailing dot
)
ATTENTION_PARAM_PREFIXES = (
    'attn.', 'attn_norm.',
)


def split_param_groups(named_params):
    """Split parameters into encoder, head, and attention groups.

    Returns:
        enc_params, head_params, attn_params
    """
    enc, head, attn = [], [], []
    for n, p in named_params:
        if n.startswith(ATTENTION_PARAM_PREFIXES):
            attn.append(p)
        elif n.startswith(HEAD_PARAM_PREFIXES):
            head.append(p)
        elif n.startswith(ENCODER_PARAM_PREFIXES):
            enc.append(p)
        else:
            # Unknown params default to encoder (conservative, low LR)
            enc.append(p)
    return enc, head, attn


class CriticNetwork(nn.Module):
    """MAPPO centralized critic network.

    Input: global state (state_dim + num_agents + comm_dim,)
    Output: scalar V(s) + optional per-target V_q(s)
    """

    def __init__(self, state_dim: int, hidden_layers: list = [256, 256],
                 num_agents: int = 4, comm_dim: int = 0, num_targets: int = 0):
        super().__init__()
        self.num_agents = num_agents
        self.comm_dim = comm_dim
        self.num_targets = num_targets
        input_dim = state_dim + num_agents + comm_dim
        # Shared trunk: full MLP
        self.shared = mlp(input_dim, hidden_layers, hidden_layers[-1])
        last_dim = hidden_layers[-1]
        # Scalar value head
        self.value_head = nn.Linear(last_dim, 1)
        # Per-target value heads (S3: target-wise critic)
        self.target_heads = None
        if num_targets > 0:
            self.target_heads = nn.ModuleList([
                nn.Linear(last_dim, 1) for _ in range(num_targets)
            ])
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor):
        """Returns scalar V(s) for backward compatibility."""
        h = self.shared(state)
        return self.value_head(h).squeeze(-1)

    def forward_with_targets(self, state: torch.Tensor):
        """Returns (scalar_v, per_target_v) for S3 diagnostics."""
        h = self.shared(state)
        scalar_v = self.value_head(h).squeeze(-1)
        target_v = None
        if self.target_heads is not None:
            target_v = torch.stack([head(h).squeeze(-1) for head in self.target_heads], dim=-1)
        return scalar_v, target_v


class GATEncoder(nn.Module):
    """Optional Graph Attention Network for neighbor message aggregation.

    Used when K is large (enabled for K >= 6 in the document).
    Phase 1 (K=4): not activated by default.
    """

    def __init__(self, node_dim: int, hidden_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        # Project input to hidden_dim, then use consistent embed_dim throughout
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        self.attn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    batch_first=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, node_features: torch.Tensor,
                adj_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            node_features: (batch, K, node_dim)
            adj_mask: (K, K) adjacency mask (optional), True = masked (ignore)

        Returns:
            aggregated: (batch, K, hidden_dim)
        """
        x = self.input_proj(node_features)

        for attn, norm in zip(self.attn_layers, self.norms):
            x2, _ = attn(
                x, x, x,
                attn_mask=adj_mask,
                need_weights=False,
            )
            x = norm(x + x2)  # residual + layer-norm

        return x
