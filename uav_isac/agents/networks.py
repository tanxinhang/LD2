"""Neural network architectures for actor and critic.

Actor: shared MLP [256, 256] → heads: dp_mean(2), dp_log_std(2), role_logits(3)
  (shared = obs → 256 → 256, then head: 256 → output)
Critic: MLP [256, 256] → scalar value
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
    """

    def __init__(self, obs_dim: int, K: int = 8, Q: int = 8,
                 entity_dim: int = 128, max_dp: float = 2.5):
        super().__init__()
        self.K, self.Q = K, Q
        self.max_dp = max_dp
        D = entity_dim

        # ── Entity encoders ──
        self.self_enc = nn.Sequential(
            nn.Linear(11, D), nn.ReLU(), nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D),
        )
        self.target_enc = nn.Sequential(
            nn.Linear(18, D), nn.ReLU(), nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D),
        )
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

    def _parse_obs(self, obs: torch.Tensor):
        """Parse flat obs. Returns entity tensors as sequences (B, N, W, D)."""
        B = obs.shape[0]
        obs_dim = obs.shape[1]
        single_dim = 227  # decentralized base dim

        # Detect 2-frame stacking → split and stack as sequences
        if obs_dim > single_dim + 20:
            w = obs_dim // single_dim  # number of stacked frames (2)
            frames = []
            for i in range(w):
                s, t, n, g, c = self._parse_one(obs[:, i*single_dim:(i+1)*single_dim], B)
                frames.append((s, t, n, g, c))
            # Stack along LAST sequence dim
            s_seq = torch.stack([f[0] for f in frames], dim=-1)  # (B, 11, W)
            t_seq = torch.stack([f[1] for f in frames], dim=-1)  # (B, Q, D_t, W)
            n_seq = torch.stack([f[2] for f in frames], dim=-1)  # (B, K-1, D_n, W)
            g_seq = torch.stack([f[3] for f in frames], dim=-1)  # (B, 2, W)
            return s_seq, t_seq, n_seq, g_seq, frames[-1][4], w
        else:
            s, t, n, g, c = self._parse_one(obs, B)
            # Add seq dim of 1 at the end
            return (s.unsqueeze(-1), t.unsqueeze(-1), n.unsqueeze(-1),
                    g.unsqueeze(-1), c, 1)

    def _parse_one(self, obs, B):
        """Parse single frame (227 dims) into entity tensors."""
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
        # Detect presence: if remaining dims > what's expected without P0, they're present
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
        _pd_hist = obs[:, ptr:ptr+Q]; ptr += Q  # P_D history
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

        return self_state, target_stack, neighbor_stack, global_feat, comm_agg

    def forward(self, obs: torch.Tensor, h_prev: torch.Tensor = None):
        """Forward with optional streaming GRU hidden state."""
        result = self._parse_obs(obs)
        self_s, targets, neighbors, global_f, comm_agg, n_frames = result
        B = obs.shape[0]
        D = self.self_enc[0].out_features
        LOG_STD_MIN, LOG_STD_MAX = -1.0, 1.0
        log_std = LOG_STD_MIN + 0.5*(LOG_STD_MAX-LOG_STD_MIN)*(torch.tanh(self.dp_log_std)+1.0)

        if targets is None:
            h = self.self_enc(torch.zeros(B, 11, device=obs.device))
            return self.dp_head(h), log_std, self.role_head(h), torch.tanh(self.comm_head(h)), torch.zeros(B, 1), None

        # Encode entities: use last timestep (dim=-1 is seq)
        se = self.self_enc(self_s[..., -1]).unsqueeze(1)          # (B, 1, D)
        te = self.target_enc(targets[..., -1])                     # (B, Q, D)
        ge = self.global_enc(global_f[..., -1]).unsqueeze(1)      # (B, 1, D)

        # Streaming GRU for neighbors: (B*Nn, 1, 9) with per-neighbor state
        Nn = neighbors.shape[1]
        n_feat = neighbors[..., -1]  # (B, Nn, 9) — last timestep
        n_input = n_feat.reshape(B * Nn, 1, 9)  # (B*Nn, 1, 9)
        if h_prev is not None:
            _, hn = self.neighbor_gru(n_input, h_prev)
        else:
            _, hn = self.neighbor_gru(n_input)
        ne = self.neighbor_proj(hn.squeeze(0)).reshape(B, Nn, -1)
        h_new = hn.detach()  # (1, B*Nn, D) — detach for Truncated BPTT

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
        # Per-target value heads (S3: diagnostic)
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
