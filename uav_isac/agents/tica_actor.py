"""TICA Actor: Temporal–Inter-Agent Factorized Attention.

Architecture:
  FrameEncoder → Causal Temporal Self-Attention (L frames)
    ├→ Target Cross-Attention (K/V: per-target tokens)
    └→ Agent Cross-Attention (K/V: per-neighbor tokens)
          → Residual + LayerNorm → Output Heads

Fixed window (L frames), no recurrent hidden state.
D1-compatible residual adapter with zero-init projection.
"""
import math
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════
# Positional Encoding
# ═══════════════════════════════════════════════════════════════════

class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding for temporal dimension."""
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D) with positional encoding added."""
        return x + self.pe[:, :x.shape[1], :]


# ═══════════════════════════════════════════════════════════════════
# Frame Encoder
# ═══════════════════════════════════════════════════════════════════

class FrameEncoder(nn.Module):
    """Encode per-frame per-UAV observation into entity tokens.

    Uses ObservationSlices for correct parsing (P0-0 fix: beliefs block
    comes BEFORE geometry block, not interleaved per-target).
    """
    def __init__(self, obs_dim: int, K: int, Q: int, D: int = 128,
                 use_p0: bool = False, use_rel_features: bool = True):
        super().__init__()
        self.K, self.Q = K, Q
        from uav_isac.environment.observation_slices import ObservationSlices
        self.slices = ObservationSlices.from_config(
            K=K, Q=Q, use_p0=use_p0, use_rel_features=use_rel_features)

        # Self-state: pos(3)+vel(3)+battery(1)+role(1)+physics(3) = 11
        self.self_enc = nn.Sequential(
            nn.Linear(11, D), nn.ReLU(), nn.Linear(D, D))

        # Target: belief(9) + geometry(8) = 17 → 18 with coverage
        target_in = 9 + (8 if use_rel_features else 0) + (1 if use_p0 else 0)
        self.target_enc = nn.Sequential(
            nn.Linear(target_in, D), nn.ReLU(), nn.Linear(D, D))
        self.pd_hist_proj = nn.Linear(1, D)

        # Neighbor — always 9 dims after padding
        self.neighbor_enc = nn.Sequential(
            nn.Linear(9, D), nn.ReLU(), nn.Linear(D, D))

        # Global: always 2 dims (zero-filled when P0 off)
        self.global_enc = nn.Linear(2, D)

        # Communication: aggregated (16-dim) for self_token;
        # per-neighbor individual comm would be added here when observation
        # builder provides per-neighbor messages (not just aggregated).
        self.comm_proj = nn.Linear(16, D)
        self.neighbor_comm_proj = nn.Linear(16, D)  # per-neighbor comm → token

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs: torch.Tensor):
        """Encode observation into entity tokens using correct block parsing."""
        B = obs.shape[0]
        K, Q = self.K, self.Q
        sl = self.slices

        # Self state: pos(3)+vel(3)+battery(1)+role(1) = 8
        self_raw = obs[:, sl.self_start:sl.self_start + sl.self_len]

        # Physics
        phys = obs[:, sl.physics_start:sl.physics_start + sl.physics_len]
        self_state = torch.cat([self_raw, phys], dim=-1)  # (B, 11)

        # Beliefs block (Q × 9) — ALL targets, then geometry block (Q × 8)
        beliefs = obs[:, sl.belief_start:sl.belief_start + Q * sl.belief_per_target]
        beliefs = beliefs.reshape(B, Q, sl.belief_per_target)  # (B, Q, 9)

        if sl.has_rel_features and sl.geom_per_target > 0:
            geometry = obs[:, sl.geom_start:sl.geom_start + Q * sl.geom_per_target]
            geometry = geometry.reshape(B, Q, sl.geom_per_target)  # (B, Q, 8)
            targets = torch.cat([beliefs, geometry], dim=-1)  # (B, Q, 17)
        else:
            targets = beliefs  # (B, Q, 9)

        # Coverage (P0)
        if sl.has_p0 and sl.coverage_len > 0:
            cov = obs[:, sl.coverage_start:sl.coverage_start + sl.coverage_len]
            targets = torch.cat([targets, cov.unsqueeze(-1)], dim=-1)

        # Neighbors block — always pad to 9 dims before encoding
        n_dim = sl.neighbor_per_agent
        n_raw = obs[:, sl.neighbor_start:sl.neighbor_start + (K-1) * n_dim]
        neighbors = n_raw.reshape(B, K - 1, n_dim)  # (B, K-1, n_dim)
        if n_dim == 8:
            neighbors = torch.cat([
                neighbors[:, :, :7],
                torch.zeros(B, K-1, 1, device=obs.device),
                neighbors[:, :, 7:]
            ], dim=-1)  # → (B, K-1, 9)

        # Global (P0 only)
        if sl.has_p0 and sl.global_len > 0:
            global_feat = obs[:, sl.global_start:sl.global_start + sl.global_len]
        else:
            global_feat = torch.zeros(B, 2, device=obs.device)

        # PD_hist
        pd_hist = obs[:, sl.pd_hist_start:sl.pd_hist_start + sl.pd_hist_len]

        # Comm
        comm_agg = obs[:, sl.comm_start:sl.comm_start + sl.comm_len]

        # Encode
        se = self.self_enc(self_state)  # (B, D)
        te = self.target_enc(targets)   # (B, Q, D)
        ne = self.neighbor_enc(neighbors)  # (B, K-1, D)
        ge = self.global_enc(global_feat)  # (B, D)
        cp = self.comm_proj(comm_agg)  # (B, D)

        # Add PD_hist modulation to target tokens
        pd_feat = self.pd_hist_proj(pd_hist.unsqueeze(-1))  # (B, Q, D)
        te = te + pd_feat

        # Combine self with global and comm
        self_token = se + ge + cp  # (B, D)

        return self_token, te, ne


# ═══════════════════════════════════════════════════════════════════
# Temporal Self-Attention
# ═══════════════════════════════════════════════════════════════════

class TemporalSelfAttention(nn.Module):
    """Causal self-attention over L-frame window of self tokens."""
    def __init__(self, D: int = 128, num_layers: int = 2, num_heads: int = 4,
                 max_len: int = 32):
        super().__init__()
        self.pos_enc = LearnedPositionalEncoding(max_len, D)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=num_heads, dim_feedforward=D*4,
            dropout=0.0, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(D)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Causal temporal self-attention.

        Window is right-aligned: position -1 always holds current frame.
        Padding positions (mask=False) are ignored via src_key_padding_mask.

        Args:
            x: (B, T, D) sequence of self tokens, right-aligned
            mask: (B, T) boolean mask, True = valid frame.
                  mask[:, -1] MUST be True.

        Returns:
            (B, D) attended token for position -1 (current frame)
        """
        B, T, D = x.shape
        x = self.pos_enc(x)

        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)

        if mask is not None:
            padding_mask = ~mask  # True = ignore
            x_out = self.encoder(
                x, mask=causal_mask,
                src_key_padding_mask=padding_mask,
                is_causal=True,
            )
        else:
            x_out = self.encoder(x, mask=causal_mask, is_causal=True)

        # Current frame always at position -1 (right-aligned)
        return self.norm(x_out[:, -1, :])


# ═══════════════════════════════════════════════════════════════════
# Target Cross-Attention
# ═══════════════════════════════════════════════════════════════════

class TargetCrossAttention(nn.Module):
    """Cross-attention from temporal self token to target tokens."""
    def __init__(self, D: int = 128, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(D, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(D)

    def forward(self, query: torch.Tensor, target_tokens: torch.Tensor):
        """Cross-attention to target tokens.

        Args:
            query: (B, D) temporal self token
            target_tokens: (B, Q, D) per-target encodings

        Returns:
            (B, D) target-attended token
        """
        q = query.unsqueeze(1)  # (B, 1, D)
        ctx, _ = self.attn(q, target_tokens, target_tokens)
        return self.norm(query + ctx.squeeze(1))


# ═══════════════════════════════════════════════════════════════════
# Agent Cross-Attention
# ═══════════════════════════════════════════════════════════════════

class AgentCrossAttention(nn.Module):
    """Cross-attention from combined token to neighbor tokens."""
    def __init__(self, D: int = 128, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(D, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(D)

    def forward(self, query: torch.Tensor, neighbor_tokens: torch.Tensor):
        """Cross-attention to neighbor tokens.

        Args:
            query: (B, D) combined self+target token
            neighbor_tokens: (B, K-1, D) per-neighbor encodings

        Returns:
            (B, D) agent-attended token
        """
        q = query.unsqueeze(1)  # (B, 1, D)
        ctx, _ = self.attn(q, neighbor_tokens, neighbor_tokens)
        return self.norm(query + ctx.squeeze(1))


# ═══════════════════════════════════════════════════════════════════
# TICA Actor
# ═══════════════════════════════════════════════════════════════════

class TICAActor(nn.Module):
    """Temporal–Inter-Agent Factorized Attention Actor.

    Fixed window (L frames), no recurrent hidden state.
    Compatible with D1 warm-start via zero-init residual adapter.

    Args:
        obs_dim: flat observation dimension
        K: number of UAVs
        Q: number of targets
        D: entity/token dimension
        L: temporal window length (frames)
        max_dp: maximum displacement
    """
    def __init__(self, obs_dim: int, K: int = 4, Q: int = 4,
                 D: int = 128, L: int = 16, max_dp: float = 2.5,
                 use_p0: bool = False, use_rel_features: bool = True):
        super().__init__()
        self.K, self.Q, self.D, self.L = K, Q, D, L
        self.max_dp = max_dp

        # Frame encoder (correct block-based parsing)
        self.frame_encoder = FrameEncoder(obs_dim, K, Q, D, use_p0, use_rel_features)

        # Temporal self-attention
        self.temporal_attn = TemporalSelfAttention(D, num_layers=2, num_heads=4, max_len=L)

        # Target cross-attention
        self.target_attn = TargetCrossAttention(D, num_heads=4)

        # Agent cross-attention
        self.agent_attn = AgentCrossAttention(D, num_heads=4)

        # Learnable fusion gates (init small for conservative D1 fine-tuning)
        self.alpha_T = nn.Parameter(torch.tensor(0.01))
        self.alpha_A = nn.Parameter(torch.tensor(0.01))

        # Output projection
        self.output_proj = nn.Linear(D, D)
        self.output_norm = nn.LayerNorm(D)

        # Heads (same as StructuredActorNetwork)
        self.dp_head = nn.Linear(D, 2)
        self.comm_head = nn.Linear(D, 16)
        self.intent_head = nn.Linear(16, Q)
        self.role_head = nn.Linear(D, 3)
        self.comm_proj_head = nn.Linear(16, D)
        self.gate = nn.Linear(D + D, 1)
        self.dp_log_std = nn.Parameter(torch.zeros(2))

        # D1 residual adapter (zero-init)
        self.d1_proj = nn.Linear(D, D)
        nn.init.zeros_(self.d1_proj.weight)
        nn.init.zeros_(self.d1_proj.bias)

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and 'd1_proj' not in name:
                g = 0.01 if 'head' in name else np.sqrt(2)
                nn.init.orthogonal_(m.weight, gain=g)
                nn.init.constant_(m.bias, 0.0)

    def encode_features(self, obs: torch.Tensor,
                         window_mask: torch.Tensor = None) -> torch.Tensor:
        """Encode observation window into feature vector (no output heads).

        Args:
            obs: (B, obs_dim) or (B, L, obs_dim)
            window_mask: (B, L) boolean, True = valid frame

        Returns:
            h: (B, D) TICA feature vector
        """
        if obs.dim() == 2:
            B = obs.shape[0]
            obs_window = obs.unsqueeze(1)
            T = 1
            mask = None
        else:
            B, T, _ = obs.shape
            obs_window = obs
            mask = window_mask

        all_self_tokens = []
        all_target_tokens = []
        all_neighbor_tokens = []
        for t in range(T):
            st, tt, nt = self.frame_encoder(obs_window[:, t, :])
            target_pool = tt.mean(dim=1)
            frame_token = st + target_pool
            all_self_tokens.append(frame_token)
            all_target_tokens.append(tt)
            all_neighbor_tokens.append(nt)

        self_seq = torch.stack(all_self_tokens, dim=1)
        z_time = self.temporal_attn(self_seq, mask)
        target_tokens = all_target_tokens[-1]
        delta_target = self.target_attn(z_time, target_tokens) - z_time
        neighbor_tokens = all_neighbor_tokens[-1]
        delta_agent = self.agent_attn(z_time, neighbor_tokens) - z_time

        h = z_time + self.alpha_T * delta_target + self.alpha_A * delta_agent
        return self.output_norm(self.output_proj(h))

    def forward(self, obs: torch.Tensor,
                h_prev: torch.Tensor = None,
                detach_h_new: bool = True,
                window_mask: torch.Tensor = None):
        """Forward pass — single-frame or window input."""
        h = self.encode_features(obs, window_mask=window_mask)
        dp_mean = self.dp_head(h)
        LOG_STD_MIN, LOG_STD_MAX = -1.0, 1.0
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (torch.tanh(self.dp_log_std) + 1.0)
        role_logits = self.role_head(h)
        comm_msg = torch.tanh(self.comm_head(h))
        pd_pred = torch.sigmoid(torch.zeros_like(dp_mean[:, :1]))
        return dp_mean, log_std, role_logits, comm_msg, pd_pred, None

    def forward_with_window(self, obs_window: torch.Tensor):
        """Forward with a pre-built observation window.

        Args:
            obs_window: (B, L, obs_dim) batch of observation windows

        Returns:
            Same as forward()
        """
        return self.forward(obs_window)

    def load_d1_adapter(self, d1_state_dict: dict, strict: bool = False):
        """Load D1's frame_encoder weights (compatible subset).

        The TICA FrameEncoder has the same parsing logic as
        StructuredActorNetwork's _parse_obs + entity encoders.
        We load matching keys and zero-init the rest.
        """
        tica_state = self.state_dict()
        # Map D1 keys to TICA frame_encoder keys where they match
        d1_prefix_map = {
            'self_enc.': 'frame_encoder.self_enc.',
            'target_enc.': 'frame_encoder.target_enc.',
            'pd_hist_proj.': 'frame_encoder.pd_hist_proj.',
            'neighbor_gru.': None,  # No GRU in TICA
            'neighbor_proj.': 'frame_encoder.neighbor_enc.',  # Approximate
            'global_enc.': 'frame_encoder.global_enc.',
            'comm_proj.': 'frame_encoder.comm_proj.',
            'attn.': None,  # Cross-attn is different
            'attn_norm.': None,
            'dp_head.': 'dp_head.',
            'comm_head.': 'comm_head.',
            'intent_head.': 'intent_head.',
            'role_head.': 'role_head.',
            'dp_log_std': 'dp_log_std',
            'gate.': 'gate.',
            'comm_proj_head.': 'comm_proj_head.',  # Note: D1 has 'comm_proj', TICA has 'comm_proj_head' + 'frame_encoder.comm_proj'
        }

        loaded = 0
        for d1_key, d1_val in d1_state_dict.items():
            for prefix, tica_prefix in d1_prefix_map.items():
                if d1_key.startswith(prefix) and tica_prefix is not None:
                    tica_key = d1_key.replace(prefix, tica_prefix, 1)
                    if tica_key in tica_state:
                        if tica_state[tica_key].shape == d1_val.shape:
                            tica_state[tica_key] = d1_val.clone()
                            loaded += 1
                    break

        self.load_state_dict(tica_state)
        print(f'[TICA] loaded {loaded} D1 parameters, zero-init adapter')
        return loaded


# ═══════════════════════════════════════════════════════════════════
# D1-TICA Residual Adapter
# ═══════════════════════════════════════════════════════════════════

class D1TICAResidualActor(nn.Module):
    """Exact-preserving TICA adapter: freeze D1 base, zero-init delta heads.

    π_0(a|o_window, h_prev) = π_D1(a|o_last, h_prev)  ← exact at init

    Architecture:
      D1 base (frozen) → base_dp, base_log_std, base_role, base_comm
      TICA branch (trainable) → h_tica
      delta_dp (zero-init) → residual added to base_dp
      delta_role (zero-init) → residual added to base_role
    """

    def __init__(self, base_actor: nn.Module, tica_actor: TICAActor):
        super().__init__()
        self.base_actor = base_actor
        self.tica_actor = tica_actor

        # Freeze D1 base
        for p in self.base_actor.parameters():
            p.requires_grad_(False)

        D = tica_actor.D
        self.delta_dp = nn.Linear(D, 2)
        self.delta_role = nn.Linear(D, 3)

        # Zero-init: residual = 0 at initialization
        nn.init.zeros_(self.delta_dp.weight)
        nn.init.zeros_(self.delta_dp.bias)
        nn.init.zeros_(self.delta_role.weight)
        nn.init.zeros_(self.delta_role.bias)

    # ── Compat properties for MAPPO trainer ──
    @property
    def neighbor_gru(self):
        """Expose D1 base's GRU for trainer's gru_hidden_dim detection."""
        return self.base_actor.neighbor_gru

    @property
    def dp_log_std(self):
        """Expose D1 base's log_std for BC/KL anchor."""
        return self.base_actor.dp_log_std

    @property
    def D(self):
        return self.tica_actor.D

    def get_recurrent_state_dim(self) -> int:
        """Return GRU hidden dim for buffer allocation."""
        return self.base_actor.neighbor_gru.hidden_size

    def parameter_groups(self) -> dict:
        """Return explicit parameter groups for per-module LR.

        Trainer should prefer this over name-based split_param_groups().
        """
        tica = self.tica_actor
        return {
            'encoder': list(tica.frame_encoder.parameters()),
            'attention': (
                list(tica.temporal_attn.parameters())
                + list(tica.target_attn.parameters())
                + list(tica.agent_attn.parameters())
            ),
            'head': (
                list(self.delta_dp.parameters())
                + list(self.delta_role.parameters())
            ),
        }

    def trainable_parameters(self):
        """Only TICA feature branch + delta heads. Excludes unused TICA output heads."""
        for p in self.tica_actor.frame_encoder.parameters():
            yield p
        for p in self.tica_actor.temporal_attn.parameters():
            yield p
        for p in self.tica_actor.target_attn.parameters():
            yield p
        for p in self.tica_actor.agent_attn.parameters():
            yield p
        for p in self.tica_actor.output_proj.parameters():
            yield p
        for p in self.tica_actor.output_norm.parameters():
            yield p
        yield self.tica_actor.alpha_T
        yield self.tica_actor.alpha_A
        for p in self.delta_dp.parameters():
            yield p
        for p in self.delta_role.parameters():
            yield p

    def forward(self, obs: torch.Tensor,
                h_prev: torch.Tensor = None,
                detach_h_new: bool = True,
                window_mask: torch.Tensor = None):
        """Forward with D1 base + TICA residual.

        Args:
            obs: (B, obs_dim) single frame or (B, L, obs_dim) window.
                 D1 base uses only the LAST frame.
            h_prev: GRU hidden state for D1 base.
            detach_h_new: passed to D1 base.

        Returns:
            dp_mean, log_std, role_logits, comm_msg, pd_pred, h_new
        """
        # D1 base: frozen forward on last observation frame
        if obs.dim() == 3:
            obs_last = obs[:, -1, :]  # (B, obs_dim)
        else:
            obs_last = obs

        with torch.no_grad():
            base_dp, base_log_std, base_role, base_comm, base_pd, h_new = \
                self.base_actor(obs_last, h_prev=h_prev,
                                detach_h_new=detach_h_new)

        # TICA branch: encode full window (or single frame) with mask
        h_tica = self.tica_actor.encode_features(obs, window_mask=window_mask)

        # Residual heads (zero-init → exact D1 at initialization)
        dp_mean = base_dp + self.delta_dp(h_tica)
        role_logits = base_role + self.delta_role(h_tica)

        # Keep D1 comm, P_D, log_std unchanged (comm-off)
        return dp_mean, base_log_std, role_logits, base_comm, base_pd, h_new
