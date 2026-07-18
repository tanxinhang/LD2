"""Multi-Head Neighbor Cross-Attention for Belief Fusion.

Per-target attention: node i queries which neighbors to trust for target q.
Outputs learned fusion weights for conservative belief fusion (CI) and
uncertainty-aware P0 score correction.

Architecture:
  Q_{i,q} = f_q(local_belief, cov, AoI, PD_hist)_q
  K/V_{j,q} = f_n(neighbor_belief, cov, AoI, PD, link_quality)_{j,q}
  α_{ijq} = MHA(Q, K, V)
  ω_{ijq} = softmax(α_{ijq} · χ_{ij})  -- fusion weights

This does NOT replace the actor — it improves belief quality for P0 scheduling.
"""
import math
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class PerTargetBeliefEncoder(nn.Module):
    """Encode one target's belief state into a query/key token."""

    def __init__(self, D: int = 64):
        super().__init__()
        # belief mean (4) + cov diag (4) + AoI (1) + PD_hist (1) + link (1) = 11
        self.encoder = nn.Sequential(
            nn.Linear(11, D), nn.ReLU(), nn.Linear(D, D))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, belief_mean, belief_cov, aoi, pd_hist,
                link_quality=None):
        """Encode per-target belief into token.

        Args:
            belief_mean: (B, 4) normalized belief mean
            belief_cov: (B, 4) normalized covariance diagonal
            aoi: (B, 1) age of information
            pd_hist: (B, 1) previous detection probability
            link_quality: (B, 1) optional link reliability [0,1]

        Returns:
            token: (B, D)
        """
        feat = torch.cat([belief_mean, belief_cov, aoi, pd_hist], dim=-1)
        if link_quality is not None:
            feat = torch.cat([feat, link_quality], dim=-1)
        else:
            feat = torch.cat([feat, torch.ones(feat.shape[0], 1, device=feat.device)], dim=-1)
        return self.encoder(feat)


class MultiHeadNeighborAttention(nn.Module):
    """Multi-head cross-attention: one query, multiple neighbor keys/values."""

    def __init__(self, D: int = 64, num_heads: int = 4,
                 uniform_init: bool = True):
        super().__init__()
        assert D % num_heads == 0
        self.D = D
        self.num_heads = num_heads
        self.head_dim = D // num_heads

        self.W_Q = nn.Linear(D, D, bias=False)
        self.W_K = nn.Linear(D, D, bias=False)
        self.W_V = nn.Linear(D, D, bias=False)
        self.W_O = nn.Linear(D, D, bias=False)

        if uniform_init:
            self._init_uniform()
        else:
            self._init_weights()

    def _init_weights(self):
        for m in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))

    def _init_uniform(self):
        """Zero-init Q/K → uniform attention weights at initialization.
        V and O kept small-random so context is non-zero but untrained.
        This ensures attention fusion = uniform mean at init (safe fallback)."""
        nn.init.zeros_(self.W_Q.weight)
        nn.init.zeros_(self.W_K.weight)
        nn.init.orthogonal_(self.W_V.weight, gain=0.01)
        nn.init.orthogonal_(self.W_O.weight, gain=0.01)

    def forward(self, query: torch.Tensor,
                neighbor_tokens: torch.Tensor,
                neighbor_mask: Optional[torch.Tensor] = None):
        """Multi-head cross-attention from query to neighbor tokens.

        Args:
            query: (B, D) local node's query for this target
            neighbor_tokens: (B, N, D) per-neighbor encodings
            neighbor_mask: (B, N) True = valid neighbor

        Returns:
            context: (B, D) attended context vector
            weights: (B, num_heads, N) per-head attention weights
        """
        B, N, D = neighbor_tokens.shape
        H = self.num_heads
        d = self.head_dim

        # Project
        Q = self.W_Q(query).view(B, 1, H, d).transpose(1, 2)  # (B, H, 1, d)
        K = self.W_K(neighbor_tokens).view(B, N, H, d).transpose(1, 2)  # (B, H, N, d)
        V = self.W_V(neighbor_tokens).view(B, N, H, d).transpose(1, 2)  # (B, H, N, d)

        # Scaled dot-product attention
        scale = math.sqrt(d)
        attn_logits = (Q @ K.transpose(-2, -1)) / scale  # (B, H, 1, N)

        if neighbor_mask is not None:
            # mask: (B, N) → (B, 1, 1, N) for broadcasting with (B, H, 1, N)
            mask_expanded = neighbor_mask.unsqueeze(1).unsqueeze(2)
            attn_logits = attn_logits.masked_fill(
                ~mask_expanded, float('-inf'))

        attn_weights = torch.softmax(attn_logits, dim=-1)  # (B, H, 1, N)

        # Weighted sum
        context = (attn_weights @ V).squeeze(2)  # (B, H, d)
        context = context.transpose(1, 2).reshape(B, D)  # (B, D)
        context = self.W_O(context)

        return context, attn_weights.squeeze(2)  # (B, D), (B, H, N)


class NeighborBeliefFusion(nn.Module):
    """Per-target multi-head neighbor attention for belief fusion.

    For each target q:
      - Encode local belief state → query
      - Encode each neighbor's belief → key/value tokens
      - Multi-head cross-attention → fusion weights
      - Conservative belief fusion (Covariance Intersection)
    """

    def __init__(self, Q: int, D: int = 64, num_heads: int = 4,
                 uniform_init: bool = True):
        super().__init__()
        self.Q = Q
        self.query_encoder = PerTargetBeliefEncoder(D)
        self.neighbor_encoder = PerTargetBeliefEncoder(D)
        self.attention = MultiHeadNeighborAttention(D, num_heads,
                                                     uniform_init=uniform_init)

    def forward(self,
                local_belief_mean,   # (B, Q, 4)
                local_belief_cov,    # (B, Q, 4)
                local_aoi,           # (B, Q, 1)
                local_pd_hist,       # (B, Q, 1)
                neighbor_belief_mean,  # (B, Q, N, 4)
                neighbor_belief_cov,   # (B, Q, N, 4)
                neighbor_aoi,          # (B, Q, N, 1)
                neighbor_pd_hist,      # (B, Q, N, 1)
                neighbor_mask=None,    # (B, N)
                link_quality=None,     # (B, N, 1)
                ):
        """Compute per-target fusion weights via multi-head attention.

        Args:
            local_belief_mean: (B, Q, 4) local belief means
            local_belief_cov: (B, Q, 4) local covariance diagonals
            local_aoi: (B, Q, 1) local AoI
            local_pd_hist: (B, Q, 1) local previous P_D
            neighbor_belief_mean: (B, Q, N, 4) per-neighbor belief means
            neighbor_belief_cov: (B, Q, N, 4) per-neighbor covariance diagonals
            neighbor_aoi: (B, Q, N, 1) per-neighbor AoI
            neighbor_pd_hist: (B, Q, N, 1) per-neighbor previous P_D
            neighbor_mask: (B, N) valid neighbor mask
            link_quality: (B, N, 1) optional link reliability

        Returns:
            fusion_weights: (B, Q, N) ω_{ijq} ∈ [0,1], sum_j ω = 1
            attention_weights: (B, Q, num_heads, N) per-head α
            fused_context: (B, Q, D) attention context per target
        """
        B, Q, N = neighbor_belief_mean.shape[:3]
        device = local_belief_mean.device

        all_fusion_weights = []
        all_attn_weights = []
        all_contexts = []

        for q in range(Q):
            # Encode local query
            q_token = self.query_encoder(
                local_belief_mean[:, q, :],
                local_belief_cov[:, q, :],
                local_aoi[:, q, :],
                local_pd_hist[:, q, :],
            )  # (B, D)

            # Encode neighbor tokens for target q
            n_tokens = []
            for j in range(N):
                lq = link_quality[:, j, :] if link_quality is not None else None
                n_token = self.neighbor_encoder(
                    neighbor_belief_mean[:, q, j, :],
                    neighbor_belief_cov[:, q, j, :],
                    neighbor_aoi[:, q, j, :],
                    neighbor_pd_hist[:, q, j, :],
                    link_quality=lq,
                )  # (B, D)
                n_tokens.append(n_token)
            n_tokens = torch.stack(n_tokens, dim=1)  # (B, N, D)

            # Multi-head cross-attention
            context, attn_w = self.attention(q_token, n_tokens, neighbor_mask)
            # attn_w: (B, H, N)

            # Mean over heads → fusion weights
            fusion_w = attn_w.mean(dim=1)  # (B, N)

            # Apply link quality gate
            if link_quality is not None:
                lq_flat = link_quality.squeeze(-1)  # (B, N)
                fusion_w = fusion_w * lq_flat

            # Mask invalid neighbors
            if neighbor_mask is not None:
                fusion_w = fusion_w.masked_fill(~neighbor_mask, 0.0)

            # Normalize
            fusion_w = fusion_w / (fusion_w.sum(dim=1, keepdim=True) + 1e-8)

            all_fusion_weights.append(fusion_w)
            all_attn_weights.append(attn_w)
            all_contexts.append(context)

        return (
            torch.stack(all_fusion_weights, dim=1),  # (B, Q, N)
            torch.stack(all_attn_weights, dim=1),    # (B, Q, H, N)
            torch.stack(all_contexts, dim=1),         # (B, Q, D)
        )

    def covariance_intersection_fusion(
        self,
        local_belief_mean,      # (B, Q, 4)
        local_belief_cov,       # (B, Q, 4)
        neighbor_belief_mean,   # (B, Q, N, 4)
        neighbor_belief_cov,    # (B, Q, N, 4)
        fusion_weights,         # (B, Q, N)
        local_weight=0.3,       # minimum weight for local belief
    ):
        """Conservative belief fusion via Covariance Intersection.

        P_f^{-1} = Σ_j ω_j P_j^{-1} + ω_local P_local^{-1}
        x_f = P_f (Σ_j ω_j P_j^{-1} x_j + ω_local P_local^{-1} x_local)

        Args:
            local_belief_mean: (B, Q, 4)
            local_belief_cov: (B, Q, 4) — diagonal only
            neighbor_belief_mean: (B, Q, N, 4)
            neighbor_belief_cov: (B, Q, N, 4) — diagonal only
            fusion_weights: (B, Q, N) — ω_{ijq} from attention
            local_weight: minimum weight preserved for local belief

        Returns:
            fused_mean: (B, Q, 4)
            fused_cov: (B, Q, 4)
        """
        B, Q, N = neighbor_belief_mean.shape[:3]
        device = local_belief_mean.device
        eps = 1e-8

        # Scale neighbor weights to leave room for local
        scaled_weights = fusion_weights * (1.0 - local_weight)  # (B, Q, N)
        all_weights = torch.cat([
            torch.full((B, Q, 1), local_weight, device=device),
            scaled_weights,
        ], dim=-1)  # (B, Q, 1+N)

        # Stack all beliefs: [local, n0, n1, ...]
        all_means = torch.cat([
            local_belief_mean.unsqueeze(2),
            neighbor_belief_mean,
        ], dim=2)  # (B, Q, 1+N, 4)

        all_covs = torch.cat([
            local_belief_cov.unsqueeze(2),
            neighbor_belief_cov,
        ], dim=2)  # (B, Q, 1+N, 4)

        # Covariance Intersection: P_f^{-1} = Σ ω_k P_k^{-1}
        precisions = all_weights.unsqueeze(-1) / (all_covs + eps)  # (B, Q, 1+N, 4)
        fused_precision = precisions.sum(dim=2)  # (B, Q, 4)
        fused_cov = 1.0 / (fused_precision + eps)

        # Weighted mean
        weighted_mean = (all_weights.unsqueeze(-1) * precisions * all_means).sum(dim=2)
        fused_mean = weighted_mean / (fused_precision + eps)

        return fused_mean, fused_cov
