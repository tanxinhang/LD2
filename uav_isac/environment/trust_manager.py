"""Trust Manager: Disagreement-gated belief fusion (Layer 2 of CG-SR).

Pure-numpy module that runs inside the env step. Manages per-neighbor
trust scores, detects belief disagreement, and enforces fusion safety
via weight bounds and quarantine.

Trust score τ_{ijq} ∈ [0,1]:
  τ = g_NIS(i) · g_NIS(j) · exp(-d_{ijq}/2) · g_age(j)

where:
  g_NIS(i) = exp(-max(0, nis_ema_i - 1))     NIS health
  d_{ijq}   = dx^T (P_i + P_j)^{-1} dx         Mahalanobis disagreement
  g_age(j)  = max(0, 1 - aoi_j / aoi_max)      age penalty

Safety mechanisms:
  - ω_max: per-neighbor CI weight cap (prevents single-node dominance)
  - local_weight_min: minimum local belief weight (always kept)
  - quarantine: neighbor isolated when NIS_fused >> NIS_local
"""

import numpy as np
from typing import Dict, Optional, Tuple


class TrustManager:
    """Per-neighbor trust scoring, gating, and quarantine."""

    def __init__(
        self,
        K: int,
        Q: int,
        disagreement_threshold: float = 6.0,
        aoi_max: float = 50.0,
        weight_max: float = 0.6,
        local_weight_min: float = 0.25,
        ema_rho: float = 0.1,
        quarantine_nis_ratio: float = 1.5,
        quarantine_duration: int = 10,
    ):
        """
        Args:
            K: Number of UAVs
            Q: Number of targets
            disagreement_threshold: d_ijq above which neighbors are suspicious
            aoi_max: AoI at which age penalty reaches 0
            weight_max: Maximum CI weight for any single neighbor
            local_weight_min: Minimum weight preserved for local belief
            ema_rho: EMA smoothing factor for trust scores
            quarantine_nis_ratio: Trigger quarantine if NIS_fused/NIS_local > this
            quarantine_duration: Frames to quarantine after trigger
        """
        self.K = K
        self.Q = Q
        self.disagreement_threshold = disagreement_threshold
        self.aoi_max = aoi_max
        self.weight_max = weight_max
        self.local_weight_min = local_weight_min
        self.ema_rho = ema_rho
        self.quarantine_nis_ratio = quarantine_nis_ratio
        self.quarantine_duration = quarantine_duration

        # Trust scores: (K, K, Q) — trust[k, j, q] = trust of k in neighbor j for target q
        # Diagonal (k==j) is unused
        self.trust_score = np.ones((K, K, Q), dtype=np.float64)

        # Per-neighbor disagreement history: (K, K, Q)
        self.disagreement = np.zeros((K, K, Q), dtype=np.float64)

        # NIS tracking for trust feedback (Layer 4)
        self.nis_fused_ema = np.ones((K, K, Q), dtype=np.float64)
        self.nis_local_ema = np.ones((K, Q), dtype=np.float64)

        # Quarantine state: (K, K, Q) — remaining frames
        self.quarantine_counter = np.zeros((K, K, Q), dtype=np.int32)

    # ── Gate computation ──

    def compute_gate_weights(
        self,
        belief_mean: np.ndarray,       # (K, Q, 4)
        belief_cov: np.ndarray,        # (K, Q, 4, 4)
        belief_aoi: np.ndarray,        # (K, Q)
        nis_ema: np.ndarray,           # (K, Q)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-neighbor trust scores and gated fusion weights.

        Args:
            belief_mean: (K, Q, 4) belief means
            belief_cov: (K, Q, 4, 4) calibrated covariance matrices
            belief_aoi: (K, Q) age of information per (k,q)
            nis_ema: (K, Q) NIS EMA from BeliefManager

        Returns:
            trust_scores: (K, K, Q) τ_{kjq} ∈ [0, 1]
            gated_weights: (K, K, Q) raw ω before normalization, 0 for quarantined
        """
        K, Q = self.K, self.Q
        trust = np.zeros((K, K, Q), dtype=np.float64)
        raw_w = np.zeros((K, K, Q), dtype=np.float64)

        for k in range(K):
            for q in range(Q):
                g_nis_k = np.exp(-max(0.0, float(nis_ema[k, q]) - 1.0))
                P_k = belief_cov[k, q]  # (4, 4)
                x_k = belief_mean[k, q]  # (4,)

                for j in range(K):
                    if j == k:
                        continue

                    # Skip quarantined neighbors
                    if self.quarantine_counter[k, j, q] > 0:
                        trust[k, j, q] = 0.0
                        raw_w[k, j, q] = 0.0
                        continue

                    # NIS health of neighbor
                    g_nis_j = np.exp(-max(0.0, float(nis_ema[j, q]) - 1.0))

                    # Pairwise Mahalanobis disagreement
                    # Use only first 4 dims [x,y,vx,vy] for compatibility
                    # with both CV (4D) and CA (6D) belief states
                    x_k_4d = x_k[:4]
                    x_j_4d = belief_mean[j, q, :4]
                    dx = x_k_4d - x_j_4d
                    P_sum_4d = (P_k[:4, :4] if P_k.shape[0] > 4 else P_k) + \
                               (belief_cov[j, q, :4, :4] if belief_cov[j, q].shape[0] > 4 else belief_cov[j, q])
                    try:
                        d_ijq = float(dx @ np.linalg.solve(P_sum_4d, dx))
                    except np.linalg.LinAlgError:
                        d_ijq = 1e6  # numerical issue → distrust

                    # Age penalty
                    aoi_j = float(belief_aoi[j, q])
                    g_age_j = max(0.0, 1.0 - aoi_j / self.aoi_max)

                    # Combined trust
                    tau = g_nis_k * g_nis_j * np.exp(-d_ijq / 2.0) * g_age_j
                    tau = float(np.clip(tau, 0.0, 1.0))

                    # EMA smoothing
                    self.trust_score[k, j, q] = (
                        (1.0 - self.ema_rho) * self.trust_score[k, j, q]
                        + self.ema_rho * tau
                    )
                    self.disagreement[k, j, q] = d_ijq

                    trust[k, j, q] = self.trust_score[k, j, q]
                    # Apply weight cap
                    raw_w[k, j, q] = min(
                        self.trust_score[k, j, q],
                        self.weight_max,
                    )

        return trust, raw_w

    def apply_gate_to_weights(
        self,
        fusion_weights: np.ndarray,   # (B, Q, N) attention weights
        trust_scores: np.ndarray,     # (B, Q, N) per-neighbor trust
    ) -> np.ndarray:
        """Gate attention fusion weights by trust scores.

        Args:
            fusion_weights: (B, Q, N) raw attention weights
            trust_scores: (B, Q, N) per-neighbor trust τ ∈ [0, 1]

        Returns:
            gated_weights: (B, Q, N) gated weights, renormalized
        """
        # Gate: ω ← ω · τ
        gated = fusion_weights * trust_scores

        # Clip to weight_max
        gated = np.clip(gated, 0.0, self.weight_max)

        # Renormalize to sum ≤ 1 - local_weight_min
        total = gated.sum(axis=-1, keepdims=True) + 1e-8
        max_total = 1.0 - self.local_weight_min
        scale = np.minimum(1.0, max_total / total)
        gated = gated * scale

        return gated

    # ── Trust feedback (Layer 4) ──

    def update_trust_from_nis(
        self,
        k: int,
        j: int,
        q: int,
        nis_fused: float,
        nis_local: float,
        d_z: int = 4,
    ) -> None:
        """Update trust score based on post-measurement NIS comparison.

        Called after Kalman update when both local and fused predictions
        were available for comparison against the actual measurement.

        Args:
            k: Local UAV index
            j: Neighbor UAV index
            q: Target index
            nis_fused: NIS of fused belief prediction vs measurement
            nis_local: NIS of local belief prediction vs measurement
            d_z: Measurement dimension
        """
        rho = self.ema_rho
        self.nis_fused_ema[k, j, q] = (
            (1.0 - rho) * self.nis_fused_ema[k, j, q]
            + rho * (nis_fused / d_z)
        )
        self.nis_local_ema[k, q] = (
            (1.0 - rho) * self.nis_local_ema[k, q]
            + rho * (nis_local / d_z)
        )

        # Trust update: exponential reward for good fused prediction
        T_new = np.exp(-nis_fused / (2.0 * d_z))
        self.trust_score[k, j, q] = (
            (1.0 - rho) * self.trust_score[k, j, q]
            + rho * T_new
        )

    def check_quarantine(
        self,
        k: int,
        j: int,
        q: int,
    ) -> bool:
        """Check if neighbor should be quarantined.

        Returns True if quarantine was just triggered.
        """
        nis_f = self.nis_fused_ema[k, j, q]
        nis_l = max(self.nis_local_ema[k, q], 1e-8)

        if nis_f > self.quarantine_nis_ratio * nis_l:
            self.quarantine_counter[k, j, q] = self.quarantine_duration
            return True
        return False

    def decay_quarantine(self) -> None:
        """Decrement all quarantine counters (call once per frame)."""
        self.quarantine_counter = np.maximum(
            0, self.quarantine_counter - 1,
        )

    # ── Diagnostics ──

    def get_trust_summary(self) -> Dict:
        """Return summary statistics for logging."""
        K, Q = self.K, self.Q
        # Only off-diagonal (j != k) entries
        mask = ~np.eye(K, dtype=bool)
        trust_off_diag = self.trust_score[mask]  # (K*(K-1), Q)

        n_quarantined = int(np.sum(self.quarantine_counter > 0))
        n_total_pairs = K * (K - 1) * Q

        return {
            'fusion_trust_mean': float(np.mean(trust_off_diag)),
            'fusion_trust_min': float(np.min(trust_off_diag)),
            'fusion_trust_max': float(np.max(trust_off_diag)),
            'fusion_disagreement_mean': float(np.mean(self.disagreement[mask])),
            'fusion_disagreement_max': float(np.max(self.disagreement[mask])),
            'quarantine_count': n_quarantined,
            'fusion_rejection_rate': n_quarantined / max(n_total_pairs, 1),
        }

    def get_trust_matrix(self, q: int) -> np.ndarray:
        """Return (K, K) trust matrix for a single target (for debugging)."""
        return self.trust_score[:, :, q].copy()

    def reset(self) -> None:
        """Reset all state to initial values."""
        self.trust_score.fill(1.0)
        self.disagreement.fill(0.0)
        self.nis_fused_ema.fill(1.0)
        self.nis_local_ema.fill(1.0)
        self.quarantine_counter.fill(0)
