"""Reward computation: team reward + marginal contribution shaping.

r_k^shaped = r_team + η_mc * [ΔU_k - (1/K) * Σ_l ΔU_l]

where:
  r_team = Σ_q ω_q * U_q(D_q^*) - λ_report * total_bits - total_penalty
  ΔU_k = marginal contribution of UAV k (delete-approximation)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from uav_isac.utils.types import P0Solution, DeflectionEntry
from uav_isac.physical.detection import compute_weighted_utility


class RewardComputer:
    """Computes team reward and shaped individual rewards."""

    def __init__(
        self,
        omega_q: np.ndarray,         # (Q,) target priorities
        P_FA: float = 0.001,
        lambda_report: float = 0.001, # communication cost weight
        eta_mc: float = 0.5,          # marginal contribution shaping coefficient
        alpha_pd: float = 0.0,        # direct P_D weight
        lambda_tail: float = 0.0,     # bottom-3 bonus weight
        lambda_tail_warmup: int = 30, # episodes before tail bonus activates
    ):
        """
        Args:
            omega_q: Target priority weights
            P_FA: False alarm probability
            lambda_report: Weight for communication cost in reward
            eta_mc: Marginal contribution shaping coefficient
            alpha_pd: Weight for direct P_D reward term (0=off)
        """
        self.omega_q = np.asarray(omega_q, dtype=np.float64)
        self.P_FA = P_FA
        self.lambda_report = lambda_report
        self.eta_mc = eta_mc
        self.alpha_pd = alpha_pd
        self.lambda_tail = lambda_tail
        self.lambda_tail_warmup = lambda_tail_warmup

    def compute_team_utility_from_deflection(
        self, D_q: np.ndarray, total_bits: float = 0.0
    ) -> float:
        """Task utility from per-target deflection (no P0 involved). Used for
        fixed-assignment difference reward: same deflection→utility mapping as
        team reward, minus the bits/penalty terms that don't change per frame."""
        utility = compute_weighted_utility(D_q, self.P_FA, self.omega_q)
        return float(utility)

    def compute_team_reward(
        self,
        D_q_star: np.ndarray,      # (Q,) cumulative Deflection
        total_bits: float,          # total soft info bits reported
        constraint_penalty: float,  # total constraint violation penalty
        P_D_q: np.ndarray = None,   # (Q,) direct detection probs (optional, for alpha_pd>0)
    ) -> float:
        """Compute team-level reward.

        r_team = (1-α_pd) * weighted_utility + α_pd * mean_P_D - lambda * bits - penalty

        Args:
            D_q_star: Per-target cumulative Deflection
            total_bits: Total bits reported this frame
            constraint_penalty: Total constraint violation penalty
            P_D_q: Per-target detection probabilities (for direct P_D term)

        Returns:
            Team reward (scalar)
        """
        utility = compute_weighted_utility(D_q_star, self.P_FA, self.omega_q)
        if self.alpha_pd > 0 and P_D_q is not None:
            pd_reward = float(np.dot(self.omega_q, P_D_q))
            # Bottom-3 bonus: gentle tail protection
            if self.lambda_tail > 0:
                sorted_pd = np.sort(P_D_q)
                bottom3 = float(np.mean(sorted_pd[:3]))
                pd_reward = pd_reward + self.lambda_tail * bottom3
            team_rew = (1.0 - self.alpha_pd) * utility + self.alpha_pd * pd_reward
        else:
            team_rew = float(utility)
        return float(team_rew - self.lambda_report * total_bits - constraint_penalty)

    def compute_marginal_contributions(
        self,
        uav_ids: List[int],         # list of UAV IDs
        selected_set: List[Tuple[int, int, int]],  # (tx, rx, target) selected
        deflection_entries: List[DeflectionEntry],
        Q: int,
    ) -> Dict[int, float]:
        """Compute marginal contribution of each UAV (delete-approximation).

        ΔU_k = U(D^*) - U(D^* without k's contributions)
        where D^* without k's contributions removes all edges where k is
        either tx or rx.

        Args:
            uav_ids: List of UAV indices
            selected_set: Selected assignments from P0
            deflection_entries: All valid DeflectionEntry objects
            Q: Number of targets

        Returns:
            Dict mapping uav_id → marginal utility contribution
        """
        # Compute full utility
        D_full = np.zeros(Q, dtype=np.float64)
        for (i, j, q) in selected_set:
            # Find the matching DeflectionEntry
            for e in deflection_entries:
                if e.i == i and e.j == j and e.q == q:
                    D_full[q] += e.d_eff
                    break

        U_full = compute_weighted_utility(D_full, self.P_FA, self.omega_q)

        # Compute utility without each UAV's contributions
        marginal_contribs = {}
        for k in uav_ids:
            D_without_k = np.zeros(Q, dtype=np.float64)
            for (i, j, q) in selected_set:
                if i == k or j == k:
                    continue  # exclude edges involving UAV k
                for e in deflection_entries:
                    if e.i == i and e.j == j and e.q == q:
                        D_without_k[q] += e.d_eff
                        break

            U_without_k = compute_weighted_utility(D_without_k, self.P_FA, self.omega_q)
            marginal_contribs[k] = U_full - U_without_k

        return marginal_contribs

    def compute_shaped_rewards(
        self,
        team_reward: float,
        marginal_contribs: Dict[int, float],
        per_agent_sensing: Optional[Dict[int, float]] = None,
        eta_sense: float = 0.1,
        diff_rewards: Optional[Dict[int, float]] = None,
        team_weight: float = 0.7,
        diff_weight: float = 0.3,
    ) -> Dict[int, float]:
        """Compute shaped individual rewards.

        r_k = r_team + η_mc * (ΔU_k - mean(ΔU)) + η_sense * sensing_k

        The marginal contribution shaping sums to zero.
        The sensing term gives each UAV a LOCAL reward for its own
        sensing quality (d_eff contributed, proximity to targets).

        Args:
            team_reward: Team reward (shared)
            marginal_contribs: Dict of uav_id → ΔU_k
            per_agent_sensing: Dict of uav_id → local sensing quality
            eta_sense: weight for per-agent sensing term

        Returns:
            Dict of uav_id → shaped reward
        """
        K = len(marginal_contribs)
        if K == 0:
            return {}

        values = list(marginal_contribs.values())
        mean_delta = np.mean(values)

        shaped = {}
        for k, delta_u in marginal_contribs.items():
            # Team reward (with marginal shaping if enabled)
            r_k = team_reward
            if self.eta_mc > 0 and marginal_contribs:
                r_k += self.eta_mc * (delta_u - mean_delta)
            # Per-agent sensing quality
            if per_agent_sensing is not None and k in per_agent_sensing:
                r_k += eta_sense * per_agent_sensing[k]
            # Fixed-assignment difference reward
            if diff_rewards is not None and k in diff_rewards:
                r_k = team_weight * r_k + diff_weight * diff_rewards[k]
            shaped[k] = float(r_k)

        return shaped
