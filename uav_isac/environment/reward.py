"""Reward computation: team reward + marginal contribution shaping.

r_k^shaped = r_team + η_mc * [ΔU_k - (1/K) * Σ_l ΔU_l]

where:
  r_team = Σ_q ω_q * U_q(D_q^*) - λ_report * total_bits - total_penalty
  ΔU_k = marginal contribution of UAV k (delete-approximation)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from uav_isac.utils.types import P0Solution, DeflectionEntry
from uav_isac.physical.detection import utility_from_D
from uav_isac.environment.maxmin_reward import (
    concave_saturating_deflection_utility,
    maxmin_dual_reward,
)


def _qos_summary(P_D_q: np.ndarray) -> Tuple[float, float, float]:
    """Return steady, bottom-3 and worst detection quality for one frame."""
    pd = np.asarray(P_D_q, dtype=np.float64).reshape(-1)
    if pd.size == 0:
        return 0.0, 0.0, 0.0
    ordered = np.sort(np.clip(pd, 0.0, 1.0))
    return (
        float(np.mean(ordered)),
        float(np.mean(ordered[:min(3, ordered.size)])),
        float(ordered[0]),
    )


def _deficit_progress(previous: float, current: float, floor: float) -> float:
    """Signed, normalized progress towards a QoS floor.

    Reducing a deficit is positive, increasing it is negative, and motion
    wholly above the floor is neutral.  Unlike a one-shot absolute bonus this
    term cannot be collected repeatedly by merely staying at the same quality.
    """
    floor = max(float(floor), 1e-8)
    previous_deficit = max(0.0, floor - float(previous))
    current_deficit = max(0.0, floor - float(current))
    return float(np.clip(
        (previous_deficit - current_deficit) / floor, -1.0, 1.0))


def compute_segmented_coordination_reward(
    previous_P_D_q: Optional[np.ndarray],
    current_P_D_q: np.ndarray,
    previous_uav_positions: np.ndarray,
    current_uav_positions: np.ndarray,
    target_positions: np.ndarray,
    *,
    stage: int,
    steady_floor: float = 0.80,
    weak3_floor: float = 0.70,
    worst_floor: float = 0.60,
    worst_weight: float = 1.0,
    duplicate_weight: float = 0.01,
    weak3_weight: float = 0.50,
    steady_weight: float = 0.25,
    min_move_m: float = 1e-6,
) -> Dict[str, float]:
    """Compute an auditable, stage-gated coordination reward.

    Stages are cumulative so a paired ablation can add exactly one mechanism:

    0. diagnostics only;
    1. worst-target deficit progress;
    2. plus avoidable duplicate-motion penalty;
    3. plus bottom-3 deficit progress;
    4. plus steady-mean deficit progress.

    A duplicate is penalized only when the team points multiple active motion
    vectors at one target while at least one below-worst-floor target receives
    no motion.  Intentional multi-UAV sensing is therefore not penalized once
    all weak targets have a prospective mover.
    """
    stage = int(stage)
    if stage < 0 or stage > 4:
        raise ValueError('coord_reward_stage must be between 0 and 4')

    current_pd = np.asarray(current_P_D_q, dtype=np.float64).reshape(-1)
    steady, weak3, worst = _qos_summary(current_pd)
    if previous_P_D_q is None:
        previous_steady, previous_weak3, previous_worst = steady, weak3, worst
    else:
        previous_steady, previous_weak3, previous_worst = _qos_summary(
            previous_P_D_q)

    worst_raw = _deficit_progress(previous_worst, worst, worst_floor)
    weak3_raw = _deficit_progress(previous_weak3, weak3, weak3_floor)
    steady_raw = _deficit_progress(previous_steady, steady, steady_floor)

    previous_xy = np.asarray(previous_uav_positions, dtype=np.float64)[:, :2]
    current_xy = np.asarray(current_uav_positions, dtype=np.float64)[:, :2]
    target_xy = np.asarray(target_positions, dtype=np.float64)[:, :2]
    movement = current_xy - previous_xy
    move_norm = np.linalg.norm(movement, axis=1)
    active = move_norm > max(float(min_move_m), 0.0)
    target_load = np.zeros(current_pd.size, dtype=np.int64)
    if np.any(active) and target_xy.shape[0] == current_pd.size:
        to_target = target_xy[None, :, :] - current_xy[:, None, :]
        target_dist = np.linalg.norm(to_target, axis=-1)
        cosine = np.einsum('kd,kqd->kq', movement, to_target)
        cosine /= np.maximum(move_norm[:, None] * target_dist, 1e-12)
        choices = np.argmax(cosine[active], axis=1)
        target_load = np.bincount(choices, minlength=current_pd.size)

    weak_uncovered = (current_pd < float(worst_floor)) & (target_load == 0)
    duplicate_excess = int(np.maximum(target_load - 1, 0).sum())
    avoidable_duplicates = min(duplicate_excess, int(np.sum(weak_uncovered)))
    if avoidable_duplicates > 0:
        uncovered_deficit = np.maximum(
            0.0, float(worst_floor) - current_pd[weak_uncovered])
        deficit_scale = float(np.mean(uncovered_deficit) / max(worst_floor, 1e-8))
        duplicate_raw = (
            avoidable_duplicates / max(current_xy.shape[0], 1)
        ) * deficit_scale
    else:
        duplicate_raw = 0.0

    worst_reward = float(worst_weight) * worst_raw if stage >= 1 else 0.0
    duplicate_penalty = (
        float(duplicate_weight) * duplicate_raw if stage >= 2 else 0.0)
    weak3_reward = float(weak3_weight) * weak3_raw if stage >= 3 else 0.0
    steady_reward = float(steady_weight) * steady_raw if stage >= 4 else 0.0
    total = worst_reward + weak3_reward + steady_reward - duplicate_penalty

    return {
        'coord_stage': float(stage),
        'coord_steady': steady,
        'coord_weak3': weak3,
        'coord_worst': worst,
        'coord_worst_progress_raw': worst_raw,
        'coord_weak3_progress_raw': weak3_raw,
        'coord_steady_progress_raw': steady_raw,
        'coord_avoidable_duplicate_raw': float(duplicate_raw),
        'coord_avoidable_duplicate_count': float(avoidable_duplicates),
        'coord_movement_collision': float(np.any(target_load > 1)),
        'coord_weak_uncovered_count': float(np.sum(weak_uncovered)),
        'coord_worst_reward': worst_reward,
        'coord_weak3_reward': weak3_reward,
        'coord_steady_reward': steady_reward,
        'coord_duplicate_penalty': duplicate_penalty,
        'coord_total': float(total),
    }


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
        utility_mode: str = "log",    # "log" | "concave" | "maxmin_dual"
        concave_kappa: float = 1.0,   # saturation scale for the concave utility
    ):
        """
        Args:
            omega_q: Target priority weights
            P_FA: False alarm probability
            lambda_report: Weight for communication cost in reward
            eta_mc: Marginal contribution shaping coefficient
            alpha_pd: Weight for direct P_D reward term (0=off)
            utility_mode: Detection-utility curvature.
                - "log": ``-log(1-P_D)`` (historical; convex in D).
                - "concave": ``1-exp(-kappa*D)`` (concave/submodular).
                - "maxmin_dual": LP-dual-price weighted reward; requires
                  ``gain_per_watt``/``sensing_budget_w`` in
                  :meth:`compute_team_reward`.
            concave_kappa: Saturation scale used by "concave" and, per target,
                by "maxmin_dual".
        """
        self.omega_q = np.asarray(omega_q, dtype=np.float64)
        self.P_FA = P_FA
        self.lambda_report = lambda_report
        self.eta_mc = eta_mc
        self.alpha_pd = alpha_pd
        self.lambda_tail = lambda_tail
        self.lambda_tail_warmup = lambda_tail_warmup
        mode = str(utility_mode)
        if mode not in ("log", "concave", "maxmin_dual", "tstar", "bargaining"):
            raise ValueError(
                "utility_mode must be one of 'log', 'concave', "
                "'maxmin_dual', 'tstar', 'bargaining'")
        self.utility_mode = mode
        self.concave_kappa = float(concave_kappa)

    def _per_target_utility(self, D_q: np.ndarray) -> np.ndarray:
        """Per-target utility under the configured curvature.

        The ``maxmin_dual`` mode falls back to the concave utility on paths
        that lack the fixed-owner gain matrix (difference reward / marginal
        shaping); the team reward itself uses the exact dual price.
        """
        values = np.asarray(D_q, dtype=np.float64).reshape(-1)
        if self.utility_mode == "log":
            return utility_from_D(values, self.P_FA)
        return concave_saturating_deflection_utility(
            values, self.concave_kappa)

    def _weighted_utility(self, D_q: np.ndarray) -> float:
        return float(np.dot(self.omega_q, self._per_target_utility(D_q)))

    def compute_team_utility_from_deflection(
        self, D_q: np.ndarray, total_bits: float = 0.0
    ) -> float:
        """Task utility from per-target deflection (no P0 involved). Used for
        fixed-assignment difference reward: same deflection→utility mapping as
        team reward, minus the bits/penalty terms that don't change per frame."""
        return self._weighted_utility(D_q)

    def compute_team_reward(
        self,
        D_q_star: np.ndarray,      # (Q,) cumulative Deflection
        total_bits: float,          # total soft info bits reported
        constraint_penalty: float,  # total constraint violation penalty
        P_D_q: np.ndarray = None,   # (Q,) direct detection probs (optional, for alpha_pd>0)
        gain_per_watt: np.ndarray = None,   # (K,Q) fixed-owner per-watt gains
        sensing_budget_w: np.ndarray = None,  # (K,) per-UAV sensing budget
    ) -> float:
        """Compute team-level reward.

        r_team = (1-α_pd) * utility + α_pd * mean_P_D - lambda * bits - penalty

        ``utility`` follows ``self.utility_mode``:
        - "log" / "concave": ``sum_q omega_q * U(D_q)``;
        - "maxmin_dual": ``sum_q lambda*_q * U(D_q)`` with ``lambda*`` the
          fixed-owner max-min LP dual price (requires ``gain_per_watt`` and
          ``sensing_budget_w``).  If those are missing the mode degrades to the
          concave utility.

        Args:
            D_q_star: Per-target cumulative Deflection
            total_bits: Total bits reported this frame
            constraint_penalty: Total constraint violation penalty
            P_D_q: Per-target detection probabilities (for direct P_D term)
            gain_per_watt: Fixed-owner per-watt gain matrix (maxmin_dual only)
            sensing_budget_w: Per-UAV sensing budget (maxmin_dual only)

        Returns:
            Team reward (scalar)
        """
        if (
            self.utility_mode == "maxmin_dual"
            and gain_per_watt is not None
            and sensing_budget_w is not None
        ):
            utility, _ = maxmin_dual_reward(
                D_q_star,
                gain_per_watt,
                sensing_budget_w,
                kappa=self.concave_kappa,
            )
        elif (
            self.utility_mode == "bargaining"
            and gain_per_watt is not None
            and sensing_budget_w is not None
        ):
            # D0.92: the reward is the bargaining value eta = min_q r_q, the
            # common fraction of each target's own achievable headroom.
            from uav_isac.coordination.bargaining_power import (
                solve_fixed_structure_bargaining_lp,
            )
            result = solve_fixed_structure_bargaining_lp(
                gain_per_watt, sensing_budget_w)
            utility = float(result.bargaining_value)
        elif self.utility_mode == "tstar":
            # Reserve-first max-min LP value mapped to detection probability.
            # When the analytical power hook runs, D_q_star is already the
            # LP-optimal deflection, so min_q P_D(D_q) equals the LP value t*.
            from uav_isac.physical.detection import (
                compute_detection_probabilities,
            )
            pd = compute_detection_probabilities(
                np.asarray(D_q_star, dtype=np.float64).reshape(-1),
                self.P_FA,
            )
            utility = float(np.min(pd))
        else:
            utility = self._weighted_utility(D_q_star)
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
        detection_fusion_mode: str = "legacy_global",
        num_agents: Optional[int] = None,
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
        from uav_isac.physical.evidence import (
            receiver_deflection_from_selected,
            select_detection_deflection,
        )

        mode = str(detection_fusion_mode).strip().lower()
        if mode == "u2u_distributed":
            raise RuntimeError(
                "mode-aware marginal credit for u2u_distributed requires "
                "delivered packet attribution")
        K = (
            int(num_agents)
            if num_agents is not None
            else max([int(k) for k in uav_ids] + [-1]) + 1
        )

        def aggregate(edges):
            receiver_d = receiver_deflection_from_selected(
                edges, deflection_entries, K, Q)
            resolved_mode = (
                "central_oracle" if mode == "legacy_global" else mode)
            return select_detection_deflection(
                resolved_mode, receiver_d)

        # Compute full utility
        D_full = aggregate(selected_set)

        U_full = self._weighted_utility(D_full)

        # Compute utility without each UAV's contributions
        marginal_contribs = {}
        for k in uav_ids:
            edges_without_k = [
                (i, j, q)
                for (i, j, q) in selected_set
                if i != k and j != k
            ]
            D_without_k = aggregate(edges_without_k)

            U_without_k = self._weighted_utility(D_without_k)
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
