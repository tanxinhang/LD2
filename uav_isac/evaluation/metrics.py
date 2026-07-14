"""Evaluation metrics for UAV-ISAC system.

Metrics:
  - Average P_D across targets
  - Worst-target P_D
  - Jain fairness index for P_D
  - Cumulative sensing energy
  - Total uplink communication (soft info bits)
  - Constraint violation rate
  - Convergence speed (episodes to reach 90% of final performance)
"""

import numpy as np
from typing import Dict, List


def compute_avg_PD(P_D_q: np.ndarray) -> float:
    """Average detection probability across targets."""
    return float(np.mean(P_D_q))


def compute_worst_PD(P_D_q: np.ndarray) -> float:
    """Worst (minimum) detection probability across targets."""
    return float(np.min(P_D_q))


def compute_jain_fairness(P_D_q: np.ndarray) -> float:
    """Jain's fairness index for P_D across targets.

    J = (Σ P_D_q)^2 / (Q * Σ P_D_q^2)
    1.0 = perfectly fair, 1/Q = completely unfair
    """
    Q = len(P_D_q)
    if Q == 0:
        return 1.0
    sum_pd = np.sum(P_D_q)
    sum_sq = np.sum(P_D_q ** 2)
    if sum_sq < 1e-15:
        return 1.0
    return float(sum_pd ** 2 / (Q * sum_sq))


def compute_cumulative_energy(
    initial_batteries: np.ndarray,
    final_batteries: np.ndarray,
) -> float:
    """Total energy consumed during episode (J)."""
    return float(np.sum(initial_batteries - final_batteries))


def compute_total_communication(p0_solutions: List) -> float:
    """Total uplink soft information bits communicated."""
    return sum(sol.total_bits for sol in p0_solutions)


def compute_constraint_violation_rate(constraint_infos: List[Dict]) -> float:
    """Fraction of frames with at least one constraint violation."""
    if not constraint_infos:
        return 0.0
    violations = sum(1 for ci in constraint_infos if ci.get('any_violation', False))
    return violations / len(constraint_infos)


def compute_episode_metrics(
    P_D_history: List[np.ndarray],       # list of (Q,) per frame
    p0_solutions: List,                   # list of P0Solution per frame
    constraint_infos: List[Dict],
    initial_batteries: np.ndarray,
    final_batteries: np.ndarray,
    team_rewards: List[float],
    steady_window: int = 20,              # frames for steady-state (last-window) average
) -> Dict[str, float]:
    """Compute all metrics for one episode.

    Args:
        P_D_history: Per-frame detection probabilities
        p0_solutions: Per-frame P0 solutions
        constraint_infos: Per-frame constraint check results
        initial_batteries: (K,) initial battery levels
        final_batteries: (K,) final battery levels
        team_rewards: Per-frame team rewards

    Returns:
        Dict of metric name → value
    """
    if not P_D_history:
        return {}

    # Average over frames
    avg_pd = np.mean([compute_avg_PD(pd) for pd in P_D_history])
    worst_pd = np.mean([compute_worst_PD(pd) for pd in P_D_history])
    jain = np.mean([compute_jain_fairness(pd) for pd in P_D_history])

    # Final frame metrics
    final_avg_pd = compute_avg_PD(P_D_history[-1])
    final_worst_pd = compute_worst_PD(P_D_history[-1])

    # Steady-state metrics: average over the last `steady_window` frames.
    # More robust than the single final frame; reflects the achievable ceiling
    # AFTER the approach transient (UAVs have reached good bistatic geometry).
    w = min(steady_window, len(P_D_history))
    steady_avg_pd = np.mean([compute_avg_PD(pd) for pd in P_D_history[-w:]])
    steady_worst_pd = np.mean([compute_worst_PD(pd) for pd in P_D_history[-w:]])

    # Energy and communication
    energy = compute_cumulative_energy(initial_batteries, final_batteries)
    comm = compute_total_communication(p0_solutions)

    # Violations
    viol_rate = compute_constraint_violation_rate(constraint_infos)

    # Reward
    mean_reward = np.mean(team_rewards) if team_rewards else 0.0
    total_reward = np.sum(team_rewards) if team_rewards else 0.0

    return {
        'avg_P_D': avg_pd,
        'worst_P_D': worst_pd,
        'final_avg_P_D': final_avg_pd,
        'final_worst_P_D': final_worst_pd,
        'steady_avg_P_D': steady_avg_pd,
        'steady_worst_P_D': steady_worst_pd,
        'jain_fairness': jain,
        'cumulative_energy_J': energy,
        'total_communication_bits': comm,
        'constraint_violation_rate': viol_rate,
        'mean_team_reward': mean_reward,
        'total_team_reward': total_reward,
    }
