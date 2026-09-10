"""Legacy MAPPO trainer retained behind V2 adapters.

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
import json
import os
from statistics import NormalDist
from copy import deepcopy
from dataclasses import replace

from uav_isac.agents.networks import CausalContributionPredictor
from uav_isac.evaluation.responsibility_audit import (
    classify_responsibility_frame,
    node_removal_marginal_pd,
    summarize_responsibility_episodes,
)
from uav_isac.evaluation.temporal_credit_audit import (
    summarize_temporal_credit_proxy,
)
from uav_isac.evaluation.target_choice_audit import (
    apply_sensing_choice_intervention,
    evaluate_joint_sensing_pair_intervention,
    evaluate_sensing_choice_intervention,
    evaluate_target_choice_intervention,
    summarize_joint_sensing_pair_interventions,
    summarize_sensing_choice_interventions,
    summarize_target_choice_interventions,
)
from uav_isac.evaluation.evidence_oracle_audit import (
    receiver_local_and_global_pd,
    summarize_evidence_oracle,
    summarize_lossless_topk_capacity,
)
from uav_isac.evaluation.physical_oracle_audit import (
    evaluate_physical_feasibility_oracles,
    summarize_physical_feasibility_oracles,
)
from uav_isac.optimization.constrained_pareto import (
    assign_constrained_pareto_gradients,
    empirical_detection_cvar_residual,
    flattened_team_detection_cvar_residual,
    inequality_augmented_lagrangian,
    projected_target_dual_update,
)
from uav_isac.optimization.temporal_unrolled_power import (
    DifferentiableTemporalPowerUnroll,
    actor_sensing_budget,
    temporal_power_objectives,
)
from uav_isac.optimization.temporal_feasible_structure import (
    enumerate_feasible_single_role_structures,
    straight_through_structure,
    temporal_feasible_structure_mixture,
)


def stable_ppo_ratio_and_approx_kl(
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        log_ratio_bound: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a numerically bounded PPO ratio and Schulman KL estimate.

    For ``x = log(pi_new / pi_old)``, ``exp(x) - 1 - x`` is non-negative
    and is the standard low-variance approximate KL used by PPO. Clamping at
    |x|=20 does not make a rejected update acceptable (the estimate is already
    orders of magnitude above any configured trust-region bound), but prevents
    ``exp`` overflow while the fail-closed guard runs. Non-finite inputs map to
    an infinite KL so callers must reject the update.
    """
    log_ratio = new_log_probs - old_log_probs
    finite = torch.isfinite(log_ratio).all()
    safe_log_ratio = torch.nan_to_num(
        log_ratio, nan=0.0,
        posinf=log_ratio_bound, neginf=-log_ratio_bound,
    ).clamp(-log_ratio_bound, log_ratio_bound)
    ratio = torch.exp(safe_log_ratio)
    approx_kl = (torch.expm1(safe_log_ratio) - safe_log_ratio).mean()
    if not bool(finite):
        approx_kl = torch.full_like(approx_kl, float('inf'))
    return ratio, approx_kl


def linear_sum_assignment_numpy(cost_matrix: np.ndarray):
    """Exact rectangular Hungarian assignment without a SciPy runtime.

    The implementation uses the primal-dual O(n^2 m) algorithm for n <= m.
    Keeping this tiny diagnostic/teacher primitive in NumPy avoids loading a
    second OpenMP runtime after PyTorch has already initialized CUDA.
    """
    cost = np.asarray(cost_matrix, dtype=np.float64)
    if cost.ndim != 2 or min(cost.shape) == 0:
        raise ValueError('assignment cost must be a non-empty matrix')
    if not np.all(np.isfinite(cost)):
        raise ValueError('assignment cost must be finite')
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    rows, columns = cost.shape
    u = np.zeros(rows + 1, dtype=np.float64)
    v = np.zeros(columns + 1, dtype=np.float64)
    matched_row = np.zeros(columns + 1, dtype=np.int64)
    predecessor = np.zeros(columns + 1, dtype=np.int64)
    for row in range(1, rows + 1):
        matched_row[0] = row
        minimum = np.full(columns + 1, np.inf, dtype=np.float64)
        used = np.zeros(columns + 1, dtype=bool)
        column0 = 0
        while True:
            used[column0] = True
            row0 = matched_row[column0]
            delta = np.inf
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                reduced = (
                    cost[row0 - 1, column - 1] - u[row0] - v[column])
                if reduced < minimum[column]:
                    minimum[column] = reduced
                    predecessor[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(columns + 1):
                if used[column]:
                    u[matched_row[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if matched_row[column0] == 0:
                break
        while True:
            column1 = predecessor[column0]
            matched_row[column0] = matched_row[column1]
            column0 = column1
            if column0 == 0:
                break
    assigned_column = np.empty(rows, dtype=np.int64)
    for column in range(1, columns + 1):
        if matched_row[column] != 0:
            assigned_column[matched_row[column] - 1] = column - 1
    assigned_row = np.arange(rows, dtype=np.int64)
    if transposed:
        return assigned_column, assigned_row
    return assigned_row, assigned_column


def mask_sender_from_token_observations(
    observations: Dict[str, np.ndarray],
    sender: int,
    slices,
) -> Tuple[Dict[str, np.ndarray], bool]:
    """Apply ``do(sender token = null)`` to all receiver observations."""
    masked = {key: np.asarray(value).copy()
              for key, value in observations.items()}
    if not getattr(slices, 'has_comm_tokens', False):
        return masked, False
    K = int(slices.K)
    sender = int(sender)
    if sender < 0 or sender >= K:
        raise ValueError('sender outside observation topology')
    tokens_per_sender = int(slices.comm_tokens_per_sender)
    token_dim = int(slices.comm_token_per_sender)
    changed = False
    for receiver in range(K):
        if receiver == sender:
            continue
        key = str(receiver)
        if key not in masked:
            continue
        peers = [peer for peer in range(K) if peer != receiver]
        slot = peers.index(sender)
        token0 = int(slices.comm_token_start +
                     slot * tokens_per_sender * token_dim)
        token1 = token0 + tokens_per_sender * token_dim
        mask0 = int(slices.comm_mask_start + slot * tokens_per_sender)
        mask1 = mask0 + tokens_per_sender
        before = masked[key][mask0:mask1].copy()
        masked[key][token0:token1] = 0.0
        masked[key][mask0:mask1] = 0.0
        changed = changed or bool(np.any(before > 0.5))
    return masked, changed

def compute_cvar_k(Q: int, tail_fraction: float = 0.25) -> int:
    """Number of worst targets for CVaR tail-risk constraint.

    Args:
        Q: number of targets
        tail_fraction: fraction of targets in the tail (default 0.25)

    Returns:
        k: number of worst targets to average over (≥1)
    """
    return max(1, int(np.ceil(tail_fraction * Q)))


def quantile_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Huber quantile-regression loss for per-target scalar distributions.

    Args:
        prediction: ``(batch, targets, quantiles)``.
        target: sampled per-target outcomes ``(batch, targets)``.
        kappa: Huber transition point.
    """
    if prediction.ndim != 3 or target.ndim != 2:
        raise ValueError('quantile prediction/target ranks must be 3 and 2')
    if prediction.shape[:2] != target.shape:
        raise ValueError('quantile prediction and target batch/target axes differ')
    num_quantiles = prediction.shape[-1]
    tau = (
        torch.arange(
            num_quantiles, dtype=prediction.dtype,
            device=prediction.device) + 0.5
    ) / float(num_quantiles)
    error = target.unsqueeze(-1) - prediction
    abs_error = error.abs()
    kappa = max(float(kappa), 1e-8)
    huber = torch.where(
        abs_error <= kappa,
        0.5 * error.square(),
        kappa * (abs_error - 0.5 * kappa),
    )
    weight = torch.abs(
        tau.view(1, 1, -1) - (error.detach() < 0).to(prediction.dtype))
    return (weight * huber / kappa).mean()


def binary_risk_calibration_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
) -> Dict[str, float]:
    """Calibration and ranking diagnostics for a binary constraint critic."""
    score = np.asarray(logits, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels, dtype=np.float64).reshape(-1) >= 0.5
    if score.size == 0 or score.size != truth.size:
        raise ValueError('binary risk logits/labels must be non-empty and aligned')
    probability = 1.0 / (1.0 + np.exp(-np.clip(score, -40.0, 40.0)))
    prediction = probability >= 0.5
    positive = int(np.sum(truth))
    negative = int(truth.size - positive)
    sensitivity = float(np.mean(prediction[truth])) if positive else 1.0
    specificity = float(np.mean(~prediction[~truth])) if negative else 1.0

    auroc = 0.5
    if positive and negative:
        order = np.argsort(probability, kind='mergesort')
        sorted_probability = probability[order]
        ranks = np.empty(probability.size, dtype=np.float64)
        start = 0
        while start < probability.size:
            end = start + 1
            while (end < probability.size
                   and sorted_probability[end] == sorted_probability[start]):
                end += 1
            # One-based average rank for ties.
            ranks[order[start:end]] = 0.5 * (start + 1 + end)
            start = end
        rank_sum = float(np.sum(ranks[truth]))
        auroc = (
            rank_sum - positive * (positive + 1) / 2.0
        ) / float(positive * negative)

    auprc = float(np.mean(truth))
    if positive:
        descending = np.argsort(-probability, kind='mergesort')
        sorted_truth = truth[descending].astype(np.float64)
        precision = np.cumsum(sorted_truth) / np.arange(
            1, truth.size + 1, dtype=np.float64)
        auprc = float(np.sum(precision * sorted_truth) / positive)

    ece = 0.0
    bins = np.linspace(0.0, 1.0, max(1, int(num_bins)) + 1)
    for idx in range(len(bins) - 1):
        if idx == len(bins) - 2:
            mask = ((probability >= bins[idx])
                    & (probability <= bins[idx + 1]))
        else:
            mask = ((probability >= bins[idx])
                    & (probability < bins[idx + 1]))
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(probability[mask]))
                - float(np.mean(truth[mask])))
    return {
        'risk_violation_auroc': float(auroc),
        'risk_violation_auprc': float(auprc),
        'risk_violation_brier': float(np.mean(
            (probability - truth.astype(np.float64)) ** 2)),
        'risk_violation_ece': float(ece),
        'risk_violation_balanced_accuracy': float(
            0.5 * (sensitivity + specificity)),
        'risk_violation_prevalence': float(np.mean(truth)),
    }


def quantile_calibration_metrics(
    quantiles: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """Held-out calibration diagnostics for equally spaced quantile heads."""
    prediction = np.asarray(quantiles, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if prediction.ndim != 3 or target.shape != prediction.shape[:2]:
        raise ValueError(
            'quantiles must be (samples, targets, N) and targets (samples, targets)')
    count = prediction.shape[-1]
    tau = (np.arange(count, dtype=np.float64) + 0.5) / float(count)
    error = target[..., None] - prediction
    pinball = np.maximum(tau * error, (tau - 1.0) * error)
    coverage = np.mean(target[..., None] <= prediction, axis=(0, 1))
    crossing = (
        float(np.mean(prediction[..., :-1] > prediction[..., 1:]))
        if count > 1 else 0.0)
    return {
        'risk_quantile_pinball': float(np.mean(pinball)),
        'risk_quantile_calibration_error': float(np.mean(
            np.abs(coverage - tau))),
        'risk_quantile_max_calibration_error': float(np.max(
            np.abs(coverage - tau))),
        'risk_quantile_crossing_rate': crossing,
    }


def compute_comm_qos_metrics(per_target_pd: np.ndarray) -> np.ndarray:
    """Return [steady mean, bottom-3 mean, worst] for one transition.

    The same definitions are used by deterministic evaluation. For Q < 3,
    weak-3 naturally becomes the mean over all available targets.
    """
    values = np.asarray(per_target_pd, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.zeros(3, dtype=np.float64)
    ordered = np.sort(values)
    weak_k = min(3, values.size)
    return np.array([
        float(np.mean(values)),
        float(np.mean(ordered[:weak_k])),
        float(ordered[0]),
    ], dtype=np.float64)


def compute_comm_qos_rate_penalties(
    rate_indices: np.ndarray,
    rate_bits_per_dim: List[int],
    qos_values: np.ndarray,
    qos_targets: np.ndarray,
    min_rate_bits: int,
    penalty_scale: float,
) -> np.ndarray:
    """Per-sender low-rate penalty while sensing QoS remains infeasible.

    Unlike the shared sensing return, this barrier is attributable to the
    sampled discrete rate action.  Its severity is the largest normalized QoS
    deficit and becomes exactly zero once all floors are met.
    """
    rates = np.asarray(rate_indices, dtype=np.int64).reshape(-1)
    levels = np.asarray(rate_bits_per_dim, dtype=np.float64).reshape(-1)
    if levels.size == 0:
        raise ValueError('rate_bits_per_dim cannot be empty')
    if np.any(rates < 0) or np.any(rates >= levels.size):
        raise ValueError('rate index outside configured rate levels')
    targets = np.asarray(qos_targets, dtype=np.float64).reshape(-1)
    values = np.asarray(qos_values, dtype=np.float64).reshape(-1)
    if targets.shape != values.shape:
        raise ValueError('qos_values and qos_targets must have equal shape')
    normalized_deficit = np.maximum(targets - values, 0.0) / np.maximum(
        targets, 1e-9)
    severity = float(np.max(normalized_deficit)) if targets.size else 0.0
    minimum = max(float(min_rate_bits), 0.0)
    if minimum == 0.0 or severity == 0.0 or penalty_scale <= 0.0:
        return np.zeros_like(rates, dtype=np.float64)
    rate_shortfall = np.maximum(minimum - levels[rates], 0.0) / minimum
    return float(penalty_scale) * severity * rate_shortfall


def compute_comm_encouragement_bonuses(
    rate_indices: np.ndarray,
    qos_values: np.ndarray,
    qos_targets: np.ndarray,
    delivery_rate: float,
    bonus_weight: float,
    floor_ratio: float = 0.10,
) -> np.ndarray:
    """Reward successful communication participation without fixing a rate.

    The bonus is directly attributable to a sender's transmit/silent action,
    but is deliberately independent of payload precision. Its gate grows with
    the largest normalized sensing-QoS deficit. The environment's existing
    bit, energy and delay charges therefore decide whether 4/8/16/32 bits are
    worth paying for, while a failed delivery earns no communication bonus.
    """
    rates = np.asarray(rate_indices, dtype=np.int64).reshape(-1)
    if np.any(rates < 0):
        raise ValueError('rate indices must be non-negative')
    targets = np.asarray(qos_targets, dtype=np.float64).reshape(-1)
    values = np.asarray(qos_values, dtype=np.float64).reshape(-1)
    if targets.shape != values.shape:
        raise ValueError('qos_values and qos_targets must have equal shape')
    if bonus_weight <= 0.0:
        return np.zeros_like(rates, dtype=np.float64)

    normalized_deficit = np.maximum(targets - values, 0.0) / np.maximum(
        targets, 1e-9)
    severity = float(np.max(normalized_deficit)) if targets.size else 0.0
    severity = float(np.clip(severity, 0.0, 1.0))
    floor = float(np.clip(floor_ratio, 0.0, 1.0))
    qos_gate = floor + (1.0 - floor) * severity
    delivered = float(np.clip(delivery_rate, 0.0, 1.0))
    active = (rates > 0).astype(np.float64)
    return float(bonus_weight) * qos_gate * delivered * active


def compute_sender_delivery_penalties(
    rate_indices: np.ndarray,
    sender_delivery_rates: np.ndarray,
    penalty_weight: float,
) -> np.ndarray:
    """Charge each active sender for its own failed broadcast links.

    This is a training-time transport-constraint signal, not an observation.
    The deployed actor must infer future feasibility from its local reciprocal
    channel history and configured deadline/SNR threshold.
    """
    rates = np.asarray(rate_indices, dtype=np.int64).reshape(-1)
    delivery = np.asarray(
        sender_delivery_rates, dtype=np.float64).reshape(-1)
    if delivery.shape != rates.shape:
        raise ValueError(
            'one sender delivery rate is required per rate action')
    if np.any(rates < 0):
        raise ValueError('rate indices must be non-negative')
    if not np.all(np.isfinite(delivery)):
        raise ValueError('sender delivery rates must be finite')
    active = (rates > 0).astype(np.float64)
    failure = 1.0 - np.clip(delivery, 0.0, 1.0)
    return max(float(penalty_weight), 0.0) * active * failure


def compute_comm_rate_exploration_bonuses(
    rate_indices: np.ndarray,
    rate_bits_per_dim: List[int],
    qos_values: np.ndarray,
    qos_targets: np.ndarray,
    delivery_rate: float,
    bonus_weight: float,
    target_bits: int,
) -> np.ndarray:
    """Softly reward payload precision while sensing QoS is infeasible.

    Credit grows linearly up to ``target_bits`` and then saturates. This can
    move a collapsed 4-bit policy toward 8/16 bit without a hard minimum;
    explicit bit, energy and delay costs still decide between high rates.
    """
    rates = np.asarray(rate_indices, dtype=np.int64).reshape(-1)
    levels = np.asarray(rate_bits_per_dim, dtype=np.float64).reshape(-1)
    if levels.size == 0:
        raise ValueError('rate_bits_per_dim cannot be empty')
    if np.any(rates < 0) or np.any(rates >= levels.size):
        raise ValueError('rate index outside configured rate levels')
    targets = np.asarray(qos_targets, dtype=np.float64).reshape(-1)
    values = np.asarray(qos_values, dtype=np.float64).reshape(-1)
    if targets.shape != values.shape:
        raise ValueError('qos_values and qos_targets must have equal shape')
    if bonus_weight <= 0.0 or target_bits <= 0:
        return np.zeros_like(rates, dtype=np.float64)

    deficits = np.maximum(targets - values, 0.0) / np.maximum(targets, 1e-9)
    severity = float(np.clip(
        np.max(deficits) if targets.size else 0.0, 0.0, 1.0))
    delivered = float(np.clip(delivery_rate, 0.0, 1.0))
    precision_credit = np.clip(levels[rates] / float(target_bits), 0.0, 1.0)
    return float(bonus_weight) * severity * delivered * precision_credit


def build_rate_conditioned_topk_mask(
    claim_scores: torch.Tensor,
    rate_actions: torch.Tensor,
    topk_by_rate: List[int],
    maximum_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert an existing PPO rate action into a target-cardinality action.

    Duplicate physical rate levels can therefore represent different values of
    k while retaining the exact categorical log-probability already stored by
    MAPPO.  Target ordering remains the actor's learned sparse-claim ordering.
    ``maximum_mask`` can restrict the adaptive action to a warm-started top-k
    support, preventing an untrained controller from exposing new targets.
    """
    if claim_scores.ndim != 2:
        raise ValueError('claim scores must have shape (batch, Q)')
    rates = rate_actions.to(
        device=claim_scores.device, dtype=torch.long).reshape(-1)
    if rates.shape[0] != claim_scores.shape[0]:
        raise ValueError('one rate action is required per claim row')
    mapping = torch.as_tensor(
        topk_by_rate, dtype=torch.long, device=claim_scores.device)
    if mapping.numel() == 0:
        raise ValueError('topk_by_rate cannot be empty')
    if torch.any(rates < 0) or torch.any(rates >= mapping.numel()):
        raise ValueError('rate action outside adaptive top-k mapping')
    num_targets = claim_scores.shape[-1]
    cardinality = mapping[rates].clamp(0, num_targets)
    ranking_scores = claim_scores
    if maximum_mask is not None:
        if maximum_mask.shape != claim_scores.shape:
            raise ValueError('maximum mask must match claim scores')
        supported = maximum_mask > 0.5
        ranking_scores = torch.where(
            supported,
            claim_scores,
            torch.full_like(claim_scores, -torch.inf),
        )
        cardinality = torch.minimum(
            cardinality, supported.sum(dim=-1).to(cardinality.dtype))
    order = torch.argsort(ranking_scores, dim=-1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(
        1,
        order,
        torch.arange(num_targets, device=claim_scores.device)
        .unsqueeze(0).expand_as(order),
    )
    mask = (rank < cardinality.unsqueeze(-1)).to(claim_scores.dtype)
    if maximum_mask is not None:
        mask = mask * (maximum_mask > 0.5).to(mask.dtype)
    return mask


def update_long_silence_penalties(
    rate_indices: np.ndarray,
    previous_streaks: np.ndarray,
    grace_decisions: int,
    penalty_per_decision: float,
    max_penalty: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Advance sender silence streaks and return capped liveness penalties.

    A non-zero rate resets that sender immediately. Short silence inside the
    grace window is free, preserving the policy's ability to save resources;
    only persistent channel abandonment is penalized.
    """
    rates = np.asarray(rate_indices, dtype=np.int64).reshape(-1)
    previous = np.asarray(previous_streaks, dtype=np.int64).reshape(-1)
    if rates.shape != previous.shape:
        raise ValueError('rate_indices and previous_streaks must match')
    if np.any(rates < 0) or np.any(previous < 0):
        raise ValueError('rates and silence streaks must be non-negative')

    current = np.where(rates == 0, previous + 1, 0).astype(np.int64)
    grace = max(int(grace_decisions), 0)
    excess = np.maximum(current - grace, 0).astype(np.float64)
    scale = max(float(penalty_per_decision), 0.0)
    cap = max(float(max_penalty), 0.0)
    penalties = np.minimum(scale * excess, cap)
    return current, penalties


def build_comm_qos_checkpoint_key(
    qos_values: np.ndarray,
    qos_targets: np.ndarray,
    bits_per_frame: float,
    *,
    feasibility_lcb: Optional[float] = None,
    worst_lcb: Optional[float] = None,
    worst_cvar: Optional[float] = None,
    trimmed_worst: Optional[float] = None,
) -> Tuple[float, ...]:
    """Lexicographic checkpoint rank led by QoS feasibility and worst P_D.

    Communication use is intentionally the final tie-breaker. This prevents a
    cheap or high-steady policy from displacing one that better protects the
    worst target.
    """
    values = np.asarray(qos_values, dtype=np.float64).reshape(-1)
    targets = np.asarray(qos_targets, dtype=np.float64).reshape(-1)
    if values.shape != targets.shape or values.size != 3:
        raise ValueError('checkpoint QoS values/targets must both have size 3')
    feasible = float(np.all(values >= targets))
    steady, weak3, worst = (float(x) for x in values)
    if worst_lcb is not None:
        return (
            float(feasibility_lcb if feasibility_lcb is not None else feasible),
            float(worst_lcb),
            float(worst_cvar if worst_cvar is not None else worst),
            worst,
            float(trimmed_worst if trimmed_worst is not None else worst),
            weak3,
            steady,
            -max(float(bits_per_frame), 0.0),
        )
    return (feasible, worst, weak3, steady, -max(float(bits_per_frame), 0.0))


def apply_cacsr_sensing_residual(
    sensing_weights: torch.Tensor,
    crisis_gate: torch.Tensor,
    semantic_quality: torch.Tensor,
    endpoint_count: torch.Tensor,
    local_capability: torch.Tensor,
    gain: float,
    desired_endpoints: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a centered, bounded CA-CSR correction under a fixed RF budget."""
    if sensing_weights.shape != crisis_gate.shape:
        raise ValueError('CA-CSR tensors must share (batch, target) shape')
    desired = max(1, int(desired_endpoints))
    deficit = torch.relu(float(desired) - endpoint_count) / float(desired)
    overload = torch.relu(endpoint_count - float(desired)) / float(desired)
    quality_gap = (1.0 - semantic_quality).clamp(0.0, 1.0)
    raw = torch.tanh(
        local_capability.clamp(0.0, 1.0) * (deficit + quality_gap)
        - overload)
    delta = crisis_gate.to(raw.dtype) * raw
    delta = delta - delta.mean(dim=-1, keepdim=True)
    logits = torch.log(sensing_weights.clamp_min(1e-8)) + float(gain) * delta
    return torch.softmax(logits, dim=-1), delta


# Seeds permanently isolated after the 2026-07-29 test-set contamination
# event (see docs/EXPERIMENT_LOG.md:1764 and
# docs/CURRENT_SYSTEM_MODEL.md:412-416).  The authoritative registry is
# config/quarantined_seeds.json; this constant is only the fail-closed
# fallback when that file is missing or malformed.  Keep the two in sync
# (tests/test_quarantined_seeds.py locks the documented set).
QUARANTINED_SEEDS_FALLBACK: frozenset[int] = frozenset({795, 747, 105, 860, 2})
_QUARANTINE_DOC_REF = (
    "docs/EXPERIMENT_LOG.md:1764, docs/CURRENT_SYSTEM_MODEL.md:412-416")


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))


def load_quarantined_seeds() -> frozenset[int]:
    """Load the quarantined-seed registry (single source of truth).

    The registry lives in config/quarantined_seeds.json so it can be
    versioned and audited independently of code.  If the file is missing or
    malformed the built-in fallback is returned so evaluation still fails
    closed on the documented seeds.
    """
    path = os.path.join(_project_root(), "config", "quarantined_seeds.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        seeds = {int(seed) for seed in payload.get("seeds", [])}
        if not seeds:
            return QUARANTINED_SEEDS_FALLBACK
        return frozenset(seeds)
    except (OSError, ValueError, TypeError):
        return QUARANTINED_SEEDS_FALLBACK


def load_stratified_seed_split(
    path: str, split: str, *, strict: bool = True,
) -> List[int]:
    """Load one non-empty, duplicate-free split from a versioned seed bank.

    With strict=True (default) any split that still contains a quarantined
    seed is rejected with an explicit error, so the documented "permanently
    isolated" seeds can never silently enter a confirmatory or final
    performance evaluation again.  Pass strict=False only for structural
    validation of legacy banks that pre-date the quarantine (see
    tests/test_robust_checkpoint_protocol.py); do not use it for any run that
    produces a reported performance number.
    """
    with open(os.path.abspath(path), "r", encoding="utf-8") as handle:
        bank = json.load(handle)
    splits = bank.get("splits", {})
    if split not in splits:
        raise KeyError(f"seed split {split!r} not found in {path}")
    seeds = [int(seed) for seed in splits[split]]
    if not seeds:
        raise ValueError(f"seed split {split!r} is empty")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"seed split {split!r} contains duplicates")
    if strict:
        quarantined = load_quarantined_seeds()
        hits = sorted(seed for seed in seeds if seed in quarantined)
        if hits:
            raise ValueError(
                f"seed split {split!r} of {path} contains quarantined seeds "
                f"{hits}; these seeds were permanently isolated "
                f"({_QUARANTINE_DOC_REF}) and must not enter confirmatory or "
                f"final performance evaluation. Regenerate the bank with "
                f"tools/seed_stratification.py (quarantine is excluded "
                f"automatically) or drop the split explicitly.")
    return seeds


def load_training_seed_pool(
    path: str,
    max_nearest_m: float = 350.0,
) -> Tuple[List[int], np.ndarray]:
    """Load training seeds while excluding every evaluation split.

    Returns seeds sorted by geometric difficulty and their aligned composite
    difficulty scores.  The nominal nearest-distance filter prevents clearly
    out-of-scope layouts from dominating prioritized replay; it is not treated
    as a proof of physical feasibility.
    """
    with open(os.path.abspath(path), "r", encoding="utf-8") as handle:
        bank = json.load(handle)
    metadata = bank.get("seed_metadata", {})
    if not metadata:
        raise ValueError(f"seed bank {path} has no seed_metadata")
    reserved = {
        int(seed)
        for values in bank.get("splits", {}).values()
        for seed in values
    }
    quarantined = load_quarantined_seeds()
    candidates = []
    for raw_seed, raw_meta in metadata.items():
        seed = int(raw_seed)
        if seed in reserved or seed in quarantined:
            continue
        nearest = float(raw_meta.get("worst_nearest_m", np.inf))
        if nearest > float(max_nearest_m):
            continue
        difficulty = float(raw_meta.get("difficulty_score_m", nearest))
        if np.isfinite(difficulty):
            candidates.append((difficulty, seed))
    if not candidates:
        raise ValueError(f"seed bank {path} has no eligible training seeds")
    candidates.sort()
    return (
        [seed for _, seed in candidates],
        np.asarray([difficulty for difficulty, _ in candidates],
                   dtype=np.float64),
    )


class PrioritizedGeometrySeedSampler:
    """Policy-adaptive seed replay with a geometry curriculum.

    Priority combines QoS deficit with change relative to the seed's running
    performance estimate.  A uniform mixture keeps coverage broad, while the
    curriculum gradually exposes harder geometries in difficulty order.
    """

    def __init__(
        self,
        seeds: List[int],
        difficulties: np.ndarray,
        *,
        worst_floor: float = 0.60,
        uniform_mix: float = 0.20,
        priority_alpha: float = 0.70,
        ema: float = 0.20,
        min_curriculum_fraction: float = 0.30,
        curriculum_frames: int = 300000,
    ):
        if len(seeds) == 0 or len(seeds) != len(difficulties):
            raise ValueError('training seeds/difficulties must be non-empty and aligned')
        self.seeds = np.asarray(seeds, dtype=np.int64)
        self.difficulties = np.asarray(difficulties, dtype=np.float64)
        self.worst_floor = max(float(worst_floor), 1e-6)
        self.uniform_mix = float(np.clip(uniform_mix, 0.0, 1.0))
        self.priority_alpha = max(float(priority_alpha), 0.0)
        self.ema = float(np.clip(ema, 1e-6, 1.0))
        self.min_curriculum_fraction = float(np.clip(
            min_curriculum_fraction, 1.0 / len(seeds), 1.0))
        self.curriculum_frames = max(int(curriculum_frames), 1)
        self.priorities = np.ones(len(seeds), dtype=np.float64)
        self.performance_ema = np.full(len(seeds), np.nan, dtype=np.float64)
        self._index = {int(seed): idx for idx, seed in enumerate(self.seeds)}

    def eligible_count(self, total_frames: int) -> int:
        progress = float(np.clip(
            total_frames / self.curriculum_frames, 0.0, 1.0))
        fraction = self.min_curriculum_fraction + (
            1.0 - self.min_curriculum_fraction) * progress
        return min(len(self.seeds), max(1, int(np.ceil(
            fraction * len(self.seeds)))))

    def sample(self, rng: np.random.Generator, total_frames: int) -> int:
        count = self.eligible_count(total_frames)
        priority = np.power(
            np.maximum(self.priorities[:count], 1e-6),
            self.priority_alpha)
        priority /= priority.sum()
        probability = (
            (1.0 - self.uniform_mix) * priority
            + self.uniform_mix / count)
        return int(rng.choice(self.seeds[:count], p=probability))

    def update(self, seed: int, episode_worst: float) -> None:
        idx = self._index.get(int(seed))
        if idx is None or not np.isfinite(episode_worst):
            return
        current = float(np.clip(episode_worst, 0.0, 1.0))
        old = self.performance_ema[idx]
        learning_change = 0.0 if np.isnan(old) else abs(current - old)
        self.performance_ema[idx] = (
            current if np.isnan(old)
            else (1.0 - self.ema) * old + self.ema * current)
        deficit = np.clip(
            (self.worst_floor - current) / self.worst_floor, 0.0, 1.0)
        target_priority = 0.05 + deficit + 2.0 * learning_change
        self.priorities[idx] = (
            (1.0 - self.ema) * self.priorities[idx]
            + self.ema * target_priority)

    def diagnostics(self, total_frames: int) -> Dict[str, float]:
        count = self.eligible_count(total_frames)
        return {
            'training_seed_eligible_count': float(count),
            'training_seed_max_difficulty_m': float(
                self.difficulties[count - 1]),
            'training_seed_mean_priority': float(
                np.mean(self.priorities[:count])),
        }


def compute_bottleneck_risk_advantage(
    per_target_advantages: torch.Tensor,
    per_target_pd: torch.Tensor,
    scalar_advantages: torch.Tensor,
    *,
    tail_fraction: float = 0.50,
    temperature: float = 0.10,
    target_floor: float = 0.60,
    scalar_mix: float = 0.25,
) -> torch.Tensor:
    """Route PPO credit through the weakest target tail of each transition."""
    if (per_target_advantages.ndim != 2
            or per_target_pd.shape != per_target_advantages.shape):
        raise ValueError('per-target advantage and P_D tensors must align')
    if scalar_advantages.ndim != 1 or scalar_advantages.shape[0] != per_target_pd.shape[0]:
        raise ValueError('scalar advantages must align with per-target rows')
    batch, num_targets = per_target_pd.shape
    tail_k = max(1, int(np.ceil(float(tail_fraction) * num_targets)))
    tail_k = min(tail_k, num_targets)

    normalized = torch.zeros_like(per_target_advantages)
    for q in range(num_targets):
        advantage_q = per_target_advantages[:, q]
        normalized[:, q] = (
            (advantage_q - advantage_q.mean())
            / advantage_q.std(unbiased=False).clamp_min(1e-8))

    # Select the strict bottom-k targets first, then use a smooth deficit score
    # only inside that tail. This avoids spreading the gradient over already
    # healthy targets while keeping it differentiable with respect to neither
    # rewards nor target selection (both are training labels).
    bottom_indices = torch.topk(
        per_target_pd, k=tail_k, dim=-1, largest=False).indices
    tail_mask = torch.zeros_like(per_target_pd, dtype=torch.bool)
    tail_mask.scatter_(1, bottom_indices, True)
    deficit = torch.relu(float(target_floor) - per_target_pd)
    scores = deficit / max(float(temperature), 1e-4)
    scores = scores.masked_fill(~tail_mask, -1e9)
    weights = torch.softmax(scores, dim=-1).detach()
    risk_advantage = (weights * normalized).sum(dim=-1)

    scalar = (
        (scalar_advantages - scalar_advantages.mean())
        / scalar_advantages.std(unbiased=False).clamp_min(1e-8))
    mix = float(np.clip(scalar_mix, 0.0, 1.0))
    combined = (1.0 - mix) * risk_advantage + mix * scalar
    return (
        (combined - combined.mean())
        / combined.std(unbiased=False).clamp_min(1e-8))


def compute_robust_checkpoint_statistics(
    episode_steady: np.ndarray,
    episode_weak3: np.ndarray,
    episode_worst: np.ndarray,
    qos_targets: np.ndarray,
    *,
    alpha: float = 0.05,
    bootstrap_samples: int = 2000,
    cvar_fraction: float = 0.20,
    bootstrap_seed: int = 20260721,
    qos_tol: float = 1e-6,
) -> Dict[str, float]:
    """Compute selection statistics without replacing strict target worst.

    Wilson is applied only to the Bernoulli per-seed QoS feasibility event.
    The continuous worst metric uses a deterministic non-parametric bootstrap
    lower bound. CVaR is the mean over the lowest scenario tail.

    ``qos_tol`` is a solver-level tolerance (advice 010): the task-constrained
    gauge pins the worst target exactly at the floor, and P_D evaluated there
    can sit 1e-16 below it, failing a strict comparison by float noise.  The
    tolerance is tiny on P_D in [0,1] but far above binary error; raw values
    and slacks are still reported so nothing below the floor is hidden.
    """
    steady = np.asarray(episode_steady, dtype=np.float64).reshape(-1)
    weak3 = np.asarray(episode_weak3, dtype=np.float64).reshape(-1)
    worst = np.asarray(episode_worst, dtype=np.float64).reshape(-1)
    targets = np.asarray(qos_targets, dtype=np.float64).reshape(-1)
    if not (steady.size == weak3.size == worst.size) or steady.size == 0:
        raise ValueError("episode QoS arrays must be non-empty and equal length")
    if targets.size != 3:
        raise ValueError("qos_targets must contain steady/weak3/worst")
    tol = float(np.clip(qos_tol, 0.0, 0.01))
    alpha = float(np.clip(alpha, 1e-6, 0.499999))
    n = int(worst.size)
    rng = np.random.default_rng(int(bootstrap_seed))
    draws = max(1, int(bootstrap_samples))
    bootstrap_mean = worst[rng.integers(0, n, size=(draws, n))].mean(axis=1)
    worst_lcb = float(np.quantile(bootstrap_mean, alpha))

    tail_n = max(1, int(np.ceil(float(cvar_fraction) * n)))
    worst_cvar = float(np.mean(np.sort(worst)[:tail_n]))
    feasible = (
        (steady >= targets[0] - tol)
        & (weak3 >= targets[1] - tol)
        & (worst >= targets[2] - tol))
    successes = int(np.sum(feasible))
    p_hat = successes / n
    # Audit 2026-08-25: the project convention (docs/CURRENT_SYSTEM_MODEL.md, tests) is the
    # 95% two-sided Wilson lower endpoint, z = inv_cdf(1-alpha/2) = 1.96, not
    # the one-sided z = inv_cdf(1-alpha) = 1.645.  The old one-sided value could
    # flip a 0.70 gate (e.g. 78/100 -> 0.705 vs 0.689) and disagreed with the
    # formal gate tools.  Aligned with the documented standard.
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    denom = 1.0 + z * z / n
    center = p_hat + z * z / (2.0 * n)
    radius = z * np.sqrt(
        p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n))
    wilson_lcb = float(max(0.0, (center - radius) / denom))
    return {
        "eval_worst_std": float(np.std(worst, ddof=0)),
        "eval_worst_lcb": worst_lcb,
        "eval_worst_cvar": worst_cvar,
        "eval_qos_feasible_rate": float(p_hat),
        "eval_qos_feasible_wilson_lcb": wilson_lcb,
        "eval_qos_tol": float(tol),
        "eval_worst_slack": float(np.mean(worst - targets[2])),
        "eval_weak3_slack": float(np.mean(weak3 - targets[1])),
        "eval_steady_slack": float(np.mean(steady - targets[0])),
        "eval_raw_worst_P_D": float(np.mean(worst)),
        "eval_raw_weak3_P_D": float(np.mean(weak3)),
        "eval_raw_steady_P_D": float(np.mean(steady)),
    }


def compute_target_allocation_regularizer(
    assignment_probs: torch.Tensor,
    num_agents: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Coverage-balance and commitment losses for decentralized assignment.

    Consecutive rows are the K agents from one stored transition. Balanced
    column load discourages duplicate target pursuit; per-agent entropy
    encourages an unambiguous local responsibility. No message field is given
    a prescribed semantic label.
    """
    if assignment_probs.ndim != 2:
        raise ValueError('assignment_probs must have shape (batch, Q)')
    batch, num_targets = assignment_probs.shape
    if batch % num_agents != 0:
        raise ValueError('assignment batch must contain complete UAV teams')
    teams = assignment_probs.reshape(-1, num_agents, num_targets)
    mean_load = teams.mean(dim=1)
    desired = torch.full_like(mean_load, 1.0 / max(num_targets, 1))
    balance_loss = (mean_load - desired).pow(2).mean()
    commitment_loss = -(
        assignment_probs
        * torch.log(assignment_probs.clamp_min(1e-8))
    ).sum(dim=-1).mean()
    return balance_loss, commitment_loss


def compute_sparse_endpoint_load_loss(
    assignment_probs: torch.Tensor,
    num_agents: int,
    commitment_topk: int,
    desired_endpoints: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Penalize under/over-subscribed targets for multi-static sensing.

    The forward pass uses a hard local top-k commitment, while its backward
    pass follows the soft assignment probabilities.  Unlike one-to-one target
    balancing, this permits the requested two sensing endpoints per target.
    """
    if assignment_probs.ndim != 2:
        raise ValueError('assignment_probs must have shape (batch, Q)')
    batch, num_targets = assignment_probs.shape
    if batch % num_agents != 0:
        raise ValueError('assignment batch must contain complete UAV teams')
    topk = min(max(int(commitment_topk), 1), num_targets)
    hard = torch.zeros_like(assignment_probs)
    hard.scatter_(
        dim=-1,
        index=torch.topk(assignment_probs, k=topk, dim=-1).indices,
        value=1.0,
    )
    commitments = hard + assignment_probs - assignment_probs.detach()
    load = commitments.reshape(-1, num_agents, num_targets).sum(dim=1)
    desired = torch.as_tensor(
        float(desired_endpoints), dtype=load.dtype, device=load.device)
    underload_loss = torch.relu(desired - load).pow(2).mean()
    overload_loss = torch.relu(load - desired).pow(2).mean()
    return underload_loss, overload_loss


def compute_target_allocation_temporal_loss(
    assignment_probs: torch.Tensor,
    num_agents: int,
    delay_teams: int,
    transition_masks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Keep each UAV's learned target commitment stable across decisions.

    Rollout teams are stored with environments interleaved, so the preceding
    decision for the same environment is ``delay_teams=num_envs`` rows back.
    Episode boundaries are excluded when transition masks are available.
    The previous distribution is detached: the current decision learns to keep
    a commitment, while balance/quality losses remain free to improve it.
    """
    if assignment_probs.ndim != 2:
        raise ValueError('assignment_probs must have shape (batch, Q)')
    if assignment_probs.shape[0] % num_agents != 0:
        raise ValueError('assignment batch must contain complete UAV teams')
    teams = assignment_probs.reshape(
        -1, num_agents, assignment_probs.shape[-1])
    lag = max(int(delay_teams), 1)
    if teams.shape[0] <= lag:
        return assignment_probs.sum() * 0.0
    per_pair = (teams[lag:] - teams[:-lag].detach()).pow(2).mean(
        dim=(1, 2))
    if transition_masks is not None:
        masks = transition_masks.reshape(-1, num_agents)
        valid = masks[:-lag].amin(dim=1) > 0.5
        if not torch.any(valid):
            return assignment_probs.sum() * 0.0
        per_pair = per_pair[valid]
    return per_pair.mean()


def compute_balanced_assignment_teacher(
    global_states: torch.Tensor,
    num_agents: int,
    num_targets: int,
    region_size: Tuple[float, float],
    max_dp: float,
    num_envs: int = 1,
    transition_masks: Optional[torch.Tensor] = None,
    switching_penalty_m: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build CTDE-only balanced coverage labels and movement actions.

    The critic state layout starts with K blocks of eight normalized UAV
    features followed by Q blocks of six normalized target features.  A
    minimum-distance linear assignment prevents several UAVs from pursuing the
    same easy target.  When K > Q, repeated target slots distribute the excess
    UAVs as evenly as possible.

    Returns flattened tensors in rollout order: labels ``(T*K,)`` and bounded
    metric movements ``(T*K, 2)``.  No teacher field is added to the actor's
    execution-time observation.
    """
    if global_states.ndim != 2:
        raise ValueError('global_states must have shape (T, global_dim)')
    if num_agents < 1 or num_targets < 1:
        raise ValueError('num_agents and num_targets must be positive')
    required = 8 * num_agents + 6 * num_targets
    if global_states.shape[1] < required:
        raise ValueError(
            f'global state has {global_states.shape[1]} fields; need {required}')

    # Assignment is a non-differentiable training label, so solve the tiny
    # combinatorial problem on CPU and move the result back to the input device.
    gs = global_states.detach().cpu().numpy()
    scale = np.asarray(region_size, dtype=np.float64).reshape(2)
    if np.any(scale <= 0):
        raise ValueError('region_size must be positive')
    labels = np.zeros((len(gs), num_agents), dtype=np.int64)
    movements = np.zeros((len(gs), num_agents, 2), dtype=np.float32)
    num_envs = max(1, int(num_envs))
    switching_penalty_m = max(0.0, float(switching_penalty_m))
    previous_labels = [None for _ in range(num_envs)]
    team_masks = None
    if transition_masks is not None:
        mask_np = transition_masks.detach().cpu().numpy().reshape(-1)
        if mask_np.size != len(gs) * num_agents:
            raise ValueError(
                'transition_masks must contain one value per teacher label')
        team_masks = mask_np.reshape(len(gs), num_agents)

    target_offset = 8 * num_agents
    for step, state in enumerate(gs):
        env_index = step % num_envs
        if (team_masks is not None and step >= num_envs
                and np.min(team_masks[step - num_envs]) < 0.5):
            previous_labels[env_index] = None
        uav_xy_norm = np.stack([
            state[8 * k:8 * k + 2] for k in range(num_agents)
        ])
        target_xy_norm = np.stack([
            state[target_offset + 6 * q:target_offset + 6 * q + 2]
            for q in range(num_targets)
        ])

        if num_agents <= num_targets:
            slot_targets = np.arange(num_targets, dtype=np.int64)
        else:
            # Every target receives floor(K/Q) slots; the first K mod Q targets
            # receive one extra.  Slot identity is irrelevant after assignment.
            slot_targets = np.arange(num_agents, dtype=np.int64) % num_targets
        displacement_m = (
            uav_xy_norm[:, None, :]
            - target_xy_norm[slot_targets][None, :, :]
        ) * scale[None, None, :]
        cost = np.linalg.norm(displacement_m, axis=-1)
        previous = previous_labels[env_index]
        if previous is not None and switching_penalty_m > 0.0:
            cost = cost + switching_penalty_m * (
                slot_targets[None, :] != previous[:, None])
        rows, cols = linear_sum_assignment_numpy(cost)
        step_labels = np.zeros(num_agents, dtype=np.int64)
        step_labels[rows] = slot_targets[cols]
        labels[step] = step_labels
        previous_labels[env_index] = step_labels.copy()

        delta = (
            target_xy_norm[step_labels] - uav_xy_norm
        ) * scale[None, :]
        norm = np.linalg.norm(delta, axis=-1, keepdims=True)
        ratio = np.minimum(1.0, float(max_dp) / np.maximum(norm, 1e-12))
        movements[step] = (delta * ratio).astype(np.float32)

    device = global_states.device
    return (
        torch.as_tensor(labels.reshape(-1), dtype=torch.long, device=device),
        torch.as_tensor(
            movements.reshape(-1, 2), dtype=torch.float32, device=device),
    )


def compute_qos_bistatic_assignment_teacher(
    global_states: torch.Tensor,
    num_agents: int,
    num_targets: int,
    region_size: Tuple[float, float],
    max_dp: float,
    *,
    commitment_frames: int = 5,
    qos_floor: float = 0.60,
    qos_weight: float = 2.0,
    height_m: float = 20.0,
    num_envs: int = 1,
    transition_masks: Optional[torch.Tensor] = None,
    switching_penalty_m: float = 0.0,
    policy_assignment_probs: Optional[torch.Tensor] = None,
    policy_alignment_m: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """QoS-aware persistent assignment for the slow kinematic layer.

    For every candidate UAV-target commitment, predict the transmitter range
    after one commitment interval and combine it with the best *other* UAV's
    current receiver range.  Targets below ``qos_floor`` receive larger cost
    weights, so the globally assigned UAV geometry preferentially improves the
    current bottleneck instead of merely minimizing mean travel distance.

    The centralized P_D values create training labels only.  They are never
    appended to the deployed local observation or token payload.
    """
    if global_states.ndim != 2:
        raise ValueError('global_states must have shape (T, global_dim)')
    if num_agents < 2:
        raise ValueError('bistatic assignment requires at least two UAVs')
    if num_targets < 1:
        raise ValueError('num_targets must be positive')
    required = 8 * num_agents + 6 * num_targets + 1 + 2 * num_targets
    if global_states.shape[1] < required:
        raise ValueError(
            f'global state has {global_states.shape[1]} fields; need {required}')

    gs = global_states.detach().cpu().numpy()
    scale = np.asarray(region_size, dtype=np.float64).reshape(2)
    if np.any(scale <= 0):
        raise ValueError('region_size must be positive')
    max_travel = max(0.0, float(max_dp)) * max(1, int(commitment_frames))
    qos_floor = float(np.clip(qos_floor, 1e-6, 1.0))
    qos_weight = max(0.0, float(qos_weight))
    height_m = max(0.0, float(height_m))
    switching_penalty_m = max(0.0, float(switching_penalty_m))
    policy_alignment_m = max(0.0, float(policy_alignment_m))
    num_envs = max(1, int(num_envs))

    policy_probs = None
    if policy_assignment_probs is not None:
        policy_probs = policy_assignment_probs.detach().cpu().numpy()
        if policy_probs.ndim == 2:
            policy_probs = policy_probs.reshape(
                len(gs), num_agents, num_targets)
        expected = (len(gs), num_agents, num_targets)
        if policy_probs.shape != expected:
            raise ValueError(
                'policy_assignment_probs must have shape '
                f'{expected} or {(len(gs) * num_agents, num_targets)}')
        policy_probs = np.maximum(policy_probs, 0.0)
        policy_probs /= np.maximum(
            policy_probs.sum(axis=-1, keepdims=True), 1e-12)

    labels = np.zeros((len(gs), num_agents), dtype=np.int64)
    movements = np.zeros((len(gs), num_agents, 2), dtype=np.float32)
    previous_labels = [None for _ in range(num_envs)]
    team_masks = None
    if transition_masks is not None:
        mask_np = transition_masks.detach().cpu().numpy().reshape(-1)
        if mask_np.size != len(gs) * num_agents:
            raise ValueError(
                'transition_masks must contain one value per teacher label')
        team_masks = mask_np.reshape(len(gs), num_agents)

    target_offset = 8 * num_agents
    pd_offset = target_offset + 6 * num_targets + 1 + num_targets
    for step, state in enumerate(gs):
        env_index = step % num_envs
        if (team_masks is not None and step >= num_envs
                and np.min(team_masks[step - num_envs]) < 0.5):
            previous_labels[env_index] = None

        uav_xy = np.stack([
            state[8 * k:8 * k + 2] for k in range(num_agents)
        ]) * scale[None, :]
        target_xy = np.stack([
            state[target_offset + 6 * q:target_offset + 6 * q + 2]
            for q in range(num_targets)
        ]) * scale[None, :]
        previous_pd = np.clip(
            state[pd_offset:pd_offset + num_targets], 0.0, 1.0)
        deficit = np.maximum(qos_floor - previous_pd, 0.0) / qos_floor
        urgency = 1.0 + qos_weight * deficit

        if num_agents <= num_targets:
            slot_targets = np.arange(num_targets, dtype=np.int64)
        else:
            slot_targets = np.arange(num_agents, dtype=np.int64) % num_targets

        cost = np.zeros((num_agents, len(slot_targets)), dtype=np.float64)
        for k in range(num_agents):
            for slot, q in enumerate(slot_targets):
                delta = target_xy[q] - uav_xy[k]
                horizontal = float(np.linalg.norm(delta))
                if horizontal > 1e-12:
                    moved = uav_xy[k] + (
                        min(max_travel, horizontal) * delta / horizontal)
                else:
                    moved = uav_xy[k]
                tx_range = float(np.sqrt(
                    np.sum((moved - target_xy[q]) ** 2) + height_m ** 2))
                receiver_indices = [j for j in range(num_agents) if j != k]
                rx_horizontal = np.linalg.norm(
                    uav_xy[receiver_indices] - target_xy[q], axis=-1)
                rx_range = float(np.min(np.sqrt(
                    rx_horizontal ** 2 + height_m ** 2)))
                # sqrt(R_tx*R_rx) is an equivalent bistatic range: lower is
                # better, while avoiding the extreme scale of the R^-4 gain.
                bistatic_range = np.sqrt(max(tx_range * rx_range, 1e-12))
                cost[k, slot] = urgency[q] * bistatic_range
                # Student-feasibility projection: among physically
                # near-equivalent matchings prefer a permutation already
                # representable by decentralized local intentions.  The
                # bounded term cannot dominate geometry by more than the
                # configured metre-scale allowance per assignment.
                if policy_probs is not None and policy_alignment_m > 0.0:
                    cost[k, slot] += policy_alignment_m * (
                        1.0 - policy_probs[step, k, q])

        previous = previous_labels[env_index]
        if previous is not None and switching_penalty_m > 0.0:
            cost = cost + switching_penalty_m * (
                slot_targets[None, :] != previous[:, None])
        rows, cols = linear_sum_assignment_numpy(cost)
        step_labels = np.zeros(num_agents, dtype=np.int64)
        step_labels[rows] = slot_targets[cols]
        labels[step] = step_labels
        previous_labels[env_index] = step_labels.copy()

        delta = target_xy[step_labels] - uav_xy
        norm = np.linalg.norm(delta, axis=-1, keepdims=True)
        ratio = np.minimum(1.0, float(max_dp) / np.maximum(norm, 1e-12))
        movements[step] = (delta * ratio).astype(np.float32)

    device = global_states.device
    return (
        torch.as_tensor(labels.reshape(-1), dtype=torch.long, device=device),
        torch.as_tensor(
            movements.reshape(-1, 2), dtype=torch.float32, device=device),
    )


def build_differentiable_u2u_inbox(
    obs: torch.Tensor,
    sender_messages: torch.Tensor,
    actor,
    num_agents: int,
    rate_index: int,
    rate_bits: int,
    num_rate_levels: int,
    delay_teams: int = 0,
    transition_masks: Optional[torch.Tensor] = None,
    sender_token_masks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Inject a full-delivery, straight-through-quantized U2U inbox.

    This is used only by the CTDE assignment auxiliary. It supports either one
    aggregate latent token or ``Q`` target-aligned latent tokens per sender.
    The straight-through estimator lets receiver assignment and movement losses
    train sender content despite the physical-channel quantizer.
    """
    if obs.ndim != 2 or sender_messages.ndim != 2:
        raise ValueError('obs and sender_messages must be rank-2 tensors')
    if obs.shape[0] != sender_messages.shape[0]:
        raise ValueError('obs and sender_messages batch sizes must match')
    if obs.shape[0] % num_agents != 0:
        raise ValueError('batch must contain complete consecutive UAV teams')
    slices = getattr(actor, '_obs_slices', None)
    if slices is None:
        raise ValueError('actor must expose communication-token observation slices')
    tokens_per_sender = int(slices.comm_tokens_per_sender)
    payload_dim = int(sender_messages.shape[1])
    if payload_dim % max(tokens_per_sender, 1) != 0:
        raise ValueError(
            'message payload must divide evenly into sender tokens')
    content_dim = payload_dim // max(tokens_per_sender, 1)
    metadata_dim = int(slices.comm_token_per_sender) - content_dim
    if metadata_dim < 5:
        raise ValueError(
            'receiver token has insufficient physical-link metadata fields')
    token_len = (
        slices.comm_token_per_sender * tokens_per_sender * (num_agents - 1))
    if token_len <= 0 or slices.comm_mask_len <= 0:
        raise ValueError('actor observation does not contain communication tokens')

    teams = obs.shape[0] // num_agents
    messages = sender_messages.reshape(
        teams, num_agents, tokens_per_sender, content_dim)
    if sender_token_masks is None:
        token_masks = torch.ones(
            teams, num_agents, tokens_per_sender,
            dtype=obs.dtype, device=obs.device)
    else:
        if sender_token_masks.shape != (
                obs.shape[0], tokens_per_sender):
            raise ValueError(
                'sender_token_masks must have shape (batch, tokens_per_sender)')
        token_masks = sender_token_masks.to(
            dtype=obs.dtype, device=obs.device).reshape(
                teams, num_agents, tokens_per_sender)
    lag = max(int(delay_teams), 0)
    if lag > 0 and teams > lag:
        # Buffer rows are environment-interleaved. Shift by num_envs (or an
        # integer multiple) to reproduce the packet available at the next
        # decentralized policy decision, while preserving the gradient to the
        # earlier sender.
        messages = torch.cat([messages[:lag], messages[:-lag]], dim=0)
        token_masks = torch.cat(
            [token_masks[:lag], token_masks[:-lag]], dim=0)
    bits = max(int(rate_bits), 1)
    quant_levels = float(2 ** bits - 1)
    hard = (
        torch.round((messages.clamp(-1.0, 1.0) + 1.0)
                    * 0.5 * quant_levels)
        / quant_levels * 2.0 - 1.0
    )
    quantized = (
        messages + (hard - messages).detach()
    ) * token_masks.unsqueeze(-1)

    receiver_tokens = []
    receiver_masks = []
    for receiver in range(num_agents):
        rows = []
        mask_rows = []
        for sender in range(num_agents):
            if sender == receiver:
                continue
            metadata = torch.zeros(
                teams, tokens_per_sender, metadata_dim,
                dtype=obs.dtype, device=obs.device)
            metadata[:, :, 0] = sender / max(num_agents - 1, 1)
            if tokens_per_sender > 1 and metadata_dim >= 6:
                metadata[:, :, 1] = torch.linspace(
                    0.0, 1.0, tokens_per_sender,
                    dtype=obs.dtype, device=obs.device)
                metadata[:, :, 2] = (
                    rate_index / max(num_rate_levels - 1, 1))
            else:
                metadata[:, :, 1] = (
                    rate_index / max(num_rate_levels - 1, 1))
            # Ideal full-delivery teacher uses zero normalized latency/AoI and
            # a neutral SNR marker; physical variations are learned by PPO.
            sender_mask = token_masks[:, sender]
            rows.append(torch.cat(
                [quantized[:, sender],
                 metadata * sender_mask.unsqueeze(-1)], dim=-1))
            mask_rows.append(sender_mask)
        receiver_tokens.append(torch.cat(rows, dim=1))
        receiver_masks.append(torch.cat(mask_rows, dim=1))
    tokens = torch.stack(receiver_tokens, dim=1).reshape(
        obs.shape[0], -1)
    masks = torch.stack(receiver_masks, dim=1).reshape(
        obs.shape[0], slices.comm_mask_len)

    token_start = slices.comm_token_start
    token_end = token_start + token_len
    mask_start = slices.comm_mask_start
    mask_end = mask_start + slices.comm_mask_len
    injected = torch.cat([
        obs[:, :token_start],
        tokens,
        obs[:, token_end:mask_start],
        masks,
        obs[:, mask_end:],
    ], dim=-1)
    if lag > 0:
        replace_team = torch.zeros(
            teams, dtype=torch.bool, device=obs.device)
        if teams > lag:
            replace_team[lag:] = True
            if transition_masks is not None:
                team_masks = transition_masks.reshape(teams, num_agents)
                replace_team[lag:] &= team_masks[:-lag].amin(dim=1) > 0.5
        replace_rows = replace_team.repeat_interleave(num_agents).unsqueeze(-1)
        injected = torch.where(replace_rows, injected, obs)
    return injected


from uav_isac.agents.mappo_agent import MAPPOAgent
from uav_isac.agents.buffer import RolloutBuffer
from uav_isac.agents.networks import (
    split_param_groups, ATTENTION_PARAM_PREFIXES,
    ENCODER_PARAM_PREFIXES, HEAD_PARAM_PREFIXES,
)
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.utils.seeding import set_seed
from uav_isac.utils.types import Action


def build_aligned_critic_input(
    global_states: torch.Tensor,
    local_obs: torch.Tensor,
    flat_indices: np.ndarray,
    num_agents: int,
    centralized: bool,
    base_state_dim: int,
    comm_dim: int = 16,
) -> torch.Tensor:
    """Rebuild critic inputs in the exact rollout field order.

    Rollout values are evaluated on ``[base, agent_one_hot, comm_summary]``.
    The replay buffer stores ``[raw_global_state, comm_summary]`` once per
    environment transition.  PPO updates must split that stored vector before
    inserting agent identity; appending identity after the unsplit vector would
    silently train on ``[base, comm_summary, agent_one_hot]`` instead.

    For IPPO, ``base`` is the aligned local observation, but the physical
    communication summary still comes from the transition-level buffer rather
    than being replaced with zeros.
    """
    if global_states.ndim != 2 or local_obs.ndim != 2:
        raise ValueError("critic states and observations must be rank-2")
    if num_agents <= 0 or comm_dim <= 0 or base_state_dim <= 0:
        raise ValueError("critic dimensions and num_agents must be positive")
    indices = torch.as_tensor(
        flat_indices, dtype=torch.long, device=global_states.device)
    transition_indices = torch.div(
        indices, int(num_agents), rounding_mode='floor')
    stored = global_states[transition_indices]
    if stored.shape[-1] < comm_dim:
        raise ValueError("stored global state is shorter than comm summary")
    comm = stored[:, -comm_dim:]
    if centralized:
        if stored.shape[-1] < base_state_dim + comm_dim:
            raise ValueError(
                "stored global state cannot contain requested base and comm")
        base = stored[:, :base_state_dim]
    else:
        if local_obs.shape[0] != indices.numel():
            raise ValueError("local observations and flat indices are unaligned")
        if local_obs.shape[-1] != base_state_dim:
            raise ValueError(
                "local observation width does not match critic base state")
        base = local_obs
    agent_ids = torch.remainder(indices, int(num_agents))
    agent_one_hot = torch.nn.functional.one_hot(
        agent_ids, int(num_agents)).to(dtype=base.dtype)
    return torch.cat([base, agent_one_hot, comm.to(dtype=base.dtype)], dim=-1)


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
        self._structure_student = None
        self._structure_student_channel_enabled = False
        self._structure_student_bits_per_dim = 8
        self._structure_student_adaptive_min_bits_per_dim = 0
        self._structure_student_min_comm_fraction = 0.01
        self._structure_student_send_mode = "every_frame"
        self._structure_student_allow_scale_migration = False
        self._dynamic_local_search_config = None

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
        self._constrained_cvar_enabled = bool(getattr(
            ma, 'constrained_cvar_ppo_enabled', False))
        self._constrained_cvar_qos_floor = float(getattr(
            ma, 'constrained_cvar_qos_floor', 0.60))
        self._constrained_cvar_tail_fraction = float(getattr(
            ma, 'constrained_cvar_tail_fraction', 0.20))
        self._constrained_cvar_penalty = float(getattr(
            ma, 'constrained_cvar_penalty', 0.10))
        self._constrained_cvar_dual_step_size = float(getattr(
            ma, 'constrained_cvar_dual_step_size', 0.10))
        self._constrained_cvar_dual_maximum = float(getattr(
            ma, 'constrained_cvar_dual_maximum', 10.0))
        self._constrained_cvar_primal_tolerance = float(getattr(
            ma, 'constrained_cvar_primal_tolerance', 0.01))
        self._constrained_cvar_power_tolerance_w = float(getattr(
            ma, 'constrained_cvar_power_tolerance_w', 1.0e-9))
        self._constrained_cvar_convergence_patience = int(getattr(
            ma, 'constrained_cvar_convergence_patience', 3))
        self._constrained_cvar_multipliers = torch.zeros(
            self.Q, dtype=torch.float32, device=self.device)
        self._constrained_cvar_update_count = 0
        self._constrained_cvar_feasible_streak = 0
        self._constrained_cvar_first_feasible_update = None
        self._temporal_unrolled_power_enabled = bool(getattr(
            ma, 'temporal_unrolled_power_enabled', False))
        self._temporal_unrolled_power_horizon = int(getattr(
            ma, 'temporal_unrolled_power_horizon', 4))
        self._temporal_unrolled_power_inner_iterations = int(getattr(
            ma, 'temporal_unrolled_power_inner_iterations', 4))
        self._temporal_unrolled_power_warm_start_mix = float(getattr(
            ma, 'temporal_unrolled_power_warm_start_mix', 0.50))
        self._temporal_unrolled_power_detach_between_frames = bool(getattr(
            ma, 'temporal_unrolled_power_detach_between_frames', False))
        self._temporal_unrolled_power_batched_vjp_enabled = bool(getattr(
            ma, 'temporal_unrolled_power_batched_vjp_enabled', True))
        self._temporal_unrolled_power_rf_tangent_enabled = bool(getattr(
            ma, 'temporal_unrolled_power_rf_tangent_enabled', False))
        self._temporal_unrolled_power_rf_tangent_slack_tolerance = float(
            getattr(ma, 'temporal_unrolled_power_rf_tangent_slack_tolerance',
                    1.0e-6))
        self._temporal_unrolled_power_rf_tangent_rank_tolerance = float(
            getattr(ma, 'temporal_unrolled_power_rf_tangent_rank_tolerance',
                    1.0e-8))
        self._temporal_unrolled_power_pareto_tolerance = float(getattr(
            ma, 'temporal_unrolled_power_pareto_tolerance', 1.0e-6))
        self._temporal_unrolled_power_pareto_max_iterations = int(getattr(
            ma, 'temporal_unrolled_power_pareto_max_iterations', 64))
        self._temporal_feasible_structure_enabled = bool(getattr(
            ma, 'temporal_feasible_structure_enabled', False))
        self._temporal_feasible_structure_temperature = float(getattr(
            ma, 'temporal_feasible_structure_temperature', 0.25))
        self._temporal_feasible_structure_inertia = float(getattr(
            ma, 'temporal_feasible_structure_inertia', 0.25))
        self._temporal_feasible_structure_max_structures = int(getattr(
            ma, 'temporal_feasible_structure_max_structures', 4096))
        self._temporal_feasible_structure_hard_forward = bool(getattr(
            ma, 'temporal_feasible_structure_hard_forward', True))
        # CTDE (centralized critic, MAPPO) vs decentralized critic (IPPO).
        # Single source of truth = how the agent's critic was actually built.
        self.centralized_critic = getattr(agents[0], 'centralized_critic', True)
        self._set_risk_critic_enabled = bool(getattr(
            agents[0].critic, 'set_risk_critic_enabled', False))
        self._risk_critic_qos_floor = float(np.clip(getattr(
            ma, 'risk_critic_qos_floor', 0.60), 0.0, 1.0))
        self._risk_critic_quantile_coef = max(0.0, float(getattr(
            ma, 'risk_critic_quantile_coef', 0.25)))
        self._risk_critic_constraint_coef = max(0.0, float(getattr(
            ma, 'risk_critic_constraint_coef', 0.10)))
        self._risk_critic_constraint_positive_weight = float(getattr(
            ma, 'risk_critic_constraint_positive_weight', 0.0))
        self._risk_critic_only_training = bool(getattr(
            ma, 'risk_critic_only_training', False))
        if (self._risk_critic_only_training
                and not self._set_risk_critic_enabled):
            raise ValueError(
                'risk_critic_only_training requires set_risk_critic_enabled')

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
        self._eval_seed_bank_path = str(getattr(
            ma, 'eval_seed_bank_path', '')).strip()
        self._eval_seed_split = str(getattr(
            ma, 'eval_seed_split', 'selection')).strip()
        if self._eval_seed_bank_path:
            self.eval_seeds = load_stratified_seed_split(
                self._eval_seed_bank_path, self._eval_seed_split)
            self.eval_episodes = len(self.eval_seeds)
        self._checkpoint_confidence_alpha = float(np.clip(getattr(
            ma, 'checkpoint_confidence_alpha', 0.05), 1e-6, 0.499999))
        self._checkpoint_bootstrap_samples = max(1, int(getattr(
            ma, 'checkpoint_bootstrap_samples', 2000)))
        self._checkpoint_cvar_fraction = float(np.clip(getattr(
            ma, 'checkpoint_cvar_fraction', 0.20), 1e-6, 1.0))
        self._checkpoint_confirmation_enabled = bool(getattr(
            ma, 'checkpoint_confirmation_enabled', False))
        self._checkpoint_confirmation_split = str(getattr(
            ma, 'checkpoint_confirmation_split', 'confirmation')).strip()
        self._checkpoint_confirmation_seeds: List[int] = []
        if self._checkpoint_confirmation_enabled:
            if not self._eval_seed_bank_path:
                raise ValueError(
                    'checkpoint confirmation requires eval_seed_bank_path')
            self._checkpoint_confirmation_seeds = load_stratified_seed_split(
                self._eval_seed_bank_path,
                self._checkpoint_confirmation_split)
        self._training_seed_sampler = None
        self._training_fixed_seed = int(getattr(
            ma, 'training_fixed_seed', -1))
        if bool(getattr(ma, 'training_seed_replay_enabled', False)):
            training_bank_path = str(getattr(
                ma, 'training_seed_bank_path', '')).strip()
            if not training_bank_path:
                training_bank_path = self._eval_seed_bank_path
            if not training_bank_path:
                raise ValueError(
                    'training seed replay requires a seed-bank path')
            training_seeds, training_difficulties = load_training_seed_pool(
                training_bank_path,
                max_nearest_m=float(getattr(
                    ma, 'training_seed_max_nearest_m', 350.0)))
            self._training_seed_sampler = PrioritizedGeometrySeedSampler(
                training_seeds,
                training_difficulties,
                worst_floor=float(getattr(
                    ma, 'comm_qos_worst_min', 0.60)),
                uniform_mix=float(getattr(
                    ma, 'training_seed_uniform_mix', 0.20)),
                priority_alpha=float(getattr(
                    ma, 'training_seed_priority_alpha', 0.70)),
                ema=float(getattr(
                    ma, 'training_seed_priority_ema', 0.20)),
                min_curriculum_fraction=float(getattr(
                    ma, 'training_seed_min_curriculum_fraction', 0.30)),
                curriculum_frames=int(getattr(
                    ma, 'training_seed_curriculum_frames', 300000)),
            )
        self.best_score = -float('inf')
        self.best_params = None
        self.best_runtime_state = None
        self.last_unrestored_params = None
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

        self._training_current_seeds: List[Optional[int]] = [
            None] * self.num_envs
        self._training_episode_worst_sum = np.zeros(
            self.num_envs, dtype=np.float64)
        self._training_episode_steps = np.zeros(
            self.num_envs, dtype=np.int64)

        # ── Fix #6: persistent env state across rollouts ──
        self._current_obs = [None] * self.num_envs

        # Steps per env to maintain total transitions ~= rollout_steps
        self.steps_per_env = max(1, self.rollout_steps // self.num_envs)
        self.macro_interval = getattr(ma, 'actor_decision_interval', 1)
        self.movement_decision_interval = max(1, int(getattr(
            ma, 'movement_decision_interval', 1)))
        self._joint_isac_power_enabled = bool(getattr(
            ma, 'joint_isac_power_enabled', False))
        if self._joint_isac_power_enabled and self.macro_interval != 1:
            raise ValueError(
                'joint ISAC power allocation requires actor_decision_interval=1; '
                'each simulator frame is one communication round')
        if self.movement_decision_interval > 1 and self.macro_interval != 1:
            raise ValueError(
                'movement_decision_interval>1 requires actor_decision_interval=1; '
                'fast communication/resource decisions must remain frame-level')
        # Per-environment movement commitments persist across rollout chunks.
        # Communication and resource actions are never held by this state.
        self._movement_phase = np.zeros(self.num_envs, dtype=np.int64)
        self._held_movement_dp = np.zeros(
            (self.num_envs, self.K, 2), dtype=np.float64)
        self._held_movement_role = np.zeros(
            (self.num_envs, self.K), dtype=np.int32)
        self._held_movement_valid = np.zeros(
            self.num_envs, dtype=bool)
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
            comm_dim=int(getattr(agents[0], 'comm_payload_dim', 16)),
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
        self._comm_cost_aware = (self._comm_mode == 'cost_aware')
        self._headwise_credit_enabled = bool(getattr(
            ma, 'headwise_credit_enabled', False))
        if self._headwise_credit_enabled and not self._comm_cost_aware:
            raise ValueError(
                'headwise_credit_enabled requires learned_comm_mode=cost_aware')
        self._headwise_credit_vf_coef = max(0.0, float(getattr(
            ma, 'headwise_credit_vf_coef', 0.5)))
        self._headwise_credit_coefs = np.asarray([
            float(getattr(ma, 'headwise_movement_coef', 1.0)),
            float(getattr(ma, 'headwise_message_coef', 1.0)),
            float(getattr(ma, 'headwise_rate_coef', 1.0)),
            float(getattr(ma, 'headwise_resource_coef', 1.0)),
        ], dtype=np.float64)
        if np.any(self._headwise_credit_coefs < 0.0):
            raise ValueError('headwise credit coefficients must be non-negative')
        self._causal_ccp_enabled = bool(getattr(
            ma, 'causal_ccp_enabled', False))
        if self._causal_ccp_enabled and not self._headwise_credit_enabled:
            raise ValueError(
                'causal_ccp_enabled requires headwise_credit_enabled')
        if (self._causal_ccp_enabled
                and str(getattr(ma, 'comm_payload_mode', '')).lower()
                != 'target_tokens'):
            raise ValueError(
                'causal CCP currently requires target-token communication')
        self._causal_ccp_stride = max(1, int(getattr(
            ma, 'causal_ccp_intervention_stride', 4)))
        self._causal_ccp_epochs = max(1, int(getattr(
            ma, 'causal_ccp_epochs', 6)))
        self._causal_ccp_tail_temperature = max(1e-3, float(getattr(
            ma, 'causal_ccp_tail_temperature', 0.10)))
        self._causal_ccp_message_mix = float(np.clip(getattr(
            ma, 'causal_ccp_message_mix', 0.75), 0.0, 1.0))
        self._causal_ccp_min_samples = max(4, int(getattr(
            ma, 'causal_ccp_min_samples', 32)))
        self._causal_ccp_min_effect = max(0.0, float(getattr(
            ma, 'causal_ccp_min_effect', 1e-5)))
        self._causal_ccp = None
        self._causal_ccp_optimizer = None
        if self._causal_ccp_enabled:
            self._causal_ccp = CausalContributionPredictor(
                obs_dim=obs_dim,
                comm_dim=int(getattr(agents[0], 'comm_payload_dim', 16)),
                num_targets=self.Q,
                num_rate_levels=len(getattr(
                    ma, 'comm_rate_bits_per_dim', [0, 4, 8, 16])),
                hidden_dim=int(getattr(ma, 'causal_ccp_hidden_dim', 128)),
            ).to(self.device)
            self._causal_ccp_optimizer = torch.optim.Adam(
                self._causal_ccp.parameters(),
                lr=max(1e-7, float(getattr(ma, 'causal_ccp_lr', 3e-4))))
        self._comm_qos_enabled = (
            self._comm_cost_aware
            and bool(getattr(ma, 'comm_qos_constrained', False))
        )
        self._comm_qos_targets = np.array([
            float(getattr(ma, 'comm_qos_steady_min', 0.80)),
            float(getattr(ma, 'comm_qos_weak3_min', 0.70)),
            float(getattr(ma, 'comm_qos_worst_min', 0.60)),
        ], dtype=np.float64)
        qos_init = float(getattr(ma, 'comm_qos_lambda_init', 0.5))
        self._comm_qos_lambdas = np.full(3, qos_init, dtype=np.float64)
        self._comm_qos_dual_lr = float(getattr(ma, 'comm_qos_dual_lr', 0.05))
        self._comm_qos_lambda_max = float(getattr(
            ma, 'comm_qos_lambda_max', 5.0))
        self._comm_qos_reward_scale = float(getattr(
            ma, 'comm_qos_reward_scale', 1.0))
        self._comm_encouragement_enabled = bool(getattr(
            ma, 'comm_encouragement_enabled', True))
        self._comm_encouragement_weight = float(getattr(
            ma, 'comm_encouragement_weight', 0.05))
        self._comm_encouragement_floor_ratio = float(getattr(
            ma, 'comm_encouragement_floor_ratio', 0.10))
        self._comm_rate_bonus_enabled = bool(getattr(
            ma, 'comm_rate_bonus_enabled', True))
        self._comm_rate_bonus_weight = max(0.0, float(getattr(
            ma, 'comm_rate_bonus_weight', 0.03)))
        self._comm_rate_bonus_target_bits = max(1, int(getattr(
            ma, 'comm_rate_bonus_target_bits', 8)))
        self._comm_rate_bonus_aux_coef = max(0.0, float(getattr(
            ma, 'comm_rate_bonus_aux_coef', 1.0)))
        self._comm_rate_bonus_aux_lr = max(0.0, float(getattr(
            ma, 'comm_rate_bonus_aux_lr', 0.01)))
        self._comm_sender_delivery_penalty_enabled = bool(getattr(
            ma, 'comm_sender_delivery_penalty_enabled', False))
        self._comm_sender_delivery_penalty_weight = max(0.0, float(getattr(
            ma, 'comm_sender_delivery_penalty_weight', 0.05)))
        self._comm_silence_penalty_enabled = bool(getattr(
            ma, 'comm_silence_penalty_enabled', True))
        self._comm_silence_grace_decisions = max(0, int(getattr(
            ma, 'comm_silence_grace_decisions', 3)))
        self._comm_silence_penalty_per_decision = max(0.0, float(getattr(
            ma, 'comm_silence_penalty_per_decision', 0.01)))
        self._comm_silence_penalty_max = max(0.0, float(getattr(
            ma, 'comm_silence_penalty_max', 0.05)))
        self._comm_silence_streaks = np.zeros(
            (self.num_envs, self.K), dtype=np.int64)
        self._comm_eval_force_silence = bool(getattr(
            ma, 'comm_eval_force_silence', False))
        self._comm_eval_force_rate_bits = int(getattr(
            ma, 'comm_eval_force_rate_bits', -1))
        if self._comm_eval_force_rate_bits >= 0:
            configured_rates = list(getattr(
                ma, 'comm_rate_bits_per_dim', [0, 4, 8, 16]))
            if self._comm_eval_force_rate_bits not in configured_rates:
                raise ValueError(
                    'comm_eval_force_rate_bits must match a configured '
                    'comm_rate_bits_per_dim entry')
            self._comm_eval_force_rate_index = configured_rates.index(
                self._comm_eval_force_rate_bits)
        else:
            self._comm_eval_force_rate_index = -1
        self._comm_eval_message_ablation = str(getattr(
            ma, 'comm_eval_message_ablation', 'none')).lower()
        if self._comm_eval_message_ablation not in {
                'none', 'zero', 'permute'}:
            raise ValueError(
                'comm_eval_message_ablation must be none/zero/permute')
        self._eval_centralized_assignment_movement = bool(getattr(
            ma, 'eval_centralized_assignment_movement', False))
        self._eval_qos_bistatic_assignment_movement = bool(getattr(
            ma, 'eval_qos_bistatic_assignment_movement', False))
        if (self._eval_centralized_assignment_movement
                and self._eval_qos_bistatic_assignment_movement):
            raise ValueError(
                'only one evaluation movement intervention may be enabled')
        self._comm_qos_min_rate_bits = int(getattr(
            ma, 'comm_qos_min_rate_bits', 0))
        self._comm_qos_rate_shortfall_penalty = float(getattr(
            ma, 'comm_qos_rate_shortfall_penalty', 0.0))
        self._comm_qos_rate_aux_coef = float(getattr(
            ma, 'comm_qos_rate_aux_coef', 0.0))
        self._best_comm_qos_key = None
        self._best_selection_screen_key = None
        self._target_allocation_enabled = bool(getattr(
            ma, 'target_allocation_enabled', False))
        self._target_allocation_balance_coef = float(getattr(
            ma, 'target_allocation_balance_coef', 0.20))
        self._target_allocation_commit_coef = float(getattr(
            ma, 'target_allocation_commit_coef', 0.01))
        self._target_allocation_differentiable_comm = bool(getattr(
            ma, 'target_allocation_differentiable_comm', False))
        self._target_allocation_comm_delay_decisions = max(0, int(getattr(
            ma, 'target_allocation_comm_delay_decisions', 1)))
        self._target_allocation_temporal_coef = max(0.0, float(getattr(
            ma, 'target_allocation_temporal_coef', 0.0)))
        self._target_allocation_aux_epochs = max(1, int(getattr(
            ma, 'target_allocation_aux_epochs', 1)))
        self._target_allocation_aux_lr = max(0.0, float(getattr(
            ma, 'target_allocation_aux_lr', 0.0)))
        self._capacity_matching_enabled = bool(getattr(
            ma, 'capacity_matching_enabled', False))
        self._capacity_matching_blend_start = float(np.clip(getattr(
            ma, 'capacity_matching_blend_start', 0.0), 0.0, 1.0))
        self._capacity_matching_blend_end = float(np.clip(getattr(
            ma, 'capacity_matching_blend_end', 1.0), 0.0, 1.0))
        self._capacity_matching_anneal_frames = max(1, int(getattr(
            ma, 'capacity_matching_anneal_frames', 100000)))
        if (self._capacity_matching_enabled
                and not hasattr(agents[0].actor,
                                'set_capacity_matching_blend')):
            raise ValueError(
                'capacity matching requires a compatible structured actor')
        self._target_allocation_movement_only = bool(getattr(
            ma, 'target_allocation_movement_only', False))
        self._target_movement_blend_anneal_frames = max(0, int(getattr(
            ma, 'target_allocation_movement_blend_anneal_frames', 0)))
        self._target_movement_blend_schedule_enabled = (
            self._target_movement_blend_anneal_frames > 0)
        self._target_movement_blend_start = float(np.clip(getattr(
            ma, 'target_allocation_movement_blend_start', 0.0), 0.0, 1.0))
        self._target_movement_blend_end = float(np.clip(getattr(
            ma, 'target_allocation_movement_blend_end', 0.0), 0.0, 1.0))
        if (self._target_movement_blend_schedule_enabled
                and not hasattr(
                    agents[0].actor,
                    'set_target_allocation_movement_blend')):
            raise ValueError(
                'movement-blend schedule requires a compatible structured actor')
        self._sparse_claim_enabled = bool(getattr(
            ma, 'sparse_claim_enabled', False))
        self._sparse_claim_commit_topk = max(1, int(getattr(
            ma, 'sparse_claim_commit_topk', 2)))
        self._sparse_claim_desired_endpoints = max(1, int(getattr(
            ma, 'sparse_claim_desired_endpoints', 2)))
        self._adaptive_topk_from_rate_enabled = bool(getattr(
            ma, 'adaptive_topk_from_rate_enabled', False))
        self._adaptive_topk_rate_mapping = [int(value) for value in getattr(
            ma, 'adaptive_topk_rate_mapping', [0, 1, 2])]
        self._adaptive_topk_worst_credit_coef = max(0.0, float(getattr(
            ma, 'adaptive_topk_worst_credit_coef', 0.0)))
        if self._adaptive_topk_from_rate_enabled:
            rate_count = len(getattr(
                ma, 'comm_rate_bits_per_dim', [0, 4, 8, 16]))
            if not self._sparse_claim_enabled:
                raise ValueError('adaptive top-k requires sparse claims')
            if len(self._adaptive_topk_rate_mapping) != rate_count:
                raise ValueError(
                    'adaptive top-k mapping must match communication rates')
            if self._adaptive_topk_rate_mapping[0] != 0:
                raise ValueError('silent rate must map to k=0')
            if any(value < 0 or value > self.Q
                   for value in self._adaptive_topk_rate_mapping):
                raise ValueError('adaptive k must be between zero and Q')
        self._sparse_claim_underload_coef = max(0.0, float(getattr(
            ma, 'sparse_claim_underload_coef', 0.0)))
        self._sparse_claim_overload_coef = max(0.0, float(getattr(
            ma, 'sparse_claim_overload_coef', 0.0)))
        self._comm_aided_sensing_enabled = bool(getattr(
            ma, 'comm_aided_sensing_enabled', False))
        self._comm_aided_sensing_aux_coef = max(0.0, float(getattr(
            ma, 'comm_aided_sensing_aux_coef', 0.0)))
        self._comm_aided_sensing_counterfactual_coef = max(
            0.0, float(getattr(
                ma, 'comm_aided_sensing_counterfactual_coef', 0.0)))
        self._comm_aided_sensing_temperature = max(1e-3, float(getattr(
            ma, 'comm_aided_sensing_temperature', 0.25)))
        self._comm_aided_sensing_margin = max(0.0, float(getattr(
            ma, 'comm_aided_sensing_margin', 0.02)))
        self._comm_semantic_cacsr_eval_enabled = bool(getattr(
            ma, 'comm_semantic_cacsr_eval_enabled', False))
        self._comm_semantic_cacsr_eval_gain = max(0.0, float(getattr(
            ma, 'comm_semantic_cacsr_eval_gain', 0.0)))
        self._comm_semantic_cacsr_quality_threshold = float(np.clip(getattr(
            ma, 'comm_semantic_cacsr_quality_threshold', 0.97), 0.0, 1.0))
        self._comm_semantic_cacsr_stagnation_frames = max(1, int(getattr(
            ma, 'comm_semantic_cacsr_stagnation_frames', 5)))
        self._comm_semantic_cacsr_improvement_epsilon = max(0.0, float(getattr(
            ma, 'comm_semantic_cacsr_improvement_epsilon', 0.01)))
        self._comm_semantic_cacsr_ema_alpha = float(np.clip(getattr(
            ma, 'comm_semantic_cacsr_ema_alpha', 0.30), 1e-3, 1.0))
        if (self._comm_semantic_cacsr_eval_enabled
                and not bool(getattr(ma, 'comm_semantic_decoder_enabled', False))):
            raise ValueError('CA-CSR evaluation requires semantic decoder')
        self._target_allocation_teacher_enabled = bool(getattr(
            ma, 'target_allocation_teacher_enabled', False))
        self._target_allocation_teacher_label_coef = float(getattr(
            ma, 'target_allocation_teacher_label_coef', 0.20))
        self._target_allocation_teacher_movement_coef = float(getattr(
            ma, 'target_allocation_teacher_movement_coef', 0.50))
        self._target_allocation_teacher_message_coef = float(getattr(
            ma, 'target_allocation_teacher_message_coef', 0.05))
        self._target_allocation_teacher_epochs = max(1, int(getattr(
            ma, 'target_allocation_teacher_epochs', 1)))
        self._target_allocation_teacher_differentiable_comm = bool(getattr(
            ma, 'target_allocation_teacher_differentiable_comm', False))
        self._target_allocation_teacher_switching_penalty_m = max(
            0.0, float(getattr(
                ma, 'target_allocation_teacher_switching_penalty_m', 0.0)))
        self._target_allocation_teacher_mode = str(getattr(
            ma, 'target_allocation_teacher_mode', 'distance')).strip().lower()
        if self._target_allocation_teacher_mode not in {
                'distance', 'qos_bistatic'}:
            raise ValueError(
                'target_allocation_teacher_mode must be distance or qos_bistatic')
        self._target_allocation_teacher_qos_floor = float(np.clip(getattr(
            ma, 'target_allocation_teacher_qos_floor', 0.60), 1e-6, 1.0))
        self._target_allocation_teacher_qos_weight = max(0.0, float(getattr(
            ma, 'target_allocation_teacher_qos_weight', 2.0)))
        self._target_allocation_teacher_commitment_frames = max(1, int(getattr(
            ma, 'target_allocation_teacher_commitment_frames',
            self.movement_decision_interval)))
        self._target_allocation_teacher_height_m = max(0.0, float(getattr(
            ma, 'target_allocation_teacher_height_m',
            self.cfg.scenario.height)))
        self._target_allocation_teacher_crisis_only_enabled = bool(getattr(
            ma, 'target_allocation_teacher_crisis_only_enabled', False))
        self._target_allocation_teacher_crisis_floor = float(np.clip(getattr(
            ma, 'target_allocation_teacher_crisis_floor', 0.60),
            0.0, 1.0))
        self._v2_modular_balance_coef = max(0.0, float(getattr(
            ma, 'architecture_v2_modular_balance_coef', 0.0)))
        self._v2_modular_specialization_coef = max(0.0, float(getattr(
            ma, 'architecture_v2_modular_specialization_coef', 0.0)))
        self._v2_modular_lr_scale = max(0.0, float(getattr(
            ma, 'architecture_v2_modular_lr_scale', 1.0)))
        self._adv_mode = getattr(ma, 'advantage_mode', 'scalar')
        self._resp_tau_m = getattr(ma, 'target_responsibility_tau_m', 50.0)
        self._risk_tail_fraction = float(np.clip(getattr(
            ma, 'risk_tail_fraction', 0.50), 1e-6, 1.0))
        self._risk_target_temperature = max(1e-4, float(getattr(
            ma, 'risk_target_temperature', 0.10)))
        self._risk_target_floor = float(getattr(
            ma, 'risk_target_floor', 0.60))
        self._risk_scalar_mix = float(np.clip(getattr(
            ma, 'risk_scalar_mix', 0.25), 0.0, 1.0))
        freeze_attn = getattr(ma, 'freeze_attention', False)
        use_per_lr = getattr(ma, 'use_per_module_lr', False)

        # Prefer actor-provided parameter groups (e.g. TICA adapter).
        # Fall back to name-based prefix matching for legacy actors.
        if hasattr(agents[0].actor, 'parameter_groups'):
            pg = agents[0].actor.parameter_groups()
            enc_params = pg.get('encoder', [])
            attn_params = pg.get('attention', [])
            head_params = pg.get('head', [])
        else:
            enc_params, head_params, attn_params = split_param_groups(
                agents[0].actor.named_parameters())

        if self._comm_off:
            # Freeze all communication-related heads
            comm_head_names = [
                'comm_head.', 'comm_target_token_head.',
                'comm_rate_head.', 'comm_set_rate_head.',
                'comm_set_attention.', 'comm_log_std',
                'isac_power_mean_head.', 'isac_set_power_head.',
                'isac_sensing_mean_head.',
                'isac_power_log_std', 'isac_sensing_log_std',
                'comm_proj.', 'gate.', 'intent_head.', 'comm_token_enc.',
                'comm_cross_attn.', 'comm_cross_norm.', 'comm_target_gate.',
                'round_phase_enc.', 'round_phase_equivariant_enc.',
                'round_target_gate.',
                'round_peer_claim_head.',
            ]
            for n, p in agents[0].actor.named_parameters():
                if any(n.startswith(prefix) for prefix in comm_head_names):
                    p.requires_grad_(False)

        if (self._adaptive_topk_from_rate_enabled
                and bool(getattr(
                    ma, 'adaptive_topk_rate_only_training', False))):
            for name, parameter in agents[0].actor.named_parameters():
                parameter.requires_grad_(
                    name.startswith((
                        'comm_rate_head.', 'comm_set_rate_head.',
                        'comm_rate_feedback_head.')))

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
            # Audit 2026-08-25: the episode LR scheduler below used to overwrite
            # every param-group lr, silently erasing this per-module isolation
            # (enc 1e-5 / attn 0-or-1e-5 / head 5e-5) from the first scheduled
            # update onward.  Record the explicit per-group lrs so the scheduler
            # preserves them (they are an ablation contract, not a default).
            self._actor_param_group_lr = {
                id(g): float(g['lr'])
                for g in agents[0].actor_optimizer.param_groups}
            frozen_count = sum(1 for p in agents[0].actor.parameters() if not p.requires_grad)
            print(f'[S1] comm={self._comm_mode} freeze_attn={freeze_attn} '
                  f'per_module_lr={use_per_lr or freeze_attn} '
                  f'enc_lr=1e-5 attn_lr={attn_lr} head_lr=5e-5 '
                  f'frozen={frozen_count} params')
        else:
            # Default single-LR optimizer: the scheduler may overwrite lr freely.
            self._actor_param_group_lr = None

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

        # Calibration stage: the behaviour policy and all legacy value heads
        # remain bitwise fixed.  Only the cardinality-independent risk branch
        # receives gradients, so changing the actor cannot make the labels
        # non-stationary or create a false performance gain.
        self._risk_critic_actor_checksum = None
        if self._risk_critic_only_training:
            for parameter in agents[0].actor.parameters():
                parameter.requires_grad_(False)
            risk_parameters = []
            for name, parameter in agents[0].critic.named_parameters():
                is_risk_parameter = name.startswith('risk_')
                parameter.requires_grad_(is_risk_parameter)
                if is_risk_parameter:
                    risk_parameters.append(parameter)
            if not risk_parameters:
                raise RuntimeError(
                    'risk-critic-only training found no risk parameters')
            critic_lr = float(
                agents[0].critic_optimizer.param_groups[0]['lr'])
            risk_optimizer = torch.optim.Adam(
                risk_parameters, lr=critic_lr)
            actor_checksum = tuple(
                float(parameter.detach().double().sum().cpu())
                for parameter in agents[0].actor.parameters())
            self._risk_critic_actor_checksum = actor_checksum
            for agent in agents:
                agent.actor_optimizer = agents[0].actor_optimizer
                agent.critic_optimizer = risk_optimizer

        # A rate-head-only optimizer makes precision exploration effective
        # without perturbing message semantics, attention, or movement. It is
        # stepped only while QoS remains infeasible.
        self._comm_rate_aux_optimizer = None
        self._comm_rate_head_module = (
            agents[0].actor.comm_set_rate_head
            if getattr(
                agents[0].actor,
                '_scale_equivariant_comm_heads_enabled', False)
            else getattr(agents[0].actor, 'comm_rate_head', None))
        if (self._comm_cost_aware
                and self._comm_rate_bonus_enabled
                and self._comm_rate_bonus_aux_lr > 0.0
                and self._comm_rate_head_module is not None):
            self._comm_rate_aux_optimizer = torch.optim.SGD(
                self._comm_rate_head_module.parameters(),
                lr=self._comm_rate_bonus_aux_lr)

        # Optional higher-rate auxiliary optimizer for the latent commitment
        # path. These parameters also receive PPO gradients; the extra step is
        # restricted to message/assignment modules so a few diagnostic updates
        # can escape the uniform-assignment stationary point without rewriting
        # the warm-started physical encoders.
        self._target_allocation_aux_optimizer = None
        if (self._target_allocation_enabled
                and self._target_allocation_aux_lr > 0.0):
            allocation_prefixes = (
                'target_assignment_head.', 'allocation_proj.',
                'movement_commitment_head.',
                'allocation_gate.', 'comm_head.',
                'comm_target_token_head.', 'comm_token_enc.',
                'comm_cross_attn.', 'comm_cross_norm.',
                'comm_target_gate.', 'intent_head.',
                'neighbor_bid_msg_proj.',
                'neighbor_bid_target_proj.',
                'round_phase_enc.', 'round_phase_equivariant_enc.',
                'round_target_gate.',
                'round_peer_claim_head.',
                'comm_sensing_gate.', 'comm_sensing_head.',
                'target_movement_head.',
                # Architecture V2 slow intent is deliberately separated from
                # its shared target encoder and sensing head.  CTDE
                # distillation may update bids and kinematics without moving
                # the physical/QoS prior or executed sensing-power policy.
                'v2_assignment_head.',
                'v2_movement_head.',
                'v2_target_attention.',
                'v2_target_context.',
                'v2_movement_gate.',
                'v2_module_router.',
                'v2_coordination_experts.',
            )
            allocation_params = [
                p for name, p in agents[0].actor.named_parameters()
                if p.requires_grad
                and any(name.startswith(prefix)
                        for prefix in allocation_prefixes)
            ]
            if allocation_params:
                modular_prefixes = (
                    'v2_module_router.',
                    'v2_coordination_experts.',
                )
                modular_params = [
                    p for name, p in agents[0].actor.named_parameters()
                    if p.requires_grad
                    and any(name.startswith(prefix)
                            for prefix in modular_prefixes)
                ]
                modular_param_ids = {id(p) for p in modular_params}
                regular_params = [
                    p for p in allocation_params
                    if id(p) not in modular_param_ids]
                optimizer_groups = []
                if regular_params:
                    optimizer_groups.append({
                        'params': regular_params,
                        'lr': self._target_allocation_aux_lr,
                    })
                if modular_params:
                    optimizer_groups.append({
                        'params': modular_params,
                        'lr': (
                            self._target_allocation_aux_lr
                            * self._v2_modular_lr_scale),
                    })
                self._target_allocation_aux_optimizer = torch.optim.Adam(
                    optimizer_groups)

    def load_frozen_structure_student(
        self,
        checkpoint: str,
        *,
        channel_enabled: bool = False,
        bits_per_dim: int = 8,
        adaptive_min_bits_per_dim: int = 0,
        min_comm_fraction: float = 0.01,
        send_mode: str = "every_frame",
        allow_scale_migration: bool = False,
    ) -> None:
        """Load an inference-only local-information structure surrogate."""
        from uav_isac.agents.frozen_structure_student import (
            FrozenStructureStudent,
        )

        student = FrozenStructureStudent(checkpoint, device=str(self.device))
        cardinality_mismatch = (
            student.num_uavs != self.K or student.num_targets != self.Q)
        if cardinality_mismatch and not allow_scale_migration:
            raise ValueError(
                "structure-student K/Q does not match the environment")
        if cardinality_mismatch:
            scale_safe_actor = all((
                bool(getattr(self.cfg.marl, 'architecture_v2_enabled', False)),
                bool(getattr(
                    self.cfg.marl,
                    'scale_equivariant_comm_heads_enabled', False)),
                bool(getattr(
                    self.cfg.marl,
                    'permutation_equivariant_round_encoding_enabled', False)),
                bool(getattr(
                    self.cfg.marl,
                    'equivariant_value_critic_enabled', False)),
                bool(getattr(student, 'cardinality_equivariant', False)),
            ))
            if not scale_safe_actor:
                raise ValueError(
                    "cross-scale structure-student evaluation requires the "
                    "cardinality-equivariant Actor, communication heads, "
                    "round encoding, Critic and Student")
        self._structure_student = student
        self._structure_student_channel_enabled = bool(channel_enabled)
        self._structure_student_bits_per_dim = max(
            1, int(bits_per_dim))
        adaptive_min = int(adaptive_min_bits_per_dim)
        self._structure_student_adaptive_min_bits_per_dim = (
            int(np.clip(
                adaptive_min,
                1,
                self._structure_student_bits_per_dim,
            ))
            if adaptive_min > 0 else 0
        )
        self._structure_student_min_comm_fraction = float(np.clip(
            min_comm_fraction, 0.0, 1.0))
        send_mode = str(send_mode).strip().lower()
        if send_mode not in {"every_frame", "p0_resolve"}:
            raise ValueError(
                "structure-student send mode must be every_frame or "
                "p0_resolve")
        self._structure_student_send_mode = send_mode
        self._structure_student_allow_scale_migration = bool(
            allow_scale_migration)

    def configure_dynamic_local_search(
        self,
        mode: str,
        *,
        ranker_checkpoint: str | None = None,
        cold_initializer: str = "role_first",
        factor_graph_checkpoint: str | None = None,
        neighbor_topk: int = 4,
        target_topk: int = 4,
        coverage_fraction: float = 0.30,
        cold_rounds: int = 8,
        warm_rounds: int = 8,
        warm_top_m: int = 3,
        rebootstrap_mode: str = "off",
        rebootstrap_interval_frames: int = 0,
        rebootstrap_rounds: int = 8,
        rebootstrap_require_deficit: bool = False,
    ) -> None:
        """Configure the dynamic Gate C1.7 evaluation controller."""
        if self._structure_student is None:
            raise ValueError(
                "dynamic local search requires a frozen structure Student")
        if self._structure_student_channel_enabled:
            raise ValueError(
                "C1.7a local candidate construction currently requires the "
                "information-equivalent Student graph")
        normalized = str(mode).strip().lower()
        if normalized not in {"previous", "oracle", "hybrid", "replicated"}:
            raise ValueError("unknown dynamic local-search mode")
        if normalized == "hybrid" and not ranker_checkpoint:
            raise ValueError("hybrid mode requires a local move ranker")
        cold_initializer = str(cold_initializer).strip().lower()
        if cold_initializer not in {"role_first", "factor_graph"}:
            raise ValueError("unknown dynamic cold initializer")
        if cold_initializer == "factor_graph" and not factor_graph_checkpoint:
            raise ValueError(
                "factor_graph cold initialization requires a checkpoint")
        rebootstrap_mode = str(rebootstrap_mode).strip().lower()
        if rebootstrap_mode not in {"off", "periodic", "oracle"}:
            raise ValueError("unknown dynamic rebootstrap mode")
        if (rebootstrap_mode == "periodic"
                and int(rebootstrap_interval_frames) <= 0):
            raise ValueError(
                "periodic rebootstrap requires a positive frame interval")
        self._dynamic_local_search_config = {
            "mode": normalized,
            "ranker_checkpoint": ranker_checkpoint,
            "cold_initializer": cold_initializer,
            "factor_graph_checkpoint": factor_graph_checkpoint,
            "neighbor_topk": max(0, int(neighbor_topk)),
            "target_topk": max(0, int(target_topk)),
            "coverage_fraction": float(np.clip(
                coverage_fraction, 0.0, 1.0)),
            "cold_rounds": max(0, int(cold_rounds)),
            "warm_rounds": max(0, int(warm_rounds)),
            "warm_top_m": max(1, int(warm_top_m)),
            "rebootstrap_mode": rebootstrap_mode,
            "rebootstrap_interval_frames": max(
                0, int(rebootstrap_interval_frames)),
            "rebootstrap_rounds": max(0, int(rebootstrap_rounds)),
            "rebootstrap_require_deficit": bool(
                rebootstrap_require_deficit),
        }

    def _effective_comm(self, comm_msgs: torch.Tensor) -> torch.Tensor:
        """Return zeroed comm if comm_off, else original. Single entry point."""
        if self._comm_off:
            return torch.zeros_like(comm_msgs)
        return comm_msgs

    def _critic_comm_summary(self, comm: torch.Tensor) -> torch.Tensor:
        """Reduce arbitrary token payloads to the critic's stable 16-D slot."""
        if comm.shape[-1] == 16:
            return comm
        if (comm.shape[-1] % max(self.Q, 1) == 0
                and str(getattr(self.cfg.marl, 'comm_payload_mode',
                                'aggregate')).lower() == 'target_tokens'):
            comm = comm.reshape(*comm.shape[:-1], self.Q, -1).mean(dim=-2)
        if comm.shape[-1] > 16:
            return comm[..., :16]
        return torch.nn.functional.pad(comm, (0, 16 - comm.shape[-1]))

    def _critic_comm_summary_np(self, comm: np.ndarray) -> np.ndarray:
        arr = np.asarray(comm)
        if arr.shape[-1] == 16:
            return arr
        if (arr.shape[-1] % max(self.Q, 1) == 0
                and str(getattr(self.cfg.marl, 'comm_payload_mode',
                                'aggregate')).lower() == 'target_tokens'):
            arr = arr.reshape(*arr.shape[:-1], self.Q, -1).mean(axis=-2)
        if arr.shape[-1] >= 16:
            return arr[..., :16]
        return np.pad(arr, [(0, 0)] * (arr.ndim - 1)
                      + [(0, 16 - arr.shape[-1])])

    def _simulate_deterministic_next_qos(
        self,
        env,
        observations: Dict[str, np.ndarray],
        env_index: int,
        next_movement_phase: int,
    ) -> np.ndarray:
        """One-step common-random-number branch for causal token labels."""
        branch = deepcopy(env)
        actor = self.agents[0].actor
        ob = np.stack([observations[str(k)] for k in range(self.K)])

        h_rows = []
        for receiver in range(self.K):
            for neighbor in range(self.K):
                if neighbor == receiver:
                    continue
                hidden = branch.core._gru_hidden.get((receiver, neighbor))
                if hidden is None:
                    hidden = np.zeros(
                        self.buffer.gru_hidden_dim, dtype=np.float32)
                h_rows.append(hidden)
        h_prev = None
        if h_rows:
            h_prev = torch.as_tensor(
                np.stack(h_rows), dtype=torch.float32,
                device=self.device).unsqueeze(0)

        actor_input = torch.as_tensor(
            ob, dtype=torch.float32, device=self.device)
        window_mask = None
        if self._use_window:
            branch_window = self._obs_ring[env_index].copy()
            branch_mask = self._window_mask[env_index].copy()
            branch_window[:, :-1] = branch_window[:, 1:]
            branch_window[:, -1] = ob
            branch_mask[:, :-1] = branch_mask[:, 1:]
            branch_mask[:, -1] = True
            actor_input = torch.as_tensor(
                branch_window, dtype=torch.float32, device=self.device)
            window_mask = torch.as_tensor(
                branch_mask, dtype=torch.bool, device=self.device)

        with torch.inference_mode():
            phase = float(next_movement_phase) / max(
                self.movement_decision_interval - 1, 1)
            phase_t = torch.full(
                (self.K,), phase, dtype=torch.float32, device=self.device)
            identity = torch.arange(
                self.K, dtype=torch.long, device=self.device)
            dp_mean, dp_log_std, role_logits, comm_mean, _, _ = actor(
                actor_input, h_prev, window_mask=window_mask,
                comm_round_phase=phase_t, agent_identity=identity)
            outgoing_token_mask = getattr(
                actor, 'last_outgoing_token_mask', None)
            if outgoing_token_mask is None:
                outgoing_token_mask = torch.ones(
                    (self.K, self.Q), dtype=comm_mean.dtype,
                    device=comm_mean.device)
            comm_action, comm_rate, _, _ = (
                self.agents[0].sample_communication(
                    comm_mean, deterministic=True))
            if self._joint_isac_power_enabled:
                (_, _, comm_fraction, sensing_weights, _, _) = (
                    self.agents[0].sample_isac_resources(
                        comm_mean, comm_rate, deterministic=True,
                        comm_fraction_min=float(getattr(
                            self.cfg.marl, 'comm_power_fraction_min', 0.0)),
                        comm_fraction_max=float(getattr(
                            self.cfg.marl, 'comm_power_fraction_max', 1.0)),
                    ))
            else:
                comm_fraction = sensing_weights = None

        dpm = dp_mean.detach().cpu().numpy()
        dps = dp_log_std.detach().cpu().numpy()
        roles = role_logits.detach().cpu().numpy()
        movement_decision = bool(next_movement_phase == 0)
        actions = {}
        for k in range(self.K):
            decoded, _ = self.agents[k].action_space.decode(
                dpm[k], dps, roles[k], deterministic=True)
            if movement_decision:
                delta_p = decoded.delta_p
                role = decoded.role
            else:
                delta_p = self._held_movement_dp[env_index, k].copy()
                role = int(self._held_movement_role[env_index, k])
            actions[str(k)] = {'delta_p': delta_p, 'role': int(role)}

        messages = {
            k: comm_action[k].detach().cpu().numpy() for k in range(self.K)}
        rates = {k: int(comm_rate[k].item()) for k in range(self.K)}
        token_masks = {
            k: outgoing_token_mask[k].detach().cpu().numpy()
            for k in range(self.K)}
        if self._joint_isac_power_enabled:
            submit_args = (
                messages, rates,
                {k: float(comm_fraction[k].item()) for k in range(self.K)},
                {k: sensing_weights[k].detach().cpu().numpy()
                 for k in range(self.K)},
            )
            if self._sparse_claim_enabled:
                branch.core.submit_learned_communications(
                    *submit_args, token_masks=token_masks)
            else:
                branch.core.submit_learned_communications(*submit_args)
        else:
            if self._sparse_claim_enabled:
                branch.core.submit_learned_communications(
                    messages, rates, token_masks=token_masks)
            else:
                branch.core.submit_learned_communications(messages, rates)
        _, _, _, _, info = branch.step(actions)
        return np.asarray(info['P_D_q'], dtype=np.float64).copy()

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

    def _next_training_seed(self, env, env_index: int) -> int:
        """Choose a reset seed without leaking evaluation-bank scenarios."""
        if self._training_fixed_seed >= 0:
            seed = self._training_fixed_seed
        elif self._training_seed_sampler is None:
            seed = int(env.rng.integers(0, 2**31 - 1))
        else:
            seed = self._training_seed_sampler.sample(
                env.rng, self.total_frames)
        self._training_current_seeds[env_index] = seed
        self._training_episode_worst_sum[env_index] = 0.0
        self._training_episode_steps[env_index] = 0
        return seed

    def _update_capacity_matching_blend(self) -> float:
        """Anneal matching authority without changing disabled baselines."""
        if not self._capacity_matching_enabled:
            return 0.0
        progress = float(np.clip(
            self.total_frames / self._capacity_matching_anneal_frames,
            0.0, 1.0))
        blend = (
            self._capacity_matching_blend_start
            + progress * (
                self._capacity_matching_blend_end
                - self._capacity_matching_blend_start))
        self.agents[0].actor.set_capacity_matching_blend(blend)
        return float(blend)

    def _update_target_movement_blend(self) -> float:
        """Anneal slow target commitment authority between PPO rollouts."""
        if not self._target_movement_blend_schedule_enabled:
            return float(getattr(
                self.cfg.marl, 'target_allocation_movement_blend', 0.0))
        progress = float(np.clip(
            self.total_frames / self._target_movement_blend_anneal_frames,
            0.0, 1.0))
        blend = (
            self._target_movement_blend_start
            + progress * (
                self._target_movement_blend_end
                - self._target_movement_blend_start))
        self.agents[0].actor.set_target_allocation_movement_blend(blend)
        return float(blend)

    def get_policy_runtime_state(self) -> Dict[str, float]:
        """Serializable non-parameter state that changes policy execution."""
        actor = self.agents[0].actor
        return {
            'total_frames': int(self.total_frames),
            'capacity_matching_blend': float(getattr(
                actor, '_capacity_matching_blend', 0.0)),
            'target_allocation_movement_blend': float(getattr(
                actor, '_target_allocation_movement_blend', 0.0)),
        }

    def restore_policy_runtime_state(self, state: Optional[Dict]) -> None:
        """Restore schedule-controlled execution state from a checkpoint."""
        if not state:
            return
        actor = self.agents[0].actor
        if ('capacity_matching_blend' in state
                and hasattr(actor, 'set_capacity_matching_blend')):
            actor.set_capacity_matching_blend(
                float(state['capacity_matching_blend']))
        if ('target_allocation_movement_blend' in state
                and hasattr(actor,
                            'set_target_allocation_movement_blend')):
            actor.set_target_allocation_movement_blend(float(
                state['target_allocation_movement_blend']))
        if 'total_frames' in state:
            self.total_frames = max(0, int(state['total_frames']))

    @torch.inference_mode()
    def collect_rollout(self) -> bool:
        """Collect a full rollout with parallel environments for GPU batching.

        N environments run in parallel. Observations from all N envs are batched
        into a single GPU forward pass (batch = N*K instead of K), dramatically
        reducing GPU kernel launch overhead and idle time.

        Returns:
            True if any episode ended during rollout
        """
        self.buffer.clear()
        # PPO requires one fixed behaviour policy throughout a rollout.  The
        # matching authority is therefore annealed only at rollout boundaries;
        # changing it every frame invalidates old-log-prob recomputation even
        # when all neural parameters are unchanged.
        self._update_capacity_matching_blend()
        self._update_target_movement_blend()

        # ── Reset accumulators for this rollout ──
        episode_ended = False
        N = self.num_envs
        K = self.K

        # Reset only envs without current state (first call or after episode end)
        all_obs = []
        for n, env in enumerate(self.envs):
            if self._current_obs[n] is None:
                obs, _ = env.reset(seed=self._next_training_seed(env, n))
                self._current_obs[n] = obs
                self._comm_silence_streaks[n].fill(0)
                self._movement_phase[n] = 0
                self._held_movement_valid[n] = False
                # Clear ring buffer on fresh reset
                if self._use_window:
                    self._obs_ring[n].fill(0)
                    self._window_mask[n].fill(False)
            all_obs.append(self._current_obs[n])

        self._rollout_team_rewards = []
        self._rollout_pd = []
        self._rollout_constraint_costs = []
        self._rollout_lagrangian_penalties = []
        self._rollout_comm_agent_vars = []    # diagnostic: cross-agent comm variance
        self._rollout_utility = []            # diagnostic: utility (before comm cost)
        self._rollout_comm_cost = []          # diagnostic: comm cost per step
        self._rollout_learned_comm_bits = []
        self._rollout_learned_comm_energy = []
        self._rollout_learned_comm_latency = []
        self._rollout_learned_comm_delivery = []
        self._rollout_learned_comm_active = []
        self._rollout_learned_comm_violation = []
        self._rollout_comm_qos_values = []
        self._rollout_comm_qos_rewards = []
        self._rollout_comm_qos_reward_components = []
        self._rollout_comm_qos_rate_penalties = []
        self._rollout_comm_encouragement_bonuses = []
        self._rollout_comm_rate_bonuses = []
        self._rollout_comm_sender_delivery_penalties = []
        self._rollout_comm_rate_counts = np.zeros(
            len(getattr(self.cfg.marl, 'comm_rate_bits_per_dim',
                        [0, 4, 8, 16])), dtype=np.int64)
        self._rollout_comm_silence_penalties = []
        self._rollout_comm_silence_streaks = []
        self._rollout_reward_components = {}
        self._rollout_headwise_rewards = {
            name: [] for name in self.buffer.credit_head_names
        }
        self._rollout_causal_teacher_effects = []
        self._rollout_pd_tensor = None        # (T, Q) tensor for aux loss lookup
        self._rollout_cvar_deficits = []      # CVaR deficit per step
        self._rollout_cvar_penalties = []
        self._rollout_power_budget_violations = []
        # Executed RF accounting, sampled from the environment after every
        # micro-frame.  These are deliberately kept separate from the
        # differentiable unroll objectives: an apparent detection gain is not
        # a systems gain if it merely consumes more of the shared RF budget.
        self._rollout_isac_resource_metrics = {
            name: [] for name in (
                'isac_comm_power_w',
                'isac_sensing_power_w',
                'isac_unused_power_w',
            )
        }
        self._rollout_temporal_power_records = []
        # Stage-2 bounded distributed-power diagnostics.  Keep only scalar
        # summaries in the trainer buffer; the environment info retains the
        # target-price vector for post-hoc audits.
        self._rollout_distributed_power_metrics = {
            name: [] for name in (
                'distributed_primal_dual_power_selected_violation',
                'distributed_primal_dual_power_candidate_violation',
                'distributed_primal_dual_power_candidate_stationarity',
                'distributed_primal_dual_power_candidate_dual_residual',
                'distributed_primal_dual_power_consensus_residual',
                'distributed_primal_dual_power_target_price_mean',
                'distributed_primal_dual_power_target_price_max',
                'distributed_primal_dual_power_fallback',
                'distributed_primal_dual_power_warm_started',
                'distributed_primal_dual_power_actor_proposal_used',
                'distributed_primal_dual_power_actor_proposal_mix',
                'distributed_primal_dual_power_candidate_accepted',
                'distributed_primal_dual_power_converged',
                'distributed_primal_dual_power_rounds',
            )
        }

        # Pre-allocate numpy buffers (reused each step)
        dp_mean_np = np.empty((N * K, 2), dtype=np.float64)
        role_logits_np = np.empty((N * K, 3), dtype=np.float64)
        values_np = np.empty(N * K, dtype=np.float64)
        credit_values_np = np.zeros((N * K, 4), dtype=np.float64)
        actions_dp = np.zeros((K, 2), dtype=np.float64)
        actions_role = np.zeros(K, dtype=np.int32)
        comm_payload_dim = int(getattr(
            self.agents[0], 'comm_payload_dim', 16))
        actions_comm = np.zeros((K, comm_payload_dim), dtype=np.float64)
        actions_comm_rate = np.zeros(K, dtype=np.int32)
        actions_comm_token_mask = np.ones(
            (K, self.Q), dtype=np.float64)
        actions_isac_power_raw = np.zeros(K, dtype=np.float64)
        actions_sensing_raw = np.zeros((K, self.Q), dtype=np.float64)
        actions_comm_fraction = np.zeros(K, dtype=np.float64)
        actions_sensing_weights = np.full(
            (K, self.Q), 1.0 / max(self.Q, 1), dtype=np.float64)
        movement_action_mask = np.ones(K, dtype=np.float64)
        comm_round_phase = np.zeros(K, dtype=np.float64)
        log_probs = np.zeros(K, dtype=np.float64)
        head_log_probs = np.zeros((K, 4), dtype=np.float64)

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
                            h = np.zeros(self.buffer.gru_hidden_dim, dtype=np.float32)
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

            round_phase_env = self._movement_phase.astype(np.float32) / max(
                self.movement_decision_interval - 1, 1)
            round_phase_batch = torch.as_tensor(
                np.repeat(round_phase_env, K), dtype=torch.float32,
                device=self.device)
            rollout_agent_identity = torch.arange(
                K, dtype=torch.long, device=self.device).repeat(N)
            dp_mean, dp_log_std, role_logits, comm_msgs, _pd_pred, h_new = self.agents[0].actor(
                actor_window if self._use_window else self._obs_gpu[:N*K],
                h_prev_batch,
                window_mask=actor_wmask if self._use_window else None,
                comm_round_phase=round_phase_batch,
                agent_identity=rollout_agent_identity)
            outgoing_token_mask_t = getattr(
                self.agents[0].actor, 'last_outgoing_token_mask', None)
            if outgoing_token_mask_t is None:
                outgoing_token_mask_t = torch.ones(
                    (N * K, self.Q), dtype=comm_msgs.dtype,
                    device=comm_msgs.device)

            # In cost-aware mode communication is part of the stochastic joint
            # action: the policy chooses both message content and rate (0=silent).
            if self._comm_cost_aware:
                (comm_actions_t, comm_rates_t, comm_log_probs_t, _,
                 comm_lp_components_t) = self.agents[0].sample_communication(
                     comm_msgs, return_components=True)
                if self._adaptive_topk_from_rate_enabled:
                    claim_scores_t = getattr(
                        self.agents[0].actor,
                        'last_sparse_claim_scores',
                        None,
                    )
                    if claim_scores_t is None:
                        raise RuntimeError(
                            'adaptive top-k requires actor claim scores')
                    outgoing_token_mask_t = build_rate_conditioned_topk_mask(
                        claim_scores_t,
                        comm_rates_t,
                        self._adaptive_topk_rate_mapping,
                        maximum_mask=outgoing_token_mask_t,
                    )
                comm_active_t = (comm_rates_t > 0).to(comm_msgs.dtype).unsqueeze(-1)
                comm_dim_mask_t = outgoing_token_mask_t.repeat_interleave(
                    max(comm_msgs.shape[-1] // max(self.Q, 1), 1), dim=-1)
                effective_comm = (
                    comm_actions_t * comm_active_t * comm_dim_mask_t)
                if self._joint_isac_power_enabled:
                    (isac_power_raw_t, sensing_raw_t,
                     comm_fraction_t, sensing_weights_t,
                     resource_log_probs_t, _) = (
                        self.agents[0].sample_isac_resources(
                            comm_msgs, comm_rates_t,
                            comm_fraction_min=float(getattr(
                                self.cfg.marl, 'comm_power_fraction_min', 0.0)),
                            comm_fraction_max=float(getattr(
                                self.cfg.marl, 'comm_power_fraction_max', 1.0)),
                        )
                    )
                else:
                    isac_power_raw_t = sensing_raw_t = None
                    comm_fraction_t = sensing_weights_t = None
                    resource_log_probs_t = torch.zeros_like(comm_log_probs_t)
            else:
                comm_actions_t = None
                comm_rates_t = None
                comm_log_probs_t = None
                effective_comm = self._effective_comm(comm_msgs)

            # P0 FIX: save h_prev numpy per env for buffer storage.
            # h_prev_list is ordered: env0_k0_nbr0, env0_k0_nbr1, ..., env0_k1_nbr0, ...
            # Reshape to (N, K, K-1, D) then index per env.
            h_prev_arr = None
            if h_prev_list:
                h_prev_arr = np.stack(h_prev_list).reshape(N, K, K-1, -1)

            # Store comm + GRU hidden state per-neighbor for next frame
            comm_np = effective_comm.detach().cpu().numpy()
            if self._comm_cost_aware:
                comm_action_np = comm_actions_t.detach().cpu().numpy()
                comm_rate_np = comm_rates_t.detach().cpu().numpy()
                comm_token_mask_np = (
                    outgoing_token_mask_t.detach().cpu().numpy())
                comm_log_prob_np = comm_log_probs_t.detach().cpu().numpy()
                message_log_prob_np = (
                    comm_lp_components_t['message'].detach().cpu().numpy())
                rate_log_prob_np = (
                    comm_lp_components_t['rate'].detach().cpu().numpy())
                if self._joint_isac_power_enabled:
                    isac_power_raw_np = isac_power_raw_t.detach().cpu().numpy()
                    sensing_raw_np = sensing_raw_t.detach().cpu().numpy()
                    comm_fraction_np = comm_fraction_t.detach().cpu().numpy()
                    sensing_weights_np = sensing_weights_t.detach().cpu().numpy()
                    resource_log_prob_np = (
                        resource_log_probs_t.detach().cpu().numpy())
                else:
                    isac_power_raw_np = np.zeros(N * K, dtype=np.float64)
                    sensing_raw_np = np.zeros((N * K, self.Q), dtype=np.float64)
                    comm_fraction_np = np.zeros(N * K, dtype=np.float64)
                    sensing_weights_np = np.full(
                        (N * K, self.Q), 1.0 / max(self.Q, 1))
                    resource_log_prob_np = np.zeros(N * K, dtype=np.float64)
            else:
                comm_action_np = comm_np
                comm_rate_np = np.zeros(N * K, dtype=np.int64)
                comm_token_mask_np = np.ones(
                    (N * K, self.Q), dtype=np.float64)
                comm_log_prob_np = np.zeros(N * K, dtype=np.float64)
                message_log_prob_np = np.zeros(N * K, dtype=np.float64)
                rate_log_prob_np = np.zeros(N * K, dtype=np.float64)
                isac_power_raw_np = np.zeros(N * K, dtype=np.float64)
                sensing_raw_np = np.zeros((N * K, self.Q), dtype=np.float64)
                comm_fraction_np = np.zeros(N * K, dtype=np.float64)
                sensing_weights_np = np.full(
                    (N * K, self.Q), 1.0 / max(self.Q, 1))
                resource_log_prob_np = np.zeros(N * K, dtype=np.float64)
            h_new_np = h_new.cpu().numpy() if h_new is not None else None
            if h_new_np is not None:
                h_new_np = h_new_np.reshape(N, K, K-1, -1)
            for n in range(N):
                for k in range(K):
                    # Legacy mode keeps the historical zero-cost direct message
                    # path. Cost-aware messages go through the physical U2U link.
                    if not self._comm_cost_aware:
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
                comm_summary = self._critic_comm_summary(effective_comm)
                comm_agg = comm_summary.reshape(N, K, 16).mean(dim=1)
                comm_agg_rep = comm_agg.repeat_interleave(K, dim=0)  # (N*K, 16)
                if self.centralized_critic:
                    base = self._gs_gpu[:N].repeat_interleave(K, dim=0)  # (N*K, gs_dim)
                else:
                    base = self._obs_gpu[:N*K]                          # IPPO: local obs
                gs_with_id = torch.cat([base, agent_oh, comm_agg_rep], dim=-1)
                # Scalar, credit and per-target heads share one deterministic
                # critic encoding.  A single pass is mathematically identical
                # to the historical duplicate calls.
                values_t, credit_values_t, target_values_t = (
                    self.agents[0].critic.forward_with_auxiliaries(
                        gs_with_id,
                        include_credit=self._headwise_credit_enabled,
                    ))
                target_v_np = target_values_t.detach().cpu().numpy() if target_values_t is not None else None

            # Copy results back to CPU (single transfer per rollout step)
            dp_mean_np[:] = dp_mean.detach().cpu().numpy()
            dp_std_np = dp_log_std.detach().cpu().numpy()  # (2,) shared param
            role_logits_np[:] = role_logits.detach().cpu().numpy()
            values_np[:] = values_t.detach().cpu().numpy()
            if credit_values_t is not None:
                credit_values_np[:] = credit_values_t.detach().cpu().numpy()

            # ── Per-env action decode + step + buffer store ──
            for n in range(N):
                env = self.envs[n]
                obs = all_obs[n]
                idx0 = n * K
                idx1 = idx0 + K
                movement_decision = bool(
                    self._movement_phase[n] == 0
                    or not self._held_movement_valid[n])
                movement_action_mask.fill(float(movement_decision))
                comm_round_phase.fill(float(round_phase_env[n]))

                # Decode actions for this env's agents
                for k in range(K):
                    local_idx = idx0 + k
                    action, lp = self.agents[k].action_space.decode(
                        dp_mean_np[local_idx], dp_std_np, role_logits_np[local_idx]
                    )
                    if movement_decision:
                        actions_dp[k] = action.delta_p
                        actions_role[k] = action.role
                    else:
                        actions_dp[k] = self._held_movement_dp[n, k]
                        actions_role[k] = self._held_movement_role[n, k]
                    actions_comm[k] = comm_action_np[local_idx]
                    actions_comm_rate[k] = int(comm_rate_np[local_idx])
                    actions_comm_token_mask[k] = (
                        comm_token_mask_np[local_idx])
                    actions_isac_power_raw[k] = isac_power_raw_np[local_idx]
                    actions_sensing_raw[k] = sensing_raw_np[local_idx]
                    actions_comm_fraction[k] = comm_fraction_np[local_idx]
                    actions_sensing_weights[k] = sensing_weights_np[local_idx]
                    head_log_probs[k] = np.asarray([
                        movement_action_mask[k] * lp,
                        float(message_log_prob_np[local_idx]),
                        float(rate_log_prob_np[local_idx]),
                        float(resource_log_prob_np[local_idx]),
                    ])
                    log_probs[k] = float(np.sum(head_log_probs[k]))
                if movement_decision:
                    self._held_movement_dp[n] = actions_dp
                    self._held_movement_role[n] = actions_role
                    self._held_movement_valid[n] = True
                if self._comm_cost_aware:
                    self._rollout_comm_rate_counts += np.bincount(
                        actions_comm_rate,
                        minlength=self._rollout_comm_rate_counts.size,
                    )[:self._rollout_comm_rate_counts.size]

                # Oracle-guided exploration: replace actions with Greedy-Approach
                # with probability α (decaying). Oracle actions are NOT trained on.
                oracle_mask = np.ones(K, dtype=np.float64)  # 1=actor, 0=oracle
                r = self.envs[n].core.rng.random()
                if movement_decision and r < self._oracle_alpha:
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
                        # Only the replaced movement head is excluded.  The
                        # communication/resource actions remain valid samples.
                        movement_action_mask[k] = 0.0
                        head_log_probs[k, 0] = 0.0
                        log_probs[k] = float(np.sum(head_log_probs[k]))
                    self._held_movement_dp[n] = actions_dp

                # Build actions dict once per macro step
                actions_dict = {
                    str(k): {'delta_p': actions_dp[k], 'role': int(actions_role[k])}
                    for k in range(K)
                }

                # Macro-loop: hold same action for macro_interval micro-frames
                macro_rewards = {k: 0.0 for k in range(K)}
                macro_pd = []
                macro_costs = []
                macro_learned_comm = []
                macro_reward_components = []
                for micro in range(self.macro_interval):
                    # One policy communication action is one packet. Movement
                    # is held for the macro interval, but retransmitting the
                    # same packet on every micro-frame would multiply bits and
                    # energy by actor_decision_interval without a new decision.
                    if self._comm_cost_aware and micro == 0:
                        messages_k = {
                            k: actions_comm[k] for k in range(K)}
                        rates_k = {
                            k: int(actions_comm_rate[k]) for k in range(K)}
                        if self._joint_isac_power_enabled:
                            submit_args = (
                                messages_k, rates_k,
                                {k: float(actions_comm_fraction[k])
                                 for k in range(K)},
                                {k: actions_sensing_weights[k]
                                 for k in range(K)},
                            )
                            if self._sparse_claim_enabled:
                                env.core.submit_learned_communications(
                                    *submit_args, token_masks={
                                    k: actions_comm_token_mask[k]
                                    for k in range(K)},
                                )
                            else:
                                env.core.submit_learned_communications(
                                    *submit_args)
                        else:
                            if self._sparse_claim_enabled:
                                env.core.submit_learned_communications(
                                    messages_k, rates_k, token_masks={
                                    k: actions_comm_token_mask[k]
                                    for k in range(K)})
                            else:
                                env.core.submit_learned_communications(
                                    messages_k, rates_k)
                    next_obs, rewards, terminated, truncated, info = env.step(actions_dict)
                    w = self.gamma_micro ** micro
                    for k in range(K):
                        macro_rewards[k] += w * float(rewards[str(k)])
                    macro_pd.append(info.get('P_D_q', np.zeros(self.Q)).copy())
                    macro_costs.append(float(info.get('constraint_info', {}).get('any_violation', 0.0)))
                    self._rollout_power_budget_violations.append(float(
                        info.get('isac_max_power_budget_violation_w', 0.0)))
                    for metric_name, values in (
                            self._rollout_isac_resource_metrics.items()):
                        if metric_name in info:
                            value = float(info[metric_name])
                            if np.isfinite(value):
                                values.append(value)
                    for metric_name, values in (
                            self._rollout_distributed_power_metrics.items()):
                        if metric_name in info:
                            value = float(info[metric_name])
                            if np.isfinite(value):
                                values.append(value)
                    macro_learned_comm.append({
                        key: value for key, value in info.items()
                        if key.startswith('learned_comm_')
                    })
                    macro_reward_components.append(dict(
                        info.get('reward_components', {})))
                    if terminated.get('__all__', False) or truncated.get('__all__', False):
                        break

                # Lagrangian penalty on mean constraint cost over macro step
                constraint_cost = float(np.mean(macro_costs)) if macro_costs else 0.0

                # ── Fix #1: convert string keys to int ──
                obs_int = {int(k): v for k, v in obs.items()}
                rewards_int = {k: float(v) for k, v in macro_rewards.items()}
                environment_rewards_int = dict(rewards_int)
                cvar_penalty = 0.0
                qos_reward = 0.0
                qos_values = None
                rate_penalties = np.zeros(K, dtype=np.float64)
                comm_bonuses = np.zeros(K, dtype=np.float64)
                rate_bonuses = np.zeros(K, dtype=np.float64)
                sender_delivery_penalties = np.zeros(K, dtype=np.float64)
                silence_penalties = np.zeros(K, dtype=np.float64)

                # Augment reward with Lagrangian constraint penalty
                lagrangian_penalty = self.lagrangian_lambda * constraint_cost
                for k in range(K):
                    rewards_int[k] -= lagrangian_penalty
                self._rollout_lagrangian_penalties.append(lagrangian_penalty)

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
                    self._rollout_cvar_penalties.append(cvar_penalty)

                # QoS-constrained communication objective. This is the
                # Lagrangian term lambda^T(P_D - floor): while a quality floor
                # is missed, its multiplier rises after the rollout and gives
                # the policy a stronger incentive to spend resources that
                # improve that metric. Once all floors are met, multipliers
                # decay and the explicit U2U cost favours silence/lower rates.
                if self._comm_qos_enabled and macro_pd:
                    qos_values = compute_comm_qos_metrics(
                        np.mean(np.asarray(macro_pd), axis=0))
                    qos_margin = qos_values - self._comm_qos_targets
                    qos_reward_components = (
                        self._comm_qos_reward_scale
                        * self._comm_qos_lambdas * qos_margin)
                    qos_reward = float(np.sum(qos_reward_components))
                    for k in range(K):
                        rewards_int[k] += qos_reward
                    rate_penalties = compute_comm_qos_rate_penalties(
                        actions_comm_rate,
                        getattr(self.cfg.marl, 'comm_rate_bits_per_dim',
                                [0, 4, 8, 16]),
                        qos_values,
                        self._comm_qos_targets,
                        self._comm_qos_min_rate_bits,
                        self._comm_qos_rate_shortfall_penalty,
                    )
                    for k in range(K):
                        rewards_int[k] -= float(rate_penalties[k])
                    # Communication is submitted only on micro-frame zero.
                    # Use that transport outcome rather than averaging it with
                    # later no-packet frames (whose delivery rate is 1 by
                    # convention). This prevents failed packets being rewarded.
                    packet_delivery = float(
                        macro_learned_comm[0].get(
                            'learned_comm_delivery_rate', 0.0)
                        if macro_learned_comm else 0.0)
                    if self._comm_encouragement_enabled:
                        comm_bonuses = compute_comm_encouragement_bonuses(
                            actions_comm_rate,
                            qos_values,
                            self._comm_qos_targets,
                            packet_delivery,
                            self._comm_encouragement_weight,
                            self._comm_encouragement_floor_ratio,
                        )
                    else:
                        comm_bonuses = np.zeros(K, dtype=np.float64)
                    if self._comm_rate_bonus_enabled:
                        rate_bonuses = compute_comm_rate_exploration_bonuses(
                            actions_comm_rate,
                            getattr(self.cfg.marl, 'comm_rate_bits_per_dim',
                                    [0, 4, 8, 16]),
                            qos_values,
                            self._comm_qos_targets,
                            packet_delivery,
                            self._comm_rate_bonus_weight,
                            self._comm_rate_bonus_target_bits,
                        )
                    else:
                        rate_bonuses = np.zeros(K, dtype=np.float64)
                    for k in range(K):
                        rewards_int[k] += float(
                            comm_bonuses[k] + rate_bonuses[k])
                    self._rollout_comm_qos_values.append(qos_values)
                    self._rollout_comm_qos_rewards.append(qos_reward)
                    self._rollout_comm_qos_reward_components.append(
                        qos_reward_components)
                    self._rollout_comm_qos_rate_penalties.extend(
                        rate_penalties.tolist())
                    self._rollout_comm_encouragement_bonuses.extend(
                        comm_bonuses.tolist())
                    self._rollout_comm_rate_bonuses.extend(
                        rate_bonuses.tolist())

                if (self._comm_cost_aware
                        and self._comm_sender_delivery_penalty_enabled):
                    sender_delivery = np.asarray(
                        macro_learned_comm[0].get(
                            'learned_comm_per_sender_delivery_rate',
                            np.ones(K, dtype=np.float64))
                        if macro_learned_comm else np.ones(
                            K, dtype=np.float64),
                        dtype=np.float64).reshape(-1)
                    sender_delivery_penalties = (
                        compute_sender_delivery_penalties(
                            actions_comm_rate, sender_delivery,
                            self._comm_sender_delivery_penalty_weight))
                    for k in range(K):
                        rewards_int[k] -= float(
                            sender_delivery_penalties[k])
                    self._rollout_comm_sender_delivery_penalties.extend(
                        sender_delivery_penalties.tolist())

                # Sender-specific liveness shaping is separate from payload
                # cost and QoS reward. It allows brief resource-saving silence
                # but prevents the discrete rate policy from abandoning the
                # communication channel for long stretches.
                if (self._comm_cost_aware
                        and self._comm_silence_penalty_enabled):
                    streaks, silence_penalties = update_long_silence_penalties(
                        actions_comm_rate,
                        self._comm_silence_streaks[n],
                        self._comm_silence_grace_decisions,
                        self._comm_silence_penalty_per_decision,
                        self._comm_silence_penalty_max,
                    )
                    self._comm_silence_streaks[n] = streaks
                    for k in range(K):
                        rewards_int[k] -= float(silence_penalties[k])
                    self._rollout_comm_silence_penalties.extend(
                        silence_penalties.tolist())
                    self._rollout_comm_silence_streaks.extend(
                        streaks.tolist())

                causal_teacher_effects = None
                causal_teacher_masks = None
                if self._causal_ccp_enabled:
                    causal_teacher_effects = np.zeros(
                        (K, self.Q), dtype=np.float64)
                    causal_teacher_masks = np.zeros(K, dtype=np.float64)
                    branch_done = bool(
                        terminated.get('__all__', False)
                        or truncated.get('__all__', False))
                    intervention_due = (
                        ((step * self.num_envs + n)
                         % self._causal_ccp_stride) == 0)
                    active_senders = np.flatnonzero(actions_comm_rate > 0)
                    if (not branch_done and intervention_due
                            and active_senders.size > 0):
                        sender = int(active_senders[
                            ((step // self._causal_ccp_stride) + n)
                            % active_senders.size])
                        masked_next_obs, token_was_delivered = (
                            mask_sender_from_token_observations(
                                next_obs, sender,
                                self.agents[0].actor._obs_slices))
                        if token_was_delivered:
                            next_phase = int((
                                self._movement_phase[n] + self.macro_interval
                            ) % self.movement_decision_interval)
                            actual_pd = self._simulate_deterministic_next_qos(
                                env, next_obs, n, next_phase)
                            masked_pd = self._simulate_deterministic_next_qos(
                                env, masked_next_obs, n, next_phase)
                            causal_teacher_effects[sender] = (
                                actual_pd - masked_pd)
                            causal_teacher_masks[sender] = 1.0
                            self._rollout_causal_teacher_effects.append(
                                causal_teacher_effects[sender].copy())

                credit_rewards = None
                if self._headwise_credit_enabled:
                    def discounted_component(name):
                        return float(sum(
                            (self.gamma_micro ** micro_idx)
                            * float(component.get(name, 0.0))
                            for micro_idx, component in enumerate(
                                macro_reward_components)))

                    learned_comm_cost = discounted_component(
                        'learned_comm_cost')
                    coord_credit = discounted_component('coord_total')
                    # Difference-reward blending attenuates the shared team
                    # cost by team_weight; undo only that part for movement.
                    comm_cost_in_movement = learned_comm_cost
                    if getattr(self.cfg.marl, 'use_difference_reward', False):
                        comm_cost_in_movement *= float(getattr(
                            self.cfg.marl, 'team_weight', 0.7))

                    rate_levels = np.asarray(getattr(
                        self.cfg.marl, 'comm_rate_bits_per_dim',
                        [0, 4, 8, 16]), dtype=np.float64)
                    rate_indices = np.clip(
                        actions_comm_rate, 0, max(rate_levels.size - 1, 0))
                    sender_weights = rate_levels[rate_indices]
                    if float(np.sum(sender_weights)) > 0.0:
                        sender_costs = (
                            learned_comm_cost * sender_weights
                            / float(np.sum(sender_weights)))
                    else:
                        sender_costs = np.zeros(K, dtype=np.float64)

                    credit_rewards = np.zeros((K, 4), dtype=np.float64)
                    for k in range(K):
                        # Movement sees sensing/coordination and physical
                        # constraints, but no rate/transport incentive.
                        credit_rewards[k, 0] = (
                            environment_rewards_int[k]
                            + comm_cost_in_movement
                            - lagrangian_penalty - cvar_penalty + qos_reward)
                        # Latent content is delivered into the next
                        # observation. Its task credit is back-filled from the
                        # next transition below, never from this same frame.
                        credit_rewards[k, 1] = 0.0
                        # Rate decides whether precision is worth its own
                        # bit/energy/delay cost and liveness terms.
                        credit_rewards[k, 2] = (
                            qos_reward
                            + float(comm_bonuses[k] + rate_bonuses[k])
                            - float(
                                rate_penalties[k] + silence_penalties[k]
                                + sender_delivery_penalties[k])
                            - float(sender_costs[k]))
                        # Joint power/sensing allocation controls both sensing
                        # quality and transport cost, so it keeps the full
                        # environment task reward but not rate-only bonuses.
                        credit_rewards[k, 3] = (
                            environment_rewards_int[k]
                            - lagrangian_penalty - cvar_penalty + qos_reward)
                    # Causal one-round message credit: the token/rate selected
                    # by this env on the previous transition produced the
                    # inbox used for the current coordination result. Buffer
                    # rows are interleaved by environment, hence ptr-N.
                    previous_env_row = self.buffer.ptr - self.num_envs
                    if previous_env_row >= 0:
                        delayed_worst_credit = (
                            self._adaptive_topk_worst_credit_coef
                            * float(qos_values[2])
                            if (self._adaptive_topk_from_rate_enabled
                                and qos_values is not None)
                            else 0.0)
                        for k in range(K):
                            if self.buffer.masks[previous_env_row, k] > 0.0:
                                self.buffer.credit_rewards[
                                    previous_env_row, k, 1] += coord_credit
                                self.buffer.credit_rewards[
                                    previous_env_row, k, 2] += coord_credit
                                self.buffer.credit_rewards[
                                    previous_env_row, k, 2] += (
                                        delayed_worst_credit)

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
                comm_agg_n = np.mean(self._critic_comm_summary_np(
                    comm_np[n*K:(n+1)*K]), axis=0)
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
                env_wmask = (self._window_mask[n].copy() if self._use_window
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
                    window_mask=env_wmask,
                    actions_comm=(actions_comm if self._comm_cost_aware else None),
                    actions_comm_rate=(actions_comm_rate if self._comm_cost_aware else None),
                    actions_isac_power_raw=(
                        actions_isac_power_raw
                        if self._joint_isac_power_enabled else None),
                    actions_sensing_raw=(
                        actions_sensing_raw
                        if self._joint_isac_power_enabled else None),
                    movement_action_mask=movement_action_mask,
                    comm_round_phase=comm_round_phase,
                    head_log_probs=(
                        head_log_probs if self._headwise_credit_enabled else None),
                    credit_rewards=credit_rewards,
                    credit_values=(
                        credit_values_np[idx0:idx1]
                        if self._headwise_credit_enabled else None),
                    causal_teacher_effects=causal_teacher_effects,
                    causal_teacher_masks=causal_teacher_masks,
                )
                if self._temporal_unrolled_power_enabled:
                    common_tensor_names = (
                        'temporal_unrolled_power_target_requirement',
                        'temporal_unrolled_power_total_rf_budget_w',
                        'temporal_unrolled_power_sensing_cap_w',
                    )
                    tensor_names = (
                        common_tensor_names + (
                            'temporal_feasible_structure_coefficient_per_watt',
                            'temporal_feasible_structure_executed_edge_mask',
                        )
                        if self._temporal_feasible_structure_enabled
                        else common_tensor_names + (
                            'temporal_unrolled_power_gain_per_watt',
                            'temporal_unrolled_power_feasible_mask',
                        )
                    )
                    missing_temporal = [
                        name for name in tensor_names if name not in info]
                    if self._temporal_feasible_structure_enabled:
                        missing_temporal.extend(
                            name for name in (
                                'temporal_feasible_structure_target_pair_limit',
                                'temporal_feasible_structure_reports_per_receiver',
                            ) if name not in info)
                    temporal_record = {
                        'buffer_row': int(self.buffer.ptr - 1),
                        'environment': int(n),
                        'terminal': bool(done_all),
                        'valid': not missing_temporal,
                    }
                    if not missing_temporal:
                        temporal_record.update({
                            'requirement': np.asarray(
                                info[common_tensor_names[0]],
                                dtype=np.float64).copy(),
                            'total_rf_budget_w': np.asarray(
                                info[common_tensor_names[1]],
                                dtype=np.float64).copy(),
                            'sensing_cap_w': np.asarray(
                                info[common_tensor_names[2]],
                                dtype=np.float64).copy(),
                        })
                        if self._temporal_feasible_structure_enabled:
                            temporal_record.update({
                                'coefficient': np.asarray(info[
                                    'temporal_feasible_structure_'
                                    'coefficient_per_watt'],
                                    dtype=np.float64).copy(),
                                'executed_edge_mask': np.asarray(info[
                                    'temporal_feasible_structure_'
                                    'executed_edge_mask'], dtype=bool).copy(),
                                'target_pair_limit': int(info[
                                    'temporal_feasible_structure_'
                                    'target_pair_limit']),
                                'reports_per_receiver': int(info[
                                    'temporal_feasible_structure_'
                                    'reports_per_receiver']),
                            })
                        else:
                            temporal_record.update({
                                'gain': np.asarray(info[
                                    'temporal_unrolled_power_gain_per_watt'],
                                    dtype=np.float64).copy(),
                                'mask': np.asarray(info[
                                    'temporal_unrolled_power_feasible_mask'],
                                    dtype=bool).copy(),
                            })
                    self._rollout_temporal_power_records.append(
                        temporal_record)
                self.total_frames += self.macro_interval
                self._movement_phase[n] = (
                    self._movement_phase[n] + self.macro_interval
                ) % self.movement_decision_interval

                # Accumulate rollout-level metrics (mean over macro frames)
                if macro_pd:
                    macro_pd_mean = np.mean(macro_pd, axis=0)
                    self._rollout_pd.append(macro_pd_mean)
                    self._training_episode_worst_sum[n] += float(
                        np.min(macro_pd_mean))
                    self._training_episode_steps[n] += 1
                self._rollout_constraint_costs.append(constraint_cost)
                # Fix: populate team_reward from env info
                self._rollout_team_rewards.append(float(info.get('team_reward', 0.0)))
                component_names = {
                    name for component in macro_reward_components
                    for name in component
                }
                for name in component_names:
                    values = [
                        float(component[name])
                        for component in macro_reward_components
                        if name in component
                    ]
                    self._rollout_reward_components.setdefault(
                        name, []).append(float(np.mean(values)))
                # Diagnostic: comm message variance (across K agents)
                self._rollout_comm_agent_vars.append(float(np.var(comm_np[n*K:(n+1)*K])))
                # Diagnostic: utility from P_D (without comm cost)
                avg_pd_frame = np.mean(info.get('P_D_q', np.zeros(self.Q)))
                self._rollout_utility.append(float(avg_pd_frame))
                # Communication cost (legacy reporting plus learned U2U traffic).
                p0_bits = info.get('total_bits', 0.0)
                lambda_r = getattr(self, '_lambda_report', 1e-5)
                learned_bits = float(sum(
                    m.get('learned_comm_bits', 0.0) for m in macro_learned_comm))
                learned_energy = float(sum(
                    m.get('learned_comm_energy_j', 0.0) for m in macro_learned_comm))
                learned_latency = float(np.mean([
                    m.get('learned_comm_mean_latency_s', 0.0)
                    for m in macro_learned_comm])) if macro_learned_comm else 0.0
                learned_delivery = float(np.mean([
                    m.get('learned_comm_delivery_rate', 0.0)
                    for m in macro_learned_comm])) if macro_learned_comm else 0.0
                learned_active = float(np.mean([
                    m.get('learned_comm_active_senders', 0.0)
                    for m in macro_learned_comm])) if macro_learned_comm else 0.0
                learned_violation = float(np.mean([
                    m.get('learned_comm_deadline_violation_rate', 0.0)
                    for m in macro_learned_comm])) if macro_learned_comm else 0.0
                comm_cost = (
                    lambda_r * float(p0_bits)
                    + getattr(self.cfg.marl, 'comm_bit_cost_weight', 0.0) * learned_bits
                    + getattr(self.cfg.marl, 'comm_energy_cost_weight', 0.0) * learned_energy
                    + getattr(self.cfg.marl, 'comm_delay_cost_weight', 0.0)
                    * (learned_latency + learned_violation)
                )
                self._rollout_comm_cost.append(float(comm_cost))
                self._rollout_learned_comm_bits.append(learned_bits)
                self._rollout_learned_comm_energy.append(learned_energy)
                self._rollout_learned_comm_latency.append(learned_latency)
                self._rollout_learned_comm_delivery.append(learned_delivery)
                self._rollout_learned_comm_active.append(learned_active)
                self._rollout_learned_comm_violation.append(learned_violation)

                # Decay entropy
                decay_progress = min(1.0, self.total_frames / self.entropy_decay_frames)
                self.entropy_coef = (
                    self.entropy_init
                    + decay_progress * (self.entropy_final - self.entropy_init)
                )

                # ── Only reset when episode actually ends ──
                if done_all:
                    episode_ended = True
                    if (self._training_seed_sampler is not None
                            and self._training_current_seeds[n] is not None
                            and self._training_episode_steps[n] > 0):
                        episode_worst = (
                            self._training_episode_worst_sum[n]
                            / self._training_episode_steps[n])
                        self._training_seed_sampler.update(
                            self._training_current_seeds[n], episode_worst)
                    new_obs, _ = env.reset(
                        seed=self._next_training_seed(env, n))
                    all_obs[n] = new_obs
                    self._comm_silence_streaks[n].fill(0)
                    self._movement_phase[n] = 0
                    self._held_movement_valid[n] = False
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
            final_round_phase = torch.as_tensor(
                np.repeat(
                    self._movement_phase.astype(np.float32)
                    / max(self.movement_decision_interval - 1, 1), K),
                dtype=torch.float32, device=self.device)
            final_agent_identity = torch.arange(
                K, dtype=torch.long, device=self.device).repeat(N)
            _, _, _, final_comm, _, _ = self.agents[0].actor(
                self._obs_gpu[:N*K], final_h_batch,
                comm_round_phase=final_round_phase,
                agent_identity=final_agent_identity)
            if self._comm_cost_aware:
                final_comm_action, final_comm_rate, _, _ = (
                    self.agents[0].sample_communication(final_comm)
                )
                final_comm = final_comm_action * (
                    final_comm_rate > 0).to(final_comm.dtype).unsqueeze(-1)
            else:
                final_comm = self._effective_comm(final_comm)
            final_comm_summary = self._critic_comm_summary(final_comm)
            final_comm_agg = final_comm_summary.reshape(
                N, K, 16).mean(dim=1).repeat_interleave(K, dim=0)

            if self.centralized_critic:
                final_gs_batch = np.stack([e.core.get_global_state() for e in self.envs])
                self._gs_gpu[:N].copy_(torch.as_tensor(final_gs_batch, dtype=torch.float32))
                base = self._gs_gpu[:N].repeat_interleave(K, dim=0)
            else:
                base = self._obs_gpu[:N*K]  # already has final obs
            gs_with_id = torch.cat([base, agent_oh, final_comm_agg], dim=-1)

            # Scalar and per-target next values
            (next_values_t, next_credit_values_t,
             next_target_values_t) = (
                self.agents[0].critic.forward_with_auxiliaries(
                    gs_with_id,
                    include_credit=self._headwise_credit_enabled,
                ))
            next_values = next_values_t.detach().cpu().numpy()
            next_credit_values = (
                next_credit_values_t.detach().cpu().numpy()
                if next_credit_values_t is not None else None)
            next_pt_values = (next_target_values_t.detach().cpu().numpy()
                              if next_target_values_t is not None else None)

        # Effective gamma between macro transitions
        gamma_eff = self.gamma_micro ** self.macro_interval
        self.buffer.gamma = gamma_eff
        self.buffer.compute_gae(
            next_values,
            next_per_target_values=next_pt_values,
            next_credit_values=next_credit_values,
        )

        return episode_ended

    def _fit_ccp_and_route_message_credit(
        self,
        data: Dict[str, torch.Tensor],
        obs: torch.Tensor,
        actions_comm: torch.Tensor,
        actions_comm_rate: torch.Tensor,
        credit_advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Fit the amortized intervention model and route tail-aware credit."""
        metrics = {
            'causal_ccp_samples': 0.0,
            'causal_ccp_loss': 0.0,
            'causal_ccp_val_corr': 0.0,
            'causal_ccp_val_sign_accuracy': 0.0,
            'causal_ccp_reliability': 0.0,
            'causal_ccp_teacher_abs_effect': 0.0,
            'causal_ccp_credit_fraction': 0.0,
        }
        required = {
            'causal_teacher_effects', 'causal_teacher_masks',
            'per_target_rewards_raw'}
        if (not self._causal_ccp_enabled
                or self._causal_ccp is None
                or not required.issubset(data)):
            return credit_advantages, metrics

        teacher = data['causal_teacher_effects'].to(self.device)
        measured = data['causal_teacher_masks'].to(
            self.device) > 0.5
        pd_raw = data['per_target_rewards_raw'].to(self.device)
        sample_idx = torch.nonzero(measured, as_tuple=False).reshape(-1)
        metrics['causal_ccp_samples'] = float(sample_idx.numel())
        if sample_idx.numel() < self._causal_ccp_min_samples:
            return credit_advantages, metrics

        # Deterministic 80/20 split provides a genuine held-out fidelity gate.
        order = torch.arange(sample_idx.numel(), device=self.device)
        val_mask = (order % 5) == 0
        train_idx = sample_idx[~val_mask]
        val_idx = sample_idx[val_mask]
        if train_idx.numel() == 0 or val_idx.numel() == 0:
            return credit_advantages, metrics

        target_train = teacher[train_idx]
        target_scale = float(1.0 / max(
            float(target_train.std(unbiased=False).item()), 1e-3))
        target_scale = min(target_scale, 1000.0)
        self._causal_ccp.train()
        final_loss = 0.0
        for _ in range(self._causal_ccp_epochs):
            permutation = train_idx[torch.randperm(
                train_idx.numel(), device=self.device)]
            for start in range(0, permutation.numel(), 256):
                idx = permutation[start:start + 256]
                prediction = self._causal_ccp(
                    obs[idx], actions_comm[idx], actions_comm_rate[idx])
                loss = torch.nn.functional.smooth_l1_loss(
                    prediction * target_scale,
                    teacher[idx] * target_scale)
                self._causal_ccp_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self._causal_ccp.parameters(), self.max_grad_norm)
                self._causal_ccp_optimizer.step()
                final_loss = float(loss.item())
        metrics['causal_ccp_loss'] = final_loss

        self._causal_ccp.eval()
        with torch.no_grad():
            val_prediction = self._causal_ccp(
                obs[val_idx], actions_comm[val_idx],
                actions_comm_rate[val_idx])
            pred_flat = val_prediction.reshape(-1)
            target_flat = teacher[val_idx].reshape(-1)
            pred_centered = pred_flat - pred_flat.mean()
            target_centered = target_flat - target_flat.mean()
            denominator = torch.sqrt(
                pred_centered.square().sum()
                * target_centered.square().sum()).clamp_min(1e-12)
            corr = float((pred_centered * target_centered).sum().item()
                         / denominator.item())
            informative = target_flat.abs() >= self._causal_ccp_min_effect
            if bool(informative.any()):
                sign_accuracy = float((
                    torch.sign(pred_flat[informative])
                    == torch.sign(target_flat[informative])
                ).float().mean().item())
            else:
                sign_accuracy = 0.5
            reliability = float(np.clip(max(corr, 0.0), 0.0, 1.0))
            reliability *= float(np.clip(
                (sign_accuracy - 0.5) / 0.5, 0.0, 1.0))
            all_prediction = self._causal_ccp(
                obs, actions_comm, actions_comm_rate)

        metrics['causal_ccp_val_corr'] = corr
        metrics['causal_ccp_val_sign_accuracy'] = sign_accuracy
        metrics['causal_ccp_reliability'] = reliability
        metrics['causal_ccp_teacher_abs_effect'] = float(
            teacher[sample_idx].abs().mean().item())

        # Measured interventions remain authoritative. Unmeasured predictions
        # are admitted only when held-out fidelity is non-trivial.
        effects = all_prediction * reliability
        effects = torch.where(measured.unsqueeze(-1), teacher, effects)
        temperature = self._causal_ccp_tail_temperature
        tail_weights = torch.softmax(-pd_raw / temperature, dim=-1)
        causal_score = (effects * tail_weights).sum(dim=-1)
        active = actions_comm_rate > 0
        informative_measured = measured & (
            teacher.abs().amax(dim=-1) >= self._causal_ccp_min_effect)
        usable = informative_measured.clone()
        if reliability >= 0.25:
            usable = usable | active
        usable = usable & active
        if int(usable.sum().item()) < 2:
            return credit_advantages, metrics

        score_used = causal_score[usable]
        score_std = score_used.std(unbiased=False)
        if float(score_std.item()) < 1e-8:
            return credit_advantages, metrics
        causal_normalized = torch.zeros_like(causal_score)
        causal_normalized[usable] = (
            score_used - score_used.mean()) / (score_std + 1e-8)

        routed = credit_advantages.clone()
        mix = self._causal_ccp_message_mix
        routed[usable, 1] = (
            (1.0 - mix) * routed[usable, 1]
            + mix * causal_normalized[usable])
        metrics['causal_ccp_credit_fraction'] = float(
            usable.float().mean().item())
        return routed, metrics

    def _update_temporal_unrolled_power(
        self,
        agent,
        obs: torch.Tensor,
        data: Dict[str, torch.Tensor],
        comm_round_phases: torch.Tensor,
    ) -> Tuple[Dict[str, float], bool]:
        """Differentiate Pareto/CVaR losses through multi-frame PDHG.

        This is one joint actor step, not an environment-side optimizer.  The
        physical tensors come from the fixed structure actually executed in
        each rollout frame; actor outputs define both the RF split and the
        finite-iteration primal initialization.  Episode boundaries terminate
        the BPTT chain.
        """
        names = (
            'negative_worst_detection',
            'negative_mean_detection',
            'communication_load',
            'sensing_energy',
            'temporal_switching',
        )
        metrics = {
            'temporal_unrolled_power_applied': 0.0,
            'temporal_unrolled_power_windows': 0.0,
            'temporal_unrolled_power_frames': 0.0,
            'temporal_unrolled_power_eligible_frames': 0.0,
            'temporal_unrolled_power_skipped_frames': 0.0,
            'temporal_unrolled_power_coverage': 0.0,
            'temporal_unrolled_power_constraint_loss': 0.0,
            'temporal_unrolled_power_cvar_residual_max': 0.0,
            'temporal_unrolled_power_raw_cvar_residual_max': 0.0,
            'temporal_unrolled_power_model_bias_mae': 0.0,
            'temporal_unrolled_power_budget_violation_w': 0.0,
            'temporal_unrolled_power_primal_violation': 0.0,
            'temporal_unrolled_power_stationarity': 0.0,
            'temporal_unrolled_power_dual_residual': 0.0,
            'temporal_unrolled_power_pareto_gradient_norm': 0.0,
            'temporal_unrolled_power_constraint_gradient_norm': 0.0,
            'temporal_unrolled_power_applied_gradient_norm': 0.0,
            'temporal_unrolled_power_gradient_finite': 1.0,
            'temporal_unrolled_power_update_time_s': 0.0,
            'temporal_unrolled_power_pareto_time_s': 0.0,
            'temporal_unrolled_power_cross_frame_bptt': float(
                not self._temporal_unrolled_power_detach_between_frames),
            'temporal_unrolled_power_actor_proximal_regularization': float(
                getattr(
                    self.cfg.marl,
                    'distributed_primal_dual_power_actor_proximal_'
                    'regularization',
                    0.0)),
            'temporal_unrolled_power_batched_vjp': float(
                self._temporal_unrolled_power_batched_vjp_enabled),
            'temporal_unrolled_power_rf_tangent_enabled': float(
                self._temporal_unrolled_power_rf_tangent_enabled),
            'temporal_unrolled_power_rf_tangent_active': 0.0,
            'temporal_unrolled_power_rf_resource_balance': 0.0,
            'temporal_unrolled_power_rf_tangent_gradient_norm': 0.0,
            'temporal_unrolled_power_rf_projection_mode': 0.0,
            'temporal_unrolled_power_rf_projection_rank': 0.0,
            'temporal_unrolled_power_rf_projection_norm_ratio': 1.0,
            'temporal_feasible_structure_enabled': float(
                self._temporal_feasible_structure_enabled),
            # Straight-through/discrete gradients are not claimed.  This flag
            # makes the deliberately biased convex-hull surrogate explicit in
            # every run record that enables it.
            'temporal_feasible_structure_biased_surrogate': float(
                self._temporal_feasible_structure_enabled),
            'temporal_feasible_structure_hard_forward': float(
                self._temporal_feasible_structure_enabled
                and self._temporal_feasible_structure_hard_forward),
            'temporal_feasible_structure_candidate_count_max': 0.0,
            'temporal_feasible_structure_empty_frames': 0.0,
            'temporal_feasible_structure_entropy': 0.0,
            'temporal_feasible_structure_hard_edge_agreement': 0.0,
            'temporal_feasible_structure_convex_violation': 0.0,
        }
        for name in names:
            metrics[f'temporal_unrolled_power_objective_{name}'] = 0.0
            metrics[f'temporal_unrolled_power_pareto_weight_{name}'] = 0.0
            metrics[
                f'temporal_unrolled_power_objective_grad_norm_{name}'] = 0.0
        metrics['temporal_unrolled_power_objective_grad_cosine_min'] = 0.0
        metrics['temporal_unrolled_power_objective_grad_cosine_max'] = 0.0
        metrics['temporal_unrolled_power_pareto_zero_weight_count'] = 0.0
        metrics['temporal_unrolled_power_model_bias_mean'] = 0.0
        metrics['temporal_unrolled_power_model_bias_std'] = 0.0
        metrics['temporal_unrolled_power_model_bias_max_abs'] = 0.0
        metrics['temporal_unrolled_power_real_detection_mean'] = 0.0
        metrics['temporal_unrolled_power_raw_detection_mean'] = 0.0
        records = list(getattr(
            self, '_rollout_temporal_power_records', ()))
        if not records:
            return metrics, False
        expected_rows = int(self.buffer.ptr)
        by_row = {
            int(record['buffer_row']): dict(record) for record in records}
        if len(by_row) != expected_rows or set(by_row) != set(range(expected_rows)):
            raise RuntimeError(
                'temporal unroll requires exactly one physical record per '
                'rollout buffer row')
        if self._temporal_feasible_structure_enabled:
            for record in by_row.values():
                if not bool(record.get('valid', False)):
                    continue
                try:
                    enumerate_feasible_single_role_structures(
                        np.asarray(record['coefficient']) > 0.0,
                        target_pair_limit=int(record['target_pair_limit']),
                        reports_per_receiver=int(
                            record['reports_per_receiver']),
                        max_structures=(
                            self._temporal_feasible_structure_max_structures),
                    )
                except ValueError as error:
                    if 'no complete single-role structure' not in str(error):
                        raise
                    # Empty physical feasible set is a real system state.  It
                    # terminates the temporal segment; inventing a tiny-gain
                    # edge here would create both false sensing and a false
                    # structure gradient.  Enumeration-overflow remains a
                    # hard RuntimeError and is intentionally not swallowed.
                    record['valid'] = False
                    record['empty_structure_feasible_set'] = True
        eligible_frames = sum(bool(record.get('valid', False))
                              for record in by_row.values())
        metrics['temporal_feasible_structure_empty_frames'] = float(sum(
            bool(record.get('empty_structure_feasible_set', False))
            for record in by_row.values()))
        metrics['temporal_unrolled_power_eligible_frames'] = float(
            eligible_frames)
        metrics['temporal_unrolled_power_skipped_frames'] = float(
            expected_rows - eligible_frames)
        metrics['temporal_unrolled_power_coverage'] = float(
            eligible_frames / max(expected_rows, 1))
        if not (hasattr(agent.actor, 'communication_parameters')
                and hasattr(agent.actor, 'isac_resource_parameters')):
            raise RuntimeError(
                'temporal unroll requires communication and ISAC actor heads')
        if len(self._rollout_pd) != expected_rows:
            raise RuntimeError(
                'temporal unroll requires one real detection vector per '
                'rollout buffer row')
        real_detection = torch.as_tensor(
            np.stack(self._rollout_pd), dtype=obs.dtype, device=self.device)

        training_input = obs
        window_mask = None
        if 'obs_window' in data:
            training_input = data['obs_window'].to(self.device)
            if 'window_mask' in data:
                window_mask = data['window_mask'].to(self.device)
        h_prev = None
        if 'h_prev' in data:
            h_full = data['h_prev'].to(self.device)
            h_prev = h_full.reshape(1, -1, h_full.shape[-1])
        flat_size = obs.shape[0]
        agent_identity = torch.arange(
            flat_size, dtype=torch.long, device=self.device) % self.K
        _, _, _, comm_mean, _, _ = agent.actor(
            training_input,
            h_prev,
            window_mask=window_mask,
            detach_h_new=True,
            comm_round_phase=comm_round_phases,
            agent_identity=agent_identity,
        )
        _, rate_logits_flat = agent.actor.communication_parameters(comm_mean)
        power_logits_flat, _, sensing_logits_flat, _ = (
            agent.actor.isac_resource_parameters(comm_mean))
        if (power_logits_flat.shape != (flat_size,)
                or sensing_logits_flat.shape != (flat_size, self.Q)):
            raise RuntimeError('actor resource heads have incompatible shapes')
        power_logits = power_logits_flat.reshape(expected_rows, self.K)
        sensing_logits = sensing_logits_flat.reshape(
            expected_rows, self.K, self.Q)
        rate_logits = rate_logits_flat.reshape(
            expected_rows, self.K, -1)

        sequences = []
        for environment in range(self.num_envs):
            pending = []
            environment_records = sorted(
                (record for record in by_row.values()
                 if int(record['environment']) == environment),
                key=lambda record: int(record['buffer_row']),
            )
            for record in environment_records:
                if not bool(record.get('valid', False)):
                    if len(pending) >= 2:
                        sequences.append(tuple(pending))
                    pending = []
                    continue
                pending.append(int(record['buffer_row']))
                if (len(pending) >= self._temporal_unrolled_power_horizon
                        or bool(record['terminal'])):
                    if len(pending) >= 2:
                        sequences.append(tuple(pending))
                    pending = []
            if len(pending) >= 2:
                sequences.append(tuple(pending))
        if not sequences:
            return metrics, False

        layer = DifferentiableTemporalPowerUnroll(
            inner_iterations=self._temporal_unrolled_power_inner_iterations,
            warm_start_mix=self._temporal_unrolled_power_warm_start_mix,
            power_cost_per_watt=float(getattr(
                self.cfg.marl,
                'distributed_primal_dual_power_cost_per_watt', 1.0e-4)),
            quadratic_regularization=float(getattr(
                self.cfg.marl,
                'distributed_primal_dual_power_regularization', 1.0e-3)),
            actor_proximal_regularization=float(getattr(
                self.cfg.marl,
                'distributed_primal_dual_power_actor_proximal_regularization',
                0.0)),
            detach_between_frames=(
                self._temporal_unrolled_power_detach_between_frames),
        )
        objective_sets = []
        detections = []
        raw_detections = []
        real_detections = []
        model_biases = []
        trajectory_results = []
        structure_results = []
        structure_agreements = []
        structure_convex_violations = []
        resource_balance_residuals = []
        for sequence in sequences:
            index = torch.as_tensor(
                sequence, dtype=torch.long, device=self.device)
            sequence_records = [by_row[row] for row in sequence]
            requirement = torch.as_tensor(
                np.stack([
                    record['requirement'] for record in sequence_records]),
                dtype=obs.dtype, device=self.device)
            total_budget = torch.as_tensor(
                np.stack([
                    record['total_rf_budget_w']
                    for record in sequence_records]),
                dtype=obs.dtype, device=self.device)
            sensing_cap = torch.as_tensor(
                np.stack([
                    record['sensing_cap_w'] for record in sequence_records]),
                dtype=obs.dtype, device=self.device)
            structural_activation = None
            if self._temporal_feasible_structure_enabled:
                pair_limits = {
                    int(record['target_pair_limit'])
                    for record in sequence_records}
                receiver_limits = {
                    int(record['reports_per_receiver'])
                    for record in sequence_records}
                if len(pair_limits) != 1 or len(receiver_limits) != 1:
                    raise RuntimeError(
                        'temporal structure limits changed inside one horizon')
                coefficient = torch.as_tensor(
                    np.stack([
                        record['coefficient']
                        for record in sequence_records]),
                    dtype=obs.dtype, device=self.device)
                planned_budget, _, _ = actor_sensing_budget(
                    power_logits.index_select(0, index),
                    rate_logits.index_select(0, index),
                    total_budget,
                    sensing_cap,
                    communication_fraction_min=float(getattr(
                        self.cfg.marl, 'comm_power_fraction_min', 0.0)),
                    communication_fraction_max=float(getattr(
                        self.cfg.marl, 'comm_power_fraction_max', 1.0)),
                )
                structure_result = temporal_feasible_structure_mixture(
                    sensing_logits.index_select(0, index),
                    coefficient,
                    target_pair_limit=next(iter(pair_limits)),
                    reports_per_receiver=next(iter(receiver_limits)),
                    temperature=(
                        self._temporal_feasible_structure_temperature),
                    inertia=self._temporal_feasible_structure_inertia,
                    max_structures=(
                        self._temporal_feasible_structure_max_structures),
                    detach_between_frames=(
                        self._temporal_unrolled_power_detach_between_frames),
                    transmitter_budget_w=planned_budget,
                )
                structure_results.append(structure_result)
                schedule = (
                    straight_through_structure(structure_result)
                    if self._temporal_feasible_structure_hard_forward
                    else structure_result.mixture)
                # Optional hard schedule in the forward pass; when selected,
                # the mixture's Jacobian remains available to the Pareto/CVaR
                # actor update.  The soft mode is retained as an ablation for
                # the within-frame time-sharing interpretation.
                gain = torch.sum(coefficient * schedule, dim=2)
                structural_activation = schedule.sum(dim=2).clamp(0.0, 1.0)
                mask = gain > 0.0
                executed = torch.as_tensor(
                    np.stack([
                        record['executed_edge_mask']
                        for record in sequence_records]),
                    dtype=torch.bool, device=self.device)
                hard = structure_result.hard_schedule > 0.5
                intersection = torch.sum(hard & executed, dim=(1, 2, 3))
                union = torch.sum(hard | executed, dim=(1, 2, 3))
                structure_agreements.append(torch.where(
                    union > 0,
                    intersection.to(obs.dtype) / union.to(obs.dtype),
                    torch.ones_like(union, dtype=obs.dtype)))
                mixture = structure_result.mixture
                pair_limit = float(next(iter(pair_limits)))
                receiver_limit = float(next(iter(receiver_limits)))
                target_load = mixture.sum(dim=(1, 2))
                receiver_load = mixture.sum(dim=(1, 3))
                diagonal = mixture[:, torch.arange(self.K),
                                   torch.arange(self.K), :]
                convex_violation = torch.stack((
                    torch.relu(1.0 - target_load).amax(),
                    torch.relu(target_load - pair_limit).amax(),
                    torch.relu(receiver_load - receiver_limit).amax(),
                    diagonal.abs().amax(),
                )).amax()
                structure_convex_violations.append(convex_violation)
            else:
                gain = torch.as_tensor(
                    np.stack([
                        record['gain'] for record in sequence_records]),
                    dtype=obs.dtype, device=self.device)
                mask = torch.as_tensor(
                    np.stack([
                        record['mask'] for record in sequence_records]),
                    dtype=torch.bool, device=self.device)
            result = layer(
                power_logits.index_select(0, index),
                sensing_logits.index_select(0, index),
                rate_logits.index_select(0, index),
                gain,
                requirement,
                mask,
                total_budget,
                sensing_cap,
                p_fa=float(self.cfg.detection.P_FA),
                communication_fraction_min=float(getattr(
                    self.cfg.marl, 'comm_power_fraction_min', 0.0)),
                communication_fraction_max=float(getattr(
                    self.cfg.marl, 'comm_power_fraction_max', 1.0)),
                structural_activation=structural_activation,
            )
            trajectory_results.append(result)
            normalized_sensing = result.power_w.sum(dim=-1) / (
                total_budget.clamp_min(torch.finfo(obs.dtype).eps))
            resource_balance_residuals.append(torch.mean(
                result.communication_fraction + normalized_sensing - 1.0))
            real_sequence_detection = real_detection.index_select(0, index)
            model_bias = (
                real_sequence_detection - result.detection_probability)
            # Zeroth-order consistency with the sampled physical system;
            # stop-gradient preserves the unrolled model's local Jacobian.
            calibrated_detection = (
                result.detection_probability + model_bias.detach())
            calibrated_result = replace(
                result, detection_probability=calibrated_detection)
            objective_sets.append(temporal_power_objectives(
                calibrated_result))
            detections.append(calibrated_detection)
            raw_detections.append(result.detection_probability)
            real_detections.append(real_sequence_detection)
            model_biases.append(model_bias.detach())

        objective_losses = tuple(torch.stack([
            getattr(objectives, name) for objectives in objective_sets
        ]).mean() for name in names)
        unrolled_detection = torch.cat(detections, dim=0)
        cvar_residual = empirical_detection_cvar_residual(
            unrolled_detection,
            self._constrained_cvar_qos_floor,
            tail_fraction=self._constrained_cvar_tail_fraction,
            sample_dim=0,
        )
        raw_cvar_residual = empirical_detection_cvar_residual(
            torch.cat(raw_detections, dim=0),
            self._constrained_cvar_qos_floor,
            tail_fraction=self._constrained_cvar_tail_fraction,
            sample_dim=0,
        )
        constraint_loss = inequality_augmented_lagrangian(
            cvar_residual,
            self._constrained_cvar_multipliers,
            penalty=self._constrained_cvar_penalty,
        )
        resource_balance = torch.stack(resource_balance_residuals).mean()
        resource_active = bool(
            self._temporal_unrolled_power_rf_tangent_enabled
            and float(resource_balance.detach().abs())
            <= self._temporal_unrolled_power_rf_tangent_slack_tolerance)
        tangent_constraints = ((resource_balance,) if resource_active else ())
        agent.actor_optimizer.zero_grad()
        pareto_started = time.perf_counter()
        assigned = assign_constrained_pareto_gradients(
            objective_losses,
            agent.actor.parameters(),
            constraint_loss=constraint_loss,
            tangent_constraint_losses=tangent_constraints,
            normalize_objective_gradients=True,
            tangent_projection_rank_tolerance=(
                self._temporal_unrolled_power_rf_tangent_rank_tolerance),
            batched_vjp=self._temporal_unrolled_power_batched_vjp_enabled,
            tolerance=self._temporal_unrolled_power_pareto_tolerance,
            max_iterations=(
                self._temporal_unrolled_power_pareto_max_iterations),
        )
        pareto_time_s = float(time.perf_counter() - pareto_started)
        clipped_norm = nn.utils.clip_grad_norm_(
            agent.actor.parameters(), self.max_grad_norm)
        finite = bool(
            torch.isfinite(constraint_loss.detach())
            and torch.isfinite(assigned.applied_gradient_norm)
            and torch.isfinite(torch.as_tensor(clipped_norm))
            and all(
                parameter.grad is None
                or bool(torch.isfinite(parameter.grad).all())
                for parameter in agent.actor.parameters()))
        if finite:
            agent.actor_optimizer.step()

        metrics.update({
            'temporal_unrolled_power_applied': float(finite),
            'temporal_unrolled_power_windows': float(len(sequences)),
            'temporal_unrolled_power_frames': float(sum(
                len(sequence) for sequence in sequences)),
            'temporal_unrolled_power_constraint_loss': float(
                constraint_loss.detach().item()),
            'temporal_unrolled_power_rf_resource_balance': float(
                resource_balance.detach().item()),
            'temporal_unrolled_power_rf_tangent_active': float(resource_active),
            'temporal_unrolled_power_rf_tangent_gradient_norm': float(
                assigned.equality_constraint_gradient_norm.detach().item()
                if assigned.equality_constraint_gradient_norm is not None
                else 0.0),
            'temporal_unrolled_power_rf_projection_mode': float({
                'identity': 0,
                'single': 1,
                'multi': 2,
            }.get(assigned.projection_mode, -1)),
            'temporal_unrolled_power_rf_projection_rank': float(
                assigned.projection_rank),
            'temporal_unrolled_power_rf_projection_norm_ratio': float(
                assigned.projection_norm_ratio.detach().item()
                if assigned.projection_norm_ratio is not None else 1.0),
            'temporal_unrolled_power_cvar_residual_max': float(
                cvar_residual.detach().max().item()),
            'temporal_unrolled_power_raw_cvar_residual_max': float(
                raw_cvar_residual.detach().max().item()),
            'temporal_unrolled_power_model_bias_mae': float(
                torch.cat(model_biases, dim=0).abs().mean().item()),
            'temporal_unrolled_power_model_bias_mean': float(
                torch.cat(model_biases, dim=0).mean().item()),
            'temporal_unrolled_power_model_bias_std': float(
                torch.cat(model_biases, dim=0).std(unbiased=False).item()),
            'temporal_unrolled_power_model_bias_max_abs': float(
                torch.cat(model_biases, dim=0).abs().max().item()),
            'temporal_unrolled_power_real_detection_mean': float(
                torch.cat(real_detections, dim=0).mean().item()),
            'temporal_unrolled_power_raw_detection_mean': float(
                torch.cat(raw_detections, dim=0).mean().item()),
            'temporal_unrolled_power_budget_violation_w': float(max(
                result.budget_violation_w.detach().max().item()
                for result in trajectory_results)),
            'temporal_unrolled_power_primal_violation': float(max(
                result.primal_violation.detach().max().item()
                for result in trajectory_results)),
            'temporal_unrolled_power_stationarity': float(max(
                result.primal_stationarity.detach().max().item()
                for result in trajectory_results)),
            'temporal_unrolled_power_dual_residual': float(max(
                result.dual_residual.detach().max().item()
                for result in trajectory_results)),
            'temporal_unrolled_power_pareto_gradient_norm': float(
                assigned.pareto.gradient_norm.detach().item()),
            'temporal_unrolled_power_constraint_gradient_norm': float(
                assigned.constraint_gradient_norm.detach().item()),
            'temporal_unrolled_power_applied_gradient_norm': float(
                assigned.applied_gradient_norm.detach().item()),
            'temporal_unrolled_power_gradient_finite': float(finite),
            'temporal_unrolled_power_pareto_converged': float(
                assigned.pareto.converged),
            'temporal_unrolled_power_pareto_iterations': float(
                assigned.pareto.iterations),
            'temporal_unrolled_power_pareto_gap': float(
                assigned.pareto.frank_wolfe_gap.detach().item()),
            'temporal_unrolled_power_pareto_time_s': pareto_time_s,
        })
        if structure_results:
            metrics.update({
                'temporal_feasible_structure_candidate_count_max': float(max(
                    result.candidate_count.detach().max().item()
                    for result in structure_results)),
                'temporal_feasible_structure_entropy': float(torch.cat([
                    result.entropy.detach()
                    for result in structure_results]).mean().item()),
                'temporal_feasible_structure_hard_edge_agreement': float(
                    torch.cat(structure_agreements).mean().item()),
                'temporal_feasible_structure_convex_violation': float(max(
                    value.detach().item()
                    for value in structure_convex_violations)),
            })
        weights = assigned.pareto.weights.detach()
        if (assigned.objective_gradient_norms is not None
                and assigned.objective_gradient_cosine is not None):
            raw_norms = assigned.objective_gradient_norms.detach()
            cosine = assigned.objective_gradient_cosine.detach()
            metrics['temporal_unrolled_power_objective_grad_cosine_min'] = (
                float(cosine.min().item()))
            metrics['temporal_unrolled_power_objective_grad_cosine_max'] = (
                float(cosine.max().item()))
            metrics['temporal_unrolled_power_pareto_zero_weight_count'] = (
                float(torch.count_nonzero(
                    assigned.pareto.weights.detach() <= 1.0e-8).item()))
            for first in range(len(names)):
                for second in range(first + 1, len(names)):
                    metrics[
                        'temporal_unrolled_power_objective_grad_cosine_'
                        f'{names[first]}__{names[second]}'] = float(
                            cosine[first, second].item())
            for objective_index, name in enumerate(names):
                metrics[
                    f'temporal_unrolled_power_objective_grad_norm_{name}'] = (
                        float(raw_norms[objective_index].item()))
        for objective_index, name in enumerate(names):
            metrics[f'temporal_unrolled_power_objective_{name}'] = float(
                objective_losses[objective_index].detach().item())
            metrics[f'temporal_unrolled_power_pareto_weight_{name}'] = float(
                weights[objective_index].item())
        return metrics, not finite

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
        actions_comm = data.get('actions_comm')
        actions_comm_rate = data.get('actions_comm_rate')
        actions_isac_power_raw = data.get('actions_isac_power_raw')
        actions_sensing_raw = data.get('actions_sensing_raw')
        movement_action_masks = data['movement_action_masks'].to(self.device)
        comm_round_phases = data['comm_round_phases'].to(self.device)
        if actions_comm is not None:
            actions_comm = actions_comm.to(self.device)
        if actions_comm_rate is not None:
            actions_comm_rate = actions_comm_rate.to(self.device)
        if actions_isac_power_raw is not None:
            actions_isac_power_raw = actions_isac_power_raw.to(self.device)
        if actions_sensing_raw is not None:
            actions_sensing_raw = actions_sensing_raw.to(self.device)
        old_log_probs = data['old_log_probs'].to(self.device)
        advantages = data['advantages'].to(self.device)
        returns = data['returns'].to(self.device)
        old_values = data['old_values'].to(self.device)
        per_target_rewards_raw = data.get('per_target_rewards_raw')
        if per_target_rewards_raw is not None:
            per_target_rewards_raw = per_target_rewards_raw.to(self.device)
        if self._set_risk_critic_enabled:
            if per_target_rewards_raw is None:
                raise RuntimeError(
                    'set risk critic requires raw per-target P_D')
        old_head_log_probs = None
        credit_advantages = None
        credit_returns = None
        causal_ccp_metrics = {}
        if self._headwise_credit_enabled:
            required = {
                'old_head_log_probs', 'credit_advantages', 'credit_returns'}
            missing = required.difference(data)
            if missing:
                raise RuntimeError(
                    f'headwise credit data missing from rollout: {sorted(missing)}')
            old_head_log_probs = data['old_head_log_probs'].to(self.device)
            credit_advantages = data['credit_advantages'].to(self.device)
            credit_returns = data['credit_returns'].to(self.device)
            if self._causal_ccp_enabled:
                if actions_comm is None or actions_comm_rate is None:
                    raise RuntimeError(
                        'causal CCP requires stored communication actions')
                credit_advantages, causal_ccp_metrics = (
                    self._fit_ccp_and_route_message_credit(
                        data, obs, actions_comm, actions_comm_rate,
                        credit_advantages))

        total_size = obs.shape[0]
        indices = np.arange(total_size)

        metrics = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'credit_critic_loss': 0.0,
            'risk_critic_quantile_loss': 0.0,
            'risk_critic_constraint_loss': 0.0,
            'risk_critic_violation_accuracy': 0.0,
            'risk_critic_violation_balanced_accuracy': 0.0,
            'risk_critic_predicted_cvar': 0.0,
            'risk_critic_empirical_violation_rate': 0.0,
            'entropy': 0.0,
            'approx_kl': 0.0,
            'clip_fraction': 0.0,
            'comm_qos_rate_aux_loss': 0.0,
            'comm_rate_exploration_aux_loss': 0.0,
            '_n_minibatches': 0,
        }
        for head_name in self.buffer.credit_head_names:
            metrics[f'actor_loss_{head_name}'] = 0.0
            metrics[f'approx_kl_{head_name}'] = 0.0
            metrics[f'clip_fraction_{head_name}'] = 0.0

        agent = self.agents[0]  # shared networks

        # Treat the complete actor update (PPO plus allocation/rate auxiliary
        # steps) as a transaction. A final full-rollout KL check below either
        # accepts it or restores both parameters and optimizer moments.
        actor_update_snapshot = None
        actor_optimizer_snapshots = []
        if (not self._risk_critic_only_training
                and self.target_kl is not None
                and float(self.target_kl) > 0.0):
            actor_update_snapshot = deepcopy(agent.actor.state_dict())
            seen_optimizers = set()
            for optimizer in (
                    agent.actor_optimizer,
                    self._target_allocation_aux_optimizer,
                    self._comm_rate_aux_optimizer):
                if optimizer is None or id(optimizer) in seen_optimizers:
                    continue
                seen_optimizers.add(id(optimizer))
                actor_optimizer_snapshots.append(
                    (optimizer, deepcopy(optimizer.state_dict())))

        qos_rate_aux_active = False
        qos_target_rate_index = 0
        exploration_target_rate_index = 0
        if self._comm_qos_enabled and self._rollout_comm_qos_values:
            rollout_qos = np.mean(
                np.asarray(self._rollout_comm_qos_values), axis=0)
            qos_rate_aux_active = bool(np.any(
                rollout_qos < self._comm_qos_targets))
            rate_levels = list(getattr(
                self.cfg.marl, 'comm_rate_bits_per_dim', [0, 4, 8, 16]))
            eligible = [idx for idx, bits in enumerate(rate_levels)
                        if bits >= self._comm_qos_min_rate_bits]
            qos_target_rate_index = (
                eligible[0] if eligible else len(rate_levels) - 1)
            exploration_eligible = [
                idx for idx, bits in enumerate(rate_levels)
                if bits >= self._comm_rate_bonus_target_bits
            ]
            exploration_target_rate_index = (
                exploration_eligible[0]
                if exploration_eligible else len(rate_levels) - 1)

        # Ensure train mode (eval may have been called between updates)
        if self._risk_critic_only_training:
            agent.actor.eval()
        else:
            agent.actor.train()
        agent.critic.train()

        # Pre-compute target-routed advantages ONCE before PPO epochs. This
        # keeps the tail selection invariant to minibatch split and shuffle.
        tw_advantages = None
        if (self._adv_mode == 'target_wise'
                and 'per_target_advantages' in data):
            tw_advantages = self._compute_target_wise_advantage(
                obs, data['per_target_advantages'].to(self.device),
                tau_d=self._resp_tau_m)
        elif self._adv_mode == 'bottleneck_risk':
            required = {'per_target_advantages', 'per_target_rewards_raw'}
            missing = required.difference(data)
            if missing:
                raise RuntimeError(
                    'bottleneck risk advantage missing rollout fields: '
                    f'{sorted(missing)}')
            tw_advantages = compute_bottleneck_risk_advantage(
                data['per_target_advantages'].to(self.device),
                data['per_target_rewards_raw'].to(self.device),
                advantages,
                tail_fraction=self._risk_tail_fraction,
                temperature=self._risk_target_temperature,
                target_floor=self._risk_target_floor,
                scalar_mix=self._risk_scalar_mix,
            )
            # NOTE: risk_advantage_tail_fraction is a per-update scalar (not a
            # per-minibatch loss); it is added AFTER the averaging loop below so
            # it is not divided by n_minibatches (audit 2026-08-17).

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
            _check_comm = (actions_comm[_check_idx]
                           if actions_comm is not None else None)
            _check_comm_rate = (actions_comm_rate[_check_idx]
                                if actions_comm_rate is not None else None)
            _check_power = (actions_isac_power_raw[_check_idx]
                            if actions_isac_power_raw is not None else None)
            _check_sensing = (actions_sensing_raw[_check_idx]
                               if actions_sensing_raw is not None else None)
            _check_movement_mask = movement_action_masks[_check_idx]
            _check_round_phase = comm_round_phases[_check_idx]
            _check_agent_identity = torch.as_tensor(
                _check_idx % self.K, dtype=torch.long, device=self.device)
            _check_h = None
            if 'h_prev' in data:
                _check_h_full = data['h_prev'][_check_idx]
                _check_h = _check_h_full.reshape(1, -1, _check_h_full.shape[-1]).to(self.device)

            _check_w = None
            _check_wm = None
            if 'obs_window' in data:
                _check_w = data['obs_window'][_check_idx].to(self.device)
            if 'window_mask' in data:
                _check_wm = data['window_mask'][_check_idx].to(self.device)

            passed, max_diff = agent.verify_old_log_prob_consistency(
                _check_w if _check_w is not None else _check_obs,
                _check_dp, _check_role, _check_old_lp,
                h_prev=_check_h,
                window_mask=_check_wm,
                actions_comm=_check_comm,
                actions_comm_rate=_check_comm_rate,
                actions_isac_power_raw=_check_power,
                actions_sensing_raw=_check_sensing,
                movement_action_mask=_check_movement_mask,
                comm_round_phase=_check_round_phase,
                agent_identity=_check_agent_identity,
            )
            if not passed and self._risk_critic_only_training:
                print(
                    '[PPO RATIO DIAGNOSTIC] frozen actor differs by '
                    f'{max_diff:.6f}; ignored because no actor parameter or '
                    'PPO advantage is updated')
            elif not passed:
                print(f'[PPO RATIO ERROR] old_log_prob != recomputed_log_prob: '
                      f'max|diff|={max_diff:.6f} > tolerance=1e-4')
                print(f'  → PPO ratio r_t ≠ 1 before any optimizer step. '
                      f'Training results are CONTAMINATED.')
                print(f'  → Likely cause: GRU h_prev mismatch between rollout and update, '
                      f'or dp_scale mismatch between ActionSpace.decode and evaluate_actions.')
            else:
                print(f'[PPO RATIO OK] old_log_prob matches recomputed: max|diff|={max_diff:.6f} < 1e-4')

        kl_stop = False
        rejected_minibatch_kl = 0.0
        constrained_cvar_loss_value = 0.0
        constrained_cvar_gradient_norm = 0.0
        constrained_cvar_gradient_finite = True
        constrained_cvar_empirical_residual = None
        force_actor_rollback = False
        for epoch in range(self.ppo_epochs):
            if kl_stop:
                break
            np.random.shuffle(indices)

            for start in range(0, total_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]

                mb_obs = obs[mb_idx]
                agent_ids_mb = torch.as_tensor(
                    mb_idx % self.K, dtype=torch.long, device=self.device)
                # Critic input: global state (MAPPO/CTDE) or local obs (IPPO) + agent one-hot.
                # obs index b → timestep row = b // K, agent = b % K
                mb_gs = build_aligned_critic_input(
                    global_states=global_states,
                    local_obs=mb_obs,
                    flat_indices=mb_idx,
                    num_agents=self.K,
                    centralized=self.centralized_critic,
                    base_state_dim=(
                        self.agents[0]._critic_base_state_dim),
                    comm_dim=16,
                )

                mb_actions_dp = actions_dp[mb_idx]
                mb_actions_role = actions_role[mb_idx]
                mb_actions_comm = (actions_comm[mb_idx]
                                   if actions_comm is not None else None)
                mb_actions_comm_rate = (actions_comm_rate[mb_idx]
                                        if actions_comm_rate is not None else None)
                mb_actions_isac_power_raw = (
                    actions_isac_power_raw[mb_idx]
                    if actions_isac_power_raw is not None else None)
                mb_actions_sensing_raw = (
                    actions_sensing_raw[mb_idx]
                    if actions_sensing_raw is not None else None)
                mb_movement_action_mask = movement_action_masks[mb_idx]
                mb_round_phase = comm_round_phases[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_old_head_log_probs = (
                    old_head_log_probs[mb_idx]
                    if old_head_log_probs is not None else None)
                mb_credit_advantages = (
                    credit_advantages[mb_idx]
                    if credit_advantages is not None else None)
                mb_credit_returns = (
                    credit_returns[mb_idx]
                    if credit_returns is not None else None)
                # advantages already normalized once globally in buffer.get_training_data();
                # do NOT re-normalize per minibatch (that was a double normalization).
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_old_values = old_values[mb_idx]
                mb_per_target_rewards_raw = (
                    per_target_rewards_raw[mb_idx]
                    if per_target_rewards_raw is not None else None)

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
                evaluated = agent.evaluate_actions(
                    mb_window if mb_window is not None else mb_obs,
                    mb_gs, mb_actions_dp, mb_actions_role,
                    h_prev=mb_h_prev,
                    window_mask=mb_wmask,
                    actions_comm=mb_actions_comm,
                    actions_comm_rate=mb_actions_comm_rate,
                    actions_isac_power_raw=mb_actions_isac_power_raw,
                    actions_sensing_raw=mb_actions_sensing_raw,
                    movement_action_mask=mb_movement_action_mask,
                    comm_round_phase=mb_round_phase,
                    agent_identity=agent_ids_mb,
                    return_log_prob_components=self._headwise_credit_enabled,
                )
                if self._headwise_credit_enabled:
                    (new_log_probs, values, entropies, dp_means, _, comm_batch,
                     head_outputs) = evaluated
                    new_head_log_probs = head_outputs['log_probs']
                    credit_value_predictions = head_outputs['credit_values']
                else:
                    (new_log_probs, values, entropies, dp_means, _,
                     comm_batch) = evaluated
                    new_head_log_probs = None
                    credit_value_predictions = None

                ratio, approx_kl_mb_tensor = (
                    stable_ppo_ratio_and_approx_kl(
                        new_log_probs, mb_old_log_probs))
                approx_kl_mb = float(approx_kl_mb_tensor.item())
                kl_limit = 1.5 * float(self.target_kl)
                if (not np.isfinite(approx_kl_mb)
                        or (float(self.target_kl) > 0.0
                            and approx_kl_mb > kl_limit)):
                    # PPO clipping bounds the surrogate objective, not the
                    # actual parameter step or resulting policy divergence.
                    # Reject this minibatch before any gradient update.
                    kl_stop = True
                    rejected_minibatch_kl = approx_kl_mb
                    break
                if self._headwise_credit_enabled:
                    head_ratios, _ = stable_ppo_ratio_and_approx_kl(
                        new_head_log_probs, mb_old_head_log_probs)
                    head_losses = []
                    for head_idx in range(len(self.buffer.credit_head_names)):
                        ratio_h = head_ratios[:, head_idx]
                        # A target-conditioned movement policy must receive the
                        # corresponding per-target critic/GAE signal. Other
                        # action heads retain their delayed causal credits.
                        if head_idx == 0 and tw_advantages is not None:
                            advantage_h = mb_advantages
                        else:
                            advantage_h = mb_credit_advantages[:, head_idx]
                        surr1_h = ratio_h * advantage_h
                        surr2_h = torch.clamp(
                            ratio_h, 1.0 - self.ppo_clip,
                            1.0 + self.ppo_clip) * advantage_h
                        head_losses.append(-torch.min(
                            surr1_h, surr2_h).mean())
                    coef_t = torch.as_tensor(
                        self._headwise_credit_coefs,
                        dtype=head_losses[0].dtype, device=self.device)
                    stacked_head_losses = torch.stack(head_losses)
                    actor_loss = torch.sum(
                        coef_t * stacked_head_losses) / torch.clamp(
                            coef_t.sum(), min=1e-8)
                else:
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(
                        ratio, 1.0 - self.ppo_clip,
                        1.0 + self.ppo_clip) * mb_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    head_ratios = None
                    head_losses = None

                # Critic loss: Huber (smooth L1), robust to return outliers
                critic_loss = torch.nn.functional.smooth_l1_loss(
                    values, mb_returns
                )
                if self._headwise_credit_enabled:
                    credit_critic_loss = torch.nn.functional.smooth_l1_loss(
                        credit_value_predictions, mb_credit_returns)
                else:
                    credit_critic_loss = torch.zeros((), device=self.device)
                risk_quantile_loss = torch.zeros((), device=self.device)
                risk_constraint_loss = torch.zeros((), device=self.device)
                risk_violation_accuracy = torch.zeros(
                    (), device=self.device)
                risk_violation_balanced_accuracy = torch.zeros(
                    (), device=self.device)
                risk_predicted_cvar = torch.zeros((), device=self.device)
                risk_empirical_violation_rate = torch.zeros(
                    (), device=self.device)
                if self._set_risk_critic_enabled:
                    risk_outputs = agent.critic.forward_risk(mb_gs)
                    risk_quantiles, risk_constraint_logits = risk_outputs
                    risk_quantile_loss = quantile_huber_loss(
                        risk_quantiles, mb_per_target_rewards_raw)
                    violation_target = (
                        mb_per_target_rewards_raw
                        < self._risk_critic_qos_floor).to(
                            risk_constraint_logits.dtype)
                    positive_count = violation_target.sum()
                    negative_count = (
                        violation_target.numel() - positive_count)
                    configured_positive_weight = (
                        self._risk_critic_constraint_positive_weight)
                    if configured_positive_weight > 0:
                        positive_weight = torch.as_tensor(
                            configured_positive_weight,
                            dtype=risk_constraint_logits.dtype,
                            device=self.device)
                    elif (float(positive_count.item()) > 0
                          and float(negative_count.item()) > 0):
                        positive_weight = torch.clamp(
                            negative_count / positive_count, min=1.0, max=20.0)
                    else:
                        positive_weight = torch.ones(
                            (), dtype=risk_constraint_logits.dtype,
                            device=self.device)
                    risk_constraint_loss = (
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            risk_constraint_logits, violation_target,
                            pos_weight=positive_weight))
                    with torch.no_grad():
                        predicted_violation = risk_constraint_logits >= 0
                        positive_mask = violation_target >= 0.5
                        negative_mask = ~positive_mask
                        risk_violation_accuracy = (
                            predicted_violation == positive_mask
                        ).float().mean()
                        sensitivity = (
                            predicted_violation[positive_mask].float().mean()
                            if bool(positive_mask.any())
                            else torch.ones((), device=self.device))
                        specificity = (
                            (~predicted_violation[negative_mask]).float().mean()
                            if bool(negative_mask.any())
                            else torch.ones((), device=self.device))
                        risk_violation_balanced_accuracy = (
                            0.5 * (sensitivity + specificity))
                        tail_count = max(1, int(np.ceil(
                            agent.critic.risk_cvar_alpha
                            * agent.critic.risk_num_quantiles)))
                        risk_predicted_cvar = torch.sort(
                            risk_quantiles, dim=-1
                        ).values[..., :tail_count].mean()
                        risk_empirical_violation_rate = (
                            violation_target.mean())

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
                elif self._comm_cost_aware:
                    # PPO and the environment transport cost provide the main
                    # learning signal. A small magnitude regularizer prevents
                    # unnecessarily large analogue message actions.
                    loss_comm = 1e-4 * comm_batch.pow(2).mean()
                    rate_aux_loss = torch.zeros((), device=self.device)
                    rate_exploration_aux_loss = torch.zeros(
                        (), device=self.device)
                    if (qos_rate_aux_active
                            and self._comm_qos_rate_aux_coef > 0.0):
                        _, rate_logits = (
                            agent.actor.communication_parameters(
                                comm_batch.detach()))
                        rate_target = torch.full(
                            (rate_logits.shape[0],), qos_target_rate_index,
                            dtype=torch.long, device=self.device)
                        rate_aux_loss = torch.nn.functional.cross_entropy(
                            rate_logits, rate_target)
                        loss_comm = (
                            loss_comm
                            + self._comm_qos_rate_aux_coef * rate_aux_loss)
                    if (qos_rate_aux_active
                            and self._comm_rate_bonus_enabled
                            and self._comm_rate_bonus_aux_coef > 0.0):
                        _, rate_logits = (
                            agent.actor.communication_parameters(
                                comm_batch.detach()))
                        exploration_target = torch.full(
                            (rate_logits.shape[0],),
                            exploration_target_rate_index,
                            dtype=torch.long, device=self.device)
                        rate_exploration_aux_loss = (
                            torch.nn.functional.cross_entropy(
                                rate_logits, exploration_target))
                        loss_comm = (
                            loss_comm
                            + self._comm_rate_bonus_aux_coef
                            * rate_exploration_aux_loss)
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
                    + self._headwise_credit_vf_coef * credit_critic_loss
                    + self._risk_critic_quantile_coef * risk_quantile_loss
                    + self._risk_critic_constraint_coef * risk_constraint_loss
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

                if not self._risk_critic_only_training:
                    agent.actor_optimizer.step()
                agent.critic_optimizer.step()

                if (qos_rate_aux_active
                        and self._comm_rate_bonus_enabled
                        and not self._risk_critic_only_training
                        and self._comm_rate_aux_optimizer is not None):
                    self._comm_rate_aux_optimizer.zero_grad()
                    _, aux_rate_logits = (
                        agent.actor.communication_parameters(
                            comm_batch.detach()))
                    aux_target = torch.full(
                        (aux_rate_logits.shape[0],),
                        exploration_target_rate_index,
                        dtype=torch.long, device=self.device)
                    aux_rate_loss = torch.nn.functional.cross_entropy(
                        aux_rate_logits, aux_target)
                    (self._comm_rate_bonus_aux_coef * aux_rate_loss).backward()
                    nn.utils.clip_grad_norm_(
                        self._comm_rate_head_module.parameters(),
                        self.max_grad_norm)
                    self._comm_rate_aux_optimizer.step()

                # Track metrics
                metrics['actor_loss'] += actor_loss.item()
                metrics['critic_loss'] += critic_loss.item()
                metrics['credit_critic_loss'] += credit_critic_loss.item()
                metrics['risk_critic_quantile_loss'] += float(
                    risk_quantile_loss.item())
                metrics['risk_critic_constraint_loss'] += float(
                    risk_constraint_loss.item())
                metrics['risk_critic_violation_accuracy'] += float(
                    risk_violation_accuracy.item())
                metrics['risk_critic_violation_balanced_accuracy'] += float(
                    risk_violation_balanced_accuracy.item())
                metrics['risk_critic_predicted_cvar'] += float(
                    risk_predicted_cvar.item())
                metrics['risk_critic_empirical_violation_rate'] += float(
                    risk_empirical_violation_rate.item())
                metrics['entropy'] += entropy.item()
                metrics['approx_kl'] += approx_kl_mb
                metrics['clip_fraction'] += ((ratio < 1.0 - self.ppo_clip) | (ratio > 1.0 + self.ppo_clip)).float().mean().item()
                if self._headwise_credit_enabled:
                    for head_idx, head_name in enumerate(
                            self.buffer.credit_head_names):
                        ratio_h = head_ratios[:, head_idx]
                        metrics[f'actor_loss_{head_name}'] += float(
                            head_losses[head_idx].item())
                        metrics[f'approx_kl_{head_name}'] += float((
                            (ratio_h - 1.0) - torch.log(ratio_h)
                        ).mean().item())
                        metrics[f'clip_fraction_{head_name}'] += float((
                            (ratio_h < 1.0 - self.ppo_clip)
                            | (ratio_h > 1.0 + self.ppo_clip)
                        ).float().mean().item())
                if self._comm_cost_aware:
                    metrics['comm_qos_rate_aux_loss'] += float(
                        rate_aux_loss.item())
                    metrics['comm_rate_exploration_aux_loss'] += float(
                        rate_exploration_aux_loss.item())
                metrics['_n_minibatches'] += 1

        # The physical detection outcome belongs to the complete team action.
        # Random PPO minibatches intentionally remain unchanged; the opt-in
        # condition gets one separate full-rollout actor pass and aggregates
        # the factorized likelihood ratio over all K UAVs before CVaR.
        if (self._constrained_cvar_enabled
                and not self._temporal_unrolled_power_enabled
                and not self._risk_critic_only_training
                and not kl_stop):
            if len(self._rollout_pd) * self.K != total_size:
                raise RuntimeError(
                    'constrained CVaR requires one complete team detection '
                    'vector per rollout transition')
            rollout_pd = torch.as_tensor(
                np.stack(self._rollout_pd), dtype=torch.float32,
                device=self.device)
            constrained_cvar_empirical_residual = (
                empirical_detection_cvar_residual(
                    rollout_pd,
                    self._constrained_cvar_qos_floor,
                    tail_fraction=self._constrained_cvar_tail_fraction,
                    sample_dim=0,
                ).detach())
            constraint_new_log_probs = []
            validation_batch_size = max(1, self.minibatch_size)
            for start in range(0, total_size, validation_batch_size):
                end = min(start + validation_batch_size, total_size)
                batch_idx = np.arange(start, end)
                constraint_obs = obs[batch_idx]
                constraint_gs = build_aligned_critic_input(
                    global_states=global_states,
                    local_obs=constraint_obs,
                    flat_indices=batch_idx,
                    num_agents=self.K,
                    centralized=self.centralized_critic,
                    base_state_dim=self.agents[0]._critic_base_state_dim,
                    comm_dim=16,
                )
                constraint_h = None
                if 'h_prev' in data:
                    constraint_h_full = data['h_prev'][batch_idx].to(
                        self.device)
                    constraint_h = constraint_h_full.reshape(
                        1, -1, constraint_h_full.shape[-1])
                constraint_window = None
                constraint_window_mask = None
                if 'obs_window' in data:
                    constraint_window = data['obs_window'][batch_idx].to(
                        self.device)
                if 'window_mask' in data:
                    constraint_window_mask = data['window_mask'][
                        batch_idx].to(self.device)
                constraint_agent_identity = torch.as_tensor(
                    batch_idx % self.K, dtype=torch.long,
                    device=self.device)
                constraint_evaluated = agent.evaluate_actions(
                    (constraint_window if constraint_window is not None
                     else constraint_obs),
                    constraint_gs,
                    actions_dp[batch_idx],
                    actions_role[batch_idx],
                    h_prev=constraint_h,
                    window_mask=constraint_window_mask,
                    actions_comm=(actions_comm[batch_idx]
                                  if actions_comm is not None else None),
                    actions_comm_rate=(actions_comm_rate[batch_idx]
                                       if actions_comm_rate is not None
                                       else None),
                    actions_isac_power_raw=(
                        actions_isac_power_raw[batch_idx]
                        if actions_isac_power_raw is not None else None),
                    actions_sensing_raw=(
                        actions_sensing_raw[batch_idx]
                        if actions_sensing_raw is not None else None),
                    movement_action_mask=movement_action_masks[batch_idx],
                    comm_round_phase=comm_round_phases[batch_idx],
                    agent_identity=constraint_agent_identity,
                )
                constraint_new_log_probs.append(constraint_evaluated[0])
            flattened_pd = rollout_pd.repeat_interleave(self.K, dim=0)
            cvar_surrogate_residual = flattened_team_detection_cvar_residual(
                flattened_pd,
                self._constrained_cvar_qos_floor,
                torch.cat(constraint_new_log_probs),
                old_log_probs,
                num_agents=self.K,
                tail_fraction=self._constrained_cvar_tail_fraction,
            )
            constrained_cvar_loss = inequality_augmented_lagrangian(
                cvar_surrogate_residual,
                self._constrained_cvar_multipliers,
                penalty=self._constrained_cvar_penalty,
            )
            agent.actor_optimizer.zero_grad()
            constrained_cvar_loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                agent.actor.parameters(), self.max_grad_norm)
            constrained_cvar_loss_value = float(
                constrained_cvar_loss.detach().item())
            constrained_cvar_gradient_norm = float(
                torch.as_tensor(gradient_norm).detach().item())
            constrained_cvar_gradient_finite = bool(
                np.isfinite(constrained_cvar_loss_value)
                and np.isfinite(constrained_cvar_gradient_norm)
                and all(
                    parameter.grad is None
                    or bool(torch.isfinite(parameter.grad).all())
                    for parameter in agent.actor.parameters()))
            if constrained_cvar_gradient_finite:
                agent.actor_optimizer.step()
            else:
                # The final transaction check restores all preceding PPO
                # minibatches and optimizer moments as one atomic update.
                force_actor_rollback = True

        # Average metrics (use ACTUAL minibatch count, not expected)
        n_actual = max(metrics.pop('_n_minibatches', 1), 1)
        for k in metrics:
            metrics[k] /= max(n_actual, 1)
        if getattr(self, '_risk_tail_fraction', None) is not None:
            metrics['risk_advantage_tail_fraction'] = (
                self._risk_tail_fraction)
        metrics.update(causal_ccp_metrics)
        metrics['kl_early_stop'] = float(kl_stop)
        metrics['kl_rejected_minibatch'] = float(rejected_minibatch_kl)

        # When enabled, this replaces the detached-outcome CVaR actor pass
        # above.  The QoS residual is now evaluated on a differentiable
        # multi-frame physical trajectory, while the same on-policy rollout
        # still drives the projected multiplier update below.
        if (self._temporal_unrolled_power_enabled
                and not self._risk_critic_only_training
                and not kl_stop):
            temporal_update_started = time.perf_counter()
            temporal_metrics, temporal_failed = (
                self._update_temporal_unrolled_power(
                    agent, obs, data, comm_round_phases))
            temporal_metrics['temporal_unrolled_power_update_time_s'] = (
                float(time.perf_counter() - temporal_update_started))
            metrics.update(temporal_metrics)
            force_actor_rollback = force_actor_rollback or temporal_failed
            constrained_cvar_loss_value = temporal_metrics[
                'temporal_unrolled_power_constraint_loss']
            constrained_cvar_gradient_norm = temporal_metrics[
                'temporal_unrolled_power_applied_gradient_norm']
            constrained_cvar_gradient_finite = bool(
                temporal_metrics[
                    'temporal_unrolled_power_gradient_finite'])

        # One full-team auxiliary step preserves the (transition, UAV) grouping
        # that random PPO minibatches destroy. Centralized state is used only to
        # construct training labels; execution still uses each local actor and
        # whatever messages were actually delivered over the physical U2U link.
        if (not self._risk_critic_only_training
                and not kl_stop
                and (self._target_allocation_enabled
             or self._target_allocation_teacher_enabled)
                and hasattr(agent.actor, 'last_target_assignment')):
            aux_input = obs
            aux_mask = None
            if 'obs_window' in data:
                aux_input = data['obs_window'].to(self.device)
                if 'window_mask' in data:
                    aux_mask = data['window_mask'].to(self.device)
            aux_h = None
            if 'h_prev' in data:
                aux_h_full = data['h_prev'].to(self.device)
                aux_h = aux_h_full.reshape(1, -1, aux_h_full.shape[-1])

            teacher_labels = teacher_dp = None
            if self._target_allocation_teacher_enabled:
                teacher_fn = (
                    compute_qos_bistatic_assignment_teacher
                    if self._target_allocation_teacher_mode == 'qos_bistatic'
                    else compute_balanced_assignment_teacher)
                teacher_kwargs = dict(
                        global_states=global_states,
                        num_agents=self.K,
                        num_targets=self.Q,
                        region_size=self.cfg.scenario.region_size,
                        max_dp=(self.cfg.uav.v_max
                                * self.cfg.scenario.dt),
                        num_envs=self.num_envs,
                        transition_masks=data.get('masks'),
                        switching_penalty_m=(
                            self._target_allocation_teacher_switching_penalty_m),
                    )
                if self._target_allocation_teacher_mode == 'qos_bistatic':
                    teacher_kwargs.update(
                        commitment_frames=(
                            self._target_allocation_teacher_commitment_frames),
                        qos_floor=self._target_allocation_teacher_qos_floor,
                        qos_weight=self._target_allocation_teacher_qos_weight,
                        height_m=self._target_allocation_teacher_height_m,
                    )
                teacher_labels, teacher_dp = teacher_fn(**teacher_kwargs)
            aux_epochs = (self._target_allocation_teacher_epochs
                          if self._target_allocation_teacher_enabled
                          else self._target_allocation_aux_epochs)
            assignment_probs = None
            transition_masks = data.get('masks')
            if transition_masks is not None:
                transition_masks = transition_masks.to(self.device)
            pd_targets_all = data.get('per_target_rewards_raw')
            if pd_targets_all is not None:
                pd_targets_all = pd_targets_all.to(self.device)
            allocation_selector = torch.ones(
                obs.shape[0], dtype=torch.bool, device=self.device)
            teacher_crisis_fraction = 1.0
            if self._target_allocation_movement_only:
                allocation_selector = movement_action_masks > 0.5
                # Team-grouped losses require complete K-row teams.  Rollout
                # scheduling sets the same phase for every UAV in an env.
                if int(allocation_selector.sum().item()) % self.K != 0:
                    raise RuntimeError(
                        'movement-only allocation mask split a UAV team')
            if (self._target_allocation_teacher_enabled
                    and self._target_allocation_teacher_crisis_only_enabled):
                if pd_targets_all is None:
                    raise RuntimeError(
                        'crisis-only teacher requires per-target rollout QoS')
                crisis_rows = (
                    pd_targets_all.amin(dim=-1)
                    < self._target_allocation_teacher_crisis_floor)
                allocation_selector = allocation_selector & crisis_rows
                teacher_crisis_fraction = float(
                    crisis_rows.float().mean().item())
                if int(allocation_selector.sum().item()) % self.K != 0:
                    raise RuntimeError(
                        'crisis-only teacher split a UAV team')
            temporal_loss = torch.zeros((), device=self.device)
            comm_sensing_aux_loss = torch.zeros((), device=self.device)
            comm_sensing_counterfactual_loss = torch.zeros(
                (), device=self.device)
            comm_sensing_weak_boost = torch.zeros((), device=self.device)
            modular_balance_loss = torch.zeros((), device=self.device)
            modular_specialization_loss = torch.zeros(
                (), device=self.device)
            modular_route_entropy = torch.zeros((), device=self.device)
            modular_route_usage_span = torch.zeros((), device=self.device)
            aux_agent_identity = torch.arange(
                obs.shape[0], dtype=torch.long,
                device=self.device) % self.K
            for _ in range(aux_epochs):
                agent.actor_optimizer.zero_grad()
                if self._target_allocation_aux_optimizer is not None:
                    self._target_allocation_aux_optimizer.zero_grad()
                training_input = aux_input
                use_differentiable_comm = (
                    (self._target_allocation_teacher_enabled
                     and self._target_allocation_teacher_differentiable_comm)
                    or (self._target_allocation_enabled
                        and self._target_allocation_differentiable_comm)
                )
                if use_differentiable_comm:
                    _, _, _, sender_comm, _, _ = agent.actor(
                        aux_input, aux_h, window_mask=aux_mask,
                        detach_h_new=True,
                        comm_round_phase=comm_round_phases,
                        agent_identity=aux_agent_identity)
                    sender_token_masks = getattr(
                        agent.actor, 'last_outgoing_token_mask', None)
                    rate_levels = list(getattr(
                        self.cfg.marl, 'comm_rate_bits_per_dim',
                        [0, 4, 8, 16]))
                    training_input = build_differentiable_u2u_inbox(
                        aux_input,
                        sender_comm,
                        agent.actor,
                        num_agents=self.K,
                        rate_index=qos_target_rate_index,
                        rate_bits=rate_levels[qos_target_rate_index],
                        num_rate_levels=len(rate_levels),
                        delay_teams=(
                            self.num_envs
                            * self._target_allocation_comm_delay_decisions
                            if self._target_allocation_enabled else 0),
                        transition_masks=transition_masks,
                        sender_token_masks=sender_token_masks,
                    )
                aux_dp_mean, _, _, aux_comm_mean, _, _ = agent.actor(
                    training_input, aux_h, window_mask=aux_mask,
                    detach_h_new=True,
                    comm_round_phase=comm_round_phases,
                    agent_identity=aux_agent_identity)
                assignment_probs = agent.actor.last_target_assignment
                if self._target_allocation_teacher_enabled:
                    movement_probs = getattr(
                        agent.actor, 'last_movement_assignment', None)
                    if movement_probs is not None:
                        assignment_probs = movement_probs
                if assignment_probs is None:
                    break
                assignment_for_loss = assignment_probs[allocation_selector]
                if assignment_for_loss.numel() == 0:
                    assignment_probs = None
                    break
                transition_masks_for_loss = (
                    transition_masks[allocation_selector]
                    if transition_masks is not None else None)
                if self._target_allocation_teacher_enabled:
                    local_bid_logits = getattr(
                        agent.actor, 'last_v2_local_bid_logits', None)
                    if (local_bid_logits is not None
                            and local_bid_logits.shape
                            == assignment_probs.shape):
                        label_loss = torch.nn.functional.cross_entropy(
                            local_bid_logits[allocation_selector],
                            teacher_labels[allocation_selector],
                        )
                    else:
                        label_loss = torch.nn.functional.nll_loss(
                            torch.log(
                                assignment_for_loss.clamp_min(1e-8)),
                            teacher_labels[allocation_selector],
                        )
                    max_dp = self.cfg.uav.v_max * self.cfg.scenario.dt
                    movement_loss = torch.nn.functional.smooth_l1_loss(
                        torch.tanh(aux_dp_mean[allocation_selector]),
                        teacher_dp[allocation_selector]
                        / max(float(max_dp), 1e-9),
                    )
                    if hasattr(agent.actor, 'intent_head'):
                        message_loss = torch.nn.functional.cross_entropy(
                            agent.actor.intent_head(
                                aux_comm_mean[allocation_selector]),
                            teacher_labels[allocation_selector],
                        )
                    else:
                        message_loss = torch.zeros((), device=self.device)
                    allocation_loss = (
                        self._target_allocation_teacher_label_coef * label_loss
                        + self._target_allocation_teacher_movement_coef
                        * movement_loss
                        + self._target_allocation_teacher_message_coef
                        * message_loss
                    )
                else:
                    balance_probs = getattr(
                        agent.actor, 'last_target_assignment_st', None)
                    if balance_probs is None:
                        balance_probs = assignment_for_loss
                    else:
                        balance_probs = balance_probs[allocation_selector]
                    balance_loss, _ = (
                        compute_target_allocation_regularizer(
                            balance_probs, self.K))
                    _, commitment_loss = (
                        compute_target_allocation_regularizer(
                            assignment_for_loss, self.K))
                    temporal_loss = compute_target_allocation_temporal_loss(
                        assignment_for_loss,
                        num_agents=self.K,
                        delay_teams=(self.num_envs
                                     * max(self._target_allocation_comm_delay_decisions, 1)),
                        transition_masks=transition_masks_for_loss,
                    )
                    allocation_loss = (
                        self._target_allocation_balance_coef * balance_loss
                        + self._target_allocation_commit_coef * commitment_loss
                        + self._target_allocation_temporal_coef * temporal_loss
                    )
                    if self._sparse_claim_enabled:
                        sparse_underload_loss, sparse_overload_loss = (
                            compute_sparse_endpoint_load_loss(
                                assignment_for_loss,
                                num_agents=self.K,
                                commitment_topk=(
                                    self._sparse_claim_commit_topk),
                                desired_endpoints=(
                                    self._sparse_claim_desired_endpoints),
                            ))
                        allocation_loss = (
                            allocation_loss
                            + self._sparse_claim_underload_coef
                            * sparse_underload_loss
                            + self._sparse_claim_overload_coef
                            * sparse_overload_loss
                        )
                if (self._comm_aided_sensing_enabled
                        and pd_targets_all is not None
                        and hasattr(agent.actor, 'isac_resource_parameters')):
                    # Learn sensing itself: low-P_D targets receive more of the
                    # fixed per-UAV sensing budget.
                    _, _, comm_sensing_mean, _ = (
                        agent.actor.isac_resource_parameters(aux_comm_mean))
                    comm_sensing_probs = torch.softmax(
                        comm_sensing_mean, dim=-1)
                    sensing_probs_for_loss = (
                        comm_sensing_probs[allocation_selector])
                    pd_for_sensing = pd_targets_all[allocation_selector]
                    desired_sensing = torch.softmax(
                        -pd_for_sensing
                        / self._comm_aided_sensing_temperature,
                        dim=-1).detach()
                    # Supervise only the message-induced residual. The base
                    # sensing head remains under PPO control, preventing this
                    # auxiliary from replacing the sensing policy itself.
                    comm_sensing_residual = getattr(
                        agent.actor, 'last_comm_sensing_logits', None)
                    if comm_sensing_residual is not None:
                        desired_residual = (
                            desired_sensing
                            - desired_sensing.mean(dim=-1, keepdim=True))
                        comm_sensing_aux_loss = (
                            torch.nn.functional.smooth_l1_loss(
                                comm_sensing_residual[allocation_selector],
                                desired_residual))

                    # Learn communication-assisted sensing explicitly. The new
                    # residual is structurally zero under a no-token virtual
                    # intervention, so its centered value is an uncontaminated
                    # message-only effect (unlike a full actor re-forward, which
                    # also changes the legacy assignment adapter).
                    slices = getattr(agent.actor, '_obs_slices', None)
                    if (self._comm_aided_sensing_counterfactual_coef > 0.0
                            and slices is not None
                            and slices.comm_mask_len > 0
                            and comm_sensing_residual is not None):
                        comm_view = (training_input if training_input.ndim == 2
                                     else training_input[:, -1])
                        valid_comm_rows = (
                            comm_view[:, slices.comm_mask_start:
                                      slices.comm_mask_start
                                      + slices.comm_mask_len] > 0.5
                        ).any(dim=-1)[allocation_selector]
                        weakest_target = pd_for_sensing.argmin(
                            dim=-1, keepdim=True)
                        centered_comm_residual = (
                            comm_sensing_residual[allocation_selector]
                            - comm_sensing_residual[allocation_selector].mean(
                                dim=-1, keepdim=True))
                        weak_boost = centered_comm_residual.gather(
                            1, weakest_target).squeeze(-1)
                        if valid_comm_rows.any():
                            selected_boost = weak_boost[valid_comm_rows]
                            comm_sensing_weak_boost = selected_boost.mean()
                            comm_sensing_counterfactual_loss = torch.relu(
                                self._comm_aided_sensing_margin
                                - selected_boost).mean()
                    allocation_loss = (
                        allocation_loss
                        + self._comm_aided_sensing_aux_coef
                        * comm_sensing_aux_loss
                        + self._comm_aided_sensing_counterfactual_coef
                        * comm_sensing_counterfactual_loss
                    )
                module_routing = getattr(
                    agent.actor, 'last_v2_module_routing', None)
                if module_routing is not None:
                    selected_routing = module_routing[allocation_selector]
                    if selected_routing.numel() > 0:
                        num_experts = selected_routing.shape[-1]
                        mean_usage = selected_routing.mean(dim=0)
                        uniform_usage = 1.0 / float(num_experts)
                        modular_balance_loss = (
                            float(num_experts)
                            * (mean_usage - uniform_usage).square().sum())
                        modular_route_entropy = (
                            -(selected_routing.clamp_min(1e-8)
                              * selected_routing.clamp_min(1e-8).log())
                            .sum(dim=-1).mean()
                            / np.log(float(num_experts)))
                        # Minimizing normalized entropy encourages a UAV to
                        # choose a distinct coordination path; the aggregate
                        # balance term prevents all UAVs choosing the same one.
                        modular_specialization_loss = modular_route_entropy
                        modular_route_usage_span = (
                            mean_usage.max() - mean_usage.min())
                        allocation_loss = (
                            allocation_loss
                            + self._v2_modular_balance_coef
                            * modular_balance_loss
                            + self._v2_modular_specialization_coef
                            * modular_specialization_loss)
                # Selective-plasticity probes may intentionally freeze every
                # target-allocation parameter while leaving the PPO rate head
                # trainable. In that case this auxiliary is observational and
                # has no graph; attempting backward would abort an otherwise
                # valid rate-only update.
                if allocation_loss.requires_grad:
                    allocation_loss.backward()
                    if self._target_allocation_aux_optimizer is not None:
                        aux_params = [
                            p for group in self._target_allocation_aux_optimizer.param_groups
                            for p in group['params']]
                        nn.utils.clip_grad_norm_(aux_params, self.max_grad_norm)
                        self._target_allocation_aux_optimizer.step()
                    else:
                        nn.utils.clip_grad_norm_(
                            agent.actor.parameters(), self.max_grad_norm)
                        agent.actor_optimizer.step()

            if assignment_probs is not None:
                metrics['target_allocation_loss'] = float(
                    allocation_loss.item())
                if self._target_allocation_teacher_enabled:
                    with torch.no_grad():
                        teacher_prediction = (
                            local_bid_logits[allocation_selector].argmax(
                                dim=-1)
                            if local_bid_logits is not None
                            else assignment_for_loss.argmax(dim=-1))
                        accuracy = (
                            teacher_prediction
                            == teacher_labels[allocation_selector]
                        ).float().mean()
                    metrics['target_teacher_label_loss'] = float(
                        label_loss.item())
                    metrics['target_teacher_movement_loss'] = float(
                        movement_loss.item())
                    metrics['target_teacher_message_loss'] = float(
                        message_loss.item())
                    metrics['target_teacher_accuracy'] = float(accuracy.item())
                    metrics['target_teacher_crisis_fraction'] = float(
                        teacher_crisis_fraction)
                    if getattr(
                            agent.actor, 'last_v2_module_routing', None
                    ) is not None:
                        metrics['v2_modular_balance_loss'] = float(
                            modular_balance_loss.item())
                        metrics['v2_modular_route_entropy'] = float(
                            modular_route_entropy.item())
                        metrics['v2_modular_route_usage_span'] = float(
                            modular_route_usage_span.item())
                        residual_norm = getattr(
                            agent.actor,
                            'last_v2_module_residual_norm',
                            None)
                        if residual_norm is not None:
                            metrics['v2_modular_residual_norm'] = float(
                                residual_norm.mean().item())
                else:
                    metrics['target_allocation_balance'] = float(
                        balance_loss.item())
                    metrics['target_allocation_entropy'] = float(
                        commitment_loss.item())
                    metrics['target_allocation_temporal_loss'] = float(
                        temporal_loss.item())
                    if self._sparse_claim_enabled:
                        metrics['sparse_claim_underload_loss'] = float(
                            sparse_underload_loss.item())
                        metrics['sparse_claim_overload_loss'] = float(
                            sparse_overload_loss.item())
                    if self._comm_aided_sensing_enabled:
                        metrics['comm_aided_sensing_aux_loss'] = float(
                            comm_sensing_aux_loss.item())
                        metrics[
                            'comm_aided_sensing_counterfactual_loss'] = float(
                                comm_sensing_counterfactual_loss.item())
                        metrics['comm_aided_sensing_weak_boost'] = float(
                            comm_sensing_weak_boost.item())
                    with torch.no_grad():
                        assignment_teams = assignment_for_loss.reshape(
                            -1, self.K, self.Q)
                        assignment_choice = assignment_teams.argmax(dim=-1)
                        assignment_load = torch.nn.functional.one_hot(
                            assignment_choice, num_classes=self.Q).sum(dim=1)
                        metrics['target_allocation_unique_fraction'] = float(
                            ((assignment_load > 0).sum(dim=-1).float()
                             / max(min(self.K, self.Q), 1)).mean().item())
                        metrics['target_allocation_collision_rate'] = float(
                            (assignment_load > 1).any(dim=-1).float().mean().item())

        # Validate the final actor, including auxiliary steps, against the
        # rollout policy. Minibatch early stopping cannot prevent a single
        # large last step; transactional rollback closes that gap.
        post_update_kl = 0.0
        actor_update_rejected = False
        if actor_update_snapshot is not None:
            try:
                recomputed_log_probs = []
                validation_batch_size = max(1, self.minibatch_size)
                with torch.no_grad():
                    for start in range(0, total_size, validation_batch_size):
                        end = min(start + validation_batch_size, total_size)
                        batch_idx = np.arange(start, end)
                        validation_obs = obs[batch_idx]
                        validation_h = None
                        if 'h_prev' in data:
                            validation_h_full = data['h_prev'][batch_idx].to(
                                self.device)
                            validation_h = validation_h_full.reshape(
                                1, -1, validation_h_full.shape[-1])
                        validation_window = None
                        validation_window_mask = None
                        if 'obs_window' in data:
                            validation_window = data['obs_window'][
                                batch_idx].to(self.device)
                        if 'window_mask' in data:
                            validation_window_mask = data['window_mask'][
                                batch_idx].to(self.device)
                        validation_agent_identity = torch.as_tensor(
                            batch_idx % self.K, dtype=torch.long,
                            device=self.device)
                        (dp_mean, dp_log_std, role_logits, comm_mean,
                         _, _) = agent.actor(
                            (validation_window
                             if validation_window is not None
                             else validation_obs),
                            validation_h,
                            window_mask=validation_window_mask,
                            comm_round_phase=comm_round_phases[batch_idx],
                            agent_identity=validation_agent_identity,
                        )
                        if not all(bool(torch.isfinite(value).all()) for value in (
                                dp_mean, dp_log_std, role_logits, comm_mean)):
                            raise FloatingPointError(
                                'non-finite actor output during KL validation')
                        batch_log_probs, _ = (
                            agent._movement_message_resource_log_probs(
                                dp_mean, dp_log_std, role_logits, comm_mean,
                                actions_dp[batch_idx],
                                actions_role[batch_idx],
                                movement_action_masks[batch_idx],
                                (actions_comm[batch_idx]
                                 if actions_comm is not None else None),
                                (actions_comm_rate[batch_idx]
                                 if actions_comm_rate is not None else None),
                                (actions_isac_power_raw[batch_idx]
                                 if actions_isac_power_raw is not None
                                 else None),
                                (actions_sensing_raw[batch_idx]
                                 if actions_sensing_raw is not None else None),
                            ))
                        recomputed_log_probs.append(batch_log_probs)
                _, post_kl_tensor = stable_ppo_ratio_and_approx_kl(
                    torch.cat(recomputed_log_probs), old_log_probs)
                post_update_kl = float(post_kl_tensor.item())
            except FloatingPointError:
                # Convert a numerical actor failure into rollback, not a
                # poisoned checkpoint. Structural/runtime errors remain loud.
                post_update_kl = float('inf')

            kl_limit = 1.5 * float(self.target_kl)
            if (force_actor_rollback
                    or not np.isfinite(post_update_kl)
                    or post_update_kl > kl_limit):
                agent.actor.load_state_dict(actor_update_snapshot)
                for optimizer, optimizer_state in actor_optimizer_snapshots:
                    optimizer.load_state_dict(optimizer_state)
                actor_update_rejected = True

        metrics['post_update_attempted_approx_kl'] = float(post_update_kl)
        metrics['post_update_approx_kl'] = float(
            0.0 if actor_update_rejected else post_update_kl)
        metrics['actor_update_rejected'] = float(actor_update_rejected)

        if self._constrained_cvar_enabled:
            if constrained_cvar_empirical_residual is None:
                # KL may have stopped the ordinary PPO pass before the
                # condition pass. The physical rollout still supplies a valid
                # primal/dual observation and must not disappear from metrics.
                rollout_pd = torch.as_tensor(
                    np.stack(self._rollout_pd), dtype=torch.float32,
                    device=self.device)
                constrained_cvar_empirical_residual = (
                    empirical_detection_cvar_residual(
                        rollout_pd,
                        self._constrained_cvar_qos_floor,
                        tail_fraction=self._constrained_cvar_tail_fraction,
                        sample_dim=0,
                    ).detach())
            old_multipliers = self._constrained_cvar_multipliers
            self._constrained_cvar_multipliers = projected_target_dual_update(
                old_multipliers,
                constrained_cvar_empirical_residual,
                step_size=self._constrained_cvar_dual_step_size,
                maximum=self._constrained_cvar_dual_maximum,
            )
            dual_residual = float(torch.max(torch.abs(
                self._constrained_cvar_multipliers
                - old_multipliers)).item())
            primal_violation = float(torch.clamp(
                constrained_cvar_empirical_residual, min=0.0).max().item())
            max_power_violation = float(max(
                self._rollout_power_budget_violations or [0.0]))
            self._constrained_cvar_update_count += 1
            feasible = bool(
                primal_violation <= self._constrained_cvar_primal_tolerance
                and max_power_violation
                    <= self._constrained_cvar_power_tolerance_w
                and not actor_update_rejected)
            if feasible:
                self._constrained_cvar_feasible_streak += 1
                if self._constrained_cvar_first_feasible_update is None:
                    self._constrained_cvar_first_feasible_update = (
                        self._constrained_cvar_update_count)
            else:
                self._constrained_cvar_feasible_streak = 0
            metrics.update({
                'constrained_cvar_loss': constrained_cvar_loss_value,
                'constrained_cvar_gradient_norm': (
                    constrained_cvar_gradient_norm),
                'constrained_cvar_gradient_finite': float(
                    constrained_cvar_gradient_finite),
                'constrained_cvar_primal_violation': primal_violation,
                'constrained_cvar_residual_mean': float(
                    constrained_cvar_empirical_residual.mean().item()),
                'constrained_cvar_residual_max': float(
                    constrained_cvar_empirical_residual.max().item()),
                'constrained_cvar_dual_residual': dual_residual,
                'constrained_cvar_lambda_max': float(
                    self._constrained_cvar_multipliers.max().item()),
                'constrained_cvar_update_count': float(
                    self._constrained_cvar_update_count),
                'constrained_cvar_first_feasible_update': float(
                    self._constrained_cvar_first_feasible_update or 0),
                'constrained_cvar_feasible_streak': float(
                    self._constrained_cvar_feasible_streak),
                'constrained_cvar_converged': float(
                    self._constrained_cvar_feasible_streak
                    >= self._constrained_cvar_convergence_patience),
                'isac_max_power_budget_violation_w': max_power_violation,
            })

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
        metrics['reward_component_constraint_lagrangian_penalty'] = float(
            np.mean(self._rollout_lagrangian_penalties or [0.0]))
        metrics['entropy_coef'] = self.entropy_coef

        if self._comm_qos_enabled and self._rollout_comm_qos_values:
            mean_qos = np.mean(
                np.asarray(self._rollout_comm_qos_values), axis=0)
            qos_deficit = self._comm_qos_targets - mean_qos
            self._comm_qos_lambdas = np.clip(
                self._comm_qos_lambdas
                + self._comm_qos_dual_lr * qos_deficit,
                0.0,
                self._comm_qos_lambda_max,
            )
            names = ('steady', 'weak3', 'worst')
            for idx, name in enumerate(names):
                metrics[f'comm_qos_{name}'] = float(mean_qos[idx])
                metrics[f'comm_qos_lambda_{name}'] = float(
                    self._comm_qos_lambdas[idx])
            if self._rollout_comm_qos_reward_components:
                mean_reward_components = np.mean(np.asarray(
                    self._rollout_comm_qos_reward_components), axis=0)
                for idx, name in enumerate(names):
                    metrics[f'reward_component_comm_qos_{name}'] = float(
                        mean_reward_components[idx])
            metrics['comm_qos_reward'] = float(np.mean(
                self._rollout_comm_qos_rewards))
            metrics['comm_qos_rate_penalty'] = float(np.mean(
                self._rollout_comm_qos_rate_penalties
                or [0.0]))
            metrics['comm_encouragement_bonus'] = float(np.mean(
                self._rollout_comm_encouragement_bonuses
                or [0.0]))
            metrics['comm_rate_exploration_bonus'] = float(np.mean(
                self._rollout_comm_rate_bonuses or [0.0]))
            metrics['comm_qos_feasible'] = float(np.all(
                mean_qos >= self._comm_qos_targets))
        metrics['comm_long_silence_penalty'] = float(np.mean(
            self._rollout_comm_silence_penalties or [0.0]))
        metrics['comm_sender_delivery_penalty'] = float(np.mean(
            self._rollout_comm_sender_delivery_penalties or [0.0]))
        metrics['comm_max_silence_streak'] = float(np.max(
            self._rollout_comm_silence_streaks or [0.0]))
        for name, values in self._rollout_reward_components.items():
            metrics[f'reward_component_{name}'] = float(np.mean(values))
        if self._headwise_credit_enabled:
            head_rewards = self.buffer.credit_rewards[:self.buffer.ptr]
            for head_idx, name in enumerate(self.buffer.credit_head_names):
                metrics[f'headwise_reward_{name}'] = float(np.mean(
                    head_rewards[:, :, head_idx]))
        rate_total = int(np.sum(self._rollout_comm_rate_counts))
        if rate_total > 0:
            rate_dist = self._rollout_comm_rate_counts / rate_total
            rate_levels = np.asarray(getattr(
                self.cfg.marl, 'comm_rate_bits_per_dim', [0, 4, 8, 16]),
                dtype=np.float64)
            metrics['learned_comm_rate_distribution'] = rate_dist.tolist()
            metrics['learned_comm_mean_bits_per_dim'] = float(
                np.dot(rate_dist, rate_levels))

        # CVaR Lagrangian update
        if hasattr(self, '_cvar_lambda') and self._rollout_cvar_deficits:
            mean_cvar = float(np.mean(self._rollout_cvar_deficits))
            cvar_epsilon = getattr(self, '_cvar_epsilon', 0.05)
            self._cvar_lambda = float(np.clip(
                self._cvar_lambda + 0.01 * (mean_cvar - cvar_epsilon), 0.0, 2.0))
            metrics['cvar_deficit'] = mean_cvar
            metrics['cvar_lambda'] = self._cvar_lambda
        metrics['reward_component_cvar_penalty'] = float(np.mean(
            self._rollout_cvar_penalties or [0.0]))
        if self._rollout_pd:
            diagnostic_cvar_residual = empirical_detection_cvar_residual(
                torch.as_tensor(
                    np.stack(self._rollout_pd), dtype=torch.float32,
                    device=self.device),
                self._constrained_cvar_qos_floor,
                tail_fraction=self._constrained_cvar_tail_fraction,
                sample_dim=0,
            ).detach()
            metrics['detection_cvar_primal_violation'] = float(
                torch.clamp(diagnostic_cvar_residual, min=0.0).max().item())
            metrics['detection_cvar_residual_mean'] = float(
                diagnostic_cvar_residual.mean().item())
        metrics.setdefault(
            'isac_max_power_budget_violation_w',
            float(max(self._rollout_power_budget_violations or [0.0])))
        for metric_name, values in getattr(
                self, '_rollout_isac_resource_metrics', {}).items():
            if values:
                values_array = np.asarray(values, dtype=np.float64)
                metrics[f'{metric_name}_per_frame'] = float(
                    np.mean(values_array))
                metrics[f'{metric_name}_max_per_frame'] = float(
                    np.max(values_array))

        # Surface bounded temporal power-controller health in the training
        # record.  Max aggregation is used for residuals/stationarity so a
        # single unsafe frame cannot be hidden by an episode mean; rates and
        # iteration counts use a mean for throughput accounting.
        residual_metric_names = {
            'distributed_primal_dual_power_selected_violation',
            'distributed_primal_dual_power_candidate_violation',
            'distributed_primal_dual_power_candidate_stationarity',
            'distributed_primal_dual_power_candidate_dual_residual',
            'distributed_primal_dual_power_consensus_residual',
            'distributed_primal_dual_power_target_price_max',
        }
        for metric_name, values in getattr(
                self, '_rollout_distributed_power_metrics', {}).items():
            if not values:
                continue
            values_array = np.asarray(values, dtype=np.float64)
            metrics[metric_name] = float(
                np.max(values_array)
                if metric_name in residual_metric_names
                else np.mean(values_array))
        # A stochastic rollout can contain no valid fixed-owner structure
        # (for example, every sampled movement leaves P0 with no TX/RX pair).
        # In that case the power controller is not called and no per-frame
        # diagnostic exists.  Keep the schema total and make the state explicit
        # instead of raising a KeyError or silently carrying a NaN.  Residuals
        # are zero because no infeasible candidate was selected; fallback=1 and
        # rounds=0 record that the controller did not execute.
        distributed_defaults = {
            'distributed_primal_dual_power_selected_violation': 0.0,
            'distributed_primal_dual_power_candidate_violation': 0.0,
            'distributed_primal_dual_power_candidate_stationarity': 0.0,
            'distributed_primal_dual_power_candidate_dual_residual': 0.0,
            'distributed_primal_dual_power_consensus_residual': 0.0,
            'distributed_primal_dual_power_target_price_mean': 0.0,
            'distributed_primal_dual_power_target_price_max': 0.0,
            'distributed_primal_dual_power_fallback': 1.0,
            'distributed_primal_dual_power_warm_started': 0.0,
            'distributed_primal_dual_power_actor_proposal_used': 0.0,
            'distributed_primal_dual_power_actor_proposal_mix': 0.0,
            'distributed_primal_dual_power_candidate_accepted': 0.0,
            'distributed_primal_dual_power_converged': 0.0,
            'distributed_primal_dual_power_rounds': 0.0,
        }
        for metric_name, default in distributed_defaults.items():
            metrics.setdefault(metric_name, float(default))

        if self._capacity_matching_enabled:
            metrics['capacity_matching_blend'] = (
                self._update_capacity_matching_blend())
        if self._target_movement_blend_schedule_enabled:
            metrics['target_allocation_movement_blend'] = (
                self._update_target_movement_blend())
        if self._training_seed_sampler is not None:
            metrics.update(self._training_seed_sampler.diagnostics(
                self.total_frames))

        # ── Diagnostic: is the critic fitting? are returns trending up/down? ──
        # value≈0 while return≈tens => critic not fitting (value-clip on raw scale).
        # return trending DOWN over training => reward/advantage direction problem.
        metrics['mean_return'] = float(returns.mean().item())
        metrics['mean_value'] = float(old_values.mean().item())
        metrics['mean_adv_abs'] = float(advantages.abs().mean().item())  # ~0.8 if normalized
        if self._risk_critic_only_training:
            current_checksum = tuple(
                float(parameter.detach().double().sum().cpu())
                for parameter in agent.actor.parameters())
            actor_unchanged = (
                current_checksum == self._risk_critic_actor_checksum)
            metrics['risk_critic_actor_unchanged'] = float(actor_unchanged)
            if not actor_unchanged:
                raise RuntimeError(
                    'actor changed during risk-critic-only training')

        return metrics

    def train_episode(self) -> Dict[str, float]:
        """Train for one episode (one rollout + one update).

        Returns:
            Dict of episode metrics
        """
        collect_started = time.perf_counter()
        episode_ended = self.collect_rollout()
        collect_time_s = float(time.perf_counter() - collect_started)
        update_started = time.perf_counter()
        metrics = self.update()
        update_time_s = float(time.perf_counter() - update_started)
        metrics['collect_time_s'] = collect_time_s
        metrics['update_time_s'] = update_time_s
        # MSE BC anchor: hold constant (no decay).
        if self._bc_beta_init > 0:
            self._bc_beta = self._bc_beta_init
        # LR decay + optional Actor freeze
        self._oracle_ep_count += 1
        freeze_after = getattr(self.cfg.marl, 'freeze_actor_after', 0)
        if self._risk_critic_only_training:
            return metrics
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
            # Audit 2026-08-17: the old `if current_actor_lr > 1e-4:` guard made
            # the final 3e-5 stage unreachable (after lr reached exactly 1e-4 the
            # guard was False forever).  Schedule by episode count instead; skip
            # only when the actor is frozen (lr already 0).
            current_actor_lr = self.agents[0].actor_optimizer.param_groups[0]['lr']
            if current_actor_lr > 0.0:
                if self._oracle_ep_count < 100: lr = 3e-4
                elif self._oracle_ep_count < 200: lr = 1e-4
                else: lr = 3e-5
                lr_map = getattr(self, '_actor_param_group_lr', None)
                for agent in self.agents:
                    # Audit 2026-08-25: preserve explicitly configured per-module
                    # lr groups (use_per_module_lr / freeze_attention ablation);
                    # the schedule only governs default single-LR groups.
                    for pg in agent.actor_optimizer.param_groups:
                        pg['lr'] = lr_map.get(id(pg), lr) if lr_map is not None else lr
                    for pg in agent.critic_optimizer.param_groups: pg['lr'] = lr * 5.0
        return metrics

    def _build_eval_env(self, seed: int) -> "UAVISACEnv":
        """Build an evaluation env for one seed (D1.10-A).

        Constructs a fresh ``UAVISACEnv`` (constructor seed = the episode seed)
        and applies the same dynamic-local-search / structure-student channel
        configuration the shared-instance protocol applies once.  With
        ``eval_independent_env=True`` the caller creates one env per episode,
        so every stochastic stream (including deflection_computer's Rician/LoS
        rng, which ``wrapper.reset`` does NOT replace) starts from the episode
        seed -- each seed is an independent draw rather than the k-th draw of a
        drifting shared stream.
        """
        eval_env = UAVISACEnv(config=self.cfg, seed=int(seed))
        if self._dynamic_local_search_config is not None:
            from uav_isac.coordination.dynamic_local_search import (
                DynamicLocalSearchCoordinator,
            )

            local_cfg = self._dynamic_local_search_config
            eval_env.core.configure_dynamic_local_search(
                DynamicLocalSearchCoordinator(
                    local_cfg["mode"],
                    ranker_checkpoint=local_cfg["ranker_checkpoint"],
                    cold_initializer=local_cfg["cold_initializer"],
                    factor_graph_checkpoint=(
                        local_cfg["factor_graph_checkpoint"]),
                    cold_rounds=local_cfg["cold_rounds"],
                    warm_rounds=local_cfg["warm_rounds"],
                    warm_top_m=local_cfg["warm_top_m"],
                    rebootstrap_mode=local_cfg["rebootstrap_mode"],
                    rebootstrap_interval_frames=(
                        local_cfg["rebootstrap_interval_frames"]),
                    rebootstrap_rounds=(
                        local_cfg["rebootstrap_rounds"]),
                    rebootstrap_require_deficit=bool(
                        local_cfg.get(
                            "rebootstrap_require_deficit", False)),
                )
            )
        if (
            self._structure_student is not None
            and self._structure_student_channel_enabled
        ):
            eval_env.core.configure_structure_student_channel(
                self._structure_student,
                bits_per_dim=(
                    self._structure_student_bits_per_dim),
                adaptive_min_bits_per_dim=(
                    self._structure_student_adaptive_min_bits_per_dim),
                min_comm_fraction=(
                    self._structure_student_min_comm_fraction),
                allow_cardinality_mismatch=(
                    self._structure_student_allow_scale_migration),
            )
        return eval_env

    def _evaluate(self, n_episodes: int = 5, steady_window: int = 20,
                  dp_deterministic: bool = True, role_deterministic: bool = True,
                  eval_seeds: Optional[List[int]] = None,
                  target_choice_audit_stride: int = 0,
                  sensing_choice_audit_stride: int = 0,
                  sensing_residual_blend: float = 0.25,
                  sensing_audit_horizon: int = 0,
                  sensing_oracle_control: bool = False,
                  joint_sensing_pair_audit_stride: int = 0,
                  physical_oracle_stride: int = 0,
                  evidence_trace_output: Optional[str] = None,
                  structure_teacher_trace_output: Optional[str] = None,
                  structure_trace_run_binding: Optional[Dict] = None,
                  n5_counterfactual_output: Optional[str] = None,
                  n5_counterfactual_max_events: int = 0,
                  n5_counterfactual_target_mode: str = "both",
                  n5_counterfactual_seed_filter: Optional[List[int]] = None,
                  n5_counterfactual_max_candidates: int = 0,
                  n5_counterfactual_require_proxy_positive: bool = False,
                  ) -> Dict[str, float]:
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
        independent = bool(getattr(
            self.cfg.marl, 'eval_independent_env', False))
        if independent:
            # D1.10-A: one fresh env per episode seed -> every stochastic
            # stream (incl. deflection_computer's persistent rng) is seeded
            # by the episode, so results are sequence-independent.
            eval_env = None
        else:
            eval_env = self._build_eval_env(seed=12345)

        ep_full_means = []        # full-episode mean P_D per episode
        ep_steady_means = []      # steady-window mean P_D per episode
        ep_worst = []             # per-episode steady-window worst target
        ep_weak3 = []             # per-episode steady-window bottom-3 avg
        ep_trimmed_worst = []      # per-episode bottom-2 target average
        ep_tstd = []              # per-episode steady-window target std
        ep_per_target = []        # (n_eps, Q) steady-window per-target means
        ep_commitment_coverage = []
        ep_hard_commitment_coverage = []
        ep_p0_target_coverage = []
        vp_frames = notx_frames = samerole_frames = total_frames = 0
        duplex_endpoint_frames = 0
        duplex_endpoint_nodes = []
        eval_comm_bits = []
        eval_comm_energy = []
        eval_comm_latency = []
        eval_comm_p95_latency = []
        eval_comm_mean_bler = []
        eval_comm_max_bler = []
        eval_comm_reliability_violation = []
        eval_comm_burst_failure = []
        eval_comm_burst_bad_link = []
        eval_comm_delivery = []
        eval_comm_active = []
        eval_comm_violation = []
        eval_comm_snr_violation = []
        eval_comm_delivery_failure = []
        eval_structure_payload_bits = []
        eval_structure_atomic_delivery = []
        eval_structure_selected_bits = []
        eval_structure_public_valid = []
        eval_structure_public_age = []
        eval_structure_insufficient_cache = []
        eval_p0_resolved = []
        eval_p0_solve_time = []
        eval_p0_topology_invalid = []
        eval_p0_topology_invalid_edges = []
        eval_p0_topology_repair = []
        eval_p0_topology_replacements = []
        eval_p0_topology_repair_worst_pd = []
        eval_p0_topology_full_fallback = []
        eval_local_search_candidates = []
        eval_local_search_verifications = []
        eval_local_search_accepted = []
        eval_local_search_cold = []
        eval_local_search_rebootstrap_attempted = []
        eval_local_search_rebootstrap_accepted = []
        eval_local_search_rebootstrap_candidates = []
        eval_local_search_rebootstrap_deficit_present = []
        eval_local_search_rebootstrap_blocked_no_deficit = []
        n5_counterfactual_events = []
        eval_evidence_comm_bits = []
        eval_evidence_comm_energy = []
        eval_evidence_comm_latency = []
        eval_evidence_comm_delivery = []
        eval_evidence_comm_active = []
        eval_evidence_comm_violation = []
        eval_evidence_comm_failure = []
        eval_evidence_comm_snr_failure = []
        eval_evidence_comm_reliability_failure = []
        eval_evidence_comm_burst_failure = []
        eval_evidence_comm_burst_bad = []
        eval_evidence_utilization = []
        eval_evidence_transmitted_entries = []
        eval_evidence_useful_entries = []
        eval_evidence_pfa = []
        eval_belief_feedback_entries = []
        eval_belief_feedback_receivers = []
        eval_belief_feedback_before = []
        eval_belief_feedback_after = []
        eval_belief_feedback_contraction = []
        eval_belief_feedback_truth_rmse_before = []
        eval_belief_feedback_truth_rmse_after = []
        eval_belief_position_rmse = []
        eval_belief_position_p95 = []
        eval_belief_position_rmse_per_target = []
        eval_belief_target_aoi_mean = []
        eval_belief_target_aoi_p95 = []
        eval_belief_target_aoi_per_target = []
        eval_belief_disagreement_end = []
        eval_isac_comm_power = []
        eval_isac_sensing_power = []
        eval_isac_balance_error = []
        eval_isac_budget_violation = []
        eval_isac_unused_power = []
        eval_isac_target_power = []
        eval_replicated_common_view = []
        eval_replicated_full_view = []
        eval_replicated_local_coverage = []
        eval_replicated_solve_time = []
        eval_replicated_per_node_solve_time = []
        eval_replicated_parallel_used = []
        eval_replicated_parallel_batch_wall = []
        eval_replicated_history_reserve_used = []
        eval_replicated_worker_warmup = []
        eval_hyperedge_visible_peers = []
        eval_hyperedge_mutual_edges = []
        eval_hyperedge_active_edges = []
        eval_hyperedge_target_coverage = []
        eval_hyperedge_protocol_used = []
        eval_hyperedge_safety_fallback = []
        eval_hyperedge_assignment_reused = []
        eval_p0_target_selected_mask = []
        eval_cacsr_gate_rate = []
        eval_cacsr_delta_abs = []
        eval_risk_residual_gate = []
        eval_risk_residual_delta_abs = []
        eval_risk_direction_scale_abs = []
        eval_adaptive_topk_values = []
        eval_adaptive_topk_switches = []
        eval_token_sensing_jaccard = []
        eval_token_sensing_exact = []
        eval_token_sensing_power_mass = []
        # Actor allocation diagnostics.  These are computed from the actor's
        # original movement output before any evaluation-only intervention.
        actor_move_unique_target = []
        actor_move_collision = []
        actor_move_pair_cosine = []
        actor_move_idle = []
        actor_move_hungarian_match = []
        actor_target_choice_counts = np.zeros((K, Q), dtype=np.int64)
        executed_move_unique_target = []
        executed_move_collision = []
        executed_target_choice_counts = np.zeros((K, Q), dtype=np.int64)
        ep_nearest_target_distance = []
        # Physical post-step separation audit.  Unlike the actor/executed
        # target-choice diagnostics above, these values are read back from the
        # environment after every movement override has actually been applied.
        eval_inter_uav_min_distance = []
        eval_inter_uav_separation_violation = []
        ep_min_inter_uav_distance = []
        eval_inter_uav_continuous_min_distance = []
        eval_inter_uav_continuous_separation_violation = []
        ep_min_inter_uav_continuous_distance = []
        eval_movement_safety_intervened = []
        eval_movement_safety_intervention_call_rate = []
        eval_movement_safety_fail_closed = []
        eval_movement_safety_fail_closed_calls = []
        eval_movement_safety_fail_closed_outside_invariant_calls = []
        eval_movement_safety_outside_invariant_calls = []
        eval_movement_safety_recovery_pair_count = []
        eval_movement_safety_solve_time_s = []
        eval_movement_safety_solve_time_s_per_node = []
        eval_movement_safety_projection_calls = []
        eval_movement_safety_reduced_linear_qp_calls = []
        eval_movement_safety_reduced_linear_qp_rate = []
        eval_movement_public_view_stale_fail_closed_rate = []
        eval_movement_public_view_mean_max_age = []
        eval_movement_public_view_max_age = []
        eval_movement_envelope_gap_selected_rate = []
        eval_movement_envelope_single_projection_rate = []
        eval_movement_envelope_selection_disagreement_rate = []
        eval_movement_primal_dual_candidate_enabled = []
        eval_movement_primal_dual_certificate_available_rate = []
        eval_movement_primal_dual_selected_rate = []
        eval_movement_primal_dual_strong_dominance_rate = []
        eval_movement_primal_dual_lower_improvement = []
        eval_movement_primal_dual_certificate_width = []
        eval_movement_primal_dual_incumbent_lower = []
        eval_movement_primal_dual_candidate_lower = []
        eval_movement_primal_dual_selection_disagreement_rate = []
        # Unified MAC slot accounting (audit priority #5): serialized
        # coordination + evidence/belief latency against the frame deadline.
        eval_protocol_total_latency = []
        eval_protocol_total_p95_latency = []
        eval_protocol_serialized_bits = []
        eval_protocol_total_deadline_violation = []
        eval_movement_assignment_target_coverage = []
        eval_movement_assignment_duplicate_rate = []
        eval_movement_assignment_entry_agreement = []
        eval_movement_anchor_target_coverage = []
        eval_movement_gain_schedule_selected_self_rate = []
        eval_movement_gain_schedule_strategy_hold_rate = []
        eval_movement_gain_schedule_no_improvement_rate = []
        eval_movement_gain_schedule_far_range_rate = []
        rate_levels = list(getattr(
            self.cfg.marl, 'comm_rate_bits_per_dim', [0, 4, 8, 16]))
        eval_comm_rate_counts = np.zeros(len(rate_levels), dtype=np.int64)
        eval_risk_quantiles = []
        eval_risk_logits = []
        eval_risk_targets = []
        eval_episode_initial_cvar = []
        eval_episode_realized_worst = []
        # Movement-boundary audit: distinguish useful bistatic overlap from
        # duplicate movers that have no realized marginal sensing value.
        eval_responsibility_episodes = []
        eval_responsibility_pd_histories = []
        eval_movement_reward_histories = []
        eval_target_choice_intervention_episodes = []
        eval_sensing_choice_intervention_episodes = []
        eval_joint_sensing_pair_intervention_episodes = []
        eval_physical_oracle_episodes = []
        target_choice_audit_counter = 0
        sensing_choice_audit_counter = 0
        eval_evidence_global_histories = []
        eval_evidence_local_histories = []
        eval_evidence_top1_histories = []
        eval_evidence_top2_histories = []
        eval_evidence_reconstruction_errors = []
        evidence_trace_receiver_d = []
        evidence_trace_owner = []
        evidence_trace_episode = []
        evidence_trace_frame = []
        evidence_trace_positions = []
        evidence_trace_comm_power = []
        structure_trace_episode = []
        structure_trace_seed = []
        structure_trace_frame = []
        structure_trace_p0_resolved = []
        structure_trace_obs = []
        structure_trace_next_obs = []
        structure_trace_delta_p = []
        structure_trace_message = []
        structure_trace_rate = []
        structure_trace_token_mask = []
        structure_trace_comm_fraction = []
        structure_trace_sensing_weights = []
        structure_trace_uav_positions = []
        structure_trace_uav_velocities = []
        structure_trace_target_states = []
        structure_trace_target_positions = []
        structure_trace_target_velocities = []
        structure_trace_sensing_power = []
        structure_trace_comm_power = []
        structure_trace_sensing_budget = []
        structure_trace_per_watt_coefficient = []
        structure_trace_task_mean_pd = []
        structure_trace_task_weak3_pd = []
        structure_trace_task_worst_pd = []
        structure_trace_local_pd = []
        structure_trace_coord_pd_ema = []
        structure_trace_coord_pd_ema_valid = []
        structure_trace_pair = []
        structure_trace_tx_target = []
        structure_trace_rx_target = []
        structure_trace_endpoint_target = []
        structure_trace_receiver_owner = []
        structure_trace_role = []
        structure_trace_candidate = []
        structure_trace_d_eff = []
        structure_trace_d_raw = []
        structure_trace_alpha = []
        structure_trace_g_dd = []
        structure_trace_chi_rep = []
        structure_trace_physical_pd = []
        structure_trace_rebootstrap_attempted = []
        structure_trace_rebootstrap_accepted = []
        W = steady_window

        for ep_seed in eval_seeds:
            if independent:
                # D1.10-A: fresh env per episode seed.
                eval_env = self._build_eval_env(seed=int(ep_seed))
            obs, reset_info = eval_env.reset(seed=int(ep_seed))
            if independent or not eval_replicated_worker_warmup:
                eval_replicated_worker_warmup.append(float(reset_info.get(
                    'distributed_replicated_power_worker_warmup_time_s',
                    0.0,
                )))
            pd_hist = []   # list of mean P_D_q per frame
            pd_per_target = []  # list of (Q,) per frame
            nearest_target_distance = []  # list of (Q,) per frame
            episode_inter_uav_min_distance = []
            episode_inter_uav_continuous_min_distance = []
            # P0 FIX: streaming GRU + TICA window during eval
            eval_h_prev = None
            eval_movement_phase = 0
            eval_held_actions = None
            eval_qos_held_actions = None
            eval_qos_teacher_states = []
            eval_cacsr_ema = None
            eval_cacsr_history = []
            eval_previous_topk = np.full(K, -1, dtype=np.int64)
            eval_sensing_oracle_hold = None
            eval_sensing_oracle_until = -1
            episode_risk_quantiles = []
            episode_risk_logits = []
            episode_responsibility_records = []
            episode_movement_rewards = []
            episode_target_choice_interventions = []
            episode_sensing_choice_interventions = []
            episode_joint_sensing_pair_interventions = []
            episode_physical_oracles = []
            episode_evidence_global = []
            episode_evidence_local = []
            episode_evidence_top1 = []
            episode_evidence_top2 = []
            episode_commitment_coverage = []
            episode_hard_commitment_coverage = []
            episode_p0_target_coverage = []
            eval_ring = None
            eval_wmask = None
            if self._use_window:
                eval_ring = np.zeros((K, self._window_len, obs_dim), dtype=np.float64)
                eval_wmask = np.zeros((K, self._window_len), dtype=bool)
            episode_done = False
            while not episode_done:
                ob = np.stack([obs[str(k)] for k in range(K)])
                # Update window ring buffer
                if self._use_window:
                    eval_ring[:, :-1] = eval_ring[:, 1:]
                    eval_ring[:, -1] = ob
                    eval_wmask[:, :-1] = eval_wmask[:, 1:]
                    eval_wmask[:, -1] = True
                with torch.inference_mode():
                    eval_phase_value = (
                        float(eval_movement_phase)
                        / max(self.movement_decision_interval - 1, 1))
                    eval_round_phase = torch.full(
                        (K,), eval_phase_value, dtype=torch.float32,
                        device=self.device)
                    eval_agent_identity = torch.arange(
                        K, dtype=torch.long, device=self.device)
                    if self._use_window:
                        ob_in = torch.as_tensor(eval_ring, dtype=torch.float32, device=self.device)
                        wm_in = torch.as_tensor(eval_wmask, dtype=torch.bool, device=self.device)
                        dp_mean, dp_log_std, role_logits, comm_mean, _, h_new = actor(
                            ob_in, eval_h_prev, window_mask=wm_in,
                            comm_round_phase=eval_round_phase,
                            agent_identity=eval_agent_identity)
                    else:
                        ob_t = torch.as_tensor(ob, dtype=torch.float32, device=self.device)
                        dp_mean, dp_log_std, role_logits, comm_mean, _, h_new = actor(
                            ob_t, eval_h_prev,
                            comm_round_phase=eval_round_phase,
                            agent_identity=eval_agent_identity)
                    risk_residual_gate = getattr(
                        actor, 'last_residual_gate', None)
                    if risk_residual_gate is not None:
                        eval_risk_residual_gate.append(float(
                            risk_residual_gate.detach().float().mean().cpu()))
                    risk_residual_delta = getattr(
                        actor, 'last_normalized_movement_delta', None)
                    if risk_residual_delta is not None:
                        eval_risk_residual_delta_abs.append(float(
                            risk_residual_delta.detach().float().abs().mean().cpu()))
                    risk_direction_scale = getattr(
                        actor, 'last_directional_movement_scale', None)
                    if risk_direction_scale is not None:
                        eval_risk_direction_scale_abs.append(float(
                            risk_direction_scale.detach().float().abs().mean().cpu()))
                    eval_token_mask = getattr(
                        actor, 'last_outgoing_token_mask', None)
                    if eval_token_mask is None:
                        eval_token_mask = torch.ones(
                            (K, Q), dtype=comm_mean.dtype,
                            device=comm_mean.device)
                    if self._comm_cost_aware:
                        eval_comm, eval_rate, _, _ = self.agents[0].sample_communication(
                            comm_mean, deterministic=True)
                        if self._comm_eval_force_rate_index >= 0:
                            eval_rate = torch.full_like(
                                eval_rate,
                                self._comm_eval_force_rate_index)
                        if self._adaptive_topk_from_rate_enabled:
                            eval_claim_scores = getattr(
                                actor, 'last_sparse_claim_scores', None)
                            if eval_claim_scores is None:
                                raise RuntimeError(
                                    'adaptive top-k requires actor claim scores')
                            eval_token_mask = build_rate_conditioned_topk_mask(
                                eval_claim_scores,
                                eval_rate,
                                self._adaptive_topk_rate_mapping,
                                maximum_mask=eval_token_mask,
                            )
                        if self._joint_isac_power_enabled:
                            (_, _, eval_comm_fraction,
                             eval_sensing_weights, _, _) = (
                                self.agents[0].sample_isac_resources(
                                    comm_mean, eval_rate, deterministic=True,
                                    comm_fraction_min=float(getattr(
                                        self.cfg.marl,
                                        'comm_power_fraction_min', 0.0)),
                                    comm_fraction_max=float(getattr(
                                        self.cfg.marl,
                                        'comm_power_fraction_max', 1.0)),
                                )
                            )
                            if self._comm_semantic_cacsr_eval_enabled:
                                semantic_evidence = getattr(
                                    actor, 'last_comm_semantic_evidence', None)
                                semantic_pd = getattr(
                                    actor, 'last_comm_semantic_pd', None)
                                slices = getattr(actor, '_obs_slices', None)
                                if (semantic_evidence is not None
                                        and semantic_pd is not None
                                        and slices is not None):
                                    comm_mask_np = slices.extract_comm_mask(ob)
                                    peer_mask = torch.as_tensor(
                                        comm_mask_np.reshape(K, K - 1, Q),
                                        dtype=comm_mean.dtype,
                                        device=self.device)
                                    peer_quality = (
                                        semantic_evidence * semantic_pd).reshape(
                                            K, K - 1, Q).amax(dim=1)
                                    local_pd = torch.as_tensor(
                                        slices.extract_pd_hist(ob),
                                        dtype=comm_mean.dtype,
                                        device=self.device)
                                    semantic_quality = torch.maximum(
                                        peer_quality, local_pd).clamp(0.0, 1.0)
                                    endpoint_count = (
                                        peer_mask.sum(dim=1)
                                        + eval_token_mask.to(peer_mask.dtype))
                                    geometry = slices.extract_geometry(ob)
                                    if geometry is None:
                                        local_capability = torch.ones_like(
                                            semantic_quality)
                                    else:
                                        local_capability = torch.as_tensor(
                                            geometry[..., 6],
                                            dtype=comm_mean.dtype,
                                            device=self.device).clamp(0.0, 1.0)
                                    quality_np = (
                                        semantic_quality.detach().cpu().numpy())
                                    if eval_cacsr_ema is None:
                                        eval_cacsr_ema = quality_np.copy()
                                    else:
                                        alpha = self._comm_semantic_cacsr_ema_alpha
                                        eval_cacsr_ema = (
                                            alpha * quality_np
                                            + (1.0 - alpha) * eval_cacsr_ema)
                                    eval_cacsr_history.append(
                                        eval_cacsr_ema.copy())
                                    history_len = (
                                        self._comm_semantic_cacsr_stagnation_frames)
                                    if len(eval_cacsr_history) > history_len:
                                        improvement = (
                                            eval_cacsr_history[-1]
                                            - eval_cacsr_history[-1 - history_len])
                                        stagnant = improvement <= (
                                            self._comm_semantic_cacsr_improvement_epsilon)
                                    else:
                                        stagnant = np.zeros((K, Q), dtype=bool)
                                    low_quality = (
                                        1.0 - quality_np
                                        >= self._comm_semantic_cacsr_quality_threshold)
                                    endpoint_underload = (
                                        endpoint_count.detach().cpu().numpy()
                                        < self._sparse_claim_desired_endpoints)
                                    has_peer_message = (
                                        peer_mask.sum(dim=(1, 2)).detach()
                                        .cpu().numpy() > 0.0)[:, None]
                                    gate_np = (
                                        endpoint_underload
                                        | (low_quality & stagnant))
                                    gate_np &= has_peer_message
                                    gate = torch.as_tensor(
                                        gate_np,
                                        dtype=torch.bool,
                                        device=self.device)
                                    eval_sensing_weights, cacsr_delta = (
                                        apply_cacsr_sensing_residual(
                                            eval_sensing_weights,
                                            gate,
                                            semantic_quality,
                                            endpoint_count,
                                            local_capability,
                                            self._comm_semantic_cacsr_eval_gain,
                                            self._sparse_claim_desired_endpoints,
                                        ))
                                    eval_cacsr_gate_rate.append(float(
                                        gate.to(torch.float32).mean().item()))
                                    eval_cacsr_delta_abs.append(float(
                                        cacsr_delta.abs().mean().item()))
                        if self._comm_eval_force_silence:
                            eval_rate = torch.zeros_like(eval_rate)
                        if self._comm_eval_message_ablation == 'zero':
                            eval_comm = torch.zeros_like(eval_comm)
                        elif self._comm_eval_message_ablation == 'permute':
                            eval_comm = torch.roll(eval_comm, shifts=1, dims=0)
                            eval_token_mask = torch.roll(
                                eval_token_mask, shifts=1, dims=0)
                        rate_np = eval_rate.detach().cpu().numpy().astype(int)
                        if self._adaptive_topk_from_rate_enabled:
                            mapping = np.asarray(
                                self._adaptive_topk_rate_mapping,
                                dtype=np.int64)
                            topk_np = mapping[np.clip(
                                rate_np, 0, mapping.size - 1)]
                            valid_previous = eval_previous_topk >= 0
                            if np.any(valid_previous):
                                eval_adaptive_topk_switches.extend(
                                    (topk_np[valid_previous]
                                     != eval_previous_topk[valid_previous])
                                    .astype(np.float64).tolist())
                            eval_adaptive_topk_values.extend(
                                topk_np.astype(np.float64).tolist())
                            eval_previous_topk = topk_np.copy()
                        eval_comm_rate_counts += np.bincount(
                            rate_np,
                            minlength=len(rate_levels),
                        )[:len(rate_levels)]
                    eval_h_prev = h_new
                dpm = dp_mean.detach().cpu().numpy()
                dps = dp_log_std.detach().cpu().numpy()
                rl = role_logits.detach().cpu().numpy()
                actions = {}
                movement_decision = bool(
                    eval_movement_phase == 0 or eval_held_actions is None)
                for k in range(K):
                    a, _ = aspace.decode(dpm[k], dps, rl[k],
                                         dp_deterministic=dp_deterministic,
                                         role_deterministic=role_deterministic)
                    actions[str(k)] = {'delta_p': a.delta_p, 'role': a.role}
                if movement_decision:
                    eval_held_actions = {
                        key: {
                            'delta_p': np.asarray(value['delta_p']).copy(),
                            'role': int(value['role']),
                        }
                        for key, value in actions.items()
                    }
                else:
                    actions = {
                        key: {
                            'delta_p': np.asarray(value['delta_p']).copy(),
                            'role': int(value['role']),
                        }
                        for key, value in eval_held_actions.items()
                    }

                # Infer which target each movement vector is approaching.  A
                # direction-based diagnostic is preferable to nearest-target
                # identity because it captures the actor's immediate intent.
                uav_xy = np.stack([
                    eval_env.core.uavs[k].pos[:2] for k in range(K)])
                target_xy = np.stack([
                    target.get_position_3d()[:2]
                    for target in eval_env.core.targets])
                actor_moves = np.stack([
                    actions[str(k)]['delta_p'] for k in range(K)])
                move_norm = np.linalg.norm(actor_moves, axis=1)
                active_move = move_norm > 1e-8
                to_target = target_xy[None, :, :] - uav_xy[:, None, :]
                target_dist = np.linalg.norm(to_target, axis=-1)
                cosine = np.einsum('kd,kqd->kq', actor_moves, to_target)
                cosine /= np.maximum(
                    move_norm[:, None] * target_dist, 1e-12)
                actor_choice = np.argmax(cosine, axis=1)
                for k in range(K):
                    if active_move[k]:
                        actor_target_choice_counts[k, actor_choice[k]] += 1
                active_choices = actor_choice[active_move]
                unique_count = len(np.unique(active_choices))
                actor_move_unique_target.append(
                    unique_count / max(min(K, Q), 1))
                choice_load = np.bincount(active_choices, minlength=Q)
                actor_move_collision.append(float(np.any(choice_load > 1)))
                actor_move_idle.append(float(np.mean(~active_move)))
                pair_cos = []
                for i in range(K):
                    for j in range(i + 1, K):
                        if active_move[i] and active_move[j]:
                            pair_cos.append(float(np.dot(
                                actor_moves[i], actor_moves[j]) /
                                (move_norm[i] * move_norm[j])))
                if pair_cos:
                    actor_move_pair_cosine.append(float(np.mean(pair_cos)))

                rows, cols = linear_sum_assignment_numpy(target_dist)
                nearest_assignment = np.full(K, -1, dtype=np.int64)
                nearest_assignment[rows] = cols
                comparable = active_move & (nearest_assignment >= 0)
                if np.any(comparable):
                    actor_move_hungarian_match.append(float(np.mean(
                        actor_choice[comparable]
                        == nearest_assignment[comparable])))
                if self._eval_centralized_assignment_movement:
                    if K <= Q:
                        slots = np.arange(Q, dtype=np.int64)
                    else:
                        slots = np.arange(K, dtype=np.int64) % Q
                    cost = np.linalg.norm(
                        uav_xy[:, None, :]
                        - target_xy[slots][None, :, :], axis=-1)
                    rows, cols = linear_sum_assignment_numpy(cost)
                    assignment = np.zeros(K, dtype=np.int64)
                    assignment[rows] = slots[cols]
                    max_dp = self.cfg.uav.v_max * self.cfg.scenario.dt
                    for k in range(K):
                        delta = target_xy[assignment[k]] - uav_xy[k]
                        norm = float(np.linalg.norm(delta))
                        move = (delta / norm * max_dp
                                if norm > 1e-9 else np.zeros(2))
                        actions[str(k)]['delta_p'] = move
                if self._eval_qos_bistatic_assignment_movement:
                    if movement_decision or eval_qos_held_actions is None:
                        eval_qos_teacher_states.append(
                            eval_env.core.get_global_state().copy())
                        teacher_states = torch.as_tensor(
                            np.stack(eval_qos_teacher_states),
                            dtype=torch.float32,
                            device=self.device,
                        )
                        _, teacher_movements = (
                            compute_qos_bistatic_assignment_teacher(
                                teacher_states,
                                K,
                                Q,
                                self.cfg.scenario.region_size,
                                self.cfg.uav.v_max * self.cfg.scenario.dt,
                                commitment_frames=(
                                    self._target_allocation_teacher_commitment_frames),
                                qos_floor=(
                                    self._target_allocation_teacher_qos_floor),
                                qos_weight=(
                                    self._target_allocation_teacher_qos_weight),
                                height_m=(
                                    self._target_allocation_teacher_height_m),
                                switching_penalty_m=(
                                    self._target_allocation_teacher_switching_penalty_m),
                            )
                        )
                        current_moves = teacher_movements.reshape(
                            -1, K, 2)[-1].detach().cpu().numpy()
                        eval_qos_held_actions = {
                            str(k): {
                                'delta_p': current_moves[k].copy(),
                                'role': int(actions[str(k)]['role']),
                            }
                            for k in range(K)
                        }
                    actions = {
                        key: {
                            'delta_p': np.asarray(value['delta_p']).copy(),
                            'role': int(value['role']),
                        }
                        for key, value in eval_qos_held_actions.items()
                    }

                # Executed-movement diagnostics are intentionally separate
                # from the actor-intent metrics above.  Under an evaluation
                # intervention they reveal the controller actually sent to
                # the physical environment.
                executed_moves = np.stack([
                    actions[str(k)]['delta_p'] for k in range(K)])
                executed_norm = np.linalg.norm(executed_moves, axis=1)
                executed_active = executed_norm > 1e-8
                executed_cosine = np.einsum(
                    'kd,kqd->kq', executed_moves, to_target)
                executed_cosine /= np.maximum(
                    executed_norm[:, None] * target_dist, 1e-12)
                executed_choice = np.argmax(executed_cosine, axis=1)
                for k in range(K):
                    if executed_active[k]:
                        executed_target_choice_counts[
                            k, executed_choice[k]] += 1
                executed_choices = executed_choice[executed_active]
                executed_move_unique_target.append(
                    len(np.unique(executed_choices))
                    / max(min(K, Q), 1))
                executed_load = np.bincount(
                    executed_choices, minlength=Q)
                executed_move_collision.append(float(
                    np.any(executed_load > 1)))
                if self._comm_cost_aware:
                    eval_messages = {
                        k: eval_comm[k].detach().cpu().numpy()
                        for k in range(K)}
                    eval_rates = {
                        k: int(eval_rate[k].item()) for k in range(K)}
                    eval_token_masks = {
                        k: eval_token_mask[k].detach().cpu().numpy()
                        for k in range(K)}
                    if self._joint_isac_power_enabled:
                        sensing_np = (
                            eval_sensing_weights.detach().cpu().numpy())
                        commitment_topk = min(max(int(getattr(
                            self.cfg.marl,
                            'distributed_target_commitment_topk',
                            1)), 1), Q)
                        sensing_order = np.argsort(
                            -sensing_np, axis=1, kind='stable')
                        sensing_mask = np.zeros((K, Q), dtype=bool)
                        sensing_mask[
                            np.arange(K)[:, None],
                            sensing_order[:, :commitment_topk],
                        ] = True
                        transmitted_mask = np.stack([
                            eval_token_masks[k] > 0.5 for k in range(K)])
                        intersection = np.logical_and(
                            transmitted_mask, sensing_mask).sum(axis=1)
                        union = np.logical_or(
                            transmitted_mask, sensing_mask).sum(axis=1)
                        eval_token_sensing_jaccard.extend(
                            (intersection / np.maximum(union, 1)).tolist())
                        eval_token_sensing_exact.extend(np.all(
                            transmitted_mask == sensing_mask,
                            axis=1).astype(np.float64).tolist())
                        eval_token_sensing_power_mass.extend(np.sum(
                            sensing_np * transmitted_mask, axis=1).tolist())
                        submit_args = (
                            eval_messages, eval_rates,
                            {k: float(eval_comm_fraction[k].item())
                             for k in range(K)},
                            {k: eval_sensing_weights[k].detach().cpu().numpy()
                             for k in range(K)},
                        )
                        if self._sparse_claim_enabled:
                            eval_env.core.submit_learned_communications(
                                *submit_args, token_masks=eval_token_masks)
                        else:
                            eval_env.core.submit_learned_communications(
                                *submit_args)
                    else:
                        if self._sparse_claim_enabled:
                            eval_env.core.submit_learned_communications(
                                eval_messages, eval_rates,
                                token_masks=eval_token_masks)
                        else:
                            eval_env.core.submit_learned_communications(
                                eval_messages, eval_rates)
                elif not self._comm_off:
                    legacy_comm = comm_mean.detach().cpu().numpy()
                    eval_env.core._comm_msgs = {
                        k: legacy_comm[k].copy() for k in range(K)
                    }
                if self._structure_student is not None:
                    if not (
                        self._comm_cost_aware
                        and self._joint_isac_power_enabled
                    ):
                        raise RuntimeError(
                            "structure student requires cost-aware target "
                            "tokens and joint ISAC resource actions")
                    from uav_isac.agents.frozen_structure_student import (
                        build_structure_student_features,
                    )

                    slices = getattr(actor, "_obs_slices", None)
                    if slices is None:
                        raise RuntimeError(
                            "structure student requires actor observation "
                            "slice metadata")
                    student_kwargs = dict(
                        outgoing_message=(
                            eval_comm.detach().cpu().numpy()),
                        outgoing_token_mask=(
                            eval_token_mask.detach().cpu().numpy()),
                        outgoing_rate=(
                            eval_rate.detach().cpu().numpy()),
                        comm_fraction=(
                            eval_comm_fraction.detach().cpu().numpy()),
                        sensing_weights=(
                            eval_sensing_weights.detach().cpu().numpy()),
                        rate_scale=self._structure_student.rate_scale,
                    )
                    student_features = build_structure_student_features(
                        ob,
                        slices,
                        **student_kwargs,
                    )
                    if self._structure_student_channel_enabled:
                        send_structure = (
                            self._structure_student_send_mode
                            == "every_frame"
                            or eval_env.core
                            .structure_student_protocol_due_next_step()
                        )
                        if send_structure:
                            endpoint_protocol = (
                                self._structure_student
                                .encode_endpoint_protocol(
                                    student_features)[0])
                            eval_env.core.submit_structure_student_endpoint_protocol(
                                endpoint_protocol)
                    else:
                        if self._dynamic_local_search_config is not None:
                            from uav_isac.evaluation.local_candidate_audit import (
                                build_local_candidate_mask,
                                delivered_target_tokens,
                                select_local_neighbors,
                                select_value_guided_neighbors,
                            )

                            local_cfg = self._dynamic_local_search_config
                            frame_obs = ob[None, ...]
                            token_visible, token_age = delivered_target_tokens(
                                frame_obs, slices)
                            local_pd = slices.extract_pd_hist(frame_obs)
                            local_neighbors = select_local_neighbors(
                                token_visible,
                                token_age,
                                neighbor_topk=local_cfg["neighbor_topk"],
                            )
                            provisional_features = (
                                build_structure_student_features(
                                    ob,
                                    slices,
                                    neighbor_subset_mask=(
                                        local_neighbors[0]),
                                    **student_kwargs,
                                )
                            )
                            provisional_values = (
                                self._structure_student.predict(
                                    provisional_features)
                            )
                            local_neighbors = select_value_guided_neighbors(
                                provisional_values,
                                local_pd,
                                token_visible,
                                token_age,
                                neighbor_topk=local_cfg["neighbor_topk"],
                                qos_floor=float(getattr(
                                    self.cfg.marl,
                                    "comm_qos_worst_min",
                                    0.60)),
                                coverage_fraction=(
                                    local_cfg["coverage_fraction"]),
                            )
                            student_features = (
                                build_structure_student_features(
                                    ob,
                                    slices,
                                    neighbor_subset_mask=(
                                        local_neighbors[0]),
                                    **student_kwargs,
                                )
                            )
                            student_edge_values = (
                                self._structure_student.predict(
                                    student_features)[0]
                            )
                            candidate_mask, _ = build_local_candidate_mask(
                                student_edge_values[None, ...],
                                local_pd,
                                local_neighbors,
                                token_visible,
                                target_topk=local_cfg["target_topk"],
                                qos_floor=float(getattr(
                                    self.cfg.marl,
                                    "comm_qos_worst_min",
                                    0.60)),
                                coverage_fraction=(
                                    local_cfg["coverage_fraction"]),
                                require_reciprocal_link=True,
                            )
                            eval_env.core.submit_structure_student_candidate_mask(
                                candidate_mask[0])
                        else:
                            student_edge_values = (
                                self._structure_student.predict(
                                    student_features)[0]
                            )
                        eval_env.core.submit_structure_student_edge_values(
                            student_edge_values)
                if (sensing_oracle_control
                        and eval_sensing_oracle_hold is not None
                        and int(eval_env.core.t) < eval_sensing_oracle_until):
                    held_agent, held_target = eval_sensing_oracle_hold
                    apply_sensing_choice_intervention(
                        eval_env,
                        held_agent,
                        held_target,
                        sensing_residual_blend,
                    )
                risk_quantiles_step = None
                risk_logits_step = None
                if self._set_risk_critic_enabled:
                    with torch.inference_mode():
                        if self._comm_cost_aware:
                            active = (eval_rate > 0).to(
                                eval_comm.dtype).unsqueeze(-1)
                            token_width = max(
                                eval_comm.shape[-1] // max(Q, 1), 1)
                            dimension_mask = (
                                eval_token_mask.repeat_interleave(
                                    token_width, dim=-1))
                            risk_comm = eval_comm * active * dimension_mask
                        elif self._comm_off:
                            risk_comm = torch.zeros_like(comm_mean)
                        else:
                            risk_comm = comm_mean
                        risk_comm_summary = self._critic_comm_summary(
                            risk_comm).mean(dim=0, keepdim=True)
                        risk_base = torch.as_tensor(
                            eval_env.core.get_global_state(),
                            dtype=torch.float32,
                            device=self.device).unsqueeze(0)
                        # The set-risk branch intentionally ignores this
                        # identity block; zeros make that boundary explicit.
                        risk_identity = torch.zeros(
                            (1, K), dtype=risk_base.dtype,
                            device=self.device)
                        risk_input = torch.cat([
                            risk_base, risk_identity, risk_comm_summary], dim=-1)
                        risk_quantiles_t, risk_logits_t = (
                            self.agents[0].critic.forward_risk(risk_input))
                        risk_quantiles_step = (
                            risk_quantiles_t[0].detach().cpu().numpy())
                        risk_logits_step = (
                            risk_logits_t[0].detach().cpu().numpy())
                # Match rollout/deployment timing. Communication/resources are
                # refreshed on every actor decision; movement may be a slower
                # local commitment controlled by movement_decision_interval.
                audit_stride = max(0, int(target_choice_audit_stride))
                pending_target_choice_intervention = None
                if (audit_stride > 0
                        and movement_decision
                        and int(eval_env.core.t) % audit_stride == 0):
                    audit_agent = (
                        target_choice_audit_counter % max(K, 1))
                    pending_target_choice_intervention = (
                        evaluate_target_choice_intervention(
                            eval_env,
                            actions,
                            agent_index=audit_agent,
                            actual_choice=int(
                                executed_choice[audit_agent]),
                            num_targets=Q,
                            horizon=self.movement_decision_interval,
                        )
                    )
                    target_choice_audit_counter += 1
                sensing_audit_stride = max(
                    0, int(sensing_choice_audit_stride))
                if (sensing_audit_stride > 0
                        and int(eval_env.core.t) % sensing_audit_stride == 0):
                    sensing_horizon = (
                        max(1, int(sensing_audit_horizon))
                        if int(sensing_audit_horizon) > 0
                        else self.movement_decision_interval
                    )
                    sensing_agent = (
                        sensing_choice_audit_counter % max(K, 1))
                    sensing_intervention = (
                        evaluate_sensing_choice_intervention(
                            eval_env,
                            actions,
                            agent_index=sensing_agent,
                            num_targets=Q,
                            horizon=sensing_horizon,
                            residual_blend=sensing_residual_blend,
                        )
                    )
                    episode_sensing_choice_interventions.append(
                        sensing_intervention)
                    if sensing_oracle_control:
                        best_sensing_choice = int(
                            sensing_intervention["best_choice"])
                        if best_sensing_choice >= 0:
                            eval_sensing_oracle_hold = (
                                sensing_agent,
                                best_sensing_choice,
                            )
                            eval_sensing_oracle_until = (
                                int(eval_env.core.t)
                                + sensing_horizon
                            )
                            apply_sensing_choice_intervention(
                                eval_env,
                                sensing_agent,
                                best_sensing_choice,
                                sensing_residual_blend,
                            )
                        else:
                            eval_sensing_oracle_hold = None
                            eval_sensing_oracle_until = -1
                    sensing_choice_audit_counter += 1
                joint_sensing_audit_stride = max(
                    0, int(joint_sensing_pair_audit_stride))
                if (joint_sensing_audit_stride > 0
                        and int(eval_env.core.t)
                        % joint_sensing_audit_stride == 0):
                    joint_sensing_horizon = (
                        max(1, int(sensing_audit_horizon))
                        if int(sensing_audit_horizon) > 0
                        else self.movement_decision_interval
                    )
                    episode_joint_sensing_pair_interventions.append(
                        evaluate_joint_sensing_pair_intervention(
                            eval_env,
                            actions,
                            num_targets=Q,
                            horizon=joint_sensing_horizon,
                            residual_blend=sensing_residual_blend,
                        )
                    )
                responsibility_pre_pd = (
                    None
                    if eval_env.core.prev_P_D is None
                    else np.asarray(
                        eval_env.core.prev_P_D,
                        dtype=np.float64).copy()
                )
                if structure_teacher_trace_output:
                    structure_input_obs = ob.copy()
                    structure_coord_pd_ema_value = getattr(
                        eval_env.core, '_coord_pd_ema', None)
                    structure_coord_pd_ema_valid = bool(
                        structure_coord_pd_ema_value is not None)
                    structure_coord_pd_ema = (
                        np.asarray(
                            structure_coord_pd_ema_value,
                            dtype=np.float32,
                        ).copy()
                        if structure_coord_pd_ema_valid
                        else np.zeros(Q, dtype=np.float32)
                    )
                    structure_delta_p = np.stack([
                        np.asarray(
                            actions[str(k)]['delta_p'],
                            dtype=np.float32,
                        )
                        for k in range(K)
                    ])
                    if self._comm_cost_aware:
                        structure_message = (
                            eval_comm.detach().cpu().numpy().astype(
                                np.float32, copy=True))
                        structure_rate = (
                            eval_rate.detach().cpu().numpy().astype(
                                np.int16, copy=True))
                        structure_token_mask = (
                            eval_token_mask.detach().cpu().numpy().astype(
                                np.uint8, copy=True))
                        if self._joint_isac_power_enabled:
                            structure_comm_fraction = (
                                eval_comm_fraction.detach().cpu().numpy().astype(
                                    np.float32, copy=True))
                            structure_sensing_weights = (
                                eval_sensing_weights.detach().cpu().numpy().astype(
                                    np.float32, copy=True))
                        else:
                            structure_comm_fraction = np.zeros(
                                K, dtype=np.float32)
                            structure_sensing_weights = np.zeros(
                                (K, Q), dtype=np.float32)
                    else:
                        structure_message = np.zeros(
                            (K, int(getattr(
                                self.cfg.marl,
                                'comm_target_token_dim',
                                16)) * Q),
                            dtype=np.float32)
                        structure_rate = np.zeros(K, dtype=np.int16)
                        structure_token_mask = np.zeros(
                            (K, Q), dtype=np.uint8)
                        structure_comm_fraction = np.zeros(
                            K, dtype=np.float32)
                        structure_sensing_weights = np.zeros(
                            (K, Q), dtype=np.float32)
                responsibility_recorded = False
                for _ in range(self.macro_interval):
                    n5_pre_step_env = None
                    if (
                        n5_counterfactual_output
                        and self._dynamic_local_search_config is not None
                        and (
                            int(n5_counterfactual_max_events) <= 0
                            or len(n5_counterfactual_events)
                            < int(n5_counterfactual_max_events)
                        )
                        and (
                            not n5_counterfactual_seed_filter
                            or int(ep_seed) in {
                                int(value)
                                for value in n5_counterfactual_seed_filter
                            }
                        )
                    ):
                        next_frame = int(eval_env.core.t) + 1
                        hold_frames = max(1, int(
                            eval_env.core._p0_maxmin_pairing_hold_frames))
                        fixed_due = bool(
                            eval_env.core._cached_p0_solution is None
                            or next_frame == 1
                            or next_frame % hold_frames == 0
                        )
                        if next_frame > 1 and fixed_due:
                            # Copy after all communication/resource/Student
                            # submissions but immediately before the physical
                            # step, so every forced branch shares the complete
                            # simulator and RNG state.
                            n5_pre_step_env = deepcopy(eval_env)
                    pre_step_xy = np.asarray([
                        uav.pos[:2] for uav in eval_env.core.uavs
                    ], dtype=np.float64)
                    obs, eval_rewards, term, trunc, info = eval_env.step(
                        actions)
                    post_step_xy = np.asarray([
                        uav.pos[:2] for uav in eval_env.core.uavs
                    ], dtype=np.float64)
                    if K >= 2:
                        pairwise_distance = np.linalg.norm(
                            post_step_xy[:, None, :]
                            - post_step_xy[None, :, :],
                            axis=-1,
                        )
                        frame_min_separation = float(np.min(
                            pairwise_distance[
                                np.triu_indices(K, k=1)]))
                    else:
                        frame_min_separation = float('inf')
                    episode_inter_uav_min_distance.append(
                        frame_min_separation)
                    eval_inter_uav_min_distance.append(
                        frame_min_separation)
                    eval_inter_uav_separation_violation.append(float(
                        frame_min_separation
                        < float(self.cfg.uav.d_safe) - 1.0e-9))
                    if K >= 2:
                        pair_i, pair_j = np.triu_indices(K, k=1)
                        relative_start = (
                            pre_step_xy[pair_i] - pre_step_xy[pair_j])
                        relative_end = (
                            post_step_xy[pair_i] - post_step_xy[pair_j])
                        relative_delta = relative_end - relative_start
                        delta_sq = np.sum(relative_delta ** 2, axis=1)
                        closest_time = np.zeros_like(delta_sq)
                        moving_pair = delta_sq > 1.0e-18
                        closest_time[moving_pair] = np.clip(
                            -np.sum(
                                relative_start[moving_pair]
                                * relative_delta[moving_pair],
                                axis=1,
                            ) / delta_sq[moving_pair],
                            0.0,
                            1.0,
                        )
                        continuous_pair_distance = np.linalg.norm(
                            relative_start
                            + closest_time[:, None] * relative_delta,
                            axis=1,
                        )
                        frame_continuous_min_separation = float(np.min(
                            continuous_pair_distance))
                    else:
                        frame_continuous_min_separation = float('inf')
                    episode_inter_uav_continuous_min_distance.append(
                        frame_continuous_min_separation)
                    eval_inter_uav_continuous_min_distance.append(
                        frame_continuous_min_separation)
                    eval_inter_uav_continuous_separation_violation.append(
                        float(
                            frame_continuous_min_separation
                            < float(self.cfg.uav.d_safe) - 1.0e-9))
                    eval_movement_safety_intervened.append(float(
                        info.get('movement_safety_intervened', 0.0)))
                    eval_movement_safety_intervention_call_rate.append(float(
                        info.get(
                            'movement_safety_intervention_call_rate', 0.0)))
                    eval_movement_safety_fail_closed.append(float(
                        info.get('movement_safety_fail_closed', 0.0)))
                    eval_movement_safety_fail_closed_calls.append(float(
                        info.get('movement_safety_fail_closed_calls', 0.0)))
                    eval_movement_safety_fail_closed_outside_invariant_calls.append(
                        float(info.get(
                            'movement_safety_fail_closed_outside_invariant_calls',
                            0.0)))
                    eval_movement_safety_outside_invariant_calls.append(float(
                        info.get('movement_safety_outside_invariant_calls', 0.0)))
                    eval_movement_safety_recovery_pair_count.append(float(
                        info.get('movement_safety_recovery_pair_count', 0.0)))
                    eval_movement_safety_solve_time_s.append(float(
                        info.get('movement_safety_solve_time_s', 0.0)))
                    eval_movement_safety_solve_time_s_per_node.append(float(
                        info.get(
                            'movement_safety_solve_time_s_per_node', 0.0)))
                    eval_movement_safety_projection_calls.append(float(
                        info.get('movement_safety_projection_calls', 0.0)))
                    eval_movement_safety_reduced_linear_qp_calls.append(float(
                        info.get(
                            'movement_safety_reduced_linear_qp_calls', 0.0)))
                    eval_movement_safety_reduced_linear_qp_rate.append(float(
                        info.get(
                            'movement_safety_reduced_linear_qp_rate', 0.0)))
                    eval_movement_public_view_stale_fail_closed_rate.append(
                        float(info.get(
                            'movement_public_view_stale_fail_closed_rate',
                            0.0)))
                    eval_movement_public_view_mean_max_age.append(float(
                        info.get(
                            'movement_public_view_mean_max_age_frames', 0.0)))
                    eval_movement_public_view_max_age.append(float(info.get(
                        'movement_public_view_max_age_frames', 0.0)))
                    eval_movement_envelope_gap_selected_rate.append(float(
                        info.get(
                            'movement_gap_baseline_envelope_gap_selected_rate',
                            0.0)))
                    eval_movement_envelope_single_projection_rate.append(float(
                        info.get(
                            'movement_gap_baseline_envelope_single_projection_rate',
                            0.0)))
                    eval_movement_envelope_selection_disagreement_rate.append(
                        float(info.get(
                            'movement_gap_baseline_envelope_selection_disagreement_rate',
                            0.0)))
                    eval_movement_primal_dual_candidate_enabled.append(float(
                        info.get(
                            'movement_gap_primal_dual_candidate_enabled',
                            0.0)))
                    eval_movement_primal_dual_certificate_available_rate.append(
                        float(info.get(
                            'movement_gap_primal_dual_certificate_available_rate',
                            0.0)))
                    eval_movement_primal_dual_selected_rate.append(float(
                        info.get(
                            'movement_gap_primal_dual_selected_rate', 0.0)))
                    eval_movement_primal_dual_strong_dominance_rate.append(
                        float(info.get(
                            'movement_gap_primal_dual_strong_dominance_rate',
                            0.0)))
                    eval_movement_primal_dual_lower_improvement.append(float(
                        info.get(
                            'movement_gap_primal_dual_lower_improvement_mean',
                            0.0)))
                    eval_movement_primal_dual_certificate_width.append(float(
                        info.get(
                            'movement_gap_primal_dual_certificate_width_mean',
                            0.0)))
                    eval_movement_primal_dual_incumbent_lower.append(float(
                        info.get(
                            'movement_gap_primal_dual_incumbent_lower_mean',
                            0.0)))
                    eval_movement_primal_dual_candidate_lower.append(float(
                        info.get(
                            'movement_gap_primal_dual_candidate_lower_mean',
                            0.0)))
                    eval_movement_primal_dual_selection_disagreement_rate.append(
                        float(info.get(
                            'movement_gap_primal_dual_selection_disagreement_rate',
                            0.0)))
                    eval_movement_assignment_target_coverage.append(float(
                        info.get(
                            'movement_executed_assignment_target_coverage',
                            0.0)))
                    eval_movement_assignment_duplicate_rate.append(float(
                        info.get(
                            'movement_executed_assignment_duplicate_rate',
                            0.0)))
                    eval_movement_assignment_entry_agreement.append(float(
                        info.get(
                            'movement_local_assignment_entry_agreement_rate',
                            0.0)))
                    eval_movement_anchor_target_coverage.append(float(
                        info.get(
                            'movement_anchor_target_coverage', 0.0)))
                    eval_movement_gain_schedule_selected_self_rate.append(
                        float(info.get(
                            'movement_gain_schedule_selected_self_rate',
                            0.0)))
                    eval_movement_gain_schedule_strategy_hold_rate.append(
                        float(info.get(
                            'movement_gain_schedule_strategy_hold_rate',
                            0.0)))
                    eval_movement_gain_schedule_no_improvement_rate.append(
                        float(info.get(
                            'movement_gain_schedule_no_improvement_rate',
                            0.0)))
                    eval_movement_gain_schedule_far_range_rate.append(
                        float(info.get(
                            'movement_gain_schedule_far_range_rate',
                            0.0)))
                    if (
                        n5_pre_step_env is not None
                        and bool(info.get('p0_resolved', False))
                    ):
                        from uav_isac.evaluation.n5_counterfactual_audit import (
                            audit_atomic_n5_event,
                            has_proxy_positive_atomic_n5,
                        )

                        try:
                            should_audit_n5 = (
                                not n5_counterfactual_require_proxy_positive
                                or has_proxy_positive_atomic_n5(eval_env)
                            )
                            if should_audit_n5:
                                n5_counterfactual_events.append(
                                    audit_atomic_n5_event(
                                        n5_pre_step_env,
                                        eval_env,
                                        actions,
                                        info,
                                        episode_seed=int(ep_seed),
                                        frame=int(eval_env.core.t),
                                        target_mode=(
                                            n5_counterfactual_target_mode),
                                        max_candidates=max(
                                            0,
                                            int(n5_counterfactual_max_candidates),
                                        ),
                                    )
                                )
                        finally:
                            n5_pre_step_env.close()
                    elif n5_pre_step_env is not None:
                        n5_pre_step_env.close()
                    pd_q = info['P_D_q'].copy()
                    pd_hist.append(np.mean(pd_q))
                    pd_per_target.append(pd_q)
                    episode_commitment_coverage.append(float(info.get(
                        'learned_comm_commitment_target_coverage', 0.0)))
                    episode_hard_commitment_coverage.append(float(info.get(
                        'learned_comm_commitment_hard_target_coverage', 0.0)))
                    episode_p0_target_coverage.append(float(info.get(
                        'p0_target_coverage', 0.0)))
                    physical_stride = max(
                        0, int(physical_oracle_stride))
                    if (physical_stride > 0
                            and int(eval_env.core.t)
                            % physical_stride == 0):
                        current_sensing_power = np.asarray(
                            eval_env.core._current_sensing_power_w,
                            dtype=np.float64,
                        )
                        current_comm_power = np.asarray(
                            eval_env.core._current_comm_power_w,
                            dtype=np.float64,
                        )
                        reports_per_receiver = (
                            max(1, int(
                                self.cfg.p0_solver.capacity_per_rx
                                // max(self.cfg.detection.B_q, 1)))
                            if eval_env.core.ground_communication_enabled
                            else Q
                        )
                        episode_physical_oracles.append(
                            evaluate_physical_feasibility_oracles(
                                eval_env.current_step_info.deflection_entries,
                                current_sensing_power,
                                pd_q,
                                eval_env.current_step_info.p0_solution.selected_set,
                                num_uavs=K,
                                num_targets=Q,
                                p_fa=self.cfg.detection.P_FA,
                                total_power_w=float(
                                    eval_env.core._isac_total_power_w),
                                communication_reserve_w=float(
                                    np.mean(current_comm_power)),
                                sensing_power_cap_w=float(
                                    self.cfg.uav.P_sense_max),
                                target_pair_limit=int(
                                    self.cfg.detection.K_q_max),
                                reports_per_receiver=reports_per_receiver,
                                detection_fusion_mode=str(
                                    eval_env.core._detection_fusion_mode),
                                seed=(
                                    int(ep_seed) * 1000
                                    + int(eval_env.core.t)
                                ),
                            )
                        )
                    evidence_oracle = receiver_local_and_global_pd(
                        eval_env.current_step_info.p0_solution.selected_set,
                        eval_env.current_step_info.deflection_entries,
                        K,
                        Q,
                        self.cfg.detection.P_FA,
                    )
                    if evidence_trace_output:
                        executed_receiver_d = np.asarray(
                            (
                                eval_env.current_step_info.learned_comm or {}
                            ).get(
                                'detection_receiver_deflection',
                                evidence_oracle['receiver_deflection'],
                            ),
                            dtype=np.float64,
                        )
                        evidence_trace_receiver_d.append(
                            executed_receiver_d.copy())
                        evidence_trace_owner.append(np.asarray(
                            (
                                eval_env.current_step_info.learned_comm or {}
                            ).get(
                                'evidence_owner',
                                np.full(Q, -1, dtype=np.int64),
                            ),
                            dtype=np.int64,
                        ).copy())
                        evidence_trace_episode.append(
                            len(eval_evidence_global_histories))
                        evidence_trace_frame.append(
                            len(episode_evidence_global))
                        evidence_trace_positions.append(np.stack([
                            eval_env.core.uavs[k].pos.copy()
                            for k in range(K)
                        ]))
                        evidence_trace_comm_power.append(np.asarray(
                            getattr(
                                eval_env.core,
                                '_current_comm_power_w',
                                np.zeros(K),
                            ),
                            dtype=np.float64,
                        ).copy())
                    if structure_teacher_trace_output:
                        from uav_isac.evaluation.structure_teacher import (
                            deflection_entry_tensors,
                            structure_teacher_labels,
                        )
                        selected = (
                            eval_env.current_step_info.p0_solution.selected_set)
                        labels = structure_teacher_labels(selected, K, Q)
                        physical = deflection_entry_tensors(
                            eval_env.current_step_info.deflection_entries,
                            K,
                            Q,
                        )
                        structure_trace_episode.append(
                            len(eval_evidence_global_histories))
                        structure_trace_seed.append(int(ep_seed))
                        structure_trace_frame.append(
                            int(eval_env.current_step_info.frame))
                        structure_trace_p0_resolved.append(bool(
                            eval_env.current_step_info.p0_resolved))
                        structure_trace_obs.append(
                            structure_input_obs.astype(
                                np.float32, copy=True))
                        structure_trace_next_obs.append(np.stack([
                            obs[str(k)] for k in range(K)
                        ]).astype(np.float32, copy=False))
                        structure_trace_delta_p.append(
                            structure_delta_p.copy())
                        structure_trace_message.append(
                            structure_message.copy())
                        structure_trace_rate.append(
                            structure_rate.copy())
                        structure_trace_token_mask.append(
                            structure_token_mask.copy())
                        structure_trace_comm_fraction.append(
                            structure_comm_fraction.copy())
                        structure_trace_sensing_weights.append(
                            structure_sensing_weights.copy())
                        structure_trace_uav_positions.append(np.stack([
                            eval_env.core.uavs[k].pos
                            for k in range(K)
                        ]).astype(np.float32))
                        structure_trace_uav_velocities.append(np.stack([
                            eval_env.core.uavs[k].vel
                            for k in range(K)
                        ]).astype(np.float32))
                        structure_trace_target_states.append(np.stack([
                            np.concatenate([
                                target.get_position_3d()[:2],
                                target.state[2:4],
                            ])
                            for target in eval_env.core.targets
                        ]).astype(np.float32))
                        structure_trace_target_positions.append(np.stack([
                            target.get_position_3d()
                            for target in eval_env.core.targets
                        ]).astype(np.float32))
                        structure_trace_target_velocities.append(np.stack([
                            np.asarray([
                                *target.get_velocity(), 0.0
                            ], dtype=np.float32)
                            for target in eval_env.core.targets
                        ]))
                        deployed_sensing_power = np.asarray(
                            eval_env.core._current_sensing_power_w,
                            dtype=np.float64).copy()
                        deployed_comm_power = np.asarray(
                            eval_env.core._current_comm_power_w,
                            dtype=np.float64).copy()
                        structure_trace_sensing_power.append(
                            deployed_sensing_power)
                        structure_trace_comm_power.append(
                            deployed_comm_power)
                        structure_trace_sensing_budget.append(np.minimum(
                            np.maximum(
                                float(eval_env.core._isac_total_power_w)
                                - deployed_comm_power,
                                0.0,
                            ),
                            float(eval_env.core._sensing_power_cap_w),
                        ))
                        structure_trace_per_watt_coefficient.append(
                            eval_env.core._per_watt_coefficient_from_entries(
                                eval_env.current_step_info.deflection_entries
                            ).copy())
                        task_pd = np.asarray(pd_q, dtype=np.float64)
                        weak_count = min(3, Q)
                        structure_trace_task_mean_pd.append(float(
                            np.mean(task_pd)))
                        structure_trace_task_weak3_pd.append(float(np.mean(
                            np.partition(task_pd, weak_count - 1)[
                                :weak_count])))
                        structure_trace_task_worst_pd.append(float(
                            np.min(task_pd)))
                        structure_trace_local_pd.append(np.stack([
                            eval_env.core.prev_P_D_local.get(
                                k, np.zeros(Q, dtype=np.float64))
                            for k in range(K)
                        ]).astype(np.float32))
                        structure_trace_coord_pd_ema.append(
                            structure_coord_pd_ema.copy())
                        structure_trace_coord_pd_ema_valid.append(
                            structure_coord_pd_ema_valid)
                        structure_trace_pair.append(labels["pair"])
                        structure_trace_tx_target.append(
                            labels["tx_target"])
                        structure_trace_rx_target.append(
                            labels["rx_target"])
                        structure_trace_endpoint_target.append(
                            labels["endpoint_target"])
                        structure_trace_receiver_owner.append(
                            labels["receiver_owner"])
                        structure_trace_role.append(labels["role"])
                        structure_trace_candidate.append(
                            physical["candidate"])
                        structure_trace_d_eff.append(physical["d_eff"])
                        structure_trace_d_raw.append(physical["d_raw"])
                        structure_trace_alpha.append(physical["alpha"])
                        structure_trace_g_dd.append(physical["g_dd"])
                        structure_trace_chi_rep.append(
                            physical["chi_rep"])
                        structure_trace_physical_pd.append(
                            np.asarray(pd_q, dtype=np.float64).copy())
                        structure_trace_rebootstrap_attempted.append(float(
                            info.get(
                                'local_search_rebootstrap_attempted', 0.0)))
                        structure_trace_rebootstrap_accepted.append(float(
                            info.get(
                                'local_search_rebootstrap_accepted', 0.0)))
                    # Keep the central-oracle label independent of the active
                    # environment evidence boundary.
                    episode_evidence_global.append(
                        evidence_oracle['global_pd'].copy())
                    episode_evidence_local.append(
                        evidence_oracle['local_best_pd'].copy())
                    episode_evidence_top1.append(
                        evidence_oracle[
                            'lossless_quality_top1_pd'].copy())
                    episode_evidence_top2.append(
                        evidence_oracle[
                            'lossless_quality_top2_pd'].copy())
                    active_fusion_mode = str(getattr(
                        self.cfg.marl,
                        'detection_fusion_mode',
                        'legacy_global',
                    )).strip().lower()
                    expected_pd = (
                        evidence_oracle['local_best_pd']
                        if active_fusion_mode == 'local_only'
                        else evidence_oracle['global_pd']
                    )
                    if active_fusion_mode != 'u2u_distributed':
                        eval_evidence_reconstruction_errors.append(float(
                            np.max(np.abs(
                                expected_pd
                                - np.asarray(pd_q, dtype=np.float64)))))
                    # Match the movement head's deployed task stream.  The
                    # transport cost is removed because rate/resource heads,
                    # not the held movement action, control it.
                    comm_cost = float(info.get(
                        'reward_components', {}).get(
                            'learned_comm_cost', 0.0))
                    comm_cost_scale = (
                        float(getattr(self.cfg.marl, 'team_weight', 0.7))
                        if getattr(
                            self.cfg.marl, 'use_difference_reward', False)
                        else 1.0
                    )
                    episode_movement_rewards.append([
                        float(eval_rewards[str(k)])
                        + comm_cost_scale * comm_cost
                        for k in range(K)
                    ])
                    if (movement_decision
                            and not responsibility_recorded
                            and responsibility_pre_pd is not None):
                        responsibility = classify_responsibility_frame(
                            executed_choice,
                            executed_active,
                            eval_env.current_step_info.p0_solution.selected_set,
                            eval_env.current_step_info.deflection_entries,
                            Q,
                            self.cfg.detection.P_FA,
                        )
                        responsibility.update({
                            'frame_index': len(pd_per_target) - 1,
                            'pre_pd': responsibility_pre_pd,
                        })
                        episode_responsibility_records.append(
                            responsibility)
                        responsibility_recorded = True
                    if pending_target_choice_intervention is not None:
                        audit_agent = int(
                            pending_target_choice_intervention[
                                'agent_index'])
                        audit_choice = int(
                            pending_target_choice_intervention[
                                'actual_choice'])
                        audit_counterfactual = node_removal_marginal_pd(
                            eval_env.current_step_info.p0_solution.selected_set,
                            eval_env.current_step_info.deflection_entries,
                            K,
                            Q,
                            self.cfg.detection.P_FA,
                        )
                        audit_load = np.bincount(
                            executed_choice[executed_active],
                            minlength=Q)
                        audit_marginal = float(
                            audit_counterfactual[
                                'node_marginal_pd'][
                                    audit_agent, audit_choice])
                        duplicate_choice = bool(
                            executed_active[audit_agent]
                            and audit_load[audit_choice] > 1)
                        pending_target_choice_intervention.update({
                            'actual_node_marginal_pd': audit_marginal,
                            'actual_is_duplicate': float(
                                duplicate_choice),
                            'actual_is_ineffective_duplicate': float(
                                duplicate_choice
                                and audit_marginal <= 1.0e-4),
                            'actual_is_productive_duplicate': float(
                                duplicate_choice
                                and audit_marginal > 1.0e-4),
                        })
                        episode_target_choice_interventions.append(
                            pending_target_choice_intervention)
                        pending_target_choice_intervention = None
                    if risk_quantiles_step is not None:
                        episode_risk_quantiles.append(
                            risk_quantiles_step.copy())
                        episode_risk_logits.append(risk_logits_step.copy())
                        eval_risk_quantiles.append(
                            risk_quantiles_step.copy())
                        eval_risk_logits.append(risk_logits_step.copy())
                        eval_risk_targets.append(pd_q.copy())
                    frame_uav_xy = np.stack([
                        eval_env.core.uavs[k].pos[:2] for k in range(K)])
                    frame_target_xy = np.stack([
                        target.get_position_3d()[:2]
                        for target in eval_env.core.targets])
                    nearest_target_distance.append(np.min(np.linalg.norm(
                        frame_uav_xy[:, None, :]
                        - frame_target_xy[None, :, :], axis=-1), axis=0))
                    total_frames += 1
                    vp_frames += int(info.get('valid_pair', False))
                    notx_frames += int(info.get('no_tx', False))
                    samerole_frames += int(info.get('all_same_role', False))
                    n_duplex = int(info.get('n_duplex', 0))
                    duplex_endpoint_frames += int(n_duplex > 0)
                    duplex_endpoint_nodes.append(n_duplex)
                    eval_comm_bits.append(float(
                        info.get('learned_comm_bits', 0.0)))
                    eval_comm_energy.append(float(
                        info.get('learned_comm_energy_j', 0.0)))
                    # Latency/delivery are packet metrics. Do not dilute them
                    # with the no-packet hold frames, but keep bits, energy and
                    # active senders as true per-simulator-frame quantities.
                    attempted = float(info.get(
                        'learned_comm_attempted_links', 0.0))
                    if attempted > 0.0:
                        eval_comm_latency.append(float(
                            info.get('learned_comm_mean_latency_s', 0.0)))
                        eval_comm_p95_latency.append(float(
                            info.get('learned_comm_p95_latency_s', 0.0)))
                        eval_comm_mean_bler.append(float(info.get(
                            'learned_comm_mean_packet_error_probability', 0.0)))
                        eval_comm_max_bler.append(float(info.get(
                            'learned_comm_max_packet_error_probability', 0.0)))
                        eval_comm_reliability_violation.append(float(info.get(
                            'learned_comm_reliability_violation_rate', 0.0)))
                        eval_comm_burst_failure.append(float(info.get(
                            'learned_comm_burst_failure_rate', 0.0)))
                        eval_comm_burst_bad_link.append(float(info.get(
                            'learned_comm_burst_bad_link_rate', 0.0)))
                        eval_comm_delivery.append(float(
                            info.get('learned_comm_delivery_rate', 0.0)))
                        eval_comm_violation.append(float(info.get(
                            'learned_comm_deadline_violation_rate', 0.0)))
                        eval_comm_snr_violation.append(float(info.get(
                            'learned_comm_snr_violation_rate', 0.0)))
                        eval_comm_delivery_failure.append(float(info.get(
                            'learned_comm_delivery_failure_rate', 0.0)))
                    eval_comm_active.append(float(
                        info.get('learned_comm_active_senders', 0.0)))
                    eval_structure_payload_bits.append(float(info.get(
                        'structure_student_payload_bits', 0.0)))
                    eval_p0_target_selected_mask.append(np.asarray(
                        info.get('p0_target_selected_mask', np.zeros(Q)),
                        dtype=np.float64))
                    if float(info.get(
                            'structure_student_atomic_attempted_senders',
                            0.0)) > 0.0:
                        eval_structure_atomic_delivery.append(float(info.get(
                            'structure_student_atomic_delivery_rate', 0.0)))
                        eval_structure_selected_bits.append(float(info.get(
                            'structure_student_selected_bits_per_dim',
                            self._structure_student_bits_per_dim)))
                    eval_structure_public_valid.append(float(info.get(
                        'structure_student_public_valid_fraction', 0.0)))
                    eval_structure_public_age.append(float(info.get(
                        'structure_student_public_mean_age_frames', 0.0)))
                    eval_structure_insufficient_cache.append(float(info.get(
                        'structure_student_insufficient_cache', 0.0)))
                    eval_p0_resolved.append(float(
                        info.get('p0_resolved', False)))
                    eval_p0_solve_time.append(float(
                        info.get('p0_solve_time_s', 0.0)))
                    eval_p0_topology_invalid.append(float(info.get(
                        'p0_topology_invalid_frame', 0.0)))
                    eval_p0_topology_invalid_edges.append(float(info.get(
                        'p0_topology_invalid_edge_count', 0.0)))
                    repair_applied = float(info.get(
                        'p0_topology_min_change_repair_applied', 0.0))
                    eval_p0_topology_repair.append(repair_applied)
                    eval_p0_topology_replacements.append(float(info.get(
                        'p0_topology_min_change_replacement_count', 0.0)))
                    repair_worst = float(info.get(
                        'p0_topology_repair_candidate_worst_pd', np.nan))
                    if repair_applied > 0.0 and np.isfinite(repair_worst):
                        eval_p0_topology_repair_worst_pd.append(repair_worst)
                    eval_p0_topology_full_fallback.append(float(info.get(
                        'p0_topology_full_fallback', 0.0)))
                    if float(info.get('local_search_resolved', 0.0)) > 0.0:
                        eval_local_search_candidates.append(float(info.get(
                            'local_search_candidate_count', 0.0)))
                        eval_local_search_verifications.append(float(info.get(
                            'local_search_exact_verifications', 0.0)))
                        eval_local_search_accepted.append(float(info.get(
                            'local_search_accepted_moves', 0.0)))
                        eval_local_search_cold.append(float(info.get(
                            'local_search_cold_start', 0.0)))
                        eval_local_search_rebootstrap_attempted.append(
                            float(info.get(
                                'local_search_rebootstrap_attempted', 0.0)))
                        eval_local_search_rebootstrap_accepted.append(
                            float(info.get(
                                'local_search_rebootstrap_accepted', 0.0)))
                        eval_local_search_rebootstrap_candidates.append(
                            float(info.get(
                                'local_search_rebootstrap_candidate_count',
                                0.0)))
                        eval_local_search_rebootstrap_deficit_present.append(
                            float(info.get(
                                'local_search_rebootstrap_deficit_present',
                                0.0)))
                        eval_local_search_rebootstrap_blocked_no_deficit.append(
                            float(info.get(
                                'local_search_rebootstrap_blocked_no_deficit',
                                0.0)))
                    eval_evidence_comm_bits.append(float(
                        info.get('evidence_comm_bits', 0.0)))
                    eval_protocol_total_latency.append(float(
                        info.get('protocol_total_latency_s', 0.0)))
                    eval_protocol_total_p95_latency.append(float(
                        info.get('protocol_total_p95_latency_s', 0.0)))
                    eval_protocol_serialized_bits.append(float(
                        info.get('protocol_serialized_bits', 0.0)))
                    eval_protocol_total_deadline_violation.append(float(
                        info.get('protocol_total_deadline_violation', 0.0)))
                    eval_evidence_comm_energy.append(float(
                        info.get('evidence_comm_energy_j', 0.0)))
                    evidence_attempted = float(info.get(
                        'evidence_comm_attempted_links', 0.0))
                    if evidence_attempted > 0.0:
                        eval_evidence_comm_latency.append(float(info.get(
                            'evidence_comm_mean_latency_s', 0.0)))
                        eval_evidence_comm_delivery.append(float(info.get(
                            'evidence_comm_delivery_rate', 0.0)))
                        eval_evidence_comm_violation.append(float(info.get(
                            'evidence_comm_deadline_violation_rate', 0.0)))
                        eval_evidence_comm_failure.append(float(info.get(
                            'evidence_comm_delivery_failure_rate', 0.0)))
                        eval_evidence_comm_snr_failure.append(float(info.get(
                            'evidence_comm_snr_violation_rate', 0.0)))
                        eval_evidence_comm_reliability_failure.append(float(
                            info.get(
                                'evidence_comm_reliability_violation_rate',
                                0.0)))
                        eval_evidence_comm_burst_failure.append(float(info.get(
                            'evidence_comm_burst_failure_rate', 0.0)))
                        eval_evidence_comm_burst_bad.append(float(info.get(
                            'evidence_comm_burst_bad_link_rate', 0.0)))
                    eval_evidence_comm_active.append(float(info.get(
                        'evidence_comm_active_senders', 0.0)))
                    eval_evidence_utilization.append(float(info.get(
                        'evidence_utilization', 0.0)))
                    eval_evidence_transmitted_entries.append(float(info.get(
                        'evidence_transmitted_entries', 0.0)))
                    eval_evidence_useful_entries.append(float(info.get(
                        'evidence_useful_unique_entries', 0.0)))
                    eval_evidence_pfa.append(float(info.get(
                        'evidence_detection_aggregate_pfa',
                        self.cfg.detection.P_FA,
                    )))
                    eval_belief_feedback_entries.append(float(info.get(
                        'belief_feedback_fused_entries', 0.0)))
                    eval_belief_feedback_receivers.append(float(info.get(
                        'belief_feedback_receivers', 0.0)))
                    eval_belief_feedback_before.append(float(info.get(
                        'belief_feedback_disagreement_before_m', 0.0)))
                    eval_belief_feedback_after.append(float(info.get(
                        'belief_feedback_disagreement_after_m', 0.0)))
                    eval_belief_feedback_contraction.append(float(info.get(
                        'belief_feedback_contraction_ratio', 1.0)))
                    eval_belief_feedback_truth_rmse_before.append(float(
                        info.get('belief_feedback_truth_rmse_before_m', 0.0)))
                    eval_belief_feedback_truth_rmse_after.append(float(
                        info.get('belief_feedback_truth_rmse_after_m', 0.0)))
                    if 'belief_position_rmse_m' in info:
                        eval_belief_position_rmse.append(float(
                            info['belief_position_rmse_m']))
                        eval_belief_position_p95.append(float(
                            info['belief_position_error_p95_m']))
                        eval_belief_position_rmse_per_target.append(np.asarray(
                            info['belief_position_rmse_per_target_m'],
                            dtype=np.float64))
                        eval_belief_target_aoi_mean.append(float(
                            info['belief_target_aoi_mean_frames']))
                        eval_belief_target_aoi_p95.append(float(
                            info['belief_target_aoi_p95_frames']))
                        eval_belief_target_aoi_per_target.append(np.asarray(
                            info['belief_target_aoi_per_target_frames'],
                            dtype=np.float64))
                        eval_belief_disagreement_end.append(float(
                            info['belief_position_disagreement_end_m']))
                    if self._joint_isac_power_enabled:
                        eval_isac_comm_power.append(float(
                            info.get('isac_comm_power_w', 0.0)))
                        eval_isac_sensing_power.append(float(
                            info.get('isac_sensing_power_w', 0.0)))
                        eval_isac_balance_error.append(float(
                            info.get('isac_max_power_balance_error_w', 0.0)))
                        eval_isac_budget_violation.append(float(
                            info.get(
                                'isac_max_power_budget_violation_w', 0.0)))
                        eval_isac_unused_power.append(float(
                            info.get('isac_unused_power_w', 0.0)))
                        eval_isac_target_power.append(np.asarray(
                            info.get('isac_target_power_w', np.zeros(Q)),
                            dtype=np.float64))
                        if float(info.get(
                            'distributed_replicated_power_enabled', 0.0
                        )) > 0.0:
                            eval_replicated_common_view.append(float(info.get(
                                'distributed_replicated_power_common_view',
                                0.0)))
                            eval_replicated_full_view.append(float(info.get(
                                'distributed_replicated_power_full_view_fraction',
                                0.0)))
                            eval_replicated_local_coverage.append(float(info.get(
                                'distributed_replicated_power_local_coverage',
                                0.0)))
                            eval_replicated_solve_time.append(float(info.get(
                                'distributed_replicated_power_solve_time_s',
                                0.0)))
                            eval_replicated_per_node_solve_time.append(float(
                                info.get(
                                    'distributed_replicated_power_per_node_solve_time_s',
                                    0.0)))
                            eval_replicated_parallel_used.append(float(info.get(
                                'distributed_replicated_power_process_parallel_used',
                                0.0)))
                            eval_replicated_parallel_batch_wall.append(float(
                                info.get(
                                    'distributed_replicated_power_parallel_batch_wall_s',
                                    0.0)))
                            eval_replicated_history_reserve_used.append(float(
                                info.get(
                                    'distributed_replicated_power_history_reserve_used_fraction',
                                    0.0)))
                    if float(info.get('hyperedge_enabled', 0.0)) > 0.0:
                        eval_hyperedge_visible_peers.append(float(info.get(
                            'hyperedge_visible_peers_per_uav', 0.0)))
                        eval_hyperedge_mutual_edges.append(float(info.get(
                            'hyperedge_mutual_edges', 0.0)))
                        eval_hyperedge_active_edges.append(float(info.get(
                            'hyperedge_active_edges', 0.0)))
                        eval_hyperedge_target_coverage.append(float(info.get(
                            'hyperedge_target_coverage', 0.0)))
                        eval_hyperedge_protocol_used.append(float(info.get(
                            'hyperedge_protocol_used', 0.0)))
                        eval_hyperedge_safety_fallback.append(float(info.get(
                            'hyperedge_safety_fallback', 0.0)))
                        eval_hyperedge_assignment_reused.append(float(info.get(
                            'hyperedge_assignment_reused', 0.0)))
                    if (term.get('__all__', False)
                            or trunc.get('__all__', False)):
                        episode_done = True
                        break
                eval_movement_phase = (
                    eval_movement_phase + self.macro_interval
                ) % self.movement_decision_interval
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
                ep_trimmed_worst.append(float(np.mean(sorted_q[:min(2, Q)])))
                ep_tstd.append(float(steady_per_target.std()))
                ep_commitment_coverage.append(float(np.mean(
                    episode_commitment_coverage[-w:] or [0.0])))
                ep_hard_commitment_coverage.append(float(np.mean(
                    episode_hard_commitment_coverage[-w:] or [0.0])))
                ep_p0_target_coverage.append(float(np.mean(
                    episode_p0_target_coverage[-w:] or [0.0])))
                if episode_risk_quantiles:
                    initial_quantiles = np.asarray(
                        episode_risk_quantiles[0])
                    tail_count = max(1, int(np.ceil(
                        self.agents[0].critic.risk_cvar_alpha
                        * self.agents[0].critic.risk_num_quantiles)))
                    initial_cvar_by_target = np.mean(
                        np.sort(initial_quantiles, axis=-1)[
                            :, :tail_count],
                        axis=-1)
                    eval_episode_initial_cvar.append(float(
                        np.min(initial_cvar_by_target)))
                    eval_episode_realized_worst.append(float(sorted_q[0]))
                steady_distance = np.array(nearest_target_distance[-w:]).mean(axis=0)
                ep_nearest_target_distance.append(steady_distance)
                ep_min_inter_uav_distance.append(float(np.min(
                    episode_inter_uav_min_distance or [float('inf')])))
                ep_min_inter_uav_continuous_distance.append(float(np.min(
                    episode_inter_uav_continuous_min_distance
                    or [float('inf')])))
            eval_responsibility_episodes.append(
                episode_responsibility_records)
            eval_responsibility_pd_histories.append(np.asarray(
                pd_per_target, dtype=np.float64))
            eval_movement_reward_histories.append(np.asarray(
                episode_movement_rewards, dtype=np.float64))
            eval_target_choice_intervention_episodes.append(
                episode_target_choice_interventions)
            eval_sensing_choice_intervention_episodes.append(
                episode_sensing_choice_interventions)
            eval_joint_sensing_pair_intervention_episodes.append(
                episode_joint_sensing_pair_interventions)
            eval_physical_oracle_episodes.append(
                episode_physical_oracles)
            eval_evidence_global_histories.append(np.asarray(
                episode_evidence_global, dtype=np.float64))
            eval_evidence_local_histories.append(np.asarray(
                episode_evidence_local, dtype=np.float64))
            eval_evidence_top1_histories.append(np.asarray(
                episode_evidence_top1, dtype=np.float64))
            eval_evidence_top2_histories.append(np.asarray(
                episode_evidence_top2, dtype=np.float64))

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

        if ep_nearest_target_distance:
            nearest_distance_mat = np.asarray(ep_nearest_target_distance)
            per_target_nearest_distance = nearest_distance_mat.mean(axis=0)
            mean_nearest_distance = float(np.mean(nearest_distance_mat))
            worst_nearest_distance = float(np.mean(
                np.max(nearest_distance_mat, axis=1)))
        else:
            per_target_nearest_distance = np.zeros(Q)
            mean_nearest_distance = worst_nearest_distance = 0.0
        actor_choice_row_sum = actor_target_choice_counts.sum(
            axis=1, keepdims=True)
        actor_target_choice_matrix = np.divide(
            actor_target_choice_counts,
            actor_choice_row_sum,
            out=np.zeros_like(actor_target_choice_counts, dtype=np.float64),
            where=actor_choice_row_sum > 0,
        )

        rate_count = int(np.sum(eval_comm_rate_counts))
        rate_dist = (eval_comm_rate_counts / rate_count
                     if rate_count > 0 else np.zeros(len(rate_levels)))
        robust_stats = compute_robust_checkpoint_statistics(
            np.asarray(ep_steady_means),
            np.asarray(ep_weak3),
            np.asarray(ep_worst),
            self._comm_qos_targets,
            alpha=self._checkpoint_confidence_alpha,
            bootstrap_samples=self._checkpoint_bootstrap_samples,
            cvar_fraction=self._checkpoint_cvar_fraction,
            qos_tol=float(getattr(self.cfg.marl, 'qos_eval_tol', 1e-6)),
        )
        responsibility_stats = summarize_responsibility_episodes(
            eval_responsibility_episodes,
            eval_responsibility_pd_histories,
            horizon=self.movement_decision_interval,
            bootstrap_samples=self._checkpoint_bootstrap_samples,
        )
        temporal_credit_stats = summarize_temporal_credit_proxy(
            eval_movement_reward_histories,
            eval_responsibility_pd_histories,
            gamma=self.gamma_micro,
            gae_lambda=self.gae_lambda,
            interval=self.movement_decision_interval,
            bootstrap_samples=self._checkpoint_bootstrap_samples,
        )
        target_choice_stats = summarize_target_choice_interventions(
            eval_target_choice_intervention_episodes,
            bootstrap_samples=self._checkpoint_bootstrap_samples,
        )
        sensing_choice_stats = summarize_sensing_choice_interventions(
            eval_sensing_choice_intervention_episodes,
            bootstrap_samples=self._checkpoint_bootstrap_samples,
        )
        joint_sensing_pair_stats = (
            summarize_joint_sensing_pair_interventions(
                eval_joint_sensing_pair_intervention_episodes,
                bootstrap_samples=self._checkpoint_bootstrap_samples,
            )
        )
        physical_oracle_stats = summarize_physical_feasibility_oracles(
            eval_physical_oracle_episodes,
            worst_floor=float(self._comm_qos_targets[2]),
            bootstrap_samples=self._checkpoint_bootstrap_samples,
        )
        evidence_oracle_stats = summarize_evidence_oracle(
            eval_evidence_global_histories,
            eval_evidence_local_histories,
            steady_window=W,
            bootstrap_samples=self._checkpoint_bootstrap_samples,
        )
        evidence_oracle_stats[
            'eval_evidence_oracle_reconstruction_max_error'
        ] = float(max(eval_evidence_reconstruction_errors or [0.0]))
        evidence_topk_capacity_stats = summarize_lossless_topk_capacity(
            eval_evidence_global_histories,
            eval_evidence_local_histories,
            {
                1: eval_evidence_top1_histories,
                2: eval_evidence_top2_histories,
            },
            steady_window=W,
            bootstrap_samples=self._checkpoint_bootstrap_samples,
        )
        if evidence_trace_output:
            trace_path = os.path.abspath(evidence_trace_output)
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            np.savez_compressed(
                trace_path,
                receiver_deflection=np.asarray(
                    evidence_trace_receiver_d, dtype=np.float64),
                fusion_owner=np.asarray(
                    evidence_trace_owner, dtype=np.int64),
                episode_index=np.asarray(
                    evidence_trace_episode, dtype=np.int32),
                frame_index=np.asarray(
                    evidence_trace_frame, dtype=np.int32),
                uav_positions=np.asarray(
                    evidence_trace_positions, dtype=np.float64),
                comm_power_w=np.asarray(
                    evidence_trace_comm_power, dtype=np.float64),
                episode_seeds=np.asarray(eval_seeds, dtype=np.int64),
                p_fa=np.asarray(
                    self.cfg.detection.P_FA, dtype=np.float64),
                num_agents=np.asarray(K, dtype=np.int32),
                num_targets=np.asarray(Q, dtype=np.int32),
            )
        if structure_teacher_trace_output:
            from uav_isac.evaluation.layer_provenance import (
                LAYER_TRACE_SCHEMA_VERSION,
                build_layer_trace_provenance,
                physics_contract_from_environment,
            )
            trace_path = os.path.abspath(structure_teacher_trace_output)
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            trace_provenance = build_layer_trace_provenance(
                run_binding=structure_trace_run_binding or {},
                physics_contract=physics_contract_from_environment(
                    eval_env.core, self.cfg),
            )
            np.savez_compressed(
                trace_path,
                schema_version=np.asarray(
                    [LAYER_TRACE_SCHEMA_VERSION], dtype=np.int16),
                provenance_json=np.asarray([
                    json.dumps(
                        trace_provenance, sort_keys=True,
                        separators=(",", ":"), ensure_ascii=False)
                ]),
                num_uavs=np.asarray([K], dtype=np.int16),
                num_targets=np.asarray([Q], dtype=np.int16),
                episode=np.asarray(structure_trace_episode, dtype=np.int32),
                seed=np.asarray(structure_trace_seed, dtype=np.int64),
                frame=np.asarray(structure_trace_frame, dtype=np.int32),
                p0_resolved=np.asarray(
                    structure_trace_p0_resolved, dtype=np.uint8),
                local_obs=np.asarray(
                    structure_trace_obs, dtype=np.float32),
                next_local_obs=np.asarray(
                    structure_trace_next_obs, dtype=np.float32),
                delta_p=np.asarray(
                    structure_trace_delta_p, dtype=np.float32),
                outgoing_message=np.asarray(
                    structure_trace_message, dtype=np.float32),
                outgoing_rate=np.asarray(
                    structure_trace_rate, dtype=np.int16),
                outgoing_token_mask=np.asarray(
                    structure_trace_token_mask, dtype=np.uint8),
                comm_fraction=np.asarray(
                    structure_trace_comm_fraction, dtype=np.float32),
                sensing_weights=np.asarray(
                    structure_trace_sensing_weights, dtype=np.float32),
                uav_positions=np.asarray(
                    structure_trace_uav_positions, dtype=np.float32),
                uav_velocities=np.asarray(
                    structure_trace_uav_velocities, dtype=np.float32),
                target_states=np.asarray(
                    structure_trace_target_states, dtype=np.float32),
                target_positions=np.asarray(
                    structure_trace_target_positions, dtype=np.float32),
                target_velocities=np.asarray(
                    structure_trace_target_velocities, dtype=np.float32),
                local_pd=np.asarray(
                    structure_trace_local_pd, dtype=np.float32),
                coord_pd_ema=np.asarray(
                    structure_trace_coord_pd_ema, dtype=np.float32),
                coord_pd_ema_valid=np.asarray(
                    structure_trace_coord_pd_ema_valid, dtype=np.uint8),
                teacher_pair=np.asarray(
                    structure_trace_pair, dtype=np.uint8),
                teacher_tx_target=np.asarray(
                    structure_trace_tx_target, dtype=np.uint8),
                teacher_rx_target=np.asarray(
                    structure_trace_rx_target, dtype=np.uint8),
                teacher_endpoint_target=np.asarray(
                    structure_trace_endpoint_target, dtype=np.uint8),
                teacher_receiver_owner=np.asarray(
                    structure_trace_receiver_owner, dtype=np.int16),
                teacher_role=np.asarray(
                    structure_trace_role, dtype=np.int8),
                deployed_selected=np.asarray(
                    structure_trace_pair, dtype=np.uint8),
                deployed_receiver_owner=np.asarray(
                    structure_trace_receiver_owner, dtype=np.int16),
                deployed_role=np.asarray(
                    structure_trace_role, dtype=np.int8),
                deployed_sensing_power_w=np.asarray(
                    structure_trace_sensing_power, dtype=np.float64),
                deployed_comm_power_w=np.asarray(
                    structure_trace_comm_power, dtype=np.float64),
                frame_sensing_budget_w=np.asarray(
                    structure_trace_sensing_budget, dtype=np.float64),
                per_watt_coefficient=np.asarray(
                    structure_trace_per_watt_coefficient,
                    dtype=np.float64),
                privileged_candidate=np.asarray(
                    structure_trace_candidate, dtype=np.uint8),
                privileged_d_eff=np.asarray(
                    structure_trace_d_eff, dtype=np.float64),
                privileged_d_raw=np.asarray(
                    structure_trace_d_raw, dtype=np.float64),
                privileged_alpha=np.asarray(
                    structure_trace_alpha, dtype=np.float64),
                privileged_g_dd=np.asarray(
                    structure_trace_g_dd, dtype=np.float64),
                privileged_chi_rep=np.asarray(
                    structure_trace_chi_rep, dtype=np.float64),
                physical_pd=np.asarray(
                    structure_trace_physical_pd, dtype=np.float64),
                task_mean_pd=np.asarray(
                    structure_trace_task_mean_pd, dtype=np.float64),
                task_weak3_pd=np.asarray(
                    structure_trace_task_weak3_pd, dtype=np.float64),
                task_worst_pd=np.asarray(
                    structure_trace_task_worst_pd, dtype=np.float64),
                local_search_rebootstrap_attempted=np.asarray(
                    structure_trace_rebootstrap_attempted,
                    dtype=np.uint8),
                local_search_rebootstrap_accepted=np.asarray(
                    structure_trace_rebootstrap_accepted,
                    dtype=np.uint8),
                p_fa=np.asarray(
                    [self.cfg.detection.P_FA], dtype=np.float64),
                qos_floor=np.asarray([
                    float(getattr(
                        self.cfg.marl,
                        "comm_qos_worst_min",
                        0.60,
                    ))
                ], dtype=np.float64),
                deficit_priority_gain=np.asarray([
                    float(getattr(
                        self.cfg.marl,
                        "p0_maxmin_deficit_priority_gain",
                        3.0,
                    ))
                ], dtype=np.float64),
                target_pair_limit=np.asarray(
                    [self.cfg.detection.K_q_max], dtype=np.int16),
                reports_per_receiver=np.asarray([
                    Q * self.cfg.detection.K_q_max
                ], dtype=np.int16),
                rf_total_power_cap_w=np.asarray([
                    float(eval_env.core._isac_total_power_w)
                ], dtype=np.float64),
                sensing_power_cap_w=np.asarray([
                    float(eval_env.core._sensing_power_cap_w)
                ], dtype=np.float64),
            )
        n5_counterfactual_stats: Dict[str, float] = {}
        if n5_counterfactual_output:
            trace_path = os.path.abspath(n5_counterfactual_output)
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            with open(trace_path, "w", encoding="utf-8") as handle:
                json.dump({
                    "schema_version": 1,
                    "protocol": "atomic_n5_same_state_crn_h1",
                    "target_mode": str(n5_counterfactual_target_mode),
                    "max_candidates": int(
                        n5_counterfactual_max_candidates),
                    "require_proxy_positive": bool(
                        n5_counterfactual_require_proxy_positive),
                    "events": n5_counterfactual_events,
                }, handle, indent=2)
            proxy_deltas = [
                float(event["proxy_choice"]["delta_worst"])
                for event in n5_counterfactual_events
                if event.get("proxy_choice") is not None
            ]
            proxy_oracle_deltas = [
                float(event["oracle_proxy_pool"]["delta_worst"])
                for event in n5_counterfactual_events
                if event.get("oracle_proxy_pool") is not None
            ]
            all_oracle_deltas = [
                float(event["oracle_all_target_pool"]["delta_worst"])
                for event in n5_counterfactual_events
                if event.get("oracle_all_target_pool") is not None
            ]
            n5_counterfactual_stats = {
                "eval_n5_cf_event_count": float(
                    len(n5_counterfactual_events)),
                "eval_n5_cf_proxy_choice_delta_worst": float(np.mean(
                    proxy_deltas or [0.0])),
                "eval_n5_cf_proxy_pool_oracle_delta_worst": float(np.mean(
                    proxy_oracle_deltas or [0.0])),
                "eval_n5_cf_all_target_oracle_delta_worst": float(np.mean(
                    all_oracle_deltas or [0.0])),
                "eval_n5_cf_false_accept_rate": float(np.mean([
                    float(event.get("false_accept_rate", 0.0))
                    for event in n5_counterfactual_events
                ] or [0.0])),
                "eval_n5_cf_material_sign_agreement": float(np.mean([
                    float(event.get("material_sign_agreement", 0.0))
                    for event in n5_counterfactual_events
                ] or [0.0])),
                "eval_n5_cf_max_geometry_error": float(max([
                    float(event.get("max_geometry_error", 0.0))
                    for event in n5_counterfactual_events
                ] or [0.0])),
            }
        risk_stats: Dict[str, float] = {}
        if eval_risk_quantiles:
            risk_quantile_array = np.asarray(eval_risk_quantiles)
            risk_logit_array = np.asarray(eval_risk_logits)
            risk_target_array = np.asarray(eval_risk_targets)
            risk_stats.update({
                f'eval_{key}': value for key, value in
                quantile_calibration_metrics(
                    risk_quantile_array, risk_target_array).items()
            })
            violation_label = (
                risk_target_array < self._risk_critic_qos_floor)
            risk_stats.update({
                f'eval_{key}': value for key, value in
                binary_risk_calibration_metrics(
                    risk_logit_array, violation_label).items()
            })
            tail_count = max(1, int(np.ceil(
                self.agents[0].critic.risk_cvar_alpha
                * self.agents[0].critic.risk_num_quantiles)))
            predicted_cvar = np.mean(
                np.sort(risk_quantile_array, axis=-1)[
                    ..., :tail_count],
                axis=-1)
            risk_stats['eval_risk_predicted_cvar_mean'] = float(
                np.mean(predicted_cvar))
            if (len(eval_episode_initial_cvar) >= 2
                    and np.std(eval_episode_initial_cvar) > 1e-12
                    and np.std(eval_episode_realized_worst) > 1e-12):
                cvar_centered = (
                    np.asarray(eval_episode_initial_cvar, dtype=np.float64)
                    - np.mean(eval_episode_initial_cvar))
                worst_centered = (
                    np.asarray(eval_episode_realized_worst, dtype=np.float64)
                    - np.mean(eval_episode_realized_worst))
                denominator = np.sqrt(
                    np.sum(cvar_centered ** 2)
                    * np.sum(worst_centered ** 2))
                risk_stats['eval_risk_episode_cvar_worst_corr'] = float(
                    np.sum(cvar_centered * worst_centered)
                    / max(float(denominator), 1e-12))
            else:
                risk_stats['eval_risk_episode_cvar_worst_corr'] = 0.0
        return {
            'eval_steady_P_D': float(np.mean(ep_steady_means)),
            'eval_full_P_D': float(np.mean(ep_full_means)),
            # Episode-wise: average of per-episode worst/bottom3/std
            'eval_worst_P_D': float(np.mean(ep_worst)),
            'eval_weak3_P_D': float(np.mean(ep_weak3)),
            # Bottom-2 remains an auxiliary stability diagnostic and never
            # replaces the strict per-episode minimum above.
            'eval_trimmed_worst_P_D': float(np.mean(ep_trimmed_worst)),
            'eval_episode_steady_P_D': [float(x) for x in ep_steady_means],
            'eval_episode_weak3_P_D': [float(x) for x in ep_weak3],
            'eval_episode_worst_P_D': [float(x) for x in ep_worst],
            'eval_episode_trimmed_worst_P_D': [
                float(x) for x in ep_trimmed_worst],
            'eval_target_std': float(np.mean(ep_tstd)),
            **robust_stats,
            # Per-target identity tracking
            'eval_per_target': per_target_avg.tolist(),
            'eval_episode_seeds': [
                int(seed) for seed in eval_seeds[:n_eps_completed]],
            'eval_episode_steady_values': [
                float(value) for value in ep_steady_means],
            'eval_episode_weak3_values': [
                float(value) for value in ep_weak3],
            'eval_episode_worst_values': [
                float(value) for value in ep_worst],
            'eval_episode_commitment_coverage': [
                float(value) for value in ep_commitment_coverage],
            'eval_episode_hard_commitment_coverage': [
                float(value) for value in ep_hard_commitment_coverage],
            'eval_episode_p0_target_coverage': [
                float(value) for value in ep_p0_target_coverage],
            'eval_episode_worst_nearest_distance_m': [
                float(np.max(value))
                for value in ep_nearest_target_distance],
            'eval_inter_uav_min_distance_m': float(np.min(
                eval_inter_uav_min_distance or [float('inf')])),
            'eval_inter_uav_mean_frame_min_distance_m': float(np.mean(
                eval_inter_uav_min_distance or [float('inf')])),
            'eval_inter_uav_separation_violation_rate': float(np.mean(
                eval_inter_uav_separation_violation or [0.0])),
            'eval_episode_min_inter_uav_distance_m': [
                float(value) for value in ep_min_inter_uav_distance],
            'eval_inter_uav_continuous_min_distance_m': float(np.min(
                eval_inter_uav_continuous_min_distance
                or [float('inf')])),
            'eval_inter_uav_continuous_mean_frame_min_distance_m': float(
                np.mean(
                    eval_inter_uav_continuous_min_distance
                    or [float('inf')])),
            'eval_inter_uav_continuous_separation_violation_rate': float(
                np.mean(
                    eval_inter_uav_continuous_separation_violation
                    or [0.0])),
            'eval_episode_min_inter_uav_continuous_distance_m': [
                float(value)
                for value in ep_min_inter_uav_continuous_distance],
            'eval_movement_safety_intervention_frame_rate': float(np.mean(
                eval_movement_safety_intervened or [0.0])),
            'eval_movement_safety_intervention_call_rate': float(np.mean(
                eval_movement_safety_intervention_call_rate or [0.0])),
            'eval_movement_safety_fail_closed_frame_rate': float(np.mean(
                eval_movement_safety_fail_closed or [0.0])),
            'eval_movement_safety_fail_closed_calls': float(np.sum(
                eval_movement_safety_fail_closed_calls or [0.0])),
            'eval_movement_safety_fail_closed_outside_invariant_calls': float(
                np.sum(
                    eval_movement_safety_fail_closed_outside_invariant_calls
                    or [0.0])),
            'eval_movement_safety_outside_invariant_calls': float(np.sum(
                eval_movement_safety_outside_invariant_calls or [0.0])),
            'eval_movement_safety_recovery_pair_count': float(np.sum(
                eval_movement_safety_recovery_pair_count or [0.0])),
            'eval_movement_safety_mean_solve_time_s_per_frame': float(np.mean(
                eval_movement_safety_solve_time_s or [0.0])),
            'eval_movement_safety_mean_solve_time_s_per_node': float(np.mean(
                eval_movement_safety_solve_time_s_per_node or [0.0])),
            'eval_movement_safety_mean_projection_calls_per_frame': float(
                np.mean(eval_movement_safety_projection_calls or [0.0])),
            'eval_movement_safety_reduced_linear_qp_calls': float(np.sum(
                eval_movement_safety_reduced_linear_qp_calls or [0.0])),
            'eval_movement_safety_reduced_linear_qp_rate': float(np.mean(
                eval_movement_safety_reduced_linear_qp_rate or [0.0])),
            'eval_movement_public_view_stale_fail_closed_rate': float(
                np.mean(
                    eval_movement_public_view_stale_fail_closed_rate
                    or [0.0])),
            'eval_movement_public_view_mean_max_age_frames': float(np.mean(
                eval_movement_public_view_mean_max_age or [0.0])),
            'eval_movement_public_view_max_age_frames': float(max(
                eval_movement_public_view_max_age or [0.0])),
            'eval_movement_gap_baseline_envelope_gap_selected_rate': float(
                np.mean(eval_movement_envelope_gap_selected_rate or [0.0])),
            'eval_movement_gap_baseline_envelope_single_projection_rate': float(
                np.mean(
                    eval_movement_envelope_single_projection_rate or [0.0])),
            'eval_movement_gap_baseline_envelope_selection_disagreement_rate': (
                float(np.mean(
                    eval_movement_envelope_selection_disagreement_rate
                    or [0.0]))),
            'eval_movement_gap_primal_dual_candidate_enabled': float(np.mean(
                eval_movement_primal_dual_candidate_enabled or [0.0])),
            'eval_movement_gap_primal_dual_certificate_available_rate': float(
                np.mean(
                    eval_movement_primal_dual_certificate_available_rate
                    or [0.0])),
            'eval_movement_gap_primal_dual_selected_rate': float(np.mean(
                eval_movement_primal_dual_selected_rate or [0.0])),
            'eval_movement_gap_primal_dual_strong_dominance_rate': float(
                np.mean(
                    eval_movement_primal_dual_strong_dominance_rate
                    or [0.0])),
            'eval_movement_gap_primal_dual_lower_improvement_mean': float(
                np.mean(eval_movement_primal_dual_lower_improvement
                        or [0.0])),
            'eval_movement_gap_primal_dual_certificate_width_mean': float(
                np.mean(eval_movement_primal_dual_certificate_width
                        or [0.0])),
            'eval_movement_gap_primal_dual_incumbent_lower_mean': float(
                np.mean(eval_movement_primal_dual_incumbent_lower
                        or [0.0])),
            'eval_movement_gap_primal_dual_candidate_lower_mean': float(
                np.mean(eval_movement_primal_dual_candidate_lower
                        or [0.0])),
            'eval_movement_gap_primal_dual_selection_disagreement_rate': float(
                np.mean(
                    eval_movement_primal_dual_selection_disagreement_rate
                    or [0.0])),
            'eval_movement_assignment_target_coverage': float(np.mean(
                eval_movement_assignment_target_coverage or [0.0])),
            'eval_movement_assignment_duplicate_rate': float(np.mean(
                eval_movement_assignment_duplicate_rate or [0.0])),
            'eval_movement_assignment_entry_agreement_rate': float(np.mean(
                eval_movement_assignment_entry_agreement or [0.0])),
            'eval_movement_anchor_target_coverage': float(np.mean(
                eval_movement_anchor_target_coverage or [0.0])),
            'eval_movement_gain_schedule_selected_self_rate': float(np.mean(
                eval_movement_gain_schedule_selected_self_rate or [0.0])),
            'eval_movement_gain_schedule_strategy_hold_rate': float(np.mean(
                eval_movement_gain_schedule_strategy_hold_rate or [0.0])),
            'eval_movement_gain_schedule_no_improvement_rate': float(np.mean(
                eval_movement_gain_schedule_no_improvement_rate or [0.0])),
            'eval_movement_gain_schedule_far_range_rate': float(np.mean(
                eval_movement_gain_schedule_far_range_rate or [0.0])),
            **risk_stats,
            **responsibility_stats,
            **temporal_credit_stats,
            **target_choice_stats,
            **sensing_choice_stats,
            **joint_sensing_pair_stats,
            **physical_oracle_stats,
            **evidence_oracle_stats,
            **evidence_topk_capacity_stats,
            **n5_counterfactual_stats,
            # Actor target-allocation and geometric coverage diagnostics.
            'eval_actor_move_unique_target_fraction': float(np.mean(
                actor_move_unique_target or [0.0])),
            'eval_actor_move_collision_frame_rate': float(np.mean(
                actor_move_collision or [0.0])),
            'eval_actor_move_pair_cosine': float(np.mean(
                actor_move_pair_cosine or [0.0])),
            'eval_actor_move_idle_fraction': float(np.mean(
                actor_move_idle or [0.0])),
            'eval_actor_move_hungarian_match': float(np.mean(
                actor_move_hungarian_match or [0.0])),
            'eval_actor_target_choice_matrix': actor_target_choice_matrix.tolist(),
            'eval_executed_move_unique_target_fraction': float(np.mean(
                executed_move_unique_target or [0.0])),
            'eval_executed_move_collision_frame_rate': float(np.mean(
                executed_move_collision or [0.0])),
            'eval_executed_target_choice_matrix': (
                executed_target_choice_counts
                / np.maximum(
                    executed_target_choice_counts.sum(axis=1, keepdims=True),
                    1)
            ).tolist(),
            'eval_mean_nearest_uav_distance_m': mean_nearest_distance,
            'eval_worst_nearest_uav_distance_m': worst_nearest_distance,
            'eval_per_target_nearest_uav_distance_m': (
                per_target_nearest_distance.tolist()),
            # Pairing diagnostics
            'valid_pair_rate': vp_frames / tf,
            'no_TX_rate': notx_frames / tf,
            'all_same_role_rate': samerole_frames / tf,
            'eval_multistatic_duplex_frame_rate': (
                duplex_endpoint_frames / tf),
            'eval_multistatic_mean_duplex_nodes': float(np.mean(
                duplex_endpoint_nodes or [0.0])),
            'eval_comm_bits_per_frame': float(np.mean(eval_comm_bits)),
            'eval_comm_energy_j_per_frame': float(np.mean(eval_comm_energy)),
            'eval_comm_mean_latency_s': float(np.mean(
                eval_comm_latency or [0.0])),
            'eval_comm_p95_latency_s': float(np.mean(
                eval_comm_p95_latency or [0.0])),
            'eval_comm_mean_packet_error_probability': float(np.mean(
                eval_comm_mean_bler or [0.0])),
            'eval_comm_max_packet_error_probability': float(np.max(
                eval_comm_max_bler or [0.0])),
            'eval_comm_reliability_violation_rate': float(np.mean(
                eval_comm_reliability_violation or [0.0])),
            'eval_comm_burst_failure_rate': float(np.mean(
                eval_comm_burst_failure or [0.0])),
            'eval_comm_burst_bad_link_rate': float(np.mean(
                eval_comm_burst_bad_link or [0.0])),
            'eval_comm_delivery_rate': float(np.mean(
                eval_comm_delivery or [1.0])),
            'eval_comm_active_senders': float(np.mean(eval_comm_active)),
            'eval_comm_deadline_violation_rate': float(
                np.mean(eval_comm_violation or [0.0])),
            'eval_comm_snr_violation_rate': float(np.mean(
                eval_comm_snr_violation or [0.0])),
            'eval_comm_delivery_failure_rate': float(np.mean(
                eval_comm_delivery_failure or [0.0])),
            'eval_structure_student_payload_bits_per_frame': float(np.mean(
                eval_structure_payload_bits or [0.0])),
            'eval_structure_student_atomic_delivery_rate': float(np.mean(
                eval_structure_atomic_delivery or [1.0])),
            'eval_structure_student_selected_bits_per_dim': float(np.mean(
                eval_structure_selected_bits or [
                    self._structure_student_bits_per_dim])),
            'eval_structure_student_public_valid_fraction': float(np.mean(
                eval_structure_public_valid or [0.0])),
            'eval_structure_student_public_mean_age_frames': float(np.mean(
                eval_structure_public_age or [0.0])),
            'eval_structure_student_insufficient_cache_rate': float(np.mean(
                eval_structure_insufficient_cache or [0.0])),
            'eval_token_sensing_jaccard': float(np.mean(
                eval_token_sensing_jaccard or [0.0])),
            'eval_token_sensing_exact_match_rate': float(np.mean(
                eval_token_sensing_exact or [0.0])),
            'eval_token_sensing_power_mass': float(np.mean(
                eval_token_sensing_power_mass or [0.0])),
            'eval_p0_resolve_frame_rate': float(np.mean(
                eval_p0_resolved or [0.0])),
            'eval_p0_solve_time_s_per_frame': float(np.mean(
                eval_p0_solve_time or [0.0])),
            'eval_p0_solve_time_s_per_resolve': float(
                np.sum(eval_p0_solve_time)
                / max(np.sum(eval_p0_resolved), 1.0)),
            'eval_p0_topology_invalid_frame_rate': float(np.mean(
                eval_p0_topology_invalid or [0.0])),
            'eval_p0_topology_invalid_edges_per_frame': float(np.mean(
                eval_p0_topology_invalid_edges or [0.0])),
            'eval_p0_topology_repair_frame_rate': float(np.mean(
                eval_p0_topology_repair or [0.0])),
            'eval_p0_topology_replacements_per_repair': float(
                np.sum(eval_p0_topology_replacements)
                / max(np.sum(eval_p0_topology_repair), 1.0)),
            'eval_p0_topology_repair_candidate_worst_pd': float(np.mean(
                eval_p0_topology_repair_worst_pd or [0.0])),
            'eval_p0_topology_full_fallback_rate': float(np.mean(
                eval_p0_topology_full_fallback or [0.0])),
            'eval_p0_per_target_selection_rate': (
                np.mean(np.asarray(eval_p0_target_selected_mask),
                        axis=0).tolist()
                if eval_p0_target_selected_mask
                else np.zeros(Q).tolist()),
            'eval_local_search_candidate_evaluations_per_resolve': float(
                np.mean(eval_local_search_candidates or [0.0])),
            'eval_local_search_exact_verifications_per_resolve': float(
                np.mean(eval_local_search_verifications or [0.0])),
            'eval_local_search_accepted_moves_per_resolve': float(
                np.mean(eval_local_search_accepted or [0.0])),
            'eval_local_search_cold_resolve_fraction': float(
                np.mean(eval_local_search_cold or [0.0])),
            'eval_local_search_rebootstrap_attempt_rate': float(
                np.mean(
                    eval_local_search_rebootstrap_attempted or [0.0])),
            'eval_local_search_rebootstrap_accept_rate': float(
                np.mean(
                    eval_local_search_rebootstrap_accepted or [0.0])),
            'eval_local_search_rebootstrap_accept_given_attempt': float(
                np.sum(eval_local_search_rebootstrap_accepted)
                / max(np.sum(
                    eval_local_search_rebootstrap_attempted), 1.0)),
            'eval_local_search_rebootstrap_candidates_per_resolve': float(
                np.mean(
                    eval_local_search_rebootstrap_candidates or [0.0])),
            'eval_local_search_rebootstrap_deficit_present_rate': float(
                np.mean(
                    eval_local_search_rebootstrap_deficit_present or [0.0])),
            'eval_local_search_rebootstrap_blocked_no_deficit_rate': float(
                np.mean(
                    eval_local_search_rebootstrap_blocked_no_deficit or [0.0])),
            'eval_evidence_comm_bits_per_frame': float(np.mean(
                eval_evidence_comm_bits or [0.0])),
            'eval_protocol_total_latency_mean_s': float(np.mean(
                eval_protocol_total_latency or [0.0])),
            'eval_protocol_total_p95_latency_s': float(np.mean(
                eval_protocol_total_p95_latency or [0.0])),
            'eval_protocol_serialized_bits_per_frame': float(np.mean(
                eval_protocol_serialized_bits or [0.0])),
            'eval_protocol_total_deadline_violation_rate': float(np.mean(
                eval_protocol_total_deadline_violation or [0.0])),
            'eval_evidence_comm_energy_j_per_frame': float(np.mean(
                eval_evidence_comm_energy or [0.0])),
            'eval_evidence_comm_mean_latency_s': float(np.mean(
                eval_evidence_comm_latency or [0.0])),
            'eval_evidence_comm_delivery_rate': float(np.mean(
                eval_evidence_comm_delivery or [1.0])),
            'eval_evidence_comm_active_senders': float(np.mean(
                eval_evidence_comm_active or [0.0])),
            'eval_evidence_comm_deadline_violation_rate': float(np.mean(
                eval_evidence_comm_violation or [0.0])),
            'eval_evidence_comm_delivery_failure_rate': float(np.mean(
                eval_evidence_comm_failure or [0.0])),
            'eval_evidence_comm_snr_violation_rate': float(np.mean(
                eval_evidence_comm_snr_failure or [0.0])),
            'eval_evidence_comm_reliability_violation_rate': float(np.mean(
                eval_evidence_comm_reliability_failure or [0.0])),
            'eval_evidence_comm_burst_failure_rate': float(np.mean(
                eval_evidence_comm_burst_failure or [0.0])),
            'eval_evidence_comm_burst_bad_link_rate': float(np.mean(
                eval_evidence_comm_burst_bad or [0.0])),
            'eval_evidence_utilization_frame_mean': float(np.mean(
                eval_evidence_utilization or [0.0])),
            'eval_evidence_utilization': float(
                np.sum(eval_evidence_useful_entries)
                / max(np.sum(eval_evidence_transmitted_entries), 1.0)),
            'eval_evidence_transmitted_entries_per_frame': float(np.mean(
                eval_evidence_transmitted_entries or [0.0])),
            'eval_evidence_useful_entries_per_frame': float(np.mean(
                eval_evidence_useful_entries or [0.0])),
            'eval_total_u2u_bits_per_frame': float(
                np.mean(eval_comm_bits)
                + np.mean(eval_evidence_comm_bits or [0.0])),
            'eval_evidence_detection_pfa': float(np.mean(
                eval_evidence_pfa or [self.cfg.detection.P_FA])),
            'eval_belief_feedback_fused_entries_per_frame': float(np.mean(
                eval_belief_feedback_entries or [0.0])),
            'eval_belief_feedback_receivers_per_frame': float(np.mean(
                eval_belief_feedback_receivers or [0.0])),
            'eval_belief_feedback_disagreement_before_m': float(np.mean(
                eval_belief_feedback_before or [0.0])),
            'eval_belief_feedback_disagreement_after_m': float(np.mean(
                eval_belief_feedback_after or [0.0])),
            'eval_belief_feedback_contraction_ratio': float(np.mean(
                eval_belief_feedback_contraction or [1.0])),
            'eval_belief_feedback_contraction_frame_rate': float(np.mean(
                np.asarray(eval_belief_feedback_contraction or [1.0])
                < 1.0)),
            'eval_belief_feedback_truth_rmse_before_m': float(np.mean(
                eval_belief_feedback_truth_rmse_before or [0.0])),
            'eval_belief_feedback_truth_rmse_after_m': float(np.mean(
                eval_belief_feedback_truth_rmse_after or [0.0])),
            'eval_belief_position_rmse_m': float(np.mean(
                eval_belief_position_rmse or [0.0])),
            'eval_belief_position_p95_m': float(np.mean(
                eval_belief_position_p95 or [0.0])),
            'eval_belief_position_rmse_per_target_m': (
                np.mean(np.asarray(eval_belief_position_rmse_per_target),
                        axis=0).tolist()
                if eval_belief_position_rmse_per_target
                else np.zeros(Q).tolist()),
            'eval_belief_target_aoi_mean_frames': float(np.mean(
                eval_belief_target_aoi_mean or [0.0])),
            'eval_belief_target_aoi_p95_frames': float(np.mean(
                eval_belief_target_aoi_p95 or [0.0])),
            'eval_belief_target_aoi_per_target_frames': (
                np.mean(np.asarray(eval_belief_target_aoi_per_target),
                        axis=0).tolist()
                if eval_belief_target_aoi_per_target
                else np.zeros(Q).tolist()),
            'eval_belief_position_disagreement_end_m': float(np.mean(
                eval_belief_disagreement_end or [0.0])),
            'eval_comm_rate_distribution': rate_dist.tolist(),
            'eval_comm_mean_bits_per_dim': float(np.dot(
                rate_dist, np.asarray(rate_levels, dtype=np.float64))),
            'eval_adaptive_topk_mean': float(np.mean(
                eval_adaptive_topk_values or [0.0])),
            'eval_adaptive_topk_switch_rate': float(np.mean(
                eval_adaptive_topk_switches or [0.0])),
            'eval_isac_comm_power_w_per_frame': float(np.mean(
                eval_isac_comm_power or [0.0])),
            'eval_isac_sensing_power_w_per_frame': float(np.mean(
                eval_isac_sensing_power or [0.0])),
            'eval_isac_max_power_balance_error_w': float(np.max(
                eval_isac_balance_error or [0.0])),
            'eval_isac_max_power_budget_violation_w': float(np.max(
                eval_isac_budget_violation or [0.0])),
            'eval_isac_unused_power_w_per_frame': float(np.mean(
                eval_isac_unused_power or [0.0])),
            'eval_isac_per_target_power_w': (
                np.mean(np.asarray(eval_isac_target_power), axis=0).tolist()
                if eval_isac_target_power else np.zeros(Q).tolist()),
            'eval_distributed_replicated_common_view_rate': float(np.mean(
                eval_replicated_common_view or [0.0])),
            'eval_distributed_replicated_full_view_fraction': float(np.mean(
                eval_replicated_full_view or [0.0])),
            'eval_distributed_replicated_local_coverage': float(np.mean(
                eval_replicated_local_coverage or [0.0])),
            'eval_distributed_replicated_solve_time_s_per_frame': float(
                np.mean(eval_replicated_solve_time or [0.0])),
            'eval_distributed_replicated_per_node_solve_time_s_per_frame': (
                float(np.mean(
                    eval_replicated_per_node_solve_time or [0.0]))),
            'eval_distributed_replicated_process_parallel_use_rate': float(
                np.mean(eval_replicated_parallel_used or [0.0])),
            'eval_distributed_replicated_parallel_batch_wall_s_per_frame': (
                float(np.mean(
                    eval_replicated_parallel_batch_wall or [0.0]))),
            'eval_distributed_replicated_worker_warmup_time_s': float(
                np.sum(eval_replicated_worker_warmup or [0.0])),
            'eval_distributed_replicated_history_reserve_used_fraction': (
                float(np.mean(
                    eval_replicated_history_reserve_used or [0.0]))),
            'eval_hyperedge_visible_peers_per_uav': float(np.mean(
                eval_hyperedge_visible_peers or [0.0])),
            'eval_hyperedge_mutual_edges_per_frame': float(np.mean(
                eval_hyperedge_mutual_edges or [0.0])),
            'eval_hyperedge_active_edges_per_frame': float(np.mean(
                eval_hyperedge_active_edges or [0.0])),
            'eval_hyperedge_target_coverage': float(np.mean(
                eval_hyperedge_target_coverage or [0.0])),
            'eval_hyperedge_protocol_use_rate': float(np.mean(
                eval_hyperedge_protocol_used or [0.0])),
            'eval_hyperedge_safety_fallback_rate': float(np.mean(
                eval_hyperedge_safety_fallback or [0.0])),
            'eval_hyperedge_assignment_reuse_rate': float(np.mean(
                eval_hyperedge_assignment_reused or [0.0])),
            'eval_cacsr_gate_rate': float(np.mean(
                eval_cacsr_gate_rate or [0.0])),
            'eval_cacsr_delta_abs': float(np.mean(
                eval_cacsr_delta_abs or [0.0])),
            'eval_risk_residual_gate_mean': float(np.mean(
                eval_risk_residual_gate or [0.0])),
            'eval_risk_residual_delta_abs': float(np.mean(
                eval_risk_residual_delta_abs or [0.0])),
            'eval_risk_direction_scale_abs': float(np.mean(
                eval_risk_direction_scale_abs or [0.0])),
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
              eval_interval: Optional[int] = None) -> List[Dict]:
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
            if self._rollout_learned_comm_bits:
                metrics['learned_comm_bits'] = float(
                    np.mean(self._rollout_learned_comm_bits))
                metrics['learned_comm_energy_j'] = float(
                    np.mean(self._rollout_learned_comm_energy))
                metrics['learned_comm_latency_s'] = float(
                    np.mean(self._rollout_learned_comm_latency))
                metrics['learned_comm_delivery_rate'] = float(
                    np.mean(self._rollout_learned_comm_delivery))
                metrics['learned_comm_active_senders'] = float(
                    np.mean(self._rollout_learned_comm_active))
                metrics['learned_comm_deadline_violation_rate'] = float(
                    np.mean(self._rollout_learned_comm_violation))
                metrics['learned_comm_silence_rate'] = float(
                    1.0 - np.mean(self._rollout_learned_comm_active) / max(self.K, 1))
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
                    + f" coord={metrics.get('reward_component_coord_total', 0):+.4f}"
                    + f" dup={metrics.get('reward_component_coord_duplicate_penalty', 0):.4f}"
                    + pd_str
                )

            # ── Convergence-based eval + early stopping ──
            # Audit 2026-08-17: use the eval_interval parameter (previously the
            # signature argument was silently ignored in favour of the config
            # attribute self.eval_interval).
            cadence = eval_interval if eval_interval is not None else self.eval_interval
            if self.early_stop and (ep % max(cadence, 1) == 0 or ep == n_episodes - 1):
                ev = self._evaluate(self.eval_episodes)
                score = ev['eval_steady_P_D']
                metrics.update(ev)
                if self._comm_qos_enabled:
                    qos_eval = np.array([
                        ev.get('eval_steady_P_D', 0.0),
                        ev.get('eval_weak3_P_D', 0.0),
                        ev.get('eval_worst_P_D', 0.0),
                    ], dtype=np.float64)
                    feasible = bool(np.all(qos_eval >= self._comm_qos_targets))
                    bits = float(ev.get('eval_comm_bits_per_frame', 0.0))
                    # QoS feasibility first; then worst-target quality. Weak-3
                    # and steady are only tie-breakers, and resource use is
                    # deliberately last. Never select a checkpoint merely for
                    # low traffic or a high steady mean.
                    qos_key = build_comm_qos_checkpoint_key(
                        qos_eval, self._comm_qos_targets, bits,
                        feasibility_lcb=ev.get(
                            'eval_qos_feasible_wilson_lcb'),
                        worst_lcb=ev.get('eval_worst_lcb'),
                        worst_cvar=ev.get('eval_worst_cvar'),
                        trimmed_worst=ev.get('eval_trimmed_worst_P_D'))
                    selected_ev = ev
                    if self._checkpoint_confirmation_enabled:
                        screen_pass = (
                            self._best_selection_screen_key is None
                            or qos_key > self._best_selection_screen_key)
                        improved = False
                        if screen_pass:
                            self._best_selection_screen_key = qos_key
                            confirmation_ev = self._evaluate(
                                len(self._checkpoint_confirmation_seeds),
                                eval_seeds=self._checkpoint_confirmation_seeds)
                            for key, value in confirmation_ev.items():
                                metrics[f'confirmation_{key}'] = value
                            confirmation_qos = np.array([
                                confirmation_ev.get('eval_steady_P_D', 0.0),
                                confirmation_ev.get('eval_weak3_P_D', 0.0),
                                confirmation_ev.get('eval_worst_P_D', 0.0),
                            ], dtype=np.float64)
                            confirmation_key = build_comm_qos_checkpoint_key(
                                confirmation_qos,
                                self._comm_qos_targets,
                                float(confirmation_ev.get(
                                    'eval_comm_bits_per_frame', 0.0)),
                                feasibility_lcb=confirmation_ev.get(
                                    'eval_qos_feasible_wilson_lcb'),
                                worst_lcb=confirmation_ev.get(
                                    'eval_worst_lcb'),
                                worst_cvar=confirmation_ev.get(
                                    'eval_worst_cvar'),
                                trimmed_worst=confirmation_ev.get(
                                    'eval_trimmed_worst_P_D'))
                            improved = (
                                self._best_comm_qos_key is None
                                or confirmation_key > self._best_comm_qos_key)
                            if improved:
                                self._best_comm_qos_key = confirmation_key
                                selected_ev = confirmation_ev
                    else:
                        improved = (
                            self._best_comm_qos_key is None
                            or qos_key > self._best_comm_qos_key)
                        if improved:
                            self._best_comm_qos_key = qos_key
                else:
                    improved = score > self.best_score + self.early_stop_min_delta
                    selected_ev = ev
                if improved:
                    self.best_score = selected_ev['eval_steady_P_D']
                    self._best_eval_metrics = dict(selected_ev)
                    self.best_params = self.agents[0].get_params()  # snapshot best policy
                    self.best_runtime_state = self.get_policy_runtime_state()
                    self._patience = 0
                else:
                    self._patience += 1
                worst = ev.get('eval_worst_P_D', 0)
                worst_lcb = ev.get('eval_worst_lcb', worst)
                worst_cvar = ev.get('eval_worst_cvar', worst)
                weak3 = ev.get('eval_weak3_P_D', 0)
                tstd = ev.get('eval_target_std', 0)
                full = ev.get('eval_full_P_D', 0)
                best_worst = getattr(
                    self, '_best_eval_metrics', {}).get('eval_worst_P_D', 0.0)
                print(f"  [eval] ep {ep}: steady={score:.3f} worst={worst:.3f} "
                      f"worst_lcb={worst_lcb:.3f} cvar={worst_cvar:.3f} "
                      f"weak3={weak3:.3f} tstd={tstd:.3f} full={full:.3f} "
                      f"(feasible={int(feasible) if self._comm_qos_enabled else '-'} "
                      f"best_worst={best_worst:.3f})")
                if self._patience >= self.early_stop_patience:
                    self.converged_episode = ep
                    criterion = (
                        'no QoS/worst checkpoint-key improvement'
                        if self._comm_qos_enabled
                        else f'no >{self.early_stop_min_delta} steady improvement'
                    )
                    print(f"  [early-stop] converged at ep {ep}: "
                          f"{criterion} for {self.early_stop_patience} evals. "
                          f"best_worst_P_D={best_worst:.3f}")
                    break

        # Preserve the final optimization state for explicitly labelled
        # diagnostics, while deployment continues to restore/select strictly by
        # the QoS/worst checkpoint key.
        self.last_unrestored_params = self.agents[0].get_params()

        # Restore best policy (return the converged optimum, not the last step)
        if self.best_params is not None:
            self.agents[0].set_params(self.best_params)
            # Runtime blend coefficients are part of the executed policy even
            # though they are not neural parameters. Restore them together.
            preserved_total_frames = self.total_frames
            self.restore_policy_runtime_state(self.best_runtime_state)
            self.total_frames = preserved_total_frames
            best_ev = getattr(self, '_best_eval_metrics', {})
            print(f"[train] restored best policy: "
                  f"steady={best_ev.get('eval_steady_P_D', self.best_score):.3f} "
                  f"weak3={best_ev.get('eval_weak3_P_D', 0.0):.3f} "
                  f"worst={best_ev.get('eval_worst_P_D', 0.0):.3f}"
                  + (f" (converged @ ep {self.converged_episode})" if self.converged_episode else ""))

        return all_metrics
