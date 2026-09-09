"""Native constrained objective for predictive structure and power learning.

Unlike the online composable-certificate transport, this module makes the
physical constraints part of the differentiable action construction and loss.
The analytical controller remains the final execution authority.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as functional

from .certified_gnn import FactorizedPrediction


@dataclass(frozen=True)
class ConstrainedObjectiveResult:
    """Loss decomposition plus the physically projected prediction."""

    total: torch.Tensor
    detection: torch.Tensor
    qos_violation: torch.Tensor
    qos_residual: torch.Tensor
    power_imitation: torch.Tensor
    temporal_stability: torch.Tensor
    risk_calibration: torch.Tensor
    projected_power_w: torch.Tensor
    robust_deflection: torch.Tensor
    detection_probability: torch.Tensor
    budget_violation_w: torch.Tensor


def project_power_to_row_budget(
    power_logits: torch.Tensor,
    sensing_budget_w: torch.Tensor,
    feasible_mask: torch.Tensor,
) -> torch.Tensor:
    """Map arbitrary logits to non-negative row-feasible sensing power.

    Rows with no feasible target return zero. Other rows use their full budget;
    this is a construction, not a penalty or post-hoc certificate.
    """
    if power_logits.ndim != 3:
        raise ValueError("power_logits must have shape (B,K,Q)")
    if sensing_budget_w.shape != power_logits.shape[:2]:
        raise ValueError("sensing_budget_w must have shape (B,K)")
    if feasible_mask.shape != power_logits.shape:
        raise ValueError("feasible_mask must match power_logits")
    if torch.any(~torch.isfinite(power_logits)):
        raise ValueError("power_logits must be finite")
    if torch.any(~torch.isfinite(sensing_budget_w)) or torch.any(
        sensing_budget_w < 0
    ):
        raise ValueError("sensing budgets must be finite and non-negative")
    mask = feasible_mask.to(dtype=torch.bool)
    safe_logits = power_logits.masked_fill(~mask, -torch.inf)
    row_has_target = mask.any(dim=-1, keepdim=True)
    safe_logits = torch.where(
        row_has_target, safe_logits, torch.zeros_like(safe_logits))
    share = torch.softmax(safe_logits, dim=-1)
    share = torch.where(mask & row_has_target, share, torch.zeros_like(share))
    # A few float32 ulps of inward slack keep the implemented sum on the safe
    # side of the hard cap instead of relying on an audit tolerance.
    inward = 1.0 - 4.0 * torch.finfo(power_logits.dtype).eps
    return sensing_budget_w.unsqueeze(-1) * inward * share


def detection_probability_from_deflection(
    deflection: torch.Tensor,
    *,
    p_fa: float,
) -> torch.Tensor:
    """Differentiable real-Gaussian-shift detector used by the simulator."""
    probability = float(p_fa)
    if not 0.0 < probability < 1.0:
        raise ValueError("p_fa must lie strictly between zero and one")
    if torch.any(~torch.isfinite(deflection)) or torch.any(deflection < 0):
        raise ValueError("deflection must be finite and non-negative")
    normal = torch.distributions.Normal(
        torch.zeros((), device=deflection.device, dtype=deflection.dtype),
        torch.ones((), device=deflection.device, dtype=deflection.dtype),
    )
    threshold = normal.icdf(torch.as_tensor(
        1.0 - probability, device=deflection.device,
        dtype=deflection.dtype))
    return 0.5 * torch.erfc(
        # Match the simulator's declared numerical convention, which applies
        # a 1e-10 floor before sqrt (and also avoids an infinite gradient at 0).
        (threshold - torch.sqrt(torch.clamp(deflection, min=1.0e-10)))
        / (2.0 ** 0.5))


def constrained_detection_objective(
    prediction: FactorizedPrediction,
    robust_gain_per_watt: torch.Tensor,
    sensing_budget_w: torch.Tensor,
    feasible_mask: torch.Tensor,
    teacher_power_w: torch.Tensor,
    previous_power_w: torch.Tensor,
    *,
    p_fa: float,
    qos_floor: float,
    target_weight: torch.Tensor | None = None,
    qos_constraint_mask: torch.Tensor | None = None,
    detection_weight: float = 1.0,
    qos_weight: float = 4.0,
    qos_dual_weight: float = 0.0,
    power_imitation_weight: float = 0.25,
    temporal_weight: float = 0.02,
    risk_weight: float = 0.05,
    softmin_temperature: float = 0.05,
) -> ConstrainedObjectiveResult:
    """Joint detection/power/constraint/temporal objective.

    ``robust_gain_per_watt`` is a physical lower envelope available from local
    state, not a transmitted certificate. QoS is imposed on the resulting
    robust detection probability, while the row power cap is satisfied exactly
    by construction.
    """
    logits = prediction.power_logits
    expected = logits.shape
    for name, value in (
        ("robust_gain_per_watt", robust_gain_per_watt),
        ("feasible_mask", feasible_mask),
        ("teacher_power_w", teacher_power_w),
        ("previous_power_w", previous_power_w),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} must match power logits")
    if not 0.0 < float(qos_floor) < 1.0:
        raise ValueError("qos_floor must lie strictly between zero and one")
    if float(softmin_temperature) <= 0.0:
        raise ValueError("softmin_temperature must be positive")
    if torch.any(~torch.isfinite(robust_gain_per_watt)) or torch.any(
        robust_gain_per_watt < 0
    ):
        raise ValueError("robust gains must be finite and non-negative")

    power = project_power_to_row_budget(logits, sensing_budget_w, feasible_mask)
    deflection = torch.sum(robust_gain_per_watt * power, dim=1)
    probability = detection_probability_from_deflection(deflection, p_fa=p_fa)
    if target_weight is None:
        weight = torch.full_like(probability, 1.0 / probability.shape[-1])
    else:
        if target_weight.shape != probability.shape:
            raise ValueError("target_weight must have shape (B,Q)")
        weight = torch.clamp(target_weight, min=0.0)
        weight = weight / torch.clamp(weight.sum(dim=-1, keepdim=True), min=1e-12)

    temperature = float(softmin_temperature)
    # Weighted smooth minimum. Adding log(weight) preserves target priorities
    # without allowing a low-priority target to disappear from the objective.
    log_weight = torch.log(torch.clamp(weight, min=1e-12))
    smooth_worst = -temperature * torch.logsumexp(
        log_weight - probability / temperature, dim=-1)
    detection_loss = torch.mean(1.0 - smooth_worst)
    qos_shortfall = torch.relu(float(qos_floor) - probability)
    if qos_constraint_mask is None:
        constraint_weight = torch.ones_like(qos_shortfall)
    else:
        if qos_constraint_mask.shape != qos_shortfall.shape:
            raise ValueError("qos_constraint_mask must have shape (B,Q)")
        constraint_weight = qos_constraint_mask.to(qos_shortfall.dtype)
    constraint_count = torch.clamp(constraint_weight.sum(), min=1.0)
    qos_residual = torch.sum(
        constraint_weight * qos_shortfall) / constraint_count
    qos_violation = torch.sum(
        constraint_weight * qos_shortfall ** 2) / constraint_count

    scale = torch.clamp(sensing_budget_w.unsqueeze(-1), min=1e-12)
    teacher_share = teacher_power_w / scale
    predicted_share = power / scale
    power_imitation = functional.smooth_l1_loss(
        predicted_share, teacher_share)
    temporal_stability = functional.smooth_l1_loss(
        predicted_share, previous_power_w / scale)

    # The risk head learns the normalized native constraint residual, making it
    # useful for scheduling without treating it as a proof or protocol field.
    risk_target = torch.relu(float(qos_floor) - probability).detach()
    risk_calibration = functional.smooth_l1_loss(
        prediction.risk_radius, risk_target)
    budget_violation = torch.relu(
        power.sum(dim=-1) - sensing_budget_w).amax()
    total = (
        float(detection_weight) * detection_loss
        + float(qos_dual_weight) * qos_residual
        + float(qos_weight) * qos_violation
        + float(power_imitation_weight) * power_imitation
        + float(temporal_weight) * temporal_stability
        + float(risk_weight) * risk_calibration
    )
    return ConstrainedObjectiveResult(
        total=total,
        detection=detection_loss,
        qos_violation=qos_violation,
        qos_residual=qos_residual,
        power_imitation=power_imitation,
        temporal_stability=temporal_stability,
        risk_calibration=risk_calibration,
        projected_power_w=power,
        robust_deflection=deflection,
        detection_probability=probability,
        budget_violation_w=budget_violation,
    )
