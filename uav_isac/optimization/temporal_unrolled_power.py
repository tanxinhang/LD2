"""Differentiable multi-frame primal--dual power optimization.

This module turns the fixed-structure power optimizer into a finite unrolled
layer.  The actor controls the communication/sensing RF split and the primal
initialization.  Primal power and target prices are then carried across frames,
so an outer Pareto/CVaR training objective differentiates through the actual
inner optimization trajectory instead of optimizing an unrelated proxy loss.

For a horizon of H frames and L inner iterations, the computation is

    actor -> (budget_t, p_t^0) -> L projected PDHG steps -> P_D,t
             ^                    |
             |---- (p, lambda) ---|  across consecutive frames.

The row-simplex projection enforces the RF budget and structural mask at every
inner iteration.  QoS remains an inequality constraint; it is intentionally
not inserted into the Pareto objective vector.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from uav_isac.prediction.constrained_objective import (
    detection_probability_from_deflection,
)


@dataclass(frozen=True)
class TemporalUnrolledPowerResult:
    """Physical trajectory produced by a differentiable horizon solve."""

    power_w: torch.Tensor
    target_prices: torch.Tensor
    sensing_budget_w: torch.Tensor
    communication_fraction: torch.Tensor
    active_probability: torch.Tensor
    deflection: torch.Tensor
    detection_probability: torch.Tensor
    target_residual: torch.Tensor
    primal_violation: torch.Tensor
    budget_violation_w: torch.Tensor
    primal_stationarity: torch.Tensor
    dual_residual: torch.Tensor


@dataclass(frozen=True)
class TemporalPowerObjectives:
    """Independent outer objectives; no fixed weighted sum is constructed."""

    negative_worst_detection: torch.Tensor
    negative_mean_detection: torch.Tensor
    communication_load: torch.Tensor
    sensing_energy: torch.Tensor
    temporal_switching: torch.Tensor

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        return (
            self.negative_worst_detection,
            self.negative_mean_detection,
            self.communication_load,
            self.sensing_energy,
            self.temporal_switching,
        )


def _as_matching_tensor(
    value: torch.Tensor | float,
    reference: torch.Tensor,
    shape: tuple[int, ...],
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(
        value, dtype=reference.dtype, device=reference.device)
    try:
        tensor = torch.broadcast_to(tensor, shape)
    except RuntimeError as error:
        raise ValueError(f"{name} is not broadcastable to {shape}") from error
    if torch.any(~torch.isfinite(tensor)):
        raise ValueError(f"{name} must be finite")
    return tensor


def project_masked_row_power_budget_torch(
    proposed_power_w: torch.Tensor,
    row_budget_w: torch.Tensor | float,
    feasible_mask: torch.Tensor,
) -> torch.Tensor:
    """Exact differentiable row projection onto a masked power simplex.

    The active set is selected piecewise, as in any Euclidean simplex
    projection.  Within an active region, PyTorch retains the exact Jacobian;
    masked entries are identically zero and every row is hard feasible.
    Leading dimensions are allowed, with the final two dimensions interpreted
    as ``(UAV, target)``.
    """
    proposed = torch.as_tensor(proposed_power_w)
    if not proposed.is_floating_point() or proposed.ndim < 2:
        raise ValueError(
            "proposed_power_w must be a floating tensor ending in (UAV,target)")
    mask = torch.as_tensor(
        feasible_mask, dtype=torch.bool, device=proposed.device)
    if mask.shape != proposed.shape:
        raise ValueError("feasible_mask must match proposed_power_w")
    if torch.any(~torch.isfinite(proposed)):
        raise ValueError("proposed power must be finite")
    budget_shape = proposed.shape[:-1]
    budgets = _as_matching_tensor(
        row_budget_w, proposed, budget_shape, "row_budget_w")
    if torch.any(budgets < 0.0):
        raise ValueError("row budgets must be non-negative")

    target_count = proposed.shape[-1]
    flat_proposed = proposed.reshape(-1, target_count)
    flat_mask = mask.reshape(-1, target_count)
    flat_budget = budgets.reshape(-1)
    positive = torch.where(
        flat_mask, torch.clamp(flat_proposed, min=0.0),
        torch.zeros_like(flat_proposed))
    row_sum = positive.sum(dim=-1)
    needs_projection = row_sum > flat_budget

    # Masked zeros must sort after every active entry.  The gathered positive
    # values remain finite, so cumsum never sees -inf and the active-set
    # projection keeps a stable gradient on GPU.
    sortable = torch.where(
        flat_mask, positive,
        torch.full_like(positive, -torch.inf))
    _, order = torch.sort(sortable, dim=-1, descending=True)
    ordered = torch.gather(positive, -1, order)
    ordered_active = torch.gather(flat_mask, -1, order)
    active_rank = torch.cumsum(ordered_active.to(proposed.dtype), dim=-1)
    cumulative = torch.cumsum(ordered, dim=-1) - flat_budget.unsqueeze(-1)
    threshold_candidates = cumulative / active_rank.clamp_min(1.0)
    support = ordered_active & (
        ordered - threshold_candidates > 0.0)
    # When budget=0, the strict support can be empty. The clamped first index
    # still yields theta=max(x), hence the exact all-zero projection.
    rho = (support.sum(dim=-1) - 1).clamp_min(0)
    threshold = torch.gather(
        threshold_candidates, -1, rho.unsqueeze(-1)).squeeze(-1)
    equality_projection = torch.where(
        flat_mask,
        torch.clamp(positive - threshold.unsqueeze(-1), min=0.0),
        torch.zeros_like(positive),
    )
    projected = torch.where(
        needs_projection.unsqueeze(-1), equality_projection, positive)
    return projected.reshape_as(proposed)


def project_capped_row_power_budget_torch(
    proposed_power_w: torch.Tensor,
    row_budget_w: torch.Tensor | float,
    entry_upper_bound_w: torch.Tensor,
    *,
    bisection_iterations: int = 48,
) -> torch.Tensor:
    """Project onto ``0 <= p_iq <= u_iq, sum_q p_iq <= b_i``.

    The KKT solution is ``p=clip(v-theta, 0, u)``.  If clipping at zero and
    ``u`` already respects the row budget, ``theta=0``; otherwise a monotone
    scalar bisection finds the unique budget multiplier.  This is the relaxed
    structure--power coupling used when ``u_iq=b_i sum_j xbar_ijq``.
    """
    proposed = torch.as_tensor(proposed_power_w)
    upper = torch.as_tensor(
        entry_upper_bound_w, dtype=proposed.dtype, device=proposed.device)
    if (not proposed.is_floating_point() or proposed.ndim < 2
            or upper.shape != proposed.shape):
        raise ValueError(
            "proposed power and entry upper bounds must have the same "
            "floating shape ending in (UAV,target)")
    if int(bisection_iterations) < 1:
        raise ValueError("bisection_iterations must be positive")
    if torch.any(~torch.isfinite(proposed)) or torch.any(
        ~torch.isfinite(upper)
    ) or torch.any(upper < 0.0):
        raise ValueError("power proposals/bounds must be finite and non-negative")
    budgets = _as_matching_tensor(
        row_budget_w, proposed, proposed.shape[:-1], "row_budget_w")
    if torch.any(budgets < 0.0):
        raise ValueError("row budgets must be non-negative")

    positive = torch.minimum(torch.clamp(proposed, min=0.0), upper)
    needs_budget_projection = positive.sum(dim=-1) > budgets
    low = torch.amin(proposed - upper, dim=-1)
    high = torch.amax(proposed, dim=-1)
    for _ in range(int(bisection_iterations)):
        middle = 0.5 * (low + high)
        mass = torch.minimum(
            torch.clamp(proposed - middle.unsqueeze(-1), min=0.0),
            upper,
        ).sum(dim=-1)
        too_much = mass > budgets
        low = torch.where(too_much, middle, low)
        high = torch.where(too_much, high, middle)
    threshold = 0.5 * (low + high)
    equality_projection = torch.minimum(
        torch.clamp(proposed - threshold.unsqueeze(-1), min=0.0), upper)
    # Avoid a tiny positive bisection remainder on exactly zero budgets.
    equality_projection = torch.where(
        (budgets <= 0.0).unsqueeze(-1),
        torch.zeros_like(equality_projection), equality_projection)
    return torch.where(
        needs_budget_projection.unsqueeze(-1),
        equality_projection, positive)


def actor_sensing_budget(
    communication_power_logits: torch.Tensor,
    rate_logits: torch.Tensor,
    total_rf_budget_w: torch.Tensor | float,
    sensing_power_cap_w: torch.Tensor | float,
    *,
    communication_fraction_min: float = 0.0,
    communication_fraction_max: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map actor heads to expected communication share and sensing budget.

    Rate index zero denotes silence.  Marginalizing the categorical rate action
    keeps this planning layer differentiable while PPO continues to train the
    sampled execution distribution with exact log probabilities.
    """
    power_logits = torch.as_tensor(communication_power_logits)
    rates = torch.as_tensor(
        rate_logits, dtype=power_logits.dtype, device=power_logits.device)
    if (not power_logits.is_floating_point() or power_logits.ndim < 1
            or rates.shape[:-1] != power_logits.shape
            or rates.shape[-1] < 1):
        raise ValueError(
            "rate_logits must append a non-empty rate dimension to power logits")
    if torch.any(~torch.isfinite(power_logits)) or torch.any(
        ~torch.isfinite(rates)
    ):
        raise ValueError("actor RF logits must be finite")
    lo = float(communication_fraction_min)
    hi = float(communication_fraction_max)
    if (not math.isfinite(lo) or not math.isfinite(hi)
            or not 0.0 <= lo <= hi <= 1.0):
        raise ValueError("communication fraction bounds must lie in [0,1]")
    total = _as_matching_tensor(
        total_rf_budget_w, power_logits, tuple(power_logits.shape),
        "total_rf_budget_w")
    cap = _as_matching_tensor(
        sensing_power_cap_w, power_logits, tuple(power_logits.shape),
        "sensing_power_cap_w")
    if torch.any(total < 0.0) or torch.any(cap < 0.0):
        raise ValueError("RF budgets and sensing caps must be non-negative")
    active_probability = 1.0 - torch.softmax(rates, dim=-1)[..., 0]
    communication_fraction = (
        lo + (hi - lo) * torch.sigmoid(power_logits)
    ) * active_probability
    sensing_budget = torch.minimum(
        total * (1.0 - communication_fraction), cap)
    return sensing_budget, communication_fraction, active_probability


class DifferentiableTemporalPowerUnroll:
    """Finite-horizon differentiable PDHG layer with cross-frame state."""

    def __init__(
        self,
        *,
        inner_iterations: int = 8,
        warm_start_mix: float = 0.50,
        power_cost_per_watt: float = 1.0e-4,
        quadratic_regularization: float = 1.0e-3,
        actor_proximal_regularization: float = 0.0,
        step_safety: float = 0.90,
        over_relaxation: float = 1.0,
        detach_between_frames: bool = False,
    ) -> None:
        if int(inner_iterations) < 1:
            raise ValueError("inner_iterations must be positive")
        values = tuple(float(value) for value in (
            warm_start_mix, power_cost_per_watt,
            quadratic_regularization, actor_proximal_regularization,
            step_safety, over_relaxation))
        if any(not math.isfinite(value) for value in values):
            raise ValueError("unroll parameters must be finite")
        if not 0.0 <= values[0] <= 1.0:
            raise ValueError("warm_start_mix must lie in [0,1]")
        if values[1] < 0.0 or values[2] < 0.0 or values[3] < 0.0:
            raise ValueError(
                "power cost, regularization and actor proximal curvature "
                "must be non-negative")
        if not 0.0 < values[4] < 1.0:
            raise ValueError("step_safety must lie in (0,1)")
        if not 0.0 <= values[5] <= 1.0:
            raise ValueError("over_relaxation must lie in [0,1]")
        self.inner_iterations = int(inner_iterations)
        self.warm_start_mix = values[0]
        self.power_cost_per_watt = values[1]
        self.quadratic_regularization = values[2]
        self.actor_proximal_regularization = values[3]
        self.step_safety = values[4]
        self.over_relaxation = values[5]
        self.detach_between_frames = bool(detach_between_frames)

    def __call__(
        self,
        communication_power_logits: torch.Tensor,
        sensing_logits: torch.Tensor,
        rate_logits: torch.Tensor,
        effective_gain_per_watt: torch.Tensor,
        target_requirement: torch.Tensor,
        feasible_mask: torch.Tensor,
        total_rf_budget_w: torch.Tensor | float,
        sensing_power_cap_w: torch.Tensor | float,
        *,
        p_fa: float,
        communication_fraction_min: float = 0.0,
        communication_fraction_max: float = 1.0,
        structural_activation: torch.Tensor | None = None,
    ) -> TemporalUnrolledPowerResult:
        """Unroll ``inner_iterations`` at every frame of one horizon.

        Tensor shapes are ``(H,K)``, ``(H,K,Q)``, ``(H,K,R)``, and
        ``(H,Q)``.  The target constraints are evaluated after each frame's
        bounded inner solve, which makes CVaR over the horizon directly
        differentiable with respect to the actor heads.
        """
        comm = torch.as_tensor(communication_power_logits)
        sensing = torch.as_tensor(
            sensing_logits, dtype=comm.dtype, device=comm.device)
        rates = torch.as_tensor(rate_logits, dtype=comm.dtype, device=comm.device)
        gain = torch.as_tensor(
            effective_gain_per_watt, dtype=comm.dtype, device=comm.device)
        mask = torch.as_tensor(feasible_mask, dtype=torch.bool, device=comm.device)
        if not comm.is_floating_point() or comm.ndim != 2:
            raise ValueError("communication_power_logits must have shape (H,K)")
        horizon, agents = comm.shape
        if sensing.ndim != 3 or sensing.shape[:2] != (horizon, agents):
            raise ValueError("sensing_logits must have shape (H,K,Q)")
        targets = sensing.shape[-1]
        if rates.ndim != 3 or rates.shape[:2] != (horizon, agents):
            raise ValueError("rate_logits must have shape (H,K,R)")
        if gain.shape != (horizon, agents, targets) or mask.shape != gain.shape:
            raise ValueError("gain and mask must have shape (H,K,Q)")
        if torch.any(~torch.isfinite(sensing)) or torch.any(
            ~torch.isfinite(rates)
        ) or torch.any(~torch.isfinite(gain)) or torch.any(gain < 0.0):
            raise ValueError("unroll inputs must be finite and gains non-negative")
        requirement = _as_matching_tensor(
            target_requirement, comm, (horizon, targets),
            "target_requirement")
        if torch.any(requirement < 0.0):
            raise ValueError("target requirements must be non-negative")
        activation = None
        if structural_activation is not None:
            activation = torch.as_tensor(
                structural_activation, dtype=comm.dtype, device=comm.device)
            if activation.shape != gain.shape:
                raise ValueError(
                    "structural_activation must have shape (H,K,Q)")
            if torch.any(~torch.isfinite(activation)) or torch.any(
                (activation < 0.0) | (activation > 1.0)
            ):
                raise ValueError("structural_activation must lie in [0,1]")

        budgets, communication_fraction, active_probability = (
            actor_sensing_budget(
                comm, rates, total_rf_budget_w, sensing_power_cap_w,
                communication_fraction_min=communication_fraction_min,
                communication_fraction_max=communication_fraction_max,
            ))
        masked_gain = torch.where(mask, gain, torch.zeros_like(gain))
        powers = []
        prices = []
        stationarity = []
        dual_residual = []
        previous_power = None
        previous_prices = None
        for frame in range(horizon):
            actor_proposal = (
                budgets[frame].unsqueeze(-1)
                * torch.softmax(sensing[frame], dim=-1))
            entry_cap = (
                None if activation is None
                else budgets[frame].unsqueeze(-1) * activation[frame])
            actor_initial = (
                project_masked_row_power_budget_torch(
                    actor_proposal, budgets[frame], mask[frame])
                if entry_cap is None else
                project_capped_row_power_budget_torch(
                    actor_proposal, budgets[frame],
                    torch.where(mask[frame], entry_cap,
                                torch.zeros_like(entry_cap)))
            )
            if previous_power is None:
                power = actor_initial
                target_prices = torch.zeros(
                    targets, dtype=comm.dtype, device=comm.device)
            else:
                carried_power = previous_power
                carried_prices = previous_prices
                if self.detach_between_frames:
                    carried_power = carried_power.detach()
                    carried_prices = carried_prices.detach()
                carried_power = (
                    project_masked_row_power_budget_torch(
                        carried_power, budgets[frame], mask[frame])
                    if entry_cap is None else
                    project_capped_row_power_budget_torch(
                        carried_power, budgets[frame],
                        torch.where(mask[frame], entry_cap,
                                    torch.zeros_like(entry_cap)))
                )
                power = (
                    self.warm_start_mix * actor_initial
                    + (1.0 - self.warm_start_mix) * carried_power)
                target_prices = carried_prices
            extrapolated = power
            # A's target rows have disjoint support, so this is its exact norm.
            operator_norm = torch.sqrt(torch.max(torch.sum(
                masked_gain[frame].square(), dim=0)))
            scale = torch.clamp(
                torch.maximum(
                    operator_norm,
                    torch.as_tensor(
                        self.quadratic_regularization
                        + self.actor_proximal_regularization,
                        dtype=comm.dtype, device=comm.device)),
                min=1.0e-12,
            )
            primal_step = self.step_safety / scale
            dual_step = self.step_safety / scale
            final_stationarity = torch.zeros(
                (), dtype=comm.dtype, device=comm.device)
            final_dual_residual = torch.zeros_like(final_stationarity)
            for _ in range(self.inner_iterations):
                old_prices = target_prices
                contribution = masked_gain[frame] * extrapolated
                target_prices = torch.relu(
                    target_prices + dual_step * (
                        requirement[frame] - contribution.sum(dim=0)))
                old_power = power
                gradient = (
                    self.power_cost_per_watt
                    + self.quadratic_regularization * old_power
                    + self.actor_proximal_regularization * (
                        old_power - actor_initial)
                    - masked_gain[frame] * target_prices.unsqueeze(0))
                proposal = old_power - primal_step * gradient
                power = (
                    project_masked_row_power_budget_torch(
                        proposal, budgets[frame], mask[frame])
                    if entry_cap is None else
                    project_capped_row_power_budget_torch(
                        proposal, budgets[frame],
                        torch.where(mask[frame], entry_cap,
                                    torch.zeros_like(entry_cap)))
                )
                extrapolated = power + self.over_relaxation * (
                    power - old_power)
                final_stationarity = torch.amax(
                    torch.abs(power - old_power)) / primal_step
                final_dual_residual = torch.amax(
                    torch.abs(target_prices - old_prices)) / dual_step
            powers.append(power)
            prices.append(target_prices)
            stationarity.append(final_stationarity)
            dual_residual.append(final_dual_residual)
            previous_power = power
            previous_prices = target_prices

        power_trajectory = torch.stack(powers, dim=0)
        price_trajectory = torch.stack(prices, dim=0)
        deflection = torch.sum(masked_gain * power_trajectory, dim=1)
        detection = detection_probability_from_deflection(
            deflection, p_fa=float(p_fa))
        residual = requirement - deflection
        primal_violation = torch.relu(residual).amax(dim=-1)
        row_excess = torch.relu(
            power_trajectory.sum(dim=-1) - budgets)
        masked_excess = torch.where(
            mask, torch.zeros_like(power_trajectory),
            torch.abs(power_trajectory)).amax(dim=(-2, -1))
        if activation is not None:
            structural_excess = torch.relu(
                power_trajectory
                - budgets.unsqueeze(-1) * activation).amax(dim=(-2, -1))
            masked_excess = torch.maximum(masked_excess, structural_excess)
        budget_violation = torch.maximum(
            row_excess.amax(dim=-1), masked_excess)
        return TemporalUnrolledPowerResult(
            power_w=power_trajectory,
            target_prices=price_trajectory,
            sensing_budget_w=budgets,
            communication_fraction=communication_fraction,
            active_probability=active_probability,
            deflection=deflection,
            detection_probability=detection,
            target_residual=residual,
            primal_violation=primal_violation,
            budget_violation_w=budget_violation,
            primal_stationarity=torch.stack(stationarity),
            dual_residual=torch.stack(dual_residual),
        )


def temporal_power_objectives(
    result: TemporalUnrolledPowerResult,
) -> TemporalPowerObjectives:
    """Build the outer multi-objective vector from one unrolled trajectory."""
    probability = result.detection_probability
    if probability.ndim != 2 or result.power_w.ndim != 3:
        raise ValueError("temporal power result has invalid trajectory ranks")
    if result.power_w.shape[0] > 1:
        switching = torch.mean(torch.abs(
            result.power_w[1:] - result.power_w[:-1]))
    else:
        switching = torch.zeros(
            (), dtype=result.power_w.dtype, device=result.power_w.device)
    return TemporalPowerObjectives(
        negative_worst_detection=-torch.mean(torch.amin(probability, dim=-1)),
        negative_mean_detection=-torch.mean(probability),
        communication_load=torch.mean(result.communication_fraction),
        sensing_energy=torch.mean(torch.sum(result.power_w, dim=(-2, -1))),
        temporal_switching=switching,
    )
