"""Teacher-free distributed power allocation for a fixed structure.

The convex block solves a regularized minimum-power problem

    min_p  <c, p> + mu/2 ||p||^2
    s.t.   demand_q - sum_k gain_kq p_kq <= 0,
           p_kq = 0 outside the fixed structure,
           p_kq >= 0, sum_q p_kq <= budget_k.

It uses a projected primal--dual hybrid-gradient update.  Every UAV's primal
projection is independent; nodes exchange only their target contribution and
the target-price vector.  The operator-normalized step sizes satisfy the PDHG
stability bound without a penalty-parameter search.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class DistributedPowerStep:
    """One auditable fixed-structure primal--dual update."""

    power_w: np.ndarray
    target_prices: np.ndarray
    local_target_contribution: np.ndarray
    aggregate_target_contribution: np.ndarray
    target_residual: np.ndarray
    primal_violation: float
    power_budget_violation_w: float
    primal_stationarity: float
    dual_residual: float
    consensus_residual: float
    iteration: int
    converged: bool


@dataclass(frozen=True)
class BoundedPowerUpdate:
    """Selected result of one bounded temporal warm-start solve."""

    power_w: np.ndarray
    target_prices: np.ndarray
    baseline_primal_violation: float
    candidate_primal_violation: float
    selected_primal_violation: float
    power_budget_violation_w: float
    consensus_residual: float
    candidate_primal_stationarity: float
    candidate_dual_residual: float
    iterations: int
    warm_started: bool
    actor_proposal_used: bool
    actor_proposal_mix: float
    candidate_accepted: bool
    converged: bool
    fallback_reason: str


def project_masked_row_power_budget(
    proposed_power_w: np.ndarray,
    row_budget_w: np.ndarray | float,
    feasible_mask: np.ndarray,
) -> np.ndarray:
    """Project each UAV row onto its masked non-negative power simplex."""
    proposed = np.asarray(proposed_power_w, dtype=np.float64)
    mask = np.asarray(feasible_mask, dtype=bool)
    if proposed.ndim != 2 or mask.shape != proposed.shape:
        raise ValueError("power and feasible_mask must have equal (UAV, target) shape")
    if not np.all(np.isfinite(proposed)):
        raise ValueError("proposed power must be finite")
    budgets = np.asarray(row_budget_w, dtype=np.float64)
    if budgets.ndim == 0:
        budgets = np.full(proposed.shape[0], float(budgets))
    if (budgets.shape != (proposed.shape[0],)
            or not np.all(np.isfinite(budgets))
            or np.any(budgets < 0.0)):
        raise ValueError("row budgets must be finite, non-negative and per-UAV")

    projected = np.zeros_like(proposed)
    for row in range(proposed.shape[0]):
        active = np.flatnonzero(mask[row])
        if active.size == 0 or budgets[row] == 0.0:
            continue
        values = np.maximum(proposed[row, active], 0.0)
        if float(values.sum()) <= float(budgets[row]):
            projected[row, active] = values
            continue
        # Euclidean projection onto {x >= 0, sum(x) = budget}.
        ordered = np.sort(values)[::-1]
        cumulative = np.cumsum(ordered) - budgets[row]
        support = np.flatnonzero(
            ordered - cumulative / np.arange(1, active.size + 1) > 0.0)
        rho = int(support[-1])
        threshold = cumulative[rho] / float(rho + 1)
        projected[row, active] = np.maximum(values - threshold, 0.0)
    return projected


class FixedStructurePowerPrimalDual:
    """Distributed PDHG controller with exact local hard projections."""

    _STATE_VERSION = 1

    def __init__(
        self,
        effective_gain_per_watt: np.ndarray,
        target_requirement: np.ndarray,
        row_budget_w: np.ndarray | float,
        feasible_mask: np.ndarray,
        *,
        power_cost_per_watt: np.ndarray | float = 1.0e-3,
        quadratic_regularization: float = 1.0e-3,
        proximal_center_w: np.ndarray | None = None,
        proximal_regularization: float = 0.0,
        step_safety: float = 0.90,
        over_relaxation: float = 1.0,
        initial_power_w: np.ndarray | None = None,
        initial_target_prices: np.ndarray | None = None,
    ) -> None:
        gain = np.asarray(effective_gain_per_watt, dtype=np.float64)
        mask = np.asarray(feasible_mask, dtype=bool)
        if gain.ndim != 2 or mask.shape != gain.shape:
            raise ValueError("gain and feasible_mask must have equal (UAV, target) shape")
        if not np.all(np.isfinite(gain)) or np.any(gain < 0.0):
            raise ValueError("effective gains must be finite and non-negative")
        requirement = np.asarray(target_requirement, dtype=np.float64)
        if (requirement.shape != (gain.shape[1],)
                or not np.all(np.isfinite(requirement))
                or np.any(requirement < 0.0)):
            raise ValueError("target requirements must be finite and non-negative")
        budgets = np.asarray(row_budget_w, dtype=np.float64)
        if budgets.ndim == 0:
            budgets = np.full(gain.shape[0], float(budgets))
        if (budgets.shape != (gain.shape[0],)
                or not np.all(np.isfinite(budgets))
                or np.any(budgets < 0.0)):
            raise ValueError("row budgets must be finite and non-negative")
        cost = np.asarray(power_cost_per_watt, dtype=np.float64)
        if cost.ndim == 0:
            cost = np.full(gain.shape, float(cost))
        else:
            cost = np.broadcast_to(cost, gain.shape).copy()
        if not np.all(np.isfinite(cost)) or np.any(cost < 0.0):
            raise ValueError("power cost must be finite and non-negative")
        regularization = float(quadratic_regularization)
        proximal = float(proximal_regularization)
        safety = float(step_safety)
        relaxation = float(over_relaxation)
        if not math.isfinite(regularization) or regularization < 0.0:
            raise ValueError("quadratic regularization must be non-negative")
        if not math.isfinite(proximal) or proximal < 0.0:
            raise ValueError(
                "proximal regularization must be non-negative")
        if not math.isfinite(safety) or not 0.0 < safety < 1.0:
            raise ValueError("step_safety must lie in (0,1)")
        if not math.isfinite(relaxation) or not 0.0 <= relaxation <= 1.0:
            raise ValueError("over_relaxation must lie in [0,1]")

        self.gain = np.where(mask, gain, 0.0)
        self.requirement = requirement.copy()
        self.budgets = budgets.copy()
        self.mask = mask.copy()
        self.cost = np.where(mask, cost, 0.0)
        self.regularization = regularization
        self.proximal_regularization = proximal
        if proximal_center_w is None:
            center = np.zeros_like(self.gain)
        else:
            center = np.asarray(proximal_center_w, dtype=np.float64)
            if (center.shape != self.gain.shape
                    or not np.all(np.isfinite(center))):
                raise ValueError(
                    "proximal center has invalid shape or values")
        self.proximal_center_w = project_masked_row_power_budget(
            center, self.budgets, self.mask)
        self.step_safety = safety
        self.over_relaxation = relaxation
        # A maps flattened (k,q) power to target evidence. Its rows have
        # disjoint support, hence ||A||^2=max_q sum_k gain_kq^2 exactly.
        operator_norm = math.sqrt(float(np.max(
            np.sum(self.gain * self.gain, axis=0), initial=0.0)))
        scale = max(
            operator_norm, regularization + proximal, 1.0e-12)
        self.primal_step = safety / scale
        self.dual_step = safety / scale

        if initial_power_w is None:
            power = np.zeros_like(self.gain)
        else:
            power = np.asarray(initial_power_w, dtype=np.float64)
            if power.shape != self.gain.shape or not np.all(np.isfinite(power)):
                raise ValueError("initial power has invalid shape or values")
        self.power_w = project_masked_row_power_budget(
            power, self.budgets, self.mask)
        if initial_target_prices is None:
            prices = np.zeros(self.gain.shape[1], dtype=np.float64)
        else:
            prices = np.asarray(initial_target_prices, dtype=np.float64)
        if (prices.shape != (self.gain.shape[1],)
                or not np.all(np.isfinite(prices))
                or np.any(prices < 0.0)):
            raise ValueError("initial target prices must be finite and non-negative")
        self.target_prices = prices.copy()
        self.extrapolated_power_w = self.power_w.copy()
        self.iteration = 0
        self._convergence_streak = 0

    def step(
        self,
        *,
        primal_tolerance: float = 1.0e-6,
        stationarity_tolerance: float = 1.0e-6,
        dual_tolerance: float = 1.0e-6,
        consensus_tolerance: float = 1.0e-12,
        convergence_patience: int = 3,
    ) -> DistributedPowerStep:
        """Execute one local-primal/target-price iteration."""
        tolerances = tuple(float(value) for value in (
            primal_tolerance, stationarity_tolerance,
            dual_tolerance, consensus_tolerance))
        if any(not math.isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("convergence tolerances must be finite/non-negative")
        if int(convergence_patience) < 1:
            raise ValueError("convergence_patience must be positive")

        local_extrapolated = self.gain * self.extrapolated_power_w
        aggregate_extrapolated = local_extrapolated.sum(axis=0)
        previous_prices = self.target_prices.copy()
        prices = np.maximum(
            previous_prices
            + self.dual_step * (
                self.requirement - aggregate_extrapolated),
            0.0,
        )
        previous_power = self.power_w.copy()
        primal_gradient = (
            self.cost
            + self.regularization * previous_power
            + self.proximal_regularization * (
                previous_power - self.proximal_center_w)
            - self.gain * prices[None, :])
        proposed = previous_power - self.primal_step * primal_gradient
        power = project_masked_row_power_budget(
            proposed, self.budgets, self.mask)
        extrapolated = power + self.over_relaxation * (power - previous_power)

        local_contribution = self.gain * power
        aggregate = local_contribution.sum(axis=0)
        residual = self.requirement - aggregate
        budget_violation = float(np.max(np.maximum(
            power.sum(axis=1) - self.budgets, 0.0), initial=0.0))
        masked_violation = float(np.max(np.abs(
            power[~self.mask]), initial=0.0))
        budget_violation = max(budget_violation, masked_violation)
        stationarity = float(np.linalg.norm(
            power - previous_power, ord=np.inf)
            / max(self.primal_step, 1.0e-12))
        dual_residual = float(np.linalg.norm(
            prices - previous_prices, ord=np.inf)
            / max(self.dual_step, 1.0e-12))
        # The aggregate is reconstructed solely from exchanged target-level
        # contributions. This explicit check catches stale/mismatched packets.
        consensus_residual = float(np.linalg.norm(
            aggregate - np.sum(local_contribution, axis=0), ord=np.inf))
        primal_violation = float(np.max(np.maximum(residual, 0.0), initial=0.0))
        accepted = bool(
            primal_violation <= tolerances[0]
            and stationarity <= tolerances[1]
            and dual_residual <= tolerances[2]
            and consensus_residual <= tolerances[3]
            and budget_violation == 0.0)
        self._convergence_streak = (
            self._convergence_streak + 1 if accepted else 0)
        self.power_w = power
        self.extrapolated_power_w = extrapolated
        self.target_prices = prices
        self.iteration += 1
        return DistributedPowerStep(
            power_w=power.copy(),
            target_prices=prices.copy(),
            local_target_contribution=local_contribution.copy(),
            aggregate_target_contribution=aggregate.copy(),
            target_residual=residual.copy(),
            primal_violation=primal_violation,
            power_budget_violation_w=budget_violation,
            primal_stationarity=stationarity,
            dual_residual=dual_residual,
            consensus_residual=consensus_residual,
            iteration=self.iteration,
            converged=(self._convergence_streak >= int(convergence_patience)),
        )

    def run(self, max_iterations: int, **step_kwargs) -> DistributedPowerStep:
        """Run until all independent convergence gates pass or budget expires."""
        if int(max_iterations) < 1:
            raise ValueError("max_iterations must be positive")
        result = None
        for _ in range(int(max_iterations)):
            result = self.step(**step_kwargs)
            if result.converged:
                break
        assert result is not None
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self._STATE_VERSION,
            "shape": self.gain.shape,
            "power_w": self.power_w.copy(),
            "extrapolated_power_w": self.extrapolated_power_w.copy(),
            "target_prices": self.target_prices.copy(),
            "iteration": self.iteration,
            "convergence_streak": self._convergence_streak,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        required = {
            "version", "shape", "power_w", "extrapolated_power_w",
            "target_prices", "iteration", "convergence_streak"}
        if missing := required.difference(state):
            raise ValueError(f"power checkpoint missing keys: {sorted(missing)}")
        if (int(state["version"]) != self._STATE_VERSION
                or tuple(state["shape"]) != self.gain.shape):
            raise ValueError("power checkpoint is incompatible")
        power = np.asarray(state["power_w"], dtype=np.float64)
        extrapolated = np.asarray(
            state["extrapolated_power_w"], dtype=np.float64)
        prices = np.asarray(state["target_prices"], dtype=np.float64)
        if (power.shape != self.gain.shape
                or extrapolated.shape != self.gain.shape
                or prices.shape != self.requirement.shape
                or not np.all(np.isfinite(power))
                or not np.all(np.isfinite(extrapolated))
                or not np.all(np.isfinite(prices))
                or np.any(prices < 0.0)):
            raise ValueError("power checkpoint tensors are invalid")
        projected = project_masked_row_power_budget(
            power, self.budgets, self.mask)
        if not np.allclose(projected, power, rtol=0.0, atol=1.0e-12):
            raise ValueError("checkpoint power violates the hard feasible set")
        iteration = int(state["iteration"])
        streak = int(state["convergence_streak"])
        if iteration < 0 or streak < 0 or streak > iteration:
            raise ValueError("power checkpoint counters are invalid")
        self.power_w = power.copy()
        self.extrapolated_power_w = extrapolated.copy()
        self.target_prices = prices.copy()
        self.iteration = iteration
        self._convergence_streak = streak


class BoundedTemporalPowerController:
    """Warm-start fixed-structure power solves with fail-closed selection.

    A bounded candidate may be deployed before numerical convergence only when
    its maximum target violation does not regress relative to the projected
    previous allocation. Hard power/structure feasibility is unconditional.
    """

    _STATE_VERSION = 1

    def __init__(
        self,
        *,
        maximum_iterations_per_frame: int = 20,
        acceptance_tolerance: float = 1.0e-10,
        solver_options: Mapping[str, Any] | None = None,
        step_options: Mapping[str, Any] | None = None,
    ) -> None:
        if int(maximum_iterations_per_frame) < 1:
            raise ValueError("maximum_iterations_per_frame must be positive")
        tolerance = float(acceptance_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("acceptance_tolerance must be non-negative")
        self.maximum_iterations_per_frame = int(maximum_iterations_per_frame)
        self.acceptance_tolerance = tolerance
        self.solver_options = dict(solver_options or {})
        self.step_options = dict(step_options or {})
        self.previous_power_w: np.ndarray | None = None
        self.previous_target_prices: np.ndarray | None = None
        self.frame_count = 0
        self.accepted_count = 0
        self.fallback_count = 0

    def solve(
        self,
        effective_gain_per_watt: np.ndarray,
        target_requirement: np.ndarray,
        row_budget_w: np.ndarray | float,
        feasible_mask: np.ndarray,
        *,
        actor_proposal_power_w: np.ndarray | None = None,
        actor_proposal_mix: float = 1.0,
    ) -> BoundedPowerUpdate:
        gain = np.asarray(effective_gain_per_watt, dtype=np.float64)
        mask = np.asarray(feasible_mask, dtype=bool)
        warm = bool(
            self.previous_power_w is not None
            and self.previous_power_w.shape == gain.shape
            and self.previous_target_prices is not None
            and self.previous_target_prices.shape == (gain.shape[1],))
        previous_power = self.previous_power_w if warm else None
        initial_prices = self.previous_target_prices if warm else None
        # Project again because per-frame communication reserve/budgets may
        # change even while the discrete structure remains fixed.
        if previous_power is None:
            baseline = np.zeros_like(gain)
        else:
            baseline = project_masked_row_power_budget(
                previous_power, row_budget_w, mask)
        proposal_used = actor_proposal_power_w is not None
        proposal_mix = float(actor_proposal_mix)
        if (not math.isfinite(proposal_mix)
                or not 0.0 <= proposal_mix <= 1.0):
            raise ValueError("actor_proposal_mix must lie in [0,1]")
        if proposal_used:
            proposal = np.asarray(
                actor_proposal_power_w, dtype=np.float64)
            if proposal.shape != gain.shape or not np.all(np.isfinite(proposal)):
                raise ValueError(
                    "actor power proposal must be finite with gain shape")
            proposal = project_masked_row_power_budget(
                proposal, row_budget_w, mask)
            initial_power = (
                proposal if not warm
                else proposal_mix * proposal + (1.0 - proposal_mix) * baseline)
        else:
            initial_power = baseline
        requirement = np.asarray(target_requirement, dtype=np.float64)
        masked_gain = np.where(mask, gain, 0.0)
        baseline_residual = requirement - np.sum(
            masked_gain * baseline, axis=0)
        baseline_violation = float(np.max(
            np.maximum(baseline_residual, 0.0), initial=0.0))
        solver_options = dict(self.solver_options)
        # The actor proposal is the centre of an optional proximal term, not
        # an unconstrained override.  It is projected before entering the
        # solver, so every proximal update remains in the same physical
        # feasible set and the bounded-controller acceptance gate is intact.
        if proposal_used and float(
                solver_options.get('proximal_regularization', 0.0)) > 0.0:
            solver_options['proximal_center_w'] = proposal
        solver = FixedStructurePowerPrimalDual(
            gain, requirement, row_budget_w, mask,
            initial_power_w=initial_power,
            initial_target_prices=initial_prices,
            **solver_options,
        )
        candidate = solver.run(
            self.maximum_iterations_per_frame, **self.step_options)
        hard_safe = bool(
            candidate.power_budget_violation_w == 0.0
            and candidate.consensus_residual
                <= float(self.step_options.get(
                    "consensus_tolerance", 1.0e-12)))
        non_regressing = bool(
            candidate.primal_violation
            <= baseline_violation + self.acceptance_tolerance)
        accepted = hard_safe and non_regressing
        if accepted:
            selected_power = candidate.power_w
            selected_prices = candidate.target_prices
            selected_violation = candidate.primal_violation
            reason = ""
            self.accepted_count += 1
        else:
            selected_power = baseline
            selected_prices = (
                np.zeros(gain.shape[1], dtype=np.float64)
                if initial_prices is None else initial_prices.copy())
            selected_violation = baseline_violation
            reason = (
                "hard_constraint" if not hard_safe
                else "primal_violation_regression")
            self.fallback_count += 1
        self.previous_power_w = selected_power.copy()
        self.previous_target_prices = selected_prices.copy()
        self.frame_count += 1
        selected_budget_violation = float(np.max(np.maximum(
            selected_power.sum(axis=1)
            - np.broadcast_to(
                np.asarray(row_budget_w, dtype=np.float64),
                (gain.shape[0],)),
            0.0), initial=0.0))
        return BoundedPowerUpdate(
            power_w=selected_power.copy(),
            target_prices=selected_prices.copy(),
            baseline_primal_violation=baseline_violation,
            candidate_primal_violation=candidate.primal_violation,
            selected_primal_violation=selected_violation,
            power_budget_violation_w=selected_budget_violation,
            consensus_residual=(candidate.consensus_residual
                                if accepted else 0.0),
            candidate_primal_stationarity=float(
                candidate.primal_stationarity),
            candidate_dual_residual=float(candidate.dual_residual),
            iterations=candidate.iteration,
            warm_started=warm,
            actor_proposal_used=proposal_used,
            actor_proposal_mix=(proposal_mix if proposal_used else 0.0),
            candidate_accepted=accepted,
            converged=(candidate.converged if accepted else False),
            fallback_reason=reason,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self._STATE_VERSION,
            "maximum_iterations_per_frame": self.maximum_iterations_per_frame,
            "acceptance_tolerance": self.acceptance_tolerance,
            "previous_power_w": (
                None if self.previous_power_w is None
                else self.previous_power_w.copy()),
            "previous_target_prices": (
                None if self.previous_target_prices is None
                else self.previous_target_prices.copy()),
            "frame_count": self.frame_count,
            "accepted_count": self.accepted_count,
            "fallback_count": self.fallback_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        required = {
            "version", "maximum_iterations_per_frame",
            "acceptance_tolerance", "previous_power_w",
            "previous_target_prices", "frame_count", "accepted_count",
            "fallback_count"}
        if missing := required.difference(state):
            raise ValueError(f"temporal power checkpoint missing: {sorted(missing)}")
        if (int(state["version"]) != self._STATE_VERSION
                or int(state["maximum_iterations_per_frame"])
                    != self.maximum_iterations_per_frame
                or float(state["acceptance_tolerance"])
                    != self.acceptance_tolerance):
            raise ValueError("temporal power checkpoint is incompatible")
        power = state["previous_power_w"]
        prices = state["previous_target_prices"]
        if (power is None) != (prices is None):
            raise ValueError("temporal power checkpoint warm state is incomplete")
        if power is not None:
            power = np.asarray(power, dtype=np.float64)
            prices = np.asarray(prices, dtype=np.float64)
            if (power.ndim != 2 or prices.shape != (power.shape[1],)
                    or not np.all(np.isfinite(power))
                    or np.any(power < 0.0)
                    or not np.all(np.isfinite(prices))
                    or np.any(prices < 0.0)):
                raise ValueError("temporal power checkpoint tensors are invalid")
        counters = tuple(int(state[name]) for name in (
            "frame_count", "accepted_count", "fallback_count"))
        if (any(value < 0 for value in counters)
                or counters[1] + counters[2] != counters[0]):
            raise ValueError("temporal power checkpoint counters are invalid")
        self.previous_power_w = None if power is None else power.copy()
        self.previous_target_prices = None if prices is None else prices.copy()
        self.frame_count, self.accepted_count, self.fallback_count = counters
