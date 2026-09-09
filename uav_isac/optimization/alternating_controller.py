"""Stateful scheduling for alternating constrained policy optimization.

The controller deliberately does not own a torch optimizer.  It keeps only
the mathematical state that ordinary optimizers do not know about: one dual
variable per physical condition and a cyclic schedule for small objective
blocks.  This makes it possible to checkpoint/resume the constrained method
without coupling it to PPO, Adam, or a particular policy architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch

from .constrained_pareto import (
    ConstrainedParetoStatus,
    constrained_pareto_status,
    inequality_augmented_lagrangian,
    projected_target_dual_update,
)


def team_aligned_minibatches(
    total_rows: int,
    num_agents: int,
    maximum_rows_per_batch: int,
    *,
    team_order: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Build flat row indices while keeping every team action intact.

    PPO buffers are flattened as ``(time * agent)``. Constraint gradients use
    a joint likelihood ratio and therefore cannot be computed from shuffled
    individual rows. The optional ``team_order`` is a permutation produced by
    the caller's seeded RNG; only complete teams are then packed into batches.
    """
    rows = int(total_rows)
    agents = int(num_agents)
    limit = int(maximum_rows_per_batch)
    if rows < 1 or agents < 1 or rows % agents != 0:
        raise ValueError("total rows must contain one or more complete teams")
    if limit < agents:
        raise ValueError("maximum batch rows must fit at least one complete team")
    team_count = rows // agents
    if team_order is None:
        order = torch.arange(team_count, dtype=torch.long)
    else:
        order = torch.as_tensor(team_order)
        if order.ndim != 1 or order.shape[0] != team_count:
            raise ValueError("team_order must contain every team exactly once")
        if order.dtype == torch.bool or order.is_floating_point():
            raise ValueError("team_order must use an integer dtype")
        order = order.to(dtype=torch.long)
        if not torch.equal(
            torch.sort(order).values,
            torch.arange(team_count, dtype=torch.long, device=order.device),
        ):
            raise ValueError("team_order must be a permutation of team indices")
    local_rows = torch.arange(agents, dtype=torch.long, device=order.device)
    aligned = (order[:, None] * agents + local_rows[None, :]).reshape(-1)
    teams_per_batch = max(1, limit // agents)
    row_count = teams_per_batch * agents
    return tuple(aligned[start:start + row_count]
                 for start in range(0, rows, row_count))


@dataclass(frozen=True)
class ConstraintStep:
    """Diagnostics from one projected dual update."""

    previous_multipliers: torch.Tensor
    multipliers: torch.Tensor
    residual: torch.Tensor
    positive_violation: torch.Tensor
    dual_change: torch.Tensor
    maximum_violation: float
    maximum_dual_change: float


class AlternatingConstrainedController:
    """Cycle objective blocks while enforcing vector-valued conditions.

    Objective groups contain stable names rather than loss weights.  A caller
    may use a singleton group for an ordinary primary-objective update or a
    small group with ``assign_constrained_pareto_gradients``.  Conditions are
    always handled separately through the augmented Lagrangian and projected
    dual ascent.
    """

    _STATE_VERSION = 1

    def __init__(
        self,
        objective_groups: Sequence[Sequence[str]],
        *,
        constraint_count: int,
        penalty: float,
        dual_step_size: float | None = None,
        dual_maximum: float | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        groups = tuple(tuple(str(name) for name in group) for group in objective_groups)
        if not groups or any(not group for group in groups):
            raise ValueError("objective_groups must contain non-empty groups")
        if any(not name.strip() for group in groups for name in group):
            raise ValueError("objective names must be non-empty")
        flattened = tuple(name for group in groups for name in group)
        if len(flattened) != len(set(flattened)):
            raise ValueError("an objective may belong to only one alternating group")
        count = int(constraint_count)
        if count < 1:
            raise ValueError("constraint_count must be positive")
        rho = float(penalty)
        step = rho if dual_step_size is None else float(dual_step_size)
        if not math.isfinite(rho) or rho <= 0.0:
            raise ValueError("penalty must be finite and positive")
        if not math.isfinite(step) or step < 0.0:
            raise ValueError("dual_step_size must be finite and non-negative")
        if dual_maximum is not None and (
            not math.isfinite(float(dual_maximum)) or float(dual_maximum) < 0.0
        ):
            raise ValueError("dual_maximum must be finite and non-negative")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError("controller dtype must be floating point")

        self.objective_groups = groups
        self.constraint_count = count
        self.penalty = rho
        self.dual_step_size = step
        self.dual_maximum = (
            None if dual_maximum is None else float(dual_maximum))
        self.multipliers = torch.zeros(count, dtype=dtype, device=device)
        self.group_index = 0
        self.update_count = 0
        self.last_residual = torch.zeros_like(self.multipliers)
        self.last_dual_change = torch.zeros_like(self.multipliers)

    @property
    def current_objectives(self) -> tuple[str, ...]:
        """Names of the objective block selected for the next primal step."""
        return self.objective_groups[self.group_index]

    def select_losses(
        self, losses: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """Select the active block and fail closed when a loss is missing."""
        missing = [name for name in self.current_objectives if name not in losses]
        if missing:
            raise KeyError(f"missing active objective losses: {missing}")
        selected = tuple(losses[name] for name in self.current_objectives)
        if any(loss.ndim != 0 for loss in selected):
            raise ValueError("objective losses must be scalar tensors")
        return selected

    def constraint_loss(self, residual: torch.Tensor) -> torch.Tensor:
        """Return the vector-condition augmented-Lagrangian loss."""
        checked = self._coerce_residual(residual)
        return inequality_augmented_lagrangian(
            checked, self.multipliers, penalty=self.penalty)

    def finish_step(self, residual: torch.Tensor) -> ConstraintStep:
        """Update dual variables and advance to the next objective block."""
        checked = self._coerce_residual(residual).detach()
        previous = self.multipliers.detach().clone()
        updated = projected_target_dual_update(
            previous,
            checked,
            step_size=self.dual_step_size,
            maximum=self.dual_maximum,
        )
        change = updated - previous
        self.multipliers = updated
        self.last_residual = checked.clone()
        self.last_dual_change = change.clone()
        self.update_count += 1
        self.group_index = self.update_count % len(self.objective_groups)
        violation = torch.clamp(checked, min=0.0)
        return ConstraintStep(
            previous_multipliers=previous,
            multipliers=updated.clone(),
            residual=checked.clone(),
            positive_violation=violation,
            dual_change=change,
            maximum_violation=float(violation.amax()),
            maximum_dual_change=float(change.abs().amax()),
        )

    def convergence_status(
        self,
        stationarity: float,
        *,
        consensus_residual: torch.Tensor | float = 0.0,
        stationarity_tolerance: float,
        primal_tolerance: float,
        dual_tolerance: float,
        consensus_tolerance: float = 0.0,
    ) -> ConstrainedParetoStatus:
        """Evaluate independent stopping conditions from the latest step."""
        return constrained_pareto_status(
            stationarity,
            self.last_residual,
            self.last_dual_change,
            consensus_residual,
            stationarity_tolerance=stationarity_tolerance,
            primal_tolerance=primal_tolerance,
            dual_tolerance=dual_tolerance,
            consensus_tolerance=consensus_tolerance,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return a portable checkpoint containing schedule and dual state."""
        return {
            "version": self._STATE_VERSION,
            "objective_groups": self.objective_groups,
            "constraint_count": self.constraint_count,
            "penalty": self.penalty,
            "dual_step_size": self.dual_step_size,
            "dual_maximum": self.dual_maximum,
            "multipliers": self.multipliers.detach().clone(),
            "last_residual": self.last_residual.detach().clone(),
            "last_dual_change": self.last_dual_change.detach().clone(),
            "group_index": self.group_index,
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state after validating mathematical compatibility."""
        required = {
            "version", "objective_groups", "constraint_count", "penalty",
            "dual_step_size", "dual_maximum", "multipliers",
            "last_residual", "last_dual_change", "group_index",
            "update_count",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(f"controller checkpoint is missing keys: {sorted(missing)}")
        if int(state["version"]) != self._STATE_VERSION:
            raise ValueError("unsupported controller checkpoint version")
        groups = tuple(tuple(group) for group in state["objective_groups"])
        if groups != self.objective_groups:
            raise ValueError("objective group schedule does not match checkpoint")
        if int(state["constraint_count"]) != self.constraint_count:
            raise ValueError("constraint count does not match checkpoint")
        scalars_match = (
            float(state["penalty"]) == self.penalty
            and float(state["dual_step_size"]) == self.dual_step_size
            and state["dual_maximum"] == self.dual_maximum
        )
        if not scalars_match:
            raise ValueError("constraint hyperparameters do not match checkpoint")

        multipliers = self._load_vector(state["multipliers"], "multipliers")
        if torch.any(multipliers < 0.0):
            raise ValueError("checkpoint multipliers must be non-negative")
        last_residual = self._load_vector(state["last_residual"], "last_residual")
        last_change = self._load_vector(state["last_dual_change"], "last_dual_change")
        update_count = int(state["update_count"])
        group_index = int(state["group_index"])
        if update_count < 0 or group_index != update_count % len(self.objective_groups):
            raise ValueError("checkpoint objective schedule is inconsistent")

        self.multipliers = multipliers
        self.last_residual = last_residual
        self.last_dual_change = last_change
        self.update_count = update_count
        self.group_index = group_index

    def _coerce_residual(self, residual: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(
            residual, dtype=self.multipliers.dtype,
            device=self.multipliers.device,
        )
        if value.shape != (self.constraint_count,):
            raise ValueError(
                f"constraint residual must have shape ({self.constraint_count},)")
        if torch.any(~torch.isfinite(value)):
            raise ValueError("constraint residual must be finite")
        return value

    def _load_vector(self, value: Any, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(
            value, dtype=self.multipliers.dtype,
            device=self.multipliers.device,
        )
        if tensor.shape != (self.constraint_count,) or torch.any(~torch.isfinite(tensor)):
            raise ValueError(f"checkpoint {name} is invalid")
        return tensor.detach().clone()
