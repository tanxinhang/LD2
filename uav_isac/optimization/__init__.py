"""Constraint-native optimization primitives used by learned policies."""

from .alternating_controller import (
    AlternatingConstrainedController,
    ConstraintStep,
    team_aligned_minibatches,
)

from .constrained_pareto import (
    AdaptiveProjectionResult,
    AssignedParetoGradient,
    ConstrainedParetoStatus,
    ParetoDirection,
    adaptive_constraint_projection,
    assign_constrained_pareto_gradients,
    collapse_repeated_team_detection,
    constrained_pareto_status,
    empirical_detection_cvar_residual,
    flattened_team_detection_cvar_residual,
    importance_weighted_detection_cvar_residual,
    inequality_augmented_lagrangian,
    joint_policy_detection_cvar_residual,
    minimum_norm_pareto_direction,
    projected_target_dual_update,
)
from .distributed_power_primal_dual import (
    BoundedPowerUpdate,
    BoundedTemporalPowerController,
    DistributedPowerStep,
    FixedStructurePowerPrimalDual,
    project_masked_row_power_budget,
)
from .temporal_unrolled_power import (
    DifferentiableTemporalPowerUnroll,
    TemporalPowerObjectives,
    TemporalUnrolledPowerResult,
    actor_sensing_budget,
    project_capped_row_power_budget_torch,
    project_masked_row_power_budget_torch,
    temporal_power_objectives,
)
from .temporal_feasible_structure import (
    TemporalFeasibleStructureResult,
    enumerate_feasible_single_role_structures,
    straight_through_structure,
    temporal_feasible_structure_mixture,
)

__all__ = [
    "AlternatingConstrainedController",
    "AdaptiveProjectionResult",
    "AssignedParetoGradient",
    "BoundedPowerUpdate",
    "BoundedTemporalPowerController",
    "ConstrainedParetoStatus",
    "ConstraintStep",
    "DistributedPowerStep",
    "DifferentiableTemporalPowerUnroll",
    "FixedStructurePowerPrimalDual",
    "team_aligned_minibatches",
    "ParetoDirection",
    "TemporalPowerObjectives",
    "TemporalUnrolledPowerResult",
    "actor_sensing_budget",
    "adaptive_constraint_projection",
    "assign_constrained_pareto_gradients",
    "collapse_repeated_team_detection",
    "constrained_pareto_status",
    "empirical_detection_cvar_residual",
    "flattened_team_detection_cvar_residual",
    "importance_weighted_detection_cvar_residual",
    "inequality_augmented_lagrangian",
    "joint_policy_detection_cvar_residual",
    "minimum_norm_pareto_direction",
    "projected_target_dual_update",
    "project_masked_row_power_budget",
    "project_masked_row_power_budget_torch",
    "project_capped_row_power_budget_torch",
    "temporal_power_objectives",
    "TemporalFeasibleStructureResult",
    "enumerate_feasible_single_role_structures",
    "straight_through_structure",
    "temporal_feasible_structure_mixture",
]
