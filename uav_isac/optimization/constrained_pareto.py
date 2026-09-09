"""Pareto and distributional constraint primitives without teacher labels.

The policy may expose one gradient row per physical objective.  A small convex
simplex problem chooses their minimum-norm combination, avoiding fixed loss
weights.  QoS remains a target-wise distributional constraint and is updated
by projected dual ascent rather than folded into the objective vector.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True)
class ParetoDirection:
    """Solution of the minimum-norm convex gradient subproblem."""

    weights: torch.Tensor
    gradient: torch.Tensor
    gradient_norm: torch.Tensor
    frank_wolfe_gap: torch.Tensor
    iterations: int
    converged: bool


@dataclass(frozen=True)
class ConstrainedParetoStatus:
    """Independent convergence gates for a constrained Pareto update."""

    pareto_stationarity: float
    primal_violation: float
    dual_residual: float
    consensus_residual: float
    converged: bool


@dataclass(frozen=True)
class AssignedParetoGradient:
    """Diagnostics after assigning Pareto plus constraint gradients."""

    pareto: ParetoDirection
    constraint_gradient_norm: torch.Tensor
    applied_gradient_norm: torch.Tensor
    # Unnormalized objective rows are retained only as diagnostics.  The
    # Pareto QP still uses the explicitly normalized rows below; exposing both
    # makes it possible to distinguish a physically weak objective from a
    # merely incompatible direction.
    objective_gradient_norms: torch.Tensor | None = None
    objective_gradient_cosine: torch.Tensor | None = None
    equality_constraint_gradient_norm: torch.Tensor | None = None
    tangent_gradient_norm: torch.Tensor | None = None
    projection_mode: str = "identity"
    projection_rank: int = 0
    projection_norm_ratio: torch.Tensor | None = None
    update_mode: str = "pareto_plus_constraint"


@dataclass(frozen=True)
class AdaptiveProjectionResult:
    """Rank-aware projection selected by the active constraint adapter."""

    gradients: torch.Tensor
    mode: str
    rank: int
    norm_ratio: torch.Tensor


def adaptive_constraint_projection(
    objective_gradients: torch.Tensor,
    equality_gradients: torch.Tensor,
    *,
    rank_tolerance: float = 1.0e-8,
) -> AdaptiveProjectionResult:
    """Project objective rows onto one or more active equality tangents.

    The adapter switches deterministically between identity, single and
    multi-constraint modes.  For multiple constraints, an SVD extracts an
    orthonormal basis of the independent Jacobian row space before projection;
    dependent or nearly dependent constraints therefore cannot amplify
    numerical noise.  This is a geometric feasibility operation, not a task
    weight or a learned gate.
    """
    gradients = torch.as_tensor(objective_gradients)
    equalities = torch.as_tensor(
        equality_gradients, dtype=gradients.dtype, device=gradients.device)
    if gradients.ndim != 2 or equalities.ndim != 2:
        raise ValueError("gradient matrices must be rank two")
    if gradients.shape[1] != equalities.shape[1]:
        raise ValueError("objective/equality gradients must share parameters")
    if not gradients.is_floating_point() or not equalities.is_floating_point():
        raise ValueError("gradient matrices must be floating point")
    if torch.any(~torch.isfinite(gradients)) or torch.any(
            ~torch.isfinite(equalities)):
        raise ValueError("gradient matrices must be finite")
    tolerance = float(rank_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("rank_tolerance must be finite and non-negative")
    original_norm = torch.linalg.vector_norm(gradients)
    if equalities.shape[0] == 0:
        return AdaptiveProjectionResult(
            gradients=gradients,
            mode="identity",
            rank=0,
            norm_ratio=torch.ones((), dtype=gradients.dtype,
                                  device=gradients.device),
        )
    _, singular_values, vh = torch.linalg.svd(
        equalities, full_matrices=False)
    if singular_values.numel() == 0:
        rank = 0
    else:
        threshold = tolerance * torch.clamp(singular_values[0], min=1.0)
        rank = int(torch.count_nonzero(singular_values > threshold).item())
    if rank == 0:
        projected = gradients
        mode = "identity"
    else:
        # Rows of Vh are orthonormal directions spanning the active normal
        # space, so this is the exact Euclidean tangent projection.
        basis = vh[:rank]
        projected = gradients - (gradients @ basis.transpose(0, 1)) @ basis
        mode = "single" if equalities.shape[0] == 1 else "multi"
    projected_norm = torch.linalg.vector_norm(projected)
    norm_ratio = projected_norm / torch.clamp(
        original_norm, min=torch.finfo(gradients.dtype).eps)
    return AdaptiveProjectionResult(
        gradients=projected,
        mode=mode,
        rank=rank,
        norm_ratio=norm_ratio.detach(),
    )


def minimum_norm_pareto_direction(
    task_gradients: torch.Tensor,
    *,
    tolerance: float = 1.0e-6,
    max_iterations: int = 128,
) -> ParetoDirection:
    """Solve ``min ||sum_i alpha_i g_i||`` over the probability simplex.

    For a small objective vector, every active face of the simplex is
    enumerated and its equality-constrained KKT system is solved.  This is an
    exact tiny convex-QP solve (up to floating-point tolerance), rather than a
    long hyperparameter-sensitive search.  Larger objective vectors fall back
    to deterministic Frank--Wolfe with an exact segment line search.  The
    returned coefficients are detached optimization variables; gradients must
    not flow through the Pareto subproblem itself.
    """
    gradients = torch.as_tensor(task_gradients)
    if gradients.ndim != 2 or gradients.shape[0] < 1:
        raise ValueError("task_gradients must have shape (objectives, parameters)")
    if not gradients.is_floating_point() or torch.any(~torch.isfinite(gradients)):
        raise ValueError("task gradients must be finite floating-point values")
    if float(tolerance) < 0.0 or int(max_iterations) < 1:
        raise ValueError("invalid Pareto solver tolerance/iteration limit")

    detached = gradients.detach()
    objectives = detached.shape[0]
    gram = detached @ detached.transpose(0, 1)
    face_count = (1 << objectives) - 1
    # Multi-task policies normally expose fewer than ten objectives.  In that
    # regime the complete active-set search is tiny (31 faces for five tasks)
    # and gives a materially stronger Pareto-stationarity certificate than a
    # fixed number of first-order QP iterations.
    if objectives <= 12 and face_count <= int(max_iterations):
        gram_numpy = gram.detach().to(
            device='cpu', dtype=torch.float64).numpy()
        best_weights_numpy = None
        best_value = None
        evaluated = 0
        feasibility_tolerance = max(float(tolerance), 1.0e-7)
        for bits in range(1, face_count + 1):
            active = [index for index in range(objectives)
                      if bits & (1 << index)]
            indices_numpy = np.asarray(active, dtype=np.int64)
            face_gram = gram_numpy[np.ix_(indices_numpy, indices_numpy)]
            count = len(active)
            # [G 1; 1' 0] [alpha; nu] = [0; 1].  lstsq also handles
            # collinear/opposing task gradients, where G is singular and the
            # minimum-norm solution lies on a flat face.
            kkt = np.zeros((count + 1, count + 1), dtype=np.float64)
            kkt[:count, :count] = face_gram
            kkt[:count, count] = 1.0
            kkt[count, :count] = 1.0
            rhs = np.zeros(count + 1, dtype=np.float64)
            rhs[count] = 1.0
            solution = np.linalg.lstsq(kkt, rhs, rcond=None)[0][:count]
            evaluated += 1
            if (np.any(~np.isfinite(solution))
                    or float(np.min(solution)) < -feasibility_tolerance
                    or abs(float(np.sum(solution)) - 1.0)
                    > 10.0 * feasibility_tolerance):
                continue
            face_weights = np.maximum(solution, 0.0)
            face_weights /= max(float(np.sum(face_weights)),
                                np.finfo(np.float64).eps)
            candidate = np.zeros(objectives, dtype=np.float64)
            candidate[indices_numpy] = face_weights
            value = float(candidate @ gram_numpy @ candidate)
            if best_value is None or value < best_value:
                best_value = value
                best_weights_numpy = candidate
        if best_weights_numpy is not None:
            best_weights = torch.as_tensor(
                best_weights_numpy, dtype=detached.dtype,
                device=detached.device)
            combined = best_weights @ detached
            objective_gradient = gram @ best_weights
            gap = torch.clamp(
                torch.dot(best_weights, objective_gradient)
                - torch.min(objective_gradient),
                min=0.0,
            )
            norm = torch.linalg.vector_norm(combined)
            return ParetoDirection(
                weights=best_weights,
                gradient=combined,
                gradient_norm=norm,
                frank_wolfe_gap=gap,
                iterations=evaluated,
                converged=bool(float(gap) <= float(tolerance)),
            )

    weights = torch.full(
        (objectives,), 1.0 / objectives,
        dtype=detached.dtype, device=detached.device,
    )
    gap = torch.full((), float("inf"), dtype=detached.dtype,
                     device=detached.device)
    converged = False
    iterations = 0
    eps = torch.finfo(detached.dtype).eps
    for iteration in range(1, int(max_iterations) + 1):
        objective_gradient = gram @ weights
        vertex_index = int(torch.argmin(objective_gradient).item())
        vertex = torch.zeros_like(weights)
        vertex[vertex_index] = 1.0
        direction = vertex - weights
        gap = torch.dot(weights - vertex, objective_gradient)
        iterations = iteration
        if float(gap) <= float(tolerance):
            converged = True
            break
        denominator = torch.dot(direction, gram @ direction)
        if float(denominator) <= float(eps):
            converged = True
            break
        step = torch.clamp(
            -torch.dot(direction, objective_gradient) / denominator,
            min=0.0, max=1.0,
        )
        weights = weights + step * direction

    # Clamp only solver-scale residue, then restore the simplex exactly.
    weights = torch.clamp(weights, min=0.0)
    weights = weights / torch.clamp(weights.sum(), min=eps)
    combined = weights @ detached
    norm = torch.linalg.vector_norm(combined)
    return ParetoDirection(
        weights=weights,
        gradient=combined,
        gradient_norm=norm,
        frank_wolfe_gap=torch.clamp(gap, min=0.0),
        iterations=iterations,
        converged=converged,
    )


def assign_constrained_pareto_gradients(
    objective_losses: list[torch.Tensor] | tuple[torch.Tensor, ...],
    parameters,
    *,
    constraint_loss: torch.Tensor | None = None,
    tangent_constraint_losses: list[torch.Tensor]
    | tuple[torch.Tensor, ...] = (),
    normalize_objective_gradients: bool = True,
    tangent_projection_rank_tolerance: float = 1.0e-8,
    feasibility_first: bool = False,
    batched_vjp: bool = True,
    tolerance: float = 1.0e-6,
    max_iterations: int = 128,
) -> AssignedParetoGradient:
    """Assign a Pareto actor gradient plus an independent constraint gradient.

    Objective rows are L2-normalized by default because detection probability,
    communication bits and latency live in different physical units. Constraint
    gradients are never included in the Pareto weight solve: they enter through
    their projected dual variables and therefore cannot be traded away by an
    objective coefficient.  An opt-in feasibility-first restoration step uses
    the normalized constraint gradient alone.  It is intended only while a
    physical inequality is violated, and removes arbitrary penalty-scale
    dependence from the restoration direction.
    """
    losses = tuple(objective_losses)
    tangent_losses = tuple(tangent_constraint_losses)
    trainable = tuple(
        parameter for parameter in parameters if parameter.requires_grad)
    if not losses or not trainable:
        raise ValueError("at least one objective and trainable parameter are required")
    for loss in losses:
        if loss.ndim != 0 or not bool(torch.isfinite(loss.detach())):
            raise ValueError("each objective loss must be a finite scalar")
    if constraint_loss is not None and (
        constraint_loss.ndim != 0
        or not bool(torch.isfinite(constraint_loss.detach()))
    ):
        raise ValueError("constraint loss must be a finite scalar")
    if feasibility_first and constraint_loss is None:
        raise ValueError("feasibility-first mode requires a constraint loss")
    if (not math.isfinite(float(tangent_projection_rank_tolerance))
            or float(tangent_projection_rank_tolerance) < 0.0):
        raise ValueError(
            "tangent_projection_rank_tolerance must be finite/non-negative")
    for loss in tangent_losses:
        if loss.ndim != 0 or not bool(torch.isfinite(loss.detach())):
            raise ValueError(
                "each tangent constraint must be a finite scalar")
    retain_for_constraints = constraint_loss is not None or bool(tangent_losses)

    # Each objective used to trigger an independent reverse traversal.  The
    # objective vector is small, so PyTorch's batched VJP computes exactly the
    # same Jacobian rows in one traversal.  A guarded fallback keeps custom
    # autograd operators and older PyTorch versions correct: this optimization
    # must never change the update semantics.
    matrix = None
    if bool(batched_vjp) and len(losses) > 1:
        try:
            objective_vector = torch.stack(losses)
            basis = torch.eye(
                len(losses), dtype=objective_vector.dtype,
                device=objective_vector.device,
            )
            batched_values = torch.autograd.grad(
                objective_vector,
                trainable,
                grad_outputs=basis,
                retain_graph=retain_for_constraints,
                allow_unused=True,
                is_grads_batched=True,
            )
            rows = []
            for parameter, value in zip(trainable, batched_values):
                if value is None:
                    rows.append(torch.zeros(
                        (len(losses), parameter.numel()),
                        dtype=parameter.dtype, device=parameter.device,
                    ))
                else:
                    if value.ndim != parameter.ndim + 1:
                        raise RuntimeError(
                            "batched VJP returned an invalid parameter shape")
                    rows.append(value.reshape(len(losses), -1))
            matrix = torch.cat(rows, dim=1)
        except (RuntimeError, NotImplementedError):
            # Some custom operators do not implement vmap batching.  The
            # scalar reverse-mode path is the reference implementation.
            matrix = None
    if matrix is None:
        gradient_rows = []
        for index, loss in enumerate(losses):
            retain = index < len(losses) - 1 or retain_for_constraints
            values = torch.autograd.grad(
                loss, trainable, retain_graph=retain, allow_unused=True)
            gradient_rows.append(torch.cat([
                (torch.zeros_like(parameter) if value is None else value).reshape(-1)
                for parameter, value in zip(trainable, values)
            ]))
        matrix = torch.stack(gradient_rows)
    raw_norms = torch.linalg.vector_norm(matrix, dim=1).detach()
    safe_norms = torch.clamp(raw_norms, min=torch.finfo(matrix.dtype).eps)
    cosine = (
        (matrix.detach() @ matrix.detach().transpose(0, 1))
        / (safe_norms[:, None] * safe_norms[None, :])
    ).detach()
    if normalize_objective_gradients:
        scale = torch.linalg.vector_norm(matrix, dim=1, keepdim=True)
        matrix = matrix / torch.clamp(
            scale, min=torch.finfo(matrix.dtype).eps)
    equality_matrix = None
    projection_mode = "identity"
    projection_rank = 0
    projection_norm_ratio = torch.ones(
        (), dtype=matrix.dtype, device=matrix.device)
    if tangent_losses:
        equality_rows = []
        for index, loss in enumerate(tangent_losses):
            retain = index < len(tangent_losses) - 1 or constraint_loss is not None
            values = torch.autograd.grad(
                loss, trainable, retain_graph=retain, allow_unused=True)
            equality_rows.append(torch.cat([
                (torch.zeros_like(parameter) if value is None else value).reshape(-1)
                for parameter, value in zip(trainable, values)
            ]))
        equality_matrix = torch.stack(equality_rows)
        # The adapter is detached from the learning graph: the projection is
        # part of the constrained Pareto subproblem, not a trainable objective.
        projection = adaptive_constraint_projection(
            matrix,
            equality_matrix.detach(),
            rank_tolerance=tangent_projection_rank_tolerance,
        )
        matrix = projection.gradients
        projection_mode = projection.mode
        projection_rank = projection.rank
        projection_norm_ratio = projection.norm_ratio
    equality_norm = (
        torch.linalg.vector_norm(equality_matrix.detach())
        if equality_matrix is not None else None)
    tangent_norm = torch.linalg.vector_norm(matrix, dim=1).detach()
    pareto = minimum_norm_pareto_direction(
        matrix, tolerance=tolerance, max_iterations=max_iterations)

    if constraint_loss is None:
        constraint_flat = torch.zeros_like(pareto.gradient)
    else:
        constraint_values = torch.autograd.grad(
            constraint_loss, trainable, allow_unused=True)
        constraint_flat = torch.cat([
            (torch.zeros_like(parameter) if value is None else value).reshape(-1)
            for parameter, value in zip(trainable, constraint_values)
        ]).detach()
    constraint_norm = torch.linalg.vector_norm(constraint_flat)
    if feasibility_first:
        if float(constraint_norm) <= float(torch.finfo(constraint_flat.dtype).eps):
            raise ValueError(
                "feasibility-first mode requires a non-zero constraint gradient")
        applied = constraint_flat / constraint_norm
        update_mode = "feasibility_first"
    else:
        applied = pareto.gradient + constraint_flat
        update_mode = "pareto_plus_constraint"
    offset = 0
    for parameter in trainable:
        count = parameter.numel()
        parameter.grad = applied[offset:offset + count].reshape_as(
            parameter).detach().clone()
        offset += count
    return AssignedParetoGradient(
        pareto=pareto,
        constraint_gradient_norm=constraint_norm,
        applied_gradient_norm=torch.linalg.vector_norm(applied),
        objective_gradient_norms=raw_norms,
        objective_gradient_cosine=cosine,
        equality_constraint_gradient_norm=equality_norm,
        tangent_gradient_norm=tangent_norm,
        projection_mode=projection_mode,
        projection_rank=projection_rank,
        projection_norm_ratio=projection_norm_ratio,
        update_mode=update_mode,
    )


def empirical_detection_cvar_residual(
    detection_probability: torch.Tensor,
    qos_floor: float | torch.Tensor,
    *,
    tail_fraction: float = 0.20,
    sample_dim: int = 0,
) -> torch.Tensor:
    """Return target-wise upper-tail CVaR of ``floor - P_D``.

    Negative values retain a physical safety margin; positive values violate
    the distributional QoS constraint.  Sorting is piecewise differentiable
    with respect to the selected worst samples and introduces no teacher data.
    """
    probability = torch.as_tensor(detection_probability)
    if not probability.is_floating_point() or probability.ndim < 1:
        raise ValueError("detection probability must be a floating tensor")
    if torch.any(~torch.isfinite(probability)) or torch.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError("detection probabilities must be finite and in [0,1]")
    fraction = float(tail_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("tail_fraction must lie in (0,1]")
    dimension = int(sample_dim) % probability.ndim
    samples = probability.shape[dimension]
    if samples < 1:
        raise ValueError("CVaR sample dimension cannot be empty")
    floor = torch.as_tensor(
        qos_floor, dtype=probability.dtype, device=probability.device)
    if torch.any(~torch.isfinite(floor)) or torch.any(
        (floor <= 0.0) | (floor >= 1.0)
    ):
        raise ValueError("QoS floor must be finite and lie in (0,1)")
    shortfall = floor - probability
    tail_count = max(1, int(math.ceil(fraction * samples)))
    worst = torch.topk(
        shortfall, k=tail_count, dim=dimension, largest=True,
        sorted=False).values
    return worst.mean(dim=dimension)


def importance_weighted_detection_cvar_residual(
    detection_probability: torch.Tensor,
    qos_floor: float | torch.Tensor,
    log_probability_ratio: torch.Tensor,
    *,
    tail_fraction: float = 0.20,
    sample_dim: int = 0,
    maximum_log_ratio: float = 20.0,
) -> torch.Tensor:
    """Return a teacher-free PPO surrogate for target-wise CVaR conditions.

    Physical detection outcomes are sampled under the rollout policy and are
    therefore not differentiable through the actor.  The selected empirical
    lower-detection tail is held fixed within one PPO update, while
    ``exp(new_log_prob - old_log_prob)`` transports its shortfall to the new
    policy.  At ratio one this equals :func:`empirical_detection_cvar_residual`.

    ``log_probability_ratio`` has one value per sample and may also include
    singleton target dimensions.  Clamping is a numerical guard, not a loss
    weight; PPO clipping and KL rollback remain the policy trust region.
    """
    probability = torch.as_tensor(detection_probability)
    if not probability.is_floating_point() or probability.ndim < 1:
        raise ValueError("detection probability must be a floating tensor")
    if torch.any(~torch.isfinite(probability)) or torch.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError("detection probabilities must be finite and in [0,1]")
    fraction = float(tail_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("tail_fraction must lie in (0,1]")
    limit = float(maximum_log_ratio)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("maximum_log_ratio must be finite and positive")
    dimension = int(sample_dim) % probability.ndim
    samples = probability.shape[dimension]
    if samples < 1:
        raise ValueError("CVaR sample dimension cannot be empty")
    floor = torch.as_tensor(
        qos_floor, dtype=probability.dtype, device=probability.device)
    if torch.any(~torch.isfinite(floor)) or torch.any(
        (floor <= 0.0) | (floor >= 1.0)
    ):
        raise ValueError("QoS floor must be finite and lie in (0,1)")

    log_ratio = torch.as_tensor(
        log_probability_ratio,
        dtype=probability.dtype,
        device=probability.device,
    )
    if torch.any(~torch.isfinite(log_ratio)):
        raise ValueError("log probability ratios must be finite")
    # Common PPO storage uses one joint-action ratio per sample. Insert target
    # singleton axes so it broadcasts over all physical conditions.
    if log_ratio.ndim == 1 and log_ratio.shape[0] == samples:
        ratio_shape = [1] * probability.ndim
        ratio_shape[dimension] = samples
        log_ratio = log_ratio.reshape(ratio_shape)
    try:
        log_ratio = torch.broadcast_to(log_ratio, probability.shape)
    except RuntimeError as exc:
        raise ValueError(
            "log probability ratio is not broadcastable to detection samples"
        ) from exc

    shortfall = floor - probability
    tail_count = max(1, int(math.ceil(fraction * samples)))
    # Tail membership is rollout data, not a trainable discrete decision.
    tail_indices = torch.topk(
        shortfall.detach(),
        k=tail_count,
        dim=dimension,
        largest=True,
        sorted=False,
    ).indices
    tail_shortfall = torch.gather(shortfall, dimension, tail_indices)
    tail_log_ratio = torch.gather(log_ratio, dimension, tail_indices)
    tail_ratio = torch.exp(torch.clamp(tail_log_ratio, -limit, limit))
    return (tail_ratio * tail_shortfall).mean(dim=dimension)


def joint_policy_detection_cvar_residual(
    detection_probability: torch.Tensor,
    qos_floor: float | torch.Tensor,
    new_agent_log_probability: torch.Tensor,
    old_agent_log_probability: torch.Tensor,
    *,
    tail_fraction: float = 0.20,
    maximum_log_ratio: float = 20.0,
) -> torch.Tensor:
    """CVaR surrogate for a factorized multi-agent joint policy.

    Inputs preserve rollout structure: ``P_D`` has shape ``(time, target)``
    and action log probabilities have shape ``(time, agent)``.  Because the
    physical outcome belongs to the complete team action, the correct
    importance log-ratio is the sum over agents.  Passing flattened or
    duplicated target outcomes is rejected instead of silently over-counting
    a transition.
    """
    probability = torch.as_tensor(detection_probability)
    new_log_probability = torch.as_tensor(new_agent_log_probability)
    old_log_probability = torch.as_tensor(
        old_agent_log_probability,
        dtype=new_log_probability.dtype,
        device=new_log_probability.device,
    )
    if probability.ndim != 2:
        raise ValueError("detection probability must have shape (time, target)")
    if new_log_probability.ndim != 2:
        raise ValueError("agent log probabilities must have shape (time, agent)")
    if new_log_probability.shape != old_log_probability.shape:
        raise ValueError("new and old agent log probabilities must align")
    if probability.shape[0] != new_log_probability.shape[0]:
        raise ValueError("physical outcomes and joint actions must align in time")
    if torch.any(~torch.isfinite(new_log_probability)) or torch.any(
        ~torch.isfinite(old_log_probability)
    ):
        raise ValueError("agent log probabilities must be finite")
    joint_log_ratio = (new_log_probability - old_log_probability).sum(dim=1)
    return importance_weighted_detection_cvar_residual(
        probability,
        qos_floor,
        joint_log_ratio,
        tail_fraction=tail_fraction,
        sample_dim=0,
        maximum_log_ratio=maximum_log_ratio,
    )


def collapse_repeated_team_detection(
    flattened_detection_probability: torch.Tensor,
    num_agents: int,
    *,
    consistency_tolerance: float = 1.0e-6,
) -> torch.Tensor:
    """Recover unique team outcomes from agent-flattened rollout storage.

    A team result may be collapsed only when all agent rows carry the same
    target vector. This fail-closed check prevents a genuinely per-agent
    measurement from being mistaken for one joint physical outcome.
    """
    flattened = torch.as_tensor(flattened_detection_probability)
    agents = int(num_agents)
    tolerance = float(consistency_tolerance)
    if flattened.ndim != 2:
        raise ValueError("flattened detection must have shape (time*agent, target)")
    if agents < 1 or flattened.shape[0] % agents != 0:
        raise ValueError("flattened detection rows must contain complete teams")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("consistency tolerance must be finite and non-negative")
    teams = flattened.reshape(-1, agents, flattened.shape[-1])
    reference = teams[:, :1, :]
    maximum_difference = torch.max(torch.abs(teams - reference))
    if float(maximum_difference) > tolerance:
        raise ValueError(
            "agent rows do not contain one consistent team detection outcome")
    return teams[:, 0, :]


def flattened_team_detection_cvar_residual(
    flattened_detection_probability: torch.Tensor,
    qos_floor: float | torch.Tensor,
    flattened_new_log_probability: torch.Tensor,
    flattened_old_log_probability: torch.Tensor,
    *,
    num_agents: int,
    tail_fraction: float = 0.20,
    consistency_tolerance: float = 1.0e-6,
    maximum_log_ratio: float = 20.0,
) -> torch.Tensor:
    """Bridge agent-flattened PPO buffers to the joint-policy CVaR estimator."""
    probability = collapse_repeated_team_detection(
        flattened_detection_probability,
        num_agents,
        consistency_tolerance=consistency_tolerance,
    )
    new_flat = torch.as_tensor(flattened_new_log_probability)
    old_flat = torch.as_tensor(
        flattened_old_log_probability,
        dtype=new_flat.dtype,
        device=new_flat.device,
    )
    if new_flat.ndim != 1 or new_flat.shape != old_flat.shape:
        raise ValueError("flattened new/old log probabilities must align")
    if new_flat.shape[0] != probability.shape[0] * int(num_agents):
        raise ValueError("log probabilities must contain one value per agent row")
    return joint_policy_detection_cvar_residual(
        probability,
        qos_floor,
        new_flat.reshape(-1, int(num_agents)),
        old_flat.reshape(-1, int(num_agents)),
        tail_fraction=tail_fraction,
        maximum_log_ratio=maximum_log_ratio,
    )


def projected_target_dual_update(
    multipliers: torch.Tensor,
    cvar_residual: torch.Tensor,
    *,
    step_size: float,
    maximum: float | None = None,
) -> torch.Tensor:
    """Perform one target-wise projected dual-ascent update."""
    dual = torch.as_tensor(multipliers)
    residual = torch.as_tensor(
        cvar_residual, dtype=dual.dtype, device=dual.device)
    if dual.shape != residual.shape:
        raise ValueError("multipliers and CVaR residual must have equal shape")
    if torch.any(~torch.isfinite(dual)) or torch.any(dual < 0.0):
        raise ValueError("dual multipliers must be finite and non-negative")
    if torch.any(~torch.isfinite(residual)):
        raise ValueError("CVaR residual must be finite")
    step = float(step_size)
    if not math.isfinite(step) or step < 0.0:
        raise ValueError("dual step size must be finite and non-negative")
    updated = torch.clamp(dual + step * residual, min=0.0)
    if maximum is not None:
        cap = float(maximum)
        if not math.isfinite(cap) or cap < 0.0:
            raise ValueError("dual maximum must be finite and non-negative")
        updated = torch.clamp(updated, max=cap)
    return updated.detach()


def inequality_augmented_lagrangian(
    constraint_residual: torch.Tensor,
    multipliers: torch.Tensor,
    *,
    penalty: float,
) -> torch.Tensor:
    """Powell--Hestenes loss for inequalities ``g(x) <= 0``.

    This form is consistent with the projected update
    ``lambda <- [lambda + penalty * g]_+`` and avoids a manually weighted QoS
    loss. Each constraint retains its own multiplier and physical residual.
    """
    residual = torch.as_tensor(constraint_residual)
    dual = torch.as_tensor(
        multipliers, dtype=residual.dtype, device=residual.device)
    if residual.shape != dual.shape:
        raise ValueError("constraint residual and multipliers must match")
    if torch.any(~torch.isfinite(residual)):
        raise ValueError("constraint residual must be finite")
    if torch.any(~torch.isfinite(dual)) or torch.any(dual < 0.0):
        raise ValueError("multipliers must be finite and non-negative")
    rho = float(penalty)
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError("augmented-Lagrangian penalty must be positive")
    shifted = torch.relu(dual.detach() + rho * residual)
    return torch.sum(shifted.square() - dual.detach().square()) / (2.0 * rho)


def constrained_pareto_status(
    pareto_gradient_norm: float,
    constraint_residual: torch.Tensor,
    dual_change: torch.Tensor,
    consensus_residual: torch.Tensor | float = 0.0,
    *,
    stationarity_tolerance: float,
    primal_tolerance: float,
    dual_tolerance: float,
    consensus_tolerance: float = 0.0,
) -> ConstrainedParetoStatus:
    """Evaluate convergence without collapsing independent conditions."""
    stationarity = float(pareto_gradient_norm)
    primal = torch.as_tensor(constraint_residual, dtype=torch.float64)
    dual = torch.as_tensor(dual_change, dtype=torch.float64)
    consensus = torch.as_tensor(consensus_residual, dtype=torch.float64)
    values = [stationarity]
    values.extend(float(value) for value in (
        primal.abs().amax(), dual.abs().amax(), consensus.abs().amax()))
    tolerances = [
        float(stationarity_tolerance), float(primal_tolerance),
        float(dual_tolerance), float(consensus_tolerance),
    ]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("convergence residuals must be finite")
    if any(not math.isfinite(value) or value < 0.0 for value in tolerances):
        raise ValueError("convergence tolerances must be finite and non-negative")
    # Only positive constraint residual is infeasible; a negative CVaR margin
    # must not count against primal convergence.
    primal_violation = float(torch.clamp(primal, min=0.0).amax())
    dual_residual = float(dual.abs().amax())
    consensus_value = float(consensus.abs().amax())
    converged = bool(
        stationarity <= tolerances[0]
        and primal_violation <= tolerances[1]
        and dual_residual <= tolerances[2]
        and consensus_value <= tolerances[3]
    )
    return ConstrainedParetoStatus(
        pareto_stationarity=stationarity,
        primal_violation=primal_violation,
        dual_residual=dual_residual,
        consensus_residual=consensus_value,
        converged=converged,
    )
