"""Teacher-free temporal relaxation over physically feasible structures.

The hard schedule is discrete.  This module does not claim that its gradient
is an unbiased gradient of that discrete decision.  Instead it enumerates the
bounded feasible schedule set and differentiates through an entropy-regularized
distribution over that same set.  Its barycentre therefore lies in the convex
hull of schedules that already satisfy:

* one global role per UAV (Tx, Rx, or idle),
* no monostatic edge,
* one fusion owner per target,
* a per-target transmitter/pair limit, and
* a per-receiver report limit.

An overlap term with the previous soft schedule supplies the temporal proximal
coupling.  Detaching that state gives an explicit truncated-BPTT ablation.
Enumeration is intentionally bounded and fails loudly when the stated bound is
too small; silently pruning structures would invalidate the convex-hull claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
import math

import numpy as np
import torch


@dataclass(frozen=True)
class TemporalFeasibleStructureResult:
    """Soft/hard temporal schedules and their induced transmitter gains."""

    mixture: torch.Tensor
    hard_schedule: torch.Tensor
    effective_gain_per_watt: torch.Tensor
    transmitter_activation: torch.Tensor
    entropy: torch.Tensor
    candidate_count: torch.Tensor


def straight_through_structure(
    result: TemporalFeasibleStructureResult,
) -> torch.Tensor:
    """Return a hard-forward/soft-backward schedule tensor.

    The forward value is exactly the MAP feasible schedule.  Its Jacobian is
    the entropy-relaxed barycentre's Jacobian, so this is intentionally a
    straight-through (biased) estimator rather than an unbiased derivative of
    the discrete argmax.
    """
    return (
        result.hard_schedule
        + result.mixture
        - result.mixture.detach()
    )


def enumerate_feasible_single_role_structures(
    physical_support: np.ndarray | torch.Tensor,
    *,
    target_pair_limit: int,
    reports_per_receiver: int,
    max_structures: int = 4096,
) -> np.ndarray:
    """Enumerate complete feasible schedules as ``(M,K,K,Q)`` booleans.

    Every target has at least one selected transmitter and exactly one owner.
    This is the complete feasible set under the stated constraints, subject to
    ``max_structures``.  The exception on overflow is part of the scientific
    contract: callers must increase the bound or disable the exact relaxation.
    """
    support = np.asarray(
        physical_support.detach().cpu().numpy()
        if isinstance(physical_support, torch.Tensor)
        else physical_support,
        dtype=bool,
    )
    if support.ndim != 3 or support.shape[0] != support.shape[1]:
        raise ValueError("physical_support must have shape (K,K,Q)")
    agents, _, targets = support.shape
    pair_limit = int(target_pair_limit)
    receiver_limit = int(reports_per_receiver)
    maximum = int(max_structures)
    if pair_limit < 1 or receiver_limit < 1 or maximum < 1:
        raise ValueError("structure and capacity limits must be positive")
    support = support.copy()
    support[np.arange(agents), np.arange(agents), :] = False

    # A selected edge determines the used role.  Enumerating explicit role
    # assignments may generate the same schedule when an unused UAV is tagged
    # Tx/Rx rather than idle, so schedule keys are deduplicated below.
    schedule_keys: set[tuple[tuple[int, int, int], ...]] = set()
    for labels in product((0, 1, 2), repeat=agents):
        transmitters = tuple(i for i, role in enumerate(labels) if role == 1)
        receivers = tuple(i for i, role in enumerate(labels) if role == 2)
        if not transmitters or not receivers:
            continue
        choices_by_target: list[tuple[tuple[tuple[int, int, int], ...], ...]] = []
        complete = True
        for target in range(targets):
            choices = []
            for receiver in receivers:
                valid_tx = tuple(
                    transmitter for transmitter in transmitters
                    if support[transmitter, receiver, target]
                )
                for count in range(1, min(pair_limit, len(valid_tx)) + 1):
                    for subset in combinations(valid_tx, count):
                        choices.append(tuple(
                            (int(transmitter), int(receiver), int(target))
                            for transmitter in subset
                        ))
            if not choices:
                complete = False
                break
            choices_by_target.append(tuple(choices))
        if not complete:
            continue

        receiver_load = np.zeros(agents, dtype=np.int64)

        def extend(target: int, selected: tuple[tuple[int, int, int], ...]) -> None:
            if target == targets:
                schedule_keys.add(tuple(sorted(selected)))
                if len(schedule_keys) > maximum:
                    raise RuntimeError(
                        "feasible structure count exceeds max_structures; "
                        "increase the explicit bound or disable the exact "
                        "temporal structure relaxation"
                    )
                return
            for option in choices_by_target[target]:
                receiver = option[0][1]
                added = len(option)
                if receiver_load[receiver] + added > receiver_limit:
                    continue
                receiver_load[receiver] += added
                extend(target + 1, selected + option)
                receiver_load[receiver] -= added

        extend(0, tuple())

    if not schedule_keys:
        raise ValueError(
            "no complete single-role structure exists on physical_support")
    ordered = sorted(schedule_keys)
    structures = np.zeros(
        (len(ordered), agents, agents, targets), dtype=bool)
    for index, schedule in enumerate(ordered):
        for transmitter, receiver, target in schedule:
            structures[index, transmitter, receiver, target] = True
    return structures


def _edge_scores(
    sensing_logits: torch.Tensor,
    coefficient: torch.Tensor,
    *,
    physical_log_weight: float,
    transmitter_budget_w: torch.Tensor | None,
) -> torch.Tensor:
    """Causal transmit power split plus a scale-free physical log score.

    Only the transmitter's target allocation appears: bistatic reception does
    not radiate a second sensing waveform, so inserting the receiver's sensing
    weight would double-count an RF resource that does not exist.
    """
    transmitter_score = torch.log_softmax(sensing_logits, dim=-1)[:, None, :]
    if transmitter_budget_w is not None:
        transmitter_score = transmitter_score + torch.log(
            transmitter_budget_w.clamp_min(1.0e-12))[:, None, None]
    target_scale = coefficient.amax(dim=(0, 1), keepdim=True).clamp_min(
        torch.finfo(coefficient.dtype).tiny)
    normalized = coefficient / target_scale
    physical_score = torch.log(normalized.clamp_min(1.0e-12))
    return transmitter_score + float(physical_log_weight) * physical_score


def temporal_feasible_structure_mixture(
    sensing_logits: torch.Tensor,
    coefficient_per_watt: torch.Tensor,
    *,
    target_pair_limit: int,
    reports_per_receiver: int,
    temperature: float = 0.25,
    inertia: float = 0.25,
    physical_log_weight: float = 1.0,
    max_structures: int = 4096,
    detach_between_frames: bool = False,
    transmitter_budget_w: torch.Tensor | None = None,
) -> TemporalFeasibleStructureResult:
    """Differentiate over the convex hull of exact feasible schedules.

    Shapes are ``sensing_logits=(H,K,Q)`` and
    ``coefficient_per_watt=(H,K,K,Q)``.  For candidate schedule ``z_m`` the
    frame score is

    ``<z_m, s_t> + inertia * <z_m, zbar_{t-1}>``.

    Softmax at positive temperature is the unique optimizer of the linear
    schedule score plus Shannon entropy over the finite feasible set.  The
    hard output is its MAP schedule and is used only for forward/execution
    audits; the mixture supplies the deliberately biased surrogate Jacobian.
    """
    logits = torch.as_tensor(sensing_logits)
    coefficient = torch.as_tensor(
        coefficient_per_watt, dtype=logits.dtype, device=logits.device)
    if not logits.is_floating_point() or logits.ndim != 3:
        raise ValueError("sensing_logits must have shape (H,K,Q)")
    horizon, agents, targets = logits.shape
    if coefficient.shape != (horizon, agents, agents, targets):
        raise ValueError("coefficient_per_watt must have shape (H,K,K,Q)")
    if torch.any(~torch.isfinite(logits)) or torch.any(
        ~torch.isfinite(coefficient)
    ) or torch.any(coefficient < 0.0):
        raise ValueError("structure inputs must be finite and non-negative")
    budgets = None
    if transmitter_budget_w is not None:
        budgets = torch.as_tensor(
            transmitter_budget_w, dtype=logits.dtype, device=logits.device)
        if budgets.shape != (horizon, agents):
            raise ValueError("transmitter_budget_w must have shape (H,K)")
        if torch.any(~torch.isfinite(budgets)) or torch.any(budgets < 0.0):
            raise ValueError("transmitter budgets must be finite/non-negative")
    values = (float(temperature), float(inertia), float(physical_log_weight))
    if any(not math.isfinite(value) for value in values):
        raise ValueError("structure relaxation parameters must be finite")
    if values[0] <= 0.0 or values[1] < 0.0 or values[2] < 0.0:
        raise ValueError(
            "temperature must be positive; inertia and physical weight "
            "must be non-negative")

    mixtures = []
    hard_schedules = []
    effective_gains = []
    activations = []
    entropies = []
    counts = []
    previous_mixture = None
    for frame in range(horizon):
        feasible_np = enumerate_feasible_single_role_structures(
            coefficient[frame] > 0.0,
            target_pair_limit=target_pair_limit,
            reports_per_receiver=reports_per_receiver,
            max_structures=max_structures,
        )
        feasible = torch.as_tensor(
            feasible_np, dtype=logits.dtype, device=logits.device)
        scores = torch.sum(
            feasible * _edge_scores(
                logits[frame], coefficient[frame],
                physical_log_weight=values[2],
                transmitter_budget_w=(
                    None if budgets is None else budgets[frame])),
            dim=(1, 2, 3),
        )
        if previous_mixture is not None and values[1] > 0.0:
            carried = (
                previous_mixture.detach()
                if detach_between_frames else previous_mixture)
            # Divide by Q*Kmax so inertia has a stable interpretation when
            # problem cardinality changes.
            normalizer = max(1, targets * int(target_pair_limit))
            scores = scores + values[1] * torch.sum(
                feasible * carried.unsqueeze(0), dim=(1, 2, 3)
            ) / float(normalizer)
        probability = torch.softmax(scores / values[0], dim=0)
        mixture = torch.sum(
            probability[:, None, None, None] * feasible, dim=0)
        hard = feasible[torch.argmax(scores)]
        # zbar_iq is the relaxed counterpart of sum_j x_ijq in the hard
        # structure--power link p_iq <= P_i sum_j x_ijq.
        activation = mixture.sum(dim=1).clamp(0.0, 1.0)
        effective_gain = torch.sum(
            coefficient[frame] * mixture, dim=1)
        entropy = -torch.sum(
            probability * torch.log(probability.clamp_min(1.0e-12)))
        mixtures.append(mixture)
        hard_schedules.append(hard)
        effective_gains.append(effective_gain)
        activations.append(activation)
        entropies.append(entropy)
        counts.append(float(feasible.shape[0]))
        previous_mixture = mixture

    return TemporalFeasibleStructureResult(
        mixture=torch.stack(mixtures),
        hard_schedule=torch.stack(hard_schedules),
        effective_gain_per_watt=torch.stack(effective_gains),
        transmitter_activation=torch.stack(activations),
        entropy=torch.stack(entropies),
        candidate_count=torch.as_tensor(
            counts, dtype=logits.dtype, device=logits.device),
    )
