"""Fail-closed shadow arbitration for one-step and horizon ISAC repairs.

The shadow router deliberately has no commit authority.  It computes the
action that a future consensus controller *would* be allowed to request while
leaving the deployed one-step decision unchanged.  Agreement binds the full
candidate action through a stable digest; two Boolean accept flags are not a
sufficient notion of agreement.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

import numpy as np


def _sha256_arrays(
    domain: bytes,
    arrays: Mapping[str, np.ndarray],
) -> str:
    digest = hashlib.sha256()
    digest.update(domain + b"\0")
    for name, array in arrays.items():
        value = np.ascontiguousarray(array)
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def isac_structure_digest(
    *,
    selected: np.ndarray,
    role: np.ndarray,
    owner: np.ndarray,
) -> str:
    """Bind the discrete structure, role partition and target ownership."""
    pair = np.asarray(selected, dtype=np.bool_)
    roles = np.asarray(role, dtype=np.int8)
    owners = np.asarray(owner, dtype=np.int64)
    if pair.ndim != 3 or pair.shape[0] != pair.shape[1]:
        raise ValueError("selected must have shape (K,K,Q)")
    if roles.shape != (pair.shape[0],):
        raise ValueError("role must have shape (K,)")
    if owners.shape != (pair.shape[2],):
        raise ValueError("owner must have shape (Q,)")
    return _sha256_arrays(
        b"uav-isac-shadow-structure-v1",
        {"selected": pair, "role": roles, "owner": owners},
    )


def isac_action_digest(
    *,
    selected: np.ndarray,
    role: np.ndarray,
    owner: np.ndarray,
    sensing_power_w: np.ndarray,
    comm_power_w: np.ndarray,
) -> str:
    """Return a shape- and dtype-bound SHA-256 digest of one complete action."""
    arrays = {
        "selected": np.asarray(selected, dtype=np.bool_),
        "role": np.asarray(role, dtype=np.int8),
        "owner": np.asarray(owner, dtype=np.int64),
        "sensing_power_w": np.asarray(sensing_power_w, dtype=np.float64),
        "comm_power_w": np.asarray(comm_power_w, dtype=np.float64),
    }
    selected_shape = arrays["selected"].shape
    if len(selected_shape) != 3 or selected_shape[0] != selected_shape[1]:
        raise ValueError("selected must have shape (K,K,Q)")
    if arrays["role"].shape != (selected_shape[0],):
        raise ValueError("role must have shape (K,)")
    if arrays["owner"].shape != (selected_shape[2],):
        raise ValueError("owner must have shape (Q,)")
    if arrays["sensing_power_w"].shape != (
        selected_shape[0], selected_shape[2]
    ):
        raise ValueError("sensing_power_w must have shape (K,Q)")
    if arrays["comm_power_w"].shape != (selected_shape[0],):
        raise ValueError("comm_power_w must have shape (K,)")
    if (
        np.any(~np.isfinite(arrays["sensing_power_w"]))
        or np.any(arrays["sensing_power_w"] < 0.0)
        or np.any(~np.isfinite(arrays["comm_power_w"]))
        or np.any(arrays["comm_power_w"] < 0.0)
    ):
        raise ValueError("action powers must be finite and non-negative")

    return _sha256_arrays(b"uav-isac-shadow-action-v1", arrays)


@dataclass(frozen=True)
class ShadowBranchObservation:
    """One controller branch observed by the read-only arbiter."""

    evaluated: bool
    accept: bool
    action_digest: str | None
    structure_digest: str | None
    latency_s: float
    control_energy_j: float
    hard_feasible: bool = True
    resource_accounting_complete: bool = True

    def __post_init__(self) -> None:
        latency = float(self.latency_s)
        energy = float(self.control_energy_j)
        if (
            not np.isfinite(latency) or latency < 0.0
            or not np.isfinite(energy) or energy < 0.0
        ):
            raise ValueError("branch latency/energy must be finite non-negative")
        for field_name in ("action_digest", "structure_digest"):
            digest = getattr(self, field_name)
            if digest is None:
                continue
            normalized = str(digest).strip().lower()
            if len(normalized) != 64 or any(
                character not in "0123456789abcdef" for character in normalized
            ):
                raise ValueError(
                    f"{field_name} must be a SHA-256 hex string")
            object.__setattr__(self, field_name, normalized)


@dataclass(frozen=True)
class ShadowRouteDecision:
    """Read-only decision plus the unchanged deployed baseline action."""

    live_action: str
    shadow_action: str
    reason: str
    branch_acceptance_agrees: bool
    candidate_identity_agrees: bool
    structure_identity_agrees: bool
    shared_control_latency_s: float
    parallel_latency_s: float
    shared_control_energy_j: float
    total_control_energy_j: float
    deadline_feasible: bool
    energy_feasible: bool
    commit_authority: bool = False


def arbitrate_shadow_horizon(
    one_step: ShadowBranchObservation,
    horizon: ShadowBranchObservation,
    *,
    control_period_s: float,
    max_control_energy_j: float | None = None,
    shared_control_latency_s: float = 0.0,
    shared_control_energy_j: float = 0.0,
    allow_horizon_power_on_structure_consensus: bool = False,
) -> ShadowRouteDecision:
    """Evaluate strict consensus without changing the live one-step action.

    The branches are assumed to execute concurrently, hence the critical-path
    branch-local latency enters through their maximum, while a calibrated
    complete-controller compute path is supplied once through
    ``shared_control_latency_s``.  Branch-local energy is additive, while energy
    measured over the complete parallel controller package is supplied once
    through ``shared_control_energy_j``.  A future consensus action is eligible
    only if both branches accept the exact same full action and all hard
    resource checks pass.  Any missing observation fails closed.
    """
    period = float(control_period_s)
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("control_period_s must be finite and positive")
    energy_limit = (
        None if max_control_energy_j is None else float(max_control_energy_j)
    )
    if energy_limit is not None and (
        not np.isfinite(energy_limit) or energy_limit < 0.0
    ):
        raise ValueError("max_control_energy_j must be finite non-negative")
    shared_latency = float(shared_control_latency_s)
    if not np.isfinite(shared_latency) or shared_latency < 0.0:
        raise ValueError(
            "shared_control_latency_s must be finite non-negative")
    shared_energy = float(shared_control_energy_j)
    if not np.isfinite(shared_energy) or shared_energy < 0.0:
        raise ValueError(
            "shared_control_energy_j must be finite non-negative")

    live_action = (
        "one_step_candidate"
        if one_step.evaluated and one_step.accept and one_step.hard_feasible
        else "noop"
    )
    parallel_latency = float(
        shared_latency
        + max(float(one_step.latency_s), float(horizon.latency_s)))
    total_energy = float(
        shared_energy
        + one_step.control_energy_j
        + horizon.control_energy_j)
    deadline_feasible = bool(parallel_latency <= period + 1.0e-12)
    energy_feasible = bool(
        energy_limit is None or total_energy <= energy_limit + 1.0e-15)
    branch_agreement = bool(one_step.accept == horizon.accept)
    identity_agreement = bool(
        one_step.action_digest is not None
        and horizon.action_digest is not None
        and one_step.action_digest == horizon.action_digest
    )
    structure_agreement = bool(
        one_step.structure_digest is not None
        and horizon.structure_digest is not None
        and one_step.structure_digest == horizon.structure_digest
    )

    if not one_step.evaluated or not horizon.evaluated:
        reason = "incomplete_branch_observation"
    elif (
        not one_step.resource_accounting_complete
        or not horizon.resource_accounting_complete
    ):
        reason = "incomplete_resource_accounting"
    elif not one_step.hard_feasible or not horizon.hard_feasible:
        reason = "branch_hard_infeasible"
    elif not deadline_feasible:
        reason = "parallel_deadline_failure"
    elif not energy_feasible:
        reason = "combined_energy_failure"
    elif not branch_agreement:
        reason = "acceptance_disagreement"
    elif not one_step.accept:
        reason = "consensus_noop"
    elif not identity_agreement:
        if (
            bool(allow_horizon_power_on_structure_consensus)
            and structure_agreement
        ):
            # The horizon full action, including its own RF power, already
            # passed the multi-step gate.  One-step acceptance is used only as
            # independent agreement on the discrete structural transition.
            reason = "structure_consensus_horizon_power"
        else:
            reason = (
                "candidate_identity_missing"
                if one_step.action_digest is None
                or horizon.action_digest is None
                else "candidate_identity_disagreement"
            )
    else:
        reason = "consensus_candidate"

    return ShadowRouteDecision(
        live_action=live_action,
        shadow_action=(
            "consensus_candidate"
            if reason == "consensus_candidate"
            else (
                "horizon_power_candidate"
                if reason == "structure_consensus_horizon_power"
                else "noop"
            )
        ),
        reason=reason,
        branch_acceptance_agrees=branch_agreement,
        candidate_identity_agrees=identity_agreement,
        structure_identity_agrees=structure_agreement,
        shared_control_latency_s=shared_latency,
        parallel_latency_s=parallel_latency,
        shared_control_energy_j=shared_energy,
        total_control_energy_j=total_energy,
        deadline_feasible=deadline_feasible,
        energy_feasible=energy_feasible,
        commit_authority=False,
    )


def shadow_reason_counts(
    decisions: tuple[ShadowRouteDecision, ...],
) -> Mapping[str, int]:
    """Return deterministic reason counts for audit serialization."""
    return {
        reason: sum(decision.reason == reason for decision in decisions)
        for reason in sorted({decision.reason for decision in decisions})
    }
