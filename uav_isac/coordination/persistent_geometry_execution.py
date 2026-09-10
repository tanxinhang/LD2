"""Persistent execution state for certified slow-geometry joint-plan tubes."""

# ----------------------------------------------------------------------
# AUDIT/RESEARCH-ONLY MODULE (2026-08-16 audit remediation)
#
# This module is consumed only by tools/ audit scripts and tests. It is
# NOT part of the deployment execution path (env_core / trainer) and its
# results must not be described as deployed behaviour. It exists to keep
# a specific research question reproducible; see
# docs/EXPERIMENT_LOG.md for the associated gate.
# ----------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from uav_isac.utils.sentinels import AGE_EXPIRED

from uav_isac.coordination.certified_geometry_repair import (
    CertifiedGeometryRepairDecision,
)
from uav_isac.coordination.causal_joint_plan import CausalJointPlanCommitment
from uav_isac.domain.communication import (
    CommunicationStepStatsLike,
    CommunicationTransport,
)


@dataclass(frozen=True)
class CertifiedGeometryTube:
    """A finite-horizon Actor commitment or its certified geometry repair."""

    decision_frame: int
    joint_plan_digest: str
    start_frame: int
    next_step: int
    displacement_m: np.ndarray
    movement_plan_m: np.ndarray
    repair_movement_plan_m: np.ndarray
    baseline_movement_plan_m: np.ndarray
    repair_position_plan_m: np.ndarray
    baseline_position_plan_m: np.ndarray
    certified_support: np.ndarray
    selected: np.ndarray
    role: np.ndarray
    sensing_power_w: np.ndarray
    communication_power_w: np.ndarray
    escrow_flight_energy_plan_j: np.ndarray
    coefficient_lower: np.ndarray
    coefficient_upper: np.ndarray
    baseline_upper_pd: np.ndarray
    candidate_lower_pd: np.ndarray
    certified_window_target_gain: np.ndarray
    certified_window_worst_gain: float
    certified_first_step_worst_gain: float
    certified_first_step_worst_gain_threshold: float
    execution_mode: str
    proof_available: bool

    @property
    def horizon_steps(self) -> int:
        return int(self.coefficient_lower.shape[0])

    @property
    def remaining_steps(self) -> int:
        return int(self.horizon_steps - self.next_step)

    def command(self, frame: int) -> np.ndarray:
        """Return the complete mobility vector for the requested tube step."""
        expected = int(self.start_frame + self.next_step)
        if int(frame) != expected:
            raise ValueError(
                f"geometry tube expected frame {expected}, got {int(frame)}")
        return np.asarray(
            self.movement_plan_m[self.next_step], dtype=np.float64).copy()

    def advance(self, frame: int) -> "CertifiedGeometryTube | None":
        """Consume exactly one certified step without skipping or replaying."""
        self.command(frame)
        next_step = int(self.next_step + 1)
        if next_step >= self.horizon_steps:
            return None
        return CertifiedGeometryTube(
            decision_frame=self.decision_frame,
            joint_plan_digest=self.joint_plan_digest,
            start_frame=self.start_frame,
            next_step=next_step,
            displacement_m=self.displacement_m.copy(),
            movement_plan_m=self.movement_plan_m.copy(),
            repair_movement_plan_m=self.repair_movement_plan_m.copy(),
            baseline_movement_plan_m=self.baseline_movement_plan_m.copy(),
            repair_position_plan_m=self.repair_position_plan_m.copy(),
            baseline_position_plan_m=self.baseline_position_plan_m.copy(),
            certified_support=self.certified_support.copy(),
            selected=self.selected.copy(),
            role=self.role.copy(),
            sensing_power_w=self.sensing_power_w.copy(),
            communication_power_w=self.communication_power_w.copy(),
            escrow_flight_energy_plan_j=(
                self.escrow_flight_energy_plan_j.copy()),
            coefficient_lower=self.coefficient_lower.copy(),
            coefficient_upper=self.coefficient_upper.copy(),
            baseline_upper_pd=self.baseline_upper_pd.copy(),
            candidate_lower_pd=self.candidate_lower_pd.copy(),
            certified_window_target_gain=(
                self.certified_window_target_gain.copy()),
            certified_window_worst_gain=float(
                self.certified_window_worst_gain),
            certified_first_step_worst_gain=float(
                self.certified_first_step_worst_gain),
            certified_first_step_worst_gain_threshold=float(
                self.certified_first_step_worst_gain_threshold),
            execution_mode=self.execution_mode,
            proof_available=bool(self.proof_available),
        )


def start_certified_geometry_tube(
    decision: CertifiedGeometryRepairDecision,
    *,
    decision_frame: int,
    selected: np.ndarray,
    role: np.ndarray,
    certified_support: np.ndarray,
    origin_position_m: np.ndarray,
    execution_mode: str = "repair",
) -> CertifiedGeometryTube:
    """Bind an accepted certificate to the exact future execution stream."""
    if not decision.accepted or decision.best_candidate is None:
        raise ValueError("only accepted geometry decisions can start a tube")
    best = decision.best_candidate
    mode = str(execution_mode)
    if mode not in {"repair", "committed_baseline"}:
        raise ValueError("geometry tube execution mode is invalid")
    proof_bounds = (
        best.physical_bounds
        if mode == "repair"
        else best.baseline_physical_bounds)
    lower = np.asarray(proof_bounds.lower, dtype=np.float64)
    upper = np.asarray(proof_bounds.upper, dtype=np.float64)
    baseline = np.asarray(
        best.wire_baseline_upper_pd, dtype=np.float64)
    candidate = np.asarray(
        best.wire_candidate_lower_pd
        if mode == "repair" else best.wire_baseline_lower_pd,
        dtype=np.float64)
    certified_window_target_gain = np.asarray(
        best.coupled_window_pd_gain, dtype=np.float64)
    certified_window_worst_gain = float(
        best.coupled_window_worst_pd_gain)
    certified_first_step_worst_gain = float(
        best.coupled_first_step_worst_pd_gain)
    certified_first_step_worst_gain_threshold = float(
        best.minimum_first_step_worst_pd_gain)
    active = np.asarray(selected, dtype=bool)
    support = np.asarray(certified_support, dtype=bool)
    roles = np.asarray(role, dtype=np.int8).reshape(-1)
    sensing = np.asarray(best.sensing_power_plan_w, dtype=np.float64)
    communication = np.asarray(
        best.communication_power_plan_w, dtype=np.float64)
    escrow_flight_energy = np.asarray(
        best.escrow_flight_energy_plan_j, dtype=np.float64)
    displacement = np.asarray(
        best.displacement_m
        if mode == "repair" else best.baseline_displacement_m,
        dtype=np.float64)
    movement_plan = np.asarray(
        best.movement_plan_m
        if mode == "repair" else best.baseline_movement_plan_m,
        dtype=np.float64)
    origin_position = np.asarray(origin_position_m, dtype=np.float64)
    repair_movement = np.asarray(best.movement_plan_m, dtype=np.float64)
    baseline_movement = np.asarray(
        best.baseline_movement_plan_m, dtype=np.float64)
    if lower.ndim != 4 or upper.shape != lower.shape:
        raise ValueError("geometry tube coefficients must have shape (H,K,K,Q)")
    horizon, K, K2, Q = lower.shape
    if (
        K != K2 or horizon < 1 or active.shape != (K, K, Q)
        or support.shape != (K, K, Q) or np.any(active & ~support)
        or roles.shape != (K,) or sensing.shape != (horizon, K, Q)
        or communication.shape != (horizon, K)
        or escrow_flight_energy.shape != (horizon, K)
        or displacement.shape != (K, 2)
        or baseline.shape != (horizon, Q)
        or candidate.shape != (horizon, Q)
        or certified_window_target_gain.shape != (Q,)
        or int(best.certificate_horizon_steps) != horizon
        or movement_plan.shape != (horizon, K, 2)
        or repair_movement.shape != (horizon, K, 2)
        or baseline_movement.shape != (horizon, K, 2)
        or origin_position.shape != (K, 3)
        or np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper))
        or np.any(upper < lower) or np.any(~np.isfinite(sensing))
        or np.any(sensing < 0.0) or np.any(~np.isfinite(communication))
        or np.any(communication < 0.0)
        or np.any(~np.isfinite(certified_window_target_gain))
        or not np.isfinite(certified_window_worst_gain)
        or not np.isfinite(certified_first_step_worst_gain)
        or not np.isfinite(certified_first_step_worst_gain_threshold)
        or certified_first_step_worst_gain_threshold < 0.0
    ):
        raise ValueError("geometry tube proof and execution dimensions disagree")
    def position_plan(plan: np.ndarray) -> np.ndarray:
        result = np.repeat(origin_position[None, ...], horizon, axis=0)
        result[:, :, :2] += np.cumsum(plan, axis=0)
        return result
    return CertifiedGeometryTube(
        decision_frame=int(decision_frame),
        joint_plan_digest=str(best.joint_plan_digest),
        start_frame=int(decision_frame) + 1,
        next_step=0,
        displacement_m=displacement.copy(),
        movement_plan_m=movement_plan.copy(),
        repair_movement_plan_m=repair_movement.copy(),
        baseline_movement_plan_m=baseline_movement.copy(),
        repair_position_plan_m=position_plan(repair_movement),
        baseline_position_plan_m=position_plan(baseline_movement),
        certified_support=support.copy(),
        selected=active.copy(),
        role=roles.copy(),
        sensing_power_w=sensing.copy(),
        communication_power_w=communication.copy(),
        escrow_flight_energy_plan_j=escrow_flight_energy.copy(),
        coefficient_lower=lower.copy(),
        coefficient_upper=upper.copy(),
        baseline_upper_pd=baseline.copy(),
        candidate_lower_pd=candidate.copy(),
        certified_window_target_gain=certified_window_target_gain.copy(),
        certified_window_worst_gain=certified_window_worst_gain,
        certified_first_step_worst_gain=certified_first_step_worst_gain,
        certified_first_step_worst_gain_threshold=(
            certified_first_step_worst_gain_threshold),
        execution_mode=mode,
        proof_available=True,
    )


def start_committed_joint_plan_tube(
    commitment: CausalJointPlanCommitment,
    *,
    selected: np.ndarray,
    role: np.ndarray,
    certified_support: np.ndarray,
    escrow_flight_energy_per_step_j: float,
) -> CertifiedGeometryTube:
    """Execute a rejected slow-layer commitment as an atomic baseline tube."""
    movement = np.asarray(commitment.movement_plan_m, dtype=np.float64)
    communication = np.asarray(
        commitment.communication_power_plan_w, dtype=np.float64)
    sensing = np.asarray(commitment.sensing_power_plan_w, dtype=np.float64)
    active = np.asarray(selected, dtype=bool)
    support = np.asarray(certified_support, dtype=bool)
    roles = np.asarray(role, dtype=np.int8).reshape(-1)
    horizon, K, _ = movement.shape
    Q = sensing.shape[2]
    escrow = float(escrow_flight_energy_per_step_j)
    if (
        active.shape != (K, K, Q) or support.shape != active.shape
        or np.any(active & ~support) or roles.shape != (K,)
        or communication.shape != (horizon, K)
        or sensing.shape != (horizon, K, Q)
        or not np.isfinite(escrow) or escrow < 0.0
    ):
        raise ValueError("committed baseline transaction dimensions disagree")
    zeros_coefficient = np.zeros((horizon, K, K, Q), dtype=np.float64)
    zeros_pd = np.zeros((horizon, Q), dtype=np.float64)
    return CertifiedGeometryTube(
        decision_frame=int(commitment.decision_frame),
        joint_plan_digest=commitment.digest,
        start_frame=int(commitment.decision_frame) + 1,
        next_step=0,
        displacement_m=movement[0].copy(),
        movement_plan_m=movement.copy(),
        repair_movement_plan_m=movement.copy(),
        baseline_movement_plan_m=movement.copy(),
        repair_position_plan_m=np.zeros((horizon, K, 3), dtype=np.float64),
        baseline_position_plan_m=np.zeros((horizon, K, 3), dtype=np.float64),
        certified_support=support.copy(),
        selected=active.copy(),
        role=roles.copy(),
        sensing_power_w=sensing.copy(),
        communication_power_w=communication.copy(),
        escrow_flight_energy_plan_j=np.full(
            (horizon, K), escrow, dtype=np.float64),
        coefficient_lower=zeros_coefficient,
        coefficient_upper=zeros_coefficient.copy(),
        baseline_upper_pd=zeros_pd,
        candidate_lower_pd=zeros_pd.copy(),
        certified_window_target_gain=np.zeros(Q, dtype=np.float64),
        certified_window_worst_gain=0.0,
        certified_first_step_worst_gain=0.0,
        certified_first_step_worst_gain_threshold=0.0,
        execution_mode="committed_baseline_only",
        proof_available=False,
    )


@dataclass(frozen=True)
class FrozenTokenInbox:
    """Receiver-specific delivered target masks with physical age."""

    token_mask: np.ndarray
    age_frames: np.ndarray
    ttl_frames: int

    @classmethod
    def from_observation(
        cls,
        visible: np.ndarray,
        normalized_age: np.ndarray,
        *,
        ttl_frames: int,
        observation_age_scale_frames: float = 10.0,
    ) -> "FrozenTokenInbox":
        mask = np.asarray(visible, dtype=bool)
        age = np.asarray(normalized_age, dtype=np.float64)
        if (
            mask.ndim != 3 or mask.shape[0] != mask.shape[1]
            or age.shape != mask.shape or int(ttl_frames) < 0
        ):
            raise ValueError("token observation must have shape (K,K,Q)")
        K = mask.shape[0]
        finite_age = np.where(
            mask, np.maximum(age, 0.0), np.inf)
        link_age = np.min(finite_age, axis=2)
        available = np.any(mask, axis=2)
        decoded_age = np.where(
            available,
            np.rint(link_age * float(observation_age_scale_frames)),
            -1.0,
        ).astype(np.int64)
        decoded_age[np.arange(K), np.arange(K)] = AGE_EXPIRED
        mask = mask.copy()
        mask[np.arange(K), np.arange(K), :] = False
        return cls(mask, decoded_age, int(ttl_frames))

    @property
    def visible(self) -> np.ndarray:
        return (
            np.asarray(self.token_mask, dtype=bool)
            & (self.age_frames[:, :, None] >= 0)
        )

    def advance(
        self,
        outgoing_message: np.ndarray,
        outgoing_rate: np.ndarray,
        outgoing_token_mask: np.ndarray,
        positions: np.ndarray,
        communication_power_w: np.ndarray,
        communication_model: CommunicationTransport,
    ) -> tuple["FrozenTokenInbox", CommunicationStepStatsLike]:
        """Re-evaluate the frozen Token action at the modified geometry."""
        messages = np.asarray(outgoing_message, dtype=np.float64)
        rates = np.asarray(outgoing_rate, dtype=np.int64).reshape(-1)
        masks = np.asarray(outgoing_token_mask, dtype=np.float64)
        pos = np.asarray(positions, dtype=np.float64)
        power = np.asarray(communication_power_w, dtype=np.float64).reshape(-1)
        K, _, Q = self.token_mask.shape
        if (
            messages.ndim != 2 or messages.shape[0] != K
            or rates.shape != (K,) or masks.shape != (K, Q)
            or pos.shape != (K, 3) or power.shape != (K,)
            or communication_model.message_dim != messages.shape[1]
        ):
            raise ValueError("frozen Token action dimensions are inconsistent")
        if communication_model.deadline_s > communication_model.dt + 1.0e-12:
            raise ValueError(
                "persistent Token replay requires deadline no longer than one frame")
        deliveries, stats = communication_model.transmit(
            {sender: messages[sender].copy() for sender in range(K)},
            {sender: int(rates[sender]) for sender in range(K)},
            pos,
            tx_powers_w={sender: float(power[sender]) for sender in range(K)},
            token_masks={sender: masks[sender].copy() for sender in range(K)},
        )
        next_age = self.age_frames.copy()
        retained = next_age >= 0
        next_age[retained] += 1
        expired = next_age > int(self.ttl_frames)
        next_age[expired] = AGE_EXPIRED
        next_mask = self.token_mask.copy()
        next_mask[expired, :] = False
        for delivery in deliveries:
            receiver = int(delivery.receiver)
            sender = int(delivery.sender)
            token_mask = (
                np.ones(Q, dtype=bool)
                if delivery.token_mask is None
                else np.asarray(delivery.token_mask, dtype=np.float64) > 0.5
            )
            next_mask[receiver, sender] = token_mask
            next_age[receiver, sender] = 0
        next_mask[np.arange(K), np.arange(K), :] = False
        next_age[np.arange(K), np.arange(K)] = AGE_EXPIRED
        return FrozenTokenInbox(
            token_mask=next_mask,
            age_frames=next_age,
            ttl_frames=int(self.ttl_frames),
        ), stats

    def age_without_delivery(self) -> "FrozenTokenInbox":
        """Advance physical age while an atomic slow transaction isolates writes."""
        next_age = self.age_frames.copy()
        retained = next_age >= 0
        next_age[retained] += 1
        expired = next_age > int(self.ttl_frames)
        next_age[expired] = AGE_EXPIRED
        next_mask = self.token_mask.copy()
        next_mask[expired, :] = False
        return FrozenTokenInbox(
            token_mask=next_mask,
            age_frames=next_age,
            ttl_frames=int(self.ttl_frames),
        )

    def advance_common_geometry_deliveries(
        self,
        outgoing_message: np.ndarray,
        outgoing_rate: np.ndarray,
        outgoing_token_mask: np.ndarray,
        repair_positions: np.ndarray,
        baseline_positions: np.ndarray,
        communication_power_w: np.ndarray,
        communication_model: CommunicationTransport,
    ) -> tuple["FrozenTokenInbox", np.ndarray]:
        """Commit only Token deliveries feasible on both geometry branches."""
        messages = np.asarray(outgoing_message, dtype=np.float64)
        rates = np.asarray(outgoing_rate, dtype=np.int64).reshape(-1)
        masks = np.asarray(outgoing_token_mask, dtype=np.float64)
        power = np.asarray(communication_power_w, dtype=np.float64).reshape(-1)
        K, _, Q = self.token_mask.shape
        if (
            messages.ndim != 2 or messages.shape[0] != K
            or rates.shape != (K,) or masks.shape != (K, Q)
            or np.asarray(repair_positions).shape != (K, 3)
            or np.asarray(baseline_positions).shape != (K, 3)
            or power.shape != (K,)
        ):
            raise ValueError("common-geometry Token dimensions disagree")
        transmit_inputs = (
            {sender: messages[sender].copy() for sender in range(K)},
            {sender: int(rates[sender]) for sender in range(K)},
        )
        keyword_inputs = {
            "tx_powers_w": {
                sender: float(power[sender]) for sender in range(K)},
            "token_masks": {
                sender: masks[sender].copy() for sender in range(K)},
        }
        repair_deliveries, repair_stats = communication_model.transmit(
            *transmit_inputs,
            np.asarray(repair_positions, dtype=np.float64),
            **keyword_inputs,
        )
        baseline_deliveries, baseline_stats = communication_model.transmit(
            *transmit_inputs,
            np.asarray(baseline_positions, dtype=np.float64),
            **keyword_inputs,
        )
        repair_keys = {
            (int(item.sender), int(item.receiver)) for item in repair_deliveries}
        common = [
            item for item in baseline_deliveries
            if (int(item.sender), int(item.receiver)) in repair_keys
        ]
        next_inbox = self.age_without_delivery()
        next_mask = next_inbox.token_mask.copy()
        next_age = next_inbox.age_frames.copy()
        for delivery in common:
            receiver = int(delivery.receiver)
            sender = int(delivery.sender)
            token_mask = (
                np.ones(Q, dtype=bool)
                if delivery.token_mask is None
                else np.asarray(delivery.token_mask, dtype=np.float64) > 0.5
            )
            next_mask[receiver, sender] = token_mask
            next_age[receiver, sender] = 0
        conservative_energy = np.asarray([
            max(
                repair_stats.per_sender_energy_j.get(agent, 0.0),
                baseline_stats.per_sender_energy_j.get(agent, 0.0),
            )
            for agent in range(K)
        ], dtype=np.float64)
        return FrozenTokenInbox(
            token_mask=next_mask,
            age_frames=next_age,
            ttl_frames=int(self.ttl_frames),
        ), conservative_energy


def replace_with_geometry_command(
    recorded_delta_m: np.ndarray,
    tube: CertifiedGeometryTube | None,
    *,
    frame: int,
) -> np.ndarray:
    """Use either the recorded action or the complete tube command, never both."""
    recorded = np.asarray(recorded_delta_m, dtype=np.float64)
    if recorded.ndim != 2 or recorded.shape[1] != 2:
        raise ValueError("recorded mobility action must have shape (K,2)")
    if tube is None:
        return recorded.copy()
    command = tube.command(int(frame))
    if command.shape != recorded.shape:
        raise ValueError("geometry command and recorded action dimensions disagree")
    return command
