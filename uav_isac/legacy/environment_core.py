"""Legacy environment core retained behind V2 adapters.

Step pipeline:
  1. Apply outer actions to UAVs (position, role)
  2. Step target dynamics
  3. Compute bistatic geometry → Deflection matrix
  4. Run inner P0 solver → selected assignments, D_q*
  5. Compute P_D per target
  6. Compute rewards (team + shaped)
  7. Check constraints → penalties
  8. Update beliefs (increment AoI, reset for observed targets)
  9. Build next observations
  10. Check termination
"""

import copy
import time
import os
import json
import logging
import numpy as np
from numbers import Integral
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from uav_isac.utils.types import Action, P0Solution, UAVState, TargetState
from uav_isac.utils.math_utils import symmetric_2x2_max_eigenvalue
from uav_isac.utils.sentinels import (
    FRAME_NEVER,
    FRAME_NOT_APPLICABLE,
    TARGET_INDEX_NONE,
)
from uav_isac.environment.uav import UAV
from uav_isac.environment.target import Target
from uav_isac.environment.belief import (
    BeliefManager,
    batched_pair_covariance_intersection,
    bistatic_range_doppler_crlb,
    bistatic_range_doppler_measurement_and_jacobian,
    generalized_covariance_intersection,
)
from uav_isac.environment.observation import ObservationBuilder
from uav_isac.prediction.markov_kinematics import predict_reflecting_cv_mean
from uav_isac.environment.action import ActionSpace
from uav_isac.environment.reward import (
    RewardComputer,
    compute_segmented_coordination_reward,
)
from uav_isac.environment.constraints import ConstraintChecker
from uav_isac.physical.deflection import DeflectionComputer
from uav_isac.physical.otfs import (
    compute_dd_phys_gain_batch,
)
from uav_isac.physical.inner_solver import InnerSolver
from uav_isac.physical.detection import (
    compute_detection_probabilities,
    compute_target_utilities,
    minimum_deflection_for_detection_probability,
)
from uav_isac.physical.feasibility_oracle import (
    repair_invalid_local_only_edges_min_change,
    solve_maxmin_single_role_pairs,
    unit_deflection_tensor,
)
from uav_isac.physical.finite_blocklength import (
    minimum_snr_normal_approximation,
)
from uav_isac.physical.evidence import (
    DETECTION_FUSION_MODES,
    DeflectionConfidenceQuantizer,
    EvidencePacketLayout,
    estimate_quantized_evidence_detection,
    local_ambiguity_top2_mask,
    local_quality_topk_mask,
    quantize_belief_feedback,
    receiver_deflection_from_broadcast_waveforms,
    receiver_deflection_from_selected,
    route_structured_evidence,
    scheduled_fusion_owner,
    select_detection_deflection,
)
from uav_isac.environment.trust_manager import TrustManager
from uav_isac.environment.communication import (
    CommunicationStepStats,
    InterUAVCommunicationModel,
)
from uav_isac.coordination.qpd import (
    local_primal_dual_update,
    update_virtual_queue,
)
from uav_isac.coordination.maxmin_power import (
    IncompleteFixedOwnerStructureError,
    MaxMinPowerResult,
    NonUniqueFixedOwnerStructureError,
    blend_row_feasible_power_with_inertia,
    fixed_owner_gain_matrix,
    fixed_owner_gain_matrix_from_selected_values,
    local_transmitter_range_minimax_share,
    replicated_local_row_maxmin_power,
    sparse_harmonic_row_power,
    solve_fixed_structure_maxmin_power_lp as _solve_fixed_structure_maxmin_power_lp,
)
from uav_isac.coordination.parallel_power_executor import (
    ReplicatedPowerProcessExecutor,
)
from uav_isac.acceleration.hyperedge import (
    create_hyperedge_acceleration_service,
)
from uav_isac.acceleration.deflection import (
    create_deflection_materialization_service,
)
from uav_isac.coordination.composable_certificate import (
    aggregate_target_responsibility_certificate,
    conservative_row_contribution,
    quantize_lower_log,
    quantize_upper_log,
    robust_movement_dominates,
    robust_primal_dual_bounds,
    targetwise_price_row_dual_upper,
    uniform_price_row_dual_upper,
)
from uav_isac.coordination.hyperedge import (
    annular_alternating_movement_delta,
    deterministic_bottleneck_cost_assignment,
    deterministic_bottleneck_matching,
    bistatic_geometry_tail_ratio,
    project_pairwise_safe_movement,
    decode_offer_stream,
    encode_offer_stream,
    factorized_endpoint_capabilities,
    gauss_southwell_bistatic_geometry_step,
    gauss_southwell_bistatic_geometry_sweep,
    mutual_endpoint_consensus,
    plan_budget_certified_hyperedges,
    plan_local_hyperedges,
    plan_refined_endpoint_hyperedges,
    plan_reserved_endpoint_hyperedges,
    plan_sparse_coalition_endpoint_hyperedges,
    role_aware_bistatic_movement_cost,
    reconstruct_bistatic_pair_value,
    update_consensus_streak,
)


def filter_deflection_by_local_commitments(
    deflection_entries: list,
    sensing_power_w: np.ndarray,
    topk: int = 1,
    require_receiver: bool = True,
    mode: str = 'hard',
    soft_floor: float = 0.25,
    uncertainty_relief: float = 1.0,
    commitment_mask: Optional[np.ndarray] = None,
) -> Tuple[list, Dict[str, object]]:
    """Apply local UAV commitments to the P0 candidate graph.

    ``sensing_power_w[k]`` is emitted by UAV ``k`` from its local observation
    and delivered-token context. P0 may still resolve geometry/capacity ties,
    but it may not redirect an endpoint to a target outside that endpoint's
    top-k commitment. Requiring receiver agreement creates a local rendezvous:
    both bistatic endpoints must independently name the same target.

    ``hard`` reproduces the original top-k deletion rule. ``soft`` is a
    confidence-calibrated graph prior: it multiplies each candidate's ranking
    deflection by a bounded local-agreement gate but never erases a physically
    valid edge. High-entropy commitments relax the gate toward one, preventing
    arbitrary top-k ties or immature tokens from starving an entire target.
    Realized detection still uses the unscaled physical deflection.
    """
    power = np.asarray(sensing_power_w, dtype=np.float64)
    if power.ndim != 2:
        raise ValueError(
            'sensing_power_w must have shape (num_uavs, num_targets)')
    K, Q = power.shape
    mode = str(mode).strip().lower()
    if mode not in {'hard', 'soft'}:
        raise ValueError('commitment mode must be hard or soft')
    soft_floor = float(np.clip(soft_floor, 0.0, 1.0))
    uncertainty_relief = float(np.clip(uncertainty_relief, 0.0, 1.0))
    if Q <= 0:
        return [], {
            'learned_comm_commitment_candidate_fraction': 0.0,
            'learned_comm_commitment_target_coverage': 0.0,
            'learned_comm_commitment_topk': 0,
            'learned_comm_commitment_require_receiver': bool(require_receiver),
            'learned_comm_commitment_mode': mode,
        }
    k_eff = int(np.clip(topk, 1, Q))
    if commitment_mask is None:
        committed = np.zeros((K, Q), dtype=bool)
        order = np.argsort(-power, axis=1, kind='stable')
        rows = np.arange(K)[:, None]
        committed[rows, order[:, :k_eff]] = True
    else:
        explicit = np.asarray(commitment_mask)
        if explicit.shape != (K, Q):
            raise ValueError(
                'commitment_mask must have the same (num_uavs, '
                'num_targets) shape as sensing_power_w')
        committed = explicit > 0.5

    ranked = []
    gates = []
    if mode == 'soft':
        row_sum = np.sum(np.maximum(power, 0.0), axis=1, keepdims=True)
        probs = np.divide(
            np.maximum(power, 0.0),
            row_sum,
            out=np.full_like(power, 1.0 / Q),
            where=row_sum > 1e-12,
        )
        relative = probs / np.maximum(np.max(
            probs, axis=1, keepdims=True), 1e-12)
        if Q > 1:
            entropy = -np.sum(
                probs * np.log(np.maximum(probs, 1e-12)), axis=1
            ) / np.log(float(Q))
        else:
            entropy = np.zeros(K, dtype=np.float64)
        certainty = np.clip(1.0 - entropy, 0.0, 1.0)

    for entry in deflection_entries:
        i, j, q = int(entry.i), int(entry.j), int(entry.q)
        sender_ok = 0 <= i < K and 0 <= q < Q and committed[i, q]
        receiver_ok = (
            not require_receiver
            or (0 <= j < K and committed[j, q]))
        if mode == 'hard':
            if sender_ok and receiver_ok:
                ranked.append(entry)
            continue

        if not (0 <= i < K and 0 <= j < K and 0 <= q < Q):
            continue
        if require_receiver:
            agreement = float(np.sqrt(relative[i, q] * relative[j, q]))
            pair_certainty = float(0.5 * (certainty[i] + certainty[j]))
        else:
            agreement = float(relative[i, q])
            pair_certainty = float(certainty[i])
        adaptive_floor = soft_floor + (
            uncertainty_relief * (1.0 - pair_certainty)
            * (1.0 - soft_floor))
        gate = float(np.clip(
            adaptive_floor + (1.0 - adaptive_floor) * agreement,
            soft_floor,
            1.0,
        ))
        gates.append(gate)
        ranked.append(entry._replace(d_eff=float(entry.d_eff) * gate))

    covered_targets = {int(entry.q) for entry in ranked}
    hard_covered_targets = {
        int(entry.q) for entry in deflection_entries
        if (0 <= int(entry.i) < K and 0 <= int(entry.q) < Q
            and committed[int(entry.i), int(entry.q)]
            and (not require_receiver
                 or (0 <= int(entry.j) < K
                     and committed[int(entry.j), int(entry.q)])))
    }
    total = len(deflection_entries)
    metrics = {
        'learned_comm_commitment_candidate_fraction': (
            float(len(ranked) / total) if total else 0.0),
        'learned_comm_commitment_target_coverage': float(
            len(covered_targets) / Q),
        'learned_comm_commitment_hard_target_coverage': float(
            len(hard_covered_targets) / Q),
        'learned_comm_commitment_topk': int(k_eff),
        'learned_comm_commitment_claims_per_uav': float(np.mean(
            committed.sum(axis=1))),
        'learned_comm_commitment_require_receiver': bool(require_receiver),
        'learned_comm_commitment_mode': mode,
        'learned_comm_commitment_soft_gate_mean': float(np.mean(
            gates or [1.0])),
        'learned_comm_commitment_soft_gate_min': float(np.min(
            gates or [1.0])),
        'learned_comm_commitment_mask': committed.copy(),
    }
    return ranked, metrics


@dataclass
class StepInfo:
    """Detailed information about one simulation step."""
    frame: int
    uav_states: List[UAVState]
    target_states: List[TargetState]
    deflection_entries: list
    p0_solution: object
    P_D_q: np.ndarray
    team_reward: float
    shaped_rewards: Dict[int, float]
    constraint_info: Dict
    dones: Dict[int, bool]
    # ── Pairing diagnostics (P0): attribute eval collapse to role assignment ──
    roles: Optional[np.ndarray] = None       # (K,) realized roles this frame
    n_tx: int = 0                            # # UAVs with role==tx(0)
    n_rx: int = 0                            # # UAVs with role==rx(1)
    n_selected: int = 0                      # # (i,j,q) triples P0 actually selected
    n_duplex: int = 0                        # # UAVs used as TX and RX in separate sub-slots
    valid_pair: bool = False                 # P0 selected >=1 bistatic pair -> sensing happened
    no_tx: bool = False                      # zero TX this frame (no sensing possible)
    all_same_role: bool = False              # all K UAVs picked the same role (degenerate)
    p0_resolved: bool = False
    p0_solve_time_s: float = 0.0
    learned_comm: Optional[Dict[str, object]] = None
    reward_components: Optional[Dict[str, float]] = None


class EnvironmentCore:
    """Stateless core executing one simulation frame."""

    def __init__(self, config, rng: Optional[np.random.Generator] = None):
        """
        Args:
            config: MasterConfig object
            rng: NumPy random generator
        """
        validator = getattr(config, "validate_runtime_boundary", None)
        if not callable(validator):
            raise ValueError(
                "environment config must expose validate_runtime_boundary()")
        validator()
        self.cfg = config
        self.rng = rng if rng is not None else np.random.default_rng()

        # Extract params
        sc = config.scenario
        ua = config.uav
        ta = config.target
        ot = config.otfs
        ch = config.channel
        de = config.detection
        p0 = config.p0_solver
        ma = config.marl

        self.K = sc.K
        self.Q = sc.Q
        self.T = sc.T
        self.dt = sc.dt
        self.area_size = (sc.region_size[0], sc.region_size[1])
        self.height = sc.height
        self.tracking_enabled = bool(getattr(ma, 'tracking_enabled', True))
        self._distributed_coordination_use_local_belief_targets = bool(
            getattr(
                ma,
                'distributed_coordination_use_local_belief_targets',
                False,
            )
        )
        # Post-G2 strict no-truth closure (audit advice/001 P0, 2026-08-26).
        # When enabled, no distributed decision path may read simulator ground
        # truth: the coordination-target resolver and the replicated-power
        # fallback fail closed instead.  The canonical post-G2 manifest pins
        # this together with ``distributed_coordination_use_local_belief_targets:
        # true`` (and therefore ``tracking_enabled: true``, which the belief
        # manager requires -- see the guard below).
        self._distributed_no_truth_fail_closed = bool(
            getattr(ma, 'distributed_no_truth_fail_closed', False))
        if (
            self._distributed_no_truth_fail_closed
            and not self._distributed_coordination_use_local_belief_targets
        ):
            raise ValueError(
                'distributed_no_truth_fail_closed requires '
                'distributed_coordination_use_local_belief_targets=true')
        if (
            self._distributed_coordination_use_local_belief_targets
            and not self.tracking_enabled
        ):
            raise ValueError(
                'distributed_coordination_use_local_belief_targets requires '
                'tracking_enabled=true')
        # C6 decision-sufficient comm (audit advice/001 §12-13, 2026-08-26):
        # under the strict distributed identity a viewer must NOT broadcast
        # full U2U state every frame.  Parse the decision-sufficient switch,
        # its quantisation dynamic range, and its certified-bit ceiling, and
        # fail closed when the switch is enabled without the strict no-truth
        # gate (decision-sufficient tokens are only sound when no simulator
        # truth can leak into them).
        self._distributed_decision_sufficient_comm_enabled = bool(
            getattr(ma, 'distributed_decision_sufficient_comm_enabled', False))
        self._distributed_decision_sufficient_dynamic_range = max(
            1e-9, float(getattr(
                ma, 'distributed_decision_sufficient_dynamic_range', 1.0)))
        self._distributed_decision_sufficient_max_bits = max(
            1, int(getattr(ma, 'distributed_decision_sufficient_max_bits', 32)))
        if (self._distributed_decision_sufficient_comm_enabled
                and not self._distributed_no_truth_fail_closed):
            raise ValueError(
                'distributed_decision_sufficient_comm_enabled requires '
                'distributed_no_truth_fail_closed=true (C6 fail-closed)')
        # C6 split switches (advice/001 §12-13): the decision-sufficient comm is
        # a two-stage pipeline -- (1) event-triggered silence (suppress the
        # token while the LOCAL margin Δ is CERTIFIED to persist) and
        # (2) adaptive decision-preserving bits (transmit the minimal certified
        # precision instead of rounding up to the rate ladder).  Each stage is
        # a named flag; the chain is fail-closed: adaptive-bits requires
        # event-trigger, event-trigger requires comm-enable, comm-enable
        # requires the strict no-truth gate above.  All three default OFF so
        # every pre-C6 run stays bit-identical.
        self._distributed_decision_sufficient_event_trigger_enabled = bool(
            getattr(
                ma, 'distributed_decision_sufficient_event_trigger_enabled',
                False))
        self._distributed_decision_sufficient_adaptive_bits_enabled = bool(
            getattr(
                ma, 'distributed_decision_sufficient_adaptive_bits_enabled',
                False))
        if (self._distributed_decision_sufficient_adaptive_bits_enabled
                and not self._distributed_decision_sufficient_event_trigger_enabled):
            raise ValueError(
                'distributed_decision_sufficient_adaptive_bits_enabled '
                'requires distributed_decision_sufficient_event_trigger_'
                'enabled=true (C6 decision-sufficient chain)')
        if (self._distributed_decision_sufficient_event_trigger_enabled
                and not self._distributed_decision_sufficient_comm_enabled):
            raise ValueError(
                'distributed_decision_sufficient_event_trigger_enabled '
                'requires distributed_decision_sufficient_comm_enabled=true '
                '(C6 decision-sufficient chain)')
        self._detection_fusion_mode = str(getattr(
            ma, 'detection_fusion_mode', 'local_only')).strip().lower()
        self._passive_multireceiver_evidence_enabled = bool(getattr(
            ma, 'passive_multireceiver_evidence_enabled', False))
        self._passive_multireceiver_belief_update_enabled = bool(getattr(
            ma, 'passive_multireceiver_belief_update_enabled', False))
        self._u2u_belief_feedback_enabled = bool(getattr(
            ma, 'u2u_belief_feedback_enabled', False))
        self._u2u_belief_feedback_mean_bits = max(1, int(getattr(
            ma, 'u2u_belief_feedback_mean_bits', 12)))
        self._u2u_belief_feedback_cov_bits = max(1, int(getattr(
            ma, 'u2u_belief_feedback_cov_bits', 8)))
        self._u2u_belief_feedback_aoi_bits = max(1, int(getattr(
            ma, 'u2u_belief_feedback_aoi_bits', 8)))
        self._evidence_legacy_service_envelope_enabled = bool(getattr(
            ma, 'evidence_legacy_service_envelope_enabled', False))
        self._evidence_legacy_service_aoi_bits = max(1, int(getattr(
            ma, 'evidence_legacy_service_aoi_bits', 8)))
        self._u2u_belief_feedback_schedule = str(getattr(
            ma, 'u2u_belief_feedback_schedule',
            'evidence_topk')).strip().lower()
        if self._u2u_belief_feedback_schedule not in {
                'evidence_topk', 'freshness'}:
            raise ValueError(
                'u2u_belief_feedback_schedule must be evidence_topk or '
                'freshness')
        self._u2u_belief_feedback_topk = max(1, int(getattr(
            ma, 'u2u_belief_feedback_topk', 2)))
        self._u2u_belief_feedback_bit_budget = max(0, int(getattr(
            ma, 'u2u_belief_feedback_bit_budget', 800)))
        self._u2u_belief_feedback_max_union_per_source = max(1, int(getattr(
            ma, 'u2u_belief_feedback_max_union_per_source', 2)))
        self._u2u_belief_feedback_aoi_weight = max(0.0, float(getattr(
            ma, 'u2u_belief_feedback_aoi_weight', 1.0 / 3.0)))
        self._u2u_belief_feedback_uncertainty_weight = max(
            0.0, float(getattr(
                ma, 'u2u_belief_feedback_uncertainty_weight', 1.0 / 3.0)))
        self._u2u_belief_feedback_task_weight = max(0.0, float(getattr(
            ma, 'u2u_belief_feedback_task_weight', 1.0 / 3.0)))
        self._u2u_belief_feedback_owner_aware = bool(getattr(
            ma, 'u2u_belief_feedback_owner_aware', True))
        self._u2u_belief_feedback_max_age_frames = max(0, int(getattr(
            ma, 'u2u_belief_feedback_max_age_frames', 5)))
        self._owner_posterior_enabled = bool(getattr(
            ma, 'distributed_owner_posterior_enabled', False))
        self._owner_posterior_mean_bits = max(1, int(getattr(
            ma, 'distributed_owner_posterior_mean_bits', 12)))
        self._owner_posterior_cov_bits = max(1, int(getattr(
            ma, 'distributed_owner_posterior_cov_bits', 8)))
        self._owner_posterior_aoi_bits = max(1, int(getattr(
            ma, 'distributed_owner_posterior_aoi_bits', 8)))
        self._owner_posterior_max_age = max(0, int(getattr(
            ma, 'distributed_owner_posterior_max_age_frames', 5)))
        owner_belief_model = str(getattr(
            ma, 'belief_motion_model', 'TARGET')).upper()
        if owner_belief_model == 'TARGET':
            owner_belief_model = str(getattr(
                self.cfg.target, 'motion_model', 'CV')).upper()
        self._owner_posterior_state_dim = (
            6 if owner_belief_model == 'CA' else 4)
        if self._owner_posterior_enabled and not self.tracking_enabled:
            raise ValueError(
                'distributed owner posterior requires tracking_enabled=true')
        if (
            self._u2u_belief_feedback_enabled
            and self._detection_fusion_mode != 'u2u_distributed'
        ):
            raise ValueError(
                'u2u_belief_feedback_enabled requires '
                'detection_fusion_mode=u2u_distributed')
        if self._u2u_belief_feedback_enabled and not self.tracking_enabled:
            raise ValueError(
                'u2u_belief_feedback_enabled requires tracking_enabled=true')
        if (
            self._u2u_belief_feedback_enabled
            and bool(getattr(ma, 'belief_nis_enabled', False))
        ):
            raise ValueError(
                'u2u belief feedback currently requires '
                'belief_nis_enabled=false because sender-private adaptive '
                'process-noise scales are not carried in the packet')
        if (
            self._passive_multireceiver_belief_update_enabled
            and not self._passive_multireceiver_evidence_enabled
        ):
            raise ValueError(
                'passive multireceiver belief updates require passive '
                'multireceiver evidence')
        if self._detection_fusion_mode not in DETECTION_FUSION_MODES:
            allowed = ', '.join(sorted(DETECTION_FUSION_MODES))
            raise ValueError(
                'detection_fusion_mode must be one of ' + allowed)
        self.ground_communication_enabled = bool(getattr(
            ma, 'ground_communication_enabled', True))
        self._comm_mode = str(getattr(ma, 'learned_comm_mode', 'on'))
        if (
            self._detection_fusion_mode == 'u2u_distributed'
            and self._comm_mode != 'cost_aware'
        ):
            raise ValueError(
                'u2u_distributed detection requires '
                'learned_comm_mode=cost_aware')
        if (
            self._detection_fusion_mode == 'u2u_distributed'
            and (
                bool(getattr(ma, 'use_centered_marginal', False))
                or bool(getattr(ma, 'use_difference_reward', False))
            )
        ):
            raise ValueError(
                'u2u_distributed currently requires centered marginal and '
                'difference rewards to be disabled; packet-level '
                'counterfactual attribution is a later gate')
        self._joint_isac_power_enabled = bool(getattr(
            ma, 'joint_isac_power_enabled', False))
        if self._joint_isac_power_enabled and self._comm_mode != 'cost_aware':
            raise ValueError(
                'joint_isac_power_enabled requires learned_comm_mode=cost_aware')
        self._analytical_sensing_power_enabled = bool(getattr(
            ma, 'analytical_sensing_power_enabled', False))
        if self._analytical_sensing_power_enabled and not self._joint_isac_power_enabled:
            raise ValueError(
                'analytical_sensing_power_enabled requires joint_isac_power_enabled')
        self._analytical_sensing_power_reserve_pd = float(getattr(
            ma, 'analytical_sensing_power_reserve_pd', 0.0))
        self._distributed_primal_dual_power_enabled = bool(getattr(
            ma, 'distributed_primal_dual_power_enabled', False))
        self._temporal_unrolled_power_enabled = bool(getattr(
            ma, 'temporal_unrolled_power_enabled', False))
        self._temporal_feasible_structure_enabled = bool(getattr(
            ma, 'temporal_feasible_structure_enabled', False))
        if self._distributed_primal_dual_power_enabled:
            if not self._analytical_sensing_power_enabled:
                raise ValueError(
                    'distributed primal-dual power requires analytical '
                    'sensing power')
            if self._analytical_sensing_power_reserve_pd <= 0.0:
                raise ValueError(
                    'distributed primal-dual power requires a positive '
                    'analytical_sensing_power_reserve_pd')
        if (self._temporal_unrolled_power_enabled
                and not self._distributed_primal_dual_power_enabled):
            raise ValueError(
                'temporal unrolled power requires distributed primal-dual power')
        if (self._temporal_feasible_structure_enabled
                and not self._temporal_unrolled_power_enabled):
            raise ValueError(
                'temporal feasible structure requires temporal unrolled power')
        self._distributed_primal_dual_power_controller = None
        self._analytical_power_geometry_reuse_enabled = bool(getattr(
            ma, 'analytical_power_geometry_reuse_enabled', False))
        if (self._analytical_power_geometry_reuse_enabled
                and not self._analytical_sensing_power_enabled):
            raise ValueError(
                'analytical power geometry reuse requires analytical '
                'sensing power')
        self._distributed_replicated_power_enabled = bool(getattr(
            ma, 'distributed_replicated_power_enabled', False))
        if (self._distributed_primal_dual_power_enabled
                and self._distributed_replicated_power_enabled):
            raise ValueError(
                'distributed primal-dual and replicated power modes are '
                'mutually exclusive')
        self._distributed_replicated_power_process_parallel_enabled = bool(
            getattr(
                ma,
                'distributed_replicated_power_process_parallel_enabled',
                False,
            ))
        self._distributed_replicated_power_process_workers = int(getattr(
            ma, 'distributed_replicated_power_process_workers', 4))
        self._distributed_replicated_power_process_timeout_s = float(getattr(
            ma, 'distributed_replicated_power_process_timeout_s', 0.1))
        self._distributed_replicated_power_parallel_fallback_to_serial = bool(
            getattr(
                ma,
                'distributed_replicated_power_parallel_fallback_to_serial',
                True,
            ))
        self._distributed_replicated_power_parallel_failure_mode = str(getattr(
            ma,
            'distributed_replicated_power_parallel_failure_mode',
            'serial',
        )).strip().lower()
        self._distributed_replicated_power_deadline_incumbent_tolerance = float(
            getattr(
                ma,
                'distributed_replicated_power_deadline_incumbent_relative_tolerance',
                0.05,
            ))
        self._distributed_replicated_power_max_consecutive_failures = int(
            getattr(
                ma,
                'distributed_replicated_power_max_consecutive_process_failures',
                3,
            ))
        self._distributed_replicated_power_certificate_shadow_global_lp_enabled = bool(
            getattr(
                ma,
                'distributed_replicated_power_certificate_shadow_global_lp_enabled',
                False,
            ))
        if (
            self._distributed_replicated_power_process_parallel_enabled
            and not self._distributed_replicated_power_enabled
        ):
            raise ValueError(
                'parallel replicated power requires distributed replicated '
                'power')
        if self._distributed_replicated_power_process_workers < 1:
            raise ValueError(
                'distributed replicated power process workers must be positive')
        if (
            not np.isfinite(
                self._distributed_replicated_power_process_timeout_s)
            or self._distributed_replicated_power_process_timeout_s <= 0.0
        ):
            raise ValueError(
                'distributed replicated power process timeout must be positive')
        if self._distributed_replicated_power_parallel_failure_mode not in {
            'serial', 'cached_or_uniform', 'cached_or_harmonic'
        }:
            raise ValueError(
                'distributed replicated power parallel failure mode must be '
                'serial, cached_or_uniform, or cached_or_harmonic')
        if (
            not np.isfinite(
                self._distributed_replicated_power_deadline_incumbent_tolerance)
            or not 0.0 <= (
                self._distributed_replicated_power_deadline_incumbent_tolerance
            ) < 1.0
        ):
            raise ValueError(
                'deadline incumbent tolerance must lie in [0,1)')
        if self._distributed_replicated_power_max_consecutive_failures < 1:
            raise ValueError(
                'maximum consecutive process failures must be positive')
        self._distributed_replicated_power_executor = (
            ReplicatedPowerProcessExecutor(
                self._distributed_replicated_power_process_workers,
                self._distributed_replicated_power_process_timeout_s,
            )
            if self._distributed_replicated_power_process_parallel_enabled
            else None
        )
        self._distributed_replicated_power_executor_warmup_time_s = 0.0
        self._distributed_replicated_power_executor_warmup_failed = False
        self._distributed_replicated_power_consecutive_failures = 0
        self._distributed_replicated_power_inertia = float(np.clip(
            getattr(ma, 'distributed_replicated_power_inertia', 0.0),
            0.0, 0.95))
        self._distributed_replicated_power_history_reserve_pd = float(
            getattr(
                ma,
                'distributed_replicated_power_history_reserve_pd',
                0.0,
            ))
        if (
            not np.isfinite(
                self._distributed_replicated_power_history_reserve_pd)
            or self._distributed_replicated_power_history_reserve_pd < 0.0
            or self._distributed_replicated_power_history_reserve_pd >= 1.0
            or (
                self._distributed_replicated_power_history_reserve_pd > 0.0
                and self._distributed_replicated_power_history_reserve_pd
                <= float(self.cfg.detection.P_FA)
            )
        ):
            raise ValueError(
                'replicated power history reserve P_D must be zero or lie '
                'strictly between P_FA and one')
        self._distributed_replicated_power_history_reserve_deflection = (
            float(minimum_deflection_for_detection_probability(
                np.asarray([
                    self._distributed_replicated_power_history_reserve_pd
                ], dtype=np.float64),
                self.cfg.detection.P_FA,
            )[0])
            if self._distributed_replicated_power_history_reserve_pd > 0.0
            else None
        )
        self._distributed_replicated_power_reuse_relative_tolerance = float(
            getattr(
                ma,
                'distributed_replicated_power_reuse_relative_tolerance',
                0.0,
            ))
        if not 0.0 <= (
            self._distributed_replicated_power_reuse_relative_tolerance
        ) < 1.0:
            raise ValueError(
                'distributed replicated power reuse tolerance must lie '
                'in [0,1)')
        self._composable_certificate_enabled = bool(getattr(
            ma, 'distributed_composable_certificate_enabled', False))
        self._composable_certificate_targetwise_upper_enabled = bool(getattr(
            ma,
            'distributed_composable_certificate_targetwise_upper_enabled',
            False,
        ))
        self._composable_certificate_bits = int(getattr(
            ma, 'distributed_composable_certificate_bits_per_target', 16))
        self._composable_certificate_frame_bits = int(getattr(
            ma, 'distributed_composable_certificate_frame_bits', 32))
        self._composable_certificate_scale = float(getattr(
            ma, 'distributed_composable_certificate_scale', 1.0e-6))
        self._composable_certificate_maximum = float(getattr(
            ma, 'distributed_composable_certificate_max_deflection', 1.0e6))
        self._composable_certificate_max_age = int(getattr(
            ma, 'distributed_composable_certificate_max_age_frames', 5))
        if self._composable_certificate_enabled and (
            self._composable_certificate_bits < 1
            or self._composable_certificate_frame_bits < 1
            or not np.isfinite(self._composable_certificate_scale)
            or self._composable_certificate_scale <= 0.0
            or not np.isfinite(self._composable_certificate_maximum)
            or self._composable_certificate_maximum <= 0.0
            or self._composable_certificate_max_age < 0
        ):
            raise ValueError(
                'composable certificate quantizer/AoI parameters are invalid')
        if (
            self._composable_certificate_targetwise_upper_enabled
            and not self._composable_certificate_enabled
        ):
            raise ValueError(
                'targetwise certificate upper requires composable certificate')
        self._distributed_replicated_power_unknown_target_reserve = float(
            np.clip(getattr(
                ma,
                'distributed_replicated_power_unknown_target_reserve',
                1.0,
            ), 0.0, 1.0))
        self._distributed_target_position_uncertainty_sigma = float(getattr(
            ma, 'distributed_target_position_uncertainty_sigma', 0.0))
        self._distributed_replicated_power_robust_gain_mix = float(getattr(
            ma, 'distributed_replicated_power_robust_gain_mix', 0.0))
        if self._distributed_target_position_uncertainty_sigma < 0.0:
            raise ValueError(
                'distributed_target_position_uncertainty_sigma must be '
                'non-negative')
        if (
            self._distributed_target_position_uncertainty_sigma > 0.0
            and not self.tracking_enabled
        ):
            raise ValueError(
                'target-position uncertainty reconstruction requires '
                'tracking_enabled=true')
        if not 0.0 <= self._distributed_replicated_power_robust_gain_mix <= 1.0:
            raise ValueError(
                'robust replicated-power gain mix must lie in [0,1]')
        if (
            self._distributed_replicated_power_robust_gain_mix > 0.0
            and self._distributed_target_position_uncertainty_sigma <= 0.0
        ):
            raise ValueError(
                'robust replicated-power gain requires a positive '
                'distributed_target_position_uncertainty_sigma')
        self._distributed_replicated_power_local_range_fallback_enabled = bool(
            getattr(
                ma,
                'distributed_replicated_power_local_range_fallback_enabled',
                False,
            ))
        self._analytical_structure_ranking_enabled = bool(getattr(
            ma, 'analytical_structure_ranking_enabled', False))
        if (self._analytical_structure_ranking_enabled
                and not self._analytical_sensing_power_enabled):
            raise ValueError(
                'analytical_structure_ranking_enabled requires '
                'analytical_sensing_power_enabled')
        # D1.10-C (advice 014 §4/§5): coupling-aware structure repair.  Only
        # meaningful when the analytical sensing power LP is the executed path.
        self._bargaining_objective_enabled = bool(getattr(
            ma, 'bargaining_objective_enabled', False))
        if (self._bargaining_objective_enabled
                and not self._analytical_sensing_power_enabled):
            raise ValueError(
                'bargaining_objective_enabled requires '
                'analytical_sensing_power_enabled')
        self._analytical_comm_power_enabled = bool(getattr(
            ma, 'analytical_comm_power_enabled', False))
        if (self._analytical_comm_power_enabled
                and not self._joint_isac_power_enabled):
            raise ValueError(
                'analytical_comm_power_enabled requires '
                'joint_isac_power_enabled')
        self._task_constrained_power_enabled = bool(getattr(
            ma, 'task_constrained_power_enabled', False))
        if (self._task_constrained_power_enabled
                and not self._analytical_sensing_power_enabled):
            raise ValueError(
                'task_constrained_power_enabled requires '
                'analytical_sensing_power_enabled')
        self._analytical_movement_enabled = bool(getattr(
            ma, 'analytical_movement_enabled', False))
        self._distributed_id_movement_enabled = bool(getattr(
            ma, 'distributed_id_movement_enabled', False))
        self._distributed_id_movement_standoff_m = max(0.0, float(getattr(
            ma, 'distributed_id_movement_standoff_m', 0.0)))
        self._distributed_greedy_matching_movement_enabled = bool(getattr(
            ma, 'distributed_greedy_matching_movement_enabled', False))
        self._distributed_greedy_matching_hold_frames = max(1, int(getattr(
            ma, 'distributed_greedy_matching_hold_frames', 150)))
        self._distributed_greedy_matching_prediction_frames = max(0, int(
            getattr(ma, 'distributed_greedy_matching_prediction_frames', 0)))
        self._distributed_movement_anchor_broadcast_enabled = bool(getattr(
            ma, 'distributed_movement_anchor_broadcast_enabled', False))
        self._distributed_movement_anchor_broadcast_period_frames = max(
            1, int(getattr(
                ma,
                'distributed_movement_anchor_broadcast_period_frames',
                5,
            )))
        self._distributed_movement_anchor_max_age_frames = max(0, int(getattr(
            ma, 'distributed_movement_anchor_max_age_frames', 10)))
        self._distributed_bottleneck_matching_movement_enabled = bool(getattr(
            ma, 'distributed_bottleneck_matching_movement_enabled', False))
        self._distributed_bistatic_bottleneck_movement_enabled = bool(getattr(
            ma, 'distributed_bistatic_bottleneck_movement_enabled', False))
        self._distributed_bistatic_complement_exponent = float(np.clip(
            getattr(ma, 'distributed_bistatic_complement_exponent', 1.0),
            0.0, 1.0))
        self._distributed_alternating_optimization_enabled = bool(getattr(
            ma, 'distributed_alternating_optimization_enabled', False))
        self._distributed_ao_inner_radius_m = max(0.0, float(getattr(
            ma, 'distributed_ao_inner_radius_m', 0.0)))
        self._distributed_ao_outer_radius_m = max(0.0, float(getattr(
            ma, 'distributed_ao_outer_radius_m', 225.0)))
        self._distributed_ao_range_frames = max(1, int(getattr(
            ma, 'distributed_ao_range_frames', 2)))
        self._distributed_ao_strategy_frames = max(1, int(getattr(
            ma, 'distributed_ao_strategy_frames', 3)))
        self._distributed_ao_desired_bistatic_angle_deg = float(getattr(
            ma, 'distributed_ao_desired_bistatic_angle_deg', 90.0))
        self._distributed_gain_scheduled_movement_enabled = bool(getattr(
            ma, 'distributed_gain_scheduled_movement_enabled', False))
        self._distributed_gain_scheduled_period_frames = max(2, int(getattr(
            ma, 'distributed_gain_scheduled_period_frames', 2)))
        self._distributed_gain_scheduled_min_log_improvement = max(
            0.0, float(getattr(
                ma,
                'distributed_gain_scheduled_min_log_improvement',
                0.0,
            )))
        configured_gain_selected_nodes = int(getattr(
            ma, 'distributed_gain_scheduled_max_selected_nodes', 0))
        if configured_gain_selected_nodes < 0:
            raise ValueError(
                'distributed gain-scheduled max selected nodes must be '
                'nonnegative')
        self._distributed_gain_scheduled_max_selected_nodes = (
            self.K
            if configured_gain_selected_nodes == 0
            else min(self.K, configured_gain_selected_nodes)
        )
        self._distributed_gain_scheduled_far_range_gate_m = max(
            0.0, float(getattr(
                ma,
                'distributed_gain_scheduled_far_range_gate_m',
                0.0,
            )))
        self._distributed_gain_scheduled_far_assignment_mode = str(getattr(
            ma,
            'distributed_gain_scheduled_far_assignment_mode',
            'local_matching',
        )).strip().lower()
        if self._distributed_gain_scheduled_far_assignment_mode not in {
            'local_matching', 'fixed_id',
        }:
            raise ValueError(
                'distributed gain-scheduled far assignment mode must be '
                'local_matching or fixed_id')
        if self._distributed_alternating_optimization_enabled:
            if not (
                self._distributed_ao_inner_radius_m
                < self._distributed_ao_outer_radius_m
            ):
                raise ValueError(
                    'distributed AO requires inner radius < outer radius')
            if not 0.0 < self._distributed_ao_desired_bistatic_angle_deg < 180.0:
                raise ValueError(
                    'distributed AO desired bistatic angle must lie in (0,180)')
        if (
            self._distributed_alternating_optimization_enabled
            and self._distributed_gain_scheduled_movement_enabled
        ):
            raise ValueError(
                'annular AO and gain-scheduled movement are separate ablations')
        self._distributed_bistatic_tail_gate_ratio = max(0.0, float(getattr(
            ma, 'distributed_bistatic_tail_gate_ratio', 0.0)))
        self._distributed_role_capacity_movement_enabled = bool(getattr(
            ma, 'distributed_role_capacity_movement_enabled', False))
        self._distributed_role_capacity_standoff_m = max(0.0, float(getattr(
            ma, 'distributed_role_capacity_standoff_m', 0.0)))
        self._distributed_gap_coverage_movement_enabled = bool(getattr(
            ma, 'distributed_gap_coverage_movement_enabled', False))
        self._distributed_gap_critical_radius_m = max(
            1.0, float(getattr(
                ma, 'distributed_gap_critical_radius_m', 320.0)))
        self._distributed_gap_capacity = max(1, int(getattr(
            ma, 'distributed_gap_capacity', 2)))
        self._distributed_gap_standoff_m = max(0.0, float(getattr(
            ma, 'distributed_gap_standoff_m', 0.0)))
        self._distributed_gap_pursuit_radius_m = max(1.0, float(getattr(
            ma, 'distributed_gap_pursuit_radius_m', 250.0)))
        self._distributed_gap_deficit_priority = bool(getattr(
            ma, 'distributed_gap_deficit_priority', True))
        self._distributed_gap_tangential_phase = bool(getattr(
            ma, 'distributed_gap_tangential_phase', False))
        self._distributed_gap_tangential_step_rad = max(
            0.0, float(getattr(
                ma, 'distributed_gap_tangential_step_rad', 0.02)))
        self._distributed_gap_nearfield_focus = bool(getattr(
            ma, 'distributed_gap_nearfield_focus', True))
        self._distributed_gap_nearfield_radius_m = max(
            1.0, float(getattr(
                ma, 'distributed_gap_nearfield_radius_m', 300.0)))
        self._distributed_gap_complete_radius_m = max(
            1.0, float(getattr(
                ma, 'distributed_gap_complete_radius_m', 350.0)))
        self._distributed_gap_focus_weight = max(
            1.0, float(getattr(
                ma, 'distributed_gap_focus_weight', 4.0)))
        self._distributed_gap_baseline_envelope_enabled = bool(getattr(
            ma, 'distributed_gap_baseline_envelope_enabled', False))
        self._distributed_gap_primal_dual_candidate_enabled = bool(getattr(
            ma, 'distributed_gap_primal_dual_candidate_enabled', False))
        self._distributed_gap_primal_dual_margin = float(getattr(
            ma, 'distributed_gap_primal_dual_margin', 0.0))
        if (
            not np.isfinite(self._distributed_gap_primal_dual_margin)
            or self._distributed_gap_primal_dual_margin < 0.0
        ):
            raise ValueError(
                'distributed gap primal-dual margin must be finite/non-negative')
        self._distributed_gap_primal_dual_strong_only = bool(getattr(
            ma, 'distributed_gap_primal_dual_strong_only', False))
        self._distributed_role_capacity_reassign_threshold = max(
            0.0, float(getattr(
                ma, 'distributed_role_capacity_reassign_threshold', 0.05)))
        self._distributed_role_capacity_hold_frames = max(1, int(getattr(
            ma, 'distributed_role_capacity_hold_frames', 20)))
        self._distributed_role_capacity_range_frames = max(1, int(getattr(
            ma, 'distributed_role_capacity_range_frames', 12)))
        self._distributed_role_capacity_strategy_frames = max(0, int(getattr(
            ma, 'distributed_role_capacity_strategy_frames', 3)))
        if (
            self._distributed_role_capacity_movement_enabled
            and (
                self._distributed_bistatic_bottleneck_movement_enabled
                or self._distributed_bottleneck_matching_movement_enabled
                or self._distributed_greedy_matching_movement_enabled
                or self._distributed_id_movement_enabled
            )
        ):
            raise ValueError(
                'role-capacity movement is a separate ablation from the '
                'legacy distributed movement modes')
        self._distributed_movement_safety_projection_enabled = bool(getattr(
            ma, 'distributed_movement_safety_projection_enabled', False))
        self._distributed_movement_safety_margin_m = max(0.0, float(getattr(
            ma, 'distributed_movement_safety_margin_m', 0.0)))
        self._distributed_movement_safety_margin_per_age_m = max(
            0.0, float(getattr(
                ma, 'distributed_movement_safety_margin_per_age_m', 0.0)))
        self._distributed_movement_local_assignment_cache_enabled = bool(
            getattr(
                ma,
                'distributed_movement_local_assignment_cache_enabled',
                False,
            ))
        self._distributed_movement_public_max_age_frames = max(0, int(
            getattr(ma, 'distributed_movement_public_max_age_frames', 5)))
        self._distributed_movement_stale_fail_closed = bool(getattr(
            ma, 'distributed_movement_stale_fail_closed', False))
        self._distributed_movement_execute_projected_public_action = bool(
            getattr(
                ma,
                'distributed_movement_execute_projected_public_action',
                False,
            ))
        if self._distributed_alternating_optimization_enabled:
            if not self._distributed_movement_safety_projection_enabled:
                raise ValueError(
                    'distributed AO requires public-view safety projection')
            if not self._distributed_movement_execute_projected_public_action:
                raise ValueError(
                    'distributed AO requires execution of the projected action')
        if self._distributed_gain_scheduled_movement_enabled:
            if not self._distributed_movement_safety_projection_enabled:
                raise ValueError(
                    'gain-scheduled movement requires public-view safety '
                    'projection')
            if not self._distributed_movement_execute_projected_public_action:
                raise ValueError(
                    'gain-scheduled movement requires execution of the '
                    'projected action')
        if self._distributed_role_capacity_movement_enabled:
            if not self._distributed_movement_safety_projection_enabled:
                raise ValueError(
                    'role-capacity movement requires public-view safety '
                    'projection')
            if not self._distributed_movement_execute_projected_public_action:
                raise ValueError(
                    'role-capacity movement requires execution of the '
                    'projected action')
        if self._distributed_gap_coverage_movement_enabled:
            if not self._distributed_movement_safety_projection_enabled:
                raise ValueError(
                    'gap-coverage movement requires public-view safety '
                    'projection')
            if not self._distributed_movement_execute_projected_public_action:
                raise ValueError(
                    'gap-coverage movement requires execution of the '
                    'projected action')
        self._distributed_movement_outside_invariant_recovery_enabled = bool(
            getattr(
                ma,
                'distributed_movement_outside_invariant_recovery_enabled',
                False,
            ))
        self._distributed_movement_outside_invariant_recovery_gain = max(
            0.0,
            float(getattr(
                ma,
                'distributed_movement_outside_invariant_recovery_gain',
                1.0,
            )),
        )
        self._distributed_movement_independently_composable_safety = bool(
            getattr(
                ma,
                'distributed_movement_independently_composable_safety',
                False,
            ))
        self._distributed_movement_analytic_composable_projection_enabled = (
            bool(getattr(
                ma,
                'distributed_movement_analytic_composable_projection_enabled',
                False,
            )))
        if (
            self._distributed_movement_analytic_composable_projection_enabled
            and not self._distributed_movement_independently_composable_safety
        ):
            raise ValueError(
                'analytic composable projection requires independently '
                'composable movement safety')
        if (self._analytical_movement_enabled
                and not self._analytical_structure_ranking_enabled):
            raise ValueError(
                'analytical_movement_enabled requires '
                'analytical_structure_ranking_enabled')
        self._analytical_movement_candidates_enabled = bool(getattr(
            ma, 'analytical_movement_candidates_enabled', False))
        self._intercept_constrained_power_enabled = bool(getattr(
            ma, 'intercept_constrained_power_enabled', False))
        if (self._intercept_constrained_power_enabled
                and not self._analytical_sensing_power_enabled):
            raise ValueError(
                'intercept_constrained_power_enabled requires '
                'analytical_sensing_power_enabled')
        if (self._distributed_primal_dual_power_enabled and any((
                self._intercept_constrained_power_enabled,
                self._task_constrained_power_enabled,
                self._bargaining_objective_enabled))):
            raise ValueError(
                'distributed primal-dual power cannot be combined with an '
                'alternative analytical power objective')
        # Use the unique max-entropy (central-path) dual price in place of the
        # basis-dependent LP vertex for geometry/structure price weighting.
        # Default OFF: preserves historical behaviour; A/B via config flag.
        self._entropic_dual_price_enabled = bool(getattr(
            ma, 'entropic_dual_price_enabled', False))
        self._entropic_dual_tau = float(
            getattr(ma, 'entropic_dual_tau', 0.0))
        self._last_intercept_mu: np.ndarray | None = None
        self._last_intercept_pd_max: float | None = None
        self._last_intercept_eps: float | None = None
        self._last_intercept_infeasible = False
        self._last_intercept_joined = False
        self._last_analytical_power_balance_error = 0.0
        # A RF budget is an upper bound.  Keep the historical absolute
        # balance diagnostic for trace compatibility, but distinguish an
        # actual cap violation from intentional under-use (e.g. covertness).
        self._last_analytical_power_budget_violation_w = 0.0
        self._last_analytical_unused_power_w = 0.0
        self._last_analytical_dual_prices = None
        # D1.1-A audit hook (env var DSH_LEX_AUDIT): per-frame lex diagnostics.
        self._lex_audit_path = (os.environ.get('DSH_LEX_AUDIT', '') or '').strip()
        self._lex_mode = 'none'
        self._lex_t_star = None
        self._last_analytical_gain = None
        self._last_analytical_budget = None
        self._last_bargaining_value = None
        self._isac_total_power_w = max(
            0.0, float(getattr(ua, 'P_isac_total',
                               ua.P_sense + getattr(ma, 'comm_tx_power_w', 0.25))))
        configured_sensing_cap = float(getattr(
            ua, 'P_sense_max', ua.P_sense))
        if (not np.isfinite(configured_sensing_cap)
                or configured_sensing_cap <= 0.0):
            raise ValueError("uav.P_sense_max must be finite and positive")
        self._sensing_power_cap_w = min(
            self._isac_total_power_w, configured_sensing_cap)
        self._comm_power_fraction_min = float(np.clip(getattr(
            ma, 'comm_power_fraction_min', 0.0), 0.0, 1.0))
        self._comm_power_fraction_max = float(np.clip(getattr(
            ma, 'comm_power_fraction_max', 1.0),
            self._comm_power_fraction_min, 1.0))
        self._comm_cross_attention_enabled = (
            self._comm_mode == 'cost_aware'
            and bool(getattr(ma, 'comm_cross_attention_enabled', False))
        )
        strict_u2u_info = (
            self._comm_mode == 'cost_aware'
            and bool(getattr(ma, 'comm_only_neighbor_information', False))
        )
        self._comm_payload_mode = str(getattr(
            ma, 'comm_payload_mode', 'aggregate')).lower()
        if (
            self._distributed_decision_sufficient_comm_enabled
            or self._distributed_decision_sufficient_event_trigger_enabled
            or self._distributed_decision_sufficient_adaptive_bits_enabled
        ):
            raise ValueError(
                'decision-sufficient environment compression requires a '
                'downstream decision certificate; learned aggregate/target '
                'tokens have no certified decoder/action Lipschitz bound, so '
                'the C6 online chain must remain disabled')
        self._comm_target_token_dim = max(1, int(getattr(
            ma, 'comm_target_token_dim', 16)))
        self._comm_payload_dim = (
            self.Q * self._comm_target_token_dim
            if self._comm_payload_mode == 'target_tokens' else 16)
        self._comm_channel_feedback_rate_enabled = bool(getattr(
            ma, 'comm_channel_feedback_rate_enabled', False))
        self._comm_channel_feedback_dim = int(getattr(
            ma, 'comm_channel_feedback_dim', 6))
        if (self._comm_channel_feedback_rate_enabled
                and self._comm_channel_feedback_dim != 6):
            raise ValueError(
                'comm_channel_feedback_dim must be 6 for the current '
                'local channel summary')

        # Action space
        self.action_space = ActionSpace(
            v_max=ua.v_max, dt=sc.dt, rng=self.rng,
            learn_roles=bool(getattr(self.cfg.marl, 'learn_roles', False)),
            dp_parameterization=str(getattr(
                self.cfg.marl, 'dp_parameterization', 'radial_clip')),
        )

        # Observation builder
        self.obs_builder = ObservationBuilder(
            K=self.K, Q=self.Q, area_size=self.area_size, height=self.height,
            use_relative_features=getattr(ma, 'rel_features', False),
            use_p0_info=(getattr(ma, 'use_p0_sinr_gated', False)
                         and not strict_u2u_info),
            expose_neighbor_state=not strict_u2u_info,
            use_comm_tokens=self._comm_cross_attention_enabled,
            comm_num_rate_levels=len(getattr(
                ma, 'comm_rate_bits_per_dim', [0, 4, 8, 16])),
            comm_rate_metadata_denominator=getattr(
                ma, 'comm_rate_metadata_denominator', None),
            comm_deadline_s=float(getattr(ma, 'comm_deadline_s', 0.005)),
            comm_payload_mode=self._comm_payload_mode,
            comm_target_token_dim=self._comm_target_token_dim,
            use_channel_feedback=(
                self._comm_channel_feedback_rate_enabled),
            channel_feedback_dim=self._comm_channel_feedback_dim,
        )

        # Deflection computer
        self.deflection_computer = DeflectionComputer(
            fc=ot.fc, delta_f=ot.delta_f, T_sym=ot.T_sym,
            M=ot.M, N=ot.N, kT=ch.kT, B=ot.B, NF_dB=ch.NF,
            P_sense=ua.P_sense, P_report=ua.P_report,
            ric_K=ch.ric_K, rcs=ta.rcs, g_min=de.g_min,
            rng=self.rng,
            g_tx_dBi=ot.g_tx_dBi, g_rx_dBi=ot.g_rx_dBi, n_cpi=ot.n_cpi,
            c_det=float(getattr(de, 'c_det', 1.0)),
            control_frame_s=sc.dt,
            use_los_prob=getattr(ch, 'use_los_prob', False),
            los_a=getattr(ch, 'los_a', 4.88), los_b=getattr(ch, 'los_b', 0.43),
            eta_los_dB=getattr(ch, 'eta_los_dB', 0.1),
            eta_nlos_dB=getattr(ch, 'eta_nlos_dB', 21.0),
            use_swerling=getattr(ch, 'use_swerling', False),
            use_report_link=self.ground_communication_enabled,
            dd_gain_mode=str(getattr(de, 'dd_gain_mode', 'binary')),
            sync_delay_error_bins=float(getattr(
                ch, 'sync_delay_error_bins', 0.0)),
            sync_doppler_error_bins=float(getattr(
                ch, 'sync_doppler_error_bins', 0.0)),
        )

        # File-backed configs validate one weight per target.  Programmatic
        # smoke/test configs historically changed Q without rebuilding the
        # list, so retain that compatibility path but never let truncation
        # silently change the total reward/detection scale.
        if len(ta.omega_q) >= self.Q:
            omega_q = np.asarray(ta.omega_q[:self.Q], dtype=np.float64)
            omega_sum = float(np.sum(omega_q))
            if not np.all(np.isfinite(omega_q)) or omega_sum <= 0.0:
                raise ValueError(
                    "target omega_q prefix must be finite with positive sum")
            omega_q = omega_q / omega_sum
        else:
            omega_q = np.ones(self.Q, dtype=np.float64) / self.Q

        # Inner P0 solver
        self.inner_solver = InnerSolver(
            K_q_max=de.K_q_max, B_q=de.B_q,
            capacity_per_rx=(p0.capacity_per_rx
                             if self.ground_communication_enabled else 10**12),
            latency_max=(p0.latency_max
                         if self.ground_communication_enabled else float('inf')),
            omega_q=omega_q,
            P_FA=de.P_FA, P_D_min=de.P_D_min,
        )

        # Reward computer
        self.reward_computer = RewardComputer(
            omega_q=omega_q,
            P_FA=de.P_FA,
            lambda_report=ma.lambda_report,
            eta_mc=ma.eta_mc,
            alpha_pd=getattr(ma, 'alpha_pd', 0.0),
            lambda_tail=getattr(ma, 'lambda_tail', 0.0),
            utility_mode=getattr(ma, 'reward_utility_mode', 'log'),
            concave_kappa=getattr(ma, 'reward_concave_kappa', 1.0),
        )

        # Auditable coordination shaping. Stage zero computes diagnostics but
        # contributes no reward, preserving historical checkpoints/configs.
        self._coord_reward_enabled = bool(getattr(
            ma, 'coord_reward_enabled', False))
        self._coord_reward_stage = int(getattr(ma, 'coord_reward_stage', 0))
        if not 0 <= self._coord_reward_stage <= 4:
            raise ValueError('coord_reward_stage must be between 0 and 4')
        self._coord_reward_ema_alpha = float(np.clip(getattr(
            ma, 'coord_reward_ema_alpha', 0.20), 0.0, 1.0))
        self._coord_reward_kwargs = {
            'steady_floor': float(getattr(
                ma, 'coord_reward_steady_floor', 0.80)),
            'weak3_floor': float(getattr(
                ma, 'coord_reward_weak3_floor', 0.70)),
            'worst_floor': float(getattr(
                ma, 'coord_reward_worst_floor', 0.60)),
            'worst_weight': float(getattr(
                ma, 'coord_reward_worst_weight', 1.0)),
            'duplicate_weight': float(getattr(
                ma, 'coord_reward_duplicate_weight', 0.01)),
            'weak3_weight': float(getattr(
                ma, 'coord_reward_weak3_weight', 0.50)),
            'steady_weight': float(getattr(
                ma, 'coord_reward_steady_weight', 0.25)),
            'min_move_m': float(getattr(
                ma, 'coord_reward_min_move_m', 1e-6)),
        }

        # Potential-based distance shaping (optional; guides UAVs toward targets
        # to overcome the sparse detection reward). Policy-invariant in theory.
        self.use_distance_shaping = getattr(ma, 'use_distance_shaping', False)
        self.shape_w = getattr(ma, 'shape_w', 0.01)
        self.gamma_shape = ma.gamma

        # Constraint checker
        self.constraint_checker = ConstraintChecker(
            d_safe=ua.d_safe, P_D_min=de.P_D_min,
            area_size=self.area_size,
        )

        # Role assignment mode: when False (default), the policy does NOT choose
        # tx/rx — deflection is role-agnostic and the inner P0 solver assigns roles
        # under a one-role-per-UAV constraint, guaranteeing valid tx-rx pairing
        # whenever geometry allows (fixes the all-same-role argmax collapse).
        self.learn_roles = bool(getattr(self.cfg.marl, 'learn_roles', False))
        self._multistatic_subslot_enabled = bool(getattr(
            self.cfg.marl, 'multistatic_subslot_enabled', False))
        if self._multistatic_subslot_enabled and self.learn_roles:
            raise ValueError(
                'multistatic sub-slots require learn_roles=false; endpoint '
                'roles are scheduled by P0 rather than the actor')
        self._distributed_target_commitment_enabled = bool(getattr(
            self.cfg.marl, 'distributed_target_commitment_enabled', False))
        self._distributed_target_commitment_topk = max(1, int(getattr(
            self.cfg.marl, 'distributed_target_commitment_topk', 1)))
        self._distributed_target_commitment_require_receiver = bool(getattr(
            self.cfg.marl,
            'distributed_target_commitment_require_receiver', True))
        self._distributed_target_commitment_mode = str(getattr(
            self.cfg.marl,
            'distributed_target_commitment_mode', 'hard')).strip().lower()
        if self._distributed_target_commitment_mode not in {'hard', 'soft'}:
            raise ValueError(
                'distributed_target_commitment_mode must be hard or soft')
        self._distributed_target_commitment_soft_floor = float(np.clip(getattr(
            self.cfg.marl,
            'distributed_target_commitment_soft_floor', 0.25), 0.0, 1.0))
        self._distributed_target_commitment_uncertainty_relief = float(
            np.clip(getattr(
                self.cfg.marl,
                'distributed_target_commitment_uncertainty_relief', 1.0),
                0.0, 1.0))
        self._distributed_target_commitment_source = str(getattr(
            self.cfg.marl,
            'distributed_target_commitment_source',
            'sensing_power')).strip().lower()
        if self._distributed_target_commitment_source not in {
                'sensing_power', 'persistent_sensing', 'sent_token', 'qpd'}:
            raise ValueError(
                'distributed_target_commitment_source must be '
                'sensing_power, persistent_sensing, sent_token or qpd')
        self._distributed_target_commitment_min_hold_frames = max(
            0, int(getattr(
                self.cfg.marl,
                'distributed_target_commitment_min_hold_frames', 0)))
        self._distributed_target_commitment_handover_frames = max(
            0, int(getattr(
                self.cfg.marl,
                'distributed_target_commitment_handover_frames', 0)))
        self._distributed_target_commitment_max_age_frames = max(
            1, int(getattr(
                self.cfg.marl,
                'distributed_target_commitment_max_age_frames',
                getattr(self.cfg.marl, 'comm_message_ttl_frames', 5))))
        self._qpd_enabled = bool(getattr(ma, 'qpd_isac_enabled', False))
        self._qpd_qos_floor = float(np.clip(getattr(
            ma, 'qpd_qos_floor', 0.60), 0.0, 1.0))
        self._qpd_queue_step = max(
            0.0, float(getattr(ma, 'qpd_queue_step', 0.25)))
        self._qpd_queue_max = max(
            1e-6, float(getattr(ma, 'qpd_queue_max', 4.0)))
        self._qpd_primal_step = max(
            0.0, float(getattr(ma, 'qpd_primal_step', 1.0)))
        self._qpd_dual_step = max(
            0.0, float(getattr(ma, 'qpd_dual_step', 0.25)))
        self._qpd_rounds = max(1, int(getattr(ma, 'qpd_rounds', 2)))
        self._qpd_row_capacity = float(np.clip(
            getattr(ma, 'qpd_row_capacity', 2.0), 0.0, max(self.Q, 1)))
        self._qpd_target_capacity = max(
            0.0, float(getattr(ma, 'qpd_target_capacity', 2.0)))
        self._qpd_price_max = max(
            1e-6, float(getattr(ma, 'qpd_price_max', 4.0)))
        self._qpd_primal_exploration_floor = max(
            0.0, float(getattr(
                ma, 'qpd_primal_exploration_floor', 0.02)))
        self._qpd_peer_deficit_gain = max(
            0.0, float(getattr(ma, 'qpd_peer_deficit_gain', 2.0)))
        self._qpd_send_threshold = max(
            0.0, float(getattr(ma, 'qpd_send_threshold', 0.05)))
        self._qpd_bid_change_weight = max(
            0.0, float(getattr(ma, 'qpd_bid_change_weight', 1.0)))
        self._qpd_power_cost = max(
            0.0, float(getattr(ma, 'qpd_power_cost', 0.02)))
        self._qpd_comm_cost = max(
            0.0, float(getattr(ma, 'qpd_comm_cost', 0.01)))
        self._qpd_switch_cost = max(
            0.0, float(getattr(ma, 'qpd_switch_cost', 0.02)))
        self._qpd_distance_scale_m = max(1e-6, float(getattr(
            ma, 'qpd_capability_distance_scale_m', 150.0)))
        self._qpd_commitment_threshold = float(np.clip(getattr(
            ma, 'qpd_commitment_threshold', 0.25), 0.0, 1.0))
        self._qpd_override_token_mask = bool(getattr(
            ma, 'qpd_override_token_mask', True))
        self._qpd_overwrite_protocol_header = bool(getattr(
            ma, 'qpd_overwrite_protocol_header', True))
        self._qpd_override_sensing_weights = bool(getattr(
            ma, 'qpd_override_sensing_weights', True))
        self._hyperedge_enabled = bool(getattr(
            ma, 'hyperedge_negotiation_enabled', False))
        self._hyperedge_acceleration = (
            create_hyperedge_acceleration_service(getattr(
                ma, 'hyperedge_acceleration_backend', 'numpy'))
        )
        self._deflection_materialization = (
            create_deflection_materialization_service(getattr(
                ma, 'deflection_materialization_backend', 'numpy'))
        )
        self._acceleration_golden_trace_enabled = bool(getattr(
            ma, 'acceleration_golden_trace_enabled', False))
        self._hyperedge_golden_snapshot: Dict[str, object] = {}
        self._hyperedge_share_topk = max(1, min(
            self.Q, int(getattr(ma, 'hyperedge_share_topk', self.Q))))
        self._hyperedge_distance_scale_m = max(1e-6, float(getattr(
            ma, 'hyperedge_distance_scale_m', 150.0)))
        self._hyperedge_capability_mode = str(getattr(
            ma, 'hyperedge_capability_mode',
            'exponential')).strip().lower()
        if self._hyperedge_capability_mode not in {
                'exponential', 'inverse_square'}:
            raise ValueError(
                'hyperedge_capability_mode must be exponential or '
                'inverse_square')
        self._hyperedge_deficit_gain = max(0.0, float(getattr(
            ma, 'hyperedge_deficit_gain', 2.0)))
        self._hyperedge_proxy_floor = max(1e-9, float(getattr(
            ma, 'hyperedge_proxy_floor', 0.25)))
        self._hyperedge_pair_score_mode = str(getattr(
            ma, 'hyperedge_pair_score_mode',
            'endpoint_proxy')).strip().lower()
        if self._hyperedge_pair_score_mode not in {
                'endpoint_proxy', 'physical_reconstructable',
                'budget_reconstructable'}:
            raise ValueError(
                'hyperedge_pair_score_mode must be endpoint_proxy or '
                'physical_reconstructable or budget_reconstructable')
        self._hyperedge_coordination_mode = str(getattr(
            ma, 'hyperedge_coordination_mode',
            'replicated_global')).strip().lower()
        if self._hyperedge_coordination_mode not in {
                'replicated_global', 'reserved_endpoint',
                'refined_endpoint', 'coalition_endpoint'}:
            raise ValueError(
                'hyperedge_coordination_mode must be replicated_global, '
                'reserved_endpoint, refined_endpoint or coalition_endpoint')
        self._hyperedge_state_stream_enabled = bool(getattr(
            ma, 'hyperedge_state_stream_enabled', False))
        self._hyperedge_protocol_only_enabled = bool(getattr(
            ma, 'hyperedge_protocol_only_enabled', False))
        self._hyperedge_state_bits_per_dim = max(0, int(getattr(
            ma, 'hyperedge_state_bits_per_dim', 0)))
        self._hyperedge_protocol_header_bits = max(0, int(getattr(
            ma, 'hyperedge_protocol_header_bits', 0)))
        crc_undetected_max = float(getattr(
            ma, 'comm_undetected_error_probability_max', 1.0e-7))
        crc_bler = float(getattr(
            ma, 'comm_finite_blocklength_target_bler', 1.0e-5))
        if (
            not np.isfinite(crc_undetected_max)
            or crc_undetected_max <= 0.0
            or not np.isfinite(crc_bler)
            or crc_bler <= 0.0
        ):
            raise ValueError('CRC reliability budgets must be positive')
        minimum_crc_bits = max(0, int(np.ceil(np.log2(
            crc_bler / crc_undetected_max))))
        if (
            0 < self._hyperedge_protocol_header_bits < minimum_crc_bits
        ):
            raise ValueError(
                'hyperedge fixed-schema CRC is too short for the '
                'configured undetected-error budget')
        self._hyperedge_beacon_round_robin_period = max(1, int(getattr(
            ma, 'hyperedge_beacon_round_robin_period', 1)))
        self._hyperedge_nearfield_residual_enabled = bool(getattr(
            ma, 'hyperedge_nearfield_residual_enabled', False))
        self._hyperedge_nearfield_residual_range_m = max(1.0e-6, float(
            getattr(ma, 'hyperedge_nearfield_residual_range_m', 32.0)))
        self._hyperedge_nearfield_residual_companding_mu = max(0.0, float(
            getattr(
                ma, 'hyperedge_nearfield_residual_companding_mu', 0.0)))
        self._hyperedge_state_relay_enabled = bool(getattr(
            ma, 'hyperedge_state_relay_enabled', False))
        self._hyperedge_tx_coalition_max = max(1, int(getattr(
            ma, 'hyperedge_tx_coalition_max', 2)))
        self._hyperedge_robust_quantization_enabled = bool(getattr(
            ma, 'hyperedge_robust_quantization_enabled', False))
        self._hyperedge_bistatic_information_ranking_enabled = bool(getattr(
            ma, 'hyperedge_bistatic_information_ranking_enabled', False))
        self._hyperedge_bistatic_information_weight = float(getattr(
            ma, 'hyperedge_bistatic_information_weight', 1.0))
        if (
            not np.isfinite(self._hyperedge_bistatic_information_weight)
            or self._hyperedge_bistatic_information_weight < 0.0
        ):
            raise ValueError('hyperedge bistatic information weight is invalid')
        self._hyperedge_reserved_tx_nodes = tuple(int(node) for node in getattr(
            ma, 'hyperedge_reserved_tx_nodes', ()))
        if any(
            node < 0 or node >= self.K
            for node in self._hyperedge_reserved_tx_nodes
        ):
            raise ValueError('hyperedge_reserved_tx_nodes contains invalid ID')
        # The budget-reconstructable protocol needs only one node-level
        # (x,y,vx,vy) sufficient statistic. It is carried once in a target
        # token packet and expanded locally, rather than repeated Q times.
        hyperedge_base_state_dim = (
            7 if self._hyperedge_nearfield_residual_enabled else 4)
        self._hyperedge_base_state_dim = hyperedge_base_state_dim
        self._hyperedge_protocol_dim = (
            (
                2 * hyperedge_base_state_dim + 1
                if self._hyperedge_state_relay_enabled
                else hyperedge_base_state_dim
            )
            if self._hyperedge_pair_score_mode == 'budget_reconstructable'
            else (7 if self._hyperedge_state_stream_enabled else 3))
        raw_state_field_bits = tuple(int(value) for value in getattr(
            ma, 'hyperedge_state_field_bits', ()))
        if raw_state_field_bits:
            if (
                len(raw_state_field_bits) != self._hyperedge_protocol_dim
                or any(value <= 0 for value in raw_state_field_bits)
            ):
                raise ValueError(
                    'hyperedge_state_field_bits must contain one positive '
                    'precision per protocol field')
            self._hyperedge_state_field_bits = raw_state_field_bits
        else:
            self._hyperedge_state_field_bits = (
                self._hyperedge_state_bits_per_dim,
            ) * self._hyperedge_protocol_dim
        self._hyperedge_state_codec_service_envelope_enabled = bool(getattr(
            ma, 'hyperedge_state_codec_service_envelope_enabled', False))
        configured_service_header = int(getattr(
            ma,
            'hyperedge_state_codec_service_reference_header_bits',
            0,
        ))
        if configured_service_header < 0:
            raise ValueError(
                'hyperedge service reference header must be non-negative')
        self._hyperedge_state_service_reference_header_bits = int(
            configured_service_header
            if configured_service_header > 0
            else self._hyperedge_protocol_header_bits)
        if (
            self._hyperedge_state_service_reference_header_bits
            < self._hyperedge_protocol_header_bits
        ):
            raise ValueError(
                'hyperedge service reference header cannot be shorter than '
                'the physical header')
        self._hyperedge_state_service_delta_bits = int(
            self._hyperedge_protocol_dim
            * self._hyperedge_state_bits_per_dim
            - sum(self._hyperedge_state_field_bits)
        )
        if self._hyperedge_state_codec_service_envelope_enabled:
            if not self._hyperedge_protocol_only_enabled:
                raise ValueError(
                    'hyperedge codec service envelope requires '
                    'protocol-only transport')
            if self._hyperedge_state_service_delta_bits < 0:
                raise ValueError(
                    'hyperedge codec service envelope cannot expand beyond '
                    'the uniform reference packet')
        self._hyperedge_consensus_rounds = max(1, int(getattr(
            ma, 'hyperedge_consensus_rounds', 2)))
        self._hyperedge_assignment_hold_frames = max(1, int(getattr(
            ma, 'hyperedge_assignment_hold_frames', 1)))
        self._hyperedge_min_target_coverage = float(np.clip(getattr(
            ma, 'hyperedge_min_target_coverage', 1.0), 0.0, 1.0))
        self._hyperedge_safety_fallback = bool(getattr(
            ma, 'hyperedge_safety_fallback_enabled', True))
        if self._qpd_enabled:
            if self._comm_mode != 'cost_aware':
                raise ValueError('QPD-ISAC requires cost-aware communication')
            if not self._joint_isac_power_enabled:
                raise ValueError('QPD-ISAC requires joint ISAC power')
            if self._comm_payload_mode != 'target_tokens':
                raise ValueError('QPD-ISAC requires target-token payloads')
            if self._comm_target_token_dim < 5:
                raise ValueError(
                    'QPD-ISAC protocol header requires >=5 dimensions/token')
            if not self._distributed_target_commitment_enabled:
                raise ValueError(
                    'QPD-ISAC requires distributed target commitments')
            if self._distributed_target_commitment_source != 'qpd':
                raise ValueError(
                    'QPD-ISAC requires commitment source=qpd')
        if self._hyperedge_enabled:
            if self._qpd_enabled:
                raise ValueError(
                    'hyperedge negotiation and QPD cannot be enabled together')
            if self._comm_mode != 'cost_aware':
                raise ValueError(
                    'hyperedge negotiation requires cost-aware communication')
            if not self._joint_isac_power_enabled:
                raise ValueError(
                    'hyperedge negotiation requires joint ISAC power')
            if self._comm_payload_mode != 'target_tokens':
                raise ValueError(
                    'hyperedge negotiation requires target-token payloads')
            if (self._hyperedge_pair_score_mode in {
                    'physical_reconstructable', 'budget_reconstructable'}
                    and not self._hyperedge_state_stream_enabled):
                raise ValueError(
                    'physical-reconstructable hyperedge scores require the '
                    'position/velocity state stream')
            if (self._hyperedge_protocol_only_enabled
                    and self._hyperedge_pair_score_mode
                    != 'budget_reconstructable'):
                raise ValueError(
                    'protocol-only hyperedge beacons currently require '
                    'budget_reconstructable scores')
            if (self._hyperedge_protocol_only_enabled
                    and self._hyperedge_state_bits_per_dim <= 0):
                raise ValueError(
                    'protocol-only hyperedge beacons require a positive '
                    'hyperedge_state_bits_per_dim')
            if self._hyperedge_beacon_round_robin_period > 1:
                if not self._hyperedge_protocol_only_enabled:
                    raise ValueError(
                        'round-robin hyperedge beacons require protocol-only '
                        'transport')
                if (
                    self._hyperedge_beacon_round_robin_period
                    > max(0, int(getattr(
                        ma, 'comm_message_ttl_frames', 3)))
                ):
                    raise ValueError(
                        'round-robin hyperedge period exceeds message TTL')
                if (
                    self._hyperedge_beacon_round_robin_period
                    > self._distributed_movement_public_max_age_frames
                ):
                    raise ValueError(
                        'round-robin hyperedge period exceeds movement '
                        'public-age cap')
            if (
                self._hyperedge_nearfield_residual_enabled
                and self._hyperedge_pair_score_mode
                != 'budget_reconstructable'
            ):
                raise ValueError(
                    'near-field residual beacons require '
                    'budget-reconstructable hyperedges')
        if self._distributed_movement_anchor_broadcast_enabled:
            if self._hyperedge_pair_score_mode != 'budget_reconstructable':
                raise ValueError(
                    'movement anchor broadcast requires the '
                    'budget-reconstructable protocol')
            if not self._hyperedge_nearfield_residual_enabled:
                raise ValueError(
                    'movement anchor broadcast requires the three-dimensional '
                    'near-field header')
        if self._distributed_replicated_power_enabled:
            if not self._analytical_sensing_power_enabled:
                raise ValueError(
                    'distributed replicated power requires analytical '
                    'sensing power')
            if (
                not self._hyperedge_enabled
                or self._hyperedge_pair_score_mode
                != 'budget_reconstructable'
            ):
                raise ValueError(
                    'distributed replicated power requires the '
                    'budget-reconstructable hyperedge state channel')
            if (
                self._intercept_constrained_power_enabled
                or self._task_constrained_power_enabled
                or self._bargaining_objective_enabled
            ):
                raise ValueError(
                    'distributed replicated power currently supports only '
                    'ordinary fixed-structure max-min power')
        if (
            self._composable_certificate_enabled
            and not self._distributed_replicated_power_enabled
        ):
            raise ValueError(
                'composable certificate requires distributed replicated power')
        self._p0_maxmin_pairing_enabled = bool(getattr(
            ma, "p0_maxmin_pairing_enabled", False))
        self._p0_maxmin_bypass_commitment_filter = bool(getattr(
            ma, "p0_maxmin_bypass_commitment_filter", False))
        self._p0_maxmin_pairing_hold_frames = max(1, int(getattr(
            ma, "p0_maxmin_pairing_hold_frames", 1)))
        self._p0_maxmin_local_fusion_enabled = bool(getattr(
            ma, "p0_maxmin_local_fusion_enabled", False))
        self._p0_maxmin_event_triggered_enabled = bool(getattr(
            ma, "p0_maxmin_event_triggered_enabled", False))
        self._p0_maxmin_event_qos_enabled = bool(getattr(
            ma, "p0_maxmin_event_qos_enabled", True))
        self._p0_topology_min_change_repair_enabled = bool(getattr(
            ma, "p0_topology_min_change_repair_enabled", False))
        self._p0_budget_coupled_structure_enabled = bool(getattr(
            ma, "p0_budget_coupled_structure_enabled", False))
        self._p0_budget_coupled_time_limit_s = float(getattr(
            ma, "p0_budget_coupled_time_limit_s", 5.0))
        self._p0_budget_coupled_secondary_enabled = bool(getattr(
            ma, "p0_budget_coupled_secondary_enabled", True))
        self._p0_budget_coupled_solver_mode = str(getattr(
            ma, "p0_budget_coupled_solver_mode", "milp")).strip().lower()
        if self._p0_budget_coupled_solver_mode not in {
                "milp", "enumerated_role_ceiling",
                "satisficing_legacy_then_enumerated"}:
            raise ValueError(
                "p0_budget_coupled_solver_mode must be 'milp' or "
                "'enumerated_role_ceiling' or "
                "'satisficing_legacy_then_enumerated'")
        if self._p0_budget_coupled_time_limit_s <= 0.0:
            raise ValueError(
                "p0_budget_coupled_time_limit_s must be positive")
        if (self._p0_budget_coupled_structure_enabled
                and not self._p0_maxmin_local_fusion_enabled):
            raise ValueError(
                "budget-coupled P0 requires local-fusion max-min pairing")
        if (self._p0_budget_coupled_structure_enabled
                and not self._analytical_sensing_power_enabled):
            raise ValueError(
                "budget-coupled P0 requires analytical sensing power")
        if (self._distributed_target_commitment_enabled
                and not self._joint_isac_power_enabled):
            raise ValueError(
                'distributed target commitments require joint ISAC power')
        # B6: P0 ranks on fused belief (deployable) vs true geometry (oracle).
        self.p0_uses_belief = bool(getattr(self.cfg.marl, 'p0_uses_belief', False))
        # B7: gate belief update by a Bernoulli(P_D) detection event.
        self.belief_detection_sampling = bool(getattr(self.cfg.marl, 'belief_detection_sampling', False))
        self._belief_expected_detection_information_enabled = bool(getattr(
            self.cfg.marl,
            'belief_expected_detection_information_enabled',
            False,
        ))
        self._belief_expected_detection_information_floor = float(getattr(
            self.cfg.marl,
            'belief_expected_detection_information_floor',
            1.0e-3,
        ))
        if not 0.0 < self._belief_expected_detection_information_floor <= 1.0:
            raise ValueError(
                'expected detection-information floor must lie in (0,1]')
        if (
            self.belief_detection_sampling
            and self._belief_expected_detection_information_enabled
        ):
            raise ValueError(
                'Bernoulli detection sampling and expected detection '
                'information are mutually exclusive')
        self._belief_measurement_model = str(getattr(
            self.cfg.marl,
            'belief_measurement_model',
            'cartesian',
        )).strip().lower()
        if self._belief_measurement_model not in {
            'cartesian', 'bistatic_range_doppler'
        }:
            raise ValueError(
                'belief_measurement_model must be cartesian or '
                'bistatic_range_doppler')
        self._belief_bistatic_crlb_efficiency = float(getattr(
            self.cfg.marl, 'belief_bistatic_crlb_efficiency', 4.0))
        self._belief_bistatic_min_effective_deflection = float(getattr(
            self.cfg.marl,
            'belief_bistatic_min_effective_deflection',
            1.0e-3,
        ))
        if (
            not np.isfinite(self._belief_bistatic_crlb_efficiency)
            or self._belief_bistatic_crlb_efficiency < 1.0
            or not np.isfinite(
                self._belief_bistatic_min_effective_deflection)
            or self._belief_bistatic_min_effective_deflection <= 0.0
        ):
            raise ValueError('bistatic tracker CRLB parameters are invalid')
        if (
            self._belief_measurement_model == 'bistatic_range_doppler'
            and self._belief_expected_detection_information_enabled
        ):
            raise ValueError(
                'bistatic measurement already uses edge Deflection; expected '
                'Cartesian detection information would double count it')
        if (
            self._belief_measurement_model == 'bistatic_range_doppler'
            and self._passive_multireceiver_belief_update_enabled
        ):
            raise ValueError(
                'bistatic tracker currently requires selected receiver-owner '
                'updates, not passive multireceiver belief updates')
        if self._hyperedge_bistatic_information_ranking_enabled and (
            self._belief_measurement_model != 'bistatic_range_doppler'
            or self._hyperedge_coordination_mode != 'reserved_endpoint'
        ):
            raise ValueError(
                'bistatic information ranking requires the bistatic tracker '
                'and reserved_endpoint coordination')
        # B8: neighbor belief fusion via multi-head attention + CI.
        self.neighbor_belief_fusion = bool(getattr(self.cfg.marl, 'neighbor_belief_fusion', False))
        # B3: Uncertainty-aware P0 scoring
        self.p0_beta_uncertainty = float(getattr(self.cfg.marl, 'p0_beta_uncertainty', 0.0))
        self.p0_eta_aoi = float(getattr(self.cfg.marl, 'p0_eta_aoi', 0.0))
        self._belief_fusion_module = None
        if self.neighbor_belief_fusion:
            import torch as _torch
            from uav_isac.environment.belief_fusion import NeighborBeliefFusion
            self._belief_fusion_module = NeighborBeliefFusion(
                Q=self.Q, D=64, num_heads=4)
            # Will be moved to device later if needed; for now CPU is fine for env

        # Layer 2: Trust manager for gated CI fusion
        self.trust_gate_enabled = bool(getattr(self.cfg.marl, 'trust_gate_enabled', False))
        self._trust_manager: Optional[TrustManager] = None
        if self.trust_gate_enabled:
            self._trust_manager = TrustManager(
                K=self.K, Q=self.Q,
                disagreement_threshold=float(getattr(self.cfg.marl, 'trust_disagreement_threshold', 6.0)),
                aoi_max=float(getattr(self.cfg.marl, 'trust_aoi_max', 50.0)),
                weight_max=float(getattr(self.cfg.marl, 'trust_weight_max', 0.6)),
                local_weight_min=float(getattr(self.cfg.marl, 'trust_local_weight_min', 0.25)),
                ema_rho=float(getattr(self.cfg.marl, 'trust_ema_rho', 0.1)),
                quarantine_nis_ratio=float(getattr(self.cfg.marl, 'trust_quarantine_nis_ratio', 1.5)),
                quarantine_duration=int(getattr(self.cfg.marl, 'trust_quarantine_duration', 10)),
            )

        # State objects (created in reset)
        self.uavs: List[UAV] = []
        self.targets: List[Target] = []
        self.belief_mgr: Optional[BeliefManager] = None
        self.fc_position: Optional[np.ndarray] = None
        self.t: int = 0
        self.prev_P_D: Optional[np.ndarray] = None
        self._coord_pd_ema: Optional[np.ndarray] = None
        # P1 FIX (2026-07-14): per-UAV LOCAL detection confidence.
        # Previously prev_P_D was the global fused P_D_q broadcast to all UAVs
        # for free — violating decentralized execution. Now each UAV sees only
        # its own local P_D contribution: P_D computed from deflection of pairs
        # where THIS UAV is tx or rx.
        self.prev_P_D_local: Dict[int, np.ndarray] = {}
        # Assignment hold: cache P0 solution to reduce reward non-stationarity
        self._cached_p0_solution = None
        self._last_solve_frame = -1
        self._assignment_switched = False
        self._last_p0_solve_time_s = 0.0
        self._prev_obs: Dict[int, np.ndarray] = {}  # per-agent previous obs for history stack
        self._comm_msgs: Dict[int, np.ndarray] = {}  # per-agent latent payloads
        # Cost-aware learned communication uses explicit outbound actions and a
        # receiver-specific inbox. Legacy `on` mode continues to use _comm_msgs.
        self._comm_mode = str(getattr(ma, 'learned_comm_mode', 'on'))
        self._pending_comm_messages: Dict[int, np.ndarray] = {}
        self._pending_comm_rates: Dict[int, int] = {}
        self._pending_comm_token_masks: Dict[int, np.ndarray] = {}
        # C6 decision-sufficient per-sender exact bit plan (adaptive-bits stage):
        # filled by ``_apply_decision_sufficient_silence`` for senders that DO
        # transmit, consumed exactly once by the transmit call, then cleared.
        self._pending_decision_sufficient_bits: Dict[int, int] = {}
        self._last_sent_comm_token_masks: Dict[int, np.ndarray] = {}
        self._last_sent_comm_target_claims: Dict[int, np.ndarray] = {}
        self._persistent_commitment_mask = np.zeros(
            (self.K, self.Q), dtype=bool)
        self._persistent_commitment_old_mask = np.zeros(
            (self.K, self.Q), dtype=bool)
        self._persistent_commitment_last_switch = np.full(
            self.K, FRAME_NOT_APPLICABLE, dtype=np.int64)
        self._persistent_commitment_last_seen = np.full(
            self.K, FRAME_NOT_APPLICABLE, dtype=np.int64)
        self._persistent_commitment_handover_remaining = np.zeros(
            self.K, dtype=np.int64)
        self._persistent_commitment_effective_mask = np.zeros(
            (self.K, self.Q), dtype=bool)
        self._persistent_commitment_resolved_frame = -1
        self._persistent_commitment_metrics: Dict[str, object] = {}
        self._qpd_queue = np.zeros((self.K, self.Q), dtype=np.float64)
        self._qpd_previous_queue = np.zeros(
            (self.K, self.Q), dtype=np.float64)
        self._qpd_bid = np.zeros((self.K, self.Q), dtype=np.float64)
        self._qpd_previous_bid = np.zeros(
            (self.K, self.Q), dtype=np.float64)
        initial_qpd = min(
            1.0, self._qpd_row_capacity / max(self.Q, 1))
        self._qpd_primal = np.full(
            (self.K, self.Q), initial_qpd, dtype=np.float64)
        self._qpd_previous_primal = self._qpd_primal.copy()
        self._qpd_target_price = np.zeros(
            (self.K, self.Q), dtype=np.float64)
        self._qpd_capability = np.zeros(
            (self.K, self.Q), dtype=np.float64)
        self._qpd_age = np.zeros((self.K, self.Q), dtype=np.int64)
        self._qpd_send_mask = np.zeros(
            (self.K, self.Q), dtype=bool)
        self._pending_qpd_protocol: Dict[int, np.ndarray] = {}
        self._qpd_commitment_mask = np.zeros(
            (self.K, self.Q), dtype=bool)
        self._qpd_received_primal = np.zeros(
            (self.K, self.K, self.Q), dtype=np.float64)
        self._qpd_received_queue = np.zeros(
            (self.K, self.K, self.Q), dtype=np.float64)
        self._qpd_received_last_seen = np.full(
            (self.K, self.K, self.Q), FRAME_NOT_APPLICABLE, dtype=np.int64)
        self._qpd_metrics: Dict[str, object] = {}
        self._qpd_last_submission_frame = -1
        self._pending_hyperedge_protocol: Dict[int, np.ndarray] = {}
        self._pending_hyperedge_protocol_frame: Dict[int, int] = {}
        self._hyperedge_local_offer = np.zeros(
            (self.K, self.Q, self._hyperedge_protocol_dim),
            dtype=np.float64)
        self._hyperedge_received_offer = np.zeros(
            (self.K, self.K, self.Q, self._hyperedge_protocol_dim),
            dtype=np.float64)
        self._hyperedge_received_last_seen = np.full(
            (self.K, self.K, self.Q), FRAME_NOT_APPLICABLE, dtype=np.int64)
        self._hyperedge_consensus_streak = np.zeros(
            (self.K, self.K, self.Q), dtype=np.int64)
        self._hyperedge_selected_set: Tuple[Tuple[int, int, int], ...] = tuple()
        self._hyperedge_last_update_frame = FRAME_NOT_APPLICABLE
        self._hyperedge_metrics: Dict[str, object] = {}
        self._hyperedge_public_gain_views = np.zeros(
            (self.K, self.K, self.Q), dtype=np.float64)
        self._hyperedge_public_full_views = np.zeros(
            self.K, dtype=bool)
        self._composable_certificate_local_lower = np.zeros(
            (self.K, self.Q), dtype=np.float64)
        self._composable_certificate_gain_views = np.zeros(
            (self.K, self.K, self.Q), dtype=np.float64)
        self._composable_certificate_gain_upper_views = np.zeros(
            (self.K, self.K, self.Q), dtype=np.float64)
        self._composable_certificate_local_dual_upper = np.zeros(
            self.K, dtype=np.float64)
        self._composable_certificate_local_targetwise_upper = np.zeros(
            (self.K, self.Q), dtype=np.float64)
        self._composable_certificate_local_frame = np.full(
            self.K, -1, dtype=np.int64)
        self._composable_certificate_last_sent_lower = np.zeros(
            (self.K, self.Q), dtype=np.float64)
        self._composable_certificate_last_sent_frame = np.full(
            self.K, -1, dtype=np.int64)
        self._composable_certificate_last_sent_dual_upper = np.zeros(
            self.K, dtype=np.float64)
        self._composable_certificate_last_sent_targetwise_upper = np.zeros(
            (self.K, self.Q), dtype=np.float64)
        self._composable_certificate_received_lower = np.zeros(
            (self.K, self.K, self.Q), dtype=np.float64)
        self._composable_certificate_received_frame = np.full(
            (self.K, self.K), -1, dtype=np.int64)
        self._composable_certificate_received_dual_upper = np.zeros(
            (self.K, self.K), dtype=np.float64)
        self._composable_certificate_received_targetwise_upper = np.zeros(
            (self.K, self.K, self.Q), dtype=np.float64)
        self._composable_certificate_metrics: Dict[str, object] = {}
        self._reset_owner_posterior_state()
        self._distributed_movement_target = np.full(
            self.K, -1, dtype=np.int64)
        self._distributed_movement_local_assignment = np.full(
            (self.K, self.K), -1, dtype=np.int64)
        self._distributed_movement_last_update = np.full(
            self.K, FRAME_NOT_APPLICABLE, dtype=np.int64)
        self._distributed_movement_anchor_target_xy = np.zeros(
            (self.K, self.K, self.Q, 2), dtype=np.float64)
        self._distributed_movement_anchor_last_seen = np.full(
            (self.K, self.K, self.Q), FRAME_NOT_APPLICABLE, dtype=np.int64)
        self._distributed_bistatic_gate_latch = np.full(
            self.K, -1, dtype=np.int8)
        self._distributed_role_capacity_tx_resp = np.zeros(
            (self.K, self.K, self.Q), dtype=np.int8)
        self._distributed_role_capacity_rx_resp = np.zeros(
            (self.K, self.K, self.Q), dtype=np.int8)
        self._distributed_role_capacity_bottleneck = np.full(
            self.K, np.inf, dtype=np.float64)
        self._distributed_role_capacity_last_reassign = np.full(
            self.K, FRAME_NOT_APPLICABLE, dtype=np.int64)
        self._distributed_gap_tx_resp = np.zeros(
            (self.K, self.K, self.Q), dtype=np.int8)
        self._distributed_gap_rx_resp = np.zeros(
            (self.K, self.K, self.Q), dtype=np.int8)
        self._pending_comm_power_fractions: Dict[int, float] = {}
        self._pending_sensing_weights: Dict[int, np.ndarray] = {}
        self._current_comm_power_w = np.zeros(self.K, dtype=np.float64)
        self._current_sensing_power_w = np.full(
            (self.K, self.Q),
            self._sensing_power_cap_w / max(self.Q, 1), dtype=np.float64)
        self._distributed_replicated_previous_power = (
            self._current_sensing_power_w.copy())
        self._distributed_replicated_local_power_cache = np.zeros(
            (self.K, self.K, self.Q), dtype=np.float64)
        self._distributed_replicated_local_price_cache = np.full(
            (self.K, self.Q), 1.0 / max(self.Q, 1), dtype=np.float64)
        self._distributed_replicated_local_cache_valid = np.zeros(
            self.K, dtype=bool)
        self._isac_sensing_battery_before = None
        self._last_isac_metrics: Dict[str, object] = {}
        self._last_movement_safety_metrics: Dict[str, object] = {}
        # Optional information-equivalent distributed structure surrogate.
        # The trainer assembles this graph only from per-UAV local features;
        # realized detection below continues to use the true physical echo.
        self._external_structure_edge_values: Optional[np.ndarray] = None
        self._external_structure_candidate_mask: Optional[np.ndarray] = None
        self._dynamic_local_search_coordinator = None
        # Optional physical transport for the same frozen surrogate.  Each UAV
        # broadcasts its per-target Tx/Rx endpoint embeddings through the
        # existing cost-aware U2U channel.  A sender's public cache entry is
        # advanced atomically only after every peer receives the broadcast,
        # so all decentralized nodes reconstruct one consistent graph.
        self._structure_student_channel_enabled = False
        self._structure_student_decoder = None
        self._structure_student_endpoint_width = 0
        self._structure_student_bits_per_dim = 8
        self._structure_student_adaptive_min_bits_per_dim = 0
        self._structure_student_min_comm_fraction = 0.01
        self._pending_structure_student_protocol: Dict[int, np.ndarray] = {}
        self._structure_student_public_protocol = np.zeros(
            (self.K, self.Q, 0), dtype=np.float64)
        self._structure_student_public_valid = np.zeros(
            self.K, dtype=bool)
        self._structure_student_public_last_seen = np.full(
            self.K, FRAME_NOT_APPLICABLE, dtype=np.int64)
        # (due_frame, sender, protocol, sent_frame)
        self._structure_student_mailbox: list = []
        self._structure_student_metrics: Dict[str, object] = {}
        self._received_comm_msgs: Dict[int, Dict[int, np.ndarray]] = {}
        self._received_comm_meta: Dict[int, Dict[int, dict]] = {}
        self._comm_message_ttl_frames = max(
            0, int(getattr(ma, 'comm_message_ttl_frames', 5)))
        # (due_frame, receiver, sender, message, metadata)
        self._comm_mailbox: list = []
        self._last_comm_stats = CommunicationStepStats()
        self._inter_uav_comm = None
        self._nominal_comm_deadline_s = float(getattr(
            ma, 'comm_deadline_s', 0.005))
        self._nominal_comm_snr_threshold_db = float(getattr(
            ma, 'comm_snr_threshold_db', 0.0))
        self._active_comm_deadline_s = self._nominal_comm_deadline_s
        self._active_comm_snr_threshold_db = (
            self._nominal_comm_snr_threshold_db)
        self._active_comm_channel_profile = -1
        self._evidence_packet_layout = None
        self._evidence_service_envelope_layout = None
        self._evidence_confidence_quantizer = None
        if self._comm_mode == 'cost_aware':
            self._inter_uav_comm = InterUAVCommunicationModel(
                rate_bits_per_dim=list(getattr(
                    ma, 'comm_rate_bits_per_dim', [0, 4, 8, 16])),
                header_bits=int(getattr(ma, 'comm_header_bits', 64)),
                bandwidth_hz=float(getattr(ma, 'comm_bandwidth_hz', 1.0e5)),
                deadline_s=float(getattr(ma, 'comm_deadline_s', 0.005)),
                processing_delay_s=float(getattr(
                    ma, 'comm_processing_delay_s', 2.0e-4)),
                snr_threshold_db=float(getattr(ma, 'comm_snr_threshold_db', 0.0)),
                antenna_gain_dbi=float(getattr(ma, 'comm_antenna_gain_dbi', 0.0)),
                carrier_hz=ot.fc,
                tx_power_w=float(getattr(ma, 'comm_tx_power_w', 0.25)),
                kT=ch.kT,
                noise_figure_db=ch.NF,
                dt=sc.dt,
                message_dim=self._comm_payload_dim,
                finite_blocklength_enabled=bool(getattr(
                    ma, 'comm_finite_blocklength_enabled', False)),
                finite_blocklength_target_bler=float(getattr(
                    ma, 'comm_finite_blocklength_target_bler', 1.0e-5)),
                finite_blocklength_max_channel_uses=int(getattr(
                    ma, 'comm_finite_blocklength_max_channel_uses',
                    100_000_000)),
                finite_blocklength_sample_errors=bool(getattr(
                    ma, 'comm_finite_blocklength_sample_errors', True)),
                finite_blocklength_coding_snr_margin_db=float(getattr(
                    ma, 'comm_finite_blocklength_coding_snr_margin_db', 0.0)),
                snr_shadowing_std_db=float(getattr(
                    ma, 'comm_snr_shadowing_std_db', 0.0)),
                snr_shadowing_correlation=float(getattr(
                    ma, 'comm_snr_shadowing_correlation', 0.0)),
                burst_loss_enabled=bool(getattr(
                    ma, 'comm_burst_loss_enabled', False)),
                burst_good_to_bad_probability=float(getattr(
                    ma, 'comm_burst_good_to_bad_probability', 0.0)),
                burst_bad_to_good_probability=float(getattr(
                    ma, 'comm_burst_bad_to_good_probability', 1.0)),
                burst_good_drop_probability=float(getattr(
                    ma, 'comm_burst_good_drop_probability', 0.0)),
                burst_bad_drop_probability=float(getattr(
                    ma, 'comm_burst_bad_drop_probability', 1.0)),
                rng=self.rng,
            )
            if (
                self._inter_uav_comm.finite_blocklength_enabled
                and self._analytical_comm_power_enabled
                and bool(getattr(ma, 'analytical_comm_optimal_bw', True))
            ):
                raise ValueError(
                    'finite-blocklength analytical L0 currently requires '
                    'analytical_comm_optimal_bw=false; the Shannon KKT '
                    'bandwidth allocator is not valid under dispersion')
        if self._detection_fusion_mode == 'u2u_distributed':
            self._evidence_topk = max(
                1, int(getattr(ma, 'evidence_packet_topk', 1)))
            self._evidence_ambiguity_top2_enabled = bool(getattr(
                ma, 'evidence_packet_ambiguity_top2_enabled', False))
            self._evidence_ambiguity_ratio = float(getattr(
                ma, 'evidence_packet_ambiguity_ratio', 0.5))
            self._evidence_ambiguity_min_deflection = float(getattr(
                ma, 'evidence_packet_ambiguity_min_deflection', 0.0))
            if not 0.0 <= self._evidence_ambiguity_ratio <= 1.0:
                raise ValueError(
                    'evidence_packet_ambiguity_ratio must lie in [0,1]')
            if self._evidence_ambiguity_min_deflection < 0.0:
                raise ValueError(
                    'evidence_packet_ambiguity_min_deflection must be '
                    'non-negative')
            self._evidence_owner_aware = bool(getattr(
                ma, 'evidence_packet_owner_aware', True))
            self._evidence_llr_bits = int(getattr(
                ma, 'evidence_packet_llr_bits', 8))
            self._evidence_confidence_bits = int(getattr(
                ma, 'evidence_packet_confidence_bits', 2))
            self._evidence_clip_max = float(getattr(
                ma, 'evidence_packet_clip_max', 0.0))
            self._evidence_threshold = float(getattr(
                ma, 'evidence_packet_standardized_threshold', 0.0))
            self._evidence_mc_draws = max(
                1, int(getattr(ma, 'evidence_packet_mc_draws', 2048)))
            self._evidence_mc_seed = int(getattr(
                ma, 'evidence_packet_mc_seed', 20260725))
            self._evidence_content_mode = str(getattr(
                ma, 'evidence_packet_content_mode', 'normal')).strip().lower()
            self._evidence_calibration_profile = str(getattr(
                ma, 'evidence_packet_calibration_profile', '')).strip()
            if self._evidence_llr_bits <= 0:
                raise ValueError(
                    'evidence_packet_llr_bits must be positive')
            if self._evidence_clip_max <= 0.0:
                raise ValueError(
                    'u2u_distributed requires a positive calibrated '
                    'evidence_packet_clip_max')
            if not self._evidence_calibration_profile:
                raise ValueError(
                    'u2u_distributed requires an explicit '
                    'evidence_packet_calibration_profile')
            if self._evidence_content_mode not in {
                    'normal', 'standardized_score', 'zero', 'value_roll'}:
                raise ValueError(
                    'evidence_packet_content_mode must be '
                    'normal, standardized_score, zero, or value_roll')
            evidence_header_bits = int(getattr(
                ma, 'evidence_fixed_schema_header_bits', 0))
            evidence_timestamp_bits = int(getattr(
                ma, 'evidence_timestamp_bits', 16))
            if evidence_header_bits < 0:
                raise ValueError(
                    'evidence_fixed_schema_header_bits must be non-negative')
            if 0 < evidence_header_bits < minimum_crc_bits:
                raise ValueError(
                    'fixed-schema evidence CRC is too short for the '
                    'configured undetected-error budget')
            if evidence_timestamp_bits < 0:
                raise ValueError(
                    'evidence_timestamp_bits must be non-negative')
            if evidence_timestamp_bits == 0 and evidence_header_bits == 0:
                raise ValueError(
                    'implicit evidence timestamp requires a fixed-schema '
                    'synchronous header')
            self._evidence_packet_layout = EvidencePacketLayout(
                num_agents=self.K,
                num_targets=self.Q,
                header_bits=(
                    evidence_header_bits if evidence_header_bits > 0
                    else int(getattr(ma, 'comm_header_bits', 64))),
                timestamp_bits=evidence_timestamp_bits,
                confidence_bits=self._evidence_confidence_bits,
                feedback_bits_per_entry=(
                    4 * self._u2u_belief_feedback_mean_bits
                    + 4 * self._u2u_belief_feedback_cov_bits
                    + self._u2u_belief_feedback_aoi_bits
                    if self._u2u_belief_feedback_enabled else 0
                ),
            )
            if self._evidence_legacy_service_envelope_enabled:
                if evidence_header_bits <= 0 or evidence_timestamp_bits != 0:
                    raise ValueError(
                        'legacy evidence service envelope requires the '
                        'synchronous compact-header codec')
                self._evidence_service_envelope_layout = EvidencePacketLayout(
                    num_agents=self.K,
                    num_targets=self.Q,
                    header_bits=int(getattr(ma, 'comm_header_bits', 64)),
                    timestamp_bits=16,
                    confidence_bits=self._evidence_confidence_bits,
                    feedback_bits_per_entry=(
                        4 * self._u2u_belief_feedback_mean_bits
                        + 4 * self._u2u_belief_feedback_cov_bits
                        + self._evidence_legacy_service_aoi_bits
                        if self._u2u_belief_feedback_enabled else 0
                    ),
                )
            boundaries = np.asarray(getattr(
                ma,
                'evidence_packet_confidence_log_boundaries',
                [],
            ), dtype=np.float64)
            representatives = np.asarray(getattr(
                ma,
                'evidence_packet_confidence_representatives',
                [],
            ), dtype=np.float64)
            levels = 1 << max(self._evidence_confidence_bits, 0)
            if (
                self._evidence_confidence_bits > 0
                and (
                    boundaries.shape != (levels - 1,)
                    or representatives.shape != (levels,)
                )
            ):
                raise ValueError(
                    'confidence calibration must provide 2^b-1 log '
                    'boundaries and 2^b representatives')
            if self._evidence_confidence_bits > 0:
                self._evidence_confidence_quantizer = (
                    DeflectionConfidenceQuantizer(
                        bits=self._evidence_confidence_bits,
                        log_boundaries=boundaries,
                        representatives=representatives,
                    )
                )
        self._gru_hidden: Dict[int, np.ndarray] = {}  # per-agent GRU hidden state (64-dim)

    def reseed(self, seed: int) -> np.random.Generator:
        """Replace the owned RNG and rebind persistent stochastic clients.

        Audit 2026-08-25: the cost-aware InterUAVCommunicationModel previously
        kept the constructor-time generator after a reset(seed=...), so its
        burst/shadowing/FBL-erasure draws followed the stale constructor stream
        and same-seed episode replay diverged (runtime probe _rng_probe.py
        confirmed shadow matrices differ across two reset(seed=91) episodes).
        Rebind it here so reset(seed=...) governs every stochastic consumer.
        """
        self.rng = np.random.default_rng(seed)
        self.action_space.rng = self.rng
        self.deflection_computer.rng = self.rng
        if self._inter_uav_comm is not None:
            self._inter_uav_comm.rng = self.rng
        return self.rng

    def close(self) -> None:
        """Release optional simulator-side persistent worker resources."""
        if self._distributed_replicated_power_executor is not None:
            self._distributed_replicated_power_executor.close()
            self._distributed_replicated_power_executor = None

    def reset(self) -> Tuple[Dict[int, np.ndarray], Dict]:
        """Reset the environment to initial state.

        Returns:
            (observations_dict, info_dict)
        """
        self.t = 0
        self._hyperedge_acceleration.reset_stats()
        self._distributed_replicated_power_consecutive_failures = 0
        if self._distributed_replicated_power_executor is not None:
            try:
                warmup = self._distributed_replicated_power_executor.warm_up(
                    self.K, self.Q)
                self._distributed_replicated_power_executor_warmup_time_s += (
                    float(warmup))
            except Exception as error:
                if not (
                    self._distributed_replicated_power_parallel_fallback_to_serial
                ):
                    raise RuntimeError(
                        'parallel private power worker warm-up failed') from error
                self._distributed_replicated_power_executor.close()
                self._distributed_replicated_power_executor = None
                self._distributed_replicated_power_executor_warmup_failed = True
        self.prev_P_D = None
        self.prev_P_D_local = {}
        self._coord_pd_ema = None
        # D1.1-A audit state (DSH_LEX_AUDIT diagnostic) — fresh per episode.
        self._lex_mode = 'none'
        self._lex_t_star = None
        self._last_analytical_gain = None
        self._last_analytical_budget = None
        # Audit 2026-08-17 (P0): these per-episode state holders must be reset,
        # otherwise the first frame(s) of every episode after the first inherit
        # the previous episode's values (cross-episode state leak / A6 replay
        # determinism violation).
        self._last_analytical_dual_prices = None
        self._last_bargaining_value = None
        self._last_analytical_power_balance_error = 0.0
        self._last_analytical_power_budget_violation_w = 0.0
        self._last_analytical_unused_power_w = 0.0
        self._prev_obs_deque = {}
        self._probe_miss_count = np.zeros(self.Q, dtype=np.int32)
        self._prev_obs = {}  # clear history on reset
        self._gru_hidden = {}  # clear GRU state on reset
        self._comm_msgs = {}
        self._pending_comm_messages = {}
        self._pending_comm_rates = {}
        self._pending_comm_token_masks = {}
        self._pending_decision_sufficient_bits = {}
        self._last_sent_comm_token_masks = {}
        self._last_sent_comm_target_claims = {}
        self._persistent_commitment_mask = np.zeros(
            (self.K, self.Q), dtype=bool)
        self._persistent_commitment_old_mask = np.zeros(
            (self.K, self.Q), dtype=bool)
        self._persistent_commitment_last_switch = np.full(
            self.K, FRAME_NOT_APPLICABLE, dtype=np.int64)
        self._persistent_commitment_last_seen = np.full(
            self.K, FRAME_NOT_APPLICABLE, dtype=np.int64)
        self._persistent_commitment_handover_remaining = np.zeros(
            self.K, dtype=np.int64)
        self._persistent_commitment_effective_mask = np.zeros(
            (self.K, self.Q), dtype=bool)
        self._persistent_commitment_resolved_frame = -1
        self._persistent_commitment_metrics = {}
        self._qpd_queue = np.zeros((self.K, self.Q), dtype=np.float64)
        self._qpd_previous_queue = np.zeros(
            (self.K, self.Q), dtype=np.float64)
        self._qpd_bid = np.zeros((self.K, self.Q), dtype=np.float64)
        self._qpd_previous_bid = np.zeros(
            (self.K, self.Q), dtype=np.float64)
        initial_qpd = min(
            1.0, self._qpd_row_capacity / max(self.Q, 1))
        self._qpd_primal = np.full(
            (self.K, self.Q), initial_qpd, dtype=np.float64)
        self._qpd_previous_primal = self._qpd_primal.copy()
        self._qpd_target_price = np.zeros(
            (self.K, self.Q), dtype=np.float64)
        self._qpd_capability = np.zeros(
            (self.K, self.Q), dtype=np.float64)
        self._qpd_age = np.zeros((self.K, self.Q), dtype=np.int64)
        self._qpd_send_mask = np.zeros(
            (self.K, self.Q), dtype=bool)
        self._pending_qpd_protocol = {}
        self._qpd_commitment_mask = np.zeros(
            (self.K, self.Q), dtype=bool)
        self._qpd_received_primal = np.zeros(
            (self.K, self.K, self.Q), dtype=np.float64)
        self._qpd_received_queue = np.zeros(
            (self.K, self.K, self.Q), dtype=np.float64)
        self._qpd_received_last_seen = np.full(
            (self.K, self.K, self.Q), FRAME_NOT_APPLICABLE, dtype=np.int64)
        self._qpd_metrics = {}
        self._qpd_last_submission_frame = -1
        self._pending_hyperedge_protocol = {}
        self._pending_hyperedge_protocol_frame = {}
        self._hyperedge_local_offer = np.zeros(
            (self.K, self.Q, self._hyperedge_protocol_dim),
            dtype=np.float64)
        self._hyperedge_received_offer = np.zeros(
            (self.K, self.K, self.Q, self._hyperedge_protocol_dim),
            dtype=np.float64)
        self._hyperedge_received_last_seen = np.full(
            (self.K, self.K, self.Q), FRAME_NOT_APPLICABLE, dtype=np.int64)
        self._hyperedge_consensus_streak = np.zeros(
            (self.K, self.K, self.Q), dtype=np.int64)
        self._hyperedge_selected_set = tuple()
        self._hyperedge_last_update_frame = FRAME_NOT_APPLICABLE
        self._hyperedge_metrics = {}
        self._hyperedge_golden_snapshot = {}
        self._hyperedge_public_gain_views = np.zeros(
            (self.K, self.K, self.Q), dtype=np.float64)
        self._hyperedge_public_full_views = np.zeros(
            self.K, dtype=bool)
        self._composable_certificate_local_lower = np.zeros(
            (self.K, self.Q), dtype=np.float64)
        self._composable_certificate_gain_views = np.zeros(
            (self.K, self.K, self.Q), dtype=np.float64)
        self._composable_certificate_gain_upper_views = np.zeros(
            (self.K, self.K, self.Q), dtype=np.float64)
        self._composable_certificate_local_dual_upper = np.zeros(
            self.K, dtype=np.float64)
        self._composable_certificate_local_targetwise_upper = np.zeros(
            (self.K, self.Q), dtype=np.float64)
        self._composable_certificate_local_frame = np.full(
            self.K, -1, dtype=np.int64)
        self._composable_certificate_last_sent_lower = np.zeros(
            (self.K, self.Q), dtype=np.float64)
        self._composable_certificate_last_sent_frame = np.full(
            self.K, -1, dtype=np.int64)
        self._composable_certificate_last_sent_dual_upper = np.zeros(
            self.K, dtype=np.float64)
        self._composable_certificate_last_sent_targetwise_upper = np.zeros(
            (self.K, self.Q), dtype=np.float64)
        self._composable_certificate_received_lower = np.zeros(
            (self.K, self.K, self.Q), dtype=np.float64)
        self._composable_certificate_received_frame = np.full(
            (self.K, self.K), -1, dtype=np.int64)
        self._composable_certificate_received_dual_upper = np.zeros(
            (self.K, self.K), dtype=np.float64)
        self._composable_certificate_received_targetwise_upper = np.zeros(
            (self.K, self.K, self.Q), dtype=np.float64)
        self._composable_certificate_metrics = {}
        self._reset_owner_posterior_state()
        self._distributed_movement_target = np.full(
            self.K, -1, dtype=np.int64)
        self._distributed_movement_local_assignment = np.full(
            (self.K, self.K), -1, dtype=np.int64)
        self._distributed_movement_last_update = np.full(
            self.K, FRAME_NOT_APPLICABLE, dtype=np.int64)
        self._distributed_movement_anchor_target_xy = np.zeros(
            (self.K, self.K, self.Q, 2), dtype=np.float64)
        self._distributed_movement_anchor_last_seen = np.full(
            (self.K, self.K, self.Q), FRAME_NOT_APPLICABLE, dtype=np.int64)
        self._distributed_bistatic_gate_latch = np.full(
            self.K, -1, dtype=np.int8)
        self._distributed_role_capacity_tx_resp = np.zeros(
            (self.K, self.K, self.Q), dtype=np.int8)
        self._distributed_role_capacity_rx_resp = np.zeros(
            (self.K, self.K, self.Q), dtype=np.int8)
        self._distributed_role_capacity_bottleneck = np.full(
            self.K, np.inf, dtype=np.float64)
        self._distributed_role_capacity_last_reassign = np.full(
            self.K, FRAME_NOT_APPLICABLE, dtype=np.int64)
        self._distributed_gap_tx_resp = np.zeros(
            (self.K, self.K, self.Q), dtype=np.int8)
        self._distributed_gap_rx_resp = np.zeros(
            (self.K, self.K, self.Q), dtype=np.int8)
        self._pending_comm_power_fractions = {}
        self._pending_sensing_weights = {}
        self._current_comm_power_w = np.zeros(self.K, dtype=np.float64)
        self._current_sensing_power_w = np.full(
            (self.K, self.Q),
            self._sensing_power_cap_w / max(self.Q, 1), dtype=np.float64)
        self._distributed_replicated_previous_power = (
            self._current_sensing_power_w.copy())
        self._distributed_replicated_local_power_cache = np.zeros(
            (self.K, self.K, self.Q), dtype=np.float64)
        self._distributed_replicated_local_price_cache = np.full(
            (self.K, self.Q), 1.0 / max(self.Q, 1), dtype=np.float64)
        self._distributed_replicated_local_cache_valid = np.zeros(
            self.K, dtype=bool)
        self._isac_sensing_battery_before = None
        self._last_isac_metrics = {}
        self._last_movement_safety_metrics = {}
        self._external_structure_edge_values = None
        self._external_structure_candidate_mask = None
        if self._dynamic_local_search_coordinator is not None:
            self._dynamic_local_search_coordinator.reset()
        self._pending_structure_student_protocol = {}
        self._structure_student_public_protocol = np.zeros(
            (self.K, self.Q, self._structure_student_endpoint_width),
            dtype=np.float64,
        )
        self._structure_student_public_valid = np.zeros(
            self.K, dtype=bool)
        self._structure_student_public_last_seen = np.full(
            self.K, FRAME_NOT_APPLICABLE, dtype=np.int64)
        self._structure_student_mailbox = []
        self._structure_student_metrics = {}
        self._received_comm_msgs = {}
        self._received_comm_meta = {}
        self._comm_mailbox = []
        self._last_comm_stats = CommunicationStepStats()
        self._sample_comm_channel_profile()
        if self._inter_uav_comm is not None:
            self._inter_uav_comm.reset_channel_state(self.K)
        if self._trust_manager is not None:
            self._trust_manager.reset()


        # Fusion center at center of area
        self.fc_position = np.array([
            self.area_size[0] / 2, self.area_size[1] / 2, self.height
        ], dtype=np.float64)
        if getattr(self.cfg.marl, 'use_p0_sinr_gated', False):
            self.obs_builder._fc_position = self.fc_position

        # Initialize UAVs at random positions with safe spacing
        self.uavs = []
        positions = self._generate_safe_uav_positions()
        for k in range(self.K):
            uav = UAV(
                uav_id=k,
                initial_pos=positions[k],
                v_max=self.cfg.uav.v_max,
                d_safe=self.cfg.uav.d_safe,
                B_max=self.cfg.uav.B_max,
                P_sense=self.cfg.uav.P_sense,
                P_report=(self.cfg.uav.P_report
                          if self.ground_communication_enabled else 0.0),
                P_fly_static=self.cfg.uav.P_fly_static,
                P_fly_coeff=self.cfg.uav.P_fly_coeff,
                dt=self.dt,
                area_size=self.area_size,
                height=self.height,
            )
            self.uavs.append(uav)

        # Initialize targets
        self.targets = []
        for q in range(self.Q):
            # Random initial position and velocity
            pos = np.array([
                self.rng.uniform(100, self.area_size[0] - 100),
                self.rng.uniform(100, self.area_size[1] - 100),
            ])
            speed = (self.rng.uniform(*self.cfg.target.speed_range)
                     if self.tracking_enabled else 0.0)
            angle = self.rng.uniform(0, 2 * np.pi)
            vel = np.array([speed * np.cos(angle), speed * np.sin(angle)])

            target = Target(
                target_id=q,
                initial_pos=pos,
                initial_vel=vel,
                sigma_a=self.cfg.target.sigma_a,
                dt=self.dt,
                area_size=self.area_size,
                rng=self.rng,
                motion_model=getattr(self.cfg.target, 'motion_model', 'CV'),
                turn_rate=getattr(self.cfg.target, 'ct_turn_rate', 0.3),
            )
            self.targets.append(target)

        # Initialize beliefs
        true_positions = np.array([t.get_position_3d() for t in self.targets])
        true_velocities = np.array([
            np.array([*t.get_velocity(), 0.0]) for t in self.targets
        ])
        belief_motion_model = str(getattr(
            self.cfg.marl, 'belief_motion_model', 'TARGET')).upper()
        if belief_motion_model == 'TARGET':
            belief_motion_model = str(getattr(
                self.cfg.target, 'motion_model', 'CV')).upper()
        if belief_motion_model not in {'CV', 'CA'}:
            raise ValueError(
                "marl.belief_motion_model must be TARGET, CV, or CA")
        self.belief_mgr = BeliefManager(
            K=self.K, Q=self.Q,
            initial_positions=true_positions,
            initial_velocities=true_velocities,
            dt=self.dt,
            sigma_a=self.cfg.target.sigma_a,
            rng=self.rng,
            motion_model=belief_motion_model,
            nis_enabled=bool(getattr(self.cfg.marl, 'belief_nis_enabled', False)),
            nis_window=float(getattr(self.cfg.marl, 'belief_nis_window', 0.05)),
            nis_inflate_k=float(getattr(self.cfg.marl, 'belief_nis_inflate_k', 2.0)),
            nis_lambda_max=float(getattr(self.cfg.marl, 'belief_nis_lambda_max', 5.0)),
            nis_deflate_rate=float(getattr(self.cfg.marl, 'belief_nis_deflate_rate', 0.95)),
            cov_floor_pos=float(getattr(self.cfg.marl, 'belief_cov_floor_pos', 25.0)),
            cov_floor_vel=float(getattr(self.cfg.marl, 'belief_cov_floor_vel', 1.0)),
            nis_enter_threshold=float(getattr(self.cfg.marl, 'belief_nis_enter_threshold', 1.8)),
            nis_exit_threshold=float(getattr(self.cfg.marl, 'belief_nis_exit_threshold', 1.2)),
            nis_enter_frames=int(getattr(self.cfg.marl, 'belief_nis_enter_frames', 3)),
            nis_exit_frames=int(getattr(self.cfg.marl, 'belief_nis_exit_frames', 5)),
        )

        # Build initial observations
        obs = self._build_observations()

        info = {
            'uav_positions': np.array([u.pos for u in self.uavs]),
            'target_positions': true_positions,
            'comm_channel_profile': self._active_comm_channel_profile,
            'comm_channel_snr_threshold_db': (
                self._active_comm_snr_threshold_db),
            'comm_channel_deadline_s': self._active_comm_deadline_s,
            'distributed_replicated_power_worker_warmup_time_s': float(
                self._distributed_replicated_power_executor_warmup_time_s),
            'distributed_replicated_power_worker_warmup_failed': float(
                self._distributed_replicated_power_executor_warmup_failed),
        }

        return obs, info

    def _sample_comm_channel_profile(self) -> None:
        """Select one reproducible episode-level U2U channel profile."""
        if self._inter_uav_comm is None:
            return

        ma = self.cfg.marl
        enabled = bool(getattr(
            ma, 'comm_channel_randomization_enabled', False))
        snr_values = list(getattr(
            ma, 'comm_channel_randomization_snr_threshold_db_values', []))
        deadline_values = list(getattr(
            ma, 'comm_channel_randomization_deadline_s_values', []))
        weights = list(getattr(
            ma, 'comm_channel_randomization_profile_weights', []))

        if enabled:
            if not snr_values or len(snr_values) != len(deadline_values):
                raise ValueError(
                    'channel randomization requires equal non-empty SNR and '
                    'deadline profile lists')
            if weights and len(weights) != len(snr_values):
                raise ValueError(
                    'channel randomization weights must match profile count')
            deadlines = np.asarray(deadline_values, dtype=np.float64)
            if np.any(~np.isfinite(deadlines)) or np.any(deadlines <= 0.0):
                raise ValueError(
                    'channel randomization deadlines must be finite and > 0')
            snrs = np.asarray(snr_values, dtype=np.float64)
            if np.any(~np.isfinite(snrs)):
                raise ValueError(
                    'channel randomization SNR thresholds must be finite')

            probabilities = None
            if weights:
                probabilities = np.asarray(weights, dtype=np.float64)
                if (np.any(~np.isfinite(probabilities))
                        or np.any(probabilities < 0.0)
                        or float(np.sum(probabilities)) <= 0.0):
                    raise ValueError(
                        'channel randomization weights must be finite, '
                        'non-negative, and have positive sum')
                probabilities = probabilities / np.sum(probabilities)
            # Draw from a cloned generator state so selecting a channel
            # profile does not perturb the geometry/target RNG stream. This
            # preserves common-random-number comparisons with fixed-channel
            # controls using the same environment seed.
            channel_rng = np.random.default_rng()
            channel_rng.bit_generator.state = copy.deepcopy(
                self.rng.bit_generator.state)
            profile = int(channel_rng.choice(
                len(snr_values), p=probabilities))
            snr_threshold_db = float(snrs[profile])
            deadline_s = float(deadlines[profile])
        else:
            profile = -1
            snr_threshold_db = self._nominal_comm_snr_threshold_db
            deadline_s = self._nominal_comm_deadline_s

        self._active_comm_channel_profile = profile
        self._active_comm_snr_threshold_db = snr_threshold_db
        self._active_comm_deadline_s = deadline_s
        self._inter_uav_comm.snr_threshold_db = snr_threshold_db
        self._inter_uav_comm.deadline_s = deadline_s
        # The token metadata normalizes latency by the active system deadline.
        self.obs_builder.comm_deadline_s = max(deadline_s, 1e-9)

    def submit_learned_communications(
        self,
        messages: Dict[int, np.ndarray],
        rate_indices: Dict[int, int],
        comm_power_fractions: Optional[Dict[int, float]] = None,
        sensing_target_weights: Optional[Dict[int, np.ndarray]] = None,
        token_masks: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        """Queue the stochastic communication actions for the next frame.

        The policy decides message semantics and rate.  Transport, quantization,
        delivery and accounting remain environment responsibilities.
        """
        def normalized_mapping(raw: object, label: str) -> Dict[int, object]:
            if not isinstance(raw, dict):
                raise ValueError(f"{label} must be a UAV-ID mapping")
            normalized: Dict[int, object] = {}
            for raw_key, value in raw.items():
                if isinstance(raw_key, bool) or not isinstance(
                    raw_key, Integral
                ):
                    raise ValueError(f"{label} keys must be integer UAV IDs")
                key = int(raw_key)
                if not 0 <= key < self.K:
                    raise ValueError(f"{label} contains invalid UAV ID {key}")
                if key in normalized:
                    raise ValueError(f"{label} contains duplicate UAV ID {key}")
                normalized[key] = value
            return normalized

        raw_messages = normalized_mapping(messages, "messages")
        raw_rates = normalized_mapping(rate_indices, "rate_indices")
        if set(raw_rates) != set(raw_messages):
            raise ValueError(
                "rate_indices must contain exactly the submitted message senders")
        validated_messages: Dict[int, np.ndarray] = {}
        validated_rates: Dict[int, int] = {}
        rate_count = len(self.cfg.marl.comm_rate_bits_per_dim)
        for sender, raw_message in raw_messages.items():
            try:
                message = np.asarray(raw_message, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"message for UAV {sender} must be numeric") from exc
            if message.shape != (self._comm_payload_dim,):
                raise ValueError(
                    f"message for UAV {sender} must have shape "
                    f"{(self._comm_payload_dim,)}")
            if not np.all(np.isfinite(message)):
                raise ValueError(f"message for UAV {sender} must be finite")
            raw_rate = raw_rates[sender]
            if isinstance(raw_rate, bool) or not isinstance(raw_rate, Integral):
                raise ValueError(
                    f"rate index for UAV {sender} must be an integer")
            rate = int(raw_rate)
            if not 0 <= rate < rate_count:
                raise ValueError(
                    f"rate index for UAV {sender} is outside [0, {rate_count})")
            validated_messages[sender] = message.copy()
            validated_rates[sender] = rate
        raw_masks = normalized_mapping(token_masks or {}, "token_masks")
        if not set(raw_masks).issubset(raw_messages):
            raise ValueError("token_masks may only name submitted message senders")
        validated_masks: Dict[int, np.ndarray] = {}
        for k, raw_mask in raw_masks.items():
            mask = np.asarray(raw_mask, dtype=np.float64)
            if mask.shape != (self.Q,):
                raise ValueError(
                    f'token mask for UAV {k} must have shape {(self.Q,)}')
            if not np.all(np.isfinite(mask)):
                raise ValueError(f'token mask for UAV {k} must be finite')
            validated_masks[int(k)] = (
                mask > 0.5).astype(np.float64)
        validated_fractions: Dict[int, float] = {}
        validated_weights: Dict[int, np.ndarray] = {}
        if self._joint_isac_power_enabled:
            fractions = normalized_mapping(
                comm_power_fractions or {}, "comm_power_fractions")
            weights = normalized_mapping(
                sensing_target_weights or {}, "sensing_target_weights")
            for k, raw_fraction in fractions.items():
                if isinstance(raw_fraction, bool):
                    raise ValueError(
                        f'communication power fraction for UAV {k} must be numeric')
                try:
                    fraction = float(raw_fraction)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f'communication power fraction for UAV {k} must be numeric'
                    ) from exc
                if not np.isfinite(fraction) or not (
                    self._comm_power_fraction_min
                    <= fraction
                    <= self._comm_power_fraction_max
                ):
                    raise ValueError(
                        f'communication power fraction for UAV {k} must be '
                        'finite and within the configured range')
                validated_fractions[k] = fraction
            for k, raw_weight in weights.items():
                weight = np.asarray(raw_weight, dtype=np.float64)
                if weight.shape != (self.Q,):
                    raise ValueError(
                        f'sensing weights for UAV {k} must have shape {(self.Q,)}')
                if not np.all(np.isfinite(weight)) or np.any(weight < 0.0):
                    raise ValueError(
                        f'sensing weights for UAV {k} must be finite and '
                        'non-negative')
                total = float(np.sum(weight))
                if total <= 1e-12:
                    raise ValueError(
                        f'sensing weights for UAV {k} must have positive sum')
                validated_weights[k] = weight / total

        # Commit the validated policy payload atomically.  No invalid optional
        # field can leave a half-updated message/rate queue behind.
        self._pending_comm_messages = validated_messages
        self._pending_comm_rates = validated_rates
        self._pending_comm_token_masks = validated_masks
        if self._qpd_enabled:
            self._prepare_qpd_submission()
        if self._joint_isac_power_enabled:
            self._pending_comm_power_fractions = {
                k: validated_fractions.get(
                    k, float(self._comm_power_fraction_min))
                for k in range(self.K)
            }
            self._pending_sensing_weights = {}
            for k in range(self.K):
                if self._qpd_enabled and self._qpd_override_sensing_weights:
                    # The optimizer changes only the target split.  The
                    # actor's communication fraction remains intact in this
                    # mechanism-only gate, and the exact 1 W projection below
                    # still owns the comm/sensing total.
                    qpd_weight = (
                        np.maximum(self._qpd_primal[k], 0.0)
                        * np.maximum(self._qpd_capability[k], 1e-6)
                    )
                    w = np.asarray(qpd_weight, dtype=np.float64)
                else:
                    w = validated_weights.get(
                        k, np.ones(self.Q, dtype=np.float64) / self.Q)
                if w.shape != (self.Q,):
                    raise ValueError(
                        f'sensing weights for UAV {k} must have shape {(self.Q,)}')
                if not np.all(np.isfinite(w)) or np.any(w < 0.0):
                    raise ValueError(
                        f'sensing weights for UAV {k} must be finite and '
                        'non-negative')
                total = float(np.sum(w))
                if total <= 1e-12:
                    raise ValueError(
                        f'sensing weights for UAV {k} must have positive sum')
                self._pending_sensing_weights[k] = w / total
        # Hyperedge state is encoded after the next frame's movement has been
        # applied, immediately before radio transport. Preparing it here would
        # attach the previous frame's position to a later transmission.

    def submit_structure_student_edge_values(
        self,
        edge_values: np.ndarray,
    ) -> None:
        """Submit a frozen student's directed K-by-K-by-Q ranking graph."""
        values = np.asarray(edge_values, dtype=np.float64)
        expected = (self.K, self.K, self.Q)
        if values.shape != expected:
            raise ValueError(
                f"structure student edge graph must have shape {expected}")
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "structure student edge graph contains non-finite values")
        values = np.maximum(values, 0.0)
        for k in range(self.K):
            values[k, k, :] = 0.0
        self._external_structure_edge_values = values.copy()

    def submit_structure_student_candidate_mask(
        self,
        candidate_mask: np.ndarray,
    ) -> None:
        """Submit the public, locally generated sparse candidate graph."""
        mask = np.asarray(candidate_mask, dtype=bool)
        expected = (self.K, self.K, self.Q)
        if mask.shape != expected:
            raise ValueError(
                f"structure candidate mask must have shape {expected}")
        mask = mask.copy()
        mask[np.arange(self.K), np.arange(self.K), :] = False
        self._external_structure_candidate_mask = mask

    def configure_dynamic_local_search(self, coordinator) -> None:
        """Install a stateful C1.7 structure controller for evaluation."""
        if not self._p0_maxmin_pairing_enabled:
            raise ValueError(
                "dynamic local search requires max-min pairing mode")
        self._dynamic_local_search_coordinator = coordinator
        self._dynamic_local_search_coordinator.reset()

    def configure_structure_student_channel(
        self,
        decoder,
        *,
        bits_per_dim: int = 8,
        adaptive_min_bits_per_dim: int = 0,
        min_comm_fraction: float = 0.01,
        allow_cardinality_mismatch: bool = False,
    ) -> None:
        """Enable physical U2U transport of frozen-student endpoint Tokens."""
        if self._comm_mode != 'cost_aware' or self._inter_uav_comm is None:
            raise ValueError(
                "structure-student transport requires cost-aware U2U "
                "communication")
        cardinality_mismatch = (
            decoder.num_uavs != self.K or decoder.num_targets != self.Q)
        if cardinality_mismatch and not allow_cardinality_mismatch:
            raise ValueError(
                "structure-student decoder K/Q does not match environment")
        if cardinality_mismatch and not bool(getattr(
                decoder, 'cardinality_equivariant', False)):
            raise ValueError(
                "structure-student decoder does not declare cardinality "
                "equivariance")
        width = 2 * int(decoder.model.endpoint_dim)
        if width <= 0:
            raise ValueError("structure-student endpoint width must be positive")
        self._structure_student_channel_enabled = True
        self._structure_student_decoder = decoder
        self._structure_student_endpoint_width = width
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
            min_comm_fraction,
            0.0,
            self._comm_power_fraction_max,
        ))
        self._pending_structure_student_protocol = {}
        self._structure_student_public_protocol = np.zeros(
            (self.K, self.Q, width), dtype=np.float64)
        self._structure_student_public_valid = np.zeros(
            self.K, dtype=bool)
        self._structure_student_public_last_seen = np.full(
            self.K, FRAME_NOT_APPLICABLE, dtype=np.int64)
        self._structure_student_mailbox = []
        self._structure_student_metrics = {}
        self._external_structure_edge_values = None

    def _select_structure_student_bits_per_dim(
        self,
        sender: int,
        uav_positions: np.ndarray,
        tx_powers_w: Optional[Dict[int, float]],
        extra_dimensions: Dict[int, int],
        active_sender_count: int,
        structure_endpoint_dimensions: int,
    ) -> tuple[int, int]:
        """Choose the highest structural precision feasible for one broadcast.

        The projection uses only quantities available to a broadcasting UAV's
        link layer: its selected RF power, packet size, deadline, bandwidth and
        peer CSI/range.  It therefore adapts representation precision without
        changing the frozen Actor or consuming privileged sensing evidence.
        The returned header width signals the selected structural code rate.
        """
        maximum = int(self._structure_student_bits_per_dim)
        minimum = int(self._structure_student_adaptive_min_bits_per_dim)
        if minimum <= 0 or minimum >= maximum:
            return maximum, 0

        num_levels = maximum - minimum + 1
        rate_header_bits = int(np.ceil(np.log2(num_levels)))
        rate_index = int(self._pending_comm_rates.get(sender, 0))
        rate_index = int(np.clip(
            rate_index,
            0,
            len(self._inter_uav_comm.rate_bits_per_dim) - 1,
        ))
        actor_bits_per_dim = int(
            self._inter_uav_comm.rate_bits_per_dim[rate_index])
        actor_dimensions = (
            self._inter_uav_comm._active_dimensions(
                self._pending_comm_token_masks.get(sender))
            + max(0, int(extra_dimensions.get(sender, 0)))
        )
        effective_bandwidth_hz = (
            self._inter_uav_comm.bandwidth_hz
            / max(1, int(active_sender_count))
        )
        sender_power_w = (
            self._inter_uav_comm.tx_power_w
            if tx_powers_w is None
            else max(float(tx_powers_w.get(
                sender, self._inter_uav_comm.tx_power_w)), 0.0)
        )

        for bits in range(maximum, minimum - 1, -1):
            payload_bits = (
                actor_dimensions * actor_bits_per_dim
                + structure_endpoint_dimensions * bits
                + rate_header_bits
            )
            packet_bits = self._inter_uav_comm.header_bits + payload_bits
            feasible = True
            for receiver in range(self.K):
                if receiver == sender:
                    continue
                snr_db, _, _, latency_s = self._inter_uav_comm._link(
                    uav_positions[sender],
                    uav_positions[receiver],
                    packet_bits,
                    effective_bandwidth_hz,
                    sender_power_w,
                )
                if (
                    snr_db < self._inter_uav_comm.snr_threshold_db
                    or latency_s > self._inter_uav_comm.deadline_s
                ):
                    feasible = False
                    break
            if feasible:
                return bits, rate_header_bits
        return minimum, rate_header_bits

    def submit_structure_student_endpoint_protocol(
        self,
        protocol: np.ndarray,
    ) -> None:
        """Queue every UAV's bounded per-target Tx/Rx endpoint stream."""
        if not self._structure_student_channel_enabled:
            raise RuntimeError(
                "structure-student channel has not been configured")
        values = np.asarray(protocol, dtype=np.float64)
        expected = (
            self.K, self.Q, self._structure_student_endpoint_width)
        if values.shape != expected:
            raise ValueError(
                f"structure-student endpoint protocol must have shape "
                f"{expected}")
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "structure-student endpoint protocol contains non-finite "
                "values")
        values = np.clip(values, -1.0, 1.0)
        self._pending_structure_student_protocol = {
            sender: values[sender].copy()
            for sender in range(self.K)
        }
        # Structural endpoints use a separate codebook from the actor's latent
        # Token.  Preserve the actor's learned 4-bit/silence distribution while
        # reserving only the explicit pilot power needed by a mandatory
        # structural packet.
        if self._joint_isac_power_enabled:
            for sender in range(self.K):
                self._pending_comm_power_fractions[sender] = max(
                    float(self._pending_comm_power_fractions.get(
                        sender, 0.0)),
                    self._structure_student_min_comm_fraction,
                )

    def structure_student_protocol_due_next_step(self) -> bool:
        """Whether the next frame can consume a newly encoded structure graph.

        Fixed Hold-N projection only reads ranking edges on its scheduled solve
        frames.  Event-triggered projection remains conservative and requests
        a packet every frame because its crisis condition is evaluated after
        the physical graph has been formed.
        """
        if not self._structure_student_channel_enabled:
            return False
        if self._p0_maxmin_event_triggered_enabled:
            return True
        if not self._p0_maxmin_pairing_enabled:
            return True
        next_frame = int(self.t) + 1
        hold_frames = max(
            1, int(self._p0_maxmin_pairing_hold_frames))
        return bool(
            self._cached_p0_solution is None
            or next_frame == 1
            or next_frame % hold_frames == 0
        )

    def _merge_structure_student_public_protocol(
        self,
        sender: int,
        protocol: np.ndarray,
        sent_frame: int,
    ) -> None:
        """Atomically publish one sender version after successful multicast."""
        sender = int(sender)
        values = np.asarray(protocol, dtype=np.float64)
        expected = (self.Q, self._structure_student_endpoint_width)
        if not (0 <= sender < self.K) or values.shape != expected:
            raise ValueError("invalid structure-student public endpoint packet")
        self._structure_student_public_protocol[sender] = values
        self._structure_student_public_valid[sender] = True
        self._structure_student_public_last_seen[sender] = int(sent_frame)

    def _refresh_structure_student_edges(self) -> None:
        """Decode the common cached endpoint table into the ranking graph."""
        if (
            not self._structure_student_channel_enabled
            or self._structure_student_decoder is None
        ):
            return
        valid = self._structure_student_public_valid.copy()
        if np.sum(valid) < 2:
            # Never fall through to the environment's native physical ranking
            # when the distributed structural channel is unavailable.  That
            # would turn packet loss into access to a privileged centralized
            # graph.  An explicit zero graph is the auditable fail-closed
            # behavior until at least one directed peer pair is reconstructable.
            self.submit_structure_student_edge_values(np.zeros(
                (self.K, self.K, self.Q), dtype=np.float64))
            return
        values = self._structure_student_decoder.predict_from_endpoint_protocol(
            self._structure_student_public_protocol,
            valid_senders=valid,
        )[0]
        self.submit_structure_student_edge_values(values)

    def _apply_decision_sufficient_silence(self) -> None:
        """C6 decision-sufficient U2U gate (audit advice/001 §12-13, 2026-08-26).

        Active ONLY under the strict distributed identity (decision-sufficient
        comm enabled AND no-truth fail-closed; both default OFF so every
        pre-C6 run is bit-identical).  For each pending sender it computes the
        LOCAL top-1 vs top-2 capability margin Δ from the sender's own
        local-belief target distances (never simulator truth), bounds the
        stale drift with a per-age Lipschitz term and the physical error with
        the local target-position uncertainty, then applies
        ``decision_sufficient_plan``:

          - Δ > 2(E_stale + E_phys) + 2 eps_B  ->  CERTIFIED persist; the
            sender is silenced (pending message/rate/mask dropped -> ZERO
            airtime, decision-preserving token suppressed);
          - otherwise the token transmits, with bits adapted to the smallest
            certified decision-preserving precision (clamped to the
            configured rate ladder).

        Fail-closed: if the strict gate is on but a pending sender's local
        belief cannot be resolved, the sender is silenced instead of reading
        simulator truth.
        """
        comm_enabled = bool(getattr(
            self, '_distributed_decision_sufficient_comm_enabled', False))
        strict = bool(getattr(
            self, '_distributed_no_truth_fail_closed', False))
        event_trigger = bool(getattr(
            self, '_distributed_decision_sufficient_event_trigger_enabled',
            False))
        adaptive_bits = bool(getattr(
            self, '_distributed_decision_sufficient_adaptive_bits_enabled',
            False))
        # Event-trigger OFF keeps the whole C6 pass inert: nothing is silenced
        # and no per-sender plan bits are recorded, so the U2U transport stays
        # bit-identical to the pre-C6 rate ladder.  The constructor already
        # fail-closes the chain adaptive -> event -> comm -> no-truth.
        if not (comm_enabled and strict and event_trigger):
            self._pending_decision_sufficient_bits = {}
            return
        from uav_isac.coordination.structure_regret import (
            decision_sufficient_plan,
            local_decision_margin,
        )
        scale = max(1.0, float(getattr(
            self, '_hyperedge_distance_scale_m', 150.0)))
        # stale drift: one frame of worst-case motion (v_max*dt) scaled by the
        # capability Lipschitz (~1/scale for exp(-d/scale)).
        v_max = float(getattr(self.cfg.uav, 'v_max', 25.0))
        dt = float(getattr(self.cfg.scenario, 'dt', 0.1))
        stale_lipschitz = max(
            1.0, float(getattr(
                self, '_distributed_movement_safety_margin_per_age_m', 0.0)))
        stale_base = max(
            1, int(getattr(
                self, '_distributed_movement_public_max_age_frames', 5)))
        configured_phys_sigma = max(0.0, float(getattr(
            self, '_distributed_target_position_uncertainty_sigma', 0.0)))
        dynamic_range = max(1e-9, float(getattr(
            self, '_distributed_decision_sufficient_dynamic_range', 1.0)))
        max_bits = max(1, int(getattr(
            self, '_distributed_decision_sufficient_max_bits', 32)))
        silenced_senders: List[int] = []
        saved_bits = 0
        for sender in list(self._pending_comm_messages):
            # C6 certifies that the sender's *local decision token* can be
            # omitted without changing its target ranking.  It does not
            # certify that control-plane state is already shared.  Silencing
            # a sender while a hyperedge/QPD/structure protocol packet is
            # pending creates a cold-start deadlock: peers never receive the
            # state needed to form a common structure, so no later decision
            # margin can repair the missing bootstrap.  Mandatory protocol
            # payloads therefore keep the carrier transmission alive; their
            # own suppression needs a protocol-specific common-view/AoI
            # certificate rather than the local top-2 margin used here.
            if (
                int(sender) in self._pending_hyperedge_protocol
                or int(sender) in self._pending_qpd_protocol
                or int(sender) in self._pending_structure_student_protocol
            ):
                continue
            try:
                position, _velocity = self._coordination_target_state_for_viewer(
                    int(sender))
            except Exception:
                # No truth fallback is allowed, but dropping the packet would
                # also pretend that the downstream decision was certified.
                # Keep the locally produced payload on air at the richest
                # available precision instead.
                if adaptive_bits:
                    self._pending_decision_sufficient_bits[int(sender)] = int(
                        max_bits)
                continue
            sender_xy = np.asarray(
                self.uavs[int(sender)].pos[:2], dtype=np.float64)
            target_xy = np.asarray(position)[:, :2]
            distances = np.linalg.norm(target_xy - sender_xy[None, :], axis=1)
            capability = np.exp(-distances / scale)
            margin = local_decision_margin(capability, top_k=2)
            stale = stale_lipschitz * stale_base * (v_max * dt / scale)
            belief_sigma = 0.0
            if self.belief_mgr is not None:
                covariance = np.asarray(
                    self.belief_mgr.cov[int(sender), :, :2, :2],
                    dtype=np.float64,
                )
                if covariance.shape == (self.Q, 2, 2) and np.all(
                    np.isfinite(covariance)
                ):
                    eigenvalues = np.linalg.eigvalsh(
                        0.5 * (covariance + np.swapaxes(covariance, -1, -2)))
                    belief_sigma = float(np.sqrt(max(
                        float(np.max(eigenvalues)), 0.0)))
                else:
                    # Invalid uncertainty cannot support a silence proof.
                    belief_sigma = float('inf')
            phys = max(configured_phys_sigma, belief_sigma) / scale

            # Silence preserves a peer decision only when every receiver still
            # holds a live token from this sender.  Use the weakest retained
            # precision; an absent/expired reference forces bootstrap.
            reference_bits = []
            for receiver in range(self.K):
                if receiver == int(sender):
                    continue
                metadata = self._received_comm_meta.get(receiver, {}).get(
                    int(sender))
                message = self._received_comm_msgs.get(receiver, {}).get(
                    int(sender))
                if metadata is None or message is None:
                    reference_bits = []
                    break
                age = int(metadata.get('age_frames', 0))
                if age > int(self._comm_message_ttl_frames):
                    reference_bits = []
                    break
                bits = metadata.get('bits_per_dim')
                if bits is None:
                    rate_index = int(metadata.get('rate_index', 0))
                    ladder = self._inter_uav_comm.rate_bits_per_dim
                    index = int(np.clip(rate_index, 0, len(ladder) - 1))
                    bits = int(ladder[index])
                if int(bits) < 1:
                    reference_bits = []
                    break
                reference_bits.append(int(bits))
            held_bits = min(reference_bits) if reference_bits else 0
            transmit, bits = decision_sufficient_plan(
                capability, dynamic_range,
                stale_drift_bound=stale,
                physical_error_bound=phys,
                max_bits=max_bits,
                reference_bits=held_bits,
            )
            if transmit and adaptive_bits:
                # Decision-preserving minimal precision for this sender's
                # token: transmit consumes it exactly (not ladder-rounded)
                # through ``exact_bits_per_dim``.
                self._pending_decision_sufficient_bits[int(sender)] = int(bits)
            if not transmit:
                rate_index = int(self._pending_comm_rates.get(int(sender), 0))
                active_dimensions = self._inter_uav_comm._active_dimensions(
                    self._pending_comm_token_masks.get(int(sender)))
                saved_bits += int(self._inter_uav_comm.payload_bits(
                    rate_index, active_dimensions=active_dimensions))
                silenced_senders.append(int(sender))
        for sender in silenced_senders:
            self._pending_comm_messages.pop(sender, None)
            self._pending_comm_rates.pop(sender, None)
            self._pending_comm_token_masks.pop(sender, None)
        self._last_isac_metrics['decision_sufficient_silenced'] = float(
            len(silenced_senders))
        self._last_isac_metrics['decision_sufficient_saved_bits'] = float(
            saved_bits)

    def _coordination_target_state_for_viewer(
        self,
        viewer: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the target state legally available to one coordinator.

        In dynamic distributed experiments this is a strict information
        boundary: invalid or unavailable local tracker state raises instead of
        silently exposing simulator ground truth.  Ground truth remains legal
        only for physical measurement generation and post-hoc scoring.

        Post-G2 strict no-truth closure (audit advice/001 P0, 2026-08-26):
        when ``distributed_no_truth_fail_closed`` is enabled the truth
        fallback below becomes UNREACHABLE -- the resolver fails closed rather
        than ever letting a distributed decision path read ``self.targets``.
        """
        viewer = int(viewer)
        if not 0 <= viewer < self.K:
            raise ValueError('viewer is outside the UAV index set')
        if self._distributed_coordination_use_local_belief_targets:
            if self.belief_mgr is None:
                raise RuntimeError(
                    'local target belief is unavailable during coordination')
            local = np.asarray(
                self.belief_mgr.mean[viewer], dtype=np.float64)
            if (
                local.ndim != 2 or local.shape[0] != self.Q
                or local.shape[1] < 4 or not np.all(np.isfinite(local))
            ):
                raise RuntimeError(
                    'local target belief is invalid during coordination')
            # CV and CA filters expose the same decision-sufficient
            # [x,y,vx,vy] prefix.  Acceleration remains tracker-private; the
            # strict wire/state boundary must not reject a valid 6-D CA state.
            local = local[:, :4]
            position = np.column_stack([
                local[:, :2],
                np.zeros(self.Q, dtype=np.float64),
            ])
            velocity = np.column_stack([
                local[:, 2:4],
                np.zeros(self.Q, dtype=np.float64),
            ])
            return position, velocity
        if self._distributed_no_truth_fail_closed:
            # Strict no-truth closure (audit advice/001 P0, 2026-08-26): a
            # distributed decision path must NEVER read simulator ground truth.
            # Local tracker state was validated above; reaching here means the
            # local-belief flag is off, so fail closed with an explicit error
            # instead of silently exposing ``self.targets`` to coordination.
            raise RuntimeError(
                'distributed_no_truth_fail_closed is enabled but '
                'distributed_coordination_use_local_belief_targets is off: '
                'refusing to expose simulator truth to a distributed decision'
            )
        position = np.asarray([
            target.get_position_3d() for target in self.targets
        ], dtype=np.float64)
        velocity = np.asarray([
            [*target.get_velocity(), 0.0] for target in self.targets
        ], dtype=np.float64)
        return position, velocity

    @staticmethod
    def _belief_position_disagreement(mean: np.ndarray) -> float:
        """Mean pairwise position disagreement over targets."""
        values = np.asarray(mean, dtype=np.float64)
        if values.ndim != 3 or values.shape[2] < 2 or values.shape[0] < 2:
            return 0.0
        differences = (
            values[:, None, :, :2] - values[None, :, :, :2])
        distances = np.linalg.norm(differences, axis=-1)
        upper = np.triu_indices(values.shape[0], k=1)
        return float(np.mean(distances[upper]))

    def _strict_no_truth_target_map(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Public median local-belief target map for strict no-truth geometry.

        Post-G2 strict closure (audit advice/001 P0, 2026-08-26): the deployed
        L3 movement hook and the frame-0 analytical warm start consume a target
        map.  Under ``distributed_no_truth_fail_closed`` that map must be the
        public reconstructable median of the per-viewer local beliefs (what a
        distributed system can assemble from delivered anchors + own tracker),
        never simulator ground truth.  Fails closed when no belief manager or
        invalid beliefs exist.
        """
        if not self._distributed_no_truth_fail_closed:
            raise RuntimeError(
                'strict target map requested while '
                'distributed_no_truth_fail_closed is off')
        if self.belief_mgr is None:
            raise RuntimeError(
                'strict no-truth geometry requires the belief manager')
        mean = np.asarray(self.belief_mgr.mean, dtype=np.float64)  # (K,Q,4)
        if mean.shape != (self.K, self.Q, 4) or not np.all(np.isfinite(mean)):
            raise RuntimeError(
                'strict no-truth geometry requires finite local beliefs')
        pos = np.median(mean[:, :, :2], axis=0)          # (Q,2) public median
        vel = np.median(mean[:, :, 2:4], axis=0)         # (Q,2) public median
        return pos, vel

    def _movement_target_state_for_viewer(
        self,
        viewer: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return local target state with physically received anchor positions.

        Anchors replace positions only while fresh. Velocity remains the
        viewer's own CV estimate and propagates a cached anchor by its public
        age. No simulator target truth or global owner table is read.
        """
        position, velocity = self._coordination_target_state_for_viewer(viewer)
        if not self._distributed_movement_anchor_broadcast_enabled:
            return position, velocity
        position = position.copy()
        last_seen = self._distributed_movement_anchor_last_seen[int(viewer)]
        fresh = (
            (last_seen > FRAME_NEVER)
            & (
                self.t - last_seen
                <= self._distributed_movement_anchor_max_age_frames
            )
        )
        for target in range(self.Q):
            sources = np.flatnonzero(fresh[:, target])
            if sources.size == 0:
                continue
            age = (
                self.t - last_seen[sources, target]
            ).astype(np.float64)
            propagated = (
                self._distributed_movement_anchor_target_xy[
                    int(viewer), sources, target]
                + age[:, None]
                * float(self.uavs[int(viewer)].dt)
                * velocity[target, :2][None, :]
            )
            position[target, :2] = np.median(propagated, axis=0)
        return position, velocity

    def _belief_freshness_schedule_mask(
        self,
        evidence_owner: np.ndarray,
        evidence_mask: np.ndarray,
    ) -> np.ndarray:
        """Physically charged sparse posterior broadcast schedule.

        Evidence LLRs and 4D posteriors are different information and cannot
        substitute for one another. Each source ranks its own posteriors by

            score_kq = w_aoi * aoi_norm + w_unc * unc_norm
                       + w_task * task_norm

        and selects entries subject to (a) a per-source union cap
        (evidence union posteriors <= ``u2u_belief_feedback_max_union_per_source``,
        which bounds the per-packet payload and keeps the broadcast inside the
        frame deadline) and (b) a global posterior-bit budget. Entries whose
        source AoI exceeds the receiver-side fusion guard or are owned fusion
        targets (when ``u2u_belief_feedback_owner_aware``) are excluded. A
        posterior may share a target ID with an evidence entry, but still pays
        its full posterior payload. Each signal
        is min-max normalized per source row over its own Q posteriors, so the
        schedule is deterministic from purely local state and needs no global
        quality matrix, no ground link and no ACK.
        """
        if self.belief_mgr is None:
            return np.zeros((self.K, self.Q), dtype=bool)
        aoi = np.asarray(self.belief_mgr.aoi, dtype=np.float64)
        cov = np.asarray(self.belief_mgr.cov, dtype=np.float64)
        uncertainty = cov[..., 0, 0] + cov[..., 1, 1]
        deficit = np.zeros((self.K, self.Q), dtype=np.float64)
        for source in range(self.K):
            local_pd = np.asarray(
                self.prev_P_D_local.get(
                    source, np.zeros(self.Q, dtype=np.float64)),
                dtype=np.float64,
            ).reshape(-1)
            if local_pd.shape == (self.Q,):
                deficit[source] = np.clip(1.0 - local_pd, 0.0, 1.0)

        def row_normalized(values: np.ndarray) -> np.ndarray:
            out = np.zeros_like(values, dtype=np.float64)
            for source in range(self.K):
                row = np.asarray(values[source], dtype=np.float64)
                lo = float(np.min(row))
                hi = float(np.max(row))
                if hi - lo > 1.0e-12:
                    out[source] = (row - lo) / (hi - lo)
            return out

        score = (
            self._u2u_belief_feedback_aoi_weight
            * row_normalized(aoi)
            + self._u2u_belief_feedback_uncertainty_weight
            * row_normalized(uncertainty)
            + self._u2u_belief_feedback_task_weight
            * row_normalized(deficit)
        )
        # Entries whose source AoI already exceeds the receiver-side fusion
        # guard cannot be fused and must not consume broadcast budget.
        stale = aoi > float(self._u2u_belief_feedback_max_age_frames)
        score[stale] = -np.inf
        owner = np.asarray(evidence_owner, dtype=np.int64).reshape(-1)
        owned = owner >= 0
        owner_mask = np.zeros((self.K, self.Q), dtype=bool)
        owner_mask[owner[owned], np.arange(self.Q)[owned]] = True
        if self._u2u_belief_feedback_owner_aware:
            score[owner_mask] = -np.inf
        evidence = np.asarray(evidence_mask, dtype=bool)
        if evidence.shape != (self.K, self.Q):
            raise ValueError('evidence_mask must have shape (K, Q)')
        topk = max(1, int(self._u2u_belief_feedback_topk))
        union_cap = max(1, int(self._u2u_belief_feedback_max_union_per_source))
        union_count = np.sum(evidence, axis=1).astype(np.int64)
        # The posterior payload must fit inside the frame window shared with
        # the coordination and evidence broadcasts (unified-MAC budget): select
        # top-ups greedily by score subject to a global posterior-bit budget.
        per_entry_bits = max(
            1,
            4 * self._u2u_belief_feedback_mean_bits
            + 4 * self._u2u_belief_feedback_cov_bits
            + (
                self._evidence_legacy_service_aoi_bits
                if self._evidence_legacy_service_envelope_enabled
                else self._u2u_belief_feedback_aoi_bits
            ),
        )
        budget_entries = (
            self._u2u_belief_feedback_bit_budget // per_entry_bits
            if self._u2u_belief_feedback_bit_budget > 0
            else self.K * self.Q
        )
        scored = []
        for source in range(self.K):
            row = score[source]
            finite_indices = np.flatnonzero(np.isfinite(row))
            if finite_indices.size == 0:
                continue
            order = finite_indices[np.argsort(
                -row[finite_indices], kind='stable')]
            for q in order[:topk]:
                scored.append((float(row[q]), int(source), int(q)))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        mask = np.zeros((self.K, self.Q), dtype=bool)
        selected_entries = 0
        for _value, source, target in scored:
            if selected_entries >= budget_entries:
                break
            adds_union_entry = not bool(evidence[source, target])
            if adds_union_entry and union_count[source] >= union_cap:
                continue
            mask[source, target] = True
            union_count[source] += int(adds_union_entry)
            selected_entries += 1
        return mask

    def _apply_u2u_belief_feedback(
        self,
        evidence_transport,
        source_mean: np.ndarray,
        source_covariance: np.ndarray,
        source_aoi: np.ndarray,
    ) -> Dict[str, float]:
        """Fuse one physically delivered cross-frame posterior round."""
        metrics = {
            'belief_feedback_enabled': float(
                self._u2u_belief_feedback_enabled),
            'belief_feedback_fused_entries': 0.0,
            'belief_feedback_receivers': 0.0,
            'belief_feedback_disagreement_before_m': 0.0,
            'belief_feedback_disagreement_after_m': 0.0,
            'belief_feedback_contraction_ratio': 1.0,
            'belief_feedback_truth_rmse_before_m': 0.0,
            'belief_feedback_truth_rmse_after_m': 0.0,
        }
        if (
            not self._u2u_belief_feedback_enabled
            or evidence_transport is None
            or self.belief_mgr is None
        ):
            return metrics
        if self.belief_mgr.state_dim != 4:
            raise RuntimeError(
                'U2U belief feedback currently requires a 4D CV belief')
        means = np.asarray(source_mean, dtype=np.float64)
        covariances = np.asarray(source_covariance, dtype=np.float64)
        ages = np.asarray(source_aoi, dtype=np.int64)
        if means.shape != (self.K, self.Q, 4):
            raise ValueError('feedback source means have invalid shape')
        if covariances.shape != (self.K, self.Q, 4, 4):
            raise ValueError('feedback source covariances have invalid shape')
        if ages.shape != (self.K, self.Q):
            raise ValueError('feedback source AoI has invalid shape')

        true_position = np.asarray([
            target.get_position_3d()[:2] for target in self.targets
        ], dtype=np.float64)
        before = self._belief_position_disagreement(self.belief_mgr.mean)
        truth_rmse_before = float(np.sqrt(np.mean(np.square(np.linalg.norm(
            self.belief_mgr.mean[:, :, :2]
            - true_position[None, :, :], axis=-1)))))
        new_mean = self.belief_mgr.mean.copy()
        new_covariance = self.belief_mgr.cov.copy()
        new_aoi = self.belief_mgr.aoi.copy()
        selected = np.asarray(
            evidence_transport.belief_schedule_mask, dtype=bool)
        delivery = np.asarray(
            evidence_transport.delivery_matrix, dtype=bool)
        decoded = {}
        fused_entries = 0
        receivers_used = set()
        for receiver in range(self.K):
            for target in range(self.Q):
                local_means = [self.belief_mgr.mean[receiver, target].copy()]
                local_covariances = [
                    self.belief_mgr.cov[receiver, target].copy()]
                local_ages = [int(self.belief_mgr.aoi[receiver, target])]
                for source in range(self.K):
                    encoded_age = min(
                        max(int(ages[source, target]), 0),
                        (1 << self._u2u_belief_feedback_aoi_bits) - 1,
                    )
                    if (
                        source == receiver
                        or not selected[source, target]
                        or not delivery[source, receiver]
                        or encoded_age
                        > self._u2u_belief_feedback_max_age_frames
                    ):
                        continue
                    key = (source, target)
                    if key not in decoded:
                        decoded_mean, decoded_covariance = (
                            quantize_belief_feedback(
                                means[source, target],
                                covariances[source, target],
                                area_size_xy=(
                                    float(self.area_size[0]),
                                    float(self.area_size[1]),
                                ),
                                velocity_bound_mps=max(
                                    float(self.cfg.uav.v_max),
                                    float(self.cfg.target.speed_range[1])
                                    + 4.0 * float(self.cfg.target.sigma_a),
                                    1.0,
                                ),
                                mean_bits=(
                                    self._u2u_belief_feedback_mean_bits),
                                covariance_bits=(
                                    self._u2u_belief_feedback_cov_bits),
                            )
                        )
                        # The packet contains the previous-round posterior.
                        # Propagate it once to the receiver's current time.
                        propagated_mean = self.belief_mgr.F @ decoded_mean
                        propagated_covariance = (
                            self.belief_mgr.F @ decoded_covariance
                            @ self.belief_mgr.F.T
                            + self.belief_mgr.Q_proc_base
                        )
                        decoded[key] = (
                            propagated_mean,
                            propagated_covariance,
                            encoded_age + 1,
                        )
                    decoded_mean, decoded_covariance, decoded_age = decoded[key]
                    local_means.append(decoded_mean)
                    local_covariances.append(decoded_covariance)
                    local_ages.append(decoded_age)
                    fused_entries += 1
                if len(local_means) <= 1:
                    continue
                fused_mean, fused_covariance = (
                    generalized_covariance_intersection(
                        np.asarray(local_means),
                        np.asarray(local_covariances),
                    )
                )
                new_mean[receiver, target] = fused_mean
                new_covariance[receiver, target] = fused_covariance
                new_aoi[receiver, target] = min(local_ages)
                receivers_used.add(receiver)
        self.belief_mgr.mean[:] = new_mean
        self.belief_mgr.cov[:] = new_covariance
        self.belief_mgr.aoi[:] = new_aoi
        after = self._belief_position_disagreement(self.belief_mgr.mean)
        truth_rmse_after = float(np.sqrt(np.mean(np.square(np.linalg.norm(
            self.belief_mgr.mean[:, :, :2]
            - true_position[None, :, :], axis=-1)))))
        metrics.update({
            'belief_feedback_fused_entries': float(fused_entries),
            'belief_feedback_receivers': float(len(receivers_used)),
            'belief_feedback_disagreement_before_m': float(before),
            'belief_feedback_disagreement_after_m': float(after),
            'belief_feedback_contraction_ratio': float(
                after / max(before, 1.0e-12)),
            'belief_feedback_truth_rmse_before_m': truth_rmse_before,
            'belief_feedback_truth_rmse_after_m': truth_rmse_after,
        })
        return metrics

    def _prepare_hyperedge_submission(self) -> None:
        """Append a physically charged local Tx/Rx/deficit offer stream."""
        if len(self.uavs) != self.K or len(self.targets) != self.Q:
            return
        uav_xy = np.asarray([uav.pos[:2] for uav in self.uavs])

        self._pending_hyperedge_protocol = {}
        self._pending_hyperedge_protocol_frame = {}
        for sender in range(self.K):
            anchor_payload = False
            sender_target_position, _ = (
                self._coordination_target_state_for_viewer(sender))
            sender_target_xy = sender_target_position[:, :2]
            sender_distance = np.linalg.norm(
                uav_xy[sender, None, :] - sender_target_xy,
                axis=-1,
            )
            comm_fraction = float(self._pending_comm_power_fractions.get(
                sender, 0.0))
            sensing_weights = np.asarray(
                self._pending_sensing_weights.get(
                    sender, np.full(self.Q, 1.0 / max(self.Q, 1))),
                dtype=np.float64,
            )
            # Tx capability includes the sender's actual remaining sensing
            # fraction and target split. Rx capability is geometric because
            # receive participation does not emit sensing RF power.
            tx_capability, rx_capability = (
                factorized_endpoint_capabilities(
                    sender_distance,
                    (1.0 - comm_fraction) * sensing_weights,
                    distance_scale_m=self._hyperedge_distance_scale_m,
                    mode=self._hyperedge_capability_mode,
                )
            )
            local_pd = np.asarray(
                self.prev_P_D_local.get(
                    sender, np.zeros(self.Q, dtype=np.float64)),
                dtype=np.float64,
            )
            if local_pd.shape != (self.Q,):
                local_pd = np.zeros(self.Q, dtype=np.float64)
            deficit = np.clip(1.0 - local_pd, 0.0, 1.0)
            if self._hyperedge_pair_score_mode == 'budget_reconstructable':
                width = max(float(self.area_size[0]), 1e-9)
                height = max(float(self.area_size[1]), 1e-9)
                speed = max(float(self.cfg.uav.v_max), 1e-9)
                state_normalized = np.array([
                    np.clip(self.uavs[sender].pos[0] / width, 0.0, 1.0),
                    np.clip(self.uavs[sender].pos[1] / height, 0.0, 1.0),
                    np.clip(
                        0.5 * (self.uavs[sender].vel[0] / speed + 1.0),
                        0.0, 1.0),
                    np.clip(
                        0.5 * (self.uavs[sender].vel[1] / speed + 1.0),
                        0.0, 1.0),
                ], dtype=np.float64)
                if self._hyperedge_nearfield_residual_enabled:
                    anchor_payload = bool(
                        self._distributed_movement_anchor_broadcast_enabled
                        and self.t
                        % self._distributed_movement_anchor_broadcast_period_frames
                        == 0
                    )
                    if anchor_payload:
                        anchor_round = int(
                            self.t
                            // self._distributed_movement_anchor_broadcast_period_frames)
                        target_code = int(
                            (sender + anchor_round) % self.Q)
                        nearfield = np.asarray([
                            float(target_code) / max(float(self.Q), 1.0),
                            np.clip(
                                sender_target_xy[target_code, 0] / width,
                                0.0, 1.0),
                            np.clip(
                                sender_target_xy[target_code, 1] / height,
                                0.0, 1.0),
                        ], dtype=np.float64)
                        anchor_bits = min(
                            self._hyperedge_state_field_bits[5:7])
                        levels = max((1 << anchor_bits) - 1, 1)
                        quantized_anchor = np.rint(
                            nearfield[1:] * levels) / levels
                        self._distributed_movement_anchor_target_xy[
                            sender, sender, target_code] = (
                                quantized_anchor
                                * np.asarray([width, height]))
                        self._distributed_movement_anchor_last_seen[
                            sender, sender, target_code] = int(self.t)
                    else:
                        nearest = int(np.argmin(sender_distance))
                        residual_range = float(
                            self._hyperedge_nearfield_residual_range_m)
                        companding_mu = float(
                            self._hyperedge_nearfield_residual_companding_mu)
                        use_residual = bool(
                            companding_mu > 0.0
                            or sender_distance[nearest] <= residual_range)
                        # Code Q is an explicit absolute-position fallback.
                        target_code = nearest if use_residual else self.Q
                        residual = (
                            uav_xy[sender] - sender_target_xy[nearest]
                            if use_residual else np.zeros(2, dtype=np.float64)
                        )
                        if companding_mu > 0.0:
                            companded = (
                                np.sign(residual)
                                * np.log1p(
                                    companding_mu
                                    * np.minimum(
                                        np.abs(residual), residual_range)
                                    / residual_range)
                                / np.log1p(companding_mu)
                            )
                        else:
                            companded = residual / residual_range
                        nearfield = np.asarray([
                            float(target_code) / max(float(self.Q), 1.0),
                            np.clip(
                                0.5 * (companded[0] + 1.0),
                                0.0, 1.0),
                            np.clip(
                                0.5 * (companded[1] + 1.0),
                                0.0, 1.0),
                        ], dtype=np.float64)
                    state_normalized = np.concatenate(
                        [state_normalized, nearfield])
                if self._hyperedge_state_relay_enabled:
                    # Rotate one cached physical state through the beacon.
                    # Source code K is the explicit no-relay sentinel. Cached
                    # values have already passed the physical link/codec.
                    relay_source = self.K
                    relay_state = np.zeros(
                        self._hyperedge_base_state_dim, dtype=np.float64)
                    seen = np.any(
                        self._hyperedge_received_last_seen[sender] > FRAME_NEVER,
                        axis=1,
                    )
                    seen[sender] = False
                    for offset in range(self.K):
                        candidate = int((self.t + sender + offset) % self.K)
                        if seen[candidate]:
                            relay_source = candidate
                            relay_state = self._hyperedge_received_offer[
                                sender, candidate, 0,
                                :self._hyperedge_base_state_dim,
                            ].copy()
                            break
                    state_normalized = np.concatenate([
                        state_normalized,
                        np.asarray([
                            float(relay_source) / max(float(self.K), 1.0)
                        ]),
                        relay_state,
                    ])
                decoded = np.repeat(
                    state_normalized[None, :], self.Q, axis=0)
                encoded = 2.0 * decoded - 1.0
            else:
                decoded = np.stack(
                    [tx_capability, rx_capability, deficit], axis=-1)
                encoded = encode_offer_stream(
                    tx_capability, rx_capability, deficit)
                if self._hyperedge_state_stream_enabled:
                    width = max(float(self.area_size[0]), 1e-9)
                    height = max(float(self.area_size[1]), 1e-9)
                    speed = max(float(self.cfg.uav.v_max), 1e-9)
                    state_normalized = np.array([
                        np.clip(self.uavs[sender].pos[0] / width, 0.0, 1.0),
                        np.clip(self.uavs[sender].pos[1] / height, 0.0, 1.0),
                        np.clip(
                            0.5 * (self.uavs[sender].vel[0] / speed + 1.0),
                            0.0, 1.0),
                        np.clip(
                            0.5 * (self.uavs[sender].vel[1] / speed + 1.0),
                            0.0, 1.0),
                    ], dtype=np.float64)
                    repeated_state = np.repeat(
                        state_normalized[None, :], self.Q, axis=0)
                    decoded = np.concatenate(
                        [decoded, repeated_state], axis=-1)
                    encoded = np.concatenate(
                        [encoded, 2.0 * repeated_state - 1.0], axis=-1)
            local_decoded = decoded
            if anchor_payload:
                # The multiplexed target anchor is consumed by the movement
                # cache.  Do not reinterpret its absolute x/y fields as a
                # near-field residual in the physical structure decoder.
                local_decoded = decoded.copy()
                local_decoded[:, 4:7] = np.asarray([1.0, 0.5, 0.5])
            self._hyperedge_local_offer[sender] = local_decoded
            scheduled = bool(
                sender % self._hyperedge_beacon_round_robin_period
                == self.t % self._hyperedge_beacon_round_robin_period)
            if scheduled:
                self._pending_hyperedge_protocol[sender] = encoded
                self._pending_hyperedge_protocol_frame[sender] = int(self.t)

            priority = deficit * np.maximum(
                tx_capability, rx_capability)
            order = np.argsort(-priority, kind='stable')
            mask = np.zeros(self.Q, dtype=np.float64)
            mask[order[:self._hyperedge_share_topk]] = 1.0
            self._pending_comm_token_masks[sender] = mask

    def _decode_hyperedge_packet(
        self,
        protocol: np.ndarray,
        token_mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        stream = np.asarray(protocol, dtype=np.float64).reshape(
            self.Q, self._hyperedge_protocol_dim)
        active = (
            np.ones(self.Q, dtype=bool)
            if token_mask is None
            else np.asarray(token_mask, dtype=np.float64).reshape(-1) > 0.5
        )
        if self._hyperedge_pair_score_mode == 'budget_reconstructable':
            # One active target token carries the node-level state header.
            # Expand it to every target because endpoint state is target
            # independent; no untransmitted target-specific value is created.
            active_indices = np.flatnonzero(active)
            if active_indices.size == 0:
                return np.zeros_like(stream), np.zeros(self.Q, dtype=bool)
            state = np.clip(
                0.5 * (stream[int(active_indices[0])] + 1.0), 0.0, 1.0)
            decoded = np.repeat(state[None, :], self.Q, axis=0)
            return decoded, np.ones(self.Q, dtype=bool)
        decoded = decode_offer_stream(stream[:, :3])
        if self._hyperedge_state_stream_enabled:
            decoded = np.concatenate([
                decoded,
                np.clip(0.5 * (stream[:, 3:] + 1.0), 0.0, 1.0),
            ], axis=-1)
        return decoded, active

    def _merge_received_hyperedge_packet(
        self,
        receiver: int,
        sender: int,
        protocol: np.ndarray,
        token_mask: Optional[np.ndarray],
        sent_frame: Optional[int] = None,
    ) -> None:
        decoded, active = self._decode_hyperedge_packet(
            protocol, token_mask)
        packet_frame = int(self.t if sent_frame is None else sent_frame)
        if (
            self._distributed_movement_anchor_broadcast_enabled
            and packet_frame
            % self._distributed_movement_anchor_broadcast_period_frames
            == 0
            and np.any(active)
        ):
            first = decoded[int(np.flatnonzero(active)[0])]
            target_code = int(np.rint(float(first[4]) * float(self.Q)))
            if 0 <= target_code < self.Q:
                self._distributed_movement_anchor_target_xy[
                    int(receiver), int(sender), target_code] = np.asarray([
                        float(first[5]) * float(self.area_size[0]),
                        float(first[6]) * float(self.area_size[1]),
                    ])
                self._distributed_movement_anchor_last_seen[
                    int(receiver), int(sender), target_code] = packet_frame
            decoded = decoded.copy()
            decoded[:, 4:7] = np.asarray([1.0, 0.5, 0.5])
        self._hyperedge_received_offer[
            int(receiver), int(sender), active] = decoded[active]
        self._hyperedge_received_last_seen[
            int(receiver), int(sender), active] = packet_frame
        if (
            self._hyperedge_pair_score_mode == 'budget_reconstructable'
            and self._hyperedge_state_relay_enabled
            and np.any(active)
        ):
            first = decoded[int(np.flatnonzero(active)[0])]
            base = self._hyperedge_base_state_dim
            relay_source = int(np.rint(float(first[base]) * float(self.K)))
            if 0 <= relay_source < self.K and relay_source != int(sender):
                relay_state = np.asarray(
                    first[base + 1:base + 1 + base],
                    dtype=np.float64,
                )
                relayed = np.zeros(
                    (self.Q, self._hyperedge_protocol_dim),
                    dtype=np.float64,
                )
                relayed[:, :base] = relay_state[None, :]
                self._hyperedge_received_offer[
                    int(receiver), relay_source] = relayed
                # The compact relay schema carries state but no original
                # source timestamp.  Never relabel cached state as fresh at
                # relay reception: use the oldest source time permitted by
                # the sender cache TTL.  This is conservative (possibly more
                # stale than reality) and therefore preserves fail-closed AoI
                # semantics without adding hidden metadata.
                self._hyperedge_received_last_seen[
                    int(receiver), relay_source] = (
                        packet_frame - self._comm_message_ttl_frames)

    def _merge_received_hyperedge_packet_many(
        self,
        receivers: np.ndarray,
        sender: int,
        protocol: np.ndarray,
        token_mask: Optional[np.ndarray],
        sent_frame: Optional[int] = None,
    ) -> None:
        """Decode one immutable broadcast once and scatter to receivers."""
        receiver_index = np.asarray(receivers, dtype=np.int64).reshape(-1)
        if receiver_index.size == 0:
            return
        if (
            self._distributed_movement_anchor_broadcast_enabled
            or self._hyperedge_state_relay_enabled
        ):
            for receiver in receiver_index:
                self._merge_received_hyperedge_packet(
                    int(receiver), sender, protocol, token_mask, sent_frame)
            return
        if (
            np.any(receiver_index < 0)
            or np.any(receiver_index >= self.K)
            or np.unique(receiver_index).size != receiver_index.size
        ):
            raise ValueError("invalid hyperedge broadcast receiver set")
        decoded, active = self._decode_hyperedge_packet(
            protocol, token_mask)
        active_index = np.flatnonzero(active)
        packet_frame = int(self.t if sent_frame is None else sent_frame)
        if active_index.size:
            self._hyperedge_received_offer[
                receiver_index[:, None],
                int(sender),
                active_index[None, :],
            ] = decoded[active_index][None, :, :]
            self._hyperedge_received_last_seen[
                receiver_index[:, None],
                int(sender),
                active_index[None, :],
            ] = packet_frame

    def _reset_owner_posterior_state(self) -> None:
        """Reset physically transported receiver-owner posterior state."""
        state_dim = int(self._owner_posterior_state_dim)
        self._owner_posterior_pending_mean = np.zeros(
            (self.K, self.Q, state_dim), dtype=np.float64)
        self._owner_posterior_pending_cov = np.zeros(
            (self.K, self.Q, state_dim, state_dim), dtype=np.float64)
        self._owner_posterior_pending_aoi = np.zeros(
            (self.K, self.Q), dtype=np.int64)
        self._owner_posterior_pending_frame = np.full(
            (self.K, self.Q), -1, dtype=np.int64)
        self._owner_posterior_pending_valid = np.zeros(
            (self.K, self.Q), dtype=bool)
        self._owner_posterior_received_mean = np.zeros(
            (self.K, self.K, self.Q, state_dim), dtype=np.float64)
        self._owner_posterior_received_cov = np.zeros(
            (self.K, self.K, self.Q, state_dim, state_dim), dtype=np.float64)
        self._owner_posterior_received_aoi = np.zeros(
            (self.K, self.K, self.Q), dtype=np.int64)
        self._owner_posterior_received_frame = np.full(
            (self.K, self.K, self.Q), -1, dtype=np.int64)
        self._owner_posterior_last_fused_frame = np.full(
            (self.K, self.K, self.Q), -1, dtype=np.int64)
        self._owner_posterior_fusion_queue = set()
        self._owner_posterior_metrics: Dict[str, object] = {}

    def _merge_composable_certificate_packet(
        self,
        receiver: int,
        sender: int,
        contribution_lower: np.ndarray,
        row_dual_upper: float,
        source_frame: int,
        row_targetwise_upper: np.ndarray | None = None,
    ) -> None:
        """Store one physically delivered row certificate at a receiver."""
        values = np.asarray(contribution_lower, dtype=np.float64).reshape(-1)
        if (
            values.shape != (self.Q,)
            or np.any(~np.isfinite(values))
            or np.any(values < 0.0)
        ):
            raise ValueError("invalid composable certificate payload")
        self._composable_certificate_received_lower[
            int(receiver), int(sender)] = values
        self._composable_certificate_received_frame[
            int(receiver), int(sender)] = int(source_frame)
        upper = float(row_dual_upper)
        if not np.isfinite(upper) or upper < 0.0:
            raise ValueError("invalid composable dual-upper payload")
        self._composable_certificate_received_dual_upper[
            int(receiver), int(sender)] = upper
        if self._composable_certificate_targetwise_upper_enabled:
            targetwise = np.asarray(
                row_targetwise_upper, dtype=np.float64).reshape(-1)
            if (
                targetwise.shape != (self.Q,)
                or np.any(~np.isfinite(targetwise))
                or np.any(targetwise < 0.0)
            ):
                raise ValueError("invalid targetwise dual-upper payload")
            self._composable_certificate_received_targetwise_upper[
                int(receiver), int(sender)] = targetwise

    def _merge_composable_certificate_packet_many(
        self,
        receivers: np.ndarray,
        sender: int,
        contribution_lower: np.ndarray,
        row_dual_upper: float,
        source_frame: int,
        row_targetwise_upper: np.ndarray | None = None,
    ) -> None:
        """Validate one row certificate once and scatter its broadcast."""
        receiver_index = np.asarray(receivers, dtype=np.int64).reshape(-1)
        values = np.asarray(contribution_lower, dtype=np.float64).reshape(-1)
        upper = float(row_dual_upper)
        if (
            values.shape != (self.Q,)
            or np.any(~np.isfinite(values))
            or np.any(values < 0.0)
            or not np.isfinite(upper)
            or upper < 0.0
            or np.any(receiver_index < 0)
            or np.any(receiver_index >= self.K)
            or np.unique(receiver_index).size != receiver_index.size
        ):
            raise ValueError("invalid composable certificate broadcast")
        self._composable_certificate_received_lower[
            receiver_index, int(sender)] = values[None, :]
        self._composable_certificate_received_frame[
            receiver_index, int(sender)] = int(source_frame)
        self._composable_certificate_received_dual_upper[
            receiver_index, int(sender)] = upper
        if self._composable_certificate_targetwise_upper_enabled:
            targetwise = np.asarray(
                row_targetwise_upper, dtype=np.float64).reshape(-1)
            if (
                targetwise.shape != (self.Q,)
                or np.any(~np.isfinite(targetwise))
                or np.any(targetwise < 0.0)
            ):
                raise ValueError("invalid targetwise dual-upper broadcast")
            self._composable_certificate_received_targetwise_upper[
                receiver_index, int(sender)] = targetwise[None, :]

    def _refresh_composable_certificate_metrics(self) -> None:
        """Compose same-frame row reports at q mod K responsibility nodes."""
        if not self._composable_certificate_enabled:
            self._composable_certificate_metrics = {}
            return
        reports = self._composable_certificate_received_lower.copy()
        frames = self._composable_certificate_received_frame.copy()
        dual_upper = self._composable_certificate_received_dual_upper.copy()
        targetwise_upper = (
            self._composable_certificate_received_targetwise_upper.copy()
            if self._composable_certificate_targetwise_upper_enabled
            else None)
        # A responsibility node does not need to transmit its row to itself.
        # Insert that node's own local report into the corresponding diagonal.
        for owner in range(self.K):
            reports[owner, owner] = (
                self._composable_certificate_last_sent_lower[owner])
            frames[owner, owner] = int(
                self._composable_certificate_last_sent_frame[owner])
            dual_upper[owner, owner] = float(
                self._composable_certificate_last_sent_dual_upper[owner])
            if targetwise_upper is not None:
                targetwise_upper[owner, owner] = (
                    self._composable_certificate_last_sent_targetwise_upper[
                        owner])
        certificate = aggregate_target_responsibility_certificate(
            reports,
            frames,
            target_owner=np.arange(self.Q, dtype=np.int64) % self.K,
            max_age_frames=self._composable_certificate_max_age,
            current_frame=int(self.t),
            row_dual_upper=dual_upper,
            row_targetwise_upper=targetwise_upper,
        )
        complete = certificate.target_complete
        target_lower = certificate.target_deflection_lower
        pd_lower = compute_detection_probabilities(
            target_lower, self.cfg.detection.P_FA)
        # Incomplete targets fail closed to zero service probability.  The
        # detector formula at D=0 equals P_FA and must not mask packet loss.
        pd_lower = np.where(complete, pd_lower, 0.0)
        self._composable_certificate_metrics = {
            'composable_certificate_enabled': 1.0,
            'composable_certificate_complete_fraction': float(np.mean(
                complete)),
            'composable_certificate_all_targets_complete': float(np.all(
                complete)),
            'composable_certificate_worst_deflection_lower': float(
                np.min(target_lower) if np.all(complete) else 0.0),
            'composable_certificate_worst_pd_lower': float(np.min(pd_lower)),
            'composable_certificate_global_optimum_upper': float(
                certificate.global_optimum_upper),
            'composable_certificate_joint_approximation_ratio_lower': float(
                certificate.joint_approximation_ratio_lower),
            'composable_certificate_target_deflection_lower': (
                target_lower.copy()),
            'composable_certificate_target_pd_lower': pd_lower.copy(),
            'composable_certificate_source_frame': (
                certificate.source_frame.copy()),
            'composable_certificate_payload_bits_per_sender': float(
                (self.Q + (
                    self.Q
                    if self._composable_certificate_targetwise_upper_enabled
                    else 1
                )) * self._composable_certificate_bits
                + self._composable_certificate_frame_bits),
        }

    def _prepare_owner_posterior_submission(
        self,
        selected_set: Tuple[Tuple[int, int, int], ...],
    ) -> None:
        """Snapshot each target receiver's causal posterior for next carrier."""
        if not self._owner_posterior_enabled:
            return
        if (
            self.belief_mgr is None
            or self.belief_mgr.state_dim
            != int(self._owner_posterior_state_dim)
        ):
            raise RuntimeError(
                'distributed owner posterior state contract does not match '
                'the local belief model')
        state_dim = int(self._owner_posterior_state_dim)
        owners = np.full(self.Q, -1, dtype=np.int64)
        for _transmitter, receiver, target in selected_set:
            j, q = int(receiver), int(target)
            if owners[q] not in (-1, j):
                raise RuntimeError(
                    'owner posterior requires one sensing receiver per target')
            owners[q] = j
        self._owner_posterior_pending_valid.fill(False)
        for q, raw_owner in enumerate(owners):
            owner = int(raw_owner)
            if owner < 0:
                continue
            self._owner_posterior_pending_mean[owner, q] = (
                self.belief_mgr.mean[owner, q, :state_dim])
            self._owner_posterior_pending_cov[owner, q] = (
                self.belief_mgr.cov[owner, q, :state_dim, :state_dim])
            self._owner_posterior_pending_aoi[owner, q] = int(
                self.belief_mgr.aoi[owner, q])
            self._owner_posterior_pending_frame[owner, q] = int(self.t)
            self._owner_posterior_pending_valid[owner, q] = True

    def _merge_owner_posterior_packet(
        self,
        receiver: int,
        sender: int,
        payload: dict,
    ) -> None:
        """Store one delivered sparse posterior packet without fusing it yet."""
        targets = np.asarray(payload.get('targets', []), dtype=np.int64)
        means = np.asarray(payload.get('means', []), dtype=np.float64)
        covariances = np.asarray(
            payload.get('covariances', []), dtype=np.float64)
        ages = np.asarray(payload.get('aoi', []), dtype=np.int64)
        frames = np.asarray(payload.get('frames', []), dtype=np.int64)
        count = targets.size
        state_dim = int(self._owner_posterior_state_dim)
        if (
            means.shape != (count, state_dim)
            or covariances.shape != (count, state_dim, state_dim)
            or ages.shape != (count,)
            or frames.shape != (count,)
            or np.any(targets < 0) or np.any(targets >= self.Q)
            or np.any(~np.isfinite(means))
            or np.any(~np.isfinite(covariances))
            or np.any(ages < 0)
        ):
            raise ValueError('invalid owner-posterior packet')
        for index, target in enumerate(targets):
            q = int(target)
            self._owner_posterior_received_mean[
                int(receiver), int(sender), q] = means[index]
            self._owner_posterior_received_cov[
                int(receiver), int(sender), q] = covariances[index]
            self._owner_posterior_received_aoi[
                int(receiver), int(sender), q] = int(ages[index])
            self._owner_posterior_received_frame[
                int(receiver), int(sender), q] = int(frames[index])
            self._owner_posterior_fusion_queue.add((
                int(receiver), int(sender), q))

    def _merge_owner_posterior_packet_many(
        self,
        receivers: np.ndarray,
        sender: int,
        payload: dict,
    ) -> None:
        """Validate a sparse posterior broadcast once and scatter it."""
        receiver_index = np.asarray(receivers, dtype=np.int64).reshape(-1)
        targets = np.asarray(payload.get('targets', []), dtype=np.int64)
        means = np.asarray(payload.get('means', []), dtype=np.float64)
        covariances = np.asarray(
            payload.get('covariances', []), dtype=np.float64)
        ages = np.asarray(payload.get('aoi', []), dtype=np.int64)
        frames = np.asarray(payload.get('frames', []), dtype=np.int64)
        count = targets.size
        state_dim = int(self._owner_posterior_state_dim)
        if (
            means.shape != (count, state_dim)
            or covariances.shape != (count, state_dim, state_dim)
            or ages.shape != (count,)
            or frames.shape != (count,)
            or np.any(targets < 0)
            or np.any(targets >= self.Q)
            or np.any(~np.isfinite(means))
            or np.any(~np.isfinite(covariances))
            or np.any(ages < 0)
            or np.any(receiver_index < 0)
            or np.any(receiver_index >= self.K)
            or np.unique(receiver_index).size != receiver_index.size
        ):
            raise ValueError('invalid owner-posterior broadcast')
        if count == 0 or receiver_index.size == 0:
            return
        receiver_grid = receiver_index[:, None]
        target_grid = targets[None, :]
        self._owner_posterior_received_mean[
            receiver_grid, int(sender), target_grid] = means[None, :, :]
        self._owner_posterior_received_cov[
            receiver_grid, int(sender), target_grid] = (
                covariances[None, :, :, :])
        self._owner_posterior_received_aoi[
            receiver_grid, int(sender), target_grid] = ages[None, :]
        self._owner_posterior_received_frame[
            receiver_grid, int(sender), target_grid] = frames[None, :]
        self._owner_posterior_fusion_queue.update(
            (int(receiver), int(sender), int(target))
            for receiver in receiver_index
            for target in targets
        )

    def _fuse_delivered_owner_posteriors(self) -> None:
        """Fuse each newly delivered posterior once using covariance intersection."""
        if not self._owner_posterior_enabled or self.belief_mgr is None:
            self._owner_posterior_metrics = {}
            return
        if self.belief_mgr.state_dim != int(self._owner_posterior_state_dim):
            raise RuntimeError(
                'distributed owner posterior state contract does not match '
                'the local belief model')
        trace_before = []
        trace_after = []
        used_ages = []
        fused_entries = 0
        receivers_used = set()
        local_epoch = int(self.t - 1)
        queued = tuple(self._owner_posterior_fusion_queue)
        self._owner_posterior_fusion_queue.clear()
        grouped: Dict[Tuple[int, int], list] = {}
        for receiver, sender, q in queued:
            grouped.setdefault((int(receiver), int(q)), []).append(
                int(sender))
        pair_records = []
        for (receiver, q), candidate_senders in grouped.items():
            means = [self.belief_mgr.mean[receiver, q].copy()]
            covariances = [self.belief_mgr.cov[receiver, q].copy()]
            ages = [int(self.belief_mgr.aoi[receiver, q])]
            consumed = []
            for sender in sorted(candidate_senders):
                if sender == receiver:
                    continue
                source_frame = int(
                    self._owner_posterior_received_frame[
                        receiver, sender, q])
                if (
                    source_frame < 0
                    or source_frame <= int(
                        self._owner_posterior_last_fused_frame[
                            receiver, sender, q])
                    or int(self.t) - source_frame
                    > self._owner_posterior_max_age
                ):
                    continue
                mean = self._owner_posterior_received_mean[
                    receiver, sender, q].copy()
                covariance = self._owner_posterior_received_cov[
                    receiver, sender, q].copy()
                propagation_steps = max(local_epoch - source_frame, 0)
                for _ in range(propagation_steps):
                    mean = self.belief_mgr.F @ mean
                    covariance = (
                        self.belief_mgr.F @ covariance
                        @ self.belief_mgr.F.T
                        + self.belief_mgr.Q_proc_base
                    )
                means.append(mean)
                covariances.append(covariance)
                packet_age = int(
                    self._owner_posterior_received_aoi[
                        receiver, sender, q]) + propagation_steps
                ages.append(packet_age)
                used_ages.append(packet_age)
                consumed.append((sender, source_frame))
            if len(means) <= 1:
                continue
            if len(means) == 2:
                pair_records.append((
                    receiver,
                    q,
                    means[0],
                    covariances[0],
                    means[1],
                    covariances[1],
                    min(ages),
                    consumed[0],
                ))
                continue
            before = float(np.trace(self.belief_mgr.cov[receiver, q]))
            fused_mean, fused_covariance = (
                generalized_covariance_intersection(
                    np.asarray(means),
                    np.asarray(covariances),
                )
            )
            self.belief_mgr.mean[receiver, q] = fused_mean
            self.belief_mgr.cov[receiver, q] = fused_covariance
            self.belief_mgr.aoi[receiver, q] = min(ages)
            trace_before.append(before)
            trace_after.append(float(np.trace(fused_covariance)))
            for sender, source_frame in consumed:
                self._owner_posterior_last_fused_frame[
                    receiver, sender, q] = source_frame
                fused_entries += 1
            receivers_used.add(receiver)
        if pair_records:
            fused_means, fused_covariances = (
                batched_pair_covariance_intersection(
                    np.asarray([item[2] for item in pair_records]),
                    np.asarray([item[3] for item in pair_records]),
                    np.asarray([item[4] for item in pair_records]),
                    np.asarray([item[5] for item in pair_records]),
                )
            )
            for index, item in enumerate(pair_records):
                receiver, q = int(item[0]), int(item[1])
                before = float(np.trace(item[3]))
                fused_covariance = fused_covariances[index]
                self.belief_mgr.mean[receiver, q] = fused_means[index]
                self.belief_mgr.cov[receiver, q] = fused_covariance
                self.belief_mgr.aoi[receiver, q] = int(item[6])
                trace_before.append(before)
                trace_after.append(float(np.trace(fused_covariance)))
                sender, source_frame = item[7]
                self._owner_posterior_last_fused_frame[
                    receiver, int(sender), q] = int(source_frame)
                fused_entries += 1
                receivers_used.add(receiver)
        self._owner_posterior_metrics = {
            'owner_posterior_enabled': 1.0,
            'owner_posterior_state_dim': float(
                self._owner_posterior_state_dim),
            'owner_posterior_fused_entries': float(fused_entries),
            'owner_posterior_receivers': float(len(receivers_used)),
            'owner_posterior_cov_trace_before': float(np.mean(
                trace_before or [0.0])),
            'owner_posterior_cov_trace_after': float(np.mean(
                trace_after or [0.0])),
            'owner_posterior_cov_trace_ratio': float(
                np.mean(trace_after) / max(np.mean(trace_before), 1.0e-12)
                if trace_before else 1.0),
            'owner_posterior_packet_aoi_mean_frames': float(np.mean(
                used_ages or [0.0])),
            'owner_posterior_packet_aoi_max_frames': float(max(
                used_ages or [0])),
        }

    def _prepare_qpd_submission(self) -> None:
        """Build the local QPD control plane before physical transmission.

        Five coordinates of each transmitted target token are the actual
        protocol header ``[queue, bid, capability, primal, age]``.  They pass
        through the same quantizer, sparse mask, channel and delay model as the
        learned latent coordinates; no free side channel is introduced.
        """
        if self._qpd_last_submission_frame == self.t:
            return
        self._qpd_last_submission_frame = self.t
        if len(self.uavs) != self.K or len(self.targets) != self.Q:
            return

        local_pd = np.zeros((self.K, self.Q), dtype=np.float64)
        for k in range(self.K):
            value = np.asarray(
                self.prev_P_D_local.get(k, np.zeros(self.Q)),
                dtype=np.float64,
            ).reshape(-1)
            if value.shape == (self.Q,):
                local_pd[k] = np.clip(value, 0.0, 1.0)

        self._qpd_previous_queue = self._qpd_queue.copy()
        self._qpd_previous_bid = self._qpd_bid.copy()
        for k in range(self.K):
            self._qpd_queue[k] = update_virtual_queue(
                self._qpd_queue[k],
                local_pd[k],
                self._qpd_qos_floor,
                self._qpd_queue_step,
                self._qpd_queue_max,
            )
        safe = local_pd >= self._qpd_qos_floor
        self._qpd_age = np.where(safe, 0, self._qpd_age + 1)

        uav_xy = np.asarray([u.pos[:2] for u in self.uavs])
        distances = np.zeros((self.K, self.Q), dtype=np.float64)
        for k in range(self.K):
            local_target_position, _ = (
                self._coordination_target_state_for_viewer(k))
            distances[k] = np.linalg.norm(
                uav_xy[k, None, :] - local_target_position[:, :2],
                axis=-1,
            )
        self._qpd_capability = np.exp(
            -distances / self._qpd_distance_scale_m)

        queue_norm = self._qpd_queue / self._qpd_queue_max
        marginal_proxy = (
            self._qpd_capability * np.maximum(1.0 - local_pd, 0.0))
        switch_amount = np.abs(
            self._qpd_primal - self._qpd_previous_primal)
        self._qpd_bid = (
            queue_norm * marginal_proxy
            - self._qpd_power_cost * (1.0 - self._qpd_capability)
            - self._qpd_comm_cost
            - self._qpd_switch_cost * switch_amount
        )

        trigger = (
            np.abs(
                self._qpd_queue - self._qpd_previous_queue
            ) / self._qpd_queue_max
            + self._qpd_bid_change_weight * np.abs(
                self._qpd_bid - self._qpd_previous_bid)
        )
        self._qpd_send_mask = trigger > self._qpd_send_threshold
        # A node with an unresolved deficit must expose at least its most
        # valuable target; otherwise an initially quiet graph cannot bootstrap.
        priority = queue_norm * np.maximum(self._qpd_capability, 1e-6)
        for k in range(self.K):
            if (not np.any(self._qpd_send_mask[k])
                    and np.max(self._qpd_queue[k], initial=0.0) > 0.0):
                self._qpd_send_mask[k, int(np.argmax(priority[k]))] = True

        if self._qpd_override_token_mask:
            self._pending_comm_token_masks = {
                k: self._qpd_send_mask[k].astype(np.float64).copy()
                for k in range(self.K)
            }

        self._pending_qpd_protocol = {}
        for k in range(self.K):
            protocol = np.zeros((self.Q, 5), dtype=np.float64)
            protocol[:, 0] = np.clip(
                2.0 * queue_norm[k] - 1.0, -1.0, 1.0)
            protocol[:, 1] = np.tanh(self._qpd_bid[k])
            protocol[:, 2] = np.clip(
                2.0 * self._qpd_capability[k] - 1.0, -1.0, 1.0)
            protocol[:, 3] = np.clip(
                2.0 * self._qpd_primal[k] - 1.0, -1.0, 1.0)
            age_norm = np.clip(
                self._qpd_age[k]
                / max(self._comm_message_ttl_frames, 1),
                0.0,
                1.0,
            )
            protocol[:, 4] = 2.0 * age_norm - 1.0
            self._pending_qpd_protocol[k] = protocol
            if (self._qpd_overwrite_protocol_header
                    and k in self._pending_comm_messages):
                payload = np.asarray(
                    self._pending_comm_messages[k],
                    dtype=np.float64,
                ).reshape(self.Q, self._comm_target_token_dim).copy()
                payload[:, :5] = protocol
                self._pending_comm_messages[k] = payload.reshape(-1)

    def _decode_qpd_protocol(
        self,
        message: np.ndarray,
        token_mask: Optional[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Decode an actually received, quantized QPD protocol header."""
        payload = np.asarray(message, dtype=np.float64).reshape(
            self.Q, self._comm_target_token_dim)
        active = (
            np.ones(self.Q, dtype=bool)
            if token_mask is None
            else np.asarray(token_mask).reshape(-1) > 0.5
        )
        if active.shape != (self.Q,):
            raise ValueError('received QPD token mask has invalid shape')
        return {
            'queue': np.where(
                active, 0.5 * (payload[:, 0] + 1.0), 0.0),
            'bid': np.where(active, payload[:, 1], 0.0),
            'capability': np.where(
                active, 0.5 * (payload[:, 2] + 1.0), 0.0),
            'primal': np.where(
                active, 0.5 * (payload[:, 3] + 1.0), 0.0),
            'age': np.where(
                active, 0.5 * (payload[:, 4] + 1.0), 0.0),
            'active': active,
        }

    def _decode_qpd_control_packet(
        self,
        protocol: np.ndarray,
        token_mask: Optional[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Decode a five-dimensional QPD stream appended to each token."""
        packet = np.asarray(protocol, dtype=np.float64).reshape(self.Q, 5)
        active = (
            np.ones(self.Q, dtype=bool)
            if token_mask is None
            else np.asarray(token_mask).reshape(-1) > 0.5
        )
        return {
            'queue': np.where(
                active, 0.5 * (packet[:, 0] + 1.0), 0.0),
            'bid': np.where(active, packet[:, 1], 0.0),
            'capability': np.where(
                active, 0.5 * (packet[:, 2] + 1.0), 0.0),
            'primal': np.where(
                active, 0.5 * (packet[:, 3] + 1.0), 0.0),
            'age': np.where(
                active, 0.5 * (packet[:, 4] + 1.0), 0.0),
            'active': active,
        }

    def _merge_received_qpd_protocol(
        self,
        receiver: int,
        sender: int,
        message: np.ndarray,
        token_mask: Optional[np.ndarray],
    ) -> None:
        """Merge an event packet without erasing silent target state."""
        decoded = self._decode_qpd_protocol(message, token_mask)
        self._merge_received_qpd_packet(
            receiver, sender, decoded, decoded['active'])

    def _merge_received_qpd_packet(
        self,
        receiver: int,
        sender: int,
        decoded: Dict[str, np.ndarray],
        active: np.ndarray,
    ) -> None:
        """Merge a decoded in-band or appended protocol control stream."""
        active = np.asarray(
            decoded.get('active', active), dtype=bool)
        self._qpd_received_primal[receiver, sender, active] = (
            decoded['primal'][active])
        self._qpd_received_queue[receiver, sender, active] = (
            decoded['queue'][active])
        self._qpd_received_last_seen[receiver, sender, active] = self.t

    def _compute_analytical_min_comm_power(
        self, uav_positions: np.ndarray,
    ) -> np.ndarray:
        """Analytic minimum broadcast power per sender (L0 comm-slack recovery).

        Direct inversion of the orthogonal-U2U Shannon link model: for sender i,
        the minimum power to meet the SNR/deadline criterion for EVERY receiver
        is max_j (Gamma_req * N0 * B_eff / g_ij).  Transport semantics are
        unchanged; only the link margin is reclaimed into the sensing budget.

        D1.1-C (2026-08-16): the per-sender bandwidth B_i is no longer the
        equal split B/n_active but the optimal allocation of the convex problem
            min sum_i f_i(B_i)   s.t.  sum_i B_i = B,  B_i >= 0,
            f_i(B) = N0 * B * max(gamma_th, 2^(r_i/B) - 1) / g_i,
        with r_i = payload_i / t_win and g_i = worst-receiver path gain of
        sender i.  f_i is convex in B (the Shannon power is convex-decreasing
        in bandwidth and max preserves convexity), so the KKT condition
        f_i'(B_i) = -lambda with sum_i B_i = B is necessary and sufficient;
        the outer bisection on lambda and the inner per-sender bisection on
        B_i solve it exactly.  Every sender still meets its SNR threshold and
        rate demand (P_i = f_i(B_i) with B_i > 0), so the allocation is
        feasible under the same orthogonal-communication semantics and the
        total power is strictly no larger than under the equal split.
        """
        comm = self._inter_uav_comm
        K = self.K
        result = np.zeros(K, dtype=np.float64)
        active = np.zeros(K, dtype=bool)
        payload = np.zeros(K, dtype=np.float64)
        for k in range(K):
            active_dimensions = comm._active_dimensions(
                self._pending_comm_token_masks.get(k))
            structure_bits = (
                self.Q * self._structure_student_endpoint_width
                * self._structure_student_bits_per_dim
                if k in self._pending_structure_student_protocol else 0
            )
            if k in self._pending_comm_messages:
                bits = comm.payload_bits(
                    self._pending_comm_rates.get(k, 0),
                    active_dimensions=active_dimensions)
                if bits > 0 or structure_bits > 0:
                    active[k] = True
                    payload[k] = float(bits + structure_bits)
        n_active = max(1, int(np.sum(active)))
        b_total = comm.bandwidth_hz
        gamma_th = float(10.0 ** (comm.snr_threshold_db / 10.0))
        design_snr_factor = float(10.0 ** (
            max(0.0, float(getattr(
                self.cfg.marl, 'analytical_comm_snr_margin_db', 0.0)))
            / 10.0))
        t_win = max(comm.deadline_s - comm.processing_delay_s, 1e-12)

        # Per-sender rate demand and worst-receiver path gain (g_i = min_j g_ij).
        rates = np.zeros(K, dtype=np.float64)
        gains = np.zeros(K, dtype=np.float64)
        for i in range(K):
            if not active[i] or payload[i] <= 0.0:
                continue
            rates[i] = payload[i] / t_win
            g_min = np.inf
            for j in range(K):
                if j == i:
                    continue
                d = max(float(np.linalg.norm(
                    uav_positions[i] - uav_positions[j])), 1.0)
                path_gain = (comm.wavelength / (4.0 * np.pi * d)) ** 2
                g_min = min(g_min, comm.antenna_gain_linear * path_gain)
            gains[i] = max(g_min, 1e-30)

        if not bool(getattr(self.cfg.marl,
                            'analytical_comm_optimal_bw', True)):
            # D0.93 L0 equal split (legacy path, kept for A/B comparison).
            b_eff = b_total / n_active
            n0_b = comm.kT * b_eff * comm.noise_figure_linear
            for i in range(K):
                if not active[i] or payload[i] <= 0.0:
                    continue
                if comm.finite_blocklength_enabled:
                    channel_uses = max(1, int(np.floor(b_eff * t_win)))
                    required = minimum_snr_normal_approximation(
                        channel_uses, int(np.ceil(payload[i])),
                        comm.finite_blocklength_target_bler)
                    gamma_rate = (
                        float('inf') if required is None else float(required))
                else:
                    r_req = rates[i]
                    gamma_rate = float(2.0 ** (r_req / b_eff) - 1.0)
                gamma_req = max(gamma_th, gamma_rate) * design_snr_factor
                result[i] = gamma_req * n0_b / gains[i]
            return result

        # D1.1-C optimal bandwidth allocation via KKT bisection.
        n0 = comm.kT * comm.noise_figure_linear
        active_idx = [i for i in range(K) if active[i] and payload[i] > 0.0]
        if not active_idx:
            return result
        m = len(active_idx)
        r_arr = rates[active_idx]
        g_arr = gains[active_idx]

        # Marginal f_i'(B) of f_i(B) = n0*B*max(gamma_th, 2^(r/B)-1)/g_i.
        # The rate term's derivative is 2^(r/B)*(1 - r*ln2/B) - 1; the SNR
        # term is linear with constant marginal gamma_th.  max preserves
        # convexity and the marginal is monotone increasing in B, so the KKT
        # bisection below is exact.
        def marginal(i: int, b: float) -> float:
            b = max(b, 1e-12)
            x = float(2.0 ** min(r_arr[i] / b, 60.0))
            rate_marg = x * (1.0 - r_arr[i] * np.log(2.0) / b) - 1.0
            snr_marg = gamma_th
            return n0 / g_arr[i] * max(rate_marg, snr_marg)

        def b_of_lambda(lam: float) -> np.ndarray:
            out = np.zeros(m, dtype=np.float64)
            for i in range(m):
                lo, hi = 1e-9, b_total
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if marginal(i, mid) > -lam:
                        hi = mid
                    else:
                        lo = mid
                out[i] = 0.5 * (lo + hi)
            return out

        lam_lo, lam_hi = 1e-12, 1e6
        for _ in range(60):
            mid = 0.5 * (lam_lo + lam_hi)
            if float(np.sum(b_of_lambda(mid))) > b_total:
                lam_lo = mid
            else:
                lam_hi = mid
        lam = 0.5 * (lam_lo + lam_hi)
        b_alloc = b_of_lambda(lam)
        s = float(np.sum(b_alloc))
        if s > 0.0 and abs(s - b_total) > 1e-6 * b_total:
            b_alloc = b_alloc * b_total / s  # exact-sum fallback
        for idx, i in enumerate(active_idx):
            b_i = max(float(b_alloc[idx]), 1e-12)
            gamma_rate = float(2.0 ** min(r_arr[idx] / b_i, 60.0) - 1.0)
            gamma_req = max(gamma_th, gamma_rate) * design_snr_factor
            result[i] = gamma_req * n0 * b_i / g_arr[idx]
        return result

    def _solve_intercept_power(
        self, gain: np.ndarray, budget: np.ndarray,
    ) -> object | None:
        """D1.1-D live covertness-constrained L1 power (T3 / advice 012).

        Solves the max-min power LP subject to the per-target counter-
        detection hard bound

            D_q^I = sum_i a^I[i,q] p_iq <= bar D^I,
            bar D^I = [Q^{-1}(P_FA^I) - Q^{-1}(eps)]^2,

        so the opponent's detection probability P_{D,w}^I <= eps is a HARD
        executed constraint (not a reward term).  Returns a MaxMinPowerResult
        (with the opponent-detection price mu attached to prices) or None when
        infeasible; the caller then falls back to the normal power path and
        the violation is recorded in ``_last_intercept_pd_max``.
        """
        from uav_isac.coordination.intercept_power import (
            INTERCEPT_CAPABILITIES,
            constrained_maxmin_lp,
            intercept_coefficients,
            intercept_deflection_limit,
        )
        from uav_isac.coordination.maxmin_power import MaxMinPowerResult
        from uav_isac.physical.detection import compute_detection_probabilities

        uav_p = np.array([u.pos[:2].copy() for u in self.uavs])
        tgt_p = np.array([t.get_position_3d()[:2] for t in self.targets])
        tier = str(getattr(self.cfg.marl, 'intercept_capability',
                           'medium')).strip().lower()
        theta = INTERCEPT_CAPABILITIES.get(
            tier, INTERCEPT_CAPABILITIES['medium'])
        a_i = intercept_coefficients(
            uav_p, tgt_p, fc=float(self.cfg.otfs.fc),
            g_tx_dBi=float(self.cfg.otfs.g_tx_dBi),
            height=float(self.cfg.scenario.height),
            kt=float(self.cfg.channel.kT), theta=theta)
        pfa_i = float(getattr(self.cfg.marl, 'intercept_pfa', 1e-3))
        eps = float(getattr(self.cfg.marl, 'intercept_eps', 0.1))
        d_bar = intercept_deflection_limit(pfa_i, eps)
        self._last_intercept_eps = eps

        # D1.1-E: when the task-constrained (lexicographic) power path is the
        # deployed inner layer, join the QoS floors and the covertness bound in
        # ONE LP (qos_constrained_maxmin_lp with intercept rows), so covertness
        # is enforced without losing the QoS-constrained max-min objective.
        joined = False
        out = None
        if (getattr(self.cfg.marl, 'task_constrained_power_enabled', False)
                and getattr(self.cfg.marl, 'task_constrained_mode', 'gauge')
                == 'lexicographic'):
            try:
                from uav_isac.coordination.capability import (
                    qos_constrained_maxmin_lp,
                )
                from uav_isac.coordination.pwl_pd import (
                    chord_lower_bound,
                    curvature_breakpoints,
                )
                from uav_isac.physical.detection import (
                    minimum_deflection_for_detection_probability,
                )
                qos_floors = tuple(float(v) for v in getattr(
                    self.cfg.marl, 'task_constrained_qos_floors',
                    (0.60, 0.70, 0.80, 3)))
                d_min = float(minimum_deflection_for_detection_probability(
                    np.asarray([qos_floors[0]]), self.cfg.detection.P_FA)[0])
                ceiling = np.sum(gain * budget[:, None], axis=0)
                if not np.any(ceiling < d_min - 1e-9):
                    d_max = float(np.max(ceiling)) + 1.0
                    bps = curvature_breakpoints(
                        self.cfg.detection.P_FA, d_min, d_max, epsilon=1e-3)
                    cs, ci = chord_lower_bound(self.cfg.detection.P_FA, bps)
                    out = qos_constrained_maxmin_lp(
                        gain, budget, self.cfg.detection.P_FA,
                        (qos_floors[0], qos_floors[1], qos_floors[2],
                         max(1, int(qos_floors[3]))),
                        cs, ci, d_min,
                        intercept_coeff=a_i,
                        intercept_ub=np.full(self.Q, d_bar))
                    joined = out is not None  # feasible joint LP only
            except (ValueError, ImportError):
                out = None
                joined = False
        if out is None:
            out = constrained_maxmin_lp(
                gain, budget, a_i, np.full(self.Q, d_bar))
        if out is None:
            self._last_intercept_infeasible = True
            self._last_intercept_mu = None
            self._last_intercept_pd_max = None
            return None
        self._last_intercept_infeasible = False
        self._last_intercept_joined = joined
        if joined:
            # qos_constrained_maxmin_lp returns (t*, p*, D*); recover the
            # opponent-detection price from the joint LP's marginals indirectly
            # is not exposed, so expose the pure covertness prices instead.
            t_star, p_star, d_star = out
            p_out = p_star
            deflection = d_star
            _t_out = float(t_star)
            lam_out = np.zeros(self.Q, dtype=np.float64)
            mu_out = np.zeros(self.Q, dtype=np.float64)
        else:
            t_star, p_star, lam, beta, mu = out
            p_out = p_star
            deflection = np.sum(gain * p_star, axis=0)
            _t_out = float(t_star)
            lam_out = lam
            mu_out = mu
        d_intercept = np.sum(a_i * p_out, axis=0)
        self._last_intercept_mu = mu_out.copy()
        self._last_intercept_pd_max = float(np.max(
            compute_detection_probabilities(d_intercept, pfa_i)))
        return MaxMinPowerResult(
            power_w=p_out,
            deflection=deflection,
            worst_deflection=float(np.min(deflection)),
            prices=lam_out,
            dual_upper_bound=_t_out,
            primal_dual_gap=0.0,
            rounds=0,
            worst_history=(float(np.min(deflection)),),
        )

    def _process_learned_communications(
        self, uav_positions: np.ndarray,
    ) -> CommunicationStepStats:
        """Deliver due messages and transmit the current learned broadcasts."""
        if self._comm_mode != 'cost_aware' or self._inter_uav_comm is None:
            self._last_comm_stats = CommunicationStepStats()
            return self._last_comm_stats

        tx_powers_w = None
        if self._joint_isac_power_enabled:
            tx_powers_w = {}
            self._current_comm_power_w = np.zeros(self.K, dtype=np.float64)
            self._current_sensing_power_w = np.zeros(
                (self.K, self.Q), dtype=np.float64)
            analytical_power = (
                self._compute_analytical_min_comm_power(uav_positions)
                if self._analytical_comm_power_enabled else None
            )
            for k in range(self.K):
                active_dimensions = self._inter_uav_comm._active_dimensions(
                    self._pending_comm_token_masks.get(k))
                structure_bits = (
                    self.Q
                    * self._structure_student_endpoint_width
                    * self._structure_student_bits_per_dim
                    if k in self._pending_structure_student_protocol
                    else 0
                )
                hyperedge_protocol_bits = (
                    self._hyperedge_protocol_dim
                    * self._hyperedge_state_bits_per_dim
                    if (
                        self._hyperedge_enabled
                        and self._hyperedge_protocol_only_enabled
                        and k in self._pending_hyperedge_protocol
                    ) else 0
                )
                active = (
                    k in self._pending_comm_messages
                    and (
                        self._inter_uav_comm.payload_bits(
                            self._pending_comm_rates.get(k, 0),
                            active_dimensions=active_dimensions) > 0
                        or structure_bits > 0
                        or hyperedge_protocol_bits > 0
                    )
                )
                # Evidence/posterior senders are selected only after sensing.
                # A silent early coordination cohort must therefore retain
                # its frame-level communication-power reservation; otherwise
                # a later evidence packet would be transmitted at zero power.
                reserve_for_late_evidence = bool(
                    self._detection_fusion_mode == 'u2u_distributed')
                radio_reserved = bool(active or reserve_for_late_evidence)
                fraction = (
                    self._pending_comm_power_fractions.get(k, 0.0)
                    if radio_reserved else 0.0)
                if analytical_power is not None and active:
                    # L0: use the analytic minimum (<= learned fraction).
                    p_comm = min(
                        self._isac_total_power_w * fraction,
                        analytical_power[k],
                    )
                else:
                    p_comm = self._isac_total_power_w * fraction
                p_sense = min(
                    self._isac_total_power_w - p_comm,
                    self._sensing_power_cap_w,
                )
                weights = self._pending_sensing_weights.get(
                    k, np.full(self.Q, 1.0 / max(self.Q, 1)))
                self._current_comm_power_w[k] = p_comm
                self._current_sensing_power_w[k] = p_sense * weights
                tx_powers_w[k] = float(p_comm)

        # Retain the latest delivered message for a bounded number of frames.
        # This provides a physical message age (AoI) without inventing free data.
        inbox: Dict[int, Dict[int, np.ndarray]] = {k: {} for k in range(self.K)}
        inbox_meta: Dict[int, Dict[int, dict]] = {k: {} for k in range(self.K)}
        if self._qpd_enabled:
            expired_qpd = (
                self.t - self._qpd_received_last_seen
                > self._comm_message_ttl_frames
            )
            self._qpd_received_primal[expired_qpd] = 0.0
            self._qpd_received_queue[expired_qpd] = 0.0
            self._qpd_received_last_seen[expired_qpd] = FRAME_NOT_APPLICABLE
        if self._hyperedge_enabled:
            expired_hyperedge = (
                self.t - self._hyperedge_received_last_seen
                > self._comm_message_ttl_frames
            )
            self._hyperedge_received_offer[expired_hyperedge] = 0.0
            self._hyperedge_received_last_seen[
                expired_hyperedge] = FRAME_NOT_APPLICABLE
        if self._structure_student_channel_enabled:
            future_structure_mail = []
            for (due_frame, sender, protocol,
                 sent_frame) in self._structure_student_mailbox:
                if int(due_frame) <= self.t:
                    self._merge_structure_student_public_protocol(
                        int(sender), protocol, int(sent_frame))
                else:
                    future_structure_mail.append((
                        due_frame, sender, protocol, sent_frame))
            self._structure_student_mailbox = future_structure_mail
            expired_structure = (
                self.t - self._structure_student_public_last_seen
                > self._comm_message_ttl_frames
            )
            self._structure_student_public_protocol[
                expired_structure] = 0.0
            self._structure_student_public_valid[
                expired_structure] = False
            self._structure_student_public_last_seen[
                expired_structure] = FRAME_NOT_APPLICABLE
        for receiver in range(self.K):
            old_msgs = self._received_comm_msgs.get(receiver, {})
            old_meta = self._received_comm_meta.get(receiver, {})
            for sender, message in old_msgs.items():
                md = dict(old_meta.get(sender, {}))
                age = int(md.get('age_frames', 0)) + 1
                if age <= self._comm_message_ttl_frames:
                    md['age_frames'] = age
                    inbox[receiver][sender] = np.asarray(message).copy()
                    inbox_meta[receiver][sender] = md
        future_mail = []
        for due_frame, receiver, sender, message, metadata in self._comm_mailbox:
            if due_frame <= self.t:
                inbox[int(receiver)][int(sender)] = np.asarray(message).copy()
                md = dict(metadata)
                md['age_frames'] = max(0, int(self.t - md.get('sent_frame', self.t)))
                inbox_meta[int(receiver)][int(sender)] = md
                if self._qpd_enabled:
                    if 'qpd_protocol' in md:
                        decoded = self._decode_qpd_control_packet(
                            md['qpd_protocol'], md.get('token_mask'))
                        self._merge_received_qpd_packet(
                            int(receiver), int(sender),
                            decoded, decoded['active'])
                    elif self._qpd_overwrite_protocol_header:
                        self._merge_received_qpd_protocol(
                            int(receiver),
                            int(sender),
                            message,
                            md.get('token_mask'),
                        )
                if (self._hyperedge_enabled
                        and 'hyperedge_protocol' in md):
                    self._merge_received_hyperedge_packet(
                        int(receiver), int(sender),
                        md['hyperedge_protocol'], md.get('token_mask'),
                        sent_frame=md.get(
                            'hyperedge_protocol_frame',
                            md.get('sent_frame')))
                if (
                    self._composable_certificate_enabled
                    and 'composable_certificate' in md
                ):
                    self._merge_composable_certificate_packet(
                        int(receiver), int(sender),
                        md['composable_certificate'],
                        float(md['composable_certificate_dual_upper']),
                        int(md['composable_certificate_frame']),
                        md.get('composable_certificate_targetwise_upper'),
                    )
                if (
                    self._owner_posterior_enabled
                    and 'owner_posterior' in md
                ):
                    self._merge_owner_posterior_packet(
                        int(receiver), int(sender),
                        md['owner_posterior'],
                    )
            else:
                future_mail.append(
                    (due_frame, receiver, sender, message, metadata))
        self._comm_mailbox = future_mail

        extra_dimensions = {}
        exact_extra_payload_bits: Dict[int, int] = {}
        protocol_header_bits: Dict[int, int] = {}
        base_payload_dimensions: Dict[int, int] = {}
        suppress_message_payload: Dict[int, bool] = {}
        quantized_qpd_protocol: Dict[int, np.ndarray] = {}
        if self._qpd_enabled and not self._qpd_overwrite_protocol_header:
            for sender, protocol in self._pending_qpd_protocol.items():
                mask = np.asarray(self._pending_comm_token_masks.get(
                    sender, np.ones(self.Q)), dtype=np.float64)
                active_targets = int(np.sum(mask > 0.5))
                extra_dimensions[sender] = 5 * active_targets
                rate_index = int(self._pending_comm_rates.get(sender, 0))
                quantized = self._inter_uav_comm.quantize_values(
                    np.asarray(protocol).reshape(-1), rate_index,
                ).reshape(self.Q, 5)
                quantized[mask <= 0.5] = 0.0
                quantized_qpd_protocol[sender] = quantized
        quantized_hyperedge_protocol: Dict[int, np.ndarray] = {}
        if self._hyperedge_enabled:
            for sender, protocol in self._pending_hyperedge_protocol.items():
                mask = np.asarray(self._pending_comm_token_masks.get(
                    sender, np.ones(self.Q)), dtype=np.float64)
                active_targets = int(np.sum(mask > 0.5))
                hyperedge_dimensions = (
                    self._hyperedge_protocol_dim
                    if (
                        self._hyperedge_pair_score_mode
                        == 'budget_reconstructable'
                        and active_targets > 0
                    )
                    else self._hyperedge_protocol_dim * active_targets
                )
                rate_index = int(self._pending_comm_rates.get(sender, 0))
                if self._hyperedge_protocol_only_enabled:
                    base_payload_dimensions[sender] = 0
                    suppress_message_payload[sender] = True
                    exact_extra_payload_bits[sender] = (
                        int(exact_extra_payload_bits.get(sender, 0))
                        + sum(self._hyperedge_state_field_bits)
                    )
                    if self._hyperedge_protocol_header_bits > 0:
                        protocol_header_bits[sender] = (
                            self._hyperedge_protocol_header_bits)
                    protocol_array = np.asarray(protocol).reshape(
                        self.Q, self._hyperedge_protocol_dim)
                    quantized = np.empty_like(
                        protocol_array, dtype=np.float64)
                    for field, bits in enumerate(
                            self._hyperedge_state_field_bits):
                        quantized[:, field] = (
                            self._inter_uav_comm.quantize_values_at_bits(
                                protocol_array[:, field], bits))
                    if (
                        self._hyperedge_state_codec_service_envelope_enabled
                        and self._hyperedge_nearfield_residual_enabled
                        and self._hyperedge_protocol_dim >= 7
                        and self._hyperedge_state_field_bits[4]
                        < self._hyperedge_state_bits_per_dim
                    ):
                        # Field 4 is categorical (target 0..Q, including the
                        # no-near-field sentinel Q). Decode its short code to
                        # the integer and look up the legacy uniform-codec
                        # representative. Thus downstream floating-point state
                        # is bit-identical to the reference codec, not merely
                        # category-equivalent.
                        target_code = np.rint(
                            0.5 * (quantized[:, 4] + 1.0)
                            * float(self.Q)
                        ).astype(np.int64)
                        target_code = np.clip(target_code, 0, self.Q)
                        legacy_source = (
                            2.0 * target_code.astype(np.float64)
                            / max(float(self.Q), 1.0) - 1.0)
                        quantized[:, 4] = (
                            self._inter_uav_comm.quantize_values_at_bits(
                                legacy_source,
                                self._hyperedge_state_bits_per_dim,
                            ))
                else:
                    extra_dimensions[sender] = (
                        int(extra_dimensions.get(sender, 0))
                        + hyperedge_dimensions)
                    quantized = self._inter_uav_comm.quantize_values(
                        np.asarray(protocol).reshape(-1), rate_index,
                    ).reshape(self.Q, self._hyperedge_protocol_dim)
                if (self._hyperedge_pair_score_mode
                        == 'budget_reconstructable'):
                    active_indices = np.flatnonzero(mask > 0.5)
                    if active_indices.size:
                        state = quantized[int(active_indices[0])].copy()
                        quantized[:] = 0.0
                        quantized[int(active_indices[0])] = state
                    else:
                        quantized[:] = 0.0
                else:
                    quantized[mask <= 0.5] = 0.0
                quantized_hyperedge_protocol[sender] = quantized
                # Consensus must use public protocol state.  If the sender
                # plans from an unquantized private copy while every neighbor
                # plans from the transmitted 4-bit copy, tiny score changes
                # can invert the discrete Tx/Rx partition and eliminate every
                # mutual edge.  The sender therefore commits to the same
                # quantized values placed on air.
                decoded, active = self._decode_hyperedge_packet(
                    quantized, mask)
                self._hyperedge_local_offer[
                    int(sender), active] = decoded[active]
        quantized_composable_certificate: Dict[int, np.ndarray] = {}
        quantized_composable_targetwise_upper: Dict[int, np.ndarray] = {}
        if self._composable_certificate_enabled:
            for sender in range(self.K):
                source_frame = int(
                    self._composable_certificate_local_frame[sender])
                if (
                    source_frame < 0
                    or sender not in self._pending_comm_messages
                ):
                    continue
                quantized_composable_certificate[sender] = quantize_lower_log(
                    self._composable_certificate_local_lower[sender],
                    bits=self._composable_certificate_bits,
                    scale=self._composable_certificate_scale,
                    maximum=self._composable_certificate_maximum,
                )
                quantized_dual_upper = float(quantize_upper_log(
                    np.asarray([
                        self._composable_certificate_local_dual_upper[sender]
                    ]),
                    bits=self._composable_certificate_bits,
                    scale=self._composable_certificate_scale,
                    maximum=self._composable_certificate_maximum,
                )[0])
                self._composable_certificate_last_sent_lower[sender] = (
                    quantized_composable_certificate[sender])
                self._composable_certificate_last_sent_frame[sender] = (
                    source_frame)
                self._composable_certificate_last_sent_dual_upper[sender] = (
                    quantized_dual_upper)
                if self._composable_certificate_targetwise_upper_enabled:
                    quantized_composable_targetwise_upper[sender] = (
                        quantize_upper_log(
                            self._composable_certificate_local_targetwise_upper[
                                sender],
                            bits=self._composable_certificate_bits,
                            scale=self._composable_certificate_scale,
                            maximum=self._composable_certificate_maximum,
                        )
                    )
                    self._composable_certificate_last_sent_targetwise_upper[
                        sender] = quantized_composable_targetwise_upper[sender]
                exact_extra_payload_bits[sender] = (
                    int(exact_extra_payload_bits.get(sender, 0))
                    + (self.Q + (
                        self.Q
                        if self._composable_certificate_targetwise_upper_enabled
                        else 1
                    )) * self._composable_certificate_bits
                    + self._composable_certificate_frame_bits
                )
        quantized_owner_posterior: Dict[int, dict] = {}
        owner_posterior_payload_bits = 0
        if self._owner_posterior_enabled:
            target_id_bits = max(1, int(np.ceil(np.log2(max(self.Q, 2)))))
            state_dim = int(self._owner_posterior_state_dim)
            entry_bits = (
                state_dim * self._owner_posterior_mean_bits
                + state_dim * self._owner_posterior_cov_bits
                + self._owner_posterior_aoi_bits
                + target_id_bits
            )
            maximum_encoded_age = (
                (1 << self._owner_posterior_aoi_bits) - 1)
            velocity_bound_mps = max(
                float(self.cfg.uav.v_max),
                float(self.cfg.target.speed_range[1])
                + 4.0 * float(self.cfg.target.sigma_a),
                1.0,
            )
            for sender in range(self.K):
                active_targets = np.flatnonzero(
                    self._owner_posterior_pending_valid[sender])
                if (
                    active_targets.size == 0
                    or sender not in self._pending_comm_messages
                ):
                    continue
                decoded_means = []
                decoded_covariances = []
                encoded_ages = []
                source_frames = []
                for target in active_targets:
                    decoded_mean, decoded_covariance = (
                        quantize_belief_feedback(
                            self._owner_posterior_pending_mean[
                                sender, target],
                            self._owner_posterior_pending_cov[
                                sender, target],
                            area_size_xy=(
                                float(self.area_size[0]),
                                float(self.area_size[1]),
                            ),
                            velocity_bound_mps=velocity_bound_mps,
                            acceleration_bound_mps2=max(
                                4.0 * float(self.cfg.target.sigma_a),
                                1.0,
                            ),
                            mean_bits=self._owner_posterior_mean_bits,
                            covariance_bits=self._owner_posterior_cov_bits,
                        )
                    )
                    decoded_means.append(decoded_mean)
                    decoded_covariances.append(decoded_covariance)
                    encoded_ages.append(min(
                        max(int(self._owner_posterior_pending_aoi[
                            sender, target]), 0),
                        maximum_encoded_age,
                    ))
                    source_frames.append(int(
                        self._owner_posterior_pending_frame[
                            sender, target]))
                packet = {
                    'targets': active_targets.astype(np.int64),
                    'means': np.asarray(decoded_means, dtype=np.float64),
                    'covariances': np.asarray(
                        decoded_covariances, dtype=np.float64),
                    'aoi': np.asarray(encoded_ages, dtype=np.int64),
                    'frames': np.asarray(source_frames, dtype=np.int64),
                }
                # One physical broadcast has one immutable payload shared by
                # all receiver-delivery records.  Per-receiver array copies
                # would emulate K-1 different encodings and add O(KQ) memory
                # traffic without changing any received bit.
                for value in packet.values():
                    np.asarray(value).setflags(write=False)
                quantized_owner_posterior[sender] = packet
                sender_bits = int(entry_bits * active_targets.size)
                owner_posterior_payload_bits += sender_bits
                exact_extra_payload_bits[sender] = (
                    int(exact_extra_payload_bits.get(sender, 0))
                    + sender_bits
                )
        quantized_structure_protocol: Dict[int, np.ndarray] = {}
        selected_structure_bits: Dict[int, int] = {}
        structure_endpoint_dimensions = (
            self.Q * self._structure_student_endpoint_width)
        if self._structure_student_channel_enabled:
            active_sender_count = 0
            for candidate_sender in range(self.K):
                if candidate_sender not in self._pending_comm_messages:
                    continue
                active_dimensions = (
                    self._inter_uav_comm._active_dimensions(
                        self._pending_comm_token_masks.get(candidate_sender))
                    + max(0, int(extra_dimensions.get(
                        candidate_sender, 0)))
                )
                learned_payload = self._inter_uav_comm.payload_bits(
                    self._pending_comm_rates.get(candidate_sender, 0),
                    active_dimensions=active_dimensions,
                )
                if (
                    learned_payload > 0
                    or candidate_sender
                    in self._pending_structure_student_protocol
                ):
                    active_sender_count += 1
            for sender, protocol in (
                    self._pending_structure_student_protocol.items()):
                selected_bits, rate_header_bits = (
                    self._select_structure_student_bits_per_dim(
                        int(sender),
                        uav_positions,
                        tx_powers_w,
                        extra_dimensions,
                        active_sender_count,
                        structure_endpoint_dimensions,
                    )
                )
                selected_structure_bits[int(sender)] = int(selected_bits)
                exact_extra_payload_bits[sender] = (
                    int(exact_extra_payload_bits.get(sender, 0))
                    + structure_endpoint_dimensions * selected_bits
                    + rate_header_bits
                )
                quantized_structure_protocol[sender] = (
                    self._inter_uav_comm.quantize_values_at_bits(
                        np.asarray(protocol).reshape(-1),
                        selected_bits,
                    ).reshape(
                        self.Q,
                        self._structure_student_endpoint_width,
                    )
                )

        # C6 decision-sufficient U2U gate (audit advice/001 §12-13, 2026-08-26):
        # suppress a pending sender's token (silence, zero airtime) whenever its
        # LOCAL top-1 vs top-2 capability margin is CERTIFIED to persist through
        # the Lipschitz stale-drift and physical-error bounds; otherwise the
        # token goes out at the decision-preserving minimal bit precision.
        if (
            self._hyperedge_protocol_only_enabled
            and self._hyperedge_beacon_round_robin_period > 1
        ):
            # Keep all pending messages through frame-power reservation and
            # analytical link-budget computation, because any node may later
            # become an evidence sender. Suppress only the early coordination
            # carrier after that reservation is fixed.
            for sender in tuple(self._pending_comm_messages):
                if sender not in self._pending_hyperedge_protocol:
                    self._pending_comm_messages.pop(sender, None)
                    self._pending_comm_rates.pop(sender, None)
        self._apply_decision_sufficient_silence()
        # C6 adaptive-bits stage: senders that DO transmit with a certified
        # decision-preserving plan carry their exact per-dimension precision
        # (instead of the rate-ladder rounding) on air; consumed once here.
        dec_suff_bits = dict(self._pending_decision_sufficient_bits)
        self._pending_decision_sufficient_bits = {}
        service_envelope_payload_bits: Dict[int, int] = {}
        if (
            self._hyperedge_state_codec_service_envelope_enabled
            and self._hyperedge_state_service_delta_bits > 0
        ):
            for sender in self._pending_hyperedge_protocol:
                if sender not in self._pending_comm_messages:
                    continue
                rate_index = int(self._pending_comm_rates.get(sender, 0))
                rate_slot = int(np.clip(
                    rate_index,
                    0,
                    len(self._inter_uav_comm.rate_bits_per_dim) - 1,
                ))
                bits_per_dimension = int(
                    dec_suff_bits.get(
                        sender,
                        self._inter_uav_comm.rate_bits_per_dim[rate_slot],
                    ))
                learned_dimensions = (
                    int(base_payload_dimensions[sender])
                    if sender in base_payload_dimensions
                    else self._inter_uav_comm._active_dimensions(
                        self._pending_comm_token_masks.get(sender))
                ) + int(extra_dimensions.get(sender, 0))
                appended_bits = int(
                    exact_extra_payload_bits.get(sender, 0))
                actual_payload = (
                    learned_dimensions * bits_per_dimension + appended_bits)
                actual_header = int(protocol_header_bits.get(
                    sender, self._inter_uav_comm.header_bits))
                if actual_payload > 0:
                    service_envelope_payload_bits[sender] = int(
                        actual_header
                        + actual_payload
                        + self._hyperedge_state_service_delta_bits)
                    service_envelope_payload_bits[sender] += max(
                        0,
                        self._hyperedge_state_service_reference_header_bits
                        - actual_header,
                    )
        deliveries, stats = self._inter_uav_comm.transmit(
            self._pending_comm_messages,
            self._pending_comm_rates,
            uav_positions,
            tx_powers_w=tx_powers_w,
            token_masks=self._pending_comm_token_masks,
            extra_payload_dimensions=extra_dimensions,
            extra_payload_bits=exact_extra_payload_bits,
            base_payload_dimensions=base_payload_dimensions,
            suppress_message_payload=suppress_message_payload,
            exact_bits_per_dim=(dec_suff_bits or None),
            header_bits_by_sender=(protocol_header_bits or None),
            service_envelope_payload_bits_by_sender=(
                service_envelope_payload_bits or None),
        )
        # A sender always knows the sparse claim mask it just put on air. Keep
        # this local state through observation construction; peer copies still
        # arrive solely through the receiver-specific physical U2U inbox.
        self._last_sent_comm_token_masks = {}
        self._last_sent_comm_target_claims = {}
        for sender in range(self.K):
            rate_index = self._pending_comm_rates.get(sender, 0)
            has_payload = sender in self._pending_comm_messages
            active_dims = self._inter_uav_comm._active_dimensions(
                self._pending_comm_token_masks.get(sender))
            if sender in base_payload_dimensions:
                active_dims = int(base_payload_dimensions[sender])
            active_dims += int(extra_dimensions.get(sender, 0))
            ladder = self._inter_uav_comm.rate_bits_per_dim
            ladder_index = int(np.clip(
                int(rate_index), 0, len(ladder) - 1))
            bits_per_dim = int(dec_suff_bits.get(
                sender, ladder[ladder_index]))
            active = (
                has_payload
                and (
                    active_dims * bits_per_dim > 0
                    or int(exact_extra_payload_bits.get(sender, 0)) > 0
                ))
            if active:
                self._last_sent_comm_token_masks[sender] = np.asarray(
                    self._pending_comm_token_masks.get(
                        sender, np.ones(self.Q, dtype=np.float64)),
                    dtype=np.float64,
                ).copy()
                if self._comm_payload_mode == 'target_tokens':
                    payload = np.asarray(
                        self._pending_comm_messages[sender],
                        dtype=np.float64).reshape(
                            self.Q, self._comm_target_token_dim)
                    self._last_sent_comm_target_claims[sender] = (
                        payload[:, 0].copy())
        immediate_hyperedge_receivers: Dict[int, list[int]] = {}
        immediate_certificate_receivers: Dict[int, list[int]] = {}
        immediate_owner_receivers: Dict[int, list[int]] = {}
        immediate_metadata_by_sender: Dict[int, dict] = {}
        for item in deliveries:
            metadata = {
                'rate_index': int(item.rate_index),
                'bits_per_dim': int(dec_suff_bits.get(
                    int(item.sender),
                    self._inter_uav_comm.rate_bits_per_dim[int(np.clip(
                        int(item.rate_index),
                        0,
                        len(self._inter_uav_comm.rate_bits_per_dim) - 1,
                    ))],
                )),
                'latency_s': float(item.latency_s),
                'snr_db': float(item.snr_db),
                'age_frames': 0,
                'sent_frame': int(self.t),
                'tx_power_w': float(item.tx_power_w),
            }
            if item.token_mask is not None:
                metadata['token_mask'] = item.token_mask.copy()
            if item.sender in quantized_qpd_protocol:
                metadata['qpd_protocol'] = (
                    quantized_qpd_protocol[item.sender].copy())
            if item.sender in quantized_hyperedge_protocol:
                metadata['hyperedge_protocol'] = (
                    quantized_hyperedge_protocol[item.sender].copy())
                metadata['hyperedge_protocol_frame'] = int(
                    self._pending_hyperedge_protocol_frame.get(
                        item.sender, self.t))
            if item.sender in quantized_composable_certificate:
                metadata['composable_certificate'] = (
                    quantized_composable_certificate[item.sender].copy())
                metadata['composable_certificate_frame'] = int(
                    self._composable_certificate_local_frame[item.sender])
                metadata['composable_certificate_dual_upper'] = float(
                    self._composable_certificate_last_sent_dual_upper[
                        item.sender])
                if item.sender in quantized_composable_targetwise_upper:
                    metadata['composable_certificate_targetwise_upper'] = (
                        quantized_composable_targetwise_upper[
                            item.sender].copy())
            if item.sender in quantized_owner_posterior:
                metadata['owner_posterior'] = (
                    quantized_owner_posterior[item.sender])
            # A sub-frame transmission is available in the next observation;
            # longer delays remain in the mailbox until their due frame.
            due_frame = self.t + max(0, item.delay_frames - 1)
            if due_frame <= self.t:
                inbox[item.receiver][item.sender] = item.message.copy()
                inbox_meta[item.receiver][item.sender] = metadata
                if self._qpd_enabled:
                    if 'qpd_protocol' in metadata:
                        decoded = self._decode_qpd_control_packet(
                            metadata['qpd_protocol'],
                            metadata.get('token_mask'))
                        self._merge_received_qpd_packet(
                            int(item.receiver), int(item.sender),
                            decoded, decoded['active'])
                    elif self._qpd_overwrite_protocol_header:
                        self._merge_received_qpd_protocol(
                            int(item.receiver),
                            int(item.sender),
                            item.message,
                            metadata.get('token_mask'),
                        )
                if (self._hyperedge_enabled
                        and 'hyperedge_protocol' in metadata):
                    immediate_hyperedge_receivers.setdefault(
                        int(item.sender), []).append(int(item.receiver))
                if (
                    self._composable_certificate_enabled
                    and 'composable_certificate' in metadata
                ):
                    immediate_certificate_receivers.setdefault(
                        int(item.sender), []).append(int(item.receiver))
                if (
                    self._owner_posterior_enabled
                    and 'owner_posterior' in metadata
                ):
                    immediate_owner_receivers.setdefault(
                        int(item.sender), []).append(int(item.receiver))
                immediate_metadata_by_sender[int(item.sender)] = metadata
            else:
                self._comm_mailbox.append((
                    due_frame, item.receiver, item.sender, item.message.copy(),
                    metadata))

        for sender, receivers in immediate_hyperedge_receivers.items():
            metadata = immediate_metadata_by_sender[sender]
            self._merge_received_hyperedge_packet_many(
                np.asarray(receivers, dtype=np.int64),
                sender,
                metadata['hyperedge_protocol'],
                metadata.get('token_mask'),
                sent_frame=metadata.get(
                    'hyperedge_protocol_frame', metadata.get('sent_frame')),
            )
        for sender, receivers in immediate_certificate_receivers.items():
            metadata = immediate_metadata_by_sender[sender]
            self._merge_composable_certificate_packet_many(
                np.asarray(receivers, dtype=np.int64),
                sender,
                metadata['composable_certificate'],
                float(metadata['composable_certificate_dual_upper']),
                int(metadata['composable_certificate_frame']),
                metadata.get('composable_certificate_targetwise_upper'),
            )
        for sender, receivers in immediate_owner_receivers.items():
            metadata = immediate_metadata_by_sender[sender]
            self._merge_owner_posterior_packet_many(
                np.asarray(receivers, dtype=np.int64),
                sender,
                metadata['owner_posterior'],
            )

        structure_attempted = 0
        structure_delivered = 0
        structure_payload_bits = 0.0
        if self._structure_student_channel_enabled:
            deliveries_by_sender: Dict[int, list] = {}
            for item in deliveries:
                deliveries_by_sender.setdefault(
                    int(item.sender), []).append(item)
            for sender, protocol in quantized_structure_protocol.items():
                structure_attempted += 1
                structure_payload_bits += float(
                    exact_extra_payload_bits[int(sender)])
                sender_deliveries = deliveries_by_sender.get(
                    int(sender), [])
                receivers = {
                    int(item.receiver) for item in sender_deliveries
                }
                if len(receivers) != max(self.K - 1, 0):
                    continue
                structure_delivered += 1
                due_frame = self.t + max(
                    0,
                    max(
                        (int(item.delay_frames)
                         for item in sender_deliveries),
                        default=1,
                    ) - 1,
                )
                if due_frame <= self.t:
                    self._merge_structure_student_public_protocol(
                        int(sender), protocol, int(self.t))
                else:
                    self._structure_student_mailbox.append((
                        due_frame,
                        int(sender),
                        protocol.copy(),
                        int(self.t),
                    ))

            self._refresh_structure_student_edges()
            valid = self._structure_student_public_valid
            ages = (
                self.t
                - self._structure_student_public_last_seen[valid]
            )
            self._structure_student_metrics = {
                'structure_student_channel_enabled': 1.0,
                'structure_student_endpoint_width': float(
                    self._structure_student_endpoint_width),
                'structure_student_bits_per_dim': float(
                    self._structure_student_bits_per_dim),
                'structure_student_adaptive_min_bits_per_dim': float(
                    self._structure_student_adaptive_min_bits_per_dim),
                'structure_student_selected_bits_per_dim': float(np.mean(
                    list(selected_structure_bits.values()) or [
                        self._structure_student_bits_per_dim])),
                'structure_student_min_comm_fraction': float(
                    self._structure_student_min_comm_fraction),
                'structure_student_payload_bits': float(
                    structure_payload_bits),
                'structure_student_atomic_attempted_senders': float(
                    structure_attempted),
                'structure_student_atomic_delivered_senders': float(
                    structure_delivered),
                'structure_student_atomic_delivery_rate': float(
                    structure_delivered / structure_attempted)
                if structure_attempted > 0 else 1.0,
                'structure_student_public_valid_fraction': float(
                    np.mean(valid)),
                'structure_student_insufficient_cache': float(
                    np.sum(valid) < 2),
                'structure_student_public_mean_age_frames': float(
                    np.mean(ages)) if ages.size else 0.0,
                'structure_student_public_max_age_frames': float(
                    np.max(ages)) if ages.size else 0.0,
            }
        else:
            self._structure_student_metrics = {}

        self._refresh_composable_certificate_metrics()
        self._fuse_delivered_owner_posteriors()
        if self._owner_posterior_enabled:
            self._owner_posterior_metrics.update({
                'owner_posterior_payload_bits': float(
                    owner_posterior_payload_bits),
                'owner_posterior_payload_entries': float(sum(
                    np.asarray(packet['targets']).size
                    for packet in quantized_owner_posterior.values()
                )),
            })

        for sender, energy_j in stats.per_sender_energy_j.items():
            if 0 <= sender < len(self.uavs):
                self.uavs[sender].battery = max(
                    0.0, self.uavs[sender].battery - float(energy_j))

        if self._joint_isac_power_enabled:
            # Sensing allocation is held for the full simulator frame; packet
            # energy uses its actual serialization airtime.
            # Save the post-communication battery state.  A later analytical
            # solver may deliberately use less than the provisional sensing
            # budget, in which case accounting is recomputed from this state.
            self._isac_sensing_battery_before = np.asarray(
                [uav.battery for uav in self.uavs], dtype=np.float64)
            for k in range(self.K):
                sensing_energy = float(
                    np.sum(self._current_sensing_power_w[k])
                    * self._sensing_slot_duration_s())
                self.uavs[k].battery = max(
                    0.0, self.uavs[k].battery - sensing_energy)
            allocated = self._current_comm_power_w + np.sum(
                self._current_sensing_power_w, axis=1)
            self._last_isac_metrics = {
                'isac_total_power_budget_w': float(
                    self.K * self._isac_total_power_w),
                'isac_comm_power_w': float(np.sum(
                    self._current_comm_power_w)),
                'isac_sensing_power_w': float(np.sum(
                    self._current_sensing_power_w)),
                'isac_max_power_balance_error_w': float(np.max(np.abs(
                    allocated - self._isac_total_power_w))),
                'isac_max_power_budget_violation_w': float(np.max(np.maximum(
                    allocated - self._isac_total_power_w, 0.0))),
                'isac_unused_power_w': float(np.sum(np.maximum(
                    self._isac_total_power_w - allocated, 0.0))),
                'isac_per_uav_comm_power_w': self._current_comm_power_w.copy(),
                'isac_per_uav_sensing_power_w': np.sum(
                    self._current_sensing_power_w, axis=1),
                'isac_target_power_w': np.sum(
                    self._current_sensing_power_w, axis=0),
            }
        else:
            self._last_isac_metrics = {}
        self._last_isac_metrics.update(
            self._structure_student_metrics)
        self._last_isac_metrics.update(
            self._composable_certificate_metrics)
        self._last_isac_metrics.update(
            self._owner_posterior_metrics)

        self._pending_comm_messages = {}
        self._pending_comm_rates = {}
        self._pending_comm_token_masks = {}
        self._pending_qpd_protocol = {}
        self._pending_hyperedge_protocol = {}
        self._pending_hyperedge_protocol_frame = {}
        self._pending_structure_student_protocol = {}
        self._pending_comm_power_fractions = {}
        self._pending_sensing_weights = {}
        self._received_comm_msgs = inbox
        self._received_comm_meta = inbox_meta
        self._last_comm_stats = stats
        return stats

    def _resolve_persistent_token_commitments(self) -> np.ndarray:
        """Return the sparse Token-derived commitment graph for this frame.

        Each sender owns one local persistent intent. A newly observed local
        sensing or transmitted mask may replace it only after the configured
        minimum hold. During a
        bounded handover, the old and new sparse masks coexist so receivers do
        not lose the previous endpoint before learning the replacement.
        Silence retains the last physically transmitted mask up to
        ``max_age_frames``; after expiry the local sensing top-k is used as a
        safe fallback. No target geometry or team reward enters this update.
        """
        if self._persistent_commitment_resolved_frame == self.t:
            return self._persistent_commitment_effective_mask.copy()

        fallback = np.zeros((self.K, self.Q), dtype=bool)
        order = np.argsort(
            -self._current_sensing_power_w, axis=1, kind='stable')
        rows = np.arange(self.K)[:, None]
        fallback[
            rows,
            order[:, :min(
                self._distributed_target_commitment_topk, self.Q)],
        ] = True

        switches = 0
        active_handover = 0
        stale_fallbacks = 0
        for k in range(self.K):
            incoming_raw = (
                fallback[k]
                if self._distributed_target_commitment_source
                == 'persistent_sensing'
                else self._last_sent_comm_token_masks.get(k))
            incoming = None
            if incoming_raw is not None:
                candidate = np.asarray(incoming_raw).reshape(-1) > 0.5
                if candidate.shape != (self.Q,):
                    raise ValueError(
                        'sent target-token mask has invalid shape')
                if np.any(candidate):
                    incoming = candidate

            current = self._persistent_commitment_mask[k]
            initialized = bool(np.any(current))
            if not initialized:
                accepted = incoming if incoming is not None else fallback[k]
                self._persistent_commitment_mask[k] = accepted
                self._persistent_commitment_old_mask[k] = False
                self._persistent_commitment_last_switch[k] = self.t
                self._persistent_commitment_last_seen[k] = self.t
                self._persistent_commitment_handover_remaining[k] = 0
            elif incoming is not None:
                self._persistent_commitment_last_seen[k] = self.t
                changed = not np.array_equal(incoming, current)
                held_frames = (
                    self.t - self._persistent_commitment_last_switch[k])
                if (changed
                        and held_frames >= (
                            self._distributed_target_commitment_min_hold_frames)):
                    self._persistent_commitment_old_mask[k] = current.copy()
                    self._persistent_commitment_mask[k] = incoming
                    self._persistent_commitment_last_switch[k] = self.t
                    self._persistent_commitment_handover_remaining[k] = (
                        self._distributed_target_commitment_handover_frames)
                    switches += 1
            elif (self.t - self._persistent_commitment_last_seen[k]
                  > self._distributed_target_commitment_max_age_frames):
                stale_fallbacks += 1
                if not np.array_equal(current, fallback[k]):
                    self._persistent_commitment_old_mask[k] = current.copy()
                    self._persistent_commitment_mask[k] = fallback[k]
                    self._persistent_commitment_last_switch[k] = self.t
                    self._persistent_commitment_handover_remaining[k] = (
                        self._distributed_target_commitment_handover_frames)
                    switches += 1
                self._persistent_commitment_last_seen[k] = self.t

            effective = self._persistent_commitment_mask[k].copy()
            if self._persistent_commitment_handover_remaining[k] > 0:
                effective |= self._persistent_commitment_old_mask[k]
                self._persistent_commitment_handover_remaining[k] -= 1
                active_handover += 1
            else:
                self._persistent_commitment_old_mask[k] = False
            self._persistent_commitment_effective_mask[k] = effective

        self._persistent_commitment_resolved_frame = self.t
        self._persistent_commitment_metrics = {
            'learned_comm_commitment_source': (
                self._distributed_target_commitment_source),
            'learned_comm_commitment_switch_rate': float(
                switches / max(self.K, 1)),
            'learned_comm_commitment_handover_rate': float(
                active_handover / max(self.K, 1)),
            'learned_comm_commitment_stale_fallback_rate': float(
                stale_fallbacks / max(self.K, 1)),
            'learned_comm_commitment_effective_claims_per_uav': float(
                np.mean(self._persistent_commitment_effective_mask.sum(
                    axis=1))),
        }
        return self._persistent_commitment_effective_mask.copy()

    def _resolve_qpd_commitments(self) -> np.ndarray:
        """Resolve one sparse commitment row per UAV from local inboxes only.

        The environment loops over UAVs for simulation efficiency, but row
        ``k`` is computed exclusively from UAV ``k``'s own QPD state and the
        delayed/quantized packets in receiver ``k``'s physical inbox.
        """
        old_primal = self._qpd_primal.copy()
        new_primal = np.zeros_like(old_primal)
        new_price = np.zeros_like(self._qpd_target_price)
        local_residuals = np.zeros(self.K, dtype=np.float64)
        local_loads = np.zeros((self.K, self.Q), dtype=np.float64)
        visible_peers = np.zeros(self.K, dtype=np.float64)
        effective_bids = np.zeros((self.K, self.Q), dtype=np.float64)

        for k in range(self.K):
            peer_rows = self._qpd_received_primal[k].copy()
            peer_rows[k] = 0.0
            peer_primal = np.sum(peer_rows, axis=0)
            visible_target = (
                self._qpd_received_last_seen[k] > FRAME_NEVER)
            visible_target[k] = False
            peer_queue = np.where(
                visible_target,
                self._qpd_received_queue[k],
                0.0,
            )
            team_queue = (
                self._qpd_queue[k] / self._qpd_queue_max
                + np.sum(peer_queue, axis=0)
            ) / (
                1.0 + np.sum(visible_target, axis=0)
            )
            scarcity = team_queue - float(np.mean(team_queue))
            effective_bid = (
                np.tanh(self._qpd_bid[k])
                + self._qpd_peer_deficit_gain * scarcity
            )
            effective_bids[k] = effective_bid
            visible_peers[k] = float(np.sum(np.any(
                self._qpd_received_last_seen[k] > FRAME_NEVER,
                axis=1,
            )) - np.any(
                self._qpd_received_last_seen[k, k] > FRAME_NEVER))

            result = local_primal_dual_update(
                effective_bid,
                peer_primal,
                old_primal[k],
                self._qpd_target_price[k],
                row_capacity=self._qpd_row_capacity,
                target_capacity=self._qpd_target_capacity,
                primal_step=self._qpd_primal_step,
                dual_step=self._qpd_dual_step,
                rounds=self._qpd_rounds,
                price_max=self._qpd_price_max,
                exploration_floor=(
                    self._qpd_primal_exploration_floor),
            )
            new_primal[k] = result.primal
            new_price[k] = result.target_price
            local_residuals[k] = result.kkt_residual
            local_loads[k] = result.estimated_target_load

        self._qpd_previous_primal = old_primal
        self._qpd_primal = new_primal
        self._qpd_target_price = new_price

        committed = np.zeros((self.K, self.Q), dtype=bool)
        max_claims = max(
            1, min(self.Q, int(np.ceil(self._qpd_row_capacity))))
        for k in range(self.K):
            order = np.argsort(-new_primal[k], kind='stable')
            committed[k, order[0]] = True
            for q in order[1:max_claims]:
                if new_primal[k, q] >= self._qpd_commitment_threshold:
                    committed[k, q] = True
        self._qpd_commitment_mask = committed

        row_sums = np.sum(new_primal, axis=1)
        target_load = np.sum(new_primal, axis=0)
        target_error = np.abs(
            target_load - np.minimum(
                target_load, self._qpd_target_capacity))
        probs = np.divide(
            new_primal,
            np.maximum(row_sums[:, None], 1e-12),
            out=np.zeros_like(new_primal),
            where=row_sums[:, None] > 1e-12,
        )
        if self.Q > 1:
            entropy = -np.sum(
                probs * np.log(np.maximum(probs, 1e-12)), axis=1
            ) / np.log(float(self.Q))
        else:
            entropy = np.zeros(self.K, dtype=np.float64)
        self._qpd_metrics = {
            'qpd_enabled': 1.0,
            'qpd_queue_mean': float(np.mean(self._qpd_queue)),
            'qpd_queue_max': float(np.max(
                self._qpd_queue, initial=0.0)),
            'qpd_bid_mean': float(np.mean(self._qpd_bid)),
            'qpd_bid_max': float(np.max(
                self._qpd_bid, initial=0.0)),
            'qpd_effective_bid_span': float(np.mean(
                np.ptp(effective_bids, axis=1))),
            'qpd_send_target_rate': float(np.mean(
                self._qpd_send_mask)),
            'qpd_active_claims_per_uav': float(np.mean(
                committed.sum(axis=1))),
            'qpd_visible_peers_per_uav': float(np.mean(visible_peers)),
            'qpd_primal_row_sum_mean': float(np.mean(row_sums)),
            'qpd_primal_row_sum_max': float(np.max(
                row_sums, initial=0.0)),
            'qpd_primal_entropy': float(np.mean(entropy)),
            'qpd_target_load_min': float(np.min(
                target_load)) if target_load.size else 0.0,
            'qpd_target_load_max': float(np.max(
                target_load, initial=0.0)),
            'qpd_target_overload': float(np.max(
                target_error, initial=0.0)),
            'qpd_local_estimated_load_mean': float(np.mean(local_loads)),
            'qpd_price_abs_mean': float(np.mean(np.abs(new_price))),
            'qpd_kkt_residual_mean': float(np.mean(local_residuals)),
            'qpd_kkt_residual_max': float(np.max(
                local_residuals, initial=0.0)),
            'qpd_local_iterations': float(self._qpd_rounds),
        }
        self._last_isac_metrics.update(self._qpd_metrics)
        return committed.copy()

    def _resolve_hyperedge_negotiation(
        self,
    ) -> Tuple[Tuple[int, int, int], ...]:
        """Resolve reciprocal plans without a simulator-truth validity filter.

        Local plans are functions only of physically delivered endpoint state,
        local target beliefs and public configuration.  A planned edge whose
        true delay/Doppler support is poor must therefore fail through its
        realized receiver statistic and subsequent AoI/belief evolution; the
        protocol may not inspect the environment's true feasible-edge set to
        erase the mistake before execution.
        """
        held = tuple(self._hyperedge_selected_set)
        hold_active = bool(
            held
            and len(held) == len(self._hyperedge_selected_set)
            and self.t - self._hyperedge_last_update_frame
            < self._hyperedge_assignment_hold_frames
        )
        if hold_active and not self._distributed_replicated_power_enabled:
            coverage = float(len({
                target for _, _, target in held
            }) / max(self.Q, 1))
            self._hyperedge_metrics = {
                **self._hyperedge_metrics,
                'hyperedge_enabled': 1.0,
                'hyperedge_active_edges': float(len(held)),
                'hyperedge_target_coverage': coverage,
                'hyperedge_protocol_used': 1.0,
                'hyperedge_safety_fallback': 0.0,
                'hyperedge_assignment_reused': 1.0,
                'hyperedge_assignment_age_frames': float(
                    self.t - self._hyperedge_last_update_frame),
                'hyperedge_assignment_hold_frames': float(
                    self._hyperedge_assignment_hold_frames),
            }
            self._last_isac_metrics.update(self._hyperedge_metrics)
            return held

        local_plans = []
        public_coefficient_views = []
        certificate_coefficient_views = []
        certificate_upper_coefficient_views = []
        public_full_views = []
        visible_peer_counts = []
        local_min_proxy = []
        selected_batch_inputs = []
        selected_batch_coefficients = None
        selected_batch_dc = None
        selected_batch_coefficient_scale = 0.0
        selected_batch_position_uncertainty_m = 0.0
        selected_batch_velocity_uncertainty_mps = 0.0
        golden_visible_views = (
            [] if self._acceleration_golden_trace_enabled else None)
        golden_endpoint_position_views = (
            [] if self._acceleration_golden_trace_enabled else None)
        golden_endpoint_velocity_views = (
            [] if self._acceleration_golden_trace_enabled else None)
        golden_target_position_views = (
            [] if self._acceleration_golden_trace_enabled else None)
        golden_target_velocity_views = (
            [] if self._acceleration_golden_trace_enabled else None)
        golden_target_position_uncertainty_views = (
            [] if self._acceleration_golden_trace_enabled else None)
        golden_target_velocity_uncertainty_views = (
            [] if self._acceleration_golden_trace_enabled else None)
        for viewer in range(self.K):
            target_position_3d, target_velocity_3d = (
                self._coordination_target_state_for_viewer(viewer))
            target_xy = target_position_3d[:, :2]
            tx_capability = np.zeros((self.K, self.Q), dtype=np.float64)
            rx_capability = np.zeros((self.K, self.Q), dtype=np.float64)
            deficit = np.zeros((self.K, self.Q), dtype=np.float64)
            visible = np.zeros((self.K, self.Q), dtype=bool)
            public_position = np.zeros(
                (self.K, self.Q, 2), dtype=np.float64)
            public_velocity = np.zeros(
                (self.K, self.Q, 2), dtype=np.float64)
            public_nearfield = np.zeros(
                (self.K, self.Q, 3), dtype=np.float64)

            visible[viewer] = True
            if self._hyperedge_pair_score_mode == 'budget_reconstructable':
                # A UAV has causal, exact access to its own post-action
                # proprioceptive state; using the previous submitted beacon
                # here would manufacture a one-frame self delay.
                public_position[viewer, :, 0] = float(
                    self.uavs[viewer].pos[0])
                public_position[viewer, :, 1] = float(
                    self.uavs[viewer].pos[1])
                public_velocity[viewer, :, 0] = float(
                    self.uavs[viewer].vel[0])
                public_velocity[viewer, :, 1] = float(
                    self.uavs[viewer].vel[1])
                if self._hyperedge_nearfield_residual_enabled:
                    public_nearfield[viewer] = (
                        self._hyperedge_local_offer[viewer, :, 4:7])
            else:
                tx_capability[viewer] = self._hyperedge_local_offer[
                    viewer, :, 0]
                rx_capability[viewer] = self._hyperedge_local_offer[
                    viewer, :, 1]
                deficit[viewer] = self._hyperedge_local_offer[viewer, :, 2]
            if (self._hyperedge_state_stream_enabled
                    and self._hyperedge_pair_score_mode
                    != 'budget_reconstructable'):
                public_position[viewer, :, 0] = (
                    self._hyperedge_local_offer[viewer, :, 3]
                    * float(self.area_size[0]))
                public_position[viewer, :, 1] = (
                    self._hyperedge_local_offer[viewer, :, 4]
                    * float(self.area_size[1]))
                public_velocity[viewer, :, 0] = (
                    (2.0 * self._hyperedge_local_offer[viewer, :, 5] - 1.0)
                    * float(self.cfg.uav.v_max))
                public_velocity[viewer, :, 1] = (
                    (2.0 * self._hyperedge_local_offer[viewer, :, 6] - 1.0)
                    * float(self.cfg.uav.v_max))

            received = (
                self._hyperedge_received_last_seen[viewer] > FRAME_NEVER)
            visible |= received
            if self._hyperedge_pair_score_mode == 'budget_reconstructable':
                public_position[:, :, 0][received] = (
                    self._hyperedge_received_offer[
                        viewer, :, :, 0][received]
                    * float(self.area_size[0]))
                public_position[:, :, 1][received] = (
                    self._hyperedge_received_offer[
                        viewer, :, :, 1][received]
                    * float(self.area_size[1]))
                public_velocity[:, :, 0][received] = (
                    (2.0 * self._hyperedge_received_offer[
                        viewer, :, :, 2][received] - 1.0)
                    * float(self.cfg.uav.v_max))
                public_velocity[:, :, 1][received] = (
                    (2.0 * self._hyperedge_received_offer[
                        viewer, :, :, 3][received] - 1.0)
                    * float(self.cfg.uav.v_max))
                if self._hyperedge_nearfield_residual_enabled:
                    public_nearfield[received] = (
                        self._hyperedge_received_offer[
                            viewer, :, :, 4:7][received])
                # Dead-reckon physically delivered peer state from its source
                # frame. This uses only the packet's quantized velocity and
                # local frame clock, then clips to the public flight box.
                source_age = np.where(
                    received,
                    np.maximum(
                        self.t
                        - self._hyperedge_received_last_seen[viewer],
                        0,
                    ),
                    0,
                ).astype(np.float64)
                public_position += (
                    public_velocity
                    * source_age[:, :, None]
                    * float(self.cfg.scenario.dt))
                public_position[:, :, 0] = np.clip(
                    public_position[:, :, 0],
                    0.0,
                    float(self.area_size[0]),
                )
                public_position[:, :, 1] = np.clip(
                    public_position[:, :, 1],
                    0.0,
                    float(self.area_size[1]),
                )
            else:
                tx_capability[received] = self._hyperedge_received_offer[
                    viewer, :, :, 0][received]
                rx_capability[received] = self._hyperedge_received_offer[
                    viewer, :, :, 1][received]
                deficit[received] = self._hyperedge_received_offer[
                    viewer, :, :, 2][received]
            if (self._hyperedge_state_stream_enabled
                    and self._hyperedge_pair_score_mode
                    != 'budget_reconstructable'):
                public_position[:, :, 0][received] = (
                    self._hyperedge_received_offer[
                        viewer, :, :, 3][received]
                    * float(self.area_size[0]))
                public_position[:, :, 1][received] = (
                    self._hyperedge_received_offer[
                        viewer, :, :, 4][received]
                    * float(self.area_size[1]))
                public_velocity[:, :, 0][received] = (
                    (2.0 * self._hyperedge_received_offer[
                        viewer, :, :, 5][received] - 1.0)
                    * float(self.cfg.uav.v_max))
                public_velocity[:, :, 1][received] = (
                    (2.0 * self._hyperedge_received_offer[
                        viewer, :, :, 6][received] - 1.0)
                    * float(self.cfg.uav.v_max))
            visible_peer_counts.append(float(np.sum(
                np.any(received, axis=1))))
            if golden_visible_views is not None:
                golden_visible_views.append(visible.copy())

            if self._hyperedge_pair_score_mode == 'budget_reconstructable':
                # Uniform priority preserves the max-min objective and makes
                # replicated plans independent of private detection history.
                target_deficit = np.zeros(self.Q, dtype=np.float64)
            else:
                # Each viewer uses a conservative maximum of the deficits it
                # can actually see; missing targets retain its local deficit.
                visible_deficit = np.where(visible, deficit, -np.inf)
                target_deficit = np.max(visible_deficit, axis=0)
                target_deficit[~np.isfinite(target_deficit)] = (
                    self._hyperedge_local_offer[viewer, :, 2][
                        ~np.isfinite(target_deficit)])
            pair_value = None
            if self._hyperedge_pair_score_mode == 'physical_reconstructable':
                pair_value = reconstruct_bistatic_pair_value(
                    tx_capability,
                    public_position,
                    target_xy,
                    visible,
                    distance_scale_m=self._hyperedge_distance_scale_m,
                )
            if self._hyperedge_pair_score_mode == 'budget_reconstructable':
                dc = self.deflection_computer
                coefficient_scale = float(
                    dc.c_det * dc.M * dc.N * dc.antenna_gain * dc.n_cpi
                    / max(dc.noise_power, 1.0e-15)
                )
                position_state_bits = min(
                    self._hyperedge_state_field_bits[:2])
                velocity_state_bits = min(
                    self._hyperedge_state_field_bits[2:4])
                residual_state_bits = (
                    min(self._hyperedge_state_field_bits[5:7])
                    if (
                        self._hyperedge_nearfield_residual_enabled
                        and len(self._hyperedge_state_field_bits) >= 7
                    ) else 0
                )
                quantization_uncertainty_m = (
                    0.5 * np.hypot(
                        float(self.area_size[0])
                        / max(
                            2 ** position_state_bits - 1, 1),
                        float(self.area_size[1])
                        / max(
                            2 ** position_state_bits - 1, 1),
                    )
                    if (
                        self._hyperedge_robust_quantization_enabled
                        and position_state_bits > 0
                    ) else 0.0
                )
                quantization_velocity_uncertainty_mps = (
                    np.sqrt(2.0) * float(self.cfg.uav.v_max)
                    / max(
                        2 ** velocity_state_bits - 1, 1)
                    if (
                        self._hyperedge_robust_quantization_enabled
                        and velocity_state_bits > 0
                    ) else 0.0
                )
                target_position_uncertainty_m = np.zeros(
                    self.Q, dtype=np.float64)
                target_velocity_uncertainty_mps = np.zeros(
                    self.Q, dtype=np.float64)
                if (
                    self._distributed_target_position_uncertainty_sigma > 0.0
                    and self.belief_mgr is not None
                ):
                    viewer_covariance = np.asarray(
                        self.belief_mgr.cov[viewer], dtype=np.float64)
                    position_covariance = 0.5 * (
                        viewer_covariance[:, :2, :2]
                        + np.swapaxes(
                            viewer_covariance[:, :2, :2], -1, -2))
                    velocity_covariance = 0.5 * (
                        viewer_covariance[:, 2:4, 2:4]
                        + np.swapaxes(
                            viewer_covariance[:, 2:4, 2:4], -1, -2))
                    sigma = float(
                        self._distributed_target_position_uncertainty_sigma)
                    target_position_uncertainty_m = sigma * np.sqrt(
                        np.maximum(
                            symmetric_2x2_max_eigenvalue(
                                position_covariance),
                            0.0,
                        )
                    )
                    target_velocity_uncertainty_mps = sigma * np.sqrt(
                        np.maximum(
                            symmetric_2x2_max_eigenvalue(
                                velocity_covariance),
                            0.0,
                        )
                    )
                if golden_endpoint_position_views is not None:
                    golden_endpoint_position_views.append(
                        public_position.copy())
                    golden_endpoint_velocity_views.append(
                        public_velocity.copy())
                    golden_target_position_views.append(
                        target_position_3d.copy())
                    golden_target_velocity_views.append(
                        target_velocity_3d.copy())
                    golden_target_position_uncertainty_views.append(
                        target_position_uncertainty_m.copy())
                    golden_target_velocity_uncertainty_views.append(
                        target_velocity_uncertainty_mps.copy())
                public_node = np.any(visible, axis=1)
                if (
                    hold_active
                    and not self._hyperedge_nearfield_residual_enabled
                ):
                    # All viewers use the same frozen COO edge set.  Defer
                    # their continuous gain refresh to one viewer-batched
                    # kernel after public-state construction finishes.
                    selected_batch_inputs.append((
                        public_position,
                        public_velocity,
                        target_position_3d,
                        target_velocity_3d,
                        visible,
                        target_position_uncertainty_m,
                        target_velocity_uncertainty_mps,
                    ))
                    selected_batch_dc = dc
                    selected_batch_coefficient_scale = coefficient_scale
                    selected_batch_position_uncertainty_m = float(
                        quantization_uncertainty_m)
                    selected_batch_velocity_uncertainty_mps = float(
                        quantization_velocity_uncertainty_mps)
                    public_full_views.append(bool(np.all(public_node)))
                    continue
                public_coefficient = (
                    self._hyperedge_acceleration.reconstruct_dense(
                        public_position,
                        public_velocity,
                        target_position_3d,
                        target_velocity_3d,
                        visible,
                        uav_height_m=float(self.cfg.scenario.height),
                        fc_hz=float(dc.fc),
                        rcs_m2=float(dc.rcs),
                        delta_f_hz=float(dc.delta_f),
                        symbol_period_s=float(dc.T_sym),
                        delay_bins=int(dc.M),
                        doppler_bins=int(dc.N),
                        dd_gate_min=float(dc.g_min),
                        coefficient_scale=coefficient_scale,
                        position_uncertainty_m=quantization_uncertainty_m,
                        dd_gain_mode=str(dc.dd_gain_mode),
                        robust_dd_uncertainty=False,
                    )
                )
                power_coefficient = public_coefficient
                if (
                    self._distributed_replicated_power_enabled
                    and self._hyperedge_nearfield_residual_enabled
                ):
                    # All receivers decode the same target-relative residual,
                    # preserving common views while refining near-field range.
                    power_position = public_position.copy()
                    power_velocity = public_velocity.copy()
                    endpoint_uncertainty = np.full(
                        self.K,
                        float(quantization_uncertainty_m),
                        dtype=np.float64,
                    )
                    levels = max(2 ** residual_state_bits - 1, 1)
                    residual_range = float(
                        self._hyperedge_nearfield_residual_range_m)
                    companding_mu = float(
                        self._hyperedge_nearfield_residual_companding_mu)
                    for node in range(self.K):
                        if not public_node[node]:
                            continue
                        if node == viewer:
                            power_position[node, :, 0] = float(
                                self.uavs[viewer].pos[0])
                            power_position[node, :, 1] = float(
                                self.uavs[viewer].pos[1])
                            power_velocity[node, :, 0] = float(
                                self.uavs[viewer].vel[0])
                            power_velocity[node, :, 1] = float(
                                self.uavs[viewer].vel[1])
                            endpoint_uncertainty[node] = 0.0
                            continue
                        near = public_nearfield[node, 0]
                        target_code = int(np.rint(
                            float(near[0]) * float(self.Q)))
                        if not (0 <= target_code < self.Q):
                            continue
                        compact = 2.0 * np.asarray(
                            near[1:3], dtype=np.float64) - 1.0
                        if companding_mu > 0.0:
                            log_mu = float(np.log1p(companding_mu))
                            residual = (
                                np.sign(compact) * residual_range
                                * np.expm1(np.abs(compact) * log_mu)
                                / companding_mu
                            )
                            compact_upper = np.minimum(
                                np.abs(compact) + 1.0 / levels, 1.0)
                            component_error = (
                                residual_range / companding_mu
                                * (
                                    np.expm1(compact_upper * log_mu)
                                    - np.expm1(
                                        np.abs(compact) * log_mu)
                                )
                            )
                            residual_uncertainty = float(
                                np.linalg.norm(component_error))
                        else:
                            residual = compact * residual_range
                            residual_uncertainty = float(
                                np.sqrt(2.0) * residual_range / levels)
                        node_age = float(source_age[node, 0])
                        relative_velocity = (
                            public_velocity[node, 0]
                            - target_velocity_3d[target_code, :2])
                        refined_xy = (
                            target_position_3d[target_code, :2]
                            + residual
                            + relative_velocity
                            * node_age
                            * float(self.cfg.scenario.dt))
                        refined_xy = np.clip(
                            refined_xy,
                            np.zeros(2, dtype=np.float64),
                            np.asarray(self.area_size, dtype=np.float64),
                        )
                        power_position[node, :, 0] = refined_xy[0]
                        power_position[node, :, 1] = refined_xy[1]
                        endpoint_uncertainty[node] = residual_uncertainty
                    power_coefficient = (
                        self._hyperedge_acceleration.reconstruct_dense(
                            power_position,
                            power_velocity,
                            target_position_3d,
                            target_velocity_3d,
                            visible,
                            uav_height_m=float(self.cfg.scenario.height),
                            fc_hz=float(dc.fc),
                            rcs_m2=float(dc.rcs),
                            delta_f_hz=float(dc.delta_f),
                            symbol_period_s=float(dc.T_sym),
                            delay_bins=int(dc.M),
                            doppler_bins=int(dc.N),
                            dd_gate_min=float(dc.g_min),
                            coefficient_scale=coefficient_scale,
                            position_uncertainty_m=endpoint_uncertainty,
                            dd_gain_mode=str(dc.dd_gain_mode),
                            robust_dd_uncertainty=False,
                        )
                    )
                certificate_position = (
                    power_position
                    if (
                        self._distributed_replicated_power_enabled
                        and self._hyperedge_nearfield_residual_enabled
                    ) else public_position
                )
                certificate_position_uncertainty = (
                    (
                        endpoint_uncertainty
                        + 2.0
                        * source_age[:, 0]
                        * np.asarray([
                            uav.v_max * uav.dt for uav in self.uavs
                        ], dtype=np.float64)
                    )
                    if (
                        self._distributed_replicated_power_enabled
                        and self._hyperedge_nearfield_residual_enabled
                    ) else quantization_uncertainty_m
                )
                certificate_coefficient = (
                    self._hyperedge_acceleration.reconstruct_dense(
                        certificate_position,
                        public_velocity,
                        target_position_3d,
                        target_velocity_3d,
                        visible,
                        uav_height_m=float(self.cfg.scenario.height),
                        fc_hz=float(dc.fc),
                        rcs_m2=float(dc.rcs),
                        delta_f_hz=float(dc.delta_f),
                        symbol_period_s=float(dc.T_sym),
                        delay_bins=int(dc.M),
                        doppler_bins=int(dc.N),
                        dd_gate_min=float(dc.g_min),
                        coefficient_scale=coefficient_scale,
                        position_uncertainty_m=(
                            certificate_position_uncertainty),
                        target_position_uncertainty_m=(
                            target_position_uncertainty_m),
                        velocity_uncertainty_mps=(
                            quantization_velocity_uncertainty_mps),
                        target_velocity_uncertainty_mps=(
                            target_velocity_uncertainty_mps),
                        dd_gain_mode=str(dc.dd_gain_mode),
                        robust_dd_uncertainty=True,
                    )
                )
                certificate_upper_coefficient = (
                    self._hyperedge_acceleration.reconstruct_upper(
                        certificate_position,
                        target_position_3d,
                        visible,
                        uav_height_m=float(self.cfg.scenario.height),
                        fc_hz=float(dc.fc),
                        rcs_m2=float(dc.rcs),
                        coefficient_scale=coefficient_scale,
                        position_uncertainty_m=(
                            certificate_position_uncertainty),
                        target_position_uncertainty_m=(
                            target_position_uncertainty_m),
                    )
                )
                public_coefficient_views.append(power_coefficient.copy())
                certificate_coefficient_views.append(
                    certificate_coefficient.copy())
                certificate_upper_coefficient_views.append(
                    certificate_upper_coefficient.copy())
                public_full_views.append(bool(np.all(public_node)))
                if hold_active:
                    # The discrete L2 structure is frozen during its declared
                    # hold interval.  Distributed L1 still needs the freshly
                    # reconstructed nominal/lower/upper gains above, but no
                    # proposal, information ranking or reciprocal consensus
                    # result can alter the held edge set.  Skip that dead
                    # control-plane work without changing the power inputs.
                    continue
                # The independent sensing PA cap is the binding budget in the
                # current system. Missing nodes receive zero budget and cannot
                # become transmitters in a viewer's replicated solve.
                public_budget = np.where(
                    public_node,
                    float(self._sensing_power_cap_w),
                    0.0,
                )
                reports_per_receiver = (
                    max(1, int(
                        self.cfg.p0_solver.capacity_per_rx
                        // max(self.cfg.detection.B_q, 1)))
                    if self.ground_communication_enabled
                    else self.Q * self.cfg.detection.K_q_max
                )
                if self._hyperedge_coordination_mode == 'reserved_endpoint':
                    tx_role_mask = np.zeros(self.K, dtype=bool)
                    if self._hyperedge_reserved_tx_nodes:
                        tx_role_mask[np.asarray(
                            self._hyperedge_reserved_tx_nodes,
                            dtype=np.int64)] = True
                    else:
                        tx_role_mask = (
                            np.arange(self.K, dtype=np.int64) % 2 == 0)
                    normalized_information = None
                    if self._hyperedge_bistatic_information_ranking_enabled:
                        normalized_information = np.zeros(
                            (self.K, self.K, self.Q, 4, 4),
                            dtype=np.float64,
                        )
                        if not bool(tx_role_mask[viewer]):
                            receiver = int(viewer)
                            receiver_position = np.concatenate((
                                public_position[receiver],
                                np.full((self.Q, 1), float(
                                    self.cfg.scenario.height)),
                            ), axis=1)
                            receiver_velocity = np.concatenate((
                                public_velocity[receiver],
                                np.zeros((self.Q, 1), dtype=np.float64),
                            ), axis=1)
                            for target in range(self.Q):
                                prior = 0.5 * (
                                    self.belief_mgr.cov[
                                        receiver, target, :4, :4]
                                    + self.belief_mgr.cov[
                                        receiver, target, :4, :4].T)
                                eigenvalues, eigenvectors = np.linalg.eigh(prior)
                                prior_sqrt = (
                                    eigenvectors
                                    @ np.diag(np.sqrt(np.maximum(
                                        eigenvalues, 0.0)))
                                    @ eigenvectors.T)
                                for transmitter in np.flatnonzero(tx_role_mask):
                                    nominal_deflection = float(
                                        public_budget[transmitter]
                                        * public_coefficient[
                                            transmitter,
                                            receiver,
                                            target,
                                        ])
                                    if nominal_deflection <= 0.0:
                                        continue
                                    transmitter_position = np.asarray([
                                        public_position[transmitter, target, 0],
                                        public_position[transmitter, target, 1],
                                        float(self.cfg.scenario.height),
                                    ])
                                    transmitter_velocity = np.asarray([
                                        public_velocity[transmitter, target, 0],
                                        public_velocity[transmitter, target, 1],
                                        0.0,
                                    ])
                                    _, jacobian = (
                                        bistatic_range_doppler_measurement_and_jacobian(
                                            self.belief_mgr.mean[
                                                receiver, target],
                                            transmitter_position,
                                            transmitter_velocity,
                                            receiver_position[target],
                                            receiver_velocity[target],
                                            float(dc.fc),
                                        ))
                                    measurement_covariance = (
                                        bistatic_range_doppler_crlb(
                                            nominal_deflection,
                                            float(dc.M * dc.delta_f),
                                            float(dc.n_cpi * dc.N * dc.T_sym),
                                            efficiency=(
                                                self._belief_bistatic_crlb_efficiency),
                                            minimum_effective_deflection=(
                                                self._belief_bistatic_min_effective_deflection),
                                        ))
                                    h4 = jacobian[:, :4]
                                    information = (
                                        h4.T
                                        @ np.linalg.inv(measurement_covariance)
                                        @ h4)
                                    normalized = (
                                        prior_sqrt @ information @ prior_sqrt)
                                    normalized_information[
                                        transmitter, receiver, target] = 0.5 * (
                                            normalized + normalized.T)
                    plan = plan_reserved_endpoint_hyperedges(
                        public_coefficient,
                        visible,
                        public_budget,
                        viewer=viewer,
                        tx_role_mask=tx_role_mask,
                        target_pair_limit=int(
                            self.cfg.detection.K_q_max),
                        normalized_edge_information=normalized_information,
                        information_weight=(
                            self._hyperedge_bistatic_information_weight),
                    )
                elif (self._hyperedge_coordination_mode
                        == 'refined_endpoint'):
                    plan = plan_refined_endpoint_hyperedges(
                        public_coefficient,
                        visible,
                        public_budget,
                        viewer=viewer,
                        target_pair_limit=int(
                            self.cfg.detection.K_q_max),
                    )
                elif (self._hyperedge_coordination_mode
                        == 'coalition_endpoint'):
                    plan = plan_sparse_coalition_endpoint_hyperedges(
                        public_coefficient,
                        visible,
                        public_budget,
                        viewer=viewer,
                        target_pair_limit=int(
                            self.cfg.detection.K_q_max),
                        max_transmitters=self._hyperedge_tx_coalition_max,
                    )
                else:
                    plan = plan_budget_certified_hyperedges(
                        public_coefficient,
                        public_budget,
                        target_deficit,
                        p_fa=float(self.cfg.detection.P_FA),
                        p_d_floor=float(getattr(
                            self.cfg.marl, 'comm_qos_worst_min', 0.60)),
                        target_pair_limit=int(self.cfg.detection.K_q_max),
                        reports_per_receiver=reports_per_receiver,
                    )
            else:
                plan = plan_local_hyperedges(
                    tx_capability,
                    rx_capability,
                    visible,
                    target_deficit,
                    target_pair_limit=int(self.cfg.detection.K_q_max),
                    deficit_gain=self._hyperedge_deficit_gain,
                    proxy_floor=self._hyperedge_proxy_floor,
                    pair_value=pair_value,
                )
            local_plans.append(plan)
            local_min_proxy.append(float(np.min(
                plan.proxy_target_value)) if self.Q else 0.0)

        if selected_batch_inputs:
            if (
                len(selected_batch_inputs) != self.K
                or selected_batch_dc is None
            ):
                raise RuntimeError(
                    "incomplete viewer batch for held hyperedge refresh")
            batch_dc = selected_batch_dc
            selected_batch_coefficients = (
                self._hyperedge_acceleration.reconstruct_selected_batch(
                    np.stack([
                        item[0] for item in selected_batch_inputs
                    ], axis=0),
                    np.stack([
                        item[1] for item in selected_batch_inputs
                    ], axis=0),
                    np.stack([
                        item[2] for item in selected_batch_inputs
                    ], axis=0),
                    np.stack([
                        item[3] for item in selected_batch_inputs
                    ], axis=0),
                    np.stack([
                        item[4] for item in selected_batch_inputs
                    ], axis=0),
                    held,
                    uav_height_m=float(self.cfg.scenario.height),
                    fc_hz=float(batch_dc.fc),
                    rcs_m2=float(batch_dc.rcs),
                    delta_f_hz=float(batch_dc.delta_f),
                    symbol_period_s=float(batch_dc.T_sym),
                    delay_bins=int(batch_dc.M),
                    doppler_bins=int(batch_dc.N),
                    dd_gate_min=float(batch_dc.g_min),
                    coefficient_scale=(
                        selected_batch_coefficient_scale),
                    nominal_position_uncertainty_m=(
                        selected_batch_position_uncertainty_m),
                    certificate_position_uncertainty_m=(
                        selected_batch_position_uncertainty_m),
                    target_position_uncertainty_m=np.stack([
                        item[5] for item in selected_batch_inputs
                    ], axis=0),
                    velocity_uncertainty_mps=(
                        selected_batch_velocity_uncertainty_mps),
                    target_velocity_uncertainty_mps=np.stack([
                        item[6] for item in selected_batch_inputs
                    ], axis=0),
                    dd_gain_mode=str(batch_dc.dd_gain_mode),
                )
            )
        if hold_active:
            # No new proposal round was executed.  Report the actually active
            # held structure and preserve the previous consensus streak for
            # the next topology-update frame.
            mutual = held
            stable = held
        else:
            mutual = mutual_endpoint_consensus(
                local_plans,
                num_uavs=self.K,
                num_targets=self.Q,
                target_pair_limit=int(self.cfg.detection.K_q_max),
            )
            self._hyperedge_consensus_streak, stable = update_consensus_streak(
                self._hyperedge_consensus_streak,
                mutual,
                consensus_rounds=self._hyperedge_consensus_rounds,
            )

        active = tuple(stable)
        covered_targets = {target for _, _, target in active}
        coverage = float(
            len(covered_targets) / max(self.Q, 1))
        use_protocol = bool(
            active
            and coverage + 1e-12
            >= self._hyperedge_min_target_coverage
        )
        if hold_active:
            # Certificate-light L1 still needs current continuous gains, but
            # must not churn the frozen discrete L2 structure. The loop above
            # refreshes each local public-state view; retain the held edges.
            selected = held
            active = held
            coverage = float(len({
                target for _, _, target in held
            }) / max(self.Q, 1))
            use_protocol = True
        else:
            selected = active if use_protocol else tuple()
        self._hyperedge_selected_set = selected
        if selected_batch_coefficients is not None and len(selected) > 0:
            try:
                gain_views, _ = fixed_owner_gain_matrix_from_selected_values(
                    selected_batch_coefficients.nominal,
                    selected,
                    num_uavs=self.K,
                    num_targets=self.Q,
                )
                certificate_gain_views, _ = (
                    fixed_owner_gain_matrix_from_selected_values(
                        selected_batch_coefficients.lower,
                        selected,
                        num_uavs=self.K,
                        num_targets=self.Q,
                    )
                )
                certificate_gain_upper_views, _ = (
                    fixed_owner_gain_matrix_from_selected_values(
                        selected_batch_coefficients.upper,
                        selected,
                        num_uavs=self.K,
                        num_targets=self.Q,
                    )
                )
            except (
                IncompleteFixedOwnerStructureError,
                NonUniqueFixedOwnerStructureError,
            ):
                gain_views = np.zeros(
                    (self.K, self.K, self.Q), dtype=np.float64)
                certificate_gain_views = np.zeros_like(gain_views)
                certificate_gain_upper_views = np.zeros_like(gain_views)
            except ValueError as error:
                raise RuntimeError(
                    "invalid sparse fixed-owner gain batch") from error
            self._hyperedge_public_gain_views = gain_views
            self._composable_certificate_gain_views = certificate_gain_views
            robust_mix = self._distributed_replicated_power_robust_gain_mix
            if robust_mix > 0.0:
                self._hyperedge_public_gain_views = (
                    (1.0 - robust_mix) * gain_views
                    + robust_mix * certificate_gain_views
                )
            self._composable_certificate_gain_upper_views = (
                certificate_gain_upper_views)
            self._hyperedge_public_full_views = np.asarray(
                public_full_views, dtype=bool)
        elif (
            len(public_coefficient_views) == self.K
            and len(selected) > 0
        ):
            gain_views = np.zeros(
                (self.K, self.K, self.Q), dtype=np.float64)
            for viewer, public_coefficient in enumerate(
                    public_coefficient_views):
                try:
                    gain_views[viewer], _ = fixed_owner_gain_matrix(
                        public_coefficient, selected)
                except (
                    IncompleteFixedOwnerStructureError,
                    NonUniqueFixedOwnerStructureError,
                ):
                    # Incomplete local coverage fails closed for that view;
                    # it does not receive the environment's true gain graph.
                    gain_views[viewer] = 0.0
                except ValueError as error:
                    raise RuntimeError(
                        "invalid public fixed-owner gain view"
                    ) from error
            self._hyperedge_public_gain_views = gain_views
            certificate_gain_views = np.zeros_like(gain_views)
            for viewer, certificate_coefficient in enumerate(
                    certificate_coefficient_views):
                try:
                    certificate_gain_views[viewer], _ = fixed_owner_gain_matrix(
                        certificate_coefficient, selected)
                except (
                    IncompleteFixedOwnerStructureError,
                    NonUniqueFixedOwnerStructureError,
                ):
                    certificate_gain_views[viewer] = 0.0
                except ValueError as error:
                    raise RuntimeError(
                        "invalid conservative fixed-owner gain view"
                    ) from error
            self._composable_certificate_gain_views = certificate_gain_views
            robust_mix = self._distributed_replicated_power_robust_gain_mix
            if robust_mix > 0.0:
                # Coherent risk interpolation: each viewer trades nominal gain
                # against its own covariance-set lower gain.  No certificate
                # or hidden global state is injected, and rho=0 is exactly the
                # historical controller.  Only the viewer's own row executes.
                self._hyperedge_public_gain_views = (
                    (1.0 - robust_mix) * gain_views
                    + robust_mix * certificate_gain_views
                )
            certificate_gain_upper_views = np.zeros_like(gain_views)
            for viewer, certificate_upper in enumerate(
                    certificate_upper_coefficient_views):
                try:
                    certificate_gain_upper_views[viewer], _ = (
                        fixed_owner_gain_matrix(certificate_upper, selected))
                except (
                    IncompleteFixedOwnerStructureError,
                    NonUniqueFixedOwnerStructureError,
                ):
                    certificate_gain_upper_views[viewer] = 0.0
                except ValueError as error:
                    raise RuntimeError(
                        "invalid upper fixed-owner gain view"
                    ) from error
            self._composable_certificate_gain_upper_views = (
                certificate_gain_upper_views)
            self._hyperedge_public_full_views = np.asarray(
                public_full_views, dtype=bool)
        else:
            self._hyperedge_public_gain_views = np.zeros(
                (self.K, self.K, self.Q), dtype=np.float64)
            self._composable_certificate_gain_views = np.zeros(
                (self.K, self.K, self.Q), dtype=np.float64)
            self._composable_certificate_gain_upper_views = np.zeros(
                (self.K, self.K, self.Q), dtype=np.float64)
            self._hyperedge_public_full_views = np.zeros(
                self.K, dtype=bool)
        if use_protocol and not hold_active:
            self._hyperedge_last_update_frame = int(self.t)
        if self._acceleration_golden_trace_enabled:
            self._hyperedge_golden_snapshot = {
                'frame': int(self.t),
                'hold_active': bool(hold_active),
                'received_last_seen': (
                    self._hyperedge_received_last_seen.copy()),
                'visible_views': np.asarray(
                    golden_visible_views, dtype=bool),
                'endpoint_position_views': np.asarray(
                    golden_endpoint_position_views, dtype=np.float64),
                'endpoint_velocity_views': np.asarray(
                    golden_endpoint_velocity_views, dtype=np.float64),
                'target_position_views': np.asarray(
                    golden_target_position_views, dtype=np.float64),
                'target_velocity_views': np.asarray(
                    golden_target_velocity_views, dtype=np.float64),
                'target_position_uncertainty_views': np.asarray(
                    golden_target_position_uncertainty_views,
                    dtype=np.float64,
                ),
                'target_velocity_uncertainty_views': np.asarray(
                    golden_target_velocity_uncertainty_views,
                    dtype=np.float64,
                ),
                'local_plans': tuple({
                    'selected': tuple(
                        tuple(int(value) for value in edge)
                        for edge in plan.selected
                    ),
                    'proxy_target_value': np.asarray(
                        plan.proxy_target_value,
                        dtype=np.float64,
                    ).copy(),
                    'proxy_scores': tuple(
                        (tuple(int(value) for value in edge), float(score))
                        for edge, score in sorted(plan.proxy_scores.items())
                    ),
                } for plan in local_plans),
                'mutual': tuple(
                    tuple(int(value) for value in edge) for edge in mutual),
                'stable': tuple(
                    tuple(int(value) for value in edge) for edge in stable),
                'selected': tuple(
                    tuple(int(value) for value in edge) for edge in selected),
                'public_full_views': self._hyperedge_public_full_views.copy(),
                'public_gain_views': self._hyperedge_public_gain_views.copy(),
                'certificate_gain_views': (
                    self._composable_certificate_gain_views.copy()),
                'certificate_gain_upper_views': (
                    self._composable_certificate_gain_upper_views.copy()),
                'consensus_streak': self._hyperedge_consensus_streak.copy(),
            }
        acceleration_stats = self._hyperedge_acceleration.stats()
        self._hyperedge_metrics = {
            'hyperedge_enabled': 1.0,
            'hyperedge_visible_peers_per_uav': float(np.mean(
                visible_peer_counts or [0.0])),
            'hyperedge_local_min_proxy': (
                float(self._hyperedge_metrics.get(
                    'hyperedge_local_min_proxy', 0.0))
                if hold_active
                else float(np.mean(local_min_proxy or [0.0]))
            ),
            'hyperedge_mutual_edges': float(len(mutual)),
            'hyperedge_stable_edges': float(len(stable)),
            'hyperedge_active_edges': float(len(active)),
            'hyperedge_target_coverage': coverage,
            'hyperedge_protocol_used': float(use_protocol),
            'hyperedge_safety_fallback': float(
                not use_protocol and self._hyperedge_safety_fallback),
            'hyperedge_assignment_reused': float(hold_active),
            'hyperedge_sparse_hold_gain_refresh': float(
                hold_active
                and not self._hyperedge_nearfield_residual_enabled),
            'hyperedge_selected_batch_used': float(
                bool(selected_batch_inputs)),
            'hyperedge_assignment_age_frames': float(
                self.t - self._hyperedge_last_update_frame
                if hold_active else 0),
            'hyperedge_assignment_hold_frames': float(
                self._hyperedge_assignment_hold_frames),
            'hyperedge_consensus_rounds': float(
                self._hyperedge_consensus_rounds),
            'hyperedge_pair_score_mode': self._hyperedge_pair_score_mode,
            'hyperedge_coordination_mode': (
                self._hyperedge_coordination_mode),
            'hyperedge_local_belief_target_state': float(
                self._distributed_coordination_use_local_belief_targets),
            'hyperedge_acceleration_backend': str(
                acceleration_stats['backend']),
            'hyperedge_acceleration_dense_calls': float(
                acceleration_stats['dense_calls']),
            'hyperedge_acceleration_upper_calls': float(
                acceleration_stats['upper_calls']),
            'hyperedge_acceleration_selected_calls': float(
                acceleration_stats['selected_calls']),
            'hyperedge_acceleration_selected_batch_calls': float(
                acceleration_stats['selected_batch_calls']),
            'hyperedge_acceleration_total_seconds': float(
                acceleration_stats['total_seconds']),
            'hyperedge_acceleration_timing_enabled': float(
                acceleration_stats['timing_enabled']),
            'deflection_materialization_backend': str(
                self._deflection_materialization.name),
        }
        self._last_isac_metrics.update(self._hyperedge_metrics)
        return selected

    def _coverage_potential(self, uav_pos: np.ndarray, tgt_pos: np.ndarray) -> float:
        """Φ(s) = -Σ_q min_k ||uav_k - target_q|| (2D). Higher (less negative)
        when every target has a UAV near it. Used for potential-based shaping."""
        if tgt_pos.shape[0] == 0 or uav_pos.shape[0] == 0:
            return 0.0
        total = 0.0
        for q in range(tgt_pos.shape[0]):
            d = np.linalg.norm(uav_pos[:, :2] - tgt_pos[q, :2], axis=1)
            total += float(np.min(d))
        return -total

    def _per_watt_coefficient_from_entries(
        self, entries: list,
    ) -> np.ndarray:
        """Power-independent per-watt gain tensor reconstructed from entries.

        ``a_ijq = 1[g_dd >= g_min] * chi_rep * alpha^2 * C`` with
        ``C = M*N*G_tx*G_rx*n_CPI/P_noise`` under the dimensionless
        energy ratio ``E_signal/E_noise``.  This is read directly
        from the entry observables (not from ``d_eff / P_sense``), so it stays
        identified even for edges the learned sensing head left unexcited.
        """
        dc = self.deflection_computer
        scale = float(
            dc.c_det * dc.M * dc.N * dc.antenna_gain * dc.n_cpi
            / max(dc.noise_power, 1.0e-15)
        )
        coefficient = np.zeros((self.K, self.K, self.Q), dtype=np.float64)
        valid_entries = []
        for entry in entries:
            i, j, q = int(entry.i), int(entry.j), int(entry.q)
            if not (0 <= i < self.K and 0 <= j < self.K and 0 <= q < self.Q):
                continue
            if i == j:
                # audit 2026-08-17 (P2-7): defensive i != j mask.  No current
                # entry path produces self-loops (geometry/deflection skip
                # them), but the owner-value and fixed-owner-LP layers all
                # rely on the tensor having no i == j support, so mask it
                # explicitly here instead of depending on the caller.
                continue
            valid_entries.append((entry, i, j, q))

        if dc.dd_gain_mode == "continuous":
            phys_gains = compute_dd_phys_gain_batch(
                np.fromiter(
                    (float(item[0].tau) for item in valid_entries),
                    dtype=np.float64,
                    count=len(valid_entries),
                ),
                np.fromiter(
                    (float(item[0].nu) for item in valid_entries),
                    dtype=np.float64,
                    count=len(valid_entries),
                ),
                dc.delta_f, dc.T_sym, dc.M, dc.N,
            )
            for (entry, i, j, q), phys_gain in zip(valid_entries, phys_gains):
                # Post-G2 physics (audit advice/001 section 5): coefficient
                # must carry ``I_support * |A|^2`` (continuous gain), not the
                # binary ``1[g_dd >= g_min]`` support gate, so deflection is
                # scaled continuously off-bin and out-of-support entries are
                # exactly zero.  The gain is reconstructed from the same
                # (tau, nu) entry observables (power-independent), matching
                # the realized d_eff = chi_rep * d_raw * I_support * |A|^2.
                if phys_gain > 0.0:
                    coefficient[i, j, q] = (
                        float(entry.chi_rep) * float(entry.alpha) ** 2 * scale
                        * float(phys_gain)
                    )
        else:  # legacy binary gate
            for entry, i, j, q in valid_entries:
                if float(entry.g_dd) >= float(dc.g_min):
                    coefficient[i, j, q] = (
                        float(entry.chi_rep) * float(entry.alpha) ** 2 * scale
                    )
        return coefficient

    def _per_watt_coefficient_from_dense_deflection(
        self, dense,
    ) -> np.ndarray:
        """Reconstruct per-watt gain from tensors in legacy arithmetic order.

        The LP may have a degenerate optimal face, so even harmless ULP-level
        changes can select a different optimizer and alter the closed-loop
        trajectory.  This method deliberately mirrors
        :meth:`_per_watt_coefficient_from_entries` field extraction and scalar
        multiplication order while avoiding temporary ``DeflectionEntry``
        allocation.
        """
        dc = self.deflection_computer
        scale = float(
            dc.c_det * dc.M * dc.N * dc.antenna_gain * dc.n_cpi
            / max(dc.noise_power, 1.0e-15)
        )
        coefficient = np.zeros((self.K, self.K, self.Q), dtype=np.float64)
        valid = dense.valid
        alpha = dense.alpha[valid]
        if dc.dd_gain_mode == "continuous":
            phys_gains = compute_dd_phys_gain_batch(
                dense.tau[valid],
                dense.nu[valid],
                dc.delta_f, dc.T_sym, dc.M, dc.N,
            )
            # ``fromiter`` retains the legacy scalar arithmetic order exactly;
            # boolean scatter then restores the dense i/j/q layout.
            coefficient[valid] = np.fromiter((
                float(path_gain) ** 2 * scale * float(phys_gain)
                if phys_gain > 0.0 else 0.0
                for path_gain, phys_gain in zip(alpha, phys_gains)
            ), dtype=np.float64, count=alpha.size)
        else:
            g_dd = dense.g_dd[valid]
            coefficient[valid] = np.fromiter((
                float(path_gain) ** 2 * scale
                if float(gain) >= float(dc.g_min) else 0.0
                for path_gain, gain in zip(alpha, g_dd)
            ), dtype=np.float64, count=alpha.size)
        return coefficient

    def _maxmin_dual_reward_gain(
        self,
        deflection_entries: list,
        selected_set: list,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Fixed-owner per-watt gain matrix and sensing budget for the reward.

        Reconstructs the realized per-watt gain ``d_eff / P_sense`` for every
        excited edge and collapses it to the fixed-owner transmitter--target
        gain used by the max-min power LP.  Unexcited edges (zero sensing
        power) contribute zero current marginal value, which is the correct
        shadow-price convention for a same-frame reward (not a counterfactual
        claim).  Returns ``(None, budget)`` when the current selection leaves a
        target without an owner so the reward degrades to the concave utility.
        """
        K, Q = self.K, self.Q
        power = np.asarray(self._current_sensing_power_w, dtype=np.float64)
        if power.shape != (K, Q):
            power = np.zeros((K, Q), dtype=np.float64)
        coefficient = np.zeros((K, K, Q), dtype=np.float64)
        for entry in deflection_entries:
            i, j, q = int(entry.i), int(entry.j), int(entry.q)
            if (
                i == j or not (0 <= i < K and 0 <= j < K and 0 <= q < Q)
                or float(entry.d_eff) <= 0.0
                or power[i, q] <= 1.0e-12
            ):
                continue
            coefficient[i, j, q] = float(entry.d_eff) / power[i, q]
        budget = np.sum(power, axis=1)
        try:
            gain, _ = fixed_owner_gain_matrix(
                coefficient, [tuple(edge) for edge in selected_set])
        except (
            IncompleteFixedOwnerStructureError,
            NonUniqueFixedOwnerStructureError,
        ):
            return None, budget
        except ValueError as error:
            raise RuntimeError(
                "invalid fixed-owner state for max-min reward"
            ) from error
        return gain, budget

    def _sensing_slot_duration_s(self) -> float:
        """Return the sensing clock duration used for battery billing.

        Post-G2 time-energy closure (audit advice/001 section 3, 2026-08-26):
        the deflection path already charges one OTFS CPI of observation time
        ``T_sense = n_cpi * N * T_sym`` per control step, but the battery path
        historically billed ``P_sense * dt`` ( about 97x longer at the canonical
        numerology).  ``cpi_frame`` closes the two clocks:

            T_slot = T_sense + T_comm + T_idle  (canonical, non-overlapping)

        with ``E_sense = P_sense * T_sense``.  ``dt_frame`` retains the legacy
        ``P_sense * dt`` convention so pre-G2 certified runs stay reproducible.
        """
        mode = str(getattr(self.cfg.scenario, 'sensing_energy_mode', 'dt_frame'))
        if mode == 'cpi_frame':
            ot = self.cfg.otfs
            return float(
                int(getattr(ot, 'n_cpi', 1))
                * int(ot.N) * float(ot.T_sym))
        return float(self.cfg.scenario.dt)

    def _new_distributed_primal_dual_controller(self):
        """Construct the bounded power controller from the live config.

        Keeping construction in one helper makes environment state restore
        use exactly the same numerical tolerances as the normal frame path.
        """
        from uav_isac.optimization import BoundedTemporalPowerController

        ma = self.cfg.marl
        return BoundedTemporalPowerController(
            maximum_iterations_per_frame=int(getattr(
                ma, 'distributed_primal_dual_power_rounds', 20)),
            acceptance_tolerance=float(getattr(
                ma, 'distributed_primal_dual_power_acceptance_tolerance',
                1.0e-10)),
            solver_options={
                'power_cost_per_watt': float(getattr(
                    ma, 'distributed_primal_dual_power_cost_per_watt',
                    1.0e-4)),
                'quadratic_regularization': float(getattr(
                    ma, 'distributed_primal_dual_power_regularization',
                    1.0e-3)),
                'proximal_regularization': float(getattr(
                    ma,
                    'distributed_primal_dual_power_actor_proximal_'
                    'regularization',
                    0.0)),
            },
            step_options={
                'primal_tolerance': float(getattr(
                    ma, 'distributed_primal_dual_power_primal_tolerance',
                    2.0e-4)),
                'stationarity_tolerance': float(getattr(
                    ma, 'distributed_primal_dual_power_stationarity_tolerance',
                    2.0e-4)),
                'dual_tolerance': float(getattr(
                    ma, 'distributed_primal_dual_power_dual_tolerance',
                    2.0e-4)),
                'convergence_patience': int(getattr(
                    ma, 'distributed_primal_dual_power_convergence_patience',
                    3)),
            },
        )

    def _finalize_analytical_power_accounting(
        self,
        sensing_budget_w: np.ndarray,
    ) -> None:
        """Close RF-cap diagnostics and energy after an analytical override.

        ``sensing_budget_w`` is the residual per-UAV cap after communication,
        not a required equality.  Ordinary monotone max-min happens to admit a
        saturated optimum; covertness/exposure constraints may require genuine
        under-use.  This method therefore reports violation and unused power
        separately and charges only the power that is actually executed.
        """
        budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
        used = np.sum(np.asarray(
            self._current_sensing_power_w, dtype=np.float64), axis=1)
        if budget.shape != (self.K,) or used.shape != (self.K,):
            raise ValueError("analytical sensing budget/accounting shape mismatch")
        violation = np.maximum(used - budget, 0.0)
        unused = np.maximum(budget - used, 0.0)
        self._last_analytical_power_balance_error = float(np.max(
            np.abs(used - budget)))
        self._last_analytical_power_budget_violation_w = float(np.max(violation))
        self._last_analytical_unused_power_w = float(np.sum(unused))
        if self._last_analytical_power_budget_violation_w > 1.0e-9:
            raise AssertionError("analytical sensing allocation exceeds RF cap")

        before = getattr(self, '_isac_sensing_battery_before', None)
        if before is not None:
            before = np.asarray(before, dtype=np.float64).reshape(-1)
            if before.shape == (self.K,):
                for k, uav in enumerate(self.uavs):
                    uav.battery = max(
                        0.0, float(before[k] - used[k] * self._sensing_slot_duration_s()))

        allocated = self._current_comm_power_w + used
        self._last_isac_metrics.update({
            'isac_comm_power_w': float(np.sum(self._current_comm_power_w)),
            'isac_sensing_power_w': float(np.sum(used)),
            # Deprecated compatibility metric: absolute slack or overflow.
            'isac_max_power_balance_error_w': float(np.max(np.abs(
                allocated - self._isac_total_power_w))),
            'isac_max_power_budget_violation_w': float(np.max(np.maximum(
                allocated - self._isac_total_power_w, 0.0))),
            'isac_unused_power_w': float(np.sum(np.maximum(
                self._isac_total_power_w - allocated, 0.0))),
            'isac_per_uav_comm_power_w': self._current_comm_power_w.copy(),
            'isac_per_uav_sensing_power_w': used.copy(),
            'isac_target_power_w': np.sum(
                self._current_sensing_power_w, axis=0),
        })

    def _initial_analytical_state(
        self,
    ) -> tuple[object | None, tuple]:
        """Frame-0 warm-start state (D1.8, advice 013).

        Computes the initial deflection entries at unit sensing power and a
        minimal feasible single-owner structure (each target owned by its
        nearest UAV, TX = the farthest UAV for a bistatic baseline) so the
        analytical L3 hook can move on frame 0 instead of waiting for the
        first resolved structure.  Returns ``(entries, selected)`` or
        ``(None, ())`` when the geometry cannot form a structure (K < 2).
        """
        if self.K < 2:
            return None, ()
        try:
            uav_p = np.array([u.pos.copy() for u in self.uavs])  # (K, 3)
            uav_v = np.array([u.vel.copy() for u in self.uavs])  # (K, 3)
            if self._distributed_no_truth_fail_closed:
                # Strict no-truth frame-0 warm start (audit advice/001 P0,
                # 2026-08-26): the L3 warm start consumes the public median
                # local-belief map (zero z, belief velocity), never truth.
                pos_xy, vel_xy = self._strict_no_truth_target_map()
                tgt_p = np.column_stack([
                    pos_xy, np.zeros(self.Q, dtype=np.float64),
                ])
                tgt_v = np.column_stack([
                    vel_xy, np.zeros(self.Q, dtype=np.float64),
                ])
            else:
                tgt_p = np.array([
                    t.get_position_3d() for t in self.targets
                ])
                tgt_v = np.array([
                    np.array([*t.get_velocity(), 0.0])
                    for t in self.targets
                ])
            roles0 = np.full(self.K, 2, dtype=int)  # idle placeholder
            entries = self.deflection_computer.compute(
                uav_p, uav_v, tgt_p, tgt_v, roles0, self.fc_position,
                role_agnostic=True,
                sensing_power_w=(
                    np.ones((self.K, self.Q), dtype=np.float64)
                    if self._joint_isac_power_enabled else None),
            )
            d = np.linalg.norm(uav_p[:, None, :2] - tgt_p[None, :, :2],
                               axis=2)  # (K, Q)
            selected = []
            for q in range(self.Q):
                jq = int(np.argmin(d[:, q]))
                iq = int(np.argmax(d[:, q]))
                if iq == jq:
                    iq = (iq + 1) % self.K
                selected.append((iq, jq, q))
            self._last_deflection_entries = entries
            self._last_selected_set = tuple(selected)
            return entries, tuple(tuple(int(v) for v in e)
                                  for e in selected)
        except Exception as error:
            # K < 2 is the only legitimate no-structure condition and is
            # handled above.  Treating a physics/configuration defect as the
            # same empty warm start silently disables the analytical L3 path
            # and can yield a plausible-looking but semantically different
            # run.  Keep the causal chain and fail loudly instead.
            raise RuntimeError(
                "failed to construct the frame-0 analytical state"
            ) from error

    def _distributed_id_movement_delta(self) -> dict:
        """Move each UAV toward its ID-reserved target using local belief.

        UAV ``k`` reads only belief ``(k, k mod Q)``. With ``K == Q`` this
        provides one persistent geometric custodian per target without a
        centralized assignment or access to target truth.
        """
        if self.belief_mgr is None or self.Q <= 0:
            return {}
        out = {}
        standoff = float(self._distributed_id_movement_standoff_m)
        for k in range(self.K):
            q = int(k % self.Q)
            belief_xy = np.asarray(
                self.belief_mgr.get_belief(k, q).mean[:2],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(belief_xy)):
                continue
            delta = belief_xy - np.asarray(self.uavs[k].pos[:2])
            distance = float(np.linalg.norm(delta))
            if distance <= standoff + 1.0e-12:
                continue
            step = min(
                float(self.uavs[k].v_max * self.uavs[k].dt),
                distance - standoff,
            )
            if step > 0.0:
                out[k] = step * delta / max(distance, 1.0e-12)
        return out

    def _distributed_tx_role_mask(self) -> np.ndarray:
        """Return the public, deterministic Tx/Rx partition used by L2/L3."""
        mask = np.zeros(self.K, dtype=bool)
        if self._hyperedge_reserved_tx_nodes:
            mask[np.asarray(
                self._hyperedge_reserved_tx_nodes,
                dtype=np.int64,
            )] = True
        else:
            mask = np.arange(self.K, dtype=np.int64) % 2 == 0
        return mask

    def _distributed_public_desired_movement(
        self,
        public_xy: np.ndarray,
        target_xy: np.ndarray,
        assignment_vector: np.ndarray,
        movement_step: float,
    ) -> Tuple[np.ndarray, Tuple[str, ...]]:
        """Build one whole-view desired action before the safety projection.

        With AO disabled this is exactly the legacy radial standoff rule.  With
        AO enabled, each row uses its assigned target and nearest public
        opposite-role complement.  Hence independently reconstructed complete
        views agree without reading a global quality/certificate vector.
        """
        desired = np.zeros((self.K, 2), dtype=np.float64)
        states = ['unassigned_hold'] * self.K
        standoff = float(self._distributed_id_movement_standoff_m)
        if self._distributed_gain_scheduled_movement_enabled:
            far_gate = self._distributed_gain_scheduled_far_range_gate_m
            if far_gate > 0.0:
                horizontal = np.linalg.norm(
                    public_xy[:, None, :] - target_xy[None, :, :],
                    axis=-1,
                )
                tx_role = self._distributed_tx_role_mask()
                target_has_near_tx = np.min(
                    horizontal[tx_role], axis=0) <= far_gate
                target_has_near_rx = np.min(
                    horizontal[~tx_role], axis=0) <= far_gate
                far_range_phase = bool(np.any(
                    ~(target_has_near_tx & target_has_near_rx)))
                if far_range_phase:
                    for node, target in enumerate(assignment_vector):
                        target = int(target)
                        if not 0 <= target < self.Q:
                            continue
                        node_delta = target_xy[target] - public_xy[node]
                        node_distance = float(np.linalg.norm(node_delta))
                        node_step = min(
                            movement_step,
                            max(node_distance - standoff, 0.0),
                        )
                        if node_step > 0.0:
                            desired[node] = (
                                node_step * node_delta
                                / max(node_distance, 1.0e-12))
                            states[node] = 'gain_far_range'
                        else:
                            states[node] = 'gain_far_range_hold'
                    return desired, tuple(states)
            geometry_phase = bool(
                self.t
                % self._distributed_gain_scheduled_period_frames
                == 0)
            if not geometry_phase:
                return desired, tuple(
                    'gain_strategy_hold' for _ in range(self.K))
            desired, winners, _gain, _before, _after = (
                gauss_southwell_bistatic_geometry_sweep(
                    public_xy,
                    target_xy,
                    self._distributed_tx_role_mask(),
                    height_m=float(self.cfg.scenario.height),
                    maximum_step_m=movement_step,
                    standoff_m=standoff,
                    minimum_log_improvement=(
                        self._distributed_gain_scheduled_min_log_improvement),
                    maximum_selected_nodes=(
                        self._distributed_gain_scheduled_max_selected_nodes),
                ))
            if not winners:
                return desired, tuple(
                    'gain_no_improvement' for _ in range(self.K))
            states = ['gain_not_selected'] * self.K
            for winner in winners:
                states[int(winner[0])] = 'gain_selected'
            return desired, tuple(states)
        if not self._distributed_alternating_optimization_enabled:
            for node, target in enumerate(assignment_vector):
                target = int(target)
                if not 0 <= target < self.Q:
                    continue
                node_delta = target_xy[target] - public_xy[node]
                node_distance = float(np.linalg.norm(node_delta))
                node_step = min(
                    movement_step,
                    max(node_distance - standoff, 0.0),
                )
                if node_step > 0.0:
                    desired[node] = (
                        node_step * node_delta
                        / max(node_distance, 1.0e-12))
                    states[node] = 'legacy_radial'
                else:
                    states[node] = 'legacy_hold'
            return desired, tuple(states)

        cycle = (
            self._distributed_ao_range_frames
            + self._distributed_ao_strategy_frames
        )
        cycle_frame = int(self.t % cycle)
        phase = (
            'range'
            if cycle_frame < self._distributed_ao_range_frames
            else 'strategy'
        )
        tx_role_mask = self._distributed_tx_role_mask()
        for node, target in enumerate(assignment_vector):
            target = int(target)
            if not 0 <= target < self.Q:
                continue
            complements = np.flatnonzero(tx_role_mask != tx_role_mask[node])
            if complements.size == 0:
                states[node] = 'strategy_hold'
                continue
            complement_distance = np.linalg.norm(
                public_xy[complements] - target_xy[target], axis=1)
            complement = int(complements[int(np.argmin(complement_distance))])
            delta, state = annular_alternating_movement_delta(
                public_xy[node],
                target_xy[target],
                public_xy[complement],
                phase=phase,
                inner_radius_m=self._distributed_ao_inner_radius_m,
                outer_radius_m=self._distributed_ao_outer_radius_m,
                maximum_step_m=movement_step,
                desired_bistatic_angle_deg=(
                    self._distributed_ao_desired_bistatic_angle_deg),
                orientation_sign=(1 if (node + target) % 2 == 0 else -1),
            )
            desired[node] = delta
            states[node] = state
        return desired, tuple(states)

    def _role_capacity_desired_movement(
        self,
        public_xy: np.ndarray,
        target_xy: np.ndarray,
        tx_resp: np.ndarray,
        rx_resp: np.ndarray,
        movement_step: float,
        strategy_phase: bool,
        standoff: float,
    ) -> Tuple[np.ndarray, Tuple[str, ...], np.ndarray]:
        """Whole-view desired action for role-capacity responsibilities.

        In the strategy phase every node holds geometry while the P0 layer
        refreshes hyperedges and sensing power.  In the range phase each node
        approaches the responsibility target with the largest current own
        range (worst-first): with a bounded per-episode travel budget this is
        a greedy descent on the per-target bottleneck range, and the approach
        alternates between the two responsibilities as their ranges change.
        The travelled distance is clamped so the node never enters the
        standoff disk of any responsibility target.  Nodes without a
        responsibility hold.
        """
        desired = np.zeros((self.K, 2), dtype=np.float64)
        states = ['rolecap_no_responsibility'] * self.K
        chosen_targets = np.full(self.K, -1, dtype=np.int64)
        if strategy_phase:
            return (
                desired,
                tuple('rolecap_strategy_hold' for _ in range(self.K)),
                chosen_targets,
            )
        for node in range(self.K):
            resp = [int(q) for q in range(self.Q)
                    if tx_resp[node, q] or rx_resp[node, q]]
            if not resp:
                continue
            if len(resp) == 1:
                target = resp[0]
            else:
                # Worst-first: approach the responsibility target whose own
                # range currently dominates the local bottleneck.
                own_distance = np.linalg.norm(
                    public_xy[node, None, :]
                    - target_xy[np.asarray(resp, dtype=np.int64)],
                    axis=1,
                )
                target = resp[int(np.argmax(own_distance))]
            delta = target_xy[target] - public_xy[node]
            distance = float(np.linalg.norm(delta))
            if distance <= 1.0e-12:
                states[node] = 'rolecap_strategy_hold'
                continue
            unit = delta / distance
            travel = min(float(movement_step), distance)
            for q in resp:
                dq = float(np.linalg.norm(public_xy[node] - target_xy[q]))
                if dq <= standoff + 1.0e-12:
                    travel = 0.0
                    states[node] = 'rolecap_standoff_hold'
                    break
                # t^2 + 2 b t + (dq^2 - standoff^2) = 0 with b = unit.(pos - z);
                # the forward root t* = -b - sqrt(b^2 - dq^2 + standoff^2)
                # is the travel that first reaches the standoff disk.
                b = float(unit @ (public_xy[node] - target_xy[q]))
                discriminant = b * b - dq * dq + standoff * standoff
                if b < 0.0 and discriminant > 0.0:
                    t_hit = -b - float(np.sqrt(discriminant))
                    travel = min(travel, max(t_hit, 0.0))
            if travel <= 1.0e-12:
                states[node] = 'rolecap_standoff_hold'
                continue
            desired[node] = travel * unit
            states[node] = 'rolecap_radial'
            chosen_targets[node] = int(target)
        return desired, tuple(states), chosen_targets

    def _distributed_gap_candidate_gain_bounds(
        self,
        viewer: int,
        public_xy: np.ndarray,
        public_velocity_xy: np.ndarray,
        public_age_frames: np.ndarray,
        target_position_3d: np.ndarray,
        target_velocity_3d: np.ndarray,
        candidate_delta_xy: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Reconstruct one viewer's robust moved-geometry gain sandwich.

        All inputs are either local proprioception/belief or physically
        delivered state.  The method adds no message field and deliberately
        returns a *local-view* certificate; independently stitched fleet rows
        are not promoted to a centralized performance theorem.
        """
        if (
            self._hyperedge_pair_score_mode != 'budget_reconstructable'
            or not self._hyperedge_selected_set
            or self._hyperedge_protocol_dim < 4
        ):
            return None
        xy = np.asarray(public_xy, dtype=np.float64)
        velocity = np.asarray(public_velocity_xy, dtype=np.float64)
        age = np.asarray(public_age_frames, dtype=np.float64).reshape(-1)
        target_position = np.asarray(target_position_3d, dtype=np.float64)
        target_velocity = np.asarray(target_velocity_3d, dtype=np.float64)
        delta = np.asarray(candidate_delta_xy, dtype=np.float64)
        if (
            xy.shape != (self.K, 2)
            or velocity.shape != (self.K, 2)
            or age.shape != (self.K,)
            or target_position.shape != (self.Q, 3)
            or target_velocity.shape != (self.Q, 3)
            or delta.shape != (self.K, 2)
            or any(np.any(~np.isfinite(value)) for value in (
                xy, velocity, age, target_position, target_velocity, delta))
            or np.any(age < 0.0)
        ):
            return None

        dt = float(self.cfg.scenario.dt)
        estimated_xy = xy + velocity * age[:, None] * dt
        # Own proprioception is exact and legal to this viewer.  Peer state is
        # dead-reckoned only from delivered velocity and the local frame clock.
        estimated_xy[int(viewer)] = self.uavs[int(viewer)].pos[:2]
        estimated_xy = np.clip(
            estimated_xy,
            np.zeros(2, dtype=np.float64),
            np.asarray(self.area_size, dtype=np.float64),
        )
        moved_xy = np.clip(
            estimated_xy + delta,
            np.zeros(2, dtype=np.float64),
            np.asarray(self.area_size, dtype=np.float64),
        )
        moved_velocity = delta / max(dt, 1.0e-12)
        position_by_target = np.broadcast_to(
            moved_xy[:, None, :], (self.K, self.Q, 2)).copy()
        velocity_by_target = np.broadcast_to(
            moved_velocity[:, None, :], (self.K, self.Q, 2)).copy()
        visible = np.ones((self.K, self.Q), dtype=bool)

        # Predict the state at which sensing is evaluated later in this frame.
        predicted_target = target_position.copy()
        predicted_target[:, :2] += target_velocity[:, :2] * dt

        position_bits = min(self._hyperedge_state_field_bits[:2])
        velocity_bits = min(self._hyperedge_state_field_bits[2:4])
        position_uncertainty = (
            0.5 * np.hypot(
                float(self.area_size[0]) / max(2 ** position_bits - 1, 1),
                float(self.area_size[1]) / max(2 ** position_bits - 1, 1),
            )
            if self._hyperedge_robust_quantization_enabled else 0.0
        )
        velocity_uncertainty = (
            np.sqrt(2.0) * float(self.cfg.uav.v_max)
            / max(2 ** velocity_bits - 1, 1)
            if self._hyperedge_robust_quantization_enabled else 0.0
        )
        endpoint_position_uncertainty = np.full(
            self.K, float(position_uncertainty), dtype=np.float64)
        endpoint_position_uncertainty += (
            2.0 * age * float(self.cfg.uav.v_max) * dt)
        endpoint_position_uncertainty[int(viewer)] = 0.0
        endpoint_velocity_uncertainty = np.full(
            self.K, float(velocity_uncertainty), dtype=np.float64)
        endpoint_velocity_uncertainty[int(viewer)] = 0.0

        target_position_uncertainty = np.zeros(self.Q, dtype=np.float64)
        target_velocity_uncertainty = np.zeros(self.Q, dtype=np.float64)
        sigma = float(self._distributed_target_position_uncertainty_sigma)
        if sigma > 0.0 and self.belief_mgr is not None:
            covariance = np.asarray(
                self.belief_mgr.cov[int(viewer)], dtype=np.float64)
            position_covariance = 0.5 * (
                covariance[:, :2, :2]
                + np.swapaxes(covariance[:, :2, :2], -1, -2))
            velocity_covariance = 0.5 * (
                covariance[:, 2:4, 2:4]
                + np.swapaxes(covariance[:, 2:4, 2:4], -1, -2))
            target_position_uncertainty = sigma * np.sqrt(np.maximum(
                symmetric_2x2_max_eigenvalue(position_covariance), 0.0))
            target_velocity_uncertainty = sigma * np.sqrt(np.maximum(
                symmetric_2x2_max_eigenvalue(velocity_covariance), 0.0))
            target_position_uncertainty += (
                dt * target_velocity_uncertainty)

        dc = self.deflection_computer
        coefficient_scale = float(
            dc.c_det * dc.M * dc.N * dc.antenna_gain * dc.n_cpi
            / max(dc.noise_power, 1.0e-15))
        try:
            lower_coefficient = (
                self._hyperedge_acceleration.reconstruct_dense(
                    position_by_target,
                    velocity_by_target,
                    predicted_target,
                    target_velocity,
                    visible,
                    uav_height_m=float(self.cfg.scenario.height),
                    fc_hz=float(dc.fc),
                    rcs_m2=float(dc.rcs),
                    delta_f_hz=float(dc.delta_f),
                    symbol_period_s=float(dc.T_sym),
                    delay_bins=int(dc.M),
                    doppler_bins=int(dc.N),
                    dd_gate_min=float(dc.g_min),
                    coefficient_scale=coefficient_scale,
                    position_uncertainty_m=endpoint_position_uncertainty,
                    target_position_uncertainty_m=(
                        target_position_uncertainty),
                    velocity_uncertainty_mps=(
                        endpoint_velocity_uncertainty),
                    target_velocity_uncertainty_mps=(
                        target_velocity_uncertainty),
                    dd_gain_mode=str(dc.dd_gain_mode),
                    robust_dd_uncertainty=True,
                ))
            upper_coefficient = (
                self._hyperedge_acceleration.reconstruct_upper(
                    position_by_target,
                    predicted_target,
                    visible,
                    uav_height_m=float(self.cfg.scenario.height),
                    fc_hz=float(dc.fc),
                    rcs_m2=float(dc.rcs),
                    coefficient_scale=coefficient_scale,
                    position_uncertainty_m=endpoint_position_uncertainty,
                    target_position_uncertainty_m=(
                        target_position_uncertainty),
                ))
        except ValueError:
            # A viewer can legitimately lack enough public state to construct
            # a conservative candidate certificate.  That candidate then
            # fails closed without borrowing simulator truth.
            return None
        try:
            lower_gain, owner = fixed_owner_gain_matrix(
                lower_coefficient, self._hyperedge_selected_set)
            upper_gain, _ = fixed_owner_gain_matrix(
                upper_coefficient, self._hyperedge_selected_set)
        except (
            IncompleteFixedOwnerStructureError,
            NonUniqueFixedOwnerStructureError,
        ):
            return None
        except ValueError as error:
            raise RuntimeError(
                "invalid fixed-owner structure in movement certificate"
            ) from error
        return lower_gain, upper_gain, owner

    def _distributed_gap_coverage_movement_delta(self) -> dict:
        """Gap-coverage movement (three-phase policy, Phase A + B safety net).

        Geometric objective: place at least one Tx endpoint and one Rx endpoint
        inside ``distributed_gap_critical_radius_m`` for every target.  This
        design radius is not a universal P_D threshold because power, DD
        support and fusion remain relevant.  Each viewer reconstructs the
        public view and assigns a recovery responsibility to every missing
        role.  Responsibilities are re-evaluated every frame, so recovery
        pursuit starts immediately instead of waiting for the legacy
        150-frame hold; physical coverage itself still takes finite travel
        time.

        Pursuit (fast convergence): each node approaches the responsibility
        target with the largest remaining gap ``gap = max(0, R - R_crit)`` at
        full speed -- a greedy descent on the P_D-gate margin -- clamped to
        the target standoff disks (safety invariant S1).  A node whose
        responsibilities are all inside the critical radius holds its range
        (near-field placeholder until the Phase C tangential steps).
        """
        from uav_isac.coordination.hyperedge import (
            deterministic_bottleneck_cost_assignment,
            gap_coverage_bottleneck_assignment,
            is_pairwise_safe_movement,
            project_pairwise_safe_movement,
            role_aware_bistatic_movement_cost,
        )
        from uav_isac.physical.movement_potential import (
            coverage_deficit_lexicographic_score,
            role_aware_bistatic_products,
            select_baseline_enveloped_movement,
        )

        safety_diagnostics = []
        stale_fail_closed_viewers = 0
        assignment_infeasible_viewers = 0
        public_view_max_ages = []
        viewer_states = []
        uncovered_counts = []
        envelope_gap_selected = []
        envelope_single_projection = []
        envelope_gap_noop = []
        envelope_baseline_noop = []
        primal_dual_available = []
        primal_dual_selected = []
        primal_dual_strong_dominance = []
        primal_dual_lower_improvement = []
        primal_dual_certificate_width = []
        primal_dual_incumbent_lower = []
        primal_dual_candidate_lower = []
        if self.Q <= 0:
            return {}
        out = {}
        r_crit = max(1.0, float(self._distributed_gap_critical_radius_m))
        cap = max(1, int(self._distributed_gap_capacity))
        standoff = float(self._distributed_gap_standoff_m)
        movement_step = float(
            self.uavs[0].v_max * self.uavs[0].dt)
        role_mask = self._distributed_tx_role_mask()
        for viewer in range(self.K):
            target_position_3d, target_velocity_3d = (
                self._movement_target_state_for_viewer(viewer))
            target_xy = target_position_3d[:, :2]
            public_xy = np.zeros((self.K, 2), dtype=np.float64)
            public_velocity_xy = np.zeros((self.K, 2), dtype=np.float64)
            public_xy[viewer, 0] = (
                self._hyperedge_local_offer[viewer, 0, 0]
                * float(self.area_size[0]))
            public_xy[viewer, 1] = (
                self._hyperedge_local_offer[viewer, 0, 1]
                * float(self.area_size[1]))
            if self._hyperedge_protocol_dim >= 4:
                public_velocity_xy[viewer, 0] = (
                    (2.0 * self._hyperedge_local_offer[viewer, 0, 2] - 1.0)
                    * float(self.cfg.uav.v_max))
                public_velocity_xy[viewer, 1] = (
                    (2.0 * self._hyperedge_local_offer[viewer, 0, 3] - 1.0)
                    * float(self.cfg.uav.v_max))
            last_seen = self._hyperedge_received_last_seen[viewer]
            peer_seen = np.all(
                (last_seen > FRAME_NEVER)
                & (
                    self.t - last_seen
                    <= self._distributed_movement_public_max_age_frames
                ),
                axis=1,
            )
            complete = bool(np.sum(peer_seen) >= self.K - 1)
            if not complete:
                stale_fail_closed_viewers += 1
                out[viewer] = np.zeros(2, dtype=np.float64)
                continue
            for node in range(self.K):
                if node == viewer:
                    continue
                public_xy[node, 0] = (
                    self._hyperedge_received_offer[viewer, node, 0, 0]
                    * float(self.area_size[0]))
                public_xy[node, 1] = (
                    self._hyperedge_received_offer[viewer, node, 0, 1]
                    * float(self.area_size[1]))
                if self._hyperedge_protocol_dim >= 4:
                    public_velocity_xy[node, 0] = (
                        (2.0 * self._hyperedge_received_offer[
                            viewer, node, 0, 2] - 1.0)
                        * float(self.cfg.uav.v_max))
                    public_velocity_xy[node, 1] = (
                        (2.0 * self._hyperedge_received_offer[
                            viewer, node, 0, 3] - 1.0)
                        * float(self.cfg.uav.v_max))
            peer_age = np.zeros(self.K, dtype=np.int64)
            for node in range(self.K):
                if node != viewer and peer_seen[node]:
                    peer_age[node] = int(np.max(self.t - last_seen[node]))
            view_max_age = int(np.max(peer_age))
            public_view_max_ages.append(view_max_age)

            local_deficit = None
            if self._distributed_gap_deficit_priority:
                local_pd = np.asarray(
                    self.prev_P_D_local.get(
                        viewer, np.zeros(self.Q, dtype=np.float64)),
                    dtype=np.float64,
                ).reshape(-1)
                if local_pd.shape == (self.Q,):
                    local_deficit = np.clip(1.0 - local_pd, 0.0, 1.0)
            tx_resp, rx_resp, uncovered = gap_coverage_bottleneck_assignment(
                public_xy,
                target_xy,
                role_mask,
                critical_radius_m=r_crit,
                capacity=cap,
                target_deficit=local_deficit,
            )
            self._distributed_gap_tx_resp[viewer] = tx_resp
            self._distributed_gap_rx_resp[viewer] = rx_resp
            uncovered_counts.append(int(np.sum(uncovered)))

            assignment_vector = np.full(self.K, -1, dtype=np.int64)
            # Per-target bistatic bottleneck for the pursuit order:
            #   score_kq = R_kq^2 * min_{opposite role} R_jq^2
            # chasing the largest product lowers the P_D-gate bottleneck
            # fastest (P_D monotone in -R_tx^2 R_rx^2).  All quantities come
            # from the public view.
            horizontal_sq = np.sum(
                (public_xy[:, None, :] - target_xy[None, :, :]) ** 2,
                axis=-1,
            )
            tx_nodes = np.flatnonzero(role_mask)
            rx_nodes = np.flatnonzero(~role_mask)
            physical_range_sq = (
                float(self.cfg.scenario.height) ** 2 + horizontal_sq)
            role_product = role_aware_bistatic_products(
                physical_range_sq,
                role_mask,
            )
            for node in range(self.K):
                node_targets = [
                    int(q) for q in range(self.Q)
                    if tx_resp[node, q] or rx_resp[node, q]]
                if not node_targets:
                    assignment_vector[node] = int(node % self.Q)
                    continue
                product = role_product[node][
                    np.asarray(node_targets, dtype=np.int64)]
                primary = node_targets[int(np.argmax(product))]
                assignment_vector[node] = primary
            self._distributed_movement_target[viewer] = int(
                assignment_vector[viewer])
            self._distributed_movement_local_assignment[viewer] = (
                np.asarray(assignment_vector, dtype=np.int64))
            self._distributed_movement_last_update[viewer] = int(self.t)

            desired = np.zeros((self.K, 2), dtype=np.float64)
            states = ['gap_no_responsibility'] * self.K
            pursuit_radius = max(
                1.0, float(self._distributed_gap_pursuit_radius_m))
            # Near-field focus (Phase B reinforcement), weighted-product form:
            # a target whose equivalent radius R_eq = (min_tx^2 * min_rx^2)^0.25
            # exceeds the pass radius while its near endpoint is within the
            # reachable radius is critical (P_D dragged by the far leg).  Its
            # responsible product is multiplied by ``focus_weight`` in the
            # pursuit order -- a soft bias, not a hard lock: the node still
            # switches to a genuinely worse responsibility, so no other
            # target is preempted (the hard-lock ablation regressed 3
            # saturated seeds).
            focus_mask = np.zeros(self.Q, dtype=bool)
            if self._distributed_gap_nearfield_focus:
                nearfield_radius = max(
                    1.0, float(self._distributed_gap_nearfield_radius_m))
                pass_radius = max(
                    1.0, float(self._distributed_gap_complete_radius_m))
                min_tx_sq = np.min(horizontal_sq[tx_nodes], axis=0)
                min_rx_sq = np.min(horizontal_sq[rx_nodes], axis=0)
                eq_radius = np.power(min_tx_sq * min_rx_sq, 0.25)
                near_endpoint = np.sqrt(np.minimum(min_tx_sq, min_rx_sq))
                focus_mask = (eq_radius > pass_radius) & (
                    near_endpoint <= nearfield_radius)
            focus_weight = max(
                1.0, float(self._distributed_gap_focus_weight))
            for node in range(self.K):
                node_targets = [
                    int(q) for q in range(self.Q)
                    if tx_resp[node, q] or rx_resp[node, q]]
                if not node_targets:
                    continue
                own_range = np.sqrt(
                    horizontal_sq[node][
                        np.asarray(node_targets, dtype=np.int64)])
                if float(np.max(own_range)) <= pursuit_radius + 1.0e-9:
                    states[node] = 'gap_near_hold'
                    continue
                product = role_product[node][
                    np.asarray(node_targets, dtype=np.int64)]
                product_focus = product.copy()
                for idx, q in enumerate(node_targets):
                    if focus_mask[q]:
                        product_focus[idx] *= focus_weight
                target = node_targets[int(np.argmax(product_focus))]
                if focus_mask[target]:
                    states[node] = 'gap_focus'
                delta = target_xy[target] - public_xy[node]
                distance = float(np.linalg.norm(delta))
                if distance <= 1.0e-12:
                    states[node] = 'gap_near_hold'
                    continue
                unit = delta / distance
                travel = min(float(movement_step), distance)
                for q in node_targets:
                    dq = float(np.linalg.norm(
                        public_xy[node] - target_xy[q]))
                    if dq <= standoff + 1.0e-12:
                        travel = 0.0
                        states[node] = 'gap_standoff_hold'
                        break
                    b = float(unit @ (public_xy[node] - target_xy[q]))
                    discriminant = b * b - dq * dq + standoff * standoff
                    if b < 0.0 and discriminant > 0.0:
                        t_hit = -b - float(np.sqrt(discriminant))
                        travel = min(travel, max(t_hit, 0.0))
                if travel <= 1.0e-12:
                    if states[node] == 'gap_no_responsibility':
                        states[node] = 'gap_standoff_hold'
                    if (states[node] == 'gap_standoff_hold'
                            and self._distributed_gap_tangential_phase):
                        # Phase C: radius-preserving tangential step at the
                        # standoff boundary.  The new position lies on the
                        # same circle radius R around the pursuit target, so
                        # the safety invariant S1 (R >= standoff) is
                        # preserved exactly; the deterministic sign
                        # (node+target parity) keeps identical public views
                        # consistent.
                        radial = public_xy[node] - target_xy[target]
                        radius = float(np.linalg.norm(radial))
                        if radius > 1.0e-9:
                            theta = float(np.arctan2(
                                radial[1], radial[0]))
                            sign = 1.0 if (node + target) % 2 == 0 else -1.0
                            max_angle = float(
                                self._distributed_gap_tangential_step_rad)
                            theta_new = theta + sign * min(
                                max_angle, movement_step / radius)
                            new_pos = target_xy[target] + radius * np.asarray(
                                [float(np.cos(theta_new)),
                                 float(np.sin(theta_new))])
                            desired[node] = new_pos - public_xy[node]
                            states[node] = 'gap_tangential'
                    continue
                desired[node] = travel * unit
                states[node] = 'gap_radial'
            viewer_states.append(states[viewer])

            safety_kwargs = {
                'minimum_distance_m': (
                    float(self.cfg.uav.d_safe)
                    + self._distributed_movement_safety_margin_m
                    + self._distributed_movement_safety_margin_per_age_m
                    * view_max_age),
                'maximum_step_m': movement_step,
                'area_size_xy': (
                    float(self.area_size[0]),
                    float(self.area_size[1]),
                ),
                'outside_invariant_recovery': (
                    self._distributed_movement_outside_invariant_recovery_enabled),
                'independently_composable': (
                    self._distributed_movement_independently_composable_safety),
                'analytic_composable_projection': (
                    self._distributed_movement_analytic_composable_projection_enabled),
            }
            if self._distributed_gap_baseline_envelope_enabled:
                baseline_cost = role_aware_bistatic_movement_cost(
                    public_xy,
                    target_xy,
                    role_mask,
                    height_m=float(self.cfg.scenario.height),
                    movement_step_m=movement_step,
                    standoff_m=standoff,
                    complement_exponent=(
                        self._distributed_bistatic_complement_exponent),
                )
                baseline_assignment = deterministic_bottleneck_cost_assignment(
                    baseline_cost)
                baseline_desired, _baseline_states = (
                    self._distributed_public_desired_movement(
                        public_xy,
                        target_xy,
                        baseline_assignment,
                        movement_step,
                    ))
                # Exact lazy projection. A certified candidate is already the
                # output of its safety projection, so solve only uncertified
                # candidates. If both are certified, score first and retain
                # one projector call for uniform safety accounting. This is
                # trajectory-identical to projecting both candidates.
                gap_noop = is_pairwise_safe_movement(
                    public_xy, desired, **safety_kwargs)
                baseline_noop = is_pairwise_safe_movement(
                    public_xy, baseline_desired, **safety_kwargs)
                envelope_gap_noop.append(bool(gap_noop))
                envelope_baseline_noop.append(bool(baseline_noop))
                diagnostics_before = len(safety_diagnostics)
                if gap_noop and baseline_noop:
                    selected_desired, use_gap, _gap_score, _baseline_score = (
                        select_baseline_enveloped_movement(
                            target_xy,
                            public_xy,
                            role_mask,
                            desired,
                            baseline_desired,
                            r_crit,
                            height_m=float(self.cfg.scenario.height),
                        ))
                    selected, safety_info = project_pairwise_safe_movement(
                        public_xy,
                        selected_desired,
                        return_diagnostics=True,
                        recovery_gain=(
                            self._distributed_movement_outside_invariant_recovery_gain),
                        **safety_kwargs,
                    )
                    safety_diagnostics.append(safety_info)
                else:
                    safe = desired
                    if not gap_noop:
                        safe, safety_info = project_pairwise_safe_movement(
                            public_xy,
                            desired,
                            return_diagnostics=True,
                            recovery_gain=(
                                self._distributed_movement_outside_invariant_recovery_gain),
                            **safety_kwargs,
                        )
                        safety_diagnostics.append(safety_info)
                    baseline_safe = baseline_desired
                    if not baseline_noop:
                        baseline_safe, baseline_safety_info = (
                            project_pairwise_safe_movement(
                                public_xy,
                                baseline_desired,
                                return_diagnostics=True,
                                recovery_gain=(
                                    self._distributed_movement_outside_invariant_recovery_gain),
                                **safety_kwargs,
                            ))
                        safety_diagnostics.append(baseline_safety_info)
                    selected, use_gap, _gap_score, _baseline_score = (
                        select_baseline_enveloped_movement(
                            target_xy,
                            public_xy,
                            role_mask,
                            safe,
                            baseline_safe,
                            r_crit,
                            height_m=float(self.cfg.scenario.height),
                        ))
                certificate_available = False
                certificate_selected = False
                certificate_strong = False
                certificate_improvement = 0.0
                certificate_width = 0.0
                if (
                    self._distributed_gap_primal_dual_candidate_enabled
                    and bool(self._distributed_replicated_local_cache_valid[
                        viewer])
                    and self._hyperedge_selected_set
                ):
                    owner = np.full(self.Q, -1, dtype=np.int64)
                    for _tx, receiver, target in self._hyperedge_selected_set:
                        owner[int(target)] = int(receiver)
                    # Prices and the feasible full-power plan are produced by
                    # the deployed public-view LP.  Its geometry gradient must
                    # therefore use that same gain matrix; multiplying those
                    # primal/dual variables by the stricter certificate gain
                    # would mix two different saddle-point problems.  The
                    # conservative/optimistic views remain the acceptance
                    # sandwich below, so a proposal cannot bypass robustness.
                    local_proposal_gain = np.asarray(
                        self._hyperedge_public_gain_views[viewer],
                        dtype=np.float64)
                    local_gain_lower = np.asarray(
                        self._composable_certificate_gain_views[viewer],
                        dtype=np.float64)
                    local_power = np.asarray(
                        self._distributed_replicated_local_power_cache[viewer],
                        dtype=np.float64)
                    local_price = np.maximum(np.asarray(
                        self._distributed_replicated_local_price_cache[viewer],
                        dtype=np.float64), 0.0)
                    price_total = float(np.sum(local_price))
                    if price_total > 0.0:
                        local_price = local_price / price_total
                    if (
                        np.all(owner >= 0)
                        and local_proposal_gain.shape == (self.K, self.Q)
                        and local_gain_lower.shape == (self.K, self.Q)
                        and local_power.shape == (self.K, self.Q)
                        and local_price.shape == (self.Q,)
                        and price_total > 0.0
                        and np.any(local_proposal_gain > 0.0)
                    ):
                        from uav_isac.coordination.capability import (
                            capability_geometry_gradient,
                        )
                        proposal_gradient = capability_geometry_gradient(
                            local_proposal_gain,
                            owner,
                            public_xy,
                            target_xy,
                            local_power,
                            local_price,
                            height_m=float(self.cfg.scenario.height),
                        )
                        dual_desired = np.zeros((self.K, 2), dtype=np.float64)
                        for node in range(self.K):
                            norm = float(np.linalg.norm(
                                proposal_gradient[node]))
                            if norm > 1.0e-12:
                                # Positive prices differentiate max-min
                                # capability, so ascent follows +gradient.
                                dual_desired[node] = (
                                    movement_step
                                    * proposal_gradient[node] / norm)
                        if np.any(dual_desired != 0.0):
                            dual_safe = dual_desired
                            if not is_pairwise_safe_movement(
                                public_xy, dual_desired, **safety_kwargs
                            ):
                                dual_safe, dual_safety_info = (
                                    project_pairwise_safe_movement(
                                        public_xy,
                                        dual_desired,
                                        return_diagnostics=True,
                                        recovery_gain=(
                                            self._distributed_movement_outside_invariant_recovery_gain),
                                        **safety_kwargs,
                                    ))
                                safety_diagnostics.append(dual_safety_info)
                            incumbent_score = (
                                coverage_deficit_lexicographic_score(
                                    target_xy,
                                    public_xy + selected,
                                    role_mask,
                                    r_crit,
                                    height_m=float(self.cfg.scenario.height),
                                ))
                            dual_score = coverage_deficit_lexicographic_score(
                                target_xy,
                                public_xy + dual_safe,
                                role_mask,
                                r_crit,
                                height_m=float(self.cfg.scenario.height),
                            )
                            incumbent_gain = (
                                self._distributed_gap_candidate_gain_bounds(
                                    viewer,
                                    public_xy,
                                    public_velocity_xy,
                                    peer_age,
                                    target_position_3d,
                                    target_velocity_3d,
                                    selected,
                                ))
                            dual_gain = (
                                self._distributed_gap_candidate_gain_bounds(
                                    viewer,
                                    public_xy,
                                    public_velocity_xy,
                                    peer_age,
                                    target_position_3d,
                                    target_velocity_3d,
                                    dual_safe,
                                ))
                            if incumbent_gain is not None and dual_gain is not None:
                                public_budget = np.full(
                                    self.K,
                                    float(self._sensing_power_cap_w),
                                    dtype=np.float64,
                                )
                                try:
                                    incumbent_certificate = (
                                        robust_primal_dual_bounds(
                                            incumbent_gain[0],
                                            incumbent_gain[1],
                                            local_power,
                                            public_budget,
                                            local_price,
                                        ))
                                    dual_certificate = robust_primal_dual_bounds(
                                        dual_gain[0],
                                        dual_gain[1],
                                        local_power,
                                        public_budget,
                                        local_price,
                                    )
                                except ValueError:
                                    incumbent_certificate = None
                                    dual_certificate = None
                                if (
                                    incumbent_certificate is not None
                                    and dual_certificate is not None
                                ):
                                    certificate_available = True
                                    certificate_improvement = float(
                                        dual_certificate.primal_witness_lower
                                        - incumbent_certificate.primal_witness_lower)
                                    primal_dual_incumbent_lower.append(float(
                                        incumbent_certificate.primal_witness_lower))
                                    primal_dual_candidate_lower.append(float(
                                        dual_certificate.primal_witness_lower))
                                    certificate_width = float(
                                        dual_certificate.uncertainty_width)
                                    certificate_strong = robust_movement_dominates(
                                        dual_certificate,
                                        incumbent_certificate,
                                        margin=(
                                            self._distributed_gap_primal_dual_margin),
                                    )
                                    witness_gate = bool(
                                        certificate_improvement
                                        > self._distributed_gap_primal_dual_margin
                                    )
                                    certificate_gate = (
                                        certificate_strong
                                        if self._distributed_gap_primal_dual_strong_only
                                        else witness_gate)
                                    proxy_pareto_non_regression = all(
                                        float(candidate_value)
                                        <= float(incumbent_value)
                                        + 1.0e-12 * max(
                                            1.0,
                                            abs(float(incumbent_value)),
                                        )
                                        for candidate_value, incumbent_value
                                        in zip(dual_score, incumbent_score)
                                    )
                                    if (
                                        proxy_pareto_non_regression
                                        and certificate_gate
                                    ):
                                        selected = dual_safe
                                        use_gap = False
                                        certificate_selected = True
                primal_dual_available.append(bool(certificate_available))
                primal_dual_selected.append(bool(certificate_selected))
                primal_dual_strong_dominance.append(bool(certificate_strong))
                if certificate_available:
                    primal_dual_lower_improvement.append(
                        float(certificate_improvement))
                    primal_dual_certificate_width.append(
                        float(certificate_width))
                envelope_single_projection.append(bool(
                    len(safety_diagnostics) - diagnostics_before == 1))
                envelope_gap_selected.append(bool(use_gap))
                out[viewer] = selected[viewer].copy()
            else:
                safe, safety_info = project_pairwise_safe_movement(
                    public_xy,
                    desired,
                    return_diagnostics=True,
                    recovery_gain=(
                        self._distributed_movement_outside_invariant_recovery_gain),
                    **safety_kwargs,
                )
                safety_diagnostics.append(safety_info)
                out[viewer] = safe[viewer].copy()

        if safety_diagnostics:
            solve_time = float(sum(
                float(item['solve_time_s'])
                for item in safety_diagnostics))
            fail_closed_calls = int(sum(
                bool(item['fail_closed'])
                for item in safety_diagnostics))
            intervention_calls = int(sum(
                bool(item['intervened'])
                for item in safety_diagnostics))
            reduced_linear_calls = int(sum(
                item.get('projection_solver') == 'reduced_linear_qp'
                for item in safety_diagnostics))
            self._last_movement_safety_metrics.update({
                'movement_safety_projection_calls': float(
                    len(safety_diagnostics)),
                'movement_safety_intervened': float(
                    intervention_calls > 0),
                'movement_safety_intervention_call_rate': float(
                    intervention_calls / len(safety_diagnostics)),
                'movement_safety_fail_closed': float(
                    fail_closed_calls > 0),
                'movement_safety_fail_closed_calls': float(
                    fail_closed_calls),
                'movement_safety_solve_time_s': solve_time,
                'movement_safety_solve_time_s_per_node': float(
                    solve_time / max(self.K, 1)),
                'movement_safety_pairwise_constraints_mean': float(np.mean([
                    float(item['pairwise_constraint_count'])
                    for item in safety_diagnostics
                ])),
                'movement_safety_reduced_linear_qp_calls': float(
                    reduced_linear_calls),
                'movement_safety_reduced_linear_qp_rate': float(
                    reduced_linear_calls / len(safety_diagnostics)),
            })
        self._last_movement_safety_metrics.update({
            'movement_public_view_stale_fail_closed_viewers': float(
                stale_fail_closed_viewers),
            'movement_public_view_stale_fail_closed_rate': float(
                stale_fail_closed_viewers / max(self.K, 1)),
            'movement_public_view_mean_max_age_frames': float(np.mean(
                public_view_max_ages or [0.0])),
            'movement_public_view_max_age_frames': float(max(
                public_view_max_ages or [0.0])),
            'movement_gap_uncovered_targets_mean': float(np.mean(
                uncovered_counts or [0.0])),
            'movement_gap_uncovered_rate': float(np.mean(
                [c / max(self.Q, 1) for c in uncovered_counts] or [0.0])),
            'movement_gap_baseline_envelope_enabled': float(
                self._distributed_gap_baseline_envelope_enabled),
            'movement_gap_baseline_envelope_gap_selected_rate': float(np.mean(
                envelope_gap_selected or [0.0])),
            'movement_gap_baseline_envelope_baseline_selected_rate': float(
                1.0 - np.mean(envelope_gap_selected or [0.0])
                if envelope_gap_selected else 0.0),
            'movement_gap_baseline_envelope_single_projection_rate': float(
                np.mean(envelope_single_projection or [0.0])),
            'movement_gap_baseline_envelope_gap_noop_rate': float(
                np.mean(envelope_gap_noop or [0.0])),
            'movement_gap_baseline_envelope_baseline_noop_rate': float(
                np.mean(envelope_baseline_noop or [0.0])),
            # Diagnostic only. Different choices reveal that local-view
            # performance envelopes did not compose into one fleet candidate;
            # independently composable safety remains valid.
            'movement_gap_baseline_envelope_selection_disagreement_rate': (
                float(
                    bool(envelope_gap_selected)
                    and any(envelope_gap_selected)
                    and not all(envelope_gap_selected)
                )),
            'movement_gap_primal_dual_candidate_enabled': float(
                self._distributed_gap_primal_dual_candidate_enabled),
            'movement_gap_primal_dual_certificate_available_rate': float(
                np.mean(primal_dual_available or [0.0])),
            'movement_gap_primal_dual_selected_rate': float(
                np.mean(primal_dual_selected or [0.0])),
            'movement_gap_primal_dual_strong_dominance_rate': float(
                np.mean(primal_dual_strong_dominance or [0.0])),
            'movement_gap_primal_dual_lower_improvement_mean': float(
                np.mean(primal_dual_lower_improvement or [0.0])),
            'movement_gap_primal_dual_certificate_width_mean': float(
                np.mean(primal_dual_certificate_width or [0.0])),
            'movement_gap_primal_dual_incumbent_lower_mean': float(
                np.mean(primal_dual_incumbent_lower or [0.0])),
            'movement_gap_primal_dual_candidate_lower_mean': float(
                np.mean(primal_dual_candidate_lower or [0.0])),
            'movement_gap_primal_dual_selection_disagreement_rate': float(
                bool(primal_dual_selected)
                and any(primal_dual_selected)
                and not all(primal_dual_selected)),
        })
        if viewer_states:
            denominator = float(len(viewer_states))
            self._last_movement_safety_metrics.update({
                'movement_gap_radial_rate': float(
                    viewer_states.count('gap_radial') / denominator),
                'movement_gap_near_hold_rate': float(
                    viewer_states.count('gap_near_hold') / denominator),
                'movement_gap_standoff_hold_rate': float(
                    viewer_states.count('gap_standoff_hold') / denominator),
                'movement_gap_no_responsibility_rate': float(
                    viewer_states.count('gap_no_responsibility')
                    / denominator),
            })
        # Responsibility agreement across viewers (gap-coverage analogue of
        # the legacy local-assignment agreement rate).
        if self.K >= 2:
            tx_agree = np.zeros((self.K, self.Q), dtype=bool)
            rx_agree = np.zeros((self.K, self.Q), dtype=bool)
            for node in range(self.K):
                for target in range(self.Q):
                    tx_entries = self._distributed_gap_tx_resp[:, node, target]
                    rx_entries = self._distributed_gap_rx_resp[:, node, target]
                    tx_agree[node, target] = bool(
                        np.all(tx_entries == tx_entries[0]))
                    rx_agree[node, target] = bool(
                        np.all(rx_entries == rx_entries[0]))
            agreement = float(np.mean(np.concatenate(
                [tx_agree.reshape(-1), rx_agree.reshape(-1)])))
        else:
            agreement = 1.0
        self._last_movement_safety_metrics.update({
            'movement_gap_responsibility_agreement_rate': agreement,
        })
        executed_targets = np.asarray(
            self._distributed_movement_target, dtype=np.int64)
        valid_executed = executed_targets[
            (executed_targets >= 0) & (executed_targets < self.Q)]
        unique_executed = int(np.unique(valid_executed).size)
        valid_views = np.asarray(
            self._distributed_movement_local_assignment,
            dtype=np.int64,
        )
        valid_views = valid_views[np.all(valid_views >= 0, axis=1)]
        entry_agreements = []
        if valid_views.shape[0] > 0:
            for node in range(self.K):
                _, counts = np.unique(
                    valid_views[:, node], return_counts=True)
                entry_agreements.append(
                    float(np.max(counts) / valid_views.shape[0]))
        self._last_movement_safety_metrics.update({
            'movement_executed_assignment_target_coverage': float(
                unique_executed / max(self.Q, 1)),
            'movement_executed_assignment_duplicate_rate': float(
                1.0 - unique_executed / max(valid_executed.size, 1)),
            'movement_local_assignment_entry_agreement_rate': float(np.mean(
                entry_agreements or [0.0])),
        })
        return out

    def _distributed_role_capacity_movement_delta(self) -> dict:
        """Role-capacitated Tx/Rx responsibility movement.

        Each viewer reconstructs the public node view from delivered offers and
        independently solves the two role subproblems of
        ``role_capacity_bottleneck_assignment`` (Tx set and Rx set, per-node
        capacity ``ceil(Q/|K_r|)``).  Every target therefore owns exactly one
        Tx and one Rx responsibility, and no viewer's local matching row is
        stitched into a fleet action.  A hysteresis rule reassigns a viewer
        only when the new worst-case responsibility range improves the
        incumbent by at least ``reassign_threshold`` after at least
        ``hold_frames``, preventing target-motion chatter.

        The whole-view desired action still passes the ``d_safe`` pairwise
        projection before execution (always executed, matching the projected
        public action semantics of the other distributed modes).
        """
        from uav_isac.coordination.hyperedge import (
            project_pairwise_safe_movement,
            role_capacity_bottleneck_assignment,
        )

        safety_diagnostics = []
        stale_fail_closed_viewers = 0
        public_view_max_ages = []
        viewer_states = []
        reassign_count = 0
        if self.Q <= 0:
            return {}
        out = {}
        standoff = float(self._distributed_role_capacity_standoff_m)
        range_frames = max(1, int(self._distributed_role_capacity_range_frames))
        strategy_frames = max(0, int(self._distributed_role_capacity_strategy_frames))
        cycle = range_frames + max(strategy_frames, 1)
        strategy_phase = bool(
            strategy_frames > 0
            and self.t % cycle >= range_frames)
        hold_frames = max(1, int(self._distributed_role_capacity_hold_frames))
        reassign_threshold = max(
            0.0, float(self._distributed_role_capacity_reassign_threshold))
        role_mask = self._distributed_tx_role_mask()
        tx_count = int(np.count_nonzero(role_mask))
        rx_count = int(self.K - tx_count)
        tx_capacity = 1
        rx_capacity = 1
        if tx_count > 0:
            tx_capacity = max(
                1, int(np.ceil(self.Q / float(tx_count))))
        if rx_count > 0:
            rx_capacity = max(
                1, int(np.ceil(self.Q / float(rx_count))))
        movement_step = float(
            self.uavs[0].v_max * self.uavs[0].dt)
        for viewer in range(self.K):
            target_position_3d, _ = (
                self._movement_target_state_for_viewer(viewer))
            target_xy = target_position_3d[:, :2]
            public_xy = np.zeros((self.K, 2), dtype=np.float64)
            public_xy[viewer, 0] = (
                self._hyperedge_local_offer[viewer, 0, 0]
                * float(self.area_size[0]))
            public_xy[viewer, 1] = (
                self._hyperedge_local_offer[viewer, 0, 1]
                * float(self.area_size[1]))
            last_seen = self._hyperedge_received_last_seen[viewer]
            peer_seen = np.all(
                (last_seen > FRAME_NEVER)
                & (
                    self.t - last_seen
                    <= self._distributed_movement_public_max_age_frames
                ),
                axis=1,
            )
            complete = bool(np.sum(peer_seen) >= self.K - 1)
            if not complete:
                # No reconstructed public view means no responsibility set:
                # fail closed to a hold (never release the unverified actor
                # movement), independent of the stale-fail-closed flag.
                stale_fail_closed_viewers += 1
                out[viewer] = np.zeros(2, dtype=np.float64)
                continue
            for node in range(self.K):
                if node == viewer:
                    continue
                public_xy[node, 0] = (
                    self._hyperedge_received_offer[viewer, node, 0, 0]
                    * float(self.area_size[0]))
                public_xy[node, 1] = (
                    self._hyperedge_received_offer[viewer, node, 0, 1]
                    * float(self.area_size[1]))
            peer_age = np.zeros(self.K, dtype=np.int64)
            for node in range(self.K):
                if node != viewer and peer_seen[node]:
                    peer_age[node] = int(np.max(self.t - last_seen[node]))
            view_max_age = int(np.max(peer_age))
            public_view_max_ages.append(view_max_age)

            current_bottleneck = float(
                self._distributed_role_capacity_bottleneck[viewer])
            has_current = bool(np.isfinite(current_bottleneck))
            since_reassign = int(
                self.t - self._distributed_role_capacity_last_reassign[viewer])
            recompute = not has_current
            if (not recompute
                    and since_reassign >= hold_frames):
                new_tx, new_rx, new_cost, new_feasible = (
                    role_capacity_bottleneck_assignment(
                        public_xy,
                        target_xy,
                        role_mask,
                        height_m=float(self.cfg.scenario.height),
                        capacity=(tx_capacity, rx_capacity),
                    ))
                if not new_feasible:
                    assignment_infeasible_viewers += 1
                    self._distributed_role_capacity_tx_resp[viewer].fill(0)
                    self._distributed_role_capacity_rx_resp[viewer].fill(0)
                    self._distributed_role_capacity_bottleneck[viewer] = np.inf
                    self._distributed_movement_target[viewer] = TARGET_INDEX_NONE
                    self._distributed_movement_local_assignment[viewer].fill(-1)
                    viewer_states.append('rolecap_assignment_infeasible')
                    out[viewer] = np.zeros(2, dtype=np.float64)
                    continue
                if (
                    new_cost
                    < current_bottleneck * (1.0 - reassign_threshold)
                ):
                    self._distributed_role_capacity_tx_resp[viewer] = new_tx
                    self._distributed_role_capacity_rx_resp[viewer] = new_rx
                    self._distributed_role_capacity_bottleneck[viewer] = (
                        new_cost)
                    self._distributed_role_capacity_last_reassign[viewer] = (
                        int(self.t))
                    reassign_count += 1
            if recompute:
                new_tx, new_rx, new_cost, new_feasible = (
                    role_capacity_bottleneck_assignment(
                        public_xy,
                        target_xy,
                        role_mask,
                        height_m=float(self.cfg.scenario.height),
                        capacity=(tx_capacity, rx_capacity),
                    ))
                if not new_feasible:
                    assignment_infeasible_viewers += 1
                    self._distributed_role_capacity_tx_resp[viewer].fill(0)
                    self._distributed_role_capacity_rx_resp[viewer].fill(0)
                    self._distributed_role_capacity_bottleneck[viewer] = np.inf
                    self._distributed_movement_target[viewer] = TARGET_INDEX_NONE
                    self._distributed_movement_local_assignment[viewer].fill(-1)
                    viewer_states.append('rolecap_assignment_infeasible')
                    out[viewer] = np.zeros(2, dtype=np.float64)
                    continue
                self._distributed_role_capacity_tx_resp[viewer] = new_tx
                self._distributed_role_capacity_rx_resp[viewer] = new_rx
                self._distributed_role_capacity_bottleneck[viewer] = new_cost
                self._distributed_role_capacity_last_reassign[viewer] = (
                    int(self.t))
                reassign_count += 1

            tx_resp = self._distributed_role_capacity_tx_resp[viewer]
            rx_resp = self._distributed_role_capacity_rx_resp[viewer]
            desired, movement_states, chosen_targets = (
                self._role_capacity_desired_movement(
                public_xy,
                target_xy,
                tx_resp,
                rx_resp,
                movement_step,
                strategy_phase,
                standoff,
            ))
            viewer_states.append(movement_states[viewer])
            safe, safety_info = project_pairwise_safe_movement(
                public_xy,
                desired,
                minimum_distance_m=(
                    float(self.cfg.uav.d_safe)
                    + self._distributed_movement_safety_margin_m
                    + self._distributed_movement_safety_margin_per_age_m
                    * view_max_age),
                maximum_step_m=movement_step,
                area_size_xy=(
                    float(self.area_size[0]),
                    float(self.area_size[1]),
                ),
                return_diagnostics=True,
                outside_invariant_recovery=(
                    self._distributed_movement_outside_invariant_recovery_enabled),
                recovery_gain=(
                    self._distributed_movement_outside_invariant_recovery_gain),
                independently_composable=(
                    self._distributed_movement_independently_composable_safety),
                analytic_composable_projection=(
                    self._distributed_movement_analytic_composable_projection_enabled),
            )
            safety_diagnostics.append(safety_info)
            out[viewer] = safe[viewer].copy()
            executed_target = int(chosen_targets[viewer])
            if float(np.linalg.norm(safe[viewer])) <= 1.0e-12:
                executed_target = TARGET_INDEX_NONE
            self._distributed_movement_target[viewer] = executed_target
            self._distributed_movement_local_assignment[viewer] = (
                np.asarray(chosen_targets, dtype=np.int64))
            self._distributed_movement_last_update[viewer] = int(self.t)

        if safety_diagnostics:
            solve_time = float(sum(
                float(item['solve_time_s'])
                for item in safety_diagnostics))
            fail_closed_calls = int(sum(
                bool(item['fail_closed'])
                for item in safety_diagnostics))
            intervention_calls = int(sum(
                bool(item['intervened'])
                for item in safety_diagnostics))
            self._last_movement_safety_metrics.update({
                'movement_safety_projection_calls': float(
                    len(safety_diagnostics)),
                'movement_safety_intervened': float(
                    intervention_calls > 0),
                'movement_safety_intervention_call_rate': float(
                    intervention_calls / len(safety_diagnostics)),
                'movement_safety_fail_closed': float(
                    fail_closed_calls > 0),
                'movement_safety_fail_closed_calls': float(
                    fail_closed_calls),
                'movement_safety_solve_time_s': solve_time,
                'movement_safety_solve_time_s_per_node': float(
                    solve_time / max(self.K, 1)),
                'movement_safety_pairwise_constraints_mean': float(np.mean([
                    float(item['pairwise_constraint_count'])
                    for item in safety_diagnostics
                ])),
            })
        self._last_movement_safety_metrics.update({
            'movement_public_view_stale_fail_closed_viewers': float(
                stale_fail_closed_viewers),
            'movement_public_view_stale_fail_closed_rate': float(
                stale_fail_closed_viewers / max(self.K, 1)),
            'movement_public_view_mean_max_age_frames': float(np.mean(
                public_view_max_ages or [0.0])),
            'movement_public_view_max_age_frames': float(max(
                public_view_max_ages or [0.0])),
            'movement_role_capacity_reassign_count': float(
                reassign_count),
            'movement_role_capacity_assignment_infeasible_viewers': float(
                assignment_infeasible_viewers),
            'movement_role_capacity_assignment_infeasible_rate': float(
                assignment_infeasible_viewers / max(self.K, 1)),
            'movement_role_capacity_strategy_phase': float(strategy_phase),
        })
        if viewer_states:
            denominator = float(len(viewer_states))
            self._last_movement_safety_metrics.update({
                'movement_role_capacity_radial_rate': float(
                    viewer_states.count('rolecap_radial') / denominator),
                'movement_role_capacity_strategy_hold_rate': float(
                    viewer_states.count('rolecap_strategy_hold')
                    / denominator),
                'movement_role_capacity_standoff_hold_rate': float(
                    viewer_states.count('rolecap_standoff_hold')
                    / denominator),
                'movement_role_capacity_no_responsibility_rate': float(
                    viewer_states.count('rolecap_no_responsibility')
                    / denominator),
            })
        # Responsibility agreement across viewers: the fraction of
        # (node, target) entries whose Tx and Rx responsibility flags agree
        # between the two reconstructed views -- the role-capacity analogue of
        # the legacy local-assignment agreement rate.
        if self.K >= 2:
            tx_agree = np.zeros((self.K, self.Q), dtype=bool)
            rx_agree = np.zeros((self.K, self.Q), dtype=bool)
            for node in range(self.K):
                for target in range(self.Q):
                    tx_entries = self._distributed_role_capacity_tx_resp[
                        :, node, target]
                    rx_entries = self._distributed_role_capacity_rx_resp[
                        :, node, target]
                    tx_agree[node, target] = bool(
                        np.all(tx_entries == tx_entries[0]))
                    rx_agree[node, target] = bool(
                        np.all(rx_entries == rx_entries[0]))
            agreement = float(np.mean(
                np.concatenate([tx_agree.reshape(-1), rx_agree.reshape(-1)])))
        else:
            agreement = 1.0
        self._last_movement_safety_metrics.update({
            'movement_role_capacity_responsibility_agreement_rate': agreement,
        })
        executed_targets = np.asarray(
            self._distributed_movement_target, dtype=np.int64)
        valid_executed = executed_targets[
            (executed_targets >= 0) & (executed_targets < self.Q)]
        unique_executed = int(np.unique(valid_executed).size)
        valid_views = np.asarray(
            self._distributed_movement_local_assignment,
            dtype=np.int64,
        )
        valid_views = valid_views[np.all(valid_views >= 0, axis=1)]
        entry_agreements = []
        if valid_views.shape[0] > 0:
            for node in range(self.K):
                _, counts = np.unique(
                    valid_views[:, node], return_counts=True)
                entry_agreements.append(
                    float(np.max(counts) / valid_views.shape[0]))
        self._last_movement_safety_metrics.update({
            'movement_executed_assignment_target_coverage': float(
                unique_executed / max(self.Q, 1)),
            'movement_executed_assignment_duplicate_rate': float(
                1.0 - unique_executed / max(valid_executed.size, 1)),
            'movement_local_assignment_entry_agreement_rate': float(np.mean(
                entry_agreements or [0.0])),
        })
        return out

    def _distributed_greedy_matching_movement_delta(self) -> dict:
        """Compute a matching from one UAV's delivered public-state cache.

        Each node independently sorts public node-target distances and accepts
        the first pair whose node and target are both unmatched.  Complete
        caches therefore reproduce the same assignment without an assignment
        server.  A node with an incomplete cache fails over to its persistent
        ID target, so packet loss cannot reveal a centralized state table.
        """
        safety_diagnostics = []
        stale_fail_closed_viewers = 0
        public_view_max_ages = []
        ao_viewer_states = []
        ao_cycle = (
            self._distributed_ao_range_frames
            + self._distributed_ao_strategy_frames
        )
        ao_strategy_phase = bool(
            self.t % ao_cycle >= self._distributed_ao_range_frames)
        gain_geometry_phase = bool(
            self.t
            % self._distributed_gain_scheduled_period_frames
            == 0)
        self._last_movement_safety_metrics = {
            'movement_safety_projection_enabled': float(
                self._distributed_movement_safety_projection_enabled),
            'movement_safety_projection_calls': 0.0,
            'movement_safety_intervened': 0.0,
            'movement_safety_intervention_call_rate': 0.0,
            'movement_safety_fail_closed': 0.0,
            'movement_safety_fail_closed_calls': 0.0,
            'movement_safety_fail_closed_outside_invariant_calls': 0.0,
            'movement_safety_outside_invariant_calls': 0.0,
            'movement_safety_recovery_pair_count': 0.0,
            'movement_safety_solve_time_s': 0.0,
            'movement_safety_solve_time_s_per_node': 0.0,
            'movement_safety_pairwise_constraints_mean': 0.0,
            'movement_safety_reduced_linear_qp_calls': 0.0,
            'movement_safety_reduced_linear_qp_rate': 0.0,
            'movement_public_view_stale_fail_closed_viewers': 0.0,
            'movement_public_view_stale_fail_closed_rate': 0.0,
            'movement_executed_assignment_target_coverage': 0.0,
            'movement_executed_assignment_duplicate_rate': 0.0,
            'movement_local_assignment_entry_agreement_rate': 0.0,
            'movement_anchor_target_coverage': 0.0,
            'movement_ao_enabled': float(
                self._distributed_alternating_optimization_enabled),
            'movement_ao_strategy_phase': float(ao_strategy_phase),
            'movement_ao_evaluated_viewers': 0.0,
            'movement_ao_far_recovery_rate': 0.0,
            'movement_ao_near_recovery_rate': 0.0,
            'movement_ao_range_hold_rate': 0.0,
            'movement_ao_strategy_tangent_rate': 0.0,
            'movement_ao_strategy_hold_rate': 0.0,
            'movement_ao_inner_radius_m': float(
                self._distributed_ao_inner_radius_m),
            'movement_ao_outer_radius_m': float(
                self._distributed_ao_outer_radius_m),
            'movement_gain_schedule_enabled': float(
                self._distributed_gain_scheduled_movement_enabled),
            'movement_gain_schedule_geometry_phase': float(
                gain_geometry_phase),
            'movement_gain_schedule_evaluated_viewers': 0.0,
            'movement_gain_schedule_selected_self_rate': 0.0,
            'movement_gain_schedule_strategy_hold_rate': 0.0,
            'movement_gain_schedule_no_improvement_rate': 0.0,
            'movement_gain_schedule_far_range_rate': 0.0,
            'movement_target_prediction_frames': float(
                self._distributed_greedy_matching_prediction_frames),
        }
        if self.Q <= 0:
            return {}
        out = {}
        standoff = float(self._distributed_id_movement_standoff_m)
        for viewer in range(self.K):
            target_position_3d, target_velocity_3d = (
                self._movement_target_state_for_viewer(viewer))
            target_xy = target_position_3d[:, :2]
            if self._distributed_greedy_matching_prediction_frames > 0:
                target_xy, _ = predict_reflecting_cv_mean(
                    target_xy,
                    target_velocity_3d[:, :2],
                    elapsed_s=(
                        self._distributed_greedy_matching_prediction_frames
                        * float(self.cfg.scenario.dt)),
                    area_size_m=tuple(
                        float(v) for v in self.cfg.scenario.region_size),
                )
            held_target = int(self._distributed_movement_target[viewer])
            held = bool(
                0 <= held_target < self.Q
                and self.t - int(
                    self._distributed_movement_last_update[viewer])
                < self._distributed_greedy_matching_hold_frames
            )
            if held:
                q = held_target
            else:
                q = int(viewer % self.Q)
            public_xy = np.zeros((self.K, 2), dtype=np.float64)
            public_xy[viewer, 0] = (
                self._hyperedge_local_offer[viewer, 0, 0]
                * float(self.area_size[0]))
            public_xy[viewer, 1] = (
                self._hyperedge_local_offer[viewer, 0, 1]
                * float(self.area_size[1]))
            last_seen = self._hyperedge_received_last_seen[viewer]
            peer_seen = np.all(
                (last_seen > FRAME_NEVER)
                & (
                    self.t - last_seen
                    <= self._distributed_movement_public_max_age_frames
                ),
                axis=1,
            )
            complete = bool(np.sum(peer_seen) >= self.K - 1)
            peer_age = np.zeros(self.K, dtype=np.int64)
            for node in range(self.K):
                if node != viewer and peer_seen[node]:
                    peer_age[node] = int(np.max(
                        self.t - last_seen[node]))
            view_max_age = int(np.max(peer_age))
            if complete and not held:
                for node in range(self.K):
                    if node == viewer:
                        continue
                    public_xy[node, 0] = (
                        self._hyperedge_received_offer[
                            viewer, node, 0, 0]
                        * float(self.area_size[0])
                    )
                    public_xy[node, 1] = (
                        self._hyperedge_received_offer[
                            viewer, node, 0, 1]
                        * float(self.area_size[1])
                    )
                distance = np.linalg.norm(
                    public_xy[:, None, :] - target_xy[None, :, :],
                    axis=-1,
                )
                if self._distributed_bistatic_bottleneck_movement_enabled:
                    tx_role_mask = self._distributed_tx_role_mask()
                    gate_threshold = self._distributed_bistatic_tail_gate_ratio
                    if (gate_threshold > 0.0
                            and self._distributed_bistatic_gate_latch[viewer] < 0):
                        tail_ratio = bistatic_geometry_tail_ratio(
                            public_xy, target_xy, tx_role_mask,
                            height_m=float(self.cfg.scenario.height))
                        self._distributed_bistatic_gate_latch[viewer] = int(
                            tail_ratio >= gate_threshold)
                    gate_open = bool(
                        gate_threshold <= 0.0
                        or self._distributed_bistatic_gate_latch[viewer] == 1)
                    if gate_open:
                        bistatic_cost = role_aware_bistatic_movement_cost(
                            public_xy,
                            target_xy,
                            tx_role_mask,
                            height_m=float(self.cfg.scenario.height),
                            movement_step_m=float(
                                self.uavs[viewer].v_max * self.uavs[viewer].dt),
                            standoff_m=standoff,
                            complement_exponent=(
                                self._distributed_bistatic_complement_exponent),
                        )
                        assignment_vector = (
                            deterministic_bottleneck_cost_assignment(
                                bistatic_cost))
                    else:
                        assignment_vector = deterministic_bottleneck_matching(
                            public_xy, target_xy)
                    assigned = int(assignment_vector[viewer])
                    q = int(assigned if assigned >= 0 else viewer % self.Q)
                elif self._distributed_bottleneck_matching_movement_enabled:
                    assignment_vector = deterministic_bottleneck_matching(
                        public_xy, target_xy)
                    assigned = int(assignment_vector[viewer])
                    q = int(assigned if assigned >= 0 else viewer % self.Q)
                else:
                    pairs = sorted(
                        (float(distance[node, target]), int(node), int(target))
                        for node in range(self.K)
                        for target in range(self.Q)
                    )
                    assigned_node = set()
                    assigned_target = set()
                    assignment = {}
                    for _distance, node, target in pairs:
                        if (node in assigned_node
                                or target in assigned_target):
                            continue
                        assignment[node] = target
                        assigned_node.add(node)
                        assigned_target.add(target)
                    assignment_vector = np.full(
                        self.K, -1, dtype=np.int64)
                    for node, target in assignment.items():
                        assignment_vector[int(node)] = int(target)
                    q = int(assignment.get(viewer, viewer % self.Q))
                if (
                    self._distributed_gain_scheduled_movement_enabled
                    and self._distributed_gain_scheduled_far_assignment_mode
                    == 'fixed_id'
                ):
                    assignment_vector = (
                        np.arange(self.K, dtype=np.int64) % self.Q)
                    q = int(assignment_vector[viewer])
                self._distributed_movement_target[viewer] = q
                self._distributed_movement_local_assignment[viewer] = (
                    np.asarray(assignment_vector, dtype=np.int64))
                self._distributed_movement_last_update[viewer] = int(self.t)
                if self._distributed_movement_safety_projection_enabled:
                    public_view_max_ages.append(view_max_age)
                    movement_step = float(
                        self.uavs[viewer].v_max * self.uavs[viewer].dt)
                    desired, movement_states = (
                        self._distributed_public_desired_movement(
                            public_xy,
                            target_xy,
                            assignment_vector,
                            movement_step,
                        ))
                    ao_viewer_states.append(movement_states[viewer])
                    safe, safety_info = project_pairwise_safe_movement(
                        public_xy,
                        desired,
                        minimum_distance_m=(
                            float(self.cfg.uav.d_safe)
                            + self._distributed_movement_safety_margin_m
                            + self._distributed_movement_safety_margin_per_age_m
                            * view_max_age),
                        maximum_step_m=movement_step,
                        area_size_xy=(
                            float(self.area_size[0]),
                            float(self.area_size[1]),
                        ),
                        return_diagnostics=True,
                        outside_invariant_recovery=(
                            self._distributed_movement_outside_invariant_recovery_enabled),
                        recovery_gain=(
                            self._distributed_movement_outside_invariant_recovery_gain),
                        independently_composable=(
                            self._distributed_movement_independently_composable_safety),
                        analytic_composable_projection=(
                            self._distributed_movement_analytic_composable_projection_enabled),
                    )
                    safety_diagnostics.append(safety_info)
                    if (
                        self._distributed_movement_execute_projected_public_action
                    ):
                        # Presence in ``out`` is the override signal.  An
                        # explicit zero is therefore required for a genuine
                        # fail-closed hold; omitting the key releases the
                        # unverified actor movement.
                        out[viewer] = safe[viewer].copy()
                        continue
                    if np.any(np.abs(safe - desired) > 1.0e-9):
                        out[viewer] = safe[viewer].copy()
                        continue
            if (complete and held
                    and self._distributed_movement_safety_projection_enabled):
                public_view_max_ages.append(view_max_age)
                for node in range(self.K):
                    if node == viewer:
                        continue
                    public_xy[node, 0] = (
                        self._hyperedge_received_offer[
                            viewer, node, 0, 0]
                        * float(self.area_size[0])
                    )
                    public_xy[node, 1] = (
                        self._hyperedge_received_offer[
                            viewer, node, 0, 1]
                        * float(self.area_size[1])
                    )
                if self._distributed_movement_local_assignment_cache_enabled:
                    assignment_vector = (
                        self._distributed_movement_local_assignment[
                            viewer].copy())
                    if np.any(assignment_vector < 0):
                        stale_fail_closed_viewers += 1
                        out[viewer] = np.zeros(2, dtype=np.float64)
                        continue
                else:
                    assignment_vector = (
                        self._distributed_movement_target.copy())
                movement_step = float(
                    self.uavs[viewer].v_max * self.uavs[viewer].dt)
                desired, movement_states = (
                    self._distributed_public_desired_movement(
                        public_xy,
                        target_xy,
                        assignment_vector,
                        movement_step,
                    ))
                ao_viewer_states.append(movement_states[viewer])
                safe, safety_info = project_pairwise_safe_movement(
                    public_xy,
                    desired,
                    minimum_distance_m=(
                        float(self.cfg.uav.d_safe)
                        + self._distributed_movement_safety_margin_m
                        + self._distributed_movement_safety_margin_per_age_m
                        * view_max_age),
                    maximum_step_m=movement_step,
                    area_size_xy=(
                        float(self.area_size[0]),
                        float(self.area_size[1]),
                    ),
                    return_diagnostics=True,
                    outside_invariant_recovery=(
                        self._distributed_movement_outside_invariant_recovery_enabled),
                    recovery_gain=(
                        self._distributed_movement_outside_invariant_recovery_gain),
                    independently_composable=(
                        self._distributed_movement_independently_composable_safety),
                    analytic_composable_projection=(
                        self._distributed_movement_analytic_composable_projection_enabled),
                )
                safety_diagnostics.append(safety_info)
                if self._distributed_movement_execute_projected_public_action:
                    out[viewer] = safe[viewer].copy()
                    continue
                if np.any(np.abs(safe - desired) > 1.0e-9):
                    out[viewer] = safe[viewer].copy()
                    continue
            if not complete and self._distributed_movement_stale_fail_closed:
                stale_fail_closed_viewers += 1
                out[viewer] = np.zeros(2, dtype=np.float64)
                continue
            delta = target_xy[q] - np.asarray(self.uavs[viewer].pos[:2])
            distance = float(np.linalg.norm(delta))
            if distance <= standoff + 1.0e-12:
                continue
            step = min(
                float(self.uavs[viewer].v_max * self.uavs[viewer].dt),
                distance - standoff,
            )
            if step > 0.0:
                out[viewer] = step * delta / max(distance, 1.0e-12)
        if safety_diagnostics:
            solve_time = float(sum(
                float(item['solve_time_s'])
                for item in safety_diagnostics))
            fail_closed_calls = int(sum(
                bool(item['fail_closed'])
                for item in safety_diagnostics))
            fail_closed_outside_invariant_calls = int(sum(
                bool(item['fail_closed']) and not bool(item['initially_safe'])
                for item in safety_diagnostics))
            outside_invariant_calls = int(sum(
                not bool(item['initially_safe'])
                for item in safety_diagnostics))
            recovery_pair_count = int(sum(
                int(item.get('recovery_pair_count', 0))
                for item in safety_diagnostics))
            intervention_calls = int(sum(
                bool(item['intervened'])
                for item in safety_diagnostics))
            self._last_movement_safety_metrics.update({
                'movement_safety_projection_calls': float(
                    len(safety_diagnostics)),
                'movement_safety_intervened': float(
                    intervention_calls > 0),
                'movement_safety_intervention_call_rate': float(
                    intervention_calls / len(safety_diagnostics)),
                'movement_safety_fail_closed': float(
                    fail_closed_calls > 0),
                'movement_safety_fail_closed_calls': float(
                    fail_closed_calls),
                'movement_safety_fail_closed_outside_invariant_calls': float(
                    fail_closed_outside_invariant_calls),
                'movement_safety_outside_invariant_calls': float(
                    outside_invariant_calls),
                'movement_safety_recovery_pair_count': float(
                    recovery_pair_count),
                'movement_safety_solve_time_s': solve_time,
                'movement_safety_solve_time_s_per_node': float(
                    solve_time / max(self.K, 1)),
                'movement_safety_pairwise_constraints_mean': float(np.mean([
                    float(item['pairwise_constraint_count'])
                    for item in safety_diagnostics
                ])),
            })
        self._last_movement_safety_metrics.update({
            'movement_public_view_stale_fail_closed_viewers': float(
                stale_fail_closed_viewers),
            'movement_public_view_stale_fail_closed_rate': float(
                stale_fail_closed_viewers / max(self.K, 1)),
            'movement_public_view_mean_max_age_frames': float(np.mean(
                public_view_max_ages or [0.0])),
            'movement_public_view_max_age_frames': float(max(
                public_view_max_ages or [0.0])),
        })
        if ao_viewer_states:
            denominator = float(len(ao_viewer_states))
            self._last_movement_safety_metrics.update({
                'movement_ao_evaluated_viewers': denominator,
                'movement_ao_far_recovery_rate': float(
                    ao_viewer_states.count('far_recovery') / denominator),
                'movement_ao_near_recovery_rate': float(
                    ao_viewer_states.count('near_recovery') / denominator),
                'movement_ao_range_hold_rate': float(
                    ao_viewer_states.count('range_hold') / denominator),
                'movement_ao_strategy_tangent_rate': float(
                    ao_viewer_states.count('strategy_tangent') / denominator),
                'movement_ao_strategy_hold_rate': float(
                    ao_viewer_states.count('strategy_hold') / denominator),
                'movement_gain_schedule_evaluated_viewers': denominator,
                'movement_gain_schedule_selected_self_rate': float(
                    ao_viewer_states.count('gain_selected') / denominator),
                'movement_gain_schedule_strategy_hold_rate': float(
                    ao_viewer_states.count('gain_strategy_hold')
                    / denominator),
                'movement_gain_schedule_no_improvement_rate': float(
                    ao_viewer_states.count('gain_no_improvement')
                    / denominator),
                'movement_gain_schedule_far_range_rate': float(
                    (
                        ao_viewer_states.count('gain_far_range')
                        + ao_viewer_states.count('gain_far_range_hold')
                    ) / denominator),
            })
        executed_targets = np.asarray(
            self._distributed_movement_target, dtype=np.int64)
        valid_executed = executed_targets[
            (executed_targets >= 0) & (executed_targets < self.Q)]
        unique_executed = int(np.unique(valid_executed).size)
        valid_views = np.asarray(
            self._distributed_movement_local_assignment,
            dtype=np.int64,
        )
        valid_views = valid_views[np.all(valid_views >= 0, axis=1)]
        entry_agreements = []
        if valid_views.shape[0] > 0:
            for node in range(self.K):
                _, counts = np.unique(
                    valid_views[:, node], return_counts=True)
                entry_agreements.append(
                    float(np.max(counts) / valid_views.shape[0]))
        self._last_movement_safety_metrics.update({
            'movement_executed_assignment_target_coverage': float(
                unique_executed / max(self.Q, 1)),
            'movement_executed_assignment_duplicate_rate': float(
                1.0 - unique_executed / max(valid_executed.size, 1)),
            'movement_local_assignment_entry_agreement_rate': float(np.mean(
                entry_agreements or [0.0])),
            'movement_anchor_target_coverage': float(np.mean(
                np.sum(
                    np.any(
                        (
                            self._distributed_movement_anchor_last_seen
                            > FRAME_NEVER
                        )
                        & (
                            self.t
                            - self._distributed_movement_anchor_last_seen
                            <= self._distributed_movement_anchor_max_age_frames
                        ),
                        axis=1,
                    ),
                    axis=1,
                ) / max(self.Q, 1)
            )),
        })
        return out

    def _analytical_movement_delta(self) -> dict:
        """Receding-horizon capability-guided movement (L3 geometry hook).

        Uses the previous frame's per-watt coefficient, owner map and sensing
        budget to take ONE step of the deficit->capability descent (D0.95): if
        the fixed-owner ceiling is below the worst floor, move along the
        steepest descent of the feasibility violation; otherwise solve the
        capability gauge and move along its price-weighted gradient.  Returns
        ``{uav_id: delta_p(2,)}``; empty on the first frame or when the
        previous structure is unavailable.
        """
        entries = getattr(self, '_last_deflection_entries', None)
        selected = tuple(
            tuple(int(v) for v in e)
            for e in getattr(self, '_last_selected_set', ())
        )
        if entries is None or not selected:
            # D1.8 feasibility-aware warm start (advice 013): on the first
            # frame the previous-frame structure is unavailable (reset does
            # not compute it), so the L3 hook used to waste frame 0.  Build
            # the initial deflection entries and a minimal single-owner
            # structure from the CURRENT geometry so the deficit descent can
            # act immediately.  If the initial capability gauge gamma_0* > 1
            # (worst floor infeasible at the initial geometry), Phase 1 below
            # drives the geometry repair from frame 0.
            if self.t <= 1:
                entries, selected = self._initial_analytical_state()
            if entries is None or not selected:
                return {}
        coefficient = self._per_watt_coefficient_from_entries(entries)
        try:
            gain, owner = fixed_owner_gain_matrix(coefficient, selected)
        except (
            IncompleteFixedOwnerStructureError,
            NonUniqueFixedOwnerStructureError,
        ):
            # A partially resolved previous graph is an expected transient;
            # the actor movement remains the documented fallback.
            return {}
        except ValueError as error:
            # Shape, finiteness, edge-support and unique-owner failures are
            # invariant violations.  Silently treating them as an unavailable
            # graph would hide corrupted physics behind actor movement.
            raise RuntimeError(
                "invalid fixed-owner state for analytical movement"
            ) from error
        budget = np.clip(
            self._isac_total_power_w - self._current_comm_power_w, 0.0, None)
        step = float(self.uavs[0].v_max * self.uavs[0].dt)
        uav = np.array([u.pos[:2].copy() for u in self.uavs])
        if self._distributed_no_truth_fail_closed:
            # Strict no-truth L3 (audit advice/001 P0, 2026-08-26): the L3
            # geometry hook consumes the public median local-belief map, never
            # simulator ground truth.
            tgt, _tgt_vel = self._strict_no_truth_target_map()
        else:
            tgt = np.array([t.get_position_3d()[:2] for t in self.targets])
        height_sq = float(self.cfg.scenario.height) ** 2
        p_fa = float(self.cfg.detection.P_FA)

        from uav_isac.coordination.capability import local_capability_gradient_k
        from uav_isac.physical.detection import (
            minimum_deflection_for_detection_probability)
        from uav_isac.coordination.maxmin_power import (
            solve_fixed_structure_maxmin_power_lp, optimal_maxmin_dual_prices,
            entropic_maxmin_dual_prices)

        d_min = float(minimum_deflection_for_detection_probability(
            np.asarray([0.60]), p_fa)[0])
        d_steady = float(minimum_deflection_for_detection_probability(
            np.asarray([0.80]), p_fa)[0])
        ceiling = np.sum(gain * budget[:, None], axis=0)

        # D1.10-B (2026-08-16): the Phase-1 trigger previously used only
        # ``ceiling = sum_k gain[k,q] * budget[k]`` -- each UAV's power treated
        # as independently available to every target.  Under the real 1 W/UAV
        # coupling (one UAV's watt serves all its owner targets), the max-min
        # LP value t* can sit far below d_min while every ceiling is above it
        # (blind100 seed 615: ceilings all >= 11.18, t* = 6.44, P_D stuck at
        # 0.29).  In that regime Phase 1 never fires, the L3 pool picks stay
        # and the geometry stalls at a power-coupling bottleneck.  Resolve the
        # fixed-structure max-min LP once here and drive the deficit descent
        # from the REAL achievable worst deflection instead of the optimistic
        # ceiling.  This is exact: t* is the deflection the executed L1 power
        # path will actually deliver at this geometry.
        try:
            res_trigger = solve_fixed_structure_maxmin_power_lp(gain, budget)
            t_star = float(res_trigger.worst_deflection)
        except ValueError as error:
            raise RuntimeError(
                "invalid max-min trigger inputs for analytical movement"
            ) from error
        phase1_deficit = (
            np.any(ceiling < d_min - 1e-9)
            or (bool(getattr(self.cfg.marl,
                             'analytical_movement_tstar_trigger', True))
                and t_star < d_min - 1e-9))

        grads: dict = {}
        if phase1_deficit:
            # Phase 1: deficit gradient (steepest descent of the violation).
            # Deficit weights use the REAL max-min shortfall when the ceiling
            # test passes but coupling still binds (t* < d_min); fall back to
            # the ceiling deficit otherwise.
            if (t_star < d_min - 1e-9
                    and not np.any(ceiling < d_min - 1e-9)
                    and bool(getattr(self.cfg.marl,
                                     'analytical_movement_tstar_trigger',
                                     True))):
                # D1.10-B (rev 2): weight the deficit by the max-min DUAL
                # price lambda* (concentrated on the bottleneck targets that
                # actually bind the coupling) instead of a uniform
                # ``d_min - t_star`` on every target.  A uniform deficit made
                # every UAV pull toward ALL targets and diluted the gradient
                # (20-seed indep A/B: QoS 14/20 vs 15/20, seed 615 0.710 ->
                # 0.331).  lambda* is exactly the marginal that a 1 W/UAV
                # power reallocation cannot fix by itself, so geometry must
                # move to raise the ceiling of the bottleneck targets.
                lam_trigger, _ = (
                    entropic_maxmin_dual_prices(
                        gain, budget, self._entropic_dual_tau)
                    if self._entropic_dual_price_enabled
                    else optimal_maxmin_dual_prices(gain, budget))
                lam_abs = np.abs(np.asarray(lam_trigger, dtype=np.float64))
                lam_sum = float(np.sum(lam_abs))
                if lam_sum > 1e-12:
                    deficit = (d_min - t_star) * (lam_abs / lam_sum)
                else:
                    deficit = np.full(self.Q, d_min - t_star)
                deficit = np.maximum(0.0, deficit)
            else:
                deficit = np.maximum(0.0, d_min - ceiling)
            for k in range(self.K):
                gk = np.zeros(2, dtype=np.float64)
                for q in range(self.Q):
                    w = deficit[q] * budget[k]
                    if abs(w) < 1e-15:
                        continue
                    a = gain[k, q]
                    horizontal = uav[k] - tgt[q]
                    range_sq = float(horizontal @ horizontal) + height_sq
                    gk += (
                        w * (2.0 * a) * horizontal
                        / max(range_sq, 1e-9))
                for q in range(self.Q):
                    if owner[q] != k:
                        continue
                    horizontal = uav[k] - tgt[q]
                    range_sq = float(horizontal @ horizontal) + height_sq
                    for i in range(self.K):
                        w = deficit[q] * budget[i]
                        if abs(w) < 1e-15:
                            continue
                        a = gain[i, q]
                        gk += (
                            w * (2.0 * a) * horizontal
                            / max(range_sq, 1e-9))
                grads[k] = gk
        else:
            # Phase 2: max-min dual price gradient (cheap LP, no PWL).  The
            # max-min dual lambda* concentrates on the bottleneck target, which
            # is exactly the price that drags the steady (temporal-mean worst).
            # Reuse the D1.10-B trigger LP result when available (same inputs).
            res = res_trigger if res_trigger is not None else (
                solve_fixed_structure_maxmin_power_lp(gain, budget))
            if (not self._analytical_movement_candidates_enabled
                    and res.worst_deflection >= d_steady - 1e-9):
                # Already at/above the steady floor: hover (preserve the good
                # geometry) instead of falling back to the actor's motion,
                # which would oscillate and re-degrade a hard seed.
                return {k: np.zeros(2, dtype=np.float64) for k in range(self.K)}
            lam, _ = (
                entropic_maxmin_dual_prices(
                    gain, budget, self._entropic_dual_tau)
                if self._entropic_dual_price_enabled
                else optimal_maxmin_dual_prices(gain, budget))
            prices = -lam  # gauge sign convention (negative marginals)
            support = {q: {i: (res.power_w[i, q], gain[i, q])
                           for i in range(self.K)} for q in range(self.Q)}
            for k in range(self.K):
                grads[k] = local_capability_gradient_k(
                    k, owner, prices, res.power_w[k], gain[k], uav[k], tgt,
                    support, height_m=float(self.cfg.scenario.height))

        # D1.1-B (advice 010): bounded multi-candidate trust-region.  Build a
        # few whole-fleet movement candidates, evaluate each at the moved
        # geometry with the exact max-min LP, and execute the best.  ``stay`` is
        # always a candidate, so the proxy score is monotone (never degrades).
        # D1.10-B (rev 3): when the t*-triggered coupling-scarce Phase 1 fired
        # (t* < d_min while every ceiling >= d_min), bypass the candidate
        # scorer and execute the lambda*-weighted deficit gradient directly.
        # RATIONALE: the candidate scorer scores by the max-min LP at the moved
        # geometry, but under power coupling moving a UAV toward the bottleneck
        # target makes ANOTHER target the new bottleneck, so t* does not rise
        # and the pool picks stay -- the scorer cannot see the structural
        # reallocation value (blind100 seed 615).  The deficit gradient is the
        # steepest descent of the feasibility violation itself, so it is the
        # correct repair direction in exactly this regime; the candidate pool
        # remains the gatekeeper everywhere else.
        if self._analytical_movement_candidates_enabled:
            if (phase1_deficit
                    and bool(getattr(self.cfg.marl,
                                     'analytical_movement_phase1_force',
                                     True))):
                delta: dict = {}
                for k in range(self.K):
                    gk = grads.get(k)
                    if gk is None:
                        continue
                    n = float(np.linalg.norm(gk))
                    if n > 1e-12:
                        delta[k] = -step * gk / n
                if delta:
                    return delta
            return self._select_best_movement_candidate(
                coefficient, selected, budget, uav, tgt, grads, step)

        delta: dict = {}
        for k in range(self.K):
            gk = grads.get(k)
            if gk is None:
                continue
            n = float(np.linalg.norm(gk))
            if n > 1e-12:
                delta[k] = -step * gk / n
        return delta

    @staticmethod
    def _friis_rescale_tensor(
        coeff: np.ndarray, uav: np.ndarray, tgt: np.ndarray,
        new_uav: np.ndarray,
        *,
        height_m: float | np.ndarray = 0.0,
    ) -> np.ndarray:
        """Rescale the (K,K,Q) per-watt tensor under 1/(R_tx^2 R_rx^2)."""
        delta = np.asarray(uav, dtype=np.float64)[:, None, :] - np.asarray(
            tgt, dtype=np.float64)[None, :, :]
        new_delta = (
            np.asarray(new_uav, dtype=np.float64)[:, None, :]
            - np.asarray(tgt, dtype=np.float64)[None, :, :])
        vertical = np.asarray(height_m, dtype=np.float64)
        if np.any(~np.isfinite(vertical)):
            raise ValueError("height_m must be finite")
        if delta.shape[2] >= 3 and np.any(vertical != 0.0):
            raise ValueError(
                "height_m must be zero when positions already contain altitude")
        try:
            vertical_sq = np.broadcast_to(
                vertical * vertical, delta.shape[:2])
        except ValueError as exc:
            raise ValueError(
                "height_m must be scalar or broadcastable to (K,Q)") from exc
        range_sq = np.sum(delta * delta, axis=2) + vertical_sq
        new_range_sq = np.sum(new_delta * new_delta, axis=2) + vertical_sq
        path_constant = (
            np.asarray(coeff, dtype=np.float64)
            * range_sq[:, None, :] * range_sq[None, :, :])
        return path_constant / (
            new_range_sq[:, None, :] * new_range_sq[None, :, :])

    def _select_best_movement_candidate(
        self,
        coefficient: np.ndarray,
        selected: tuple,
        budget: np.ndarray,
        uav: np.ndarray,
        tgt: np.ndarray,
        grads: dict,
        step: float,
    ) -> dict:
        """D1.1-B: bounded whole-fleet trust-region candidate selection.

        Candidates: stay, the single-step gradient (D0.95), half of it, and
        radial steps toward the two weakest targets (by max-min ceiling).  Each
        candidate is scored at the moved geometry with the exact max-min power
        LP (structure fixed at the current P0 selection); the best is executed.
        ``stay`` is always included, so the proxy score is monotone.

        D1.1-B+ dual pruning (2026-08-16): before running the exact LP for a
        candidate, its weak-duality upper bound is computed with the current
        frame's optimal dual price ``lambda*``:

            U_lambda(g') = sum_i b_i * max_q lambda*_q * a'_iq
                           >= t*(g') = max-min deflection at g'

        (weak duality: any feasible simplex price bounds the max-min value from
        above).  P_D = Q(Q^{-1}(P_FA) - sqrt(D)) is strictly monotone in D, so
        ``U_lambda(g') <= best_deflection`` implies the candidate's worst P_D
        cannot exceed the current best and its exact LP is provably dominated.
        The pruning is exact: it never changes the selected candidate, it only
        skips LP evaluations that cannot win.  The price is computed once per
        frame on the current geometry (O(KQ) per candidate check).
        """
        K, Q = self.K, self.Q
        area = tuple(float(v) for v in self.cfg.scenario.region_size)
        d_safe = float(getattr(self.cfg.uav, 'd_safe', 20.0))
        p_fa = float(self.cfg.detection.P_FA)
        from uav_isac.coordination.maxmin_power import (
            fixed_owner_gain_matrix,
            optimal_maxmin_dual_prices,
            solve_fixed_structure_maxmin_power_lp,
            entropic_maxmin_dual_prices,
        )
        from uav_isac.physical.detection import (
            compute_detection_probabilities,
        )

        base = np.zeros((K, 2), dtype=np.float64)
        for k in range(K):
            gk = grads.get(k)
            if gk is None:
                continue
            n = float(np.linalg.norm(gk))
            if n > 1e-12:
                base[k] = -step * gk / n

        try:
            gain_cur, owner = fixed_owner_gain_matrix(coefficient, selected)
        except (
            IncompleteFixedOwnerStructureError,
            NonUniqueFixedOwnerStructureError,
        ):
            return {}
        except ValueError as error:
            raise RuntimeError(
                "invalid fixed-owner state in movement candidate selection"
            ) from error
        ceiling = np.sum(gain_cur * budget[:, None], axis=0)
        weak_order = np.argsort(ceiling)

        # D1.1-B+ dual pruning price (optimal dual of the CURRENT geometry).
        # lambda* is a feasible simplex price for every candidate geometry, so
        # weak duality holds per candidate regardless of where the UAVs move.
        dual_prune = bool(getattr(
            self.cfg.marl, 'analytical_movement_dual_prune', True))
        lam = None
        if dual_prune:
            try:
                lam, _ = (
                    entropic_maxmin_dual_prices(
                        gain_cur, budget, self._entropic_dual_tau)
                    if self._entropic_dual_price_enabled
                    else optimal_maxmin_dual_prices(gain_cur, budget))
            except ValueError as error:
                raise RuntimeError(
                    "invalid max-min dual inputs in movement candidate selection"
                ) from error

        def radial(weak_q: int) -> np.ndarray:
            d = np.zeros((K, 2), dtype=np.float64)
            for k in range(K):
                v = tgt[weak_q] - uav[k]
                n = float(np.linalg.norm(v))
                if n > 1e-9:
                    d[k] = step * v / n
            return d

        def radial_away(weak_q: int) -> np.ndarray:
            # D1.1-F standoff candidate: AWAY from the weakest target.  Under
            # the covertness constraint the opponent's per-watt gain is
            # a^I ~ 1/d^2, so moving away relaxes the counter-detection bound
            # and lets more power through, at the cost of sensing gain
            # (1/(R_tx^2 R_rx^2)).  The exact max-min LP decides the trade-off.
            d = np.zeros((K, 2), dtype=np.float64)
            for k in range(K):
                v = uav[k] - tgt[weak_q]
                n = float(np.linalg.norm(v))
                if n > 1e-9:
                    d[k] = step * v / n
            return d

        candidates = [
            np.zeros((K, 2), dtype=np.float64),
            base,
            0.5 * base,
            radial(int(weak_order[0])),
        ]
        if Q >= 2:
            candidates.append(radial(int(weak_order[1])))
        # D1.9 (2026-08-16): bottleneck lookahead scoring.  The radial
        # candidates above are scored at the 1-step geometry; at
        # R ~ 300-450 m a 2.5 m step barely moves the ceiling, so the exact
        # LP cannot discriminate the approach direction and the pool often
        # picks stay -- the blind100 left-tail mechanism.  Re-score the weak
        # radial candidates at the H-frame sustained-approach geometry
        # (uav + H*step*dir), executing only 1 step (receding horizon).  The
        # per-watt tensor is rescaled by the exact 1/R^4 law; d_safe and the
        # stay candidate keep the proxy monotone.
        lookahead_h = int(getattr(
            self.cfg.marl, 'analytical_movement_lookahead_frames', 0))
        if lookahead_h > 0:
            for wq in range(min(2, Q)):
                d_la = np.zeros((K, 2), dtype=np.float64)
                for k in range(K):
                    v = tgt[int(weak_order[wq])] - uav[k]
                    n = float(np.linalg.norm(v))
                    if n > 1e-9:
                        d_la[k] = step * v / n
                # same direction as radial(weak_q), but scored after H
                # sustained frames (receding horizon): executing this
                # candidate still clamps to one 2.5 m step, while the score
                # sees the H-frame approach geometry where 1/R^4 P_D
                # discrimination is strong.
                candidates.append(float(lookahead_h) * d_la)
        # D1.1-F: when the live covertness path is active, the standoff
        # directions (away from the two weakest targets) join the candidate
        # set so the geometry can trade sensing range for covertness headroom.
        if (self._intercept_constrained_power_enabled
                and bool(getattr(self.cfg.marl,
                                 'analytical_movement_standoff_candidates',
                                 True))):
            candidates.append(radial_away(int(weak_order[0])))
            if Q >= 2:
                candidates.append(radial_away(int(weak_order[1])))

        # D1.1-B++ gauge-price step: the capability-gauge dual pi* is the
        # shadow price of the full task set (worst+bottom-k+steady), while
        # lambda* concentrates on the worst target.  Both price-driven
        # directions (plus a convex dual x radial combination) join the
        # candidate set and compete under the exact max-min LP; the D0.95
        # "L1 uses pi / L3 uses lambda*" split is replaced by candidate
        # competition so the better direction wins on the actual physics.
        gauge_step = bool(getattr(
            self.cfg.marl, 'analytical_movement_gauge_price_step', False))
        if gauge_step:
            try:
                from uav_isac.coordination.capability import (
                    capability_gauge_pwl_lp_full,
                    capability_geometry_gradient,
                )
                from uav_isac.coordination.pwl_pd import (
                    chord_lower_bound,
                    curvature_breakpoints,
                )
                from uav_isac.physical.detection import (
                    minimum_deflection_for_detection_probability,
                )
                qos_floors = tuple(float(v) for v in getattr(
                    self.cfg.marl, 'task_constrained_qos_floors',
                    (0.60, 0.70, 0.80, 3)))
                d_min = float(minimum_deflection_for_detection_probability(
                    np.asarray([qos_floors[0]]), p_fa)[0])
                if not np.any(ceiling < d_min - 1e-9):
                    d_max = float(np.max(ceiling)) + 1.0
                    bps = curvature_breakpoints(
                        p_fa, d_min, d_max, epsilon=1e-3)
                    cs, ci = chord_lower_bound(p_fa, bps)
                    out = capability_gauge_pwl_lp_full(
                        gain_cur, budget, p_fa,
                        (qos_floors[0], qos_floors[1], qos_floors[2],
                         max(1, int(qos_floors[3]))),
                        cs, ci, d_min)
                    if out is not None and out[0] <= 1.0 + 1e-6:
                        _, p_star, pi_star = out
                        grad_pi = capability_geometry_gradient(
                            gain_cur, owner, uav, tgt, p_star, pi_star,
                            height_m=float(self.cfg.scenario.height))
                        pi_step = np.zeros((K, 2), dtype=np.float64)
                        for k in range(K):
                            gk = grad_pi[k]
                            n = float(np.linalg.norm(gk))
                            if n > 1e-12:
                                pi_step[k] = -step * gk / n
                        candidates.append(pi_step)
                        candidates.append(
                            0.5 * pi_step + 0.5 * radial(int(weak_order[0])))
            except (ValueError, ImportError) as error:
                # Best-effort candidate generation: a degenerate geometry should
                # not abort the step, but the failure must not vanish silently.
                logger.debug("PWL candidate generation skipped: %r", error)

        best = np.zeros((K, 2), dtype=np.float64)
        best_s = float('-inf')
        best_deflection: float | None = None
        # D1.1-B+++ score/execute consistency: use the SAME L1 solver that
        # will be executed to score each candidate.  The D1.1-B scorer used
        # the plain max-min LP even under a lexicographic inner layer, so a
        # candidate could win the pure max-min score yet degrade under the
        # QoS floors actually executed.  When lexicographic mode is active we
        # score with qos_constrained_maxmin_lp (Stage B) and fall back to
        # reserve-first max-min on infeasible geometries, mirroring the
        # executed power path.  Weak-duality pruning stays exact: lex t* is
        # at most the plain max-min t*, which is at most U_lambda.
        use_lex_scoring = bool(
            getattr(self.cfg.marl, 'analytical_movement_lex_scoring', True)
            and getattr(self.cfg.marl, 'task_constrained_power_enabled', False)
            and getattr(self.cfg.marl, 'task_constrained_mode', 'gauge')
            == 'lexicographic')
        lex_l1 = None
        if use_lex_scoring:
            try:
                from uav_isac.coordination.capability import (
                    qos_constrained_maxmin_lp,
                )
                from uav_isac.coordination.pwl_pd import (
                    chord_lower_bound,
                    curvature_breakpoints,
                )
                from uav_isac.physical.detection import (
                    minimum_deflection_for_detection_probability,
                )
                qos_floors = tuple(float(v) for v in getattr(
                    self.cfg.marl, 'task_constrained_qos_floors',
                    (0.60, 0.70, 0.80, 3)))
                d_min = float(minimum_deflection_for_detection_probability(
                    np.asarray([qos_floors[0]]), p_fa)[0])
                d_max = float(np.max(ceiling)) + 1.0
                bps = curvature_breakpoints(p_fa, d_min, d_max, epsilon=1e-3)
                cs, ci = chord_lower_bound(p_fa, bps)
                lex_l1 = (qos_constrained_maxmin_lp,
                          (qos_floors[0], qos_floors[1], qos_floors[2],
                           max(1, int(qos_floors[3]))), cs, ci, d_min)
            except (ValueError, ImportError) as error:
                # Lex-QoS scoring is optional; on a degenerate geometry fall back
                # to plain max-min scoring, but keep the drop observable.
                logger.debug("lex-QoS L1 scoring unavailable: %r", error)
                lex_l1 = None

        for cand in candidates:
            nu = np.clip(uav + cand, 0.0, area)
            # Collision / proximity guard: no UAV may come closer than d_safe
            # to any target (the 1/R^4 gain would otherwise explode and the
            # greedy candidate search would keep ramming UAVs into targets).
            dist = np.linalg.norm(
                nu[:, None, :] - tgt[None, :, :], axis=2)  # (K,Q)
            if float(np.min(dist)) < d_safe - 1e-9:
                continue
            coeff_c = self._friis_rescale_tensor(
                coefficient, uav, tgt, nu,
                height_m=float(self.cfg.scenario.height))
            try:
                g_c, _ = fixed_owner_gain_matrix(coeff_c, selected)
            except (
                IncompleteFixedOwnerStructureError,
                NonUniqueFixedOwnerStructureError,
            ):
                continue
            except ValueError as error:
                raise RuntimeError(
                    "invalid candidate fixed-owner state"
                ) from error
            # D1.1-B+ dual pruning: U_lambda(g') <= best_deflection proves the
            # candidate's max-min deflection (hence worst P_D) cannot beat the
            # incumbent, so the exact LP evaluation is provably dominated.
            if lam is not None and best_deflection is not None:
                u_lambda = float(np.sum(
                    budget * np.max(lam[None, :] * g_c, axis=1)))
                if u_lambda <= best_deflection + 1e-9:
                    continue
            if lex_l1 is not None:
                solver, xi, cs, ci, d_min = lex_l1
                out = solver(g_c, budget, p_fa, xi, cs, ci, d_min)
                if out is None:
                    # Stage-B infeasible: the executed path falls back to
                    # reserve-first max-min, so score that fallback too.
                    res = solve_fixed_structure_maxmin_power_lp(g_c, budget)
                    pd = compute_detection_probabilities(
                        res.deflection, p_fa)
                    s = float(np.min(pd))
                    cand_def = res.worst_deflection
                else:
                    t_star, _p_star, d_star = out
                    pd = compute_detection_probabilities(d_star, p_fa)
                    s = float(np.min(pd))
                    cand_def = float(np.min(d_star))
            else:
                res = solve_fixed_structure_maxmin_power_lp(g_c, budget)
                # Score in P_D space (saturates at 1), so once a target is
                # already saturated, ramming UAVs closer yields no further
                # score gain.
                pd = compute_detection_probabilities(res.deflection, p_fa)
                s = float(np.min(pd))
                cand_def = res.worst_deflection
            if s > best_s + 1e-9:
                best_s, best = s, cand
                best_deflection = cand_def
        out: dict = {}
        for k in range(K):
            if np.any(best[k] != 0.0):
                out[k] = best[k]
        return out

    def _serialized_protocol_accounting(
        self,
        comm_stats,
        evidence_transport,
    ) -> Dict[str, float]:
        """Unified MAC slot accounting for the two protocol broadcasts.

        The coordination broadcast (learned messages plus the hyperedge state
        stream) and the evidence/belief broadcast share one frame window.
        When they are transmitted serially in two sub-slots the total protocol
        latency is the sum of the two class latencies; the sum must itself fit
        the frame deadline for the protocol to be certificate-valid.  This is
        pure latency/bits bookkeeping -- it does not move traffic into a
        different slot, it makes the slot split explicit and testable.
        """
        deadline = max(1.0e-9, float(self._active_comm_deadline_s))
        coord_latency = float(getattr(comm_stats, 'mean_latency_s', 0.0))
        coord_p95 = float(getattr(comm_stats, 'p95_latency_s', 0.0))
        coord_bits = float(getattr(comm_stats, 'total_bits', 0.0))
        if evidence_transport is None:
            evidence_latency = 0.0
            evidence_p95 = 0.0
            evidence_bits = 0.0
        else:
            evidence_latency = float(
                getattr(evidence_transport, 'mean_latency_s', 0.0))
            evidence_p95 = float(
                getattr(evidence_transport, 'p95_latency_s', 0.0))
            evidence_bits = float(
                getattr(evidence_transport, 'total_bits', 0.0))
        serialized_mean = coord_latency + evidence_latency
        serialized_p95 = coord_p95 + evidence_p95
        return {
            'protocol_coord_latency_s': coord_latency,
            'protocol_evidence_latency_s': evidence_latency,
            'protocol_total_latency_s': serialized_mean,
            'protocol_total_p95_latency_s': serialized_p95,
            'protocol_serialized_bits': coord_bits + evidence_bits,
            'protocol_frame_deadline_s': deadline,
            'protocol_total_deadline_violation': float(
                serialized_mean > deadline),
            'protocol_total_p95_deadline_violation': float(
                serialized_p95 > deadline),
        }

    def _validate_step_actions(
        self,
        actions: Dict[int, Action],
    ) -> Dict[int, Action]:
        if not isinstance(actions, dict):
            raise ValueError("actions must be a mapping for every UAV")
        if any(
            isinstance(key, bool) or not isinstance(key, Integral)
            for key in actions
        ):
            raise ValueError("core action keys must be integer UAV IDs")
        actual = {int(key) for key in actions}
        if len(actual) != len(actions):
            raise ValueError("duplicate core action keys after integer coercion")
        expected = set(range(self.K))
        if actual != expected:
            missing = sorted(expected.difference(actual))
            extra = sorted(actual.difference(expected))
            raise ValueError(
                "actions must cover every UAV exactly once; "
                f"missing={missing}, extra={extra}")
        validated: Dict[int, Action] = {}
        for k in range(self.K):
            action = actions[k]
            if not isinstance(action, Action):
                raise ValueError(f"action for UAV {k} must be an Action")
            delta = np.asarray(action.delta_p, dtype=np.float64)
            if delta.shape != (2,):
                raise ValueError(
                    f"delta_p for UAV {k} must have shape (2,), got {delta.shape}")
            if not np.all(np.isfinite(delta)):
                raise ValueError(f"delta_p for UAV {k} must be finite")
            if isinstance(action.role, bool) or not isinstance(
                action.role, Integral
            ):
                raise ValueError(f"role for UAV {k} must be an integer")
            role = int(action.role)
            if not 0 <= role < 3:
                raise ValueError(
                    f"role for UAV {k} must be one of 0, 1, 2")
            validated[k] = Action(delta_p=delta.copy(), role=role)
        return validated

    def step(self, actions: Dict[int, Action]) -> Tuple[Dict, Dict, Dict, StepInfo]:
        """Execute one simulation frame.

        Args:
            actions: Dict mapping uav_id → Action

        Returns:
            (next_observations, rewards_dict, dones_dict, step_info)
        """
        actions = self._validate_step_actions(actions)
        if self.fc_position is None or self.belief_mgr is None:
            raise RuntimeError(
                "environment must be reset before the first step")
        step_wall_started = time.perf_counter()
        self.t += 1

        # Capture pre-move UAV positions (for potential-based shaping)
        prev_uav_positions = np.array([u.pos.copy() for u in self.uavs])

        # D0.95 L3: override the learned trajectory with the capability-guided
        # movement (receding horizon).  Structure (L2) is re-optimised by P0 at
        # the moved geometry later in this frame.
        movement_compute_started = time.perf_counter()
        analytical_delta: dict = {}
        if self._distributed_gap_coverage_movement_enabled:
            analytical_delta = (
                self._distributed_gap_coverage_movement_delta())
        elif self._distributed_role_capacity_movement_enabled:
            analytical_delta = (
                self._distributed_role_capacity_movement_delta())
        elif (self._distributed_bistatic_bottleneck_movement_enabled
                or self._distributed_bottleneck_matching_movement_enabled
                or self._distributed_greedy_matching_movement_enabled):
            analytical_delta = (
                self._distributed_greedy_matching_movement_delta())
        elif self._distributed_id_movement_enabled:
            analytical_delta = self._distributed_id_movement_delta()
        elif self._analytical_movement_enabled:
            analytical_delta = self._analytical_movement_delta()
        movement_compute_time_s = float(
            time.perf_counter() - movement_compute_started)

        # 1. Apply UAV actions
        uav_positions = np.zeros((self.K, 3), dtype=np.float64)
        uav_velocities = np.zeros((self.K, 3), dtype=np.float64)
        roles = np.zeros(self.K, dtype=np.int32)

        for k in range(self.K):
            delta_p = analytical_delta[k] if k in analytical_delta \
                else actions[k].delta_p
            delta_p = np.asarray(delta_p, dtype=np.float64)
            if delta_p.shape != (2,) or not np.all(np.isfinite(delta_p)):
                raise RuntimeError(
                    f"movement controller produced an invalid delta_p for UAV {k}")
            self.uavs[k].apply_action(
                delta_p, actions[k].role,
                account_radio_energy=not self._joint_isac_power_enabled)
            uav_positions[k] = self.uavs[k].pos
            uav_velocities[k] = self.uavs[k].vel
            roles[k] = self.uavs[k].role
        if (
            not np.all(np.isfinite(uav_positions))
            or not np.all(np.isfinite(uav_velocities))
            or any(not np.isfinite(float(uav.battery)) for uav in self.uavs)
        ):
            raise RuntimeError(
                "non-finite UAV state after applying validated actions")

        # Encode the physically transmitted beacon from current post-action
        # local proprioception. Pending power/target-weight decisions were
        # already queued by submit_actions; no peer or simulator target truth
        # is introduced by moving this local encoding to transmission time.
        if self._hyperedge_enabled:
            self._prepare_hyperedge_submission()

        # Transport the messages chosen from the previous observation. They are
        # receiver-specific and appear only after satisfying link/deadline
        # constraints. Radio energy is deducted from the sending UAV battery.
        comm_processing_started = time.perf_counter()
        comm_stats = self._process_learned_communications(uav_positions)
        comm_processing_wall_time_s = float(
            time.perf_counter() - comm_processing_started)

        # 2. Step target dynamics
        if self.tracking_enabled:
            for target in self.targets:
                target.step()

        target_positions = np.array([t.get_position_3d() for t in self.targets])
        target_velocities = np.array([
            np.array([*t.get_velocity(), 0.0]) for t in self.targets
        ])

        # 3. Compute Deflection matrix. When the policy does not choose roles,
        #    every UAV is a candidate tx and rx; the P0 solver picks roles.
        role_agnostic = not self.learn_roles

        # TRUE-geometry deflection = the physical echo; always the realized signal.
        # D0.89-B: when the structure ranking is analytical, P0 ranks on the
        # per-watt gain a_ijq, so the deflection used for ranking is computed at
        # unit sensing power (the realized powered deflection is recomputed after
        # the LP in the analytical power hook below).
        ranking_power_w = self._current_sensing_power_w
        if self._analytical_structure_ranking_enabled:
            ranking_power_w = np.ones((self.K, self.Q), dtype=np.float64)
        # O3: deterministic U2U physics stays as a dense (i,j,q) tensor until
        # a legacy list consumer actually needs it.  On the deployable
        # belief-ranking path the true unit-power list is never needed, which
        # avoids constructing K(K-1)Q Python objects only to replace all of
        # them after the analytical power solve.
        true_unit_deflection_dense = None
        use_dense_unit_geometry = bool(
            self._analytical_structure_ranking_enabled
            and not self.deflection_computer.use_report_link
            and not self.deflection_computer.use_swerling
        )
        if use_dense_unit_geometry:
            true_unit_deflection_dense = self.deflection_computer.compute_dense(
                uav_positions, uav_velocities,
                target_positions, target_velocities,
                roles, self.fc_position,
                role_agnostic=role_agnostic,
                sensing_power_w=(ranking_power_w
                                 if self._joint_isac_power_enabled else None),
            )
            defer_true_entries = bool(
                self.p0_uses_belief and self.tracking_enabled)
            deflection_entries = (
                [] if defer_true_entries
                else self._deflection_materialization.materialize(
                    true_unit_deflection_dense)
            )
        else:
            deflection_entries = self.deflection_computer.compute(
                uav_positions, uav_velocities,
                target_positions, target_velocities,
                roles, self.fc_position,
                role_agnostic=role_agnostic,
                sensing_power_w=(ranking_power_w
                                 if self._joint_isac_power_enabled else None),
            )
        true_unit_deflection_entries = (
            deflection_entries
            if (self._analytical_structure_ranking_enabled
                and true_unit_deflection_dense is None)
            else None
        )
        self._last_deflection_entries = deflection_entries  # for obs coordination features

        # B6: choose what P0 RANKS candidates on. Oracle = true geometry; deployable
        # = belief estimate (P0 cannot see true targets at deployment). The
        # realized D_q*/P_D below always come from the TRUE deflection of the picks.
        #
        # Layer 3 (Safe P0): when fusion confidence is low, fall back to LOCAL
        # belief for the ranking geometry (not just disable B3). This ensures
        # P0's base deflection comes from safe local estimates, not corrupted
        # fused beliefs. The fused belief is only used when it's trusted.
        if self.p0_uses_belief and self.tracking_enabled:
            # Determine whether to use local or fused belief for ranking
            use_fused_for_ranking = (
                self.neighbor_belief_fusion
                and self._belief_fusion_module is not None
            )

            # Safe P0: check fusion confidence — if any target is untrusted,
            # fall back to local belief for ALL ranking entries (conservative).
            safe_p0_active = bool(getattr(self.cfg.marl, 'p0_safe_fallback', False))
            if use_fused_for_ranking and safe_p0_active and self._trust_manager is not None:
                conf_min = float(getattr(self.cfg.marl, 'p0_fusion_confidence_min', 0.3))
                # Per-target confidence: mean trust over all (k,j) pairs for each target
                conf_q = np.zeros(self.Q, dtype=np.float64)
                for q in range(self.Q):
                    mask = ~np.eye(self.K, dtype=bool)
                    conf_q[q] = float(np.mean(self._trust_manager.trust_score[:, :, q][mask]))
                # If ANY target is below confidence threshold, use local belief
                if np.any(conf_q < conf_min):
                    use_fused_for_ranking = False

            if use_fused_for_ranking:
                fused = self._fuse_beliefs_attention()  # attention-weighted CI fusion
            else:
                fused = self.belief_mgr.mean.mean(axis=0)  # uniform mean (local belief)
            est_pos = np.stack([fused[:, 0], fused[:, 1], np.zeros(self.Q)], axis=1)
            est_vel = np.stack([fused[:, 2], fused[:, 3], np.zeros(self.Q)], axis=1)
            ranking_entries = self.deflection_computer.compute(
                uav_positions, uav_velocities,
                est_pos, est_vel,
                roles, self.fc_position,
                role_agnostic=role_agnostic,
                sensing_power_w=(self._current_sensing_power_w
                                 if self._joint_isac_power_enabled else None),
            )
        else:
            ranking_entries = deflection_entries

        if self._external_structure_edge_values is not None:
            student_values = self._external_structure_edge_values
            ranking_entries = [
                entry._replace(
                    d_eff=float(student_values[
                        int(entry.i), int(entry.j), int(entry.q)]))
                for entry in ranking_entries
            ]
            self._last_isac_metrics.update({
                "structure_student_edge_control": 1.0,
                "structure_student_edge_mean": float(np.mean(
                    student_values[
                        ~np.eye(self.K, dtype=bool), :])),
            })
        unfiltered_ranking_entries = ranking_entries

        # The policy's per-target sensing split is its local commitment. P0 may
        # rank only the resulting subgraph, while realized P_D is still read
        # from true-geometry entries after assignment.
        if self._distributed_target_commitment_enabled:
            explicit_commitment_mask = None
            commitment_scores = self._current_sensing_power_w
            if self._distributed_target_commitment_source == 'qpd':
                explicit_commitment_mask = self._resolve_qpd_commitments()
                # QPD's continuous primal is the graph-ranking confidence.
                # Keep it separate from physical sensing power so the first
                # gate can attribute scheduler and resource effects.
                commitment_scores = self._qpd_primal
            elif self._distributed_target_commitment_source in {
                    'persistent_sensing', 'sent_token'}:
                explicit_commitment_mask = (
                    self._resolve_persistent_token_commitments())
            filtered_ranking_entries, commitment_metrics = (
                filter_deflection_by_local_commitments(
                    ranking_entries,
                    commitment_scores,
                    topk=self._distributed_target_commitment_topk,
                    require_receiver=(
                        self._distributed_target_commitment_require_receiver),
                    mode=self._distributed_target_commitment_mode,
                    soft_floor=(
                        self._distributed_target_commitment_soft_floor),
                    uncertainty_relief=(
                        self._distributed_target_commitment_uncertainty_relief),
                    commitment_mask=explicit_commitment_mask,
                ))
            commitment_metrics['learned_comm_commitment_source'] = (
                self._distributed_target_commitment_source)
            if explicit_commitment_mask is not None:
                if self._distributed_target_commitment_source == 'qpd':
                    commitment_metrics.update(self._qpd_metrics)
                else:
                    commitment_metrics.update(
                        self._persistent_commitment_metrics)
            self._last_isac_metrics.update(commitment_metrics)
            if not (self._p0_maxmin_pairing_enabled
                    and self._p0_maxmin_bypass_commitment_filter):
                ranking_entries = filtered_ranking_entries
            else:
                ranking_entries = unfiltered_ranking_entries

        hyperedge_selected: Tuple[Tuple[int, int, int], ...] = tuple()
        hyperedge_compute_time_s = 0.0
        if self._hyperedge_enabled:
            hyperedge_compute_started = time.perf_counter()
            hyperedge_selected = self._resolve_hyperedge_negotiation()
            hyperedge_compute_time_s = float(
                time.perf_counter() - hyperedge_compute_started)

        # 4. Inner P0 solver with assignment hold (reduces reward non-stationarity)
        hold_frames = (
            self._p0_maxmin_pairing_hold_frames
            if self._p0_maxmin_pairing_enabled
            else getattr(self.cfg.marl, 'assignment_hold_frames', 1)
        )
        topology_repair_solution = None
        topology_repair_started = None
        topology_replacement_count = 0
        topology_invalid_edge_count = 0
        topology_repair_candidate_worst_pd = np.nan
        # Commitments change at communication rate. Reusing an old solution
        # would bypass the current local choices, so always resolve this mode.
        if (self._p0_maxmin_pairing_enabled
                and self._p0_maxmin_event_triggered_enabled
                and self._cached_p0_solution is not None):
            ranking_lookup = {
                (int(entry.i), int(entry.j), int(entry.q)):
                    float(entry.d_eff)
                for entry in ranking_entries
                if float(entry.d_eff) > 0.0
            }
            cached_receiver_D = np.zeros(
                (self.K, self.Q), dtype=np.float64)
            cached_graph_valid = bool(
                self._cached_p0_solution.selected_set)
            invalid_cached_edges = []
            for edge in self._cached_p0_solution.selected_set:
                key = tuple(int(value) for value in edge)
                if key not in ranking_lookup:
                    cached_graph_valid = False
                    invalid_cached_edges.append(key)
                else:
                    cached_receiver_D[key[1], key[2]] += ranking_lookup[key]
            topology_invalid_edge_count = len(invalid_cached_edges)
            maximum_hold_due = (
                self.t - self._last_solve_frame >= hold_frames)
            if (
                invalid_cached_edges
                and self._p0_topology_min_change_repair_enabled
                and self._p0_budget_coupled_structure_enabled
                and self._p0_maxmin_local_fusion_enabled
                and not maximum_hold_due
            ):
                topology_repair_started = time.perf_counter()
                reports_per_receiver = (
                    max(1, int(
                        self.cfg.p0_solver.capacity_per_rx
                        // max(self.cfg.detection.B_q, 1)))
                    if self.ground_communication_enabled
                    else self.Q * self.cfg.detection.K_q_max
                )
                repair = repair_invalid_local_only_edges_min_change(
                    unit_deflection_tensor(
                        ranking_entries, self.K, self.Q),
                    self._cached_p0_solution.selected_set,
                    np.sum(self._current_sensing_power_w, axis=1),
                    P_FA=self.cfg.detection.P_FA,
                    target_pair_limit=self.cfg.detection.K_q_max,
                    reports_per_receiver=reports_per_receiver,
                )
                if repair is not None:
                    topology_repair_candidate_worst_pd = float(
                        np.min(repair.P_D_q))
                    z_repaired = np.zeros(
                        (self.K, self.K, self.Q), dtype=np.int32)
                    for i, j, q in repair.selected_set:
                        z_repaired[i, j, q] = 1
                    topology_repair_solution = P0Solution(
                        z_selected=z_repaired,
                        D_q_star=np.asarray(
                            repair.D_q, dtype=np.float64),
                        U_q=compute_target_utilities(
                            repair.D_q, self.cfg.detection.P_FA),
                        selected_set=list(repair.selected_set),
                        total_bits=float(
                            len(repair.selected_set)
                            * self.cfg.detection.B_q),
                        total_latency=(
                            self._cached_p0_solution.total_latency),
                    )
                    topology_replacement_count = len(invalid_cached_edges)
                    ranking_lookup = {
                        (int(entry.i), int(entry.j), int(entry.q)):
                            float(entry.d_eff)
                        for entry in ranking_entries
                        if float(entry.d_eff) > 0.0
                    }
                    cached_graph_valid = True
                    cached_receiver_D.fill(0.0)
                    for edge in repair.selected_set:
                        i, j, q = (int(value) for value in edge)
                        if (i, j, q) not in ranking_lookup:
                            cached_graph_valid = False
                            break
                        cached_receiver_D[j, q] += ranking_lookup[(i, j, q)]
            if (cached_graph_valid
                    and self._p0_budget_coupled_structure_enabled):
                # Scale-safe event certificate: the ranking entries are
                # per-watt gains, so evaluating their raw sum would silently
                # restore the old fictitious unit-power assumption.  Re-solve
                # the cheap fixed-structure LP under the current per-UAV RF
                # budget and trigger only when that executable structure can
                # no longer meet the task floor.
                cached_gain = np.zeros(
                    (self.K, self.Q), dtype=np.float64)
                certificate_selected = (
                    topology_repair_solution.selected_set
                    if topology_repair_solution is not None
                    else self._cached_p0_solution.selected_set
                )
                for edge in certificate_selected:
                    i, j, q = (int(value) for value in edge)
                    cached_gain[i, q] = ranking_lookup[(i, j, q)]
                try:
                    cached_power = _solve_fixed_structure_maxmin_power_lp(
                        cached_gain,
                        np.sum(self._current_sensing_power_w, axis=1),
                    )
                    cached_D_q = cached_power.deflection
                except RuntimeError:
                    cached_graph_valid = False
                    cached_D_q = np.zeros(self.Q, dtype=np.float64)
            else:
                cached_D_q = (
                    np.max(cached_receiver_D, axis=0)
                    if self._p0_maxmin_local_fusion_enabled
                    else np.sum(cached_receiver_D, axis=0)
                )
            cached_worst_pd = float(np.min(
                compute_detection_probabilities(
                    cached_D_q, self.cfg.detection.P_FA)))
            event_floor = float(getattr(
                self.cfg.marl, "comm_qos_worst_min", 0.60))
            event_due = (
                not cached_graph_valid
                or (
                    self._p0_maxmin_event_qos_enabled
                    and cached_worst_pd < event_floor
                )
            )
            should_resolve = bool(event_due or maximum_hold_due)
        else:
            should_resolve = (
                (
                    self._distributed_target_commitment_enabled
                    and not self._p0_maxmin_pairing_enabled
                )
                or self.t == 1
                or self.t % hold_frames == 0
                or self._cached_p0_solution is None
            )
        if should_resolve:
            p0_solve_started = time.perf_counter()
            # B3: build uncertainty inputs for P0
            # Note: cov and AoI always come from LOCAL beliefs (BeliefManager).
            # The fused belief is NOT used here — B3 scoring is applied on top
            # of local belief geometry, gated by fusion_confidence.
            p0_cov = None; p0_aoi = None
            if self.p0_beta_uncertainty > 0 or self.p0_eta_aoi > 0:
                # Average covariance diagonal per target (from local beliefs)
                p0_cov = np.array([
                    np.mean([np.abs(self.belief_mgr.get_belief(k, q).cov_diag)
                             for k in range(self.K)], axis=0)
                    for q in range(self.Q)
                ])  # (Q, 4)
                p0_aoi = np.array([
                    np.mean([self.belief_mgr.get_belief(k, q).aoi
                             for k in range(self.K)])
                    for q in range(self.Q)
                ])  # (Q,)

            # ── Layer 3: Safe P0 — compute per-target fusion confidence ──
            p0_fusion_conf = None
            p0_conf_min = float(getattr(self.cfg.marl, 'p0_fusion_confidence_min', 0.3))
            safe_p0 = bool(getattr(self.cfg.marl, 'p0_safe_fallback', False))
            if safe_p0 and self._trust_manager is not None:
                # Per-target trust: mean over all off-diagonal (k,j) pairs
                tm = self._trust_manager
                conf = np.zeros(self.Q, dtype=np.float64)
                for q in range(self.Q):
                    mask = ~np.eye(self.K, dtype=bool)
                    conf[q] = float(np.mean(tm.trust_score[:, :, q][mask]))
                p0_fusion_conf = conf

            if (self._hyperedge_enabled
                    and (hyperedge_selected
                         or not self._hyperedge_safety_fallback)):
                lookup = {
                    (int(entry.i), int(entry.j), int(entry.q)):
                        float(entry.d_eff)
                    for entry in unfiltered_ranking_entries
                    if float(entry.d_eff) > 0.0
                }
                # Protocol membership is authoritative for the discrete
                # schedule.  A mutually retained edge that is absent from the
                # current physical table remains scheduled and realizes zero
                # gain below; silently deleting it here would give the
                # distributed protocol an oracle truth-set filter.
                selected = tuple(hyperedge_selected)
                hyperedge_D_q = np.zeros(self.Q, dtype=np.float64)
                z_selected = np.zeros(
                    (self.K, self.K, self.Q), dtype=np.int32)
                for i, j, q in selected:
                    z_selected[i, j, q] = 1
                    hyperedge_D_q[q] += lookup.get((i, j, q), 0.0)
                p0_solution = P0Solution(
                    z_selected=z_selected,
                    D_q_star=hyperedge_D_q,
                    U_q=compute_target_utilities(
                        hyperedge_D_q, self.cfg.detection.P_FA),
                    selected_set=list(selected),
                    total_bits=0.0,
                    total_latency=0.0,
                )
            elif self._p0_maxmin_pairing_enabled:
                reports_per_receiver = (
                    max(1, int(
                        self.cfg.p0_solver.capacity_per_rx
                        // max(self.cfg.detection.B_q, 1)))
                    if self.ground_communication_enabled
                    else self.Q * self.cfg.detection.K_q_max
                )
                if self._dynamic_local_search_coordinator is not None:
                    if self._external_structure_edge_values is None:
                        raise RuntimeError(
                            "dynamic local search requires Student edge values")
                    if self._external_structure_candidate_mask is None:
                        raise RuntimeError(
                            "dynamic local search requires a local candidate mask")
                    edge_value = np.zeros(
                        (self.K, self.K, self.Q), dtype=np.float64)
                    physically_ranked = np.zeros_like(
                        edge_value, dtype=bool)
                    for entry in ranking_entries:
                        edge = (int(entry.i), int(entry.j), int(entry.q))
                        edge_value[edge] = max(float(entry.d_eff), 0.0)
                        physically_ranked[edge] = True
                    local_result = (
                        self._dynamic_local_search_coordinator.resolve(
                            edge_value,
                            self._external_structure_candidate_mask
                            & physically_ranked,
                            target_pair_limit=(
                                self.cfg.detection.K_q_max),
                            reports_per_receiver=reports_per_receiver,
                            p_fa=self.cfg.detection.P_FA,
                            p_d_floor=float(getattr(
                                self.cfg.marl,
                                "comm_qos_worst_min",
                                0.60)),
                            target_priority=(
                                np.exp(np.clip(
                                    float(getattr(
                                        self.cfg.marl,
                                        "p0_maxmin_deficit_priority_gain",
                                        3.0))
                                    * (float(getattr(
                                        self.cfg.marl,
                                        "comm_qos_worst_min",
                                        0.60)) - self._coord_pd_ema),
                                    -6.0,
                                    6.0))
                                if self._coord_pd_ema is not None
                                else np.ones(
                                    self.Q, dtype=np.float64)),
                            target_deficit=(
                                float(getattr(
                                    self.cfg.marl,
                                    "comm_qos_worst_min",
                                    0.60)) - self._coord_pd_ema
                                if self._coord_pd_ema is not None
                                else None),
                            frame_index=int(self.t),
                        )
                    )
                    selected_array = local_result.selected
                    selected = [
                        tuple(int(value) for value in edge)
                        for edge in np.argwhere(selected_array)
                    ]
                    maxmin_D_q = np.sum(
                        np.where(selected_array, edge_value, 0.0),
                        axis=(0, 1),
                    )
                    self._last_isac_metrics.update(
                        local_result.diagnostics)
                    self._last_isac_metrics.update({
                        "local_search_enabled": 1.0,
                        "local_search_resolved": 1.0,
                        "local_search_candidate_edges": int(np.sum(
                            self._external_structure_candidate_mask
                            & physically_ranked)),
                    })
                else:
                    selected, maxmin_D_q = solve_maxmin_single_role_pairs(
                        ranking_entries,
                        num_uavs=self.K,
                        num_targets=self.Q,
                        target_pair_limit=self.cfg.detection.K_q_max,
                        reports_per_receiver=reports_per_receiver,
                        p_fa=self.cfg.detection.P_FA,
                        p_d_floor=float(getattr(
                            self.cfg.marl, "comm_qos_worst_min", 0.60)),
                        target_priority=(
                            self._last_analytical_dual_prices
                            if (
                                self._analytical_structure_ranking_enabled
                                and self._last_analytical_dual_prices is not None
                            )
                            else (
                                np.exp(np.clip(
                                    float(getattr(
                                        self.cfg.marl,
                                        "p0_maxmin_deficit_priority_gain",
                                        3.0,
                                    ))
                                    * (
                                        float(getattr(
                                            self.cfg.marl,
                                            "comm_qos_worst_min",
                                            0.60,
                                        ))
                                        - self._coord_pd_ema
                                    ),
                                    -6.0,
                                    6.0,
                                ))
                                if self._coord_pd_ema is not None
                                else np.ones(self.Q, dtype=np.float64)
                            )
                        ),
                        fusion_mode=(
                            "local_only"
                            if self._p0_maxmin_local_fusion_enabled
                            else "central_oracle"
                        ),
                        per_uav_sensing_budget_w=(
                            np.sum(self._current_sensing_power_w, axis=1)
                            if (
                                self._p0_budget_coupled_structure_enabled
                                and
                                self._p0_maxmin_local_fusion_enabled
                                and self._analytical_sensing_power_enabled
                            )
                            else None
                        ),
                        joint_time_limit_s=(
                            self._p0_budget_coupled_time_limit_s),
                        joint_secondary_objective_enabled=(
                            self._p0_budget_coupled_secondary_enabled),
                        joint_solver_mode=(
                            self._p0_budget_coupled_solver_mode),
                        task_qos_floors=(
                            getattr(
                                self.cfg.marl,
                                "p0_legacy_guard_qos_floors",
                                None,
                            )
                            or getattr(
                                self.cfg.marl,
                                "task_constrained_qos_floors",
                                None,
                            )
                        ),
                    )
                z_selected = np.zeros(
                    (self.K, self.K, self.Q), dtype=np.int32)
                for i, j, q in selected:
                    z_selected[i, j, q] = 1
                p0_solution = P0Solution(
                    z_selected=z_selected,
                    D_q_star=maxmin_D_q,
                    U_q=compute_target_utilities(
                        maxmin_D_q, self.cfg.detection.P_FA),
                    selected_set=list(selected),
                    total_bits=float(
                        len(selected) * self.cfg.detection.B_q),
                    total_latency=0.0,
                )
            else:
                p0_solution = self.inner_solver.solve(
                    ranking_entries, Q=self.Q, K=self.K,
                    enforce_single_role=(
                        role_agnostic
                        and not self._multistatic_subslot_enabled),
                    belief_cov_diag=p0_cov,
                    belief_aoi=p0_aoi,
                    beta_uncertainty=self.p0_beta_uncertainty,
                    eta_aoi=self.p0_eta_aoi,
                    fusion_confidence=p0_fusion_conf,
                    fusion_confidence_min=p0_conf_min,
                    du_enabled=bool(getattr(self.cfg.marl, 'du_enabled', False)),
                    du_ambiguity_threshold=float(getattr(self.cfg.marl, 'du_ambiguity_threshold', 3.0)),
                    du_ambiguity_bonus=float(getattr(self.cfg.marl, 'du_ambiguity_bonus', 0.1)),
                )
            if not self.ground_communication_enabled:
                p0_solution = p0_solution._replace(
                    total_bits=0.0, total_latency=0.0)
            self._cached_p0_solution = p0_solution
            self._assignment_switched = True
            self._last_solve_frame = self.t
            self._last_p0_solve_time_s = float(
                time.perf_counter() - p0_solve_started)
        elif topology_repair_solution is not None:
            p0_solution = topology_repair_solution
            self._assignment_switched = True
            self._last_p0_solve_time_s = float(
                time.perf_counter() - topology_repair_started)
        else:
            p0_solution = self._cached_p0_solution
            self._assignment_switched = False
            self._last_p0_solve_time_s = 0.0
        self._last_isac_metrics.update({
            "p0_topology_invalid_edge_count": float(
                topology_invalid_edge_count),
            "p0_topology_invalid_frame": float(
                topology_invalid_edge_count > 0),
            "p0_topology_min_change_repair_applied": float(
                topology_repair_solution is not None and not should_resolve),
            "p0_topology_min_change_replacement_count": float(
                topology_replacement_count
                if topology_repair_solution is not None and not should_resolve
                else 0),
            "p0_topology_repair_candidate_worst_pd": float(
                topology_repair_candidate_worst_pd),
            "p0_topology_full_fallback": float(
                topology_invalid_edge_count > 0 and should_resolve),
        })
        # A topology bridge is a fast physical-execution repair, not a slow
        # L2 commitment.  Feeding its one-frame edge into the next frame's L3
        # geometry controller violates the intended time-scale separation and
        # can turn a sparse DD-support loss into a persistent trajectory shift.
        # Keep L3 anchored to the hold-window structure; only a scheduled/full
        # P0 solve is allowed to change that slow coordination state.
        self._last_selected_set = (
            self._cached_p0_solution.selected_set
            if topology_repair_solution is not None and not should_resolve
            else p0_solution.selected_set
        )

        # ── D0.89: analytical inner sensing power ──
        # The learned per-target sensing head is ignored for execution; after
        # P0 fixes role/owner/edge, the fixed-owner max-min power LP allocates
        # sensing power.  The actor still controls the sensing *budget* through
        # P_comm (b_i = 1 - P_comm).  Only the sensing power changes; motion,
        # structure and Token decisions are untouched.
        if self._analytical_sensing_power_enabled:
            if true_unit_deflection_dense is not None:
                coefficient = (
                    self._per_watt_coefficient_from_dense_deflection(
                        true_unit_deflection_dense))
            else:
                coefficient = self._per_watt_coefficient_from_entries(
                    deflection_entries)
            selected = tuple(
                tuple(int(v) for v in edge)
                for edge in p0_solution.selected_set
            )
            budget = np.sum(self._current_sensing_power_w, axis=1)
            reserve = None
            if self._analytical_sensing_power_reserve_pd > 0.0:
                reserve = minimum_deflection_for_detection_probability(
                    np.full(self.Q, float(
                        self._analytical_sensing_power_reserve_pd)),
                    self.cfg.detection.P_FA,
                )
            if self._temporal_unrolled_power_enabled:
                temporal_metrics = {
                    'temporal_unrolled_power_target_requirement': (
                        reserve.copy()),
                    'temporal_unrolled_power_total_rf_budget_w': np.full(
                        self.K, self._isac_total_power_w,
                        dtype=np.float64),
                    'temporal_unrolled_power_sensing_cap_w': np.full(
                        self.K, self._sensing_power_cap_w,
                        dtype=np.float64),
                }
                if self._temporal_feasible_structure_enabled:
                    executed_edge_mask = np.zeros(
                        (self.K, self.K, self.Q), dtype=bool)
                    for transmitter, receiver, target in selected:
                        executed_edge_mask[
                            transmitter, receiver, target] = True
                    reports_per_receiver = (
                        max(1, int(
                            self.cfg.p0_solver.capacity_per_rx
                            // max(self.cfg.detection.B_q, 1)))
                        if self.ground_communication_enabled
                        else self.Q * self.cfg.detection.K_q_max
                    )
                    temporal_metrics.update({
                        # Simulator coefficient is exposed to the training
                        # computation only (CTDE); it is never appended to a
                        # decentralized actor observation.
                        'temporal_feasible_structure_coefficient_per_watt': (
                            coefficient.copy()),
                        'temporal_feasible_structure_executed_edge_mask': (
                            executed_edge_mask),
                        'temporal_feasible_structure_target_pair_limit': int(
                            self.cfg.detection.K_q_max),
                        'temporal_feasible_structure_reports_per_receiver': int(
                            reports_per_receiver),
                    })
                self._last_isac_metrics.update(temporal_metrics)
            try:
                gain, _owners = fixed_owner_gain_matrix(coefficient, selected)
            except (
                IncompleteFixedOwnerStructureError,
                NonUniqueFixedOwnerStructureError,
            ):
                gain = None
            except ValueError as error:
                raise RuntimeError(
                    "invalid fixed-owner state for analytical sensing power"
                ) from error
            if gain is not None:
                self._last_analytical_gain = gain.copy()
                self._last_analytical_budget = budget.copy()
                actor_sensing_power_proposal = (
                    self._current_sensing_power_w.copy())
                bounded_primal_dual_power = None
                replicated_power = None
                if self._distributed_primal_dual_power_enabled:
                    if self._distributed_primal_dual_power_controller is None:
                        self._distributed_primal_dual_power_controller = (
                            self._new_distributed_primal_dual_controller())
                    bounded_primal_dual_power = (
                        self._distributed_primal_dual_power_controller.solve(
                            gain,
                            reserve,
                            budget,
                            gain > 0.0,
                            actor_proposal_power_w=(
                                actor_sensing_power_proposal
                                if self._temporal_unrolled_power_enabled
                                else None),
                            actor_proposal_mix=float(getattr(
                                self.cfg.marl,
                                'temporal_unrolled_power_warm_start_mix',
                                0.50)),
                        ))
                    self._current_sensing_power_w = (
                        bounded_primal_dual_power.power_w.copy())
                    self._last_analytical_power_balance_error = float(np.max(
                        np.maximum(
                            np.sum(self._current_sensing_power_w, axis=1)
                            - budget,
                            0.0,
                        )))
                    self._last_analytical_dual_prices = (
                        bounded_primal_dual_power.target_prices.copy())
                    self._last_isac_metrics.update({
                        'distributed_primal_dual_power_enabled': 1.0,
                        'distributed_primal_dual_power_warm_started': float(
                            bounded_primal_dual_power.warm_started),
                        'distributed_primal_dual_power_actor_proposal_used': float(
                            bounded_primal_dual_power.actor_proposal_used),
                        'distributed_primal_dual_power_actor_proposal_mix': float(
                            bounded_primal_dual_power.actor_proposal_mix),
                        'distributed_primal_dual_power_candidate_accepted': float(
                            bounded_primal_dual_power.candidate_accepted),
                        'distributed_primal_dual_power_converged': float(
                            bounded_primal_dual_power.converged),
                        'distributed_primal_dual_power_rounds': float(
                            bounded_primal_dual_power.iterations),
                        'distributed_primal_dual_power_baseline_violation': float(
                            bounded_primal_dual_power.baseline_primal_violation),
                        'distributed_primal_dual_power_candidate_violation': float(
                            bounded_primal_dual_power.candidate_primal_violation),
                        'distributed_primal_dual_power_selected_violation': float(
                            bounded_primal_dual_power.selected_primal_violation),
                        'distributed_primal_dual_power_budget_violation_w': float(
                            bounded_primal_dual_power.power_budget_violation_w),
                        'distributed_primal_dual_power_consensus_residual': float(
                            bounded_primal_dual_power.consensus_residual),
                        'distributed_primal_dual_power_candidate_stationarity': float(
                            bounded_primal_dual_power.candidate_primal_stationarity),
                        'distributed_primal_dual_power_candidate_dual_residual': float(
                            bounded_primal_dual_power.candidate_dual_residual),
                        # Keep the vector for post-hoc target-wise audits and
                        # expose scalar summaries for rollout CSVs.
                        'distributed_primal_dual_power_target_prices': (
                            bounded_primal_dual_power.target_prices.copy()),
                        'distributed_primal_dual_power_target_price_mean': float(
                            np.mean(bounded_primal_dual_power.target_prices)),
                        'distributed_primal_dual_power_target_price_max': float(
                            np.max(bounded_primal_dual_power.target_prices,
                                   initial=0.0)),
                        'distributed_primal_dual_power_fallback': float(
                            not bounded_primal_dual_power.candidate_accepted),
                    })
                    if self._temporal_unrolled_power_enabled:
                        # Training-only CTDE tensors.  These are the physical
                        # coefficients used by the executed inner problem, not
                        # observations or teacher actions.  The actor never
                        # receives them during decentralized execution.
                        self._last_isac_metrics.update({
                            'temporal_unrolled_power_gain_per_watt': (
                                gain.copy()),
                            'temporal_unrolled_power_target_requirement': (
                                reserve.copy()),
                            'temporal_unrolled_power_feasible_mask': (
                                gain > 0.0),
                        })
                elif self._distributed_replicated_power_enabled:
                    # Every transmitter independently solves from its own
                    # quantized public-state cache and executes only its row.
                    # The fixed sensing PA cap is public; the final row is
                    # projected onto the actual post-communication budget.
                    public_budget = np.full(
                        self.K,
                        float(self._sensing_power_cap_w),
                        dtype=np.float64,
                    )
                    incomplete_prior = None
                    if (
                        self._distributed_replicated_power_local_range_fallback_enabled
                    ):
                        transmitter_positions = np.asarray([
                            uav.pos for uav in self.uavs
                        ], dtype=np.float64)
                        if self._distributed_no_truth_fail_closed:
                            # Strict no-truth closure (audit advice/001 P0,
                            # 2026-08-26): the historical local-range fallback
                            # read ``self.targets`` (simulator truth).  Failed
                            # closed, the prior is skipped and the unknown-target
                            # reserve alone covers incomplete public views.
                            incomplete_prior = None
                        else:
                            fallback_targets = np.asarray([
                                target.get_position_3d()
                                for target in self.targets
                            ], dtype=np.float64)
                            incomplete_prior = (
                                local_transmitter_range_minimax_share(
                                    transmitter_positions,
                                    fallback_targets,
                                )
                            )
                    deadline_safe_row_gain = None
                    if self._composable_certificate_enabled:
                        deadline_safe_row_gain = np.asarray([
                            self._composable_certificate_gain_views[node, node]
                            for node in range(self.K)
                        ], dtype=np.float64)
                    replicated_solve_started = time.perf_counter()
                    replicated_power = replicated_local_row_maxmin_power(
                        self._hyperedge_public_gain_views,
                        public_budget,
                        budget,
                        self._distributed_replicated_power_unknown_target_reserve,
                        incomplete_prior,
                        previous_local_power_w=(
                            self._distributed_replicated_local_power_cache),
                        previous_local_prices=(
                            self._distributed_replicated_local_price_cache),
                        previous_local_cache_valid=(
                            self._distributed_replicated_local_cache_valid),
                        reuse_relative_tolerance=(
                            self._distributed_replicated_power_reuse_relative_tolerance),
                        parallel_executor=(
                            self._distributed_replicated_power_executor),
                        parallel_fallback_to_serial=(
                            self._distributed_replicated_power_parallel_fallback_to_serial),
                        parallel_failure_mode=(
                            self._distributed_replicated_power_parallel_failure_mode),
                        deadline_incumbent_relative_tolerance=(
                            self._distributed_replicated_power_deadline_incumbent_tolerance),
                        force_deadline_fallback=(
                            self._distributed_replicated_power_executor_warmup_failed),
                        deadline_safe_row_gain_per_watt=(
                            deadline_safe_row_gain),
                        history_reserve_deflection_cap=(
                            self._distributed_replicated_power_history_reserve_deflection),
                    )
                    replicated_solve_time_s = float(
                        time.perf_counter() - replicated_solve_started)
                    if (
                        replicated_power.parallel_fallback_to_serial
                        and self._distributed_replicated_power_executor
                        is not None
                    ):
                        self._distributed_replicated_power_consecutive_failures += 1
                        if (
                            self._distributed_replicated_power_consecutive_failures
                            >= self._distributed_replicated_power_max_consecutive_failures
                        ):
                            # A single late batch may be transient; repeated
                            # failures are treated as a broken pool and latched
                            # into the no-LP deadline fallback.
                            self._distributed_replicated_power_executor.close(
                                wait=False)
                            self._distributed_replicated_power_executor = None
                            self._distributed_replicated_power_executor_warmup_failed = True
                    elif replicated_power.parallel_execution_used:
                        self._distributed_replicated_power_consecutive_failures = 0
                    self._current_sensing_power_w = (
                        blend_row_feasible_power_with_inertia(
                            replicated_power.power_w,
                            self._distributed_replicated_previous_power,
                            budget,
                            self._distributed_replicated_power_inertia,
                        ))
                    deadline_actual_deflection_floor = 0.0
                    deadline_actual_pd_floor = float(
                        compute_detection_probabilities(
                            np.asarray([0.0], dtype=np.float64),
                            self.cfg.detection.P_FA,
                        )[0])
                    deadline_safe_target_coverage = 0.0
                    deadline_safe_min_contributors = 0.0
                    deadline_safe_active_transmitter_fraction = 0.0
                    deadline_sparse_harmonic_pd_floor = float(
                        self.cfg.detection.P_FA)
                    deadline_shadow_minimum_reserve_fraction = float("inf")
                    deadline_shadow_reserve_feasible = 0.0
                    deadline_shadow_global_lp_pd_floor = float(
                        self.cfg.detection.P_FA)
                    deadline_shadow_global_lp_time_s = 0.0
                    deadline_shadow_harmonic_approximation_ratio = 0.0
                    if self._composable_certificate_enabled:
                        executor_gain_lower = deadline_safe_row_gain
                        safe_support = executor_gain_lower > 0.0
                        contributors = np.sum(safe_support, axis=0)
                        deadline_safe_target_coverage = float(np.mean(
                            contributors > 0))
                        deadline_safe_min_contributors = float(np.min(
                            contributors))
                        deadline_safe_active_transmitter_fraction = float(
                            np.mean(np.any(safe_support, axis=1)))
                        deadline_actual_deflection_floor = float(np.min(
                            np.sum(
                                executor_gain_lower
                                * self._current_sensing_power_w,
                                axis=0,
                            )
                        ))
                        deadline_actual_pd_floor = float(
                            compute_detection_probabilities(
                                np.asarray(
                                    [deadline_actual_deflection_floor],
                                    dtype=np.float64,
                                ),
                                self.cfg.detection.P_FA,
                            )[0])
                        actual_budget = np.sum(
                            self._current_sensing_power_w, axis=1)
                        harmonic_shadow_power = sparse_harmonic_row_power(
                            executor_gain_lower,
                            actual_budget,
                            zero_support_fallback_power_w=(
                                self._current_sensing_power_w),
                        )
                        base_target_lower = np.sum(
                            executor_gain_lower
                            * self._current_sensing_power_w,
                            axis=0,
                        )
                        harmonic_target_lower = np.sum(
                            executor_gain_lower * harmonic_shadow_power,
                            axis=0,
                        )
                        harmonic_floor = float(np.min(
                            harmonic_target_lower))
                        deadline_sparse_harmonic_pd_floor = float(
                            compute_detection_probabilities(
                                np.asarray([harmonic_floor]),
                                self.cfg.detection.P_FA,
                            )[0])
                        if (
                            self._distributed_replicated_power_certificate_shadow_global_lp_enabled
                            and np.all(contributors > 0)
                        ):
                            shadow_lp_started = time.perf_counter()
                            shadow_global = (
                                _solve_fixed_structure_maxmin_power_lp(
                                    executor_gain_lower,
                                    actual_budget,
                                ))
                            deadline_shadow_global_lp_time_s = float(
                                time.perf_counter() - shadow_lp_started)
                            deadline_shadow_global_lp_pd_floor = float(
                                compute_detection_probabilities(
                                    np.asarray([
                                        shadow_global.worst_deflection
                                    ], dtype=np.float64),
                                    self.cfg.detection.P_FA,
                                )[0])
                            deadline_shadow_harmonic_approximation_ratio = float(
                                np.clip(
                                    harmonic_floor
                                    / max(
                                        shadow_global.worst_deflection,
                                        1.0e-300,
                                    ),
                                    0.0,
                                    1.0,
                                ))
                        required_floor = float(
                            minimum_deflection_for_detection_probability(
                                np.asarray([float(getattr(
                                    self.cfg.marl,
                                    'comm_qos_worst_min',
                                    0.60,
                                ))]),
                                self.cfg.detection.P_FA,
                            )[0])
                        deficient = base_target_lower < required_floor
                        if not np.any(deficient):
                            deadline_shadow_minimum_reserve_fraction = 0.0
                            deadline_shadow_reserve_feasible = 1.0
                        else:
                            improvement = (
                                harmonic_target_lower - base_target_lower)
                            if np.all(improvement[deficient] > 0.0):
                                required_fraction = float(np.max(
                                    (required_floor
                                     - base_target_lower[deficient])
                                    / improvement[deficient]
                                ))
                                deadline_shadow_minimum_reserve_fraction = (
                                    required_fraction)
                                deadline_shadow_reserve_feasible = float(
                                    required_fraction <= 1.0)
                        self._composable_certificate_local_lower = (
                            conservative_row_contribution(
                                self._current_sensing_power_w,
                                executor_gain_lower,
                            )
                        )
                        executor_gain_upper = np.asarray([
                            self._composable_certificate_gain_upper_views[
                                node, node]
                            for node in range(self.K)
                        ], dtype=np.float64)
                        self._composable_certificate_local_dual_upper = (
                            uniform_price_row_dual_upper(
                                executor_gain_upper,
                                budget,
                            )
                        )
                        if (
                            self._composable_certificate_targetwise_upper_enabled
                        ):
                            self._composable_certificate_local_targetwise_upper = (
                                targetwise_price_row_dual_upper(
                                    executor_gain_upper,
                                    budget,
                                )
                            )
                        self._composable_certificate_local_frame.fill(
                            int(self.t))
                    self._distributed_replicated_previous_power = (
                        self._current_sensing_power_w.copy())
                    self._distributed_replicated_local_power_cache = (
                        replicated_power.local_full_power_w.copy())
                    self._distributed_replicated_local_price_cache = (
                        replicated_power.local_prices.copy())
                    self._distributed_replicated_local_cache_valid = (
                        replicated_power.local_cache_valid.copy())
                    self._last_analytical_power_balance_error = float(np.max(
                        np.abs(np.sum(
                            self._current_sensing_power_w, axis=1) - budget)))
                    self._last_analytical_dual_prices = None
                    self._lex_mode = 'distributed_replicated'
                    self._lex_t_star = None
                    self._last_isac_metrics.update({
                        'distributed_replicated_power_enabled': 1.0,
                        'distributed_replicated_power_inertia': float(
                            self._distributed_replicated_power_inertia),
                        'distributed_replicated_power_history_reserve_pd': (
                            self._distributed_replicated_power_history_reserve_pd),
                        'distributed_replicated_power_history_reserve_used_fraction': (
                            replicated_power.history_reserve_used_fraction),
                        'distributed_replicated_power_unknown_target_reserve': (
                            float(
                                self._distributed_replicated_power_unknown_target_reserve)),
                        'distributed_replicated_power_robust_gain_mix': float(
                            self._distributed_replicated_power_robust_gain_mix),
                        'distributed_replicated_power_local_range_fallback': (
                            float(
                                self._distributed_replicated_power_local_range_fallback_enabled)),
                        'distributed_replicated_power_solve_time_s': (
                            replicated_solve_time_s),
                        'distributed_replicated_power_per_node_solve_time_s': (
                            replicated_solve_time_s / max(self.K, 1)),
                        # The simulator executes private nodes serially in one
                        # process.  A real distributed fleet executes them in
                        # parallel, so the power-layer decision critical path
                        # is the slowest node, not the serial wrapper time.
                        'distributed_replicated_power_node_mean_compute_time_s': (
                            float(np.mean(
                                replicated_power.local_compute_time_s))),
                        'distributed_replicated_power_node_p95_compute_time_s': (
                            float(np.percentile(
                                replicated_power.local_compute_time_s, 95))),
                        'distributed_replicated_power_parallel_critical_path_s': (
                            float(np.max(
                                replicated_power.local_compute_time_s))),
                        'distributed_replicated_power_common_view': float(
                            replicated_power.common_view),
                        'distributed_replicated_power_unique_local_problems': (
                            float(
                                replicated_power.unique_local_problem_count)),
                        'distributed_replicated_power_exact_dedup_fraction': (
                            float(
                                1.0
                                - replicated_power.unique_local_problem_count
                                / max(self.K, 1))),
                        'distributed_replicated_power_process_parallel_used': (
                            float(replicated_power.parallel_execution_used)),
                        'distributed_replicated_power_process_workers': float(
                            replicated_power.parallel_worker_count),
                        'distributed_replicated_power_parallel_batch_wall_s': (
                            float(replicated_power.parallel_batch_wall_time_s)),
                        'distributed_replicated_power_parallel_fallback': float(
                            replicated_power.parallel_fallback_to_serial
                            or self._distributed_replicated_power_executor_warmup_failed),
                        'distributed_replicated_power_worker_warmup_time_s': (
                            float(
                                self._distributed_replicated_power_executor_warmup_time_s)),
                        'distributed_replicated_power_deadline_incumbent_fraction': (
                            float(
                                replicated_power.deadline_incumbent_used_fraction)),
                        'distributed_replicated_power_deadline_incumbent_certificate_fraction': (
                            float(
                                replicated_power.deadline_incumbent_certificate_fraction)),
                        'distributed_replicated_power_deadline_uniform_fraction': (
                            float(
                                replicated_power.deadline_uniform_fallback_fraction)),
                        'distributed_replicated_power_deadline_harmonic_fraction': (
                            float(
                                replicated_power.deadline_harmonic_fallback_fraction)),
                        # Recompute after post-radio budget projection and
                        # inertia.  Target responsibility composes the sparse
                        # row reports into min_q sum_k a^-_{kq} p_{kq}.
                        'distributed_replicated_power_deadline_composable_deflection_floor': (
                            deadline_actual_deflection_floor),
                        'distributed_replicated_power_deadline_composable_pd_floor': (
                            deadline_actual_pd_floor),
                        'distributed_replicated_power_deadline_safe_target_coverage': (
                            deadline_safe_target_coverage),
                        'distributed_replicated_power_deadline_safe_min_contributors': (
                            deadline_safe_min_contributors),
                        'distributed_replicated_power_deadline_safe_active_transmitter_fraction': (
                            deadline_safe_active_transmitter_fraction),
                        # Shadow only: this does not alter execution. It asks
                        # whether a same-frame convex reserve toward the
                        # separable harmonic allocation could certify the
                        # configured Worst-P_D floor.
                        'distributed_replicated_power_deadline_sparse_harmonic_shadow_pd_floor': (
                            deadline_sparse_harmonic_pd_floor),
                        'distributed_replicated_power_deadline_shadow_minimum_reserve_fraction': (
                            deadline_shadow_minimum_reserve_fraction),
                        'distributed_replicated_power_deadline_shadow_reserve_feasible': (
                            deadline_shadow_reserve_feasible),
                        'distributed_replicated_power_deadline_shadow_global_lp_enabled': (
                            float(
                                self._distributed_replicated_power_certificate_shadow_global_lp_enabled)),
                        'distributed_replicated_power_deadline_shadow_global_lp_pd_floor': (
                            deadline_shadow_global_lp_pd_floor),
                        'distributed_replicated_power_deadline_shadow_global_lp_time_s': (
                            deadline_shadow_global_lp_time_s),
                        'distributed_replicated_power_deadline_shadow_harmonic_approximation_ratio': (
                            deadline_shadow_harmonic_approximation_ratio),
                        'distributed_replicated_power_consecutive_process_failures': (
                            float(
                                self._distributed_replicated_power_consecutive_failures)),
                        'distributed_replicated_power_full_view_fraction': (
                            float(np.mean(
                                self._hyperedge_public_full_views))),
                        'distributed_replicated_power_local_coverage': float(
                            np.mean(
                                replicated_power.local_full_coverage)),
                        'distributed_replicated_power_local_worst_mean': float(
                            np.mean(
                                replicated_power.local_worst_deflection)),
                        'distributed_replicated_power_local_worst_min': float(
                            np.min(
                                replicated_power.local_worst_deflection)),
                        'distributed_replicated_power_reuse_tolerance': float(
                            self._distributed_replicated_power_reuse_relative_tolerance),
                        'distributed_replicated_power_resolve_fraction': float(
                            np.mean(replicated_power.local_resolved)),
                        'distributed_replicated_power_reuse_fraction': float(
                            np.mean(
                                replicated_power.local_cache_valid
                                & ~replicated_power.local_resolved)),
                        'distributed_replicated_power_reuse_gap_max': float(
                            np.max(np.where(
                                np.isfinite(
                                    replicated_power.local_relative_gap),
                                replicated_power.local_relative_gap,
                                0.0,
                            ))),
                    })
                # D1.1-D live covertness (T3): max-min subject to the
                # counter-detection hard bound P_{D,w}^I <= eps.  When the
                # constrained LP is feasible it REPLACES the power path; the
                # dual price mu is the opponent-detection cost per watt.
                intercept_lp = None
                if (
                    bounded_primal_dual_power is None
                    and replicated_power is None
                    and self._intercept_constrained_power_enabled
                ):
                    intercept_lp = self._solve_intercept_power(gain, budget)
                    if intercept_lp is None:
                        raise RuntimeError(
                            "covertness-constrained power solve failed; "
                            "refusing unconstrained fallback")
                if bounded_primal_dual_power is not None:
                    pass
                elif replicated_power is not None:
                    pass
                elif intercept_lp is not None:
                    lp = intercept_lp
                    self._current_sensing_power_w = lp.power_w.copy()
                    self._last_analytical_power_balance_error = float(np.max(
                        np.abs(np.sum(lp.power_w, axis=1) - budget)))
                elif self._task_constrained_power_enabled:
                    # D0.93-A1: task-constrained power (QoS floors as hard
                    # constraints via the PWL LP).  If gamma* <= 1 the returned
                    # power satisfies worst+bottom-k+steady; else fall back to
                    # reserve-first max-min (best effort).
                    from uav_isac.coordination.capability import (
                        capability_gauge_pwl_lp_full,
                    )
                    from uav_isac.coordination.pwl_pd import (
                        chord_lower_bound,
                        curvature_breakpoints,
                    )
                    qos_floors = tuple(float(v) for v in getattr(
                        self.cfg.marl, 'task_constrained_qos_floors',
                        (0.60, 0.70, 0.80, 3)))
                    worst_floor = qos_floors[0]
                    weak3_floor = qos_floors[1]
                    steady_floor = qos_floors[2]
                    k_tail = max(1, int(qos_floors[3]))
                    d_min = float(minimum_deflection_for_detection_probability(
                        np.asarray([worst_floor]),
                        self.cfg.detection.P_FA)[0])
                    ceiling = np.sum(gain * budget[:, None], axis=0)
                    if np.any(ceiling < d_min - 1e-9):
                        lp = None
                    else:
                        d_max = float(np.max(ceiling)) + 1.0
                        bps = curvature_breakpoints(
                            self.cfg.detection.P_FA, d_min, d_max, epsilon=1e-3)
                        cs, ci = chord_lower_bound(
                            self.cfg.detection.P_FA, bps)
                        out = capability_gauge_pwl_lp_full(
                            gain, budget, self.cfg.detection.P_FA,
                            (worst_floor, weak3_floor, steady_floor, k_tail),
                            cs, ci, d_min)
                        lp = None
                        if out is not None and out[0] <= 1.0 + 1e-6:
                            gamma_val, p_star, pi_star = out
                            from uav_isac.coordination.maxmin_power import (
                                MaxMinPowerResult,
                            )
                            # D1.1-A (advice 010): lexicographic mode maximises
                            # the worst within the feasible region instead of
                            # executing the gauge's satisficing allocation.
                            if (getattr(self.cfg.marl,
                                        'task_constrained_mode', 'gauge')
                                    == 'lexicographic'):
                                from uav_isac.coordination.capability import (
                                    qos_constrained_maxmin_lp,
                                )
                                st = qos_constrained_maxmin_lp(
                                    gain, budget, self.cfg.detection.P_FA,
                                    (worst_floor, weak3_floor, steady_floor,
                                     k_tail),
                                    cs, ci, d_min)
                                if st is not None:
                                    _t_star, p_star, _d_star = st
                                    gamma_val = float(_t_star)
                                    self._lex_mode = 'stage_b'
                                    self._lex_t_star = float(_t_star)
                                    # Stage B returns no dual; expose the
                                    # true max-min shadow prices instead of
                                    # mislabelling the deflection vector.
                                    from uav_isac.coordination.maxmin_power import (
                                        optimal_maxmin_dual_prices,
                                        entropic_maxmin_dual_prices,
                                    )
                                    pi_star, _ = (
                                        entropic_maxmin_dual_prices(
                                            gain, budget,
                                            self._entropic_dual_tau)
                                        if self._entropic_dual_price_enabled
                                        else optimal_maxmin_dual_prices(
                                            gain, budget))
                                else:
                                    self._lex_mode = 'stage_b_infeasible'
                                    self._lex_t_star = None
                            else:
                                self._lex_mode = 'gauge'
                                self._lex_t_star = None
                            deflection = np.sum(gain * p_star, axis=0)
                            lp = MaxMinPowerResult(
                                power_w=p_star,
                                deflection=deflection,
                                worst_deflection=float(np.min(deflection)),
                                prices=pi_star,
                                dual_upper_bound=gamma_val,
                                primal_dual_gap=0.0,
                                rounds=0,
                                worst_history=(float(np.min(deflection)),),
                            )
                    if lp is None:
                        from uav_isac.coordination.maxmin_power import (
                            solve_fixed_structure_maxmin_power_lp,
                        )
                        lp = solve_fixed_structure_maxmin_power_lp(
                            gain, budget, minimum_deflection=reserve)
                        self._lex_mode = 'reserve_fallback'
                        self._lex_t_star = None
                    self._current_sensing_power_w = lp.power_w.copy()
                    self._last_analytical_power_balance_error = float(np.max(
                        np.abs(np.sum(lp.power_w, axis=1) - budget)))
                elif self._bargaining_objective_enabled:
                    # D0.92: reference-normalized bargaining power allocation.
                    # The reserve (if any) is dropped: opportunity fairness
                    # already encodes "each target gets a fair share of its own
                    # headroom", so there is no separate reserve counterbalance.
                    from uav_isac.coordination.bargaining_power import (
                        solve_fixed_structure_bargaining_lp,
                    )
                    lp = solve_fixed_structure_bargaining_lp(gain, budget)
                    self._last_bargaining_value = lp.bargaining_value
                    self._last_analytical_dual_prices = lp.prices
                    self._current_sensing_power_w = lp.power_w.copy()
                    self._last_analytical_power_balance_error = float(
                        lp.power_balance_error_w)
                else:
                    from uav_isac.coordination.maxmin_power import (
                        solve_fixed_structure_maxmin_power_lp,
                    )
                    try:
                        lp = solve_fixed_structure_maxmin_power_lp(
                            gain, budget, minimum_deflection=reserve)
                    except RuntimeError:
                        # Reserve infeasible: fall back to pure max-min and
                        # record the shortfall so the reward can penalise it
                        # rather than crashing the environment.
                        lp = solve_fixed_structure_maxmin_power_lp(gain, budget)
                        if reserve is not None:
                            shortfall = float(np.max(np.maximum(
                                reserve - lp.deflection, 0.0)))
                        else:
                            shortfall = 0.0
                        lp = MaxMinPowerResult(
                            power_w=lp.power_w,
                            deflection=lp.deflection,
                            worst_deflection=lp.worst_deflection,
                            prices=lp.prices,
                            dual_upper_bound=lp.dual_upper_bound,
                            primal_dual_gap=lp.primal_dual_gap,
                            rounds=lp.rounds,
                            worst_history=lp.worst_history,
                            reserve_feasible=False,
                            reserve_shortfall=shortfall,
                            minimum_deflection=tuple(
                                () if reserve is None else reserve.tolist()),
                        )
                    self._current_sensing_power_w = lp.power_w.copy()
                    self._last_analytical_power_balance_error = float(np.max(
                        np.abs(np.sum(lp.power_w, axis=1) - budget)))
                    if self._analytical_structure_ranking_enabled:
                        # Expose the bottleneck dual price lambda* to the next
                        # frame's structure ranking (lagged primal-dual weight).
                        from uav_isac.coordination.maxmin_power import (
                            optimal_maxmin_dual_prices,
                            entropic_maxmin_dual_prices,
                        )
                        dual, _ = (
                            entropic_maxmin_dual_prices(
                                gain, budget, self._entropic_dual_tau)
                            if self._entropic_dual_price_enabled
                            else optimal_maxmin_dual_prices(gain, budget))
                        self._last_analytical_dual_prices = dual
                # Deflection is linear in sensing power for fixed geometry.
                # In deterministic U2U-only execution, reuse the unit-power
                # entries already computed above and scale only d_raw/d_eff;
                # stochastic report/Swerling paths retain a fresh draw.
                can_reuse_geometry = bool(
                    self._analytical_power_geometry_reuse_enabled
                    and (true_unit_deflection_dense is not None
                         or true_unit_deflection_entries is not None)
                    and not self.deflection_computer.use_report_link
                    and not self.deflection_computer.use_swerling
                )
                if can_reuse_geometry:
                    if true_unit_deflection_dense is not None:
                        deflection_entries = (
                            self._deflection_materialization.materialize(
                                true_unit_deflection_dense,
                                self._current_sensing_power_w))
                    else:
                        deflection_entries = [
                            entry._replace(
                                d_raw=(
                                    float(entry.d_raw)
                                    * float(self._current_sensing_power_w[
                                        int(entry.i), int(entry.q)])),
                                d_eff=(
                                    float(entry.d_eff)
                                    * float(self._current_sensing_power_w[
                                        int(entry.i), int(entry.q)])),
                            )
                            for entry in true_unit_deflection_entries
                        ]
                else:
                    deflection_entries = self.deflection_computer.compute(
                        uav_positions, uav_velocities,
                        target_positions, target_velocities,
                        roles, self.fc_position,
                        role_agnostic=role_agnostic,
                        sensing_power_w=self._current_sensing_power_w,
                    )
                self._last_isac_metrics[
                    'analytical_power_geometry_reused'] = float(
                        can_reuse_geometry)
                self._last_deflection_entries = deflection_entries
            self._finalize_analytical_power_accounting(budget)

        # Realized per-target deflection = TRUE d_eff of the SELECTED pairs.
        d_true = {
            (int(e.i), int(e.j), int(e.q)): float(e.d_eff)
            for e in deflection_entries
        }
        if self.p0_uses_belief or self._joint_isac_power_enabled:
            D_q_star = np.zeros(self.Q, dtype=np.float64)
            for (i, j, q) in p0_solution.selected_set:
                D_q_star[q] += d_true.get((i, j, q), 0.0)
        else:
            D_q_star = p0_solution.D_q_star

        # 4b. If P0 assigned roles, derive endpoint participation from the
        #     selection.  Code 3 denotes a dual endpoint in the time-slotted
        #     model (TX in one sub-slot, RX in another); action roles remain the
        #     historical {0=TX, 1=RX, 2=idle} when roles are learned.
        if role_agnostic:
            derived = np.full(self.K, 2, dtype=np.int32)  # default idle
            tx_nodes = {int(i) for (i, _, _) in p0_solution.selected_set}
            rx_nodes = {int(j) for (_, j, _) in p0_solution.selected_set}
            for k in tx_nodes - rx_nodes:
                derived[k] = 0
            for k in rx_nodes - tx_nodes:
                derived[k] = 1
            for k in tx_nodes & rx_nodes:
                derived[k] = 3
            for k in range(self.K):
                self.uavs[k].role = int(derived[k])
            roles = derived
            # V3-C0 (advice 015, 2026-08-17): with learn_roles=False the policy
            # role was the idle placeholder, so apply_action charged NO radio
            # energy -- but the P0-derived roles actually transmit.  Charge the
            # radio energy against the EXECUTED role here (TX transmits the
            # sensing waveform P_sense*dt, RX reports P_report*dt; code 3 dual
            # endpoint charges the TX sub-slot -- the RX sub-slot is passive
            # listening of the same broadcast).  No double charge: the
            # placeholder role charged zero above.  The joint-ISAC path charges
            # energy from the actual LP power instead (comm/evidence transport).
            if not self._joint_isac_power_enabled:
                sensing_slot_s = self._sensing_slot_duration_s()
                for k in range(self.K):
                    if derived[k] in (0, 3):
                        # Post-G2 (audit advice/001 section 3): the sensing
                        # waveform is billed on the OTFS clock
                        # ``T_sense = n_cpi*N*T_sym`` in ``cpi_frame`` mode;
                        # legacy ``dt_frame`` keeps ``P_sense*dt``.
                        self.uavs[k].battery -= (
                            self.cfg.uav.P_sense * sensing_slot_s)
                    elif derived[k] == 1:
                        # RX is a passive listener of the same broadcast.  In
                        # the canonical post-G2 U2U scope the RX executes no
                        # own RF transmit, so it is billed no sensing RF energy;
                        # legacy ``dt_frame`` keeps the historical
                        # ``P_report*dt`` charge for reproducibility.
                        if self.cfg.scenario.sensing_energy_mode == 'dt_frame':
                            self.uavs[k].battery -= (
                                self.cfg.uav.P_report * sensing_slot_s)
                    if self.uavs[k].battery < 0.0:
                        self.uavs[k].battery = 0.0

        # ── Layer 4: Event-triggered active probing ──
        # Probe only when a target's accumulated risk score exceeds threshold.
        # This avoids wasting sensing resources when all targets are well-served.
        probe_triggered = False
        probe_target = TARGET_INDEX_NONE
        if (bool(getattr(self.cfg.marl, 'active_probe_enabled', False))
                and self.belief_mgr is not None
                and self.belief_mgr.nis_enabled):
            # Track consecutive miss count per target
            if not hasattr(self, '_probe_miss_count'):
                self._probe_miss_count = np.zeros(self.Q, dtype=np.int32)
            selected_targets = {q for (_, _, q) in p0_solution.selected_set}
            for q in range(self.Q):
                if q in selected_targets:
                    self._probe_miss_count[q] = 0
                else:
                    self._probe_miss_count[q] += 1

            probe_aoi_w = float(getattr(self.cfg.marl, 'active_probe_aoi_weight', 0.5))
            probe_unc_w = float(getattr(self.cfg.marl, 'active_probe_uncertainty_weight', 0.3))
            probe_nis_w = float(getattr(self.cfg.marl, 'active_probe_nis_weight', 0.2))
            probe_miss_w = 0.4   # penalty per consecutive miss
            probe_threshold = float(getattr(self.cfg.marl, 'active_probe_threshold', 3.0))

            probe_score = np.zeros(self.Q, dtype=np.float64)
            for q in range(self.Q):
                aoi_mean = float(np.mean(self.belief_mgr.aoi[:, q]))
                cov_trace = float(np.mean([
                    np.trace(self.belief_mgr.cov[k, q])
                    for k in range(self.K)
                ]))
                nis_max = float(np.max(self.belief_mgr.nis_ema[:, q]))
                miss_penalty = self._probe_miss_count[q]
                # Disagreement bonus: high disagreement → needs independent verification
                disagreement = 0.0
                if self._trust_manager is not None:
                    disagreement = float(np.mean(
                        self._trust_manager.disagreement[:, :, q]))
                probe_score[q] = (
                    probe_aoi_w * aoi_mean
                    + probe_unc_w * cov_trace
                    + probe_nis_w * nis_max
                    + probe_miss_w * miss_penalty
                    + 0.1 * disagreement
                )

            # Event-triggered: only probe when max score exceeds threshold
            if np.max(probe_score) > probe_threshold:
                q_star = int(np.argmax(probe_score))
                if q_star not in selected_targets:
                    # Find best available (tx, rx) pair for q_star
                    best_entry = None
                    best_d = -1.0
                    for e in deflection_entries:
                        if e.q == q_star and e.d_eff > 0:
                            if e.d_eff > best_d:
                                best_d = e.d_eff
                                best_entry = e
                    if best_entry is not None:
                        p0_solution.selected_set.append(
                            (best_entry.i, best_entry.j, best_entry.q))
                        D_q_star[q_star] += best_entry.d_eff
                        if role_agnostic:
                            tx_nodes = {
                                int(i) for (i, _, _)
                                in p0_solution.selected_set}
                            rx_nodes = {
                                int(j) for (_, j, _)
                                in p0_solution.selected_set}
                            derived.fill(2)
                            for k in tx_nodes - rx_nodes:
                                derived[k] = 0
                            for k in rx_nodes - tx_nodes:
                                derived[k] = 1
                            for k in tx_nodes & rx_nodes:
                                derived[k] = 3
                            for k in range(self.K):
                                self.uavs[k].role = int(derived[k])
                            roles = derived
                        probe_triggered = True
                        probe_target = q_star
                        self._probe_miss_count[q_star] = 0

        # 5. Resolve the evidence boundary before computing any detector,
        #    reward, constraint, or tracking statistic. P0 remains a scheduler;
        #    its cumulative score is not permission to fuse receiver evidence.
        if self._passive_multireceiver_evidence_enabled:
            receiver_D = receiver_deflection_from_broadcast_waveforms(
                p0_solution.selected_set,
                deflection_entries,
                self.K,
                self.Q,
            )
        else:
            receiver_D = receiver_deflection_from_selected(
                p0_solution.selected_set,
                deflection_entries,
                self.K,
                self.Q,
            )
        central_D = np.sum(receiver_D, axis=0)
        # The public sensing schedule, not the simulator's receiver-quality
        # matrix, defines the fusion directory.  Keep this value as the sole
        # owner source for execution, diagnostics, and calibration traces.
        evidence_owner = scheduled_fusion_owner(
            p0_solution.selected_set,
            self.K,
            self.Q,
        )
        belief_schedule_mask = None
        if self._u2u_belief_feedback_schedule == 'freshness':
            # Replicate the evidence selection the router will compute so the
            # posterior scheduler can share target metadata where useful and
            # enforce the per-source union cap. Every selected posterior still
            # pays its full state payload.
            owned_target = evidence_owner >= 0
            owner_selection_mask = np.zeros(
                (self.K, self.Q), dtype=bool)
            owner_selection_mask[
                evidence_owner[owned_target],
                np.arange(self.Q)[owned_target],
            ] = True
            selection_quality = receiver_D.copy()
            selection_quality[:, ~owned_target] = 0.0
            if self._evidence_owner_aware:
                selection_quality[owner_selection_mask] = 0.0
            evidence_selection = local_quality_topk_mask(
                selection_quality, self._evidence_topk)
            if self._evidence_ambiguity_top2_enabled:
                evidence_selection = local_ambiguity_top2_mask(
                    selection_quality,
                    base_topk=self._evidence_topk,
                    second_ratio=self._evidence_ambiguity_ratio,
                    second_min_deflection=(
                        self._evidence_ambiguity_min_deflection),
                )
            belief_schedule_mask = self._belief_freshness_schedule_mask(
                evidence_owner, evidence_selection)
        evidence_transport = None
        evidence_detection = None
        belief_feedback_source_mean = None
        belief_feedback_source_covariance = None
        belief_feedback_source_aoi = None
        if self._u2u_belief_feedback_enabled:
            if self.belief_mgr is None:
                raise RuntimeError(
                    "belief feedback is enabled without a belief manager")
            belief_feedback_source_mean = (
                self.belief_mgr.mean[:, :, :4].copy())
            belief_feedback_source_covariance = (
                self.belief_mgr.cov[:, :, :4, :4].copy())
            belief_feedback_source_aoi = self.belief_mgr.aoi.copy()
        if self._detection_fusion_mode == 'u2u_distributed':
            if self._inter_uav_comm is None or self._evidence_packet_layout is None:
                raise RuntimeError(
                    "distributed evidence fusion requires communication and "
                    "packet-layout state")
            evidence_power = (
                self._current_comm_power_w.copy()
                if self._joint_isac_power_enabled
                else np.full(
                    self.K,
                    float(getattr(
                        self.cfg.marl, 'comm_tx_power_w', 0.25)),
                    dtype=np.float64,
                )
            )
            evidence_transport = route_structured_evidence(
                receiver_D,
                uav_positions,
                evidence_power,
                observation_frame=self.t,
                topk=self._evidence_topk,
                llr_bits=self._evidence_llr_bits,
                layout=self._evidence_packet_layout,
                link_model=self._inter_uav_comm,
                fusion_owner=evidence_owner,
                owner_aware=self._evidence_owner_aware,
                ambiguity_second_ratio=(
                    self._evidence_ambiguity_ratio
                    if self._evidence_ambiguity_top2_enabled else None),
                ambiguity_second_min_deflection=(
                    self._evidence_ambiguity_min_deflection),
                belief_selected_mask=belief_schedule_mask,
                service_envelope_layout=(
                    self._evidence_service_envelope_layout),
            )
            evidence_detection = estimate_quantized_evidence_detection(
                receiver_D,
                evidence_transport,
                llr_bits=self._evidence_llr_bits,
                clip_max=self._evidence_clip_max,
                standardized_threshold=self._evidence_threshold,
                p_fa=self.cfg.detection.P_FA,
                confidence_quantizer=(
                    self._evidence_confidence_quantizer),
                draws=self._evidence_mc_draws,
                seed=self._evidence_mc_seed + int(self.t),
                content_mode=self._evidence_content_mode,
            )
            detection_D_q = np.asarray(
                evidence_detection['equivalent_deflection'],
                dtype=np.float64,
            )
            P_D_q = np.asarray(
                evidence_detection['pd'], dtype=np.float64)
            for sender, energy_j in enumerate(
                    evidence_transport.energy_j_by_sender):
                if energy_j > 0.0:
                    self.uavs[sender].battery = max(
                        0.0,
                        self.uavs[sender].battery - float(energy_j),
                    )
        else:
            detection_D_q = select_detection_deflection(
                self._detection_fusion_mode,
                receiver_D,
                legacy_global_deflection=D_q_star,
            )
            P_D_q = compute_detection_probabilities(
                detection_D_q, self.cfg.detection.P_FA)
        self._last_isac_metrics.update({
            'detection_fusion_mode': self._detection_fusion_mode,
            'passive_multireceiver_evidence_enabled': float(
                self._passive_multireceiver_evidence_enabled),
            'detection_deflection_q': detection_D_q.copy(),
            'detection_central_oracle_deflection_q': central_D.copy(),
            'detection_receiver_deflection': receiver_D.copy(),
            'detection_fusion_owner': evidence_owner.copy(),
        })
        if evidence_transport is not None and evidence_detection is not None:
            self._last_isac_metrics.update(
                evidence_transport.as_dict())
            self._last_isac_metrics.update({
                'evidence_detection_pfa': np.asarray(
                    evidence_detection['pfa'],
                    dtype=np.float64,
                ).copy(),
                'evidence_detection_aggregate_pfa': float(
                    evidence_detection['aggregate_pfa']),
                'evidence_detection_clip_rate': float(
                    evidence_detection['clip_rate']),
                'evidence_detection_quantization_mse': float(
                    evidence_detection['quantization_mse']),
                'evidence_detection_sign_flip_rate': float(
                    evidence_detection['sign_flip_rate']),
                'evidence_detection_available_true_deflection': np.asarray(
                    evidence_detection['available_true_deflection'],
                    dtype=np.float64,
                ).copy(),
                'evidence_detection_threshold_deflection': np.asarray(
                    evidence_detection['threshold_deflection'],
                    dtype=np.float64,
                ).copy(),
                'evidence_detection_mc_draws': float(
                    evidence_detection['draws']),
                'evidence_packet_llr_bits': float(
                    self._evidence_llr_bits),
                'evidence_packet_topk': float(self._evidence_topk),
                'evidence_packet_calibration_profile': (
                    self._evidence_calibration_profile),
                'evidence_packet_content_mode': (
                    self._evidence_content_mode),
            })
        self._last_isac_metrics.update(
            self._serialized_protocol_accounting(
                comm_stats, evidence_transport))

        # 6. Check constraints
        batteries = np.array([u.battery for u in self.uavs])
        constraint_info = self.constraint_checker.check_all(
            uav_positions, batteries, P_D_q
        )

        # 7. Compute rewards (constraint penalties handled by Lagrangian in trainer)
        reward_gain = None
        reward_budget = None
        if self.reward_computer.utility_mode == 'maxmin_dual':
            reward_gain, reward_budget = self._maxmin_dual_reward_gain(
                deflection_entries, p0_solution.selected_set)
        team_reward = self.reward_computer.compute_team_reward(
            detection_D_q,            # evidence available under selected mode
            p0_solution.total_bits,
            0.0,  # constraint_penalty removed; Lagrangian handles constraints
            P_D_q=P_D_q,              # for direct P_D reward term (alpha_pd > 0)
            gain_per_watt=reward_gain,
            sensing_budget_w=reward_budget,
        )
        reward_components = {
            'base_detection_and_report': float(team_reward),
            'learned_comm_cost': 0.0,
            'evidence_comm_cost': 0.0,
            'distance_shaping': 0.0,
        }
        if self._comm_mode == 'cost_aware':
            ma = self.cfg.marl
            comm_cost = (
                float(getattr(ma, 'comm_bit_cost_weight', 1.0e-5))
                * comm_stats.total_bits
                + float(getattr(ma, 'comm_energy_cost_weight', 1.0))
                * comm_stats.total_energy_j
                + float(getattr(ma, 'comm_delay_cost_weight', 10.0))
                * (min(
                    comm_stats.mean_latency_s,
                    self._active_comm_deadline_s)
                   + comm_stats.deadline_violation_rate)
            )
            team_reward -= comm_cost
            reward_components['learned_comm_cost'] = float(comm_cost)
            if evidence_transport is not None:
                evidence_cost = (
                    float(getattr(
                        ma, 'comm_bit_cost_weight', 1.0e-5))
                    * evidence_transport.total_bits
                    + float(getattr(
                        ma, 'comm_energy_cost_weight', 1.0))
                    * evidence_transport.total_energy_j
                    + float(getattr(
                        ma, 'comm_delay_cost_weight', 10.0))
                    * (
                        min(
                            evidence_transport.mean_latency_s,
                            self._active_comm_deadline_s)
                        + evidence_transport.deadline_violation_rate
                    )
                )
                team_reward -= evidence_cost
                reward_components['evidence_comm_cost'] = float(
                    evidence_cost)

        # Potential-based distance shaping: F = γΦ(s') - Φ(s), Φ = -Σ_q min_k dist.
        # Action-dependent part rewards moving UAVs toward targets; policy-invariant.
        if self.use_distance_shaping:
            phi_old = self._coverage_potential(prev_uav_positions, target_positions)
            phi_new = self._coverage_potential(uav_positions, target_positions)
            distance_shaping = self.shape_w * (
                self.gamma_shape * phi_new - phi_old)
            team_reward += distance_shaping
            reward_components['distance_shaping'] = float(distance_shaping)

        # Use a short EMA to prevent frame-level fading from overwhelming the
        # movement/communication credit signal. The raw and weighted terms are
        # still reported separately for stage-wise attribution.
        if self._coord_pd_ema is None:
            coord_pd = P_D_q.copy()
        else:
            alpha = self._coord_reward_ema_alpha
            coord_pd = alpha * P_D_q + (1.0 - alpha) * self._coord_pd_ema
        coord_components = compute_segmented_coordination_reward(
            self._coord_pd_ema,
            coord_pd,
            prev_uav_positions,
            uav_positions,
            target_positions,
            stage=(self._coord_reward_stage
                   if self._coord_reward_enabled else 0),
            **self._coord_reward_kwargs,
        )
        self._coord_pd_ema = coord_pd.copy()
        team_reward += coord_components['coord_total']
        reward_components.update(coord_components)
        reward_components['team_total'] = float(team_reward)

        # Marginal contributions (gated; disabled in team-only baseline)
        if getattr(self.cfg.marl, 'use_centered_marginal', False):
            marginal = self.reward_computer.compute_marginal_contributions(
                list(range(self.K)),
                p0_solution.selected_set,
                deflection_entries,
                self.Q,
                detection_fusion_mode=self._detection_fusion_mode,
                num_agents=self.K,
            )
        else:
            marginal = {k: 0.0 for k in range(self.K)}

        # Fixed-assignment difference reward: Δ_k = F(actual) - F(noop_k)
        # Uses the CURRENT P0 assignment (fixed, no re-solve). Measures how much
        # UAV k's own movement contributes to task utility, independent of others.
        diff_rewards = {}
        use_diff = getattr(self.cfg.marl, 'use_difference_reward', False)
        if use_diff:
            # Compute utility for the actual configuration
            actual_receiver_D = self._compute_assigned_receiver_deflection(
                uav_positions, p0_solution.selected_set, deflection_entries,
                uav_velocities, target_positions, target_velocities, roles)
            actual_D_q = select_detection_deflection(
                ('central_oracle'
                 if self._detection_fusion_mode == 'legacy_global'
                 else self._detection_fusion_mode),
                actual_receiver_D,
            )
            actual_util = self.reward_computer.compute_team_utility_from_deflection(
                actual_D_q, p0_solution.total_bits)

            for k in range(self.K):
                # No-op: UAV k stays at its PREVIOUS position
                cf_positions = uav_positions.copy()
                cf_positions[k] = prev_uav_positions[k]
                cf_receiver_D = self._compute_assigned_receiver_deflection(
                    cf_positions, p0_solution.selected_set, deflection_entries,
                    uav_velocities, target_positions, target_velocities, roles)
                cf_D_q = select_detection_deflection(
                    ('central_oracle'
                     if self._detection_fusion_mode == 'legacy_global'
                     else self._detection_fusion_mode),
                    cf_receiver_D,
                )
                cf_util = self.reward_computer.compute_team_utility_from_deflection(
                    cf_D_q, p0_solution.total_bits)
                diff_rewards[k] = actual_util - cf_util
        else:
            diff_rewards = {k: 0.0 for k in range(self.K)}

        # Per-agent sensing quality (optional; gated by eta_sense > 0)
        eta_sense = getattr(self.cfg.marl, 'eta_sense', 0.0)
        per_agent_sensing = {}
        if eta_sense > 0:
            for k in range(self.K):
                d_sum = sum(e.d_eff for e in deflection_entries
                           if e.d_eff > 0 and (e.i == k or e.j == k))
                n_entries = max(1, sum(1 for e in deflection_entries
                                       if e.d_eff > 0 and (e.i == k or e.j == k)))
                per_agent_sensing[k] = d_sum / n_entries if n_entries > 0 else 0.0

        shaped_rewards = self.reward_computer.compute_shaped_rewards(
            team_reward, marginal,
            per_agent_sensing=(per_agent_sensing if eta_sense > 0 else None),
            eta_sense=eta_sense,
            diff_rewards=diff_rewards,
            team_weight=getattr(self.cfg.marl, 'team_weight', 0.7),
            diff_weight=getattr(self.cfg.marl, 'diff_weight', 0.3),
        )

        # 8. Update beliefs
        if self.tracking_enabled:
            self.belief_mgr.step()  # CV predict + increment AoI

        belief_feedback_metrics = self._apply_u2u_belief_feedback(
            evidence_transport,
            belief_feedback_source_mean,
            belief_feedback_source_covariance,
            belief_feedback_source_aoi,
        ) if self._u2u_belief_feedback_enabled else {
            'belief_feedback_enabled': 0.0,
            'belief_feedback_fused_entries': 0.0,
            'belief_feedback_receivers': 0.0,
            'belief_feedback_disagreement_before_m': 0.0,
            'belief_feedback_disagreement_after_m': 0.0,
            'belief_feedback_contraction_ratio': 1.0,
            'belief_feedback_truth_rmse_before_m': 0.0,
            'belief_feedback_truth_rmse_after_m': 0.0,
        }

        # Kalman update for observed targets (noisy measurement of true state)
        true_states = np.array([
            [t.state[0], t.state[1], *t.get_velocity()]
            for t in self.targets
        ])
        # B7: optionally gate the belief update by a detection event. With
        # belief_detection_sampling, target q's belief updates only if
        # delta_q ~ Bernoulli(P_D_q) fires (sampled once per target); otherwise
        # the pair is "missed" -> predict-only, AoI keeps growing. Default off =
        # optimistic (selected pair always observes).
        detected_q: Dict[int, bool] = {}
        if self.tracking_enabled:
            if self._belief_measurement_model == 'bistatic_range_doppler':
                information_gain = np.zeros(self.Q, dtype=np.float64)
                minimum_eigenvalue = np.zeros(self.Q, dtype=np.float64)
                for target_index in range(self.Q):
                    owner = int(evidence_owner[target_index])
                    if owner < 0:
                        continue
                    prior = 0.5 * (
                        self.belief_mgr.cov[owner, target_index, :4, :4]
                        + self.belief_mgr.cov[
                            owner, target_index, :4, :4].T)
                    eigenvalues, eigenvectors = np.linalg.eigh(prior)
                    prior_sqrt = (
                        eigenvectors
                        @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
                        @ eigenvectors.T)
                    information = np.zeros((4, 4), dtype=np.float64)
                    for tx, rx, q in p0_solution.selected_set:
                        if int(q) != target_index or int(rx) != owner:
                            continue
                        edge_deflection = float(d_true.get(
                            (int(tx), int(rx), int(q)), 0.0))
                        if edge_deflection <= 0.0:
                            continue
                        _, jacobian = (
                            bistatic_range_doppler_measurement_and_jacobian(
                                self.belief_mgr.mean[owner, target_index],
                                uav_positions[int(tx)],
                                uav_velocities[int(tx)],
                                uav_positions[int(rx)],
                                uav_velocities[int(rx)],
                                float(self.deflection_computer.fc),
                            ))
                        measurement_covariance = bistatic_range_doppler_crlb(
                            edge_deflection,
                            float(self.deflection_computer.M
                                  * self.deflection_computer.delta_f),
                            float(self.deflection_computer.n_cpi
                                  * self.deflection_computer.N
                                  * self.deflection_computer.T_sym),
                            efficiency=self._belief_bistatic_crlb_efficiency,
                            minimum_effective_deflection=(
                                self._belief_bistatic_min_effective_deflection),
                        )
                        h4 = jacobian[:, :4]
                        information += (
                            h4.T
                            @ np.linalg.inv(measurement_covariance)
                            @ h4)
                    normalized = prior_sqrt @ information @ prior_sqrt
                    normalized = 0.5 * (normalized + normalized.T)
                    normalized_eigenvalues = np.maximum(
                        np.linalg.eigvalsh(normalized), 0.0)
                    information_gain[target_index] = 0.5 * float(np.sum(
                        np.log1p(normalized_eigenvalues)))
                    minimum_eigenvalue[target_index] = float(
                        np.min(normalized_eigenvalues))
                self._last_isac_metrics.update({
                    'bistatic_tracker_information_gain_mean': float(
                        np.mean(information_gain)),
                    'bistatic_tracker_information_gain_worst': float(
                        np.min(information_gain)),
                    'bistatic_tracker_full_rank_target_fraction': float(
                        np.mean(minimum_eigenvalue > 1.0e-8)),
                    'bistatic_tracker_min_normalized_information_eigenvalue': (
                        float(np.min(minimum_eigenvalue))),
                })
            if self._passive_multireceiver_belief_update_enabled:
                local_detection_probability = compute_detection_probabilities(
                    receiver_D, self.cfg.detection.P_FA)
                for receiver in range(self.K):
                    for q in range(self.Q):
                        if receiver_D[receiver, q] <= 0.0:
                            continue
                        obs = bool(
                            self.rng.random()
                            < float(local_detection_probability[receiver, q])
                        ) if self.belief_detection_sampling else True
                        detected_q[q] = bool(
                            detected_q.get(q, False) or obs)
                        self.belief_mgr.update_after_observation(
                            receiver,
                            q,
                            obs,
                            true_states[q],
                            detection_probability=(
                                float(local_detection_probability[receiver, q])
                                if self._belief_expected_detection_information_enabled
                                else None
                            ),
                            detection_probability_floor=(
                                self._belief_expected_detection_information_floor),
                        )
            else:
                for (i, j, q) in p0_solution.selected_set:
                    if q not in detected_q:
                        if self.belief_detection_sampling:
                            detected_q[q] = bool(
                                self.rng.random() < float(P_D_q[q]))
                        else:
                            detected_q[q] = True
                    obs = detected_q[q]
                    ts = true_states[q]
                    if self._belief_measurement_model == (
                        'bistatic_range_doppler'
                    ):
                        # Only the scheduled receiver observes the echo.  The
                        # transmitter does not receive a free Cartesian target
                        # state; peers learn the receiver posterior later via
                        # the physically billed owner-posterior protocol.
                        edge_deflection = float(
                            d_true.get((int(i), int(j), int(q)), 0.0))
                        self.belief_mgr.update_after_bistatic_observation(
                            j,
                            q,
                            bool(obs and edge_deflection > 0.0),
                            ts,
                            transmitter_position_m=uav_positions[i],
                            transmitter_velocity_mps=uav_velocities[i],
                            receiver_position_m=uav_positions[j],
                            receiver_velocity_mps=uav_velocities[j],
                            carrier_hz=float(self.deflection_computer.fc),
                            effective_deflection=edge_deflection,
                            bandwidth_hz=float(
                                self.deflection_computer.M
                                * self.deflection_computer.delta_f),
                            coherent_time_s=float(
                                self.deflection_computer.n_cpi
                                * self.deflection_computer.N
                                * self.deflection_computer.T_sym),
                            crlb_efficiency=(
                                self._belief_bistatic_crlb_efficiency),
                            minimum_effective_deflection=(
                                self._belief_bistatic_min_effective_deflection),
                        )
                    else:
                        expected_probability = (
                            float(P_D_q[q])
                            if self._belief_expected_detection_information_enabled
                            else None
                        )
                        self.belief_mgr.update_after_observation(
                            i,
                            q,
                            obs,
                            ts,
                            detection_probability=expected_probability,
                            detection_probability_floor=(
                                self._belief_expected_detection_information_floor),
                        )
                        self.belief_mgr.update_after_observation(
                            j,
                            q,
                            obs,
                            ts,
                            detection_probability=expected_probability,
                            detection_probability_floor=(
                                self._belief_expected_detection_information_floor),
                        )

            # The designated bistatic receiver is the target's posterior
            # owner for this frame.  Snapshot only its causal filter state;
            # the packet is quantized, delayed and charged on the next U2U
            # carrier by _process_learned_communications().
            self._prepare_owner_posterior_submission(tuple(
                (int(i), int(j), int(q))
                for i, j, q in p0_solution.selected_set
            ))

            true_position = true_states[:, :2]
            belief_position_error = (
                self.belief_mgr.mean[:, :, :2]
                - true_position[None, :, :]
            )
            position_error_norm = np.linalg.norm(
                belief_position_error, axis=-1)
            belief_feedback_metrics.update({
                'belief_position_rmse_m': float(np.sqrt(np.mean(
                    position_error_norm * position_error_norm))),
                'belief_position_error_p95_m': float(np.percentile(
                    position_error_norm, 95)),
                'belief_position_error_max_m': float(np.max(
                    position_error_norm, initial=0.0)),
                'belief_position_rmse_per_target_m': np.sqrt(np.mean(
                    position_error_norm * position_error_norm,
                    axis=0,
                )),
                'belief_target_aoi_mean_frames': float(np.mean(
                    self.belief_mgr.aoi)),
                'belief_target_aoi_p95_frames': float(np.percentile(
                    self.belief_mgr.aoi, 95)),
                'belief_target_aoi_max_frames': float(np.max(
                    self.belief_mgr.aoi, initial=0)),
                'belief_target_aoi_per_target_frames': np.mean(
                    self.belief_mgr.aoi, axis=0),
                'belief_position_disagreement_end_m': (
                    self._belief_position_disagreement(
                        self.belief_mgr.mean)),
            })
        self._last_isac_metrics.update(belief_feedback_metrics)

        # ── Layer 4: Trust feedback — update trust based on post-measurement NIS ──
        if (
            self.trust_gate_enabled
            and self._trust_manager is not None
            and not self._passive_multireceiver_belief_update_enabled
        ):
            for (i, j, q) in p0_solution.selected_set:
                if q in detected_q and detected_q[q]:
                    # Use the NIS just computed in update_after_observation
                    nis_i = float(self.belief_mgr._last_nis[i, q])
                    nis_j = float(self.belief_mgr._last_nis[j, q])
                    # Both tx and rx observed this target; update mutual trust
                    self._trust_manager.update_trust_from_nis(i, j, q, nis_i, nis_j)
                    self._trust_manager.update_trust_from_nis(j, i, q, nis_j, nis_i)
                    # Check quarantine
                    self._trust_manager.check_quarantine(i, j, q)
                    self._trust_manager.check_quarantine(j, i, q)

        # 9. Compute previous P_D BEFORE building next observations.
        #    Fixes off-by-one: previously prev_P_D was updated AFTER
        #    _build_observations, so next_obs got P_D_{t-1} not P_D_t.
        self.prev_P_D = P_D_q.copy()

        # D1.1-A audit hook: dump one JSONL record per frame when enabled.
        if self._lex_audit_path:
            try:
                with open(self._lex_audit_path, 'a', encoding='utf-8') as _f:
                    _f.write(json.dumps({
                        "t": int(self.t),
                        "mode": self._lex_mode,
                        "lex_t_star": self._lex_t_star,
                        "balance_error_w": float(
                            self._last_analytical_power_balance_error),
                        "budget_violation_w": float(
                            self._last_analytical_power_budget_violation_w),
                        "unused_power_w": float(
                            self._last_analytical_unused_power_w),
                        "P_D_q": [float(v) for v in P_D_q],
                        "comm_power_w": [
                            float(v) for v in self._current_comm_power_w],
                        "sensing_row_sums_w": [
                            float(np.sum(self._current_sensing_power_w[i]))
                            for i in range(self.K)],
                        "gain": (self._last_analytical_gain.tolist()
                                 if self._last_analytical_gain is not None
                                 else None),
                        "budget": (self._last_analytical_budget.tolist()
                                   if self._last_analytical_budget is not None
                                   else None),
                    }) + "\n")
            except Exception as error:
                raise RuntimeError(
                    "failed to append enabled lex audit record; refusing to "
                    "continue with an incomplete evidence trace"
                ) from error

        # P1 FIX: per-UAV LOCAL detection confidence (RX-only by default).
        # Only the RX UAV of each bistatic pair gets local P_D credit;
        # this respects the physical reality that RX performs detection.
        # TX must learn target status via neighbor comm messages or
        # explicit RX→TX feedback (not yet modeled).
        self.prev_P_D_local = {}
        for k in range(self.K):
            D_k = np.zeros(self.Q, dtype=np.float64)
            for (i, j, q) in p0_solution.selected_set:
                if j == k:  # RX-only (方案A)
                    for e in deflection_entries:
                        if e.i == i and e.j == j and e.q == q:
                            D_k[q] += e.d_eff
                            break
            from uav_isac.utils.math_utils import compute_PD
            # `compute_PD(0, P_FA)` returns the detector false-alarm floor.
            # That is a global detector statistic, not evidence locally
            # available to a UAV that was not an RX. Keep absent local sensing
            # exactly zero so the sender must communicate any useful result.
            P_D_local_k = np.array([
                compute_PD(D_k[q], self.cfg.detection.P_FA)
                if D_k[q] > 0.0 else 0.0
                for q in range(self.Q)
            ])
            self.prev_P_D_local[k] = P_D_local_k

        # 10. Build next observations (now sees correct prev_P_D_local from step 9)
        next_obs = self._build_observations()

        # 11. Check termination
        dones = {}
        all_dead = all(not u.is_alive() for u in self.uavs)
        time_up = self.t >= self.T
        done_flag = all_dead or time_up
        for k in range(self.K):
            dones[k] = done_flag or not self.uavs[k].is_alive()
        dones['__all__'] = done_flag

        # Build step info
        uav_states = [u.get_state() for u in self.uavs]
        target_states = [t.get_state_as_target_state() for t in self.targets]

        # ── Pairing diagnostics (P0) ──
        # A bistatic candidate requires >=1 TX and >=1 RX; valid_pair is the
        # ground truth that sensing actually occurred (P0 selected a triple).
        if role_agnostic:
            tx_nodes = {int(i) for (i, _, _) in p0_solution.selected_set}
            rx_nodes = {int(j) for (_, j, _) in p0_solution.selected_set}
            n_tx = len(tx_nodes)
            n_rx = len(rx_nodes)
            n_duplex = len(tx_nodes & rx_nodes)
        else:
            n_tx = int(np.sum(roles == 0))
            n_rx = int(np.sum(roles == 1))
            n_duplex = 0
        n_selected = len(p0_solution.selected_set)
        all_same_role = bool(
            not self._multistatic_subslot_enabled
            and np.all(roles == roles[0])) if self.K > 0 else False

        power_parallel_time_s = float(self._last_isac_metrics.get(
            'distributed_replicated_power_parallel_critical_path_s', 0.0))
        controller_compute_critical_path_s = float(
            movement_compute_time_s
            + hyperedge_compute_time_s
            + self._last_p0_solve_time_s
            + power_parallel_time_s
        )
        radio_critical_path_s = float(comm_stats.max_latency_s)
        closed_loop_critical_path_upper_s = float(
            controller_compute_critical_path_s + radio_critical_path_s)
        simulation_step_wall_s = float(
            time.perf_counter() - step_wall_started)

        step_info = StepInfo(
            frame=self.t,
            uav_states=uav_states,
            target_states=target_states,
            deflection_entries=deflection_entries,
            p0_solution=p0_solution,
            P_D_q=P_D_q,
            team_reward=team_reward,
            shaped_rewards=shaped_rewards,
            constraint_info=constraint_info,
            dones=dones,
            roles=roles.copy(),
            n_tx=n_tx,
            n_rx=n_rx,
            n_selected=n_selected,
            n_duplex=n_duplex,
            valid_pair=bool(n_selected > 0),
            no_tx=bool(n_tx == 0),
            all_same_role=all_same_role,
            p0_resolved=bool(self._assignment_switched),
            p0_solve_time_s=float(self._last_p0_solve_time_s),
            learned_comm={
                **comm_stats.as_dict(),
                **self._last_isac_metrics,
                **self._last_movement_safety_metrics,
                'learned_comm_channel_profile': (
                    self._active_comm_channel_profile),
                'learned_comm_channel_snr_threshold_db': (
                    self._active_comm_snr_threshold_db),
                'learned_comm_channel_deadline_s': (
                    self._active_comm_deadline_s),
                'learned_comm_per_sender_delivery_rate': (
                    comm_stats.sender_delivery_rates(self.K)),
                # Disjoint timing semantics: simulator wall time is never
                # labeled deployment latency.  The controller proxy contains
                # only online movement/structure/P0/slowest-node power work;
                # the physical radio latency is exposed separately and the
                # conservative closed-loop bound adds the dependent stages.
                'timing_simulator_step_wall_s': simulation_step_wall_s,
                'timing_comm_model_processing_wall_s': (
                    comm_processing_wall_time_s),
                'timing_controller_movement_compute_s': (
                    movement_compute_time_s),
                'timing_controller_structure_compute_s': (
                    hyperedge_compute_time_s),
                'timing_controller_power_parallel_s': power_parallel_time_s,
                'timing_controller_compute_critical_path_s': (
                    controller_compute_critical_path_s),
                'timing_radio_serialization_critical_path_s': (
                    radio_critical_path_s),
                'timing_closed_loop_critical_path_upper_s': (
                    closed_loop_critical_path_upper_s),
            },
            reward_components=reward_components,
        )

        return next_obs, shaped_rewards, dones, step_info

    def _compute_assigned_deflection(
        self, uav_positions: np.ndarray,
        selected_set: list,
        _unused_entries: list,  # ignored; we recompute below
        uav_velocities: np.ndarray = None,
        target_positions: np.ndarray = None,
        target_velocities: np.ndarray = None,
        roles: np.ndarray = None,
    ) -> np.ndarray:
        """Per-target fused deflection for a FIXED set of (i,j,q) pairs.
        Recomputes the FULL deflection geometry with the given positions,
        then filters to only the selected pairs. Does NOT re-run P0."""
        return np.sum(self._compute_assigned_receiver_deflection(
            uav_positions,
            selected_set,
            _unused_entries,
            uav_velocities,
            target_positions,
            target_velocities,
            roles,
        ), axis=0)

    def _compute_assigned_receiver_deflection(
        self, uav_positions: np.ndarray,
        selected_set: list,
        _unused_entries: list,
        uav_velocities: np.ndarray = None,
        target_positions: np.ndarray = None,
        target_velocities: np.ndarray = None,
        roles: np.ndarray = None,
    ) -> np.ndarray:
        """Receiver-local deflection for one fixed P0 assignment."""
        if uav_velocities is None or target_positions is None:
            # Fallback: use cached entries (actual positions only)
            return receiver_deflection_from_selected(
                selected_set, _unused_entries, self.K, self.Q)

        # Recompute deflection for ALL pairs, then filter to selected
        all_entries = self.deflection_computer.compute(
            uav_positions, uav_velocities,
            target_positions, target_velocities,
            roles if roles is not None else np.zeros(self.K, dtype=np.int32),
            self.fc_position,
            role_agnostic=not self.learn_roles,
            sensing_power_w=(self._current_sensing_power_w
                             if self._joint_isac_power_enabled else None),
        )
        return receiver_deflection_from_selected(
            selected_set, all_entries, self.K, self.Q)

    def _fuse_beliefs_attention(self) -> np.ndarray:
        """Fuse per-UAV beliefs via multi-head neighbor attention + CI.

        When trust_gate_enabled, applies disagreement-based gating
        and weight caps before CI fusion (Layer 2).

        Returns:
            fused_mean: (Q, 4) fused belief mean per target
        """
        import torch
        Q, K = self.Q, self.K
        D = 4  # belief mean dimension

        # Extract per-agent per-target belief summaries
        loc_mean = np.zeros((K, Q, D), dtype=np.float64)
        loc_cov = np.zeros((K, Q, D), dtype=np.float64)
        loc_aoi = np.zeros((K, Q, 1), dtype=np.float64)
        loc_pd = np.zeros((K, Q, 1), dtype=np.float64)

        for k in range(K):
            for q in range(Q):
                b = self.belief_mgr.get_belief(k, q)
                loc_mean[k, q] = b.mean
                loc_cov[k, q] = np.abs(b.cov_diag)
                loc_aoi[k, q, 0] = float(b.aoi)
            if hasattr(self, 'prev_P_D_local') and k in self.prev_P_D_local:
                loc_pd[k, :, 0] = self.prev_P_D_local[k]

        # ── Layer 2: Trust gate (numpy, before torch) ──
        trust_enabled = (self.trust_gate_enabled and self._trust_manager is not None
                         and self.belief_mgr is not None)
        if trust_enabled:
            # Get full covariance matrices for Mahalanobis computation
            belief_cov_full = self.belief_mgr.get_calibrated_covariance()  # (K,Q,4,4)
            belief_aoi_raw = self.belief_mgr.aoi.astype(np.float64)  # (K,Q)
            nis_ema_raw = self.belief_mgr.nis_ema  # (K,Q)
            trust_scores, _ = self._trust_manager.compute_gate_weights(
                loc_mean, belief_cov_full, belief_aoi_raw, nis_ema_raw,
            )  # trust_scores: (K, K, Q)
            weight_max_val = self._trust_manager.weight_max
            local_w_min = self._trust_manager.local_weight_min
            # Also decay quarantines each frame
            self._trust_manager.decay_quarantine()
        else:
            trust_scores = None
            weight_max_val = 1.0
            local_w_min = 0.25

        # Use raw belief values (meters, meters^2) — CI needs consistent units
        loc_mean_t = torch.as_tensor(loc_mean, dtype=torch.float32)
        loc_cov_t = torch.as_tensor(np.abs(loc_cov), dtype=torch.float32)
        loc_aoi_t = torch.as_tensor(loc_aoi, dtype=torch.float32)
        loc_pd_t = torch.as_tensor(loc_pd, dtype=torch.float32)

        # For each agent as query, other agents as neighbors
        all_fused = np.zeros((Q, D), dtype=np.float64)
        for k in range(K):
            # Query: agent k
            q_mean = loc_mean_t[k:k+1]  # (1, Q, D)
            q_cov = loc_cov_t[k:k+1]
            q_aoi = loc_aoi_t[k:k+1]
            q_pd = loc_pd_t[k:k+1]

            # Neighbors: all other agents → (1, Q, N, D)
            nb_idx = [j for j in range(K) if j != k]
            nb_mean = loc_mean_t[nb_idx].permute(1, 0, 2).unsqueeze(0)  # (N,Q,D)→(Q,N,D)→(1,Q,N,D)
            nb_cov = loc_cov_t[nb_idx].permute(1, 0, 2).unsqueeze(0)
            nb_aoi = loc_aoi_t[nb_idx].permute(1, 0, 2).unsqueeze(0)
            nb_pd = loc_pd_t[nb_idx].permute(1, 0, 2).unsqueeze(0)
            nb_mask = torch.ones(1, K-1, dtype=torch.bool)

            # Build trust tensor for this agent's neighbors
            # Trust is used as binary gate (quarantine only), not weight scaling.
            # Weight scaling would kill useful but disagreeing neighbors.
            # The weight cap (weight_max) is the primary safety mechanism.
            ts_k = None
            if trust_scores is not None:
                ts_k_np = (trust_scores[k, nb_idx, :] > 0.01).astype(np.float64)  # (N, Q)
                ts_k = ts_k_np.T[np.newaxis, :, :]     # (1, Q, N)

            with torch.no_grad():
                fw, _, _ = self._belief_fusion_module(
                    q_mean, q_cov, q_aoi, q_pd,
                    nb_mean, nb_cov, nb_aoi, nb_pd,
                    nb_mask)
                fused_m, _ = self._belief_fusion_module.covariance_intersection_fusion(
                    q_mean, q_cov, nb_mean, nb_cov, fw,
                    local_weight=local_w_min,
                    trust_scores=ts_k,
                    weight_max=weight_max_val,
                )

            fused_m_np = fused_m[0].cpu().numpy()  # already in raw coordinates
            all_fused += fused_m_np

        # Average over all agents' fused beliefs
        return all_fused / K

    def _build_local_channel_feedback(
        self, received_meta: Optional[Dict[int, dict]],
    ) -> np.ndarray:
        """Summarize locally measurable reciprocal-link conditions.

        No acknowledgement or fusion-centre information is fabricated here.
        The first four entries use only packets physically received by this
        UAV; the final two are the receiver's known channel configuration:

        0. fresh sender fraction,
        1. mean received-SNR margin,
        2. mean latency slack,
        3. mean packet age,
        4. active/nominal deadline log-ratio,
        5. active/nominal SNR-threshold shift.
        """
        metadata = received_meta or {}
        peer_count = max(self.K - 1, 1)
        fresh_fraction = (
            sum(float(md.get('age_frames', 0.0)) <= 0.0
                for md in metadata.values())
            / peer_count)

        if metadata:
            snr_margin = float(np.mean([
                np.tanh((
                    float(md.get('snr_db',
                                 self._active_comm_snr_threshold_db))
                    - self._active_comm_snr_threshold_db) / 10.0)
                for md in metadata.values()
            ]))
            latency_slack = float(np.mean([
                np.clip(
                    1.0 - float(md.get('latency_s', 0.0))
                    / max(self._active_comm_deadline_s, 1e-9),
                    -1.0, 1.0)
                for md in metadata.values()
            ]))
            mean_aoi = float(np.mean([
                np.clip(float(md.get('age_frames', 0.0)) / 10.0, 0.0, 1.0)
                for md in metadata.values()
            ]))
        else:
            snr_margin = 0.0
            latency_slack = 0.0
            mean_aoi = 1.0

        deadline_ratio = float(np.clip(np.log10(
            max(self._active_comm_deadline_s, 1e-9)
            / max(self._nominal_comm_deadline_s, 1e-9)), -1.0, 1.0))
        threshold_shift = float(np.tanh((
            self._active_comm_snr_threshold_db
            - self._nominal_comm_snr_threshold_db) / 10.0))
        return np.asarray([
            fresh_fraction, snr_margin, latency_slack, mean_aoi,
            deadline_ratio, threshold_shift,
        ], dtype=np.float64)

    def _build_observations(self) -> Dict[int, np.ndarray]:
        """Build local observations for all UAVs."""
        uav_states = [u.get_state() for u in self.uavs]
        if self.belief_mgr is None:
            raise RuntimeError(
                "observation construction requires reset belief state")
        # Observation construction consumes read-only local posterior arrays.
        # Passing those arrays directly avoids materializing K*Q BeliefState
        # objects per frame; the information boundary is unchanged because
        # agent k still indexes only row k unless neighbor state was explicitly
        # enabled by the configuration.
        belief_mean = np.asarray(
            self.belief_mgr.mean[:, :, :4], dtype=np.float64)
        belief_cov_diag = np.diagonal(
            self.belief_mgr.cov[:, :, :4, :4],
            axis1=-2,
            axis2=-1,
        )
        belief_aoi = np.asarray(self.belief_mgr.aoi, dtype=np.float64)

        # Diagnostic: feed true target state instead of beliefs
        # In tracking-free mode target locations are mission-known sensing
        # objects, not privileged tracking truth. Reuse the zero-covariance
        # observation layout to preserve network dimensions/checkpoints.
        oracle = (getattr(self.cfg.marl, 'oracle_obs', False)
                  or not self.tracking_enabled)
        oracle_targets = None
        if oracle:
            oracle_targets = np.array([
                [t.state[0], t.state[1], *t.get_velocity()]
                for t in self.targets
            ], dtype=np.float64)

        obs = {}
        history_frames = getattr(self.cfg.marl, 'obs_history_frames', 1)
        if not hasattr(self, '_prev_obs_deque'):
            self._prev_obs_deque: dict = {}  # {agent_id: deque of prev frames}
        strict_batch = None
        if (
            bool(getattr(
                self.cfg.marl, 'strict_batch_observation_enabled', True))
            and
            self._comm_mode == 'cost_aware'
            and not self.obs_builder.expose_neighbor_state
            and not self.obs_builder.use_p0_global_info
            and self.obs_builder.use_relative_features
            and self.obs_builder.use_comm_tokens
            and self.obs_builder.comm_payload_mode == 'target_tokens'
            and not self.obs_builder.use_channel_feedback
        ):
            local_detection = np.stack([
                np.asarray(
                    self.prev_P_D_local.get(
                        node, np.zeros(self.Q, dtype=np.float64)),
                    dtype=np.float64,
                )
                for node in range(self.K)
            ], axis=0)
            strict_batch = self.obs_builder.build_strict_local_obs_batch(
                uav_states,
                prev_p_d=local_detection,
                belief_mean=belief_mean,
                belief_cov_diag=belief_cov_diag,
                belief_aoi=belief_aoi,
                comm_msgs=self._received_comm_msgs,
                comm_metadata=self._received_comm_meta,
                own_token_masks=self._last_sent_comm_token_masks,
                own_target_claims=self._last_sent_comm_target_claims,
                oracle_targets=oracle_targets,
            )
        for k in range(self.K):
            # P1 FIX: per-UAV LOCAL detection confidence (RX-only).
            # No fallback to global prev_P_D — strict decentralized mode.
            # First frame (t=0) or UAV with no RX role gets zeros.
            local_pd = self.prev_P_D_local.get(k)
            if local_pd is None:
                local_pd = np.zeros(self.Q, dtype=np.float64)
            if strict_batch is not None:
                cur = strict_batch[k]
            else:
                if self._comm_mode == 'cost_aware':
                    received = self._received_comm_msgs.get(k, {})
                    received_meta = self._received_comm_meta.get(k, {})
                else:
                    received = self._comm_msgs if self._comm_msgs else None
                    received_meta = None
                channel_feedback = (
                    self._build_local_channel_feedback(received_meta)
                    if self._comm_channel_feedback_rate_enabled else None)
                cur = self.obs_builder.build_local_obs(
                    k, uav_states, None, local_pd,
                    oracle_targets=oracle_targets,
                    selected_set=getattr(self, '_last_selected_set', []),
                    deflection_entries=getattr(
                        self, '_last_deflection_entries', None),
                    comm_msgs=received,
                    comm_metadata=received_meta,
                    own_token_mask=self._last_sent_comm_token_masks.get(k),
                    own_target_claims=(
                        self._last_sent_comm_target_claims.get(k)),
                    channel_feedback=channel_feedback,
                    belief_mean=belief_mean,
                    belief_cov_diag=belief_cov_diag,
                    belief_aoi=belief_aoi,
                )
            if history_frames > 1:
                import collections
                if k not in self._prev_obs_deque:
                    self._prev_obs_deque[k] = collections.deque(maxlen=history_frames-1)
                dq = self._prev_obs_deque[k]
                # Fill with zeros on first call
                while len(dq) < history_frames - 1:
                    dq.append(np.zeros_like(cur))
                # Concat all prev frames + current
                parts = list(dq) + [cur]
                cur = np.concatenate(parts)
                dq.append(cur[-(len(cur)//history_frames):])  # store only current frame portion
            obs[k] = cur
        return obs

    def _persistent_rngs(self) -> List[np.random.Generator]:
        """Distinct Generator objects held by PERSISTENT (non-snapshotted)
        components, which must be restored in place.

        Subtlety: wrapper.reset(seed) replaces self.rng with a fresh generator,
        but components built in __init__ (notably deflection_computer, whose
        Rician/LoS fading draws from its OWN rng) keep a reference to the
        ORIGINAL generator. That stream is therefore separate from self.rng and
        must be snapshotted independently, or replays diverge in d_eff/P_D even
        though positions are identical. We scan self.rng plus every component's
        `.rng` attribute and de-duplicate by object identity.
        """
        seen: Dict[int, np.random.Generator] = {}
        candidates = [self.rng]
        for v in self.__dict__.values():
            r = getattr(v, 'rng', None)
            if isinstance(r, np.random.Generator):
                candidates.append(r)
        for g in candidates:
            if isinstance(g, np.random.Generator):
                seen.setdefault(id(g), g)
        return list(seen.values())

    def get_state(self) -> Dict:
        """Snapshot all mutable simulation state for exact replay (P0).

        Captures the five mutable state objects (uavs, targets, belief_mgr, t,
        prev_P_D) plus fc_position and EVERY distinct RNG stream the sim draws
        from. The snapshot is fully deep-copied so it can be restored repeatedly
        and is decoupled from subsequent stepping. Used to confirm train/eval
        consistency by replaying the SAME frame through different decode modes.
        """
        return {
            't': self.t,
            'prev_obs': copy.deepcopy(self._prev_obs),
            'prev_obs_deque': copy.deepcopy(self._prev_obs_deque),
            'gru_hidden': copy.deepcopy(self._gru_hidden),
            'probe_miss_count': self._probe_miss_count.copy(),
            'active_comm_deadline_s': float(self._active_comm_deadline_s),
            'active_comm_snr_threshold_db': float(
                self._active_comm_snr_threshold_db),
            'cached_p0_solution': copy.deepcopy(self._cached_p0_solution),
            'last_solve_frame': int(self._last_solve_frame),
            'assignment_switched': bool(self._assignment_switched),
            'last_p0_solve_time_s': float(self._last_p0_solve_time_s),
            'last_selected_set': copy.deepcopy(getattr(
                self, '_last_selected_set', [])),
            'prev_P_D': None if self.prev_P_D is None else self.prev_P_D.copy(),
            'coord_pd_ema': (None if self._coord_pd_ema is None
                             else self._coord_pd_ema.copy()),
            'prev_P_D_local': {k: v.copy() for k, v in self.prev_P_D_local.items()},
            'comm_msgs': {k: v.copy() for k, v in self._comm_msgs.items()},
            'pending_comm_messages': {
                k: v.copy() for k, v in self._pending_comm_messages.items()},
            'pending_comm_rates': dict(self._pending_comm_rates),
            'pending_decision_sufficient_bits': dict(
                self._pending_decision_sufficient_bits),
            'pending_comm_token_masks': {
                k: v.copy() for k, v in
                self._pending_comm_token_masks.items()},
            'last_sent_comm_token_masks': {
                k: v.copy() for k, v in
                self._last_sent_comm_token_masks.items()},
            'last_sent_comm_target_claims': {
                k: v.copy() for k, v in
                self._last_sent_comm_target_claims.items()},
            'persistent_commitment_mask': (
                self._persistent_commitment_mask.copy()),
            'persistent_commitment_old_mask': (
                self._persistent_commitment_old_mask.copy()),
            'persistent_commitment_last_switch': (
                self._persistent_commitment_last_switch.copy()),
            'persistent_commitment_last_seen': (
                self._persistent_commitment_last_seen.copy()),
            'persistent_commitment_handover_remaining': (
                self._persistent_commitment_handover_remaining.copy()),
            'persistent_commitment_effective_mask': (
                self._persistent_commitment_effective_mask.copy()),
            'persistent_commitment_resolved_frame': int(
                self._persistent_commitment_resolved_frame),
            'persistent_commitment_metrics': copy.deepcopy(
                self._persistent_commitment_metrics),
            'qpd_queue': self._qpd_queue.copy(),
            'qpd_previous_queue': self._qpd_previous_queue.copy(),
            'qpd_bid': self._qpd_bid.copy(),
            'qpd_previous_bid': self._qpd_previous_bid.copy(),
            'qpd_primal': self._qpd_primal.copy(),
            'qpd_previous_primal': self._qpd_previous_primal.copy(),
            'qpd_target_price': self._qpd_target_price.copy(),
            'qpd_capability': self._qpd_capability.copy(),
            'qpd_age': self._qpd_age.copy(),
            'qpd_send_mask': self._qpd_send_mask.copy(),
            'pending_qpd_protocol': {
                k: v.copy()
                for k, v in self._pending_qpd_protocol.items()},
            'qpd_commitment_mask': self._qpd_commitment_mask.copy(),
            'qpd_received_primal': self._qpd_received_primal.copy(),
            'qpd_received_queue': self._qpd_received_queue.copy(),
            'qpd_received_last_seen': (
                self._qpd_received_last_seen.copy()),
            'qpd_metrics': copy.deepcopy(self._qpd_metrics),
            'qpd_last_submission_frame': int(
                self._qpd_last_submission_frame),
            'pending_hyperedge_protocol': {
                k: v.copy()
                for k, v in self._pending_hyperedge_protocol.items()},
            'pending_hyperedge_protocol_frame': dict(
                self._pending_hyperedge_protocol_frame),
            'hyperedge_local_offer': self._hyperedge_local_offer.copy(),
            'hyperedge_received_offer': (
                self._hyperedge_received_offer.copy()),
            'hyperedge_received_last_seen': (
                self._hyperedge_received_last_seen.copy()),
            'hyperedge_consensus_streak': (
                self._hyperedge_consensus_streak.copy()),
            'hyperedge_selected_set': tuple(
                self._hyperedge_selected_set),
            'hyperedge_last_update_frame': int(
                self._hyperedge_last_update_frame),
            'hyperedge_metrics': copy.deepcopy(self._hyperedge_metrics),
            'hyperedge_public_gain_views': (
                self._hyperedge_public_gain_views.copy()),
            'hyperedge_public_full_views': (
                self._hyperedge_public_full_views.copy()),
            'distributed_movement_target': (
                self._distributed_movement_target.copy()),
            'distributed_movement_local_assignment': (
                self._distributed_movement_local_assignment.copy()),
            'distributed_movement_last_update': (
                self._distributed_movement_last_update.copy()),
            'distributed_movement_anchor_target_xy': (
                self._distributed_movement_anchor_target_xy.copy()),
            'distributed_movement_anchor_last_seen': (
                self._distributed_movement_anchor_last_seen.copy()),
            'distributed_bistatic_gate_latch': (
                self._distributed_bistatic_gate_latch.copy()),
            'distributed_role_capacity_tx_resp': (
                self._distributed_role_capacity_tx_resp.copy()),
            'distributed_role_capacity_rx_resp': (
                self._distributed_role_capacity_rx_resp.copy()),
            'distributed_role_capacity_bottleneck': (
                self._distributed_role_capacity_bottleneck.copy()),
            'distributed_role_capacity_last_reassign': (
                self._distributed_role_capacity_last_reassign.copy()),
            'distributed_gap_tx_resp': (
                self._distributed_gap_tx_resp.copy()),
            'distributed_gap_rx_resp': (
                self._distributed_gap_rx_resp.copy()),
            'pending_comm_power_fractions': dict(
                self._pending_comm_power_fractions),
            'pending_sensing_weights': {
                k: v.copy() for k, v in self._pending_sensing_weights.items()},
            'current_comm_power_w': self._current_comm_power_w.copy(),
            'current_sensing_power_w': self._current_sensing_power_w.copy(),
            'distributed_primal_dual_power_controller': (
                None
                if self._distributed_primal_dual_power_controller is None
                else self._distributed_primal_dual_power_controller.state_dict()),
            'distributed_replicated_previous_power': (
                self._distributed_replicated_previous_power.copy()),
            'distributed_replicated_local_power_cache': (
                self._distributed_replicated_local_power_cache.copy()),
            'distributed_replicated_local_price_cache': (
                self._distributed_replicated_local_price_cache.copy()),
            'distributed_replicated_local_cache_valid': (
                self._distributed_replicated_local_cache_valid.copy()),
            'isac_sensing_battery_before': (
                None if self._isac_sensing_battery_before is None
                else self._isac_sensing_battery_before.copy()),
            'last_analytical_power_balance_error': float(
                self._last_analytical_power_balance_error),
            'last_analytical_power_budget_violation_w': float(
                self._last_analytical_power_budget_violation_w),
            'last_analytical_unused_power_w': float(
                self._last_analytical_unused_power_w),
            'last_isac_metrics': copy.deepcopy(self._last_isac_metrics),
            'last_movement_safety_metrics': copy.deepcopy(
                self._last_movement_safety_metrics),
            'external_structure_edge_values': (
                None
                if self._external_structure_edge_values is None
                else self._external_structure_edge_values.copy()),
            'external_structure_candidate_mask': (
                None
                if self._external_structure_candidate_mask is None
                else self._external_structure_candidate_mask.copy()),
            'dynamic_local_search_state': (
                None
                if self._dynamic_local_search_coordinator is None
                else self._dynamic_local_search_coordinator.get_state()),
            'pending_structure_student_protocol': {
                k: v.copy()
                for k, v in
                self._pending_structure_student_protocol.items()},
            'structure_student_public_protocol': (
                self._structure_student_public_protocol.copy()),
            'structure_student_public_valid': (
                self._structure_student_public_valid.copy()),
            'structure_student_public_last_seen': (
                self._structure_student_public_last_seen.copy()),
            'structure_student_mailbox': copy.deepcopy(
                self._structure_student_mailbox),
            'structure_student_metrics': copy.deepcopy(
                self._structure_student_metrics),
            'received_comm_msgs': copy.deepcopy(self._received_comm_msgs),
            'received_comm_meta': copy.deepcopy(self._received_comm_meta),
            'comm_mailbox': copy.deepcopy(self._comm_mailbox),
            'last_comm_stats': copy.deepcopy(self._last_comm_stats),
            'comm_transport_state': (
                None if self._inter_uav_comm is None
                else self._inter_uav_comm.get_channel_state()),
            'fc_position': None if self.fc_position is None else self.fc_position.copy(),
            'uavs': copy.deepcopy(self.uavs),
            'targets': copy.deepcopy(self.targets),
            'belief_mgr': copy.deepcopy(self.belief_mgr),
            # Hold the actual persistent generator objects + their states so we
            # can restore each in place (they survive set_state unchanged).
            'rng_refs': [(g, copy.deepcopy(g.bit_generator.state))
                         for g in self._persistent_rngs()],
        }

    def set_state(self, state: Dict) -> None:
        """Restore a snapshot produced by get_state().

        Deep-copies on restore so the snapshot stays reusable. Restores every
        persistent RNG stream in place, then re-aliases targets/belief_mgr onto
        self.rng (a naive deepcopy would fork independent generators and
        desynchronize sampling).
        """
        self.t = state['t']
        self._prev_obs = copy.deepcopy(state.get('prev_obs', {}))
        self._prev_obs_deque = copy.deepcopy(
            state.get('prev_obs_deque', {}))
        self._gru_hidden = copy.deepcopy(state.get('gru_hidden', {}))
        self._probe_miss_count = np.asarray(state.get(
            'probe_miss_count', np.zeros(self.Q, dtype=np.int32)),
            dtype=np.int32).copy()
        self._active_comm_deadline_s = float(state.get(
            'active_comm_deadline_s', self._nominal_comm_deadline_s))
        self._active_comm_snr_threshold_db = float(state.get(
            'active_comm_snr_threshold_db',
            self._nominal_comm_snr_threshold_db))
        self._cached_p0_solution = copy.deepcopy(
            state.get('cached_p0_solution'))
        self._last_solve_frame = int(state.get('last_solve_frame', -1))
        self._assignment_switched = bool(
            state.get('assignment_switched', False))
        self._last_p0_solve_time_s = float(
            state.get('last_p0_solve_time_s', 0.0))
        self._last_selected_set = copy.deepcopy(
            state.get('last_selected_set', []))
        self.prev_P_D = None if state['prev_P_D'] is None else state['prev_P_D'].copy()
        coord_pd_ema = state.get('coord_pd_ema')
        self._coord_pd_ema = (None if coord_pd_ema is None
                              else np.asarray(coord_pd_ema).copy())
        self.prev_P_D_local = {k: v.copy() for k, v in state.get('prev_P_D_local', {}).items()}
        self._comm_msgs = {
            k: v.copy() for k, v in state.get('comm_msgs', {}).items()}
        self._pending_comm_messages = {
            k: v.copy() for k, v in state.get('pending_comm_messages', {}).items()}
        self._pending_comm_rates = dict(state.get('pending_comm_rates', {}))
        self._pending_decision_sufficient_bits = {
            int(k): int(v) for k, v in state.get(
                'pending_decision_sufficient_bits', {}).items()
        }
        self._pending_comm_token_masks = {
            k: v.copy() for k, v in state.get(
                'pending_comm_token_masks', {}).items()}
        self._last_sent_comm_token_masks = {
            int(k): np.asarray(v, dtype=np.float64).copy()
            for k, v in state.get(
                'last_sent_comm_token_masks', {}).items()}
        self._last_sent_comm_target_claims = {
            int(k): np.asarray(v, dtype=np.float64).copy()
            for k, v in state.get(
                'last_sent_comm_target_claims', {}).items()}
        self._persistent_commitment_mask = np.asarray(state.get(
            'persistent_commitment_mask',
            np.zeros((self.K, self.Q), dtype=bool)),
            dtype=bool).copy()
        self._persistent_commitment_old_mask = np.asarray(state.get(
            'persistent_commitment_old_mask',
            np.zeros((self.K, self.Q), dtype=bool)),
            dtype=bool).copy()
        self._persistent_commitment_last_switch = np.asarray(state.get(
            'persistent_commitment_last_switch',
            np.full(self.K, FRAME_NOT_APPLICABLE, dtype=np.int64)),
            dtype=np.int64).copy()
        self._persistent_commitment_last_seen = np.asarray(state.get(
            'persistent_commitment_last_seen',
            np.full(self.K, FRAME_NOT_APPLICABLE, dtype=np.int64)),
            dtype=np.int64).copy()
        self._persistent_commitment_handover_remaining = np.asarray(state.get(
            'persistent_commitment_handover_remaining',
            np.zeros(self.K, dtype=np.int64)),
            dtype=np.int64).copy()
        self._persistent_commitment_effective_mask = np.asarray(state.get(
            'persistent_commitment_effective_mask',
            self._persistent_commitment_mask),
            dtype=bool).copy()
        self._persistent_commitment_resolved_frame = int(state.get(
            'persistent_commitment_resolved_frame', -1))
        self._persistent_commitment_metrics = copy.deepcopy(state.get(
            'persistent_commitment_metrics', {}))
        self._qpd_queue = np.asarray(state.get(
            'qpd_queue', np.zeros((self.K, self.Q))),
            dtype=np.float64).copy()
        self._qpd_previous_queue = np.asarray(state.get(
            'qpd_previous_queue', np.zeros((self.K, self.Q))),
            dtype=np.float64).copy()
        self._qpd_bid = np.asarray(state.get(
            'qpd_bid', np.zeros((self.K, self.Q))),
            dtype=np.float64).copy()
        self._qpd_previous_bid = np.asarray(state.get(
            'qpd_previous_bid', np.zeros((self.K, self.Q))),
            dtype=np.float64).copy()
        initial_qpd = min(
            1.0, self._qpd_row_capacity / max(self.Q, 1))
        self._qpd_primal = np.asarray(state.get(
            'qpd_primal',
            np.full((self.K, self.Q), initial_qpd)),
            dtype=np.float64).copy()
        self._qpd_previous_primal = np.asarray(state.get(
            'qpd_previous_primal', self._qpd_primal),
            dtype=np.float64).copy()
        self._qpd_target_price = np.asarray(state.get(
            'qpd_target_price', np.zeros((self.K, self.Q))),
            dtype=np.float64).copy()
        self._qpd_capability = np.asarray(state.get(
            'qpd_capability', np.zeros((self.K, self.Q))),
            dtype=np.float64).copy()
        self._qpd_age = np.asarray(state.get(
            'qpd_age', np.zeros((self.K, self.Q))),
            dtype=np.int64).copy()
        self._qpd_send_mask = np.asarray(state.get(
            'qpd_send_mask', np.zeros((self.K, self.Q))),
            dtype=bool).copy()
        self._pending_qpd_protocol = {
            int(k): np.asarray(v, dtype=np.float64).copy()
            for k, v in state.get(
                'pending_qpd_protocol', {}).items()}
        self._qpd_commitment_mask = np.asarray(state.get(
            'qpd_commitment_mask', np.zeros((self.K, self.Q))),
            dtype=bool).copy()
        self._qpd_received_primal = np.asarray(state.get(
            'qpd_received_primal',
            np.zeros((self.K, self.K, self.Q))),
            dtype=np.float64).copy()
        self._qpd_received_queue = np.asarray(state.get(
            'qpd_received_queue',
            np.zeros((self.K, self.K, self.Q))),
            dtype=np.float64).copy()
        self._qpd_received_last_seen = np.asarray(state.get(
            'qpd_received_last_seen',
            np.full((self.K, self.K, self.Q), FRAME_NOT_APPLICABLE)),
            dtype=np.int64).copy()
        self._qpd_metrics = copy.deepcopy(state.get('qpd_metrics', {}))
        self._qpd_last_submission_frame = int(state.get(
            'qpd_last_submission_frame', -1))
        self._pending_hyperedge_protocol = {
            int(k): np.asarray(v, dtype=np.float64).copy()
            for k, v in state.get(
                'pending_hyperedge_protocol', {}).items()}
        self._pending_hyperedge_protocol_frame = {
            int(k): int(v)
            for k, v in state.get(
                'pending_hyperedge_protocol_frame', {}).items()}
        self._hyperedge_local_offer = np.asarray(state.get(
            'hyperedge_local_offer',
            np.zeros((
                self.K, self.Q, self._hyperedge_protocol_dim))),
            dtype=np.float64).copy()
        self._hyperedge_received_offer = np.asarray(state.get(
            'hyperedge_received_offer',
            np.zeros((
                self.K, self.K, self.Q,
                self._hyperedge_protocol_dim))),
            dtype=np.float64).copy()
        self._hyperedge_received_last_seen = np.asarray(state.get(
            'hyperedge_received_last_seen',
            np.full((self.K, self.K, self.Q), FRAME_NOT_APPLICABLE)),
            dtype=np.int64).copy()
        self._hyperedge_consensus_streak = np.asarray(state.get(
            'hyperedge_consensus_streak',
            np.zeros((self.K, self.K, self.Q))),
            dtype=np.int64).copy()
        self._hyperedge_selected_set = tuple(
            tuple(int(value) for value in edge)
            for edge in state.get('hyperedge_selected_set', ()))
        self._hyperedge_last_update_frame = int(state.get(
            'hyperedge_last_update_frame', FRAME_NOT_APPLICABLE))
        self._hyperedge_metrics = copy.deepcopy(state.get(
            'hyperedge_metrics', {}))
        self._hyperedge_public_gain_views = np.asarray(state.get(
            'hyperedge_public_gain_views',
            np.zeros((self.K, self.K, self.Q))),
            dtype=np.float64).copy()
        self._hyperedge_public_full_views = np.asarray(state.get(
            'hyperedge_public_full_views',
            np.zeros(self.K, dtype=bool)),
            dtype=bool).copy()
        self._distributed_movement_target = np.asarray(state.get(
            'distributed_movement_target',
            np.full(self.K, -1, dtype=np.int64)),
            dtype=np.int64).copy()
        self._distributed_movement_local_assignment = np.asarray(state.get(
            'distributed_movement_local_assignment',
            np.full((self.K, self.K), -1, dtype=np.int64)),
            dtype=np.int64).copy()
        self._distributed_movement_last_update = np.asarray(state.get(
            'distributed_movement_last_update',
            np.full(self.K, FRAME_NOT_APPLICABLE, dtype=np.int64)),
            dtype=np.int64).copy()
        self._distributed_movement_anchor_target_xy = np.asarray(state.get(
            'distributed_movement_anchor_target_xy',
            np.zeros((self.K, self.K, self.Q, 2), dtype=np.float64)),
            dtype=np.float64).copy()
        self._distributed_movement_anchor_last_seen = np.asarray(state.get(
            'distributed_movement_anchor_last_seen',
            np.full(
                (self.K, self.K, self.Q), FRAME_NOT_APPLICABLE, dtype=np.int64)),
            dtype=np.int64).copy()
        self._distributed_bistatic_gate_latch = np.asarray(state.get(
            'distributed_bistatic_gate_latch',
            np.full(self.K, -1, dtype=np.int8)),
            dtype=np.int8).copy()
        self._distributed_role_capacity_tx_resp = np.asarray(state.get(
            'distributed_role_capacity_tx_resp',
            np.zeros((self.K, self.K, self.Q), dtype=np.int8)),
            dtype=np.int8).copy()
        self._distributed_role_capacity_rx_resp = np.asarray(state.get(
            'distributed_role_capacity_rx_resp',
            np.zeros((self.K, self.K, self.Q), dtype=np.int8)),
            dtype=np.int8).copy()
        self._distributed_role_capacity_bottleneck = np.asarray(state.get(
            'distributed_role_capacity_bottleneck',
            np.full(self.K, np.inf, dtype=np.float64)),
            dtype=np.float64).copy()
        self._distributed_role_capacity_last_reassign = np.asarray(state.get(
            'distributed_role_capacity_last_reassign',
            np.full(self.K, FRAME_NOT_APPLICABLE, dtype=np.int64)),
            dtype=np.int64).copy()
        self._distributed_gap_tx_resp = np.asarray(state.get(
            'distributed_gap_tx_resp',
            np.zeros((self.K, self.K, self.Q), dtype=np.int8)),
            dtype=np.int8).copy()
        self._distributed_gap_rx_resp = np.asarray(state.get(
            'distributed_gap_rx_resp',
            np.zeros((self.K, self.K, self.Q), dtype=np.int8)),
            dtype=np.int8).copy()
        self._pending_comm_power_fractions = dict(
            state.get('pending_comm_power_fractions', {}))
        self._pending_sensing_weights = {
            k: v.copy() for k, v in state.get(
                'pending_sensing_weights', {}).items()}
        self._current_comm_power_w = np.asarray(state.get(
            'current_comm_power_w', np.zeros(self.K)), dtype=np.float64).copy()
        self._current_sensing_power_w = np.asarray(state.get(
            'current_sensing_power_w', np.full(
                (self.K, self.Q),
                self._sensing_power_cap_w / max(self.Q, 1))),
            dtype=np.float64).copy()
        controller_state = state.get(
            'distributed_primal_dual_power_controller')
        if self._distributed_primal_dual_power_enabled:
            if controller_state is None:
                self._distributed_primal_dual_power_controller = None
            else:
                controller = self._new_distributed_primal_dual_controller()
                previous_power = controller_state.get('previous_power_w')
                previous_prices = controller_state.get('previous_target_prices')
                if previous_power is not None and (
                        np.asarray(previous_power).shape != (self.K, self.Q)):
                    raise ValueError(
                        'distributed primal-dual controller state has an '
                        'incompatible power shape')
                if previous_prices is not None and (
                        np.asarray(previous_prices).shape != (self.Q,)):
                    raise ValueError(
                        'distributed primal-dual controller state has an '
                        'incompatible price shape')
                controller.load_state_dict(controller_state)
                self._distributed_primal_dual_power_controller = controller
        else:
            self._distributed_primal_dual_power_controller = None
        self._distributed_replicated_previous_power = np.asarray(state.get(
            'distributed_replicated_previous_power',
            self._current_sensing_power_w), dtype=np.float64).copy()
        self._distributed_replicated_local_power_cache = np.asarray(state.get(
            'distributed_replicated_local_power_cache',
            np.zeros((self.K, self.K, self.Q), dtype=np.float64),
        ), dtype=np.float64).copy()
        self._distributed_replicated_local_price_cache = np.asarray(state.get(
            'distributed_replicated_local_price_cache',
            np.full(
                (self.K, self.Q), 1.0 / max(self.Q, 1), dtype=np.float64),
        ), dtype=np.float64).copy()
        self._distributed_replicated_local_cache_valid = np.asarray(state.get(
            'distributed_replicated_local_cache_valid',
            np.zeros(self.K, dtype=bool),
        ), dtype=bool).copy()
        battery_before = state.get('isac_sensing_battery_before')
        self._isac_sensing_battery_before = (
            None if battery_before is None
            else np.asarray(battery_before, dtype=np.float64).copy())
        self._last_analytical_power_balance_error = float(state.get(
            'last_analytical_power_balance_error', 0.0))
        self._last_analytical_power_budget_violation_w = float(state.get(
            'last_analytical_power_budget_violation_w', 0.0))
        self._last_analytical_unused_power_w = float(state.get(
            'last_analytical_unused_power_w', 0.0))
        self._last_isac_metrics = copy.deepcopy(
            state.get('last_isac_metrics', {}))
        self._last_movement_safety_metrics = copy.deepcopy(
            state.get('last_movement_safety_metrics', {}))
        external_structure = state.get(
            'external_structure_edge_values')
        self._external_structure_edge_values = (
            None if external_structure is None
            else np.asarray(
                external_structure, dtype=np.float64).copy())
        external_candidate = state.get(
            'external_structure_candidate_mask')
        self._external_structure_candidate_mask = (
            None if external_candidate is None
            else np.asarray(external_candidate, dtype=bool).copy())
        dynamic_state = state.get('dynamic_local_search_state')
        if (self._dynamic_local_search_coordinator is not None
                and dynamic_state is not None):
            self._dynamic_local_search_coordinator.set_state(dynamic_state)
        self._pending_structure_student_protocol = {
            int(k): np.asarray(v, dtype=np.float64).copy()
            for k, v in state.get(
                'pending_structure_student_protocol', {}).items()}
        self._structure_student_public_protocol = np.asarray(state.get(
            'structure_student_public_protocol',
            np.zeros((
                self.K,
                self.Q,
                self._structure_student_endpoint_width,
            ))),
            dtype=np.float64,
        ).copy()
        self._structure_student_public_valid = np.asarray(state.get(
            'structure_student_public_valid',
            np.zeros(self.K, dtype=bool)),
            dtype=bool,
        ).copy()
        self._structure_student_public_last_seen = np.asarray(state.get(
            'structure_student_public_last_seen',
            np.full(self.K, FRAME_NOT_APPLICABLE, dtype=np.int64)),
            dtype=np.int64,
        ).copy()
        self._structure_student_mailbox = copy.deepcopy(state.get(
            'structure_student_mailbox', []))
        self._structure_student_metrics = copy.deepcopy(state.get(
            'structure_student_metrics', {}))
        self._received_comm_msgs = copy.deepcopy(
            state.get('received_comm_msgs', {}))
        self._received_comm_meta = copy.deepcopy(
            state.get('received_comm_meta', {}))
        self._comm_mailbox = copy.deepcopy(state.get('comm_mailbox', []))
        self._last_comm_stats = copy.deepcopy(
            state.get('last_comm_stats', CommunicationStepStats()))
        if self._inter_uav_comm is not None:
            transport_state = state.get('comm_transport_state')
            if transport_state is None:
                self._inter_uav_comm.reset_channel_state(self.K)
            else:
                self._inter_uav_comm.set_channel_state(transport_state)
        self.fc_position = None if state['fc_position'] is None else state['fc_position'].copy()
        self.uavs = copy.deepcopy(state['uavs'])
        self.targets = copy.deepcopy(state['targets'])
        self.belief_mgr = copy.deepcopy(state['belief_mgr'])
        # Restore each persistent RNG stream in place (same objects as snapshot).
        for g, st in state['rng_refs']:
            g.bit_generator.state = copy.deepcopy(st)
        # Re-alias the shared RNG onto restored sub-objects (see docstring).
        for tgt in self.targets:
            if hasattr(tgt, 'rng'):
                tgt.rng = self.rng
        if self.belief_mgr is not None and hasattr(self.belief_mgr, 'rng'):
            self.belief_mgr.rng = self.rng

    def get_obs_dim(self) -> int:
        """Effective observation dim (includes history stacking)."""
        base = self.obs_builder.get_obs_dim()
        h = getattr(self.cfg.marl, 'obs_history_frames', 1)
        return base * h

    def get_global_state(self) -> np.ndarray:
        """Build global state for centralized critic."""
        uav_states = [u.get_state() for u in self.uavs]
        target_positions = np.array([t.get_position_3d() for t in self.targets])
        target_velocities = np.array([
            np.array([*t.get_velocity(), 0.0]) for t in self.targets
        ])
        time_frac = self.t / max(self.T, 1)
        # Per-target mean belief uncertainty (trace of covariance)
        # Audit 2026-08-17: the comment says "mean over UAVs" but only UAV-0's
        # trace was fed to the centralized critic; average over all UAVs.
        if self.belief_mgr is not None:
            belief_cov_trace = np.array([
                np.mean([np.trace(self.belief_mgr.cov[k, q])
                         for k in range(self.K)])
                for q in range(self.Q)
            ])
        else:
            belief_cov_trace = None
        return self.obs_builder.build_global_state(
            uav_states, target_positions, target_velocities, self.prev_P_D,
            time_frac=time_frac, belief_cov_trace=belief_cov_trace,
        )

    def _generate_safe_uav_positions(self) -> List[np.ndarray]:
        """Generate initial UAV positions satisfying safe distance constraint."""
        positions = []
        max_attempts = 1000

        for k in range(self.K):
            for attempt in range(max_attempts):
                pos = np.array([
                    self.rng.uniform(50, self.area_size[0] - 50),
                    self.rng.uniform(50, self.area_size[1] - 50),
                    float(self.height),
                ], dtype=np.float64)

                # Check safety with already placed UAVs
                safe = True
                for other_pos in positions:
                    if np.linalg.norm(pos - other_pos) < self.cfg.uav.d_safe:
                        safe = False
                        break

                if safe or attempt == max_attempts - 1:
                    positions.append(pos)
                    break

        return positions
