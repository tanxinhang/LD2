"""Environment core: orchestrates one simulation step.

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
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from uav_isac.utils.types import Action, P0Solution, UAVState, TargetState
from uav_isac.environment.uav import UAV
from uav_isac.environment.target import Target
from uav_isac.environment.belief import BeliefManager
from uav_isac.environment.observation import ObservationBuilder
from uav_isac.environment.action import ActionSpace
from uav_isac.environment.reward import (
    RewardComputer,
    compute_segmented_coordination_reward,
)
from uav_isac.environment.constraints import ConstraintChecker
from uav_isac.physical.deflection import DeflectionComputer
from uav_isac.physical.inner_solver import InnerSolver
from uav_isac.physical.detection import (
    compute_detection_probabilities,
    compute_target_utilities,
)
from uav_isac.physical.feasibility_oracle import (
    solve_maxmin_single_role_pairs,
)
from uav_isac.physical.evidence import (
    DETECTION_FUSION_MODES,
    DeflectionConfidenceQuantizer,
    EvidencePacketLayout,
    estimate_quantized_evidence_detection,
    receiver_deflection_from_selected,
    route_structured_evidence,
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
    MaxMinPowerResult,
    fixed_owner_gain_matrix,
)
from uav_isac.coordination.hyperedge import (
    decode_offer_stream,
    encode_offer_stream,
    factorized_endpoint_capabilities,
    mutual_endpoint_consensus,
    plan_local_hyperedges,
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
        self._detection_fusion_mode = str(getattr(
            ma, 'detection_fusion_mode', 'local_only')).strip().lower()
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
        self._analytical_structure_ranking_enabled = bool(getattr(
            ma, 'analytical_structure_ranking_enabled', False))
        if (self._analytical_structure_ranking_enabled
                and not self._analytical_sensing_power_enabled):
            raise ValueError(
                'analytical_structure_ranking_enabled requires '
                'analytical_sensing_power_enabled')
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
        if (self._analytical_movement_enabled
                and not self._analytical_structure_ranking_enabled):
            raise ValueError(
                'analytical_movement_enabled requires '
                'analytical_structure_ranking_enabled')
        self._analytical_movement_candidates_enabled = bool(getattr(
            ma, 'analytical_movement_candidates_enabled', False))
        self._last_analytical_power_balance_error = 0.0
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
            use_los_prob=getattr(ch, 'use_los_prob', False),
            los_a=getattr(ch, 'los_a', 4.88), los_b=getattr(ch, 'los_b', 0.43),
            eta_los_dB=getattr(ch, 'eta_los_dB', 0.1),
            eta_nlos_dB=getattr(ch, 'eta_nlos_dB', 21.0),
            use_swerling=getattr(ch, 'use_swerling', False),
            use_report_link=self.ground_communication_enabled,
        )

        # Truncate/pad omega_q to match Q
        omega_q = np.array(ta.omega_q[:self.Q]) if len(ta.omega_q) >= self.Q \
                  else np.ones(self.Q) / self.Q

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
                'endpoint_proxy', 'physical_reconstructable'}:
            raise ValueError(
                'hyperedge_pair_score_mode must be endpoint_proxy or '
                'physical_reconstructable')
        self._hyperedge_state_stream_enabled = bool(getattr(
            ma, 'hyperedge_state_stream_enabled', False))
        self._hyperedge_protocol_dim = (
            7 if self._hyperedge_state_stream_enabled else 3)
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
            if (self._hyperedge_pair_score_mode == 'physical_reconstructable'
                    and not self._hyperedge_state_stream_enabled):
                raise ValueError(
                    'physical-reconstructable hyperedge scores require the '
                    'position/velocity state stream')
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
        if (self._distributed_target_commitment_enabled
                and not self._joint_isac_power_enabled):
            raise ValueError(
                'distributed target commitments require joint ISAC power')
        # B6: P0 ranks on fused belief (deployable) vs true geometry (oracle).
        self.p0_uses_belief = bool(getattr(self.cfg.marl, 'p0_uses_belief', False))
        # B7: gate belief update by a Bernoulli(P_D) detection event.
        self.belief_detection_sampling = bool(getattr(self.cfg.marl, 'belief_detection_sampling', False))
        # B8: neighbor belief fusion via multi-head attention + CI.
        self.neighbor_belief_fusion = bool(getattr(self.cfg.marl, 'neighbor_belief_fusion', False))
        # B3: Uncertainty-aware P0 scoring
        self.p0_beta_uncertainty = float(getattr(self.cfg.marl, 'p0_beta_uncertainty', 0.0))
        self.p0_eta_aoi = float(getattr(self.cfg.marl, 'p0_eta_aoi', 0.0))
        self._belief_fusion_module = None
        if self.neighbor_belief_fusion:
            import torch as _torch
            from uav_isac.agents.neighbor_attention import NeighborBeliefFusion
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
        self._last_sent_comm_token_masks: Dict[int, np.ndarray] = {}
        self._last_sent_comm_target_claims: Dict[int, np.ndarray] = {}
        self._persistent_commitment_mask = np.zeros(
            (self.K, self.Q), dtype=bool)
        self._persistent_commitment_old_mask = np.zeros(
            (self.K, self.Q), dtype=bool)
        self._persistent_commitment_last_switch = np.full(
            self.K, -10**9, dtype=np.int64)
        self._persistent_commitment_last_seen = np.full(
            self.K, -10**9, dtype=np.int64)
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
            (self.K, self.K, self.Q), -10**9, dtype=np.int64)
        self._qpd_metrics: Dict[str, object] = {}
        self._qpd_last_submission_frame = -1
        self._pending_hyperedge_protocol: Dict[int, np.ndarray] = {}
        self._hyperedge_local_offer = np.zeros(
            (self.K, self.Q, self._hyperedge_protocol_dim),
            dtype=np.float64)
        self._hyperedge_received_offer = np.zeros(
            (self.K, self.K, self.Q, self._hyperedge_protocol_dim),
            dtype=np.float64)
        self._hyperedge_received_last_seen = np.full(
            (self.K, self.K, self.Q), -10**9, dtype=np.int64)
        self._hyperedge_consensus_streak = np.zeros(
            (self.K, self.K, self.Q), dtype=np.int64)
        self._hyperedge_selected_set: Tuple[Tuple[int, int, int], ...] = tuple()
        self._hyperedge_last_update_frame = -10**9
        self._hyperedge_metrics: Dict[str, object] = {}
        self._pending_comm_power_fractions: Dict[int, float] = {}
        self._pending_sensing_weights: Dict[int, np.ndarray] = {}
        self._current_comm_power_w = np.zeros(self.K, dtype=np.float64)
        self._current_sensing_power_w = np.full(
            (self.K, self.Q), float(ua.P_sense), dtype=np.float64)
        self._last_isac_metrics: Dict[str, object] = {}
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
            self.K, -10**9, dtype=np.int64)
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
            )
        if self._detection_fusion_mode == 'u2u_distributed':
            self._evidence_topk = max(
                1, int(getattr(ma, 'evidence_packet_topk', 1)))
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
                    'normal', 'zero', 'value_roll'}:
                raise ValueError(
                    'evidence_packet_content_mode must be '
                    'normal, zero, or value_roll')
            self._evidence_packet_layout = EvidencePacketLayout(
                num_agents=self.K,
                num_targets=self.Q,
                header_bits=int(getattr(ma, 'comm_header_bits', 64)),
                timestamp_bits=16,
                confidence_bits=self._evidence_confidence_bits,
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

    def reset(self) -> Tuple[Dict[int, np.ndarray], Dict]:
        """Reset the environment to initial state.

        Returns:
            (observations_dict, info_dict)
        """
        self.t = 0
        self.prev_P_D = None
        self.prev_P_D_local = {}
        self._coord_pd_ema = None
        # D1.1-A audit state (DSH_LEX_AUDIT diagnostic) — fresh per episode.
        self._lex_mode = 'none'
        self._lex_t_star = None
        self._last_analytical_gain = None
        self._last_analytical_budget = None
        self._prev_obs = {}  # clear history on reset
        self._gru_hidden = {}  # clear GRU state on reset
        self._comm_msgs = {}
        self._pending_comm_messages = {}
        self._pending_comm_rates = {}
        self._pending_comm_token_masks = {}
        self._last_sent_comm_token_masks = {}
        self._last_sent_comm_target_claims = {}
        self._persistent_commitment_mask = np.zeros(
            (self.K, self.Q), dtype=bool)
        self._persistent_commitment_old_mask = np.zeros(
            (self.K, self.Q), dtype=bool)
        self._persistent_commitment_last_switch = np.full(
            self.K, -10**9, dtype=np.int64)
        self._persistent_commitment_last_seen = np.full(
            self.K, -10**9, dtype=np.int64)
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
            (self.K, self.K, self.Q), -10**9, dtype=np.int64)
        self._qpd_metrics = {}
        self._qpd_last_submission_frame = -1
        self._pending_hyperedge_protocol = {}
        self._hyperedge_local_offer = np.zeros(
            (self.K, self.Q, self._hyperedge_protocol_dim),
            dtype=np.float64)
        self._hyperedge_received_offer = np.zeros(
            (self.K, self.K, self.Q, self._hyperedge_protocol_dim),
            dtype=np.float64)
        self._hyperedge_received_last_seen = np.full(
            (self.K, self.K, self.Q), -10**9, dtype=np.int64)
        self._hyperedge_consensus_streak = np.zeros(
            (self.K, self.K, self.Q), dtype=np.int64)
        self._hyperedge_selected_set = tuple()
        self._hyperedge_last_update_frame = -10**9
        self._hyperedge_metrics = {}
        self._pending_comm_power_fractions = {}
        self._pending_sensing_weights = {}
        self._current_comm_power_w = np.zeros(self.K, dtype=np.float64)
        self._current_sensing_power_w = np.full(
            (self.K, self.Q), float(self.cfg.uav.P_sense), dtype=np.float64)
        self._last_isac_metrics = {}
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
            self.K, -10**9, dtype=np.int64)
        self._structure_student_mailbox = []
        self._structure_student_metrics = {}
        self._received_comm_msgs = {}
        self._received_comm_meta = {}
        self._comm_mailbox = []
        self._last_comm_stats = CommunicationStepStats()
        self._sample_comm_channel_profile()
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
            np.array([t.state[2], t.state[3], 0.0]) for t in self.targets
        ])
        self.belief_mgr = BeliefManager(
            K=self.K, Q=self.Q,
            initial_positions=true_positions,
            initial_velocities=true_velocities,
            dt=self.dt,
            sigma_a=self.cfg.target.sigma_a,
            rng=self.rng,
            motion_model=str(getattr(self.cfg.marl, 'belief_motion_model',
                            getattr(self.cfg.target, 'motion_model', 'CV'))),
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
        self._pending_comm_messages = {
            int(k): np.asarray(v, dtype=np.float64).copy()
            for k, v in messages.items()
        }
        self._pending_comm_rates = {
            int(k): int(v) for k, v in rate_indices.items()
        }
        self._pending_comm_token_masks = {}
        for k, raw_mask in (token_masks or {}).items():
            mask = np.asarray(raw_mask, dtype=np.float64).reshape(-1)
            if mask.shape != (self.Q,):
                raise ValueError(
                    f'token mask for UAV {k} must have shape {(self.Q,)}')
            self._pending_comm_token_masks[int(k)] = (
                mask > 0.5).astype(np.float64)
        if self._qpd_enabled:
            self._prepare_qpd_submission()
        if self._joint_isac_power_enabled:
            fractions = comm_power_fractions or {}
            weights = sensing_target_weights or {}
            self._pending_comm_power_fractions = {
                k: float(np.clip(fractions.get(k, 0.0),
                                 self._comm_power_fraction_min,
                                 self._comm_power_fraction_max))
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
                    w = np.asarray(weights.get(
                        k, np.ones(self.Q, dtype=np.float64)),
                        dtype=np.float64)
                if w.shape != (self.Q,):
                    raise ValueError(
                        f'sensing weights for UAV {k} must have shape {(self.Q,)}')
                w = np.maximum(w, 0.0)
                total = float(np.sum(w))
                self._pending_sensing_weights[k] = (
                    w / total if total > 1e-12
                    else np.full(self.Q, 1.0 / max(self.Q, 1)))
        if self._hyperedge_enabled:
            self._prepare_hyperedge_submission()

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
            self.K, -10**9, dtype=np.int64)
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

    def _prepare_hyperedge_submission(self) -> None:
        """Append a physically charged local Tx/Rx/deficit offer stream."""
        if len(self.uavs) != self.K or len(self.targets) != self.Q:
            return
        uav_xy = np.asarray([uav.pos[:2] for uav in self.uavs])
        target_xy = np.asarray([
            target.get_position_3d()[:2] for target in self.targets])
        distance = np.linalg.norm(
            uav_xy[:, None, :] - target_xy[None, :, :], axis=-1)

        self._pending_hyperedge_protocol = {}
        for sender in range(self.K):
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
                    distance[sender],
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
            self._hyperedge_local_offer[sender] = decoded
            self._pending_hyperedge_protocol[sender] = encoded

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
    ) -> None:
        decoded, active = self._decode_hyperedge_packet(
            protocol, token_mask)
        self._hyperedge_received_offer[
            int(receiver), int(sender), active] = decoded[active]
        self._hyperedge_received_last_seen[
            int(receiver), int(sender), active] = self.t

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
        target_xy = np.asarray([
            target.get_position_3d()[:2] for target in self.targets])
        distances = np.linalg.norm(
            uav_xy[:, None, :] - target_xy[None, :, :], axis=-1)
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
        b_eff = comm.bandwidth_hz / n_active
        gamma_th = float(10.0 ** (comm.snr_threshold_db / 10.0))
        n0_b = comm.kT * b_eff * comm.noise_figure_linear
        t_win = max(comm.deadline_s - comm.processing_delay_s, 1e-12)
        for i in range(K):
            if not active[i] or payload[i] <= 0.0:
                continue
            r_req = payload[i] / t_win
            gamma_rate = float(2.0 ** (r_req / b_eff) - 1.0)
            gamma_req = max(gamma_th, gamma_rate)
            for j in range(K):
                if j == i:
                    continue
                d = max(float(np.linalg.norm(
                    uav_positions[i] - uav_positions[j])), 1.0)
                path_gain = (comm.wavelength / (4.0 * np.pi * d)) ** 2
                g = comm.antenna_gain_linear * path_gain
                result[i] = max(result[i], gamma_req * n0_b / max(g, 1e-30))
        return result

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
                active = (
                    k in self._pending_comm_messages
                    and (
                        self._inter_uav_comm.payload_bits(
                            self._pending_comm_rates.get(k, 0),
                            active_dimensions=active_dimensions) > 0
                        or structure_bits > 0
                    )
                )
                fraction = (self._pending_comm_power_fractions.get(k, 0.0)
                            if active else 0.0)
                if analytical_power is not None and active:
                    # L0: use the analytic minimum (<= learned fraction).
                    p_comm = min(
                        self._isac_total_power_w * fraction,
                        analytical_power[k],
                    )
                else:
                    p_comm = self._isac_total_power_w * fraction
                p_sense = self._isac_total_power_w - p_comm
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
            self._qpd_received_last_seen[expired_qpd] = -10**9
        if self._hyperedge_enabled:
            expired_hyperedge = (
                self.t - self._hyperedge_received_last_seen
                > self._comm_message_ttl_frames
            )
            self._hyperedge_received_offer[expired_hyperedge] = 0.0
            self._hyperedge_received_last_seen[
                expired_hyperedge] = -10**9
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
                expired_structure] = -10**9
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
                        md['hyperedge_protocol'], md.get('token_mask'))
            else:
                future_mail.append(
                    (due_frame, receiver, sender, message, metadata))
        self._comm_mailbox = future_mail

        extra_dimensions = {}
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
                extra_dimensions[sender] = (
                    int(extra_dimensions.get(sender, 0))
                    + self._hyperedge_protocol_dim * active_targets
                )
                rate_index = int(self._pending_comm_rates.get(sender, 0))
                quantized = self._inter_uav_comm.quantize_values(
                    np.asarray(protocol).reshape(-1), rate_index,
                ).reshape(self.Q, self._hyperedge_protocol_dim)
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
        quantized_structure_protocol: Dict[int, np.ndarray] = {}
        selected_structure_bits: Dict[int, int] = {}
        structure_endpoint_dimensions = (
            self.Q * self._structure_student_endpoint_width)
        exact_extra_payload_bits: Dict[int, int] = {}
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
                    structure_endpoint_dimensions
                    * selected_bits
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

        deliveries, stats = self._inter_uav_comm.transmit(
            self._pending_comm_messages,
            self._pending_comm_rates,
            uav_positions,
            tx_powers_w=tx_powers_w,
            token_masks=self._pending_comm_token_masks,
            extra_payload_dimensions=extra_dimensions,
            extra_payload_bits=exact_extra_payload_bits,
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
            active_dims += int(extra_dimensions.get(sender, 0))
            active = (
                has_payload
                and self._inter_uav_comm.payload_bits(
                    rate_index, active_dimensions=active_dims) > 0)
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
        for item in deliveries:
            metadata = {
                'rate_index': int(item.rate_index),
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
                    self._merge_received_hyperedge_packet(
                        int(item.receiver),
                        int(item.sender),
                        metadata['hyperedge_protocol'],
                        metadata.get('token_mask'),
                    )
            else:
                self._comm_mailbox.append((
                    due_frame, item.receiver, item.sender, item.message.copy(),
                    metadata))

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

        for sender, energy_j in stats.per_sender_energy_j.items():
            if 0 <= sender < len(self.uavs):
                self.uavs[sender].battery = max(
                    0.0, self.uavs[sender].battery - float(energy_j))

        if self._joint_isac_power_enabled:
            # Sensing allocation is held for the full simulator frame; packet
            # energy uses its actual serialization airtime.
            for k in range(self.K):
                sensing_energy = float(
                    np.sum(self._current_sensing_power_w[k]) * self.dt)
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

        self._pending_comm_messages = {}
        self._pending_comm_rates = {}
        self._pending_comm_token_masks = {}
        self._pending_qpd_protocol = {}
        self._pending_hyperedge_protocol = {}
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
                self._qpd_received_last_seen[k] > -10**8)
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
                self._qpd_received_last_seen[k] > -10**8,
                axis=1,
            )) - np.any(
                self._qpd_received_last_seen[k, k] > -10**8))

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
        physical_entries: list,
    ) -> Tuple[Tuple[int, int, int], ...]:
        """Resolve reciprocal directed plans from receiver-local offer views."""
        physical_lookup = {
            (int(entry.i), int(entry.j), int(entry.q))
            for entry in physical_entries
            if int(entry.i) != int(entry.j) and float(entry.d_eff) > 0.0
        }
        held = tuple(
            edge for edge in self._hyperedge_selected_set
            if edge in physical_lookup)
        hold_active = bool(
            held
            and len(held) == len(self._hyperedge_selected_set)
            and self.t - self._hyperedge_last_update_frame
            < self._hyperedge_assignment_hold_frames
        )
        if hold_active:
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

        target_xy = np.asarray([
            target.get_position_3d()[:2] for target in self.targets
        ], dtype=np.float64)
        local_plans = []
        visible_peer_counts = []
        local_min_proxy = []
        for viewer in range(self.K):
            tx_capability = np.zeros((self.K, self.Q), dtype=np.float64)
            rx_capability = np.zeros((self.K, self.Q), dtype=np.float64)
            deficit = np.zeros((self.K, self.Q), dtype=np.float64)
            visible = np.zeros((self.K, self.Q), dtype=bool)
            public_position = np.zeros(
                (self.K, self.Q, 2), dtype=np.float64)

            tx_capability[viewer] = self._hyperedge_local_offer[
                viewer, :, 0]
            rx_capability[viewer] = self._hyperedge_local_offer[
                viewer, :, 1]
            deficit[viewer] = self._hyperedge_local_offer[viewer, :, 2]
            visible[viewer] = True
            if self._hyperedge_state_stream_enabled:
                public_position[viewer, :, 0] = (
                    self._hyperedge_local_offer[viewer, :, 3]
                    * float(self.area_size[0]))
                public_position[viewer, :, 1] = (
                    self._hyperedge_local_offer[viewer, :, 4]
                    * float(self.area_size[1]))

            received = (
                self._hyperedge_received_last_seen[viewer] > -10**8)
            visible |= received
            tx_capability[received] = self._hyperedge_received_offer[
                viewer, :, :, 0][received]
            rx_capability[received] = self._hyperedge_received_offer[
                viewer, :, :, 1][received]
            deficit[received] = self._hyperedge_received_offer[
                viewer, :, :, 2][received]
            if self._hyperedge_state_stream_enabled:
                public_position[:, :, 0][received] = (
                    self._hyperedge_received_offer[
                        viewer, :, :, 3][received]
                    * float(self.area_size[0]))
                public_position[:, :, 1][received] = (
                    self._hyperedge_received_offer[
                        viewer, :, :, 4][received]
                    * float(self.area_size[1]))
            visible_peer_counts.append(float(np.sum(
                np.any(received, axis=1))))

            # Each viewer uses a conservative maximum of the deficits it can
            # actually see; missing targets retain its own local deficit.
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

        active = tuple(
            edge for edge in stable if edge in physical_lookup)
        covered_targets = {target for _, _, target in active}
        coverage = float(
            len(covered_targets) / max(self.Q, 1))
        use_protocol = bool(
            active
            and coverage + 1e-12
            >= self._hyperedge_min_target_coverage
        )
        selected = active if use_protocol else tuple()
        self._hyperedge_selected_set = selected
        if use_protocol:
            self._hyperedge_last_update_frame = int(self.t)
        self._hyperedge_metrics = {
            'hyperedge_enabled': 1.0,
            'hyperedge_visible_peers_per_uav': float(np.mean(
                visible_peer_counts or [0.0])),
            'hyperedge_local_min_proxy': float(np.mean(
                local_min_proxy or [0.0])),
            'hyperedge_mutual_edges': float(len(mutual)),
            'hyperedge_stable_edges': float(len(stable)),
            'hyperedge_active_edges': float(len(active)),
            'hyperedge_target_coverage': coverage,
            'hyperedge_protocol_used': float(use_protocol),
            'hyperedge_safety_fallback': float(
                not use_protocol and self._hyperedge_safety_fallback),
            'hyperedge_assignment_reused': 0.0,
            'hyperedge_assignment_age_frames': 0.0,
            'hyperedge_assignment_hold_frames': float(
                self._hyperedge_assignment_hold_frames),
            'hyperedge_consensus_rounds': float(
                self._hyperedge_consensus_rounds),
            'hyperedge_pair_score_mode': self._hyperedge_pair_score_mode,
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
        ``C = T_sym*M*N*G_tx*G_rx*n_CPI/sigma_z^2``.  This is read directly
        from the entry observables (not from ``d_eff / P_sense``), so it stays
        identified even for edges the learned sensing head left unexcited.
        """
        dc = self.deflection_computer
        scale = float(
            dc.T_sym * dc.M * dc.N * dc.antenna_gain * dc.n_cpi
            / max(dc.noise_power, 1.0e-15)
        )
        coefficient = np.zeros((self.K, self.K, self.Q), dtype=np.float64)
        for entry in entries:
            i, j, q = int(entry.i), int(entry.j), int(entry.q)
            if not (0 <= i < self.K and 0 <= j < self.K and 0 <= q < self.Q):
                continue
            if float(entry.g_dd) >= float(dc.g_min):
                coefficient[i, j, q] = (
                    float(entry.chi_rep) * float(entry.alpha) ** 2 * scale
                )
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
        except ValueError:
            return None, budget
        return gain, budget

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
            return {}
        coefficient = self._per_watt_coefficient_from_entries(entries)
        try:
            gain, owner = fixed_owner_gain_matrix(coefficient, selected)
        except ValueError:
            return {}
        budget = np.clip(
            self._isac_total_power_w - self._current_comm_power_w, 0.0, None)
        step = float(self.uavs[0].v_max * self.uavs[0].dt)
        uav = np.array([u.pos[:2].copy() for u in self.uavs])
        tgt = np.array([t.get_position_3d()[:2] for t in self.targets])
        p_fa = float(self.cfg.detection.P_FA)

        from uav_isac.coordination.capability import local_capability_gradient_k
        from uav_isac.physical.detection import (
            minimum_deflection_for_detection_probability)
        from uav_isac.coordination.maxmin_power import (
            solve_fixed_structure_maxmin_power_lp, optimal_maxmin_dual_prices)

        d_min = float(minimum_deflection_for_detection_probability(
            np.asarray([0.60]), p_fa)[0])
        d_steady = float(minimum_deflection_for_detection_probability(
            np.asarray([0.80]), p_fa)[0])
        ceiling = np.sum(gain * budget[:, None], axis=0)

        grads: dict = {}
        if np.any(ceiling < d_min - 1e-9):
            # Phase 1: deficit gradient (steepest descent of the violation).
            deficit = np.maximum(0.0, d_min - ceiling)
            for k in range(self.K):
                gk = np.zeros(2, dtype=np.float64)
                for q in range(self.Q):
                    w = deficit[q] * budget[k]
                    if abs(w) < 1e-15:
                        continue
                    a = gain[k, q]
                    r = float(np.linalg.norm(uav[k] - tgt[q]))
                    gk += w * (2.0 * a) * (uav[k] - tgt[q]) / max(r ** 2, 1e-9)
                for q in range(self.Q):
                    if owner[q] != k:
                        continue
                    r = float(np.linalg.norm(uav[k] - tgt[q]))
                    for i in range(self.K):
                        w = deficit[q] * budget[i]
                        if abs(w) < 1e-15:
                            continue
                        a = gain[i, q]
                        gk += w * (2.0 * a) * (uav[k] - tgt[q]) / max(r ** 2, 1e-9)
                grads[k] = gk
        else:
            # Phase 2: max-min dual price gradient (cheap LP, no PWL).  The
            # max-min dual lambda* concentrates on the bottleneck target, which
            # is exactly the price that drags the steady (temporal-mean worst).
            res = solve_fixed_structure_maxmin_power_lp(gain, budget)
            if (not self._analytical_movement_candidates_enabled
                    and res.worst_deflection >= d_steady - 1e-9):
                # Already at/above the steady floor: hover (preserve the good
                # geometry) instead of falling back to the actor's motion,
                # which would oscillate and re-degrade a hard seed.
                return {k: np.zeros(2, dtype=np.float64) for k in range(self.K)}
            lam, _ = optimal_maxmin_dual_prices(gain, budget)
            prices = -lam  # gauge sign convention (negative marginals)
            support = {q: {i: (res.power_w[i, q], gain[i, q])
                           for i in range(self.K)} for q in range(self.Q)}
            for k in range(self.K):
                grads[k] = local_capability_gradient_k(
                    k, owner, prices, res.power_w[k], gain[k], uav[k], tgt,
                    support)

        # D1.1-B (advice 010): bounded multi-candidate trust-region.  Build a
        # few whole-fleet movement candidates, evaluate each at the moved
        # geometry with the exact max-min LP, and execute the best.  ``stay`` is
        # always a candidate, so the proxy score is monotone (never degrades).
        if self._analytical_movement_candidates_enabled:
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
    ) -> np.ndarray:
        """Rescale the (K,K,Q) per-watt tensor under 1/(R_tx^2 R_rx^2)."""
        r = np.linalg.norm(uav[:, None, :] - tgt[None, :, :], axis=2)      # (K,Q)
        rn = np.linalg.norm(new_uav[:, None, :] - tgt[None, :, :], axis=2)
        c = coeff * (r[:, None, :] ** 2) * (r[None, :, :] ** 2)
        return c / (rn[:, None, :] ** 2 * rn[None, :, :] ** 2)

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
            gain_cur, _ = fixed_owner_gain_matrix(coefficient, selected)
        except ValueError:
            gain_cur = np.zeros((K, Q))
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
                lam, _ = optimal_maxmin_dual_prices(gain_cur, budget)
            except ValueError:
                lam = None

        def radial(weak_q: int) -> np.ndarray:
            d = np.zeros((K, 2), dtype=np.float64)
            for k in range(K):
                v = tgt[weak_q] - uav[k]
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

        best = np.zeros((K, 2), dtype=np.float64)
        best_s = float('-inf')
        best_deflection: float | None = None
        for cand in candidates:
            nu = np.clip(uav + cand, 0.0, area)
            # Collision / proximity guard: no UAV may come closer than d_safe
            # to any target (the 1/R^4 gain would otherwise explode and the
            # greedy candidate search would keep ramming UAVs into targets).
            dist = np.linalg.norm(
                nu[:, None, :] - tgt[None, :, :], axis=2)  # (K,Q)
            if float(np.min(dist)) < d_safe - 1e-9:
                continue
            coeff_c = self._friis_rescale_tensor(coefficient, uav, tgt, nu)
            try:
                g_c, _ = fixed_owner_gain_matrix(coeff_c, selected)
            except ValueError:
                continue
            # D1.1-B+ dual pruning: U_lambda(g') <= best_deflection proves the
            # candidate's max-min deflection (hence worst P_D) cannot beat the
            # incumbent, so the exact LP evaluation is provably dominated.
            if lam is not None and best_deflection is not None:
                u_lambda = float(np.sum(
                    budget * np.max(lam[None, :] * g_c, axis=1)))
                if u_lambda <= best_deflection + 1e-9:
                    continue
            res = solve_fixed_structure_maxmin_power_lp(g_c, budget)
            # Score in P_D space (saturates at 1), so once a target is already
            # saturated, ramming UAVs closer yields no further score gain.
            pd = compute_detection_probabilities(res.deflection, p_fa)
            s = float(np.min(pd))
            if s > best_s + 1e-9:
                best_s, best = s, cand
                best_deflection = res.worst_deflection
        out: dict = {}
        for k in range(K):
            if np.any(best[k] != 0.0):
                out[k] = best[k]
        return out

    def step(self, actions: Dict[int, Action]) -> Tuple[Dict, Dict, Dict, StepInfo]:
        """Execute one simulation frame.

        Args:
            actions: Dict mapping uav_id → Action

        Returns:
            (next_observations, rewards_dict, dones_dict, step_info)
        """
        self.t += 1

        # Narrow Optional types (guaranteed set by reset())
        assert self.fc_position is not None
        assert self.belief_mgr is not None

        # Capture pre-move UAV positions (for potential-based shaping)
        prev_uav_positions = np.array([u.pos.copy() for u in self.uavs])

        # D0.95 L3: override the learned trajectory with the capability-guided
        # movement (receding horizon).  Structure (L2) is re-optimised by P0 at
        # the moved geometry later in this frame.
        analytical_delta: dict = {}
        if self._analytical_movement_enabled:
            analytical_delta = self._analytical_movement_delta()

        # 1. Apply UAV actions
        uav_positions = np.zeros((self.K, 3), dtype=np.float64)
        uav_velocities = np.zeros((self.K, 3), dtype=np.float64)
        roles = np.zeros(self.K, dtype=np.int32)

        for k in range(self.K):
            if k in actions:
                delta_p = analytical_delta[k] if k in analytical_delta \
                    else actions[k].delta_p
                self.uavs[k].apply_action(
                    delta_p, actions[k].role,
                    account_radio_energy=not self._joint_isac_power_enabled)
            uav_positions[k] = self.uavs[k].pos
            uav_velocities[k] = self.uavs[k].vel
            roles[k] = self.uavs[k].role

        # Transport the messages chosen from the previous observation. They are
        # receiver-specific and appear only after satisfying link/deadline
        # constraints. Radio energy is deducted from the sending UAV battery.
        comm_stats = self._process_learned_communications(uav_positions)

        # 2. Step target dynamics
        if self.tracking_enabled:
            for target in self.targets:
                target.step()

        target_positions = np.array([t.get_position_3d() for t in self.targets])
        target_velocities = np.array([
            np.array([t.state[2], t.state[3], 0.0]) for t in self.targets
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
        deflection_entries = self.deflection_computer.compute(
            uav_positions, uav_velocities,
            target_positions, target_velocities,
            roles, self.fc_position,
            role_agnostic=role_agnostic,
            sensing_power_w=(ranking_power_w
                             if self._joint_isac_power_enabled else None),
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
        if self._hyperedge_enabled:
            hyperedge_selected = self._resolve_hyperedge_negotiation(
                unfiltered_ranking_entries)

        # 4. Inner P0 solver with assignment hold (reduces reward non-stationarity)
        hold_frames = (
            self._p0_maxmin_pairing_hold_frames
            if self._p0_maxmin_pairing_enabled
            else getattr(self.cfg.marl, 'assignment_hold_frames', 1)
        )
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
            for edge in self._cached_p0_solution.selected_set:
                key = tuple(int(value) for value in edge)
                if key not in ranking_lookup:
                    cached_graph_valid = False
                    break
                cached_receiver_D[key[1], key[2]] += ranking_lookup[key]
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
                or cached_worst_pd < event_floor
            )
            maximum_hold_due = (
                self.t - self._last_solve_frame >= hold_frames)
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
                selected = tuple(
                    edge for edge in hyperedge_selected if edge in lookup)
                hyperedge_D_q = np.zeros(self.Q, dtype=np.float64)
                z_selected = np.zeros(
                    (self.K, self.K, self.Q), dtype=np.int32)
                for i, j, q in selected:
                    z_selected[i, j, q] = 1
                    hyperedge_D_q[q] += lookup[(i, j, q)]
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
        else:
            p0_solution = self._cached_p0_solution
            self._assignment_switched = False
            self._last_p0_solve_time_s = 0.0
        self._last_selected_set = p0_solution.selected_set  # for next obs

        # ── D0.89: analytical inner sensing power ──
        # The learned per-target sensing head is ignored for execution; after
        # P0 fixes role/owner/edge, the fixed-owner max-min power LP allocates
        # sensing power.  The actor still controls the sensing *budget* through
        # P_comm (b_i = 1 - P_comm).  Only the sensing power changes; motion,
        # structure and Token decisions are untouched.
        if self._analytical_sensing_power_enabled:
            coefficient = self._per_watt_coefficient_from_entries(
                deflection_entries)
            selected = tuple(
                tuple(int(v) for v in edge)
                for edge in p0_solution.selected_set
            )
            budget = np.sum(self._current_sensing_power_w, axis=1)
            reserve = None
            if self._analytical_sensing_power_reserve_pd > 0.0:
                from uav_isac.physical.detection import (
                    minimum_deflection_for_detection_probability,
                )
                reserve = minimum_deflection_for_detection_probability(
                    np.full(self.Q, float(
                        self._analytical_sensing_power_reserve_pd)),
                    self.cfg.detection.P_FA,
                )
            try:
                gain, _owners = fixed_owner_gain_matrix(coefficient, selected)
            except ValueError:
                gain = None
            if gain is not None:
                self._last_analytical_gain = gain.copy()
                self._last_analytical_budget = budget.copy()
                if self._task_constrained_power_enabled:
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
                    from uav_isac.physical.detection import (
                        minimum_deflection_for_detection_probability,
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
                                    )
                                    pi_star, _ = optimal_maxmin_dual_prices(
                                        gain, budget)
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
                        )
                        dual, _ = optimal_maxmin_dual_prices(gain, budget)
                        self._last_analytical_dual_prices = dual
                # Recompute the realized Deflection with the LP power.  This is
                # deterministic when the reporting link and Swerling are off.
                deflection_entries = self.deflection_computer.compute(
                    uav_positions, uav_velocities,
                    target_positions, target_velocities,
                    roles, self.fc_position,
                    role_agnostic=role_agnostic,
                    sensing_power_w=self._current_sensing_power_w,
                )
                self._last_deflection_entries = deflection_entries

        # Realized per-target deflection = TRUE d_eff of the SELECTED pairs.
        if self.p0_uses_belief or self._joint_isac_power_enabled:
            d_true = {(e.i, e.j, e.q): e.d_eff for e in deflection_entries}
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

        # ── Layer 4: Event-triggered active probing ──
        # Probe only when a target's accumulated risk score exceeds threshold.
        # This avoids wasting sensing resources when all targets are well-served.
        probe_triggered = False
        probe_target = -1
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
        receiver_D = receiver_deflection_from_selected(
            p0_solution.selected_set,
            deflection_entries,
            self.K,
            self.Q,
        )
        central_D = np.sum(receiver_D, axis=0)
        evidence_transport = None
        evidence_detection = None
        if self._detection_fusion_mode == 'u2u_distributed':
            assert self._inter_uav_comm is not None
            assert self._evidence_packet_layout is not None
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
                owner_aware=self._evidence_owner_aware,
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
            'detection_deflection_q': detection_D_q.copy(),
            'detection_central_oracle_deflection_q': central_D.copy(),
            'detection_receiver_deflection': receiver_D.copy(),
            'detection_fusion_owner': np.argmax(
                receiver_D, axis=0).astype(np.int64),
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

        # Kalman update for observed targets (noisy measurement of true state)
        true_states = np.array([
            [t.state[0], t.state[1], t.state[2], t.state[3]]
            for t in self.targets
        ])
        # B7: optionally gate the belief update by a detection event. With
        # belief_detection_sampling, target q's belief updates only if
        # delta_q ~ Bernoulli(P_D_q) fires (sampled once per target); otherwise
        # the pair is "missed" -> predict-only, AoI keeps growing. Default off =
        # optimistic (selected pair always observes).
        detected_q: Dict[int, bool] = {}
        if self.tracking_enabled:
            for (i, j, q) in p0_solution.selected_set:
                if q not in detected_q:
                    if self.belief_detection_sampling:
                        detected_q[q] = bool(self.rng.random() < float(P_D_q[q]))
                    else:
                        detected_q[q] = True
                obs = detected_q[q]
                ts = true_states[q]
                self.belief_mgr.update_after_observation(i, q, obs, ts)
                self.belief_mgr.update_after_observation(j, q, obs, ts)

        # ── Layer 4: Trust feedback — update trust based on post-measurement NIS ──
        if self.trust_gate_enabled and self._trust_manager is not None:
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
            except Exception:
                pass

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
                'learned_comm_channel_profile': (
                    self._active_comm_channel_profile),
                'learned_comm_channel_snr_threshold_db': (
                    self._active_comm_snr_threshold_db),
                'learned_comm_channel_deadline_s': (
                    self._active_comm_deadline_s),
                'learned_comm_per_sender_delivery_rate': (
                    comm_stats.sender_delivery_rates(self.K)),
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
        assert self.belief_mgr is not None
        beliefs = [self.belief_mgr.get_all_beliefs(k) for k in range(self.K)]

        # Diagnostic: feed true target state instead of beliefs
        # In tracking-free mode target locations are mission-known sensing
        # objects, not privileged tracking truth. Reuse the zero-covariance
        # observation layout to preserve network dimensions/checkpoints.
        oracle = (getattr(self.cfg.marl, 'oracle_obs', False)
                  or not self.tracking_enabled)
        oracle_targets = None
        if oracle:
            oracle_targets = np.array([
                [t.state[0], t.state[1], t.state[2], t.state[3]]
                for t in self.targets
            ], dtype=np.float64)

        obs = {}
        history_frames = getattr(self.cfg.marl, 'obs_history_frames', 1)
        if not hasattr(self, '_prev_obs_deque'):
            self._prev_obs_deque: dict = {}  # {agent_id: deque of prev frames}
        for k in range(self.K):
            # P1 FIX: per-UAV LOCAL detection confidence (RX-only).
            # No fallback to global prev_P_D — strict decentralized mode.
            # First frame (t=0) or UAV with no RX role gets zeros.
            local_pd = self.prev_P_D_local.get(k)
            if local_pd is None:
                local_pd = np.zeros(self.Q, dtype=np.float64)
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
                k, uav_states, beliefs, local_pd,
                oracle_targets=oracle_targets,
                selected_set=getattr(self, '_last_selected_set', []),
                deflection_entries=getattr(self, '_last_deflection_entries', None),
                comm_msgs=received,
                comm_metadata=received_meta,
                own_token_mask=self._last_sent_comm_token_masks.get(k),
                own_target_claims=(
                    self._last_sent_comm_target_claims.get(k)),
                channel_feedback=channel_feedback,
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
            'prev_P_D': None if self.prev_P_D is None else self.prev_P_D.copy(),
            'coord_pd_ema': (None if self._coord_pd_ema is None
                             else self._coord_pd_ema.copy()),
            'prev_P_D_local': {k: v.copy() for k, v in self.prev_P_D_local.items()},
            'comm_msgs': {k: v.copy() for k, v in self._comm_msgs.items()},
            'pending_comm_messages': {
                k: v.copy() for k, v in self._pending_comm_messages.items()},
            'pending_comm_rates': dict(self._pending_comm_rates),
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
            'pending_comm_power_fractions': dict(
                self._pending_comm_power_fractions),
            'pending_sensing_weights': {
                k: v.copy() for k, v in self._pending_sensing_weights.items()},
            'current_comm_power_w': self._current_comm_power_w.copy(),
            'current_sensing_power_w': self._current_sensing_power_w.copy(),
            'last_isac_metrics': copy.deepcopy(self._last_isac_metrics),
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
            np.full(self.K, -10**9, dtype=np.int64)),
            dtype=np.int64).copy()
        self._persistent_commitment_last_seen = np.asarray(state.get(
            'persistent_commitment_last_seen',
            np.full(self.K, -10**9, dtype=np.int64)),
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
            np.full((self.K, self.K, self.Q), -10**9)),
            dtype=np.int64).copy()
        self._qpd_metrics = copy.deepcopy(state.get('qpd_metrics', {}))
        self._qpd_last_submission_frame = int(state.get(
            'qpd_last_submission_frame', -1))
        self._pending_hyperedge_protocol = {
            int(k): np.asarray(v, dtype=np.float64).copy()
            for k, v in state.get(
                'pending_hyperedge_protocol', {}).items()}
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
            np.full((self.K, self.K, self.Q), -10**9)),
            dtype=np.int64).copy()
        self._hyperedge_consensus_streak = np.asarray(state.get(
            'hyperedge_consensus_streak',
            np.zeros((self.K, self.K, self.Q))),
            dtype=np.int64).copy()
        self._hyperedge_selected_set = tuple(
            tuple(int(value) for value in edge)
            for edge in state.get('hyperedge_selected_set', ()))
        self._hyperedge_last_update_frame = int(state.get(
            'hyperedge_last_update_frame', -10**9))
        self._hyperedge_metrics = copy.deepcopy(state.get(
            'hyperedge_metrics', {}))
        self._pending_comm_power_fractions = dict(
            state.get('pending_comm_power_fractions', {}))
        self._pending_sensing_weights = {
            k: v.copy() for k, v in state.get(
                'pending_sensing_weights', {}).items()}
        self._current_comm_power_w = np.asarray(state.get(
            'current_comm_power_w', np.zeros(self.K)), dtype=np.float64).copy()
        self._current_sensing_power_w = np.asarray(state.get(
            'current_sensing_power_w', np.full(
                (self.K, self.Q), self.cfg.uav.P_sense)),
            dtype=np.float64).copy()
        self._last_isac_metrics = copy.deepcopy(
            state.get('last_isac_metrics', {}))
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
            np.full(self.K, -10**9, dtype=np.int64)),
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
            np.array([t.state[2], t.state[3], 0.0]) for t in self.targets
        ])
        time_frac = self.t / max(self.T, 1)
        # Per-target mean belief uncertainty (trace of covariance)
        if self.belief_mgr is not None:
            belief_cov_trace = np.array([
                np.trace(self.belief_mgr.cov[0, q]) for q in range(self.Q)
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
