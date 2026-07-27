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
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from uav_isac.utils.types import Action, UAVState, TargetState
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
from uav_isac.physical.detection import compute_detection_probabilities
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


def filter_deflection_by_local_commitments(
    deflection_entries: list,
    sensing_power_w: np.ndarray,
    topk: int = 1,
    require_receiver: bool = True,
    mode: str = 'hard',
    soft_floor: float = 0.25,
    uncertainty_relief: float = 1.0,
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
    committed = np.zeros((K, Q), dtype=bool)
    order = np.argsort(-power, axis=1, kind='stable')
    rows = np.arange(K)[:, None]
    committed[rows, order[:, :k_eff]] = True

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
        self._pending_comm_power_fractions: Dict[int, float] = {}
        self._pending_sensing_weights: Dict[int, np.ndarray] = {}
        self._current_comm_power_w = np.zeros(self.K, dtype=np.float64)
        self._current_sensing_power_w = np.full(
            (self.K, self.Q), float(ua.P_sense), dtype=np.float64)
        self._last_isac_metrics: Dict[str, object] = {}
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
        self._coord_pd_ema = None
        self._prev_obs = {}  # clear history on reset
        self._gru_hidden = {}  # clear GRU state on reset
        self._comm_msgs = {}
        self._pending_comm_messages = {}
        self._pending_comm_rates = {}
        self._pending_comm_token_masks = {}
        self._last_sent_comm_token_masks = {}
        self._last_sent_comm_target_claims = {}
        self._pending_comm_power_fractions = {}
        self._pending_sensing_weights = {}
        self._current_comm_power_w = np.zeros(self.K, dtype=np.float64)
        self._current_sensing_power_w = np.full(
            (self.K, self.Q), float(self.cfg.uav.P_sense), dtype=np.float64)
        self._last_isac_metrics = {}
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
                w = np.asarray(weights.get(
                    k, np.ones(self.Q, dtype=np.float64)), dtype=np.float64)
                if w.shape != (self.Q,):
                    raise ValueError(
                        f'sensing weights for UAV {k} must have shape {(self.Q,)}')
                w = np.maximum(w, 0.0)
                total = float(np.sum(w))
                self._pending_sensing_weights[k] = (
                    w / total if total > 1e-12
                    else np.full(self.Q, 1.0 / max(self.Q, 1)))

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
            for k in range(self.K):
                active_dimensions = self._inter_uav_comm._active_dimensions(
                    self._pending_comm_token_masks.get(k))
                active = (
                    k in self._pending_comm_messages
                    and self._inter_uav_comm.payload_bits(
                        self._pending_comm_rates.get(k, 0),
                        active_dimensions=active_dimensions) > 0
                )
                fraction = (self._pending_comm_power_fractions.get(k, 0.0)
                            if active else 0.0)
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
            else:
                future_mail.append(
                    (due_frame, receiver, sender, message, metadata))
        self._comm_mailbox = future_mail

        deliveries, stats = self._inter_uav_comm.transmit(
            self._pending_comm_messages,
            self._pending_comm_rates,
            uav_positions,
            tx_powers_w=tx_powers_w,
            token_masks=self._pending_comm_token_masks,
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
            # A sub-frame transmission is available in the next observation;
            # longer delays remain in the mailbox until their due frame.
            due_frame = self.t + max(0, item.delay_frames - 1)
            if due_frame <= self.t:
                inbox[item.receiver][item.sender] = item.message.copy()
                inbox_meta[item.receiver][item.sender] = metadata
            else:
                self._comm_mailbox.append((
                    due_frame, item.receiver, item.sender, item.message.copy(),
                    metadata))

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

        self._pending_comm_messages = {}
        self._pending_comm_rates = {}
        self._pending_comm_token_masks = {}
        self._pending_comm_power_fractions = {}
        self._pending_sensing_weights = {}
        self._received_comm_msgs = inbox
        self._received_comm_meta = inbox_meta
        self._last_comm_stats = stats
        return stats

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

        # 1. Apply UAV actions
        uav_positions = np.zeros((self.K, 3), dtype=np.float64)
        uav_velocities = np.zeros((self.K, 3), dtype=np.float64)
        roles = np.zeros(self.K, dtype=np.int32)

        for k in range(self.K):
            if k in actions:
                self.uavs[k].apply_action(
                    actions[k].delta_p, actions[k].role,
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
        deflection_entries = self.deflection_computer.compute(
            uav_positions, uav_velocities,
            target_positions, target_velocities,
            roles, self.fc_position,
            role_agnostic=role_agnostic,
            sensing_power_w=(self._current_sensing_power_w
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

        # The policy's per-target sensing split is its local commitment. P0 may
        # rank only the resulting subgraph, while realized P_D is still read
        # from true-geometry entries after assignment.
        if self._distributed_target_commitment_enabled:
            ranking_entries, commitment_metrics = (
                filter_deflection_by_local_commitments(
                    ranking_entries,
                    self._current_sensing_power_w,
                    topk=self._distributed_target_commitment_topk,
                    require_receiver=(
                        self._distributed_target_commitment_require_receiver),
                    mode=self._distributed_target_commitment_mode,
                    soft_floor=(
                        self._distributed_target_commitment_soft_floor),
                    uncertainty_relief=(
                        self._distributed_target_commitment_uncertainty_relief),
                ))
            self._last_isac_metrics.update(commitment_metrics)

        # 4. Inner P0 solver with assignment hold (reduces reward non-stationarity)
        hold_frames = getattr(self.cfg.marl, 'assignment_hold_frames', 1)
        # Commitments change at communication rate. Reusing an old solution
        # would bypass the current local choices, so always resolve this mode.
        should_resolve = (self._distributed_target_commitment_enabled or
                          self.t == 1 or
                          self.t % hold_frames == 0 or
                          self._cached_p0_solution is None)
        if should_resolve:
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
        else:
            p0_solution = self._cached_p0_solution
            self._assignment_switched = False
        self._last_selected_set = p0_solution.selected_set  # for next obs

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
        team_reward = self.reward_computer.compute_team_reward(
            detection_D_q,            # evidence available under selected mode
            p0_solution.total_bits,
            0.0,  # constraint_penalty removed; Lagrangian handles constraints
            P_D_q=P_D_q,              # for direct P_D reward term (alpha_pd > 0)
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
            'pending_comm_power_fractions': dict(
                self._pending_comm_power_fractions),
            'pending_sensing_weights': {
                k: v.copy() for k, v in self._pending_sensing_weights.items()},
            'current_comm_power_w': self._current_comm_power_w.copy(),
            'current_sensing_power_w': self._current_sensing_power_w.copy(),
            'last_isac_metrics': copy.deepcopy(self._last_isac_metrics),
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
