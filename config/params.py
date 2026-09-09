"""Configuration parameter dataclasses and YAML loader."""

import dataclasses
import math
import os
import types
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, get_args, get_origin

import yaml


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(loader, node, deep=False):
    if not isinstance(node, yaml.MappingNode):
        raise yaml.constructor.ConstructorError(
            None,
            None,
            f"expected a mapping node, got {node.id}",
            node.start_mark,
        )
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ValueError(
                "duplicate YAML mapping key "
                f"{key!r} at line {key_node.start_mark.line + 1}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_unique_yaml(stream):
    """Parse safe YAML while refusing duplicate-key last-value wins."""
    return yaml.load(stream, Loader=_UniqueKeySafeLoader)


@dataclass
class ScenarioParams:
    region_size: Tuple[float, float] = (400.0, 400.0)  # Device-free MARL ISAC (TVT 2024): 400x400 m
    height: float = 20.0                                # TVT 2024: H=20 m
    K: int = 4
    Q: int = 2
    T: int = 150                                        # traversable: 2.5*150=375m > half-diag 283m
    dt: float = 0.1
    C: int = 1
    # Post-G2 time-energy closure (audit advice/001 section 3, 2026-08-26).
    # ``cpi_frame``: one control step executes n_cpi OTFS frames, so sensing
    # energy is charged on the OTFS clock ``T_sense = n_cpi*N*T_sym``
    # (≈1.024 ms at the canonical numerology) instead of the 100 ms `dt` slot;
    # deflection and battery then share the same clock.  ``dt_frame`` is the
    # legacy pre-G2 billing (P_sense*dt), retained to reproduce historical
    # certified results.  The canonical post-G2 manifest pins ``cpi_frame``.
    sensing_energy_mode: str = "dt_frame"


@dataclass
class UAVParams:
    v_max: float = 25.0
    d_safe: float = 20.0
    P_sense: float = 0.0251   # W nominal sensing waveform power
    # G2-0.7: hardware/waveform cap anchored at 14 dBm.  The 1 W joint RF cap
    # is an upper bound, not permission to silently amplify the sensing PA.
    P_sense_max: float = 0.0251
    P_report: float = 0.25    # W — TVT 2024 comm power set {0.25..1 W}
    # Per-UAV RF budget for joint U2U-ISAC allocation.  When enabled:
    # P_comm[k] + sum_q P_sense[k,q] <= P_isac_total in every frame.
    P_isac_total: float = 1.0
    B_max: float = 50000.0
    P_fly_static: float = 80.0
    P_fly_coeff: float = 0.05


@dataclass
class TargetParams:
    motion_model: str = "CV"   # "CV" | "CT" | "CA"
    ct_turn_rate: float = 0.3   # rad/s, only for CT model
    # bistatic pair -> task is learnable, P_D can approach the ~1.0 ceiling.
    # Ramp up to (0,20) only after MAPPO reaches high P_D on slow targets.
    speed_range: Tuple[float, float] = (0.0, 5.0)
    sigma_a: float = 0.5
    rcs: float = 1.0
    omega_q: List[float] = field(default_factory=lambda: [0.5, 0.5])


@dataclass
class OTFSParams:
    fc: float = 2.8e+10
    B: float = 1.0e+6        # 1 MHz — Device-free MARL ISAC (TVT 2024)
    delta_f: float = 1.5625e+4  # B/M = 1e6/64
    M: int = 64
    N: int = 16
    T_sym: float = 6.4e-5    # 1/delta_f
    g_tx_dBi: float = 16.0   # 64-elem UAV phased array (lit: 8 dBi single / "large arrays")
    g_rx_dBi: float = 16.0   # rx array gain
    # G2-0.6: one control action emits one explicit OTFS sensing frame. Values
    # above one require a separately certified multi-look execution model.
    n_cpi: int = 1


@dataclass
class ChannelParams:
    NF: float = 4.0          # kT*B*NF = 4e-21*1e6*2.51 = 1e-14 W = -110 dBm (TVT 2024 N0)
    kT: float = 4.0e-21
    ric_K: float = 6.0
    # Low-altitude (H=20m) report-link blockage: Al-Hourani LoS/NLoS (suburban). ON.
    use_los_prob: bool = True
    los_a: float = 4.88
    los_b: float = 0.43
    eta_los_dB: float = 0.1
    eta_nlos_dB: float = 21.0
    # Swerling-II RCS fading on the sensing return (default OFF).
    use_swerling: bool = False


@dataclass
class DetectionParams:
    P_FA: float = 0.001
    # G2-0.5: H0 Z~N(0,1), H1 Z~N(sqrt(D),1), so D=c_det*Es/En.
    c_det: float = 1.0
    g_min: float = 0.5
    # Post-G2 DD model (audit advice/001 section 5, 2026-08-26):
    #   binary      -- legacy ``1[g_dd >= g_min]`` support gate (historical).
    #   continuous  -- physical ``I_support * |A(tau,nu)|^2`` gain (canonical
    #                  post-G2 manifest; all new formal runs use this).
    dd_gain_mode: str = "binary"
    K_q_max: int = 3
    B_q: int = 64
    # Long-term fairness floor (constraint D4). 0.8 was unreachable even for the
    # best scripted policy (Greedy steady-state P_D ~0.2-0.4), which pinned the
    # Lagrangian multiplier and biased training. Set to a reachable-but-binding
    # value; re-tune from run_baselines.py 'steady_P_D' output.
    P_D_min: float = 0.2


@dataclass
class P0SolverParams:
    capacity_per_rx: int = 256
    latency_max: float = 0.005


@dataclass
class MARLParams:
    hidden_layers: List[int] = field(default_factory=lambda: [256, 256])
    lr: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95  # 0.98→0.95: per-agent reward adds variance, lower λ reduces advantage noise
    ppo_clip: float = 0.1     # 0.2→0.1: gentler updates near BC anchor
    ppo_epochs: int = 2       # 4→2: fewer gradient steps/rollout, less drift
    rollout_steps: int = 2048
    minibatch_size: int = 256
    entropy_init: float = 0.03  # 0.08→0.03: lower entropy preserves BC behavior
    entropy_final: float = 0.005  # 0.08→0.005: decay to near-deterministic
    entropy_decay_frames: int = 500_000  # faster decay
    eta_mc: float = 0.5
    eta_sense: float = 0.0  # per-agent sensing: 0=off (team-only baseline), 0.1=on
    # Detection-evidence boundary:
    # legacy_global preserves historical unconditional global fusion;
    # central_oracle reconstructs it explicitly; local_only forbids
    # cross-receiver fusion; u2u_distributed consumes delivered evidence.
    detection_fusion_mode: str = "local_only"
    # Tracking-model identity used by every local belief filter.  TARGET keeps
    # the historical behaviour (match target.motion_model); CV/CA explicitly
    # pin the estimator so maneuver-model mismatch experiments do not silently
    # change both plant and filter at once.  CT targets are tracked by CV when
    # CV is selected because the current belief manager has no CT filter.
    belief_motion_model: str = "TARGET"
    # Structured receiver-evidence packet. Calibration values are deliberately
    # unset by default; a u2u_distributed experiment must name an explicit
    # calibration profile rather than inherit hidden stress-set constants.
    # A selected sensing waveform is a wireless broadcast: every currently
    # reserved Rx endpoint may form a local statistic.  Only its later U2U
    # evidence report consumes additional RF resources.
    passive_multireceiver_evidence_enabled: bool = False
    # Physically local tracking and cross-frame posterior feedback.  Feedback
    # is piggybacked on delivered structured evidence broadcasts; no ACK or
    # free neighbour-belief channel is introduced.
    passive_multireceiver_belief_update_enabled: bool = False
    u2u_belief_feedback_enabled: bool = False
    u2u_belief_feedback_mean_bits: int = 12
    u2u_belief_feedback_cov_bits: int = 8
    u2u_belief_feedback_aoi_bits: int = 8
    u2u_belief_feedback_max_age_frames: int = 5
    # Posterior broadcast schedule.  evidence_topk (legacy) rides the belief on
    # the evidence top-k selection; freshness decouples the belief schedule and
    # ranks each source's own posteriors by the joint AoI / uncertainty /
    # task-loss score, broadcasting its top-k entries (audit priority #3).
    u2u_belief_feedback_schedule: str = "evidence_topk"
    u2u_belief_feedback_topk: int = 2
    # Global posterior payload budget (bits/frame) shared with the coordination
    # and evidence broadcasts; the freshness schedule must fit the frame
    # deadline (unified-MAC).  0 disables the cap.
    u2u_belief_feedback_bit_budget: int = 800
    # Per-source union cap (evidence entries + freshness top-ups).  Bounds the
    # per-packet payload so a decoupled schedule cannot silently blow the
    # per-class deadline by growing individual broadcasts.
    u2u_belief_feedback_max_union_per_source: int = 2
    u2u_belief_feedback_aoi_weight: float = 1.0 / 3.0
    u2u_belief_feedback_uncertainty_weight: float = 1.0 / 3.0
    u2u_belief_feedback_task_weight: float = 1.0 / 3.0
    u2u_belief_feedback_owner_aware: bool = True
    evidence_packet_topk: int = 1
    evidence_packet_ambiguity_top2_enabled: bool = False
    evidence_packet_ambiguity_ratio: float = 0.5
    evidence_packet_ambiguity_min_deflection: float = 0.0
    evidence_packet_owner_aware: bool = True
    evidence_packet_llr_bits: int = 8
    evidence_packet_confidence_bits: int = 2
    # Optional fixed-schema evidence header. Zero retains comm_header_bits.
    # In the synchronous sensing sub-slot a 16-bit CRC plus the explicitly
    # charged source index is sufficient; no dynamic rate/schema is inferred.
    evidence_fixed_schema_header_bits: int = 0
    # Explicit on-air observation-frame field. Set to zero only when the
    # fixed synchronous sub-slot makes the current frame uniquely inferable.
    evidence_timestamp_bits: int = 16
    # Compatibility envelope for protocol-only compression. The short packet
    # is physically serialized, but delivery admission, erasure coupling and
    # billed RF airtime use the legacy generic evidence packet. This preserves
    # the deployed policy's service distribution while reclaiming bits/latency.
    evidence_legacy_service_envelope_enabled: bool = False
    # AoI width of the reference packet retained by the legacy service
    # envelope. This may exceed the lossless physical categorical width.
    evidence_legacy_service_aoi_bits: int = 8
    evidence_packet_clip_max: float = 0.0
    evidence_packet_standardized_threshold: float = 0.0
    evidence_packet_confidence_log_boundaries: List[float] = field(
        default_factory=list)
    evidence_packet_confidence_representatives: List[float] = field(
        default_factory=list)
    evidence_packet_mc_draws: int = 2048
    evidence_packet_mc_seed: int = 20260725
    evidence_packet_content_mode: str = "normal"
    evidence_packet_calibration_profile: str = ""
    use_centered_marginal: bool = False  # centered marginal contribution shaping
    use_difference_reward: bool = True   # fixed-assignment no-op difference reward
    team_weight: float = 0.7             # team reward weight (E3 baseline)
    diff_weight: float = 0.3             # difference reward weight (E3 baseline)
    use_distance_shaping: bool = False  # potential-based "approach target" shaping
    shape_w: float = 0.01               # shaping weight (action signal ~ shape_w * 2.5/step)
    # Communication-cost weight in reward. MUST be small vs detection utility
    # (U_q=-log(1-P_D) ~ 0.3-2). At 1e-3, lambda*bits (~0.4) exceeded utility,
    # so "do nothing" paid better than detecting -> P_D collapsed. Comms is also
    # hard-capped by p0_solver.capacity_per_rx, so this is only a light regularizer.
    lambda_report: float = 1.0e-5
    alpha_pd: float = 0.0                    # direct P_D reward weight (0=utility-only, 0.5=hybrid)
    lambda_tail: float = 0.0                 # bottom-3 bonus weight
    # Detection-utility curvature for the team reward.
    #   "log"         -> -log(1-P_D) (historical; convex in Deflection)
    #   "concave"     -> 1-exp(-kappa*D) (concave/submodular, fixes B8)
    #   "maxmin_dual" -> LP-dual-price weighted reward aligned with the
    #                    fixed-structure max-min power coordinator.
    reward_utility_mode: str = "log"
    reward_concave_kappa: float = 1.0        # saturation scale for concave/dual modes
    # Analytical inner sensing-power solver (D0.89).  When enabled, the
    # learned per-target sensing head is ignored for execution and the fixed-
    # owner max-min power LP allocates sensing power after P0 selects the
    # structure.  The learned policy still controls the communication/sensing
    # budget split (P_comm -> b_i = 1 - P_comm).  reserve_pd > 0 adds a
    # reserve-first per-target floor derived from that detection probability.
    analytical_sensing_power_enabled: bool = False
    analytical_sensing_power_reserve_pd: float = 0.0
    # Fixed-structure teacher-free distributed primal--dual alternative to the
    # centralized analytical power LP.  Default off; requires a positive
    # analytical_sensing_power_reserve_pd to define the physical demand.
    distributed_primal_dual_power_enabled: bool = False
    distributed_primal_dual_power_rounds: int = 20
    distributed_primal_dual_power_acceptance_tolerance: float = 1.0e-10
    distributed_primal_dual_power_cost_per_watt: float = 1.0e-4
    distributed_primal_dual_power_regularization: float = 1.0e-3
    # Optional actor-centred proximal curvature.  Zero preserves the original
    # bounded solver; a positive value makes the executed KKT trajectory use
    # the same proposal centre as the differentiable unroll.
    distributed_primal_dual_power_actor_proximal_regularization: float = 0.0
    distributed_primal_dual_power_primal_tolerance: float = 2.0e-4
    distributed_primal_dual_power_stationarity_tolerance: float = 2.0e-4
    distributed_primal_dual_power_dual_tolerance: float = 2.0e-4
    distributed_primal_dual_power_convergence_patience: int = 3
    # Differentiable multi-frame learning-to-optimize layer.  This is a
    # training-only outer loop around the bounded execution solver and remains
    # opt-in so historical PPO runs/replays are unchanged.
    temporal_unrolled_power_enabled: bool = False
    temporal_unrolled_power_horizon: int = 4
    temporal_unrolled_power_inner_iterations: int = 4
    temporal_unrolled_power_warm_start_mix: float = 0.50
    temporal_unrolled_power_detach_between_frames: bool = False
    # Exact implementation switch for the Pareto Jacobian.  Batched VJP is
    # algebraically identical to the scalar reference and falls back
    # automatically for unsupported custom operators.
    temporal_unrolled_power_batched_vjp_enabled: bool = True
    # Optional active-set projection for the shared RF-budget equality.  The
    # default off path retains the existing objective-only Pareto update.
    temporal_unrolled_power_rf_tangent_enabled: bool = False
    temporal_unrolled_power_rf_tangent_slack_tolerance: float = 1.0e-6
    temporal_unrolled_power_rf_tangent_rank_tolerance: float = 1.0e-8
    temporal_unrolled_power_pareto_tolerance: float = 1.0e-6
    temporal_unrolled_power_pareto_max_iterations: int = 64
    # Optional teacher-free discrete-structure closure.  The environment
    # retains the actor's transmitter-side target power in the existing hard
    # P0 feasibility solve.  The training pass differentiates over the exact
    # bounded feasible-set convex
    # hull and enforces p_iq <= b_i sum_j xbar_ijq.
    temporal_feasible_structure_enabled: bool = False
    temporal_feasible_structure_temperature: float = 0.25
    temporal_feasible_structure_inertia: float = 0.25
    temporal_feasible_structure_max_structures: int = 4096
    temporal_feasible_structure_hard_forward: bool = True
    # Reuse unit-power true-geometry entries after the analytical LP and scale
    # only d_raw/d_eff by p_kq. Valid only for deterministic linear-in-power
    # sensing (no report-link draw or Swerling draw); otherwise execution
    # automatically falls back to a fresh physical computation.
    analytical_power_geometry_reuse_enabled: bool = False
    # Certificate-light distributed L1. Each UAV solves from its own delivered
    # public-state cache and executes only its own power row. No primal/dual
    # gap or ACK certificate is exchanged. Common views recover the
    # deterministic LP optimum; partial views preserve the local RF budget.
    distributed_replicated_power_enabled: bool = False
    distributed_replicated_power_inertia: float = 0.0
    # Optional causal two-stage envelope for the private LP.  With a valid
    # previous local full-plan cache, target q receives the feasible reserve
    # min{D_q(previous plan under the current private view), D(P_D=cap)} before
    # the usual max-min objective is optimized.  Zero disables the envelope.
    distributed_replicated_power_history_reserve_pd: float = 0.0
    # Simulator-only execution organization. Independent private LPs may run
    # in persistent isolated processes; this implies no radio concurrency or
    # additional inter-node information. Off preserves historical replay.
    distributed_replicated_power_process_parallel_enabled: bool = False
    distributed_replicated_power_process_workers: int = 4
    distributed_replicated_power_process_timeout_s: float = 0.1
    distributed_replicated_power_parallel_fallback_to_serial: bool = True
    # On worker failure, "serial" preserves the legacy exact recomputation;
    # "cached_or_uniform" performs no LP after the sub-deadline and executes a
    # projected incumbent/uniform safeguard. "cached_or_harmonic" replaces an
    # uncertified incumbent row by the closed-form conservative row max-min.
    distributed_replicated_power_parallel_failure_mode: str = "serial"
    distributed_replicated_power_deadline_incumbent_relative_tolerance: float = 0.05
    distributed_replicated_power_max_consecutive_process_failures: int = 3
    # Diagnostic only: solve one common conservative-gain LP to measure the
    # ceiling available to a composable allocator. Never used for execution.
    distributed_replicated_power_certificate_shadow_global_lp_enabled: bool = False
    # Per-node cross-frame LP reuse.  Each UAV evaluates its own cached primal
    # row allocation and cached simplex prices on its current private gain
    # view.  The LP is skipped only when the resulting local primal/dual
    # relative gap is below this tolerance. Zero preserves every-frame solves.
    distributed_replicated_power_reuse_relative_tolerance: float = 0.0
    # Target-responsibility certificate for the *assembled executed rows*.
    # Each transmitter broadcasts Q downward-quantized conservative
    # contributions a_lower[k,q]*p[k,q]. Target q is owned by q mod K and is
    # certified only after all K same-frame reports arrive within the AoI cap.
    # This proves a joint achieved-QoS lower bound; it does not claim global LP
    # optimality. Disabled by default for historical run reproducibility.
    distributed_composable_certificate_enabled: bool = False
    distributed_composable_certificate_targetwise_upper_enabled: bool = False
    distributed_composable_certificate_bits_per_target: int = 16
    distributed_composable_certificate_frame_bits: int = 32
    distributed_composable_certificate_scale: float = 1.0e-6
    distributed_composable_certificate_max_deflection: float = 1.0e6
    distributed_composable_certificate_max_age_frames: int = 5
    # Under an incomplete local public graph, reserve this fraction for a
    # uniform unknown-target floor and optimize the remainder over targets
    # reachable in that cache. One preserves the legacy all-uniform fallback.
    distributed_replicated_power_unknown_target_reserve: float = 1.0
    # Robust public-gain reconstruction: add this many local posterior
    # position standard deviations to each target-endpoint range.
    distributed_target_position_uncertainty_sigma: float = 0.0
    # Experimental nominal-to-robust risk homotopy for the replicated LP:
    # a_rho=(1-rho)*a_point+rho*a_lower.  rho=0 exactly preserves the nominal
    # controller; rho=1 uses the full covariance-set lower coefficient already
    # computed for the composable certificate.  The certificate itself always
    # retains the full lower bound, independent of this control preference.
    distributed_replicated_power_robust_gain_mix: float = 0.0
    # When the public graph is incomplete, use the row-separable minimax
    # transmitter-range prior p_kq proportional to R_kq^2 for the non-uniform
    # share. It needs only self position and the common target map.
    distributed_replicated_power_local_range_fallback_enabled: bool = False
    # D0.92: reference-normalized bargaining objective for the inner power
    # layer.  When enabled (requires analytical_sensing_power_enabled), the
    # fixed-owner power LP maximizes the common normalized headroom gain
    # eta = min_q (D_q - D^0_q)/(D^I_q - D^0_q) instead of pure max-min, so a
    # hard target with a small reachable ceiling gets a fair share of its own
    # opportunity rather than being starved.  steady/worst/weak3 remain
    # evaluation metrics only.
    bargaining_objective_enabled: bool = False
    # D0.93 L0: analytical minimum communication power.  When enabled (requires
    # joint_isac_power_enabled), each active sender's communication power is the
    # analytic minimum needed to meet the SNR/deadline delivery criterion for
    # every receiver, instead of the learned fraction.  This reclaims the link
    # margin into the sensing budget (b_i = 1 - P_comm^min) before capability is
    # evaluated.  Transport semantics (active set, token bits, receiver set,
    # deadline) are unchanged.
    analytical_comm_power_enabled: bool = False
    # One-sided design fade subtracted from the nominal link budget.  The L0
    # power inversion multiplies required receive SNR by 10^(margin/10).
    analytical_comm_snr_margin_db: float = 0.0
    # D1.1-C (2026-08-16): optimal orthogonal-bandwidth allocation for the L0
    # minimum comm power.  The original L0 splits the U2U bandwidth equally
    # among active senders (b_eff = B/n_active).  When a sender's rate demand
    # r_i = payload_i/(deadline - processing) approaches its share of the
    # bandwidth, the Shannon power N0*B_i*(2^(r_i/B_i)-1)/g_i explodes, so the
    # equal split over-charges high-load senders and wastes the 1 W budget
    # (numerically: up to ~90% of P_comm^min in the capacity-limited regime,
    # 99.8% under a heavy single sender).  The optimal allocation solves
    #     min sum_i N0*B_i*max(gamma_th, 2^(r_i/B_i)-1)/g_i   s.t. sum B_i = B
    # which is convex in B_i; the KKT condition f_i'(B_i) = -lambda is solved
    # by bisection on lambda (outer) and per sender (inner).  The result is a
    # feasible orthogonal allocation with the SAME SNR/deadline semantics
    # (each sender still meets its worst-receiver SNR and rate), so the
    # reclaimed sensing budget b_i = 1 - P_comm^min is never smaller than
    # under the equal split -- a strict no-waste improvement.  Default on.
    analytical_comm_optimal_bw: bool = True
    # D1.1-D (2026-08-16): live covertness-constrained power (T3 / advice 012
    # first step from oracle to deployment).  When enabled (requires
    # analytical_sensing_power_enabled), the L1 power allocation is the
    # max-min LP subject to the per-target counter-detection hard bound
    #   D_q^I = sum_i a^I[i,q] p_iq <= bar D^I,  bar D^I = [Q^-1(P_FA^I)-Q^-1(eps)]^2,
    # i.e. the opponent's detection probability P_{D,w}^I <= eps becomes a
    # HARD executed constraint (not a reward term), and the dual price mu_w
    # is the opponent-detection cost of each watt.  If the constraint makes
    # the LP infeasible the power falls back to the normal (unconstrained)
    # path and the violation is recorded in _last_intercept_pd_max.
    intercept_constrained_power_enabled: bool = False
    intercept_capability: str = "medium"   # weak | medium | strong
    intercept_eps: float = 0.1             # opponent detection prob. bound
    intercept_pfa: float = 1e-3            # opponent false-alarm prob.
    # D1.1-F (2026-08-16): standoff movement candidates under covertness.
    # The opponent's per-watt gain is a^I ~ 1/d^2, so moving AWAY from the
    # weakest target relaxes the counter-detection bound and lets more power
    # through (the standoff knob couples naturally through the constraint,
    # T3 SS2), at the cost of sensing gain 1/(R_tx^2 R_rx^2); the exact
    # max-min LP decides the trade-off per candidate.  Only meaningful when
    # intercept_constrained_power_enabled; default on, set False to A/B.
    analytical_movement_standoff_candidates: bool = True
    # D0.93-A1: task-constrained power allocation.  When enabled (requires
    # analytical_sensing_power_enabled), the inner solver solves the capability
    # gauge (min gamma s.t. worst+bottom-k+steady floors, budget <= gamma*b) via
    # the PWL LP; if gamma* <= 1 the returned power satisfies the FULL task, else
    # it falls back to reserve-first max-min (best effort).  This makes QoS a
    # hard constraint rather than a scalar weight.
    task_constrained_power_enabled: bool = False
    # Gauge floor targets for task_constrained_power_enabled, as
    # [worst, weak3(k-floor mean), steady, k].  A small worst margin (e.g. 0.61)
    # above the evaluation floor 0.60 makes the realized worst robustly clear
    # the strict `>= 0.60` check (the solver otherwise pins worst exactly at the
    # floor, failing the exact comparison by float noise ~1e-16).
    task_constrained_qos_floors: List[float] = field(
        default_factory=lambda: [0.60, 0.70, 0.80, 3])
    # D1.1-A (advice 010): inner L1 mode under task_constrained_power_enabled.
    #   "gauge"         -> capability gauge only (satisficing at the floors).
    #   "lexicographic" -> Stage A gauge proves the floors feasible; Stage B
    #                      maximises the worst within the feasible region
    #                      (QoS-constrained max-min); infeasible frames fall back
    #                      to reserve-first max-min (best effort).
    task_constrained_mode: str = "gauge"
    # D1.1-A (advice 010): solver-level tolerance in the QoS feasibility check.
    # The task-constrained gauge pins the worst target exactly at the 0.60 floor
    # (target of the LP), and P_D = Q(Q^-1(P_FA) - sqrt(D)) evaluated there can
    # land 1e-16 below the floor, failing a strict `>=` comparison by float
    # noise.  1e-6 is negligible on P_D in [0,1] but far above binary error.
    qos_eval_tol: float = 1e-6
    # D0.89-B: make the P0 structure ranking power-independent.  When enabled
    # (requires analytical_sensing_power_enabled), the deflection fed to P0 is
    # computed at unit sensing power so P0 ranks on the per-watt gain a_ijq
    # instead of the (now LP-determined) powered deflection, breaking the
    # structure<-power<-structure circular dependency.  The bottleneck target
    # priority uses the previous frame's max-min LP dual price lambda*.
    analytical_structure_ranking_enabled: bool = False
    # Scale-safe P0 structure/power co-design for local fusion.  The legacy P0
    # ranks a structure at fictitious unit power and imposes the actual
    # per-UAV RF budget only in L1; at K=Q=6/8 this can select one transmitter
    # for every target and strand 5/6 or 7/8 of the available fleet power.
    # This option jointly optimizes binary roles/owners/edges and continuous
    # edge power under the physical per-UAV budgets.  It is explicit and OFF
    # by default so frozen baseline configurations retain their semantics.
    p0_budget_coupled_structure_enabled: bool = False
    p0_budget_coupled_time_limit_s: float = 5.0
    # Stage 2 preserves the Stage-1 max-min optimum and improves capped
    # priority-weighted target quality.  Disable it for the lower-latency
    # primary-only variant; all physical constraints remain unchanged.
    p0_budget_coupled_secondary_enabled: bool = True
    # "milp" solves the joint mixed-integer model; "enumerated_role_ceiling"
    # enumerates Tx/Rx partitions and certifies each feasible support with the
    # exact fixed-structure power LP (fast primal lower-bound controller);
    # "satisficing_legacy_then_enumerated" retains a legacy structure only if
    # its current-budget LP already meets the worst-target task floor.
    p0_budget_coupled_solver_mode: str = "milp"
    # Optional robust task gate for the satisficing legacy portfolio.  Empty
    # means reuse task_constrained_qos_floors.  Non-empty values are a
    # development/calibration margin, not a universal physical constant.
    p0_legacy_guard_qos_floors: List[float] = field(default_factory=list)
    # Event-trigger decomposition: graph invalidity is always a hard trigger;
    # this switch controls the additional soft current-worst-P_D trigger.
    # Keeping it separate avoids conflating physical support loss with a
    # myopic performance trigger that can cause harmful L2/L3 churn.
    p0_maxmin_event_qos_enabled: bool = True
    # If a cached edge leaves the physical graph between P0 updates, preserve
    # all Tx/Rx roles and target owners and search only one-for-one replacement
    # edges.  Each candidate is certified by the current-budget exact LP;
    # impossible repairs fall back to the full event-triggered P0 solve.
    p0_topology_min_change_repair_enabled: bool = False
    # D0.95: analytical capability-guided movement (L3 geometry layer).  When
    # enabled (requires analytical_sensing_power_enabled and
    # analytical_structure_ranking_enabled), the learned trajectory increment is
    # overridden by a receding-horizon price-mediated geometry step: using the
    # previous frame's per-watt coefficient, owner map and sensing budget, the
    # capability gauge is solved and each UAV moves one step along the
    # deficit->capability gradient (steepest descent of the feasibility
    # violation, then of gamma*).  This is the distributed L3 hook; structure
    # (L2) is re-optimised by P0 at the moved geometry each frame.
    analytical_movement_enabled: bool = False
    distributed_id_movement_enabled: bool = False
    distributed_id_movement_standoff_m: float = 0.0
    distributed_greedy_matching_movement_enabled: bool = False
    distributed_greedy_matching_hold_frames: int = 150
    # Low-rate, ACK-free second-level movement beacon. At the configured
    # period each node multiplexes one absolute local target-belief anchor into
    # the already charged three-dimensional near-field header.
    distributed_movement_anchor_broadcast_enabled: bool = False
    distributed_movement_anchor_broadcast_period_frames: int = 5
    distributed_movement_anchor_max_age_frames: int = 10
    distributed_bottleneck_matching_movement_enabled: bool = False
    distributed_bistatic_bottleneck_movement_enabled: bool = False
    distributed_bistatic_complement_exponent: float = 1.0
    # Optional distributed block-coordinate movement.  Range infeasibility is
    # repaired in every frame; once an assigned UAV lies inside the engineering
    # annulus, a common frame clock alternates between radial hold and a
    # radius-preserving Tx/Rx angular-separation step.  This feature is
    # deliberately default-off because the inner radius is an engineering
    # constraint until an adversarial intercept model has calibrated it.
    distributed_alternating_optimization_enabled: bool = False
    distributed_ao_inner_radius_m: float = 0.0
    distributed_ao_outer_radius_m: float = 225.0
    distributed_ao_range_frames: int = 2
    distributed_ao_strategy_frames: int = 3
    distributed_ao_desired_bistatic_angle_deg: float = 90.0
    # Safe Gauss--Southwell block-coordinate movement. On geometry frames each
    # public view selects the single UAV-target radial step with the largest
    # predicted reduction of the worst nearest-Tx/nearest-Rx range product;
    # intervening frames hold geometry so L1/L2 can update. The selected action
    # is still executed only after the public d_safe projection.
    distributed_gain_scheduled_movement_enabled: bool = False
    distributed_gain_scheduled_period_frames: int = 2
    distributed_gain_scheduled_min_log_improvement: float = 0.0
    # 0 means one distinct coordinate per UAV at most (a full greedy sweep).
    distributed_gain_scheduled_max_selected_nodes: int = 0
    # Positive values give reachability lexicographic priority: until every
    # public-belief target has both a Tx and an Rx within this horizontal
    # radius, execute the full assignment-based range block every frame.
    distributed_gain_scheduled_far_range_gate_m: float = 0.0
    # local_matching keeps each viewer's cache-derived bottleneck assignment;
    # fixed_id is an ACK/certificate-free bijection for K=Q diagnostics.
    distributed_gain_scheduled_far_assignment_mode: str = "local_matching"
    # Optional episode-latched severe-tail gate.  A positive value activates
    # role-aware movement only when the public nearest-Tx/nearest-Rx squared
    # range-product has max/median at least this threshold; otherwise the
    # ordinary distance-bottleneck assignment is retained.
    distributed_bistatic_tail_gate_ratio: float = 0.0
    # Role-capacitated dual-role geometric responsibility (audit priority #2).
    # Each target owns exactly one Tx and one Rx responsibility; within each
    # public role the assignment minimizes the worst-case squared range subject
    # to a per-node capacity of ceil(Q/|K_r|).  No viewer matching row is
    # stitched into a fleet action.  Range phase moves toward the planar
    # midpoint of the responsibility set; strategy phase holds geometry while
    # P0 refreshes hyperedges/power.  Reassignment requires a relative
    # bottleneck improvement after a hold period (hysteresis).
    distributed_role_capacity_movement_enabled: bool = False
    distributed_role_capacity_standoff_m: float = 0.0
    distributed_role_capacity_reassign_threshold: float = 0.05
    distributed_role_capacity_hold_frames: int = 20
    distributed_role_capacity_range_frames: int = 12
    distributed_role_capacity_strategy_frames: int = 3
    # Gap-coverage movement (three-phase policy, Phase A/B): coverage
    # invariant C1 requires every target to have a Tx and an Rx endpoint
    # within ``critical_radius_m`` (the P_D >= 0.6 operational radius).
    # Uncovered targets receive deterministic Tx/Rx responsibilities every
    # frame (safety net against missed targets); each node pursues the
    # responsibility with the largest remaining gap at full speed.
    distributed_gap_coverage_movement_enabled: bool = False
    distributed_gap_critical_radius_m: float = 320.0
    distributed_gap_capacity: int = 2
    distributed_gap_standoff_m: float = 0.0
    # Saturation-protection pursuit radius.  When > 0, a node whose every
    # responsibility target is within this radius holds (prevents pulling
    # endpoints off saturated targets); 0 disables the hold (always pursue to
    # the standoff clamp).  The 25-seed ablation showed the hold regresses
    # more seeds than it protects (18/25 vs 21/25 without), so the default is
    # 0 (disabled); the parameter is kept for the ablation.
    distributed_gap_pursuit_radius_m: float = 0.0
    # Phase B: order the responsibility assignment by local target deficit
    # (1 - local P_D) so weaker targets receive their Tx/Rx endpoints first
    # and saturated targets release endpoints when capacity is exhausted.
    distributed_gap_deficit_priority: bool = True
    # Phase C: at the standoff boundary, apply a radius-preserving
    # tangential step (safety invariant S1 preserved exactly) to improve the
    # near-field bistatic geometry / DD-gate margin.  Default off until the
    # Phase A/B validation is complete.
    distributed_gap_tangential_phase: bool = False
    distributed_gap_tangential_step_rad: float = 0.02
    # Phase B near-field focus (weighted-product form): a target whose
    # equivalent radius R_eq = (min_tx^2 * min_rx^2)^0.25 exceeds
    # ``complete_radius`` while its near endpoint is within
    # ``nearfield_radius`` is critical; the responsible product is multiplied
    # by ``focus_weight`` in the pursuit order -- a soft bias, not a hard
    # lock (hard-lock ablation regressed saturated seeds; soft bias preserves
    # other responsibilities).  Fastest 1/R^4 product descent, dP_D/dR ~ -4/R^5.
    distributed_gap_nearfield_focus: bool = True
    distributed_gap_nearfield_radius_m: float = 300.0
    distributed_gap_complete_radius_m: float = 350.0
    distributed_gap_focus_weight: float = 4.0
    # Baseline-enveloped gap control. Both the gap-coverage action and the
    # legacy role-aware bottleneck action are reconstructed from the same
    # delivered public view and evaluated under the same safety operator.
    # Analytically certified no-op candidates skip redundant projection;
    # uncertified candidates are projected before comparison.
    # Gap control executes only when its lexicographic reachability potential
    # strictly improves on baseline; exact ties retain baseline.
    distributed_gap_baseline_envelope_enabled: bool = False
    # Optional third candidate inside the existing baseline envelope.  A
    # local robust lower/upper gain reconstruction, cached row-feasible power
    # witness and simplex target price form a primal--dual certificate without
    # adding any wire fields.  The candidate may execute only when the public
    # geometry score is no worse and its robust witness lower bound improves.
    distributed_gap_primal_dual_candidate_enabled: bool = False
    distributed_gap_primal_dual_margin: float = 0.0
    # Research gate: require the much stronger L_candidate > U_incumbent
    # condition instead of witness monotonicity.  Usually conservative.
    distributed_gap_primal_dual_strong_only: bool = False
    # Dynamic-target coordination must be computed from each viewer's own
    # tracker state.  When enabled, no target ground truth is used to create
    # hyperedge offers, reconstruct pair coefficients, or plan movement.
    distributed_coordination_use_local_belief_targets: bool = False
    # Post-G2 strict no-truth closure (audit advice/001 P0, 2026-08-26):
    # when enabled, distributed decision paths (coordination target resolver,
    # replicated-power range fallback) may NEVER read simulator ground truth;
    # they fail closed instead of silently exposing ``self.targets``.
    # Requires ``distributed_coordination_use_local_belief_targets=true``
    # (which itself requires ``tracking_enabled=true`` for the belief manager).
    # The canonical post-G2 manifest pins this true.
    distributed_no_truth_fail_closed: bool = False
    # Experimental scalar score-order communication primitive.  The current
    # learned aggregate/target-token payloads do not have a certified decoder
    # and action-margin bound, so EnvironmentCore rejects this online switch.
    # Keep OFF until an explicit score-token protocol closes that proof chain.
    distributed_decision_sufficient_comm_enabled: bool = False
    # Decision-preserving quantization dynamic range for the local margin
    # (uniform quant error eps_B = R / (2(2^B - 1))).
    distributed_decision_sufficient_dynamic_range: float = 1.0
    # Max bits for the decision-preserving token when the margin cannot certify
    # silence (falls back to the richest certified precision).
    distributed_decision_sufficient_max_bits: int = 32
    # Experimental event-trigger stage; unavailable for learned payloads until
    # the downstream decision certificate above exists.
    distributed_decision_sufficient_event_trigger_enabled: bool = False
    # Experimental adaptive-bit stage; exact-bit transport remains separately
    # tested, but learned-payload bit selection is fail-closed.
    distributed_decision_sufficient_adaptive_bits_enabled: bool = False
    distributed_movement_safety_projection_enabled: bool = False
    distributed_movement_safety_margin_m: float = 0.0
    # Additional two-endpoint motion uncertainty per public-cache age frame.
    distributed_movement_safety_margin_per_age_m: float = 0.0
    # ACK-free self-stabilizing movement. Every viewer stores only the full
    # assignment reconstructed from its own delivered public-state cache.
    distributed_movement_local_assignment_cache_enabled: bool = False
    distributed_movement_public_max_age_frames: int = 5
    distributed_movement_stale_fail_closed: bool = False
    # Execute exactly the public action checked by the robust projection,
    # including its non-intervention branch.
    distributed_movement_execute_projected_public_action: bool = False
    distributed_movement_outside_invariant_recovery_enabled: bool = False
    distributed_movement_outside_invariant_recovery_gain: float = 1.0
    # Endpoint-split CBF constraints remain valid when independently computed
    # per-UAV action rows are assembled after packet loss.
    distributed_movement_independently_composable_safety: bool = False
    # Proof-gated reduction for independently composable movement. If every
    # affine/bound constraint contains zero, Euclidean projection cannot
    # increase the reference norm, so per-node speed balls are redundant and
    # the nonlinear program reduces exactly to a linearly constrained QP.
    distributed_movement_analytic_composable_projection_enabled: bool = False
    # D1.1-B (advice 010): bounded multi-candidate trust-region L3.  Instead of
    # one normalized gradient step per frame, generate ~6 whole-fleet movement
    # candidates (stay, dual-price step, half step, capability-price step, and
    # radial steps toward the two weakest targets), evaluate each at the moved
    # geometry with the exact max-min power LP, and execute the best.  Adds a
    # small fixed number of LP evaluations per frame; a no-op when the
    # single-step gradient is already optimal.
    analytical_movement_candidates_enabled: bool = False
    # D1.1-B+ (2026-08-16, dual-pruned candidates): when the multi-candidate L3
    # is enabled, prune candidates by the weak-duality upper bound before
    # running the exact max-min LP.  For any simplex price lambda (e.g. the
    # current frame's optimal dual lambda*), the candidate geometry's max-min
    # deflection is bounded above by U_lambda = sum_i b_i max_q lambda_q a'_iq
    # (weak duality), and P_D is strictly monotone in deflection, so
    # U_lambda <= best_deflection proves the candidate cannot beat the current
    # best and its exact LP can be skipped.  The pruning is exact (never
    # changes the chosen candidate), it only removes provably-dominated LP
    # evaluations.  Default on; set False to reproduce the unpruned path.
    analytical_movement_dual_prune: bool = True
    # D1.1-B++ (2026-08-16): enrich the multi-candidate L3 candidate set with
    # a capability-gauge-price step and a dual x radial combination step.
    # Theory: by the envelope theorem d gamma*/d a_iq = -pi_q* p_iq*, the
    # gauge dual pi* is the shadow price of the FULL task (worst+bottom-k+
    # steady), whereas lambda* concentrates on the worst target only.  The
    # D0.95 design decision "L1 uses pi, L3 uses lambda*" is generalised here
    # to candidate competition: both price-driven directions (plus a convex
    # combination with the geometric radial step) are scored by the exact
    # max-min LP, so the better direction wins on the actual physics instead
    # of being pre-selected.  The candidate set is a superset of the D1.1-B
    # set and stay remains a candidate, so the chosen proxy score is monotone
    # (never degrades); dual pruning (above) bounds the LP budget.
    #
    # NEGATIVE RESULT (2026-08-16, recorded honestly): a 2-seed x 40-frame
    # end-to-end comparison showed mean worst gain ~0 (no measurable benefit).
    # Theory explains it: in lexicographic mode max-min equilibrates so
    # weak3 == steady == worst = t* and lambda* is supported on ALL targets
    # (D1_1A SS6.2), so pi carries no information lambda* lacks and the
    # D0.95 "L3 driven by lambda*" separation is optimal.  Kept as a
    # default-off option for reuse under non-equilibrating inner modes.
    analytical_movement_gauge_price_step: bool = False
    # D1.1-B+++ (2026-08-16): score the multi-candidate L3 candidates with
    # the SAME L1 solver that will be executed (lex Stage-B under a
    # lexicographic inner layer).  RATIONALE: the D1.1-B scorer used the plain
    # max-min LP even under lex execution, a score/execute objective
    # mismatch.  NEGATIVE RESULT (verified, default OFF): a 2-seed x 40-frame
    # comparison degraded badly (seed 503 final worst 0.996 -> 0.662), and the
    # theory explains it -- the mismatch is NOT a defect but the layered
    # design.  L3's role is to steer GEOMETRY, and the plain max-min score
    # keeps providing an improving gradient above t* even when the QoS floors
    # are met, whereas Stage-B pins t* at the floor once the floors bind, so
    # its score loses all sensitivity to movement and the geometry stalls.
    # This is the same "metric selects the price" separation as D0.95
    # (L1 uses the gauge, L3 uses lambda*).  Keep OFF; recorded for future
    # non-equilibrating inner modes only.
    analytical_movement_lex_scoring: bool = False
    # D1.9 (2026-08-16, bottleneck lookahead): evaluate the radial weak-target
    # candidates at the geometry AFTER H frames of sustained full-speed
    # approach (uav + H*step*dir) instead of after the single 1-step move,
    # while still executing only 1 step (receding horizon).  Theory: P_D =
    # Q(Q^{-1}(P_FA)-sqrt(D)) with D ~ 1/(R_tx^2 R_rx^2), so at R ~ 300-450 m a
    # single 2.5 m step changes the ceiling by ~0 (no LP discrimination, the
    # candidate pool often picks stay) while H frames of approach close the
    # gap to the 200 m regime where P_D saturates.  The blind100 left tail
    # (QoS 0.40 at 450-550 m worst_nearest vs 0.91 below 350 m) is exactly the
    # regime where the 1-step trust region stalls.  Physics: the lookahead
    # geometry rescales the per-watt tensor by the exact 1/R^4 law, respects
    # d_safe, and stay remains a candidate so the score is monotone.  H = 0
    # reproduces the D1.1-B single-step scoring.
    analytical_movement_lookahead_frames: int = 0
    # D1.10-B (2026-08-16, NEGATIVE RESULT -- default OFF): drive the L3
    # Phase-1 deficit trigger from the REAL fixed-structure max-min LP value t*
    # in addition to the optimistic per-UAV ceiling.  The motivation was the
    # coupling-scarce regime (blind100 seed 615: ceilings all >= d_min while
    # t* = 6.44 < d_min, P_D stuck at 0.29).  rev1 (uniform deficit) degraded
    # 20-seed QoS 15->14; rev2 (lambda*-weighted deficit) still degraded
    # specific seeds (615 0.71->0.29, 298 0.80->0.30, 886 0.50->0.39) despite
    # the same aggregate QoS, because any Phase-1 deficit gradient changes the
    # candidate-pool input and the Phase-2 dual-price gradient (lookahead40)
    # is the better direction on the actual physics.  The coupling-scarce fix
    # belongs in a per-UAV structural-reallocation candidate (D1.10-C), not a
    # global trigger.  Kept for A/B reuse; OFF is the certified configuration.
    analytical_movement_tstar_trigger: bool = False
    # D1.10-B (rev 3, 2026-08-16): when the t*-triggered Phase 1 fires (max-min
    # t* < d_min under power coupling), execute the lambda*-weighted deficit
    # gradient DIRECTLY, bypassing the multi-candidate scorer.  The scorer
    # judges moves by the max-min LP at the moved geometry, but under coupling
    # moving a UAV toward the bottleneck makes another target the new
    # bottleneck, so t* never rises and the pool picks stay -- the scorer
    # cannot see structural-reallocation value.  The deficit gradient is the
    # steepest descent of the violation itself.  The candidate pool stays the
    # gatekeeper in every other regime.
    #
    # NEGATIVE RESULT (2026-08-16, recorded honestly, default OFF): on the
    # blind100 residual seeds it fixes seed 615 (steady 0.29 -> 0.81, a true
    # coupling-scarce case) but DEGRADES seed 298 (0.33 -> 0.22) and 185
    # (0.75 -> 0.47) -- the deficit gradient pushes UAVs more spread (spread
    # +10-60%) in every case, which helps only when the bottleneck is
    # "isolated target lacking UAV supply" and hurts when the geometry is
    # already near-optimal.  No simple observable (t*/ceiling ratio, window,
    # spread) separates the two regimes, so forcing is not safe as a default.
    # Kept OFF; the seed-615 success motivates a per-UAV structural-reallocation
    # candidate (D1.10-C) designed for the coupling-scarce regime only.
    analytical_movement_phase1_force: bool = False
    # D1.10-C (advice 014 §4/§5, 2026-08-17): coupling-aware structure repair.
    # When the fixed-structure max-min LP value t_fix is below the worst QoS
    # Stage-wise, auditable coordination shaping. Stage 0 is diagnostic-only;
    # 1 adds worst progress; 2 adds avoidable duplicate penalty; 3 adds weak3
    # progress; 4 adds steady progress. Historical configs remain unchanged.
    coord_reward_enabled: bool = False
    coord_reward_stage: int = 0
    coord_reward_ema_alpha: float = 0.20
    coord_reward_steady_floor: float = 0.80
    coord_reward_weak3_floor: float = 0.70
    coord_reward_worst_floor: float = 0.60
    coord_reward_worst_weight: float = 1.0
    coord_reward_duplicate_weight: float = 0.01
    coord_reward_weak3_weight: float = 0.50
    coord_reward_steady_weight: float = 0.25
    coord_reward_min_move_m: float = 1e-6
    # CTDE-only action-head credit assignment. The deployed actor remains
    # decentralized; separate advantages update movement, latent message,
    # rate and joint-resource log-probabilities.
    headwise_credit_enabled: bool = False
    headwise_credit_vf_coef: float = 0.5
    headwise_movement_coef: float = 1.0
    headwise_message_coef: float = 1.0
    headwise_rate_coef: float = 1.0
    headwise_resource_coef: float = 1.0
    # Training-only paired virtual interventions and amortized token credit.
    # The predictor is discarded for decentralized execution.
    causal_ccp_enabled: bool = False
    causal_ccp_hidden_dim: int = 128
    causal_ccp_lr: float = 3e-4
    causal_ccp_epochs: int = 6
    causal_ccp_intervention_stride: int = 32
    causal_ccp_tail_temperature: float = 0.10
    causal_ccp_message_mix: float = 0.75
    causal_ccp_min_samples: int = 24
    causal_ccp_min_effect: float = 1e-5
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lagrangian_lr: float = 0.002     # lowered: binary any_violation made λ climb monotonically -> non-stationary returns -> critic blow-up
    max_violation_rate: float = 0.1  # Lagrangian target: max fraction of steps with constraint violations
    lagrangian_max: float = 1.0      # lowered cap so λ can't dominate the reward
    target_kl: float = 0.02          # 0.03 -> 0.02: tighter per-rollout policy drift bound
    num_episodes: int = 1000         # max episodes (hard cap; early-stop usually stops sooner)
    num_envs: int = 8  # parallel envs for GPU batching (tuned; stable MAPPO)
    assignment_hold_frames: int = 1  # 1=every frame; 5=hold P0 for 5 frames
    actor_decision_interval: int = 1  # 1=per-frame; 5=macro-action (hold action 5 frames)
    # Decouple slow physical motion from fast U2U negotiation.  The actor and
    # communication/resource heads still run every actor decision; only the
    # movement/role action is held for this many simulator frames.
    movement_decision_interval: int = 1
    obs_history_frames: int = 1  # 1=current only; 2=stack prev+current frame obs
    oracle_obs: bool = False  # diagnostic: feed true target pos (not beliefs) to actor
    rel_features: bool = True  # explicit per-target (dx,dy,dist,bearing) in actor obs
    structured_actor: bool = True  # entity-attention actor (vs flat MLP)
    # Tracking-free sensing mode. When False, targets are fixed sensing objects
    # whose locations are mission-known; the actor receives their current
    # coordinates directly and the Kalman predict/update loop is bypassed.
    # Post-G2 identity (audit advice/001 C0, 2026-08-26): the canonical system
    # is tracking-free; the dataclass default now matches default.yaml so the
    # no-YAML construction path cannot silently enable a legacy Kalman loop.
    tracking_enabled: bool = False
    # Physically billed receiver-owner posterior broadcast for strict local
    # tracking.  The selected bistatic receiver sends its causal 4D posterior;
    # peers fuse it by covariance intersection, which remains conservative
    # under unknown cross-correlation.  This is independent of detection
    # fusion and never shares simulator truth.
    distributed_owner_posterior_enabled: bool = False
    distributed_owner_posterior_mean_bits: int = 12
    distributed_owner_posterior_cov_bits: int = 8
    distributed_owner_posterior_aoi_bits: int = 8
    distributed_owner_posterior_max_age_frames: int = 5
    # Current paper scope uses only UAV-to-UAV links. When disabled, the old
    # receiver-to-ground-fusion reporting link, its energy, bit penalty and
    # capacity/latency constraints are removed from the sensing pipeline.
    # Post-G2 identity (audit advice/001 C0): U2U-only is the canonical scope;
    # the dataclass default now matches default.yaml so direct construction
    # cannot silently reintroduce the ground reporting link.
    ground_communication_enabled: bool = False
    critic_lr_mult: float = 5.0  # critic LR = lr * this (critic needs to track moving returns)
    bc_beta_init: float = 0.05  # BC anchor strength; sweet spot: prevents collapse, allows improvement
    use_p0_sinr_gated: bool = False  # gate P0 features by SINR threshold
    freeze_actor_after: int = 0       # freeze Actor at this episode (0=disabled)
    # Role assignment. learn_roles=False (default) DROPS the policy role head from
    # the objective and lets the inner P0 solver assign tx/rx: deflection enumerates
    # all i!=j ordered pairs (any UAV may tx or rx), P0 selects (i,j,q) with a
    # one-role-per-UAV constraint, and roles are derived from the selection. This
    # removes the degenerate all-same-role argmax collapse (diagnosed: deterministic
    # role argmax -> 95% frames single role -> zero tx-rx pairing -> P_D collapse).
    # learn_roles=True restores the original learned-role behavior (for comparison).
    learn_roles: bool = False
    # B5 / advice 014 (2026-08-17): movement-action parameterization.
    #   radial_clip (default, backward compatible): Gaussian -> tanh box -> scale ->
    #     radial projection onto the disk. The projection is a many-to-one map, so
    #     log pi(executed action) is NOT the true density of the executed action
    #     (~21% of the box lies outside the inscribed circle and triggers it) ->
    #     the PPO importance ratio is not mathematically exact.
    #   smooth_disk: bijection R^2 -> open disk  dp = d_max * z / sqrt(1+|z|^2),
    #     Jacobian |det d(dp)/dz| = d_max^2/(1+|z|^2)^2, so the exact log-prob
    #     correction is +2*log(1+|z|^2) - 2*log(d_max). PPO ratio becomes exact.
    #     New training runs should use this; existing checkpoints/results were
    #     produced under radial_clip and keep their numbers.
    dp_parameterization: str = "radial_clip"
    # Time-slotted multistatic endpoint scheduling.  When enabled together with
    # learn_roles=False, a UAV may transmit a sensing waveform in one sub-slot
    # and receive a peer echo in another sub-slot of the same simulator frame.
    # The per-frame communication+sensing power budget is unchanged; only the
    # mutually-exclusive endpoint-role constraint in P0 is removed.
    multistatic_subslot_enabled: bool = False
    # Hard connection between decentralized per-target sensing decisions and
    # the physical P0 assignment. P0 may rank only targets inside each local
    # top-k commitment; optionally both bistatic endpoints must agree.
    distributed_target_commitment_enabled: bool = False
    distributed_target_commitment_topk: int = 1
    distributed_target_commitment_require_receiver: bool = True
    # ``hard`` deletes every off-claim bistatic edge. ``soft`` retains the
    # physical graph and confidence-weights its ranking score; uncertain local
    # claims automatically defer toward geometry instead of starving targets.
    distributed_target_commitment_mode: str = "hard"
    distributed_target_commitment_soft_floor: float = 0.25
    distributed_target_commitment_uncertainty_relief: float = 1.0
    # Commitment graph source. ``sensing_power`` preserves the historical
    # instantaneous top-k rule. ``persistent_sensing`` applies the local
    # hold/handover state to that same sensing intent without changing Token
    # content. ``sent_token`` instead consumes each physically transmitted
    # sparse target-token mask and retains it locally across silence.
    distributed_target_commitment_source: str = "sensing_power"
    distributed_target_commitment_min_hold_frames: int = 0
    distributed_target_commitment_handover_frames: int = 0
    distributed_target_commitment_max_age_frames: int = 5
    # Queue-driven distributed primal-dual Token negotiation (QPD-ISAC).
    # The first gate is deliberately learning-free: geometric local capability
    # stands in for the learned marginal detection-value scorer.  The mechanism
    # is opt-in and leaves historical checkpoints bitwise unchanged when off.
    qpd_isac_enabled: bool = False
    qpd_qos_floor: float = 0.60
    qpd_queue_step: float = 0.25
    qpd_queue_max: float = 4.0
    qpd_primal_step: float = 1.0
    qpd_dual_step: float = 0.25
    qpd_rounds: int = 2
    qpd_row_capacity: float = 2.0
    qpd_target_capacity: float = 2.0
    qpd_price_max: float = 4.0
    qpd_primal_exploration_floor: float = 0.02
    qpd_peer_deficit_gain: float = 2.0
    qpd_send_threshold: float = 0.05
    qpd_bid_change_weight: float = 1.0
    qpd_power_cost: float = 0.02
    qpd_comm_cost: float = 0.01
    qpd_switch_cost: float = 0.02
    qpd_capability_distance_scale_m: float = 150.0
    qpd_commitment_threshold: float = 0.25
    qpd_override_token_mask: bool = True
    qpd_overwrite_protocol_header: bool = True
    qpd_override_sensing_weights: bool = True
    # Learning-free local-view negotiation over directed (Tx, Rx, target)
    # hyperedges. The protocol is appended to the ordinary learned Token, so
    # latent content is preserved while all additional coordinates are charged.
    hyperedge_negotiation_enabled: bool = False
    # Replaceable coefficient reconstruction service. ``numpy`` is the
    # audited exact implementation; custom backends register a factory before
    # EnvironmentCore construction.
    hyperedge_acceleration_backend: str = "numpy"
    # DenseDeflection -> DeflectionEntry compatibility boundary.
    deflection_materialization_backend: str = "numpy"
    # Development-only golden trace for acceleration equivalence audits.
    acceleration_golden_trace_enabled: bool = False
    strict_batch_observation_enabled: bool = True
    hyperedge_share_topk: int = 4
    hyperedge_distance_scale_m: float = 150.0
    hyperedge_capability_mode: str = "exponential"
    hyperedge_deficit_gain: float = 2.0
    hyperedge_proxy_floor: float = 0.25
    hyperedge_pair_score_mode: str = "endpoint_proxy"
    # replicated_global requires matching whole-graph snapshots;
    # reserved_endpoint uses stable ID roles/owners and two-endpoint consent.
    hyperedge_coordination_mode: str = "replicated_global"
    hyperedge_state_stream_enabled: bool = False
    hyperedge_protocol_only_enabled: bool = False
    hyperedge_state_bits_per_dim: int = 0
    # Optional fixed per-field precision for protocol-only beacons. An empty
    # list preserves uniform hyperedge_state_bits_per_dim. Both endpoints know
    # this manifest-level codec, so no adaptive-rate header is required.
    hyperedge_state_field_bits: List[int] = field(default_factory=list)
    # Preserve the uniform-codec service distribution when a shorter fixed
    # field codec is deployed. Actual bits/latency use the shorter packet;
    # admission, erasure coupling and RF billing use the old uniform length.
    hyperedge_state_codec_service_envelope_enabled: bool = False
    # Header length of the reference packet preserved by the codec service
    # envelope. Zero uses the actual hyperedge_protocol_header_bits.
    hyperedge_state_codec_service_reference_header_bits: int = 0
    # Optional compact fixed-schema header for protocol-only beacon packets.
    # Zero retains the generic communication header. A fixed orthogonal slot
    # identifies the sender/rate/schema; the compact header remains on-air and
    # is intended for CRC/error detection rather than hidden free metadata.
    hyperedge_protocol_header_bits: int = 0
    # Deterministic sender-cohort schedule for protocol-only state beacons.
    # Period H sends node k when frame mod H == k mod H; no schedule bits are
    # required. H must fit both the transport TTL and movement public-age cap.
    hyperedge_beacon_round_robin_period: int = 1
    # Optional common near-field position refinement. The beacon appends the
    # nearest static target ID and a bounded (dx,dy) residual; this is physical
    # state, not an optimizer certificate. Absolute state remains the fallback.
    hyperedge_nearfield_residual_enabled: bool = False
    hyperedge_nearfield_residual_range_m: float = 32.0
    hyperedge_nearfield_residual_companding_mu: float = 0.0
    # Optional store-and-forward dissemination of one cached peer physical
    # state per beacon. The relay record is source ID + the same state fields;
    # it contains no optimizer solution, dual price, gap, or ACK certificate.
    hyperedge_state_relay_enabled: bool = False
    hyperedge_tx_coalition_max: int = 2
    hyperedge_robust_quantization_enabled: bool = False
    # For the physical bistatic EKF candidate, let each reserved receiver
    # greedily maximize a monotone submodular sum of log(1+Deflection) and
    # prior-normalized log-det Fisher information over its incoming edges.
    hyperedge_bistatic_information_ranking_enabled: bool = False
    hyperedge_bistatic_information_weight: float = 1.0
    hyperedge_reserved_tx_nodes: Tuple[int, ...] = ()
    hyperedge_consensus_rounds: int = 2
    hyperedge_assignment_hold_frames: int = 1
    hyperedge_min_target_coverage: float = 1.0
    hyperedge_safety_fallback_enabled: bool = True
    # Diagnostic architecture gate: replace the average-utility greedy P0 with
    # an exact single-role max-min MILP. Optionally bypass the learned local
    # commitment filter to separate graph loss from solver-objective loss.
    p0_maxmin_pairing_enabled: bool = False
    p0_maxmin_bypass_commitment_filter: bool = False
    p0_maxmin_pairing_hold_frames: int = 1
    p0_maxmin_local_fusion_enabled: bool = False
    p0_maxmin_event_triggered_enabled: bool = False
    p0_maxmin_deficit_priority_gain: float = 3.0
    # P0 information source (B6). False (default) = ORACLE inner scheduler: P0 ranks
    # candidates on TRUE target geometry (upper bound). True = DEPLOYABLE: P0 ranks on
    # the fused belief estimate, while the realized deflection/P_D of the selected
    # pairs still uses TRUE geometry (the physical echo). See docs/KNOWN_ISSUES.md B6.
    p0_uses_belief: bool = False
    # Belief–P_D coupling (B7). False (default) = optimistic (selected pair always
    # updates belief). True = Bernoulli detection gating: target q's belief updates
    # only if a detection event delta_q ~ Bernoulli(P_D_q) fires; else predict-only
    # and AoI keeps growing. See docs/KNOWN_ISSUES.md B7.
    belief_detection_sampling: bool = False
    # Deterministic expected-information alternative to Bernoulli gating.
    # A detection probability p contributes expected measurement information
    # p*R^-1, implemented as R_eff=R/max(p,p_floor).  This is opt-in because it
    # changes the historical optimistic tracker even without random misses.
    belief_expected_detection_information_enabled: bool = False
    belief_expected_detection_information_floor: float = 1.0e-3
    # Tracker measurement physics. ``cartesian`` is the historical optimistic
    # [x,y,vx,vy] observation. ``bistatic_range_doppler`` uses only the
    # selected Tx/Rx geometry and OTFS delay/Doppler information in an EKF.
    belief_measurement_model: str = "cartesian"
    # Practical loss above the ideal known-signal CRLB (>=1), and a numerical
    # SNR/Deflection floor used only to keep a finite fail-weak covariance.
    belief_bistatic_crlb_efficiency: float = 4.0
    belief_bistatic_min_effective_deflection: float = 1.0e-3
    # Fixed evaluation scenarios (reused every eval + across decode modes).
    eval_seeds: List[int] = field(default_factory=lambda: [10001, 10002, 10003, 10004, 10005])
    # Optional versioned geometry-stratified bank. When set, ``eval_seed_split``
    # replaces eval_seeds and eval_episodes for checkpoint evaluation.
    eval_seed_bank_path: str = ""
    eval_seed_split: str = "selection"
    final_eval_seed_split: str = "test"
    checkpoint_confidence_alpha: float = 0.05
    checkpoint_bootstrap_samples: int = 2000
    checkpoint_cvar_fraction: float = 0.20
    checkpoint_confirmation_enabled: bool = False
    checkpoint_confirmation_split: str = "confirmation"
    # D1.10-A (2026-08-16): build a FRESH env instance per evaluation episode
    # instead of reusing one shared env across seeds.  The shared-instance
    # protocol leaves deflection_computer's Rician/LoS rng stream (constructed
    # in __init__, NOT replaced by wrapper.reset) drifting across episodes, so
    # the k-th seed's stochastic draws depend on how many episodes ran before
    # it (blind100 audit: seed 298 0.326 in-sequence vs 0.856 solo).  Default
    # False keeps every historical result numerically identical; enable for
    # fresh certification runs where each seed must be an independent draw.
    eval_independent_env: bool = False
    # Training-only prioritized replay over the geometry seed bank. Evaluation
    # splits are excluded automatically to prevent selection/test leakage.
    training_seed_replay_enabled: bool = False
    # Engineering/reproducibility probe only: non-negative values reset every
    # completed training episode to the same geometry seed.  Normal training
    # keeps -1 and samples fresh seeds (or uses the prioritized seed bank).
    training_fixed_seed: int = -1
    training_seed_bank_path: str = ""
    training_seed_uniform_mix: float = 0.20
    training_seed_priority_alpha: float = 0.70
    training_seed_priority_ema: float = 0.20
    training_seed_min_curriculum_fraction: float = 0.30
    training_seed_curriculum_frames: int = 300000
    training_seed_max_nearest_m: float = 350.0
    # Convergence-based early stopping (deterministic eval on a plateau)
    early_stop: bool = True
    eval_interval: int = 50          # run a deterministic eval every N episodes
    eval_episodes: int = 3           # deterministic eval episodes to average
    early_stop_patience: int = 12    # stop after this many evals with no improvement
    early_stop_min_delta: float = 0.005  # min steady_P_D gain counted as improvement
    # CTDE centralized critic (MAPPO, critic sees global state) vs decentralized
    # critic (IPPO, critic sees only local obs). de Witt et al. 2020 / Yu et al. 2022.
    centralized_critic: bool = True
    # Communication mode:
    #   off        -> no learned messages (historical Full/EH isolation)
    #   on         -> legacy free continuous messages
    #   cost_aware -> stochastic learned message + learned silence/rate action,
    #                 quantized and transported through the inter-UAV channel.
    learned_comm_mode: str = 'on'
    # Prevent free access to neighbour position/velocity/role/intent in the
    # actor observation. The feature slots remain (zero-filled) so learned U2U
    # messages are the only inter-node information channel without changing
    # observation dimensions or invalidating existing network parsers.
    comm_only_neighbor_information: bool = False
    # Preserve every delivered sender message as an independent token and let
    # each target entity query the inbox with masked cross-attention.
    comm_cross_attention_enabled: bool = False
    comm_message_ttl_frames: int = 5
    # Cost-aware emergent inter-UAV communication. A rate action chooses the
    # number of quantization bits per one of the 16 learned message components;
    # zero bits is the learned silence action. The payload is broadcast once
    # and may be received by multiple UAV neighbours.
    # Dataclass fallback retains the historical four-level action for old YAML
    # files; default.yaml explicitly enables the optional 32-bit fifth level.
    comm_rate_bits_per_dim: List[int] = field(
        default_factory=lambda: [0, 4, 8, 16])
    comm_header_bits: int = 64
    comm_bandwidth_hz: float = 1.0e5
    comm_tx_power_w: float = 0.25
    comm_deadline_s: float = 0.005
    comm_processing_delay_s: float = 2.0e-4
    comm_snr_threshold_db: float = 0.0
    comm_antenna_gain_dbi: float = 0.0
    # Optional Polyanskiy-style second-order normal approximation for a
    # complex AWGN block.  Disabled by default to preserve historical Shannon
    # traces.  When enabled, packet serialization must meet both deadline and
    # target BLER; this remains an analytical approximation, not a code-level
    # reliability guarantee.
    comm_finite_blocklength_enabled: bool = False
    comm_finite_blocklength_target_bler: float = 1.0e-5
    # Maximum accepted probability of an erroneous codeword passing the CRC.
    # A fixed r-bit CRC must satisfy target_bler*2^-r <= this budget.
    comm_undetected_error_probability_max: float = 1.0e-7
    comm_finite_blocklength_max_channel_uses: int = 100_000_000
    comm_finite_blocklength_sample_errors: bool = True
    # Code blocklength is selected at nominal SNR minus this design fade.
    comm_finite_blocklength_coding_snr_margin_db: float = 0.0
    # Actual (not threshold-only) correlated U2U SNR shadowing.
    comm_snr_shadowing_std_db: float = 0.0
    comm_snr_shadowing_correlation: float = 0.0
    # Gilbert-Elliott directed-link erasures.  The Markov channel evolves once
    # per simulator frame and applies to the complete physical broadcast.
    comm_burst_loss_enabled: bool = False
    comm_burst_good_to_bad_probability: float = 0.0
    comm_burst_bad_to_good_probability: float = 1.0
    comm_burst_good_drop_probability: float = 0.0
    comm_burst_bad_drop_probability: float = 1.0
    # Optional episode-level channel-domain randomization.  The two lists
    # define paired (SNR threshold, deadline) profiles and must have equal
    # length.  Sampling happens only at reset and is reproducible from the
    # environment RNG.  Keeping this disabled preserves every legacy run.
    comm_channel_randomization_enabled: bool = False
    comm_channel_randomization_snr_threshold_db_values: List[float] = field(
        default_factory=list)
    comm_channel_randomization_deadline_s_values: List[float] = field(
        default_factory=list)
    comm_channel_randomization_profile_weights: List[float] = field(
        default_factory=list)
    comm_message_log_std_init: float = -1.0
    comm_entropy_scale: float = 1.0
    # Payload representation. "aggregate" keeps one 16-D latent vector per
    # UAV. "target_tokens" sends Q independent network tokens; their values
    # are unconstrained and consumed directly by receiver cross-attention.
    comm_payload_mode: str = 'aggregate'
    comm_target_token_dim: int = 16
    # Continuous communication power plus a learned multi-target sensing split.
    # Silence assigns zero communication power and the full budget to sensing.
    joint_isac_power_enabled: bool = False
    comm_power_fraction_min: float = 0.0
    comm_power_fraction_max: float = 1.0
    isac_power_log_std_init: float = -1.0
    sensing_allocation_log_std_init: float = -1.0
    # Explicit environment costs. P0 reporting bits retain lambda_report;
    # these weights apply only to learned UAV-to-UAV messages.
    comm_bit_cost_weight: float = 1.0e-5
    comm_energy_cost_weight: float = 1.0
    comm_delay_cost_weight: float = 10.0
    # Minimise U2U resource use subject to sensing-quality constraints. The
    # trainer adapts one dual multiplier per metric: multipliers rise while a
    # rollout is below its floor and decay once the floor is satisfied.
    comm_qos_constrained: bool = False
    comm_qos_steady_min: float = 0.80
    comm_qos_weak3_min: float = 0.70
    comm_qos_worst_min: float = 0.60
    # R15 (roadmap 2026-08-29): single source for the FORMAL acceptance gates.
    # tools/assert_gate_thresholds.py (MEDIUM_FLOORS) and
    # tools/report_blind_certification.py read these values when --config is
    # given.  Training reward floors (coord_reward_*) and communication
    # constraint floors (comm_qos_*) are numerically identical to this triple
    # by default, and that equality is LOCKED by tests/test_qos_gate_provenance.py
    # (a test, not a shared reference: the three sources stay semantically
    # independent -- acceptance / reward / constraint, C1).
    # Order: [worst, weak3, steady].
    qos_acceptance_floors: List[float] = field(
        default_factory=lambda: [0.60, 0.70, 0.80])
    qos_acceptance_wilson_lcb_floor: float = 0.70
    comm_qos_dual_lr: float = 0.05
    comm_qos_lambda_init: float = 0.5
    comm_qos_lambda_max: float = 5.0
    comm_qos_reward_scale: float = 1.0
    # Soft, directly attributable exploration reward for the discrete rate
    # action. Successful active senders receive a QoS-deficit-gated bonus;
    # explicit bit/energy/delay costs still discourage waste. The floor keeps
    # a small exploration signal after feasibility without prescribing a rate.
    comm_encouragement_enabled: bool = True
    comm_encouragement_weight: float = 0.05
    comm_encouragement_floor_ratio: float = 0.10
    # QoS-gated soft precision exploration. Credit rises from 4 to the target
    # bits/dimension and then saturates, so higher precision is not mandatory.
    comm_rate_bonus_enabled: bool = True
    comm_rate_bonus_weight: float = 0.03
    comm_rate_bonus_target_bits: int = 8
    comm_rate_bonus_aux_coef: float = 1.0
    comm_rate_bonus_aux_lr: float = 0.01
    # Communication liveness regularizer. Silence is allowed for a short grace
    # window; every additional silent policy decision incurs a growing,
    # capped sender-specific penalty and any active rate resets the streak.
    comm_silence_penalty_enabled: bool = True
    comm_silence_grace_decisions: int = 3
    comm_silence_penalty_per_decision: float = 0.01
    comm_silence_penalty_max: float = 0.05
    # Evaluation-only causal ablation: preserve the cost-aware observation and
    # actor architecture, but replace every sampled rate with silence.
    comm_eval_force_silence: bool = False
    # Evaluation-only rate diagnostic. A non-negative value must match one
    # entry in comm_rate_bits_per_dim and overrides only the deterministic
    # rate action.
    comm_eval_force_rate_bits: int = -1
    # Evaluation-only content intervention with identical rate/cost/metadata:
    # "none" | "zero" | "permute" (cyclic sender-content permutation).
    comm_eval_message_ablation: str = "none"
    # Evaluation-only movement intervention: centralized minimum-distance
    # one-to-one UAV/target assignment; actor roles and communication unchanged.
    eval_centralized_assignment_movement: bool = False
    # Evaluation-only diagnostic for the QoS-aware slow-layer teacher.  The
    # centralized teacher directly controls movement while the actor still
    # controls communication, sensing resources and roles.  This isolates
    # teacher quality from decentralized distillation error.
    eval_qos_bistatic_assignment_movement: bool = False
    # Optional hard rate barrier retained only for ablation/reproduction. It is
    # disabled by default because forcing 8 bit increased traffic without
    # improving the fixed-seed sensing probe.
    comm_qos_min_rate_bits: int = 0
    comm_qos_rate_shortfall_penalty: float = 0.0
    comm_qos_rate_aux_coef: float = 0.0
    # Communication-aware decentralized target allocation. Each UAV produces
    # a target-responsibility distribution from local target entities and its
    # delivered-message context. Training regularizes team coverage without
    # fixing the semantic content of the messages.
    target_allocation_enabled: bool = False
    target_allocation_balance_coef: float = 0.20
    target_allocation_commit_coef: float = 0.01
    # End-to-end decentralized commitment options.  The payload remains a
    # learned latent vector; these controls only regularize the resulting team
    # allocation and connect it to the movement head.
    target_allocation_temperature: float = 1.0
    target_allocation_straight_through: bool = False
    target_allocation_movement_blend: float = 0.0
    # Confidence-gated physical authority for the discrete movement
    # commitment.  The top-1/top-2 probability margin suppresses uncertain
    # hard choices, while confident commitments retain the configured blend.
    target_allocation_movement_confidence_gating_enabled: bool = False
    target_allocation_movement_confidence_floor: float = 0.0
    target_allocation_movement_confidence_power: float = 2.0
    # Optional rollout-boundary schedule for the physical authority of the
    # slow target commitment.  A positive anneal length overrides the fixed
    # blend above and preserves PPO likelihood consistency within a rollout.
    target_allocation_movement_blend_start: float = 0.0
    target_allocation_movement_blend_end: float = 0.0
    target_allocation_movement_blend_anneal_frames: int = 0
    # Keep a one-target kinematic commitment separate from the capacitated
    # two-endpoint sensing assignment.  This removes the contradictory row-sum
    # constraints that otherwise make the commitment teacher diverge.
    hierarchical_dual_assignment_enabled: bool = False
    # Distributed one-to-one optimal-transport projection for the slow
    # movement responsibility. Every UAV reconstructs the same team bid graph
    # from received target tokens; sensing endpoints keep their independent
    # capacity-two projection.
    movement_team_matching_enabled: bool = False
    movement_team_matching_temperature: float = 0.35
    movement_team_matching_iterations: int = 16
    movement_team_matching_blend: float = 0.0
    movement_team_matching_intrinsic_bid_mix: float = 0.0
    # Map locally negotiated responsibilities into physical sensing logits.
    # Zero keeps the historical auxiliary-only path.
    target_allocation_resource_blend: float = 0.0
    target_allocation_differentiable_comm: bool = False
    target_allocation_comm_delay_decisions: int = 1
    target_allocation_temporal_coef: float = 0.0
    target_allocation_aux_epochs: int = 1
    target_allocation_aux_lr: float = 0.0
    # Two-stage local negotiation.  Each UAV knows only the shared frame phase
    # (proposal/response), its own entities and delivered peer tokens.  Peer
    # claims are decoded from latent tokens and used as a local exclusion term.
    round_negotiation_enabled: bool = False
    round_negotiation_strength: float = 0.5
    round_negotiation_temperature: float = 0.5
    # Sparse neighborhood claim exchange. Only the locally preferred target
    # tokens are transported; delivered token masks act as peer claims. The
    # desired load is two endpoints per target for bistatic sensing.
    sparse_claim_enabled: bool = False
    sparse_claim_share_topk: int = 2
    sparse_claim_commit_topk: int = 2
    sparse_claim_desired_endpoints: int = 2
    sparse_claim_full_penalty: float = 2.0
    sparse_claim_vacant_bonus: float = 0.5
    sparse_claim_temperature: float = 0.25
    sparse_claim_underload_coef: float = 4.0
    sparse_claim_overload_coef: float = 1.0
    # Reuse the PPO categorical communication-rate action as a joint
    # cardinality action. Duplicate bit-rate levels may map to different k,
    # e.g. [0,8,8] with mapping [0,1,2].
    adaptive_topk_from_rate_enabled: bool = False
    adaptive_topk_rate_mapping: List[int] = field(
        default_factory=lambda: [0, 1, 2])
    adaptive_topk_rate_only_training: bool = False
    adaptive_topk_reset_rate_head: bool = False
    adaptive_topk_worst_credit_coef: float = 0.0
    comm_rate_metadata_denominator: Optional[float] = None
    # Strictly local reciprocal-link summary exposed only to the rate head.
    # It contains delivered-packet quality/AoI plus the receiver's configured
    # deadline and SNR threshold; no fusion-centre or free ACK is introduced.
    comm_channel_feedback_rate_enabled: bool = False
    comm_channel_feedback_dim: int = 6
    # Training-only, sender-specific transport constraint. Failed outgoing
    # links penalize only the sampled sender rate head; execution still uses
    # local reciprocal-link observations and has no free acknowledgement.
    comm_sender_delivery_penalty_enabled: bool = False
    comm_sender_delivery_penalty_weight: float = 0.05
    # Explicit message-to-sensing path. Received target tokens alter executed
    # sensing logits, while an auxiliary counterfactual teaches useful changes.
    comm_aided_sensing_enabled: bool = False
    comm_aided_sensing_blend: float = 1.0
    comm_aided_sensing_aux_coef: float = 1.0
    comm_aided_sensing_counterfactual_coef: float = 0.5
    comm_aided_sensing_temperature: float = 0.25
    comm_aided_sensing_margin: float = 0.02
    # Diagnostic/deployable decoder for the latent portion of each target
    # token. Dimension 0 remains the explicit movement bid; dimensions 1..D-1
    # are decoded into sender-evidence validity and conditional local P_D.
    # The decoder is observational only until a separately gated CA-CSR path
    # is enabled, preserving legacy policy behaviour exactly.
    comm_semantic_decoder_enabled: bool = False
    # Decoded peer endpoint quality refines sparse neighbor bids immediately
    # before capacity-two projection. The row-centred correction is opt-in and
    # has no direct path to movement or the total power split.
    comm_semantic_capacity_bid_enabled: bool = False
    comm_semantic_capacity_bid_gain: float = 0.0
    # Keep the stable top-k contract and allow at most one extra target token
    # when decoded peer quality indicates endpoint underload and local geometry
    # provides useful sensing capability.
    comm_semantic_extra_token_enabled: bool = False
    comm_semantic_extra_token_threshold: float = 0.10
    # Frozen-policy structural probe for CA-CSR. This evaluation-only switch
    # lets a pretrained observational decoder alter sensing logits after a
    # local temporal crisis gate; it is never enabled during PPO collection.
    comm_semantic_cacsr_eval_enabled: bool = False
    comm_semantic_cacsr_eval_gain: float = 0.0
    comm_semantic_cacsr_quality_threshold: float = 0.9704283028841019
    comm_semantic_cacsr_stagnation_frames: int = 5
    comm_semantic_cacsr_improvement_epsilon: float = 0.01
    comm_semantic_cacsr_ema_alpha: float = 0.30
    # Endpoint-Deficit Semantic Kinematic Field (ED-SKF). Delivered sparse
    # target claims attract assistance only while a bistatic target lacks an
    # endpoint. Full targets exert zero physical force; their competition is
    # handled by the target-assignment logits.
    semantic_kinematic_field_enabled: bool = False
    semantic_kinematic_field_gain: float = 0.15
    # Learnable communication-conditioned target movement (CTMH). Each target
    # token produces radial/tangential motion coefficients; negotiated local
    # responsibilities combine the candidates. The final layer is zero-init so
    # old checkpoints remain behaviorally identical until PPO learns the path.
    target_conditioned_movement_enabled: bool = False
    target_conditioned_movement_gain: float = 0.15
    target_allocation_movement_only: bool = False
    # CTDE-only balanced-coverage teacher.  During training, the centralized
    # state supplies a minimum-distance one-to-one UAV/target assignment; at
    # execution the actor still receives only its local observation and U2U
    # inbox.  The message payload itself remains a learned 16-D representation.
    target_allocation_teacher_enabled: bool = False
    target_allocation_teacher_label_coef: float = 0.20
    target_allocation_teacher_movement_coef: float = 0.50
    target_allocation_teacher_message_coef: float = 0.05
    target_allocation_teacher_epochs: int = 1
    target_allocation_teacher_differentiable_comm: bool = False
    target_allocation_teacher_switching_penalty_m: float = 0.0
    # Slow-layer teacher: "distance" retains the historical Hungarian label;
    # "qos_bistatic" assigns the best reachable bistatic geometry to targets
    # with the largest P_D deficit, with no teacher input exposed at execution.
    target_allocation_teacher_mode: str = "distance"
    target_allocation_teacher_qos_floor: float = 0.60
    target_allocation_teacher_qos_weight: float = 2.0
    target_allocation_teacher_commitment_frames: int = 5
    target_allocation_teacher_height_m: float = 20.0
    target_allocation_teacher_crisis_only_enabled: bool = False
    target_allocation_teacher_crisis_floor: float = 0.60
    target_allocation_sinkhorn_enabled: bool = False
    # Capacitated decentralized matching. Sparse physical target tokens form
    # local rows of a UAV-target bid graph; alternating Sinkhorn projection
    # enforces endpoint loads. Coupling is annealed to avoid one-shot collapse.
    capacity_matching_enabled: bool = False
    capacity_matching_row_capacity: int = 2
    capacity_matching_column_capacity: int = 2
    capacity_matching_temperature: float = 0.35
    capacity_matching_iterations: int = 32
    capacity_matching_blend_start: float = 0.0
    capacity_matching_blend_end: float = 1.0
    capacity_matching_anneal_frames: int = 100000
    # Per-module LR: encoder=1e-5, attention=1e-5, head=5e-5 (Full).
    # When freeze_attention=True: attention LR→0. False = single LR for all.
    use_per_module_lr: bool = False
    # Neighbor belief fusion via multi-head attention + CI.
    neighbor_belief_fusion: bool = False
    # B3: Uncertainty-aware P0 scoring weights.
    p0_beta_uncertainty: float = 0.0   # uncertainty penalty (0=off)
    p0_eta_aoi: float = 0.0           # AoI urgency bonus (0=off)
    # ── Calibrate–Gate–Schedule–Recover: belief fusion safety system ──
    # Layer 1: NIS-driven covariance calibration.
    belief_nis_enabled: bool = False        # master switch for NIS calibration
    belief_nis_window: float = 0.1          # ρ for NIS EMA (0.1 ≈ 10-frame window)
    belief_nis_inflate_k: float = 2.0       # k in λ = 1 + k·max(r̄−1, 0) (linear, gentle)
    belief_nis_lambda_max: float = 5.0      # max inflation factor
    belief_nis_deflate_rate: float = 0.95   # per-frame decay when r̄ ≤ 1
    belief_cov_floor_pos: float = 25.0      # σ²_min for position (5 m σ)
    belief_cov_floor_vel: float = 1.0       # σ²_min for velocity (1 m/s σ)
    # NIS state machine hysteresis
    belief_nis_enter_threshold: float = 1.3  # r̄ above this → suspect
    belief_nis_exit_threshold: float = 1.2   # r̄ below this → recovering
    belief_nis_enter_frames: int = 3         # consecutive frames to enter SUSPECT
    belief_nis_exit_frames: int = 5          # consecutive frames to exit to NORMAL
    # Layer 2: Disagreement-gated CI fusion.
    trust_gate_enabled: bool = False        # master switch for trust gating
    trust_disagreement_threshold: float = 6.0   # chi² 4-dof trigger threshold
    trust_aoi_max: float = 50.0             # max AoI before age penalty
    trust_weight_max: float = 0.6           # ω_max per neighbor in CI
    trust_local_weight_min: float = 0.25    # minimum local belief weight
    trust_ema_rho: float = 0.1              # EMA ρ for trust scores
    trust_quarantine_nis_ratio: float = 1.5 # quarantine if NIS_fused > ratio × NIS_local
    trust_quarantine_duration: int = 10     # frames to quarantine after trigger
    # Layer 3: Safe P0 with bounded fusion correction.
    p0_safe_fallback: bool = False          # master switch for safe P0
    p0_fusion_confidence_min: float = 0.3   # min fusion confidence to use correction
    # DU-P0: Decision-Uncertainty-aware scheduling
    du_enabled: bool = False                # master switch for DU-P0
    du_ambiguity_threshold: float = 3.0     # A_q above this triggers probe mode
    du_ambiguity_bonus: float = 0.1         # gamma: bonus weight per unit ambiguity
    # Layer 4: Active probing + trust feedback recovery.
    active_probe_enabled: bool = False      # master switch for active probing
    active_probe_threshold: float = 3.0     # event-triggered: min score to probe
    active_probe_uncertainty_weight: float = 0.3  # c₁: covariance trace weight
    active_probe_nis_weight: float = 0.2         # c₂: NIS anomaly weight
    active_probe_aoi_weight: float = 0.5         # AoI urgency weight
    # Freeze attention (attn.* + attn_norm.*) — EH mode.
    # Only meaningful when use_per_module_lr=True.
    freeze_attention: bool = False
    # Advantage mode: 'scalar', 'target_wise', or 'bottleneck_risk'.
    # target_wise: per-target advantages aggregated via UAV-target
    # responsibility weights (inverse-distance softmax, detached).
    advantage_mode: str = 'scalar'
    # Temperature for inverse-distance target responsibility (meters).
    # Only used when advantage_mode='target_wise'.
    target_responsibility_tau_m: float = 50.0
    # Bottleneck-risk mode uses the current bottom target tail to route
    # per-target GAE, while retaining a scalar-advantage mixture for stability.
    risk_tail_fraction: float = 0.50
    # Legacy CVaR constraint used by the trainer. These fields previously
    # existed only in YAML and were silently discarded by the dataclass loader.
    cvar_tau: float = 0.0
    cvar_epsilon: float = 0.05
    # Teacher-free target-wise CVaR condition for PPO.  This is deliberately
    # separate from the legacy reward penalty above and remains default-off.
    constrained_cvar_ppo_enabled: bool = False
    constrained_cvar_qos_floor: float = 0.60
    constrained_cvar_tail_fraction: float = 0.20
    constrained_cvar_penalty: float = 0.10
    constrained_cvar_dual_step_size: float = 0.10
    constrained_cvar_dual_maximum: float = 10.0
    constrained_cvar_primal_tolerance: float = 0.01
    constrained_cvar_power_tolerance_w: float = 1.0e-9
    constrained_cvar_convergence_patience: int = 3
    risk_target_temperature: float = 0.10
    risk_target_floor: float = 0.60
    risk_scalar_mix: float = 0.25
    # Safe warm-start adaptation: freeze the foundation actor and train only a
    # bounded, locally gated movement residual.
    risk_residual_delta_max: float = 0.06
    risk_residual_hidden_dim: int = 32
    risk_residual_gate_bias: float = -2.0
    risk_residual_learning_rate: float = 3.0e-5
    risk_residual_directional_basis_enabled: bool = False
    # Architecture V2 removes parameter-level K/Q dependence from the
    # deployed actor. Every target-dependent decision is produced by a shared
    # scorer and pooled with a set operation; absolute UAV IDs are not inputs.
    architecture_v2_enabled: bool = False
    architecture_v2_prior_gain: float = 1.0
    architecture_v2_distance_weight: float = 0.25
    architecture_v2_qos_floor: float = 0.60
    architecture_v2_comm_prior_gain: float = 2.0
    architecture_v2_comm_crisis_threshold: float = 0.25
    architecture_v2_consensus_enabled: bool = True
    architecture_v2_matching_temperature: float = 0.35
    architecture_v2_movement_consensus_blend: float = 1.0
    architecture_v2_endpoint_consensus_gain: float = 2.0
    architecture_v2_bid_residual_scale: float = 0.25
    architecture_v2_sensing_aligned_claims_enabled: bool = False
    # ADMN-inspired, but identity-free, modular coordination.  Shared experts
    # alter only the slow bid/movement latent.  A set-pooled local router
    # chooses their mixture from locally observable target/QoS context, so
    # decentralized execution and target permutation equivariance are kept.
    architecture_v2_modular_coordination_enabled: bool = False
    architecture_v2_modular_num_experts: int = 3
    architecture_v2_modular_gain: float = 0.25
    architecture_v2_modular_temperature: float = 0.75
    architecture_v2_modular_balance_coef: float = 0.01
    architecture_v2_modular_specialization_coef: float = 0.002
    architecture_v2_modular_lr_scale: float = 1.0
    # Cardinality-independent local communication decisions. The set pooling
    # consumes only one UAV's own outgoing per-target tokens.
    scale_equivariant_comm_heads_enabled: bool = False
    # Replace K-long agent one-hot input in round negotiation with the shared
    # proposal/response phase, preserving permutation equivariance.
    permutation_equivariant_round_encoding_enabled: bool = False
    # Set-based scalar/per-target MAPPO value function. Unlike the auxiliary
    # risk head, this is the main PPO baseline and has no fixed-width K/Q MLP.
    equivariant_value_critic_enabled: bool = False
    # CTDE-only set/distributional risk critic.  Shared UAV/target encoders
    # remove absolute node identities; the module is absent from deployment.
    set_risk_critic_enabled: bool = False
    risk_critic_hidden_dim: int = 128
    risk_critic_num_quantiles: int = 16
    risk_critic_cvar_alpha: float = 0.20
    risk_critic_monotonic_quantiles_enabled: bool = False
    risk_critic_qos_floor: float = 0.60
    risk_critic_quantile_coef: float = 0.25
    risk_critic_constraint_coef: float = 0.10
    # <=0 selects an automatic negative/positive ratio per minibatch.
    risk_critic_constraint_positive_weight: float = 0.0
    # Freeze the actor and legacy scalar/target critics; fit only the set-risk
    # branch on trajectories from the fixed behaviour policy.
    risk_critic_only_training: bool = False


@dataclass
class MasterConfig:
    scenario: ScenarioParams = field(default_factory=ScenarioParams)
    uav: UAVParams = field(default_factory=UAVParams)
    target: TargetParams = field(default_factory=TargetParams)
    otfs: OTFSParams = field(default_factory=OTFSParams)
    channel: ChannelParams = field(default_factory=ChannelParams)
    detection: DetectionParams = field(default_factory=DetectionParams)
    p0_solver: P0SolverParams = field(default_factory=P0SolverParams)
    marl: MARLParams = field(default_factory=MARLParams)
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])

    def validate_runtime_boundary(self) -> None:
        """Validate a possibly programmatically resized live configuration.

        File-backed configurations require one normalized target weight per
        target.  Tests and small programmatic probes historically resize K/Q
        after loading: the environment normalizes a weight prefix (or uses a
        uniform vector when Q grows), and a reporter cap above a reduced K is
        simply non-binding.  Those two resizing consequences are the only
        relaxed invariants; every other physical and protocol check remains
        active.
        """
        self.validate(allow_target_weight_resize=True)

    def validate(self, *, allow_target_weight_resize: bool = False) -> None:
        """Fail closed on malformed or physically invalid core parameters.

        Every declared field is type-checked so YAML booleans, strings, and
        null values cannot silently enter numeric code. Semantic checks are
        intentionally limited to the stable scenario/physics/training spine;
        experimental options keep their existing freedom beyond their type.
        """
        _validate_dataclass_types(self, "config")
        marl = self.marl
        _require_positive(
            "config.marl.distributed_decision_sufficient_dynamic_range",
            marl.distributed_decision_sufficient_dynamic_range,
        )
        for name in (
            "distributed_decision_sufficient_max_bits",
            "u2u_belief_feedback_mean_bits",
            "u2u_belief_feedback_cov_bits",
            "u2u_belief_feedback_aoi_bits",
        ):
            _require_positive(f"config.marl.{name}", getattr(marl, name))

        scenario = self.scenario
        for name in ("K", "Q", "T", "C"):
            _require_positive(f"config.scenario.{name}", getattr(scenario, name))
        for index, extent in enumerate(scenario.region_size):
            _require_positive(f"config.scenario.region_size[{index}]", extent)
        _require_positive("config.scenario.height", scenario.height)
        _require_positive("config.scenario.dt", scenario.dt)
        _require_choice(
            "config.scenario.sensing_energy_mode",
            scenario.sensing_energy_mode,
            {"dt_frame", "cpi_frame"},
        )

        uav = self.uav
        for name in (
                "v_max", "d_safe", "P_sense", "P_report",
                "P_fly_static", "P_fly_coeff"):
            _require_nonnegative(f"config.uav.{name}", getattr(uav, name))
        for name in ("P_sense_max", "P_isac_total", "B_max"):
            _require_positive(f"config.uav.{name}", getattr(uav, name))
        if uav.P_sense > uav.P_sense_max:
            raise ValueError(
                "config.uav.P_sense must not exceed config.uav.P_sense_max")
        if uav.P_sense_max > uav.P_isac_total:
            raise ValueError(
                "config.uav.P_sense_max must not exceed "
                "config.uav.P_isac_total")

        target = self.target
        _require_choice(
            "config.target.motion_model", target.motion_model, {"CV", "CT", "CA"})
        _require_finite("config.target.ct_turn_rate", target.ct_turn_rate)
        for index, speed in enumerate(target.speed_range):
            _require_nonnegative(f"config.target.speed_range[{index}]", speed)
        if target.speed_range[0] > target.speed_range[1]:
            raise ValueError(
                "config.target.speed_range minimum must not exceed maximum")
        _require_nonnegative("config.target.sigma_a", target.sigma_a)
        _require_positive("config.target.rcs", target.rcs)
        if not target.omega_q:
            raise ValueError("config.target.omega_q must not be empty")
        if allow_target_weight_resize:
            effective_weights = (
                target.omega_q[:scenario.Q]
                if len(target.omega_q) >= scenario.Q
                else target.omega_q
            )
        else:
            if len(target.omega_q) != scenario.Q:
                raise ValueError(
                    "config.target.omega_q must contain exactly "
                    "config.scenario.Q weights")
            effective_weights = target.omega_q
        for index, weight in enumerate(effective_weights):
            _require_nonnegative(f"config.target.omega_q[{index}]", weight)
        if sum(effective_weights) <= 0.0:
            raise ValueError(
                "config.target.omega_q effective weights must have "
                "positive sum")
        if (not allow_target_weight_resize
                and not math.isclose(
                    sum(effective_weights), 1.0, rel_tol=1.0e-8,
                    abs_tol=1.0e-8)):
            raise ValueError("config.target.omega_q must sum to 1")

        otfs = self.otfs
        for name in ("fc", "B", "delta_f", "T_sym"):
            _require_positive(f"config.otfs.{name}", getattr(otfs, name))
        for name in ("M", "N", "n_cpi"):
            _require_positive(f"config.otfs.{name}", getattr(otfs, name))
        for name in ("g_tx_dBi", "g_rx_dBi"):
            _require_finite(f"config.otfs.{name}", getattr(otfs, name))
        if not math.isclose(otfs.B, otfs.M * otfs.delta_f,
                            rel_tol=1.0e-6):
            raise ValueError(
                "config.otfs.B must equal config.otfs.M * config.otfs.delta_f")
        if not math.isclose(otfs.T_sym * otfs.delta_f, 1.0,
                            rel_tol=1.0e-6):
            raise ValueError(
                "config.otfs.T_sym must equal 1 / config.otfs.delta_f")
        cpi_duration = otfs.n_cpi * otfs.N * otfs.T_sym
        if cpi_duration > scenario.dt:
            raise ValueError(
                "OTFS CPI duration must not exceed config.scenario.dt")

        channel = self.channel
        for name in ("NF", "kT"):
            _require_positive(f"config.channel.{name}", getattr(channel, name))
        for name in (
                "ric_K", "los_a", "los_b", "eta_los_dB", "eta_nlos_dB"):
            _require_nonnegative(f"config.channel.{name}", getattr(channel, name))

        detection = self.detection
        _require_probability("config.detection.P_FA", detection.P_FA, strict=True)
        _require_positive("config.detection.c_det", detection.c_det)
        _require_probability("config.detection.g_min", detection.g_min)
        _require_choice(
            "config.detection.dd_gain_mode", detection.dd_gain_mode,
            {"binary", "continuous"})
        _require_positive("config.detection.K_q_max", detection.K_q_max)
        if (not allow_target_weight_resize
                and detection.K_q_max > scenario.K):
            raise ValueError(
                "config.detection.K_q_max must not exceed config.scenario.K")
        _require_positive("config.detection.B_q", detection.B_q)
        _require_probability("config.detection.P_D_min", detection.P_D_min)

        p0_solver = self.p0_solver
        _require_positive(
            "config.p0_solver.capacity_per_rx", p0_solver.capacity_per_rx)
        _require_positive("config.p0_solver.latency_max", p0_solver.latency_max)
        if p0_solver.latency_max > scenario.dt:
            raise ValueError(
                "config.p0_solver.latency_max must not exceed "
                "config.scenario.dt")

        marl = self.marl
        if not marl.hidden_layers:
            raise ValueError("config.marl.hidden_layers must not be empty")
        for index, width in enumerate(marl.hidden_layers):
            _require_positive(f"config.marl.hidden_layers[{index}]", width)
        # Evaluation-only profiles intentionally use lr=0 and/or ppo_epochs=0
        # to make accidental training a no-op.  These sentinels are valid;
        # negative values are never meaningful.
        _require_nonnegative("config.marl.lr", marl.lr)
        _require_positive("config.marl.ppo_clip", marl.ppo_clip)
        _require_probability("config.marl.gamma", marl.gamma, strict=True)
        _require_probability("config.marl.gae_lambda", marl.gae_lambda)
        for name in ("ppo_epochs", "num_episodes"):
            _require_nonnegative(f"config.marl.{name}", getattr(marl, name))
        if marl.training_fixed_seed < -1:
            raise ValueError(
                "config.marl.training_fixed_seed must be -1 or non-negative")
        if (marl.training_fixed_seed >= 0
                and marl.training_seed_replay_enabled):
            raise ValueError(
                "training_fixed_seed cannot be combined with "
                "training_seed_replay_enabled")
        for name in (
                "rollout_steps", "minibatch_size", "num_envs",
                "assignment_hold_frames", "actor_decision_interval",
                "movement_decision_interval", "obs_history_frames"):
            _require_positive(f"config.marl.{name}", getattr(marl, name))
        for name in ("entropy_init", "entropy_final"):
            _require_nonnegative(f"config.marl.{name}", getattr(marl, name))
        if marl.entropy_final > marl.entropy_init:
            raise ValueError(
                "config.marl.entropy_final must not exceed "
                "config.marl.entropy_init")
        _require_choice(
            "config.marl.detection_fusion_mode", marl.detection_fusion_mode,
            {"legacy_global", "central_oracle", "local_only", "u2u_distributed"})

        _require_probability(
            "config.marl.constrained_cvar_qos_floor",
            marl.constrained_cvar_qos_floor, strict=True)
        _require_probability(
            "config.marl.constrained_cvar_tail_fraction",
            marl.constrained_cvar_tail_fraction, strict=True)
        _require_positive(
            "config.marl.constrained_cvar_penalty",
            marl.constrained_cvar_penalty)
        _require_nonnegative(
            "config.marl.constrained_cvar_dual_step_size",
            marl.constrained_cvar_dual_step_size)
        _require_nonnegative(
            "config.marl.constrained_cvar_dual_maximum",
            marl.constrained_cvar_dual_maximum)
        _require_nonnegative(
            "config.marl.constrained_cvar_primal_tolerance",
            marl.constrained_cvar_primal_tolerance)
        _require_nonnegative(
            "config.marl.constrained_cvar_power_tolerance_w",
            marl.constrained_cvar_power_tolerance_w)
        _require_positive(
            "config.marl.constrained_cvar_convergence_patience",
            marl.constrained_cvar_convergence_patience)
        if marl.constrained_cvar_ppo_enabled and marl.cvar_tau > 0.0:
            raise ValueError(
                "constrained_cvar_ppo_enabled cannot be combined with the "
                "legacy cvar_tau reward penalty")
        if (marl.constrained_cvar_ppo_enabled
                and marl.target_kl is not None
                and marl.target_kl <= 0.0):
            raise ValueError(
                "constrained_cvar_ppo_enabled requires a positive target_kl "
                "for transactional actor rollback")

        # Fixed-structure distributed primal--dual power control is an
        # explicitly bounded execution path.  Validate its numerical knobs
        # here so malformed YAML cannot reach the inner solver and silently
        # weaken the RF-budget guarantee.  The mode remains opt-in and all
        # defaults preserve the historical analytical/learned power paths.
        _require_probability(
            "config.marl.analytical_sensing_power_reserve_pd",
            marl.analytical_sensing_power_reserve_pd)
        _require_positive(
            "config.marl.distributed_primal_dual_power_rounds",
            marl.distributed_primal_dual_power_rounds)
        _require_nonnegative(
            "config.marl.distributed_primal_dual_power_acceptance_tolerance",
            marl.distributed_primal_dual_power_acceptance_tolerance)
        _require_positive(
            "config.marl.distributed_primal_dual_power_cost_per_watt",
            marl.distributed_primal_dual_power_cost_per_watt)
        _require_nonnegative(
            "config.marl.distributed_primal_dual_power_regularization",
            marl.distributed_primal_dual_power_regularization)
        _require_nonnegative(
            "config.marl.distributed_primal_dual_power_actor_proximal_"
            "regularization",
            marl.distributed_primal_dual_power_actor_proximal_regularization)
        for name in (
                "distributed_primal_dual_power_primal_tolerance",
                "distributed_primal_dual_power_stationarity_tolerance",
                "distributed_primal_dual_power_dual_tolerance"):
            _require_nonnegative(f"config.marl.{name}", getattr(marl, name))
        _require_positive(
            "config.marl.distributed_primal_dual_power_convergence_patience",
            marl.distributed_primal_dual_power_convergence_patience)
        if marl.distributed_primal_dual_power_enabled:
            if not marl.analytical_sensing_power_enabled:
                raise ValueError(
                    "distributed_primal_dual_power_enabled requires "
                    "analytical_sensing_power_enabled")
            if not marl.joint_isac_power_enabled:
                raise ValueError(
                    "distributed_primal_dual_power_enabled requires "
                    "joint_isac_power_enabled")
            if marl.analytical_sensing_power_reserve_pd <= 0.0:
                raise ValueError(
                    "distributed_primal_dual_power_enabled requires a "
                    "positive analytical_sensing_power_reserve_pd")
        if (marl.distributed_primal_dual_power_enabled
                and marl.distributed_replicated_power_enabled):
            raise ValueError(
                "distributed_primal_dual_power_enabled cannot be combined "
                "with distributed_replicated_power_enabled")
        _require_positive(
            "config.marl.temporal_unrolled_power_horizon",
            marl.temporal_unrolled_power_horizon)
        _require_positive(
            "config.marl.temporal_unrolled_power_inner_iterations",
            marl.temporal_unrolled_power_inner_iterations)
        _require_probability(
            "config.marl.temporal_unrolled_power_warm_start_mix",
            marl.temporal_unrolled_power_warm_start_mix)
        _require_nonnegative(
            "config.marl.temporal_unrolled_power_pareto_tolerance",
            marl.temporal_unrolled_power_pareto_tolerance)
        _require_nonnegative(
            "config.marl.temporal_unrolled_power_rf_tangent_slack_tolerance",
            marl.temporal_unrolled_power_rf_tangent_slack_tolerance)
        _require_nonnegative(
            "config.marl.temporal_unrolled_power_rf_tangent_rank_tolerance",
            marl.temporal_unrolled_power_rf_tangent_rank_tolerance)
        _require_positive(
            "config.marl.temporal_unrolled_power_pareto_max_iterations",
            marl.temporal_unrolled_power_pareto_max_iterations)
        _require_positive(
            "config.marl.temporal_feasible_structure_temperature",
            marl.temporal_feasible_structure_temperature)
        _require_nonnegative(
            "config.marl.temporal_feasible_structure_inertia",
            marl.temporal_feasible_structure_inertia)
        _require_positive(
            "config.marl.temporal_feasible_structure_max_structures",
            marl.temporal_feasible_structure_max_structures)
        if marl.temporal_unrolled_power_enabled:
            if marl.temporal_unrolled_power_horizon < 2:
                raise ValueError(
                    "temporal_unrolled_power_enabled requires horizon >= 2")
            if not marl.distributed_primal_dual_power_enabled:
                raise ValueError(
                    "temporal_unrolled_power_enabled requires "
                    "distributed_primal_dual_power_enabled")
            if not marl.constrained_cvar_ppo_enabled:
                raise ValueError(
                    "temporal_unrolled_power_enabled requires "
                    "constrained_cvar_ppo_enabled")
            if marl.target_kl is None or marl.target_kl <= 0.0:
                raise ValueError(
                    "temporal_unrolled_power_enabled requires a positive "
                    "target_kl for transactional rollback")
        if marl.temporal_feasible_structure_enabled:
            if not marl.temporal_unrolled_power_enabled:
                raise ValueError(
                    "temporal_feasible_structure_enabled requires "
                    "temporal_unrolled_power_enabled")
            if marl.target_allocation_teacher_enabled:
                raise ValueError(
                    "temporal feasible structure is teacher-free and cannot "
                    "be combined with target_allocation_teacher_enabled")
            if marl.analytical_structure_ranking_enabled:
                raise ValueError(
                    "temporal feasible structure requires executed P0 ranking "
                    "to retain actor sensing power; disable "
                    "analytical_structure_ranking_enabled")

        if marl.distributed_no_truth_fail_closed:
            if not marl.tracking_enabled:
                raise ValueError(
                    "config.marl.distributed_no_truth_fail_closed requires "
                    "config.marl.tracking_enabled")
            if not marl.distributed_coordination_use_local_belief_targets:
                raise ValueError(
                    "config.marl.distributed_no_truth_fail_closed requires "
                    "config.marl.distributed_coordination_use_local_belief_targets")

        _require_positive(
            "config.marl.comm_bandwidth_hz", marl.comm_bandwidth_hz)
        _require_nonnegative(
            "config.marl.comm_tx_power_w", marl.comm_tx_power_w)
        _require_positive("config.marl.comm_deadline_s", marl.comm_deadline_s)
        _require_nonnegative(
            "config.marl.comm_processing_delay_s", marl.comm_processing_delay_s)
        if marl.comm_processing_delay_s >= marl.comm_deadline_s:
            raise ValueError(
                "config.marl.comm_processing_delay_s must be less than "
                "config.marl.comm_deadline_s")
        if not marl.comm_rate_bits_per_dim:
            raise ValueError("config.marl.comm_rate_bits_per_dim must not be empty")
        for index, bits in enumerate(marl.comm_rate_bits_per_dim):
            _require_nonnegative(
                f"config.marl.comm_rate_bits_per_dim[{index}]", bits)
        _require_probability(
            "config.marl.comm_power_fraction_min", marl.comm_power_fraction_min)
        _require_probability(
            "config.marl.comm_power_fraction_max", marl.comm_power_fraction_max)
        if marl.comm_power_fraction_min > marl.comm_power_fraction_max:
            raise ValueError(
                "config.marl.comm_power_fraction_min must not exceed "
                "config.marl.comm_power_fraction_max")

        qos_floors = (
            marl.comm_qos_worst_min,
            marl.comm_qos_weak3_min,
            marl.comm_qos_steady_min,
        )
        for name, value in zip(
                ("comm_qos_worst_min", "comm_qos_weak3_min",
                 "comm_qos_steady_min"), qos_floors):
            _require_probability(f"config.marl.{name}", value)
        if tuple(sorted(qos_floors)) != qos_floors:
            raise ValueError(
                "communication QoS floors must satisfy worst <= weak3 <= steady")

        if not self.seeds:
            raise ValueError("config.seeds must not be empty")
        for index, seed in enumerate(self.seeds):
            _require_nonnegative(f"config.seeds[{index}]", seed)
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("config.seeds must not contain duplicates")

    def to_small_config(self) -> "MasterConfig":
        """Return a reduced config for fast smoke tests."""
        small = MasterConfig()
        small.scenario.K = 2
        small.scenario.Q = 1
        small.scenario.T = 10
        small.marl.hidden_layers = [32, 32]
        small.marl.rollout_steps = 64
        small.marl.minibatch_size = 16
        small.marl.num_episodes = 10
        small.marl.ppo_epochs = 2
        return small


def _validate_dataclass_types(value, path: str) -> None:
    """Validate one dataclass tree against its runtime type annotations."""
    if not dataclasses.is_dataclass(value) or isinstance(value, type):
        raise ValueError(f"{path} must be a configuration mapping")
    for item in dataclasses.fields(value):
        _validate_value_type(getattr(value, item.name), item.type,
                             f"{path}.{item.name}")


def _validate_value_type(value, expected_type, path: str) -> None:
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if dataclasses.is_dataclass(expected_type):
        if not isinstance(value, expected_type):
            raise ValueError(f"{path} must be a configuration mapping")
        _validate_dataclass_types(value, path)
        return
    if origin is list:
        if type(value) is not list:
            raise ValueError(f"{path} must be a list")
        for index, element in enumerate(value):
            _validate_value_type(element, args[0], f"{path}[{index}]")
        return
    if origin is tuple:
        if type(value) is not tuple:
            raise ValueError(f"{path} must be a tuple-compatible sequence")
        if len(args) == 2 and args[1] is Ellipsis:
            item_types = [args[0]] * len(value)
        else:
            if len(value) != len(args):
                raise ValueError(
                    f"{path} must contain exactly {len(args)} elements")
            item_types = args
        for index, (element, item_type) in enumerate(zip(value, item_types)):
            _validate_value_type(element, item_type, f"{path}[{index}]")
        return
    if origin in (Union, types.UnionType):
        for option in args:
            try:
                _validate_value_type(value, option, path)
                return
            except ValueError:
                pass
        raise ValueError(f"{path} has an invalid type")
    if expected_type is type(None):
        if value is not None:
            raise ValueError(f"{path} must be null")
        return
    if expected_type is bool:
        if type(value) is not bool:
            raise ValueError(f"{path} must be a boolean")
        return
    if expected_type is int:
        if type(value) is not int:
            raise ValueError(f"{path} must be an integer")
        return
    if expected_type is float:
        if type(value) not in (int, float):
            raise ValueError(f"{path} must be a number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{path} must be finite")
        return
    if expected_type is str:
        if type(value) is not str:
            raise ValueError(f"{path} must be a string")
        return
    if not isinstance(value, expected_type):
        raise ValueError(f"{path} has an invalid type")


def _require_finite(path: str, value) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{path} must be finite")


def _require_positive(path: str, value) -> None:
    _require_finite(path, value)
    if value <= 0:
        raise ValueError(f"{path} must be greater than zero")


def _require_nonnegative(path: str, value) -> None:
    _require_finite(path, value)
    if value < 0:
        raise ValueError(f"{path} must be nonnegative")


def _require_probability(path: str, value, *, strict: bool = False) -> None:
    _require_finite(path, value)
    valid = 0.0 < value < 1.0 if strict else 0.0 <= value <= 1.0
    if not valid:
        interval = "(0, 1)" if strict else "[0, 1]"
        raise ValueError(f"{path} must be in {interval}")


def _require_choice(path: str, value: str, choices: set) -> None:
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{path} must be one of: {allowed}")


def _dict_to_dataclass(cls, d: dict, path: str = "config"):
    """Recursively convert a mapping and reject unknown configuration keys.

    Silent key dropping is unsafe for experiments: a misspelled switch can
    otherwise produce a valid-looking run under a different controller.
    """
    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, value in d.items():
        if key not in field_types:
            raise ValueError(f"unknown configuration key: {path}.{key}")
        ft = field_types[key]
        if dataclasses.is_dataclass(ft) and isinstance(value, dict):
            kwargs[key] = _dict_to_dataclass(ft, value, f"{path}.{key}")
        elif hasattr(ft, '__origin__') and ft.__origin__ in (list, List):
            kwargs[key] = value
        elif hasattr(ft, '__origin__') and ft.__origin__ in (tuple, Tuple):
            if type(value) not in (list, tuple):
                raise ValueError(
                    f"{path}.{key} must be a list/tuple sequence")
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _deep_merge_dict(base: dict, override: dict) -> dict:
    """Recursively merge config dictionaries without mutating either input."""
    merged = dict(base)
    for key, value in override.items():
        if (key in merged and isinstance(merged[key], dict)
                and isinstance(value, dict)):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_config(path: str, seen: set) -> dict:
    resolved = os.path.abspath(path)
    if resolved in seen:
        raise ValueError(f'cyclic config inheritance involving {resolved}')
    seen.add(resolved)
    with open(resolved, 'r', encoding='utf-8') as f:
        raw = load_unique_yaml(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f'config root must be a mapping: {resolved}')
    parent = raw.pop('extends', None)
    if parent is None:
        seen.remove(resolved)
        return raw
    if not isinstance(parent, str):
        raise ValueError('config extends must be a relative or absolute path')
    parent_path = (parent if os.path.isabs(parent) else
                   os.path.join(os.path.dirname(resolved), parent))
    base = _load_raw_config(parent_path, seen)
    seen.remove(resolved)
    return _deep_merge_dict(base, raw)


def load_config(path: str) -> MasterConfig:
    """Load YAML, optionally inheriting another file via ``extends``."""
    config = _dict_to_dataclass(MasterConfig, _load_raw_config(path, set()))
    config.validate()
    return config


def get_default_config() -> MasterConfig:
    """Load the default config from the package."""
    default_path = os.path.join(os.path.dirname(__file__), 'default.yaml')
    return load_config(default_path)
