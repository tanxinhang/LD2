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
from uav_isac.environment.reward import RewardComputer
from uav_isac.environment.constraints import ConstraintChecker
from uav_isac.physical.deflection import DeflectionComputer
from uav_isac.physical.inner_solver import InnerSolver
from uav_isac.physical.detection import compute_detection_probabilities


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
    valid_pair: bool = False                 # P0 selected >=1 bistatic pair -> sensing happened
    no_tx: bool = False                      # zero TX this frame (no sensing possible)
    all_same_role: bool = False              # all K UAVs picked the same role (degenerate)


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

        # Action space
        self.action_space = ActionSpace(
            v_max=ua.v_max, dt=sc.dt, rng=self.rng,
            learn_roles=bool(getattr(self.cfg.marl, 'learn_roles', False)),
        )

        # Observation builder
        self.obs_builder = ObservationBuilder(
            K=self.K, Q=self.Q, area_size=self.area_size, height=self.height,
            use_relative_features=getattr(ma, 'rel_features', False),
            use_p0_info=getattr(ma, 'use_p0_sinr_gated', False),
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
        )

        # Truncate/pad omega_q to match Q
        omega_q = np.array(ta.omega_q[:self.Q]) if len(ta.omega_q) >= self.Q \
                  else np.ones(self.Q) / self.Q

        # Inner P0 solver
        self.inner_solver = InnerSolver(
            K_q_max=de.K_q_max, B_q=de.B_q,
            capacity_per_rx=p0.capacity_per_rx,
            latency_max=p0.latency_max,
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

        # State objects (created in reset)
        self.uavs: List[UAV] = []
        self.targets: List[Target] = []
        self.belief_mgr: Optional[BeliefManager] = None
        self.fc_position: Optional[np.ndarray] = None
        self.t: int = 0
        self.prev_P_D: Optional[np.ndarray] = None
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
        self._comm_msgs: Dict[int, np.ndarray] = {}  # per-agent communication messages (16-dim)
        self._gru_hidden: Dict[int, np.ndarray] = {}  # per-agent GRU hidden state (64-dim)

    def reset(self) -> Tuple[Dict[int, np.ndarray], Dict]:
        """Reset the environment to initial state.

        Returns:
            (observations_dict, info_dict)
        """
        self.t = 0
        self.prev_P_D = None
        self._prev_obs = {}  # clear history on reset
        self._gru_hidden = {}  # clear GRU state on reset

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
                P_report=self.cfg.uav.P_report,
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
            speed = self.rng.uniform(*self.cfg.target.speed_range)
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
        )

        # Build initial observations
        obs = self._build_observations()

        info = {
            'uav_positions': np.array([u.pos for u in self.uavs]),
            'target_positions': true_positions,
        }

        return obs, info

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
                self.uavs[k].apply_action(actions[k].delta_p, actions[k].role)
            uav_positions[k] = self.uavs[k].pos
            uav_velocities[k] = self.uavs[k].vel
            roles[k] = self.uavs[k].role

        # 2. Step target dynamics
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
        )
        self._last_deflection_entries = deflection_entries  # for obs coordination features

        # B6: choose what P0 RANKS candidates on. Oracle = true geometry; deployable
        # = fused belief estimate (P0 cannot see true targets at deployment). The
        # realized D_q*/P_D below always come from the TRUE deflection of the picks.
        if self.p0_uses_belief:
            if self.neighbor_belief_fusion and self._belief_fusion_module is not None:
                fused = self._fuse_beliefs_attention()  # attention-weighted CI fusion
            else:
                fused = self.belief_mgr.mean.mean(axis=0)  # uniform mean (legacy)
            est_pos = np.stack([fused[:, 0], fused[:, 1], np.zeros(self.Q)], axis=1)
            est_vel = np.stack([fused[:, 2], fused[:, 3], np.zeros(self.Q)], axis=1)
            ranking_entries = self.deflection_computer.compute(
                uav_positions, uav_velocities,
                est_pos, est_vel,
                roles, self.fc_position,
                role_agnostic=role_agnostic,
            )
        else:
            ranking_entries = deflection_entries

        # 4. Inner P0 solver with assignment hold (reduces reward non-stationarity)
        hold_frames = getattr(self.cfg.marl, 'assignment_hold_frames', 1)
        should_resolve = (self.t == 1 or
                          self.t % hold_frames == 0 or
                          self._cached_p0_solution is None)
        if should_resolve:
            # B3: build uncertainty inputs for P0
            p0_cov = None; p0_aoi = None
            if self.p0_beta_uncertainty > 0 or self.p0_eta_aoi > 0:
                if self.p0_uses_belief and self.neighbor_belief_fusion:
                    fused_belief = self._fuse_beliefs_attention()
                else:
                    fused_belief = self.belief_mgr.mean.mean(axis=0)
                # Average covariance diagonal per target
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

            p0_solution = self.inner_solver.solve(
                ranking_entries, Q=self.Q, K=self.K,
                enforce_single_role=role_agnostic,
                belief_cov_diag=p0_cov,
                belief_aoi=p0_aoi,
                beta_uncertainty=self.p0_beta_uncertainty,
                eta_aoi=self.p0_eta_aoi,
            )
            self._cached_p0_solution = p0_solution
            self._assignment_switched = True
            self._last_solve_frame = self.t
        else:
            p0_solution = self._cached_p0_solution
            self._assignment_switched = False
        self._last_selected_set = p0_solution.selected_set  # for next obs

        # Realized per-target deflection = TRUE d_eff of the SELECTED pairs.
        if self.p0_uses_belief:
            d_true = {(e.i, e.j, e.q): e.d_eff for e in deflection_entries}
            D_q_star = np.zeros(self.Q, dtype=np.float64)
            for (i, j, q) in p0_solution.selected_set:
                D_q_star[q] += d_true.get((i, j, q), 0.0)
        else:
            D_q_star = p0_solution.D_q_star

        # 4b. If P0 assigned roles, derive each UAV's role from the selection
        #     (tx if used as transmitter, rx if used as receiver, else idle) and
        #     write it back for observations / diagnostics / next frame.
        if role_agnostic:
            derived = np.full(self.K, 2, dtype=np.int32)  # default idle
            for (i, j, q) in p0_solution.selected_set:
                derived[i] = 0  # tx
                derived[j] = 1  # rx
            for k in range(self.K):
                self.uavs[k].role = int(derived[k])
            roles = derived

        # 5. Compute P_D per target (from REALIZED true-geometry deflection)
        P_D_q = compute_detection_probabilities(D_q_star, self.cfg.detection.P_FA)

        # 6. Check constraints
        batteries = np.array([u.battery for u in self.uavs])
        constraint_info = self.constraint_checker.check_all(
            uav_positions, batteries, P_D_q
        )

        # 7. Compute rewards (constraint penalties handled by Lagrangian in trainer)
        team_reward = self.reward_computer.compute_team_reward(
            D_q_star,                 # realized (true-geometry) deflection of the picks
            p0_solution.total_bits,
            0.0,  # constraint_penalty removed; Lagrangian handles constraints
            P_D_q=P_D_q,              # for direct P_D reward term (alpha_pd > 0)
        )

        # Potential-based distance shaping: F = γΦ(s') - Φ(s), Φ = -Σ_q min_k dist.
        # Action-dependent part rewards moving UAVs toward targets; policy-invariant.
        if self.use_distance_shaping:
            phi_old = self._coverage_potential(prev_uav_positions, target_positions)
            phi_new = self._coverage_potential(uav_positions, target_positions)
            team_reward += self.shape_w * (self.gamma_shape * phi_new - phi_old)

        # Marginal contributions (gated; disabled in team-only baseline)
        if getattr(self.cfg.marl, 'use_centered_marginal', False):
            marginal = self.reward_computer.compute_marginal_contributions(
                list(range(self.K)),
                p0_solution.selected_set,
                deflection_entries,
                self.Q,
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
            actual_D_q = self._compute_assigned_deflection(
                uav_positions, p0_solution.selected_set, deflection_entries,
                uav_velocities, target_positions, target_velocities, roles)
            actual_util = self.reward_computer.compute_team_utility_from_deflection(
                actual_D_q, p0_solution.total_bits)

            for k in range(self.K):
                # No-op: UAV k stays at its PREVIOUS position
                cf_positions = uav_positions.copy()
                cf_positions[k] = prev_uav_positions[k]
                cf_D_q = self._compute_assigned_deflection(
                    cf_positions, p0_solution.selected_set, deflection_entries,
                    uav_velocities, target_positions, target_velocities, roles)
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
            P_D_local_k = np.array([compute_PD(D_k[q], self.cfg.detection.P_FA)
                                    for q in range(self.Q)])
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
        n_tx = int(np.sum(roles == 0))
        n_rx = int(np.sum(roles == 1))
        n_selected = len(p0_solution.selected_set)
        all_same_role = bool(np.all(roles == roles[0])) if self.K > 0 else False

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
            valid_pair=bool(n_selected > 0),
            no_tx=bool(n_tx == 0),
            all_same_role=all_same_role,
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
        if uav_velocities is None or target_positions is None:
            # Fallback: use cached entries (actual positions only)
            D_q = np.zeros(self.Q, dtype=np.float64)
            entry_map = {(e.i, e.j, e.q): e.d_eff for e in _unused_entries}
            for (i, j, q) in selected_set:
                D_q[q] += entry_map.get((i, j, q), 0.0)
            return D_q

        # Recompute deflection for ALL pairs, then filter to selected
        all_entries = self.deflection_computer.compute(
            uav_positions, uav_velocities,
            target_positions, target_velocities,
            roles if roles is not None else np.zeros(self.K, dtype=np.int32),
            self.fc_position,
            role_agnostic=not self.learn_roles,
        )
        D_q = np.zeros(self.Q, dtype=np.float64)
        entry_map = {(e.i, e.j, e.q): e.d_eff for e in all_entries}
        for (i, j, q) in selected_set:
            D_q[q] += entry_map.get((i, j, q), 0.0)
        return D_q

    def _fuse_beliefs_attention(self) -> np.ndarray:
        """Fuse per-UAV beliefs via multi-head neighbor attention + CI.

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

            with torch.no_grad():
                fw, _, _ = self._belief_fusion_module(
                    q_mean, q_cov, q_aoi, q_pd,
                    nb_mean, nb_cov, nb_aoi, nb_pd,
                    nb_mask)
                fused_m, _ = self._belief_fusion_module.covariance_intersection_fusion(
                    q_mean, q_cov, nb_mean, nb_cov, fw, local_weight=0.25)

            fused_m_np = fused_m[0].cpu().numpy()  # already in raw coordinates
            all_fused += fused_m_np

        # Average over all agents' fused beliefs
        return all_fused / K

    def _build_observations(self) -> Dict[int, np.ndarray]:
        """Build local observations for all UAVs."""
        uav_states = [u.get_state() for u in self.uavs]
        assert self.belief_mgr is not None
        beliefs = [self.belief_mgr.get_all_beliefs(k) for k in range(self.K)]

        # Diagnostic: feed true target state instead of beliefs
        oracle = getattr(self.cfg.marl, 'oracle_obs', False)
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
            cur = self.obs_builder.build_local_obs(
                k, uav_states, beliefs, local_pd,
                oracle_targets=oracle_targets,
                selected_set=getattr(self, '_last_selected_set', []),
                deflection_entries=getattr(self, '_last_deflection_entries', None),
                comm_msgs=self._comm_msgs if self._comm_msgs else None,
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
            'prev_P_D_local': {k: v.copy() for k, v in self.prev_P_D_local.items()},
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
        self.prev_P_D_local = {k: v.copy() for k, v in state.get('prev_P_D_local', {}).items()}
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
