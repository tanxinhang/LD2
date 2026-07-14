"""Observation builder: constructs per-agent local observations.

Observation per UAV k:
  o_k = {
    self_pos (3), self_vel (3), battery (1), role (1),
    beliefs: for each target q: mean(4), cov_diag(4), aoi(1)  = 9 per target
    neighbor_msgs: for each other UAV: relative_pos(2), role(1), battery(1) = 4 per neighbor
    detection_status: P_D per target from previous frame (Q)
  }
"""

import numpy as np
from typing import Dict, List, Optional
from uav_isac.utils.types import BeliefState


class ObservationBuilder:
    """Builds local observations for each UAV agent.

    The observation is a flat vector encoding everything the UAV can locally
    observe. Global state (for the critic) is built separately.
    """

    def __init__(
        self,
        K: int,
        Q: int,
        area_size: tuple = (1000.0, 1000.0),
        height: float = 100.0,
        use_relative_features: bool = False,
        use_p0_info: bool = False,
    ):
        """
        Args:
            K: Number of UAVs
            Q: Number of targets
            area_size: Region dimensions (width, height) for normalization
            height: Fixed flight altitude
        """
        self.K = K
        self.Q = Q
        self.area_w, self.area_h = area_size
        self.height = height
        self.use_relative_features = use_relative_features
        self.use_history = False  # set True to stack previous frame features
        self._prev_rel = {}       # agent_id -> previous relative features array
        # God-view features: P0 solution info (SINR-gated for v2-physical)
        self.use_p0_global_info = use_p0_info

        # Observation dimension per agent
        self.self_dim = 8    # pos(3) + vel(3) + battery(1) + role_onehot(1)
        self.belief_dim = 9  # per target: mean(4) + cov_diag(4) + aoi(1)
        self.neighbor_dim = 8  # per neighbor: rel_pos(2)+rel_vel(2)+role(1)+heading(2)+nearest_dist(1)
        if self.use_p0_global_info:
            self.neighbor_dim += 1  # +in_pair(1) when P0 info enabled
        self.detection_dim = 1  # per target: P_D history
        self.relative_dim = 8  # per target: dx,dy,dist,sin_bearing,cos_bearing,dist_s1,dist_s2,in_range

        self.obs_dim = (
            self.self_dim
            + Q * self.belief_dim
            + (K - 1) * self.neighbor_dim
            + Q * self.detection_dim
        )
        if use_relative_features:
            self.obs_dim += Q * self.relative_dim
        if self.use_p0_global_info:
            self.obs_dim += 2          # global: n_pairs, n_targets_sensed
            self.obs_dim += Q           # per-target coverage count (from P0)
            self.obs_dim += (K - 1)     # per-neighbor pairing feasibility
        self.obs_dim += 16       # attention-aggregated neighbor comm messages
        self.obs_dim += 3        # explicit physics: nearest_target_dist(1) + bearing_sin+cos(2)

    def build_local_obs(
        self,
        agent_id: int,
        uav_states: List,       # List of UAVState for all K UAVs
        beliefs: List[List[BeliefState]],  # [K][Q] beliefs
        prev_P_D: Optional[np.ndarray] = None,  # (Q,) previous frame P_D
        oracle_targets: Optional[np.ndarray] = None,  # (Q, 4) true [px,py,vx,vy]; None=use beliefs
        selected_set: Optional[list] = None,  # P0 selected (i,j,q) triples
        deflection_entries: Optional[list] = None,  # for pairing feasibility
        comm_msgs: Optional[Dict[int, np.ndarray]] = None,  # per-agent comm messages
    ) -> np.ndarray:
        """Build local observation for one agent.

        Args:
            agent_id: Index of this UAV
            uav_states: List of UAVState for all UAVs
            beliefs: Per-UAV, per-target belief states
            prev_P_D: Previous frame detection probabilities (Q,)
            selected_set: P0 selected bistatic pairs [(tx,rx,target), ...]

        Returns:
            Flat observation vector
        """
        obs_parts = []
        selected_set = selected_set or []

        # --- Self state ---
        self_state = uav_states[agent_id]
        # Normalize position to [0, 1]
        obs_parts.append(self_state.pos / np.array([self.area_w, self.area_h, self.height]))
        obs_parts.append(self_state.vel / 25.0)  # normalize by v_max
        obs_parts.append([self_state.battery / 50000.0])  # normalize by B_max
        obs_parts.append([float(self_state.role)])  # already in {0,1,2}

        # --- Beliefs (per target) or oracle ---
        use_oracle = oracle_targets is not None
        for q in range(self.Q):
            if use_oracle:
                # Oracle: true target state, zero covariance, zero AoI
                tgt = oracle_targets[q]  # [px, py, vx, vy]
                oracle_mean = np.array([tgt[0], tgt[1], tgt[2], tgt[3]])
                obs_parts.append(oracle_mean / np.array([self.area_w, self.area_h, 25.0, 25.0]))
                obs_parts.append(np.zeros(4))  # zero covariance
                obs_parts.append([0.0])  # zero AoI
            else:
                b = beliefs[agent_id][q]
                obs_parts.append(b.mean / np.array([self.area_w, self.area_h, 25.0, 25.0]))
                obs_parts.append(b.cov_diag / np.array([self.area_w**2, self.area_h**2, 625.0, 625.0]))
                obs_parts.append([float(b.aoi) / 100.0])  # normalize AoI

        # --- Explicit relative geometry (per target) ---
        if self.use_relative_features:
            my_pos = self_state.pos[:2]  # (2,) UAV x,y
            diag = np.sqrt(self.area_w**2 + self.area_h**2)
            for q in range(self.Q):
                if use_oracle:
                    tgt_pos = oracle_targets[q, :2]
                else:
                    tgt_pos = beliefs[agent_id][q].mean[:2]
                dx = (tgt_pos[0] - my_pos[0]) / self.area_w
                dy = (tgt_pos[1] - my_pos[1]) / self.area_h
                dist_raw = np.linalg.norm(tgt_pos - my_pos)
                dist = dist_raw / diag
                # Positional encoding: bearing as (sin, cos) pair (not single [-1,1])
                angle = np.arctan2(dy * self.area_h, dx * self.area_w)
                sin_b = np.sin(angle)
                cos_b = np.cos(angle)
                # Multi-scale distance: exp(-d / scale) at 3 scales
                d_s1 = np.exp(-dist_raw / 50.0)
                d_s2 = np.exp(-dist_raw / 150.0)
                d_s3 = np.exp(-dist_raw / 400.0)
                obs_parts.append([dx, dy, dist, sin_b, cos_b, d_s1, d_s2, d_s3])

        # --- Explicit physical features: nearest-target distance + bearing ---
        # Gives Actor direct knowledge of the dominant physical relationship
        # (distance-to-target is the #1 determinant of P_D per R^{-4} radar eq.)
        my_pos = self_state.pos[:2]
        nearest_q = 0
        nearest_d = float('inf')
        for q in range(self.Q):
            if use_oracle:
                tgt_p = oracle_targets[q, :2]
            else:
                tgt_p = beliefs[agent_id][q].mean[:2]
            d_q = float(np.linalg.norm(tgt_p - my_pos))
            if d_q < nearest_d:
                nearest_d = d_q
                nearest_q = q
        if use_oracle:
            nearest_pos = oracle_targets[nearest_q, :2]
        else:
            nearest_pos = beliefs[agent_id][nearest_q].mean[:2]
        d_nearest_norm = nearest_d / diag  # normalized distance
        bearing = np.arctan2(nearest_pos[1] - my_pos[1], nearest_pos[0] - my_pos[0])
        # Nearest target features: 3 dims (dist, sin_bearing, cos_bearing)
        obs_parts.append([d_nearest_norm, np.sin(bearing), np.cos(bearing)])

        # --- Rich neighbor intent features ---
        # Pre-compute each neighbor's nearest target (intent proxy)
        neighbor_target_dirs = {}  # k -> (dx,dy) normalized toward nearest target
        for k in range(self.K):
            if k == agent_id: continue
            n_pos = uav_states[k].pos[:2]
            # Find nearest target to this neighbor (from belief or oracle)
            best_d, best_dir = float('inf'), np.zeros(2)
            for q in range(self.Q):
                if use_oracle:
                    tgt_p = oracle_targets[q, :2]
                else:
                    tgt_p = beliefs[k][q].mean[:2]
                d = np.linalg.norm(n_pos - tgt_p)
                if d < best_d:
                    best_d = d
                    best_dir = (tgt_p - n_pos) / max(d, 1e-6)
            neighbor_target_dirs[k] = best_dir
        # P0 pairs: which UAVs are in active pairs this frame
        paired_uavs = set()
        for (i, j, q) in selected_set:
            paired_uavs.add(i); paired_uavs.add(j)
        n_pairs = len(selected_set)
        n_targets_sensed = len(set(q for (_, _, q) in selected_set))

        # --- SINR-gated P0 info: only available when comm link is good ---
        # If SINR < threshold, P0 features are zeroed (forced to rely on physics)
        comm_available = True  # default: comm is available
        if self.use_p0_global_info and uav_states is not None:
            try:
                from uav_isac.physical.channel import compute_comm_sinr_db
                fc_pos = getattr(self, '_fc_position', None)
                if fc_pos is not None:
                    all_pos = np.array([s.pos for s in uav_states])
                    all_roles = np.array([s.role for s in uav_states])
                    # SINR from agent to fusion center
                    noise_pwr = 4e-21 * 1e6 * 10**(4.0/10)  # kT*B*NF
                    sinr_db = compute_comm_sinr_db(
                        self_state.pos, fc_pos, all_pos, all_roles,
                        2.8e10, 0.25, noise_pwr)
                    comm_available = sinr_db >= 6.0  # QPSK demod threshold
            except ImportError:
                pass  # fallback: always available

        # --- P0 global info (optional): target coverage + pairing feasibility ---
        if self.use_p0_global_info:
            coverage = np.zeros(self.Q, dtype=np.float64)
            if comm_available:
                for (_, _, q) in selected_set:
                    if q < self.Q:
                        coverage[q] += 1.0
                coverage /= max(1.0, self.K / 2.0)
            obs_parts.append(coverage)                                                 # Q dims

            pairing_with = np.zeros(self.K - 1, dtype=np.float64)
            if comm_available and deflection_entries:
                neighbor_idx = 0
                for k in range(self.K):
                    if k == agent_id: continue
                    can_pair = 0.0
                    for e in deflection_entries:
                        if e.d_eff > 0 and e.i == agent_id and e.j == k:
                            can_pair = 1.0; break
                    pairing_with[neighbor_idx] = can_pair
                    neighbor_idx += 1
            obs_parts.append(pairing_with)                                              # K-1 dims

        for k in range(self.K):
            if k == agent_id: continue
            n_state = uav_states[k]
            rel_pos = (n_state.pos[:2] - self_state.pos[:2]) / np.array([self.area_w, self.area_h])
            rel_vel = (n_state.vel[:2] - self_state.vel[:2]) / 25.0
            obs_parts.append(rel_pos)                          # 2
            obs_parts.append(rel_vel)                          # 2
            obs_parts.append([float(n_state.role)])            # 1
            obs_parts.append(neighbor_target_dirs.get(k, np.zeros(2)))  # 2: heading intent
            if self.use_p0_global_info:
                in_pair_val = 1.0 if (k in paired_uavs and comm_available) else 0.0
                obs_parts.append([in_pair_val])   # 1: actively sensing (SINR-gated)
            n_tgt_d = min(np.linalg.norm(n_state.pos[:2] - (beliefs[k][q].mean[:2] if not use_oracle else oracle_targets[q,:2])) for q in range(self.Q))
            obs_parts.append([n_tgt_d / np.sqrt(self.area_w**2+self.area_h**2)])  # 1: dist to nearest target

        # Global coordination summary (P0, SINR-gated)
        if self.use_p0_global_info:
            obs_parts.append([float(n_pairs) / self.K if comm_available else 0.0])
            obs_parts.append([float(n_targets_sensed) / max(self.Q,1) if comm_available else 0.0])

        # --- Previous detection status ---
        if prev_P_D is not None:
            obs_parts.append(prev_P_D)
        else:
            obs_parts.append(np.zeros(self.Q, dtype=np.float64))

        # --- Distance-weighted neighbor communication messages ---
        if comm_msgs is not None and len(comm_msgs) > 0:
            my_p = self_state.pos[:2]
            weights = []
            msgs = []
            for k in range(self.K):
                if k == agent_id: continue
                if k in comm_msgs:
                    other_p = uav_states[k].pos[:2]
                    d = np.linalg.norm(my_p - other_p) / 200.0  # distance weight
                    w = np.exp(-d)  # closer = higher weight
                    weights.append(w)
                    msgs.append(comm_msgs[k])
            if msgs:
                w_arr = np.array(weights)
                w_arr = w_arr / (w_arr.sum() + 1e-6)
                agg_msg = np.sum([w * m for w, m in zip(w_arr, msgs)], axis=0)  # (16,)
            else:
                agg_msg = np.zeros(16)
        else:
            agg_msg = np.zeros(16)
        obs_parts.append(agg_msg)  # 16 dims

        obs = np.concatenate([np.atleast_1d(p).ravel() for p in obs_parts])
        return obs.astype(np.float64)

    def build_global_state(
        self,
        uav_states: List,
        target_positions: np.ndarray,   # (Q, 3)
        target_velocities: np.ndarray,  # (Q, 3)
        prev_P_D: Optional[np.ndarray] = None,
        time_frac: float = 0.0,
        belief_cov_trace: Optional[np.ndarray] = None,  # (Q,) per-target mean cov trace
    ) -> np.ndarray:
        """Build global state for the centralized critic.

        Includes full information about all UAVs and targets.

        Args:
            uav_states: List of UAVState for all UAVs
            target_positions: True target positions (Q, 3)
            target_velocities: True target velocities (Q, 3)
            prev_P_D: Previous frame P_D (Q,)

        Returns:
            Flat global state vector
        """
        parts = []

        # All UAV states stacked
        for state in uav_states:
            parts.append(state.pos / np.array([self.area_w, self.area_h, self.height]))
            parts.append(state.vel / 25.0)
            parts.append([state.battery / 50000.0])
            parts.append([float(state.role)])

        # All target states
        for q in range(self.Q):
            parts.append(target_positions[q] / np.array([self.area_w, self.area_h, 1.0]))
            parts.append(target_velocities[q] / 25.0)

        # Time-to-go: critic needs to know how close to termination we are
        parts.append([time_frac])

        # Aggregate belief uncertainty per target (mean covariance trace)
        if belief_cov_trace is not None:
            parts.append(belief_cov_trace / 2500.0)  # normalize by initial pos variance
        else:
            parts.append(np.ones(self.Q, dtype=np.float64))

        # Detection status
        if prev_P_D is not None:
            parts.append(prev_P_D)
        else:
            parts.append(np.zeros(self.Q, dtype=np.float64))

        global_state = np.concatenate([np.atleast_1d(p).ravel() for p in parts])
        return global_state.astype(np.float64)

    def get_obs_dim(self) -> int:
        """Return observation dimension."""
        return self.obs_dim

    def get_global_state_dim(self) -> int:
        """Return global state dimension."""
        per_uav = 3 + 3 + 1 + 1  # pos + vel + battery + role
        per_target = 3 + 3  # pos + vel
        return (self.K * per_uav + self.Q * per_target +
                1 + self.Q + self.Q)  # time_frac + belief_cov_trace(Q) + P_D(Q)
