"""Gymnasium Environment wrapper for the UAV-ISAC simulation."""

import numpy as np
from numbers import Integral
from typing import Any, Dict, Optional, Tuple, Union
import gymnasium
from gymnasium import spaces

from uav_isac.environment.env_core import EnvironmentCore
from uav_isac.utils.types import Action
from config.params import MasterConfig


class UAVISACEnv(gymnasium.Env):
    """Gymnasium environment for multi-UAV cooperative ISAC sensing.

    Multi-agent environment with:
    - Observation space: Dict of Box spaces (one per agent)
    - Action space: Dict of Dict spaces (one per agent: delta_p + role)
    - Reward: Dict of float (one per agent)

    Follows the Gymnasium PettingZoo-style multi-agent API.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        config: Optional[MasterConfig] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Args:
            config: MasterConfig; loads default if None
            seed: Random seed
            render_mode: "human" or "rgb_array" or None
        """
        super().__init__()

        if config is None:
            from config.params import get_default_config
            config = get_default_config()

        self.cfg = config
        self.render_mode = render_mode
        self.seed_val = seed if seed is not None else 42

        # Create RNG
        self.rng = np.random.default_rng(self.seed_val)

        # Create core
        self.core = EnvironmentCore(config, rng=self.rng)

        # Define spaces
        self.K = config.scenario.K
        self.Q = config.scenario.Q
        self.max_dp = config.uav.v_max * config.scenario.dt

        # Per-agent observation space
        # Audit 2026-08-17: use core.get_obs_dim() (accounts for history
        # stacking) instead of obs_builder.get_obs_dim() (single-frame only),
        # otherwise obs_history_frames>1 violates the declared Box shape.
        obs_dim = self.core.get_obs_dim()
        self.observation_space = spaces.Dict({
            str(k): spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_dim,), dtype=np.float64
            )
            for k in range(self.K)
        })

        # Per-agent action space: delta_p (2,) + role (1, discrete)
        self.action_space = spaces.Dict({
            str(k): spaces.Dict({
                'delta_p': spaces.Box(
                    low=-self.max_dp, high=self.max_dp,
                    shape=(2,), dtype=np.float64
                ),
                'role': spaces.Discrete(3),  # 0=tx, 1=rx, 2=idle
            })
            for k in range(self.K)
        })

        # State storage
        self.current_obs: Optional[Dict] = None
        self.current_step_info = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict, Dict]:
        """Reset environment.

        Args:
            seed: Random seed (overrides constructor seed)
            options: Additional options (unused)

        Returns:
            (observations, info)
        """
        if seed is not None:
            self.seed_val = seed
            self.rng = self.core.reseed(seed)

        obs, info = self.core.reset()
        self.current_obs = obs
        self.current_step_info = None

        # Convert to string keys for Gymnasium compatibility
        obs_str = {str(k): v for k, v in obs.items()}
        return obs_str, info

    def step(  # type: ignore[override]
        self,
        actions: Dict[str, Any],
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Execute one step.

        Args:
            actions: Dict mapping agent_id (str) to action dict
                     {'delta_p': np.ndarray (2,), 'role': int}

        Returns:
            (observations, rewards, terminated, truncated, info)
        """
        if not isinstance(actions, dict):
            raise ValueError("actions must be a mapping for every UAV")

        # Convert string keys to int and parse actions.  Validate the complete
        # batch before mutating simulator state: NaN/Inf otherwise bypass the
        # movement clamp and can make a corrupted episode look successful.
        actions_int = {}
        for raw_key, a in actions.items():
            if isinstance(raw_key, bool):
                raise ValueError("boolean UAV action keys are invalid")
            try:
                k = int(raw_key)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid UAV action key: {raw_key!r}") from exc
            if k in actions_int:
                raise ValueError(f"Duplicate UAV action key after parsing: {k}")
            if isinstance(a, dict):
                if set(a) != {"delta_p", "role"}:
                    raise ValueError(
                        f"Action for UAV {k} must contain exactly delta_p and role")
                try:
                    delta_p = np.asarray(a["delta_p"], dtype=np.float64)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"delta_p for UAV {k} must be numeric") from exc
                raw_role = a["role"]
                if isinstance(raw_role, bool) or not isinstance(
                    raw_role, Integral
                ):
                    raise ValueError(f"role for UAV {k} must be an integer")
                role = int(raw_role)
                actions_int[k] = Action(delta_p=delta_p, role=role)
            elif isinstance(a, Action):
                actions_int[k] = a
            else:
                raise ValueError(f"Invalid action type: {type(a)}")

        expected_agents = set(range(self.K))
        if set(actions_int) != expected_agents:
            missing = sorted(expected_agents.difference(actions_int))
            extra = sorted(set(actions_int).difference(expected_agents))
            raise ValueError(
                f"actions must cover every UAV exactly once; "
                f"missing={missing}, extra={extra}")

        # Step the core
        next_obs, rewards, dones, step_info = self.core.step(actions_int)

        # Convert to string keys
        obs_str = {str(k): v for k, v in next_obs.items()}
        rewards_str = {str(k): v for k, v in rewards.items()}

        # Determine termination
        terminated = {str(k): dones.get(k, False) for k in range(self.K)}
        truncated = {str(k): False for k in range(self.K)}  # no truncation
        terminated['__all__'] = dones.get('__all__', False)
        truncated['__all__'] = False

        # Build info
        info = {
            'frame': step_info.frame,
            'team_reward': step_info.team_reward,
            'P_D_q': step_info.P_D_q,
            'total_bits': step_info.p0_solution.total_bits,
            'constraint_info': step_info.constraint_info,
            'uav_positions': np.array([u.pos for u in self.core.uavs]),
            'target_positions': np.array([t.get_position_3d() for t in self.core.targets]),
            # ── Pairing diagnostics (P0) ──
            'roles': step_info.roles,
            'n_tx': step_info.n_tx,
            'n_rx': step_info.n_rx,
            'n_selected': step_info.n_selected,
            'p0_target_coverage': float(
                len({
                    int(q) for _, _, q
                    in step_info.p0_solution.selected_set
                }) / max(self.Q, 1)),
            'p0_target_selected_mask': np.asarray([
                any(int(edge_q) == q for _, _, edge_q
                    in step_info.p0_solution.selected_set)
                for q in range(self.Q)
            ], dtype=np.float64),
            'n_duplex': step_info.n_duplex,
            'multistatic_subslot_enabled': bool(
                self.core._multistatic_subslot_enabled),
            'valid_pair': step_info.valid_pair,
            'no_tx': step_info.no_tx,
            'all_same_role': step_info.all_same_role,
            'p0_resolved': step_info.p0_resolved,
            'p0_solve_time_s': step_info.p0_solve_time_s,
        }
        if step_info.learned_comm:
            info.update(step_info.learned_comm)
            info['total_bits_all'] = (
                float(step_info.p0_solution.total_bits)
                + float(step_info.learned_comm.get('learned_comm_bits', 0.0))
                + float(step_info.learned_comm.get('evidence_comm_bits', 0.0))
            )
        if step_info.reward_components:
            # Keep both the structured dictionary and flat scalar keys. The
            # latter flow directly into rollout CSVs without special parsing.
            info['reward_components'] = dict(step_info.reward_components)
            info.update({
                f'reward_{name}': float(value)
                for name, value in step_info.reward_components.items()
            })

        self.current_obs = next_obs
        self.current_step_info = step_info

        return obs_str, rewards_str, terminated, truncated, info

    def get_state(self) -> Dict:
        """Snapshot full env state for exact replay (P0). See EnvironmentCore.get_state."""
        return self.core.get_state()

    def set_state(self, state: Dict) -> None:
        """Restore a snapshot from get_state(). Keeps wrapper.rng aliased to core.rng."""
        self.core.set_state(state)
        # core.set_state restores the generator in place; keep the wrapper's
        # handle pointing at the same object.
        self.rng = self.core.rng

    def render(self) -> Optional[Union[str, np.ndarray]]:  # type: ignore[override]
        """Render the environment (text-based for now)."""
        if self.render_mode is None:
            return None

        if self.current_step_info is None:
            return None

        info = self.current_step_info
        lines = []
        lines.append(f"=== Frame {info.frame} ===")
        lines.append(f"Team Reward: {info.team_reward:.4f}")
        lines.append(f"P_D per target: {info.P_D_q}")

        for k, u in enumerate(self.core.uavs):
            # Audit 2026-08-17: P0 can assign role code 3 (dual TX+RX endpoint),
            # which is out of range for the 3-element list -> IndexError.
            role_str = ['tx', 'rx', 'idle', 'duplex'][u.role % 4]
            lines.append(
                f"  UAV{k}: pos=({u.pos[0]:.0f},{u.pos[1]:.0f}) "
                f"battery={u.battery:.0f}J role={role_str}"
            )

        lines.append("Targets:")
        for q, t in enumerate(self.core.targets):
            lines.append(
                f"  T{q}: pos=({t.state[0]:.0f},{t.state[1]:.0f}) "
                f"vel=({t.state[2]:.1f},{t.state[3]:.1f})"
            )

        text = "\n".join(lines)

        if self.render_mode == "human":
            print(text)
            return None
        elif self.render_mode == "rgb_array":
            return text  # placeholder

        return None

    def close(self):
        """Cleanup resources."""
        self.core.close()
