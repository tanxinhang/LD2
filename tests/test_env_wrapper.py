"""Gymnasium environment compliance and smoke tests."""

import numpy as np
import pytest
from uav_isac.environment.env_wrapper import UAVISACEnv
from config.params import get_default_config


class TestEnvCreation:
    def test_create_default(self):
        env = UAVISACEnv()
        assert env.K == 4
        assert env.Q == 2
        env.close()

    def test_create_small(self, small_config):
        env = UAVISACEnv(config=small_config)
        assert env.K == 2
        assert env.Q == 1
        env.close()


class TestReset:
    def test_reset_returns_valid_obs(self):
        env = UAVISACEnv()
        obs, info = env.reset(seed=42)

        # Check observation structure
        assert isinstance(obs, dict)
        assert len(obs) == 4  # K=4 agents
        for k in range(4):
            assert str(k) in obs
            assert isinstance(obs[str(k)], np.ndarray)
            assert np.all(np.isfinite(obs[str(k)]))

        assert 'uav_positions' in info
        assert 'target_positions' in info
        env.close()

    def test_reset_deterministic(self):
        """Same seed → same initial state."""
        env1 = UAVISACEnv()
        obs1, _ = env1.reset(seed=42)
        env2 = UAVISACEnv()
        obs2, _ = env2.reset(seed=42)

        for k in range(4):
            assert np.allclose(obs1[str(k)], obs2[str(k)])
        env1.close()
        env2.close()


class TestStep:
    def test_random_actions_no_crash(self):
        """Random policy rollout for 50 steps — no crashes."""
        env = UAVISACEnv()
        obs, _ = env.reset(seed=42)

        for _ in range(50):
            actions = {}
            for k_str in obs:
                k = int(k_str)
                delta_p = env.rng.uniform(-env.max_dp, env.max_dp, size=2)
                role = env.rng.integers(0, 3)
                actions[k_str] = {'delta_p': delta_p, 'role': role}

            obs, rewards, terminated, truncated, info = env.step(actions)

            # All observation values should be finite
            for k_str, o in obs.items():
                assert np.all(np.isfinite(o)), f"Non-finite obs for agent {k_str}"

            # Rewards should be finite
            for k_str, r in rewards.items():
                assert np.isfinite(r), f"Non-finite reward for agent {k_str}"

            # P_D in [0, 1]
            assert np.all(info['P_D_q'] >= 0)
            assert np.all(info['P_D_q'] <= 1)

            if terminated.get('__all__', False):
                break

        env.close()

    def test_full_episode_completes(self):
        """A full 100-frame episode should complete without errors."""
        cfg = get_default_config()
        cfg.scenario.T = 100
        env = UAVISACEnv(config=cfg)
        obs, _ = env.reset(seed=123)

        frames = 0
        while True:
            actions = {}
            for k_str in obs:
                k = int(k_str)
                delta_p = env.rng.uniform(-env.max_dp, env.max_dp, size=2)
                role = env.rng.integers(0, 3)
                actions[k_str] = {'delta_p': delta_p, 'role': role}

            obs, rewards, terminated, truncated, info = env.step(actions)
            frames += 1

            if terminated.get('__all__', False):
                break

        assert frames <= 100
        env.close()


class TestEnergy:
    def test_energy_decreases(self):
        """Battery energy should monotonically decrease."""
        env = UAVISACEnv()
        obs, _ = env.reset(seed=42)

        prev_batteries = [u.battery for u in env.core.uavs]

        for step_idx in range(10):
            actions = {}
            for k_str in obs:
                k = int(k_str)
                actions[k_str] = {
                    'delta_p': env.rng.uniform(-env.max_dp, env.max_dp, size=2),
                    'role': env.rng.integers(0, 3),
                }
            obs, rewards, terminated, truncated, info = env.step(actions)

            curr_batteries = [u.battery for u in env.core.uavs]
            for k in range(env.K):
                assert curr_batteries[k] <= prev_batteries[k] + 1e-10, \
                    f"Battery increased for UAV {k}"
            prev_batteries = curr_batteries

            if terminated.get('__all__', False):
                break

        env.close()
