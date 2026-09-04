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
        assert np.allclose(env.core.reward_computer.omega_q, [1.0])
        env.close()

    def test_programmatic_config_type_bypass_is_rejected(self):
        cfg = get_default_config()
        cfg.marl.tracking_enabled = "false"
        with pytest.raises(ValueError, match="tracking_enabled must be a boolean"):
            UAVISACEnv(config=cfg)

    def test_programmatic_config_semantic_bypass_is_rejected(self):
        cfg = get_default_config()
        cfg.scenario.K = 0
        with pytest.raises(
            ValueError, match=r"config\.scenario\.K must be greater than zero"
        ):
            UAVISACEnv(config=cfg)


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
    def test_step_before_reset_fails_without_advancing_time(self):
        env = UAVISACEnv()
        actions = {
            str(k): {"delta_p": np.zeros(2), "role": 2}
            for k in range(env.K)
        }
        assert env.core.t == 0
        with pytest.raises(
            RuntimeError, match="environment must be reset before the first step"
        ):
            env.step(actions)
        assert env.core.t == 0
        env.close()

    @pytest.mark.parametrize(
        "defect",
        ["nan", "inf", "shape", "role_range", "role_float", "missing"],
    )
    def test_invalid_action_batch_fails_before_state_mutation(self, defect):
        env = UAVISACEnv()
        env.reset(seed=42)
        actions = {
            str(k): {"delta_p": np.zeros(2), "role": 2}
            for k in range(env.K)
        }
        if defect == "nan":
            actions["0"]["delta_p"] = np.array([np.nan, 0.0])
        elif defect == "inf":
            actions["0"]["delta_p"] = np.array([np.inf, 0.0])
        elif defect == "shape":
            actions["0"]["delta_p"] = np.zeros(3)
        elif defect == "role_range":
            actions["0"]["role"] = 3
        elif defect == "role_float":
            actions["0"]["role"] = 1.5
        else:
            del actions["0"]

        before_t = env.core.t
        before_positions = np.array([u.pos.copy() for u in env.core.uavs])
        with pytest.raises(ValueError):
            env.step(actions)
        assert env.core.t == before_t
        assert np.array_equal(
            np.array([u.pos for u in env.core.uavs]), before_positions)
        env.close()

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
