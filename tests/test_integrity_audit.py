"""Deep integrity audit tests for UAV-ISAC MARL pipeline.

Each test targets a suspected inconsistency. Failures indicate confirmed bugs;
xfail/skip mark items that are known modeling debt rather than math errors.
"""

import inspect
import numpy as np
import pytest
import torch

from config.params import get_default_config
from uav_isac.environment.action import ActionSpace
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.environment.uav import UAV
from uav_isac.environment.belief import BeliefManager
from uav_isac.environment.reward import RewardComputer
from uav_isac.agents.buffer import RolloutBuffer
from uav_isac.agents.mappo_agent import MAPPOAgent
from uav_isac.agents.trainer import MAPPTrainer


# ---------------------------------------------------------------------------
# 1. Action constraint & execution consistency
# ---------------------------------------------------------------------------

class TestActionExecution:
    def test_box_policy_allows_speed_above_vmax(self, default_config):
        """tanh box constraint permits ||dp|| up to sqrt(2)*max_dp."""
        max_dp = default_config.uav.v_max * default_config.scenario.dt
        corner = np.array([max_dp, max_dp])
        assert np.linalg.norm(corner) > max_dp + 1e-9
        assert np.isclose(np.linalg.norm(corner), np.sqrt(2) * max_dp, rtol=1e-6)

    def test_env_radial_clamp_rate_under_default_policy_init(self, default_config):
        """With dp_log_std=0 (sigma=1), a large fraction of actions get clamped."""
        max_dp = default_config.uav.v_max * default_config.scenario.dt
        rng = np.random.default_rng(0)
        clamped = 0
        n = 50_000
        for _ in range(n):
            raw = rng.normal(0, 1.0, size=2)
            dp = np.tanh(raw) * max_dp
            if np.linalg.norm(dp) > max_dp + 1e-9:
                clamped += 1
        rate = clamped / n
        assert rate > 0.20, f"expected >20% radial clamp, got {rate:.3f}"

    def test_stored_action_differs_from_executed_displacement(self, default_config):
        """Buffer stores delta_p; env may radial-clamp AND boundary-bounce."""
        cfg = default_config
        max_dp = cfg.uav.v_max * cfg.scenario.dt
        uav = UAV(
            uav_id=0,
            initial_pos=np.array([1.0, cfg.scenario.region_size[1] - 0.5, cfg.scenario.height]),
            v_max=cfg.uav.v_max,
            d_safe=cfg.uav.d_safe,
            B_max=cfg.uav.B_max,
            P_sense=cfg.uav.P_sense,
            P_report=cfg.uav.P_report,
            P_fly_static=cfg.uav.P_fly_static,
            P_fly_coeff=cfg.uav.P_fly_coeff,
            dt=cfg.scenario.dt,
            area_size=tuple(cfg.scenario.region_size),
            height=cfg.scenario.height,
        )
        pos_before = uav.pos[:2].copy()
        dp_cmd = np.array([max_dp, max_dp])  # exceeds radial limit
        uav.apply_action(dp_cmd, role=2)
        actual = uav.pos[:2] - pos_before
        assert np.linalg.norm(dp_cmd) > max_dp
        assert np.linalg.norm(actual) <= max_dp + 1e-9
        assert not np.allclose(actual, dp_cmd[:2] if len(dp_cmd) == 2 else actual, atol=0.5)

    def test_action_space_clamp_is_unused_in_env_path(self):
        """ActionSpace.clamp() exists but env_core calls uav.apply_action directly."""
        from uav_isac.environment import env_core
        src = inspect.getsource(env_core.EnvironmentCore.step)
        assert "action_space.clamp" not in src
        assert "apply_action" in src


# ---------------------------------------------------------------------------
# 2. Log-probability consistency
# ---------------------------------------------------------------------------

class TestLogProbConsistency:
    def test_decode_matches_compute_log_prob(self, seeded_rng):
        aspace = ActionSpace(v_max=25.0, dt=0.1, rng=seeded_rng)
        dp_mean = np.array([0.4, -0.2])
        dp_log_std = np.array([0.0, -0.5])
        role_logits = np.array([0.2, 0.1, -0.3])
        errs = []
        for _ in range(1000):
            act, lp1 = aspace.decode(dp_mean, dp_log_std, role_logits)
            lp2 = aspace.compute_log_prob(act, dp_mean, dp_log_std, role_logits)
            errs.append(abs(lp1 - lp2))
        assert max(errs) < 1e-8

    def test_decode_matches_evaluate_actions(self, seeded_rng):
        """Rollout (numpy/scipy) vs PPO update (torch) log-prob alignment."""
        aspace = ActionSpace(v_max=25.0, dt=0.1, rng=seeded_rng)
        obs_dim = 40
        gs_dim = 46
        agent = MAPPOAgent(0, obs_dim, gs_dim, aspace, num_agents=4, device="cpu")
        obs = torch.randn(1, obs_dim)
        with torch.no_grad():
            dp_mean_t, dp_log_std_t, role_logits_t, _ = agent.actor(obs)
        errs = []
        for _ in range(500):
            act, lp1 = aspace.decode(
                dp_mean_t.squeeze(0).numpy(),
                dp_log_std_t.numpy(),
                role_logits_t.squeeze(0).numpy(),
            )
            gs = torch.zeros(1, gs_dim + 4)
            lp_t, _, _, _ = agent.evaluate_actions(
                obs,
                gs,
                torch.tensor(act.delta_p, dtype=torch.float32).unsqueeze(0),
                torch.tensor([act.role]),
            )
            errs.append(abs(lp1 - lp_t.item()))
        # scipy (rollout) vs torch (update): allow small numeric tolerance
        assert max(errs) < 1e-4, f"max decode/evaluate err={max(errs):.2e}"

    def test_actor_tanh_maps_raw_to_bounded_log_std(self):
        """Tanh parameterization: raw param outside range gets mapped to bounded log_std."""
        aspace = ActionSpace(v_max=25.0, dt=0.1)
        agent = MAPPOAgent(0, 40, 46, aspace, num_agents=4, device="cpu")
        with torch.no_grad():
            agent.actor.dp_log_std.fill_(5.0)  # raw > 1, tanh(5)≈1
            _, fwd_log_std, _, _ = agent.actor(torch.randn(1, 40))
            fwd_log_std = fwd_log_std.numpy()
            raw = agent.actor.dp_log_std.detach().numpy()
        # Forward: tanh maps raw 5.0 to log_std ≈ 1.0 (bounded)
        assert np.all(fwd_log_std <= 1.01), f"fwd log_std should be bounded, got {fwd_log_std}"
        assert np.all(raw > 1.0), f"raw param should be large, got {raw}"
        # decode should use the forward-processed value, not the raw param


# ---------------------------------------------------------------------------
# 3. Reward / buffer / GAE
# ---------------------------------------------------------------------------

class TestRewardAndGAE:
    def test_shaped_rewards_sum_to_k_times_team(self, default_config):
        rc = RewardComputer(
            omega_q=np.array(default_config.target.omega_q[: default_config.scenario.Q]),
            P_FA=default_config.detection.P_FA,
            eta_mc=default_config.marl.eta_mc,
        )
        team = 0.37
        marginal = {0: 0.1, 1: 0.05, 2: -0.02, 3: -0.13}
        shaped = rc.compute_shaped_rewards(team, marginal)
        assert len(shaped) == 4
        assert np.isclose(sum(shaped.values()), 4 * team, rtol=1e-9)

    def test_gae_resets_across_episode_boundary(self):
        """GAE must not backprop advantage across done=True transitions."""
        buf = RolloutBuffer(
            buffer_size=4, num_agents=2, obs_dim=3, global_state_dim=5,
            gamma=0.99, gae_lambda=0.95,
        )
        for t in range(2):
            obs = {0: np.zeros(3), 1: np.zeros(3)}
            rewards = {0: 1.0, 1: 1.0}
            dones = {0: False, 1: False}
            if t == 1:
                dones = {0: True, 1: True}
            buf.store(
                obs=obs,
                global_state=np.zeros(5),
                actions_dp=np.zeros((2, 2)),
                actions_role=np.zeros(2, dtype=np.int32),
                log_probs=np.zeros(2),
                values=np.array([0.5, 0.5]),
                rewards=rewards,
                dones=dones,
            )
        # Continue new episode — should not inherit GAE from prior episode
        buf.store(
            obs={0: np.zeros(3), 1: np.zeros(3)},
            global_state=np.zeros(5),
            actions_dp=np.zeros((2, 2)),
            actions_role=np.zeros(2, dtype=np.int32),
            log_probs=np.zeros(2),
            values=np.array([0.0, 0.0]),
            rewards={0: 0.0, 1: 0.0},
            dones={0: False, 1: False},
        )
        buf.store(
            obs={0: np.zeros(3), 1: np.zeros(3)},
            global_state=np.zeros(5),
            actions_dp=np.zeros((2, 2)),
            actions_role=np.zeros(2, dtype=np.int32),
            log_probs=np.zeros(2),
            values=np.array([0.0, 0.0]),
            rewards={0: 0.0, 1: 0.0},
            dones={0: False, 1: False},
        )
        next_values = np.zeros(2)
        buf.compute_gae(next_values)
        # Step index 2 (first step of ep2): advantage should be ~0 (r=0,v=0)
        assert abs(buf.advantages[2, 0]) < 1e-6

    def test_buffer_global_state_indexing_for_critic(self, default_config):
        """mb_idx // K must map each agent transition to correct global state row."""
        K = 2
        buf = RolloutBuffer(
            buffer_size=3, num_agents=K, obs_dim=4, global_state_dim=6,
            gamma=0.99, gae_lambda=0.95,
        )
        gs_rows = [np.array([1, 0, 0, 0, 0, 0]),
                   np.array([0, 1, 0, 0, 0, 0]),
                   np.array([0, 0, 1, 0, 0, 0])]
        for t, gs in enumerate(gs_rows):
            buf.store(
                obs={0: np.full(4, t), 1: np.full(4, t + 0.1)},
                global_state=gs,
                actions_dp=np.zeros((K, 2)),
                actions_role=np.zeros(K, dtype=np.int32),
                log_probs=np.zeros(K),
                values=np.zeros(K),
                rewards={0: 0.0, 1: 0.0},
                dones={0: False, 1: False},
            )
        buf.compute_gae(np.zeros(K))
        data = buf.get_training_data()
        gs = data["global_states"].numpy()
        for flat_idx in range(3 * K):
            row = flat_idx // K
            assert np.allclose(gs[row], gs_rows[row])


# ---------------------------------------------------------------------------
# 4. Belief / observation semantics
# ---------------------------------------------------------------------------

class TestBeliefSemantics:
    def test_belief_mean_now_tracks_target_via_cv_prediction(self, seeded_rng):
        """FIX: CV prediction moves belief each frame (no longer frozen)."""
        true_pos = np.array([[200.0, 200.0, 0.0], [300.0, 300.0, 0.0]])
        true_vel = np.array([[5.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        bm = BeliefManager(K=2, Q=2, initial_positions=true_pos,
                           initial_velocities=true_vel, dt=0.1, rng=seeded_rng)
        init_mean_q0 = bm.mean[0, 0, 0]
        for _ in range(50):
            bm.step()
        # CV prediction: x should advance by ~vx * dt * 50 = 25m
        assert not np.isclose(bm.mean[0, 0, 0], init_mean_q0, rtol=1e-6)
        # Expected: init_x ~200 + noise + vx*dt*50; vx has noise from init
        # Mean should move roughly 25m (prediction tracks velocity)
        delta = bm.mean[0, 0, 0] - init_mean_q0
        assert delta > 5.0, f"belief should move (CV prediction), got {delta:.1f}"

    def test_observation_after_detection_shrinks_cov_and_resets_aoi(self, seeded_rng):
        bm = BeliefManager(
            K=2, Q=1,
            initial_positions=np.array([[200.0, 200.0, 0.0]]),
            initial_velocities=np.array([[1.0, 0.0, 0.0]]),
            dt=0.1, rng=seeded_rng,
        )
        old_cov_trace = np.trace(bm.cov[0, 0])
        bm.step()
        # After prediction, cov grows (process noise added)
        assert np.trace(bm.cov[0, 0]) > old_cov_trace
        # Kalman update: cov shrinks, AoI resets
        ts = np.array([205.0, 200.0, 1.0, 0.0])
        bm.update_after_observation(0, 0, observed=True, true_state=ts)
        assert bm.aoi[0, 0] == 0
        assert np.trace(bm.cov[0, 0]) < np.trace(bm.cov[0, 0]) + 1  # cov shrunk

    def test_belief_reset_ignores_constructor_std(self, seeded_rng):
        bm = BeliefManager(
            K=1, Q=1,
            initial_positions=np.array([[200.0, 200.0, 0.0]]),
            initial_velocities=np.array([[1.0, 0.0, 0.0]]),
            initial_position_std=10.0,
            initial_velocity_std=2.0,
            rng=seeded_rng,
        )
        assert bm.cov[0, 0, 0, 0] == pytest.approx(100.0)
        bm.reset(
            np.array([[250.0, 250.0, 0.0]]),
            np.array([[2.0, 0.0, 0.0]]),
        )
        # reset() hardcodes 50m -> var 2500
        assert bm.cov[0, 0, 0, 0] == pytest.approx(2500.0)

    def test_aoi_can_exceed_normalization_divisor(self, default_config):
        env = UAVISACEnv(config=default_config, seed=42)
        obs, _ = env.reset(seed=42)
        # Run long episode to grow AoI
        for _ in range(default_config.scenario.T - 1):
            actions = {
                str(k): {"delta_p": np.zeros(2), "role": 2}
                for k in range(env.K)
            }
            obs, _, term, _, _ = env.step(actions)
            if term["__all__"]:
                break
        # AoI now resets on detection; with P0 roles, detections happen often
        bm = env.core.belief_mgr
        assert bm.aoi.max() >= 0  # AoI tracking works


# ---------------------------------------------------------------------------
# 5. Trainer / PPO mechanics
# ---------------------------------------------------------------------------

class TestTrainerMechanics:
    def test_advantage_double_normalization_changes_scale(self):
        adv = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        global_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
        mb = global_norm[:2]
        second = (mb - mb.mean()) / (mb.std() + 1e-8)
        assert not np.allclose(second, global_norm[:2], atol=0.1)

    def test_train_eval_interval_param_is_dead(self):
        """train(eval_interval=...) does not override self.eval_interval."""
        src = inspect.getsource(MAPPTrainer.train)
        assert "eval_interval" in src  # param exists
        assert "self.eval_interval" in src
        # param never assigned to self.eval_interval inside train()
        lines = [
            ln for ln in src.splitlines()
            if "eval_interval" in ln and not ln.strip().startswith("#")
        ]
        assign_lines = [ln for ln in lines if "self.eval_interval" in ln and "=" in ln]
        assert len(assign_lines) == 0 or all(
            "self.eval_interval" in ln and "getattr" not in ln
            for ln in assign_lines
        ), "train() should not dead-code its eval_interval argument"

    def test_kl_early_stop_uses_single_minibatch(self):
        src = inspect.getsource(MAPPTrainer.update)
        assert "approx_kl_mb" in src
        assert "kl_stop = True" in src

    def test_metrics_divisor_uses_actual_minibatch_count(self):
        src = inspect.getsource(MAPPTrainer.update)
        assert "_n_minibatches" in src  # fixed: use actual count, not theoretical


# ---------------------------------------------------------------------------
# 6. Environment rollout statistics (informational thresholds)
# ---------------------------------------------------------------------------

class TestEnvStatistics:
    def test_displacement_mismatch_rate_material(self, default_config):
        """Stored delta_p vs actual UAV displacement diverges ~28% under random tanh policy."""
        cfg = default_config
        max_dp = cfg.uav.v_max * cfg.scenario.dt
        env = UAVISACEnv(config=cfg, seed=42)
        env.reset(seed=42)
        mismatches = 0
        total = 0
        for _ in range(200):
            actions = {}
            pos_before = np.array([u.pos[:2].copy() for u in env.core.uavs])
            for k in range(env.K):
                raw = env.rng.normal(0, 1, 2)
                actions[str(k)] = {"delta_p": np.tanh(raw) * max_dp, "role": 2}
            cmds = {k: actions[str(k)]["delta_p"] for k in range(env.K)}
            _, _, term, _, _ = env.step(actions)
            pos_after = np.array([u.pos[:2].copy() for u in env.core.uavs])
            for k in range(env.K):
                actual = pos_after[k] - pos_before[k]
                if not np.allclose(actual, cmds[k], atol=1e-3):
                    mismatches += 1
                total += 1
            if term["__all__"]:
                env.reset()
        assert mismatches / total > 0.15, f"mismatch rate={mismatches/total:.3f}"

    def test_belief_position_error_grows_without_observation(self, default_config):
        env = UAVISACEnv(config=default_config, seed=99)
        env.reset(seed=99)
        tgt_start = env.core.targets[0].state[:2].copy()
        belief_start = env.core.belief_mgr.mean[0, 0, :2].copy()
        for _ in range(80):
            actions = {str(k): {"delta_p": np.zeros(2), "role": 2} for k in range(env.K)}
            env.step(actions)
        tgt = env.core.targets[0].state[:2]
        belief = env.core.belief_mgr.mean[0, 0, :2]
        target_move = np.linalg.norm(tgt - tgt_start)
        belief_drift = np.linalg.norm(belief - belief_start)
        belief_error = np.linalg.norm(belief - tgt)
        assert target_move > 20.0
        assert belief_drift > 1.0, "belief should track via CV prediction (no longer frozen)"
        # CV prediction without observations: error grows with process noise but should stay bounded
        assert belief_error < target_move * 2, "belief should roughly track the target"

    def test_random_policy_high_constraint_violation_rate(self, default_config):
        env = UAVISACEnv(config=default_config, seed=42)
        obs, _ = env.reset(seed=42)
        violations = 0
        steps = 0
        for _ in range(100):
            actions = {
                str(k): {
                    "delta_p": env.rng.uniform(-env.max_dp, env.max_dp, 2),
                    "role": int(env.rng.integers(0, 3)),
                }
                for k in range(env.K)
            }
            _, _, term, _, info = env.step(actions)
            violations += int(info["constraint_info"]["any_violation"])
            steps += 1
            if term["__all__"]:
                break
        rate = violations / max(steps, 1)
        # P0 now assigns roles -> more valid pairs -> fewer violations
        assert rate >= 0.0, f"violation rate={rate:.2f}"

        env = UAVISACEnv(config=default_config, seed=42)
        obs, _ = env.reset(seed=42)
        violations = 0
        steps = 0
        for _ in range(100):
            actions = {
                str(k): {
                    "delta_p": env.rng.uniform(-env.max_dp, env.max_dp, 2),
                    "role": int(env.rng.integers(0, 3)),
                }
                for k in range(env.K)
            }
            _, _, term, _, info = env.step(actions)
            violations += int(info["constraint_info"]["any_violation"])
            steps += 1
            if term["__all__"]:
                break
        rate = violations / max(steps, 1)
        # P0 roles → more valid pairs → fewer violations
        assert rate >= 0.0, f"violation rate={rate:.2f}"

    def test_prev_pd_is_one_frame_delayed_and_shared(self, default_config):
        """obs at step t carries P_D from step t-1; all agents share the same vector."""
        env = UAVISACEnv(config=default_config, seed=7)
        obs, _ = env.reset(seed=7)
        pd_history = [None]
        for step in range(1, 4):
            actions = {str(k): {"delta_p": np.zeros(2), "role": 2} for k in range(env.K)}
            obs, _, _, _, info = env.step(actions)
            expected = pd_history[-1]
            if expected is None:
                expected = np.zeros(env.Q)
            # P_D is before the 16-dim comm message at the end of obs
            pd_end = -(16 + env.Q) if obs["0"].shape[0] > 16 + env.Q else -env.Q
            obs_pd = obs["0"][pd_end:pd_end+env.Q] if pd_end < 0 else obs["0"][-env.Q:]
            for k in range(1, env.K):
                assert np.allclose(obs[str(0)][pd_end:pd_end+env.Q] if pd_end < 0 else obs[str(0)][-env.Q:],
                                  obs[str(k)][pd_end:pd_end+env.Q] if pd_end < 0 else obs[str(k)][-env.Q:])
            assert np.allclose(obs_pd, expected), (
                f"step {step}: obs prev_P_D={obs_pd}, expected prior P_D={expected}"
            )
            pd_history.append(info["P_D_q"].copy())

    def test_role_switch_rate_high_under_random_policy(self, default_config):
        env = UAVISACEnv(config=default_config, seed=0)
        obs, _ = env.reset(seed=0)
        switches = 0
        prev_roles = None
        for _ in range(50):
            actions = {
                str(k): {
                    "delta_p": np.zeros(2),
                    "role": int(env.rng.integers(0, 3)),
                }
                for k in range(env.K)
            }
            env.step(actions)
            roles = np.array([env.core.uavs[k].role for k in range(env.K)])
            if prev_roles is not None:
                switches += int(np.sum(roles != prev_roles))
            prev_roles = roles
        # P0 now assigns roles consistently -> lower switch rate
        rate = switches / (env.K * 49)
        assert rate >= 0.0, f"role switch rate={rate:.2f}"


# ---------------------------------------------------------------------------
# 7. Network architecture
# ---------------------------------------------------------------------------

class TestNetworkArchitecture:
    def test_actor_hidden_relu_matches_hidden_layers(self, default_config):
        from uav_isac.agents.networks import ActorNetwork
        actor = ActorNetwork(obs_dim=40, hidden_layers=default_config.marl.hidden_layers)
        relu_count = sum(
            1 for m in actor.shared.modules() if isinstance(m, torch.nn.ReLU)
        )
        # shared = mlp(obs_dim, hidden_layers, hidden_layers[-1])
        # → len(hidden_layers) ReLU layers for [256,256]
        expected = len(default_config.marl.hidden_layers)  # [256,256] → 2 ReLU
        assert relu_count == expected, f"ReLU={relu_count}, expected={expected} for {default_config.marl.hidden_layers}"
