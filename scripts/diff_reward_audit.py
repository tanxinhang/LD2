"""Compare 1-step vs H-step difference reward signal strength.

Offline audit — no training, just random rollout + counterfactual computation.
Outputs per-agent statistics for both modes.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.params import load_config
from uav_isac.environment.env_wrapper import UAVISACEnv


def compute_fixed_utility(env, uav_positions, uav_velocities, roles,
                          target_positions, target_velocities,
                          selected_set):
    """Utility for fixed assignment (no P0 re-solve)."""
    entries = env.core.deflection_computer.compute(
        uav_positions, uav_velocities,
        target_positions, target_velocities,
        roles, env.core.fc_position,
        role_agnostic=not env.core.learn_roles,
    )
    D_q = np.zeros(env.Q, dtype=np.float64)
    emap = {(e.i, e.j, e.q): e.d_eff for e in entries}
    for (i, j, q) in selected_set:
        D_q[q] += emap.get((i, j, q), 0.0)
    return env.core.reward_computer.compute_team_utility_from_deflection(D_q)


def run_audit(cfg, n_windows=100):
    """Collect assignment windows with random actions, compute 1-step and H-step diff."""
    env = UAVISACEnv(config=cfg, seed=42)
    H = cfg.marl.assignment_hold_frames
    K, Q, T = cfg.scenario.K, cfg.scenario.Q, cfg.scenario.T

    one_step_diffs = {k: [] for k in range(K)}
    h_step_diffs = {k: [] for k in range(K)}

    obs, _ = env.reset(seed=42)
    window_count = 0

    while window_count < n_windows and env.core.t < T:
        # Start of window
        window_start = env.core.t
        window_states = []   # [(uav_pos, uav_vel, roles, tgt_pos, tgt_vel, selected_set, actions)]
        window_utils = []     # [actual_utility per frame]

        for h in range(H):
            if env.core.t >= T:
                break
            # Store pre-step state
            uav_pos = np.array([u.pos.copy() for u in env.core.uavs])
            uav_vel = np.array([u.vel.copy() for u in env.core.uavs])
            roles = np.array([u.role for u in env.core.uavs], dtype=np.int32)
            tgt_pos = np.array([t.get_position_3d() for t in env.core.targets])
            tgt_vel = np.array([[t.state[2], t.state[3], 0.0] for t in env.core.targets])

            # Random actions
            actions = {}
            for k in range(K):
                dp = env.rng.uniform(-2.5, 2.5, 2)
                actions[str(k)] = {'delta_p': dp, 'role': 0}

            obs, _, _, _, info = env.step(actions)
            si = env.current_step_info
            if si is None:
                break

            window_states.append({
                'uav_pos': uav_pos, 'uav_vel': uav_vel, 'roles': roles,
                'tgt_pos': tgt_pos, 'tgt_vel': tgt_vel,
                'actions': {k: actions[str(k)]['delta_p'].copy() for k in range(K)},
                'selected_set': list(si.p0_solution.selected_set),
            })
            window_utils.append(
                compute_fixed_utility(env, uav_pos, uav_vel, roles,
                                      tgt_pos, tgt_vel,
                                      si.p0_solution.selected_set))

        # Compute 1-step and H-step differences for each frame in window
        # For each frame t, the ACTUAL next state is at window_states[t+1] (or end).
        # The counterfactual keeps agent k at its PRE-step position.
        L = len(window_states)
        for t in range(L):
            # ACTUAL positions for frame t: after action, UAVs moved to next positions.
            # Use the NEXT frame's uav_pos (if available) as the post-action state.
            if t + 1 < L:
                actual_pos = window_states[t+1]['uav_pos'].copy()
                actual_util = window_utils[t+1]
            else:
                actual_pos = window_states[t]['uav_pos'].copy()
                for kk in range(K):
                    actual_pos[kk][:2] += window_states[t]['actions'][kk]
                actual_util = window_utils[t]

            for k in range(K):
                # --- 1-step diff: remove just agent k's action at t ---
                cf1_pos = actual_pos.copy()
                cf1_pos[k][:2] -= window_states[t]['actions'][k]
                cf1_util = compute_fixed_utility(
                    env, cf1_pos,
                    window_states[t]['uav_vel'],
                    window_states[t]['roles'],
                    window_states[t]['tgt_pos'],
                    window_states[t]['tgt_vel'],
                    window_states[t]['selected_set'],
                )
                one_step_diffs[k].append(actual_util - cf1_util)

                # --- H-step diff: remove agent k's action at t, replay subsequent actions ---
                cfH_pos = actual_pos.copy()
                cfH_pos[k][:2] -= window_states[t]['actions'][k]
                weighted_gap = 0.0
                weight_sum = 0.0

                # Frame t (post-action)
                cfH_util = compute_fixed_utility(
                    env, cfH_pos,
                    window_states[t]['uav_vel'],
                    window_states[t]['roles'],
                    window_states[t]['tgt_pos'],
                    window_states[t]['tgt_vel'],
                    window_states[t]['selected_set'],
                )
                weight = 1.0
                weighted_gap += weight * (actual_util - cfH_util)
                weight_sum += weight

                # Frames t+1 to end of window
                for h in range(t+1, L):
                    # Apply action at frame h (same for actual and cf)
                    for kk in range(K):
                        cfH_pos[kk][:2] += window_states[h]['actions'][kk]

                    # Actual utility at frame h
                    if h + 1 < L:
                        actual_h_pos = window_states[h+1]['uav_pos']
                        actual_h_util = window_utils[h+1]
                    else:
                        actual_h_pos = window_states[h]['uav_pos'].copy()
                        for kk in range(K):
                            actual_h_pos[kk][:2] += window_states[h]['actions'][kk]
                        actual_h_util = window_utils[h]

                    cfH_util_h = compute_fixed_utility(
                        env, cfH_pos,
                        window_states[h]['uav_vel'],
                        window_states[h]['roles'],
                        window_states[h]['tgt_pos'],
                        window_states[h]['tgt_vel'],
                        window_states[h]['selected_set'],
                    )
                    weight = cfg.marl.gamma ** (h - t + 1)
                    weighted_gap += weight * (actual_h_util - cfH_util_h)
                    weight_sum += weight

                h_step_diffs[k].append(weighted_gap / max(weight_sum, 1e-8))

        window_count += 1

    env.close()
    return one_step_diffs, h_step_diffs


def report(one_step, h_step):
    print(f"{'Metric':<20s} {'1-step':>12s} {'H-step':>12s} {'Ratio':>8s}")
    print('-' * 55)

    all_1 = np.concatenate([np.array(v) for v in one_step.values()])
    all_H = np.concatenate([np.array(v) for v in h_step.values()])

    for name, data in [('1-step', all_1), ('H-step', all_H)]:
        pass

    print(f"{'mean':<20s} {all_1.mean():12.6f} {all_H.mean():12.6f} {all_H.mean()/max(abs(all_1.mean()),1e-8):8.2f}x")
    print(f"{'std':<20s} {all_1.std():12.6f} {all_H.std():12.6f} {all_H.std()/max(all_1.std(),1e-8):8.2f}x")
    print(f"{'p95':<20s} {np.percentile(np.abs(all_1),95):12.6f} {np.percentile(np.abs(all_H),95):12.6f}")
    print(f"{'positive ratio':<20s} {(all_1>0).mean():11.1%} {(all_H>0).mean():11.1%}")
    print(f"{'zero ratio':<20s} {(abs(all_1)<1e-10).mean():11.1%} {(abs(all_H)<1e-10).mean():11.1%}")

    # Agent-wise std
    agent_std_1 = np.std([np.mean(v) for v in one_step.values()])
    agent_std_H = np.std([np.mean(v) for v in h_step.values()])
    print(f"{'agent-wise mean std':<20s} {agent_std_1:12.6f} {agent_std_H:12.6f}")

    signal_ratio = all_H.std() / max(all_1.std(), 1e-8)
    print(f"\nSignal amplification: {signal_ratio:.2f}x")
    if signal_ratio > 1.5:
        print("=> H-step significantly amplifies difference signal — worth implementing.")
    elif signal_ratio > 1.1:
        print("=> H-step provides modest amplification.")
    else:
        print("=> H-step does NOT amplify signal. Check implementation.")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/exp_800_q4.yaml')
    ap.add_argument('--windows', type=int, default=100)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.marl.assignment_hold_frames = 5
    print(f"config={args.config} H={cfg.marl.assignment_hold_frames} windows={args.windows}")
    one, h = run_audit(cfg, n_windows=args.windows)
    report(one, h)
