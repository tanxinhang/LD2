"""Reward audit (§2): trace every reward component frame-by-frame.

Outputs:
  results/reward_audit_steps.csv   — per-frame, per-agent reward breakdown
  results/reward_audit_summary.csv — per-episode statistics
  results/assignment_switch_audit.csv — switch vs non-switch reward variance

Answers:
  1. Which reward component has the largest variance?
  2. Is absolute_sensing_k dominated by P0 selection?
  3. Does assignment switching increase reward variance?
  4. Does centered marginal give negative reward to positively contributing UAVs?
  5. Does team reward drown out individual reward?
"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config.params import load_config
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.environment.action import ActionSpace


def collect_audit_data(cfg, n_episodes=5, seed_offset=0):
    """Run episodes with random actions, collect per-step reward components."""
    rows = []
    aspace = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt,
                         learn_roles=cfg.marl.learn_roles)
    K, Q, T = cfg.scenario.K, cfg.scenario.Q, cfg.scenario.T

    for ep in range(n_episodes):
        seed = 1000 + seed_offset + ep
        env = UAVISACEnv(config=cfg, seed=seed)
        obs, _ = env.reset(seed=seed)

        prev_selected = set()
        for step in range(T):
            # Random actions
            acts = {}
            for k in range(K):
                dp = env.rng.uniform(-2.5, 2.5, 2)
                acts[str(k)] = {'delta_p': dp, 'role': 0}

            obs, rewards, term, _, info = env.step(acts)
            si = env.current_step_info
            if si is None:
                break

            # --- Extract all reward components ---
            # Team-level
            team_utility = si.team_reward  # already net of comm cost
            comm_cost = si.p0_solution.total_bits * cfg.marl.lambda_report
            safety_penalty = si.constraint_info.get('safety_penalty', 0.0)
            energy_penalty = si.constraint_info.get('energy_penalty', 0.0)
            fairness_penalty = si.constraint_info.get('fairness_penalty', 0.0)
            boundary_penalty = si.constraint_info.get('boundary_penalty', 0.0)
            total_penalty = si.constraint_info.get('total_penalty', 0.0)

            # P_D and deflection
            P_D_q = si.P_D_q
            fused_deflection = si.p0_solution.D_q_star

            # Assignment
            selected = set(tuple(t) for t in si.p0_solution.selected_set)
            switched = selected != prev_selected
            assignment_id = hash(frozenset(selected)) % 10000
            n_pairs = len(selected)
            prev_selected = selected

            # Per-agent
            for k in range(K):
                r_k = float(rewards[str(k)])
                # Action norm
                action_norm = float(np.linalg.norm(acts[str(k)]['delta_p']))

                # Marginal contribution (approximate from shaped reward diff)
                # shaped_k = team_reward + eta_mc * (ΔU_k - mean(ΔU))
                # So ΔU_k - mean(ΔU) = (shaped_k - team_reward) / eta_mc
                centered_marginal = (r_k - team_utility) / max(cfg.marl.eta_mc, 1e-6)

                # Per-agent sensing: d_eff for entries involving this UAV
                n_entries = 0
                sensing_sum = 0.0
                if hasattr(si, 'deflection_entries'):
                    for e in si.deflection_entries:
                        if e.d_eff > 0 and (e.i == k or e.j == k):
                            sensing_sum += e.d_eff
                            n_entries += 1
                absolute_sensing_k = sensing_sum / max(n_entries, 1)

                row = {
                    'episode': ep, 'step': step, 'agent': k,
                    'team_utility': float(team_utility),
                    'comm_cost': float(comm_cost),
                    'energy_penalty': float(energy_penalty),
                    'safety_penalty': float(safety_penalty),
                    'fairness_penalty': float(fairness_penalty),
                    'boundary_penalty': float(boundary_penalty),
                    'total_penalty': float(total_penalty),
                    'centered_marginal': float(centered_marginal),
                    'absolute_sensing_k': float(absolute_sensing_k),
                    'total_reward_k': float(r_k),
                    'fused_deflection_q0': float(fused_deflection[0]) if len(fused_deflection) > 0 else 0.0,
                    'P_D_q0': float(P_D_q[0]) if len(P_D_q) > 0 else 0.0,
                    'P_D_mean': float(np.mean(P_D_q)),
                    'assignment_id': assignment_id,
                    'assignment_switched': int(switched),
                    'n_pairs': n_pairs,
                    'action_norm': float(action_norm),
                }
                rows.append(row)

            if term.get('__all__', False):
                break
        env.close()

    return rows


def compute_summary(rows, output_dir):
    """Compute statistics and write CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # Step-level CSV
    keys = list(rows[0].keys())
    with open(os.path.join(output_dir, 'reward_audit_steps.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to results/reward_audit_steps.csv")

    # Summary statistics
    numeric_keys = [k for k in keys if k not in ('episode', 'step', 'agent')]
    summary_rows = []
    for k in numeric_keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        summary_rows.append({
            'field': k,
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'p05': float(np.percentile(vals, 5)),
            'p50': float(np.percentile(vals, 50)),
            'p95': float(np.percentile(vals, 95)),
        })
    with open(os.path.join(output_dir, 'reward_audit_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['field', 'mean', 'std', 'min', 'max', 'p05', 'p50', 'p95'])
        w.writeheader()
        w.writerows(summary_rows)
    print("Wrote reward_audit_summary.csv")

    # Assignment switch audit
    switched = np.array([r['assignment_switched'] for r in rows], dtype=bool)
    for field in ['total_reward_k', 'absolute_sensing_k', 'centered_marginal', 'P_D_mean']:
        vals = np.array([r[field] for r in rows])
        print(f"  {field}: switch_mean={vals[switched].mean():.4f}  "
              f"no_switch_mean={vals[~switched].mean():.4f}  "
              f"switch_std={vals[switched].std():.4f}  no_switch_std={vals[~switched].std():.4f}  "
              f"ratio={vals[switched].std()/max(vals[~switched].std(),1e-6):.2f}x")

    # Correlations
    print("\n=== Reward component correlations ===")
    pd_mean = np.array([r['P_D_mean'] for r in rows])
    for field in ['team_utility', 'centered_marginal', 'absolute_sensing_k', 'total_reward_k', 'action_norm']:
        vals = np.array([r[field] for r in rows])
        corr = np.corrcoef(vals, pd_mean)[0, 1]
        print(f"  corr({field}, P_D_mean) = {corr:.3f}")

    # Per-agent breakdown
    print("\n=== Per-agent reward statistics ===")
    K = max(r['agent'] for r in rows) + 1
    for k in range(K):
        agent_rows = [r for r in rows if r['agent'] == k]
        r_vals = np.array([r['total_reward_k'] for r in agent_rows])
        adv_vals = np.array([r['centered_marginal'] for r in agent_rows])
        print(f"  Agent {k}: reward mean={r_vals.mean():.4f} std={r_vals.std():.4f}  "
              f"marginal mean={adv_vals.mean():.4f} std={adv_vals.std():.4f}")

    # Key questions
    print("\n=== KEY FINDINGS ===")
    team = np.array([r['team_utility'] for r in rows])
    indv = np.array([r['total_reward_k'] for r in rows])
    sensing = np.array([r['absolute_sensing_k'] for r in rows])
    marg = np.array([r['centered_marginal'] for r in rows])

    # Q1: largest variance component among reward pieces
    components = {
        'team_utility': team[:len(team)//4],  # sample one agent's values
        'centered_marginal': marg[:len(marg)//4],
        'absolute_sensing': sensing[:len(sensing)//4],
    }
    largest = max(components, key=lambda c: components[c].std())
    print(f"Q1: Largest variance component = {largest} (std={components[largest].std():.4f})")

    # Q2: is absolute_sensing dominated by P0 selection?
    in_pair = np.array([1 if r['n_pairs'] > 0 else 0 for r in rows])
    sensing_in = sensing[in_pair == 1]
    sensing_out = sensing[in_pair == 0]
    if len(sensing_in) > 0 and len(sensing_out) > 0:
        print(f"Q2: sensing_k when in_pair={sensing_in.mean():.4f}, "
              f"no_pair={sensing_out.mean():.4f}, ratio={sensing_in.mean()/max(sensing_out.mean(),1e-6):.1f}x")

    # Q3: assignment switch increases reward variance?
    s_std = np.array([r['total_reward_k'] for r in rows if r['assignment_switched']]).std()
    ns_std = np.array([r['total_reward_k'] for r in rows if not r['assignment_switched']]).std()
    print(f"Q3: reward std switch={s_std:.4f} vs no_switch={ns_std:.4f}, ratio={s_std/max(ns_std,1e-6):.2f}x")

    # Q4: centered marginal gives negative reward to positive contributors?
    pos_sense = sensing > np.median(sensing)
    neg_marg_given_pos = (marg[pos_sense] < 0).mean()
    print(f"Q4: P(marginal<0 | sensing>median) = {neg_marg_given_pos:.1%}")

    # Q5: team reward drowns individual?
    corr_team_indv = np.corrcoef(team[::4], indv[::4])[0, 1]  # decimate: 4 agents
    print(f"Q5: corr(team_reward, individual_reward) = {corr_team_indv:.3f}")
    print(f"    team_reward fraction of total = {abs(team[:len(indv)//4]).mean()/max(abs(indv).mean(),1e-6):.2f}x")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Reward audit (§2)")
    ap.add_argument('--config', default='config/exp_800_q4.yaml')
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--output', default='results')
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"config={args.config} K={cfg.scenario.K} Q={cfg.scenario.Q} T={cfg.scenario.T} "
          f"learn_roles={cfg.marl.learn_roles} eta_mc={cfg.marl.eta_mc} eta_sense={cfg.marl.eta_sense}")

    rows = collect_audit_data(cfg, n_episodes=args.episodes)
    compute_summary(rows, args.output)
    print("Done.")
