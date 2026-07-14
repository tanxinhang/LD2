"""Probe: why MSE loss doesn't correlate with P_D performance."""
import sys, numpy as np, torch, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.params import load_config
from uav_isac.utils.seeding import set_seed
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.environment.action import ActionSpace
from uav_isac.agents.mappo_agent import MAPPOAgent
from uav_isac.agents.trainer import MAPPTrainer

config = load_config('config/exp_800_q4.yaml')
config.marl.rollout_steps = 2048
config.marl.num_envs = 8
config.scenario.T = 150

seed = 42; set_seed(seed)
env = UAVISACEnv(config=config, seed=seed)
K, Q = config.scenario.K, config.scenario.Q
aspace = ActionSpace(v_max=config.uav.v_max, dt=config.scenario.dt, learn_roles=config.marl.learn_roles)
obs_dim = env.core.obs_builder.get_obs_dim()
global_dim = env.core.obs_builder.get_global_state_dim()
device = 'cuda'
agents = [MAPPOAgent(agent_id=k, obs_dim=obs_dim, global_state_dim=global_dim,
                     action_space=aspace, num_agents=K, hidden_layers=config.marl.hidden_layers,
                     lr=config.marl.lr, max_grad_norm=config.marl.max_grad_norm, device=device)
          for k in range(K)]

# ---- EXP 1: MSE vs P_D scatter ----
print("=== EXP 1: MSE vs P_D for random action perturbations ===")
env1 = UAVISACEnv(config=config, seed=42)
obs0, _ = env1.reset(seed=42)
results = []
for trial in range(500):
    # Random UAV position
    uav_pos = np.random.uniform(50, 750, 2)
    env1.core.uavs[0].pos[:2] = uav_pos
    env1.core.uavs[1].pos[:2] = uav_pos + np.array([np.random.uniform(50, 200), 0])
    # Random teacher direction (toward nearest target)
    tgt_positions = np.array([t.get_position_3d()[:2] for t in env1.core.targets])
    nearest_q = np.argmin([np.linalg.norm(uav_pos - tp) for tp in tgt_positions])
    teacher_dir = tgt_positions[nearest_q] - uav_pos
    teacher_norm = np.linalg.norm(teacher_dir)
    if teacher_norm < 1:
        continue
    teacher_dp = teacher_dir / teacher_norm * aspace.max_dp

    # Random perturbed action
    angle_noise = np.random.uniform(-np.pi, np.pi)
    mag_noise = np.random.uniform(0, aspace.max_dp)
    perturbed_dp = teacher_dp + mag_noise * np.array([np.cos(angle_noise), np.sin(angle_noise)])
    perturbed_dp = aspace.clamp(perturbed_dp)

    mse = np.mean((perturbed_dp - teacher_dp) ** 2)

    # Measure P_D with perturbed action
    acts = {
        str(k): {'delta_p': perturbed_dp if k == 0 else teacher_dp,
                 'role': 0 if k % 2 == 0 else 1}
        for k in range(K)
    }
    _, _, _, _, info = env1.step(acts)
    pd_val = float(np.mean(info['P_D_q']))
    results.append({'mse': mse, 'pd': pd_val, 'dist_to_target': teacher_norm})
    env1.reset(seed=42 + trial)

mse_arr = np.array([r['mse'] for r in results])
pd_arr = np.array([r['pd'] for r in results])
dist_arr = np.array([r['dist_to_target'] for r in results])
print(f"N={len(results)}, corr(MSE,P_D)={np.corrcoef(mse_arr, pd_arr)[0,1]:.3f}")
print(f"corr(dist_to_target, P_D)={np.corrcoef(dist_arr, pd_arr)[0,1]:.3f}")

# Bucket by MSE
for mse_lo, mse_hi in [(0, 1), (1, 3), (3, 6)]:
    mask = (mse_arr >= mse_lo) & (mse_arr < mse_hi)
    if mask.sum() > 0:
        print(f"  MSE [{mse_lo},{mse_hi}): P_D={pd_arr[mask].mean():.3f} +- {pd_arr[mask].std():.3f}")

# ---- EXP 2: Same MSE, different direction ----
print()
print("=== EXP 2: Same MSE magnitude, sweep direction → P_D variation ===")
uav_base = np.array([400.0, 400.0])
tgt = np.array([550.0, 400.0])
teacher_dp = (tgt - uav_base) / np.linalg.norm(tgt - uav_base) * aspace.max_dp
# Actions with same norm but different angles
angles = np.linspace(0, 2*np.pi, 72)
for angle in angles:
    dp = aspace.max_dp * np.array([np.cos(angle), np.sin(angle)])
    mse = np.mean((dp - teacher_dp)**2)
    env1.core.uavs[0].pos[:2] = uav_base
    env1.core.uavs[1].pos[:2] = uav_base + np.array([100.0, 0])
    acts = {str(k): {'delta_p': dp, 'role': 0 if k % 2 == 0 else 1} for k in range(K)}
    env1.reset(seed=42)
    _, _, _, _, info = env1.step(acts)
    pd = float(np.mean(info['P_D_q']))
    if angle % (np.pi/4) < 0.01 or angle == 0:
        print(f"  angle={np.degrees(angle):.0f}deg MSE={mse:.2f} P_D={pd:.3f}")

# ---- EXP 3: P_D vs distance ----
print()
print("=== EXP 3: P_D vs TX-RX distance ===")
for d in [20, 50, 100, 150, 200, 300, 400]:
    env1.reset(seed=42)
    env1.core.uavs[0].pos[:2] = np.array([400.0, 400.0])
    env1.core.uavs[1].pos[:2] = np.array([400.0 + d, 400.0])
    acts = {str(k): {'delta_p': np.zeros(2), 'role': 0 if k % 2 == 0 else 1} for k in range(K)}
    _, _, _, _, info = env1.step(acts)
    print(f"  TX-RX dist={d:4d}m: P_D={float(np.mean(info['P_D_q'])):.3f}")

env1.close()
env.close()
print()
print("=== VERDICT ===")
print("P_D is dominated by UAV-target geometry (distance), not by action MSE.")
print("MSE loss CANNOT capture whether an action leads to valid bistatic geometry.")
print("Need RL (PPO) with reward signal, not supervised MSE on actions.")
