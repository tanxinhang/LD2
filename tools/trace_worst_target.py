"""Trace the worst-target geometry for a failing seed (deep-audit diagnostic).

Records per-frame: nearest/farthest UAV distance to the worst target, the
responsible (nearest) UAV id, the mean belief error, and the executed
assignment view of the responsible UAV.  Prints a compact trajectory.
"""
import sys
import numpy as np

sys.path.insert(0, r'D:\BYLW\LD3')
from config.params import load_config
from uav_isac.environment.env_wrapper import UAVISACEnv


def main():
    seed = int(sys.argv[1])
    q = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    cfg = load_config(
        'config/exp_800_k12q12_distributed_v2_dynamic_u2u_robustbelief_freshness_pilot.yaml')
    env = UAVISACEnv(cfg, seed=seed)
    obs, info = env.reset()
    core = env.core
    if q < 0:
        uav = np.asarray([u.pos[:2] for u in core.uavs])
        tgt = np.asarray([t.get_position_3d()[:2] for t in core.targets])
        d0 = np.linalg.norm(uav[:, None, :] - tgt[None, :, :], axis=-1)
        q = int(np.argmax(d0.min(axis=0)))
    traj = []
    for t in range(core.T):
        actions = {str(k): {'delta_p': np.zeros(2), 'role': 2}
                   for k in range(core.K)}
        core.submit_learned_communications(
            messages={k: np.zeros(192, dtype=np.float64)
                      for k in range(core.K)},
            rate_indices={k: 1 for k in range(core.K)},
            comm_power_fractions={k: 0.05 for k in range(core.K)},
            sensing_target_weights={
                k: np.ones(core.Q, dtype=np.float64) for k in range(core.K)},
        )
        env.step(actions)
        uav = np.asarray([u.pos[:2] for u in core.uavs])
        tgt_true = np.asarray([t.get_position_3d()[:2] for t in core.targets])
        d = np.linalg.norm(uav - tgt_true[q], axis=1)
        bel = core.belief_mgr.mean[:, q, :2]
        assigned = int(core._distributed_movement_target[core.K - 1]) \
            if hasattr(core, '_distributed_movement_target') else -1
        traj.append((t, d.min(), int(d.argmin()), d.max(),
                     float(np.linalg.norm(bel.mean(0) - tgt_true[q])),
                     assigned))
    print('seed=%d worst_target=%d' % (seed, q))
    print(' t   nearest(m) UAV  farthest(m) belief_err(m)  execTarget(viewer11)')
    for t, dmin, kmin, dmax, berr, assigned in traj[::25] + [traj[-1]]:
        print('%3d  %7.1f  %3d  %8.1f  %9.1f  %d' % (
            t, dmin, kmin, dmax, berr, assigned))
    env.close()


if __name__ == '__main__':
    main()
