"""Diagnose the detection chain of one target in a failing seed.

Records per frame the deflection entries involving the target, the DD-gate
status of the best pair, and the per-target detection probability, to
distinguish distance, pairing, DD-gate, and power causes.
"""
import sys
import numpy as np

sys.path.insert(0, r'D:\BYLW\LD3')
from config.params import load_config
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.physical.otfs import compute_dd_effectiveness
from uav_isac.physical.geometry import compute_all_bistatic_params


def main():
    seed = int(sys.argv[1])
    q = int(sys.argv[2])
    cfg = load_config(
        'config/exp_800_k12q12_distributed_v2_dynamic_u2u_robustbelief_gapcoverage_pilot.yaml')
    env = UAVISACEnv(cfg, seed=seed)
    obs, info = env.reset()
    core = env.core
    otfs = cfg.otfs
    det = cfg.detection
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
        if t % 25 != 0 and t != core.T - 1:
            continue
        uav = np.asarray([u.pos for u in core.uavs])
        uav_vel = np.asarray([u.vel for u in core.uavs])
        tgt = np.asarray([tgt_.get_position_3d() for tgt_ in core.targets])
        tgt_vel = np.asarray([np.asarray([*t_.get_velocity(), 0.0])
                              for t_ in core.targets])
        d2u = np.linalg.norm(uav[:, :2] - tgt[q][:2], axis=1)
        # DD gate of the best (i, j) pair involving the two nearest UAVs.
        roles = np.zeros(core.K, dtype=np.int64)
        best = []
        for i in range(core.K):
            for j in range(core.K):
                if i == j:
                    continue
                tau, nu, _ = compute_all_bistatic_params(
                    uav[i:i+1], uav_vel[i:i+1], tgt[q:q+1], tgt_vel[q:q+1],
                    roles, otfs.fc, cfg.target.rcs, role_agnostic=True)
                g_dd = compute_dd_effectiveness(
                    float(tau[0, 0, 0]), float(nu[0, 0, 0]),
                    otfs.delta_f, otfs.T_sym, det.B_q, 16, det.g_min)
                best.append((float(g_dd), int(i), int(j)))
        best.sort(reverse=True)
        g_best, i_b, j_b = best[0]
        psel = core._last_isac_metrics.get(
            'p0_target_selected_mask', None)
        pd_q = core._last_isac_metrics.get('detection_deflection_q', None)
        pdv = None
        if pd_q is not None:
            from uav_isac.physical.detection import (
                compute_detection_probabilities)
            pdv = float(compute_detection_probabilities(
                np.asarray([pd_q[q]]), cfg.detection.P_FA)[0])
        print('t=%3d 目标%d: 最近UAV距离=[%.0f,%.0f]m (UAV%d,%d) 最佳对DD-gate=%.3f (UAV%d->UAV%d) P_D=%.4f' % (
            t, q, d2u[np.argsort(d2u)][0], d2u[np.argsort(d2u)][1],
            int(np.argsort(d2u)[0]), int(np.argsort(d2u)[1]),
            g_best, i_b, j_b, pdv if pdv is not None else -1))
    env.close()


if __name__ == '__main__':
    main()
