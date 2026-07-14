#!/usr/bin/env python
"""Probe the TRUE achievable ceiling of the env at ideal geometry (torch-free).

Places a static target and a tx/rx UAV pair near it (varying horizontal offset),
then prints per-candidate g_dd / chi_rep / d_raw / d_eff and the inner-solver
D_q / P_D. Answers: does the env actually deliver high P_D at good geometry, or
is something (e.g. the g_min gate, OTFS eta) capping it well below ~1.0?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config.params import get_default_config
from uav_isac.physical.deflection import DeflectionComputer
from uav_isac.physical.inner_solver import InnerSolver
from uav_isac.utils.math_utils import compute_PD

cfg = get_default_config()
ot, ch, ua, de, ta = cfg.otfs, cfg.channel, cfg.uav, cfg.detection, cfg.target
rng = np.random.default_rng(0)

dc = DeflectionComputer(
    fc=ot.fc, delta_f=ot.delta_f, T_sym=ot.T_sym, M=ot.M, N=ot.N,
    kT=ch.kT, B=ot.B, NF_dB=ch.NF, P_sense=ua.P_sense, P_report=ua.P_report,
    ric_K=ch.ric_K, rcs=ta.rcs, g_min=de.g_min, rng=rng,
    g_tx_dBi=getattr(ot, 'g_tx_dBi', 0.0), g_rx_dBi=getattr(ot, 'g_rx_dBi', 0.0),
    n_cpi=getattr(ot, 'n_cpi', 1),
)
solver = InnerSolver(
    K_q_max=de.K_q_max, B_q=de.B_q, P_FA=de.P_FA,
    omega_q=np.array(ta.omega_q[:1]),
    capacity_per_rx=cfg.p0_solver.capacity_per_rx,
)

H = cfg.scenario.height
tgt = np.array([[500.0, 500.0, 0.0]])          # one static target at area center
tgt_vel = np.zeros((1, 3))
fc = np.array([500.0, 500.0, 0.0])
roles = np.array([0, 1])                         # UAV0 = tx, UAV1 = rx

print(f"g_min={de.g_min}, P_FA={de.P_FA}, H={H}, n_cpi={getattr(ot,'n_cpi','?')}, "
      f"G_tx/rx_dBi={getattr(ot,'g_tx_dBi','?')}/{getattr(ot,'g_rx_dBi','?')}")
print(f"\n{'offset(m)':>9} {'leg(m)':>8} {'g_dd':>8} {'chi':>7} {'d_raw':>11} {'d_eff':>11} {'D_q':>9} {'P_D':>7}")
print("-"*78)
for off in [10, 30, 60, 100, 150, 200, 300]:
    # tx directly over target shifted by 'off' horizontally; rx symmetric other side
    uav = np.array([
        [500.0 - off/2, 500.0, H],
        [500.0 + off/2, 500.0, H],
    ])
    uav_vel = np.zeros((2, 3))
    entries = dc.compute(uav, uav_vel, tgt, tgt_vel, roles, fc)
    leg = np.sqrt((off/2)**2 + H**2)
    if not entries:
        print(f"{off:>9} {leg:>8.1f}   <no valid bistatic entries>")
        continue
    e = max(entries, key=lambda x: x.d_raw)   # the tx->rx->target candidate
    sol = solver.solve(entries, Q=1, K=2)
    Dq = float(sol.D_q_star[0])
    pd = float(compute_PD(np.array(Dq), de.P_FA))
    print(f"{off:>9} {leg:>8.1f} {e.g_dd:>8.3f} {e.chi_rep:>7.3f} {e.d_raw:>11.3e} "
          f"{e.d_eff:>11.3e} {Dq:>9.2f} {pd:>7.3f}")

print("\n解读：")
print(" - 若 P_D 在小偏移(腿~100m)处≈1.0 => env 天花板高, Greedy 只是没占好位/控制问题。")
print(" - 若 g_dd 普遍<g_min 导致 d_eff=0 / P_D 上不去 => 是 g_min 门控或 OTFS eta 把上限卡死了。")
