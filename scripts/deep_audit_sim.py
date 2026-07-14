#!/usr/bin/env python
"""Deep simulation diagnostics for the suspected issues (read-only; no fixes).

Unlike pass/fail unit tests, each probe QUANTIFIES magnitude + downstream impact
on the ACTUAL code (imports the real package), then prints a verdict.

Run:  python scripts/deep_audit_sim.py
Torch is optional (only Probe 5b uses it; skipped if unavailable).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from config.params import get_default_config
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.environment.action import ActionSpace

cfg = get_default_config()
K, Q = cfg.scenario.K, cfg.scenario.Q
max_dp = cfg.uav.v_max * cfg.scenario.dt
RNG = np.random.default_rng(0)
LINE = "-" * 70


def banner(t): print(f"\n{LINE}\n{t}\n{LINE}")


# ===================================================================== Probe 1
def probe_action_fidelity():
    """A1/A2: how often + how much does the env rewrite the policy's action?"""
    banner("PROBE 1  动作执行保真度 (策略输出 Δp vs 环境实际位移)")
    aspace = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt)
    aspace.rng = RNG
    N = 20000
    clamp_cnt, errs, diag_shrink = 0, [], []
    for _ in range(N):
        # realistic policy outputs: tanh-Gaussian, mean~0, std~exp(0) (untrained)
        dp_mean = RNG.normal(0, 1, 2)
        a, _ = aspace.decode(dp_mean, np.zeros(2), RNG.normal(0, 1, 3))
        stored = a.delta_p
        n = np.linalg.norm(stored)
        executed = stored * (max_dp / n) if n > max_dp else stored
        if n > max_dp + 1e-9:
            clamp_cnt += 1
            diag_shrink.append(n / max_dp)  # how far outside the disk
        errs.append(np.linalg.norm(stored - executed))
    errs = np.array(errs)
    print(f"  采样动作数: {N}")
    print(f"  被径向 clamp 改写比例: {clamp_cnt/N*100:.1f}%   (box 对角可达 {np.sqrt(2):.3f}·max_dp)")
    print(f"  位移误差 ||stored-executed||: 均值 {errs.mean():.3f}m, 95% {np.percentile(errs,95):.3f}m, max {errs.max():.3f}m  (max_dp={max_dp:.2f}m)")
    if diag_shrink:
        print(f"  被改写动作平均超出圆盘 {np.mean(diag_shrink):.2f}× → 系统性削弱对角(斜向)机动")
    print(f"  判定: {'⚠ 显著' if clamp_cnt/N>0.1 else 'OK'} — buffer 存 stored、环境执行 executed，PPO 按 stored 学梯度")


# ===================================================================== Probe 2
def probe_action_fidelity_rollout():
    """Same as P1 but in a real episode (includes boundary bounce)."""
    banner("PROBE 2  整局动作保真度 (含边界 bounce, 真实 env)")
    env = UAVISACEnv(config=cfg, seed=1)
    aspace = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt); aspace.rng = RNG
    obs, _ = env.reset()
    mism, total, pos_err = 0, 0, []
    for t in range(cfg.scenario.T):
        pre = np.array([u.pos[:2].copy() for u in env.core.uavs])
        acts = {}
        cmd = {}
        for k in range(K):
            a = aspace.sample()
            acts[str(k)] = {'delta_p': a.delta_p, 'role': a.role}
            cmd[k] = a.delta_p.copy()
        obs, _, term, trunc, _ = env.step(acts)
        post = np.array([u.pos[:2] for u in env.core.uavs])
        for k in range(K):
            actual = post[k] - pre[k]
            e = np.linalg.norm(actual - cmd[k]); pos_err.append(e)
            if e > 1e-6: mism += 1
            total += 1
        if term.get('__all__') or trunc.get('__all__'): break
    pos_err = np.array(pos_err)
    print(f"  步数×UAV: {total}")
    print(f"  实际位移≠命令 Δp 的比例: {mism/total*100:.1f}%")
    print(f"  位移偏差: 均值 {pos_err.mean():.3f}m, 95% {np.percentile(pos_err,95):.3f}m, max {pos_err.max():.3f}m")
    print(f"  判定: {'⚠ MDP 动作语义不一致' if mism/total>0.1 else 'OK'}")


# ===================================================================== Probe 3
def probe_belief_drift():
    """B: belief mean vs true target over the episode; when does it exceed R_sense?"""
    banner("PROBE 3  Belief 漂移 (信念目标位置 vs 真值)")
    env = UAVISACEnv(config=cfg, seed=2)
    env.reset()
    R_sense = 120.0  # approx horizontal sensing radius (from probe_ceiling)
    bm = env.core.belief_mgr
    true0 = np.array([t.get_position_3d()[:2] for t in env.core.targets])
    rows = []
    blind_from = None
    for t in range(cfg.scenario.T):
        for tgt in env.core.targets: tgt.step()
        bm.step()
        truep = np.array([tg.get_position_3d()[:2] for tg in env.core.targets])
        # belief mean for uav 0
        errs = []
        for q in range(Q):
            b = bm.get_belief(0, q)
            errs.append(np.linalg.norm(b.mean[:2] - truep[q]))
        me = float(np.mean(errs))
        if t in (0, 24, 49, 99, cfg.scenario.T - 1):
            aoi = bm.get_belief(0, 0).aoi
            rows.append((t, np.linalg.norm(truep[0]-true0[0]), me, aoi))
        if blind_from is None and me > R_sense:
            blind_from = t
    print(f"  {'frame':>6}{'目标移动(m)':>14}{'belief误差(m)':>16}{'AoI':>6}")
    for t, mv, me, aoi in rows:
        print(f"  {t:>6}{mv:>14.1f}{me:>16.1f}{aoi:>6}")
    if blind_from is not None:
        print(f"  belief 误差超过感知半径(~{R_sense:.0f}m)的起始帧: {blind_from}  → 此后 agent 目标信息基本失效")
    print(f"  判定: {'⚠ 开环(信念不传播/不更新)' if rows[-1][2] > R_sense*0.5 else 'OK'}")


# ===================================================================== Probe 4
def probe_role_and_feasibility():
    """B: random-policy role churn + fraction of frames with NO valid tx-rx pair."""
    banner("PROBE 4  角色抖动 & 双基地可行性 (随机策略)")
    env = UAVISACEnv(config=cfg, seed=3)
    aspace = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt); aspace.rng = RNG
    obs, _ = env.reset()
    prev_roles = None; switches = 0; steps = 0; no_pair = 0; viol = 0
    for t in range(cfg.scenario.T):
        acts = {}; roles = []
        for k in range(K):
            a = aspace.sample(); acts[str(k)] = {'delta_p': a.delta_p, 'role': a.role}; roles.append(a.role)
        obs, _, term, trunc, info = env.step(acts)
        roles = np.array(roles)
        if (roles == 0).sum() == 0 or (roles == 1).sum() == 0:
            no_pair += 1   # need >=1 tx AND >=1 rx for any bistatic candidate
        if prev_roles is not None:
            switches += (roles != prev_roles).sum()
        prev_roles = roles; steps += 1
        ci = env.current_step_info.constraint_info if env.current_step_info else {}
        if ci.get('any_violation', False): viol += 1
        if term.get('__all__') or trunc.get('__all__'): break
    print(f"  角色切换率: {switches/(steps*K)*100:.1f}% /agent/frame")
    print(f"  无有效 tx-rx 配对的帧比例: {no_pair/steps*100:.1f}%  (这些帧根本不可能有感知候选)")
    print(f"  约束违反率: {viol/steps*100:.1f}%")
    print(f"  判定: 角色每帧独立采样 → 高抖动 + {no_pair/steps*100:.0f}% 帧零感知, 放大方差")


# ===================================================================== Probe 5
def probe_logprob_consistency():
    """log-prob: numpy decode vs compute_log_prob (and torch path if available)."""
    banner("PROBE 5  log-prob 一致性 (numpy 路径; torch 路径若可用)")
    aspace = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt); aspace.rng = RNG
    maxerr = 0.0
    for _ in range(5000):
        dpm = RNG.normal(0, 1, 2); dps = RNG.normal(-1, 0.5, 2); rl = RNG.normal(0, 1, 3)
        a, lp = aspace.decode(dpm, dps, rl)
        lp2 = aspace.compute_log_prob(a, dpm, dps, rl)
        maxerr = max(maxerr, abs(lp - lp2))
    print(f"  decode vs compute_log_prob 最大误差: {maxerr:.2e}  (阈值 1e-4)")
    print(f"  判定: {'OK (ratio 安全)' if maxerr < 1e-4 else '⚠ 检查'}")


# ===================================================================== Probe 6
def probe_double_advnorm():
    """C5: numerically demonstrate buffer-global + minibatch double normalization."""
    banner("PROBE 6  Advantage 双重归一化数值演示")
    adv = RNG.normal(2.0, 5.0, 4096)              # raw advantages (nonzero mean/var)
    g = (adv - adv.mean()) / (adv.std() + 1e-8)   # buffer global norm (get_training_data)
    # minibatch re-norm (update loop)
    mb = g[:256]; mb2 = (mb - mb.mean()) / (mb.std() + 1e-8)
    single = g[:256]
    print(f"  原始 adv: mean={adv.mean():.2f} std={adv.std():.2f}")
    print(f"  全局归一化后该 minibatch: mean={single.mean():.3f} std={single.std():.3f}")
    print(f"  再做 minibatch 归一化后: mean={mb2.mean():.3f} std={mb2.std():.3f}")
    print(f"  两者逐元素最大差异: {np.abs(mb2 - single).max():.3f}  → 第二次归一化确实改变了相对 scale")
    print(f"  判定: ⚠ 双重归一化 (buffer.get_training_data + update 各一次), 应只保留一次")


# ===================================================================== Probe 5b
def probe_torch_paths():
    banner("PROBE 7  (可选) torch 路径: actor 有效层数 + log_std clamp + critic 维度")
    try:
        import torch
        from uav_isac.agents.networks import ActorNetwork, CriticNetwork
        a = ActorNetwork(obs_dim=40, hidden_layers=cfg.marl.hidden_layers)
        n_relu = sum(1 for m in a.modules() if isinstance(m, torch.nn.ReLU))
        n_lin = sum(1 for m in a.modules() if isinstance(m, torch.nn.Linear))
        print(f"  Actor: Linear 层={n_lin}, ReLU(非线性)层={n_relu}  (期望 2 层非线性?)")
        # log_std effective range via forward
        o = torch.zeros(1, 40)
        _, ls, _ = a(o)
        print(f"  forward 输出 log_std 范围: [{ls.min().item():.2f},{ls.max().item():.2f}] (forward clamp 应是 [-2,1])")
        print(f"  判定: {'⚠ Actor 仅 1 层非线性' if n_relu < 2 else 'OK'}")
    except Exception as e:
        print(f"  (torch 不可用, 跳过: {e})")


# ===================================================================== Probe 8
def probe_obs_ranges():
    """Audit: observation component ranges (normalization sanity)."""
    banner("PROBE 8  观测归一化范围 (随机 rollout 下各分量是否越界/尺度失衡)")
    env = UAVISACEnv(config=cfg, seed=4)
    aspace = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt); aspace.rng = RNG
    obs, _ = env.reset()
    mn, mx = None, None
    for t in range(cfg.scenario.T):
        acts = {str(k): (lambda a: {'delta_p': a.delta_p, 'role': a.role})(aspace.sample()) for k in range(K)}
        obs, _, term, trunc, _ = env.step(acts)
        v = np.stack([obs[str(k)] for k in range(K)])
        mn = v.min(0) if mn is None else np.minimum(mn, v.min(0))
        mx = v.max(0) if mx is None else np.maximum(mx, v.max(0))
        if term.get('__all__') or trunc.get('__all__'): break
    over = int(np.sum((mx > 1.5) | (mn < -1.5)))
    print(f"  观测维度: {len(mx)}")
    print(f"  全局范围: min={mn.min():.2f}, max={mx.max():.2f}")
    print(f"  |分量|>1.5 的维度数: {over}  (归一化良好的话应≈0; >1 可能是 AoI/100 越界或 belief 误差)")
    print(f"  目标速度课程 {cfg.target.speed_range} 但 belief 速度按 /25 归一化 → 速度分量数量级 ~{cfg.target.speed_range[1]/25:.2f}（偏小，尺度失衡）")
    print(f"  判定: {'⚠ 有分量越界/尺度失衡' if over > 0 else '范围基本可控'}")


# ===================================================================== Probe 9
def probe_dt_coherence():
    """dt vs sensing coherence: is n_cpi coherent integration physically valid?"""
    banner("PROBE 9  dt 与感知相干性一致性 (n_cpi 相干积累是否过乐观)")
    c = 3e8; fc = cfg.otfs.fc; lam = c / fc
    vmax_tgt = cfg.target.speed_range[1]
    T_sym = cfg.otfs.T_sym; ncpi = getattr(cfg.otfs, 'n_cpi', 1)
    coh_time = lam / (2 * max(vmax_tgt, 1e-3))      # rough coherence time
    cpi_time = ncpi * cfg.otfs.N * T_sym             # time spanned by n_cpi OTFS frames
    print(f"  载频 {fc/1e9:.0f}GHz, λ={lam*100:.2f}cm, 目标最大速 {vmax_tgt} m/s")
    print(f"  相干时间 ~ λ/(2v) = {coh_time*1e3:.2f} ms")
    print(f"  dt(决策步) = {cfg.scenario.dt*1e3:.0f} ms;  n_cpi={ncpi} 帧 ≈ {cpi_time*1e3:.1f} ms 的积累窗")
    ratio = cpi_time / coh_time
    print(f"  积累窗 / 相干时间 = {ratio:.0f}×")
    print(f"  判定: {'⚠ 相干积累窗远超相干时间 → 128× 相干增益对动目标过乐观(需块平稳假设或改非相干)' if ratio > 3 else 'OK'}")
    print(f"  注: dt={cfg.scenario.dt*1e3:.0f}ms 作'控制步长'正常; 问题在把它当一次相干 CPI 且 n_cpi 偏大")


if __name__ == "__main__":
    print("="*70 + "\nDEEP SIMULATION AUDIT (read-only)\n" +
          f"config: region={cfg.scenario.region_size} H={cfg.scenario.height} dt={cfg.scenario.dt} "
          f"v_max={cfg.uav.v_max} max_dp={max_dp:.2f} T={cfg.scenario.T} target_speed={cfg.target.speed_range}\n" + "="*70)
    for p in (probe_action_fidelity, probe_action_fidelity_rollout, probe_belief_drift,
              probe_role_and_feasibility, probe_logprob_consistency,
              probe_double_advnorm, probe_torch_paths,
              probe_obs_ranges, probe_dt_coherence):
        try:
            p()
        except Exception as e:
            import traceback; print(f"\n[{p.__name__}] 失败: {e}"); traceback.print_exc()
    print("\n" + "="*70 + "\n汇总: 上面带 ⚠ 的为仿真确认的可疑点 (幅度见各 probe).\n" + "="*70)
