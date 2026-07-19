# 架构、数据流与张量维度

---

## 1. 模块职责与调用关系

```text
scripts/run_*          训练/评估入口:建 env、agents、trainer,跑循环
  └─ agents/trainer.py        MAPPTrainer:rollout 收集 + GAE + PPO 更新 + 评估/早停
       ├─ agents/mappo_agent.py  MAPPOAgent:act / act_batch / evaluate_actions(共享 actor+critic)
       │    └─ agents/networks.py  ActorNetwork(dp_mean/dp_log_std/role_logits)、CriticNetwork
       ├─ agents/buffer.py        RolloutBuffer:存 transition + compute_gae + 展平/归一化
       └─ environment/env_wrapper.py  UAVISACEnv(Gym 接口,reset/step/get_state/set_state)
            └─ environment/env_core.py  EnvironmentCore:一帧主循环(见 §3)
                 ├─ environment/action.py     ActionSpace:decode / log_prob / entropy
                 ├─ environment/uav.py         apply_action(移动+角色+能耗)
                 ├─ environment/target.py      CV/CA/CT 运动 + 过程噪声
                 ├─ physical/deflection.py     K×K×Q 双基地 deflection(role-agnostic 可选)
                 │    ├─ physical/geometry.py    τ/ν/α
                 │    └─ physical/channel.py     Rician + LoS/NLoS 上报可靠性(用 RNG)
                 ├─ physical/inner_solver.py   P0 贪心选择配对(单角色约束 + B3 + Safe P0)
                 ├─ physical/detection.py      D_q* → P_D、U_q
                 ├─ environment/constraints.py 安全/能量/公平/边界检查
                 ├─ environment/reward.py      团队奖励 + 边际贡献 shaping
                 ├─ environment/belief.py      Kalman 预测 + NIS 校准(CG-SR L1) + AoI
                 ├─ environment/trust_manager.py TrustManager: gate + weight cap + quarantine (CG-SR L2/L4)
                 └─ agents/neighbor_attention.py  Multi-Head Cross-Attention + CI fusion (CG-SR L2 门控)
```

执行(CTDE):训练时 critic 看全局状态(MAPPO)或局部 obs(IPPO);执行时各 UAV 只用自己的局部 obs + 共享 actor。

---

## 2. Actor 与 P0 的职责边界(最常见误解)

- **Actor(MAPPO)只输出每架 UAV 的二维位移 `Δp`**(以及 `learn_roles=True` 时的角色概率)。它决定"飞到哪"。
- **P0(inner_solver)决定"谁发谁收、和哪个目标配对"**:在当前几何下贪心选 `(tx, rx, target)` 三元组。`learn_roles=False`(默认)时角色完全由 P0 分配,Actor 不碰角色。
- 因此:位移是**学习**出来的(策略),配对是**每帧重新优化**出来的(P0)。评估时若看到某指标异常,先判断是策略问题还是 P0 输入问题。

---

## 3. 一帧执行顺序(`env_core.step`)与陷阱

严格顺序(默认 `learn_roles=False`):

```text
1.  t += 1
2.  记录移动前 UAV 位置 prev_uav_positions(供势函数 shaping)
3.  apply_action:每架 UAV 按 Δp 移动、更新速度/能耗;角色先置占位(Idle)
        → 得到 uav_positions/velocities(移动后),roles(占位)
4.  target.step():每个目标 CV 运动 + 过程噪声(目标在 UAV 移动之后才步进)
        → target_positions/velocities(移动后)
5.  deflection.compute(role_agnostic=True):用移动后的 UAV 与目标几何,枚举所有 i≠j 对 × Q
6.  inner_solver.solve(enforce_single_role=True):贪心选 (i,j,q),每 UAV 单角色
7.  从 selected_set 反推角色,写回 uav.role(供 obs/诊断/下一帧)
8.  P_D = compute_detection_probabilities(D_q*, P_FA)
9.  constraints.check_all(uav_pos, batteries, P_D) → 仅产出 info,不直接进奖励
10. team_reward = utility − λ_report·bits − 0.0;(可选)距离势 shaping;
        边际贡献 ΔU_k;shaped_rewards r_k
11. belief:先 predict(CV)+ AoI++,再对每个被选 (i,j,q) 做 Kalman 更新(真值+量测噪声)
12. next_obs = build_observations()
13. dones / 存 prev_P_D / 组 StepInfo(含配对诊断:roles, n_tx/n_rx, n_selected,
        valid_pair, no_tx, all_same_role)
```

**容易出错的几个时序点(已核对源码):**

- **先移动 UAV(3)再步进目标(4),感知(5)用的是两者移动后的几何。** 但策略决定 `Δp` 用的是**上一帧**的 obs,所以策略看到的目标位置必然比感知时刻旧一帧——这是序贯调度的固有延迟,不是 bug。
- **P0(6)使用移动后的几何**(步骤 5 的 deflection)。
- **belief 先 predict 后 update**(11):标准 Kalman 顺序。
- **reward 用当前帧的 `P_D`**(步骤 8→10)。
- **检测不是采样**:belief 更新对 P0 选中的配对一律按"观测成功"处理(传 `True` + 真值带噪),不按 `P_D` 概率掷骰。
- **obs 里的"上一帧 P_D"实际滞后一帧(off-by-one,需留意):** `next_obs` 在步骤 12 构建时用的 `self.prev_P_D` 是**上一帧**的 P_D;当前帧的 `P_D_q` 在步骤 13 才写入 `self.prev_P_D`。也就是说,用于决定 `a_{t+1}` 的 obs 里携带的是 `P_D_{t-1}` 而非 `P_D_t`。若要改成无滞后,应在 `_build_observations()` 之前更新 `self.prev_P_D`。(记录于 `KNOWN_ISSUES`。)
- **done 时存的是终止前的 transition**:`collect_rollout` 先 `store(...)` 当前 transition(obs 为决策时的旧 obs),再判断 `done_all` 并 `reset`;reset 后的新 obs 用于下一帧决策,不写进当前 transition。
- **GAE 的 `next_value`**:来自 rollout 结束后对**最终状态**(最终全局状态 / 最终局部 obs)的 critic 估值;episode 边界由 `mask=1−done` 截断(见 `TRAINING §3`)。

帧级时序(简图):

```text
obs_t ──Actor──▶ Δp ──┐
                       ├─[UAV移动]─[目标移动]─[deflection]─[P0配对]─[角色]
                       │            │
                  [P_D, 约束, 奖励] ◀┘ ──[belief predict→update]──▶ obs_{t+1}
```

---

## 4. 张量维度

### 4.1 局部观测 `obs`(每个 agent 一条,`observation.py: build_local_obs`)

设区域宽高 `aw,ah`、高度 `h`、`v_max=25`、`B_max=50000`。

| 索引(K=4,Q=2 时) | 内容 | 维度 | 归一化 |
|---|---|---|---|
| 0:3 | 自身位置 | 3 | ÷ [aw, ah, h] |
| 3:6 | 自身速度 | 3 | ÷ v_max |
| 6 | 电池 | 1 | ÷ B_max |
| 7 | 角色(标量 0/1/2,**非 one-hot,不归一化**) | 1 | — |
| 8 起,每目标 9 维 | belief.mean(4) | 4 | ÷ [aw,ah,25,25] |
| | belief.cov 对角(4) | 4 | ÷ [aw²,ah²,625,625] |
| | AoI(1) | 1 | ÷ 100 |
| 之后,每邻居 4 维(K−1 个) | 相对位置(2) | 2 | ÷ [aw,ah] |
| | 邻居角色(1) | 1 | — |
| | 邻居电池(1) | 1 | ÷ B_max |
| 末尾 | 上一帧 P_D(Q,见 §3 的 off-by-one) | Q | ∈[0,1] |

**公式:** `obs_dim = 8 + 9Q + 4(K−1) + Q = 8 + 10Q + 4(K−1)`
- K=4, Q=2 → `8 + 20 + 12 = 40`
- K=4, Q=4 → `8 + 40 + 12 = 60`

> 注意:代码注释把 role 写成 "role_onehot(1)",实际是**单个标量**(`[float(role)]`),不是 one-hot。

### 4.2 全局状态 `global_state`(`build_global_state`,MAPPO critic 用)

| 内容 | 每项维度 |
|---|---|
| 每 UAV:pos(3)÷[aw,ah,h]、vel(3)÷25、battery(1)÷B_max、role(1) | 8 × K |
| 每目标:**真实** pos(3)÷[aw,ah,1]、**真实** vel(3)÷25 | 6 × Q |
| 上一帧 P_D | Q |

**公式:** `global_state_dim = 8K + 7Q`
- K=4, Q=2 → `32 + 14 = 46`
- K=4, Q=4 → `32 + 28 = 60`

全局状态包含**真实目标状态**(privileged),局部 obs 只有 belief。

### 4.3 Critic 实际输入(`trainer`)

critic 输入 = base + agent one-hot(K 维),用于参数共享下区分 agent:
- MAPPO(`centralized_critic=True`):`global_state_dim + K`(K=4,Q=2 → 50)
- IPPO(`centralized_critic=False`):`obs_dim + K`(K=4,Q=2 → 44)

### 4.4 动作格式

`Action(delta_p: np.ndarray(2,), role: int)`(`utils/types.py`)。三处保存的内容:

| 阶段 | 保存什么 |
|---|---|
| 网络侧 | `dp_mean(2), dp_log_std(2), role_logits(3)` |
| decode 产出 | `delta_p`(已 tanh+缩放+圆盘投影)、`role`(占位或采样)、`log_prob`(连续;角色仅 `learn_roles=True`) |
| buffer 侧 | `actions_dp(2), actions_role(标量), log_probs(标量)` |
| 环境执行 | 即 buffer 的 `delta_p`(投影使 env 裁剪为 no-op,**存储==执行**) |

### 4.5 Buffer 张量(`buffer.py`,多环境交织)

buffer 行布局为 `[env0_step0, env1_step0, …, env_{N-1}_step0, env0_step1, …]`,
`buffer_size = steps_per_env × num_envs`(默认 256×8 = 2048)。

| 字段 | 形状 |
|---|---|
| `obs` | (buffer_size, K, obs_dim) |
| `global_states` | (buffer_size, global_state_dim) |
| `actions_dp` | (buffer_size, K, 2) |
| `actions_role` | (buffer_size, K) |
| `log_probs / values / rewards / dones / masks` | (buffer_size, K) |
| `advantages / returns`(GAE 后) | (buffer_size, K) |
| `h_prev` (P0 fix, StructuredActor only) | (buffer_size, K, K-1, gru_hidden_dim) |
| `per_target_rewards / per_target_values` | (buffer_size, K, Q) |
| `per_target_advantages / per_target_returns` (GAE 后) | (buffer_size, K, Q) |

`get_training_data` 把 (T,K,·) 展平成 (T·K,·),并对 advantage 做**一次全局**标准化。

### 4.6 GRU Hidden State 一致性路径 (P0 fix, 2026-07-14)

```
Rollout (collect_rollout):
  env._gru_hidden[(k,kk)] → h_prev_batch (1, N*K*(K-1), D)
    → actor(obs, h_prev_batch) → h_new
      → store h_prev per-env to buffer  ← NEW
      → store h_new back to env._gru_hidden

Update (update):
  buffer.get_training_data() → data['h_prev']  ← NEW
    → per-minibatch: reshape to (1, mb*(K-1), D) .to(device)
      → actor(obs, mb_h_prev)  ← NEW (was actor(obs) before fix)

Consistency assertion (before first optimizer step):
  verify_old_log_prob_consistency(obs, actions, old_lp, h_prev):
    recompute log_prob with stored h_prev
    assert max|old_lp - recomputed_lp| < 1e-4
```

### 4.7 PD_hist 数据流 (P1 fix, 2026-07-14)

```
ObservationBuilder.build_local_obs:
  prev_P_D (Q,) → obs_parts.append → flat obs

StructuredActorNetwork._parse_one:
  pd_hist = obs[:, ptr:ptr+Q]  → RETURNED (was discarded before fix)

StructuredActorNetwork.forward:
  pd_last = pd_hist[..., -1]   (last timestep, shape (B, Q))
  pd_feat = pd_hist_proj(pd_last.unsqueeze(-1))  (B, Q, D)
  te = target_enc(targets) + pd_feat               ← residual modulation

Note: PD_hist in local obs is assumed to be the UAV's LOCAL detection
confidence. If changed to global fused P_D from fusion centre, the
communication cost (delay, bits, AoI) MUST be accounted for.

### 4.8 Checkpoint 兼容性：zero_init_new_layers (P0 fix, 2026-07-14)

加载旧 DAgger checkpoint 时，新增层（如 pd_hist_proj）不在 checkpoint 中，
`load_state_dict(..., strict=False)` 让它们保持 `_init_weights()` 的随机正交初始化。
这会改变策略，使 "DAgger baseline" 不再等于真正的 DAgger。

```python
actor.load_state_dict(old_ckpt, strict=False)
actor.zero_init_new_layers(known_keys=set(old_ckpt.keys()))
# → pd_hist_proj.weight ← 0, pd_hist_proj.bias ← 0
# → e_{kq} = e_{kq}^{base} + 0 = e_{kq}^{base}  (严格复现)
```

### 4.9 Streaming GRU 评估路径 (P0 fix, 2026-07-14)

```
_evaluate() per episode:
  eval_h_prev = None  ← zero-init each episode
  for each frame:
    actor(obs, eval_h_prev) → dp_mean, ..., h_new
    eval_h_prev = h_new  ← maintain across frames
```

此前评估调用 `actor(obs)` 每帧从零初始化 GRU（与 rollout 不一致）。
现在评估的 recurrent 路径与 rollout 完全一致。

### 4.10 Actor `detach_h_new` 参数 (2026-07-14)

`StructuredActorNetwork.forward(detach_h_new=True)`：
- **True（默认）**：返回 `hn.detach()`，用于 rollout、evaluation 和 PPO 单步更新。
  梯度不跨帧传播——每帧独立反向。
- **False**：返回 `hn`（保留计算图），用于 DAgger chunk BPTT 训练。
  梯度在 chunk 内部跨帧传播（最多 L 帧）。调用者必须在 chunk 边界显式 `detach()`。

DAgger chunk BPTT 时序：
```
episode: h=0
  chunk 0 [0:L]:   actor(obs, h, detach_h_new=False) → ... → h_end
                    h_next = h_end.detach()  ← chunk boundary
  chunk 1 [L:2L]:  actor(obs, h_next, detach_h_new=False) → ...
                    h_next = h_end.detach()
  ...
  loss.backward()   ← once per episode
  optimizer.step()
```

### 4.11 single_frame_dim 动态检测 (P1 fix, 2026-07-14)

`StructuredActorNetwork._parse_obs()` 不再硬编码 `single_dim=227`。
改为从 `self.single_frame_dim` 读取（0 = 自动检测为 obs_dim）。

设置优先级：
1. 构造时显式传入 `single_frame_dim`（MAPPOAgent → StructuredActorNetwork）
2. Trainer 自动从 `env.core.obs_builder.get_single_frame_dim()` 检测
3. 兜底：`obs_dim` 本身（单帧场景）

这使得多帧 stacking 解析自动适配 K=4/8、Q=4/8、P0 有无等不同配置。

### 4.12 NIS 校准 + 自适应 Q 数据流 (CG-SR Layer 1, 2026-07-19)

`BeliefManager` 新增 NIS 状态机、自适应 Q、和协方差校准：

```
update_after_observation (Joseph form, PSD-preserving):
  S = cov + R
  ν = z − mean
  nis = ν^T S^{-1} ν
  r̄ ← (1−ρ)r̄ + ρ·(nis/d_z)              (d_z=4, ρ=0.1 ≈ 10帧窗口)

  K = cov @ inv(S)
  mean⁺ = mean + K @ ν
  I_KH = I - K
  cov⁺ = I_KH @ cov @ I_KH^T + K @ R @ K^T    // Joseph form
  cov⁺ = 0.5 * (cov⁺ + cov⁺^T)                // symmetrize

step (predict, with adaptive Q):
  Q_eff = q_scale · Q_0
  P_raw = F·P·F^T + Q_eff

  // NIS state machine (hysteresis):
  NORMAL  → SUSPECT    when r̄ ≥ 1.3 for 3 frames
  SUSPECT → RECOVERING when r̄ <  1.1 for 5 frames
  RECOVERING → NORMAL  when r̄ <  1.1 for 5 frames

  // Adaptive-Q:
  SUSPECT: q_scale → min(r̄, 100)     (EMA α=0.05)
  NORMAL:  q_scale → 1.0            (slow decay)

  // Linear multiplicative inflation:
  λ = 1 + k·max(r̄−1, 0)    clamped to [1, 5.0]
  P_cal = λ · P_raw + δI             // δI = diag[σ²_pos, σ²_pos, σ²_vel, σ²_vel]
```

关键设计：
- **Joseph form** 保证 PSD：`(I-KH)P(I-KH)^T + KRK^T` 替代简化 `(I-KH)P`
- λ 用**线性** (非指数) multiplicative：`1 + k·(r̄−1)` 防止过度膨胀
- 下限用 `np.maximum(diag, floor)` 而非 `+=`
- **自适应 Q** 从根源修正过程噪声（不只是修补协方差）
- 滞回状态机防止单次异常反复触发

### 4.13 TrustManager 门控数据流 (CG-SR Layer 2, 2026-07-19)

纯 numpy 模块，在 `env_core._fuse_beliefs_attention()` 中调用：

```
compute_gate_weights:
  g_nis(i) = exp(−max(0, nis_ema_i − 1))
  d_ijq = (x̂_i−x̂_j)^T (P_i+P_j)^{-1} (x̂_i−x̂_j)    Mahalanobis
  g_age(j) = max(0, 1 − aoi_j/aoi_max)
  τ = g_nis(i)·g_nis(j)·exp(−d_ijq/2)·g_age(j)
  τ ← (1−ρ)τ + ρ·τ_new

门控策略（二值，非连续缩放）:
  if quarantined: weight = 0
  else:          weight = min(τ, ω_max)   (ω_max=0.6)
  renormalize so Σω ≤ 1 − local_weight_min

观测后信任反馈 (Layer 4):
  T_new = exp(−nis_fused/(2·d_z))
  τ ← (1−ρ)τ + ρ·T_new
  if nis_fused > ratio·nis_local: quarantine(duration=10)
```

### 4.14 Safe P0 数据流 (CG-SR Layer 3, 2026-07-19)

**Dual-score Safe P0** — 低置信度时完整回退 local belief 几何 (非仅关 B3)：

```
Step 1: 选择 ranking 几何源
  conf_q = mean(τ_{kjq} over all k≠j)
  if any(conf_q < 0.3):
      ranking_entries = deflection_from(local_belief_mean)   // safe fallback
  else:
      ranking_entries = deflection_from(fused_belief_mean)   // trusted

Step 2: B3 scoring (per candidate in greedy loop)
  if fusion_confidence[e.q] ≥ confidence_min (0.3):
      apply B3: gain −= β·√(cov)/100 + η·AoI
  // else: B3 disabled, B0 baseline only
```

关键设计：
- **几何回退优先**：低置信时 P0 的基础 deflection 来自 local belief，不依赖 fused
- B3 bonus 是增量优化——仅在基础几何可信时开启
- B3 的 cov/AoI 输入始终来自 local BeliefManager (非 fused)

### 4.15 Event-Triggered Probe 数据流 (CG-SR Layer 4, 2026-07-19)

`EnvironmentCore.step()` 在 P0 求解后、belief 更新前插入。**事件触发** (非固定周期)：

```
Per-frame probe score:
  G_q = w_aoi·AoI_q + w_cov·tr(P_q) + w_nis·max(NIS_q)
      + w_miss·consecutive_miss_count_q
      + w_disagree·mean(disagreement[:,:,q])

Event trigger:
  if max(G_q) > threshold (3.0):
      q* = argmax(G_q)
      if q* not selected by P0:
          force-pair: find best (tx,rx) with d_eff>0 for q*
          reset miss_count[q*] = 0

Miss tracking:
  for each target q:
      if q in P0 selected: miss_count[q] = 0
      else:                miss_count[q] += 1
```

信任反馈 (观测后):
```
for each selected (i,j,q) where detected:
  nis_i = belief_mgr._last_nis[i,q]
  nis_j = belief_mgr._last_nis[j,q]
  trust_manager.update_trust_from_nis(i,j,q, nis_i, nis_j)
  trust_manager.check_quarantine(i,j,q)
trust_manager.decay_quarantine()
```

关键设计：
- **事件触发**: Easy/Medium 基本不触发，Hard 漂移累积后触发，不浪费正常 sensing 资源
- **Miss count**: 连续漏检目标获得递增优先级，打破 "不被选→无观测→更不被选" 死循环
- **Disagreement bonus**: 节点间争议大的目标更需独立验证
