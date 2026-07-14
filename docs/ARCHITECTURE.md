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
                 ├─ environment/target.py      CV 运动 + 过程噪声
                 ├─ physical/deflection.py     K×K×Q 双基地 deflection(role-agnostic 可选)
                 │    ├─ physical/geometry.py    τ/ν/α
                 │    └─ physical/channel.py     Rician + LoS/NLoS 上报可靠性(用 RNG)
                 ├─ physical/inner_solver.py   P0 贪心选择配对(单角色约束可选)
                 ├─ physical/detection.py      D_q* → P_D、U_q
                 ├─ environment/constraints.py 安全/能量/公平/边界检查
                 ├─ environment/reward.py      团队奖励 + 边际贡献 shaping
                 └─ environment/belief.py      Kalman 预测 + 观测更新 + AoI
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

`get_training_data` 把 (T,K,·) 展平成 (T·K,·),并对 advantage 做**一次全局**标准化。
