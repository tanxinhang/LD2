# 系统数学模型与符号—代码映射

本章统一定义数学符号,并给出与代码变量的一一对应。所有公式都对应到具体文件,便于核对。

---

## 0. 系统边界(做什么 / 不做什么)

**系统负责:** 多 UAV 轨迹决策;TX/RX 角色分配(默认由 P0,见 §3.3);双基地感知配对;多目标检测概率;能量/通信/安全/公平约束;目标 belief 的预测与 Kalman 更新。

**当前不负责(刻意抽象掉):** 完整 OTFS 调制/解调;原始 IQ 波形与回波处理;真实通信协议栈;多目标数据关联(本项目按目标索引 `q` 直接关联,无关联歧义);硬件飞控接口。28 GHz 等射频参数**确实进入**路径损耗、多普勒、波长、天线增益、OTFS DD 网格的解析计算(见 §3.4),但**不**生成时域波形。

---

## 1. UAV 状态

```
s^UAV_{k,t} = [ p_{k,t}, v_{k,t}, E_{k,t}, r_{k,t} ]
```

- `p_{k,t}`:位置 (x,y,z),单位 m。`z` 固定为飞行高度 `height`(默认 20 m),只有 (x,y) 受 Δp 改变。
- `v_{k,t}`:速度 (vx,vy,0),单位 m/s,由位移估计 `v = Δp/dt`。
- `E_{k,t}`:剩余电量,单位 J。
- `r_{k,t}`:角色 ∈ {0=TX, 1=RX, 2=Idle}。
- 更新顺序:先按动作移动并(由 P0)确定角色,再步进目标——见 `ARCHITECTURE §3`。
- 边界:越界做软反弹(`uav.py: apply_action`)。

代码:`uav.py`(`UAV`),状态快照 `UAVState`(`utils/types.py`)。

---

## 2. 目标状态

```
滤波/belief 状态:  x_{q,t} = [ x, y, vx, vy ]^T        # 4D 平面 CV 状态(Kalman)
仿真真值表示:        position = [x, y, 0],  velocity = [vx, vy, 0]   # 3D(z=0 平面)
```

> 维度说明:目标在仿真里以 **3D**(z=0)位置/速度参与几何计算(`get_position_3d`),而 belief/Kalman 用 **4D** 平面状态 `[x,y,vx,vy]`。两者一致(z 恒 0),全局状态里写的 "pos(3)+vel(3)" 即 3D 真值表示,不是维度矛盾。

- 运动:常速度模型 + 加速度过程噪声 `sigma_a`(默认 0.5 m/s²),`target.step()` 用 `rng.multivariate_normal` 采样。
- 速度范围 `speed_range`(默认 [0,5] m/s,慢目标课程)。
- 边界处理:区域内反弹。

代码:`target.py`(`Target`),`TargetState`(`utils/types.py`)。

---

## 3. 决策与物理映射

### 3.1 动作

```
a_{k,t} = ( Δp_{k,t}, r_{k,t} )
```

链路(`action.py: ActionSpace`):
1. 网络原始输出:`dp_mean`(2),`dp_log_std`(2,共享参数,经 tanh 映射到 [-1,1] 的 log-std),`role_logits`(3)。
2. 采样并 tanh 压缩:`dp01 = clip(tanh(N(dp_mean, σ)), ±0.999)`,再乘 `dp_scale = max_dp = v_max·dt`(默认 2.5 m)。
3. **径向投影**:若 `‖Δp‖ > max_dp`,投影回圆盘 `Δp·(max_dp/‖Δp‖)`。这使**存储动作 == 环境执行动作**(env 的裁剪成为 no-op),从而 `old_log_prob == new_log_prob`(修复历史 P1/P5 bug)。
4. 确定性(评估)位移:`Δp = tanh(dp_mean)·dp_scale` 再投影(不做 ±0.999 预裁剪,以与历史评估一致)。
5. 角色:`learn_roles=False`(默认)时,decode 输出占位 Idle(2),真实角色由 P0 在环境内分配;`learn_roles=True` 时,确定性取 argmax、随机取 Categorical 采样。
6. log-prob:连续部分是 tanh-squashed Gaussian(含 Jacobian 修正 `-Σ log(1-dp01²)`);角色项**仅在 `learn_roles=True` 时**计入;两路都确定时返回 0.0。

### 3.2 双基地链路

三元组 `(i, j, q)` = (发射 UAV i, 接收 UAV j, 目标 q),要求 `i ≠ j`。

> **几何用的是目标真值,不是 belief(重要,已核对)。** `env_core.step` 把 `target_positions = [t.get_position_3d()]`(**真实**位置)传给 `deflection.compute` 和 `inner_solver.solve`。因此**内层 P0 调度是一个 oracle 调度器**:它按真实目标几何选配对,belief 误差不影响配对选择。信息可用性现状:**Actor 只看 belief,Critic 看真值(CTDE 特权),P0 也看真值。** 部署场景不具备 P0 的真值条件,这会影响模型的科学可信度。**现已提供两模式开关**:默认 oracle(真值排序,上界);`marl.p0_uses_belief=True` 切 deployable(P0 在融合 belief 上排序,实现 P_D 仍用真值几何)。详见 `KNOWN_ISSUES B6` 与 `EXPERIMENTS`。

对每条链路计算(`geometry.py`,`deflection.py`):
- 双基地距离 `R = ‖p_i − x_q‖ + ‖x_q − p_j‖`(tx→目标→rx)。
- 时延 `τ`、多普勒 `ν`(tx/目标/rx 三段贡献,`× fc/c`)。
- 路径增益 `α`(Friis,`∝ λ²·σ_rcs / (R_tx²·R_rx²)`,含天线增益)。
- 原始 deflection `d_raw`(感知功率/噪声功率,含 CPI 积累 `n_cpi`)。
- DD 有效性 `g_dd`(需 ≥ `g_min`,否则该链路 `d_eff=0`)。
- 上报链路可靠性 `χ_rep`(Rician + 可选 Al-Hourani LoS/NLoS,`channel.py`,**消耗 RNG**——见 `KNOWN_ISSUES` 的双 RNG 流提醒)。
- 有效 deflection `d_eff = χ_rep · d_raw`(当 `g_dd ≥ g_min`,否则 0)。

### 3.3 角色分配(默认 P0 承担)

`learn_roles=False`(默认):deflection 以 **role-agnostic** 模式枚举所有 `i≠j` 有序对(每架 UAV 既是候选 TX 也是候选 RX),P0 在选择时施加**每 UAV 单角色**约束(一架不能同帧既 TX 又 RX),选完后从 `selected_set` 反推每架 UAV 的角色写回。
`learn_roles=True`:沿用旧设计,角色由策略输出,deflection 按 `role==0/1` 门控。
这条边界很关键:**Actor 只决定"飞到哪",P0 决定"谁发谁收、配对哪个目标"**。

### 3.4 检测模型(deflection → P_D)

每目标累积 deflection `D_q* = Σ_{(i,j) 被选} d_{ijq}`(各链路 deflection 直接相加,模块化)。

```
P_D^q = Q( Q^{-1}(P_FA) − sqrt(D_q*) )          # math_utils.compute_PD
U_q   = − log( 1 − P_D^q + eps )                # math_utils.utility_from_D
```

- `P_FA`:虚警率,配置给定(默认 1e-3),全局常数。
- 多链路 deflection **相加累积**(等价假设链路在 deflection 统计量上可加)。
- **检测用的是概率值 `P_D`,不做随机采样**。
- belief 更新用的是 **P0 选中的配对**(`selected_set`):对每个选中的 `(i,j,q)`,把目标 `q` 当作被成功观测(`update_after_observation(..., True, true_state)`,带量测噪声),**与 `P_D` 无关**——即便 `P_D=0.05`,只要被选中,belief 照常更新。这是一个**乐观观测假设**(选中即成功观测),不是按检测概率掷骰。**现已提供 Bernoulli 检测门控开关**:`marl.belief_detection_sampling=True` 时按 `δ_q~Bernoulli(P_D,q)` 决定是否更新。详见 `KNOWN_ISSUES B7`。

> **效用函数的凹凸性与 P0 的近似保证(重要更正,已数值核对)。**
> `U(P_D) = −log(1−P_D)` 关于 `P_D` 是**严格凸**(`U''=1/(1−P_D)²>0`),**不是凹**。
> 更关键的是把它复合上 `P_D(D)` 后:经数值检验,`U(D) = −log(1−P_D(D))` 在相关 `D` 区间内**单调递增但并非凹**(`U''>0` 占约 99.6% 的区间),且对固定 `d_eff` 的边际增益 `ΔU(D_current)` 随 `D_current` **递增而非递减**。
> 因此**不能**声称 P0 目标是"单调次模"或贪心有近似保证。P0 当前应被准确描述为:**基于边际检测效用的启发式贪心调度**(见 `TRAINING §6`)。
> 若要恢复次模/近似保证,需换成对 `D` 真正凹且饱和的效用,例如 `U(D)=1−exp(−κD)`(其 `U''=−κ²exp(−κD)<0`),并重新验证单调性、凹性与对加性 deflection 的次模性。这是 `KNOWN_ISSUES B8` 的开放项。

### 3.5 优化问题

外层(策略 π)+ 内层(选择集合 S_t)联合:

```
max_{π, {S_t}}  E[ Σ_t γ^t · R(s_t, a_t, S_t) ]
```

约束(`constraints.py`,detection/p0_solver 配置):
- 轨迹:`‖Δp‖ ≤ v_max·dt`(由动作投影保证)。
- 安全:任意两 UAV 距离 ≥ `d_safe`(默认 20 m)。
- 能量:电量 ≥ 0。
- 边界:UAV 在区域内。
- 检测公平:每目标 `P_D^q ≥ P_D_min`(默认 0.2)。
- P0 选择约束:接收容量 `capacity_per_rx`、上报时延 `latency_max`、每目标最大配对数 `K_q_max`;`learn_roles=False` 时额外的每 UAV 单角色约束。

**MAPPO 解外层(轨迹/长期回报),P0 解内层(给定几何下的最优配对)。** 约束的处理方式见 `TRAINING §4`(奖励里只有 `utility − λ_report·bits`;违反通过 Lagrangian 以二值 `any_violation` 惩罚,`constraints` 的解析 penalty 当前不直接进奖励)。

---

## 4. 符号 — 代码变量 — 形状 — 单位 映射表

| 数学符号 | 代码变量 | 形状 | 单位 | 文件 |
|---|---|---|---|---|
| p_k | `uav.pos` / `UAVState.pos` | (3,) | m | `uav.py` |
| v_k | `uav.vel` / `UAVState.vel` | (3,) | m/s | `uav.py` |
| E_k | `uav.battery` | scalar | J | `uav.py` |
| r_k | `uav.role` / `Action.role` | scalar∈{0,1,2} | — | `uav.py`,`types.py` |
| Δp_k | `Action.delta_p` | (2,) | m/帧 | `action.py` |
| x_q | `target.state` | (4,) | m, m/s | `target.py` |
| τ,ν,α | `tau, nu, alpha` | (K,K,Q) | s, Hz, 无量纲 | `geometry.py` |
| d_eff(i,j,q) | `DeflectionEntry.d_eff` | 每条目 | 无量纲 | `deflection.py`,`types.py` |
| D_q* | `p0_solution.D_q_star` | (Q,) | 无量纲 | `inner_solver.py` |
| P_D^q | `info['P_D_q']` / `StepInfo.P_D_q` | (Q,) | 概率∈[0,1] | `detection.py` |
| U_q | `p0_solution.U_q` | (Q,) | 无量纲 | `detection.py` |
| x_{ijq}(选择) | `p0_solution.selected_set` | list[(i,j,q)] | — | `inner_solver.py` |
| x̂_{k,q} | `BeliefState.mean` | (4,) | m, m/s | `belief.py` |
| Σ_{k,q} | `BeliefState.cov`(对角入 obs) | (4,4) | 混合 | `belief.py` |
| AoI_{k,q} | `BeliefState.aoi` | scalar | 帧 | `belief.py` |
| 约束惩罚 | `constraint_info['*_penalty']` | scalar | 归一化 | `constraints.py` |
| 违反标志 | `constraint_info['any_violation']` | bool | — | `constraints.py` |
| r_team | `StepInfo.team_reward` | scalar | — | `reward.py` |
| r_k(shaped) | `StepInfo.shaped_rewards[k]` | scalar | — | `reward.py` |
| ΔU_k | `marginal[k]`(内部) | scalar | — | `reward.py` |

形状中的 `K`=UAV 数,`Q`=目标数(默认 4, 2)。
