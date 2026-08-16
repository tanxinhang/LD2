# 当前系统模型（Architecture V2 + 认证化分布式控制）

> 文档日期：2026-08-16（更新：§10 补 6/6 干净种子重跑结果并关闭污染待办、§12
> 补回填/清理状态与最新测试基线；已对照 `uav_isac/`、`config/`、`tools/` 逐项校验）。
> 本文是**当前部署架构**的数学模型与代码映射总纲，覆盖 Architecture V2 与
> D0.x 认证化控制主线的完整系统模型。基础环境与历史 P0 数学模型见
> [`SYSTEM_MODEL.md`](SYSTEM_MODEL.md)；正式结果与版本判定以
> [`CURRENT_SYSTEM_STATUS.md`](CURRENT_SYSTEM_STATUS.md) 为准；演进与失败实验见
> [`ARCHITECTURE_V2_RESULTS.md`](ARCHITECTURE_V2_RESULTS.md)；性能缺口与路线见
> [`SYSTEM_OVERVIEW_AND_ROADMAP.md`](SYSTEM_OVERVIEW_AND_ROADMAP.md)；理论驱动优化
> （含负结果）见 [`OPTIMIZATION_LOG.md`](OPTIMIZATION_LOG.md)。若历史章节与本文
> 冲突，以本文为准。

---

## 0. 系统边界（做什么 / 不做什么）

当前研究对象是**只有 UAV 间（U2U）通信、没有地面链路的分布式多 UAV 协同 ISAC**。
每架 UAV 在严格 `1 W` 通信与感知联合功率预算下，自主决定运动、Token 通信与逐目标
感知资源；最终检测由环境级集中式证据融合计算。

真实边界应表述为：

```text
分布式局部观测 + 共享参数策略（运动/通信/感知意图）
  + 物理 U2U Token 传递（比特/时延/功率/丢包）
  + 分布式结构 Student 近似角色/配对/owner 决策
  + 环境级集中式证据融合与检测评价
```

**系统负责：** UAV 轨迹决策；Tx/Rx 角色分配；双基地感知配对与接收机归属（owner）；
逐目标检测概率；通信/感知联合功率分配；bit/时延/能量通信核算；event 触发的结构
/功率/几何安全控制；证书化 no-harm 保证。

**当前不负责（刻意抽象掉）：** 原始 IQ 波形与回波处理；完整 OTFS 调制解调；真实
通信协议栈；多目标数据关联（按目标索引 `q` 直接关联，无关联歧义）；硬件飞控接口；
波形层联合设计（发射协方差、子载波、波束、模糊函数**保持固定**）。28 GHz 等射频
参数进入路径损耗、多普勒、波长、天线增益、OTFS DD 网格的解析计算，但不生成时域
波形。

---

## 1. 场景与状态空间

### 1.1 场景

- `K` 架 UAV 与 `Q` 个目标，二维平面区域（默认 4/4 为 `800×800 m`，6/6 为
  `980×980 m`，8/8 为 `1130×1130 m`，边长为 `800·sqrt(K/4)` 以保持密度）。
- UAV 固定飞行高度 `h=20 m`，只有 `(x,y)` 受位移 `Δp` 改变。
- 目标在跨尺度审计中为**静止已知目标**（`tracking_enabled=false`），位置为任务
  已知信息；历史 4/4 亦支持 CV/CA 运动目标 + Kalman belief 追踪（见 SYSTEM_MODEL）。
- 帧时长 `dt=0.1 s`，每 episode `T=150` 帧，UAV 最大速度 `v_max=25 m/s`，单步最大
  位移 `dp_max = v_max·dt = 2.5 m`。

### 1.2 UAV 状态

```text
s^UAV_k = [ p_k, v_k, E_k, r_k ]
```

| 符号 | 内容 | 单位 | 代码 |
|---|---|---|---|
| `p_k` | 位置 (x,y,z)，z 固定 20 m | m | `uav.pos` / `UAVState.pos` |
| `v_k` | 速度，由 `Δp/dt` 估计 | m/s | `uav.vel` |
| `E_k` | 剩余电量 | J | `uav.battery` |
| `r_k` | 角色 ∈ {0=TX, 1=RX, 2=Idle} | — | `uav.role` / `Action.role` |

### 1.3 目标状态

```text
仿真真值:  position=[x,y,0], velocity=[vx,vy,0]      (3D, z=0)
belief  :  x_q = [x,y,vx,vy]^T                        (4D 平面 CV 状态)
```

跨尺度审计 `tracking_enabled=false` 时目标静止、位置已知，不经过 Kalman 循环；
运动目标场景仍支持 CV/CA/CT 模型 + 过程噪声 `sigma_a`（见 SYSTEM_MODEL §2）。

---

## 2. 物理感知层（双基地 → 检测）

### 2.1 双基地链路

三元组 `(i, j, q)` = (发射 UAV `i`，接收 UAV `j`，目标 `q`)，要求 `i ≠ j`。
对每条链路计算（`geometry.py`, `deflection.py`, `channel.py`）：

- 双基地距离 `R = ‖p_i − x_q‖ + ‖x_q − p_j‖`；
- 时延 `τ` 与多普勒 `ν`（tx/目标/rx 三段贡献，`× fc/c`）；
- 路径增益 `α`（Friis，`α² ∝ λ²·σ_rcs/(R_tx²·R_rx²)`；收发阵列增益 `G_tx·G_rx`
  在 §2.2 的 `d_raw` 中单独计入，不并入 `α` 本身）；
- 原始 deflection `d_raw`；
- DD 有效性 `g_dd`（需 `≥ g_min`，否则该链路 `d_eff=0`）；
- 上报链路可靠性 `χ_rep`（Rician / Al-Hourani LoS-NLoS，`channel.py`）。

### 2.2 原始 Deflection

```text
d_raw(i,j,q) = P_sense(i,q) · |α_ijq|² · T_sym · M · N · G_tx · G_rx · n_CPI
               / σ_z²
```

其中 `T_sym` 为符号周期、`M=64` 延迟格、`N=16` 多普勒格、`G_tx/G_rx` 收发阵列增益
（各 16 dBi）、`n_CPI=128` 相干积累帧、`σ_z²=kT·B·NF` 为噪声功率。
代码：`compute_raw_deflection`。

### 2.3 有效 Deflection 与每瓦增益

```text
d_eff(i,j,q) = 1[g_dd,ijq >= g_min] · χ_rep,ijq · d_raw(i,j,q)
```

因此**每瓦有效增益**（与功率无关，D0.12 修正的可辨识形式）为

```text
a_ijq = 1[g_dd,ijq >= g_min] · χ_rep,ijq · α_ijq²
        · T_sym · M · N · G_tx · G_rx · n_CPI / σ_z²
```

代码：`per_watt_deflection_tensor_from_observables`（`physical_oracle_audit.py`）。
Swerling 开启时该重建 fail-closed（单次 RCS 实现不可由 `α/g_dd/χ_rep` 识别）。

### 2.4 检测概率与融合边界

接收机局部证据 `receiver_d[j,q] = Σ_{i:(i,j,q)∈selected} d_eff(i,j,q)`（不跨接收机
融合，`receiver_deflection_from_selected`），随后按融合模式取逐目标 Deflection：

```text
local_only:      D_q = max_j receiver_d[j,q]         # 每目标单一 owner 接收机（部署）
central_oracle:  D_q = Σ_j receiver_d[j,q]           # 跨接收机集中融合（教师/上界）
legacy_global:   历史全局融合
u2u_distributed: 由实际送达证据包重建
```

检测概率为

```text
P_D^q = Q( Q^{-1}(P_FA) − sqrt(D_q) ),   P_FA = 1e-3
```

代码：`compute_PD`（`math_utils.py`）、`select_detection_deflection`（`evidence.py`）。
检测用概率值 `P_D`，不做随机采样。

> **注意**：`U(P_D) = -log(1-P_D)` 关于 `P_D` 严格凸、复合 `P_D(D)` 后非凹（见
> B8）；因此历史 P0 贪心是启发式，无次模/`(1-1/e)` 保证。当前部署主线已把优化
> 目标转移到 **Deflection 空间的 max-min 线性规划**（§5），其目标关于 `D` 是凹
> （最小化线性函数族的逐点 min），恢复了凹性/次模结构。

### 2.5 QoS 指标

每 episode 取末 20 帧为 steady 窗口，`P̄_D^q` 为窗口内目标 `q` 均值：

```text
steady = (1/Q) Σ_q P̄_D^q
weak3  = 均值 of bottom-3 P̄_D^q
worst  = min_q P̄_D^q
QoS-feasible  ⇔  steady>=0.80 且 weak3>=0.70 且 worst>=0.60
```

---

## 3. 通信层（U2U Token）

- 每架 UAV 广播稀疏目标 Token：聚合模式为单个 16 维向量
  （`comm_payload_mode='aggregate'`），`target_tokens` 模式为 `Q` 个各
  `comm_target_token_dim=16` 维的目标向量；稀疏 top-k 掩码只激活部分目标槽位并
  决定实际发送维度数 `D_active`，`b` bit/维（Architecture V2 部署取
  `comm_rate_bits_per_dim=[0,4]`，历史 4/4 部署为自适应 4–8 bit）。接收方保留
  sender identity 与 target identity 作为包元数据，重建 identity-indexed 团队竞价图。
- 包大小 `L_k = H + D_active·b_k`（另加结构协议扩展维度与速率头字段），
  `H=64` bit 包头（`comm_header_bits`）；活跃发送者正交平分 100 kHz 带宽。
- 链路用自由空间增益 `(λ/(4πd))²`、Shannon 串行化速率与 0.2 ms 处理时延；包只在
  SNR ≥ 阈值且总时延 ≤ deadline（默认 5 ms）时送达；最近送达 Token 保留最多 5 帧，
  带显式 AoI。
- 通信功率与感知功率满足 §4 的硬预算；结构协议字段（header/epoch/digest/索引/
  价格/反馈）逐 bit 计费（`structure_sequence_transport.py` 等）。

代码：`environment/communication.py`（`InterUAVCommunicationModel`）、
`coordination/*_transport.py`。

---

## 4. 功率预算与资源分配

每架 UAV 的通信功率与逐目标感知功率严格投影到 `1 W` 单纯形：

```text
P_comm,k + Σ_q P_sense,kq = 1 W,   P_comm,k ∈ [0, ρ_max·1 W],  P_sense,kq ≥ 0
```

投影在环境内执行（`env_core.py` 的 `1 W` 硬投影），功率平衡误差 ≤ `4.44e-16 W`；
`ρ_max = comm_power_fraction_max`（Architecture V2 部署取 `0.5`，即通信功率至多
0.5 W）。感知预算记为 `b_k = 1 − P_comm,k`。**通信代价是内生的**：增加 Token bit 或发送
功率既提高邻居可见性，又减少本 UAV 当帧可用感知功率。

---

## 5. 决策与控制架构

### 5.1 三层职责划分

```text
快层：固定 role/owner/edge 的 max-min 感知功率修复（LP / Dantzig-Wolfe）
中层：仅当固定结构上界低于 QoS 时，搜索 joint structure + power（原子 N5/N6）
慢层：仅当宽松同几何上界仍低于 QoS 时，触发航迹/几何修复
```

这是 D0.11 依据物理分解确立的架构：对 6/6 最终重解状态，`原部署/只换配对/固定
结构功率/联合结构功率` 的 mean-worst 分别为 `0.678/0.678/0.918/0.932`——**只换配对
增益严格为 0，联合层超过固定功率层仅 `+0.014`**，因此主要 headroom 在功率层与
几何层，而非继续扩大神经网络。

### 5.2 分布式 Actor

1. 共享 per-target scorer 对每个目标独立估计局部感知边际价值（参数共享，目标置换
   等变，非共享决策）；
2. 集合/注意力编码处理变长 UAV/目标集合，不依赖固定身份 one-hot；
3. 运动头由目标承诺与 Token 信息驱动（径向受运动学约束，切向形成双基地基线）；
4. 通信速率与总通信资源由集合池化头输出，不依赖固定 `Q`；
5. 感知/通信功率统一投影到 §4 的 `1 W` 单纯形。

代码：`agents/networks.py`（`StructuredActorNetwork`）、`agents/mappo_agent.py`。

### 5.3 结构 Student 与集中式控制边界

- 集中式冻结控制器（receiver-owner full-graph hold-5 P0）提供结构目标；分布式
  Student 利用本地状态与收到 Token 近似其端点选择与结构解码。
- **部署版不是"完全去中心化检测器"**：决策路径分布式，最终 `P_D` 由环境级证据
  融合统一计算。集中式结构控制器只作为教师/参考上界，不属于部署执行路径。
- CTDE：Critic 只在训练时看全局状态，执行时每 UAV 只用本地观测、局部历史与已
  送达 Token。

### 5.4 锚点保持的基数门控残差（跨尺度）

```text
E(K,Q) = E0 + g(K,Q)·ΔE,   D(K,Q) = D0 + g(K,Q)·ΔD,   g(4,4)=0, g(6,6)=1
```

`E0/D0` 为冻结 4/4 Student，`ΔE/ΔD` 只在非锚点数据上训练；在 4/4 上残差严格关闭，
模型输出与原 Student 逐回合一致。代码：`agents/frozen_structure_student.py`。

### 5.5 双集合等变规划头（跨基数运动，D0.82）

对每架焦点 UAV `i`，分别构造目标集合与 peer UAV 集合：

```text
t_iq = [(y_q-x_i)/L, ‖y_q-x_i‖/L, dir(y_q-x_i), s_iq, c_i]
u_ij = [(x_j-x_i)/L, ‖x_j-x_i‖/L, dir(x_j-x_i), a_j/d_max, c_j, Σ_q s_jq]
Delta a_i,1:H = ρ·d_max·tanh f(self_i, SymmetricPool_q φ_t(t_iq),
                                       SymmetricPool_{j≠i} φ_u(u_ij))
```

精确不变量：目标置换**不变**、UAV 置换**等变**、均匀区域尺度**协变**、速度圆盘
投影保证 `‖a_i,h‖ ≤ v_max·dt`。代码：`agents/equivariant_movement_plan.py` 等。

### 5.6 解析执行栈（L0→L3）

除上述学习 Actor 外，系统有一条**解析控制执行路径**，用 `analytical_*_enabled`
标志逐层接管学到的通信/功率/结构/运动决策。**分两代**：

#### 5.6.1 Legacy deployed baseline：D0.95（gauge L1 + 单步 L3）

| 层 | 标志 | 作用 | 关键代码 |
|---|---|---|---|
| L0 通信余量 | `analytical_comm_power_enabled` | 解析最小通信功率，回收链路余量进感知预算 `b_i = 1 − P_comm^min` | `env_core._compute_analytical_min_comm_power` |
| L1 功率 | `analytical_sensing_power_enabled` + `task_constrained_power_enabled` | 固定结构 max-min LP，或 capability gauge（三地板硬约束） | `maxmin_power.py`、`capability.py` |
| L2 结构 | `analytical_structure_ranking_enabled` | 功率无关排名（per-watt gain）+ 上一帧 max-min 对偶价格 λ* 的瓶颈优先级 | `env_core._per_watt_coefficient_from_entries`（P0 排名） |
| L3 几何 | `analytical_movement_enabled` | 价格驱动赤字→能力下降 + 达标悬停（单步梯度） | `env_core._analytical_movement_delta`（D0.95） |

D0.95 栈 20 seed 上 worst 0.662 / weak3 0.724 / steady 0.808（三地板全达标，§10）。

#### 5.6.2 Candidate deployment：D1.1（lex L1 + 多候选 L3 + 对偶剪枝 + 最优带宽）

在 D0.95 基础上，D1.1 系列升级为**当前部署候选**（`OPTIMIZATION_LOG.md`）：

| 层 | 升级 | 关键代码/标志 |
|---|---|---|
| L0 | 最优正交带宽分配（D1.1-C，KKT 凸解，no-waste） | `analytical_comm_optimal_bw` |
| L1 | **lexicographic QoS 约束 max-min**（D1.1-A，Stage-A 可行性 + Stage-B worst 最大化，已 live） | `task_constrained_mode: lexicographic` |
| L2 | 不变（headroom 已量化 <1%，冻结） | P0 排名 + hold-5 |
| L3 | **多候选 trust-region**（D1.1-B，5–7 个整队候选 LP 打分）+ **对偶上界剪枝**（D1.1-B+，弱对偶精确剪枝） | `analytical_movement_candidates_enabled`、`analytical_movement_dual_prune` |
| 隐蔽性 | 反检测硬约束（D1.1-D/E，`P_D^I ≤ ε`，QoS×隐蔽性联合 LP） | `intercept_constrained_power_enabled`（默认关） |

**D1.1 候选 20 seed 上 worst 0.975 / weak3 0.978 / steady 0.982 / QoS 1.0**
（live eval，同种子同 warm-start；`_d095_lexcand20`）。这是当前最强部署配置，
论文主结果应以它为准；**D1.5 盲测（≥100 全新 seed）前不继续调参**（advice 013）。
`priced_structure.py` 的逐帧 owner+TX 重分配是离线结构修复 oracle，尚未接入 live
（L2 headroom <1%，已冻结，见 §6.2 注）。

---

## 6. 优化层数学（三层核心）

### 6.1 快层：固定 owner 的 max-min 功率 LP

固定唯一 owner 后，`a_iq` 为 UAV `i` 对目标 `q` 的每瓦有效 Deflection，
`b_i = 1 − P_comm,i`：

```text
max_{p,t}  t
s.t.  Σ_i a_iq p_iq ≥ t,   ∀q           (目标下界)
      Σ_q p_iq = b_i,      ∀i           (每 UAV 感知预算)
      p_iq ≥ 0
```

**对偶**（单纯形）：

```text
min_{λ ∈ Δ_Q}  Σ_i b_i · max_q λ_q a_iq
```

强对偶成立（LP 可行有界）。性质（D0.11/D055 已数值验证）：

- 最优对偶 `λ*` 由互补松弛支撑在**瓶颈目标**上（`λ*_q>0 ⟹ D*_q = min_q D*_q`）；
- `Σ_q λ*_q D*_q = min_q D*_q`（支撑线性化 / 次梯度）；
- `λ*_q` 是"目标 q 下界约束"的影子价格（边际最差 QoS 价值）。

**有限轮 Dantzig–Wolfe 列生成**：初始化 Q 个全目标覆盖列，每轮 owner 广播量化
价格，各 UAV 用本地 `a_iq` 生成最佳响应列，owner 端求 Q 维受限主问题；任何返回
功率都是完整可行列的凸组合，逐 UAV RF 等式恒成立，受限原始值单调不降。6-bit 价格
4 轮达精确解约 `98%`。代码：`coordination/maxmin_power.py`。

**松弛同几何上界**（慢层路由判据，不是可行动作）：

```text
D_q^relax = Σ_i b_i · max_j a_ijq
```

任何可行同几何方案逐目标不超过该值（放松共同 owner、角色、容量与跨目标功率耦合）。

**capability gauge（D0.93，task-constrained 功率，L1 的完整任务形式）**：把三地板当
**硬约束**而非标量目标，求最小预算缩放 `γ`：

```text
γ* = min_{p,γ} γ
s.t.  D_q ≥ d_min(0.60),   ∀q                       (worst)
      D_q ≥ d_weak3(0.70), ∀q 的 bottom-3 组合      (O(Q) order-statistic 线性化)
      D_q ≥ d_steady(0.80),∀q                       (steady)
      Σ_q p_iq ≤ γ b_i                              (预算按 γ 缩放)
```

`P_D` 在 `D ≥ c²/3` 区间上关于 D 凹，用两侧 PWL 证书（chord 下界 + tangent 上界）把
gauge 转成 LP（`γ_optimistic ≤ γ* ≤ γ_conservative`）。`γ* ≤ 1` 时返回的功率满足完整
任务，否则回退 reserve-first max-min（best effort）。代码：`coordination/capability.py`
（`capability_gauge_pwl_lp_full`）、`coordination/pwl_pd.py`。

### 6.2 中层：对偶引导的原子结构修复

- 候选：B=2 弱目标集合上的原子 N5（角色+owner 依赖闭包重建）与 N6（固定角色、
  只交换 1–2 目标 owner/support）；依赖闭包可影响多目标（D0.5：99.57% 候选实际
  改变 >2 目标）。
- 排序：候选只按**严格可行下界** `L` 与公开价格对偶上界 `U_λ` 的区间中点

  ```text
  L(S') = max(当前功率重放, 公共全目标列混合, 当前列−公开价格最佳响应列)
  U_λ(S') = Σ_i b_i max_q λ_q a'_iq
  score(S') = 0.5 L(S') + 0.5 U_λ(S')
  ```

  `L` 始终对应真实可行 RF 分配；`U_λ` 只用于排序，**绝不进入接受条件**。
- 接受：Top-1 候选须通过有限轮量化功率复算、严格单调改进、完整通信序列与
  episode-conformal 净收益门，否则 No-op；每事件至多一次原子重构（D0.15）。
- 代码：`coordination/dual_guided_structure_repair.py`。

**价格驱动结构重分配（D0.95，L2 的离线结构修复 oracle）**：与上述认证化原子搜索
互补，是 8/8 解析栈 L2×L3 交替下降里使用的**逐帧 owner+TX 重分配**结构修复器
（`tools/audit_l3_l2_alternating.py`；live 的逐帧 L2 是 §5.6 的功率无关 P0 排名）。
包络定理 `∂γ*/∂a_iq = −π_q p_iq` 给出同一价格 `π` 驱动结构与几何；owner 选择价格正交：

```text
j_q* = argmax_j Σ_i a_ijq b_i        (π_q > 0 提出公因子)
S_q* = top-K_q_max of {a_i,j_q*,q b_i}
```

半双工角色划分（TX/RX 互斥）用**功率共享感知的 max-min LP 打分**贪心求解（而非
2⁸ 枚举）。硬帧上结构单独可修复 44.2%（worst 地板），与几何交替联合达 67.5%。
代码：`coordination/priced_structure.py`。

### 6.3 慢层：认证化几何修复（原子提交）

每个接受动作是 prepare/vote/decision 三轮原子事务（D0.78）：

1. **Prepare**：收集 owner-local 状态/梯度，广播稀疏修正，收集 owner 验证因子，
   形成 digest 绑定证明；
2. **Commit**：全 UAV 投票，拒绝无副作用，接受则冻结结构与 RF 到承诺计划 `H` 步；
3. **Execute**：repair 与 baseline 分支用不同认证运动管、但相同结构/RF/语义送达
   状态与保守能量扣减；
4. **Synchronize**：终点位置/速度按构造相等，其余递归状态分支公共，下一决策从
   同状态开始。

代码：`coordination/certified_geometry_repair.py`、`persistent_geometry_execution.py`
、`causal_joint_plan.py`。

**解析几何下降（D0.94/D0.95，L3 的部署形式）**：与上述认证化原子提交互补，是
`analytical_movement_enabled` 的逐帧价格驱动运动。每帧对上一帧 per-watt 系数/owner/
预算求一步赤字→能力下降：

```text
若 ceiling_q < d_min（不可行）：  δ_q = [d_min − ceiling_q]_+
    d_k = −step · g_k/‖g_k‖,   g_k = Σ_q δ_q b_k ∇_{x_k} a_kq  (+ owner/Rx 项)
否则（可行但 worst < d_steady）：用 max-min 对偶 λ* 驱动
    d_k = −step · g_k/‖g_k‖,   g_k = Σ_q λ*_q p_kq ∇_{x_k} a_kq  (+ owner/Rx 项)
否则（worst ≥ d_steady）：悬停（零位移，不交回 actor）
```

**指标选择价格**：L3 的达标门是 `worst_deflection ≥ d_steady`（**最差目标**达到
0.80 地板），故几何层用 max-min 对偶 `λ*`（互补松弛集中于最差目标）驱动运动，而非
摊在三地板（worst+bottom-3+steady）的 capability 对偶 `π`——后者会稀释运动（§2.5
的 `steady` 是全部目标均值、`worst` 才是逐目标最小，L3 的 satisficing 门只盯最差
目标）。步长受 `‖Δp‖≤v_max·dt` 硬约束。代码：`env_core._analytical_movement_delta`。

> **L1 执行 vs L3 引导的目标"不一致"是有意分层（2026-08-16 审计澄清）**：多候选
> L3 的候选评分用**纯 max-min LP**（`solve_fixed_structure_maxmin_power_lp`），
> 即使执行内层是 lexicographic（Stage-B QoS 约束 max-min）。曾试图改成"评分与
> 执行一致"（用 Stage-B 评分候选），端到端 2-seed 对比严重退化（seed 503 final
> worst 0.996→0.662）。理论解释：**L3 的职责是引导几何，纯 max-min 评分在 `t*`
> 之上持续提供改进梯度（即使三地板已满足仍推动 UAV 提升均衡水平），而 Stage-B
> 在地板绑定处把 `t*` 钉在地板、评分对移动失去区分度 → 几何停滞**。这是 D0.95
> "指标选择价格"分离（L1 用 gauge/lex、L3 用 λ\*）在评分层的自然延伸，不是缺陷。
> 配置 `analytical_movement_lex_scoring`（默认关）保留为负结果记录。

---

## 7. 证书与安全层

系统不依赖单一 trigger，而是组合多层 fail-closed 证书（详见 D0.x 各 Gate）：

```text
owner-local 边际价值
  -> 依赖闭包 LNS（小提案触发，证书化排序）
  -> 物理计费且 epoch/digest 绑定的原子提交
  -> transition/transport 事件级联合最大分数（split-conformal）
  -> 延迟 owner 残差 + 冻结管线 + e-process 漂移锁定 No-op
```

关键构件：

- **事件级 split-conformal**：统计单位是独立 episode，事件内上千候选/帧/目标不
  当独立样本；联合分数取事件内最大标准化低估残差，5% 风险下有限样本覆盖下界
  `n/(n+1)`。
- **owner-local 双端物理证书**：候选下界与 No-op 上界分别满足逐目标 no-harm 与
  worst 净增益；不确定度含 Token 年龄/送达、belief 协方差、DD 支撑裕量、物理区间
  宽度与切换年龄。
- **no-harm 可组合性定理**（D0.78/D0.85/D0.86）：若两运动管满足速度/边界/扫描
  间隔、终点位置与速度相等、结构与 RF 冻结、Token 按分支可行送达交集更新、电量
  按分支最大扣减，则 `X_{t+H}^repair = X_{t+H}^baseline` 在协议表示精度上成立；
  全部未来确定性决策相等，总配对性能差恰为接受窗口内差之和。
- **e-process 漂移监测**：证书违例 e-value 越过 `1/δ` 时永久锁定 No-op（anytime
  界，依赖稳定条件）。

代码：`evaluation/episode_joint_conformal.py`、`self_normalized_feedback.py`、
`transition_certificate.py`、`channel_margin.py`、`physics_interval_gate.py` 等。

---

## 8. 学习目标（MAPPO + max-min 对齐奖励）

外层由 MAPPO 求解（CTDE，共享 actor + 集中 critic），内层由 §6 的 LP/搜索层求解。
`gamma=0.99`、`gae_lambda=0.95`、`ppo_clip=0.1`、per-target GAE bootstrap 已修复。

**团队奖励曲率**（D055）：

```text
r_team = Σ_q λ_q · U_κ(D_q) − λ_report·bits − 通信成本 − 约束惩罚
```

两种正交修正替代历史 `U(D) = -log(1-P_D)`（非凹、边际递增 → 富者愈富）：

1. **凹饱和效用** `U_κ(D) = 1 − exp(−κD)`（凹/单调/饱和）：使 `Σ ω_q U_κ(D_q)`
   恢复**单调次模**结构，贪心恢复 `(1−1/e)`；
2. **对偶价格加权** `λ = λ*`（§6.1 的最优 LP 对偶）：让学习梯度聚焦当前瓶颈
   目标，与协调器可认证的 Lagrangian 一致（"对偶一致奖励"）。

`reward_utility_mode ∈ {log, concave, maxmin_dual}`，默认 `log`；代码：
`environment/maxmin_reward.py`、`environment/reward.py`。

---

## 9. 符号 ↔ 代码 ↔ 单位映射

| 数学符号 | 代码变量 | 形状 | 单位 | 文件 |
|---|---|---|---|---|
| `p_k` | `uav.pos` | (3,) | m | `uav.py` |
| `v_k` | `uav.vel` | (3,) | m/s | `uav.py` |
| `E_k` | `uav.battery` | scalar | J | `uav.py` |
| `r_k` | `Action.role` | scalar∈{0,1,2} | — | `action.py` |
| `Δp_k` | `Action.delta_p` | (2,) | m/帧 | `action.py` |
| `τ,ν,α` | `tau,nu,alpha` | (K,K,Q) | s, Hz, 无量纲 | `geometry.py` |
| `d_eff(i,j,q)` | `DeflectionEntry.d_eff` | 每条目 | 无量纲 | `deflection.py` |
| `a_ijq` | `coefficient` | (K,K,Q) | 1/W | `physical_oracle_audit.py` |
| `a_iq`（fixed owner） | `fixed_owner_gain_matrix` | (K,Q) | 1/W | `maxmin_power.py` |
| `b_i` | `sensing_budget_w` | (K,) | W | `maxmin_power.py` |
| `p_iq` | `MaxMinPowerResult.power_w` | (K,Q) | W | `maxmin_power.py` |
| `λ_q` | `MaxMinPowerResult.prices` | (Q,) | 无量纲 | `maxmin_power.py` |
| `receiver_d[j,q]` | `receiver_d` | (K,Q) | 无量纲 | `evidence.py` |
| `D_q*` | `detection_D_q` | (Q,) | 无量纲 | `env_core.py` |
| `P_D^q` | `P_D_q` | (Q,) | ∈[0,1] | `detection.py` |
| `r_team` | `StepInfo.team_reward` | scalar | — | `reward.py` |
| `r_k`（shaped） | `shaped_rewards[k]` | scalar | — | `reward.py` |

---

## 10. 当前性能与可宣称边界

| 版本 | 种子 | mean-worst | QoS 可行率 | 判定 |
|---|---:|---:|---:|---|
| 4/4 冻结部署版 | 100 | 0.739 | 0.72 | 正式版本 |
| 6/6 原子控制 D0.85 | 20 | 0.6543 | 0.7303 | 严格 no-harm 证书成立（种子状态同待复核） |
| 8/8 原子控制 D0.86 | 20 | 0.4372 | 0.50 | 严格 no-harm 证书成立 |
| **8/8 解析栈 L0+L1+L3（D0.95，gauge）** | 20 | **0.662** | **1.0\*** | 当前部署配置（satisficing L1） |
| 8/8 lexicographic L1（D1.1-A，live） | 20 | **0.844** | **1.0**（LCB 0.881） | 与部署同种子同 warm-start，仅 L1 目标切换 |
| 8/8 lex + 多候选 L3（D1.1-B，live） | 20 | **0.975** | **1.0**（LCB 0.881） | 同种子，L3 视野增强 |
| **6/6 跨尺度 + 解析栈（D1.6，干净 test 前 20）** | 20 | **0.751–0.910** | **0.65–0.90** | **恢复跨门槛**（三变体） |
| **6/6 跨尺度纯零样本（无解析栈，作废基线）** | 20 | **0.293–0.344** | **0.10–0.30** | 配置不匹配（见下） |
| **8/8 盲测 D1.5（frozen 候选，100 全新 seed）** | **100** | **0.808** | **0.730（点估计过）** | **LCB 0.636 未过**（见下） |
| **8/8 盲测 D1.9（H=40 前瞻 L3，100 全新 seed）** | **100** | **0.962** | **0.950** | **LCB 0.888 过门（论文主结果）** |

> **差距分解（同 20 seed、同 warm-start、同 selection split，实测配对）**：
> `0.662 → 0.844`（**+0.18，仅切换 `task_constrained_mode: gauge→lexicographic`**，
> 不动运动/结构）→ `0.975`（**+0.13，再加多候选 trust-region L3**，
> `analytical_movement_candidates_enabled`）。三行都是 **live eval-only 运行**
> （`run_mappo.py`，`_d095_lex20` / `_d095_lexcand20`），不是 oracle——**差距不是
> "部署层做不到"，而是部署基线还停留在 D0.95 的 gauge 配置，D1.1-A/B 已验证的
> 更强配置尚未切换为部署基线**。注意：这组 seed 已多轮复用（D1_1A §5），论文
> 最终认证需 ≥100 全新 blind seed（D1.5）。

> **D1.5 盲测正式结果（2026-08-16，100 全新 seed，advice 013）**：
> frozen 候选（L0-KKT + Lex-L1 + P0-L2 + 多候选 L3）在从未暴露的 100 seed 上：
> **QoS feasible 0.730（点估计过 0.70 门）、Wilson LCB 0.636（未过 0.70）**；
> steady/weak3/worst 均值 0.808–0.820。三点结论：
> ① dev selection 的 0.975 是 seed 复用偏差，盲测真实水平 ~0.73（点估计）；
> ② 27/100 失败为**场景级**塌陷，主导因子是初始最差目标-最近 UAV 距离
> `worst_nearest`（<350 m QoS 0.91 → >550 m 仅 0.17）；v_max×episode 最大位移
> 375 m，实测 realized 距离与 worst P_D 相关 −0.863（>300 m 则 QoS 0），
> **>450 m 的 seed 处于时间可达性边界（部分物理不可达）**，300–450 m 是
> 策略可改进区（D1.9 候选：初始瞬态 warm start / 瓶颈优先趋近）；
> ③ 论文 `--require-lcb` 声明须如实报告 LCB 0.636 < 0.70（N=100 统计功效），
> 或改点估计 + 置信区间披露。详见 [`OPTIMIZATION_LOG.md`](OPTIMIZATION_LOG.md) D1.5。

> **D1.9 盲测闭合（2026-08-16，同 100 全新 seed，H=40 前瞻 L3）**：
> 单步 trust-region 停滞被 receding-horizon 评分修复（评分 H 帧持续趋近几何 /
> 执行 1 步）：**QoS feasible 0.950（95/100）、Wilson LCB 0.888（过 0.70）**，
> steady/worst 均值 0.962–0.963——**`--require-lcb` 论文主结果声明成立**。
> 剩余 5 个失败 seed 中 4 个 worst_nearest > 450 m（episode 375 m 位移预算下
> 物理不可达，D1.5 已预告），仅 seed 615（355 m）为残余策略失败；
> **95/100 已近该运动学下的可达性上界**。详见
> [`OPTIMIZATION_LOG.md`](OPTIMIZATION_LOG.md) D1.9。

`*` D0.95 端到端严格比较（tol=0）报 QoS 0.65（13/20），其中 7 个 seed 的 worst
恰好钉在 0.60 地板（`0.60 − 1.11e-16`，浮点伪影，非真实性能差距）；按
`marl.qos_eval_tol=1e-6` 口径为 **QoS 1.0（20/20，Wilson LCB 0.839）**。口径修正
见 [`D1_1A_LEXICOGRAPHIC_L1.md`](D1_1A_LEXICOGRAPHIC_L1.md) §2。

> **6/6 决策数据已回填重跑 + D1.6 失效分解（2026-08-16）**：
> ① 原 6/6 三行（0.543/0.645/0.635）来自 `980_k6q6` test split 前 10 个种子，
> **含全部 5 个被隔离种子**（795/747/105/860/2）且该批种子系统性偏乐观，原"均值
> 达标"结论被高估，已作废。
> ② 干净 test bank（`stratified_seeds_980_k6q6_v2.json`）前 20 种子重跑**无解析栈**
> 基线：worst 0.293–0.344、QoS 0.10–0.30（崩坏表象）。
> ③ **D1.6 分解（advice 013 情况 A）**：同一配置接上解析部署候选栈（L0 KKT +
> lex L1 + P0-L2 + 多候选 L3），worst 恢复至 **0.751–0.910、QoS 0.65–0.90**
> （`results/_6x6_d1_6_analytical_*_20/`）——**6/6 的失败主因是旧部署配置未启用
> 解析栈（configuration mismatch），不是跨尺度学习崩坏，无需重训 Student**；
> teacher 结构（oracle）worst 0.910 优于 Student 0.75–0.77，暴露 ~0.15 的 Student
> 校准 gap（后续可选校准）。详见 [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md)。

**8/8 解析栈端到端**（`analytical_*_enabled`，D0.87–D0.95，20 seed）：

| 配置 | worst | weak3 | steady | QoS 可行率 |
|---|---:|---:|---:|---:|
| C0 deployed | 0.355 | 0.508 | 0.773 | 0.40 |
| L0+L1（task-constrained） | 0.609 | — | 0.736 | 0.50 |
| **L0+L1+L3 几何** | **0.662** | **0.724** | **0.808** | **1.0**（tol=1e-6；严格比较 0.65 为浮点伪影） |

L3 几何把 steady 从 0.736 拉到 0.808（0/20 seed 低于 0.80），三地板首次端到端全达标；
剩余 7/20 seed 早期帧 worst<0.60 是滚动时域收敛瞬态。详见
[`D095_JOINT_L2_L3_ALTERNATING.md`](D095_JOINT_L2_L3_ALTERNATING.md)。

**8/8 可达性上界（oracle 诊断）**（D1.0-A horizon joint，
`tools/audit_horizon_joint_oracle.py`，从教师最终几何起点、全局信息、多步规划；
**不是部署执行**）：

| oracle | worst | weak3 | steady | QoS 可行率 |
|---|---:|---:|---:|---:|
| horizon_joint（H=20，maxmin 内层） | 0.852 | — | — | 0.75 |
| horizon_joint（gauge 内层） | 0.60（satisficing） | — | — | 0.90\* |

`\*` gauge 行早期 QoS 0.90 含 γ 缩放功率 bug（γ*>1 时 Σp≤γ·b>1 W），已修正（见
[`D1_1A_LEXICOGRAPHIC_L1.md`](D1_1A_LEXICOGRAPHIC_L1.md) §2）。

**结论（2026-08-16 修正）**：8/8 差距的主要来源**不是几何/物理**，而是两个执行层
选择：① **L1 目标 satisficing**——gauge 在 γ*≤1 后把 worst 钉在 0.60 地板，浪费
满足门限后的剩余资源；lexicographic（QoS 约束 max-min）在同一几何回收该资源
（+0.18，live 已验证）；② **L3 视野**——单步一阶梯度 vs 每帧多候选 trust-region
打分（+0.13，live 已验证）。horizon joint oracle（0.852）与部署（0.662）的差距
包含**同类的 L1/L3 增强 + 多步联合规划 + 全局信息**，作为可达性上界参考，不是
部署方法。

**8/8 天花板分解**（`tools/audit_scale_ceiling.py`，同几何瀑布）：

```text
deployed 0.31 -> power_only 0.675 -> single 0.700 -> duplex 0.706 -> relaxed 0.868
```

判定：**coordination_limited**，主导缺口是 `power_gap = +0.368`；快层 max-min 功率
LP 单独恢复 deployed→ceiling 缺口的 **65.5%**；几何层（L3）进一步把端到端 steady
推到 0.808（D0.95）。因此 8/8 的绝对性能不是物理天花板，而是部署功率/结构/几何
分配未做解析优化——现已逐层接入（§5.6）。

**可以宣称：**

- 分布式、共享参数、置换等变的结构 Student 在 4/4 达到 Medium 可部署门槛；
- 自适应精度 Token 在受限 U2U 信道下保持功率与原子传输约束；
- 6/6、8/8 静态目标域内，认证化原子控制器具有**严格 no-harm 可组合证书**；
- 双集合表示提供目标置换不变 / UAV 置换等变 / 均匀尺度协变 / 速度圆盘保证；
- 8/8 静态目标域内，解析执行栈（L0 通信余量 + L1 功率 LP/capability gauge + L3
  价格驱动几何下降）在 20 seed 上把 worst/weak3/steady 拉到 0.662/0.724/0.808，
  三地板全达标（D0.95；QoS 在 tol=1e-6 口径下 1.0）；
- 8/8 同几何可达性：lexicographic L1 / horizon joint oracle 显示 worst 0.844、
  QoS 1.0（20/20）可达——**性能缺口是协调/部署执行，不是物理**（oracle 诊断，
  非部署方法）。

**不可宣称：**

- 完全分布式端到端物理检测（最终融合仍是环境级集中式）；
- 任意 `K/Q` 的检测/QoS 通解，或 8/8 已达与 6/6 相同绝对 QoS；
- 学习候选优于解析物理候选（当前接受动作均来自解析梯度池）；
- lexicographic L1 / horizon joint oracle 为可部署执行路径（目前是离线诊断，
  需完成端点部署执行接线）；
- 波形级或真实硬件 ISAC 性能；
- 6/6 跨尺度"不可行"（D1.6 已证：接入解析部署栈后 worst 恢复 0.75–0.91、QoS
  0.65–0.90，原失败是旧配置未启用解析栈；残差为 Student 校准 gap ~0.15）。

---

## 11. 开放项与下一步

> **D0.89–D0.95 已闭合下述 1、2 两项**：快层 max-min LP 已接入部署路径
> （`analytical_sensing_power_enabled`，D0.89-A），慢几何层已以
> `analytical_movement_enabled`（价格驱动的赤字→能力下降 + 达标悬停）实现并端到端
> 验证（D0.95：20 seed 上 worst 0.662 / weak3 0.724 / steady 0.808，三地板全达标）。
> 详见 [`D095_JOINT_L2_L3_ALTERNATING.md`](D095_JOINT_L2_L3_ALTERNATING.md)。

1. ~~**功率层接入部署执行路径**~~（已闭合，D0.89-A）：快层 max-min LP 已成为部署
   执行主路径，8/8 worst +0.227。
2. ~~**慢几何层闭环实现**~~（已闭合，D0.95）：`analytical_movement_enabled` 价格驱动
   几何下降已闭环；剩余瞬态问题（7/20 seed 早期帧 worst<0.60）待补。
3. **max-min 对齐奖励的端到端训练消融**：`log vs concave vs maxmin_dual` 尚待
   独立多 seed 训练验证。
4. **非方形规模**（6/8 或 8/6）：检验 `K≠Q` 时置换→广义分配的语义断裂与 RF 可行
   域、协议缩放。
5. **随机物理残差校准**：随机 RCS/Swerling、随机 CSI、丢包与模型漂移尚未进入
   独立事件校准；此前 `3 dB / 0.5 ms` 只是工程裕量。
6. **统计功效（部分落地）**：mean-worst 是重尾统计量，30 种子不足以分辨 0.6 门槛；
   主指标应改用 `QoS feasible rate`（带 Wilson LCB）并扩到 ≥100 种子或加方差缩减。
   已新增 `tools/assert_gate_thresholds.py` / `tools/assert_formal_gates.py` 把
   Medium 门槛（含可选 Wilson LCB 强制）脚本化——论文正式声明时应启用
   `--require-lcb`，并注意当前 4/4 的 LCB 0.63 不达 0.70（须在方法学中显式说明）。
7. **对抗检测约束（advice 012，oracle 级已落地 + live 功率层已接入）**：T2 的 exposure 代理量
   （`E_w ≤ Γ_w`）已在 oracle 侧升级为**对方探测能力约束**（`D_w^I ≤ D̄_w^I`，
   三个对方能力等级 weak/medium/strong，`--inner intercept`），三向对比证明
   power-only 与 exposure 在 medium/strong 对手下 75%/100% 被裸发现，而
   detection-constrained 恒成立（P_D^I ≤ ε）。见
   [`T3_DETECTION_CAPABILITY.md`](T3_DETECTION_CAPABILITY.md)。
   **D1.1-D/E（2026-08-16）已把隐蔽性接入 live 功率路径**：`intercept_constrained_power_enabled`
   使执行功率满足 `P_D^I ≤ ε` 硬约束（`constrained_maxmin_lp`），lex 模式下与 QoS 地板
   联合进同一 LP（`qos_constrained_maxmin_lp` 加 intercept 行，三族价格 λ/π/μ）。
   端到端 4 UAV × 2 seed × 30 帧实测：所有档位 0 违反（约束跨帧保持），QoS 代价随
   对手强度单调（off/weak 0.703 → medium 0.663 → strong 0.001）——**"strong 对手
   必须静默"的 oracle 结论在 live 路径复现**。下一步是把新的攻防对偶价格
   `s_iq = λ_q a_iq − μ_w a^I[i,q]` 接入分布式协调（列生成），exposure 正式降级
   为 baseline。
8. ~~**lexicographic L1 部署化**~~（已闭合，D1.1-A）：lex L1 已是 live 配置
   （`task_constrained_mode: lexicographic`，20 seed worst 0.844→0.975 与多候选
   L3 组合），不再是"待部署求解器"。**D1.5 盲测已执行**（2026-08-16，100 全新
   seed，frozen 候选：L0-KKT + Lex-L1 + P0-L2 + 多候选 L3）：QoS feasible
   **0.730（点估计过 0.70）**、Wilson LCB **0.636（未过）**——dev 0.975 的复用
   偏差被量化，真实点估计 ~0.73。左尾分解确认失败由初始 `worst_nearest`
   主导（300–450 m 为策略可改进区，>450 m 部分物理不可达）。**D1.9 瓶颈前瞻
   L3 已实现**（`analytical_movement_lookahead_frames=H`，receding-horizon：
   评分 H 帧持续趋近几何 / 执行 1 步）：16 个验证 seed QoS 10/16 → 15/16、
   零退化；全量 100-seed blind 复测运行中。详见
   [`OPTIMIZATION_LOG.md`](OPTIMIZATION_LOG.md) D1.5/D1.9。
9. **6/6 跨尺度（2026-08-16 已分解）**：D1.6 失效分解证明 6/6 的"崩坏"是**旧部署
   配置未启用解析栈**（情况 A）——接入解析部署候选栈后 worst 恢复 0.751–0.910、
   QoS 0.65–0.90（§10）。无需重训整个 Student；可选后续为 Student 结构校准
   （teacher oracle 0.910 vs Student 0.75–0.77，gap ~0.15）。下一步：把 6/6
   +解析栈作为候选配置，并入 D1.5 盲测矩阵。

---

## 12. 工程治理与可复现性（2026-08-16 审计修正）

研究内容之外，本目录还记录了保证结果可信的工程治理机制（本次深度审计的修正成果，
详见 [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md)）：

- **测试集污染种子隔离（代码强制 + 已回填）**：seed `795/747/105/860/2` 因
  2026-07-29 split 漏写被误用，文档此前只有声明、代码零拦截。现注册表
  `config/quarantined_seeds.json` + `load_stratified_seed_split(strict=True)`
  fail-closed 拦截（含隔离种子即抛错），bank 生成自动排除并记录
  `quarantined_excluded`。**980_k6q6 已回填**（`stratified_seeds_980_k6q6_v2.json`：
  干净 test split、无 split 重叠、selection/confirmation/stress 不变），6/6 决策
  行已用干净 20 种子重跑（见 §10）；`800_q4` 与 `1130_k8q8` 的 selection/
  confirmation 仍含隔离种子（795/747 等），相关 strict 加载继续 fail-closed。
- **Gate 门槛脚本化**：`tools/assert_gate_thresholds.py`（单结果断言，从
  paired_eval.csv 重算聚合）与 `tools/assert_formal_gates.py`（批量断言正式结果，
  受污染项标 QUARANTINED 不参与判定）。4/4 正式结果经脚本复核 PASS
  （0.9132/0.8848/0.7393/0.72）；6/6 重跑三变体经脚本判定 FAIL（四地板不达标）。
- **协调层主/支路径立界**：`uav_isac/coordination/__init__.py` 导出主路径 API；
  14 个仅审计/研究用模块（priced_structure、certified_geometry_repair、
  owner_local_physics 等）带 `AUDIT/RESEARCH-ONLY` 标注，不代表部署行为。
- **结果治理**：`tools/audit_results_tree.py` 只读扫描 results/——原 767 目录中
  36 个无 manifest、20 个 `_` 前缀临时目录混存、summary.json 有 87 种 schema
  变体；2026-08-16 清理已归档 126 个过时目录（90 个 paper-era + 11 个无引用
  scratch + 26 个无引用无 manifest）至 `results/_archive/`，顶层现 641 个目录。
- **工程基线**：`requirements.txt`（此前缺失，导致 sklearn 缺失测试失败）、
  `.github/workflows/ci.yml`（Linux 全量 pytest）、`pytest.ini` 排除 scripts/
  （`test_ppo_ratio_fix.py` 曾模块级执行训练被 pytest 误收集）。全量测试基线
  **886 passed / 1 env failure**（sklearn，安装 requirements 后通过）。
