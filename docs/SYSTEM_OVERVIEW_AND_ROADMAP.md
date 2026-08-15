# 当前系统全景：现状、问题与应对思路

> 文档日期：2026-08-15。
> 本文是**总纲**，汇总当前系统的整体架构、已量化的性能缺口、问题清单、理论框架与
> 分阶段应对路线。各 Gate 的详细证据见 `CURRENT_SYSTEM_STATUS.md`、
> `D087_POWER_DEPLOYMENT.md`、`D088_WARMSTART_STALENESS.md`、
> `D089_ANALYTICAL_INNER_POWER.md`、`D089B_STRUCTURE_RANKING.md`、
> `D089C_TSTAR_REWARD.md`、`D091_STEADY_CEILING.md`、`D092_BARGAINING_OBJECTIVE.md`、
> `D093_CAPABILITY_GAUGE.md`、`D093_L2_STRUCTURE.md`、`D093_POWER_SIDE_E2E.md`、
> `D094_L3D_DISTRIBUTED_GEOMETRY.md`、`D095_JOINT_L2_L3_ALTERNATING.md`。

---

## 第一部分：系统整体现状

### 1.1 研究对象与边界

多 UAV 分布式 ISAC：`K` 架 UAV、`Q` 个目标，二维平面，只有 U2U 通信、无地面链路，
每 UAV 严格 `1 W` 通信+感知联合功率预算。边界：

```text
分布式局部观测 + 共享参数策略（运动/通信/感知意图）
  + 物理 U2U Token 传递（bit/时延/功率/丢包）
  + 分布式结构 Student 近似角色/配对/owner
  + 环境级集中式证据融合与检测评价
```

**不负责**：原始 IQ 波形、完整协议栈、飞控、波形层联合设计（发射协方差/子载波/
波束/模糊函数固定）。

### 1.2 三层控制架构（快/中/慢）

```text
快层：固定 role/owner/edge 的 max-min 感知功率 LP（Dantzig–Wolfe 列生成）
中层：仅当固定结构上界低于 QoS 时，joint structure+power 原子 N5/N6 搜索
慢层：仅当宽松同几何上界仍低于 QoS 时，航迹/几何修复（D0.95 已实现
      `analytical_movement_enabled` 价格驱动下降 + 达标悬停）
```

外加一层**证书/安全**：事件级 split-conformal、owner-local 双侧物理证书、原子
依赖闭包提交、no-harm 可组合定理、e-process 漂移锁 No-op。

### 1.3 三尺度结果

| 版本 | 种子 | worst | steady | QoS 可行率 |
|---|---:|---:|---:|---:|
| 4/4 冻结部署版 | 100 | **0.739** | 0.913 | 0.72 |
| 6/6 原子控制 D0.85 | 20 | 0.6543 | 0.871 | 0.7303 |
| 8/8 原子控制 D0.86 | 20 | 0.4372 | — | 0.50 |

**8/8 是主要缺口**：worst 0.437 距 0.60 差 0.163，QoS 可行率 0.50 距 0.70 差 0.20。

---

## 第二部分：量化性能缺口（8/8 天花板分解）

### 2.1 同几何瀑布（test20）

| 层级 | worst | 增量 | 含义 |
|---|---:|---:|---|
| deployed | 0.355 | — | 当前部署 |
| C1（LP 功率，固定结构） | 0.582 | +0.227 | **功率缺口**（最大） |
| single-duplex（最优结构+功率） | 0.700 | +0.118 | **结构缺口** |
| full-duplex | 0.706 | +0.006 | 全双工（可忽略） |
| relaxed ceiling | 0.955 | +0.249 | 几何/去耦合上界 |

### 2.2 steady 天花板（D0.91，决定性）

| 指标 | relaxed 天花板 | 要求 |
|---|---:|---:|
| steady | **0.993** | ≥0.80 |
| worst | **0.955** | ≥0.60 |

**结论：0.60 worst 与 0.80 steady 都物理可达。性能缺口是协调，不是物理。**

### 2.3 三个关键张力

1. **worst vs steady 张力**：pure max-min LP 把 steady 从 0.773 压到 0.692（
   D0.89-A 实测），因为 max-min 削峰填谷牺牲强目标。reserve-first 是必要解。
2. **结构层 headroom 被 D0.93-L2 低估**：D0.93-L2 报"17% 结构可修复"是在 escalate
   窄口径下测得，**已被 D0.95 修正**——全部硬帧上最佳 owner + top-3 TX（含功率共享）
   结构单独可修复 44.2%，与几何（L3）互补后交替联合达 67.5%。结构不是主瓶颈的定性
   结论不变，但"只救 17%"的定量结论作废。
3. **几何 headroom 最大**：relaxed 0.955 里超出 single-duplex 0.700 的 +0.249 是
   几何/去耦合上界，尚未被任何已实现层吃掉。

---

## 第三部分：问题清单（按性质分层）

### A. 性能问题

- **A1**：8/8 worst 0.437（deployed 0.355）距 0.60 缺口大，QoS 0.50 距 0.70 缺口大。
- **A2**：4/4 尾部危险（5/100 种子 worst<0.1，bottom-20% CVaR 0.288）。
- **A3**：近一个月主线增益塌缩到 1e-5（D0.85/0.86 严格配对增益），优化锁死在
  fail-closed 不动点。

### B. 架构—实现错位

- **B1（已修，D0.89-A）**：快层 LP 此前只当审计/上界，未进部署路径。现已接入
  `analytical_sensing_power_enabled`，8/8 worst +0.227。
- **B2（主缺口）**：P0 结构选择是"贪心选结构 → 单独跑功率 LP"，不是 joint
  structure+power；结构次优导致 C1 只有 0.582 而非 single-duplex 0.700。
- **B3（已实现，D0.95）**：慢几何层此前"只路由、未实现"，现已以
  `analytical_movement_enabled`（价格驱动的赤字→能力下降 + 达标悬停）落地并端到端
  验证：20 seed 上 worst 0.662 / weak3 0.724 / steady 0.808，三地板全达标。
- **B4（负结果，D0.89-B）**：功率无关结构排名 + λ* 硬支撑优先级不改善 worst，且
  使 steady 塌到 0.640。λ* 硬支撑过强，需软化。

### C. 理论问题

- **C1（B8 遗留）**：`U(P_D)=-log(1-P_D)` 在 D 上非凹，历史 P0 贪心无次模保证。
  已用 max-min Deflection LP（凹）与 concave 效用部分修复。
- **C2**：single-duplex oracle 是**交替启发式**（自述"not a proof of global
  optimality"），被误当上界使用。真实结构缺口需要 McCormick 精确 MILP 量化。
- **C3**：λ* 只在功率层是精确对偶证书；结构层是松弛对偶（MILP 有对偶间隙）；几何
  层是方向导数（λ* 退化非唯一）。三层对偶地位不同，不能一刀切。
- **C4**：DD 门 `g_dd≥g_min` 是支撑不连续，几何 Lipschitz/可微性失效。
- **C5（D0.88）**：认证 staleness 界 2B 松约 900 倍，只能当安全证书、不能当触发。

### D. 学习问题

- **D1**：reward 曲率富者愈富（`-log(1-P_D)` 边际递增），已用 maxmin_dual/concave/
  tstar 三模式部分修复（D0.55/D0.89-C）。
- **D2**：学习候选从未胜过解析候选（D0.85 47/47、D0.86 143/143 全由解析胜出）。
  learner 价值应在运动预测（D0.82 显示 8.6%/29%/6.6% 零样本改善），不在物理动作源。
- **D3（D0.89-C 冒烟）**：外层变量 headroom 小，重训 t* 收敛慢（0.292→0.324 / 5ep），
  符合"LP 已做重活"的预期，但也说明外层学习不是性能主贡献。

### E. 统计/评估问题

- **E1**：mean-worst 是 Q 取 min 的重尾统计量，30 种子分辨不了 0.60 门槛（D0.18
  cal/val 差 0.142）。主指标应改 QoS feasible rate + Wilson LCB，尾部指标 ≥100 种子。
- **E2**：测试种子反复被揭盲（D0.10/0.16/0.17），评估预算被烧。需冻结 versioned
  独立 test bank。

### F. 通信/计算问题

- **F1**：8/8 协议 20.3 kbit / 43.3 ms，随 KQ 上涨。
- **F2**：全图 MILP 结构求解是教师/参考，不可部署（D0.89-B 实测 unit-power 排名使
  MILP 候选集扩张 ~3–4x 变慢）。需局部候选图或分布式列生成。

---

## 第四部分：核心理论框架（应对思路）

### 4.1 主线：dual-consistent 分布式 ISAC 控制

把三层统一在**对偶证书**上，而不是三个独立启发式：

```text
λ*, μ*（完整对偶价格）
  → 功率：p* = argmax Σ_i b_i max_q (λ*+μ*)_q a_iq          （精确，凸 LP）
  → 结构：列生成，reduced cost = (λ*+μ*)_q·a_ijq − value    （LP 松弛引导 + MILP 兜底）
  → 几何：∇_x t* = Σ_q (λ*+μ*)_q Σ_i p*_iq ∇_x a_iq        （Danskin 方向导数）
```

**关键修正（回应"λ* 是否会偏向"）**：λ* 只编码 worst 约束的影子价，丢掉
reserve/steady、容量、角色影子价，会系统性偏向最差目标（D0.89-B 已实测 steady 塌
到 0.64）。正确价格是**完整对偶**：

```text
有效价格 π_q = λ*_q（max-min） + μ*_q（reserve，反解自 P_floor） + 容量/角色对偶
```

**μ* 是 counterbalance λ* 的那个量**：λ* 拉最差目标，μ* 保每个目标不低于地板。

### 4.2 完整对偶能否"天然成立"——不能，需分层处理

| 层 | 问题结构 | 强对偶 | 对偶地位 | 不完美度量 |
|---|---|---|---|---|
| 功率 | 固定结构 LP | **成立** | 精确证书 | gap=0 |
| 结构 | joint MILP（x∈{0,1}） | **不成立** | 松弛上界 | duality gap |
| 几何 | LP/MILP 值函数 | 只给方向导数 | 次梯度 | 退化半径+步长 |

**不完美解决思路**（branch-and-bound 哲学）：

```text
结构层：LP 松弛对偶 λ* → 乐观上界 V_relax（引导列生成）
        可行 MILP 解      → 可达下界 V_feasible（fail-closed 验证）
        gap = V_relax − V_feasible（显式度量"对偶多不完美"）
几何层：方向导数 + line search（不当梯度乱迈）
        DD 跨门当事件强制重解（D0.88 已实现 dd_gate_crossing）
退化：  熵/近端正则化使 λ* 唯一（ε→0 恢复真对偶）
```

### 4.3 统一拓展：非凸性的价格（price of non-convexity）

三层各自的"不完美"落到一个同维证书：

```text
功率层 gap = 0（凸，天然零）
结构层 gap = V_relax − V_MILP（整数性的价格）
几何层 gap = 次梯度步长 + 支撑裕量 + 退化半径（非光滑/不连续的价格）
```

这是可辩护的创新点：**不是发明新数学，是把 McCormick 精确线性化（二值 x 无松弛
间隙）、DW 列生成 ε-近似、Danskin 方向导数 + 熵正则化统一成一个 gap certificate**。

### 4.4 四条控制纪律（数值/理论必需）

1. **退化选解**：并列瓶颈时 λ* 非唯一，加最小范数 λ* 或熵正则化。
2. **可行性裕量**：显式携带 reserve_shortfall（D0.89-C 已实现）作为 fail-closed。
3. **时域一致**：λ* 逐帧抖动，需动量/滞回（deficit EMA 已软、λ* 硬）。
4. **步长控制**：几何次梯度步长 trust-region，受 `‖Δp‖≤v_max·dt` 硬约束。

---

## 第五部分：分阶段应对方案（带门槛，可验证）

### 已完成的 Gate

| Gate | 结论 |
|---|---|
| D0.87 | LP 功率接入部署路径，8/8 worst +0.2447（trace）/ +0.2264（live 20-seed） |
| D0.88 | H=5 功率 hold 保住 +0.203、省 65% 重解；认证界 2B 松 900x 只当安全证书 |
| D0.89-A | 删 sensing-power 头 + 内层 LP，复现 D0.87，功率平衡 <1e-12 |
| D0.89-B | 功率无关排名 + λ* 负结果：不改善 worst、steady 塌缩、MILP 慢 3–4x |
| D0.89-C | tstar 奖励 + reserve 不可行回退接线完成；训练管线冒烟通过 |
| D0.91 | steady 天花板 0.993 / worst 0.955：两地板物理可达，缺口是协调 |
| D0.92 | Kalai–Smorodinsky 讨价还价功率 LP：worst 0.465 / steady 0.857（谱两端） |
| D0.93 | capability gauge（三地板硬约束）+ epoch 包络 + L0 通信余量 + PWL 证书 |
| D0.94 | L3 分布式价格几何（T0/T2/T4 单测）+ 长时域 48% 硬帧修复 |
| D0.95 | 联合 L2(结构)+L3(几何) 交替下降 67.5% 硬帧修复；端到端 worst 0.662 / steady 0.808 |

### 下一步（按优先级，每步可单测验证）

> **Step 1–4 已在 D0.93–D0.95 链闭合**：single-duplex steady（D0.91/D0.93）、
> 结构 MILP/价格驱动结构重分配（D0.93-L2/D0.95 `priced_structure.py`）、几何层
> （D0.94/D0.95 `analytical_movement_enabled`）均已实现并验证。当前剩余：

**Step 5（统计修复，仍待做）**：主 Gate 改 QoS feasible rate + 单侧 95% Wilson LCB，
≥100 blind seeds；冻结 versioned 独立 test bank；mean-worst/CVaR 降为尾部指标。

**Step 6（端到端瞬态补齐）**：D0.95 端到端三地板均值达标但 7/20 seed 早期帧
worst<0.60（滚动时域收敛瞬态）。候选：提前触发运动 / warm-start 几何 / 粘性瓶颈。

**Step 7（结构层部署化收尾）**：`priced_structure.py` 的价格驱动贪心（25.7%）距
oracle（~40%）仍有 gap，属"分布式 vs 全局规划"的代价；如需逼近可做价格加权局部
搜索（RX 交换邻域）。

---

## 第六部分：创新性定位（克制）

- 单看 McCormick、Benders、Danskin、熵正则化、DW、conformal 都已有先例。
- 可辩护的是**组合**：把三层统一在**完整对偶证书（λ*+μ*+…）**上，并把三层的
  "不完美"统一成**非凸性的价格（gap certificate）**。这是"dual-consistent 分布式
  ISAC 控制"的完整闭环，不是三个独立启发式，也不是波形层或普通 MAPPO 网络层。
- 在 McCormick MILP 量化结构 gap、完整对偶（含 μ*）实现、及独立多 seed 端到端
  验证前，不作"首次提出"断言。

---

## 第七部分：风险与待决问题

1. **路 A vs 路 B**：可部署优先（λ* 排序 + MILP 兜底，结构最优性只能 ε-经验）vs
   理论优先（branch-and-price，结构有精确证书但计算重）。倾向**路 A + 局部候选图
   ε-近似界**。
2. **结构 gap 到底多大**：取决于 Step 2 的 McCormick MILP 结果。若 gap 小（交替式
   已近最优），结构层不值得大改；若 gap 大，才是真正"大调整"。
3. **几何是否必要（已答，D0.95）**：joint structure+power 单独只有 26.7% 硬帧修复，
   steady 停在 0.736；几何层（+ 结构交替）把硬帧修复推到 67.5%、steady 拉到 0.808。
   **几何是性能必需，不只是稳健性。**
4. **随机物理**：Rician/Swerling/随机 CSI/丢包尚未进独立事件校准；此前 3dB/0.5ms 是
   工程裕量。
5. **C2/C3 训练结果**（进行中）：将给出"缩小动作空间 / 换 reward"的干净消融，但
   预期不是性能主贡献。
