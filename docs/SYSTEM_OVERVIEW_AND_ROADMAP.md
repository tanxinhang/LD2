# 当前系统全景：现状、问题与应对思路

> 文档日期：2026-08-16（审计修正：并入 T2/T3/D1A/D1_1A，修正 8/8 主缺口表述；
> 建立 D0.87–D0.95 ≡ D0.10–D0.18 编号映射，见 §1.4）。
> 本文是**总纲**，汇总当前系统的整体架构、已量化的性能缺口、问题清单、理论框架与
> 分阶段应对路线。各 Gate 的详细证据见 `CURRENT_SYSTEM_STATUS.md`、
> `D087_POWER_DEPLOYMENT.md`、`D088_WARMSTART_STALENESS.md`、
> `D089_ANALYTICAL_INNER_POWER.md`、`D089B_STRUCTURE_RANKING.md`、
> `D089C_TSTAR_REWARD.md`、`D091_STEADY_CEILING.md`、`D092_BARGAINING_OBJECTIVE.md`、
> `D093_CAPABILITY_GAUGE.md`、`D093_L2_STRUCTURE.md`、`D093_POWER_SIDE_E2E.md`、
> `D094_L3D_DISTRIBUTED_GEOMETRY.md`、`D095_JOINT_L2_L3_ALTERNATING.md`、
> `D1A_HORIZON_JOINT_ORACLE.md`、`D1_1A_LEXICOGRAPHIC_L1.md`、
> `T2_CENTRAL_ORACLE.md`、`T3_DETECTION_CAPABILITY.md`、`CURRENT_SYSTEM_MODEL.md`。

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
| 8/8 解析栈端到端 D0.95（gauge） | 20 | 0.662 | 0.808 | 0.65* |
| 8/8 lexicographic L1（D1.1-A，live） | 20 | **0.844** | 0.845 | **1.0**（LCB 0.881） |
| 8/8 lex + 多候选 L3（D1.1-B，live） | 20 | **0.975** | 0.982 | **1.0**（LCB 0.881） |

`*` D0.95 的 0.65 是严格比较的浮点伪影：7 个 seed 的 worst 恰好钉在 0.60 地板
（`0.60 − 1.11e-16`）；容差 ≥1e-12 时 QoS=1.0。口径修正见
[`D1_1A_LEXICOGRAPHIC_L1.md`](D1_1A_LEXICOGRAPHIC_L1.md) §2。

**8/8 差距性质（2026-08-16 修正，关键）**：lex L1 与 lex+多候选 L3 两行是
**live eval-only 运行**（`_d095_lex20` / `_d095_lexcand20`，与 D0.95 部署
**同 20 seed、同 warm-start、同 selection split**，仅配置不同）——不是 oracle。
实测配对差距：`0.662 → 0.844`（**+0.18，仅切换 L1 目标 gauge→lexicographic**）→
`0.975`（**+0.13，加多候选 trust-region L3**）。**"部署层做不到 0.844"的准确表述
是"部署基线还停留在 D0.95 的 gauge 配置，D1.1-A/B 已验证的更强配置尚未切换为
部署基线"**；horizon joint oracle（0.852）才是教师几何起点的可达性上界。
注意：该 20 seed 已多轮复用，最终认证需 ≥100 全新 blind seed（D1.5）。

**8/8 缺口状态（2026-08-16 修正）**：旧表述"8/8 worst 0.437 是主要缺口"对应
D0.86 原子控制（20 seed）。D0.95 解析栈端到端已将 8/8 均值 worst 提到 **0.662**、
steady 0.808（三地板均值达标，7/20 seed 早期帧存在滚动时域收敛瞬态）；D1.1-A
lexicographic L1 oracle 进一步显示 8/8 在**教师最终几何**上可达 worst **0.844**、
严格 QoS **1.0（20/20）**。因此：**8/8 的物理可行域足够，缺口收敛为"有限视野
一阶分布式控制器 → 多步联合优化器（lex L1 / horizon joint）"的算法差距与
"端点部署执行"差距**，不再是"不可达"。

### 1.4 Gate 编号映射（审计修正）

同一 8/8 解析功率/几何链在项目内存在两套编号，本总纲统一使用 **D0.87–D0.95**
（官方 Gate 编号），与 `CURRENT_SYSTEM_STATUS.md` 的 **D0.10–D0.18** 为同一物理内容：

| 总纲编号 | 状态文档编号 | 内容 |
|---|---|---|
| D0.87 | D0.10 | LP 功率接入部署路径 |
| D0.88 | D0.11 | H=5 功率 hold + 认证 staleness 界 |
| D0.89-A/B/C | D0.12–D0.13 | 删头复现 / 排名负结果 / tstar 接线 |
| D0.91 | D0.14 | steady 天花板 0.993 / worst 0.955 |
| D0.92 | D0.15 | Kalai–Smorodinsky 讨价还价 |
| D0.93 | D0.16 | capability gauge + epoch 包络 + L0 余量 |
| D0.94 | D0.17 | L3 分布式价格几何 |
| D0.95 | D0.18 | 联合 L2+L3 交替下降 + 端到端 |

两套编号并存是历史原因（文档各自演进），不改变任何冻结数值；后续新 Gate 一律使用
**D0.9x 单套编号**并在总纲登记。

---

## 第二部分：量化性能缺口（8/8 天花板分解）

### 2.1 同几何瀑布（test20）

| 层级 | worst | 增量 | 含义 |
|---|---:|---:|---|
| deployed（D0.86 原子控制） | 0.355 | — | 旧部署口径（D1A 复测为 0.44） |
| C1（LP 功率，固定结构） | 0.582 | +0.227 | **功率缺口**（最大） |
| single-duplex（最优结构+功率） | 0.700 | +0.118 | **结构缺口** |
| full-duplex | 0.706 | +0.006 | 全双工（可忽略） |
| relaxed ceiling | 0.955 | +0.249 | 几何/去耦合上界 |
| **horizon_joint oracle（D1.0-A，H=20）** | **0.852** | — | 允许移动后的多步联合上界（从部署几何出发） |

> 注意：瀑布各台阶不允许 UAV 移动；D1.0-A 增加移动维度后 mean worst 0.852 /
> QoS 0.75（`docs/D1A_HORIZON_JOINT_ORACLE.md`）。因此 **0.60 门槛在可执行几何上
> 也是可达的**（20 seed 中 85% 达 ≥0.72、80% 达 ≥0.75）。

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

- **A1（2026-08-16 修正）**：8/8 部署执行仍落后于可达上界——D0.95 解析栈端到端
  mean worst 0.662（20 seed，7/20 早期帧瞬态 <0.60），而 lex L1 oracle 显示同几何
  可达 0.844 / QoS 1.0。缺口性质从"不可达"修正为"**有限视野一阶控制器 → 多步
  联合优化器（lex L1 / horizon joint）的算法差距 + 端点部署执行差距**"。
- **A2**：4/4 尾部危险（5/100 种子 worst<0.1，bottom-20% CVaR 0.288）。
- **A3**：近一个月主线增益塌缩到 1e-5（D0.85/0.86 严格配对增益），优化锁死在
  fail-closed 不动点（D0.95 解析栈已以 ~0.2 的端到端增益打破该不动点）。

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
| D1.0-A/B/C | Horizon joint oracle（H=20）：worst 0.852 / QoS 0.75；内层求解器消融；责任分配联合移动为负结果 |
| D1.1-A | **Lexicographic L1（QoS 优先 ≻ worst 最大化）：live 20-seed mean worst 0.844、严格 QoS 1.0（20/20，LCB 0.881），Gate 大幅通过**（与 D0.95 部署同种子同 warm-start，仅 L1 目标切换 +0.18）；浮点 QoS 口径修正（0.65→1.0） |
| D1.1-B | 多候选 trust-region L3（`analytical_movement_candidates_enabled`）：lex 基础上 worst 0.844→**0.975**、steady 0.982，worst≥0.95 率 17/20 |
| T2 | 安全/低暴露统一母问题（standoff + Γ）：oracle worst 保持 0.79–0.81；**standoff 是主要限制**（需从满足约束的初始几何重规划） |
| T3 | **Detection-Capability-Constrained（advice 012）**：三档对方能力三向对比——power-only 在 medium/strong 下被 75%/100% 发现，exposure 在 strong 下同样失效，只有 detection 约束恒成立（strong 下必须近乎静默） |

### 下一步（按优先级，每步可单测验证）

> **Step 1–4 已在 D0.93–D0.95 链闭合**：single-duplex steady（D0.91/D0.93）、
> 结构 MILP/价格驱动结构重分配（D0.93-L2/D0.95 `priced_structure.py`）、几何层
> （D0.94/D0.95 `analytical_movement_enabled`）均已实现并验证。D1.1-A 已证明
> lex L1 在 oracle 级达成 Phase 2（worst≥0.75、QoS≥0.90）。当前剩余：

**Step 5（统计修复，审计后 P1）**：主 Gate 改 QoS feasible rate + 单侧 95% Wilson LCB，
≥100 blind seeds；冻结 versioned 独立 test bank；mean-worst/CVaR 降为尾部指标。
**审计警示：6/6（10 seed）、8/8（20 seed）Gate 目前仍以 mean-worst≥0.60 为主指标，
与 E1 自相矛盾，须在论文正式声明前切换。**

**Step 6（端到端瞬态补齐）**：D0.95 端到端三地板均值达标但 7/20 seed 早期帧
worst<0.60（滚动时域收敛瞬态）。候选：提前触发运动 / warm-start 几何 / 粘性瓶颈。

**Step 7（结构层部署化收尾）**：`priced_structure.py` 的价格驱动贪心（25.7%）距
oracle（~40%）仍有 gap；D1.1-A 的 lex L1 目前是 **oracle 诊断**（教师几何起点、
全局信息），**尚未作为部署执行路径接入**（与 D1A §2 的"端点部署执行差距"一致）。
下一步是把 lex L1 的 Stage-B QoS-constrained max-min 做成逐帧可部署求解器。

**Step 8（T3 分布式化）**：T3 的 detection-capability 约束目前是中央 oracle；按
advice 012 §11，用统一三价格 `s_iq = λ_q a_iq − μ_w a^I[i,q]` 做本地 bid 的分布式
列生成是后续主线（μ 即对方探测能力价格）。strong 对手下"必须静默"是 1 W/28 GHz
的物理结论，需波形层（扩频/LPI）或大幅降功率。

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
2. **结构 gap 到底多大**：D1.1-A 已给出关键数据——**L1 内层从 gauge（satisficing）
   换成 lexicographic（QoS 优先 ≻ worst 最大化）就把 20-seed mean worst 从 0.671
   推到 0.844**（不动 L3/结构/轨迹）。这证明"结构+功率+几何的联合机会"主要卡在
   内层目标的字典序选择上，而不是结构枚举本身；剩余 gap 是部署执行（Step 7）。
3. **几何是否必要（已答，D0.95）**：joint structure+power 单独只有 26.7% 硬帧修复，
   steady 停在 0.736；几何层（+ 结构交替）把硬帧修复推到 67.5%、steady 拉到 0.808。
   **几何是性能必需，不只是稳健性。**
4. **随机物理**：Rician/Swerling/随机 CSI/丢包尚未进独立事件校准；此前 3dB/0.5ms 是
   工程裕量。
5. **C2/C3 训练结果**（进行中）：将给出"缩小动作空间 / 换 reward"的干净消融，但
   预期不是性能主贡献。
6. **T3 隐蔽性与 QoS 的对抗**：detection-constrained 在 strong 对手下 QoS 坍缩到
   0.001（必须静默）——这是 1 W/28 GHz 下的物理结论；若要同时保 QoS 与隐蔽性，
   必须进入波形层设计（扩频/LPI/波束成形），超出当前解析层边界。
7. **统计口径纪律**（审计 E1）：8/8 的"worst 0.662 达标"建立在 20 个种子上，0.662
   与 0.60 的差距小于 bootstrap 噪声尺度；正式声明必须配 95% bootstrap CI 或改用
   QoS+Wilson LCB 主指标，不能以均值达标口径头条呈现。
