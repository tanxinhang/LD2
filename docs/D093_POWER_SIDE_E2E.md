# D0.93 功率侧端到端链：L0（通信余量）+ L1（功率 LP）

> 状态：L0+L1(max-min) 完成；L0+L1(bargaining) / L0+L1(reserve) 待收。
> 这是把离线审计（D0.93-F L0 / D0.89-A L1）**真正接进 live 环境**后的端到端仿真对比。

## 结果（8/8 test20，20 seed，同 seed 配对）

| 配置 | worst | steady | weak3 | QoS 可行率 |
|---|---:|---:|---:|---:|
| C0 deployed | 0.355 | 0.773 | 0.508 | 0.40 |
| L1 功率 LP（max-min） | 0.582 | 0.692 | 0.595 | 0.55 |
| **L0+L1（max-min）** | 0.642 | 0.753 | 0.647 | 0.55 |
| L0+L1（reserve 0.6） | 0.645 | 0.755 | 0.649 | 0.55 |
| L0+L1（bargaining） | 0.465 | **0.857** | 0.652 | 0.35 |

## 关键结论（功率侧端到端闭环）

1. **L0 在 L1 之上再 +0.060 worst / +0.061 steady**（回收 ~2 W 通信 link margin），
   worst 跨过 0.60。reserve 0.6 与 max-min 几乎一致（worst 0.645≈0.642），因为
   max-min 已把 worst 推到 0.64>0.60，reserve 不 binding。
2. **bargaining 把 steady 拉到 0.857（跨过 0.80）但 worst 掉到 0.465**——正好是
   D0.92-A 的"公平-效率谱两端"在端到端复现。
3. **单一标量目标（max-min/reserve/bargaining 任一）都无法同时过 0.60/0.80**。这
   正是 D0.93-R/G 的诊断：两个地板是**约束**，不是标量目标；需要 D0.93-A1 的
   task-constrained allocation（QoS 作硬约束，bargaining 作层内目标），且功率侧
   只能解决 ~27% 帧，其余 73% 是结构/几何瓶颈。

## 意义

功率侧（L0+L1）已端到端闭环并量化到极限：**max-min 给 worst 0.642 / steady 0.753，
bargaining 给 worst 0.465 / steady 0.857**。两者之间的"双地板"缺口必须靠
task-constrained 分配 + L2 结构 + L3 几何补齐，而不是继续换功率侧标量目标。

## 补记：L0+L1+L3（task-constrained 功率 + 几何运动）端到端（D0.95）

在 task-constrained 功率栈上叠加 `analytical_movement_enabled`（L3 几何运动，
max-min 对偶价格 + 达标悬停），20 seed 结果：

| 配置 | worst | weak3 | steady | QoS 可行率 |
|---|---:|---:|---:|---:|
| C0 deployed | 0.355 | 0.508 | 0.773 | 0.40 |
| L0+L1（task-constrained） | 0.609 | — | 0.736 | 0.50 |
| **L0+L1+L3 几何** | **0.662** | **0.724** | **0.808** | **0.65** |

L3 几何把 **steady 从 0.736 拉到 0.808（跨过 0.80，0/20 seed 低于地板）**，worst
跨过 0.60，weak3 跨过 0.70——**三个 QoS 地板首次在端到端全达标**。剩余小缺口：7/20
seed 的 worst 在早期帧（运动收敛前）仍 <0.60，QoS feasible 0.65 距 0.70 还差一点，
均源于滚动时域的收敛瞬态。详见 [`D095_JOINT_L2_L3_ALTERNATING.md`](D095_JOINT_L2_L3_ALTERNATING.md)。
