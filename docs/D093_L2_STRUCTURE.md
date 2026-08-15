# D0.93 L2：结构能力扩张审计（关键重定向：瓶颈是几何，不是结构）

> **⚠ 已被 D0.95 修正**：本文"17% 结构可修复 / 83% 几何受限"是在 **escalate 帧
> 12 采样 + 词典序完整任务判定 + 只重排 owner 不重排 TX 配对** 的窄口径下得到的，
> 严重低估了结构层 headroom。D0.95 在全部 1572 个硬帧（γ=∞）上重测：
> - 最佳 owner + 每目标 top-3 TX（含功率共享）**结构单独可修复 44.2%**（worst 地板），
>   完整单角色 oracle 全任务为 26.7%，价格驱动贪心为 25.7%；
> - 结构（L2）与几何（L3）**互补而非替代**：交替联合达 **67.5%**。
> 因此"瓶颈是几何、不是结构"的定性结论仍成立，但"结构只救 17%"的**定量结论作废**。
> 详见 [`D095_JOINT_L2_L3_ALTERNATING.md`](D095_JOINT_L2_L3_ALTERNATING.md)。

> 状态：完成（8/8 escalate 帧采样 12 个）。
> 工具：`tools/audit_d093_l2_structure.py`。
> 结论：**escalate 帧里只有 17% 是"结构可修复"，83% 是"几何受限"；即便最优
> single-duplex 结构，这些帧 worst 也只有 0.411。真正的瓶颈是 L3 几何，不是 L2 结构。**

## 1. 方法

对 power-only capability gauge 判为 escalate（γ_P*>1 或目标 ceiling<d_0.60）的帧，
用联合 pair+power oracle `solve_joint_pair_power_oracle(full_duplex=False)`（枚举
全部单双工角色分区）求**最优结构 + 最优功率**，检查其 (worst, weak3, steady) 是否
满足完整任务：

- 满足 → 结构可修复（L2 能解决）；
- 不满足 → 几何受限（即便最优结构也救不了，需 L3 移动）。

## 2. 结果（12 个 escalate 帧采样）

| 指标 | 值 |
|---|---:|
| 结构可修复 | 2 / 12 = **16.7%** |
| 几何受限 | 10 / 12 = **83.3%** |
| oracle worst 均值 | 0.411 |
| oracle steady 均值 | 0.414 |

## 3. 决定性含义（重定向主线）

1. **"77% 结构瓶颈"（D0.93-F）实为"77% 几何瓶颈"**：在 escalate 帧上，即便枚举
   所有角色/owner/edge 的最优结构，worst 仍只有 0.411——目标在几何上不可达
   （DD 不可行或距离过远），**结构怎么换都救不了**。

2. **D0.87 的"结构缺口 +0.118"是"容易帧"上的现象**：0.700 是 final-resolve（几何
   已收敛）帧上的 single-duplex；在 escalate（硬）帧上，single-duplex 只有 0.411。
   结构层的真实 headroom 远小于之前的估计。

3. **主线重定向到 L3 几何**：性能要从 0.63 继续往上，靠的是 **dual-sensitivity
   几何**（`∇_x γ*`，或 advice 001 §5 的 `∇_x t* = Σ_q λ*_q Σ_i p*_iq ∇_x a_iq`），
   即"UAV 该往哪飞才能最快消掉 capability deficit"。这正是从第一轮 advice 就在
   强调、但一直未实现的那一层。

## 4. 完整分层结论（四层能力阶梯的实证）

```text
L0 通信余量：回收 ~2W，但只解释 23% escalate（不是主因）
L1 功率 LP：  +0.227 worst（主收益，已部署）
L2 结构：     只修复 17% escalate（窄口径，D0.95 修正为 44% 硬帧；见顶部修正注）
L3 几何：     83% escalate 的真正瓶颈（D0.94/D0.95 已实现并验证）
```

**这条链把"该动哪一层"从猜测变成了有因果证据的分层判定。**

## 5. 下一步

起 **L3 dual-sensitivity 几何**：给定 escalate 帧的 LP 解 `(λ*, p*)`，计算
`∇_x γ*`（能力 gauge 对几何的方向导数，配熵正则化处理 λ* 退化、DD 跨门当事件），
验证"沿该方向移动 UAV 能单调降低 γ*"。这是把 relaxed ceiling（0.955）里那部分
几何 headroom 吃掉的唯一路径。
