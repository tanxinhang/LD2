# D0.94-L3D：Distributed Capability-Sensitivity Geometry（理论 Gate）

> 状态：规格冻结，T2 单测验证。
> 依据：`advice/008.md`。
> 定位：**price-mediated local capability descent**（价格中介的局部能力下降），
> 不是"全局 γ\* 梯度的分布式近似"。

## 1. 一句话定位

```text
distributed price update → quantized π Token → local bistatic sensitivity
→ local safe trust-region action
```

集中式 `γ*`、`∇_x γ*` 只保留为**离线 oracle**，回答"在线局部动作损失了多少全局能力下降"；
不是部署动作。

## 2. 五个 Gate

| Gate | 证明/验证什么 | 状态 |
|---|---|---|
| **T0 Price provenance** | π 只能来自实际 U2U 消息；peer-to-peer dual consensus 收敛到集中式对偶 | ✅ 单测（max-min dual） |
| **T1 Gradient decomposition** | `g = col(g_1,...,g_K)`，每个 `g_k` 只用本地 + 送达消息 | ✅ 规格已定 |
| **T2 Oracle equivalence** | ideal comm 下 col(g_k) 与集中式梯度数值一致 | ✅ 单测 |
| **T3 Robust price error** | 量化/AoI 下 `‖ĝ−g‖` 界 + robust descent 条件 | 待做 |
| **T4 Support recovery** | DD 不可行时不取零梯度，进入 constraint-restoration | ✅ 单测 |

## 3. 四个必须写进实现的修正

### (1) π* 的产生（T0：已选 peer-to-peer，单测通过）

`tests/test_l3_price_consensus.py` 用**环图 + Metropolis 双随机矩阵 + 分布式对偶
子梯度**验证：每架 UAV 只与邻居交换本地价格，经 consensus + 本地子梯度迭代，平均
价格收敛到集中式 max-min 对偶 λ\*（err < 0.05），对偶值一致。**"无中心协调器"的
价格生成由此有了硬证据**（当前验证的是 max-min dual λ^mm；capability dual π^cap
的 peer-to-peer 收敛是同一机制的扩展，需单独再验证）。

```text
π_k^{r+1} = Π_Δ[Σ_{ℓ∈N_k} W_{kℓ} π_ℓ^r − α_r g_k^r],   g_k^r = b_k e_{argmax_q π_q a_kq}
```

### (2) 局部梯度分解（含通信几何项）

```text
g_k = g_k^Tx + g_k^Rx + g_k^comm
g_k^Tx   = −Σ_q π_q p_kq ∇_{x_k} a_{k,j_q,q}
g_k^Rx   = −Σ_{q:j_q=k} π_q Σ_i p_iq ∇_{x_k} a_{i,k,q}
g_k^comm = −γ* Σ_i β_i ∇_{x_k} b_i
```

owner 项**不是零通信**：owner k 需要 support 发射机的 `p_iq` 与链路几何，最小消息
`m_{i→j_q} = {q, p_iq, epoch, AoI}`；未送达必须 stale/bound/fail-closed，不能用
真值补齐。

### (3) 梯度写法不固化 `−2a(x−x_q)/R²`

系数里还有 `χ_rep`、双基地 Tx/Rx 距离、DD support，正确写法是

```text
∇_x a = a · ∇_x log a
```

Friis 项系数是 −2 还是 −4 以代码里 α 是"幅度增益"还是"功率增益"为准，用
finite-difference 单链路先验证，不猜符号。

### (4) λ^mm ≠ π^cap（不能宣称"一个价格贯穿三层"）

快层价格是 max-min dual λ^mm；L3 用完整 task capability 的 KKT 价格 π^cap。二者
一般不相等。只有把 power/structure/geometry 全改成**同一个 capability/PWL primal**
后才可写 "one common price coordinates all layers"；在那之前只能写
"common price architecture, layer-specific dual semantics"。

## 4. L3 两个 Mode（DD 门是最大风险）

- **Mode I（capability descent）**：目标有有效 support 且 `g_dd−g_min>δ`，用
  `d_k=−g_k`。
- **Mode II（support restoration）**：高价格目标 `π_q>0` 但无有效 support（`a=0`，
  零梯度"吸不动"），改用恢复势：

```text
Ψ_k = Σ_q π_q [g_min − g_dd,kq]_+,   d_k^restore = −∇_{x_k} Ψ_k
```

先把高价值目标送回 feasible manifold，跨回 `g_dd≥g_min` 后重建 support、重求
power、更新 π、回到 Mode I。

## 5. 信息边界（硬测试）

部署函数禁止访问：full `A[K,K,Q]`、全 UAV 真值位置、oracle `γ*`、未送达 Token。
测试一旦读这些对象就 fail。这是"分布式"主张最有说服力的证据。

## 6. T2 单测（已完成）

`tests/test_l3_distributed.py::test_local_gradient_matches_oracle`：把集中式
`capability_geometry_gradient`（= per-UAV 梯度 col）与"Tx + Rx 逐项求和"的局部
分解对照，理想通信下数值一致（<1e-9）。这是 T1 分解正确性的基础证据。

## 7. T4 Mode II support restoration（已完成）

`tests/test_l3_support_restoration.py`（3 passed）：

1. **恢复势非零**：`g_dd < g_min` 时 `Ψ_k = π_q[g_min−g_dd]_+ > 0`（DD 门外）。
2. **恢复梯度减 gap**：沿 `−∇Ψ` 移动使 `g_dd` 回升（把目标拉回 feasible manifold）。
3. **对照**：DD 门外 `a=0` → capability 梯度为零（吸不动），正是需要 Mode II 的原因。

**关键物理发现**：静态目标（ν=0）时 `g_dd = |sinc(l_offset)| ≥ |sinc(0.5)|=0.637 >
g_min`，**纯延迟失配永远不跨门**；只有延迟+多普勒同时失配（运动 UAV，ν≠0）才使
`g_dd ≈ 0.637² = 0.406 < g_min`。因此 Mode II 只在运动 UAV 场景才真正触发，与 8/8
trace 的 gate-crossing 观察一致。

## 8. 落地与端到端结果（D0.95 补记）

本规格的 price-mediated descent 已落地为两层：

1. **长时域几何**（`tools/audit_l3_long_horizon.py`）：赤字→能力两阶段下降，在硬帧
   （γ=∞）上 **48% 达到 γ≤1、79% 变为可解**（100 步 × 2.5 m，平均 63 m/UAV）。
2. **交替结构-几何**（`tools/audit_l3_l2_alternating.py` + `priced_structure.py`）：
   在硬帧上 **67.5% 达到 γ≤1**（L3 单独 47.5%、L2 单独 26.7%）。

**端到端**（`analytical_movement_enabled` 钩子接入 `env_core.step()`，20 seed）：

| 指标 | task-constrained 基线 | + L3 运动 | 地板 |
|---|---:|---:|---:|
| worst | 0.609 | **0.662** | ≥0.60 |
| weak3 | — | **0.724** | ≥0.70 |
| steady | 0.736 | **0.808** | ≥0.80 |
| QoS feasible | 0.50 | **0.65** | ≥0.70 |

**两条关键修正**（写进实现，见 `_analytical_movement_delta`）：

1. **指标选择价格**：`steady` = 末 20 帧**最差目标** P_D 的时间平均，故几何层应被
   **max-min 对偶 `λ*`**（互补松弛集中于最差目标）驱动，而不是摊在三地板的
   capability 对偶 `π`——后者会稀释运动（实测 steady +0.006 vs λ* 的 +0.059）。
   这与 §3(4)"λ^mm ≠ π^cap"一致：几何层用 λ^mm，功率层才用 π^cap。
2. **达标即悬停**：最差目标 ≥ 0.80 地板后返回**零位移**（而非回退 actor 运动），
   否则满分 seed（worst 1.0）会被硬拉坏到 0.48、硬 seed 会与 actor 振荡。
