# D0.93-R：固定结构 (worst, steady) 能力区域（决定性结果）

> 状态：完成（8/8 test20 前 3 seed）。
> 工具：`tools/audit_d093_capability_region.py`。
> 结论：**功率层单独就存在同时满足 worst≥0.60 且 steady≥0.80 的分配**。D0.92 的
> "steady-vs-worst 张力"是 scalarization 伪象，不是功率自由度耗尽。

## 1. 方法（依据 advice/002.md）

对每个固定结构帧，扫描 worst 地板 `w`，解"最大 steady 受 worst ≥ w 约束"：

```text
max_p  mean_q P_D(Σ_i a_iq p_iq)
s.t.   Σ_i a_iq p_iq ≥ d_w  ∀q（worst ≥ w）,  Σ_q p_iq = b_i,  p ≥ 0
```

`d_w` 由 `minimum_deflection_for_detection_probability(w)` 反解。`w ≥ 0.60` 时
`d_w ≈ 8.05 > c²/3`，`P_D(D)=Q(c−√D)` 在 `D ≥ d_w` 上凹，故这是凹最大化（凸规划），
SLSQP 得全局最优。

## 2. 结果（mean steady at each worst floor）

| worst 地板 w | mean steady | 可行帧数 |
|---:|---:|---:|
| 0.30 | 0.885 | 230 |
| 0.50 | 0.893 | 176 |
| **0.60** | **0.901** | 152 |
| 0.70 | 0.900 | 145 |
| 0.80 | 0.908 | 126 |

**关键判定：`steady_at_w=0.60 = 0.901 ≥ 0.80`，`power_layer_satisfies_both = true`。**

## 3. 决定性含义

1. **"steady-vs-worst 张力"是伪象**：max-min `(0.636, 0.746)` 与 bargaining
   `(0.446, 0.826)` 只是**两种 scalarization 各自选了 Pareto 前沿的一个端点**；
   前沿**中间**存在 `(≥0.60, ≥0.90)` 的点同时满足两个地板。D0.92 的结论"单一标量
   无法兼顾"被推翻——问题是没选对分配，不是不存在。

2. **功率层未耗尽**：`w=0.60` 时稳态可达 0.90，`w=0.80` 时仍 0.908。所以**不需要
   升级结构/几何层来满足 QoS 地板**（至少对 ~66% 的帧）。

3. **~34% 的帧是结构瓶颈**：`w=0.60` 的可行帧数从 230 降到 152，说明约 1/3 帧里
   某目标在固定结构下**达不到 0.60**（`D_q^I < d_60`）。这部分才真正需要结构层。

## 4. 正确的架构（advice 的 "Feasibility outside, Bargaining inside"）

```text
路由（能力判定）：   R_P ∩ A ≠ ∅ ?   →  是（~66% 帧）
分配（层内 Pareto）： max_p η(p)  s.t.  worst(p)≥0.60, steady(p)≥0.80
                    （QoS = 约束，bargaining η = 目标）
升级（仅对不可行帧）： R_P ∩ A = ∅  →  结构层（~34% 帧）
```

这比 D0.92 的"η*≤0 路由"严格得多：η* 是 bargaining value，不是 layer-sufficiency
certificate（advice 002 §1 的批评完全正确——`η*≥0` 恒成立，永远不触发结构）。

## 5. 下一步（D0.93-A：约束分配）

实现"constrained bargaining"：

```text
max_{p,η} η  s.t.  D_q(p) ≥ D_q^0 + η·h_q（bargaining 公平）
                  D_q(p) ≥ d_w（worst 地板，硬约束）
                  mean_q P_D(D_q) ≥ 0.80（steady 地板，硬约束）
                  Σ_q p_iq = b_i
```

其中 steady 地板是凹约束（`mean P_D ≥ 0.80` 关于 D 凹），所以整体是凸规划，可用
现有 LP + 凹约束迭代，或直接 SLSQP。**这一步直接验证"约束分配能拿到 (0.60+, 0.80+)
这个前沿中间点"，并量化相对 max-min/bargaining 的收益。**
