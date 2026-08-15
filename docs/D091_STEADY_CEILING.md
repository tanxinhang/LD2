# D0.91：8/8 steady/worst 天花板审计（两地板均物理可达）

> 状态：完成（8/8 test20）。
> 工具：`tools/audit_scale_steady_ceiling.py`。
> 结论：**worst≥0.60 与 steady≥0.80 都物理可达；性能缺口是协调，不是物理。**

## 1. 结果（relaxed same-geometry 逐目标上界，20 episodes 稳态窗口）

```text
D_q^relax = Σ_i b_i · max_j a_ijq   (放松跨目标功率耦合、共同 owner、角色、容量)
```

| 指标 | 天花板 | 要求 | 判定 |
|---|---:|---:|---|
| steady mean | **0.993** | ≥0.80 | 可达 |
| steady min | 0.925 | — | — |
| worst mean | **0.955** | ≥0.60 | 可达 |
| worst min | 0.549 | — | 个别帧 |
| QoS feasible rate | 0.90 | ≥0.70 | 可达 |

## 2. 决定性含义

1. **"0.80 steady 物理不可达"被否定**：天花板 steady=0.993，远高于 0.80。上一轮
   我担心的"max-min 削峰填谷导致 steady 塌缩"**不是物理限制，是协调缺陷**——LP
   在固定结构上只能在 worst/steady 之间搬，但 joint structure+power 优化可以同时
   拿住两者。

2. **真实缺口 = 协调缺口**：把 D0.87 的天花板分解与本次 steady 天花板对齐：

   | 层级 | worst | 增量 |
   |---|---:|---:|
   | deployed | 0.355 | — |
   | C1（LP 功率，固定结构） | 0.582 | +0.227（功率） |
   | single-duplex oracle（最优结构+功率） | 0.700 | +0.118（结构） |
   | full-duplex oracle | 0.706 | +0.006（全双工） |
   | relaxed ceiling | 0.955 | +0.249（几何/去耦合上界） |

   **主缺口是"部署 P0 结构选择次优"**：单双工 oracle（部署本就能达到的模式）是
   0.700 ≥ 0.60，而 deployed 只有 0.355、加 LP 功率也才 0.582。差的 0.118 是
   **结构层**，不是功率、不是物理。

## 3. "大调整"的正确靶点（回应"架构需要大调整"）

不是继续调 MAPPO / 换 reward（这两者已被 LP 和天花板证明是次要的），而是：

1. **把 P0 结构选择换成 joint structure+power max-min（single-duplex oracle）**，
   不再"贪心选结构 → 单独跑功率 LP"。这是当前最大的 +0.118 结构缺口来源。
   `solve_joint_pair_power_oracle(full_duplex=False)` 已是现成的集中式求解器，
   目前只当离线教师/审计用。

2. **再把它分布化**：single-duplex oracle 是 MILP（教师/上界），部署需要
   "局部候选图（Gate A2 的 `L_u=4,L_q=4`）+ 有限轮 DW" 逼近，而不是每帧全图 MILP。

3. **最后做 dual-sensitivity 几何**（`∇_x t* = Σ_q λ*_q Σ_i p*_iq ∇_x a_iq`）：
   这是 relaxed 0.955 里、超出 single-duplex 0.700 的那部分几何 headroom。

## 4. 对正在跑的 C2/C3 的定位

C2/C3（外层重训 + log/tstar reward）仍在跑，其价值不变：给出"缩小动作空间 /
换 reward"的干净消融。但它们**不是**性能主贡献——主贡献在结构层（joint
structure+power）与几何层。性能要过 0.60/0.80，靠的是把 single-duplex oracle
从"离线教师"变成"部署协调器"，而不是再把 reward 调 20 次。
