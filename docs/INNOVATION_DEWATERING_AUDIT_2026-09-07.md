# 创新点脱水与精炼审计（2026-09-07）

## 结论

当前系统**没有已经被实验证明的算法首创点**。现阶段可以成立的是：系统把多个已有模块放入同一仿真链，并通过 oracle ladder 与正交消融定位了性能损失；这属于有价值的系统研究证据，但还不能写成“提出了新的 MARL 算法”。

因此本版把创新表述分成三类：

| 等级 | 含义 | 当前归类 |
|---|---|---|
| 已证实 | 有新的数学对象/算法，并有独立基线、反事实和统计证据 | 无 |
| 候选 | 被故障定位指向，但尚无新算法或定理闭环 | candidate-aware / hold-aware episode-QoS-regret 接口 |
| 工程集成 | 模块组合、执行顺序或实现约束 | MAPPO+LP、attention/equivariant、projection、Student/teacher、factor message |

## 为什么现有表述“水分高”

近三年工作已经分别覆盖了当前几个表面上的“创新词”：

- 通信学习已覆盖何时发送、发送什么、如何编码，并明确考虑信道状态和消息顺序；因此“任务导向通信/低维消息”本身不是新问题。[Hu et al., 2023](https://proceedings.mlr.press/v189/hu23a.html)；通信 MARL 综述也按消息生成、传输、集成和学习过程系统归类。[Zhu et al., 2024](https://link.springer.com/article/10.1007/s10458-023-09633-6)
- 多智能体信用分配已从 agent 维度扩展到 temporal-agent 维度，并出现图结构、反事实和奖励重分配方法；因此“task-regret loss/多尺度 credit assignment”不能单独构成贡献。[Li et al., 2025](https://proceedings.mlr.press/v258/li25e.html)、[XMIX, 2025](https://doi.org/10.1016/j.neucom.2025.131471)、[TAR², 2025](https://arxiv.org/abs/2502.04864)
- 约束强化学习已有 primal-dual、CVaR/no-violation 和未知约束 regret 结果；连续动作 masking/projection 也已有直接方法。因此“惩罚 + 投影 + 证书”是约束实现选项，不是首创对象。[Tabas et al., 2023](https://proceedings.mlr.press/v211/tabas23a.html)、[Maddux & Kamgarpour, 2024](https://proceedings.mlr.press/v238/maddux24a.html)、[Stolz et al., 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/acf4a08f67724e9d2de34099f57a9c25-Abstract-Conference.html)

这些文献并不意味着本项目不能采用这些模块；它们只说明模块名字不能被当作创新结论。

## 逐项降级与保留

### 1. “多时间尺度混合控制”

当前实现是：外层学习/结构提议，内层解析功率，安全层做硬约束。它解释了模块责任，**但还不是新算法**。要升级为贡献，必须给出一个可复现的接口性质，例如：外层只输出长期变量和有界 residual；内层对固定结构给出精确解；两者的梯度/信用不再奖励被执行层覆盖的动作。没有该形式化和对照时，只能写“decision-authority-consistent architecture”。

### 2. “candidate-aware、hold-aware episode-QoS-regret rank”

这是当前唯一由实验明确指向的候选对象，但仍是**研究假设**。现有证据是：full candidate 对 CE rank 的 worst 提升 `+0.00561`，而 task-regret rank 在 full candidate 上反而使 worst 下降 `−0.02182`；交互项为 `−0.02743`。这证明 task-regret 不能作为独立可插拔 loss，不证明新的 rank 算法已经成立。

若要升级，必须同时定义：

1. 候选来源/覆盖率变量，而不是把 teacher pool 当 oracle；
2. hold-5 下的 episode 级 regret，而不是单帧 reward；
3. candidate miss 到 QoS regret 的可计算上界或校准关系；
4. 在相同 candidate budget、bit、时延和训练 seed 下击败 CE、counterfactual、QMIX/attention critic 等基线。

缺一项，就只能称为“候选接口设计”。

### 3. “factor message + certificate”

它目前既没有自然 M4-D candidate，也没有等 bit 闭环证明。它可能提供的独特性质是 decision-sufficient bit bound、outward interval 和 fail-closed admission；不是“消息更短”或“更可解释”。在完成 learned latent / no-communication / factor interval 等 bit 对照前，不应把它列为正式创新。

### 4. “LP/projection/safety layer”

固定结构下 max-min LP 的价值是精确性和可审计性，projection 的价值是逐帧可行性；二者是**性质驱动的工程选择**。当前 projection oracle 为严格 identity，说明它没有带来性能增益；K=16 的 replicated LP 仍有 P95 `127.3 ms`，说明解析解也未解决计算扩展性。不能把“用了 LP/投影”写成性能创新。

## 当前论文可用的最小诚实表述

> 本工作提出一个面向多 UAV-ISAC 仿真的、决策权一致的分层执行架构，并通过责任矩阵、oracle ladder 与候选来源×排序损失正交消融，证明结构候选覆盖与 episode-level 排序之间存在耦合；在此基础上提出 candidate-aware、hold-aware QoS-regret 接口作为待验证算法假设。本文不声称 MAPPO、attention、LP、projection、通信压缩或 certificate 单独具有首创性。

这段话的关键是把“提出”限制在**架构和可检验接口**，把真正的算法创新暂列为 candidate，直到新算法、定理和独立盲测完成。

## 下一道创新门槛

只有满足下面四项，才允许把候选升级成正式创新：

1. **新对象**：给出严格的 candidate-supported episode regret 定义和训练/推断算法；
2. **新性质**：证明或校准 candidate coverage 与 QoS regret/LCB 的关系；
3. **必要性**：同观测、同动作、同 bit、同计算预算下，现有 MARL 内生修复无法提供该性质；
4. **独立证据**：预注册 seed、置信区间、失败案例和算力/通信代价同时改善。

在此之前，后续工作应优先做定义、基线和必要性实验，而不是继续增加网络头、通信协议或安全模块。

## 当前推进状态

第一步已完成：`uav_isac/evaluation/episode_regret.py` 将 episode regret 固定为
`(steady_drop, weak3_drop, worst_drop)`，并把候选不支持与排序/执行残差分开记录。现有
18-seed trace 上没有出现 reference-feasible → method-infeasible 的 QoS flip，因此目前
只有连续 regret 证据，没有“候选缺失导致 QoS 失败”的因果证据。下一步应在该指标冻结后，
再做候选生成器与排序器的同预算独立盲测。

第二步的非因果上界也已完成：以两个训练 seed 选择 future-hold `q20(d_eff)` 排序，固定后在
18个未参与校准 seed 上使 worst `0.5668→0.6100`、QoS feasible `0.444→0.500`。这把候选
假设进一步精炼为“预测 hold 段 lower-tail edge evidence”，但 CVaR 几乎不变，且 oracle 使用
未来真值，仍不能升级为算法创新。完整边界见
[`HOLD_QOS_RANK_AUDIT_2026-09-07.md`](HOLD_QOS_RANK_AUDIT_2026-09-07.md)。
