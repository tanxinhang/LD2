# 部署版创新性与研究范围审计

> 审计日期：2026-07-29  
> 审计性质：面向论文立项的机制级快速审计，不等同于穷尽式系统综述或专利查新。  
> 当前正式部署结果：`steady=0.890`、`weak3=0.854`、`worst=0.679`、
> `CVaR=0.053`、QoS 可行率 `0.70`、`768 bit/frame`、`1.40 ms/frame`，
> 且严格满足单 UAV `1 W` 通信–感知总功率约束。

## 1. 结论

当前部署版适合作为**强工程基线和安全回退控制器**，但不适合把现有模块组合直接
包装成论文唯一主算法。主要原因不是系统没有价值，而是以下单项都已有明确先例：

- 连续 Token、自主语义、注意力接收、选择通信对象、多轮通信和动态静默；
- Set/GNN 置换等变结构和跨规模参数共享；
- MAPPO、约束 Critic、CVaR/最差性能优化和原始–对偶训练；
- 可微投影、Sinkhorn/优化展开和学习增强的资源分配；
- 两机器人联盟、分布式拍卖、通信感知的联盟形成；
- UAV-ISAC 中的任务分配、轨迹、功率及感知质量联合优化。

因此，论文不能声称“首次使用 Token/Attention/CVaR/原始–对偶/两节点联盟”。
这些只能作为实现组件或相关工作。

现有证据支持扩大研究范围，但扩大方式不是继续堆模块，而是改变优化的基本变量：

\[
\boxed{
\text{节点–目标权重 }x_{i,q}
\;\longrightarrow\;
\text{有向双基地超边 }y_{i_{\rm Tx},j_{\rm Rx},q}
}
\]

建议主线收敛为：

> **通信–感知共功率约束下，面向 Worst/CVaR 的分布式非对称双基地联盟协商。**

更准确的英文问题表述可为：

> **Risk-aware endogenous-communication hypergraph negotiation for
> distributed multi-UAV bistatic ISAC.**

其潜在创新不来自任何一个组件，而来自目前检索到的文献尚未同时覆盖的四重耦合：

1. 一个感知动作必须由角色非对称的 Tx/Rx 两端共同形成；
2. Token 传输既提供联盟可见性，又占用同一 UAV 的 `1 W` RF 预算，因此通信代价
   是**内生的**，而非外加的 bit penalty；
3. 目标是动态 Worst-QoS/CVaR，而非总收益或平均任务完成时间；
4. 执行只依赖经过时延、丢包、量化和功率约束后实际到达的邻居 Token。

这是一条**有条件成立**的创新方向，不是已经证明的新颖性。最终仍需更系统的
引文追踪与正式实验支持。

## 2. 当前系统的准确边界

当前系统可准确描述为：

1. 每个 UAV 使用共享、置换等变 Actor，根据本地状态、目标集合、历史检测和
   物理到达的 U2U Token 输出运动、通信和感知倾向；
2. 通信具有 bit、时延、信道和 RF 功率成本；
3. 每个 UAV 的通信功率与感知功率严格投影到 `1 W`；
4. 环境级 P0 使用全局候选图执行确定性的 deficit-aware max-min 双基地配对；
5. 环境对选中接收端的检测证据做集中式融合。

因此它是：

\[
\text{分布式运动/通信意图}
+\text{集中式配对与证据融合}
\]

而不是完全分布式端到端 ISAC。论文可以声称 Actor 的执行输入是局部的，但不能
在保留 P0 时声称整个配对与检测链路完全去中心化。

## 3. 机制与已有工作的对照

| 当前或曾提议的机制 | 代表性先例 | 审计判断 |
|---|---|---|
| 自主学习连续通信内容 | [CommNet, NeurIPS 2016](https://proceedings.neurips.cc/paper/2016/hash/55b1927fdafef39c48e5b73b5d61ea60-Abstract.html)；[DIAL/RIAL, NeurIPS 2016](https://proceedings.neurips.cc/paper_files/paper/2016/hash/c7635bfd99248a2cdef8249ef7bfbef4-Abstract.html) | 不能作为独立创新 |
| 学习“说什么、对谁说”、多轮 Token | [TarMAC, ICML 2019](https://proceedings.mlr.press/v97/das19a.html) | 不能作为独立创新 |
| 学习何时通信和注意力群组 | [ATOC, NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/6a8018b3a00b69c008601b8becae392b-Abstract.html) | 不能作为独立创新 |
| 学习调度/稀疏通信/静默 | [SchedNet, ICLR 2019](https://openreview.net/pdf?id=HUAnBToP_a)；[Information Bottleneck Communication, ICML 2020](https://proceedings.mlr.press/v119/wang20i.html)；[TMC, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/c82b013313066e0702d58dc70db033ca-Abstract.html) | Top-k、自适应轮次和长静默惩罚均不是方法级新颖性 |
| 因果/反事实通信归因 | [Individually Inferred Communication, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/fb2fcd534b0ff3bbed73cc51df620323-Abstract.html) | CCP/反事实头必须证明 ISAC 特有的新因果对象，否则属于延伸 |
| 置换等变、跨规模无线资源分配 | [REGNN, IEEE TSP 2020](https://arxiv.org/abs/1909.01865) | Set scorer 和规模无关参数共享是必要设计，不是核心创新 |
| 图上的局部价值分解与消息传递 | [Deep Coordination Graphs, ICML 2020](https://proceedings.mlr.press/v119/boehmer20a.html) | Pairwise/graph critic 本身不新 |
| 可微优化层和硬约束投影 | [OptNet, ICML 2017](https://proceedings.mlr.press/v70/amos17a.html) | 投影/展开是工具，需有新的问题结构或理论 |
| 分布式原始–对偶资源分配 | [Distributed wireless resource learning](https://arxiv.org/abs/1905.13378)；[REGNN](https://arxiv.org/abs/1909.01865)；[Communication-efficient primal–dual, AISTATS 2021](https://proceedings.mlr.press/v130/chen21c.html) | QPD 不能以“首次学习增强原始–对偶”表述 |
| 对偶变量通过网络协调任务分配 | [State-augmented multi-agent assignment, L4DC 2024](https://proceedings.mlr.press/v242/agorio24a.html) | “价格 Token + 分布式任务分配”已有直接先例 |
| 约束 MAPPO/安全 MARL | [MAFOCOPS, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7b64c47dcb067efd6be5eee854c14835-Abstract.html)；[Scal-MAPPO-L, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fa76985f05e0a25c66528308dda33de0-Abstract-Conference.html) | Lagrangian/约束 Critic 不是核心创新 |
| 概率/CVaR 约束 | [Primal–dual constrained MARL interpretation, L4DC 2023](https://proceedings.mlr.press/v211/tabas23a.html)；[RiskQ](https://arxiv.org/abs/2311.01753) | CVaR Critic 和 worst 加权属于已有风险敏感 MARL 范畴 |
| 每项任务恰需两个节点的联盟拍卖 | [Distributed auction with two-robot coalitions, IEEE T-RO 2024](https://ieeexplore.ieee.org/abstract/document/10706011) | “两节点联盟/三集合打包”本身已有直接先例 |
| 通信故障和事件触发下的联盟形成 | [Communication-aware coalition formation, IEEE Access 2021](https://doi.org/10.1109/ACCESS.2021.3061149) | “事件 Token + 通信感知联盟”本身也不够 |
| UAV 感知任务与发射功率联合优化 | [Multi-UAV collaborative sensing and communication, IEEE TWC 2023](https://arxiv.org/abs/2207.04498) | 联合任务/功率分配已有高度相关工作 |
| UAV-ISAC 轨迹、波束/功率和 sensing QoS | [Energy-aware UAV-ISAC, 2023](https://arxiv.org/abs/2302.10124) | 单纯增加联合优化变量不能构成创新 |
| Attention-MAPPO 用于多 UAV 感知/通信/计算 | [Trajectory design and resource allocation, 2024](https://arxiv.org/abs/2410.04151) | “MAPPO + Attention + ISAC”已存在 |
| 多节点 ISAC 图学习、模式选择、关联和估计 | [Graph Learning for Cooperative Cell-Free ISAC, IEEE TWC 2026](https://ieeexplore.ieee.org/abstract/document/11449475/) | “异构图 + ISAC 联合优化”也已形成直接竞争 |
| 多基地检测与功率分配 | [Multi-static target detection and power allocation](https://arxiv.org/abs/2305.12523)；[Hybrid radar fusion, IEEE TWC 2024](https://ieeexplore.ieee.org/abstract/document/10417003) | 多基地检测/集中融合是物理基础，不是算法新颖性 |

## 4. 现有部署版的创新性评级

| 方面 | 评级 | 理由 |
|---|---:|---|
| 工程完整性 | 4/5 | 同时实现本地 Actor、真实 U2U 传输、bit/时延/功率、双基地检测、固定测试库和严格资源核算 |
| 系统建模 | 3.5/5 | “无地面通信 + U2U 辅助感知 + 每 UAV 共用 1 W”具有清晰应用特色 |
| 单模块算法新颖性 | 1.5/5 | 几乎每个组件都能找到直接先例 |
| 组合方法新颖性 | 2/5 | 当前核心增益主要来自集中式 deficit max-min P0，而非被证实有效的新型分布式学习机制 |
| 证据严谨性 | 3.5/5 | 已有分层种子、LCB/CVaR、配对干预和失败模块审计，但正式样本仍需扩到 100 种子 |
| 扩展为论文主线的潜力 | 4/5 | 配对独立上界显著，且恰好对应角色非对称超边这一尚未解决的结构瓶颈 |

## 5. 为什么应该扩大到“双基地超边协商”

### 5.1 已有因果审计

旧的 10 种子末帧分解中，只改变配对就把 `worst` 从 `0.399` 提到
`0.881`，而只改变功率仅到 `0.519`。

最初的 20/100 种子物理 Oracle 把不同接收机的 deflection 直接求和，而部署环境
使用 `local_only` 检测，即每个目标只能使用单个接收机内累积的证据。因此旧表把
“集中式证据融合”误计入了 pair-only 收益，不能继续作为 Gate A 证据。对应文件
`architecture_v2_deployed_physical_oracle_gate20` 和
`architecture_v2_reproducible_physical_oracle_gate100` 只保留为无效尺子的诊断记录。

修正后，pair MILP 显式选择每个目标的接收机所有者；发射机证据只能在该接收机内
累积。功率 MILP、单角色联合上界和全双工上界使用同一融合边界。加载归档
`best_restored.pt`，冻结策略且不训练，在前 30 个固定 test 种子上得到：

| 30-seed same-frame controller | worst | gap vs deployed | gap 95% CI | QoS feasible |
|---|---:|---:|---:|---:|
| 当前配对/功率 | 0.650 | -- | -- | -- |
| 仅优化端点/配对，固定当前功率 | **0.862** | **+0.213** | **[0.115, 0.321]** | **0.867** |
| 仅优化功率，固定当前配对 | 0.729 | +0.080 | [0.038, 0.127] | 0.733 |
| 联合配对与功率，单角色 | **0.910** | **+0.260** | **[0.147, 0.387]** | **0.867** |
| 联合配对与功率，全双工 | 0.926 | 单角色之上 +0.0157 | [0.002, 0.036] | -- |

融合一致的 pair-only 增益下界 `0.115 > 0.03`，且平均增益约为 power-only
的 2.7 倍；全双工相对单角色的平均增益没有达到 `0.03` 门槛。因此当前证据支持
把首要变量收窄为**单角色端点分配与接收机归属**，但不支持把全双工作为主线。
正式 100 种子 Gate A 仍应在源码冻结后完成，30 种子结果只用于架构准入。

融合一致结果文件：
`results/architecture_v2_fusion_consistent_oracle_gate30/paired_eval.csv`。

需要单独记录一个复现限制：归档检查点在当前代码上的 episode 指标为
`0.833/0.777/0.528`，没有复现历史 `0.890/0.854/0.679`。两个 manifest
除新加入但关闭的模块、warm-start 路径和运行帧计数外没有实质参数差异。根因是
Architecture V2 产生历史结果时仍位于未提交工作树；manifest 只保存了基底提交
`fa16abf`，而没有保存当时的源码快照。因此：

- 历史高分保留为工程记录，暂不能作为最终论文可复现证据；
- 30 种子融合一致 Gate A 的结构性结论有效，因为控制器在每个种子上共享完全
  相同的同帧状态；
- 从 Gate B 起，所有方法必须以当前冻结、可复现执行链为共同基线，并保存源代码
  commit、完整配置、Actor/运行时状态和逐 episode 结果。

### 5.2 QPD 失败也支持改变变量

现有 node-target QPD 用每行独立更新和 Top-k 舍入，无法保证两个不同 UAV 同时
在同一目标上形成 Tx/Rx 会合。双基地收益并不是

\[
D_{i,q}+D_{j,q},
\]

而是依赖 Tx、Rx、目标三者几何及角色的不可分离函数

\[
D_{i,j,q}=g(p_i,p_j,p_q,\rho_i^{\rm sense},\text{channel}).
\]

因此失败不是“QPD 步长没调好”，而是决策变量不表达真实物理对象。继续优化
`x_{i,q}` 或增加 per-target MLP，不会自动修复这一结构错配。

### 5.3 融合一致、带滞回的集中式结构教师

进一步审计发现，当前 P0 的 max-min 目标仍把不同接收机的证据相加，而部署检测
使用 `local_only`。为隔离这一错配，实现了默认关闭的 receiver-owner P0，并做了
三个逐步对照：

1. **过滤承诺图 + receiver-owner**：10 种子 `worst=0.440`。目标一旦被 learned
   Top-2 裁掉，严格最小值退化；
2. **完整候选图 + 每帧重解**：`worst=0.654`、CVaR `0.145`。覆盖恢复，但角色与
   接收机所有者逐帧抖动，安全场景退化；
3. **完整候选图 + 5 帧保持**：同时修复融合边界、图可达性和切换抖动。

第三个对照在完整 100 个固定 test 种子上得到：

| controller | steady | weak3 | worst | CVaR | QoS feasible | bit/frame |
|---|---:|---:|---:|---:|---:|---:|
| 冻结可复现基线 | 0.833 | 0.777 | 0.528 | 0.057 | 0.46 | 768.0 |
| receiver-owner full-graph hold-5 | **0.913** | **0.884** | **0.751** | **0.354** | **0.73** | **712.7** |

配对增益及 95% bootstrap CI 分别为：

- steady：`+0.080 [0.053, 0.108]`；
- weak3：`+0.107 [0.070, 0.144]`；
- worst：`+0.223 [0.149, 0.296]`。

QoS 新增成功 34 个、损失 7 个，exact McNemar `p=2.53e-5`。单 UAV 1 W
约束误差仍为 `4.44e-16`。代价是集中式求解每帧均摊从 `1.44 ms` 增至
`3.29 ms`，每次重解约 `15.9 ms`。

该版本已经在 100 种子上达到 Medium 三项性能门槛，但它读取完整物理候选图，
因此只能定义为**结构教师/集中式强基线**，不能定义为分布式部署方法。它给出了
下一阶段必须同时逼近的三个机制：

\[
\boxed{
\text{receiver ownership}
+\text{role-feasible graph recovery}
+\text{hysteretic switching}
}
\]

## 6. 建议的新问题与算法边界

### 6.1 超边变量

定义有向双基地超边：

\[
y_{i,j,q}^t\in\{0,1\},\quad i\ne j,
\]

表示 UAV \(i\) 作为 Tx、UAV \(j\) 作为 Rx，共同感知目标 \(q\)。
约束至少包括：

- 单 UAV 单时隙角色/容量约束；
- 每目标最少/最多端点约束；
- 只允许由实际到达 Token 构成的可见超边；
- 每 UAV 通信与感知总功率的 `1 W` 单纯形约束；
- 必要时的切换、保持和安全运动约束。

### 6.2 内生通信代价

通信变量不只是目标函数中的固定罚项。若 UAV 增加 Token bit 或发送功率，
它会同时：

1. 提高邻居看见该超边报价的概率、降低报价过期风险；
2. 增加序列化时延和拥塞；
3. 减少本 UAV 当帧可用于感知的功率。

因此通信动作会改变**信息集、可行超边集和物理检测收益**。这是区别于一般
通信感知联盟与一般 ISAC 功率分配的关键建模点。

### 6.3 风险价格与局部报价

每个目标维护可通过 Token 一致化的缺口价格 \(z_q^t\)。候选超边报价可写为：

\[
B_{i,j,q}^t =
z_q^t\widehat{\Delta D}_{i,j,q}^t
-\lambda_b C_{i,j,q}^{\rm bit}
-\lambda_\ell C_{i,j,q}^{\rm delay}
-\lambda_p C_{i,j,q}^{\rm RF}
-\lambda_s C_{i,j,q}^{\rm switch}.
\]

学习模块只估计局部不可解析的边际检测价值
\(\widehat{\Delta D}_{i,j,q}\)，分布式拍卖/原始–对偶层负责形成角色一致的
超边并执行硬投影。这样 MAPPO 从最终动作生成器退回到局部价值估计器，减少
学习搜索空间。

### 6.4 论文贡献必须满足的四项

1. **问题贡献**：明确形式化内生通信、非对称双基地联盟和 Worst/CVaR 的联合
   随机优化问题；
2. **机制贡献**：提出只能使用物理到达 Token 的分布式超边协商，而不是在
   中央 P0 后再加一个学习残差；
3. **理论贡献**：至少证明硬约束可行性不变式，并在静态单时隙/有界时延条件下
   给出有限轮收敛、\(\epsilon\)-竞争性或相对集中式上界的误差界之一；
4. **实证贡献**：在未见过的规模、信道和几何下，同时改善 worst/CVaR 和
   通信代价，而不只在 4/4 固定环境中调高均值。

缺少第 2 项时，它仍是集中式 P0 的工程增强；缺少第 3、4 项时，SCI 三区以上
的说服力仍然有限。

## 7. 不建议扩大的方向

- 不再以增加 Token 维度、Cross-Attention 层数或 MLP 深度作为主线；
- 不再继续单独调 Top-1/Top-2 或固定轮次；
- 不再把普通 CVaR Critic、Lagrangian、Sinkhorn 或 Gumbel-Softmax 单列为
  contribution；
- 不重新引入依赖集中式教师的蒸馏作为部署主路径；
- 不同时加入异构 UAV、目标跟踪、地面通信和全双工。它们会扩大变量数量，
  但不会修复当前最核心的双端点结构错配；
- 在没有分布式超边机制前，不做长周期端到端 PPO 训练。

## 8. 分阶段实验与停止门槛

### Gate A：上界复核

先在冻结 Actor 上完成融合一致的 30 种子筛选；源码冻结后再完成 100 个分层测试
种子的末帧 pair-only、power-only 和 joint oracle 正式分解。

通过条件：

- pair-only 增益 95% CI 下界 `> 0.03`；
- pair-only 增益显著大于 power-only；
- 至少 70% episode 存在可由单角色超边修复的 QoS 缺口。

30 种子筛选已通过 pair-only 余量和单角色优先级条件；“pair-only 显著优于
power-only”的配对差值置信区间以及正式 100 种子复核仍待完成。后续不得用上界
证明部署性能，只能把它作为 Gate B 的 oracle-gap 分母。

### Gate B：无学习的分布式超边协议

先实现 2--3 轮、物理 Token 约束下的确定性超边拍卖，不训练神经网络。

10 种子准入条件：

- 相对当前部署版 `Δworst >= 0.03`；
- `steady`、`weak3` 各自退化不超过 `0.01`；
- QoS 可行率不低于部署版；
- `1 W` 误差保持在数值精度内；
- 通信 bit、时延和求解时间全部实际核算。

已完成的两次朴素协议筛选均失败：

- 3 标量端点代理：`0.529/0.463/0.126`，协议使用率 `0.34`；
- 7 维可重构物理状态：`0.492/0.363/0.054`，协议使用率 `0.717`。

它们都保持 1 W 约束并实际核算通信量，但局部 Top-k/互选没有显式的
“每目标单接收机所有者”和全局单角色一致性，导致更多可见边反而分散局部证据。
因此这两版 Gate B 被拒绝。下一次 Gate B 只能测试接收机所有者变量、角色价格和
容量投影，不再继续调整自由超边分数。接收机所有者 + 公共量化状态版本虽将
`worst` 从旧协议的 `0.126` 恢复到 `0.465`，但覆盖仅 `0.597`、可行率仅
`0.30`，仍被拒绝。后续学习头必须以 100 种子结构教师为监督/上界，并显式学习
所有者、角色和切换价值；不得在现有失败协议上继续堆叠。

### Gate C：学习增强而非学习替代

只有 Gate B 通过后，才用共享超边 scorer 学习
\(\widehat{\Delta D}_{i,j,q}\) 或自适应步长。必须与纯物理报价做消融。

100 种子正式准入条件：

- paired bootstrap `Δworst` 95% CI 下界 `>0`；
- CVaR 绝对提升至少 `0.05`；
- feasible rate 的 Wilson LCB 不降低；
- 通信成本–性能 Pareto 前沿优于固定 Top-1、Top-2 和当前 P0。

### Gate D：泛化

至少报告：

- 4/4 训练，6/6 与 8/8 零样本测试；
- SNR、丢包、时延、量化 bit 和 `1 W` 分配边界扰动；
- Easy/Medium/Hard 几何分层；
- 纯注意力 MAPPO、固定 Top-k、通信感知联盟拍卖、当前集中式 P0 和集中式
  MILP oracle；
- oracle-gap recovery：

\[
\frac{J_{\rm method}-J_{\rm deployed}}
{J_{\rm oracle}-J_{\rm deployed}}.
\]

## 9. 论文定位建议

部署版不应被删除。它有三种重要角色：

1. 可复现实验平台和安全回退；
2. 强基线：证明普通 MAPPO/Attention 和独立 node-target QPD 的上限；
3. 新方法失败时的保底论文材料，包括物理模型、评估协议和负结果审计。

但若目标是以“方法创新”投稿 SCI 三区及以上，推荐把新论文主角换成超边协商，
部署版改为 `Deployed-Hybrid Baseline`。最稳妥的贡献叙事不是“我们用了更多
深度学习模块”，而是：

> 现有 learned communication 不表达双基地角色可行性；现有 coalition
> assignment 不处理通信与感知共享 RF 功率；现有 UAV-ISAC 优化通常使用集中式
> 全局信息或平均目标。本文把三者统一为具有实际分组、时延、量化和尾部 QoS
> 约束的分布式有向超图协商问题。

只有当 Gate B 和 Gate D 同时成立后，才建议把这一表述作为最终论文主线。
