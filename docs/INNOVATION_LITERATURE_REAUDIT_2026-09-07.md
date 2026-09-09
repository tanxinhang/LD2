# MARL/DRL 模块级文献演进与创新必要性审计（2026-09-07）

## 0. 结论先行

上一版把文献演进写成了“别人做过什么”的功能清单，不能回答本项目为什么必须采用某个
模块。本版改用**模块故障与必要性**作为审计单位。结论有四点：

1. “已有工作使用过某模块”不构成否定；真正需要排除的是：原有 MARL/DRL 模块本来可以
   通过合理改造解决，却在没有证据的情况下直接叠加 LP、Student、证书或新通信协议。
2. 当前仓库存在两条不能混写的执行链：`Structured MAPPO` 是学习研究链；当前严格正式
   profile 则是 **hold action + 确定性 U2U 协议 + 解析结构/功率/运动控制**，不是 learned-policy
   结果。正式 runner 已在源码中明确声明这一点。
3. 现有失败并不指向一个统一的“MARL 不够强”：6/6 主要暴露结构 Student 的跨尺度校准与
   闭环耦合；目标数增加暴露 sensing capacity/fairness；CT/CA 暴露 belief 动力学失配；K=16
   暴露每个私有视图各解一次 LP 的计算扩展性。四种故障必须分别处理。
4. 因而现在还不能冻结“factor message + certificate”作为唯一创新主线。它只可能解决
   **可认证通信与执行准入**，不能解决 belief 失配、策略信用分配、结构泛化或 K=16 求解时延。

创新判断必须遵守下列顺序：

```text
可观测失败
  -> 定位到具体模块和性质
  -> 先穷尽同范式内的最小修复
  -> 证明仍缺少某项结构性质（精确性、逐时隙可行性、可认证 bit 或时限）
  -> 才允许引入外部模块
  -> 用正交消融证明新增模块只修复该故障且收益大于代价
```

## 1. 审计边界与“无法解决”的严格含义

检索窗口为 2023-09 至 2026-09，覆盖 UAV-ISAC、集中/分布式 DRL、MAPPO、学习通信、
networked sensing、offline/safe MARL。文献以 IEEE/出版社页面和 arXiv 原文为主；检索用于
建立最近邻和演进链，不冒充穷尽式专利查新。

这里不使用“某方法在数学上绝对不可能解决”这种无法证实的表述。允许引入新模块，只能基于
以下三类**性质不匹配**：

- **表示不匹配**：局部观测没有包含决定任务所需的信息。调 reward、critic 或 PPO 超参数不能
  恢复从未被观测/传输的信息。
- **保证不匹配**：期望回报或期望约束不能自动推出每帧、每目标的硬可行性。若系统要求执行前
  可验证保证，则需要投影、求解器或证书；但若只要求平均性能，外部证书并非必要。
- **计算不匹配**：一个固定结构内的凸子问题已有可在 deadline 内完成的精确解时，让神经网络
  重新近似通常没有必要；反之，若精确解超过 deadline，也不能因为“有解析解”就强行使用。

每个新模块必须通过五道门：F1 故障可复现；F2 故障可归因到该模块；F3 已比较同范式修复；
F4 新模块提供了原模块缺少的明确性质；F5 正交消融显示端到端净收益。

## 2. 首要架构事实：当前正式系统并不是 MARL 执行系统

当前 canonical profile 是 `config/exp_strict_distributed_k16q16.yaml`，继承 strict pilot。其
正式入口 `tools/run_strict_distributed_pilot.py` 明确写明：

- runner 没有 learned actor，所有 UAV 提交零位移 hold action；
- learned message 是零速率、零内容载体，`hyperedge_protocol_only` 在其上附加确定性状态包；
- 运动由 distributed bistatic controller 覆盖；
- 结构由 hyperedge/P0 路径生成；
- 固定结构内感知功率由 replicated max-min LP 生成。

学习分支中的 actor 虽有 movement、assignment、sensing、message、rate、comm-power 等头，但在
strict 执行链中：movement 不执行，role 不学习，learned latent 被 protocol-only 抑制，逐目标
sensing allocation 被 LP 覆盖。严格基线中剩余的 RF split 也是 runner 固定提交，不来自 actor。

因此目前不能同时声称“提出一个 MARL 控制算法”和“正式结果由 strict profile 证明”。正确的
论文架构只能二选一：

- **解析分布式控制论文**：删除 MAPPO 主贡献叙事，把学习模块降为离线 proposer/shadow；或
- **混合 MARL 论文**：明确 actor 独占哪些自由度，只允许解析层处理可证明需要的内层变量，
  然后重新做 learned-policy 正式认证。

在这个边界未确定前，讨论“MAPPO + LP + certificate 的联合创新”没有可识别的因果对象。

## 3. 2023–2026 的模块演进，而不是名词演进

| 时间与代表工作 | 暴露的 DRL/MARL 模块问题 | 论文优先采用的内生修复 | 尚未提供的性质 | 对本项目的含义 |
|---|---|---|---|---|
| Qin et al., TWC 2023 | 联合关联、轨迹、功率的连续联合动作维度高；集中式策略扩展差，分布式策略又面临协同困难 | SAC；利用问题对称性做 replay augmentation；比较集中与多智能体 SAC | 私有 belief、真实 U2U payload、逐目标执行证书 | 先用参数共享、置换等变和分布式 actor 修复规模问题，不能直接以“多 UAV + DRL”作贡献 |
| Mason et al., TCCN 2024 | 控制与通信被分开优化时，发送内容/时机不一定服务控制目标 | CP-POMDP，把通信 action 与物理 action 放在同一决策过程中联合训练 | 面向特定物理约束的可认证最小 bit、逐目标 QoS 保持 | “任务导向通信”可以由 MARL 内生学习；只有需要可证明决策保持时，确定性消息才可能必要 |
| Networked ISAC 2024 | 原始数据汇聚压垮 leader，并形成单点瓶颈 | adapt-then-combine，交换低维中间估计；分析收敛与 MSE | 动态控制决策保持、弱目标 max-min、FBL 下的执行准入 | 低维物理消息本身不是创新；必须说明为何估计误差指标不足以保护当前控制决策 |
| TANAGERS 2024 / Atsu et al. 2025 | 双基地 swarm 的局部可见性、随机 U2U 丢包和通信干扰 | LSTM + attention 的 emergent message；同时学习移动和通信功率/速率相关动作；丢包状态进入接收端 | 消息真实 bit 语义、最弱目标约束、硬可行性、证书 | learned communication 是必须先击败的直接内生基线，而不是因不可解释就直接排除 |
| Peng et al., TCOM 2025 | Gaussian 连续策略有边界偏差；全连接 critic 随智能体增长；异构变量相互干扰 | bounded Beta actor、attention critic、MU/UAV 异构 agent 分解；闭式可解变量不交给策略 | reward penalty 仍不是逐帧硬保证；训练信息仍较强 | 文献展示了“先修 actor/critic/action decomposition，再引入求解器”的正确顺序 |
| AERIS 2026 preprint | online exploration 有风险；离线 critic 在 OOD joint action 上过估计；纯 BC 继承旧策略缺陷 | twin centralized critic + IQL 支撑、局部候选 rectification、trust-gated distillation | 证明只覆盖 offline-support 上的 critic surrogate 期望改进，不等于每帧物理 QoS 可行 | Student/teacher、局部修复和 trust gate 也已有近邻；本项目证书必须证明其保证对象不同 |

这条演进说明：近三年不是简单地从 DRL 走到“DRL + 更多模块”，而是在逐层修复
**动作边界、实体扩展、部分可观测、通信资源、离线支撑和执行保证**。本项目必须进入同样的
模块深度，才能避免 fake innovation。

## 4. 本项目的模块级故障与必要性判定

### 4.1 任务目标与 reward：sum utility 还是 weakest-target QoS

**现有问题。** Atsu 的共同 reward 是各 UAV/目标 sensing SNR 的总和；AERIS 将通信、感知
pass/margin 与违约惩罚合为 slot reward；Peng 也通过惩罚表示 latency、碰撞和感知约束。这些
设计可提升平均任务，但 sum reward 可以牺牲少数弱目标，固定 penalty 还会随 K/Q、几何和
信道尺度改变有效权重。

**先做的 MARL 内生修复。** 直接把 episode 三地板定义成 constrained Markov game；比较
Lagrangian MAPPO、worst/soft-min utility、distributional/CVaR critic、per-target value head 和
dual-weighted advantage。只有这些方法在同观测、同动作、同预算下仍产生逐帧弱目标违约，
才能证明 reward/critic 调整不够。

**何时才需要 LP/证书。** 若要求的是“执行前保证每个目标达到硬地板”，expected return 无法
推出该性质，此时验证器或可行投影有必要；若论文只主张平均 QoS，证书不是必要模块。

**当前证据。** K=8 而 Q 从 8 增至 12/16 时 worst 从 0.616 降至 0.454/0.313，属于资源和公平
分配故障；它不能由通信压缩自动修复。应先验证 dual/worst-aware policy 或 active-set allocation。

### 4.2 观测、belief 与记忆：POMDP 的状态估计故障

**现有问题。** 当前 structured actor 已接收每目标 belief mean、covariance diagonal、AoI，使用
neighbor GRU 和 cross-attention；这比只加一层 LSTM 更完整。但 strict motion 审计固定 CV
tracker 后，CT 10–20 m/s 的 worst 仅 0.290，CA 从最低速度档就失败，且 runtime 始终达标。
这直接定位为动力学/调度鲁棒性，而不是 actor 容量或 LP 时延。

**先做的内生修复。** 比较 matched CV/CT/CA、IMM、多模型 recurrent state encoder、history
length、process-noise calibration，以及让 critic 预测 belief error/innovation 的辅助任务。要用
oracle-state actor 作为上界：若 oracle 仍失败才轮到控制策略；若 oracle 成功而 local belief
失败，则问题在 estimator/information。

**何时才需要额外通信。** 只有在本地历史无法识别模式、而远端观测显著降低 posterior
uncertainty 时，belief exchange 才必要。需要 no-comm / latent-comm / posterior-comm 的等 bit
消融。factor capability message 并不携带完整运动模式信息，不能被宣称为 CT/CA 修复。

### 4.3 Actor 与动作空间：边界、异构变量和梯度冲突

**现有问题。** 当前 actor 把移动、结构评分、感知分配、消息、码率和 RF split 置于共享表示或
关联头上。这些动作跨越不同时间尺度，且连续、离散、集合匹配和 simplex 变量混合。更严重的
是多个头在 strict execution 被覆盖，训练梯度优化的动作与最终执行动作并不一致。

**先做的内生修复。** 仿照 Peng 的顺序，先做 bounded distribution（Beta/logistic-normal）、
慢/快 actor 分离、autoregressive 或 Sinkhorn matching、per-head advantage、gradient conflict
诊断，以及 residual policy（解析基线给 nominal，actor 只学有界 residual）。必须报告每个头的
action intervention：强制改变该头时，最终物理 action 和回报是否真的改变。

**何时才需要外部求解器。** 只有某个动作子块存在可高效精确求解的结构，或离散容量约束在
actor 输出后仍频繁不可行，才把这一子块交给求解器。不能让求解器覆盖全部 head 后仍把性能
归因给 MAPPO。

### 4.4 Critic 与信用分配：集中训练并不自动解决规模泛化

**现有问题。** 当前代码已有 permutation-equivariant value critic、team/target feature 和
per-target value 输出，方向上覆盖了 attention critic 的近期演进。现在缺的不是再加 GNN/attention
名词，而是证明 critic 在不同 K/Q 上仍能正确排序 joint-action counterfactual。

**先做的内生修复。** 在 K=4 训练、K=6/8/12 测试上报告 TD error、value calibration、candidate
ranking top-k recall、permutation/cardinality consistency；比较 flat critic、attention critic、
equivariant critic 与 factorized/QMIX 类 critic。用 centralized oracle critic 区分“actor 看不到”
和“critic 教错了”。

**何时需要解析 upper bound。** 若目标是安全剪枝，learned critic 的排序精度不能构成“不漏掉
最优候选”的保证，解析 upper bound 有独立价值；但它只能作为剪枝证书，不能被说成改善了
critic 的学习能力。

### 4.5 通信 encoder/decoder：learned latent、估计消息与 capability message

**已有内生方案。** Mason 已证明通信时机/动作可与控制联合学习；Atsu 在双基地 UAV swarm
中用 LSTM-attention 产生连续消息并适应随机链路；Networked ISAC 用模型结构导出的中间估计
降低 raw-data 汇聚。这三类分别覆盖 task-oriented latent、emergent U2U 和物理中间量。

**当前项目的具体矛盾。** structured actor 能产生 target tokens，接收端有 masked cross-attention；
但 strict profile 的 `hyperedge_protocol_only` 明确抑制 learned payload，发送的是确定性 endpoint
state。于是“学习通信提升控制”和“确定性 factor 消息提供证书”属于两条替代路线，不能把两者
都写成同一正式机制的贡献。

**同范式修复优先级。** 先比较 message dropout/noise training、码率 action、event-trigger、
information bottleneck、message reconstruction/decision auxiliary loss、receiver attention，以及
将真实 FBL/BLER/AoI 放入训练。若 learned latent 在相同 over-air bit 下已经保持动作与 QoS，
就没有必要另造 factor protocol。

**capability message 真正可能必要的条件。** 论文要求的是：给定物理不确定集，在发送前计算
足以保持 downstream argmax/feasibility decision 的 bit 下界，并在无法证明时 fail closed。
黑盒 latent 通常不直接给出这种可验证性质；此时 factor + outward interval quantization 才有
清晰角色。它的贡献对象是**可认证决策充分性**，不是“压缩消息”或“任务导向通信”本身。

### 4.6 结构分配：Student、matching 与 P0 的责任冲突

**当前证据。** 6/6 blind100 中原 Student QoS 为 0.530；teacher 在 47 个失败 seed 中改善
43 个并达到 0.723。multi-scale CE 把点估计推到 0.780，但 LCB 仍为 0.689。仓库自己的结论是
Student 误差 × 弱几何 × 闭环敏感性，而不是“神经网络容量不足”这一单因解释。

**先做的内生修复。** 先比较 set-equivariant Student、capacity-aware matching head、
autoregressive endpoint assignment、DAgger/on-policy relabel、task-regret/dual-weighted loss、
uncertainty-aware abstention。需要把三类错误分开：candidate recall、edge ranking、hard matching
projection。若 teacher 候选不在 Student pool，改 loss 无效；若候选在但排序错，才是 critic/
Student 校准；若排序对但投影后错，才是 matching 模块。

**何时需要 P0/局部搜索。** endpoint capacity、单角色和覆盖约束要求每帧可行，而 Student
持续产生不可行结构时，最小可行投影有必要。但若 P0 每帧完整替代 Student，Student 就只是
提议器，必须以“相对 P0 减少多少计算/通信且保留多少质量”评价，不能归因总体 QoS。

### 4.7 固定结构内功率：为什么 LP 合理，但当前实现仍未过关

固定结构与几何下，deflection 对 sensing power 线性，max-min allocation 是结构明确的 LP。
若它在 deadline 内求解，直接求精确解比让 actor 学一个近似 simplex 更合理：它提供预算平衡、
弱目标最优和 primal/dual 可核验性。这是性质驱动的模块化，而非因为“传统算法更可靠”。

但当前 K=16 每个节点基于不同私有 cache 各解一个 LP，最终 P95 仍为 127.3 ms，高于 100 ms；
因此“解析”并不等于“可部署”。应先做结构化闭式/active-set、warm start、增量对偶、deadline
incumbent 和复杂度界。若精确 LP 持续超时，才比较 amortized learned solver，并保留可行投影与
事后 gap，而不是无条件坚持 16 次精确解。

### 4.8 运动模块：MARL、解析 L3 与 model-predictive residual

strict profile 的运动由 bistatic bottleneck 规则覆盖 actor。它在静态/慢速几何可工作，但 CT/CA
失败说明“当前 belief 下的一步弱目标趋近”不足以应对模式变化。直接加入更多 L3 heuristic 会
继续扩大创新孤岛。

正确比较顺序是：learned movement only → matched-belief learned movement → analytical L3 only
→ analytical nominal + learned residual → short-horizon model-predictive actor。安全边界投影可统一
保留。只有证明 learned residual 无法满足逐帧 kinematic/coverage constraint，解析 safety filter
才是必要层；不能用 safety filter 替代长期策略后还称其为 actor 改进。

### 4.9 安全层与证书：必须区分保证对象

AERIS 的定理是在离线数据支撑状态分布上，假设 critic 局部排序、路径正则性和 distillation
误差有界时，保证 centralized-critic surrogate 的**期望改进**。它不等于每帧每目标 QoS 保证；
其奖励仍用 sensing/collision violation penalty。Anytime-constrained MARL 则说明 joint action
硬约束本身可能带来计算困难。

本项目至少有四类不同“安全”概念，不能混成一个 certificate：

1. actor 更新安全：KL/trust region；
2. offline support 安全：不信任 OOD critic；
3. 物理动作可行：功率、速度、角色、deadline；
4. 任务 no-harm：相对 baseline 的逐目标 QoS 不降低。

composable certificate 只在其观测、量化、belief coverage 和消息新鲜度假设下认证贡献下界；
它不会修正错误 belief，也不会使长期 episode 回报单调。证书只有在系统明确要求 3 或 4 时才
必要，而且必须报告 abstention/false-certificate/coverage，而不只报告通过后的性能。

## 5. 当前失败到模块的重新归因

| 已观察失败 | 首要责任模块 | 首先排除的替代解释 | 当前不应添加的模块 | 下一项有判别力的实验 |
|---|---|---|---|---|
| 6/6 Student QoS 0.530；CE LCB 0.689 | 结构 Student/candidate/matching/闭环转移 | actor 容量、candidate recall、projection 三者混淆 | 新通信 certificate | teacher-forced pool × Student rank × exact matching 三因子消融 |
| K=8,Q=12/16 worst 0.454/0.313 | sensing capacity + weakest-target allocation | 是否存在物理可行 witness | 更深 critic 或更多 token | oracle feasibility + dual/worst-aware allocation + active-set |
| CT/CA worst 0.290/0.338 及更低 | belief motion model + horizon | runtime 已达标，故非计算瓶颈 | factor message、LP 证书 | oracle-state / matched-model / IMM / recurrent belief 四级对照 |
| K=16 sensing好但 P95 127.3 ms | replicated private-view LP + local projection | 不是中心信息缺失，也不是 policy inference | 额外 neural head | solver profile、warm-start/active-set、deadline incumbent、learned solver A/B |
| M4-D 无自然 L2 candidate | candidate generator/问题定义 | 不是 factor rate 本身失败 | 人工构造可通过候选 | 先建立不看 factor/dense 编码的自然 candidate provenance |

这张表说明：当前没有任何一个观测结果证明“只有 factor communication + certificate 才能解决
系统性能”。它们解决的是一个更窄的、尚待 live 闭环证明的通信保证问题。

## 6. 创新主张的重新分级

### 6.1 已被否定的主张

- 多 UAV + ISAC + MARL/CTDE；
- attention/equivariant actor-critic；
- learned U2U communication 或 task-oriented communication；
- MAPPO 与解析优化器混合；
- Student/teacher、局部候选修复、trust gate 或“加安全层”；
- 低维物理中间量交换。

这些不是“别人做过所以不能用”，而是它们只能作为模块选择，不能单独解释新的科学性质。

### 6.2 有条件成立的候选贡献

**候选 A：可认证的 decision-sufficient physical message。** 与 learned latent 和 estimation
message 的区别必须是：从 bistatic physics 导出 exact factorization；把 task distortion 定义为
downstream weakest-target decision 改变；给出 outward interval、margin-bit sufficient bound 和
fail-closed 规则。当前 M4-B/C 只支持固定计划/route 的机制证据，M4-D 尚未形成 live 结论。

**候选 B：私有 belief 下的 composable QoS admission。** 价值不在“用了 certificate”，而在
局部 contribution interval 能否在不共享完整全局状态时组合为逐目标下界，并与原子提交绑定。
必须严格区分 conditional robust set 与无条件真实世界保证，并与 centralized projection、
Lagrangian MAPPO、AERIS 类 trust-gated rectification 比较。

**候选 C：异构时间尺度的可识别混合控制。** 若最终保留 MARL，可以把长期运动/通信调度交给
policy，把固定结构凸功率交给 LP，把硬边界交给 safety filter；关键创新必须是可证明的接口与
端到端 credit consistency，而不是三个模块并列。当前实现因多头被覆盖，尚未达到该条件。

## 7. 在继续理论/算法优化前必须完成的判别实验

1. **责任矩阵**：对每个 actor head 做 intervention，记录最终执行动作是否改变；删除或冻结所有
   被确定性层完全覆盖的 head。
2. **四段 oracle ladder**：oracle state、oracle candidate pool、oracle rank、oracle feasibility
   projection，逐层确定性能损失发生在哪一段。
3. **通信三基线**：无通信、等 bit learned latent、等 bit factor interval；统一 FBL、BLER、AoI、
   RF power 和 decoder 计算预算。
4. **约束三基线**：reward penalty/Lagrangian、可行投影、certificate admission；分别报告平均
   QoS、逐帧违约、拒绝率和计算时延。
5. **求解器决策门**：LP exact、warm-start active-set、deadline incumbent、learned amortized
   solver；只有在相同可行率下比较 QoS/时延。
6. **跨尺度与动态测试**：K/Q cardinality、CV/CT/CA motion 和信道预算必须正交变化，禁止用
   一个改动同时掩盖结构泛化、belief 失配和资源不足。

只有上述实验表明：同范式修复无法提供所需性质，而新增模块以可接受代价补上该性质，才进入
理论推导和算法优化。否则优先简化架构，而不是继续增加模块。

## 8. 重新审计后的最终判定

当前系统的工程与物理建模深度较强，但**算法创新仍未闭环**。最主要的问题已经不是文献覆盖
不足，而是论文对象不唯一：learned MAPPO 分支和 strict analytical baseline 被同一“系统”名称
包裹，导致性能、创新和责任边界不可识别。

进一步脱水后，结论应写得更严格：截至当前，没有一项算法主张同时满足“新数学对象、同预算
必要性、独立盲测收益”三项条件。因而“决策权一致的多时间尺度混合控制”只能作为架构描述，
“candidate-aware、hold-aware 的 episode-QoS-regret 排序接口”只能作为待验证假设，不能在摘要
或贡献列表中写成已证实的算法创新。LP、projection、factor message、certificate、attention
和 MAPPO 均降为性质驱动的模块选择或工程集成。

建议暂时冻结如下问题，而不是冻结某个解法：

> 在私有 belief、有限 U2U bit 和逐目标 QoS 要求下，MARL 的哪一个信息或保证缺口无法由
> recurrent/equivariant actor-critic、constrained objective 与 learned communication 在同预算下
> 修复；物理 factor message、解析内层和 composable certificate 分别以什么最小代价补上该缺口？

在责任矩阵和 oracle ladder 完成前，不新增理论模块。完成后再根据故障结果选择一条主线：
belief-aware MARL、可认证通信、结构 Student，或 deadline-aware distributed optimization；不再把
四条线同时包装为一个算法贡献。

阶段 1 actor-head responsibility matrix 已完成，结果见
[`ACTOR_HEAD_RESPONSIBILITY_AUDIT_2026-09-07.md`](ACTOR_HEAD_RESPONSIBILITY_AUDIT_2026-09-07.md)。
三 seed × 20 帧干预确认 strict profile 中只有 communication-power 输入能改变 RF 记账，其余
actor-like 字段均被正式 runner 固定或被解析执行层覆盖；这不等于这些 head 在其他 research
profile 中无用。下一阶段按计划进入 oracle ladder。

阶段 2 oracle ladder 已完成，完整定义、18 个未参与训练 seed 的结果和归因见
[`ORACLE_LADDER_AUDIT_2026-09-07.md`](ORACLE_LADDER_AUDIT_2026-09-07.md)。结论不是
“oracle 越强越好”：state 干预无增益；完整 candidate pool 使 episode-worst 提升
`+0.00561`、weak3 提升 `+0.01169`；exact rank 提升 steady/weak3 但使 worst 和
episode QoS 可行率下降；projection 干预为严格 identity。因此当前最有证据的候选
主线应精炼提升为分层系统创新：以**决策权一致的多时间尺度混合控制**为核心架构，
以 **episode-QoS-regret 对齐的 candidate localization + 稳定排序接口**作为已被
ladder 指向的性能增强；factor message/certificate 保留为条件性保证层，只有在等 bit
闭环对照证明普通 Student 无法保持决策或约束性质后，才提升为正式贡献。暂不把这些
模块并列包装，也不提前删除任何尚未完成必要性检验的扩展。

随后完成的 candidate provenance × task-regret rank 正交消融进一步表明两者存在显著
交互：CE rank 使用 full candidate 时 worst 提升 `+0.00561`，但 task-regret rank 使用
同一 full candidate 时 worst 下降 `−0.02182`；task-regret 仅在 local candidate 上带来
`+0.01639` worst 提升。故性能创新应精炼为 **candidate-aware、hold-aware 的
episode-QoS-regret rank interface**，而不是把 task-regret loss 当作可独立插拔模块。
完整结果见 [`CANDIDATE_REGRET_ABLATION_2026-09-07.md`](CANDIDATE_REGRET_ABLATION_2026-09-07.md)。

## 9. 主要文献

- Y. Qin et al., “Deep Reinforcement Learning Based Resource Allocation and Trajectory Planning in
  Integrated Sensing and Communications UAV Network,” IEEE TWC 22(11), 2023,
  [DOI](https://doi.org/10.1109/TWC.2023.3260304).
- F. Mason et al., “Multi-Agent Reinforcement Learning for Coordinating Communication and Control,”
  IEEE TCCN 10(4), 2024, [论文](https://arxiv.org/abs/2302.14399),
  [DOI](https://doi.org/10.1109/TCCN.2024.3384492).
- “Networked Integrated Sensing and Communications for 6G Wireless Systems,” 2024,
  [原文](https://arxiv.org/html/2405.16398v1).
- Z. Peng et al., “Trajectory Design and Resource Allocation for Multi-UAV-Assisted Sensing,
  Communication, and Edge Computing Integration,” IEEE TCOM, 2025,
  [原文](https://arxiv.org/html/2410.04151v1).
- A. Atsu et al., “Reinforcement Learning for Enhancing Sensing Estimation in Bistatic ISAC Systems
  with UAV Swarms,” 2025, [原文](https://arxiv.org/html/2501.06454v1).
- Z. Wang et al., “AERIS: Offline Policy Improvement for Multi-UAV Integrated Sensing and
  Communication,” preprint, 2026, [原文](https://arxiv.org/html/2608.25477v1).
- J. McMahan and X. Zhu, “Anytime-Constrained Multi-Agent Reinforcement Learning,” 2024,
  [论文](https://arxiv.org/abs/2410.23637).
