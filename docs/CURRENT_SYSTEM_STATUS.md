# 当前系统状态与研究决策

> 状态日期：2026-08-01（本页聚焦 4/4/6/6 结构协调器主线）。
>
> **⚠ 8/8 解析功率/几何链（D0.87–D0.95）未并入本页**，最新总纲见
> [`SYSTEM_OVERVIEW_AND_ROADMAP.md`](SYSTEM_OVERVIEW_AND_ROADMAP.md)。要点：8/8 在
> `analytical_sensing_power_enabled` + `analytical_comm_power_enabled` +
> `analytical_movement_enabled` 解析栈上，20 seed 达 worst 0.662 / weak3 0.724 /
> steady 0.808（三 QoS 地板全达标）；详见
> [`D095_JOINT_L2_L3_ALTERNATING.md`](D095_JOINT_L2_L3_ALTERNATING.md)。

> 本文是当前代码、模型和实验结论的快速入口。演进过程和失败实验保留在
> [ARCHITECTURE_V2_RESULTS.md](ARCHITECTURE_V2_RESULTS.md)，基础环境公式见
> [SYSTEM_MODEL.md](SYSTEM_MODEL.md)。若历史章节与本文表述冲突，以本文为准。

## 1. 一句话结论

当前系统已经形成一个可冻结的 `4 UAV / 4 target` 分布式部署版本；它在正式
100 种子协议上达到 `steady=0.913`、`weak3=0.885`、`worst=0.739` 和
`QoS feasible=0.72`。新增的基数门控残差保持了 4/4 锚点逐回合完全不变，并将
6/6 的平均 worst 从 `0.543` 提高到 `0.645`，但 6/6 可行率仍只有 `0.60`。
因此：**4/4 版本继续冻结，基数残差仅保留为跨尺度候选，暂不升级部署，也暂不
对 8/8 作成功宣称。**

## 2. 当前场景和系统边界

- 场景包含 `K` 架 UAV 和 `Q` 个运动目标，UAV 在二维平面运动并对目标执行双基地/
  多基地协同感知。
- 只建模 UAV 节点间通信，不考虑 UAV 与地面的通信。
- 每架 UAV 的通信功率与各目标感知功率之和严格为 `1 W`；网络输出资源倾向，
  单纯形投影负责满足硬预算。
- 通信可发生多轮、面向多个邻居；发送功率、Token 数和精度可以自适应，传输具有
  比特、功率、时延及丢包代价。
- Token 内容通过训练得到并实际接入运动、感知及结构决策网络；当前结构协议还保留
  端点/竞标等可解释字段，防止自由 Token 退化为无意义标识。
- 环境仍使用解析链路与集中式证据融合计算最终检测概率，不生成原始 IQ 波形，也不
  实现完整通信协议栈或飞控硬件接口。

当前系统的真实边界应表述为：

```text
分布式局部观测与共享参数策略
  + 物理 U2U Token 传递
  + 分布式结构 Student 近似配对/资源决策
  + 环境级集中式证据融合与检测评价
```

冻结的集中式结构控制器仍作为教师和参考上界，不属于部署执行路径。CTDE Critic 只在
训练时使用全局信息，执行时每架 UAV 只使用本地观测、局部历史和已送达 Token。

## 3. 当前部署架构

### 3.1 分布式 Actor

1. 共享的 per-target scorer 对每个目标独立估计局部感知边际价值。参数在目标间共享，
   不等于共享目标决策，也不会破坏分布式执行。
2. 集合/注意力编码处理变长 UAV 和目标集合；不再依赖固定长度 UAV 身份 one-hot，
   对节点和目标置换保持等变或不变。
3. 运动头由目标承诺和 Token 信息共同驱动；径向分量受运动学约束，切向分量用于形成
   双基地几何基线。
4. 通信速率和总通信资源由集合池化头输出，不依赖固定 `Q`。自适应 4--8 bit 协议在
   稳定场景减少开销，在危机目标上保留较高精度。
5. 感知、通信功率统一投影到每 UAV 的 1 W 单纯形上。

### 3.2 结构 Student 与集中式控制边界

集中式冻结控制器提供结构目标；分布式 Student 利用本地状态和收到的 Token 近似其
端点选择与结构解码。部署版不是“完全去中心化检测器”：决策路径分布式，但最终
`P_D` 仍由环境级证据融合模块统一计算。这一边界必须在论文中明确。

### 3.3 锚点保持的基数门控残差

跨尺度候选不直接重训或覆盖 4/4 基模型，而是在冻结锚点外增加残差：

```text
E(K,Q) = E0 + g(K,Q) * DeltaE
D(K,Q) = D0 + g(K,Q) * DeltaD
g(4,4) = 0,  g(6,6) = 1
```

其中 `E0/D0` 为冻结 4/4 Student，`DeltaE/DeltaD` 只在非锚点数据上训练。
这个设计把“4/4 不退化”从奖励期望提升为架构保证：在 4/4 上残差严格关闭，模型输出
和逐回合评价数组与原 Student 完全一致。

实现位于
[`uav_isac/agents/frozen_structure_student.py`](../uav_isac/agents/frozen_structure_student.py)，
训练入口为
[`tools/train_multiscale_structure_student.py`](../tools/train_multiscale_structure_student.py)。

## 4. 当前可复核结果

### 4.1 结果总表

| 协议与版本 | 种子 | steady | weak3 | mean worst | CVaR | QoS feasible | 决策 |
|---|---:|---:|---:|---:|---:|---:|---|
| 4/4 冻结部署版，50 kbit/s，自适应 4--8 bit | 100 | 0.913 | 0.885 | 0.739 | 0.277 | 0.72 | 当前正式版本 |
| 4/4 基数残差安全门诊断 | 10 | 0.954 | 0.939 | 0.858 | 0.477 | 0.90 | 与原 Student 逐回合完全相同 |
| 6/6 原跨尺度 Student | 10 | 0.893 | 0.787 | 0.543 | 0.158 | 0.60 | mean worst 不达标 |
| 6/6 锚点保持基数残差 | 10 | 0.895 | 0.802 | 0.645 | 0.202 | 0.60 | 均值达标、可行率未达标 |
| 6/6 冻结集中式控制 | 10 | 0.896 | 0.792 | 0.635 | 0.065 | 0.70 | 参考控制，不是部署策略 |

注意：第一行是正式 100 种子结论；其余 10 种子结果是机制筛选，不能与正式结果混称。

> **⚠ 2026-08-16 污染披露**：下表 6/6 三行（0.543/0.645/0.635）来自
> `architecture_v2_scale_k6q6_structure_student_adaptive_b4b8_gate10` 与
> `..._cardinality_residual46_gate10`，其 paired_eval.csv 的 10 个种子**含全部 5 个
> 被隔离种子**（795/747/105/860/2，980_k6q6 test split 前 10 个）——不可作为正式
> 证据；"6/6 暂不升级"的决策需在 bank 回填后重跑确认（见 [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md)）。

### 4.2 配对统计

在相同的 6/6 十个种子上，基数残差相对原跨尺度 Student：

- mean worst 提升 `+0.1025`；
- 配对 bootstrap 置信区间为 `[+0.0499, +0.1574]`；
- 9 个种子改善，0 个退化，1 个持平；
- 但 `QoS feasible=0.60 < 0.70`，因此不满足升级门槛。

基数残差相对 6/6 集中式参考的 mean worst 仅高 `+0.0105`，置信区间
`[-0.1377, +0.1360]` 跨零，不能宣称优于集中式控制。

### 4.3 资源与约束

4/4 正式部署版平均总通信量为 `1143.259 bit/frame`，其中结构通信约
`419.534 bit/frame`，平均选择精度为 `7.883 bit`。结构发送具有原子性和缓存约束；
每 UAV 总功率误差不超过 `4.44e-16`。

### 4.4 6/6 局部候选与有限轮协调审计

局部候选审计不训练新网络，而是检查：只使用本地状态、实际收到的目标 Token，以及在
局部邻居集合上重新计算的聚合量，能否生成足以承载高质量接收端所有权解的稀疏超边
候选。开发 10 种子上的原始 exact-edge Gate A 保持为失败，不能用事后指标覆盖。

在与开发集不重叠的后 10 个种子上，事前固定的等价证据 Gate A2 使用
`L_u=4, L_q=4`、单次价值/覆盖重排和 `0.95` 证据比例。warm-start 后：

- 任意 owner 等价目标召回为 `0.992`，同 owner 等价目标召回为 `0.952`；
- owner 召回为 `0.996`，空目标率为 `0.00056`；
- 精确边召回仍只有 `0.869`，所以严格 Gate A 仍然失败；
- A2 的等价目标、owner、空目标和性能剪枝检查全部通过。

独立 10 种子上的同求解器重放为：

| 控制器 | steady | weak3 | mean worst | CVaR | QoS feasible |
|---|---:|---:|---:|---:|---:|
| 完整物理图教师参考 | 0.880 | 0.762 | 0.536 | 0.211 | 0.40 |
| 候选受限教师参考 | 0.880 | 0.769 | 0.551 | 0.211 | 0.40 |
| 候选 Student + 集中式可行投影（归因对照） | 0.877 | 0.764 | 0.558 | 0.197 | 0.50 |
| 1 轮局部价格协商 | 0.740 | 0.502 | 0.266 | 0.053 | 0.20 |
| 3 轮局部价格协商 | 0.727 | 0.494 | 0.267 | 0.073 | 0.10 |
| 公共候选图复制式确定性求解（复杂度对照） | 0.875 | 0.752 | 0.543 | 0.197 | 0.50 |

这组归因给出三个明确结论。第一，候选剪枝和冻结 Student 边值不是主要瓶颈；第二，
简单目标价格/角色价格更新损失约 `0.29` mean worst，且增加轮数不单调改善，应停止继续
调价格步长和奖励权重；第三，各节点得到同一候选图后独立运行同一确定性求解，mean
worst 距候选受限教师参考仅 `0.007`，说明给定当前候选图和同状态轨迹时，缺失的是
可扩展的组合协调器，而不是更多局部特征。
目标缺口队列对照也没有改善 CVaR，因此不进入主路径。

这里的“教师参考”不是数学上界：Candidate Student 加集中式投影的 worst 为 `0.558`，
高于固定教师的 `0.551`，表明二者的边值或优化目标不同。只有在相同状态、相同边值、
相同目标并证明候选集合上的全局最优后，才能使用“候选上界”这一术语。

物理 Token 拓扑审计显示，在 300 个非空求解帧中，一轮候选泛洪后的公共视图率和边
一致率均为 `0.9967`，第二至第四轮没有额外收益。冷启动首次求解仍以 fail-closed 处理，
占全部求解帧 `3.23%`；新协商字段目前只做了离线重放和通信量估算，尚未接入环境的
功率、时延与丢包过程。因此 **Gate A2 通过不等于端到端协议门通过**。

可复现报告见 [A2 候选审计](../results/architecture_v2_scale_k6q6_local_candidate_value_lu4_lq4_r1_gate_a2_holdout10/summary.json)、
[价格协议与泛洪审计](../results/architecture_v2_scale_k6q6_price_protocol_gossip_audit_holdout10/summary.json) 和
[复制式求解对照](../results/architecture_v2_scale_k6q6_replicated_consensus_holdout10/summary.json)。
已有 v1 JSON 中的 `candidate_upper_minus_protocol` 是历史字段名；后续 v2 输出已更名为
`candidate_reference_minus_protocol`，不改变已冻结数值。

### 4.5 Gate C1 因子图协调器首轮筛选

实现了 UAV 节点、目标节点和 `(Tx,Rx,target)` 超边节点组成的共享参数有限轮因子图，
以及只执行角色、owner、容量裁剪和确定性 tie-break 的轻量投影。训练使用前 10 个种子
中的 8 个，2 个验证；后 10 个种子首次评估后又用于多项结构筛选，因此以下只能称为
开发 holdout，不能再作为最终独立测试。

| C1 开发筛选 | 参数量 | steady | weak3 | worst | CVaR | QoS | repair rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原始 3 轮因子图 | 105k | 0.840 | 0.684 | **0.446** | 0.124 | 0.40 | 0.176* |
| 一次联合 logits 耦合 | 105k | 0.806 | 0.633 | 0.408 | 0.124 | 0.40 | 0.181 |
| 3 轮均值场耦合 | 105k | 0.828 | 0.662 | 0.417 | 0.124 | 0.40 | 0.157 |
| 12k 紧凑模型 | 12k | 0.815 | 0.652 | 0.418 | 0.085 | 0.30 | 0.130 |
| 跨目标全局上下文 | 124k | 0.838 | 0.684 | 0.446 | 0.124 | 0.40 | 0.274 |
| 等价 owner/物理证据监督 | 105k | 0.806 | 0.621 | 0.384 | 0.121 | 0.30 | 0.108 |
| 联合近等价证书 listwise | 105k | 0.827 | 0.663 | 0.411 | **0.157** | 0.30 | 0.116 |

`*` 原始模型的 repair rate 用当前统一的“已按预测角色/owner 形成 proposal，再统计硬投影
改写”口径重算；冻结的旧 summary 使用了更严格的历史 proposal 定义。

候选受限教师参考为 `0.875/0.752/0.543/0.197/0.50`。当前最佳 C1 的 worst/CVaR
差分别为 `0.097/0.073`，等价证据召回 `0.731`，投影改写 `0.176`，只满足硬冲突为零；
所以 Gate C1 明确失败，Gate C2 不启动。该结果仍优于价格协议的 `worst=0.266`，说明
消息传递有价值，但尚未学会联合角色—owner 组合。

两项归因尤其重要：仅替换为教师角色得到 worst `0.412`，仅替换为教师 owner 得到
`0.439`，同时替换才恢复到 `0.538`。因此不能继续独立提高某一个分类头；单目标等价
证据监督也不足以保证全局角色可组合性。

复制式教师现可导出完整潜在角色分区、owner、逐目标值和联合近等价结构。在 10 个训练
种子的 300 个求解帧上，95% worst/总值门槛下平均有 `1.99` 个联合证书；`56.7%`
存在不同角色分区，`54.3%` 存在不同 owner。联合 listwise 监督将开发 CVaR 从 `0.124`
提高到 `0.157`，并降低投影改写，但 mean worst 降为 `0.411`，仍未通过任何 C1 性能
差门。这证明联合等价性真实存在，也证明一次性摊销分类仍不足以代替局部组合改进过程。

### 4.6 Gate C1.5 可行局部邻域审计

在不训练交换网络的条件下，新增了 fail-closed Oracle 局部搜索。每个中间结构均显式
检查候选图、单 Tx/Rx 角色、单 owner、receiver 容量与目标端点上限；只有与复制式教师
完全相同的词典序边值目标严格改善时才接受原子动作。N1 是同 owner 边增删/替换，N2
是角色交换并重建支持边，N3 是 owner 迁移并重建支持边，N5 是对 2--3 个 UAV 和 1--2
个当前弱目标的局部精确重优化。N5 已覆盖本阶段需要验证的原子链式重排，因此未单独
实现只能表达部分链的 N4。

开发 holdout 上的 mean worst 如下；zero-step 均为各初始化自己的零步基线：

| 初始化 | zero-step | N1 | N1+N2 | N1+N2+N3 | +N5 |
|---|---:|---:|---:|---:|---:|
| 因子图 + 硬投影 | 0.446 | 0.446 | 0.479 | 0.483 | **0.544** |
| role-first | 0.147 | 0.147 | 0.384 | 0.515 | **0.529** |
| 上一帧结构 + 必要修复 | 0.072 | 0.072 | 0.424 | **0.542** | 0.537 |

复制式参考为 `0.543`。因子图 + N5 的
`steady/weak3/worst/CVaR/QoS=0.871/0.745/0.544/0.197/0.50`，恢复组合差距
`100.5%`，平均接受 `1.66` 次修改，所有教师目标轨迹严格单调，因此 Gate C1.5 通过。
但它每个求解帧平均枚举 `492.3` 个候选，仍是搜索上界而不是部署算法。

初始化审计揭示了更有用的时序结构：上一帧热启动只用 N1--N3 就达到
`0.875/0.754/0.542/0.197/0.50`，平均接受 `0.58` 次修改、评估 `41.3` 个候选；常规
帧不需要运行 N5。相反，从因子图或冷 role-first 启动时，只有能够原子调整角色、owner
和支持边的小块 N5 才能跨过组合能垒。不同初始化最终均可接近参考，说明局部邻域有
足够 headroom；但计算效率明显依赖时序热启动。

还需注意，逐帧严格单调的是教师的边值代理目标，不是重新计算后的 episode 检测指标；
例如 previous 的 `+N5` worst 略低于 N1--N3。这一差异要求后续 C1.6 同时报告代理 regret
和真实检测 regret，不能把精确验证器的代理改进直接写成闭环物理改进。

### 4.7 Gate C1.6 学习排序与事件触发冷启动

在生成训练标签前，正增益判定已进一步收紧：确定性边索引 tie-break 只用于多个正增益
动作之间的稳定排序，不再把“物理目标和边数完全相同、仅索引更优”当成值得切换的动作。
固定宽度的 29 维 move descriptor 只包含动作类型、角色/owner 改变量、冻结 Student 边值
的局部增删统计和当前稀缺度，不输入 post-move 全局最小值或精确词典序键；因此没有把
验证答案直接泄漏给排序器。共享 MLP 仅有 `6,210` 个参数。

上一帧热启动、N1--N3 的开发 holdout 闭环结果为：

| 排序器 | Top-M | worst | CVaR | 平均精确验证 |
|---|---:|---:|---:|---:|
| 全邻域 Oracle | all | 0.542 | 0.197 | 41.29 |
| random（3 次均值） | 5 | 0.481 | 0.136 | 5.85 |
| total-gain | 5 | 0.373 | 0.013 | 5.43 |
| scarcity | 3 | 0.514 | 0.140 | 3.68 |
| scarcity | 5 | **0.538** | 0.177 | 5.80 |
| learned | 1 | 0.388 | 0.059 | 1.40 |
| learned | **3** | **0.526** | **0.197** | **3.66** |
| learned | 5 | 0.526 | 0.197 | 5.86 |

learned Top-3 的 `steady/weak3/worst/CVaR/QoS` 为
`0.874/0.751/0.526/0.197/0.50`，worst 距 Oracle `0.016<0.02`，CVaR 完全恢复，精确
验证减少 `90.8%`。相同 Top-3 预算下它显著优于 scarcity；Top-5 不再改善闭环结果，
因此常规时序维护选择 learned Top-3，warm Gate C1.6 通过。

该结论不能外推到冷启动 N5。以因子图为初解时，scarcity Top-5、learned Top-3 和
learned Top-5 的 worst 分别只有 `0.485/0.475/0.498`，而 N5 Oracle 为 `0.544`；纯学习
N5 块排序明确失败并停止调参。离线 Top-3 命中率高但闭环失败，原因是早期次优动作会
改变后续候选分布，静态组内准确率不能代替路径级 regret。

事件触发混合控制则通过了离线性能屏：每个 episode 首次求解（10/300，`3.33%`）使用
完整 N5 Oracle 建立可行结构，后续 290 次求解使用 previous + learned Top-3，得到
`0.870/0.744/0.538/0.197/0.50`，worst 距复制式参考仅 `0.0053`。平均有效精确评估为
`21.05` 次，相对相同路径的全标签评估下降 `55.5%`。这不是“纯学习协调器已解决”：
它证明学习排序适合常规维护，冷启动仍需要确定性局部初始化，而且尚未经过真实 Token、
状态反演化和 Hold/switch-cost 的 C1.7。

### 4.8 Gate C1.7a 动态状态反演化

C1.7a 已把协调器接入真实闭环：Actor、UAV/目标运动、U2U Token 历史、本地观测、
Student 边值和 `L_u=4,L_q=4` 候选图逐帧重新生成；结构仍按部署基线 Hold-5 更新，
暂不加入事件触发和切换成本。部署执行路径只为学习排序后的 Top-M 候选计算精确目标，
不再为了离线诊断偷偷计算全邻域标签。

10 个开发种子的同序配对结果为：

| 控制器 | steady | weak3 | worst | CVaR | QoS | 精确验证/resolve | ms/resolve |
|---|---:|---:|---:|---:|---:|---:|---:|
| previous-only | 0.754 | 0.565 | 0.251 | 0.001 | 0.20 | 19.75 | 8.21 |
| dynamic Oracle N1--N3 | 0.882 | 0.771 | 0.531 | 0.138 | 0.50 | 50.97 | 11.10 |
| Hybrid learned Top-3 | 0.878 | 0.763 | 0.507 | 0.138 | 0.40 | 23.01 | 15.24 |
| **Hybrid learned Top-5** | **0.882** | **0.772** | **0.533** | **0.138** | **0.50** | **25.10** | **14.73** |
| replicated candidate reference | 0.915 | 0.832 | 0.632 | 0.236 | 0.50 | 85.81 | 96.66 |

Top-3 在 seed `483` 上漏掉关键正增益修改，使 episode worst 从 Oracle 的 `0.608`
降至 `0.349`；Top-5 在 10 个种子上恢复 Oracle 路径，并在 seed `243` 略优。故动态
版本将验证预算修正为 Top-5。相对 dynamic Oracle，其 mean worst 差为 `-0.0014`、
CVaR 差为 `0`，精确验证减少 `50.75%`，QoS 可行率相同，Gate C1.7a 通过。结构可行性
在每次解算后断言，单 UAV 1 W 最大误差为 `4.44e-16 W`。

该通过结论只针对“学习排序是否能逼近动态局部 Oracle”。相对 replicated 全局重解，
Top-5 的 steady/weak3/worst/CVaR 仍低 `0.033/0.060/0.099/0.098`，说明全局与局部
协调之间仍有显著 headroom。此外，当前 Python rank/enumerate 实现的墙钟时间是 Oracle
的 `1.33x`；验证次数下降尚不能写成实际加速。新鲜最终测试种子仍未使用。

### 4.9 Gate C1.7b 全局重构 headroom

为检验剩余 `0.099` worst 缺口是否可由“低频大邻域重构 + 日常 Top-5 维护”关闭，动态
协调器加入了 periodic/oracle N5 rebootstrap。Hold=5 条件下，`Periodic-N5(H=5)` 在
每个可更新 warm resolve 后检查完整 N5，并且只接受公共 Student 代理目标严格改善的动作；
因此它同时是现有 N5 的最高频 headroom 检查，低频 H=10/20 不可能提供更高的 N5 上限。

10 个相同开发种子的结果为：

| 控制器 | steady | weak3 | worst | CVaR | QoS | 精确验证/resolve | ms/resolve |
|---|---:|---:|---:|---:|---:|---:|---:|
| Local-only Top-5 | 0.882 | 0.772 | 0.533 | 0.138 | 0.50 | 25.10 | 14.73 |
| **Periodic-N5 H=5** | **0.915** | **0.831** | **0.573** | **0.178** | **0.50** | **170.00** | **85.86** |
| Replicated global reference | 0.915 | 0.832 | 0.632 | 0.236 | 0.50 | 85.81 | 96.66 |

H=5 改善了 steady/weak3 和平均尾部，但没有通过预注册的 `mean worst>=0.60` 门槛，
距 replicated global 仍差 `0.0588` worst 和 `0.0576` CVaR。N5 在 `93.55%` resolve
上被检查，但仅 `3.79%` 的检查接受动作。逐 episode 按 `1e-4` 容差统计为 3 个改善、
4 个持平、3 个恶化；paired mean-worst 增益 `0.0404` 的 bootstrap 95% CI 为
`[-0.0835,0.2097]`。seed `989` 改善 `+0.7065`，但 seed `483` 恶化 `-0.2957`，说明
“单帧公共代理正增益”不能保证“多帧物理 worst 正增益”。

因此停止 H=10/20 与 Hold 大扫描，也暂不训练 rebootstrap trigger：触发器无法修复错误的
重构价值标签。下一 Gate 先在相同状态、相同随机数和冻结后续动作下审计 N5 代理增益与
1/5 帧真实 worst 增益的一致性；若一致性仍不足，再进入“学习选择 UAV/目标块 + 块内精确
LNS”，而不是继续扩大 Top-M 或反复调整切换权重。该结论仍未消耗新鲜最终测试种子。

### 4.10 Gate D0/D1 N5 反事实初筛

反事实工具现可保存动态协调器实际使用的公共 value/mask，分别枚举现有 `proxy_weak` N5
和诊断性的 `all-target` 单/双目标块，并在相同 P0 前状态、相同动作和相同 RNG 下强制执行
任一候选。seed `483` 的机械 smoke 得到 `max_geometry_error=0`，确认 UAV/目标运动及随机流
严格一致。以下结论只来自首次代理正机会 frame `75`，属于事件级初筛而非总体 N5 上限。

| seed 483 / frame 75 | 现有 proxy-weak 池 | all-target 诊断池 |
|---|---:|---:|
| 原子候选数 | 209 | 2046 |
| 一帧物理 Oracle worst 增益 | +0.082 | +0.323 |
| 一帧物理 Oracle weak3 增益 | +0.180 | +0.412 |
| 一帧物理 Oracle steady 增益 | +0.090 | +0.206 |

all-target 最优动作不在现有池中，额外 worst headroom 为 `0.2406`，证明“只围绕公共代理
最弱两个目标生成块”至少在该灾难事件上会漏掉关键重构。与此同时，现有池 13 个代理正
候选中有 3 个使同帧物理 worst 恶化，错误接受率为 `23.1%`；但代理 Top-1 恰好也是现有池
的一帧物理最优，且现有池 10 个物理正候选均被代理正集合覆盖，故不能概括成“代理完全无
排序能力”。实际重构序列在该事件只接受一个动作，
已排除连续多步过冲。

五开发种子的 local-only 与 Periodic-N5 轨迹进一步完成闭环对齐。seed `483` 在 frame 75
前结构逐帧完全一致且物理 PD 最大误差为 `0`。首次重构后：

| 闭环窗口 | mean worst 增益 | 时域最小 worst 增益 | target-frame bottom-20% 增益 |
|---|---:|---:|---:|
| H=1 | +0.0821 | +0.0821 | +0.1749 |
| H=5 | +0.0555 | -0.0013 | +0.0438 |
| H=10 | -0.1049 | -0.0013 | -0.0036 |

目标随机状态始终完全一致。frame 76 首先出现 sensing weight 分叉，frame 77 开始出现运动
动作与 UAV 几何分叉；到 frame 80，Periodic 路径单帧 worst 比 Local 低 `0.4175`。因此
该事件同时暴露两个瓶颈：候选块漏失，以及短期结构收益被通信辅助感知/运动闭环响应反转。

当前仍不能直接训练 trigger 或宣布转向 learned block-LNS：完整 all-target 只审计了一个
事件，尚未完成冻结未来动作的 H=5 重放，也未建立顺序 LNS 上限。下一步只在 5 种子轨迹
中分层抽取少量“代理接受、代理漏失、deficit spike”事件重复 D0，并补冻结动作对照；新鲜
最终测试种子继续保持封存。

### 4.11 Gate D0.2--D0.3 冻结控制归因与 deficit 安全门

冻结控制重放已经完成。No-op 与 Forced-N5 从同一 pre-step 状态出发，未来使用完全相同的
运动、Token、速率、掩码、通信/感知功率和 Student 公共图。seed `483` 的前缀重放物理
`P_D` 误差为 `0`、观测最大误差为 `2.98e-8`，且 1 W 功率平衡误差不超过
`2.22e-16 W`。严格区分干预当帧与之后五帧后：

| seed 483 / frame 75 路径 | future-5 mean worst 增益 | future-5 终点增益 |
|---|---:|---:|
| 全部 No-op 控制冻结 | +0.0956 | 0.0000 |
| 仅恢复运动反馈 | +0.0896 | -0.0255 |
| 仅恢复通信/资源/Student 反馈 | +0.1462 | -0.0000 |
| 恢复完整记录闭环 | -0.0444 | -0.4175 |

因此 seed `483` 的结构动作本身是正增益，单独的运动或通信/资源路径也不能解释崩塌；
movement × radio/resource/Student 的非线性交互项为 `-0.1847`。这支持“联合过渡一致性”
而不是独立平滑某一个 Actor head。混合路径属于 cross-world 诊断，不能解释为可加的自然
间接效应。

随后对五开发种子中四个相互独立的首次 N5 接受事件重复 Gate。结果不是单一机制：

| 类别 | 事件 |
|---|---|
| 同帧代理误接受 | seed 291 / frame 95 |
| 正动作被闭环反转 | seed 483 / frame 75 |
| 稳定正增益 | seed 566 / frame 10；seed 103 / frame 15 |

seed `291` 在事件前两条轨迹逐值一致，实际 N5 却把 worst 从 `0.8401` 降到 `0.5795`。
完整同状态池显示 proxy/all-target 分别有 `172/1719` 个候选，但两池的物理最优都是 No-op；
唯一代理正候选就是物理有害候选。因此该事件是纯代理/No-op 接受错误，不是候选块漏失。

基于这一诊断增加了默认关闭的 deficit-only N5 审计门。10 个 `selection` 种子上，它只拦截
seed `291`，其余九个种子逐值不变：

| 控制器 | steady | weak3 | mean worst | CVaR | QoS | exact/resolve |
|---|---:|---:|---:|---:|---:|---:|
| Periodic-N5 H=5 | 0.9155 | 0.8309 | 0.5730 | 0.1784 | 0.50 | 170.00 |
| + deficit guard | 0.9153 | 0.8306 | 0.5777 | 0.1784 | 0.50 | 144.14 |

guard 使 mean worst 增加 `0.00476`，精确验证减少 `15.2%`，但 `0.5777<0.60`，CVaR 与
可行率均未改善。它可以保留为安全/计算保护，却不是性能主贡献。并且当前环境级
`coord_pd_ema` 不是免费分布式信号；正式实现必须用 owner 局部 QoS Token 或有限轮 max/min
共识替代。

协议审计同时记录一次测试集污染：首次 seed `291` 全池命令漏写显式 `selection` split，N5
过滤得到 0 个事件且该结果未参与方法判断，但普通 paired evaluator 仍运行了默认 `test`
split 的前 5 个种子 `795/747/105/860/2`。这 5 个种子必须永久隔离，不能再进入确认性或
最终性能声明；后续只能使用未查看的锁定剩余池或重新预注册替代测试池。正确的 seed `291`
审计随后在独立输出目录中用显式 `selection` split 完成。

### 4.12 Gate D0.4 嵌套弱目标预算与 No-op 安全证书

N5 的候选目标集合已从固定 Top-2 推广为嵌套预算 `B in {1,2,3,Q}`。这里 `B` 只决定可从
多少个公共代理最弱目标中选择单目标/双目标原子块，并不把一次原子重构扩展成不可控的多目标
全局修改。候选集合严格嵌套，`B=Q` 仅作为离线诊断上界。每个候选继续满足 delivered
candidate graph、角色/owner/容量约束以及单 UAV `communication + sensing = 1 W`；物理
重放使用同状态 common random numbers。

接受规则显式包含 No-op。若 baseline 的 steady/weak3 已达到阈值，候选不得跌破阈值；若
baseline 尚未达标，候选不得继续恶化该指标。只有同时满足该保护并提高物理 worst 的候选
才算 QoS-safe。三个开发危机事件结果为：

| B | 三事件候选总数 | 存在安全正动作的事件 | 平均 No-op-safe worst 增益 |
|---:|---:|---:|---:|
| 1 | 174 | 1/3 | +0.0274 |
| 2 | 616 | 1/3 | +0.0274 |
| 3 | 1204 | 1/3 | +0.0274 |
| Q=6 | 4615 | 1/3 | +0.1076 |

seed `103/566` 即使扩大到 `B=Q`，也只能用 weak3 下降换取 worst 上升，安全证书应选择
No-op。seed `483` 确有候选遗漏：同帧安全 worst 增益从 `+0.0821` 增至 `+0.3227`，且
steady/weak3 同时提高；但冻结 No-op 控制后的 future-5 mean 仅从 `+0.0956` 增至
`+0.0984`，边际持久收益只有 `+0.00283`，并在下一次普通结构重解时归零。`B=Q` 的三事件
候选量则是 `B=2` 的 `7.49x`。因此统一扩大候选池被否决；嵌套预算只能作为按需搜索机制。

下一方法不应是固定更大的 Top-k，而应是置信约束、事件触发的局部大邻域搜索。每个 target
owner 用实际收到且带时延/年龄的 Token 构造检测 logit 下界：

```text
P_lower(q,S) = sigmoid(P_logit_hat(q,S) - beta * uncertainty(q,S))
```

候选 `S'` 只有在所有目标的下界不低于各自受保护基准，并且最差目标下界的增益超过结构
切换、bit、时延和功率成本时才允许执行；No-op 永远属于可行集。`uncertainty` 必须由
校准残差、Token age、丢包和量化误差得到，不能访问真实全局几何。该机制在数学上属于
带置信安全集的分布式重构，而不是 Reward Shaping；在收集足够的 owner-local 校准事件并
完成事件级交叉验证前，不接入部署控制器。

### 4.13 Gate D0.5--D0.6 依赖闭包、尾缺口主序与时域证书

Gate D0.5 纠正了此前“一个 N5 原子动作至多影响两个目标”的表述。历史 N5 虽然只允许
1--2 个目标改变 owner，但角色变化后会重建全部支持边；三个危机事件的 `4615` 个候选中，
`4595` 个（`99.57%`）实际改变超过两个目标，平均影响 `5.46` 个目标，最大为 `Q=6`。
因此历史 N5 应称为**由小提案触发的多目标依赖闭包**，不能称为双目标局部动作。

实现中新增了两个只用于开发审计、默认不进入部署的严格局部对照：`target_block` N5 冻结
目标块外全部边，N6 固定角色分区、只交换 1--2 个目标的 owner/support。`B=Q` 时三事件
分别只剩 `46/72` 个候选，两者均无逐目标安全正动作，也无尾缺口主序安全正动作。这说明
当前物理增益确实来自跨目标角色耦合，而不是可由简单局部 owner 调整替代。

逐目标 identity-preserving 约束

```text
P_q(S') >= min(P_q(S), P_floor),  for every q
```

在当前等权目标模型中过于保守：历史全闭包 N5 仅保留 seed `483` 的 `+0.00270` worst，且
该候选在 `B=1` 已存在。为同时尊重 max--min、公平尾部和目标置换等变性，新增检测缺口的
worst-k 累积主序。令

```text
d_q(S) = max(P_floor - P_lower_q(S), 0)
D_k(S) = sum of the k largest entries of d(S)
```

候选必须对所有 `k=1,...,Q` 满足 `D_k(S') <= D_k(S)`。在等权离散目标下，`k=1`
保护最大缺口，每个 `D_k/k` 对应一个离散 upper-tail deficit CVaR，`k=Q` 保护总缺口。
该规则在三个事件上拒绝 seed `103/566`，同时恢复 seed `483` 的 `+0.0821`（`B<=3`）和
`+0.3227`（`B=Q`）同帧物理上界；它允许目标身份间的公平重分配，但不允许任何最坏-k
尾部恶化。若未来目标权重不等，必须先推广为加权尾风险，不能直接沿用等权主序。

Gate D0.6 证明同帧主序仍非闭环证书。seed `483` 的 `B=2` 动作在匹配记录闭环的 future-5
仅 `1/5` 帧保持尾缺口主序，最大 worst-k 累积缺口违规为 `0.5147`，终点 worst 增益为
`-0.4175`；`B=Q` 在冻结 No-op 路径上 `5/5` 帧保持主序，但没有与之匹配的闭环反馈轨迹，
不能外推。所以下一证书对象改为候选相对 No-op 的 H 步尾缺口差，并按事件定义最大标准化
低估残差：

```text
R_e = max over candidate c, horizon h and tail k
      (G_true[e,c,h,k] - G_hat[e,c,h,k]) / u[e,c,h,k]
```

split-conformal 的 `beta` 只从独立事件级 `R_e` 校准，进而使用
`G_upper = G_hat + beta*u <= 0` 作为 fail-closed 条件。这样同一事件中的候选、目标尾部和
时刻不会被错误当成独立样本，并能覆盖冻结候选生成管线内的后选择。30 个独立事件仅作为
pilot；在管线、H、损失定义和候选预算冻结前，不开始正式校准，也不改变部署控制器。

## 5. 门槛判定

当前 Medium 均值门槛为：

```text
steady >= 0.80
weak3  >= 0.70
mean worst >= 0.60
```

跨尺度候选还必须满足：

```text
QoS feasible >= 0.70
4/4 anchor exact preservation
```

据此，6/6 基数残差满足前三项和锚点保持，但没有满足可行率门槛。它是值得继续验证的
候选，而不是已经完成的部署升级。

## 6. 可以与不可以写入论文的结论

目前可以支持：

- 分布式、共享参数和置换等变的结构 Student 可在 4/4 达到 Medium 可部署门槛；
- 自适应精度 Token 在受限 U2U 信道下保持功率和原子传输约束；
- 基数门控残差在严格保持 4/4 锚点的同时，显著改善了本次 6/6 诊断集上的 mean worst；
- 架构保持比单纯奖励保护更适合处理“旧规模不能退化”的约束。

目前不可以宣称：

- 已实现完全分布式的端到端物理检测；
- 6/6 已达到稳定部署标准；
- 已证明基数残差优于集中式控制；
- 4/4 训练模型已经可靠零样本扩展到 8/8；
- 当前结果代表波形级或真实硬件 ISAC 性能。

## 7. 下一阶段实验协议

主线现在收敛为“局部候选图 → 一轮公共视图 → 可扩展的角色/owner 组合推理 → 硬可行
投影”。4/4 部署版继续冻结，基数残差保留为跨尺度基线。

1. 冻结 `L_u=4, L_q=4`、单次 refinement 的候选器和 A2 指标；不再围绕 exact-edge
   复刻或二次重排继续调参，因为性能等价边已经足够。
2. 停止目标价格、角色价格和简单缺口队列的参数扫描。现有负结果已证明这类可分离
   局部贪心无法恢复 Tx/Rx 角色互斥、receiver 容量和 owner 唯一性的组合耦合。
3. 将复制式确定性求解器只保留为“公共候选图上的教师/可达性能参考”，不能作为部署
   算法；它接近候选受限教师参考，但枚举与动态规划复杂度不具备大规模演进性。
4. 下一实现应是 ADMN 思路启发、但不直接照搬的共享参数消息传递/展开式协调器：网络
   学习候选超边提案、有限轮更新和停止条件，最终角色、owner、容量及单 UAV 1 W 约束
   仍由轻量确定性硬投影保证。监督损失以 owner、等价目标证据和参考目标差距为主，
   exact edge 只作为辅助诊断，避免学习任意 tie-break。若多个 owner 在证据和目标值上
   等价，则 owner 标签也采用集合监督；证据损失同时加入边数/bit 成本，防止用稠密全选
   取得平凡的高召回。
5. 硬投影只能执行单角色冲突修复、owner 唯一化、容量裁剪、确定性 tie-break 和 1 W
   单纯形投影，不得重新调用动态规划或 MILP。新增 projection 修改边比例与 projection
   目标增益，要求硬冲突为零、修改边比例原则上低于 `5%`，且目标性能不能主要由投影器
   产生。
6. **Gate C1（组合逼近门）**：先在离线公共候选图上要求 mean worst 距教师参考不超过
   `0.02`、CVaR 差不超过 `0.01`、所有硬约束零违反，并报告 A2 等价证据召回、owner
   准确率、目标差距、每轮收敛率和 projection repair rate。
7. **Gate C2（物理通信门）**：C1 通过后才接入 bit、通信功率、时延、丢包、量化和
   冷启动 fail-closed；只有真实 Token 条件下仍接近 C1，才能称有限轮分布式协调器。
8. 当前 6/6 候选受限教师参考本身只有 `0.551` mean worst 和 `0.40` QoS 可行率。
   C1/C2 只解决分布式执行逼近问题；之后仍需分别审计运动几何、6/6 边值校准及慢时标
   承诺，不能暗示协调器单独即可达到 Medium。
9. Gate C1.7a 已证明动态 Top-3 预算不足、Top-5 通过相对 dynamic Oracle 的性能与
   `50%` 精确验证缩减门。冻结 ranker checkpoint 和 Top-5，不再继续扫描 M 或网络宽度。
10. Gate C1.7b 的高频 N5 已失败：`worst=0.573<0.60`，且单帧代理接受的重构可使整段
    episode 恶化。停止 H=10/20、Hold 大扫描和 trigger 训练，不把低频/触发机制误当作
    N5 邻域能力提升。
11. Gate D0/D1 初筛已在 seed `483` 定位“候选块漏失 + 闭环收益反转”双重瓶颈。下一步
    不做全种子穷举，只对少量事件分层复核完整候选池，并增加冻结未来动作 H=5 对照，报告
    proxy/all-target Oracle、错误接受/漏失、H1/H5/H10 mean/min/bottom-tail 与累计 regret。
    只有重复证据支持后才在“修块选择、修短时域验证器、加保护期”之间选一条主线。
12. Gate D0.2--D0.3 进一步确认失败机制至少有两类：seed `291` 是代理对 No-op 的同帧
    误接受，seed `483` 是运动与通信/资源/Student 的负交互反转。deficit guard 仅把10种子
    mean worst 从 `0.5730` 提到 `0.5777`，仍不达 `0.60`；下一阶段必须同时保留 No-op
    安全验证和联合过渡一致性，而不能把单一 trigger 或单头平滑当成完整修复。
13. Gate D0.4 否决统一扩大弱目标预算：`B=Q` 的候选量为 `B=2` 的 `7.49x`，三个危机
    事件中仍只有一个存在 QoS-safe 正动作，且其相对 `B=2` 的冻结 future-5 mean 额外增益
    仅 `0.00283`。下一 Gate 只收集 owner-local 检测下界的校准残差，采用事件级而非候选级
    划分验证置信安全证书；证书通过前不训练 trigger、不改变部署协调器。
14. Gate D0.5--D0.6 修正下一步：历史 N5 的 `99.57%` 候选实际影响超过两个目标，严格
    双目标 N5 与固定角色 N6 均无尾缺口安全正动作，因此不能把主线包装成低成本双目标
    LNS。保留等权目标下的 worst-k 累积缺口主序作为公平安全序，但同帧规则不足；正式主线
    改为稀疏触发的角色依赖闭包与候选相关 H 步证书。只有事件级联合 conformal 上界通过后
    才能训练 trigger 或接入部署。

## 8. 复现实物与验证状态

- 4/4 正式结果：
  [`paired_eval.csv`](../results/architecture_v2_structure_student_u2u_resolve_bw50k_adaptive_b4b8_failclosed_gate100/paired_eval.csv)
- 6/6 基数残差结果：
  [`paired_eval.csv`](../results/architecture_v2_scale_k6q6_structure_student_cardinality_residual46_gate10/paired_eval.csv)
- 4/4 锚点安全门：
  [`paired_eval.csv`](../results/architecture_v2_structure_student_cardinality_residual46_bw50k_gate10/paired_eval.csv)
- 基数残差模型：
  [`frozen_structure_student_cardinality_residual46.pt`](../results/architecture_v2_structure_student_cardinality_residual46_gate/frozen_structure_student_cardinality_residual46.pt)
- 6/6 配置：
  [`exp_800_k6q6_architecture_v2_maxmin_local_fusion_fullgraph_hold5_scale.yaml`](../config/exp_800_k6q6_architecture_v2_maxmin_local_fusion_fullgraph_hold5_scale.yaml)
- 8/8 配置：
  [`exp_800_k8q8_architecture_v2_maxmin_local_fusion_fullgraph_hold5_scale.yaml`](../config/exp_800_k8q8_architecture_v2_maxmin_local_fusion_fullgraph_hold5_scale.yaml)
- 局部候选与等价类审计：
  [`audit_local_candidate_upper_bound.py`](../tools/audit_local_candidate_upper_bound.py)
- 有限轮协调器与复制式教师参考：
  [`finite_round_hyperedge.py`](../uav_isac/coordination/finite_round_hyperedge.py)
- 协议重放入口：
  [`run_distributed_hyperedge_negotiation.py`](../tools/run_distributed_hyperedge_negotiation.py)
- Gate C1 因子图协调器：
  [`factor_graph_coordinator.py`](../uav_isac/coordination/factor_graph_coordinator.py)
- Gate C1 训练与审计入口：
  [`train_factor_graph_coordinator.py`](../tools/train_factor_graph_coordinator.py)
- 当前最佳 C1 筛选（未通过）：
  [`summary.json`](../results/architecture_v2_scale_k6q6_factor_graph_gate_c1_screen/summary.json)
- 联合结构证书：
  [`summary.json`](../results/architecture_v2_scale_k6q6_joint_certificate_train10/summary.json)
- 联合证书 listwise 筛选（未通过）：
  [`summary.json`](../results/architecture_v2_scale_k6q6_factor_graph_gate_c1_certificate_screen/summary.json)
- Gate C1.5 可行局部搜索实现：
  [`local_exchange_oracle.py`](../uav_isac/coordination/local_exchange_oracle.py)
- Gate C1.5 审计入口：
  [`audit_oracle_local_search.py`](../tools/audit_oracle_local_search.py)
- 三初始化完整 Gate C1.5 报告（通过）：
  [`summary.json`](../results/architecture_v2_scale_k6q6_oracle_local_search_gate_c1_5_all_initials/summary.json)
- Gate C1.6 排序特征与模型：
  [`local_move_ranker.py`](../uav_isac/coordination/local_move_ranker.py)、
  [`learned_move_ranker.py`](../uav_isac/coordination/learned_move_ranker.py)
- warm learned Top-3 开发报告：
  [`summary.json`](../results/architecture_v2_scale_k6q6_ranked_local_search_gate_c1_6_learned_dev10/summary.json)
- pure learned N5 失败报告：
  [`summary.json`](../results/architecture_v2_scale_k6q6_ranked_local_search_gate_c1_6_learned_cold_n5/summary.json)
- 事件触发冷启动混合报告：
  [`summary.json`](../results/architecture_v2_scale_k6q6_hybrid_local_search_gate_c1_6/summary.json)
- Gate C1.7a 动态四路对照与准入结论：
  [`summary.json`](../results/architecture_v2_scale_k6q6_dynamic_local_search_gate_c1_7a/summary.json)
- Gate C1.7a 动态协调器与汇总入口：
  [`dynamic_local_search.py`](../uav_isac/coordination/dynamic_local_search.py)、
  [`summarize_dynamic_local_search_gate.py`](../tools/summarize_dynamic_local_search_gate.py)
- Gate C1.7b 高频 N5 重构失败报告：
  [`summary.json`](../results/architecture_v2_scale_k6q6_rebootstrap_headroom_gate_c1_7b/summary.json)
- Gate C1.7b 汇总入口：
  [`summarize_rebootstrap_headroom_gate.py`](../tools/summarize_rebootstrap_headroom_gate.py)
- Gate D0/D1 N5 反事实初筛报告：
  [`summary.json`](../results/architecture_v2_scale_k6q6_n5_counterfactual_gate_d0_d1_preliminary/summary.json)
- Gate D0/D1 重放与汇总入口：
  [`n5_counterfactual_audit.py`](../uav_isac/evaluation/n5_counterfactual_audit.py)、
  [`summarize_n5_counterfactual_gate.py`](../tools/summarize_n5_counterfactual_gate.py)
- Gate D0.2--D0.3 冻结控制/deficit guard 报告：
  [`summary.json`](../results/architecture_v2_scale_k6q6_n5_frozen_controller_gate_d0_2_d0_3/summary.json)
- Gate D0.2 路径重放入口：
  [`audit_n5_frozen_controller.py`](../tools/audit_n5_frozen_controller.py)
- Gate D0.4 嵌套预算/No-op 安全报告：
  [`summary.json`](../results/architecture_v2_scale_k6q6_n5_weak_budget_gate_d0_4/summary.json)
- Gate D0.4 事件审计与汇总入口：
  [`audit_n5_weak_target_budget.py`](../tools/audit_n5_weak_target_budget.py)、
  [`summarize_n5_weak_target_budget_gate.py`](../tools/summarize_n5_weak_target_budget_gate.py)
- Gate D0.5 依赖闭包/证书口径报告：
  [`summary.json`](../results/architecture_v2_scale_k6q6_n5_certificate_alignment_gate_d0_5/summary.json)
- Gate D0.6 时域尾缺口报告：
  [`transition_tail_summary.json`](../results/architecture_v2_scale_k6q6_n5_certificate_alignment_gate_d0_5/transition_tail_summary.json)
- Gate D0.5--D0.6 汇总与联合证书实现：
  [`summarize_n5_certificate_alignment_gate.py`](../tools/summarize_n5_certificate_alignment_gate.py)、
  [`summarize_n5_transition_tail_gate.py`](../tools/summarize_n5_transition_tail_gate.py)、
  [`transition_certificate.py`](../uav_isac/evaluation/transition_certificate.py)

历史验证状态为 `385` 项主回归测试与 `14` 项 belief 校准测试通过；本次 Gate C1.5--C1.6
相关的局部搜索、排序器、协调器、候选审计和 Student 路径共 `49` 项定向回归通过；Gate
D0.4 修改后的 N5/Student 定向回归为 `32 passed`；Gate D0.5--D0.6 的局部性、尾缺口、
事件级 conformal 及相关 Student/协调器扩展回归为 `69 passed`。详细命令、
历史对照和所有失败路线见 [ARCHITECTURE_V2_RESULTS.md](ARCHITECTURE_V2_RESULTS.md)。

### 4.14 Gate D0.7：物理原子提交与控制功率预留

本 Gate 将重构判据从“异量纲加权和”改为三个可行域的交集：

```text
A_safe = A_role/owner/capacity/power
       ∩ A_link/deadline/version
       ∩ A_H-step-tail
```

bit、秒、焦耳、瓦和检测概率不再直接相加。代价只允许在已经满足全部硬约束的
候选之间排序；owner 不可达、状态版本不一致、超时或功率越界均直接回退 No-op。

实现采用 prepare/vote/decision 三轮原子提交。依赖闭包包括角色发生变化的 UAV、
所有切换边的两个端点，以及受影响目标的新旧 owner。消息显式携带角色、owner、
切换边和全目标检测下界记录。16 bit 检测下界向下量化、不确定度向上量化，单项
保守网格误差不超过 `1/(2^16-1)=1.526e-5`。每条链路统一调用环境的 Shannon
链路预算；投票轮多个发送者正交平分带宽。硬条件为：每包 SNR 和时限合格、三轮
总时延不超过一个 `0.1 s` 控制帧、参与者版本一致，以及逐 UAV 满足

```text
P_comm,k + sum_q P_sense,kq <= 1 W.
```

seed `483` / frame `75` 的三个代表性正增益候选（index `0/4/8`）均形成 6-UAV
依赖闭包。原执行分配中 UAV 2 的通信功率为 0，无法返回 vote，故全部正确拒绝。
若对闭包参与者预留 `0.25 W` 控制功率，则 index `0/4` 的完整三轮消息为
`1419 bit`、`1.948 ms`、`0.914 mJ`；index `8` 为 `1454 bit`、`1.985 ms`、
`0.923 mJ`，均满足当前 `5 ms` 单包时限和 `0.1 s` 控制帧。

`0.25 W` 只是固定可行上界，不是最终资源设计。新增的单调二分在每次迭代都重新投影
剩余感知功率，并求满足全部链路/时限条件的最小共同通信功率下限。本事件的名义下限
为 `2.119e-6 W`：index `0/4` 的三轮总时延/能量为 `6.170 ms/0.824 mJ`，index
`8` 为 `6.210 ms/0.834 mJ`；瓶颈 vote 包时延为 `4.9995 ms`。该值紧贴 5 ms
边界，只是特定几何和名义信道下的理论下界，不能直接作为部署值；正式裕量必须由
独立信道事件残差校准，不能从单事件任意指定。

预留功率没有被当作免费资源。审计重新执行通信/感知 1 W 硬投影并重跑同状态物理
检测；三个候选的 worst 增益仍分别为 `+0.0821/+0.3227/+0.0821`，尾缺口主序
仍成立，最大功率平衡误差为 `2.22e-16 W`。原因是该事件中 UAV 2 被削减的感知
功率未被候选支持结构使用；这只是事件级机制证据，不能推广为一般零代价结论。

当前研究定位进一步明确：主线属于**资源分配与分布式结构控制层**，联合优化目标
支持、Tx/Rx 角色、融合 owner、通信 bit/rate 以及通信-感知功率分配；波形层目前
仍是固定波形和解析信道抽象，尚未优化发射协方差、子载波、波束或模糊函数。因此
当前不能声称波形级联合设计。创新候选收敛为“事件触发控制功率预留 + 原子依赖
闭包提交 + H 步尾缺口证书”，在独立事件级 conformal 校准完成前继续默认关闭。

新增实现与证据：

- [`dependency_commit.py`](../uav_isac/coordination/dependency_commit.py)
- [`audit_n5_dependency_commit.py`](../tools/audit_n5_dependency_commit.py)
- [`seed483_frame75_commit.json`](../results/architecture_v2_scale_k6q6_n5_physical_commit_gate_d0_7/seed483_frame75_commit.json)

扩展的局部性、时域证书、物理提交与 Student/协调器定向回归为 `78 passed`；本 Gate
未使用新的最终测试种子。

### 4.15 Gate D0.8：事件平衡残差反馈与漂移锁定

反馈已纳入循环，但采用快慢双时间尺度，禁止当前事件的结果修改授权当前事件的证书。
一个部署 epoch 内残差模型与 conformal 乘子完全冻结；已执行事件的 owner 反馈只能进入
下一 epoch。对 Token 年龄、实际丢包比例、量化步长、切换规模、时域位置和尾部位置等
owner 可观测特征，采用

```text
G_tilde[e,c,h,k] = G_hat[e,c,h,k] + f_theta(x[e,c,h,k]).
```

残差回归按事件平衡：每个事件总权重相同，事件内复制上千候选不会增加该事件权重。
模型训练事件与校准事件 ID 强制互斥；校准仍以每个事件的
`max_(c,h,k)(G_true-G_tilde)/u` 作为唯一分数。5% 误覆盖下至少需要 19 个独立校准
事件才有有限 conformal 乘子，因此 30 个事件只能作为训练/校准 pilot，不能授权部署。

另增加证书违例 e-process 漂移监测。在稳定条件
`P(violation_t | past) <= alpha` 下，多个固定备择似然比的混合为非负超鞅；当 e-value
越过 `1/delta` 时永久锁定 No-op。该 anytime 界依赖上述条件，不能由当前单事件推断。

反馈链路没有免费化。K=Q=6、H=5 时，反馈包共享字段为 `153 bit`，每个
`(target,horizon)` 条目为 `23 bit`。seed `483` / frame `75` 中 owner 1/2 分别
携带 `24/12` 个条目，总无线发送量为 `1134 bit`。仅针对 commit 求得的
`2.119e-6 W` 下限无法发送反馈：最小 SNR 为 `-1.165 dB`、时延为 `10.671 ms`。
固定 `0.25 W` 可在 `1.551 ms/0.468 mJ` 完成反馈。

进一步联合优化 proposer、三轮 commit 与延迟反馈后，名义共同功率下限为
`6.795e-6 W`，协调者为 UAV 3。index `0/4` 的 commit 时延为 `3.982 ms`，index
`8` 为 `4.022 ms`；反馈时延均为 `4.99988 ms`，反馈能量 `0.338 mJ`。使用该联合
功率重新投影并重跑检测后，三个候选的 worst 增益及尾缺口主序保持不变。由于反馈
仍紧贴 5 ms 时限，该值只是名义理论下界，部署裕量必须由独立信道残差校准。

当前创新主线由此变为：**学习器估计局部边际价值 → 确定性物理层原子提交 → owner
返回延迟有界残差 → 下一独立 epoch 更新预测器 → e-process 监测漂移并优先 No-op**。
它属于双向、证书化、事件触发的分布式资源分配与结构控制，而不是普通的在线误差
反传，也不是波形优化。当前继续默认关闭，未使用新的最终测试种子。

新增实现与证据：

- [`certified_feedback.py`](../uav_isac/evaluation/certified_feedback.py)
- [`test_certified_feedback.py`](../tests/test_certified_feedback.py)
- [`seed483_frame75_feedback.json`](../results/architecture_v2_scale_k6q6_certified_feedback_gate_d0_8/seed483_frame75_feedback.json)

### 4.16 Gate D0.9：事件级联合校准的鲁棒信道闭环

名义 `6.795e-6 W` 联合功率下限紧贴反馈链路 `5 ms` 时限，不能把人为
指定的 `3 dB` 当作部署裕量。本 Gate 对每个独立事件内的全部链路、协议
轮次与反馈包先取一个联合最坏信道分数：

```text
R_channel,e = max_i {0,
  (SNR_hat-SNR_obs)/r_snr,
  (delay_obs-delay_hat)/r_delay }.
```

`r_snr` 与 `r_delay` 是校准前固定的物理分辨尺度，不是把 dB 和秒相加的
代价权重。同一事件内复制再多链路也只贡献一个校准样本。保守 SNR 在
Shannon 公式之前施加，随后重新计算速率、串行化时延、RF 能量、三轮原子
提交、owner 反馈和 `通信功率+感知功率=1 W` 投影。

同时修正了风险预算：两个各为 5% 的独立证书只能直接给出 10% 的 union
bound；若各压到 2.5%，有限 split-conformal 分位数至少需要 39 个事件，
计划中的 30 个事件不够。当前首选做法是对同一事件使用

```text
R_joint,e = max(R_transition,e, R_channel,e),
```

并冻结一个共同乘子。该方法不假设检测误差与信道误差独立，在一个 5%
风险预算内同时约束二者；训练事件与联合校准事件 ID 仍强制互斥。

seed 483/frame 75 仅用于预校准敏感性重放。诊断乘子从 0 增至 6（分辨尺度
为 `1 dB/0.1 ms`）时，共同 RF 下限单调从 `6.795 µW` 增至
`34.424 µW`（`5.07×`）。最大点的反馈鲁棒 SNR 为 `4.943 dB`、时延仍低于
`5 ms`，感知功率最大减少 `8.742 µW`。重新运行物理检测后，三个候选的
即时 worst 增益仍为 `+0.0821/+0.3227/+0.0821`，tail-safe 判定不变，功率
平衡误差不超过 `2.22e-16 W`。

这些点只是“误差乘子—所需功率”曲线，不是部署裕量。至少 30 个独立开发
事件完成联合校准，并由后续互斥事件验证前，控制器继续默认关闭；本 Gate
没有使用新的最终测试种子。

当前 U2U 传输的预测与交付仍共用确定性 Friis 几何模型，本身不能产生有
意义的非零“观测减预测”信道残差。下一 Gate 必须先增加独立、可复现的观测
信道/影子 CSI，并审计 Rician 归一化，再收集 30 个事件；否则零残差校准会
形成循环论证。这属于资源分配证书的物理验证基础设施，不代表研究主线已经
转为波形优化。

新增实现与证据：

- [`channel_margin.py`](../uav_isac/evaluation/channel_margin.py)
- [`certified_feedback.py`](../uav_isac/evaluation/certified_feedback.py)
- [`seed483_frame75_margin_curve.json`](../results/architecture_v2_scale_k6q6_robust_channel_gate_d0_9/seed483_frame75_margin_curve.json)
- [`test_channel_margin.py`](../tests/test_channel_margin.py)

隔离回归为非 belief 测试 `454 passed`、belief 校准测试 `14 passed`。单进程
合并运行仍会在既有 Windows MKL `numpy.linalg.eigvalsh` 路径原生中止，故
采用进程隔离；两组均无断言失败。

### 4.17 Gate D0.10：证书深度审计与重放 provenance 纠错

本 Gate 不以 D0.9 结果为前提，重新审计统计、通信协议和信道模型。发现并
修复四类实质问题：

1. `+inf` 事件分数过去会被 conformal 分位数函数删除；现在保留 `+inf`，
   NaN/负无穷直接报错，无法解析的阈值强制 No-op。
2. 声明为必需但缺失的 future-H 结果或链路包过去可能被静默跳过；现在缺失
   或未送达映射为 `+inf`。只有预先注册的结构 padding mask 可以排除条目。
3. 原总时延残差重复包含了低 SNR 已经造成的 Shannon 串行化增长；现在先用
   实测 SNR 扣除 `bits/R(SNR)`，只校准额外排队、调度与处理时延。
4. prepare/vote/decision 虽然计入 epoch/digest bit，但过去只比较状态版本；
   现在 commit 与 owner feedback 同时要求状态版本、冻结证书 epoch 和 64-bit
   候选摘要一致。proposal-pipeline digest 变化同样直接 No-op。

同时发现 Rician helper 计算了 `sqrt(1/(K+1))` 却未乘到 NLoS 项，导致

```text
E|h|^2 = path_gain * (K/(K+1) + 1)
```

而不是正确的 `E|h|^2=path_gain`。在 `K=6 dB` 时平均信道功率约高估
`79.9%`。归一化已经修正，并用固定随机种子在 `K=-10/0/6/20 dB` 上验证
二阶矩。

逐步测试推翻了最初的失效判断。`0.59446` 差异来自使用了与 trace 清单不一致的
ranker/factor-graph checkpoint，造成控制前缀分叉；它不能解释为 Rician 修正导致的
物理误差。当前配置的 `ground_communication_enabled=false`，检测 Deflection 不走
Rician reporting-link 分支。使用 trace 记录的精确流水线后，seed 483/frame 75 的
observation 前缀误差为 `2.98e-8`，物理 `P_D` 重放误差为 `0`。因此 D0.9 的该事件
即时增益和确定性 U2U bit/Shannon 曲线没有因本次 Rician helper 修正而失效；启用
ground/reporting stochastic link 的其他实验仍须单独重放。

审计工具新增强制 provenance 绑定，在计算任何重放残差前核对 trace、config、
Student、ranker、factor graph、搜索模式、Top-k、覆盖率和轮数。故意传入错误模型
时工具在仿真前终止且不写结果；正确流水线则完整通过。这样把“物理模型不一致”和
“控制器版本不一致”分开，避免用不可比较的轨迹作物理结论。

创新性也重新定级：联合通感功率分配、MARL、conformal 安全控制和多智能体
LNS 都不能单独宣称新颖。当前可辩护的潜在创新是它们的证书化组合：

```text
owner-local 边际价值
 -> 依赖闭包 LNS
 -> 物理计费且 epoch/digest 绑定的原子提交
 -> transition/transport 事件级联合最大分数
 -> 延迟 owner 残差与冻结管线
 -> e-process 漂移锁定 No-op。
```

这仍属于资源分配与分布式结构控制层，不是波形优化。完成系统相关工作检索、
修正信道实验和逐组件消融前，只表述为“潜在集成创新”，不作首创性断言。

新增实现与证据：

- [`transition_certificate.py`](../uav_isac/evaluation/transition_certificate.py)
- [`channel_margin.py`](../uav_isac/evaluation/channel_margin.py)
- [`dependency_commit.py`](../uav_isac/coordination/dependency_commit.py)
- [`channel.py`](../uav_isac/physical/channel.py)
- [`summary.json`](../results/architecture_v2_scale_k6q6_deep_certificate_audit_gate_d0_10/summary.json)
- [`seed483_frame75_revalidated.json`](../results/architecture_v2_scale_k6q6_deep_certificate_audit_gate_d0_10/seed483_frame75_revalidated.json)
- [`test_channel_physics.py`](../tests/test_channel_physics.py)
- [`test_audit_replay_provenance.py`](../tests/test_audit_replay_provenance.py)

隔离回归为 `466 passed + 14 passed`；没有使用新的最终测试种子。

局部性、时域证书、原子提交、反馈与 Student/协调器联合定向回归为 `84 passed`。

### 4.18 Gate D0.11：证书路由的两时间尺度功率/结构/几何架构

本 Gate 先否决了“再叠加一个结构专家”的路线。同一状态下把 cardinality 与
multiscale 两个 Student 按目标仲裁，selection 前 10 个 episode 的 mean worst 仅从
`0.608294` 增至 `0.608388`（`+9.49e-5`），而 payload-only 边值从每发送者
`3072 bit` 增至 `6144 bit`。因此不以双模型和双倍 Token 负担掩盖真正瓶颈。

物理分解显示当前主瓶颈是固定结构内的连续感知功率，而不是统一扩大 Top-k。
对 selection 前 20 个 episode 的最终重解状态，原部署、只换配对、固定结构功率、
联合单工结构/功率的 mean worst 分别为
`0.677636/0.677636/0.918167/0.932318`。只换配对的增益严格为 0；联合层超过
固定结构功率层的均值仅 `0.014151`。因此架构改为三层而不是继续扩大神经网络：

```text
快层：固定 role/owner/edge 的 max-min 感知功率修复
中层：仅当固定结构上界低于 QoS 时才搜索 joint structure + power
慢层：仅当宽松同几何上界仍低于 QoS 时才触发航迹/几何修复
```

固定唯一 owner 后，令 `a_iq` 为 UAV `i` 对目标 `q` 的每瓦有效 Deflection，
`b_i=1-P_comm,i`。快层是线性规划

```text
max_{p,t} t
s.t. sum_i a_iq p_iq >= t,                 for every q,
     sum_q p_iq = b_i, p_iq >= 0,          for every i.
```

其单纯形对偶为

```text
min_{lambda in simplex(Q)} sum_i b_i max_q lambda_q a_iq.
```

强对偶来自该 LP 的可行有界性。owner 只广播目标价格 `lambda_q`；每个 UAV 用
本地 `a_iq` 选择 `argmax_q lambda_q a_iq`。有限轮实现不再使用会漏目标的纯
次梯度极点平均，而采用 Dantzig--Wolfe 列生成：初始化 `Q` 个全目标覆盖列，
每轮增加一个分布式最佳响应列，并在 owner 端求 `Q` 维受限主问题。任何返回的
功率都是完整可行列的凸组合，因此每一轮均满足非负与每 UAV 精确 RF 等式；受限
原始值单调不降，量化公共价格处的精确本地响应给出有效对偶上界。

selection 前 10 个 episode 的全部 `310` 个结构重解事件上，6-bit 价格时的结果为：

| 方法 | mean worst | QoS 可行率 | 相对精确 Deflection | payload-only 上界 |
|---|---:|---:|---:|---:|
| 原部署 | 0.430307 | 0.3226 | - | - |
| 精确固定结构 LP | 0.704167 | 0.6323 | 1.0000 | 离线上界 |
| 1 轮列生成 | 0.628041 | 0.5516 | 0.8014 | 228 bit/重解 |
| 2 轮列生成 | 0.675462 | 0.6032 | 0.9199 | 360 bit/重解 |
| 4 轮列生成 | 0.698827 | 0.6258 | 0.9827 | 624 bit/重解 |
| 8 轮列生成 | 0.702509 | 0.6290 | 0.9944 | 1152 bit/重解 |

4-bit/6-bit/8-bit 的 4 轮 mean worst 分别为
`0.694920/0.698827/0.698825`，故暂用 6 bit；这只是同状态物理潜力审计，尚未计入
header、竞争、重传和实测时延。功率守恒误差不超过 `4.44e-16 W`。

算法冻结后，仅在尚未用于本轮设计的 selection 后 10 个 episode 上查看结果：
原部署、精确固定结构 LP、4 轮列生成 mean worst 分别为
`0.298254/0.569872/0.563907`，对应 4 轮 QoS 可行率 `0.4581`。这确认功率层收益
可迁移，但也证明它不能单独解决所有事件。

为纳入反馈，使用上一结构重解事件的 per-watt 增益作为最简单的因果估计。
selection 后 10 个 episode 上，4 轮 causal lag-1 的 mean worst 为 `0.499996`；
在全部 20 个 episode 上，lag-1 有 `8.83%` 事件比同事件 No-op 更差。用前 10 个
episode 的事件分数做诊断性 one-sided conformal 余量后，后 10 个 episode 的门控
策略 mean worst 为 `0.521787`，但仍有 `3.67%` 的约束违例。由于 300 个重解点只
聚类在 10 个校准 episode 内，且尚未计入切换/bit/时延代价，该结果明确标记为
`certificate_ready=false`，不能接入部署控制器。

中慢层采用嵌套物理上界路由，避免把启发式可行解误称为上界。宽松同几何上界允许
每个目标独占每架 UAV 的全部感知预算、每个发射机为每个目标自由选择最佳接收机，
并放松共同 owner、角色、容量和跨目标功率耦合：

```text
D_q^relax = sum_i b_i max_j a_ijq.
```

任何可行同几何方案均逐目标不超过该值。最终重解的 20 个 selection episode 被
证书路由为 `12 No-op / 6 fixed-power / 1 joint-structure-power / 1 slow-geometry`。
seed 606 的宽松上界仅 `0.3986`，所以慢几何触发有严格依据；seed 138 的固定结构
上界为 `0.5228`、宽松上界为 `0.9563`，且联合求解已找到 `0.7412` 的可行解，
所以应先换结构。joint alternating 的失败本身不再被用于证明几何不可行。

创新边界保持克制：max-min ISAC 功率分配、列生成/目标价格、两时间尺度资源分配、
conformal 安全门和 MARL 都是已有方法。当前可辩护的是潜在集成创新，即
“owner-local 检测反馈和边际增益 -> 始终 RF 可行的有限轮列生成 -> 嵌套物理上界
只路由到最小必要控制层 -> epoch 隔离的残差下界和 No-op 安全层”。在完成系统文献
检索、独立 episode 校准、真实 Token 时延/丢包重放和闭环航迹消融前，不作首创断言。
当前研究仍是资源分配与分布式控制层，不是波形设计；波形、波束、子载波和模糊函数
仍保持固定。

新增实现与证据：

- [`maxmin_power.py`](../uav_isac/coordination/maxmin_power.py)
- [`bottleneck_router.py`](../uav_isac/coordination/bottleneck_router.py)
- [`expert_arbitration.py`](../uav_isac/evaluation/expert_arbitration.py)
- [`audit_distributed_power_repair_trace.py`](../tools/audit_distributed_power_repair_trace.py)
- [`audit_structure_trace_physical_bottleneck.py`](../tools/audit_structure_trace_physical_bottleneck.py)
- [`summary.json`](../results/architecture_v2_scale_k6q6_targetwise_expert_arbitration_gate_d0_11/summary.json)
- [`selection20_all_resolves_power_feedback_gate.json`](../results/architecture_v2_scale_k6q6_targetwise_expert_arbitration_gate_d0_11/selection20_all_resolves_power_feedback_gate.json)
- [`selection20_final_resolve_physical_bottleneck.json`](../results/architecture_v2_scale_k6q6_targetwise_expert_arbitration_gate_d0_11/selection20_final_resolve_physical_bottleneck.json)

### 4.19 Gate D0.12--D0.13：可观测物理反馈、通信计费与事件触发原子结构修复

本轮首先修正了反事实物理审计的一个根本性可辨识性错误。过去用
`d_eff / 当前感知功率` 恢复每瓦增益；当某 UAV--目标的当前功率为零时，
这会把“未激励、不可由该比值识别”错误写成物理零。当前非 Swerling 模型中，
trace 已保存与功率无关的 `alpha`、`g_dd` 和 `chi_rep`，因此现在直接使用

```text
a_ijq = 1[g_dd,ijq >= g_min] chi_rep,ijq alpha_ijq^2
        * T_sym M N G_tx G_rx n_CPI / sigma_z^2 .
```

该式与 `DeflectionComputer` 完全同源，并在 50 个重解事件上把重放误差压到
`1.36e-12`。Swerling 开启时该重建会失败关闭，因为单次 RCS 随机实现不能由
上述三个量识别。延迟到下一事件的增益继续按双基地距离律传输：

```text
a_ijq(t) R_iq(t)^2 R_jq(t)^2
  = a_ijq(t-1) R_iq(t-1)^2 R_jq(t-1)^2 .
```

当前 `g_dd >= g_min` 同时作为硬支持门，而不是被平均进一个连续代理。这个修正
证明当前主线仍是**资源分配与分布式控制层**：波形、子载波、波束和模糊函数
保持固定，OTFS/双基地模型只提供物理可行域与每瓦边增益，不能称为波形联合设计。

固定 owner 后的快层仍求解 max--min LP。有限轮 Dantzig--Wolfe 现在不仅量化
6-bit 目标价格，也真正以 IEEE binary16 量化 owner Deflection 反馈；未计数的
双精度侧信道已被移除。通信协议显式包含 64-bit header、16-bit epoch、64-bit
digest、节点/轮次/目标索引、价格与反馈字段。owner 反馈正交分带、协调者价格
广播，完整时延为

```text
T_protocol = T_feedback,0 + R (T_price + T_feedback),
```

并在 Shannon 速率、3 dB SNR 工程裕量、0.5 ms 时延工程裕量和 0.1 s 控制周期下
重新投影 `P_comm,k + sum_q P_sense,kq = 1 W`。selection20 的 600 个因果事件全部
通信可行：平均 `1537 bit`，平均最大单包时延 `1.071 ms`，最坏完整协议时延
`28.805 ms`，平均能量 `0.455 mJ`，最大额外通信功率 `3.20e-4 W`，功率平衡误差
不超过 `4.44e-16 W`。binary16 相对双精度只造成约 `7.6e-5` 的 mean-worst 损失。

20 个非保留训练 episode 与 20 个冻结 selection episode 按 episode 而非候选做
联合 split-conformal 校准。每个 episode 只贡献候选、No-op 与配对增益残差的一个
联合最大值；5% 风险下有限样本覆盖下限为 `20/21=95.24%`。功率层联合裕量为
`2.03e-6`。计入切换、逐事件真实 bit、最大包时延和 RF 能量成本后，selection20
接受率为 `89.5%`，策略事件 mean-worst 为 `0.63740`，接受动作的约束违例、净收益
违例和 episode-any 违例均为 0。该证书仍因确定性信道及工程裕量未由独立随机
CSI/丢包残差校准而标记 `certificate_ready=false`。

中层没有重新统一扩大 Top-k，而是采用最多三步的原子 B=2 N5/N6。每一步只改变
一/两个目标的依赖闭包。候选先由两个严格可行下界排序：复用当前功率得到的下界，
以及令所有 UAV 依次集中到目标 q、再进行公共列时间共享得到的

```text
c_q = sum_i b_i a_iq,
t_mix = 1 / sum_q (1/c_q).
```

代理只负责排序；每步最多 Top-8 候选必须再经过 4 轮 6/16-bit 列生成验证，且只
接受严格单调改进，No-op 始终显式保留。训练池曾证伪“用目标中位数补全所有未观测
替代边”：该做法造成 `75.5%` harm；将未观测结构边下界置零并加入 DD 硬门后，
harm 降为 0。这一负结果说明置信域约束比扩大候选池更重要。

冻结 selection20 上，三步结构层在 267 个触发事件中 260 个找到严格改进；触发
事件的 power-only/structure mean-worst 为 `0.31950/0.48242`，接受动作平均增益
`+0.16731`，harm 为 0，预测绝对误差 `1.37e-7`。三层原始路由在 600 个因果事件
上由原部署 `0.37389/26.0%` 和全事件功率层 `0.63859/55.17%` 提升到
`0.68078/68.5%`（mean-worst/QoS 率）。包含每个 episode 首个不可因果事件时，
检测残差门后的 overall 值为 `0.66127/66.29%`，联合裕量仍为 `2.03e-6`，检测约束
与净增益违例均为 0。

更新后的最终事件物理 Oracle 进一步表明：原部署、固定结构功率、单工联合结构/
功率、全双工 mean-worst 分别为 `0.67764/0.91817/0.98475/0.98815`；单工联合可行解
20/20 达到 0.6，而全双工平均只再增加 `0.00340`。因此当前主要差距是低成本分布式
结构搜索逼近误差，不是必须转向波形层或全双工。放松同几何上界也不能再直接用于
宣称慢几何必要性，因为它放松了共同 owner、角色、容量和跨目标功率耦合。

创新性保持克制：UAV-ISAC 轨迹/资源联合优化、Dantzig--Wolfe/列生成、conformal
控制和多智能体 LNS 均已有先例。当前可辩护的是组合机制：**可观测双基地因果修正
-> 始终 RF 可行的量化有限轮功率修复 -> 嵌套物理上界路由 -> B=2 原子结构搜索
-> episode 联合候选/No-op/增益残差门**。它属于潜在集成创新，不作“首次提出”断言。

当前仍有三个部署阻塞项：多步结构 dependency-commit 的 header/bit/时延/能量尚未
逐步计入统一收益门；3 dB/0.5 ms 仍是工程裕量而非随机 CSI 校准量；慢几何动作只被
路由、尚未实现。因此结构控制器继续默认关闭，也没有使用新的 test 或 confirmation
种子。

新增实现与证据：

- [`geometry_gain_predictor.py`](../uav_isac/coordination/geometry_gain_predictor.py)
- [`power_repair_transport.py`](../uav_isac/coordination/power_repair_transport.py)
- [`dual_guided_structure_repair.py`](../uav_isac/coordination/dual_guided_structure_repair.py)
- [`episode_joint_conformal.py`](../uav_isac/evaluation/episode_joint_conformal.py)
- [`audit_power_repair_transport_trace.py`](../tools/audit_power_repair_transport_trace.py)
- [`audit_dual_guided_structure_trace.py`](../tools/audit_dual_guided_structure_trace.py)
- [`cross_split_transport_certificate.json`](../results/architecture_v2_scale_k6q6_geometry_feedback_gate_d0_12/cross_split_transport_certificate.json)
- [`cross_split_routed_detection_certificate.json`](../results/architecture_v2_scale_k6q6_geometry_feedback_gate_d0_12/cross_split_routed_detection_certificate.json)
- [`selection20_structure_frozen_validation.json`](../results/architecture_v2_scale_k6q6_geometry_feedback_gate_d0_12/selection20_structure_frozen_validation.json)
- [`selection20_final_updated_physical_oracle.json`](../results/architecture_v2_scale_k6q6_geometry_feedback_gate_d0_12/selection20_final_updated_physical_oracle.json)
- [`summary.json`](../results/architecture_v2_scale_k6q6_geometry_feedback_gate_d0_12/summary.json)

本轮隔离回归为非 belief `507 passed`、belief 校准 `14 passed`，无断言失败。

本 Gate 未使用新的最终测试或 confirmation 种子，新增控制器继续默认不接入部署。
隔离全回归为 `487 passed + 14 passed`；没有断言失败。

### 4.20 Gate D0.14：通信有界 Top-1 与对偶区间反馈排序

D0.13 的 Top-8 结构搜索仍遗漏了一项关键成本：每个候选的精确功率验证都需要
owner feedback/price 往返，已接受原子动作还需要 prepare/vote/decision。若只计算
最终动作而不计算搜索过程，约 13 次精确验证会把通信复杂度隐藏在控制器之外。本 Gate
将结构周期定义为严格串行的完整序列：

```text
当前结构功率验证
  -> [Top-1 候选功率验证 -> 若接受则原子提交]，最多 3 步。
```

序列证书逐步累计 header、epoch、digest、索引、6-bit 价格、binary16 Deflection、
正交反馈争用、Shannon 时延、RF 能量和原子提交，并同时检查总时延不超过 `0.1 s`、
单包不超过链路 deadline，以及每架 UAV
`P_comm + sum_q P_sense,q = 1 W`。解析最小通信功率采用机器精度级单向上取整，修复了
门限复算约 `-3.4e-15 dB` 的浮点误判；SNR 门限本身没有放宽。

仅使用可行下界的 Top-1 会降低排序质量。为此利用功率层已经公开的量化目标价格
`lambda in simplex(Q)`，对每个候选构造严格区间：

```text
L(S') <= t*(S') <= U_lambda(S'),
U_lambda(S') = sum_i b_i max_q lambda_q a'_iq.
```

`L` 是当前功率重放、公共全目标列混合和“当前列--公开价格最佳响应列”最优凸组合三者
中的最大值，因而始终对应真实可行 RF 分配。`U_lambda` 来自 max--min LP 的单纯形
对偶，对任意公开价格均是有效上界。候选只按

```text
score(S') = 0.5 L(S') + 0.5 U_lambda(S')
```

排序；上界绝不进入接受条件。Top-1 候选仍须通过有限轮量化功率复算、严格单调改进、
完整通信序列和 episode-conformal 净收益门，否则执行 No-op。训练池中
`kappa=0.25/0.5/1.0` 的 raw mean-worst 分别为
`0.72147/0.72157/0.71468`，纯上界明显过度乐观，因此冻结区间中点，不再用 selection
结果调参。

冻结 selection20 的 600 个因果事件中，267 个触发结构层，完整序列 267/267 可行；
平均 `6282.63 bit`、平均总协议时延 `27.70 ms`、最大 `46.44 ms`，功率恒等式误差不超过
`2.22e-16 W`。相对只用可行下界的通信闭合 Top-1，raw mean-worst 从 `0.67129`
提高到 `0.67895`，QoS 率从 `0.66333` 提高到 `0.68167`，已接受动作 harm 为 0；它也
接近尚未完整计费的 Top-8 `0.68078`，说明对偶反馈改善了排序，而不是靠扩大候选池。

calibration20 与 selection20 继续按 episode 隔离。5% 风险下有限样本覆盖下界为
`20/21=95.24%`，联合检测余量为 `2.03e-6`。计入切换、逐事件真实 bit、最大单包时延
和 RF 能量后，443 个候选动作接受 414 个；包含每个 episode 首个不可因果事件在内的
总体 mean-worst/QoS 率为 `0.65507/65.97%`。接受动作的检测约束违例、净收益违例及
episode-any 违例均为 0。

研究主线仍是**分布式结构与感知功率分配/控制层**，不是波形层；OTFS 波形、子载波、
波束和模糊函数仍固定。可辩护的创新性是“公开对偶价格形成可证区间排序 + 通信有界
Top-1 原子 LNS + 完整验证/提交计费 + episode 级 No-op 安全门”的组合机制，仍只表述为
潜在集成创新，不作首创断言。

部署证书仍为 `false`。当前 trace 工具集中持有因果预测增益张量；虽然区间的加和项可按
UAV/owner 分解，owner-local 最佳提案 Token、量化字段及候选竞争通信尚未逐 bit 实现。
此外，确定性 U2U 重放尚无独立随机 CSI/丢包残差校准，`3 dB/0.5 ms` 仍是工程裕量，
slow-geometry 动作也尚未实现。控制器继续默认关闭，本 Gate 没有使用新的最终测试或
confirmation 种子。

新增实现与证据：

- [`structure_sequence_transport.py`](../uav_isac/coordination/structure_sequence_transport.py)
- [`dual_guided_structure_repair.py`](../uav_isac/coordination/dual_guided_structure_repair.py)
- [`audit_dual_guided_structure_trace.py`](../tools/audit_dual_guided_structure_trace.py)
- [`train20_structure_top1_dual_interval_k0p5_development.json`](../results/architecture_v2_scale_k6q6_geometry_feedback_gate_d0_12/train20_structure_top1_dual_interval_k0p5_development.json)
- [`selection20_structure_top1_dual_interval_k0p5_frozen.json`](../results/architecture_v2_scale_k6q6_geometry_feedback_gate_d0_12/selection20_structure_top1_dual_interval_k0p5_frozen.json)
- [`cross_split_routed_top1_dual_interval_k0p5_transport_certificate.json`](../results/architecture_v2_scale_k6q6_geometry_feedback_gate_d0_12/cross_split_routed_top1_dual_interval_k0p5_transport_certificate.json)
- [`gate_d0_14_summary.json`](../results/architecture_v2_scale_k6q6_geometry_feedback_gate_d0_12/gate_d0_14_summary.json)

隔离全回归为非 belief `512 passed`、belief 校准 `14 passed`；无断言失败。

### 4.21 Gate D0.15：owner 单提案协议与完整决策时延门

D0.14 已计入候选验证与原子提交，但 trace 重放仍集中持有全部候选区间分数，且软收益门
只使用最大单包时延。后者只适合检查单链路 deadline，不能表示多阶段重构动作完成前的
信息年龄。因此 D0.15 明确取代 D0.14 的软时延结果，并采用

```text
T_decision = T_baseline
           + sum_s (T_proposal,s + T_verification,s + T_commit,s)
```

计算 `lambda_delay T_decision`。最大单包时延继续作为硬链路约束，不再作为完整动作的
软时延代理。专门的防回归测试保证只要完整协议字段存在，收益门就不能退回最大单包字段。

候选竞争改为有界 owner 单提案协议。B=2 弱目标集合被划分给其当前 owner；每个 owner
只发送本组区间分数最高的一个 N5/N6 原子候选。未量化时该分解不改变全局 Top-1：

```text
max_o max_{S' in C_o} score(S') = max_{S' in union_o C_o} score(S').
```

因为所有候选只改变两个弱目标的单/双目标块，每步 owner 提案数不超过 `B=2`。每个
目标旧边和新边各不超过 `L_pair`，所以边翻转数满足
`E_toggle <= 4 L_pair`；角色变化不超过 3 个 UAV，owner 变化不超过 2 个目标，依赖闭包
参与者不超过 K。Token 显式包含 header、epoch、digest、proposer、N5/N6 类型、受影响
目标、角色/owner/边差分和 `[L,U]`。`L` 用 IEEE binary16 向下取整，`U` 向上取整；
不支持的量化宽度直接报错，而不是使用未计数的精度侧信道。

协议时序固定为

```text
当前结构功率验证 -> owner 正交提案 -> Top-1 功率验证
                   -> 若严格改进则 prepare/vote/decision。
```

calibration20 上，每事件最多 1/2/3 个原子步骤的 raw mean-worst 为
`0.71319/0.72028/0.72150`，平均完整结构时延为 `18.14/26.48/29.58 ms`。在冻结的
切换、bit、完整决策时延和 RF 能量权重下，平均净效用反而为
`+0.13372/+0.08256/+0.06122`，正净效用率为 `70.29%/63.24%/58.82%`。因此第二、
第三次验证的边际检测收益小于信息年龄成本，最终冻结为**每个事件最多一次原子结构
重构**，而不是继续堆叠 LNS 步数。

冻结 selection20 的 267 个结构触发事件全部通过完整序列硬证书。每事件 owner 提案平均
`1.36` 个、最多 `2` 个，提案空口平均 `76.33 bit`、最大 `259 bit`；全部结构协议平均
`4085.42 bit`、最大 `6383 bit`，平均时延 `17.86 ms`、最大 `22.63 ms`，功率恒等式
误差不超过 `2.22e-16 W`。raw mean-worst/QoS 率为 `0.67336/67.33%`，已接受动作
harm 为 0。

calibration20 到 selection20 的 episode split-conformal 覆盖下界仍为 `20/21=95.24%`，
联合检测余量为 `2.03e-6`。使用完整决策时延收费后，443 个候选动作接受 347 个；包含
每个 episode 首个不可因果事件的总体 mean-worst/QoS 率为 `0.62787/64.68%`。接受动作
的检测约束违例、净收益违例和 episode-any 违例均为 0。作为反证，三步策略在相同完整
时延口径下只有 `0.59942/63.87%`，说明用更深搜索提高 raw 检测值会降低真实控制净收益。

本 Gate 的潜在创新性是“owner 分组的定向区间 Token + 通信有界单步 Top-1 + 完整决策
信息年龄收费 + episode 级 No-op 安全门”的组合，而不是新的波形或普通 MAPPO 网络层。
研究重点仍是分布式结构/感知功率分配和安全控制；OTFS 波形、子载波、波束与模糊函数
保持固定，不作首创性断言。

部署证书继续为 `false`。当前重放已发送候选描述与区间，但 owner 候选缓存仍由集中 trace
张量模拟，尚未从实际送达 Token 的年龄、丢包和版本状态逐项重建；随机 CSI/丢包残差也
没有独立校准，`3 dB/0.5 ms` 仍是工程裕量，slow-geometry 动作尚未实现。控制器继续
默认关闭，本 Gate 未使用新的最终测试或 confirmation 种子。

新增实现与证据：

- [`owner_proposal_transport.py`](../uav_isac/coordination/owner_proposal_transport.py)
- [`structure_sequence_transport.py`](../uav_isac/coordination/structure_sequence_transport.py)
- [`dual_guided_structure_repair.py`](../uav_isac/coordination/dual_guided_structure_repair.py)
- [`audit_geometry_feedback_cross_split.py`](../tools/audit_geometry_feedback_cross_split.py)
- [`train20_owner_proposal_steps1_development.json`](../results/architecture_v2_scale_k6q6_geometry_feedback_gate_d0_12/train20_owner_proposal_steps1_development.json)
- [`selection20_owner_proposal_steps1_frozen.json`](../results/architecture_v2_scale_k6q6_geometry_feedback_gate_d0_12/selection20_owner_proposal_steps1_frozen.json)
- [`cross_split_owner_proposal_steps1_full_delay_certificate.json`](../results/architecture_v2_scale_k6q6_geometry_feedback_gate_d0_12/cross_split_owner_proposal_steps1_full_delay_certificate.json)
- [`gate_d0_15_summary.json`](../results/architecture_v2_scale_k6q6_geometry_feedback_gate_d0_12/gate_d0_15_summary.json)

最终隔离全回归为非 belief `518 passed`、belief 校准 `14 passed`；无断言失败。

### 4.22 Gate D0.16：owner-local 两部分物理证书与反馈控制变量审计

D0.16 对 D0.15 的信息边界做了反向审计。此前 lag-1 系数由 trace 中的完整
`privileged_alpha/g_dd/chi_rep` 张量解析重建，因而“上一帧系数为正”并不等价于
“该边曾被本地观测”。在 calibration20 中，完整物理支持平均为 `175.73` 条边，
真正由上一帧已选且有正感知功率的边平均仅 `11.28` 条；约 `93.6%` 的物理边并无
直接反馈来源。当前帧 `privileged_candidate/g_dd` 和全局真值目标状态也不能作为部署输入。

为避免把零增益支持错误混入普通回归误差，本 Gate 将物理证书拆为两部分。首先只在
双向实际送达 Token 的端点对上形成候选，并由接收 owner 的本地 belief 计算 OTFS
延迟--多普勒有效度，采用严格支持判据

```text
g_dd_hat(i,j,q) - m_phys > g_min.
```

其次，上一轮已选且有正感知功率的边反馈目标级双基地雷达方程充分统计量

```text
kappa_q = a_ijq R_iq^2 R_jq^2,
a_hat_ijq = kappa_q / (R_iq_hat^2 R_jq_hat^2),
```

并形成 `[a_hat exp(-m_phys), a_hat exp(+m_phys)]`。一个独立 episode 内对所有帧、候选边、
DD 假阳性以及双侧对数增益误差共同取最大值，只校准一个联合半径。这样不会把同一事件的
上千相关候选当成独立样本，也不会把上下两个 5% 证书误称为 95% 联合证书。

40 个互不重叠、且与 selection/stress/test/confirmation 均无交集的开发事件上，5% 探索性
半径为 `0.245581`，阶次 `39/41`，对应乘法区间 `[0.78225, 1.27836]`。但端到端还包含
检测安全门和链路门，必须满足 union bound 风险预算

```text
alpha_total >= alpha_physics + alpha_detection + alpha_link.
```

因此下一 Gate 的物理层冻结风险改为 `2.5%`；在 40 个校准事件下只能使用最大次序统计量，
覆盖下界为 `40/41=97.56%`，半径为 `0.253913`。链路随机残差尚未校准，所以这仍不是
部署证书。

反馈闭环使用当前协调检测 EMA 作为 No-op 控制变量，而不是让几何模型同时替代已部署基线：

```text
P_hat(S') = clip(P_feedback(S) + P_phys_lower(S') - P_phys_lower(S), 0, 1).
```

这修复了候选与 No-op 使用不同偏差基线的问题，但没有消除逐目标风险。在最后一次未揭盲的
开发 holdout20 上，平均 Token 候选/物理准入边数为 `179.9/100.88`，准入物理精度
`99.9934%`，物理 episode-any 越界为 `10%`。单步 Top-1 LNS 的 raw mean-worst/QoS 为
`0.54851/46.17%`，相对 No-op 伤害率 `0.167%`；完整结构协议平均 `3614.10 bit`、
`16.41 ms`，最大 `22.79 ms`，1 W 功率平衡误差不超过 `2.22e-16 W`。因此本轮没有
通过性能或安全 Gate。

进一步的事件联合检测残差达到 `0.73608`，单一全局加性半径会拒绝所有动作。原因是某些
非瓶颈目标可能在 worst 指标改善时显著下降；不能通过取消 `forall q` 约束来隐藏该问题。
下一步冻结为“目标级自归一化反馈证书”：不确定度显式包含 Token 年龄/送达状态、本地 belief
协方差、距 `g_min` 的 DD 支持裕量、物理区间宽度、检测反馈--物理预测分歧和结构切换年龄。
MAPPO/Student 仍只估计局部边际价值，owner Token 携带 `kappa_q`、检测反馈与不确定度，
确定性硬投影层独占提交权。已揭盲 holdout 不再复用；归一化分数冻结后必须换新的独立开发事件。

本 Gate 的研究重点仍是**分布式结构、感知功率分配与事件触发安全控制**，不是波形优化。
OTFS 波形、子载波、波束和模糊函数保持固定，只通过 DD 有效度、双基地雷达方程和 1 W RF
预算进入硬约束。控制器继续关闭，未使用新的最终 test 或 confirmation 种子。

新增实现与证据：

- [`owner_local_physics.py`](../uav_isac/coordination/owner_local_physics.py)
- [`audit_owner_local_physics_certificate.py`](../tools/audit_owner_local_physics_certificate.py)
- [`audit_dual_guided_structure_trace.py`](../tools/audit_dual_guided_structure_trace.py)
- [`audit_geometry_feedback_cross_split.py`](../tools/audit_geometry_feedback_cross_split.py)
- [`d016_feedback_holdout20_owner_local_physics_frozen.json`](../results/architecture_v2_scale_k6q6_geometry_feedback_gate_d0_12/d016_feedback_holdout20_owner_local_physics_frozen.json)
- [`d016_feedback_holdout20_owner_local_lns_feedback_frozen.json`](../results/architecture_v2_scale_k6q6_geometry_feedback_gate_d0_12/d016_feedback_holdout20_owner_local_lns_feedback_frozen.json)
- [`gate_d0_16_summary.json`](../results/architecture_v2_scale_k6q6_geometry_feedback_gate_d0_12/gate_d0_16_summary.json)

### 4.23 Gate D0.17：目标储备、双侧反馈证书与快慢环并行审计

本 Gate 继续聚焦**分布式结构、感知功率分配和事件触发安全控制**，不进入波形层。OTFS 波形、子载波、波束和模糊函数均保持固定；物理层只通过 DD 支撑、双基地增益、检测概率单调映射以及单 UAV `通信功率+感知功率=1 W` 进入硬约束。

首先修复了自归一化分数中的一个数学错误。旧分数额外校准逐目标概率差，但控制器并未消费该量；同时候选概率经过 `[0,1]` 裁剪后，反馈锚定项不再代数消去，少数事件产生 `1e4--1e5` 的伪异常分数。新证书只校准实际用于决策的逐目标双侧包络：

```text
A_q = max(F_q-L_0,q, 0) + eps,
B_q = max(U_0,q-F_q, 0) + eps,
L_c,q = clip(P_hat_c,q-m A_q, 0, 1),
L_0,q = clip(F_q-m A_q, 0, 1),
U_0,q = clip(F_q+m B_q, 0, 1).
```

由最小值算子的单调性，

```text
min_q P_c,q - min_q P_0,q
  >= min_q L_c,q - min_q U_0,q.
```

因此不再使用“所有目标最大误差之和”惩罚 worst 增益。事件内所有帧和目标仍只取一个最大分数，40 个独立事件在 `alpha=0.05` 下使用第 `39/41` 阶统计量，覆盖下界为 `95.12%`。

候选生成也从事后拒绝改为储备优先。约束

```text
L_c,q >= min(L_0,q, P_floor)
```

先利用检测函数的严格单调性反解为每个目标的最小 Deflection，再加入有限轮 Dantzig--Wolfe 主问题。主问题先满足储备，再最大化 worst；储备约束和 max--min 约束的对偶乘子合并为公开目标价格，每个 UAV 仍只用本地增益生成完整 RF 列。所有列及其凸组合保持逐 UAV 感知功率预算，最终 `1 W` 平衡误差不超过 `4.44e-16 W`。结构 Top-1 代理增加了储备优先的公共列水填充下界；这是真实可实施时分凸组合，不把 relaxed upper 当成可行动作。

首次揭示的全新训练池 holdout20 与既有 80 个开发事件及 selection/stress/test/confirmation 全部无交集。未加储备的 owner-local LNS raw mean-worst/QoS 为 `0.61509/54.83%`，动作伤害率为 0，说明搜索层有价值而证书是瓶颈。揭示后该集合只用于架构诊断，不再称为独立最终验证。

储备、双侧证书和 slow-geometry 快速临时修复组合后，诊断集合上的硬传输约束策略从 No-op `0.39971/30.48%` 提高到 `0.47302/40.81%`（mean-worst/QoS），候选接受率 `22.89%`。接受事件的同步区间越界、认证服务下界违例、真实逐目标 no-harm 违例和 worst 净增益违例均为 0。owner-local DD 准入精度为 `99.9965%`，episode-any DD 支撑越界为 `5%`。但是该结果仍明显低于 `0.60`，且属于揭示后的开发诊断，不能作为部署证书。

快慢环并行只把认证 mean-worst 从 `0.47265` 提高到 `0.47302`，因此 slow route 等待期 No-op 不是主瓶颈。统一扩大 Top-k 仍被否决；下一主线必须是：在闭环重放中持久化已接受的结构/功率/Token/反馈状态，并实现带转移证书的慢几何优化。bit、时延和能量当前已作为容量、deadline 和 RF 功率硬约束；在没有约束对偶价格前，不再把任意常数权重重复加到检测概率上。

部署状态仍为 `false`。剩余阻挡项包括：策略改变后需要新的独立校准和未揭示开发集；闭环状态尚未跨事件传播；慢几何动作尚未实现；随机 CSI、丢包及 Swerling/模型漂移残差尚未联合校准。没有使用新的最终 test 或 confirmation 种子。

新增实现与证据：

- [`self_normalized_feedback.py`](../uav_isac/evaluation/self_normalized_feedback.py)
- [`maxmin_power.py`](../uav_isac/coordination/maxmin_power.py)
- [`dual_guided_structure_repair.py`](../uav_isac/coordination/dual_guided_structure_repair.py)
- [`audit_self_normalized_feedback_cross_split.py`](../tools/audit_self_normalized_feedback_cross_split.py)
- [`d017_self_normalized_holdout_seeds_k6q6.json`](../config/d017_self_normalized_holdout_seeds_k6q6.json)
- [`gate_d0_17_summary.json`](../results/architecture_v2_scale_k6q6_geometry_feedback_gate_d0_12/gate_d0_17_summary.json)

隔离回归为非 belief `531 passed`、belief `14 passed`，无失败。

隔离全回归为非 belief `521 passed`、belief 校准 `14 passed`，无断言失败。一次合并进程在既有
belief `eigvalsh` 用例中触发 MKL 原生中止；拆分重跑后两组均通过，因此记录为运行库并发问题，
不记录为算法回归通过前的断言失败。

### 4.24 Gate D0.18：动作时刻对齐、持久闭环与物理区间双证书

本 Gate 完成了一次架构级修正，而不是继续修改 Top-k 或 MAPPO 网络。研究重点明确为
**分布式感知结构分配、连续 RF 功率再优化和事件触发安全控制**；OTFS 波形、子载波、波束与
模糊函数保持固定。波形物理只以延迟--多普勒可分辨性、双基地雷达增益和单 UAV
`通信功率+感知功率=1 W` 的硬约束进入控制层。

#### 动作时刻对齐

D0.17 trace 的 `local_obs` 对应动作前状态，而 `g_dd/alpha/P_D` 对应动作执行后的物理状态。
用动作前几何预测动作后检测量会产生系统性时序偏差。D0.18 按环境中的同一运动学更新对
owner-local 状态作因果投影：

```text
p_i^+ = Project_Omega(p_i + Delta p_i),
v_i^+ = BounceAndClip(Delta p_i / dt, v_max).
```

目标位置若具有椭球不确定集
`(p-p_hat)^T Sigma_p^{-1}(p-p_hat) <= rho^2`，令
`r_p=rho sqrt(lambda_max(Sigma_p))`，则双基地时延满足确定性界

```text
|Delta tau| <= 2 r_p / c.
```

多普勒界由单位视线向量在位置球内的扰动上界和目标/UAV 速度界共同得到；把所得时延、
多普勒归一化区间代入 `sinc^2`，在区间端点、零点和整数零陷处取严格最小值，形成
`g_dd^L`。超出 DD 支撑的边下界置零并 fail closed，不再把“当前已选边”无条件继承为安全边。
动作对齐后，固定目标基准的全边联合对数残差从约 `0.220` 降至不超过
`4.39e-6`，DD 准入物理精度为 `100%`，60 个独立 episode 中没有 episode-any DD
支撑越界。这表明此前的主要物理模型误差来自时间索引错位，不是需要更宽的经验裕量。

#### 有适用域的目标不变量缓存

双基地充分统计量 `kappa_q=a_ijq R_iq^2 R_jq^2` 按目标独立保存，并携带 age/version；禁止
跨目标填补，过期后下界归零。本基准中 `tracking_enabled=false`、目标固定、`RCS=1`、
`use_swerling=false`，所以 `kappa_q` 在 episode 内保持不变，150 帧缓存具有物理依据。
该结论**不能**外推到运动目标、随机 RCS 或 Swerling 起伏；这些场景必须引入目标状态转移和
模型漂移不确定度，而不能继续沿用长缓存。

#### 持久闭环和双证书

审计不再把每一帧视为独立反事实：只有通过证书的结构才进入下一帧已部署状态，连续 RF
功率则按冻结 trace 的当前局部提案作每事件 recourse。候选替代 No-op 的反馈证书仍满足

```text
L_c,q >= min(L_0,q, P_floor),  for every q,
min_q L_c,q - min_q U_0,q > 0.
```

新增物理区间证书使用候选物理下界和 No-op 物理上界满足同一逐目标条件。控制器当前允许
`feedback OR physics`，但两个分支的并集风险尚未做联合 episode 级分配，所以仍是开发证书；
不能由“每个分支单独看起来安全”推出并集已具有相同覆盖率。

冻结参数为 4 轮、6-bit 价格、16-bit 反馈、Top-1、每事件最多一个原子结构步骤，边残差
双侧余量 `1e-5`，反馈和控制归一化余量均为 `1.0`。两个 30-seed 集与历史开发集及
selection/stress/test/confirmation 均不重叠，统计单位为独立 episode seed，不把 episode
中的候选或帧误当成独立校准样本：

| 独立划分 | 因果帧数 | 持久闭环 mean-worst | QoS 率 | 接受动作 |
|---|---:|---:|---:|---:|
| calibration30 | 900 | 0.49869 | 52.89% | 467 |
| validation30 | 900 | 0.64067 | 69.33% | 578 |
| 合并 60 seeds | 1800 | **0.56968** | 61.11% | 1045 |

1045 个接受动作中，worst 伤害、反馈区间越界、物理区间越界、逐目标认证服务违例和真实
逐目标 no-harm 违例均为 0；最大完整协议时延 `55.97 ms`，功率平衡误差不超过
`4.44e-16 W`。安全层因此通过开发审计。但 validation 单独超过 0.60、calibration 只有
0.49869，两者均值差 `0.14198`；合并 mean-worst `0.56968<0.60`。因此 Gate 状态是
**安全通过、性能失败**，不得挑选 validation 结果宣称架构达标，部署控制器继续关闭。

本轮还否决了两条表面上更“深”的修改：每事件做两个顺序 LNS 步骤会增加 bit/时延且性能
略降；无条件继承已选边的 DD 支撑会产生假安全接受。真正有创新价值的部分是“动作时刻一致的
owner-local 物理预测 + 有版本的目标不变量 Token + 确定性 DD/检测区间 + 持久化 No-op
基线 + 学习器只提供边际价值、硬安全层独占提交权”的组合。

下一 Gate D0.19 不再放宽证书或扩大 Top-k，而应针对划分不稳定性实现**带转移证书的时域候选
生成**：在短时域内联合传播 UAV/目标状态不确定度，优化保守 worst 检测下界，并显式收费
切换、bit 和决策时延；结构仍只执行首个原子动作，下一事件滚动重算。随机丢包、随机 CSI、
Swerling/模型漂移、实际 Token 中的 `kappa/age/version` 量化成本以及 live controller RF
recourse 必须进入独立事件校准。在安全与合并性能同时通过前，不使用最终 test 或 confirmation
种子。

新增实现与证据：

- [`owner_local_physics.py`](../uav_isac/coordination/owner_local_physics.py)
- [`physics_interval_gate.py`](../uav_isac/evaluation/physics_interval_gate.py)
- [`audit_dual_guided_structure_trace.py`](../tools/audit_dual_guided_structure_trace.py)
- [`d018_action_aligned_closed_loop_seeds_k6q6.json`](../config/d018_action_aligned_closed_loop_seeds_k6q6.json)
- [`d018_action_aligned_calibration30_frozen_closed_loop.json`](../results/architecture_v2_scale_k6q6_geometry_feedback_gate_d0_12/d018_action_aligned_calibration30_frozen_closed_loop.json)
- [`d018_action_aligned_validation30_frozen_closed_loop.json`](../results/architecture_v2_scale_k6q6_geometry_feedback_gate_d0_12/d018_action_aligned_validation30_frozen_closed_loop.json)
- [`gate_d0_18_summary.json`](../results/architecture_v2_scale_k6q6_geometry_feedback_gate_d0_12/gate_d0_18_summary.json)

完整回归按 MKL 稳定边界拆分执行：非 belief `530 passed`，belief 相关 `22 passed`，合计
`552 passed`，无断言失败。合并 belief 进程在 Windows/MKL `eigvalsh` 内发生原生中止，
所有用例隔离重跑均通过；该事件记录为运行库批处理问题，不伪装成一次完整单进程通过。
