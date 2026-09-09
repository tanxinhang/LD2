# 多 UAV 分布式 ISAC：当前系统总文档

> 状态日期：2026-09-01。本文同时给出当前模型、部署算法、最新数据、证据边界和下一步，
> 是项目的首要阅读入口。公式与代码级细节见 `CURRENT_SYSTEM_MODEL.md`；算法为什么演进
> 到当前形态见 `ALGORITHM_EVOLUTION.md`；所有历史数据和失败实验见 `EXPERIMENT_LOG.md`。

> **2026-09-07 架构升级提案**：面向K16/Q16整帧20--30 ms目标，见
> [`PREDICTIVE_CERTIFIED_ARCHITECTURE_2026-09-07.md`](PREDICTIVE_CERTIFIED_ARCHITECTURE_2026-09-07.md)。
> 方案采用统一高维时空状态、异步预测慢平面和selected-only精确证书快平面。

> **2026-09-01 身份门禁收口**：`config/system_manifest.yaml` 是 post-G2 基础契约，
> 不是可单独用于正式运行的完整场景；当前 executable profile 为
> `config/exp_strict_distributed_k16q16.yaml`，并绑定 K16/Q16
> reset-distribution/v2 blind bank。CI 与正式运行均对该 profile 做 strict 双向指纹校验。

> **G2-0 科学口径变更（2026-08-20）**：当前代码已把 raw deflection 从“能量除以
> 噪声功率”的量纲不闭合写法，修正为无量纲能量比 `E_signal/E_noise`。因此下表全部
> 100-seed 数据都是 **pre-G2 历史证据**，不能直接作为当前物理口径下的性能声明；
> 在可比的 pre-G2 结果中，V3-C0 `0.910/0.838` 是晚于 D1.10 修复后的科学基线。
>
> **G2-0.5 结论**：实高斯检测统计量已锁定 `c_det=1`，50 万样本 ROC 与解析式吻合；
> 但 link-budget falsification 显示默认 `n_CPI=128` 使 100–800 m 理想单链路全部
> `P_D>0.9999999`，且按 OTFS frame 解释的积分时间 `0.131072 s > dt=0.1 s`。
> 该阶段的阻塞点是 processing-gain/CPI 语义，不是算法性能；随后由 G2-0.6 闭合。
>
> **G2-0.6 闭合**：执行器每个控制动作只产生一个显式 OTFS 感知帧，故当前认证模型
> 固定 `n_CPI=L_eff=1`；`T_F=N·T_sym=1.024 ms < dt=100 ms`。审计距离的
> `P_D=1.000/0.989/0.124/0.00981`，不再全饱和，机器门禁现为
> `ready_for_g2_1=true`。剩余控制帧时间不是观测证据；多 look 必须另行实现并认证。
>
> **G2-0.7 功率语义闭合**：`1 W` 是通信与感知的联合 RF 上限，不能覆盖感知波形自身
> `14 dBm=25.1 mW/UAV` 的硬件锚点。现增加 `P_sense_max=0.0251 W`，允许 RF 预算
> 留空。单-seed paired smoke 中，8/8 与 6/6 不再全饱和；结果仅用于验证动态范围，
> 不作为正式性能声明。
>
> **规模化路线冻结为 CIS-ISAC**：保留 Exact L0/L1 与现有认证 L2/L3，未来只在 L1
> 后增加 Robust Task Slack 和包含 No-op 的 fail-open active set。该机制在 G2-1A/1B
> 完成前不进入 live；Student 只作 proposer，无线协调仍必须通过现有物理 admission。

## 1. 四文档体系

| 文档 | 唯一职责 | 更新时机 |
|---|---|---|
| `README.md` | 当前模型、算法、最新数据和决策的总览 | 任一正式 Gate 或系统边界改变 |
| `CURRENT_SYSTEM_MODEL.md` | 数学模型、算法定义、执行顺序、单位与代码映射 | 代码改变模型或算法语义 |
| `ALGORITHM_EVOLUTION.md` | 机制演进、理论动机、反例、保留/关闭决策 | 完成一轮算法判断 |
| `EXPERIMENT_LOG.md` | 过往实验数据、seed 协议、Gate 状态和产物路径 | 每次实验完成后追加 |

不得再为单次实验或单个想法新增活动 Markdown；需要保存的内容必须进入上述对应文档。

## 2. 系统边界

研究对象是只有 UAV 间通信、没有地面链路的多 UAV 协同 ISAC：`K` 架 UAV 在二维区域
内跟踪 `Q` 个目标，通过双基地/多基地链路积累检测证据。每架 UAV 每帧满足严格联合预算

```math
P_{k,\mathrm{comm}} + \sum_q P_{kq,\mathrm{sense}} \le 1\ \mathrm{W}.
```

系统显式建模：UAV/目标运动、Rician/Swerling 随机性、OTFS/DD 可用支撑、双基地路径
损耗、U2U Shannon 容量、量化 bit、时延、丢包、通信功率、感知功率、检测概率和
截获者检测能力。系统不生成原始 IQ 波形，不实现完整 MAC/路由协议、飞控、波束赋形、
波形协方差或多目标数据关联。

“分布式部署”的准确边界是：各 UAV 的动作和 Token 由共享策略基于本地可获得信息产生；
结构 Student 近似集中式教师；最终检测概率仍由环境级集中式证据融合模块计算。集中式 P0、
MILP、horizon oracle 和完整图教师只能作为训练/诊断参考，不能写成部署组件。

## 3. 当前算法

### 3.1 分布式学习外层

- 共享参数的集合式 Actor 处理可变 UAV/目标集合，避免固定身份 one-hot 成为主表达。
- 连续运动采用有界映射；新训练可选 smooth-disk 双射以保持 PPO 执行动作密度正确。
- Token 经过真实量化、容量、时延、功率和丢包路径；未到达的信息不能进入决策。
- 结构 Student 输出 owner/端点/目标意图；集中式教师只提供训练或参考标签。
- MAPPO/GAE 负责跨时域回报，解析控制器只解当前帧定义明确的子问题。

### 3.2 解析执行栈

```text
L0 通信资源：正交 U2U Shannon 反解 / no-waste 带宽
  ↓
L1 功率：固定结构下 QoS 字典序 max-min LP
  ↓
L2 结构：P0/Student 结构；只有通过证书的原子修复才允许提交
  ↓
L3 几何：capability/KKT 敏感度、多候选与 receding-horizon 前瞻
  ↓
安全层：1 W 联合上限、25.1 mW/UAV 感知波形上限、DD 支撑、三项 QoS 地板、隐蔽性、时延和 bit 门
```

固定结构下的功率问题是线性规划，可对该子问题声称全局最优；结构含离散变量，几何含
非凸信道和 DD 门，只能声称候选内精确比较、局部下降或已验证的上界，不能借用 LP 的
最优性扩张为全系统最优。

检测 QoS 同时使用：

- steady mean `>=0.80`；
- bottom-3 mean `>=0.70`；
- worst target `>=0.60`；
- episode QoS 可行率及其 Wilson 下置信界。

隐蔽性约束采用镜像 deflection/检测模型。若截获约束功率问题无解，执行路径 fail-closed，
禁止回退到无约束 max-min 功率。

### 3.3 默认关闭的研究机制

V3 dual-priced L2、coupling-aware congestion relief、task-regret Student 校准、议价 L1
和多种因子图/有限轮协调器均不是当前默认部署算法。原因分别包括闭环轨迹分叉、通信量/
时延过大、小样本混合结果或 LCB 未过门。代码保留用于审计，不代表已经认证。

## 4. 最新数据与证据状态

### 4.1 正式结果

| 系统 | N | steady | weak3 | worst | QoS | Wilson LCB | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| 4/4 frozen deployment | 100 | 0.9132 | 0.8848 | 0.7393 | 0.72 | 0.625 | 点估计 Gate PASS |
| 8/8 D1.10 independent blind | 100 | 0.9664 | 0.9633 | 0.9633 | 0.94 | 0.875 | 历史 pre-V3/PRE-G2 |
| 8/8 V3-C0 re-cert | 100 | 0.9240 | 0.9156 | 0.9150 | 0.91 | 0.838 | 最新可比 pre-G2 PASS |
| 6/6 V3-C0 re-cert | 100 | — | — | — | 0.59 | 0.492 | DISCLOSED FAIL |
| 6/6 Gate 1 multi-scale CE | 100 | 0.8633 | 0.8445 | 0.8419 | 0.78 | 0.689 | LCB FAIL |

不能再把 D1.10 `0.940/0.875` 称为当前主结果：它早于 V3-C0 的 `χ_rep` 抽样和能量
记账修复。按时间和物理语义，**V3-C0 `0.910/0.838` 是 pre-G2 最新科学基线**；但
G2-0 又修正了 deflection 量纲，故当前代码尚无 post-G2 的 100-seed 认证值。4/4 的
历史 LCB 未达到 `0.70`；6/6 Gate 1 为 `0.689<0.70`，也未完成置信认证。本文采用的
“Wilson LCB”统一指 `z=1.96` 的 **95% 双侧 Wilson 区间下端点**，不是 95% 单侧界。

### 4.2 已知尾部

8/8 D1.9/D1.10 的失败主要包括运动学不可达远目标和少量结构—功率—几何闭环耦合事件。
已经查看过的失败 seed 只能做根因分析，不能继续用于选择超参数。6/6 的主要问题不是单纯
增加网络容量即可解决，而是 Student 结构误差、弱几何和闭环敏感性的共同作用。

### 4.3 工程验证

- 2026-08-20 G3 capability routing 工作树分片全回归：非 belief
  `1036 passed`，belief `14 passed`，合计 `1050 passed`，无断言失败。
- 2026-08-25 现树全量回归刷新：`1198 passed`（171 个测试文件，81.7 s，
  无 skip/xfail；仅 tica transformer `norm_first` 警告 8 条）。自 08-20 后
  新增 belief-freshness/scale/role-capacity 等 148 项测试。
- Windows/MKL 混合长进程仍可能在 `numpy.linalg.eigvalsh` 内原生中止；因此不宣称
  单进程全量通过。
- 正式 Gate 会拒绝空数组、长度错位、非有限/越界概率和重复 episode seed。
- 每个新运行保存确定性源码快照、解析后配置、Git dirty 状态、依赖版本、命令行和输入
  检查点 SHA-256。

## 5. 当前科学结论

可以宣称：

1. 固定结构功率 LP、capability/PWL 证书及 KKT 几何敏感度具有明确数学定义和回归证据。
2. pre-G2 解析快层、结构层和几何层的组合曾在 8/8 独立 blind 协议上通过 QoS 与 LCB
   Gate；当前能量归一化口径必须重新认证，暂不外推该性能结论。
3. 通信资源、bit、时延、功率、DD 支撑和检测约束已进入可审计执行路径。
4. 多个直觉性方案被反例否决，包括功率无关结构排序、简单价格协商、无条件结构 relief
   和只看提交帧的单调门；这些负结果构成算法边界证据。

不能宣称：

1. 全系统或 L2/L3 非凸问题的全局最优；
2. 集中式教师、环境融合或诊断 Oracle 是分布式部署组件；
3. 6/6 已通过置信认证；
4. 普通 Token、Attention、MAPPO、CVaR、拍卖或原始—对偶方法本身具有首创性；
5. 解析链路仿真等价于波形级或硬件级验证。

## 6. 创新性定位

创新判断不再采用“某模块已有先例，因此不能再做”的表层口径，而采用“模块故障 →
同范式修复 → 缺失性质 → 外部模块必要性 → 正交消融”的口径。完整的 MARL/DRL 模块级
演进与重新审计见
[`INNOVATION_LITERATURE_REAUDIT_2026-09-07.md`](INNOVATION_LITERATURE_REAUDIT_2026-09-07.md)。
创新表述的脱水分级、近三年模块证据与升级门槛见
[`INNOVATION_DEWATERING_AUDIT_2026-09-07.md`](INNOVATION_DEWATERING_AUDIT_2026-09-07.md)。

当前必须首先区分两条执行链：

- `Structured MAPPO` 是学习研究链，包含 movement/assignment/sensing/message/rate/RF-split
  等策略头；
- 当前 canonical strict runner 是 **hold action + 固定控制载波 + 分布式解析协议栈**，没有
  learned actor，且运动、结构和感知功率分别由确定性模块执行。

因此 strict 性能不能归因给 MAPPO；MAPPO、Attention、Student、LP、factor message 和
certificate 也都不能以简单组合形成创新。当前只保留三个**有条件候选**：

1. 物理 exact factorization + decision-margin bit bound + outward interval 所形成的可认证
   decision-sufficient message；
2. 私有 belief 下局部 contribution interval 的可组合逐目标 QoS admission；
3. 若重新启用 learned policy，长期策略、凸内层和硬准入之间可识别且 credit-consistent 的
   混合控制接口。

以上均尚未闭环。M4-B/C 只支持机制级结果，M4-D 无自然候选；现有数据也未证明 factor
communication 能修复 Student 跨尺度、CT/CA belief 失配、目标增长的 sensing capacity 或
K16 私有视图 LP 时延。进入新理论前，必须先完成 actor-head responsibility matrix、oracle
ladder，以及等 bit learned-latent / factor-message 对照。

actor-head responsibility matrix 已完成，见
[`ACTOR_HEAD_RESPONSIBILITY_AUDIT_2026-09-07.md`](ACTOR_HEAD_RESPONSIBILITY_AUDIT_2026-09-07.md)：
strict runner 不构造 learned actor；三 seed × 20 帧反事实干预中，movement、role、message、
rate 和 sensing-weight 均不改变执行结果，communication-power 只改变 RF 记账而未改变检测。
下一阶段为 oracle state/candidate/rank/projection ladder。

oracle ladder 已完成，见
[`ORACLE_LADDER_AUDIT_2026-09-07.md`](ORACLE_LADDER_AUDIT_2026-09-07.md)。18 个未参与训练
seed 的 same-trace 结果显示：state 无增益，完整 candidate pool 对 episode-worst 有
`+0.00561` 增益，exact rank 对 steady/weak3 有增益但降低 QoS-feasible，projection
为 identity。创新方向采用“精炼提升”而非模块删减：以**决策权一致的多时间尺度混合
控制**为核心，以 **episode-QoS-regret candidate localization + 稳定排序**为性能增强，
factor message/certificate 保留为完成等 bit 闭环必要性检验后的条件性保证层。

candidate provenance × task-regret rank 联合消融已完成，见
[`CANDIDATE_REGRET_ABLATION_2026-09-07.md`](CANDIDATE_REGRET_ABLATION_2026-09-07.md)：CE rank
受益于 full candidate，但 task-regret rank 在 full candidate 上反而恶化，证明两者存在
显著交互。下一步创新对象应是 **candidate-aware、hold-aware 的 episode-QoS-regret
排序接口**，而非孤立替换 Student loss。

更严格的脱水结论是：截至 2026-09-07，没有已经被实验证明的算法首创点；“多时间尺度混合
控制”是架构表述，“candidate-aware、hold-aware episode-QoS-regret”是待验证算法假设，
均不能直接写成已完成的创新声明。具体降级理由与升级门槛见
[`INNOVATION_DEWATERING_AUDIT_2026-09-07.md`](INNOVATION_DEWATERING_AUDIT_2026-09-07.md)。

## 7. 下一步

多帧性能与闭环收敛指标已经接入 strict runner，定义与证据边界见
[`TEMPORAL_PERFORMANCE_AND_CONVERGENCE_2026-09-07.md`](TEMPORAL_PERFORMANCE_AND_CONVERGENCE_2026-09-07.md)。
多帧检测按窗口累积 Deflection，而不是平均 `P_D`；收敛同时检查尾窗均值、方差、斜率、
振荡、settling frame、QoS首次达到和持续保持率。该指标改变的是评价维度，不改变当前算法。

下一轮进入 **G2-1A paired bridge**，之后按 G2-1B blind confirmation 协议
重认证，再推进 6/6 尾部置信缺口。候选机制必须同时满足：

1. 在未查看、预注册的独立 seed 上评估；
2. 目标函数直接对应 episode 级三地板和 LCB，而非仅当前帧 `t*`；
3. 明确计算结构信息所需 bit 下界，并进入 Shannon 容量、串行化时延和 1 W 预算；
4. 使用 No-op/冻结 8/8 锚点作为安全基线；
5. 先做 falsification，小样本只决定“是否值得继续”，不产生正式性能结论；
6. 若 LCB 未过，结果保留为 `DISCLOSED FAIL`，不追尾调参。

candidate-regret 的第一步已经完成：`episode_regret.py` 固定三维 episode regret，并把
candidate unsupported 与 rank/execution residual 分开统计；现有 18-seed trace 没有 QoS
可行性翻转证据。下一步先冻结该指标，再做候选生成器/排序器的同预算独立盲测。

hold-aware 排序上界已完成，见
[`HOLD_QOS_RANK_AUDIT_2026-09-07.md`](HOLD_QOS_RANK_AUDIT_2026-09-07.md)：用训练 seed
校准后冻结的 future-hold q20 oracle，在18个未参与校准 seed 上将 worst 从 `0.5668` 提升到
`0.6100`，QoS feasible 从 `0.444` 提升到 `0.500`；CVaR仅提升 `0.00037`。该结果定位了
hold-time-scale 排序缺口，但因读取未来 realized `d_eff` 不能部署。下一步是训练 local-belief
hold-q20 predictor，并与当前 CE/task-regret Student 在新 seed 上同预算比较。

统计顺序固定为：G2-1A 使用已查看的原 100 seeds 做 paired bridge audit，只报告物理
修正造成的分布变化，不称 blind；随后冻结代码和阈值，G2-1B 使用全新、预注册且未查看
的 seed bank 做真正 confirmation。4/4、8/8 V3-C0、6/6 multi-scale CE 三个系统
必须一起跑；在 G2-1B 前保持 task-regret 默认关闭。

`tools/audit_detector_normalization.py --assert-ready` 是进入 G2-1 的可执行前置门禁；
G2-0.6 当前返回成功且 `ready_for_g2_1=true`；若未来配置重新引入未认证多 look、
全距离饱和或时间超限，门禁仍会 fail-closed。

G2-1A 现有执行器为 `tools/run_g2_1a_bridge.py`：自动锁定四个历史 100-seed bank、
冻结 checkpoint/Student 类型与物理口径，以单 worker 逐 seed 隔离并支持断点续跑。
当前四个系统均完成 **10/100**；`tools/audit_g2_1a_batch.py` 已对 episode 数组、三地板
QoS/Wilson 重算、通信守恒和 `K×0.0251 W` 感知总功率执行可复现审计且全通过。
该批次状态仍为 `IN_PROGRESS`，只用于发现错误和规模风险，不是 G2-1A 性能结论。当前
4×4 QoS 为 `10/10`，6×6 V3-C0、8×8 V3-C0 与 6×6 CE 均为 `1/10`；CE 相对
同 seed V3-C0 的 worst 均值差为 `-0.1114`，不支持提前启用 CE。

不同规模 worst 深度审计进一步确认：当前 4/6/8 配置不是 scale-only A/B，100-seed bank
的初始 matching bottleneck 均值约为 `363/523/555 m`。在合法 `0.0251 W/UAV`
sensing PA cap 下，代表性联合物理探针呈 4×4 可达、6×6/8×8 仍受限的方向；主要机制是
目标数 order statistic、双基地第二端点/匹配尾部、相对机动半径下降和 max-min 共享预算
水床效应。原 physical oracle 漏 sensing cap 的诊断口径已修正，旧的 worst=1 乐观结果作废。

后续算法路线现统一为 **Capability-Aware Hierarchical Scaling**：先完成 G2-1A/G2-1B，
再执行 S0-P 自然物理 scaling、S0-M 难度匹配 scale-only、S0-F common-controller
factorial 和 S0-O cap-aware attribution。CIS live 正式后移；只有同几何存在可行 witness、
但固定结构不行的 Region II 才进入 L2/CIS。relaxed physical ceiling 已证明不可能的
Region III 路由 L3；证据不足进入 unresolved shadow，不把启发式求解失败冒充物理不可行。
S2 anytime 仍必须先证明 capability-gauge 上下界覆盖 central LP；不得直接把 RMP dual
当完整问题下界。

优先研究方向是“置信约束的 task-regret Student + 决策保持自适应量化”，而不是继续增加
网络深度或无条件扩大通信轮数。其理论核心是将结构误差的能力损失上界、排序保持所需最小
量化精度和 episode 尾部风险统一到一个可验证门内。

G3-A/G3-B 的最新 post-G2 开发审计进一步收窄了方向：750 帧中 owner-consistent
capability 对实际 worst 的帧级 Spearman 仅 `0.184`，低于 matching-distance 的 `0.440`，
因此不实现以该量为直接目标的几何 SOCP。25 个 cap-aware 抽样帧的认证路由为
`I=4、II=21、III=0、U=0`；21 个 fixed failure 均存在同几何联合结构—功率可行 witness。
当前证据优先支持 L2 原子结构—功率联合部署，而不是 L3 移动。该结果来自 5 个已查看开发
seed，只是机制归因，不改变 G2-1A 10/100 的 IN PROGRESS 状态。

G4-A 随后在 21 个 Region-II 帧上完成 exact minimum-intervention MILP：21/21 获得
conservative 三地板可行 witness；role flip/edge toggle/参与 UAV 的中位数分别为 `1/1/2`，
16/21 的完整 closure 参与者不超过 3。结果支持下一步做 dependency-closed nested block，
但不支持固定 Top-2 小块覆盖所有失败：个别帧 closure cardinality 达 9。当前 oracle 是
集中式 shadow，尚未经过真实 L0 admission、量化/AoI 下界或闭环 paired replay。

**2026-08-21 provenance 更正：**以上 G3/G4-A/B/B2 数字是历史物理版本证据，不能继续
作为当前 L2 样本。最终 `T_sym` cancellation、`n_cpi=1`、`c_det=1` 口径下，对相同 5 个
开发 seed 的 25 帧重放，25/25 原离散结构均可在合法预算内通过纯功率重新部署满足任务，
当前 L2 structure-repair 帧为 0。trace 又缺少 sensing-power tensor 与 environment-level
`physical_pd`，所以当前 deployed-plan 指标不能从 `local_pd` 代替重建。历史 R1/S0 的图结构
边界仍可作条件性结构证据，但“21 个当前 Region-II/L2 repair 帧”这一标签已经撤销。

G4-A 同时发现 high-dynamic-range PWL 证书退化：跨数万 Deflection 的单条 chord 会过度
保守。新增 `P_D=0.999` saturating chord 尾段，以检测单调性保持严格下界；目前只用于
shadow，不修改 G2-1A 冻结 live 算法。

G4-B 的 nested-block shadow 最终都能恢复 21/21 witness，但局部发现效率未达预期：两种
排序首次可行块的中位数均为 5 UAV×6 targets，只有 4/21 在 ≤3 UAV、≤3 target 内可行。
裸 gain 更早恢复，closure-efficiency 更常匹配 exact 最小 closure，但后者需要更多 MILP
尝试；两者不存在一致占优。当前判定为 completeness PASS、locality-efficiency FAIL，
不从已查看开发 seed 选择排序，也不越级进入通信 admission/live。

G4-B2b 进一步证明 localization gap 可由组合可行性反馈显著缩小：exact fixing-deletion
filter 将 permission block 中位降至 `2 UAV×1 target×1 role`，15/21 为 ≤3×3 小块，
16/21 命中 G4-A minimum closure。但它需要 333 次 exact feasibility 调用；二分删除更差，
达到 471 次。因此该机制定位有效、计算过重，只作为离线冲突标签 oracle，不进入 live 或
G4-C。当前环境未提供 IIS/Farkas 接口，文档不把该结果冒充正式 IIS 证书。

G4-B2c 的顺序无关 backbone 在 13/21 帧非空、中位 2 个权限，且角色权限从未成为全局
必要项；这支持“目标/UAV 核心较稳定、具体角色可替代”的集合标签方向。但 raw complement
conflict-cut master 在 16-call/帧门内仅 1/21 收敛，强制 backbone 后没有改善。当前结论是
backbone PASS、raw master FAIL：valid cut 不等于 informative cut，不提高调用上限追结果。

G4-B2d 将该负结果升级为 oracle-certified irreducible conflict core 与 minimum hitting-set
master。真实 21 帧证明 core 确实稀疏（中位 3、最大 5），等权/字典序 master 都把 16-call
内收敛从 1/21 提高到 16/21；字典序 support 还达到 16/21 minimum closure 命中。但逐核
sequential shrinking 分别需要 2824/2822 次 exact feasibility 调用，且未达到预注册的
21/21、至少 18/21 收敛和低于 B2b 333 次调用。因此本轮结论是 conflict-core mechanism
PASS、online efficiency FAIL；QuickXplain 与 physics-order smoke 均更差并关闭。新增代码
保持 shadow/offline，未修改 live L0–L3。

G4-B2e（OLCS）进一步把目标从“小 core”改成“严格抬高 support-lex 下界”。tri-state oracle
现只允许 primal-validated feasible 或 solver `status=2` infeasible 形成证书，unresolved 永不
生成 cut；证书口径限定为 conservative-PWL-MILP，而非原始连续物理不可行。五个 B2d
触顶帧的 objective-face cut 为 5/5 certified-infeasible，只用 5 次 physical query；sublevel
二分用 19 次查询将五帧下界分别跨越 6–7 个标量层，验证 OLCS 理论有效。但从空 core 启动
的 greedy bundle 在五帧 16 轮内仍为 0/5：8-support 版 407 次 physical query，单-support
版虽降至 100 次却下界停滞。因此 face/sublevel separation PASS、greedy bundle FAIL，按门
不运行 21 帧、不进入 live。

通信/感知鲁棒性下一步必须包含两项敏感性而非直接改默认参数：短包链路用有限块长 normal
approximation 扫描 blocklength/目标 BLER，相对 Shannon 容量报告 bit/时延余量；感知侧对
残余 timing/frequency offset 扰动 DD active set，报告门翻转率、worst/weak3/QoS 与
fail-closed 触发率。两项完成前，不把解析 Shannon/DD 几何结果外推为硬件鲁棒性。

## 8. 使用与维护

常用验证：

```powershell
# 当前 post-G2 门；在新的 clean/blind100 产物注册前应 fail closed（exit 2）
python tools/assert_formal_gates.py
# 只复核 pre-G2 历史表，不代表当前系统认证通过
python tools/assert_formal_gates.py --evidence-epoch historical
pytest tests/test_config_validation.py tests/test_run_provenance.py `
  tests/test_assert_gate_thresholds.py tests/test_assert_formal_gates.py -q
```

文档维护规则：模型语义只改 `CURRENT_SYSTEM_MODEL.md`；算法判断只追加
`ALGORITHM_EVOLUTION.md`；实验完成后只追加 `EXPERIMENT_LOG.md`；正式状态变化同步更新
本文。旧专题内容由 Git 保存，不再恢复为活动文档。

## 9. R1 交互宽度审计（2026-08-21）

R1 冻结并检验“安全物理剪枝后的有效交互宽度远小于网络规模”假设。审计没有用
connected-components 代替 treewidth，而是在 21 个 Region-II 帧上分别构造 raw-target、
resource-target、summary-lifted incidence 和 task-mode clique 四种图，并对最多 12 个节点的
图做精确 elimination-DP treewidth。结果每帧依次为 `5/5/6/6`；summary-lifted 图始终是
`K6,6`，单一 12 节点巨分量，所有 UAV 的 target degree 均为 6。安全的 isolated-mode
ceiling 剪枝后仍保留中位 `793/900=88.1%` 模式，overlap HHI 中位 `0.1673`，接近六个
UAV 均匀参与的下界 `1/6`。

因此当前 K=Q=6 证据否决小宽度假设：lifting 只是把 clique 复杂度移入 incidence/frontier
state，并未产生 separator 优势。按预注册纪律停止 R2 task-mode equivalence、R3 dual
pruning 和 separator solver；该结论不外推为其他规模定理，但足以关闭当前数据上的
bounded-width 主线。

## 10. S0 安全结构约简（2026-08-21）

R1 已正式关闭 treewidth/separator 理论路线，不再寻找“更聪明的图表示”。S0-A 转而检验
resource-complete context-free safe screening。由于 sensing PA、角色和 owner/capacity 都按
UAV 分量独立记账，跨 UAV replacement 必然把替代 UAV 的某个资源分量从 0 增为正，因而
不存在 context-free componentwise dominance。当前模型内唯一非平凡充分规则是严格零增益
edge：全部零增益边可保持 feasibility；要无条件保持 minimum-intervention lex optimum，
还必须要求该 edge 在 reference structure 中未选中。

21 帧 exact 对照中，D-F 与 D-O 都保持 21/21，且三层 lex objective 完全一致；但只删除
`96/3780=2.54%` edges，每帧 `0/4/16`（min/median/max），正系数 UAV-target incidence
仍为 36 条、exact width 仍恒为 6。因此 S0-A 是 safety PASS、sparsification FAIL，不重开
width 路线。

S0-D 进一步发现组合致密不等于系数高维。由几何重建、允许对角线的逆距离外积为 126/126
精确 rank-1；禁止 `i=j` 后 operational 矩阵的 95% energy rank 中位为 2。加入当前 U2U-only
report reliability 不改变奇异值；DD gate 后仍有 `109/126=86.5%` 的矩阵 `r95<=2`，其余为
14 个 rank-3、3 个 rank-4。该结果只支持“dense low-energy-rank”研究假设；rank-1 截断误差
中位约 0.65，且截断 SVD 不保持逐项上下界，尚不能进入 feasibility oracle 或 live 算法。

逐元素审计进一步给出比 truncated SVD 更强的精确结构。receiver-only reporting 下，每目标
系数可写为 `u_q v_q^T - diag(u_q*v_q) - E_DD,q`，其中 `E_DD,q` 只保存 hard-DD
失活的 off-diagonal entries。126 个矩阵的最大相对重建误差为 `8.08e-16`，DD exceptions
共 96 个（2.54%）。这允许把系数描述量从 `O(K^2 Q)` 降到 `O(KQ+|S_DD|)`，同时不近似
感知物理；但它目前只是 exact capability representation，不代表组合 MILP 复杂度已经降低，
也未通过量化、AoI、FBL 或 decision-preservation 门。

## 11. M0–M4 Decision-Adaptive Factorized Capability Communication（2026-08-21）

当前结论严格区分两类 scaling：组合计算仍致密且未解决；完整 capability 信息的表示可精确
factorize。这里的 payload 是“传输完整 sensing-capability information”的 semantic shadow
record，当前 live 从未广播 dense `A`，所以不声称把现有 live 开销从 `O(K^2Q)` 降到
`O(KQ)`。当前共同 log scale 也由集中式 frame 数据确定，分布式 scale 协商尚未解决。

当前研究主张收敛为：**dense decision coupling does not imply dense decision information**。
证书不是系统目的，而是零容错决策语义下的安全工具；真正优化对象是，在给定决策裕量时，
为保持 L1/L2 正确决策至少需要多少物理能力信息。M4-C 已验证 L1 的一个固定结构特例，
但“Decision-Adaptive Factorized Capability Communication”仍是候选架构名，不是 live 成果。
统一目标暂记为

```text
B*_delta(s) = min { B : Pr[d(theta_B) != d(theta)] <= delta },
epsilon_information(B) < m_decision.
```

现有 interval certificate 只是 `delta=0` 的严格特例；尚未实现或验证 `delta>0` 风险模式。

M1 明确计入 protocol/epoch/bit-depth header、float32 scale、factor codes，以及 DD sparse-list/
full-mask 两种编码的较小者。8/12/14/16 bit 时 factor payload 中位分别为
`707/995/1139/1283 bit`，相对同 bit-depth direct-dense semantic record 少
`52.8%/55.1%/55.8%/56.3%`。这不是同任务证书公平比较，只是精确 payload accounting。

M2-0 使用 outward-rounded float32 log range 和 closed quantization bins，3–16 bit 在 21 帧
所有系数上均为 0 containment violation；范围传播只覆盖 `h=0`、exact DD exception support。
既有 DD Lipschitz 定理是 frame 内 position-only certificate，明确不支持跨帧速度变化，因此
AoI 尚未过门，不能把 M2-0 写成完整 M2。

M3 将逐项系数区间严格传播到 Deflection、`P_D` 和 worst/weak3/steady。21 个 G4-A witness
及 0/0.25/0.5/0.75/1.0 功率缩放在所有 bit-depth 下均为 0 false-feasible、0
false-infeasible。minimum-power witness 精确贴 nominal QoS floor，故有限保守区间始终
0/21 CERT_FEASIBLE。引入设计 headroom 后出现可验证 trade-off；gauge-balanced 更新结果
与公平 dense 对照见下方 M4-A/B。nominal minimum-power 的零裕量结论保持不变。

当前配置还满足 `P_comm<=0.5 W`、`P_isac,total=1 W`、`P_sense,max=0.0251 W`，所以
`min(1-P_comm,0.0251)=0.0251 W` 恒成立；本配置当帧通信功率不挤占 sensing PA cap。
factor 消息的近期代价是 serialization、5 ms deadline、delivery、AoI、energy 与后续闭环，
不是当帧 sensing power。M5 前不得使用“多发 bit 会降低本帧 sensing budget”的表述。

M4-A/B 首先修复 factor gauge：外积对 `u->cu,v->v/c` 不变，旧实现把 detector scale 全放
在 `v` 中却共用一个 log range，造成表示性动态范围浪费。逐目标对齐 TX/RX log-range
center 后系数逐项不变，8-bit coefficient log-interval 中位宽度由 `0.412` 降到 `0.0487`。

公平 baseline 让 dense-linear、dense-log、factor-log 共享 27-bit header 与同一 DD codec；
dense 只编码 active entries，三者都逐级选择首个 CERT_FEASIBLE payload。所有方法、bit、
headroom 均为 0 containment violation、0 false certificate。最强 dense 是 dense-log：

| headroom | factor adaptive median | best dense-log median | factor/dense |
|---:|---:|---:|---:|
| +0.0025 | 995 bit | 2235 bit | 0.445 |
| +0.005 | 851 bit | 1891 bit | 0.450 |
| +0.01 | 851 bit | 1891 bit | 0.450 |

三点均为 21/21 同一 static three-floor CERT_FEASIBLE，故 M4-A safety 与 M4-B
communication-value Gate 在开发机制集上 PASS。该优势不是 log quantization 对 linear 的
优势，而是在相同 log baseline 下由 72 个 factors 对约 176 个 active coefficients 的生成
结构带来。

M4-C-light 进一步固定 G4-A structure，只要求保持 L1 路由，不要求 power vector 或 LP basis
相同。由 `A^-<=A<=A^+` 与 capability 单调性得到
`gamma(A^+)<=gamma(A)<=gamma(A^-)`；仅当右端 `<=1` 才进入 L1，仅当左端 `>1`
才 fallback，其余 fail-closed。对每帧用 residual-budget 缩放构造
`gamma*=0.90/0.97/0.99/1.01/1.03/1.10`，126/126 场景均在 16 bit 内正确认证，
coefficient containment、gauge bracket violation、false route 都为 0。裕量 0.10、0.03、0.01
的典型首次认证约由 6–8 bit、8–10 bit 上升到 10 bit；margin 与 min-bit 的描述性
Spearman `rho=-0.824`。这些场景来自 5 个相关 episode，不作显著性或泛化结论。

因此 M4-C-light safety/route-value Gate PASS；M4-D L2 lex-optimal equivalence set、固定物理
scale/分布式 scale 协商、AoI/FBL 与 live transport 仍未知，不将 M4 外推为部署通信优势。

M4-D-light 随后复用可严格重放的 G4-A 与两条 G4-B2b permission filter，候选池在量化前
冻结；依赖缺失 `physical_pd` 的两条 G4-B ladder 被排除。三个来源每帧全部回到同一原结构，
21/21 candidate count=1；旧 G4-A
为 20/21 非零 closure、中位 3，而当前重放为 0/21、中位 0。`0 bit` “决定”只是 singleton
退化，不能解释为 L2 information gain。M4-D safety 子程序为 0 containment、0 power-bracket、
0 nonoptimal，但 candidate-pool、decision 与 communication-value Gates 均 FAIL；状态为
`INVALID_INPUT_DEGENERATE_CANDIDATE_POOL`，不得为得到正结果而人工添加 one-toggle 候选。

### P0 当前模型 Layer Provenance（2026-08-21）

M4-D 的正式状态进一步统一为 **`BLOCKED_BY_LAYER_PROVENANCE`**；
`INVALID_INPUT_DEGENERATE_CANDIDATE_POOL` 是这次旧输入的具体表现，不是当前 L2 的性能结论。
新增 schema-3 trace 契约与 fail-closed 审计，逐帧绑定：seed/episode/frame、UAV/目标位置速度、
实际结构/role/owner、实际 sensing/communication power、per-watt coefficient、`g_DD`、
`chi_rep`、环境 `P_D`、mean/weak3/worst 与 residual sensing budget；run 级同时绑定
code/config/physics/checkpoint 四个 SHA-256。physics hash 显式覆盖 `c_det`、`n_cpi`、无隐藏
`L_eff`、`T_sym`/带宽约定、`P_sense,max`、`P_FA`、DD 门、report reliability 和 fusion mode。

旧 selection20 trace 经 P0 工具检查为 schema 1，按预期 FAIL，产物为
[`_p0_layer_provenance_legacy_gate.json`](../results/_p0_layer_provenance_legacy_gate.json)。
后续严格按以下顺序推进：

```text
P0 fresh natural current-policy trace + provenance PASS
  -> P1 D / L1 / L2 / III / U re-triage
  -> N_L2 > 0: 仅真实 L2 帧进入 M4-D
  -> N_L2 = 0: 关闭 M4-D，L2 降为稀有应急层
  -> 再启动 M5（AoI/FBL/live transport）
```

证据等级同步冻结：解析 factorization 是 model-level；M4-B 是 mechanism-level；M4-C 是
controlled boundary audit；R1/S0 只描述历史条件几何，不再代表当前 L2。fresh trace 先取
自然当前轨迹，不做 difficulty filtering；stress 轨迹只能另标为 diagnostic。

P0 完成后已用相同 5 个 exposed development seeds 生成 750 帧 fresh natural trace；schema、
四类哈希和 frame-axis Gate 全部 PASS。在不按结果筛选的 `frame mod 15 = 0` 系统抽样上，P1
得到 50 帧：`D/L1/L2/III/U = 17/1/32/0/0`。其中 32 个 L2 是 conservative-PWL joint
MILP 的构造 witness，随后用真实 Gaussian Deflection `P_D` 再验；最小
worst/bottom-3/average 为 `0.610000/0.710002/0.810055`，最大 per-UAV power violation
`1.53e-16 W`。fixed `gamma` 在 L2 帧最小 `1.0181`，repair-witness `gamma` 最大
`0.9931`，所以不是边界容差误分。

这意味着历史 M4-D 产物仍标为 `BLOCKED_BY_LAYER_PROVENANCE`，但**新的 M4-D 输入路线已经
解锁**：只允许使用这 32 个 fresh L2 帧重新冻结自然算法候选。统计单位仍仅 5 个相关
episode；`64%` 不能外推为总体 L2 发生率，也不授权 live 修改。

### M4-D0/D1：oracle-free candidate Gate（2026-08-21）

M4-D 被拆为 D0 candidate provenance、D1 candidate sufficiency、D2 decision rate。D0 对
全部 50 个系统抽样帧运行，不读取 P1 label 或 witness；候选先序列化、SHA-256 并记录 freeze
timestamp，新的 exact referee 只在冻结后启动，时间顺序 Gate PASS。canonical L1 dual 同时
输出 target price 与 budget multiplier，并执行 `sum_i eta_i b_i=1` KKT 检查。

第一版 whole-role-partition completion 在 32 个 L2 帧上 Pool A/B 均 0 coverage，暴露其会
重建过多目标、closure 偏大。未读取 witness 内容，只按该结构性缺陷补入所有合法单步 owner
replacement、TX support replacement/add/remove 和全局 role swap，冻结 v2：

| 判定 | 帧数 |
|---|---:|
| Pool A 覆盖 exact lex-equivalence set | 0/32 |
| Pool B 覆盖、Pool A 不覆盖（localization gap） | 9/32 |
| Pool B 也不覆盖（move-grammar gap） | 23/32 |
| D2 eligible | 0/32 |

因此当前状态是 **`STOP_AT_CANDIDATE_LOCALIZATION`**。这不是 factor/dense rate 失败，也不能把
candidate miss 写成 `B*=infinity`。不运行 bit sweep，不复用 B2b feasibility feedback，不继续
在同 5 个 episode 上追加 heuristic。M4-A/B/C 保留；M4-D 给出有边界的负结果。
