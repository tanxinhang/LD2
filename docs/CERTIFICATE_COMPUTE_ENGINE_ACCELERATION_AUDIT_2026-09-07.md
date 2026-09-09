# 证书计算引擎与稀疏化总体方案审计（2026-09-07）

## 结论

方案的主方向成立：应把证书上界用于候选剪枝、不可行预检和求解复用，并把协议的
`K_q_max=3` 转化为 selected-edge 数据结构约束。但原稿不能原样实施。建议评级为：

- 架构方向：A-；
- 数学严谨性：B-；
- 当前代码映射：B；
- 性能收益可信度：C+；
- 经本文修正后的可执行性：A-。

必须先修正四点：

1. `1/R^2` 只提供衰减和可证剪枝界，不提供精确的有限邻域。距离截断是带误差预算的
   近似，不是硬物理稀疏。
2. 路损项可由端点量相乘，但 DD 项依赖 Tx/Rx 距离和径向速度之和后的 sinc，不能写成
   两个端点 DD 因子的乘积。端点表避免重复几何，但边级求和、round 和 sinc 仍需执行。
3. 当前固定结构 LP 使用 `K*Q+1` 个变量；K=Q=16 时为257，不是128。压缩后变量上界
   是 `|unique(tx,q)|+1 <= 3Q+1=49`，约25只能作为实测均值，不能作为结构保证。
4. `EnvironmentCore.step()` 的真实回波路径仍生成完整 `DenseDeflection` 并物化对象。
   只改超边系数不能满足“生产路径零 pair-dense 物化”。

因此，更准确的一句话是：

> 把证书从事后报告升级为前置的可采纳上界引擎；把协议稀疏性作为精确约束，把距离
> 衰减作为可证剪枝条件，并让候选、证书、LP 和真实回波共享同一 selected-edge 表示。

## 修订后的六条不变量

### I1：禁止的是 pair-dense，不是某个恰好相同的数组形状

生产 hold 路径不得生成按 `(tx,rx,target)` 索引的完整张量。允许：

- packet/base endpoint table；
- 每 viewer 的 endpoint state、visibility 和 uncertainty；
- selected-edge COO，严格不超过 `K_q_max*Q`；
- topology-update 帧中的有界工作块 `[Bv,Bt,Br,Bq]`。

不应把 `[V,Kt,Kr,Q]` 整块列为长期允许形态：K=Q=100、V=100时，即使固定奇偶角色也
有2500万元素，float64约200 MB，仍为高阶增长。工作块必须受固定字节预算约束。

### I2：端点表是唯一几何来源，但边 DD 代数不是完全可分离的

每个 `(viewer,node,target)` 只计算一次：range、unit vector、radial velocity、range bounds
和 radial bounds。边级计算允许且仅允许：

- `tau=(R_tx+R_rx)/c`；
- `nu=(rho_tx+rho_rx)fc/c`；
- 两端点路损因子相乘；
- 对 tau/nu 的区间做 support、round 与 sinc 运算。

这样把范数和点积从 pair 层移到 endpoint 层，但不伪称 sinc 可分离。

### I3：上界先行，且区分 exact pruning 与 epsilon pruning

`a_plus` 可使用最小距离和 `DD<=1` 构造。若候选上界严格低于当前第3名的已知下界，
则 branch-and-bound 剪枝是精确的。若按距离直接丢弃边，则必须记录 omitted upper mass
或 score bound，并把它写入 epsilon 证书；这条路径不能进入“精确等价”验收组。

同一上界可用于：

- topology top-3 精确剪枝；
- reserve/QoS 不可达预检；
- LP 对偶上界和跨帧复用判定。

上界只能证明“不可行”或“无需继续搜索”；不能单独证明原问题可行。

### I4：下界必须对 sinc 周期结构成立

当前实现把偏移约化到最近整数后的 `[-0.5,0.5]`，并检查区间是否跨半整数。原稿中
“`|x|>=1` 就令下界为0”的规则是保守松弛，不是精确最小值：例如区间 `[1.1,1.2]`
不包含 sinc 零点，其最小值严格大于0。

可接受两种实现：

- exact periodic interval：判断所有整数零点/round 分界并取端点与临界点；
- conservative zero：只声称保守，不声称 exact，并监控证书间隙。

任何 float32 内核都必须采用 outward rounding、`nextafter` 或经证明的误差膨胀；在已经
用 float32 算错的界上最后转 float64，不能恢复包含关系。

### I5：稀疏结构贯穿到 LP 和真实回波

新增 sparse fixed-owner gain builder，直接消费 `(edges, values)`，不先填零张量再调用
`fixed_owner_gain_matrix()`。LP 仅为 unique `(tx,q)` 建变量，inactive transmitter 的
剩余功率按现有确定性规则补齐，以保持输出兼容。

hold 帧的真实 deflection 也必须有 selected-only 内核；否则系统仍会在每帧生成完整
`DenseDeflection`，违反 I1。topology-update 帧才允许块流式候选计算。

### I6：分布式信息边界不可因缓存或批处理改变

共享缓存不能只用 `(node_id,source_frame)`。当前状态按 target token 传输，且不同 viewer
具有不同收包集合/AoI。安全 key 至少应包含：

`(sender,target,source_frame,codec_profile,payload_hash)`。

缓存可共享“相同 payload 的解码和基础外推”，但 viewer 的 receipt mask、自身精确覆写、
belief target state 和 uncertainty 必须在 viewer 层处理。

## 对原13环节的审计

| 环节 | 结论 | 修订动作 |
|---|---|---|
| 状态打包 | 保留 | 继续作为协议固定载荷，不纳入计算稀疏化收益 |
| U2U缓存 | 条件成立 | 改用 packet级key；不得跨不同target token或不同payload复用 |
| 公共视图 | 成立 | 分成共享packet base和viewer overlay，避免复制完整表 |
| 不确定半径 | 成立 | 2x2闭式特征值可做；需和现有对称化/非负截断对拍 |
| 系数重建 | 部分成立 | 端点几何复用；DD仍在边/块级计算；块受字节上限约束 |
| 三套系数 | 部分成立 | nominal/lower/upper共享端点量；不能假定一个完全可分离公式 |
| 角色固定 | 成立 | 使用 `tx_idx/rx_idx`，不要硬编码奇偶，避免未来角色表改变 |
| 本地提议 | 原表述不成立 | Tx/Rx切片只是候选输入；仍须保留容量、唯一owner、排序与tie-break |
| 共识 | 可优化 | AND之外还要保留两端score均值、owner聚合、top-3和字典序消歧 |
| 五帧hold | 成立 | 候选复用、连续系数重算；当前O2/O3已验证 |
| gain压缩 | 强烈建议 | sparse builder直接生成每viewer的 `(tx,q,value)` |
| 功率LP | 强烈建议 | 257变量压到最多49；private-view语义和inactive-row补齐必须保持 |
| 回波/P_D | 原方案遗漏 | 新增 selected-only true-deflection 路径，否则无法实现零pair-dense |

原稿中的源码行号已经漂移，应在任务和测试中用函数符号作为锚点，例如
`EnvironmentCore._resolve_hyperedge_negotiation`、
`reconstruct_bistatic_coefficient_from_public_state`、
`mutual_endpoint_consensus`、`solve_fixed_structure_maxmin_power_lp`。

## 修订后的实施阶段

### 当前进度

- P0 已完成第一版：`--include-trace` 会启用默认关闭的 acceleration golden；
- 已覆盖一个 topology-update 帧和随后四个 hold 帧；
- 已记录 receipt/visibility、local plans、mutual/stable/selected、三套 gain、consensus
  streak、LP cache/price、执行功率、deflection 和 P_D；
- `tools/compare_acceleration_golden.py` 支持结构严格比较及 `atol/rtol` 数值比较；
- seed 11033 两次独立5帧运行在零容差下完全一致，exact mismatch=0，全部数值
  max-abs=0。

基准文件为：

- `results/acceleration_golden_p0_seed11033_a.json`；
- `results/acceleration_golden_p0_seed11033_b.json`；
- `results/acceleration_golden_p0_selfcheck.json`。

单份5帧JSON约4.7 MB，因此它用于短窗口结构对拍；10-seed性能门继续使用不含完整golden
的轻量结果，避免把诊断序列化成本混入生产时延。

### P0：golden、安全网和分段基线

现有固定seed回归继续作为基础，但 golden 需要补充非最终输出，否则候选排序错误可能被
最终top-3偶然掩盖。每个 viewer、每个 topology frame 至少冻结：

- receipt/AoI mask、端点表和 uncertainty；
- candidate eligibility、top-k score及tie margin；
- local plan、mutual set、consensus streak、selected COO；
- selected nominal/lower/upper；
- sparse gain、LP primal/dual、reserve precheck；
- true selected deflection和P_D。

验收分两组：exact组要求边、LP active set和P_D判定一致；epsilon组要求 omitted bound、
证书包含和QoS门同时通过。不得用“允许epsilon内排序翻转”掩盖无证据的翻转；只有当两边
分数区间确实重叠时才允许node-id消歧。

### P1：endpoint kernel + packet base/viewer overlay

先实现 endpoint dataclass/SoA 和 packet级共享缓存，再做 topology 帧的有界块组装。不要
一次生成完整 `[V,Kt,Kr,Q]`。共识向量化在 local-plan等价之后单独合入。

验收：exact golden；峰值工作内存上限；每个viewer的receipt/AoI不变。

### P2：certificate-first API

建议API返回：

`endpoint_bounds -> candidate_upper -> prune_mask -> selected_exact_bundle`

其中 selected bundle 一次给出 nominal/lower/upper。加入运行时抽检，而不是每帧对所有边
计算 `a_true`：开发模式全检，生产模式固定比例抽检和失败熔断。

验收：`lower <= true <= upper`；不可达预检无false reject；精确剪枝与未剪枝top-3一致。

### P3：sparse gain + sparse LP + selected-only true deflection

这是闭合 I1 的阶段，三项必须一起完成：

1. sparse fixed-owner gain；
2. unique `(tx,q)` LP布局与inactive-row确定性回填；
3. hold帧 selected-only true-deflection/P_D。

先串行求解验证变量压缩收益，再决定是否保留进程池。只有当 IPC/pickle 仍占主要比例时
才上共享内存。当前四进程池可能在变量压缩后反而不划算，应重新比较 serial、thread、
process 三种模式。

验收：变量数、矩阵nnz、solver wall、IPC wall分别报告；primal/dual、full power matrix、
P_D逐seed对拍；worst和QoS门不变。

### P4：大K的精确流式搜索与可选epsilon模式

仅在 K=32/64 档 profile 后启用。先做精确 upper-bound BnB 和固定字节块；距离截断作为
单独epsilon模式，默认关闭。复杂度报告必须区分 worst-case 与 observed active degree。

“近线性”只能在以下经验条件同时成立时声称：Q相对K的缩放关系已声明、每节点有效候选
度有界、每viewer收到的状态量有界、且AoI/覆盖不恶化。算法最坏复杂度仍不能直接写成
近线性。

### P5：内核和数值

Numba适合独立元素内核，但 `prange` 不自动保证跨平台位级一致。禁止 `fastmath` 进入证书
路径；分别验收 deterministic repeatability 和 numerical containment。float32仅用于
nominal ranking或带外向误差界的端点量，证书最终比较使用经外向舍入的float64界。

GPU维持禁用，除非大K实测数据规模和批次足以覆盖传输、launch与同步开销。

### P6：信道/AoI轨道

方向成立，但触发器不能只看 median AoI。至少使用 `(viewer,sender,target)` 的 p50/p95、
stale fail-closed率、target覆盖率和端点不对称率。建议触发条件为：p95 AoI连续多个窗口
超过2帧，且已能解释QoS或coverage退化；否则不改MAC。

## 可信的性能预期

当前O5 profile中，30帧累计：超边解析约443 ms，其中selected重建约160 ms，dense约77
ms、upper约9 ms；DenseDeflection物化约139 ms。当前完整10-seed回归约60.10 ms/帧，
控制关键路径约17.20 ms。

因此原稿“P1收益主体10--50x”和“K=16系统毫秒级”没有Amdahl证据。合理目标应分层写：

- coefficient/consensus子阶段：争取3--10x；
- LP子阶段：按变量与IPC实测争取2--5x；
- K16控制关键路径：先以低于10 ms为阶段目标；
- K16完整仿真帧：先以低于45--50 ms为阶段目标；
- K扩展：先证明 K=16/32/64 的斜率，再讨论K上百。

这些目标仍然有价值，而且比未经验证的“10--50x、上百近线性”更适合论文和工程验收。

## 触发器修订

| 信号 | 动作 |
|---|---|
| sparse LP后solver小于IPC | 先串行/线程，再评估共享内存；不默认保留进程池 |
| p95 AoI持续大于2且coverage/QoS退化 | 启动P6 |
| 可见性不对称导致mutual/coverage下降 | 先诊断交集丢失；再单独消融rounds=2 |
| 证书间隙偏大 | 分解为距离、量化、外推、DD四类gap后定向收紧 |
| 非重叠score区间仍发生排序翻转 | 视为bug，禁止用epsilon margin豁免 |
| 工作块超过内存预算 | 缩小Bv/Bt/Br/Bq，不物化全块 |

## 关于“优化买来协议升级空间”

这个叙事方向合理，但必须通过消融来成立。当前K16配置确实关闭了information ranking、
near-field residual，并把consensus rounds设为1、hold设为5。它们不能被统一解释为“仅因
计算预算关闭”：near-field还增加通信字段，rounds=2改变决策时延，information ranking
当前包含特征分解、Jacobian、CRLB和矩阵逆，并非一次廉价的分数替换。

正确论文路径是：先报告加速后释放的时延预算，再分别启用一个增强模块，比较新增耗时、
通信、AoI、worst/QoS和证书间隙。只有获得质量提升且仍满足deadline，才能声称加速支持了
更强协议，而不是事后叙事。

## Go / No-Go

按修订版推进：Go。

原样推进：No-Go。尤其不得将物理距离衰减写成精确硬稀疏、不得把主瓣外统一置零写成
精确下界、不得在未改真实deflection路径时宣称生产路径零pair-dense、不得预先承诺
K上百近线性。

## 逐步实施进度

- P0已完成：固定seed的五帧golden覆盖拓扑帧与四个hold帧，比较结构、三套gain、LP、
  价格、证书与P_D；独立重复运行在零容差下完全一致。
- P1第一段已完成：hold帧将viewer-local selected重建合为一个`(V,E)`批量内核；零容差
  golden通过。
- P1第二段已完成：selected值直接折叠为`(V,K,Q)` fixed-owner gain，不再中间物化
  `V`份`(K,K,Q)`；零容差golden通过。
- K16/Q16、10 seed × 30 frame上，相对O5控制关键路径均值17.20降至11.55 ms，整帧
  60.10降至53.94 ms；W=3 worst_min维持0.844917且QoS通过率100%。当前尚未达到控制
  路径低于10 ms或整帧45--50 ms的阶段目标，下一瓶颈需重新profile后再选择共享端点缓存
  或P3 sparse LP，不能按原计划盲目推进。
