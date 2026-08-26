# L2 联合结构定位器：模型、推导与证据边界

## 核心论点

在固定几何、per-watt 双基地系数和残余感知预算下，现有 L2 失败主要来自候选语法无法表达
跨目标的角色复用，而不是连续功率不可行；本文实现的联合结构定位器通过“分族原子改写 +
跨目标多样性 beam”显著提高开发轨迹上的 exact-witness recall，但尚未构成独立 blind 认证。

## 术语表

| 规范术语 | 定义 |
|---|---|
| formal episode floors | 正式评价阈值 `(0.60, 0.70, 0.80)` |
| optimizer reserve floors | 优化器内部鲁棒阈值；当前相关 trace 为 `(0.61, 0.71, 0.81)` |
| Pool A | 仅使用部署结构、物理系数、预算和冻结 L1 对偶量生成的 oracle-free 候选 |
| exact referee | 只在候选冻结后用于判定可行性和字典序最优性的 MILP/LP 裁判 |
| closure | 受影响目标数与参与 UAV 数之和 |
| development replay | 已暴露 exact digest 上的回放；不得解释为 blind 泛化 |

## 1. 问题形式化

令 `E` 为双基地边集合，边 `(i,j,q)` 表示 UAV `i` 向目标 `q` 发射、UAV `j` 接收。
固定几何下，每瓦 Deflection 系数为 `a_ijq >= 0`，UAV `i` 的残余感知预算为 `b_i`。
固定合法结构后，连续功率层是线性规划；L2 的困难来自：

1. 每个 UAV 不能同时承担 TX 和 RX；
2. 每个目标只有一个 owner；
3. receiver report capacity 与 target support capacity 有限；
4. 同一 TX 预算在多个目标之间共享；
5. 优化目标先最小化 closure，再最小化协议 bit，最后最小化感知总功率。

旧 locator 对所有目标改写使用单一 global top-k。它会把“一条边的 support augmentation”与
所有全新一边/两边结构直接竞争，也无法表达“旧 RX 晋升为 TX，同时迁移 owner”的无合法中间态
动作。因此 top-k 既不保持最小干预顺序，也不对跨目标角色复用闭合。

## 2. 新的分族动作语法

对每个目标分别生成以下动作族，并在族内按 raw capability 和 L1 reduced capability 双重排序：

- 保留 owner 的单边 support addition/removal/replacement；
- 保留 TX support 的 owner migration；
- `old RX -> TX` 的 receiver promotion；
- owner migration 与 old-owner promotion 的同步复合动作；
- 少量 generic raw/dual 高能力逃逸结构。

所有单边 addition/removal 均强制保留，而不再受 global top-k 截断。原因是一个局部仅排名第三的
TX，可能已被另一目标激活；复用该节点可减少 role change 和 participant 数，从而在字典序目标中
优于局部 top-1。

## 3. Oracle-free 联合代理

对候选结构 `E`，定义目标 `q` 的独立乐观能力上界

\[
C_q=\max_j\sum_{i\in\operatorname{TopS}_j(q)} b_i a_{ijq},
\]

其中 `S` 是 target support 上限。定义归一化能力

\[
\rho_q(E)=\frac{\sum_{(i,j,q)\in E}b_i a_{ijq}}{C_q}\in[0,1].
\]

beam 的物理排序依次使用 `min rho`、bottom-3 mean、global mean 和
`sum_q pi_q sum_(i,j,q in E) b_i a_ijq`。同时按
`(closure proxy, toggled edges, participant signature)` 分桶，避免高能力但高干预结构清除低干预
候选。

该代理有意忽略共享功率竞争，因此不是可行性证书。它只负责定位；是否满足三地板、所需总功率
和字典序等价性仍由后续 exact referee 判定。这一隔离保持了 oracle-free 语义。

## 4. 开发回放结果

在既有 5-episode、32 个 genuine-L2 开发帧上，旧 v2 的 exact-digest recall 为 Pool A `0/32`，
Pool-B union `9/32`。第一版联合 locator 的回放为：

- Pool A exact-digest recall：`28/32`；
- candidate count：`1582--3433/frame`，中位 `2396.5`；
- 32 帧总 CPU 时间：约 `195.5 s`；
- 证据等级：`DEVELOPMENT_ONLY_NOT_BLIND`。

剩余 4 帧显示多个目标通过同一 UAV 复用 participant closure。进一步把
`|union_q V(Delta E_q)|` 的集合并代价显式化：对每个共享 UAV 运行一个按已改目标数分层的动态
beam，一次扫描同时产生 radius 2--4，而不是逐目标子集重复枚举。端点条件化 DP 的完整回放为：

- Pool A exact-digest recall：`32/32`；
- candidate count：`2089--4330/frame`，中位 `3054`；
- 32 帧总 CPU 时间：约 `297.1 s`；
- 证据等级仍为 `DEVELOPMENT_ONLY_NOT_BLIND`。

因此 move grammar 在已暴露开发集上已实现 exact-digest closure，但候选规模和离线耗时同步上升。
该结果把下一个瓶颈从“候选不可达”推进到“候选压缩、在线增量计算与通信执行”，并不等价于实时
locator 已完成。

exact-digest recall 是严格但不完备的开发指标：未命中 exact digest 不代表候选池不存在相同
closure/bit/power 的等价结构；反之，命中也不能证明独立 seed 上的定位能力。

## 5. 当前限制与下一 Gate

1. 结果使用了已暴露 exact digest，只能说明语法修复有效，不能重写旧 M4-D1 的正式负结论。
2. radius-3/4 beam 增加了 CPU 与候选传输成本，当前更适合作为 locator teacher 或离线候选生成器，
   不能直接宣称满足实时 deadline。
3. 当前 32/32 是对已暴露 digest 的开发闭合，仍可能是针对该结构族的过拟合。
4. 尚未完成新 development split 的预注册、候选冻结和随后独立 exact referee。

下一阶段必须按以下顺序执行：新 development episodes 上冻结 locator 和预算；运行 exact referee；
只对 Pool-A-covered 帧测 decision-sufficient bit rate；最后在完全独立 blind episodes 上确认。任何
factor/dense 通信率结论都不得早于候选充分性 Gate。

## Claim–evidence map

| 声明 | 证据 | 状态 |
|---|---|---|
| 新语法可表达旧语法遗漏的跨目标角色复用 | 合成单元测试与开发 exact-digest replay | supported on development data |
| Pool A recall 从 0/32 提升到 32/32 | 哈希绑定的 endpoint-DP development replay | supported, not blind |
| 新 locator 可实时部署 | 尚无端到端协议 deadline 与复杂度证书 | needs evidence |
| 新 locator 在独立 seed 泛化 | 尚无预注册 blind confirmation | needs evidence |
