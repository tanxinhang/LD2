# 6×6 / 8×8 预算耦合结构优化与开发验证

日期：2026-08-22  
证据等级：开发集因果验证；尚未完成预注册 100-seed blind 认证。

## 1. 已修复的根因

旧 P0 用单位功率系数选择离散结构，再由 L1 执行真实逐机功率预算。对
K=Q=6/8，这会把同一架发射机视为可同时对每个目标使用一单位功率，实际
却必须满足

\[
\sum_q p_{iq}\le b_i,\qquad b_i\simeq 0.0251\ {\rm W}.
\]

结果是 P0 在抽样帧中退化为单发射机覆盖全部目标，只有 1/K 的机队感知
预算位于有效 Tx 角色。修复后的 P0 在结构层同时处理实际预算，不再把
不可执行的单位功率松弛当作结构目标。

## 2. 严格预算耦合模型

对边 e=(i,j,q) 定义准入变量 x_e、边功率流 f_e、Tx 角色 r_i、目标本地
owner 变量 o_{jq} 和最差 Deflection y。主要约束为

\[
\sum_{e:\,tx(e)=i}f_e\le b_i,\quad
0\le f_e\le b_i x_e,
\]

\[
x_{ijq}\le r_i,\quad x_{ijq}\le1-r_j,\quad x_{ijq}\le o_{jq},
\]

\[
\sum_j o_{jq}=1,\quad
\sum_{ij}x_{ijq}\le K_q^{\max},\quad
\sum_{iq}x_{ijq}\le C_j,
\]

\[
y\le D_q=\sum_{ij}a_{ijq}f_{ijq},\qquad \max y.
\]

可选第二阶段在保持 y 的条件下最大化 QoS-floor 截断后的加权目标质量。
限时 MILP 只有在独立复核变量界、整数性和全部线性约束后才接收 incumbent；
连续流还会向逐机功率单纯形径向投影，因此执行功率严格不超预算。返回支撑
对下游固定结构 L1 可行，故 L1 的最优最差 Deflection 不低于返回的可执行
incumbent 值。限时 incumbent 不被表述为全局最优证书。

## 3. 实时分解算法

严格 MILP 修复了性能，但 8×8 仍约 2.98 s/次。实时路径枚举所有非平凡
Tx/Rx 角色划分；K=8 时仅 2^8-2=254 个。对每个划分：

1. 用预算加权 ceiling
   \(c_{jq}=\sum_{i\in\operatorname{TopL}}b_i a_{ijq}\)
   选择目标 q 的本地 owner 和至多 Kqmax 条边；
2. 执行接收容量检查；
3. 用精确固定结构 max-min LP 分配功率并给候选评分；
4. 删除 LP 中零功率边，避免虚增报告负载；
5. 对最佳角色划分做两轮 owner 坐标改进，只有精确 LP 词典序分数严格提高
   才接受。

该分解是物理可行的 primal lower bound，不声称求得联合 MILP 全局最优。

6×6 另外使用任务满足型组合护栏：先构造低复杂度单-Tx legacy 候选，并
用真实预算 LP 检查鲁棒开发门 [worst, bottom-3, steady] =
[0.80, 0.85, 0.90]。只有全部通过才保留与冻结策略/L3 协调良好的旧结构，
否则切换多-Tx 枚举修复。该门槛是开发期抗 5-frame 陈旧裕度，不是理论
常数，必须在 selection split 校准后才能进入 blind 认证。

保持 5 帧时还存在第二类尺度问题：几何与 DD 支撑可使缓存边在两次 P0
之间失效。最终开发控制器采用**硬拓扑有效性事件**：只要缓存边不再属于
当前物理图，就提前运行完整预算耦合 P0；瞬时 worst-PD 软触发关闭。后者在
dev10 中造成过度结构抖动（QoS 由 9/10 降至 7/10），不能与物理失效混为
同一事件。

## 4. 闭环结果

### 单种子严格因果对照

| 场景/路径 | worst PD | QoS | P0 s/frame | P0 s/resolve |
|---|---:|---:|---:|---:|
| 6×6 旧 P0，seed 237 | 0.368034 | 0 | 0.0261 | 0.1262 |
| 6×6 预算耦合 MILP | 0.849441 | 1 | 0.1077 | 0.5209 |
| 6×6 最终 satisficing+enumerated | 0.899274 | 1 | 0.0197 | 0.0952 |
| 8×8 旧 P0，seed 248 | 0.102918 | 0 | 1.4517 | 7.0244 |
| 8×8 预算耦合 MILP | 0.9999999 | 1 | 0.6151 | 2.9762 |
| 8×8 最终 enumerated | 0.9999890 | 1 | 0.0825 | 0.3991 |

### 3-seed 已暴露开发集复核

6×6 seeds=[237,14,274]：

- 旧 P0 mean worst=0.477365，QoS=1/3；
- 最终组合 mean worst=0.853934，mean weak3=0.867831，mean steady=0.914533，
  QoS=3/3；
- P0 加权均值约 0.01430 s/frame、0.06918 s/resolve。

8×8 seeds=[248,196,241]：

- 旧 P0 mean worst=0.264696，QoS=0/3；
- 最终枚举 mean worst=0.999858，QoS=3/3；
- P0=0.08211 s/frame、0.39729 s/resolve。

两种最终路径的摊销 P0 时间均低于 0.1 s 控制周期。这里只核算代码记录的
P0 求解时间，不把整套环境、审计和 GPU 推理墙钟时间冒充在线 P0 延迟。

### 6×6 拓扑事件 dev10（当前最终开发候选）

seeds=[14,237,274,410,479,586,705,725,752,866]：

- mean worst/weak3/steady = 0.906602/0.906602/0.914849；
- QoS=9/10，Wilson 95% LCB=0.652281；唯一失败 seed 586 的
  worst/weak3/steady=0.772302/0.772302/0.780919；
- 拓扑失效/完整回退帧率=0.072667，P0 resolve 帧率=0.240667；
- P0=0.02046 s/frame、0.08501 s/resolve；
- U2U 交付率=0.876462，时限违约率=0.123538，1085.568 bit/frame，
  平均通信功率=0.001812 W；
- 逐帧证据中 10 个种子均未再出现 worst PD≈PFA=0.001 的失效边硬崩塌，
  但若干种子仍有瞬时 PD<0.6，故不能声称逐帧 QoS 得到硬保证。

### 8×8 当前代码回归

seeds=[248,196,241]：mean worst/weak3/steady =
0.999859/0.999859/0.999893，QoS=3/3；P0=0.07893 s/frame、
0.38194 s/resolve；U2U 交付率=0.999206、时限违约率=0.000794。
结果复现此前数值，表明 6×6 事件诊断没有污染默认关闭事件的 8×8 主路径。

## 5. 通信与感知边界

- 保留逐机 sensing PA 上限和联合 RF 余量语义，没有把约 0.97 W 的非感知
  余量错误转给 sensing PA。
- local-only 检测只在唯一 owner 接收机内累积证据，不跨接收机隐式融合。
- 每目标配对上限、接收报告容量和 Tx/Rx 单角色互斥均为硬约束。
- 零功率边被删除；最终平均边数约为 6×6 的 9 条和 8×8 的 13 条，未用
  冗余边虚增性能。
- 闭环中的现有 U2U 交付率/时限违约仍按链路模型执行；结构控制本身仍使用
  集中可见的 per-watt coefficient。故当前结果是“集中教师可执行结构”，
  不是已完成协议传输、同步、AoI 和丢包闭环的完全分布式部署证明。

## 6. 负结果与剩余风险

- 仅关闭 MILP 第二阶段虽快约 30%，但仍不满足 8×8 实时性。
- 单纯把 time limit 压到 0.4 s 会产生 y=0 的无用可行 incumbent，不能作为
  稳健实时方案。
- 6×6 纯枚举在 seed 14 从旧 P0 的 0.998 降到 0.550；严格 MILP 可恢复到
  0.950，表明单帧结构代理与冻结策略/L3 存在闭环协调问题。最终任务满足型
  护栏解决了三个开发种子，但其鲁棒门仍需独立校准。
- 开发 trace 的逐帧 physical worst 可含初始瞬态，不能用 episodic steady
  指标掩盖；正式报告必须同时给出瞬态和稳态定义。
- 当前 3-seed 结果只能证明方向和工程可行性，不能替代 100-seed Wilson/CI、
  独立 blind bank、波形级接收机或真实硬件验证。
- 试验了保持角色、owner、容量和边数不变的最小变更桥接：所有候选均经
  当前预算精确 LP 认证，且能消除 PFA 硬崩塌；但直接桥接 dev10 仅 8/10
  QoS。20 回合 repair-aware 微调在三诊断种子达到 3/3，却在 dev10 将失败
  从 seed 586 转移到 seed 274，整体仍为 9/10 且 mean worst 仅 0.864060，
  低于硬事件控制器 0.906602。因此桥接与微调均保留为默认关闭的实验分支，
  不进入最终候选。这是负结果，不得选择性隐去。
- seed 586 说明 episode 汇总可掩盖物理瞬态：无事件配置 steady 约 0.998，
  但证据 trace 在帧 39/62/69 出现 worst PD≈0.001。硬事件/桥接消除了这些
  PFA 帧，却暴露冻结策略对安全修复的长期闭环适配不足。后续应在独立
  selection split 上训练显式双时间尺度策略，而非容忍无效边来追求汇总分。

## 7. 复现入口

- 严格模型与快速分解：`uav_isac/physical/feasibility_oracle.py`
- 在线接入与当前预算事件证书：`uav_isac/environment/env_core.py`
- 6×6 最终开发配置：
  `config/exp_800_k6q6_budget_coupled_p0_satisficing_event.yaml`
- 6×6 默认关闭的桥接实验：
  `config/exp_800_k6q6_budget_coupled_p0_satisficing_bridge_experimental.yaml`
- 8×8 最终开发配置：
  `config/exp_800_k8q8_budget_coupled_p0_enumerated.yaml`
- 离线基准：`tools/benchmark_budget_coupled_p0.py`
- 当前代码全量回归：1114 passed，8 warnings；Conda/MKL 混合入口曾在
  `eigvalsh` 直接 abort，改用项目虚拟环境 Python 后完整通过，非断言失败。
