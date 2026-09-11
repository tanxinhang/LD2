# UAV-ISAC 实验文档：协议、指标、统计与最新结果

更新时间：2026-09-10
配套总报告：`docs/CURRENT_SYSTEM_MODEL.md`
数据目录：`results/current/`

## 1. 实验目标与证据等级

实验不是单纯比较平均 `P_D`，而是逐层回答五个问题：系统是否按冻结身份执行；硬约束是否在
每条轨迹上成立；弱目标 QoS 是否在独立 episode 上稳定通过；通信和计算代价是否满足部署门槛；
研究支线的改善是否超出随机波动。

证据分为三级：

| 等级 | 用途 | 必要条件 | 可否支持正式结论 |
|---|---|---|---|
| 单元/性质测试 | 验证公式、边界和不变量 | 固定输入、确定预期、失败关闭 | 否 |
| diagnostic / shadow | 验证闭环接线、定位机制 | 明确配置与种子；允许 dirty tree | 否 |
| formal confirmatory | 版本正式性能 | clean commit、冻结协议、100 test seeds、完整 provenance | 是 |

任何 diagnostic 数据即使数值很好，也不能升级为正式证据。正式结果如果绑定旧 commit，也不能
用来描述当前源码。

## 2. 研究问题与可检验假设

### RQ1：严格分布式控制是否真正不读取真值？

- `H1a`：hyperedge、功率和移动决策只依赖本地 belief 与实际送达 U2U 字段。
- 验证：truth permutation/invariance 测试、配置身份检查、缺失 belief 的 fail-closed 测试。
- 失败条件：改变隐藏目标真值但保持可见 belief 不变时，提交结构或动作发生变化。

### RQ2：解析功率块是否满足物理预算并正确改善 bottleneck？

- `H2a`：固定结构 LP 的 primal allocation 非负且每行不超过可用 sensing budget。
- `H2b`：最弱 Deflection 等于 LP objective，dual upper bound 不低于 primal。
- 验证：随机问题性质测试、退化零增益测试、episode 最大 RF 违反量。
- 失败条件：功率违反 `>10^-9 W`、非有限解、错误 owner 或不完整目标覆盖。

### RQ3：移动控制是否在端点和帧内连续轨迹上安全？

- `H3a`：每帧投影动作满足 pairwise affine barrier。
- `H3b`：所有同时执行的直线轨迹最小距离均不低于 20 m。
- 验证：构造碰撞测试、随机投影性质测试、逐 episode endpoint/swept minimum。
- 失败条件：任何 seed、任何 UAV 对低于 20 m；投影失败后仍执行未验证命令。

### RQ4：严格 K16/Q16 主线能否稳定满足 QoS？

- episode success：tail-50 同时满足 `steady>=0.80`、`weak3>=0.70`、`worst>=0.60`。
- 正式假设：100 个独立 test episode 的 success rate 单侧 95% Wilson LCB `>=0.80`。
- 当前状态：尚未在当前 clean commit 上运行，不做结论。

### RQ5：Markov 图候选是否优于等预算 blind 搜索？

- 配对零假设：`E[improvement_graph-improvement_blind]<=0`。
- 筛查标准：固定同一 case、incumbent、候选预算和物理评分，报告 paired delta 与 bootstrap CI。
- 当前结果：95% CI 跨 0，不能拒绝零优势解释。

### RQ6：fixed-lag residual 是否提供 CV 之外的可预测信息？

- 配对零假设：white acceleration 下 residual forecast 的误差不低于 CV forecast。
- 晋级条件：独立 seeds 的平均改善 CI 完全大于 0，并在真实闭环中保留收益。
- 当前结果：white 模型 CI 跨 0，因此 residual forecast 不晋级。

### RQ7：通信、分布式拼接与能量语义是否闭合？

- `H7a`：一次广播对任意接收者数量只产生一个发送 airtime，且 `E_i=P_i*T_tx,i`。
- `H7b`：只有 byte-identical 公共增益模型才允许独立 LP 行拼接；否则 fallback fraction 为 1。
- `H7c`：所有已执行扣能的截断前缺口恒为 0，而不是只观察截断后 battery 非负。
- 验证：异距离多接收者广播、分歧视图、低电量强制透支与分析功率重算测试。

### RQ8：几何 Deflection 抽象何时可代表波形检测？

- 检查 `Delta_tau=1 us`、`Delta_nu=976.5625 Hz` 下目标对的 DD 分离与 ambiguity 主瓣重叠。
- 比较波形 Monte Carlo 与解析 `D→P_D` ROC，报告系数相对误差、证据相关性和分层校准误差。
- 门禁未通过前，不把几何级结果表述为真实 28 GHz 多目标可分辨性能。

## 3. 冻结正式实验设计

### 3.1 系统配置

正式配置为 `config/exp_strict_distributed_k16q16.yaml`，传递继承
`config/exp_strict_distributed_no_truth_pilot.yaml` 和 `config/system_manifest.yaml`。有效配置必须
由 loader 解析后哈希，不能只对最外层 YAML 求哈希。

关键条件：K=16、Q=16、T=150、dt=0.1 s、tail window=50、carrier period=3、episode workers=1。
节点内部本地 LP 可以使用 4 个进程，但不允许再并行 episode，避免嵌套进程争用导致运行时指标
失真。

### 3.2 种子与统计单位

统计单位是完整独立 episode，不是帧、目标或 UAV。正式种子来自
`config/stratified_seeds_1130_k16q16_blind.json` 的 ordered `test` split，共 100 个唯一种子。
以下做法均属于伪重复或数据泄漏：

- 把 150 帧当成 150 个独立样本；
- 把同一 episode 的 16 个目标当成独立性能样本；
- 根据 test seed 表现调参后仍称其为 blind；
- 打乱种子顺序后与旧 manifest 拼接；
- 用 development/quarantined seed 替换失败 seed。

### 3.3 对照公平性

算法对照必须固定：场景初态、目标噪声、通信随机流、OTFS/信道参数、episode horizon、tail window、
种子顺序和候选评估预算。只有待比较模块允许变化。配对比较优先使用 common random numbers，
并同时报告两方法绝对指标和 paired delta。

### 3.4 预热和评价窗

正式 episode 运行 150 帧，只用最后 50 帧形成主检测指标。这允许 tracker、结构 hold、owner
posterior 和移动责任在前 100 帧建立状态。30 帧 smoke 的 tail-20 不具备同样稳态含义，因此
不得与正式阈值直接比较。

## 4. 指标的精确定义

### 4.1 检测指标

对 episode 的 tail 集合 `W`，先计算每个目标的帧均检测概率：

```text
p_bar_q = (1/|W|) * sum_{t in W} P_D,q,t
steady = (1/Q) * sum_q p_bar_q
weak3 = mean(three smallest p_bar_q)
worst = min_q p_bar_q
```

episode QoS success 定义为三个条件的交集：

```text
I_success = 1[steady>=0.80 and weak3>=0.70 and worst>=0.60]
```

不能用 steady 的高值补偿 worst 失败，也不能只报告多帧融合窗口而隐藏单帧基线。

### 4.2 QoS 成功率与 Wilson 下界

若 100 个 episode 中有 `s` 个成功，点估计 `p_hat=s/n`。单侧 95% Wilson 下界采用
`z=Phi^-1(0.95)`：

```text
LCB = [p_hat + z^2/(2n)
       - z*sqrt(p_hat(1-p_hat)/n + z^2/(4n^2))]
      / [1 + z^2/n]
```

正式门槛使用 `LCB>=0.80`，而不是只要求 `p_hat>=0.80`。这把有限样本不确定性显式计入验收。
对 `n=100`，至少需要 `87/100` 成功才达到该门槛；`86/100` 仍不通过。

### 4.3 通信指标

```text
delivery_rate = delivered_links / attempted_links
deadline_violation_rate = deadline_failed_links / attempted_links
bits_per_frame = total physically billed packet bits / frames
```

无尝试链路的帧对 delivery 使用中性值 1，但正式报告还必须同时给出 active sender 和 attempted
links，避免“全静默获得 100% delivery”的误读。正式门槛为平均 delivery `>=0.99`、平均 deadline
violation `<=0.01`。

### 4.4 运行时指标

每个 episode 记录闭环关键路径逐帧时间，再取 P95；正式门槛要求每个 seed 的 P95 均
`<=100 ms`。运行时声明必须来自 episode worker=1 的环境；多 episode 并行结果只可用于吞吐，
不能作为单帧延迟证据。K16 profile 将 power worker soft timeout 降为 55 ms，不再允许单个
子求解器占满整个 100 ms；报告同时区分 simulator wall clock、node-local solver latency 和
`max_i T_i` 的 distributed critical-node latency。

### 4.5 安全指标

Endpoint minimum：所有帧末端、所有 UAV 对距离的全局最小值。

Swept minimum：对每帧同时执行的线段，解析求
`min_{tau∈[0,1]} ||r_0+tau*du||`，再对 UAV 对和帧取全局最小值。两者正式门槛均为 20 m。
只报 endpoint 会漏掉换位穿越，因此不够。

### 4.6 功率与电量

```text
power_violation_i,t = max(P_comm,i,t + sum_q P_sense,iq,t - 1 W, 0)
episode_power_violation = max_i,t power_violation_i,t
episode_min_battery = min_i,t B_i,t
episode_energy_deficit = max_i,t max(E_used_i,t - B_before_i,t, 0)
```

正式门槛为功率违反 `<=10^-9 W`、截断前能量缺口 `=0 J`。`episode_min_battery>=0` 只是状态
完整性诊断，因为状态本身会截断到 0，不能单独证明能量因果性。同时保留感知 PA 上限检查，
不能把联合 RF slack 全部错误注入 sensing PA。

### 4.7 Belief 与结构诊断

诊断字段包括 belief position RMSE、NIS、posterior covariance trace ratio、packet AoI、target
coverage、唯一 owner 完整率、composable certificate complete fraction、primal–dual gap、fallback
比例和 movement intervention/fail-closed 比例。这些不是主 QoS 的替代品，而是解释失败根因。

## 5. 算法验证矩阵

| 模块主张 | 必须观察的证据 | 关键消融/反例 | 当前状态 |
|---|---|---|---|
| 无真值分布式决策 | truth invariance、缺 belief fail closed | 隐藏真值置换 | 单元测试通过 |
| 固定结构功率全局最优 | primal 可行、dual upper、gap | 零增益、极小系数、reserve 不可达 | 单元测试通过 |
| 安全投影有效 | barrier、endpoint、swept | 迎面穿越、初始近距离、陈旧节点 hold | 新门禁已接入 |
| 物理通信付费 | bits、airtime、energy、delivery | 静默、超 deadline、丢包 | 闭环字段已接入 |
| 公共模型后再拼接 LP 行 | exact model digest、fallback fraction | 一节点系数分歧 | 已接入 strict profile |
| 能量因果有效 | pre-clamp deficit | 低电量、候选重算 | 已接入正式门禁 |
| DD 抽象可校准 | ROC、相关性、分辨率分层误差 | 同栅格/邻栅格双目标 | 待波形离线校准 |
| Hyperedge 有效 | 唯一 owner、全覆盖、公共视图一致 | 丢失提议、冲突 owner | 当前解析主线 |
| Markov 候选优于 blind | paired delta CI | 等候选预算 blind | 未证明 |
| Fixed-lag 可预测未来 | forecast delta CI | white vs persistent acceleration | white 模型未证明 |
| Predictive-GNN 可替代求解器 | 物理复核、fallback、闭环消融 | 错排序、OOD、证书失败 | 未接入主线 |

## 6. 2026-09-10 审计发现与修复

### 6.1 发现的问题

1. 旧 formal evidence 绑定历史 commit，不能代表当前代码。
2. 运动安全曾缺少正式逐 seed endpoint/swept 硬门禁。
3. Markov 物理分配只检查部分几何条件，且 benchmark 使用硬编码物理参数。
4. pilot/bank 没有统一输出功率最大违反和最低电量。
5. 若干完整性测试包含恒真断言、重复随机循环或只检查字段存在。
6. `results`、`docs` 和 `tools` 混有大量过时分支与历史输出，难以判断当前证据。
7. `alpha` 的文档曾把阵列增益写入传播系数，容易与 Deflection 外部 `G_tx*G_rx` 形成双计歧义。
8. 旧 battery 门禁检查的是已经截断的状态，逻辑上不能发现透支。
9. 独立节点 LP 在公共系数视图不一致时仍可拼接，缺少共同模型充分条件。

### 6.2 已实施修复

- 在系统 manifest 中强制 movement safety projection、投影动作执行、本地 assignment cache、
  stale fail closed 和 independently composable safety。
- Markov transition 复用同一 pairwise projection；evaluation 同时检查 UAV–target 与 UAV–UAV 距离。
- pilot 新增 endpoint/swept distance、最大 RF violation、最低 battery 字段。
- bank 与 formal gate 新增四项逐 seed 硬门禁。
- Markov benchmark 从当前 L4 profile 读取高度、区域、channel、OTFS、P_FA、RCS 和 movement step。
- 修正无效测试并补充构造碰撞测试。
- 更新因安全语义变化而失效的 characterization fingerprint。
- 核验实现中 `alpha` 仅含传播/RCS，阵列增益只在 Deflection 外乘一次，并纠正文档。
- 通信统计新增发送者唯一 airtime；多接收者测试证明能量不是逐接收者累加。
- strict profile 开启 byte-exact common-model certificate；端点状态与唯一 owner posterior 均从
  同一不可变量化广播包重建，发送者使用零空口开销的同包回环；失败时改用逐行 harmonic
  composable fallback。
- UAV 保存最大截断前能量缺口；pilot、bank 与 formal gate 改用该字段证明能量因果性。

## 7. 验证记录

### 7.1 修改前基线

| 检查 | 结果 | 解释 |
|---|---:|---|
| 全量测试 | 1764 passed, 2 skipped | 修改前代码基线 |
| Architecture V2 | PASS | 依赖边界有效 |
| K16/Q16 identity 字段 | PASS | 配置身份一致 |
| 当前 formal gate | FAIL | 旧证据 stale，不允许跨提交复用 |

### 7.2 修改中定向回归

安全、formal-gate、Markov 和 bank 定向回归：`89 passed, 2 skipped`。

### 7.3 清理后最终回归

```text
1768 passed, 6 skipped, 7 warnings
Architecture V2: PASS
JSON parse and git diff check: PASS
```

相较上一稳定提交新增的 8 个测试覆盖 source-root 映射、唯一层归属、相对导入、依赖环、
atomic epoch 的非载波 loopback/make-before-break，以及二维安全投影的正 AoI margin 与不可行
短路。保留工具的依赖闭包经过 test collection 和全量回归验证。
重新生成的 current data
catalog 只登记 4 个现存结果文件，entrypoint catalog 登记 81 个现存入口；Architecture V2 的
数据完整性与入口新鲜度检查均通过。正式门禁仍按预期失败，因为没有当前提交、clean tree 的
blind-100 证据，不能把本节 smoke 伪装成正式结论。

## 8. 最新诊断实验

### 8.1 K16/Q16 十种子、30 帧独立诊断 smoke

命令：

```powershell
pytrch_ven\Scripts\python.exe -m uav_isac.interfaces.cli pilot -- --config config/exp_strict_distributed_k16q16.yaml --seeds 20001,20002,20003,20004,20005,20006,20007,20008,20009,20010 --frames 30 --tail-window 20 --quiet
```

该运行使用 dirty tree，且帧数/种子数低于正式协议，故 `formal_eligible=false`。

| 指标 | 十种子聚合/最坏值 | 门槛 |
|---|---:|---:|
| QoS success | 10/10 | diagnostic only |
| delivery rate | mean 1.0 | `>=0.99` |
| endpoint / swept / pre-execution minimum | 24.424 m | `>=20 m` |
| max RF violation | 0 W | `<=10^-9 W` |
| minimum battery | 49,669.367 J | 诊断字段 |
| pre-clamp energy deficit | 0 J | `=0 J` |
| common-model certificate | 每种子 0.9667 | 首帧 bootstrap 外全部通过 |
| common-model safe fallback | 每种子 0 | 无协议诱导回退 |
| closed-loop P95 | max 76.132 ms | `<=100 ms` |
| closed-loop frame maximum | 89.520 ms | `<=100 ms`（诊断） |
| online deadline miss | 每种子 0 | `=0` |

聚合检测与通信：

| 指标 | 数值 |
|---|---:|
| steady | 0.961748 |
| weak3 | 0.937515 |
| worst | 0.924460 |
| short-window QoS rate | 1.0（10/10） |
| bits/frame | 2513.067 |
| deadline violation | 0 |
| hyperedge coverage | 0.9667（final=1.0） |

逐帧 trace 证明旧 0.30 模式的直接根因是：非载波帧只刷新了发送者本地 endpoint loopback，peer
没有物理广播可接收，所以端点源帧和 dead-reckoned position 每三帧中有两帧分歧。修复后本地
回环只有在 sender 实际广播时才更新。新结构先进入 pending epoch；owner posterior、endpoint
payload、源帧和完整模型在所有 viewer 间一致后才 make-before-break commit。acquisition 阶段逐帧
发载波并完整计费，首个 epoch 激活后恢复三帧周期。安全投影使用可分二维精确求解，并对速度界
内不可行的 AoI barrier 直接 fail closed，避免 SLSQP 跑满迭代后才返回同一零动作。结果是首帧
以外证书全通过、harmonic fallback 为 0、独立诊断 QoS 为 10/10，且十种子所有帧低于 100 ms；
仍须 clean-tree blind-100 才能形成正式统计结论。

### 8.2 Markov/KNN 物理影子基准

条件：固定种子 20260910，K16/Q16，32 synthetic cases；graph 和 blind 每 case 都评估 32 个
候选；二者共享 incumbent 和精确 fixed-structure LP。

| 指标 | 数值 |
|---|---:|
| graph acceptance rate | 0.90625 |
| graph beats blind rate | 0.4375 |
| graph / tie / loss | 14 / 9 / 9 |
| graph mean improvement | 2.4164e-4 |
| blind mean improvement | 2.3149e-4 |
| paired delta mean | 1.0147e-5 |
| paired delta median | 0 |
| paired delta P10 | -6.2299e-5 |
| paired delta bootstrap 95% CI | [-1.2407e-5, 3.4669e-5] |
| graph/blind mean time | 0.1032 / 0.1015 s |

结论：候选接受机制不会接受比分 incumbent 更差的动作，但相对等预算 blind 的平均优势小、区间
跨 0，且时间没有优势。支线保留用于真实 belief snapshot 的后续 falsification，不晋级主线。

### 8.3 Fixed-lag smoother 影子基准

条件：32 seeds；分别模拟 white acceleration 与具有时间持续性的 acceleration。主问题是历史
平滑和 residual forecast 是否分别有效。

| 模型 | filtered history MSE | smoothed history MSE | ratio | residual forecast delta 95% CI |
|---|---:|---:|---:|---:|
| white | 65.813 | 17.642 | 0.2681 | [-0.01120, 0.01560] |
| persistent | 54.856 | 18.802 | 0.3427 | [0.00864, 0.04231] |

解释：RTS 对历史去噪在两种模型中都有效；只有人为 persistent 模型的 residual forecast CI 大于
0。当前 L4 真实身份是 white acceleration，因此不能用 persistent 合成结果为在线预测背书。

## 9. 当前数据文件

| 文件 | 内容 | 证据等级 |
|---|---|---|
| `results/current/strict_k16q16_smoke.json` | 十种子短闭环、安全/资源/运行时诊断 | diagnostic |
| `results/current/markov_graph_shadow.json` | 32-case Markov 对 blind 配对基准 | shadow |
| `results/current/fixed_lag_shadow.json` | 32-seed smoothing/forecast 机制筛查 | shadow |
| `results/current/README.md` | 数据边界说明 | metadata |

原始 smoke 运行保存在 `artifacts/runs/pilot-8c1fa34d2ac28263fa3d/`。`results/current` 只保存小型
可解释汇总，不替代受管 artifact 的 provenance。

## 10. 正式 blind-100 执行协议

### 10.1 执行前检查

1. 工作树形成 clean commit；记录 commit hash。
2. 运行全量 pytest 和 Architecture V2 检查。
3. strict identity 检查必须无 dirty、配置或 seed-bank mismatch。
4. 确认 100 个 ordered test seeds 与 bank 完全一致，无 quarantined/development seeds。
5. 固定 workers=1、tail=50、carrier period=3 和运行包/线程环境。

### 10.2 执行中原子性

每个 seed 完成后原子写入 episode 结果。正式运行禁止从可变的半成品 episode JSON resume；只有
manifest、commit、source tree、effective config、seed bank、参数和 seed 顺序完全相同的受管
checkpoint 才能恢复。单个 seed 失败不能静默丢弃或重新抽样。

### 10.3 执行后验收

Formal gate 重算 CSV/JSON 中的 episode 数组，不信任预先写好的 summary。它检查：

- 100 条 episode 顺序、唯一性和 QoS boolean；
- 三项检测门槛与 Wilson LCB；
- delivery、deadline、每 seed runtime P95；
- endpoint/swept distance、power violation、pre-clamp energy deficit 与 battery 诊断；
- artifact hash、completion status、formal eligibility、commit 和有效配置；
- registry 的 evidence blob 与当前 release 内容绑定。

只有所有 enforced gate 通过，才允许把结果写成“当前版本正式通过”。

## 11. 下一轮实验计划（按证据依赖排序）

> 2026-09-11 收缩说明：下列 P2--P6 是历史预注册路线，目前全部冻结。C1/C2 后的有效顺序只有
> waveform-to-evidence 外部有效性、actual orthogonal packet realization、以及前两项存活后才允许
> 开始的 quantization/rate allocation。不得继续并行推进结构、SOCP、移动或联合 oracle 来“救”
> correlation-aware 结果。

### P0：语义闭合与架构门禁回归（诊断与全量回归已完成）

- atomic epoch 已把 structure、owner、endpoint state 与 owner posterior 作为同一候选纪元提交；
  依赖不完整时保留旧完整纪元，启动期无旧纪元时 fail-closed。
- architecture checker 已执行 `source_root` 全覆盖、exactly-one ownership、相对导入解析和层依赖
  环检测；未迁移包显式归入 `legacy_runtime`，不再落在门禁之外。
- 十种子 smoke 的共同模型证书均为 `0.9667`，协议 fallback 均为 `0`；首帧为显式 acquisition，
  不是结构/后验错配。全量回归及既有构造安全/陈旧消息测试均通过；正式统计结论仍等待 clean
  commit 的 blind-100。

### P1：当前确定性基线的 clean formal

- 只有 P0 全量回归、Architecture V2、identity 与新门禁全部通过后，才运行 K16/Q16 blind-100。
- 失败按 belief/structure/model-certificate/power/movement/communication/energy/runtime 分层归因，
  不查看 test seed 后调参；修复必须建立新 commit 与 evidence epoch。

### P1.5：最小 waveform / correlation calibration gate（初始诊断已由 C2 完成）

- 在不建设完整 transceiver 的前提下，对 same DD bin、adjacent DD bin、far DD bin 以及
  `rho in {0, 0.1, 0.3, 0.5}` 做小规模 Monte Carlo。
- 比较解析 `a_ijq -> D_q -> P_D`、波形级检测排序与相关证据模型
  `mu_q^T Sigma_q^{-1} mu_q`；报告排序保持率、偏差和覆盖率。
- 若 additive Deflection 明显偏乐观，先修正感知目标与证书；校准未通过时禁止进入稳健功率和
  对偶结构优化，也禁止把当前结果外推为真实 OTFS 波形性能。

### P2：primal-dual structural exchange

- 固定当前 certified LP 为 incumbent，使用 target/RF/airtime/AoI 对偶价格筛选少量
  one-edge / owner-exchange 候选。
- 候选 reduced cost 只负责筛选；每个候选必须重新执行 exact LP 与全部协议、RF、能量和安全
  门禁，只有真实改善才 atomic commit。
- 先在小 K/Q joint oracle 上报告结构最优性 gap，再决定是否进入正式规模。

### P3：belief-robust SOCP

- 用 selection split 标定系数均值/协方差、`epsilon_q` 与 Gaussian/Cantelli 选择。
- 对比 nominal LP、lower-bound LP、SOCP：机会约束违反率、worst `P_D`、保守损失、求解时间。
- 先做小规模精确 Monte Carlo coverage；覆盖不足即停止，不进入 blind-100。

### P4：对偶移动与安全 QP

- 在 P2/P3 固化后，将 target price 传入 bottleneck movement，将 barrier price 纳入 QP；
  执行前验证 swept certificate。
- 压力测试报告安全余量、干预率、hold 率、能耗和检测收益，不以 endpoint 安全替代 swept 安全。

### P5：小规模联合 oracle

- 在 P2 开始时先作为结构算法诊断；此阶段扩展场景覆盖，量化 finite-round exchange + LP/SOCP
  的最优性差距。只把 gap 当诊断，不把小规模最优性外推到 K16/Q16。

### P6：完整波形级 DD 验证

- 在 P1.5 最小校准通过后，再扩展为多目标、clutter、DD sidelobe、残余干扰、振荡器/信道误差
  下的完整波形实验，并复核解析模型适用域。

Markov/KNN、fixed-lag residual、Predictive-GNN 与 temporal-unroll 已降为低优先级 shadow：除非
上述主线出现明确瓶颈且支线先通过独立 falsification，否则不占用正式种子或主报告结论。

### 2026-09-11：A2 OTFS Gram 相关下界接线诊断

- 新 profile：`config/exp_strict_distributed_k16q16_correlation_calibrated.yaml`；只改变多边证据
  covariance 语义，结构、通信、移动和随机种子均继承 K16/Q16 严格基线。
- 软件门禁：相关矩阵 Hermitian/PSD、同模板极限 `c_q=n`、整数 DD-bin 正交极限 `c_q=1`、零增益
  边和重复边处理均有单元测试；全仓回归为 `1773 passed, 6 skipped`。
- seed 7、8 帧、tail 5 的同种子诊断：相关 profile 的 `c_q` 帧均值 `1.2921`、全程最大
  `1.9328`；独立/相关两组的 bits/frame 均为 `2536`、平均 coverage 均为 `0.875`，因而短程
  差异不是免费增加通信或结构覆盖造成的。
- 同一诊断中 steady/weak3/worst `P_D` 从独立模型的 `0.97/0.93/0.92` 降为
  `0.90/0.72/0.65`（四舍五入）。这只说明独立相加在该解析 Gram 模型下明显乐观，不是新算法
  性能提升，也不是统计结论；正式比较仍需 waveform/ROC calibration 与 paired blind gate。
- 两组均未出现 RF 预算违反或通信 deadline 违反；相关组最小 swept UAV 距离为 `94.55 m`。

### 2026-09-11：B1 相关感知对偶交换内核

- 新增 `correlation_aware_dual_pruned_exact_exchange`：复用既有 N5/N6 有界邻域，结构候选逐一
  重算 OTFS Gram 因子；incumbent LP 对偶价格只负责排序，弱对偶上界只负责可证明剪枝，Top-M
  最终由精确 fixed-structure LP 验证。
- 构造门禁将 target 0 的 incumbent 设置为两条完全重合的高增益模板，并提供一组稍低原始增益、
  但相差一个整数 delay bin 的正交替换。内核接受 N6 交换，相关因子由 `[2,1]` 降至 `[1,1]`，
  且校准后的精确 worst Deflection 严格提高；无候选版本返回显式 no-op。
- 本阶段只证明研究内核的数学接线和单步不降，不代表 K16/Q16 在线收益。下一 gate 是随机小规模
  exhaustive N5/N6 oracle：报告 Top-M recall、邻域最优性 gap、LP 次数和 Gram 计算时间；通过后
  才设计 packet-reconstructed atomic commit 接入，禁止直接从集中式张量执行候选。
- 30 个固定随机 K6/Q4 case（每例 32 个 eligible candidates）的首次 falsification 显示，旧的
  incumbent-power replay 排序在 Top-8 只有 `73.3%` exact recall。改用
  `U_lambda(E')-eta*` 主排序、replay sensitivity 次排序后，Top-4/8/12 recall 分别为
  `73.3%/96.7%/100%`；平均邻域最优值比例为 `98.10%/99.86%/100%`，最差比例为
  `82.09%/95.86%/100%`。Top-12 平均验证 11.0 个 LP，而安全剪枝后的 exhaustive 平均为
  22.3 个；因此后续协议原型暂定 Top-12，Top-8 仍不满足零漏检门禁。该结论仅限当前 30 个
  synthetic case，复现脚本为 `tools/audit_correlation_aware_exchange.py`。

### 2026-09-11：B2 reconstruct-then-hash 候选证书

- 新增相关候选规范记录，绑定 generation、完整依赖版本向量、结构/角色/owner、OTFS numerology、
  Gram 因子、校准增益、预算/reserve、精确 LP 功率/Deflection 以及 primal-dual 证书；规范编码使用
  固定大端整数、IEEE-754 binary64 和固定 bit order，不依赖 JSON/repr。
- 所有依赖闭包节点必须从已送达数据独立重建，并同时满足 SHA-256 identity 与规范记录逐字节
  相等。构造测试中，单个节点一个 ULP 的因子差异会拒绝；即使测试注入恒定 256-bit hash 来
  模拟碰撞，不同完整记录仍然拒绝；预算单纯形或 Deflection 恒等式破坏也 fail closed。
- 证书复用既有 prepare/vote/decision 物理传输检查，但把 digest 扩为 256 bit、generation 扩为
  32 bit。相对旧布局每个实际发送包增加 208 bit，闭包大小为 `m` 时额外空口量为
  `208(m+1)` bit；稠密 gain/power 不上空口，而由节点从已计费依赖本地重建。
- 针对相关校准、结构交换、依赖提交和新证书的联合测试为 `23 passed`。本阶段仍未将研究候选
  接入在线 pending/active epoch；因此只证明协议语义和物理提交条件闭合，不声称在线收益。

### 2026-09-11：B3 atomic shadow、物理反例与闭包预筛

- 相关 profile 新增默认关闭的 atomic-epoch shadow；只在实际控制载波帧计算，且没有结构状态写
  句柄。K16/Q16 seed 7 的首次版本误在无载波帧尝试提交，零通信功率导致时延发散并 fail closed；
  调度修正后不再假设免费控制信道。
- 30 个固定 K6/Q4 通信压力 case 中，“感知 Top-12 后做物理 gate”被 falsify：Top-4/8/12 对
  通信可行 exhaustive 的 recall 为 `36.67%/56.67%/80%`，最差最优值比例分别为
  `40.31%/40.31%/61.41%`。原因是高收益、无控制 reserve 的 receiver 候选挤占验证预算。
- 修正采用维度分离的 feasibility-first 逻辑：先按 dependency closure 检查 SNR、deadline 和共享
  RF 单纯形，再对通过者计算 Gram/LP，精确解后复核同一物理约束。相同 cases 平均预筛 22/32
  个候选、post-LP 拒绝 0；Top-4/8/12 recall 为 `93.33%/100%/100%`，Top-12 平均验证 9 个 LP。
  失败/修正 artifacts 分别为 `correlation_exchange_topm_physical.json` 和
  `correlation_exchange_topm_physical_prefilter.json`。
- K16/Q16 seeds `7/19/43`、每 seed 12 帧的 shadow artifact 为
  `pilot-d01e584c83f9401b0f53`。每 seed 尝试 3 次，exact accept 与 would-commit 都为
  `2/3、3/3、3/3`；接受候选副本一致率和 active 不变率均为 1。平均提交量约
  `4350/3587/3450 bit`，协议时延 `1.583/1.426/1.407 ms`。
- 计算门禁未通过：三组 Gram 时间均值约 `124.7/168.0/169.9 ms`，LP 另需
  `15.8/22.2/21.2 ms`，超过 100 ms frame。且 shadow 输入仍标记为
  `centralized_physical_diagnostic`，不能把同源副本一致率写成真实 packet-local 共识。

### 2026-09-11：B4 精确增量 Gram 与两级对偶剪枝

- 对每个 target 的 selected-edge mask 建立帧内精确 Gram-factor cache，并由候选与 incumbent 的
  XOR 只重算 1--2 个改变目标；未变目标继承同一个 binary64 因子，不近似特征值。
- 新增 raw-gain optimistic dual gate。由 `c_q>=1` 得 `g_corr<=g_raw`，所以 raw gain 在同一
  simplex 价格下的对偶值是相关候选的合法上界；上界不超过 incumbent 时可在 Gram 前删除。
  Top-M 内再按下一候选上界 branch-and-bound，达到上界即提前停止。
- 30-case communication-feasible oracle 中，Top-12 仍为 100% recall；branch-and-bound 将平均候选
  LP 从 9.0 降到 6.03，修正版 artifact 为
  `artifacts/diagnostic/correlation_exchange_topm_physical_prefilter_bnb.json`。
- K16/Q16 `7/19/43`、12-frame 重跑 artifact 为 `pilot-02eb8976d69006c2b891`。与 B3 相比，
  exact accept/commit 和 active-unchanged 均不变；Gram 时间降至 `6.71/8.41/8.29 ms`，LP 为
  `14.43/20.36/19.89 ms`，单节点重建 critical path 为 `3.19/3.42/2.97 ms`。短程计算门禁由
  明确失败改善为约 `24--32 ms`，但输入仍是 centralized diagnostic，尚未晋级在线。
- 为避免继续用 test-bank 风格种子调优，另以独立诊断种子 `20001--20005` 各跑 12 帧，artifact
  为 `pilot-97900bc33756661c2f80`。15 次 shadow 尝试全部 exact accept、物理 would-commit，active
  不变率为 1；各 seed 的 Gram/LP/单节点重建/协议时延合计约 `25.8--38.9 ms`。这加强了短程
  运行时可行性证据，但仍不是 packet-local 或正式统计门禁。

### 2026-09-11：C1 科学主线收缩与 Correlation x Budget 门禁

- 新增 `physical/correlated_soft_evidence.py`，用同一个共同协方差高斯模型给出精确线性融合
  `D=delta^T Sigma^-1 delta`、`w=Sigma^-1 delta` 和 Schur-complement conditional gain。随机 SPD
  测试验证边际公式与直接重算一致；独立极限严格退化为 local Deflection。
- 研究主线收缩为 local evidence、task-oriented selection、soft fusion。现有 atomic/provenance/
  replay 继续保留代码和门禁，但降级为 assurance shell；不再新增 lineage 协议。当前 evidence
  路由只有 receiver source-local broadcast，没有 multi-hop fused-evidence forwarding 接口。
- `tools/audit_correlation_budget_mechanism.py` 在 32 个配对 case 上扫描
  `rho={0,0.2,0.5,0.8,0.95}`。原先由预算比例向下取整产生的 `102/204/307/409 bit` 不是
  64-bit evidence 的可实现包长，现改为整数 evidence 网格
  `{64,128,192,256,320,384,448,512}` bit。`rho=0` 时 proposed 与 correlation-unaware 的
  `Delta P_D` 精确为 0；在 `rho=0.8`、192/256 bit 时平均 `Delta P_D` 分别为
  `0.1144/0.1337`，95% paired bootstrap CI 为 `[0.1042,0.1256]` 和
  `[0.1180,0.1505]`；`rho=0.95` 时分别为 `0.1343/0.1657`，CI 为
  `[0.1236,0.1460]` 和 `[0.1464,0.1844]`。
- 在校正后的离散预算网格上达到平均 `P_D>=0.90`，`rho=0.8/0.95` 的 unaware baseline 需要
  320 bit，conditional-Deflection selector 需要 192 bit，合成网格节省 128 bit；原“409 对
  204、节省 205 bit”结论撤回。所有上述点 proposed 与 exhaustive reference 数值一致。但这是
  有意激活冗余的 synthetic mechanism evidence，不能外推
  为真实 OTFS/channel 性能。artifact 为
  `artifacts/diagnostic/correlation_budget_mechanism_v1.json`。

### 2026-09-11：C2 ideal-cyclic OTFS waveform-to-evidence G1--G3

- 新增最小 unitary OTFS block 和双分数 DD circular channel；显式加入 deterministic multipath、
  shared common clutter、receiver-local clutter 与 complex AWGN，再从每个 receiver 自己的 DD
  observation 提取 coherent matched statistic。该模块是 offline falsification，不接在线路径。
- 10,000 samples/hypothesis/split、独立 calibration/validation seeds 下，目标 `P_FA=0.01` 的
  validation PFA 为 `0.0069`；目标幅度 scale `0.4/0.7/1.0/1.3` 的 P_D 单调为
  `0.2009/0.6202/0.9291/0.9952`。
- H0 evidence 中相似视角 pair 的样本相关为 `0.6641`，远视角为 `-0.0131`。equal-covariance
  相对误差 `0.0113`，held-out H0 covariance 相对误差 `0.0251`；15 个 subset 的 calibrated D
  与 validation P_D 的 Spearman `rho=0.9964`。
- target amplitude fluctuation 反例产生 `1.5148` covariance mismatch，成功触发 H0 fixed-PFA
  fallback，证明实现没有无条件套用 pooled covariance。
- 以上 artifact 为 `artifacts/diagnostic/waveform_evidence_closure_v1.json`。它仍依赖理想循环块、
  coherent target phase 和人为 common-clutter loadings，且 `P_FA=0.01` 不是正式 `10^-3` 门禁；
  因此只证明链条内部自洽，不证明真实传播中 Survival Gate A 已通过。

### 2026-09-11：C3 未知相位、`P_FA=10^-3` 与负机制消融

- 移除 receiver-specific common-clutter loading，全部置 1；相似/远 DD template 的 H0 correlation
  仍为 `0.6746/-0.2136`，说明异质性由 waveform template overlap 产生，而非直接注入 pairwise
  rho 或 receiver attenuation。clutter path placement 仍是人工 stress geometry。
- 对随机 target phase 使用 matched-output energy。equal-covariance mismatch 为 `0.9690`，按规则
  切换到 H0 fixed-PFA fallback；validation `P_FA=0.0101`，幅度扫描 P_D 为
  `0.0602/0.2555/0.6364/0.9152`，15-subset Spearman 为 `0.9714`。
- 以 100,000 samples/hypothesis/split 重跑 `P_FA=10^-3`。二项四 sigma 绝对容差为 `0.0003998`；
  coherent/noncoherent validation PFA 为 `0.00094/0.00077`，基准幅度 P_D 为
  `0.80237/0.35787`。artifact 为
  `artifacts/diagnostic/waveform_evidence_closure_pfa1e3_v2.json`。
- 128-bit 2x2 消融给出负结果：aware/unaware selection 均选 `(2,3)`，selection gain `0`；aware
  fusion 仅带来约 `0.0005 P_D`。共同 clutter 已使相关 pair 的单点质量下降，强 quality baseline
  本身会避开冗余节点。故当前只能保留 waveform/statistical closure，correlation-aware selection
  的 Survival Gate 标记为 `NOT_ACTIVATED`，不得通过调场景或弱化 baseline 追求正结果。

### 2026-09-11：C4 双基地几何—OTFS 波形物理桥

- 新增 `physical/bistatic_waveform.py`，严格复用既有双基地 `tau/nu/alpha`，不另造传播模型。
  映射使用 `delay_bin=tau*M*Delta_f`、`doppler_bin=nu*N/Delta_f`，接收路径功率满足双基地
  距离律并对 `p_iq` 线性。
- 针对性测试验证：两段传播距离同时加倍时路径幅度降至约 `1/4`、发射功率加倍时接收功率
  加倍、DD bin 映射与 OTFS 分辨率一致、未调度 edge component 不能进入 `Y_j`。
- 该桥不接受人为 `desired_local_deflection`。现有 C3 受控场景仍保留为 diagnostic，但不能作为
  geometry-grounded 结果；correlation-aware Survival Gate 继续保持 `NOT_ACTIVATED`。
- 多发射机 MAC/正交 resource 尚未冻结，所以本阶段不改变在线 sensing airtime，也不声称多个
  `(i,q)` 波形可免费共享同一 OTFS 块。

### 2026-09-11：C5 解析—波形 detector 归一化与单边距离门禁

- 冻结 detector 保持 `real_gaussian_shift,c_det=1`，不修改历史在线口径。新增显式换算
  `sigma_complex^2=2*P_noise(real-equivalent)`；由此 ideal coherent waveform 与解析
  `D=MN*p*G*|alpha|^2/P_noise` 完全一致。
- 在默认 `M=64,N=16,P_FA=10^-3` 和 20,000 samples/hypothesis 下，对称双基地单腿距离
  `100/300/500/800 m` 的解析 Deflection 为 `2345.37/28.96/3.75/0.573`；波形 Monte Carlo
  最大相对误差为 `1.30%`，解析交叉检查误差低于 `2e-16`。
- 对应单边 `P_D` 为 `1.000/0.989/0.124/0.0098`。因此当前 `P_D_min=0.2` 只在该对称、单边、
  无 clutter 理想模型约 `464 m` 单腿距离内可达；500 m 和 800 m 不满足预期。该负结论不能用
  correlation-aware selection 修饰，必须依靠经资源计费的多边融合、功率/几何改善，或下调任务需求。

### 2026-09-11：C6 单 Tx / passive-Rx 随机几何性能边界

- 使用 500 个未按结果筛选的 `1130 m x 1130 m` 均匀随机几何；每个 case 只有一个 Tx 对一个
  target 发射一次满额 25.1 mW 波形，其余 UAV 被动接收，所以感知功率和 `1.024 ms` 时钟只计
  一次。该实验无 common clutter、blockage、量化或 packet loss，且假设连续 DD 模板已知。
- K=8 时 best-single 的平均 `P_D=0.6595`、达到 `P_D>=0.2` 的几何比例为 `77.0%`；七个 passive
  Rx 在独立热噪声假设下全部融合后为 `0.8340/94.8%`。paired `Delta P_D=0.1745`，95% CI
  `[0.1556,0.1930]`，但仍有 26/500 个几何连理想 all-passive reference 都未达 floor。
- K=16 时 best-single 为 `0.8257/94.0%`，all-passive 为 `0.9721/99.8%`，paired gain `0.1464`，
  95% CI `[0.1255,0.1681]`；仍有 1/500 个理想几何未达 floor。
- 这说明 passive cooperation 在物理量级上有价值，但尚不能宣布系统性能满足预期：结果只有一个
  target 使用整笔 Tx sensing power。Q16 同时服务时受 `sum_q p_iq<=25.1 mW` 和波形资源约束，
  不能把该单目标结果复制 16 次；相关 clutter 和真实 evidence transport 也只会降低该理想上界。

### 2026-09-11：C7 一般 sensing resource 与可分性契约

- 新增离线 `physical/sensing_resource.py`，显式记录 `(K,L,T,F)` 占用、流能量和功率包络；时频
  开销使用流并集，峰值包络使用同时活动流之和，零能量预算是合法边界并会拒绝正能量分配。
- shared-waveform 模式读取实际总 RF 能量与功率包络，拒绝同时提供 additive communication
  ledger；disjoint 模式才允许能量相加，并要求非零通信能量具有逐时隙功率包络。两者均检查共享
  PA 峰值，但仍不是 sample-level PAPR 证书。
- 可分性改用噪声白化、列归一 Gram；数值诊断对正幅度缩放不变，并报告 mutual coherence、最小
  特征值、条件数与秩。模式标签不能替代门槛证书，hypothesis ID 不能来自 target truth。
- 新增测试覆盖正交、近冗余、缩放不变、资源并集、共享波形能量、非法身份和预算超限。定向结果
  为 `27 passed`，Architecture V2 通过。该阶段没有接入在线 scheduler/LP，也没有产生新的性能
  数值；800 m/Q16 可达性仍未被证明，Survival Gate 保持 `NOT_ACTIVATED`。

### 2026-09-11：C8 target-separable LP specialization gate

- 新增组合证书，将资源守恒与签名可分性绑定在同一次调用中；只有目标可分、hypothesis/LP 列
  一一对应、共享 PA 不超限且 Gram 门槛通过时，才输出 `p_iq=E_iq/T_epoch`。
- `tools/audit_lp_specialization_contract.py` 的正交案例通过并得到 `[[0.2,0.2]] W`；common-probe
  案例因不存在逐目标映射被拒绝；近重复案例以 `mu=0.999950`、`lambda_min=4.9996e-5` 被拒绝；
  通信功率使总峰值由 `0.4 W` 增至 `0.6 W`、超过 `0.5 W` PA 上限，也被拒绝。
- 审计已纳入 pytest；相关定向回归为 `31 passed`。这组结果只验证反例门禁，不比较两种波形的
  检测性能或通信速率，也不改变在线 LP、800 m 结论或 Survival Gate 状态。

### C19 无ACK新似然消息与统一历史融合

- 用户明确排除ACK与重传。新增physical/markov_evidence.py日志域前向递推，
  本地、远端新观测似然在同一隐状态下融合，历史通过上一状态质量传递。没有把发送端历史后验
  作为新独立证据。跨节点噪声条件独立为明确前提；相关clutter需联合似然替换。
- audit_noack_markov_fusion.py采用三UAV和五个真实位置假设（静止目标、身份转移矩阵），
  各格点独立调用双基地桥，不把五个假设作为同时照射的五份功率。Rx匹配输出保留完整Gram
  协方差，随机目标相位用Bessel函数精确边缘化。发送5个float32新似然，元数据/包头后258 bit。
- 每块仅一次广播，无ACK、无重传，失败消息不进入接收端。三种split各100000样本、各八个
  独立链路轨迹。8块48.192 ms，通信2064 bit；本地/逐块独立融合/统一历史融合P_D为
  0.60003/0.79791/0.80083，P_FA为0.00088/0.00111/0.00102。统一融合条件Hoeffding下界
  0.79560（十二点校正），未通过严格P_D>0.8；更不能从均值轻微越界宣称成功。
- 统一历史相对逐块独立的配对增益0.00292，按八个链路轨迹bootstrap的95%区间
  [-0.0038505,0.01000025]跨0，未证明稳定增益。H1位置负对数评分由本地1.59033改善至1.04631，
  支持远端信息改善目标状态推断。1块融合P_FA=0.00161亦提示阈值跨链路泛化尚需更充分审计。
- 此轮只证明通信辅助推断路径可运行；感知功率0.15 W、通信0.1 W固定且属于不同节点，尚无
  后验驱动的感知动作优化、感知辅助U2U物理链路或运动目标实证。5项递推/缺包/单次发送测试通过。

### C17 IM-v1 信息状态 Markov 框架版本

- 版本定义固定在 docs/INFORMATION_MARKOV_V1.md：终端检测概率最大化，0.8仅作验收，资源作为
  可行域，不使用人工加权成本奖励。说明多目标 min E 与 E min 不可混淆。
- physical/information_markov.py 给出有限状态、有限动作参考Bellman递推；对每状态每节点核验
  sensing+communication<=1 W，拒绝非法转移概率及无可行延续状态。没有宣称有限信息状态充分性。
- 两步合成反例中，当前收益相同但参考求解器选择先acquire后send；该例仅验证递推，不作为
  800 m性能。四项参考求解/历史/实际链路集成测试通过。
- 既有底座重跑：固定通信功率0/1/10/100 mW，平均P_D为0.61216/0.81946/0.87798/0.88773；
  最差采样链路理论P_D仍为0.60899/0.67688/0.78672/0.78672。该数据属于固定功率基线，
  不属于新策略。IM-v1物理动作转移及分布式历史策略尚未接线，不能认定版本性能已达标。

### C16 实际模拟链路到 owner 历史的接线

- `tools/audit_link_history_detection.py` 使用三 UAV 几何：Tx=(-800,0,0)、owner=(800,0,0)、
  remote Rx=(0,800,0)，target=(0,0,0)。感知桥给出两条真实几何路径的统计强度；Tx 每块 0.15 W，
  owner 保存本地观测，remote Rx 在独立通信子时隙发送 32-bit 能量统计量（float32 量化）。
- 复用 route_structured_evidence/InterUAVCommunicationModel，启用有限码长、4 dB 相关阴影
  （相关系数 0.7）、1 MHz 带宽、16 dBi 两端天线、5 ms 截止时间；remote 到 owner 距离约
  1131.37 m。每块1.024 ms感知后预留完整5 ms通信，8块合计48.192 ms，失败包不抹除时间。
- 新增 DeliveredEnergyHistory，仅接纳实际送达或 owner 本地的观测，以(frame,source,target)
  去重并按观测时间清除过期记录，冲突重复/未来记录拒绝。路由传元数据，测试传递成功解码后的
  本地 payload；融合器没有使用未送达远端统计量，也没有反馈链路。
- 八个独立链路种子，每种功率 H0/H1 各合计100000次，通信功率0/0.001/0.01/0.1 W的 P_D
  为0.61216/0.81946/0.87798/0.88773。对固定链路银行的条件独立非同分布样本使用 Hoeffding
  下界（四点Bonferroni），为0.60748/0.81478/0.87330/0.88305。不能用池化二项CP忽略分层。
- 有通信时每窗口 attempted bits=920；所有节点功率分别为[0.15,0,P_comm]且均<=1 W，不能将
  不同节点预算相加冒充共享约束。这里只检验角色正确的固定功率链，不是同节点自主角色/功率选择。
- 尾部没有通过：1 mW最差链路轨迹的理论P_D约0.67688；10/100 mW最差约0.78672。平均超过
  0.8不能宣称所有链路环境达标。仍缺随机几何、感知相关杂波、物理同步误差和Q16。
- 20项历史/链路相关测试通过，零通信功率会静默而非产生无限延迟包。

### C15 更新为严格 P_D>0.8 与独立检测验证

- 用户新要求取代 C12--C14 的旧 P_D=0.2 门槛。新功率/积累脚本以严格 0.8 判断，旧章节仅保留
  历史含义。800 m 单腿、P_FA=0.001 条件不变；等于 0.8 的数值解明确标记为边界而非通过。
- 达到 0.8 边界所需感知功率：单 Rx/单块 0.861905 W；单 Rx/8 块 0.188624 W；15 Rx/8 块
  全送达 0.035993 W；送达率 0.9 时 0.038056 W。
- 在观察 Monte Carlo 前固定设计 P_D=0.85，对应功率为 0.940446/0.202806/0.038135/0.040341 W。
  各点 H0/H1 独立 100000 样本；经验 P_D 为 0.85098/0.85178/0.85110/0.85138。
  单侧 Clopper-Pearson 下界按四点 Bonferroni（每点 alpha=0.0125）校正，为
  0.848438/0.849244/0.848559/0.848841，均严格高于 0.8。
- 实测虚警率为 0.00109/0.00129/0.00113/0.00113。理论阈值按已知独立噪声精确设定；上述数据
  不能证明实际 P_FA 的置信上界不超过 0.001，更不涵盖协方差失配或多目标搜索虚警。
- 保留旧 25.1 mW 感知上限时，理论首次超过 0.8 需要单 Rx 230 块、15 Rx 全送达 16 块、
  15 Rx/90% 送达 17 块。分别计实际包头和串行空口时间；这些最小块数点尚未做检测率置信认证。
- 最新脚本仍为物理可达性与成本诊断，不是自主策略性能。旧固定功率的 0.42/0.37 结果均不满足
  新要求，且不能把放宽感知上限产生的增益归于智能优化。

### C14 功率上限与决策算法收益的分离审计

- `tools/audit_800m_power_feasibility.py` 重新从双基地桥计算 800 m 单腿系数
  D/W=22.8127096，采用未知相位非相干检测、P_FA=0.001，数值求解 P_D=0.2 所需功率。
  单 Rx/单块需 0.323346 W，单 Rx/8 块需 0.084637 W；15 Rx/8 块全送达需 0.018862 W，
  送达率 0.9 时需 0.019858 W。方程残差均小于 1e-9。
- 单 Rx/单块感知功率 0.5 W 时 P_D=0.422074，1 W 时为 0.880777。放宽旧 25.1 mW 感知上限
  本身就能产生很大提升，不能归因于历史决策或滚动优化。代码明确标记 autonomous_policy_validated=false。
- 旧 params.py 将 P_sense_max 明确标注为硬件/波形上限。用户要求的仅共享 1 W 条件应作为新的
  硬件假设进行对照，不能静默替换历史结果。C13 的感知 1 W 合成动作不属于旧上限下可执行动作。
- 本轮沿用单目标、独立热噪声、等质量 Rx，送达率未与通信功率闭环；不同节点不能共用功率预算。
  尚未完成真实 inbox/历史驱动的自主策略和固定策略公平比较，不能据此宣布自主分配性能达标。
- 五项功率门槛、丢包及历史机制测试通过。下一项必需工作是建立角色正确的 Tx/Rx/融合端闭环，
  绑定通信功率到真实送达模型，再评价历史预测；避免在送达概率手填的代理模型上声称算法提升。

### C13 用户确认的 1 W 上限与历史条件信息决策

- 用户确认“1”为每节点 1 W 总功率上限，并要求节点结合历史判断信息价值。当前旧配置另有
  25.1 mW sensing cap；新增离线 `physical/history_power_choice.py` 默认只施加共享 1 W 约束，
  尚未改变旧在线配置。每次候选动作显式给出 sensing/communication 功率及预测可用概率。
- 历史上下文使用同一目的节点、同一检测窗口的本地证据或实际收到的 delivery confirmation；
  未知远端历史不得作为已知信息。按 observation frame 剔除过期记录，重复 ID 不增加证据。
- 一份候选新证据的收益为 `p_available * Delta D0(candidate | history)`，复用 Schur complement。
  该期望只适用于一次新增证据、可用事件不依赖未实现的证据数值，且所有动作使用同一冻结模型。
  预测 mean/covariance 与功率、链路的关系由调用方提供，本模块尚未实现物理预测器或联合调度。
- 机制反例：无历史时选择 1 W sensing；已有强相关历史（rho=0.99）后，新 sensing 条件增益
  从 4 降至约 0.0201，改选 0.4 W report（预测送达概率 0.9、条件期望增益 2.025）。当该 report
  已确认送达时不重复计收益；全部证据已有时 idle。1.1 W 候选始终被拒绝。
- 相关定向测试 15 passed，覆盖历史改变选择、重复送达、过期记录、未知上下文及每节点预算。
  这些是合成决策机制结果，不构成实际自主分配的 800 m 检测率认证。下一步需要接入功率相关的
  感知/传输预测与实际 inbox，再做多帧闭环对照。

### C12 800 m 时间与跨节点能量积累诊断

- `tools/audit_800m_spatiotemporal_accumulation.py` 复用双基地桥，两条传播腿各 800 m，单 Tx
  25.1 mW、每块 1.024 ms、D_edge=0.572599。各块各 Rx 噪声独立，未知相位能量统计量之和
  满足 `T|n,H0~chi2(2n)`、`T|n,H1~ncx2(2n,n*D_edge)`；按实际送达的观测数 n 调整阈值。
  本地观测始终保留，远端独立丢包；重复统计量不作为额外样本。本实验直接抽取充分统计量分布，
  不声称逐波形 Monte Carlo 或实际 packet router 已接线。
- 每个远端每块一次发送，沿用 EvidencePacketLayout，32-bit 数值对应总包长 120 bit。声明串行
  有效传输率 1 Mbps、通信发射功率 0.1 W，全部 attempted packets 均计能耗和时长，无重传。
  收集感知观测与发送串行计费；这是传输压力假设，并非物理链路速率/可靠性认证。
- 固定每块功率时，8 块的理论 P_D：本地 1 Rx 为 0.013332；15 Rx 全送达为 0.420829，送达率
  0.9 时为 0.370913。每组 H0/H1 各 100000 样本，经验 P_D 为 0.01342/0.42148/0.37091，
  P_FA 为 0.00084/0.00085/0.00085，均在预设采样容差内。
- 15 Rx/8 块的感知能量 0.2056192 mJ、attempted bits=13440、通信发射能量 1.344 mJ，串行
  总时长 21.632 ms（不含处理/接收电路能耗）。该单目标条件下超过 P_D=0.2；固定总感知能量
  为单块 0.0257024 mJ 时，15 Rx/8 块/全送达的 P_D 仅 0.003761，不能宣称延长窗口必然改善。
- 15 个 Rx 采用相同链路质量复制，尚未构建对应 K16 几何；不含跨节点/时间相关杂波，已计包长
  但未模拟数值量化。Q16 调度、通信实际交互以及轨迹/同步状态信息的增益尚未验证，因此不能
  宣布系统 800 m 已达标。本轮两项计数/丢包边界测试通过。

### C11 非白杂波与未知相位检测闭环

- `tools/audit_clutter_projected_detection.py` 从总接收观测生成检测输出。固定单位杂波签名 c，
  `C=I+rho*c*c^H`，白化矩阵 `W=I+((1+rho)^(-1/2)-1)*c*c^H`；令
  `v=P_perp(W*t)*W*s`，滤波器 `w=W*v/||v||` 满足 `w^H*C*w=1`、`w^H*t=0`。
  统计量 `|w^H*y|^2` 在 H0 为单位均值指数分布，H1 下两倍统计量为自由度 2、非中心参数
  `2*A^2*||v||^2` 的非中心卡方。每次 H1 trial 随机化目标相位，A=3 为归一化幅度。
- calibration H0、validation H0/H1 各 100000 样本且使用独立随机种子。阈值仅由 calibration H0
  的 0.999 分位数决定；六个可辨识案例 validation PFA 为 0.00084--0.00133，通过包含校准和验证
  两侧采样误差的五标准误诊断容差；这不是虚警率严格不超过 0.001 的置信认证。
- 两轴间距 0.1/0.5/1 bin 时，无杂波 P_D 为 0.00654/0.48220/0.72328；rho=4 时为
  0.00477/0.14027/0.13749。全部与冻结阈值下非中心卡方理论预测在预设采样容差内一致。
  两个零间距案例返回 UNIDENTIFIABLE，P_D/P_FA 为空，不把不可辨识性写成检测成功。
- 定向数学测试与已有 DD 扫描测试均通过（2 passed），架构检查通过。C 是已知真协方差，DD
  模板与杂波位置也是受控输入；尚未验证有限训练样本、协方差失配、搜索多重比较或 800 m 链路。

### C10 分数 DD 双目标可辨识性与 C9 证据范围纠正

- C9 的 2 倍由预设能量分配公式直接得到，未从总接收观测独立验证；其资源表的时长、带宽也未与
  OTFS numerology 绑定。撤回“公平波形级对照已完成”的表述，C9 仅为条件性的代数记账例子。
- `tools/audit_fractional_dd_separation.py` 使用实际 cyclic OTFS 路径响应构造总观测
  `y=a*s+b*t+n`。白噪声下，将未知复幅度 nuisance `b` 所在子空间投影掉：
  `P_perp=I-t*t^H`（单位 t），目标保留能量 `||P_perp*s||^2=1-|t^H*s|^2`。
  这是可辨识性与非中心参数的能量因子，不能直接套用 coherent Gaussian ROC 充当未知相位 P_D。
- M16/N8、Delta_f=15625 Hz 对应 B=250000 Hz、块长 0.512 ms。固定两个绝对分数 DD 坐标，
  同时扫描两轴间距；QPSK 在 0/0.1/0.5/1 bin 时保留比例为 0/0.046659/0.694888/0.977338。
  DD impulse 对应为 0/0.063384/0.833691/1。签名完全重合时，单目标幅度无法与未知另一目标分开。
- 16 个案例均验证投影恒等式和总观测中的 nuisance 消除，误差低于 1e-12；相关测试 17 passed。
  未产生 800 m 性能或通信 QoS 认证。模板来源、非白 clutter 和独立 H0 阈值校准仍需后续闭合。

### 2026-09-11：C9 common-probe 能量均分反例

- `tools/audit_common_probe_energy_counterexample.py` 构造相同物理预算：通信/感知时分占用
  `0.25/0.75` epoch，两种模式均使用 `0.3 J` sensing、`0.1 J` communication 和 `0.4 W` 峰值；
  规定通信接收 SNR=10 时，两者理想 AWGN Shannon 上界均为 `864857.9 bit`，高于相同
  `500000 bit` payload。该项只说明 capacity 未排除可行性，不认证有限码长 QoS。
- ideal-cyclic DD impulse 的两个整数 DD 响应 mutual coherence 为 `8.31e-33`。common probe 一次
  广播照射两个格点，解析 Deflection 为 `[0.6,0.294]`；无波束增益的两路完全隔离定向流平分同一
  能量后为 `[0.3,0.147]`。因此当前受控案例中的比值为 2。
- grill：这个 2 倍不能写成 common probe 的普遍优势，但它足以否证“无条件把 common waveform
  energy 均分成逐目标 `p_iq`”。审计保持离线，且明确缺少 fractional leakage、clutter、beam
  gain/sidelobe、PAPR/RF impairment 和 finite-blocklength delivery/deadline。

## 12. 结果解释与禁止表述

允许表述：

- “十种子独立诊断 smoke 中未观察到安全或 RF 预算违反。”
- “Markov graph 的 paired CI 跨 0，当前没有稳定优于 blind 的证据。”
- “Fixed-lag 改善历史估计，但 white acceleration 下未证明预测收益。”

禁止表述：

- “当前 K16/Q16 已正式通过”——当前没有 clean-commit blind-100。
- “算法全局最优”——只有固定结构功率子问题全局最优。
- “系统完全分布式”——仿真推进和统计汇总仍集中完成。
- “保证永不碰撞”——保证限于离散控制模型、投影前提和当前连续线段验证。
- “Markov/GNN 显著提升性能”——现有数据不支持。

## 13. 复现命令

```powershell
# 软件回归
pytrch_ven\Scripts\python.exe -m pytest -q
pytrch_ven\Scripts\python.exe tools/check_architecture_v2.py

# 系统身份（dirty tree 会按设计失败）
pytrch_ven\Scripts\python.exe tools/check_system_identity.py --manifest config/exp_strict_distributed_k16q16.yaml --strict

# 十种子 independent diagnostic smoke
pytrch_ven\Scripts\python.exe -m uav_isac.interfaces.cli pilot -- --config config/exp_strict_distributed_k16q16.yaml --seeds 20001,20002,20003,20004,20005,20006,20007,20008,20009,20010 --frames 30 --tail-window 20 --quiet

# 研究支线影子基准
pytrch_ven\Scripts\python.exe tools/benchmark_markov_graph_assignment.py --cases 32 --cardinality 16 --neighbors 3 --blind-candidates 32 --output results/current/markov_graph_shadow.json
pytrch_ven\Scripts\python.exe tools/benchmark_fixed_lag_smoother.py --seed-count 32 --output results/current/fixed_lag_shadow.json

# 当前 formal gate；在 blind-100 刷新前应拒绝 stale evidence
pytrch_ven\Scripts\python.exe tools/assert_formal_gates.py
```

正式 bank 必须从项目受管 CLI/执行器启动；不得通过直接编辑 registry、summary 或 completion 文件
绕过 provenance。
