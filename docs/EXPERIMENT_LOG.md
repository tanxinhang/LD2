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

### P1.5：最小 waveform / correlation calibration gate

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
