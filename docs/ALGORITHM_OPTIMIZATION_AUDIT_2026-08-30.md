# LD3 算法性能审计：Baseline-Enveloped Gap Coverage（2026-08-30）

## 1. 结论

本轮不是只做代码加速，而是针对 baseline 的 missed-target 尾部失败，引入
**baseline-enveloped gap coverage**：每个 UAV 仅凭实际送达的公开几何与本地 target belief，
同时重建 gap-coverage 候选和 legacy role-aware baseline 候选；在相同安全集合下，只有当
gap 候选的可达性势函数严格优于 baseline 时才执行 gap，平局回退 baseline。

相同 checkpoint、相同 10 个 CRN seeds、K=Q=12、T=150、零训练 episode 的开发诊断表明：

表中使用 2026-08-30 re-audit 后的通信守恒实现：posterior 不再免费搭载在 evidence LLR 中，
Tx mover 也只允许使用 Rx complement（反之亦然）。

| 控制器 | steady P_D | weak3 P_D | worst P_D | QoS pass | worst CVaR20 | 总协议 bit/frame |
|---|---:|---:|---:|---:|---:|---:|
| audited baseline | 0.7067 | 0.3881 | 0.3091 | 0.30 | 0.0026 | 2389.5 |
| audited enveloped gap | **0.8406** | **0.7343** | **0.6749** | **0.60** | **0.2343** | **2338.2** |

相对 current baseline 的 paired mean delta 与 episode-bootstrap 95% CI（100,000 次）：

- steady：+0.1339，95% CI [0.0439, 0.2253]，7/10 seeds 改善；
- weak3：+0.3462，95% CI [0.1256, 0.5734]，8/10 seeds 改善；
- worst：+0.3658，95% CI [0.0980, 0.6370]，8/10 seeds 改善。

这是开发集证据，不是正式盲测认证；工作树为 dirty，10 seeds 也不足以替代项目的正式 gate。
表中的 bit/deadline 是协议优化前用于隔离感知算法效应的控制版本，不能代表当前主配置。随后
§4.3--§4.4 在同源困难种子上完成因果 RR2 与可靠性约束下的无损转码：最终物理协议 P95 为
3.835 ms、deadline violation 为 0，且检测、控制与安全轨迹相对 RR2 服务控制逐字段不变。因此
当前结论是“开发集感知算法优势 + 困难集通信门闭合”；正式 blind 感知优势仍待 §7 的独立门禁。

## 2. 数理依据

在固定 OTFS 支持、固定功率和确定性 bistatic 几何下，target q 的 deflection 系数满足

\[
a_{ijq}\propto
\frac{G_{\rm DD}(\tau_{ijq},\nu_{ijq})}
{R_{iq}^{2}R_{jq}^{2}},\qquad
D_q=\sum_{(i,j)}a_{ijq}p_{iq}.
\]

检测模型

\[
P_{D,q}=Q\!\left(Q^{-1}(P_{FA})-\sqrt{D_q}\right)
\]

对 \(D_q\) 单调递增。因此，在不改变功率和 DD 支持的局部几何比较中，降低最近 Tx/Rx
的 bistatic range product 会提高 target 的可用增益上界。为避免任意加权和掩盖真正漏检，
候选 next-state 使用如下字典序势函数：

\[
S(x)=\left(
N_{\rm uncovered},
\max_q\delta_q,
\sum_q\delta_q,
\max_q\log\Phi_q,
\sum_q\log\Phi_q
\right),
\]

其中

\[
\delta_q=\max\left(
\frac{r_{q,\min}^{\mathrm{Tx}}}{R_{crit}}-1,
\frac{r_{q,\min}^{\mathrm{Rx}}}{R_{crit}}-1,
0\right),\quad
\Phi_q=(H^2+r_{q,\min}^{\mathrm{Tx}\,2})
(H^2+r_{q,\min}^{\mathrm{Rx}\,2}).
\]

前三项优先满足“每个 target 同时有可达 Tx/Rx endpoint”的通信感知覆盖原则；只有覆盖亏损
相同时，后两项才优化 bistatic gain ceiling。使用 log product 等价保持正数乘积排序，同时避免
远距离乘积的数值尺度问题。

pursuit 排序对 mover k 使用
\(R_{kq}^2\min_{j:\operatorname{role}(j)\ne\operatorname{role}(k)}R_{jq}^2\)。
旧实现错误地取 \(\min(R_{q,\min}^{\mathrm{Tx}},R_{q,\min}^{\mathrm{Rx}})\)，可能把 Tx--Tx
或 Rx--Rx 当作互补链路；re-audit 已修正并加入反例锁定。配置的 320 m 仅是几何设计半径，
不是普适的 \(P_D\ge0.6\) 半径，因为功率、DD gate、融合和另一端距离都参与检测。

## 3. Baseline envelope 的保证与边界

令 \(\Pi_{safe}\) 为相同公开视图上的 pairwise CBF-like 安全投影，gap 和 baseline 的
投影后动作分别为 \(u_g,u_b\)。执行规则为

\[
u^*=\arg\min_{u\in\{u_g,u_b\}} S(x+u),
\]

严格平局选择 baseline。对单个 viewer 重建的完整候选，由于集合显式包含 \(u_b\)，逐帧必有

\[
S(x+u^*)\le_{lex}S(x+u_b).
\]

该结论是对“同一 viewer 已送达 public belief 下的一步完整编队几何代理”的确定性保证，
不是当前分布式拼接动作的全局性能定理。实际执行由每个 viewer 仅取自己那一行再拼接；tracking
模式下各节点 belief 可以不同，拼接结果未必等于任何 viewer 评分过的完整候选。独立可组合 barrier
仍保证安全，但性能支配必须依赖闭环配对证据，并显式报告 selection disagreement。

它也不冒充真实 \(P_D\) 的逐帧支配：belief error、未来 target motion、DD support 和后续离散
assignment 都可能使长期真实回报偏离代理。因此仍必须使用 CRN paired evaluation 检验闭环收益；
本轮 10-seed 结果只是对该边界的经验补充。

信息边界方面，算法不读取 simulator target truth，不新增 ground link，也不假设未送达 peer state；
公开视图不完整时原有 fail-closed hold 仍生效。learned payload 仍为 1440 bit/frame，但这不是
完整通信成本：计入 evidence/posterior 后，audited gap 为 2338.2 bit/frame。posterior schedule
现在是独立、稀疏且全量计费的 400-bit 预算；LLR target ID 可以与 posterior 共享，但 4D posterior
状态的 88 bit 不得免费。该轮 H=1 通用头部控制组的总协议 deadline 尚未闭合；第 4.3 节的
因果 RR2 固定模式协议已在同源困难种子审计中闭合该门禁。

## 4. 安全投影解析快速路径

直接包络每 viewer 每帧需要两个安全投影。新增的解析证书检查候选是否同时满足：

1. 每节点速度球和飞行区域约束；
2. 所有 next-state pairwise endpoint separation；
3. 与投影器完全相同的 affine barrier；
4. strict distributed 模式下的 independently-composable endpoint barrier；
5. 若初态在安全集外且启用 recovery，则禁止走快速路径。

候选通过时，投影器的严格凸 least-change 问题以原候选为可行零损失解，因此
\(\Pi_{safe}(u)=u\)。惰性实现只求解未获证候选；两个都未获证时才执行两个投影。
这不是近似筛选，4-seed 和 6-seed 两组都验证了 steady/weak3/worst 及连续最小间距数组与
双投影版本逐元素完全相等。

| 10-seed 平均 | role-correct eager | role-correct lazy | 变化 |
|---|---:|---:|---:|
| safety projections/frame | 23.11 | 19.31 | -16.5% |
| safety solve time/frame | 6.67 ms | 5.87 ms | -12.0% |

惰性版本与 eager 版本的检测数组和连续最小间距逐元素完全相等。audited gap 10 seeds 的
最小连续 UAV 间距为 30.73 m，连续分离违约率为 0。

### 4.1 独立可组合投影的冗余约束消除

对 endpoint-split barrier，安全初态给出每个局部半空间右端 (b/2\le0)；安全集外的渐进
recovery 明确令 (b=0)。飞行区域内的 box bounds 同样包含零动作。因此线性可行集
(C) 是含原点的闭凸集。参考动作 (x) 已预先截断到速度球 (|x_k|\le v_{max}\Delta t)。
令 (p=P_C(x))，欧氏投影的变分不等式对 (z=0\in C) 给出

\[
\langle x-p,0-p\rangle\le0
\quad\Longrightarrow\quad
\|p\|^2\le\langle x,p\rangle\le\|x\|\,\|p\|,
\]

故 (|p\|\le|x\|\le v_{max}\Delta t)。每节点速度球不会改变最优解，可以从 SLSQP 中
严格删除，把带 (K) 个非线性圆盘约束的问题化为线性约束二次投影。代码在运行时逐项验证
(b\le0)、box 包含零、参考动作位于速度球；任一前提不成立即保留完整问题。

审计过程先否决了逐 UAV 的 Python 活动集枚举：虽在 200 个随机反例搜索中与 SLSQP 的最大
动作误差仅 (1.0\times10^{-14})，但 12-UAV 压力微基准从 0.80 ms 退化到 7.16 ms，故未采用。
最终冗余约束消除在 2,000 个随机 12-UAV 场景中动作最大误差为 0，平均投影时间由
0.790 ms 降至 0.694 ms（-12.2%）。

6 个困难种子的端到端 CRN A/B 结果保存在
`.arts/algorithm_audit/projection_control_hard6` 与
`.arts/algorithm_audit/projection_reduced_hard6`：steady/weak3/worst、CVaR、QoS、通信 bit、
连续轨迹最小间距和全部非计时字段完全一致；安全投影时间由 6.205 ms/frame 降至
6.098 ms/frame（-1.73%）。干预调用仅占 7.19%，故 Amdahl 上限使系统收益明显小于压力微基准；
该优化释放少量实时预算，但不宣称提高检测质量或解决 4 ms 协议 deadline。

### 4.2 单位功率物理张量复用

固定几何、角色和 DD support 时，bistatic echo deflection 对 target-wise Tx 功率一阶齐次：

\[
D_{ijq}(p_{iq})=p_{iq}D_{ijq}(1).
\]

解析 power LP 本来已为 gain reconstruction 计算单位功率的 dense
((K,K,Q)) 张量；功率求解后再次执行 delay/Doppler/path-gain/DD 计算是重复工作。候选配置复用
单位功率张量，只对 `d_raw` 与 `d_eff` 广播乘以 ((K,Q)) 功率矩阵。该路径仅在 U2U-only、
`use_swerling=false` 且单位功率张量存在时启用；report-link 或 Swerling 含有状态随机抽样，必须
回退完整物理计算。

单种子和 6 个困难种子的 CRN 输出中，所有非计时字段与未复用版本完全一致。代表性 K=Q=12
物理内核微基准中，已有单位张量后的边际物化由 2.007 ms 降至 1.843 ms（-8.2%）。完整 6-seed
进程 wall time 受 GPU/系统负载波动影响，未观察到可稳定归因的显著下降，因此这里只接受为
零语义内核优化，不声称端到端加速幅度。

### 4.3 因果两队列信标与固定模式头部

通用协议原来每帧让全部 K=12 个节点发送 7 个 8-bit 状态字段，并为每个包收取 64-bit
通用头部，因此结构信标负载为

\[
B_{H=1}=K(64+7\times8)=1440\ \text{bit/frame}.
\]

候选把节点按 (k\bmod H) 分为确定性正交 TDMA 队列，帧 t 只发送
(k\bmod H=t\bmod H) 的队列。固定时隙已经标识 sender，配置 manifest 固定字段顺序、精度和
调度，所以不需要再次发送动态 sender/rate/schema 元数据；仍在物理链路中显式收取 16-bit CRC。
取 H=2 时

\[
B_{H=2}=\frac K2(16+7\times8)=432\ \text{bit/frame},
\]

与审计日志逐 bit 一致。它不是把头部设为零，也没有把 evidence/posterior 藏进免费 side channel；
16-bit 头部、FBL codeword、BLER、时延和能耗全部进入相同通信模型。若改为竞争接入、动态码率或
动态 schema，这个隐式标识前提失效，必须退回通用头部。

时间语义也同时修正：信标在当前帧 UAV 动作执行后，以当前局部 proprioception 编码并立即传输；
接收缓存记录 packet source frame，而非 reception frame。旧实现把上一 submit 帧的状态标成当前帧，
违反因果年龄定义。收到的 peer 位置按已量化速度作常速度外推
(\hat p(t)=\hat p(t-a)+a\Delta t\hat v)，越界时裁剪到飞行域；量化误差用于 nominal endpoint，
额外 stale worst-case bound 仅进入安全证书。没有原始时间戳的实验性 relay 不能伪装成新鲜数据，
现按 sender TTL 的最坏可能年龄标记并 fail closed。

同一代码、checkpoint、6 个困难种子和 fresh-encode 语义下，H=1 通用头部控制组与最终候选为：

| 6-seed 平均 | fresh H=1/64-bit | RR2/16-bit | 变化 |
|---|---:|---:|---:|
| steady P_D | 0.725087 | 0.744376 | +0.019289 |
| weak3 P_D | 0.618932 | 0.630636 | +0.011704 |
| worst P_D | 0.574479 | 0.583252 | +0.008773 |
| worst CVaR20 | 0.166418 | 0.176465 | +0.010046 |
| 总 U2U bit/frame | 2334.83 | 1314.40 | -43.7% |
| 总串行协议均值时延 | 5.194 ms | 3.487 ms | -32.9% |
| 总协议 deadline 违约率 | 1.000 | 0.000 | -1.000 |
| evidence delivery rate | 0.99158 | 0.98872 | -0.00285 |
| 连续最小 UAV 间距 | 28.44 m | 35.42 m | +6.98 m |
| 连续分离违约率 | 0 | 0 | 0 |

mean public max-age 从 0.997 增到 1.974 frame，stale fail-closed 从 0.00676 增到 0.01454；
这些代价被完整报告，而不是以“缓存免费”掩盖。检测四项均提升说明在本组种子上，较低信令拥塞和
可预测 AoI 带来的闭环收益超过了少量 evidence delivery 损失。不过这仍是经验结果，不是
“降低信标频率必然提高 P_D”的普适定理；blind bank 仍需重新认证。

### 4.4 可靠性约束下的无损协议转码与服务包络

RR2/CRC16 解决了 mean deadline，但 6-seed 的保守串行 p95 仍为 5.111 ms。直接把 evidence
通用头压为固定 schema 后，单种子 bit 和 delivery 均改善，worst 却从 0.02720 降到 0.01675。
这是一个关键反例：较短包救回了原本超时的随机 LLR，改变了闭环样本路径；“更多 evidence”只在
统计期望和模型正确时增加信息，不是任意有限样本下的逐路径支配定理。因此最终方案采用
wire codec 与 policy service 分离的兼容包络。

旧 evidence 包对 n 个 evidence entries、m 个 posterior entries 收取

\[
B_{e,old}=64+\lceil\log_2K\rceil+16
 +n(\lceil\log_2Q\rceil+8+2)+88m.
\]

同步 sensing sub-slot 使 observation frame 可由接收时隙唯一推断，source/target ID 仍显式收取。
freshness guard 只允许 AoI (a\in\{0,1,2,3\})，故 2 bit AoI 无损。新 wire 长度为

\[
B_{e,new}=10+\lceil\log_2K\rceil
 +n(\lceil\log_2Q\rceil+8+2)+82m.
\]

CRC 长度不是经验拍定。当前 FBL target BLER 为 (10^{-3})，配置的每包未检出错误预算为
(10^{-6})；若 CRC 错误图样近似均匀，则

\[
P_{undetected}\le P_{BLER}2^{-r},\qquad
r\ge\left\lceil\log_2(10^{-3}/10^{-6})\right\rceil=10.
\]

约 11 packet/frame、150 frame 的 episode union bound 为 (1.65\times10^{-3})。代码由实际
BLER/budget 自动计算最小 CRC；低于下界的配置 fail closed。

coordination 的连续字段仍为 8 bit；target code 只有 Q+1=13 个符号。令 c\in\{0,\ldots,Q\}，
4-bit 均匀码的回译误差满足

\[
\left|Qq_4(c/Q)-c\right|\le\frac{Q}{2(2^4-1)}=0.4<0.5,
\]

故 round 后逐类别精确。为了不只保持类别、还保持旧浮点执行轨迹，decoder 再查表映射到相同的
8-bit legacy representative。每个 RR2 beacon 因而从 16+7*8=72 bit 降到
10+(8+8+8+8+4+8+8)=62 bit。

service envelope 同时锁定五个可能改变策略的通道：

1. freshness scheduler 仍按旧 AoI8 的 400-bit budget 选择相同 posterior；
2. deadline admission 使用 legacy packet length；
3. 同一均匀随机数下，短码选择满足 (P_e^{short}\le P_e^{legacy}) 的最小整数 blocklength；
4. battery 仍保守扣除 legacy RF airtime，不把省电混入感知增益；
5. actor inbox 的 latency/BLER metadata 保持 legacy service clock，而协议统计报告 physical clock。

对固定 SNR 和 blocklength n，normal-approximation error 随 information bits 单调不减；短包信息位
更少，故达到 legacy error 所需的 (n_{short}\le n_{legacy})，物理时延不会增加。准入、随机擦除、
能耗与 actor metadata 均保持旧服务后，检测与控制样本路径保持不变；实际 wire bit/latency 单独下降。

同一 Python/Torch、顺序 MKL、checkpoint 和 6 个困难种子的最终 A/B 为：

| 指标 | RR2/CRC16 control | CRC10 + categorical/AoI codec | 变化 |
|---|---:|---:|---:|
| steady / weak3 / worst | 0.744376 / 0.630636 / 0.583252 | 完全相同 | 0 |
| worst CVaR20 / LCB | 0.176465 / 0.366553 | 完全相同 | 0 |
| coordination bit/frame | 432.00 | 372.00 | -13.9% |
| evidence bit/frame | 882.40 | 508.62 | -42.4% |
| total U2U bit/frame | 1314.40 | 880.62 | -33.0% |
| total mean latency | 3.487 ms | 2.565 ms | -26.5% |
| conservative total p95 | 5.111 ms | 3.835 ms | -25.0% |
| evidence deadline violation | 1.106% | 0% | -1.106 pp |
| coordination / evidence delivery | 0.999781 / 0.988724 | 完全相同 | 0 |
| continuous min separation | 35.420 m | 35.420 m | 0 |

除 wire cost、physical latency 与显式依赖通信代价的 temporal proxy 外，字段级差分没有任何非通信
差异。主配置已推广该方案；旧协议由 `...enveloped_pre_crc10_control.yaml` 显式保留。

### 4.5 私有功率 LP 的确定性分片并行

当前分布式功率层由 K 个节点分别求解自己的私有视图 LP；节点 k 的问题只依赖不可变的
(A_k,b)，不读取另一 worker 的解、通信 RNG 或物理状态。串行映射

\[
(S(A_1,b),\ldots,S(A_K,b))
\]

因此是一个直积映射，分片并行与按节点序号装配可交换。实现使用 spawn 隔离进程、显式回传原
problem index，并在主进程按 canonical viewer order 恢复结果。四个 worker 内部再固定为一个
native numerical thread，防止 process x BLAS 的递归超配。K=12 的 12 个细粒度 IPC 进一步按
round-robin 合并为 4 个确定性 shard；这只减少 pickle/future 调度，不合并任何 LP 约束或私有视图。

worker 数不是按均值拍定。生产 executor 的 seed 451、12 次 LP batch 微基准中，4 workers 的
mean/P95 为 6.341/9.160 ms；6 workers 的 mean/P95 为 6.675/12.785 ms，故尾时延门禁选择 4。
完整 learned-policy 的 3 个困难种子配对结果为：

| 指标 | serial | process-4 + 4 shards | 变化 |
|---|---:|---:|---:|
| replicated power wall/frame | 18.508 ms | 8.091 ms | -56.3% |
| steady / weak3 / worst P_D | 0.601737 / 0.466292 / 0.414673 | 完全相同 | 0 |
| 非计时字段差异数 | 0 | 0 | 0 |

Windows spawn 会导入入口模块。`run_mappo.py` 原先在 module scope 导入 Torch/CUDA、trainer 和完整
environment，即使 worker 只求 LP 也会加载整套训练栈；现已改为 `main()` 内延迟导入。修正后正式
入口测得一次性 4-worker warm-up 为 0.583 s，与 LP-only benchmark 的 0.566 s 接近。按困难集
每帧节省 10.417 ms，break-even 约 56 帧，小于一个 150-frame episode，因此 process-4 已推广到
主配置。分片本身在同一 seed 又把原 12-task process 路径 8.566 ms/frame 降到 8.074 ms/frame
（-5.7%）。一次单-seed evaluation 的记录窗口由 26.25 s 降到 25.71 s；跨独立进程的 wall 数字
仍受系统负载影响，主要结论以内部同帧计时与 exact trace 为准。这些数字是 simulator throughput，
不冒充无线闭环部署时延；真实机群本来就在各节点并行执行私有 LP。

### 4.6 QoS 截断历史储备的局部定理与闭环反例

直接把功率惯性从 0.25 降到 0，在 seed 451 上把 worst 从 0.02720 提到 0.03154，却使 steady
从 0.21287 降到 0.17050；因此“当前 private LP 更优”不等于闭环 Pareto 改善。为避免任意折中，
进一步实现了一个关闭默认的两阶段储备 LP。viewer k 先把自己的上一帧完整局部计划 H_k 投影到
当前公开预算，并构造

\[
r_{kq}=\min\left\{\sum_i a^{(k)}_{iq}H^{(k)}_{iq},
D(P_D=0.60)\right\}.
\]

H_k 本身是该 reserve 的可行见证；随后在 (D_q\ge r_{kq}) 下最大化 (\min_q D_q)。故对 viewer k
求得的完整计划 P_k^*，严格有

\[
D_q^{(k)}(P_k^*)\ge r_{kq},\qquad
\min_qD_q^{(k)}(P_k^*)\ge\min_qD_q^{(k)}(H_k).
\]

但真实分布式执行只拼接第 k 个 viewer 的第 k 行，
(P_{1,1,:}^*,\ldots,P_{K,K,:}^*)，它通常不等于任何一个 P_k^*。主配置的 common-view rate 为 0，
所以不能把上述完整计划定理移植到拼接矩阵。闭环反例验证了这一边界：

| seed 451 | inertia 0.25 | inertia 0 | history reserve + inertia 0 |
|---|---:|---:|---:|
| steady P_D | 0.21287 | 0.17050 | **0.24814** |
| weak3 P_D | 0.03183 | **0.03392** | 0.01502 |
| worst P_D | 0.02720 | **0.03154** | 0.00733 |
| belief RMSE | 40.55 m | 38.18 m | **32.62 m** |

储备方案改善了均值与跟踪，却严重损害尾部，故不推广。代码只保留为 opt-in/common-view 研究能力，
并有“历史计划是可行见证、reserve/最弱目标不退化、并行 reserve 解严格等价”的单元测试。下一次
若要获得全局储备定理，必须先通过有计费的 target-owner contribution 聚合/commit，不能免费读取
集中拼接矩阵。

### 4.7 固定迭代容量对偶的 CUDA Graph 与稀疏 evidence 算术

容量 Sinkhorn 的可行性投影写成

\[
X_{ij}=\sigma\!\left(Z_{ij}+\alpha_i+\beta_j\right),\qquad
\sum_jX_{ij}=c_r,\quad \sum_iX_{ij}=c_c .
\]

原实现用固定 16 次单调二分交替更新每个对偶变量；K=Q=12 的 rollout 每次调用会发射数百个很小的
CUDA kernel，算术量不大，launch latency 才是主项。本轮没有改成 Newton、CPU 求解或低精度近似，
而是捕获**同一组** FP32 二分操作并 replay。CPU offload 曾被实际尝试，但 seed 372 出现约
\(10^{-4}\) 的 CPU/GPU sigmoid/二分差异，故立即否决。最终图路径只在 CUDA FP32、元素数不超过
4096、且外层不处于 graph capture 时自动启用；cache 同时绑定 device、dtype、shape、容量、迭代数
和 stream，带锁、限 16 项，capture 失败则退回原路径。

对偶求解仍位于 `no_grad` 内，最终直通梯度仍为固定对偶下
\(\partial X/\partial Z=\sigma'(Z+\alpha+\beta)\)，因此训练定义没有变化。测试锁定 direct、显式
replay、自动 replay 的输出 `torch.equal`，梯度也 `torch.equal`；第二组输入另行验证 static buffer
确实被刷新。12×12×12 微基准为：首次 capture+执行 161.507 ms，direct 31.270 ms，稳定 replay
5.984 ms（5.23×，-80.9%），约 7 次调用摊销 capture。

normal evidence Monte Carlo 的旧路径虽然每个目标只有一个 owner 和少量送达 peer，却对全部
\((M,K,Q)\) 单元计算并量化 LLR。新路径仍按旧 shape、旧顺序生成完整 H0/H1 随机张量，因而不改变
有限 Monte Carlo 随机流；只 gather 活跃 peer/owner 做算术，再 scatter 到零贡献张量，并继续按旧
receiver axis 求和。这样保留 owner-overlap 优先级、C-order 诊断扁平化和浮点 reduction 顺序。
若每目标平均只有 E 个送达 peer，LLR/量化算术从 \(O(MKQ)\) 降为约 \(O(M(EQ+Q))\)；其他 content
mode 完全走旧分支。K=Q=12、M=2048、每目标约 1 owner+5 peer 的微基准由 14.211 降到
7.867 ms（1.806×，-44.6%），普通、空 peer、owner/peer 重叠、缺 owner、零/微小 deflection
均与旧 full-tensor reference 逐数组相等。

同一 seed-451、150-frame cProfile（process-4 已固定）组合效果为：总时间 41.543→35.617 s
（-14.3%），actor forward 7.151→3.082 s，capacity projection 4.998→2.055 s，evidence MC
2.464→1.086 s。CUDA 异步会移动 cProfile 归因，故总时间和独立同步微基准是主证据。前后 CSV
只有 8 个 wall/solve-time 字段变化，全部非计时字段完全一致。最终 6 个困难 seed 与上一 CRC10
候选相比同样只有 6 个计时字段变化；steady/weak3/worst 仍为
0.744376/0.630636/0.583252，worst CVaR/LCB 仍为 0.176465/0.366553，bit、latency、delivery、
安全距离和完整逐 seed 数组均不变。

### 4.8 先验审计否决的移动候选：几何 sweep 与动态捕获时间

Gauss--Southwell sweep 的单步命题是正确的：在固定公开视图下，接受坐标严格降低
\(\max_q R_{Tx,q}^2R_{Rx,q}^2\)。但它不能推出动态闭环检测率支配。已有 seed-451 反例中，旧静态
baseline 的 steady/weak3/worst 为 0.45836/0.04606/0.02129，独立 sweep 变成
0.09157/0.00242/0.00193；而且在每个 viewer 中枚举它会给当前移动热点增加约
\(O(K^2Q^2)\) 候选工作。因信息增量和复杂度门禁同时失败，本轮没有把它加入 envelope 第三臂。

随后只以 opt-in 形式实现并测试了常速度最短捕获时间。对相对位置 r、目标速度 v、UAV 速度上界 s
和覆盖半径 R，最早可达时刻是下式首个非负根：

\[
(\|v\|^2-s^2)t^2+2(r^Tv-Rs)t+\|r\|^2-R^2=0.
\]

公式、迎向/远离/不可达边界、矩形镜像反射和默认静态分配完全等价均先通过单测；在线分支仅使用
已经计费的本地 \([x,y,v_x,v_y]\)，没有新增通信字段。但决定性 seed 451 得到：steady
0.21287→0.26973，weak3 0.03183→0.02971，worst 0.02720→0.02358，U2U bit/frame
849.31→851.65。即平均段改善而尾部与通信代价同时退化，触发预注册拒绝条件；未做参数搜索、未跑
hard-6，在线代码/配置/测试已经撤回。反例保存在
`.arts/algorithm_audit/dynamic_capture_seed451` 的 manifest/source snapshot 中。

这也限定了可声称的理论范围：上述根只证明单个 endpoint 在名义 CV 模型下进入一个圆盘的最短时间，
而实际 detector tail 同时依赖 Tx/Rx 成对乘积、责任容量、DD 支持、功率分配、belief 反馈和私有视图
逐行拼接；局部捕获时间不能冒充 weak3/worst 的 Pareto 证书。

## 5. 负结果与没有采用的方案

- 无包络 gap 已将 worst 从 0.3091 提到 0.6185，但有 2/10 worst 回退，最大回退 -0.4003；
  不能仅凭均值上线。
- 尝试显式复刻历史 gap v1（always-pursue、关闭 deficit/near-field/tangential）在 6 个困难 seeds
  上得到 steady/weak3/worst=0.7012/0.6031/0.5561，低于 current gap 的
  0.7396/0.6338/0.5825。历史 25-seed 文件来自旧 source/default，不能当作当前代码证据。
- episode-latched tail gate 虽然计算更省，但会用单个初态阈值永久决定整集策略；在 target 动态和
  belief 更新下缺少逐帧代理保证，因此本轮没有用它替代 envelope。
- 初版 freshness 调度让 evidence-selected target 的 posterior “零额外比特搭载”，但 LLR/置信度
  字段不包含该 4D 状态，融合端却读取了它。移除未计费信息后，4-seed gap worst 从 0.835 降到
  0.752，证明旧结果受到信息泄漏混杂；旧的 +0.3775 worst delta 已降级为历史诊断，不再作为结论。
- viewer selection disagreement 平均为 0.432，故全局性能包络没有成立。若未来需要全局定理，必须
  设计有比特、时延和丢包记账的 propose/commit，而不能在模拟器中免费集中裁决。
- H=3 虽可关闭 deadline，但 stale public state 使 steady 明显下降；否决。
- RR2 叠加 300-bit posterior budget 以及 7-field mixed-precision codec 都曾降低 worst tail；虽然省 bit，
  但违反“不能以尾部感知换吞吐”的门禁，未进入主配置。独立 field codec 能力保留用于后续有证明的
  rate-distortion 设计。
- 早期在生成 protocol 前删除未调度 carrier，会让同载波 evidence 以零通信功率发送，delivery 从
  约 98.5% 降到 23.2%；该实现已否决。现在先按实际 evidence 需求保留/求解 RF 功率，仅在进入
  transport 时抑制未调度的结构 payload。
- 直接 fixed-schema evidence 虽把单种子 p95 从 4.905 ms 降到 4.076 ms、delivery 提高 0.769 pp，
  但 worst 下降 0.01045，否决；更好的链路不自动给出有限样本尾部支配。
- target-code 4 bit 虽类别无损，但若不保持 coordination 服务集合，额外到达的公共状态仍改变闭环，
  单种子 worst 下降 0.01057，否决。最终版本必须同时使用 legacy representative 与 service envelope。
- 初拟 CRC8 使用了错误的 (10^{-5}) BLER 假设；当前配置实际为 (10^{-3})，验证器因此拒绝启动。
  最终 CRC10 由真实 BLER 和 (10^{-6}) 未检出预算重新推导。
- 6-process LP 的 mean 比 4-process 慢 5.3%，P95 又高 39.6%；按实时系统的尾门禁否决，未用更多
  worker 数冒充更高并行度。
- 直接取消 25% power inertia 虽提高 seed-451 weak3/worst，却令 steady 下降 0.04237；拒绝。
  QoS 截断 history reserve 又出现 steady/RMSE 改善但 worst 下降 0.01987 的反例；局部完整计划证书
  不能支配私有视图逐行拼接，保持关闭默认。
- 独立 Gauss--Southwell 几何 sweep 有严格单帧势下降，但 seed-451 三档检测率全部大幅回退且计算量
  落在移动热点，未作为第三候选接入。
- CV 最短捕获时间候选提高 seed-451 steady，却降低 weak3/worst 并增加闭环通信 bit；按尾部优先门禁
  否决，试验在线分支已经撤码，只保留 artifact 作为反例。

## 6. 实现与复现

- 算法势函数与候选选择：`uav_isac/physical/movement_potential.py`；
- 安全恒等证书：`uav_isac/coordination/hyperedge.py`；
- 严格 distributed 集成与诊断：`uav_isac/environment/env_core.py`；
- posterior/evidence 全量记账：`uav_isac/physical/evidence.py`；
- 私有功率 LP、历史储备研究门：`uav_isac/coordination/maxmin_power.py`；
- 确定性进程分片与 native-thread 限制：`uav_isac/coordination/parallel_power_executor.py`；
- Windows spawn 延迟导入与正式入口：`scripts/run_mappo.py`；
- 推荐候选配置：
  `config/exp_800_k12q12_distributed_v2_dynamic_u2u_robustbelief_gapcoverage_enveloped_pilot.yaml`；
- 测试：`tests/test_movement_potential.py`、`tests/test_hyperedge_negotiation.py`、
  `tests/test_finite_blocklength_communication.py`、`tests/test_maxmin_power.py`、
  `tests/test_parallel_power_executor.py`、`tests/test_cost_aware_u2u_comm.py`、
  `tests/test_detection_fusion_modes.py`。当前全量回归为 1387 passed（8 个既有 warning）。

评估统一使用 checkpoint
`results/_audit_k10_tail40_safe_robust2/best_restored.pt`，seed=725，episodes=0；困难种子为
451、1992563642、330692561、1224507828、1755461577、1914864024，通用种子为
1019466100、1466409252、272618232、2005825564。结果保存在
`.arts/algorithm_audit/baseline_audited_*`、`.arts/algorithm_audit/gap_envelope_audited_*`；
困难 baseline 与 re-audit 前相同，因为 legacy coupled schedule 已经为每个 posterior 计费。

## 7. 下一阶段门禁

1. 固定已通过 4 ms p95 的 CRC10/service-envelope source snapshot；下一通信门禁是 burst-loss、
   CRC 错误图样偏离均匀假设的 stress，而不是继续无条件压 bit；
2. 若要求全局性能包络，设计有物理传输的 propose/commit；否则明确保留“局部代理 + 经验闭环”定位；
3. 在同一可用依赖运行时中重跑当前 baseline，再在未参与选择的 geometry-stratified bank 上做至少
   25-seed blind paired eval；
4. 报告 episode-cluster bootstrap、Wilson LCB、worst CVaR、逐 seed regressions、总协议 bit/latency、
   reliability、minimum continuous separation 和 wall-clock，而非只报均值；
5. blind worst-delta CI 下界仍为正后，再进行同预算、同 seed、同停止规则的独立训练。
6. 主配置使用 process-4 时，外层 episode worker 必须为 1；同时独立报告一次性 worker warm-up、
   在线 power wall/frame 与总 wall time，不把初始化成本从结果中删除。
