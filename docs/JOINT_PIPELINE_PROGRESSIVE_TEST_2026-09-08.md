# 联合管线逐步测试（2026-09-08）

## 口径

- 配置：`config/exp_strict_distributed_k16q16.yaml`。
- 端到端诊断：10个固定开发seed，每个30帧，tail window 10。
- 当前GNN仅shadow，不控制生产动作。
- 30 ms仅作工程参照，不作为优化约束或SLA。
- cProfile仅用于热点归因，不用于物理性能比较。

## 阶段结果

### S1 正确性回归：通过

prediction、refresh gate、teacher exporter、性能等价、hyperedge acceleration service和
deflection acceleration service共22项测试全部通过（1.16 s）。

### S2 时序GNN影子验证：部分通过

模型为18维输入、12,420参数，8个seed训练、2个seed完全留出。驻留内存CPU单线程基准：
特征5.18 ms、GNN均值2.22 ms（p95 3.18 ms）、条件解码1.36 ms，共约8.77 ms。

留出集owner@2为100%，edge recall为96.69%。非刷新转换18/18完整覆盖；4个刷新边界的
edge recall为81.77%，完整覆盖0/4。因此模型可以异步排序候选，但不能取代解析补洞或exact
fallback，`production_eligible=false`保持正确。

### S3 当前生产管线端到端：计算未通过，三帧物理性能通过

结果文件：`results/joint_pipeline_current_10x30_20260908.json`。

| 指标 | 结果 | 判断 |
|---|---:|---|
| 全仿真step均值 | 50.53 ms | 未到20--30 ms工程参照 |
| seed内step P95均值 | 67.62 ms | 未通过 |
| 部署关键路径估计均值 | 12.69 ms | 低于参照，但不等于完整仿真 |
| 部署关键路径估计P95均值 | 27.48 ms | 低于参照 |
| 单帧worst均值 | 0.8441 | 均值较好 |
| 单帧worst_min | 0.3308 | 未通过0.75期望 |
| 三帧worst_min | 0.8502 | 通过 |
| 三帧QoS | 100% | 通过 |
| worst闭环收敛率 | 10% | 未通过 |
| belief position RMSE均值 | 9.50 m | 记录值，尚无冻结门限 |

三帧结果只能证明累计检测有效，不能替代后续超边、LP、belief联合收敛验证。

### S4 证书与最优性：未通过

certificate complete fraction均值为90%，但certificate joint approximation ratio lower均值
仅`9.37e-4`（范围`6.32e-5`--`5.45e-3`）。当前全局上界过松，无法据此证明当前可行解
接近物理优化问题的最优值。下一轮应优先收紧可分目标/owner条件上界，使`U-L`成为有判别力
的收敛量，而不是先训练GNN近似这个松证书。

### S5 功率并行A/B：保留并行，但继续压IPC

3 seed × 30帧serial/process-4配对轨迹全部逐元素一致。process将power batch wall由约28 ms
降至约9.2 ms，整帧平均加速1.43倍。因此不能直接撤掉进程池；后续目标是将约9.2 ms向
worker critical path约2.76 ms靠近，优先采用常驻共享数组、批量任务描述和结果原位写回。

## cProfile归因（单seed、30帧）

以下是累计时间折算的近似每帧值，函数存在包含关系，禁止求和：

| 环节 | 约耗时/帧 |
|---|---:|
| max-min power入口 | 9.14 ms |
| 通信处理 | 8.10 ms |
| 超边协商与重建 | 7.72 ms |
| 严格观测构建 | 7.57 ms |
| Dense→entries（两次） | 4.39 ms |
| bistatic coefficient重建 | 2.59 ms |
| movement matching | 2.33 ms |
| dense deflection→per-watt | 1.86 ms |

## 总体判断与下一轮顺序

当前系统不是“整体满足预期”：三帧worst/QoS达标，部署关键路径估计较低；但完整仿真耗时、
单帧最差性能、闭环收敛率、刷新边界预测完整性和认证最优性均未达标。

下一轮按依赖顺序执行：

1. 收紧物理优化上界，形成可用的`U-L`收敛曲线；
2. 贯通同一edge COO至selected physics、per-watt gain和LP，删除Dense→entries；
3. 在hold帧实测selected-only物理，不改变协议索引；
4. 保留process-4并把gain/active-set放入常驻共享数组，减少IPC；
5. GNN仅学习每项计算动作对认证间隙的下降量，刷新边界不足由解析上界补洞；
6. 重跑10×30，要求物理轨迹等价，并分别报告全仿真和部署关键路径，禁止混用。

## 证书收紧迭代 S6

新增确定性的one-hot simplex dual证书。每行上传`b_k*a_upper[k,q]`，责任节点计算
`min_q sum_k b_k*a_upper[k,q]`；它是合法simplex dual点族的最小值，不改变执行功率。
聚合器新增强制不变量：若可行下界`L`在数值容差外高于上界`U`，立即抛错，禁止用ratio
clip掩盖错误。

10 seed × 30帧真实通信A/B结果：

| 指标 | uniform upper | targetwise upper |
|---|---:|---:|
| global upper均值 | 8714.68 | 84.11 |
| joint approximation ratio lower均值 | 0.000937 | 0.032629 |
| certificate payload/sender | 304 bit | 544 bit |
| 总通信量 | 2412.8 bit/frame | 3564.8 bit/frame |
| 物理指标最大绝对差 | — | 0 |

所有seed的upper均不变松、ratio均不下降，delivery和coverage保持1。step均值观测为
50.53→48.14 ms，但证书路径增加了计算与通信，该下降视为运行噪声，不作为加速收益。
由于通信增加47.7%且认证比绝对值仍仅3.26%，该配置保留为实验配置，不启用canonical。

进一步实现了全simplex最优dual LP并用随机exact LP验证强对偶。120帧离线结果表明它相对
one-hot upper只再下降约25.8%，认证比均值0.00867→0.01277（teacher口径）；因此上界的
主要松弛不再来自price选择，不把额外LP接入在线路径。

物理区间审计显示：选中边全部可见且nominal为正，但lower为正的比例仅81.82%；
`a-/a`中位数0.271，`a+/a`中位数2.97、P95 81.72。新增DD区间最大值上界并通过随机
uncertainty-ball包含测试；它使one-hot upper均值仅124.87→119.66，收益有限，证明多数
DD不确定区间确实覆盖峰值。下一步应校准时序状态协方差并验证集合coverage，禁止直接缩小
sigma或不确定半径。

本阶段最终相关回归：103 passed。新增实验配置为
`config/exp_strict_distributed_k16q16_targetwise_certificate.yaml`，默认功能保持关闭。
