# 计算路径“近似—AI超分—证书修复”逐步审计（2026-08-28）

## 结论

当前不应把 AI 直接接管严格分布式感知功率分配。轻量等变网络的单节点
CPU 推理足够快，但独立测试种子上的最差目标质量和原始—对偶证书均未达到
部署门槛；加入缩减 LP 后能恢复质量，却因 SciPy/HiGHS 固定启动开销而比原
LP 更慢。

本轮可安全保留的改进有两项：

1. 精确 LP 使用缓存的稠密布局、HiGHS 双单纯形并关闭小问题预求解；
2. 精确 LP 返回真实的目标对偶价格，为后续本地事件触发提供可验证上界。

跨帧复用在单一公共视图上有效，但在严格分布式系统中，各 UAV 私有 LP 的
局部近优证书不能约束“每个 UAV 只执行自己一行”之后的全局拼接性能。因此
复用开关默认仍为 0，不作为当前主算法。

更重要的时延审计结论是：单进程模拟器串行执行 K 个私有 LP，其总时间是
仿真吞吐成本，不是实际分布式部署的决策关键路径。K=Q=16 的测量中，16 个
LP 串行总计约 23.51 ms，而最慢单节点约 1.74 ms。以后必须并列报告：

- simulator wall time：单进程仿真吞吐；
- node critical path：最慢 UAV 的本地计算；
- radio/protocol latency：真实传输与共识时延。

不能再把三者直接相加或把串行模拟器总时间称为 UAV 在线时延。

## 数学边界

固定结构功率问题为

\[
t^*(A,b)=\max_{p\ge 0}\min_q\sum_i a_{iq}p_{iq},\qquad
\sum_qp_{iq}\le b_i.
\]

其对偶为

\[
t^*(A,b)=\min_{\lambda\in\Delta_Q}
\sum_i b_i\max_q\lambda_q a_{iq}.
\]

对任意可行旧功率 \(p^h\) 和任意单纯形价格 \(\lambda^h\)，在新实例上有

\[
L=\min_q\sum_i a'_{iq}p^h_{iq}\le t^*(A',b')\le
U=\sum_i b'_i\max_q\lambda^h_q a'_{iq}.
\]

所以公共视图下，当 \((U-L)/U\le\varepsilon\) 时复用具有严格相对近优保证。
但是严格分布式执行只拼接每个私有解的一行。不同节点的 \(L_k,U_k\) 对应
不同私有问题，不能推出拼接矩阵的全局 \(L,U\)。本轮大规模测试已经观察到
这一理论边界，不能将局部证书错误扩展为全局证书。

## 分阶段结果

### Stage 1：无 AI 粗解基线

在记录的 K=Q=8 物理轨迹上，解析 harmonic safe start 和 1--64 轮分布式
赢家通吃对偶更新均无法同时满足质量、时间和证书门槛。低轮方法很快，但
p05 效用比约 0.36；64 轮无量化对照已经慢于精确 LP，p05 仍约 0.39。

结果：
`results/_strict_distributed_sweep/coarse_to_fine_power_stage1_smoke_k8q8.json`
和
`results/_strict_distributed_sweep/coarse_to_fine_power_stage1_smoke_unquantized_k8q8.json`。

### Stage 2：等变 AI 直接细化

模型是 2 层二部图消息传递网络，输出逐 UAV softmax 功率份额和目标价格
simplex；训练/测试按场景种子隔离。600 个轨迹样本、300 轮训练后：

- 单节点 CPU 推理 p50 0.299 ms，约为 LP 的 4.23 倍纯推理加速；
- RF 可行率 100%；
- 效用比 p05 0.767、p50 0.890；
- 0.5% 原始—对偶证书通过率 3.33%；
- 严格回退率 96.67%，含回退期望时间 1.675 ms，高于直接 LP。

结论：直接 AI 超分不部署。

结果：
`results/_strict_distributed_sweep/coarse_to_fine_power_stage2_ai_k8q8_600.json`。

### Stage 3：AI 活跃集 + 缩减精确 LP

AI 只筛选变量，并用确定性目标覆盖修复；缩减 LP 的对偶价格在完整增益矩阵
上重新评估，因此证书仍针对完整问题。

- Top-4 保留 50.23% 变量：效用比 p05 0.999872，RF 可行率 100%，证书通过
  94.67%；
- Top-6 保留 75.02% 变量：效用和证书均达到 100%；
- 但 Top-4 的 AI+LP p50 为 1.523 ms，原 LP 为 1.277 ms；Top-6 更慢。

结论：AI 能识别活跃集，但当前小 LP 的固定求解器开销压过降维收益。

结果：
`results/_strict_distributed_sweep/coarse_to_fine_power_stage3_active_set_k8q8_300.json`。

### Stage 4：精确 LP 底层优化

128 个真实轨迹实例上，缓存稠密约束布局 + `highs-ds` + 关闭 presolve：

- 原实现 p50 1.225 ms；
- 候选 p50 1.029 ms，约 19% 改善；
- 128/128 目标值匹配，RF 可行率 100%。

融合后再次测量的生产函数 p50 约 1.071 ms。相关 68 项功率/陈旧性测试通过。

结果：
`results/_strict_distributed_sweep/power_lp_backend_stage4_k8q8_128.json`
和
`results/_strict_distributed_sweep/power_lp_backend_stage4_after_integration_k8q8_128.json`。

### Stage 5：对偶认证跨帧复用

公共视图影子审计，20 场景、2950 帧：

- 2% 阈值：跳过 29.76% LP，估算 1.40 倍加速，效用比 p05 0.9870；
- 5% 阈值：跳过 55.90% LP，估算 2.21 倍加速，效用比 p05 0.9643；
- 两档均为对偶上界违规 0、相对保证违规 0、RF 可行率 100%。

严格私有视图 K=Q=16、3 种子、每种子 150 帧：

- 2% 阈值：复用 3.76%，整帧均值 69.97 -> 69.42 ms；
- 5% 阈值：复用 21.76%，整帧均值 69.97 -> 65.71 ms；
- 5% 阈值的平均 worst 由 0.6862 降到 0.6536，种子 43 由 0.5149
  降到 0.4569。

结论：公共视图证书成立；私有行拼接没有同等全局保证，默认关闭。

结果：
`results/_strict_distributed_sweep/certified_power_reuse_stage5_k8q8_20seed.json`
和
`results/_strict_distributed_sweep/private_reuse_k16q16_3seed_t150_tol005.json`。

## 当前收敛后的下一步

1. 不继续扩大直接功率网络；保留模型和工具作为负结果/活跃集证据。
2. 不启用私有 LP 跨帧复用，除非增加可计费的目标级聚合或证明行拼接界。
3. 将在线指标拆成节点关键路径、协议时延、仿真吞吐三部分。
4. 下一轮只剖析单 UAV 的约 1.7 ms 功率路径，以及真正属于控制器的本地观测/
   消息处理；物理真值生成和奖励计算不计入部署决策时延。
5. 若仍使用 AI，优先做事件/活跃集排序或暖启动，并要求硬投影和完整证书；
   不允许用未认证的神经输出直接替代通信—感知功率可行域。

## 验证

完整测试：1237 passed，8 条既有 Transformer nested-tensor 警告，无失败。
