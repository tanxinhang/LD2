# K16/Q16 盲测与仿真器墙钟优化协议

## 1. 统计单位和正式判据

正式统计单位是一个完整的 seeded episode，而不是帧或目标。即使每个 episode 有 150 帧和 16 个目标，100 个种子仍只有 100 个独立 Bernoulli QoS 样本。QoS 成功定义保持不变：

\[
I_s=\mathbf 1\{\operatorname{steady}_s\geq0.8,
\operatorname{weak3}_s\geq0.7,
\operatorname{worst}_s\geq0.6\}.
\]

若 \(x=\sum_s I_s\)、\(n\) 为 episode 数，正式 QoS 门限使用 95% 单侧 Wilson 下界

\[
L_W=\frac{\hat p+z^2/(2n)-z\sqrt{\hat p(1-\hat p)/n+z^2/(4n^2)}}
{1+z^2/n},
\quad z=\Phi^{-1}(0.95),
\]

并要求 \(L_W\geq0.8\)。例如观测到 80/100 成功时，点估计虽然为 0.8，Wilson 下界仍低于 0.8，不能宣称通过。

连续指标按 100 个 episode 报告均值、中位数、P5、最小值和最大值。不得把 15000 个帧或 1600 个 seed-target 组合当作独立重复。

## 2. 冻结的 K16/Q16 种子库

K8/Q8 银行的几何指纹不能用于 K16/Q16。专用配置为 `config/exp_strict_distributed_k16q16.yaml`，专用银行为 `config/stratified_seeds_1130_k16q16_blind.json`：

- 候选种子范围：10000--11999；
- 几何候选数：1998；
- 仅用初始几何难度分层，不读取算法 QoS 结果；
- easy/medium/hard 候选数：500/998/500；
- test：按 1:2:1 分层冻结 100 个种子；
- 抽样种子：20260828；
- 两个执行器冒烟种子 10213、11751 已登记为 development excluded，并在最终抽样前排除。

当前工作区是 dirty 状态，因此不得立即消费这 100 个种子并称为正式结果。正确顺序是：完成实现和回归、提交冻结、设置单线程数值环境，然后用 `--formal` 一次性执行完整 test split。

## 3. 可断点并行执行

`tools/run_strict_distributed_bank.py` 支持：

- 按冻结顺序读取完整 split；
- 每完成一个 episode 原子写入 checkpoint；
- `--resume` 时核对有效配置哈希和种子顺序；
- 多进程提高统计吞吐量；
- 正式模式拒绝少于 100 个种子、dirty 工作区、非冻结 split 和非单线程数值库。

并行 worker 会竞争 CPU，因此其 episode 墙钟不能用于时延结论。执行器会把多 worker 结果标记为 `timing_claim_eligible=false`，时延认证必须另跑单 worker。

冻结后推荐命令：

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python tools/run_strict_distributed_bank.py \
  --config config/exp_strict_distributed_k16q16.yaml \
  --workers 4 --resume --formal
```

统计吞吐量运行完成后，再以 `--workers 1` 对冻结硬件做时延复验。

## 4. 墙钟剖析

K16/Q16、严格配置、开启 profiler 的 20--25 帧诊断显示，主要累计成本为：

| 阶段 | 典型每帧成本 |
|---|---:|
| 私有超边结构重构 | 约 31 ms（profiler）/ 22 ms（无 profiler） |
| 16 个私有 max-min LP 的串行仿真 | 约 27 ms |
| 两次偏转几何计算 | 约 16 ms |
| 本地观测构造 | 约 16 ms（优化前） |
| 物理通信处理 | 平均约 4.8 ms；载波帧约 13.2 ms |
| owner 后验 CI | 约 6 ms（优化前） |

这些是单进程仿真器成本。部署时 16 个节点的私有 LP 和结构计算并行执行，因此闭环关键路径取最慢节点，而不是 16 次求解之和。

## 5. 已实施的数学等价优化

### 5.1 二元 CI 批处理

控制载波正常投递时，每个 `(receiver,target)` 通常只有一个新责任后验。将 240 个彼此独立的二元 CI 组成批量 4x4 线性代数：

\[
P_f^{-1}=\tfrac12P_l^{-1}+\tfrac12P_r^{-1},\qquad
\hat x_f=P_f\left(\tfrac12P_l^{-1}\hat x_l+
\tfrac12P_r^{-1}\hat x_r\right).
\]

批维之间没有信息交换；它与逐项广义 CI 在数值容差 \(10^{-12}\) 内一致。多于一个新源的罕见情况仍走通用 CI。

### 5.2 严格相同 LP 的公共子表达式消除

若两个节点的局部增益矩阵逐 bit 相同，它们定义同一个确定性 LP，仿真器可复用一次求解结果。只要有任意 bit 不同就分别求解，不以“接近”代替“相同”，因此不引入近似误差，也不把一个节点的私有信息提供给另一个节点。严格私有视图下多数帧不能触发去重，这一保守行为是预期结果。

### 5.3 观测构造向量化与广播对象共享

目标 belief、相对几何、nearest-target、零邻居块和 target-token 元数据改为数组运算，保持原字段顺序。一个物理广播的不可变 owner 后验对象由所有 delivery 记录共享，避免把一次广播错误地模拟成 \(K-1\) 次编码和内存复制。

## 6. 当前墙钟结果

同种子、同物理结果的 50 帧 A/B 中：

- 优化前：平均约 92.3 ms，P95 约 108.9 ms；
- CI 批处理后：平均约 90.8 ms，P95 约 101.5 ms；
- 加入观测向量化后：平均约 89.1 ms，P95 约 101.5 ms。

150 帧单 worker 复验得到总体平均 87.74 ms、总体 P95 99.12 ms、最大值 107.85 ms。载波帧 P95 约 104.05 ms，非载波帧 P95 约 89.99 ms；因此下一优化重点应是载波帧通信对象构造和结构重构，而不是删减感知物理或奖励定义。

墙钟 P95 已进入 100 ms，但裕量较小，不能据此宣称跨硬件实时。闭环控制关键路径仍约 31 ms；仿真器墙钟和部署关键路径继续分别报告。

## 7. 10-seed 开发诊断

在不消费冻结 blind100 的前提下，使用开发种子
`7,14,19,29,43,101,211,307,401,509` 完成 K16/Q16、150 帧单 worker
诊断：

| 指标 | 10-seed 结果 |
|---|---:|
| QoS 通过率 | 10/10（仅点估计，不替代 blind100 Wilson） |
| Steady \(P_D\) 均值 | 0.9984 |
| Weak-3 \(P_D\) 均值 | 0.9912 |
| Worst \(P_D\) 均值 / P5 / 最小 | 0.9839 / 0.9491 / 0.9401 |
| 本地 RMSE 均值 / 最大 | 3.40 m / 3.57 m |
| 运动覆盖率均值 / 最小 | 95.31% / 93.13% |
| 仿真墙钟均值 | 87.51 ms |
| episode 内墙钟 P95：跨 seed 均值 / 最大 | 99.22 ms / 101.29 ms |
| 闭环关键路径 P95：跨 seed 均值 / 最大 | 30.93 ms / 31.38 ms |

结果文件为
`results/_strict_distributed_sweep/k16q16_10seed_t150_owner_bistatic_diagnostic.json`，
其 manifest 明确标记 `formal_result_eligible=false`。

载波周期 4/5 的三种子消融可把通信量从 2581.76 bit/frame 降至
1952.00/1532.16 bit/frame，但 RMSE 分别升至约 3.89/4.26 m，墙钟 P95 也没有
稳定降低。因此主配置保留周期 3；降低通信频率必须等待目标级信息损失证书，而不能只按
平均吞吐量选择。
