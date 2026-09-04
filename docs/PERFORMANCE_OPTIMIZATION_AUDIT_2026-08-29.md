# LD3 性能深审计与零语义优化记录（2026-08-29）

## 1. 结论

本轮只接受满足“物理模型不变、随机流不变、通信记账不变、闭环输出可对拍”的优化。
已经落地四类改动：rollout 推理图裁剪、critic 共享编码复用、2×2 协方差闭式谱计算、
belief/deflection 稠密张量传递。K=16 的四进程解析功率路径保留，并通过三 seed 串并行逐帧对拍。

本轮不把训练环境的 `env.step` 直接改成线程并行：该段以 Python 控制流为主，GIL 下没有可靠收益；
也不立即改成进程池，因为环境含 RNG、GRU 隐状态、消息队列、belief cache 和 episode 生命周期，
没有完整状态隔离协议时会破坏马尔可夫轨迹与可复现性。

## 2. 数学与物理边界

### 2.1 2×2 对称协方差的闭式最大特征值

对实对称矩阵

\[
S=\begin{bmatrix}a&b\\b&d\end{bmatrix},\qquad
\lambda_{\max}(S)=\frac{a+d+\sqrt{(a-d)^2+4b^2}}{2}.
\]

位置和速度不确定度只读取 2×2 协方差块，所以通用 `eigvalsh` 属于过度求解。实现用
`hypot(a-d, 2b)` 保持数值稳定，并在批维上一次计算。它不改变协方差、置信半径或超边安全规则。

### 2.2 固定几何下 deflection 对功率严格线性

确定性 U2U、固定 \((\tau,\nu,\alpha)\) 时，

\[
d^{\rm raw}_{ijq}(P_{iq})=C\alpha_{ijq}^{2}P_{iq},\qquad
d^{\rm eff}_{ijq}=d^{\rm raw}_{ijq}I_{\rm support}|A(\tau,\nu)|^2.
\]

因此有严格关系

\[
d^{\rm eff}_{ijq}(P_{iq})=P_{iq}d^{\rm eff}_{ijq}(1).
\]

系统先保存单位功率的 \((K,K,Q)\) 张量，解析功率 LP 完成后再缩放并物化旧接口对象。
接收机到融合中心的报告可靠度和 Swerling 散射包含有序随机抽样，明确禁止走该快捷路径，仍执行原标量路径。

LP 在退化最优面附近可能放大 1 ULP 的系数变化。为保持闭环身份，稠密路径没有直接把
`d_eff(1)` 当作 LP 系数，而是按旧实现相同的 \(\alpha^2\times C\times |A|^2\) 标量运算次序重建；
实测逐元素 bit-exact。这是“张量存储 + 浮点身份保持”的关键约束。

### 2.3 critic 单编码多头读出

令共享 critic 表征为 \(h=f_\theta(s)\)，标量价值、credit 和逐目标价值分别是
\(g_v(h),g_c(h),g_q(h)\)。原 rollout 重复计算了同一个 \(f_\theta(s)\)。新接口只计算一次
\(h\)，再执行三个确定性读出；代数输出完全相同。`collect_rollout` 整体运行在
`torch.inference_mode()` 下，训练更新和梯度路径不受影响。

### 2.4 稠密 belief 观测

`BeliefState` 列表原本只是 `mean/cov_diag/aoi` 三个数组的逐元素包装。观测构造器现在可直接读取
`(K,Q,4)` mean、`(K,Q,4)` covariance diagonal 和 `(K,Q)` AoI；归一化常数预缓存。
对象接口仍保留，并由逐数组精确相等测试锁定。

## 3. 代码改动

- `uav_isac/utils/math_utils.py`：批量 2×2 最大特征值闭式。
- `uav_isac/environment/env_core.py`：超边协方差批处理、稠密 belief 输入、单位功率 deflection 延迟物化、
  LP 系数旧运算序重建。
- `uav_isac/environment/observation.py`：稠密 belief 接口和归一化常量缓存。
- `uav_isac/physical/deflection.py`：`DenseDeflection` 与确定性 U2U tensor kernel；随机通信/散射 fail-closed。
- `uav_isac/agents/networks.py`：critic 单编码多辅助头接口。
- `uav_isac/agents/trainer.py`：rollout 全域 inference mode，删除重复 critic trunk 前向。
- `tests/test_performance_equivalence.py`、`tests/test_deflection_vectorized_equiv.py`：数值、顺序、接口和线性缩放门禁。

## 4. 性能证据

硬件：Ryzen 7 7800X3D（8C/16T），Python 3.14.6，NumPy 2.5，SciPy 1.18，
PyTorch 2.12.1+cu130。K=Q=16。

| 证据 | 优化前 | 优化后 | 解释 |
|---|---:|---:|---|
| 30 帧 cProfile 总时间 | 5.160 s | 4.552 s | -11.8% |
| `EnvironmentCore.step` 累计 | 3.741 s | 3.226 s | -13.8% |
| Python 调用数 | 4,683,865 | 3,994,581 | -14.7% |
| 每瓦系数重建累计 | 0.187 s | 0.057 s | -69.5% |
| 小矩阵 `eigvalsh` 调用/30 帧 | 15,378 | 18 | 仅保留非目标热点调用 |

四进程功率求解的同版本三 seed、每 seed 50 帧配对结果：串行平均 step 94.37 ms，
并行 78.32 ms，平均加速 1.205×；并行 P95 三 seed 平均约 90.92 ms。3/3 seed 无 fallback，
检测、belief RMSE、bits、delivery、sensing power 和 UAV position 的逐帧最大绝对误差均为 0。

注意：cProfile 会改变真实墙钟截止策略，profiled 与 unprofiled 的最差检测概率不可直接横向比较。
性能数字属于开发诊断，不是正式认证结果；当前工作树 dirty，`formal_result_eligible=false`。

### 4.1 K12 私有 LP 分片与 spawn 启动路径复审（2026-08-31）

生产 executor 不再为 12 个小 LP 分别做 IPC，而是按 problem index round-robin 组成 4 个固定 shard，
worker 返回 index 后由主进程恢复 canonical viewer order。每个 worker 通过 `threadpoolctl` 固定一个
native thread。相同 3 个困难 seed 中，replicated-power wall 从 18.508 降到 8.091 ms/frame
（-56.3%），全部非计时字段完全一致。4-worker 的独立 batch mean/P95 为 6.341/9.160 ms；6-worker
为 6.675/12.785 ms，故拒绝 6-worker。

Windows spawn 还会导入入口模块。`scripts/run_mappo.py` 的 Torch/CUDA/trainer/environment import 已从
module scope 移到 `main()`，LP worker 不再加载训练栈。正式入口报告的 4-worker warm-up 为 0.583 s，
LP-only 工具为 0.566 s；按每帧节省 10.417 ms，约 56 帧即可摊销。主配置因此启用 process-4；外层
episode 并行仍必须关闭，避免嵌套进程超配。

### 4.2 小张量 CUDA Graph 与稀疏 evidence 算术（2026-08-31）

K=Q=12 的 capacity Sinkhorn 算术规模很小，但固定 16×outer-iteration 的二分会发射数百个 CUDA
kernel。新路径 capture 并 replay 完全相同的 FP32 操作序列，不改变迭代数、阈值、对偶 stop-gradient
或最终 sigmoid。仅对 CUDA FP32、numel≤4096 的小张量启用；大 PPO minibatch、嵌套 capture、cache
满或 runtime capture 失败均退回原实现。cache 按 device/dtype/shape/capacity/iterations/stream
隔离并加锁。direct/replay/auto 的输出和 direct gradient 均 `torch.equal`。

同步微基准：capture+first 161.507 ms，direct 31.270 ms，replay 5.984 ms（5.23×），约 7 次摊销。
一次 150-frame 生产 profile 中 capacity 累计 4.998→2.055 s，actor forward 7.151→3.082 s。
尝试 CPU dual offload 曾产生约 1e-4 的跨设备数值差，因此未采用。

normal evidence 路径保留完整 H0/H1 RNG shape 和生成顺序，只对送达 peer 与 owner gather 计算 LLR/
量化，再 scatter 后沿原 receiver 轴求和。故随机流、owner precedence、诊断顺序和 reduction order
不变；其他 payload mode 不走新分支。K=Q=12、draws=2048 的 full-reference/稀疏路径为
14.211/7.867 ms（1.806×），覆盖空 mask、重叠 mask、缺 owner 与零 deflection 的测试均逐数组相等。

两项组合的同 seed、同 process-4 cProfile 总时间 41.543→35.617 s（-14.3%），evidence MC
2.464→1.086 s。前后评估 CSV 的全部非计时字段相同；6-seed hard trace 的完整检测、通信、感知和
安全结果也相同。因此这两项属于零语义吞吐优化，不作为算法增益计入 baseline gap。

## 5. 验证门禁

- 定向物理/性能/strict 协议回归：74 passed。
- 当时全量测试：1363 passed；2026-08-31 协议/分片/储备审计后为 1382 passed；本轮 CUDA Graph/
  稀疏 evidence 审计后为 1387 passed，8 个既有 PyTorch nested-tensor warning，59.10 s。
- 2×2 闭式与 `eigvalsh`：随机批次误差门 `rtol=atol=2e-15`。
- 稠密/对象 belief 观测：`array_equal`。
- critic 旧多次调用/新单编码：PyTorch `assert_close`，普通与等变 critic、credit 开关全组合。
- deflection：标量物理参考、直接功率张量、单位功率缩放、canonical i→j→q 顺序全部对拍。
- 串/并行功率执行：三 seed 六类闭环 trace 全部 bit-exact。

## 6. 下一阶段：训练环境进程并行协议

训练 rollout 当前仍在 Python 中逐环境调用 `env.step`。下一步只在满足以下门禁后启用进程并行：

1. 每个 worker 独占完整环境、NumPy RNG、action RNG、belief/message cache 和 episode seed；不共享可变状态。
2. 主进程只批量执行 actor/critic 张量前向；动作按固定 env index 分发，结果按同一 index 收集，禁止按完成顺序拼接。
3. reset seed 定义为纯函数 `seed(base, rollout, env_id, episode_id)`，worker 重启后可重放。
4. `state_dict`/异常恢复覆盖 GRU hidden state、observation、done/truncation 和 pending communication。
5. 用串行与 2/4 worker 在相同动作 trace 下逐步对拍 observation/reward/done/info；通过后才测吞吐。
6. 只有当包含 IPC 的 rollout steps/s 提升至少 20%、P95 不恶化且全部 trace 等价时默认开启。

GPU/CPU 张量策略保持简单：小 MLP 批量送 GPU 或大批 CPU GEMM；环境中的小数组不搬 GPU，避免 PCIe/launch
开销大于计算。线程数按进程数分配，防止 4 个 worker 各自再开 16 个 BLAS 线程造成过度订阅。

## 7. 后续优先级

1. 把 belief-ranking 的 `DeflectionEntry` 消费者逐层改为 tensor view，目标是移除剩余每帧一次
   `K(K-1)Q` 对象物化；必须先给 InnerSolver/hyperedge lookup 增加等价接口。
2. 实施上述训练环境进程协议，并单独测 inference batching 与 environment parallelism 的贡献。
3. 对 `_resolve_hyperedge_negotiation` 做候选维批处理；任何改变 tie-break 顺序的方案一律不合入。
4. 结果目录约 13 GB，另行做内容寻址 checkpoint 去重；这是 I/O/存储治理，不与算法性能修改混批。
