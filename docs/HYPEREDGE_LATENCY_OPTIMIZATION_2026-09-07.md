# 超边与物理系数时延优化记录（2026-09-07）

## 验收门

每一步优化必须同时满足：

1. 同seed的steady、weak3、worst、QoS、通信和覆盖指标不变；
2. W=3的`worst_min`不得低于优化前的0.844917；
3. certificate robust路径不得放宽；
4. 并行功率回退率保持为0；
5. 分别报告物理内核、控制关键路径和仿真整帧耗时，不能混为一个指标。

## O1：跳过名义系数路径的无效robust-DD计算

`reconstruct_bistatic_coefficient_from_public_state(...,
robust_dd_uncertainty=False)`原先仍执行位置/速度不确定性传播、两次sinc下界计算，随后将结果
丢弃并恢复名义ambiguity。本次将分支提前；路径损耗的量化保守项保持不变，robust
certificate路径也保持原实现。

K16/Q16固定输入、1000次调用的微基准：

| 路径 | 优化前 | 优化后 | 变化 |
|---|---:|---:|---:|
| nominal reconstruction | 412.9 us | 228.6 us | -44.7% |
| robust reconstruction | 412.1 us | 393.3 us | 基本不变 |

相同K16盲库前10 seed、30 frame端到端对照：

| 指标 | 优化前 | 优化后 |
|---|---:|---:|
| 性能字段最大绝对误差 | — | 0 |
| 控制关键路径均值 | 27.60 ms | 25.37 ms |
| 仿真整帧均值 | 75.76 ms | 75.40 ms |
| W=3 worst_min | 0.844917 | 0.844917 |
| W=3 worst通过率 | 100% | 100% |
| 功率并行回退率 | 0% | 0% |

微内核收益没有等比例转化为整帧收益，证明下一瓶颈不再只是名义DD重建。下一步应拆分
`完整拓扑解析`和`hold帧已选边增益刷新`：结构保持帧跳过全图提议/共识，只计算最多48条
已选边的nominal/lower/upper增益。该步骤必须继续执行上述同seed零漂移验收。

## O2：结构保持帧跳过无效提议和共识

当离散L2结构处于5帧hold时，新生成的本地proposal和mutual consensus不能改变执行边集；
但旧实现仍逐viewer执行排序、提议和集合共识。本次保持每帧完整连续增益、certificate和
分布式功率输入不变，只跳过这些无效控制流，并保留既有consensus streak供下一结构更新帧
继续使用。

相同K16盲库前10 seed、30 frame相对O1的端到端结果：

| 指标 | O1 | O2 |
|---|---:|---:|
| 性能字段最大绝对误差 | — | 0 |
| 控制关键路径均值 | 25.37 ms | 22.58 ms |
| 仿真整帧均值 | 75.40 ms | 71.49 ms |
| 额外整帧加速 | — | 1.055x |
| W=3 worst_min | 0.844917 | 0.844917 |
| W=3 worst通过率 | 100% | 100% |
| 功率并行回退率 | 0% | 0% |

O1+O2把控制关键路径由原始27.60 ms降到22.58 ms，性能未变化。O3将只改变hold帧的
系数表示：对最多48条`selected_edges`直接计算连续nominal/lower/upper增益，不再构造
3840条完整图；完整图仍在真正的拓扑更新帧计算。

## O3：hold帧只重建已选边增益

新增selected-edge物理内核，在一次端点几何计算中同时生成名义系数、certificate下界和
certificate上界。只有`hold_active`且近场残差关闭时使用该路径；真正的拓扑更新帧以及
其他配置继续使用原完整图实现。稀疏结果已逐边与三个原始dense函数比较。

K16/Q16、48条已选边、1000次调用微基准：三次完整图重建为673.3 us，稀疏联合重建为
274.2 us，降低59.3%。相同K16盲库前10 seed、30 frame相对O2的端到端结果：

| 指标 | O2 | O3 |
|---|---:|---:|
| 性能字段最大绝对误差 | — | 1.47e-14 |
| 控制关键路径均值 | 22.58 ms | 16.21 ms |
| 仿真整帧均值 | 71.49 ms | 64.02 ms |
| 额外整帧加速 | — | 1.117x |
| W=3 worst_min | 0.844917 | 0.844917 |
| W=3 worst通过率 | 100% | 100% |
| 功率并行回退率 | 0% | 0% |

相对O1之前的基线，O1--O3累计把控制关键路径27.60 ms降至16.21 ms（-41.3%），整帧
75.76 ms降至64.02 ms（1.183x）。1.47e-14属于浮点运算排列产生的机器精度差异，没有
改变任何QoS判定或W=3性能。

## O4：可替换的加速服务包

新增 `uav_isac.acceleration.hyperedge` 服务包，将环境与物理重建内核解耦。环境只依赖
`HyperedgeAccelerationService` 契约，包含 `reconstruct_dense`、`reconstruct_upper` 和
`reconstruct_selected` 三条路径；默认 `numpy` 后端直接绑定已审计内核，生产路径不做
逐调用计时或计数。需要 profiling 时可显式构造
`NumpyHyperedgeAccelerationService(collect_timing=True)`。

替换方式有两种：进程内调用
`register_hyperedge_acceleration_backend("name", factory)`，或在配置中填写
`package.module:factory`。因此更换 Numba/CuPy/C++ 后端不需要修改 `EnvironmentCore`。
新后端必须保持 dense 形状、selected 顺序、可见性 fail-closed、nominal/lower/upper
保守语义、确定性和 public-state-only 输入；上线前用 `numpy` 作为 oracle 逐项对比并跑
同一 QoS/worst 回归。

封装回归（K16/Q16，盲库前10 seed，30 frame，tail-window=20）如下：

| 指标 | O3 直接内核 | O4 numpy 服务 |
|---|---:|---:|
| 性能字段最大绝对误差 | — | 0 |
| 控制关键路径均值 | 16.21 ms | 17.14 ms |
| 仿真整帧均值 | 64.02 ms | 66.77 ms |
| W=3 worst_min | 0.844917 | 0.844917 |
| W=3 QoS 通过率 | 100% | 100% |
| 功率并行回退率 | 0% | 0% |

O4 结果文件：`results/k16_hyperedge_service_numpy_blind10x30_diagnostic.json`。
17.14/66.77 ms 是当前机器一次回归的观测值，主要受进程并行调度影响；数值和 QoS
完全逐 seed 对齐，服务默认直通路径不会改变算法行为。

## O5：DenseDeflection 到对象的向量化物化

profile 表明 `DenseDeflection.to_entries()` 每帧调用两次，是剩余最大的 Python 对象热点：
旧实现通过 `i -> j -> q` 三重 Python 遍历，对每条边执行7次数组标量索引和转换。新实现
用 `np.nonzero` 一次提取有效非对角索引，批量转换各字段，再用 `starmap` 构造
`DeflectionEntry`；C-order 保证历史顺序不变。

K16/Q16、3840条有效边、100次物化微基准：6.14 ms降至2.39 ms，降低61.1%。相同
10 seed × 30 frame回归相对O4：

| 指标 | O4 | O5内核 | O5双服务最终包 |
|---|---:|---:|---:|
| 性能字段最大绝对误差 | — | 0 | 0 |
| 控制关键路径均值 | 17.14 ms | 16.47 ms | 17.20 ms |
| 仿真整帧均值 | 66.77 ms | 57.39 ms | 60.10 ms |
| 整帧加速 | — | 1.164x | 1.111x |
| W=3 worst_min | 0.844917 | 0.844917 | 0.844917 |
| W=3 QoS通过率 | 100% | 100% | 100% |

对象物化也被封装为独立的 `uav_isac.acceleration.deflection` 服务，避免和物理系数重建
职责耦合。默认后端通过 `staticmethod(DenseDeflection.to_entries)` 零转发调用；可通过
`deflection_materialization_backend` 注册名或 `module:factory` 替换。O5算法回归文件为
`results/k16_dense_entries_o5_blind10x30_diagnostic.json`，双服务封装回归文件为
`results/k16_acceleration_services_o5_blind10x30_diagnostic.json`。重复运行的整帧观测范围为
57.39--60.10 ms；表中最终包采用较保守的60.10 ms，进程池调度波动不计作算法收益。

## O6--O7：viewer批量重建与稀疏gain直达

O6把hold帧原先16次`reconstruct_selected`合并为一次
`reconstruct_selected_batch`。内核仍在每个viewer的public state上计算完整的双基地路径和
DD耦合量，没有把sinc错误拆成端点乘积。K16五帧golden以零容差比较结构、三套gain、LP
功率、价格、证书和P_D，全部`max_abs=0`。

O7进一步删除`(V,E) -> V份(K,K,Q) -> (V,K,Q)`的往返，新增
`fixed_owner_gain_matrix_from_selected_values`，按原边顺序直接累加到viewer批量gain。该步骤
同样通过零容差golden，并由dense/COO对拍单测覆盖。

相同10 seed × 30 frame配对结果：

| 指标 | O5双服务 | O6批量重建 | O7稀疏gain直达 |
|---|---:|---:|---:|
| 控制关键路径均值 | 17.20 ms | 12.97 ms | 11.55 ms |
| 控制关键路径P95均值 | 26.51 ms | 25.30 ms | 25.48 ms |
| 仿真整帧均值 | 60.10 ms | 54.92 ms | 53.94 ms |
| 仿真整帧P95均值 | 78.80 ms | 73.99 ms | 73.86 ms |
| W=3 worst_min | 0.844917 | 0.844917 | 0.844917 |
| W=3 QoS通过率 | 100% | 100% | 100% |

O7相对O5使控制关键路径均值降低32.9%，整帧均值降低10.3%。10个seed的控制关键路径
均相对O6下降0.79--2.23 ms；P95与LP墙钟没有同步改善，说明尾延迟已主要受拓扑更新帧和
求解器/调度影响，不能把O7描述成尾延迟优化。

结果文件分别为
`results/k16_acceleration_p1_selected_batch_blind10x30_diagnostic.json`和
`results/k16_acceleration_p1_sparse_gain_blind10x30_diagnostic.json`；零容差对拍为
`results/acceleration_p1_sparse_gain_compare.json`。
