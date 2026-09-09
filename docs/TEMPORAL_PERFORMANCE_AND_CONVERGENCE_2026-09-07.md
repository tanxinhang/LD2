# 多帧性能与算法收敛指标（2026-09-07）

## 1. 指标边界

单帧性能差并不必然意味着任务失败，但多帧积累改变了检测时延和统计假设，不能直接用
多帧 `P_D` 替换单帧 `P_D`。因此正式结果同时保留：

1. 单帧即时检测能力；
2. W帧证据积累能力；
3. 闭环轨迹是否稳定；
4. QoS达到后能否持续保持。

## 2. 多帧检测指标

当前检测器为独立白化 Gaussian-shift 模型。若连续W帧证据条件独立、目标身份保持一致，
Deflection可以相加：

```math
D_q^{(W)}(t)=\sum_{k=t-W+1}^{t}D_q(k),\qquad
P_{D,q}^{(W)}=Q\!\left(Q^{-1}(P_{FA})-\sqrt{D_q^{(W)}}\right).
```

实现不对 `P_D` 求和或平均，而是先累积 Deflection，再映射为检测概率。runner固定报告
`W={1,2,3,5,10,20}`，同时给出：steady、weak3、worst、QoS success、rolling QoS rate 和
`W·dt` 观测时延。

该结果是**条件性多帧能力**：若帧间噪声、Swerling散射或跟踪误差相关，则必须引入有效
相关矩阵或有效样本数；重叠窗口反复判决还需要单独控制序贯/族错误率。因此多帧结果不能
用于掩盖单帧物理模型问题。

## 3. 闭环收敛指标

对每帧的 steady、weak3、worst 序列分别报告：

- `terminal_mean/std`：尾部均值和波动；
- `terminal_slope_per_frame`：尾窗线性趋势，判断仍在改善或退化；
- `terminal_oscillation`：尾窗相邻帧平均绝对变化；
- `settling_frame`：滚动均值进入终值带、滚动标准差足够小并连续保持后的起始时刻；
- `first_hit_frame`：第一次达到门槛；
- `sustained_threshold_frame`：连续保持门槛的起始时刻；
- `terminal_threshold_hold_rate`：尾窗门槛保持比例。

联合QoS另外报告首次达到帧、持续达到帧、尾窗保持率和是否曾持续满足。这样可以区分：

```text
一次偶然越线 ≠ 达到QoS ≠ 稳定保持QoS ≠ 闭环收敛
```

这里衡量的是**回合内闭环性能收敛**。`convergence_profile` 也可以用于训练checkpoint序列，
但只有传入跨epoch独立评估值后，才能讨论MARL训练收敛。

## 4. 短回合诊断

K8/Q8、seed 7、30帧诊断中，单帧结果为 steady/weak3/worst
`0.665/0.629/0.610`，单帧QoS失败；独立证据假设下5帧积累达到约
`1.000/1.000/1.000`，QoS通过。与此同时三个单帧闭环序列均未达到稳定收敛门。

该结果只说明“多帧任务定义可能成立”，也暴露出独立累积很快饱和，必须先核验帧间相关性
和允许检测时延，不能把该30帧诊断写成正式性能提升。

## 5. `worst >= 0.75` 的门槛判定

K6/Q6的18个未校准评估 seed 上，`W=2`仅使 episode-worst 的均值达到0.809，逐episode
通过率只有72.2%；`W=3`通过率为94.4%，但最差 seed 仍只有0.567。最小的全episode通过
窗口是`W=5`：当前帧 oracle rank 下 `min(worst)=0.855`，通过率100%。

因此正式门禁必须记录 `minimum_window_with_all_episodes_pass`，不能只记录
`minimum_window_with_mean_worst_pass`。该结果仍受独立证据假设和0.5 s时延约束，后续必须在
K16/Q16盲测 seed bank 上复核，才能升级为系统性能声明。

strict runner 的多帧QoS合同现固定为 `steady>=0.80, weak3>=0.70, worst>=0.75`，汇总同时
输出跨seed的 `worst_min`、`worst_floor_pass_rate` 与
`minimum_multiframe_window_with_all_episode_worst_pass`。单帧旧门槛仍单独保留为历史诊断，
不会用于宣称0.75目标已经实现。

## 6. 缩帧与计算优化判据

对K6/Q6失败 seed 138 的反事实审计表明，将结构刷新周期由5帧缩短至1/2/3帧时，W=3的
worst仅为0.575/0.573/0.575；候选内逐目标独立最优上界也只有0.575。要使其达到0.75，
三帧累计Deflection至少还需提高约32.0%。因此不采用高频重排，而将下一算法对象定义为
“三帧剩余证据缺口”，由功率和机动共同补足。

计算侧，K16四进程私有LP在3 seed × 30 frame配对实验中保持全部检测、功率、位置、通信
轨迹逐元素相同且无回退；功率批处理约由30 ms降至10 ms，整帧均值约由97 ms降至76 ms，
平均加速1.27倍。该无损并行路径已进入K16主配置；一次性worker warm-up约0.70 s，不计入
稳态帧时延结论。

15 ms激进子截止时间的首次主配置诊断产生1.1%近似回退，因此未予保留。主配置采用已完成
逐元素等价配对验证的100 ms worker超时和串行保底；速度结论只统计无回退稳态帧。

K16盲测库前10个未见 seed 的30帧诊断中，单帧 `worst_min=0.306`、通过率80%；W=3达到
`worst_min=0.845`、worst通过率和联合QoS率均为100%。并行执行无回退，整帧均值76.7 ms，
跨seed最大p95为99.4 ms。该结果把候选正式窗口由5帧缩到3帧，但仍是10-seed短回合诊断，
必须通过100 seed × 150 frame冻结盲测后才能成为正式结论。

补充W=2后，其 `worst_min=0.647`、worst与联合QoS通过率均为90%，因此不能将窗口继续压到
2帧。当前最小诚实候选仍为W=3；下一步算法优化应专门修复W=2的单个尾部seed，而不是
继续提高已经饱和的其余9个seed。

实现：[`uav_isac/evaluation/temporal_performance.py`](../uav_isac/evaluation/temporal_performance.py)。
正式runner接入：[`tools/run_strict_distributed_pilot.py`](../tools/run_strict_distributed_pilot.py)。
