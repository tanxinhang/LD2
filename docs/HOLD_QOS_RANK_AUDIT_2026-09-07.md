# Hold-aware QoS 排序上界审计（2026-09-07）

## 结论

当前 worst/QoS 的主要可改进方向是**排序时间尺度与 hold-5 执行时间尺度不一致**。在两个
checkpoint 训练 seed 上校准、随后固定参数到18个未参与校准 seed 后，使用 hold 段未来
`d_eff` 的20%分位数排序，相对当前帧 oracle rank 得到：

| 条件 | steady | weak3 | worst | CVaR | QoS feasible |
|---|---:|---:|---:|---:|---:|
| 当前帧 oracle rank | 0.902757 | 0.808497 | 0.566790 | 0.216882 | 0.444 |
| hold-aware q20 oracle rank | 0.918233 | 0.836728 | 0.610017 | 0.217255 | 0.500 |
| 增量 | +0.015477 | +0.028232 | **+0.043227** | +0.000372 | **+0.0556** |

这证明 future-hold lower-tail ranking 确实能同时提升 mean worst 和 episode QoS feasible；但
CVaR 几乎不变，QoS 仍只有0.50，因此它没有解决最深尾部，也不是可部署性能结果。

## 方法

当前结构每个 resolve frame 选择一次，随后保持到下一个 resolve frame。审计对每条候选边
计算当前 hold 段的证据序列，并构造：

```text
score = (1-w) * hold_mean(d_eff) + w * hold_quantile_q(d_eff)
```

在训练 seed `291, 606` 上按 `(QoS feasible, worst, weak3, steady)` 字典序选择
`q=0.20, w=1.0`，然后冻结参数，在其余18个 seed、2700帧、558个 resolve frame 上评估。
候选池、exact projection、QoS 门槛、hold 时序和物理结果均保持不变。

## 证据边界

该排序读取整个未来 hold 段的 realized `d_eff`，因此是非因果 oracle，不能进入部署链。它的
作用是排除错误方向：

- 单纯提高当前帧 edge rank 精度不足以保护 episode worst；
- 排序标签必须对应动作实际保持的时间段；
- 训练目标应预测 hold 段 lower-tail，而不是拟合单帧 teacher edge value；
- CVaR 不变说明还需要单独处理几何不可达/资源不足 episode，不能把所有尾部归因给 ranker。

## `worst >= 0.75` 硬门槛

不能用跨 episode 的 `worst` 均值替代逐 episode 硬门槛。在18个未参与校准的评估 seed 上，
对同一目标连续积累 Deflection 后得到：

| 排序条件 | 窗口 | 时延 | mean(worst) | min(worst) | 5%分位 | 逐episode通过率 |
|---|---:|---:|---:|---:|---:|---:|
| 当前帧 oracle rank | 1 | 0.1 s | 0.566790 | 0.114594 | 0.200634 | 38.9% |
| 当前帧 oracle rank | 2 | 0.2 s | 0.808848 | 0.338446 | 0.529948 | 72.2% |
| 当前帧 oracle rank | 3 | 0.3 s | 0.926838 | 0.567433 | 0.774377 | 94.4% |
| 当前帧 oracle rank | 5 | 0.5 s | 0.988941 | **0.855469** | 0.959508 | **100%** |
| hold-aware oracle rank | 1 | 0.1 s | 0.610017 | 0.114594 | 0.200634 | 44.4% |
| hold-aware oracle rank | 2 | 0.2 s | 0.828514 | 0.338446 | 0.529948 | 72.2% |
| hold-aware oracle rank | 3 | 0.3 s | 0.932741 | 0.574665 | 0.775462 | 94.4% |
| hold-aware oracle rank | 5 | 0.5 s | 0.989887 | **0.875087** | 0.962451 | **100%** |

因此，若门槛指“评估集中每个 episode 的 worst 均不低于0.75”，最小已验证窗口是
`W=5`，而不是均值首次达标的`W=2`。`W=3`虽通过5%分位，但 seed 138 仍失败。

这一结论只在当前K6/Q6、18 seed和帧间独立白化证据假设下成立；它不是K16/Q16正式系统的
盲测结论，也不能声称单帧已达标。将`W=5`纳入正式任务意味着接受0.5 s观测时延，并需先
校验帧间相关性、目标身份连续性以及重叠窗口的虚警控制。

## 下一步算法门

下一步只实现一个部署可用的对象：从当前 local belief、AoI、目标速度和候选历史，预测
`q20(d_eff[t:t+H])`，并在相同 candidate budget 下替换当前单帧 Student rank。必须保留：

1. 当前 CE Student；
2. current-frame task-regret Student；
3. deployable hold-q20 predictor；
4. non-causal hold-q20 oracle 上界。

只有第3项在新 seed 上同步改善 worst、QoS 和CVaR，才能称为算法改进。否则结论应是现有
观测不足以预测 hold-tail，需要转向 belief dynamics 或资源可行性，而不是继续改 loss。

可复现工具：[`tools/audit_hold_qos_rank.py`](../tools/audit_hold_qos_rank.py)。机器结果：
`results/hold_qos_rank_k6q6_clean18/summary.json`。
