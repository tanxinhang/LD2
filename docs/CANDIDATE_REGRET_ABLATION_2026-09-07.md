# Candidate provenance × task-regret rank 联合消融（2026-09-07）

## 1. 目的

oracle ladder 显示 candidate 与 rank 都有影响，但两者可能存在交互。这里固定同一
条 trace、同一 exact projection、同一 hold-5 和同一 realized physical deflection，
构造正交 2×2：

| | CE rank | task-regret rank |
|---|---|---|
| local sparse candidate | local_ce | local_task_regret |
| full physical candidate | full_ce | full_task_regret |

local candidate 只由 CE Student 生成一次，再用于两个 ranker，避免把候选变化错误归因
为排序变化。评估为 18 个未参与训练 seed、2700 帧、558 resolve 帧的离线 same-state
诊断，不是闭环部署证明。

可复现工具：
[`tools/audit_candidate_regret_ablation.py`](../tools/audit_candidate_regret_ablation.py)。

## 2. 结果

| 条件 | steady | weak3 | worst | CVaR | QoS feasible |
|---|---:|---:|---:|---:|---:|
| local + CE | 0.885004 | 0.776410 | 0.563733 | 0.200153 | 0.500 |
| full + CE | 0.892327 | 0.788104 | 0.569344 | 0.201942 | 0.500 |
| local + task-regret | 0.885701 | 0.777881 | 0.580121 | 0.207927 | 0.500 |
| full + task-regret | 0.882440 | 0.767711 | 0.558299 | 0.169697 | 0.500 |

### 正交对比

| 对比 | Δsteady | Δweak3 | Δworst | ΔCVaR | ΔQoS feasible |
|---|---:|---:|---:|---:|---:|
| candidate 增益（CE rank） | +0.007323 | +0.011694 | +0.005611 | +0.001788 | 0 |
| candidate 增益（task-regret rank） | −0.003261 | −0.010170 | −0.021822 | −0.038230 | 0 |
| task-regret 增益（local candidate） | +0.000697 | +0.001471 | +0.016388 | +0.007774 | 0 |
| task-regret 增益（full candidate） | −0.009888 | −0.020392 | −0.011045 | −0.032244 | 0 |

candidate×rank 的 worst-P_D 交互项为
`−0.021822 − (+0.005611) = −0.027433`，不是可忽略的数值噪声。task-regret
rank 在 local candidate 上改善 worst，但换成 full candidate 后反而恶化；说明它
不是可以独立插拔的 ranker。

## 3. 判定

1. **candidate 与 rank 应联合设计，而不是先后堆叠。** CE rank 能利用更完整候选，
   task-regret rank 却对候选分布敏感；当前接口没有保持输入分布/候选语义一致。
2. **task-regret 的正确创新对象不是一个新 loss，而是 candidate-aware、hold-aware
   的 rank interface。** rank 必须知道候选覆盖、target-empty 风险和 hold-5 切换代价，
   并直接优化 episode 三地板。
3. **仍不能声称 QoS 已提升。** 四个条件的 episode QoS-feasible 都是 0.500；目前
   只证明了 worst/weak3/CVaR 的结构性变化。
4. **factor message/certificate 仍属于后续保证层。** 当前失败由候选—排序交互解释
   得更充分，不能用通信模块掩盖接口问题。

## 4. 精炼后的系统创新表述

完整系统的创新不应写成“Student + candidate + certificate”的模块清单，而应写成：

> **一个决策权一致的多时间尺度混合控制框架，其中 candidate provenance、task-regret
> ranking 与 hold-aware projection 通过可追踪的 episode-level credit 接口协同，
> 并在通信必要性被证明后扩展为 decision-sufficient certificate。**

原始机器可读结果：
[`results/candidate_regret_ablation_k6q6_clean18/summary.json`](../results/candidate_regret_ablation_k6q6_clean18/summary.json)。

## 5. 第一步：把候选接口变成可计算的 episode regret

新增 [`uav_isac/evaluation/episode_regret.py`](../uav_isac/evaluation/episode_regret.py)，
定义了不做任意标量化的三维 regret 向量：
`(steady_drop, weak3_drop, worst_drop)`。同时按严格规则记录：

- `candidate_unsupported`：某 episode 的任一 resolved hold 帧缺少参考结构边；
- `candidate_miss_qos_regret`：候选不支持且参考 episode 可行、方法 episode 不可行；
- `rank_or_execution_qos_regret`：候选支持但仍发生上述可行性翻转。

将 full oracle rank 作为参考的 18-seed 结果中，local CE 的平均向量 regret 为
`(0.02379, 0.04345, 0.02389)`，local task-regret 为
`(0.02224, 0.04164, 0.01816)`；两者 `qos_flip_rate=0`。这意味着当前 trace 只能证明
连续性能损失和候选覆盖不足，尚未证明候选缺失导致 QoS 可行性翻转，更不能据此宣称新算法。

该指标已接入 `audit_candidate_regret_ablation.py` 的机器结果，测试为 3 passed。下一步才是
在同一指标下做候选生成器/排序器的预注册独立盲测。
