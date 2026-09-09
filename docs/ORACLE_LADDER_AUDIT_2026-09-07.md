# Oracle ladder：state → candidate → rank → projection（2026-09-07）

## 1. 目的与边界

本审计把结构 Student 的性能缺口拆成四个可干预层，而不是把所有 oracle 结果
合并成一个“上界”：

```text
记录的 local state
  → oracle state
  → oracle candidate pool
  → oracle rank
  → oracle projection
```

所有层使用同一条 6×6 teacher trace、同一 `P_FA`/QoS 地板、同一 hold-5 规则、
同一 `privileged_d_eff` 实现结果。评估使用 checkpoint 未参与训练的 18 个 seed、
2700 帧和 558 个 resolve 帧；训练重叠的 291、606 被排除。结果是离线 same-state
归因，不是闭环部署性能证明。

可复现命令：

```text
E:\anaconda\conda\python.exe tools/audit_oracle_ladder.py \
  --trace results/architecture_v2_scale_k6q6_teacher_trace_selection20/teacher_trace.npz \
  --config config/exp_800_k6q6_architecture_v2_maxmin_local_fusion_fullgraph_hold5_scale.yaml \
  --student-checkpoint results/_gate1_multiscale_ce/frozen_structure_student_multiscale_ce.pt \
  --output results/oracle_ladder_k6q6_clean18/summary.json --max-seeds 18
```

## 2. 四级干预定义

| 层 | 替换内容 | 保持不变 |
|---|---|---|
| baseline | 记录的 local observation、local sparse candidate、Student rank | exact receiver-owner projection |
| oracle state | 用当前目标 state 重建 belief；若 trace 已为真值则保持 time-aligned geometry | 候选规则、Student、projection |
| oracle candidate | 将 local sparse pool 换为完整物理支持图 | oracle-state Student、projection |
| oracle rank | 将 Student 分数换为 realized `d_eff` | 完整候选图、projection |
| oracle projection | 再次显式调用同一 exact max-min single-role projection | oracle rank 的候选与分数 |

最后一级是 identity audit：当前 live P0 已经使用该 exact projection，因此不能
人为换一个更强求解器后再宣称“projection 创新”。

## 3. 结果

### 3.1 汇总性能

| rung | steady | weak3 | worst | CVaR | episode QoS feasible |
|---|---:|---:|---:|---:|---:|
| baseline | 0.885004 | 0.776410 | 0.563733 | 0.200153 | 0.500 |
| oracle state | 0.885004 | 0.776410 | 0.563733 | 0.200153 | 0.500 |
| oracle candidate | 0.892327 | 0.788104 | 0.569344 | 0.201942 | 0.500 |
| oracle rank | 0.902757 | 0.808497 | 0.566790 | 0.216882 | 0.444 |
| oracle projection | 0.902757 | 0.808497 | 0.566790 | 0.216882 | 0.444 |

### 3.2 增量归因

| 增量 | Δsteady | Δweak3 | Δworst | ΔCVaR | ΔQoS-feasible |
|---|---:|---:|---:|---:|---:|
| state → oracle state | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000 |
| state → oracle candidate | +0.007323 | +0.011694 | +0.005611 | +0.001788 | 0.000 |
| candidate → oracle rank | +0.010429 | +0.020393 | −0.002554 | +0.014941 | −0.056 |
| rank → oracle projection | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000 |

补充证据：local candidate 对 oracle-rank 解的 edge recall 为 **0.8002**，owner
recall 为 **0.9659**，target empty rate 为 **0.0332**；完整候选只占物理支持边的
约 51.6%。结构变化率为 state→candidate **82.5%**、candidate→rank **97.6%**，
而 rank→projection 为 0、pair Jaccard 为 1。

## 4. 判定

1. **state 不是当前 6×6 结构瓶颈。** 该 trace 的 belief/geometry 已经是决策时
   真值；oracle-state 干预最大观测差 `5.96e-8`，没有候选或 QoS 增益。
2. **candidate localization 是稳定的结构缺口。** 完整候选带来 `+0.00561`
   episode-worst、`+0.01169` weak3，但尚未改变 QoS-feasible rate；当前 local
   candidate 的 edge recall 只有 0.8002，且存在 3.32% target-empty frames。
3. **rank 是跨目标/跨帧的第二缺口，而不是简单的“越精确越好”。** exact rank
   提高 steady/weak3 和 CVaR，却使 episode-worst 略降、QoS-feasible 从 0.500
   降至 0.444，说明当前 Student 的局部排序与 hold-5/episode 三地板之间存在
   temporal credit mismatch。不能直接把“oracle rank 更强”写成方法收益。
4. **projection 不是缺口。** 当前 projection 已是 exact operator；重复 oracle
   投影完全相同。因此新增 certificate/projection 层没有现有证据支持。

## 5. 对创新主线的精炼提升

本结果不要求把系统收缩成单一 candidate 模块，而要求把整体方案改写成有因果接口
的三层创新：

1. **核心架构：决策权一致的多时间尺度混合控制。** 长期运动、通信调度和候选生成
   由 learned policy 负责；固定结构下的凸资源分配由解析层负责；硬边界由 projection
   负责。每个 actor head 必须拥有真实执行权，并能追踪到 episode-level credit。
2. **已被 ladder 指向的性能增强：episode-QoS-regret candidate/rank。** 用自然的
   local candidate provenance 保证 target/owner 覆盖，再让 rank 直接优化 hold-5
   后的 worst/weak3，而不是只拟合逐帧 edge value。candidate 是当前 worst-P_D 的
   首要缺口，rank 是 steady/weak3 与 temporal stability 的第二缺口。
3. **条件性保证层：decision-sufficient factor communication/certificate。** 它不被
   删除，也不再被预先宣称为收益；只有等 bit learned-latent 与 factor interval 在
   同一 QoS、AoI、FBL、计算预算下完成闭环对照，并证明普通 Student 无法保持决策/约束
   性质时，才把它提升为正式保证机制。

因此“精炼提升”不是减少模块数量，而是把模块从并列堆叠改为：

```text
决策权一致的混合架构
        ↓
episode-QoS-regret 候选/排序增强
        ↓（仅在必要性实验通过后）
decision-sufficient factor message + certificate
```

下一步应做 candidate provenance × task-regret rank 的联合消融，报告 candidate recall、
target-empty、结构切换率、episode worst/weak3、QoS 可行率和通信代价；通信 certificate
保留为条件性扩展，不提前降级为“无关模块”。

原始机器可读结果：
[`results/oracle_ladder_k6q6_clean18/summary.json`](../results/oracle_ladder_k6q6_clean18/summary.json)。
