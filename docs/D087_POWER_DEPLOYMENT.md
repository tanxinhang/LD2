# D0.87：快层 max-min 功率接入部署路径（闭环节点反事实）

> 状态：完成（8/8 test20，20 个 episode，全帧）。
> 工具：`tools/audit_d087_power_deployment.py`。
> 结论：**架构—实现错位确认**。把 deployed 感知功率替换为固定 owner max-min
> LP，episode 级 mean-worst 提升 `+0.2447`，20/20 episode 改善、0 退化。

## 1. 实验设计

严格配对三模式，只改感知功率分配，role/owner/edge/通信功率/几何完全不动：

| Mode | 感知功率 |
|---|---|
| A deployed | trace 记录的 `(1-P_comm)·normalized sensing_weights` |
| B exact LP | 固定 owner max-min 功率 LP（HiGHS） |
| C Dantzig-Wolfe | 4 轮 6-bit 量化列生成 |

同一 analytic 每瓦重建（`per_watt_deflection_tensor_from_observables`）、同一
`local_only` 融合边界，逐帧重算检测，按 steady 窗口（末 20 帧）聚合。

## 2. 结果（8/8 d079_test20，20 episodes）

| Mode | mean-worst | mean-steady | mean-weak3 | QoS 可行率 | Wilson LCB | worst_min | CVaR20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A deployed | 0.3916 | 0.8048 | 0.5706 | 0.30 | 0.164 | 0.0077 | 0.0304 |
| B exact LP | **0.6363** | 0.7460 | 0.6395 | **0.55** | **0.372** | 0.0459 | 0.1357 |
| C 4×6-bit DW | 0.6266 | 0.7137 | 0.6391 | 0.55 | 0.372 | 0.0428 | 0.1238 |

配对 power gap（bootstrap 95% CI）：

| 配对 | mean gap | 95% CI | 改善/退化 |
|---|---:|---:|---:|
| LP − deployed | **+0.2447** | [+0.1680, +0.3296] | 20 / 0 |
| DW − deployed | **+0.2350** | [+0.1566, +0.3177] | 20 / 0 |

`max_deployed_replay_error = 4.44e-16`（重算 deployed 与 trace `physical_pd` 一致到
机器精度，验证重建精确且融合边界一致）；`lp_infeasible_frames = 0`。

## 3. 判定

1. **`+0.368` 的天花板 headroom 能转化为 episode 级收益，但不会全额复现**：
   ceiling audit 的 `0.307→0.675` 是特定同几何状态的离线上界；正式 episode 聚合
   gain 为 `+0.2447`，落在预期的 `+0.15~+0.25` 区间。二者必须分开报告（预注册）。
2. **20/20 改善、0 退化**：LP 功率覆盖在 episode 级是严格 Pareto 改善，不是
   "均值改善、尾部恶化"。
3. **QoS 可行率 0.30→0.55 翻倍**，Wilson LCB 0.164→0.372。
4. **DW ≈ LP 的 98%**：`0.6266/0.6363 = 0.985`，有限轮量化列生成已足够接近，
   支持 D0.88 用 DW 作为真正快层而非 centralized LP。
5. **注意**：LP 的 mean-steady 反而低于 deployed（0.746 vs 0.805），因为 max-min
   用"削峰填谷"牺牲强目标保弱目标——这正是与 QoS 门（含 steady≥0.80）的张力，
   后续 D0.89 的 `t*` 奖励需显式权衡 steady 地板，不能只最大化 `min_q`。

## 4. 对下一步的指引

- **D0.88**：DW 已证明 98% LP；扫 cadence（每帧/每 2 帧/每 5 帧/event × 1/2/4 轮）
  与 warm-start（持久价格 + incumbent 列），把 8/8 的 bit/时延从 every-frame-4-round
  压下来。
- **D0.89**：删除 sensing-power 学习头。**注意**：P0 结构选择当前按 powered
  deflection `d_eff=a_ijq·P_sense` 排序；移除 power 头后，P0 必须改按功率无关增益
  `a_ijq`（或 LP 修复后值）排序，否则结构选择失去定义。
- **D0.90**：λ* 引导慢几何（对偶灵敏度）。Danskin 定理要求最优对偶唯一；退化平局
  需选解规则或方向导数。
