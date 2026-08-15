# D0.89-A：解析内层功率 / 学习外层协调（删 sensing-power 自由度）

> 状态：A 阶段实现完成，冻结 Actor 复现待接 8/8 checkpoint。
> 实现：`config/params.py`（`analytical_sensing_power_enabled` / `_reserve_pd`）、
> `uav_isac/coordination/maxmin_power.py`（reserve-first 精确 LP）、
> `uav_isac/environment/env_core.py`（LP 覆盖 hook）。
> 结论：**架构定义已落地**——sensing power 从学习自由度中移除，改由固定 owner
> max-min LP 分配；Actor 只保留 `Δp, P_comm, Token, structure`，通过
> `P_comm → b_i = 1 − P_comm` 仍控制感知预算。

## 1. 核心改变（一句话）

```text
旧:  Actor 同时输出 P_comm 和 p_iq (逐目标感知功率)
新:  Actor 输出 P_comm (通信/感知预算切分)；内层 LP 决定 p* = argmax min_q D_q
```

Actor 并未失去功率控制力：它通过 `P_comm` 控制 `b_i = 1 − P_comm`，再交给 LP 在
目标间最优分配。这恰好是"学习负责改变 LP 可行域/系数矩阵的变量（几何、通信、
结构），解析负责 LP 内可解析的功率分配"的层次分工。

## 2. 实现位置（env_core.step）

在 P0 固定 role/owner/edge 之后、检测之前插入：

```text
1. coefficient = a_ijq = 1[g_dd>=g_min] · chi_rep · alpha² · C   (功率无关)
2. gain, owner = fixed_owner_gain_matrix(coefficient, selected)
3. budget = sum_q P_sense,kq  (= 1 − P_comm)
4. p*, t*, λ* = MaxMinLP(gain, budget, reserve)
5. 覆盖 _current_sensing_power_w = p*，重算 Deflection → P_D
```

关键点：

- **功率无关重建**：`a_ijq` 直接从 entry 的 `alpha/g_dd/chi_rep` 解析重建
  （`_per_watt_coefficient_from_entries`），**不做 `d_eff/P_sense` 除法**，因此
  未激励边（Actor 给 0 功率的边）仍然可识别——这正是 D0.12 强调的可辨识性问题。
- **结构不变**：LP 覆盖发生在 P0 之后，所以 role/owner/edge/Token/motion 在当帧
  完全一致（测试 `test_analytical_power_does_not_change_structure` 已验证
  frame0 的 roles/owner/n_selected 逐值相同）。
- **确定性**：`ground_communication_enabled=false` 且 `use_swerling=false` 时
  重算 Deflection 不消耗 RNG，重放确定。
- **功率平衡**：LP 的 `_fill_budget` 保证 `Σ_q p_iq = b_i` 精确成立，误差
  `< 1e-12`（测试已断言）。

## 3. reserve-first（为 D0.89-C 的 steady 张力预留）

`solve_fixed_structure_maxmin_power_lp` 新增 `minimum_deflection`：

```text
max_p,t  t
s.t.  Σ_i a_iq p_iq >= t     (max-min)
      Σ_i a_iq p_iq >= r_q   (reserve, 可选)
      Σ_q p_iq = b_i
```

`r_q` 由 `analytical_sensing_power_reserve_pd` 经
`minimum_deflection_for_detection_probability` 从 `P_D` 地板反解。这保证 D0.89-C
训练时用 "reserve feasibility ≻ t* ≻ comm cost" 的层级目标，而不是纯 max-min
削峰填谷牺牲 steady。

## 4. 验证状态

- `tests/test_analytical_sensing_power.py`：5 passed（reserve LP、infeasible
  报错、env 集成 + 功率平衡 `<1e-12`、系数重建、结构不变性）。
- 全量相关回归（env/reward/detection/maxmin）通过；默认 flag=False 时行为不变。

## 5. D0.89-A 复现结果（8/8 test20，冻结 Actor，20 seeds）

配置：`config/exp_800_k8q8_architecture_v2_analytical_power_d089a.yaml`；
checkpoint `results/architecture_v2_scale_k8q8_teacher_trace_d079_test20/best_restored.pt`；
`--episodes 0 --max-final-eval-seeds 20`，同 seed 严格配对。

| Metric | deployed（flag OFF） | analytical LP（flag ON） | Δ |
|---|---:|---:|---:|
| worst P_D | 0.3554 | 0.5818 | **+0.2264** |
| weak3 P_D | 0.5084 | 0.5945 | +0.0861 |
| steady P_D | 0.7731 | 0.6923 | −0.0808 |
| QoS feasible rate | 0.40 | 0.55 | +0.15 |
| QoS Wilson LCB | 0.242 | 0.372 | +0.130 |
| worst CVaR20 | 0.0131 | 0.0872 | +0.0741 |
| 功率平衡误差 | 3.3e-16 | 4.4e-16 | 精确 |

**判定：**

1. **复现成立**：live env 的 `+0.2264` worst 与 D0.87 trace 级的 `+0.2447` 同量级
   同方向（差异来自 trace 工具与 eval 的聚合口径/teacher 结构细节），机制等价已
   由"系数重建同源 + 结构不变"保证。
2. **功率平衡精确**（`<1e-12`）：`1 W` 预算不因 LP 覆盖破坏。
3. **steady 张力实锤**：pure max-min 把 steady 从 `0.773` 压到 `0.692`（削峰填谷），
   这正是 D0.89-C 必须用 reserve-first 的原因；否则 QoS 门（steady≥0.80）永远过不去。
4. **Gate 1 门槛未达**：`mean-worst(ON)=0.5818` 仍低于 `0.60`，且稳态地板未满足。
   这符合预期——**D0.89-A 只是"实现追上架构"，不追求性能门槛**；性能缺口留给
   B（结构排序 λ*）与 C（reserve-first 重训 + outer 学习）。

## 6. 下一步（按可归因 Gate）

| Gate | 内容 | 门槛 |
|---|---|---|
| **A（本步，已过）** | 删 power head + frozen Actor + exact LP 复现 | 结构/Token/`P_comm` 一致、功率平衡 `<1e-12`、复现 ~D0.87 的 `+0.22~+0.24` |
| **B** | `a_ijq` 为 primitive score，结构边际价值用 `v_ijq = λ*_q a_ijq` | 避免 `d_eff` 循环依赖与"强目标吸走结构" |
| **C** | 用 exact LP（`H_train=1`）+ reserve/max-min reward 重训 outer | `mean-worst > frozen+LP`，paired gain ≥ `1e-2` |
| **2×2** | C0 old/C1 exact-LP/C2 retrain-旧reward/C3 retrain-新reward | 分离 LP 贡献 vs 学习贡献 vs reward 贡献 |

4/4 anchor 需重报：删 power head 后即便 Actor 不变，4/4 执行层也已升级为统一
max-min solver，不能再说"4/4 严格输出一致"，只能表述"structure/motion/comm
anchor 保持，功率执行层升级"。
