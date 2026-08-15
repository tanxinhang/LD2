# D1.1-A — Lexicographic QoS-Constrained Max-Min L1（advice 010）

> 文档日期：2026-08-15。依据 [`advice/010.md`](../advice/010.md) 的 D1.1-A。
> 关键代码：`uav_isac/coordination/capability.py::qos_constrained_maxmin_lp`、
> `uav_isac/environment/env_core.py`（`task_constrained_mode="lexicographic"`）、
> 配置 `config/exp_800_k8q8_analytical_l0l1_movement_lex.yaml`（margin 地板
> `[0.61,0.71,0.81]` + lex 模式）。

## 1. 算法：Gauge 做 feasibility certificate，MaxMin 在可行域内做 performance ascent

```text
Stage A: capability gauge  →  γ* ≤ 1 ?    （证明 worst/weak3/steady 三地板可行）
Stage B: QoS-constrained max-min
         max t  s.t. D_q ≥ t, D_q ≥ d(worst), weak3 ≥ weak3_floor,
                    steady ≥ steady_floor, Σ_q p_iq ≤ b_i
         （gauge 的分配本身是 Stage-B 的可行点 ⇒ lex 在 QoS 与 worst 上都不劣于 gauge）
不可行帧: 回退 reserve-first max-min（worst 地板储备，best effort）
```

与 penalty hybrid 的本质区别：不是 `J = J_maxmin − ρ·QoSViolation`（有权衡），而是
**字典序** `QoS feasibility first ≻ worst maximization second`。

## 2. 对 Round 1「gauge QoS 0.90」结论的重要修正

**Round 1 的 oracle gauge 内层有个 bug**：`capability_gauge_pwl_lp_full` 在 γ*>1 时
返回的是**按 γ 缩放后的功率分配**（Σp ≤ γ·b > 1 W），oracle 的 `_eval` 无条件用它算
Deflection，导致显示 QoS 被**预算违规的乐观功率**抬高（例如 seed 15 的 "gauge 1.0"
实际用了 69.6× 预算）。已修正：γ*>1 时显示物理可行的 max-min best effort（分数仍为
−γ，驱动可行性）。

**修正后的固定几何对比**（20 seed 最终 resolved frame，priced structure + 各 L1）：

| 内层 | mean worst | QoS feasible | 说明 |
|---|---:|---:|---|
| gauge（修正后） | 0.782 | **13/20** | γ>1 帧回退 max-min，与下面相同 |
| lexicographic | 0.801 | 13/20 | Stage B 可行时推高 worst |
| maxmin | 0.801 | 13/20 | — |

**关键结论**：在教师轨迹最终几何上，三种 L1 的 QoS 相同（13/20）——**7 个 seed 的
三地板在教师最终几何上物理不可行（15/34/402/488/591/751/922），瓶颈是几何而非 L1**。
唯一种子 878 展示机制：gauge 可行（γ≤1）但分配被钉在地板（worst 0.610），lex/maxmin
在可行域内把 worst 推到 1.0——**这就是"满足门限后的剩余资源"**。

因此 Round 1 的「gauge 擅长可行性、maxmin 擅长绝对性能」应修正为：
**L1 的选择在固定几何上影响很小（γ>1 时都回退 max-min）；真正的主导杠杆是 L3 几何
（live 150 帧 L3 把几何推到三地板可行的位置，QoS 才达到 20/20）**。lex L1 的价值在
**好几何上的剩余资源回收**。

## 3. Live 验证（D1.1-A）

live 接入：`task_constrained_mode="lexicographic"` + margin 地板 `[0.61,0.71,0.81]`。
Stage A γ≤1 时用 Stage-B 功率（推高 worst），否则回退 reserve-first max-min。

**2-seed 冒烟**（`run_mappo.py` eval-only，同一 warm-start / test 种子库）：

| seed | gauge-margin worst | **lex worst** | steady | QoS |
|---|---|---|---|---|
| 503 | 0.61（地板钉死） | **0.835** | 0.835 | ✅ |
| 700 | 0.923 | 0.923 | 0.923 | ✅ |

**seed 503 的 worst 从 0.61 → 0.835**：gauge 在可行域内 satisficing 把 worst 钉在地板，
lex 在**同一几何**用剩余资源把 worst 推高——advice 010 §6 预测的机制在 live 系统得到
确认。

**20-seed 全量 lex 复测**（`results/_d095_lex20/paired_eval.csv`，同一 warm-start /
test 种子库，`run_mappo.py` eval-only）：

| 指标 | D095 基线 | + margin (gauge) | **+ lexicographic L1** |
|---|---:|---:|---:|
| mean worst | 0.662 | 0.671 | **0.844** |
| mean weak3 | 0.724 | 0.733 | **0.844** |
| mean steady | 0.808 | 0.817 | **0.845** |
| 严格 QoS feasible | 0.65（浮点伪影） | 1.0 | **1.0（20/20，LCB 0.881）** |
| worst ≥ 0.75 率 | — | — | **20/20** |

配对 Δworst（lex − margin）：mean **+0.173**，17/20 seed 改善，0 个退化；即使 seed
751（冻结 DD 下此前最难的 seed）也从 0.61 提到 0.817。**机制**：Stage-B max-min 把
8 个目标均衡到同一水平 t*，且 t* 高于三地板（0.844 > 0.81），因此 weak3 = steady =
worst = t*，三地板全部以真实余量通过。

**判定：D1.1-A Gate 大幅通过**——`P_QoS = 1.0`（≥ 目标），`mean(Δworst) = +0.173 > 0`
（≥ 目标），mean worst 0.844 远超 D1.1-A 的 0.69~0.70 目标，**并且直接达成 advice
Phase 2（worst ≥ 0.75、QoS ≥ 0.90）**。仅改 L1（不动 L3/结构/轨迹）就回收了 gauge
satisficing 浪费的全部"满足门限后剩余资源"。

## 6. 审计（Round 3）：修改正确性核验 + 机制澄清

对 lexicographic L1 的修改做了严格审计（含生产 harness 逐帧重放 seed 503，
`DSH_LEX_AUDIT` 钩子，150 帧记录）：

### 6.1 正确性审计：全部通过

| 检查项 | 结果 |
|---|---|
| Stage-B LP 公式（max t s.t. D_q≥t, D_q≥d_min, PWL chord 三地板, Σp≤b） | ✓ 与 gauge 同构，逐项核对 |
| 功率物理有效：1W 平衡误差 | ✓ ≤ **9.99e-16**（机器精度） |
| per-UAV comm+sensing 总功率 | ✓ 恰为 **1.0 W**（max 1.0000000000000002） |
| budget 来源 | ✓ 帧首 `p_sense·weights` 求和 = **1 − P_comm** |
| P_D 真实性 | ✓ 报告 min P_D = **P_D(t\*) 到 9.99e-16**（Stage-B 功率直接产生报告的 P_D） |
| 代码路径 | ✓ `_current_sensing_power_w = lp.power_w` → deflection 重算 → P_D，无覆盖 |
| QoS 自洽 | ✓ 20/20 由 episode 原始数组重算一致 |
| lex 分支实际调用 | ✓ 150 帧中 142 帧 stage_b、8 帧 reserve_fallback（早期瞬态） |

### 6.2 机制澄清（重要，诚实表述）

lex20 数据里**所有 seed 的 max-min ≥ 0.817 > 0.81（steady 地板）**，且 weak3 == worst
（均衡特征）——说明在 live 几何上**三地板从不绑定**，Stage-B 数学上等价于**纯 max-min
L1**。因此：

- 0.844 的 worst 是 **max-min 对"满足门限后剩余资源"的回收**（gauge satisficing 把
  worst 钉在 0.60 浪费的部分），真实且可复现；
- QoS 20/20 是高 max-min 水平（≥0.81）的**自然结果**，不是地板约束强制的结果；
- 诚实表述应为"**带三地板守卫的 max-min L1**"；若要看到 lex 地板真正绑定，需在
  max-min 水平 < 0.81 的几何上运行（更差起点 / 更多目标）。

### 6.3 修复与遗留

- 修复：env_core lex 分支 `pi_star` 原接收 Stage-B 的 deflection 向量（误标为
  prices，无执行影响），已改为真实 max-min 对偶价格 `optimal_maxmin_dual_prices`。
- 遗留（非 lex 引入）：早期 ~8-19 帧瞬态（L3 滚动收敛，worst/steady 低于地板），
  D095 已知问题；eval 的 steady-window（末 20 帧）指标不受影响。
- 审计钩子 `DSH_LEX_AUDIT`（环境变量门控，默认关闭）保留，供后续审计复用。

## 7. D1.1-B — Bounded Multi-Candidate Trust-Region L3（Round 3，验证中）

针对"worst 上来了但整体（weak3/steady）没跟上"（max-min 均衡化把 steady 钉在 worst
水平 ~0.844）：

1. **帕累托分析**（live 几何，max ΣD subject to worst 地板）：worst 地板 0.61→0.80 时
   steady 仅 0.828→0.843（甚至略低于 max-min 的 0.845）——**非均衡分配不能提升整体**，
   因为功率在 max-min 最优处耗尽（审计实测 slack=0）；瓶颈是**几何**（所有目标增益的
   绝对值），非分配方式。
2. **实现**（`env_core._analytical_movement_delta` + `_select_best_movement_candidate`，
   参数 `analytical_movement_candidates_enabled`）：每帧评估 ~5 个**整队运动候选**
   （stay / λ* 双价格步 / 半步 / 朝最弱两个目标的径向步），每个候选在移动后几何用
   精确 max-min LP 打分（P_D 空间，饱和到 1），选最优执行；stay 恒在候选集 → 代理
   score 单调。
3. **教训（重要）**：初版候选评估用原始 Deflection score + 无防撞保护，导致 UAV 无限
   逼近目标（gain 达 9e4，P_D 虚高到 1.0）且贪心振荡（worst 瞬时塌到 0.001）。修复：
   **d_safe=20m 防撞保护**（剔除使任意 UAV 距任意目标 < d_safe 的候选）+ **P_D 饱和
   score**（饱和后继续逼近无收益）。
4. **2-seed 验证**（seed 503/700，确定性复现两次一致）：seed 503 的 worst/weak3/steady
   从 0.835/0.835/0.835 → **0.952/0.952/0.958**，QoS 1.0，功率平衡 1.8e-15。单帧
   worst 在末段偶有瞬时塌陷（~0.09，结构重排引起），被 steady-window 时序均值抹平
   （与 D095 早期瞬态同类）。

**20-seed 全量**（`results/_d095_lexcand20/paired_eval.csv`，确定性）：

| 指标 | margin (gauge) | lex L1 | **lex + 多候选 L3** |
|---|---:|---:|---:|
| mean worst | 0.671 | 0.844 | **0.975** |
| mean weak3 | 0.733 | 0.844 | **0.978** |
| mean steady | 0.817 | 0.845 | **0.982** |
| 严格 QoS | 1.0 | 1.0 | **1.0（LCB 0.881）** |
| worst ≥ 0.90 率 | — | — | **19/20** |
| worst ≥ 0.95 率 | — | — | **17/20** |

**"整体性能"问题解决**：max-min 均衡化曾把 steady 钉在 worst 水平（0.844）；多候选
L3 提升所有目标的几何 ceiling，使 worst/weak3/steady 一起升到 **0.975/0.978/0.982**，
全部超过 advice 010 的 Phase 1-3 目标。审计确认：无 NaN、无 d_safe 违规、功率平衡
≤1.8e-15、2-seed 确定性复现。

配置：`config/exp_800_k8q8_analytical_l0l1_movement_lex_candidates.yaml`（lex L1 +
margin 地板 + 多候选 L3）。

## 4. 配套：QoS 评估容差（advice 010 "必须修"）

trainer 的 QoS feasible 检查原为严格 `x >= floor`，导致 gauge 把 worst 精确钉在
0.60 时以 `0.60 − 1.11e-16` 被误判失败（D095 报 QoS 0.65 的浮点伪影）。已修复：

- 新参数 `marl.qos_eval_tol = 1e-6`（solver 级容差，`compute_robust_checkpoint_statistics`
  新增 `qos_tol` 参数）。
- 输出新增 `eval_qos_tol`、`eval_{worst,weak3,steady}_slack`（raw − floor）、
  `eval_raw_{worst,weak3,steady}_P_D`——容差不隐藏真实低于门限。
- 验证：模拟 D095 边界情形（7 seed worst=0.60−1e-16），tol=0 → QoS 0.65/LCB 0.467
  （与 D095 报告完全一致），tol=1e-6 → QoS 1.0/LCB 0.881。

## 5. 统计口径提醒（advice 010 §末尾）

这组 stratified seeds 已用于浮点问题发现、margin 选择与多轮验证，**不能再称最终
blind test bank**；D1.x 固定后应换 ≥100 全新 seed 做最终统计认证（D1.5）。
