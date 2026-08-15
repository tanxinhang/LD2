# D0.92：参考点归一化的议价功率分配（Bargaining-Consistent ISAC）

> 状态：理论 Gate（T1–T4）+ 实现完成，单测通过；端到端性能待 D0.92-A 跑。
> 实现：`uav_isac/coordination/bargaining_power.py`、
> `config/params.py`（`bargaining_objective_enabled`）、`env_core.py` hook、
> `reward.py`（`utility_mode="bargaining"`）。
> 依据：`advice/001.md` 的抽象层升级。

## 1. 动机：不再让目标函数随 8/8 数字改写

旧主线把 worst/weak3/steady 本身当优化目标，于是 `0.6/0.7/0.8` 不断被塞进
目标函数（reserve、λ+μ 补丁），理论抽象性不足。本 Gate 把它抽象成：

```text
有限通信—感知资源下，如何分配 D_1,...,D_Q，兼顾公平性与效率？
```

其中 `D_q = Σ_i a_iq p_iq` 是目标 q 的统计可分性（Deflection），对固定 `P_FA`
单调映射到 `P_D`。优化 `D_q` 在检测理论上有明确意义，且不依赖某段 `P_D(D)` 曲率。
worst/weak3/steady **降级为评价指标**，只用于验收与层级路由，不进目标函数。

## 2. 目标：归一化机会公平（Kalai–Smorodinsky 议价）

对每个目标定义两个量：

```text
D_q^I = max_{p∈F} D_q(p) = Σ_i b_i a_iq      （理想点，同一可行域 F）
D_q^0 = Σ_i (b_i/Q) a_iq                      （disagreement，均匀分裂，可实现）
h_q   = D_q^I − D_q^0                        （可实现 headroom）
```

归一化机会 `r_q(p) = (D_q − D_q^0)/h_q ∈ [0,1]`，主问题：

```text
max_{p,η}  η   s.t.  D_q(p) ≥ D_q^0 + η·h_q  ∀q,  Σ_q p_iq = b_i,  p ≥ 0
```

物理解释：**让每个目标获得相同比例的"可实现感知提升机会"**，而不是相等 Deflection。

### 关键优势：仍是 LP

`D_q(p)` 线性 → 约束 `D_q − η·h_q ≥ D_q^0` 关于 `(p,η)` 线性。**固定结构下仍是
标准 LP**，D0.87/D0.88 的强对偶、DW、staleness、量化价格、event-trigger 全部继承。

## 3. 四个定理（T1–T4）

### T1（D^0, D^I 合法性与存在性）

`D^I_q = Σ_i b_i a_iq` 由 `p_iq = b_i·1[q=q]` 的极值点实现，属同一可行域 `F`，
不是 relaxed oracle；`D^0_q = Σ_i (b_i/Q) a_iq` 由均匀分配实现，可计算、不依赖
未来 oracle、对目标置换一致。`h_q ≥ 0`，`h_q = 0 ⟺ a_iq = 0 ∀i`（目标不可达）。
（已测：`test_ideal_and_disagreement_legitimacy`）

### T2（可行有界 LP + 强对偶）

`η = 0` 由均匀分配实现 → 可行；`r_q ≤ 1` → `η ≤ 1` 有界。LP 强对偶，对偶为

```text
min_{λ,μ}  Σ_i b_i μ_i − Σ_q λ_q D_q^0
s.t.  μ_i ≥ λ_q a_iq,   Σ_q λ_q h_q = 1,   λ ≥ 0, μ ≥ 0
```

stationarity `Σ_q λ_q h_q = 1`：价格 `λ_q` 定价"哪个目标的**未恢复 headroom** 最
稀缺"，而非"谁现在最弱"。**不再需要 λ+μ 人为 counterbalance。**
（已测：`test_bargaining_lp_strong_duality`、`test_bargaining_dual_stationarity`）

### T3（公理公平 / Pareto / 退化）

- A1 单调、A3 置换不变、A4 Schur-凹（min of 线性函数），A2 边际递减由 `r_q ≤ 1`
  的饱和 + 分母 `h_q` 保证——**避免旧 `-log(1-P_D)` 的富者愈富**；
- `η*` 是最大共同归一化增益 → Pareto 最优（任一目标再提升必损另一目标）；
- `D^0 = 0` 且 `h_q` 相等时退化为纯 max-min（比例）。
（已测：`test_bargaining_opportunity_fairness_against_maxmin` 验证 `r_0 = r_1`）

### T4（结构 gap + 几何方向导数）

- 结构：`G_S = V_relax − V_feasible`（MILP 整数性价格），LP 松弛给上界、可行结构给
  下界；
- 几何：`D^0, D^I` 在 trust region 内冻结时，`∇_x V = Σ_q λ*_q (Σ_i p*_iq ∇_x a_iq)`
  （Danskin），配熵正则化（λ* 退化）、支撑裕量（DD 门）、步长 trust-region；
- `g_dd = g_min` 跨门 → 停止局部微分模型、重新锚定 LP（继承 D0.88）。

## 4. 与旧 max-min 的继承关系

```text
max-min LP           →  bargaining LP（把 min_q D_q 升级为 min_q r_q）
MaxMinPowerResult    →  BargainingPowerResult（新增 η、r_q、h_q、λ）
Dantzig–Wolfe 列生成  →  完全继承（约束仍线性）
staleness 界         →  完全继承（值函数 Lipschitz 论证同构）
event-trigger        →  完全继承（跨 DD 门 + η 下降触发）
```

## 5. 使用与验证

```yaml
marl:
  analytical_sensing_power_enabled: true   # 先启内层解析功率
  bargaining_objective_enabled: true        # 内层目标改为归一化议价
  reward_utility_mode: bargaining           # 学习奖励 = η
```

- `tests/test_bargaining_power.py`：7 passed（T1/T2/T3 + 零 headroom 排除）。
- 相关回归（reward/analytical power/maxmin）25 passed。

## 6. D0.92-A 冒烟结果（8/8 test20，冻结 Actor，同 seed）

| 指标 | max-min LP | bargaining LP | Δ |
|---|---:|---:|---:|
| worst | **0.6363** | 0.4456 | −0.191 |
| steady | 0.7460 | **0.8263** | **+0.080** |
| η（议价值） | — | 0.3157 | — |

**判定：**

1. **bargaining 解决 steady 张力**：steady 从 0.746（max-min）跨过 0.80 到 0.826。
   这正是 advice 预言的——归一化机会公平不牺牲 easy 目标，`h_q` 分母给困难目标
   更高权重而不 starve 强目标。
2. **但牺牲 worst**：worst 从 0.636 掉到 0.446。原因是困难目标的 `D^I` 小，其
   "公平份额"（`r_q = 份额`）的**绝对** Deflection 仍小，故绝对 worst P_D 下降。
3. **结论：单一标量目标无法同时满足两个地板**。max-min 偏 worst、bargaining 偏
   steady，二者在同一条"公平—效率"谱两端。**这正好验证 advice §8/§12 的核心**：
   worst/steady 必须是**评价指标 + 升级触发**，不是单一优化目标；优化器求单一公平
   目标（η 或 min D），地板在仿真后验收，不达标则升级结构/几何层。

## 7. 下一步（D0.92-A 正式 + 层级路由）

- 正式 D0.92-A：多 seed 对比 max-min vs bargaining 的 worst/steady/QoS，确认
  steady 张力被系统化解（不依赖 reserve 补丁）。
- 层级路由改用 advice §12 的 **Pareto-improvement exhaustion**：快层 η* ≤ 0（无
  严格 Pareto 改进）→ 触发中层结构；结构后仍 η* ≤ 0 → 触发几何。不再依赖
  `0.6/0.7/0.8` 数字做路由。
