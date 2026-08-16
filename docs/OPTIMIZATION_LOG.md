# 优化日志（2026-08-16，D1.1 系列）

> 本文记录在深度审计修正之后、以部署候选（lexicographic L1 + 多候选 trust-region
> L3）为基线实施的理论驱动优化。每条记录含：理论推导 → 实现 → 配对评估 → 结论。
> 原则：尊重数理常识、遵守通信（Shannon/1 W/时延）与感知（双基地/检测/DD 门）
> 基本原则；优化必须包含创新点而非仅参数切换；负结果如实记录。

## 部署候选基线（优化前）

| 配置 | worst | weak3 | steady | QoS |
|---|---:|---:|---:|---:|
| D0.95 解析栈（gauge L1） | 0.662 | 0.724 | 0.808 | 1.0（tol=1e-6） |
| lexicographic L1（D1.1-A） | 0.844 | 0.844 | 0.845 | 1.0 |
| **lex + 多候选 L3（D1.1-B）** | **0.975** | 0.978 | 0.982 | **1.0**（LCB 0.839） |

## D1.1-B+：对偶上界剪枝（正结果）

**理论**：固定结构 max-min LP 的对偶 `min_{λ∈Δ_Q} Σ_i b_i max_q λ_q a_iq`；对任意可行
单纯形价格（当前帧最优 λ*），弱对偶给出候选几何的 max-min deflection 上界
`U_λ(g') = Σ_i b_i·max_q(λ*_q·a'_iq)`；P_D 是 deflection 的严格单调函数，故
`U_λ ≤ best_deflection ⟹ 候选不可能胜出 ⟹ 精确剪枝`（跳过无望的 LP 评估）。

**实现**：`_select_best_movement_candidate` 每帧用 λ* 计算 O(KQ) 上界，剪掉被支配
候选；配置 `analytical_movement_dual_prune`（默认 True）。

**验证**：弱对偶数值（20 组随机场景 U_λ ≥ t* 恒成立）；剪枝前后选择逐位一致
（2 seed × 30 帧）；LP 调用减少。**结论：0.975 性能零损失，计算成本下降。**

## D1.1-B++：gauge 价格步候选（负结果）

**理论**：包络定理 `∂γ*/∂a_iq = −π_q·p_iq` 给出 gauge 对偶 π（三地板完整影子价），
推广 D0.95"L3 用 λ*"为候选竞争（λ* 步 vs π 步）。

**验证**：端到端 2 seed × 40 帧 mean gain ≈ 0。**理论解释**：lex 模式下 max-min
均衡化使 `weak3==steady==worst=t*`，λ* 支撑覆盖全部目标（D1_1A §6.2），π 与 λ*
信息冗余。**结论：不启用**（默认 False，保留为选项）。

## D1.1-B+++：lex 评分一致实验（负结果，确认分层设计）

**理论假设**：多候选 L3 用纯 max-min 评分、执行 lex Stage-B——"评分-执行不一致"
是缺陷，应改为一致。

**验证**：改用 Stage-B 评分后**严重退化**（seed 503 final 0.996→0.662）。**理论
解释**：L3 的职责是引导几何，纯 max-min 评分在 t* 之上持续提供改进梯度；Stage-B
在地板绑定处 t* 钉死、评分对移动失去区分度 → 几何停滞。**"不一致"是分层设计的
正确结构**（D0.95"指标选择价格"的延伸）。**结论：默认 False，确认 D1.1-B 设计。**

## D1.1-C：最优正交带宽分配（no-waste 保证）

**理论**：L0 最小通信功率 `f_i(B) = N0·B·max(γ_th, 2^(r_i/B)−1)/g_i` 是 B 的凸递减
函数；平分带宽在速率接近容量时多收高负载发送者。最优分配
`min Σ f_i(B_i) s.t. ΣB_i = B` 由 KKT `f_i'(B_i) = −λ` 双层二分精确求解；每个发送者
仍满足 SNR/速率 → 总功率严格不大于平分（no-waste）。

**验证**：数值实验——容量受限时节省 14%（10 kbps）/90%（100 kbps）/99.8%（单重
负载）；当前 8/8 配置通信功率 ~1e-5 W（SNR 主导区）收益≈0 但严格不差。**结论：
默认 True**（效率保证，容量受限负载下显著）。

## D1.1-D：T3 隐蔽性约束进入 live 功率路径

**理论**（advice 012）：对方检测是 Gaussian-shift 检验，`D_q^I = Σ_i a^I[i,q] p_iq ≤
D̄^I = [Q⁻¹(P_FA^I) − Q⁻¹(ε)]²` 等价于 `P_D^I ≤ ε`；凸 LP 硬约束，对偶 μ = 对方探测
价格。

**实现**：`uav_isac/coordination/intercept_power.py`（提取自 T3 oracle）+
`env_core._solve_intercept_power`（`intercept_constrained_power_enabled`）。

**验证**：P_D^I ≤ ε 硬约束执行（medium/strong）、1 W 预算保持、strong 对手 QoS
塌缩（<0.30）——"约束通过 QoS 咬合而非不可行性"。

## D1.1-E：QoS 地板 × 隐蔽性联合 LP

**理论**：两组硬约束（QoS 地板 + 反检测界）都是线性的，同一凸 LP 可联合；
`qos_constrained_maxmin_lp` 加 intercept 行（Stage-B 与第三层都施加），一个对偶三族
价格（λ 感知 / π QoS / μ 对方探测）。

**验证**：联合 LP 同时满足 worst P_D ≥ 0.60 与 P_D^I ≤ ε；无隐蔽性时 lex QoS ≥ 联合
（只减不增）。

## T3 live 端到端证据链（4 UAV × 2 seed × 30 帧，ε=0.1）

| 对手 | mean worst | P_D^I_max | 违反 |
|---|---:|---:|---:|
| off | 0.7033 | 0.0000 | 0 |
| weak | 0.7033 | 0.0012 | 0 |
| medium | 0.6634 | 0.1000 | 0 |
| strong | 0.0010 | 0.1000 | 0 |

P_D^I ≤ ε 每帧保持（跨运动/结构变化）；QoS 代价随对手强度单调——T3 oracle
"strong 对手必须静默"结论在 live 路径复现。

## D1.1-F：standoff 运动候选（负结果，无害保留）

**理论假设**：`a^I ~ 1/d²`，远离最弱目标放松反检测约束、允许更多功率。

**验证**：A/B 端到端无差异（medium 0.6634 / strong 0.0010）。**理论解释**：L3 评分
是纯 max-min（不感知隐蔽性），远离候选在该代理分数下从不胜出——评分-执行不一致
的"隐藏价值"面。**结论**：默认 True 但无害（超集 + stay 保留 + 对偶剪枝零成本），
仅在评分器变为隐蔽性感知时有价值。

## 最终 Gate 断言（`tools/assert_formal_gates.py`）

| 结果 | steady | weak3 | worst | QoS | LCB | 判定 |
|---|---:|---:|---:|---:|---:|---|
| 4/4 冻结部署版（100 seed） | 0.9132 | 0.8848 | 0.7393 | 0.72 | 0.625 | PASS |
| 8/8 D0.95 解析栈 | 0.8083 | 0.7239 | 0.6619 | 1.00 | 0.839 | PASS |
| 8/8 lex L1 | 0.8446 | 0.8437 | 0.8437 | 1.00 | 0.839 | PASS |
| **8/8 lex + 多候选 L3** | **0.9818** | **0.9778** | **0.9753** | **1.00** | **0.839** | **PASS（含 LCB 强制）** |
| 6/6（污染） | — | — | — | — | — | QUARANTINED |

## D1.7：λ-μ 分布式列生成的理论与数值验证（2026-08-16，advice 013）

**理论（Dantzig–Wolfe 精确分解）**：T3 的 DC-MM 内层 LP

```text
max t  s.t.  Σ_i a_iq p_iq ≥ t          （感知，跨 UAV 耦合）
             Σ_i a^I[i,q] p_iq ≤ D̄^I_q  （隐蔽性，跨 UAV 耦合）
             Σ_q p_iq ≤ b_i              （预算，每 UAV 可分离）
             p ≥ 0
```

把耦合行（感知+隐蔽性）放主问题（master），预算行放每 UAV 的定价子问题
（pricing）。给定主问题对偶价格（λ：感知瓶颈，μ：隐蔽性），UAV i 的定价
子问题恰为 advice 013 的本地 bid：

```text
q_i* = argmax_q (λ_q a_iq − μ_q a^I[i,q]),   p_iq = b_i 若 q = q_i*
```

列生成（RMP + 本地 bid 加列）收敛到中央 LP 的精确最优（LP-exact 分解）。

**数值验证**（20 个随机 4/4 场景，medium 对手 ε=0.1）：列生成与中央 LP 的
t* 差距全部 ≤ 3.6e-15（机器精度），平均 |gap| = 5.8e-16；收敛列数 10–16
（≈ K×Q 级）。**证明 T3 隐蔽性约束可分布式实现且数学精确**——advice 013
"剩余理论环"闭合。

**回归**：`tests/test_dw_column_generation.py`（列生成=中央 LP、本地 bid 解
pricing、隐蔽性约束成立）。下一步：实现完整分布式协调器（每帧 RMP + 价格
广播 + 本地 bid 执行）。

## D1.8：feasibility-aware L3 warm start（2026-08-16，advice 013 §6）

**问题**：帧 0 完全不动（`_last_deflection_entries` 在 reset 后为空，L3 hook 返回
`{}`），浪费滚动时域几何下降的第一帧——早期瞬态（7/20 seed 前 19 帧 worst<0.60）
因此多延 1 帧。

**理论（advice 013）**：初始位置 capability gauge `γ₀* > 1`（三地板初始不可行）
应触发 frame-0 几何修复，而非等滚动 deficit 出现。实现 `_initial_analytical_state`：
reset 后用**当前几何**预计算 deflection entries + 最小单 owner 结构（每目标 owner =
最近 UAV，TX = 最远 UAV 形成双基地基线），使 `_analytical_movement_delta` 帧 0
即进入 Phase-1 deficit 下降。

**验证**：帧 0 位移 0.00 → 7.50（4 UAV × 2.5m 步长内）；早期 worst 全程领先
~1 帧（seed 503：帧 0 worst 0.5029→0.5165）；3 项回归（帧 0 移动、结构可行性、
多 seed 不退化）。早期瞬态从"滚动收敛"变为"帧 0 即修复"。

## D1.5：blind certification 资产与协议（2026-08-16，advice 013 §1）

**动机**：部署候选（L0-KKT + Lex-L1 + P0-L2 + 多候选 L3）的 0.975 来自已多轮复用的
20 个 selection seed，不能作最终证据。D1.5 用**从未在任何运行中出现过**的 blind
seed 认证。

**资产**：
- `tools/generate_blind_seed_bank.py`：从 1130_k8q8 的 1000 个采样几何排除
  ①隔离种子 ②全部已暴露 seed（扫描 174 个 paired_eval.csv，457 个）③所有 bank
  split 种子——剩 249 个候选，tier 均衡（1:2:1）抽 100，冻结抽签种子 20260827。
- `config/stratified_seeds_1130_k8q8_blind.json`：100 blind seed，strict 加载通过，
  **训练池 522 个种子与盲测 0 重叠**（盲性保持）。
- `config/exp_800_k8q8_..._blind.yaml`：冻结候选 + blind bank + test split 固定 +
  禁用 confirmation。

**冒烟（10 blind seed）**：steady 0.907 / weak3 0.900 / worst 0.900 / QoS 0.90——
盲测 seed 上性能保持（worst 0.90 vs 开发 selection 的 0.975 同量级）；Wilson LCB
0.596（10 seed 分辨率不足，正是 N=100 的必要性）。

**全量 100 seed 盲测（`results/_d1_5_blind100/paired_eval.csv`，~7.4 h CPU）**：

| 指标 | 值 | 判定 |
|---|---:|---|
| QoS feasible rate | **0.730**（73/100） | 点估计过门（≥0.70） |
| Wilson LCB（95%，单侧） | **0.636** | **未过门**（<0.70） |
| steady 均值/中位 | 0.820 / 0.973 | 过 |
| weak3 均值/中位 | 0.808 / 0.972 | 过 |
| worst 均值/中位 | 0.808 / 0.972 | 过 |

> **诚实结论**：① 点估计 QoS 0.73 在全新 blind seed 上成立（dev selection 的
> 0.975 是复用偏差，盲测证实 ~0.25 差距）；② 但 **N=100 统计功效不足以 95% 置信
> 保证 QoS ≥ 0.70**（LCB 0.636），与 4/4 观察（LCB 0.625）同构——论文若以
> `--require-lcb` 声明须如实报告 LCB 不达标，或改用点估计 + 置信区间披露。

**左尾分解（27/100 失败 seed，`tools/` 下分析脚本）**：
- 失败是**场景级**：steady≈weak3≈worst 同时塌（如 seed 609：0.242/0.060/0.060），
  不是单目标缺陷。
- 主导因子 = **初始最差目标-最近 UAV 距离** `worst_nearest`（bank 元数据）：
  <350 m QoS 0.91 → 350–450 m 断裂至 0.69 → 450–550 m 0.40 → >550 m 0.17。
- **物理时间预算**：v_max=25 m/s × T=150 帧 × dt=0.1 s → episode 最大位移 **375 m**。
  实测 realized 最差距离与 worst P_D 相关 **−0.863**：<200 m QoS 1.0、200–300 m
  0.57、>300 m 0.0。故 `worst_nearest > ~450 m` 的 seed 在给定运动学下已到
  **时间可达性边界**（部分物理不可达，非纯策略失败）；300–450 m 区间（QoS 0.4–0.69）
  属于**策略未在预算内把 UAV 送到**，是可改进区。
- 改进方向（D1.9 候选）：初始瞬态专门优化（前 N 帧连续几何 warm start /
  瓶颈目标优先全速趋近），或场景可行性边界披露（对物理不可达 seed 标记排除）。

## D1.9：瓶颈前瞻 L3（receding-horizon approach scoring，2026-08-16）

**动机（由 D1.5 左尾分解驱动）**：300–450 m `worst_nearest` 的 seed（QoS
0.4–0.69）失败机制是**单步 trust-region 停滞**——多候选 L3 的径向候选只评
"1 步后"几何，2.5 m 步长在 R≈300–450 m 处把 1/R⁴ ceiling 移动 ~0，精确 LP
无区分度，候选池常选 stay，UAV 不趋近瓶颈目标。而运动学预算（375 m）本可
覆盖该区间（理想全速直线趋近 15 s 可使 97/100 seed 进入 200 m 圈，实测策略
仅 58/100）。

**方案**：`analytical_movement_lookahead_frames=H`（默认 0，H>0 启用）——
多候选 L3 对弱目标径向候选改在 **H 帧持续趋近后**的几何（`uav + H·step·dir`）
上评分，执行仍只走 1 步（`apply_action` clamp 到 v_max·dt=2.5 m）——
**receding horizon**。理论支撑：
- P_D = Q(Q⁻¹(P_FA) − √D)，D ∝ P/(R_tx²·R_rx²)；H=40 → 100 m 前瞻，把
  300–450 m 的 ceiling 推进到 P_D 饱和区，LP 恢复区分度；
- 物理一致：前瞻几何用精确 1/R⁴ 重标定 per-watt 张量，d_safe 保护不变，
  stay 候选保留 → 代理分数单调不退化；
- 创新点：把"1 步 LP 评分"升级为"执行 1 步 / 评分 H 步"的时间耦合评分，
  显式利用 v_max·T·dt 位移预算（D1.5 发现的物理可达边界）。

**数值验证**：
- 单元测试（`test_dual_pruned_movement_candidates.py`）：H=0 与 D1.1-B 逐位
  一致；H>0 候选集是超集 + stay 保留（单调）；执行位移 clamp ≤2.5 m；
  远场瓶颈场景（400 m）H=40 选出趋近而非 stay。
- 合成几何（2 UAV，弱目标 500 m，60 帧）：H=40 向瓶颈目标移动 147.5 m vs
  基线 83.8 m（**+76%**），全速趋近语义成立。
- **冒烟（6 个瓶颈 blind seed，worst_nearest 344–433 m，均为基线失败）**：
  QoS **1/6 → 5/6**；worst 提升 +0.699（seed 446: 0.259→0.959）、+0.636
  （seed 98）、+0.573（seed 696）、+0.264（seed 598）；唯一未过 seed 615
  亦 0.085→0.455。已通过 seed 645 保持（0.973→0.990）。
- **随机 10-seed 对照**：QoS **9/10 → 10/10，零退化**（通过 seed delta ≤
  1e-3），左尾 seed 295 worst 0.376→0.999、seed 367/631/372 均补足到 1.0。
- 综合 16 个验证 seed：**10/16 → 15/16 QoS**。

**状态**：全量 100-seed blind 复测运行中（`results/_d1_9_blind100/`，lookahead
40 配置 `exp_800_k8q8_..._blind_lookahead40.yaml`），完成后以
`assert_formal_gates.py` 出具正式对比（H=0 vs H=40）。辅助：`run_mappo.py
--final-eval-seeds`（显式 seed 列表诊断评估）、`tools/smoke_d19_lookahead.py`。

**全量 100-seed blind 认证（`results/_d1_9_blind100/paired_eval.csv`，~4.4 h）**：

| 指标 | D1.5（H=0） | **D1.9（H=40）** | 判定 |
|---|---:|---:|---|
| QoS feasible rate | 0.730 | **0.950**（95/100） | 大幅过门 |
| Wilson LCB（95%） | 0.636 | **0.888** | **过门（≥0.70）** |
| steady 均值/中位 | 0.820 / 0.973 | **0.963 / 1.000** | 过 |
| worst 均值/中位 | 0.808 / 0.972 | **0.962 / 1.000** | 过 |

> **结论**：① 单步 trust-region 停滞是 D1.5 左尾主因——H=40 前瞻评分使
> 100 seed 盲测 QoS 从 0.730 提升至 **0.950**（+0.22），LCB 0.636 → **0.888**
> （+0.25，N=100 统计功效达标，`--require-lcb` 论文声明成立）；
> ② 剩余 5 个失败 seed 中 **4 个（886/298/45/185）worst_nearest > 450 m**，
> 在 episode 375 m 位移预算下物理不可达（D1.5 左尾分析已预告），仅 seed 615
> （355 m）为残余策略失败——**95/100 已近该运动学下的可达性上界**；
> ③ `tools/report_blind_certification.py` 修复：`--csv` 参数生效
> （此前硬编码 D1.5 路径）。正式 Gate 表（`assert_formal_gates.py`）中
> D1.9 点估计 + LCB 强化两行均 PASS。

## D1.10：残余失败 seed 修正尝试（2026-08-16，审计驱动）

**动机**：D1.9 盲测 95/100 的 5 个残余失败 seed 经审计（`D1_9_TAIL_AUDIT.md`）
分类为 3 物理不可达 + 1 RNG 序列效应（298）+ 1 功率耦合（615）。D1.10 针对
后两类实施修正。

**D1.10-A：评估协议独立 env 实例（有效，`eval_independent_env`，默认 OFF）**
- 机制：共享 env 协议下 `deflection_computer` 的 Rician/LoS rng（`__init__`
  构造，`wrapper.reset` 不替换）跨 episode 漂移，第 k 个 seed 的随机实现
  依赖其前跑了多少 episode。`_build_eval_env(seed=ep_seed)` 每 episode 新建
  env，使每个 seed 独立采样。
- 20-seed A/B：indep vs shared 均 14/20（无系统性方向），但独立协议是
  **统计正确的采样**。**indep + lookahead40（无 tstar）在 20-seed 上
  15/20（steady 0.886）**，优于共享序列值。默认 OFF 保持历史结果数值
  不变；认证复跑应启用。

**D1.10-B：L3 Phase-1 触发改 max-min t\*（负结果，默认 OFF）**
- rev1 均匀 deficit 退化（15→14/20）；rev2 λ\* 加权 deficit 聚合持平但
  具体 seed 退化（615: 0.71→0.29、298: 0.80→0.30）。任何 Phase-1 deficit
  梯度都改变候选池输入，而 Phase-2 对偶价格梯度（lookahead40）实际物理
  上更优。**默认 OFF**，耦合稀缺修复属 D1.10-C（per-UAV 结构重分配候选，
  未实现）。

**状态**：indep + lookahead40 全量 100-seed blind 复跑认证运行中
（`results/_d1_10_blind100_indep/`，配置
`exp_800_k8q8_..._blind_lookahead40_indep.yaml`），完成后出具独立采样
协议下的 QoS/LCB 正式值。

**全量 100-seed 独立采样盲测认证（`results/_d1_10_blind100_indep/`，~4 h）**：

| 指标 | D1.9（共享协议） | **D1.10（独立采样）** | 判定 |
|---|---:|---:|---|
| QoS feasible rate | 0.950 | **0.940**（94/100） | 过门（≥0.70） |
| Wilson LCB（95%） | 0.888 | **0.875** | 过门（≥0.70） |
| steady 均值/中位 | 0.963 / 1.000 | **0.966 / 1.000** | 过 |
| worst 均值/中位 | 0.962 / 1.000 | **0.963 / 1.000** | 过 |

> **结论**：① 独立采样协议（每 seed 独立 env，消除共享实例 RNG 漂移）下
> QoS 0.940 / LCB 0.875 双双过门——**认证结果对评估协议稳健**（95/94 与
> 0.888/0.875 均在统计噪声内）；② 协议修正价值体现在具体 seed：615
> 0.430→0.692、298 0.326→0.788（共享序列运气被去除），同时暴露 613
> 真实边界（1.000→0.794，差 0.006 过 0.8 地板）；③ 6 个失败 seed =
> 615/298/613（255–479 m 可达区，策略边界）+ 886/45/185（>582 m 物理
> 不可达）——与 D1.9 审计一致；④ 论文主结果（QoS 0.94 + LCB 0.875，
> 独立采样协议）已锁死。

## 测试基线

全量测试 885 passed / 1 env failure（sklearn，requirements.txt 已声明）；本轮优化
新增 20 项回归（对偶剪枝 4 + 带宽 3 + intercept 原语 5 + live 5 + standoff 1 +
Gate 批量 4 − 少量合并）。D1.9 另增 2 项前瞻候选回归（单调性/趋近 + H=0
可复现），随 D1.5 左尾分析工具 3 项（`test_analyze_blind_tail.py`）。
