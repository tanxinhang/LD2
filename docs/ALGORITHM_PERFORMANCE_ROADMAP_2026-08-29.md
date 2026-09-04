# Algorithm Performance Roadmap (Theory-Driven) — 2026-08-29

> 持续优化路线图。所有优化项以四条不可妥协约束为宪章，以已确认瓶颈证据为起点，
> 以可验证门禁为终点。本文件是工作文档：每完成一项更新状态，每新增一项先填宪章表。

---

## 1. 优化宪章（四条约束，每条可操作化）

### C1 尊重基本数理常识
- 不等式方向、单位（W/bit/s/m）、凸性/凹性与 KKT 条件判定必须正确；
- 不把启发式当定理：凡声称保证（近似比/收敛/可行性）必须有推导引用（本文件定理编号 T#）；
- 不引入 inv/log(0)/NaN 路径；已有防护模式（clip/maximum/solve 而非 inv）为新代码基线。

### C2 理论推导支撑
- 每项优化必须给出：问题形式化 → 推导（可证性质）→ 算法引理/定理 → 复杂度。
- 已有可复用理论资产：`P_D=Q(Q⁻¹(P_FA)−√D)`（detection.py:83）、线性叠加 `D_q=Σ_edges a_ijq·p_ijq`（manifest）、
  capability gauge `γ*(A,b)` 关于 (A,b) 单调（capability.py:15-27 已证）、max-min LP 精确求解、
  CI 无独立性假设（belief.py）、`structure_regret` 决策保持 bit 下界 `B > log₂(1+R/Δ)`（§13-B 已实现）。

### C3 满足通信与感知基本原则（每条优化必须声明"不违反"）
1. 联合 RF 上限：每 UAV `P_isac_total=1 W`，comm+sensing 共享（env_core:985-994 区域）；
2. 物理 U2U 信道：带宽/deadline 5 ms/tx 0.25 W/FBL，诚实记账（communication.py）；
3. 感知能量时钟：`cpi_frame`，`T_sense=N·T_sym=1.024 ms`；
4. no-truth / 本地信念：任何优化不得读取 simulator truth（env_core 2835-2888 fail-closed 边界）；
5. 检测门物理可行性：`P_D≥0.80/0.70/0.60` 必须由可达 deflection 支撑（不可用不可达目标凑数）；
6. 分布式对称性：所有 UAV 同构协议，无中心化隐藏依赖。

### C4 创新性
- 每一项至少有一个"可辩护的新颖性声明"：把既有性质（如单调性）转化为运行时协议、
  把经典方法（信任域/剪枝/势函数）适配到多 UAV 双基地 ISAC 的具体结构，
  且声明必须是"该项目此前未落地"的（有 git/文档反查）。

### 宪章表（新优化项必须填写）
| 字段 | 内容 |
|---|---|
| ID / 标题 | |
| 问题与现状证据 | （文件:行 / 实测数字） |
| 理论推导（定理编号） | |
| 原则符合性声明（C3 六条逐条） | |
| 新颖性声明（C4） | |
| 验证门禁 | （等价性回归 / 性质测试 / 配对 A/B） |
| 批次 / 状态 | P0-P2 / 待实施-已实施-已证伪 |

---

## 2. 已确认瓶颈（证据，2026-08-29 实测）

| 瓶颈 | 证据 | 性质 |
|---|---|---|
| 感知 QoS 未达标（hold-action） | steady 0.549 / weak3 0.473 / worst 0.444（实测；门限 0.80/0.70/0.60） | 协议/实时闭环健康，任务性能缺口在感知层 |
| 弱几何是 6/6 失败主因 | §13-A：失败帧 `η_max 1.5–5.1` vs PASS `9–63`；teacher 0.723 vs Student 0.530 | 几何/覆盖主导，结构次之 |
| K16 计算超时 | CONVERGED 审计：优化后 P95 127–218 ms > 100 ms 门 | 计算路径（deflection 慢路径命中 canonical continuous 模式，deflection.py:320-382） |
| C7 训练不稳定 | `_c6_strict_blind100_run.log`：KL 0.12→3.27M 后 NaN（target_kl=0.02 失效） | RL 信任域失效（多头输出无共享信任域） |

**推论**：性能优化的最高杠杆在「感知层（几何/功率/融合的算法）」与「训练信任域」，计算层为次；全部必须在 no-truth 边界内。

---

## 3. 理论基座速览（优化引用的定理资产）

- **T0（检测反演）**：`D_c(q) ≥ (Q⁻¹(P_FA) − Q⁻¹(P_D_gate))²` 是目标 q 过门所需累计 deflection 的**充要闭式**（detection.py:99-122 已实现反演）。→ 一切感知优化以 T0 为预算单位。
- **T1（双基地功率-距离域）**：对链路 (i,j,q)，`a_ijq ≈ p/(R_tx²·R_rx²)·I_support·|A|²`（manifest 连续 DD 增益），联合 RF 1 W 与实际感知功率上限（0.0251 W/UAV，`K·P_sense_max` 守恒审计 §14）下，单目标可达 deflection 的**上限可由极值几何闭式给出**（见 O1 推导）。
- **T2（能力单调性）**：`γ*(A,b)` 关于 (A,b) 逐元素单调非增（capability.py:15-27）。→ 单调准入/剪枝不改变解集（O4）。
- **T3（决策保持 bit 下界）**：`B > log₂(1+R/Δ)` 保持结构决策（structure_regret，§13-B 已落地）。→ C6 通信 fail-closed 可直接转为"协议证书齐全后上线"（O6 感知侧延伸）。
- **T4（CI 无独立性假设）**：CI 信息域加权不假设估计独立（belief.py:155-237）。→ 证据融合算法可在 CI 框架内扩展。
- **T5（incumbent 复用证书，O8 已落地+锁定）**：`replicated_local_row_maxmin_power`（maxmin_power.py:354）的 reuse/热启动不是无证书启发式：
  1. **构造性可行性**：held 分配按行归一化后按**当前** budget 缩放（:592-600）→ 非负且行和==budget，可行性由构造保证（无需重解）；
  2. **ε-最优性证书**：`lower = min_q Σ a·p`（主值）与 `upper = Σ b·max_q(price·a)`（对偶上界，:601-612）满足弱对偶 `upper ≥ OPT ≥ lower`；复用判定以 `(upper−lower)/upper ≤ reuse_tolerance` 为门（:613-619）→ 复用解是**带可计算 gap 证书的 ε-最优**；
  3. **同几何不变性**：视图一致时组装==集中 LP 最优（:379-383）；gap=0 复用即该最优。
  锁定：`tests/test_maxmin_power_reuse_theorem.py`（4/4；含默认路径 tol=0 逐位不变——零语义）。→ O8 从"机制存在"升级为"有定理+测试"。
- **T6（可达性天花板，O1 已落地+锁定）**：给定公共增益视图 `A`（K,Q）与联合 RF 感知预算 `b`（K,），目标 q 的**全队最大 deflection** 为 `D_q^max = Σ_i A[i,q]·b_i`（线性叠加、全预算投 q 的上限）；T0 门限 `D_c(q) = (Q⁻¹(P_FA)−Q⁻¹(P_D))²`。**q 可达 iff `D_q^max ≥ D_c(q)`**。两个被性质测试锁定的推论：
  1. **单调剪枝（T2 化）**：不可达 q 对任何功率分配（`Σ_i p_iq ≤ Σ_i b_i`）的 deflection ≤ `D_q^max < D_c(q)` → **排除不可达目标不丢任何可行解**（准入剪枝正确性）；
  2. **视图单调**：`A1 ≤ A2` 逐元素 ⟹ 不可达集单调收缩 → 准入集帧一致（无振荡）。
  锁定：`tests/test_reachability_ceiling.py`（6/6：T0 一致性、充要闭式、不可达永不过门、视图单调、最差 deficit 供 O5、no-truth 签名锁）+ 零 RNG（确定性）。→ O1 的证书层完成；超边准入接线（默认 OFF 旗标）为下一实施步。

---

## 4. 优化项路线图

### P0 零语义微优化（立即做，不改变任何输出；冻结前并入）

**O2 deflection 慢路径向量化**（对齐 advice/002 2.1/2.2）
- 理论：批量与标量是同一函数的两种调度，**等价性可逐点验证**（`g_dd_all`/`d_eff_all` 已存在于快速路径 deflection.py:289-318，慢路径 320-382 复用同一 `compute_dd_phys_gain_batch`）。
- 原则：不读取新输入、不改记账 → C3 六条天然不违反。
- 新颖性：低（性能对齐）；价值：canonical continuous 模式每帧都走慢路径。
- 门禁：`test_physics_closure_c0c1.py`（含批/标量等价断言）+ 全量 1314 回归。

**O7 缓存 `Q_inverse(P_FA)`**（advice/002 2.3）
- 理论：`Q⁻¹(P_FA)` 是全局常量，缓存是纯函数 memoization，结果逐位相同。
- 门禁：数值逐位断言 + 全量回归。

**O8 maxmin 对偶/热启动传递**（低风险）
- 理论：LP 基解热启动不改变最优值（对偶可行→原可行传递已有 `distributed_replicated_power_reuse` 雏形）；
  限定在同帧功率缓存的**对偶价格作为下一帧初始点**，正确性由"可行性保持不变→最优值不变或更优"（需 T5 化）。
- 注意：仅在"诊断身份"下评估；改变数值路径但不改收敛值，等价性按 LP 最优值容差验证。

### P1 理论驱动大项（算法级，改变口径 → 冻结后以非冻结/冻结+1 身份评估）

**O1 双基地物理可达域 → 结构层单调准入（感知 QoS 高杠杆，理论+创新为主）**
- 问题：6/6 失败帧弱几何（η_max 1.5-5.1）、hold-action worst 0.444——部分目标物理不可达（§13-A 3 物理不可达）。
- 推导（T1 细化）：给定目标 q 的最近 Tx/Rx 双基地距离 `(R_tx, R_rx)`，单 UAV 对 q 的 deflection 贡献上限
  `a_max(q) ≤ c·P_sense_max·I·|A|² / (R_tx²·R_rx²)`；门限由 T0 给出 `D_c(q)`；则 **q 可被全队覆盖的充要条件**
  `Σ_{(i,j)∈E_q} a_ijq^(max) ≥ D_c(q)` 可在结构生成前以闭式检查。
- 落地：把该检查作为 hyperedge candidate 的**准入剪枝（单调，T2 保证不丢可行解）** + 暴露"最不可达目标"作为
  运动层势函数输入（接 O5）。在 manifest 冻结身份内**不改变协议语义**，仅增加准入/优先级（默认 OFF 旗标，
  冻结身份保持现状，A/B 评估后再冻结+1）。
- 原则：只用本地 belief 位置 + 物理参数；不读 truth；诚实记账不变。
- 新颖性：双基地椭圆可达域的**结构生成前准入 + 单调剪枝证书**是该体系未落地的组合（git 反查无此类模块）。
- 门禁：A/B（同 seed 配对，independent-env）；性质测试（剪枝不减可行性：对随机几何枚举全/剪枝解集相等）。

**O3 多头解耦 KL 信任域（训练稳定，C7）**
- 问题：C7 KL 0.12→3.27M 爆炸；结构化 actor 输出头（dp/role/comm/token/resource）共享单一 target_kl。
- 推导：总策略 KL 对独立条件分布可加 `D_KL(π‖π_old) = Σ_head w_h·D_KL(π_h‖π_old,h)`（按头条件独立时）→
  将 target_kl 预算按头分配（自适应权重），逐头做 clip/投影；结合已有 R2 事务回滚，理论保证每头 KL 受控。
- 原则：纯训练侧机制（不改推理解释、不改变冻结在线身份）；C3 不适用但需声明不触碰评估路径。
- 新颖性：多头结构化行动器的**按头自适应 KL 预算分配**（经典 TRPO 信任域的多头分解适配，本项目未落地）。
- 门禁：单头注入 KL 尖峰的合成测试（KL 不超限）+ 1-episode 训练稳定性冒烟 + 与现有 PPO 回归（奖励不降）。
- 风险：若头间相关性强（共享 encoder），按头投影的联合界需假设条件独立——推导中显式声明假设并测其敏感性。

**O4 capability 单调性 → active-set 剪枝证书（计算性能，K16 方向）**
- 问题：K16 结构/功率求解超 100 ms 门；ai_candidate_screener/active-set 已有雏形但无正确性证书。
- 推导（T2 化）：候选集剪枝若满足"被剪元素对任何可行解都不是唯一支撑"，则剪枝不改变最优值；
  由 `γ*(A,b)` 单调性导出充分条件（保留使 γ* 最紧的下界支撑集）。
- 落地：`_restricted_column_master`/`_restricted_reserve_master`（maxmin_power.py）的**运行时准入**接入单调剪枝。
- 新颖性：把文档已证的单调性**转成在线剪枝证书**（该项目未落地）。
- 门禁：K∈{8,12,16} 剪枝 vs 全量最优值等价（容差 1e-6）+ P95 时间下降 ≥35%。

**O5 运动层势函数前瞻（弱几何根治的方向，理论最重）**
- 问题：worst 目标弱几何（η_max 低）→ 运动层需主动改善瓶颈目标的 TX/RX 距离。
- 推导：定义势函数 `Φ_q = (R_tx(q)² + R_rx(q)²)` 或 `R_tx(q)·R_rx(q)`（双基地距离乘积，对应 a∝(R_txR_rx)⁻²）；
  每次移动候选使瓶颈目标 `Φ` 下降——证明**存在性**：若某 UAV 沿"指向瓶颈目标的角色路径"移动则 Φ 严格下降
  （双基地椭圆上梯度方向）；收敛性：Φ 非负有下界 + 单调下降（Lyapunov 类论证），有限步到局部极小。
- 落地：扩展现有 `distributed_bistatic_bottleneck_movement`（D1.9 前瞻启发式）为**带 Φ 下降检验的接受规则**（拒绝上升步）。
- 新颖性：双基地"距离乘积势函数 + 接受规则"（替代纯距离/几何启发式），项目未落地形态。
- 门禁：合成几何下 Φ 单调非增断言 + 6/6 失败 seed 的 worst η_max 配对 A/B。

**O6 感知证据量化 → 门保持界（通信-感知交叉，直接回应 C6 证书缺口）**
- 问题：C6 决策充分通信因缺证书 fail-closed；感知证据量化（confidence bits）对下行 P_D 门的影响无界。
- 推导：量化 LLR/deflection 反馈后融合 deflection `D_q^(quan)` 与真实 `D_q` 的偏差界
  `|D_q^(quan)−D_q| ≤ Δ_q(bits)`（由量化器网格与 CI 权重导出）；再由 T0 反演得 **门保持条件**
  `D_q ≥ D_c(q) + Δ_q(bits)` ⟹ 量化后仍过门（该界是"感知侧决策保持"证书，与 structure_regret 的通信侧配对）。
- 落地：作为现有 fail-closed 协议的**首个科学证书候选**（先文档化推导与数值验证，不直接上线）。
- 新颖性：感知量化"门保持界"证书——C6 缺的正是此类；若推导成立，可把 fail-closed 从"全关"放宽为"有界场景开启"。
- 门禁：数值界验证（蒙特卡洛 vs 界）+ 现有 quantized_evidence 测试扩展。

### P2 治理/一致性（支撑可信优化而非性能本身）

- **R14 哨兵统一**、**R15 QoS 门限单一来源**：按 §13 已定方案实施；保证优化评估的指标口径一致（否则 A/B 无可比性）。

---

## 5. 验证协议（每个优化项的强制步骤）

1. **等价性回归**（零语义项）：与现状逐位/容差断言，全量 1314 通过。
2. **性质测试**（算法项）：推导中每条可证性质至少一个专门测试（单调/剪枝不减/Φ 非增/KL 上限/界成立）。
3. **配对 A/B**（算法项）：同 seed 配对、independent-env、paired bootstrap（复用 eval_independent_env 与
   `assert_formal_gates`/`summarize_paired_horizon_confirmation` 方法论）；先 10-20 seed 冒烟、再决定全量。
4. **冻结身份保护**：改口径项默认 OFF 旗标 + 诊断身份评估；冻结 manifest 的 `effective_sha256` 在优化前后
   必须不变（除非显式冻结+1 流程）；`check_system_identity.py --strict` 保持 0 hard-pin 失败（除 dirty 位）。
5. **理论审计**：每项最终结论必须引用定理编号；"经验观察到提升"不能作为宣称（可作线索，不可作结论）。

## 6. 执行时序

1. **现在（本批次）**：P0 三项（O2/O7/O8）——零语义、可立即并入下一次冻结提交；R15 门限单一来源先行
   （为 A/B 提供统一指标口径）。
2. **冻结后（P1-第一批）**：O1 可行域准入 + O5 势函数（同属感知-几何方向，互为输入）+ 对应配对 A/B。
3. **P1-第二批**：O3（训练信任域）在训练身份上验证；O4（剪枝证书）在计算身份上验证。
4. **P1-第三批**：O6 证书推导与数值验证（先文档后代码）。
5. **每个批次后**：更新本路线图状态表 + 审计文档 §16（结果/证伪记录，沿用"零动作探针/负结果如实记录"惯例）。

---

## 7. 禁止事项（与宪章反向约束）

- 不通过放宽门限、降低 P_FA、抬高功率上限、引入 truth 或中心化来"刷"QoS——违反 C3 且违背认证意义。
- 不把"非冻结身份下的提升"宣称成正式结果（provenance 与 `formal_result_eligible` 契约）。
- 不在没有配对 A/B 或性质测试时宣称优化有效（违反 C2）。
- 不改动冻结 manifest 的 `effective_sha256` 而不走冻结+1 显式流程。

---

## 8. 状态表

| ID | 标题 | 批次 | 状态 | 证据 | 备注 |
|---|---|---|---|---|---|
| O7 | 缓存 `Q_inverse` 标量路径 | P0 | **已实施 ✓** | 性质测试 `tests/test_q_inverse_cache.py` 4/4；基准 200k 调用 520→379 ms（1.37x，命中 199999/200000）；全量 **1318 passed / 46.11s** | 零语义 memoization；详见审计文档 §16.1 |
| O2 | deflection 慢路径向量化（含 canonical continuous 命中） | P0 | **已实施 ✓** | 对拍 `tests/test_deflection_vectorized_equiv.py` 5/5（含随机几何/sensing 矩阵/out-of-support/binary/Swerling）；基准 K=Q=8：批量 0.99 vs 标量 1.61 ms/frame（**1.62x**）；全量 **1323 passed / 46.75s** | canonical（continuous+report_link+非 Swerling）走批量；binary/Swerling 保留原标量（RNG 顺序）；d_raw 与标量**同序表达式**（1-2 ULP 等价，不宣称逐位——C1） |
| O8 | maxmin 对偶/热启动传递 | P0 | **已实施 ✓**（T5 化） | 性质测试 `tests/test_maxmin_power_reuse_theorem.py` 4/4（构造性可行性保持、ε-最优 gap 证书、同几何最优不变、默认路径 tol=0 逐位一致）；T5 定理已入 §3 | 机制已存在（`replicated_local_row_maxmin_power` 的 incumbent/对偶复用）；本轮将其**理论化+锁定**（C1/C2）；零语义（无源码改动，默认路径不变）；开启策略与整帧收益验证列 O4 关联 |
| O1 | 双基地物理可达域 → 结构层单调准入 | P1 | **部分实施 ✓（证书层完成）** | 新模块 `uav_isac/physical/reachable_deflection.py`（`target_reachability_ceiling`/`admission_mask_and_worst_deficit`）；T6 定理入 §3；性质测试 `tests/test_reachability_ceiling.py` 6/6（T0 一致性、充要闭式、不可达永不过门、视图单调、最差 deficit 供 O5、no-truth 签名锁+零 RNG）；全量 **1350 passed / 47.74s** | 证书函数+定理+测试已闭环；**超边准入接线（默认 OFF 旗标 `hyperedge_reachability_admission_enabled=false`）+ 诊断身份 A/B** 为下一实施步 |
| O3 | 多头解耦 KL 信任域 | P1 | 待实施 | — | 依赖 R2 事务回滚；条件独立假设需显式声明与敏感性测试 |
| O4 | capability 单调性 → active-set 剪枝证书 | P1 | 待实施 | — | T2 化；K{8,12,16} 等价验证 |
| O5 | 运动层势函数前瞻（Φ 下降接受规则） | P1 | 待实施 | — | Lyapunov 类收敛论证；与 O1 互为输入 |
| O6 | 感知证据量化 → 门保持界证书 | P1 | 待实施 | — | 与 structure_regret 决策保持界配对；先文档后代码 |
| R14 | 哨兵统一（缺失/过期/无分配显式枚举） | P2 | **已实施 ✓** | 新模块 `utils/sentinels.py`（FRAME_NEVER/FRAME_NOT_APPLICABLE/AGE_EXPIRED/AGE_NO_CACHE/TARGET_INDEX_NONE/OWNER_INDEX_NONE/VALUE_UNBOUNDED_*）；env_core（时间哨兵 11 处+movement 4 处）、persistent（AGE_EXPIRED×4）、owner（AGE_NO_CACHE）、evidence 系（OWNER_INDEX_NONE×5）全部具名；性质测试 `tests/test_sentinel_semantics.py` 10/10（值保持=零语义、同值异域行为方向、时间契约、源码锁定 6 模块） | 数值全部不变（-1e8/-1e9/-1 保持）；-1 三义显式命名隔离；历史数值混用记录为"待统一（serde 两侧需同步，行为改变）"—后续项 |\n| R15 | QoS 门限单一来源（验收门） | P2 | **已实施 ✓** | `params.py: qos_acceptance_floors + qos_acceptance_wilson_lcb_floor`；`assert_gate_thresholds.py` 的 `acceptance_floors_from_config`（fail-loud）+ `medium_gate_checks/assert_medium_gate/assert_gate_from_csv` floors 参数化；`report_blind_certification.py --config` 读取；性质测试 `tests/test_qos_gate_provenance.py` 6/6（默认==params、三源数值一致锁定、config 驱动门、语义独立、fail-loud）；全量 **1330 passed / 46.56s** | 验收门单源化；奖励/通信地板保持独立语义、数值一致由测试锁定；影子默认清理列 backlog（P3 卫生） |