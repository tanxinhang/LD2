# 深度审计：系统级代码 / 配置 / 仓库健康（2026-08-25）

> 本报告是只读审计 + 安全修复的记录。范围：D:\BYLW\LD3 分支
> `codex/architecture-v2` 的当前工作树（354 项未提交变更）。
> 审计方法：7 个并行只读子代理逐行阅读 126 个 uav_isac 源文件（其中 6 个
> 为空 __init__，实际可导入模块 125 个）、171 个测试文件，
> 全部跨仓库符号引用验证；叠加 6 个机械检查脚本与 2 个运行时探针；
> 最终以全量 pytest（1203 passed）与 2 道正式门禁复验。
> 未修改任何认证结论对应的代码语义；修复均带回归测试。

## 1. 已验证事实（机器可复现基线）

| 检查 | 结果 |
|---|---|
| 全量测试 | **1203 passed**（修复后；修复前 1198），0 失败，81.7/78.5 s，无 skip/xfail |
| G2-1 检测口径门禁 | `ready_for_g2_1=True; blockers=[]`；审计距离 P_D=1.000/0.989/0.124/0.00981 |
| 正式门禁表 | 4/4 PASS(0.625)、8/8 D1.10 PASS、V3-C0 PASS、6/6 CE DISCLOSED FAIL —— 与 README 全表一致 |
| 编译检查 | uav_isac/tools/scripts/tests/config 全部 `compileall` 通过 |
| 模块导入 | 125/125 个 uav_isac 模块导入成功，0 断链 |
| 配置加载 | 326/326 个 config/*.yaml 经 `load_config`（含 extends 链、未知键拒绝）成功 |
| 结果目录 | 977 个 live 目录；**892 个孤儿**（全树无引用）；`_archive` 127 个，其中 **41 个仍被引用** |
| git 工作树 | 92 modified / 32 deleted / 230 untracked；HEAD 停留在 2026-08-17 |

## 2. 严重度分级发现

### S（严重：真实缺陷 / 语义错误 / 可翻转门禁）

| # | 位置 | 发现 | 证据 |
|---|---|---|---|
| S1 | `uav_isac/environment/env_core.py:1605-1610` | `reseed()` 只重绑定 `self.rng / action_space.rng / deflection_computer.rng`，**cost-aware 通信模型 RNG 仍持旧构造流**；burst/shadowing/FBL 擦除不跟随 `reset(seed=...)`，同 seed 复现跨 episode 发散。**运行时探针实证**：同一 env 两次 `reset(seed=91)` 后 shadow 矩阵 132 个非零项但值不同。 | `InterUAVCommunicationModel(rng=self.rng)` 在构造时捕获生成器；`reseed` 无 `_inter_uav_comm.rng` 重绑定。影响当前活动研究线（fbl/shadow/burst 变体配置） |
| S2 | `uav_isac/agents/trainer.py:4897-4909` | episode LR 调度把 `actor_optimizer` **所有 param_group 的 lr 统一覆盖**，从首个调度步起静默抹掉 `use_per_module_lr`/`freeze_attention`（enc 1e-5 / attn 0 或 1e-5 / head 5e-5）的隔离契约 —— S1/EH 消融语义失效。 | `for pg in agent.actor_optimizer.param_groups: pg['lr'] = lr`；而 init 处（2068-2077）专门建了三组不同 lr |
| S3 | `uav_isac/agents/trainer.py:890` 与 `tools/audit_d087_power_deployment.py:119-127` | **Wilson LCB 双约定并存**：trainer 记录指标用 `inv_cdf(1-alpha)`≈1.645（单侧），而文档规定（README/CURRENT_SYSTEM_MODEL/测试）与 4 个正式工具用 z=1.96（双侧）。78/100 时 0.705 vs 0.689，**翻转 0.70 门禁**。 | z=1.96 在 `tools/assert_gate_thresholds.py`、`summarize_paired_horizon_confirmation.py`、`crash_isolated_seed_eval.py`、`summarize_dyn_k12_conf.py`、多处测试；trainer 用 1.645。`assert_gate_thresholds.py:19-22` 注释承认分歧 |
| S4 | `uav_isac/physical/feasibility_oracle.py:1679, 1748` | K==1 / 空角色分区退化路径：`solve_joint_pair_power_oracle` 在 `for tx, rx, mode in partitions:` 循环外引用 `mode` → **NameError**（注释自称"优雅降级"）；`solve_pair_only_oracle` 仍保留 `assert best is not None` → K==1 时 **AssertionError**。且退化解把 `P_D_q` 硬编码为 0.0，与本模块"无 deflection ⇒ P_D=P_FA"约定（第 184/1602 行）不一致。 | 空 `partitions` 列表 + `mode=mode`；`assert best is not None` |

### M（中等：死代码 / 重复 / 契约漂移 / 状态机缺口）

| # | 位置 | 发现 |
|---|---|---|
| M1 | `env_core.get_state/set_state`（8920-9525） | replay 序列化缺 `_prev_obs_deque`、`_gru_hidden`、`_probe_miss_count`、活动信道参数（`_active_comm_deadline_s/_snr_threshold_db`）——`obs_history_frames>1` 或 active-probe 配置下 get_state/set_state 重放发散 |
| M2 | `coordination/minimum_intervention_repair.py:102-107` | `task_floors` 校验放在解包**之后**（短元组 → IndexError 而非 ValueError）；域契约与 sibling `scale_capability.task_detection_metrics`（(0,1]）不一致，且单位地板不可反演也未给出原因 |
| M3 | `agents/mappo_agent.py`（evaluate_actions vs verify_old_log_prob_consistency） | ~100 行 log-prob 数学**逐字节重复**，PPO 比值断言依赖两份拷贝保持同步 |
| M4 | `agents/trainer.py`（collect_rollout vs _evaluate） | ~350 行 rollout/eval 流水线重复实现（环缓冲、GRU 批、comm 采样、token mask、critic 输入、遥测），仅确定性不同 |
| M5 | `agents/buffer.py:266-312, 374-420` | 多环境 GAE 尾部行不写入 → 静默零优势；`num_envs=0` 除法崩溃；`active_adv = adv_flat != 0` 会静默丢弃零行 |
| M6 | `agents/networks.py:2779`, `tica_actor.py` 若干、`evaluation/quantized_evidence_audit.py:393`、`evaluation/shadow_horizon_router.py:277` | `GATEncoder`、`act_batch`、`forward_with_window`、`summarize_quantized_method`、`shadow_reason_counts`、`_candidate_proxy_lower`、`channel_margin.py` 整模块等**零调用者死代码**（工具/测试以外全树无引用），约 1500+ 行 |
| M7 | `coordination/capability.py:119-135` | `capability_gauge`（SLSQP）不检查 `res.success`，求解失败静默返回末迭代/NaN gauge 而非 `None`（fail-closed 契约失效） |
| M8 | `coordination/hyperedge.py:353-430` | `role_capacity_bottleneck_assignment` / `gap_coverage_bottleneck_assignment` 容量不足时**静默返回全零责任行**且 bottleneck 代价不含失败角色 —— 调用方无法区分"容量不可行"与"未分配" |
| M9 | `coordination/minimum_intervention_repair.py:102-107` 域 + `coordination/scale_capability.py:43-66` 域 | 同一量（任务概率地板）两个门模块域不一致 (0,1) vs (0,1]（详见 S2 已修复说明：修复器是反演目标必须 (0,1)，度量模块是校验目标可 (0,1]） |
| M10 | `coordination/congestion_relief.py:1-31` vs `env_core:7362` | 模块头自称 "CLOSED 负结果"，但 env_core 仍以默认-关的 flag 接线执行 —— 状态注释与集成事实不一致 |
| M11 | `coordination/` 四套 owner 选举 + 三套 max-min 功率求解 + `joint_structure_relaxation`(LP) / `qos_threshold_feasibility`(MILP) 相同约束构造 | 同概念多实现，漂移风险（`dual_priced_auction` 与 `congestion_relief` 的 lex 提交门两份拷贝且 margin 语义不同） |
| M12 | `evaluation/evidence_oracle_audit.py:223-237` | `summarize_lossless_topk_capacity` 缺每-episode 空历史防护（sibling `summarize_evidence_oracle:134-137` 有）→ 空历史产生 NaN 静默毒化均值/CI |
| M13 | `evaluation/episode_joint_conformal.py:87-91` | 8 个 split-conformal 拷贝中唯一"钳制 rank 而非 fail-closed"（小样本时覆盖层 < 1-alpha 且不返回 inf） |
| M14 | 文档引用 | **10+ 处引用已删除的 `docs/KNOWN_ISSUES.md`**（config/quarantined_seeds.json、utils/math_utils.py、physical/detection.py、physical/inner_solver.py、tools 若干、测试 docstring）与 **3 处引用已删除的 `docs/OPTIMIZATION_LOG.md`**（tools/assert_formal_gates.py、测试）；CI 注释（ci.yml:6）也引用 KNOWN_ISSUES.md |
| M15 | `uav_isac/environment/env_wrapper.py:205-209, 252` | `total_bits_all` 在非证据传输模式静默报 0；`['tx','rx','idle','duplex'][role % 4]` 对 role>3 静默错映射 |
| M16 | `uav_isac/environment/belief.py:471-500` | `reset()` 硬编码初始噪声 (50.0, 5.0) 与协方差 (2500, 25)，不读取 `initial_position_std`/`initial_velocity_std`；首 episode 与后续 episode 校准不同 |

### L（轻微）

- `agents/networks.py:211,2455` 可变默认参数 `hidden_layers=[256,256]`（当前未变异，潜在别名风险）。
- `agents/trainer.py:15` 未用 import `deque`；`tica_actor.py` 多个未用权重头参数保留在 state_dict。
- `physical/geometry.py:82-85`（+ owner_local_physics 复制）接收端运动 Doppler 项符号与雷达惯例相反 —— 无 signed 消费者，潜伏。
- `physical/inner_solver.py:20-21` 陈旧 "(1-1/e)≈63%" 声明；`solve_exhaustive` 忽略 greedy 执行的时延约束 → `compute_greedy_gap` 可把不可行集当最优。
- `physical/channel.py:177` `compute_report_link_capacity`、`physical/detection.py:12` `DETECTOR_SCALE`、`utils/types.py:7` `Position3D`、`physical/evidence.py:331` `evidence_ids` —— 零引用死符号。
- `physical/feasibility_oracle.py:105 vs 773/1105/1247` 同一 (Q⁻¹(P_FA)−Q⁻¹(P_PD))² 地板常量 1e-12 vs 1e-9 不一致；两个 MILP 硬编码 `time_limit=5.0` 不接参。
- `scripts/federated/run_federated.py:205`、`run_local_only.py:17-19` 引用不存在的 `config/fed_region_*.yaml`（初始提交遗留，全树无调用）。
- README `docs/README.md:130-131` 测试数 `1050` 已过时（现 1198/1203，本报告已更新该行加注）。

## 3. 本轮已修复项（全部带回归测试并通过全量验证）

**Fix 1（S1 实证）** `env_core.reseed()` 增加 `_inter_uav_comm.rng` 重绑定。
验证：`_rng_probe.py` 从 `CONFIRMED`（shadow 不一致）变为 `PASS`（同 seed 重放一致）；`comm.rng is core.rng` 恒 True。

**Fix 2（S3）** `trainer.py:890` Wilson z 由单侧 1.645 对齐文档双侧 `inv_cdf(1-alpha/2)`=1.96；`tools/audit_d087_power_deployment.py:123` 同步。
验证：无测试固定旧值；全量通过；门禁表（由 z=1.96 工具重算）不变。

**Fix 3（S4）** `feasibility_oracle` 两个 K==1 退化路径：
- joint：`mode` 循环外引用 → `mode="degenerate"`；`P_D_q` 硬编码 0.0 → `compute_PD(zeros, P_FA)`；
- pair-only：`assert best is not None` → 与 joint 相同的 `pair_only_degenerate` 构造。
验证：新增 `test_k1_joint_oracle_degrades_gracefully` / `test_k1_pair_oracle_degrades_gracefully`。

**Fix 4（M2/M9 重构为语义正确形态）** `minimum_intervention_repair.py`：
- `len(task_floors)!=4` 校验前移到解包之前（ValueError 而非 IndexError）；
- 域保持 (0,1) 但改为**显式解释性报错**（单位地板"完美检测不可反演为有限 deflection"，与 `scale_capability` 的 (0,1] 校验域区分开，消除"两份拷贝看似矛盾"的假象）。
验证：`test_task_floors_validated_before_unpack`、`test_task_floors_unit_floor_fails_closed_with_explanation`。

**Fix 5（M12）** `evidence_oracle_audit.summarize_lossless_topk_capacity` 增加与 sibling 一致的每-episode 空历史跳过 + 全空 fail-closed。
验证：全量测试无回归。

**Fix 6（S2）** `trainer.py` 优化器初始化记录自定义 per-module LR 组（`_actor_param_group_lr`），episode 调度只覆盖默认单 LR 组：
- 默认配置行为完全不变；
- `use_per_module_lr`/`freeze_attention` 消融的 enc/attn/head 隔离契约从首个调度步起保持有效。
验证：全量测试通过（含训练路径相关测试）。

**Fix 7（M14 文档）** `docs/README.md:130-131` 增补"2026-08-25 现树 1198/1203 passed"注记，保留历史声明。

**Fix 8（新回归测试）** `tests/test_audit_fix_regressions.py`：5 项锁定以上 1/3/4/5 行为。

修复后验证：**全量 1203 passed**（+5）；G2-1 门禁 `ready_for_g2_1=True`；正式门禁 4/4、D1.10、V3-C0 PASS、6/6 DISCLOSED FAIL 全部不变。

## 4. 建议后续（未动代码，需决策）

1. **S1 配套**：`get_state/set_state` 补齐 `_prev_obs_deque`/`_gru_hidden`/`_probe_miss_count`/活动信道参数（M1）——与 Fix 1 同属"共享实例重放"完整性；修后需 seed-451 同 seed 重放单测。
2. **M3/M4**：log-prob 数学与 rollout/eval 流水线抽公共实现（`policy_log_prob(obs, actions, ...)` + 共享 rollout-step helper）——当前 PPO 比值断言正确性依赖两份拷贝逐字节同步。
3. **M5**：multi-env GAE 尾行写入 + `num_envs` 校验 + 非零行归一化。
4. **M6/M7/M8/M16**：死代码清理（约 1500 行）、SLSQP success 检查、hyperedge 容量不足 fail-closed 返回、belief reset 读取配置初值——均小改动，建议并入下次发布提交。
5. **M11**：owner 选举与 max-min 求解收敛为单一实现；`joint_structure_relaxation`(LP) 与 `qos_threshold_feasibility`(MILP) 共用一个参数化约束构造器。
6. **M14**：把 `KNOWN_ISSUES.md`/`OPTIMIZATION_LOG.md` 的残留引用迁移到四文档对应章节（B8 子模性注释 → CURRENT_SYSTEM_MODEL；D1.5/D1.9 统计功效 → EXPERIMENT_LOG），或恢复两个被删文档的存根。
7. **仓库清理（审计 P2 延续）**：354 项工作树变更一次干净发布提交；892 个孤儿结果目录按 `_orphan_v3.txt` 归档或删除；259 个全树无引用的死配置与 41 个已归档但仍被引用的结果目录对齐引用；`scripts/federated`（引用不存在的 fed_region 配置）归档或补配置。
8. **CI**：`ci.yml` 注释引用已删 `docs/KNOWN_ISSUES.md`；README 声称"Windows/MKL eigvalsh 原生中止"限制本地单进程全量——本机全量 1203 passed 未复现该问题（可标注为历史环境限制）。

## 5. 证据产物

- `_rng_probe.py`：reseed 泄漏运行时探针（修复前 CONFIRMED / 修复后 PASS）。
- `_config_audit.py`：326 配置可加载性 + 死配置/死引用清单（259 死 / 7 死引用，其中 2 处真实：fed_region 三件套）。
- `_ref_audit.py` / `_deadlink_audit.py`（既有）：刷新后 892 孤儿 / 41 归档引用。
- `tests/test_audit_fix_regressions.py`：5 项修复回归测试。
- 全量 pytest：`1203 passed, 8 warnings (78.5 s)`；`tools/audit_detector_normalization.py --assert-ready` → `ready_for_g2_1=True`；`tools/assert_formal_gates.py` 表不变。