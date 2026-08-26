# 代码结构地图与重构蓝图（2026-08-25）

> 本文档把"代码逻辑混乱"转成可读的现状地图，并给出目标架构与分阶段
> 重构路线。数据来源：AST 依赖图审计（`_dep_audit.py`）+ 7 路只读子代理
> 审计（2026-08-25 DEEP_AUDIT_SYSTEM 报告 §2/§4）。本文档不改变认证语义，
> 只描述"现状是什么、该变成什么、怎么变"。

## 1. 现状量化快照

| 指标 | 值 | 含义 |
|---|---:|---|
| uav_isac 源文件 | 126（实模块 125） | 其中 6 个为空 `__init__` |
| 内部 import 边 | 213 | 模块间直接依赖 |
| import 环 | **0** | 依赖图是 DAG（比预期健康） |
| 部署主干（env_core+trainer 闭包） | **59 模块** | `environment/*` + `physical/*` + 协调核心 |
| 主干外模块 | 61 | 研究/工具链候选 |
| 零导入者模块 | 41 | 几乎全由 `tools/`、`tests/` 直接引用 |
| UTF-8 BOM 文件 | **13** | coordination 研究链；`ast.parse` 直接失败 |
| 巨型聚合器 | 2 | `env_core.py`（9535 行）、`trainer.py`（7872 行） |
| 参数声明拷贝 | 3 份 | `MARLParams` → `MAPPTrainer.__init__` → `MAPPOAgent.__init__` → `StructuredActorNetwork.__init__` |

## 2. 当前三环结构（现状）

```text
核心环（59）：部署主干
  env_core / trainer
  └─ environment: env_core, env_wrapper, action, observation(+slices), belief,
                  communication, reward, constraints, trust_manager, target, uav
  └─ physical: deflection, detection, evidence, feasibility_oracle, inner_solver,
               channel, otfs, geometry, finite_blocklength
  └─ coordination 核心: maxmin_power, hyperedge, qpd, intercept_power, pwl_pd,
                        power_staleness, capability, bargaining_power,
                        congestion_relief(默认关), dual_priced_auction(+structure),
                        dynamic_local_search
  └─ config.params（唯一参数源）

研究环（coordination 内 25+）：标 AUDIT／RESEARCH-ONLY
  bottleneck_router, certified_*（修复栈 5 件）, bounded_repetition,
  causal_*（控制器/联合计划）, certified_factor_message, digest_rendezvous,
  finite_round_hyperedge, fixing_conflict_filter, geometry_gain_predictor,
  horizon_atomic_repair, coefficient_structure, interaction_width,
  local_exchange_oracle(被核心用), scale_capability(被工具用), …

工具/审计环（tools/ + tests/ 直接引用）：
  evaluation/*（29 模块，多数只被 tools/tests 用）+ agents 四件
  （equviariant_movement_plan, p0_fixed_agent, residual_actor, tica_actor）
```

## 3. 混乱的具体来源（审计实证）

1. **两座巨型聚合器**：`env_core.py` 实际是六个子系统挤在一个类里
   （配置校验、step 编排、协议收发、运动控制、功率协调、状态序列化）；
   `trainer.py` 含 ~350 行 rollout 与 ~2700 行 eval 遥测重复流水线。
   两者之间还有第三个"影子"：`MAPPOAgent.update()` 是文档化 no-op，
   全部学习逻辑实际在 trainer。
2. **同一概念多份实现**（每份都是独立维护的死重）：
   - owner 选举 4 套：hyperedge / owner_bid_transport / permission_cut_master /
     oracle_free_candidate_locator；
   - max-min 功率求解 ≥3 套：linprog / 镜像对偶 / 列生成 / 行本地 + qpd 队列对偶；
   - 结构修复栈 2 条平行体系：部署链（congestion_relief+dual_priced_auction）vs
     研究链（certified_hierarchical_controller → certified_geometry_repair /
     dual_guided_structure_repair / horizon_atomic_repair），两份 lex 提交门语义不同；
   - log-prob 数学 2 份逐字节拷贝（evaluate_actions vs verify_old_log_prob_consistency）；
   - split-conformal 上界 8 份、Clopper-Pearson 4 份、episode bootstrap 6 份、
     steady/weak3/worst 6 份（其中 1 份行为分歧）。
3. **死代码 ~1500+ 行**：`GATEncoder`、`act_batch`、`_parse_one`、
   `nearfield_focus_targets`、`channel_margin.py` 整模块、`summarize_quantized_method`、
   `shadow_reason_counts` 等（零调用者，全树 grep 实证）。
4. **状态漂移型契约缺口**：
   - `get_state/set_state` 缺 5 个字段（_prev_obs_deque/_gru_hidden/_probe_miss_count/
     活动信道参数）→ replay 发散（本次已探针实证的 reseed 问题同族）；
   - `capability_gauge`（SLSQP）不查 `res.success`，失败静默返回末迭代/NaN；
   - `hyperedge` 容量不足静默返回全零责任行，调用方无法区分"容量不可行/未分配"；
   - `-1` 哨兵语义冲突（owner_local_physics"新鲜" vs target_invariant_transport"过期"）。
5. **依赖方向反转**：`trainer.py` 不 import `networks/mappo_agent/buffer`——它们由
   `scripts/run_mappo.py` 构造后注入。巨型聚合器反而"不声明"对核心学习组件的依赖，
   是学习链路可维护性差的直接原因。

## 4. 目标架构（重构后）

```text
config.params（唯一参数源）
  │
  ├─ 物理核心层（无状态函数）：channel, otfs, geometry, deflection, detection
  │        finite_blocklength, evidence(量化/路由), feasibility_oracle(教师)
  ├─ 环境层（有状态，只管仿真）：
  │        env_state（状态机）+ action/observation/belief/communication/reward
  │        └─ 从 env_core 拆出：ProtocolTransportLayer / MovementController /
  │             PowerAllocationLayer / StateSnapshot（dataclass 驱动的 serde）
  ├─ 协调层（每机制一个原语，单一实现）：
  │        owner_assignment（唯一入口，4 套收敛）│ maxmin_power_solve（唯一入口）
  │        lex_commit_gate（唯一入口）│ capability_Gauge（带 success 检查）
  │        ├─ 核心机制（被 env 调用）
  │        └─ 研究机制（tools/tests 调用，头注释标记，不再混入核心环）
  ├─ 学习层（trainer 只做 PPO/GAE/遥测）：
  │        policy_log_prob（唯一入口：evaluate_actions/verify 共用）
  │        rollout_step（collect_rollout/_evaluate 共用）
  │        ├─ networks（含 per-module LR 契约）
  │        └─ buffer（含 multi-env 尾行修复）
  └─ 工具/审计环（evaluation/*，只依赖上面公共层）
```

分层规则（提交时强制）：
- 下层不得 import 上层（coordination 不得 import environment；environment 不得
  import agents）；`config.params` 可被任意层读，但 `params.py` 不 import uav_isac。
- 每个"原语"一个模块一个入口函数；同概念第二份实现视为 bug。
- 巨型聚合器只保留编排；任何 >1.5k 行的文件拆分前必须能单独 import 且测试全绿。

## 5. 分阶段路线

| 阶段 | 内容 | 验收标准 | 风险 |
|---|---|---|---|
| P0 | 冻结现状（本文档 + 依赖图） | 地图与代码一致 | 无 |
| P1 | 机械清理：修 13 个 BOM；删 ~1500 行零引用死代码；迁移 KNOWN_ISSUES/OPTIMIZATION_LOG 残留引用 | `py_compile` 全过、AST 导入 125/125、1203 passed 不变 | 低 |
| P2 | 提取公共原语（行为不变）：`policy_log_prob`、`owner_assignment`、`maxmin_power_solve`、`lex_commit_gate`、`split_upper`（收敛 8 份→1） | 全量测试绿；回归测试覆盖新旧一致性（seed-451 同 seed 逐值） | 中（纯重构） |
| P3 | 拆 env_core → ProtocolTransport/Movement/MovementController/PowerAllocation/StateSnapshot；拆 trainer → rollout 核心 + eval 遥测 | 每步提交可独立 import、测试绿 | 中高（需 P2 先就位） |
| P4 | 收敛研究环：dead 研究链归档或并入核心；agent 四件迁入 coordination 抽象；`_archive` 策略统一 | 主干 59→~35 模块；研究环显式标记 | 中 |

顺序依赖：P1 必须先于 P2（清理后提取才有干净基底）；P2 必须先于 P3（先有原语再拆聚合器）。

## 6. 评审决策（2026-08-25 用户拍板）

| 评审点 | 决策 | 对路线的影响 |
|---|---|---|
| A 研究环策略 | **直接删除已证伪机制**（负结果条目保留在文档层） | P1 范围扩大：不再只是隔离，而是删除负结果机制；删除前须在 ALGORITHM_EVOLUTION/EXPERIMENT_LOG 确认负结果已有记录可追溯；被删除机制对应的 tests 同步移除 |
| B P2 第一原语 | **`policy_log_prob` 先行**（收敛 log-prob 双拷贝，训练正确性守卫） | P2 顺序：先学习层原语 → owner 选举 → max-min 求解 → lex 提交门 → split-upper 收敛 |
| C env_core 拆分 | **四子系统认可**（协议/运动/功率/快照，推荐） | P3 按四子系统拆分；每子系统独立子类/模块，先 P2 原语就位再动 |

删除已证伪机制的前置清单（审计证据）：DR1.10-C congestion_relief 负结果、README §9 R1 treewidth 关闭、S0-A sparsification FAIL、M4-D STOP_AT_CANDIDATE_LOCALIZATION、G4-B locality FAIL、G4-B2d online FAIL、G4-B2e greedy FAIL、V3 dual-priced 默认关。以上条目须先在文档有负结果记录，再删代码与对应测试。

## 6.1 执行记录（2026-08-25 决策后）

**P1a — 机械 BOM 修复（完成）**
13 个 coordination 研究链源文件剥除 UTF-8 BOM（`bounded_repetition`、`causal_hierarchical_controller`、`causal_joint_plan`、`certified_geometry_repair`、`digest_rendezvous`、`finite_round_hyperedge`、`geometry_gain_predictor`、`horizon_atomic_repair`、`owner_local_physics`、`owner_parallel_runtime`、`persistent_geometry_execution`、`priced_structure`、`protocol_fingerprint`）。验收：`compileall` 全过；全量 pytest 1203 passed（与修复前一致）。

**P1b-1 — 零调用死符号删除（已执行，验证中）**
| 文件 | 删除符号 | 行数 |
|---|---|---|
| `agents/networks.py` | `GATEncoder` 整类（2779-2834） | 56+107 |
| `agents/networks.py` | `_parse_one` 死分支（1030-1136）；`_parse_obs` 折叠为恒真分支 `_parse_one_corrected` | 107 |
| `agents/mappo_agent.py` | `act_batch`（350-385） | 36 |
| `coordination/hyperedge.py` | `nearfield_focus_targets`（353-430） | 78 |
| `evaluation/quantized_evidence_audit.py` | `summarize_quantized_method`（393-456） | 64 |
| `evaluation/shadow_horizon_router.py` | `shadow_reason_counts`（277-283，含尾部孤立 `}` 补删） | 7 |

（注：`_parse_one` 删除时发现其调用点 925-926 是恒真 `if self._use_corrected_parser` 分支——该 flag 在 416 硬置 True，故折叠为直接调用 `_parse_one_corrected`；`quantized_evidence_audit.py:410` 的残留注释同步改为指向现存配对函数。）

**P1b-2 — 已证伪机制删除（已完成 2026-08-25）**

已证伪/负结果机制（advice 014/015 负结果 + 用户拍板直接删除）：

| 机制 | 负结果证据 | 删除内容 |
|---|---|---|
| D1.10-C congestion relief（偶合感知结构修复） | 头注释 CLOSED negative：613 救回/615 退化/298 崩溃 | env_core 接线+serde 397→已删段、`config/params.py` congrelief 键段、`config/exp_800_k8q8_..._indep_relief.yaml` 启用行、`congestion_relief.py` 模块、`test_congestion_relief.py`、`tools/audit_structure_regret.py`/`audit_l3_l2_alternating.py`/`audit_horizon_joint_oracle.py` |
| V3-T4 dual-priced L2（双价结构协调） | ALGORITHM_EVOLUTION 负结果：866 轨迹分歧、lookahead/persist 两轮 A/B 均未达标 | env_core V3 接线+serde 段、`params.py` v3_* 键段、两个 v3 YAML 的 7+6 键行、`dual_priced_auction.py`/`dual_priced_structure.py`/`priced_structure.py` 模块、5 个 v3/congrelief 测试、`tools/verify_v3_t3_on_trace.py` |

验收：compileall 全过、AST 导入 121/0、326/326 config 全部可加载（load failures 0）、全量 pytest 回归通过（见上方 P1b-1 后 1203 + 本轮删除后复验）。负结果叙述保留在 ALGORITHM_EVOLUTION/EXPERIMENT_LOG 文档层，不回删文档。

**P1b-2 后续 — 待执行（预留，非本轮）**：已证伪机制删除波及的 env_core 行区间（397 行）不含的——不，已全部完成。
用户拍板"直接删除已证伪机制"。精确集合（有明确负结果记录的，**不是**所有"默认关闭"项）：
- `congestion_relief.D1.10-C`（头注释 CLOSED negative）+ `env_core:7362-7399` 的接线 gate + `params.analytical_structure_congestion_relief_enabled` 键及其 326 个 YAML 中的引用；
- V3 dual-priced L2（ALGORITHM_EVOLUTION 记录 A/B 无效负结果）：`dual_priced_auction`/`dual_priced_structure`/`priced_structure` + `env_core:7418` 接线 + 对应 config 键与 YAML。
排除项（**不删**）：task-regret Student（advice/016 活动研究主线，README §7 当前优先方向）、议价 L1（未完全证伪，仅小样本混合）、因子图/有限轮协调器（未认证 ≠ 负结果）、`bounded_repetition` 等（仅"非默认"）。每个删除项须先确认负结果已记入 ALGORITHM_EVOLUTION/EXPERIMENT_LOG 再动代码（文档纪律）。

**P2-1 — 已完成（2026-08-25）**：`policy_log_prob` 单一原语落地。
- 新增 `MAPPOAgent._movement_message_resource_log_probs(dp_mean, dp_log_std, role_logits, comm_msgs, actions_dp, actions_role, movement_action_mask, actions_comm, actions_comm_rate, actions_isac_power_raw, actions_sensing_raw, detach_head_context=False, return_components=False)` —— 原 `evaluate_actions` 第 644-778 段（7184 字节）与 `verify_old_log_prob_consistency` 第 822-894 段（3931 字节）逐字节重复的数学，现唯一实现于此。
- `evaluate_actions`：普通路径调用 helper；`return_log_prob_components` 路径调用 helper(return_components=True) 取 mov/message/rate/resource 四分量构建 head_outputs。
- `verify_old_log_prob_consistency`：no_grad 下调用同一 helper 后与存储 old_log_probs 比对（PPO 比值断言正确性由构造保证，不再依赖两份拷贝同步）。
- **验收（P2-1）**：compileall 全过；`log_prob_dp = -0.5*(` 与 `log_probs_role = torch.log_softmax(role_logits` 残留各仅 1 处（即 helper 内部）；helper def 1、调用点 3（evaluate 双分支 + verify）。verify 修复（helper 调用传 `comm_mean` 与解包一致、补回 diff/passed 计算）后**全量 pytest 1148 passed / 0 失败（56.5s）**——与 P1b-2 后基线完全一致，零行为变化。（首次 pytest 的 NameError 与假绿背景任务均已在修复+裸命令重跑后结算。）

**P2-2 起（下一轮）**：owner 选举 4 套（hyperedge / owner_bid_transport / permission_cut_master / oracle_free_candidate_locator）→ 单一原语 `owner_gain_ceiling(receiver, target, gain, budget, pair_limit)`；max-min 求解 ≥3 套 → 单一入口；lex 提交门双份 → 单一实现；split-upper 8 份 → 1 份。同步清理 P1 后残余单次脚本 `_strip_falsified.py`/`_trim_v3_yaml.py`/`_p2_dedup_logprob.py`（一次性工具，验收后删除）。

## 7. 纪律与边界

- 任何一步不改变认证语义（G2 口径、n_cpi/c_det/P_sense_max、Wilson z=1.96、
  冻结 checkpoint 行为）；需要改语义的改动必须单独预注册。
- 每步提交前：全量 `pytest`（当前 1203 passed）为最低门槛；涉及重放/一致性
  的步骤补 seed-451 同 seed 逐值回归。
- 文档纪律：本文档与 DEEP_AUDIT_SYSTEM 是工程文档；模型语义仍只写
  CURRENT_SYSTEM_MODEL.md，实验结果只写 EXPERIMENT_LOG.md。