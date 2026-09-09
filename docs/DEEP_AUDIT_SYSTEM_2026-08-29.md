# Deep Audit of the LD3 System — 2026-08-29

> 深度审计报告（最终稿 2026-08-29）。
> 审计对象：D:\BYLW\LD3 多无人机分布式 ISAC 仿真系统（uav_isac 包 + config 实验矩阵 + results ≈12.9 GiB / 6326 文件实验树）。
> 方法：6 个并行审计子代理 + 全量 pytest 实测 + 关键断言一手复现验证 + git 状态核验 + 独立扫描脚本。
> 注：配置与文档两个子代理未在时限内返回，其核心结论由一手核验/独立扫描覆盖；行文不依赖其未返回部分。

---

## 0. 执行摘要

本次深度审计（2026-08-29）覆盖代码/测试/配置/依赖/文档/数据六个维度：6 个并行审计子代理 + 全量 pytest 实测（**1303 passed / 8 warnings / 63.87s**）+ 关键断言一手复现 + git 状态核验。

**总体判断：测试与配置基础设施健康（严格加载 341/341 全绿、0 import 环、全量测试通过），但认证主路径当前被阻塞，工作树处于"重构在途未提交"状态，CI 存在设计缺陷。**

三个最优先问题（P0）：
1. **冻结 manifest + strict pilot 无法构造训练 actor**（一手复现 ValueError；`results/_c6_strict_blind100_RUN.md` 同日记录）→ C7 100-seed 盲测认证无法重启。
2. **C7 训练不稳定**：PPO KL 爆炸至数百万后 NaN 崩溃（`_c6_strict_blind100_run.log` 实测尾部证据）→ 修复构造前题仍需稳定训练。
3. **HEAD ≠ 当前系统**：8 个未跟踪源码模块 + 12 个未跟踪测试（39M/15D/70??），干净检出无法跑当前套件；CI 验证的是旧树，且 `test_assert_gate_thresholds.py:104-115` 硬读未跟踪 CSV 使干净检出上 CI 必然失败。

数据层主要风险：results/（≈12.9 GiB）中约 57.9% 字节（7.48GB）为调试/一次性目录；约 9.7GB 为同 hash checkpoint 冗余复制；summary 产物覆盖仅 23.5%。文档层：核心中文文档存在 GBK 双编码乱码（EXPERIMENT_LOG 等），且多处断言滞后于代码（已修项未更新、死代码清单失效）。代码层：`env_core.py` 单类 10,837 行 / `step()` 2,180 行、`env_core.py:10136 except Exception: pass` 全吞、hyperedge 容量不足静默全零写入部署状态、capability_gauge 不查 `res.success`。依赖层：venv（Python 3.14.6 / torch 2.12.1+cu130）缺 scikit-learn（requirements 注释与事实不符），requirements 仅下界未锁定。

风险登记与修复建议见 §10（P0/P1/P2/P3 四档，R1-R20）。

---

## 1. 已验证的关键发现（一手复现，非转述）

### C1【高危】冻结 manifest + strict pilot 无法构造训练主路径的 actor
- 组合：`config/exp_strict_distributed_no_truth_pilot.yaml` 继承 `config/system_manifest.yaml`，其中 `marl.architecture_v2_enabled: true`（pilot 第 68 行），而 `target_allocation_enabled`、`target_allocation_teacher_enabled`、`comm_target_token_enabled` 均为默认 False（pilot/manifest 未覆盖）。
- `uav_isac/agents/networks.py:553-559` 的条件：`architecture_v2_enabled and not (use_target_allocation and comm_target_token_enabled and use_comm_cross_attention)` → raise `ValueError('architecture_v2_enabled requires target allocation, per-target token communication and cross-attention')`。
- **一手复现**（与 run_mappo.py:323-484 完全相同接线，含 `action_space.structured_actor = True`、`architecture_v2_enabled=bool(getattr(config.marl,...))`、use_target_allocation=target_allocation_enabled or teacher_enabled）：
  - `architecture_v2_enabled=True, target_allocation=False, comm_cross_attention=True, comm_target_token=False` → **CONSTRUCTION FAILED: ValueError**（networks.py:557）。
  - 对照实验：不设 `structured_actor=True` 时构造 OK（走 ActorNetwork 分支，无此检查）→ 证明问题只在训练主路径。
- 影响：`scripts/run_mappo.py` 用冻结 manifest 启动的任何训练/认证运行在构造期即失败。`results/_c6_strict_blind100_RUN.md`（2026-08-28）已记录同一结论（三次独立构造尝试均 raise）。**当前文件状态下 C7 无法重启**。
- 建议：在 manifest/pilot 中统一 `target_allocation_enabled: true`（或先关闭 architecture_v2_enabled，直到 target-allocation 栈真正部署），并加一个"用 run_mappo.py 同款接线构造 MAPPOAgent"的回归测试（当前测试只覆盖解析协议栈，不覆盖此路径——见 T2）。

### C2【高危】C7 100-seed 盲测认证失败：训练不稳定（KL 爆炸 → NaN）
- `results/_c6_strict_blind100_run.log` 尾部实测：`Normal(loc: torch.Size([256,128])) ... tensor([[nan,nan,...]]...)`（cuda:0）。
- 报告（2026-08-28 关闭）：KL 0.12 → 19.7@Ep70 → 5002@Ep80 → 25.6k@Ep90 → 3.27M@Ep160 → 231k@Ep180，后 NaN 崩溃（mappo_agent.py:550 evaluate_actions）。target_kl=0.02 完全无效。
- 影响：即使修复 C1，训练不稳定性也阻止有意义的盲测认证——冻结算法身份要求可复现训练。
- 待办（报告列出的 a/b/c 均未执行）：修 manifest 构造前置；稳定训练（降 actor lr / warm-start / 收紧 PPO kl-clip）；重启 C7 后跑 `tools/report_blind_certification.py`。

### C3【高危】重构在途未提交：HEAD 检出 ≠ 当前系统
- git 状态：39 M / 15 D / 70 ??；`git diff --stat`：54 文件 +5,326 / -11,169 行。
- uav_isac 磁盘 131 个 .py，git 跟踪仅 123 → **8 个未跟踪**，其中 6 个 coordination 新模块：`ai_candidate_screener.py`、`composable_certificate.py`、`distributed_compute_fusion.py`、`parallel_power_executor.py`、`progressive_information_transport.py`、`spectral_information_reuse.py`（另有 `environment/interference_certificate.py`、`utils/reproducibility.py`）。
- tests 磁盘 180 个 test_*.py，跟踪 170 → **12 个未跟踪**（strict/C6 家族 + test_reproducibility 等）；另有 7 个已修改。当前 1303 测试套件中的一部分（估算过百）来自未跟踪+已修改文件。
- 影响：从 HEAD 全新检出 -> 缺模块 -> **当前套件跑不起来**；CI（push/pull_request）验证的是旧树，其"绿"不能代表当前系统。`results/_c6_strict_blind100_RUN.md` 亦证实文档惯例"dirty workspace 与 Git HEAD 不同、以 source_snapshot.zip 为恢复源"。
- 建议：在下一个提交前完成 P1/P2 阶段性收口（含 12 个新测试文件 + 6 个新模块 + 7 个修改测试），或明确把当前工作树作为认证基线并记录 hash。

### C4【高危】CI 设计缺陷：干净检出上全量 pytest 必挂
- `tests/test_assert_gate_thresholds.py:104-115` 硬读 `results/architecture_v2_structure_student_u2u_resolve_bw50k_adaptive_b4b8_failclosed_gate100/paired_eval.csv`——该文件**未被 git 跟踪**（results/ 仅 45 个历史文件被跟踪，此 CSV 不在其中），无 skip 保护。
- .github/workflows/ci.yml 曾在干净检出上跑 `python -m pytest tests/ -q` → FileNotFoundError → **CI 红**。
- （对照：test_tica_exact_adapter.py:189 读 `results/dagger_corrected/dagger_D1.pt` — 该文件被跟踪，CI 可过。）
- 建议：该测试改为 tmp_path 合成数据，或显式 `pytest.skip(..., reason=缺少正式 artifact)`；并把 CI 已跑通的证据纳入仓库。

### C5【中高】测试套件覆盖缺口：训练路径构造无测试
- 全量 pytest 实测（本机 venv，2026-08-29）：**1303 passed, 8 warnings, 63.87s**。
- 但 strict pilot 测试（tests/test_strict_distributed_pilot.py，未跟踪）只覆盖 `EnvironmentCore`/`_episode`（解析协议栈），从不构造 MAPPOAgent → C1 的构造失败没有任何测试拦截。
- 建议：新增"run_mappo.py 同款接线构造 MAPPOAgent + 冻结 manifest"的回归测试。

---

## 2. 系统概览与入口

- **系统**：多 UAV 分布式 ISAC 仿真——仅 UAV 间通信（U2U）的协同感知，K 架 UAV 跟踪 Q 个目标；解析执行栈（L0 通信资源 → L1 固定结构 QoS 字典序 max-min 功率 LP → L2 结构 → L3 几何候选/前瞻 → 安全层）+ 分布式学习外层（等变 MAPPO actor + 结构 student）。(基线：子代理 A 摘要)
- **入口**：
  - 无单一主脚本；实验由 YAML 驱动。训练：`scripts/run_mappo.py`（构造 MAPPTrainer）；解析/认证：`tools/run_strict_distributed_pilot.py --config config/exp_strict_distributed_no_truth_pilot.yaml`；门禁断言：`tools/assert_formal_gates.py`、`tools/check_system_identity.py --strict --commit`、`tools/audit_detector_normalization.py --assert-ready`；认证报告：`tools/report_blind_certification.py --csv <paired_eval.csv>`（需 `PYTHONPATH=.`）。
  - 冻结身份：`config/system_manifest.yaml`（严格加载：键必须存在于 params.py；实测 341/341 全部 yaml 严格加载成功）。
- **模块**：`uav_isac`：environment（env_core 10,837 行/ trainer 7,565 行两大聚合器）、physical（无状态物理函数）、coordination（58 模块，核心机制 + 研究环）、agents（4 件）、evaluation（29 模块）、utils（types/seeding/provenance/reproducibility）。
- **科学状态（2026-08-29）**：
  - 历史 D1.10 独立采样盲测：QoS 0.940 / LCB 0.875（pre-V3/pre-G2 历史认证，**不是当前主结果**）。
  - V3-C0 重认证（2026-08-18）：QoS 0.910 / LCB 0.838 —— 最新可比 pre-G2 基线。
  - **post-G2（deflection 量纲修复后）尚无 100-seed 认证值**；legacy 6/6 为 DISCLOSED FAIL（LCB 0.492 < 0.70）。(CURRENT_SYSTEM_MODEL.md / DEEP_AUDIT 2026-08-25)
  - strict 分布式线（C6/C7）：**C7 盲测认证失败**（构造失败 + 训练 KL 爆炸 → NaN，见 C1/C2）；当前文件状态无法重启。
  - paper/ 手稿停留在 2026-07-22（把 D1.10 当主结果，未反映 post-G2 状态）——文档已登记的论文债 (CURRENT_SYSTEM_MODEL.md §12)。

## 3. 历史审计脉络（2026-08-25 → 08-28）

- **DEEP_AUDIT_SYSTEM_2026-08-25**：只读代码/配置/仓库健康审计（126 源文件、171 测试、326 配置、977 结果目录）。S1-S4（reseed RNG 泄漏、LR 调度、Wilson z 约定、K=1 退化）+ M1-M16 中等问题；结论认证门禁不动 1203 passed。**后续代码验证：S1-S4 与多数 M 已修，M14（文档引用已知问题）与 M11（多实现收敛 P2-2）仍开放。**
- **THREE_PHASE_POLICY_DESIGN 2026-08-25**：Phase A 漏追修复（worst 0.011→0.47）、Phase B 近场聚焦（0.589→0.956）；物理航程极限种子仍失败（差 8-30m）。
- **CONVERGED_LARGE_SCALE 2026-08-27**：3-seed×150 帧规模诊断——K8 受感知容量限、K16 受计算时间限（优化后 P95 127-218ms 仍不过 100ms 门）；CT 高速模型失配、CA 全败。
- **COARSE_TO_FINE_AI_POWER 2026-08-28**：AI 加速计算路径——AI 直接超分不可部署（证书仅 3.33%、回退 96.67%）；仅保留精确 LP（p50 −19%）。
- **C6/C7（2026-08-26→28）**：C6 决策充分通信链 wiring 完成（分 flag 默认 OFF、fail-closed 链）；**C7 100-seed 盲测失败**（见 C1/C2）。

---

## 4. 代码质量与架构审计（子代理 d5f1bfa5 + 一手核验）

- import 环：**0 环属实**（241 条内部 import 全解析，SCC 无环）。
- 巨型文件：`env_core.py` 10,837 行（EnvironmentCore 单类 90 方法，`step()` 2,180 行、`__init__` 1,621 行）；`trainer.py` 7,565 行（MAPPTrainer `_evaluate` 2,646 行、`update` 1,135 行）。
- 死代码：126 个零外部引用顶层符号（多为模块内 dataclass）；真可疑：`evaluation/metrics.py` 6 个叶子函数、`physical/channel.py` 5 个函数、`utils/types.py Position3D`、`evaluation/channel_margin.py` 整模块仅被自身单测引用。
- 危险模式：0 处 `except: pass` 直连，但 9 处吞异常 handler，**最重 `env_core.py:10136 except Exception: pass`（step 尾部全吞）**；TODO/FIXME 0 处；trainer.py 14 处无条件 print（`[PPO RATIO *]` 每次 update 打印）；QoS 阈值三元组 (0.80,0.70,0.60) ≥5 处独立硬编码。
- 哨兵冲突（文档点名属实）：`owner_local_physics.py:972` `target_age=-1`（永不触发过期）vs `target_invariant_transport.py:264` `age=max_age+1`（立即过期）。
- 已知问题核验：
  - `capability_gauge` 不查 `res.success` —— **仍存在**（capability.py:119-135；同文件 264/435/568 均有检查可抄）。
  - hyperedge 容量不足静默全零 —— **仍存在且更严重**（hyperedge.py:196-201 直接 return 全零行，无 feasible 标志；env_core.py:6898-6936 把 cost=0 当有效瓶颈写入部署状态）。
  - get/set_state 缺 5 字段 —— **已修复**（get_state 10578-10582 / set_state 10770-10780 完全对齐；文档滞后）。
- API 卫生：6 个 0 字节 `__init__.py`，仅 coordination 有 curated `__all__`。
- 滞留危险脚本：`_strip_dead.py`（行号硬编码删除清单，目标符号已不存在，误跑删错行）。

---

## 5. 测试套件健康度审计（子代理 c7400bcf + 一手 pytest 实测）

- 实测：1303 collected / 1303 passed / 8 warnings / 63.87s；`--collect-only` 1303 无收集错误；scripts/ 0 泄漏（testpaths=tests 生效）。
- 覆盖缺口（高）：6 个模块零测试触达 — `agents/base_agent.py`、`agents/neighbor_attention.py`、`agents/p0_fixed_agent.py`、`coordination/learned_move_ranker.py`、`evaluation/metrics.py`、`utils/seeding.py`。
- results/ 依赖（高）：test_assert_gate_thresholds.py:106 硬读未跟踪 CSV 无 skip（= CI 破坏者，见 C4）；test_tica_exact_adapter.py:189 有 skip 保护。
- 随机性纪律（中）：126/180 测试文件无种子位，其中 7 个含 torch.randn/np.random 且无种子（comm_off_consistency、dynamic_kq_shapes、local_pd_boundary、observation_parsing、recurrent_logprob_consistency、target_wise_advantage、tica_exact_adapter）。
- 断言纪律：199 处精确浮点 ==（32 处非平凡）；approx/allclose ~380 处；5 文件抽查断言普遍认真（机制验证而非只跑通）。
- skip/xfail：仅 2 处 pytest.skip（环境依赖型），0 xfail —— 无债务但 skip 会掩盖未测。
- 组织：31/180 audit 相关测试文件（17%）；一次性产物测试（test_d1_10_audit_fixes、test_summarize_* 等）应移出或标记 diagnostic。

---

## 6. 配置一致性审计（一手核验 + 独立扫描）

**一手核验（已确认）**：
- **严格加载全绿**：341/341 个 config yaml 用项目 loader 严格加载成功（0 失败）——keys 与 params.py 的 schema 一致性良好（manifest 的"键必须存在于 params.py"机制有效）。
- **C1 的配置层根源**：pilot 设 `architecture_v2_enabled: true`（第 68 行）但未设 target-allocation / target-token（默认 False）→ 与 networks.py:553-559 前置条件矛盾（见 C1）。
- **配置膨胀**：341 个 yaml 中约 265 个未在任何代码/文档中按名引用（一手全树 grep 扫描；排除 results/；注意结果目录名会变形、bench_* 可能被 run_benchmark_matrix.py 通配引用、部分经 extends 链继承使用，故该数有上偏——精确的"有配置无结果"数为数据审计的 115）。大量 `*_pilot/*_eval/*_diag/*_smoke/*_gate` 一次性配置堆叠（如 exp_800_k6q6_distributed_diag_tx{0..5,40..45}.yaml 系列、exp_strict_distributed_* 系列）。
- **seed bank 结构**：`stratified_seeds_1130_k16q16_blind.json` 含 fingerprint_version + scenario_fingerprint（正式）；其余 v2 bank 无指纹（test_reproducibility.py 会拒绝——已读断言 103-108）。k16 bank test split = 100 seeds。

---

## 7. 依赖与运行时环境审计（子代理 c8b472a8 + 一手验证）

- 环境：Python 3.14.6 / numpy 2.5.0 / scipy 1.18.0 / torch 2.12.1+cu130（CUDA 13.0 可用，GPU 冒烟通过）/ gymnasium 1.3.0 / pytest 9.1.1；pip check 无破损。
- 【高】venv **未装 scikit-learn**：实测 `find_spec('sklearn') → None`、`import sklearn.metrics → ModuleNotFoundError`。requirements.txt 头注释自称"2026-08-16 已修复"，**注释与事实不符**。
- 但注意（一手核验修正）：sklearn 仅在 `tools/probe_crisis_gate.py:183/221` 等函数体延迟导入；测试路径使用模块自带 `_balanced_accuracy`/`_binary_roc_auc`（127/136 行）→ 1303 全过不受影响。"测试必挂"是过度声称，真实风险是**延迟导入的工具路径在缺 sklearn 的 venv 中会挂**，且 venv 与 requirements 长期漂移。
- requirements（9 项）vs venv（41 包）不锁定；psutil 声明但 0 处 import。
- torch.load `weights_only` 用法不一致（torch 2.12 默认 True，(run_mappo.py:555 等未指定)）。
- .pt 权重无 torch 版本元数据，跨版本不可校验。
- 可移植性：Windows/MKL eigvalsh 原生 abort 已知（certified_feedback.py:124-127 标量 Cholesky 规避；crash_isolated_seed_eval.py MKL_NUM_THREADS=1 隔离）；uav_isac 全绝对导入（1030 处）、0 相对导入、无 _private 导入。
- CI 用 python 3.11 vs 本机 3.14 —— 版本带差（CI 只测 3.11）。

---

## 8. 文档与代码一致性审计（一手核验）

**一手发现（数据完整性）**：主要文档 CJK 内容存在 **GBK 双编码乱码**：
- 实测（UTF-8 可解码但含乱码字形）：EXPERIMENT_LOG.md（166 个乱码字形）、CURRENT_SYSTEM_MODEL.md（186）、ALGORITHM_EVOLUTION.md（119）、README.md（27）、CODE_STRUCTURE_MAP（35）。
- 例：EXPERIMENT_LOG.md 于 93646 offset 处 `..."29.3 Gate �ж�..."` 段出现 `���� minimum-floor power ...` 乱码序列。
- 数字、ASCII、机器可读内容完好；中文叙述段落不可完整恢复 → 实验记录存在信息损失风险。
- 建议：用原始 source_snapshot.zip 或 git 历史重建干净文本；今后文档写入明确 UTF-8/换行策略（git 已出现 CRLF warning 噪音）。

**一手断链检查（6 篇核心文档）**：10 处指向已不存在文件的引用，集中在 3 篇：
- `tools/audit_structure_regret.py`（CODE_STRUCTURE_MAP / CURRENT_SYSTEM_MODEL / ALGORITHM_EVOLUTION 三处）、`tools/verify_v3_t3_on_trace.py`、`tools/audit_horizon_joint_oracle.py`、`tools/audit_l3_l2_alternating.py`、`uav_isac/coordination/congestion_relief.py` —— 均为 2026-08-25 清理（已证伪机制删除）中被移除的模块，文档未同步。
- 两处 `config/exp_800_k8q8_..._indep_relief.yaml` 类名是文档中的省略写法（非真实路径），应补全或加引注。
- 结论：断链规模小（10 处），但全部指向"已删除机制"的文档残留——与 M14（KNOWN_ISSUES/OPTIMIZATION_LOG 引用已删文档）同类的文档债。

---

## 9. 结果数据与可复现性审计（子代理 02fdc474）

- 规模：results/ 6326 文件 / 12.92GB；979 顶层目录。
- 【高】权重冗余复制 ≈9.7GB：eval-only 目录逐 seed 复制同一基座 checkpoint（best_restored.pt 652 份 + risk_critic_final.pt 664 份，同族 sha 全同验证）。
- 【高】产物体系脆弱：summary.json 覆盖率仅 23.5%，`*_summary.md` 全树仅 1 个，summary schema 80 种变体。
- 【高】调试目录与正式结果混放：`_` 前缀 scratch 330 目录 + smoke 171 目录 = 478 目录 / 7.48GB（占字节 57.9%）。
- 【中】种子链路：仅 `stratified_seeds_1130_k16q16_blind.json` 带 fingerprint_version（正式）；其余 v2 bank 无指纹被 validate_formal_run 拒绝；正式 bank 100 个 test 种子 0/100 有结果目录。
- 【中】test_reproducibility.py **不验证"同种子复跑输出一致"**，只验证绑定与 fail-closed。
- 【中】配置↔结果：115/341 配置无结果；7 个 manifest 引用不存在的配置名；单配置 265 个 manifest 膨胀。
- 【低中】仓库根 10+ 调试文件未忽略且未跟踪（`_c6_test_fail.log`、`_pytest_full*.log`、`_verify_*.py` 等），`_orphan_v3.txt` 已被跟踪且在修改态。
- 正向：run_manifest 机制扎实（git commit/dirty、source/config/checkpoint sha、命令行），抽查 source_snapshot.zip sha 与 manifest 完全一致；正式目录 worker.log 健康。

---

## 10. 风险登记与修复优先级

### P0 — 阻断认证/复现的核心问题（本周内）
| # | 问题 | 证据 | 修复建议 |
|---|---|---|---|
| R1 | 冻结 manifest + strict pilot 无法构造训练 actor | networks.py:553-559；一手复现 ValueError（C1） | manifest/pilot 统一 `target_allocation_enabled: true` 或先关 `architecture_v2_enabled`；新增 run_mappo 接线构造回归测试 |
| R2 | 训练不稳定：PPO KL 爆炸 → NaN（target_kl 失效） | `results/_c6_strict_blind100_run.log` NaN 尾；报告 08-28 | 降 actor lr / warm-start / 收紧 kl-clip；先修复再认证（C2） |
| R3 | 重构未提交：HEAD 检出现缺模块（8 个未跟踪 .py + 12 个未跟踪测试） | git status 39M/15D/70??；git ls-files 对比 | P1/P2 阶段收口提交；或将当前树作为认证基线记录 hash（C3） |
| R4 | CI 设计缺陷：干净检出全量 pytest 必挂 | test_assert_gate_thresholds.py:104-115 读未跟踪 CSV | 该测试改 tmp_path 合成数据或显式 skip（C4） |

### P1 — 数据/产物层（安全与可信度）
| # | 问题 | 证据 | 修复建议 |
|---|---|---|---|
| R5 | 权重冗余 ≈9.7GB（652+664 份同 hash checkpoint） | 数据审计 #1 | 去重为符号链接/引用清单；核实各 seed 是否真为 eval-only |
| R6 | summary 产物覆盖 23.5%、schema 80 种变体 | 数据审计 #2 | 统一 summary schema；audit_results_tree 输出强制 |
| R7 | 调试目录 478 个 / 7.48GB 与正式结果混放 | 数据审计 #3 | 物理归档 `_` 前缀与 smoke 目录（audit 工具只报告不清理） |
| R8 | 正式种子 bank 的 100 个 test 种子无结果产出 | 数据审计 #4 | C7 重启后按 bank 计划跑出 seed_<n> 结果；补 fingerprint_version 缺失的 v2 bank |

### P2 — 代码质量
| # | 问题 | 证据 | 修复建议 |
|---|---|---|---|
| R9 | EnvironmentCore 单类 10,837 行 / step() 2,180 行 | env_core.py:292-11154 | 拆协议/移动/功率/序列化层（文档 blueprint 已列） |
| R10 | `env_core.py:10136 except Exception: pass` 全吞 | env_core.py:10136 | 至少记日志并 fail-closed |
| R11 | hyperedge 容量不足静默全零 + cost=0 当有效值写入 | hyperedge.py:196-201; env_core.py:6898-6936 | 返回 feasible 标志；部署路径拒绝 0 行 |
| R12 | capability_gauge 不查 res.success | capability.py:119-135 | 抄同文件 264/435/568 先例 |
| R13 | `_strip_dead.py` 滞留（行号已失效，误跑删错行） | 仓库根 44KB | 删除或归档到 _archive |
| R14 | -1 哨兵语义冲突（永不过期 vs 立即过期） | owner_local_physics.py:972; target_invariant_transport.py:264 | 统一为显式缺失枚举 |
| R15 | 阈值三元组 (0.80,0.70,0.60) ≥5 处硬编码 | env_core.py:4239-4241 等 | 提取为 params 常量 |

### P3 — 卫生与债务
| # | 问题 | 证据 | 修复建议 |
|---|---|---|---|
| R16 | venv 缺 scikit-learn（requirements 注释与事实不符） | 实测 find_spec None | `pip install -r requirements.txt` 同步 venv 或补环境记录 |
| R17 | torch.load weights_only 不一致 | run_mappo.py:555 等 | 统一显式 `weights_only=` |
| R18 | 文档 CJK 乱码（GBK 双编码） | EXPERIMENT_LOG/CURRENT_SYSTEM_MODEL/ALGORITHM_EVOLUTION（166/186/119 乱码字形） | 从 source_snapshot/git 历史重建干净文本；统一 UTF-8 写入策略 |
| R19 | 根目录 15+ 调试文件未忽略未清理；_orphan_v3.txt 已跟踪 | git status | 加 .gitignore 项或归档 |
| R20 | 6 个空 __init__.py、命名不一致、metrics/channel 死叶模块 | 代码审计 #5 | 包级导出策略统一；清理或标注 research-only |

---

## 11. 附录：一手验证命令与证据清单

1. 全量 pytest（venv `D:\BYLW\LD3\pytrch_ven\Scripts\python.exe`）：`python -m pytest -x -q` → **1303 passed, 8 warnings, 63.87s**。
2. 构造复现（run_mappo 同款接线，`structured_actor=True` + manifest/pilot flag）：`CONSTRUCTION FAILED: ValueError architecture_v2_enabled requires target allocation, per-target token communication and cross-attention`；对照（structured_actor=False）`CONSTRUCTION OK`。
3. 配置严格加载健康检查（`_audit_cfg_load.py`）：341/341 OK，0 失败。
4. git 状态：39 M / 15 D / 70 ??；`git diff --stat`: 54 文件 +5,326/-11,169；uav_isac 131 .py on disk vs 123 tracked；tests 180 vs 170 tracked。
5. 文档编码：UTF-8 可解码但含 GBK 双编码乱码字形（EXPERIMENT_LOG 166 / CURRENT_SYSTEM_MODEL 186 / ALGORITHM_EVOLUTION 119）。
6. C7 日志证据：`results/_c6_strict_blind100_run.log` 尾部 `Normal(loc: torch.Size([256,128])) ... tensor([[nan,nan,...]])`（cuda:0）。
7. sklearn 缺装：`python -c "import importlib.util; importlib.util.find_spec('sklearn')"` → None。
8. results/ 跟踪核对：git ls-files results/ = 45（历史资产），测试依赖的 `architecture_v2_.../paired_eval.csv` 不在其中。
9. 本审计产生的辅助脚本：`_audit_cfg_load.py`（可删）、`_audit_enc_check.py`（可删）。

---

## 12. 审计后第一轮修复与复验（2026-08-29）

本节记录针对上述风险登记的实际修复，不覆盖原始审计证据。所有结论均以当前脏工作树为对象，因此只构成开发诊断，**不构成正式盲测认证**。

### 12.1 已关闭风险

| 风险 | 修复 | 可执行证据 |
|---|---|---|
| R1 | `system_manifest.yaml` 明确启用 `target_allocation_enabled`，使 architecture-v2 的目标分配、逐目标 Token、跨注意力三个前置条件同时成立；C0 身份检查新增硬 pin | `test_strict_pilot_constructs_run_mappo_structured_actor` 按 `run_mappo.py` 接线成功构造 actor；strict 单步 smoke 通过 |
| R2 | 恢复 PPO 的 KL 提前停止；用稳定形式 `exp(x)-1-x` 估计 KL，并将非有限 log-ratio 映射为拒绝；在 backward 前拒绝越界 minibatch；对 PPO 与策略辅助更新实施整段 rollout 的事务式 KL 验证，越过 `1.5*target_kl` 时同时回滚 actor 参数与优化器矩 | 机械性质测试验证 KL 非负、NaN/Inf fail-closed、守卫位于 backward 前；真实小批量 actor+critic 更新验证接受后的 `post_update_approx_kl <= 1.5*target_kl` |
| R4 | 去除 CI 对未跟踪正式结果 CSV 的硬依赖，改用 `tmp_path` 中确定性构造的历史阈值夹具 | 干净数据依赖不再存在；门限聚合回归通过 |
| R10 | 启用词典序审计时，JSONL 证据写入失败不再 `except Exception: pass`，而是抛出带路径的 RuntimeError | 目录冒充审计文件的故障注入测试通过 |
| R11 | 角色容量分配显式返回 `feasible`；容量不足、惩罚边被选或覆盖不完整时返回 `cost=inf` 并清空职责；环境部署路径记录 infeasible viewer 并 fail closed | 容量不足性质测试与环境调用回归通过 |
| R12 | capability gauge 校验求解器 `success`、解向量形状及有限性；失败不再产生伪证书 | 强制 `success=False` 的求解器故障注入测试通过 |

### 12.2 全量与闭环复验

- 全量测试：`1308 passed, 8 warnings, 44.70s`；8 条均为既有 TICA Transformer nested-tensor 提示，无失败。
- C0 系统身份检查：0 个 hard-pin 失败。
- strict seed=7、10 帧、carrier period=3 诊断：948 bit/frame，交付率 1.0，通信截止违约率 0，超边覆盖率 1.0，闭环关键路径 P95 约 9.45 ms，100 ms 在线预算违约率 0。
- 上述 strict 诊断的 hold-action 感知 QoS 为 steady≈0.586、weak3≈0.451、worst≈0.408，未达到任务门限；这说明通信/协议/实时性闭环健康，**不说明感知性能已达标**。
- run manifest 正确标记 `formal_result_eligible=false`，原因为 dirty workspace；R3 仍是进入正式 100-seed 认证前的阻断项。

### 12.3 下一优先级

1. 收口并冻结当前源码、配置、seed bank 与依赖环境，关闭 R3 后才允许正式盲测。
2. 统一 R14 的缺失/永不过期哨兵语义，并补跨模块序列化性质测试。
3. 将 R15 的 QoS 阈值三元组提升为单一配置值，避免训练奖励、通信约束与验收门不一致。
4. 清理或隔离 R5/R7 的重复权重与调试产物，但在哈希清单完成前不执行破坏性删除。

---

## 13. 第一轮修复复验 + 第二轮优化建议（2026-08-29，独立实测）

### 13.1 第一轮修复复验（一手核验，非转述）

| 声称 | 独立验证 | 状态 |
|---|---|---|
| R1 manifest 加 `target_allocation_enabled` 后训练 actor 可构造 | 按 run_mappo 真实接线复现：`use_target_allocation=True` 且 pilot 的 `comm_payload_mode=target_tokens` 映射 `target_tokens=True`、cross-attention=True → **CONSTRUCTION OK**；对照（structured_actor=False）仍走旧分支 | ✓ 闭环 |
| R2 PPO KL 提前停止 + 稳定估计 + fail-closed + 事务回滚 | `trainer.py:51-75 stable_ppo_ratio_and_approx_kl`（expm1 稳定式、非有限 log-ratio→inf KL）；`:4113-4125` minibatch 在 backward 前拒绝；`:4825-4902` 整段 rollout 事务式 KL 验证与 actor/优化器回滚（`1.5*target_kl`） | ✓ 闭环 |
| R4 CI 不再依赖未跟踪 CSV | `test_assert_gate_thresholds.py:104+` 改为 tmp_path 合成夹具（历史 4×4 数值内联），无 results/ 依赖 | ✓ 闭环 |
| R10 审计日志写入失败不静默 | `env_core.py:10161-10165`：JSONL 追加失败抛带路径 RuntimeError（"refusing to continue"） | ✓ 闭环 |
| R11 容量不足显式不可行 + 部署 fail-closed | `hyperedge.py:123-205` 返回 `(tx,rx,cost=inf,feasible)`；`env_core.py:6915-6924/6945-6954` 置 `bottleneck=inf`、清职责、记录 `rolecap_assignment_infeasible` | ✓ 闭环 |
| R12 capability 失败不再出伪证书 | `capability.py:124-133`：`res.success`/形状/有限性任一不满足 → raise RuntimeError | ✓ 闭环 |
| 全量 1308 passed | 本机复跑 **1308 passed / 8 warnings / 46.64s**（用户 44.70s） | ✓ |
| C0 身份检查 0 硬约束失败 | 全部 hard-pin `[OK]`；**但命令整体 FAIL**：`git workspace dirty` + `manifest is not tracked` | ⚠ 需与 13.2 一起读 |
| strict 10 帧诊断 | 本机复现：948 bit/frame、delivery 1.0、deadline 0、coverage 1.0、闭环 P95≈9.67ms（用户 9.45，同量级）、online miss 0；hold-action QoS 0.549/0.473/0.444（用户 0.586/0.451/0.408，同量级）→ **协议与实时闭环健康成立，任务性能未达标成立** | ✓ |

### 13.2 第二轮新增硬缺口（正式认证前置，一手实测）

1. **冻结 manifest 本身未被 git 跟踪**（`git ls-files config/system_manifest.yaml` = 0）。冻结提交必须包含它，否则"冻结身份"无版本锚点。
2. **manifest 钉选的种子库与 strict 身份不匹配（双重）**：
   - `config/stratified_seeds_1130_k8q8_blind.json`：`schema_version=1`、**`fingerprint_version=None`**、`sampling_seed=None`（有 `blind_draw_seed`）；
   - `source_config = config/exp_800_k8q8_dcb_top1_scale.yaml`（**DCB-top1 训练/评估身份，非 strict**）；bank `scenario_fingerprint=c7dd74ad93a9fbaf` ≠ strict pilot 期望 `f3530f4c26265081`（实测 `scenario_fingerprint(pilot_cfg, bank.source_config)`）。
   - 后果：即使工作树冻结提交，`validate_formal_run`（`uav_isac/utils/reproducibility.py:214-225`）**必然拒绝**——要求 `fingerprint_version=="reset-distribution/v2"` 且 bank 指纹匹配运行身份。
3. **正式运行线程环境要求**：`validate_formal_run`（`:234-242`）要求 OMP/MKL/OPENBLAS/NUMEXPR 四变量**全部为 "1"**；当前诊断 run manifest 显示四者均为 `UNSET`。
4. **种子顺序约束**：正式认证的种子列表必须与 bank 的 `final_eval_seed_split`（test）**按序完全一致**（`:226-233`）。

### 13.3 第二轮执行顺序（建议）

1. **Step 0 — 冻结预检脚本（先做，~0.5 h）**：输出一张 blocker 清单（git 全清？manifest 已跟踪？bank 指纹/身份/顺序？线程变量？requirements 锁？），每项有命令与判据；任一未过即不进正式认证。
2. **Step 1 — R14 哨兵统一（先于冻结，因为改跨模块行为）**：
   - 现状（一手扫描 82 处匹配中的真哨兵）：`-1` 双语义（`owner_local_physics.py:972` 未收到反馈=永不过期 vs `persistent_geometry_execution.py:333/382/396/409` 已过期）、env_core `-1e8/-1e9` last_seen/过期、`-inf` stale score、`np.inf` 无答案；`movement_target=-1` 为"无分配"。
   - 关键判断：owner 的"永不过期"与 transport 的"立即过期"是**两种故意策略（保守 vs fail-closed）**——统一目标不是抹平行为，而是**同一协议内显式区分"从未收到/已过期/无分配"**：引入 `uav_isac/utils/sentinels.py` 枚举（如 `NEVER_SEEN / EXPIRED / TARGET_NONE`），消除"同一数值两种语义"，并补"各模块读写同一常量 + get/set_state 序列化往返"性质测试。
3. **Step 2 — R15 QoS 门限单一来源**：
   - 冲突面（一手证据）：`tools/assert_gate_thresholds.py:36` `MEDIUM_FLOORS={0.80,0.70,0.60}` 硬编码（验收门）；`tools/report_blind_certification.py:54/56` `p_hat>=0.70/lcb>=0.70` 硬编码；`params.py:670-672` `coord_reward_*_floor`（奖励）与 `:1013-1015` `comm_qos_*_min`（通信）互不引用；全树 ~30 处 `getattr(..., 0.80/0.70/0.60)` 影子默认（如 env_core.py:1113-1117、trainer.py:1820-1822）。
   - 建议：新增唯一权威 `marl.qos_acceptance_floors: List[float]`（默认 [0.60,0.70,0.80]）；`assert_gate_thresholds.py`/`report_blind_certification.py` 从冻结配置读取（工具显式传参或读 manifest）；删除影子默认；加 `test_qos_gate_provenance` 一致性回归（改 param 一方即失败）。
   - 语义注意：验收门 weak3（最弱 3 目标均值）与 `task_constrained_qos_floors[3]=3`（bottom-k 顺序统计）数值同 3 但语义略异——统一时文档化，避免"数值相同=语义相同"错觉。
4. **Step 3 — 冻结提交 + 种子库重建 + 正式认证**：
   - 一次性提交（manifest、12 新测试、6 新模块、修复后测试、CI、requirements 锁、文档 §12/§13）。
   - 用 seed 重建工具在 strict 身份下重建 k8q8 blind bank（`fingerprint_version="reset-distribution/v2"` + `scenario_fingerprint=strict` + 固定采样参数），**先验证新 bank test split 的 100 个 seed 与旧 bank 是否一致**：若一致则仅升级指纹；若不一致须显式记录旧 bank 退役、新 bank 入 manifest。
   - 线程环境统一导出 `OMP/MKL/OPENBLAS/NUMEXPR=1` 后：`check_system_identity.py --strict --commit <sha>` → 干净树全量 pytest → `tools/run_strict_distributed_pilot.py --formal --seeds <bank.test 按序>` → `tools/report_blind_certification.py --csv paired_eval.csv`。
   - **预期**：hold-action QoS 0.55/0.47/0.44 远低于 0.80/0.70/0.60 门限，正式认证大概率不过——但它的价值是锁定"协议+实时闭环健康"的正式基线，把差距转化为科学问题。
5. **Step 4 — 科学方向（认证后）**：分析 hold-action 下 steady≈0.55 的根因（几何覆盖不足 / 功率预算内感知上限 / 仅 local_only 证据融合无邻居增益，对照 legacy D1.10 0.94 的差异来源），决定 strict 身份下需启用哪些分布式协议或预算调整——在此之后再谈 P2（巨型文件拆分）与 P3（数据清理）。

### 13.4 本轮不改动项

- R5/R7 数据清理保持只读审计，不做破坏性删除（哈希清单先行）。
- env_core.py（10,837 行）/ trainer.py（7,565 行）拆分列入 backlog，不阻塞认证。

---

## 14. advice/002.md 问题逐条确认（2026-08-29，一手代码核验）

`D:\BYLW\LD3\advice\002.md` 存在（9,417 字节），是一份以计算/性能优化为主的深度审计建议（9 个优化点）。逐条对照当前代码核验如下：

| # | advice/002 主张 | 当前代码证据 | 状态 |
|---|---|---|---|
| 1.3 | 有限块长二分重复计算 capacity/dispersion | `physical/finite_blocklength.py`：`achievable_information_bits`(58-72) 与二分 `minimum_blocklength_normal_approximation`(75-110)/`minimum_snr_normal_approximation`(113-146)；while×3，无跨迭代缓存 | **仍成立**（性能，P1） |
| 1.4 | 通信链路预算逐链路 Python 循环 | `environment/communication.py transmit()`(509-701)：`for sender in active`(580) × `for receiver in range(K)`(613) 双重循环，每链路 `_link()`+`packet_error_probability`；K=8 → 56 链路/帧 | **仍成立**（性能，P2） |
| 2.1 | deflection 慢速路径三重循环 + 标量 dd | `physical/deflection.py:320-382`：`for i×j×q` 三重循环，每元标量 `compute_raw_deflection`(336)/`compute_dd_effectiveness`(344)/`compute_dd_phys_gain`(360)。**注意：`dd_gain_mode=continuous`（冻结 manifest 的 canonical 模式）恰走此慢路径** → 这不是可选优化，而是当前冻结身份的每帧成本 | **仍成立（且命中 canonical 路径，优先级上调）** |
| 2.2 | OTFS 标量/批量双实现逻辑重复 | `physical/otfs.py`：`compute_dd_phys_gain`(121-143 标量) 与 `compute_dd_phys_gain_batch`(146-196 批量) 并存；deflection 慢路径仍用标量版 | **仍成立**（结构/性能，P2） |
| 2.3 | `compute_PD` 每次调用重算 `Q^-1(P_FA)`（erfinv） | `utils/math_utils.py`：`compute_PD`(32-48) 调 `Q_inverse`；`lru_cache` 出现 0 次 | **仍成立**（热路径 erfinv，P1，修复成本最低） |
| 3.1 | P0 贪心非子模，无 (1-1/e) 保证 | `physical/inner_solver.py:14-21` **头注释自承认**：`U=-log(1-P_D)` 非凹非子模，greedy 是 heuristic（KNOWN_ISSUES B8）；并自写"1-exp(-kD) 可恢复子模性" | **仍成立，但属已知设计选择**——切换效用会改变行为/数值，属科学决策而非 bug，认证前不宜动 |
| 3.2 | 贪心主循环 Python 循环 | `physical/inner_solver.py:232 while True:` → `:236 for candidate_index, e in enumerate(candidates):` —— 逐候选约束检查/argmax 为 Python 循环（已用 `marginal_utility_gain_batch` 做了 per-target 批量增益） | **仍成立**（性能，P2） |
| 3.3 | maxmin_power 多套实现并存 | `coordination/maxmin_power.py` 顶层函数 ≥6 套：`solve_fixed_structure_maxmin_power_lp`、`replicated_local_row_maxmin_power`、`distributed_dual_maxmin_power`、`distributed_column_generation_maxmin_power`、`local_transmitter_range_minimax_share`、`relaxed_same_geometry_target_ceiling` 等 | **仍成立**（维护成本，P2；未隔离） |
| 3.4 | capability_gauge 不查 `res.success` 静默失败 | `coordination/capability.py`：`res.success` 出现 2 次；`capability_gauge`(56-145) 在 :124-133 检查 success/形状/有限性并 raise | **已修复**（第一轮 R12） |
| 3.5 | env_core 巨型聚合；get/set_state 缺 5 字段 | `environment/env_core.py` 11,182 行、`step()`(8115-10297)=2,183 行；get_state(10594-10786)/set_state(10788-11126)；**缺字段问题已修复**（前轮审计确认全键对齐） | **巨型聚合仍成立**（backlog）；**缺字段已修复** |
| 3.6 | feasibility_oracle MILP+LP 交替组合爆炸 | `physical/feasibility_oracle.py`：`milp`×23/`linprog`×2（`_select_pairs_milp`、`_select_pairs_local_only_milp`、`_optimize_power_lp`）；已有参数超时 `p0_budget_coupled_time_limit_s=5.0` | **结构仍成立，部分缓解**（已有 time-limit 参数） |

**确认结论**：advice/002.md 的问题**大部分仍然存在**（9 个优化点中 8 个仍成立/部分成立，1 个已修复，另 1 个为已知设计选择）。它与 R14/R15（哨兵/QoS 门限）是**不重叠的两组问题**：advice/002 全是计算性能/结构维度。

**据此修订后续优先级**：
1. 认证主线不变：R3 冻结（含 bank 重建）→ R14 哨兵 → R15 QoS 门限（§13.2-13.3）。
2. 计算层**顺手修复**（低风险、不改变语义，可在冻结前完成并纳入冻结提交）：
   - **2.3 缓存 `Q_inverse(P_FA)`**（15 分钟级，热路径 erfinv，零行为变化）；
   - **2.1/2.2 deflection 慢路径改用批量 `compute_dd_phys_gain_batch`**（canonical continuous 模式每帧成本，需先做等价性回归——当前 1308 套件含 `test_physics_closure_c0c1.py` 可作天然门禁）；
   - 1.3 有限块长二分外提 capacity/dispersion；1.4/3.2 向量化（K=8 时收益有限，列为 backfill）。
3. 3.1（效用切换）**不在认证前做**——改变数值口径，属科学决策；3.3/3.5 维持已有 backlog。

---

## 15. 外部深度审计报告核验（2026-08-29，一手交叉验证）

提交方提供了一份对 CURRENT_SYSTEM_MODEL.md §13 的深度复核报告（通过项 + 已知问题复核 + 3 个新发现）。逐条一手核验如下：

### 15.1 测试数字

| 报告声称 | 独立实测 | 判定 |
|---|---|---|
| 1314 tests collected | `--collect-only` **1314 collected**（比第一轮 1308 多 6，报告基于更新状态） | ✅ 准确 |
| 819+ passed（0 failed） | 完整运行应为 1314 全过；本轮全量实测 **1314 passed / 8 warnings / 47.41s** | ⚠ "819+"非完整运行数字，不可引用 |

### 15.2 已知问题复核（§13 原文已读，CURRENT_SYSTEM_MODEL.md:1281-1392）

| §13 条目 | 文档原文 | 当前代码证据 | 判定 |
|---|---|---|---|
| A（6/6 未过盲测） | QoS 0.530 / LCB 0.433；teacher 0.723 @47 seed；Student 跨尺度校准为绑定瓶颈；V3 dual-priced L2 A/B 无效默认 OFF（:1295-1301） | 与文档一致 | ✅ 报告准确（"A/B 无效/默认 OFF"即文档口径） |
| C3（λ* 非唯一） | 文档写"熵/近端正则化**待接线**"（2026-08-17 快照） | **已接线**：`maxmin_power.py:1510 entropic_maxmin_dual_prices` + env_core 7+ 调用点（7626/7679-7681/7726-7728/7833/7865-7867/9287-9293/9377-9382），`entropic_dual_price_enabled` 默认 False | ✅ **报告正确、文档滞后** |
| D（遗留代码） | `_oracle_alpha` 从未接线、residual_actor 无法工作、maxmin LP prices 占位值（:1378-1383） | trainer.py:2136 `_oracle_alpha=0.0` + :3038 `r < 0.0` 恒假（死功能实测）；residual_actor.py 存在 | ✅ 属实 |
| E（MKL eigvalsh） | Windows 并发中止，进程隔离兜底、未根治（:1391-1392） | 代码多处 eigvalsh；与文档一致 | ✅ 属实 |

### 15.3 通过项抽查

- `detection.py:77-96` P_D=Q(Q⁻¹(P_FA)−√D) ✅；`deflection.py:355-368` + `otfs.py:121-134` DD 模型 ✅；`belief.py:155-237` CI 融合（eigvalsh 正定性检查 + inv）✅；均与此前审计一致。

### 15.4 新发现核验

| 编号 | 报告主张 | 一手核验 | 判定 |
|---|---|---|---|
| NEW-1 | belief.py:592 EKF 用 `np.linalg.inv(S)` 而非 solve | `K_gain = cov @ self.H.T @ np.linalg.inv(S)`（592 行，属实）；同文件 598/694 已有 `np.linalg.solve` 先例 | ✅ 属实；P2 建议合理（1 行改动） |
| NEW-2 | belief CI 中 eigvalsh 无 try-except、MKL 崩溃无 fallback | 167/219/220 有显式 `<=0 → ValueError` **正定性检查**，无 try 包裹；但 §13-E 已有**进程级**兜底（crash_isolated 包装器） | ⚠ 部分属实：函数层缺 try 对；"无 fallback"与 §13-E 冲突，应改为"依赖进程级兜底" |
| NEW-3 | maxmin_power.py:988/1025 量化价格用 1e-300 | `log_prices = np.log(np.maximum(prices, 1.0e-300))`（1025，属实；另 615/740 亦 1e-300） | ✅ 属实；建议 1e-12 与全局保护一致（微调合理） |

### 15.5 未验证/需标注项

- "entropic 开销 <2%"：未做基准，标注为估计。
- "strict_pilot 28.5 ms/frame（<50ms 实时约束）"：实测 26.5 ms / P95 29.2 ms（同量级 ✅）；但在线预算为 `online_budget_ms=100`，"<50ms"出处不明，建议统一引用 100 ms 预算口径。
- 报告整体可信度：**高**——主要 claim 均与代码/文档一致，少数瑕疵（"819+"数字、"无 fallback"措辞、"待接线/已接线"的文档滞后方向）已在上表标注。

---

## 16. 性能优化执行记录（路线图 ALGORITHM_PERFORMANCE_ROADMAP_2026-08-29）

### 16.1 O7 已实施（2026-08-29，零语义）——缓存 `Q_inverse` 标量路径

- **改动**：`uav_isac/utils/math_utils.py` 增加 `@lru_cache(maxsize=64) _q_inverse_scalar`；`Q_inverse` 对 size==1 输入走缓存、数组输入保持原向量化公式；输出形状逐位保持（`np.full_like`）。
- **理论依据**：`Q⁻¹(p)=√2·erfinv(1−2p)` 是确定性纯函数；热路径调用点 ~32/33 为 size==1（`compute_PD`、内层求解 marginal-gain 循环、feasibility_oracle/拦截功率/几何修复的 P_FA 反演）。缓存是纯 memoization，**不改变任何数值**（同一平台/library 下 erfinv 逐位确定）。
- **性质测试**：`tests/test_q_inverse_cache.py` 4 项——① 标量缓存路径与向量化公式**逐位相等**；② 数组路径不受缓存影响（与单元素调用逐位一致）；③ 输出形状镜像输入（0-d/(1,)/(n,)）；④ `compute_PD` 与手算 `Q(Q⁻¹(P_FA)−√D)` 逐位一致。
- **基准（诚实记录）**：200k 次重复标量调用 520→379 ms（**1.37x**，缓存命中 199,999/200,000）。收益真实但温和——scipy `erfinv` 本身较快，实际训练/解析热路径收益取决于调用占比（不夸大）。
- **回归**：全量 pytest **1318 passed / 8 warnings / 46.11s**（1314 存量 + 4 新测试），0 failed。
- **宪章符合**：C1（不改变单位/不等式/数值）、C2（memoization 推导）、C3（不触碰任何协议/记账/truth 路径——纯 utils 函数）、C4（低创新，标注为性能对齐项）。
- **冻结影响**：零——`effective_sha256` 仅在冻结 manifest 校验；O7 属 utils 内部、不改变 config 语义。

### 16.2 O2 已实施（2026-08-29，零语义）——deflection canonical 路径向量化

- **改动**：`uav_isac/physical/deflection.py` 新增 canonical 批量分支（`dd_gain_mode=="continuous" and not use_swerling`）：`d_raw_all`/`phys_gain_all`/`chi_rep_all` 以 (K,K,Q) 数组计算，条目按原 (i→j→q) 顺序组装。**binary 与任何 Swerling 组合保留原标量循环**（Swerling RNG 抽取顺序是流状态，不得重排）。
- **关键设计一（浮点诚实）**：批量 d_raw 改用**与标量 `compute_raw_deflection` 完全相同的表达式序列**（`P·α²·G·T_sym·N·n_cpi / max(noise_psd, 1e-15/implied_bw)`）——实测差异仅 1-2 ULP（浮点交换律/指令级），**不宣称逐位**（C1）；对拍断言 rtol=1e-13。
- **关键设计二（广播语义）**：`power[:, None, :]` 必须显式（(K,Q)→(K,1,Q)）——曾误写 `power * alpha**2` 致 i 维右对齐错位到 j 维（对拍测试立即暴露，第一版 3/5 失败即此因；这是"对拍性质测试"价值的直接证据）。
- **关键设计三（batch 的 inf 约束）**：`compute_dd_phys_gain_batch` 在 support 掩码前拒绝非有限 tau/nu（otfs.py:169 raise）——几何层对无效对（i==j 等）产出 inf tau，批量分支先 `np.where(isfinite, tau, 0)` 掩码（gain=0，语义与标量 skip 一致），组装列表推导保留 `if not np.isinf(tau[i,j,q])`（复刻标量 continue）。
- **性质测试**：`tests/test_deflection_vectorized_equiv.py` 5 项——canonical 随机几何、sensing 功率矩阵、out-of-support（tau≥1/Δf 场景；**注意 Doppler 维度由 UAV 速度贡献、目标速度不产生 nu**——初版用 1e6 m/s 目标速度构造失败即此，改用 30km 距离目标）、binary 保标量、Swerling 保标量与 RNG 顺序（双独立同 seed 实例）。
- **基准**：K=Q=8（128 条二基地链路）：批量 0.99 ms vs 标量 1.61 ms（**1.62x**，deflection 子阶段）。诚实标注：这是 deflection 计算子阶段收益，占单帧 step 总耗时（~26 ms）的一小部分；其对 K16 计算超时（P95 127-218 ms）的缓解需整帧核算后确认（记录于路线图 O4 关联）。
- **回归**：全量 **1323 passed / 8 warnings / 46.75s**（1318 + 5 新），0 failed。
- **宪章符合**：C1（浮点诚实，不宣称逐位；广播语义正确）、C2（批量/标量同公式推导 + 对拍）、C3（不触协议/记账/truth 路径；binary/Swerling 分支逐位不动）、C4（性能对齐项，低创新）。
- **冻结影响**：零——canonical 数值与标量 1-2 ULP 等价、结构不变；`effective_sha256` 不变。

### 16.3 R15 已实施（2026-08-29）——QoS 验收门限单一来源

- **改动**：
  - `config/params.py`：新增 `marl.qos_acceptance_floors: List[float] = [0.60, 0.70, 0.80]`（顺序 worst/weak3/steady）与 `marl.qos_acceptance_wilson_lcb_floor: float = 0.70`——验收门的**唯一权威源**。
  - `tools/assert_gate_thresholds.py`：`acceptance_floors_from_config(path)`（无 path → canonical 默认 == params 默认；有 path → 严格 loader 读取，失败 **fail-loud**（RuntimeError），绝不静默回退默认——防止伪造门判定）；`medium_gate_checks`/`assert_medium_gate` 增加 `floors` 参数；`assert_gate_from_csv` 增 `floors`；CLI 增 `--config`。（`MEDIUM_FLOORS` 保留为默认副本，值不变。）
  - `tools/report_blind_certification.py`：`--config` 读取 acceptance floors + LCB floor；feasible 掩码、point/LCB 门与 `assert_medium_gate` 均用 config 源。
- **语义设计（C1）**：奖励地板（`coord_reward_*`）与通信约束地板（`comm_qos_*`）**保持独立语义**（训练/在线约束 vs 统计认证），默认数值一致（0.6/0.7/0.8）由 `tests/test_qos_gate_provenance.py` **以测试锁定**（非共享引用）——改训练奖励不会悄悄移动认证门，反之亦然；验收门的 weak3=最弱 3 目标均值 与 `task_constrained_qos_floors[3]=3`（bottom-k 顺序统计）数值同 3 但语义不同，已在注释中显式区分。
- **性质测试**：`tests/test_qos_gate_provenance.py` 6 项——默认==params（worst/weak3/steady 逐值）、LCB floor==params、四源（acceptance/reward/comm）数值一致锁定、config 门驱动（值过默认门不过收紧 config 门 → 断言失败；同值默认门放行）、语义独立（改 acceptance 不动 reward/comm）、坏值 fail-loud。
- **全量回归**：**1330 passed / 8 warnings / 46.56s**（1323 + 7 新，0 failed）。
- **冻结影响**：零——未改任何 manifest 键语义；新增键默认值与既有三地板数值一致；`effective_sha256` 不变（需冻结提交时与 O7/O2 一并收口）。

### 16.4 O8 已实施（2026-08-29）——maxmin 复用/热启动 T5 化（P0 收尾）

- **勘察结论**：reuse/热启动机制**已存在**——`replicated_local_row_maxmin_power`（maxmin_power.py:354）支持 `previous_local_power_w/prices/cache_valid` + `reuse_relative_tolerance`（默认 0.0）；对偶价格热启动 `distributed_dual_maxmin_power(initial_prices=...)`（:911）与列生成 `incumbent_power_w`/`reuse_primal_duals`（:1193/:1066）亦然。**本轮不做新机制，而是 T5 化**：把现有复用从无证书启发式提升为可证机制（C1/C2），并补性质测试锁定。
- **T5 定理（三性质，均已落地实现中）**：
  1. **构造性可行性**：held 按行归一化后按**当前** budget 缩放（:592-600）→ 非负且行和==budget；可行性由构造保证，复用不需重解；
  2. **ε-最优性证书**：`lower = min_q Σ a·p`（主值）与 `upper = Σ b·max_q(price·a)`（对偶上界，:601-612），弱对偶 `upper ≥ OPT ≥ lower`；复用以 `(upper−lower)/upper ≤ reuse_tolerance` 为门（:613-619）——**gap 是可计算证书，非信仰**；合法域 [0,1) 由校验强制（:462-468）；
  3. **同几何不变性**：视图一致时组装==集中 LP 最优（:379-383）；gap=0 复用即最优。
- **性质测试**：`tests/test_maxmin_power_reuse_theorem.py` 4/4——① 复用输出行和==budget 且非负（构造性）；② 同视图 tol=0.1 全复用、gap≤0.1 且组装 worst==集中 LP 最优（证书 + 最优性）；③ tol=0 时 gap=0 才复用（同几何不变）且 worst==OPT；④ **默认路径零语义**：tol=0（默认）提供缓存也不触发复用，输出与无缓存路径**逐位一致**。
- **全量回归**：**1334 passed / 8 warnings / 45.39s**（1330 + 4 新，0 failed）。
- **宪章符合**：C1（弱对偶方向、合法域 [0,1)、不宣称无证书优化）、C2（三性质均可证 + 测试）、C3（复用不读新输入/不越记账；默认路径不动）、C4（把既有启发式提升为带证书机制）。**零语义**（无源码改动——本轮纯理论化+锁定）。
- **下一步关联**：开启 reuse（tol>0）的整帧收益需在 K16 计算身份上 A/B 验证（路线图 O4 关联）；P0 批次至此收尾（O2/O7/O8 + R15 口径），时序进入"冻结 → P1（O1/O5）"。

### 16.5 R14 已实施（2026-08-29）——哨兵统一（命名+域隔离，零语义）

- **语义表（全树扫描 ~88 处真哨兵，归并 7 组）**：
  1. **时间域**：env_core 的 last_seen 类字段用 `-10**8`（比较侧 `> -10**8` 判"已收到"）与 `-10**9`（初始化/serde 侧）表达同一语义（"从未收到时间戳"）——异值同义（历史债务）；
  2. **年龄域**：`-1` 两处语义——persistent_geometry_execution（"已过期、冻结不再老化"，`next_age[expired]=-1`、对角 `= -1`）、owner_local_physics（"无缓存年龄"，`target_age=np.full(Q,-1)`，缓存时覆盖为非负）——同值异义；
  3. **索引域**：`-1`——env_core 运动目标（_distributed_movement_target/executed_target/probe_target，"无目标分配"）、evidence 系 owner（`owner < -1` 拒绝非法，"无 owner"）——同值异义（索引域）；
  4. **数值域**：`±np.inf`（LP 上下界/候选最优/可行性阈值）语义统一、无歧义，仅文档化。
- **实施（全部数值不变→零语义）**：
  - 新模块 `uav_isac/utils/sentinels.py`：FRAME_NEVER=-10**8、FRAME_NOT_APPLICABLE=-10**9、AGE_EXPIRED=-1、AGE_NO_CACHE=-1、TARGET_INDEX_NONE=-1、OWNER_INDEX_NONE=-1、VALUE_UNBOUNDED_NEG/POS；模块文档记录域标签与"同值异域"警示；
  - env_core：`-10**9` 初始化/serde（~22 处）→ FRAME_NOT_APPLICABLE、`> -10**8` 比较（7 处）→ `> FRAME_NEVER`、运动目标 `= -1`（4 处）→ TARGET_INDEX_NONE；
  - persistent_geometry_execution：年龄置位（4 处）→ AGE_EXPIRED；
  - owner_local_physics：无缓存年龄（1 处）→ AGE_NO_CACHE；
  - evidence/quantized_evidence_audit/evidence_oracle_audit：`owner < -1` 判定（3 处）→ `owner < OWNER_INDEX_NONE`。
- **性质测试** tests/test_sentinel_semantics.py 10/10：七常量历史值保持（零语义）；同值异域行为方向（AGE_EXPIRED≥0=False=冻结不老化 vs AGE_NO_CACHE>max_age=False=永不过期——同一 -1 两种相反行为）；时间契约（FRAME_NOT_APPLICABLE>FRAME_NEVER=False=初始化值正确判"未收到"）；源码锁定 6 模块（防回退裸字面）。
- **全量回归**：**1344 passed / 8 warnings / 49.06s**（1334 + 10 新，0 failed）。
- **宪章符合**：C1（零语义：数值不变；异值同义/同值异义显式化而非静默统一——统一是行为改变需 serde 双侧同步，列后续）、C2（域隔离语义表）、C3（不触协议/记账/truth）、C4（命名哨兵+域隔离+源码锁定）。
- **后续项（登记不实施）**：时间哨兵 -10**8/-10**9 数值统一（env_core get/set_state 双侧同步 + replay 哈希变更，走冻结+1）；`-1` 三域彻底分离（改 AGE_NO_CACHE 独立值，冻结后批次）。

### 16.6 O1 已落地（2026-08-29）——双基地可达性天花板证书（P1 第一批，证书层）

- **新模块** `uav_isac/physical/reachable_deflection.py`（无状态、零 RNG）：
  - `target_reachability_ceiling(A, b, p_fa, pd_gate) -> (reachable, d_max)`：`D_q^max = Σ_i A[i,q]·b_i`（全队把全部预算投 q 的 deflection 上限，线性叠加）与 T0 门限 `D_c(q) = (Q⁻¹(P_FA)−Q⁻¹(P_D))²`（复用 detection.minimum_deflection_for_detection_probability 反演）比较 → `reachable[q] ⟺ D_q^max ≥ D_c(q)`；
  - `admission_mask_and_worst_deficit(...) -> (admission, worst_unreachable_q, worst_deficit)`：准入掩码 + **最不可达目标**（最大 gate deficit，O5 运动层输入）。
- **T6 定理（两个被测试锁定的推论）**：
  1. **单调剪枝（T2 化）**：不可达 q 对任何功率分配（`Σ_i p_iq ≤ Σ_i b_i`）的 deflection `≤ D_q^max < D_c(q)` → 排除不可达目标**不丢任何可行解**——O1 准入剪枝的正确性证书；
  2. **视图单调**：`A1 ≤ A2` 逐元素 ⟹ 不可达集单调收缩 → 准入集帧一致（无帧间振荡）。
- **no-truth 边界（C3 源码锁）**：模块签名仅接受公共增益视图 + 预算 + 检测常量——无 simulator truth、无 RNG（测试用 inspect 断言签名与禁词）。
- **性质测试** `tests/test_reachability_ceiling.py` 6/6：T0 一致性（D_c 与 detection 反演逐位一致）、充要闭式、**不可达永不过门**（任何分配下 deflection < D_c，直接验证 T6-1）、视图单调（不可达集收缩）、最差 deficit 供 O5、no-truth 签名锁。
- **全量回归**：**1350 passed / 8 warnings / 47.74s**（1344 + 6 新，0 failed）。
- **宪章符合**：C1（充要闭式、弱对偶方向、不把启发式当定理）、C2（T6 三推论均被性质测试锁定）、C3（无 truth、零 RNG、预算为联合 RF 上限）、C4（把\"物理可达性几何论证\"转成**结构生成前的单调准入证书**——该项目此前仅有事后可行性检查）。
- **下一步（已登记）**：超边准入接线——默认 OFF 旗标 `hyperedge_reachability_admission_enabled=false` 接入候选生成（不可达目标不入候选、最不可达目标喂给运动层 O5），随后诊断身份 A/B（先 10-20 seed，paired bootstrap）。
