# Deep Audit of the LD3 System — 2026-09-04

> 深度审计报告（2026-09-04）。
> 审计对象：D:\BYLW\LD3 多无人机分布式 ISAC 仿真系统。
> 方法：5 个并行维度子代理（源码 / 测试 / 配置 / 依赖-CI / 文档）+ 全量 pytest 实测 + 正式门槛与身份校验一手重测 + git 状态核验。
> 核心纪律：**不信任 2026-08-29 与 2026-09-03 审计报告的结论——关键项全部独立重测**。审计只读，未修改任何代码 / 配置 / 文档 / 数据。

---

## 0. 执行摘要

**总体判断：当前文件状态与 2026-09-03 审计快照基本一致，其登记的 P0（重复 YAML 键）仍未修复——全量测试当前为 2 failed，不是全绿；主风险仍是"工作树长期不提交"。另在本轮新发现一个高危 `git add -A` 污染/丢失隐患。**

- 全量测试（本机 venv 实测）：**1515 collected / 2 FAILED / 1513 passed / 8 warnings / 107.9s**。2 个失败与 09-03 的 P0 同根因（`config/exp_800_k8q8.yaml` 重复键 `target_kl`：行 99=0.003、行 104=0.02）。
- 正式认证门槛当前 **fail-closed**（`assert_formal_gates.py` exit 1，无 post-G2 认证产物登记）。
- **可执行 profile 的 strict 身份校验除 2 项外全部通过**——功能合规由工作树修补达成（k16q16 bank 已为 schema_version=2 / reset-distribution/v2 / 指纹匹配），残余 2 项均属"无版本锚点"（git 工作树脏、manifest/profile 未跟踪）。**这比 09-03 审计报告的配置/种子合规状态显著改善。**但配置子代理另报：**97 个使用 `1130_k8q8_blind` bank 的配置中有 65 个（k12/k10 分布式系族）是 identity 错配**（K12Q12/1386m 等却按 K8Q8/1130m 银行评估），其正式认证同样不可达——见 N6。
- results/ 数据树冻结于 08-28：979 顶层目录 / 6326 文件 / 12.92 GB；`best_restored.pt` 923 / `risk_critic_final.pt` 680 —— 08-28 后零新实验。

---

## 1. 新发现（本轮一手确认，前两次审计未登记或已变化）

### N1【P0 持续】重复 YAML 键 `target_kl` 仍未修复，全量测试 2 红
- 一手扫描全部 config/*.yaml（362 个）重复键：**仅此一个文件**存在重复映射键，位于 `marl:` 块内：行 99 `target_kl: 0.003`、行 104 `target_kl: 0.02`（PPO KL 目标，语义互斥）。
- 全量 pytest 实测失败两处（同根因）：
  - `tests/test_config_validation.py::test_all_versioned_yaml_configs_match_the_schema`
  - `tests/test_audit_fix_regressions.py::test_federated_region_configs_are_standalone_and_inherit_base`
  - 错误：`ValueError: duplicate YAML mapping key 'target_kl' at line 104`（`config/params.py:38`）。
- 机理：工作树 `config/params.py` 的 `_UniqueKeySafeLoader` 已能拒绝重复键，但该文件的两行冲突是**先天存量缺陷**（初始提交即含），不是加载器损坏。`fed_region_A/B/C.yaml` 以 `extends: exp_800_k8q8.yaml` 连带中毒。
- 裁决提示：PyYAML last-wins 下**历史生效值恒为 0.02**（行 104 后覆盖行 99）；保留 0.02（删 0.003）为零语义变更，保留 0.003 则改变实际生效值。哪个为本意需作者裁决。

### N2【新·高危】`git add -A` 会连带 1.45 GB 代理产物入库
- `.gitignore` **未覆盖** `.arts/`（609 文件 / 1.45 GB）、`.codegraph/`（40 MB）、`.workbuddy/`、`.agents/`、`.claude/`、`.codeartsdoer/` 及根目录 `_*.py`/`_*.log` 调试脚本。
- 一手实测 `git add -A --dry-run`：将暂存 **840 个文件**，其中约 **608 个来自 `.arts/`**（`algorithm_audit` 555 + `perf_audit` 53，含 263 `.pt`/91 `.zip`/91 `.csv`）。一次不谨慎的 `git add -A` 会把代理产物与调试垃圾提交入库，且可能冲掉未提交源码。
- 建议：提交收口前必须补 `.gitignore`（`.arts/`、`.codegraph/`、`.workbuddy/`、根调试文件），并**杜绝裸 `git add -A`**。

### N3【持续】R3 工作树与 HEAD 的裂口维持（未跟踪项继续增加）
- HEAD 仍为 `56e8e2c`（Author Date 一手确认 2026-08-26）。08-26 之后全部修复（严格加载器 +597、CI 门禁、`constraints-ci.txt`、新模块/新测试/新文档）尚未入库。
- dirty 组成一手实测：**91 M / 14 D / 129 ?**（09-03 为 122 ?，未跟踪新增 7 项）；`advice/003-016` 14 个删除未提交。
- 未跟踪 uav_isac 源码 12 个；未跟踪测试 23 个；未跟踪 config yaml 36 个；未跟踪 docs .md 15 个。
- **关键悖论仍成立**：R1/R2/R10-R15 等修复证据（测试）本身就是未跟踪文件——从 HEAD 检出既缺模块、也缺"修复存在"的证明。

### N4【改善】配置/种子库合规已由工作树达成（较 09-03 变化）
- 09-03 报告"manifest 钉选 k8q8 bank 三不合规"。一手重测：可执行 profile `exp_strict_distributed_k16q16.yaml` `extends: exp_strict_distributed_no_truth_pilot.yaml`，并钉选 `config/stratified_seeds_1130_k16q16_blind.json`（schema_version=**2**、fingerprint_version=**reset-distribution/v2**、sampling_seed=20260828、source_config=exp_strict_distributed_k16q16.yaml、指纹 0f5474b33951e199）。
- `check_system_identity.py --strict --manifest config/exp_strict_distributed_k16q16.yaml` 一手实测：全部身份/OTFS/c_det/seed/extends 项 **OK**，**仅 2 项 FAIL**：`git workspace is dirty`、`manifest is not tracked by git`（profile 自身未跟踪）。
- 结论：功能合规已达成，缺的只是**版本锚点**（入库后 strict 即可通过）。manifest 自身仍保留历史兼容指针并单独 strict 时按设计 fail（文档已声明）。

### N5【持续】正式认证（C7）线程前置变量仍未满足
- 一手实测进程环境：`OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS / NUMEXPR_NUM_THREADS` **全部为空**；`validate_formal_run`（`uav_isac/utils/reproducibility.py:313`）要求四变量均为 `"1"`。
- 结合 N3/N4，C7 正式 100-seed 盲测当前仍无法启动，阻塞项为：P0 键冲突、工作树脏 + manifest/profile 未跟踪、线程变量未设。

### N6【新·P0】97 个 bank 消费者的 identity 错配（配置子代理一手全量扫描）
- 全量 362 yaml 严格加载：**358 PASS / 4 FAIL**（4 个失败全为 N1 重复键：`exp_800_k8q8.yaml` + 经 `extends:` 继承的 `fed_region_A/B/C.yaml`）。
- **97 个配置钉选 `config/stratified_seeds_1130_k8q8_blind.json`，其中仅 32 个 K8Q8/1130×1130 匹配——其余 65 个 identity 错配**：57× K12Q12/reg1386×1386、5× K10Q10/reg1265×1265、2× K10/K12（800×800）、1× K4Q2（manifest 自身，其 bank 指针仅为历史兼容）。即 k12/k10 分布式系族当前在 K8Q8 区域 bank 上评估，正式指纹门将拒绝这些配置的认证运行。`trainer.load_stratified_seed_split` 只校验 quarantine、不校验指纹，故它们可非正式运行而不告警。
- 其它语义失配（子代理）：346/362 仍用旧 `dt_frame` 感知计费（仅 strict 链 12 个用 manifest 钉选的 `cpi_frame`）；122 个 `checkpoint_confirmation_enabled=true`（manifest=false）；280 个 `use_difference_reward=true`（manifest=false）；44 个含 ground/central 融合（违反 strict 无地面链范围）；manifest 钉选 100 kHz 带宽但冻结 profile 继承 500 kHz（未被身份门校验，属"死文本"）。
- 所有旧 bank（含 `_v2`）schema/指纹均不合规；`980_k6q6` 的 test 分裂仍包含全部 5 个 quarantined 种子。
- 加载器本身强健（8/10）：`_UniqueKeySafeLoader` 拒绝重复键、未知键拒绝、`validate()` 全量递归类型 + 物理不变量（P_sense≤P_sense_max≤P_isac_total、CPI≤dt、QoS 地板排序、no-truth⇒tracking∧local-belief、种子唯一等）；缺口为加载期不校验 bank 存在/指纹、无 manifest 链绑定、`n_cpi>1` 仅注释未设闸。

---

## 2. 上次风险登记（R1-R20，09-03）独立复核表

| # | 上次问题 | 本轮一手复核证据 | 判定 |
|---|---|---|---|
| R1 | manifest+pilot 无法构造训练 actor | `check_system_identity.py --strict` 对 profile 的目标/comm/tracking 全 OK；profile 链可解析 | ✅ 配置层已达成 |
| R2 | PPO KL 爆炸→NaN | 全量套件内 trainer 相关测试全绿（含 stable_ppo_ratio 回归） | ✅ |
| R3 | 重构未提交 HEAD≠当前 | dirty 91M/14D/129??；HEAD 08-26；manifest/profile 仍未跟踪 | ❌ **持续** |
| R4 | CI 硬读未跟踪 CSV | CI 在 Windows 参考环境装包带 constraints、不依赖 results/；test_assert_gate_thresholds 用 tmp_path 合成 | ✅ 已修复 |
| R5 | 权重冗余 | 一手实测 best_restored.pt=923、risk_critic_final.pt=680；树未增长（12.92GB） | ⚠ 未处理，口径修正 |
| R6 | summary 覆盖 23.5% | 一手：230/979 目录含 summary | ❌ 仍存在 |
| R7 | 调试目录混放 | `_` 前缀 + smoke 目录持续存在 | ❌ 仍存在 |
| R8 | 正式 bank 100 种子无产出 | results 树 08-28 后零新实验 | ❌ 仍存在 |
| R9 | env_core 超大 | 一手 `wc -l`：env_core **12,535** / trainer **8,170** | ❌ 维持（未继续增长） |
| R10 | env_core 全吞异常 | 源码子代理：仅存 `:9165 except (ValueError, ImportError): pass` 与 `:3054 bare except Exception` 低危点 | ⚠ 基本修复，残两点 |
| R11 | hyperedge 静默全零 | 源码子代理：feasible/inf cost/fail-closed 已落地 | ✅ 已修复 |
| R12 | capability 不查 success | 源码子代理：success/形状/有限性 → RuntimeError | ✅ 已修复 |
| R13 | `_strip_dead.py` 滞留 | 根目录调试脚本仍在（见 N2 高危面） | ❌ 仍存在（恶化面向） |
| R14 | 哨兵同值异义 | `utils/sentinels.py` 存在且 71 处一致引用；FRAME_NEVER/FRAME_NOT_APPLICABLE 未统一（文档已声明） | ✅ 已实施 |
| R15 | QoS 门限多源 | 存在 `params.py` floors + wilson LCB；但源码子代理发现 **QoS 地板字面量散 13+ 处**、边界集三处不一致 | ⚠ 部分（新增一致性质疑） |
| R16 | venv 缺 sklearn / 注释失实 | requirements 含 scikit-learn；`constraints-ci.txt` 提供精确锁定；ci.yml 用 `-c` | ✅ 已修复 |
| R17 | torch.load 不一致 | `utils/checkpoint_loading.py:373` 显式 weights_only=True 无回退；回归测试锁定 | ✅ 已修复 |
| R18 | 文档 GBK 乱码 | 一手扫描核心文档 0 U+FFFD（仅 09-03 文档内 6 个为引用的历史乱码证据） | ✅ 已修复 |
| R19 | 根目录调试文件 | `_*.py`/`_*.log` 约 20 个仍在；且未忽略、并入 git add -A 面 | ❌ 仍存在（恶化面向） |
| R20 | 空 __init__/死叶模块 | 未逐项复核（低危，backlog）；源码子代理判定 11 个小模块为合理原语而非 stub | — 未复核 |

**小结：10 项已闭环 / 8 项仍存在或部分 / 1 项未复核（口径与 09-03 略有出入，因其自表亦有计数误差——见 §3 文档维度）。**

---

## 3. 各维度一手复核要点

### 源码（uav_isac）
- **导入健康极好**：128/128 模块在 pytrch_ven 下 import 成功，无运行时断链。
- 巨型单体：`env_core.py` 12,535 行（74 方法）、`trainer.py` 8,170 行（29 顶层函数）；`networks.py` 2,690 / `hyperedge.py` 2,094 / `maxmin_power.py` 1,986 / `feasibility_oracle.py` 1,724 等。
- **结构风险**：仅 40/128 模块在部署运行时路径（env_core+trainer）可达；其余 88 个（约 2/3）只被 tests/scripts/tools 引用，构成未接线的"认证/研究并行层"（`certified_hierarchical_controller`、`persistent_geometry_execution`、`certified_geometry_repair`、`pwl_pd`、transport/digest 等），存在与 env_core 内联逻辑**重复实现漂移**风险。
- 具体缺陷：`env_core.py:10506-10507` 死分支 `if replicated_power is not None: pass`（功率求解静默无操作）；`:9165` 吞错；`:3054` 裸 `except Exception:`；QoS 地板 `0.60` 字面量散 13+ 处；QoS 地板边界集在 3 处控制器间不一致（`0≤x<1` / `0<x<1` / `(0,1]`）；未播种 RNG 回退于 `belief.py:341`、`target.py:41`、`evidence.py:1136`。
- 健康分：**7/10**。

### 测试（tests）
- 191 个测试文件 / 约 1374 个 `test_` 函数；**0 skip / 0 xfail / 0 空体**；协调/功率/通信断言密度高（hyperedge 143、cost_aware 176 asserts）。
- 门槛测试是全套最强部分（篡改变异、SHA-256 绑定、真实 `git init` 的 HEAD-blob 绑定），但 autouse fixture **stub 掉 `_validate_release_source_binding`**（最严格的"当前证据来自干净已提交源码"属性从未真测），且 5 个门槛测试依赖 `results/` 产物、干净检出会静默跳过。
- 弱测试：`test_audit_certified_hierarchical_controller.py`（5 行）、`test_joint_structural_locator_replay.py`（inspect 源码子串）；空断言 `test_safe_p0.py:112`（`len>=0`）、`test_integrity_audit.py:340`（`aoi.max()>=0`）、`test_chunk_bptt_consistency.py:275`（`assert True`）；`test_local_pd_boundary.py:136-139` 自我吞判据。
- 覆盖缺口：`utils/seeding.py` 零引用；无端到端 `MAPPTrainer.train()` 多 episode；OTFS 仅间接覆盖。
- 健康分：**7.5/10**（扣分主因：1 个 P0 红、约 6 处真空断言、stub 的 release-binding、少量覆盖缺口）。

### 配置（一手 + 配置子代理）
- 362 yaml 严格加载 **358 PASS / 4 FAIL**（4 失败全为 N1：基文件 + 3 个 `extends:` 继承者）；加载器强健 8/10。
- 可执行 profile 链 + k16q16 bank 合规（N4）；manifest 与 profile **均未跟踪**；全矩阵仅此 **1 份合规 bank/profile**。
- **N6 risk**：65 个 k12/k10 配置钉选错配的 `1130_k8q8` bank——当前树内 **0 份可正式认证配置**（dirty/未跟踪 + 60 余错配消费者）。
- 语义失配：346/362 `dt_frame`、122 checkpoint-confirm、280 difference-reward、44 ground-fusion，均与 manifest 冻结相悖（strict 链除外）。
- 配置健康分：**6.5/10**（加载器脊柱健康，矩阵面与契约卫生不足）。

### 依赖 / 安全 / CI
- 无 2024-2026 已知高危/严重 CVE（torch 2.12.1、numpy 2.5.0、scipy 1.18、PyYAML 6.0.3 近当前版）；torch 安全加载器无降级回退且被回归锁定；Python 3.14.6 受支持。
- **中危**：无哈希钉死 lockfile；`constraints-ci.txt` 非参考 venv 精确冻结（传递版本漂移、缺 CUDA/可视化包）；requirements 仅下界、裸装不可复现；CI 装 CPU torch、参考为 +cu130。
- **低危**：`psutil` 声明但 0 处使用；约 10 处 `np.load` 未显式 `allow_pickle=False`。
- CI 优点：干净环境（无 venv 缓存）、不依赖 results/、全量测试 + `check_system_identity --strict` + `audit_detector_normalization --assert-ready` 都跑。
- 健康分：**7/10**。

### 文档
- 编码全干净（24 篇核心文档 0 U+FFFD）；README§4.1 与 EXPERIMENT_LOG 数字逐字一致；README 文件路径 claim 全有效。
- **失效引用**：`CURRENT_SYSTEM_MODEL.md`(:766/:572/:1106) 与 `ALGORITHM_EVOLUTION.md`(:496/:501/:781/:1069) 以现在时引用 08-25 已删除模块（`audit_structure_regret`、`audit_l3_l2_alternating`、`audit_horizon_joint_oracle`、`congestion_relief`、`verify_v3_t3`）；仅 CODE_STRUCTURE_MAP 正确按"删除记录"表述。
- **文档纪律漂移**：约 17 篇单实验/单机制专题 .md 违反"四文档体系、不得新增活动 .md"策略。
- **内部计数矛盾**：09-03 自述"20 项中 10 闭环 / 6 仍存在 / 3 恶化"，与其自家 R 表（11 闭环 / 5 仍存在 / 2 恶化、合计 21）不符，略微高估未闭环项。
- 健康分：**6.5/10**。

---

## 4. 优先级建议

### P0（本周，进入任何正式流程之前）
1. **裁决并修复 `exp_800_k8q8.yaml` 重复 `target_kl`**（N1）——推荐删行 99（0.003）保 0.02（零语义变更）；随后全量 pytest 应回 1515/1515。
2. **收口提交 + 补 `.gitignore`**（N2/N3）：一次性提交 12 源码模块 + 23 测试 + params.py 严格加载器 + manifest + profile + CI + constraints-ci.txt；提交前忽略 `.arts/`/`.codegraph/`/`.workbuddy/` 与根调试脚本；**杜绝裸 `git add -A`**。
3. manifest/profile 入库后，重建/核验 C7 前置：线程四变量置 `1`、按已就绪的 k16q16 身份推进。
4. **裁决并绑定 scale-matched seed bank（N6）**：为 k12q12/reg1386 与 k10q10/reg1265 系族重建并钉选对应 schema_version=2 bank（或将其改为只经 manifest 链正式运行），否则这些系族的认证运行被指纹门拒绝；同时规范化旧 bank（含 `_v2`）并修复 `980_k6q6` 的 quarantined 种子泄漏。

### P1
5. 停止 env_core/trainer 反向增长并拆分；对齐"认证/研究并行层"88 模块与 env_core 内联逻辑的一致性（重复实现漂移，N-source）。
6. 修文档 8 处失效引用与 09-03 内部计数；将约 17 篇专题 .md 并入 EXPERIMENT_LOG 或归档。
7. 测试补强：把门槛测试的 release-source-binding stub 还原为真测；`results/` 依赖跳过显式化/CI 门控；QoS 地板收敛为单一命名常量（源码子代理 §15）。

### P2（不再阻塞认证）
8. 清理根目录调试脚本；移除或标注 `psutil`；统一 `np.load allow_pickle=False`；补充 belief/target/evidence 的 RNG 回退播种。

---

## 5. 复验命令（全部本机实测可重放）

```
# 全量测试（修复 N1 后应 1515 passed）
pytrch_ven\Scripts\python.exe -m pytest tests/ -q --tb=line
# 重复键扫描（仅 exp_800_k8q8.yaml）
grep -n target_kl config/exp_800_k8q8.yaml
# strict 身份（可执行 profile，仅剩 dirty+未跟踪 2 项）
pytrch_ven\Scripts\python.exe tools/check_system_identity.py --strict --manifest config/exp_strict_distributed_k16q16.yaml
# 正式门槛（当前 fail-closed exit 1）
pytrch_ven\Scripts\python.exe tools/assert_formal_gates.py
# git add -A 隐患面
git add -A --dry-run
# 数据树
find results -maxdepth 1 -type d | wc -l ; find results -name best_restored.pt | wc -l  # 923
# 线程前置变量（应全为 1）
echo %OMP_NUM_THREADS% %MKL_NUM_THREADS% %OPENBLAS_NUM_THREADS% %NUMEXPR_NUM_THREADS%
```

---

## 6. 本次未改动项声明

- 本审计**只读**：未修改任何代码 / 配置 / 文档 / 数据文件；仅清理了本人执行全量测试产生的临时日志 `_audit_pytest_20260904_083641.log`。
- N1 两个 `target_kl` 值哪个为本意，需作者裁决——本报告不代选。
- R20（空 __init__ / 死叶模块）未逐项复查，维持 backlog 定位。
- 配置维度子代理已在本轮完成并纳入（§1 N6、§3 配置）。

---

## 7. 审计后测试优化记录（同日追加）

> 本审计为只读快照；下文记录**审计后**按"逐步测试优化"执行的修复，用于追溯。修复本身已修改工作树（未提交），与 §0/§3 的"审计时状态"不冲突——§7 记录的是审计后的增量。

**第 1 步（P0 收口）**：删除 `config/exp_800_k8q8.yaml` 中重复键 `target_kl`（删行 99 的 0.003，保留 0.02——历史上 last-wins 生效值，零语义变更）。全量重跑：**1515 passed**（原 2 failed 全消，`fed_region_A/B/C.yaml` 经 extends 连带解封）。

**第 2 步（补针对性 config 覆盖）**：`tests/test_config_validation.py` 新增 `test_exp_800_k8q8_ppo_kl_target_is_pinned_and_unique`——超越 glob 扫描，明确断言该文件 `target_kl==0.02` 且无重复键（P0 回归守卫）。→ 该文件 16 passed。

**第 3 步（收紧真空/空洞断言，审计 §测试维度 b2）**：
- `tests/test_safe_p0.py:112` `len(selected_set)>=0` → 断言 `0<len(selected_set)<=len(entries)` 且 `selected_set⊆` 输入 entries 的 `(i,j,q)` 集合（不虚构条目）。
- `tests/test_integrity_audit.py:340` `aoi.max()>=0` → 断言 AOI 非负、有限（NaN 会使 min>=0 失败）、且 `max()>0`（追踪确实存活）。
- `tests/test_chunk_bptt_consistency.py:275` `assert True` 占位 → 断言 chunk-BPTT backward 确实产生有限梯度（非仅形状通过）。
- 三处改写均通过对应测试文件，无红。

**第 4 步（补零引用模块覆盖）**：新增 `tests/test_seeding.py`（`utils/seeding.py` 此前零测试引用）——4 个确定性测试：三次种子可复现、不同种子产生不同序列、无 CUDA 不抛错、torch 序列可复现；用 autouse fixture 恢复全局 RNG 状态避免串扰。

**第 5 步（全量复核）**：全量 pytest **1520 passed / 8 warnings**（1515 → +5：1 config + 4 seeding），exit 0。

**遗留建议**：`test_local_pd_boundary.py:136-139` 自我吞判据为作者显式注释的已知边界（非空 `any_positive` 时验证），维持不动；`test_audit_fix_regressions.py` 与 5 个依赖 `results/` 的门槛测试的"静默跳过"建议后续显式化/CI 门控；chunk-BPTT 的"分块 vs 全序列梯度等价"深测可作为后续独立步骤。所有修复尚未提交，随下个收口提交一并入库。

---

## 8. 审计后工程修复记录（同日追加，继 §7 测试优化后的代码/依赖修复）

> 本节记录"逐步修复"对 §1 新发现与 §2 风险表的工程性处置。每个改动均已做不回归核验。改动尚未提交（整树未收口）。

**第 1 步（阻断 `git add -A` 高危面，§1 N2 / §2 R19）**：`.gitignore` 追加 `.arts/`、`.codegraph/`、`.workbuddy/`、`.agents/`、`.claude/`、`.codeartsdoer/` 及根目录 `/_*.log`、`/_*.py`。核验：`git add -A --dry-run` 应暂存文件数 **840 → 223**，其中 `.arts/` 相关 **608 → 0**（1.45 GB 代理产物不再进入暂存区）。

**第 2 步（env_core 死分支 / 静默吞错，§3 源码维度 b1/b2）**：
- `env_core.py:10506` 的 `if replicated_power is not None: pass` 经复核为**意图性 dispatch 而非缺陷**：replicated 路径在更上方已完整应用功率/缓存/`_lex_mode`（原 :10348-10362），`pass` 用于跳过 intercept/task/maxmin 备选分支。**保留未改**（改动会影响分布式 replicated 功率研究线）。复核已注明。
- 新增模块级 `logging` logger；将两处候选生成静默吞错由 `except (ValueError, ImportError): pass / lex_l1=None` 改为 `logger.debug(...)`（原 :9165 候选生成、:9209 lex-L1），**行为不变、失败可观测**。
- 核验：`env_core` import OK；`test_integrity_audit.py` + `test_env_wrapper.py` 43 passed。

**第 3 步（未播种 RNG 回退，§3 源码维度 f）**：复核 `belief.py:341`、`target.py:41`、`evidence.py:1136` 的 `rng is None → default_rng()`：主 env 路径恒传入 `self.rng`（`env_core` 构造 Target:2420、BeliefManager:2445），回退只在**独立/standalone 构造**时触发——属意图性模式（且 `action.py`、`communication.py` 的默认已播种）。**不改动**（强制确定性会改变独立构造语义、可能回归其测试），判定低于本论修复优先级。

**第 4 步（声明未用依赖，§依赖维度）**：`requirements.txt` 删除从未被任何 first-party 代码 import 的 `psutil>=5.9`（grep 确认 0 处使用），并加追溯注释。核验：无 `psutil` import 依赖。注：`constraints-ci.txt` 仍钉 `psutil 7.2.2`，仅作版本上限、无害。

**第 5 步（全量复核）**：全量 pytest **1520 passed / 8 warnings / exit 0**——env_core 日志化与 requirements 改动均无回归（收集数不变）。

**遗留待授权**：git 收口提交（含 §7+§8 全部改动）需在作者授权后执行；提交前建议核验 `git add -A` 已收敛且不包含 `.arts/`。

---

## 9. `.arts/` 处置决定（作者 2026-09-04 拍板）

> 审计结论见本节下方要点；**处置决定：`.arts/` 保持被 `.gitignore` 忽略、不提交**（作者指令）。

**核实（2026-09-04 一手复验）**
- `.gitignore` 已含 `.arts/` ✓；`git ls-files .arts` = 空（从未被跟踪）✓；`git add -A --dry-run` = 0 个 `.arts/` 文件入暂存 ✓。

**审计要点（只读）**
- `.arts/` = architecture-v2/P1-P2 期间的**合法实验/性能产物存储**（`algorithm_audit` 555 文件 + `perf_audit` 53 文件，共 86 个运行目录；1.45 GB）。每目录为标准 6 件套（`best_restored.pt`/`risk_critic_final.pt`/`last_unrestored_diagnostic.pt`/`paired_eval.csv`/`run_manifest.json`(provenance)/`source_snapshot.zip`），`run_manifest` 记录 `git_commit 56e8e2c` + `git_dirty=true`，符合仓库运行 provenance 纪律。
- **无凭据**：收紧模式扫描 0 处真实凭据（早前 200 个命中均为 `target_tokens`/`token` 配置键误报）。
- **不受代码引用**：config/scripts/tools/uav_isac 均无 `.arts` 引用——外部工具写入，非仓库管理产物。
- **results/ 08-28 后"零新实验"的口径修正**：P1/P2 实验确实跑了，产物落在此 scratch 区而非 results/。
- **冗余**：263 个 `.pt`（每目录两份 ~7.2MB）主导 1.4GB；86 个变体目录高度同构。这些证据落在未登记、忽略、不进 results/ 管道、CI 不可见的区域内。
- **后续（如需）**：留作证据→先哈希清单再迁 results/formal_evidence 经正式门径注册；清理→先哈希清单备份再删，预估可回收 1GB+。当前不动作。

---

## 10. 开发模块合理性审计（同日追加，作者拍板：保留为主）

> 审计对象：`uav_isac` 全部 **128 个模块**（其中 81 个不在部署运行时路径）。本文的"开发模块"指这 81 个。
> 审计方法：全量引用扫描（import 图，128 模块逐一核验至少一处引用）+ 测试存在性（`tests/` 专属/间接覆盖核对）+ `docs/README.md` §3.3 与 EXPERIMENT_LOG / ALGORITHM_EVOLUTION 的机制登记交叉核对。
> 与 §3 源码维度（"88 个未接线认证/研究并行层"）口径的差异：本次按"模块级引用"而非"运行时可达性"逐项分类，81 与 88 之差来自 `__init__`/常量/工具脚本的计数归属不同，判定方向一致。

### 10.1 总判定

**绝大多数（约 75/81）合理保留**——它们是项目明确保留、带专属测试、有实验日志登记的研究/认证机制；**无真孤儿**（128 模块全部至少被一处引用，与 2026-08-25 源码审计一致，无死模块可删）；真正需要处置的只有一小撮（§10.3）。

### 10.2 A 类：合理保留（研究/认证机制层，均有专属测试与文档登记）

- **认证控制器簇**：`certified_hierarchical_controller`、`certified_maxmin_power_controller`、`certified_geometry_repair`、`persistent_geometry_execution`、`owner_local_physics`、`causal_hierarchical_controller`、`causal_joint_plan`——L2/L3 证书机制，各带专属测试（`test_certified_*` / `test_persistent_geometry_execution` / `test_causal_*`）。
- **结构修复/MILP 簇**：`minimum_intervention_repair`、`nested_task_repair`、`horizon_atomic_repair`、`fixing_conflict_filter`、`permission_cut_master`、`dual_guided_structure_repair`、`bottleneck_router`、`oracle_free_candidate_locator`——G4/OLCS 系列，实验日志有明确负/正结果记录（如 B2d/B2e 的 cert 判定）。
- **分布式信息/传输簇**：`progressive_information_transport`、`spectral_information_reuse`、`distributed_compute_fusion`、`digest_rendezvous`、`owner_*_transport`、`power_repair_transport`、`geometry_repair_transport`、`structure_sequence_transport`、`target_invariant_transport`、`protocol_fingerprint`、`ai_candidate_screener`——M 系/ISCC 研究线。
- **evaluation/audit 层（约 20）**：`certified_feedback`、`self_normalized_feedback`、`horizon_future_audit`、`horizon_transition_gate`、`transition_certificate`、`episode_joint_conformal`、`quantized_evidence_audit`、`finite_sample_feasibility`、`physics_interval_gate`、各类 calibration/wiring、`expert_arbitration`、`channel_margin` 等——支撑正式门禁的审计/校准工具，几乎全带专属测试。
- **agent 变体**：`equivariant_movement_plan`、`tica_actor`、`residual_actor`、`p0_fixed_agent`、`base_agent`——架构变体研究线，均有测试或工具引用。

### 10.3 B 类：存疑——两类需处置（4 个无专属测试小模块 + 1 个结构性风险）

**B1. 无专属测试的小模块（补测或归档）**
- `evaluation/metrics.py`——仅 `scripts/run_baselines.py` 引用、无任何测试（**最弱**）；
- `coordination/learned_move_ranker.py`、`coordination/local_move_ranker.py`——无专属测试（间接覆盖：`test_safe_checkpoint_loading.py` 装载/推理路径、`test_local_exchange_oracle.py` 特征函数）；
- `coordination/safe_structural_reduction.py`——S0-A 机制（README 已登记安全 PASS / 稀疏化 FAIL），无专属测试（间接覆盖：`test_minimum_intervention_repair.py` 的 `context_free_zero_gain_screen`）。

**B2. 结构性风险（非删除，需对齐）**：认证控制器簇与 `env_core` 内联逻辑存在**重复实现漂移**——`certified_hierarchical_controller` / `persistent_geometry_execution` / `certified_geometry_repair` 等是"影子认证"，而 `env_core` 用另一套内联控制流执行同一算法（`env_core` 未 import 它们）。两边语义漂移会让"证书"与"运行时"脱节。处置方向：把它们固化绑到运行时调用点，或显式声明为 shadow 并加"与 env_core 运行时语义一致"的对照测试（见 §10.5，未执行）。

### 10.4 C 类：死模块

**无**——128 模块全部至少被一处引用（与 2026-08-25 源码审计一致），无孤儿可删。

### 10.5 处置决定（作者拍板 2026-09-04）

- **保留为主，不删开发层**：它是"默认关闭但保留审计"的研究资产（`docs/README.md` §3.3 明示）。
- **小动作（已执行，见 §10.6）**：给 B1 的 4 个模块补最小行为测试。
- **中动作（未执行，待后续独立步骤）**：给影子认证簇加"与 env_core 运行时语义一致"的对照测试（目前各自测试只验证模块自身，不验证与运行时的等价性）；或按 §10.3 B2 显式声明 shadow 并固化绑定。

### 10.6 补测执行记录（同日追加）

**新增三个测试文件（§10.3 B1 处置）**：

1. `tests/test_evaluation_metrics.py`（13 个测试）——`metrics.py` 全部 7 个函数的行为锁定：avg/worst P_D、Jain 公平性（相等→1、单目标→1、空/全零→1、非均匀∈(1/Q,1)）、累积能量守恒、通信位求和、违约率（空→0）、`compute_episode_metrics` 完整 schema 与 steady 窗口尾部切片语义。
2. `tests/test_move_ranker.py`（11 个测试）——`local_move_ranker.py` 特征契约（宽度==`FEATURE_NAMES`、确定性、kind 独热顺序、owner 变化分数、增删边分数、形状校验、有限性）+ `learned_move_ranker.py` 的 `LocalMoveRanker`（归一化宽度校验、rank/positive 双头输出形状、std 钳制除零安全）。`FrozenLocalMoveRanker` 装载路径已有 `test_safe_checkpoint_loading.py` 覆盖，不重复。
3. `tests/test_safe_structural_reduction.py`（8 个测试）——S0 层专属语义：零增益边全覆盖离对边、feasibility/optimality 分割、选中零边报 repair toggle、正增益永不入屏、**精确零无容差**（1e-12 不入屏）、负/非有限/形状校验。

**独立运行验证**：三文件 **33 passed / 0 failed**（含 `--cache-clear` 复跑）。

**全量复核（受限环境下如实记录）**：本会话沙箱对平台临时目录的 `os.scandir` 拒绝（`WinError 5`），pytest `tmp_path` fixture 不可用，且 git/multiprocessing 子进程被拦截——全量结果为 **1442 passed / 104 errors / 7 failed**，其中 **1442 已包含全部 33 个新测试**，104 errors 全为 tmp_path 环境性失败，7 failed 全为 git/multiprocessing 子进程类环境性失败（`test_parallel_power_executor` ×2、`test_physics_closure_c0c1` entrypoint ×1、`test_reproducibility` ×4），**无一来自本次改动**（收集数 1553 = §8 基线 1520 + 33，完全对账）。在无沙箱限制环境按 §5 复验命令重跑预期 **1553 passed**。

**§10 自身状态**：审计结论（§10.1–§10.5）为只读登记；§10.6 为本轮补测执行记录（对应 §7/§8 的「同日追加」惯例）。B2 影子认证簇对照测试仍为**后续独立步骤**（§10.5 中动作），未在本轮执行。
