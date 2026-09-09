# Deep Audit of the LD3 System — 2026-09-03

> 深度审计报告（2026-09-03）。
> 审计对象：D:\BYLW\LD3 多无人机分布式 ISAC 仿真系统。
> 方法：4 个并行维度子代理（文档 / 配置 / 依赖 / 数据）+ 关键代码断言一手复核 + 全量 pytest 实测 + git 状态核验。
> 核心纪律：**不信任 2026-08-29 审计报告（含其 §12/§13/§16 修复声明）的任何结论——全部独立重测**。

---

## 0. 执行摘要

**总体判断：上次审计登记的 20 项风险中 10 项已闭环（R1/R2/R4/R10/R11/R12/R14/R15/R16/R17/R18），但 R3（工作树未收口）显著恶化，且本次发现一个新的 P0 级配置回归：重复 YAML 键已提交进 HEAD，当前全量测试 2 failed（上次审计以来首次不全绿）。**

- 全量测试（本机 venv 实测）：**1515 collected / 2 FAILED / 1513 passed / 8 warnings / 55.83s**（上次 1350 passed → +165 测试，修复工作持续但未提交）。
- 两个失败同源：`config/exp_800_k8q8.yaml` 第 99/104 行重复键 `target_kl`（**0.003 vs 0.02 语义冲突**）。
- 正式盲测认证（C7）三重前置阻断**全部仍在**：manifest 未跟踪、种子库三不合规、线程环境变量未满足——且 `validate_formal_run` 本轮还**收紧**了（新增 `schema_version==2` 门槛）。
- results/ 树自 08-28 起**零新实验**——上次审计后的全部工作都在代码 / 测试 / 文档层。

---

## 1. 新发现（本次一手确认，上两次审计均未记录）

### N1【P0】重复 YAML 键 `target_kl` 已提交进 HEAD，当前 2 个测试红

- `config/exp_800_k8q8.yaml:99` → `target_kl: 0.003`；`:104` → `target_kl: 0.02`。两值均为 PPO KL 目标、语义互斥。
- 失败测试（实测复现）：
  - `tests/test_config_validation.py::test_all_versioned_yaml_configs_match_the_schema`
  - `tests/test_audit_fix_regressions.py::test_federated_region_configs_are_standalone_and_inherit_base`
  - 错误：`ValueError: duplicate YAML mapping key 'target_kl' at line 104`
- **回归链（git 考古确认，含对初稿结论的修正）**：
  1. 重复键是**先天缺陷而非近期回归**：`git show d58e2a9:config/exp_800_k8q8.yaml`（初始提交）即含两行（:100=0.003、:105=0.02）；HEAD（56e8e2c）未触碰这两行。
  2. HEAD 的 `config/params.py` **不含**重复键拒绝逻辑（`git show HEAD:config/params.py | grep -c "duplicate YAML mapping key"` = 0）→ 旧加载器静默 last-wins，CI 在 HEAD 上"绿"，但 **exp_800_k8q8.yaml 自项目诞生起实际生效值就是被静默覆盖的 0.02**（两键同在 `marl:` 映射内，PyYAML 后者覆盖前者）。
  3. 工作树 `config/params.py` 有 **+597 行未提交**改动，新增 `_UniqueKeySafeLoader`（拒绝重复键）→ 本轮暴露先天冲突。
  4. `config/fed_region_A/B/C.yaml`（3 个未跟踪新文件）`extends: exp_800_k8q8.yaml` → 连带中毒。
  5. 全 config/ 扫描：仅此一个文件存在重复 target_kl；manifest 无 target_kl 键。
- **影响**：无论加载器是否收紧，exp_800_k8q8.yaml 都存在一个被静默丢弃的配置值——训练配置语义不可信。需作者裁决保留哪个值（0.003 是该文件历史值；0.02 是 manifest/strict 线口径）。**注意**：由于 PyYAML last-wins，全部历史运行的实际生效值都是 0.02——保留 0.02（删 0.003）才是零语义变更；保留 0.003 则改变生效值，属语义变更。
- **修复建议**：删除 :99 或 :104 之一（裁决语义）；fed_region_A/B/C.yaml 随基础文件修复自动通过；`_UniqueKeySafeLoader` 改动应随下个提交收口（它正是防止此类静默冲突的闸门）。

### N2【P0 持续】正式认证三重前置阻断全部仍在（一手重测）

1. **冻结 manifest 仍未被 git 跟踪**：`git ls-files config/system_manifest.yaml` = 空，且 `git status` 显示 `?? config/system_manifest.yaml`。"冻结身份"依然没有版本锚点。
2. **manifest 钉选种子库三不合规**（`config/stratified_seeds_1130_k8q8_blind.json` 实测）：`schema_version=1`、`fingerprint_version=None`、`sampling_seed=None`、`source_config=exp_800_k8q8_dcb_top1_scale.yaml`（DCB-top1 身份而非 strict）、`scenario_fingerprint=c7dd74ad93a9fbaf` ≠ strict 期望 `f3530f4c26265081`。manifest:101 该指针已加注释 "Historical compatibility pointer only"，但键仍钉着不合规 bank。
3. **validate_formal_run 反而更严**：移至 `uav_isac/utils/reproducibility.py:313`（原 :214），新增 `schema_version==2` 硬门槛（:343），其余要求不变（指纹匹配 :367-385、种子按序 :386-393、OMP/MKL/OPENBLAS/NUMEXPR 四变量全 "1" :419-427）。
- **结论**：C7 正式 100-seed 盲测在当前文件状态下依然无法启动，且门槛比上次审计更高。

### N3【恶化】R3 工作树与 HEAD 的裂口扩大

- git dirty：39M/15D/70??（08-29）→ **91M/14D/122??，共 227 项**（09-03）。
- **HEAD 提交日期实测为 2026-08-26**（56e8e2c，Author Date 一手确认）——即 08-26 之后的**全部**修复与重构（严格加载器 +597 行、CI 门禁、constraints-ci.txt、R1/R2/R10-R15 修复、R14/R15 新模块与测试）均未入库，积压已超一周；这些工作当前只存在于工作树与 `results/*/source_snapshot.zip` 中，一次误操作即可能丢失。
- 未跟踪 uav_isac 源码模块 8 → **12** 个（新增 `physical/movement_potential.py`、`physical/reachable_deflection.py`、`utils/checkpoint_loading.py`、`utils/sentinels.py`）。
- 未跟踪测试 12 → **23** 个（sentinel/qos_gate/reuse_theorem/physics_closure 等全部修复验证测试都在未跟踪之列）。
- **关键悖论**：R1/R2/R10-R15 的修复证据（测试）本身就是未跟踪文件——从 HEAD 检出既缺模块、也缺"修复存在"的证明。
- advice/ 目录工作树删除 003-016（14 个 D）未提交。

### N4【恶化】巨型文件在"架构重构"中继续膨胀

- `env_core.py`：10,837 行（08-29）→ **12,535 行**（+1,698，"P1/P2 in progress" 提交之后反而更大）。
- `trainer.py`：从 `environment/`（7,565 行）迁至 `agents/`，现为 **8,170 行**（+605）。
- 重构目前是"移动 + 继续往 env_core 堆"，拆分目标未达成（维持 backlog 定位，但应停止反向增长）。

### N5【事实修正】上次数据审计的 checkpoint 份数偏低

- 实测：`best_restored.pt` **923** 份（上次记 652）、`risk_critic_final.pt` **680** 份（上次记 664）。上次 ≈9.7GB 冗余估算应按新份数上调口径。
- 其余与上次一致：979 顶层目录 / 6326 文件 / 12.94GB / summary 覆盖 23.5%（230/979）/ `_` 前缀 330 + smoke 171 目录。
- results/ 树 08-28 后**零新增顶层目录**（`-newermt 2026-08-29` 命中 0）——审计后的工作全部在代码/测试/文档层，无新实验数据。

### N6【新】配置矩阵增长且首现严格加载失败

- config yaml：341 → **362**（+21）。严格加载（`config/params.py:1899 load_config`，工作树收紧版）：**358/362 通过，4 失败**（exp_800_k8q8 + 3 个 fed_region，全部同源 N1）。
- 上次"341/341 全绿"的记录对应旧加载器；本次失败是收紧后的加载器暴露的存量冲突，不是加载器本身损坏。

---

## 2. 上次风险登记（R1-R20）独立复核总表

| # | 上次问题 | 本次一手复核证据 | 判定 |
|---|---|---|---|
| R1 | manifest+pilot 无法构造训练 actor | manifest:92 `target_allocation_enabled: true`；pilot `comm_payload_mode=target_tokens`；按 run_mappo 接线三前置齐备 | ✅ 已修复 |
| R2 | PPO KL 爆炸→NaN | `agents/trainer.py:51 stable_ppo_ratio_and_approx_kl`（expm1 稳定式）、:4114/:4128 minibatch 拒绝、:4885 事务回滚 | ✅ 已修复 |
| R3 | 重构未提交 HEAD≠当前 | dirty 227 项；未跟踪源码 12 / 测试 23；manifest 仍未跟踪 | ❌ **恶化** |
| R4 | CI 硬读未跟踪 CSV | `test_assert_gate_thresholds.py` 全部 tmp_path 合成夹具，0 results/ 依赖；Windows 参考环境运行完整门禁 | ✅ 已修复 |
| R5 | 权重冗余 ≈9.7GB | 树未增长；份数修正为 923+680（上次 652+664 偏低） | ⚠ 未处理，口径修正 |
| R6 | summary 覆盖 23.5% | 230/979 = 23.5% 不变 | ❌ 仍存在 |
| R7 | 调试目录混放 | `_` 330 + smoke 171 不变 | ❌ 仍存在 |
| R8 | 正式 bank 100 test 种子无产出 | results 树 08-28 后零新实验 | ❌ 仍存在 |
| R9 | env_core 10,837 行 | **12,535 行** | ❌ **恶化** |
| R10 | env_core except-pass 全吞 | 现 4 处 except Exception 全部显式处理（条件 raise / 策略注释 / RuntimeError）；仅剩 :9165 `except (ValueError, ImportError): pass` 一处低危 | ✅ 已修复 |
| R11 | hyperedge 静默全零 | `hyperedge.py:73/220` feasible 标志 + inf cost + fail-closed | ✅ 已修复 |
| R12 | capability 不查 success | `capability.py:124-133` success/形状/有限性 → RuntimeError | ✅ 已修复 |
| R13 | `_strip_dead.py` 滞留 | 仓库根仍在（连同 _config_audit/_dep_audit 等调试脚本） | ❌ 仍存在 |
| R14 | 哨兵同值异义 | `utils/sentinels.py` 七常量 + 源码锁定测试（但模块本身未跟踪） | ✅ 已实施 |
| R15 | QoS 门限多源 | `params.py:1138-1140 qos_acceptance_floors` + wilson LCB floor + provenance 测试 | ✅ 已实施 |
| R16 | venv 缺 sklearn / 注释失实 | `find_spec('sklearn')=True`；requirements 10 项含 scikit-learn>=1.2；新增 **constraints-ci.txt**（10 直接 + 16 传递依赖精确锁定） | ✅ 已修复 |
| R17 | torch.load weights_only 不一致 | 项目级唯一调用 `utils/checkpoint_loading.py:373` 显式 True 无回退；run_mappo 改用 safe_torch_load | ✅ 已修复 |
| R18 | 文档 GBK 乱码 | 7 篇核心文档 U+FFFD=0、双编码 run=0（166/186/119 全部归零） | ✅ 已修复 |
| R19 | 根目录调试文件 | `_strip_dead.py`/`_c6_test_fail.log`/`_pytest_full*.log` 等仍在 | ❌ 仍存在 |
| R20 | 空 __init__/死叶模块 | 未逐项复查（低危，backlog） | — 未复查 |

**小结：11 项闭环 / 6 项仍存在 / 3 项恶化 / 1 项口径修正。**

---

## 3. 四维度子代理复核要点

### 文档（一手重测）
- 乱码：**已全部修复**（EXPERIMENT_LOG/CURRENT_SYSTEM_MODEL/ALGORITHM_EVOLUTION/README/CODE_STRUCTURE_MAP/FORMAL_PROTOCOL/COMPOSABLE 七篇全部干净）。
- 断链：**10 处未修**（与上次持平），集中在 CURRENT_SYSTEM_MODEL / ALGORITHM_EVOLUTION / CODE_STRUCTURE_MAP 三篇，指向 08-25 清理删除的 5 个模块（audit_structure_regret.py 等）。
- 新增 2 篇审计文档：ALGORITHM_OPTIMIZATION_AUDIT_2026-08-30、ALGORITHM_PRIMAL_DUAL_AUDIT_2026-08-31。
- CURRENT_SYSTEM_MODEL:1402 "paper/ 手稿仍为 2026-07-22 旧稿" **属实**（paper/ 仍只有 archive/）；:1385 C3 "λ* 熵正则化待接线" 与代码一致（未接线），声明无滞后。

### 依赖与运行时（一手重测）
- venv：Python 3.14.6 / torch 2.12.1+cu130 / numpy 2.5.0 / scipy 1.18.0 / pytest 9.1.1，46 包，与 constraints-ci.txt 锁定一致。
- psutil：requirements 声明但项目代码 0 处 import（仅 joblib/loky 机会性受益）——唯一残留的声明不一致，低危。
- CI：Windows py3.14 参考环境，装包带 constraints 锁，加跑 `check_system_identity.py --strict` 与 `audit_detector_normalization.py --assert-ready`；不依赖 results/。
- 线程四变量断言仍在（assert_formal_gates / reproducibility / parallel_power_executor / crash_isolated）。

### 数据（results/，只读）
- 树冻结于 08-28：979 目录 / 6326 文件 / 12.94GB；summary 覆盖 23.5%；git 仅跟踪 45 个历史资产文件。
- C2 证据 `results/_c6_strict_blind100_run.log` 仍在（16KB，08-28）。

### 配置（一手重测）
- 362 yaml，严格加载 358/362（4 失败同源 N1）；seed bank 钉选三不合规（N2）；strict pilot 关键 flag 生效值正确（N2 反向印证 R1 修复）。

---

## 4. 优先级建议

### P0（本周，进入任何正式流程之前）
1. **裁决并修复 `exp_800_k8q8.yaml` 重复 `target_kl`**（N1）——一行删除 + 作者确认语义；随后全量 pytest 应回到 1515/1515。
2. **收口提交**（R3）：一次性提交 12 个源码模块 + 23 个测试 + params.py 严格加载器 + manifest + CI + constraints-ci.txt。当前 HEAD 的"绿"是假象（加载器容忍重复键 + 缺模块缺测试）。
3. manifest 入库后，按 §13.3 流程重建 strict 身份 seed bank（schema_version=2 + fingerprint reset-distribution/v2 + strict 指纹）。

### P1
4. 停止 env_core.py 反向增长（N4）：任何新逻辑不得再进 env_core，拆分按文档 blueprint 执行。
5. 断链 10 处与 advice/003-016 删除一并随收口提交清理。
6. results/ 冗余与调试目录（R5/R7）维持只读，待哈希清单后处理。

### P2（不阻塞认证）
7. `_strip_dead.py` 等根目录调试脚本归档；psutil 声明移除或标注为传递受益；env_core:9165 的吞异常补注释或日志。

---

## 5. 附录：复验命令（全部本机实测可重放）

```
# 全量测试（预期修复 N1 后 1515 passed）
pytrch_ven\Scripts\python.exe -m pytest tests/ -q --tb=no
# 收集数
pytrch_ven\Scripts\python.exe -m pytest tests/ --collect-only -q | tail -1
# 重复键定位
grep -n target_kl config/exp_800_k8q8.yaml
# HEAD 已含冲突 / HEAD 加载器无拒绝（回归链证据）
git show HEAD:config/exp_800_k8q8.yaml | grep -n target_kl
git show HEAD:config/params.py | grep -c "duplicate YAML mapping key"
# 认证阻断
git ls-files config/system_manifest.yaml        # 空 = 未跟踪
python -c "import json;b=json.load(open('config/stratified_seeds_1130_k8q8_blind.json'));print(b['schema_version'],b['fingerprint_version'],b['source_config'])"
# 数据树
find results -maxdepth 1 -type d | wc -l ; find results -type f | wc -l
find results -name best_restored.pt | wc -l     # 923
find results -maxdepth 2 -name summary.json | wc -l  # 230
```

## 6. 本次未改动项声明

- 本审计**只读**：未修改任何代码 / 配置 / 文档 / 数据文件；未清理任何 results/ 内容（遵守"哈希清单先行，不做破坏性删除"原则）。
- R20（空 __init__ / 死叶模块）未逐项复查，维持上次 backlog 定位。
- exp_800_k8q8.yaml 两个 target_kl 值哪个是本意，需作者裁决——本报告不代选。
