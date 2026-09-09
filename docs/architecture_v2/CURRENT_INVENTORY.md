# 当前架构盘点基线（2026-09-09）

本页是迁移起点，不是永久性能声明。统计包含缓存或生成文件时会注明。

| 区域 | 快照 |
|---|---:|
| `uav_isac/` Python 模块 | 190 |
| `tools/` Python 脚本 | 162 |
| `tests/` Python 文件 | 232 |
| `config/` YAML | 383 |
| `results/` 文件 | 6763（本轮只读复核） |
| `results/` 总量 | 15,484,782,200 bytes（本轮只读复核） |
| `env_core.py` | 13316 行 |
| `trainer.py` | 9192 行 |

主要结构债务：

1. 环境和训练聚合器承担配置、状态、协议、求解、遥测等多种职责；
2. 正式、研究、审计实现之间边界松散，协调层至少 15 处反向依赖 environment/evaluation；
3. 配置和工具数量过多，活动入口与历史复现入口没有统一生命周期；
4. 结果目录巨大，raw/derived/published/scratch 混放，不能安全地按目录名直接清理；
5. 当前工作树有大量未提交改动，迁移必须避免无依据覆盖或批量移动。

协调层原有 15 个反向依赖点已纳入 `uav_isac/governance/data/architecture_rules.yaml` 棘轮：14 个环境
类型依赖已改为 domain port，纯安全门已从 evaluation 迁入 domain。当前例外数为 0；检查器
会拒绝任何新的反向依赖以及已经失效但未删除的例外。

`ActionSpace` 与 `ObservationSlices` 已迁入 domain，旧 environment 路径仅作兼容导出；agents
不再为这两个公共契约依赖 environment。其余物理、环境和 trainer 反向依赖已进入同一棘轮，
后续按用例切片消除。

两个巨型聚合器 `env_core.py` 与 `trainer.py` 已隔离到 `uav_isac/legacy/`；旧模块路径只作
兼容入口。V2 application 通过 adapter 使用 legacy runtime，新增功能不得直接扩展这两个
聚合器。

383 份 YAML 中，V2 活动 registry 只登记 `default_dev` 与 `strict_k16q16`。其他 YAML 暂时
保留为 legacy reproduction 输入，不能因文件存在就成为活动实验入口。

结果数据已完成 metadata 和 SHA-256 盘点。经用户明确授权，只删除 43 个未引用且有
SHA-256 完全一致保留副本的冗余 `.log`，释放 737,779,883 bytes；没有删除 checkpoint、
trace、frozen/blind 证据或未分类数据。逐文件恢复路径记录在不可覆盖的清理日志中。清理后
重新盘点为 6761 文件、15,483,082,473 bytes；仍存在的 2280 个重复项继续保持
`deletion_authorized=false`。

架构迁移审计已通过：依赖、语义基线、profile、打包、数据目录和入口目录均通过，发布终审
为 `1697 passed`。此前构建 wheel `uav_isac-2.0.0a0`（SHA-256
`085e81605291a320fba3fe65afe547b5f36f0994638cffa4c7911a8fd0ac34dd`），在隔离安装目录中重放
两条黄金基线均一致。项目阶段因此进入 `result_refresh`；算法优化仍关闭。

K16/Q16 正式刷新已经从冻结提交
`080016ac0b2964dc88f2a88971efd08aad20d3d4` 完成：100/100 seed QoS 通过，单侧 95% Wilson
下界 0.973657，投递率 1.0，截止违约率 0，逐 seed 闭环 P95 最大值 38.884 ms。完成 envelope
SHA-256 为 `7f566dd52b9b1349470c82945bd4252f6bdda1faa937586ae62355847a5a35d4`，paired CSV
SHA-256 为 `f5581a54be678431c359a717c8e1e2b56087fc67075c4867a1a73119b6e6e79b`；二者已登记到
`formal_evidence/registry.json`，并由当前 post-G2 正式门禁重新计算为 PASS。V2 发布索引与不可变
副本位于 `artifacts/published/` 和 `artifacts/runs/formal-k16q16-blind100-080016a-7f566dd5/`。
刷新后的 results 再盘点为 6770 文件、15,485,299,077 bytes，并全部完成 SHA-256 catalog。

2026-09-09 发布后深度审计再次验证架构边界、依赖、黄金语义基线、严格系统身份、正式
post-G2 证据和 Git 对象完整性。随后精确清理 49 个目标、3384 个文件、102,241,676 bytes：
包括可重建的 Python/pytest/build/egg-info/codegraph 缓存、失败的隔离安装副本、未引用旧探针，
以及已由正式 `FORMAL_COMPLETE` envelope 取代的 7 个中间结果。被文档引用的
`.arts/algorithm_audit`、活动虚拟环境、成功的隔离复现、正式结果、checkpoint、trace、frozen
证据及未分类数据均保留。复现分层后 results 为 6761 文件、15,483,932,320 bytes，已重新生成
全文件 SHA-256 catalog；2 个已验证且保留规范副本的零字节日志被删除。剩余重复项为
2278 个、8,756,091,454 bytes，其中 2229 个位于含标准复现产物的目录，439 个被受版本控制
文本引用；它们已单列为复现或待复核资产，不会当作垃圾删除。

清理后的结果树治理审计仍识别出 1044 个一级结果目录，其中 38 个不含
`summary.json`、`paired_eval.csv` 或 `run_manifest.json`，331 个为下划线前缀目录，现存
`summary.json` 有 88 种顶层 schema。这些目录可能包含历史复现证据，当前统一视为待分类的
legacy debt，不据名称或结构缺失自动删除。审计工具的 `--json-output` 路径另有 tuple-key
序列化缺陷；本次通过只读兼容导出生成审计报告，暂不修改正式发布绑定的源代码。

根目录 7 个一次性审计/探针脚本、旧 `_orphan_v3.txt` 和失败测试日志已删除；相应结论由
版本化文档与 V2 catalog 接管。基础 `system_manifest.yaml` 的历史 K8/Q8 bank 指针已替换为
场景匹配的 `stratified_seeds_400_k4q2_v2.json`，K4/Q2 基础身份现在可通过严格 fingerprint、
source-config 和 K/Q/region/dynamics 检查；K16/Q16 profile 继续使用独立的正式 bank。

M0 已建立项目阶段门禁。默认 K4/Q2 与正式 K16/Q16 短帧语义指纹均已连续重放稳定；墙钟
求解、执行器预热和模拟器耗时字段被显式排除并保留给性能审计。因此
`baseline_characterized` 已过门。架构迁移、审计与隔离安装复现均已通过，当前仅开放结果
刷新与受控数据清理；算法优化继续冻结。迁移后以测试全绿而不是固定测试数量为准。

## 尚未完成的研究型收口

截至本次复核，V2 foundation 可以视为完成，但原计划中的 M3 及运行契约仍有实质性债务：
175 个可执行入口中有 163 个仍标记为 `legacy_retained`；正式运行还同时使用 V2 artifact
manifest 与 strict-distributed manifest 两套契约；训练和正式刷新仍经 adapter 分派旧脚本。
因此在这些项目完成前，不把系统标记为“研究主干重构完成”，也不开放算法优化。

M4 的保留边界已经明确：`results/` 整体是只读的 pre-V2 历史研究库。此前 5044 个
`unclassified` 文件现标记为 `legacy_unmanifested`，表示可以保留和人工复现，但没有 V2
manifest，不能直接支持新论文结论。新运行只能写入 `artifacts/runs/<run_id>/`，并以不可变
manifest 和 completion 哈希闭合。该分类不授权删除任何历史资产。
