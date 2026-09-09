# 当前架构盘点基线（2026-09-09）

本页是迁移起点，不是永久性能声明。统计包含缓存或生成文件时会注明。

| 区域 | 快照 |
|---|---:|
| `uav_isac/` Python 模块 | 158 |
| `tools/` Python 脚本 | 154 |
| `tests/` Python 文件 | 219 |
| `config/` YAML | 383 |
| `results/` 文件 | 6770（清理并完成刷新后） |
| `results/` 总量 | 15,485,299,077 bytes（清理并完成刷新后） |
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

M0 已建立项目阶段门禁。默认 K4/Q2 与正式 K16/Q16 短帧语义指纹均已连续重放稳定；墙钟
求解、执行器预热和模拟器耗时字段被显式排除并保留给性能审计。因此
`baseline_characterized` 已过门。架构迁移、审计与隔离安装复现均已通过，当前仅开放结果
刷新与受控数据清理；算法优化继续冻结。迁移后以测试全绿而不是固定测试数量为准。
