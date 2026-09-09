# 架构迁移计划

迁移以可回滚的小切片推进。任何切片同时改变架构和科学语义都视为失败，必须拆开。

## M0 冻结与基线（当前）

- 暂停算法优化、全量结果刷新和破坏性数据清理；
- 保存当前工作树状态，不整理他人未提交改动；
- 建立结构、入口、配置、数据和依赖基线；
- 选择少量 characterization traces，记录输入、输出、随机状态和容差。

逐值行为基线排除墙钟耗时（例如 `p0_solve_time_s`）；耗时单独作为性能预算审计，不能与
物理状态、动作、奖励、约束或随机结果混入同一确定性指纹。

退出条件：基线可被一条命令验证，核心行为有黄金指纹，待迁移对象有 owner/status。

旧入口同样受机器门禁约束：`scripts/run_mappo.py` 禁止训练优化，
`tools/run_strict_distributed_sweep.py` 禁止全量刷新；单次 strict pilot 仅可用于迁移审计。

## M1 建立 V2 壳层

- 定义 domain 值对象、application 端口、artifact store 和 canonical CLI；
- 把现有入口包装成 legacy adapter，先保持逐值行为一致；
- 统一运行 manifest、配置解析和随机源注入。

退出条件：旧/新入口在选定 seeds 上逐值或在声明容差内一致；失败可回到旧入口。

## M2 迁移稳定主干

按“物理纯函数 → 约束/协调原语 → 环境状态机 → 训练/评估编排”顺序迁移。每次只迁一个
用例闭包；超过 1500 行的聚合器只保留编排，不直接重写全部内部逻辑。

退出条件：正式主干不反向依赖 research；依赖规则、契约测试和全量回归通过。

## M3 收敛研究与工具面

- 活跃机制迁入 `research/` 并登记成熟度；
- 已证伪机制保留文字、配置哈希和结果索引后删除可执行实现；
- 154 个工具脚本收敛为少量子命令；一次性探针进入归档清单而非活动源码。

退出条件：每个可执行入口都有唯一用途、owner、输入契约和产物契约。

入口状态由 `artifacts/legacy/entrypoints.jsonl` 盘点：只有 V2 CLI 是 canonical；架构检查工具
属于 governance；三个旧脚本是受阶段门禁控制的 adapter backend；其余入口统一标记为
legacy retained，不再视为活动入口。

科研系统不以删除历史探针数量作为 M3 完成条件。当前活动面由
`research_programs.yaml` 和 canonical CLI 决定；163 个 `legacy_retained` 入口保持不可从
canonical CLI 到达，用于负结果、消融和历史复现。三个稳定数值执行器以 managed adapter
保留，输出路径由 V2 强制注入到单一 run namespace，并由 completion 哈希闭合。

## M4 数据迁移

先生成清单和校验和，再按 `DATA_LIFECYCLE.md` 分类。迁移使用 copy-verify-switch-delete：
复制/移动到新布局，验证哈希和引用，切换索引，最后才批准删除旧副本。

退出条件：活动数据 100% 有 run manifest；论文/审计引用可解析；清理候选经人工审阅。

## M5 切换与审计

依次执行结构审计、契约审计、可复现审计、科学语义审计、数据溯源审计和性能预算审计。
缺陷修复后必须重跑受影响门。全部通过后才把阶段改为 `result_refresh`。

## M6 全量结果刷新

冻结代码提交、依赖锁、正式 profiles 和 seed banks；先小规模 dry-run，再执行全量更新。
新结果进入独立 V2 命名空间，审计完成前不覆盖旧发布结果。
canonical `refresh-results` 唯一映射到冻结 seed-bank runner；多规模/运动 sweep 保留为
研究诊断入口，不再承担正式全面刷新语义。

## 完成定义

- canonical 入口和数据流唯一；
- 新主干符合依赖规则，legacy 仅剩有期限的适配器；
- 干净环境可按 manifest 重放；
- 审计报告无未处置的 blocker；
- 结果刷新门禁由 false 显式改为 true，并记录批准依据。
