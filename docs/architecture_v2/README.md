# 软件架构 V2 控制页

当前状态：`result_refresh`。V2 迁移壳层、架构审计、隔离复现和首轮正式结果刷新已经通过；
算法优化仍冻结。这里的“通过”表示旧系统已被可审计地包入 V2 边界，不表示 legacy 核心、
研究入口和历史数据已经完成最终收敛。

本目录只维护四份活动文档：

- `TARGET_ARCHITECTURE.md`：最终边界与依赖规则；
- `MIGRATION_PLAN.md`：迁移顺序、验收门和回滚原则；
- `DATA_LIFECYCLE.md`：结果数据的保留、归档、发布与清理规则；
- `CURRENT_INVENTORY.md`：迁移基线和已知结构债务。

机器门禁位于 `uav_isac/governance/data/project_phase.yaml`。检查命令：

```powershell
python tools/check_project_phase.py
python tools/check_project_phase.py --operation full_result_refresh
python tools/check_architecture_v2.py
python tools/audit_data_lifecycle_v2.py
pytrch_ven\Scripts\python.exe -m uav_isac.interfaces.cli phase
pytrch_ven\Scripts\python.exe -m uav_isac.interfaces.cli research-programs
pytrch_ven\Scripts\python.exe -m uav_isac.interfaces.cli refactor-status
pytrch_ven\Scripts\python.exe -m uav_isac.interfaces.cli characterize --seed 451 --frames 3
pytrch_ven\Scripts\python.exe -m uav_isac.interfaces.cli verify-baseline
```

统一入口还提供 `train`、`pilot` 和 `refresh-results` 子命令。它们在分派旧后端前检查项目
阶段。当前允许 `pilot`、`refresh-results` 和 characterization；`train` 仍被机器门禁禁止，
直到研究主路径、运行 manifest 和数据生命周期完成收口并通过新的算法优化批准门。

安装项目后可直接使用 `uav-isac`；源码树中等价入口为
`python -m uav_isac.interfaces.cli`。依赖下限和包资源由根目录 `pyproject.toml` 声明，精确
复现版本仍由 `constraints-ci.txt` 锁定。

只有按顺序完成“架构迁移 → 架构审计 → 可复现验证”，才能开放
`full_result_refresh`。禁止通过临时参数绕过门禁。
