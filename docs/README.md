# 多 UAV 分布式 ISAC 项目文档

> 文档状态：2026-08-01。本文只负责导航，不重复维护实验数字。
> 当前性能、部署决策和下一阶段门槛统一以
> [CURRENT_SYSTEM_STATUS.md](CURRENT_SYSTEM_STATUS.md) 为准。

## 推荐阅读顺序

1. [CURRENT_SYSTEM_STATUS.md](CURRENT_SYSTEM_STATUS.md)：当前系统边界、正式结果、
   版本判定和下一阶段协议。
2. [ARCHITECTURE_V2_RESULTS.md](ARCHITECTURE_V2_RESULTS.md)：Architecture V2 的完整
   演进、机制筛选、失败实验和跨尺度结果。
3. [NOVELTY_AND_SCOPE_AUDIT.md](NOVELTY_AND_SCOPE_AUDIT.md)：创新性、相关方法边界、
   可宣称范围和论文风险。
4. [SYSTEM_MODEL.md](SYSTEM_MODEL.md)：基础环境、物理链路、检测和约束数学模型。
5. [ARCHITECTURE.md](ARCHITECTURE.md)：旧基础实现的数据流与张量映射；阅读时需结合
   Architecture V2 文档，不代表当前部署结构全貌。
6. [TRAINING.md](TRAINING.md)：MAPPO、GAE、约束和历史训练实现说明。
7. [EXPERIMENTS.md](EXPERIMENTS.md)：基础实验协议、配置和指标定义。
8. [KNOWN_ISSUES.md](KNOWN_ISSUES.md)：历史缺陷、修复记录和仍开放的技术债务。

## 当前系统摘要

当前研究对象是只有 UAV 间通信、没有地面通信的多 UAV 协同 ISAC 系统。每架 UAV
在严格 `1 W` 通信与感知联合功率预算下，自主决定运动、Token 通信和逐目标感知资源。
执行策略使用共享参数、集合/注意力编码和分布式结构 Student；最终检测概率仍由环境级
集中式证据融合模块计算，集中式结构控制器只作为教师和参考控制。

当前正式部署版本是 4 UAV / 4 target 冻结模型。跨尺度基数残差在 6/6 上提高了平均
worst，但尚未通过 `QoS feasible >= 0.70` 门槛，因此没有升级为部署版。

## 文档维护规则

- 正式 100 种子结果写入 `CURRENT_SYSTEM_STATUS.md`；机制筛选必须显式标注种子数。
- 完整实验过程统一追加到 `ARCHITECTURE_V2_RESULTS.md`，不再为每轮筛选新建独立文档。
- 数学模型变化更新 `SYSTEM_MODEL.md`；论文创新边界变化更新
  `NOVELTY_AND_SCOPE_AUDIT.md`。
- 已否决方案只保留在 Architecture V2 的历史章节和 Git 记录中，避免旧结果与当前结论
  并列展示。
- 不得把集中式教师、环境级证据融合或诊断 Oracle 表述为分布式部署组件。

## 目录清理说明

2026-08-01 已删除 34 份旧的单轮实验、临时方案和重构前快照文档。这些文件的有效结论
已合并到 Architecture V2 演进记录；原文仍可通过 Git 历史恢复。后续原则上保持本目录
只有上述 9 份主文档。
