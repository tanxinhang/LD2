# 多 UAV 分布式 ISAC 项目文档

> 文档状态：2026-08-16（清理修正：导航补入新文档与 D0.9x 系列；修正"9 份主文档"声明）。
> 本文只负责导航，不重复维护实验数字。
> 当前性能、部署决策和下一阶段门槛统一以
> [CURRENT_SYSTEM_STATUS.md](CURRENT_SYSTEM_STATUS.md) 与
> [SYSTEM_OVERVIEW_AND_ROADMAP.md](SYSTEM_OVERVIEW_AND_ROADMAP.md) 为准。

## 推荐阅读顺序

1. [SYSTEM_OVERVIEW_AND_ROADMAP.md](SYSTEM_OVERVIEW_AND_ROADMAP.md)：**总纲**——系统全景、
   性能缺口、问题清单、理论框架与分阶段路线（最新权威入口）。
2. [CURRENT_SYSTEM_STATUS.md](CURRENT_SYSTEM_STATUS.md)：当前系统边界、正式结果、
   版本判定和下一阶段协议（含 8/8 解析栈与认证化控制链）。
3. [CURRENT_SYSTEM_MODEL.md](CURRENT_SYSTEM_MODEL.md)：当前部署架构的数学模型与代码
   映射总纲（Architecture V2 + 认证化控制 + T3 隐蔽性）。
4. [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md)：D1.1 系列理论驱动优化日志（对偶剪枝/
   最优带宽/隐蔽性 live 化，含负结果）。
5. [ARCHITECTURE_V2_RESULTS.md](ARCHITECTURE_V2_RESULTS.md)：Architecture V2 的完整
   演进、机制筛选、失败实验和跨尺度结果。
6. [NOVELTY_AND_SCOPE_AUDIT.md](NOVELTY_AND_SCOPE_AUDIT.md)：创新性、相关方法边界、
   可宣称范围和论文风险。
7. [KNOWN_ISSUES.md](KNOWN_ISSUES.md)：历史缺陷、修复记录和仍开放的技术债务（含
   2026-08-16 测试污染隔离与 8 月 Gate 问题登记）。
8. [SYSTEM_MODEL.md](SYSTEM_MODEL.md)（历史）：基础环境、物理链路、检测和约束数学模型
   （被 CURRENT_SYSTEM_MODEL.md 取代，保留为基础公式参考）。
9. [TRAINING.md](TRAINING.md)（部分过时）：MAPPO、GAE、约束和历史训练实现说明。
10. [EXPERIMENTS.md](EXPERIMENTS.md)（部分过时）：基础实验协议、配置和指标定义。
11. [ARCHITECTURE.md](ARCHITECTURE.md)（历史）：旧基础实现的数据流与张量映射；不代表
    当前部署结构全貌。

### D0.9x Gate 系列（8/8 解析功率/几何链）

D0.87–D0.95 逐 Gate 证据：`D087_POWER_DEPLOYMENT.md`、`D088_WARMSTART_STALENESS.md`、
`D089_ANALYTICAL_INNER_POWER.md`、`D089B_STRUCTURE_RANKING.md`、`D089C_TSTAR_REWARD.md`、
`D091_STEADY_CEILING.md`、`D092_BARGAINING_OBJECTIVE.md`、`D093_CAPABILITY_GAUGE.md`、
`D093_L2_STRUCTURE.md`、`D093_POWER_SIDE_E2E.md`、`D094_L3D_DISTRIBUTED_GEOMETRY.md`、
`D095_JOINT_L2_L3_ALTERNATING.md`；后续：`D1A_HORIZON_JOINT_ORACLE.md`、
`D1_1A_LEXICOGRAPHIC_L1.md`；隐蔽性主线：`T2_CENTRAL_ORACLE.md`、
`T3_DETECTION_CAPABILITY.md`。（编号映射见总纲 §1.4。）

## 当前系统摘要

当前研究对象是只有 UAV 间通信、没有地面通信的多 UAV 协同 ISAC 系统。每架 UAV
在严格 `1 W` 通信与感知联合功率预算下，自主决定运动、Token 通信和逐目标感知资源。
执行策略使用共享参数、集合/注意力编码和分布式结构 Student；最终检测概率仍由环境级
集中式证据融合模块计算，集中式结构控制器只作为教师和参考控制。

当前正式部署版本是 4 UAV / 4 target 冻结模型（100 seed）。8/8 部署候选为
lexicographic L1 + 多候选 L3 解析栈（20 seed worst 0.975，QoS 1.0）；T3 隐蔽性约束
（P_D^I ≤ ε）已可接入 live 功率路径。跨尺度基数残差在 6/6 上的决策数据受测试集污染
影响待重跑，未升级为部署版。

## 文档维护规则

- 正式 100 种子结果写入 `CURRENT_SYSTEM_STATUS.md`；机制筛选必须显式标注种子数。
- 完整实验过程统一追加到 `ARCHITECTURE_V2_RESULTS.md`，不再为每轮筛选新建独立文档。
- 数学模型变化更新 `CURRENT_SYSTEM_MODEL.md`（基础公式参考 `SYSTEM_MODEL.md`）；论文
  创新边界变化更新 `NOVELTY_AND_SCOPE_AUDIT.md`。
- 理论驱动优化（含负结果）记录到 `OPTIMIZATION_LOG.md`。
- 已否决方案只保留在 Architecture V2 的历史章节和 Git 记录中，避免旧结果与当前结论
  并列展示。
- 不得把集中式教师、环境级证据融合或诊断 Oracle 表述为分布式部署组件。

## 目录清理说明

2026-08-01 已删除 34 份旧的单轮实验文档；2026-08-16 清理将过时的 DCB-MAPPO 论文稿
归档至 `paper/archive/`，results/ 过时产物归档至 `results/_archive/`。历史均保留在
Git 记录与归档目录中。
