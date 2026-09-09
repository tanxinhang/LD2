# 软件架构 V2

## 1. 设计目标

V2 只优化四件事：边界清楚、入口唯一、实验可复现、扩展不修改稳定核心。它不追求把每个
细节都抽象成框架，也不在迁移过程中改变物理语义或算法结论。

## 2. 分层与依赖方向

```text
interfaces (CLI / batch entrypoints)
    -> application (train / evaluate / audit use cases)
        -> domain (physics, constraints, coordination contracts)
        <- adapters (YAML, solver, checkpoint, artifact store)

research -> application/domain public contracts only
```

目标目录：

```text
uav_isac/
  domain/          # 无框架状态的物理、约束、值对象和协调原语
  application/     # 用例编排；一次运行只有一个明确入口
  adapters/        # YAML、SciPy/Torch、文件系统、旧实现适配器
  interfaces/      # CLI 参数解析和退出码，不含业务逻辑
  research/        # 未认证机制；不得被正式运行反向依赖
  governance/      # 阶段、身份、provenance 和发布门禁
```

现有 `physical/coordination/environment/agents/evaluation` 是迁移源，不立即整体改名。采用
strangler 迁移：先定义契约，再逐用例切换；旧模块只有适配器能进入新主干。迁移期间禁止
为了目录整齐进行大爆炸式重写。

## 3. 六条硬规则

1. 正式训练、评估、审计各有一个 canonical CLI；脚本只调用 application use case。
2. Domain 不读取 YAML、不写文件、不依赖 environment/agents/evaluation。
3. Application 只依赖端口和 domain；文件、求解器、Torch 由 adapters 注入。
4. Research 代码默认不可达，启用时必须有显式 profile、状态标签和独立产物命名空间。
5. 配置解析后生成规范快照和哈希；禁止隐式默认值、重复键和运行时静默改写。
6. 每个 run 生成不可变 manifest；发布指标只能引用通过审计的 run_id。

## 4. 稳定契约

稳定核心只暴露小型数据结构和协议：`ScenarioSpec`、`FrameState`、`ActionPlan`、
`ConstraintReport`、`RunSpec`、`RunManifest`。NumPy/Torch 数组可作为载荷，但 shape、单位、
坐标系、随机源和缺失值语义必须由契约声明。避免建立“万能上下文对象”。

## 5. 扩展方式

- 新算法：实现 domain/application 端口，在 research 注册；不得在环境巨类增加开关链。
- 新场景：增加组合式 `ScenarioSpec`，不复制完整 YAML。
- 新指标：实现 evaluation observer，只读事件流，不回调或修改仿真状态。
- 新存储：实现 artifact-store adapter，不改变训练/评估用例。

## 6. 明确不做

本轮不优化模型表现，不刷新论文结果，不统一所有历史研究 API，不保留已证伪机制的可执行
副本，也不把 383 份历史 YAML 全部升级为活动配置。历史复现通过冻结快照或归档实现。

