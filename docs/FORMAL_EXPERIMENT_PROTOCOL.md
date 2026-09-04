# 严格分布式实验冻结协议（v2）

K16/Q16 的专用 100-seed 银行、Wilson 单侧门限、故障隔离执行和墙钟认证
分离规则见 `docs/BLIND100_AND_SIMULATOR_PERFORMANCE.md`。K8/Q8 与 K16/Q16
必须使用各自的场景指纹和种子库，不得混用。

## 运行身份

正式结果必须由 `tools/run_strict_distributed_pilot.py`、
`tools/run_strict_distributed_sweep.py` 或
`tools/run_strict_distributed_bank.py` 生成。每个结果自动嵌入
`strict-distributed-run-manifest/v2`，至少包含：

- Git commit、branch、dirty 状态及状态哈希；
- `config/`、`scripts/`、`tools/`、`uav_isac/` 中所有实验源文件，以及
  `requirements.txt`、`constraints-ci.txt`、`pytest.ini`、`.gitignore` 的内容树
  SHA-256；正式门禁验证器 `tools/assert_formal_gates.py` 也包含在该绑定中；
- 配置源文件哈希、继承展开后的完整配置快照及其 SHA-256；
- 实际种子列表、种子库路径及种子库 SHA-256；
- 算法版本、Python 路径/版本、关键包版本、CPU/平台和线程环境变量。

只要工作区非干净状态，`formal_result_eligible=false`。这不会阻止诊断运行，但该
JSON 不得用于确认性结论。当前工作区包含大量既有未提交/未跟踪文件，因此目前只能
产生诊断结果；需要由项目所有者整理并提交后，才能进行正式 100-seed 冻结运行。
正式执行必须增加 `--formal`；该开关会在运行前检查 canonical manifest 的传递继承、
全部物理/协议身份 pins、双向场景指纹、schema v2、Git 跟踪和内容哈希、完整且顺序一致
的 100-seed 冻结 split，以及 `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`。任一条件
不满足会直接拒绝运行，而不是生成一个看似正式的结果；通过后产物状态才标记为
`FORMAL_COMPLETE`。

正式模式禁止 `--resume`：可编辑的中间 JSON 不能成为确认性证据输入。断点续跑只用于
诊断模式，其完成状态为 `DIAGNOSTIC_ONLY`。正式任务必须从冻结输入重新执行全部 100
个 seed，并在完成时再次校验运行身份和 manifest。

## 正式证据注册

当前 `post_g2` 证据只从 `formal_evidence/registry.json` 加载。该文件采用
`formal-evidence-registry/v1`，顶层字段固定为 `schema_version`、
`evidence_epoch` 和 `entries`；每条记录必须显式给出名称、`results/` 相对的 CSV 与
完成 envelope 路径、两者 SHA-256、`quarantined`、`enforced` 和说明。未知字段、重复
JSON key、错误 schema 或错误类型一律拒绝，不能依赖宽松默认值。

注册表必须纳入 Git。只要 `entries` 非空，门禁会要求工作区中的注册表与当前 `HEAD`
Git blob 完全一致；未提交或暂存但未提交的注册注入不会被接受。注册表目录故意不属于
实验 source-tree hash 范围，因为结果只能在运行完成后登记；验证器本身则属于该范围，
因此既消除了“登记结果改变实验源码哈希”的循环，也不能在不使既有正式证据失效的情况
下修改验证逻辑。空 v1 注册表表示尚无当前正式证据，默认门禁应以退出码 2 失败关闭；
历史 `pre_g2` 列表仍可用 `--evidence-epoch historical` 独立审计，但不能使当前发布变绿。

## 唯一配置与种子库

- K8/Q8 严格入口：`config/exp_strict_distributed_no_truth_pilot.yaml`
- K16/Q16 严格入口：`config/exp_strict_distributed_k16q16.yaml`
- K16/Q16 进程并行候选：
  `config/exp_strict_distributed_k16q16_process4.yaml`（不得替代基线配置）
- 系统基线：`config/system_manifest.yaml`
- K8/Q8 盲测库：`config/stratified_seeds_1130_k8q8_blind.json`
- K16/Q16 盲测库：`config/stratified_seeds_1130_k16q16_blind.json`
- 正式 split：`test`
- 正式样本数：不少于 100；每个 seed 使用独立环境实例。

种子库必须声明 `fingerprint_version=reset-distribution/v2`。该指纹覆盖会改变
`reset()` 场景分布的 K/Q、区域、高度、时域、速度、安全距离、跟踪开关与目标运动
参数。K16/Q16 库已由其完全一致的严格配置迁移到 v2；现有 K8/Q8 库由
`tracking.enabled=false` 的历史配置生成，与当前严格入口的 `true` 不一致，因此是
legacy 库，禁止正式运行。K8/Q8 必须在代码冻结后重新分层生成，不能只改 JSON 标签。

开发 seed、短冒烟和历史结果只能生成假设，不能与冻结盲测混合汇总。不同 K/Q、
运动模型、速度或运动控制器必须形成独立 case，并在完全相同 seed 映射和运行协议下
成对比较。

## 时延口径

结果同时保留三类互不替代的量：

1. `step_time_*`：仿真器整步墙钟时间，包含真值推进、物理生成、奖励和观测；
2. `controller_compute_critical_path_*`：运动、分布式结构、P0 和并行节点中最慢功率
   求解的在线计算路径；
3. `closed_loop_critical_path_*`：控制器计算路径加物理无线序列化关键路径的保守上界。

100 ms 门限只作用于第 3 类。`online_deadline_miss_rate` 和 sweep 的 P95 gate 已改用
该口径；仿真器整步时间保留为工程性能指标，不再被称为部署端时延。

若启用节点私有 LP 的进程并行，必须额外报告 worker 数、一次性预热时间、在线批次
墙钟、超时、缓存/调和降级比例以及降级时的可组合 `P_D` 下界。预热发生在任务时钟
之前但不能从总实验耗时中删除。正式基线仍采用 100 ms 防御性超时与精确串行回退；
15 ms 子截止时间、连续 3 次失败锁存和稀疏调和降级仅属于 K16/Q16 开发候选，必须与
基线成对报告，不能静默替换。seed-bank 外层 worker 必须为 1，禁止嵌套进程池。
同机进程结果只能支持“仿真器计算组织”的结论；真实 UAV 部署时延仍需独立硬件节点
测量最慢完成时间、调度抖动和通信依赖。

## 统计门限

每个正式 case 至少报告逐 seed Steady、Weak-3、Worst、QoS 通过标志、证书完整率、
证书 Worst \(P_D\) 下界、消息 bit、投递/截止时间、跟踪 RMSE、运动覆盖率以及三类
时延。若启用责任后验，还必须报告状态维数以及均值/协方差/AoI的实际载荷；CV的
4维结果不得与CA的6维结果按“相同通信预算”描述。若声明Service阶段，还必须报告
因果Acquisition进入时刻和进入后的证书失效率，不得事后固定删除前N帧。均值不能
掩盖左尾，必须同时报告失败 seed 和二项比例置信下界。只有全部原始
逐 seed 记录、manifest 和汇总代码哈希一致时，汇总表才有效。
