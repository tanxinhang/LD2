# 约束原生 GNN—时序—探测—优化统一架构（2026-09-08）

## 1. 架构结论

当前系统应从“GNN提出候选、证书批准、LP执行、探测事后评价”的串联结构，改为：

```text
局部公共状态与历史
  -> 时空二部 GNN
  -> 结构可行投影（owner/coverage/cardinality）
  -> 执行节点行拼接
  -> 功率单纯形投影（p>=0, sum_q p_kq<=b_k）
  -> nominal/robust-mix Deflection
  -> P_D、QoS残差、时序变化和稀疏LP残差
  -> 投影原—对偶更新
  -> 下一帧状态
```

在线证书 payload 不再处于动作链。保守 lower envelope 只作为小比例风险项；正式
containment/最优性证书保留在离线审计层。

## 2. 已实现的融合

### 2.1 模型输出

`CertifiedBipartiteGNN` 保留旧类名以兼容接口，新增：

- `power_logits (B,K,Q)`：由原生行单纯形投影变成功率；
- `detection_logits (B,Q)`：下一帧实际 P_D 辅助预测；
- `detection_delta (B,Q)`：跨帧 P_D 变化量；
- 原 owner、Tx、dual warm start、risk residual 继续保留。

### 2.2 原生硬约束

功率不再依赖损失惩罚或在线证书确认：

```text
p_kq = b_k * softmax_q(z_kq) * feasible_mask_kq
```

无可行目标的行输出零；其余行在浮点内向余量下满足非负和预算上限。单测覆盖 mask、
零支撑行、梯度和最大预算超限；当前留出实测超限为 0。

### 2.3 联合目标

当前训练目标为：

```text
L = L_owner + L_tx + 0.1 L_dual
  + L_detection-softmin
  + lambda * g_QoS + rho * g_QoS^2
  + 0.25 L_power-imitation
  + 0.02 L_power-temporal
  + 0.05 L_risk
  + alpha L_PD-forecast + beta L_PD-delta.
```

其中 `lambda` 每 epoch 做 `[0,lambda_max]` 投影对偶上升。QoS 对偶只作用于 teacher
提供构造性可行 witness 的帧；不可行帧继续优化 detection soft-min，防止乘子追逐不可行
约束。

### 2.4 teacher v2

`predictive-teacher-dataset/v2-native-objectives` 新增：

- executed joint power；
- executor nominal gain；
- executor physical lower gain；
- actual Deflection/P_D；
- protocol bits 和 delivery。

它们来自实际执行轨迹，不需要在线传输证书。旧 per-view lower tensor 仅保留为局部信息，
不再冒充全局探测标签。

## 3. 逐步实验

数据为 10 seed × 12 帧，8 seed 训练、2 seed 完全留出；留出仅 22 个转换和 4 个刷新
边界，因此全部结果是架构筛选，不是上线认证。

### 3.1 关键反例

- per-view lower-gain teacher：85.2% viewer-frame 至少一个目标为零，QoS 通过率 0%；
- executor-row lower-gain teacher：worst P_D 均值 0.0054，QoS 通过率 0%；
- 结论：100% lower envelope 不是可用主目标，只能作为风险项。

### 3.2 25% robust mix

`0.75 nominal + 0.25 lower` 与系统级 robust25 消融一致：

| 版本 | edge recall | boundary recall | blended worst P_D | QoS完整率 | power share MAE | 预算超限 |
|---|---:|---:|---:|---:|---:|---:|
| joint fixed penalty | 96.12% | 78.65% | 0.3058 | 22.7% | 0.0535 | 0 |
| 对所有帧做dual | 96.31% | 79.69% | 0.2917 | 18.2% | 0.0532 | 0 |
| 可行帧投影dual | 96.31% | 79.69% | **0.3062** | **22.7%** | 0.0534 | 0 |
| +低权重P_D时序辅助 | 96.12% | 78.65% | **0.3101** | **22.7%** | 0.0533 | 0 |

可行帧 dual 的训练 QoS 残差降到 `3.76e-4`，最终乘子 0.123。120 epoch 出现留出
退化，故当前选择 60 epoch。

### 3.3 收敛判定

训练收敛不再等同于 loss 下降。当前联合门同时要求：

1. 最近窗口 loss 相对跨度低于阈值；
2. QoS 残差跨度低于阈值；
3. 最终 QoS 残差低于约束阈值；
4. 留出探测和结构不退化。

60/120 epoch 均未通过完整联合收敛门。120 epoch 的训练残差更低但留出 worst P_D 和
QoS 下降，判定为过拟合。

## 4. 当前判断

### 已通过

- 在线 composable-certificate payload 可删除；
- 功率预算已经内生为动作投影；
- GNN、执行行、功率、Deflection、P_D、QoS 对偶和时序辅助梯度已贯通；
- 25% robust mix 优于 100% lower-envelope 目标；
- 不可行帧与可行帧已在对偶更新中分流。

### 未通过

- boundary complete frame rate 仍为 0%；
- P_D forecast MAE 约 0.30，辅助头不可用于生产刷新；
- 当前留出只有 4 个边界转换，无法支持收敛或泛化声明；
- GNN 尚未接管环境，只能 shadow；
- 稀疏 LP warm-start 的 KKT residual 仍未进入在线闭环。

## 5. 后续实施顺序

1. 采集至少 30 seed × 30 帧的 certificate-free teacher；按 seed 划分并保证留出边界
   转换不少于 30；
2. 用多任务梯度投影或交替更新，避免结构与探测损失相互伤害；
3. GNN 只提出结构与 primal/dual warm start，结构/预算/QoS/KKT 由原生投影和 residual
   接受；
4. 将 predicted power 的 executor rows 接入 shadow 环境，逐帧与 exact LP 对拍；
5. 依次开放 hold、refresh boundary、LP warm-start，任何一层不过门均回 exact；
6. 最终比较 MLP、静态 GNN、时序 GNN、联合约束 GNN，若时序 GNN无显著优势则删去。

当前推荐 checkpoint 是“60 epoch、25% robust mix、可行帧 projected dual”；低权重
P_D 时序版仅作后续扩数据起点，不具备 production eligibility。

## 6. 10×30 原生 teacher 复验

新增 runner 内存直出路径：`--teacher-output` 会直接生成 v2 NPZ，并从最终 JSON 删除大
trace，避免先写数百 MB JSON 再转换。10×30 数据约 9.9 MB，报告约 116 KB。

同一低权重时序辅助、25% robust mix、可行帧 projected-dual 配置结果：

| 指标 | 10×12 | 10×30 |
|---|---:|---:|
| validation transitions | 22 | 58 |
| boundary transitions | 4 | 10 |
| edge recall | 96.12% | 96.37% |
| boundary edge recall | 78.65% | 78.96% |
| boundary complete | 0% | 0% |
| blended worst P_D | 0.3101 | **0.3864** |
| QoS complete | 22.7% | **39.7%** |
| P_D forecast MAE | 0.3008 | **0.2706** |
| P_D delta MAE | 0.1299 | **0.0901** |
| power share MAE | 0.0533 | **0.0506** |
| budget violation | 0 | 0 |

扩数据显著改善探测、QoS、时序预测和功率，但没有解决刷新边界完整覆盖。检查实际标签
相位后，全部 50 次结构变化均发生在 source phase=0，其余相位变化为0，因此不存在一帧
错位。

边界样本重复权重从4提高到8的消融反而使 boundary recall 78.96%→77.29%、blended
worst P_D 0.3864→0.3488、QoS 39.7%→29.3%，故拒绝。下一步应测试边界专用 residual
decoder、两帧/多帧状态或多任务梯度投影，不能继续靠样本加权或候选膨胀。
