# Scientific Core：通信受限的相关软证据选择与融合

状态：P0 收缩重构；研究内核，尚无在线执行权

## 唯一科学问题

在多 UAV OTFS-ISAC 中，不同 UAV 的本地软感知证据可能因共同视角、clutter、干扰或有限
delay--Doppler 分辨率而相关。目标是在通信 bit、deadline 和 delivery reliability 约束下，只交换
少量 source-local evidence，并最大化实际送达集合的检测性能，或以最小通信量达到目标检测率。

主线只保留三个科学模块：

```text
OTFS local evidence -> conditional-information selection -> soft fusion
```

Atomic epoch、provenance、replay 和 formal gate 保留为 assurance shell，不再列作算法贡献。

## 统一统计模型

设候选证据在两个假设下具有共同的对称正定协方差：

```text
z_S | H_k ~ N(mu_k,S, Sigma_S),
delta_S = mu_1,S - mu_0,S.
```

最优线性融合权重与 Deflection 为

```text
w_S = Sigma_S^{-1} delta_S,
D(S) = delta_S^T Sigma_S^{-1} delta_S.
```

实现使用线性方程求解而不显式计算逆矩阵；非对称、非有限或非正定协方差直接 fail closed。
加入证据 `j` 的 Schur-complement 增量为

```text
Delta D_j(S) =
  (delta_j - Sigma_jS Sigma_S^{-1} delta_S)^2
  / (Sigma_jj - Sigma_jS Sigma_S^{-1} Sigma_Sj).
```

正定性保证分母严格为正，因此 `Delta D>=0`，且该式与 `D(S union {j})-D(S)` 精确相等。
当 `Sigma_jS=0` 时，它退化为 local Deflection `delta_j^2/Sigma_jj`；所以低相关场景下与
quality-only baseline 接近是理论预期，而不是需要用附加协议掩盖的问题。

## 通信约束与算法边界

当前 greedy 以

```text
p_success,j * Delta D_j(S)
---------------------------------------
bits_j + lambda_latency * latency_j
```

排序，同时将 bit budget 和 deadline 保持为独立硬约束。该比率是可解释的 task-oriented
启发式，不宣称全局最优，也不宣称 `D(S)` 一般具有子模性。最终报告的融合 Deflection 使用真实
协方差和实际 selected/delivered set 精确重算。小规模 exhaustive oracle 用于量化 greedy gap。

空口主路径只允许 source-local evidence：当前 `EvidencePacket` 由 sensing receiver 直接广播，
身份为 `(observation_frame, source_rx, target_id)`，路由 API 不接收已融合 packet 作为新 sensing
source，也没有 relay/origin 分裂字段。同一 generation 的融合结果只能在 owner 本地使用，不能
再次包装成源证据。因此主线不需要 dependency DAG 或 lineage bitmap；未来 multi-hop 只能作为
单独 extension 重新建模。

## 主文 baseline

1. Single-UAV；
2. All-neighbor fusion（高开销参考）；
3. local-Deflection/quality Top-K；
4. correlation-unaware greedy（`Sigma` 对角化后选择，真实 `Sigma` 下融合）；
5. exact subset oracle（仅小规模）。

Random/nearest、结构 LP、Top-M 和提交协议只用于 appendix、feasibility repair 或基础设施审计。

## P0 机制门禁

`tools/audit_correlation_budget_mechanism.py` 控制四组双源视角的组内相关系数
`rho={0,0.2,0.5,0.8,0.95}`，扫描通信预算比例 `{0.2,0.4,0.6,0.8,1.0}`，并比较 proposed、
correlation-unaware 和 exact oracle。门禁要求：

- `rho=0` 时 proposed 与 unaware 数值一致；
- 高相关且非退化预算下，paired `Delta P_D` 的 95% bootstrap CI 下界大于 0；
- proposed 不超过 exact oracle；
- 高相关时达到指定 `P_D` 所需的离散扫描 bits 更少。

这只是合成机制激活实验。它确认算法逻辑是否值得继续，不校准真实 OTFS `delta,Sigma`，也不能
替代 waveform/IQ、common-clutter、量化、丢包和 clean-commit blind-bank 证据。

## 下一门槛

从 OTFS 波形或单独校准 trace 估计 `delta` 与 `Sigma`，在不复用评估数据的条件下冻结 calibration
artifact；随后让 owner 仅以实际送达的 `EvidencePacket` 构造集合并运行相同融合器。若真实高相关
工作点仍不能产生可重复的 Pareto 改善，则停止把 correlation-aware selection 作为主创新。
