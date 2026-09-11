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

## 最小 waveform-to-evidence 闭环（G1--G3）

`physical/waveform_evidence.py` 增加了一个离线、理想循环块模型：QPSK DD pilot 经 unitary ISFFT、
OFDM 调制、双分数 delay/Doppler 路径、OFDM 解调与 SFFT 后形成 DD response；多径、共同杂波、
本地杂波和复 AWGN 显式进入 receiver-local coherent matched statistic。该模型不接在线环境。

`physical/evidence_calibration.py` 在独立 calibration/validation split 上估计
`mu_0,mu_1,Sigma_0,Sigma_1`。只有相对 covariance mismatch 通过预设门槛时才使用 pooled
covariance；否则使用 `Sigma_0` 作为 fixed-P_FA 线性 fallback。若条件数过大，只做最小的
diagonal shrinkage，并继续用 linear solve/Cholesky，不显式求逆。H0 fallback 不是 QDA 最优性
声明，必须另过 held-out ROC 排序门禁。

10,000-sample/split 的诊断结果为：固定校准阈值在 validation 上得到 `P_FA=0.0069`（目标
`0.01`）；目标幅度 scale `0.4/0.7/1.0/1.3` 的 `P_D` 为
`0.2009/0.6202/0.9291/0.9952`。共同杂波产生 `rho_01=0.6641`，而 `rho_03=-0.0131`；H0/H1
covariance 相对误差 `0.0113`，calibration/held-out H0 covariance 相对误差 `0.0251`。15 个非空
subset 的 calibrated `D` 与 held-out `P_D` Spearman correlation 为 `0.9964`。加入 target
amplitude fluctuation 的反例使 covariance mismatch 达 `1.5148`，实现正确切换到 H0 fallback。

这些数值属于 `ideal_cyclic_coherent_otfs_offline_calibration`，仍有以下硬质疑：

- 共同杂波 loading 是人为构造的机制场景，不能证明真实几何自然产生相同异质性；
- coherent statistic 假设目标相位已校准，尚未覆盖 unknown-phase GLRT 及其非高斯统计；
- circular fractional delay 隐含充分 cyclic extension，尚未验证 CP 不足、脉冲成形和同步误差；
- `P_FA=0.01` 是受 10,000 样本尾部精度限制的诊断点，不等于正式配置的 `10^-3` 认证；
- calibration 使用受控 H0/H1 标签合理，但 runtime context 尚未证明完全不含 target truth；
- 尚未进入 actual packet delivery，不能据此报告真实 airtime 或链路可靠性收益。

因此 Survival Gate A 只完成了“可执行性证明”，没有完成外部物理有效性证明。下一步应使用冻结、
不含 test truth 的多几何 waveform trace，加入 unknown-phase detector，并以至少能稳定估计
`P_FA=10^-3` 的样本量重复 covariance 异质性、稳定性和 subset ranking 门禁。

## 不可违反的开发规则

1. Detection threshold 只由 calibration H0 决定；主结果固定 `P_FA` 比较 `P_D`。
2. Runtime 禁止 target truth、future sample 和 validation label。
3. 只融合实际送达、同 generation、去重后的 source-local evidence。
4. 发送失败仍计 attempted bits/airtime/energy。
5. Airtime 公式必须先声明 MAC；串行正交才允许直接求和。
6. Runtime covariance 只能来自冻结 calibration 与可观测 context，不能按 test truth 查表。
7. 同 generation 的 fused evidence 不得递归转发。
8. Exhaustive reference 与 proposed 信息集完全相同，唯一优势只能是组合枚举。
9. 相关矩阵病态或失配时降级，不用裸矩阵逆掩盖问题。
10. 新模块若不能产生独立、可统计认证的 Pareto 改善，应删除或降级。
