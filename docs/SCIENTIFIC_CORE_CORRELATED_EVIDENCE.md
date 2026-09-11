# Scientific Core：通信受限的相关软证据选择与融合

状态：P0 收缩重构；研究内核，尚无在线执行权

## 唯一科学问题

在多 UAV OTFS-ISAC 中，不同 UAV 的本地软感知证据可能因共同视角、clutter、干扰或有限
delay--Doppler 分辨率而相关。目标是在通信 bit、deadline 和 delivery reliability 约束下，只交换
少量 source-local evidence，并最大化实际送达集合的检测性能，或以最小通信量达到目标检测率。

主线只保留三个科学模块：

```text
OTFS local evidence -> conditional-Deflection selection -> soft fusion
```

Atomic epoch、provenance、replay 和 formal gate 保留为 assurance shell，不再列作算法贡献。

## 统一统计模型

令候选证据的均值差为 `delta=mu_1-mu_0`，`Sigma_0` 为 `H0` 下的对称正定
协方差。对任意线性统计 `w^T z`，H0-Deflection 为：

```text
D_0(w;S) = (w^T delta_S)^2 / (w^T Sigma_0,S w).
```

最优线性融合权重与 H0-Deflection 为

```text
w_S = Sigma_0,S^{-1} delta_S,
D_0(S) = delta_S^T Sigma_0,S^{-1} delta_S.
```

实现使用线性方程求解而不显式计算逆矩阵；非对称、非有限或非正定协方差直接 fail closed。
加入证据 `j` 的 Schur-complement 增量为

```text
Delta D_j(S) =
  (delta_j - Sigma_0,jS Sigma_0,S^{-1} delta_S)^2
  / (Sigma_0,jj - Sigma_0,jS Sigma_0,S^{-1} Sigma_0,Sj).
```

正定性保证分母严格为正，因此 `Delta D_0>=0`，且该式与
`D_0(S union {j})-D_0(S)` 精确相等。这个线性准则本身不要求证据高斯，也不要求
`Sigma_0=Sigma_1`；但非高斯或异方差时，不能再由 Deflection 套用解析 Gaussian ROC，最终
`P_D` 必须由独立 held-out、fixed-P_FA 实验给出。当 `Sigma_0,jS=0` 时，它退化为 local
Deflection `delta_j^2/Sigma_0,jj`；所以低相关场景下与
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
协方差和实际 selected/delivered set 精确重算。小规模 exhaustive reference 与 greedy 使用完全
相同的信息和约束，仅通过枚举可行子集量化 greedy gap，不具有 oracle side information。

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
5. exhaustive subset reference（仅小规模）。

Random/nearest、结构 LP、Top-M 和提交协议只用于 appendix、feasibility repair 或基础设施审计。

## P0 机制门禁

`tools/audit_correlation_budget_mechanism.py` 控制四组双源视角的组内相关系数
`rho={0,0.2,0.5,0.8,0.95}`，按 64-bit evidence 的整数个数扫描预算
`{64,128,192,256,320,384,448,512}` bit，并比较 proposed、correlation-unaware 和 exhaustive
reference。门禁要求：

- `rho=0` 时 proposed 与 unaware 数值一致；
- 高相关且非退化预算下，paired `Delta P_D` 的 95% bootstrap CI 下界大于 0；
- proposed 不超过 exhaustive reference；
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

10,000-sample/split 的诊断结果为：固定校准阈值在 validation 上得到 `P_FA=0.0070`（目标
`0.01`）；目标幅度 scale `0.4/0.7/1.0/1.3` 的 `P_D` 为
`0.2239/0.6541/0.9487/0.9972`。共同杂波产生 `rho_01=0.6746`，而 `rho_03=-0.2136`；H0/H1
covariance 相对误差 `0.0104`，calibration/held-out H0 covariance 相对误差 `0.0237`。15 个非空
subset 的 calibrated `D` 与 held-out `P_D` Spearman correlation 为 `0.9964`。加入 target
amplitude fluctuation 的反例使 covariance mismatch 达 `1.3854`，实现正确切换到 H0 fallback。

这些数值属于 `ideal_cyclic_coherent_otfs_offline_calibration`，仍有以下硬质疑：

- receiver clutter loading 已统一为 1，相关差异来自 DD-template overlap；但 clutter 路径位置仍是
  人为构造的机制场景，不能证明真实几何自然产生相同异质性；
- coherent statistic 假设目标相位已校准，尚未覆盖 unknown-phase GLRT 及其非高斯统计；
- circular fractional delay 隐含充分 cyclic extension，尚未验证 CP 不足、脉冲成形和同步误差；
- `P_FA=0.01` 是受 10,000 样本尾部精度限制的诊断点，不等于正式配置的 `10^-3` 认证；
- calibration 使用受控 H0/H1 标签合理，但 runtime context 尚未证明完全不含 target truth；
- 尚未进入 actual packet delivery，不能据此报告真实 airtime 或链路可靠性收益。

因此 Survival Gate A 只完成了“可执行性证明”，没有完成外部物理有效性证明。下一步应使用冻结、
不含 test truth 的多几何 waveform trace，加入 unknown-phase detector，并以至少能稳定估计
`P_FA=10^-3` 的样本量重复 covariance 异质性、稳定性和 subset ranking 门禁。

### 双基地几何到波形的物理桥

`physical/bistatic_waveform.py` 复用既有双基地距离、Doppler 与 radar-equation path gain，将
`tau/nu/alpha` 映射为连续 OTFS delay/Doppler bin，并用实际 `p_iq` 与天线增益生成接收路径幅度。
该接口接受目标级 RCS，但不接受期望 Deflection；`delta/Sigma_0` 必须在生成接收机总观测后由
calibration 数据估计。

物理边 `(i,j,q)` 仅是生成器内部的可审计分量。接收机可见量必须先按调度集合聚合为
`Y_j=sum_(i,q) Y_ijq`，再形成 `z_jq`；未调度分量非零会被拒绝。当前桥只封闭传播量纲和可观测
边界，尚未封闭多发射机正交资源，所以不能据此宣称多波形能在同一个 `1.024 ms` 块内免费分离。

在该边界上运行的单 Tx / passive-Rx geometry bank 只回答单目标物理可达性。K8/K16 的理想
all-passive floor rate 分别为 `94.8%/99.8%`，优于 best-single 的 `77.0%/94.0%`；但它给每个 case
的唯一 target 使用完整 25.1 mW，且没有 common clutter 或 evidence transport。因此它支持继续
研究 cooperative sensing，却不支持声称 Q16 系统已满足性能预期。

### 未知相位与正式虚警点压力测试

随机化每次 trial 的 target phase 后，coherent mean-shift 不再适用，因此改用 matched-output
energy 作为 GLRT-like local evidence。该证据非高斯且明显异方差：10,000-sample 诊断的 H0/H1
covariance mismatch 为 `0.9690`，实现按规则使用 `Sigma_0`；validation `P_FA=0.0101`，幅度
scale `0.4/0.7/1.0/1.3` 的 `P_D` 为 `0.0602/0.2555/0.6364/0.9152`，subset ranking Spearman
仍为 `0.9714`。

进一步在正式虚警点 `P_FA=10^-3` 使用 100,000 samples/hypothesis/split，并按二项尾部标准误设置
四 sigma 容差。coherent/noncoherent validation PFA 分别为 `0.00094/0.00077`；对应基准幅度的
P_D 为 `0.80237/0.35787`，两条幅度扫描均单调。该结果支持阈值与 surrogate 的数值闭环，但未知
相位造成的性能下降是真实代价，不能用 coherent 结果替代。

最关键的负结果来自 128-bit 2x2 消融：aware 与 unaware selection 都选择 source `(2,3)`，
selection gain 为 `0`，aware fusion 仅增加约 `0.0005 P_D`。因此当前 waveform 场景虽存在异质
相关，却没有激活 correlation-aware selection；Survival Gate 当前状态是 `NOT_ACTIVATED`。
不得通过调 clutter loading 或弱化 baseline 把这个负结果“优化掉”。

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
