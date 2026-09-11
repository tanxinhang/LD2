# Correlation-aware atomic shadow V1 审计交接

日期：2026-09-11
状态：研究审计候选（research audit candidate），不是正式 blind-100 发布版

## 1. 审计对象

本候选版在冻结的 K16/Q16、U2U-only、严格分布式解析基线上增加一条默认关闭的实验路径：

1. 用有限 OTFS delay--Doppler steering atom 的 Gram 矩阵校准同一 owner 下的相关证据；
2. 在 N5/N6 一步结构邻域中，用弱对偶上界筛选候选并以 fixed-structure max--min LP 精确验收；
3. 在通信物理闭包预筛后，为候选生成 reconstruct-then-hash 原子提交证书；
4. 仅在真实控制载波帧执行 shadow，记录候选、代价和 would-commit 结果，不修改 active/pending epoch。

默认配置 `config/default.yaml` 使用 `fusion_correlation_mode: independent` 且关闭 shadow；实验入口是
`config/exp_strict_distributed_k16q16_correlation_calibrated.yaml`。

## 2. 数学与物理不变量

对目标 `q` 的归一化证据协方差 `R_q`，实现采用

```text
D_joint,q = mu_q^H R_q^-1 mu_q
          >= ||mu_q||_2^2 / lambda_max(R_q)
          = sum_e a_eq p_eq / c_q,
c_q = max(1, lambda_max(R_q)).
```

因此相关校准只把固定结构 LP 的系数替换为 `a_eq/c_q`，不改变其线性和凸性。完全重合的 `n`
个模板给出 `c_q=n`，整数 DD-bin 正交模板给出 `c_q=1`。LP 与执行检测器使用同一校准系数。

候选剪枝使用两个可证明安全的上界：

```text
eta_corr*(E') <= U_lambda(g_corr(E')) <= U_lambda(g_raw(E')),
U_lambda(g) = sum_i b_i max_q(lambda_q g_iq).
```

只有精确 LP 的最差目标值严格优于 incumbent 才接受；no-op 始终可用。通信可行性不与 sensing
收益做人为加权，而按 SNR、逐包与端到端时延、RF 能量以及通信--感知共享功率单纯形分别检查。

## 3. 已验证内容

- 单元与集成回归：`1784 passed, 6 skipped`；覆盖 Gram PSD/Hermitian、完全相关/正交极限、
  增量相关因子等价性、物理预筛顺序、Top-M 与不剪枝结果等价、规范序列化、1-ULP 分歧、
  模拟哈希碰撞、generation/功率/Deflection/链路失败闭合。
- Architecture V2 检查通过；默认路径保持关闭且旧回归无行为漂移。
- K6/Q4、30-case 通信感知压力审计中，未经物理预筛的 sensing Top-12 recall 仅为 `80%`；
  加入物理预筛后 Top-8/Top-12 recall 均为 `100%`。这既记录反例，也验证修正方向。
- 加入 dual-bound branch-and-bound 后，Top-12 仍保持 `100%` recall，候选 LP 均值由 `9`
  降为 `6.0333`。
- 独立诊断种子 `20001--20005` 的 K16/Q16、12-frame shadow 共 15 次尝试，全部 exact-accept、
  would-commit，且 active structure 全程不变。平均 Gram、LP、证书重建和协议时延合计约
  `25.8--38.9 ms`，低于本项目为非射频部分预留的 `45 ms` 短诊断预算。

对应诊断 artifact：

- `artifacts/diagnostic/correlation_exchange_topm_physical.json`
- `artifacts/diagnostic/correlation_exchange_topm_physical_prefilter.json`
- `artifacts/diagnostic/correlation_exchange_topm_physical_prefilter_bnb.json`
- `artifacts/runs/pilot-97900bc33756661c2f80/`

artifact 目录按项目规则不进入 Git；上列结果均为 dirty-tree diagnostic evidence，不能用作正式
统计性能声明。

## 4. 明确不作出的声明

- 不声称 Gram 模型已经替代 waveform/IQ 仿真或真实 28 GHz 接收机校准。
- 不声称局部 N5/N6 交换是联合结构--功率全局最优算法。
- 不声称 Top-12 在任意场景或任意分布上零漏检；目前只对已记录的通信可行压力审计成立。
- 不声称 replica agreement 已证明 packet-local 分布式共识。当前 shadow 输入范围明确标记为
  `centralized_physical_diagnostic`，多个副本共享已构造的物理张量。
- 不声称 shadow 已获得 active epoch 写权限；其接口没有状态机句柄。
- 不声称短帧诊断替代 clean-commit blind-100 正式门禁。

## 5. 复现与审计命令

```powershell
pytrch_ven\Scripts\python.exe -m pytest -q
pytrch_ven\Scripts\python.exe tools\check_architecture_v2.py
pytrch_ven\Scripts\python.exe tools\audit_correlation_aware_exchange.py --cases 30 --top-m 4,8,12 --output artifacts\diagnostic\correlation_exchange_topm_physical_prefilter_bnb.json
pytrch_ven\Scripts\python.exe -m uav_isac.interfaces.cli pilot -- --config config\exp_strict_distributed_k16q16_correlation_calibrated.yaml --seeds 20001,20002,20003,20004,20005 --frames 12 --tail-window 8 --quiet
pytrch_ven\Scripts\python.exe tools\check_system_identity.py --manifest config\exp_strict_distributed_k16q16_correlation_calibrated.yaml
```

正式基线的 strict identity 与 formal gate 应继续针对
`config/exp_strict_distributed_k16q16.yaml` 执行。实验 profile 有意改变融合语义，旧 blind-100
必须被 provenance gate 拒绝，不能跨版本复用。

## 6. 下一晋级门槛

下一个版本应让每个副本只从其实际送达、代次一致的 packet-local dependency closure 重建候选
张量，并验证丢包、AoI、量化误差和重建失败均 fail closed。完成该门槛后，才适合讨论受限在线
canary；正式结论仍须另行运行 clean-commit paired blind bank 与 waveform/ROC 校准。
