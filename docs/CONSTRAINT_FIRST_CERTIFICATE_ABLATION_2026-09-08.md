# 约束优先、证书旁路逐步测试（2026-09-08）

## 目标与口径

目标是验证在线 composable-certificate payload 能否退出物理协议，并将保守性直接放入
执行功率优化。全部结果均为 dirty-workspace 开发诊断，不是正式 blind 认证。

- 主配置：`config/exp_strict_distributed_k16q16.yaml`；
- 规模：K16/Q16，30 帧，carrier period=3；
- 小规模严格对拍：seed 7/19/43，逐帧 trace；
- 扩展筛选：与 `joint_pipeline_current_10x30_20260908.json` 相同的 10 个开发 seed；
- QoS 门：steady/weak3/worst = 0.80/0.70/0.60；
- 三帧 worst 期望门：0.75。

## S0：只移除证书传输

配置：`config/exp_strict_distributed_k16q16_certificate_transport_off.yaml`。

3 seed × 30 帧的 certificate ON/OFF 配对通过零容差 golden 比较：

- exact mismatch = 0；
- sensing power、Deflection、P_D、selected structure、local plan、gain views、LP cache 和
  dual price 的 max absolute difference 全部为 0；
- 2412.8 -> 953.6 bit/frame，减少 60.48%；
- 三个 seed 的 worst 指标完全不变。

结论：当前在线证书 payload 对正常物理动作是冗余旁路，可以从协议移除。该结论不覆盖
故障/Byzantine 执行证明。

严格等价结果：`results/constraint_first_s0_physical_equivalence.json`。

## S1：100% 保守执行增益

配置：`config/exp_strict_distributed_k16q16_constraint_first_robust.yaml`，设置
`distributed_replicated_power_robust_gain_mix=1.0`。

3-seed 短测中 worst 均值改善，但扩展到 10×30 后出现明显退化：

| 指标 | 历史同组证书基线 | 100% robust |
|---|---:|---:|
| QoS rate | 0.90 | 0.70 |
| steady mean | 0.9155 | 0.9191 |
| weak3 mean | 0.8669 | 0.8451 |
| worst mean | 0.8441 | 0.8072 |
| worst min | 0.3308 | 0.2414 |
| 3-frame worst min | 0.8502 | 0.8165 |

离线按 trace 重算的保守 Deflection 在全部帧中仅 47% 达到 `P_D=0.6` 对应的
`D_min=11.1795`；跳过前 10 帧后为 60%。100% 保守化不能视为已经闭合的硬 QoS
约束，也不应进入 canonical profile。

## S2：刷新率消融

100% robust 下比较 carrier period 3/2/1：

| period | bit/frame | 跳过前10帧后的 robust D 门通过率 | 结论 |
|---:|---:|---:|---|
| 3 | 953.6 | 60% | 最低通信成本 |
| 2 | 1454.9 | 67% | 收益不足以抵消额外通信 |
| 1 | 2958.9 | 72% | 比原证书基线通信量更高，仍未闭合 |

因此剩余缺口不是简单提高 beacon 频率可以解决；公开状态覆盖和保守包络宽度才是主要
约束。

## S3：25% 保守权重候选

配置：`config/exp_strict_distributed_k16q16_constraint_first_robust25.yaml`。

| 指标 | 历史同组证书基线 | constraint-first robust25 | 差值 |
|---|---:|---:|---:|
| QoS rate | 0.90 | 0.90 | 0 |
| steady mean | 0.915514 | 0.916018 | +0.000504 |
| weak3 mean | 0.866889 | 0.862565 | -0.004324 |
| worst mean | 0.844059 | 0.836473 | -0.007586 |
| worst min | 0.330787 | 0.317883 | -0.012904 |
| 3-frame worst min | 0.850209 | 0.860946 | +0.010737 |
| bit/frame | 2412.8 | 953.6 | -60.48% |

为了隔离 transport 成本，又运行了同代码、同 25% robust 控制器的 certificate ON/OFF
10×30 配对：QoS、steady、weak3、worst 逐 seed 差值全部为 0；OFF 的平均 step
42.22 -> 41.98 ms，step P95 57.21 -> 55.65 ms，闭环关键路径 P95
23.60 -> 22.85 ms。时延收益小于通信收益，且单次墙钟仍受运行噪声影响。

## 判定

1. **通过**：证书传输可从在线协议删除；其正常动作等价性已在 90 帧逐元素零容差对拍。
2. **部分通过**：25% 保守执行权重是当前最佳候选，保持 10-seed QoS rate 和三帧门，
   同时降低 60.48% 通信量。
3. **未通过**：不能把当前实现宣称为“硬鲁棒 QoS 约束”；保守门并非逐帧可行。
4. **拒绝**：100% robust 和 carrier period=1 均不符合综合预期。

## 下一步门禁

- canonical 暂不切换；先补至少 30-seed paired CI，确认 worst mean 的约 0.76 个百分点
  退化是否显著；
- 将 online certificate transport 与 conservative gain construction 在核心配置中正式
  解耦，避免关闭 payload 同时关闭 deadline-safe fallback；
- 对 `public_full_views=false` 和 lower-gain=0 的帧分类，优先收紧包络/覆盖，而不是提高
  全局 carrier 频率；
- 只有当逐帧 reserve feasibility 达到预注册门限后，才把该路线命名为 hard constraint；
- seed 11033 的 nominal 路径曾触发约 `9.8e-7` 相对 primal-dual 差异并 fail closed，
  应作为独立数值缺陷修复，禁止通过放宽证书容差掩盖。
