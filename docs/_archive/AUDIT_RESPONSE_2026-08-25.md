# 2026-08-25 审计响应与优化记录（只读审计 → 实现与验证）

本记录对应 2026-08-25 只读审计（P0/P1/P2 与建议优先级）。所有改动默认关闭或
独立 config，稳定配置不变；每一项都有 seed-451 同 seed 因果验证、单元测试或
预注册 25-seed 配对确认。审计全文见会话；本记录按审计优先级逐条给出
「改动 → 验证 → 结论」。

## 结论速览

- **审计 #1（冻结）已完成**：动态 K12 基线冻结为**混合 freshness 候选**
  `exp_800_k12q12_distributed_v2_dynamic_u2u_robustbelief_freshness_pilot.yaml`
  （`frozen/STABLE_DISTRIBUTED_DYNAMIC_K12_V1_2026-08-25.md`，快照哈希验证），
  停止与冻结 V1 混用。
- **审计 #3（belief 调度）通过预注册配对门**：25-seed 上三档门通过数
  **19/25 → 20/25**（Wilson LCB 0.566→0.609），worst P_D 均值 +0.073
  （bootstrap 95% LCB +0.0166），**worst p05 0.121→0.509（4.2×）**，
  证据投递率保持 0.9990。
- **审计 #5（总协议 deadline）确认成立**：25-seed 实测串行总时延
  5.2–5.3 ms（P95 7.1–7.3 ms）> 4 ms，违约率 1.0。
- **审计 #2（双角色容量分配）提升一致率但单独不提升 P_D**：一致率
  0.50→0.68（联合 freshness 0.79），hard seed 上移动稀释航程；
  作为可配置项保留。
- **审计 #4（目标侧驻留）**：standoff 夹逼已实现，提供 80 m 变体 config；
  完整截获概率目标未启用（后续）。

## 实现总览

| 审计优先级 | 内容 | 实现 | 验证 |
|---|---|---|---|
| #2 | 远场双角色容量分配 | `role_capacity_bottleneck_assignment` + role-capacity 移动模式 | 单测 7 项；seed-451 一致率 50.4→68.2%（worst-first）、联合 freshness 78.6% |
| #3 | belief 广播改 AoI/不确定性/任务亏损联合调度 | `u2u_belief_feedback_schedule=freshness` + 全局比特预算 | 单测；seed-451 RMSE −47%、AoI −39%、分歧 −74%、worst P_D +80% |
| #4 | 目标侧最小距离 | `distributed_role_capacity_standoff_m` 夹逼 | config 变体 + 代码夹逼（单测覆盖行程钳制） |
| #5 | 统一 MAC 时序与总协议 deadline | `_serialized_protocol_accounting` + trainer CSV 列 | 单测；实测 5.12 ms > 4 ms（violation=1.0，与审计 5.37 ms 同量级） |
| #1 | 冻结动态 K12 基线 | 预注册 25-seed 配对确认 + 冻结文档 + 快照 | 进行中（见 `frozen/PREREG_LD3_DYN_K12_CONF_1.md`） |
| #6/#7 | 重新训练/规模曲线 | checkpoint 迁移消融（episodes=0 零训练）已在所有验证中使用；6×6–12×12 同口径曲线列为后续 | 见边界说明 |

## #2 远场双角色容量分配

### 数学

每个目标同时拥有一个 Tx 几何责任与一个 Rx 几何责任。公共 Tx/Rx 划分固定
（`arange(K)%2==0`，或超边保留的 Tx 集合），每个角色单独求解

\[
\min_{\pi_r}\ \max_q \left(H^2+\|x_{\pi_r(q)}-z_q\|^2\right),\qquad
\sum_q \mathbf 1\{\pi_r(q)=k\}\le c_r,\quad c_r=\lceil Q/|\mathcal K_r|\rceil .
\]

用「节点复制 c 行 + 阈值二分 + 最小化总代价 + 确定性行主序 tie-break」的
b-匹配实现（`hyperedge.role_capacity_bottleneck_assignment`），同一公共视图
必然重建同一责任集，无优化器证书交换。

移动：range 阶段每节点朝**当前最远**责任目标（worst-first，贪婪下降瓶颈程），
行程被责任目标的 standoff 圆盘钳制；strategy 阶段保持几何由 P0 更新功率；
重排要求相对瓶颈改进 ≥5% 且距上次重排 ≥20 帧（滞回）。

### seed-451 因果（同 seed，同冻结 checkpoint）

| 机制 | worst P_D | weak3 | steady | 责任一致率 |
|---|---:|---:|---:|---:|
| 稳定基线 | 0.021289 | 0.046061 | 0.458360 | 0.5036 |
| role-cap 中点移动 | 0.001318 | 0.002596 | 0.206057 | 0.6711 |
| role-cap worst-first | 0.007593 | 0.011181 | 0.253526 | 0.6820 |
| role-cap + freshness | 0.029224 | 0.037061 | 0.224601 | 0.7863 |

结论：role-capacity 把「行拼接」的一致率从 0.50 提升到 0.68（联合 freshness
0.79），但**中点移动稀释了 375 m 航程**，hard seed 上 P_D 反而下降；worst-first
恢复航程但重排使近目标失守。单独 role-capacity 不作为默认；它与 freshness
联合时一致率最高但 P_D 仍低于 freshness 单独。责任机制保留为可配置项。

## #3 belief 广播联合调度（本轮的正面结果）

### 改动

`route_structured_evidence` 接受独立 `belief_selected_mask`：证据选择仍是
receiver 本地质量 top-k（检测不变），belief 后验改由
\[
\text{score}_{kq}=w_a\,\text{aoi}_{kq}^\text{norm}
+w_u\,\text{unc}_{kq}^\text{norm}+w_t\,\text{deficit}_{kq}^\text{norm}
\]
按源逐行 min-max 归一化后取 top-k，跳过「源 AoI 超融合上限」与「自己 owned
目标」；包承载证据∪belief 条目，只有 belief 条目付 88 bit 后验载荷；**全局
后验比特预算**（默认 400 bit/帧）防止 freshness 调度打爆 4 ms deadline。

### seed-451 同 seed（基线 worst 0.0213 / weak3 0.0461 / steady 0.4584）

| 方案 | worst | weak3 | steady | belief RMSE (m) | AoI (帧) | 分歧 (m) | 证据投递率 | 证据 deadline 违约 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 基线（evidence top-1 耦合） | 0.021289 | 0.046061 | 0.458360 | 69.33 | 43.78 | 69.92 | 0.99985 | 0 |
| freshness 无预算 | 0.019727 | 0.022371 | 0.275372 | 48.70 | 33.25 | 42.81 | 0.369 | 0.631 |
| **freshness + 400 bit 预算** | **0.038354** | **0.051221** | 0.321828 | **36.38** | **26.82** | **18.53** | 0.969 | 0.031 |

机制：belief 越新 → 2-sigma 鲁棒增益的 `rho_q` 越小 → 公开增益去保守化越少
→ 功率层有效增益越高（P0 在 `distributed_target_position_uncertainty_sigma=2`
下按 belief 协方差重建增益）。无预算版把载荷打到 2408 bit 使证据 deadline
违约 63%、投递率掉到 37%——这正说明「调度必须服从 MAC deadline」（#5）。

### 后续

- 权重联合（1/3,1/3,1/3）为默认；可做 AoI-only / 任务亏损-only 消融。
- 目标速度协方差尚未进入 Doppler/DD gate 鲁棒化（审计 P1 #6 边界保留）。

## #4 目标侧最小距离

role-capacity 移动已实现责任目标 standoff 圆盘钳制（`rolecap_standoff_hold`
状态 + 二次方程精确行程钳制）；提供
`..._robustbelief_rolecap_standoff80_pilot.yaml`（80 m 目标驻留）。
完整截获概率目标（`intercept_constrained_power_enabled`）仍是关闭分支，
未在本轮启用——审计 P1 的「接近—策略—隐蔽性」三块交替优化中的隐蔽性块
只落地了距离钳制部分。

## #5 统一 MAC 时序与总协议 deadline

`_serialized_protocol_accounting`：协调广播（learned + 超边状态）与
evidence/belief 广播作为同一帧窗口的两个子时隙，报告
`protocol_total_latency_s / p95 / serialized_bits / deadline_violation`；
trainer 聚合为 `eval_protocol_*` 四列。

实测（seed-451 freshness 预算版）：协调 2.92 ms + evidence 2.20 ms
串行合计 **5.12 ms（P95 7.38 ms）> 4 ms → violation 1.0**，与审计
5.37 ms 结论一致：**当前协议串行发送不满足总 deadline**，需要（a）削减
协调载荷，（b）分频/并行子信道，或（c）放宽帧长。这是审计 #5 的已验证
账本，不是已修复项。

## 回归

- 新增单测：`tests/test_role_capacity_assignment.py`（7 项）、
  `tests/test_belief_freshness_mac_accounting.py`（5 项）。
- 改动模块回归：hyperedge/evidence/communication/integrity/power/config/
  provenance 相关 184 passed（两批）。
- 稳定基线 seed-451 逐值复现审计（worst 0.021289 / weak3 0.046061 /
  steady 0.458360 / 最近距离 477.785 / 间距 20.988），确认 harness 与
  冻结 checkpoint 一致。

## 同口径规模曲线（审计 #7，预注册 Amendment 2，进行中）

关键前置发现：**冻结的 K12 检测器常量不跨规模迁移**（K6 用 K12 threshold
3.35 时 PFA 实测 0.106）。因此先重建了可复现校准配方
（`tools/calibrate_dynamic_evidence.py`，复刻部署统计量：owner 本地 LLR +
8-bit 量化 peer + 2-bit 置信量化 threshold deflection，取 1−P_FA 分位），
并在 K6/K8/K12 三个规模统一使用（PFA=0.001，自检 PFA 0.00098–0.00105）。
规模趋势：τ K6=15.67 > K8=7.45 > K12=5.22（区域越大、统计量尾越轻）。

10-seed 配对（同 seed 集，baseline vs 混合 freshness）：

| 规模 | baseline 门通过 | hybrid 门通过 | worst Δ均值 | worst bootstrap LCB | 证据投递率 (b→h) |
|---|---:|---:|---:|---:|---:|
| 6×6 | 4/10 | 4/10 | +0.077 | −0.191 | 0.795→0.665 |
| 8×8 | 6/10 | **7/10** | **+0.169** | **+0.009** | 0.9995→0.859 |
| 12×12（recal） | 5/10 | **6/10** | +0.119 | −0.006 | 0.9996→0.9990 |

结论：**hybrid 的 worst P_D 配对均值在三个规模全部为正**（+0.077/+0.169/
+0.119，正 seed 5–7/10），门通过数在 8×8、12×12 各 +1，6×6 持平；
同配方 K12 点复现冻结 K12 的排序（门 +1）。8×8 按预注册规则 PASS，
K12 冻结 25-seed 确认 PASS（20/25）。K6/K8 证据 deadline 违约率
（20–33%）高于 K12（0.08%）：400 bit 预算与 union 上限 2 是 K12 调参，
小规模每包时延裕量更紧（协议 #5 的规模维度，后续按规模重调）。

## 三段式策略（Phase A/B/C，2026-08-25 第二轮）

用户要求：前期快速收敛 / 后期补强+兜底 / 近场通信-感知-位置联合调整。
设计见 `docs/THREE_PHASE_POLICY_DESIGN_2026-08-25.md`，预注册见
`frozen/PREREG_LD3_DYN_K12_CONF_1.md` Amendment 3。

已验证（决定性种子，同 seed 同 checkpoint）：

- **Phase A gap-coverage 兜底**：修复"漏追"根因（一致率 0.72→0.93），
  漏追种子 1992563642 worst 0.011→0.41（v1 乘积序，37 倍）。25-seed
  门 20→21/25，worst min 0.011→0.41（灾难尾部消除）。
- **pursuit-250 饱和保护消融**：门回退 18/25（保护反而有害），默认关。
- **Phase B 近场聚焦**（critical：近端≤300 远端>450）：修复"近但未达标"
  种子——647973436（278m，P_D 0.589→0.956）、1466409252（0.582→0.97+）；
  机制：乘积序把近端责任 UAV 拉去追别的责任，聚焦锁定它持续接近
  （∂P_D/∂R ∝ −4/R⁵ 的最陡乘积下降）。
- **Phase C 切向**（standoff 处半径保持，默认关）：近场 DD-gate 优化的
  预备实现。

## 边界与后续

1. **25-seed 配对确认（预注册，含 Amendment 1）**：基线 19/25；解耦
   top-k 版 18/25（NOT PASS，4 个强 seed 回退源于证据投递率 0.9997→0.948）；
   **混合 free-ride+top-up 版 20/25（PASS）**——证据条目免费携带后验，
   top-up 与证据不相交且每源 union ≤2，投递率回到 0.9990。
   冻结目标已更新为混合候选（见 `frozen/STABLE_DISTRIBUTED_DYNAMIC_K12_V1_2026-08-25.md`）。
2. **规模曲线（Amendment 2）**：6×6 门未动（均值改善）、8×8 **PASS**
   （门 +1，worst LCB +0.009）；K12-recal 对运行中。K6/K8 的 400 bit 预算/
   union 上限 2 是 K12 调参，小规模证据 deadline 裕量更紧，后续按规模重调
   通信预算。
3. 重新训练（审计 #6）：当前全部验证是 K10 checkpoint 在 K12 动态环境
   的零训练迁移消融（episodes=0, lr=0）；完整重训是独立一轮。
4. 仓库清理（审计 P2）：322 项工作树变更待一次干净的发布提交；
   冻结文档已绑定 `source_snapshot.zip` 哈希（611 文件，
   86c4f196…），恢复方式与 V1 一致。
