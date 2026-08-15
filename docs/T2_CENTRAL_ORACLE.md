# T2 Central Oracle — 安全/低暴露约束下的统一母问题可达性（advice 011）

> 文档日期：2026-08-16。依据 [`advice/011.md`](../advice/011.md) 的 T1/T2 阶段。
> 工具：`tools/audit_horizon_joint_oracle.py`（新增 `--inner exposure`、
> `--d-sep`、`--d-standoff`、`--exposure-gamma`、`--comm-coupling`、
> `--packet-bits`）。结果 JSON：`results/_t2_pareto_{baseline,standoff50,
> standoff100,exp9,exp10,both}/summary.json`。

## 1. 统一母问题 (P) 与 Inner (I) 的实现

把 D1.x 小模块收敛为 advice 011 的母问题：

```text
max_{X,Z,P,t} t
s.t.  D_q(X,Z,P) >= t,  ∀q                    （max-min 感知）
      P_comm,i + Σ_q p_iq <= P_max            （1 W 联合 RF 预算）
      U2U communication feasible              （b_i = 1 − P_comm^min(X,Z)）
      Z ∈ Z_HD                                （半双工结构）
      X ∈ X_safe                              （UAV 间距 d_sep + 目标 standoff d_standoff）
      E_w(X,Z,P) <= Γ_w,  ∀w                  （低电磁暴露）
      p_iq >= 0
```

**Inner (I)**（固定 X,Z 的 LP，`_exposure_maxmin_lp`）：

```text
max_{p,t} t
s.t.  Σ_i a_iq p_iq >= t,  ∀q
      Σ_q p_iq <= b_i,     ∀i
      Σ_i c[i,q] p_iq <= Γ_q,  ∀q   （目标侧暴露，观察者 = 目标）
      p >= 0
```

对偶返回**统一三价格**：`λ`（感知瓶颈）、`β`（本地 RF 资源）、`μ`（暴露）。
局部净价值 `s_iq = λ_q a_iq − μ_q c[i,q]`（瓶颈感知收益 − 隐蔽暴露代价）。

**暴露系数** `c[i,q] = G_tx·(λ_c/(4π d_3d))²`（每瓦泄漏增益，含 UAV 高度）。

**通信耦合** `b_i = 1 − P_comm^min(X,Z)`（`_min_comm_power_budget`，镜像
env_core 的解析最小通信功率：SNR 阈值 + Shannon 串行化 + deadline）。

## 2. 审计中发现的并修复的两个数值问题

1. **HiGHS 相对容差**：Γ~1e-9 的约束绝对尺度太小，HiGHS 默认容差允许 ~2.3e-10
   的绝对违反（相对 Γ 是 23%）。修复：**归一化暴露约束**（除以 Γ，b_ub=1），
   使约束尺度 O(1)。
2. **budget fill 违反暴露**：原 fill 把 slack 加到 argmax-gain 目标，可能把 E 推过
   Γ。修复：fill 到**暴露余量最大**的目标（尊重 Γ）。

修复后验证：Γ ∈ {1e-8..1e-12} 全部严格满足暴露约束（E ≤ Γ + 1e-12），功率平衡
≤1e-16。

## 3. T2 Pareto sweep（20 seed，H=10，教师最终几何起点）

| 配置 | mean worst | weak3 | steady | QoS 可行率 | 总移动 |
|---|---|---|---:|---:|---:|---:|
| baseline（无约束） | 0.832 | 0.832 | 0.843 | 0.75 | 149 m |
| standoff 50 m | 0.807 | 0.807 | 0.816 | 0.65 | 8 m |
| standoff 100 m | 0.801 | 0.801 | 0.811 | 0.65 | 0 m |
| exposure Γ=1e-9 | 0.819 | 0.819 | 0.832 | 0.75 | 115 m |
| exposure Γ=1e-10 | 0.790 | 0.790 | 0.805 | 0.65 | 83 m |
| standoff 50 + Γ=1e-9 | 0.801 | 0.801 | 0.811 | 0.65 | 3 m |

## 4. 结论（回答 advice 011 T2 的问题）

> **安全 + 低暴露约束下，强 oracle 的 worst 仍保持 0.79~0.81（≥ 0.75 保守线，
> 接近 0.80）——物理可行域足够，不是靠算法包装。**

- **安全间距是主要限制**：standoff 50-100 m 时 move→0（部署轨迹终点 UAV 已侵入
  安全区，任何移动被硬约束拒绝），几何优化被安全约束封顶在 worst ~0.80。
  启示：**若系统纳入 standoff 硬约束，需从满足约束的初始几何/轨迹重新规划**，
  不能从违反约束的部署终点出发。
- **低暴露约束**：Γ=1e-9 影响小（−0.013），Γ=1e-10 影响明显（−0.043）但仍 ≥0.75。
- **组合约束**（standoff50 + Γ=1e-9）：worst 0.801。

## 5. 对当前系统的处理方式（advice 011 §12 路线）

- **T2 完成**（本 Gate）：统一母问题、inner (I) + 三价格、安全/暴露硬约束、
  Pareto 面均已落地并验证。
- **T3 方向明确**：用统一三价格 (λ*, μ*, β*) 做分布式协调——不广播全局增益张量，
  而是低维价格 → 本地 bid（s_iq = λ a − μ c）→ 有限轮列生成/结构更新。Inner 是
  精确 LP，可复用现有 Dantzig-Wolfe 基础。
- **保留/降级**（advice 011 §末尾）：保留 max-min LP、真实 U2U 通信、1 W 硬预算、
  对偶价格、原子安全提交、统计证书；capability gauge 降级为 feasibility/safety
  guard；hybrid objective、hard responsibility assignment 不再发展；MAPPO 只保留
  预测/warm-start 辅助角色。

## 6. 复现

```bash
python tools/audit_horizon_joint_oracle.py --seed-limit 20 --horizon 10 --rounds 3 \
  --inner maxmin   --output results/_t2_pareto_baseline/summary.json
python tools/audit_horizon_joint_oracle.py --seed-limit 20 --horizon 10 --rounds 3 \
  --inner maxmin --d-standoff 100 --output results/_t2_pareto_standoff100/summary.json
python tools/audit_horizon_joint_oracle.py --seed-limit 20 --horizon 10 --rounds 3 \
  --inner exposure --exposure-gamma 1e-10 --output results/_t2_pareto_exp10/summary.json
```
