# D0.93 L3-T1/T2：Capability-Sensitivity 几何梯度（KKT + finite-difference 验证）

> 状态：完成（T1 推导 + T2 数值验证）。
> 实现：`capability.py::capability_gauge_pwl_lp_full`（显式 D_q + 对偶 π*）、
> `tests/test_capability.py`（envelope 梯度 finite-difference）。
> 依据：`advice/007.md`。
> 结论：**∂γ\*/∂a_iq = π_q\* · p_iq\* 验证通过；capability primal 的 KKT 价格 π\*
> 是正确的几何灵敏度，不是旧 max-min 的 λ\*。**

## 1. L3 核心问题（按 advice 007 §1）

```text
min_x γ*(x; S)    其中 γ* = 能力 gauge（完整任务 worst+bottom-k+steady 的预算缩放）
```

UAV 移动追求 `γ*(x+Δx) < γ*(x)`，最终 `γ* ≤ 1`。L0/L1/L2/L3 语言统一为"**哪个控制
自由度能消除 task capability deficit**"，不是"谁把 P_D 做最大"。

## 2. L3-T1：从 capability primal 的 KKT 重新推导（不是旧 λ*）

把能力 LP 改成显式 `D_q` 变量 + 耦合等式 `D_q − Σ_i a_iq p_iq = 0`。该等式的对偶
乘子就是 **target sensitivity price π_q\***。由 envelope theorem：

```text
∂γ*/∂a_iq = −π_q* · p_iq*
```

这与旧 max-min 的 `λ* p` 同构，但 π\* 来自**完整任务集**（worst + bottom-k +
steady + budget scaling），不是只最差目标的 λ\*。scipy `eqlin.marginals` 用相反
符号，实证恒等式是 `∂γ/∂a = +marginals_q · p_iq`。

## 3. L3-T2：finite-difference 验证（理论—代码一致性）

| 验证 | 结果 |
|---|---|
| 单条目 ∂γ/∂a_iq（3 个 (i,q)） | finite-diff 与 π_q·p_iq 一致（atol 1e-3） |
| 随机方向导数（h=1e-3/1e-4/1e-5） | finite-diff 与 g·d 一致（atol 1e-2） |

物理符号正确：`∂γ/∂a_iq < 0`（增益越大，所需预算缩放越小），几何更新沿 `−∇_x γ`
移动（靠近目标、改善 sensing gain）单调降 γ。

## 4. 与 advice 007 的关键一致点

- **π\* 不是 λ\***：π\* 编码了 steady/bottom-k 的影子价，λ\* 只编码 worst。
- **双基地双身份**：`a_iq = a_{i,j_q,q}`，UAV k 移动时 Tx（k=i）与 Rx/owner（k=j_q）
  两个身份的 `∇_x a` 都非零，正式实现必须都加（advice 007 §4）。
- **DD 门边界**：`∇_x a` 只在 DD support 不变时有效，跨门当事件重解（advice 007 §5）。
- **通信耦合项**：`∇_x γ = −Σ π p ∇_x a + γ Σ β ∇_x b`（sensing geometry + comm
  geometry），第一版 L3 可先冻结 b 做 sensing-only，正式版再加（advice 007 §3）。

## 5. 验证状态

- `tests/test_capability.py`：9 passed（含 2 个 envelope 梯度测试）。
- `tests/test_pwl_pd.py`：3 passed。

## 6. L3-G1：one-step trust-region（已验证）

`capability_geometry_gradient`（`capability.py`）实现 Friis 几何梯度，**同时含 Tx
与 Rx/owner 双身份效应**（advice 007 §4）：

```text
dγ*/dx_k = Σ_{i,q} π_q p_iq · [1[k=i]·(−2a_iq(x_i−x_q)/R_iq²)
                             + 1[k=owner_q]·(−2a_iq(x_owner−x_q)/R_owner²)]
```

`tests/test_l3_geometry.py`：2 passed——

1. **one-step 严格下降**：沿 `−∇_x γ*` 做 trust-region 步（步长按半径 cap），10 个
   随机几何上 γ* 单调不增（严格下降）；
2. **梯度—几何 finite-difference 一致**：单 UAV 单维扰动下，`∂γ*/∂x_k` 与 Friis
   梯度一致（atol 5e-2），确认 Tx/Rx 双身份都被正确计入。

## 7. L3-G2：capability-sensitivity vs baselines（已验证）

`tests/test_l3_geometry_baselines.py` 在 60 个随机几何上，one-step（2 m）移动的
γ\* 平均下降：

| baseline | mean Δγ |
|---|---:|
| no movement | 0.0 |
| toward-worst-target | 1.67 |
| Friis distance（靠近最近目标） | 3.08 |
| **capability-dual（π\* 加权）** | **3.75** |

**判定**：capability-dual 比最强简单 baseline（Friis 距离）高 **+22%**，比
toward-worst 高 **+124%**。说明 π\*（完整任务对偶价：worst+bottom-k+steady）提供了
额外决策信息，不是"离目标近一点自然就好"。注意：这是合成 Friis 几何（无 DD 门），
真实 8/8 硬帧的验证是 L3-G3。

## 8. 下一步（advice 007 的 Gate 顺序）

- **L3-G3**：在 8/8 真实硬帧上做 5–10 step receding-horizon，看多少 γ>1 状态被
  推到 γ≤1。
- **L3-C**：加通信梯度项 `γ Σ β ∇_x b`，验证 sensing-only 与 true ISAC geometry
  的差异。
