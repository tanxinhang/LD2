# D0.93-A2：双侧 PWL capability certificate（可部署 LP）

> 状态：完成（函数级 sandwich + LP 化 + γ 三级 sandwich 验证）。
> 实现：`uav_isac/coordination/pwl_pd.py`、`capability.py::capability_gauge_pwl_lp`。
> 依据：`advice/003.md §7`、`advice/004.md §3–6`。
> 结论：**P_D 的 chord 下界 + tangent 上界使 capability gauge 变成 LP，且
> γ_optimistic ≤ γ* ≤ γ_conservative 验证通过。**

## 1. 双侧 PWL（凹函数的 chord/tangent）

`P_D(D)=Φ(√D−c)` 在 `D≥c²/3` 凹。对凹函数：
- **chord**（两断点线性插值）在函数**下方** → 下界 `f̄ ≤ P_D`；
- **tangent**（断点切线）在函数**上方** → 上界 `f̄ ≥ P_D`。

二者都是凹 PWL，可写成**仿射片的逐点 min**：`f(D) = min_m (slope_m·D + intercept_m)`。
于是 `f(D_q) ≥ ρ` ⟺ `slope_m·D_q + intercept_m ≥ ρ ∀m`（一组线性约束）。

## 2. 断点：曲率自适应（advice 004 §5）

chord 误差界 `0 ≤ P_D−chord ≤ (M_m/8)(Δd)²`，故 `Δd ≤ √(8ε/M_m)`，断点高曲率区密、
低曲率区稀，不是固定 8/16 段扫描。`M_m=|P_D''|` 用解析式（近端点曲率小、远端点大），
fixed-point 迭代取远端曲率（增曲率区更保守）。已测：ε 越小误差严格单调降，且
chord ≤ P_D ≤ tangent、误差 < 1e-2。

## 3. LP 化（可部署）

capability gauge 用 PWL 界后变成 **linprog**：

```text
min γ  s.t.  D_q = Σ_i a_iq p_iq
      y_q ≤ slope_m·D_q + intercept_m  ∀m,q      （PWL）
      Σ_q y_q ≥ Q·ρ_avg                          （steady）
      D_q ≥ d_min                                 （worst，精确线性）
      z_q ≥ τ−y_q, k·τ−Σ z_q ≥ k·ρ_tail          （bottom-k，O(Q)）
      Σ_q p_iq ≤ γ·b_i                            （budget）
```

worst 约束**不需要 PWL**（`P_D` 单调可精确反解 `D_q≥d_ρmin`），保守误差只来自
steady/bottom-k（advice 004 §6）。

## 4. 三级 sandwich（验证通过）

```text
γ_optimistic = γ*(tangent 上界)  ≤  γ*_exact  ≤  γ_conservative = γ*(chord 下界)
```

20 个随机 (A,b) 上验证通过。三态路由（advice 004 §4）：
`γ_conservative ≤ 1` → certified sufficient；`γ_optimistic > 1` → certified
insufficient；否则 → ambiguous（defer/refine，不误触发结构重构）。

## 5. 意义与定位

- **LP 化是关键**：capability gauge 从 SLSQP 变成 linprog，于是 D0.87/D0.88 的
  强对偶、Dantzig–Wolfe、量化价格、staleness、event-trigger **全部可继承**。
- **严格可行性界**：chord 通过 ⟹ 真实 QoS 通过；tangent 失败 ⟹ 真实不足；中间
  才需要 refine。
- **部署形态**：PWL 是"低复杂度 capability certificate generator"，在 epoch/event
  时运行（不是逐帧），并可用价格/列交换分布式实现。

## 6. 验证状态

- `tests/test_pwl_pd.py`：3 passed（chord≤P_D≤tangent、ε 单调、高曲率区更密）。
- `tests/test_capability.py`：7 passed（含 PWL LP 三级 sandwich）。
