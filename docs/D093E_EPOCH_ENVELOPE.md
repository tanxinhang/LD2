# D0.93-E：Epoch-Level Capability Envelope（三态认证路由）

> 状态：完成（8/8 test20 前 3 seed，90 个 epoch）。
> 实现：`uav_isac/coordination/capability.py`（库模块 + 单调性）、
> `tools/audit_d093e_epoch_envelope.py`（epoch 包络审计）。
> 依据：`advice/005.md`。
> 结论：**单调性 Proposition 验证通过（0/152 违例）；三态路由 stay 6.7% /
> escalate 57.8% / ambiguous 35.6%。**

## 1. 理论：capability gauge 单调性（Proposition）

```text
A^(1) ⪯ A^(2), b^(1) ⪯ b^(2)（逐元素）  ⟹  γ*(A^(2),b^(2)) ≤ γ*(A^(1),b^(1)).
```

理由：较差 gain/budget 下可行的分配，在更好 gain/budget 下仍可行，故最小预算
缩放不可能增大。于是对窗口内的系数区间 `A^- ≤ A_t ≤ A^+`、`b^- ≤ b_t ≤ b^+`：

```text
γ^L := γ*(A^+, b^+)  ≤  γ*_t  ≤  γ^U := γ*(A^-, b^-)   ∀t∈窗口.
```

**验证**：100 个随机单调对 + 152 个 trace 逐帧 sandwich 检查，**0 违例**。

## 2. 三态认证路由（无经验阈值）

| 状态 | 条件 | 动作 |
|---|---|---|
| certified stay | `γ^U ≤ 1` | 保持 power-only，中层不启动 |
| certified escalate | `γ^L > 1` | 结构能力确定不足，启动结构层 |
| ambiguous | `γ^L ≤ 1 < γ^U` | 不确定 → defer/refine，不触发结构 |

**关键原则：不确定性本身不触发昂贵结构动作。**

## 3. 结果（8/8，90 epoch，H_c=5）

| 状态 | epoch 数 | 比例 |
|---:|---:|---:|
| stay（γ^U≤1） | 6 | 6.7% |
| escalate（γ^L>1） | 52 | 57.8% |
| **ambiguous（γ^L≤1<γ^U）** | **32** | **35.6%** |

- **sandwich 0 违例**：γ^L ≤ γ*_t ≤ γ^U 在所有 152 个可解帧上成立，单调性包络
  理论上 sound。
- **~1/3 的 epoch 是 ambiguous**：区间包络无法证明行/不行，正确动作是 defer（等待
  下一次观测或缩短 check），而非立刻 structure rebuild——避免用"单帧 γ>1"的抖动触发
  昂贵结构重构。这正是 advice 005 的核心：`γ*_t` 是离线 oracle，在线用 epoch 包络。

## 4. 多时标架构（advice 005 §10，正式收敛）

```text
fast   : sensing power adaptation   frame/short hold   p*
meso   : capability certification   epoch/event        [γ^L, γ^U]
slow   : structure repair           γ^L > 1            S
slower : geometry repair            γ_S^L > 1          x
```

路由触发是**持久/epoch 级 certified capability deficit**，不是单帧 γ_t。

## 5. 创新点升级

从"capability gauge"升级为 **Interval-Certified Multi-Timescale Capability
Routing**：

```text
A^- ≤ A_t ≤ A^+  →（monotonicity）→  γ^L ≤ γ*_t ≤ γ^U  →（三态认证）→
stay / escalate / defer，DD/结构/信道事件触发更新。
```

同时解决：不能逐帧求 oracle、channel/geometry 不确定、结构抖动、U2U 信令开销，
且保留严格数学保证。

## 6. 下一步

- **D0.93-A2（双侧 PWL）**：`γ_U ≤ γ* ≤ γ_L`（chord 下界 + tangent 上界），把
  capability envelope 的"乐观/保守"端变成可部署的 LP 证书生成器。
- **L0 通信余量回收**：γ>1 的帧先区分"结构不够" vs "Actor 给通信留了过多功率"，
  再决定是否上结构层。
