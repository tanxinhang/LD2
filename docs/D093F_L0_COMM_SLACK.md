# D0.93-F（L0）：通信余量回收审计（结构层获得强因果依据）

> 状态：完成（8/8 test20 前 3 seed）。
> 工具：`tools/audit_d093f_l0_comm_slack.py`。
> 依据：`advice/006.md`。
> 结论：**通信余量真实且巨大（均值 1.99 W），但只解释 23% 的 escalate epoch；
> 77% 是真正的 sensing/structure deficit。**

## 1. 方法（保持协议语义不变，只回收 link margin）

固定 active Tx、Token 内容/量化、packet bits、receiver 集、带宽分配、role/owner/
structure、deadline、delivery 判据，只反解最小广播功率：

```text
P_comm,i^min = max_{j∈R_i}  Γ_req · N0 · B_eff / g_ij,
Γ_req = max(Γ_th, 2^(L_i/(B_eff·(T_ddl−T_proc))) − 1),
g_ij = antenna_gain · (λ/(4πd_ij))²,  B_eff = B / n_active.
```

这是当前 orthogonal-U2U Shannon 链路模型的直接反解，不是 heuristic。回收后的
`b'_i = 1 − P_comm,i^min ≥ b_i`，由单调性 `γ*(A,b') ≤ γ*(A,b)`（严格 no-harm）。

## 2. 结果

| 指标 | 值 |
|---|---:|
| 平均通信余量 | **1.99 W** |
| escalate epoch 总数 | 52 |
| escalate→stay | 1 |
| escalate→ambiguous | 11 |
| escalate→escalate | **40** |
| **通信可回收比例** | **23.1%** |

## 3. 判定（负结果，但推动架构收敛）

1. **通信功率严重过度分配**：Actor 用 ~0.25 W/sender，解析最小功率 ~69 µW
   （~3600x），平均每帧 ~1.99 W 通信功率是 link margin，可回收为感知预算。
2. **但通信余量不是主因**：回收后 52 个 escalate epoch 只有 1 个变 stay、11 个变
   ambiguous，**40 个（77%）仍是 genuine sensing/structure deficit**。
3. **结论**：advice 006 的"负结果"分支命中——`通信 slack 不是主因，结构层升级得到
   更强因果依据`。这堵住了审稿质疑"为什么结构重构前不先回收通信功率"：**回收了，
   但 77% 还是不够，所以结构层是必要的**。

## 4. 对创新主线的意义

L0 不包装成主创新（本质是 link-margin-aware minimum power），它让
"minimum-necessary-layer activation" 真正成立：

```text
L0 回收通信 slack → L1 认证 sensing capability → L2 结构 → L3 几何
```

正/负结果都推动收敛：**结果是 77% 结构瓶颈 → 结构层（min γ_S）获得强因果依据，
是下一步的主方向**。

## 5. 下一步

- **D0.93-A2（双侧 PWL capability certificate）**：`γ_U ≤ γ* ≤ γ_L`，把 L1 的
  能力包络变成可部署 LP 证书生成器。
- **L2 结构层（min γ_S）**：对 77% 的 escalate epoch，用 McCormick 精确 MILP 求
  "改变 structure 能消掉多少 capability deficit"。
