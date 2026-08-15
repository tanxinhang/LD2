# D0.88：认证 staleness 界 + 功率 hold-H 节奏

> 状态：完成（8/8 test20）。理论已数值验证，工程结论已出。
> 实现：`uav_isac/coordination/power_staleness.py`、
> `tools/audit_d088_event_trigger.py`。
> 结论：**认证最坏界过于松，不能当重解触发；但实际 staleness 很小，功率
> hold-H=5（对齐结构节奏）只损失 0.042 mean-worst，省 65% 重解通信。**

## 1. 理论：max-min LP 值的扰动 Lipschitz 界

固定 owner 的 max-min 功率 LP 值函数

```text
t*(A) = max_{p>=0} min_q sum_i a_iq p_iq  s.t. sum_q p_iq = b_i
```

关于增益矩阵扰动满足两个**已数值验证**的界（`power_staleness.py`）：

```text
(1) 值函数 Lipschitz:  |t*(A') - t*(A)| <= B,   B = sum_i b_i max_q |a'_iq - a_iq|
(2) 持有功率 staleness: t*(A') - t_held <= 2B,  t_held = min_q sum_i a'_iq p*_iq
```

(1) 来自对偶 `t*(A)=min_λ Σ_i b_i max_q λ_q a_iq` 与 `max` 的 1-Lipschitz 性；
(2) 是 (1) 与 `p*_iq<=b_i` 的一步三角不等式。二者都不要求可微。

几何漂移的一阶 Lipschitz（远离 DD 门）：

```text
|Δa_iq| <= a_iq * (2 du_i/R_iq + 2 du_owner(q)/R_owner(q),q
                    + 2 dt_q (1/R_iq + 1/R_owner(q),q))
```

DD 门 `g_dd>=g_min` 是**支撑不连续**：跨门使 `a_iq` 跳零，Lipschitz 界失效，
因此跨门必须强制重解（`dd_gate_crossing`）。

验证：200 个随机增益扰动上 (1)(2) 均为真上界；100 个 Friis 几何扰动上一阶界为
真上界（`tests/test_power_staleness.py`，5 passed）。

## 2. 结果：hold-H 节奏（8/8 test20，20 episodes）

| hold_frames | mean-worst | 重解次数 | 重解/帧 | 实际 staleness(D) | 认证界 2B(D) |
|---:|---:|---:|---:|---:|---:|
| 1（每帧） | **0.6363** | 3000 | 1.00 | 0 | 0 |
| 2 | 0.6093 | 1587 | 0.53 | 1.79 | 1628 |
| 5 | 0.5946 | 1048 | 0.35 | 2.26 | 1775 |
| 10 | 0.5430 | 754 | 0.25 | 4.16 | 3270 |
| 25 | 0.4641 | 621 | 0.21 | 5.44 | 5228 |

对照：deployed（不接 LP）mean-worst `0.3916`。

## 3. 判定

1. **H=5（对齐结构 hold）是最佳工程点**：mean-worst `0.5946`，仍比 deployed 高
   `+0.203`，而重解次数降到 35%（每重解约 832 bit，即 1048 次 → 0.87 Mbit/episode
   量级，vs 每帧 2.5 Mbit）。
2. **认证界 2B 约松 900 倍**（1628 vs 实际 1.79）：最坏界把 Σ_i b_i max_q|Δa| 的
   对抗性上界当成重解触发，导致几乎每帧都触发。结论：**2B 是"持有功率不 stale"
   的安全证书，不是通信节约的触发规则**。
3. **实际 staleness 随 H 近线性增长**（1.79→2.26→4.16→5.44），且常数远小于最坏界，
   因为几何漂移是平滑、相关的（Friis 连续），不是对抗的。因此实用触发应是
   "结构变化 + 选中边跨 DD 门"强制重解，平滑漂移在 H=5 内可忽略。
4. **warm-start 未验证有效**：朴素 `incumbent_power_w` 热启动在冒烟里未减少轮数
   （且出现异常略降 worst，疑似 16-bit 列量化与 `_fill_budget` 交互），D0.88 的
   "event 4 轮 + 平稳 1 轮 warm-start" 需要更仔细工程，不能当免费午餐。

## 4. 对下一步的指引

- 功率层接入部署：**每结构周期（H=5）一次 exact LP / 4 轮 DW + 选中边跨门强制重解**，
  是当前性价比最高的部署形态；H=2 可恢复 0.6093（重解 53%），H=10 掉到 0.543。
- 若要"1 轮 warm-start"通信，需要先修列量化与 incumbent 的交互，并配持久价格，
  否则 4 轮仍是诚实口径。
- 认证界保留为 no-harm 安全边（"持有功率的 certified 上界 ≤ 门限"），不作为触发。
