# TC-SP-MAPPO vs 当前系统差距分析

> 日期：2026-07-14

---

## 差距总览

| 类别 | 需重构 | 需修改 | 已有 |
|------|:---:|:---:|:---:|
| Actor | 1 | 1 | 1 |
| Critic | 3 | 0 | 0 |
| Reward/Credit | 2 | 2 | 0 |
| Safety Gates | 2 | 1 | 0 |
| 实现审计 | 2 | 1 | 0 |
| **合计** | **10** | **5** | **1** |

---

## 一、Actor

### 1.1 架构 ✅ 不需改

当前 StructuredActorNetwork (Self/Target/Neighbor Encoders + Cross-Attention + Gate + Heads) 与提案一致。DAgger 0.71-0.74 steady 证明架构容量足够。

### 1.2 Attention 冻结 ✅ 已完成 (S1, 2026-07-14)

**当前**：通过 `split_param_groups()` 显式分组，`ATTENTION_PARAM_PREFIXES = ('attn.', 'attn_norm.')`。

```python
enc_params, head_params, attn_params = split_param_groups(actor.named_parameters())
for p in attn_params:
    p.requires_grad_(False)
# → 6 params frozen: attn.in_proj_*, attn.out_proj.*, attn_norm.*
```

**改动位置**：`networks.py`（split_param_groups）, `trainer.py`（freeze 逻辑）。

### 1.3 Encoder LR 🔴 需降

**当前**：Encoder LR = 3e-4 (与 Heads 相同)。

**提案**：Encoder LR = 1e-5，Heads LR = 5e-5，Attention LR = 0。

---

## 二、Critic

### 2.1 单标量 → Per-target 向量 ✅ 已完成 (S3, 2026-07-14)

**当前** (`networks.py:CriticNetwork`)：
```python
self.value_head = nn.Linear(last_dim, 1)           # scalar V(s)
self.target_heads = nn.ModuleList([                # per-target V_q(s)
    nn.Linear(last_dim, 1) for _ in range(Q)
])
# forward_with_targets() → (scalar_v, per_target_v)
```

### 2.2 GAE 单标量 → Per-target GAE ✅ 已完成 (S3c, 2026-07-14)

**当前** (`buffer.py:compute_gae`)：
```python
# Scalar GAE (unchanged)
delta = rewards[t,k] + gamma * next_v * masks[t,k] - values[t,k]

# Per-target GAE (NEW)
delta_q = per_target_rewards[t,k,q] + gamma * next_v_pt * mask - per_target_values[t,k,q]
A_q = delta_q + gamma * lambda * mask * A_next_q
# Returns per_target_advantages (T, K, Q) and per_target_returns (T, K, Q)
```

**遗留**: Actor loss 仍使用 scalar advantage。S4 需要将 `per_target_advantages` 加权聚合为 per-agent advantage 并接入 actor loss。
```python
# S4 (待实施):
A_k = sum_q w_q * alpha_{kq} * A_q - lambda_comm * A_comm
```

### 2.3 Critic 预训练 🔴 需新增

**当前**：Actor 和 Critic 同步从 DAgger 或随机初始化开始训练。

**提案**：先冻结 Actor 5-10 rollouts，只训练 target-wise Critic。要求 explained variance 不再显著为负后才开放 Actor。

---

## 三、Reward / Credit

### 3.1 UAV-目标反事实贡献 🔴 需新增

**当前**：所有 UAV 获得相同团队奖励（或简单差异奖励）。

**提案**：
```python
# 对每个 UAV k 和每个目标 q:
P_full = compute_P_D(all_uavs)           # 所有 UAV 参与
P_minus_k = compute_P_D(all_uavs \ {k})  # 移除 UAV k
m_{kq} = P_full[q] - P_minus_k[q]        # 边际贡献
alpha_{kq} = max(m_{kq}, 0) / (sum_i max(m_{iq}, 0) + eps)
```

计算量：8 UAV × 8 目标 = 64 次反事实 P_D 评估。需要在 `env_core.py` 中实现 `compute_counterfactual_P_D()`。

### 3.2 弱目标平滑加权 🔴 需新增

**当前**：mean-P_D (所有目标等权)。

**提案**：
```python
d_q = max(0, tau - P_D_q)               # deficit, tau=0.3
w_q = (1-rho)/Q + rho * (d_q+eps)/sum(d_q+eps)  # rho=0.2
```

与 Bottom-3/CVaR/SoftMin 的区别：80% 保留平均性能导向，20% 倾斜弱目标——不会主导训练导致崩溃。

### 3.3 通信成本独立 head 🟡 需修改

**当前**：通信成本混入团队奖励 `-lambda_report * total_bits`。

**提案**：独立 comm value head 计算通信 advantage，不混入目标检测奖励。

---

## 四、Safety Gates

### 4.1 更新回滚机制 🔴 需新增

**当前**：无回滚，PPO 更新直接应用。

**提案**：
```python
# Before update: save state
actor_backup = copy.deepcopy(actor.state_dict())
# After update: validate
if delta_weak3 < -0.02:
    actor.load_state_dict(actor_backup)  # rollback
    actor_lr *= 0.5                      # halve LR
    discard_rollout()                    # don't reuse
```

### 4.2 Validation/Test 分离 🟡 需修改

**当前**：5-seed online eval 同时用于监控和选 checkpoint。

**提案**：
- Online validation: 5 seeds (训练监控, early-stop)
- Independent test bank: 100 seeds (最终报告, 永不用于训练决策)

---

## 五、实现审计

### 5.1 Neighbor GRU slot 一致性 🔴 需审计

**当前**：未检查邻居排序是否跨帧一致。

**风险**：如果邻居按距离排序，每帧顺序变化，GRU 把不同 UAV 当成同一时序实体，直接破坏协同语义。

**检查项**：
1. Episode 开始 hidden state 清零
2. 不同 UAV hidden state 独立
3. 同一 neighbor slot 始终对应同一 UAV
4. 邻居进入/离开时 mask 正确

### 5.2 Communication gate 饱和度 🔴 需审计

**当前**：未记录 gate 行为。

**检查项**：
1. gate 均值/方差/entropy
2. 0/1 饱和比例
3. `comm_agent_var` vs `comm_batch_var`
4. gate 与 weak3 失败的 correlation

### 5.3 Actor-Critic 信息边界 🟡 需确认

**当前**：Critic 用全局状态 (153-dim)，Actor 用局部 obs (454-dim)。

**确认项**：Actor forward 中无隐藏 global state 使用；normalization 统计不混合训练期全局变量。

---

## 六、实施进度

> **2026-07-14 P0 修复**：发现并修复了 GRU/PPO 状态一致性 bug（此前所有 PPO 训练均在 ratio 非法状态下进行）。修复后 1 次 PPO 更新不再破坏 DAgger。以下进度已更新。

| 阶段 | 内容 | 状态 | 结果 |
|:---:|------|:---:|------|
| **P0** | **GRU/PPO 状态一致性修复** | **✅ 完成** | ratio 验证 max\|diff\|<1e-5, Full/EH 1 update 不破坏 DAgger |
| **P0** | **Attention 冻结补全 (attn_norm)** | **✅ 完成** | 6 params frozen, 显式 param group |
| **P0** | **Q 硬编码移除** | **✅ 完成** | num_targets 全链路 config 驱动, 新增 k8q8 config |
| **P1** | **PD_hist 接入 Actor** | **✅ 完成** | pd_hist_proj → target entity residual modulation |
| **S1** | Attention 冻结 + 降低 LR | **✅ 完成** | 已补全 attn_norm, 显式 param groups |
| S2 | 更新回滚机制 | ⏳ 待实施 | — |
| S3 | Per-target Critic heads (诊断) | ✅ 完成 | heads 已添加, forward_with_targets() 就绪 |
| **S3b** | **Per-target storage + values** | **✅ 完成** | buffer 存储 per-target rewards + values |
| **S3c** | **Per-target GAE** | **✅ 完成** | buffer.compute_gae 计算 per-target advantages/returns; Actor loss 仍用 scalar advantage, target-wise integration 待 S4 |
| S4 | 反事实贡献 + 弱目标加权 | ⏳ 待实施 | — |
| S5 | Neighbor GRU + Gate 审计 | ⏳ 待实施 | — |

### S1 详细结果 (2026-07-14, P0 修复后重新测试)

**第一轮（h=0 eval, 非严格配对）：**
| 模式 | Δsteady | Δweak3 | Δworst | PPO ratio OK? |
|:---:|:---:|:---:|:---:|:---:|
| Full PPO 1 update | +0.002 | +0.004 | -0.006 | ✅ max\|diff\|=8e-6 |
| EH PPO 1 update | +0.000 | -0.001 | +0.000 | ✅ (attn frozen) |

**第二轮（streaming GRU eval, strict paired, zero-init ckpt, 2026-07-14 晚）：**
| 模式 | Δsteady | Δweak3 | PPO ratio OK? |
|:---:|:---:|:---:|:---:|
| Full PPO 1 update | -0.008 | -0.020 | ✅ max\|diff\|=7e-6 |
| EH PPO 1 update | -0.001 | -0.003 | ✅ max\|diff\|=5e-6 |

**结论 (更正)**：streaming GRU 评估下 DAgger baseline 的 weak3 更高 (0.283 vs h=0 eval 的 0.180)，说明此前评估低估了 DAgger 真实性能。修复后 Full PPO weak3 降幅 ~2%（vs 修复前 ~26%），EH 几乎不变。GRU/PPO ratio 修复已确认有效。长期训练稳定性仍需 >20 updates 验证。
