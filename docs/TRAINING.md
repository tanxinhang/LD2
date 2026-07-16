# 训练:MAPPO / GAE / Lagrangian / 奖励 / P0 求解器

---

## 1. 参数共享

- **Actor 共享**:所有 UAV 用同一个 `ActorNetwork`(`agents[0].actor`)。
- **Critic 共享**:同一个 `CriticNetwork`。MAPPO 看全局状态,IPPO 看局部 obs。
- **优化器**:actor / critic 各一个 Adam,作用于共享网络。
- `MAPPOAgent` 列表里每个对象**引用同一对网络**(`agents[0]` 持有,其余复用)。
- **Agent 身份**:critic 输入拼接 `K` 维 one-hot 区分 agent(`trainer` 中 `torch.cat([base, agent_oh])`)。

网络结构(`networks.py`):
- 共享主干 `MLP(obs_dim → 256 → 256 → 256)` + ReLU。
- 头:`dp_mean_head(256→2)`;`dp_log_std`(可学习 (2,) 参数,经 `tanh` 平滑映射到 `[LOG_STD_MIN,LOG_STD_MAX]=[-1,1]`,保证梯度处处非零、避免熵单向塌缩);`role_head(256→3)`。
- Critic:`MLP(state_dim+K → 256 → 256 → 1)`。
- 初始化:隐藏层正交 `gain=√2`,输出头 `gain=0.01`。

---

## 2. Rollout 数据形状

并行 `num_envs=8` 个环境,每环境 `steps_per_env = rollout_steps // num_envs = 2048//8 = 256` 帧,
总 `buffer_size = 2048`。buffer 按 **env 交织**存储(见 `ARCHITECTURE §4.5`):
行索引 `b` → 时间步 `b // K`?不——展平发生在 `get_training_data`;buffer 行本身是 `(buffer_size, K, ·)`,
其中 env n 的第 s 步在行 `n + s·N`。展平后批样本 `b`:agent = `b % K`,全局状态行 = `b // K`。

> "8 envs × 256 steps = 2048" 与 "rollout_steps=2048" 是同一个量:`rollout_steps` 是总帧数,`steps_per_env` 由它除以 `num_envs` 得到。文档/配置以 `rollout_steps` 为准。

---

## 3. GAE 与 PPO

GAE(`buffer.py: compute_gae`,逐 agent、逐 env 反向):

```
mask_t = 1 − done_t
δ_t    = r_t + γ · V(s_{t+1}) · mask_t − V(s_t)
A_t    = δ_t + γλ · mask_t · A_{t+1}
R_t    = A_t + V(s_t)
```

- `γ=0.99`,`λ=0.95`。
- **多环境交织**:每个 env n 的轨迹在行 `n, n+N, n+2N, …` 上独立反向回溯,互不串味。
- 末步 bootstrap `V(s_{t+1})` 用 rollout 结束后对最终状态的 critic 估值(`next_values`)。
- `mask` 在 episode 末置 0,截断跨 episode 的优势泄漏。

PPO 更新(`trainer.update`):
- **优势标准化**:在 `get_training_data` 里**一次性全局**做 `(A−mean)/std`;**不**在 minibatch 内重复标准化(历史上重复归一化是个 bug)。
- ratio `= exp(new_logp − old_logp)`;裁剪 `ε=0.2`:`−min(ratio·A, clip(ratio,1±ε)·A)`。
- **KL 早停**:`approx_kl > 1.5·target_kl`(`target_kl=0.02`)或出现 NaN 即停止本轮剩余更新。
- **Critic 损失**:纯 MSE `0.5·(V−R)²`,**无 value clipping**(历史上 value-clip 在原始 return 尺度上限制更新量,导致 critic 永远追不上 return → 崩溃,已移除)。
- **熵奖励**:系数 `entropy_coef`(`entropy_init=entropy_final=0.08`,默认不衰减)。
- 总损失:`actor_loss + vf_coef·critic_loss − entropy_coef·entropy`(`vf_coef=0.5`)。
- 梯度裁剪 `max_grad_norm=0.5`;`ppo_epochs=4`,`minibatch_size=256`。
- **连续 + 角色 log-prob 相加**:`learn_roles=True` 时 `logp = logp_dp + logp_role`;`learn_roles=False` 时只有 `logp_dp`(`evaluate_actions` 与 `decode/compute_log_prob` 两侧一致剔除角色项,保证 ratio 正确)。

### GRU Hidden State 一致性 (P0 fix, 2026-07-14)

StructuredActorNetwork 使用 GRU 编码邻居时序。**Rollout 和 PPO 更新必须使用相同的 GRU hidden state**，否则 PPO ratio 条件分布不同，r_t ≠ 1 在第一次 optimizer step 之前就已失效。

- **Rollout 阶段**：每帧从 `env._gru_hidden[(k, kk)]` 读取 h_prev，actor forward 返回 h_new，写回 env 并存入 buffer。
- **Update 阶段**：从 buffer 读取 h_prev，reshape 为 `(1, mb*(K-1), D)`，传入 `evaluate_actions()` → `actor(obs, h_prev)`。
- **一致性断言**：训练开始时自动运行 `verify_old_log_prob_consistency()` — 用 buffer 中的 h_prev 重新计算 log_prob，与 rollout 时存储的 old_log_probs 对比。若 `max|diff| > 1e-4`，打印 `[PPO RATIO ERROR]` 并警告训练结果被污染。
- **Flat-MLP Actor**：`ActorNetwork`（无 StructuredActor）没有 GRU，h_prev 路径不生效（`gru_hidden_dim=0`），ratio 一致性由 tanh 重构路径保证（见 A2）。

---

## 4. 奖励分解

每帧团队奖励(`reward.py` / `env_core` 步骤 10):

```
r_team = Σ_q ω_q · U_q(D_q*) − λ_report · total_bits − constraint_penalty
```

- `U_q = −log(1 − P_D^q)`(`P_D∈[0,1]` → `U_q∈[0,∞)`,典型 0.3–3)。**注意:** 该效用关于 `P_D` 是凸的,且复合后关于 `D` 经验上非凹(见 `SYSTEM_MODEL §3.4` 与 `KNOWN_ISSUES B8`),故 P0 不享有次模/近似保证。
- `ω_q` 目标优先级(默认等权,和为 1)。
- `λ_report = 1e-5`(很小,远低于检测效用;历史上 1e-3 会让"少上报"比"多检测"更划算 → P_D 崩)。`total_bits = 选中链路数 × B_q`(B_q=64)。
- **`constraint_penalty` 在环境内传入 0.0**:`constraints.py` 算出的解析惩罚**不直接进奖励**,只放进 `info` 供诊断;约束违反改由 Lagrangian 处理(见 §下)。

个体 shaping(边际贡献,`compute_shaped_rewards`):

```
ΔU_k = U(D*) − U(D* 去掉所有涉及 k 的链路)      # delete-approximation
r_k  = r_team + η_mc · (ΔU_k − mean_l ΔU_l)      # η_mc = 0.5,shaping 项零和
```

**Lagrangian 约束惩罚(在 `trainer.collect_rollout`,对每个 agent 的奖励再扣):**

```
r_k ← r_k − λ_lag · any_violation        # any_violation ∈ {0,1}(本帧是否有任意约束违反)
```

λ 在每次 update 末按 rollout 平均违反率更新(`trainer.update`):

```
λ_lag ← clip( λ_lag + lr_lag · (mean_violation − max_violation_rate), 0, λ_max )
```
`lr_lag=0.002`,`max_violation_rate=0.1`,`λ_max=1.0`。

各项尺度/符号速查:

| 项 | 符号 | 典型范围 | 权重 | 团队共享 | 进 Lagrangian |
|---|---|---|---|---|---|
| 检测效用 | +Σω_q U_q | 0 ~ ~6 | ω_q | 是 | 否 |
| 通信开销 | −λ_report·bits | ~−1e-3 量级 | 1e-5 | 是 | 否 |
| 解析约束惩罚 | (传 0.0) | — | — | — | 否(当前不进奖励) |
| 边际贡献 shaping | η_mc·(ΔU_k−mean) | 小,零和 | 0.5 | 否(个体) | 否 |
| 约束违反 | −λ_lag·any_violation | 0 ~ −1 | λ_lag≤1 | 是 | 是 |

**两点技术债务(见 `KNOWN_ISSUES B10`、`EXPERIMENTS §6`):**
- **二值 Lagrangian**:轻微越界、严重碰撞、单目标略低于门限、全目标检测失败——都映射到同一个 `any_violation=1`、共用一个 λ,无法区分约束类型与严重程度。更合理的是分离的连续违反量 `r_k − Σ_m λ_m·g̃_m`。
- **通信成本几乎可忽略**:每条链路 `B_q=64 bits × λ_report=1e-5 ≈ 6.4e-4`,而检测效用典型 0.3–3。所以 `λ_report` 只是**弱正则**,真正的通信约束是 `capacity_per_rx` 硬上限。当前项目**未重点优化** bits–detection 的 Pareto 权衡;若论文要强调该权衡,需做 `λ_report` 敏感性实验。

---

## 5. 训练 vs 评估差异

| 项目 | 训练(rollout) | 评估(`_evaluate`) |
|---|---|---|
| 位移动作 Δp | 高斯采样 | 均值(确定性);四模式可切采样 |
| 角色 | `learn_roles=True`:Categorical 采样;`False`:P0 分配 | `learn_roles=True`:argmax;`False`:P0 分配 |
| 环境种子 | 每次 reset 随机抽 | 固定 `eval_env seed=12345` + 固定 `eval_seeds`(默认 [10001..10005]) |
| 每集 reset 种子 | 随机 | **固定**逐集种子(每次评估重放同一批场景) |
| 网络模式 | 训练(含熵探索) | `torch.inference_mode`,无探索噪声 |
| GRU 隐状态 | 持续维护,每帧更新后存回 env | 持续维护,每帧传入 `h_prev` 并保存 `h_new`,episode 边界清零;不更新网络参数 |

四种评估模式(`_evaluate_modes`,用于归因 eval 崩塌)。**为避免字母歧义,直接用描述性名称**(与代码 key 一致),不要用裸 A/B/C/D:

| 代码 key(== 日志名) | 位移 dp | 角色 role |
|---|---|---|
| `dp_det_role_stoch` | 确定(均值) | 采样 |
| `dp_stoch_role_det` | 采样 | 确定(argmax) |
| `full_greedy` | 确定 | 确定 |
| `full_stochastic` | 采样 | 采样 |

在**同一批固定场景**上对比。`learn_roles=False` 下角色恒由 P0 分配,故四模式仅在 dp 确定性上有别(role 维度退化)。评估同时聚合 `valid_pair_rate / no_TX_rate / all_same_role_rate` 三个配对诊断。

> 注:历史讨论中曾用过不同的 A/B/C/D 字母约定,易混淆;**以上述描述性名称为准**,代码 `_evaluate_modes` 的 key 也用全名。它正是定位旧版角色 argmax 全同角色 → 零配对的工具(见 `KNOWN_ISSUES A1`)。

---

## 6. P0 内层求解器(`physical/inner_solver.py: solve`)

**输入:** `deflection_entries`(所有候选链路的 `DeflectionEntry`,含 `d_eff`)、`reporting_rates`(可选)、`Q`、`K`、`enforce_single_role`。几何/角色经由 `deflection.compute` 已编码进候选集合。

**候选集合:**
```
C = { (i,j,q) : i≠j, d_eff>0, 满足约束 }
```
- `learn_roles=False`(role-agnostic):每架 UAV 既是候选 TX 也是候选 RX,枚举所有有序 `i≠j` 对 × Q,共 `K(K−1)Q` 条(默认 24,exp 场景 48)。
- `learn_roles=True`:候选按角色门控,`n_tx × n_rx × Q`。
- **半双工 / 单角色**:`enforce_single_role=True`(默认场景)时,一架 UAV 同帧不能既 TX 又 RX(可对多个目标同任一角色);一个目标可被多条链路服务,上限 `K_q_max=3`。

**贪心边际增益排序:**
```
ΔU_{ijq} = ω_q · [ U(D_q + d_eff) − U(D_q) ]      # 加权边际效用增益
```
每轮选增益最大且可行的候选;`marginal_utility_gain`(`math_utils.py`)给出该边际效用。**这是启发式贪心**:当前效用 `U(D)` 经验上非凹(边际增益不递减),所以**没有**子模/`(1−1/e)` 近似保证(见 `KNOWN_ISSUES B8`)。

**约束(每轮可行性检查):** 接收容量 `capacity_per_rx`(每选一条扣 `B_q`)、上报时延 `latency_max`(由 `reporting_rates` 推)、每目标基数 `K_q_max`、单角色一致性(若启用)。

**停止条件:** 无正边际增益(`best_gain ≤ 1e-12`),或所有候选均不可行/已选满。

**输出 `P0Solution`(`utils/types.py`):**
```python
P0Solution(
    z_selected:   np.ndarray,            # (K,K,Q) 0/1 选择张量
    D_q_star:     np.ndarray,            # (Q,) 每目标累积 deflection
    U_q:          np.ndarray,            # (Q,) 每目标效用
    selected_set: list[tuple[int,int,int]],  # 选中的 (tx, rx, target)
    total_bits:   float,                 # = len(selected)·B_q
    total_latency: float,
)
```
角色(`learn_roles=False` 时)由 `env_core` 从 `selected_set` 反推:出现在某选中三元组 i 位 → TX,j 位 → RX,否则 Idle。

---

## 7. Recurrent DAgger 变体训练 (D0/D1, 2026-07-14 v2)

两个 local-PD 输入模式的 recurrent DAgger 对照：

```bash
python scripts/train_dagger_variants.py --mode all \
    --config config/exp_800_q4.yaml \
    --dagger-iters 5 --teacher-eps 60 --student-eps 40 \
    --val-episodes 20 --test-episodes 100 \
    --out-dir results/dagger_variants
```

| Mode | PD_hist | Comm | 用途 |
|------|:---:|:---:|------|
| D0 | zeros | zeros | 无检测历史基线 |
| D1 | RX-only local | zeros | local confidence（通信推迟到 PPO） |

**协议要点 (v3, chunk BPTT)**：
- 数据保存为 episode 序列（不存储 h_prev）
- 训练：chunk-based TBPTT (chunk_size=16)
  - h=0 仅在 episode 边界
  - Chunk 内部：`detach_h_new=False`，梯度跨帧传播（最多 L=16 帧）
  - Chunk 边界：`h.detach()` 截断梯度；Optimizer step 在 episode 结束后执行一次
- 每轮报告 hidden drift 诊断值
- Validation (20 eps) 按 max weak3 (steady ≥ base-0.01) 选择 checkpoint
- Test (100 eps, 独立 bank)；ep_fail 主门限 τ=0.3
- D2 已移除：通信训练推迟到 PPO

**D0/D1 结果 (K=4,Q=4, seed=42, 2026-07-15)**：D0 (0.701/0.601), D1 (0.703/0.604), Δ<eval noise。D1 选为 PPO 初始化（接口一致性）。

**回归测试** (`tests/test_chunk_bptt_consistency.py`)：7/7 通过
1. Full-sequence == chunked 输出 (max|diff| < 1e-5)
2. Chunk 边界 carry state (carried ≠ reset)
3. Episode 边界 reset hidden (fresh ≠ leaked)
4. Actor h_new.grad_fn is None (内置 TBPTT detach)

**建议**：选 D1 作为 PPO 初始化——符合分布式信息结构，`pd_hist_proj` 从 DAgger 阶段参与训练。

---

## 8. Full/EH 长期对照实验 (2026-07-15)

### 实验设计

```bash
# Full PPO (Attention trainable)
python scripts/run_mappo.py \
  --config config/exp_800_q4_full.yaml \
  --warm-start results/dagger_variants/dagger_D1.pt \
  --warm-start-mode direct --seed 42

# EH PPO (Attention frozen)
python scripts/run_mappo.py \
  --config config/exp_800_q4_eh.yaml \
  --warm-start results/dagger_variants/dagger_D1.pt \
  --warm-start-mode direct --seed 42
```

| | Encoder LR | Attention LR | Head LR | Comm |
|---|---|---|---|---|
| **Full** | 1e-5 | 1e-5 | 5e-5 | off |
| **EH** | 1e-5 | **0** | 5e-5 | off |

唯一变量：Attention 是否参与训练。`learned_comm_mode='off'` 关闭通信消息、损失和 comm head 梯度。
配置中已设 `num_episodes: 300`，无需命令行 override。

### 3-Seed 结果 (2026-07-16)

| Seed | Full best_steady | EH best_steady |
|:---:|:---:|:---:|
| 42 | 0.5023 | 0.5035 |
| 123 | 0.4994 | 0.5046 |
| 456 | 0.5028 | 0.5024 |
| **Mean ± Std** | **0.5015 ± 0.0015** | **0.5035 ± 0.0009** |

Online weak3: Full 0.3255 ± 0.0053, EH 0.3280 ± 0.0010。
Δ(EH−Full) < 0.003 — 不可区分。

PPO 不再破坏 DAgger。Full 与 EH 在 3 seeds 下表现等价。
GRU/PPO 状态一致性 bug 是此前崩塌的根因，冻结 Attention 不必要。

## 9. PPO Ratio 重验证协议 (P0 fix, 2026-07-14)

修复 GRU/PPO 状态一致性后，所有基于 PPO 的实验结论需要用修复后代码重新验证。建议验证矩阵：

| Case | 目的 | 优先级 |
|------|------|:---:|
| DAgger frozen eval | 新 baseline（ratio 修复后） | P0 |
| Full PPO 1 update | 确认不再立即破坏 DAgger | P0 |
| Full PPO 20 updates | 验证长期训练不退化 | P0 |
| EH 1 update | 确认选择性可塑性有效 | P1 |
| EH 20 updates | 长期 EH 稳定性 | P1 |
| Per-module ablation (100ep) | 用正确 ratio 重新归因 | P2 |

每次训练启动时自动运行 `verify_old_log_prob_consistency()` — 若输出 `[PPO RATIO OK]` 则 ratio 正确；若输出 `[PPO RATIO ERROR]` 则立即停止并排查。

快速验证命令：
```bash
python scripts/test_ppo_ratio_fix.py  # DAgger → 1 PPO update → 对比
```

### 严格配对实验要求 (2026-07-14)

任何比较 "PPO 是否破坏 DAgger" 的实验必须满足：

1. **Checkpoint 兼容**：加载旧 DAgger 后调用 `zero_init_new_layers()`，保证新增层权重为 0（不改变策略）。
2. **独立 trainer**：每个 case（Full/EH/E-only/H-only）使用全新 env + agents + trainer，不从其他 case 复用。
3. **同一快照**：所有 case 从相同的 actor + critic + optimizer state 恢复。
4. **Streaming GRU 评估**：`_evaluate()` 维护跨帧 GRU hidden state，与 rollout 的 recurrent 路径一致。
5. **Ratio 断言**：每个 case 的首次 PPO update 自动运行 `verify_old_log_prob_consistency()`，输出 `[PPO RATIO OK]` 或 `[PPO RATIO ERROR]`。
6. **同一 rollout seed**：所有 case 使用相同的环境种子，消除 rollout 方差。
7. **同一 test bank**：所有 case 在相同的 eval seeds 上评估。

违反以上任何一条，比较结果不可信。
