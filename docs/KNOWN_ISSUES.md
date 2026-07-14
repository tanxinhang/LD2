# 已知问题与技术债务

> **2026-07-14 基础设施审计闭环。** 截至 commit `c77f3e0`，recurrent PPO rollout–buffer–update 条件一致性、streaming GRU 评估、动态 K/Q、per-target GAE bootstrap、RX-only 局部检测置信度边界均已通过自动化回归测试（8 个测试文件，全部硬断言）。此前基于非法 PPO ratio 和全局 P_D 历史输入得到的训练退化与模块归因结果不再作为有效证据。后续实验统一基于修复后的 local-PD 代码路径。

统一格式:问题 / 现象 / 原因 / 状态 / 诊断方法 / 优先级 / 相关文件。
**已修复**与**开放**分列。

---

## DAgger 变体对照 (D0/D1, recurrent protocol, 2026-07-14)

对 local-PD 进行了 recurrent DAgger 训练对照（`scripts/train_dagger_variants.py`，K=4,Q=4）：

**协议**：student rollout 使用 streaming GRU；监督训练存储并传入 per-frame `h_prev`；validation (20 eps) 选择 checkpoint；test (100 eps) 独立报告。ep_fail 主门限 τ=0.3。

**限制**：单一 training seed (seed=42)；per-frame stored h_prev（非 chunk BPTT）；通信训练推迟到 PPO 阶段（D2 已移除）。

| Variant | PD_hist | Comm | steady | weak3 | ep_fail_030 | ep_fail_005 |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| D0 | zeros | zeros | — | — | — | — |
| D1 | RX-only | zeros | — | — | — | — |

**结论**：local-PD 对 nearest-target DAgger 无可测量增益（nearest-target teacher 基于真实目标位置，不依赖检测历史）。local-PD 的价值应在 PPO fine-tuning + target-wise advantage 阶段检验。建议选 D1（RX-only local-PD）作为 PPO 初始化——符合分布式信息结构，`pd_hist_proj` 从 DAgger 阶段参与训练。

**复现产物**：`results/dagger_variants/` 含 `run_manifest.json`、`test_episodes_D{0,1}.csv`、`val_history_D{0,1}.csv`、checkpoint `.pt` 文件。

---

## 训练崩塌归因(诊断闭环,2026-06; 2026-07-14 更正)

**现象**:MAPPO 训练后 eval `steady_P_D`≈0.018、`avg_P_D`≈0.12(**低于随机 0.16、低于静止 0.20**);`entropy` Ep50 塌到下限、`kl`→0(策略冻结);训练日志显示 `avg_P_D` 从 Ep0 的 0.21 随熵塌缩**一路下降**。无论 400/800、oracle/belief 都同样崩。

### 根因 (2026-07-14 更正)

**P0 级 bug：GRU/PPO 循环状态不一致导致 PPO ratio 非法。** Rollout 时 Actor 接收持续维护的 GRU hidden state `h_{t-1}`，但 PPO 更新时 `evaluate_actions()` 调用 `actor(obs)` 没有传入对应的 `h_prev`。更新阶段 GRU 从零初始化，导致：

- rollout: π(a_t | o_t, h_{t-1})
- update:  π(a_t | o_t, h=0)
- PPO ratio r_t ≠ 1 在第一次 optimizer step 之前就已失效

这可以解释此前观察到的所有现象：DAgger 本身很好但一次 PPO 就破坏 weak3；降 LR/clip/KL 都无法保护；模块归因不稳定、符号翻转。

**2026-07-14 已修复**（commit `7993770`），修复后验证：1 次 PPO 更新不再破坏 DAgger（Δweak3 < 0.005）。

### 逐项排除(均有证据)
- ❌ 角色 argmax 崩溃 → 已修(A1)。
- ❌ 确定性评估 bug → 已修(P0 诊断)。
- ❌ P0 次模/最优性 → 贪心经验=穷举最优 100%(B8 审计)。
- ❌ critic 欠拟合 → 日志 `val` Ep50 即追上 `ret` 且同量级,`critic_loss` 1627→~0 单调收敛。
- ❌ 观测编码/表达力 → **关键证据**:一个"只从 obs 向量读目标+homing"的策略达 **0.71(oracle)/0.72(belief)**,映射近乎线性 → obs 充分、好策略可表达。
- ❌ belief 质量 → belief-obs homing = 0.72,与 oracle 几乎一致。
- ❌ BC≈random 不能证明"学不到":`sup_mse` 收敛到 0.15 却 eval≈random,是**开环 BC 分布偏移**(covariate shift)的标准特征,不是编码问题。
- ✅ **根因 (2026-07-14)**: **GRU/PPO 循环状态不一致** → PPO ratio 非法 → 任何 PPO 更新都等价于对随机扰动后的分布做梯度步。

**结论(已闭环,2026-07-14 更正)**:这是**GRU/PPO 状态一致性问题**,不是探索/优化失败,不是 obs/critic/表达力不足。修复后 PPO 不再立即破坏 DAgger。长程导航的探索困难可能仍存在,但不是崩塌的主因。

**对应工具/修法(已交付,待 GPU 验证)**:
- `scripts/dagger_warmstart.py`:DAgger 克隆可达 obs 的 nearest teacher(天花板 0.504)进 actor,修分布偏移,存 `results/warmstart_actor.pt`。
- `scripts/run_mappo.py --warm-start <pt>`:用克隆权重初始化 actor 再 PPO。
- **决定性判据**:warm-start 后 PPO **保住 ~0.5** → 探索是唯一问题(warm-start 即解);**摧毁→随机** → PPO 更新本身有问题(再查优势/critic/LR)。
- 备选:`use_distance_shaping=True`(势函数稠密梯度,角色 bug 已修后可重测)。
- 后续:要冲 0.72(index teacher)需给 actor 观测加 agent-id(当前 actor 观测无身份,只有 critic 有)——见 [[B9]] 同源。

---

## A. 已修复

### A0. GRU/PPO 循环状态不一致 → PPO ratio 非法 🔴 P0 (2026-07-14)
- **现象**:DAgger 策略本身很好 (steady 0.65-0.74)，但一次 PPO 更新立即破坏 weak3（下降 ~26%）。降 LR、加 KL anchor、clip 都无法保护。模块归因结果不稳定、符号翻转。
- **原因**:Rollout 时 Actor 使用持续维护的 GRU hidden state `h_{t-1}`，但 PPO 更新 `evaluate_actions()` 调用 `actor(obs)` 时没有传入 `h_prev`。更新阶段 GRU 从零初始化 → rollout 和 update 的 log-prob 条件不同 → PPO ratio r_t ≠ 1 在第一次 optimizer step 之前就已失效。
- **状态**:已修复。Buffer 存储每帧 GRU hidden state `(K, K-1, D)`，PPO 更新时从 buffer 读取并传入 `evaluate_actions()`。新增 `verify_old_log_prob_consistency()` 断言，训练开始时自动检查 `max|old_lp - recomputed_lp| < 1e-4`。
- **修复后验证**:1 次 Full PPO update: Δsteady=+0.002, Δweak3=+0.004, Δworst=-0.006。PPO 不再立即破坏 DAgger。
- **相关文件**:`buffer.py`, `mappo_agent.py`, `trainer.py`, `networks.py`。测试脚本:`scripts/test_ppo_ratio_fix.py`。

### A0b. Attention 冻结遗漏 attn_norm (2026-07-14)
- **现象**:文档声称 "Attention 冻结成功"，但 `attn_norm.weight/bias` 不以 `attn.` 开头，未被 `requires_grad_(False)` 覆盖。
- **状态**:已修复。`networks.py` 新增 `split_param_groups()` 显式列出三个参数组（`ATTENTION_PARAM_PREFIXES = ('attn.', 'attn_norm.')`, `ENCODER_PARAM_PREFIXES`, `HEAD_PARAM_PREFIXES`）。`trainer.py` 的 freeze 逻辑改用显式分组。
- **相关文件**:`networks.py`, `trainer.py`。

### A0c. Q=8 硬编码移除 (2026-07-14)
- **现象**:`buffer.py` 写死 `self.num_targets = 8`，与正式配置 K=4,Q=4 不一致；`action.py` 同样硬编码。per-target 存储/GAE 在 Q≠8 时 shape 不匹配。
- **状态**:已修复。`num_targets` 从 config 全链路传入：`config.scenario.Q → MAPPOAgent(num_targets=Q) → RolloutBuffer(num_targets=Q) → CriticNetwork(num_targets=Q) → buffer per_target 存储`。新增 `config/exp_800_k8_q8.yaml` 明确绑定 8×8 实验配置。
- **相关文件**:`buffer.py`, `mappo_agent.py`, `trainer.py`, `action.py`, 各 `run_*.py`, `config/exp_800_k8_q8.yaml`。

### A0d. PD_hist 接入 Actor target encoding (2026-07-14)
- **现象**:`StructuredActorNetwork._parse_one()` 读取 `_pd_hist`（上一帧每目标 P_D）后直接丢弃，Actor 无法直接知道哪些目标上一帧检测概率低。
- **状态**:已修复。`pd_hist` 从 `_parse_one` 返回，在 `forward()` 中通过新增 `pd_hist_proj` 层投影后作为残差调制加到 target entity encoding。备注了通信假设（当前 PD_hist 在本地 obs 中，若改为融合中心广播需计入通信成本）。
- **相关文件**:`networks.py`, `observation.py`。

### A0e. Per-target GAE 数据管道就绪 (2026-07-14)
- **状态**:已完成。`buffer.compute_gae()` 新增 per-target GAE 计算（per-target TD error → per-target advantage）；`get_training_data()` 返回 `per_target_advantages` 和 `per_target_returns`。**Actor loss 仍使用 scalar advantage**，S3c target-wise advantage 集成待后续完成。
- **相关文件**:`buffer.py`, `mappo_agent.py`, `trainer.py`。

### A0f. 验证框架严格化：checkpoint 兼容 + streaming 评估 + 配对实验 (2026-07-14 晚)
- **现象**:第一轮验证脚本存在四个问题：(1) EH 复用 Full PPO 的 trainer 状态而非独立创建；(2) 旧 DAgger checkpoint 加载时 `pd_hist_proj` 层随机初始化，baseline actor 与 trainer actor 可能得到不同随机权重；(3) 评估调用 `actor(obs)` 无 GRU hidden state，与 rollout 的 recurrent 路径不一致；(4) Full 和 EH 的初始状态不完全相同。
- **状态**:已修复。
  - `zero_init_new_layers(known_keys)`:加载旧 checkpoint 后将新增层权重和 bias 置零，保证 `e_{kq}=e_{kq}^{base}`，严格复现原 DAgger。
  - `_evaluate()`:维护 streaming GRU hidden state，每帧传入 `actor(obs, h_prev)` 并保存 `h_new`，评估路径与 rollout 一致。
  - `test_ppo_ratio_fix.py` v2:每个 case 独立创建 fresh env/agents/trainer，从同一 snapshot (actor+critic+optimizer) 恢复，同一 rollout seed 和 test bank。
  - **修复后 strict paired 结果 (streaming GRU, 3 seeds)**:DAgger baseline steady=0.693, weak3=0.283; Full PPO Δweak3=-0.020; EH PPO Δweak3=-0.003。两项 PPO RATIO OK (max|diff|<1e-5)。
- **相关文件**:`networks.py` (zero_init_new_layers), `trainer.py` (_evaluate streaming GRU, bootstrap state fix), `scripts/test_ppo_ratio_fix.py`。

### A0g. Per-target GAE bootstrap 修复 (2026-07-14 晚)
- **现象**:per-target GAE 在 rollout 末尾（即使 episode 未终止）将 bootstrap 值硬编码为 0，即 `V_q(s_{t+1})=0`。只有真实 episode 终止时才应为 0。
- **状态**:已修复。`compute_gae()` 接受 `next_per_target_values` 参数，非终止截断使用 critic 的 `V_q(s_{t+1})` bootstrap；trainer 在 rollout 结束时用正确的 final obs + final GRU hidden state 计算 scalar 和 per-target next values。
- **相关文件**:`buffer.py`, `trainer.py`。

### A0h. 其他硬编码与状态一致性问题 (2026-07-14 晚)
- **CVaR top-k**:`cvar_k = 2` → `max(1, int(ceil(0.25 * Q)))`，Q=4→1, Q=8→2。
- **single_dim=227**:替换为 ObservationBuilder 提供的 `single_frame_dim`，自动适配不同 K/Q/P0 配置。
- **Bootstrap stale obs**:rollout 结束时不再复用 `_obs_gpu`（可能包含旧 obs），改为从 `all_obs` 重建 final obs batch，并提取 env 中当前 GRU hidden state 做正确的 final actor forward。
- **相关文件**:`trainer.py`, `networks.py`, `observation.py`, `mappo_agent.py`。

### A1. 确定性评估角色 argmax → 全同角色 → 零配对 → P_D 崩塌
- **现象**:训练采样性能尚可,但确定性评估 `steady_P_D` 极低(C 模式 0.027);`all_same_role` 95.5%,`valid_pair_rate` 0.04。
- **原因**:共享 actor 在 argmax 下把几乎所有 agent 映射到同一角色 → 没有 TX/RX 分裂 → P0 无候选 → P_D=0。随机采样靠熵偶尔凑出分裂掩盖了问题。`i≠j` 只排除自配对,**不**保证一 TX 一 RX。
- **状态**:已修复。`marl.learn_roles=False`(默认)下移除策略角色头出目标,改由 P0 分配:deflection role-agnostic 枚举所有 `i≠j`,inner_solver 加每 UAV 单角色约束,角色从 `selected_set` 反推。
- **效果**:四模式 0.92–0.98,`valid_pair_rate`=1.00,`all_same_role`=0,角色冲突 0。
- **诊断方法**:四模式评估(`_evaluate_modes`)+ 固定场景重放。
- **相关文件**:`action.py`、`mappo_agent.py`、`env_core.py`、`physical/deflection.py`、`physical/geometry.py`、`physical/inner_solver.py`、`config/params.py`。

### A2. 存储动作 ≠ 执行动作(log-prob 数值不一致,旧 P1/P5)
- **现象**:`old_log_prob ≠ new_log_prob`,PPO ratio 失真。
- **原因**:动作经 tanh 压缩后被 env 的盒式裁剪改写,采样时与更新时重构不一致。
- **状态**:**部分修复**。decode 内对压缩动作做 ±0.999 预裁剪 + 圆盘投影,使 env 裁剪成 no-op;`compute_log_prob` 与 `evaluate_actions` 用同一 arctanh 重构路径,**保证了数值重算一致(`old_log_prob == new_log_prob`)**。
- **遗留**:这只解决了"代码前后一致",**没有**解决"投影后动作的真实概率密度"——圆盘投影是多对一映射,重算的 log-prob 不等于执行动作在投影分布下的真实密度。见开放问题 `B5`。
- **相关文件**:`action.py`、`mappo_agent.py`。

### A3. Critic value-clipping 致 critic 追不上 return
- **现象**:`val_mean << ret_mean`(差 ~70),优势变垃圾 → 策略崩。
- **原因**:value-clip 用 `ppo_clip=0.2` 在原始 return 尺度上限制每次更新的 value 变化量。
- **状态**:已修复,改纯 MSE,无 value clipping。
- **相关文件**:`trainer.py`。

### A4. 优势重复标准化
- **状态**:已修复。只在 `buffer.get_training_data` 全局标准化一次,minibatch 内不再重复。
- **相关文件**:`buffer.py`、`trainer.py`。

### A5. 奖励权重 / 约束惩罚调整
- **状态**:已处理。`lambda_report` 1e-3 → 1e-5(避免"少上报优于多检测");`constraint_penalty` 不再直接进奖励(env 传 0.0),改由 Lagrangian 以二值 `any_violation` 惩罚;`P_D_min` 0.8 → 0.2(0.8 不可达会致 Lagrangian 单调爆)。
- **相关文件**:`reward.py`、`env_core.py`、`config/default.yaml`、`trainer.py`。

### A6. 状态快照漏掉第二条 RNG 流
- **现象**:`get_state/set_state` 重放时,位置与 `core.rng` 都一致,但 `d_eff`/`P_D` 仍有 ~1e-4 偏差。
- **原因**:`reset(seed)` 替换 `core.rng` 为新对象,但 `deflection_computer` 在 `__init__` 持有的是**原始** RNG → Rician/LoS 衰落用的是一条独立、未被快照的流。
- **状态**:已修复。`get_state` 扫描所有组件去重收集每条 Generator,逐条按引用还原;targets/belief 重新指回 `core.rng`。已验证 8 帧逐位一致。
- **相关文件**:`env_core.py`(`_persistent_rngs/get_state/set_state`)。

---

## B. 开放

### B1. 默认场景过易,正式训练待跑
- **现象**:400×400/Q=2 下 stationary 0.976 ≈ random ≈ greedy → 策略无可学空间。
- **原因**:感知半径(~120 m)相对区域过大,初始摆位即覆盖目标。
- **当前状态**:已提供有 headroom 的 `config/exp_800_q4.yaml`(random≈0.18,greedy≈0.73);**正式 5-seed MAPPO/IPPO 训练尚未在 GPU 上跑**。
- **诊断方法**:area 单变量扫描(见 `EXPERIMENTS §3`);训练时盯 `eval_steady_P_D` 是否从 ~0.16 向 ~0.73 爬。
- **优先级**:P0(决定论文主结论)。
- **相关文件**:`config/exp_800_q4.yaml`、`scripts/run_experiments.py`。

### B2. `learn_roles=False` 训练梯度回路未端到端验证
- **现象**:env 侧(role-agnostic + P0 + 派生角色)已完整测试通过;但策略训练回路(collect/update/evaluate_actions 在 `learn_roles=False` 下)只做了代码审查 + `py_compile`,**未在装 torch 的机器上跑过完整 step**。
- **风险**:角色项被一致剔除、张量形状未变,理论上低风险;仍需一次冒烟确认 loss/熵/eval 正常。
- **诊断方法**:`run_experiments --config config/exp_800_q4.yaml --seeds 1` 冒烟。
- **优先级**:P0。
- **相关文件**:`mappo_agent.py`、`action.py`、`trainer.py`。

### B3. obs 的"上一帧 P_D"存在一帧额外滞后(off-by-one)
- **现象**:`next_obs` 在 `_build_observations()` 时用的 `self.prev_P_D` 是上一帧的 P_D;当前帧 `P_D_q` 在其后才写入 `self.prev_P_D`。即用于决定 `a_{t+1}` 的 obs 携带 `P_D_{t-1}` 而非 `P_D_t`。
- **影响**:检测状态特征比实际多滞后一帧;不致命,但与直觉不符。
- **修复方向**:在 `_build_observations()` 之前更新 `self.prev_P_D`(注意同步评估/复现基线)。
- **优先级**:P2(需确认是否为有意设计)。
- **相关文件**:`env_core.py`(step 步骤 12–13)。

### B5. 动作径向投影后的概率密度未严格建模 [科学正确性]
- **现象**:`old_log_prob == new_log_prob` 只证明代码重算一致,不证明投影后动作分布的真实 log-prob 正确。
- **原因**:`Δp ← (max_dp/‖Δp‖)·Δp` 是多对一映射(圆盘外同方向多点映到同一圆周点)。`dp_scale=max_dp` 时,tanh 压缩动作落在 `[-0.999,0.999]²` 盒内,其中约 21%(盒角减去内切圆)的区域 `‖Δp‖>max_dp` 会触发投影 → 触发并不罕见。重算密度与真实执行动作密度存在偏差。
- **修复方向**:改成光滑可逆的径向变换,如 `Δp = max_dp·(tanh ρ /(ρ+ε))·u`(`u=z`,`ρ=‖z‖`);或极坐标动作 `v=σ(z_v), θ=π·tanh(z_θ), Δp=max_dp·v·[cosθ,sinθ]`,使变换可逆、Jacobian 可写。
- **优先级**:P0(正式训练前至少明确列为限制)。
- **相关文件**:`action.py`、`mappo_agent.py`。

### B6. P0 / deflection 使用目标真值(oracle 内层调度) [科学正确性 — 已实现两模式,待训练验证]
- **现象**:`env_core.step` 默认把 `get_position_3d()`(**真实**目标位置/速度)传给 P0 用于排序选择。
- **影响**:内层配对相当于 oracle;部署场景不具备同等信息。
- **状态**:**已实现 deployable 模式(flag,默认 oracle 不变)**。`marl.p0_uses_belief=True` 时:P0 在**融合 belief**(各 UAV belief 均值)几何上排序/选择,而被选配对的**实现 deflection/P_D 仍用真值几何**(物理回波);二者分离实现于 `env_core.step`(两遍 deflection)。env 侧已验证 **oracle ≥ deployable**(真值排序是上界),角色一致性保持。
- **用法**:oracle 上界 = 默认;deployable 方法 = `run_experiments.py --config config/exp_800_q4.yaml --p0-belief`。
- **待办**:GPU 上跑 oracle vs deployable 5-seed 对照(随机策略下差距小,因目标慢且 belief≈真值;训练后定位能力上来差距会放大)。融合规则当前用**均匀均值**,可换信息(协方差)加权。
- **相关文件**:`env_core.py`(step 步骤 3–5)、`config/params.py`、`scripts/run_experiments.py`。

### B7. belief 更新"选中即成功观测"(与 P_D 脱钩) [科学正确性 — 已实现 Bernoulli 门控,待训练验证]
- **现象**:默认对每个 P0 选中的 `(i,j,q)` 一律按成功观测更新,**与 `P_D` 无关**。
- **状态**:**已实现检测门控(flag,默认 off=乐观)**。`marl.belief_detection_sampling=True` 时,每目标采样 `δ_q~Bernoulli(P_D,q)`(用 `core.rng`,固定评估种子可复现);`δ_q=0` 则该帧不更新 belief、只做 CV 预测、AoI 继续增长。实现于 `env_core.step` 的 belief 更新循环。
- **用法**:`run_experiments.py --config config/exp_800_q4.yaml --p0-belief --detect-sample`(与 B6 合用即"部署闭环":坏 belief→坏配对→漏检→更坏 belief)。
- **待办**:训练验证其对收敛/AoI 的影响(随机策略下对 steady_P_D 影响在噪声内,因 P_D 实现仍用真值几何,门控只改 belief 质量这一二阶通道)。备选更平滑方案:量测协方差随 deflection 加权(B7-B,未实现)。
- **相关文件**:`env_core.py`(step belief 更新)、`config/params.py`。

### B8. 效用非凹 → P0 无次模/近似保证 [科学正确性 — 已更正表述 + 已用经验最优性 gap 替代理论保证]
- **核对结果**:`U(P_D)=−log(1−P_D)` 关于 `P_D` **严格凸**;复合后 `U(D)` **单调递增但非凹**(数值 `U''>0` 占 ~99.6%),固定 `d_eff` 边际增益随 `D` **递增**。故**不存在**子模/`(1−1/e)` 理论保证。
- **状态**:**已处理(方案 A)**。(1) 所有 docs 与代码 docstring 已删除"凹/次模/近似保证"措辞,改述为启发式;(2) 用已有 `solve_exhaustive`/`compute_greedy_gap` 写了经验审计 `scripts/greedy_gap_audit.py`。**结果:在 300 帧(K=3,Q=2)上 greedy == 穷举最优的比例 = 100.0%,relative gap = 0.0000** —— 即虽无理论保证,**贪心经验上达到最优**,可作论文证据替代次模性声明。
- **审计局限**:穷举受 `≤20 候选`限制,故在缩小场景(K=3,Q=2,12 候选)上做;且穷举未含"单角色"约束(只 capacity+cardinality)。结构性结论对 K,Q 不敏感,具指示性。
- **可选(方案 B,未做)**:换饱和效用 `U(D)=1−exp(−κD)` 恢复理论保证,但需重调奖励、破坏可比性。
- **相关文件**:`scripts/greedy_gap_audit.py`、`utils/math_utils.py`、`physical/inner_solver.py`、`detection.py`(docstring 已更正)。

### B9. 角色标量编码引入人为序数关系
- **现象**:局部 obs 里角色用单标量 `0=TX,1=RX,2=Idle`,未归一化、非 one-hot,网络会隐式认为 `TX<RX<Idle` 有序。
- **修复方向**:改 one-hot(3 维)或 learned embedding;`learn_roles=False` 且角色只作历史信息时,可用 `is_tx/is_rx/is_idle` 三个二值量。
- **优先级**:P1(非致命)。
- **相关文件**:`observation.py`。

### B10. 单一二值 Lagrangian 无法区分约束类型/严重程度 [技术债务]
- **现象**:安全/能量/公平/边界违反都映射到同一个 `any_violation∈{0,1}`,共用一个 `λ_lag`;轻微越界与严重碰撞同权。
- **影响**:学不到有意义的约束权衡;此外通信成本 `λ_report=1e-5` 过小近乎可忽略(见 `EXPERIMENTS §6`)。
- **修复方向**:分离的连续违反量 + 各自乘子 `r_k^aug = r_k − Σ_m λ_m·g̃_m`(`constraints.py` 已分项算出各 penalty,可直接接入)。
- **优先级**:P2。
- **相关文件**:`trainer.py`、`constraints.py`、`env_core.py`。

### B11. 角色 head 仍构建但不参与目标(`learn_roles=False`)
- **现象**:为降低 blast-radius,`learn_roles=False` 时 `ActorNetwork.role_head` 仍存在,只是被排除出 log-prob/熵且 env 忽略其输出 → 该 head 不接收梯度(死参数)。
- **影响**:少量无用参数与前向计算;功能上等价于"删除角色头"。
- **修复方向**:确认新模式训练健康后,可把 role_head 真正从 `networks.py` 移除(需同步 `act/act_batch/evaluate_actions/collect_rollout/_evaluate` 的 `role_logits` 为 None 的处理)。
- **优先级**:P2。
- **相关文件**:`networks.py`、`mappo_agent.py`、`trainer.py`。

---

## C. 测试覆盖(`tests/`)与诊断脚本

- **单元**:动作范围/投影、log-prob 一致性、Kalman 维度、P0 合法性、reward 分解、约束计算、几何、deflection、检测、种子可复现。
- **集成/完整性**:`test_integrity_audit.py`(需 torch)审计端到端不变量。
- **数值**:`P_D∈[0,1]`、deflection 非负、协方差、无 NaN/Inf 等散布在各测试中。
- **诊断脚本**:`scripts/deep_audit_sim.py` 用多个 probe 量化(动作保真度、角色抖动、双基地配对可行性、log-prob 一致性、obs 范围等),输出带 ⚠ 的可疑点。

运行:`pytest tests/ -q`(`test_integrity_audit.py` 需要 torch;无 torch 环境可 `--ignore=tests/test_integrity_audit.py`,其余 46 项可跑)。

---

## 指标正常范围速查(日志排查)

| 指标 | 正常 | 异常 |
|---|---|---|
| actor_loss | 接近 0 有波动 | 长期严格 0 |
| critic_loss | 初期大、逐步降 | 持续爆炸 |
| entropy | 缓慢下降 | 几轮内塌到 0 |
| approx_kl | 在 target_kl(0.02)附近 | 长期 0 或突然很大(触发早停) |
| valid_pair_rate | 接近 1(P0 分配下应 ~1) | 大量空配对 |
| eval/train gap | 有限 | 相差一个数量级(查 train/eval 差异表) |
| λ_lag | 稳定在 [0,1] | 总顶到 1(违反率持续 > 目标) |
