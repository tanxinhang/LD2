# 实验:配置、基线、指标、复现

---

## 1. 配置管理与优先级

参数加载顺序(后者覆盖前者):

```text
dataclass 默认值 (config/params.py)
   ↓ 被覆盖
YAML 配置        (config/default.yaml 或 --config 指定的文件)
   ↓ 被覆盖
脚本内显式赋值    (如 run_experiments 里 cfg.marl.oracle_obs = False)
   ↓ 被覆盖
命令行参数        (run_experiments 的 --config / --seeds / --results)
```

- **单一真源**:`config/default.yaml` 是默认场景(400×400, Q=2),被 `tests/` 依赖,**不要随意改它**。
- `_dict_to_dataclass`(`params.py`)只映射 dataclass 里有的字段,YAML 多余键被忽略、缺失键用默认值——所以新增配置项必须先加进 dataclass。
- `get_default_config()` 读 `default.yaml`;`load_config(path)` 读任意 YAML。
- **实验场景** `config/exp_800_q4.yaml`:800×800、Q=4、`learn_roles=false`、`omega_q` 四等分,其余同默认。用于正式训练(default 场景过易,见 §3)。

### 影响网络维度的配置(改了必须重训)

`scenario.K`、`scenario.Q`(决定 `obs_dim=8+10Q+4(K-1)`、`global_state_dim=8K+7Q`)、`marl.hidden_layers`、`marl.learn_roles`(决定是否计角色 log-prob,且改变 env 角色逻辑)、`centralized_critic`(改 critic 输入维度)。

### 仅影响训练/评估、不改维度

`lr/gamma/gae_lambda/ppo_clip/ppo_epochs/minibatch/entropy_*/lambda_report/eta_mc/lagrangian_*/target_kl/num_episodes/num_envs/rollout_steps`、`eval_*`、`early_stop_*`、`eval_seeds`。

> 建议:每次实验把完整配置快照写入 `results/run_xxx/config.yaml`(当前 `run_experiments` 把结果写 `results/experiments.json`,可扩展为按 run 存目录)。

---

## 2. 关键参数表(来源与作用)

| 参数 | 默认值 | 单位 | 作用 | 来源/性质 |
|---|---|---|---|---|
| `region_size` | [400,400](exp:800) | m | 区域;相对感知半径决定任务难度 | 仿真设定 |
| `height` | 20 | m | 双基地几何(固定高度) | 仿真设定 |
| `K` | 4 | — | UAV 数 | 场景 |
| `Q` | 2(exp:4) | — | 目标数 | 场景 |
| `T` | 150 | 帧 | 回合长度(2.5×150=375 m 位移预算) | 场景 |
| `dt` | 0.1 | s | 帧长 | 场景 |
| `v_max` | 25 | m/s | 最大速度 → `max_dp=2.5 m/帧` | 平台约束 |
| `P_sense` | 0.0251 | W | 感知功率(14 dBm) | 文献锚定(TVT 2024) |
| `fc` | 2.8e10 | Hz | 路损/多普勒/波长/天线增益(**实际参与**计算) | 场景(28 GHz) |
| `n_cpi` | 128 | — | 相干积累 → 感知半径 ~120 m | 系统配置 |
| `M,N` | 64,16 | — | OTFS DD 网格(进 deflection) | 系统配置 |
| `P_FA` | 1e-3 | — | 虚警率 → P_D 映射 | 算法设定 |
| `P_D_min` | 0.2 | — | 公平性下限 | 算法设定(0.8 不可达会致 Lagrangian 爆) |
| `B_max` | 50000 | J | 能量预算 | 仿真设定 |
| `B_q` | 64 | bits | 每条上报软信息位 | 系统配置 |
| `capacity_per_rx` | 256 | bits/帧 | 接收容量约束 | 系统配置 |
| `K_q_max` | 3 | — | 每目标最大配对数 | 算法设定 |
| `learn_roles` | false | — | 角色是否由策略学(否=P0 分配) | 算法设定 |

> 28 GHz 确实进入路径损耗、多普勒、波长、天线增益与 OTFS 网格的解析计算(`geometry.py`/`deflection.py`/`otfs.py`),但不生成时域波形(见 `SYSTEM_MODEL §0`)。

---

## 3. 基线、当前结果与可信度

非学习基线——**统一命名**(代码里 `run_experiments` 的 key 为 `Random / P0-Fixed / Greedy-Approach`,headroom 扫描脚本里另有 `stationary`,下表给出规范名以免混用):

| 规范名 | 含义 | 代码对应 |
|---|---|---|
| **Stationary-P0** | UAV 不移动,只跑 P0 | headroom 脚本 `stationary` |
| **Random** | 随机位移 + P0 | `make_random_fn` |
| **P0-Fixed** | UAV 固定环绕队形,位置不学,仅靠 P0 配对 | `make_p0fixed_fn` |
| **Greedy-Oracle** | 用**真实目标位置**朝目标飞 → 特权上界(= 旧文中 "Greedy-Approach"/"full-greedy") | `make_greedy_fn` |
| **Greedy-Belief** | 朝 **belief** 均值飞(部署条件下界对照,尚未实现) | 待补 |

> "full-greedy" 是临时叫法,正式文档/图表请统一用 **Greedy-Oracle**。

> **整体定性:目前的"结果"主要是环境审计结论,不是最终学习算法结果。** 已能证明:环境可运行、角色冲突修复有效、场景存在 headroom、基线有显著差异。**尚不能**证明:MAPPO 优于 Random、MAPPO 优于 IPPO、CTDE 有效、belief 输入足够、训练稳定、方法可发表——这些都要等正式 5-seed 训练(见下)。

已确认结果:
- **角色修复有效**:`full_greedy` 确定性评估 `steady_P_D` 0.027 → 0.977,`valid_pair_rate` 0.04 → 1.00,`all_same_role` 0.955 → 0.000(四模式全部 ~0.92–0.98)。**可信**(env 侧实测)。
- **默认 400×400 场景过易**:stationary 0.976 / random 0.977 / greedy 1.000,gap 仅 0.02 → **策略无可学空间**,在此训练几乎必得 MAPPO≈Random 的 null result。**可信**(env 侧实测)。
- **P0 贪心经验最优性(B8)**:`scripts/greedy_gap_audit.py` 在 300 帧(K=3,Q=2)上,**greedy == 穷举最优占 100.0%,relative gap=0.0000**。当前效用虽非凹(无理论次模保证),贪心**经验达最优**——可作论文证据。穷举受 ≤20 候选限制,故缩小场景且未含单角色约束。
- **Oracle vs Deployable(B6/B7,机制已验证,数值待训练)**:env 侧随机策略——oracle `steady_P_D`≈0.156 ≥ deployable(belief 排序)≈0.145(方向正确,真值排序是上界)。随机策略下差距小(目标慢、belief≈真值),**训练后定位能力上来差距会放大**;检测门控影响在噪声内(P_D 实现仍用真值几何)。
- **场景 headroom 扫描**(单变量 area;列名用规范基线名):

| area | Q | Stationary-P0 | Random | Greedy-Oracle | gap |
|---|---|---|---|---|---|
| 400 | 2 | 0.976 | 0.977 | 1.000 | 0.02(平凡) |
| 800 | 2 | 0.221 | 0.210 | 0.875 | 0.66 |
| 800 | 4 | 0.197 | 0.175 | 0.727 | 0.55(**exp 选用**) |
| 1200 | 2 | 0.042 | 0.037 | 0.703 | 0.66 |
| 1600 | 2 | 0.008 | 0.007 | 0.403 | 0.40(过难,真位置都够不到) |

**待诊断/未完成**:
- `exp_800_q4` 的正式 5-seed MAPPO/IPPO 训练**尚未跑**(env 侧已验证 random≈0.18/greedy≈0.73、Q=4 维度正确、无角色冲突;但 `learn_roles=False` 下的训练梯度回路未在 GPU 上跑过完整 run)。结果出来前,**学习策略的最终性能不可下结论**。

---

## 4. 复现:种子与随机性

`set_seed(seed)`(`utils/seeding.py`)统一控制:Python `random`、`numpy` 全局、`torch` CPU、`torch.cuda`(并设 `cudnn.deterministic=True`)。

环境内部另有独立 RNG:
- `UAVISACEnv` 持 `self.rng = default_rng(seed)`,`reset(seed)` 会**新建**该 RNG。
- 目标运动、belief 噪声共享 env 的 `core.rng`。
- **`deflection_computer` 持有一条独立 RNG 流**(Rician/LoS 衰落),`reset` 替换 `core.rng` 后它仍指向原始 generator —— 见 `KNOWN_ISSUES`。状态快照 `get_state/set_state` 已对所有独立 RNG 流分别快照。

固定场景复现:评估用固定 `eval_seeds`(默认 `[10001..10005]`),每次评估、每个 decode 模式都重放同一批场景;主实验种子 `seeds=[42,123,456,789,1024]`。
精确逐帧重放:用 `env.get_state()` 取快照,`env.set_state(snap)` 还原(已验证 8 帧逐位一致、快照可重复使用)。

---

## 5. 运行命令与预期产物

```bash
# 单训练(默认场景)
python scripts/run_mappo.py

# 基线对照
python scripts/run_baselines.py        # 打印 Random/P0-Fixed/Greedy 的 steady_P_D 等

# 正式主实验(推荐场景)— 先冒烟后全量
python scripts/run_experiments.py --config config/exp_800_q4.yaml --seeds 1
python scripts/run_experiments.py --config config/exp_800_q4.yaml --seeds 5
#   产出: results/experiments.json(增量、可断点续跑);
#   末尾打印 Random/P0-Fixed/IPPO/MAPPO/Greedy 的 mean±std 及 MAPPO vs IPPO 配对 t 检验

# Oracle 上界 vs Deployable 方法对照(B6/B7;同一场景,改 P0 信息源)
python scripts/run_experiments.py --config config/exp_800_q4.yaml --seeds 5                      # ORACLE 上界(默认)
python scripts/run_experiments.py --config config/exp_800_q4.yaml --seeds 5 --p0-belief --detect-sample  # DEPLOYABLE 闭环
#   建议两者写到不同 --results 路径再对比,例如 --results results/exp_oracle.json / results/exp_deploy.json

# P0 贪心 vs 穷举最优 审计(B8,无需 GPU)
python scripts/greedy_gap_audit.py --frames 300 --region 800

# 环境/物理完整性诊断(无需 GPU)
python scripts/deep_audit_sim.py        # 多 probe 量化:动作保真、角色抖动、配对可行性…

# 测试
pytest tests/ -q
```

冒烟训练观察三点(`exp_800_q4`):① `eval_steady_P_D` 从 ~0.16 向 Greedy-Oracle ~0.73 爬;② `valid_pair_rate` 全程 ~1.0;③ entropy 不立即塌到 0、critic loss 收敛。若 MAPPO 最终 ≈ Random(~0.16),回查 obs/reward,而非接受结果。

---

## 6. 通信成本与检测—通信权衡(说明)

当前奖励里通信项 `−λ_report·total_bits`,`λ_report=1e-5`、`B_q=64`,每条链路成本 ≈ `6.4e-4`,而检测效用典型 0.3–3 —— **通信成本在奖励中近乎可忽略**。真正的通信约束是 **`capacity_per_rx=256 bits/帧` 的硬上限**(P0 可行性检查)。

因此当前实现里:**通信容量是硬约束,`λ_report` 只是弱正则,项目并未重点优化 bits–detection 的 Pareto 权衡**。若论文要主打"检测—通信开销权衡",需要:(a) 提高/扫描 `λ_report` 做敏感性实验,或 (b) 收紧 `capacity_per_rx` 让容量约束真正绑定。相关技术债务见 `KNOWN_ISSUES B10`。

---

## 7. 复现产物建议

当前 `run_experiments` 把聚合结果写 `results/experiments.json`。建议每次正式 run 额外保存完整配置快照,便于复现:`results/run_<tag>/config.yaml`(可在 `run_experiments.main` 里把 `cfg` dump 出来)。这样他人仅凭该目录即可重建场景与超参。
