# 多 UAV 协同 ISAC 序贯感知调度 — 项目文档

> 本 `docs/` 目录的目标:让**没看过代码的人,仅靠文档就能理解、运行、检查、继续修改本项目**。
> 阅读顺序建议:`README` → `SYSTEM_MODEL` → `ARCHITECTURE` → `TRAINING` → `EXPERIMENTS` → `KNOWN_ISSUES`。

---

## 1. 项目摘要(1 页)

**研究问题。** 在一个二维区域内,用 `K` 架无人机(UAV)对 `Q` 个匀速运动目标做**双基地(bistatic)协同感知**:每架 UAV 既可作为发射机(TX)也可作为接收机(RX),一对 (TX, RX) 对某个目标形成一条双基地观测链路。系统要在**通信容量、能量、安全间距、检测公平性**等约束下,通过控制 UAV 轨迹和感知配对,最大化各目标的检测概率 `P_D`。

**规模(默认配置 `config/default.yaml`)。** `K=4` UAV,`Q=2` 目标,区域 `400×400 m`,飞行高度 `20 m`,每回合 `T=150` 帧,帧长 `dt=0.1 s`。载频 28 GHz。在当前整套射频/检测/CPI 参数组合下有效感知距离约 120 m(受发射功率、噪声、RCS、天线增益、载频、双基地距离、DD 有效性、P_D/P_FA 阈值共同决定;`n_cpi=128` 是其中重要的增益来源之一,但非唯一决定因素)。

**输入 / 决策 / 内层优化 / 输出。**

```text
输入:   UAV 初始状态、目标运动状态、信道/OTFS 参数、能量与通信约束
决策:   每架 UAV 的二维位移 Δp;(可选)TX/RX/Idle 角色 —— 见下方"完成程度"
内层:   给定几何,P0 贪心选择双基地 (tx, rx, target) 配对,受容量/时延/基数约束
输出:   每目标 P_D、通信开销(bits)、能耗、公平性、约束违反率、团队/个体奖励
```

**核心算法。** 外层 MAPPO / IPPO(多智能体 PPO,CTDE)学习 UAV 轨迹;内层 P0 **启发式贪心**做感知配对(基于边际检测效用;注意当前效用非凹,**无**次模/近似保证,见 `SYSTEM_MODEL §3.4`);Kalman(CV 模型)做目标 belief 预测与更新。

**项目定位(重要,避免误读)。** 这是一个**几何驱动 + 链路级抽象的研究原型仿真**,**不是**波形级 OTFS 仿真。感知质量用解析的 "deflection"(检测统计量)→`P_D` 映射建模,没有原始 IQ 波形、没有真实调制解调、没有飞控硬件接口。边界详见 `SYSTEM_MODEL.md §0`。

**当前完成程度(统一表述,避免歧义)。**
- 环境、物理/几何/检测/belief/约束/奖励、P0 与角色派生逻辑:**已完成并通过环境侧验证**(单元测试)。
- MAPPO/IPPO 训练代码:**已实现**。
- `learn_roles=False` + D1 direct warm-start 的端到端 Full/EH 训练已完成 3-seed × 300 episodes（PPO ratio 正确，策略未崩塌，Full≈EH）。

**当前主要结果**：
- 角色修复后确定性评估不再崩塌（0.027→0.977）。
- DAgger D1 warm-start: steady=0.70 (100-ep test)。
- **Full/EH 3-seed 长期对照**：PPO 不再破坏 DAgger。Full 与 EH 不可区分（Δ<0.005）。GRU/PPO 状态一致性 bug 是此前崩塌的根因。
- **S4 target-wise advantage**：distance-responsibility 实现并完成 3-seed × 50-ep 测试。未通过稳定性检验（seed 456 退化），scalar advantage 保留为当前稳定主线。搁置 S4，优先推进 IPPO baseline 和 K=8/Q=8 可扩展性。

**当前已知问题(详见 `KNOWN_ISSUES.md`)。** 已修复:GRU/PPO 循环状态一致性(P0,2026-07-14)、Attention 冻结补全(P0)、Q 硬编码移除(P0)、PD_hist 接入 Actor(P1)、Per-target GAE 管道(P2)、角色 argmax 崩溃、动作存储/执行一致性、critic value-clip、优势重复归一化、奖励权重、状态快照漏 RNG 流。开放(影响科学可信度):**P0 用目标真值(oracle 调度)**、**belief 选中即成功观测(乐观)**、**效用非凹 → 贪心无近似保证**、动作投影概率密度未严格建模、二值 Lagrangian、角色标量序数编码。Scalar MAPPO 为当前稳定主线（3-seed 长期稳定性已验证）。

**2026-07-14 重要更正**:此前所有 PPO 训练结果均在 GRU/PPO 状态不一致导致的非法 PPO ratio 下测得。修复后验证:1 次 PPO 更新不再破坏 DAgger 策略(Δweak3 < 0.005)。

**2026-07-16 Full/EH 3-seed 长期对照**:300 episodes × 3 seeds。PPO 不再破坏 DAgger。Full 与 EH 不可区分（Δbest_steady=+0.002）。GRU/PPO 状态一致性 bug 是此前崩塌的根因，冻结 Attention 不必要。详见 `KNOWN_ISSUES.md`。

---

## 2. 安装

代码用 Python + PyTorch。推荐用项目自带虚拟环境或新建一个:

```bash
# 关键依赖(见 requirements.txt)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

主要依赖:`numpy`, `scipy`, `torch`, `gymnasium`, `matplotlib`, `pyyaml`, `pytest`。
GPU 非必需但训练强烈建议(`torch.cuda` 自动检测,无 GPU 退回 CPU)。

---

## 3. 快速运行

```bash
# 单次训练(默认 400×400 场景,seed=42)
python scripts/run_mappo.py

# K=8, Q=8 大场景训练 (2026-07-14 新增)
python scripts/run_mappo.py --config config/exp_800_k8_q8.yaml

# 非学习基线对照(Random / P0-Fixed / Greedy)
python scripts/run_baselines.py

# PPO ratio 修复验证 (DAgger → 1 PPO update → 对比)
python scripts/test_ppo_ratio_fix.py

# Recurrent DAgger 变体训练 (D0/D1 local-PD 对照)
python scripts/train_dagger_variants.py --mode all
# 单独训练某一变体:
python scripts/train_dagger_variants.py --mode D1 --dagger-iters 5

# 正式多 seed 主实验(MAPPO vs IPPO + 基线),推荐用有 headroom 的场景:
python scripts/run_experiments.py --config config/exp_800_q4.yaml --seeds 5
# 先冒烟一遍确认梯度路径正常:
python scripts/run_experiments.py --config config/exp_800_q4.yaml --seeds 1

# 环境/物理完整性诊断
python scripts/deep_audit_sim.py

# 测试
pytest tests/ -q
```

运行命令、预期产物、复现细节见 `EXPERIMENTS.md`。

---

## 4. 目录速览

```text
config/        参数(default.yaml 为单一真源;exp_800_q4.yaml 为训练用大场景)
uav_isac/
  environment/ env_core(一帧主循环)、env_wrapper(Gym 接口)、action、observation、
               reward、constraints、belief、target、uav
  physical/    geometry、deflection、inner_solver(P0)、detection、channel、otfs
  agents/      mappo_agent、networks、buffer、trainer、myopic/p0_fixed/base agent
  utils/       math_utils、seeding、types
scripts/       run_mappo / run_ippo / run_baselines / run_experiments / deep_audit_sim
tests/         单元 + 集成 + 完整性审计测试
docs/          本文档
```

模块职责、调用关系、一帧数据流见 `ARCHITECTURE.md`。

---

## 5. 文档能否回答这 10 个问题(自检索引)

| # | 问题 | 去哪看 |
|---|------|--------|
| 1 | Actor 控制什么? | `ARCHITECTURE §2`,`SYSTEM_MODEL §3.3` |
| 2 | P0 控制什么? | `TRAINING §6`(P0 求解器) |
| 3 | 每帧先移动 UAV 还是先算感知? | `ARCHITECTURE §3`(帧序 + 陷阱) |
| 4 | `obs_dim` 每一维是什么? | `ARCHITECTURE §4.1` |
| 5 | 训练动作 vs 评估动作? | `TRAINING §5`(train/eval 差异表) |
| 6 | `P_D` 如何从几何/deflection 得到? | `SYSTEM_MODEL §3.4–3.5` |
| 7 | 奖励每项的数值范围? | `TRAINING §4`(奖励分解) |
| 8 | 为什么某帧没有有效 TX–RX 配对? | `KNOWN_ISSUES`(角色崩溃)+ `TRAINING §6` |
| 9 | 如何用完全相同的随机场景复现? | `EXPERIMENTS §4`(种子与复现) |
| 10 | 当前结果哪些可信、哪些在诊断? | `KNOWN_ISSUES` + `EXPERIMENTS §3` |
