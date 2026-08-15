# D0.89-C：reserve-first t* 奖励 + outer 重训（训练管线已通，正式训练待跑）

> 状态：奖励接线 + reserve 不可行回退 + 训练管线冒烟完成；正式多-episode 训练与
> 2×2 归因是剩余的长跑实验。
> 实现：`reward.py`（`utility_mode="tstar"`）、`env_core.py`（reserve 不可行回退）、
> `config/exp_800_k8q8_architecture_v2_analytical_power_d089c.yaml`。

## 1. 奖励设计（层级目标，不是加权 penalty）

按 "reserve feasibility ≻ t* ≻ comm cost" 的层级定义，检测效用项直接取 LP 值：

```text
r_team = min_q P_D(D*_q)  − 通信成本 − 约束惩罚
```

其中 `D*_q` 是 reserve-first max-min LP 的最优 deflection（`analytical_sensing_power`
打开后 `detection_D_q == LP deflection`，故 `min_q P_D == P_D(t*)`）。reserve 约束
`Σ_i a_iq p_iq ≥ r_q`（`r_q` 由 `reserve_pd` 经检测函数反解）在 LP 内强制执行，所以
steady 地板**进入可行域**而不是被当作一个大权重 penalty 重新调。

实现：`RewardComputer` 新增 `utility_mode="tstar"`，返回 `min_q P_D(D_q)`。

## 2. reserve 不可行回退

reserve 超过可达天花板时，精确 LP 会不可行。回退策略：退回纯 max-min LP，记录
`reserve_shortfall`，奖励照常（worst 低于地板自然反映在 `min_q P_D` 里），不崩溃、
不掩盖。

## 3. 训练冒烟结果（5 episodes，8/8，warm-start 冻结 Actor）

| 指标 | Ep 0 | Ep 4 |
|---|---:|---:|
| actor LR | 3.0e-5 | 3.0e-5 |
| PPO ratio | max diff 8e-6 | — |
| approx KL | 0.0010 | 0.0006 |
| reward | 0.080 | 0.136 |
| t* utility（worst P_D） | 0.292 | 0.324 |
| avg P_D | 0.292 | 0.324 |

**管线判定：**
1. **lr 已恢复**：继承的冻结教师链 `lr: 0.0` 已在 C 配置覆盖为 `3e-5`（对齐
   Architecture V2 训练约定），actor 实际在学（`actor_loss≠0`）。
2. **PPO 稳定**：ratio 合法、KL 受控（~1e-3），说明 t* 奖励（非光滑的 `min_q P_D`）
   没有立即破坏训练。
3. **方向正确**：worst P_D（t*）从 0.292 缓慢上升到 0.324，与部署 worst ~0.35 同
   量级，方向一致但收敛慢——这正是"learner 只负责外层变量"的预期（LP 已做重活）。

## 4. 下一步：正式训练 + 2×2 归因

| Case | sensing power | outer retrain | reward |
|---|---|---|---|
| C0 | old head | No | log |
| C1 | exact LP | No | log（=D0.89-A） |
| C2 | exact LP | Yes | log |
| **C3** | exact LP | **Yes** | **tstar（reserve-first）** |

- `C1−C0` = 解析优化器贡献（已得 +0.2264 worst，D0.89-A）；
- `C2−C1` = 缩小动作空间后的学习贡献；
- `C3−C2` = reward 对齐贡献（tstar vs log）。

正式训练需数百 episode（Architecture V2 历史用 300），且需 4 个 case 各跑多 seed；
这是长跑实验，管线已打通，可直接 `run_mappo.py --config ... --warm-start
best_restored.pt --episodes 300` 起。

## 5. 验证状态

- `tests/test_maxmin_reward.py` 新增 tstar 单测 + env 集成（reserve_pd=0.6）：
  18 passed（含 analytical power 全套）。
- reserve 不可行回退在 env 集成中被隐式覆盖（部分几何 reserve=0.6 不可行时回退）。
