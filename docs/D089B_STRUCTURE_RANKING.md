# D0.89-B：功率无关结构排序 + 瓶颈对偶价格（负结果，含可操作洞察）

> 状态：实现完成，3-seed 冒烟完成；20-seed 因 MILP 候选集扩张而显著变慢。
> 实现：`config/params.py`（`analytical_structure_ranking_enabled`）、
> `env_core.py`（unit-power 排名 + 滞后 λ* 优先级）。
> 结论：**功能正确但未改善 worst，且 λ* 硬瓶颈权重导致 steady 塌缩；全图 MILP
> 在功率无关排名下候选集扩张 ~3–4x 变慢。**

## 1. 实现

D0.89-B 做了两件事（均 config 门控，默认关闭）：

1. **功率无关排名（unit-power）**：`analytical_structure_ranking_enabled=True` 时，
   P0 的 deflection 以单位感知功率计算，`d_eff = a_ijq`（每瓦增益），打破
   "structure ← power ← structure" 循环依赖。
2. **λ* 瓶颈优先级**：上一帧 max-min LP 的最优对偶价格 `λ*` 作为 `target_priority`
   （替换 deficit EMA），让结构选择的次要词典序项聚焦瓶颈目标。

## 2. 结果（8/8 test20 前 3 seed，冻结 Actor，同 seed 配对）

| 配置 | worst | steady | weak3 | QoS 可行率 | 功率平衡 |
|---|---:|---:|---:|---:|---:|
| deployed（无 LP） | 0.2774 | 0.7467 | 0.4543 | 0.333 | 3.3e-16 |
| A（powered 排名 + LP） | **0.6688** | **0.7379** | **0.6876** | 0.667 | 4.4e-16 |
| B（unit-power 排名 + λ* + LP） | 0.6382 | 0.6404 | 0.6382 | 0.667 | 3.3e-16 |

**B 相对 A：worst −0.031、steady −0.098、weak3 −0.049，QoS 可行率持平。**

## 3. 判定（诚实负结果）

1. **功率无关排名不改善 worst**：A（powered 排名 + LP）在 worst 上反而优于 B
   （0.6688 vs 0.6382）。原因：冻结 Actor 的感知功率近似均匀，`d_eff ≈ a_ijq/8`，
   powered 排名与 unit-power 排名结构高度相似；差异不足以带来增益。
2. **λ* 硬瓶颈权重过强**：`λ*` 支撑在单一最弱目标上，作为次要词典序项
   （权重 0.10）仍把结构选择**过度**偏向瓶颈，导致 steady 从 0.738 塌到 0.640。
   deficit EMA（`exp(gain·(floor−P_D))`）是**软**权重（覆盖所有低于地板目标），
   反而更稳。**"功率无关 ≠ 公平性无关"是对的，但 λ* 作为结构权重需要软化**
   （如 softmax(λ*/τ) 或 λ* 与 deficit EMA 的凸组合），不能直接上硬支撑。
3. **性能**：unit-power 排名把 MILP 候选集从"已激励边"扩张到"全部 DD 可行边"
   （K=8,Q=8 约 448 条），P0 求解慢 ~3–4x。这再次印证 D0.11 的结论：全图 MILP
   是教师/参考，不是可部署协调器；部署必须走局部候选图（Gate A2 的 `L_u=4,
   L_q=4`）或分布式列生成，而不是每帧全图 MILP。

## 4. 对下一步的指引

- **B 的"功率无关"方向保留**，但**结构求解器**必须换成局部候选图（复用 Gate A2
  的 `L_u/L_q` 邻域）或有限轮分布式协调，不能每帧全图 MILP。
- **结构优先级保持 deficit EMA 或软 λ\***：硬 λ* 支撑过强。若要用对偶，用
  `softmax(λ*/τ)` 或 `λ* ⊙ deficit` 的组合，避免单一瓶颈吸走全部结构资源。
- **先做 C（reserve-first 重训）** 比继续调 B 更划算：A 已拿到 +0.2264 worst，
  C 的 reserve-first 直接解决 steady 张力（A 已暴露 steady 0.773→0.692）。

## 5. 验证状态

- `tests/test_analytical_sensing_power.py`：7 passed（含 B 的 flag 校验、env 集成、
  滞后 λ* 为有效单纯形、功率平衡 `<1e-12`）。
