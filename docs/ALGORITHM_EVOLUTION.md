# 算法演进（更新至 2026-08-20）

> 本文记录在深度审计修正之后、以部署候选（lexicographic L1 + 多候选 trust-region
> L3）为基线实施的理论驱动优化。每条记录含：理论推导 → 实现 → 配对评估 → 结论。
> 原则：尊重数理常识、遵守通信（Shannon/1 W/时延）与感知（双基地/检测/DD 门）
> 基本原则；优化必须包含创新点而非仅参数切换；负结果如实记录。

本文是算法演进的唯一活动文档。它记录“为什么改、改了什么、理论依据、验证结果、
最终去留”；逐次实验数字与产物路径统一放在 `EXPERIMENT_LOG.md`，当前数学定义和执行
算法统一放在 `CURRENT_SYSTEM_MODEL.md`。

## 总体演进图

```text
学习式动作/角色/P0 启发式
  └─ Architecture V2：集合等变 Actor + 结构 Student + 物理 Token
      └─ D0.87–D0.89：固定结构解析 max-min 功率，删除不必要的学习功率自由度
          └─ D0.91–D0.93：能力上界、议价目标、capability gauge/PWL 证书
              └─ D0.94–D0.95：KKT 几何敏感度 + L2/L3 交替控制
                  └─ D1.1：QoS 字典序 L1 + 多候选/前瞻 L3
                      └─ D1.7–D1.10：分布式列生成、blind 认证与尾部审计
                          └─ V3 / advice-016：dual-priced L2、task-regret 与最小通信信息
                              └─ G2-0：RF 上限/量纲/规范对偶闭合；post-G2 等待重认证
```

## G2-0：基础科学闭合（2026-08-20，当前）

本轮不以调参提高旧指标，而是先消除会使任何优化结论失真的四个基础问题。

1. **功率是上限，不是物理等式。** 当前约束为
   `P_comm,k+Σ_q P_sense,kq≤1 W`。普通非负增益、无功率惩罚的 max-min 存在用满
   预算的最优解，故“fill”只是一种 tie-break；隐蔽/暴露约束绑定时可严格少发。
   环境现在分别记录 cap violation 与 unused power，并按最终执行功率重算电池。
2. **Deflection 量纲闭合。** 旧式 `P|α|²T_symMN/P_noise` 的单位是秒。当前采用
   `E_signal/E_noise`：分母为 `P_noise·T_sym`，故 `T_sym` 两侧抵消，raw deflection
   无量纲。未改射频参数，但绝对数值尺度改变，因此所有旧 Gate 冻结为 pre-G2 历史证据。
3. **明确 LP 的适用前提。** 只有固定结构、固定 DD active set/CSI，且 per-watt
   系数与本轮功率无关（条件可分离证据）时，L1 才是 LP；有 sensing interference
   或功率相关支撑时不能沿用强对偶声明。
4. **稳定学习标签与通信证书。** 非唯一最优对偶通过 `min ||λ||²₂, λ∈Λ*` 规范化；
   决策 bit 证书使用 `Δ_eff=Δ−2(E_stale+E_phys)`，`Δ_eff≤0` 时返回不可认证并
   fail-closed，而非把退化值误当成 1-bit 证书。
5. **修正 task-regret 训练语义。** 旧实现用 teacher-selected mask 计算所谓 Student
   regret，无法惩罚 Student 自己造成的结构翻转；现改为 Student logits 的 softmax
   可微结构解码。teacher gain 也从 `comm_fraction` 恢复真实剩余瓦数，不再把归一化
   `sensing_weights` 误当实际功率。

本轮理论主线凝练为：

```text
capability margin → structure regret → decision-preserving communication
                  → closed-loop confidence
```

**验证**：新增隐蔽约束必须少发的反例、RF cap/unused 指标测试、J/J 量纲测试、规范
对偶对称性/置换等变测试，以及 stale+physical error 的 fail-closed 测试。性能层必须
在 post-G2 同 seed、独立 env、预注册协议下重新跑，旧数值不得用于新模型选参。

## G2-1A 基础设施与 CIS-ISAC 路线冻结（2026-08-20）

新审计将规模化主线收敛为 CIS-ISAC：Exact L0/L1 保持不动，未来在 L1 后用 Robust
Task Slack 判断是否需要 L2，再用始终包含当前结构/No-op 的嵌套 active set 逐级扩展。
其原则是任务性能 fail-closed、计算复杂度 fail-open；物理 upper bound 在随机量无确定
上界时仅作 shadow ranking。Student 保持 proposer，L3 V1 不做 active-mover 重构。

严格 Gate 顺序仍是 G2-1A→G2-1B→S0→S1A shadow→S1B shadow→S1C live；因此本轮
没有把 Scaling Guard 接入环境。新增 `tools/run_g2_1a_bridge.py` 自动核查四系统各自的
100 个历史 seed、checkpoint、`n_CPI=1`、25.1 mW sensing cap、1 W RF cap，并禁止
task-regret checkpoint。底层逐 seed 隔离器支持 resume，固定单 worker 避免 Windows/MKL
与单 GPU 并发污染。

同时修复一个统计错误：旧 `_merge_rows` 对 episode 数组虽会拼接，但标量 steady/weak3/
worst/QoS/LCB 会沿用首个完成 seed。现在标量从合并后的数组重算，Wilson 使用 z=1.96，
并按 seed 排序保证确定性。当前只完成 4/4 2 seeds、8/8 1 seed、6/6 两臂各 1 seed 的
runner 验证，不称 G2-1A 完成。

## G2-0.7：联合 RF 上限不覆盖感知 PA 上限（2026-08-20）

G2-1A 初始 smoke 暴露出 8/8 每帧总感知功率 `7.9917 W`、6/6 `5.9989 W`，即解析
L1 几乎给每架 UAV 分配整瓦感知功率；而 `P_sense=0.0251 W` 原本是 14 dBm 感知
波形锚点。约束 `P_comm+Σ_q p_kq≤1 W` 只是联合 RF 上限，不能推出感知功放允许
`Σ_q p_kq≈1 W`。

因此新增独立硬约束 `Σ_q p_kq≤P_sense,max=0.0251 W`，L1 实际预算变为
`b_k=min(1 W-P_comm,k,P_sense,max)`。这是两个硬件可行域的交集，剩余 RF 功率允许不用，
不再通过“等式平衡”强迫进入感知。相同历史 seed 的 smoke 变为：4/4 worst `0.887`；
8/8 V3-C0 worst `0.0914`；6/6 V3-C0 worst `0.1376`；6/6 multi-scale CE worst
`0.3965`。它恢复了结构/几何区分度，也显示 CE 相对同 seed V3-C0 的方向性优势；单 seed
只用于 falsification，不产生性能结论。

## G2-0.6：从“空余时间”到“已执行证据”（2026-08-20）

旧 `n_CPI=128` 的问题不只在于 `128·N·T_sym>dt`，还在于执行器每个控制动作只生成
一份 OTFS 观测。时间窗最多容纳 97 帧，并不意味着系统已经获得 97 份独立噪声样本或
维持了 97 帧相干相位。因此定义 `L_time=floor(dt/(N T_sym))=97`、`L_exec=1`，采用
`L_eff=min(L_config,L_exec,L_time)=1`。

同时把能量推导写成 `D=c_det E_s/N0`，其中 `E_s=P_r N T_sym L_eff`、
`N0=P_noise/B`、`B=M/T_sym`，故 `D=c_det(P_r/P_noise)MN L_eff`。这既保留正确的
时宽积，也不把 `M` 个并行子载波误写成额外发射时长。

审计结果变为 100/300/500/800 m 的 `P_D=1.000/0.989/0.124/0.00981`；
`T_F=1.024 ms<100 ms`，全距离饱和和时间超限均消失，`--assert-ready` 返回成功。
该改变不是为恢复旧性能而调参，而是删除未执行的 phantom looks。未来多帧路线必须在
“独立 LLR 相加”或“相干和 + 残余 Doppler/相位损失”二者中明确选一并重新认证。

## G2-0.5：检测归一化与 processing-gain falsification（2026-08-20）

**统计推导**：固定实高斯充分统计量 `H0:N(0,1)`、`H1:N(sqrt(D),1)`，由 NP 阈值
直接得到 `D=(μ1-μ0)²/σ0²` 与 `P_D=Q(Q^{-1}(P_FA)-sqrt(D))`。这选择了实等效
matched-filter convention，故 `c_det=1`；复噪声能量 convention 可能出现的因子 2
不再隐含混用。50 万样本、`D={5,10,15,20}` 的经验 PFA/PD/Deflection 全部通过
解析对照。

**低成本 falsification**：100/300/500/800 m 对称双基地 link-budget 表把增益拆成
`P_r/P_n → ×MN → ×n_CPI`。`MN` 后的 P_D 依次约为
`1.000/0.989/0.124/0.0098`，具有距离区分度；再乘 `n_CPI=128` 后全部
`>0.9999999`。同时，若 128 个 looks 是完整 OTFS frames，积分时长
`128·N·T_sym=0.131072 s` 超过 `dt=0.1 s`。

**判定**：G2-0.5 只闭合了 `E_s/E_n ↔ D ↔ ROC` 的常数和统计假设，尚未证明
`n_CPI` 是帧内可实现、相互独立或相干的 128 倍增益。该反例阻断 G2-1A/1B；不得
通过立即跑 100 seeds 或重新调距离/RCS 来掩盖。下一步 G2-0.6 必须从观测时长、
相干时间、目标运动/相位补偿和独立 look 数推导有效 processing gain，再做认证。

之后统计协议固定为：旧 100 seeds 仅作 G2-1A paired bridge；冻结后另取未查看 bank
作 G2-1B blind confirmation。4/4、8/8 V3-C0、6/6 multi-scale CE 同时评估，且
在结果分叉前 task-regret 保持 OFF。

阻断条件同时被实现为 `tools/audit_detector_normalization.py --assert-ready`：若 ROC
不吻合、审计距离全饱和或声明 CPI 超出控制帧，命令非零退出。当前两个物理阻塞项
触发退出码 2，因此流程状态不是“待人工确认”，而是“不可进入 G2-1”。

## D0.87–D0.95：解析控制栈形成

| 阶段 | 核心改变 | 数理依据 | 结论 |
|---|---|---|---|
| D0.87 | 将固定结构下的感知功率改为 max-min LP | 线性预算约束与瓶颈公平 | 正向，进入快层 |
| D0.88 | 功率 warm-start 与 hold-H | LP 值函数扰动界；DD 门跨界不连续 | 理论界有效但偏松 |
| D0.89-A | 删除学习 sensing-power 自由度 | 给定结构时解析 LP 优于无约束网络输出 | 保留为当前 L1 基础 |
| D0.89-B | 功率无关结构排序 | 瓶颈对偶价格 | 负结果，关闭 |
| D0.89-C | reserve-first t-star 奖励 | feasibility 优先的层级目标 | 仅管线验证，训练路线关闭 |
| D0.91 | steady/worst 同几何上界 | relaxed capability ceiling | 证明物理资源可达，控制器未达 |
| D0.92 | 参考点归一化议价 LP | Kalai--Smorodinsky 机会公平 | 理论候选，不是默认 L1 |
| D0.93 | capability gauge 与双侧 PWL | 单调值函数、chord/tangent sandwich | 作为路由/证书保留 |
| D0.93-F | Shannon 通信余量回收 | 正交链路容量反解 | 负结果：该场景通信 slack 非主因 |
| D0.94 | capability-sensitivity 几何步 | KKT/对偶次梯度 + finite difference | 进入慢层候选 |
| D0.95 | L2 结构与 L3 几何交替 | block-coordinate improvement | 形成解析部署栈 |

这组演进最重要的纠错是：不能把局部 LP 改善直接等同于闭环 episode 改善。结构提交会
改变观察、通信和冻结 Student 的后续决策，因此后续所有 live 修改都必须同时验证提交帧
物理单调性和跨时域闭环结果。

## Architecture V2：从固定身份网络到集合式部署

早期网络依赖固定 `K/Q`、集中式 P0 角色分配和不完整的通信语义。Architecture V2 改为：

- 共享参数、UAV/目标集合编码和置换等变 critic；
- 结构 Student 学习集中式教师，但部署输入只使用本地观察和真实到达 Token；
- Token 显式经历量化、容量、时延、丢包和功率约束；
- owner/端点/目标结构保留可解释字段，避免自由 Token 退化为身份标签；
- 跨基数只允许在独立 Gate 通过后升级，不把参数共享当作零样本泛化证明。

该阶段先后否决了 persistent-intent admission、普通模块化协调、QPD-ISAC、简单有限轮
价格协商和多种因子图/排序器方案。它们的共同问题不是网络不够深，而是局部代理目标与
真实双端点检测能力、资源耦合及闭环时域目标不一致。

## 当前算法选择

当前默认保留：L0 解析通信资源、QoS 字典序 L1、P0/Student 结构路径、多候选/前瞻
L3、严格 1 W 预算、DD 支撑和环境级检测融合。V3 dual-priced L2、拥塞 relief、
task-regret Student 校准均保持默认关闭，直到独立 seed 上同时满足三项感知地板、通信
bit/时延/功率和 Wilson LCB。

算法演进遵循四条硬规则：

1. 先证明单位、定义域、预算和可行域正确，再比较性能。
2. 解析层只对其真实凸子问题声称最优；结构与几何层不借用不存在的全局最优保证。
3. 开发 seed 只能生成假设，不能再次充当确认集。
4. 点估计通过但置信下界未过时，状态必须是 `DISCLOSED FAIL`。

## 部署候选基线（优化前）

| 配置 | worst | weak3 | steady | QoS |
|---|---:|---:|---:|---:|
| D0.95 解析栈（gauge L1） | 0.662 | 0.724 | 0.808 | 1.0（tol=1e-6） |
| lexicographic L1（D1.1-A） | 0.844 | 0.844 | 0.845 | 1.0 |
| **lex + 多候选 L3（D1.1-B）** | **0.975** | 0.978 | 0.982 | **1.0**（LCB 0.839） |

## D1.1-B+：对偶上界剪枝（正结果）

**理论**：固定结构 max-min LP 的对偶 `min_{λ∈Δ_Q} Σ_i b_i max_q λ_q a_iq`；对任意可行
单纯形价格（当前帧最优 λ*），弱对偶给出候选几何的 max-min deflection 上界
`U_λ(g') = Σ_i b_i·max_q(λ*_q·a'_iq)`；P_D 是 deflection 的严格单调函数，故
`U_λ ≤ best_deflection ⟹ 候选不可能胜出 ⟹ 精确剪枝`（跳过无望的 LP 评估）。

**实现**：`_select_best_movement_candidate` 每帧用 λ* 计算 O(KQ) 上界，剪掉被支配
候选；配置 `analytical_movement_dual_prune`（默认 True）。

**验证**：弱对偶数值（20 组随机场景 U_λ ≥ t* 恒成立）；剪枝前后选择逐位一致
（2 seed × 30 帧）；LP 调用减少。**结论：0.975 性能零损失，计算成本下降。**

## D1.1-B++：gauge 价格步候选（负结果）

**理论**：包络定理 `∂γ*/∂a_iq = −π_q·p_iq` 给出 gauge 对偶 π（三地板完整影子价），
推广 D0.95"L3 用 λ*"为候选竞争（λ* 步 vs π 步）。

**验证**：端到端 2 seed × 40 帧 mean gain ≈ 0。**理论解释**：lex 模式下 max-min
均衡化使 `weak3==steady==worst=t*`，λ* 支撑覆盖全部目标（D1_1A §6.2），π 与 λ*
信息冗余。**结论：不启用**（默认 False，保留为选项）。

## D1.1-B+++：lex 评分一致实验（负结果，确认分层设计）

**理论假设**：多候选 L3 用纯 max-min 评分、执行 lex Stage-B——"评分-执行不一致"
是缺陷，应改为一致。

**验证**：改用 Stage-B 评分后**严重退化**（seed 503 final 0.996→0.662）。**理论
解释**：L3 的职责是引导几何，纯 max-min 评分在 t* 之上持续提供改进梯度；Stage-B
在地板绑定处 t* 钉死、评分对移动失去区分度 → 几何停滞。**"不一致"是分层设计的
正确结构**（D0.95"指标选择价格"的延伸）。**结论：默认 False，确认 D1.1-B 设计。**

## D1.1-C：最优正交带宽分配（no-waste 保证）

**理论**：L0 最小通信功率 `f_i(B) = N0·B·max(γ_th, 2^(r_i/B)−1)/g_i` 是 B 的凸递减
函数；平分带宽在速率接近容量时多收高负载发送者。最优分配
`min Σ f_i(B_i) s.t. ΣB_i = B` 由 KKT `f_i'(B_i) = −λ` 双层二分精确求解；每个发送者
仍满足 SNR/速率 → 总功率严格不大于平分（no-waste）。

**验证**：数值实验——容量受限时节省 14%（10 kbps）/90%（100 kbps）/99.8%（单重
负载）；当前 8/8 配置通信功率 ~1e-5 W（SNR 主导区）收益≈0 但严格不差。**结论：
默认 True**（效率保证，容量受限负载下显著）。

## D1.1-D：T3 隐蔽性约束进入 live 功率路径

**理论**（advice 012）：对方检测是 Gaussian-shift 检验，`D_q^I = Σ_i a^I[i,q] p_iq ≤
D̄^I = [Q⁻¹(P_FA^I) − Q⁻¹(ε)]²` 等价于 `P_D^I ≤ ε`；凸 LP 硬约束，对偶 μ = 对方探测
价格。

**实现**：`uav_isac/coordination/intercept_power.py`（提取自 T3 oracle）+
`env_core._solve_intercept_power`（`intercept_constrained_power_enabled`）。

**验证**：P_D^I ≤ ε 硬约束执行（medium/strong）、1 W 预算保持、strong 对手 QoS
塌缩（<0.30）——"约束通过 QoS 咬合而非不可行性"。

## D1.1-E：QoS 地板 × 隐蔽性联合 LP

**理论**：两组硬约束（QoS 地板 + 反检测界）都是线性的，同一凸 LP 可联合；
`qos_constrained_maxmin_lp` 加 intercept 行（Stage-B 与第三层都施加），一个对偶三族
价格（λ 感知 / π QoS / μ 对方探测）。

**验证**：联合 LP 同时满足 worst P_D ≥ 0.60 与 P_D^I ≤ ε；无隐蔽性时 lex QoS ≥ 联合
（只减不增）。

## T3 live 端到端证据链（4 UAV × 2 seed × 30 帧，ε=0.1）

| 对手 | mean worst | P_D^I_max | 违反 |
|---|---:|---:|---:|
| off | 0.7033 | 0.0000 | 0 |
| weak | 0.7033 | 0.0012 | 0 |
| medium | 0.6634 | 0.1000 | 0 |
| strong | 0.0010 | 0.1000 | 0 |

P_D^I ≤ ε 每帧保持（跨运动/结构变化）；QoS 代价随对手强度单调——T3 oracle
"strong 对手必须静默"结论在 live 路径复现。

## D1.1-F：standoff 运动候选（负结果，无害保留）

**理论假设**：`a^I ~ 1/d²`，远离最弱目标放松反检测约束、允许更多功率。

**验证**：A/B 端到端无差异（medium 0.6634 / strong 0.0010）。**理论解释**：L3 评分
是纯 max-min（不感知隐蔽性），远离候选在该代理分数下从不胜出——评分-执行不一致
的"隐藏价值"面。**结论**：默认 True 但无害（超集 + stay 保留 + 对偶剪枝零成本），
仅在评分器变为隐蔽性感知时有价值。

## 最终 Gate 断言（`tools/assert_formal_gates.py`）

| 结果 | steady | weak3 | worst | QoS | LCB | 判定 |
|---|---:|---:|---:|---:|---:|---|
| 4/4 冻结部署版（100 seed） | 0.9132 | 0.8848 | 0.7393 | 0.72 | 0.625 | PASS |
| 8/8 D0.95 解析栈 | 0.8083 | 0.7239 | 0.6619 | 1.00 | 0.839 | PASS |
| 8/8 lex L1 | 0.8446 | 0.8437 | 0.8437 | 1.00 | 0.839 | PASS |
| **8/8 lex + 多候选 L3** | **0.9818** | **0.9778** | **0.9753** | **1.00** | **0.839** | **PASS（含 LCB 强制）** |
| 6/6（污染） | — | — | — | — | — | QUARANTINED |

## D1.7：λ-μ 分布式列生成的理论与数值验证（2026-08-16，advice 013）

**理论（Dantzig–Wolfe 精确分解）**：T3 的 DC-MM 内层 LP

```text
max t  s.t.  Σ_i a_iq p_iq ≥ t          （感知，跨 UAV 耦合）
             Σ_i a^I[i,q] p_iq ≤ D̄^I_q  （隐蔽性，跨 UAV 耦合）
             Σ_q p_iq ≤ b_i              （预算，每 UAV 可分离）
             p ≥ 0
```

把耦合行（感知+隐蔽性）放主问题（master），预算行放每 UAV 的定价子问题
（pricing）。给定主问题对偶价格（λ：感知瓶颈，μ：隐蔽性），UAV i 的定价
子问题恰为 advice 013 的本地 bid：

```text
q_i* = argmax_q (λ_q a_iq − μ_q a^I[i,q]),   p_iq = b_i 若 q = q_i*
```

列生成（RMP + 本地 bid 加列）收敛到中央 LP 的精确最优（LP-exact 分解）。

**数值验证**（20 个随机 4/4 场景，medium 对手 ε=0.1）：列生成与中央 LP 的
t* 差距全部 ≤ 3.6e-15（机器精度），平均 |gap| = 5.8e-16；收敛列数 10–16
（≈ K×Q 级）。**证明 T3 隐蔽性约束可分布式实现且数学精确**——advice 013
"剩余理论环"闭合。

**回归**：`tests/test_dw_column_generation.py`（列生成=中央 LP、本地 bid 解
pricing、隐蔽性约束成立）。下一步：实现完整分布式协调器（每帧 RMP + 价格
广播 + 本地 bid 执行）。

## D1.8：feasibility-aware L3 warm start（2026-08-16，advice 013 §6）

**问题**：帧 0 完全不动（`_last_deflection_entries` 在 reset 后为空，L3 hook 返回
`{}`），浪费滚动时域几何下降的第一帧——早期瞬态（7/20 seed 前 19 帧 worst<0.60）
因此多延 1 帧。

**理论（advice 013）**：初始位置 capability gauge `γ₀* > 1`（三地板初始不可行）
应触发 frame-0 几何修复，而非等滚动 deficit 出现。实现 `_initial_analytical_state`：
reset 后用**当前几何**预计算 deflection entries + 最小单 owner 结构（每目标 owner =
最近 UAV，TX = 最远 UAV 形成双基地基线），使 `_analytical_movement_delta` 帧 0
即进入 Phase-1 deficit 下降。

**验证**：帧 0 位移 0.00 → 7.50（4 UAV × 2.5m 步长内）；早期 worst 全程领先
~1 帧（seed 503：帧 0 worst 0.5029→0.5165）；3 项回归（帧 0 移动、结构可行性、
多 seed 不退化）。早期瞬态从"滚动收敛"变为"帧 0 即修复"。

## D1.5：blind certification 资产与协议（2026-08-16，advice 013 §1）

**动机**：部署候选（L0-KKT + Lex-L1 + P0-L2 + 多候选 L3）的 0.975 来自已多轮复用的
20 个 selection seed，不能作最终证据。D1.5 用**从未在任何运行中出现过**的 blind
seed 认证。

**资产**：
- `tools/generate_blind_seed_bank.py`：从 1130_k8q8 的 1000 个采样几何排除
  ①隔离种子 ②全部已暴露 seed（扫描 174 个 paired_eval.csv，457 个）③所有 bank
  split 种子——剩 249 个候选，tier 均衡（1:2:1）抽 100，冻结抽签种子 20260827。
- `config/stratified_seeds_1130_k8q8_blind.json`：100 blind seed，strict 加载通过，
  **训练池 522 个种子与盲测 0 重叠**（盲性保持）。
- `config/exp_800_k8q8_..._blind.yaml`：冻结候选 + blind bank + test split 固定 +
  禁用 confirmation。

**冒烟（10 blind seed）**：steady 0.907 / weak3 0.900 / worst 0.900 / QoS 0.90——
盲测 seed 上性能保持（worst 0.90 vs 开发 selection 的 0.975 同量级）；Wilson LCB
0.596（10 seed 分辨率不足，正是 N=100 的必要性）。

**全量 100 seed 盲测（`results/_d1_5_blind100/paired_eval.csv`，~7.4 h CPU）**：

| 指标 | 值 | 判定 |
|---|---:|---|
| QoS feasible rate | **0.730**（73/100） | 点估计过门（≥0.70） |
| Wilson 下端点（95% 双侧，z=1.96） | **0.636** | **未过门**（<0.70） |
| steady 均值/中位 | 0.820 / 0.973 | 过 |
| weak3 均值/中位 | 0.808 / 0.972 | 过 |
| worst 均值/中位 | 0.808 / 0.972 | 过 |

> **诚实结论**：① 点估计 QoS 0.73 在全新 blind seed 上成立（dev selection 的
> 0.975 是复用偏差，盲测证实 ~0.25 差距）；② 但 **N=100 统计功效不足以 95% 置信
> 保证 QoS ≥ 0.70**（LCB 0.636），与 4/4 观察（LCB 0.625）同构——论文若以
> `--require-lcb` 声明须如实报告 LCB 不达标，或改用点估计 + 置信区间披露。

**左尾分解（27/100 失败 seed，`tools/` 下分析脚本）**：
- 失败是**场景级**：steady≈weak3≈worst 同时塌（如 seed 609：0.242/0.060/0.060），
  不是单目标缺陷。
- 主导因子 = **初始最差目标-最近 UAV 距离** `worst_nearest`（bank 元数据）：
  <350 m QoS 0.91 → 350–450 m 断裂至 0.69 → 450–550 m 0.40 → >550 m 0.17。
- **物理时间预算**：v_max=25 m/s × T=150 帧 × dt=0.1 s → episode 最大位移 **375 m**。
  实测 realized 最差距离与 worst P_D 相关 **−0.863**：<200 m QoS 1.0、200–300 m
  0.57、>300 m 0.0。故 `worst_nearest > ~450 m` 的 seed 在给定运动学下已到
  **时间可达性边界**（部分物理不可达，非纯策略失败）；300–450 m 区间（QoS 0.4–0.69）
  属于**策略未在预算内把 UAV 送到**，是可改进区。
- 改进方向（D1.9 候选）：初始瞬态专门优化（前 N 帧连续几何 warm start /
  瓶颈目标优先全速趋近），或场景可行性边界披露（对物理不可达 seed 标记排除）。

## D1.9：瓶颈前瞻 L3（receding-horizon approach scoring，2026-08-16）

**动机（由 D1.5 左尾分解驱动）**：300–450 m `worst_nearest` 的 seed（QoS
0.4–0.69）失败机制是**单步 trust-region 停滞**——多候选 L3 的径向候选只评
"1 步后"几何，2.5 m 步长在 R≈300–450 m 处把 1/R⁴ ceiling 移动 ~0，精确 LP
无区分度，候选池常选 stay，UAV 不趋近瓶颈目标。而运动学预算（375 m）本可
覆盖该区间（理想全速直线趋近 15 s 可使 97/100 seed 进入 200 m 圈，实测策略
仅 58/100）。

**方案**：`analytical_movement_lookahead_frames=H`（默认 0，H>0 启用）——
多候选 L3 对弱目标径向候选改在 **H 帧持续趋近后**的几何（`uav + H·step·dir`）
上评分，执行仍只走 1 步（`apply_action` clamp 到 v_max·dt=2.5 m）——
**receding horizon**。理论支撑：
- P_D = Q(Q⁻¹(P_FA) − √D)，D ∝ P/(R_tx²·R_rx²)；H=40 → 100 m 前瞻，把
  300–450 m 的 ceiling 推进到 P_D 饱和区，LP 恢复区分度；
- 物理一致：前瞻几何用精确 1/R⁴ 重标定 per-watt 张量，d_safe 保护不变，
  stay 候选保留 → 代理分数单调不退化；
- 创新点：把"1 步 LP 评分"升级为"执行 1 步 / 评分 H 步"的时间耦合评分，
  显式利用 v_max·T·dt 位移预算（D1.5 发现的物理可达边界）。

**数值验证**：
- 单元测试（`test_dual_pruned_movement_candidates.py`）：H=0 与 D1.1-B 逐位
  一致；H>0 候选集是超集 + stay 保留（单调）；执行位移 clamp ≤2.5 m；
  远场瓶颈场景（400 m）H=40 选出趋近而非 stay。
- 合成几何（2 UAV，弱目标 500 m，60 帧）：H=40 向瓶颈目标移动 147.5 m vs
  基线 83.8 m（**+76%**），全速趋近语义成立。
- **冒烟（6 个瓶颈 blind seed，worst_nearest 344–433 m，均为基线失败）**：
  QoS **1/6 → 5/6**；worst 提升 +0.699（seed 446: 0.259→0.959）、+0.636
  （seed 98）、+0.573（seed 696）、+0.264（seed 598）；唯一未过 seed 615
  亦 0.085→0.455。已通过 seed 645 保持（0.973→0.990）。
- **随机 10-seed 对照**：QoS **9/10 → 10/10，零退化**（通过 seed delta ≤
  1e-3），左尾 seed 295 worst 0.376→0.999、seed 367/631/372 均补足到 1.0。
- 综合 16 个验证 seed：**10/16 → 15/16 QoS**。

**状态**：全量 100-seed blind 复测已完结（`results/_d1_9_blind100/`，lookahead
40 配置 `exp_800_k8q8_..._blind_lookahead40.yaml`），正式对比（H=0 vs H=40）
见下。辅助：`run_mappo.py
--final-eval-seeds`（显式 seed 列表诊断评估）、`tools/smoke_d19_lookahead.py`。

**全量 100-seed blind 认证（`results/_d1_9_blind100/paired_eval.csv`，~4.4 h）**：

| 指标 | D1.5（H=0） | **D1.9（H=40）** | 判定 |
|---|---:|---:|---|
| QoS feasible rate | 0.730 | **0.950**（95/100） | 大幅过门 |
| Wilson LCB（95%） | 0.636 | **0.888** | **过门（≥0.70）** |
| steady 均值/中位 | 0.820 / 0.973 | **0.963 / 1.000** | 过 |
| worst 均值/中位 | 0.808 / 0.972 | **0.962 / 1.000** | 过 |

> **结论**：① 单步 trust-region 停滞是 D1.5 左尾主因——H=40 前瞻评分使
> 100 seed 盲测 QoS 从 0.730 提升至 **0.950**（+0.22），LCB 0.636 → **0.888**
> （+0.25，N=100 统计功效达标，`--require-lcb` 论文声明成立）；
> ② 剩余 5 个失败 seed 中 **4 个（886/298/45/185）worst_nearest > 450 m**，
> 在 episode 375 m 位移预算下物理不可达（D1.5 左尾分析已预告），仅 seed 615
> （355 m）为残余策略失败——**95/100 已近该运动学下的可达性上界**；
> ③ `tools/report_blind_certification.py` 修复：`--csv` 参数生效
> （此前硬编码 D1.5 路径）。正式 Gate 表（`assert_formal_gates.py`）中
> D1.9 点估计 + LCB 强化两行均 PASS。

## D1.10：残余失败 seed 修正尝试（2026-08-16，审计驱动）

**动机**：D1.9 盲测 95/100 的 5 个残余失败 seed 经审计（数据见 `EXPERIMENT_LOG.md`）
分类为 3 物理不可达 + 1 RNG 序列效应（298）+ 1 功率耦合（615）。D1.10 针对
后两类实施修正。

**D1.10-A：评估协议独立 env 实例（有效，`eval_independent_env`，默认 OFF）**
- 机制：共享 env 协议下 `deflection_computer` 的 Rician/LoS rng（`__init__`
  构造，`wrapper.reset` 不替换）跨 episode 漂移，第 k 个 seed 的随机实现
  依赖其前跑了多少 episode。`_build_eval_env(seed=ep_seed)` 每 episode 新建
  env，使每个 seed 独立采样。
- 20-seed A/B：indep vs shared 均 14/20（无系统性方向），但独立协议是
  **统计正确的采样**。**indep + lookahead40（无 tstar）在 20-seed 上
  15/20（steady 0.886）**，优于共享序列值。默认 OFF 保持历史结果数值
  不变；认证复跑应启用。

**D1.10-B：L3 Phase-1 触发改 max-min t\*（负结果，默认 OFF）**
- rev1 均匀 deficit 退化（15→14/20）；rev2 λ\* 加权 deficit 聚合持平但
  具体 seed 退化（615: 0.71→0.29、298: 0.80→0.30）。任何 Phase-1 deficit
  梯度都改变候选池输入，而 Phase-2 对偶价格梯度（lookahead40）实际物理
  上更优。**默认 OFF**，耦合稀缺修复属 D1.10-C（per-UAV 结构重分配候选，
  未实现）。

**状态**：indep + lookahead40 全量 100-seed blind 复跑认证已完结
（`results/_d1_10_blind100_indep/`，配置
`exp_800_k8q8_..._blind_lookahead40_indep.yaml`），独立采样
协议下的 QoS/LCB 正式值见下。

**全量 100-seed 独立采样盲测认证（`results/_d1_10_blind100_indep/`，~4 h）**：

| 指标 | D1.9（共享协议） | **D1.10（独立采样）** | 判定 |
|---|---:|---:|---|
| QoS feasible rate | 0.950 | **0.940**（94/100） | 过门（≥0.70） |
| Wilson LCB（95%） | 0.888 | **0.875** | 过门（≥0.70） |
| steady 均值/中位 | 0.963 / 1.000 | **0.966 / 1.000** | 过 |
| worst 均值/中位 | 0.962 / 1.000 | **0.963 / 1.000** | 过 |

> **结论**：① 独立采样协议（每 seed 独立 env，消除共享实例 RNG 漂移）下
> QoS 0.940 / LCB 0.875 双双过门——**认证结果对评估协议稳健**（95/94 与
> 0.888/0.875 均在统计噪声内）；② 协议修正价值体现在具体 seed：615
> 0.430→0.692、298 0.326→0.788（共享序列运气被去除），同时暴露 613
> 真实边界（1.000→0.794，差 0.006 过 0.8 地板）；③ 6 个失败 seed =
> 615/298/613（255–479 m 可达区，策略边界）+ 886/45/185（>582 m 物理
> 不可达）——与 D1.9 审计一致；④ 该值是当时的历史认证，后续已被 V3-C0
> `0.910/0.838` 的物理修复口径取代，并在 G2-0 量纲修正后等待重认证。

## D1.10-C：coupling-aware structure repair（2026-08-17，advice 014 §4–§6，**负结果，关闭**）

**动机**：seed 615 的失败本质是"有限 UAV 功率预算下的结构性资源拥塞"（UAV 2 独拥
4 个瓶颈目标，其 1 W 被多个目标竞争；`worst_deflection = 6.44 << d_min`）。D1.10-B
证明任何全局 t\* trigger 都会伤害正常 seed；D1.10-C 按 advice 014 改为**仅在结构拥塞
被证明时做稀疏 per-UAV 责任重分配**：路由 `t_fix < d_req ≤ t_relax`（结构可修）→
对偶热点检测（`λ*_q>0` 且 `p_iq>0` 且 UAV 服务 ≥2 个瓶颈目标）→ 1-swap 候选
`(i,q)→(k,q)`（owner 迁移/支持边剔除拥塞 TX）→ 严格字典序接受
（QoS feasibility ≻ floor violation ≻ t_min）+ churn 守卫（warmup=10、hold=20、
候选须保持 lex 可行 ceiling≥d_min、改善 ≥ 0.2·d_req 或翻转可行）。

**实现**：`uav_isac/coordination/congestion_relief.py`（纯函数 + 驱动）、
`env_core.step` 注入（P0 选择后、解析功率 LP 前）、配置
`analytical_structure_congestion_relief_{enabled,hold_frames,warmup,min_gain}`
（**默认 OFF**）、A/B 配置
`exp_800_k8q8_..._blind_lookahead40_indep_relief.yaml`、
回归 `tests/test_congestion_relief.py`（9 项：热点/候选/字典序/合成 seed-615 场景
/never-worsen，全过）。

**A/B 结果（独立采样协议，3 个 dev seed，relief ON vs OFF）**：

| seed | 基线 worst / steady | relief worst / steady | 判定 |
|---|---:|---:|---|
| 615 | 0.6923 / 0.7099 | **0.4908 / 0.4916** | 严重退化 |
| 298 | 0.7880 / 0.7956 | **0.1703 / 0.1732** | 崩坏 |
| 613 | 0.7501 / 0.7938 | **0.9001 / 0.9376** | 救活（rescued） |

三版迭代（aggressive 每帧 → hold/warmup → 严格门槛+ceiling 过滤）均为同一模式：
**LP 的帧内"改善"在执行路径（L3 几何评分、下帧 P0 重导、lex L1 执行）下不落地，
结构更换对近达标 seed 的伤害超过拥塞缓解收益**——`N_rescued=1`（613）但对
615/298 造成崩坏级退化（298: 0.79→0.17），与 D1.10-B rev3 失败模式同构。

**结论（按 advice 014 §六）**：D1.10-C 作为部署修复器**关闭**（flag 保持默认 OFF）。
模块与单元测试保留为可审计研究产物（语义正确的组件，但逐帧集成不成立）。
seed-615 类耦合稀缺的修复仍需不同机制（例如 P0 owner 偏好偏置而非帧后置交换），
登记为开放项。该负结果不改变当时 D1.10 历史值，但不能越过后续 V3-C0/G2-0 修订线。

## P1-4：C2 精确 MILP 结构上界审计（2026-08-17，advice 014 §八，完成）

**问题**：single-duplex oracle（交替启发式）此前被当作结构上界，但自述"not a
proof of global optimality"。advice 014 C2 要求小规模 exact MILP 量化
heuristic-oracle vs exact 的 gap。

**实现**：复用既有精确基础设施 `uav_isac/coordination/qos_threshold_feasibility.py`
（阈值可行性 MILP + 单调二分，scipy.milp/HiGHS，proven 可行下界 + proven 不可行
上界；perspective 约束避免分数 owner 复制 RF 功率）。`tools/audit_qos_threshold_boundary.py`
在 `architecture_v2_scale_k6q6_teacher_trace_d035_.../teacher_trace.npz` 的 20 个
代表性 **final-resolved** 帧上求精确联合结构+功率边界（补丁：退化帧初始地板不可行时
跳过并记录，不中断审计）；`tools/audit_joint_structure_relaxation.py --exact-reference`
对照稀疏启发式候选。

**结果（20 帧，worst P_D，mean）**：

| 层级 | worst P_D | 说明 |
|---|---:|---|
| recorded（策略） | 0.5646 | 实测 |
| fixed（固定结构功率 LP） | 0.7924 | 仅功率层启发式 |
| candidate（对偶引导稀疏候选） | 0.8775 | 结构启发式（23.9% 边，稀疏） |
| **exact joint（MILP 下界/上界）** | **0.9154 / 0.9197** | **证明紧致（宽度 0.0043）** |
| LP relaxation 上界 | 0.9221 | 整数 gap 仅 0.0067 |

- 结构 headroom（exact − fixed）= **+0.123**；启发式候选恢复 **76.1%** 结构增益，
  QoS feasible rate 与 exact 相同（0.90）——稀疏候选在可行率分类上无损。
- **C2 措辞结论**：联合结构+功率上界现已由精确 MILP **证明**（0.915–0.920，
  代表性帧），不再是"交替启发式自述"；论文可将该上界表述为 exact/proven，
  并如实报告稀疏启发式与其的 gap（~0.038 P_D / 未恢复 24% 结构增益）。

## P1-3：6/6 独立采样盲测（2026-08-17，advice 014 §七）

**设置**：`tools/generate_blind_seed_bank.py --bank 980_k6q6 --draw-seed 20260828`
生成 `config/stratified_seeds_980_k6q6_blind.json`（100 seed，tier 均衡，排除全部
bank split/已暴露/隔离种子）；配置
`exp_800_k6q6_analytical_l0l1_movement_lex_candidates_v2_blind_indep.yaml`
（D1.6 候选 + 盲测 bank + `eval_independent_env=true`）。**关键参数**
`--structure-student-adaptive-min-bits-per-dim 4`（D1.6 manifest 必带；缺失时
student 端点传输塌缩到 P_FA 地板——诊断中复现并确认）。

**结果（100 全新 blind seed，独立采样，冻结 Student + 解析栈）**：

| 指标 | 值 | 判定 |
|---|---:|---|
| QoS feasible rate | **0.530**（53/100） | **远低于 0.70** |
| Wilson LCB（95%） | **0.433** | **远低于 0.70** |
| mean / median worst | 0.659 / 0.746 | — |

> **结论：6/6 当前冻结 Student + 解析栈无法通过独立采样盲测认证**（决定性
> 失败；10-seed 冒烟 0.40 与全量一致）。按 advice 014 §七判定规则，正在
> 47 个失败 seed 上跑 **teacher（P0 集中式结构，无 student）对照**；若 teacher
> 显著通过 → **Student calibration 是下一科学瓶颈**（~0.15 校准 gap 成为决定性
> 因素）；若 teacher 同样失败 → 6/6 瓶颈在几何/物理而非 Student。

**Teacher 对照（47 个 Student 失败 seed，独立采样，同一解析栈）**：

| 结构来源 | QoS（47 失败子集） | 说明 |
|---|---:|---|
| Student（冻结） | 0/47 | 全部失败 |
| **Teacher（P0 集中式）** | **34/47 = 0.723** | 43/47 显著改善（>+0.05），均值 +0.49 |

> **最终判定（advice 014 §七）**：teacher 在同 seed 上**决定性通过**（34 个
> rescued；保守外推全 100-seed teacher QoS ≥ (53+34)/100 ≈ 0.87，LCB ≥ ~0.79
> ≥ 0.70）——**Student 结构校准（4/4 锚点基数残差的 6/6 结构近似，gap ~0.15）
> 是 6/6 认证的下一科学瓶颈**，而非几何/物理（同一解析栈 + 集中式结构即可
> 达门）。论文中 6/6 的"配置不匹配（D1.6）"结论保留，但必须补充：盲测真实
> 水平 0.53（Student）/~0.87（Teacher 外推），Student 是绑定环节。后续工作：
> Student 跨尺度校准或更优结构代理，再重跑 100-seed 独立采样认证。

**附带发现（诊断过程）**：① D1.6 的 20-seed 结果（QoS 0.70）基于 SELECTION
split（偏乐观）；盲测 tier 均衡 bank 下真实水平 ~0.53。② 同一 seed+配置在
不同进程配置下 P_D 波动可达 ±0.2（seed 29：认证 1.00 / 直跑 0.84 / MKL=1 0.76）——
解析栈数值对 BLAS/MKL 线程数敏感（见 P1-5），近地板 seed 的判定对运行环境
敏感，进一步支持 E1（统计口径纪律）。

## P1-5：MKL 崩溃隔离（2026-08-17，advice 014 §九，完成）

**实现**：`tools/crash_isolated_seed_eval.py`——每 seed 一个子进程（spawn），
`MKL_NUM_THREADS=1`/`OMP_NUM_THREADS=1`，父进程收集合并（`paired_eval.csv`
保持单行数组 schema，下游聚合器兼容），worker 崩溃 → 该 seed **fail-closed
记录**（`crashed_seeds_fail_closed`），不中断整次认证。验证：3+3 seed 双轮
（合并 QoS 2/3、3/3 与直跑一致）。

**重要发现（数值线程依赖）**：同一 seed+配置下，默认线程 vs `MKL_NUM_THREADS=1`
结果不同（seed 29：worst 0.8385 → 0.7567；认证共享进程 1.00）——解析栈的
LP/BLAS 归约对线程数敏感，近地板 seed 判定随运行环境摆动。**含义**：崩溃隔离
协议的认证数值须在其自身线程配置（MKL=1）下报告，不能与默认线程历史数值混比；
这使"单 seed 崩溃不杀认证"的鲁棒性目标与数值可复现性需要同一套进程级协议
（P2 后建议正式认证统一走 crash-isolated 包装器）。

## V3 主线：dual-priced 分布式结构协调（2026-08-17，advice 015，进行中）

**V3 定位**：冻结 L0/L1/L3，重构 L2 为对偶一致的分布式结构协调（同一组任务对偶
价格统一功率-结构-几何）；8/8 D1.10 数值保留为历史认证，物理修复后建立新 V3
baseline。以下为已完成的 V3-C0/V3-T1/V3-T2。

### V3-C0 科学正确性冻结（已修复，量化影响）

- **χ_rep 每接收机单抽**（`deflection.py`）：上报链路可靠性只依赖接收机 j→FC，
  原代码在 (i,j) 内层每 TX 重抽 Rician（同链路不同可靠性、每帧 |tx|×|rx| 次
  RNG）→ 改为每 j 抽一次、跨 TX 复用（V3-C0）。
- **能量记账按 P0 实际角色**（`env_core.py`）：learn_roles=False 时策略角色是
  idle 占位、apply_action 不扣无线电能量 → 在 P0 角色推导后按实际角色扣
  （TX 扣 P_sense·dt、RX 扣 P_report·dt、dual 端点扣 TX 子时隙）。
- 统一协议：独立 seed env、MKL=1（崩溃隔离包装器）、评估 tol 已在认证管线。
- **量化影响**（8/8 独立采样，3 seed）：29: 0.999→0.809、241: 0.9999→0.9999、
  437: 0.999→0.929——**物理修复改变数值分布（符合预期），需建立新 V3 baseline
  （不调参）**；全量测试 937 passed / 1 env failure 无回归。

### Gate 0：V3-C0 全量重认证（2026-08-18，advice/016 §18 Gate 0，完成）

**协议**：8/8 与 6/6 各做独立采样 blind100 重认证，**不调参**；与 D1.10/P1-3
**同 seed、同配置、同 warm-start、同协议**（8/8 为 eval-only `--episodes 0`，
6/6 沿用 P1-3 的 config `num_episodes=1` 单训练 episode），唯一变化是 V3-C0
物理修复已进入工作树。由于 8/8 blind bank 在 D1.10 后被 v2 回填改动，用
`--final-eval-seeds` 显式钉住 D1.10/P1-3 实际评估的 100 个 seed，保证逐 seed
配对可比。结果目录：`results/_gate0_8x8_v3c0_indep100/`、
`results/_gate0_6x6_v3c0_indep100/`；汇总：
`results/_gate0_v3c0_summary.json`；A/B 工具：
`tools/summarize_gate0_v3c0.py`。

**8/8（V3-C0 vs D1.10，同 100 seed，配对）**：

| 指标 | D1.10（历史） | **V3-C0 重认证** | 判定 |
|---|---:|---:|---|
| QoS feasible rate | 0.940（94/100） | **0.910**（91/100） | **过门（≥0.70）** |
| Wilson LCB（95%） | 0.875 | **0.838** | **过门（≥0.70）** |
| steady / worst 均值 | 0.966 / 0.963 | **0.924 / 0.915** | 过 |
| 配对翻转 | — | PASS→FAIL 4（316/609/391/605）、FAIL→PASS 1（613） | worst Δ 均值 −0.048 |

> **结论**：V3-C0 物理修复后 8/8 仍**双过门**（QoS 0.910 / LCB 0.838 ≥ 0.70），
> 但数值分布按预期整体下移（D1.10 的 0.940/0.875 保留为历史认证，不覆盖）。
> 4 个 PASS→FAIL 翻转（316/609/391/605）均为 realized worst_nearest
> 246–315 m 的策略边界 seed；613 被救活（0.750→0.959）。**V3-C0 新 baseline
> 已建立**；论文若引用 8/8 认证值须改用本行（0.910/0.838）并注明物理修复
> 后的认证口径。

**6/6（V3-C0 vs P1-3，同 100 seed，配对）**：

| 指标 | P1-3（历史） | **V3-C0 重认证** | 判定 |
|---|---:|---:|---|
| QoS feasible rate | 0.530（53/100） | **0.590**（59/100） | **仍未过 0.70** |
| Wilson LCB（95%） | 0.433 | **0.492** | **未过 0.70** |
| steady / worst 均值 | 0.775 / 0.659 | **0.789 / 0.688** | — |
| 配对翻转 | — | PASS→FAIL 8、FAIL→PASS 14 | worst Δ 均值 +0.029 |

> **结论**：V3-C0 修复没有改变 6/6 的**绑定瓶颈定性**——冻结 Student + 解析栈
> 在独立采样 blind100 上仍远低于 0.70 门（QoS 0.59 / LCB 0.49，统计噪声内
> 与 P1-3 的 0.53/0.43 一致）。P1-3 已证明 teacher 在同 seed 上可达 ~0.87
> （47 失败子集 34/47），故 **Student 结构代理仍是下一科学瓶颈**；Gate 0
> 确认该结论对 V3-C0 物理修复稳健。**进入 Gate 1（multi-scale CE Student
> falsification）**。

### Gate 1：multi-scale CE Student falsification（2026-08-18，advice/016 §18 Gate 1，完成）

**设计**：关键 falsification 实验——**只**加 4/4+6/6+8/8 teacher 数据、损失仍为
**普通 imitation**（balanced multicardinality edge-value CE），验证"补数据是否已
够"（advice/016 §17/§18：若 6/6 QoS LCB ≥ 0.70 说明问题大部分只是 dataset
shift，**不要过度发明新算法**）。训练：`tools/train_multiscale_structure_student.py`
（普通 imitation，preservation_weight=0，无 task-regret/dual 权重）；数据：干净
trace（4/4 gate100 3100 帧 + 6/6 d035 620 帧 + 8/8 d079 620 帧，全部零隔离种子、
与 blind bank 零重叠）；锚点：4/4 `frozen_structure_student_endpoint8.pt`
（endpoint_dim=8，16-wide 协议，与 P1-3 baseline 同宽）。评估：P1-3 同款
blind100 独立采样协议（同 config/同 warm-start/同 100 seed/同 1 训练 episode），
仅替换 Student checkpoint。

**⚠ 第一次训练用错锚点（诚实记录）**：首轮用 `frozen_structure_student.pt`
（endpoint_dim=32）做 preservation 锚点 → Student 输出 **64-wide endpoint 协议**，
超出物理 U2U 信道（atomic delivery rate 0.0、`insufficient_cache_rate=1.0`、
public_valid=0）→ 100 seed 全塌缩到 P_FA 地板（QoS 0.000）——这不是算法失败而
是**协议宽度错配**（与 P1-3"adaptive bits 必带否则塌缩"同属物理信道错误类）。
改用 endpoint8 锚点（16-wide，821 bit/frame，与 P1-3 一致）后 channel
delivery=1.0、public_valid=1.0，恢复正确执行。**教训**：Student endpoint 宽度
是物理信道约束，训练时须与部署基线一致。

**结果（100 全新 blind seed，独立采样，普通 imitation multi-scale Student）**：

| 指标 | P1-3（冻结 Student） | Gate 0（V3-C0 同 Student） | **Gate 1（multi-scale CE）** | 判定 |
|---|---:|---:|---:|---|
| QoS feasible rate | 0.530 | 0.590 | **0.780**（78/100） | **点估计过 0.70** |
| Wilson LCB（95%） | 0.433 | 0.492 | **0.689** | **差 0.011 未过 0.70** |
| steady / worst 均值 | 0.775 / 0.659 | 0.789 / 0.688 | **0.863 / 0.842** | — |
| 配对 worst Δ vs P1-3 | — | +0.029 | **+0.183（58/100 改善 >0.05，8 恶化）** | 大幅改善 |

> **结论（falsification 成立）**：**普通补数据已恢复 6/6 的大部分缺口**——
> QoS 0.53/0.59 → **0.78**（+0.19/+0.25），mean worst 0.66/0.69 → **0.842**
> （+0.15/+0.18），58/100 seed 显著改善、仅 8 个显著恶化。**点估计已过 0.70
> 门**；LCB 0.689 差 0.011（1 个 seed 翻转即 79/100 → LCB 0.700）未达严格
> `--require-lcb` 门。按 advice/016 Gate 1 决策规则：**dataset shift 是 6/6
> 失败的主因（advice/016 §17 的"multi-scale CE 是否已够"答案：基本够、差
> 统计尾巴）**——不宜把 task-regret/dual-weighted 当作唯一必要机制；Gate 2
> 的价值收窄为"用任务敏感权重补最后 ~0.01–0.05 的 LCB 尾巴 + 降低错判代价"。
> 结果目录：`results/_gate1_multiscale_ce/`（训练）、
> `results/_gate1_6x6_multiscale_ce_blind100/`（认证）。**下一步（Gate 2）**：
> 同一 multi-scale 数据 + `R_γ;lex;R_t` + dual-weighted 边权，目标 E[R_γ] 与
> P(γ_θ>1 | γ*≤1) 下降、QoS LCB ≥ 0.70。

### V3-T1 资源竞争诊断（`tools/diagnose_structure_competition.py`）

6/6 盲测失败 seed（479/866/725/237/410/274）逐帧统计对偶价格 π_q、UAV 稀缺价格
η_i=max_q λ*_q a_iq、TX 复用、owner 负载、支持集、竞争帧占比：

| seed | worst | η_max | TX复用 | owner负载 | 竞争帧(全/稳态) |
|---|---:|---:|---:|---:|---:|
| 479 | 0.311 | 3.9 | 2.2 | 6.0 | 0.09 / 0.00 |
| 866 | 0.174 | 3.0 | 2.1 | 4.6 | 0.58 / 1.00 |
| 725 | 0.202 | 1.5 | 2.2 | 5.7 | 0.65 / 1.00 |
| 237 | 0.259 | 5.1 | 2.3 | 4.9 | 0.61 / 0.25 |
| 410 | 0.180 | 2.0 | 2.7 | 5.3 | 0.55 / 1.00 |
| 274 | 0.579 | 5.1 | 2.3 | 3.1 | 0.61 / 0.00 |
| PASS 14 | 1.000 | 62.7 | 2.3 | 3.6 | 0.84 / 1.00 |
| PASS 752 | 0.979 | 12.8 | 2.4 | 3.2 | 1.00 / 1.00 |
| PASS 705 | 0.569 | 9.2 | 2.3 | 4.4 | 0.90 / 0.75 |

**发现**：① 失败 seed 的 **owner 负载严重（单个接收机拥有 4.6–6.0/6 个目标）**，
TX 复用 2.1–2.7——"一个稀缺 UAV 被多个目标争夺"的结构性拥塞确实存在；② 但
**竞争帧占比在 PASS seed 同样高（0.84–1.00）**——竞争是普遍现象，不是失败
充分条件；③ 区分度在**稀缺价格的量级**：PASS seed 的 η_max 高达 9–63（近目标
高增益），失败 seed 仅 1.5–5.1（几何上缺乏补偿能力）——**失败 = 竞争 × 弱几何
（t* 塌到地板下），竞争本身由 L2 结构承担、但几何（L3）决定最终是否可行**。
诊断修正了"仅靠结构重分配即可救 6/6"的朴素假设。

### V3-T2 dual-priced owner assignment 原型（`coordination/dual_priced_structure.py`）

**数学**：固定结构 max-min LP 对偶给出目标价 π_q 与 UAV 稀缺价
η_i=max_q λ*_q a_iq（1 W 预算的精确 Lagrange 乘子）；候选边 reduced-cost
`s_ijq = π_q a_ijq − η_i`；固定角色下 owner 赋值成为二分 (b-)matching（owner 值
`V_jq = Σ_{i≠j} [π_q a_ijq − η_i]_+`，**i≠j 物理约束硬编码**），LP relaxation
全幺模 → 整数解；κ>0 切换惩罚用精确 MILP。L1 在赋值后精确重解。

**关键理论发现（tests/test_dual_priced_structure.py，8 项）**：
1. **盈余捕获成立**：瓶颈目标识别出有真实盈余的 owner（合成拥塞场景 target 0
   → owner 3），仅提交该重分配把 t* 从 6.0 提到 7.5（ceiling capture）。
2. **naive 全量匹配非 t\* 单调**：当前对偶下强 TX 的 reduced-cost 恰为 0
   （`η_i = max_q π_q a_iq` 精确抵消），匹配会把 owner 分散、碎片化强 TX 预算、
   反而降 t*（6.0→4.03）——**必须"盈余提交 + κ 稳定性"联合使用**（与
   D1.10-B/C 的教训一致：结构层任何非证明性改动都会伤正常帧）。
3. 对偶一致性：`Σ_i b_i η_i == t*`（强对偶）数值验证。

**下一步（V3-T3/T4）**：分布式 auction（量化价格 + 本地 bid，D1.7 λ-μ DW 复用）、
盈余提交门（无退化证明）、6/6 100-seed 重盲测。理论链条将闭合为"同一组任务对偶
价格统一功率（L1）—结构（L2）—几何（L3）"。

### V3-T3/T4 分布式价格迭代 + 盈余提交门（已完成，`coordination/dual_priced_auction.py`）

**T3 分布式价格交换**：TX 广播量化稀缺价 η_i 与逐瓦增益 a_ijq（6-bit，公共尺度
归一后量化，D1.7 约定），目标本地累计 owner 值 `V_jq = Σ_{i≠j}[π_q a_ijq − η_i]_+`
并竞价；无人工接收机容量时 per-target argmax 即 TU 整数解；L1 在赋值后精确重解。
每轮通信 bit 有界：`rounds·(|tx|·(Q·6+6) + |owners|·6 + 6)`。

**T4 盈余提交门（单调局部搜索 + 精确认证）**：吸收 V3-T2"naive 全量匹配非 t\*
单调"教训，迭代改为**单目标坐标更新**——每轮只提交"owner 值盈余最大"的一个
目标（`q* = argmax(V[desired]−V[current])`），候选用**精确 max-min LP + 字典序
QoS key**（可行性 ≻ 地板违规 ≻ t_min）判定，接受仅当严格更优 → **接受序列 t*
单调非降（构造性无退化）**；`min_improvement` 近端守卫拒绝边际改动；结构
hold 窗口保护 L3。

**单测**（`tests/test_dual_priced_auction.py`，9 项）：量化误差界/公共尺度归一、
局部 argmax 赋值、字典序门、合成拥塞 6.0→7.5+、随机帧单调性、近端守卫、通信
bit 界。**真实帧验证**（`tools/verify_v3_t3_on_trace.py`，6/6 失败 seed 的
178 个解析帧）：**123 帧 t* 提升（69.1%，均值 +3.17、最大 +12.65 deflection），
0 帧退化，55 帧 no-op**——对偶价格迭代在大部分瓶颈帧找到结构余量且构造性零
退化（与 D1.10-C 的退化对照）。**剩余问题**：live 集成后结构变更与 L3 几何循环
的交互（hold 窗口/提交频率）需 episode 级验证——下一轮 V3-T4 live。

### V3-T4 live 集成（已完成，env_core + env_wrapper + 单测 + A/B）

**实现（避免 D1.10-C 教训的三重防线）**：

1. **单调门**（`dual_priced_auction.price_iteration` + `monotone_commit`）：每
   `v3_structure_hold_frames` 帧从当前 P0 结构出发跑分布式价格迭代，只提交
   **精确 max-min LP 字典序严格更优**（可行性 ≻ 地板违规 ≻ t_min）的单目标
   坐标更新 → 接受序列 t* 单调非降，**构造性零退化**（真实帧 0/178 退化）。
2. **hold 窗口**（`v3_structure_hold_frames=20`）：结构在窗口内冻结，L3 几何
   循环对固定结构交替下降（D0.95 风格），排除 D1.10-C 的逐帧结构 churn。
3. **通信核算**：量化价格 bit 有界（`distributed_owner_values` 返回
   `comm_bits = K·(Q·6+6) + K·6 + 6` 每轮，6-bit 公共尺度量化）。

**接线**：`config/params.py` 新增 `v3_dual_priced_structure_enabled`
（默认 False）、`v3_structure_hold_frames=20`、`v3_price_bits=6`；
`env_core.__init__`/`reset` 读取标志并初始化计数器
（`_v3_iterations`/`_v3_commits`/`_v3_last_commit_frame`），step 中在 D1.10-C
块后、D0.89 功率块前注入 V3 块；`env_wrapper` info 暴露
`v3_iterations`/`v3_commits`。**名称遮蔽坑**：V3 块内局部
`from ... import fixed_owner_gain_matrix` 会把该名字变成整个 `step()` 的
函数局部变量，D0.89 块（模块级 import）在 V3 未触发帧上 UnboundLocalError →
改为复用模块级 import，仅局部 import `solve_fixed_structure_maxmin_power_lp`。

**单测**（`tests/test_v3_dual_priced_live.py`，8 项）：标志/计数器接线与
info 暴露、V3 依赖 analytical 校验、per-episode 计数器重置、hold 窗口限制
迭代频率（相邻迭代帧距 ≥ hold）、提交有限且有界（0 ≤ commits ≤ iterations
≤ T）、长 episode（120 帧）结构有效且 1 W 预算保持、提交仅出现在迭代帧、
OFF 路径零迭代零提交。**全量测试 954 passed / 1 env failure（sklearn，既有）**。

**A/B 设置**：6/6 独立采样盲测配置（P1-3 同款，Student + 解析栈 +
`eval_independent_env=true`）上，V3 ON（
`exp_800_k6q6_..._v2_blind_indep_v3.yaml`，仅加 V3 标志）vs OFF（父配置），
9 seed（6 失败 479/866/725/237/410/274 + 3 PASS 14/752/705），crash-isolated
每 seed 子进程 MKL=1。seed 479 冒烟：**worst 0.328→0.379（+0.051）、weak3
0.365→0.443（+0.078）、steady 0.566→0.602（+0.036）**；150 帧内 8 次价格
迭代、2 次提交（hold=20 生效）。

**A/B 结果（9 seed，独立采样，V3 ON vs OFF）——混合结果，如实记录**：

| seed | OFF worst | ON worst | Δ worst | Δ weak3 | Δ steady | 类型 |
|---|---:|---:|---:|---:|---:|---|
| 237 | 0.270 | 0.439 | **+0.169** | +0.238 | +0.164 | 失败→改善 |
| 274 | 0.123 | 0.334 | **+0.211** | +0.114 | +0.056 | 失败→改善 |
| 479 | 0.328 | 0.379 | +0.051 | +0.078 | +0.036 | 失败→改善 |
| 725 | 0.155 | 0.249 | +0.094 | +0.122 | +0.004 | 失败→改善 |
| 752 | 0.797 | 0.856 | +0.060 | +0.076 | +0.064 | PASS→改善 |
| 866 | 0.820 | 0.785 | **−0.035** | −0.029 | −0.023 | 失败→微降 |
| 410 | 0.259 | 0.185 | **−0.073** | −0.078 | −0.077 | 失败→降 |
| 14 | 0.950 | 0.800 | **−0.150** | −0.083 | −0.033 | PASS→降 |
| 705 | 0.995 | 0.937 | **−0.058** | −0.039 | −0.033 | PASS→降 |

均值：worst **+0.030**、weak3 **+0.044**、steady **+0.017**；QoS 计数 4/9 → 4/9
（不变）。**5/9 改善（4 个失败 seed 显著 +0.05~+0.21）、4/9 退化**（含失败
seed 410 与 PASS seed 14/705）。

**结论（关键科学发现）**：开环真实帧验证（69.1% 帧提升、0 退化）**不能迁移到
闭环 live**——单调门只对"提交帧快照"的精确 LP 保证 t* 非降，但 live 中：
① 提交改变该帧的执行结构/功率/观测 → 下一帧的解析运动（L3）状态轨迹分叉 →
episode 级结果可能更差（410/14/705 正是轨迹分叉机制）；② hold 窗口没有真正
"冻结"结构——P0 每 `p0_maxmin_pairing_hold_frames=5` 帧重新求解并覆盖执行
结构，V3 提交只影响提交帧到下次 P0 重解（≤5 帧）；③ live 接线传
`min_improvement=0.0`，**近端守卫被旁路**——边际重分配（V3-T2 明确"伤几何
循环"）也被提交，与 V3-T2 设计意图相悖。

### V3-T4 rev2：三项 live 集成修正（已实现，A/B 复测）

针对 rev1 的 3 个机制问题逐一修正（`config/params.py` + `env_core.py`）：

1. **deficit routing（`v3_deficit_only=true`）**：只有当前 P0 结构的精确
   max-min t* 低于 QoS 地板 d_req（结构不可行）才跑价格迭代；可行/PASS 状态
   完全不触碰（rev1 在 14/705 上因扰动可行配置而退化）。**物理依据**：结构
   协调是"不可行救援器"，不是"可行配置扰动机"。
2. **近端守卫恢复（`v3_min_improvement_frac=0.02`）**：live 传
   `min_improvement = 0.02·d_req ≈ 0.22 deflection`（不再是 0.0），拒绝
   数值无意义的边际重分配。
3. **提交持久化**：提交后写入 `_cached_p0_solution`，让改进结构真正执行到
   下次 P0 重解（rev1 只改局部 `p0_solution`、缓存仍返回原 P0 结构 → 提交
   仅 1 帧生效，且 `_last_selected_set`（观测）指向提交结构而执行回退原结构
   = 观测/执行错位）。

**单测**（`tests/test_v3_dual_priced_live.py` 增至 11 项）：新增 rev2 三项
（deficit-only 门控不提交可行帧、缓存持久化后结构仍为合法全覆盖固定 owner
图、近端守卫接线正值）。

**rev2 A/B 复测（同 9 seed，V3 ON rev2 vs OFF，crash-isolated MKL=1）**：

| seed | OFF worst | rev1 ON worst | rev2 ON worst | Δ rev2 | 判定 |
|---|---:|---:|---:|---:|---|
| 14 | 0.9501 | 0.8002 | **0.9501** | +0.0000 | **修复可行扰动**（rev1 退化复原） |
| 237 | 0.2695 | 0.4389 | 0.3258 | +0.0562 | 改善 |
| 274 | 0.1231 | 0.3337 | **0.6172** | **+0.4942** | **大幅改善**（0.123→0.617） |
| 410 | 0.2587 | 0.1854 | 0.1911 | −0.0676 | 仍退化 |
| 479 | 0.3280 | 0.3790 | 0.4422 | +0.1141 | 改善 |
| 705 | 0.9951 | 0.9370 | **0.9999** | +0.0049 | **修复可行扰动** |
| 725 | 0.1549 | 0.2490 | 0.1430 | −0.0119 | 基本持平（微降） |
| 752 | 0.7966 | 0.8562 | 0.8501 | +0.0535 | 改善 |
| 866 | 0.8195 | 0.7846 | 0.5530 | **−0.2665** | **新退化**（QoS PASS→FAIL） |

均值 worst：OFF 0.5217 → rev2 0.5636（**+0.042**，优于 rev1 +0.030）；QoS 计数
4/9 → **3/9**（866 由 PASS 翻转为 FAIL，抵消 274 的改善）。seed 866 复测确定
（0.5530 与 A/B 一致，非数值噪声）。

**rev2 结论（科学判定）**：deficit routing 正确修复了可行状态扰动（14 精确
复现 OFF、705 超 OFF——证明"只救不可行帧"路由有效且不碰可行配置）；近端守卫
恢复后 274/479/752 的改善保留且 274 大幅跃升（0.123→0.617）。**但 866 仍
退化且更重**（−0.267，rev1 仅 −0.035）——缓存持久化把提交结构保持到下次 P0
重解（≤5 帧），放大了"提交帧快照最优、几何移动后失效"的轨迹分叉效应：
单调门只保证提交帧精确 LP 的 t* 非降，不保证闭环 episode 级非降。**866 是
rev1 诊断中"弱几何 + 竞争"的典型**（η_max 3.0、owner 负载 4.6），其结构
修复在提交帧成立、但在 L3 移动后的后续帧失效——几何（L3）仍决定最终可行性
（与 V3-T1 结论一致：竞争 × 弱几何，结构可缓解但不可根治）。

### V3-T4 rev3：前瞻提交门（已实现，A/B 复测）

**针对 rev2 剩余瓶颈（866 类弱几何 seed）**：单调门只认证"提交帧快照"，但
几何在下次 P0 重解前被 L3 移动漂移。rev3 新增 `forward_geometry_gate`
（`dual_priced_auction.py`）：候选在快照认证通过后，再在**外推几何**（当前
L3 移动 delta + 目标速度，外推 `v3_lookahead_steps=1` 帧，位置 area-clip）
下重算基线/候选的精确 max-min LP，**候选在外推几何下不得严格劣于基线**
（任一外推步劣化即拒绝，fail-closed）；`steps=0` 关闭前瞻（=rev2 行为）。
`env_core` 接线：读 `v3_lookahead_steps`、拒绝计数 `_v3_lookahead_rejects`
暴露到 info；`_rescale_coefficient` 同时外推 UAV 与目标运动（扩展
`_friis_rescale_tensor`）。

**单测**（`tests/test_dual_priced_auction.py` 增至 13 项 + live 增至 13 项，
共 34 项）：前瞻门 steps=0=快照认证、漂移翻转拒绝（owner 3 被拉开）、
fail-closed 无效结构、零漂移与快照一致、live 接线/计数器/关闭开关。

**rev3/rev3b/rev4 诊断（决定性）**：

- **rev3 前瞻门零拒绝**（steps=1/5/20 全部 0 rejects）：候选结构在任意外推
  几何下都不劣于基线——**"几何漂移使结构失效"假设被证伪**。866 退化不是
  结构质量问题。
- **零动作探针因果隔离**：V3 无提交路径与 OFF **逐位一致**（tail20=0.3187），
  有提交略好（0.3257）——**V3 代码路径本身零扰动**；真实评估（冻结 Student
  策略）里 866 却大幅退化 → **退化机制 = 提交→执行结构→观测→冻结策略的
  comm/预算决策分叉→轨迹分叉**（闭环策略耦合，任何纯结构侧单调门都不可见）。
- **rev3b（无持久化）866 更差**（−0.600 vs rev2 持久化 −0.267）：持久化不是
  退化主因，单帧 flapping 更糟。
- **提交增益量级**：866 的 4 次提交全部仅 +0.507 deflection（4.5% d_req），
  274 的提交 +1.995（17.8% d_req）——**边际提交扰动策略、无物理收益**，正是
  V3-T2 近端守卫教训。rev4 把守卫提到 0.05·d_req≈0.56：探针下 866 提交全
  被拒（逐位复现 OFF）、274 保留。

**rev4 A/B（同 9 seed，V3 ON rev4 vs OFF）**：

| seed | OFF worst | rev2 worst | rev4 worst | Δ rev4 | 判定 |
|---|---:|---:|---:|---:|---|
| 14 | 0.9501 | 0.9501 | 0.9501 | +0.0000 | 保持（可行未扰动） |
| 237 | 0.2695 | 0.3258 | 0.3258 | +0.0562 | 改善 |
| 274 | 0.1231 | 0.6172 | 0.2346 | +0.1116 | 改善但大增益丢失 |
| 410 | 0.2587 | 0.1911 | 0.2625 | +0.0038 | 基本持平 |
| 479 | 0.3280 | 0.4422 | 0.3123 | −0.0157 | 微降 |
| 705 | 0.9951 | 0.9999 | 0.9143 | **−0.0807** | 新退化（PASS） |
| 725 | 0.1549 | 0.1430 | 0.2802 | +0.1252 | 改善 |
| 752 | 0.7966 | 0.8501 | 0.8501 | +0.0535 | 改善 |
| 866 | 0.8195 | 0.5530 | 0.6633 | **−0.1562** | 退化减半未根治 |

均值 worst 0.5326（略高于 OFF 0.5217）；QoS 3/9（仍低于 OFF 4/9）。

**V3-T4 最终科学结论**：五轮迭代（rev1 单帧/rev2 持久化+deficit/rev3 前瞻
门/rev3b 无持久化/rev4 relax 路由+提高守卫）在 9-seed 独立采样 A/B 上**全部
为混合结果**——有 seed 大幅改善（274/725/752/237），有 seed 退化
（866/705/410/479），QoS 计数从未超过 OFF（4/9）。**根因是闭环策略耦合**：
提交改变执行结构→观测→冻结 Student 策略的 comm/预算决策分叉→轨迹分叉；
单调门只保证提交帧精确 LP 的 t* 非降，无法保证闭环 episode 级非降。这与
V3-T1 诊断一致（失败=竞争×弱几何，L3 几何/策略决定最终可行性，结构层只能
缓解）。**结论：结构侧单调门系列（快照/前瞻/路由/守卫）无法根治 866 类
退化——瓶颈在冻结 Student 策略对结构扰动的敏感性，属于 P1-3 已确认的
"Student calibration 是 6/6 下一科学瓶颈"的闭环表现。V3-T4 保持默认 OFF
（不引入未认证退化），如实记录为混合结果。**

## 通信+感知原理审计与修复（2026-08-17，V3-T4 后审计轮）

按"审计文档与代码是否符合通信/感知基本原理"的要求，3 组并行审计
（通信原理/感知原理/数理一致性）+ 自查交叉验证。**未发现 P0 物理违规**
（1 W 预算、双基地 Friis 1/(R_tx²R_rx²)、P_D=Q(Q⁻¹(P_FA)−√D)、i≠j 硬约束、
DD 门一致、强对偶 ~1e-15 均成立），但确认 3 项 P1 理论不一致并修复：

### P1-1 修复：公共尺度量化在 η 主导时压碎 π（数理审计核心）

- **问题**：`distributed_owner_values` 旧实现 `scale=max(max(pi),max(eta))`
  共尺度归一——PASS seed 的 η_max 达 60+、π~0.25 时，π/scale≈0.004 在 6-bit
  （63 级）下量化为 0，V_jq 全 0、argmax 完全退化；docstring 声称的
  "preserve the argmax ordering" 在 η 主导时是错的。
- **修复**：**双尺度量化**——π（单纯形，天然 [0,1]）与 η（预算对偶，可任意
  大）各用自身 max 归一量化，再以精确有理缩放重建公共物理尺度
  `V=Σ_i b_i[(π_q·s_pi)a_ijq−(η_i·s_eta)]_+`。每个价格在其自身尺度上携带
  相同的相对量化误差 Δ/2，互不压碎。实测：η 主导场景 V 从 0.149（旧，π 被
  压）恢复到 0.765（新，精确 0.749），argmax 保持。
- **测试**：`test_two_scale_quantization_survives_eta_dominance`（新增）。

### P1-2 修复：comm_bits 公式低估约 K 倍 + 增益量化口径（通信审计核心）

- **问题**：旧公式 `K·(Q·6+6)+K·6+6` 每 TX 只计 Q 个增益，但 owner_values
  需广播对**全部 (owner j, target q)** 组合的增益（K·Q 个）。诚实量
  `K·(K·Q·6+6)+K·Q·6+2·6=1560 bit/轮` vs 旧 294 bit（低估 5.31×）；且
  1560 bit 在 L0 模型下（1 km、168 kbps）串行化 ~9.3 ms **> 5 ms deadline**
  ——诚实核算下价格交换在典型距离物理不可达，旧公式恰好掩盖了这一点。
- **修复**：`distributed_owner_values` 改为诚实公式（每 TX 广播 K·Q 增益 +
  价格、per-owner per-target bid、两个公共尺度），docstring 记录 deadline
  可行性结论；`price_iteration` 的 comm_total 现被 env_core 计入
  `_v3_comm_bits_total` 指标（诚实核算可见；价格消息尚未由物理 U2U Token
  信道承载——已文档化为显式边界）。
- **测试**：`test_comm_bits_bounded`/`test_distributed_owner_values_within_quantization`
  更新为诚实公式。

### P1-3 修复：V3 结构层 worst-only 路由（感知审计核心）

- **问题**：V3 的 deficit 路由/单调门/近端守卫只用 0.60 单地板（d_req=11.18），
  而 QoS 判定是 steady≥0.80 ∧ weak3≥0.70 ∧ worst≥0.60 三地板；worst∈
  [0.60,0.80) 的状态被 V3 视为"可行"而放弃——只救最差目标、不救 weak3/steady
  （max-min LP 对偶 λ* 只钉最差目标，结构层天然看不见 weak3/steady）。
- **修复**：`_lex_key`/`monotone_commit`/`price_iteration`/`forward_geometry_gate`
  支持 `d_floors` 多地板向量（可行性=三地板全满足；violation=Σ_floor Σ_q
  max(d_floor−D_q,0)）；env_core 从 `task_constrained_qos_floors` 读取三地板
  P_D 门槛映射为 d_floor 向量（11.18/13.07/15.46）传入。deficit 路由仍以
  worst 地板为主门槛（max-min LP 自身目标），提交门/前瞻门验证全地板。
- **测试**：`test_monotone_commit_multi_floor_sees_weak3_steady`（新增，
  单地板接受/多地板拒绝的区分用例）。

### 附带修复（P2 一致性）

- `compute_dd_effectiveness` 的 g_min 参数从未在函数体内使用（门在调用点）
  → docstring 修正为"由调用方施加门"（`otfs.py`）。
- `monotone_commit` 参数顺序陷阱（`forward_geometry_gate` 传
  (candidate, baseline)）→ 加显式注释防止按名误改。
- `scarcity_prices` 生产代码加强对偶断言 `Σ_i b_i η_i == t*`（容差
  1e-6·max(1,t*)）——使"对偶一致"成为可执行契约（随机实例验证 ~5.8e-15）。
- **近端守卫量级同步**（P2-1）：`v3_min_improvement_frac` 默认从 0.02 提到
  **0.05**——rev4 A/B 已验证 0.05·d_req≈0.56 拒 866 边际提交（+0.507）、留
  274 实质提交（+1.995）；旧默认 0.224 会放过 866 类边际扰动（文档与代码
  此前不一致）。
- **owner_values 的 b_i 加权入文档**（P2-3）：`V_jq=Σ_i b_i[π a−η]_+` 的
  b_i（UAV i 感知预算瓦数）明确写入 docstring（此前文档省略；该式是乐观
  单边估计，LP 预算跨目标共享）。
- **前瞻门外推改 3D**（P2-6）：`forward_geometry_gate` 的 uav/tgt 位置传
  完整 3D（UAV 高 20 m、目标在地面），消除 2D 距离低估导致的 α² 乐观。
- **系数重建显式 i≠j 掩码**（P2-7）：`_per_watt_coefficient_from_entries`
  显式跳过 i==j（防御性；当前无路径产生自环，但 owner 值与固定 owner LP
  均依赖张量无 i==j 支撑）。

### DD 门感知前瞻门（P1-2 感知，审计第二轮修复）

感知审计 P1-2 指出：`forward_geometry_gate` 外推几何时**只重缩放 α²、冻结
DD 门支撑**——门边界翻转在平滑 Friis 外推中不可见，违反 §6.5"门边界
fail-closed 精确重算"处方。修复：

- `forward_geometry_gate` 新增可选 `g_dd`/`g_min`/`dd_L` 参数：每外推步用
  `dd_gate_active_set_certificate`（`power_staleness.py`，Lipschitz 证书）
  判定 `|g_dd−g_min| > L_g·r`（r = 累计外推位移）；**不确定边 fail-closed
  拒绝**（可能跨门 → 平滑外推无效），认证边保持门恒定后做 α² 外推。
- env_core 从 deflection entries 重建 (K,K,Q) g_dd 张量、用
  `dd_gate_position_lipschitz_constant`（M/Δf/T_sym/N/fc/v_max/target_v_max/
  r_min 取当前物理参数）计算 L_g，传入 gate。
- **机制解释**（为何 rev3 前瞻门实测零拒绝）：1 帧外推位移
  （v_max·dt≈0.5–2 m）远小于门翻转阈值（L_g·r≈10⁻⁴–10⁻²），证书认证门
  恒定——不是门失效，而是位移确实小；866 类退化的真实机制仍是闭环策略
  轨迹分叉（V3-T4 结论不变）。
- **测试**：`test_forward_gate_dd_uncertain_edge_fails_closed`（新增）——
  大位移 × 大 L_g 使所有边不确定 → 门 fail-closed 拒绝；小位移全部认证。

**验证**：V3 单测 39 项全过（auction 16 + structure 8 + live 15）；相关回归
45 项全过；全量测试结果见下。**未改动的记录项**：V3 价格消息未接入物理 U2U
Token 信道（P1-1 通信——接入需将价格流量纳入 L0 payload/功率/时延核算，属
下一轮；已通过 `_v3_comm_bits_total` 诚实记账暴露量级）、χ_rep 帧条件性
（P1-3 感知——6/6 为 U2U-only 场景 χ_rep=1 恒等，实际不生效）。

**验证**：V3 单测 38 项全过（auction 15 + structure 8 + live 15）；相关回归
32 项全过；全量测试结果见下。**未改动的记录项**：V3 价格消息未接入物理 U2U
Token 信道（P1-1 通信——接入需将价格流量纳入 L0 payload/功率/时延核算，属
下一轮；已通过 `_v3_comm_bits_total` 诚实记账暴露量级）、前瞻门外推丢失
DD 门/χ_rep（P1-2 感知——外推几何只重缩放 α²，门边界翻转不可见，建议后续
复用 `dd_gate_position_lipschitz_constant` 证书）、χ_rep 帧条件性（P1-3
感知——6/6 为 U2U-only 场景 χ_rep=1 恒等，实际不生效）。

## advice/016：Closed-Loop Task-Regret 主线（2026-08-17，前两步落地）

advice/016 建议把系统从"结构预测准确率"转向"**闭环任务 regret 可控的分布式
跨基数结构学习 + 通信可保持决策不变的最小信息量**"。当前阶段明确关闭 V3
dual-auction（live 无收益+通信不可达）、追 8/8 94%→100%、换 GNN/MARL/ADMM。
本步落地前两个理论可验证项（实现：`uav_isac/coordination/structure_regret.py`、
审计：`tools/audit_structure_regret.py`）：

### 1. 结构代理误差 → max-min 检测能力损失的 Lipschitz 界

对偶形式 `t*(A) = min_λ Σ_i b_i max_q λ_q a_iq` 下，由
`|min f − min g| ≤ sup|f−g|` 与 max 在单纯形权重上的 1-Lipschitz 性：

```text
|t*(A) − t*(Â)| ≤ Σ_i b_i max_q |a_iq − â_iq|      (加权 max-norm 界)
```

**QoS 保持条件**：若 teacher margin `m = t*(A) − d_req > 0` 且误差界严格低于
m，则 Student 结构不可能把 worst 地板推到门限以下（`ε_struct < m ⟹ floor
preserved`）。**验证**：300 个真实 6/6 帧（teacher trace），强扰动（丢最强
TX 边/目标）下**界零违例**（worst violation = 0.0）——理论正确可部署；界偏松
（bound/actual 均值 ~200×，安全证书性质）。

**真实数据关键发现（修正"Student 是唯一瓶颈"的误判）**：这些 6/6 失败帧的
**teacher 结构本身 max-min t\* 均值 −7.42、0% 帧过地板**（d_req=11.18）——
即使完美结构在这些帧也达不到 worst 地板，瓶颈在**几何/预算**（弱几何 × 资源
竞争），与 advice/016 §1"Student 结构误差 × 弱几何 × 闭环策略敏感性"三重机制
一致，Student 校准是必要但不充分。

### 2. Decision-preserving Token：最小 bit 下界 + 事件触发

对两个候选 score `s1>s2`（margin `Δ`）与量化动态范围 `R`，B-bit 均匀量化
每 score 误差 `ε_B ≤ R/(2(2^B−1))`，排序保持条件 `Δ > 2ε_B` 给出

```text
B ≥ ⌈log₂(1 + R/Δ)⌉
```

**事件触发**：`Δ > 2 E_stale(h) + 2ε_B ⟹ 不发 Token`（漂移界 E_stale 来自
运动/CSI/AoI）。实现：`decision_preserving_bits` / `should_transmit` /
`preserved_without_transmission`。**真实 6/6 数据**：844 个 per-target owner
决策上 mean 6.6 bit、min 2 bit、max 13 bit、1.2% 可完全抑制——**margin 大的
决策用更少 bit，结构决策不确定时多发**（advice/016 §13 主张成立）。
**固定 6-bit vs 自适应 bit 对比（真实数据，论文级量化）**：固定 6-bit 下
**50.7% 的 owner 决策未认证**（margin ≤ 2ε₆，排序可能翻转），自适应 bit 下
**0% 未认证**（每个决策用恰好足够的 bit），代价仅多用 10.2% 总 bit——固定
量化"省 bit"是以半数决策无排序保证为代价，这是"保持结构决策不变所需最小
通信量"的可推导形式（advice/016 §14 的核心指标 P_flip = P(Ŝ_q ≠ S_full)）。
**实际 argmax 翻转（更强证据）**：在真实 6/6 帧上，固定 6-bit 的 owner 决策
**28.5% 实际翻转**（260 帧中 74 帧 argmax 与全精度不同）——量化误差不是
理论风险而是真实发生的决策错误；`adaptive_owner_bits`（`dual_priced_auction.py`）
按 margin 选择每目标 bit（`B = ⌈log₂(1+R/Δ)⌉`），保证 0 翻转（协议级 per-value
宽度 = max_b，固定 6-bit 公式 `K·(K·Q·6+6)+K·Q·6+12` 是 max_b=6 的特例）。

### 3. Dual-weighted 训练权重（advice/016 §6，供未来 Student 校准）

包络定理 `∂t*/∂a_iq = π_q·p_iq` 给出 edge 级任务敏感权重 `w_iq = π_q·p_iq`。
**真实数据**：瓶颈目标（min gain 和）的对偶权重为全局均值 **4.2×**——训练
mass 正确集中在瓶颈目标；`task_regret_loss` 提供字典序
（R_γ = feasibility-flip 计 1、R_t = 残余 max-min 损失）。注意 max-min 分配
下 p_iq 对非支撑边为 0 → 权重稀疏（已加 `min_power_floor` 避免零梯度退化，
实测并文档化）。

**验证**：`tests/test_structure_regret.py` 10 项全过（Lipschitz 界 300 随机
实例零违例、单边扰动紧度、QoS 保持、bit 公式、排序保持、事件触发、dual 权重
瓶颈集中、task-regret 可行性翻转）。**未接入在线路径**（advice/016 §21：
对偶价格进训练/离线监督，不强行为在线通信协议）。

## 深度代码审计（2026-08-17，P0 闭环）

对 `uav_isac/` 全量模块做 6 组并行静态逻辑审计（trainer/agents/coordination/
environment/evaluation/physical-utils），35 条 findings 中 **12 项真实逻辑问题已
修复**、6 项记录不改；本节即为该轮工程审计的保留摘要。要点：

- **观测解析器错位（networks.py）**：legacy `_parse_one` 逐目标交错读取 vs
  block-wise 布局（初始提交起），修正 parser 强制启用；**核实所有论文/认证配置
  均用 corrected parser，无认证结果受影响**（新增 2 项回归证明）。
- **隐蔽性 slack fill 违规（capability.py）**：intercept 行存在时 slack 填到
  argmax gain 可精确违反 P_D^I ≤ ε → headroom-aware fill（镜像 intercept_power）。
- **CT 目标速度误用（env_core 6 处）** → `get_velocity()`。
- **跨 episode 状态泄漏（4 项 reset 缺失）** → 统一清理（A6 确定性）。
- **trainer LR 3e-5 死区 + tail_fraction 指标 ÷n** → 修复。
- **3 处 fail-open 校准验证（inf 界报零失败）** → fail-closed。
- 其余：n5/quantized 崩溃守卫、env_wrapper 观测空间/render、belief_cov_trace、
  K=1 oracle、SINR 索引、CT 反弹。
- 记录不改：deflection χ_rep 每 (i,j) 重抽、能量记账占位角色、maxmin 占位价格、
  residual_actor 遗留、oracle-α 死功能、buffer 尺寸——均会作废已认证结果或属
  遗留，文档化。

当时验证基线为 **927 passed / 1 环境依赖失败**；该历史数字已被下方 2026-08-20
可信执行审计取代。

## 测试基线

2026-08-20 G2-1A 基础设施工作树按 Windows/MKL 稳定边界分片验证：非 belief
`1036 passed`、belief `14 passed`，合计 **1050 passed**、无断言失败。单进程仍可能在
`numpy.linalg.eigvalsh` 内原生中止，因此只报告分片事实。

## 可信执行前置优化（2026-08-20）

在继续提高 QoS 前，先闭合六类会让“优化结果”失去科学含义的基础缺陷：严格配置
schema、动作/环境/感知 RNG 所有权、源码与检查点 provenance、Gate episode 完整性、
隐蔽功率 fail-closed，以及结构控制器精确快照。危机 Gate 的 AUC/平衡准确率和 tail
Pearson 相关按定义直接计算，消除非必要 sklearn/MKL 路径。

本轮**不产生新的性能主张**。正式 Gate 重放显示：D1.5 LCB、6/6 V3-C0 和 6/6
Gate 1 LCB 仍为 `DISCLOSED FAIL`；Gate 1 虽有 QoS 点估计 `0.780`，但 Wilson LCB
`0.689<0.70`。下一轮创新必须围绕独立 seed 上的尾部风险，同时显式满足 Shannon
容量/时延/bit、单 UAV 1 W 联合预算、DD 支撑与三项检测地板；不得用开发 seed 追尾调参。

## G2-1A：多 seed 聚合修正与 5-seed falsification（2026-08-20）

逐 seed 隔离器原先在合并时复制首个 worker 的标量遥测，这会令通信 bit、时延、功率以及
QoS/Wilson 在多 seed 下失真。现改为以 episode 数组为统计事实源：尾部指标和三地板 QoS
完全重算，均值型遥测跨 seed 求均值，最大误差型遥测取最大值，输出按 seed 确定性排序。
Wilson 下界额外投影到概率区间 `[0,1]`。该修正只改变汇总正确性，不改变单 episode 动力学。

新增独立审计器 `tools/audit_g2_1a_batch.py`，同时检查：概率有限且在 `[0,1]`、seed 唯一且
数组对齐、三地板与 Wilson 可重构、通信 delivery/deadline 概率守恒、总感知功率匹配
`K×0.0251 W`，并强制 6×6 CE 与 V3-C0 使用相同有序 seed 后才能计算配对差分。

5-seed falsification 没有支持立即启用 CE：CE 相对 V3-C0 的 worst 配对均值差为
`-0.08036`，只在 `2/5` seed 获胜。由于样本已查看且 `n=5`，不做显著性声明、不据此调参；
其价值是把后续创新问题收敛为“如何在弱几何和高通信负载下形成可证的非伤害 active set”，
而不是继续无条件扩大 Student payload 或价格迭代轮数。CIS-ISAC live 路径继续冻结到
G2-1B 之后。

## 跨尺度 worst 深度审计（2026-08-20）

审计否定了“4×4→6×6→8×8 的差只由网络规模造成”这一过强解释。三组配置虽然节点密度
相差不足 0.3%、`K/Q=1`，但使用不同 seed bank、checkpoint 和控制器；4×4 关闭解析
power/movement，8×8 使用 40 帧 movement lookahead。因此当前表格不是 scale-only A/B。

机制证据支持三层结论：

1. **几何尾部与 bank shift**：精确历史 100-seed bank 的初始 bottleneck-matching 均值为
   4×4 `363 m`、6×6 `523 m`、8×8 `555 m`；三个 bank 没有按物理难度配平。
2. **max-min 水床效应**：5-seed 中单个目标占总 sensing power 的平均最大份额由 4×4
   `0.28` 增到 6×6 `0.86`、8×8 `0.90`；解析层把多数目标等化到同一低 worst，说明
   极弱双基地目标消耗共享预算，而不是 P0 漏目标（基线 P0 coverage 均为 1）。
3. **cap-aware 上界**：每尺度一个 episode、每 30 帧采样的机制探针中，联合结构+功率
   worst 为 4×4 `0.982`、6×6 `0.352`、8×8 `0.123`。后两者在合法 sensing PA cap
   下仍达不到 0.60，支持“几何尾部×PA cap×联合耦合”为绑定机制；样本不足以形成 Gate。

审计还发现并修正 physical oracle 的口径漏洞：旧实现只用 1 W 联合预算减通信 reserve，
未施加 0.0251 W sensing PA cap，会错误给出 worst=1 的不可行上界。新 API 保留兼容默认，
但当前 trainer 必须显式传入 sensing cap。该修正改变的是诊断可信度，不改变在线控制器。

6×6 CE 的额外退化不是纯规模效应：seed 237/866 的 P0 target coverage 从 1 降到 0，
对应 `P_D≈P_FA=0.001`；CE 平均 payload 约 5736 bit/frame，且所有五个 seed 的最终
worst-nearest 都比 V3-C0 更远。后续优化应把 CE transport/admission 与运动轨迹影响作为
独立故障域，不得用扩大 payload 掩盖。

## Capability-Aware Hierarchical Scaling 路线冻结（2026-08-20）

依据跨尺度审计，创新主线从孤立的 CIS 前移为“任务能力驱动的分层规模化控制”。新增
`scale_capability.py` 实现 cap-aware budget、单对双基地 `Gamma_geo`、coupling-relaxed
逐目标 ceiling、`C_max/HHI` 以及 shadow-only 四区域 router。相关 6 项新测试连同既有
max-min/scale 审计共 37 项通过。

路由采用双侧证书而非单一启发式分数：fixed allocation 达标是 Hold 的构造性证书；合法
joint allocation 达标是 L2 可修的构造性证书；relaxed upper ceiling 未达标是必须 L3 的
不可行证书。启发式 joint oracle 未找到解但 relaxed ceiling 尚可行时进入 Region U，不得
自动启动 live L3 或宣称物理不可达。

路线顺序正式改为：

```text
G2-1A -> G2-1B -> S0-P natural physical scaling
                  -> S0-M difficulty-matched scale-only
                  -> S0-F common-controller factorial
                  -> S0-O cap-aware oracle attribution
                  -> S1-G bistatic geometry shadow
                  -> S1-CIS robust-slack/active-set shadow
                  -> S2 evidence-selected live mechanism
```

其中 S0 全部禁止调算法。S1-G 只研究基于 `R_tx^2 R_rx^2`、DD 与 relative mobility 的
bistatic geometry mechanism，不扫描 H；若未来需要 horizon，使用
`rho_H=H v_max dt/L(K)` 的无量纲定义和 receding horizon。CIS 只在 Region II 有理论
意义；Region III 优先 L3，Region I Hold，Region U 保持 shadow/fail-closed。

## G2-1A 10/100 分步性能判定（2026-08-20）

冻结四系统分别从 5 扩到 10 个历史 seed，新增 20 个隔离 worker 全部完成，无 crash。
执行正确性符合预期；性能出现明确分层：4×4 worst `0.9245`、QoS `10/10`；6×6
V3-C0 worst `0.2898`、8×8 V3-C0 `0.2697`，两者均仅 `1/10`；6×6 CE worst
`0.1784`、同样 `1/10`。因此整体系统尚不满足跨规模三地板目标，不能以 4×4 成功外推。

机制方向继续支持 capability-first 路线：6×6/8×8 基线 P0 coverage 均为 1，8×8
delivery 约 0.999 且 deadline violation 约 0.001，仍然低 worst；其单目标最大 sensing
power share 分别约 0.861/0.889。CE 则 delivery 降至 0.733、deadline violation 升至
0.267，10 个 seed 中 4 个 P0 coverage=0、1 个仅 0.2；CE−V3 paired worst 为
`-0.1114`，仅赢 `2/10`。故基线规模失败首先属于 physical capability，CE 另有
transport/admission collapse；两类问题继续分离，不做追尾调参。

## G3-A/G3-B：能力路由归因与 owner 指标否证（2026-08-20）

附件提出的 Capability Router + Bistatic Geometry Control 被拆成先证伪、后实现：先在
5 个已查看开发 seed、750 帧 post-G2 trace 上比较物理描述量，再对每 episode 每 30 帧
抽样运行 cap-aware 同几何联合 oracle。该顺序避免把一个有物理外观但缺少闭环解释力的
指标直接写入 L3。

G3-B 的帧级 Spearman 为：single-pair capability `0.536`、owner-consistent capability
`0.184`、fully relaxed capability `0.310`、最佳距离代理（matching）`0.440`。三种能力量
的理论次序在 750 帧上零违例，但 owner-consistent 指标没有达到“优于距离代理”的方向性
预期；episode 仅 n=5，也没有支持信号。因此关闭“以 `C_q^owner` 为目标直接实现
capability-rate SOCP”的分支，保留它作为 shadow descriptor。

同时修复 router 的任务语义：旧逐目标 `D_req,q` gauge 不能等价表示 worst/bottom-3/
average 三地板，新 `r_task` 直接对三项统计量取最大归一化比；它只是 task-feasibility
ratio，不是正齐次 resource gauge。G3-A 25 帧得到
`I=4、II=21、III=0、U=0`，所有 21 个固定结构失败均由同几何 joint witness 构造性救活；
没有 relaxed 上界失败帧。当前开发证据因此把优先级从 L3 几何重构改回 L2 原子结构—功率
联合部署。该结论只决定下一机制方向，不替代 G2-1A 继续扩至 100，也不授权 live 修改。

## G4-A：Minimum-Intervention Exact Oracle（2026-08-20）

新增 TS-AJR 的第一阶段 shadow oracle，把角色、owner、support 和 edge power 放入同一个
MILP，并用 conservative PWL 完整表达 worst/bottom-3/average。字典序依次最小化真实依赖
闭包、prepare payload bit 和 sensing power，不用任意加权和。对偶价格尚未进入本 Gate；
未来只能用于嵌套 block 排序，不能替代 exact solve 或 commit admission。

首次运行暴露了旧 PWL 的动态范围退化：当逐目标乐观 ceiling 达数万 Deflection 时，自适应
曲率步长直接跨到上界，只剩一条极长 chord，导致 20/21 个已知 joint witness 被错误判为
不可行。加入有单调性证明的 `P_D=0.999` 饱和常数尾段后，21/21 均获 MILP optimal
witness。该过程说明 falsification Gate 同时审计了证书本身，而不是为预期结论放宽约束。

最终 21 帧结果：role flips 中位数 `1`（18 帧为 1，2 帧为 2，1 帧为 0）；owner changes
中位数 `0`；edge toggles 中位数 `1`；participants 中位数 `2`；closure cardinality 中位数
`3`。16/21 帧同时满足 role flips<=2、owner changes<=2、participants<=3，支持
dependency-closed nested local block；但 seed 103 多数帧 closure 为 6--9，否决固定
Top-2/Top-3 能覆盖全部失败的假设。下一步是 G4-B 嵌套扩张 shadow，不进入 live。

## G4-B：Nested Block Discovery（2026-08-20，部分通过）

实现 task-ratio active-branch 子梯度排序、role-aware dependency closure 和严格嵌套的受限
MILP。首次实现曾把所有 block participants 都当成 role-changeable，导致中位 block 立即
扩到 5 UAV×6 targets；审计后拆成 edge participants 与 role-change set，只有真实角色
候选才传播全部 incident dependencies。

修正后比较两个无连续超参数的 proposer：

| 排序 | 最终恢复 | full block 前恢复 | 匹配 exact 最小 closure | ≤3 UAV 且 ≤3 target | 总 MILP 尝试 |
|---|---:|---:|---:|---:|---:|
| task-weighted gain | 21/21 | 16/21 | 13/21 | 4/21 | 54 |
| gain / marginal closure cost | 21/21 | 13/21 | 15/21 | 4/21 | 82 |

closure-efficiency 提高最小干预命中，却降低提前恢复并增加求解次数；不存在开发集上一致占优
的选择。两者首次可行块中位均为 5 UAV×6 targets，所以当前结果是 completeness PASS、
locality-efficiency FAIL。按防追尾纪律不再组合权重或挑 seed 选择模式，也不提前进入 G4-C/
live。下一研究问题收敛为如何用可验证 cut/label 找到小 closure，而不是继续增加通信轮数。

## G4-B2b：Fixing-Conflict Deletion Filter（2026-08-20，shadow 正结果）

本地环境没有可调用的 `highspy`，且 SciPy wrapper 不返回 IIS/Farkas ray，所以没有假称实现
G4-B2a。改用 exact MILP feasibility oracle 做 fixing-deletion filter：从 full permission
开始依次恢复 target、UAV、role fixings，可行才删除；保留项都有单项删除即不可行的局部
反事实证书。该输出不保证全局 minimum permission set，也可能随删除顺序变化。

顺序删除把 G4-B 首次可行块中位 `5 UAV×6 targets` 降到 permission 中位
`2 UAV×1 target×1 role`，小块率由 `4/21` 提高到 `15/21`，minimum closure 命中由最佳
`15/21` 提高到 `16/21`。代价为 333 次 exact feasibility 调用，远高于 gain 的 54 次。
二分 group deletion 未改善输出且调用升至 471，作为负结果关闭。结论是 conflict-derived
labels 有明显定位价值，但当前算法只适合作为离线 label oracle，不能进入 live/G4-C。

## G4-B2c：Backbone 正结果、Raw Conflict Master 负结果（2026-08-20）

补充了顺序无关 exact backbone。13/21 帧存在非空全局必要权限，中位大小 2；backbone
集中在 target/UAV permissions，没有 role permission，表明“需要释放哪个目标/UAV”比
“必须翻转成哪个具体角色”稳定。seed 103/483 分别稳定指向 target 2/3，但统计仍只有
5 个相关帧/episode，不能训练或泛化。

同时实现 logic-based permission master。每个 restricted-infeasible block 产生 valid cut
`sum_{g in U\B}u_g>=1`；master 对 permission cardinality 有有限收敛与首个可行解全局最优
保证，但这一目标不等于 G4-A 实际 closure 成本。无限制试跑已表现出长时间枚举，随后冻结
为每帧 16 次 oracle 的 falsification 门：仅 1/21 收敛，20/21 触顶，321 次 master query。
预先强制 backbone 后数字完全不变。结论是 raw complement cut 理论正确但太弱，当前 CCAR
实现关闭；下一步若继续，只允许研究选择性 conflict shrinking 或等价 repair 集，不提高 cap。

## G4-B2d：Core-Guided Repair（2026-08-20，机制正结果/效率负结果）

把 B2c 的大补集 cut 收缩为 `C subseteq U\B`，并由同一个 exact post-G2 oracle 认证
`Phi(U\C)=0`。加入 `sum_{g in C}u_g>=1` 后，master 成为 conflict-core minimum hitting
set；首次 exact-feasible master 解对所声明 permission objective 全局最优。UAV/role 依赖
进入 closed-set 语义，冻结 UAV 会同时冻结其 role，避免用不合法反事实产生 cut。新增
2-core、3-core、singleton-backbone、依赖闭包和全局最优性单元测试。

G4-B2d-1 的 21 帧 sequential shrink 找到 194 个 irreducible cores，大小中位 3、最大 5，
证明 conflict sparsity 假设成立；16-call master 收敛提高到 16/21，closure 命中 15/21。
但需要 2824 次 exact feasibility，故只完成机制验证。QuickXplain 在预先规定的 3 帧 smoke
仅 1/3 收敛、316 次 exact，physics-ordered sequential 也仅 1/3、396 次；两者均比 index
sequential 的 3/3、315 次更差，关闭且不做全量追尾。

随后将等权目标升级为严格字典序 `(N_UAV+N_target,N_role)`，用 `K+1` 保证第一层完全
支配第二层，并加入单调 antichain exact inference。全量仍只有 16/21 收敛，但 closure 命中
提升到 16/21；exact 调用仍为 2822。预注册的 21/21 feasible、至少 18/21 在 16 candidate
内、closure hit >=16/21、exact calls <333 四门仅通过 closure 一项。因此不进入 G4-B3/C
或 live。保留 core/hitting-set 理论与实现作为研究基础，停止排序、tie-break 和 cap 调参；
下一步必须获得比黑盒逐项 shrinking 更便宜、同样可认证的物理冲突信息。

## G4-B2e / OLCS：下界分离正结果、Greedy Bundle 负结果（2026-08-21）

根据 B2d 的 target-conflict 漏失，将 separation 目标从 inclusion-minimal core 改为
objective-layer lower-bound raise。先修复证书语义：feasibility-only incumbent 重新做
primal/integrality 检查；只有 `status=2` 产生 infeasible certificate；time limit、数值异常
等 unresolved 状态永不产生 cut。论文口径同步降为 conservative-PWL-model certificate。

在未扰动 support-lex 最优面定义 common-zero permissions `Z(tau)`。若开放其补集仍被证明
不可行，则一条 nonminimal cut 可删除全部 `J<=tau` 解并严格抬高下界。五个预声明失败帧
的 face separation 为 5/5，通过 5 次 physical query 消灭全部 target-free 最优面；sublevel
二分共 19 次查询，把下界由 `22/37/30/29/31` 分别推到 `29/44/37/36/38`。最终 cuts 主要
由 target permissions 构成，但来源是 objective face 证书而非类型权重，机制判正。

完整 OLCS 从空 core 启动后，在 union-feasible face 使用 bounded greedy bundle。8-support
版本五帧产生 28 个 sublevel cuts、52 个 bundle cuts，407 次 physical query 后仍 0/5；
单-support 对照将查询降至 100，但 0/5 且下界长期停在 15/22。这证明多候选 bundle 有
信息但太贵，单 no-good 便宜但退化枚举。按 smoke Gate 关闭 greedy bundle，不调宽度、不
增加 16 轮上限、不跑 21 帧。下一问题若继续，应是可证明覆盖整个最优面的 master-side
partition/face-splitting certificate，而不是再枚举 support 或 shrink permission。

## R1：Interaction Width Audit（2026-08-21，否证并停止）

在 B2e 之后没有继续添加 face-splitting heuristic，而是先冻结可证伪假设：post-G2
Region-II 结构经安全物理剪枝后，应呈现远小于 K 的有效交互宽度。实现 owner+TX-subset
task modes；只有 mode 的独占资源乐观 ceiling 低于检测地板时才删除。随后对四种保真度
不同的交互图计算 exact treewidth，并预先规定本地门为 effective width 中位不超过 3、
最大不超过 4。

21 帧结果完全一致：raw target/resource target/summary incidence/mode clique 的 treewidth
为 `5/5/6/6`，有效宽度中位和最大均为 6，门失败。安全剪枝仍保留中位 793/900 modes；
所有 UAV-target incidence 都存在，HHI 约等于 `1/6`。这说明当前物理可参与关系是稠密、
均匀而非局部簇状；summary variable 只改变因子图表示，没有降低需要跨 separator 传递的
状态复杂度。

seed 103 与 566 的 mode 中位不同（662 对 794），但 exact width 相同，故不能把两者 worst
差异归因为 interaction width。按阶段门，R1 未通过即不运行 R2 task-mode equivalence、R3
dual certificate pruning 或实现 tree-decomposition solver。这是该路线的实质性负结果，
不是待调参数；后续若重启，前提是得到保持通信 QoS、功率共享、single-role、owner 与检测
地板联合可行域的解析 dominance rule，而不是按 dual/gain 阈值强行稀疏化。

## S0-A / S0-D：安全约简不足，但发现致密低能量秩（2026-08-21）

R1 后正式关闭 treewidth 表示路线。S0-A 没有实现 pairwise gain dominance，而是先证明跨
UAV context-free substitution 与独立资源偏序不相容。实现的第一条充分规则只固定严格
零增益 incidence，并将 D-F 与 D-O 分开：D-F 可筛全部零边，D-O 还要求 reference edge
为 0。新增 `forbidden_edges` 只服务 shadow exact audit，不改变 live controller。

21 帧中 96/3780 edges 被筛除，D-F 可行性和 D-O 三层 lex optimum 均为 21/21 保持；但
pruning 仅 2.54%，正系数 incidence width 仍为 6。故 S0-A certificate PASS、natural
sparsification FAIL。不能因为零误删就声称有复杂度收益，也不进入 width solver。

S0-B 需要 search state 给出每个替代 UAV 的 guaranteed residual；当前 root 下其他目标可以
耗尽任一独立 PA，所以保证余量为 0。使用 incumbent/current slack 会失去“对任意 branch
completion”保证，当前不这样做。S0-C dual/probing 尚未启动。

作为并列的物理诊断，S0-D 从几何重建理想路径 completion，验证其 126/126 exact rank-1；
single-role 的非对角 mask 后，实际矩阵虽通常满代数秩，但 95% energy rank 在 109/126
cases 不超过 2。report 阶段在当前配置不改变谱，DD 仅把 17 个 case 推到 rank 3/4。
这支持新的可证伪方向：构造 dense low-energy-rank 的 conservative capability envelope。
它不是直接做 truncated SVD；只有 envelope 对每目标 Deflection 给出可审计上下界，并在
三 QoS 地板与 lex optimum 上过门，才可进入 solver。

随后没有采用 SVD 截断，而是利用 bistatic inverse-range law 推出 exact representation：
receiver-only report 只改变外积右因子，single-role 删除对角线，DD gate 形成 sparse exception。
126 个真实矩阵的最大相对重建误差 `8.08e-16`，exception ratio 2.54%。这将候选创新从
“再做一次低秩近似”收敛为“physics-exact factorized capability messaging”。下一 Gate 应
比较 dense coefficient payload 与 factor+exception payload 的 bit 数、量化后 Deflection
上下界和 decision preservation；在完成前不接入 live，也不声称加速 exact MILP。

## M0–M3：精确因子消息与静态任务证书（2026-08-21）

研究主线从 sparsification 转为 capability-information scaling。M1 先冻结 semantic record
layout，避免把 scalar count 当 bit 结论；M2-0 使用 outward float32 log range 与 interval
bins，禁止中心值重建后再用经验误差。3–16 bit 全部 0 coefficient containment violation。

M3 沿 `A interval -> D interval -> P_D interval -> three floors` 做三态分类。对 G4-A witness
及四个低功率反事实，所有 bit 均 0 false-feasible/false-infeasible。nominal minimum-power
witness 因零 QoS margin 永远 unresolved，证明“只加 bit”不是完整方案。随后固定三档 design
headroom 并重新求 exact 三层 lex witness，而不是事后缩放功率。随后 M4 的 exact gauge
balance 去除了人为动态范围；更新后 +0.005/12 bit、+0.0025/12 bit 和 +0.01/10 bit 均达
21/21 CERT_FEASIBLE。旧的 14/16-bit 需求不再作为正式结果。

这一结果支持 `execution headroom <-> message precision` 的任务级联合设计。M4-A/B 的公平
dense baseline 见下节；L1/L2 decision interval、M2-1 AoI 与 M5 Shannon/FBL transport
仍未执行。当前算法保持 shadow，不替换 Token/owner/price protocol。

## M4-A/B：同任务认证率下 factor 胜过 dense-log（2026-08-21）

在进入 baseline 前先发现并修复 gauge artifact。利用外积缩放自由度对齐 TX/RX log centers，
精确保持 `A`，却把 8-bit interval 宽度缩小约 8.5 倍。该改动不是调量化范围追结果，而是
消除 factor 表示非唯一性导致的无意义通信成本，并增加 gauge-invariance 单测。

随后实现 dense-linear、dense-log、factor-log 三种 closed-interval codec。三者共享 header、
DD mask/list，并只编码 active dense entries；都使用 adaptive first-certified rate。M4-A 所有
containment 与 false-cert 硬门为零。M4-B 在三档 headroom 的 21/21 CERT_FEASIBLE 共同点上，
factor 中位 payload 为 995/851/851 bit，最强 dense-log 为 2235/1891/1891 bit，比例
0.445/0.450/0.450。factor 不只是赢 dense-linear，因此 communication-value Gate PASS。

本结果仍是 centralized-scale semantic shadow，只证明 capability information 的 task-equivalent
rate 优势。M4-C/D 不随之自动通过；尤其不能用功率向量逐元素相同定义 L1 decision，也不能
用 index tie-break 定义 L2 winner。下一步必须分别审计 interval gauge route 与 lex-optimal
equivalence set；若失败，应保留 M4-A/B 而不包装成完整 coordination protocol。

## M4-C-light：从固定精度消息转向决策裕量驱动（2026-08-21）

M4-C 不再问“量化后是否还原同一个 power vector”，而只问固定结构的正确 L1 route 是否保持。
核心序关系是

```text
A^- <= A <= A^+  =>  gamma(A^+) <= gamma(A) <= gamma(A^-).
```

这使 capability message 成为三态 decision interval：区间整体在 1 左侧则 L1 feasible，整体在
1 右侧则 fallback，跨越 1 则继续加 bit 或 unresolved。实现审计还暴露并修复 PWL gauge 的
早退矛盾：`gamma>1` 是 gauge 的有效输出，不应因当前物理 budget 不足而返回 `None`。

21 帧各构造六个 `gamma*` 边界，共 126 场景；3–16 bit coarse-to-fine 后 126/126 正确认证，
0 containment violation、0 gauge-bracket violation、0 false route。margin=0.10 时首次认证主要
为 6–8 bit，margin=0.03 为 8–10 bit，margin=0.01 主要为 10 bit；描述性
`rho(|1-gamma*|,B*)=-0.824`。这给出当前最明确的算法演进：

```text
固定广播精度
  -> outward factor interval
  -> capability-gauge decision interval
  -> 按 |1-gamma*| 自适应 refinement
```

该正结果只关闭 M4-C-light，不完成 over-air 协议。下一步是 M4-D-light：冻结有限候选集合，
逐层比较 dependency closure、prepare payload、sensing power 的 interval lex-optimal 等价集；
不得追求 solver basis 或浮点 power vector 完全相同。其后才进入 fixed physics scale/scale
negotiation、decision rate、AoI/FBL 与 live transport。Student/MAPPO 与 G2-1A 继续独立。

## M4-D-light：候选集退化暴露 provenance drift（2026-08-21）

按预注册思想，候选只来自可重放的 G4-A 和两条 G4-B2b filter；依赖缺失 `physical_pd` 的
G4-B ladders 不用 `local_pd` 冒充恢复。候选先冻结、去重，再以 dense-log/factor-log 区间逐级
消除 competitor。固定结构的第三阶段新增 exact LP：在每
UAV sensing cap 与三地板下最小化总 sensing power，利用 gain 单调性形成 power interval。

算法本身的 containment/power bracket/nonoptimal safety 计数均为 0，但候选审计立即退化：
21 帧 candidate count 全为 1，capability-dependent case 为 0。追查发现旧 G4-A artifact 的
20/21 nonzero closure、median 3，在当前最终物理归一化和代码下重放为 0/21、median 0；同一
5 个 exposed seeds 的全部 25 帧也都是 fixed-structure power feasible。旧 trace 缺少当前
deployed power 与 environment `physical_pd`，不能完整重建 deployed performance。

因此 `0 bit` 不是 easy L2 result，而是无 competitor 的 vacuous result。M4-D 正式记为
`INVALID_INPUT_DEGENERATE_CANDIDATE_POOL`：safety 子模块 PASS，candidate/decision/communication
Gates FAIL。该负结果否决了直接用历史 G4 repair 产物关闭 M4 的计划，也说明 provenance 必须
先于 decision-rate audit。下一步冻结为：

```text
fresh current-model exposed trace
  -> explicit deployed sensing power + environment P_D
  -> deployed / L1 / L2 / L3 re-triage
  -> only genuine L2 frames enter M4-D
```

禁止通过随机结构、one-toggle 或扩大旧候选池制造 capability-separated cases；若新鲜开发集
仍没有 L2 帧，则 M4-D 路线应自然关闭，论文只保留 M4-A/B/C 的表示与 L1 结论。

## P0：Current-Model Layer Provenance Audit（2026-08-21，已实现契约）

路线没有进入 M5，而是先修证据生成链。新增 schema-3 `teacher_trace`：原有结构/物理字段
继续兼容，同时新增部署 sensing/comm power、UAV/目标速度、显式位置、per-watt coefficient、
frame sensing budget、mean/weak3/worst 以及嵌入式 provenance JSON。运行绑定采用确定性源码
snapshot hash、resolved-config hash、当前 deployed actor tensor hash与辅助 checkpoint 清单的
bundle hash；物理口径另做 canonical JSON hash。

新增 fail-closed Gate 会检查 schema、必需字段、共同 frame axis、有限非负功率和四类 64 位
SHA-256，并重算 physics hash。对历史 selection20 主 trace 实测得到：

```text
gate:              FAIL
legacy schema:     1
required schema:   3
scientific status: BLOCKED_BY_LAYER_PROVENANCE
```

这使 M4-D 的路线状态从“尝试失败”精确化为“尚无合法当前层输入”。下一阶段 P1 只在自然
current-policy 轨迹上按 D→fixed-L1→joint-L2→relaxed-III→U 顺序分类，并预注册
`f_D,f_L1,f_L2,f_III,f_U`、`Delta P_L1=P_deployed-P_min,fixed`、
`Delta_struct=gamma_fixed-gamma_joint`。不得按难度筛选自然样本；若需要 stress set，必须另标
diagnostic。若 `N_L2=0`，M4-D 正式关闭且 L2 降为 rare emergency layer，而不是继续制造候选。

## P1：fresh natural layer re-triage（2026-08-21，路线解锁）

同一 5 个 exposed seeds 已按当前代码重新产生 750 帧 schema-3 trace，P0 PASS。先用
`stride=30` 的 25 帧得到 `D/L1/L2/III/U=10/0/15/0/0`；不改阈值，扩大为同一系统抽样
规则的 `stride=15` 共 50 帧后得到 `17/1/32/0/0`，方向稳定。全部来自自然 current-policy
trajectory，没有 difficulty filtering。

每个 L2 witness 同时通过 conservative chord、真实 Gaussian-Deflection 三地板、角色互斥
和 per-UAV residual budget 验证。32 帧最小 exact worst/bottom-3/average 为
`0.610000/0.710002/0.810055`，最大预算违反 `1.53e-16 W`。fixed gauge 最小
`1.0181>1`，repair-witness gauge 最大 `0.9931<1`；closure 为 3--8，中位 5，因此不是
容差误分或 singleton/no-op。

路线从 `BLOCKED_BY_LAYER_PROVENANCE` 转为 `RESUME_ONLY_ON_OBSERVED_L2`。下一步 M4-D 只冻结
fresh L2 帧上由真实定位算法产生的候选；exact witness 只作裁判，不得泄漏为候选定位特征。
旧 G4/R1/S0 标签继续保留为历史条件证据，不与 fresh P1 合并统计。

## M4-D0/D1：候选 provenance 与 sufficiency（2026-08-21，停止于定位）

D0 增加 full-task capability gauge 的 canonical budget dual `eta`，并用
`sum eta_i b_i=1` 自检。oracle-free locator API 没有 exact witness/oracle/MILP/closure 参数；
它对全部 50 帧先运行，随后冻结候选 JSON、code hash、state hash 和 timestamp。D1 exact MILP
后启动，仅作裁判，temporal-separation Gate PASS。

D0-v1 whole-role completion 在 32/32 帧均为 `MOVE_GRAMMAR_GAP`；候选最优 closure 比全局高
1--6。由于 whole-partition completion 本身违背 local grammar 设计，v2 仅作一次原则性修复：
增加 exhaustive atomic owner/support edit 与 global role swap，然后重新冻结、重新裁判。

v2 得到 Pool A `0/32` coverage，Pool B `9/32` coverage；分类为 9 个
`LOCALIZATION_GAP` 和 23 个 `MOVE_GRAMMAR_GAP`。D2 eligible=0，因此不做 bit sweep，不能
blame factor communication，也不把 miss 记为 infinite bits。

按 Stop Rule，M4-D 在此关闭为 candidate-localization negative result；不在相同 5 个 episode
上继续加 heuristic。未来若重开 locator，必须在独立 development split 上预注册，不能复用
本轮 exact feedback 追尾。

## M4-D2-development：联合结构语法重构（2026-08-22，开发回放）

针对审计指出的 `MOVE_GRAMMAR_GAP`，新增 intervention-aware target families：保 owner 的
support add/remove/replace、owner migration、old-RX promotion，以及 migration+promotion 的
同步复合动作。跨目标组合不再只按 global top-k，而按独立能力上界归一化的
worst/bottom-3/mean、L1 dual value 与 participant-signature diversity 组成 oracle-free beam。
定位器 API 仍无 witness/oracle/MILP/feasibility-query 输入。

在已暴露的 32 个 genuine-L2 帧上，exact-digest development recall 为 Pool A `28/32`；候选数
`1582--3433/frame`、中位 `2396.5`，总 CPU 时间约 `195.5 s`。相较旧 Pool A `0/32`，这说明
跨目标角色复用确实是主要语法缺口；但本轮使用了既有 exact digest 作回放标签，artifact 强制标为
`DEVELOPMENT_ONLY_NOT_BLIND`，不得改写旧 D1 的 temporal-separation 负结论，也不得进入正式
factor/dense bit-rate 声明。下一次正式重开必须换独立 development episodes 后先冻结再裁判。

随后对剩余 4 帧的共同 participant 结构作一次底层修正：将 dependency closure 的集合并代价
`|union_q V(Delta E_q)|` 写入 endpoint-conditioned dynamic beam。每个共享 UAV 只扫描一次全部
目标，并以 changed-target radius 为状态生成 radius 2--4，替代重复 subset beam。开发回放达到
Pool A exact-digest `32/32`；候选 min/median/max 为 `2089/3054/4330`，总耗时 `297.08 s`。
语法在已暴露开发集上闭合，但计算/候选成本尚不满足在线部署，证据等级继续保持
`DEVELOPMENT_ONLY_NOT_BLIND`。

## C-FBL：有限码长 U2U 解析闭环（2026-08-22，默认关闭）

新增复 AWGN 二阶正常近似
`k=nC-sqrt(nV)Q^-1(epsilon)`，支持 BLER、最小 blocklength 和所需 SNR 的双向反演。
learned message 与 detection evidence packet 复用同一 BLER/擦除门禁；失败包计费但不进入 actor
或 owner fusion。历史 Shannon 路径默认不变。

旧 optimal-bandwidth L0 的 KKT 基于 Shannon 特定凸函数，不能直接外推到包含 dispersion 的
FBL 目标；同时启用时配置 fail closed。FBL analytical L0 当前只允许 equal-bandwidth 反演。
这构成解析可靠性闭环，不构成具体编码、MAC、同步或硬件验证。
