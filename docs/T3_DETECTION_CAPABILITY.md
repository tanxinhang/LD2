# T3 Detection-Capability-Constrained — 对抗检测约束下的统一母问题（advice 012）

> 文档日期：2026-08-16。依据 [`advice/012.md`](../advice/012.md) 的
> Detection-Capability-Constrained 模型升级 T2。
> 工具：`tools/audit_horizon_joint_oracle.py`（新增 `--inner intercept`、
> `--intercept-capability {weak,medium,strong}`、`--intercept-eps`、
> `--intercept-pfa`）。结果 JSON：`results/_d012_{pw,ex,dc}_{weak,medium,strong}.json`。

## 1. 从“低电磁暴露”升级为“对方探测能力”约束

advice 011 的暴露约束 `E_w ≤ Γ_w` 本质是**物理代理量**：同样的接收功率，对方如果
拥有更大的阵列、更长的积累时间、更低的噪声、更准确的波形先验，其探测能力完全不同。
advice 012 建议把隐蔽性定义成**对抗检测问题**：对方（目标自身或外部监听节点 w）面对

```text
H0: UAV 未执行可识别的主动感知/传输
H1: UAV 正在执行主动 ISAC 感知/通信
```

并运行自己的检测器。我们关心的不再是 `E_w ≤ Γ_w`，而是**对方发现我方活动的概率**
`P_{D,w}^I ≤ ε_w`。

### 1.1 镜像 Deflection 模型

沿用己方感知的 Gaussian-shift/Deflection 抽象（建模假设，与当前
`P_D = Q(Q⁻¹(P_FA) − √D)` 完全同构）：

```text
D_w^I(X,Z,P) = Σ_i a_{iwq}^I(X,Z) · p_iq        （对方截获 Deflection）
P_{D,w}^I = Q( Q⁻¹(P_FA,w^I) − √(D_w^I) )        （对方检测概率）
```

隐蔽要求 `P_{D,w}^I ≤ ε_w` 等价于 Deflection 硬约束：

```text
D_w^I ≤ D̄_w^I := [ Q⁻¹(P_FA,w^I) − Q⁻¹(ε_w) ]²
```

### 1.2 对方能力参数 θ_w 与每瓦截获增益

```text
a_{iwq}^I = K_w · G_tx · G_w · (λ/(4π d_3d))² · T_int,w / (kT · B_w · NF_w)
```

| 符号 | 含义 |
|---|---|
| `G_tx` | UAV 发射天线增益（线性，16 dBi） |
| `G_w` | 对方接收阵列增益（线性） |
| `B_w` | 对方监听带宽（Hz） |
| `NF_w` | 对方噪声系数（dB） |
| `T_int,w` | 对方积累时间（s） |
| `K_w` | 对方对我方波形/时隙/载频的先验增益（matched-filter 增益） |
| `d_3d` | UAV→对方 3D 距离（含 UAV 高度 20 m） |

`a^I` 与暴露系数 `c[i,q] = G_tx·(λ/(4π d_3d))²` 的关系：**a^I 是 c 乘上对方的
接收机处理增益** `K_w·G_w·T_int,w/(kT·B_w·NF_w)`，这正是暴露代理量缺失的部分——
同样的每瓦泄漏，对方积累 1 ms 还是 100 ms、用 1 kHz 还是 100 MHz 带宽去听，
探测能力差 5 个数量级以上。

### 1.3 三个对方能力等级（θ_w ∈ Θ_w）

审计修正（2026-08-16）：原 tier 参数把 weak/strong 的带宽设反了——在
`1/(kT·B_w·NF_w)` 分母形式下，**监听带宽越窄越敏感**（窄带 matched filter），
越宽越不敏感（宽带能量搜索）。按物理敏感性重新排序：

| tier | `G_w` | `B_w` (Hz) | `NF_w` (dB) | `T_int,w` (s) | `K_w` | 物理画像 |
|---|---|---:|---:|---:|---:|---|
| weak | 1 | 1e8 | 12 | 1e-5 | 1 | 宽带能量搜索（~100 MHz），不知波形，短驻留，高 NF，无阵列 |
| medium | 3 | 1e7 | 8 | 1e-4 | 2 | 知道频段（~10 MHz），中等驻留/阵列，部分波形结构 |
| strong | 100 | 1e3 | 3 | 1e-1 | 10 | 对精确 pilot/子载波做 matched filter，长相干积累，大阵列，完整波形先验 |

对应 `a^I` 每瓦量级（28 GHz，1 W 预算，100–600 m）：weak ~1e-6..1e-3，medium
~1e-1..1e1，strong ~1e6..1e9——跨度 15 个数量级，正好覆盖“完全不约束”到
“每瓦都致命”的整个区间。

## 2. 统一母问题 (DC-MM) 与 Inner (I') 的实现

advice 012 §10 的母问题：

```text
max_{X,Z,P,t} t
s.t.  D_q^L(X,Z,P) >= t,  ∀q                    （我方 max-min 感知）
      D_w^I(X,Z,P;θ_w) <= D̄_w^I,  ∀w           （对方检测 Deflection 上限）
      P_comm,i + Σ_q p_iq <= P_max              （1 W 联合 RF 预算）
      U2U communication feasible                （b_i = 1 − P_comm^min(X,Z)）
      Z ∈ Z_HD,  X ∈ X_safe                     （半双工 + d_sep/d_standoff）
      p_iq >= 0
```

**Inner (I')**（固定 X,Z 的 LP，复用 `_exposure_maxmin_lp`，把 `c`/`Γ` 换成
`a^I`/`D̄^I`）：

```text
max_{p,t} t
s.t.  Σ_i a_iq p_iq >= t,  ∀q
      Σ_q p_iq <= b_i,     ∀i
      Σ_i a^I[i,q] p_iq <= D̄^I_q,  ∀q   （对方检测约束，观察者 = 目标 q）
      p >= 0
```

对偶返回统一三价格：`λ`（感知瓶颈）、`β`（本地 RF）、**`μ`（对方检测能力代价）**。
局部净价值从 advice 011 的“能量成本”升级为**“对方检测性能成本”**：

```text
s_iq = λ_q a_iq − μ_q a^I[i,q]
```

即：每瓦功率的价值 = 我方感知增益 − 对方可探测性增益。这正是 advice 012 §11 的
攻防对偶解释。

**与 exposure 的唯一区别是系数矩阵**：exposure 用 `c[i,q]`（每瓦泄漏）与 `Γ`，
intercept 用 `a^I[i,q]`（对方每瓦 Deflection 增益）与 `D̄^I`。LP 结构、归一化
技巧、budget-fill 尊重约束余量的逻辑全部复用（见 T2 文档 §2 的两个数值修复）。

**实现要点**：`a^I` 在每次 `_eval` 用**当前 UAV 几何**重新计算（与 exposure `c`
一致的一阶约定），因此“拉开 standoff”通过约束自然耦合——距离越远 → a^I 越小 →
允许功率越高，无需人为设 `d_stealth`。

## 3. 三向对比实验设计（advice 012 §9）

同一个 20-seed 教师测试集（H=10，教师最终几何起点，`results/
architecture_v2_scale_k8q8_teacher_trace_d079_test20/`）：

| 列 | inner | 约束 | 应该发生 |
|---|---|---|---|
| Power-constrained | `maxmin` | 仅 Tx 功率上限 | QoS 最高，但对方 P_D^I ≈ 1（被完全发现） |
| Exposure-constrained | `exposure` (Γ=1e-9) | 限制对方接收功率 | QoS 略降，但对方 P_D^I 仍可能 ≈ 1 |
| Detection-constrained | `intercept` (ε=0.1) | 限制对方实际 P_D^I | P_D^I ≤ ε 恒成立，QoS 代价随 tier 增大 |

每一列都报告**最终功率分配下对方实际看到的 P_D^I**（`intercept_pd_max`、
`intercept_violated`），因此三向对比直接回答：为什么不能只看发射功率或电磁暴露，
而必须建模对方实际探测能力。

## 4. 结果（20 seed，H=10，教师最终几何起点）

`P_D^I` 为**最终功率分配下对方实际看到的检测概率**（max over targets）；`viol%`
为 `P_D^I_max > ε` 的 seed 比例；QoS 为 steady-window 三档（worst/weak3/steady）。

### 4.1 三向对比总表

| tier | inner | worst | weak3 | steady | P_D^I_max | P_D^I_mean | viol% | 总移动 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| weak | Power-only | 0.832 | 0.832 | 0.843 | 0.0015 | 0.0011 | 0 | 149 m |
| weak | Exposure (Γ=1e-9) | 0.772 | 0.772 | 0.775 | 0.0011 | 0.0011 | 0 | 115 m |
| weak | Detection (ε=0.1) | 0.832 | 0.832 | 0.832 | 0.0014 | 0.0011 | 0 | 149 m |
| medium | Power-only | 0.832 | 0.832 | 0.843 | **0.4949** | 0.1047 | **75** | 149 m |
| medium | Exposure (Γ=1e-9) | 0.772 | 0.772 | 0.775 | 0.0536 | 0.0210 | 0 | 115 m |
| medium | Detection (ε=0.1) | 0.808 | 0.808 | 0.808 | 0.0761 | 0.0247 | 0 | 123 m |
| strong | Power-only | 0.832 | 0.832 | 0.843 | **1.0000** | 1.0000 | **100** | 149 m |
| strong | Exposure (Γ=1e-9) | 0.772 | 0.772 | 0.775 | **1.0000** | 1.0000 | **100** | 115 m |
| strong | Detection (ε=0.1) | 0.001 | 0.001 | 0.001 | 0.1000 | 0.1000 | 0 | 2 m |

### 4.2 解读

**1. Power-only 在 medium/strong 下完全失效。** 不做任何隐蔽约束时，medium 对手在
75% 的 seed 下以 P_D^I ≥ 0.1 发现我方（最坏 0.49），strong 对手 100% 发现
（P_D^I = 1.0）。QoS 虽高（0.832），但系统形同裸奔。

**2. Exposure 约束在 strong 下同样失效——这正是 advice 012 的核心论点。**
Γ=1e-9 已经把对方接收功率压到 1e-9 W（比 baseline 更严），但 strong 对手的
处理增益 `K_w·G_w·T_int/(kT·B_w·NF_w) ~ 1.25e19` 把这点泄漏放大成
D^I ~ 1e10 ≫ D̄^I=3.27，P_D^I 仍然是 1.0。**限制接收功率 ≠ 限制对方探测能力**：
暴露代理量缺失了对方接收机（阵列/积累/噪声/波形先验）的全部信息。

**3. Detection-constrained 恒成立（viol=0），且优于 exposure。**
- weak：约束完全不紧（a^I ~ 1e-6..1e-3），Detection = Power-only（QoS 0.832），
  而 exposure 反而无谓牺牲了 0.06 的 QoS——对方本来就看不见我们。
- medium：Detection（0.808）在同样 0% 违反下**比 exposure（0.772）QoS 更高**。
  detection 只削减“会被对方看到的那些瓦”（按 a^I 加权），而 exposure 无差别地
  削减所有泄漏功率——这就是“按检测能力定价”优于“按能量定价”的量化证据。
- strong：只有 Detection 能控制 P_D^I ≤ 0.1，代价是 QoS 坍缩到 0.001（对方对
  精确 pilot 做 matched filter + 大阵列 + 100 ms 积累时，1 W/28 GHz 在 100–600 m
  内**每瓦都致命**，唯一可行的解是把功率压到 ~1e-9 W，我方 P_D 回到 P_FA 底）。
  移动也只有 2 m：几何无法救 strong 场景。

**4. 为什么 exposure 会“时灵时不灵”——暴露代理量隐含的 D^I 上限是对方依赖的。**
因为 `a^I[i,q] = c[i,q] × K_w·G_w·T_int/(kT·B_w·NF_w)`，exposure 约束
`Σ c·p ≤ Γ` 隐含 `D^I ≤ Γ × K_w·G_w·T_int/(kT·B_w·NF_w)`：
- medium：`Γ×2.38e9 = 2.38 ≤ D̄^I=3.27` → 碰巧控制住（0% 违反，但 QoS 多付 0.036）；
- strong：`Γ×1.25e19 = 1.25e10 ≫ 3.27` → 完全失控（100% 违反）。
同一个 Γ 对 medium 有效、对 strong 无效——**Γ 无法感知对方是谁**，而
detection-constrained 对任何 θ_w 都把 D^I 钉在 3.27 以下。这正是 advice 012 §5
“对方能力未知时应取 sup 约束”的动机。

**5. 每瓦净价值对偶。** detection 内层把功率从“离对方最近/对方最强”的目标上
撤走（μ_w a^I[i,q] 大 → s_iq 降），而不是平均降压——20 个 seed 的
`pd_max` 全部被压在 0.008–0.100 之间（中位 ~0.09），说明约束在**目标层面**逐点
生效。

### 4.3 结论（回答 advice 012 §9）

> **为什么不能只看发射功率或电磁暴露，而必须建模对方实际探测能力：**
> 在 weak 对手下三者 QoS 相同（对方根本看不见）；在 medium 下 power-only 被 75%
> 发现、exposure 虽控制住但多付 0.036 QoS、detection 用同样的安全性拿到更高 QoS；
> 在 strong 下 power-only 和 exposure **双双 100% 被裸发现**，只有 detection 恒成立
> （代价是必须保持近乎静默——这是 1 W/28 GHz 下对抗 matched-filter 拦截的物理真实）。

## 5. 结论与对当前系统的处理方式

- **T3 完成（本 Gate）**：把 T2 的 exposure 约束升级为 detection-capability 约束，
  三个对方能力等级 + 三向对比全部落地并验证。对偶价格从“能量成本”升级为
  “对方检测性能成本”，统一母问题 (DC-MM) 的 inner 仍是精确 LP，可复用。
- **exposure 降级为 baseline**（advice 012 §11 的正式修正）：不再作为核心创新，
  只保留为第 2 列的对照；文档与模型中以 `Exposure-constrained` 明确标注其为代理量。
- **detection-constrained 成为新的隐蔽性主线**：后续分布式协调用
  `s_iq = λ_q a_iq − μ_w a^I[i,q]` 做本地 bid（列生成），μ 就是对方探测能力价格。
- **诚实边界**：这是 Deflection 层面的 Detection-Capability 模型，不是完整
  covert/LPD 理论（无 waveform-level 模型，不声称 KL/DEP 约束）。strong 场景
  “必须静默”是当前参数（1 W / 28 GHz / 100–600 m）下的物理结论，不是算法缺陷；
  若需在 strong 对手下工作，须走**波形层设计**（扩频/低截获波形）或大幅降功率，
  属于后续方向。
- **保留**：max-min LP、真实 U2U 通信、1 W 硬预算、安全间距、原子提交、
  统计证书；capability gauge 继续作 feasibility guard。

## 6. 复现

```bash
python tools/audit_horizon_joint_oracle.py --seed-limit 20 --horizon 10 --rounds 3 \
  --inner maxmin --intercept-capability medium \
  --output results/_d012_pw_medium.json
python tools/audit_horizon_joint_oracle.py --seed-limit 20 --horizon 10 --rounds 3 \
  --inner exposure --exposure-gamma 1e-9 --intercept-capability medium \
  --output results/_d012_ex_medium.json
python tools/audit_horizon_joint_oracle.py --seed-limit 20 --horizon 10 --rounds 3 \
  --inner intercept --intercept-capability medium --intercept-eps 0.1 \
  --output results/_d012_dc_medium.json
```
