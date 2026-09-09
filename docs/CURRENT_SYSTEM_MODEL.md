# 当前系统模型（Architecture V2 + 认证化分布式控制）

> **2026-08-26 系统身份冻结（advice/001 C0，post-G2 基础契约）**：本文档描述的
> 物理/协议基础身份由 **[`config/system_manifest.yaml`](../config/system_manifest.yaml)**
> 声明；当前可执行正式 profile 是
> **[`config/exp_strict_distributed_k16q16.yaml`](../config/exp_strict_distributed_k16q16.yaml)**，
> 它传递继承该 manifest，并绑定 K16/Q16 的 reset-distribution/v2 盲测库。基础 manifest
> 单独加载仍继承 legacy K4/Q2 默认值和旧 K8 bank 指针，只用于历史兼容，严格身份门会
> 明确拒绝。冻结契约包括 U2U-only（`ground=false`）、strict no-ground-fusion、joint RF power=true、
> distributed local-belief=true、OTFS numerology（fc=28 GHz, B=1 MHz, Δf=15.625 kHz,
> M=64, N=16, T_sym=6.4e-5, n_cpi=1）、detector `real_gaussian_shift`（c_det=1）、
> DD 连续增益 `I_support·|A|²`、sensing clock `cpi_frame`（T_sense=N·T_sym≈1.024 ms）、
> fusion covariance = independent/whitened（additive D）、seed scheme = 100-seed
> 独立盲测 bank + paired CRN。**正式论文结果只能由传递继承该 manifest 且通过 strict
> bank fingerprint 校验的 runtime profile 启动**；下文 §5.6.3/§6.8
> 中的 V3 小节是已删除机制的**历史审计记录**（默认 OFF、已从代码移除），不再是当前
> 系统语义。一致性由 `tools/check_system_identity.py` 强制校验。

> 文档日期：2026-08-20（2026-08-16 更新：§10 补 6/6 干净种子重跑结果并关闭污染
> 待办、§12 补回填/清理状态与最新测试基线；2026-08-17 刷新：§10 盲测三行与 LCB
> 口径（z=1.96）、§11 D1.7–D1.10 状态；2026-08-17 二次刷新：§5.6.2 并入 D1.8/D1.9
> 部署栈、§10 补 D1.10 独立采样认证与正式 Gate 断言、§11/§12 状态与测试基线；
> **2026-08-17 三次刷新（深度审计后）**：§6 补理论子节（字典序 L1 完整对偶、
> active-set 定理、精确结构上界、λ-μ 分布式列生成）、§8 补 smooth-disk 双射、
> §10 补 P1-4 结构上界与 MKL 复现性数据、§12 测试基线 927 passed、**新增 §13
> 当前问题与不足（分层清单）**；**2026-08-17 四次刷新（V3-T4 + 通信/感知审计）**：
> §5.6.3 新增 V3 dual-priced L2（默认 OFF）、§6.8 新增 V3 数学（scarcity 价格、
> owner 值、单调门、双尺度量化、DD 门前瞻）、§10 补 V3-T4 9-seed A/B 混合结果
> 与通信/感知审计修复数据、§12 测试基线 968 passed、§13 更新 V3-T4 live 状态
> （A/B 无效判定、审计修复闭环、剩余理论边界）；**2026-08-17 五次刷新
> （advice/016 task-regret 主线前两步）**：§6.9 新增 task-regret 证书与
> decision-preserving Token（Lipschitz 界、QoS 保持条件、bit 下界、事件触发、
> dual 权重）、§13 补真实数据发现（6/6 失败帧 teacher 结构本身不过地板——
> Student 校准必要但不充分）；**2026-08-17 六次刷新（adaptive owner bits
> 整合）**：§6.9 补 `adaptive_owner_bits`（真实 6/6 帧固定 6-bit 28.5% owner
> 决策实际翻转 vs adaptive 0 翻转）、§12 测试基线 980 passed；**2026-08-18
> 七次刷新（advice/016 Gate 0）**：§10 补 V3-C0 重认证两行（8/8 QoS 0.910 /
> LCB 0.838 过门为 V3-C0 新 baseline；6/6 QoS 0.590 / LCB 0.492 未过，Student
> 仍为绑定瓶颈）、§13 更新 6/6 状态与下一步（Gate 1 multi-scale CE Student）；
> **2026-08-18 八次刷新（advice/016 Gate 1 falsification）**：§10 补
> multi-scale CE Student 认证行（6/6 QoS 0.780 点估计过门 / LCB 0.689 差
> 0.011——dataset shift 为主因）、§13 更新 6/6 下一步（Gate 2 task-regret
> 补尾巴）；已对照 `uav_isac/`、`config/`、`tools/` 逐项校验）。
> 编号说明：历史 Gate 段落沿用原编号（D0.11 等）；D0.87–D0.95 ≡ D0.10–D0.18
> 双套编号映射见总纲 §1.4，后续新 Gate 一律用 D0.9x。
> 本文是**当前部署架构**的数学模型与代码映射总纲，覆盖 Architecture V2 与
> D0.x 认证化控制主线的完整系统模型，也是基础环境、物理链路、训练目标和执行
> 顺序的唯一活动数学规范。正式结果与版本判定以
> [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) 为准；算法演进、失败机制与工程审计见
> [`ALGORITHM_EVOLUTION.md`](ALGORITHM_EVOLUTION.md)；最新性能缺口与路线见
> [`README.md`](README.md)。若历史章节与本文
> 冲突，以本文为准。
>
> **2026-08-20 G2-0–G2-0.7 修订**：1 W 被严格定义为物理上限而非必须用满的等式；raw
> deflection 改为无量纲能量比；固定结构 LP 增加“条件可分离证据”适用前提；学习标签
> 使用最小 L2 范数规范对偶；决策保持通信同时计入陈旧误差与物理误差；检测器固定为
> 实高斯 shift-in-mean 充分统计量并由 Monte Carlo ROC 验证。所有既有
> 100-seed 数据均为 pre-G2 历史数据，等待同协议重认证。

---

## 0.0 系统身份（System Identity，post-G2 基础契约 + K16 可执行 profile）

> **2026-08-26 C0 收敛（advice/001 审计，2026-09-01 门禁收口）**：当前系统的冻结
> 基础身份由 **`config/system_manifest.yaml`** 声明；正式执行入口为传递继承它的
> **`config/exp_strict_distributed_k16q16.yaml`**。运行配置还必须通过 seed-bank schema、
> fingerprint、K/Q/region/dynamics 双向一致性检查。与该基础契约不一致的历史章节
> （如 V3 dual-priced L2 等已被删除机制）
> 一律视为历史记录，不再是当前系统语义。一致性由
> `tools/check_system_identity.py` 强制校验（并纳入测试套件）。

冻结基础身份要点（规模与 bank 由可执行 profile 绑定）：

| 身份项 | 冻结值 | 意义 |
|---|---|---|
| 系统边界 | **U2U-only，`ground=false`** | 没有 UAV→地面融合链路 |
| 融合边界 | `detection_fusion_mode ∈ {local_only, u2u_distributed}` | strict no-ground fusion |
| RF 功率 | `joint_isac_power_enabled=true`，`P_isac_total=1 W` | comm+sensing 共享 1 W 上限 |
| 分布式信息 | `distributed_coordination_use_local_belief_targets=true` | 协调只用局部 belief，无 simulator truth |
| OTFS | `fc=28 GHz, B=1 MHz, delta_f=15.625 kHz, M=64, N=16, T_sym=6.4e-5, n_cpi=1` | 论文 numerology |
| 感知时钟 | `sensing_energy_mode=cpi_frame`，`T_sense = n_cpi·N·T_sym ≈ 1.024 ms` | battery 与 deflection 同钟（不再 `P_sense·dt` 100 ms） |
| 检测器 | `c_det=1`（real Gaussian shift 约定），`P_FA=0.001`，`g_min=0.5` | detector convention 冻结 |
| DD 模型 | `dd_gain_mode=continuous`：`d_eff = χ_rep·d_raw·I_support·|A(τ,ν)|²` | 不再二值 `g_dd≥g_min` 门；无歧义 support + 连续 ambiguity² |
| 融合协方差 | 独立接收机证据（whitened）→ additive `D_q = Σ_edges a_ijq·p_ijq` | fusion covariance assumption |
| 种子方案 | 100-seed blind bank，eval/final 用 test split | 无训练/确认泄漏 |

**物理契约（C1 封口，2026-08-26）**：

- Doppler 符号：`ν = (f_c/c)·[v_txᵀ·u_iq − v_rxᵀ·u_qj + v_qᵀ·(u_qj − u_iq)]`；“Tx/目标静止、
  Rx 朝目标飞行 ⇒ positive Doppler”。`geometry.compute_doppler` 与
  `owner_local_physics` 向量拷贝已同步修正并有契约测试。
- OTFS support：`I_support = 1[0 ≤ τ < 1/Δf ∧ |ν| ≤ 1/(2T_sym)]`，走出无歧义区即使
  alias 后落到整数 bin 也判定为 0（修复“fractional-only 可能给出 g_dd≈1”）。
- 时间-能量：一个控制步执行 `n_cpi` 个 OTFS frame，`E_sense = P_sense·T_sense`，
  与 deflection 中的观测能量同钟。

---

## 0. 系统边界（做什么 / 不做什么）

当前研究对象是**只有 UAV 间（U2U）通信、没有地面链路的分布式多 UAV 协同 ISAC**。
每架 UAV 在 `1 W` 通信与感知联合功率**上限**下，自主决定运动、Token 通信与逐目标
感知资源；最终检测由环境级集中式证据融合计算。

真实边界应表述为：

```text
分布式局部观测 + 共享参数策略（运动/通信/感知意图）
  + 物理 U2U Token 传递（比特/时延/功率/丢包）
  + 分布式结构 Student 近似角色/配对/owner 决策
  + 环境级集中式证据融合与检测评价
```

**系统负责：** UAV 轨迹决策；Tx/Rx 角色分配；双基地感知配对与接收机归属（owner）；
逐目标检测概率；通信/感知联合功率分配；bit/时延/能量通信核算；event 触发的结构
/功率/几何安全控制；证书化 no-harm 保证。

**当前不负责（刻意抽象掉）：** 原始 IQ 波形与回波处理；完整 OTFS 调制解调；真实
通信协议栈；多目标数据关联（按目标索引 `q` 直接关联，无关联歧义）；硬件飞控接口；
波形层联合设计（发射协方差、子载波、波束、模糊函数**保持固定**）。28 GHz 等射频
参数进入路径损耗、多普勒、波长、天线增益、OTFS DD 网格的解析计算，但不生成时域
波形。

---

## 1. 场景与状态空间

### 1.1 场景

- `K` 架 UAV 与 `Q` 个目标，二维平面区域（默认 4/4 为 `800×800 m`，6/6 为
  `980×980 m`，8/8 为 `1130×1130 m`，边长为 `800·sqrt(K/4)` 以保持密度）。
- UAV 固定飞行高度 `h=20 m`，只有 `(x,y)` 受位移 `Δp` 改变。
- 目标在跨尺度审计中为**静止已知目标**（`tracking_enabled=false`），位置为任务
  已知信息；历史 4/4 亦支持 CV/CA 运动目标 + Kalman belief 追踪（见 SYSTEM_MODEL）。
- 帧时长 `dt=0.1 s`，每 episode `T=150` 帧，UAV 最大速度 `v_max=25 m/s`，单步最大
  位移 `dp_max = v_max·dt = 2.5 m`。

### 1.2 UAV 状态

```text
s^UAV_k = [ p_k, v_k, E_k, r_k ]
```

| 符号 | 内容 | 单位 | 代码 |
|---|---|---|---|
| `p_k` | 位置 (x,y,z)，z 固定 20 m | m | `uav.pos` / `UAVState.pos` |
| `v_k` | 速度，由 `Δp/dt` 估计 | m/s | `uav.vel` |
| `E_k` | 剩余电量 | J | `uav.battery` |
| `r_k` | 角色 ∈ {0=TX, 1=RX, 2=Idle} | — | `uav.role` / `Action.role` |

### 1.3 目标状态

```text
仿真真值:  position=[x,y,0], velocity=[vx,vy,0]      (3D, z=0)
belief  :  x_q = [x,y,vx,vy]^T                        (4D 平面 CV 状态)
```

跨尺度审计 `tracking_enabled=false` 时目标静止、位置已知，不经过 Kalman 循环；
运动目标场景仍支持 CV/CA/CT 模型 + 过程噪声 `sigma_a`（见 SYSTEM_MODEL §2）。

---

## 2. 物理感知层（双基地 → 检测）

### 2.1 双基地链路

三元组 `(i, j, q)` = (发射 UAV `i`，接收 UAV `j`，目标 `q`)，要求 `i ≠ j`。
对每条链路计算（`geometry.py`, `deflection.py`, `channel.py`）：

- 双基地距离 `R = ‖p_i − x_q‖ + ‖x_q − p_j‖`；
- 时延 `τ` 与多普勒 `ν`（tx/目标/rx 三段贡献，`× fc/c`）；
- 路径增益 `α`（Friis，`α² ∝ λ²·σ_rcs/(R_tx²·R_rx²)`；收发阵列增益 `G_tx·G_rx`
  在 §2.2 的 `d_raw` 中单独计入，不并入 `α` 本身）；
- 原始 deflection `d_raw`；
- DD 物理增益 `I_support·|A(τ,ν)|²`（走出无歧义支撑即为 0，支撑内按
  连续 ambiguity 能量损失缩放）；`g_dd=|A|` 与 `g_min` 只保留为 legacy binary
  结果的兼容诊断量；
- 上报链路可靠性 `χ_rep`（Rician / Al-Hourani LoS-NLoS，`channel.py`）。

### 2.2 原始 Deflection

```text
d_raw(i,j,q) = E_signal / N0
             = c_det·[P_sense(i,q) · |α_ijq|² · G_tx · G_rx · N · T_sym · L_eff]
               / [P_noise/B]
             = c_det·(P_r/P_noise)·M·N·L_eff,   B=M/T_sym.
```

其中 `P_noise=kT·B·NF` 是带内噪声功率（W），`N0=P_noise/B` 是有效噪声功率谱密度
（W/Hz=J），而 `E_signal=P_r·N·T_sym·L_eff` 也是 J，故 `[d_raw]=1`。
`B·N·T_sym=M·N` 是单 OTFS 帧的时宽积。这里显式采用 matched-filter 能量归一化，避免
旧实现将 J 除以 W 所留下的秒量纲。代码：`compute_raw_deflection`；量纲回归：
`tests/test_deflection.py::test_energy_normalization_is_dimensionless`。

### 2.2.1 检测统计量约定（G2-0.5）

当前不再把 `D→P_D` 当作 convention-free 公式，而是明确声明标准化充分统计量

```text
H0: Z ~ N(0,1),        H1: Z ~ N(sqrt(D),1).
```

Neyman–Pearson 阈值为 `η=Q^{-1}(P_FA)`，因此

```text
D = (μ1-μ0)^2/σ0^2 = c_det·E_signal/E_noise,
P_D = Pr[Z>η|H1] = Q(Q^{-1}(P_FA)-sqrt(D)).
```

本文选择**实等效 matched-filter 噪声方差**约定，故 `c_det=1`。若改用
`E|n_complex|²` 定义复噪声能量，可能得到常数 2；两种 convention 不得混用。
`c_det` 已成为显式配置和每瓦系数的一部分。`D={5,10,15,20}` 的 50 万样本 Monte
Carlo 得到经验 `P_FA=0.001010`，经验 `P_D={0.197046,0.528500,0.783206,0.916508}`，
与解析值最大绝对差 `<5.4e-4`；经验 Deflection 与输入最大差 `<0.082`。

**Link-budget falsification（理想单链路，双基地两腿等长，`P_s=0.0251 W`）**：

| 每腿距离 | `P_r/P_n` | `×MN` | `P_D(MN)` | `×n_CPI` 后 D | 最终 `P_D` |
|---:|---:|---:|---:|---:|---:|
| 100 m | 2.2904 | 2345.37 | 1.000000 | 300207 | 1.000000000 |
| 300 m | 0.02828 | 28.955 | 0.989011 | 3706.26 | 1.000000000 |
| 500 m | 0.003665 | 3.7526 | 0.124440 | 480.33 | 1.000000000 |
| 800 m | 0.000559 | 0.5726 | 0.009810 | 73.29 | 0.999999978 |

该表不是性能结果，而是**阻断性反例**：`MN` 后仍有合理距离动态范围，但再乘默认
`n_CPI=128` 后 100–800 m 全部饱和；若把每个 CPI look 解释为一个 `N·T_sym`
OTFS frame，则总积分时间 `0.131072 s` 还超过控制帧 `0.1 s`。因此 `c_det=1` 和
Gaussian ROC 已闭合，但 `n_CPI` 是否代表独立/相干且帧内可实现的处理增益尚未闭合。
在该问题解决前，禁止启动 G2-1A/1B 性能认证，也不能把饱和解释为算法增强。证据：
`results/_g2_0_5_detector_normalization/audit.json`。
该结论已由 `tools/audit_detector_normalization.py --assert-ready` 固化成机器可执行门禁；
G2-0.5 当时的状态为 `ready_for_g2_1=false`；随后由 G2-0.6 闭合，而不是改写该
反例的历史记录。

### 2.2.2 G2-0.6：可执行 CPI 与有效 look 数

一个 OTFS 帧的持续时间为

```text
T_F = N·T_sym = 16·64 μs = 1.024 ms.
```

控制帧给出的纯时间上界是 `L_time=floor(dt/T_F)=97`，但当前执行器每个动作只生成一份
OTFS 观测，故 `L_exec=1`。未被调度的时间不能自动成为独立样本或相干能量，当前采用

```text
L_eff = min(L_config, L_exec, L_time) = 1.
```

这不是经验调参，而是“证据只能来自已执行观测”的守恒条件。若未来使用独立多 look，
必须从逐 look LLR 得到 `D_total=Σ_l D_l`；若使用相干积累，还必须显式加入残余多普勒/
相位误差导致的 Dirichlet-kernel 损失，不能仅乘帧数。当前 100/300/500/800 m 的
`P_D` 为 `1.000/0.989/0.124/0.00981`，CPI 时长 `1.024 ms`，门禁状态已更新为
`ready_for_g2_1=true`。执行检查由 `validate_cpi_schedule` 和
`tools/audit_detector_normalization.py --assert-ready` 完成。

### 2.3 有效 Deflection 与每瓦增益

```text
d_eff(i,j,q) = I_support(τ_ijq,ν_ijq) · |A(τ_ijq,ν_ijq)|²
               · χ_rep,ijq · d_raw(i,j,q)
```

因此**每瓦有效增益**（与功率无关，D0.12 修正的可辨识形式）为

```text
a_ijq = I_support(τ_ijq,ν_ijq) · |A(τ_ijq,ν_ijq)|² · χ_rep,ijq · α_ijq²
        · c_det · M · N · G_tx · G_rx · L_eff / P_noise
```

代码：`per_watt_deflection_tensor_from_observables`（`physical_oracle_audit.py`）。
Swerling 开启时该重建 fail-closed（单次 RCS 实现不可由
`α/I_support/|A|/χ_rep` 识别）。旧公式
`1[g_dd≥g_min]·χ_rep·d_raw` 仅属于 `dd_gain_mode=binary` 的 pre-G2 重放口径，
不得用于当前 manifest 或新论文结果。

### 2.4 检测概率与融合边界

接收机局部证据 `receiver_d[j,q] = Σ_{i:(i,j,q)∈selected} d_eff(i,j,q)`（不跨接收机
融合，`receiver_deflection_from_selected`），随后按融合模式取逐目标 Deflection：

```text
local_only:      D_q = max_j receiver_d[j,q]         # 每目标单一 owner 接收机（部署）
central_oracle:  D_q = Σ_j receiver_d[j,q]           # 跨接收机集中融合（教师/上界）
legacy_global:   历史全局融合
u2u_distributed: 由实际送达证据包重建
```

检测概率为

```text
P_D^q = Q( Q^{-1}(P_FA) − sqrt(D_q) ),   P_FA = 1e-3
```

代码：`compute_PD`（`math_utils.py`）、`select_detection_deflection`（`evidence.py`）。
检测用概率值 `P_D`，不做随机采样。

> **注意**：`U(P_D) = -log(1-P_D)` 关于 `P_D` 严格凸、复合 `P_D(D)` 后非凹（见
> B8）；因此历史 P0 贪心是启发式，无次模/`(1-1/e)` 保证。当前部署主线已把优化
> 目标转移到 **Deflection 空间的 max-min 线性规划**（§5），其目标关于 `D` 是凹
> （最小化线性函数族的逐点 min），恢复了凹性/次模结构。

### 2.5 QoS 指标

每 episode 取末 20 帧为 steady 窗口，`P̄_D^q` 为窗口内目标 `q` 均值：

```text
steady = (1/Q) Σ_q P̄_D^q
weak3  = 均值 of bottom-3 P̄_D^q
worst  = min_q P̄_D^q
QoS-feasible  ⇔  steady>=0.80 且 weak3>=0.70 且 worst>=0.60
```

---

## 3. 通信层（U2U Token）

- 每架 UAV 广播稀疏目标 Token：聚合模式为单个 16 维向量
  （`comm_payload_mode='aggregate'`），`target_tokens` 模式为 `Q` 个各
  `comm_target_token_dim=16` 维的目标向量；稀疏 top-k 掩码只激活部分目标槽位并
  决定实际发送维度数 `D_active`，`b` bit/维（Architecture V2 部署取
  `comm_rate_bits_per_dim=[0,4]`，历史 4/4 部署为自适应 4–8 bit）。接收方保留
  sender identity 与 target identity 作为包元数据，重建 identity-indexed 团队竞价图。
- 包大小 `L_k = H + D_active·b_k`（另加结构协议扩展维度与速率头字段），
  `H=64` bit 包头（`comm_header_bits`）；活跃发送者正交平分控制带宽。默认配置仍为
  100 kHz；严格 K16/Q16 配置显式使用 500 kHz，以容纳端点状态、可组合证书和稀疏责任
  后验，并仍受 5 ms 截止时间约束。该带宽差异必须随结果报告。
- 链路用自由空间增益 `(λ/(4πd))²`、Shannon 串行化速率与 0.2 ms 处理时延；包只在
  SNR ≥ 阈值且总时延 ≤ deadline（默认 5 ms）时送达；最近送达 Token 保留最多 5 帧，
  带显式 AoI。
- 通信功率与感知功率满足 §4 的硬预算；结构协议字段（header/epoch/digest/索引/
  价格/反馈）逐 bit 计费（`structure_sequence_transport.py` 等）。
- 严格配置中，目标的物理感知接收机可在下一控制载波广播其 4D 因果后验。均值量化误差
  和协方差非对角项被保守吸收到解码协方差，接收端以协方差交集融合相关未知估计；每条
  后验消息计入包长、AoI、丢包、截止时间和能耗。完整推导与消融见
  `docs/OWNER_POSTERIOR_TRACKING.md`。

代码：`environment/communication.py`（`InterUAVCommunicationModel`）、
`coordination/*_transport.py`。

---

## 4. 功率预算与资源分配

每架 UAV 的通信功率与逐目标感知功率满足 `1 W` 上限：

```text
P_comm,k + Σ_q P_sense,kq ≤ 1 W,
Σ_q P_sense,kq ≤ P_sense,max = 0.0251 W,
P_comm,k ∈ [0, ρ_max·1 W],  P_sense,kq ≥ 0.
```

投影在环境内执行。分别报告联合预算 violation、感知波形 cap violation 和未用功率；
`ρ_max = comm_power_fraction_max`（Architecture V2 部署取 `0.5`，即通信功率至多
0.5 W）。post-G2 感知上限统一记为
`b_k = min(max(1 W − P_comm,k,0), P_sense,max)`。无额外暴露上界/功率代价的普通单调
max-min 问题存在一个用满 `b_k` 的最优解，代码可把它作为确定性 tie-break；这不是
物理等式。隐蔽/暴露上界绑定时允许少发或静默，`unused_power_w` 与真实超限
`budget_violation_w` 分开记录，能耗按最终执行功率结算。**通信代价是内生的**：增加 Token bit 或发送
功率既提高邻居可见性，又减少本 UAV 当帧可用感知功率。

---

## 5. 决策与控制架构

### 5.1 三层职责划分

```text
快层：固定 role/owner/edge 的 max-min 感知功率修复（LP / Dantzig-Wolfe）
中层：仅当固定结构上界低于 QoS 时，搜索 joint structure + power（原子 N5/N6）
慢层：仅当宽松同几何上界仍低于 QoS 时，触发航迹/几何修复
```

这是 D0.11 依据物理分解确立的架构：对 6/6 最终重解状态，`原部署/只换配对/固定
结构功率/联合结构功率` 的 mean-worst 分别为 `0.678/0.678/0.918/0.932`——**只换配对
增益严格为 0，联合层超过固定功率层仅 `+0.014`**，因此主要 headroom 在功率层与
几何层，而非继续扩大神经网络。

### 5.2 分布式 Actor

1. 共享 per-target scorer 对每个目标独立估计局部感知边际价值（参数共享，目标置换
   等变，非共享决策）；
2. 集合/注意力编码处理变长 UAV/目标集合，不依赖固定身份 one-hot；
3. 运动头由目标承诺与 Token 信息驱动（径向受运动学约束，切向形成双基地基线）；
4. 通信速率与总通信资源由集合池化头输出，不依赖固定 `Q`；
5. 感知/通信功率统一投影到 §4 的 capped simplex / power-budget polytope
   `{p≥0: 1ᵀp≤min(1 W-P_comm,P_sense,max)}`。

代码：`agents/networks.py`（`StructuredActorNetwork`）、`agents/mappo_agent.py`。

### 5.3 结构 Student 与集中式控制边界

- 集中式冻结控制器（receiver-owner full-graph hold-5 P0）提供结构目标；分布式
  Student 利用本地状态与收到 Token 近似其端点选择与结构解码。
- **部署版不是"完全去中心化检测器"**：决策路径分布式，最终 `P_D` 由环境级证据
  融合统一计算。集中式结构控制器只作为教师/参考上界，不属于部署执行路径。
- CTDE：Critic 只在训练时看全局状态，执行时每 UAV 只用本地观测、局部历史与已
  送达 Token。

### 5.4 锚点保持的基数门控残差（跨尺度）

```text
E(K,Q) = E0 + g(K,Q)·ΔE,   D(K,Q) = D0 + g(K,Q)·ΔD,   g(4,4)=0, g(6,6)=1
```

`E0/D0` 为冻结 4/4 Student，`ΔE/ΔD` 只在非锚点数据上训练；在 4/4 上残差严格关闭，
模型输出与原 Student 逐回合一致。代码：`agents/frozen_structure_student.py`。

### 5.5 双集合等变规划头（跨基数运动，D0.82）

对每架焦点 UAV `i`，分别构造目标集合与 peer UAV 集合：

```text
t_iq = [(y_q-x_i)/L, ‖y_q-x_i‖/L, dir(y_q-x_i), s_iq, c_i]
u_ij = [(x_j-x_i)/L, ‖x_j-x_i‖/L, dir(x_j-x_i), a_j/d_max, c_j, Σ_q s_jq]
Delta a_i,1:H = ρ·d_max·tanh f(self_i, SymmetricPool_q φ_t(t_iq),
                                       SymmetricPool_{j≠i} φ_u(u_ij))
```

精确不变量：目标置换**不变**、UAV 置换**等变**、均匀区域尺度**协变**、速度圆盘
投影保证 `‖a_i,h‖ ≤ v_max·dt`。代码：`agents/equivariant_movement_plan.py` 等。

### 5.6 解析执行栈（L0→L3）

除上述学习 Actor 外，系统有一条**解析控制执行路径**，用 `analytical_*_enabled`
标志逐层接管学到的通信/功率/结构/运动决策。**分两代**：

#### 5.6.1 Legacy deployed baseline：D0.95（gauge L1 + 单步 L3）

| 层 | 标志 | 作用 | 关键代码 |
|---|---|---|---|
| L0 通信余量 | `analytical_comm_power_enabled` | 解析最小通信功率，并与 sensing PA cap 取交集：`b_i=min(max(1 W-P_comm^min,0),P_sense,max)` | `env_core._compute_analytical_min_comm_power` |
| L1 功率 | `analytical_sensing_power_enabled` + `task_constrained_power_enabled` | 固定结构 max-min LP，或 capability gauge（三地板硬约束） | `maxmin_power.py`、`capability.py` |
| L2 结构 | `analytical_structure_ranking_enabled` | 功率无关排名（per-watt gain）+ 上一帧 max-min 对偶价格 λ* 的瓶颈优先级 | `env_core._per_watt_coefficient_from_entries`（P0 排名） |
| L3 几何 | `analytical_movement_enabled` | 价格驱动赤字→能力下降 + 达标悬停（单步梯度） | `env_core._analytical_movement_delta`（D0.95） |

D0.95 栈 20 seed 上 worst 0.662 / weak3 0.724 / steady 0.808（三地板全达标，§10）。

#### 5.6.2 Candidate deployment：D1.1 + D1.8/D1.9（lex L1 + 多候选/前瞻 L3）

在 D0.95 基础上，D1.1 系列升级为**当前部署候选**，D1.8/D1.9 补早期瞬态与视野
（证据见 [`ALGORITHM_EVOLUTION.md`](ALGORITHM_EVOLUTION.md)）：

| 层 | 升级 | 关键代码/标志 |
|---|---|---|
| L0 | 最优正交带宽分配（D1.1-C，KKT 凸解，no-waste） | `analytical_comm_optimal_bw` |
| L1 | **lexicographic QoS 约束 max-min**（D1.1-A，Stage-A 可行性 + Stage-B worst 最大化，已 live） | `task_constrained_mode: lexicographic` |
| L2 | 历史 pre-G2 headroom 曾量化 <1%；post-G2 不再据此排序，等待 capability attribution | P0 排名 + hold-5 |
| L3 | **多候选 trust-region**（D1.1-B，5–7 个整队候选 LP 打分）+ **对偶上界剪枝**（D1.1-B+，弱对偶精确剪枝）+ **前瞻评分**（D1.9，`analytical_movement_lookahead_frames=40`，receding-horizon：评分 H 帧持续趋近几何 / 执行 1 步）+ **帧-0 warm start**（D1.8，初始 capability gauge γ₀\*>1 即触发几何修复） | `analytical_movement_candidates_enabled`、`analytical_movement_dual_prune`、`analytical_movement_lookahead_frames`、`_initial_analytical_state` |
| 隐蔽性 | 反检测硬约束（D1.1-D/E，`P_D^I ≤ ε`，QoS×隐蔽性联合 LP） | `intercept_constrained_power_enabled`（默认关） |

**20-seed live 结果**（同种子同 warm-start，`_d095_lexcand20`）：worst 0.975 /
weak3 0.978 / steady 0.982 / QoS 1.0（LCB 0.839）。**盲测认证（100 全新 seed，
D1.5→D1.9→D1.10）**：QoS 0.730（LCB 0.636 未过）→ 前瞻 L3 后 **0.950 / LCB
0.888** → 独立采样协议 **0.940 / LCB 0.875**。这些是历史 pre-fix 数据；之后
V3-C0 `0.910/0.838` 才是 pre-G2 最新可比基线，G2-0 后须重认证（详见 §10）。
advice 013"盲测前不继续调参"已执行完毕：frozen 候选 + blind bank 认证，不因 blind
表现调参。`priced_structure.py` 的逐帧 owner+TX 重分配是离线结构修复 oracle，尚未
接入 live（L2 headroom <1%，已冻结，见 §6.2 注）。

#### 5.6.3 ~~V3：dual-priced 分布式结构协调（L2 重构）~~ —— 已删除机制（历史审计记录）

advice/015 主线：**冻结 L0/L1/L3，重构 L2 为对偶一致的分布式结构协调**——让 L2
使用与 L1 同一组任务级 shadow price（L1 产生资源稀缺价格 → 各 UAV 本地计算结构
reduced-cost → 分布式竞价/匹配 → L1 重新精确分配功率），使功率-结构-几何首次用
同一套"物理价值价格"。**默认 OFF**（`v3_dual_priced_structure_enabled: false`，
不引入未认证退化）；数学见 §6.8，A/B 数据见 §10，live 状态见 §13。

| 层 | V3 变更 | 关键代码/标志 |
|---|---|---|
| L0/L1/L3 | **冻结**（不调参） | 同 §5.6.2 |
| L2 | **dual-priced 结构协调**：每 `v3_structure_hold_frames=20` 帧从 P0 结构出发跑分布式价格迭代（L1 对偶 π/η → 量化 owner 值 → 单目标坐标更新 → 精确 max-min LP 字典序门），只提交**认证严格更优**的结构 | `dual_priced_auction.py::price_iteration`、`v3_dual_priced_structure_enabled`、`v3_structure_hold_frames`、`v3_price_bits=6` |
| 通信核算 | 量化价格 bit 诚实记账（双尺度 6-bit，`_v3_comm_bits_total`） | `dual_priced_auction.py::distributed_owner_values` |
| 感知门 | 三地板 lex 门（worst/weak3/steady 全验证）+ DD 门 Lipschitz 前瞻证书 | `v3_deficit_only`、`v3_min_improvement_frac=0.05`、`v3_lookahead_steps=1`、`v3_relax_route` |

**V3-T4 live A/B（9 seed：6 失败 479/866/725/237/410/274 + 3 PASS 14/752/705，
独立采样，crash-isolated MKL=1）**：五轮迭代（rev1 单帧 → rev2 持久化+deficit →
rev3 前瞻门 → rev3b 无持久化 → rev4 relax 路由+提高守卫）**全部混合结果**，
QoS 计数从未超过 OFF（4/9）——**如实判定无效，默认 OFF**。根因是**闭环策略
耦合**：提交改变执行结构→观测→冻结 Student 策略的 comm/预算决策分叉→轨迹
分叉；单调门只保证提交帧精确 LP 的 t* 非降。详见
[`ALGORITHM_EVOLUTION.md`](ALGORITHM_EVOLUTION.md) V3-T4 章节与 §13。

---

## 6. 优化层数学（三层核心）

### 6.1 快层：固定 owner 的 max-min 功率 LP

固定唯一 owner 后，`a_iq` 为 UAV `i` 对目标 `q` 的每瓦有效 Deflection，
`b_i = min(max(1 W − P_comm,i,0), P_sense,max)`：

> **Assumption A（条件可分离证据）**：在一次 L1 内，结构/owner、DD active set 与
> 当前 CSI 固定；不同 TX 的检测统计量经过可分离处理，且 `a_iq` 不随本轮优化变量
> `p_iq` 改变。只有在该前提下 `D_q=Σ_i a_iq p_iq` 才是线性的，以下强对偶和影子
> 价格才成立。若存在 sensing interference、功率相关 CSI/DD 支撑、非线性接收机或
> 波形协方差耦合，必须升级物理模型，不能继续把 L1 称为 LP。

```text
max_{p,t}  t
s.t.  Σ_i a_iq p_iq ≥ t,   ∀q           (目标下界)
      Σ_q p_iq ≤ b_i,      ∀i           (每 UAV 感知上限)
      p_iq ≥ 0
```

**对偶**（单纯形）：

```text
min_{λ ∈ Δ_Q}  Σ_i b_i · max_q λ_q a_iq
```

强对偶成立（LP 可行有界）。性质（D0.11/D055 已数值验证）：

- 最优对偶 `λ*` 由互补松弛支撑在**瓶颈目标**上（`λ*_q>0 ⟹ D*_q = min_q D*_q`）；
- `Σ_q λ*_q D*_q = min_q D*_q`（支撑线性化 / 次梯度）；
- `λ*_q` 是"目标 q 下界约束"的影子价格（边际最差 QoS 价值）。

对偶最优解可能不唯一。用于 Student 监督和敏感度标签时采用二级问题
`min ||λ||²₂ s.t. λ∈Λ*` 的唯一规范解（`canonical_maxmin_dual_prices`），避免 LP
求解器基选择和目标排列造成标签跳变；在线主控制仍保持原始 max-min 目标不被扰动。

**有限轮 Dantzig–Wolfe 列生成**：初始化 Q 个全目标覆盖列，每轮 owner 广播量化
价格，各 UAV 用本地 `a_iq` 生成最佳响应列，owner 端求 Q 维受限主问题；任何返回
功率都是完整可行列的凸组合，逐 UAV RF 上限恒满足；普通单调问题可选择用满预算的
最优列，约束型问题不强制等式。受限原始值单调不降。6-bit 价格
4 轮达精确解约 `98%`。代码：`coordination/maxmin_power.py`。

**松弛同几何上界**（慢层路由判据，不是可行动作）：

```text
D_q^relax = Σ_i b_i · max_j a_ijq
```

任何可行同几何方案逐目标不超过该值（放松共同 owner、角色、容量与跨目标功率耦合）。

**capability gauge（D0.93，task-constrained 功率，L1 的完整任务形式）**：把三地板当
**硬约束**而非标量目标，求最小预算缩放 `γ`：

```text
γ* = min_{p,γ} γ
s.t.  D_q ≥ d_min(0.60),   ∀q                       (worst)
      D_q ≥ d_weak3(0.70), ∀q 的 bottom-3 组合      (O(Q) order-statistic 线性化)
      D_q ≥ d_steady(0.80),∀q                       (steady)
      Σ_q p_iq ≤ γ b_i                              (预算按 γ 缩放)
```

`P_D` 在 `D ≥ c²/3` 区间上关于 D 凹，用两侧 PWL 证书（chord 下界 + tangent 上界）把
gauge 转成 LP（`γ_optimistic ≤ γ* ≤ γ_conservative`）。`γ* ≤ 1` 时返回的功率满足完整
任务，否则回退 reserve-first max-min（best effort）。代码：`coordination/capability.py`
（`capability_gauge_pwl_lp_full`）、`coordination/pwl_pd.py`。

### 6.2 中层：对偶引导的原子结构修复

- 候选：B=2 弱目标集合上的原子 N5（角色+owner 依赖闭包重建）与 N6（固定角色、
  只交换 1–2 目标 owner/support）；依赖闭包可影响多目标（D0.5：99.57% 候选实际
  改变 >2 目标）。
- 排序：候选只按**严格可行下界** `L` 与公开价格对偶上界 `U_λ` 的区间中点

  ```text
  L(S') = max(当前功率重放, 公共全目标列混合, 当前列−公开价格最佳响应列)
  U_λ(S') = Σ_i b_i max_q λ_q a'_iq
  score(S') = 0.5 L(S') + 0.5 U_λ(S')
  ```

  `L` 始终对应真实可行 RF 分配；`U_λ` 只用于排序，**绝不进入接受条件**。
- 接受：Top-1 候选须通过有限轮量化功率复算、严格单调改进、完整通信序列与
  episode-conformal 净收益门，否则 No-op；每事件至多一次原子重构（D0.15）。
- 代码：`coordination/dual_guided_structure_repair.py`。

**价格驱动结构重分配（D0.95，L2 的离线结构修复 oracle）**：与上述认证化原子搜索
互补，是 8/8 解析栈 L2×L3 交替下降里使用的**逐帧 owner+TX 重分配**结构修复器
（`tools/audit_l3_l2_alternating.py`；live 的逐帧 L2 是 §5.6 的功率无关 P0 排名）。
包络定理 `∂γ*/∂a_iq = −π_q p_iq` 给出同一价格 `π` 驱动结构与几何；owner 选择价格正交：

```text
j_q* = argmax_j Σ_i a_ijq b_i        (π_q > 0 提出公因子)
S_q* = top-K_q_max of {a_i,j_q*,q b_i}
```

半双工角色划分（TX/RX 互斥）用**功率共享感知的 max-min LP 打分**贪心求解（而非
2⁸ 枚举）。硬帧上结构单独可修复 44.2%（worst 地板），与几何交替联合达 67.5%。
代码：`coordination/priced_structure.py`。

### 6.3 慢层：认证化几何修复（原子提交）

每个接受动作是 prepare/vote/decision 三轮原子事务（D0.78）：

1. **Prepare**：收集 owner-local 状态/梯度，广播稀疏修正，收集 owner 验证因子，
   形成 digest 绑定证明；
2. **Commit**：全 UAV 投票，拒绝无副作用，接受则冻结结构与 RF 到承诺计划 `H` 步；
3. **Execute**：repair 与 baseline 分支用不同认证运动管、但相同结构/RF/语义送达
   状态与保守能量扣减；
4. **Synchronize**：终点位置/速度按构造相等，其余递归状态分支公共，下一决策从
   同状态开始。

代码：`coordination/certified_geometry_repair.py`、`persistent_geometry_execution.py`
、`causal_joint_plan.py`。

**解析几何下降（D0.94/D0.95，L3 的部署形式）**：与上述认证化原子提交互补，是
`analytical_movement_enabled` 的逐帧价格驱动运动。每帧对上一帧 per-watt 系数/owner/
预算求一步赤字→能力下降：

```text
若 ceiling_q < d_min（不可行）：  δ_q = [d_min − ceiling_q]_+
    d_k = −step · g_k/‖g_k‖,   g_k = Σ_q δ_q b_k ∇_{x_k} a_kq  (+ owner/Rx 项)
否则（可行但 worst < d_steady）：用 max-min 对偶 λ* 驱动
    d_k = −step · g_k/‖g_k‖,   g_k = Σ_q λ*_q p_kq ∇_{x_k} a_kq  (+ owner/Rx 项)
否则（worst ≥ d_steady）：悬停（零位移，不交回 actor）
```

**指标选择价格**：L3 的达标门是 `worst_deflection ≥ d_steady`（**最差目标**达到
0.80 地板），故几何层用 max-min 对偶 `λ*`（互补松弛集中于最差目标）驱动运动，而非
摊在三地板（worst+bottom-3+steady）的 capability 对偶 `π`——后者会稀释运动（§2.5
的 `steady` 是全部目标均值、`worst` 才是逐目标最小，L3 的 satisficing 门只盯最差
目标）。步长受 `‖Δp‖≤v_max·dt` 硬约束。代码：`env_core._analytical_movement_delta`。

> **L1 执行 vs L3 引导的目标"不一致"是有意分层（2026-08-16 审计澄清）**：多候选
> L3 的候选评分用**纯 max-min LP**（`solve_fixed_structure_maxmin_power_lp`），
> 即使执行内层是 lexicographic（Stage-B QoS 约束 max-min）。曾试图改成"评分与
> 执行一致"（用 Stage-B 评分候选），端到端 2-seed 对比严重退化（seed 503 final
> worst 0.996→0.662）。理论解释：**L3 的职责是引导几何，纯 max-min 评分在 `t*`
> 之上持续提供改进梯度（即使三地板已满足仍推动 UAV 提升均衡水平），而 Stage-B
> 在地板绑定处把 `t*` 钉在地板、评分对移动失去区分度 → 几何停滞**。这是 D0.95
> "指标选择价格"分离（L1 用 gauge/lex、L3 用 λ\*）在评分层的自然延伸，不是缺陷。
> 配置 `analytical_movement_lex_scoring`（默认关）保留为负结果记录。

### 6.4 字典序 L1 与完整对偶证书（D1.1-A，live）

固定 owner 结构下，L1 内层把"可行性"与"性能"字典序分离：

```text
Stage A: capability gauge  →  γ* ≤ 1 ?   （证明 worst/weak3/steady 三地板可行）
Stage B: QoS-constrained max-min
         max t  s.t. D_q ≥ t, D_q ≥ d_min, weak3 ≥ d_weak3, steady ≥ d_steady,
                    Σ_q p_iq ≤ b_i        （gauge 的分配是 Stage-B 可行点）
不可行帧: 回退 reserve-first max-min（best effort，fail-closed）
```

Stage-B 的 `t*` 恰是"满足门限后的剩余资源"的回收：gauge 在 γ*≤1 后把 worst 钉在
0.60 地板（satisficing），lex 在可行域内最大化 worst（20-seed 配对 +0.18，live 验证）。
**完整对偶证书**：单靠 max-min 对偶 `λ*`（worst 约束影子价）会系统性偏向最差目标
（D0.89-B：steady 塌到 0.64）；正确价格是完整对偶

```text
有效价格 π_q = λ*_q（max-min） + μ*_q（reserve/地板，反解自 P_floor） + 容量/角色对偶
```

其中 `μ*` 是 counterbalance `λ*` 的量：`λ*` 拉最差目标、`μ*` 保每个目标不低于地板。
代码：`capability.py::qos_constrained_maxmin_lp`（Stage-B）、`optimal_maxmin_dual_prices`。

### 6.5 局部支撑保持定理（canonical continuous DD；legacy binary 另记）

当前系数
`a_ijq=I_support(τ,ν)·|A(τ,ν)|²·χ_rep·α²·C` 只在无歧义支撑边界处
不连续；支撑内部的 ambiguity 能量增益连续变化。局部候选必须用 delay/Doppler
到支撑边界的余量，证明整个 trust region 内 `I_support` 不变；无法证明时即
fail-closed，并在移动后几何上精确重算系数。处理**不是**把支撑边界平滑化。

对 legacy `dd_gain_mode=binary` 重放，旧 active-set 证书仍可写为：设
`m_ijq=g_dd,ijq-g_min`，`L_g` 为 `g_dd` 关于位置位移（**速度保持**，即帧内
trust region）的 Lipschitz 常数：

```text
L_g = (4/π) · [ M·Δf·2/c + N·T_sym·(fc/c)·(2 v_max + 2 v_t)/R_min ]
```

（延迟项来自 τ=R/c、双基地路径变化 ≤2r；多普勒项来自单位向量旋转
`|Δν| ≤ (fc/c)·r·(|v_tx|+2|v_t|+|v_rx|)/R_min`；`|sinc'| ≤ 4/π`。）

若 `|m_ijq| > L_g·r`，则半径 `r` 的 trust region 内 legacy
`1[g_dd ≥ g_min]` 恒定、其 `a_ijq(x)` 局部光滑/Lipschitz；否则 certificate
fail-closed。该二值定理只用于历史复现，不能描述 canonical continuous-DD 系统。
跨帧速度模型（v=step/dt）
使多普勒对齐逐帧摆动 O(1)，**无跨帧证书**——逐帧精确重解即 fail-closed 语义。
实现：`coordination/power_staleness.py::dd_gate_position_lipschitz_constant` /
`dd_gate_active_set_certificate`；数值验证：解析 `L_g` 为真上界、已认证边在 trust
region 内永不翻转（`tests/test_dd_gate_active_set.py`，6 项）。

### 6.6 精确结构上界（阈值可行性 MILP + 二分，C2 / advice 014 P1-4）

single-duplex 交替启发式不是最优性证明；用小规模**精确 MILP** 量化真实结构上界：
阈值可行性 MILP（binary 角色/owner/边 + 连续功率，perspective 约束避免分数 owner
复制 RF 功率）+ 单调二分求 QoS 边界，给出 **proven 可行下界 / proven 不可行上界**。
20 个代表性 final-resolved 帧：联合结构+功率 worst P_D **0.9154/0.9197（宽度
0.0043，紧致）** vs 固定结构功率 0.7924（结构 headroom +0.123）；对偶引导稀疏
启发式达 0.8775（恢复 76.1% 结构增益、QoS 可行率分类与 exact 相同，仅用 23.9% 边）。
**论文措辞结论**：联合结构上界可由"交替启发式自述"升级为 **exact/proven**，并如实
报告稀疏启发式 gap（~0.038 P_D）。实现：`coordination/qos_threshold_feasibility.py`、
`tools/audit_qos_threshold_boundary.py`、`tools/audit_joint_structure_relaxation.py`。

### 6.7 攻防对偶分布式列生成（D1.7，理论闭合）

T3 隐蔽性约束的 DC-MM 内层 LP（感知行 `Σ_i a_iq p_iq ≥ t`、隐蔽行
`Σ_i a^I[i,q] p_iq ≤ d̄_q`、预算行可分离）用 Dantzig–Wolfe 精确分解：耦合行进
master，预算行进每 UAV 的 pricing 子问题。给定主问题对偶价格（λ 感知 / μ 对方探测），
本地 bid 恰为

```text
q_i* = argmax_q (λ_q a_iq − μ_q a^I[i,q]),   p_iq = b_i 若 q = q_i*
```

列生成（RMP + 本地 bid 加列）与中央 LP 收敛至机器精度（20 随机场景 gap ≤3.6e-15、
均值 5.8e-16）——T3 的 detection-capability 约束可**精确分布式化**（理论闭合）。
完整分布式协调器（每帧 RMP + 价格广播 + 本地 bid 执行）为下一步实现。实现：
`coordination/dw_column_generation.py`、`tests/test_dw_column_generation.py`。

### 6.8 ~~V3：dual-priced 对偶一致 L2（advice/015）~~ —— 已删除机制（历史审计记录）

L2 重构为与 L1 同一套对偶价格的分布式结构协调（实现：`coordination/
dual_priced_structure.py`、`dual_priced_auction.py`；配置 `v3_*`，默认 OFF）。

**对偶价格（固定结构 max-min LP 的对偶）**：

```text
max t  s.t. Σ_i a_iq p_iq ≥ t,  Σ_q p_iq ≤ b_i,  p ≥ 0
对偶:  min_λ Σ_i b_i max_q λ_q a_iq   (λ ∈ simplex(Q))
  π_q = λ*_q                    (目标 q 的稀缺价格)
  η_i = max_q π_q a_iq          (UAV i 的实际感知预算稀缺价 = 精确 Lagrange 乘子)
```

**强对偶**：`Σ_i b_i η_i == t*`（数值验证 ~5.8e-15，`scarcity_prices` 生产代码
含断言）。候选边 reduced-cost `s_ijq = π_q a_ijq − η_i`；固定角色下 owner 值

```text
V_jq = Σ_{i≠j} b_i [ π_q a_ijq − η_i ]_+    (i≠j 物理约束硬编码)
```

bipartite matching LP relaxation **全幺模（TU）→ 整数 owner**（无人工容量；
advice/015：物理模型没有 C_j 就绝不人为加）。

**分布式价格交换（T3）**：TX 广播量化价格（**双尺度 6-bit**：π 单纯形与 η 预算
对偶各用自身 max 归一，再精确重建公共尺度——审计 2026-08-17 修复了旧公共尺度
在 η 主导时压碎 π 的缺陷），目标本地累计 owner 值并竞价；每轮通信量诚实核算
`K·(K·Q·6+6) + K·Q·6 + 12` bit（@K=Q=6 为 1560 bit/轮，1 km 处串行化 ~9.3 ms
> 5 ms deadline——**价格交换仅短距/少轮可行**，通信-算法耦合显式化）。

**盈余提交门（T4，构造性零退化）**：每轮只提交 owner 值盈余最大的一个目标
（`q* = argmax V[desired]−V[current]`），候选用**精确 max-min LP + 字典序
QoS key** 判定，接受仅当严格更优：

```text
lex key = (feasible, −Σ_floor Σ_q max(d_floor − D_q, 0), min(D))
```

（审计 2026-08-17：从单地板升级为**三地板**——worst/weak3/steady 对应 d_floor
11.18/13.07/15.46，从 `task_constrained_qos_floors` 读取；消除"QoS 判定三地板、
救援只救最差"盲区）。近端守卫 `min_improvement = 0.05·d_req` 拒绝边际重分配；
hold 窗口（20 帧）保护 L3 几何循环；提交序列 t* 单调非降。

**DD 门感知前瞻门（rev3，审计 2026-08-17 补全）**：提交前在**外推几何**（当前
L3 移动 delta + 目标速度，外推 `v3_lookahead_steps` 帧，3D 位置）上重验候选
"lex 不劣于基线"；每外推步用 §6.5 的 `dd_gate_active_set_certificate` 判定门是否
可能在位移内翻转，**不确定边 fail-closed 拒绝**（外推只重缩放 α²、冻结 DD 支撑
在平滑模型下不可见门翻转，证书补全该缺口）。1 帧外推位移（v_max·dt）远小于门
翻转阈值，实测证书恒认证门恒定。

**V3-T4 live 判定**：A/B 混合结果（QoS 从未超过 OFF），**默认 OFF**；根因 =
闭环策略耦合（提交→观测→冻结 Student 策略轨迹分叉），结构侧单调门系列无法
根治——瓶颈在 Student 校准（§13 A）。

### 6.9 Task-regret 证书与 decision-preserving Token（advice/016，默认 OFF）

advice/016 新主线：Student 从"结构模仿器"改为"**任务 regret 代理**"，通信从
"semantic Token"改为"**保持 ISAC 结构/QoS 决策不变所需最小信息量**"。本步落地
两个理论可验证项（实现：`coordination/structure_regret.py`、
`tools/audit_structure_regret.py`）：

**① 结构误差 → max-min 能力损失 Lipschitz 界**：固定结构 max-min 对偶
`t*(A) = min_λ Σ_i b_i max_q λ_q a_iq`，由 `|min f − min g| ≤ sup|f−g|` 与
max 在单纯形权重上的 1-Lipschitz 性：

```text
|t*(A) − t*(Â)| ≤ Σ_i b_i max_q |a_iq − â_iq|      (加权 max-norm)
```

**QoS 保持条件**：`ε_struct = Σ_i b_i‖Δa_i‖∞ < m = t*(A) − d_req ⟹ floor
preserved`。真实 6/6 失败帧（300 帧 teacher trace）验证：强扰动下**界零违例**；
同时暴露**关键事实——这些帧的 teacher 结构本身 t\* 均值 −7.42、0% 过地板**，
即 6/6 失败是 Student 误差 × 弱几何 × 闭环敏感性的三重机制（advice/016 §1），
Student 校准必要但不充分。

**② Decision-preserving Token bit 下界 + 事件触发**：候选 score margin `Δ`、
量化动态范围 `R`，B-bit 均匀量化误差 `ε_B ≤ R/(2(2^B−1))`，排序保持条件
`Δ > 2ε_B` 给出严格整数条件

```text
B > log₂(1 + R/Δ_eff),
B_min = floor(log₂(1 + R/Δ_eff)) + 1,
Δ_eff = Δ − 2(E_stale(h)+E_phys),
且 Δ > 2(E_stale+E_phys)+2ε_B ⟹ 不发 Token
```

不能把第一行替换成普通 `ceil`：当对数恰为整数时，`ceil` 只达到等号，两个量化
score 仍可能打平。2026-08-29 的边界反例测试覆盖 B=1/2/3/4/8/16。

真实 6/6 数据：844 个 per-target owner 决策 mean 6.6 bit / min 2 / max 13、
1.2% 可抑制——**margin 大少发、不确定多发**（通信-算法耦合的可推导形式）。
**真实决策翻转证据**：固定 6-bit 量化下 **28.5% 的 owner 决策实际翻转**
（260 真实帧中 74 帧 argmax 与全精度不同）——量化误差是真实发生的决策错误，
非理论风险；`adaptive_owner_bits`（`dual_priced_auction.py`）按 margin 选每
目标 bit（协议级 per-value 宽度 = max_b，固定 6-bit 是 max_b=6 特例），保证
0 翻转。按 advice/016 §21，对偶价格进训练/离线监督、**不强行变在线协议**
（默认 OFF）。

**在线接线状态（2026-08-29 纠偏）**：上述标量证明只适用于被量化的 score 本身。
当前 `target_tokens` 是任意神经载荷；没有 decoder Lipschitz 常数及最终动作 margin，
“距离能力 score 的间隔”不能证明 token 静默或降比特后接收端动作不变，也不能证明
证书、后验等控制载荷可被省略。因此系统 manifest 将 C6 三个在线开关全部置 OFF，
`EnvironmentCore` 对 learned-token+C6 组合构造期拒绝。精确比特传输和标量数学原语
保留供未来显式 score-token 协议使用，但不再计入当前通信收益。

当 `Δ_eff≤0` 时不存在有限 bit 数可以给出排序证书，执行必须发送更丰富的信息、保持
上一安全决策或 fail-closed；不得把退化情形返回的“1 bit”误写成已认证。

**③ Dual-weighted 训练权重**（供未来 Student 校准）：包络定理
`∂t*/∂a_iq = π_q·p_iq` 给 edge 级权重 `w_iq`（`π` 使用最小 L2 规范对偶；真实数据
瓶颈目标 4.2× 集中）；
`task_regret_loss` 提供字典序（R_γ = feasibility-flip、R_t = 残余 max-min）。

### 6.10 CIS-ISAC 增量规模化边界（冻结路线，尚未 live）

CIS-ISAC 不替换 L0--L3，只计划在 L1 后、L2 前增加 Scaling Guard。它必须等
G2-1A/1B 建立 post-G2 baseline 后，按 G2-S0/S1 shadow 协议验证，当前执行路径不变。

固定当前 PWL branch、order-statistic active set 与 DD active set 时，将任务行写为

```text
g_r^T D >= h_r,       s_r = g_r^T D - h_r.
```

只有各目标存在已认证扰动界 `|δD_q|≤e_q` 且

```text
s_r > Σ_q |g_rq| e_q,   对所有当前任务行 r,
```

才允许 shadow 判定“当前结构无需扩大搜索”。若任何 `e_q` 不可识别，证书自动失败，
不能用经验 margin 代替。物理 ceiling 使用实际预算
`b_i=min(P_sense,max,1 W-P_comm,i)` 与乐观放松
`D_q^relax=Σ_i b_i max_{j≠i}a_ijq`；若 relaxed detection vector 仍不满足任务，直接路由
现有 L3，不浪费 L2 搜索。

若 slack 不足，候选集必须嵌套扩展 `A0⊂A1⊂...⊂E_full`，每一级均包含当前结构
`S_t` 和 No-op，候选只做 speculative evaluation，绝不先执行再回滚。无法认证时扩展到
full graph，因此是 **performance fail-closed / complexity fail-open**；典型复杂度可写为
`O(C_t d_R d_T)`，最坏情况仍为 `O(K²Q)`，不得宣称线性复杂度。

未来 anytime coordination 只有在 capability-gauge 列生成完成 pricing 并验证完整列集
dual feasibility 后，才能给出 `lower_gamma≤gamma*≤upper_gamma`。Restricted Master dual
本身不是完整问题 bound。Student 仅作为 proposer；所有 scale 消息仍须通过现有 L0 的
bit、带宽、功率、时延和丢包 admission。

---

## 7. 证书与安全层

系统不依赖单一 trigger，而是组合多层 fail-closed 证书（详见 D0.x 各 Gate）：

```text
owner-local 边际价值
  -> 依赖闭包 LNS（小提案触发，证书化排序）
  -> 物理计费且 epoch/digest 绑定的原子提交
  -> transition/transport 事件级联合最大分数（split-conformal）
  -> 延迟 owner 残差 + 冻结管线 + e-process 漂移锁定 No-op
```

关键构件：

- **事件级 split-conformal**：统计单位是独立 episode，事件内上千候选/帧/目标不
  当独立样本；联合分数取事件内最大标准化低估残差，5% 风险下有限样本覆盖下界
  `n/(n+1)`。
- **owner-local 双端物理证书**：候选下界与 No-op 上界分别满足逐目标 no-harm 与
  worst 净增益；不确定度含 Token 年龄/送达、belief 协方差、DD 支撑裕量、物理区间
  宽度与切换年龄。
- **no-harm 可组合性定理**（D0.78/D0.85/D0.86）：若两运动管满足速度/边界/扫描
  间隔、终点位置与速度相等、结构与 RF 冻结、Token 按分支可行送达交集更新、电量
  按分支最大扣减，则 `X_{t+H}^repair = X_{t+H}^baseline` 在协议表示精度上成立；
  全部未来确定性决策相等，总配对性能差恰为接受窗口内差之和。
- **e-process 漂移监测**：证书违例 e-value 越过 `1/δ` 时永久锁定 No-op（anytime
  界，依赖稳定条件）。

代码：`evaluation/episode_joint_conformal.py`、`self_normalized_feedback.py`、
`transition_certificate.py`、`channel_margin.py`、`physics_interval_gate.py` 等。

---

## 8. 学习目标（MAPPO + max-min 对齐奖励）

外层由 MAPPO 求解（CTDE，共享 actor + 集中 critic），内层由 §6 的 LP/搜索层求解。
`gamma=0.99`、`gae_lambda=0.95`、`ppo_clip=0.1`、per-target GAE bootstrap 已修复。

**团队奖励曲率**（D055）：

```text
r_team = Σ_q λ_q · U_κ(D_q) − λ_report·bits − 通信成本 − 约束惩罚
```

两种正交修正替代历史 `U(D) = -log(1-P_D)`（非凹、边际递增 → 富者愈富）：

1. **凹饱和效用** `U_κ(D) = 1 − exp(−κD)`（凹/单调/饱和）：使 `Σ ω_q U_κ(D_q)`
   恢复**单调次模**结构，贪心恢复 `(1−1/e)`；
2. **对偶价格加权** `λ = λ*`（§6.1 的最优 LP 对偶）：让学习梯度聚焦当前瓶颈
   目标，与协调器可认证的 Lagrangian 一致（"对偶一致奖励"）。

`reward_utility_mode ∈ {log, concave, maxmin_dual}`，默认 `log`；代码：
`environment/maxmin_reward.py`、`environment/reward.py`。

**动作分布密度精确化（B5 / advice 014 P0-1，smooth_disk 参数化）**：历史运动动作
路径为 Gaussian → tanh 盒 → 径向投影（`Δp ← (max_dp/‖Δp‖)·Δp`，多对一，~21% 盒外
区域触发），`log π(执行动作)` 不是执行动作的真实密度 → PPO importance ratio 数学上
不严格。修复参数化 `dp_parameterization: smooth_disk`（默认 `radial_clip` 保持
向后兼容）用一一光滑映射：

```text
Δp = d_max · z / √(1 + |z|²),   |det d(Δp)/dz| = d_max²/(1+|z|²)²
log π_Δp(Δp) = log N(z; μ, σ) + 2·log(1+|z|²) − 2·log(d_max)
逆映射 z = Δp / √(d_max² − |Δp|²)   （|Δp| < d_max 严格）
```

PPO ratio 恢复严格意义（近边缘动作也精确）。工程要点：decode/compute_log_prob 将
动作量化到 float32（与 buffer 训练精度一致，逆映射在近边缘病态，float64/float32
1-ULP 差会被放大）；evaluate_actions/verify 用 float64 中间量与 numpy 完全一致的
公式（`standardized = (z−μ)/max(σ,1e-12)`，避免 `var+ε` 扰动在 6σ 极端动作处放大）。
实现：`environment/action.py`（decode/compute_log_prob/decode_deterministic）、
`agents/mappo_agent.py`（evaluate_actions/verify）；回归
`tests/test_smooth_disk_action.py`（9 项：双射/Jacobian/密度公式/PPO ratio 精确性/
buffer 管线）。**待训练验证**：新训练跑 smooth_disk 后做一轮正确性验证 + 4/4 anchor
non-regression + 解析栈兼容性检查（advice 014：不重新调参）。

---

## 9. 符号 ↔ 代码 ↔ 单位映射

| 数学符号 | 代码变量 | 形状 | 单位 | 文件 |
|---|---|---|---|---|
| `p_k` | `uav.pos` | (3,) | m | `uav.py` |
| `v_k` | `uav.vel` | (3,) | m/s | `uav.py` |
| `E_k` | `uav.battery` | scalar | J | `uav.py` |
| `r_k` | `Action.role` | scalar∈{0,1,2} | — | `action.py` |
| `Δp_k` | `Action.delta_p` | (2,) | m/帧 | `action.py` |
| `τ,ν,α` | `tau,nu,alpha` | (K,K,Q) | s, Hz, 无量纲 | `geometry.py` |
| `d_eff(i,j,q)` | `DeflectionEntry.d_eff` | 每条目 | 无量纲 | `deflection.py` |
| `a_ijq` | `coefficient` | (K,K,Q) | 1/W | `physical_oracle_audit.py` |
| `a_iq`（fixed owner） | `fixed_owner_gain_matrix` | (K,Q) | 1/W | `maxmin_power.py` |
| `b_i` | `sensing_budget_w` | (K,) | W | `maxmin_power.py` |
| `p_iq` | `MaxMinPowerResult.power_w` | (K,Q) | W | `maxmin_power.py` |
| `λ_q` | `MaxMinPowerResult.prices` | (Q,) | 无量纲 | `maxmin_power.py` |
| `receiver_d[j,q]` | `receiver_d` | (K,Q) | 无量纲 | `evidence.py` |
| `D_q*` | `detection_D_q` | (Q,) | 无量纲 | `env_core.py` |
| `P_D^q` | `P_D_q` | (Q,) | ∈[0,1] | `detection.py` |
| `r_team` | `StepInfo.team_reward` | scalar | — | `reward.py` |
| `r_k`（shaped） | `shaped_rewards[k]` | scalar | — | `reward.py` |

---

## 10. 当前性能与可宣称边界

| 版本 | 种子 | mean-worst | QoS 可行率 | 判定 |
|---|---:|---:|---:|---|
| 4/4 冻结部署版 | 100 | 0.739 | 0.72 | 正式版本 |
| 6/6 原子控制 D0.85 | 20 | 0.6543 | 0.7303 | 历史基线（D0.85 原子控制；种子状态已隔离审计，见 §12） |
| 8/8 原子控制 D0.86 | 20 | 0.4372 | 0.50 | 严格 no-harm 证书成立 |
| **8/8 解析栈 L0+L1+L3（D0.95，gauge）** | 20 | **0.662** | **1.0\*** | 历史部署基线（satisficing L1，已被 D1.1-A 取代） |
| 8/8 lexicographic L1（D1.1-A，live） | 20 | **0.844** | **1.0**（LCB 0.839） | 与部署同种子同 warm-start，仅 L1 目标切换 |
| 8/8 lex + 多候选 L3（D1.1-B，live） | 20 | **0.975** | **1.0**（LCB 0.839） | 同种子，L3 视野增强 |
| **6/6 跨尺度 + 解析栈（D1.6，v2 bank test 前 20）** | 20 | **0.751–0.910** | **0.65–0.90** | 分解证据（v2 bank 前 20，偏乐观；三变体） |
| **6/6 跨尺度 + 解析栈（P1-3 独立采样 blind100）** | **100** | **0.659** | **0.530（LCB 0.433）** | **未过 0.70 门**（teacher @47 失败子集 QoS 0.723，Student 为绑定瓶颈） |
| **6/6 跨尺度纯零样本（无解析栈，作废基线）** | 20 | **0.293–0.344** | **0.10–0.30** | 配置不匹配（见下） |
| **8/8 盲测 D1.5（frozen 候选，100 全新 seed）** | **100** | **0.808** | **0.730（点估计过）** | **LCB 0.636 未过**（见下） |
| **8/8 盲测 D1.9（H=40 前瞻 L3，100 全新 seed）** | **100** | **0.962** | **0.950** | **LCB 0.888（历史 pre-fix）** |
| **8/8 盲测 D1.10（独立采样协议，100 全新 seed）** | **100** | **0.963** | **0.940** | **LCB 0.875（历史 pre-V3/pre-G2）** |
| **8/8 V3-C0 重认证（2026-08-18，同 D1.10 的 100 seed，独立采样，不调参）** | **100** | **0.915** | **0.910** | **LCB 0.838（最新可比 pre-G2 baseline）** |
| **6/6 V3-C0 重认证（2026-08-18，同 P1-3 的 100 seed，独立采样，不调参）** | **100** | **0.688** | **0.590** | **LCB 0.492 未过——冻结 Student 仍为瓶颈** |
| **6/6 Gate 1 multi-scale CE Student（2026-08-18，4/4+6/6+8/8 普通 imitation，同 100 seed）** | **100** | **0.842** | **0.780** | **点估计过 0.70；LCB 0.689 差 0.011（dataset shift 为主因，falsification 成立）** |
| **6/6 V3-T4 live A/B（9 seed：6 失败 + 3 PASS，V3 ON rev4 vs OFF）** | 9 | 0.5326（ON）vs 0.5217（OFF） | 3/9 vs 4/9 | **混合结果，如实判定无效，V3 默认 OFF**（见 §5.6.3/§6.8） |

> **证据冻结线**：本表全部性能值由 G2-0 能量归一化修正之前的代码产生。它们仍可用于
> 审计算法演进，但不能证明当前代码的绝对性能。表内“PASS”仅表示当时预注册 Gate；
> post-G2 当前状态为 `RE-CERTIFICATION REQUIRED`。Wilson 数字统一为 `z=1.96` 的
> 95% 双侧区间下端点。

> **历史 Gate 复核（`tools/assert_formal_gates.py --evidence-epoch historical`，
> 2026-08-16/17 输出）**：
> 4/4 冻结部署版、8/8 D0.95、8/8 lex L1、8/8 lex+多候选 L3 全 **PASS**（含 LCB
> 强制）；D1.9 / D1.10 blind 行在脚本中另列 **LCB 强制 PASS**（QoS 0.950/0.888、
> 0.940/0.875，见 [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) 最新 Gate 表）；**8/8
> V3-C0 重认证（2026-08-18）LCB 强制 PASS（QoS 0.910 / LCB 0.838）——V3-C0
> baseline**；6/6 污染项标 **QUARANTINED**（不参与判定），6/6 V3-C0 重认证
> **LCB 0.492 未过门**（Student 仍为绑定瓶颈）。这些状态全部属于 `pre_g2` epoch；
> 无参数运行 `tools/assert_formal_gates.py` 只检查 `post_g2` 当前证据，在新的 clean commit
> blind100 产物注册前必须以退出码 2 fail closed，历史 PASS 不再染绿当前门禁。

> **差距分解（同 20 seed、同 warm-start、同 selection split，实测配对）**：
> `0.662 → 0.844`（**+0.18，仅切换 `task_constrained_mode: gauge→lexicographic`**，
> 不动运动/结构）→ `0.975`（**+0.13，再加多候选 trust-region L3**，
> `analytical_movement_candidates_enabled`）。三行都是 **live eval-only 运行**
> （`run_mappo.py`，`_d095_lex20` / `_d095_lexcand20`），不是 oracle——**D1.1-A/B
> 现已作为部署配置接入 live 路径**（lex L1 + 多候选/前瞻 L3），原"更强配置尚未
> 切换为部署基线"的表述已随 D1.1-A 部署化关闭。注意：这组 seed 已多轮复用
> （D1_1A §5），**最终认证已由 D1.5–D1.10 完成**（100 全新 blind seed，见下表
> 盲测三行与 [`ALGORITHM_EVOLUTION.md`](ALGORITHM_EVOLUTION.md) D1.5/D1.9/D1.10）。

> **D1.5 盲测正式结果（2026-08-16，100 全新 seed，advice 013）**：
> frozen 候选（L0-KKT + Lex-L1 + P0-L2 + 多候选 L3）在从未暴露的 100 seed 上：
> **QoS feasible 0.730（点估计过 0.70 门）、Wilson LCB 0.636（未过 0.70）**；
> steady/weak3/worst 均值 0.808–0.820。三点结论：
> ① dev selection 的 0.975 是 seed 复用偏差，盲测真实水平 ~0.73（点估计）；
> ② 27/100 失败为**场景级**塌陷，主导因子是初始最差目标-最近 UAV 距离
> `worst_nearest`（<350 m QoS 0.91 → >550 m 仅 0.17）；v_max×episode 最大位移
> 375 m，实测 realized 距离与 worst P_D 相关 −0.863（>300 m 则 QoS 0），
> **>450 m 的 seed 处于时间可达性边界（部分物理不可达）**，300–450 m 是
> 策略可改进区（D1.9 候选：初始瞬态 warm start / 瓶颈优先趋近）；
> ③ 论文 `--require-lcb` 声明须如实报告 LCB 0.636 < 0.70（N=100 统计功效），
> 或改点估计 + 置信区间披露。详见 [`ALGORITHM_EVOLUTION.md`](ALGORITHM_EVOLUTION.md) D1.5。

> **D1.9 盲测闭合（2026-08-16，同 100 全新 seed，H=40 前瞻 L3）**：
> 单步 trust-region 停滞被 receding-horizon 评分修复（评分 H 帧持续趋近几何 /
> 执行 1 步）：**QoS feasible 0.950（95/100）、Wilson LCB 0.888（过 0.70）**，
> steady/worst 均值 0.962–0.963——**该 pre-fix 运行的 `--require-lcb` 成立**。
> 剩余 5 个失败 seed 中 4 个 worst_nearest > 450 m（episode 375 m 位移预算下
> 物理不可达，D1.5 已预告），仅 seed 615（355 m）为残余策略失败；
> **95/100 已近该运动学下的可达性上界**。详见
> [`ALGORITHM_EVOLUTION.md`](ALGORITHM_EVOLUTION.md) D1.9。

> **D1.10 独立采样协议认证（2026-08-17，同 100 全新 seed）**：共享 env 协议下
> `deflection_computer` 的 Rician/LoS rng 跨 episode 漂移（第 k 个 seed 的随机实现
> 依赖其前跑了多少 episode），D1.10-A 改为每 seed 独立 `UAVISACEnv` 实例
> （`eval_independent_env`，`_build_eval_env(seed=ep_seed)`）——**统计正确的采样，
> 非性能提升**（20-seed A/B 无系统性方向），默认 OFF 保持历史数值，论文复跑认证
> 应启用。认证结果：**QoS feasible 0.940（94/100）、Wilson LCB 0.875（≥0.70）
> 双过门**，steady/worst 均值 0.966/0.963、中位 1.000；与共享协议（0.950/0.888）
> 统计噪声内一致——**主结果对评估协议稳健**。残余 6 个失败 seed = **3 物理不可达**
> （886/45/185，worst_nearest 582–666 m > 375 m 位移预算）+ **3 策略边界**
> （615/298/613，255–479 m 可达区；seed 615 为功率耦合结构瓶颈，UAV 2 独拥 4 目标）。
> **D1.10-B**：L3 Phase-1 触发改 max-min t\* 三版（均匀 deficit / λ\* 加权 /
> phase1_force）均为**负结果（默认 OFF）**；**D1.10-C** per-UAV 结构重分配
> （对偶热点 + 稀疏 1-swap + 字典序接受，`congestion_relief.py`，默认 OFF）
> 已实现并 A/B 验证为**负结果（2026-08-17 关闭**：613 救活 vs 615/298 崩坏，
> LP 帧内改善不落地）。详见 [`ALGORITHM_EVOLUTION.md`](ALGORITHM_EVOLUTION.md)
> D1.10/D1.10-C。

> **P1-4 精确结构上界审计（2026-08-17，C2 / advice 014）**：20 个代表性
> final-resolved 帧（d035 teacher trace）上，阈值可行性 MILP + 二分给出
> **proven 联合结构+功率上界 worst P_D 0.9154 / 0.9197（宽度仅 0.0043，紧致）**，
> vs 固定结构功率 0.7924（**结构 headroom +0.123**）；对偶引导稀疏启发式达
> 0.8775（**恢复 76.1% 结构增益**、QoS 可行率分类与 exact 相同、仅用 23.9% 边）。
> 论文措辞升级为 exact/proven 上界 + 稀疏启发式 gap ~0.038（§6.6、
> [`ALGORITHM_EVOLUTION.md`](ALGORITHM_EVOLUTION.md) P1-4）。

> **复现性发现（2026-08-17，P1-5 诊断）**：同一 seed+配置下默认线程 vs
> `MKL_NUM_THREADS=1` 结果不同（8/8 seed 29：worst 0.84 → 0.76；共享认证进程
> 1.00）——解析栈的 LP/BLAS 归约对线程数敏感，**近地板 seed 的 QoS 判定随运行
> 环境摆动**。正式认证应统一走崩溃隔离进程协议（`tools/crash_isolated_seed_eval.py`，
> MKL=1 + per-seed worker + fail-closed），保证鲁棒性与数值可复现性一致。

> **V3-T4 live A/B（2026-08-17，advice/015）**：9 seed（6 失败 479/866/725/237/
> 410/274 + 3 PASS 14/752/705）独立采样、crash-isolated MKL=1。五轮迭代
> （rev1 单帧 → rev2 持久化+deficit → rev3 前瞻门 → rev3b 无持久化 → rev4
> relax 路由+提高守卫）**全部混合结果**：mean worst 最高 +0.042（rev2）但 QoS
> 计数从未超过 OFF（4/9）；rev4 均值 worst 0.5326 vs OFF 0.5217、QoS 3/9。逐 seed：
> 274 大幅改善（rev2 +0.494：0.123→0.617）、479/725/752/237 改善、866/410/705/14
> 退化。**判定：无效，默认 OFF**（不引入未认证退化）。根因 = 闭环策略耦合
> （提交→观测→冻结 Student 策略 comm/预算决策分叉→轨迹分叉；零动作探针证明
> V3 无提交路径与 OFF 逐位一致，退化 100% 来自提交）。

> **通信/感知原理审计修复（2026-08-17，3 组并行审计）**：未发现 P0 物理违规；
> 修复 4 项 P1 + 7 项 P2（详见 [`ALGORITHM_EVOLUTION.md`](ALGORITHM_EVOLUTION.md)
> "通信+感知原理审计与修复"）。关键数据：① 双尺度量化修复 η 主导时 π 压碎
> （V 0.149→0.765 恢复，argmax 保持）；② `comm_bits` 诚实公式（低估 5.29×：
> 294→1560 bit/轮 @K=Q=6，1 km 处串行化 9.3 ms > 5 ms deadline）；③ V3 三地板
> 路由（d_floor 11.18/13.07/15.46，消除 worst-only 盲区）；④ DD 门 Lipschitz
> 前瞻证书（`dd_gate_active_set_certificate`，fail-closed）。测试基线 **968
> passed / 1 env failure**（sklearn 既有）。

`*` D0.95 端到端严格比较（tol=0）报 QoS 0.65（13/20），其中 7 个 seed 的 worst
恰好钉在 0.60 地板（`0.60 − 1.11e-16`，浮点伪影，非真实性能差距）；按
`marl.qos_eval_tol=1e-6` 口径为 **QoS 1.0（20/20，Wilson LCB 0.839）**。口径修正
见 [`ALGORITHM_EVOLUTION.md`](ALGORITHM_EVOLUTION.md) D1.1 章节。

> **6/6 决策数据已回填重跑 + D1.6 失效分解（2026-08-16）**：
> ① 原 6/6 三行（0.543/0.645/0.635）来自 `980_k6q6` test split 前 10 个种子，
> **含全部 5 个被隔离种子**（795/747/105/860/2）且该批种子系统性偏乐观，原"均值
> 达标"结论被高估，已作废。
> ② 干净 test bank（`stratified_seeds_980_k6q6_v2.json`）前 20 种子重跑**无解析栈**
> 基线：worst 0.293–0.344、QoS 0.10–0.30（崩坏表象）。
> ③ **D1.6 分解（advice 013 情况 A）**：同一配置接上解析部署候选栈（L0 KKT +
> lex L1 + P0-L2 + 多候选 L3），worst 恢复至 **0.751–0.910、QoS 0.65–0.90**
> （`results/_6x6_d1_6_analytical_*_20/`）——**6/6 的失败主因是旧部署配置未启用
> 解析栈（configuration mismatch），不是跨尺度学习崩坏，无需重训 Student**；
> teacher 结构（oracle）worst 0.910 优于 Student 0.75–0.77，暴露 ~0.15 的 Student
> 校准 gap（后续可选校准）。详见 [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md)。

**8/8 解析栈端到端**（`analytical_*_enabled`，D0.87–D0.95，20 seed）：

| 配置 | worst | weak3 | steady | QoS 可行率 |
|---|---:|---:|---:|---:|
| C0 deployed | 0.355 | 0.508 | 0.773 | 0.40 |
| L0+L1（task-constrained） | 0.609 | — | 0.736 | 0.50 |
| **L0+L1+L3 几何** | **0.662** | **0.724** | **0.808** | **1.0**（tol=1e-6；严格比较 0.65 为浮点伪影） |

L3 几何把 steady 从 0.736 拉到 0.808（0/20 seed 低于 0.80），三地板首次端到端全达标；
剩余 7/20 seed 早期帧 worst<0.60 是滚动时域收敛瞬态。详见
[`ALGORITHM_EVOLUTION.md`](ALGORITHM_EVOLUTION.md) D0.95 章节。

**8/8 可达性上界（oracle 诊断）**（D1.0-A horizon joint，
`tools/audit_horizon_joint_oracle.py`，从教师最终几何起点、全局信息、多步规划；
**不是部署执行**）：

| oracle | worst | weak3 | steady | QoS 可行率 |
|---|---:|---:|---:|---:|
| horizon_joint（H=20，maxmin 内层） | 0.852 | — | — | 0.75 |
| horizon_joint（gauge 内层） | 0.60（satisficing） | — | — | 0.90\* |

`\*` gauge 行早期 QoS 0.90 含 γ 缩放功率 bug（γ*>1 时 Σp≤γ·b>1 W），已修正（见
[`ALGORITHM_EVOLUTION.md`](ALGORITHM_EVOLUTION.md) D1.1 章节）。

**结论（2026-08-16 修正，2026-08-17 更新部署口径）**：8/8 差距的主要来源**不是
几何/物理**，而是两个执行层选择：① **L1 目标 satisficing**——gauge 在 γ*≤1 后把
worst 钉在 0.60 地板，浪费满足门限后的剩余资源；lexicographic（QoS 约束 max-min）
在同一几何回收该资源（+0.18，live 已验证）；② **L3 视野**——单步一阶梯度 vs 每帧
多候选 trust-region 打分（+0.13，live 已验证），再由 D1.9 前瞻评分（H=40）闭合
单步停滞（盲测 QoS 0.730→0.950）。horizon joint oracle（0.852）与部署基线的差距
包含**同类的 L1/L3 增强 + 多步联合规划 + 全局信息**，作为可达性上界参考，不是
部署方法；**当前部署候选已达 20-seed 0.975 / 盲测 0.940（§10 上表），缺口收敛为
盲测 6/100 尾部（3 物理不可达 + 3 策略边界；D1.10-C 结构修复已 A/B 验证为负结果
关闭，耦合修复需 P0 偏好级机制，开放）**。

**8/8 天花板分解**（`tools/audit_scale_ceiling.py`，同几何瀑布）：

```text
deployed 0.31 -> power_only 0.675 -> single 0.700 -> duplex 0.706 -> relaxed 0.868
```

判定：**coordination_limited**，主导缺口是 `power_gap = +0.368`；快层 max-min 功率
LP 单独恢复 deployed→ceiling 缺口的 **65.5%**；几何层（L3）进一步把端到端 steady
推到 0.808（D0.95）。因此 8/8 的绝对性能不是物理天花板，而是部署功率/结构/几何
分配未做解析优化——现已逐层接入（§5.6）。

**可以宣称：**

- 分布式、共享参数、置换等变的结构 Student 在 4/4 达到 Medium 可部署门槛；
- 自适应精度 Token 在受限 U2U 信道下保持功率与原子传输约束；
- 6/6、8/8 静态目标域内，认证化原子控制器具有**严格 no-harm 可组合证书**；
- 双集合表示提供目标置换不变 / UAV 置换等变 / 均匀尺度协变 / 速度圆盘保证；
- 8/8 静态目标域内，解析执行栈（L0 通信余量 + L1 功率 LP/capability gauge + L3
  价格驱动几何下降）在 20 seed 上把 worst/weak3/steady 拉到 0.662/0.724/0.808，
  三地板全达标（D0.95；QoS 在 tol=1e-6 口径下 1.0）；
- **8/8 部署候选盲测认证（历史 pre-V3/pre-G2）**：L0-KKT + lex L1 + P0-L2 +
  多候选/前瞻 L3 在 100 全新 blind seed 上独立采样认证 QoS **0.940** / Wilson
  LCB **0.875**（≥0.70），steady/worst 均值 0.963（D1.10；D1.9 共享协议
  0.950/0.888，两协议统计噪声内一致）；
- 8/8 同几何可达性：lexicographic L1（live）worst 0.844、QoS 1.0（20/20）可达；
  horizon joint oracle（离线诊断上界）worst 0.852——**性能缺口是协调/部署执行，
  不是物理**。

**不可宣称：**

- 完全分布式端到端物理检测（最终融合仍是环境级集中式）；
- 任意 `K/Q` 的检测/QoS 通解，或 8/8 已达与 6/6 相同绝对 QoS；
- 学习候选优于解析物理候选（当前接受动作均来自解析梯度池）；
- horizon joint oracle 为可部署执行路径（目前是离线诊断上界，非部署方法；
  lexicographic L1 已是 live 部署配置，不属本条）；
- 波形级或真实硬件 ISAC 性能；
- 6/6 跨尺度"不可行"（D1.6 已证：接入解析部署栈后 worst 恢复 0.75–0.91、QoS
  0.65–0.90，原失败是旧配置未启用解析栈；残差为 Student 校准 gap ~0.15）。

---

## 11. 开放项与下一步

> **D0.89–D0.95 已闭合下述 1、2 两项**：快层 max-min LP 已接入部署路径
> （`analytical_sensing_power_enabled`，D0.89-A），慢几何层已以
> `analytical_movement_enabled`（价格驱动的赤字→能力下降 + 达标悬停）实现并端到端
> 验证（D0.95：20 seed 上 worst 0.662 / weak3 0.724 / steady 0.808，三地板全达标）。
> 详见 [`ALGORITHM_EVOLUTION.md`](ALGORITHM_EVOLUTION.md) D0.95 章节。

1. ~~**功率层接入部署执行路径**~~（已闭合，D0.89-A）：快层 max-min LP 已成为部署
   执行主路径，8/8 worst +0.227。
2. ~~**慢几何层闭环实现**~~（已闭合，D0.95；早期瞬态由 D1.8/D1.9 进一步闭合）：
   `analytical_movement_enabled` 价格驱动几何下降已闭环；原 7/20 seed 早期帧
   worst<0.60 的滚动收敛瞬态由 D1.8（帧-0 warm start）+ D1.9（前瞻评分）消除
   （100 blind seed QoS 0.730→0.950，§10）。
3. **max-min 对齐奖励的端到端训练消融**：`log vs concave vs maxmin_dual` 尚待
   独立多 seed 训练验证。
4. **非方形规模**（6/8 或 8/6）：检验 `K≠Q` 时置换→广义分配的语义断裂与 RF 可行
   域、协议缩放。
5. **随机物理残差校准**：随机 RCS/Swerling、随机 CSI、丢包与模型漂移尚未进入
   独立事件校准；此前 `3 dB / 0.5 ms` 只是工程裕量。
6. **统计功效（主结果已切换，其余 Gate 待补）**：mean-worst 是重尾统计量，30 种子
   不足以分辨 0.6 门槛；主指标应改用 `QoS feasible rate`（带 Wilson LCB）并扩到
   ≥100 种子或加方差缩减。已新增 `tools/assert_gate_thresholds.py` /
   `tools/assert_formal_gates.py` 把 Medium 门槛（含可选 Wilson LCB 强制）脚本化。
   **8/8 主结果已完成切换**（D1.10 盲测 QoS 0.940 / LCB 0.875，`--require-lcb`
   成立）；**仍未切换**：6/6（D1.6 仅 20 seed 分解）与旧 20-seed Gate；4/4 的
   LCB 0.625 不达 0.70（方法学中须显式说明）。
7. **对抗检测约束（advice 012，oracle 级已落地 + live 功率层已接入）**：T2 的 exposure 代理量
   （`E_w ≤ Γ_w`）已在 oracle 侧升级为**对方探测能力约束**（`D_w^I ≤ D̄_w^I`，
   三个对方能力等级 weak/medium/strong，`--inner intercept`），三向对比证明
   power-only 与 exposure 在 medium/strong 对手下 75%/100% 被裸发现，而
   detection-constrained 恒成立（P_D^I ≤ ε）。见
   [`ALGORITHM_EVOLUTION.md`](ALGORITHM_EVOLUTION.md) T3 章节。
   **D1.1-D/E（2026-08-16）已把隐蔽性接入 live 功率路径**：`intercept_constrained_power_enabled`
   使执行功率满足 `P_D^I ≤ ε` 硬约束（`constrained_maxmin_lp`），lex 模式下与 QoS 地板
   联合进同一 LP（`qos_constrained_maxmin_lp` 加 intercept 行，三族价格 λ/π/μ）。
   端到端 4 UAV × 2 seed × 30 帧实测：所有档位 0 违反（约束跨帧保持），QoS 代价随
   对手强度单调（off/weak 0.703 → medium 0.663 → strong 0.001）——**"strong 对手
   必须静默"的 oracle 结论在 live 路径复现**。**D1.7（2026-08-16）已证明攻防对偶
   价格 `s_iq = λ_q a_iq − μ_w a^I[i,q]` 的本地 bid + Dantzig–Wolfe 列生成与中央
   LP 收敛至机器精度**（对偶 gap ≤3.6e-15，均值 5.8e-16，`tests/test_dw_column_generation.py`
   回归通过）——T3 隐蔽性约束可精确分布式化（理论闭合）；**完整分布式协调器
   （每帧 RMP + 价格广播 + 本地 bid 执行）为下一步实现**，exposure 正式降级
   为 baseline。
8. ~~**lexicographic L1 部署化**~~（已闭合，D1.1-A）：lex L1 已是 live 配置
   （`task_constrained_mode: lexicographic`，20 seed worst 0.844→0.975 与多候选
   L3 组合），不再是"待部署求解器"。**D1.5 盲测已执行**（2026-08-16，100 全新
   seed，frozen 候选：L0-KKT + Lex-L1 + P0-L2 + 多候选 L3）：QoS feasible
   **0.730（点估计过 0.70）**、Wilson LCB **0.636（未过）**——dev 0.975 的复用
   偏差被量化，真实点估计 ~0.73。左尾分解确认失败由初始 `worst_nearest`
   主导（300–450 m 为策略可改进区，>450 m 部分物理不可达）。**D1.9 瓶颈前瞻
   L3 已实现并全量认证**（`analytical_movement_lookahead_frames=H`，
   receding-horizon：评分 H 帧持续趋近几何 / 执行 1 步）：100 blind seed
   QoS **0.950**、LCB **0.888**（过门）。**D1.10 独立采样协议认证**：
   QoS **0.940**、LCB **0.875** 双过门，与共享协议一致；这是历史 D1.10 结论，
   已由 V3-C0 物理修复口径和 G2-0 量纲修订依次覆盖。
   残余 6/100 seed = 3 物理不可达（>582 m）+ 3 策略边界（255–479 m）；
   D1.10-B 的 t\* trigger 三版均为负结果（默认 OFF）；**D1.10-C per-UAV
   结构重分配已实现但 A/B 为负结果（613 救活 vs 615/298 崩坏，关闭**，
   耦合修复需 P0 owner 偏好级机制，开放项）。详见
   [`ALGORITHM_EVOLUTION.md`](ALGORITHM_EVOLUTION.md) D1.5/D1.9/D1.10。
9. **6/6 跨尺度（2026-08-16 已分解；P1-3 盲测定性完成，2026-08-17）**：D1.6
   失效分解证明 6/6 的"崩坏"是**旧部署配置未启用解析栈**（情况 A）——接入解析
   部署候选栈后 worst 恢复 0.751–0.910、QoS 0.65–0.90（§10，v2 bank test 前 20）。
   **P1-3 独立采样 100 全新 blind seed 认证（冻结 Student + 解析栈）：QoS
   0.530（53/100）、Wilson LCB 0.433——决定性失败**；**teacher（P0 集中式结构）
   在 47 个 Student 失败 seed 上 QoS 34/47 = 0.723（43/47 显著改善，均值 +0.49，
   保守外推全 100 teacher QoS ≥ ~0.87）——Student 结构代理（4/4 锚点基数残差
   的 6/6 近似，gap ~0.15）是 6/6 认证的绑定瓶颈**。诊断要点：
   `--structure-student-adaptive-min-bits-per-dim 4` 为 6/6 必带参数（缺失时
   student 通道塌缩到 P_FA 地板）。**Gate 0（2026-08-18，V3-C0 重认证，同 100
   seed 不调参）确认该定性对物理修复稳健**：6/6 QoS 0.590 / LCB 0.492 仍未过
   0.70（与 P1-3 的 0.53/0.43 统计噪声内一致）。**Gate 1（2026-08-18，
   multi-scale CE falsification）**：4/4+6/6+8/8 干净 trace + 普通 imitation
   训练后同 100 seed 认证 **QoS 0.780（点估计过 0.70）/ LCB 0.689（差 0.011）**，
   mean worst 0.66→0.84——**dataset shift 是 6/6 失败主因**（advice/016 §17
   答案：普通补数据基本够，差统计尾巴）；**下一步（Gate 2）**：同一数据 +
   task-regret/dual-weighted 校准补 LCB 尾巴。
10. **V3 dual-priced L2（advice/015，已执行到 T4，A/B 无效）**：dual-priced
    分布式结构协调（§5.6.3/§6.8）已完成 T1（资源竞争诊断）→T2（原型）→T3
    （分布式价格迭代）→T4（live 集成 + 单调门 + 有限切换），9-seed A/B 五轮
    迭代**全部混合结果**（QoS 从未超过 OFF）——**如实判定无效，默认 OFF**。
    科学结论：单调门保证提交帧精确 LP 的 t* 非降，但 live 中提交→观测→冻结
    Student 策略轨迹分叉（闭环策略耦合），结构层可缓解（274）不可根治（866）；
    与 V3-T1"竞争×弱几何"一致。**下一步候选**：① Student 跨尺度校准（P1-3
    绑定瓶颈，最高优先）；② V3 baseline 全量重跑（χ_rep/能量修复后 8/8 + 6/6
    独立采样认证，不调参）；③ 若启用 V3，先接入价格消息到物理 U2U 信道
    （当前诚实记账未承载）。

---

## 12. 工程治理与可复现性（2026-08-16 审计修正）

研究内容之外，本目录还记录了保证结果可信的工程治理机制（本次深度审计的修正成果，
详见 [`ALGORITHM_EVOLUTION.md`](ALGORITHM_EVOLUTION.md) 的工程审计章节）：

- **测试集污染种子隔离（代码强制 + 已回填）**：seed `795/747/105/860/2` 因
  2026-07-29 split 漏写被误用，文档此前只有声明、代码零拦截。现注册表
  `config/quarantined_seeds.json` + `load_stratified_seed_split(strict=True)`
  fail-closed 拦截（含隔离种子即抛错），bank 生成自动排除并记录
  `quarantined_excluded`。**980_k6q6 已回填**（`stratified_seeds_980_k6q6_v2.json`：
  干净 test split、无 split 重叠、selection/confirmation/stress 不变），6/6 决策
  行已用干净 20 种子重跑（见 §10）；**`800_q4` 与 `1130_k8q8` 已回填 v2 bank
  （2026-08-17，P2：隔离种子替换为同难度干净种子，strict 加载通过）**——历史
  manifest 引用的旧 bank 路径仍指向含隔离种子的原文件，新运行须显式切 v2。
- **Gate 门槛脚本化**：`tools/assert_gate_thresholds.py`（单结果断言，从
  paired_eval.csv 重算聚合）与 `tools/assert_formal_gates.py`（默认只断言 post-G2 当前
  epoch；`--evidence-epoch historical` 才显示历史表）。受污染项标 QUARANTINED 不参与
  判定；历史/探索性未过项标 `DISCLOSED_FAIL`，且整个 pre-G2 表不能改变当前门禁退出码。
  输入 episode 数组必须等长、有限、
  概率位于 `[0,1]`，若含 seed 列则必须逐 episode 唯一。4/4 正式结果经脚本复核 PASS
  （0.9132/0.8848/0.7393/0.72）；6/6 重跑三变体经脚本判定 FAIL（四地板不达标）。
- **协调层主/支路径立界**：`uav_isac/coordination/__init__.py` 导出主路径 API；
  14 个仅审计/研究用模块（priced_structure、certified_geometry_repair、
  owner_local_physics 等）带 `AUDIT/RESEARCH-ONLY` 标注，不代表部署行为。
- **结果治理**：`tools/audit_results_tree.py` 只读扫描 results/——原 767 目录中
  36 个无 manifest、20 个 `_` 前缀临时目录混存、summary.json 有 87 种 schema
  变体；2026-08-16 清理已归档 126 个过时目录（90 个 paper-era + 11 个无引用
  scratch + 26 个无引用无 manifest）至 `results/_archive/`，顶层现 641 个目录。
- **运行证据链（2026-08-20）**：`scripts/run_mappo.py` 在启动时生成确定性
  `source_snapshot.zip`，并在 manifest 写入源码哈希、解析后配置及其哈希、Git commit/
  dirty 状态、命令行、Python/平台/依赖版本以及 warm-start/Student/ranker/factor-graph
  等检查点 SHA-256。配置加载器严格拒绝未知键，避免拼写错误或旧 YAML 字段被静默丢弃。
- **随机性与精确重放（2026-08-20）**：训练动作空间显式播种；
  `env.reset(seed=...)` 会重绑定 core、动作映射和感知衰落计算器的同一 RNG。环境快照
  现包含 P0 cache、结构选择、V3/拥塞修复计数、错误状态和通信 bit 计数，保证同状态
  反事实不因隐藏控制状态分叉。
- **工程基线**：`requirements.txt`、`constraints-ci.txt`、
  `.github/workflows/ci.yml`（Windows 参考环境全量 pytest 与严格身份门禁）、`pytest.ini` 排除 scripts/
  （`test_ppo_ratio_fix.py` 曾模块级执行训练被 pytest 误收集）。全量测试基线
  2026-08-20 G2-1A 基础设施在 Windows/MKL 稳定边界下分两进程回归：非 belief
  `1036 passed`、belief `14 passed`，合计 **1050 passed**、无断言失败。单进程仍可能在 MKL
  `eigvalsh` 内原生中止，因此不得把分片结果伪装成一次单进程通过。

---

## 13. 当前问题与不足（分层清单，2026-08-17）

> 本清单汇总系统当前存在的问题与不足，按性质分层；详细证据见
> [`ALGORITHM_EVOLUTION.md`](ALGORITHM_EVOLUTION.md) 与
> [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md)。

### A. 性能与认证缺口

- **8/8 D1.10 历史盲测尾部 6/100**（pre-fix QoS 0.940 / LCB 0.875，但非 100%）：
  3 物理不可达（886/45/185，`worst_nearest` 582–666 m > 375 m 位移预算，属
  场景-运动学可达性边界，需改 v_max/episode 时长才能 100%）+ 3 策略边界
  （615/298/613；seed 615 为功率耦合结构瓶颈）。**D1.10-B（t\* trigger）与
  D1.10-C（per-UAV 结构重分配）均为负结果关闭**；耦合稀缺修复需 P0 owner
  偏好级机制（开放）。
- **6/6 未过独立采样盲测**：冻结 Student + 解析栈 QoS **0.530 / LCB 0.433**
  （100 全新 blind seed）；**teacher（P0 集中式结构）在 47 个 Student 失败 seed
  上 QoS 0.723**——**Student 结构代理（4/4 锚点基数残差的 6/6 近似，gap ~0.15）
  是 6/6 认证的绑定瓶颈**。Student 跨尺度校准为下一科学瓶颈。**V3 结构层
  （dual-priced L2，§5.6.3/§6.8）A/B 无效**（9-seed 混合结果，默认 OFF）——
  与 V3-T1 诊断一致：**竞争 × 弱几何（η_max 1.5–5.1 vs PASS 9–63）才是失败
  成因，结构重分配单独不足以救 6/6**（结构可缓解、几何/策略层根治）。
- **统计口径未全量切换**：6/6（P1-3 100-seed 已用 QoS+LCB）与旧 20-seed Gate 中
  部分仍以 mean-worst≥0.60 为主指标；4/4 的 LCB 0.625 不达 0.70（方法学须显式
  说明）。近地板 seed 判定对 BLAS/MKL 线程数敏感（复现性，见 §10）。

### B. 科学正确性（待办 / 已修复）

- **B5 smooth_disk 待训练验证**：参数化已实现（默认 OFF），新训练须做一轮正确性
  验证 + 4/4 anchor non-regression + 解析栈兼容性检查（advice 014：不重新调参）。
- **~~deflection χ_rep 每 (i,j) 重抽~~（已修复，V3-C0）**：改每接收机单抽；该修复
  改变随机实现 → 8/8 D1.10 保留历史，建立新 V3 baseline。
- **~~能量记账占位角色~~（已修复，V3-C0）**：P0 角色推导后按实际角色扣能量。
- **V3-T1 诊断发现**：6/6 失败 seed 的 owner 负载严重（单接收机拥 4.6–6.0/6 目标）、
  TX 复用 2.1–2.7——资源竞争真实存在；但 PASS seed 竞争占比同样高——**竞争 × 弱
  几何（η_max 1.5–5.1 vs PASS 9–63）才是失败成因**，结构重分配单独不足以救 6/6。
- **V3-T2 对偶一致 L2 原型**：`dual_priced_structure.py`（bipartite matching、
  TU 整数性、i≠j 物理约束、κ 切换惩罚）；盈余捕获成立（t* 6.0→7.5），naive
  全量匹配非 t\* 单调（强 TX reduced-cost 恰为 0）——需"盈余提交 + κ 稳定性"。
- **V3-T3/T4 分布式价格迭代 + 盈余提交门**（`dual_priced_auction.py`）：TX 广播
  量化价格，目标本地累计 `V_jq=Σ_{i≠j}[π a−η]_+` 竞价；单目标
  坐标更新 + 精确 t* 字典序门 → 接受序列 t* 单调非降（构造性零退化）。真实帧
  验证（6/6 失败 seed 178 帧）：**123 帧 t* 提升（69.1%，均值 +3.17）、0 退化**。
- **~~V3-T4 live 集成~~（已实现，A/B 无效，默认 OFF）**：`v3_dual_priced_structure_enabled`
  （默认 False）+ hold 窗口 20 + 单调门 + 有限切换 + `_v3_comm_bits_total` 诚实
  记账；9-seed 独立采样 A/B 五轮迭代**全部混合结果**（QoS 从未超过 OFF 4/9，
  rev4 3/9），**如实判定无效**。根因 = **闭环策略耦合**（提交→观测→冻结
  Student 策略 comm/预算决策分叉→轨迹分叉；单调门只保证提交帧精确 LP 的 t*
  非降，无法保证闭环 episode 级非降）——与 V3-T1"竞争×弱几何"一致，瓶颈在
  Student 校准（§13 A）。零动作探针证明 V3 无提交路径与 OFF 逐位一致
  （退化 100% 来自提交）。
- **V3 通信/感知审计修复（2026-08-17）**：① 价格量化改**双尺度**（π 单纯形与
  η 预算对偶各用自身 max 归一，再精确重建公共尺度）——旧公共尺度在 η 主导时
  压碎 π（V 全 0、argmax 退化）；② `comm_bits` 改诚实公式（每 TX 广播 K·Q 个
  (owner,target) 增益，1560 bit/轮 @K=Q=6，而非旧 294 bit；1 km 处串行化
  ~9.3 ms > 5 ms deadline，价格交换仅短距/少轮可行——通信-算法耦合显式化，
  量级经 `_v3_comm_bits_total` 记账）；③ V3 结构层从单地板升级为**三地板**
  （worst/weak3/steady 对应 d_floor 11.18/13.07/15.46，从
  `task_constrained_qos_floors` 读取）——单调门/前瞻门/近端守卫验证全地板，
  消除"QoS 判定三地板、救援只救最差"的结构性盲区；④ 前瞻门补 **DD 门
  Lipschitz 证书**（外推几何用 `dd_gate_active_set_certificate` 判定门是否可能
  翻转，不确定边 fail-closed——外推只重缩放 α² 时门翻转不可见）。
- **随机物理未独立校准**：Rician/Swerling/随机 CSI/丢包尚未进独立事件校准；
  此前 3 dB / 0.5 ms 只是工程裕量（P2 待办）。
- **advice/016 task-regret 证书（2026-08-17，前两步落地）**：`structure_regret.py`
  实现结构误差 → max-min 能力损失 Lipschitz 界（`|t*(A)−t*(Â)| ≤ Σ_i b_i
  ‖Δa_i‖∞`，真实 6/6 帧零违例）+ QoS 保持条件（ε_struct < m ⟹ floor
  preserved）+ decision-preserving bit 下界（`B > log₂(1+R/Δ)`，即
  `floor(log₂(1+R/Δ))+1`，真实数据
  mean 6.6 / min 2 / max 13 bit）+ 事件触发（margin > 2E_stale+2ε_B ⟹ 不发）
  + dual-weighted 边权（`w_iq=π_q·p_iq`，瓶颈目标 4.2× 集中）。**真实数据
  关键发现**：6/6 失败帧的 teacher 结构本身 t\* 均值 −7.42、0% 过地板——
  Student 校准必要但不充分，瓶颈是 Student 误差 × 弱几何 × 闭环敏感性的
  三重机制（修正"Student 是唯一瓶颈"的简化判断）。标量原语保留，但 learned
  token 在线接线因缺少 payload→action 证书而 fail-closed，供未来显式 score-token
  协议与 Student 校准使用。

### C. 理论边界

- **C2/C5**：结构层 MILP 有对偶间隙（已用精确 MILP 量化启发式 gap ~0.038，
  §6.6）；认证 staleness 界 2B 松约 900 倍，只能当安全证书、不能当触发。
- **C3**：三层对偶地位不同（功率精确 / 结构松弛 / 几何次梯度），λ* 在几何层
  退化时非唯一（熵/近端正则化待接线）。
- **V3 结构侧单调门系列（快照/前瞻/路由/守卫）无法根治弱几何退化**：单调门只
  保证提交帧精确 LP 的 t* 非降，live 中提交→观测→冻结 Student 策略轨迹分叉
  （866 类）→ episode 级无法保证——结构层可缓解（274 +0.494）、不可根治
  （866 −0.267）。**价格交换未接入物理 U2U Token 信道**（诚实记账
  `_v3_comm_bits_total` 暴露量级，但消息未纳入 L0 payload/功率/时延核算——
  接入属工程预备，V3 OFF 前提下无实际影响）。
- **T3 隐蔽性对抗**：strong 对手下 QoS 坍缩（live <0.30、oracle ~0.001）——
  1 W/28 GHz 下"必须静默"是物理结论；同时保 QoS+隐蔽性需波形层设计（扩频/LPI/
  波束成形），超出当前解析层边界。

### D. 工程与数据债

- **隔离种子回填**：`800_q4`/`1130_k8q8` 已回填 v2 bank（strict 加载通过）；
  历史 manifest 中引用的旧 bank 路径仍指向含隔离种子的原文件（回填后新运行须
  显式切 v2）。
- **文档/论文债**：`paper/` 手稿仍为 2026-07-22 旧稿（仍把 D1.10 当主结果、未按 6/6
  机理、C2 措辞更新）；文档编号 D0.87–D0.95 ≡ D0.10–D0.18 双轨并存（总纲 §1.4
  有映射）。旧 `_orphan_v3.txt` 一次性快照已由 V2 全量 catalog、SHA-256 catalog 和
  results-tree 治理审计替代并删除；未分类历史结果继续按 catalog 状态隔离，不再依赖根目录
  临时清单。
- **遗留代码**：`residual_actor.py` 无法工作（遗留 opt-in，崩溃即 fail-loud）；
  trainer oracle 引导探索（`_oracle_alpha`）从未接线（死功能）；maxmin 精确 LP
  的 `prices` 返回占位值（API 陷阱，现无消费者）。

### E. 已知且接受（设计权衡）

- 4/4 冻结部署版的尾部（5/100 seed worst<0.1，bottom-20% CVaR 0.288）——
  接受为已认证版本的已知分布，不作追尾优化。
- 通信/计算：8/8 协议 20.3 kbit / 43.3 ms 随 KQ 上涨（F1）；全图 MILP 结构求解
  是教师/参考、不可部署（F2，局部候选图/分布式列生成是后续）。
- MKL `eigvalsh` 并发中止（Windows 运行库问题）——已用进程隔离包装器兜底，
  未根治。

## 14. G2-1A 当前执行口径（2026-08-20）

每个 episode 的正式 QoS 事件定义为三个检测地板同时满足：

```text
I_e = 1{steady_e >= 0.80-tol,
        weak3_e  >= 0.70-tol,
        worst_e  >= 0.60-tol}.
```

点估计为 `p_hat = sum_e I_e/n`，置信量使用 Bernoulli episode 的 Wilson 下界；数值实现
强制截断到 `[0,1]`，避免 0 成功时出现负的机器舍入残差。不得把帧或目标当作额外独立样本。
跨 seed 汇总必须从 `eval_episode_*` 数组重算 steady/weak3/worst、QoS 与 Wilson；其余均值型
遥测取 seed 均值，真正的最大误差量取最大值，禁止沿用第一个 worker 的标量摘要。

感知功率审计要求总功率精确匹配 `K P_sense_max`，其中 `P_sense_max=0.0251 W/UAV`；
4/6/8 UAV 分别为 `0.1004/0.1506/0.2008 W/frame`。`max_power_balance_error_w`
是逐节点分配恒等式的残差/未用 headroom 诊断，不能不经定义就解释成预算违规。
当前 10-seed 批次四系统均通过数组、统计、通信概率守恒与感知总功率检查，但尚不能代替
逐 UAV cap telemetry；后续正式 100-seed 认证必须保留该限制说明。

### 14.1 跨尺度 worst 的理论边界

保持 UAV 密度和 `K/Q=1` 不足以保证 worst 尺度不变。在 IID 理想化目标难度模型中，
若单目标最近 UAV 距离的 CDF 为 `F_R(r)`，则 Q 个目标的最差最近距离满足

```text
P(R_max <= r) = F_R(r)^Q.
```

该式只解释基本极值趋势；真实目标共享 UAV、角色、matching 和功率预算，difficulty 相关，
不得把 `F_R^Q` 当作真实 bank 的精确分布。真实尺度效应必须另行报告 empirical
second-nearest、matching-bottleneck 和 capability distribution。对近似对称双基地链路，
`D_q ∝ p/(R_tx^2 R_rx^2) ≈ p/r^4`，距离尾部会被四次方放大。当前区域边长按
`sqrt(K)` 增长，绝对机动半径保持 375 m，使机动半径/区域边长从 4×4 的 0.469 降为
6×6 的 0.383 和 8×8 的 0.332。双基地任务还依赖第二近端点、TX/RX 角色和匹配瓶颈，
所以单一 nearest-distance 不是充分统计量。

物理 feasibility oracle 现在显式同时接受联合 RF 上限与独立 sensing PA 上限；每 UAV
可用于 oracle 的 sensing budget 为

```text
b_i = min(P_isac_total - P_comm,reserve, P_sense_max).
```

禁止再把约 0.97 W 的联合预算未用余量解释成可追加感知功率。旧 oracle 若未传
`P_sense_max=0.0251 W`，只能视为不满足当前 PA 约束的乐观诊断，不能用于性能归因。

### 14.2 Cap-aware 双基地能力与 shadow routing

对已经包含双腿传播、DD 门和 reporting 衰减的 per-watt 系数 `a_ijq`，定义

```text
b_i = min(max(P_isac_total - P_comm,i, 0), P_sense,max),
Gamma_geo = min_q max_{i!=j} b_i a_ijq,
C_q^owner = max_j sum_{i!=j} b_i a_ijq,
D_q^relax = sum_i b_i max_{j!=i} a_ijq.
```

`Gamma_geo` 是单对双基地瓶颈能力，`D_q^relax` 允许每个目标独占所有 UAV 预算，并放松
跨目标共享、公共 owner、角色和容量约束，因此是逐目标同几何上界。中间量
`C_q^owner` 要求同一目标的贡献汇聚到一个公共接收 owner，但仍放松跨目标预算共享、
单角色和接收容量，严格满足 `single-pair <= owner <= relaxed`；它只是描述量，不是可部署
分配。

由于实际任务同时约束 worst、bottom-k mean 和全局 mean，不能把它错误压成统一逐目标
地板。对 `P_D` 定义保留三项语义的 normalized task-feasibility ratio：

```text
r_task(P_D) = max(rho_min/min_q P_D,q,
                  rho_tail/mean(bottom-k(P_D)),
                  rho_avg/mean(P_D)).
```

`r_task<=1` 当且仅当三项地板全部满足。它不是 Minkowski/resource gauge：`P_D(D)` 非线性，
该比值不具备功率正齐次性。资源缩放意义只保留给 L1 的 `gamma_res*`。规范 shadow router
应采用的严格逻辑是：

```text
fixed gauge <= 1                         -> Region I / Hold
fixed gauge > 1 and feasible joint <= 1 -> Region II / L2 repair
relaxed upper-bound gauge > 1            -> Region III / L3 geometry
otherwise                                -> Region U / unresolved shadow
```

Region U 必须保留：当前 joint oracle 是交替混合离散—连续求解，未找到可行解不是全局不可行
证明。只有 relaxed 上界也过不了地板才可 certified route L3。实现位于
`uav_isac/coordination/scale_capability.py`，当前没有 live controller hook。

历史 G3-A 工具实际使用 trace 的 deployed `physical_pd` 作为第一层，而不是重新求
fixed-structure capability gauge；因此旧 Region II 只证明“deployed plan 失败、joint witness
成功”，不能单独归因为 L2。当前 trace 又不再含 `physical_pd` 和 sensing-power tensor，禁止
以 receiver-local `local_pd` 冒充当前 environment-level deployed detection。

2026-08-20 post-G2 开发 trace 的 G3-A 抽样归因为 `I=4/25、II=21/25、III=0、U=0`：
所有固定结构失败帧都有同几何联合结构—功率可行 witness，因此当前样本支持 L2 修复，
不支持 L3 几何重构。G3-B 帧级 Spearman 中 single-pair/owner/relaxed/matching-distance
分别为 `0.536/0.184/0.310/0.440`；owner 指标未优于距离代理，禁止作为直接 L3 目标。
两项均为 5 个已查看开发 seed 的机制审计，不是性能 Gate 或泛化结论。

功率水床用 `c_q=P_q/sum_r P_r`、`C_max=max_q c_q` 和 `H_P=sum_q c_q^2` 描述。
这些是瓶颈诊断量，不是正则化目标；在 max-min 任务中高集中度可能是正确 KKT 响应，禁止
为了让功率“更均匀”而牺牲三地板公平性。

### 14.3 G4-A：最小干预三地板联合修复 oracle（shadow only）

对 Region-II 帧，在同一 MILP 内联合决定 binary TX/RX role、每目标唯一 receiver owner、
双基地 support edge 与 perspective-linked edge power。功率流满足
`0<=w_ijq<=b_i x_ijq`、`sum_jq w_ijq<=b_i`，故同一 TX 功率不会被多个接收边重复计数。
检测概率使用 conservative chord PWL 下界，三地板由 worst rows、全局 average row 和
bottom-k 的精确 order-statistic 线性化共同约束。

目标不是继续最大化单帧性能，而是三阶段字典序 satisficing：

```text
min_lex( affected targets + participating UAVs,
         current-layout prepare payload bits,
         total sensing power )
subject to r_task(conservative P_D) <= 1.
```

参与者 closure 包括所有 changed-role UAV、切换边的两个端点以及 changed target 的新旧
owner；因此角色变化触发的隐含全图改写不能伪装成小块。当前实现
`minimum_intervention_repair.py` 是集中式全信息设计 oracle，没有 live hook，也尚未执行
L0 over-air admission。当前 bit 项精确对应已有 role/owner/edge/certificate wire layout，
但尚未包含未来可能需要的显式量化 power record；因此它是当前布局计数，不是完整 TS-AJR
通信开销结论。

本轮还发现原 curvature PWL 在 `d_max` 达数万时可能只生成一条从 worst floor 跨到 ceiling
的长弦，把真实 `P_D≈0.9998` 压成约 `0.6102`。新增 saturating chord：只在达到
`P_D=0.999` 前自适应分段，之后加入常数 `0.999` 下界；由 `P_D(D)` 单调性，该常数在高
Deflection 尾部仍是严格下界。此修复目前只供 G4 shadow oracle 使用，未改变冻结 G2-1A
live 路径。

### 14.4 G4-B：任务排序的依赖闭包嵌套 block（shadow only）

令 `B_l=(U_l,Q_l,R_l)` 分别表示允许参与边/owner 改动的 UAV、允许改变的目标和允许角色
翻转的 UAV。构造保证三集合逐层包含，MILP 在集合外冻结相应 binary 变量，因此
`F(B_l) subseteq F(B_{l+1})`；一旦某层有可行 witness，所有更大层至少保留该 witness。
“参与某条切换边”与“允许角色翻转”被严格分开：只有后者才把该 UAV 的全部既有 incident
targets 纳入 dependency closure，避免人为把局部边操作扩成全图。

候选目标权重来自当前 `r_task` 最大违约分支的一个非负子梯度支持；候选边使用
`omega_q b_i a_ijq`。G4-B 比较两种不带调参权重的排序：裸 gain 与 gain/新增 closure
成本。两者都仅负责 proposer；每层是否可行仍由完整三地板 MILP 决定。

在 21 个开发帧上，两者最终均恢复 21/21，但首次可行块的中位规模都是 5 UAV、6 targets，
且只有 4/21 在不超过 3 UAV、3 targets 的块内可行。裸 gain 在 full block 前恢复 16/21、
匹配 exact minimum closure 13/21；closure-efficiency 分别为 13/21 和 15/21，存在明确
搜索范围—最小干预权衡。因此 G4-B 只通过 completeness，未通过 locality efficiency；
不得从这 5 个已查看 seed 中选择排序并进入 live。

### 14.5 G4-B2b：exact fixing-deletion conflict filter（shadow label oracle）

当前 Python 环境没有 `highspy`，SciPy `milp` 结果也不暴露可验证 IIS/Farkas ray，因此未把
solver failure 冒充 LP 或整数冲突证书。替代实现从 full permission block 开始，逐项恢复
target/UAV/role fixing；仅当受限 exact MILP 仍可行时才删除该权限。最终每个保留权限 `g`
都有一个局部必要性反事实：在当时的超集以及其后的任意更小子集上移除 `g` 均不可行。这依赖
受限可行域单调性，但只是 one-deletion irreducible certificate，不是全局最小 IIS。

21 帧中，顺序删除均保持可行，permission block 中位数为 2 UAV、1 target、1 role；
15/21 不超过 3 UAV×3 targets，16/21 得到与 G4-A 相同的 minimum repair closure。相比
G4-B 的 4/21 小块，组合可行性反馈明显更有效，但需要 333 次 feasibility MILP，因此只适合
生成冲突标签/审计基线。二分批量删除保持相同质量却增加到 471 次调用，在当前 K=Q=6 关闭。

### 14.6 G4-B2c：global backbone 与 permission conflict-cut master

对 permission universe `U`，定义 `Phi(B)` 为受限 exact repair 是否可行。单调性给出顺序
无关 backbone

```text
C = {g in U : Phi(U\{g}) = 0}.
```

`g in C` 当且仅当所有可行 permission block 都包含 `g`。21 帧中 13 帧 backbone 非空，
大小中位数 2；没有 role permission 成为 backbone，说明具体 TX/RX 翻转存在组合替代性。
同一 episode 内核心目标较稳定：seed 103 的 5 帧均含 target 2，seed 483 的 5 帧均含
target 3。该现象仍只有开发集机制意义。

若 `B` 不可行，则 `H=U\B` 产生严格有效 cut `sum_{g in H}u_g>=1`。实现的小型 master
最小化 permission cardinality，并显式满足 `u_role,i<=u_uav,i`。必须区分：首次 exact-feasible
master 解只对 permission cardinality 全局最优，不自动等于 G4-A 的 realized closure/bit
最优。实测未缩减大补集 cut 在每帧 16-call 门内仅 1/21 收敛；20 帧触顶，共 321 次 master
oracle query。强制 exact backbone 后结果完全不变。因此 cut 有效性成立，但算法信息强度
不足，当前 master 关闭；不通过增加 call cap 掩盖失败。

### 14.7 G4-B2d：Core-Guided Capability Repair（shadow 机制成立、效率未过门）

令 `C` 是仍被冻结的 permission 集合，并用依赖闭包解释 `U\C`：冻结 `uav:i` 必须同时
冻结 `role:i`，从而始终满足 `u_role,i<=u_uav,i`。若 exact physical oracle 认证

```text
Phi(U\C) = 0,
```

则任何可行 repair `B` 都必须满足 `B intersect C != empty`，所以
`sum_{g in C}u_g>=1` 是严格有效 cut。顺序删除只在删除后仍不可行时缩小 `C`；最终每个
成员单独删除都会使最大合法开放块可行，称为 oracle-certified one-deletion irreducible
permission conflict，不称 IIS/Farkas。singleton core 正是 backbone。所有 core 的 hitting
set master 若第一次返回 exact-feasible 解，则它对 master 声明的 permission objective
全局最优：已有 core 给出全局下界，而该 feasible 解同时给出同值上界。

实现支持两种精确目标：等权 permission cardinality，以及字典序
`(N_UAV+N_target, N_role)`。后者用 `K+1` 精确标量化第一层，因 role 差最多为 `K`，不是
经验权重。可行性 oracle 仍完整使用 post-G2 通信后 residual power、25.1 mW sensing cap、
single-role、unique owner、pair/receiver capacity、local fusion 和 worst/bottom-3/steady
三地板；物理分数只曾用于查询顺序，不能生成 hard cut。

单调 antichain cache 仅做逻辑推断：已知 feasible `F subseteq B` 可认证 `B` feasible；已知
infeasible `I supseteq B` 可认证 `B` infeasible。其余块才进入 exact MILP。21 帧结果显示
core size 中位 3、最大 5，但等权和 support-lex 都只有 16/21 在 16 个 master candidate 内
收敛，exact 调用分别为 2824 和 2822；support-lex 的 minimum closure 命中为 16/21。
因此 core 表示正确且显著强于 raw cut，但逐核 exact shrinking 仍不具在线价值。当前不进入
G4-C/live，也不提高 cap；下一理论缺口是从物理 MILP 内部取得可审计的小冲突证书，或构造
可证明的专用解析 conflict oracle，而不是继续改变删除顺序。

### 14.8 G4-B2e：Objective-Layer Conflict Separation（部分成立，完整算法未过门）

support-lex 使用未扰动整数成本

```text
J(u) = (K+1)(N_UAV+N_target) + N_role.
```

`K+1` 保证与字典序严格等价；`1e-6` tie-break 只选代表点，不参与 objective face 定义。
给定已有 cuts 与子水平集 `F(tau)={u:J(u)<=tau}`，定义所有这些 master 解共同为零的权限

```text
Z(tau) = {g : u_g=0 for every u in F(tau)}.
```

若最大合法开放块 `U\Z(tau)` 被 conservative PWL MILP 证明不可行，则
`sum_{g in Z(tau)}u_g>=1` 有效，并删除整个 `J<=tau` 子水平集，所以新 master 下界严格大于
`tau`。反之，若 `U\Z(tau)` 可行，则任何 `C subseteq Z(tau)` 都不是该单调模型的 conflict；
此时一条 cut 无法消灭整个子水平集。`Z(tau)` 通过最多 18 个小 permission-master MILP
求得，不消耗 structure-power physical oracle；物理查询只用于认证 union-open block。

底层 feasibility oracle 改为三态：有 incumbent 时独立检查 bounds、全部线性约束和整数性，
通过才是 `CERT_FEASIBLE`；仅 solver `status=2` 是 `CERT_INFEASIBLE`；其余均为
`UNRESOLVED`，不得缓存为单调证书或产生 cut。由于检测约束仍是 conservative chord PWL，
当前 `CERT_INFEASIBLE` 只证明该保守 MILP 模型不可行，不能外推为原始连续检测物理不可行。

五个预声明 B2d 触顶帧上，当前最优面的 union-open block 全部 `status=2`，5/5 face cuts
严格抬升下界；进一步对 `tau` 做单调二分，19 次 physical query 找到最大不可行子水平，
单次 cut 将标量下界跨越 6–7。该结果解决了 target-free face 漏失且没有 target 人工加权。
但是从空 cuts 启动后，大量 objective face 的 union 已可行，需要 face splitting/bundle。
当前 greedy 8-support bundle 在五帧耗费 407 次 physical query 仍 0/5 收敛；单-support
no-good 只需 100 次但同样 0/5，并长期停在同一 objective layer。因此 B2e 尚未形成可用
端到端算法；保留 tri-state 与 sublevel theorem，关闭 greedy bundle，不跑全 21 帧。

### 14.9 R1：安全剪枝后的交互宽度审计（假设被否决）

对每个目标 `q`，task mode 定义为 `(owner, nonempty TX subset)`；只使用一个方向严格安全的
物理剪枝：若该 mode 在独占完整 sensing budget 时的乐观线性 Deflection ceiling 仍低于
`D_floor`，则它不可能单独满足该目标地板，可以删除。此处没有使用 dual score、经验阈值或
已知 witness。该 isolated-mode 剪枝只回答“单 mode 是否有能力”，并不证明保留下来的
modes 可联合实现。

R1 区分四种图语义：raw target primal 包含未 lifting 的 steady/bottom-k 全目标因子；
resource-target primal 由共享 UAV 端点形成 target clique；summary-lifted incidence 把 target
与 UAV summary 分成二部图；task-mode clique primal 对 mode 的 target/owner/TX scope 成团。
精确 treewidth 使用 elimination subset-DP，而不是用分量大小、最小度或近似 upper bound
代替。summary lifting 后较低的 target-side clique 不自动表示复杂度下降，因为 separator
frontier 还必须携带共享 power、role、capacity 和 QoS summary state。

21 个 K=Q=6 Region-II 帧中，四图 treewidth 固定为 `5/5/6/6`；resource-target 是 `K6`，
summary incidence 是 `K6,6`，task-mode 图含 51 条边并形成 12 节点单一巨分量。每个 UAV
端点的 target degree 都为 6。每帧理论 mode 上限为 900，安全剪枝后总数为
`630/793/835`（min/median/max），每目标为 `55/132.5/150`；overlap HHI 中位 `0.1673`。
有效宽度的中位/最大均为 6，未通过预声明的 `median<=3, max<=4` 门。

所以 `H_width` 在当前搜索宇宙被否决。seed 103 的 mode 中位为 662，seed 566 为 794，
但两者宽度均为 6；mode 数差异不能解释为 separator width 差异。当前没有不同 K/Q 的
同口径 Region-II trace，故不声称 `width=K` 是规模律，也不为凑齐曲线生成合成结论。
R2 mode-equivalence、R3 dual-set pruning 和 separator solver 停止，除非未来先出现可证明
保持联合约束的更强物理支配规则并重新通过 R1。

### 14.10 S0-A：resource-complete context-free safe screening

定义候选 incidence `e=(i,j,q)`，资源偏序逐 UAV 保留 sensing power、TX/RX role、owner、
receiver capacity 和通信负载，不允许把不同 UAV 的资源合并成 total cost。于是跨 UAV 替换
会增加替代 UAV 原本为零的资源分量；在没有 search-state implication 与 guaranteed residual
时，不可能满足 context-free `R(m')<=R(m)`。因此 gain 更大不是系统级 dominance。

对当前非负线性 Deflection 模型，若 `a_ijq=0`（严格等于零，不用 tolerance），将 `x_ijq`
与 flow 固定为零只会放松 power、pair、receiver 和 role implication 上界，且不改变任何
`D_q`，所以保持 conservative-PWL feasibility。若 reference structure 中 `x^0_ijq=0`，
该 fixing 还不会制造 edge toggle；任何含它的解删掉该边后 closure/prepare/power 均不增，
故至少保留一个三层 lex optimum。反之，若 `x^0_ijq=1`，feasibility theorem 仍成立，但
不能无条件声称 optimality-preserving。

21 帧共有 3780 个有向非对角 incidence，严格零增益 96 个，且本批 reference structure
恰好没有选择这些零边。因此 D-F/D-O 都筛除 2.54%，exact re-solve 分别保持 21/21 QoS
feasibility 和 21/21 `(closure, prepare bits, sensing power)`。但每帧正系数 incidence 在
UAV-target 投影上仍为完整 36 条边，treewidth 恒为 6。结论是证书正确、结构收益不足。

### 14.11 S0-D：致密系数的能量秩结构

固定目标 q，理想双基地路径律在允许 `i=j` 的数学 completion 下为

```text
A_q^ideal = c_q u_q u_q^T,   u_q(i)=1/R_iq^2,
```

所以精确 rank-1。实际架构禁止同一 UAV 同时 TX/RX，相当于对角 mask：
`A_q=A_q^ideal-diag(A_q^ideal)`；该矩阵一般具有满代数秩，不能宣称 exact rank-2。
但在当前 21 帧×6 targets 上，其奇异值能量高度集中：operational `r95` 为 rank-2/3/4 的
数量分别是 `109/14/3`，中位 stable rank 为 1.74。U2U-only 配置中 `chi_rep=1`，所以 report
阶段奇异值完全不变；hard DD mask 只增加少量尾秩。

该诊断不构成安全压缩定理。rank-1 相对 Frobenius 误差中位 0.65，低秩 SVD 近似还可能在
不同 entries 上正负交错，既不是 Deflection 下界也不是上界。任何后续 dense low-rank
capability 表示必须构造逐项或任务级 conservative envelope，并重新验证 worst/weak3/
steady 三地板；在此之前不修改 G4 oracle 或 live L0-L3。

当前模型还具有一个精确表示恒等式。令 `u_iq` 吸收 TX-target inverse-range factor，`v_jq`
吸收 RX-target factor、detector scale 与 receiver-only `chi_rep,j`，则 DD 前

```text
A_q^base = u_q v_q^T - diag(u_q .* v_q).
```

令 `S_DD,q` 为 hard DD gate 失活的非对角集合，并令 `E_DD,q` 只在该集合复制 base entry，
则 `A_q=A_q^base-E_DD,q`。这不是低秩近似：当前 126 个矩阵的 factor+exception 重建最大
逐元素相对误差为 `8.08e-16`。96 个 exceptions/3780 个非对角 entries 与 S0-A 严格零边对应。

该 exact diagonal-plus-rank-one-plus-sparse 表示保留每个 `a_ijq`，不改变 Deflection 线性式
或三个任务 QoS；潜在收益是 coefficient storage/message 从 dense `K^2Q` 降为两个 factor
vectors 加 sparse DD exceptions。它尚未压缩 edge binaries、single-role、owner 或 capacity
约束，所以不能据此重开 treewidth，也不能声称 solver complexity 同阶下降。进入通信实现
前还必须证明 factor quantization/AoI 误差形成保守上下 envelope。

### 14.12 M0–M3：certificate-carrying factor message（shadow）

M0 使用 §14.11 的 exact factor+DD-exception identity。M1 的对象不是当前 live packet，而是
“完整 capability tensor”这一语义任务的两种 shadow wire records。设 `N=QK(K-1)`，当前
冻结的 aggregated layout 为：dense=`25-bit header + 32-bit global linear scale + NB_A`；
factor=`27-bit header + 64-bit outward log endpoints + 2KQB + B_DD`，其中

```text
B_DD = min(ceil(log2(N+1)) + s*ceil(log2(N)), N).
```

header 包含 protocol、epoch、bit-depth 和 codec selector。共同 scale 目前由集中式 frame
求得，因此只是 semantic-equivalence baseline，不是已部署的分布式 protocol。

对正 factors 的 log 值使用 closed uniform bins；range endpoints 先向外舍入到 float32。
若 TX/RX log intervals 分别为 `[xL,xU]`、`[yL,yU]`，exact-active edge 的系数区间为
`[exp(xL+yL),exp(xU+yU)]`；对角和 exact DD-inactive entries 都为 `[0,0]`。这给出 M2-0
静态逐项 containment。3/4/6/8/10/12/14/16 bit 在 21 帧均为 0 violations。该证明不覆盖
packet loss 或 AoI；尤其现有 `dd_gate_active_set_certificate` 是同帧、velocity-held 的
position trust-region 结论，不能跨帧套用。跨帧 DD 必须 unresolved/refresh，除非另有证书。

固定 owner 与非负 power 时，逐项区间由单调性给出 `DL<=D<=DU`，再经单调 `P_D(D)` 和
三个分量单调聚合器传播。仅当 lower 三指标全过地板才 `CERT_FEASIBLE`；upper 任一仍不过
才 `CERT_INFEASIBLE`；其余 `UNRESOLVED`。105 个固定方案/bit-depth 的审计均无错误证书。

nominal G4-A 的第三层目标最小化 sensing power，使 worst 恰落在 0.61；所以任何非零向下
interval 都不能认证 nominal witness，即使 16 bit 也为 0/21。这不是 containment failure。
以更严格设计 floors 重解 exact lex MILP 后：

| headroom | 8 bit | 10 bit | 12 bit | median extra power |
|---|---:|---:|---:|---:|
| +0.0025 | 0/21 | 2/21 | 21/21 | 0.168 mW |
| +0.005 | 0/21 | 19/21 | 21/21 | 0.328 mW |
| +0.01 | 2/21 | 21/21 | 21/21 | 0.663 mW |

这说明 certificate precision 必须与执行裕量联合设计。同 certification rate 的 dense 对照
由 §14.13 完成；L1/L2 decision preservation、transport deadline、FBL 与 live replacement
仍未验证。

当前活动配置中 `comm_power_fraction_max=0.5`，故 `P_comm<=0.5 W`；配合 1 W RF cap 与
25.1 mW sensing PA cap，有 `1-P_comm>=0.5 W>0.0251 W`。因此本配置的 cap-aware sensing
budget 恒由 sensing PA cap 决定。当帧通信—感知耦合不经过 power competition，而主要经过
serialization、deadline/drop、AoI、battery energy 和后续闭环。一般模型仍保留 RF coupling，
但不能把它冒充当前实验中的活跃机制。

### 14.13 M4-A/B：task-equivalent rate audit（开发机制 PASS）

外积 gauge 不唯一：`(u,v)` 与 `(cu,v/c)` 生成同一 `A`。若将 detector scale 任意全部放入
RX factor，同时对 TX/RX 使用共同量化 range，会为无物理意义的 factor normalization 付 bit。
当前采用确定性 log-center balance：令 TX/RX log interval centers 为 `m_u,m_v`，取
`log c=(m_v-m_u)/2`。这使两中心重合，并最小化两区间在相反平移下的公共跨度；不改变任何
乘积、DD support 或消息字段数。balance 后 8-bit 乘积 log interval 中位宽由 0.412 降为
0.0487，containment 仍逐项成立。

M4 的公平 wire record 统一为 common 27-bit header、epoch/codec 和同一 exact DD support：

```text
dense-linear:  header + 32-bit scale + B_DD + N_active * B_a
dense-log:     header + 64-bit log endpoints + B_DD + N_active * B_a
factor-log:    header + 64-bit log endpoints + B_DD + KQ(B_u+B_v)
```

当前 factor 对所有 TX/RX 使用同一 `B`，所以最后一项为 `2KQB`。三种方法都允许
3/4/6/8/10/12/14/16 bit coarse-to-fine，逐帧选择实际 payload 最小的首个
`CERT_FEASIBLE`。dense methods 不发送 DD-zero coefficients，避免人为偏袒 factor。

21 帧三档 exact headroom witnesses 上，全部方法 coefficient containment violations=0、
false CERT_FEASIBLE=0。dense-linear 在 16 bit 内只覆盖 11/14/17 帧，不构成共同 21/21
operating point；dense-log 与 factor-log 都为 21/21。adaptive 结果：

| headroom | factor min-bit distribution | factor median | dense-log distribution | dense median |
|---:|---|---:|---|---:|
| +0.0025 | 10:2, 12:19 | 995 | 10:9, 12:12 | 2235 |
| +0.005 | 10:19, 12:2 | 851 | 8:1, 10:19, 12:1 | 1891 |
| +0.01 | 8:2, 10:19 | 851 | 8:6, 10:15 | 1891 |

factor/best-dense 中位 bit ratio 为 0.445/0.450/0.450。忽略公共 overhead 时，factor
break-even 条件是 `2b_f<(K-1)b_d`；K=6 时允许 factor 每 scalar 使用 dense 的 2.5 倍 bit。
实测两者首次认证 bit 同阶，而 factor scalar 数为 72、dense active 数约 176，故生成结构
优势超过乘积误差累积成本。

该 Gate 仍使用 centralized dynamic scale，统计单位只有 5 个相关 development episodes。
它只完成 M4-A dense baselines 与 M4-B fixed-plan QoS-rate frontier。M4-D 必须在 fixed
candidate set 上按 lex stages 认证最优等价集，而不是要求 solver basis/power vector 相同。

### 14.14 M4-C-light：interval capability route（开发机制 PASS）

对固定 owner/edge structure，从三维 coefficient interval 抽取 `K*Q` 固定结构 gain，定义

```text
gamma_L = gamma_PWL(A^+, b),
gamma_U = gamma_PWL(A^-, b).
```

由于 gain 增大只扩张可行功率集合，严格有
`gamma_L <= gamma_PWL(A,b) <= gamma_U`。路由语义冻结为：

```text
gamma_U <= 1  -> CERT_L1_FEASIBLE
gamma_L >  1  -> CERT_L1_FALLBACK
otherwise     -> UNRESOLVED
```

任一 LP 返回缺失、非有限值或违反单调夹逼都不签发证书。实现时同时修正一个定义冲突：旧
PWL gauge 在物理预算 `b` 达不到 worst floor 时提前返回 `None`，但 gauge 的优化变量本来允许
`gamma>1` 来度量 extra budget；现已删除该提前退出，使 fallback margin 有定义。该改动只涉及
PWL LP；旧非线性诊断函数未在本轮扩展。

审计以 21 个 G4-A 固定结构为基础，通过只减小 residual sensing budget，按 gauge 的正齐次性
构造六个边界：`0.90/0.97/0.99/1.01/1.03/1.10`。共 126 个相关场景，结果为：

| true gamma | 首次认证 bit 分布 | median semantic payload |
|---:|---|---:|
| 0.90 | 6:13, 8:8 | 579 bit |
| 0.97 | 8:14, 10:7 | 723 bit |
| 0.99 | 8:1, 10:19, 12:1 | 867 bit |
| 1.01 | 8:2, 10:18, 12:1 | 851 bit |
| 1.03 | 8:17, 10:4 | 723 bit |
| 1.10 | 6:10, 8:11 | 675 bit |

126/126 在最高 16 bit 前得到正确 route；所有 coefficient containment、gauge bracket 与 false
route 计数均为 0。`|1-gamma*|` 对首次认证 bit 的描述性 Spearman 为 `-0.824`，支持近边界
需要更细信息这一决策率规律。独立统计单位仍只有 5 个 episode，因此不使用场景级 p 值。

这是 centralized dynamic-scale、counterfactual-budget semantic shadow，不要求 power vector、
dual 或 LP basis 相同，也未证明 distributed scale agreement、air bits、deadline/AoI 或闭环性能。
M4-D 完成前仍不称完整 decision-preserving coordination protocol。

### 14.15 M4-D-light provenance Gate（INVALID INPUT，2026-08-21）

M4-D 按冻结候选集口径实现三阶段 L2 比较：前两级
`(dependency closure, prepare bits)` 精确比较，第三级对固定结构求满足三任务地板的最小总
sensing power，并由 `A^-<=A<=A^+` 单调传播
`P_min(A^+)<=P_min(A)<=P_min(A^-)`。full-precision 等价最优集使用 `1e-8 W` 数值容差；只有
全部 nonoptimal candidates 被区间逐级排除才允许停止 refinement。

候选源限定为可严格重放的 G4-A 与 G4-B2b sequential/bisect permission filter，不加入随机或
one-toggle 人工候选；依赖缺失 `physical_pd` 的 G4-B ladders 明确排除。实际 21 帧上三个来源
逐帧全部生成同一结构：candidate count 恒为 1，capability-dependent frame 为 0。旧 artifact
报告 G4-A nonzero closure `20/21`、median `3`；当前代码/物理口径重放为 `0/21`、median `0`。

进一步对旧 G3-A 的相同 5 个 exposed seeds、25 帧重放 fixed-structure minimum-intervention：
25/25 closure=0，当前 L2 frame=0。由于 trace 没有 sensing-power tensor 与 environment-level
`physical_pd`，当前 deployed-plan performance 无法重建。故：

```text
M4-D interval safety machinery: PASS
candidate-pool provenance:       FAIL
L2 decision-rate Gate:           FAIL / NOT EVALUABLE
communication-value Gate:        FAIL / NOT EVALUABLE
```

singleton 的 `B*=0` 不进入科学结论。M4-A/B/C 仍可作为这些固定帧上的静态表示、fixed-plan 与
受控 L1 boundary 机制证据，但不能再称为当前 Region-II L2 样本上的完整递进链。下一步不是
扩展 M4 certificate，而是先生成带 current sensing power、environment `P_D`、物理/config/code
hash 的新鲜 exposed development trace，再重新执行 deployed→L1→L2→L3 四层归因；blind bank
与 G2-1A 保持隔离。

### 14.16 P0 当前算法层 provenance 契约（2026-08-21）

当前系统不再仅给数据集或物理模型做 provenance，**算法层标签本身也必须有 provenance**。
`teacher_trace.npz` 升级为 schema 3；一次合法 P1 分层输入必须同时包含：

1. 同一帧的 UAV/目标位置与速度；
2. 实际执行的 selected edges、role、receiver owner；
3. 实际 `p^sense_{kq}`、`P^comm_k` 及由联合 RF cap 得到的 residual sensing budget；
4. 同帧 `a_{ijq}`、`g_DD`、`chi_rep` 和环境级 `P_D,q`；
5. frame mean/weak3/worst；
6. code/config/physics/checkpoint 的内容哈希。

物理哈希不是一句版本号，而是规范 JSON 的 SHA-256；其中显式记录无量纲 matched-filter
energy-ratio Deflection、`c_det`、`n_cpi`、`L_eff=1`（无额外 look）、`M,N,T_sym` 与
`B=M/T_sym` 约定、噪声功率、联合 RF/感知 PA cap、`P_FA`、Gaussian-deflection detector、
DD `g_min`、report-link reliability 和 fusion mode。哈希载荷与哈希不一致时 fail-closed。

P1 区域语义冻结为：D=部署方案已满足三地板；L1=部署失败但固定离散结构的
`gamma_fixed<=1`；L2=仅当 L1 失败且 joint structure-power 给出可行 witness；III=同几何
componentwise relaxed ceiling 仍失败；U=其余未决。任何 heuristic miss 都不能证明 III。
因此 M4-D 当前总状态为 `BLOCKED_BY_LAYER_PROVENANCE`；只有 fresh natural trace 通过 P0 且
`N_L2>0` 才可恢复。旧 trace 缺任一字段时不得用 `local_pd` 或重构功率补洞。

### 14.17 P1 fresh natural layer re-triage（开发机制证据）

上述恢复条件现已满足。用当前 6×6 配置、同一 warm-start 和 5 个已公开开发 seed
`291/566/99/103/483` 重新生成 750 帧；没有使用 blind bank。P0 schema-3 Gate PASS。
P1 预先采用均匀系统抽样 `frame mod 15=0`，不看 difficulty 或结果，共 50 帧：

| D | L1 | L2 | III | U |
|---:|---:|---:|---:|---:|
| 17 (34%) | 1 (2%) | 32 (64%) | 0 | 0 |

数理判定顺序为：环境三地板直接验证 D；固定结构 conservative chord-PWL
`gamma_fixed<=1` 验证 L1；仅当其失败才运行 role/owner/edge/power 联合 MILP，并以真实
`P_D` 和 per-UAV budget 二次验证 L2；只有 componentwise relaxed ceiling 失败才记 III。
32 个 L2 witness 的最小 exact `(worst,bottom-3,average)` 为
`(0.610000,0.710002,0.810055)`，最大功率违反 `1.53e-16 W`；L2 帧最小
`gamma_fixed=1.0181`，最大 `gamma_joint,witness=0.9931`。同帧 deployed replay 从
`selected*a*p` 重算 `P_D` 的最大误差 `2.22e-16`。

因此阻塞只保留在旧 M4-D artifact；fresh M4-D 可以在这 32 帧上继续。由于只有 5 个独立
episode，比例是开发机制描述，不是总体发生率、显著性或部署性能声明。

### 14.18 M4-D0/D1 候选定位边界（D2 未启动）

候选生成与 exact referee 已拆成两个 artifact 阶段。D0 函数只接受
`(S_deployed,A,b,gamma_fixed,pi,eta,hard constraints)`；禁止 witness、closure、MILP variable、
feasibility feedback、B2b conflict 和 P1 label。候选载荷在 referee 前冻结并哈希；D1 重新求
exact lex objective，仅判断冻结候选是否与最优等价集相交。等价性按
`(dependency closure,prepare bits,minimum sensing power)`，不是单一 solver structure。

若 `mu_i<=0` 是 capability LP budget-row marginal，定义 `eta_i=-mu_i>=0`；gamma KKT
stationarity 为 `1-sum_i eta_i b_i=0`。target equality marginal 取正价格约定后，locator 使用
`pi_q a_ijq-eta_i`，没有把 max-min proxy 冒充 full-task dual。

D0-v2 加入 oracle-free atomic local grammar 后，32 个 genuine L2 帧中 Pool A coverage
`0/32`、Pool B coverage `9/32`，余下 `23/32` 是 move-grammar gap。D2 eligible=0，触发
`STOP_AT_CANDIDATE_LOCALIZATION`。该结论只否定当前 locator/grammar 的充分性，不否定 L2
存在，也不评价 factor interval 或 air-interface rate。
