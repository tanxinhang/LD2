# UAV-ISAC 最新系统总报告：系统模型、理论与算法

更新时间：2026-09-11
系统身份：K16/Q16、U2U-only、严格分布式、解析控制基线
正式状态：`algorithm_research`；当前提交尚无新的 blind-100 正式证据

冻结语义清单为 `config/system_manifest.yaml`，可执行正式 profile 为
`config/exp_strict_distributed_k16q16.yaml`。

## 1. 核心论点与证据边界

本系统研究无地面融合链路条件下，多架无人机如何仅依靠本地目标 belief 与实际送达的 U2U
消息，联合完成双基地感知结构选择、感知功率分配和安全移动。当前实现表明：严格分布式解析
基线可以把信息来源、通信代价、功率约束和运动安全统一放入一个可审计闭环；本轮十种子
smoke 验证了新接入的硬安全与资源门禁，但尚不足以证明当前版本的统计性能。

结论边界如下：

- 系统生成几何级和充分统计量级的 ISAC 结果，不生成原始 IQ 样本或完整 OTFS 波形。
- “分布式”限定控制决策的可见信息；仿真器仍集中推进真值并汇总指标。
- 正式在线控制器是解析式基线，不是训练得到的神经网络策略。
- Markov、fixed-lag、predictive-GNN 和 temporal optimization 均为影子研究支线，没有在线动作权限。
- 当前数据只能支持实现正确性和机制筛查，不能替代 clean-commit blind-100 正式结论。

## 2. 术语与符号

| 符号/术语 | 定义 |
|---|---|
| `K` / `Q` | UAV 数量 / 目标数量；当前均为 16 |
| `i,j,q,t` | 发射 UAV、接收 UAV、目标和控制帧索引 |
| target belief | 节点对目标状态的局部概率估计，而非仿真真值 |
| 双基地边 `(i,j,q)` | UAV `i` 发射、UAV `j` 接收目标 `q` 回波的有向感知关系，要求 `i≠j` |
| owner receiver | 对一个目标负责形成最终本地检测证据的唯一接收 UAV |
| `a_{ijq}` | 双基地边单位感知功率的有效 Deflection 系数 |
| `p_{iq}` | 发射 UAV `i` 分配给目标 `q` 的感知功率 |
| `D_q` | 目标 `q` 的累积 Deflection，检测统计量的非中心参数 |
| `P_D,q` / `P_FA` | 检测概率 / 固定虚警概率 |
| `u_i,t` | UAV `i` 在一帧内的二维位移 |
| public view | 由本地 belief 和已送达消息重建的公共决策视图 |
| formal evidence | 与 clean Git commit、配置、种子银行和运行环境完整绑定的证据 |
| diagnostic evidence | 用于接线、排错或机制筛查，但不满足正式 provenance 的结果 |

## 3. 任务定义

### 3.1 输入、输出和目标

每个控制帧中，节点输入包括自身位置、速度、电量、角色、本地目标 belief、消息 AoI 和实际
送达的邻居消息。控制器输出：

1. Tx/Rx/idle 角色；
2. 每个目标的唯一 owner 和双基地边集合；
3. 通信消息、量化精度与协议字段；
4. 每个 UAV 的目标级感知功率；
5. 经安全投影后的二维位移。

系统希望提高所有目标中的弱目标检测性能，同时满足通信、功率、时延、电量和飞行安全
约束。当前主线采用词典序思路：先保证信息和结构有效，再最大化最弱目标 Deflection，最后
才允许移动改善几何；任何必要信息过期或证书不完整时保持不动或采用已验证 incumbent。

### 3.2 冻结场景参数

| 类别 | 当前 K16/Q16 值 |
|---|---|
| 区域与高度 | `1130 m × 1130 m`；固定高度 `20 m` |
| 控制时域 | `T=150` 帧；`dt=0.1 s`；总时长 15 s |
| UAV | `v_max=25 m/s`；单帧最大位移 `2.5 m`；安全距离 `20 m` |
| 电量 | 初始 `50,000 J`；悬停功率 `80 W`；速度系数 `0.05` |
| 目标 | CV 模型；速度范围 `0–5 m/s`；白加速度标准差 `0.5 m/s²`；RCS `1 m²` |
| OTFS | `fc=28 GHz`，`B=1 MHz`，`M=64`，`N=16`，`Δf=15.625 kHz` |
| 感知时钟 | `T_sym=64 μs`；`n_cpi=1`；每控制动作实际执行 `1.024 ms` 感知 |
| RF 功率 | 每 UAV 联合通信+感知上限 `1 W`；感知 PA 上限 `25.1 mW` |
| 通信 | 一跳广播；`500 kHz`；`0/4 bit/维`；处理延迟 `0.2 ms`；deadline `5 ms` |
| 检测 | `P_FA=10^-3`；real Gaussian shift；连续 DD 增益 |
| 评估 | 独立 test split 的固定 100-seed bank；tail window 50 帧 |

## 4. 动力学与 belief 模型

### 4.1 UAV 运动学

UAV 高度固定，只控制平面位置：

```text
x_i,t+1 = Reflect(x_i,t + u_i,t)
||u_i,t||_2 <= v_max * dt = 2.5 m
```

`Reflect` 对矩形边界执行镜像反射，并同步翻转对应速度分量。实际存储速度由执行位移计算：

```text
v_i,t+1 = (x_i,t+1 - x_i,t) / dt
```

因此公开状态描述的是实际执行轨迹，而不是边界反射前的命令速度。

### 4.2 目标 CV Markov 模型

每个目标的二维状态为 `s_q=[x,y,v_x,v_y]^T`：

```text
s_q,t+1 = F s_q,t + w_q,t
w_q,t ~ N(0,Q_cv)

F = [[1,0,dt,0],
     [0,1,0,dt],
     [0,0,1, 0],
     [0,0,0, 1]]
```

白加速度噪声协方差按轴独立：

```text
Q_cv = sigma_a^2 *
[[dt^4/4, 0,       dt^3/2, 0      ],
 [0,       dt^4/4, 0,       dt^3/2],
 [dt^3/2, 0,       dt^2,   0      ],
 [0,       dt^3/2, 0,       dt^2  ]]
```

上式第一行中的 `dt^3/2` 为位置—速度协方差项。目标越界时使用与仿真器一致的轴对称反射。

### 4.3 局部 belief 递推

节点维护高斯 belief `b_q,t=N(m_q,t,P_q,t)`。预测满足 Chapman–Kolmogorov 递推：

```text
m^-_q,t = F m_q,t-1
P^-_q,t = F P_q,t-1 F^T + Q_cv
```

获得测量 `y_q,t` 时，使用标准 Kalman 更新：

```text
r_t = y_t - H m^-_t
S_t = H P^-_t H^T + R_t
K_t = P^-_t H^T S_t^-1
m_t = m^-_t + K_t r_t
P_t = (I-K_tH)P^-_t(I-K_tH)^T + K_tR_tK_t^T
```

最后一行采用 Joseph 形式保持数值半正定。无测量时直接保留预测 belief。owner posterior 在下一
控制载波广播；不同来源间可能相关，因此使用 covariance intersection，而不是把重复信息当作
独立证据相加。正式分布式路径禁止读取目标真值；belief 缺失或超龄时 fail closed。

## 5. 通信与信息约束模型

### 5.1 物理传输

活跃发送者正交分享带宽。若一帧有 `n_a` 个活跃发送者，则每个发送者的有效带宽为
`B_k=B_comm/n_a`。距离 `d_ij` 下的自由空间增益和接收 SNR 为：

```text
g_ij = (lambda / (4*pi*d_ij))^2
SNR_ij = P_comm * G_tx * G_rx * g_ij / (kT * B_k * NF)
R_ij = B_k * log2(1 + SNR_ij)
```

数据包位数为 `L=h+d*b`，其中 `h` 为显式头部，`d` 为有效维数，`b∈{0,4}`；`b=0` 表示
静默。序列化与总时延为：

```text
t_ser = L / R_ij
t_link = t_ser + 0.2 ms
```

同一发送者的一次广播只有一个物理发射时钟。令服务包络在接收者 `j` 上所需的序列化时间为
`t_ser,ij^svc`，则本次发送统一占空时间为：

```text
T_tx,i = max_j min(t_ser,ij^svc, T_deadline)
E_comm,i = P_comm,i * T_tx,i
```

系统绝不对接收者分别累加 airtime 或能量；`per_sender_airtime_s` 是这一共同时间的直接审计
字段。接收结果仍然依赖接收者：只有同时满足 SNR 门槛、`t_link<=5 ms`、服务包络和可靠性
门槛的消息才进入该接收者 inbox。换言之，发送事件是一个，解码事件有多个。当前正式配置关闭有限
码长、阴影衰落和突发丢包扩展，因此结果只代表冻结的确定性链路身份。

### 5.2 协议语义

`target_tokens` 不只是自由神经消息。严格解析栈把端点状态、目标局部摘要、结构协商字段、
composable certificate 和 owner posterior 显式装入物理数据包并收费。协议关键设置为：

- 每维端点状态 8 bit；每节点只共享 top-1 目标摘要；
- 一轮 hyperedge consensus；结构至少覆盖 100% 目标；
- assignment 保持 5 帧，降低结构抖动；
- composable certificate 每目标 16 bit，frame 字段 32 bit，最大 AoI 5 帧；
- owner posterior 均值 12 bit、协方差 8 bit、AoI 8 bit，最大年龄 5 帧；
- 控制载波周期 3 帧；非载波帧复用仍有效且未超龄的本地缓存。

没有“免费邻居信息”旁路。若 packet 未送达、公共视图不完整或缓存超龄，相关结构、功率或移动
动作必须回退到可证明安全的状态。

## 6. 感知与检测理论

### 6.1 双基地几何

一条边 `(i,j,q)` 对应发射机 `i`、接收机 `j` 和目标 `q`。双基地时延与 Doppler 由两段路径
共同决定。实现中的路径幅度严格定义为传播—散射项：

```text
|alpha_ijq|^2 = lambda^2 * sigma_rcs
                  / ((4*pi)^3 * R_iq^2 * R_jq^2)
```

它只包含两段距离衰减、波长与 RCS，约定 `G_tx=G_rx=1`，不包含阵列增益。阵列乘积
`G_tx*G_rx` 只在 Deflection 公式中外乘一次，因此当前实现没有重复计入增益。`i=j` 的对角边
不进入严格双基地结构；以后若改用已含方向图的复信道系数，必须同时移除外部增益因子并做量纲
回归，不能两处同时保留。

### 6.2 原始与有效 Deflection

在冻结的 real-equivalent matched-filter 约定下，原始 Deflection 为：

```text
d_raw,ijq = c_det * P_iq * |alpha_ijq|^2 * G_tx*G_rx * (N*T_sym*n_cpi) / N0
N0 = P_noise / B,    P_noise = kT * B * 10^(NF/10)
```

由于 `B=M/T_sym`，它也等价于按实际执行 OTFS look 累积的预处理 SNR。这里 `n_cpi=1`，不能用
未执行的“虚拟相干帧”放大结果。

一般模型的有效系数进一步乘以可选报告链路可靠性和连续 delay–Doppler 增益：

```text
d_eff,ijq = chi_rep,j * d_raw,ijq * I_support(tau,nu) * |A(tau,nu)|^2
```

`I_support` 在无歧义 delay/Doppler 支持域外为 0；支持域内使用连续 `|A|²`，不再使用历史
二值 `g_dd>=g_min` 开关。当前 U2U-only 严格配置令 legacy UAV→ground report link 关闭，因而
该项取 `chi_rep=1`；U2U 消息是否可见由第 5 节的逐链路 delivery 单独决定。固定几何、角色、
结构和 CSI 后，Deflection 对感知功率线性：

```text
d_eff,ijq(P_iq) = a_ijq * P_iq
```

这条条件线性是后续 LP 正确性的关键；若未来引入功率相关 CSI、感知互扰或非线性 RF 耦合，
必须重建优化模型，不能继续把 `a_ijq` 当常数。

### 6.3 Delay–Doppler 分辨率与模型边界

冻结参数 `M=64, Delta_f=15.625 kHz, N=16, T_sym=64 us` 给出名义栅格：

```text
Delta_tau = 1/(M*Delta_f) = 1 us
Delta_nu  = 1/(N*T_sym) = 976.5625 Hz
```

`c*Delta_tau=300 m` 是双基地总路径差尺度；若按单基地往返距离解释，对应约 `150 m` 距离分辨率。
在 28 GHz 下，单基地径向速度尺度约为 `lambda*Delta_nu/2=5.23 m/s`。连续 ambiguity gain
消除了栅格边界的人为跳变，但不等于解决多目标不可分辨：当两个目标落入同一主瓣或证据相关时，
简单相加 Deflection 的条件独立假设可能失效。因此几何级主线只可称为“充分统计量仿真”；波形
外推前必须通过离线 DD 可分辨性、相关系数和 ROC 校准门禁。

### 6.4 多边证据融合和检测概率

每个目标只有一个 owner receiver。同一 owner 下，经白化且条件独立的边证据可加：

```text
D_q = sum_i a_i,owner(q),q * p_iq
```

冻结检测器为：

```text
H0: Z_q ~ N(0,1)
H1: Z_q ~ N(sqrt(D_q),1)
gamma = Q^-1(P_FA)
P_D,q = Q(gamma - sqrt(D_q))
```

反解达到给定检测概率 `p<1` 所需的最小 Deflection：

```text
D_min(p) = max(Q^-1(P_FA) - Q^-1(p), 0)^2
```

该映射单调，但效用 `-log(1-P_D)` 对 `D` 并非全局凹函数。因此旧 P0 greedy 不能声称
`(1-1/e)` 子模近似保证；当前文档将其严格标记为启发式，只把固定结构功率 LP 称为精确解。

### 6.5 实验性相关校准下界

`exp_strict_distributed_k16q16_correlation_calibrated.yaml` 不再把同一目标的多条选中边默认视为
独立证据。令归一化匹配滤波统计的均值为 `mu_q`、协方差为 `R_q`，联合 Deflection 为：

```text
D_joint,q = mu_q^H R_q^-1 mu_q
```

对正定 `R_q`，Rayleigh 商给出：

```text
D_joint,q >= ||mu_q||_2^2 / lambda_max(R_q)
          = sum_e a_eq p_eq / c_q,    c_q=lambda_max(R_q)>=1
```

实现用有限 `M×N` delay–Doppler steering atom 的归一化 Gram 矩阵构造 `R_q`。二维 atom 的
内积等于 delay 和 Doppler 两个有限 Dirichlet 内积之积，因此矩阵按构造为 Hermitian PSD；
完全重合的 `n` 个模板给出 `c_q=n`，相差整数 DD bin 的模板给出 `c_q=1`。奇异极限按加入任意
小独立噪声后取极限理解。该下界把每个目标的单位功率系数统一替换为 `a_eq/c_q`，所以固定结构
下仍是线性 LP；功率求解和最终检测使用同一个 `c_q`，不会出现优化证书与执行记分不一致。

这一模式目前只覆盖 selected-only receiver evidence。passive multireceiver 会改变协方差维度，
配置组合被显式拒绝，不能把 selected-edge 的因子冒充其证书。该 profile 仍是诊断候选，不改变
冻结基线；在 waveform/ROC 校准和 paired blind gate 通过前，不声称它是真实接收机的精确相关矩阵。

### 6.6 收缩后的相关软证据科学内核

当前研究主线已收缩为 `OTFS local evidence -> conditional-information selection -> soft fusion`。
对共同协方差的高斯软统计，直接使用

```text
D(S)=delta_S^T Sigma_S^-1 delta_S,
w_S=Sigma_S^-1 delta_S.
```

候选 `j` 的增量由 Schur complement 精确给出；当 `Sigma_jS=0` 时严格退化为 local Deflection，
从理论上解释低相关场景与 quality Top-K 的小差距。实现位于
`physical/correlated_soft_evidence.py`，目前只用于机制审计，不替换在线检测器。通信 bits、deadline
和 reliability 是约束/排序输入；atomic、provenance 与 replay 降级为 assurance shell。

空口 evidence 保持单跳 source-local：同一 generation 内不得把已融合 evidence 再包装成新源。
详细假设、baseline 和可证伪门禁见 `docs/SCIENTIFIC_CORE_CORRELATED_EVIDENCE.md`。

### 6.7 最小物理—统计闭环

新增离线 ideal-cyclic OTFS block，将 DD pilot 经 modulation、fractional delay/Doppler multipath、
common/local clutter、AWGN、demodulation 和 coherent DD matching 生成 receiver-local `z`。独立
split 上再估计 `mu_0,mu_1,Sigma_0,Sigma_1`，验证 fixed-P_FA ROC、covariance 异质性/稳定性和
`D(S)` 对 held-out `P_D` 的 subset 排序。equal-covariance 失败时只切换到 H0-covariance 线性
fallback，不宣称这是异方差检测的最优解。

该闭环不含 CP/pulse shaping、同步与 RF 缺陷，也假设 coherent phase；目前只属于离线
falsification。它没有接入在线控制器或 packet path，不能外推为真实 OTFS 或通信 Pareto 结果。

未知相位压力路径使用 matched-output energy，而不把随机相位强行塞入 coherent Gaussian 模型。
其 H0/H1 covariance 明显不同，故选择 `Sigma_0` fallback，并在独立 trace 上验证 subset 排序。
当前 waveform 的 2x2 消融中 aware/unaware 都选择 `(2,3)`，selection gain 为 0；因此物理相关
存在，但主选择机制尚未激活，不能把合成 rho-grid 的优势升级为系统结论。

## 7. 联合功率约束与精确 max–min LP

### 7.1 可用感知预算

每个 UAV 的通信和感知共享 `1 W`：

```text
0 <= P_comm,i + sum_q p_iq <= P_isac_total = 1 W
0 <= sum_q p_iq <= P_sense_max = 0.0251 W
```

求解器接收已经扣除通信占用并截断到 PA 上限的 `b_i`。任何执行结果都重新检查
`P_comm,i+sum_q p_iq`，允许的数值违反不超过 `10^-9 W`。

### 7.2 固定结构 LP

唯一 owner 结构冻结后，把同一 `(i,q)` 的有效边系数累加为 `g_iq`：

```text
maximize    eta
subject to  sum_i g_iq p_iq >= eta,               for every q
            sum_q p_iq <= b_i,                     for every i
            p_iq >= 0
```

相关校准 profile 使用 `g_iq/c_q` 替换 `g_iq`；`c_q` 在结构冻结后与功率无关，因此不改变
上述问题的线性、凸性或 HiGHS primal/dual 证书语义。

若给定 QoS reserve `r_q`，先附加 `sum_i g_iq p_iq>=r_q`，再最大化最弱目标。这避免只推高
最差目标时暗中破坏已经达到的任务门槛。实现使用 HiGHS dual simplex，并对极小/极大物理
系数进行尺度保护；解后检查非负性、预算和 reserve 可达性。

### 7.2.1 相关感知的精确局部交换研究内核

`coordination/correlation_aware_exchange.py` 实现了当前结构周围的一步 N5/N6 交换门禁。对每个
候选结构 `E'`，先重算其 Gram 因子和校准增益 `g'(E')`，再用 incumbent 精确 LP 的目标价格
计算候选的对偶机会值，并以 incumbent 功率重放的一阶变化作为次级排序键：

```text
S(E') = U_lambda*(E') - eta*
r(E') = lambda*^T [D(E',p*) - D(E,p*)]
```

该分数不构成接受证书。任意 simplex 价格给出的

```text
U_lambda(E') = sum_i b_i max_q lambda_q g'_iq
```

是候选精确 max-min 值的弱对偶上界；仅当 `U_lambda(E')` 不可能超过 incumbent 时才安全剪枝。
其余候选按分数选 Top-M，并逐个执行精确 fixed-structure LP。no-op 始终隐含在候选集中，只有
精确目标严格改善超过容差的交换才能返回，因此该研究内核在自身模型内具有单步不降性质。

目前它尚未获得在线执行权。下一小节的证书层已经闭合“候选是什么意思、各节点是否重建出同一
候选、空口是否足以完成原子提交”，但尚未把该研究内核接入主线环境状态机。因此它仍只允许用于
小规模结构最优性和 Top-M recall 审计，不能把集中式研究内核称为在线分布式算法。

### 7.2.2 相关候选的 reconstruct-then-hash 原子证书

`coordination/correlation_candidate_commit.py` 定义了候选的完整规范记录。SHA-256 的域分离输入
同时覆盖 generation、每个依赖的版本向量、完整结构/角色/owner、OTFS numerology、Gram 相关
因子、校准增益、UAV 感知预算、QoS reserve、精确功率解、逐目标 Deflection、最差目标值、LP
目标对偶价格与对偶上界。整数使用固定大端宽度，浮点使用规范化的 IEEE-754 binary64，布尔结构
按确定 bit order 打包；不使用 `repr` 或 JSON 浮点文本。

每个参与节点必须从已经物理送达的依赖包独立调用同一重建过程，再执行以下闭合检查：

```text
D_q = sum_i g_iq p_iq
eta = min_q D_q
sum_q p_iq <= b_i
lambda >= 0, sum_q lambda_q = 1
U(lambda) = sum_i b_i max_q(lambda_q g_iq) >= eta
```

协议采用“摘要上空口、完整记录本地重建”：prepare/vote/decision 携带完整 256-bit digest 和
32-bit generation，稀疏结构变更仍由原有布局计费；不重复发送两个稠密 `K×Q` 表。相对原
64-bit digest/16-bit epoch，每个实际发送包增加 208 bit。若依赖闭包含 `m` 个参与者，三轮协议
的额外 over-air bits 为 `208(m+1)`。这不是免费通信假设：本地重建所需端点/owner posterior
必须此前已经通过物理信道送达且代次匹配，原子协议本身仍逐链路检查 SNR、serialization latency、
端到端 deadline、RF 能量和通信—感知共享功率单纯形。

SHA-256 相等只是快速身份检查，不单独产生执行权；ACK 前仍要求参与者的规范记录逐字节相等。
因此即使抽象地假设哈希碰撞，不同物理模型或功率方案也会 fail closed。当前模块只是可执行前的
协议证书内核，尚未接入 `environment_core` 的 pending/active epoch，故不改变当前在线结果。

### 7.2.3 Atomic-epoch shadow 与通信闭包预筛

实验性相关 profile 现每 3 帧、且仅在真实控制载波存在时运行一次 counterfactual shadow。它读取
当前 active epoch，枚举相关感知 N5/N6 候选，执行证书重建并报告改善、摘要大小、空口 bits、
物理时延、Gram/LP 时间和副本一致性；函数没有 pending/active epoch 写句柄，并逐帧核验 active
structure 未改变。无载波帧不会假设零成本控制信道。

首次“先按感知排序 Top-12、再检查通信”的 30-case K6/Q4 压力测试被反例推翻：一个无控制
功率的 receiver 产生高感知收益候选并占据前排，使 Top-12 对通信可行邻域最优的 recall 只有
`80%`，最差值比为 `61.41%`。修正版不把 bit、W、s 与 Deflection 混为人为加权和，而先验证
候选 dependency closure 的原生 SNR、逐包/端到端时延和联合 RF 单纯形。固定行预算下，该物理
可行性与 target-wise 功率分配无关，故可在 Gram/LP 前安全预筛；精确功率解后仍重复同一检查。
修正版在相同 30 cases 上预筛平均 22/32 个物理不可行候选，post-LP 物理拒绝为 0；Top-4/8/12
recall 为 `93.33%/100%/100%`，Top-12 平均只解 9 个候选 LP，和通信可行 exhaustive 相同。

K16/Q16 的 3-seed、12-frame shadow 中，每 seed 尝试 3 次；exact accept/physical commit 比例依次
为 `2/3、3/3、3/3`，所有被接受候选的副本一致率和 active-structure 不变率均为 1。平均候选
提交空口量约为 `3.45--4.35 kbit`，独立三轮协议时延约 `1.41--1.58 ms`。但 Gram 枚举平均耗时
约 `125--170 ms`，已经超过 100 ms 控制帧，LP 另需约 `16--22 ms`；因此当前实现明确不能晋级
在线执行。该 shadow 的输入范围标记为 `centralized_physical_diagnostic`：当前副本由同一已构造
物理张量重建，一致率只验证证书/状态机接线，尚不能替代 packet-local 全候选张量重建证据。

随后加入两层不改变目标值的精确计算优化。第一层按 `(target, selected-edge mask)` 缓存相关因子，
并由 `E' xor E` 只访问 N5/N6 实际改变的 1--2 个目标；未变目标直接继承 incumbent 因子。第二层
利用 `c_q>=1` 推出 `g_corr(E')<=g_raw(E')`，先计算廉价上界：

```text
eta_corr*(E') <= U_lambda*(g_corr(E')) <= U_lambda*(g_raw(E'))
```

若最右项不超过 incumbent 即可在构造 Gram 前安全剪枝。对已排序候选还执行 branch-and-bound：
当前最好精确值达到下一候选的对偶上界时，所有余项都不可能更优，可提前终止。K16/Q16 同一
`7/19/43` 三种子 shadow 的 accept/commit 决策保持不变，Gram 时间降至约 `6.7--8.4 ms`，LP
约 `14.4--20.4 ms`，单节点证书重建约 `3.0--3.4 ms`。三项之和约 `24--32 ms`，在这组短程
诊断上低于预留的 45 ms 非无线预算；但仍需更多种子、packet-local 输入和与运动/安全阶段的联合
critical-path 门禁，不能据此启用在线结构写入。

### 7.3 对偶解释

令 `lambda_q>=0, sum_q lambda_q=1` 为目标价格，则对偶上界为：

```text
U(lambda) = sum_i b_i * max_q(lambda_q * g_iq)
eta* <= min_lambda U(lambda)
```

LP 返回的目标约束边际量归一化为 `lambda`。由此可计算可审计的 dual upper bound 和
primal–dual gap。零可达目标会使最优最小 Deflection 精确为 0，此时价格集中于零 ceiling 目标，
避免在退化基上伪造正证书。

### 7.4 严格分布式执行

每个节点从自己的 public view 构造同一类 LP，但“各自最优”并不推出“拼接后全局最优”。当前
主线先对完整 `(K,Q)` 增益张量执行逐字节公共模型证书：只有所有节点持有完全相同的系数视图，
才允许节点各执行本地解的自身一行；此时确定性求解使拼接矩阵等于同一个集中式 LP 解。

公共模型不是把节点私有 belief 强行视为相同，而是由无线协议包重建：所有节点（包括发送者）
都从端点状态的同一份量化广播 payload 恢复位置、速度和近场残差；目标状态则从当前结构中唯一
receiver/owner 发出的量化 posterior 恢复。发送者本地回环读取已经量化且设为不可变的 payload，
不读取未量化私有状态；该回环不占用额外空口 airtime，也不再次融合自身 posterior。若任一目标
没有唯一 owner、包未到达、时间戳来自未来或年龄超过 5 帧，则该 viewer 的共同模型无效。
所有 viewer 最终构造的完整增益张量仍须 byte-exact 相等，协议来源相同只是必要条件而非替代证书。

若证书失败，系统禁止拼接不一致 LP 行，也不声称 max–min 最优，而是逐发送者使用
`sparse_harmonic_row_power`：在该行有正保守增益的目标上令 `p_iq` 与 `1/a_iq^-` 成比例，使
`a_iq^- p_iq` 等化；无安全支持时均匀分配。该回退只依赖逐行下界，可由责任节点把贡献下界
相加形成 composable certificate。审计字段包括 `common_model_certificate` 与
`common_model_fallback_fraction`。

K16 配置允许 4 个本地 LP 进程并行，但 episode worker 固定为 1，防止嵌套进程池污染延迟
测量。并行超时或连续失败时回退 serial/incumbent/harmonic；任何回退都必须保留功率可行性。

### 7.5 下一代 belief-robust SOCP（已预注册，尚未成为当前证据）

对目标 `q`，将 belief 经 cubature/采样传播为随机系数向量 `a_q` 的均值 `mu_q` 与协方差
`Sigma_q`。对线性聚合 `a_q^T p`，单目标机会约束可用下式保守化：

```text
mu_q^T p - kappa_q * ||Sigma_q^(1/2) p||_2 >= eta
```

若只依赖均值方差并使用单侧 Cantelli 界，`kappa_q=sqrt((1-epsilon_q)/epsilon_q)`；若额外验证
近高斯性，可预注册 `kappa_q=Phi^-1(1-epsilon_q)`。目标间联合风险通过预先固定的
`sum_q epsilon_q<=epsilon_total` 分配控制。这是二阶锥约束，与每 UAV 功率单纯形共同构成 SOCP。
它只有在系数矩、校准覆盖率和数值条件门禁通过后才能替换确定性 LP；当前文档不把它写成已实现
性能。

### 7.6 统一对偶价格控制（研究主线定义）

下一代控制器不再为通信、结构、功率和移动分别堆启发式阈值，而用同一拉格朗日账本：

```text
target price lambda_q : 最弱目标/机会约束短缺
RF price rho_i        : P_comm + sum_q p_iq 的节点功率预算
energy price beta_i   : 电池因果预算
airtime/AoI price xi  : 控制包占空与信息陈旧
safety price zeta_ij  : 线性化 barrier 余量
```

功率块解 LP/SOCP；hyperedge 以物理收益减去 RF、能量和协议价格的 reduced cost 排序；事件触发
仅在预计对偶收益超过 bits/airtime/AoI 成本时广播；移动块解带 `zeta_ij` 的安全 QP。价格更新使用
投影次梯度并限制步长、量化与 AoI。该框架当前是下一代可证伪设计，不是本轮 smoke 的执行算法。

## 8. Hyperedge 结构与 owner 选择算法

### 8.1 设计动机

联合枚举角色、owner、边和功率具有组合爆炸。主线先用可由端点状态和本地 belief 重建的得分
协商稀疏结构，再对冻结结构精确求功率。这样把不可审计的联合黑盒拆为“有限轮结构协议 +
凸功率子问题”。

### 8.2 一轮有限协商

每个节点为候选 `(i,j,q)` 计算预算可重建的双基地能力与目标 deficit，广播 top-1 提议。
接收者只使用按时送达的提议；冲突按确定性排序规则解决。最终结构必须满足：

- 发射与接收端不同；
- 每个目标恰有一个 owner receiver；
- 所有目标均被至少一条有效边覆盖；
- 角色、owner 和边在各公共视图间一致；
- 结构保持窗内不因微小得分扰动频繁切换。

若完整性或一致性失败，strict profile 禁用 centralized safety fallback，因此保持旧的有效结构或
fail closed，而不是读取全局真值补齐。

## 9. 连续轨迹安全投影理论

### 9.1 约束推导

对 UAV 对 `(i,j)`，记当前相对位置 `r_ij=x_i-x_j`，相对位移
`du_ij=u_i-u_j`。要求下一端点满足：

```text
||r_ij + du_ij||^2 >= d_safe^2
```

展开并去掉非负项 `||du_ij||²`，得到充分仿射条件：

```text
r_ij^T du_ij >= (d_safe^2 - ||r_ij||^2)/2
```

投影器在位移范数球、矩形区域和上述 barrier 下，最小化与期望位移的平方偏差。距离大于
`d_safe+2*u_max` 的 UAV 对一帧内不可能碰撞，无需加入约束。

### 9.2 可独立组合安全

分布式执行不能假设所有节点同时拿到同一联合解。系统把一条 pair constraint 平分为两个局部
责任：

```text
r_ij^T u_i >= b_ij/2
(-r_ij)^T u_j >= b_ij/2
```

两式相加恢复联合 barrier。当前处于安全不变集时 `b_ij<=0`，因此陈旧节点执行零位移仍是可行
动作。视图过期、投影失败或返回非有限值时强制 hold。

### 9.3 可分二维精确投影

endpoint-split 后，每条约束只包含一个节点的二维位移，联合可行域是 `K` 个二维凸集的笛卡尔积，
平方距离目标也按节点可加。因此 32 维整队投影可严格分解为 16 个独立二维投影。每个二维集合由
仿射半空间、区域 box 与速度圆盘相交构成；最优点只能是：未修改的期望点、单条仿射边界上的
正交投影、两条仿射边界交点或仿射边界与速度圆的交点。枚举这些有限候选并取距离期望点最近者
即为精确解。

若候选集合为空，则该 public view 的 AoI 膨胀 barrier 在单帧速度界内不可行，系统直接返回
fail-closed hold。旧实现仍调用 SLSQP 直到迭代结束，最后也返回同一 hold；新实现删除了这段无效
延迟，但不放宽 barrier、速度、区域或 swept 约束。

### 9.4 为什么还测 swept distance

端点安全不自动排除两架 UAV 在帧内直线路径相交。正式 runner 对每对线段
`r(t)=r_0+t*du, t∈[0,1]` 解析计算：

```text
t* = clip(-r_0^T du / ||du||^2, 0, 1)
d_swept = ||r_0 + t*du||
```

严格配置还把公共位置年龄 `a` 转为两端点最坏运动不确定性：

```text
rho_pair(a) = 2*v_max*dt*a = 5a m
d_safe_eff = 20 m + rho_pair(a)
```

超过允许 AoI 的视图直接 hold。各 public-view projector 后，执行内核在任何 UAV 状态改变前对
最终拼装的整队命令解析计算 swept minimum；若低于 20 m，整队命令 fail closed 为 hold。这个
最后检查只读取 UAV 当前物理状态、不读取目标真值，但它是集中式仿真安全内核，因此不能据此
声称现实部署已经实现完全分布式 swept certificate。正式门禁同时要求执行前证书、endpoint 和
执行后 swept distance 均不低于 20 m。

## 10. 能量模型

单帧飞行能耗为：

```text
v_i = ||u_i||/dt
E_fly,i = (P_hover + c_v*v_i^2) * dt
```

通信能耗按实际广播 airtime 计费；deadline 失败包最多计一个 deadline 的发射 airtime，避免
无穷序列化时间。感知能耗按实际 OTFS 时钟计费：

```text
T_sense = n_cpi*N*T_sym = 1.024 ms
E_sense,i = sum_q p_iq * T_sense
B_i,t+1 = max(B_i,t - E_fly - E_comm - E_sense, 0)
```

公开电量状态为保证仿真稳定而截断，但门禁不再检查这个恒真的量。每次扣能先计算：

```text
B_raw,i,t+1 = B_i,t - E_fly - E_comm - E_sense
delta_E,i,t = max(-B_raw,i,t+1, 0)
energy_causality_violation = max_i,t delta_E,i,t
```

正式门禁要求 `energy_causality_violation_j=0`。`minimum_battery_j` 仅保留为剩余能量诊断。若分析
功率先按候选值暂扣、随后按最终解重算，实现会恢复暂扣前的 deficit 状态再记账，避免把未执行
候选误报为能量违规。

## 11. 主线算法完整流程

严格分布式解析基线每帧执行：

```text
输入：各节点自状态、本地 target belief、缓存消息、上一有效结构/功率

1. belief predict/update；删除超龄信息
2. 正常阶段每 3 帧建立控制载波；尚无完整 epoch 时进入逐帧 acquisition carrier
3. 物理 U2U 广播；按 SNR、容量、deadline 和可靠性决定逐链路 delivery
4. 各节点从实际 inbox 重建 public view
5. 若到达结构更新帧：执行一轮 finite-round hyperedge negotiation，结果先进入 pending epoch
6. 仅当 structure、owner map、endpoint packet、owner posterior 和源帧在所有 viewer 间逐字节一致时原子激活
7. 新 epoch 不完整则继续使用上一 active epoch；首次启动无 active epoch 时保持空结构 fail closed
8. 计算 active epoch 每条有效边的 a_ijq；再次核验完整增益张量的公共模型证书
9. 证书通过则精确求解同一 max–min LP；否则执行逐行 composable harmonic fallback
10. 每 20 帧用 Tx/Rx-aware bistatic bottleneck matching 更新移动责任
11. 将期望位移投影到可独立组合的 pairwise-safe 集合
12. 执行投影位移与功率，推进目标真值和测量过程
13. 计算 D_q、P_D,q、通信、时延、能耗、endpoint/swept 安全与运行时指标

输出：下一状态、检测指标、完整审计字段与 provenance
```

该流程中，策略提交的 movement 是 hold；实际移动来自解析 bottleneck 控制器。报告中不得把结果
称为 learned-policy 性能。

### 11.1 Atomic epoch 与软件架构边界

Atomic epoch 是 make-before-break 状态机，而不是仅添加一个整数标签。候选结构 `e` 只有在唯一
owner、完整量化端点包、对应 owner posterior、非未来且未超龄的源帧以及全部模型生成输入逐字节
一致时才 commit；随后派生 coefficient tensor 仍须通过独立 byte-exact 证书才能执行 LP，否则进入
逐行安全 fallback。依赖不完整时 active epoch `e-1` 保持可见。发送者回环只有在实际存在物理
广播时才能更新，禁止“本地准备了新 payload”被误当成“已经发到空口”。启动阶段额外载波会被
正常计入 bits、airtime、RF 功率和能量。

Architecture V2 当前准确状态是“边界已建立、核心迁移未完成”。checker 现在读取 `source_root`，
要求其中每个 Python 模块恰好属于一个 ownership layer；横切 legacy 约束使用非归属 layer，避免
重复所有权。相对导入先解析成绝对模块再检查，并对实际层依赖图检测环。`acceleration`、`agents`、
`coordination`、`environment`、`evaluation`、`legacy`、`optimization`、`physical`、`prediction` 和
`utils` 明确归入 `legacy_runtime`；后续采用 strangler seam 逐步迁移，不把当前状态写成重构完成。

## 12. 活动研究支线

### 12.1 Markov/KNN assignment

Markov 支线把状态写为 `(K+Q,6)`，每个 UAV 动作为目标索引。目标按 CV belief 预测，UAV 朝
分配目标移动不超过 2.5 m，并经过同一安全投影。对每个候选未来状态重新计算条件均值 Rician
物理系数和固定结构 LP，而不是用欧氏距离代理最终检测。

存在协方差时，四维目标 belief 使用 `2n=8` 个 spherical-radial cubature 点：

```text
xi_k = m ± sqrt(n) L[:,k],    weight = 1/(2n)
E[a] ≈ sum_l weight_l * a(xi_l)
```

可选保守系数为 `max(E[a]-beta*Std[a],0)`。stage cost 为
`-(worst_P_D + w_weak*weak_P_D)`。KNN 只生成交换候选；候选必须经过完整物理重评分且严格改善
incumbent 才接受。当前 32-case 结果置信区间跨 0，故仍为 shadow。

### 12.2 Fixed-lag smoother

该支线对截至当前帧的窗口运行 Kalman forward 和 Rauch–Tung–Striebel backward：

```text
J_t = P_t|t F^T (P_t+1|t)^-1
m_t|T = m_t|t + J_t(m_t+1|T - m_t+1|t)
```

最后一个 smoothed state 必然等于最后一个 filtered state，因此它不能凭空提供未来信息。其价值
是去噪历史和计算过程残差。当前白加速度模型的残差未来均值为 0；实验未证明 residual forecast
优于 CV，所以仅保留诊断用途。

### 12.3 Predictive constraint-native GNN

目标是学习候选排序或 warm start，而不替代物理约束。任何输出都必须经过原生物理重评分、
预算投影和 fallback certificate。当前尚未完成运行时接线和正式消融，因此不能进入主线。

### 12.4 Temporal constrained optimization

该支线研究跨帧功率 unroll、结构惯性和 BPTT。现有 feasibility-first 筛查没有证明正式 tail
指标改善，而且 surrogate-to-execution 单调性未闭合，因此默认关闭。

## 13. 理论上能保证什么、不能保证什么

### 13.1 已成立的条件保证

- 固定唯一 owner、固定系数时，max–min 功率问题是线性规划，可获得全局最优解和对偶上界。
- 分布式行拼接只有在公共模型证书通过时继承上述最优性；证书失败只保证逐行预算与可组合下界。
- 安全投影满足仿射 barrier、速度球和区域约束时，下一端点安全；runner 另行验证连续线段距离。
- 可独立组合约束允许各节点局部验证其动作，求和后恢复联合 barrier。
- 证据在检测模型的条件独立/白化假设下以 Deflection 相加。
- provenance gate 可以阻止跨提交、跨配置或跨种子银行复用正式结果。

### 13.2 尚不成立的保证

- Hyperedge 结构协议不是联合结构—功率全局最优算法。
- 旧 marginal-utility greedy 没有 `(1-1/e)` 保证。
- `3σ` belief 区域是统计覆盖假设，不是对所有目标轨迹的确定性保证。
- 合成 Markov 影子基准不能证明真实闭环收益。
- 软件 wall-clock 不能直接代表嵌入式飞控硬实时性能。
- 几何/充分统计量模拟不能外推为真实 28 GHz 硬件或波形级性能。
- 连续 ambiguity gain 不能单独证明多目标 DD 可分辨或多边证据独立。

## 14. 当前数据与结论

### 14.1 K16/Q16 十种子独立诊断 smoke

Atomic epoch 闭合后，使用不属于正式 blind bank 的 `20001--20010` 十个独立诊断种子，每个
30 帧、tail 20。delivery=1.0，通信 deadline violation=0，平均 bits/frame=2513.067；每个种子的
公共模型证书均为 0.9667，唯一未通过帧是首帧 fail-closed bootstrap，激活后没有
protocol-induced harmonic fallback。聚合 steady/weak3/worst 为
0.961748/0.937515/0.924460，短窗 QoS 为 10/10。

安全投影进一步利用 endpoint-split barrier 的可分性，将原 32 维 SLSQP 严格分解为 16 个二维
凸集投影；仿射边界、box 与 speed circle 的有限候选给出精确投影，无候选则证明该帧速度界内
不可行并直接 fail closed。跨种子 endpoint、执行后 swept 和执行前 swept minimum 均为
24.424 m，功率违反与截断前能量缺口均为 0。最大 per-seed closed-loop P95 为 76.132 ms，
最大单帧为 89.520 ms，所有种子的 online deadline miss 均为 0。该 smoke 达到了 P0 诊断目标，
但仍不是 clean-tree blind-100 正式证据。

### 14.2 Markov 影子基准

32 个 K16/Q16 合成 case 中，图候选 14 胜、9 平、9 负，接受率 0.90625。相对等计算预算
blind 的配对改善均值为 `1.0147e-5`，bootstrap 95% CI 为
`[-1.2407e-5,3.4669e-5]`。区间跨 0，只能说明机制可运行。

### 14.3 Fixed-lag 影子基准

32 seeds 下，white acceleration 的 smoothing history MSE 从 65.813 降至 17.642；但 residual
forecast 改善均值为 0.00233，95% CI 为 `[-0.01120,0.01560]`。因此保留 smoother 历史诊断，
拒绝残差预测在线晋级。

### 14.4 软件验证

本轮全量回归为 `1803 passed, 6 skipped, 7 warnings`。Architecture V2 检查现在还覆盖
source-root、唯一归属、相对导入和依赖环。strict
identity 的系统字段全部一致；clean-Git 项失败。formal gate 仍拒绝旧 blind-100，因为它绑定的
执行源码不是当前工作树。

## 15. 正式验收条件

当前版本必须在 clean commit 上完成完整 100-seed test bank，并同时满足：

- 每 episode：`steady>=0.80`、`weak3>=0.70`、`worst>=0.60` 才记为 QoS success；
- 100 个 episode success 的单侧 95% Wilson lower bound `>=0.80`；
- 平均 delivery `>=0.99`，平均 deadline violation `<=0.01`；
- 每个 seed 的 closed-loop critical-path P95 `<=100 ms`；
- K16 power-worker soft timeout 为 55 ms；其余 45 ms 留给 belief、协议、系数重建、安全与序列化；
- 每个 seed endpoint 和 swept UAV 最小距离均 `>=20 m`；
- 最大联合 RF 功率违反 `<=10^-9 W`；截断前最大能量缺口 `=0 J`；
- 最小电量只作诊断，不再作为能量因果性的充分证据；
- commit、source-tree、effective-config、seed-bank、运行包与线程环境哈希完全匹配。

## 16. 复现入口

```powershell
pytrch_ven\Scripts\python.exe -m pytest -q
pytrch_ven\Scripts\python.exe tools/check_architecture_v2.py
pytrch_ven\Scripts\python.exe tools/check_system_identity.py --manifest config/exp_strict_distributed_k16q16.yaml --strict
pytrch_ven\Scripts\python.exe tools/assert_formal_gates.py
```

正式 bank 必须通过受管 CLI/执行器运行，不能通过修改输出 JSON、跳过 clean-tree 校验或复用旧
episode 文件获得“通过”。详细实验设计和数据解释见 `docs/EXPERIMENT_LOG.md`。
