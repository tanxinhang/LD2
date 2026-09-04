# LD3 鲁棒原—对偶底层优化审计（2026-08-31）

## 1. 本轮结论

本轮完成的是可复用的数学底座修复，而不是把一个未经尾部验证的新启发式强行加入主配置：

1. 修复 fixed-owner max-min power 的对偶数值尺度，并实现真正唯一、目标排列等变的
   minimum-L2 canonical price；熵价格改为带显式最优值上界的 maximum-entropy 解。
2. 修复解析移动梯度与 Friis 重标定中的三维斜距，使其与当前固定高度的物理增益一致。
3. 在现有 gap/baseline envelope 内接入一个 opt-in 原—对偶候选；候选只使用已经送达的公开状态、
   本地 belief、已有功率和价格，不新增协议字段或通信 bit。
4. 增加 \(O(KQ)\) 的鲁棒上下界证书。严格候选在当前 binary-DD 不确定集下会正确地 fail closed，
   因为大量 \(A^-_{iq}=0\) 使下界退化为零；因此它保留为审计能力，默认关闭，不声称在线性能提升。
5. 一个放宽 DD 支撑条件的实验候选在 hard6 上出现决定性尾部回退，已撤回其在线实现，不能晋级。
6. 修复 Windows 上 Torch 私有 OpenMP 与 Conda MKL OpenMP 同时装载导致的 native abort，且没有使用
   `KMP_DUPLICATE_LIB_OK` 这类掩盖运行时冲突的开关。

因此，本轮可上线的是数值正确性、三维物理一致性、运行稳定性和审计基础设施；新移动候选尚未获得
“相对 baseline 拉开差距”的证据，主控制行为保持不变。

## 2. 对偶价格的正确数学底座

固定 owner 与 DD 支撑后，功率分配为

\[
\begin{aligned}
\max_{P,t}\quad &t\\
\mathrm{s.t.}\quad&\sum_i a_{iq}P_{iq}\ge t,\quad q=1,\ldots,Q,\\
&\sum_q P_{iq}\le b_i,\quad P_{iq}\ge0.
\end{aligned}
\]

令每个 target 约束的乘子为 \(\lambda_q\ge0\)。对 \(t\) 取上确界要求
\(\mathbf 1^T\lambda=1\)，对每个 transmitter 的功率单纯形取上确界，得到

\[
t^*(x)=\min_{\lambda\in\Delta_Q}f_x(\lambda),\qquad
f_x(\lambda)=\sum_i b_i\max_q\{\lambda_q a_{iq}(x)\}.
\]

原实现把一个 HiGHS 返回的任意 LP 顶点与均匀分布线性混合。这个混合一般既不留在最优面，
也不满足其注释所声称的 \(\tau\log Q\) 误差界。反例

\[
A=[1,1000],\quad b=1,\quad\tau=0.01
\]

中，精确值为 \(f^*=0.999000999\)，旧混合值为 \(5.989011\)，误差
\(4.990010\) 是声称上界 \(0.006931\) 的约 720 倍。

### 2.1 唯一 canonical price

当前实现先求精确 LP 值 \(f^*\)，再求

\[
\begin{aligned}
\min_{\lambda,u}\quad&\tfrac12\|\lambda\|_2^2\\
\mathrm{s.t.}\quad&u_i\ge a_{iq}\lambda_q,\quad\forall i,q,\\
&\sum_i b_i u_i\le f^*+\varepsilon_{num},\\
&\lambda\ge0,\quad\mathbf 1^T\lambda=1.
\end{aligned}
\]

可行域是闭凸集，目标对 \(\lambda\) 严格凸，所以解唯一；目标重编号只会对约束和解做同一排列，
因此价格排列等变且不依赖 LP basis。上述反例返回
\(\lambda=(0.999001,0.000999)\)；对 \(A=I_3\) 返回严格对称的
\((1/3,1/3,1/3)\)。100 组随机矩阵、增益尺度从 \(10^{-10}\) 到 \(10^{10}\) 的测试中，
最大排列等变误差为 \(2.61\times10^{-11}\)。

### 2.2 有诚实误差界的 entropy price

实现采用 maximum-entropy 的凸 cap 形式

\[
\min_{\lambda,u}\ \sum_q\lambda_q\log\lambda_q
\quad\mathrm{s.t.}\quad
\lambda\in\Delta_Q,\quad
f_x(\lambda)\le f^*+\tau\log Q.
\]

误差界现在是可行性约束本身，而不是对线性混合的错误推断：

\[
0\le f_x(\lambda_\tau)-f^*\le\tau\log Q.
\]

同一反例在 \(\tau=0.01\) 时得到 \(f=1.0059324708\)，误差
\(0.0069314718\)，位于数值容差内的理论边界。需要注意：这是“最大熵 + 显式 sublevel cap”，
不能误写成无约束 Tikhonov 熵正则目标。

## 3. 原始功率 LP 的量纲稳定性

物理上将所有增益乘以正数 \(c\) 不应改变最优功率：

\[
P^*(cA,b)=P^*(A,b),\qquad t^*(cA,b)=c\,t^*(A,b).
\]

旧求解在 \(A=10^{-12}[1,1000]\) 时受 HiGHS 绝对容差影响，可能返回最差目标值为零的顶点。
现在仅当 \(\max A<10^{-6}\) 或 \(\max A>10^6\) 时，以 \(\max A\) 归一化 LP，并在返回时恢复
deflection 量纲；正常物理范围不改矩阵，避免改变退化 LP 的历史 basis 和闭环轨迹。

400 个随机问题（100 个矩阵分别乘 \(10^{-12},10^{-8},10^8,10^{12}\)）无求解失败；相对未缩放
参考解的最大功率绝对误差为 \(1.18\times10^{-14}\)，恢复尺度后的最大目标误差为
\(3.67\times10^{-15}\)。seed 451 的默认控制在 316 个公共 CSV 字段上除 8 个 timing 字段外完全一致，
steady/weak3/worst、通信 bit 和连续最小间距均未改变。

## 4. 三维斜距梯度

对固定高度差 \(H\)，单端路径损耗项应使用

\[
\rho_{iq}=\|x_i-z_q\|_2^2+H^2,\qquad
a_{ijq}=\frac{C_{ijq}}{\rho_{iq}\rho_{jq}}.
\]

在 DD 支撑不变的光滑区域内，

\[
\nabla_{x_i}\log a_{ijq}
=-\frac{2(x_i-z_q)}{\|x_i-z_q\|_2^2+H^2}.
\]

解析 capability gradient、phase-1 endpoint gradient 与 Friis tensor rescale 已统一使用该斜距，
并以中心差分锁定。DD gate 仍是离散边界：解析梯度只负责提出候选，不能穿越 gate 后继续沿用旧导数；
候选必须重新构造增益和证书。

## 5. 鲁棒原—对偶移动证书

对同一个合法本地信息视图，设

\[
A^-(x')\le A^{true}(x')\le A^+(x'),
\]

并复用一个行可行功率 \(P\)、预算 \(b\) 和任意 simplex 价格 \(\lambda\)。定义

\[
L(x';P)=\min_q\sum_iA^-_{iq}(x')P_{iq},\qquad
U(x';\lambda)=\sum_i b_i\max_q\{\lambda_qA^+_{iq}(x')\}.
\]

由增益序关系、原可行性与弱对偶性，严格有

\[
L(x';P)\le t^*(A^{true}(x'))\le U(x';\lambda).
\]

在线候选先沿现有 deployed gain、功率和价格构造 saddle-consistent 方向，再经过已有安全投影；验收要求：

\[
L(x';P)>L(x;P)+\delta.
\]

可选的强支配门进一步要求

\[
L(x';P)>U(x;\lambda)+\delta.
\]

同时保留现有五分量 gap-coverage 势的逐分量 Pareto 非回退，防止一个加权 deflection 改善掩盖
Tx/Rx 可达性退化。整个计算为 \(O(KQ)\)，使用本地缓存和已送达状态，没有逐候选 LP、全局真值、
新 packet 或新共识轮次。只增加一个小型代数证书文件，没有新增运行子系统。

### 5.1 为什么严格候选暂不晋级

当前 robust belief 将不确定 DD support 的保守端置零。只要某个 target 的所有可行见证在
\(A^-\) 上出现零支持，\(L=0\)，严格改善门就会 fail closed。seed 451 与 seed 1755461577 中，
证书约 96.6% 帧可构造，但 incumbent/candidate \(L\) 均为零、候选选择率和强支配率均为零。
这不是实现错误，而是当前不确定集合过宽时正确但无信息量的证书。

曾实验性地在“名义 DD support 不变”时只对 pathloss 做条件下界，并附加 coverage-Pareto 门。
hard6 中候选选择率仅 0.892%，仍得到：

| 指标 | 当前主控制 | 条件支撑候选 | 差值 |
|---|---:|---:|---:|
| steady P_D | 0.744376 | 0.739823 | -0.004553 |
| weak3 P_D | 0.630636 | 0.631730 | +0.001095 |
| worst P_D | 0.583252 | 0.581299 | -0.001953 |

其中 seed 1914864024 的 weak3/worst 分别下降 0.055599/0.152075，构成决定性反例。该放宽分支已从
在线代码删除，仅保留 artifact 作为负结果；严格分支默认关闭。

## 6. 运行速度与系统稳定性

单线程、\(K=Q=12\) 的当前机器微基准如下（热启动后平均）：

| 运算 | 平均耗时 |
|---|---:|
| 正常尺度 power LP | 0.703 ms |
| \(10^{-12}\) 增益的自适应缩放 LP | 0.708 ms |
| minimum-L2 canonical price | 2.652 ms |
| capped maximum-entropy price | 4.229 ms |
| 单个鲁棒 \(L/U\) 证书 | 0.0477 ms |

结论是：证书适合在线 envelope；canonical/entropy 应在结构更新时低频求解并缓存，不能默认在每个
viewer、每帧重复求解。正常尺度 fast path 避免无意义的数组除法和拷贝。

Windows 评估入口现在只装载 Conda 的一份 `libiomp5md.dll`，Torch、NumPy/MKL 与 HiGHS 交错调用
已通过；此前的 `OMP Error #15` native abort 不再出现。crash-isolated 合并器还会按请求 seed 顺序
恢复结果，避免数值排序破坏配对统计。

## 7. 验证结果与理论边界

- 目标功率/对偶测试：70 passed；本轮相关测试：111 passed。
- 全量回归：1402 passed，8 个既有 warning。
- 随机量纲 stress：400/400 通过。
- seed 451 默认关闭候选时，steady/weak3/worst 为
  0.212874/0.031828/0.027197，与修复前控制完全一致；U2U bit/frame 同为 849.307。
- 新候选不增加通信字段或 bit，但当前证据不允许声称其闭环性能优于 baseline。
- common-view rate 为零时，各节点对自己的本地矩阵求解并只执行自己的功率行。严格能声称的是
  行功率可行和每个本地视图内的证书成立；不能把拼接结果称为集中式全局最优。

## 8. 下一步最值得深挖的理论方向

下一步不应再增加移动启发式，而应让现有 DD 不确定集产生非零、可校准的聚合下界。建议在同一
模块内将 binary support worst-case 改为“置信预算约束的 aggregate deflection”：

\[
\min_{\xi\in\mathcal U_\alpha}
\min_q\sum_i a_{iq}(\xi)P_{iq},
\]

其中 \(\mathcal U_\alpha\) 由已有 delay/Doppler posterior 和总失配概率预算 \(\alpha\) 构造，
而不是逐边独立地把所有不确定支撑同时置零。可用 union bound 或 distributionally robust ambiguity set
保证整体失效概率，不需要新增通信字段。只有当这一聚合下界通过覆盖率校准、hard6 尾部门和 blind
paired CI 后，才重新打开原—对偶候选。这条路径解决的是证书的保守性根因，而不是继续堆规则。
