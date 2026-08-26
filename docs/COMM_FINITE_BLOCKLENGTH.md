# U2U 有限码长通信：模型、闭环与声明边界

## 核心论点

在保持历史 Shannon 基线可复现的前提下，系统新增了可选的有限码长正常近似，使 learned
message 与检测 evidence packet 共同受到 blocklength、BLER、deadline 和实际擦除的约束；该层
闭合了解析通信可靠性，但不等价于具体信道编码、MAC 或硬件链路验证。

## 术语表

| 规范术语 | 定义 |
|---|---|
| channel use | 一个复 AWGN 信道使用；实现中按 `bandwidth × serialization time` 计数 |
| BLER | block error rate，整个 packet/codeword 的错误概率 |
| FBL normal approximation | finite-blocklength second-order normal approximation |
| analytical L0 | 满足 U2U 物理约束所需的最小通信功率层 |
| reliability erasure | 按 BLER 进行的可复现 packet 丢弃；丢弃包不进入 actor 或检测融合器 |

## 1. 数学模型

对线性 SNR `gamma`、复信道使用数 `n`、信息 bit 数 `k` 和目标 BLER `epsilon`，采用二阶正常近似

\[
k \simeq nC(\gamma)-\sqrt{nV(\gamma)}Q^{-1}(\epsilon),
\]

其中

\[
C(\gamma)=\log_2(1+\gamma),\qquad
V(\gamma)=\left(1-\frac{1}{(1+\gamma)^2}\right)(\log_2 e)^2.
\]

给定 `gamma,k,epsilon` 时，整数二分得到最小 `n`；给定 deadline 对应的 `n` 时，对 `gamma`
二分得到最小所需 SNR。BLER 反演为

\[
\epsilon_{\rm NA}=Q\!\left(
\frac{nC(\gamma)-k}{\sqrt{nV(\gamma)}}
\right).
\]

实现没有添加未经证明的 `0.5 log2(n)` 三阶收益，因此相对于包含正三阶项的近似更保守。但正常
近似本身仍不是具体编码族的可靠性保证。

## 2. 链路闭环

启用 FBL 后，每条 U2U 链路依次执行：

1. 由距离、载波、天线增益、带宽、噪声温度和发射功率计算 SNR；
2. 求满足目标 BLER 的最小 blocklength 与 serialization time；
3. 同时检查 SNR threshold、deadline 和 BLER；
4. 可选地使用环境 RNG 按 BLER 采样 codeword erasure；
5. 失败包计费但不进入接收 actor；检测 evidence packet 使用同一可靠性接口，失败证据不进入 owner
   fusion。

这避免了“learned message 有可靠性、检测上报仍理想无损”的双重口径。

## 3. 与 L0 功率层的一致性

固定等带宽 `B_i` 时，deadline 给出

\[
n_i=\left\lfloor B_i(T_{\rm deadline}-T_{\rm proc})\right\rfloor,
\]

系统用正常近似反求 `gamma_i^FBL`，再执行

\[
P_i^{\min}=\frac{
\max(\gamma_{\rm threshold},\gamma_i^{\rm FBL})N_0B_i
}{g_i^{\rm worst}}.
\]

原 Shannon optimal-bandwidth KKT 使用
`B(2^(r/B)-1)` 的特定凸结构；加入 channel dispersion 后，该导数和最优性证明不再成立。因此
配置同时启用 FBL、analytical L0 和旧 optimal-bandwidth KKT 时会 fail closed。当前只支持
FBL + analytical L0 + equal bandwidth；新的 FBL 带宽优化必须另行证明凸性或采用有证书的数值解。

## 4. 配置与复现

```yaml
comm_finite_blocklength_enabled: false
comm_finite_blocklength_target_bler: 0.00001
comm_finite_blocklength_max_channel_uses: 100000000
comm_finite_blocklength_sample_errors: true
```

默认关闭，因而旧 artifact 的 Shannon latency、power 和随机序列保持不变。FBL 擦除复用环境 RNG，
相同 seed 和相同 packet 调用顺序可复现。

## 5. 已验证与未验证

已验证：容量/色散极限、最小 blocklength、SNR 反演、FBL latency 大于同条件 Shannon latency、
deadline fail-closed、可复现擦除、learned packet 与 evidence packet 共用可靠性门禁，以及不相容
Shannon KKT 的配置拒绝。

尚未验证：具体 LDPC/Polar 编码、调制解调、信道估计、同步、衰落下的 coding gain、HARQ、MAC
冲突、路由、队列和外场链路。因此当前可声称“解析 FBL 可靠性进入仿真闭环”，不能声称“真实
通信协议或硬件可靠性完成认证”。

## Claim–evidence map

| 声明 | 证据 | 状态 |
|---|---|---|
| FBL serialization 和 BLER 进入 learned U2U 链路 | 单元测试与通信回归 | supported analytically |
| evidence packet 受相同擦除约束 | evidence routing integration test | supported analytically |
| FBL 与旧 Shannon KKT 可直接组合 | 实现明确拒绝该组合 | false / fail closed |
| 链路达到具体信道编码的 `1e-5` BLER | 无码级仿真或硬件数据 | needs evidence |
