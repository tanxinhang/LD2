# 高维预测—证书双平面架构升级方案

## 1. 目标重置

O6/O7保留为底层算子，但不再作为达到20--30 ms整帧目标的主方案。K16/Q16当前实测为
53.94 ms/帧、控制关键路径11.55 ms；即使删除全部超边协商，仍不足以消除约24--34 ms
的差距。新的优化对象必须是整帧数据依赖和计算频率。

新架构为：

`统一时空状态库 -> 高维预测慢平面 -> selected-only精确证书快平面 -> 执行与反馈`

预测提前给出候选、active set和warm start；证书负责可见性、物理上下界、功率可行性和
QoS fail-closed。预测错误只增加fallback，不能产生未经验证的动作。

## 2. 为什么不能直接增加DRL/GNN

当前瓶颈不是缺少复杂策略，而是同一状态被对象层、通信层、观测层和物理层重复解码、复制
和重建。直接在旧管线前增加GNN会新增推理开销，却不删除旧路径。

近期工作中真正可迁移的模块是：belief-state预测处理部分可观测资源分配；图网络表达用户/
信道拓扑并减少CSI和信令开销；学习warm start或active set减少求解迭代；预测调度把计算
移到资源请求之前；预测误差显式进入鲁棒约束或证书门。

本系统的创新点应定义为**证书校准的时空超图预测**：同一预测器给出离散边候选、连续gain
区间、LP active set/dual warm start和风险半径，再由现有代数证书在selected COO上验证。

对应的近三年模块证据包括：2024年的部分可观测ISAC model-based online learning将预测放在
belief-state MDP中；2024年的预测波束成形使用时变感知结果预测下一时隙CSI；2024年的
PCGNN明确研究不同图属性所需CSI/信令开销与性能的权衡；2023年的learning-to-warm-start
把网络输出接到固定点算法而非替代算法；2025年的GNN active-set预测进一步展示了跨问题
规模的solver warm start；2024年的ElaSe用预测调度减少资源匹配延迟。它们支持这里的模块
选择，但没有任何一项单独构成本系统的创新声明。

## 3. 统一高维数据层

禁止构造历史`[H,V,K,K,Q]`。高维信息使用关系分解：

| 数据块 | 建议形状 | 主要字段 |
|---|---|---|
| endpoint history | `[H,V,K,Q,F_e]` | 相对位置/速度、AoI、可见性、量化半径、belief均值与协方差谱 |
| selected edge history | `[H,V,E,F_s]`, `E<=3Q` | tx/rx/target、nominal/lower/upper、DD余量、切换年龄 |
| resource history | `[H,V,K,Q,F_r]` | 功率、row slack、target deficit、dual price、active-set标志 |
| protocol history | `[H,V,K,F_p]` | packet age、delivery、bit budget、发送/接收角色 |
| target/global tokens | `[H,V,Q,F_t]`, `[H,F_g]` | covariance谱、QoS余量、worst/weak3、frame phase、hold age |

这些数组进入固定长度ring buffer，按frame版本只写一次。ObservationBuilder、通信打包、超边
预测和证书读取同一份SoA数据，不再分别从Python UAV/Target/packet对象重建状态。

历史窗先测`H={2,4,8}`，不能因为“高维”默认使用更长历史。消融必须报告每个数据组的预测
增益、推理耗时和内存。

## 4. 预测慢平面

采用轻量因子化时空超图网络，而不是全连接Transformer：

1. endpoint encoder编码`(viewer,node,target)`，参数在K/Q维共享；
2. temporal gated convolution只沿H维执行；
3. tx--target与rx--target两类消息传递后在候选边融合；
4. 四个输出头共享主干：
   - `edge_head`：下一帧/下一hold周期的top-M候选及owner logits；
   - `gain_head`：预测log nominal和相对解析上下界的残差；
   - `solver_head`：预测非零`(tx,q)` active set、dual price和primal warm start；
   - `risk_head`：预测误差分位数/OOD score，决定是否允许快路径。

模型同时预测`t+1`和`t+H_hold`。前者服务逐帧功率，后者服务拓扑预取。推理在frame t结束
后写入双缓冲，frame t+1只读取已完成结果；模型推理不得重新串入关键路径。

## 5. 精确证书快平面

每帧在线流程压缩为：

1. 读取上一帧完成的候选、active set和warm start；
2. 对候选并集执行selected-only真实几何及nominal/lower/upper重建；
3. 用`a+`验证被排除边不可能进入top-3，失败则扩展候选或回退完整拓扑；
4. 将预测active set送入稀疏LP，缺失约束由dual/primal residual检测；
5. projection保证row simplex、owner唯一性和target覆盖；
6. 证书给出worst/QoS下界；非有限值、OOD、界违反或deadline miss均使用上一份已认证
   incumbent。

模型从不直接签发可行性，因此预测降低平均计算量，而安全语义仍由解析层定义。

## 6. 必须删除的旧同步工作

- 每帧两次DenseDeflection对象物化，改为selected COO直接计算P_D；
- 每个agent独立从对象构造观测，改为统一tensor view和mask；
- packet逐对象解码/融合，改为结构化数组上的批量scatter/gather；
- 每帧冷启动K个LP，改为预测active set和上帧primal/dual warm start；
- 拓扑更新帧全图同步阻塞，改为提前预测、selected验证、必要时full fallback。

如果新模型上线后这些路径仍执行，实验应判定为架构失败。

## 7. 分阶段实施与硬验收

### A0：高维数据集与延迟分账

从exact teacher生成ring-buffer样本。标签包括selected边、三套gain、LP active set、primal、
dual、P_D和证书gap。记录seed/frame/K/Q/可见性/AoI；同一episode不得跨训练和测试集合。

验收：重放样本逐项恢复当前golden；采集额外开销低于1 ms/帧；无hidden truth泄漏。

### A1：统一状态库和批量观测/通信

先不启用预测，让旧算法读取统一tensor，隔离数据架构收益。

目标：整帧53.94降到40--44 ms；golden零差异或仅机器精度差；worst/QoS不变。

### A2：预测器shadow mode

训练四头模型但不控制系统。报告top-M recall、owner准确率、log-gain误差、active-set recall、
warm-start residual、OOD calibration和推理时间。

硬门：selected edge recall >=99.9%；owner recall=100%或由规则补齐；active-set recall
>=99.5%；CPU异步推理p95低于一帧周期；未见seed和未见K分别报告。

### A3：证书门控的预测拓扑

预测候选替代同步全图排序，但selected物理和上界剪枝仍精确；先只在hold更新帧启用。

目标：控制均值低于9 ms，fallback低于5%，拓扑更新帧P95下降；W=3 worst_min仍高于0.75。

### A4：预测active set + sparse LP

比较serial、thread和process；模型只warm-start，residual失败自动扩集。报告变量数、nnz、
solver wall、IPC wall和projection wall。

目标：功率阶段均值低于3 ms、p95低于5 ms，LP目标与exact teacher的相对gap满足门限。

### A5：selected-only真实物理 + 异步整帧

删除非必要pair-dense真实deflection和对象物化，形成最终快路径。

目标：K16/Q16同一10-seed诊断集整帧均值20--30 ms、p95低于35 ms；W=3
worst_min>=0.75、QoS=100%。随后用未见100-seed确认集验证，诊断集不能作为最终论文结果。

## 8. 防止fake创新的消融

必须比较：无预测仅统一数据层、MLP、单帧GNN、时空GNN、去掉risk head、完整预测+证书、
完整预测但保留旧dense路径。

只有完整方案在未见seed/K上同时改善延迟、fallback和worst/QoS，才能声称架构创新。若MLP
与时空GNN相当，应选择MLP并放弃高维图模型叙事。

## 9. Go / No-Go

- 继续微调O6/O7：No-Go，收益上限不足。
- 直接用DRL输出功率/边：No-Go，缺少可行性和OOD保证。
- 统一状态库 + shadow预测 + selected证书：Go。
- 未收集数据就确定网络深度、H或attention结构：No-Go。

## 10. A0当前进度

已扩展诊断teacher trace并新增`tools/export_predictive_teacher_dataset.py`。首个K16/Q16、
5-frame样本包含`[5,16,16,16,2]` endpoint position及同形速度、target状态/不确定度、
visibility、三套gain、selected COO、power、dual price和cache-valid标签。压缩NPZ为156,936
bytes，最大边数48；与原golden零容差对拍全部一致。单独复制/堆叠147,456 bytes高维载荷的
5000次微基准均值为0.013 ms，低于A0的1 ms门槛。该数字只说明采集内存操作，不包含离线
JSON序列化，也不作为在线推理耗时。

## 11. A1当前进度

第一段将严格U2U target-token观测从16次`build_local_obs`改为共享belief/几何张量和SoA
token grid。逐agent单测及系统golden均为逐元素零差异。相同cProfile口径下，观测构建由
10.65降到8.32 ms/帧。交错legacy/batch A/B的10 seed均值显示整帧56.16降到54.20 ms、
P95 78.80降到73.08 ms；进程池噪声较大，因此隔离profile是该子阶段的主要归因证据。

第二段利用广播payload不可变性，将hyperedge、composable certificate和owner posterior从
每个receiver重复解码/验证改为每个sender一次解码，再只向物理delivery确认的receiver集合
批量scatter。未来帧mailbox、移动anchor和relay配置继续走原逐receiver路径。通信阶段的
cProfile由12.10降到8.20 ms/帧。

O7基线与A1最终10 seed × 30 frame对照：

| 指标 | O7/A0 | A1最终 |
|---|---:|---:|
| 整帧均值 | 53.94 ms | 48.52 ms |
| 整帧P95均值 | 73.86 ms | 65.09 ms |
| 控制关键路径均值 | 11.55 ms | 11.37 ms |
| W=3 worst_min | 0.844917 | 0.844917 |
| W=3 QoS | 100% | 100% |

10个seed的整帧均值全部下降。A1相对A0降低10.1%，但未达到40--44 ms阶段门，因此状态为
“部分通过”。这也表明剩余差距主要在真实deflection对象化、owner posterior fusion、belief
更新和LP等待，而不是继续压缩观测布局。

## 12. A2 GNN v0实测与架构修正（2026-09-08）

已实现`uav_isac.prediction`影子包：置换等变的UAV--target消息传递主干、Rx条件化的低秩Tx
解码、下一帧prefetch cache及fail-closed refresh gate。条件解码只对少量Rx beam计算
Tx兼容性，不生成`[V,K,K,Q]`。模型仅12,420参数。

teacher扩为10 seed × 12帧（120帧），严格按seed切分8个训练、2个留出；标签右移一帧，
杜绝同轨迹随机拆帧泄漏。单帧几何版在留出集owner recall仅61.65%、edge recall 28.03%、
完整帧覆盖0%，因此否定“静态几何GNN直接替代协商”。加入当前owner/Tx、hold相位、
`a/a-/a+`、power及dual price后，18维时序残差版取得：

| 留出指标 | 单帧几何GNN | 时序残差GNN |
|---|---:|---:|
| owner@2 recall | 61.65% | 100% |
| edge recall（96候选） | 28.03% | 96.69% |
| 完整帧覆盖率 | 0% | 81.82% |
| 非hold边界完整覆盖率 | — | 100%（18/18） |
| hold边界edge recall | — | 81.77%（4次转换） |
| hold边界完整覆盖率 | — | 0% |

当前边直接延用的全数据edge recall为95.36%、完整帧覆盖率81.82%；这证明网络提升主要来自
读取协议状态，而未解决刷新边界。把beam扩大到3×5和4×6会将候选增至240/384，留出残差
recall仅97.16%/97.44%，完整帧覆盖仍为81.82%，因此否决以候选膨胀换召回。

驻留内存CPU单线程基准：18维特征构建5.82 ms、GNN均值2.39 ms（p95 3.08 ms）、条件
解码1.59 ms，总计约9.80 ms。它必须在hold周期内异步执行，不能同步叠加到刷新帧。

由此将A3执行语义修正为：4个hold帧只重算selected-edge精确物理；GNN为第5帧刷新异步
预生成候选；刷新时依次检查selected物理、omitted-edge解析上界和LP residual，三门全过才
使用稀疏刷新，否则完整exact fallback。当前checkpoint明确`production_eligible=false`。
下一硬任务不是加深GNN，而是接通selected-only物理和omitted-edge补洞；在这两项完成前，
不能声称达到20--30 ms，也不能让预测器控制环境。
