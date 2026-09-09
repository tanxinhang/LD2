# Actor-head responsibility audit（阶段 1，2026-09-07）

## 1. 审计问题

本审计不问“网络是否定义了某个输出头”，而问两个可执行问题：

1. canonical strict runner 是否实例化并调用 learned actor；
2. 若把 actor-like 输出字段反事实地提交给同一 strict environment，该字段是否能穿过
   protocol/solver override，改变最终位置、结构、检测、通信功率、感知功率或 bit。

可复现工具：`tools/audit_actor_head_responsibility.py`。

## 2. 环境校正

系统 PyTorch 位于 `E:\anaconda\conda\python.exe`，实测版本 `2.11.0+cu128`，CUDA 可用。
此前 PATH 首项 `E:\anaconda\3_11_python\python.exe` 没有 PyTorch；该解释器缺包不能代表
系统缺少 PyTorch。使用正确解释器后：

```text
tests/test_architecture_v2.py + tests/test_analytical_sensing_power.py
27 passed
```

## 3. 正式执行责任

`tools/run_strict_distributed_pilot.py` 没有加载 checkpoint 或构造 actor，而是提交 hold movement、
idle role、zero-content/zero-rate carrier、固定通信功率比例和 uniform sensing weights。因此当前
正式结果中不存在可归因给 actor head 的性能增益。

| 学习分支中的 head | 预期策略变量 | strict 正式 runner 中的实际责任方 | actor 是否拥有正式执行权 |
|---|---|---|---|
| `dp_head` / `v2_movement_head` | UAV 位移 | distributed bistatic movement | 否 |
| `role_head` | Tx/Rx/idle | `learn_roles=false`，P0/role-agnostic structure | 否 |
| `v2_assignment_head` | target/endpoint responsibility | deterministic hyperedge/P0 | 否 |
| `comm_target_token_head` | learned target message | `hyperedge_protocol_only` fixed state payload | 否 |
| `comm_rate_head` / `comm_set_rate_head` | silence/rate | runner 固定 rate 0；fixed-bit protocol 仍发送 | 否 |
| `isac_power_mean_head` / `isac_set_power_head` | communication RF fraction | runner 固定 carrier fraction | 否 |
| `v2_sensing_head` | per-target sensing simplex | runner uniform input；执行由 max-min LP 重分配 | 否 |
| centralized/equivariant critic heads | value/credit | 只在训练使用，strict runner 无 actor/critic | 否 |

## 4. Counterfactual field intervention

配置为 `config/exp_strict_distributed_no_truth_pilot.yaml`。对 seeds 101/211/307 分别运行 20 帧；
每次只改变 UAV 0 的一个 actor-like 输出字段，其余状态、随机种子和输入保持一致。

| 干预字段 | 三 seed 中位置/结构/检测是否变化 | 功率/bit 是否变化 | 判定 |
|---|---|---|---|
| movement | 否，所有 max absolute delta = 0 | 否 | 被 L3 完全覆盖 |
| role | 否，所有 max absolute delta = 0 | 否 | 在 role-agnostic P0 中无执行权 |
| learned message content | 否，所有 max absolute delta = 0 | 否 | 被 protocol-only 抑制 |
| learned rate 0→1 | 否，所有 max absolute delta = 0 | 否 | fixed-bit protocol 绕过 learned payload rate |
| communication-power fraction 0.25→0.75 | 位置/结构/检测不变 | UAV 0 通信功率变化 0.5 W；感知功率与 bit 不变 | 字段能到达 RF 记账，但本实验内没有任务因果效应 |
| sensing weights uniform→one-hot | 否，所有 max absolute delta = 0 | 否 | 被固定结构 max-min LP 覆盖 |

这里的“无变化”仅对应该 strict 配置和 3×20 帧责任探针，不外推到所有 research profile。
尤其是未开启 analytical override 的 learned-policy profile 中，这些 head 仍可能有执行作用。

## 5. 阶段 1 结论

1. 当前 strict 系统是解析分布式控制基线，不是 MARL policy deployment。
2. 在 strict profile 内，actor 的 movement、role、assignment/message、rate、sensing heads 均无
   可观测的最终执行权；communication-power 字段只能改变 RF 功率记账，而 runner 又把它固定。
3. 因此不能通过“冻结几个 dead head”把 strict 系统变成可归因的混合 MARL。必须先选择论文
   对象：解析系统，或重新接回一个拥有明确长期动作自由度的 learned policy。
4. 暂不从共享 actor 类中删除这些 heads，因为它们仍服务其他 learned research profiles；直接
   删除会把“正式 profile 无责任”错误推广成“整个仓库无用途”。

下一阶段进入 oracle state/candidate/rank/projection ladder，定位 6/6 性能损失究竟来自信息、
候选生成、排序还是可行投影。

