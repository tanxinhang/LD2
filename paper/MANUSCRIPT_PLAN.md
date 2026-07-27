# Manuscript Plan: Sparse U2U Semantic Consensus Bidding for Multi-UAV ISAC

## Status and scope

**Method status:** frozen for manuscript preparation. The working method is the
distributed consensus-bid configuration in
`config/exp_800_q4_u2u_hierarchical_multistatic_distributed_matching_hybrid50_eval.yaml`.

**Paper type:** algorithmic paper with a systems validation component.

**Current evidence level:** proof of mechanism on one frozen trained checkpoint,
evaluated on 100 predefined test-bank geometry seeds. This is sufficient to define the
method, the main hypothesis, and the final experiment matrix, but it is not yet
sufficient for submission because independent end-to-end training runs and
strong learning baselines are missing.

## One-sentence paper argument

In a tracking-free multi-UAV ISAC mission without a ground link, sparse,
identity-consistent U2U target tokens can turn locally inferred target
preferences into a shared assignment graph, and an inertia-mixed consensus
projection can coordinate motion and sensing under an exact 1 W per-UAV RF
budget, raising the mean worst-target detection probability from 0.543 to 0.798
on 100 predefined evaluation geometries while exposing a remaining severe lower-tail failure
mode.

## Claim boundary

The defensible claim is currently narrower than “fully decentralized ISAC.”

- The actor executes from local mission geometry and physically delivered U2U
  messages; free neighbour-state and UAV-to-ground information are disabled.
- Each UAV reconstructs the same UAV-ID-indexed movement bid graph and applies
  the same deterministic permutation projection when all packets arrive.
- The simulator still uses an environment-level feasibility scheduler to choose
  multistatic transmitter/receiver edges inside the policy-constrained sensing
  subgraph. Therefore, the present result demonstrates distributed motion and
  resource-intent coordination, not a completely coordinator-free sensing
  scheduler.
- The final consensus mechanism has been evaluated as a frozen-policy structural
  intervention. End-to-end retraining of the final architecture is required
  before claiming a complete learned method.

## Working terminology ledger

| Concept | Canonical term | Symbol / abbreviation | Avoid |
|---|---|---|---|
| Overall method | Distributed Consensus-Bid MAPPO | DCB-MAPPO (working name) | Causal-Comm, CCP, CTMH unless used as ablations |
| System | U2U-assisted multi-UAV ISAC | U2U-ISAC | UAV-ground ISAC |
| Message unit | target token | \(z_{kq}\) | packet feature, belief packet |
| Sparse transmission decision | top-2 target claim | \(m_{kq}\) | fixed target message |
| Unprojected local preference | intrinsic bid | \(b^{\mathrm{int}}_{kq}\) | raw commitment |
| Consensus result | projected commitment | \(c_{kq}\) | centralized assignment |
| Bid memory/correction trade-off | bid-inertia mixing | \(\beta\) | communication weight |
| One-to-one motion allocation | permutation projection | \(\Pi_{\mathrm{perm}}\) | Hungarian oracle |
| Two-endpoint sensing allocation | capacity-two projection | \(\Pi_{2}\) | one-to-one Sinkhorn |
| Mean target quality | steady detection probability | steady \(P_D\) | average reward |
| Bottom-three quality | weak-three detection probability | weak3 \(P_D\) | weak QoS |
| Strict weakest-target quality | worst-target detection probability | worst \(P_D\) | trimmed worst |
| Per-scenario deployment event | QoS-feasible episode | \(F_e\) | average feasibility |

The acronym DCB-MAPPO is provisional and should be confirmed before the title is
finalized.

## Proposed contribution claims

1. **Identity-consistent sparse semantic consensus.** Each agent broadcasts only
   its two highest target-token bids. Delivered messages are scattered into
   globally indexed UAV rows using sender identity, so all receivers can obtain
   the same team bid graph without exchanging raw observations.

2. **Task-separated graph projections.** A hard one-to-one permutation defines
   the slow movement commitment and guides the executed actor motion, whereas a
   differentiable capacity-two projection guides fast multistatic sensing
   intent. This separates the
   one-UAV-per-target kinematic objective from the two-endpoint-per-target
   sensing requirement.

3. **Inertia-mixed bidding under a hard joint RF budget.** The transmitted bid is
   a 0.5 mixture of the live intrinsic preference and the projected commitment,
   which avoids both projected-bid lock-in and unstable purely intrinsic
   reallocation. Communication power and target-wise sensing power satisfy
   \(P_k^{\mathrm{comm}}+\sum_qP_{kq}^{\mathrm{sen}}=1\,\mathrm{W}\) exactly.

4. **Worst-QoS-first validation.** The evaluation reports strict per-episode
   worst-target quality, its bootstrap lower confidence bound and bottom-20%
   CVaR, plus a Wilson lower bound on simultaneous satisfaction of the
   steady/weak3/worst deployment floors. Communication volume is only a final
   tie-breaker.

Claims 1--3 require end-to-end multi-seed confirmation. Claim 4 is already
implemented and auditable.

## Evidence map

| Claim | Current evidence | Status |
|---|---|---|
| Consensus improves average QoS | Matched frozen-checkpoint comparison over 100 predefined test geometries: steady 0.8599→0.9452, weak3 0.8133→0.9269, worst 0.5429→0.7984 | Supported for this checkpoint |
| Consensus improves deployment feasibility | QoS-feasible rate 0.41→0.78; one-sided 95% Wilson LCB 0.3325→0.7050 | Supported for this checkpoint |
| Bid inertia is necessary | \(\beta=0,0.5,1\) test: hybrid has the highest worst, LCB, CVaR and feasible rate | Supported structurally; training-seed replication missing |
| U2U communication improves weakest-target quality | On 100 paired geometries, forced silence reduces worst 0.7984→0.7055 even though all RF power returns to sensing; paired difference CI [0.0472, 0.1415] | Supported for this checkpoint |
| Capacity-two sensing projection is necessary | Removing the projection reduces worst 0.7984→0.5362; paired difference CI [0.2131, 0.3116] | Supported for this checkpoint |
| Correct message identity matters | On 100 paired geometries, permuting sender payload+mask reduces worst 0.7984→0.7041; paired difference CI [0.0530, 0.1364] | Supported for this checkpoint |
| Direct communication-to-sensing residual helps | Removal changes worst by 0.0045; paired difference CI [−1.6×10⁻⁵, 0.0112] includes zero | Not supported as an independent contribution |
| Continuous latent dimensions independently help | Zeroing token values reduces the point estimate by 0.0309, but the paired CI [−0.0070, 0.0695] includes zero | Inconclusive; do not claim independently |
| Sparse token topology carries semantics | Forced silence and sender permutation degrade worst, whereas zero token values retain most of the gain | Supported as a topology/identity mechanism |
| Exact 1 W per-UAV RF feasibility | Maximum balance error \(2.22\times10^{-16}\) W on 100 tests | Supported |
| Full decentralized execution | Multistatic edge selection still uses an environment-level scheduler | Not yet supported |

## Completed result table

All values below use the same source checkpoint. The pre-consensus and final
rows use the same 100 predefined test-bank geometries.

| Variant | steady | weak3 | worst | worst LCB | worst CVaR | feasible | Wilson LCB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Matched pre-consensus policy | 0.8599 | 0.8133 | 0.5429 | 0.4911 | 0.0850 | 0.41 | 0.3325 |
| Projected bid, \(\beta=0\) | 0.9188 | 0.8918 | 0.7011 | 0.6374 | 0.0540 | 0.67 | 0.5891 |
| Intrinsic bid, \(\beta=1\) | 0.9046 | 0.8728 | 0.6844 | 0.6348 | 0.2384 | 0.55 | 0.4679 |
| DCB-MAPPO, \(\beta=0.5\) | **0.9452** | **0.9269** | **0.7984** | **0.7497** | **0.2882** | **0.78** | **0.7050** |
| Deployment floor | 0.80 | 0.70 | 0.60 | — | — | simultaneous | — |

The final row also uses 1280 bit/frame, four active senders, 8 bit/dimension,
100% link delivery, and 1.152 ms mean latency. Across the four-UAV team, mean
communication and sensing powers are 0.994 and 3.006 W, respectively.

The complete matched baseline and ablation table is reported in
`paper/BASELINE_RESULTS.md`. The strongest new structural findings are: forced
silence reaches only 0.7055 worst, removal of capacity-two projection reaches
0.5362, sender permutation reaches 0.7041, and the centralized motion ceiling
reaches 0.9499.

## Submission-critical experiment matrix

| Experiment | Required design | Current state | Priority |
|---|---|---|---|
| Independent training seeds | Train the final DCB-MAPPO architecture from at least 5 random seeds; report mean, standard deviation and confidence intervals | Missing | Critical |
| Untouched final test bank | Generate a second, versioned 100-seed geometry bank after all method choices are frozen | Missing | Critical |
| Strong learning baselines | Standard MAPPO without U2U messages; attention-only MAPPO; sparse-token MAPPO without consensus; independent PPO where applicable | Frozen-policy comparisons complete; independent retraining missing | Critical |
| Component ablations | Remove permutation projection, capacity-two projection, bid inertia, identity indexing, comm-aided sensing and joint-power learning one at a time | Core communication/matching ablations complete; joint-power learning ablation missing | High |
| End-to-end versus frozen intervention | Compare final-layer retraining against the current frozen structural intervention | Missing | Critical |
| Channel stress | Packet loss/SNR thresholds, bandwidth, latency deadline, bit depth and communication-power cap | Missing | High |
| Scalability | At minimum 6-UAV/6-target; preferably 8-UAV/8-target with an approximate projection | Missing | High |
| Scheduler dependence | Replace or decentralize the environment-level multistatic feasibility scheduler, or clearly position it as a common deterministic PHY service | Missing | High |
| Runtime/complexity | Actor latency, communication complexity and projection time versus team size | Missing | High |
| Mechanism figures | Trajectories, target assignments, token masks, power split and failure-tail examples | Missing | High |

## Section outline

1. **Introduction** — identify the coordination failure caused by locally
   ambiguous target responsibilities under costly U2U communication; state the
   need to couple semantic messages, motion and multistatic sensing.
2. **Related Work** — distributed MARL communication; multi-UAV ISAC resource
   allocation; differentiable matching and assignment; risk-aware/worst-user
   optimization. All citations remain placeholders until a verified literature
   search is completed.
3. **System Model** — tracking-free 4-UAV/4-target mission, U2U physical channel,
   bistatic detection model, exact per-UAV RF budget and QoS definitions.
4. **DCB-MAPPO** — target-token encoder, sparse top-2 broadcast,
   identity-consistent graph reconstruction, dual projections, bid inertia,
   MAPPO action/resource heads and training objectives.
5. **Experimental Protocol** — stratified geometry banks, robust checkpoint
   ranking, matched baselines, causal interventions and statistical reporting.
6. **Results** — main test result, bid-mixing study, semantic intervention,
   communication/power accounting, geometric diagnosis and failure tail.
7. **Discussion** — what is learned, what is supplied structurally, remaining
   P0/scheduler dependence, scaling and generalization.
8. **Conclusion** — bounded claim and next technical step.

## Chinese author notes

- 当前应当继续写论文骨架，同时只补“投稿必需实验”，不再开放式改架构。
- 最关键的新增结果不是再提高一次 0.798，而是证明该结果能在最终架构的多次独立训练中稳定复现。
- 论文中不能宣称“16 维连续 Token 都学出了语义”；现有证据只支持“稀疏目标掩码与发送者身份对应关系有用”。
- 若暂不去掉环境内 TX/RX 可行性调度器，题目和贡献必须限定在 distributed motion/resource-intent coordination。
