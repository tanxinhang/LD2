# Sparse U2U Semantic Consensus Bidding for Multi-UAV Integrated Sensing and Communication

> Working manuscript draft. Numerical claims are restricted to completed
> experiments. `[CITATION REQUIRED]` and `[TODO]` markers identify evidence that
> must be supplied before submission.

## Abstract

Distributed multi-unmanned-aerial-vehicle (UAV) integrated sensing and
communication (ISAC) requires each node to decide not only where to move and
how much power to sense with, but also what information is worth transmitting
to its neighbours. Local policies can nevertheless converge to duplicated
target choices, leaving one target persistently under-served even when the
aggregate sensing and communication resources are sufficient. We present a
working distributed consensus-bid MAPPO framework for a tracking-free,
UAV-to-UAV-only ISAC mission. Each UAV encodes every mission-known target into a
learned token, broadcasts only its two strongest claims through a bandwidth-,
delay- and power-constrained physical channel, and reconstructs an
identity-indexed team bid graph from received packets. A hard permutation
projection defines and guides the slow one-UAV-per-target movement commitment, while
a differentiable capacity-two projection represents the two endpoints required
for multistatic sensing. The transmitted bid mixes the projected commitment
with the current intrinsic target preference, allowing assignments to persist
without preventing correction. Communication power and target-wise sensing
power are jointly selected under an exact 1 W RF budget per UAV. On 100
predefined 4-UAV/4-target test-bank geometries, applying the consensus mechanism to a fixed
trained checkpoint increases steady, bottom-three and strict worst-target
detection probabilities from 0.860/0.813/0.543 to 0.945/0.927/0.798. The rate of
episodes simultaneously satisfying 0.80/0.70/0.60 QoS thresholds rises from
41% to 78%, with a one-sided 95% Wilson lower bound of 0.705. However, the
bottom-20% worst-target CVaR remains 0.288, and ablations support the importance
of sparse token identity rather than the independent utility of every latent
payload dimension. These results establish a mechanism-level benefit but also
define the remaining requirements for end-to-end multi-seed validation and a
fully distributed multistatic scheduler.

## 1. Introduction

Multi-UAV ISAC systems can improve spatial coverage by coupling mobile sensing
geometry with cooperative radio processing. The coupling is also a source of
difficulty: every UAV must allocate finite radio power between communication
and sensing, select which targets deserve sensing power, and move in a way that
complements rather than duplicates neighbouring nodes. These decisions must be
made repeatedly as the geometry changes, while exchanging information through
links with finite rate, delay and energy cost. `[CITATION REQUIRED: multi-UAV
ISAC and distributed resource allocation]`

Multi-agent reinforcement learning (MARL) provides a natural tool for this
joint decision problem, but standard centralized-training/decentralized-
execution policies do not by themselves guarantee differentiated target
responsibilities. In our preliminary system, a trained MAPPO policy remained
active but repeatedly directed multiple UAVs towards the same target. The
result was not an absolute lack of sensing or communication power; replacing
only the motion decisions with a one-to-one assignment was sufficient to cross
the requested QoS thresholds. The bottleneck was therefore the failure to turn
local target preferences and received messages into a consistent team
allocation.

Existing learned-communication approaches commonly aggregate neighbour
messages through attention or append a communication cost to the team reward.
Such mechanisms can learn whether a message correlates with return, yet two
receivers may still interpret the same set of sender tokens as different team
assignments. Soft assignment regularizers are particularly vulnerable to a
symmetric stationary point in which every agent assigns similar probability to
every target. `[CITATION REQUIRED: differentiable MARL communication,
attention-based communication and differentiable matching]` A useful
coordination protocol must break this symmetry, preserve message provenance,
and distinguish the one-target semantics of motion from the multiple-endpoint
semantics of multistatic sensing.

We address this problem with distributed consensus bidding. Each node derives a
target-specific latent token from its local actor and transmits only the two
highest target claims. Sender identity and target identity are retained as
physical packet metadata. A receiver places the delivered bids into globally
indexed UAV rows and runs the same deterministic permutation projection as its
peers. The resulting commitment changes the direction of motion, while a
separate capacity-two projection influences target-wise sensing allocation.
The projected decision is mixed with the actor's current intrinsic preference
before the next message is formed. This bid-inertia mechanism prevents a
one-hot assignment from recursively reinforcing itself while retaining enough
memory for stable coordination.

The current study makes the following bounded contributions:

1. We formulate an identity-consistent sparse target-token protocol that lets
   distributed actors reconstruct a common movement bid graph from physically
   delivered U2U packets rather than free neighbour observations.
2. We separate kinematic and sensing coordination through a hard one-to-one
   movement commitment that guides the actor output and a differentiable
   capacity-two multistatic sensing projection.
3. We introduce inertia-mixed bids and couple them to continuous
   communication/sensing power allocation under an exact per-UAV RF constraint.
4. We evaluate the mechanism with strict weakest-target metrics, bootstrap and
   Wilson lower bounds, and packet/power accounting on a stratified 100-scenario
   test bank.

The present experiments establish these contributions as a frozen-policy
structural intervention. End-to-end retraining with independent random seeds is
left as a submission-critical experiment rather than being implied by the
reported result.

## 2. Related Work

### 2.1 Multi-UAV ISAC and cooperative sensing

`[TODO: contrast joint trajectory/resource optimization, multistatic UAV
sensing and learning-based ISAC. Identify whether prior studies assume a ground
fusion centre or unconstrained inter-agent information.]`

### 2.2 Learned communication in MARL

`[TODO: compare continuous message learning, attention-based communication,
discrete/budgeted communication and emergent topology selection. The key gap is
receiver-consistent assignment, not merely message compression.]`

### 2.3 Assignment, optimal transport and risk-aware MARL

`[TODO: position Sinkhorn/optimal-transport assignment, auction or matching
protocols, and worst-user/CVaR objectives. Do not describe the current greedy
PHY scheduler as an optimal or submodular solver.]`

## 3. System Model and Problem Formulation

### 3.1 Scenario

We consider \(K=4\) UAVs operating at a fixed altitude of 20 m in an
\(800\times800\) m region containing \(Q=4\) stationary, mission-known sensing
targets. An episode contains 150 frames with duration \(\Delta t=0.1\) s. Each
UAV has a maximum horizontal speed of 25 m/s. Target tracking and UAV-to-ground
communication are disabled; the only inter-node information path is the
physical U2U channel. Target positions are mission information available to
each UAV, whereas neighbouring UAV state and intent are not provided as a free
side channel.

At frame \(t\), UAV \(k\) selects a joint action

\[
a_{k,t}=\left(\Delta \mathbf p_{k,t},\;\mathbf z_{k,t},\;r_{k,t},\;
\rho_{k,t},\;\mathbf w_{k,t}\right),
\]

where \(\Delta\mathbf p_{k,t}\) is the horizontal displacement command,
\(\mathbf z_{k,t}=\{\mathbf z_{kq,t}\}_{q=1}^{Q}\) contains target tokens,
\(r_{k,t}\) selects a communication precision from
\(\{0,4,8,16,32\}\) bit/dimension, \(\rho_{k,t}\) is the communication-power
fraction, and \(\mathbf w_{k,t}\) is a non-negative target-wise sensing
allocation. Communication and resource decisions are updated every frame,
whereas a movement commitment is held for five frames, giving five causal U2U
exchanges per 0.5 s motion decision.

### 3.2 Joint communication-sensing power

Every UAV has a hard instantaneous RF budget \(P^{\mathrm{tot}}=1\) W. For an
active sender,

\[
P^{\mathrm{comm}}_{k,t}=\rho_{k,t}P^{\mathrm{tot}},\qquad
P^{\mathrm{sen}}_{kq,t}=\left(P^{\mathrm{tot}}-
P^{\mathrm{comm}}_{k,t}\right)
\frac{w_{kq,t}}{\sum_{q'}w_{kq',t}},
\]

with \(0\leq\rho_{k,t}\leq0.5\). If the selected rate is zero, communication
power is set to zero and the entire budget is available for sensing. Thus,

\[
P^{\mathrm{comm}}_{k,t}+\sum_qP^{\mathrm{sen}}_{kq,t}
=P^{\mathrm{tot}}
\]

holds by construction rather than through a soft penalty.

### 3.3 U2U channel and packet cost

Each active UAV broadcasts one sparse target-token packet. With a top-two mask,
16 dimensions per selected target and \(b_{k,t}\) bits per dimension, the
charged packet size is

\[
L_{k,t}=H+2\times16\times b_{k,t},
\]

where \(H=64\) bit is the packet header. Active senders share 100 kHz bandwidth
orthogonally. For sender-receiver distance \(d_{kj,t}\), the link model uses
free-space gain \((\lambda/(4\pi d_{kj,t}))^2\), Shannon serialization rate,
and a 0.2 ms processing delay. A packet is delivered only if its SNR is at least
0 dB and its total latency is at most 5 ms. The most recent delivered token is
retained for up to five frames with an explicit age-of-information field.

### 3.4 Multistatic detection and QoS

For transmitter \(i\), receiver \(j\) and target \(q\), the raw deflection
coefficient is

\[
d^{\mathrm{raw}}_{ijq,t}=
\frac{P^{\mathrm{sen}}_{iq,t}|\alpha_{ijq,t}|^2T_{\mathrm{sym}}MN
G_{\mathrm{tx}}G_{\mathrm{rx}}N_{\mathrm{CPI}}}{\sigma_z^2}.
\]

An OTFS delay-Doppler feasibility gate produces the effective contribution
\(d^{\mathrm{eff}}_{ijq,t}\). Because no ground-report link exists, sensing
quality is not multiplied by a UAV-ground reliability term. The cumulative
deflection of target \(q\) is the sum of selected bistatic contributions,
\(D^*_{q,t}=\sum_{(i,j)\in\mathcal E_{q,t}}d^{\mathrm{eff}}_{ijq,t}\), and the
detection probability is

\[
P^q_{D,t}=Q\!\left(Q^{-1}(P_{FA})-\sqrt{D^*_{q,t}}\right),
\qquad P_{FA}=10^{-3}.
\]

For each episode, the last 20 frames define the steady window. Let
\(\bar P^q_D\) be the mean detection probability of target \(q\) over this
window. We report

\[
P_D^{\mathrm{steady}}=\frac{1}{Q}\sum_q\bar P^q_D,\qquad
P_D^{\mathrm{weak3}}=\frac{1}{3}\sum_{q\in\mathcal B_3}\bar P^q_D,\qquad
P_D^{\mathrm{worst}}=\min_q\bar P^q_D,
\]

where \(\mathcal B_3\) contains the three lowest-quality targets. An episode is
QoS-feasible if these quantities simultaneously exceed 0.80, 0.70 and 0.60.

### 3.5 Optimization objective

The learning problem is to maximize sensing utility and weakest-target quality
while charging packet bits, radio energy and communication delay, subject to
the RF equality constraint and physical packet-delivery conditions. MAPPO uses
a centralized critic during training and a shared local actor during execution.
The final model employs bottleneck-risk advantages that emphasize the bottom
half of target qualities while retaining 25% of the scalar team advantage.
`[TODO: give the exact final composite loss after the final architecture is
retrained; the present checkpoint predates the final consensus intervention.]`

## 4. Distributed Consensus-Bid MAPPO

### 4.1 Target-token actor

The shared actor encodes local target geometry and physically received
sender-target tokens. Each received token contains a 16-dimensional quantized
payload together with sender ID, target ID, rate, latency, SNR and age. Masked
cross-attention conditions the local target entities on valid peer tokens.
Separate heads produce movement preference, target-wise sensing allocation,
communication content, rate and communication power.

### 4.2 Sparse target claims

Let \(b^{\mathrm{int}}_{kq,t}\) denote UAV \(k\)'s intrinsic target preference
before team projection, and let \(c_{kq,t}\) denote its current projected
movement commitment. The bid transmitted in the next round is

\[
b^{\mathrm{tx}}_{kq,t}=\beta b^{\mathrm{int}}_{kq,t}
+(1-\beta)c_{kq,t},\qquad \beta=0.5.
\]

Only the two largest components are transmitted. The first dimension of each
selected target token is reserved for the comparable signed bid
\(2b^{\mathrm{tx}}_{kq,t}-1\); the remaining 15 dimensions stay latent and can
condition sensing. This design gives the graph projection a common numerical
meaning while leaving most token content unconstrained.

### 4.3 Identity-consistent graph reconstruction

Every packet retains sender identity. Receiver \(k\) scatters peer bids into
their global UAV rows and places its locally remembered transmitted bid in row
\(k\), producing \(\mathbf B_t\in\mathbb R^{K\times Q}\). With complete timely
delivery, all UAVs therefore evaluate the same matrix. A small deterministic
lexicographic term resolves exact ties identically at every node.

This construction differs from receiver-specific message aggregation. The
messages are not interpreted as an unordered set of additional target tokens;
they are claims made by specific team members about specific targets.

### 4.4 One-to-one movement projection

For \(K=Q\leq8\), the movement layer enumerates the feasible permutations and
selects

\[
\mathbf C_t=\arg\max_{\mathbf P\in\mathcal P_K}
\langle\mathbf P,\mathbf B_t\rangle,
\]

where \(\mathcal P_K\) is the set of \(K\times K\) permutation matrices. During
training, a Gibbs distribution over permutations supplies a straight-through
soft gradient, while the executed forward value is hard. The row indexed by
the local UAV defines its one-target movement commitment. Let
\(\hat{\mathbf d}_{k,t}\) be the unit direction from UAV \(k\) to the committed
target and \(\boldsymbol\mu^{\mathrm{act}}_{k,t}\) the unconstrained actor
movement mean. The executed latent movement mean is

\[
\boldsymbol\mu_{k,t}=(1-\gamma_{k,t})
\boldsymbol\mu^{\mathrm{act}}_{k,t}+
\gamma_{k,t}\operatorname{atanh}(\hat{\mathbf d}_{k,t}),
\]

where \(\gamma_{k,t}=0.5(\pi^{(1)}_{k,t}-\pi^{(2)}_{k,t})^2\) is a
confidence-gated blend based on the two largest commitment probabilities. A
complete hard consensus has unit margin and therefore uses a 0.5 guidance
coefficient; incomplete communication falls back towards the local actor.
Because all nodes use the same indexed graph and tie-break, no central
assignment message is required for this motion decision. For larger teams the
implementation falls back to a Sinkhorn approximation; its scalability is not
evaluated in the current paper draft.

### 4.5 Capacity-two sensing projection

Multistatic sensing requires two endpoints per target, so reusing a
one-to-one movement assignment would impose the wrong semantics. We instead
project the local and received sensing logits to \(\mathbf X_t\in(0,1)^{K\times
Q}\) with row and column sums equal to two. The box-constrained entropic
projection has the form

\[
X_{kq,t}=\sigma\!\left(B^{\mathrm{sen}}_{kq,t}/\tau+u_k+v_q\right),
\]

where alternating dual updates enforce the marginals. The projected local row
is normalized and blended with the actor's sensing allocation. This layer
represents endpoint demand; a physical feasibility scheduler subsequently
selects valid transmitter-receiver edges inside the policy-constrained
subgraph. This final scheduling step is not yet fully decentralized.

### 4.6 Why bid inertia is required

Pure projected bidding (\(\beta=0\)) rebroadcasts the previous one-hot winner.
The resulting positive feedback can lock the team into an early permutation,
producing high average performance but catastrophic failures on a small subset
of geometries. Pure intrinsic bidding (\(\beta=1\)) can correct errors but gives
up assignment persistence. The hybrid bid preserves half of each signal. The
choice \(\beta=0.5\) was made on the confirmation split before the final
100-scenario evaluation.

## 5. Experimental Protocol

### 5.1 Physical and learning parameters

The carrier frequency is 28 GHz and the sensing bandwidth is 1 MHz. The OTFS
grid contains \(M=64\) delay bins and \(N=16\) Doppler bins with a 64 μs symbol
period and 128 coherent integration frames. Both transmit and receive array
gains are 16 dBi. The final actor selects 8 bit/dimension for every active
sender. Four active broadcasts therefore consume 1280 bit/frame.

`[TODO: add a complete parameter table generated from the resolved final YAML,
including optimizer settings and the provenance of every literature-derived
physical parameter.]`

### 5.2 Geometry splits and checkpoint selection

Scenario seeds are generated offline and stratified by the largest initial
nearest-UAV distance among targets. Training, selection, confirmation, test and
stress splits are disjoint in the seed-bank file. Checkpoints are ranked
lexicographically by: the Wilson lower bound of QoS feasibility, the bootstrap
lower bound of mean worst-target quality, bottom-20% worst-target CVaR, strict
mean worst, trimmed worst, weak3, steady, and finally negative communication
bits. The continuous worst lower bound is the fifth percentile of 2000
non-parametric bootstrap means. The feasible-rate lower bound is the one-sided
95% Wilson bound.

The current final result uses 100 predefined seeds from the existing test split.
Because this test split was observed during architecture development, a second
untouched final bank must be generated before submission.

### 5.3 Compared variants

The current completed comparison uses a common frozen checkpoint:

- matched pre-consensus soft-commitment policy;
- projected-only bidding, \(\beta=0\);
- intrinsic-only bidding, \(\beta=1\);
- hybrid DCB, \(\beta=0.5\).

The final matched structural comparison contains forced silence, removal of the
capacity-two projection, removal of the direct communication-to-sensing
residual, zero token values with the sparse mask retained, joint permutation of
token values and masks across sender identities, and a centralized one-to-one
motion ceiling. All were evaluated on the same 100-scenario test bank. Paired
95% confidence intervals were obtained from 10,000 non-parametric bootstrap
resamples of per-scenario worst-target differences.

`[TODO-CRITICAL: add independently trained final DCB-MAPPO models and standard
MAPPO/attention/no-communication baselines. The frozen variants are mechanism
probes, not sufficient standalone learning baselines.]`

## 6. Results

### 6.1 Main QoS result

| Method | steady \(P_D\) | weak3 \(P_D\) | worst \(P_D\) | worst LCB | worst CVaR | QoS feasible | Wilson LCB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pre-consensus | 0.8599 | 0.8133 | 0.5429 | 0.4911 | 0.0850 | 0.41 | 0.3325 |
| DCB, \(\beta=0.5\) | **0.9452** | **0.9269** | **0.7984** | **0.7497** | **0.2882** | **0.78** | **0.7050** |
| Required floor | 0.80 | 0.70 | 0.60 | — | — | simultaneous | — |

The consensus intervention improves the three mean QoS metrics by 0.0852,
0.1136 and 0.2555, respectively. It also raises simultaneous per-scenario
feasibility by 37 percentage points. Thus the final aggregate metrics exceed
the specified Medium deployment floors, whereas the matched pre-consensus
policy fails the strict worst-target floor.

The gain is associated with better spatial differentiation. The fraction of
unique target choices increases from 0.6705 to 0.7093, the movement collision
frame rate decreases from 0.9243 to 0.8680, the mean nearest-UAV distance falls
from 63.8 to 57.5 m, and the episode-worst nearest-target distance falls from
184.4 to 157.4 m. The collision diagnostic remains high, indicating that the
projection improves target coverage without eliminating all duplicated
instantaneous motion directions.

### 6.2 Bid-mixing ablation

| Bid source | \(\beta\) | steady | weak3 | worst | worst LCB | worst CVaR | feasible |
|---|---:|---:|---:|---:|---:|---:|---:|
| Projected commitment only | 0 | 0.9188 | 0.8918 | 0.7011 | 0.6374 | 0.0540 | 0.67 |
| Hybrid | 0.5 | **0.9452** | **0.9269** | **0.7984** | **0.7497** | **0.2882** | **0.78** |
| Intrinsic preference only | 1 | 0.9046 | 0.8728 | 0.6844 | 0.6348 | 0.2384 | 0.55 |

Both endpoints improve mean worst-target quality over the pre-consensus policy,
but neither matches the hybrid. The projected-only endpoint has an especially
low CVaR despite a mean worst value above 0.70, consistent with assignment
lock-in on a small set of adverse geometries. The intrinsic endpoint has a
healthier tail but lower feasibility and mean performance. Bid inertia provides
the best observed balance between persistence and correction.

### 6.3 Matched structural baselines

| Variant | steady | weak3 | worst | Full − variant worst [95% paired CI] | feasible |
|---|---:|---:|---:|---:|---:|
| Pre-consensus | 0.8599 | 0.8133 | 0.5429 | 0.2555 [0.2112, 0.3011] | 0.41 |
| No U2U communication | 0.9177 | 0.8903 | 0.7055 | 0.0930 [0.0472, 0.1415] | 0.67 |
| No capacity-two projection | 0.8554 | 0.8072 | 0.5362 | 0.2622 [0.2131, 0.3116] | 0.37 |
| No direct communication-to-sensing residual | 0.9435 | 0.9247 | 0.7939 | 0.0045 [−1.6×10⁻⁵, 0.0112] | 0.78 |
| Full DCB-MAPPO | **0.9452** | **0.9269** | **0.7984** | — | **0.78** |
| Centralized movement ceiling | 0.9870 | 0.9827 | 0.9499 | −0.1515 [−0.2152, −0.0899] | 0.92 |

Forced silence assigned the full 4 W team RF budget to sensing but reduced
worst-target quality by 0.0930, with a paired interval excluding zero. Thus the
communication gain was not caused by extra sensing power. Removing the
capacity-two projection produced the largest deployable ablation loss and
increased the movement collision rate from 0.8680 to 0.9567. By contrast,
removing the direct communication-to-sensing residual changed worst by only
0.0045, and the paired interval included zero. The residual was therefore not
an independently supported source of the final gain.

The centralized movement ceiling reached a worst-target value of 0.9499 and a
92% feasible rate, compared with 0.7984 and 78% for the distributed method. The
remaining performance gap was consequently associated primarily with motion
coordination rather than physical sensing feasibility.

### 6.4 What information in a token is useful?

| Intervention | steady | weak3 | worst | Full − intervention worst [95% paired CI] | feasible |
|---|---:|---:|---:|---:|---:|
| Full token | **0.9452** | **0.9269** | **0.7984** | — | **0.78** |
| Zero all token values; retain top-2 mask | 0.9332 | 0.9109 | 0.7675 | 0.0309 [−0.0070, 0.0695] | 0.73 |
| Permute values and masks across sender identities | 0.9137 | 0.8849 | 0.7041 | 0.0943 [0.0530, 0.1364] | 0.68 |

Zeroing all transmitted values while preserving the sparse top-two mask lowered
the point estimate, but its paired interval included zero. The experiment
therefore did not establish an independent contribution from the continuous
token values. In contrast, breaking the correspondence between sender identity
and target claims caused a 0.0943 decrease with a paired interval excluding
zero. The supported interpretation is that sparse target topology and
identity-consistent claims carry the primary communication semantics; learned
continuous values may provide a secondary benefit that remains inconclusive.

### 6.5 Communication and power accounting

The final policy activates all four senders at 8 bit/dimension. Each sender
transmits two 16-dimensional target tokens plus a 64-bit header, giving 320 bit
per broadcast and 1280 bit/frame for the team. Mean packet latency is 1.152 ms,
delivery is 100%, and no deadline violations occur in the 100 evaluated
geometries.

The four-UAV mean RF split is 0.994 W for communication and 3.006 W for sensing,
equivalent to approximately 0.249 and 0.751 W per UAV. The maximum observed
per-UAV power-balance error is \(2.22\times10^{-16}\) W. Hence the QoS gain is
not obtained by violating the stipulated power budget. Communication efficiency
is not yet optimized: the current point deliberately favours reliable
coordination and should be compared with lower-rate or event-triggered variants
only after robustness is confirmed.

### 6.6 Remaining failure tail

Mean QoS alone overstates robustness. Five of the 100 evaluated scenarios have
worst-target \(P_D<0.1\), the minimum is 0.0245, and the bottom-20% CVaR is only
0.288. The method therefore satisfies the requested aggregate deployment
floors but does not guarantee every geometry. Failure examples should be
analysed through initial geometry, assignment lock-in, token delivery and
nearest-target distance before submission. `[TODO: add paired trajectory and
bid-graph visualizations for median and tail scenarios.]`

## 7. Discussion

The main result changes the diagnosis of this system. Communication quantity
was not the binding resource: even forced silence, which returned all radio
power to sensing, underperformed the full method in worst-target quality. The
gain instead required converting delivered target claims into a
receiver-consistent team structure and imposing the correct capacity-two
semantics for multistatic sensing. Removing the direct communication-to-sensing
residual had no independently resolved effect, so the evidence supports
communication-assisted coordination rather than a direct learned sensing-power
correction as the principal mechanism.

The result also clarifies what has not yet been learned. The top-two target mask,
the reserved bid dimension and sender identity impose a lightweight protocol
structure. Zeroing all token values retained most of the benefit and produced a
paired interval that included zero, whereas permuting sender correspondence
caused a clearly resolved loss. A strong paper should therefore describe the method
as structured semantic communication with learned secondary content, not as
fully emergent continuous communication semantics.

Three limitations determine the next experiments. First, the final mechanism
must be retrained end-to-end from multiple random seeds; a single frozen
checkpoint cannot establish optimization stability. Second, the current
multistatic feasibility scheduler has access to environment-level candidate
geometry. Either this scheduler must be replaced by a distributed protocol, or
the paper must treat it explicitly as a common deterministic PHY service and
limit the decentralization claim to motion and resource intent. Third, exact
permutation enumeration scales factorially and is currently bounded to at most
eight agents; approximate assignment quality and runtime must be evaluated for
larger swarms.

Finally, the test distribution has already informed method development. The
reported bootstrap and Wilson bounds quantify variation within this bank but do
not remove model-selection bias. A versioned untouched test bank is therefore
required for the final manuscript.

## 8. Conclusion

This study identifies inconsistent target responsibility, rather than absolute
radio scarcity, as the principal bottleneck in a distributed multi-UAV ISAC
scenario. Sparse identity-preserving U2U claims, dual task-specific projections
and inertia-mixed bidding raise strict weakest-target quality while respecting
an exact 1 W per-UAV communication-sensing budget. On 100 predefined test-bank geometries,
the mechanism exceeds the requested aggregate steady, weak3 and worst-target
floors and nearly doubles the rate of QoS-feasible episodes. The remaining
lower-tail failures, missing independent training runs and residual centralized
PHY scheduling delimit the present claim. Addressing these points is the
shortest path from a successful structural result to a defensible journal
submission.

## Claim-to-evidence checklist

| Manuscript statement | Evidence source | Ready? |
|---|---|---|
| Main 100-seed QoS values | `results/distributed_consensus_hybrid50_bid_frozen_test100/paired_eval.csv` | Yes |
| Matched pre-consensus values | `results/qos_pretrained_soft_gated_move50_test100/paired_eval.csv` | Yes |
| \(\beta\) ablation | `results/distributed_consensus_bid_token_frozen_test100/` and `results/distributed_consensus_intrinsic_bid_frozen_test100/` | Yes |
| Matched communication/matching ablations | `results/baseline_*_test100/` and `paper/BASELINE_RESULTS.md` | Yes for the frozen checkpoint |
| Token interventions | `results/baseline_zero_payload_test100/` and `results/baseline_permute_identity_test100/` | Yes for the frozen checkpoint |
| Exact RF balance | final 100-seed CSV and environment assertion path | Yes |
| End-to-end final-method convergence | none | No |
| Independent training statistics | none | No |
| Literature positioning | citations not yet searched and verified | No |

## Chinese author notes

本稿按照“问题—结构机制—性能证据—失败边界”的算法论文逻辑搭建。摘要中的数字均来自现有文件，但正式投稿前必须用最终架构完成多随机种子训练，并增加一个从未参与架构选择的新测试库。若不替换环境内的多基地配对调度器，则全文只能主张分布式运动与资源意图协同，不能主张完全去中心化执行。
