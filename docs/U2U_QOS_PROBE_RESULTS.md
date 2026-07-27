# U2U QoS-attention probe (Medium)

## Protocol

- Scenario: `bench_medium`, K=4, Q=4, 800 m x 800 m.
- Tracking disabled; mission-known stationary sensing targets.
- UAV-ground communication disabled.
- Free neighbour state disabled.
- Initial physical policy: corrected D1 DAgger checkpoint.
- U2U policy: learned 16-D content, learned silence/4/8/16-bit rate,
  physical delivery, per-sender tokens and target-conditioned cross-attention.
- QoS floors: steady >= 0.80, weak3 >= 0.70, worst >= 0.60.
- Probe training seed: 42; fixed five-seed validation bank used for checkpoint
  screening. A separate 20-seed bank was used for the communication ablation.

This is a mechanism probe, not a final multi-seed paper result.

> Evaluation timing correction (2026-07-20): the original tables below were
> produced before the evaluator matched the training macro interval. They
> retransmitted at every simulator frame and therefore overstate deployed
> traffic by about 5x. The corrected soft-objective probe at the end of this
> document is the current reporting basis.

## Five-seed checkpoint screening

| Checkpoint | steady | weak3 | worst | bit/frame | active sender/frame |
|---|---:|---:|---:|---:|---:|
| D1 start, silence | 0.7039 | 0.6051 | 0.2073 | 0.0 | 0.00 |
| QoS-attention best (continuation update 30) | 0.7126 | 0.6168 | 0.2619 | 509.6 | 3.98 |
| Target-wise continuation update 30 | 0.7154 | 0.6205 | 0.2533 | 512.0 | 4.00 |

The QoS dual variables caused communication to emerge after an initially
silent policy. The learned policy selected almost exclusively the lowest active
rate (4 bit/dimension), but activated nearly every sender because all three QoS
constraints remained violated. Target-wise advantage did not improve the best
worst-target result and was stopped.

## Independent 20-seed causal ablation

The same best actor was evaluated with its learned communication and with all
messages forcibly silenced:

| Mode | steady | weak3 | worst | bit/frame | active sender/frame |
|---|---:|---:|---:|---:|---:|
| Learned communication | 0.6982 | 0.5976 | 0.1765 | 510.7 | 3.99 |
| Forced silence | 0.6958 | 0.5944 | 0.1509 | 0.0 | 0.00 |
| Communication effect | +0.0024 | +0.0033 | **+0.0256** | +510.7 | +3.99 |

The message path therefore has a positive causal effect, strongest on the worst
target, but its present resource efficiency is poor.

## Higher-bit probe

Reinterpreting the same four rate actions as `[0, 8, 16, 32]` increased traffic
from 510.7 to 765.2 bit/frame on the 20-seed bank, while steady/weak3/worst
changed from 0.6982/0.5976/0.1765 to 0.6979/0.5972/0.1770. The change is
negligible relative to the extra traffic. Consequently, the default action set
keeps the low-cost 4/8/16-bit levels and appends an optional 32-bit level:
`[0, 4, 8, 16, 32]`. This lets training select high precision when useful
without forcing every active packet to pay the higher minimum.

## Decision

The learned U2U mechanism is technically valid and scientifically testable, but
the present performance is not deployable and does not meet the requested QoS
floors. The main bottleneck is no longer the link model or PPO probability
bookkeeping; it is weak-target credit assignment and distributed target
allocation. More episodes with the same objective are unlikely to close a
worst-target gap of roughly 0.42.

The next justified research step is an explicit decentralized target-allocation
mechanism (for example, differentiable assignment/auction tokens followed by
cross-attention), evaluated against a no-message and forced-silence ablation.
Only after that mechanism reaches the QoS floors should communication pruning
or a tighter bit budget be optimized for a journal result.

## Allocation-enhancement probes (2026-07-20)

A centralized minimum-distance one-to-one assignment followed by direct target
approach reached **0.9046/0.8728/0.7308** (steady/weak3/worst) on the fixed
five-seed Medium bank. Thus the requested 0.80/0.70/0.60 floors are physically
feasible; the dominant gap is distributed target allocation, not sensing power
or link capacity. A deterministic spatial ordering was less robust on 20
independent seeds: 0.8154/0.7539/0.4135, with only 30% of episodes meeting all
three floors.

| Variant | steady | weak3 | worst | bit/frame | Result |
|---|---:|---:|---:|---:|---|
| Soft balanced responsibility, update 30 | 0.7059 | 0.6079 | 0.2194 | 484.9 | uniform stationary point; reject |
| QoS 8-bit barrier + teacher, update 6 | 0.6958 | 0.5944 | 0.2032 | 768.0 | rate learned; quality degraded |
| Differentiable-message BC | 0.6777 | 0.5702 | 0.3105 | 767.9 | offline 100% assignment, physical domain gap |
| One physical DAgger iteration | 0.6577 | 0.5436 | 0.1884 | 768.0 | real-message assignment only 74.9% |
| Sinkhorn-message BC | 0.6097 | 0.4796 | 0.0770 | 768.0 | offline 99.98%, recurrent domain gap |

The QoS rate barrier solved the consumption-collapse failure: while sensing was
infeasible, deterministic communication moved from silence to the smallest
allowed precision (8 bit/dimension), producing 768 bit/frame for four active
broadcasts. It did not improve sensing by itself, confirming that extra bits
help only after message semantics and closed-loop assignment are reliable.

These allocation modules are retained behind disabled/experimental flags for
reproducibility, but none of their checkpoints is recommended. The current
recommended learned checkpoint remains `seed_42_cont/best_probe.pt`. A
publishable next experiment must train through the recurrent physical channel
(previous-message inbox, quantization, TTL and student-state distribution) from
the beginning; offline assignment accuracy is not a valid selection metric.

## Soft communication encouragement (current default)

The hard 8-bit barrier is now disabled. A successful active sender receives a
small QoS-deficit-gated bonus, identical for every non-zero rate, while bit,
energy and latency penalties remain active. A six-update conservative probe
confirmed that this produces a non-zero sender-attributable reward (about
0.037 per active sampled action) without imposing a minimum payload precision.
It did not improve the best fixed-seed sensing checkpoint in six updates, so it
should be treated as an anti-collapse/exploration mechanism rather than a
solved performance enhancement.

After correcting evaluation to one communication action per five-frame policy
interval, the recommended actor on the independent 20-seed bank obtained
steady/weak3/worst = **0.6946/0.5928/0.1748**, with **99.1 bit per simulator
frame** and **0.774 active senders per simulator frame**. Packet delivery was
100% in this bank. These values remain below the 0.80/0.70/0.60 deployment
floors.

The matched forced-silence intervention preserves the same actor and 189-D
cost-aware observation, changing only every transmitted rate to zero:

| Corrected 20-seed mode | steady | weak3 | worst | bit/frame | active/frame |
|---|---:|---:|---:|---:|---:|
| Learned communication | 0.6946 | 0.5928 | 0.1748 | 99.1 | 0.774 |
| Forced silence | 0.6961 | 0.5948 | 0.1515 | 0.0 | 0.000 |
| Communication effect | -0.0015 | -0.0020 | **+0.0233** | +99.1 | +0.774 |

Communication is therefore causally useful for the weakest target but has not
yet improved the aggregate metrics. The next training runs should select
checkpoints by QoS feasibility/worst-target quality, not communication activity
or steady mean alone.

The implemented selector now ranks checkpoints by `(QoS feasible, worst,
weak3, steady, -bits)`. Training also permits three consecutive silent policy
decisions before applying a sender-specific 0.01-per-decision penalty capped at
0.05. An active rate resets the streak. These changes need a new long-run,
multi-seed training result; historical checkpoints are not retroactively
relabelled as improved.

## Worst/QoS selector plus long-silence penalty probe

A 10-update warm-start probe (`qos_worst_silence_v1`, seed 42, actor learning
rate 1e-4) exercised the new objective and selector. On the fixed five-seed
screening bank, update 1 scored **0.7050/0.6067/0.2624**
(steady/weak3/worst), while update 10 scored 0.700/0.601/0.193. The selector
correctly restored update 1 because its worst-target quality was higher.

On the independent 20-seed bank, the restored checkpoint obtained:

| Mode | steady | weak3 | worst | bit/frame | active/frame |
|---|---:|---:|---:|---:|---:|
| Learned communication | 0.6934 | 0.5912 | 0.1759 | 101.1 | 0.790 |
| Forced silence | 0.7007 | 0.6009 | 0.1708 | 0.0 | 0.000 |
| Communication effect | -0.0073 | -0.0097 | +0.0050 | +101.1 | +0.790 |
| Required Medium floor | 0.8000 | 0.7000 | 0.6000 | - | - |

The run is not QoS-feasible. Long-silence penalties were almost zero (rollout
means 0 to 7.4e-5, maximum observed streak 7 decisions), because the warm-start
actor already communicated frequently. Thus the new liveness term is working
but does not address the dominant failure: learned message content and movement
coordination provide only a small worst-target benefit and slightly reduce the
aggregate sensing metrics.

## Increased-rate probes

The rate-neutral participation bonus did not move the deterministic actor away
from 4 bit/dimension. A rate-head-only QoS-infeasible auxiliary optimizer was
therefore tested at matched 8- and 16-bit targets. Both retained the same
physical bit/energy/delay charges and 100% packet delivery.

| Rate target | steady | weak3 | worst | bit/frame | deterministic rate |
|---|---:|---:|---:|---:|---:|
| Existing low-rate actor | 0.6934 | 0.5912 | 0.1759 | 101.1 | mostly 4 bit |
| 8-bit target | 0.6903 | 0.5871 | 0.1762 | 153.6 | 100% 8 bit |
| 16-bit target | 0.6889 | 0.5852 | 0.1741 | 256.0 | 100% 16 bit |

The 8-bit point raises communication by 52% with essentially unchanged worst
(+0.0003), whereas 16 bit raises traffic by 153% and degrades all three sensing
metrics. The default target is therefore 8 bit; 16 bit is retained only as a
high-communication ablation. Neither point meets the Medium QoS floors.

## Stepwise root-cause diagnosis (fixed 20-seed bank)

The current 8-bit actor was replayed on seeds 30001--30020 with one factor
changed at a time. All communication interventions preserve the actor,
movement and role outputs.

| Message intervention | steady | weak3 | worst | full | bit/frame |
|---|---:|---:|---:|---:|---:|
| Learned payload | 0.6903 | 0.5871 | 0.1762 | 0.5501 | 153.6 |
| Zero 16-D payload | 0.6902 | 0.5869 | 0.1748 | 0.5490 | 153.6 |
| Permute payloads between UAVs | 0.6919 | 0.5892 | 0.1760 | 0.5511 | 153.6 |
| Force every sender silent | 0.6911 | 0.5881 | 0.1336 | 0.5734 | 0.0 |

Zeroing or permuting the learned payload has a negligible effect. The 16-D
content therefore does not yet encode a useful distributed assignment
protocol. Removing the entire packet does reduce worst by 0.0426, indicating
that the delivered token metadata (sender/rate/delay/SNR/AoI) helps fairness,
but it is not sufficient. The 100% delivery rate rules out packet loss as the
dominant bottleneck.

The decisive intervention replaced only the actor's movement vectors with a
centralized minimum-distance one-to-one assignment and direct approach. The
roles and learned U2U communication were left unchanged.

| Movement controller | steady | weak3 | worst | mean nearest distance | episode-worst nearest distance |
|---|---:|---:|---:|---:|---:|
| Learned MAPPO movement | 0.6903 | 0.5871 | 0.1762 | 89.8 m | 241.1 m |
| Central one-to-one movement | **0.9292** | **0.9055** | **0.7243** | **20.1 m** | **60.4 m** |
| Required Medium floor | 0.8000 | 0.7000 | 0.6000 | - | - |

All three QoS floors are exceeded on the full 20-seed bank. Hence the sensing
physics, UAV mobility envelope and scenario are feasible. The learned actor,
not the physical environment, creates the performance ceiling.

Direction-based diagnostics explain the learned failure. The actor was never
idle, yet only covered 0.6729 of the four target slots per decision (2.69
unique targets on average). At least two UAVs selected the same target in
92.5% of decision frames, and the inferred target choice matched the
minimum-distance Hungarian assignment only 37.5% of the time. This is wasted
active control rather than conservative resource use.

A three-update `target_wise` advantage probe did not repair the allocation:
steady/weak3/worst remained 0.6892/0.5856/0.1776, collision frames increased to
93.7%, and the episode-worst nearest distance remained 240.7 m. Soft
distance-weighted credit is therefore insufficient; mutual exclusion and a
persistent decentralized target commitment must be represented explicitly.

Training diagnostics show a secondary, but not dominant, stability issue.
Warm-start files contain the actor only, so the first update starts with return
161.4 versus critic value -0.61 and critic loss 134.4. By update three the
critic reaches value 207.6 for return 247.7 and PPO KL remains below 0.003.
Moreover, a previous ten-update scalar continuation reduced critic loss while
its five-seed worst score fell from 0.2624 to 0.1932. Thus the numerical PPO
trust region is functioning, but the shared scalar team reward permits average
return improvement while one target is abandoned. The five-seed checkpoint
score (0.2522 worst) also overestimates the independent 20-seed result (0.1762),
so future selection must use a larger robust validation bank.

The evidence ranks the current bottlenecks as follows:

1. distributed one-to-one target allocation and persistent commitment;
2. message semantics/closed-loop use of received payloads;
3. per-agent credit assignment for the weakest target;
4. actor-only warm-start and small validation-bank variance.

Increasing payload precision is not on the critical path. The next justified
model should communicate learned assignment bids/intent without hard-coding
their semantic meaning, enforce differentiable collision/exclusivity pressure,
and retain the centralized assignment only as an oracle upper bound.

## Latent commitment probes

An end-to-end experimental branch was added after the root-cause diagnosis.
Each UAV produces a low-temperature target commitment from its local target
entities and physically received 16-D latent messages. A straight-through hard
choice can directly guide movement. Centralized training applies team coverage,
temporal commitment and delayed straight-through 8-bit message losses; execution
still has no global state or centralized Hungarian solver. All flags remain off
in the default configuration.

| 20-seed variant | steady | weak3 | worst | movement collision rate |
|---|---:|---:|---:|---:|
| Current 8-bit baseline | 0.6903 | 0.5871 | 0.1762 | 0.925 |
| Unsupervised commitment, 3 updates | 0.6743 | 0.5658 | 0.1815 | 0.955 |
| Local reconstructed-bid Sinkhorn | 0.5913 | 0.4551 | 0.0779 | 0.943 |
| CTDE teacher, joint PPO | 0.6463 | 0.5284 | 0.1477 | 0.950 |
| Semantic pretrain, QoS-selected deploy | **0.6949** | **0.5932** | **0.1781** | 0.930 |
| Semantic pretrain final diagnostic | 0.6842 | 0.5789 | 0.1746 | 0.935 |

The initial 0.85 movement blend caused PPO KL to spike to 0.96 and was rejected.
A conservative 0.25 blend kept KL manageable. Straight-through team balance
removed the exact zero-gradient failure of uniform soft assignments, but the
unsupervised and local Sinkhorn variants still failed to produce consistent
choices across receivers.

The two-stage CTDE semantic pretraining result is diagnostically important.
With PPO disabled and movement blend limited to 0.05, assignment accuracy rose
from chance to approximately 70%, 80% and 83% over three updates, while the
message decoding loss fell below its `ln(4)` random baseline. Thus the latent
payload can learn assignment-related information. However, higher local label
accuracy did not reduce physical team collisions or improve robust worst-target
quality. Different UAVs reconstruct inconsistent team decisions from their own
delayed inbox views.

This rules out additional bit precision or longer training of the same
single-broadcast architecture as the next step. A subsequent method needs an
explicit consensus process across receiver views, such as multi-round learned
auction/acknowledgement messages or a shared decentralized slot protocol, while
keeping message values latent and charging every communication round.

## Multi-round joint-power correction

The subsequent implementation removes the single-packet/fixed-power/single-
commitment assumptions.  Each simulator frame is now a communication round;
MAPPO selects continuous communication power and a sensing-power distribution
over all four targets under the exact per-UAV constraint
`P_comm + sum_q P_sense,q = 0.2751 W`. The payload remains latent.

Using the previous 8-bit policy as a movement/message warm start and neutral
new resource heads, the 20-seed result is 0.7049 steady, 0.6065 weak-3 and
0.2274 worst, versus 0.6903/0.5871/0.1762 for the corrected-timing baseline.
Traffic is 768 bits/frame (four active 8-bit senders), delivery is 1.0 and mean
latency is 0.801 ms.  This is a real tail improvement, but it remains well below
the 0.80/0.70/0.60 deployment floors.  Movement collision remains 94.1% and the
episode-worst nearest-target distance is 202.3 m, so joint power allocation
does not remove the dominant spatial coordination bottleneck.

The next implementation replaced the single 16-D aggregate payload with four
16-D target-entity tokens and connected all received sender-target tokens to
the actor's masked cross-attention. After one 2,048-frame warm-start update,
the independent 20-seed bank produced 0.7175/0.6233/0.2086
(steady/weak3/worst), versus 0.7163/0.6218/0.2112 for the matched aggregate-
token joint-power run. Payload cost increased from 768 to 2,304 bits/frame and
latency from about 0.80 to 1.97 ms, with 100% delivery. Thus the token plumbing
is functional, but one update is not enough for the newly initialized target-
token projection to improve worst-target coordination.

## Two-round decentralized movement commitment

The next scheduling probe decoupled fast token/resource decisions from slower
motion: communication and joint ISAC power remain frame-level, but each UAV
holds its movement for two frames. This provides two causally ordered token
exchanges per movement commitment without adding a central coordinator.

On the identical target-token checkpoint, changing only deployment timing had
almost no effect: the 20-seed steady/weak3/worst values changed from
0.71746/0.62328/0.20865 to 0.71750/0.62333/0.20904. Collision frames decreased
from 94.60% to 93.67%. After one 2,048-frame PPO adaptation rollout, the same
bank reached 0.71793/0.62390/0.21332 and 92.57% collision frames. Traffic stayed
at 2,304 bits/frame, delivery at 1.0 and mean latency at 1.97 ms because only
movement timing changed.

The PPO pre-update consistency error was 8e-6, below the 1e-4 validity bound.
Thus the multi-round implementation is correct and slightly improves tail
quality, but it still fails all deployment floors. Most importantly, a
two-percentage-point collision reduction leaves 92.6% of frames with duplicate
target choices. The evidence supports retaining multi-round communication as
infrastructure, while the next model change must make the rounds perform an
explicit distributed agreement operation (e.g. recurrent proposal/response
tokens or mutual-exclusion attention), not merely hold movement longer.

## Proposal/response agreement probe

A fully decentralized experimental branch then added: (i) a local two-round
phase input, (ii) response-token generation after proposal-token attention,
(iii) a shared latent-token claim decoder, (iv) the locally known UAV ID as a
symmetry-breaking input, and (v) a movement-frame-only unlabeled exclusivity
loss. The 64-D differentiable auxiliary inbox was corrected to preserve all
four target tokens and their physical metadata. PPO old/new log-probability
agreement remained within 8e-6.

| Independent 20-seed variant | steady | weak3 | worst | movement collision |
|---|---:|---:|---:|---:|
| Two-round target-token baseline | **0.7179** | **0.6239** | **0.2133** | **0.9257** |
| Initial proposal/response head | 0.7038 | 0.6050 | 0.1911 | 0.9637 |
| Shared own/peer claim scale | 0.7177 | 0.6236 | 0.1985 | 0.9360 |
| Shared claim + local UAV ID | 0.7166 | 0.6222 | 0.1979 | 0.9507 |

The internal allocation collision diagnostic improved from 98.3% in the first
version to 91.7% with identity symmetry breaking, proving that the agreement
auxiliary changed local responsibilities. However, responsibility entropy
remained approximately `ln(4)`, and the small 0.05 movement residual did not
turn that internal change into robust spatial separation. Five-seed evaluation
was misleadingly optimistic (up to 0.747/0.662/0.380), while the independent
bank rejected every agreement checkpoint.

Therefore this branch is retained as a research ablation, not selected for
deployment. More continuous attention layers or longer training of this soft
assignment objective are not justified by the evidence. A subsequent method
would need a genuinely discrete distributed reservation/acknowledgement state
or a recurrent finite-state handshake whose successful reservation directly
gates movement, with checkpoint selection performed on the 20-seed QoS bank.

## Artifacts

- Initial QoS probe: `results/u2u_qos_attention_probe/seed_42/`
- Best scalar continuation: `results/u2u_qos_attention_probe/seed_42_cont/`
- Target-wise branch: `results/u2u_qos_attention_probe/target_wise/`
- Failed soft-allocation branch: `results/u2u_qos_attention_probe/allocation/`
- QoS rate/teacher probes: `hungarian_teacher_v2/`, `hungarian_teacher_v3/`,
  `hungarian_teacher_v4/`
- Offline/physical distillation probes: `hungarian_bc/`,
  `hungarian_bc_dagger/`, `sinkhorn_bc/`
- Soft encouragement probe: `soft_encouragement/`
- Timing-corrected 20-seed evaluation: `soft_encouragement_macro_eval/`
- Matched forced-silence evaluation:
  `soft_encouragement_silence_macro_eval/`
- Worst/QoS + silence-penalty probe: `qos_worst_silence_v1/`
- Its matched forced-silence evaluation:
  `qos_worst_silence_v1_forced_silence/`
- Rate probes: `qos_soft_rate16_v1/`, `qos_rate16_aux_v1/`,
  `qos_rate8_head_v1/`, `qos_rate16_head_v1/`
- Stepwise message interventions: `results/u2u_stepwise/step1_*`
- Central assignment and allocation diagnostics:
  `results/u2u_stepwise/step2_central_assignment/`,
  `results/u2u_stepwise/step3_*_diagnostics/`
- Target-wise credit probe:
  `results/u2u_stepwise/step4_targetwise_3updates/`
- Latent commitment probes and diagnostics:
  `results/u2u_stepwise/step5_commitment_*`
- Two-round timing-only evaluation:
  `results/u2u_multiround_m2_evalonly/`
- Two-round one-rollout adaptation:
  `results/u2u_multiround_m2_smoke/`
- Proposal/response agreement probes:
  `results/u2u_round_agreement_smoke/`,
  `results/u2u_round_agreement_shared_claim/`,
  `results/u2u_round_agreement_identity/`
