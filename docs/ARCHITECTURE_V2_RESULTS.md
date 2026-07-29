# Architecture V2: distributed set-based ISAC policy

The method-level prior-art boundary and the evidence-based decision to explore
role-asymmetric bistatic hyperedge negotiation are documented separately in
[`NOVELTY_AND_SCOPE_AUDIT.md`](NOVELTY_AND_SCOPE_AUDIT.md).

## Why the architecture changed

The previous actor contained fixed-cardinality output parameters and could
learn communication that did not reliably affect deployed movement. The main
critic was also a fixed-width MLP. Architecture V2 replaces those paths rather
than stacking another reward term on the legacy actor.

## Deployed actor

For each UAV, the shared actor uses only its local state, local target set,
local detection history, and physically delivered U2U tokens.

1. A shared per-target scorer produces a bounded correction to a
   QoS-deficit/distance prior.
2. Target tokens reserve one comparable bid coordinate; the remaining
   coordinates remain learned latent content.
3. Each receiver reconstructs the same delayed sparse bid set from its local
   sent-token memory and U2U inbox. No fixed-length UAV one-hot is consumed.
4. A row-permutation-equivariant assignment produces one movement target per
   UAV. Exact assignment is used only up to six agents; larger teams use the
   Sinkhorn fallback.
5. A separate capacity-two transport projection controls multistatic sensing
   endpoints. Movement and sensing therefore no longer overload one head.
6. Movement is target-conditioned and kinematically constrained: the learned
   radial component cannot point away from the committed target, while a
   bounded tangential component can create sensing baseline.
7. Communication rate uses a set-pooled learned head plus a local QoS-crisis
   prior. This removes the stochastic-training/greedy-evaluation silence tie.
8. One UAV still has an exact 1 W joint communication-plus-sensing budget.

All target-dependent neural parameters are shared across targets. The same
actor state dictionary therefore loads at different K/Q values.

## CTDE critic

The main MAPPO value model is now a set critic:

- shared UAV encoder plus attention pooling;
- shared target encoder plus attention pooling;
- scalar invariant team value;
- shared equivariant per-target values for bottleneck advantages;
- calibrated binary QoS-violation auxiliary.

The uncalibrated quantile/CVaR auxiliary remains disabled. Central information
is used only during training; execution remains local and distributed.

## Verified gates

- Actor parameter keys and tensor shapes are identical for 4/4 and 6/6.
- Target permutation permutes target outputs and leaves team outputs unchanged.
- UAV permutation leaves the team critic value unchanged.
- Communication, sensing, movement, and value paths have finite gradients.
- PPO rollout/recomputed log probabilities agree within `8e-6`.
- 358 regression tests pass (the Windows MKL eigensolver test is run in a
  separate process because the combined process can abort inside native code).
- A 4/4 V2 checkpoint loads directly into the 6/6 actor and critic.

## Bounded experimental results

The most informative comparison uses the same 20 fixed seeds.

| Variant | steady | weak3 | worst | feasible | move collision | unique target |
|---|---:|---:|---:|---:|---:|---:|
| Unbounded PPO, 40 updates | 0.600 | 0.485 | 0.246 | 0.25 | 0.867 | 0.717 |
| Frozen V2 prior, distance weight 2.0 | **0.804** | **0.740** | **0.513** | **0.50** | **0.226** | **0.939** |
| Protected PPO, 10 updates | 0.772 | 0.696 | 0.401 | 0.40 | 0.233 | 0.937 |

The frozen V2 architecture meets the Medium steady and weak3 requirements but
not the required mean worst of 0.60. Its trimmed worst is 0.620, while strict
worst remains 0.513 and the worst lower confidence bound is 0.361.

Resource audit for the frozen V2 gate:

- 768 bit/frame;
- four active senders;
- system communication power 1.0 W/frame;
- system sensing power 3.0 W/frame;
- maximum per-UAV budget error `4.44e-16` W.

## Initial training decision

Ordinary end-to-end PPO fine-tuning is rejected for this architecture: both
the full-rate and protected-rate runs degrade the stronger physical-consensus
initial policy. The next training stage must preserve the prior explicitly,
for example with a matching/weak-target teacher, residual KL constraint, and
checkpoint selection on worst/QoS feasibility. Until that stage passes a
fixed-bank gate, the frozen V2 policy is the correct architecture checkpoint,
not the trained checkpoint.

## Residual controllability audit

Before adding another policy head, two common-random-number interventions were
run with the frozen V2 policy.

### Movement residual

Ten independent episodes (150 interventions) enumerated one UAV's target
direction while holding communication, sensing power, other UAV movements and
simulator randomness fixed. The mean actual-to-oracle strict-worst gap was
`0.0126` with a 95% cluster interval of `[0.0013, 0.0273]`; only `4.7%` of
interventions exceeded `0.02`. The pre-registered controllability gate failed.
No movement residual value head is justified.

### Sensing-power residual

The initial sensing experiment forced every candidate to redirect 50% of one
UAV's sensing mass. It did not include no-op and therefore measured candidate
spread rather than causal gain over the deployed policy. That preliminary
gate is invalid and must not be reported as algorithm evidence.

The corrected audit explicitly includes the unmodified sensing allocation.
A residual is admissible only if its rollout does not reduce either steady
mean or weak3 relative to no-op. With a 10-frame horizon, 10 independent
episodes and 150 interventions:

- no-op was the safe optimum in `82%` of intervention states;
- the admissible actual-to-oracle strict-worst gap was `0.0281`;
- its 95% cluster interval was `[0.0082, 0.0550]`;
- only `14%` of interventions offered more than `0.02`;
- the formal residual controllability gate failed.

A two-seed closed-loop smoke test also showed the temporal limitation. A
one-step greedy verified residual changed steady/weak3/worst from
`0.6183/0.4911/0.0566` to `0.6087/0.4783/0.0477`. A 10-frame commitment
changed them to `0.6140/0.4854/0.0648`: the worst gain was small and still
traded away the deployment constraints.

### Consequence

The current bottleneck is not a missing local MAPPO head. Local movement and
single-UAV sensing residuals do not have a reliable constraint-preserving
control margin. The next architecture experiment must act on the coupled
object that produces strict worst: a temporally committed multi-UAV
endpoint/evidence assignment. It must compare against explicit no-op, preserve
the frozen V2 policy as the safety baseline, and pass a paired episode-level
worst/QoS gate before any trainable module is admitted.

## Physical upper-bound decomposition

The preceding local tests do not imply that the geometry is infeasible. A
same-geometry physical oracle was therefore evaluated at the final frame of
each of 10 independent episodes. It preserves the observed geometry, channel,
1 W per-UAV RF budget and communication reserve. It alternates an exact MILP
max-min pair update with an LP max-min sensing-power update. This is a
centralized diagnostic upper benchmark, not a deployable result.

| Same-frame controller | worst | gap vs deployed | QoS feasible |
|---|---:|---:|---:|
| Deployed V2 | 0.399 | -- | -- |
| Pair only, deployed power fixed | 0.881 | 0.482 | 0.80 |
| Power only, deployed pairs fixed | 0.519 | 0.120 | 0.40 |
| Joint pair + power, single role | 0.937 | 0.538 | 1.00 |
| Joint pair + power, full duplex | 0.987 | 0.588 | 1.00 |

The pair-only gap has a 95% episode-bootstrap interval of
`[0.240, 0.726]`. The remaining joint-over-best-isolated gain is `0.056`
with interval `[0.0001, 0.1123]`. Full duplex adds `0.050` over the
single-role oracle, but its interval lower bound is effectively zero, so the
duplex architecture gate fails.

The existing evidence-fusion audit also reports zero strict-worst gain from
unconditional centralized receiver fusion over the best local receiver.
Consequently:

1. the scenario is physically feasible at the current geometry;
2. centralized evidence summation and duplex operation are not the primary
   missing mechanisms;
3. the dominant failure is the split between distributed target commitments,
   heuristic P0 pairing and independently emitted sensing power.

The next architecture should replace direct independent target weights with an
unrolled, permutation-equivariant max-min pair/power negotiation. U2U tokens
should carry locally computed endpoint bids and target scarcity prices, while
each round applies explicit single-role, target-capacity and 1 W projections.
The frozen V2 movement controller remains the safety foundation. Oracle
distillation may initialize the negotiation, but admission still depends on
paired worst/QoS improvement under unseen seeds and communication
perturbations.

## Deterministic deficit-aware max-min pairing

The first coupled intervention replaced the average-utility greedy P0 pairing
with an exact single-role MILP. Its objective is lexicographic:

1. maximize instantaneous minimum target detection;
2. maximize QoS-capped target coverage, weighted by each target's historical
   detection deficit;
3. apply a deterministic total-gain/index tie break.

The solver retains the distributed hard Top-2 commitment graph and enforces
one role per UAV, target cardinality and receiver-capacity constraints. A
plain instantaneous max-min objective was rejected because equivalent MILP
solutions produced temporally unstable trajectories. Fixed five- and
ten-frame assignment holds were also rejected because they reduced worst
detection. The deficit-weighted objective is deterministic across repeated
runs.

Formal results on the same 20-seed test bank:

| Variant | steady | weak3 | worst | trimmed worst | worst LCB | CVaR | feasible |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen V2 + greedy P0 | 0.804 | 0.740 | 0.513 | 0.620 | 0.361 | 0.023 | 0.50 |
| Frozen V2 + deficit max-min P0 | **0.890** | **0.854** | **0.679** | **0.787** | **0.554** | **0.053** | **0.70** |

The new deployable controller therefore satisfies the requested mean
thresholds (`steady >= 0.80`, `weak3 >= 0.70`, `worst >= 0.60`) while
preserving 768 bit/frame and the exact 1 W per-UAV RF budget. Its mean solve
time is 1.40 ms/frame. It is not yet tail-robust: the strict-worst CVaR and
lower confidence bound remain below 0.60.

An earlier higher-scoring max-min run is excluded from formal evidence. It
depended on an unspecified equivalent MILP solution and was not reproducible
after deterministic tie breaking.

## Cross-timescale interaction audit

Tail diagnosis on the deployable controller found that episode worst has
correlation `-0.825` with the farthest target's nearest-UAV distance, `+0.629`
with hard commitment coverage, and `+0.663` with P0 target coverage. Four
factorial interventions then isolated the coupling between slow movement
intent and the fast pairing graph.

| 10-seed variant | steady | weak3 | worst | worst LCB | CVaR | feasible |
|---|---:|---:|---:|---:|---:|---:|
| Hard commitments + deployed movement | 0.893 | 0.858 | 0.678 | 0.499 | 0.063 | 0.80 |
| Soft commitments only | 0.871 | 0.828 | 0.577 | 0.401 | 0.060 | 0.60 |
| QoS movement teacher only | 0.824 | 0.765 | 0.444 | -- | 0.028 | 0.40 |
| QoS movement teacher + soft commitments | **0.952** | **0.936** | **0.829** | **0.691** | **0.431** | **0.80** |

Neither component is useful alone. Soft commitments fill graph holes but
destabilize already-correct pairings and increase solve time by roughly one
order of magnitude. The movement teacher shortens distance but invalidates the
frozen actor's hard communication commitments. Together, the soft graph lets
the pairing layer follow the teacher's changed intent, producing a strong
non-additive interaction.

The combined non-deployable upper bound was confirmed on all 20 seeds:

| Metric | Deployable hard graph | Teacher + soft graph upper bound |
|---|---:|---:|
| steady | 0.890 | **0.971** |
| weak3 | 0.854 | **0.961** |
| worst | 0.679 | **0.894** |
| trimmed worst | 0.787 | **0.941** |
| worst LCB | 0.554 | **0.818** |
| worst CVaR | 0.053 | **0.577** |
| QoS feasible rate | 0.70 | **0.90** |
| Wilson feasible LCB | 0.516 | **0.738** |
| solve time/frame | **1.40 ms** | 11.01 ms |

The paired gains are `+0.080` steady (95% bootstrap interval
`[0.019, 0.147]`), `+0.107` weak3 (`[0.025, 0.196]`) and `+0.215` worst
(`[0.039, 0.393]`). The four original disaster seeds all improve, but several
easy seeds regress. The result is therefore a causal architecture upper bound,
not a deployable score.

## Revised architecture direction

The remaining gap is no longer attributed to a larger MAPPO backbone or an
additional local residual head. It is a cross-timescale consistency problem:
the slow movement layer changes target intent while the fast communication and
pairing layers continue to act on stale hard commitments.

The next candidate should be a distributed persistent-intent negotiation:

1. each UAV token carries a target intention, endpoint role, confidence and
   remaining commitment horizon;
2. intention changes use a bounded handover protocol rather than an immediate
   hard Top-2 replacement;
3. the receiver keeps a sparse primary graph and opens fallback edges only
   when a locally observable deficit/timeout event occurs;
4. CTDE distills the QoS teacher's intention and handover labels, while the
   deployed policy uses only local state and delivered tokens;
5. admission requires paired improvement in worst, CVaR and feasible-rate LCB,
   with no significant steady/weak3 loss and solve time restored near the
   sparse 1.40 ms/frame baseline.

This direction has an explicit inferential chain: measured tail failures imply
distance and graph holes; isolated interventions fail; their interaction
succeeds; therefore the missing mechanism is synchronization between intent
and graph state. It is a mechanism contribution rather than another generic
attention block or reward-weight change.

## Persistent-intent admission experiments

The upper-bound interaction motivated a deployable persistent-intent
prototype. The implementation was opt-in and preserved the formal controller
by default. It added three auditable choices:

- build the P0 commitment graph from transmitted sparse Token masks instead of
  inferring another Top-2 variable from sensing power;
- retain a local intention for a minimum hold;
- use a bounded old/new union during target handover.

Frozen-policy gates rejected every static protocol hot-swap on the first two
test seeds:

| Frozen intervention | steady | weak3 | worst |
|---|---:|---:|---:|
| Current sensing-power hard graph | 0.756 | 0.675 | 0.398 |
| Existing Token mask as graph source | 0.658 | 0.544 | 0.142 |
| Token mask + 5-frame hold + 2-frame handover | 0.660 | 0.546 | 0.143 |
| Sensing intent + 5-frame hold + 2-frame handover | 0.799 | 0.732 | 0.249 |
| Sensing intent + 2-frame handover only | 0.640 | 0.537 | 0.097 |

The existing Token mask and sensing Top-2 appear close in aggregate
(`0.983` Jaccard, `0.976` exact-match rate and `0.947` sensing-power mass),
but the small asynchronous fraction changes the deficit-aware scheduler's
history and causes a closed-loop collapse. Forcing exact sensing-aligned Token
masks also failed (`worst=0.122`) because it changed the message distribution
seen by the frozen receiver. Persistence cannot be introduced as an
environment-side rule or a zero-shot protocol change.

### Direct causal-bid distillation

The existing teacher auxiliary originally supervised the post-matching team
assignment. Its NLL was `7.7--10.2`, indicating saturated credit through the
hard matching result. Architecture V2 slow-intent parameters were also absent
from the isolated auxiliary optimizer.

The corrected training path:

1. exposes the non-detached local comparable bid logits that are actually
   encoded in the Token header;
2. directly supervises those logits with the QoS bistatic teacher;
3. adds only V2 assignment/movement coordination heads to the auxiliary
   optimizer, leaving the target encoder, sensing head, main PPO and physical
   prior frozen;
4. applies teacher gradients only to rollout states below the `0.60` crisis
   floor;
5. selects checkpoints on ten stratified seeds.

The direct loss falls to `1.25--1.31`, with teacher classification accuracy
`0.57--0.61`. Formal 20-test-seed results after three updates are:

| Controller | steady | weak3 | worst | trimmed | worst LCB | CVaR | feasible |
|---|---:|---:|---:|---:|---:|---:|---:|
| Deployable deficit max-min baseline | 0.890 | 0.854 | **0.679** | 0.787 | 0.554 | 0.053 | **0.70** |
| Crisis-gated direct-bid distillation | **0.905** | **0.873** | 0.679 | **0.814** | **0.565** | **0.226** | 0.55 |

Communication remains 768 bit/frame and solve time remains about
`1.34 ms/frame`. The paired mean-worst change is `-0.0005` with a 95%
bootstrap interval of `[-0.134, 0.131]`; it is not a significant average
improvement. The lower tail becomes smoother, but more episodes remain just
below the hard `0.60` threshold, reducing feasible rate.

### Admission decision

The deterministic hard-graph deficit max-min controller remains the deployment
checkpoint because it has the better QoS feasible rate and equal mean worst.
Crisis-gated direct-bid distillation is retained as a tail-shaping ablation,
not promoted as the final method.

The next trainable mechanism must predict whether a proposed intention change
will cross the QoS boundary, not merely whether the current state is in crisis.
That is a narrower causal question: supervise the bid update only when a paired
teacher intervention has positive constraint-preserving value. This follows
the measured failure mode and avoids another unconditional teacher or static
handover rule.

## ADMN-inspired modular-coordination screen

ADMN's useful hypothesis is that full parameter sharing can suppress
state-dependent agent diversity. It was not copied directly. The experimental
V2 variant:

- retains one shared, decentralized actor and all physical U2U constraints;
- removes ADMN's absolute agent-identity input;
- routes three shared coordination experts from a permutation-invariant
  summary of the UAV's local target set and local QoS deficits;
- applies the modular residual only to slow target bids and movement, leaving
  sensing power, communication rate and the max-min scheduler unchanged;
- uses batch-level route balance plus per-UAV specialization instead of
  ADMN's observation-transition mutual-information reward.

The router is uniform at initialization, so every historical parameter and
all actor outputs are identical to the non-modular V2 model for the same seed.
This requirement exposed and removed an initial false positive: constructing
the optional modules in the middle of the actor changed random-number
consumption, producing an apparent `worst=0.763` that was unrelated to the
near-zero modular residual. Optional experts are now registered after every
historical module under an isolated RNG scope; common tensors are bitwise
identical.

Fair ten-test-seed results under the same three-update crisis direct-bid
protocol:

| Variant | steady | weak3 | worst | LCB | CVaR | feasible |
|---|---:|---:|---:|---:|---:|---:|
| Crisis direct-bid | **0.9101** | **0.8802** | **0.6842** | 0.5138 | 0.2454 | 0.60 |
| Identity-free modular route | 0.9089 | 0.8785 | 0.6795 | 0.5099 | 0.2453 | 0.60 |
| Sharper/high-LR activation probe | 0.9084 | 0.8779 | 0.6773 | **0.5177** | **0.2596** | 0.60 |

The ordinary modular route remained uniform (`normalized entropy=1.000`,
expert-usage span below `0.0004`, residual norm below `4e-6`). Giving only the
new router/experts a 20x auxiliary learning rate and reducing routing
temperature increased the usage span to `0.0256`, but entropy remained
`0.99946`, residual norm remained below `3e-4`, and worst detection regressed.
It failed the predeclared learnability criteria (entropy below `0.95`, usage
span above `0.05`, and no worst regression).

Therefore generic dynamic specialization is not promoted. In this homogeneous
4-UAV/4-target regime, lack of actor capacity or agent diversity is not the
measured bottleneck. The ADMN-inspired code remains default-off for a future
large/heterogeneous-scale ablation, where ADMN's original evidence suggests
the hypothesis is more relevant. The main path remains paired,
constraint-preserving value-of-intention supervision.

## QPD-ISAC mechanism screen

A queue-driven distributed primal-dual control plane was implemented as an
opt-in mechanism.  The QoS queue uses the corrected deficit sign

`z <- clip(z + eta * (P_floor - P_local), 0, z_max)`,

so locally unsafe targets accumulate price.  Each receiver performs two local
primal-dual iterations using only its own state and per-target protocol fields
that arrived through the simulated U2U channel.  Sparse event packets merge
per target and expire through the existing TTL, rather than erasing every
silent target when a new sparse packet arrives.  The continuous primal is
projected onto the capped row simplex; the existing exact 1 W per-UAV
comm/sensing projection is unchanged.

The first hard-graph screen exposed an implementation issue: replacing the
latest sparse packet erased the retained state of every silent target.  After
changing retention to per-target merge plus TTL, deterministic frame traces
still showed hard target coverage oscillating between `0.50` and `0.75`
rather than remaining at `1.0`.  With two claims per UAV, local Top-2 rounding
frequently produced zero or one endpoint for a target, so the hard graph
deleted every feasible bistatic edge for that target.  The invalid pre-fix
score is not used as performance evidence.

A soft safety floor retained off-primal edges while QPD reweighted the graph.
The direct five-coordinate protocol hot-swap improved the two-seed mean worst
from the matched baseline's `0.398` to `0.448`, but reduced steady from
`0.756` to `0.696` and weak3 from `0.675` to `0.621`.  This still failed the
Pareto gate and increased the full-graph max-min solve time.

To separate protocol distribution shift from the optimizer, a physically
accounted dual stream was then tested with the same queue/price hyperparameters:
the checkpoint's learned 16-D latent token remained unchanged, while five QPD
coordinates per active target were appended to the same packet and charged in
payload bits, serialization delay, and RF energy.  This compatibility control
recovered steady and weak3, but its target-power/commitment dynamics starved a
different target in both episodes:

| Two-seed screen | steady | weak3 | worst | CVaR | feasible | bits/frame |
|---|---:|---:|---:|---:|---:|---:|
| Matched deployed baseline | 0.756 | 0.675 | 0.398 | 0.001 | 0.50 | 768 |
| QPD soft graph, direct header | 0.696 | 0.621 | 0.448 | 0.001 | 0.50 | 509 |
| QPD soft graph, dual stream | 0.755 | 0.674 | 0.028 | 0.022 | 0.00 | 598 |

All QPD variants preserved the 1 W constraint to numerical precision.  The
dual stream proves that the rejection is not explained only by overwriting
the old latent content.  It preserves the old movement/receiver distribution
well enough to recover mean performance, yet the current local price dynamics
and independent row rounding still do not reconstruct the coupled
two-endpoint rendezvous required by bistatic sensing.  A ten-seed promotion
run and learned marginal-value head were therefore not started.  QPD remains
default-off; the deployed
`0.890/0.854/0.679` checkpoint remains unchanged.

## Fusion-consistent endpoint upper-bound gate

Loading the archived `best_restored.pt` into the current execution tree does
not reproduce the historical aggregate: the current 100-seed episode metrics
are `0.833/0.777/0.528` instead of the historical
`0.890/0.854/0.679`.  Architecture V2 was still uncommitted when the
historical run was generated, while its manifest recorded only the earlier
`fa16abf` baseline commit.  The historical score is therefore retained as an
engineering record, not treated as final reproducible paper evidence.

The first 100-seed physical-oracle run was later found to use a different
evidence boundary from deployment: the oracle summed deflection across
receivers, whereas deployment uses the best single receiver for each target.
That table therefore mixed endpoint assignment with free centralized evidence
fusion and is retained only as an invalid-instrument diagnostic.

The oracle now introduces a receiver-owner variable per target and constrains
all evidence for that target to accumulate at that receiver.  Pair, power,
single-role joint, and duplex variants use the same `local_only` detector as
deployment.  A frozen, no-training 30-seed screen gives:

| Same-frame controller | worst | gap | gap 95% CI | feasible |
|---|---:|---:|---:|---:|
| Deployed graph and power | 0.650 | -- | -- | -- |
| Pair/endpoint only | **0.862** | **+0.213** | **[0.115, 0.321]** | **0.867** |
| Power only | 0.729 | +0.080 | [0.038, 0.127] | 0.733 |
| Joint single-role | **0.910** | **+0.260** | **[0.147, 0.387]** | **0.867** |
| Joint full-duplex | 0.926 | +0.0157 over single-role | [0.002, 0.036] | -- |

The corrected evidence still identifies endpoint assignment and receiver
ownership as the largest controllable deficit.  It does not support full
duplex as the next architectural expansion.  Formal 100-seed confirmation is
deferred until the source tree is frozen.

Two learning-free hyperedge screens were also rejected.  A three-scalar
endpoint proxy reached `0.529/0.463/0.126`; a seven-dimensional physical-state
stream reached `0.492/0.363/0.054`.  Both preserved the 1 W constraint, but
local Top-k mutual selection spread evidence across receivers because it did
not represent a unique receiver owner per target.  Gate B is therefore
narrowed to distributed receiver ownership, endpoint-role prices, and a hard
capacity projection; another free-scoring hyperedge variant is not warranted.

## Fusion-consistent hysteretic structure teacher

The deployed max-min P0 also optimized a central sum while the detector used
the best single receiver.  A default-off receiver-owner P0 was therefore
screened without PPO updates.  The causal sequence was:

| 10-seed control | steady | weak3 | worst | CVaR | feasible |
|---|---:|---:|---:|---:|---:|
| Frozen reproducible baseline | 0.893 | 0.858 | 0.678 | 0.063 | 0.80 |
| Local-fusion P0, filtered commitment graph | 0.776 | 0.708 | 0.440 | 0.001 | 0.30 |
| Local-fusion P0, full graph, hold 1 | 0.886 | 0.848 | 0.654 | 0.145 | 0.60 |
| Local-fusion P0, full graph, hold 5 | **0.919** | **0.893** | **0.734** | **0.154** | **0.80** |

The filtered graph fails because learned Top-2 commitments sometimes delete
every edge for a target.  The per-frame full graph restores coverage but
churns endpoint roles.  A five-frame hold supplies the missing temporal
hysteresis.

Formal 100-test-seed results confirm the effect:

| Controller | steady | weak3 | worst | CVaR | feasible | bit/frame |
|---|---:|---:|---:|---:|---:|---:|
| Frozen reproducible baseline | 0.833 | 0.777 | 0.528 | 0.057 | 0.46 | 768.0 |
| Receiver-owner full-graph hold-5 | **0.913** | **0.884** | **0.751** | **0.354** | **0.73** | **712.7** |

Paired bootstrap deltas are `+0.080 [0.053, 0.108]` for steady,
`+0.107 [0.070, 0.144]` for weak3, and
`+0.223 [0.149, 0.296]` for worst.  QoS gains 34 scenarios and loses seven
(exact McNemar `p=2.53e-5`).  The 1 W error remains `4.44e-16`.
Mean P0 time rises from `1.44 ms/frame` to `3.29 ms/frame`; each hold-5
re-solve costs about `15.9 ms`.

This controller meets the Medium performance thresholds, but it is a
centralized structure teacher, not the final distributed method: it reads the
complete physical candidate graph.  Its proper role is to supervise and bound
a Token-constrained approximation with three explicit states: receiver
ownership, single-role endpoint prices, and switch hysteresis.  Treating the
teacher itself as the paper's distributed contribution would be incorrect.

## Distributed receiver-owner Gate B2

After freezing the teacher, a stricter distributed approximation was tested.
Every node ranked edges only from its own public quantized offer, actually
delivered neighbor offers, and mission-known target positions.  The true
physical table was used only to reject infeasible execution edges, never as a
score.  Receiver ownership and a five-frame hold were explicit.

| Two-seed protocol | steady | weak3 | worst | coverage | bit/frame |
|---|---:|---:|---:|---:|---:|
| Matched frozen baseline | 0.756 | 0.675 | **0.398** | -- | 768 |
| 4-bit absolute-position reconstruction | 0.751 | 0.668 | 0.070 | 0.979 | 1728 |
| Exponential endpoint proxy | 0.798 | 0.731 | 0.341 | 0.958 | 1467 |
| Factorized inverse-square capability | **0.832** | **0.776** | 0.344 | **0.991** | 1472 |

The absolute-position version fails because four-bit coordinates over 800 m
have roughly 53 m resolution.  Factorized capability removes this
quantization pathology and improves mean metrics, yet its episode worst values
are `0.094/0.594`: one target can still starve despite almost complete graph
coverage and assignment reuse near 0.75.  Gate B2 therefore stops before ten
seeds.  The remaining missing state is a target-deficit price shared across
nodes, not more Token dimensions or another distance-scale sweep.
