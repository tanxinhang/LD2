# Architecture V2: distributed set-based ISAC policy

The method-level prior-art boundary and the evidence-based decision to explore
role-asymmetric bistatic hyperedge negotiation are documented separately in
[`NOVELTY_AND_SCOPE_AUDIT.md`](NOVELTY_AND_SCOPE_AUDIT.md).

## Current status snapshot (2026-08-01)

The authoritative Chinese status, system boundary, deployment decision, and
next-stage protocol are summarized in
[`CURRENT_SYSTEM_STATUS.md`](CURRENT_SYSTEM_STATUS.md). The sections below
retain the full experimental evolution; early bounded results are historical
screens rather than the current deployment claim.

| Protocol | Seeds | steady | weak3 | mean worst | CVaR | QoS feasible | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| 4/4 frozen deployment, 50 kbit/s, adaptive 4--8 bit | 100 | 0.913 | 0.885 | 0.739 | 0.277 | 0.72 | frozen release |
| 4/4 cardinality-residual anchor gate | 10 | 0.954 | 0.939 | 0.858 | 0.477 | 0.90 | exactly equal to the original Student |
| 6/6 old cross-scale Student | 10 | 0.893 | 0.787 | 0.543 | 0.158 | 0.60 | rejected |
| 6/6 anchor-preserving residual | 10 | 0.895 | 0.802 | 0.645 | 0.202 | 0.60 | candidate, not deployable |
| 6/6 frozen centralized control | 10 | 0.896 | 0.792 | 0.635 | 0.065 | 0.70 | reference only |

The residual improves paired 6/6 worst by `+0.1025` over the old Student
(bootstrap CI `[+0.0499, +0.1574]`) while preserving every 4/4 episode exactly.
It nevertheless fails the `QoS feasible >= 0.70` promotion gate. No 6/6 or
8/8 deployment claim is made. Formal 100-seed results and 10-seed mechanism
screens must not be pooled.

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
- 385 main regression tests and 14 belief-calibration tests pass.
- A 4/4 V2 checkpoint loads directly into the 6/6 actor and critic.

## Historical bounded experimental results

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

## Frozen-teacher distillability audit

The next stage now targets the frozen receiver-owner full-graph Hold-5
controller directly. A new synchronized trace records, at each P0 decision:

- the exact directed teacher edges, Tx/Rx target incidence, receiver owner and
  single-role labels;
- the local observation, received Token state, outgoing Token mask, resource
  action and local detection state that a distributed student can use;
- the same-frame full deflection graph in a separately named `privileged_*`
  block, which is prohibited as a deployment input.

The two-seed exporter reproduced the archived teacher exactly
(`0.875/0.834/0.613`), so trace instrumentation does not change the policy or
environment. The ten-seed audit retained the earlier frozen-teacher result
(`0.919/0.893/0.734`) and exposed a cardinality mismatch:

| 10-seed resolve-frame audit | value |
|---|---:|
| Teacher directed edges per decision | 7.00 (range 4--12) |
| Teacher node-target endpoints per team | 11.00 |
| Fixed Top-2 endpoint recall | 0.677 |
| Fixed Top-2 full teacher-edge support | 0.346 |
| Fixed Top-2 team-exact endpoint rate | 0.042 |

A team of four Top-2 senders can expose at most eight node-target endpoints,
while the teacher uses eleven on average. Therefore Top-2 is not merely an
optimization choice during teacher approximation; it is an information
bottleneck that makes exact imitation structurally impossible.

A seed-grouped offline shared per-target scorer was trained on eight seeds and
tested on two unseen seeds. Using only the pre-decision local information, its
role accuracy was `0.629`, team-exact role rate `0.113`, and receiver-owner
accuracy `0.415`. A simple pooled-observation probe did not improve these
figures. In contrast, copying the previous Hold-5 structure without a neural
network achieved role accuracy `0.749`, team-exact role rate `0.547`, receiver
owner accuracy `0.592`, and pair Jaccard `0.539` at re-solve boundaries.

Direct switch-label learning was then tested and rejected. Adding the previous
teacher structure as a feature, or as a zero-initialized persistence residual,
did not beat copying the previous structure on the two unseen seeds. The
discrete MILP label remains too discontinuous for the current local inputs.

The teacher projection was therefore replayed from the traced double-precision
deflection graph and traced QoS price state. It reconstructed every edge,
role and receiver owner on all 310 ten-seed re-solve frames (`exact=1.000`).
This validates both the trace instrument and a cleaner distillation boundary:
learn continuous directed edge value, keep the combinatorial constraints
analytic.

A first factorized graph probe used shared Tx and Rx endpoint encoders plus a
directed pair decoder. On two unseen seeds it achieved:

| Factorized continuous graph probe | value |
|---|---:|
| Per-frame edge-value Spearman | 0.922 |
| Projected role exact rate | 0.694 |
| Projected receiver-owner exact rate | 0.710 |
| Projected full-graph exact rate | 0.306 |
| Projected edge F1 | 0.680 |

This is substantially stronger than direct receiver-owner classification
(`0.427` owner accuracy), although still only a ten-seed learnability result.

The trace was then expanded to 30 seeds. The first 30 seed IDs and every
per-episode teacher metric are bit-for-bit identical to the frozen 100-seed
table. A 23/1/6 seed-grouped fit/validation/test split gives:

| 30-seed offline graph control | edge rank | role exact | owner exact | edge F1 | realized worst |
|---|---:|---:|---:|---:|---:|
| Frozen teacher | 1.000 | 1.000 | 1.000 | 1.000 | 0.696 |
| Factorized learned edge value | 0.868 | **0.801** | **0.731** | 0.780 | **0.620** |
| Calibrated inverse-range physics | **0.919** | 0.758 | 0.688 | **0.809** | 0.591 |
| Physics plus learned residual | 0.893 | 0.796 | **0.731** | 0.759 | 0.609 |

The learned graph realizes `mean/weak3/worst = 0.862/0.818/0.620` on the
teacher's held-out physical states, versus `0.886/0.849/0.696` for the
teacher. Its QoS frame-feasibility rate is `0.548`, versus `0.629`.
Thus it passes only a preliminary same-state `worst >= 0.60` gate; it has not
yet sufficiently approximated the teacher for closed-loop promotion.

The physical control recovers more literal edges but loses the bottleneck
target more often. A zero-initialized neural residual over that physical law
falls between the two controls and does not dominate either. The next edge
student should therefore retain the shared factorization but train on a larger
teacher bank with a bottleneck/constraint-aware edge-value objective. Merely
adding the inverse-range prior is not promoted.

The immediate student design is consequently narrowed to:

1. transmit all-Q endpoint embeddings during the teacher-approximation phase;
2. reconstruct \(\widehat d_{i,j,q}\) with a directed factorized decoder;
3. retain the previous feasible structure and five-frame hold as hard state;
4. apply the frozen receiver-owner/single-role max-min projection locally at
   every node;
5. only after closed-loop graph imitation succeeds, distil the all-Q stream
   into an event-triggered sparse Token protocol.

This is a staged centralized-to-distributed transfer experiment, not a claim
that the centralized teacher itself is deployable.

## Clean-reset frozen teacher and continuous-graph student

The first closed-loop student gate exposed an evaluation-state leak that
supersedes the frozen-teacher and distillability numbers above for formal
reporting. `EnvironmentCore.reset()` cleared the team `prev_P_D` but did not
clear the per-UAV `prev_P_D_local` dictionary. Consequently the first actor
decision of an episode inherited local detection history from the preceding
seed. The result was deterministic for one fixed seed order, but it was not
episode independent: changing the number or order of evaluated seeds changed
the first action and could amplify into a different trajectory.

The reset now clears `prev_P_D_local`. A regression test injects a non-zero
history, repeats the same reset seed, and requires the complete initial
observation to be bit-for-bit identical to a clean reset. The rebuilt
100-seed teacher trace satisfies:

- 15,000 frames, 3,100 Hold-5 re-solve frames and 100 unique episodes;
- zero local detection-history entries on every episode's first frame;
- exact per-episode agreement between the first ten episodes of the 100-seed
  run and an independently executed ten-seed run.

The corrected frozen structure teacher is:

| Clean 100-seed teacher | steady | weak3 | worst | CVaR | feasible | bit/frame |
|---|---:|---:|---:|---:|---:|---:|
| Receiver-owner full-graph Hold-5 | 0.909 | 0.880 | 0.731 | 0.261 | 0.70 | 716.9 |

The previous `0.913/0.884/0.751` table is retained only as historical audit
evidence and must not be used as the formal teacher result.

The clean trace uses an 80/10/10 seed-grouped fit/validation/test split.
Ordinary continuous edge regression misses the weak3 gap gate by a small
margin. A bottleneck-weighted regression with a small soft teacher-edge
distribution term is selected by projected worst/QoS, not by edge rank:

| Clean held-out same-state projection | mean | weak3 | worst | feasible |
|---|---:|---:|---:|---:|
| Frozen teacher | 0.831 | 0.774 | 0.520 | 0.423 |
| Selected factorized student | 0.819 | 0.759 | 0.490 | 0.400 |
| Student minus teacher | -0.011 | -0.015 | -0.030 | -0.023 |

The selected student has projected role/receiver-owner exact rates
`0.794/0.729`. Its saved artifact reproduces the audit output exactly after
reload, including normalization.

An information-equivalent closed-loop gate then assembles all UAV endpoint
features in one process, predicts the directed edge graph, and applies the
same frozen local-fusion max-min projection. This gate does **not** yet charge
the endpoint embeddings to the stochastic Token channel, so it proves
closed-loop sufficiency of distributed inputs but is not the final deployable
protocol.

| Clean 10-seed closed loop | steady | weak3 | worst | CVaR | feasible | bit/frame |
|---|---:|---:|---:|---:|---:|---:|
| Frozen teacher | 0.947 | 0.929 | 0.836 | 0.451 | 0.90 | 711.7 |
| Factorized student | 0.931 | 0.909 | 0.807 | 0.371 | 0.80 | 703.0 |
| Student minus teacher | -0.015 | -0.020 | -0.029 | -0.080 | -0.10 | -8.7 |

Mean worst meets the approximation gap, but CVaR and feasibility do not.
One DAgger-style round relabelled the student's visited states with the frozen
teacher and repeated those states four times. It was rejected before
closed-loop evaluation because held-out role/owner exact rates fell from
`0.794/0.729` to `0.694/0.581`, and worst/weak3 gaps worsened from
`0.030/0.015` to `0.035/0.020`. Repeating the ten observed scenes therefore
caused scenario adaptation rather than robust teacher approximation.

The current stopping point is:

1. keep the clean structure-assisted student as the active surrogate;
2. do not promote the rejected DAgger artifact;
3. next transmit the all-target endpoint embeddings through the actual
   delayed/lossy Token channel and account for their bits and power;
4. require a larger clean paired gate before any sparse/event-triggered
   compression or paper-level performance claim.

## Physical endpoint-Token transfer and solve-aligned communication

The all-target endpoint surrogate has now been moved from the
information-equivalent process boundary to the physical U2U channel. Each UAV
encodes one transmitter and one receiver endpoint per target. The endpoint
stream is quantized, appended to the existing broadcast packet and charged to
the same serialization delay, deadline, link budget, radio energy and 1 W
communication/sensing simplex as the learned latent Token.

To preserve a common distributed graph, a sender version is published only
when its broadcast reaches every other UAV. Failed versions do not partially
advance the graph; the nodes retain the last common version until the existing
five-frame TTL expires. The frozen directed decoder then reconstructs the
same \(K\times K\times Q\) ranking graph from this public endpoint table.
Realized detection continues to use the physical deflection of the selected
edges.

Two implementation failures were rejected before formal evaluation:

1. A 32-dimensional Tx/Rx endpoint at four bits was deliverable, but coarse
   endpoint quantization reduced the two-seed worst from the
   information-equivalent `0.614` to `0.482`.
2. Raising the shared packet rate to eight bits doubled the structural packet
   and missed every 5 ms deadline. More importantly, it also changed the
   quantization distribution of the actor's original latent Token, creating
   an unintended protocol/actor coupling.

The corrected protocol therefore uses two separately coded streams in one
physically charged packet:

- the actor's original latent Token retains its trained four-bit/silence
  decision;
- a low-rank structural stream uses an independent eight-bit codebook;
- the endpoint rank is reduced from 32 to 8 per Tx/Rx endpoint;
- a one-percent communication-power floor is applied only when a mandatory
  structural packet would otherwise have zero transmit power.

The low-rank endpoint is not merely a bandwidth heuristic. On the held-out
same-state projection it increased exact graph recovery from `0.358` to
`0.532`, retained owner exact accuracy `0.739`, and produced a worst gap of
`-0.0317`. Its eight-bit quantization has endpoint MAE `0.00195`, changes
edge-rank correlation only from `0.9107` to `0.9104`, and changes projected
worst from `0.4882` to `0.4848`. The large positive and negative differences
seen in individual closed-loop episodes are consequently due to discrete
scheduling boundaries amplifying small ordering changes, not a large
same-state regression error.

The first physical upper control sent structural endpoints every frame:

| Clean 100-seed result | steady | weak3 | worst | CVaR | feasible | bit/frame |
|---|---:|---:|---:|---:|---:|---:|
| Frozen centralized structure teacher | 0.909 | 0.880 | 0.731 | 0.261 | 0.70 | 716.9 |
| E=8, 8-bit U2U endpoint, every frame | 0.912 | 0.883 | 0.735 | 0.253 | 0.70 | 2789.7 |

The student's paired deltas versus the teacher are
`+0.0034/+0.0034/+0.0040` for steady/weak3/worst. Their episode-bootstrap 95%
intervals are respectively `[-0.0130, 0.0198]`,
`[-0.0185, 0.0253]`, and `[-0.0418, 0.0506]`. Thus the correct claim is
statistical equivalence, not superiority.

Finally, the structural stream was aligned with the frozen Hold-5 projection.
It is transmitted only immediately before a predictable P0 re-solve; the
cached public graph is retained between solves. If crisis/event-triggered P0
is enabled, the implementation conservatively returns to every-frame
structural transmission because the next solve cannot be predicted before
the physical graph is formed.

| Clean 100-seed result | steady | weak3 | worst | CVaR | feasible | bit/frame | structural bit/frame |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen centralized structure teacher | 0.909 | 0.880 | 0.731 | 0.261 | 0.70 | 716.9 | 0 |
| U2U endpoint, every frame | 0.912 | 0.883 | 0.735 | 0.253 | 0.70 | 2789.7 | 2048.0 |
| U2U endpoint, P0-resolve aligned | 0.910 | 0.881 | 0.738 | 0.307 | 0.72 | 1146.4 | 423.3 |

Resolve alignment reduces the structural payload by `79.3%` and total learned
communication by `58.9%` relative to every-frame transmission. It exactly
matches the measured P0 resolve-frame rate `31/150 = 0.2067`, retains 100%
atomic multicast delivery and public-cache validity, and has a mean cached
age of `1.97` frames. The maximum per-UAV 1 W power-balance error is
`4.44e-16` W.

Against the teacher, the resolve-aligned paired steady/weak3/worst deltas are
`+0.0015/+0.0014/+0.0076`, with bootstrap 95% intervals
`[-0.0146, 0.0182]`, `[-0.0200, 0.0236]`, and
`[-0.0357, 0.0528]`. It satisfies the deployment thresholds
`steady >= 0.80`, `weak3 >= 0.70`, and mean `worst >= 0.60`, while matching
the teacher's mean performance and slightly improving the measured tail and
QoS feasibility on this fixed 100-seed bank.

This resolve-aligned, dual-codebook, low-rank endpoint protocol is the current
frozen distributed approximation candidate. Larger-team and channel-shift
tests remain necessary before claiming scale or zero-shot channel
generalization; no further architecture module should be added until those
tests are complete.

## Channel-shift audit and capacity-aware structural quantization

The first deadline stress test exposed a privileged fallback bug rather than
a policy failure. When fewer than two structural endpoint caches were valid,
the environment set the external graph to `None`; the P0 implementation then
silently resumed its native centralized physical ranking. A 2 ms deadline
therefore appeared to achieve nearly perfect detection while structural
atomic delivery and cache validity were both zero. This result is invalid and
is excluded. The channel now fails closed to an explicit zero edge graph, and
the evaluator reports the fraction of frames with fewer than two cached
senders. Under the corrected implementation the same 2 ms/two-seed control
has `steady=weak3=worst=0.001`, zero feasibility, zero structural cache
validity and an insufficient-cache rate of one. The nominal 100-seed result
above is unaffected because its atomic delivery and cache validity were both
one on every evaluated frame.

Three corrected ten-seed channel screens were then compared with the same
nominal seed bank:

| Channel screen (10 seeds) | steady | weak3 | worst | CVaR | feasible | atomic delivery | cache valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nominal | 0.948 | 0.931 | 0.849 | 0.453 | 0.90 | 1.000 | 1.000 |
| 3 ms deadline | 0.929 | 0.906 | 0.818 | 0.404 | 0.80 | 0.949 | 0.956 |
| 30 dB receive threshold | 0.907 | 0.878 | 0.769 | 0.402 | 0.70 | 0.930 | 0.939 |
| 50 kHz bandwidth | 0.909 | 0.878 | 0.726 | 0.189 | 0.80 | 0.901 | 0.909 |

The 50 kHz condition is the strongest and least redundant perturbation, so it
was promoted to the formal 100-seed audit. Fixed eight-bit structural coding
still meets the mean Medium thresholds, but it does not preserve the QoS
boundary:

| Clean 100-seed result | steady | weak3 | worst | CVaR | feasible | bit/frame | structural bit/frame | atomic delivery | cache valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen centralized structure teacher | 0.909 | 0.880 | 0.731 | 0.261 | 0.70 | 716.9 | 0 | n/a | n/a |
| Nominal, fixed 8 bit | 0.910 | 0.881 | 0.738 | 0.307 | 0.72 | 1146.4 | 423.3 | 1.000 | 1.000 |
| 50 kHz, fixed 8 bit | 0.883 | 0.846 | 0.665 | 0.208 | 0.63 | 1148.5 | 423.3 | 0.923 | 0.930 |
| 50 kHz, fixed 6 bit | 0.909 | 0.880 | 0.728 | 0.239 | 0.72 | 1040.9 | 317.4 | 0.997 | 0.997 |
| 50 kHz, adaptive 4--8 bit | **0.913** | **0.885** | **0.739** | **0.277** | **0.72** | 1143.3 | 419.5 | **1.000** | **1.000** |

Simply increasing the structural power floor was rejected. On the two-seed
screen, floors of `0.30`, `0.40` and `0.50` improved packet delivery but did
not Pareto-improve sensing; the 0.50 floor changed atomic delivery from
`0.895` to `1.000` while reducing worst from `0.624` to `0.614`. This is the
expected consequence of taking power away from sensing under the exact 1 W
simplex and rules out a fixed high-power repair.

The accepted repair is instead a capacity-aware precision projection. For
each structural broadcast, sender (i) selects the largest integer precision

\[
b_i^t = \max\left\{b\in[b_{\min},b_{\max}]:
  \gamma_{ij}^t\ge\gamma_{\min},\;
  \ell_{ij}^t(b)\le\ell_{\max}\quad\forall j\ne i\right\}.
\]

The latency uses the complete packet size: actor payload, structural
endpoints, ordinary packet header and a three-bit structural-rate field for
the five levels `{4,5,6,7,8}`. The calculation consumes only the sender's
selected communication power and peer link CSI/range; it has no access to
target truth, detection evidence or centralized P0 values. If even the
minimum rate is infeasible, the packet is transmitted at the minimum and the
normal fail-closed cache rule remains active.

Offline same-state auditing explains why this is safe. At six bits, endpoint
MAE is `0.00790`, but edge-rank Spearman is `0.91076`, versus `0.91039` at
eight bits. Projected same-state worst changes only from `0.48479` to
`0.48344`; the main cost is receiver-owner exact rate `0.735 -> 0.713`.
Consequently lower precision is useful only on capacity-limited packets, not
as an unconditional replacement for the nominal eight-bit stream.

In the formal 50 kHz run the adaptive selector averages `7.883` bits per
endpoint dimension: it normally retains high precision and reduces the rate
only near the link-feasibility boundary. Against fixed eight-bit 50 kHz, its
paired steady/weak3/worst gains are `+0.0306/+0.0387/+0.0739`; the respective
episode-bootstrap 95% intervals are `[+0.0158,+0.0463]`,
`[+0.0197,+0.0587]`, and `[+0.0363,+0.1137]`. QoS improves in 13 paired
episodes and regresses in four (exact McNemar `p=0.049`). Against the frozen
teacher, its paired worst delta is `+0.0086` with 95% interval
`[-0.0341,+0.0521]`, so the defensible claim remains statistical equivalence,
not superiority.

The capacity-aware, solve-aligned physical Token version is therefore the
new frozen **4/4 channel-robust approximation**. It satisfies the requested
Medium mean thresholds and restores the frozen teacher's QoS feasibility at
half the nominal bandwidth without retraining or violating the 1 W balance
(`4.44e-16` W maximum error).

This does not yet establish scale generalization. The deployed Actor already
uses a shared per-target policy, Set/Attention communication-power heads,
permutation-equivariant round encoding and an equivariant value critic. The
factorized structure student is also mathematically cardinality-equivariant.
However, the runtime observation layout is instantiated for one K/Q, and the
student artifact loader deliberately rejects cardinalities different from its
4/4 training metadata. These guards must not be removed until a cross-scale
normalization and calibration audit is available. The next stage is therefore
an explicit 4/4-to-6/6 and 4/4-to-8/8 migration experiment, not another 4/4
Actor/reward modification.

## Controlled-density zero-shot scale migration

The scale audit uses fixed stratified seed banks that preserve UAV/target
density rather than placing more entities into the original area. The 6/6
region is `980 x 980 m` and the 8/8 region is `1130 x 1130 m`, corresponding
to (800\sqrt{6/4}) and (800\sqrt{8/4}). No Actor, Critic or Student weight
is updated. The 4/4 checkpoint is loaded directly into the shared per-target
and Set/Attention architecture.

Cross-scale Student loading is deliberately opt-in. The default loader still
rejects a K/Q mismatch. The audit flag is accepted only when the Actor uses
the Architecture-V2 shared target trunk, scale-equivariant communication
heads, permutation-equivariant round encoding and equivariant Critic, and the
Student declares its shared endpoint/pair decoder cardinality equivariant.
This prevents an old fixed-width model from being silently treated as a
scale-general policy.

Environment and checkpoint construction pass without parameter reshaping:

| scale | local observation | global state | finite reset | Actor/Critic load |
|---|---:|---:|---:|---:|
| 6/6 | 865 | 97 | yes | exact-compatible |
| 8/8 | 1515 | 129 | yes | exact-compatible |

For 6/6, the first physical-Student seed passed, but the ten-seed result did
not satisfy the requested mean-worst deployment gate:

| 6/6, 10 seeds | steady | weak3 | worst | CVaR | feasible | bit/frame | selected structural bits | atomic/cache |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Centralized physical structure control | 0.896 | 0.792 | 0.635 | 0.065 | 0.70 | 1147.6 | n/a | n/a |
| Zero-shot physical Student | 0.893 | 0.787 | 0.543 | **0.158** | 0.60 | 1986.6 | 7.001 | 1.000/1.000 |

The paired Student-minus-control worst difference is `-0.0921`, with a
ten-seed episode-bootstrap 95% interval `[-0.2226,+0.0132]`. The Student
improves three seeds and degrades seven. Importantly, it improves the hardest
cases (for example seed 2 worst `0.038 -> 0.203`) while over-correcting several
easy cases (seed 860 `0.998 -> 0.451`). This explains why CVaR improves while
mean worst and feasibility fall. Since physical atomic delivery and cache
validity are both one, communication transport is not the 6/6 bottleneck.

A same-state audit on the 6/6 teacher trace supports this diagnosis. The
4/4 Student retains a high edge-value Spearman correlation of about `0.896`,
but projected full-structure exact rate is only `0.032` and receiver-owner
exact rate is `0.258` at eight bits on the held-out seed. Ranking error that
was tolerable over (4^2Q) candidate edges is amplified by the larger
(K^2Q) combinatorial projection. Six-bit quantization does not repair it;
the failure is cross-cardinality structure calibration, not code precision.

The 8/8 one-seed gate was stopped before ten-seed promotion:

| 8/8 isolation | worst | atomic delivery | cache valid |
|---|---:|---:|---:|
| Centralized physical structure control | 0.142 | n/a | n/a |
| Fixed 100 kHz total bandwidth | 0.001 | 0.254 | 0.251 |
| 200 kHz total bandwidth | 0.097 | 1.000 | 1.000 |
| Perfect channel | 0.054 | 1.000 | 1.000 |

The fixed-total-bandwidth result exposes a real communication scaling limit,
but restoring transport does not restore worst. Even the centralized control
is below the deployment floor on this seed, so 8/8 currently combines an
extreme-value geometry/horizon limit with Student calibration error. It is not
valid to claim zero-shot 8/8 generalization or to attribute the failure only
to communication.

The resulting development boundary is precise:

1. keep the 4/4 capacity-aware physical Token version frozen;
2. do not promote the present zero-shot 6/6 or 8/8 Student;
3. construct a multi-cardinality teacher bank with disjoint 4/4, 6/6 and 8/8
   seed groups;
4. train the same shared endpoint encoder/decoder by alternating cardinality
   batches, with an explicit easy-scene preservation loss and worst/QoS
   selection rule;
5. re-run 6/6 before returning to 8/8, and report fixed-total-bandwidth and
   constant-per-UAV-bandwidth regimes separately.

This is a useful negative result: the policy network is structurally capable
of accepting new cardinalities, but architectural equivariance alone does not
guarantee calibrated combinatorial decisions under a larger candidate graph.

### Mixed-cardinality calibration gate

The next gate trained one shared endpoint encoder/decoder on new 4/4 and 6/6
centralized-teacher traces.  Both traces use the `selection` split rather than
the formal `test` split and contain 20 seeds and 620 P0 resolve frames.  The
single overlapping numeric seed (`304`) was excluded from the later-loaded
6/6 dataset.  Within each cardinality, fit/validation/test seeds are disjoint,
and every epoch contains the same number of 4/4 and 6/6 batches; no fixed-size
padding or UAV/target identifier is introduced.

The mixed model is initialized by analytically transforming the first-layer
weights for the new balanced feature normalization.  At epoch zero this
preserves the old raw 4/4 Student function exactly.  Checkpoints are selected
lexicographically by minimum-cardinality QoS feasibility, minimum-cardinality
worst, mean QoS and mean worst.  A second version additionally preserves the
old 4/4 decoded edge values and transmitted endpoint protocol on every 4/4
frame and retains the original endpoint quantization scale.

The ten-seed closed-loop gate is:

| Scale and model | steady | weak3 | worst | CVaR | feasible | atomic/cache |
|---|---:|---:|---:|---:|---:|---:|
| 4/4 frozen Student, 50 kHz | 0.954 | 0.939 | **0.858** | **0.477** | **0.90** | 1.000/1.000 |
| 4/4 mixed Student, 50 kHz | 0.932 | 0.910 | 0.752 | 0.170 | 0.70 | 1.000/1.000 |
| 4/4 protocol-guarded mixed Student, 50 kHz | 0.931 | 0.908 | 0.751 | 0.170 | 0.70 | 1.000/1.000 |
| 6/6 frozen 4/4 Student | 0.893 | 0.787 | 0.543 | 0.158 | 0.60 | 1.000/1.000 |
| 6/6 mixed Student | **0.901** | **0.806** | **0.640** | 0.156 | **0.70** | 1.000/1.000 |
| 6/6 protocol-guarded mixed Student | 0.890 | 0.787 | 0.618 | **0.179** | 0.60 | 1.000/1.000 |
| 6/6 centralized physical structure control | 0.896 | 0.792 | 0.635 | 0.065 | 0.70 | n/a |

Thus the unguarded mixed Student crosses the 6/6 mean deployment gate and is
close to the centralized control.  Relative to the frozen 4/4 Student, its
paired 6/6 worst gain is `+0.0970`, with ten-seed bootstrap interval
`[-0.0871,+0.2537]`; nine seeds improve and one degrades.  This is a promising
mechanism signal, but not a significant superiority result.

It is not deployable because the same weights cause closed-loop forgetting at
4/4.  The paired 4/4 worst difference is `-0.1055`, with interval
`[-0.3100,+0.0541]`, and QoS loses two of ten episodes.  More importantly, the
protocol guard does not repair this (`worst=0.751`, feasible `0.70`).  Offline
same-state auditing had predicted only a roughly `0.004` worst change, so the
large closed-loop loss is caused by discrete projection and trajectory
feedback amplifying small shared-parameter changes, not by communication
loss: atomic delivery and cache validity remain one.

Both mixed checkpoints are therefore rejected and the original 4/4 artifact
remains frozen.  The result also rules out further preservation-loss tuning as
the next main experiment.  A defensible successor must enforce structural
non-interference: retain the exact frozen 4/4 endpoint path and add a
permutation-invariant, cardinality-conditioned endpoint residual whose gate is
identically zero at the anchor cardinality.  Only the residual may be trained
on 6/6 (and later 8/8).  This converts approximate behavior preservation into
an architectural guarantee and directly addresses the discontinuity exposed
by this gate.

### Anchor-preserving cardinality residual

The structural non-interference proposal was then implemented as a schema-v2
Student.  For endpoint encoder (E) and directed decoder (D), it uses

\[
E_{K,Q}(x)=E_0(x)+g(K,Q)\,\Delta E(x),\qquad
D_{K,Q}(e)=D_0(e)+g(K,Q)\,\Delta D(e),
\]

where (E_0,D_0) are the immutable deployed 4/4 Student, and (g(K,Q)) is a
clipped, permutation-invariant cardinality distance.  Here (g(4,4)=0) and
(g(6,6)=1).  The base normalization and endpoint quantization scale are also
copied without modification.  Only the residual endpoint encoders and decoder
receive gradients from 6/6 selection-split frames; 4/4 data is used solely for
validation.

This changes the preservation claim from empirical to structural.  On real
4/4 teacher-trace features, old and new raw edge arrays and endpoint protocol
arrays are exactly equal (`array_equal=True`, maximum difference zero).  The
ten-seed 50 kHz closed-loop run is also bit-for-bit identical in every episode:

| 4/4 model | steady | weak3 | worst | CVaR | feasible | bit/frame |
|---|---:|---:|---:|---:|---:|---:|
| Frozen schema-v1 Student | 0.954 | 0.939 | 0.858 | 0.477 | 0.90 | 1131.0 |
| Cardinality-residual schema-v2 Student | 0.954 | 0.939 | 0.858 | 0.477 | 0.90 | 1131.0 |

The corresponding 6/6 diagnostic is:

| 6/6 model | steady | weak3 | worst | CVaR | worst LCB | feasible |
|---|---:|---:|---:|---:|---:|---:|
| Frozen 4/4 Student | 0.893 | 0.787 | 0.543 | 0.158 | 0.397 | 0.60 |
| Shared mixed Student (rejected at 4/4) | 0.901 | 0.806 | 0.640 | 0.156 | 0.474 | **0.70** |
| **Anchor-preserving residual** | 0.895 | **0.802** | **0.645** | **0.202** | **0.484** | 0.60 |
| Centralized physical structure control | **0.896** | 0.792 | 0.635 | 0.065 | 0.448 | **0.70** |

Against the old cross-scale Student, the residual worst gain is `+0.1025`
with paired ten-seed bootstrap interval `[+0.0499,+0.1574]`; nine seeds improve,
none degrades and one is unchanged.  Against centralized structure control,
the worst difference is `+0.0105`, but its interval
`[-0.1377,+0.1360]` does not establish superiority.  Physical atomic delivery
and cache validity remain one, and the exact 1 W projection remains active.

The residual therefore solves the previously observed architectural failure:
it preserves the deployed anchor exactly while transferring a statistically
clear worst-target improvement to 6/6.  It also satisfies the requested 6/6
mean Medium thresholds.  It is not yet promoted, because six rather than seven
of ten episodes satisfy all three QoS thresholds.  The remaining boundary
failure is concentrated in seed 860 (`worst=0.492`), while the genuinely hard
seeds remain below the floor for both the Student and centralized control.

The next experiment should not alter the frozen base or add another reward
module.  The present selection trace leaves only two 6/6 validation seeds, both
with zero QoS feasibility, so the checkpoint key cannot learn the deployment
boundary.  A larger disjoint 6/6 calibration/validation bank is required before
another residual fit: retain at least ten validation seeds spanning both sides
of the QoS boundary, select by feasible-rate lower bound followed by worst, and
then repeat this same 4/4-exact/6/6-Pareto gate.  No 8/8 claim is made yet.

### Local-candidate composability gate

The next scale experiment was changed from repeated 4/4 subgraph inference to
a stricter question: can a sparse hyperedge set built from deployment-local
information still contain a high-quality receiver-owner solution? Repeating a
factorized per-edge decoder on overlapping 4/4 slices is redundant when the
endpoint features are unchanged, and pooling all `K-1` peers before slicing
also leaks the full-cardinality context. The new audit therefore:

1. admits only peers represented by physically delivered target Tokens;
2. recomputes received mean, maximum and delivery fraction on the selected
   local peer set;
3. reserves candidate capacity for locally weak targets and fills the rest by
   the frozen shared edge scorer;
4. restricts the same receiver-owner, single-role, local-fusion solver to the
   resulting union; and
5. compares it with a full-physical-graph replay using the identical solver.

The naive `L_u=3,L_q=4` AoI/availability selector failed immediately on one
seed: exact edge recall was `0.511`, owner recall `0.715`, and the candidate
worst lost `0.181`. This is not a capacity impossibility: the teacher uses at
most three distinct receivers per transmitter. A single value-and-coverage
refinement raises owner recall, while a second refinement is rejected because
it amplifies cross-cardinality score errors (`worst 0.443 -> 0.343` in the
one-seed diagnostic).

The ten-seed capacity sweep is:

| Candidate | physical edges retained | exact edge recall | owner recall | steady | weak3 | worst | CVaR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `L_u=3,L_q=4`, one value refinement | 0.387 | 0.657 | 0.920 | 0.904 | 0.821 | 0.592 | 0.126 |
| `L_u=4,L_q=4`, one value refinement | 0.516 | 0.767 | 0.959 | 0.895 | 0.806 | 0.592 | 0.165 |
| `L_u=5,L_q=4`, availability upper anchor | 0.644 | 0.847 | 0.964 | 0.899 | 0.812 | 0.600 | 0.165 |

For the selected diagnostic point `L_u=4,L_q=4`, the aligned full-graph replay
is `0.909/0.820/0.594` for steady/weak3/worst with `CVaR=0.165`. The restricted
candidate replay is `0.895/0.806/0.592`, with `CVaR=0.165`; hence the aligned
pruning gaps are `0.0143/0.0145/0.0015/0.0006`. Excluding only the first
resolve of each episode for diagnosis, owner recall is `0.991`, target-empty
rate is `0.0011`, and exact edge recall is `0.805`. The overall empty rate of
`0.0333` is therefore a cold-start protocol problem, not steady-state target
starvation.

The predeclared mechanism gate nevertheless **fails**. Exact teacher-edge
recall remains below `0.90`, and the unbootstrapped overall empty-target rate
exceeds `0.02`. The small performance gaps show that many omitted teacher
edges have receiver-owner-equivalent substitutes, but this equivalence was
not declared before the experiment and cannot be substituted post hoc for the
failed exact-recall criterion. This strict Gate A remains recorded as failed;
it motivated a separately specified equivalence-aware Gate A2 on disjoint
seeds, reported below.

Implementation and reproducible artifacts:

- candidate audit logic: `uav_isac/evaluation/local_candidate_audit.py`;
- audit entry point: `tools/audit_local_candidate_upper_bound.py`;
- selected report:
  `results/architecture_v2_scale_k6q6_local_candidate_value_lu4_lq4_r1_gate10/summary.json`;
- per-episode metrics and per-frame masks are saved beside the report.

### Equivalence-aware held-out Gate A2

Gate A2 compares target evidence rather than requiring the candidate graph to
reproduce the exact receiver edge selected by a discrete solver. To prevent a
solver-path mismatch from being mistaken for a candidate error, the reference
owner and candidate owner are replayed from the same state with the same
solver. A candidate target is equivalent when it retains at least `0.95` of
the reference-supported evidence, either for the same owner or for any
feasible owner. The thresholds and `L_u=4,L_q=4`, one-refinement candidate
were fixed before evaluating the unused final ten trace seeds
`369,424,483,522,566,581,606,707,984,989`.

Warm-start results are:

| held-out candidate metric | value | threshold | pass |
|---|---:|---:|---:|
| any-owner equivalent target recall | 0.9917 | >=0.90 | yes |
| same-owner equivalent target recall | 0.9522 | diagnostic | -- |
| receiver-owner recall | 0.9961 | >=0.90 | yes |
| empty-target rate | 0.00056 | <=0.02 | yes |
| exact edge recall | 0.8695 | >=0.90 (strict Gate A) | no |

The aligned full replay is `0.8801/0.7624/0.5357/0.2114/0.40` and the
candidate replay is `0.8797/0.7685/0.5505/0.2114/0.40` for
steady/weak3/worst/CVaR/QoS feasibility. Thus every predeclared A2 check
passes, while exact-edge Gate A stays failed. This result admits an audit of
the coordination layer, not an end-to-end deployment claim: the physical
cold-start publication of the new candidate fields is still absent.

Reproducible report:
`results/architecture_v2_scale_k6q6_local_candidate_value_lu4_lq4_r1_gate_a2_holdout10/summary.json`.

### Finite-round coordination attribution

A deterministic fail-closed coordinator was implemented with target prices,
Tx/Rx role prices, owner persistence and one to four communication rounds. A
separate common-graph control lets every UAV independently run the same
role-partition enumeration and receiver-capacity dynamic program after
candidate gossip. The latter is not a deployable scalable algorithm; it is a
control that separates information transport from combinatorial inference.

| held-out controller | steady | weak3 | worst | CVaR | QoS feasible |
|---|---:|---:|---:|---:|---:|
| candidate-restricted teacher reference | 0.880 | 0.769 | 0.551 | 0.211 | 0.40 |
| candidate Student + central feasible projection | 0.877 | 0.764 | 0.558 | 0.197 | 0.50 |
| one-round price protocol | 0.740 | 0.502 | 0.266 | 0.053 | 0.20 |
| three-round price protocol | 0.727 | 0.494 | 0.267 | 0.073 | 0.10 |
| replicated common-graph computation | 0.875 | 0.752 | 0.543 | 0.197 | 0.50 |
| replicated computation + target queue | 0.878 | 0.758 | 0.542 | 0.197 | 0.50 |

The central Student projection shows that the frozen local edge scorer is not
the principal bottleneck. The price protocol selects only about seven edges
per resolve and loses approximately `0.29` mean worst; more rounds are not
monotonic. Conversely, replicated computation is only `0.0072` below the
candidate-restricted teacher reference in worst, although its CVaR gap of
`0.0143` narrowly fails the `0.01` gate. This reference is not a mathematical
upper bound: the Student plus central feasible projection reaches `0.558`
worst, above the fixed teacher's `0.551`, because its edge score and/or
optimization target differs. Adding a target queue does not improve the tail
and is rejected.

Candidate gossip is also not the steady-state bottleneck. Across 300 nonempty
resolve frames, one physical-neighbour flooding round gives a common candidate
view and identical edge set in `0.9967` of frames; rounds two to four add no
measurable agreement. Cold-start frames remain fail-closed (`3.23%` of
resolves), and the negotiation payload is only charged by an offline bit
estimate rather than the environment channel model.

The evidence therefore rejects further price-step, queue-weight or reward
tuning. The next architecture target is a shared, finite-round message-passing
or learned-unrolled approximation of the replicated role/owner solver. Its
primary supervision must use owner, equivalent target evidence and objective
gap; exact-edge imitation is auxiliary because Gate A2 exposes many equivalent
tie solutions. Owner supervision is set-valued when multiple owners are
evidence/objective equivalent, and evidence recall is paired with an edge/bit
cost so that selecting every candidate is not a trivial optimum. The
deterministic projection may repair only single-role,
unique-owner, receiver-capacity and 1 W simplex violations; it must not hide a
dynamic program, MILP or another full combinatorial search.

This defines two separate gates. Gate C1 uses the offline common candidate
graph and requires worst/reference gap at most `0.02`, CVaR/reference gap at
most `0.01`, zero hard violations, and a low projection repair rate (target
`<5%`). It also reports equivalent-evidence recall, owner accuracy, objective
gap and per-round convergence. Gate C2 is attempted only after C1 and charges
the actual bit budget, communication power, delay, loss, quantization and
fail-closed cold start. This is ADMN-inspired local reasoning rather than a
direct reuse of ADMN.

Passing C1/C2 would only recover the present candidate-reference performance.
It cannot by itself meet Medium: the held-out reference is only `0.551` mean
worst with `0.40` QoS feasibility. Geometry, cross-scale value calibration and
slow-timescale commitment remain separate post-coordination bottlenecks.

Implementation and reports:

- finite-round coordinator: `uav_isac/coordination/finite_round_hyperedge.py`;
- replay entry point: `tools/run_distributed_hyperedge_negotiation.py`;
- price/gossip audit:
  `results/architecture_v2_scale_k6q6_price_protocol_gossip_audit_holdout10/summary.json`;
- replicated-computation control:
  `results/architecture_v2_scale_k6q6_replicated_consensus_holdout10/summary.json`.

The frozen v1 reports retain the historical JSON field
`candidate_upper_minus_protocol`. Future v2 runs emit
`candidate_reference_minus_protocol`; this schema correction does not alter
the recorded measurements.

### Gate C1 factor-graph screening

A shared-parameter coordinator was then implemented over UAV nodes, target
nodes and directed `(Tx,Rx,target)` hyperedge nodes. Its decoder performs only
role filtering, unique-owner selection, receiver-capacity trimming and
deterministic ties; it contains no role enumeration, dynamic program or MILP.
The first ten seeds provide eight fitting and two validation episodes. The
second ten seeds were independent for the first screen, but repeated model
comparisons subsequently make them a development holdout rather than a valid
final test set.

| development screen | parameters | steady | weak3 | worst | CVaR | feasible | projection repair |
|---|---:|---:|---:|---:|---:|---:|---:|
| three-round factor graph | 105k | 0.840 | 0.684 | **0.446** | 0.124 | 0.40 | 0.176* |
| one-shot joint logits | 105k | 0.806 | 0.633 | 0.408 | 0.124 | 0.40 | 0.181 |
| three-round mean field | 105k | 0.828 | 0.662 | 0.417 | 0.124 | 0.40 | 0.157 |
| compact two-round model | 12k | 0.815 | 0.652 | 0.418 | 0.085 | 0.30 | 0.130 |
| cross-target global context | 124k | 0.838 | 0.684 | 0.446 | 0.124 | 0.40 | 0.274 |
| set-valued owner/evidence supervision | 105k | 0.806 | 0.621 | 0.384 | 0.121 | 0.30 | 0.108 |
| joint-certificate listwise supervision | 105k | 0.827 | 0.663 | 0.411 | **0.157** | 0.30 | 0.116 |

`*` The initial checkpoint's repair rate was recomputed with the final common
definition, in which a proposal already respects the predicted role and owner.
Its frozen original report used a stricter historical proposal definition.

The replicated teacher reference is `0.875/0.752/0.543/0.197/0.50`. The best
coordinator therefore remains `0.097` below the reference in worst and `0.073`
below it in CVaR, with equivalent-evidence recall `0.731` and repair rate
`0.176`. Gate C1 fails and Gate C2 is not admitted.

Counterfactual head replacement localizes the failure to the joint decision:
teacher roles alone give worst `0.412`, teacher owners alone `0.439`, and both
together `0.538`. Hence neither another independent classification head nor a
fixed receiver cardinality is justified. Set-valued per-target evidence also
improves owner consistency and repair rate while reducing closed-loop worst;
independent target equivalence is not sufficient for globally composable
role-owner equivalence.

The next admissible C1 experiment must first extend the replicated teacher to
emit a joint certificate: latent role partition, owner assignment, per-target
objective values and a set of near-equivalent feasible structures. A model can
then learn a joint structure energy or local feasible improvement, rather than
independent exact-edge/owner cross-entropies. Additional pooling, coupling
strength sweeps and Gate C2 physical-channel tests are stopped.

That certificate interface was subsequently implemented. Across all 300
training resolve frames, the `0.95` worst-and-sum filter retains `1.99`
structures per frame on average; `56.7%` of frames have more than one joint
certificate and a different latent role partition, while `54.3%` have a
different receiver-owner assignment. These are complete feasible plans, so
their components are never mixed independently during supervision.

Best-of-set/listwise training improves development CVaR from `0.124` to
`0.157` and reduces projection repair from `0.176` to `0.116`, but mean worst
falls from `0.446` to `0.411`; the reference is `0.543/0.197` for worst/CVaR.
Thus joint equivalence is real and useful for tail consistency, but a one-shot
amortized classifier still does not reproduce the combinatorial solution.
Gate C1 remains failed and no fresh final-test seeds or Gate C2 channel run are
consumed.

The next admissible method is no longer another classifier. It should learn
the value of locally feasible role/owner exchanges from the certificate set,
apply only positive-gain non-conflicting exchanges for a fixed number of
message rounds, and verify the objective after every round. This preserves a
distributed evolutionary path from a feasible initial plan and exposes where
each improvement arises.

Implementation and screening artifacts:

- model and lightweight projection:
  `uav_isac/coordination/factor_graph_coordinator.py`;
- training/audit entry point: `tools/train_factor_graph_coordinator.py`;
- first-screen report:
  `results/architecture_v2_scale_k6q6_factor_graph_gate_c1_screen/summary.json`;
- equivalent-supervision rejection:
  `results/architecture_v2_scale_k6q6_factor_graph_gate_c1_equivalent_screen/summary.json`.
- joint-certificate exporter: `tools/export_replicated_plan_certificates.py`;
- ten-seed certificate audit:
  `results/architecture_v2_scale_k6q6_joint_certificate_train10/summary.json`;
- listwise-certificate rejection:
  `results/architecture_v2_scale_k6q6_factor_graph_gate_c1_certificate_screen/summary.json`.

### Gate C1.5 oracle feasible-neighborhood audit

Before training another network, an oracle best-improvement audit tested
whether feasible local search has enough headroom. Every intermediate plan is
checked against the candidate graph, single Tx/Rx role, unique receiver owner,
receiver capacity, and target-pair limits. A move is accepted only when the
same lexicographic edge-value objective used by the replicated reference
strictly improves. The neighborhoods are same-owner edge replacement (N1),
atomic Tx/Rx role exchange with support rebuilding (N2), owner migration with
support rebuilding (N3), and exact repair over 2--3 UAVs and 1--2 weak targets
(N5). N5 subsumes the required atomic chain behavior, so a weaker standalone
N4 chain was not needed for this headroom decision.

| initialization | zero step | N1 | N1+N2 | N1+N2+N3 | +N5 |
|---|---:|---:|---:|---:|---:|
| factor graph + projection | 0.446 | 0.446 | 0.479 | 0.483 | **0.544** |
| role first | 0.147 | 0.147 | 0.384 | 0.515 | **0.529** |
| previous plan + required repair | 0.072 | 0.072 | 0.424 | **0.542** | 0.537 |

The replicated reference is `0.543` mean worst. From the factor-graph
initialization, N5 gives
`steady/weak3/worst/CVaR/feasible = 0.871/0.745/0.544/0.197/0.50`, recovers
`100.5%` of the initial combinatorial gap, accepts `1.66` moves per resolve on
average, and keeps every exact proxy-objective trajectory monotone. It passes
Gate C1.5, but evaluates `492.3` candidate moves per resolve and is therefore
an oracle search bound rather than a deployable coordinator.

The temporal warm start is substantially cheaper. Previous-plan initialization
with N1--N3 reaches `0.875/0.754/0.542/0.197/0.50`, accepts only `0.58` moves,
and evaluates `41.3` candidates per resolve. This supports a hierarchical
implementation: use the previous feasible plan and light N1--N3 repair on
ordinary frames, and invoke block N5 only after a deficit, residual, or ranking
confidence trigger. Different initializations can all approach the reference
once the neighborhood is sufficiently expressive, but their search costs are
not equivalent.

Monotonicity here applies to the frozen Student edge-value proxy used by the
replicated reference, not automatically to realized episode detection. The
slight decrease from previous-plan N1--N3 (`0.542`) to +N5 (`0.537`) illustrates
this calibration gap. Gate C1.6 must therefore report both proxy regret and
realized-detection regret.

The next learning target is local block/action ranking, not full-plan or
single-edge classification. A fail-closed executor should rank candidates,
exactly verify only Top-M, and accept the first positive feasible action. It
must compare learned ranking against random ranking, a physical heuristic, the
full-neighborhood oracle, and the replicated reference. Required diagnostics
are Top-1 positive-gain accuracy, Top-3 positive-action coverage, listwise
regret, exact-verification count, and final worst/CVaR gap. Gate C1.7 adds
Hold-5 and state evolution; Gate C2 remains blocked and fresh final seeds remain
unconsumed.

Implementation and report:

- feasible local-search engine:
  `uav_isac/coordination/local_exchange_oracle.py`;
- Gate C1.5 audit entry point: `tools/audit_oracle_local_search.py`;
- all-initialization report:
  `results/architecture_v2_scale_k6q6_oracle_local_search_gate_c1_5_all_initials/summary.json`.

### Gate C1.6 learned ranking and event-triggered cold start

Before exporting ranking labels, the acceptance predicate was corrected to
exclude deterministic index-only tie improvements. The index suffix now orders
multiple positive moves deterministically, but a switch is accepted only when
the worst, capped-deficit, total-value, or edge-count prefix strictly improves.
This prevents the learner from assigning value to physically equivalent index
changes.

Each move is represented by 29 cardinality-agnostic aggregate features: move
type, affected-target fraction, role/owner and edge changes, local frozen-edge
value deltas, and current scarcity statistics. The descriptor excludes the
post-move global minimum and exact lexicographic verification key. A shared
6,210-parameter MLP was trained on the first ten seeds with group-wise listwise
ranking and positive-gain classification.

For previous-plan initialization and N1--N3, the development closed-loop
screen is:

| ranker | Top-M | worst | CVaR | exact verifications/resolve |
|---|---:|---:|---:|---:|
| full-neighborhood oracle | all | 0.542 | 0.197 | 41.29 |
| random, three-run mean | 5 | 0.481 | 0.136 | 5.85 |
| total-gain heuristic | 5 | 0.373 | 0.013 | 5.43 |
| scarcity heuristic | 3 | 0.514 | 0.140 | 3.68 |
| scarcity heuristic | 5 | **0.538** | 0.177 | 5.80 |
| learned | 1 | 0.388 | 0.059 | 1.40 |
| learned | **3** | **0.526** | **0.197** | **3.66** |
| learned | 5 | 0.526 | 0.197 | 5.86 |

Learned Top-3 gives
`steady/weak3/worst/CVaR/feasible = 0.874/0.751/0.526/0.197/0.50`. Its worst
is `0.016` below the local oracle, its CVaR is identical, and simulated exact
verification falls by `90.8%`. It outperforms scarcity at the same Top-3
budget. Top-5 adds verification without changing the realized trajectory, so
the warm-maintenance Gate C1.6 checkpoint is frozen at Top-3.

Cold N5 ranking does not generalize in the same way. From the factor-graph
initial plan, scarcity Top-5, learned Top-3, and learned Top-5 obtain worst
`0.485`, `0.475`, and `0.498`, respectively, versus `0.544` for the N5 oracle.
The pure learned N5 path is therefore rejected. Its strong static held-in
ranking metrics were misleading because an early suboptimal accepted move
changes all subsequent candidate groups; group accuracy is not path regret.

An event-triggered hybrid isolates this problem. Exact N5 is used only at the
first resolve of each episode (`10/300 = 3.33%`); the remaining 290 resolves
use previous-plan learned Top-3. It obtains
`0.870/0.744/0.538/0.197/0.50`, only `0.0053` below the replicated reference in
worst. Effective exact evaluations average `21.05` per resolve, a `55.5%`
reduction relative to full labels along the same paths.

This is evidence for a hierarchical coordinator, not a claim that pure learned
coordination is complete: deterministic local initialization handles rare cold
starts, while learned ranking maintains a feasible temporal plan. Gate C1.7
must now re-run the environment with Hold-5, switching cost, Actor/motion state
evolution, and event triggering. Frozen-trace replay cannot establish those
closed-loop claims, and Gate C2 plus fresh final seeds remain blocked.

Implementation and reports:

- move features: `uav_isac/coordination/local_move_ranker.py`;
- shared scorer: `uav_isac/coordination/learned_move_ranker.py`;
- label export: `tools/export_local_move_ranking_dataset.py`;
- training: `tools/train_local_move_ranker.py`;
- ranked replay: `tools/audit_ranked_local_search.py`;
- hybrid replay: `tools/audit_hybrid_local_search.py`;
- warm learned report:
  `results/architecture_v2_scale_k6q6_ranked_local_search_gate_c1_6_learned_dev10/summary.json`;
- rejected cold learned report:
  `results/architecture_v2_scale_k6q6_ranked_local_search_gate_c1_6_learned_cold_n5/summary.json`;
- accepted offline hybrid report:
  `results/architecture_v2_scale_k6q6_hybrid_local_search_gate_c1_6/summary.json`.

### Gate C1.7a dynamic state re-evolution

The C1.6 controller was integrated into the actual evaluation loop. Actor
inference, UAV and target motion, received Token history, local observations,
Student edge values, and the `L_u=4,L_q=4` candidate graph now re-evolve every
frame. Structure resolution retains the deployed Hold-5 schedule; event
triggering and switch costs remain disabled so this gate isolates state
re-evolution. The deployment-ranked path computes exact objective keys only
for Top-M candidates. Full-neighborhood exact labels are no longer evaluated
behind the reported verification count.

The paired ten-seed development results are:

| controller | steady | weak3 | worst | CVaR | feasible | exact/resolve | ms/resolve |
|---|---:|---:|---:|---:|---:|---:|---:|
| previous only | 0.754 | 0.565 | 0.251 | 0.001 | 0.20 | 19.75 | 8.21 |
| dynamic Oracle N1--N3 | 0.882 | 0.771 | 0.531 | 0.138 | 0.50 | 50.97 | 11.10 |
| learned Top-3 | 0.878 | 0.763 | 0.507 | 0.138 | 0.40 | 23.01 | 15.24 |
| **learned Top-5** | **0.882** | **0.772** | **0.533** | **0.138** | **0.50** | **25.10** | **14.73** |
| replicated candidate reference | 0.915 | 0.832 | 0.632 | 0.236 | 0.50 | 85.81 | 96.66 |

Top-3 fails the registered `0.02` dynamic-Oracle worst-gap gate because seed
483 loses `0.259` episode worst. Top-5 exactly recovers that seed and produces
a mean worst gap of `-0.0014`, zero CVaR gap, equal QoS feasibility, and a
`50.75%` exact-verification reduction. It therefore becomes the frozen dynamic
maintenance budget and Gate C1.7a passes. Every returned structure is checked
against candidate, role, owner, receiver-capacity, and target-cardinality
constraints; the maximum 1 W projection error is `4.44e-16 W`.

This is a local-maintenance result, not closure of global coordination. The
replicated candidate reference remains ahead by `0.033/0.060/0.099/0.098` in
steady/weak3/worst/CVaR. The current Python learned executor is also `1.33x`
slower than the dynamic local Oracle despite fewer exact verifications, because
move enumeration, feature construction, and small CPU MLP calls dominate. A
runtime-speedup claim is therefore rejected until vectorization or incremental
candidate updates are measured. Fresh final seeds remain untouched.

Implementation and report:

- dynamic stateful controller:
  `uav_isac/coordination/dynamic_local_search.py`;
- dynamic CLI integration:
  `scripts/run_mappo.py`;
- gate summarizer:
  `tools/summarize_dynamic_local_search_gate.py`;
- formal report:
  `results/architecture_v2_scale_k6q6_dynamic_local_search_gate_c1_7a/summary.json`.

### Gate C1.7b periodic N5 rebootstrap headroom

The remaining `0.099` mean-worst gap motivated a direct test of a two-timescale
controller: learned Top-5 maintains the temporal plan, while an exact N5 pass
periodically attempts a larger structural reconfiguration. The implementation
supports `off`, `periodic`, and `oracle` rebootstrap modes. Rebootstrap starts
from the already maintained feasible structure, uses the current public frozen
Student values and candidate graph only, preserves the current role vector, and
accepts only a strict positive public-proxy improvement. No replicated MILP,
true geometry, or hidden centralized evidence is exposed to this path.

Under the deployed Hold-5 schedule, `Periodic-N5(H=5)` checks N5 at every
eligible warm resolve, so it is also the maximum-frequency N5 headroom test.
The paired ten-seed development results are:

| controller | steady | weak3 | worst | CVaR | feasible | exact/resolve | ms/resolve |
|---|---:|---:|---:|---:|---:|---:|---:|
| local-only learned Top-5 | 0.882 | 0.772 | 0.533 | 0.138 | 0.50 | 25.10 | 14.73 |
| **periodic N5, H=5** | **0.915** | **0.831** | **0.573** | **0.178** | **0.50** | **170.00** | **85.86** |
| replicated global reference | 0.915 | 0.832 | 0.632 | 0.236 | 0.50 | 85.81 | 96.66 |

Periodic N5 improves mean worst by `0.0404`, but its paired episode-bootstrap
95% interval is `[-0.0835, 0.2097]`, and the registered `worst>=0.60` gate
fails. It attempts rebootstrap on `93.55%` of resolves, accepts only `3.79%` of
attempts, and raises exact verification from `25.10` to `170.00`. Using a
`1e-4` material-difference tolerance, three episodes improve, four tie, and
three degrade. The strongest gain is seed 989 (`+0.7065`), while seed 483
degrades by `-0.2957`. Thus a positive instantaneous public-proxy move is not
a reliable label for multi-frame physical worst.

H=10 and H=20 are stopped rather than tuned: reducing the frequency cannot
establish N5 headroom that the every-eligible-resolve test lacks. A learned
trigger is also deferred because it would learn labels from a misaligned
reconfiguration objective. The next gate must replay accepted/rejected N5
moves from a common state with common random numbers and frozen subsequent
actions, reporting proxy gain, one-frame and five-frame physical worst gain,
sign agreement, and regret. If the proxy remains unreliable, the structural
path changes to learned deficit-block selection plus exact block LNS rather
than wider Top-M ranking or Hold/switch-cost scans. Fresh final seeds remain
untouched.

Implementation and report:

- rebootstrap implementation and diagnostics:
  `uav_isac/coordination/dynamic_local_search.py`;
- CLI and evaluation aggregation:
  `scripts/run_mappo.py`, `uav_isac/agents/trainer.py`;
- gate summarizer:
  `tools/summarize_rebootstrap_headroom_gate.py`;
- formal report:
  `results/architecture_v2_scale_k6q6_rebootstrap_headroom_gate_c1_7b/summary.json`.

### Preliminary Gate D0/D1 N5 counterfactual audit

The rebootstrap failure does not by itself distinguish candidate omission,
proxy misranking, and closed-loop persistence. A same-state replay instrument
was therefore added. It stores the exact public Student value/candidate pair
used by the dynamic coordinator, enumerates either the deployed N5 target
blocks or every singleton/target pair, and forces one candidate into a deep-
copied pre-P0 environment. All branches share actions, channel state, target
motion, detector noise, and persistent RNG streams. The seed-483 smoke has
zero UAV/target geometry error across every replayed branch.

At the first proxy-positive event (seed 483, frame 75), the atomic one-frame
ceiling is:

| candidate pool | candidates | worst gain | weak3 gain | steady gain |
|---|---:|---:|---:|---:|
| deployed proxy-weak target blocks | 209 | +0.082 | +0.180 | +0.090 |
| diagnostic all-target blocks | 2046 | +0.323 | +0.412 | +0.206 |

The all-target physical Oracle is outside the deployed pool and adds `0.2406`
one-frame worst headroom. Thus proxy-weak target-block generation can omit an
important repair even when the UAV block, role, owner, and edge reconstruction
machinery is unchanged. Proxy acceptance is also imperfect: 3 of 13 atomic
proxy-positive candidates reduce physical same-frame worst (`23.1%` false
acceptance). This does not imply total proxy failure, because the proxy Top-1
is also the physical Top-1 within the deployed pool at this event, and all ten
physically positive deployed-pool candidates are proxy-positive. The actual
eight-round rebootstrap accepts only that single atomic move here, excluding
sequential overshoot as the immediate cause.

Paired local-only and Periodic-N5 traces on the first five development seeds
then isolate the closed-loop response. For seed 483, every structure and
physical PD is exactly equal before frame 75. Starting at the accepted event:

| horizon | mean-worst delta | temporal-min delta | bottom-20% target-frame delta |
|---|---:|---:|---:|
| H=1 | +0.0821 | +0.0821 | +0.1749 |
| H=5 | +0.0555 | -0.0013 | +0.0438 |
| H=10 | -0.1049 | -0.0013 | -0.0036 |

Target states remain bit-identical. Sensing weights first diverge at frame 76;
movement and UAV geometry diverge at frame 77. By frame 80, the periodic path
loses `0.4175` single-frame worst relative to local-only. The accepted N5 move
therefore has genuine immediate and five-frame mean benefit, but that benefit
does not survive the communication-assisted sensing/movement closed loop.

This is intentionally a preliminary event-level result. Only one all-target
pool has been fully enumerated, frozen-future-action H=5 replay is still
missing, all-target enumeration is not deployable, and an atomic pool is not a
sequential LNS ceiling. The next experiment samples a small stratified set of
accepted, missed, and deficit-spike events from the same development seeds and
adds the frozen-action control. Trigger training and learned block selection
remain blocked until those labels are reliable. Fresh final seeds remain
untouched.

Implementation and report:

- exact public-problem snapshot and one-shot audit move:
  `uav_isac/coordination/dynamic_local_search.py`;
- proxy-weak/all-target atomic enumeration:
  `uav_isac/coordination/local_exchange_oracle.py`;
- same-state CRN evaluator:
  `uav_isac/evaluation/n5_counterfactual_audit.py`;
- report builder:
  `tools/summarize_n5_counterfactual_gate.py`;
- preliminary report:
  `results/architecture_v2_scale_k6q6_n5_counterfactual_gate_d0_d1_preliminary/summary.json`.

### Gate D0.2--D0.3 frozen-controller mediation and deficit guard

The missing frozen-future-action control is now complete. Baseline and
Forced-N5 branches start from the same pre-step state and receive the same
recorded movement, Token, rate, mask, communication/sensing allocation, and
public Student graph. For seed 483/frame 75, prefix physical-PD replay error is
zero, the maximum observation error is `2.98e-8`, and the 1 W balance error is
at most `2.22e-16 W`.

The intervention frame is reported separately from frames `t+1..t+5`:

| seed 483 path intervention | future-5 mean worst delta | endpoint delta |
|---|---:|---:|
| full no-op controller frozen | +0.0956 | 0.0000 |
| recorded movement feedback only | +0.0896 | -0.0255 |
| recorded radio/resource/Student feedback only | +0.1462 | -0.0000 |
| all recorded feedback restored | -0.0444 | -0.4175 |

Thus the N5 structure is not intrinsically harmful at this event. Neither
feedback path alone reproduces the collapse; their nonlinear interaction is
`-0.1847`. The mixed paths are cross-world diagnostics, not additive natural
indirect effects, but they reject a one-head-only explanation and motivate a
joint transition-consistency mechanism.

Repeating the replay on four independent first accepted events exposes a
second failure class. Seed 291/frame 95 is bit-identical across controllers
before intervention, yet the accepted N5 changes same-frame worst from
`0.8401` to `0.5795`. Its complete proxy/all-target pools contain `172/1719`
candidates; both physical Oracles select no-op (`delta=0`), while the only
proxy-positive candidate is harmful (`delta=-0.2606`). This is proxy/no-op
misacceptance, not candidate omission. Seeds 566/frame 10 and 103/frame 15 are
stable-positive controls; seed 483 remains the feedback-reversal case.

A default-off diagnostic guard was therefore added: scheduled N5 is permitted
only while at least one public QoS estimate has positive deficit. On ten
development seeds it blocks only seed 291 and leaves the other nine episodes
unchanged:

| controller | steady | weak3 | mean worst | CVaR | feasible | exact/resolve |
|---|---:|---:|---:|---:|---:|---:|
| Periodic-N5 H=5 | 0.9155 | 0.8309 | 0.5730 | 0.1784 | 0.50 | 170.00 |
| + deficit guard | 0.9153 | 0.8306 | 0.5777 | 0.1784 | 0.50 | 144.14 |

This is a safe computational guard, not the performance contribution: mean
worst remains below `0.60`, and CVaR/feasibility do not improve. Moreover, the
environment-level QoS EMA cannot be advertised as a free decentralized
signal. Deployment requires the same guard to be reconstructed from
owner-local QoS Tokens or finite-round max/min consensus.

The resulting architecture decision is two-layered: (i) no-op-safe deficit
and value validation rejects unnecessary reconfiguration; (ii) crisis
reconfiguration needs a joint movement--radio/resource transition trust
region. Candidate-block coverage remains a separate crisis-event question.
Trigger training remains deferred. A protocol incident must be carried into
all future reporting: the first seed-291 full-pool command omitted the explicit
`selection` split. Its N5 filter produced zero events and the output was not
used for method decisions, but the ordinary evaluator still exposed the first
five default-test seeds (`795/747/105/860/2`). These seeds are quarantined and
cannot support later confirmatory/final claims; only an untouched locked
remainder or a newly preregistered replacement pool may be used.

Artifacts:

- frozen-controller replay:
  `tools/audit_n5_frozen_controller.py`;
- four-event and ten-seed report:
  `results/architecture_v2_scale_k6q6_n5_frozen_controller_gate_d0_2_d0_3/summary.json`;
- seed-291 full-pool audit:
  `results/architecture_v2_scale_k6q6_n5_cf_d0b_alltarget_seed291_selection/n5_counterfactual.json`;
- deficit-guard paired evaluation:
  `results/architecture_v2_scale_k6q6_rebootstrap_deficit_guard_d0_3/paired_eval.csv`.

### Gate D0.4 nested weak-target budgets and no-op safety

The fixed two-target N5 search universe was generalized to nested budgets
`B in {1,2,3,Q}`. Budget `B` only selects the `B` weakest public-proxy
targets from which singleton/two-target atomic blocks may be formed. It does
not increase the atomic block dimension, and `B=Q` remains an offline ceiling.
All candidates still belong to the delivered candidate graph and satisfy role,
owner, receiver-capacity, and per-UAV 1 W communication+sensing constraints.

No-op is now explicit in the physical headroom rule. A candidate must improve
physical worst while preserving a baseline steady/weak3 value that is already
below its floor, or remaining above the floor when the baseline is safe. Three
development crisis events give:

| B | total candidates | events with a QoS-safe gain | mean safe worst gain |
|---:|---:|---:|---:|
| 1 | 174 | 1/3 | +0.0274 |
| 2 | 616 | 1/3 | +0.0274 |
| 3 | 1204 | 1/3 | +0.0274 |
| Q=6 | 4615 | 1/3 | +0.1076 |

Seeds 103 and 566 have no safe reconfiguration even at `B=Q`: every worst
improvement further degrades weak3. Seed 483 does contain an omitted safe
candidate: its same-frame worst gain increases from `+0.0821` at `B=2` to
`+0.3227` at `B=Q`, with simultaneous steady/weak3 gains. However, under the
frozen no-op controller, the future-five mean gain changes only from `+0.0956`
to `+0.0984`, an incremental `+0.00283`, and becomes zero at the next ordinary
structure resolve. The all-target search costs `7.49x` as many candidates.
Uniform candidate expansion is therefore rejected.

The next defensible mechanism is confidence-constrained event-triggered LNS,
not a larger fixed Top-k. Each target owner should construct a lower detection
bound from delivered, delayed, quantized Tokens,

```text
P_lower(q,S) = sigmoid(P_logit_hat(q,S) - beta * uncertainty(q,S)),
```

where uncertainty is calibrated from event-held-out residuals and increases
with Token age, loss, and quantization error. A candidate can replace no-op
only if all protected target bounds remain safe and the bottleneck lower-bound
gain exceeds switching, bit, latency, and power costs. This is a distributed
robust feasible-set decision rather than reward shaping. It will not enter the
deployed controller until owner-local calibration data pass event-level—not
candidate-level—cross-validation.

Artifacts:

- report: `results/architecture_v2_scale_k6q6_n5_weak_budget_gate_d0_4/summary.json`;
- event replay: `tools/audit_n5_weak_target_budget.py`;
- aggregation: `tools/summarize_n5_weak_target_budget_gate.py`;
- generalized candidate enumeration:
  `uav_isac/coordination/local_exchange_oracle.py`.

### Gate D0.5--D0.6 dependency closure and transition certificate

Gate D0.5 corrected the locality contract.  Historical N5 changes owner on at
most two declared targets, but a role change calls a full support rebuild.  Of
the 4,615 candidates in the three crisis events, 4,595 (99.57%) change more
than two physical target supports; the mean affected-target count is 5.46 and
the maximum is Q=6.  Historical N5 is therefore a small proposal with a
multi-target dependency closure, not a two-target atomic action.

Two exact-local controls were added for mechanism diagnosis only.  The
`target_block` N5 freezes every edge outside the declared one/two-target block,
and N6 preserves the role partition while exchanging only local owner/support.
At B=Q they retain 46 and 72 candidates across the three events, respectively,
but neither exposes a targetwise-safe or tail-deficit-safe positive action.
The physical headroom depends on the role-induced closure rather than a cheap
owner-only exchange.

The identity-preserving targetwise constraint is also too restrictive for the
equal-priority max--min objective.  Define the lower-bound detection deficit
and its worst-k cumulative curve by

```text
d_q(S) = max(P_floor - P_lower_q(S), 0)
D_k(S) = sum of the k largest entries of d(S).
```

Requiring `D_k(S') <= D_k(S)` for every k preserves maximum deficit, every
discrete upper-tail deficit CVaR, and total deficit, while remaining invariant
to target labels.  The exact-physical audit rejects seeds 103 and 566 and
recovers the seed-483 immediate worst gains of +0.0821 for B<=3 and +0.3227
for B=Q.  This contract currently assumes equal-priority targets; unequal
service weights require a weighted-tail generalization.

Gate D0.6 then rejects a same-frame-only certificate.  The B=2 seed-483 move
satisfies immediate deficit dominance, but on its matched recorded closed-loop
path only one of five future frames dominates no-op.  The maximum worst-k
cumulative-deficit violation is 0.5147 and endpoint worst delta is -0.4175.
The B=Q move dominates on all five frozen-no-op frames, but no matched B=Q
feedback trace exists, so that path cannot support a closed-loop claim.

The next certificate target is the candidate-versus-no-op transition loss
`G[e,c,h,k]`.  One calibration example is one independent event,

```text
R_e = max_{c,h,k} (G_true[e,c,h,k] - G_hat[e,c,h,k]) / u[e,c,h,k].
```

A finite-sample split-conformal multiplier yields the joint upper loss
`G_hat + beta*u`; a candidate is fail-closed unless every horizon/tail entry is
non-positive after switching/resource costs.  This reduction preserves the
dependence among thousands of candidates from the same event and supports
post-selection within a frozen proposal pipeline.  Thirty events are a pilot,
not authorization to deploy or to consume final-test seeds.

Artifacts:

- locality/certificate-alignment report:
  `results/architecture_v2_scale_k6q6_n5_certificate_alignment_gate_d0_5/summary.json`;
- H=5 transition-tail report:
  `results/architecture_v2_scale_k6q6_n5_certificate_alignment_gate_d0_5/transition_tail_summary.json`;
- strict-local and N6 implementation:
  `uav_isac/coordination/local_exchange_oracle.py`;
- event-level conformal upper certificate:
  `uav_isac/evaluation/transition_certificate.py`;
- extended local-search/Student/coordinator regression: 69 passed.

### Gate D0.7 physical atomic commit and control-power reserve

Gate D0.7 removes the dimensionally invalid idea that a detection-probability
gain can directly offset a number of bits, seconds, joules, or watts in one
weighted sum.  Reconfiguration is now defined by an intersection of feasible
sets,

```text
A_safe = A_role/owner/capacity/power
       intersect A_link/deadline/version
       intersect A_H-step-tail.
```

Costs may rank candidates only after all three sets are satisfied.  They can
never compensate for an unreachable owner, a stale state version, a missed
deadline, or a violation of the per-UAV power simplex.

The dependency closure contains every changed-role UAV, both endpoints of
every toggled sensing edge, and the old/new owner of every affected target.
It is committed by a digest-bound prepare/vote/decision protocol.  The wire
layout counts the complete role, owner, edge, and all-target lower-bound
records.  The 16-bit probability lower bound is rounded downward; uncertainty
is rounded upward, so each quantizer adds at most `1/(2^16-1)=1.526e-5` in the
conservative direction.  For each physical link,

```text
R_ij = B_round log2(1 + SNR_ij),
latency_ij = packet_bits/R_ij + processing_delay.
```

Prepare and decision are one-sender broadcasts.  All votes are concurrent and
share the U2U bandwidth orthogonally.  Every link must meet the active SNR and
5 ms packet deadline, the sum of the three round durations must fit the 0.1 s
control frame, all participants must have the same state version, and every
UAV must satisfy

```text
P_comm,k + sum_q P_sense,kq <= 1 W.
```

Three representative positive, immediate-tail-safe seed-483/frame-75
candidates were replayed.  Their role closures all contain all six UAVs.
Under the allocation that produced the original detection labels, UAV 2 is
silent (`P_comm,2=0`), so its vote has zero SNR and every candidate correctly
aborts.  Reserving 0.25 W control power for closure participants makes the
transport feasible:

| candidate | immediate worst delta | wire bits | commit latency | RF energy |
|---:|---:|---:|---:|---:|
| 0 | +0.0821 | 1,419 | 1.948 ms | 0.914 mJ |
| 4 | +0.3227 | 1,419 | 1.948 ms | 0.914 mJ |
| 8 | +0.0821 | 1,454 | 1.985 ms | 0.923 mJ |

The reserve is not treated as free.  The communication/sensing powers are
reprojected onto the exact 1 W simplex and the same-state physical detector is
rerun.  In these three moves, UAV 2's reduced sensing allocation is not used by
the executed support, so the worst deltas remain unchanged and all three still
satisfy tail-deficit dominance; maximum power-balance error is `2.22e-16 W`.
This is event-specific mechanism evidence, not a general zero-cost claim.

The fixed 0.25 W reserve is only a feasible upper control.  A monotone
bisection now minimizes the common participant communication-power floor while
reprojecting the remaining power to sensing at every iterate.  At this event
the nominal minimum is `2.119e-6 W`: candidates 0/4 then need `6.170 ms` and
`0.824 mJ`, while candidate 8 needs `6.210 ms` and `0.834 mJ`.  The limiting
vote packet arrives at `4.9995 ms`, immediately below the 5 ms packet
deadline.  This is a geometry/model-specific lower bound, not a safe operating
point; deployment reserve margin must be calibrated from independent channel
events rather than selected arbitrarily from this event.

The resulting research contribution is at the resource-allocation and
distributed structural-control layer: target support, Tx/Rx roles, fusion
owner, communication rate/bit budget, and communication-versus-sensing power.
The waveform/channel layer remains an analytical fixed-waveform abstraction;
there is currently no waveform covariance, subcarrier, beam, or ambiguity-
function optimization.  A waveform co-design claim would therefore be out of
scope without a new signal model and a new inner solver.

The deployable candidate is now an event-triggered *control-reserve plus
atomic dependency-closure commit plus H-step tail certificate*.  It remains
default-off: one development event cannot establish transition coverage, and
the required independent event-level conformal calibration has not yet been
collected.

Artifacts:

- physical commit implementation:
  `uav_isac/coordination/dependency_commit.py`;
- shared communication link budget:
  `uav_isac/environment/communication.py`;
- event replay:
  `tools/audit_n5_dependency_commit.py`;
- seed-483 physical report:
  `results/architecture_v2_scale_k6q6_n5_physical_commit_gate_d0_7/seed483_frame75_commit.json`;
- expanded locality/transition/commit regression: 78 passed.

### Gate D0.8 event-balanced residual feedback and drift lock

Feedback is now incorporated through a two-timescale certificate rather than
an unconstrained online optimizer.  During one deployed epoch the residual
model and conformal multiplier are immutable.  An event's observed error is
available only to a later epoch, so the outcome cannot retroactively change
the certificate that authorized its own action.

For owner-observable feature vector `x` (Token age, delivered packet-loss
fraction, quantization step, switch fraction, normalized horizon and tail
index), the transition prediction is corrected by

```text
G_tilde[e,c,h,k] = G_hat[e,c,h,k] + f_theta(x[e,c,h,k]).
```

The ridge fit gives every independent event equal total weight,

```text
min_theta (1/E) sum_e (1/n_e) sum_i
          (G_true[e,i] - G_hat[e,i] - f_theta(x[e,i]))^2
          + rho ||theta||_2^2.
```

Consequently, copying one event's thousands of candidates does not increase
its statistical weight.  Model-training and calibration event IDs must be
disjoint.  Calibration still reduces each event to

```text
R_e = max_{c,h,k} (G_true - G_tilde)/u,
```

and uses the finite-sample conformal order statistic.  Thus feedback can
improve the center prediction without weakening the joint post-selection
upper bound.  With 5% miscoverage at least 19 independent calibration events
are needed for a finite multiplier; a 30-event collection is only a pilot
split, not a deployment study.

A second fail-closed loop monitors realized certificate exceedances.  If
`I_t = 1{R_t > beta}`, then for each fixed alternative `q > alpha`,

```text
E_t(q) = product_s (q/alpha)^I_s
                    ((1-q)/(1-alpha))^(1-I_s).
```

Under the explicit stable-regime condition
`P(I_t=1 | past) <= alpha`, each process is a non-negative
supermartingale.  Their mixture is an e-process; crossing `1/delta` locks the
controller to No-op, with Ville false-alarm bound `delta`.  This guarantee is
conditional on that sequential null and is not inferred from the current
single event.

Owner feedback is also physically charged.  For K=Q=6 and H=5, a bundled
feedback packet has 153 shared bits and 23 bits per `(target,horizon)` entry.
At seed 483/frame 75, owner 1 carries 24 entries and owner 2 carries 12, for
1,134 over-air bits.  The commit-only minimum reserve `2.119e-6 W` is
insufficient: the feedback path has `-1.165 dB` minimum SNR and `10.671 ms`
latency.  A fixed 0.25 W reserve succeeds in `1.551 ms` and `0.468 mJ`.

Proposer election, commit transport and delayed feedback were therefore
optimized jointly.  The nominal common reserve floor becomes `6.795e-6 W`
with coordinator 3.  Candidates 0/4 commit in `3.982 ms` and candidate 8 in
`4.022 ms`; all return the 1,134-bit feedback in `4.99988 ms`.  Feedback RF
energy is `0.338 mJ`; commit energy is `0.824/0.834 mJ`.  Reprojecting this
joint reserve and replaying the physical detector leaves the three immediate
worst gains and tail dominance unchanged.  Because the feedback packet again
sits on the nominal deadline, this is a lower bound, not an operating point;
channel-residual calibration must set the deployment margin.

The resulting novel mechanism is a bidirectional certified LNS loop: learned
local marginal values propose a dependency closure, deterministic physics
commits it atomically, owners return delayed bounded residuals, and only a
future event-disjoint epoch can update the predictor.  Statistical drift has
absolute No-op priority.  The mechanism remains default-off until independent
event data support both transition calibration and channel margin.

Artifacts:

- feedback model, e-process and physical feedback transport:
  `uav_isac/evaluation/certified_feedback.py`;
- joint commit-feedback event audit:
  `results/architecture_v2_scale_k6q6_certified_feedback_gate_d0_8/seed483_frame75_feedback.json`;
- feedback regression:
  `tests/test_certified_feedback.py`;
- combined locality/transition/commit/feedback regression: 84 passed.

### Gate D0.9 event-calibrated robust channel margin

The nominal joint reserve from Gate D0.8 lay on the 5 ms feedback deadline,
so an arbitrary fixed 3 dB margin would not be a defensible operating rule.
Gate D0.9 instead defines one unsafe physical error score per independent
event. Every link, protocol round and feedback packet inside event `e` is
kept dependent and reduced to

```text
R_channel,e = max_i { 0,
    (SNR_hat[e,i] - SNR_obs[e,i]) / r_snr,
    (delay_obs[e,i] - delay_hat[e,i]) / r_delay }.
```

The positive resolution scales `r_snr` and `r_delay` must be fixed before
calibration. They only normalize physical units; SNR and seconds are never
added as economic substitutes. Split conformal calibration is performed over
event scores, not packets. Replicating thousands of links in one event
therefore cannot fabricate sample size.

The conservative SNR is applied before the Shannon calculation,

```text
SNR_robust,dB = SNR_hat,dB - beta r_snr,
R_robust = B log2(1 + 10^(SNR_robust,dB/10)),
delay_robust = bits/R_robust + processing + beta r_delay.
```

Prepare/vote/decision latency, feedback latency, RF energy, proposer election
and the common communication-power floor are all recomputed under this same
realization. Infinite finite-sample quantiles produce zero rate/infinite
delay and fail closed.

A risk-accounting correction is important. Two separately calibrated 5%
certificates provide only a direct 10% union-bound guarantee. Bonferroni
allocation to 2.5% each would require at least 39 calibration events for a
finite split-conformal order statistic, so the proposed 30-event pilot would
be insufficient. The implemented preferred construction uses one score

```text
R_joint,e = max(R_transition,e, R_channel,e)
```

and one frozen multiplier for both transition and physical margins. This
gives simultaneous marginal coverage at the selected miscoverage level
without assuming independence between detector and channel error. The
residual model's training event IDs remain disjoint from joint-calibration
event IDs. Model, multiplier and physical resolution scales form one atomic
feedback epoch; realized data can update only a later epoch.

Before collecting those events, seed 483/frame 75 was replayed only as a
development sensitivity audit. For diagnostic `beta = 0,1,2,3,4,6` with
`r_snr=1 dB` and `r_delay=0.1 ms`, the minimum common commit-plus-feedback
reserve was monotone:

| beta | SNR/delay margin | minimum common RF floor |
|---:|---:|---:|
| 0 | 0 dB / 0 ms | 6.795 microW |
| 1 | 1 dB / 0.1 ms | 8.876 microW |
| 2 | 2 dB / 0.2 ms | 11.608 microW |
| 3 | 3 dB / 0.3 ms | 15.201 microW |
| 4 | 4 dB / 0.4 ms | 19.932 microW |
| 6 | 6 dB / 0.6 ms | 34.424 microW |

At `beta=6`, candidates 0/4 commit in 5.753 ms and candidate 8 in
5.804 ms; feedback remains just below its 5 ms link deadline because it is
the active bisection constraint. Feedback minimum robust SNR is 4.943 dB,
feedback RF energy is 0.417 mJ, and commit energy is 1.009/1.022 mJ.
The required floor is 5.07 times the nominal floor but only reduces the
largest per-target sensing allocation by 8.742 microW. Exact detector replay
preserves the immediate worst gains `+0.0821/+0.3227/+0.0821` and tail-deficit
dominance, with maximum power-balance error `2.22e-16 W`.

These beta points are a power-versus-uncertainty curve, not hand-selected
deployment margins. Deployment remains default-off until at least 30
independent development events support the frozen joint quantile and a later
disjoint validation confirms it. No new final test seed was consumed.

The present U2U transport still uses deterministic Friis geometry for both
prediction and delivery. It therefore cannot generate meaningful nonzero
observed-minus-predicted channel residuals by itself. The next gate must add a
separate, reproducible observed-channel/shadow-CSI stream and audit its Rician
normalization before the 30-event collection; otherwise a zero-residual
calibration would be circular. That physical observation model is validation
infrastructure, not a claim of waveform optimization.

Artifacts:

- event-level channel and joint calibration:
  `uav_isac/evaluation/channel_margin.py`;
- joint frozen feedback/channel epoch:
  `uav_isac/evaluation/certified_feedback.py`;
- robust Shannon and protocol integration:
  `uav_isac/environment/communication.py`,
  `uav_isac/coordination/dependency_commit.py`;
- development margin curve:
  `results/architecture_v2_scale_k6q6_robust_channel_gate_d0_9/seed483_frame75_margin_curve.json`;
- isolated regression: 454 non-belief tests plus 14 belief-calibration tests
  passed. A combined Windows process still encounters the existing MKL
  native abort in `numpy.linalg.eigvalsh`, so the suites are run separately.

### Gate D0.10 deep audit: fail-closed statistics and replay provenance

Gate D0.10 did not assume that the D0.9 implementation was safe. It audited
the finite-sample statistic, physical latency decomposition, distributed
protocol identity, stochastic channel normalization and replay provenance.
Four substantive defects were found.

| Severity | Finding | Correction |
|---|---|---|
| critical | positive-infinite event scores were removed before the conformal order statistic | retain `+inf`; reject NaN/`-inf`; unresolved quantiles fail closed |
| critical | required transition/link observations with non-finite values could be silently omitted | required missing outcomes and undelivered packets map to `+inf` |
| high | total latency error included the serialization increase already explained by lower SNR | calibrate only excess queue/scheduling/processing delay after removing observed-SNR Shannon serialization |
| high | wire layouts counted epoch/digest bits but the certificate compared only state versions | require state-version, frozen-epoch and 64-bit proposal-digest agreement in commit and owner feedback |

For one event with required constraints `i`, the preferred physical score is

```text
d_excess,i = d_observed,i
             - bits_i/[B_i log2(1 + SNR_observed,i)]
             - d_processing,i,

R_channel,e = max_i {0,
    (SNR_hat,i - SNR_observed,i)/r_snr,
    (d_excess,i - d_excess_hat,i)/r_excess }.
```

The deployment budget then uses

```text
d_robust,i = bits_i/[B_i log2(1 + SNR_robust,i)]
             + d_processing,i + d_excess_hat,i + beta r_excess.
```

This decomposition is conservative without charging one fade twice. A
required undelivered packet has score `+inf`; it is not treated as padding.
Only a pre-registered structural mask may exclude a nonexistent item.

The finite-sample rule remains

```text
k = ceil((n+1)(1-alpha)),  beta = R_(k),  R_(n+1) := +inf.
```

Hence, with 19 calibration events and 5% miscoverage, one required missing
event makes the maximum order statistic infinite and the controller returns
No-op. With larger samples, a finite number of failures can only be tolerated
when allowed by the chosen event-level miscoverage; failures are never erased.
The guarantee still requires exchangeable independent events generated by the
same frozen proposal pipeline. The frozen feedback epoch now stores the
pipeline digest, and a runtime digest mismatch also forces No-op.

The physical audit found a separate Rician normalization error. The code
computed `sqrt(1/(K+1))` but did not multiply it into the NLoS Gaussian term.
The former model therefore had

```text
E|h|^2 = path_gain * (K/(K+1) + 1),
```

instead of `E|h|^2 = path_gain`. At `K=6 dB`, mean channel power was about
1.799 times the configured path gain. The missing factor is now restored and
seeded moment tests cover `K=-10,0,6,20 dB`.

Stepwise replay testing corrected an initially wrong invalidation claim.  The
reported `0.59446` detector discrepancy came from replaying the trace with a
different ranker and factor-graph checkpoint, which changed the controller
prefix.  It was not an interpretable physical residual.  Moreover, the audited
configuration has `ground_communication_enabled=false`, so its detection
Deflection does not call the Rician reporting-link branch.  With the exact
pipeline recorded in the trace manifest, seed 483/frame 75 agrees to `2.98e-8`
in local observations and exactly in physical `P_D`.  Therefore:

- the D0.9 event-level detection gains and deterministic U2U Shannon curve are
  not invalidated by this Rician helper correction;
- experiments that enable the ground/reporting stochastic link still require
  separate regeneration or replay because they do use the corrected helper;
- no replay residual may be interpreted unless the trace-producing pipeline
  is identical.

The audit tool now binds the trace, config, Student checkpoint, ranker,
factor-graph checkpoint, search mode, Top-k values, coverage fraction and
search-round counts to the sibling run manifest before simulator creation.
A deliberate checkpoint mismatch fails before residual interpretation and
writes no candidate output; the manifest-matched replay passes.  This
provenance gate is necessary because a controller-version mismatch can look
like a large physical-model error even when each simulator is internally
consistent.

Innovation audit. Joint UAV-ISAC power/resource allocation is established,
including multi-UAV communication/sensing power optimization
([Zheng et al., 2024](https://arxiv.org/abs/2410.02122)); UAV-swarm bistatic
ISAC with MARL is also established
([Atsu et al., 2025](https://arxiv.org/abs/2501.06454)). Conformal safety
filters and trajectory-level conformal control guarantees are likewise
established concepts
([Lindemann et al., 2024](https://arxiv.org/abs/2409.00536);
[Stamouli et al., 2024](https://proceedings.mlr.press/v242/stamouli24a.html)).
LNS over coupled multi-agent assignment is not new either
([Parimi and Williams, 2026](https://arxiv.org/abs/2603.28968)).

The defensible potential contribution is therefore not any component alone.
It is the system-level construction that combines learned owner-local marginal
values, dependency-closure LNS, physically charged digest-bound atomic commit,
one post-selection event score over transition and transport constraints,
delayed owner residuals, frozen-pipeline binding and an anytime No-op drift
lock. This is a plausible integration and certification novelty, but it must
be described as *potentially novel* until a systematic related-work search,
corrected-channel experiments and component ablations exclude close prior art.
It is still a resource-allocation/distributed-control contribution, not a
waveform-design contribution.

### Gate D0.46 exact QoS bottleneck decomposition and certified fast power layer

The 4/4-only success is no longer treated as sufficient evidence of scale
generality.  A new exact threshold-feasibility MILP and a finite-round
transport-aware controller separate three causes of 6/6 failure: sensing
power, reporting structure and geometry.  On the weakest resolved frame from
20 independent 6/6 episodes, recorded/fixed-power/joint-structure-power mean
worst `P_D` is `0.0713/0.4772/0.6130`; the corresponding QoS rates are
`0.00/0.35/0.55`.  Mean recoverable gains are `+0.4059` from power and
`+0.1358` from structure.  Nine of twenty weakest frames remain below 0.60
after joint same-geometry optimization and therefore require motion.

The deployed fast-layer candidate uses four Dantzig--Wolfe rounds, 6-bit
prices and binary16 owner feedback.  Across 20 seeds and 600 causal events it
raises event-mean worst `P_D` from `0.3762` to `0.6338`, with zero same-event
harm, physical-transport feasibility one, mean `1462.2 bit/event`, maximum
protocol latency `28.76 ms` in the `100 ms` control period, mean RF energy
`0.429 mJ/event`, and maximum 1 W balance error `4.44e-16 W`.

This is an architecture-development result, not a deployment certificate:
the coefficient interval still needs independent random-channel/model-drift
calibration and the new controller is not yet wired into the causal next-event
environment path.  The method and promotion gate are specified in
`docs/CURRENT_SYSTEM_STATUS.md` (Gate D0.7–D0.9).

Artifacts:

- fail-closed event statistics:
  `uav_isac/evaluation/transition_certificate.py`;
- orthogonal SNR/excess-delay score:
  `uav_isac/evaluation/channel_margin.py`;
- frozen pipeline and owner-feedback binding:
  `uav_isac/evaluation/certified_feedback.py`;
- epoch/digest-bound atomic protocol:
  `uav_isac/coordination/dependency_commit.py`;
- corrected Rician channel:
  `uav_isac/physical/channel.py`;
- corrected deep-audit report:
  `results/architecture_v2_scale_k6q6_deep_certificate_audit_gate_d0_10/summary.json`;
- provenance-matched event replay:
  `results/architecture_v2_scale_k6q6_deep_certificate_audit_gate_d0_10/seed483_frame75_revalidated.json`;
- isolated regression: 466 non-belief tests and 14 belief-calibration tests
  passed. No new final test seed was consumed.
