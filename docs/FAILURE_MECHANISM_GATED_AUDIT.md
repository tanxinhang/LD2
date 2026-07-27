# Failure-mechanism gated audit

Date: 2026-07-23

## Question

Before adding SMDP-MAPPO, COMA-style commitment credit, or another duplicate
penalty, test whether the observed failure has the controllability and causal
structure required by those mechanisms.

Policy under test:

- config:
  `config/exp_800_q4_u2u_hierarchical_multistatic_distributed_matching_hybrid50_paper_top1_eval.yaml`
- actor:
  `results/paper_top1_test100/best_restored.pt`
- diagnostic split: first 20 seeds of the fixed `stress` bank
- deterministic actor, 150 frames per episode
- movement hold: 5 frames
- bootstrap: 2,000 resamples of complete episodes

These stress seeds are used only for mechanism diagnosis. They are not used to
report the formal 100-seed test result.

## Audit 1: is movement overlap useful or wasteful?

For every movement boundary and UAV-target pair, remove all selected bistatic
edges incident on UAV \(k\) and recompute target detection:

\[
\Delta P_{D,kq}
=P_{D,q}(\mathcal E)
-P_{D,q}(\mathcal E\setminus\mathcal E_k).
\]

An overlapping responsibility is labelled ineffective only when two or more
UAVs move toward the same target and the audited UAV has
\(\Delta P_{D,kq}\le 10^{-4}\). This is different from the historical
`move_collision` flag, which labels every repeated target direction.

| Diagnostic | Result |
|---|---:|
| Movement-boundary samples | 580 |
| Historical duplicate-frame rate | 0.8741 |
| Ineffective fraction among duplicate responsibilities | 0.2359 |
| Effective multistatic-overlap target rate | 0.1483 |
| Same-role-overlap target rate | 0.0151 |
| Uncovered movement target rate | 0.3073 |
| Uncovered but still sensed by P0 rate | 0.3073 |
| Productive duplicate marginal \(P_D\) | 0.5331 |
| Ineffective duplicate marginal \(P_D\) | \(5.50\times10^{-7}\) |

Association with future five-frame improvement:

| Statistic | Estimate | episode-cluster 95% CI |
|---|---:|---:|
| Ineffective duplication vs future worst, Spearman | -0.2356 | [-0.3744, -0.0589] |
| Present-minus-absent future worst improvement | -0.0129 | [-0.0232, -0.0034] |
| Ineffective duplication vs future weak3, Spearman | -0.2736 | [-0.3746, -0.1555] |
| Present-minus-absent future weak3 improvement | -0.0066 | [-0.0106, -0.0031] |

The observational gate passes, but the table also shows why the old 0.87
collision rate must not be directly penalized: a material part is useful
multistatic overlap, and P0 often senses a target even when no UAV movement
vector points toward it.

Result file:
`results/temporal_credit_audit_stress20/paired_eval.csv`.

## Audit 2: does five-frame temporal credit fix worst credit?

Historical formal checkpoints stored the actor but not the learned standard
critic. The exact historical movement advantages therefore cannot be
reconstructed. To avoid presenting a random newly initialized critic as
evidence, the audit compares two explicitly labelled reward-only,
zero-baseline GAE proxies:

1. frame GAE applies \(\lambda\) on every simulator frame;
2. macro GAE first aggregates the five rewards caused by one held movement
   with micro discounting, then applies \(\lambda\) once per movement boundary
   and uses \(\gamma^5\).

| Future five-frame metric | Frame correlation | Macro correlation | Gain | gain 95% CI |
|---|---:|---:|---:|---:|
| worst improvement | -0.0562 | 0.0017 | +0.0579 | [-0.1175, 0.2113] |
| weak3 improvement | -0.3541 | -0.0865 | +0.2675 | [0.1880, 0.3574] |

The pre-registered worst gate required a gain greater than 0.10 with a
positive 95% lower bound. It fails. Macro credit may improve weak3 credit, but
the present evidence does not justify implementing an SMDP movement critic for
the paper's primary worst-QoS objective.

All newly produced MAPPO checkpoints now persist actor, critic, runtime state,
and critic type so that a future trained run can support an exact learned
advantage audit.

## Audit 3: is target commitment sufficiently controllable?

At every 25th frame, rotate the audited UAV and enumerate all four target
directions. Each branch starts from an identical simulator state. Other UAV
movements, message content, rate, communication power, and sensing allocation
are held fixed for five frames. This is a common-random-number physical
intervention, not a critic prediction.

| Diagnostic | Result | episode-cluster 95% CI |
|---|---:|---:|
| Interventions | 120 | 20 episodes |
| Candidate worst range | 0.0109 | [0.0049, 0.0197] |
| Candidate worst standard deviation | 0.0046 | [0.0020, 0.0083] |
| Actual-to-oracle worst gap | 0.0074 | [0.0017, 0.0162] |
| Actual choice is best | 0.2583 | [0.1833, 0.3417] |
| Oracle gap greater than 0.02 | 0.0500 | -- |

The action-controllability gate required candidate range LCB \(>0.05\) and
oracle-gap LCB \(>0.02\). It fails by a wide margin. The actor's target choice
is close to random in rank, but choosing the one-step oracle target has little
five-frame value. Therefore an explicit categorical commitment head plus COMA
would mostly improve classification of a weakly consequential action.

## Targeted causal check: can an ineffective duplicate be repaired?

The intervention sample contains 18 audited ineffective-duplicate nodes from
13 episodes.

| Conditional diagnostic | Result | episode-cluster 95% CI |
|---|---:|---:|
| Candidate worst range | 0.00130 | [0.00010, 0.00364] |
| Actual-to-oracle worst gap | 0.00107 | [0.00003, 0.00336] |
| Weak3 oracle gap | 0.00038 | [0.00002, 0.00113] |
| Oracle gap greater than 0.02 | 0/18 | -- |

This rejects the causal interpretation of Audit 1 at the tested five-frame
action horizon. Ineffective duplication is a symptom of hard geometry and weak
short-horizon movement controllability, not a repairable cause of the worst-QoS
gap. Adding a duplicate penalty is not supported.

Result file:
`results/target_choice_cf_stratified_stress20/paired_eval.csv`.

## Decision

Do not add any of the following to the current fixed-target system:

- a blanket movement-collision/duplicate penalty;
- an SMDP movement critic solely to improve worst QoS;
- an explicit stochastic commitment head plus COMA;
- a P0 teacher/DAgger layer as the main claimed contribution.

The three audits show that the present bottleneck is upstream of MAPPO credit
assignment: over one movement hold, changing a UAV's target direction has too
little effect on worst QoS. Additional credit machinery cannot create action
controllability.

The next scientifically defensible branch is a problem-formulation change:

1. enable target tracking and per-UAV private noisy sensing evidence;
2. make communication carry information unavailable to peers rather than
   mission-known target geometry;
3. move target/pair intent into distributed policies;
4. retain P0 only as a physical feasibility projection, not a global semantic
   decision maker;
5. retrain jointly and first require a significant Full-minus-no-U2U gap on
   unseen tracking/channel/scale conditions.

If that communication-necessity gate does not pass, stop architecture
development and write the current conservative paper around the validated
capacity-two matching and movement-consensus results.

## Follow-up: communication-value identifiability

The first corrected evidence Oracle gate has now passed. With selected edges,
movement, and power held identical, replacing the current centralized
deflection sum by the best receiver-local detector reduces stress-20 mean worst
from 0.6717 to 0.5169. The free centralized-fusion gain is +0.1547 with paired
95% CI [0.0940, 0.2201].

This changes the diagnosis: sensing evidence has substantial value, but the
environment currently supplies that value independently of Token transport.
The next task is therefore to make the global detection statistic conditional
on delivered U2U sufficient statistics before adding dynamic tracking or a new
MAPPO architecture. See `docs/COMMUNICATION_VALUE_IDENTIFIABILITY_GATE.md`.
