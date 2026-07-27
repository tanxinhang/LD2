# Paired-intervention causal contribution experiment

## Scope

This experiment adds a training-only Causal Contribution Predictor (CCP) to
the headwise MAPPO implementation. It does not change the deployed actor or
claim link-level traffic savings under the current one-hop broadcast model.

For a sparsely sampled active sender, the trainer clones the same post-step
environment twice. Both branches share the same simulation state and random
number stream. The factual branch retains the sender's delivered target tokens;
the intervention branch clears the sender's token content, metadata and
validity bits from every receiver observation. Both branches then execute the
next deterministic decentralized decision. Their per-target label is

`P_D(factual next decision) - P_D(do(sender token = null), next decision)`.

A lightweight MLP maps the sender's local observation, sampled target-token
payload and rate to the vector of per-target effects. A held-out correlation
and sign-accuracy gate prevents unreliable predictions from replacing PPO
message credit. Measured intervention labels remain authoritative. Tail weights
are a soft minimum over the current per-target detection probabilities.

## Verification

The focused joint-power, recurrent-log-probability, communication and reward
suite passes 54 tests. New coverage checks complete token/mask intervention,
the predictor's neutral initialization and gradient, and end-to-end collection
and PPO update with paired labels.

## Fair smoke result

Both methods use seed 42, the same 2,048-transition rollout, the same target-
token warm start and the same 20 independent evaluation seeds.

| Method | steady | weak3 | mean worst | collision | bits/frame |
| --- | ---: | ---: | ---: | ---: | ---: |
| Headwise causal-timing baseline | 0.71651 | 0.62201 | 0.20056 | 0.94200 | 2304 |
| Paired-intervention CCP | 0.71788 | 0.62384 | 0.21031 | 0.94233 | 2304 |
| Difference | +0.00137 | +0.00182 | +0.00976 | +0.00033 | 0 |
| Deployment target | 0.80000 | 0.70000 | 0.60000 | - | - |

The small worst-target improvement is not enough to pass any deployment gate
and is not a multi-training-seed significance claim.

## Attribution diagnostics

The rollout collected 64 paired labels. Their mean absolute per-target effect
was only `3.96e-5`. CCP held-out correlation was `0.198`, sign accuracy was
`0.286`, and the reliability gate therefore remained zero. Only measured,
non-negligible teacher labels affected message credit, covering `0.256%` of
actor samples. Thus the experiment mostly behaved like the headwise baseline.

An independent deployment ablation using the resulting actor then replaced all
transmitted token content with zero while preserving the communication
rate/power decisions:

| Content at evaluation | steady | weak3 | mean worst | collision |
| --- | ---: | ---: | ---: | ---: |
| Learned token | 0.71788 | 0.62384 | 0.21031 | 0.94233 |
| Zero token | 0.71780 | 0.62374 | 0.21054 | 0.93500 |

The negligible difference confirms that the current policy does not use token
content for task performance.

## Decision

CCP is numerically safe and its intervention labels are causally well-defined,
but causal attribution is downstream of causal influence. In the present
tracking-free task, target locations are mission-known. Moreover, with
`learn_roles=false`, the environment's centralized P0 solver globally chooses
the transmitter-receiver-target tuples. This removes much of the private
information and sensing decision authority that U2U tokens would otherwise
need to convey.

The next structural experiment should first create a communication-necessary
distributed sensing problem: expose only local sensing confidence/evidence,
move role/pair/target commitment out of the centralized P0 selection, and let
received tokens enter that local commitment explicitly. CCP should be retained
as the training-time attribution and audit layer after a no-token ablation
shows a non-zero performance gap. Increasing CCP capacity, reward weight or
teacher-query count before that change would fit noise rather than solve the
system bottleneck.
