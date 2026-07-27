# Headwise credit assignment experiment

## Purpose

The joint ISAC policy produces four different decisions: UAV movement, token
content, communication rate, and sensing/communication power allocation.  The
legacy PPO objective summed all four log probabilities and multiplied the sum
by one scalar advantage.  Consequently, a communication reward could update
the movement policy, while a sensing or collision reward could rewrite the
token encoder.

This experiment separates policy credit without changing the distributed
execution interface.  Every UAV still acts from its local observation and
received tokens.

## Implemented separation

The actor and rollout buffer now keep four log-probability and advantage
streams:

| Head | Main training signal |
| --- | --- |
| Movement | Sensing/coordination performance and physical constraints |
| Message token | Delayed next-decision coordination progress |
| Communication rate | Communication QoS, rate use, silence penalty, and sender cost |
| Power allocation | End-to-end sensing/communication task trade-off |

Four CTDE-only critic heads estimate these returns.  The message signal is
backfilled by one decision because a token transmitted at decision `t` first
enters another UAV's observation at `t+1`.  Backfilling is masked at episode
boundaries.  Rate and power losses condition on the transmitted token but
detach that context, so they cannot alter token content through an unintended
gradient path.

The feature is guarded by `headwise_credit_enabled` and remains disabled in the
default configuration.

## Verification

Focused regression coverage checks that:

- the four component log probabilities exactly sum to the legacy joint log
  probability;
- four independent GAE streams are produced and consumed by PPO;
- an oracle-held movement action masks only movement credit;
- the final action in a truncated rollout receives no false same-frame message
  credit;
- the communication-rate loss updates the rate head but not the token-output
  head.

The focused suite passed 51 tests.  The full repository suite was not used as a
release gate because of the pre-existing MKL eig/SVD process abort.

## Smoke evaluation

The final causal implementation was trained with the experimental configuration
`config/exp_800_q4_u2u_headwise_credit.yaml` and evaluated on 20 independent
seeds.  This is a diagnostic smoke run, not a publication-grade multi-seed
training result.

| Method | steady | weak3 | mean worst | collision | bits/frame |
| --- | ---: | ---: | ---: | ---: | ---: |
| S1 isolated reward | 0.71909 | 0.62545 | 0.21386 | 0.94333 | 2304 |
| Headwise causal credit | 0.71651 | 0.62201 | 0.20056 | 0.94200 | 2304 |
| Difference | -0.00258 | -0.00344 | -0.01330 | -0.00133 | 0 |

All per-head PPO KL values remained small (movement 0.00330, message 0.00707,
rate 0.00611, resource 0.00011), so the regression is not explained by an
unstable PPO update.  Communication delivery remained 1.0 with about 1.97 ms
latency.

## Decision

Headwise credit fixes a real optimization defect, but it is not sufficient to
meet the deployment target.  The remaining message reward is still a shared
team signal: it tells the system that communication helped, but not which
sender or token was responsible.  The next structural experiment should add a
sender-specific counterfactual communication advantage (for example, comparing
the next decision with and without each sender's received token) while retaining
the headwise separation.  Increasing reward weights or communication volume
again would not resolve that attribution ambiguity.
