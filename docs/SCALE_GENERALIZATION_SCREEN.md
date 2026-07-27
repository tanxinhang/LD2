# Controlled-Density Scale Generalization Screen

## Question

Does increasing the number of UAVs and targets reveal a larger advantage for
the proposed distributed communication-and-capacity architecture over its
structural baselines?

This screen preserves the earlier recommendation to replace mean-value MAPPO
with a risk-constrained, distributional and variable-cardinality method. It
tests whether scale alone is sufficient to rescue the current method before
that larger redesign is attempted.

## Protocol

Node and target density are approximately controlled by scaling the square
region side with `sqrt(K/4)`:

| Scale | Region | Test status |
|---|---:|---|
| 4 UAV / 4 targets | 800 m | Existing formal reference |
| 6 UAV / 6 targets | 980 m | Screened |
| 8 UAV / 8 targets | 1130 m | Prepared, not launched |

Independent geometry-stratified seed banks were generated for 6/6 and 8/8.
Each contains disjoint 20-seed selection, 60-seed confirmation, 100-seed test,
and 50-seed stress splits. The 6/6 screen uses the same first five selection
seeds for every variant.

The 4/4 foundation actor is migrated without PPO updates. Of its 98 actor
tensors, 89 retain the same shape at 6/6 and 8/8. Approximately 98% of scalar
parameters transfer, but nine scale-dependent tensors require migration or
reinitialization: communication variance, round identity input, communication
rate, communication power, per-target sensing power, and intent outputs.
Consequently this is a compatibility screen, not final evidence of zero-shot
scale generalization.

## 6/6 Results

Five paired selection seeds, zero PPO updates:

| Variant | steady | weak3 | worst | worst CVaR | QoS feasible | bit/frame |
|---|---:|---:|---:|---:|---:|---:|
| Full transferred DCB Top-1 | 0.8429 | 0.6858 | 0.2927 | 0.0836 | 0.20 | 1152 |
| No U2U communication | **0.8897** | **0.7794** | **0.3881** | 0.0675 | **0.40** | 0 |
| No capacity matching | 0.8441 | 0.6882 | **0.3821** | **0.1060** | 0.20 | 1152 |

Paired mean `full - baseline` deltas:

| Baseline | delta steady | delta weak3 | delta worst |
|---|---:|---:|---:|
| No U2U | -0.0468 | -0.0936 | -0.0954 |
| No capacity matching | -0.0012 | -0.0024 | -0.0894 |

The full system obeys the per-UAV unit power constraint exactly: aggregate
communication and sensing powers are 1.4961 W and 4.5039 W for six UAVs, with
maximum numerical balance error `2.22e-16` W. All six UAVs transmit and message
delivery is 1.0. Nevertheless, movement collision/duplicate-target rate is
0.992, identical to the no-U2U value to the reported precision. Communication
therefore consumes about 24.9% of every UAV's ISAC power without producing a
corresponding scale-level coordination benefit.

## Interpretation

The larger scenario does separate the methods, but in the wrong direction.
This falsifies the explanation that the current contribution merely looks weak
because the 4/4 scene is too small. The scale screen instead exposes three
limitations:

1. the communication and sensing action heads are cardinality-dependent;
2. communication-refined assignments trained at 4/4 do not remain calibrated
   at 6/6;
3. communication cost grows with the number of senders and targets while the
   observed movement conflict remains saturated.

## Decision and next gate

The projected-only and no-movement-consensus 6/6 ablations, and all 8/8 runs,
are stopped because both primary baselines already outperform the full method.
Running more seeds would estimate the magnitude of a method that has failed
the direction gate; it would not turn it into evidence of scalability.

Before resuming 8/8, the actor should be made cardinality-equivariant:

- shared per-target sensing-resource scorer instead of a `Q x (Qd)` linear
  output;
- pooled/set-based communication rate and total-power heads;
- identity-free round encoding or permutation-equivariant node embeddings;
- distributional target-risk critic and explicit CVaR/QoS constraints.

The minimum re-entry criterion at 6/6 is positive paired `worst` improvement
over both no-U2U and no-capacity baselines, with no `steady` regression and a
lower conflict rate. Only then should the prepared 8/8 protocol be launched.
