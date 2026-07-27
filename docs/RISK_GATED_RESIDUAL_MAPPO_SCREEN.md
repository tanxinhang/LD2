# Risk-Gated Residual MAPPO: Short-Budget Screening

## Purpose

Test whether MAPPO can improve the strongest frozen Top-1 policy without
destroying its learned communication, power-allocation, commitment, or
distributed matching behavior.

The acceptance gate was defined before the directional run: paired mean
`worst` improvement of at least `0.005` on the same ten test scenarios, with
no material `steady` or `weak3` regression. A passing result would be followed
by three training seeds and the formal 100-scenario protocol.

## Architecture

`RiskGatedResidualActor` freezes the complete foundation actor and trains only
a bounded movement adapter. The adapter uses local history, assignment
entropy, token availability, and the frozen policy latent. Its second revision
contains two residual paths:

1. a free two-dimensional correction;
2. a weak-target radial basis constructed from local target geometry and the
   communication-refined distributed responsibility assignment.

The learned risk gate controls the total intervention. The correction is
applied in normalized action space and hard-bounded. Centralized information
is used only by the training critic; execution remains decentralized.

## Verification

- The original 98 actor tensors were bitwise unchanged after PPO training.
- The adapter is exactly neutral at initialization.
- Rollout/update log-probability consistency error was `8e-6` in the real
  environment smoke test.
- 81 related regression tests passed after adding the directional branch.

## Screen 1: Free 2-D Risk Residual

Six PPO updates, ten common test scenarios per training seed.

| Training seed | Frozen worst | Residual worst | Delta | QoS delta | Mean action correction |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.843028 | 0.843045 | +0.000017 | 0.00 | 0.000039 |
| 123 | 0.842570 | 0.842590 | +0.000019 | 0.00 | 0.000024 |

The adapter learned, but its executed correction was too small to have a
meaningful effect.

## Screen 2: Weak-Target Directional Residual

Seed 42, six PPO updates, the same ten test scenarios.

| Metric | Frozen | Directional residual | Paired delta |
|---|---:|---:|---:|
| steady | 0.960757 | 0.961152 | +0.000395 |
| weak3 | 0.947676 | 0.948202 | +0.000526 |
| worst | 0.843028 | 0.844607 | +0.001579 |
| QoS feasible rate | 0.80 | 0.90 | +0.10 |

The mean gate was `0.22995`, the mean absolute directional scale was
`0.02221`, and the mean absolute executed correction was `0.001166` under the
hard `0.08` bound.

The paired `worst` result was heterogeneous: two improvements, two
regressions, and six effectively unchanged scenarios. The largest improvement
was `+0.0684`, while the two meaningful regressions were `-0.0297` and
`-0.0229`. The paired bootstrap 95% interval for mean `worst` delta was
`[-0.0112, 0.0182]`; the median delta was effectively zero. The QoS increase
came from one near-threshold scenario crossing `worst=0.6`, while the hardest
scenario deteriorated from `0.0802` to `0.0573`.

## Decision

The directional revision does not pass the predeclared `+0.005 worst` gate and
does not show a consistent tail improvement. Therefore the third training
seed and formal 100-scenario run are intentionally not launched. The adapter
is retained as an experimental implementation, not promoted to the proposed
main method or used as evidence of superiority.

The result indicates that the remaining limitation is not simply insufficient
MAPPO capacity. A local movement correction can rescue one boundary case but
redistributes risk across hard geometries. Any further algorithmic work should
optimize a distributional/tail constraint explicitly (for example a
Lagrangian CVaR critic with scenario-conditioned replay) rather than further
increasing this residual's gain.
