# Predictive two-frame and gradient-conflict ablation (2026-09-08)

## Changes

- Added causal two-frame residual features: current native state, previous-frame
  residual, and an episode-history validity flag. The first frame of every seed
  is explicitly zeroed, so no cross-episode or future state can enter training.
- Added a boundary-gated residual decoder for owner/transmitter structure.
- Kept the boundary power residual optional. The accepted structural ablation
  does not let the boundary decoder alter power logits; row-budget projection
  remains the only power feasibility mechanism.
- Added physical-anchor PCGrad diagnostics and an optional `pcgrad` mode. It
  projects away only the structural gradient component that conflicts with the
  native physical objective.

## 10 x 30 validation results

| version | boundary edge recall | boundary complete | worst P_D | QoS complete | power MAE | budget violation |
|---|---:|---:|---:|---:|---:|---:|
| existing one-frame joint | 78.96% | 0% | 0.3864 | 39.66% | 0.0506 | 0 |
| two-frame + boundary power + joint | 83.54% | 0% | 0.3715 | 36.21% | 0.0505 | 0 |
| two-frame + PCGrad | 80.42% | 0% | 0.3712 | 32.76% | 0.0491 | 0 |
| two-frame, structure-only boundary + joint | **82.50%** | 0% | **0.3861** | 37.93% | 0.0493 | 0 |
| structure-only, boundary repeat=2 | 83.13% | 0% | 0.3843 | 37.93% | **0.0495** | 0 |

The structure-only boundary branch is retained as the current shadow candidate:
it recovers most of the boundary gain without changing the physical power
path. It is not production eligible because complete boundary coverage remains
zero and the optimizer convergence gate still fails.

## Interpretation

The first PCGrad smoke run observed negative structure/physical gradient cosine
on 52.3% of batches; the full run reduced this to 43.4%, confirming real
multi-task conflict. However, projecting gradients alone does not solve the
teacher-QoS ceiling. The next accepted step must add exact executor-row/LP/KKT
shadow comparison and a fail-closed residual repair path; no learned action
should bypass those native gates.
