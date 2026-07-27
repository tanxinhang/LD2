# Gate 1: distributed evidence-transport feasibility

Date: 2026-07-23

## Scope

This gate does not retrain MAPPO and does not yet replace the environment's
legacy global-deflection reward path. It replays the fixed formal actor and
tests an offline counterfactual detector on exactly the same physical
trajectories.

The experiment asks whether receiver-local stochastic evidence can recover the
centralized fusion gain after explicit target selection, finite-rate
quantization, packet accounting, U2U SNR/deadline delivery, and content-only
interventions.

## Protocol

- actor: `results/paper_top1_test100/best_restored.pt`;
- calibration split: 20 fixed `selection` seeds;
- test split: 20 fixed `stress` seeds;
- 150 frames per episode and a 20-frame steady window;
- Gaussian LLR consistent with the existing deflection detector;
- \(10^6\) samples for clipping and H0 threshold calibration;
- 2,048 independent H0/H1 draws per test frame and target;
- target-specific alternatives: when evaluating \(H_{1,q}\), other targets
  remain under H0;
- shared random draws across normal, zero-content, and value-target mismatch;
- fixed team-level \(P_{FA}=0.001\);
- clipping and 2-bit confidence classes fitted only on `selection`;
- method thresholds calibrated with independent H0 samples on the test
  geometries; no H1 outcome enters threshold calibration.

The packet has a 64-bit header, 2-bit source ID, 16-bit timestamp, 2-bit
target ID per entry, 8-bit quantized LLR per entry, and 2-bit confidence class
per entry.

## Gate 1a: stochastic detector validation

One million samples per deflection/hypothesis were tested over
\(D\in\{0.5,1,2,4,8,12,20\}\).

| Check | Maximum error | Tolerance | Result |
|---|---:|---:|---|
| empirical vs target \(P_{FA}\) | \(4.2\times10^{-5}\) | \(2.0\times10^{-4}\) | pass |
| empirical vs theoretical \(P_D\) | \(6.77\times10^{-4}\) | \(2.45\times10^{-3}\) | pass |

Result:
`results/gate1a_llr_validation/summary.json`.

## Gate 1b0: selection-capacity screen

This first screen assumed lossless, unquantized evidence transport.

| Stress-20 method | steady | weak3 | worst | worst oracle recovery |
|---|---:|---:|---:|---:|
| best local receiver | 0.8483 | 0.7977 | 0.5169 | 0% |
| central oracle | 0.8975 | 0.8633 | 0.6717 | 100% |
| naive local-quality Top-1 | 0.8653 | 0.8204 | 0.5536 | 23.7% |
| naive local-quality Top-2 | 0.8838 | 0.8450 | 0.6175 | 65.0% |

Naive Top-1 fails the pre-registered 50% recovery gate. The failure is
structural: a sender often spends its only slot transmitting evidence already
available locally at the target's fusion owner.

## Owner-aware non-redundant routing

The corrected selector first removes targets for which the sender is already
the predesignated airborne fusion owner, then applies local-quality Top-k to
the remaining peer contributions. Owner selection occurs before random
frame-level evidence is observed.

This reduces the useful peer LLR clipping range from approximately 9,676 to
45--47 and makes 8-bit quantization viable.

## Final Gate 1c result

The final result uses 8-bit LLR, 2-bit confidence, the existing per-UAV
communication power allocation, and the configured physical U2U link.

| Method | steady | weak3 | worst | worst recovery | bits/frame |
|---|---:|---:|---:|---:|---:|
| best local receiver | 0.8483 | 0.7977 | 0.5169 | 0% | 0 |
| owner-aware Top-1 | **0.8899** | **0.8532** | **0.6417** | **80.6%** | 251.5 |
| owner-aware Top-2 | **0.8976** | **0.8635** | **0.6719** | **~100%** | 261.1 |
| central oracle | 0.8975 | 0.8633 | 0.6717 | 100% | unconstrained |

Small Top-2/central differences are Monte Carlo error, not evidence beyond the
oracle.

Both structured methods satisfy the requested Medium deployment thresholds:

\[
\mathrm{steady}\ge0.80,\qquad
\mathrm{weak3}\ge0.70,\qquad
\mathrm{worst}\ge0.60.
\]

Transport diagnostics:

| Method | delivery | p95 latency | max latency | energy/frame | measured \(P_{FA}\) |
|---|---:|---:|---:|---:|---:|
| Top-1 8-bit + conf2 | 100% | 0.526 ms | 0.578 ms | \(1.64\times10^{-4}\) J | 0.001021 |
| Top-2 8-bit + conf2 | 100% | 0.529 ms | 0.627 ms | \(1.71\times10^{-4}\) J | 0.001008 |

The deadline is 5 ms.

## Content-only causal controls

Packet targets, packet count, bits, sender power, link success, and latency are
held fixed. Only LLR content is intervened upon.

| Normal method minus control | worst delta | paired 95% CI |
|---|---:|---:|
| Top-1 minus zero content | +0.2227 | [0.1280, 0.3224] |
| Top-1 minus value-target mismatch | +0.3219 | [0.1957, 0.4579] |
| Top-2 minus zero content | +0.2426 | [0.1401, 0.3500] |
| Top-2 minus value-target mismatch | +0.3580 | [0.2262, 0.4970] |

All lower bounds are positive. The recovered performance therefore depends on
the transmitted stochastic evidence and its correct target semantics, not
merely on communication activity or metadata.

Formal result:
`results/gate1c_conf2_controls_owner_aware_8bit_stress20/summary.json`.

## Interpretation and remaining boundary

Gate 1 passes as a fixed-policy feasibility result. It supports a real method
change: owner-aware, non-redundant, finite-rate evidence routing can replace
most or all of the current free centralized evidence-fusion gain.

It does not yet prove end-to-end decentralized deployment:

1. the main environment still rewards the legacy global-deflection detector;
2. the airborne fusion owner is predesignated from physical quality and must
   be exposed without post-evidence selection or centralized evidence access;
3. the current target model is fixed and receiver LLRs are conditionally
   independent;
4. delayed evidence semantics, correlation, dynamic targets, and AoI remain
   future gates.

The next implementation step is to add explicit detector modes
`legacy_global`, `local_only`, `u2u_distributed`, and `central_oracle`, make
`u2u_distributed` consume only delivered structured packets, and rerun the
fixed actor before any MAPPO retraining.
