# Gate 2B: Online U2U evidence-packet closure

## Question

After removing free centralized evidence fusion, can finite, quantized, delayed
U2U evidence recover a material part of the central-oracle sensing gain?

The actor, movement, joint ISAC power, latent coordination tokens, P0-selected
edges, and stress seed bank are fixed. No MAPPO update is performed.

## Online information path

For every sensing frame:

1. each selected RX forms receiver-local deflection and a corresponding random
   Gaussian LLR distribution;
2. one fusion owner per target is designated before observing random LLRs;
3. each non-owner RX selects its local-quality Top-1 target;
4. an 8-bit LLR and 2-bit confidence class are placed in a structured packet
   with source, target and timestamp metadata;
5. packet success is evaluated using current UAV positions, actor-allocated
   communication power, SNR, serialization delay and the 5 ms deadline;
6. the owner fuses only unique, target-matched, successfully delivered entries;
7. 2048 H0/H1 draws estimate frame-level PFA and P_D at a fixed team PFA.

The evidence ID is `(observation_frame, source_rx, target_id)`. Duplicate
statistics cannot be fused twice.

## Fixed protocol

- Actor: `results/paper_top1_test100/best_restored.pt`
- Actor SHA-256:
  `73f9d144263bb00d4fb13178d627b88f1cb5519579abfcaca8c0d7d7e20fe2ba`
- Geometry: fixed stress20, 4 UAVs / 4 targets, 150 frames
- LLR: 8 bit, symmetric clipping at 46.9266408
- Confidence: 2 bit, calibrated on selection20
- Threshold: 3.0705367, calibrated using independent H0 draws on stress20
  geometry; H1 outcomes are not used
- Evidence selection: owner-aware local-quality Top-1

## Detection results

| Evidence path | steady | weak3 | worst | worst CVaR | QoS feasible |
|---|---:|---:|---:|---:|---:|
| Local-only | 0.8483 | 0.7977 | 0.5169 | 0.0322 | 0.45 |
| Online U2U evidence | 0.8898 | 0.8531 | 0.6419 | 0.0442 | 0.55 |
| Central oracle | 0.8975 | 0.8633 | 0.6717 | 0.1166 | 0.65 |

Paired online-U2U improvement over local-only:

| Metric | Mean delta | Paired bootstrap 95% CI | Oracle recovery |
|---|---:|---:|---:|
| steady | +0.0415 | [0.0263, 0.0567] | 84.38% |
| weak3 | +0.0554 | [0.0350, 0.0756] | 84.38% |
| worst | +0.1250 | [0.0672, 0.1877] | 80.76% |

The online path meets the requested Medium averages:
steady >= 0.80, weak3 >= 0.70, and worst >= 0.60.

## Content controls

All controls preserve actor actions, communication edges, target IDs, packet
lengths, powers, delivery events and delays. Each control has its own
independent H0 threshold calibration to keep team PFA fixed.

| Method | steady | weak3 | worst | measured PFA |
|---|---:|---:|---:|---:|
| Correct LLR–target content | 0.8898 | 0.8531 | 0.6419 | 0.001003 |
| Zero LLR content | 0.8154 | 0.7538 | 0.4188 | 0.001009 |
| Value–target roll | 0.7791 | 0.7055 | 0.3200 | 0.001137 |

Correct content minus zero:

- steady +0.0744, 95% CI [0.0475, 0.1008];
- weak3 +0.0992, 95% CI [0.0637, 0.1338];
- worst +0.2231, 95% CI [0.1283, 0.3240].

Correct content minus value–target roll:

- steady +0.1107, 95% CI [0.0772, 0.1435];
- weak3 +0.1476, 95% CI [0.1037, 0.1921];
- worst +0.3219, 95% CI [0.2025, 0.4523].

Thus the gain is attributable to the delivered evidence values and their
correct target correspondence, not merely to transmitting packets.
All three thresholds target team PFA 0.001. The value-roll test realizes a
small positive absolute deviation (+0.000137), which is reported rather than
treated as exact equality.

## Communication and power

- Structured evidence: 251.513 bit/frame
- Existing learned latent coordination stream: 768 bit/frame
- Conservative combined total: 1019.513 bit/frame
- Evidence energy: 0.0001644 J/frame
- Mean evidence latency: 0.4023 ms
- Delivery rate: 100%
- Deadline violation rate: 0%
- Unique useful/transmitted evidence entries: 100%
- Measured team PFA: 0.001003
- Maximum per-UAV power-balance error: 2.22e-16 W

Evidence packets are currently accounted in addition to the old latent token.
A later dual-stream codec can embed the structured fields into reserved token
dimensions and reduce the 1019.5-bit combined total; this optimization is not
credited in the current result.

## Decision and limitations

Gate 2B passes as an information-flow and average-QoS gate. MAPPO architecture
changes are still not justified until the following two issues are tested:

1. **Threshold generalization:** the team-PFA threshold was calibrated with
   stress20 H0 geometry. A selection-only/global-LLR threshold must be tested
   on unseen geometry, channel and scale shifts.
2. **Tail reliability:** worst CVaR is only 0.0442 and episode-level QoS
   feasibility is 55%, despite average worst exceeding 0.60.

The fusion owner is also predesignated from physical quality. The evidence
path is U2U, but owner election is not yet a learned/distributed protocol.
This must be stated explicitly in any paper claim.

## Artifacts

- `results/gate2b_u2u_evidence_stress20_final/paired_eval.csv`
- `results/gate2b_u2u_evidence_zero_stress20/paired_eval.csv`
- `results/gate2b_u2u_evidence_value_roll_stress20/paired_eval.csv`
- `config/exp_800_q4_u2u_hierarchical_multistatic_distributed_matching_hybrid50_paper_top1_u2u_evidence_eval.yaml`
