# Gate 2A: Online detection-boundary regression

## Purpose

This gate separates physical P0 scheduling from sensing-evidence fusion. The
actor, movement, U2U latent messages, joint power, and selected P0 edges are
held fixed. Only the evidence available to the final detector changes:

- `local_only`: best single airborne receiver per target;
- `central_oracle`: explicit sum across every selected receiver;
- `legacy_global`: historical environment-level cumulative deflection.

`u2u_distributed` is fail-closed until the online delivered EvidencePacket
path is connected. It cannot silently fall back to centralized fusion.

## Fixed protocol

- Actor: `results/paper_top1_test100/best_restored.pt`
- Actor SHA-256:
  `73f9d144263bb00d4fb13178d627b88f1cb5519579abfcaca8c0d7d7e20fe2ba`
- Split: fixed `stress`, 20 seeds
- Episode length: 150 frames
- UAVs/targets: 4/4
- No training or actor update

## Results

| Evidence boundary | steady | weak3 | worst | worst CVaR | QoS feasible |
|---|---:|---:|---:|---:|---:|
| Local-only | 0.8483 | 0.7977 | 0.5169 | 0.0322 | 0.45 |
| Central oracle | 0.8975 | 0.8633 | 0.6717 | 0.1166 | 0.65 |
| Legacy global | 0.8975 | 0.8633 | 0.6717 | 0.1166 | 0.65 |
| Central minus local | +0.0492 | +0.0656 | +0.1547 | +0.0843 | +0.20 |

The explicit central path is numerically identical to the legacy path for all
three primary metrics. Both mode-specific reconstruction errors are zero.
The paired episode-level central-minus-local worst gain is 0.154749, matching
Gate 0; its previously computed paired bootstrap 95% CI is
[0.093953, 0.220060].

## Decision

Gate 2A passes. The old worst score is not a property of distributed U2U
coordination alone; it includes a material centralized evidence-fusion gain.
The default configuration is therefore `local_only`. Historical paper-family
configs explicitly request `legacy_global` for regression only.

The next admissible gate is to make `u2u_distributed` consume only unique,
target-matched, unexpired evidence contained in successfully delivered
structured packets. MAPPO retraining remains blocked until this information
flow is closed.

## Artifacts

- `results/gate2a_local_only_stress20/paired_eval.csv`
- `results/gate2a_central_oracle_stress20/paired_eval.csv`
- `config/exp_800_q4_u2u_hierarchical_multistatic_distributed_matching_hybrid50_paper_top1_local_only_eval.yaml`
- `config/exp_800_q4_u2u_hierarchical_multistatic_distributed_matching_hybrid50_paper_top1_central_oracle_eval.yaml`
