# Distributed Consensus-Bid U2U-ISAC Results

## Final deployable scenario

- Four UAVs and four sensing targets; target tracking and ground links are off.
- Every UAV executes a shared decentralized MAPPO actor from local geometry,
  local detection history, its own last transmission, and physically delivered
  U2U target tokens.
- Each UAV has an exact 1 W RF budget per frame. Communication power plus the
  sum of its target sensing powers is constrained to 1 W.
- Four U2U broadcasts are active per frame at 8 bit/dimension. The measured
  system payload is 1280 bit/frame with physical latency and delivery checks.

## Final mechanism

The actor sends sparse top-2 target tokens. One channel carries a comparable
target bid and the remaining channels remain latent sensing/coordination
content. Each receiver reconstructs the same UAV-ID-indexed bid graph from its
local sent-token memory and U2U inbox. An exact one-to-one permutation layer
selects movement responsibilities, while the separate capacity-two projection
guides multistatic sensing-endpoint intent. The environment-level physical
feasibility scheduler still chooses valid transmitter/receiver edges within the
policy-constrained subgraph.

The selected bid uses inertia mix 0.5:

```text
transmitted bid = 0.5 * projected commitment
                + 0.5 * pre-projection intrinsic preference
```

The projected term prevents rapid switching; the intrinsic term lets later
observations correct a poor initial permutation. The movement-consensus layer
is decentralized at execution: no centralized critic, ground station, or true
global state is used by the actor. This statement does not extend to the
environment-level multistatic feasibility scheduler.

## Frozen-checkpoint test result

Configuration:
`config/exp_800_q4_u2u_hierarchical_multistatic_distributed_matching_hybrid50_eval.yaml`

Checkpoint:
`results/qos_pretrained_soft_gated_move50_test100/best_restored.pt`

Predefined 100-seed test-bank result:
`results/distributed_consensus_hybrid50_bid_frozen_test100/paired_eval.csv`

| Metric | Previous system | Final hybrid bid | Requirement |
|---|---:|---:|---:|
| steady P_D | 0.8599 | **0.9452** | >= 0.80 |
| weak3 P_D | 0.8133 | **0.9269** | >= 0.70 |
| mean worst P_D | 0.5429 | **0.7984** | >= 0.60 |
| bootstrap worst LCB | 0.4911 | **0.7497** | -- |
| worst CVaR | 0.0850 | **0.2882** | -- |
| QoS-feasible rate | 0.4100 | **0.7800** | -- |
| feasible Wilson LCB | 0.3325 | **0.7050** | -- |
| mean nearest distance (m) | 63.79 | **57.46** | lower is better |
| worst nearest distance (m) | 184.43 | **157.44** | lower is better |

Worst-episode distribution for the final method: minimum 0.0245, P10 0.2986,
P20 0.5828, P25 0.6376, median 0.9876. Five percent of seeds remain below
0.1, so the method meets the requested Medium mean criteria but is not yet a
hard per-scenario reliability guarantee.

## Resource audit

- U2U bits/frame: 1280
- Active senders/frame: 4
- Delivery rate: 1.0
- Mean latency: 1.152 ms
- System communication power/frame: 0.9941 W
- System sensing power/frame: 3.0059 W
- Maximum per-UAV 1 W balance error: 0

## Matched 100-scenario baselines

Every row below uses the same frozen source checkpoint and geometry bank. The
centralized movement row is an optimistic ceiling rather than a deployable
baseline.

| Variant | steady | weak3 | worst | feasible | bit/frame |
|---|---:|---:|---:|---:|---:|
| Pre-consensus | 0.8599 | 0.8133 | 0.5429 | 0.41 | 1280 |
| No U2U communication | 0.9177 | 0.8903 | 0.7055 | 0.67 | 0 |
| No capacity-two projection | 0.8554 | 0.8072 | 0.5362 | 0.37 | 1280 |
| No direct communication-to-sensing residual | 0.9435 | 0.9247 | 0.7939 | 0.78 | 1280 |
| Zero token values, retain top-2 mask | 0.9332 | 0.9109 | 0.7675 | 0.73 | 1280 |
| Permute values/masks across sender identities | 0.9137 | 0.8849 | 0.7041 | 0.68 | 1280 |
| Projected-only bid | 0.9188 | 0.8918 | 0.7011 | 0.67 | 1280 |
| Intrinsic-only bid | 0.9046 | 0.8728 | 0.6844 | 0.55 | 1280 |
| **Final hybrid bid** | **0.9452** | **0.9269** | **0.7984** | **0.78** | 1280 |
| Centralized movement ceiling | 0.9870 | 0.9827 | 0.9499 | 0.92 | 1280 |

Paired bootstrap intervals exclude zero for the full method's worst-target gain
over silence, no capacity-two projection, sender permutation and both bid
endpoints. They include zero for the direct communication-to-sensing residual
and zero-token-value interventions. The supported mechanism is therefore sparse
identity-consistent communication plus consensus/capacity matching; independent
utility of every latent token dimension is not established. Full intervals and
artifact paths are recorded in `paper/BASELINE_RESULTS.md`.

## Verification

All 23 formal test files were run in isolated Python processes to avoid a local
NumPy/MKL/Torch multi-file runtime conflict: 244 tests passed. The target motion
noise sampler was changed from repeated low-rank multivariate-normal SVDs to an
exact latent acceleration/jerk sampler with the same covariance.
