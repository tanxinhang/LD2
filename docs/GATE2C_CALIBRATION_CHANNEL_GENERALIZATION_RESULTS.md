# Gate 2C: Calibration isolation and channel operating boundary

## Question

Does the online U2U evidence result survive when the detector threshold is
calibrated without access to the stress-test geometry, and over what channel
range does the current frozen policy remain deployable?

The actor, movement policy, power split, evidence codec and stress20 geometry
are fixed. No PPO update is performed.

## Selection-only threshold calibration

The standardized team detector threshold is calibrated only from independent
H0 samples in the disjoint selection20 trace:

- selection-only threshold: `3.0951552649`;
- previous stress20-H0 threshold: `3.0705367`;
- evidence codec: owner-aware Top-1, 8-bit LLR and 2-bit confidence;
- target team PFA: `0.001`.

H1 labels and all stress20 observations are excluded from threshold fitting.
The resulting threshold is then frozen for every experiment below.

### Offline independent stress20 check

| Path | steady | weak3 | worst | measured PFA |
|---|---:|---:|---:|---:|
| Quantized Top-1 evidence | 0.8886 | 0.8515 | 0.6375 | 0.000939 |

The offline worst-oracle recovery is 77.91%, with a paired-bootstrap 95%
interval of [60.89%, 90.56%]. Thus the nominal Gate 2B result is not explained
by fitting the detector threshold on the stress geometry.

## Online nominal stress20

| Evidence path | steady | weak3 | worst | worst CVaR | QoS feasible |
|---|---:|---:|---:|---:|---:|
| Local-only | 0.8483 | 0.7977 | 0.5169 | - | - |
| Online U2U evidence | 0.8886 | 0.8514 | 0.6379 | 0.0418 | 0.55 |
| Central oracle | 0.8975 | 0.8633 | 0.6717 | - | - |

Online U2U improves mean worst by `+0.1210` over the same-trajectory
local-only detector and recovers 78.18% of the central-minus-local gap. The
measured PFA is `0.000926`. All three requested Medium mean thresholds are
met, but the tail requirement is not: scenario CVaR is only 0.0418 and only
55% of episodes are jointly QoS-feasible.

## Unseen channel perturbations

All rows use the same selection-only detector threshold.

| Channel condition | steady | weak3 | worst | CVaR | QoS | Evidence delivery | Latent-token delivery | Worst recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Nominal | 0.8886 | 0.8514 | 0.6379 | 0.0418 | 0.55 | 1.000 | 1.000 | 78.18% |
| SNR threshold 30 dB | 0.8868 | 0.8491 | 0.6308 | 0.0418 | 0.55 | 0.975 | 1.000 | 73.61% |
| SNR threshold 35 dB | 0.8646 | 0.8194 | 0.5553 | 0.0229 | 0.50 | 0.667 | 0.812 | 67.45% |
| Deadline 0.8 ms | 0.8264 | 0.7685 | 0.4308 | 0.0175 | 0.40 | 1.000 | 0.535 | 59.44% |
| Deadline 0.4 ms | 0.7910 | 0.7213 | 0.3342 | 0.0147 | 0.25 | 0.620 | 0.000 | 40.01% |

The current average-QoS operating envelope therefore includes the nominal
channel and the 30 dB SNR screen, but excludes the 35 dB and sub-millisecond
deadline screens.

The deadline failure is not primarily a detector-threshold failure. At
0.8 ms every structured evidence packet arrives, but only 53.5% of the older
latent coordination tokens arrive. This changes movement and pairing: even
the same-trajectory central oracle falls from nominal worst 0.6717 to 0.4946.
At 0.4 ms all latent tokens expire and the central oracle falls further to
0.4295. The evidence path still adds value over local-only in both cases, but
it cannot recover a policy trajectory already degraded by missing coordination
messages.

At 35 dB, both streams lose packets. The evidence path still improves
same-trajectory worst from 0.4614 to 0.5553, but the central oracle itself is
only 0.6006. This condition lies at the physical/policy feasibility boundary
rather than being a pure evidence-codec failure.

## Scale claim

No K=6 or K=8 scale-generalization claim is admitted at this gate. The prior
controlled-density K=6 screen already showed that the transferred 4/4 actor
regresses against no-U2U (`worst 0.2927` versus `0.3881`) and that the
equivariant communication-head hot swap does not repair it (`0.2916`).
Those experiments isolate a cardinality/coordination-policy failure that
precedes the detector threshold and evidence fusion.

Accordingly, rerunning the current evidence codec on the same failed
transferred actor would confound evidence scalability with actor migration.
K=8 remains stopped until a permutation-augmented, variable-cardinality actor
passes the K=6 direction gate.

## Decision

Gate 2C passes the **calibration-independence** and **nominal/moderate-channel
average-QoS** gates. It does not pass tail robustness, tight-deadline
robustness or scale generalization.

The next justified model change is narrow:

1. train the existing actor with latency/SNR domain randomization and explicit
   latent-token dropout, rather than changing the detector;
2. use the already calibrated binary QoS-violation critic as a training-time
   crisis gate;
3. optimize an action-conditioned measured improvement signal for the weak
   target, not the uncalibrated distributional CVaR estimate;
4. retest K=4 nominal, 30/35 dB and 0.8 ms before reopening K=6.

The evidence codec, selection-only detector threshold and structured content
controls should remain frozen during that experiment.

## Artifacts

- `results/gate2c_selection_only_threshold_offline_stress20/summary.json`
- `results/gate2c_selection_calibrated_online_stress20/paired_eval.csv`
- `results/gate2c_selection_calibrated_snr30_stress20/paired_eval.csv`
- `results/gate2c_selection_calibrated_snr35_stress20/paired_eval.csv`
- `results/gate2c_selection_calibrated_deadline0p8ms_stress20/paired_eval.csv`
- `results/gate2c_selection_calibrated_deadline0p4ms_stress20/paired_eval.csv`
