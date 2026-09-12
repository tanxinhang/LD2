# STOP-only multi-channel diagnostic

Run `python -m tools.audit_stop_only_channels`. Prespecified adjacent link
seeds 650001,650002,650003 are not selected by sensing performance. Each uses
50k independent H0 calibration and 50k per validation split, fixed policy and
64 rollouts. Per-trace thresholds are diagnostic conditional calibrations, NOT
an implementable unknown-channel universal threshold. Reported intervals are
within-trace, not simultaneous across the three-channel family.

The first two channels erase STOP; the third delivers it. All have four
successful prefix reports. First channel then loses six consecutive reports;
second loses seven of the last nine. Channel code evolves separate directional
shadowing innovations, so forward report reception cannot establish reverse
STOP reliability. Do not impose reciprocity without a justified physical model.

On erased-STOP traces the score must remain exactly the full-report score:
only calibration thresholds can change detection decisions. Integrated tests
check this plus the positive cost of attempted STOP requests. Nine relevant
tests pass with the verified PyTorch environment.

## Seed 650001

Candidate PD 0.87414, PFA 0.00064; PFA upper interval 0.001061825, so the joint
requirement is not certified. Full reference PD 0.86892, PFA 0.00058. At common
threshold scores give identical decisions (zero evidence delta).
H1 mean bits 1676.928, H0 1671.99744 versus 1664 full reports. No savings.
The apparent PD gain is wholly threshold change, not evidence/fusion benefit.

## Consequence

## Remaining results

| Seed | STOP delivered | Candidate PD | PFA | H1 bits | H0 bits |
|---|---|---:|---:|---:|---:|
| 650002 | No | 0.86316 | 0.00042 | 1677.248 | 1672.07936 |
| 650003 | Yes | 0.86826 | 0.00058 | 1559.18336 | 1600.86016 |

650002 meets its within-trace detection bounds but increases communication.
Its evidence contrast is exactly zero: all PD change is threshold change.
650003 meets within-trace bounds (PD lower 0.86362978, PFA upper 0.000985695),
and saves 6.30% H1 bits and 3.79% H0 bits. At the full reference threshold,
its evidence PD delta is -0.00018, interval [-0.000696510,0.000337354]; no
equal-PFA/no-loss claim follows. Two failed controls out of three selected
fixed traces are NOT a population failure-rate estimate.

## Next design boundary

STOP-only removes unconditional control overhead, not the risk that a paid
control request fails. Cost advantage requires control reliability and should
not be assumed from report-link predictions. A future admissible policy needs
observable reverse-link evidence or a justified uncertainty bound; no ACK,
retry, hidden CSI, or retrospectively observed STOP success may be invented.
No policy promotion follows from this diagnostic; defaults unchanged.
