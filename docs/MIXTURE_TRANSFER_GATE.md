# Frozen mixture transfer audit

The uncertainty model remains fixed at velocity standard deviation 6m/s and
nine positive quadrature points. Prediction x-velocity error is scanned over
0,3,6,12,18m/s and link seeds 660000 through 660002. Each case has independent
calibration/H0/H1 streams with 30000 samples. Geometry and 13-frame resources
remain fixed. No parameter is fitted to this grid.

| Velocity error | Range of cooperative P_D | Range of paired point differences |
| --- | --- | --- |
| 0 | 0.9370–0.9813 | +0.1095 to +0.1561 |
| 3 | 0.9082–0.9630 | +0.0743 to +0.1084 |
| 6 | 0.8425–0.8698 | +0.0214 to +0.0295 |
| 12 | 0.8302–0.8538 | -0.0156 to +0.0034 |
| 18 | 0.8206–0.8291 | -0.0145 to +0.0038 |

Mean grid difference is +0.048482; four of fifteen point differences are
negative. This is a descriptive fixed grid, not an iid population sample.
Negative point differences near zero do not establish statistically significant
harm. The worst observed cooperative P_D is 0.820633; this does not certify
worst-case P_D>0.8 or simultaneous false-alarm control across the whole grid.

Link traces deliver 13,6,9 of thirteen reports, respectively. Each policy still
attempts 1664 bits and incurs the same model communication energy. High error
therefore spends resources for little or uncertain marginal benefit. Source
selection requires receiver-visible diagnostics and independently calibrated
final thresholds; true mismatch cannot be used as an online switch.

The local opposed-Tx/Rx geometry remains unusually insensitive to this velocity
error, so the audit measures mainly remote loss. Expand geometry before general
claims. At P_FA around 0.001, 30000 calibration samples are exploratory; refine
critical cases with larger independent calibration before formal comparisons.

Reproduce: `python tools/audit_mixture_transfer.py`.
