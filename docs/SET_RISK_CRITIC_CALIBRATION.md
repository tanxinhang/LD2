# Set-Risk Critic Calibration

## Experimental boundary

The established top-1 actor was loaded from
`results/paper_top1_test100/best_restored.pt` and frozen bitwise. Only the
set-based CTDE risk critic was fitted. The final actor tensors have maximum
absolute difference exactly `0.0` from the source checkpoint.

The risk branch predicts:

1. next-frame per-target P_D quantiles;
2. per-target probability that next-frame P_D is below the QoS floor 0.60.

Training used geometry seeds excluded from all evaluation splits. The main
calibration result uses 20 fixed `confirmation` seeds.

## Training convergence

| Risk metric | update 0 | update 5 |
|---|---:|---:|
| Quantile loss | 0.06184 | 0.00467 |
| Constraint BCE | 0.56922 | 0.25773 |
| Constraint balanced accuracy | 0.9103 | 0.9503 |
| Actor unchanged | 1.0 | 1.0 |

The checkpoint containing both the frozen actor and fitted critic is
`results/set_risk_critic_calibration6_fixed/risk_critic_final.pt`.

## Independent 20-seed calibration

### QoS-violation head

| Metric | Value |
|---|---:|
| AUROC | 0.9849 |
| AUPRC | 0.9238 |
| Balanced accuracy | 0.9534 |
| Brier score | 0.0342 |
| ECE | 0.0264 |
| Violation prevalence | 0.1708 |

The constraint head is sufficiently discriminative and calibrated for use as
a training-time crisis detector.

### Distributional/CVaR head

| Metric | Value |
|---|---:|
| Pinball loss | 0.0608 |
| Mean quantile calibration error | 0.1801 |
| Maximum quantile calibration error | 0.4443 |
| Quantile crossing rate | 0.4693 |
| Initial predicted CVaR vs episode worst correlation | -0.1066 |

The unrestricted distributional head is not reliable enough to route actor
updates.

## Monotonic quantile ablation

A simplex-spacing parameterization was tested to force every predicted
quantile into `[0,1]` and guarantee ordering.

| Five-seed screen | Unrestricted | Monotonic |
|---|---:|---:|
| Pinball loss | 0.0501 | 0.0204 |
| Calibration error | 0.1479 | 0.3377 |
| Crossing rate | 0.4733 | 0.0000 |
| CVaR-worst correlation | 0.5191 | 0.3856 |

The structural constraint removes crossing and improves pinball loss, but
worsens held-out coverage calibration substantially. It therefore fails the
stage gate and is not promoted to a 20-seed experiment.

## Fixed-actor sensing result

On the 20 confirmation seeds, the unchanged actor obtains:

| steady | weak3 | mean worst | worst LCB | scenario CVaR20 | QoS feasible |
|---:|---:|---:|---:|---:|---:|
| 0.9113 | 0.8818 | 0.6919 | 0.5559 | 0.1080 | 0.70 |

Mean QoS clears the requested 0.80/0.70/0.60 thresholds, but tail robustness
does not: worst LCB remains below 0.60 and scenario CVaR20 is only 0.108.

## Decision

Do not connect the predicted CVaR to the actor or Lagrange multiplier. The
distributional estimate is not calibrated and has no positive episode-worst
correlation on the 20-seed set.

Retain the well-calibrated binary constraint head. Its next valid use is a
conservative training-time crisis gate combined with measured per-target
advantages or counterfactual action effects. The classifier alone is
predictive, not causal, so it must not directly reward an action without an
action-conditioned improvement signal.

## Engineering verification

- 99 relevant tests pass.
- Pure NumPy Hungarian matching replaced the evaluation-only delayed SciPy
  import that triggered a duplicate OpenMP runtime.
- The Gaussian action log-density is now computed analytically in NumPy.
- Risk checkpoints are saved before final evaluation, preventing loss of a
  completed calibration run if later diagnostics fail.
