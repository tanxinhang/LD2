# Permutation-Equivariant Architecture Screen

## Decision

The per-target sensing-power head is **not** replaced in this screen. A shared
per-target scorer would not itself violate decentralized execution when every
UAV scores only its local target representations and delivered U2U tokens.
However, replacing a trained fixed-order head is a policy change, not a safe
checkpoint migration. It therefore requires permutation-augmented retraining
and is deferred.

Three opt-in components were implemented:

1. local Set/Attention pooling for the communication-rate and total
   communication/sensing-power decisions;
2. proposal/response phase encoding that does not consume an absolute UAV
   one-hot identity;
3. a CTDE-only set/distributional CVaR and QoS-constraint critic.

All switches default to `false`; the established paper baseline is unchanged.

## Actor hot-swap result

The Set heads and identity-free round encoder are structurally valid but are
not safe zero-shot replacements for the trained fixed-cardinality actor.

| K/Q=4/4, first five test seeds | steady | weak3 | worst |
|---|---:|---:|---:|
| Existing actor | 0.967502 | 0.956669 | 0.870008 |
| Set heads only | 0.854271 | 0.805695 | 0.497916 |
| Identity-free round encoder only | 0.853000 | 0.804001 | 0.492691 |
| Both | 0.852955 | 0.803940 | 0.492667 |

Communication remained approximately 768 bit/frame and 0.997 W, while the
movement-conflict rate increased from about 0.870 to 0.928--0.929. This
identifies a learned implicit index/identity protocol rather than a power or
bit-budget failure.

At K/Q=6/6 the combined hot swap changed mean worst from 0.292707 to 0.291576;
it did not repair the transferred policy's scale failure.

Consequently these actor switches must remain experimental until trained with
target/UAV permutation augmentation or teacher distillation.

## Set/distributional risk critic

The risk critic parses the centralized training state as:

- a set of K shared 8-D UAV nodes;
- a set of Q shared 8-D target nodes (kinematics, uncertainty, previous P_D);
- time-to-go and the sender-averaged communication summary.

It deliberately ignores the appended absolute-agent one-hot. Shared encoders
and attention pooling make the UAV path permutation invariant. A shared
target-conditioned output produces:

- per-target next-frame P_D quantiles for lower-tail CVaR estimation;
- per-target logits for `P_D < QoS floor`.

The module contributes only auxiliary critic losses during PPO training and is
absent from the deployed actor. Thus it preserves decentralized execution.
Actor risk routing is not enabled until held-out calibration is demonstrated.

Configuration:

```yaml
marl:
  set_risk_critic_enabled: true
  risk_critic_hidden_dim: 128
  risk_critic_num_quantiles: 16
  risk_critic_cvar_alpha: 0.20
  risk_critic_qos_floor: 0.60
  risk_critic_quantile_coef: 0.25
  risk_critic_constraint_coef: 0.10
  risk_critic_constraint_positive_weight: 0.0  # auto class balancing
```

## Verification

- 95 relevant regression tests pass.
- Exact UAV-permutation invariance and target-permutation equivariance pass.
- Quantile and constraint gradients are finite.
- A real 16-frame PPO integration update completed.
- With automatic positive-class balancing, smoke losses were quantile 1.3293
  and constraint BCE 1.1671.
- Initial raw/balanced violation accuracies were 0.4844/0.5635 at a violation
  prevalence of 0.1563. These one-update values are not evidence of
  calibration.

The one-update random-policy evaluation is intentionally not treated as a
performance comparison.

## Required next experiment

Train only the new critic first while keeping the established actor frozen.
Use disjoint train/selection/test geometry seeds and report:

1. pinball loss on held-out transitions;
2. violation AUROC/AUPRC and calibration error;
3. correlation between predicted lower-tail return and realized episode worst;
4. K/Q=4/4 to 6/6 risk-head transfer.

Only if those diagnostics are positive should predicted risk be allowed to
route PPO advantages or update the Lagrange multiplier. This prevents an
uncalibrated critic from degrading the current strong policy.

This experiment has now been completed. The binary QoS-violation head passed
calibration, while the distributional CVaR head failed. See
`docs/SET_RISK_CRITIC_CALIBRATION.md`; CVaR-based actor routing remains
disabled.
