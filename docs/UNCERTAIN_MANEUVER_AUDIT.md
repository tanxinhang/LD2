# Paired maneuver reference audit

The 12 seeds (310000 through 310011), five epochs, and frozen 0.15 W
sensing allocation compare point-prediction conditional selection with a
five-velocity max-min increment selector. Both use the same nominal motion
candidate bank per state. Initial velocity and its measurement are perturbed;
one random acceleration changes velocity at epoch 2. This is not yet a
recursive stochastic tracking experiment.

Results from `tools/audit_uncertain_maneuver_ensemble.py`:

| Quantity | Value |
| --- | ---: |
| Point policy sample minimum model P_D | 0.99071688 |
| Max-min policy sample minimum model P_D | 0.97181512 |
| Mean paired difference | -0.00174949 |
| Seed-bootstrap 95% interval for mean difference | [-0.00555740, 0.00077056] |

The max-min policy has no demonstrated gain. A finite velocity set with
arbitrary 6 m/s offsets neither covers the continuous uncertainty distribution
nor certifies a lower bound. Greedy worst incremental information is also not
equivalent to maximizing worst terminal information when hypothesis-specific
history values differ. These are reasons to retain it as an ablation only.

## Correction to earlier interpretation

The active-geometry, short-history and moving-target audits pool evidence from
multiple physical receivers. Without a modeled transport path that evidence
cannot be called locally available at one node. The final detector furthermore
uses true signatures and covariance. Its high model P_D does not establish
performance of a receiver with uncertain target state. Earlier descriptions
of these values as a distributed detection requirement being satisfied are
withdrawn. The new audit explicitly sets deployable_detection_certified=false.

The bootstrap quantifies the paired mean difference across these sampled
conditions, not a lower confidence bound on worst-case P_D. Actual false alarm
rates are not measured here: the scalar Gaussian tail formula presumes the
declared matched model. Motion duration, acceleration feasibility, waveform
phase convention and communication costs require closure before system claims.

The next comparison should use receiver-visible, prediction-matched statistics
with independently calibrated H0 thresholds and source/message ownership.
Any hypothesis set must be calibrated from observation uncertainty, and its
coverage and terminal-information objective audited before calling it robust.
