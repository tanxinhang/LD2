# Paired evidence/threshold ablation of the rejected joint gate

Reproduce with `python -m tools.audit_gate_factorial`. This reuses the failed
candidate's 100k evaluation split (760001) for diagnosis, not new validation.
Frozen policy, link model, observations and 64 nested rollouts are unchanged.

| Evidence | Threshold | PD | PFA |
|---|---:|---:|---:|
| Full | 4.44792653 | 0.85876 | 0.00039 |
| Gated | 4.44792653 | 0.85858 | 0.00039 |
| Full | 5.63842991 | 0.79827 | 0.00009 |
| Gated | 5.63842991 | 0.79801 | 0.00009 |

Along full/full -> gated/full -> gated/gate, signed PD differences telescope:

    -0.06075 = -0.00018 (evidence) -0.06057 (threshold).

Evidence contrast: 30 losses, 12 gains; interval [-0.000415425,0.000056541].
Threshold contrast: 6057 losses, no gains; interval [-0.06247893,-0.05864995].
Each uses conservative paired intervals with error .025, giving at least 95%
joint coverage conditional on the frozen experiment. Evidence-loss nonsignificance
does not prove no loss. In the reversed path, threshold delta is -0.06049 and
evidence delta -0.00026: interactions make attribution path-dependent.

Of 6081 full-reference-only detections, 5794 never stopped transmitting. Their
loss cannot be attributed to removed reports. This particular failure is
dominated by threshold change, unlike the earlier more aggressive prefix gate.
The finite-sample origin of the excessive threshold remains to be tested using
independent large H0 calibration sets; the factorial alone does not establish it.

## Resource admission

An offline helper checks an expected-bit budget using a separately justified
lower bound q_L on effective stop probability (including control delivery):

    B_upper = B_full + B_control - B_cancel * q_L <= B_budget.

The bound is only as valid as q_L and applies to the fixed-size schedule, not
per-trial hard budgets or variable-rate packets. Without independent admission
evidence q_L=0, so the control-bearing candidate is rejected for a budget of
1664 bits. Do not derive q_L from the same evaluation H1 outcomes. H0 and H1
need separate checks unless an operational occurrence distribution is supplied.
This is an audit admission helper, not an online controller integration.

Even the observed H1 point stop rate .06862 is below 1/9 break-even: mean bits
1712.94976; H0 mean bits 1746.33472. Fixing threshold calibration alone cannot
make this candidate economical. Keep it off and full reporting as reference.

Next: independently enlarge H0 calibration and check threshold variability;
do not tune a new gate on this diagnostic H1 split. Validate rollout convergence
and establish admission bounds before any budget-constrained policy promotion.
