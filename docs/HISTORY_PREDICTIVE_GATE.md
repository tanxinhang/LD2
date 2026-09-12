# History-predictive gate: first-stage negative result

An opt-in `history_predictive=True` audit variant replaces prefix moment SNR
with a shared-state H1-conditional posterior. For state k and received history D:

    w_k(D) = softmax(log prior_k + sum_j log f1(e_j|snr_jk)/f0(e_j)).

This is a static-state (identity transition) belief recursion, not yet a dynamic
tracker. Local and delivered remote energies update the same state exactly once.
There is no invented target-existence prior. The existing declared velocity
quadrature remains an assumption, not a learned or calibrated prior.

Given the observed energy sum S, each state's remaining energy sum has a
noncentral chi-square distribution: 2*E_future ~ chi2(2*n, 2*sum snr).
The proxy evaluates posterior-averaged survival probability above T_n-S for
stopping versus continuing. T_n is the fixed-schedule Gamma noise threshold at
design alpha 0.0005. It does not replace whole-policy H0 calibration.
Forecasts assume future reports arrive; actual detection uses only simulated
delivered reports. This delivery-model mismatch is an additional limitation.

## Frozen independent validation

`audit(samples=100000,trial_seed=730001,report_gate=True,history_predictive=True)`
uses link seed 650000 and 13 frames; no fitting to this validation seed.

| Metric | Full reports | History proxy |
|---|---:|---:|
| PD | 0.86422 | 0.85659 |
| PFA | 0.00039 | 0.00040 |
| H1 mean bits | 1664 | 924.12928 |
| H0 mean bits | 1664 | 984.54016 |

Paired PD difference -0.00763, conditional conservative 95% interval
[-0.00880397, -0.00645521]. Candidate PD interval [0.853211,0.859923],
PFA interval [0.000235555,0.000630849]. Resource savings are about 44.5% under
H1 and 40.8% under H0, including the single control packet. No ACK/retry or
extra sensing power is introduced. Reserved latency is unchanged from the
control-bearing gate, not improved. Results are conditional, not deployment
certification; this is not a paired superiority comparison to previous gates.

## Decision

Keep this variant experimental, default off. The posterior update addresses
uncertainty representation but the forecast still optimizes an energy-test
surrogate rather than the actual shared-state mixture decision. More uncertain
state representation alone cannot guarantee a better policy. No causal claim
that this mismatch explains the entire loss is established by this experiment.

The next bounded step is posterior predictive rollout of the actual terminal
mixture statistic and observation-available link uncertainty, with independently
frozen thresholds. It must not inspect future realized delivery masks, future
energies or truth. A zero-loss requirement with no resource budget generally
does not justify stopping informative messages; a resource/accuracy Pareto
evaluation should expose that tradeoff rather than hide it in arbitrary weights.
