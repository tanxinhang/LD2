# Fixed-resource multi-template receiver diagnostic

## Scope and implementation

This experiment freezes geometry and roles, uses the same 13 acquisition blocks
at 0.15 W, and changes only receiver processing. It does not implement a deployed
cooperative protocol. Two receivers ideally pool projected observations; message
bits, transport latency, processing cost and covariance estimation are not yet
charged. Nine templates cannot be assumed to fit the earlier scalar-report budget.

The declared template bank is y in {0,30,60} m and vx in {0,6,12} m/s.
The single-template prediction uses (30,6); the simulated target is (0,0).
Thus the bank contains an exactly matching template, an optimistic coverage
condition. Online bank maximization does not select using target truth. The
separate oracle-template row is a diagnostic only.

Projected noise is generated jointly using the physical complex template Gram
matrix, not independently for nine templates. Receiver-local temporal noise is
stationary proper complex AR(1), with known rho in {0,0.6}. Two signal phase
trajectories (constant and 1.7 radians/block rotation) are stress tests, not a
learned or physically exhaustive phase model.

## Receiver construction

For unit-noise complex projections z, the energy covariance across time is
rho^(2|t-s|). The energy branch uses its inverse to weight the time sequence,
and predicted template SNR to weight receivers. It searches all nine templates.

The second branch applies w[0]=z[0] and
w[t]=(z[t]-rho*z[t-1])/sqrt(1-rho^2), then sums |w|^2 over time and receivers
and searches templates. For each individual template its H0 mean and variance
are both 26. Cross-template dependence remains and is retained in calibration.

The hybrid is max(bank_energy, (bank_whitened-26)/sqrt(26)). The complete
maximum, including both branches and every template, has its own independent
H0 calibration. Branch normalization is not a Gaussian-tail approximation.
It is a research detector, not a claim of an optimal likelihood-ratio test.

For a constant signal the whitened interior amplitude is multiplied by
sqrt((1-rho)/(1+rho)); at rho=0.6 its energy factor is 0.25. For a rotating
signal the factor is |exp(j*omega)-rho|^2/(1-rho^2). Consequently noise
whitening does not imply phase-independent detection improvement. Keeping an
unwhitened branch mitigates this failure but incurs a search-threshold penalty.

## Independent validation after hybrid design

Command: E:\anaconda\conda\python.exe -m tools.audit_multitemplate_receiver
--samples 50000 --geometry-seed 1032000

Four fresh geometries; independent calibration, H0 evaluation and H1 evaluation
streams, each 50,000 trials per condition. Methods share trial realizations.
Calibration targets empirical PFA=0.0005; evaluation is separate. The earlier
development experiment used geometry seed 1031000. No retuning on these results.

| Geometry | rho | Phase | Single energy PD | Bank energy PD | Hybrid PD |
|---|---|---|---:|---:|---:|
| 0 | 0 | rotating | .94828 | .99320 | .99310 |
| 0 | 0 | constant | .94784 | .99322 | .99348 |
| 1 | 0 | rotating | .83142 | .96028 | .96180 |
| 1 | 0 | constant | .83034 | .96148 | .96298 |
| 2 | 0 | rotating | .96972 | .99278 | .99210 |
| 2 | 0 | constant | .96906 | .99280 | .99240 |
| 3 | 0 | rotating | .93098 | .99302 | .99250 |
| 3 | 0 | constant | .92974 | .99358 | .99344 |
| 0 | .6 | rotating | .59176 | .84320 | 1.00000 |
| 1 | .6 | rotating | .30854 | .58462 | 1.00000 |
| 2 | .6 | rotating | .62248 | .81474 | 1.00000 |
| 3 | .6 | rotating | .59848 | .85994 | 1.00000 |
| 0 | .6 | constant | .52854 | .70158 | .70164 |
| 1 | .6 | constant | .36450 | .55000 | .53868 |
| 2 | .6 | constant | .53690 | .67288 | .67160 |
| 3 | .6 | constant | .53624 | .71210 | .68704 |

Hybrid minus single paired intervals are positive in all 16 conditions;
these intervals are not familywise-adjusted across geometries and phases.
All reported hybrid empirical PFAs are .00020-.00076. The per-condition
adjusted interval upper bounds for correlated geometries 0 and 1 are
.0010122 and .0011858: reliable PFA<=.001 certification is NOT established.
PD=1 means no misses in this finite sample, not a population guarantee.
Methods are independently calibrated, not verified at exactly equal true PFA.

## Decision

Keep this receiver as an optional diagnostic; do not promote it to the online
controller. Template mismatch is recoverable without new acquisitions in this
covered-target experiment. However all four constant-phase correlated cases
remain below PD=.8. The hybrid can lose to bank energy (e.g. .71210 to .68704)
because joint calibration charges for the additional search. The next receiver
audit must cover off-grid states and phase evolution, estimated noise covariance,
and message/compute costs before claiming system-wide improvement or choosing
additional sensing/communication power.

## Off-grid and estimated-covariance stress test

Run with --samples 50000 --geometry-seed 1033000 --stress. Four fresh
geometries, target (y,vx)=(15,3), four phase paths (0, .2t, .05t^2, 1.7t),
and both noise correlations give 32 conditions. Noise rho is estimated once
from 32 independent target-free sequences of length 13, clipped to [0,.99],
then frozen for every method and both calibration/evaluation. Training samples
and their acquisition costs are additional and excluded from the 13 signal blocks.
Calibration still sees the same stationary noise law as evaluation: this does
not test noise-distribution drift. Only one training realization per geometry
and rho was tested, so estimation-risk robustness is not established.

The stress mode relabels template zero as a reference, not an oracle: the true
target is no longer in the bank. Delay/Doppler templates are frozen across
blocks and phase stress paths are imposed separately. These are not complete
kinematically consistent moving-target trajectories.

For rho=.6 the estimated values are .6094, .5962, .6004, .5511.
Hybrid PD by geometry:

| Geometry | Constant | Slow | Chirp | Rotating |
|---|---:|---:|---:|---:|
| 0 | .52896 | .52928 | .63482 | 1.00000 |
| 1 | .55242 | .55424 | .62898 | 1.00000 |
| 2 | .73686 | .74518 | .85388 | 1.00000 |
| 3 | .80268 | .80650 | .86504 | 1.00000 |

For rho=0 hybrid PD spans .95564-.99850. Hybrid empirical PFA across all
conditions spans .00030-.00066; these point estimates do not certify the bound.
Eight of 32 hybrid point estimates remain below .8. Unlike the covered-target
experiment, the hybrid is not uniformly better than the single-template
baseline: correlated geometry 1 constant is .58266 (single) versus .55242
(hybrid). Even bank energy is only .55308 there. Additional templates can
increase the search penalty without improving target alignment enough.

These changed conditions are a combined stress test, not a causal ablation
isolating covariance-estimation loss from target mismatch. No detector or
threshold was tuned after observing these trials. This result blocks default
promotion of unconditional template/branch expansion. A follow-on comparison
should freeze paired observations while varying the bank and rho estimator
separately, and then couple inter-block phase to physical trajectory before
introducing an adaptive search rule. Nine targeted kernel tests pass, including
the exact phase-dependent whitening gain at three temporal frequencies.
