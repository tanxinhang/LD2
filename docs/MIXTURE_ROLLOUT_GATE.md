# Terminal-mixture rollout: experimental, not promoted

## Implemented model

`mixture_rollout_continue` forms H1-conditional state weights from local and
delivered remote prefix likelihoods. The static shared state is retained across
all simulated future frames and both receivers. For each state it generates
future complex-Gaussian observation energies, then evaluates exactly the same
energy log-likelihood and state log-sum-exp used by the terminal receiver:

    T(D,F) = log sum_k prior_k exp(LLR_k(D) + LLR_k(F)).
    gain(D) = sum_k posterior_k(D) E[1(T_continue>h)-1(T_stop>h) | k,D].

Only a strictly negative estimated gain requests STOP. Monte Carlo ties default
to CONTINUE. Common random numbers couple both actions; future local evidence
is identical for both, and remote energies are float32 as in the payload model.
This estimates H1-conditional detection probability, not mutual information,
not unconditional Bayes risk, and not an optimal resource-aware policy.

Future delivery uses a declared Beta(1+s,1+f) Bernoulli probability model based
solely on the four known scheduled prefix slots and received packet count.
One sampled probability is shared across the future window, incorporating
epistemic uncertainty. This exchangeable model does NOT reproduce the actual
radio's time-correlated shadowing. Neither future actual masks, packet-error
probabilities nor truth SNR are supplied to the gate. Beta(1,1) is an explicit
model assumption, not a calibrated prior or an arbitrary reward weight.

## Threshold boundary

The forecast threshold 4.322597843473792 is frozen from the prior full-report
calibration (trial seed 710001), and exposed as `rollout_threshold`. It is not
fitted to current evaluation samples. Whole-policy H0 calibration and independent
H0/H1 validation remain unchanged. Thus the statistic mismatch is addressed,
but the forecast threshold still differs from the final recalibrated adaptive
policy threshold. Solving that self-consistency issue is outstanding. Changing
window or geometry requires an independent design calibration, not blind reuse
of this threshold. Finite rollout sign error is also not controlled by a bound.

## Validation: 30,000 per split, trial seed 740001

Command: `audit(samples=30000,trial_seed=740001,report_gate=True,
mixture_rollout=True,rollouts=64)`. Link seed 650000, 13 frames.

Full PD 0.86570; gate PD 0.8601667; both PFA 0.00050.
Paired delta -0.0055333, conditional conservative 95% interval
[-0.00727193,-0.00379117]. Full-only detections 227, gate-only 61.
Gate H1 mean bits 1569.8944 (5.66% reduction from 1664);
H0 mean bits 1701.0304 (2.23% increase).
The PFA simultaneous interval upper endpoint is 0.00102745, so this sample
does not certify the joint PD/PFA requirement despite favorable point values.
End-to-end audit runtime was 22.72 seconds for three 30k splits on this host;
this is batch audit cost, not measured deployed per-node latency.

## Decision and remaining work

### Rollout-count sensitivity on the same validation samples

At 256 rollouts: PD 0.86540, PFA 0.00050, paired delta -0.00030,
95% interval [-0.000657103,0.0000700623]. Full-only 10, gate-only 1.
H1 stop fraction 0.06690, mean bits 1714.9312; H0 mean bits 1765.312.
Batch audit runtime 90.57 seconds. Failure to resolve a difference is not
proof of noninferiority. The PFA upper confidence bound still exceeds 0.001.

Both rollout counts share real evaluation samples but not nested internal
random draws. This is sensitivity evidence, not a convergence proof or an
isolated estimate of Monte Carlo error. The sharply changed stopping rate
precludes promotion of either version without further numerical validation.

With one 128-bit control packet and nine possible 128-bit canceled reports,
expected communication savings require stop probability > 1/9 = 11.11%.
The 256-rollout H1 stop rate of 6.69% fails this break-even condition; total
bits rise 3.06% under H1 and 6.09% under H0. Forecasting terminal detection
more faithfully does not itself make the control protocol economical.

Keep default off. No claim of improvement over prior gates is supported by
unpaired cross-seed comparisons. Communication, control cost, no ACK/retry and
physical sensing configuration remain unchanged; no latency gain is proved.
Matching the statistic is necessary for this forecast but is not sufficient
for good decisions. Next work must independently fit the observable link
predictor, check threshold self-consistency and rollout convergence, and then
validate the resource/accuracy tradeoff across channels and geometries.
