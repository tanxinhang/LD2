# Report-gate loss attribution and conservative candidate

## Reproduction

Run `audit(samples=200000, trial_seed=710001, report_gate=True)` from
`tools.audit_unified_receiver_transport`. Link seed remains 650000, window 13.
Calibration, H0 validation and H1 validation are independent splits.

The full detector finds 1253 targets missed by the gate; the gate finds 335
missed by full reporting. Net PD difference is -0.00459, with conditional
conservative paired 95% interval [-0.00519543, -0.00398426].

## What was lost?

- 1091 of the 1253 remain missed when gated scores use the full detector's
  threshold. The other 162 cross this diagnostic threshold but fail the
  separately calibrated gate threshold (4.349886 vs 4.322598).
- 124 lost trials never stopped: their score is unchanged, but the threshold
  increased. This is a subset of the 162, not an additional loss category.
- Mean full-minus-gated mixture log score among lost trials is 2.19405.
- Lost-trial mean remote prefix energy is 1.01370; future energy is 2.21846
  (noise mean 1). Actual fixed projected remote SNR is 0.49431, so unconditional
  expected energy is 1.49431. The channel's sensing quality did not improve
  later: conditioning on a weak prefix and eventual lost detection selects
  different noise realizations. These conditional means do not prove a
  time-varying target or an unbiased forecast error magnitude.
- Model hypothesis entropy increases by 0.71378 nats among lost trials.
  This indicates less concentrated model evidence, not demonstrated loss
  of true-state accuracy: truth is not necessarily a bank hypothesis.

The common-threshold counterfactual separates score loss from calibration
effects; it is not a replacement detector with validated false-alarm control.
Removed mixture score is a nonlinear aggregate contribution, not a sum of
independent per-report information gains.

## Candidate, not a final optimal policy

Optional `gate_confirmation=True` stops only when the full four-frame prefix
and both disjoint two-frame prefixes recommend stopping. Missing evidence in
either half defaults to continuing. It consumes no future data, extra message,
ACK or retransmission. Each policy retains its own H0 calibration. This is a
conservative diagnostic heuristic, not an information-theoretic guarantee.

On the same 200k split, losses fall from 1253 to 490, with 111 reverse gains;
net PD loss becomes 0.001895 (95% interval [-0.00226595, -0.00152371]).
H1 mean bits rise from 1184.71 to 1461.37, versus 1664 full reporting.
The improvement costs resources and still has statistically resolved loss.

Independent trial seed 720001, 100k per split, same frozen link/geometry:
full PD 0.85906; candidate PD 0.85680; candidate PFA 0.00039.
Candidate simultaneous PD interval [0.853423, 0.860131], PFA interval
[0.000227954, 0.000618477]. Paired delta interval [-0.00282243, -0.00169665].
H1 mean bits 1459.97 (12.26% saving), H0 mean bits 1699.28 (2.12% increase).
Thus meeting PD > 0.8 in this conditional experiment is not evidence of
lossless compression or unconditional resource savings. Reserved delay still
includes the extra 5 ms control slot; no speedup has been demonstrated.

## Next design consequence

Do not promote prefix mean energy to a reliable remote-quality state. A
principled replacement must propagate uncertainty in remote evidence quality
and shared target state, and evaluate future reports by their conditional
effect on the final detection decision. Training/selection and validation must
be split; online inputs remain only locally observed or already delivered
evidence. Keep full reporting as the reference and this candidate opt-in.
Claims remain conditional on the frozen static geometry, declared hypothesis
bank, independent AWGN, schedule-exogenous channel trace and simulated link.
