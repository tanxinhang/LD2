# Reverse-control observability and conservative admission

The STOP sender (node 1) observes incoming reports 2->1. Existing directional
shadowing is independent across directions; packet delivery/SNR at node 2 is
not locally available to node 1 without feedback. No feedback, ACK, probe or
new measurement is introduced. A runtime instantaneous reverse predictor would
therefore be unsupported by current information availability.

Implemented alternative: a declared offline reliability lower bound p_L,
with STOP admission only if p_L*R > C. Here C=128 control bits and R=1152
cancellable report bits. This guarantees a positive lower bound on expected
bit savings per requested STOP only when p_L bounds delivery conditional on
request and the current information stratum. An unconditional estimate is not
generally enough for a selective policy. Under the present fixed geometry,
independent directional fading and sensing noise, request selection does not
observe reverse fading, supporting a model-conditional application. This does
not protect every individual failed-control trace or bound sensing losses.

`reverse_delivery_lower` is opt-in. A zero lower bound rejects all requests,
retains full evidence and removes control costs. Missing bound leaves the
existing experimental behavior unchanged for backward compatibility; it is
NOT automatically interpreted as certified admission. Defaults remain off.

`delivery_lower_bound` computes a one-sided Clopper-Pearson bound from independent
offline outcomes. No-data or zero-success inputs return zero. The helper never
receives hidden CSI or the actual current STOP outcome.

Run `python -m tools.audit_reverse_control_admission`: 400 independent simulated
traces at seeds 920000..920399 for calibration, and 400 at 930000..930399 for
independent checking, with no H1 fitting. Calibration observes 299/400 successes,
one-sided 95% lower bound 0.70921792, exceeding 1/9 break-even. These are simulated
offline receiver outcomes, not measurements available to the online sender and
not real-world channel certification. Geometry, power, timing and distribution
shift can invalidate transfer. Per-stratum calibration would require adequate
independent samples, not treating four forward packets as reverse evidence.

Independent check: 263/400 successes (0.6575), lower bound 0.61644069.
Both batches support the coarse 1/9 break-even test within the declared model,
but the second point estimate is below the first batch's 0.7092 lower bound.
This finite-sample discrepancy must not be hidden: it warrants further seed
and distribution-stability checks before treating 0.7092 as a transferable
reliability guarantee. Do not refit on the check batch and call it validation.
The calibrated bound remains an experimental input, not a promoted default.

Seven targeted tests pass in the verified PyTorch environment. Rejected
admission is tested to produce zero requested-control cost and identical
evidence scores to full reporting. No new PD or universal savings claim is made.
