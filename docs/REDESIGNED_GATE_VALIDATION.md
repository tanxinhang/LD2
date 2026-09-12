# Large-sample redesign and independent calibration

Reproduce with `python -m tools.audit_redesigned_gate`.

Design uses only H0, 100,000 samples, seed 840001, fixed observed-link model
and 64 rollouts. Iterations: 4.1 -> 4.4126012868 -> 4.4168183444 ->
4.4168183444. This is an empirical fixed point, not a convergence theorem.

The policy forecast threshold is then frozen. Independent H0 calibration uses
100,000 samples, seed 850001. Any difference in terminal detection threshold
is explicitly retained; independence is not sacrificed to force numerical
equality. This evaluates a two-threshold pair, not exact prediction/decision
self-consistency. The entire pair is frozen before 100,000-per-split final
validation at seed 860001. No H1 data select thresholds or policies.

Same frozen geometry, link seed 650000, powers and report/control schedule;
no ACK/retry. Results are conditional on this model/channel, not population
certification. Resource acceptance requires counting the control packet under
both H0 and H1; successful detection alone is insufficient for promotion.

Regression checks: 9 targeted tests pass with `pytest --noconftest`. The normal
test entry point currently fails because its shared conftest imports unavailable
PyTorch. This is not a full-suite pass. No dependencies were installed or changed.

## Frozen validation results

Independent calibration threshold: 4.1014653409; fixed-policy pointwise 95%
tail-probability upper bound 0.000621670. Forecast remains 4.4168183444.

| Metric | Full reference | Frozen gate |
|---|---:|---:|
| PD | 0.86404 | 0.87343 |
| PFA | 0.00049 | 0.00059 |
| H1 mean bits | 1664 | 1683.45856 |
| H0 mean bits | 1664 | 1736.79616 |

Gate simultaneous PD interval [0.87022215,0.87658974], PFA interval
[0.000384644,0.000861380]. Both meet the conditional detection requirements.
Paired PD delta 0.00939, interval [0.00854717,0.01023070]. This is NOT evidence
that pruning improves the detector: the full reference has a higher threshold
(4.31513965) and lower achieved PFA. The operating points differ, and the pairwise
result describes these two frozen procedures, not equal-PFA algorithm superiority.

Resource requirement still fails: H1 mean bits increase 1.17%, H0 4.37%.
H1 stop fraction 9.422% is below 11.11% break-even. No latency benefit or rollout
convergence is established. Keep default off; detection calibration recovered,
but communication optimization did not. Next work should prioritize avoiding
uneconomical control or independently budget-admitting policies, rather than
further threshold tweaking to manufacture a PD gain. Equal-PFA comparisons
require fresh independently calibrated evaluation, not tuning to this test set.
