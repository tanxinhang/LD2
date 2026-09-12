# STOP-only physical-branch detector integration

The audit opt-in `stop_only_transport=True` now uses ReservedSlotTransport's
slot-keyed physical radio outcomes. Every trial selects its branch from the
received prefix only: no STOP request -> silent reserved control slot and all
reports; delivered STOP -> prefix reports; failed STOP -> all reports, with
control cost retained. Detector evidence and actual mean bits/energy use the
same branch. Branch selection is vectorized over sensing trials sharing one
frozen exogenous channel, not a separate radio simulation per sensing sample.

This equivalence relies on tested action-independent slot innovations, fixed
reserved schedule and no cross-slot action-induced interference. It does not
apply to general variable-duration transport. The full-reference report trace
is coupled to the same reserved control slot; no extra control bits are charged
to that reference. Reserved delay is unchanged and no acceleration is claimed.

Reproduce `python -m tools.audit_stop_only_detection`. Policy forecast threshold
4.4168183444 and 64 rollouts are predeclared; the previous observed-link Markov
model is transferred without fitting to evaluation outcomes. Its accuracy on
the new slot-keyed trace distribution is not newly certified.

Independent H0 calibration: 100k, seed 870001, threshold 4.1153959800.
Validation: 100k per split, seed 880001; link seed 650000 now denotes the NEW
slot-keyed coupling, not the old legacy trace. No old PD/bits comparison across
trace generators is a paired improvement claim. No target truth enters policy.

Ten targeted tests pass in E:/anaconda/conda/python.exe, including physical
silent-slot coupling and integrated request/effective-stop cost accounting.
Default behavior is unchanged; tests are not a full-suite certification.

## Independent validation outcome

| Metric | Full reports | STOP-only |
|---|---:|---:|
| PD | 0.87005 | 0.87308 |
| PFA | 0.00041 | 0.00046 |
| H1 mean bits | 1664 | 1558.90688 |
| H0 mean bits | 1664 | 1601.04448 |

Candidate simultaneous PD interval [0.86986842,0.87624352], PFA interval
[0.000281754,0.000704509]. Conditional requirements are met. H1 mean bits
decrease 6.32%; H0 3.78%, including requested control. Control succeeds on
this frozen trace; ten of thirteen full reports arrive. This does not establish
performance across control erasures or channel populations.

Paired PD delta 0.00303, conservative interval [0.00246465,0.00359399]. It is
not proof of superior fusion: candidate threshold 4.115396 is lower than full
reference 4.192976 and achieved PFA differs. No equal-PFA/no-loss claim is made.
This is a first conditional end-to-end branch-model result, not deployment
certification. Next: multi-channel control-success/failure validation and
independent equal-PFA comparisons; do not tune using this held-out split.
