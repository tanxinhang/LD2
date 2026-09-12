# Fixed schedule headroom and proposal corrections

The supplied advice correctly identifies forecast/receiver mismatch, future
delivery optimism and control overhead. Several implementation claims do not
match 6d89602: there is no dprime2 option, no multi-remote ALL/NONE action,
and no two-layer device/report delivery mask in this audit. There is one
remote source and a moment-SNR energy-test forecast. Multi-UAV subset oracles
would degenerate to the existing binary choice, so they are not implemented.

The new audit compares all fixed report cutoffs k=0..13 on shared calibration,
validation and H1 samples (100000 each), using the final uncertainty mixture
receiver and same FBL channel trace. A schedule is fixed for all trials; the
audit does not select k using the trial's truth or future observations.

| Cutoff | Report bits | P_D | Difference to full reporting |
| --- | ---: | ---: | ---: |
| 0 | 0 | 0.85185 | -0.02364 |
| 4 | 512 | 0.85338 | -0.02211 |
| 6 | 768 | 0.86578 | -0.00971 |
| 8 | 1024 | 0.86698 | -0.00851 |
| 10 | 1280 | 0.87080 | -0.00469 |
| 13 | 1664 | 0.87549 | 0 |

Simultaneous 95% conditional Hoeffding intervals for all nontrivial paired
differences have radius about 0.01125. The cutoff-10 interval is
[-0.01594,+0.00656]. Thus an allowed loss of 0.005 is NOT certified by its
point estimate. No epsilon_D or weighted resource penalty is silently chosen.

These fixed schedules assume preannouncement and charge no runtime STOP
control. They cannot be compared as if an adaptive protocol had free control.
They are a restricted schedule-family reference, not an upper bound for
adaptive stopping or a proof of optimality. Selecting a schedule after seeing
these results requires a fresh selection/evaluation split before deployment.

The prior adaptive gate's 1180 expected bits and 0.87026 PD have observation-
dependent scheduling, control overhead and a different protocol trajectory;
the fixed-cutoff table is not a dominance certificate over that policy.

Further cautions: likelihood distance from a threshold is not calibrated
certainty; multiplying marginal delivery probabilities by fixed gains ignores
history-dependent gains and correlated future arrivals. A proper forecast
must integrate over conditional future delivery/evidence paths. No oracle
future masks should enter an online decision. Reusing an existing control
message can only save overhead after that message and its capacity/deadline
have been demonstrated in the actual schedule.

Reproduce: `python tools/audit_fixed_stop_headroom.py`.
