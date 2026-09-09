# L4 assignment-timescale optimization

The first L4 algorithm screen varied only the commitment time of the existing
distributed bistatic assignment. Power, communication, sensing budget, seed,
horizon and disturbances were paired and unchanged.

The selected development candidate shortens the assignment hold from 20 to 5
frames. On the four calibration seeds, L4 steady/weak3/worst changed from
0.900/0.829/0.805 to 0.916/0.862/0.852. The worst-seed value changed from
0.6250 to 0.6283. Communication remained 2581.76 bit/frame with delivery 1.0.

This is not a universal replacement. On the clean L0 guard, worst changed from
0.9700 to 0.9401, although all four episodes still passed QoS. The candidate is
therefore retained only as an L4-specific comparator and must not replace the
frozen L0 controller.

An eight-seed paired holdout did **not** confirm a general mean improvement.
Baseline versus H5 worst was 0.86610 versus 0.86651 (mean delta +0.00041;
95% paired t interval [-0.03394, 0.03476], bootstrap interval
[-0.02457, 0.02864]); wins/losses were 4/4. Steady and weak3 mean deltas were
only +0.00532 and +0.00160. The correct claim is therefore not “H5 improves
L4 on average.”

The holdout does expose a testable lower-tail effect. The minimum worst-target
score rose from 0.6362 to 0.7128 and QoS rate from 0.875 to 1.0. Candidate gain
was negatively associated with baseline difficulty (Pearson r=-0.855,
p=0.0068): the baseline-bottom four gained +0.0200 on average, while the top
four lost -0.0192. Because this split is diagnostic and post hoc, it is a
hypothesis for a new preregistered tail experiment, not confirmatory evidence.

Fixed 10/15-frame variants, local-speed/local-P_D adaptive holds, a geometry
improvement event trigger, and gain-scheduled movement were screened and
rejected. Their common failure was seed-dependent negative transfer, especially
on the hardest target geometry. The next research step should predict the
change in executed worst-target Deflection, rather than use distance-product or
one-node P_D as a proxy for reassignment benefit.

A causal certificate-triggered H20/H5 switch was also tested and removed: its
calibration worst reached 0.829, but the minimum fell to 0.5909. A fixed H3
boundary screen reached 0.822 with minimum 0.5789. Thus faster reassignment is
not monotonically better, and the lagged composable certificate is not yet a
valid switching statistic.

All figures here are development estimates. Any next confirmation must
preregister a lower-tail endpoint (for example CVaR or a fixed quantile), its
sample size and acceptance rule on a newly untouched paired seed bank. Mean
worst-target performance and the frozen L0 guard must remain co-primary safety
checks.
