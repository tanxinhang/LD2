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

Fixed 10/15-frame variants, local-speed/local-P_D adaptive holds, a geometry
improvement event trigger, and gain-scheduled movement were screened and
rejected. Their common failure was seed-dependent negative transfer, especially
on the hardest target geometry. The next research step should predict the
change in executed worst-target Deflection, rather than use distance-product or
one-node P_D as a proxy for reassignment benefit.

All figures here are development estimates. Confirmation requires a newly
sealed paired seed bank and confidence intervals.
