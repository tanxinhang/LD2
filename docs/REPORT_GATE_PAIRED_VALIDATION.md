# Independent paired validation of reporting loss

200000 samples per split, seeds 710001/710002/710003, frozen link seed
650000, 13 frames, report_gate=True and stop_headroom=True. The link trace
includes the scheduled control exchange. This is conditional validation on
that trace, not a link-population guarantee.

New paired intervals use discordant counts: p(candidate only)-p(reference
only). Two Clopper-Pearson intervals with a union bound retain trial pairing.
Fixed-schedule comparisons allocate error across all fourteen cutoffs.

| Policy | P_D | PD simultaneous interval |
| --- | ---: | --- |
| Full reports | 0.86359 | [0.861057,0.866095] |
| Prefix gate | 0.85900 | [0.856432,0.861541] |

The gated-minus-full difference is -0.00459, with its separately prespecified
paired 95% interval [-0.005195,-0.003984]. Candidate-only count=335,
reference-only count=1253. The gate's loss is resolved as negative; losslessness
is rejected for this case. Individual gate PFA upper bound is 0.0005571.
Absolute conditional PD>0.8 and PFA<=0.001 still pass.

H1 mean bits=1184.70592 including control, approximately 28.8% below 1664;
mean communication energy=0.000011206618J. Reserved time remains 83.312ms,
5ms longer than the original always-send protocol.

Fixed cutoff differences, simultaneously bounded across cutoffs:

| Cutoff | Paired PD difference | Interval |
| --- | ---: | --- |
| 4 | -0.00854 | [-0.010272,-0.006807] |
| 6 | -0.00479 | [-0.006409,-0.003170] |
| 8 | -0.001025 | [-0.002499,+0.000449] |
| 10 | -0.00095 | [-0.002243,+0.000344] |
| 12 | -0.00179 | [-0.002789,-0.000790] |

These fixed schedule results motivate a comparison with the adaptive gate;
they are not automatic permission to select one after inspection. A new
validation split and multiple link/geometry conditions are needed. No user
loss tolerance is inferred from these results. Calibration threshold variation
also changes absolute PD relative to earlier runs; comparisons above are
within this single calibration/validation experiment only.
