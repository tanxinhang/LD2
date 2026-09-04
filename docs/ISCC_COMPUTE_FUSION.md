# Distributed ISCC compute-fusion diagnostic

## Scope

This diagnostic adds computation as a charged resource beside communication
and sensing.  It is deliberately a shadow model: it does not yet mutate the
strict pilot's physical action.  Results therefore establish compute/transport
resource closure, not an end-to-end ISCC deployment result.

Each candidate batch has an owner, cycle demand, deadline, data-locality flag,
request/result bits and analytical priority.  Every UAV has a heterogeneous
cycle rate and joules-per-cycle coefficient.  Remote execution includes
request airtime, link processing, remote queue/compute, response airtime,
compute energy and RF energy.  Non-migratable work remains at its data owner.

One batch is addressed by a deterministic candidate-set hash.  Its executor
reconstructs the batch from already delivered public tokens and returns one
certified batch result.  This avoids the invalid first attempt that sent one
request/result pair for every candidate and consumed 6--11 kbit per event.

## Adaptive policy

For every frame, the shadow controller compares:

1. analytical upper-bound ordering with local exact verification;
2. local AI proposal with exact certification;
3. AI proposal with distributed batch execution.

A mode is eligible only when its hard-feasibility and oracle-utility ratios
are at least 0.99, its task miss rate is at most 0.01, its makespan meets the
compute deadline and its amortized offload traffic is at most 320 bit/frame.
An optimality certificate is now an anytime diagnostic, not an execution
gate: an exactly evaluated feasible incumbent may execute when the proof
budget expires.  Among eligible modes, the controller minimizes modeled
compute plus offload RF energy.

## Fifty-scene scale screen

Assumptions are explicit in
`results/_strict_distributed_sweep/iscc_compute_fusion_50scene_50ms.json`:
50 ms compute sub-deadline, 120 Mcycle/s reference weak processor, 250 kcycle
exact candidate evaluation, 100 kbit/s U2U rate, 0.2 ms packet processing and
a five-frame assignment hold.

| K/Q | adaptive P95 (ms) | certificate | oracle ratio | added bit/frame | selected mode rate (bound/local-AI/fused-AI) |
|---|---:|---:|---:|---:|---:|
| 8/8 | 43.14 | 1.000 | 1.000 | 0.0 | 1.00/0.00/0.00 |
| 8/12 | 43.93 | 1.000 | 1.000 | 0.0 | 0.98/0.02/0.00 |
| 12/8 | 46.45 | 1.000 | 1.000 | 5.6 | 0.84/0.14/0.02 |
| 12/12 | 48.43 | 1.000 | 1.000 | 10.2 | 0.74/0.22/0.04 |

All compute-side gates pass at 50 ms.  A separate 45 ms stress screen fails
the K=Q=12 P95 deadline gate, so 50 ms is a measured diagnostic operating
point rather than an arbitrary claim that any deadline can be met.  The result
also shows that permanent offload is wasteful: the energy-aware controller
uses fusion in only 2--4% of the K=12 scenes.

## Physical-stack cross-check

The independent 150-frame strict physical run is stored in
`results/_strict_distributed_sweep/iscc_physical_baseline_4case_t150.json`.
All four cases pass sensing QoS, delivery rate is 1.0, communication deadline
violation is zero and online deadline miss is zero.  Physical-path P95 wall
times are 35.77, 42.95, 62.52 and 80.06 ms for 8/8, 8/12, 12/8 and 12/12.

The physical wall time and shadow compute makespan overlap conceptually and
must not be added or subtracted.  A physical `H/R` candidate adapter and a
single instrumented execution DAG are still required before claiming that the
combined end-to-end system meets 100 ms.  The present conclusion is narrower:
the physical gate and compute-resource gate each pass, while the integration
gate remains open.

## Progressive information boundary

The original fixed 64-bit result message is now complemented by certified
progressive transport in
`uav_isac/coordination/progressive_information_transport.py`.  Each candidate
has nested gain intervals for summary, coarse factor, fine factor and full
sufficient-statistic layers.  Refinements are batched in groups of at most four
to trade packet-processing delay against extra bits.  Transmission stops when
one lower gain dominates every competing upper gain; failure to prove this
condition returns the feasible incumbent.

The first version of this screen selected a mode after observing the current
scene's realized certificate, oracle utility, latency and energy.  That router
was non-causal and its earlier 140 kbit/s pass is withdrawn.  The corrected
screen uses 50 disjoint calibration scenes to freeze one mode for each
observable candidate-load bin, then evaluates 50 new scenes.  Progressive
gain intervals now come from uniform Gram-matrix quantization plus a Weyl
spectral-error bound; exact gain is not an interval-construction input.

With a fixed PyTorch/NumPy seed, a 50 ms compute deadline and 165 kbit/s links,
the corrected causal result is:

| K/Q | certificate | oracle ratio | P95 (ms) | task miss | added bit/frame |
|---|---:|---:|---:|---:|---:|
| 8/8 | 0.9950 | 0.9956 | 45.96 | 0.0050 | 0.0 |
| 8/12 | 1.0000 | 1.0000 | 42.49 | 0.0000 | 0.0 |
| 12/8 | 1.0000 | 1.0000 | 45.43 | 0.0000 | 0.0 |
| 12/12 | 0.9950 | 0.9956 | 49.97 | 0.0050 | 160.6 |

All declared shadow gates pass at 165 kbit/s.  At 140 kbit/s, K=Q=12 fails
(certificate 0.9833, oracle ratio 0.9849, P95 62.17 ms); at 150 kbit/s it still
fails the P95 gate at 51.06 ms.  Thus the corrected tested transition lies
between 150 and 165 kbit/s.

This remains a measured policy boundary for the stated synthetic workload,
not a universal radio-rate constant.  An independent seed-9101 replay with an
unlimited verifier does not reproduce the four-case pass: P95 makespans are
50.13--54.27 ms and K=Q=12 has hard-feasibility 0.9833.  The earlier 165
kbit/s result must therefore be treated as seed-specific diagnostic evidence,
not a robust operating boundary.

## Budgeted anytime verification

The converged protocol caps exact verification at eight candidates per UAV.
Hard feasibility still gates execution, while expiration of the performance-
proof budget executes the best exactly evaluated feasible action and records
it as uncertified.  On 50 disjoint calibration plus 50 seed-9101 evaluation
scenes at 165 kbit/s:

| K/Q | hard feasible | optimality certified | oracle ratio | P95 (ms) | added bit/frame |
|---|---:|---:|---:|---:|---:|
| 8/8 | 1.0000 | 0.9925 | 0.99999 | 30.45 | 0.0 |
| 8/12 | 1.0000 | 0.9875 | 1.00000 | 30.57 | 0.0 |
| 12/8 | 1.0000 | 0.9850 | 0.99999 | 30.64 | 0.0 |
| 12/12 | 1.0000 | 0.9717 | 0.99993 | 30.77 | 0.0 |

All shadow gates pass, even though 0.75--2.83% of decisions execute without a
completed optimality proof.  Every frozen load-bin mode is analytical bound
plus local verification; neither AI ranking nor compute offload is selected.
Consequently this result supports budgeted analytical pruning, not an AI or
compute-fusion speedup claim.

This remains a measured policy boundary for the stated synthetic workload,
not a universal radio-rate constant.  The physical adapter must recompute it
from actual interval widths, packet rates, AoI and H/R sufficient-statistic
sizes.  The AI scorer also provides no demonstrated ranking lift on this
generator: after removing determinant/Gram-norm label-sufficient features,
both AI and the analytic upper bound retain 1.0 top-k recall.  The current
speedup should therefore be attributed to analytical pruning, not AI.
