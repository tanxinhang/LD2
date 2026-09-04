# Converged large-scale diagnostic, 2026-08-27

## Claim boundary

This is a diagnostic, not a formal deployment claim.  The strict physical
runner uses hold actions plus the local-belief distributed protocol stack.  It
does not use simulator truth or ground fusion.  The ISCC experiment remains a
synthetic shadow resource model and does not mutate physical actions.

The online policy is now deliberately asymmetric:

- hard physical feasibility and deadline checks gate execution;
- an optimality proof is an optional anytime diagnostic;
- exact global optimality is reserved for small-scale offline audit.

## Protocol

Physical scale and motion experiments use three evaluation seeds (101, 211,
307), 150 frames, a 50-frame tail, a fixed 1130 m square, carrier period 3 and
role-capacity movement.  The scale screen covers fixed UAV/more targets, more
UAV/fixed targets and joint growth through K,Q in {8,12,16}.  The motion screen
keeps K=Q=8, fixes the belief tracker to CV and varies true CV/CT/CA motion over
0--5, 5--10 and 10--20 m/s.

A case passes only if QoS succeeds on at least 80% of seeds, delivery is at
least 0.99, radio and online deadline-miss rates are at most 0.01, and P95 wall
time does not exceed the 100 ms control period.

## Physical scale result

| K/Q | QoS seed rate | steady | weak-3 | worst | P95 step (ms) | online miss | pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8/8 | 0.333 | 0.698 | 0.651 | 0.616 | 40.5 | 0.000 | no |
| 8/12 | 0.333 | 0.520 | 0.473 | 0.454 | 53.1 | 0.000 | no |
| 8/16 | 0.000 | 0.401 | 0.336 | 0.313 | 50.0 | 0.000 | no |
| 12/8 | 0.667 | 0.949 | 0.889 | 0.804 | 93.6 | 0.091 | no |
| 12/12 | 0.667 | 0.923 | 0.790 | 0.703 | 83.4 | 0.018 | no |
| 16/8 | 1.000 | 1.000 | 0.999 | 0.997 | 241.0 | 0.387 | no |
| 16/16 | 1.000 | 0.961 | 0.918 | 0.882 | 158.7 | 1.000 | no |

Communication delivery is 1.0 and radio deadline violation is zero in every
case.  Fixed-K target growth is therefore sensing-capacity/fairness limited,
whereas K=16 is compute-time limited despite excellent sensing performance.
No scale case passes the complete gate.

## Motion result

| true motion, speed | QoS seed rate | steady | weak-3 | worst | belief RMSE (m) | P95 step (ms) |
|---|---:|---:|---:|---:|---:|---:|
| CV, 0--5 | 0.333 | 0.698 | 0.651 | 0.616 | 52.9 | 35.5 |
| CV, 5--10 | 0.333 | 0.599 | 0.548 | 0.520 | 54.6 | 35.7 |
| CV, 10--20 | 0.333 | 0.522 | 0.460 | 0.436 | 53.3 | 35.3 |
| CT, 0--5 | 0.333 | 0.682 | 0.607 | 0.590 | 56.8 | 35.2 |
| CT, 5--10 | 0.333 | 0.614 | 0.508 | 0.464 | 68.7 | 35.3 |
| CT, 10--20 | 0.000 | 0.548 | 0.378 | 0.290 | 93.3 | 36.5 |
| CA, 0--5 | 0.000 | 0.527 | 0.416 | 0.338 | 65.8 | 35.5 |
| CA, 5--10 | 0.000 | 0.460 | 0.359 | 0.297 | 66.3 | 35.5 |
| CA, 10--20 | 0.000 | 0.374 | 0.282 | 0.244 | 66.2 | 35.2 |

Runtime remains within budget, so the motion failure is caused by target
tracking/scheduling robustness rather than online compute time.  Speed hurts
weak-target performance, high-speed CT causes severe CV-model mismatch, and CA
fails from the lowest tested speed band.

## Budgeted ISCC shadow result

An unlimited verifier on independent seed 9101 fails all four cases at 165
kbit/s, with P95 makespans of 50.13--54.27 ms.  Capping exact verification at
eight candidates per UAV gives the following 50-calibration/50-evaluation
result:

| K/Q | hard feasible | optimality certified | oracle ratio | P95 (ms) | pass |
|---|---:|---:|---:|---:|---:|
| 8/8 | 1.0000 | 0.9925 | 0.99999 | 30.45 | yes |
| 8/12 | 1.0000 | 0.9875 | 1.00000 | 30.57 | yes |
| 12/8 | 1.0000 | 0.9850 | 0.99999 | 30.64 | yes |
| 12/12 | 1.0000 | 0.9717 | 0.99993 | 30.77 | yes |

All frozen load bins select analytical upper-bound ordering plus local exact
verification.  The result supports budgeted analytical pruning.  It does not
demonstrate an AI-ranking or compute-offload advantage, and it is not yet an
end-to-end physical result.

## Single next optimization direction

The next iteration is restricted to a **weak-target-first, compute-budgeted
sparse active set**.  Per frame it should:

1. rank target urgency from local belief uncertainty, AoI and achieved
   information floor;
2. activate only the targets needed to protect the weak-target floor;
3. retain only a bounded local UAV neighborhood per active target using cheap
   analytical upper bounds;
4. spend exact evaluation cycles only within the resulting active set;
5. execute the best hard-feasible incumbent at the frame deadline, whether or
   not an optional performance proof has completed.

This single mechanism addresses both observed failures: it directs scarce K=8
resources toward weak targets as Q grows, while preventing K=16 from expanding
all pair-target interactions.  The next paired gate is to improve weak-3/worst
performance for K=8,Q=12/16 and reduce K=16 P95 below 100 ms on the same seeds,
without reducing delivery or violating physical constraints.

## Compute-path optimization checkpoint

The first result-equivalent optimization pass removed two sources of repeated
special-function work without changing the physical model or greedy rule:

1. continuous DD support/ambiguity gain is evaluated as one broadcast-safe
   vector batch instead of one Python call per Tx--Rx--target entry;
2. P0 candidate utility is cached by target.  Initially every candidate is
   evaluated once; after selecting an edge, only scores belonging to that
   edge's target are refreshed because only that target's cumulative
   deflection changes.

The second change reduces expensive utility work from
`O(|S||E|)` to `O(|E| + sum_q |S_q||E_q|)`.  Nine frozen synthetic replays
(three seeds crossed with plain, single-role/rate, and B3+DU branches) match
the old selected set, cumulative deflection and utility byte for byte.  Strict
K=8/Q=8 and K=16/Q=16 replays also match detection, power, selected edges and
UAV positions exactly.  The full repository regression is 1226 passed.

On the same Python runtime, a profiled K=16/Q=16, seed-101, eight-frame run
dropped from 1765.1 to 288.6 ms/frame while preserving its detection result;
the inner solver itself dropped from 12.38 to 0.463 s over the eight frames.
The remaining apparent online bottleneck was then traced to
`use_difference_reward`: a training-only fixed-assignment credit signal that
performed K+1 counterfactual physics evaluations per frame but did not feed
actions, scheduling, power, communication, sensing or tracking.  The frozen
deployment manifest now disables it, and the strict online validator rejects
configs that accidentally enable it.  A 30-frame K=16/Q=16 A/B changed no
detection or communication metric and reduced unprofiled runtime from
152.83 ms/frame (P95 168.70, deadline miss 1.0) to 47.00 ms/frame
(P95 54.40, deadline miss 0.0).

These timing checks are an optimization checkpoint, not a replacement for the
three-seed 150-frame scale table above.  That table remains the last complete
performance experiment until it is rerun under the new deployment contract.

The scale sweep was subsequently rerun with the same three seeds and 150-frame
protocol.  Detection, communication, coverage and tracking metrics reproduce
the old table, as required.  After DD batching and target-local utility-score
caching, K=16/Q=16 improved from P95 158.7 ms and miss rate 1.000 to P95
132.8 ms and miss rate 0.609, but still failed the 100 ms gate.  A further
dead-work audit found that strict U2U observations emit zero neighbor-state
blocks while still calculating hidden neighbor intent K times.  Guarding that
calculation preserves four consecutive K=16 observation tensors byte for byte.
The final K-only three-seed rerun gives:

| K/Q | steady | weak-3 | worst | mean step (ms) | P95 (ms) | online miss |
|---|---:|---:|---:|---:|---:|---:|
| 16/8 | 1.000 | 0.999 | 0.997 | 105.6 | 217.9 | 0.336 |
| 16/16 | 0.961 | 0.918 | 0.882 | 98.8 | 127.3 | 0.304 |

Thus K=16 still does not pass.  The remaining strict-profile hotspot is not a
central information bottleneck: the single-process simulator serializes 16
different private-view max-min LPs (roughly 1.8--2.4 ms per node) plus local
movement projections.  All 16 public-cache views were distinct in the audit,
so sharing one LP solution would violate strict distributed semantics.  An
outer thread experiment was rejected after exceeding 90 seconds because
HiGHS thread oversubscription was much slower than the 30--38 ms serial batch.
The next valid compute task is therefore a structure-aware local max-min solver
or a calibrated distributed critical-path metric, not centralized view fusion.

## Artifacts

- `results/_strict_distributed_sweep/converged_scale_k16_3seed_t150.json`
- `results/_strict_distributed_sweep/converged_motion_3seed_t150.json`
- `results/_strict_distributed_sweep/iscc_anytime_50cal_50test_seed9101_165kbps.json`
- `results/_strict_distributed_sweep/iscc_anytime_budget8_50cal_50test_seed9101_165kbps.json`
- `results/_strict_distributed_sweep/converged_scale_compute_optimized_3seed_t150.json`
- `results/_strict_distributed_sweep/converged_scale_compute_obs_optimized_k16_3seed_t150.json`
