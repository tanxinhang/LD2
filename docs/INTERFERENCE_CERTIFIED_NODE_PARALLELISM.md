# Interference-certified node parallelism

## Scope

Node-local control computation may execute in parallel without implying that
the U2U radios transmit concurrently.  The strict baseline continues to use
orthogonal bandwidth.  `uav_isac/environment/interference_certificate.py` is
a read-only shadow test and does not change packet delivery, power, bits, QoS,
or random-number consumption.

## Conservative communication certificate

For desired link `s -> r` and simultaneous sender group `A`, admission requires

\[
\frac{p_s\underline g_{sr}}
{kTB F+\sum_{j\in A\setminus\{s,r\}}p_j\overline g_{jr}
 +\mathbf 1_{r\in A}I_{\mathrm{self},r}}
\ge \gamma_{\mathrm{req}}.
\]

The desired-gain margin is applied downward and the interference-gain margin
upward.  Every insertion is checked against cumulative interference.  A graph
whose edges are only pairwise conflicts is reported for diagnosis but is not
used as a sufficient certificate for groups of size three or larger.

Under half duplex, a receiver cannot transmit in the same resource.  Therefore
the current all-to-all broadcast semantics structurally permits no co-channel
reuse: every pair of active senders is also a pair of required receivers.  To
obtain reuse without assuming ideal full duplex, the protocol must first prove
that a narrower receiver set preserves the target-level composable certificate.

## Staged gates

1. **Shadow communication gate**: all required links pass robust SINR; no
   executed transport change.
2. **Receiver-set gate**: owner-directed/routed packets reconstruct the same
   target certificate as all-to-all delivery, including AoI and relay bits.
3. **Sensing gate**: add waveform leakage to the target-level lower deflection
   bound; require no degradation of the configured Worst `P_D` floor.
4. **Paired execution gate**: on identical seeds, require no QoS-seed loss,
   no RMSE regression beyond a frozen tolerance, delivery >= 0.99, and lower
   radio critical-path P95.
5. **Blind gate**: evaluate once on the frozen >=100-seed bank after a clean
   commit.  A parallel multi-process run is not eligible for latency claims.

Run the first diagnostic with:

```powershell
python tools/diagnose_u2u_interference_reuse.py `
  --config config/exp_strict_distributed_k16q16.yaml `
  --seed 7 --robust-margin-db 3
```

The `disjoint_pair_optimistic_bound` is a routing research bound only.  It is
not evidence that the current all-to-all protocol can safely enable reuse.

## Compute-only parallel benchmark

The private LPs can be benchmarked in parallel while leaving the radio model
unchanged:

```powershell
python tools/benchmark_node_parallel_power.py `
  --config config/exp_strict_distributed_k16q16.yaml `
  --seed 7 --repeats 10 --workers 1 2 4 8
```

The benchmark first captures immutable causal private views, warms up HiGHS in
isolated worker processes, and then requires the serial and parallel power
matrices and worst-deflection values to agree within `1e-12`.  It requests one
native numerical-library thread per node worker to avoid recursive
oversubscription.  A shared-process thread-pool probe was rejected after it
failed to complete within 90 seconds; the SciPy/HiGHS call path must therefore
not be placed in the simulator's Python thread pool.  Results measure only
this workstation's simulator throughput and are not deployment claims.

### Development measurements (2026-08-28)

K16/Q16, seed 7, five timed repetitions after worker warm-up:

| workers | mean LP batch | P95 | speed-up | numerical error |
|---:|---:|---:|---:|---:|
| 1 | 23.34 ms | 23.72 ms | 1.00x | 0 |
| 2 | 14.21 ms | 14.75 ms | 1.64x | 0 |
| 4 | 8.53 ms | 11.24 ms | 2.74x | 0 |
| 8 | 11.50 ms | 13.02 ms | 2.03x | 0 |

Seeds 19 and 43 independently gave 4-worker speed-ups of 2.72x and 2.63x,
again with zero observed power/deflection discrepancy.  Four workers are
therefore the current candidate; eight workers are rejected because shared
host contention dominates.  This benchmark excludes process startup and does
not yet justify changing the formal runner.  A production experiment requires
persistent workers, deterministic cleanup, replay tests, and end-to-end P95
measurement rather than creating a fresh process pool every frame.

## Persistent process candidate

`config/exp_strict_distributed_k16q16_process4.yaml` enables four persistent
isolated workers.  Only complete, unresolved private LP inputs are dispatched.
The following operations remain deterministic in the main process:

- private cache-gap evaluation and AoI handling;
- incomplete-view unknown-target reserve;
- exact byte-key common-subexpression elimination;
- executed-row RF-simplex projection;
- composable certificate generation;
- communication, tracking, movement and sensing physics.

Let `S(A_k,b)` be the deterministic LP solution computed by node `k` from its
private gain view `A_k` and public budget `b`.  Serial assembly is

\[
P_{k,:}=\Pi_{b_k}\bigl([S(A_k,b)]_{k,:}\bigr),\quad k=1,\ldots,K.
\]

Because `S(A_k,b)` has no dependence on another worker's output, evaluating
the product map `(S(A_1,b),...,S(A_K,b))` concurrently commutes with row
assembly.  Parallelism therefore changes scheduling only, provided inputs are
immutable and results are returned in deterministic node order.  This was
checked on complete traces rather than inferred from equal aggregate scores.

### Paired K16/Q16 evidence

For seeds 7/19/43 over 150 frames, serial and four-process execution were
exactly equal (maximum absolute error zero) for every frame of:

- target detection probability;
- local belief RMSE;
- UAV positions;
- executed sensing power;
- transmitted bits and packet delivery.

| seed | serial mean | process mean | serial P95 | process P95 | Worst `P_D` | RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 92.74 ms | 75.90 ms | 105.01 ms | 86.41 ms | 0.9800 | 3.421 m |
| 19 | 92.90 ms | 76.19 ms | 105.69 ms | 88.12 ms | 0.9800 | 3.336 m |
| 43 | 92.36 ms | 75.62 ms | 104.87 ms | 87.17 ms | 0.9600 | 3.570 m |

The one-time worker/HiGHS initialization cost was 780--791 ms and is reported
separately before the mission clock starts. It is not deleted or amortized
inside per-frame latency. The production baseline retains a 100 ms defensive
timeout and exact serial fallback. The development candidate instead reserves
only 15 ms for the LP batch so that the measured non-LP path and an `O(KQ)`
deadline fallback fit inside the 100 ms frame. One late batch is retried on the
next frame; three consecutive failures close the pool and latch the algebraic
fallback. This candidate behavior is opt-in and is not yet a formal default.

Across the existing ten development seeds (150 frames), process execution had
QoS 10/10, delivery 1.0, zero fallbacks, mean episode P95 87.71 ms and maximum
episode P95 91.10 ms. Mean Worst `P_D` remained 0.98387 and mean local tracking
RMSE 3.397 m. These are dirty-worktree diagnostics, not blind confirmation.

Nested process pools are forbidden: when internal node parallelism is enabled,
the seed-bank runner requires one outer episode worker. Otherwise CPU
oversubscription invalidates timing attribution.

## Deadline-safe sparse harmonic fallback

A permanent stale-cache policy was explicitly rejected. In a seed-7, 30-frame
fault injection, holding the old private LP rows after the process pool failed
drove Worst `P_D` to approximately `0.001`, the configured false-alarm floor.
Fresh geometry therefore has to enter the fallback even when no LP is solved.

Let `a^-_{kq}` be node `k`'s current conservative gain for the targets in its
executed sparse hyperedge row, and let
`S_k={q: a^-_{kq}>0}`. The fallback uses

\[
p_{kq}^{\mathrm{harm}}=
\begin{cases}
b_k (a^-_{kq})^{-1}/\sum_{r\in S_k}(a^-_{kr})^{-1},&q\in S_k,\\
0,&q\notin S_k.
\end{cases}
\]

For a fixed row this is the exact solution of

\[
\max_{p_k\ge0,\,\mathbf 1^Tp_k=b_k}
\min_{q\in S_k}a^-_{kq}p_{kq},
\]

because every reachable target contribution is equal at optimum. Zero-gain
targets must not be included in the row minimum: no single sparse transmitter
can cover all targets. Instead, the ordinary target-responsibility aggregation
composes the executed rows into the valid same-frame lower bound

\[
D^-_{\min}=\min_q\sum_k a^-_{kq}p_{kq},\qquad
P^-_{D,\min}=Q\!\left(Q^{-1}(P_{FA})-\sqrt{D^-_{\min}}\right).
\]

The bound is recomputed after communication-budget projection and power
inertia, not on the pre-projection candidate. Rows whose current cached LP has
a relative primal-dual gap at most 5% may retain that incumbent; all other rows
with nonempty conservative support use the sparse harmonic formula. This is a
feasibility and performance-floor certificate, not a global LP approximation
ratio.

### Development fault evidence (2026-08-28)

Paired K16/Q16 runs used seeds 7/19/43 and 150 frames under the same protocol.
The fault case forced three consecutive timeouts from frame 10, which closed
the pool and exercised the algebraic fallback for the remainder.

| mode | QoS seeds | Steady | Weak-3 | Worst `P_D` | RMSE | wall P95 | modelled deadline misses |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal process LP | 3/3 | 0.99746 | 0.98644 | 0.97336 | 3.442 m | 97.62 ms | 0 |
| permanent fallback | 3/3 | 0.99393 | 0.97433 | 0.96266 | 3.442 m | 89.15 ms | 0 |

The paired mean Worst loss is `0.01070`; tracking RMSE is bit-identical because
the fallback changes neither filtering nor motion. The mean reported
composable `P_D` floor is 0.7938 normally and 0.5551 under the permanent fault,
so the average certificate is conservative but informative. It is not yet a
uniform-in-time QoS certificate: the mean per-episode floor P05 is only 0.0151
normally and 0.0651 under fault, and the minimum is the `P_FA` floor 0.001 in
both cases. Startup/incomplete sparse responsibility therefore remains a
separate P0 problem. Wall-clock maxima can still exceed 100 ms (normal mean
per-seed maximum 106.94 ms); the modeled controller plus radio critical-path
P95 is 35.69 ms normally. These dirty-worktree, three-seed results establish
graceful degradation only. They do not replace the frozen >=100-seed blind
test or hardware timing.
