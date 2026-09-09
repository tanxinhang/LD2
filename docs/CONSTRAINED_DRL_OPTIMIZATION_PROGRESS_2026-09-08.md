# Teacher-free constrained DRL: progressive optimization record

Date: 2026-09-08

## Decision

The online learning route remains **GNN + temporal encoder + DRL**. It does
not consume teacher labels or certificate fields. An exact physical solver is
retained only for offline comparison and a separately audited fallback.

The optimizer is not a lexicographic cascade and is not a flat weighted sum of
unrelated losses. The operating point is defined by one primary policy
objective and explicit physical conditions:

\[
\min_\theta\;L_{\mathrm{PPO}}(\theta),
\qquad
g_q(\theta)=\operatorname{CVaR}_{\alpha}
\left(P_{D,\min}-P_{D,q}\right)\leq 0,
\]

with additional resource, communication, latency, topology-change and
consensus residuals added as vector conditions when their rollout signals are
available. Each condition owns its multiplier; it cannot be hidden by a loss
weight chosen for another metric.

For inequality residual vector \(g\), the actor-side condition loss is the
Powell--Hestenes form

\[
\mathcal{L}_{c}(\theta,\lambda)=
\sum_q\frac{[\lambda_q+\rho g_q(\theta)]_+^2-\lambda_q^2}{2\rho},
\qquad
\lambda_q\leftarrow[\lambda_q+\rho g_q]_+.
\]

Primal and dual updates alternate. Pareto minimum-norm gradients are optional
inside a two- or three-objective block; they do not choose the operating point
and do not absorb hard conditions.

## Implemented primitives

- `uav_isac/optimization/constrained_pareto.py`
  - minimum-norm Pareto direction with deterministic Frank--Wolfe solve;
  - objective-gradient unit normalization;
  - empirical target-wise detection CVaR;
  - PPO importance-weighted detection-CVaR surrogate;
  - factorized multi-UAV joint-policy likelihood ratio;
  - projected target-wise dual update;
  - inequality augmented Lagrangian;
  - independent stationarity, primal, dual and consensus convergence gates.
- `uav_isac/optimization/alternating_controller.py`
  - cyclic small-objective blocks;
  - vector constraint multipliers;
  - schedule/dual checkpoint and strict resume validation;
  - no dependency on PPO, Adam, a teacher or a certificate.

## Correct multi-agent estimator

The rollout buffer stores one team detection vector on every UAV row. A naive
per-row CVaR would duplicate the same physical outcome and use the wrong
single-agent importance ratio. For a factorized team policy the estimator must
first form

\[
\log r_t=\sum_{k=1}^{K}
\left(\log\pi_{\theta,k}(a_{t,k}|o_{t,k})-
\log\pi_{\mathrm{old},k}(a_{t,k}|o_{t,k})\right),
\]

then apply the CVaR condition once to the unique \(P_D(t,q)\). The implemented
joint-policy function enforces `(time, target)` and `(time, agent)` shapes to
prevent silent duplicate counting.

## Tests and timing

- Constrained optimization tests: **20 passed**.
- Predictive/constraint regression tests: **17 passed**.
- Total current focused suite: **37 passed**.
- Synthetic actor: 132k parameters, batch 256, CPU single-thread.

| Update | Mean time | Relative to one weighted backward | Solver convergence |
|---|---:|---:|---:|
| One weighted backward | 2.01--2.07 ms | 1.00x | n/a |
| Two-objective Pareto block | 4.74 ms | 2.36x | 100% |
| Three-objective Pareto block | 6.58 ms | 3.17x | 100% |
| Six-UAV/four-target joint CVaR forward + backward | 0.39 ms | 0.19x | n/a |

The earlier five-objective solve cost about 7.96x and exhausted its iteration
budget under an unnecessarily strict tolerance. It is rejected. Two/three
objective alternating blocks use tolerance `1e-6`, converge in the benchmark,
and bound the extra training cost.

## Current integration gate

Do not insert the condition into randomly shuffled per-agent PPO minibatches.
Each constraint minibatch must contain complete UAV teams in temporal rollout
order, or a separate full-rollout actor pass must compute the joint ratio.
Before enabling the path in production training, the next stage must pass:

1. **Passed:** team-aligned minibatch reconstruction with duplicate-`P_D`
   consistency checks;
2. default-off trainer integration with identical legacy behavior when off;
3. finite-gradient, PPO ratio and KL-rollback tests;
4. a short fixed-seed smoke run showing reduced detection-CVaR violation
   without power-budget violation or excessive KL;
5. paired multi-seed evaluation before any claim of system improvement.

This gate prevents a mathematically plausible but incorrectly factorized
constraint loss from being promoted merely because its training curve falls.

## Default-off PPO integration and fixed-seed smoke

The target-wise condition is now connected to `MAPPTrainer` behind
`marl.constrained_cvar_ppo_enabled: false`.  Enabling it adds one full-rollout
actor pass after the ordinary PPO minibatches.  The pass reconstructs the
factorized joint-action ratio over complete UAV teams; the existing final KL
transaction accepts or restores the ordinary PPO and condition steps together.
The trainer also reports target-wise CVaR residuals, dual progress, finite
gradient status, exact RF-budget excess and updates-to-first-feasibility.

An engineering-only `training_fixed_seed` option is also default-off (`-1`).
The smoke profile pins every episode reset to seed 31415; ordinary training
continues to draw fresh geometries or use the configured seed bank.

Six fixed-seed updates were run for both the condition-enabled profile and its
legacy-control profile.  The enabled final CVaR violation was 0.579275 versus
0.585716 for control (improvement 0.006441), while the six-update mean was
0.586928 versus 0.587473 (improvement 0.000545).  Every condition gradient was
finite, peak RF-budget excess was 0 W, peak attempted KL was 0.020422 below the
normal transactional limit of 0.03, and no normal-run rollback was required.
Excluding first-update warm-up, the enabled path averaged 0.3243 s/update
versus 0.2928 s/update for control (10.7% overhead in this small GPU smoke).
No update reached the 0.01 primal tolerance, so first-feasible-update remains
unset and the short-run convergence gate is **not passed**.

A deliberate `target_kl=0.0001` fault injection produced attempted KL
0.007780, set `actor_update_rejected=1`, and restored accepted KL to 0.  Thus
the rollback path is executable, but the present short-run evidence supports
only correct/safe integration, not CVaR convergence or policy improvement.

## Progressive optimization stage 1: fixed-structure distributed power

The first alternating block is now implemented independently of GNN, GRU,
PPO, teachers and certificates in
`uav_isac/optimization/distributed_power_primal_dual.py`.  It uses an
operator-normalized primal--dual hybrid-gradient update for a regularized
minimum-power problem. Each UAV projects its own row onto the intersection of
the non-negative RF simplex and the fixed structural support; nodes exchange
only target-level contribution vectors and target prices.

Hard properties are locked by tests: no power outside the structure, no row
budget excess, target/UAV permutation equivariance, explicit infeasibility,
independent primal/dual/consensus convergence gates, and exact checkpoint
resume. A seed-20260908 benchmark over 100 randomly generated feasible cases
(K=3--8, Q=2--6) converged in 100/100 cases with maximum primal violation
1.797e-4, zero power-budget excess and zero aggregation mismatch. Median cold
start convergence was 627 message rounds, p95 1217.2, at 39.86 ms/case.

The correctness gate passes, but the cold-start communication-round count was
too high for direct per-frame deployment.  The temporal warm-start and bounded
acceptance/fallback rule below is the required deployment gate before any PPO
alternating update.

## Progressive optimization stage 2: bounded temporal execution

`BoundedTemporalPowerController` now carries the previous feasible power and
target-price iterates across frames.  Every frame gets a fixed iteration
budget; a candidate is accepted only when it preserves the hard mask/budget
and does not increase the maximum target residual beyond the projected
incumbent (within `acceptance_tolerance`).  Otherwise the incumbent is
returned.  This gives the execution path a deterministic wall-clock bound and
a fail-closed fallback without introducing a teacher or a hidden central LP.
The controller state is included in the environment snapshot, so
counterfactual or evaluation replay restores the warm-start iterate and target
prices exactly.

The seed-20260908 temporal benchmark used 50 sequences × 20 frames and only
20 primal--dual rounds per frame.  Warm starts accepted 884/1000 candidates;
116 frames used the incumbent fallback.  The accepted warm path had mean
maximum target residual `1.413e-6` (maximum `1.76e-4`, below the configured
`2e-4` primal gate), while preserving zero hard budget/mask violation.  Mean
power was 2.10158 W versus 2.30675 W for the deliberately cold incumbent
comparison, a 0.20517 W reduction.  The cold comparison can remain exactly
feasible by overspending, so the power reduction—not a raw violation count—is
the meaningful temporal result.

An iteration-budget sweep on the same seed showed the expected latency/energy
trade-off: with only 8 rounds per frame, 50 sequences × 20 frames accepted
991/1000 updates, mean residual was `9.56e-8` (maximum `5.03e-5`) and mean
power saving versus cold was 0.10486 W.  Twenty rounds saved 0.20517 W but
accepted 884/1000 updates.  Thus 8 rounds is a viable low-latency candidate
for a later deployment profile, while 20 remains the conservative audit
setting; neither choice changes the hard budget gate.

The controller is connected to the environment only behind
`marl.distributed_primal_dual_power_enabled: false`.  It requires the existing
analytical sensing-power path and a positive `analytical_sensing_power_reserve_pd`,
and is mutually exclusive with replicated, intercept-constrained, task-gauge,
and bargaining power objectives.  `config/exp_distributed_primal_dual_power_smoke.yaml`
raises the sensing cap solely for a deterministic reachability smoke (seed 1):
two real environment frames passed with zero RF-budget excess, zero consensus
residual, a successful warm start on frame 2, and target residual below
`2e-4`.  The focused regression suite is **70 passed** after adding trainer
diagnostic and state-roundtrip coverage.

This stage therefore passes the bounded-execution and hard-safety gate.  The
rollout diagnostics and the paired PPO coupling smoke are recorded next; the
joint alternating update remains gated on their results.

## Progressive optimization stage 3: PPO coupling smoke

The rollout now records the selected/candidate primal residuals, candidate
stationarity and dual residual, consensus residual, target-price summaries,
warm-start rate, fallback rate, convergence rate and iteration count.  A
three-update fixed-seed PPO pair was then run with the same seed 31415:

| profile | mean P_D | max CVaR violation | max RF excess | max attempted KL | warm-start | fallback |
|---|---:|---:|---:|---:|---:|---:|
| bounded primal--dual | 0.7264 | 0.2650 | 0 W | 0.022927 | 98.96% | 0% |
| central analytical control | 0.9685 | 0.3141 | 0 W | 0.023345 | n/a | n/a |

The distributed block used the full 32-round budget in this smoke and had
zero selected target residual, zero consensus residual and finite gradients;
no PPO actor rollback occurred.  The lower mean `P_D` is expected here: the
distributed objective is minimum sensing power subject to the `P_D=0.6`
reserve, whereas the central control maximizes max-min deflection and spends
the remaining sensing budget.  This is therefore a resource/performance
trade-off demonstration, not evidence of detection superiority.  The CVaR
convergence gate is **not passed** (one update still had 0.265 residual), and
the dual/stationarity gates are not yet reached within 32 rounds.  Keep this
profile as an engineering diagnostic until a multi-seed Pareto comparison
shows an explicitly chosen energy--QoS operating point.

## Progressive optimization stage 4: differentiable multi-frame closure

The earlier stages proved individual safety properties but did not make the
inner optimization trajectory part of the learning objective.  The opt-in
`DifferentiableTemporalPowerUnroll` now closes that gap.  At frame `t`, the
actor's rate logits and RF-power logit define the expected sensing budget,
while its sensing logits define `p_t^0`.  Four finite projected primal--dual
iterations produce `p_t^L` and `lambda_t^L`; both are carried into frame
`t+1`.  Consequently, a loss at the last frame has a gradient to the first
frame unless `temporal_unrolled_power_detach_between_frames=true` explicitly
requests truncated BPTT.  Every inner iterate is exactly projected onto the
fixed structural mask and per-UAV RF simplex.

The outer update keeps five losses independent: negative worst-target
detection, negative mean detection, expected communication load, sensing
energy and frame-to-frame power switching.  Their normalized gradients enter
the stated minimum-norm simplex QP.  For the current five objectives, all 31
simplex faces are enumerated and solved in double precision, giving an exact
small-QP active-set solution rather than a fixed loss-weight vector or an
iteration-limited first-order approximation.  Target-wise detection CVaR is
added afterward as an augmented-Lagrangian constraint gradient and cannot be
traded away by a Pareto coefficient.  PPO, this unrolled step and any enabled
auxiliary update remain one transaction under the final full-rollout KL
rollback.

The physical unroll is not treated as a perfectly calibrated simulator.  The
training detection used for objectives and tail selection is

\[
\widetilde P_D=P_D^{model}+\operatorname{stopgrad}
  (P_D^{rollout}-P_D^{model}).
\]

Its forward value and CVaR tail therefore match the real rollout on eligible
frames, while its local derivative still passes through the multi-frame
optimizer.  Incomplete fixed-owner structures have no valid gain matrix; they
terminate the BPTT segment instead of receiving fabricated coefficients, and
the trainer reports eligible/skipped frames and coverage.  Those invalid
structure frames remain a separate structure-learning failure mode.

The seed-31415 three-update smoke is in
`results/temporal_unrolled_power_seed31415_v6/train_metrics.csv`.  All three
multi-frame Pareto QPs converged (maximum gap `1.99e-8`), all unrolled and
executed RF-budget violations were zero, and the maximum attempted KL was
`0.02293` below the transactional limit `0.03`; no rollback occurred.  Mean
unroll coverage was 95.83% (one update skipped four incomplete-structure
frames).  The unrolled model's mean absolute detection bias was about 0.2505,
which validates the need for the rollout-value correction.  Real target-wise
CVaR still violated by 0.265 on one update and the three-update convergence
gate did not pass.  Mean wall time was 1.22 s/update versus 0.54 s/update for
the earlier bounded-power PPO smoke; the five reverse-mode objective passes
are now the principal training-time cost.

Unit and integration coverage is now 77 focused tests, including exact
projection equivalence, cross-frame gradient presence/explicit truncation,
inner-iteration residual behavior, five-objective Pareto plus CVaR gradient
assignment, default-off/fail-closed configuration, real rollout wiring and KL
transaction checks.

A temporal-path-specific rollback injection used `target_kl=0.01`.  Ordinary
PPO completed without early stop (`approx_kl=0.00205`), the unrolled Pareto
step was applied, and the combined attempted KL reached `0.02293`, above the
transaction limit `0.015`.  The trainer restored the actor and optimizer
state, reporting `actor_update_rejected=1` and accepted KL `0`, so the new
multi-frame update is covered by the rollback rather than bypassing it.

### Stage-4 execution-alignment correction

The first Stage-4 implementation used the actor sensing allocation as
`p_t^0` in the differentiable layer, but the real bounded controller started
only from its previous iterate (or zero on a cold frame).  That made an actor
variable trainable in the model while causally ineffective in execution.  The
real controller now takes the same actor/history convex combination and
projects it before the finite PDHG rounds.  Its fallback baseline remains the
projected previous feasible solution, so the existing no-regression and hard
budget guarantees are unchanged.

On the same seed-31415 three-update binary-DD smoke this reduced mean model
`P_D` bias from about `0.2505` (`v6`) to `0.03190` (`v7`) and increased mean
executed `P_D` from `0.7264` to `0.9397`.  The latter is not a free performance
claim: the newly added executed RF audit shows mean sensing, communication and
unused powers of `0.49037`, `0.78883` and `0.72080` W per frame respectively.
Maximum RF-budget excess remained zero.  One update still had a target-CVaR
residual of `0.28984`, so the binary-DD convergence gate remained failed.

## Progressive optimization stage 5: feasible-structure convex hull

The optional `temporal_feasible_structure` layer extends the multi-frame graph
to the discrete owner/Tx schedule without pretending that argmax or MILP has
an unbiased derivative.  Let `C_t` be the finite set of complete schedules
that obey global single-role Tx/Rx/idle operation, no self-edge, one owner per
target, `K_q_max`, receiver report capacity and current physical support.  The
relaxed temporal policy is

\[
 \pi_t(z)=\frac{\exp\{[\langle s_t,z\rangle+
 \kappa\langle z,\bar x_{t-1}\rangle]/\tau\}}
 {\sum_{z'\in C_t}\exp\{[\langle s_t,z'\rangle+
 \kappa\langle z',\bar x_{t-1}\rangle]/\tau\}},\qquad
 \bar x_t=\mathbb E_{z\sim\pi_t}[z].
\]

This is the unique maximizer over schedule distributions of expected score
plus `tau` times Shannon entropy.  Because every hard schedule is feasible and
all schedule constraints are linear, `xbar_t` is in `conv(C_t)` and satisfies
those constraints exactly.  The power layer then uses the capped-simplex KKT
projection

\[
 p_{iq}=\operatorname{clip}(v_{iq}-\vartheta_i,0,
 b_i\sum_j\bar x_{ijq}),\qquad
 \sum_qp_{iq}\le b_i,
\]

where a scalar monotone bisection finds `vartheta_i`.  Thus the relaxed native
structure--power condition and the total RF budget hold at every inner
iteration.  The actor score uses only transmitter-side sensing allocation:
a bistatic receiver does not radiate a second sensing waveform.  Simulator
coefficients are CTDE training inputs and never enter decentralized actor
observations.  The run record explicitly sets
`temporal_feasible_structure_biased_surrogate=1`; the implementation claims an
exact gradient of the entropy relaxation, not of the discrete hard scheduler.
By default `temporal_feasible_structure_hard_forward=true` uses the
straight-through hard schedule in the physical forward pass; setting it false
is retained only as a clearly named within-frame time-sharing ablation.

The enumerator is bounded.  If the number of feasible schedules exceeds the
declared bound it raises instead of silently pruning and retaining a false
convex-hull claim.  If `C_t` is genuinely empty, that physical frame breaks
the BPTT segment and increments `temporal_feasible_structure_empty_frames`;
zero-gain edges are never fabricated.

### DD-physics correction and fixed-seed gate

The legacy binary model `1[g_DD >= g_min]` produced five empty-set frames in
the 96-frame seed-31415 smoke.  It was not suitable for a movement/structure
gradient: an arbitrarily small displacement can delete an entire edge.  The
Stage-5 profile therefore uses the already implemented canonical coefficient

\[
 a_{ijq}=I_{support}(\tau,\nu)|A(\tau,\nu)|^2
 C_{ijq}/(R_{iq}^2R_{jq}^2),
\]

which retains a strict zero outside the OTFS unambiguous region and continuous
ambiguity energy inside it.  This is a physical-model correction, not an
algorithm-versus-baseline improvement.

`results/temporal_feasible_structure_seed31415_v4/train_metrics.csv` gives:

- zero empty-set frames and 100% temporal coverage;
- zero structure-convex-hull and relaxed structure--power violation;
- zero executed and unrolled RF-budget excess;
- maximum target-CVaR residual `-0.01089`; the three-update feasibility streak
  reached three and the configured outer risk gate reported converged;
- maximum attempted KL `0.02312 < 0.03`, with no rollback;
- mean executed sensing/communication/unused power
  `0.51780/0.78881/0.69339` W per frame;
- mean `P_D=0.86732`, mean worst-target `P_D=0.77375`, and unrolled model-bias
  MAE `0.07775`.

This is only one fixed geometry repeated three times.  It proves execution of
the gates, not seed generalization.  The two K=2 role orientations also retain
entropy `log(2)` and only about 52% edge identity agreement: under symmetric
bistatic geometry these schedules are physically equivalent, so forcing a
specific Tx/Rx label would be label-learning rather than systems optimization.

### Convergence/latency ablations

Keeping the reverse graph at four inner layers and increasing only real PDHG
work from 32 to 64 rounds reduced worst observed execution stationarity from
`0.08184` to `0.001113` and dual residual from `2.1409` to `0.17789`, at mean
training time `6.96` versus `6.16` s/update (about 13% higher).  `P_D`, CVaR,
power and KL were essentially unchanged.  Going to 128 rounds did not reduce
the stationarity plateau and increased time again; it is rejected.  Increasing
the reverse graph from four to eight layers doubled training time to `12.30`
s/update without improving the unrolled residual; it is rejected.  A common
100x scaling of the convex objective preserves its infinite-iteration argmin
but changed the bounded trajectory, reduced `P_D`, and worsened stationarity;
it too is rejected.

The retained convergence audit profile is therefore **64 execution rounds / 4
reverse-mode layers**.  It improves finite-round numerical residuals but still
does not pass the strict inner stationarity/dual convergence gates.  Outer
CVaR feasibility and exact hard constraints must not be reported as inner
solver convergence.

The design is consistent with differentiable optimization layers based on KKT
or unrolled solves and with entropy relaxations of discrete assignments, but
neither ingredient alone is novel.  Any paper-level novelty claim must be
limited to—and separately ablate—the combined teacher-free temporal feasible-
structure convex hull, native structure--power projection, target-wise CVaR
dual condition, Pareto gradient QP and bounded distributed execution.

### Stage-5 hard-forward and rollback verification (2026-09-08)

The default structure path now uses a straight-through estimator: the forward
simulation receives the exact MAP schedule, while the backward pass uses the
Jacobian of the entropy-relaxed convex-hull mixture.  This is recorded as a
biased surrogate rather than being presented as an exact gradient of the
discrete scheduler.  On the seed-31415 three-update run, hard-forward changed
mean executed `P_D` by less than `1e-5` relative to the soft-forward audit and
kept all structural, RF-budget, CVaR and KL gates unchanged.

A second fixed seed (31416) also passed the outer risk gate: maximum CVaR
residual `-0.00997`, full temporal coverage, zero empty feasible frames, zero
convex-hull and RF-budget violations, and maximum attempted KL `0.02395`.
These are reproducibility checks for the execution path, not a claim of seed
generalization.

Finally, a deliberate KL fault-injection smoke with `target_kl=0.01` produced
attempted KL values `0.02311/0.03059/0.02187`; all three PPO transactions were
rejected atomically (`accepted_kl=0`), while RF-budget excess remained zero.
This verifies that rollback protects the coupled structure--power update and
does not leave a partially applied actor/critic state.

The focused regression set currently passes (`56 passed`), including exact
feasible-structure enumeration, convex-hull constraints, capped-simplex KKT
projection, straight-through gradients, temporal unrolling, CVaR integration,
and distributed power accounting.

### Performance cause audit and debug instrumentation (2026-09-08)

Before adding another optimization block, the execution path was instrumented
with objective-gradient norms/cosines, model-bias moments, real-vs-raw
detection means, and phase timings.  The seed-31415 debug run shows:

* The raw unroll overestimates the realized mean detection by `0.0462` on the
  first update and still has bias standard deviation `0.12--0.14` with maximum
  absolute error `0.36--0.39`.  The error is therefore state/structure
  dependent, not a removable scalar offset.  In particular, raw CVaR can be
  positive (`+0.093`) while the detached-calibrated rollout residual is
  negative (`-0.036`), so the current calibrated gradient is a local surrogate,
  not a physical feasibility certificate.
* The communication-load and sensing-energy gradients have cosine
  `-0.998/-0.999`, while mean/worst detection point toward sensing energy with
  cosine about `-0.4--0.5`.  The minimum-norm Pareto QP consequently assigns
  zero weight to detection objectives in the 32-round smoke.  This is the
  mathematically expected result of treating complementary RF resources as
  independent objectives, but it is a performance failure if detection is the
  desired improvement direction.
* A causal sensing intervention held movement, communication, total RF and the
  other UAV fixed.  On one test episode, `96.875%` of interventions had more
  than `0.002` worst-target opportunity, the worst-target oracle gap was
  `0.02344`, and the actor's current allocation was best only `3.125%` of the
  time.  This confirms headroom in target-level sensing allocation; it does not
  by itself prove a multi-seed gain.
* Runtime is update-bound: 6.70 s/update versus 0.45 s rollout in a one-update
  profile.  The temporal block consumes 6.48 s, of which 5.09 s is the five
  separate autograd traversals used to form the Pareto gradient rows.  The
  physical P0 solve is not the dominant runtime bottleneck.

The code path also makes two causal boundaries explicit: analytical sensing
power overwrites the actor's target split after using it as a finite-solver
proposal, and the feasible-structure schedule is currently a training
surrogate while P0 remains the execution authority.  These explain why a
nonzero gradient or a lower surrogate loss need not translate to higher
executed `P_D`.  The next optimization must therefore first remove or model
these two mismatches and redesign the redundant RF objective set; simply
increasing unroll rounds or network width is not justified by the audit.

### Stage-6 exact Pareto-gradient path (2026-09-08)

The first performance change keeps the objective vector, simplex QP and
constraint-gradient separation unchanged.  `assign_constrained_pareto_gradients`
now requests a batched vector-Jacobian product from PyTorch, producing all
objective rows in one reverse-mode call.  If a custom operator or an older
runtime does not support batched VJPs, the scalar reverse-pass implementation
is selected automatically.  The optional `batched_vjp=False` path is retained
as the reference for tests and ablations; no optimization coefficient or
physical constraint is changed.

The equivalence test covers both objective rows and the independent CVaR
constraint gradient: weights, applied gradients, raw norms and pairwise
cosines agree to floating-point tolerance.  The focused suite passes `56`
tests.  On a one-thread CPU microbenchmark (5 objectives, 132,869 parameters,
20 timed updates), mean Pareto-gradient time changed from `6.55 ms` (scalar)
to `6.40 ms` (batched), with identical stationarity norm; the gain is modest
because the exact active-set QP and Python diagnostics remain.  The fixed-seed
temporal smoke still reports zero RF-budget excess, calibrated CVaR residual
`-0.01089`, and attempted KL `0.00179`; therefore this stage is classified as
an implementation acceleration, not an algorithmic result.

On the actual CUDA temporal smoke, three scalar and two batched reference
runs gave mean Pareto times `4.31` and `4.38` s/update respectively (run-to-run
noise is larger than the difference), while all `P_D`, CVaR, KL and RF values
were identical.  Batched VJP is consequently an opt-in implementation choice,
not evidence of a guaranteed wall-clock gain on this hardware.

The next stage remains gated: replace the redundant communication/sensing
objective tradeoff by an active RF-budget constraint/dual price.  That change
must be evaluated separately from the implementation acceleration and the
actor-centred closure below.

### Stage-7 actor--execution proximal closure (2026-09-08)

The bounded fixed-structure solver now accepts an optional actor-centred
quadratic term

\[
  \frac{\rho}{2}\lVert p-p_{\rm actor}\rVert_2^2,
\]

so its primal gradient contains `rho * (p - p_actor)`.  The actor proposal is
projected onto the same masked row simplex before it is used as the centre;
therefore the proximal term cannot create negative, structurally invalid, or
over-budget power.  The differentiable temporal unroll uses the identical
curvature and current-frame proposal centre.  With `rho=0` the previous
algorithm is exactly recovered, and the option is disabled by default.

The K=2/Q=2 seed-31415 stress ablation (`rho=0.1`, three updates) improved
mean executed `P_D` from `0.86731` to `0.86886`, mean worst-target `P_D` from
`0.77377` to `0.77509`, and model-bias MAE from `0.08222` to `0.08100`.
Maximum attempted KL changed only from `0.000930` to `0.000934`, RF-budget
excess stayed exactly zero, and mean update time increased by about `1.1%`
(`5.30` to `5.36` s).  This is a promising causal-closure signal, not a
generalization result: it is one repeated geometry and a deliberately strong
stress coefficient.  The production/default path therefore remains
unchanged; the opt-in profile is retained for multi-seed and coefficient
ablation before any default switch.

The focused regression set passes `56` tests, including the new VJP equivalence,
total-schema fallback, and actor-proximal hard-feasibility checks.

### Stage-8 active RF-budget tangent (2026-09-08)

The Pareto assignment now has an optional `tangent_constraint_losses` input.
For an active equality residual `h(p)=0`, objective rows are projected with

\[
P_{\mathcal T}=I-J_h^\top(J_hJ_h^\top)^+J_h,
\]

before the same simplex minimum-norm solve.  The projection is detached from
the learning graph (no second derivatives), and the independent CVaR
inequality gradient is still added through its dual condition.  A unit test
verifies that an active RF equality removes the normal component exactly.

The trainer computes the measured normalized resource residual
`mean(comm_fraction + sensing_power/total_budget - 1)` and activates the
projection only inside a configured slack band.  In the normal seed-31415
smoke the residual is `-0.32884`, so the strict `1e-6` gate correctly remains
inactive.  A deliberately labelled `0.40` activity-band stress run exercised
the projection: tangent-gradient norm fell from `0.03187` to `4.9e-8`, while
CVaR residual (`-0.01089`), RF excess (`0`), KL (`0.00179`) and all physical
metrics stayed finite.  It did not improve detection because the projected
resource directions were already the dominant directions; this is evidence
that the redundant RF objectives cannot be fixed by a projection alone.

The tangent mode is therefore opt-in and not a default optimization.  The
remaining research step is to make the QoS/CVaR target binding (or introduce
an explicit communication service constraint) and then compare the resulting
constrained Pareto frontier across multiple seeds, rather than tuning an
inactive-budget tolerance.

### Stage-9 adaptive projection adapter (2026-09-08)

The single projection is now wrapped by a rank-aware adapter.  It automatically
selects `identity`, `single`, or `multi` mode from the number of active
equalities and the numerical rank of their Jacobian.  Multiple constraints are
projected through an SVD orthonormal normal basis, so duplicated RF/KL/QoS
constraints do not introduce a second copy of the same normal direction.  The
adapter reports mode, rank and tangent-norm ratio for debugging; it never
learns a task weight and never bypasses a hard projection.

The unit tests cover a dependent two-constraint case (mode `multi`, rank `1`)
and the combined tangent-plus-CVaR gradient path.  The stress smoke recorded
mode `single`, rank `1`, norm ratio `0.5634`, RF excess `0`, CVaR residual
`-0.01089`, and KL `0.00179`.  This validates automatic switching and
diagnostics, but does not claim a detection gain: the underlying active
resource manifold still leaves essentially no tangential descent in this
geometry.

### Stage-10 actor-proximal multi-seed ablation (2026-09-09)

The Stage-7 signal was expanded without changing code identity. Five
independent training seeds (`31415--31419`) were run for both `rho=0` and the
stress value `rho=0.1`; every trained policy was evaluated on the same ten
holdout seeds (`30001--30010`). The training seed, rather than each nested
evaluation episode, is the independent statistical unit. Every V2 completion
hash and resolved proximal coefficient is checked by
`tools/summarize_actor_proximal_ablation.py`; the compact result is stored in
`artifacts/research/actor_proximal_ablation_v1.json`.

| Training seed | steady delta | weak3 delta | worst delta |
|---:|---:|---:|---:|
| 31415 | +0.0007218 | +0.0007218 | +0.0010892 |
| 31416 | +0.0000000 | +0.0000000 | +0.0000000 |
| 31417 | +0.0000013 | +0.0000013 | +0.0000025 |
| 31418 | +0.0000000 | +0.0000000 | +0.0000000 |
| 31419 | -0.0000000 | -0.0000000 | -0.0000000 |

Across training seeds, the mean worst-target delta is `+0.0002183`, its median
is effectively zero, and its equivalence-censored cluster-bootstrap 95%
interval is `[0, +0.0006540]`. At the `1e-8` numerical-equivalence boundary,
two clusters are positive and three are ties (one-sided exact sign
`p=0.25`). Only seed `31415` exceeds the explicitly post-hoc practical
threshold of `+0.0001`. Pooling the 50 nested evaluation outcomes would
incorrectly inflate the sample size and is deliberately not used.

All evaluation runs retain QoS feasibility `1.0` and exactly zero measured RF
budget excess. Attempted KL remains below the transactional `0.03` bound.
The short training audit is not uniformly CVaR-feasible, however: the strong
arm reaches residual `0.06762` on seed `31416`, above the `0.01` tolerance.
Consequently `rho=0.1` fails both the practical generalization rule and the
candidate safety rule. It remains an opt-in diagnostic and is not promoted
to the default algorithm. This is a useful negative result: a fixed proximal
coefficient closes the actor--solver path on one favorable seed but does not
solve the underlying causal mismatch generally.

The next algorithm experiment should therefore target a binding constraint
mechanism, not tune `rho` on this ceiling-prone smoke. Before execution it
must freeze a harder non-saturated scenario bank, a CVaR-feasibility rule, a
minimum relevant tail-effect size, and independent training seeds. Candidate
selection and confirmation should use disjoint seed banks.
