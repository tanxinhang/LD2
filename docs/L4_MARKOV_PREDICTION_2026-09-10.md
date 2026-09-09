# L4 causal Markov prediction

The L4 plant and local belief both use the same linear-Gaussian CV transition
with acceleration standard deviation 1.5 m/s2. The continuous target state is
therefore already Markov. Introducing an unsupported discrete CT/CA regime
chain would change the model rather than improve inference. This experiment
instead applies the exact Chapman--Kolmogorov recursion and uses the predicted
one-second target mean in the existing distributed bistatic assignment.

The predictor is causal, consumes each viewer's existing local belief mean and
velocity, reflects the mean at the simulator boundary, and does not use future
measurements or target truth. Frozen profiles retain zero prediction frames.
The assignment hold remains 20 frames; power, communication, sensing budget
and candidate count are unchanged.

On four calibration seeds, H10 changed steady/weak3/worst from
0.9000/0.8287/0.8051 to 0.9092/0.8577/0.8460 and raised the minimum worst score
from 0.6250 to 0.6791. H20 improved the mean less and caused a 0.06 single-seed
regression. H40 reduced worst to 0.7650 and the minimum to 0.4588, showing that
long open-loop mean prediction is harmful.

Before confirmation, eight untouched non-test seeds were selected with fixed
RNG seed 20260910 across the stored easy/medium/hard geometry strata. The exact
seed list and acceptance rules are frozen in
`artifacts/research/l4_markov_prediction_h10_v1.json`. This is development
confirmation, not formal evidence; a larger bank is required for a narrow
confidence interval.

The preregistered development confirmation rejected the endpoint-mean H10
candidate. On the eight untouched seeds, baseline versus candidate
steady/weak3/worst was 0.94847/0.89385/0.87547 versus
0.94277/0.88661/0.86939. The candidate worst minimum was 0.72881 versus
0.75218, with four wins and four losses. SCORE, WEAK3 and TAIL therefore
failed; QoS, L0 and resource guards passed. The next hypothesis must integrate
the path distribution and belief covariance rather than tune another fixed
endpoint horizon.

A follow-up compressed path-risk proxy (three horizon samples plus one
position-standard-deviation radial margin) was rejected on the four calibration
seeds: worst changed by -0.0031 and the minimum fell from 0.6250 to 0.5336
(`pilot-a49cefd3f74f4a9739b1`). The proxy assigned persistent priority to
uncertain targets without re-solving the downstream structure and power at
each future state. Its implementation was removed.

The active next stage is a finite-state Markov assignment dynamic program in
`uav_isac/prediction/markov_assignment.py`. Its states are complete assignment
vectors; stage costs will come from future physical solves, and transitions pay
normalized endpoint churn. The dynamic-programming kernel is deterministic and
tested, but it has no live action authority yet. No performance claim is made
until a shadow evaluator demonstrates that its predicted assignment ranking
agrees with executed closed-loop Deflection.

The optimizer now also contains an action-conditioned scenario-tree beam
search. Unlike the first exogenous-cost DP, every branch carries its own
continuous state, so future UAV geometry depends on the complete preceding
assignment path. Transition and stage-cost callbacks are required to be finite
and deterministic. This closes the state-aliasing error that would otherwise
score different assignment histories at the same fictitious geometry. The
physical callback is now implemented in
`uav_isac/prediction/markov_physical_assignment.py`. Each reached geometry is
scored by a conditional-mean channel calculation followed by the exact
fixed-structure max-min sensing-power LP. Swerling-II uses its exact unit-mean
exponential multiplier. Rician reporting reliability is integrated with
deterministic tensor-product Gauss--Hermite quadrature, which agrees with a
500,000-sample Monte Carlo check and does not consume the live simulator RNG.

The target belief is no longer collapsed to its mean. Equal-weight
spherical-radial cubature points reproduce each target's full `[x,y,vx,vy]`
mean and covariance, then pass independently through bistatic geometry, OTFS
support and expected report-link physics. Since a target's coefficient tensor
depends only on that target's marginal state at fixed UAV geometry, this costs
`8Q` physical evaluations rather than an exponentially large joint target
tree. The stage can use the expected coefficient or a non-negative
`mean - beta*standard_deviation` lower-confidence coefficient before the exact
power solve.

This is still a shadow kernel, not an online-authority result. The reporting
structure is frozen over the planning horizon, target covariance horizons must
come from causal local beliefs, and no closed-loop L4 gain is claimed yet. The
candidate generator now constructs a deterministic target-priority swap
neighborhood around the existing causal movement assignment. Every swap
preserves the complete per-target responsibility histogram; the incumbent is
always candidate zero and K16/Q16 is capped at 32 candidates instead of
enumerating `16^16` assignments. The next gate is to feed causal local-belief
covariance horizons and these candidates into shadow logging, compare predicted
rankings with actually executed next-stage outcomes, and only then preregister
an authority experiment.

The fixed swap neighborhood is only a control, not the intended optimizer: it
does not know geometry or bistatic/power coupling. A second proposal layer in
`uav_isac/prediction/markov_graph_assignment.py` builds a target-wise KNN
UAV--target adjacency graph, retains every incumbent edge, and gives each graph
edge a count-preserving two-exchange value from the full expected-physics and
max-min-power solve. A global bipartite matching composes those edge messages;
because coupled exchange values are not additive, the joint proposal receives
one final full physical solve and is rejected if it does not strictly improve
on the incumbent. Thus graph sparsification can change ranking but cannot force
a proxy-degrading shadow action.

This physics-weighted KNN layer is deliberately an offline shadow oracle: even
after KNN pruning, solving the full covariance-integrated physics problem for
every edge is too expensive for online K16 control. Its accepted/rejected
two-exchange outcomes will provide rollout-grounded supervision for a sparse
message-passing edge residual. That later GNN will learn actual counterfactual
return residuals, not imitate the old controller, and its single composed
proposal will retain the exact final physical rejection gate.

The first fixed-seed mechanism tests rejected direct Hungarian composition:
on 12 K8/Q8 states it never beat the bounded blind-swap optimum and delivered
only 0.00417 mean improvement versus 0.00776. Replacing composition with
budgeted best-improvement local search changed the result. On 256 K16/Q16
states at 32 evaluations, distance-ranked KNN improved the stage objective by
0.005032 versus 0.004782 for blind swaps; the paired mean advantage was
0.000250 with bootstrap 95% interval `[0.000053, 0.000462]`.

Reusing the incumbent max-min LP dual prices as target messages improved the
same-state result to 0.005132, a paired advantage of 0.000350 with interval
`[0.000174, 0.000529]`. This remains a mechanism screen on generated geometry,
not L4 closed-loop evidence. A 16-state covariance/LCB stress screen was also
positive but too small for confirmation. Marginal-target evaluation and cached
receiver report reliability reduced its mean graph-search time from 1.91 s to
0.55 s without changing any score. The exact sequence, including the rejected
variant, is recorded in
`artifacts/research/markov_graph_assignment_mechanism_v1.json`.

## Fixed-lag inverse-estimation screen

The forward Markov model and the noisy inverse problem are now separated in
code. `uav_isac/environment/fixed_lag_smoother.py` provides an exact
centralized Kalman/RTS fixed-lag reference with missing-observation support,
time-varying measurement covariance, Joseph covariance updates and explicit
smoothed process residuals. It is not connected to action authority. The last
smoothed state is deliberately tested to equal the last filtered state: a
causal window cannot manufacture a future measurement, so its online value is
the denoised history and evidence about temporal model mismatch.

A 256-seed synthetic mechanism screen found that smoothing reduced historical
position MSE to 0.338 of filtering under the current L4-style independent
white-acceleration model. However, extrapolating its recent residual
acceleration changed the H10 endpoint error by -0.0017 m (95% bootstrap
interval `[-0.0056, 0.0021]`, win rate 0.469). This is a No-Go for adding
residual acceleration to the current L4 controller: `Target._step_cv` samples
acceleration independently each frame, so there is no latent acceleration
memory to predict.

Under an explicitly labelled AR(1), coefficient-0.9 persistent-acceleration
control, the same causal estimator improved endpoint error by 0.0170 m
(`95% CI [0.0117, 0.0222]`, win rate 0.656). That positive control verifies the
mechanism but is not evidence for the current simulator. ADMM decomposition or
a switching Markov model is therefore gated on first defining and detecting a
scientifically justified persistent motion regime. Full results are recorded
in `artifacts/research/fixed_lag_smoother_mechanism_v1.json`.
