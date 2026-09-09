# Constraint-native predictive control

`CertifiedBipartiteGNN` retains its historical class name for checkpoint/API
compatibility, but is now a shadow-mode constraint-native proposal model. It uses
factorized `(viewer,node,target)` tokens and returns owner logits, transmitter
logits, projected-power logits, dual warm starts, a learned risk residual,
next-frame detection logits and a temporal detection delta. A receiver-conditioned
low-rank decoder scores transmitters only for the small receiver beam, so it
can learn bistatic pairing effects without emitting a
`(viewer,tx,rx,target)` dense tensor.

The required online order is:

1. frame `t` public state starts asynchronous inference for `t+1`;
2. frame `t+1` consumes the prefetched proposal if ready;
3. structural projection repairs owner/coverage/cardinality;
4. selected-only native physics recomputes the execution gains;
5. row-simplex projection enforces non-negative per-UAV power budgets by construction;
6. native QoS residuals and sparse LP/KKT residuals decide whether the proposal is admissible;
7. any failure expands the pool or uses the last feasible incumbent.

Online composable-certificate payloads are not model inputs and are not required
for normal execution. Conservative gain envelopes are risk terms; offline
certificates remain evaluation artifacts rather than protocol fields.

The current v0 supplies steps 1--3 and the model/service contracts.  It is not
authorized to control the environment until steps 4--7 are wired and shadow
recall/OOD gates pass on unseen seeds and unseen K.

`PredictiveRefreshGate` additionally codifies the five-frame protocol: hold
frames reuse the certified edge indices and recompute selected physics only;
the predictor is consumed only at a refresh boundary. A boundary proposal is
accepted only if selected physics, omitted-edge bounds and solver residuals
all pass, otherwise the result is an explicit full-exact fallback.
