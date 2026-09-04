# Spectrally Certified Information Reuse (SCIR)

## 1. Why DD conflict colouring is only a baseline

A binary conflict graph discards how much two returns interfere, cannot credit
partial reuse, and usually assumes pairwise conflicts are sufficient.  More
importantly, colouring does not define how shared waveform energy, correlated
receiver noise, or target-estimation uncertainty enter the objective.  It can
therefore be computationally useful without being a physically closed ISAC
model.

SCIR instead treats one feasible Tx--waveform--Rx-set execution as an
information-producing experiment.  Reuse means that this one physically
charged experiment reduces uncertainty for several targets.  It never means
copying one target's deflection or power into another target for free.

## 2. Joint measurement and information accounting

For target states collected in `x=[x_1,...,x_Q]`, one action `a` produces the
locally linearized matched-filter observation

```
z_a = h_a(x) + n_a,       n_a ~ N(0, R_a),
H_a = dh_a/dx.
```

`R_a` includes receiver thermal noise, waveform cross-ambiguity, residual
clutter, quantization and any correlation between target bins.  The exact
conditional information increment is

```
J_a = H_a' R_a^{-1} H_a.
```

Independent receiver evidence may be added.  Correlated evidence must first be
stacked and whitened through its joint `R_a`; it must not be added as if it were
independent.

## 3. Continuous spectral reuse certificate

Partition `R_a = D_a + E_a`, where `D_a` retains the within-target covariance
blocks and `E_a` contains cross-target coupling.  Define

```
rho_a = ||D_a^{-1/2} E_a D_a^{-1/2}||_2.
```

Because `R_a <= (1+rho_a)D_a`, inversion reverses Loewner order:

```
H_a' R_a^{-1} H_a
    >= (1+rho_a)^{-1} H_a' D_a^{-1} H_a
    = J_a_lower.
```

Thus `gamma_a=(1+rho_a)^{-1}` is a continuous reuse discount.  Orthogonal
returns have `rho=0` and receive full credit.  Strongly coupled returns receive
less credit.  The certificate is rejected if the covariance is not positive
definite.  This is implemented and numerically checked in
`uav_isac/coordination/spectral_information_reuse.py`.

## 4. Bayesian dynamic objective

Each local tracker predicts

```
P_q^- = F_q P_q F_q' + Q_q,       Y_q^- = (P_q^-)^{-1}.
```

For a selected action set `S`, use the risk-aware information utility

```
F(S) = sum_q w_q [log det(Y_q^- + sum_{a in S} J_aq_lower)
                  - log det(Y_q^-)].
```

For fixed PSD action increments this utility is normalized, monotone and
submodular: the marginal log-det gain decreases as the accumulated information
matrix grows.  A cardinality-only greedy scheduler therefore has the classical
`1-1/e` guarantee.  Under the actual intersection of role, RF-energy,
communication and receiver-capacity constraints, that guarantee does not carry
over automatically; the implementation must state the approximation theorem
for the exact constraint family it uses.

## 5. Physical and communication constraints

Every candidate action must carry a resource vector and remain feasible:

```
tr(S_ka) + P_comm,k <= P_ISAC,k,
payload_bits / delivered_rate <= deadline,
S_ka >= 0,
one UAV cannot execute incompatible Tx/Rx roles in the same sub-slot,
OTFS CPI duration <= control-frame duration.
```

One waveform action is charged once.  Its information may cover several
targets only through `J_a_lower`.  Communication rate uses the physical U2U
SINR/deadline model; a missing packet contributes no remote information.

## 6. Strictly distributed realization

Each UAV builds candidates from its local posterior and physically delivered
peer state.  It broadcasts only:

- a candidate/action hash;
- its resource vector;
- low-rank factors `U_aq` with `J_aq_lower = U_aq U_aq'`;
- a frame/AoI tag and a spectral-certificate scalar `rho_a`.

No simulator truth, global coefficient tensor, or central fallback is allowed.
Resource prices are updated by projected local primal--dual steps.  Convergence
claims apply first to the convex fractional relaxation; discrete rounding and
finite-round consensus require separate approximation and disagreement bounds.

## 7. Online computation

For low-rank `J=UU'`, marginal utility uses the determinant lemma

```
Delta = log det(I + U' Y^{-1} U),
```

and posterior covariance uses Woodbury

```
P_new = P - P U (I + U' P U)^{-1} U' P.
```

This changes repeated dense inversions into rank-`r` updates.  A lazy upper
bound on `Delta` supports anytime selection: the controller may stop at its
deadline and execute the best already-feasible set.  Cross-frame reuse is
allowed only when a perturbation bound proves that no stale candidate can
overtake the current selection margin.

## 8. Certified AI acceleration

AI is used as a proposal mechanism, not as an authority.  A shared scorer
consumes local, size-invariant candidate features such as certified spectral
coupling, information summaries, energy, payload, AoI and queue deficit.  It
may accelerate four parts of the online pipeline:

- rank candidate actions before expensive physical `H_a/R_a` construction;
- propose a small active set for resource allocation and assignment;
- warm-start primal--dual prices and actor hidden state;
- predict whether the remaining compute budget justifies another exact batch.

Every executable action still passes physical-resource feasibility and an
analytic information certificate.  For `G=U' P U >= 0`, let `t=tr(G)` and let
`r` be the column count of `U`.  Concavity gives the rank-aware upper bound

```
Delta = sum_i log(1 + lambda_i(G))
      <= r log(1 + t/r)
      <= t.
```

The model first proposes `top-k` candidates.  Exact verification then proceeds
in descending certified-upper-bound batches until the largest unverified upper
bound is no greater than the best verified gain.  If a hard evaluation budget
expires before that condition, the controller executes the supplied feasible
incumbent.  Therefore a distribution shift or adversarial model ranking can
increase latency or trigger fallback, but cannot certify an invalid choice.

The synthetic benchmark in `tools/benchmark_certified_ai_screening.py` is a
mechanism and latency test, not yet an end-to-end physical ISAC result.  On the
local RTX 5070, 50 scenes with 512 candidates required a mean 17.12 exact
candidates and 0.334 ms total on CPU; GPU dispatch was slower at this size.
For 20 scenes with 8192 candidates, a mean 66.2 candidates were exactly
checked; GPU inference was 0.465 ms and total certified screening was 1.513 ms.
All tested scenes returned the teacher optimum with a completed certificate.
However, a single fully batched rank-two log-det was still faster than the AI
pipeline.  AI is justified only when pruning avoids full channel/Jacobian,
covariance, feasibility or combinatorial evaluation whose per-pruned-candidate
cost exceeds the measured break-even value.  The physical adapter must measure
that condition rather than assume an AI speedup.

## 9. Implementation gates

1. Unit tests: PSD lower bound, determinant lemma and Woodbury equivalence.
2. Synthetic two-target audit: vary cross-correlation continuously and verify
   that credited information decreases monotonically.
3. Physical adapter: derive `H_a` and `R_a` from the existing bistatic OTFS
   model without changing energy accounting.
4. Single-node oracle comparison: dense joint FIM versus the certified bound.
5. Distributed finite-round candidate exchange and price updates.
6. K/Q/speed/motion A/B with QoS, communication, information-gap and P95
   compute gates.

Formal deployment requires Gates 1--6; passing only the first two establishes
the mathematical primitive, not an end-to-end ISAC result.

## 10. Source basis

- Xiong et al., ISAC Gaussian-channel sensing/communication tradeoff,
  DOI 10.1109/TIT.2023.3284449.
- Liu et al., fundamental limits of ISAC,
  DOI 10.1109/COMST.2022.3149272.
- Liu et al., Fisher-information sensor selection and information ellipsoids,
  DOI 10.1109/TSP.2023.3283047.
- Rusu et al., sensor scheduling under time, energy and communication limits,
  DOI 10.1109/TSP.2017.2773429.
- Grimsman et al., information limits in distributed submodular maximization,
  DOI 10.1109/TCNS.2018.2889005.
