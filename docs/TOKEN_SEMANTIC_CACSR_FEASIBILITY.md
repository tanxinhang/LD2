# Token Semantic / CA-CSR Feasibility Study

Date: 2026-07-22

This is a staged frozen-policy diagnostic.  The selection split supplies the
probe/gate training rows and the confirmation split supplies the reported
diagnostic rows.  Neither split is an untouched final test after this study.

## 1. Does the existing latent stream contain useful information?

Eight selection episodes and eight disjoint confirmation episodes produced
9,600 physically quantized top-2 token rows per split.  All rows use the
environment's actual rate-dependent quantizer.

| Feature view | Current RX evidence AUC | Conditional local P_D R2 | Distance R2 |
|---|---:|---:|---:|
| Protocol metadata | 0.533 | -0.095 | -0.055 |
| Metadata + dimension-0 bid | 0.567 | 0.105 | 0.433 |
| Latent dimensions 1..15 | **0.892** | **0.905** | **0.826** |
| Full token | 0.890 | 0.900 | 0.821 |
| Full model evaluated after semantic zeroing | 0.550 | -0.346 | 0.304 |
| Full model evaluated after target-conditional semantic permutation | 0.514 | -1.081 | -0.534 |

Conclusion: the sender already encodes local evidence and geometry in the
latent dimensions.  A new sender reconstruction objective is not the first
bottleneck; receiver utilization is.

## 2. Can a lightweight deployable decoder recover the semantics?

A 15-32-16-2 MLP (1,074 parameters) was trained offline while every legacy
actor parameter remained frozen.

| Metric | Confirmation result |
|---|---:|
| RX evidence AUC | 0.886 |
| RX evidence balanced accuracy | 0.830 |
| Conditional local P_D MAE | 0.070 |
| Conditional local P_D R2 | 0.940 |

With decoder behavioural authority fixed to zero, the base and decoder
checkpoints were exactly equal on the same eight confirmation episodes.  Every
parseable scalar metric had zero difference, including steady, weak3, worst,
CVaR, QoS feasibility, communication bits, and power balance.

## 3. Can decoded semantics identify crisis frames?

The crisis label is next-frame team target P_D < 0.60.  Thresholds and temporal
parameters were selected only on the selection rows and applied to the
confirmation rows.

| Detector | AUC / binary balanced accuracy | Recall | Activation |
|---|---:|---:|---:|
| Endpoint deficit only | 0.687 | 0.576 | 0.267 |
| Protocol bid crisis | 0.774 | 0.707 | 0.424 |
| Semantic quality crisis | **0.828** | 0.860 | 0.449 |
| Instantaneous combined score | **0.844** | 0.988 | 0.596 |
| EMA + 5-frame stagnation gate | 0.767 | 0.824 | **0.382** |

The semantic score adds 0.070 AUC over the protocol-only score.  Temporal
hysteresis reduces intervention frequency, although the confirmation split has
twice the endpoint-underload rate of the selection split (0.267 versus 0.126),
so its activation rate remains higher than the 0.25 selection cap.

## 4. Does a power-only CA-CSR intervention improve QoS?

The pretrained actor and decoder were frozen.  The gate altered only the
Q-way sensing-power softmax; total per-UAV RF power remained exactly 1 W.

| Gain | steady | weak3 | mean worst | worst CVaR | QoS feasible | Gate rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (base) | **0.91548** | **0.88730** | **0.76339** | 0.12403 | 0.75 | 0 |
| 0.10 | 0.91521 | 0.88695 | 0.76240 | 0.12484 | 0.75 | 0.268 |
| 0.25 | 0.91489 | 0.88652 | 0.76125 | 0.12673 | 0.75 | 0.266 |
| 0.50 | 0.91425 | 0.88566 | 0.75893 | **0.12776** | 0.75 | 0.268 |

The power-only intervention slightly improves the lowest tail but reduces mean
worst and steady as gain increases.  One moderately difficult episode is
over-corrected, whereas the two severe episodes receive only small gains.
Communication remains 1,280 bit/frame and maximum RF balance error remains
2.22e-16 W.

## Decision

- PASS: the existing 15-D latent stream contains useful non-contract state.
- PASS: a 1,074-parameter neural decoder generalizes across episodes.
- PASS: decoded semantics improve crisis prediction over contract-only bids.
- PASS: an inactive decoder is exactly backward compatible and the active
  residual preserves the hard 1 W budget.
- FAIL: sensing-power correction alone does not improve the joint mean/tail QoS
  objective.

The next mechanism should apply a bounded semantic correction to endpoint bids
before the capacity-two projection, so a low-quality endpoint can be replaced
rather than merely assigning more power after matching.  The power residual
should remain a secondary, low-gain branch.  No claim of end-to-end performance
improvement is supported yet.
