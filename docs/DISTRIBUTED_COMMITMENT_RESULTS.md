# Distributed Commitment and Token-Necessity Probe

## Objective

The historical tracking-free U2U actor achieved reasonable mean detection, but
P0 was free to choose every transmitter-receiver-target triple. This experiment
tests whether the deployed UAVs themselves can choose targets, whether latent
tokens affect those choices, and whether CCP improves the worst-target QoS.

Deployment targets remain:

- steady P_D >= 0.80
- weak3 P_D >= 0.70
- mean worst P_D >= 0.60

All reported deployment results below use the same independent 20-seed bank
(30001--30020), 150 frames per seed, unless stated otherwise.

## Structural changes

1. `filter_deflection_by_local_commitments` constrains P0 to targets in the
   locally emitted sensing-power top-k. With receiver agreement enabled, both
   bistatic endpoints must independently include the target. There is no oracle
   fallback and the constrained graph is resolved every frame.
2. Three switches isolate transmitter commitment, endpoint rendezvous, and
   fully learned roles.
3. The agreement-to-resource adapter converts the decentralized target
   responsibility distribution into centered log-odds and adds it to the
   physical sensing-allocation logits. Thus token-conditioned agreement has
   execution authority instead of remaining an auxiliary representation.
4. Target-token Sinkhorn matching was corrected to use one row per sender. The
   old implementation incorrectly treated Q tokens from one sender as Q UAVs.

## Structural diagnosis before adaptation

| Mode | steady | weak3 | worst | valid pair |
|---|---:|---:|---:|---:|
| Historical unconstrained P0 + CCP checkpoint | 0.717878 | 0.623838 | 0.210313 | 1.000 |
| TX top-1 commitment, P0 roles | 0.152642 | 0.001000 | 0.001000 | 1.000 |
| TX/RX top-2 rendezvous, P0 roles | 0.355044 | 0.163143 | 0.001000 | 1.000 |
| Top-2 rendezvous + learned roles | 0.001000 | 0.001000 | 0.001000 | 0.010 |

The old policy did not learn decentralized target coverage. Its apparently high
performance depended on P0 repairing target and role decisions. Enabling the
previously untrained role head caused `no_TX_rate=0.988`, confirming that roles
must be introduced only after a separate coordination stage.

## Agreement-to-resource result

The soft local agreement adapter was trained for one rollout, then continued
for one matched rollout without CCP. The second checkpoint is selected because
checkpoint ordering is `(QoS feasible, worst, weak3, steady, -bits)`.

| Checkpoint | steady | weak3 | worst | full | collision |
|---|---:|---:|---:|---:|---:|
| Adapter, first rollout | 0.496124 | 0.328165 | 0.001000 | 0.433508 | 0.931333 |
| Adapter, second rollout (selected) | 0.506158 | 0.341544 | 0.006368 | 0.438856 | 0.921333 |
| Same selected actor, zero token content | 0.489830 | 0.319773 | 0.003725 | 0.427187 | 0.942000 |
| Learned minus zero | +0.016328 | +0.021771 | +0.002643 | +0.011669 | -0.020667 |

Rates, bits, transmission power, delivery and delay are identical in the
learned/zero comparison (`2304 bit/frame`, delivery 1.0). The nonzero gap is
therefore attributable to token content, not extra communication expenditure.
This is the first tested architecture in which the learned payload measurably
helps sensing and collision avoidance.

## CCP control

Starting from the same first-rollout adapter checkpoint, one additional rollout
was run with and without CCP.

| Continuation | steady | weak3 | worst | collision |
|---|---:|---:|---:|---:|
| No CCP control | 0.506158 | 0.341544 | 0.006368 | 0.921333 |
| CCP | 0.518046 | 0.357395 | 0.002047 | 0.948333 |
| CCP minus control | +0.011888 | +0.015851 | -0.004321 | +0.027000 |

CCP validation correlation was 0.0489, reliability 0.00445, and only 0.427% of
message credits were modified. CCP improves averages but worsens the primary
worst-QoS and collision criteria, so it is rejected for checkpoint selection.

## Rejected symmetry-breaking probes

- Straight-through hard commitment reduced allocation collision from 0.711 to
  0.673 in the training rollout, but independent steady/worst fell to
  0.458425/0.001365.
- Corrected local Sinkhorn matching adds new peer-bid projections. Coupling
  these untrained projections immediately to physical resources caused a sharp
  transient collapse (steady 0.061 in validation; independent performance also
  far below the control). It requires a resource-decoupled pretraining phase and
  gradual coupling, not a one-shot switch.

## Remaining bottleneck

The selected actor still fails all deployment thresholds. Mean sensing power is
already almost uniform over targets, so total power is not the immediate cause.
The dominant bottlenecks are geometry and intermittent coverage:

- per-target P_D: `[0.7348, 0.5191, 0.5205, 0.2503]`
- nearest-UAV distance (m): `[60.9, 126.4, 79.6, 77.6]`
- movement collision rate: `0.9213`
- training weak-uncovered target count: `1.2119` per frame
- target responsibility entropy remains near `ln(4)` and assignment collision
  remains about `0.71`

The next justified stage is curriculum training: pretrain token-conditioned
matching with the physical adapter at zero strength, then anneal adapter strength
and commitment top-k, and only afterward introduce decentralized TX/RX roles.
This preserves learned token semantics and avoids a central target teacher while
preventing random new modules from directly controlling physical power.

## Capacity-matched sparse target claims

The sender now transmits only its locally selected target-token blocks. Missing
blocks are removed from the physical packet, so the same decision reduces bits,
serialization delay, and radio energy. The receiver treats delivered token masks
as peer target proposals: zero peer claims receives a vacancy bonus, one claim
permits a second bistatic endpoint, and two claims suppress further competition.
Training additionally penalizes target endpoint loads below or above two.

`top-3` is structurally over-subscribed for this scenario: four UAVs generate 12
claims for only eight required endpoints. The capacity-matched setting is
therefore `share_topk=commit_topk=2`.

| Paired evaluation | steady | weak3 | worst | collision | bit/frame |
|---|---:|---:|---:|---:|---:|
| Previous dense-token control | 0.506158 | 0.341544 | 0.006368 | 0.921333 | 2304 |
| Sparse top-3, no adaptation | 0.366579 | 0.269079 | 0.057858 | 0.931000 | 1792 |
| Sparse top-2, no adaptation (selected) | 0.406442 | 0.333465 | 0.147315 | 0.936333 | 1280 |
| Sparse top-2, zero token values | 0.401758 | 0.327219 | 0.151568 | 0.932000 | 1280 |
| Sparse top-2, complete silence | 0.512631 | 0.350175 | 0.001000 | 0.921667 | 0 |
| Sparse top-2, one-rollout adaptation (rejected) | 0.393775 | 0.324900 | 0.101867 | 0.958000 | 1280 |

The selected sparse rule cuts payload by 44.4% and materially protects the
worst target, but it does not meet deployment thresholds. Complete silence
recovers the mean while losing the worst target. Under the historical 0.2751 W
per-UAV budget, the warm-started resource head assigned about 0.507 W of the
1.1 W system budget to communication throughout the frame, even though mean
packet latency was only about 1.18 ms.

## One-watt per-UAV adaptive ISAC budget

The hard budget is now 1 W independently for every UAV:
`P_comm[k] + sum_q P_sense[k,q] = 1 W`. The policy action spans the full
communication fraction `[0,1]`; silence maps to zero communication power and
all remaining power is distributed across sensing targets by the learned
softmax action.

| One-watt experiment | steady | weak3 | worst | system comm W | system sensing W |
|---|---:|---:|---:|---:|---:|
| Same checkpoint, no adaptation (selected) | 0.425758 | 0.367543 | 0.223214 | 2.000810 | 1.999190 |
| One-rollout adaptation (rejected) | 0.427449 | 0.378124 | 0.197499 | 1.998054 | 2.001946 |

The maximum per-UAV conservation error is below `2.3e-16 W`. The larger budget
improves all three QoS metrics, especially worst-QoS, but one rollout does not
move the old approximately 50:50 power split and reduces paired worst-QoS.
Accordingly the adapted checkpoint is rejected by the worst/QoS checkpoint
rule; longer resource-specific adaptation is still required.

## Explicit communication-aided sensing path

The actor now separates three learnable functions while keeping decentralized
execution:

1. target-token content, rate, sparse topology and communication power;
2. base target-sensing power allocation under PPO;
3. a received-token cross-attention residual applied directly to the executed
   per-target sensing logits.

The third path is zero under a no-token virtual intervention. Its auxiliary
loss therefore supervises only the message-induced residual toward the locally
weakest target; it cannot overwrite the base sensing policy. The differentiable
training inbox now also preserves the physical top-2 token mask.

| 20-seed paired mode | steady | weak3 | worst | collision | bit/frame |
|---|---:|---:|---:|---:|---:|
| One-watt actor before joint adaptation | 0.425758 | 0.367543 | 0.223214 | 0.948333 | 1280 |
| Explicit comm-aided sensing | 0.468692 | 0.399814 | 0.211181 | 0.930333 | 1280 |
| Same actor, zero token values | 0.447984 | 0.380529 | 0.147327 | 0.919000 | 1280 |
| Same actor, complete silence | 0.570201 | 0.426934 | 0.034256 | 0.942000 | 0 |

Learned token values add `+0.020707` steady, `+0.019285` weak3 and
`+0.063854` worst over the zero-value control at identical communication cost.
Compared with silence, communication raises worst by `+0.176925`, demonstrating
communication-assisted sensing rather than mere independent communication and
sensing heads. However, the paired worst remains slightly below the previous
0.223214 checkpoint, so the new checkpoint remains an experimental group and
does not replace the deployable candidate under worst/QoS ordering.
