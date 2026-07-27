# Communication-value identifiability gate

Date: 2026-07-23

## Structural finding

The current fixed-target environment contains a free centralized sensing-fusion
path:

\[
D_q^{\mathrm{env}}=\sum_{(i,j,q)\in\mathcal E}d_{ijq},
\qquad
P_{D,q}^{\mathrm{env}}=f(D_q^{\mathrm{env}}).
\]

Every selected receiver's deflection is added before the system detection
probability is computed. This happens independently of learned U2U message
content, delivery, rate, and age. Consequently, the environment obtains the
benefit of exchanging receiver sufficient statistics even when no sensing
evidence was carried by a Token.

There is a second potential free-information path in tracking mode:
`p0_uses_belief=true` with `neighbor_belief_fusion=false` currently ranks
geometry using the uniform mean of all UAV beliefs. That is centralized access
to private beliefs rather than a local-only deployment baseline.

## Corrected Gate 0

The proposed comparison "fix movement, power, and P0, then change beliefs" is
identically zero under the current physical detector: once the selected edges
are fixed, belief does not enter \(P_D\).

The meaningful fixed-control evidence oracle is instead:

- Local-only: for each target, use the best single receiver's accumulated
  deflection;
- Centralized evidence: sum every selected receiver's deflection, matching the
  current environment;
- keep selected edges, movement, sensing power, communication power, and random
  scenario identical.

This measures the exact value of exchanging detection sufficient statistics,
without allowing a scheduler or controller to confound the result.

## Stress-20 result

Policy and protocol:

- formal Top-1 actor:
  `results/paper_top1_test100/best_restored.pt`;
- first 20 seeds of the fixed stress bank;
- deterministic execution, 150 frames, 20-frame steady window;
- 2,000 paired episode-bootstrap resamples.

| Metric | Best local RX | Current global fusion | Oracle gain | 95% CI |
|---|---:|---:|---:|---:|
| steady | 0.8483 | 0.8975 | +0.0492 | [0.0324, 0.0669] |
| weak3 | 0.7977 | 0.8633 | +0.0656 | [0.0432, 0.0893] |
| worst | 0.5169 | 0.6717 | **+0.1547** | **[0.0940, 0.2201]** |

The reconstructed global detector matches the environment to a maximum absolute
error of \(3.33\times10^{-16}\).

The pre-registered gate required:

\[
\Delta_{\mathrm{oracle}}^{\mathrm{worst}}\ge 0.03
\]

with paired 95% lower bound greater than 0.01. The result passes by a wide
margin.

Result:
`results/evidence_oracle_free_fusion_stress20/paired_eval.csv`.

## Interpretation

The earlier conclusion "Token content is weak" does not show that sensing
evidence has little value. It shows that the environment already grants the
evidence-fusion result outside the Token path.

Therefore:

1. current U2U mainly learns intent/topology coordination because evidence
   exchange is unnecessary for receiving the global fused reward;
2. zeroing Token values cannot remove the free global-deflection path;
3. no-U2U still loses coordination performance, but it does not lose the
   centralized detector;
4. the formal Top-1 performance is valid for a centralized evidence-fusion
   detector, but is not yet a strictly U2U-only distributed sensing result.

## Required system-definition choice

Before implementing structured evidence Tokens, define where the final
detection statistic exists.

Recommended first-pass U2U-only distributed semantics:

\[
D_{kq}^{+}
=D_{kq}^{\mathrm{local}}
+\sum_{j\in\mathcal N_k}
\mathbf 1[\text{delivered, fresh, target-matched}]
\widehat D_{jq},
\]

\[
k_q^{\mathrm{owner}}
=
\arg\max_k D_{kq}^{\mathrm{predicted}},
\qquad
P_{D,q}^{\mathrm{team}}
=f(D_{k_q^{\mathrm{owner}}q}^{+}).
\]

Here each UAV can fuse only its own receiver evidence and evidence actually
delivered through U2U. The airborne fusion owner is chosen before observing the
random frame-level evidence, using geometry or the already selected feasible
edges. This prevents post-observation owner selection from inflating detection
through a multiple-testing effect. No ground fusion centre is assumed.

Alternatives must be stated explicitly:

- team OR/max detection, with method-specific thresholds calibrated under
  \(H_0\) to the same team-level false-alarm probability;
- consensus requirement across all UAVs;
- a ground fusion centre, which contradicts the current no-ground-
  communication scope unless restored as a modeled link.

## Next gate

Do not retrain MAPPO yet. First generate receiver-local random evidence
consistent with the deflection detector:

\[
z_{jq}\mid H_0\sim\mathcal N(0,1),\qquad
z_{jq}\mid H_1\sim\mathcal N(\sqrt{d_{jq}},1),
\]

\[
\ell_{jq}=\sqrt{d_{jq}}z_{jq}-\frac{d_{jq}}{2}.
\]

Local, distributed, and central variants must reuse the same underlying random
draws. Then implement deterministic structured evidence transport on the
existing selected edges:

1. aggregate selected-edge deflection per receiver and target, then sample one
   receiver-target LLR;
2. target ID, quantized LLR, source, timestamp, and evidence ID;
3. actual U2U delivery, delay, packet bits, and Token mask;
4. evidence de-duplication and owner-side fusion only after successful
   delivery;
5. \(H_0\)-calibrated thresholding at a fixed team-level \(P_{\mathrm{FA}}\).

Then compare, on paired trajectories:

- best local RX;
- lossless all-evidence oracle;
- structured Top-1 evidence Token;
- structured Top-2 evidence Token;
- zero, shuffled, stale, and target-ID-mismatched evidence.

Use:

\[
\eta_{\mathrm{token}}
=
\frac{W_{\mathrm{token}}-W_{\mathrm{local}}}
{W_{\mathrm{central}}-W_{\mathrm{local}}}.
\]

Proceed to learned Token compression only if structured Top-1/Top-2 recovers at
least 50% of the oracle worst gain and significantly exceeds zero/shuffle.

Top-k is not fully defined until its target-selection rule is specified. Report
both a deployable local selector (rank by pre-observation local quality) and an
oracle marginal-value selector. If only the oracle selector passes, the channel
has enough information capacity but a learned evidence-routing policy is still
required.

Dynamic hidden targets and private noisy measurements remain the next extension,
but they should be added after this detector-fusion boundary is corrected. The
current free-fusion issue already supplies a strong and physically interpretable
communication-necessity signal without expanding the paper into full tracking.

## Gate 1 status

Gate 1a--1c have now passed in the fixed-policy offline counterfactual audit.
The final owner-aware 8-bit evidence Token with a 2-bit confidence class,
configured physical transport, and fixed team-level \(P_{FA}\) reaches:

- Top-1: steady 0.8899, weak3 0.8532, worst 0.6417;
- Top-2: steady 0.8976, weak3 0.8635, worst 0.6719;
- 100% delivery and approximately 0.53 ms p95 latency;
- significant paired degradation under zero-content and value-target
  mismatch controls.

Full protocol, caveats, and results:
`docs/GATE1_DISTRIBUTED_EVIDENCE_RESULTS.md`.
