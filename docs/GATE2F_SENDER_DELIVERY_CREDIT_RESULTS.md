# Gate 2F: Sender-specific delivery credit

## Question

Given the local channel observation introduced in Gate 2E, is the remaining
failure caused only by using a team-average delivery signal instead of an
attributable sender-specific transport outcome?

Gate 2F records attempted, delivered and expired links for every active
sender. During centralized training, sender \(i\)'s rate reward receives

\[
  -\lambda_{\mathrm{del}}\,
  \mathbf{1}[r_i>0]\,(1-d_i),
\]

where \(d_i\) is the fraction of that sender's broadcast links delivered in
the current communication decision and
\(\lambda_{\mathrm{del}}=0.05\). Silence is neutral. This signal is
training-only; deployment still uses only the local Gate 2E observation and
does not receive a free acknowledgement.

Movement, message content, target cardinality, power allocation, evidence
transport/fusion and detector settings remain frozen.

## Verification

- 83 relevant regression tests pass.
- Per-sender delivery accounting identifies the failed sender and leaves
  silent senders neutral.
- The one-update mean sender penalty is `0.00333`, consistent with a 6.8%
  rollout link-failure rate.
- PPO likelihood consistency remains below `1e-4`.
- The four-update run is numerically stable; the final critic loss is `121.9`.

## Frozen stress20 result

| Condition | Method | steady | weak3 | worst | CVaR | QoS | Latent delivery | Total bit/frame | Active rate mix |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Nominal | Fixed 4 bit | 0.8877 | 0.8503 | 0.6331 | 0.0413 | 0.50 | 1.000 | 766.1 | 100% 4 |
| Nominal | Gate 2F | 0.8887 | 0.8516 | 0.6375 | 0.0458 | 0.55 | 1.000 | 892.8 | 49.7% 4 / 50.3% 8 |
| 0.8 ms | Fixed 4 bit | 0.8876 | 0.8501 | 0.6348 | 0.0434 | 0.55 | 1.000 | 766.3 | 100% 4 |
| 0.8 ms | Gate 2F | 0.8526 | 0.8034 | 0.5284 | 0.0383 | 0.45 | 0.708 | 899.8 | 43.3% 4 / 56.7% 8 |

At 0.8 ms:

- Gate-2F minus Gate-2E worst is `+0.00087`, bootstrap 95% CI
  `[-0.00138, 0.00355]`;
- Gate-2F minus fixed-4 worst is `-0.10638`, bootstrap 95% CI
  `[-0.23660, 0.00318]`.

The result is statistically and operationally indistinguishable from Gate 2E
and fails the average-worst, delivery and traffic direction criteria.

## Interpretation

The current precision action is weakly identified:

- at nominal conditions, 8 bit has no significant sensing advantage over
  4 bit;
- under 0.8 ms, 8 bit is physically less deliverable and substantially worse;
  and
- 4 bit uses less traffic in both cases.

Consequently, the rate-control problem in the present environment is close to
a dominated-action problem rather than a meaningful adaptive information-value
trade-off. A more elaborate PPO reward, delivery predictor or causal-value
head would be expected to converge to fixed 4 bit, not establish a distinct
algorithmic contribution.

The defensible deployment point remains fixed 4 bit. Before developing a
counterfactual information-value controller, the environment must demonstrate
at least one preregistered condition where additional token precision yields a
reproducible task benefit that offsets its cost. Otherwise adaptive precision
should be pruned from the paper rather than optimized further.

## Artifacts

- `results/gate2f_sender_delivery_rate_only_smoke1/`
- `results/gate2f_sender_delivery_rate_only_train4/`
- `results/gate2f_sender_delivery_train4_nominal_stress20/paired_eval.csv`
- `results/gate2f_sender_delivery_train4_deadline0p8ms_stress20/paired_eval.csv`
