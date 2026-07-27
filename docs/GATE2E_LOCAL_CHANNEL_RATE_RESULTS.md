# Gate 2E: Local-channel-observable rate control

## Question

Was the Gate 2D adaptive-rate failure caused by the categorical rate head
being unable to observe the active channel deadline and SNR condition?

Gate 2E adds one strictly local six-dimensional summary to each UAV
observation:

1. fresh received-sender fraction;
2. mean received-SNR margin;
3. mean received-packet latency slack;
4. mean received-packet age;
5. configured deadline relative to the nominal deadline; and
6. configured receive-SNR threshold relative to the nominal threshold.

The first four entries use only packets received by that UAV. The last two are
local receiver configuration. No fusion-centre state, ground communication or
free acknowledgement is introduced.

Only the old categorical rate head and a zero-initialized linear adapter from
these six features are trainable. Movement, target-token values, Top-1 token
cardinality, ISAC power, evidence codec/fusion and detector threshold remain
frozen.

## Verification

- 67 observation, communication and adaptive-rate regression tests pass.
- The zero-initialized adapter leaves movement, token values and rate logits
  exactly unchanged.
- A one-update smoke test passes the PPO old/new likelihood check with maximum
  absolute error `1.0e-5`.
- The new adapter weight norm changes from zero to `0.0146` after one update
  and `0.0522` after four updates.
- The bounded transport fix remains stable: the four-update critic loss is
  finite (`122.1` on the last update).

## Frozen stress20 result

| Condition | Method | steady | weak3 | worst | CVaR | QoS | Latent delivery | Total bit/frame | Active rate mix |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Nominal | Fixed 4 bit | 0.8877 | 0.8503 | 0.6331 | 0.0413 | 0.50 | 1.000 | 766.1 | 100% 4 |
| Nominal | Gate 2E | 0.8887 | 0.8516 | 0.6375 | 0.0458 | 0.55 | 1.000 | 892.7 | 49.7% 4 / 50.3% 8 |
| 0.8 ms | Fixed 4 bit | 0.8876 | 0.8501 | 0.6348 | 0.0434 | 0.55 | 1.000 | 766.3 | 100% 4 |
| 0.8 ms | Gate 2E | 0.8537 | 0.8049 | 0.5275 | 0.0352 | 0.45 | 0.717 | 900.7 | 43.2% 4 / 56.8% 8 |

Paired Gate-2E-minus-fixed-4 worst differences:

- nominal: `+0.00447`, bootstrap 95% CI
  `[-0.00213, 0.01345]`;
- 0.8 ms: `-0.10725`, bootstrap 95% CI
  `[-0.23785, 0.00239]`.

Gate 2E therefore fails the predeclared direction criteria. Its 0.8 ms
average worst is more than 0.10 below fixed 4 bit and below the required 0.60.
Traffic is also 17.6% above fixed 4 bit at 0.8 ms.

## Diagnosis and decision

Local channel observability was necessary but not sufficient. The learned
deadline feature changes the 4-bit-versus-8-bit logit by less than `0.001` in
the tight-deadline direction, while delayed team-level PPO credit still makes
unsafe 8-bit actions attractive. Under 0.8 ms the policy consequently uses
*more* 8-bit actions than at nominal conditions.

Longer training of this objective is not justified. The next bounded test must
assign each sender its own transport-constraint outcome before adding a more
expensive counterfactual information-value estimator.

## Artifacts

- `results/gate2e_local_channel_rate_only_smoke1/`
- `results/gate2e_local_channel_rate_only_train4/`
- `results/gate2e_local_channel_train4_nominal_stress20/paired_eval.csv`
- `results/gate2e_local_channel_train4_deadline0p8ms_stress20/paired_eval.csv`
