# Token Semantic Capacity Feasibility Study

## Question

Can the pretrained latent-token decoder improve bistatic endpoint selection by
refining neighbor bids before the capacity-two projection, without retraining
the frozen MAPPO policy?

All probes preserve the 1 W per-UAV ISAC budget.  The new mechanisms are
disabled by default and do not directly alter movement or the communication /
sensing power split.

## Mechanisms tested

1. **Row-centred semantic bid correction**: decoded sender evidence multiplied
   by decoded conditional local P_D refines each peer's sparse target bids.
   Row centring preserves aggregate sender authority.
2. **Full-token diagnostic**: all four target tokens are exposed to separate a
   sparse-topology bottleneck from a semantic-decoding bottleneck.
3. **Capacity-aware extra token**: preserve top-2 and add at most one target
   token when peer endpoints are under capacity and local geometry is useful.

## Confirmation-8 screening

| Variant | steady | weak3 | worst | worst CVaR | feasible | bit/frame | collision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen baseline | 0.915477 | 0.887303 | 0.763385 | 0.124035 | 0.750 | 1280.0 | 0.8675 |
| Semantic bid, gain 0.25 | 0.915497 | 0.887329 | 0.763383 | 0.124025 | 0.750 | 1280.0 | 0.8683 |
| Semantic bid, gain 1.00 | 0.915589 | 0.887453 | 0.763387 | 0.124052 | 0.750 | 1280.0 | 0.8650 |
| Full four tokens | 0.900493 | 0.867323 | 0.711269 | 0.055306 | 0.625 | 2304.0 | 0.8700 |
| Full tokens + semantic bid | 0.900861 | 0.867815 | 0.712538 | 0.057453 | 0.625 | 2304.0 | 0.8758 |
| Top-2 + semantic extra token | 0.919029 | 0.892039 | 0.758359 | 0.103720 | 0.750 | 1301.7 | 0.8400 |
| Extra token + semantic bid | 0.919029 | 0.892038 | 0.758358 | 0.103725 | 0.750 | 1301.7 | 0.8400 |

The extra-token branch improves steady, weak3, communication efficiency versus
full broadcast, and the movement collision diagnostic.  It nevertheless
reduces both worst and tail CVaR, so it fails the worst/QoS checkpoint rule.

## Separate stress-8 diagnostic

| Variant | steady | weak3 | worst | worst CVaR | feasible | worst LCB |
|---|---:|---:|---:|---:|---:|---:|
| Frozen baseline | 0.866700 | 0.822267 | 0.59638368 | 0.0620849 | 0.500 | 0.388451 |
| Semantic bid, gain 1.00 | 0.866681 | 0.822242 | 0.59638335 | 0.0620843 | 0.500 | 0.388446 |

This split was used only as a separate diagnostic after confirmation screening;
it is not claimed as an untouched publication test set.  The near-identical
results reproduce the lack of structural effect.

## Interpretation

The latent token is decodable, but decodability is not the same as causal
control.  With top-2 transmission and row capacity two, almost every transmitted
edge is already forced into the feasible solution.  There is little remaining
freedom for a within-row quality correction.  Sending all candidate edges
removes that saturation but breaks the learned sparse coordination topology.
Adding only one crisis edge produces a real mean-performance effect, yet the
untrained gate selects harmful endpoints in tail cases.

Therefore the complete dual-stream idea is **representationally feasible but
not deployable as a frozen-policy adapter**.  The next justified experiment is
joint training of (a) the endpoint gate and semantic bid, (b) an explicit tail
or CVaR objective, and (c) separate movement-commitment and sensing-endpoint
contracts.  Further frozen gain or threshold sweeps are not justified by these
results.

## Artifacts

- `results/token_semantic_bid25_eval8/`
- `results/token_semantic_bid100_eval8/`
- `results/token_fulltoken_base_eval8/`
- `results/token_fulltoken_semantic_bid100_eval8/`
- `results/token_semantic_extra10_eval8/`
- `results/token_semantic_extra10_bid100_eval8/`
- `results/token_semantic_stress8_base/`
- `results/token_semantic_stress8_bid100/`
