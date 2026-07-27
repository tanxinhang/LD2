# Adaptive Top-k Communication Feasibility

## Objective

Test whether the fixed top-2 target-token contract should be replaced by an
adaptive cardinality policy while preserving the exact 1 W per-UAV ISAC budget.
Checkpoint and scenario pairs are held fixed. Confirmation and stress subsets
are diagnostic sets, not untouched publication test sets.

## Fixed-cardinality controls

### Confirmation-8

| k | steady | weak3 | worst | worst CVaR | feasible | bit/frame |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.919731 | 0.892975 | 0.774492 | 0.305244 | 0.750 | 768 |
| 2 | 0.915477 | 0.887303 | 0.763385 | 0.124035 | 0.750 | 1280 |
| 3 | 0.906161 | 0.874881 | 0.719301 | 0.113248 | 0.625 | 1792 |
| 4 | 0.900493 | 0.867323 | 0.711269 | 0.055306 | 0.625 | 2304 |

The per-episode oracle over k in {1,2,3} reaches steady 0.927767, weak3
0.903690, and worst 0.812545. Its feasible rate remains 0.75 because the two
hardest geometries are not repaired by communication cardinality alone.

### Stress-8

| k | steady | weak3 | worst | worst CVaR | feasible | bit/frame |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.888495 | 0.851327 | 0.687953 | 0.054909 | 0.750 | 768 |
| 2 | 0.866700 | 0.822267 | 0.596384 | 0.062085 | 0.500 | 1280 |

Fixed top-1 is therefore a real candidate rather than a confirmation-set
accident. Some individual scenarios still prefer top-2, which establishes a
non-zero adaptive upper bound.

## PPO-native adaptive action

The existing categorical rate action is reused as a joint cardinality action:

- action 0: silence;
- action 1: 8-bit top-1;
- action 2: 8-bit top-2.

Both active choices use identical quantization. A configurable metadata
denominator preserves the legacy normalized rate field, and the selected k is
included in the exact stored/recomputed PPO log-probability. Zero-training
evaluation reproduces the old top-2 metrics to numerical precision.

Only `comm_rate_head` is trainable in the structural probe. Every movement,
message-content, sensing-power, encoder, and attention parameter is frozen.

## Short training results

| Variant | top-1 share | top-2 share | steady | weak3 | worst | CVaR | feasible | bit/frame |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Old-biased, 4 updates | 0.000 | 1.000 | 0.915477 | 0.887303 | 0.763385 | 0.124035 | 0.750 | 1280 |
| Balanced, 4 updates | 0.011 | 0.989 | 0.915120 | 0.886827 | 0.761956 | 0.118318 | 0.750 | 1274 |
| Worst-credit, 6 updates | 0.557 | 0.443 | 0.912103 | 0.882804 | 0.729773 | 0.224646 | 0.625 | 995 |

Worst-specific delayed credit successfully prevents collapse to top-2 and
improves the tail CVaR, but reduces mean worst and QoS feasibility. One paired
scenario falls from approximately 1.0 under either fixed k to 0.363 under the
mixed distributed policy. The learned k switch rate is 0.0558, so rapid
temporal switching is not the only cause; sender-specific deletion mistakes and
heterogeneous local k decisions are the more important unresolved issue.

## Conclusion

Adaptive top-k is feasible and useful, but the current team-level worst credit
is insufficient for safe decentralized cardinality control. Fixed top-1 is the
best deployable candidate found in this screening. A paper-grade adaptive
method should train a sender-specific counterfactual k critic: for a sampled
sender, replay the same simulator state/RNG with top-1 and top-2 masks, then use
the paired change in worst/weak3 as the rate-head advantage. Temporal persistence
or a switching penalty should be evaluated only after this attribution problem
is solved.

## Artifacts

- `results/adaptive_topk_fixed1_eval8/`
- `results/adaptive_topk_fixed3_eval8/`
- `results/adaptive_topk_fixed1_stress8/`
- `results/adaptive_topk12_gateoff_eval8_v3/`
- `results/adaptive_topk12_rateonly_train4/`
- `results/adaptive_topk12_balanced_train4/`
- `results/adaptive_topk12_tailcredit_train6/`
- `results/adaptive_topk12_tailcredit_train6_diag/`
