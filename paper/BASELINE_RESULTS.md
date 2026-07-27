# Matched Baseline and Ablation Results

## One-sentence conclusion

On the same 100 predefined 4-UAV/4-target test geometries, the full
Distributed Consensus-Bid MAPPO (DCB-MAPPO) configuration outperformed the
matched pre-consensus, forced-silence, capacity-projection, sender-identity and
bid-mixing baselines in strict worst-target detection probability, while the
direct communication-to-sensing residual and the non-reserved token values did
not show an independently resolved effect.

## Protocol

- All learned-policy rows use the same frozen source checkpoint:
  `results/qos_pretrained_soft_gated_move50_test100/best_restored.pt`.
- All rows use the `test` split of `config/stratified_seeds_800_q4.json`, with
  exactly 100 scenarios, deterministic actions and the final 20-frame steady
  window.
- Unless explicitly ablated, the physical channel, 1 W per-UAV RF budget,
  target geometry, sensing scheduler, packet rate and actor are unchanged.
- The reported paired interval is a two-sided 95% non-parametric bootstrap
  confidence interval over 10,000 resamples of the per-scenario difference
  \(P_D^{\mathrm{worst}}(\text{full})-P_D^{\mathrm{worst}}(\text{baseline})\),
  with random seed 20260722.
- The intervals are descriptive and are not adjusted for multiple comparisons;
  the text therefore reports whether an interval includes zero rather than
  treating every row as a separate confirmatory hypothesis test.
- The centralized motion row is an optimistic diagnostic ceiling, not a
  deployable method.

## Primary comparison

| Variant | steady | weak3 | worst | Full − variant worst [95% paired CI] | QoS feasible | bit/frame |
|---|---:|---:|---:|---:|---:|---:|
| Pre-consensus attention/soft commitment | 0.8599 | 0.8133 | 0.5429 | 0.2555 [0.2112, 0.3011] | 0.41 | 1280 |
| No U2U communication (forced silence) | 0.9177 | 0.8903 | 0.7055 | 0.0930 [0.0472, 0.1415] | 0.67 | 0 |
| No capacity-two sensing projection | 0.8554 | 0.8072 | 0.5362 | 0.2622 [0.2131, 0.3116] | 0.37 | 1280 |
| No direct communication-to-sensing residual | 0.9435 | 0.9247 | 0.7939 | 0.0045 [−1.6×10⁻⁵, 0.0112] | 0.78 | 1280 |
| Zero all transmitted token values; retain top-2 mask | 0.9332 | 0.9109 | 0.7675 | 0.0309 [−0.0070, 0.0695] | 0.73 | 1280 |
| Permute token values and masks across sender identities | 0.9137 | 0.8849 | 0.7041 | 0.0943 [0.0530, 0.1364] | 0.68 | 1280 |
| Projected-only bid, \(\beta=0\) | 0.9188 | 0.8918 | 0.7011 | 0.0973 [0.0378, 0.1601] | 0.67 | 1280 |
| Intrinsic-only bid, \(\beta=1\) | 0.9046 | 0.8728 | 0.6844 | 0.1140 [0.0700, 0.1568] | 0.55 | 1280 |
| **Full DCB-MAPPO, \(\beta=0.5\)** | **0.9452** | **0.9269** | **0.7984** | — | **0.78** | 1280 |
| Centralized one-to-one motion ceiling | 0.9870 | 0.9827 | 0.9499 | −0.1515 [−0.2152, −0.0899] | 0.92 | 1280 |

The full model exceeded every deployable structural baseline in mean
worst-target quality. The paired intervals exclude zero for pre-consensus,
silence, no capacity-two projection, sender permutation and both bid endpoints.
They include zero for removal of the direct communication-to-sensing residual
and for zero token values; these two components therefore do not have an
independently established effect under the current frozen-policy protocol.

## Robustness and geometry

| Variant | worst LCB | worst CVaR | feasible Wilson LCB | movement collision | mean nearest distance (m) |
|---|---:|---:|---:|---:|---:|
| Pre-consensus | 0.4911 | 0.0850 | 0.3325 | 0.9243 | 63.79 |
| No U2U communication | 0.6443 | 0.0881 | 0.5891 | 0.9255 | 61.90 |
| No capacity-two projection | 0.4838 | 0.0788 | 0.2950 | 0.9567 | 68.97 |
| No direct communication-to-sensing residual | 0.7464 | 0.2844 | 0.7050 | 0.8646 | 57.44 |
| Zero token values | 0.7187 | 0.2567 | 0.6516 | 0.8867 | 63.24 |
| Permuted sender semantics | 0.6512 | 0.1607 | 0.5994 | 0.8408 | 76.14 |
| Projected-only bid | 0.6374 | 0.0540 | 0.5891 | 0.8703 | 60.62 |
| Intrinsic-only bid | 0.6348 | 0.2384 | 0.4679 | 0.8627 | 52.41 |
| **Full DCB-MAPPO** | **0.7497** | **0.2882** | **0.7050** | 0.8680 | 57.46 |
| Centralized movement ceiling | 0.9264 | 0.7497 | 0.8635 | 0.0000 | 11.02 |

## Evidence interpretation

1. **U2U communication is useful despite its power cost.** Forced silence gives
   all 4 W of team RF power to sensing, yet the full method improves worst by
   0.0930 with a paired interval of [0.0472, 0.1415]. The benefit therefore
   cannot be explained by additional sensing power.
2. **Capacity-two projection is indispensable in this implementation.** Its
   removal causes the largest deployable ablation loss: worst decreases from
   0.7984 to 0.5362 and the collision rate rises to 0.9567.
3. **Sender-target correspondence carries the strongest token-level evidence.**
   Permuting both values and masks across senders decreases worst by 0.0943 and
   increases mean nearest-target distance by 18.68 m.
4. **The direct sensing residual is not a supported contribution.** Removing it
   changes worst by only 0.0045, and the paired interval includes zero. It
   should not be presented as a main innovation.
5. **Continuous token values have at most a secondary contribution.** Zeroing
   values while preserving topology lowers the point estimate by 0.0309, but
   the paired interval includes zero. The safe claim is that sparse topology
   plus sender identity is useful; additional value semantics remain
   inconclusive.
6. **There remains a substantial motion-coordination gap.** The centralized
   motion ceiling improves worst by 0.1515 and feasible rate by 14 percentage
   points over the full distributed method.

## Artifacts

- Full method: `results/distributed_consensus_hybrid50_bid_frozen_test100/`
- Pre-consensus: `results/qos_pretrained_soft_gated_move50_test100/`
- Forced silence: `results/baseline_silence_test100/`
- No capacity-two projection: `results/baseline_no_capacity_test100/`
- No direct communication-to-sensing residual:
  `results/baseline_no_comm_sensing_test100/`
- Zero token values: `results/baseline_zero_payload_test100/`
- Permuted sender semantics: `results/baseline_permute_identity_test100/`
- Centralized motion ceiling: `results/baseline_central_movement_oracle_test100/`
- Reproducible paired summary: `tools/summarize_paper_baselines.py`

## Remaining baseline gap

These matched frozen-policy interventions isolate mechanisms but do not replace
independently trained algorithm baselines. Before submission, the final test
must still include multi-seed training for standard MAPPO without learned U2U
messages, attention-only communication MAPPO, sparse-token MAPPO without
consensus, and the proposed final architecture.
