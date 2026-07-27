# IPPO fine-tuning seed stability

Each critic-aligned IPPO run uses 40 PPO updates (81,920 frames) and is compared with the frozen commitment policy carrying the same commitment-head seed. All rows use the same 100 fixed test scenarios.

| Seed | IPPO steady | IPPO weak3 | IPPO worst | worst LCB | CVaR20 | QoS | frozen worst | IPPO-frozen worst (95% paired CI) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.9614 | 0.9485 | 0.8511 | 0.8050 | 0.3533 | 0.86 | 0.8129 | +0.0382 [-0.0016, +0.0805] |
| 123 | 0.9443 | 0.9258 | 0.8056 | 0.7528 | 0.2232 | 0.80 | 0.8216 | -0.0161 [-0.0748, +0.0420] |
| 456 | 0.9460 | 0.9280 | 0.7992 | 0.7434 | 0.2037 | 0.78 | 0.8211 | -0.0219 [-0.0560, +0.0113] |

Across the three seeds:

- IPPO worst: 0.8186 +/- 0.0283
- frozen worst: 0.8185 +/- 0.0049
- seed-level IPPO-frozen worst delta: +0.0001 +/- 0.0332
- IPPO CVaR20: 0.2601 +/- 0.0813
- frozen CVaR20: 0.2713 +/- 0.0165

All IPPO seeds satisfy the Medium average thresholds, but the seed42 gain does not replicate: seeds 123 and 456 both reduce worst and tail performance relative to their frozen counterparts. IPPO fine-tuning therefore adds variance without a reproducible mean gain under the current protocol. The frozen policy remains the defensible primary method until a more stable fine-tuning rule is demonstrated.

These are fine-tuning/head seeds with a shared foundation actor, not full end-to-end random-initialization seeds.
