# Three-seed algorithm stability

Frozen, critic-aligned MAPPO, and critic-aligned IPPO use matching commitment-head seeds and the same ordered 100-scenario test bank. Each PPO method receives 40 updates (81,920 frames) per seed.

| Seed | Frozen worst | MAPPO worst | IPPO worst | MAPPO-frozen | IPPO-frozen | IPPO-MAPPO |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.8129 | 0.8057 | 0.8511 | -0.0072 | +0.0382 | +0.0454 |
| 123 | 0.8216 | 0.8083 | 0.8056 | -0.0133 | -0.0161 | -0.0027 |
| 456 | 0.8211 | 0.8452 | 0.7992 | +0.0241 | -0.0219 | -0.0460 |

## Across-seed results

| Method | steady | weak3 | worst | CVaR20 | QoS feasible |
|---|---:|---:|---:|---:|---:|
| Frozen | 0.9504 +/- 0.0003 | 0.9338 +/- 0.0004 | 0.8185 +/- 0.0049 | 0.2713 +/- 0.0165 | 0.8367 +/- 0.0058 |
| MAPPO | 0.9507 +/- 0.0062 | 0.9343 +/- 0.0082 | 0.8198 +/- 0.0221 | 0.2598 +/- 0.0387 | 0.8133 +/- 0.0231 |
| IPPO | 0.9506 +/- 0.0094 | 0.9341 +/- 0.0125 | 0.8186 +/- 0.0283 | 0.2601 +/- 0.0813 | 0.8133 +/- 0.0416 |

Seed-level mean worst differences (mean +/- sample standard deviation):

- MAPPO - frozen: +0.0012 +/- 0.0201
- IPPO - frozen: +0.0001 +/- 0.0332
- IPPO - MAPPO: -0.0011 +/- 0.0457

All nine method-seed rows satisfy the requested Medium average thresholds. However, neither PPO variant produces a reproducible gain over the frozen policy: mean worst is almost unchanged, while both PPO variants have substantially larger seed variance and lower mean CVaR/QoS feasibility. The seed42 IPPO-over-MAPPO advantage reverses at seed456, so it must not be presented as an algorithmic superiority result.

The frozen commitment policy is therefore the defensible primary method under the current evidence. PPO fine-tuning belongs in the baseline/sensitivity section, not in the claimed performance path.

These replications vary fine-tuning and commitment-head seeds while sharing the foundation actor; they are not full end-to-end random initializations.
