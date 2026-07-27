# Critic-aligned algorithm baseline results

The formal comparison uses commitment-head seed 42, a shared foundation actor, identical Top-1 configuration, 40 PPO updates (81,920 frames) for each trained method, and the same ordered 100 test scenarios. The frozen row receives zero PPO updates.

| Method | updates | steady | weak3 | worst | worst LCB | CVaR20 | QoS feasible | bit/frame |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen commitment (0 PPO) | 0 | 0.9500 | 0.9333 | 0.8129 | 0.7603 | 0.2522 | 0.83 | 768 |
| MAPPO fine-tune (critic aligned) | 40 | 0.9474 | 0.9298 | 0.8057 | 0.7530 | 0.2376 | 0.80 | 768 |
| IPPO fine-tune (critic aligned) | 40 | 0.9614 | 0.9485 | 0.8511 | 0.8050 | 0.3533 | 0.86 | 768 |

All three rows meet the requested Medium average thresholds (steady >= 0.80, weak3 >= 0.70, mean worst >= 0.60). That threshold result is distinct from proving that one learning algorithm is superior.

## Matched test-bank differences

| Left minus right | worst delta (95% paired bootstrap CI) | steady delta | weak3 delta | QoS delta | exact McNemar p |
|---|---:|---:|---:|---:|---:|
| MAPPO minus frozen | -0.0072 [-0.0378, +0.0248] | -0.0026 | -0.0035 | -0.03 | 0.5078 |
| IPPO minus frozen | +0.0382 [-0.0020, +0.0802] | +0.0114 | +0.0152 | +0.03 | 0.5078 |
| IPPO minus MAPPO | +0.0454 [+0.0129, +0.0795] | +0.0140 | +0.0187 | +0.06 | 0.0703 |

On this one training seed, aligned IPPO improves mean worst over the frozen policy by +0.0382 and over aligned MAPPO by +0.0454. Aligned MAPPO is effectively tied with the frozen policy: its difference is -0.0072, with a paired interval that must be consulted above. The result does not support a claim that centralized-critic MAPPO is currently better than IPPO at this budget.

This seed42 observation is retained for auditability, but subsequent seed123/456 replications show that it is not stable. Across three seeds, MAPPO, IPPO, and frozen mean worst are nearly identical, while both PPO variants have larger seed variance and lower mean CVaR/QoS. See `docs/ALGORITHM_SEED_STABILITY.md` for the formal multi-seed conclusion.

## Excluded legacy runs

Earlier 40-update MAPPO/IPPO outputs are diagnostic only and are excluded from every formal table. During PPO update, critic fields were assembled in a different order from rollout; legacy IPPO also replaced the communication summary with zeros. Regression tests now enforce rollout/update alignment for both critic variants.
