# Supplementary experiment status

## Completed: three training-seed replications

The trainable commitment head was independently optimized with seeds 42, 123,
and 456. All runs used the same shared foundation actor, 30 stratified training
scenarios, 15 commitments per scenario, 5-frame holds, 40 epochs, and learning
rate 1e-3. Each resulting model was evaluated on the same 100 fixed test
scenarios.

| Seed | steady | weak3 | worst | worst LCB | CVaR20 | QoS feasible |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.9500 | 0.9333 | 0.8129 | 0.7603 | 0.2522 | 0.83 |
| 123 | 0.9506 | 0.9341 | 0.8216 | 0.7709 | 0.2809 | 0.84 |
| 456 | 0.9505 | 0.9339 | 0.8211 | 0.7707 | 0.2809 | 0.84 |
| Mean +/- SD | 0.9504 +/- 0.0003 | 0.9338 +/- 0.0004 | 0.8185 +/- 0.0049 | 0.7673 +/- 0.0061 | 0.2713 +/- 0.0165 | 0.8367 +/- 0.0058 |

The original model's worst value (0.8186) is centred in the replication
distribution. This experiment measures the stability of the newly trained
commitment head, not end-to-end random initialization of the shared foundation
actor.

## Retained sensitivity result

An initially attempted 20-epoch/default-lower-LR protocol produced a stable but
lower mean worst value of 0.7558. Audit of the historical training curve showed
that the original model used 40 epochs and learning rate 1e-3. Re-running seed
20260722 with these settings reproduced the historical per-epoch NLL exactly.
The shorter runs are retained as training-budget sensitivity, not mixed into
the formal seed aggregate.

## Completed: critic-aligned algorithm baselines, three seeds

An audit found that the first 40-update MAPPO/IPPO baseline attempt assembled
critic inputs in a different field order during PPO update than during rollout;
legacy IPPO also replaced the communication summary with zeros. Those outputs
are retained only as invalid diagnostics and are excluded from the paper.
Regression tests now enforce exact rollout/update critic alignment.

The corrected MAPPO and IPPO runs use matching pretrained commitment actors for
seeds 42, 123, and 456, the same Top-1 configuration, 40 PPO updates (81,920
frames) per trained method/seed, the same selection and confirmation protocol,
and 100 fixed test scenarios.

| Method | steady | weak3 | worst | CVaR20 | QoS feasible |
|---|---:|---:|---:|---:|---:|
| Frozen commitment | 0.9504 +/- 0.0003 | 0.9338 +/- 0.0004 | 0.8185 +/- 0.0049 | 0.2713 +/- 0.0165 | 0.8367 +/- 0.0058 |
| MAPPO, aligned critic | 0.9507 +/- 0.0062 | 0.9343 +/- 0.0082 | 0.8198 +/- 0.0221 | 0.2598 +/- 0.0387 | 0.8133 +/- 0.0231 |
| IPPO, aligned critic | 0.9506 +/- 0.0094 | 0.9341 +/- 0.0125 | 0.8186 +/- 0.0283 | 0.2601 +/- 0.0813 | 0.8133 +/- 0.0416 |

The seed42 IPPO-minus-MAPPO worst advantage (+0.0454) does not replicate:
seed123 gives -0.0027 and seed456 gives -0.0460. Across seeds, MAPPO-minus-
frozen is +0.0012 +/- 0.0201 and IPPO-minus-frozen is +0.0001 +/- 0.0332.
All method-seed rows meet the Medium average thresholds, but neither PPO
variant improves the frozen policy reproducibly. Both increase seed variance
and reduce mean tail/QoS performance. The frozen commitment policy is the
primary method; PPO fine-tuning is retained as a baseline/sensitivity result.

## Pending

1. Fixed-broadcast/no-communication training baselines under the same budget,
   only if a trained-algorithm comparison is required by the target journal.
2. Controlled-density 6-UAV/6-target and 8-UAV/8-target scale experiments.
3. Full foundation-actor random-initialization seeds if claiming end-to-end
   training stability rather than commitment-head/fine-tuning stability.
