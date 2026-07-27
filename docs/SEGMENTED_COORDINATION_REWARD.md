# Segmented coordination reward

The coordination reward is disabled by default. The staged experiment family
sets the old combined QoS reward scale to zero while retaining QoS measurement,
communication encouragement, rate exploration, silence penalties and physical
communication costs. This prevents the old steady/weak3/worst scalar from
masking the term under test. All variants start from the same target-token
U2U-ISAC architecture and are cumulative:

| Stage | Newly enabled term | Purpose |
|---|---|---|
| 0 | none (paired control) | measure PPO drift without coordination shaping |
| 1 | signed worst-target deficit progress | improve the deployment bottleneck first |
| 2 | avoidable duplicate-motion penalty | discourage duplicate pursuit only while a below-floor target is uncovered |
| 3 | signed bottom-3 deficit progress | improve weak-target robustness after worst-target credit exists |
| 4 | signed steady-mean deficit progress | recover average quality last |

The progress terms use the reduction in QoS deficit rather than the absolute
QoS value. Holding the same quality therefore cannot repeatedly collect the
shaping reward. A per-target EMA (`coord_reward_ema_alpha`) suppresses
frame-level channel fading before the progress is calculated.

Run the four paired variants from the same checkpoint and seed:

```powershell
python scripts/run_mappo.py --config config/exp_800_q4_u2u_coord_s0_baseline.yaml --warm-start <actor.pt> --seed 42
python scripts/run_mappo.py --config config/exp_800_q4_u2u_coord_s1_worst.yaml --warm-start <actor.pt> --seed 42
python scripts/run_mappo.py --config config/exp_800_q4_u2u_coord_s2_duplicate.yaml --warm-start <actor.pt> --seed 42
python scripts/run_mappo.py --config config/exp_800_q4_u2u_coord_s3_weak3.yaml --warm-start <actor.pt> --seed 42
python scripts/run_mappo.py --config config/exp_800_q4_u2u_coord_s4_steady.yaml --warm-start <actor.pt> --seed 42
```

`train_metrics.csv` contains every raw and weighted term under the
`reward_component_*` prefix. In particular, inspect:

- `reward_component_coord_total`;
- `reward_component_coord_worst_reward`;
- `reward_component_coord_duplicate_penalty`;
- `reward_component_coord_weak3_reward`;
- `reward_component_coord_steady_reward`;
- `reward_component_coord_movement_collision`;
- `reward_component_coord_weak_uncovered_count`.

Checkpoint selection remains lexicographic QoS feasibility, worst, weak3,
steady, then lower traffic. Reward size never selects a checkpoint directly.

Do not advance automatically through the ladder. Use the same 20 evaluation
seeds and retain a new stage only when its paired effect matches its purpose:

- S1: worst improves without a material steady/weak3 regression;
- S2: movement collision falls and worst does not regress materially;
- S3: weak3 improves after S2 has passed;
- S4: steady improves after S3 has passed.

Also track `eval_actor_move_idle_fraction`: a collision reduction obtained by
stopping movement is a failed S2 result, not coordination.

## One-update diagnostic (seed 42, independent 20-seed bank)

These numbers are a smoke diagnostic, not a final performance claim:

| Variant | steady | weak3 | worst | movement collision | bits/frame |
|---|---:|---:|---:|---:|---:|
| S0: no combined QoS scalar, no segmented term | 0.71019 | 0.61359 | 0.22304 | 0.93133 | 2304 |
| S1: S0 + worst-deficit progress | 0.71909 | 0.62545 | 0.21386 | 0.94333 | 2304 |
| S1 - S0 | +0.00890 | +0.01186 | -0.00918 | +0.01200 | 0 |

S1 did not pass the stage gate: average and bottom-3 quality improved slightly,
but the metric it was intended to improve regressed and collision increased.
S2-S4 should therefore not be interpreted until S1 is redesigned or validated
over multiple training seeds.

An earlier magnitude calibration with the old combined QoS scalar still active
showed why the duplicate weight was reduced from 0.10 to 0.01: at 0.10 its
mean penalty (0.0160) was about 8.6 times the worst-progress term (0.00187), and
independent worst fell to 0.17117. At 0.01, independent performance recovered
to 0.71881/0.62508/0.21431 and collision was 0.9230, but this calibration is not
part of the isolated S0-S4 ladder.
