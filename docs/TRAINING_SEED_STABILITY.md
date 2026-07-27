# Training-seed stability experiment

This is a three-seed replication of the trainable commitment head using a shared foundation actor. Each resulting model is evaluated on the same 100 fixed test scenarios. It is not presented as full end-to-end random-initialization variance.

| Training seed | steady | weak3 | worst | worst LCB | CVaR20 | QoS feasible | bit/frame |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.9500 | 0.9333 | 0.8129 | 0.7603 | 0.2522 | 0.83 | 768 |
| 123 | 0.9506 | 0.9341 | 0.8216 | 0.7709 | 0.2809 | 0.84 | 768 |
| 456 | 0.9505 | 0.9339 | 0.8211 | 0.7707 | 0.2809 | 0.84 | 768 |

Across training seeds (mean +/- sample standard deviation):

- steady: 0.9504 +/- 0.0003
- weak3: 0.9338 +/- 0.0004
- worst: 0.8185 +/- 0.0049
- CVaR20: 0.2713 +/- 0.0165
- QoS feasible: 0.8367 +/- 0.0058

All three formal seeds meet the Medium average thresholds. The original model (worst=0.8186) lies at the centre of the replication distribution rather than being an exceptional selected seed.

## Training-budget sensitivity

With only 20 epochs at the default lower learning rate, mean worst falls to 0.7558. This pilot is retained as a training-budget sensitivity result and excluded from the formal three-seed aggregate.
