# Stable distributed U2U-ISAC baseline

## What

This snapshot consolidates the reproducible distributed UAV ISAC research
system before the next architecture redesign. It includes:

- decentralized UAV movement, message generation, target commitment and
  joint communication-sensing power allocation;
- token-based U2U coordination with explicit packet rate, SNR, latency,
  delivery and energy accounting;
- environment-level multistatic evidence fusion and fixed-PFA detection;
- MAPPO/IPPO training, robust checkpoint selection, stratified seed banks,
  baselines, ablations and perturbation protocols;
- permutation-aware set/attention modules, distributed risk/CVaR diagnostics
  and the bounded Gate 2A--2F experiments;
- paper drafts and result summaries required to reproduce the reported
  conclusions.

Generated checkpoints, raw logs and bulk experiment outputs under `results/`
are intentionally excluded from this snapshot. Existing historical result
artifacts already tracked by the repository are left unchanged.

## Why

The current architecture has reached a defensible intermediate operating
point. Freezing it on a dedicated branch provides a reproducible reference
for the planned structural redesign and prevents future experiments from
silently changing the baseline.

The audit also establishes an important negative result: local channel
observability and sender-specific delivery credit do not make adaptive
4/8-bit precision useful in the present environment. The 8-bit action is
close to dominated, so a more complex causal rate controller is not justified
until the environment contains a preregistered condition where extra
precision yields a reproducible sensing benefit.

## Validated operating point

The current deployable choice is owner-aware Top-1 U2U evidence with a fixed
4-bit latent coordination stream and a detector threshold calibrated only
from the disjoint selection seed bank.

| Condition | steady | weak3 | worst | worst CVaR | QoS feasible | latent delivery |
|---|---:|---:|---:|---:|---:|---:|
| Nominal | 0.8877 | 0.8503 | 0.6331 | 0.0413 | 0.50 | 1.000 |
| SNR threshold 30 dB | 0.8857 | 0.8477 | 0.6252 | 0.0413 | 0.50 | 1.000 |
| Deadline 0.8 ms | 0.8876 | 0.8501 | 0.6348 | 0.0434 | 0.55 | 1.000 |

The mean Medium thresholds are met in these three conditions:
`steady >= 0.80`, `weak3 >= 0.70`, and `mean worst >= 0.60`.

## Claim boundary

This snapshot does **not** claim:

- tail reliability: scenario-level worst CVaR remains about 0.04;
- robustness at a 35 dB receive threshold: fixed-4 mean worst is 0.5370;
- feasibility at a 0.4 ms deadline;
- zero-shot K=6/K=8 scale generalization;
- a benefit from adaptive token precision.

The next architecture should therefore target the tail and variable-cardinality
coordination bottlenecks, while preserving the fixed-4 operating point as a
frozen control. Detection calibration and the structured evidence codec should
not be changed unless a controlled failure-mechanism test implicates them.

## Checks

- Targeted U2U observation, communication, adaptive-rate and joint-power
  regression suite: 83 passed.
- Formal test suite: 334 tests collected across 47 files; all 334 pass when
  each file is executed in an isolated process.
- On the current Windows/Conda runtime, a monolithic cross-file pytest process
  aborts inside NumPy/MKL `eigvalsh` after earlier tests have run. The affected
  belief-calibration file passes all 14 tests in isolation. `pytest.ini`
  restricts discovery to the formal `tests/` directory so executable research
  diagnostics under `scripts/` are not imported as tests.
- Staged diff whitespace validation and secret-pattern audit are required
  before publication.

Detailed evidence is available in:

- `docs/GATE2C_CALIBRATION_CHANNEL_GENERALIZATION_RESULTS.md`
- `docs/GATE2D_LATENCY_AWARE_RATE_RESULTS.md`
- `docs/GATE2E_LOCAL_CHANNEL_RATE_RESULTS.md`
- `docs/GATE2F_SENDER_DELIVERY_CREDIT_RESULTS.md`
- `docs/FINAL_PAPER_EXPERIMENTS.md`
