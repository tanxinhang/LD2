# Stable Distributed Dynamic K12 V1

Freeze date: 2026-08-25
Scope: dynamic-target (CV, 2–5 m/s) K=Q=12 distributed candidate with
FBL/shadowed U2U transport, ACK-free AoI cache, conservative GCI posterior
feedback, 2-sigma robust bistatic gain, and the decoupled belief-freshness
posterior schedule with a hybrid free-ride + top-up budget (audit #3 +
unified-MAC accounting, audit #5).

This freeze target was selected by the pre-registered paired confirmation
`LD3-dyn-k12-conf-1` (Amendment 1): the hybrid candidate passed the gate
(20/25 vs 19/25 baseline, paired worst delta bootstrap 95% LCB > 0), so the
freeze target is the hybrid candidate, not the pre-fix engineering candidate.

This is a hash-verified research freeze, not a claim that the dirty workspace
at Git HEAD is identical.  The exact recoverable source is the identical
`source_snapshot.zip` stored in the confirmation result directories.

## Frozen entry points

- `config/exp_800_k12q12_distributed_v2_dynamic_u2u_robustbelief_freshness_pilot.yaml`
  - resolved_config_sha256:
    `77ee09962d627f3724304a595cdebcfc923ba930246f5b5ac247e7fdd35bf536`
  - frozen checkpoint:
    `results/_audit_k10_tail40_safe_robust2/best_restored.pt`
    - SHA-256:
      `8b3bb61fc7bcb04042c29a640880291c6f658e09a2d7426b18bda28607c1453d`
  - mechanism flags (all default-off elsewhere):
    `u2u_belief_feedback_schedule: freshness`,
    `u2u_belief_feedback_topk: 2`,
    `u2u_belief_feedback_bit_budget: 400`,
    `u2u_belief_feedback_max_union_per_source: 2`

Git reference at freeze time: `b4ec133c5d11dc43f2a5828129d0d101d663aa04`
(dirty; not sufficient alone).

## Recoverable source snapshot

- `results/_conf25_hybrid/source_snapshot.zip`
- `results/_hybrid_8seed/source_snapshot.zip`
- Identical SHA-256:
  `86c4f196eb52e66a8e3918ecdbe5c0475ddbdc18366dd059cde0ba2a6bb0d70d`
  (611 files; verified on disk at freeze time)

Do not extract this archive over the active workspace. Restore it into a new
directory and verify the archive hash first.

## Frozen evidence — 25-seed paired confirmation (LD3-dyn-k12-conf-1)

Seeds: `LD3-tailgate-safe-confirm-v1` K12 stage set (25 seeds, SHA-256
derived, no difficulty screening).  Warm start: K10 checkpoint, `episodes=0`
zero-training transfer ablation.  Computed with
`tools/summarize_dyn_k12_conf.py`.

Baseline (pre-fix engineering candidate):

- worst mean/min/p05: `0.6971 / 0.0053 / 0.1213`
- weak-3 mean/min: `0.7913 / 0.0191`
- steady mean/min: `0.9312 / 0.6659`
- three-gate pass: `19/25`, Wilson 95% LCB `0.5657`
- evidence deadline violation: `0.0`, delivery `0.9997`

Frozen hybrid candidate:

- worst mean/min/p05: `0.7701 / 0.0111 / 0.5091`
- weak-3 mean/min: `0.8500 / 0.0178`
- steady mean/min: `0.9427 / 0.5258`
- three-gate pass: `20/25`, Wilson 95% LCB `0.6087`
- paired same-seed deltas: worst `+0.0731` (16/25 positive), weak-3
  `+0.0587` (17/25), steady `+0.0114` (16/25); worst bootstrap 95% LCB
  `+0.0166`
- verdict per protocol: **PASS**
- evidence deadline violation: `0.0008`, delivery `0.9990`
- continuous separation violation: `0.0` (min 27.76 m)

Unified-MAC accounting (both runs): serialized coordination + evidence latency
mean `5.2–5.3 ms`, p95 `7.1–7.3 ms` vs the 4 ms frame deadline →
**total-protocol violation rate `1.0`** (audit #5 finding; the per-class
deadlines pass, the serialized sum does not — the MAC slot split is the open
protocol item).

## Applicability boundary

Promoted: K=Q=12, T=150, dynamic CV targets (2–5 m/s), FBL 4 ms shadowed
U2U, no ground link, engineering detector calibration
(`k12-passiverx-rawllr-h0-seed1019466100-500k-engineering`), K10-checkpoint
zero-training transfer ablation.

Not promoted:

- 6×6/8×8/10×10 dynamic same-caliber curves (per-scale calibration and
  configs not yet built; audit #7).
- Full MAPPO retraining under the dynamic stack (audit #6).
- IQ-waveform / raw-detector / hardware validation (audit P2).
- The static-target frozen V1 (`frozen/STABLE_DISTRIBUTED_V1_2026-08-22.md`)
  remains the 6×6/8×8 static boundary; this freeze does not replace it.
