# Stable Distributed V1

Freeze date: 2026-08-22  
Scope: certificate-light replicated L1 + 8-bit near-field physical state +
25% row-local temporal inertia, for the validated 6×6 and 8×8 nominal
0 dB U2U operating regime.

This is a hash-verified research freeze, not a claim that the dirty workspace
at Git HEAD is identical. The workspace had 275 pre-existing/active changes,
so a mixed Git commit would not be a trustworthy release boundary. The exact
recoverable source is the identical `source_snapshot.zip` stored in both
frozen result directories.

## Frozen entry points

- `config/exp_800_k6q6_distributed_replicated_l1_nearfield32_inertia25.yaml`
  - SHA-256: `E3816755A0FBBDF261BF3EC1F916ECF16369D3C9DDDF71C51AD5E694FA737855`
- `config/exp_800_k8q8_distributed_replicated_l1_nearfield32_inertia25.yaml`
  - SHA-256: `AFD0A048E4689A67F616FE96D6EBE75972B461B1577A43ACF79A6110A81FAF13`

Git reference at freeze time:
`b4ec133c5d11dc43f2a5828129d0d101d663aa04` (dirty; not sufficient alone).

## Recoverable source snapshot

- `results/_distributed_replicated_l1_nearfield32_inertia25_k6_holdout5/source_snapshot.zip`
- `results/_distributed_replicated_l1_nearfield32_inertia25_k8_holdout5/source_snapshot.zip`
- Identical SHA-256:
  `312EE5A88BB58CA81B5F2EF6FEA46C7705C75BB8A029DDB79568E847EA0FA9D3`

Do not extract this archive over the active workspace. Restore it into a new
directory and verify the archive hash first.

## Frozen evidence

6×6 seeds `[843,401,762,960,783]`:

- worst mean/min: `0.955470 / 0.886781`
- weak-3 mean/min: `0.984565 / 0.962260`
- steady mean/min: `0.992191 / 0.981130`
- communication: `720 bit/frame`, delivery `1.0`
- analytical sensing-budget violation: `0 W`
- paired CSV SHA-256:
  `9AF6BAB38B7072438557C98CFAC2D3773B5956E71D70D321A2AB3D02EA7542A2`

8×8 seeds `[181,409,429,970,479]`:

- worst mean/min: `0.987739 / 0.938697`
- weak-3 mean/min: `0.995913 / 0.979566`
- steady mean/min: `0.998467 / 0.992337`
- communication: `960 bit/frame`, delivery `1.0`
- analytical sensing-budget violation: `0 W`
- paired CSV SHA-256:
  `1C86EBABB5B856CC2822A36B34B0352F9D2E1938F339493F2FDF22136AE061E3`

Run manifests:

- K6 SHA-256:
  `8618EAB775F69F2FB01B1EABDE71E278733455DCFA6F7EEDEE088D0D70847315`
- K8 SHA-256:
  `A37AEF228F14A512D8C375548475EC16AC19D5EBC5A26A119A949FB041C1703E`

## Applicability boundary

Promoted: 6×6/8×8, nominal 0 dB threshold, full public-state delivery,
constant-density regions used by the listed configs.

Not promoted: 10×10/12×12 constant-density expanded-area operation,
approximately 40 dB range-outage operation, partial-view relay, or local-range
fallback experiments. Those remain development branches and must not replace
this freeze without a new independent confirmation result and a new manifest.
