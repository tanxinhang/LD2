# Current results

This directory contains only outputs regenerated for the current 2026-09-10 audit.

- `strict_k16q16_smoke.json`: two-seed, 30-frame diagnostic regenerated after
  packet-model rendezvous and pre-clamp energy closure; byte-exact common-model
  certification succeeds on 30% of frames and the remaining 70% safely falls
  back. It is not formal evidence.
- `markov_graph_shadow.json`: 32-case K16/Q16 configuration-aligned shadow microbenchmark; not formal evidence.
- `fixed_lag_shadow.json`: 32-seed history-smoothing/residual-forecast mechanism screen; not formal evidence.

No current-commit blind-100 formal result is present. A formal refresh requires a clean committed tree and the managed provenance workflow.
