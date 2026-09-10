# Current results

This directory contains only outputs regenerated for the current 2026-09-10 audit.

- `strict_k16q16_smoke.json`: ten independent diagnostic seeds, 30 frames each,
  regenerated after atomic epoch activation and exact separable 2-D safety-QP
  closure; every seed has 96.67% common-model certification, zero protocol
  fallback, zero deadline miss and QoS success. It is not formal evidence.
- `markov_graph_shadow.json`: 32-case K16/Q16 configuration-aligned shadow microbenchmark; not formal evidence.
- `fixed_lag_shadow.json`: 32-seed history-smoothing/residual-forecast mechanism screen; not formal evidence.

No current-commit blind-100 formal result is present. A formal refresh requires a clean committed tree and the managed provenance workflow.
