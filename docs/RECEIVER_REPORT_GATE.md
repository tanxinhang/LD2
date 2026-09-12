# Receiver-visible report continuation gate

The owner waits for four sensing/report slots. Its gate receives only local
prefix energies and remote energies delivered within that prefix. Effective
SNR estimates are max(mean energy - 1, 0). A noncoherent test-power forecast
compares stopping reports with continuing, using these estimates and remaining
frame counts. This is an intentionally simple surrogate, not a certified
mixture-information objective. No future observations, target truth or future
delivery masks enter the decision. With no received evidence it continues.

One 128-bit stop/continue schedule-control message is sent after the prefix.
This is not an acknowledgment or retransmission request. Lost control means
continue. Its power, energy, bits and extra 5ms reserved slot are charged.
The sender halts future report attempts only when stop control arrives. The
offline counterfactual link trace advances on the same schedule; unattempted
report outcomes are not exposed as evidence. Model channel statistics do not
depend on energy payload contents. This is still not a bit-level codec test.

Full adaptive-score H0 calibration includes the observation-dependent gate,
under H0 and H1 separately. All earlier local evidence and delivered prefix
evidence remain in the terminal mixture. Dropped future evidence never enters.

Default case, 100000 trials per split:

| Metric | Always report | Prefix gate |
| --- | ---: | ---: |
| P_D | 0.87549 | 0.87026 |
| P_FA | 0.00044 | 0.00044 |
| Mean attempted bits under H1 | 1664 | 1179.70048 |
| Mean communication energy under H1, J | 0.00001560 | 0.000011159692 |
| Reserved elapsed ms | 78.312 | 83.312 |

Under H0 the gated mean bits are 1457.20576, unlike H1. Reporting a single
unconditional expected cost would require an explicit hypothesis prior.
The 0.00523 PD loss point estimate means no lossless claim is justified.

A descriptive 0/6/18 m/s error by three-link-seed scan (30000 samples/split)
gives H1 bits about 1617–1668, 1187–1209, and 911–923 respectively. Large
error saves approximately 45%, but weak error can cost more than always
reporting. Sample PD differences range roughly -0.0106 to +0.0006.
This supports feasibility of receiver-visible throttling, not optimality or
population certification. All nine scanned control packets arrived; lost
control fallback is tested separately at the decision boundary.

Both strategies reserve the same data slots; the gate only saves RF traffic,
not elapsed sensing time. The gate's forecast assumes all future reports
arrive; it needs calibrated link prediction before deployment. Moment estimates
from a four-frame prefix are noisy. Any replacement must be retested with
whole-policy calibration and include the cost of control.

Reproduce default: `audit(report_gate=True)` in
`tools/audit_unified_receiver_transport.py`. The feature defaults off.
