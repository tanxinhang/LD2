# STOP-only control: cost shadow, not transport promotion

The system has PyTorch 2.11.0+cu128 in
`E:/anaconda/conda/python.exe`. The previously selected
`E:/anaconda/3_11_python/python.exe` did not have torch. Prior reports of a
missing dependency apply to that interpreter, not the machine. Fourteen
targeted tests now pass through the normal pytest/conftest entry point using
the conda interpreter; this is not a full-suite run.

First bounded protocol step: expose the cost of transmitting a STOP packet
only when the receiver requests stopping. Silence or failed STOP means the
sender continues on the known schedule. No ACK or retransmission is needed.

Let q be requested-stop probability and d effective-stop probability including
control delivery. With full scheduled bits B, control cost C and canceled report
bits R, expected bits are B + q*C - d*R, not B + d*C - d*R. Failed requests
still cost airtime/energy. The audit records both quantities and analogous RF
energy, preserving the original control-bearing baseline.

This is COST-ONLY SHADOW ACCOUNTING. The existing simulator still sends the
control packet for every trial; no changed delivery or detection claim follows.
The reserved control slot remains, so there is no latency improvement. Actual
transport integration must preserve channel evolution under silent slots and
verify collision/interference assumptions, timeout behavior and payload costs.
Never simply skip a random-number-consuming transmit call and call its changed
future delivery mask a protocol gain.

For the preceding frozen validation, control delivery was successful on the
single frozen trace, so q=d. H1 d=.09422 implies shadow bits
1664 - 1024*.09422 = 1567.51872 (5.80% saving), versus baseline gate
1683.45856. This is arithmetic on the old trace, NOT a new independent
performance result. Across channels, savings require d/q > C/R = 1/9 when
q>0, and must be independently validated. Default behavior is unchanged.
