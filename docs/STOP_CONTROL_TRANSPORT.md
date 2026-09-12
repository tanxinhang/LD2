# Reserved-slot STOP transport integration

`ReservedSlotTransport` is an opt-in wrapper around the existing physical
`InterUAVCommunicationModel.transmit`. An absent STOP is an empty transmission
call, not a fictitious free packet. It incurs zero bits and advances the channel
once. An actual STOP uses the existing 128-bit message budget and 0.1 W RF power;
the sender stops only on simulated receipt. Failure continues without retry.

Code inspection showed channel state advances on every transmit call, including
empty calls, but packet errors and channel innovations share an RNG. Simply
removing control traffic would shift future random draws. The wrapper uses
slot-keyed streams on a dedicated radio so actions cannot shift future channel
innovations. This defines NEW coupled seeded traces, not equivalence to legacy
seeded audit results. It assumes reserved slots and exogenous channel dynamics;
no interference-aware or variable-time scheduling claim is made.

Tests pair active and silent control over 30 channel seeds and 9 later report
slots each: future shadowing and report deliveries match. Silence costs zero;
requested-but-rejected STOP still costs 128 bits. Physical transport is exercised,
not just shadow cost arithmetic. Seven targeted tests pass with the PyTorch
environment. The wrapper does not yet drive the large-sample mixture detector:
next integration must carry trial-specific prefix decisions through its actual
report schedule and recalibrate against these new traces. No new PD, total-bit
savings or latency claim follows from these protocol tests. Defaults unchanged.
