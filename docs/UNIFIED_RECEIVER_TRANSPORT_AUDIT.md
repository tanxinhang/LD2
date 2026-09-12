# Unified frozen-scene receiver and transport decomposition

13 OTFS frames; exact 800m slant legs at 20m altitude; Tx power 0.15W.
Prediction position offset is [0,30,0]m and velocity is [6,0,0]m/s versus
a static true target. Shared projected waveform samples and deterministic
frame phases are used across all five methods. Independent calibration,
H0 validation and H1 streams have 100000 samples each. Thresholds target
0.0005; simultaneous binomial intervals cover ten PD/PFA endpoints at 95%.

| Method | P_D | P_FA |
| --- | ---: | ---: |
| True template, local multiframe | 0.82841 | 0.00046 |
| Predicted template, coherent sum | 0.00272 | 0.00042 |
| Predicted template, multiframe energy | 0.82900 | 0.00046 |
| All remote energies delivered | 0.80931 | 0.00034 |
| Simulated FBL delivery | 0.80748 | 0.00040 |

The actual-delivery PD interval is [0.803958,0.810968]; PFA interval is
[0.000245,0.000613]. This passes only the frozen conditional model gate.
The near-equality of true and predicted local energy is explained by local
template overlap 0.999977 in this symmetric geometry. It does not resolve
general prediction mismatch. Remote overlap is only 0.537502.

Blindly adding remote energy degrades PD because weak signal adds degrees
of freedom. Ideal delivery is therefore not an upper performance bound for
this fusion rule. The slight difference between local oracle and predicted
PD is sampling/calibration fluctuation, not superiority over oracle.

Actual simulated link delivers 12 of 13 fresh reports. Each is 128 bits:
float32 energy, uint16 frame, uint16 schema and 64-bit header. Receiver uses
the float32 energy represented by that payload. No ACK or retransmission.
Attempted bits=1664; model communication energy=0.0000156J; sensing
energy=0.0019968J. The reserved schedule is 78.312ms (13 sensing blocks plus
13 separate 5ms reporting slots), with per-node RF peaks [0.15,0,0.1]W.

Transport uses the existing FBL simulator's delivery and energy accounting;
no bit-level codec is simulated. One fixed channel trace is shared between
splits, so this is not a field/channel-population confidence certificate.
Noise is independent AWGN, waveform cyclic and target static. The appropriate
next change is calibrated receiver-side evidence fusion and its uncertainty,
followed by multiple geometry/channel traces. No general 800m claim is made.

Reproduce: `python tools/audit_unified_receiver_transport.py`.
