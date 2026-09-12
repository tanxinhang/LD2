# Fixed-geometry OTFS multiframe statistical gate

Two 800 m slant legs, altitude 20 m, one transmitter and one receiver.
The existing physical radar equation and fractional OTFS DD response give
per-frame matched complex SNR 1.71095322 at 0.15 W. Frames last 1.024 ms.
The independent deterministic amplitude GLRT is sum |z_t|^2, whose null law
is Gamma(W,1). A predeclared design P_FA=0.0005 leaves room to test the required
0.001 using independent observations, without retuning against validation.

200000 trials per split; exact binomial intervals use Bonferroni across
16 reported PD/PFA intervals for joint coverage at least 95 percent.

| Frames | Fixed 0.15W P_D | Fixed total energy P_D | Held-out P_FA |
| --- | ---: | ---: | ---: |
| 1 | 0.03084 | 0.03084 | 0.000555 |
| 2 | 0.07895 | 0.01769 | 0.000440 |
| 4 | 0.21456 | 0.00914 | 0.000515 |
| 8 | 0.54004 | 0.00463 | 0.000480 |

All reported P_FA upper bounds fall below 0.001. No PD gate passes. Analytic
scanning at the same threshold rule first exceeds 0.8 at 13 frames: model
P_D=0.82210856 and useful observation time 13.312 ms. This is an analytic
boundary only, not a validated minimum-window certification.

Eight fixed-power frames consume 0.0012288 J sensing RF energy. Fixed window
energy is 0.0001536 J, implemented by power 0.15/W. No evidence transport is
needed in this single receiver experiment. The research 0.15W PA assumption
respects the requested 1W RF ceiling but exceeds the legacy 25.1mW sensing cap.

The cyclic waveform model excludes guard/CP overhead, clutter, synchronization
and DD prediction error. Time is useful observation time, not end-to-end latency.
Noise and template matching are ideal. This reference cannot establish a
distributed real-system 800m detection claim. A bank or early stopping requires
a new global threshold. Reproduce with audit_otfs_multiframe_fixed_geometry.py.
