# Research disturbance ladder (K16/Q16)

## Purpose

The frozen clean K16/Q16 result is a control, not the main algorithm-development
environment. Its high score is conditional on a favorable closed-world regime:
matched K=Q resources, CV targets limited to 0--5 m/s with 0.5 m/s^2 process
noise, deterministic 1 m^2 RCS, no receiver synchronization error, no exogenous
sensing interference, and a 500 kHz U2U link that delivered every calibration
packet. The reported metric is a one-frame detection probability averaged over
the final 50 of 150 frames and over an in-scope geometry-filtered seed bank. It
does not establish robustness to stronger maneuvers, receiver mismatch,
interference, different cardinalities, or out-of-distribution geometry.

## Frozen ladder

All levels retain K=Q=16, a 1130 m square, 150 frames, a 50-frame tail, carrier
period 3, the same analytical distributed controller and paired seeds. No level
uses Gilbert--Elliott or other burst packet loss.

| Level | Added mechanism | steady | weak3 | worst | min worst | QoS | delivery |
|---|---|---:|---:|---:|---:|---:|---:|
| L0 | clean frozen control | 0.997 | 0.983 | 0.970 | 0.940 | 1.00 | 1.00 |
| L1 | Swerling-II RCS fluctuation | 0.978 | 0.950 | 0.933 | 0.895 | 1.00 | 1.00 |
| L2 | L1 + fixed 0.35-bin delay/Doppler mismatch | 0.973 | 0.944 | 0.933 | 0.886 | 1.00 | 1.00 |
| L3 | L2 + speed 2--10 m/s and sigma_a=1.5 | 0.949 | 0.899 | 0.878 | 0.794 | 1.00 | 1.00 |
| L4 | L3 + receiver NF 8 dB | 0.900 | 0.829 | 0.805 | 0.625 | 0.75 | 1.00 |

These are four-seed calibration estimates, not confirmatory claims. L4 is the
algorithm-development level because its mean worst score lies in the planned
0.70--0.85 interval, at least one seed fails the old QoS gate, and the system
does not collapse universally. L0--L3 are retained for mechanism ablation.

The fixed synchronization error is intentionally simple. It is a signed
receiver matched-filter offset in fractional OTFS bins, not a full oscillator,
clock-drift, or CSI-estimation model. The 8 dB noise figure is an equivalent
receiver noise/unresolved-interference load, not an explicit spatial jammer.

## Algorithm comparison rule

Every proposed algorithm must use paired seeds and identical disturbance draws,
budgets, frames and stopping rules. Report L4 as the primary development result,
L0 as a clean-regime regression guard, and L1--L3 as attribution ablations. Do
not tune on the frozen formal test split. A later confirmation needs a newly
sealed seed bank and confidence intervals; calibration numbers cannot be
promoted to formal evidence.
