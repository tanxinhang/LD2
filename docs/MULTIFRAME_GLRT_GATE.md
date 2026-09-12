# Multiframe deterministic-amplitude GLRT gate

The new detector whitens each complete trajectory design and projects onto
its numerical column space. Independent complex frame amplitudes are nuisance
parameters, not phase priors. A maximum is taken across complete trajectories,
never independently across frames. Duplicate columns cannot inflate rank.

Synthetic two-route experiment, 50000 samples per independent split:

| Frames | P_D, fixed per-frame energy | P_D, fixed total energy | Held-out P_FA |
| --- | ---: | ---: | ---: |
| 1 | 0.23164 | 0.23164 | 0.00094 |
| 2 | 0.45648 | 0.11596 | 0.00102 |
| 4 | 0.86806 | 0.07498 | 0.00152 |
| 8 | 0.99618 | 0.03664 | 0.00140 |

The energy unit is normalized received energy, not watts or joules. With fixed
per-frame energy the total increases with window length. With fixed total
energy, independent amplitude degrees of freedom and bank search costs hurt
detection. Thus extra frames do not automatically improve resource efficiency.
This counterexample motivates explicit energy/time accounting in OTFS tests.

The empirically calibrated thresholds did not keep all held-out P_FA estimates
below 0.001. No detection certification is made, including the apparent four-
frame P_D above 0.8. This is a synthetic statistical gate, not an 800 m result.
The next step is frozen-geometry OTFS trajectory designs with power-duration
accounting and conservative global false-alarm calibration.

Validation includes duplicated columns, independent column phase/scale
invariance, forbidden cross-trajectory frame stitching and colored Gaussian
noise normalization. Reproduce with `python tools/audit_multiframe_glrt.py`.
