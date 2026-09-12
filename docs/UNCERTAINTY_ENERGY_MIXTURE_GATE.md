# Shared-state multiframe energy mixture

The receiver declares x/y velocity error N(0,6^2) m^2/s^2, using positive
3x3 Gauss-Hermite quadrature. Position remains the predicted fixed position.
Each hypothesis generates a physical OTFS template and its overlap with the
receiver's fixed projection, producing an effective-SNR hypothesis. No true
overlap or H1 validation labels are used in these weights.

For each hypothesis, frame and delivered-source energy log likelihoods are
added FIRST. One logsumexp over hypothesis probabilities is then taken for
the entire window. A shared state cannot be independently redrawn per frame
or per node without changing the statistical model. Deterministic unknown
phase is handled by the invariant energy density, not a phase prior.

Same 13-frame scene, same 12/13 delivery mask, same bits/power, 100000 trials:

| Receiver | P_D | P_FA |
| --- | ---: | ---: |
| Local point-SNR likelihood | 0.85181 | 0.00044 |
| Delivered point-SNR likelihood | 0.81020 | 0.00046 |
| Local uncertainty mixture | 0.85185 | 0.00044 |
| Delivered uncertainty mixture | 0.87549 | 0.00044 |

The paired mixture cooperation difference is 0.02364. Its one-sided 95%
Hoeffding lower bound is approximately 0.01590, conditional on frozen
calibration thresholds, one link trace and the physical model. This supports
cooperation for this case only. Individual PD/PFA intervals now account for
18 endpoints, replacing the earlier 14.

The 6m/s standard deviation is declared, not learned or calibrated by a
tracker. Nine quadrature points are a numerical approximation, not exact
integration over a continuous posterior. SNR uncertainty is preconfigured
at the receiver; communicating such a model is outside this experiment.
The improvement must be tested across mismatch magnitudes and independent
link/geometry seeds before being promoted to an operational policy.

Reproduce with tools/audit_unified_receiver_transport.py.
